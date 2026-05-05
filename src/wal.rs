//! Write-Ahead Log.
//!
//! Append-only file containing length-prefixed records. Each record carries
//! an LSN (monotonically increasing), the txn_id that wrote it, and a typed
//! payload (Begin / Commit / Abort / Insert / Delete). The WAL invariant —
//! "log on disk before page on disk" — is enforced by `BufferPool` calling
//! [`WalManager::flush_to`] before writing any dirty page.
//!
//! This file is the *write side*; recovery (replay / undo on restart) lands
//! in the next day.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};

use crate::page::{PageId, Rid};

pub type Lsn = u64;

#[derive(Debug, Clone, PartialEq)]
pub enum WalRecordType {
    Begin,
    Commit,
    Abort,
    Insert {
        rid: Rid,
        data: Vec<u8>,
    },
    Delete {
        rid: Rid,
        data: Vec<u8>,
    },
    /// Compensation Log Record. Written during undo (normal rollback or
    /// recovery's undo phase) so that a crash mid-undo is recoverable.
    /// `undo_next_lsn` is the prev_lsn of the record this CLR compensates
    /// — recovery uses it to skip over already-undone work.
    Clr {
        undo_next_lsn: Lsn,
        redo: ClrRedo,
    },
    /// Fuzzy checkpoint. Persists the Active Transaction Table and the
    /// Dirty Page Table at the time the checkpoint started — recovery uses
    /// these as its starting point.
    Checkpoint {
        att: HashMap<u64, Lsn>,
        dpt: HashMap<PageId, Lsn>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClrRedo {
    UndoInsert { rid: Rid },
    UndoDelete { rid: Rid, data: Vec<u8> },
}

const TAG_BEGIN: u8 = 0;
const TAG_COMMIT: u8 = 1;
const TAG_ABORT: u8 = 2;
const TAG_INSERT: u8 = 3;
const TAG_DELETE: u8 = 4;
const TAG_CLR: u8 = 5;
const TAG_CHECKPOINT: u8 = 6;

const CLR_UNDO_INSERT: u8 = 0;
const CLR_UNDO_DELETE: u8 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub txn_id: u64,
    /// LSN of the previous record written by the same txn, or 0 if first.
    pub prev_lsn: Lsn,
    pub record_type: WalRecordType,
}

impl WalRecord {
    /// Layout (little-endian):
    /// `[lsn:8][txn_id:8][prev_lsn:8][tag:1][payload]`
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.lsn.to_le_bytes());
        buf.extend_from_slice(&self.txn_id.to_le_bytes());
        buf.extend_from_slice(&self.prev_lsn.to_le_bytes());
        match &self.record_type {
            WalRecordType::Begin => buf.push(TAG_BEGIN),
            WalRecordType::Commit => buf.push(TAG_COMMIT),
            WalRecordType::Abort => buf.push(TAG_ABORT),
            WalRecordType::Insert { rid, data } => {
                buf.push(TAG_INSERT);
                let (pid, slot) = *rid;
                buf.extend_from_slice(&pid.to_le_bytes());
                buf.extend_from_slice(&slot.to_le_bytes());
                buf.extend_from_slice(data);
            }
            WalRecordType::Delete { rid, data } => {
                buf.push(TAG_DELETE);
                let (pid, slot) = *rid;
                buf.extend_from_slice(&pid.to_le_bytes());
                buf.extend_from_slice(&slot.to_le_bytes());
                buf.extend_from_slice(data);
            }
            WalRecordType::Clr { undo_next_lsn, redo } => {
                buf.push(TAG_CLR);
                buf.extend_from_slice(&undo_next_lsn.to_le_bytes());
                match redo {
                    ClrRedo::UndoInsert { rid } => {
                        buf.push(CLR_UNDO_INSERT);
                        let (pid, slot) = *rid;
                        buf.extend_from_slice(&pid.to_le_bytes());
                        buf.extend_from_slice(&slot.to_le_bytes());
                    }
                    ClrRedo::UndoDelete { rid, data } => {
                        buf.push(CLR_UNDO_DELETE);
                        let (pid, slot) = *rid;
                        buf.extend_from_slice(&pid.to_le_bytes());
                        buf.extend_from_slice(&slot.to_le_bytes());
                        buf.extend_from_slice(data);
                    }
                }
            }
            WalRecordType::Checkpoint { att, dpt } => {
                buf.push(TAG_CHECKPOINT);
                buf.extend_from_slice(&(att.len() as u32).to_le_bytes());
                for (txn_id, last_lsn) in att {
                    buf.extend_from_slice(&txn_id.to_le_bytes());
                    buf.extend_from_slice(&last_lsn.to_le_bytes());
                }
                buf.extend_from_slice(&(dpt.len() as u32).to_le_bytes());
                for (pid, rec_lsn) in dpt {
                    buf.extend_from_slice(&pid.to_le_bytes());
                    buf.extend_from_slice(&rec_lsn.to_le_bytes());
                }
            }
        }
        buf
    }

    fn decode(body: &[u8]) -> Result<Self> {
        if body.len() < 25 {
            bail!("WAL record too short: {} bytes", body.len());
        }
        let lsn = u64::from_le_bytes(body[0..8].try_into().unwrap());
        let txn_id = u64::from_le_bytes(body[8..16].try_into().unwrap());
        let prev_lsn = u64::from_le_bytes(body[16..24].try_into().unwrap());
        let tag = body[24];
        let rest = &body[25..];
        let record_type = match tag {
            TAG_BEGIN => WalRecordType::Begin,
            TAG_COMMIT => WalRecordType::Commit,
            TAG_ABORT => WalRecordType::Abort,
            TAG_INSERT | TAG_DELETE => {
                if rest.len() < 6 {
                    bail!("Insert/Delete record missing rid bytes");
                }
                let pid = u32::from_le_bytes(rest[0..4].try_into().unwrap());
                let slot = u16::from_le_bytes(rest[4..6].try_into().unwrap());
                let data = rest[6..].to_vec();
                if tag == TAG_INSERT {
                    WalRecordType::Insert {
                        rid: (pid, slot),
                        data,
                    }
                } else {
                    WalRecordType::Delete {
                        rid: (pid, slot),
                        data,
                    }
                }
            }
            TAG_CLR => {
                if rest.len() < 9 {
                    bail!("CLR record missing fields");
                }
                let undo_next_lsn = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let redo_tag = rest[8];
                let redo_rest = &rest[9..];
                let redo = match redo_tag {
                    CLR_UNDO_INSERT => {
                        if redo_rest.len() < 6 {
                            bail!("CLR UndoInsert missing rid");
                        }
                        let pid = u32::from_le_bytes(redo_rest[0..4].try_into().unwrap());
                        let slot = u16::from_le_bytes(redo_rest[4..6].try_into().unwrap());
                        ClrRedo::UndoInsert { rid: (pid, slot) }
                    }
                    CLR_UNDO_DELETE => {
                        if redo_rest.len() < 6 {
                            bail!("CLR UndoDelete missing rid");
                        }
                        let pid = u32::from_le_bytes(redo_rest[0..4].try_into().unwrap());
                        let slot = u16::from_le_bytes(redo_rest[4..6].try_into().unwrap());
                        let data = redo_rest[6..].to_vec();
                        ClrRedo::UndoDelete {
                            rid: (pid, slot),
                            data,
                        }
                    }
                    other => bail!("unknown CLR redo tag: {other}"),
                };
                WalRecordType::Clr { undo_next_lsn, redo }
            }
            TAG_CHECKPOINT => {
                if rest.len() < 4 {
                    bail!("Checkpoint record missing att length");
                }
                let att_len = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
                let mut p = 4;
                let mut att = HashMap::with_capacity(att_len);
                for _ in 0..att_len {
                    if rest.len() < p + 16 {
                        bail!("Checkpoint att truncated");
                    }
                    let id = u64::from_le_bytes(rest[p..p + 8].try_into().unwrap());
                    let lsn = u64::from_le_bytes(rest[p + 8..p + 16].try_into().unwrap());
                    att.insert(id, lsn);
                    p += 16;
                }
                if rest.len() < p + 4 {
                    bail!("Checkpoint missing dpt length");
                }
                let dpt_len = u32::from_le_bytes(rest[p..p + 4].try_into().unwrap()) as usize;
                p += 4;
                let mut dpt = HashMap::with_capacity(dpt_len);
                for _ in 0..dpt_len {
                    if rest.len() < p + 12 {
                        bail!("Checkpoint dpt truncated");
                    }
                    let pid = u32::from_le_bytes(rest[p..p + 4].try_into().unwrap());
                    let rec_lsn = u64::from_le_bytes(rest[p + 4..p + 12].try_into().unwrap());
                    dpt.insert(pid, rec_lsn);
                    p += 12;
                }
                WalRecordType::Checkpoint { att, dpt }
            }
            other => bail!("unknown WAL tag: {other}"),
        };
        Ok(WalRecord {
            lsn,
            txn_id,
            prev_lsn,
            record_type,
        })
    }
}

pub struct WalManager {
    writer: Mutex<BufWriter<File>>,
    next_lsn: AtomicU64,
    flushed_lsn: AtomicU64,
}

impl WalManager {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
            next_lsn: AtomicU64::new(1), // LSN 0 is reserved for "no WAL record"
            flushed_lsn: AtomicU64::new(0),
        })
    }

    /// Append a record. Returns its LSN. Does NOT fsync — caller decides
    /// when durability is required (commit, page eviction, etc).
    pub fn append(&self, txn_id: u64, prev_lsn: Lsn, record_type: WalRecordType) -> Result<Lsn> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let record = WalRecord {
            lsn,
            txn_id,
            prev_lsn,
            record_type,
        };
        let body = record.encode();
        let mut w = self.writer.lock().unwrap();
        w.write_all(&(body.len() as u32).to_le_bytes())?;
        w.write_all(&body)?;
        Ok(lsn)
    }

    /// Force everything written so far to durable storage.
    pub fn flush(&self) -> Result<()> {
        let mut w = self.writer.lock().unwrap();
        w.flush()?;
        w.get_ref().sync_all()?;
        let durable_through = self.next_lsn.load(Ordering::SeqCst).saturating_sub(1);
        // Monotone update; another thread may have flushed past us already.
        let _ = self.flushed_lsn.fetch_max(durable_through, Ordering::SeqCst);
        Ok(())
    }

    /// Ensure all records up to `lsn` are durable. Cheap if already flushed.
    pub fn flush_to(&self, lsn: Lsn) -> Result<()> {
        if self.flushed_lsn.load(Ordering::SeqCst) >= lsn {
            return Ok(());
        }
        self.flush()
    }

    #[allow(dead_code)]
    pub fn flushed_lsn(&self) -> Lsn {
        self.flushed_lsn.load(Ordering::SeqCst)
    }

    /// Bump both counters past `lsn`. Called by recovery so newly-appended
    /// records continue past whatever was on disk at startup.
    pub fn set_next_lsn(&self, lsn: Lsn) {
        self.next_lsn.store(lsn, Ordering::SeqCst);
        let _ = self
            .flushed_lsn
            .fetch_max(lsn.saturating_sub(1), Ordering::SeqCst);
    }
}

/// Read all records from a WAL file. Returns `Ok(vec![])` if the file
/// doesn't exist. Used by recovery.
#[allow(dead_code)]
pub fn read_records<P: AsRef<Path>>(path: P) -> Result<Vec<WalRecord>> {
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut r = BufReader::new(file);
    let mut out = Vec::new();
    loop {
        let mut len = [0u8; 4];
        match r.read_exact(&mut len) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let n = u32::from_le_bytes(len) as usize;
        let mut body = vec![0u8; n];
        r.read_exact(&mut body)?;
        out.push(WalRecord::decode(&body)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-wal-{label}-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn append_and_read_back_with_prev_lsn_chain() {
        let p = temp_path("rt");
        {
            let w = WalManager::open(&p).unwrap();
            let l1 = w.append(42, 0, WalRecordType::Begin).unwrap();
            let l2 = w
                .append(
                    42,
                    l1,
                    WalRecordType::Insert {
                        rid: (3, 7),
                        data: vec![1, 2, 3],
                    },
                )
                .unwrap();
            let l3 = w
                .append(
                    42,
                    l2,
                    WalRecordType::Clr {
                        undo_next_lsn: l1,
                        redo: ClrRedo::UndoInsert { rid: (3, 7) },
                    },
                )
                .unwrap();
            let l4 = w.append(42, l3, WalRecordType::Abort).unwrap();
            assert!(l1 < l2 && l2 < l3 && l3 < l4);
            w.flush().unwrap();
        }
        let recs = read_records(&p).unwrap();
        assert_eq!(recs.len(), 4);
        assert_eq!(recs[0].prev_lsn, 0);
        assert_eq!(recs[1].prev_lsn, recs[0].lsn);
        assert_eq!(recs[2].prev_lsn, recs[1].lsn);
        assert!(matches!(
            recs[2].record_type,
            WalRecordType::Clr { redo: ClrRedo::UndoInsert { rid: (3, 7) }, .. }
        ));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn flush_to_is_no_op_if_already_durable() {
        let p = temp_path("flush-to");
        let w = WalManager::open(&p).unwrap();
        let begin_lsn = w.append(1, 0, WalRecordType::Begin).unwrap();
        let lsn = w.append(1, begin_lsn, WalRecordType::Commit).unwrap();
        w.flush().unwrap();
        assert!(w.flushed_lsn() >= lsn);
        w.flush_to(lsn).unwrap();
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn missing_file_reads_empty() {
        let p = temp_path("missing");
        let _ = std::fs::remove_file(&p);
        assert!(read_records(&p).unwrap().is_empty());
    }
}
