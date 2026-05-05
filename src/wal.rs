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

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};

use crate::page::Rid;

pub type Lsn = u64;

#[derive(Debug, Clone, PartialEq)]
pub enum WalRecordType {
    Begin,
    Commit,
    Abort,
    Insert { rid: Rid, data: Vec<u8> },
    Delete { rid: Rid, data: Vec<u8> },
}

const TAG_BEGIN: u8 = 0;
const TAG_COMMIT: u8 = 1;
const TAG_ABORT: u8 = 2;
const TAG_INSERT: u8 = 3;
const TAG_DELETE: u8 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub txn_id: u64,
    pub record_type: WalRecordType,
}

impl WalRecord {
    /// On-disk record body (the per-record length prefix is added by the
    /// WalManager when writing). Layout (little-endian):
    /// `[lsn:8][txn_id:8][tag:1][payload]`
    /// Insert/Delete payload: `[page_id:4][slot_id:2][data:..]`
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.lsn.to_le_bytes());
        buf.extend_from_slice(&self.txn_id.to_le_bytes());
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
        }
        buf
    }

    fn decode(body: &[u8]) -> Result<Self> {
        if body.len() < 17 {
            bail!("WAL record too short: {} bytes", body.len());
        }
        let lsn = u64::from_le_bytes(body[0..8].try_into().unwrap());
        let txn_id = u64::from_le_bytes(body[8..16].try_into().unwrap());
        let tag = body[16];
        let rest = &body[17..];
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
            other => bail!("unknown WAL tag: {other}"),
        };
        Ok(WalRecord {
            lsn,
            txn_id,
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
    pub fn append(&self, txn_id: u64, record_type: WalRecordType) -> Result<Lsn> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let record = WalRecord {
            lsn,
            txn_id,
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
    fn append_and_read_back() {
        let p = temp_path("rt");
        {
            let w = WalManager::open(&p).unwrap();
            let l1 = w.append(42, WalRecordType::Begin).unwrap();
            let l2 = w
                .append(
                    42,
                    WalRecordType::Insert {
                        rid: (3, 7),
                        data: vec![1, 2, 3],
                    },
                )
                .unwrap();
            let l3 = w
                .append(
                    42,
                    WalRecordType::Delete {
                        rid: (3, 7),
                        data: vec![9],
                    },
                )
                .unwrap();
            let l4 = w.append(42, WalRecordType::Commit).unwrap();
            assert!(l1 < l2 && l2 < l3 && l3 < l4);
            w.flush().unwrap();
        }
        let recs = read_records(&p).unwrap();
        assert_eq!(recs.len(), 4);
        assert_eq!(recs[0].record_type, WalRecordType::Begin);
        assert!(matches!(
            recs[1].record_type,
            WalRecordType::Insert { rid: (3, 7), .. }
        ));
        assert!(matches!(
            recs[2].record_type,
            WalRecordType::Delete { rid: (3, 7), .. }
        ));
        assert_eq!(recs[3].record_type, WalRecordType::Commit);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn flush_to_is_no_op_if_already_durable() {
        let p = temp_path("flush-to");
        let w = WalManager::open(&p).unwrap();
        let _ = w.append(1, WalRecordType::Begin).unwrap();
        let lsn = w.append(1, WalRecordType::Commit).unwrap();
        w.flush().unwrap();
        assert!(w.flushed_lsn() >= lsn);
        // Calling flush_to with an LSN already on disk should not error.
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
