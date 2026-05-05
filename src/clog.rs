//! Commit Log (CLOG): 2-bits-per-txn persistent status store.
//!
//! Each txn uses 2 bits:
//!   `00 InProgress` (default — never written), `01 Committed`, `10 Aborted`.
//! Pages are 4 KiB; that's 16384 txns per page. Pages are loaded on first
//! access and held in memory until process exit or `flush()`. Flushing is
//! invoked by a CHECKPOINT.
//!
//! Why no LRU here: even a million txns is ~256 KiB on 64 pages — no
//! eviction needed at this scope. Reference implementation keeps an
//! 8-frame LRU; we drop that complexity.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Result;

use crate::transaction_manager::TxnStatus;
use crate::tuple::TxnId;

const FILENAME: &str = "clog.db";
const PAGE_SIZE: usize = 4096;
const TXNS_PER_PAGE: u64 = (PAGE_SIZE * 4) as u64; // 4 statuses/byte * 4096

const STATUS_IN_PROGRESS: u8 = 0b00;
const STATUS_COMMITTED: u8 = 0b01;
const STATUS_ABORTED: u8 = 0b10;

#[derive(Debug)]
pub struct Clog {
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    path: PathBuf,
    /// Lazily-loaded pages keyed by page_id (= `txn_id / TXNS_PER_PAGE`).
    pages: HashMap<u64, [u8; PAGE_SIZE]>,
    /// Pages modified since last flush.
    dirty: HashSet<u64>,
    /// Number of pages currently on disk (file_size / PAGE_SIZE).
    on_disk_pages: u64,
}

impl Clog {
    /// Test-only: an in-memory CLOG that never touches disk. The path it
    /// would write to is a unique temp file that's never created (we never
    /// call `flush()` in tests that use this).
    #[cfg(test)]
    pub fn in_memory() -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-clog-mem-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        Self {
            inner: Mutex::new(Inner {
                path: p,
                pages: HashMap::new(),
                dirty: HashSet::new(),
                on_disk_pages: 0,
            }),
        }
    }

    pub fn open<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let path = data_dir.as_ref().join(FILENAME);
        let on_disk_pages = match std::fs::metadata(&path) {
            Ok(m) => m.len() / PAGE_SIZE as u64,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e.into()),
        };
        Ok(Self {
            inner: Mutex::new(Inner {
                path,
                pages: HashMap::new(),
                dirty: HashSet::new(),
                on_disk_pages,
            }),
        })
    }

    pub fn get(&self, txn_id: TxnId) -> Result<TxnStatus> {
        let mut inner = self.inner.lock().unwrap();
        let page_id = txn_id / TXNS_PER_PAGE;
        inner.ensure_loaded(page_id)?;
        let page = inner.pages.get(&page_id).expect("just loaded");
        let (off, shift) = bit_position(txn_id);
        let bits = (page[off] >> shift) & 0b11;
        Ok(match bits {
            STATUS_COMMITTED => TxnStatus::Committed,
            STATUS_ABORTED => TxnStatus::Aborted,
            _ => TxnStatus::InProgress,
        })
    }

    pub fn set(&self, txn_id: TxnId, status: TxnStatus) -> Result<()> {
        let bits = match status {
            TxnStatus::Committed => STATUS_COMMITTED,
            TxnStatus::Aborted => STATUS_ABORTED,
            TxnStatus::InProgress => STATUS_IN_PROGRESS,
        };
        let mut inner = self.inner.lock().unwrap();
        let page_id = txn_id / TXNS_PER_PAGE;
        inner.ensure_loaded(page_id)?;
        let (off, shift) = bit_position(txn_id);
        let page = inner.pages.get_mut(&page_id).expect("just loaded");
        page[off] = (page[off] & !(0b11 << shift)) | (bits << shift);
        inner.dirty.insert(page_id);
        Ok(())
    }

    /// Persist all dirty pages to disk + fsync.
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if inner.dirty.is_empty() {
            return Ok(());
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&inner.path)?;
        let dirty: Vec<u64> = inner.dirty.iter().copied().collect();
        for pid in dirty {
            let bytes = inner
                .pages
                .get(&pid)
                .expect("dirty page must be loaded")
                .to_owned();
            file.seek(SeekFrom::Start(pid * PAGE_SIZE as u64))?;
            file.write_all(&bytes)?;
            if pid >= inner.on_disk_pages {
                inner.on_disk_pages = pid + 1;
            }
        }
        file.sync_all()?;
        inner.dirty.clear();
        Ok(())
    }
}

impl Inner {
    fn ensure_loaded(&mut self, page_id: u64) -> Result<()> {
        if self.pages.contains_key(&page_id) {
            return Ok(());
        }
        let mut buf = [0u8; PAGE_SIZE];
        if page_id < self.on_disk_pages {
            let mut f = File::open(&self.path)?;
            f.seek(SeekFrom::Start(page_id * PAGE_SIZE as u64))?;
            f.read_exact(&mut buf)?;
        }
        self.pages.insert(page_id, buf);
        Ok(())
    }
}

fn bit_position(txn_id: TxnId) -> (usize, u8) {
    let off = ((txn_id % TXNS_PER_PAGE) / 4) as usize;
    let shift = ((txn_id % 4) * 2) as u8;
    (off, shift)
}

pub fn delete<P: AsRef<Path>>(data_dir: P) -> Result<()> {
    let path = data_dir.as_ref().join(FILENAME);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-clog-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn round_trip_via_disk() {
        let d = temp_dir("rt");
        {
            let c = Clog::open(&d).unwrap();
            c.set(1, TxnStatus::Committed).unwrap();
            c.set(2, TxnStatus::Aborted).unwrap();
            c.set(50_000, TxnStatus::Committed).unwrap(); // forces a 2nd page
            c.flush().unwrap();
        }
        // Reopen — values should be on disk.
        let c = Clog::open(&d).unwrap();
        assert_eq!(c.get(1).unwrap(), TxnStatus::Committed);
        assert_eq!(c.get(2).unwrap(), TxnStatus::Aborted);
        assert_eq!(c.get(3).unwrap(), TxnStatus::InProgress);
        assert_eq!(c.get(50_000).unwrap(), TxnStatus::Committed);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn pre_flush_changes_visible_in_same_process() {
        let d = temp_dir("pre");
        let c = Clog::open(&d).unwrap();
        c.set(10, TxnStatus::Committed).unwrap();
        // Not flushed yet — but in-memory pages still serve reads.
        assert_eq!(c.get(10).unwrap(), TxnStatus::Committed);
        std::fs::remove_dir_all(&d).ok();
    }
}
