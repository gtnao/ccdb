use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;

use crate::page::{PAGE_SIZE, Page, PageId};

/// Lock-free over the file handle: we use `read_at`/`write_at` (pread/pwrite)
/// so concurrent threads don't fight over a shared seek cursor. `page_count`
/// is an `AtomicU32` advanced atomically in `allocate_page`. The only thing
/// that still needs serialization is the `sync` call's relationship with
/// readers: not strictly correct under concurrent appenders, but safe enough
/// for the buffer pool's checkpoint/eviction usage where higher layers
/// already coordinate.
pub struct DiskManager {
    file: std::fs::File,
    page_count: AtomicU32,
}

impl DiskManager {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let len = file.metadata()?.len();
        let page_count = (len / PAGE_SIZE as u64) as u32;
        Ok(Self {
            file,
            page_count: AtomicU32::new(page_count),
        })
    }

    pub fn page_count(&self) -> u32 {
        self.page_count.load(Ordering::SeqCst)
    }

    pub fn read_page(&self, page_id: PageId, buf: &mut [u8; PAGE_SIZE]) -> Result<()> {
        self.file
            .read_exact_at(buf, page_id as u64 * PAGE_SIZE as u64)?;
        Ok(())
    }

    pub fn write_page(&self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        self.file
            .write_all_at(data, page_id as u64 * PAGE_SIZE as u64)?;
        Ok(())
    }

    /// Force every preceding write to durable storage. Per-page writes are
    /// not fsynced individually; callers (checkpoint, shutdown, B-Tree leaf
    /// init) invoke this when they need durability. Recovery's redo
    /// covers any page write that didn't reach disk before a crash.
    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    pub fn allocate_page(&self) -> Result<PageId> {
        let new_id = self.page_count.fetch_add(1, Ordering::SeqCst);
        self.write_page(new_id, Page::new(new_id).as_bytes())?;
        Ok(new_id)
    }
}
