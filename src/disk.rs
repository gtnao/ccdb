use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;

use crate::page::{PAGE_SIZE, Page, PageId};

pub struct DiskManager {
    file: std::fs::File,
    page_count: u32,
}

impl DiskManager {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        // Trailing partial page (if any) is ignored; we treat full pages as authoritative.
        let len = file.metadata()?.len();
        let page_count = (len / PAGE_SIZE as u64) as u32;
        Ok(Self { file, page_count })
    }

    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    pub fn read_page(&mut self, page_id: PageId, buf: &mut [u8; PAGE_SIZE]) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(page_id as u64 * PAGE_SIZE as u64))?;
        self.file.read_exact(buf)?;
        Ok(())
    }

    pub fn write_page(&mut self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(page_id as u64 * PAGE_SIZE as u64))?;
        self.file.write_all(data)?;
        Ok(())
    }

    /// Force every preceding write to durable storage. Caller invokes this at
    /// safe points (checkpoint, shutdown). Per-page writes are no longer
    /// fsynced individually — that was making bulk insert (COPY, CREATE INDEX
    /// scan) hundreds of times slower than necessary. Crash safety still
    /// holds because every page write is preceded by a WAL flush of records
    /// covering it, and recovery's redo pass replays those records.
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    pub fn allocate_page(&mut self) -> Result<PageId> {
        let new_id = self.page_count;
        self.page_count += 1;
        // Write a properly-initialised empty page rather than zeros. Zeros decode
        // as `free_space_offset = 0` (so the page looks "full" the moment it's
        // read back) and `next_page_id = 0` (which falsely points at page 0).
        // This matters on restart: if the buffer pool hasn't re-touched the
        // page since it was allocated, recovery's `fetch_page` reads what's on
        // disk verbatim — so the on-disk image must already be a valid empty
        // page.
        self.write_page(new_id, Page::new(new_id).as_bytes())?;
        Ok(new_id)
    }
}
