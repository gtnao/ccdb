use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;

use crate::page::{PAGE_SIZE, PageId};

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
        self.file.sync_all()?;
        Ok(())
    }

    pub fn allocate_page(&mut self) -> Result<PageId> {
        let new_id = self.page_count;
        self.page_count += 1;
        let blank = [0u8; PAGE_SIZE];
        self.write_page(new_id, &blank)?;
        Ok(new_id)
    }
}
