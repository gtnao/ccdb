use anyhow::{Result, bail};

use crate::disk::DiskManager;
use crate::page::{PAGE_SIZE, Page, PageId, SlotId};
use crate::tuple::{Schema, Value, deserialize_tuple, serialize_tuple};

pub type Rid = (PageId, SlotId);

pub struct Table {
    disk: DiskManager,
    schema: Schema,
}

impl Table {
    pub fn new(disk: DiskManager, schema: Schema) -> Self {
        Self { disk, schema }
    }

    pub fn page_count(&self) -> u32 {
        self.disk.page_count()
    }

    pub fn insert(&mut self, values: &[Value]) -> Result<Rid> {
        let bytes = serialize_tuple(values, &self.schema)?;

        // Try to append to the last page first.
        let n = self.disk.page_count();
        if n > 0 {
            let last = n - 1;
            let mut buf = [0u8; PAGE_SIZE];
            self.disk.read_page(last, &mut buf)?;
            let mut page = Page::from_bytes(&buf);
            if let Ok(slot) = page.insert(&bytes) {
                self.disk.write_page(last, page.as_bytes())?;
                return Ok((last, slot));
            }
        }

        // Allocate a new page.
        let new_id = self.disk.allocate_page()?;
        let mut page = Page::new(new_id);
        let slot = page.insert(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "tuple too large to fit in an empty page (tuple={} bytes, page={}): {e}",
                bytes.len(),
                PAGE_SIZE
            )
        })?;
        self.disk.write_page(new_id, page.as_bytes())?;
        Ok((new_id, slot))
    }

    pub fn scan(&mut self) -> Result<Vec<Vec<Value>>> {
        let mut out = Vec::new();
        for pid in 0..self.disk.page_count() {
            let mut buf = [0u8; PAGE_SIZE];
            self.disk.read_page(pid, &mut buf)?;
            let page = Page::from_bytes(&buf);
            for slot in 0..page.tuple_count() {
                let raw = page
                    .get_tuple(slot)
                    .ok_or_else(|| anyhow::anyhow!("missing slot {slot} on page {pid}"))?;
                out.push(deserialize_tuple(raw, &self.schema)?);
            }
        }
        Ok(out)
    }

    pub fn get(&mut self, rid: Rid) -> Result<Option<Vec<Value>>> {
        let (pid, slot) = rid;
        if pid >= self.disk.page_count() {
            bail!("page {pid} out of range");
        }
        let mut buf = [0u8; PAGE_SIZE];
        self.disk.read_page(pid, &mut buf)?;
        let page = Page::from_bytes(&buf);
        match page.get_tuple(slot) {
            Some(raw) => Ok(Some(deserialize_tuple(raw, &self.schema)?)),
            None => Ok(None),
        }
    }
}
