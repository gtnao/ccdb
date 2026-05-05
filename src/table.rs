use anyhow::{Result, bail};

use crate::buffer_pool::BufferPoolManager;
use crate::page::{PAGE_SIZE, PageId, SlotId};
use crate::tuple::{Schema, Value, deserialize_tuple, serialize_tuple};

pub type Rid = (PageId, SlotId);

pub struct Table {
    bpm: BufferPoolManager,
    schema: Schema,
}

impl Table {
    pub fn new(bpm: BufferPoolManager, schema: Schema) -> Self {
        Self { bpm, schema }
    }

    pub fn page_count(&self) -> u32 {
        self.bpm.page_count()
    }

    pub fn flush(&mut self) -> Result<()> {
        self.bpm.flush_all()
    }

    pub fn insert(&mut self, values: &[Value]) -> Result<Rid> {
        let bytes = serialize_tuple(values, &self.schema)?;

        // Try last page first.
        let n = self.bpm.page_count();
        if n > 0 {
            let last = n - 1;
            let mut guard = self.bpm.fetch_page(last)?;
            if let Ok(slot) = guard.page_mut().insert(&bytes) {
                return Ok((last, slot));
            }
            // Doesn't fit; release before allocating a new page.
            drop(guard);
        }

        let mut guard = self.bpm.new_page()?;
        let pid = guard.page_id();
        let slot = guard.page_mut().insert(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "tuple too large to fit in an empty page (tuple={} bytes, page={}): {e}",
                bytes.len(),
                PAGE_SIZE
            )
        })?;
        Ok((pid, slot))
    }

    pub fn scan(&mut self) -> Result<Vec<Vec<Value>>> {
        let mut out = Vec::new();
        for pid in 0..self.bpm.page_count() {
            let guard = self.bpm.fetch_page(pid)?;
            let n = guard.page().tuple_count();
            for slot in 0..n {
                let raw = guard
                    .page()
                    .get_tuple(slot)
                    .ok_or_else(|| anyhow::anyhow!("missing slot {slot} on page {pid}"))?;
                out.push(deserialize_tuple(raw, &self.schema)?);
            }
        }
        Ok(out)
    }

    #[allow(dead_code)]
    pub fn get(&mut self, rid: Rid) -> Result<Option<Vec<Value>>> {
        let (pid, slot) = rid;
        if pid >= self.bpm.page_count() {
            bail!("page {pid} out of range");
        }
        let guard = self.bpm.fetch_page(pid)?;
        match guard.page().get_tuple(slot) {
            Some(raw) => Ok(Some(deserialize_tuple(raw, &self.schema)?)),
            None => Ok(None),
        }
    }
}
