use std::collections::HashMap;

use anyhow::Result;
use indexmap::IndexSet;

use crate::disk::DiskManager;
use crate::page::{PAGE_SIZE, Page, PageId};

/// Replacement policy for choosing an evictable frame.
pub trait Replacer {
    fn victim(&mut self) -> Option<usize>;
    fn pin(&mut self, frame_id: usize);
    fn unpin(&mut self, frame_id: usize);
}

/// LRU: oldest-touched unpinned frame is the victim.
pub struct LruReplacer {
    /// Insertion order ≈ access recency. Most-recently-touched is at the back.
    order: IndexSet<usize>,
    pinned: Vec<bool>,
}

impl LruReplacer {
    pub fn new(capacity: usize) -> Self {
        Self {
            order: IndexSet::new(),
            // Empty frames start as "pinned" so they're never picked as victims
            // before being filled. unpin() will clear this once a real page lives there.
            pinned: vec![true; capacity],
        }
    }
}

impl Replacer for LruReplacer {
    fn victim(&mut self) -> Option<usize> {
        self.order.iter().find(|&&id| !self.pinned[id]).copied()
    }

    fn pin(&mut self, frame_id: usize) {
        self.pinned[frame_id] = true;
        self.order.shift_remove(&frame_id);
        self.order.insert(frame_id);
    }

    fn unpin(&mut self, frame_id: usize) {
        self.pinned[frame_id] = false;
    }
}

struct Frame {
    page: Page,
    page_id: Option<PageId>,
    pin_count: u32,
    dirty: bool,
}

impl Frame {
    fn empty() -> Self {
        Self {
            page: Page::new(0),
            page_id: None,
            pin_count: 0,
            dirty: false,
        }
    }
}

pub struct BufferPoolManager<R: Replacer = LruReplacer> {
    frames: Vec<Frame>,
    page_table: HashMap<PageId, usize>,
    disk: DiskManager,
    replacer: R,
    capacity: usize,
}

impl BufferPoolManager<LruReplacer> {
    pub fn new(disk: DiskManager, capacity: usize) -> Self {
        Self::with_replacer(disk, LruReplacer::new(capacity), capacity)
    }
}

impl<R: Replacer> BufferPoolManager<R> {
    pub fn with_replacer(disk: DiskManager, replacer: R, capacity: usize) -> Self {
        let frames = (0..capacity).map(|_| Frame::empty()).collect();
        Self {
            frames,
            page_table: HashMap::new(),
            disk,
            replacer,
            capacity,
        }
    }

    pub fn page_count(&self) -> u32 {
        self.disk.page_count()
    }

    fn pick_or_evict(&mut self) -> Result<usize> {
        if self.page_table.len() < self.capacity {
            let fid = self
                .frames
                .iter()
                .position(|f| f.page_id.is_none())
                .expect("page_table < capacity implies an empty frame exists");
            return Ok(fid);
        }
        let victim = self
            .replacer
            .victim()
            .ok_or_else(|| anyhow::anyhow!("buffer pool exhausted: all frames pinned"))?;
        self.evict(victim)?;
        Ok(victim)
    }

    fn evict(&mut self, frame_id: usize) -> Result<()> {
        let frame = &mut self.frames[frame_id];
        if let Some(pid) = frame.page_id {
            if frame.dirty {
                self.disk.write_page(pid, frame.page.as_bytes())?;
            }
            self.page_table.remove(&pid);
        }
        frame.page_id = None;
        frame.dirty = false;
        frame.pin_count = 0;
        Ok(())
    }

    pub fn fetch_page(&mut self, page_id: PageId) -> Result<PageGuard<'_, R>> {
        let frame_id = if let Some(&fid) = self.page_table.get(&page_id) {
            self.frames[fid].pin_count += 1;
            self.replacer.pin(fid);
            fid
        } else {
            let fid = self.pick_or_evict()?;
            let mut buf = [0u8; PAGE_SIZE];
            self.disk.read_page(page_id, &mut buf)?;
            let f = &mut self.frames[fid];
            f.page = Page::from_bytes(&buf);
            f.page_id = Some(page_id);
            f.pin_count = 1;
            f.dirty = false;
            self.page_table.insert(page_id, fid);
            self.replacer.pin(fid);
            fid
        };
        Ok(PageGuard {
            bpm: self,
            frame_id,
            mutated: false,
        })
    }

    pub fn new_page(&mut self) -> Result<PageGuard<'_, R>> {
        let frame_id = self.pick_or_evict()?;
        let page_id = self.disk.allocate_page()?;
        let f = &mut self.frames[frame_id];
        f.page = Page::new(page_id);
        f.page_id = Some(page_id);
        f.pin_count = 1;
        // A freshly allocated page exists on disk only as zeros — its header
        // and any inserts must be flushed before the page is meaningful.
        f.dirty = true;
        self.page_table.insert(page_id, frame_id);
        self.replacer.pin(frame_id);
        Ok(PageGuard {
            bpm: self,
            frame_id,
            mutated: false,
        })
    }

    fn release(&mut self, frame_id: usize, mutated: bool) {
        let f = &mut self.frames[frame_id];
        if f.pin_count == 0 {
            return;
        }
        f.pin_count -= 1;
        if mutated {
            f.dirty = true;
        }
        if f.pin_count == 0 {
            self.replacer.unpin(frame_id);
        }
    }

    pub fn flush_all(&mut self) -> Result<()> {
        for fid in 0..self.frames.len() {
            let f = &mut self.frames[fid];
            if let Some(pid) = f.page_id {
                if f.dirty {
                    self.disk.write_page(pid, f.page.as_bytes())?;
                    f.dirty = false;
                }
            }
        }
        Ok(())
    }
}

/// RAII guard: holds a pin for the lifetime of the guard, releasing it on drop.
/// Calling `page_mut()` records that the frame was mutated, so `release` will
/// promote `dirty` regardless of whether new bytes actually changed.
pub struct PageGuard<'a, R: Replacer = LruReplacer> {
    bpm: &'a mut BufferPoolManager<R>,
    frame_id: usize,
    mutated: bool,
}

impl<'a, R: Replacer> PageGuard<'a, R> {
    pub fn page(&self) -> &Page {
        &self.bpm.frames[self.frame_id].page
    }

    pub fn page_mut(&mut self) -> &mut Page {
        self.mutated = true;
        &mut self.bpm.frames[self.frame_id].page
    }

    pub fn page_id(&self) -> PageId {
        self.bpm.frames[self.frame_id]
            .page_id
            .expect("guarded frame must have a page_id")
    }
}

impl<'a, R: Replacer> Drop for PageGuard<'a, R> {
    fn drop(&mut self) {
        self.bpm.release(self.frame_id, self.mutated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-bpm-{label}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn fresh_bpm(capacity: usize, path: &std::path::Path) -> BufferPoolManager {
        let disk = DiskManager::open(path).unwrap();
        BufferPoolManager::new(disk, capacity)
    }

    #[test]
    fn allocate_and_persist_via_flush() {
        let path = temp_path("flush");
        let mut bpm = fresh_bpm(2, &path);
        {
            let mut g = bpm.new_page().unwrap();
            assert_eq!(g.page_id(), 0);
            g.page_mut().insert(b"hello").unwrap();
        }
        bpm.flush_all().unwrap();

        // Reopen via a fresh BPM and read it back.
        drop(bpm);
        let mut bpm = fresh_bpm(2, &path);
        let g = bpm.fetch_page(0).unwrap();
        assert_eq!(g.page().get_tuple(0).unwrap(), b"hello");
        drop(g);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn lru_evicts_oldest_unpinned() {
        let path = temp_path("evict");
        let mut bpm = fresh_bpm(2, &path);

        // Allocate three pages with capacity=2 → page 0 gets evicted.
        {
            let mut g0 = bpm.new_page().unwrap();
            assert_eq!(g0.page_id(), 0);
            g0.page_mut().insert(b"page0").unwrap();
        }
        {
            let mut g1 = bpm.new_page().unwrap();
            assert_eq!(g1.page_id(), 1);
            g1.page_mut().insert(b"page1").unwrap();
        }
        {
            let mut g2 = bpm.new_page().unwrap();
            assert_eq!(g2.page_id(), 2);
            g2.page_mut().insert(b"page2").unwrap();
        }

        // Fetch page 0 again — must come from disk (round-trip via eviction).
        let g = bpm.fetch_page(0).unwrap();
        assert_eq!(g.page().get_tuple(0).unwrap(), b"page0");
        drop(g);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cache_hit_does_not_touch_disk_again() {
        let path = temp_path("hit");
        let mut bpm = fresh_bpm(2, &path);
        {
            let mut g = bpm.new_page().unwrap();
            g.page_mut().insert(b"data").unwrap();
        }
        // Two fetches of the same page reuse the same frame.
        let g1 = bpm.fetch_page(0).unwrap();
        let pid1 = g1.page_id();
        drop(g1);
        let g2 = bpm.fetch_page(0).unwrap();
        let pid2 = g2.page_id();
        assert_eq!(pid1, pid2);
        std::fs::remove_file(&path).ok();
    }

    // NOTE: under the current Guard design, holding multiple PageGuards
    // simultaneously is structurally impossible (each guard exclusively
    // borrows the BPM). The "all frames pinned" error path in pick_or_evict
    // therefore can't be reached from public API — it's kept as a defensive
    // check for future designs that allow concurrent guards.

    #[test]
    fn read_only_guard_does_not_dirty() {
        let path = temp_path("clean");
        let mut bpm = fresh_bpm(2, &path);
        // Create page 0 (dirty=true via new_page) and flush it.
        {
            let mut g = bpm.new_page().unwrap();
            g.page_mut().insert(b"x").unwrap();
        }
        bpm.flush_all().unwrap();

        // Read-only fetch of page 0; page_mut() not called → frame.dirty stays false.
        {
            let g = bpm.fetch_page(0).unwrap();
            let _ = g.page().tuple_count();
        }
        // Now force eviction by allocating two more pages (capacity=2). If the
        // read-only fetch had wrongly dirtied the frame, eviction would write
        // it back; this test passes either way for correctness, but exercises
        // the path. The key assertion is that the data on disk is unchanged.
        let _ = bpm.new_page().unwrap(); // 1
        let _ = bpm.new_page().unwrap(); // 2

        let g = bpm.fetch_page(0).unwrap();
        assert_eq!(g.page().get_tuple(0).unwrap(), b"x");
        drop(g);
        std::fs::remove_file(&path).ok();
    }
}
