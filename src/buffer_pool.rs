//! Thread-safe buffer pool.
//!
//! `BufferPool` is a clonable handle (cheap `Arc` clone) that internally
//! protects the page table, frames, and disk manager with a single Mutex.
//! Each frame's `Page` lives behind its own `RwLock` so that, after the
//! short fetch/new_page critical section, callers from different threads
//! can read or write *different* pages in parallel.
//!
//! Pin lifecycle is RAII via [`PageGuard`]: the guard holds an Arc clone of
//! the pool plus the page id; on drop it briefly re-locks the pool to
//! decrement the pin count. Page contents are accessed via `read()` /
//! `write()`, which return standard `RwLock` guards.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use indexmap::IndexSet;

use crate::disk::DiskManager;
use crate::page::{PAGE_SIZE, Page, PageId};
use crate::wal::{Lsn, WalManager};

const ORDER: Ordering = Ordering::SeqCst;

/// Replacement policy for choosing an evictable frame.
trait Replacer {
    fn victim(&mut self) -> Option<usize>;
    fn pin(&mut self, frame_id: usize);
    fn unpin(&mut self, frame_id: usize);
}

struct LruReplacer {
    order: IndexSet<usize>,
    pinned: Vec<bool>,
}

impl LruReplacer {
    fn new(capacity: usize) -> Self {
        Self {
            order: IndexSet::new(),
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
    page: Arc<RwLock<Page>>,
    page_id: Option<PageId>,
    pin_count: u32,
    dirty: bool,
}

impl Frame {
    fn empty() -> Self {
        Self {
            page: Arc::new(RwLock::new(Page::new(0))),
            page_id: None,
            pin_count: 0,
            dirty: false,
        }
    }
}

struct Inner {
    frames: Vec<Frame>,
    page_table: HashMap<PageId, usize>,
    disk: DiskManager,
    replacer: LruReplacer,
    capacity: usize,
    wal: Arc<WalManager>,
    /// Dirty Page Table: `page_id → rec_lsn`. `rec_lsn` is the LSN at
    /// which the page first became dirty since its last flush. Used by
    /// fuzzy checkpoint to bound recovery's redo phase.
    dpt: HashMap<PageId, Lsn>,
}

#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Mutex<Inner>>,
}

impl BufferPool {
    pub fn new(disk: DiskManager, capacity: usize, wal: Arc<WalManager>) -> Self {
        let frames = (0..capacity).map(|_| Frame::empty()).collect();
        Self {
            inner: Arc::new(Mutex::new(Inner {
                frames,
                page_table: HashMap::new(),
                disk,
                replacer: LruReplacer::new(capacity),
                capacity,
                wal,
                dpt: HashMap::new(),
            })),
        }
    }

    /// Snapshot of the current DPT for inclusion in a Checkpoint record.
    pub fn dpt_snapshot(&self) -> HashMap<PageId, Lsn> {
        self.inner.lock().unwrap().dpt.clone()
    }

    pub fn page_count(&self) -> u32 {
        self.inner.lock().unwrap().disk.page_count()
    }

    pub fn fetch_page(&self, page_id: PageId) -> Result<PageGuard> {
        let mut inner = self.inner.lock().unwrap();
        let (page_arc, _frame_id) = inner.fetch_locked(page_id)?;
        Ok(PageGuard::new(self.clone(), page_id, page_arc))
    }

    pub fn new_page(&self) -> Result<PageGuard> {
        let mut inner = self.inner.lock().unwrap();
        let (page_id, page_arc) = inner.new_page_locked()?;
        Ok(PageGuard::new(self.clone(), page_id, page_arc))
    }

    pub fn flush_all(&self) -> Result<()> {
        self.inner.lock().unwrap().flush_all_locked()
    }

    /// Flush a single page synchronously. Used when the page's *structural*
    /// state (page_kind, etc.) must be on disk before a crash, because there
    /// is no WAL record that would let recovery rebuild it. The B+Tree leaf
    /// allocation is the current case: `init_leaf` only mutates the in-memory
    /// page, so without this, a crash before eviction leaves the on-disk
    /// page tagged as Heap.
    pub fn flush_page(&self, page_id: PageId) -> Result<()> {
        self.inner.lock().unwrap().flush_page_locked(page_id)
    }

    fn release(&self, page_id: PageId, mutated: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.release_locked(page_id, mutated);
    }
}

impl Inner {
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
                let page_guard = frame.page.read().unwrap();
                self.wal.flush_to(page_guard.page_lsn())?;
                self.disk.write_page(pid, page_guard.as_bytes())?;
            }
            self.page_table.remove(&pid);
            self.dpt.remove(&pid);
        }
        frame.page_id = None;
        frame.dirty = false;
        frame.pin_count = 0;
        Ok(())
    }

    fn fetch_locked(&mut self, page_id: PageId) -> Result<(Arc<RwLock<Page>>, usize)> {
        if let Some(&fid) = self.page_table.get(&page_id) {
            self.frames[fid].pin_count += 1;
            self.replacer.pin(fid);
            return Ok((Arc::clone(&self.frames[fid].page), fid));
        }
        let fid = self.pick_or_evict()?;
        let mut buf = [0u8; PAGE_SIZE];
        self.disk.read_page(page_id, &mut buf)?;
        // Replace the inner Page atomically so any prior holders of the Arc
        // don't see torn state. Since the frame was just made empty by evict()
        // (or this is its first use), no one should be holding the Arc, but
        // we use write() to be safe.
        {
            let mut p = self.frames[fid].page.write().unwrap();
            *p = Page::from_bytes(&buf);
        }
        let f = &mut self.frames[fid];
        f.page_id = Some(page_id);
        f.pin_count = 1;
        f.dirty = false;
        self.page_table.insert(page_id, fid);
        self.replacer.pin(fid);
        Ok((Arc::clone(&f.page), fid))
    }

    fn new_page_locked(&mut self) -> Result<(PageId, Arc<RwLock<Page>>)> {
        let fid = self.pick_or_evict()?;
        let page_id = self.disk.allocate_page()?;
        {
            let mut p = self.frames[fid].page.write().unwrap();
            *p = Page::new(page_id);
        }
        let f = &mut self.frames[fid];
        f.page_id = Some(page_id);
        f.pin_count = 1;
        // Freshly allocated → must be flushed (disk has only zeros).
        f.dirty = true;
        self.page_table.insert(page_id, fid);
        self.replacer.pin(fid);
        Ok((page_id, Arc::clone(&f.page)))
    }

    fn release_locked(&mut self, page_id: PageId, mutated: bool) {
        if let Some(&fid) = self.page_table.get(&page_id) {
            let f = &mut self.frames[fid];
            if f.pin_count == 0 {
                return;
            }
            f.pin_count -= 1;
            if mutated {
                let was_clean = !f.dirty;
                f.dirty = true;
                if was_clean {
                    // First-write since last flush: rec_lsn = current page_lsn
                    // (which the writer has just stamped via set_page_lsn).
                    let page_lsn = f.page.read().unwrap().page_lsn();
                    self.dpt.entry(page_id).or_insert(page_lsn);
                }
            }
            if f.pin_count == 0 {
                self.replacer.unpin(fid);
            }
        }
    }

    fn flush_page_locked(&mut self, page_id: PageId) -> Result<()> {
        let Some(&fid) = self.page_table.get(&page_id) else {
            return Ok(()); // not resident — disk image is already authoritative
        };
        let f = &mut self.frames[fid];
        if !f.dirty {
            return Ok(());
        }
        let pg = f.page.read().unwrap();
        self.wal.flush_to(pg.page_lsn())?;
        self.disk.write_page(page_id, pg.as_bytes())?;
        drop(pg);
        f.dirty = false;
        self.dpt.remove(&page_id);
        Ok(())
    }

    fn flush_all_locked(&mut self) -> Result<()> {
        // Flush the entire WAL first — every page we're about to write is
        // covered by `flushed_lsn >= page.page_lsn` after this.
        self.wal.flush()?;
        for fid in 0..self.frames.len() {
            let f = &mut self.frames[fid];
            if let Some(pid) = f.page_id {
                if f.dirty {
                    let pg = f.page.read().unwrap();
                    self.disk.write_page(pid, pg.as_bytes())?;
                    drop(pg);
                    f.dirty = false;
                    self.dpt.remove(&pid);
                }
            }
        }
        Ok(())
    }
}

/// RAII handle holding a pin on a buffer-pool frame. Drop releases the pin.
/// Page contents are accessed via [`PageGuard::read`] / [`PageGuard::write`],
/// which return standard `RwLock` guards over the page bytes. Multiple
/// `PageGuard`s for *different* pages can coexist across threads; multiple
/// for the *same* page coordinate via the inner RwLock.
pub struct PageGuard {
    pool: BufferPool,
    page_id: PageId,
    page: Arc<RwLock<Page>>,
    mutated: AtomicBool,
}

impl PageGuard {
    fn new(pool: BufferPool, page_id: PageId, page: Arc<RwLock<Page>>) -> Self {
        Self {
            pool,
            page_id,
            page,
            mutated: AtomicBool::new(false),
        }
    }

    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    pub fn read(&self) -> RwLockReadGuard<'_, Page> {
        self.page.read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, Page> {
        self.mutated.store(true, ORDER);
        self.page.write().unwrap()
    }
}

impl Drop for PageGuard {
    fn drop(&mut self) {
        let mutated = self.mutated.load(ORDER);
        self.pool.release(self.page_id, mutated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Barrier;
    use std::thread;

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

    fn fresh_pool(capacity: usize, path: &std::path::Path) -> BufferPool {
        let disk = DiskManager::open(path).unwrap();
        let wal_path = path.with_extension("wal");
        let wal = Arc::new(WalManager::open(&wal_path).unwrap());
        BufferPool::new(disk, capacity, wal)
    }

    #[test]
    fn allocate_and_persist_via_flush() {
        let path = temp_path("flush");
        let pool = fresh_pool(2, &path);
        {
            let g = pool.new_page().unwrap();
            assert_eq!(g.page_id(), 0);
            g.write().insert(b"hello").unwrap();
        }
        pool.flush_all().unwrap();

        // Reopen via a fresh pool and read it back.
        drop(pool);
        let pool = fresh_pool(2, &path);
        let g = pool.fetch_page(0).unwrap();
        assert_eq!(g.read().get_tuple(0).unwrap(), b"hello");
        drop(g);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn lru_evicts_oldest_unpinned() {
        let path = temp_path("evict");
        let pool = fresh_pool(2, &path);
        {
            let g = pool.new_page().unwrap();
            assert_eq!(g.page_id(), 0);
            g.write().insert(b"page0").unwrap();
        }
        {
            let g = pool.new_page().unwrap();
            assert_eq!(g.page_id(), 1);
            g.write().insert(b"page1").unwrap();
        }
        {
            let g = pool.new_page().unwrap();
            assert_eq!(g.page_id(), 2);
            g.write().insert(b"page2").unwrap();
        }
        // Fetching page 0 must round-trip via disk.
        let g = pool.fetch_page(0).unwrap();
        assert_eq!(g.read().get_tuple(0).unwrap(), b"page0");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_only_guard_does_not_dirty() {
        let path = temp_path("clean");
        let pool = fresh_pool(2, &path);
        {
            let g = pool.new_page().unwrap();
            g.write().insert(b"x").unwrap();
        }
        pool.flush_all().unwrap();
        {
            let g = pool.fetch_page(0).unwrap();
            let _ = g.read().tuple_count();
        }
        // Force eviction by filling the pool with new pages.
        let _ = pool.new_page().unwrap();
        let _ = pool.new_page().unwrap();
        let g = pool.fetch_page(0).unwrap();
        assert_eq!(g.read().get_tuple(0).unwrap(), b"x");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn concurrent_readers_and_writers_on_different_pages() {
        let path = temp_path("concurrent");
        let pool = fresh_pool(8, &path);
        {
            let g = pool.new_page().unwrap();
            g.write().insert(b"a").unwrap();
        }
        {
            let g = pool.new_page().unwrap();
            g.write().insert(b"b").unwrap();
        }
        let barrier = Arc::new(Barrier::new(2));

        let p1 = pool.clone();
        let b1 = Arc::clone(&barrier);
        let h1 = thread::spawn(move || {
            b1.wait();
            for _ in 0..50 {
                let g = p1.fetch_page(0).unwrap();
                assert_eq!(g.read().get_tuple(0).unwrap(), b"a");
            }
        });

        let p2 = pool.clone();
        let b2 = Arc::clone(&barrier);
        let h2 = thread::spawn(move || {
            b2.wait();
            for _ in 0..50 {
                let g = p2.fetch_page(1).unwrap();
                assert_eq!(g.read().get_tuple(0).unwrap(), b"b");
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();
        std::fs::remove_file(&path).ok();
    }
}
