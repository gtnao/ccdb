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

use crate::disk::DiskManager;
use crate::page::{PAGE_SIZE, Page, PageId};
use crate::wal::{Lsn, WalManager};

const ORDER: Ordering = Ordering::SeqCst;
const NIL: usize = usize::MAX;

/// Replacement policy for choosing an evictable frame.
trait Replacer {
    fn victim(&mut self) -> Option<usize>;
    fn pin(&mut self, frame_id: usize);
    fn unpin(&mut self, frame_id: usize);
}

/// O(1) LRU using a doubly-linked list embedded in two Vec arrays.
/// `head` is the least-recently-used (eviction candidate), `tail` is
/// the most recently unpinned. Pinned frames are simply unlinked from
/// the list — `in_list[fid]` says whether a frame is currently
/// participating, so pin/unpin become O(1) instead of an O(n) scan.
struct LruReplacer {
    prev: Vec<usize>,
    next: Vec<usize>,
    in_list: Vec<bool>,
    head: usize,
    tail: usize,
}

impl LruReplacer {
    fn new(capacity: usize) -> Self {
        Self {
            prev: vec![NIL; capacity],
            next: vec![NIL; capacity],
            in_list: vec![false; capacity],
            head: NIL,
            tail: NIL,
        }
    }

    fn unlink(&mut self, fid: usize) {
        if !self.in_list[fid] {
            return;
        }
        let p = self.prev[fid];
        let n = self.next[fid];
        if p != NIL {
            self.next[p] = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.prev[n] = p;
        } else {
            self.tail = p;
        }
        self.prev[fid] = NIL;
        self.next[fid] = NIL;
        self.in_list[fid] = false;
    }

    fn push_back(&mut self, fid: usize) {
        if self.in_list[fid] {
            return;
        }
        self.prev[fid] = self.tail;
        self.next[fid] = NIL;
        if self.tail != NIL {
            self.next[self.tail] = fid;
        } else {
            self.head = fid;
        }
        self.tail = fid;
        self.in_list[fid] = true;
    }
}

impl Replacer for LruReplacer {
    fn victim(&mut self) -> Option<usize> {
        if self.head == NIL {
            return None;
        }
        let fid = self.head;
        self.unlink(fid);
        Some(fid)
    }
    fn pin(&mut self, frame_id: usize) {
        self.unlink(frame_id);
    }
    fn unpin(&mut self, frame_id: usize) {
        self.unlink(frame_id);
        self.push_back(frame_id);
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

/// One partition of the buffer pool. Per-shard Mutex isolates fetch /
/// new / release on different page-id ranges so multi-threaded workloads
/// stop serialising on a single global lock. `disk` and `wal` are shared
/// because both already accept `&self` (DiskManager via pread/pwrite,
/// WalManager via its internal Mutex).
struct Shard {
    frames: Vec<Frame>,
    page_table: HashMap<PageId, usize>,
    replacer: LruReplacer,
    /// Per-shard slice of the global capacity.
    capacity: usize,
    dpt: HashMap<PageId, Lsn>,
    free_list: Vec<PageId>,
    free_frames: Vec<usize>,
}

const NUM_SHARDS: usize = 16;

#[derive(Clone)]
pub struct BufferPool {
    shards: Arc<Vec<Mutex<Shard>>>,
    disk: Arc<DiskManager>,
    wal: Arc<WalManager>,
}

fn shard_idx(page_id: PageId) -> usize {
    (page_id as usize) % NUM_SHARDS
}

impl BufferPool {
    pub fn new(disk: DiskManager, capacity: usize, wal: Arc<WalManager>) -> Self {
        let per_shard = capacity.div_ceil(NUM_SHARDS).max(1);
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            let frames: Vec<Frame> = (0..per_shard).map(|_| Frame::empty()).collect();
            let free_frames = (0..per_shard).rev().collect();
            shards.push(Mutex::new(Shard {
                frames,
                page_table: HashMap::new(),
                replacer: LruReplacer::new(per_shard),
                capacity: per_shard,
                dpt: HashMap::new(),
                free_list: Vec::new(),
                free_frames,
            }));
        }
        Self {
            shards: Arc::new(shards),
            disk: Arc::new(disk),
            wal,
        }
    }

    fn shard(&self, page_id: PageId) -> &Mutex<Shard> {
        &self.shards[shard_idx(page_id)]
    }

    /// Hand a page back to the free list of *its* shard. Caller (VACUUM)
    /// has already removed every reachable reference.
    pub fn recycle_page(&self, page_id: PageId) {
        let mut s = self.shard(page_id).lock().unwrap();
        s.free_list.push(page_id);
    }

    /// Snapshot of the current DPT (merged across shards). Used by the
    /// fuzzy checkpoint record.
    pub fn dpt_snapshot(&self) -> HashMap<PageId, Lsn> {
        let mut out = HashMap::new();
        for sh in self.shards.iter() {
            let s = sh.lock().unwrap();
            for (pid, lsn) in s.dpt.iter() {
                out.insert(*pid, *lsn);
            }
        }
        out
    }

    pub fn page_count(&self) -> u32 {
        self.disk.page_count()
    }

    pub fn fetch_page(&self, page_id: PageId) -> Result<PageGuard> {
        let mut s = self.shard(page_id).lock().unwrap();
        let (page_arc, _frame_id) = s.fetch_locked(page_id, &self.disk, &self.wal)?;
        Ok(PageGuard::new(self.clone(), page_id, page_arc))
    }

    pub fn new_page(&self) -> Result<PageGuard> {
        // Try to reuse a recycled page id from any shard before extending
        // the file. We look at our routing shard's free_list first; if
        // it's empty, fall through to disk allocation.
        // (Not strictly required for correctness — recycle_page deposits
        // into the page's home shard, so by the time someone calls
        // new_page that shard is the right place to look.)
        // First pass: try to find a recycled page id quickly.
        for sh_idx in 0..NUM_SHARDS {
            let mut s = self.shards[sh_idx].lock().unwrap();
            if let Some(page_id) = s.free_list.pop() {
                if shard_idx(page_id) != sh_idx {
                    // The page id belongs to a different shard. Re-route.
                    drop(s);
                    let mut home = self.shard(page_id).lock().unwrap();
                    let (pid, arc) =
                        home.new_recycled(page_id, &self.disk, &self.wal)?;
                    return Ok(PageGuard::new(self.clone(), pid, arc));
                }
                let (pid, arc) = s.new_recycled(page_id, &self.disk, &self.wal)?;
                return Ok(PageGuard::new(self.clone(), pid, arc));
            }
        }
        // No recycled id available; extend the file.
        let page_id = self.disk.allocate_page()?;
        let mut s = self.shard(page_id).lock().unwrap();
        let arc = s.new_fresh(page_id, &self.disk, &self.wal)?;
        Ok(PageGuard::new(self.clone(), page_id, arc))
    }

    pub fn flush_all(&self) -> Result<()> {
        // Flush WAL once first — covers every dirty page about to be
        // written from any shard.
        self.wal.flush()?;
        for sh in self.shards.iter() {
            let mut s = sh.lock().unwrap();
            s.flush_all_locked(&self.disk)?;
        }
        self.disk.sync()?;
        Ok(())
    }

    /// Flush a single page synchronously (caller wants it durable now).
    pub fn flush_page(&self, page_id: PageId) -> Result<()> {
        let mut s = self.shard(page_id).lock().unwrap();
        s.flush_page_locked(page_id, &self.disk, &self.wal)
    }

    fn release(&self, page_id: PageId, mutated: bool) {
        let mut s = self.shard(page_id).lock().unwrap();
        s.release_locked(page_id, mutated);
    }
}

impl Shard {
    fn pick_or_evict(&mut self, disk: &DiskManager, wal: &WalManager) -> Result<usize> {
        if let Some(fid) = self.free_frames.pop() {
            return Ok(fid);
        }
        let victim = self
            .replacer
            .victim()
            .ok_or_else(|| anyhow::anyhow!("buffer pool exhausted: all frames pinned"))?;
        self.evict(victim, disk, wal)?;
        Ok(victim)
    }

    fn evict(&mut self, frame_id: usize, disk: &DiskManager, wal: &WalManager) -> Result<()> {
        let frame = &mut self.frames[frame_id];
        if let Some(pid) = frame.page_id {
            if frame.dirty {
                let page_guard = frame.page.read().unwrap();
                wal.flush_to(page_guard.page_lsn())?;
                disk.write_page(pid, page_guard.as_bytes())?;
            }
            self.page_table.remove(&pid);
            self.dpt.remove(&pid);
        }
        frame.page_id = None;
        frame.dirty = false;
        frame.pin_count = 0;
        Ok(())
    }

    fn fetch_locked(
        &mut self,
        page_id: PageId,
        disk: &DiskManager,
        wal: &WalManager,
    ) -> Result<(Arc<RwLock<Page>>, usize)> {
        if let Some(&fid) = self.page_table.get(&page_id) {
            self.frames[fid].pin_count += 1;
            self.replacer.pin(fid);
            return Ok((Arc::clone(&self.frames[fid].page), fid));
        }
        let fid = self.pick_or_evict(disk, wal)?;
        let mut buf = [0u8; PAGE_SIZE];
        disk.read_page(page_id, &mut buf)?;
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

    /// Set up a frame for a recycled page id. Caller has already ensured
    /// this is the home shard for `page_id`. If a stale resident copy
    /// existed, evict it first.
    fn new_recycled(
        &mut self,
        page_id: PageId,
        disk: &DiskManager,
        wal: &WalManager,
    ) -> Result<(PageId, Arc<RwLock<Page>>)> {
        if let Some(&fid) = self.page_table.get(&page_id) {
            self.replacer.pin(fid);
            self.evict(fid, disk, wal)?;
            self.free_frames.push(fid);
        }
        let fid = self.pick_or_evict(disk, wal)?;
        {
            let mut p = self.frames[fid].page.write().unwrap();
            *p = Page::new(page_id);
        }
        let f = &mut self.frames[fid];
        f.page_id = Some(page_id);
        f.pin_count = 1;
        f.dirty = true;
        self.page_table.insert(page_id, fid);
        self.replacer.pin(fid);
        Ok((page_id, Arc::clone(&f.page)))
    }

    fn new_fresh(
        &mut self,
        page_id: PageId,
        disk: &DiskManager,
        wal: &WalManager,
    ) -> Result<Arc<RwLock<Page>>> {
        let fid = self.pick_or_evict(disk, wal)?;
        {
            let mut p = self.frames[fid].page.write().unwrap();
            *p = Page::new(page_id);
        }
        let f = &mut self.frames[fid];
        f.page_id = Some(page_id);
        f.pin_count = 1;
        f.dirty = true;
        self.page_table.insert(page_id, fid);
        self.replacer.pin(fid);
        Ok(Arc::clone(&f.page))
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
                    let page_lsn = f.page.read().unwrap().page_lsn();
                    self.dpt.entry(page_id).or_insert(page_lsn);
                }
            }
            if f.pin_count == 0 {
                self.replacer.unpin(fid);
            }
        }
    }

    fn flush_page_locked(
        &mut self,
        page_id: PageId,
        disk: &DiskManager,
        wal: &WalManager,
    ) -> Result<()> {
        let Some(&fid) = self.page_table.get(&page_id) else {
            return Ok(());
        };
        let f = &mut self.frames[fid];
        if !f.dirty {
            return Ok(());
        }
        let pg = f.page.read().unwrap();
        wal.flush_to(pg.page_lsn())?;
        disk.write_page(page_id, pg.as_bytes())?;
        drop(pg);
        f.dirty = false;
        self.dpt.remove(&page_id);
        disk.sync()?;
        Ok(())
    }

    fn flush_all_locked(&mut self, disk: &DiskManager) -> Result<()> {
        for fid in 0..self.frames.len() {
            let f = &mut self.frames[fid];
            if let Some(pid) = f.page_id {
                if f.dirty {
                    let pg = f.page.read().unwrap();
                    disk.write_page(pid, pg.as_bytes())?;
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
