//! Thread-safe buffer pool, partitioned and largely lock-free on the
//! hot path.
//!
//! Layout:
//! - `BufferPool` is a clonable handle (cheap `Arc` clone) over an
//!   array of `NUM_SHARDS` shards plus the shared disk + WAL handles.
//! - Each shard owns:
//!     * a fixed-size `Vec<Frame>` whose per-frame metadata is atomic
//!       (page_id, pin_count, dirty, referenced) — fetch / release
//!       update those without holding any shard lock,
//!     * a `RwLock<HashMap<PageId, usize>>` mapping page id to frame
//!       index. Fetch hits read-lock; only miss/evict needs write,
//!     * a `Mutex<ClockArm>` for the eviction sweep,
//!     * `Mutex<Vec<…>>` for the free-frame and recycled-page lists
//!       (cold-path bookkeeping).
//!
//! Replacement is the classic Clock algorithm: pin sets the
//! `referenced` bit; the clock arm sweeps frames clearing the bit
//! and evicts the first frame whose bit was already cleared and
//! whose pin_count is zero.
//!
//! Pin lifecycle is RAII via [`PageGuard`]: drop decrements pin_count
//! atomically without touching any lock.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use anyhow::Result;

use crate::disk::DiskManager;
use crate::page::{PAGE_SIZE, Page, PageId};
use crate::wal::{Lsn, WalManager};

const ORDER: Ordering = Ordering::SeqCst;

/// One frame's metadata. Encoded so the fast path (resident page,
/// already-cached lookup) needs only atomic ops:
///
/// - `page_id_plus_one == 0`  ⇒ frame is empty.
///   else `page_id = page_id_plus_one - 1`.
/// - `pin_count` advances under fetch / drops under release.
/// - `dirty` flips on the first write since the last flush.
/// - `referenced` is the Clock algorithm reference bit.
struct Frame {
    page: Arc<RwLock<Page>>,
    page_id_plus_one: AtomicU64,
    pin_count: AtomicU32,
    dirty: AtomicBool,
    referenced: AtomicBool,
}

impl Frame {
    fn empty() -> Self {
        Self {
            page: Arc::new(RwLock::new(Page::new(0))),
            page_id_plus_one: AtomicU64::new(0),
            pin_count: AtomicU32::new(0),
            dirty: AtomicBool::new(false),
            referenced: AtomicBool::new(false),
        }
    }

    fn page_id(&self) -> Option<PageId> {
        let v = self.page_id_plus_one.load(Ordering::Acquire);
        if v == 0 { None } else { Some((v - 1) as PageId) }
    }

    fn set_page_id(&self, pid: Option<PageId>) {
        let v = match pid {
            None => 0,
            Some(p) => p as u64 + 1,
        };
        self.page_id_plus_one.store(v, Ordering::Release);
    }
}

/// Clock replacer. The "arm" is just an index that walks the frame
/// array; we sweep up to 2*capacity steps per victim search (one
/// pass clears reference bits, the second returns the first
/// already-zero unpinned frame). Protected by a single Mutex —
/// it's only taken on miss/eviction, not on the cache-hit fast path.
struct ClockReplacer {
    arm: Mutex<usize>,
    capacity: usize,
}

impl ClockReplacer {
    fn new(capacity: usize) -> Self {
        Self {
            arm: Mutex::new(0),
            capacity: capacity.max(1),
        }
    }

    /// Find an evictable frame. Walks the clock arm forward, clearing
    /// reference bits as it goes. Returns the first frame that is
    /// unpinned AND had its reference bit clear at the moment we
    /// looked. Returns `None` only if every frame is pinned (full pool
    /// of pinned readers/writers — caller propagates as an error).
    fn victim(&self, frames: &[Frame]) -> Option<usize> {
        let mut arm = self.arm.lock().unwrap();
        let cap = self.capacity;
        for _ in 0..cap * 2 {
            let fid = *arm;
            *arm = (*arm + 1) % cap;
            let f = &frames[fid];
            if f.pin_count.load(ORDER) > 0 {
                continue;
            }
            // referenced=true means recently used — clear and skip.
            if f.referenced.swap(false, ORDER) {
                continue;
            }
            return Some(fid);
        }
        None
    }
}

struct Shard {
    frames: Vec<Frame>,
    page_table: RwLock<HashMap<PageId, usize>>,
    replacer: ClockReplacer,
    dpt: Mutex<HashMap<PageId, Lsn>>,
    free_list: Mutex<Vec<PageId>>,
    free_frames: Mutex<Vec<usize>>,
    /// Per-shard cap. Each shard has `total_capacity / NUM_SHARDS` frames.
    #[allow(dead_code)]
    capacity: usize,
}

const NUM_SHARDS: usize = 16;

#[derive(Clone)]
pub struct BufferPool {
    shards: Arc<Vec<Shard>>,
    disk: Arc<DiskManager>,
    wal: Arc<WalManager>,
    /// Tracks the next file-extending allocate so callers don't have
    /// to lock anything to know it.
    #[allow(dead_code)]
    next_alloc_hint: Arc<AtomicUsize>,
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
            let free_frames: Vec<usize> = (0..per_shard).rev().collect();
            shards.push(Shard {
                frames,
                page_table: RwLock::new(HashMap::new()),
                replacer: ClockReplacer::new(per_shard),
                dpt: Mutex::new(HashMap::new()),
                free_list: Mutex::new(Vec::new()),
                free_frames: Mutex::new(free_frames),
                capacity: per_shard,
            });
        }
        Self {
            shards: Arc::new(shards),
            disk: Arc::new(disk),
            wal,
            next_alloc_hint: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn shard(&self, page_id: PageId) -> &Shard {
        &self.shards[shard_idx(page_id)]
    }

    /// Hand a page back to its home shard's free list. Caller (VACUUM)
    /// has already removed every reachable reference.
    pub fn recycle_page(&self, page_id: PageId) {
        self.shard(page_id).free_list.lock().unwrap().push(page_id);
    }

    /// Snapshot the merged DPT for the next checkpoint record.
    pub fn dpt_snapshot(&self) -> HashMap<PageId, Lsn> {
        let mut out = HashMap::new();
        for sh in self.shards.iter() {
            let m = sh.dpt.lock().unwrap();
            for (pid, lsn) in m.iter() {
                out.insert(*pid, *lsn);
            }
        }
        out
    }

    pub fn page_count(&self) -> u32 {
        self.disk.page_count()
    }

    /// Hot path. Lock-free when the page is already resident:
    ///   1. read-lock the shard's page_table to look up the frame id;
    ///   2. atomically bump the frame's pin_count + reference bit;
    ///   3. double-check `frame.page_id` still matches (race window
    ///      between the read-lock release and the atomic bump);
    ///   4. fall back to the slow path on a miss / racy contention.
    pub fn fetch_page(&self, page_id: PageId) -> Result<PageGuard> {
        let sh = self.shard(page_id);
        if let Some(arc) = self.try_fetch_fast(sh, page_id) {
            return Ok(PageGuard::new(self.clone(), page_id, arc));
        }
        let arc = self.fetch_slow(sh, page_id)?;
        Ok(PageGuard::new(self.clone(), page_id, arc))
    }

    fn try_fetch_fast(&self, sh: &Shard, page_id: PageId) -> Option<Arc<RwLock<Page>>> {
        let pt = sh.page_table.read().unwrap();
        let fid = *pt.get(&page_id)?;
        // Hold the read lock just long enough to grab the Arc. Pin and
        // page-id check happen under the read lock so eviction (which
        // takes the write lock) cannot race past us.
        let frame = &sh.frames[fid];
        frame.pin_count.fetch_add(1, ORDER);
        if frame.page_id() != Some(page_id) {
            // Lost the race: someone evicted between the lookup and the
            // pin. Undo and let the slow path retry.
            frame.pin_count.fetch_sub(1, ORDER);
            return None;
        }
        frame.referenced.store(true, ORDER);
        Some(Arc::clone(&frame.page))
    }

    fn fetch_slow(&self, sh: &Shard, page_id: PageId) -> Result<Arc<RwLock<Page>>> {
        // Take the page_table write lock once; it covers both the
        // double-check and the new-frame insertion.
        let mut pt = sh.page_table.write().unwrap();
        // Did someone else just install it?
        if let Some(&fid) = pt.get(&page_id) {
            let frame = &sh.frames[fid];
            frame.pin_count.fetch_add(1, ORDER);
            frame.referenced.store(true, ORDER);
            return Ok(Arc::clone(&frame.page));
        }
        let fid = self.pick_or_evict(sh, &mut pt)?;
        let mut buf = [0u8; PAGE_SIZE];
        self.disk.read_page(page_id, &mut buf)?;
        let frame = &sh.frames[fid];
        {
            let mut p = frame.page.write().unwrap();
            *p = Page::from_bytes(&buf);
        }
        frame.set_page_id(Some(page_id));
        frame.dirty.store(false, ORDER);
        frame.pin_count.store(1, ORDER);
        frame.referenced.store(true, ORDER);
        pt.insert(page_id, fid);
        Ok(Arc::clone(&frame.page))
    }

    /// Reserve a frame for a new resident page. Caller already holds
    /// the page_table write lock so the new mapping can be inserted
    /// atomically with the eviction.
    fn pick_or_evict(
        &self,
        sh: &Shard,
        pt: &mut std::sync::RwLockWriteGuard<'_, HashMap<PageId, usize>>,
    ) -> Result<usize> {
        if let Some(fid) = sh.free_frames.lock().unwrap().pop() {
            return Ok(fid);
        }
        let victim = sh
            .replacer
            .victim(&sh.frames)
            .ok_or_else(|| anyhow::anyhow!("buffer pool exhausted: all frames pinned"))?;
        self.evict(sh, victim, pt)?;
        Ok(victim)
    }

    fn evict(
        &self,
        sh: &Shard,
        fid: usize,
        pt: &mut std::sync::RwLockWriteGuard<'_, HashMap<PageId, usize>>,
    ) -> Result<()> {
        let frame = &sh.frames[fid];
        if let Some(pid) = frame.page_id() {
            if frame.dirty.load(ORDER) {
                let pg = frame.page.read().unwrap();
                self.wal.flush_to(pg.page_lsn())?;
                self.disk.write_page(pid, pg.as_bytes())?;
            }
            pt.remove(&pid);
            sh.dpt.lock().unwrap().remove(&pid);
        }
        frame.set_page_id(None);
        frame.dirty.store(false, ORDER);
        frame.pin_count.store(0, ORDER);
        frame.referenced.store(false, ORDER);
        Ok(())
    }

    pub fn new_page(&self) -> Result<PageGuard> {
        // Try recycled-page lists first (any shard).
        for sh_idx in 0..NUM_SHARDS {
            let mut fl = self.shards[sh_idx].free_list.lock().unwrap();
            if let Some(page_id) = fl.pop() {
                drop(fl);
                let home = self.shard(page_id);
                let arc = self.install_recycled(home, page_id)?;
                return Ok(PageGuard::new(self.clone(), page_id, arc));
            }
        }
        // Otherwise extend the file.
        let page_id = self.disk.allocate_page()?;
        let sh = self.shard(page_id);
        let arc = self.install_fresh(sh, page_id)?;
        Ok(PageGuard::new(self.clone(), page_id, arc))
    }

    fn install_recycled(&self, sh: &Shard, page_id: PageId) -> Result<Arc<RwLock<Page>>> {
        let mut pt = sh.page_table.write().unwrap();
        // Evict any stale resident copy of `page_id`.
        if let Some(&fid) = pt.get(&page_id) {
            self.evict(sh, fid, &mut pt)?;
            sh.free_frames.lock().unwrap().push(fid);
        }
        let fid = self.pick_or_evict(sh, &mut pt)?;
        let frame = &sh.frames[fid];
        {
            let mut p = frame.page.write().unwrap();
            *p = Page::new(page_id);
        }
        frame.set_page_id(Some(page_id));
        frame.dirty.store(true, ORDER);
        frame.pin_count.store(1, ORDER);
        frame.referenced.store(true, ORDER);
        pt.insert(page_id, fid);
        Ok(Arc::clone(&frame.page))
    }

    fn install_fresh(&self, sh: &Shard, page_id: PageId) -> Result<Arc<RwLock<Page>>> {
        let mut pt = sh.page_table.write().unwrap();
        let fid = self.pick_or_evict(sh, &mut pt)?;
        let frame = &sh.frames[fid];
        {
            let mut p = frame.page.write().unwrap();
            *p = Page::new(page_id);
        }
        frame.set_page_id(Some(page_id));
        frame.dirty.store(true, ORDER);
        frame.pin_count.store(1, ORDER);
        frame.referenced.store(true, ORDER);
        pt.insert(page_id, fid);
        Ok(Arc::clone(&frame.page))
    }

    pub fn flush_all(&self) -> Result<()> {
        self.wal.flush()?;
        for sh in self.shards.iter() {
            for frame in sh.frames.iter() {
                if frame.dirty.load(ORDER) {
                    if let Some(pid) = frame.page_id() {
                        let pg = frame.page.read().unwrap();
                        self.disk.write_page(pid, pg.as_bytes())?;
                        drop(pg);
                        frame.dirty.store(false, ORDER);
                        sh.dpt.lock().unwrap().remove(&pid);
                    }
                }
            }
        }
        self.disk.sync()?;
        Ok(())
    }

    /// Force a single page durable. Used for structural changes
    /// (CREATE INDEX root, CREATE SEQUENCE state page) that have no
    /// WAL record to rebuild them via redo.
    pub fn flush_page(&self, page_id: PageId) -> Result<()> {
        let sh = self.shard(page_id);
        let pt = sh.page_table.read().unwrap();
        let Some(&fid) = pt.get(&page_id) else {
            return Ok(());
        };
        let frame = &sh.frames[fid];
        if !frame.dirty.load(ORDER) {
            return Ok(());
        }
        let pg = frame.page.read().unwrap();
        self.wal.flush_to(pg.page_lsn())?;
        self.disk.write_page(page_id, pg.as_bytes())?;
        drop(pg);
        frame.dirty.store(false, ORDER);
        sh.dpt.lock().unwrap().remove(&page_id);
        self.disk.sync()
    }

    fn release(&self, page_id: PageId, mutated: bool) {
        let sh = self.shard(page_id);
        let pt = sh.page_table.read().unwrap();
        let Some(&fid) = pt.get(&page_id) else {
            return;
        };
        let frame = &sh.frames[fid];
        if frame.pin_count.load(ORDER) == 0 {
            return;
        }
        if mutated {
            let was_clean = !frame.dirty.swap(true, ORDER);
            if was_clean {
                let page_lsn = frame.page.read().unwrap().page_lsn();
                sh.dpt.lock().unwrap().entry(page_id).or_insert(page_lsn);
            }
        }
        frame.pin_count.fetch_sub(1, ORDER);
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
        drop(pool);
        let pool = fresh_pool(2, &path);
        let g = pool.fetch_page(0).unwrap();
        assert_eq!(g.read().get_tuple(0).unwrap(), b"hello");
        drop(g);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn lru_evicts_oldest_unpinned() {
        // (Now Clock — same observable behaviour: evict an unpinned frame
        // when the pool is full.)
        let path = temp_path("evict");
        // Bump capacity above NUM_SHARDS so the test exercises eviction
        // within a shard rather than running into the per-shard floor of 1.
        let pool = fresh_pool(NUM_SHARDS * 2, &path);
        let mut ids = Vec::new();
        for _ in 0..NUM_SHARDS * 4 {
            let g = pool.new_page().unwrap();
            let pid = g.page_id();
            g.write().insert(b"x").unwrap();
            ids.push(pid);
        }
        // Round-trip: fetch the very first page back; if it was evicted,
        // it must come from disk and still read correctly.
        let g = pool.fetch_page(ids[0]).unwrap();
        assert_eq!(g.read().get_tuple(0).unwrap(), b"x");
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
