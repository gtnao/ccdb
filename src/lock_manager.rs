//! Row-level lock manager for Strict Two-Phase Locking (S2PL).
//!
//! Each (page_id, slot_id) row has a [`LockState`] tracking current holders
//! and a FIFO wait queue. Two-phase locking means locks are acquired during
//! statement execution and released only at COMMIT / ROLLBACK.
//!
//! # Status in this codebase
//! The component is fully implemented and tested under multi-threaded
//! contention, but it is **not yet wired into the executor pipeline**.
//! The executor stays single-threaded for now (one connection at a time);
//! integrating S/X locks per row requires also restructuring the buffer
//! pool (`Arc<RwLock<Page>>`) and the connection layer (per-connection
//! threads). That is a focused future change. The lock manager is ready
//! when we get there.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use crate::page::Rid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Shared (reader). Compatible with other Shared locks.
    Shared,
    /// Exclusive (writer). Compatible with no other lock.
    Exclusive,
}

#[derive(Debug)]
pub enum LockError {
    Timeout,
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Timeout => {
                write!(f, "lock acquisition timeout (possible deadlock)")
            }
        }
    }
}

impl std::error::Error for LockError {}

#[derive(Debug)]
struct LockRequest {
    txn_id: u64,
    mode: LockMode,
}

#[derive(Debug)]
struct LockState {
    /// Current holders. Each txn appears at most once.
    holders: HashMap<u64, LockMode>,
    /// FIFO queue of waiters.
    wait_queue: VecDeque<LockRequest>,
}

impl LockState {
    fn new() -> Self {
        Self {
            holders: HashMap::new(),
            wait_queue: VecDeque::new(),
        }
    }

    /// Can `txn_id` acquire `mode` *right now*, considering current holders
    /// and FIFO ordering against the wait queue?
    fn can_grant(&self, txn_id: u64, mode: LockMode) -> bool {
        if let Some(&held) = self.holders.get(&txn_id) {
            // Already holding. Re-acquire is OK if the existing lock covers it.
            if held == LockMode::Exclusive || mode == LockMode::Shared {
                return true;
            }
            // Upgrade S → X: only if we are the sole holder.
            return self.holders.len() == 1;
        }

        // Not yet a holder. Must be compatible with all current holders AND
        // not jump ahead of any waiter (FIFO).
        if !self.wait_queue.is_empty() {
            return false;
        }
        match mode {
            LockMode::Shared => self.holders.values().all(|m| *m == LockMode::Shared),
            LockMode::Exclusive => self.holders.is_empty(),
        }
    }
}

pub struct LockManager {
    table: Mutex<HashMap<Rid, LockState>>,
    cond: Condvar,
    timeout: Duration,
}

impl LockManager {
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            table: Mutex::new(HashMap::new()),
            cond: Condvar::new(),
            timeout,
        }
    }

    /// Acquire `mode` on `rid` for `txn_id`. Blocks until granted or until
    /// the configured timeout elapses (interpreted as a deadlock signal).
    pub fn lock(&self, txn_id: u64, rid: Rid, mode: LockMode) -> Result<(), LockError> {
        let mut table = self.table.lock().unwrap();

        let state = table.entry(rid).or_insert_with(LockState::new);

        if state.can_grant(txn_id, mode) {
            state.holders.insert(txn_id, mode);
            return Ok(());
        }

        // Must wait. Enqueue, then sleep on the Condvar.
        state.wait_queue.push_back(LockRequest { txn_id, mode });

        let (mut table, timed_out) = self
            .cond
            .wait_timeout_while(table, self.timeout, |t| {
                let s = t.get(&rid).expect("rid entry was removed while we slept");
                // Keep waiting while we are NOT a holder yet.
                !s.holders.contains_key(&txn_id)
            })
            .expect("condvar poisoned");

        if timed_out.timed_out() {
            // Remove ourselves from the queue so we don't get a phantom grant.
            if let Some(s) = table.get_mut(&rid) {
                s.wait_queue.retain(|r| r.txn_id != txn_id);
            }
            return Err(LockError::Timeout);
        }
        Ok(())
    }

    /// Release every lock held by `txn_id`. Wakes up any waiters that can
    /// now be granted.
    pub fn unlock_all(&self, txn_id: u64, held: &HashSet<Rid>) {
        let mut table = self.table.lock().unwrap();

        for rid in held {
            if let Some(state) = table.get_mut(rid) {
                state.holders.remove(&txn_id);
                grant_waiting(state);
                if state.holders.is_empty() && state.wait_queue.is_empty() {
                    table.remove(rid);
                }
            }
        }
        // notify_all is over-broadcast, but each thread re-checks its own
        // predicate, and lock contention is the slow path anyway.
        self.cond.notify_all();
    }

    #[cfg(test)]
    fn holders_of(&self, rid: Rid) -> Vec<(u64, LockMode)> {
        let table = self.table.lock().unwrap();
        table
            .get(&rid)
            .map(|s| s.holders.iter().map(|(k, v)| (*k, *v)).collect())
            .unwrap_or_default()
    }
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk the wait queue and grant every request that's now compatible.
/// Stops at the first ungrantable Exclusive request (FIFO ordering for
/// writers — keeps them from being starved by a stream of readers).
fn grant_waiting(state: &mut LockState) {
    let mut i = 0;
    while i < state.wait_queue.len() {
        let req = &state.wait_queue[i];
        let txn_id = req.txn_id;
        let mode = req.mode;

        let grantable = if state.holders.is_empty() {
            true
        } else {
            match mode {
                LockMode::Shared => state.holders.values().all(|m| *m == LockMode::Shared),
                LockMode::Exclusive => {
                    // Upgrade case: sole holder is the requesting txn itself.
                    state.holders.len() == 1 && state.holders.contains_key(&txn_id)
                }
            }
        };

        if grantable {
            state.holders.insert(txn_id, mode);
            state.wait_queue.remove(i);
        } else {
            if mode == LockMode::Exclusive {
                // Prevent reader-starvation of this waiting writer.
                break;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    fn rid(p: u32, s: u16) -> Rid {
        (p, s)
    }

    #[test]
    fn single_thread_lock_unlock() {
        let lm = LockManager::new();
        let r = rid(0, 0);
        lm.lock(1, r, LockMode::Shared).unwrap();
        lm.lock(1, r, LockMode::Shared).unwrap(); // re-acquire is fine
        let mut held = HashSet::new();
        held.insert(r);
        lm.unlock_all(1, &held);
        assert!(lm.holders_of(r).is_empty());
    }

    #[test]
    fn two_shared_locks_coexist() {
        let lm = LockManager::new();
        let r = rid(0, 0);
        lm.lock(1, r, LockMode::Shared).unwrap();
        lm.lock(2, r, LockMode::Shared).unwrap();
        assert_eq!(lm.holders_of(r).len(), 2);
    }

    #[test]
    fn exclusive_blocks_until_release() {
        let lm = Arc::new(LockManager::new());
        let r = rid(0, 0);
        lm.lock(1, r, LockMode::Exclusive).unwrap();

        let lm2 = Arc::clone(&lm);
        let (tx_started, rx_started) = mpsc::channel();
        let (tx_done, rx_done) = mpsc::channel();
        let h = thread::spawn(move || {
            tx_started.send(()).unwrap();
            lm2.lock(2, r, LockMode::Exclusive).unwrap();
            tx_done.send(()).unwrap();
        });

        rx_started.recv().unwrap();
        // Give the other thread time to enter the wait state.
        thread::sleep(Duration::from_millis(50));
        // It must NOT have acquired the lock yet.
        assert!(rx_done.try_recv().is_err());

        // Release. The waiter should now be granted.
        let mut held = HashSet::new();
        held.insert(r);
        lm.unlock_all(1, &held);
        rx_done.recv_timeout(Duration::from_secs(2)).unwrap();
        h.join().unwrap();
    }

    #[test]
    fn upgrade_succeeds_when_sole_holder() {
        let lm = LockManager::new();
        let r = rid(0, 0);
        lm.lock(1, r, LockMode::Shared).unwrap();
        lm.lock(1, r, LockMode::Exclusive).unwrap();
        let holders = lm.holders_of(r);
        assert_eq!(holders, vec![(1, LockMode::Exclusive)]);
    }

    #[test]
    fn timeout_on_unreleased_exclusive() {
        let lm = LockManager::with_timeout(Duration::from_millis(100));
        let r = rid(0, 0);
        lm.lock(1, r, LockMode::Exclusive).unwrap();
        let started = Instant::now();
        let err = lm.lock(2, r, LockMode::Exclusive);
        assert!(matches!(err, Err(LockError::Timeout)));
        // Verify it actually waited (not an instant rejection).
        assert!(started.elapsed() >= Duration::from_millis(80));
        // After timeout, the queue is cleaned up and original holder remains.
        assert_eq!(lm.holders_of(r), vec![(1, LockMode::Exclusive)]);
    }

    #[test]
    fn deadlock_resolves_via_timeout() {
        // Two threads each hold one rid and try to acquire the other.
        let lm = Arc::new(LockManager::with_timeout(Duration::from_millis(150)));
        let r1 = rid(0, 0);
        let r2 = rid(0, 1);

        let lm_a = Arc::clone(&lm);
        let lm_b = Arc::clone(&lm);
        let (sa, ra) = mpsc::channel();
        let (sb, rb) = mpsc::channel();

        let h1 = thread::spawn(move || {
            lm_a.lock(1, r1, LockMode::Exclusive).unwrap();
            sa.send(()).unwrap();
            // Wait for other side to grab r2 before contending.
            rb.recv().unwrap();
            let res = lm_a.lock(1, r2, LockMode::Exclusive);
            let mut held = HashSet::new();
            held.insert(r1);
            // If we acquired r2 (impossible here), include it before unlocking.
            if res.is_ok() {
                held.insert(r2);
            }
            lm_a.unlock_all(1, &held);
            res
        });
        let h2 = thread::spawn(move || {
            lm_b.lock(2, r2, LockMode::Exclusive).unwrap();
            sb.send(()).unwrap();
            ra.recv().unwrap();
            let res = lm_b.lock(2, r1, LockMode::Exclusive);
            let mut held = HashSet::new();
            held.insert(r2);
            if res.is_ok() {
                held.insert(r1);
            }
            lm_b.unlock_all(2, &held);
            res
        });

        let r1_res = h1.join().unwrap();
        let r2_res = h2.join().unwrap();
        // At least one of them must have timed out.
        assert!(
            matches!(r1_res, Err(LockError::Timeout))
                || matches!(r2_res, Err(LockError::Timeout)),
            "expected at least one timeout, got {r1_res:?} and {r2_res:?}"
        );
    }

    #[test]
    fn pending_exclusive_blocks_subsequent_shared() {
        // Verify FIFO: a pending X must not be jumped by a later S request.
        let lm = Arc::new(LockManager::with_timeout(Duration::from_millis(150)));
        let r = rid(0, 0);
        // T1 holds S.
        lm.lock(1, r, LockMode::Shared).unwrap();

        // T2 requests X; will wait behind T1.
        let lm2 = Arc::clone(&lm);
        let h2 = thread::spawn(move || lm2.lock(2, r, LockMode::Exclusive));
        // Give T2 time to enqueue.
        thread::sleep(Duration::from_millis(30));

        // T3 requests S; FIFO says it must wait behind T2 (the pending X)
        // even though S would otherwise be compatible with T1's S.
        let lm3 = Arc::clone(&lm);
        let h3 = thread::spawn(move || lm3.lock(3, r, LockMode::Shared));

        // Both should time out because T1 never releases.
        let r2 = h2.join().unwrap();
        let r3 = h3.join().unwrap();
        assert!(matches!(r2, Err(LockError::Timeout)));
        assert!(matches!(r3, Err(LockError::Timeout)));
    }
}
