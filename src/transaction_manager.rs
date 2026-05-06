//! Global transaction registry: allocates txn_ids and maintains the
//! Active Transaction Table (ATT) used by checkpoint and recovery.
//!
//! ATT entries are `txn_id → last_lsn` for every txn that has begun but
//! not yet committed/aborted. A checkpoint record persists a snapshot of
//! the ATT so recovery's analysis phase can start from a known good
//! state instead of scanning from the beginning of the WAL.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::clog::Clog;
use crate::tuple::TxnId;
use crate::wal::Lsn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    InProgress,
    Committed,
    Aborted,
}

/// Snapshot of the database state as of a transaction's start, used by
/// MVCC visibility to honour Snapshot Isolation.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The owning transaction's id.
    pub txn_id: TxnId,
    /// Highest txn_id that the snapshot considers "future" — anything >=
    /// xmax was started after our snapshot.
    pub xmax: TxnId,
    /// Transactions that were in progress when the snapshot was taken.
    /// Their writes (xmin) and deletes (xmax) are *not* visible.
    pub active: HashSet<TxnId>,
}

#[derive(Debug)]
pub struct TransactionManager {
    next_txn_id: AtomicU64,
    /// Active Transaction Table: txn_id → last_lsn.
    att: Mutex<HashMap<u64, Lsn>>,
    /// Persistent commit/abort status. Read on every visibility check, so
    /// the inner page cache absorbs the load. Reaches disk only at flush
    /// (driven by CHECKPOINT).
    clog: Arc<Clog>,
}

impl TransactionManager {
    pub fn new(clog: Arc<Clog>) -> Self {
        Self {
            next_txn_id: AtomicU64::new(1),
            att: Mutex::new(HashMap::new()),
            clog,
        }
    }

    pub fn clog(&self) -> &Arc<Clog> {
        &self.clog
    }

    pub fn next_txn_id(&self) -> u64 {
        self.next_txn_id.load(Ordering::SeqCst)
    }

    pub fn begin(&self) -> u64 {
        let id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        self.att.lock().unwrap().insert(id, 0);
        id
    }

    pub fn update_last_lsn(&self, txn_id: u64, lsn: Lsn) {
        if let Some(entry) = self.att.lock().unwrap().get_mut(&txn_id) {
            *entry = lsn;
        }
    }

    pub fn commit(&self, txn_id: u64) {
        self.att.lock().unwrap().remove(&txn_id);
        // First decision wins: don't overwrite an existing terminal status
        // (e.g. a stale refresh_autocommit() after ROLLBACK).
        if matches!(
            self.clog.get(txn_id).unwrap_or(TxnStatus::InProgress),
            TxnStatus::InProgress
        ) {
            let _ = self.clog.set(txn_id, TxnStatus::Committed);
        }
    }

    pub fn abort(&self, txn_id: u64) {
        self.att.lock().unwrap().remove(&txn_id);
        if matches!(
            self.clog.get(txn_id).unwrap_or(TxnStatus::InProgress),
            TxnStatus::InProgress
        ) {
            let _ = self.clog.set(txn_id, TxnStatus::Aborted);
        }
    }

    pub fn att_snapshot(&self) -> HashMap<u64, Lsn> {
        self.att.lock().unwrap().clone()
    }

    pub fn set_next_txn_id(&self, id: u64) {
        self.next_txn_id.store(id, Ordering::SeqCst);
    }

    /// Take a visibility snapshot. Captures every txn currently in the ATT
    /// (i.e. in-progress) and the xmax frontier.
    pub fn snapshot(&self, txn_id: TxnId) -> Snapshot {
        let att = self.att.lock().unwrap();
        let active: HashSet<TxnId> = att.keys().copied().collect();
        let xmax = self.next_txn_id.load(Ordering::SeqCst);
        Snapshot {
            txn_id,
            xmax,
            active,
        }
    }

    /// Look up a transaction's commit/abort status. Reads from CLOG;
    /// defaults to `Aborted` for unknown txn_ids that aren't in the ATT
    /// (matches the "unfinished = aborted" recovery convention).
    ///
    /// Hot path: visibility checks call this once per tuple per scan.
    /// We memoise terminal results (Committed / Aborted) in a thread-
    /// local cache — those statuses never change once observed, so no
    /// invalidation is needed. Capped at 4096 entries to keep the
    /// per-thread footprint bounded; on overflow the cache is just
    /// flushed (cold path becomes the next miss).
    pub fn status(&self, txn_id: TxnId) -> TxnStatus {
        thread_local! {
            static CLOG_CACHE: std::cell::RefCell<
                std::collections::HashMap<TxnId, TxnStatus>,
            > = std::cell::RefCell::new(std::collections::HashMap::new());
        }
        if let Some(s) = CLOG_CACHE.with(|c| c.borrow().get(&txn_id).copied()) {
            return s;
        }
        let s = self.status_uncached(txn_id);
        if matches!(s, TxnStatus::Committed | TxnStatus::Aborted) {
            CLOG_CACHE.with(|c| {
                let mut m = c.borrow_mut();
                if m.len() >= 4096 {
                    m.clear();
                }
                m.insert(txn_id, s);
            });
        }
        s
    }

    fn status_uncached(&self, txn_id: TxnId) -> TxnStatus {
        match self.clog.get(txn_id).unwrap_or(TxnStatus::InProgress) {
            TxnStatus::Committed => TxnStatus::Committed,
            TxnStatus::Aborted => TxnStatus::Aborted,
            TxnStatus::InProgress => {
                if self.att.lock().unwrap().contains_key(&txn_id) {
                    TxnStatus::InProgress
                } else {
                    TxnStatus::Aborted
                }
            }
        }
    }

    /// Recovery uses this to seed status for txns observed in the WAL.
    pub fn record_status(&self, txn_id: TxnId, status: TxnStatus) {
        let _ = self.clog.set(txn_id, status);
    }

    /// Smallest txn_id of any in-progress transaction. Used by VACUUM as
    /// the cutoff: a tuple whose `xmax` is committed and `< oldest_xmin`
    /// can no longer be seen by anyone, so it's safe to physically reclaim.
    /// When no transactions are active, returns `next_txn_id` (everything
    /// is in the past).
    pub fn oldest_active_xmin(&self) -> TxnId {
        let att = self.att.lock().unwrap();
        att.keys()
            .copied()
            .min()
            .unwrap_or_else(|| self.next_txn_id.load(Ordering::SeqCst))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn ids_unique_under_concurrency() {
        let tm = Arc::new(TransactionManager::new(std::sync::Arc::new(crate::clog::Clog::in_memory())));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let tm = Arc::clone(&tm);
            handles.push(thread::spawn(move || {
                (0..100).map(|_| tm.begin()).collect::<Vec<_>>()
            }));
        }
        let mut all_ids = std::collections::HashSet::new();
        for h in handles {
            for id in h.join().unwrap() {
                assert!(all_ids.insert(id), "duplicate id {id}");
            }
        }
        assert_eq!(all_ids.len(), 800);
    }

    #[test]
    fn att_snapshot_reflects_lifecycle() {
        let tm = TransactionManager::new(std::sync::Arc::new(crate::clog::Clog::in_memory()));
        let a = tm.begin();
        let b = tm.begin();
        tm.update_last_lsn(a, 10);
        tm.update_last_lsn(b, 20);
        let snap = tm.att_snapshot();
        assert_eq!(snap.get(&a).copied(), Some(10));
        assert_eq!(snap.get(&b).copied(), Some(20));
        tm.commit(a);
        let snap = tm.att_snapshot();
        assert!(!snap.contains_key(&a));
        assert!(snap.contains_key(&b));
    }
}
