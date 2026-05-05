//! Global transaction registry: allocates txn_ids and maintains the
//! Active Transaction Table (ATT) used by checkpoint and recovery.
//!
//! ATT entries are `txn_id → last_lsn` for every txn that has begun but
//! not yet committed/aborted. A checkpoint record persists a snapshot of
//! the ATT so recovery's analysis phase can start from a known good
//! state instead of scanning from the beginning of the WAL.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::wal::Lsn;

#[derive(Debug)]
pub struct TransactionManager {
    next_txn_id: AtomicU64,
    att: Mutex<HashMap<u64, Lsn>>,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            next_txn_id: AtomicU64::new(1),
            att: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate a fresh txn_id and add it to the ATT with last_lsn=0.
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
    }

    pub fn abort(&self, txn_id: u64) {
        self.att.lock().unwrap().remove(&txn_id);
    }

    /// Snapshot of the ATT for inclusion in a Checkpoint record.
    pub fn att_snapshot(&self) -> HashMap<u64, Lsn> {
        self.att.lock().unwrap().clone()
    }

    /// Used by recovery to advance the counter past txn_ids already on disk.
    pub fn set_next_txn_id(&self, id: u64) {
        self.next_txn_id.store(id, Ordering::SeqCst);
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn ids_unique_under_concurrency() {
        let tm = Arc::new(TransactionManager::new());
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
        let tm = TransactionManager::new();
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
