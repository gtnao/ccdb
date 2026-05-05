//! Per-connection transaction state with a physical undo log and a
//! held-locks set for the LockManager.
//!
//! Each `Transaction` always has a `txn_id` (refreshed on every BEGIN and
//! every implicit auto-commit boundary). The undo log is populated during
//! DML and applied in reverse on ROLLBACK. The held_locks set tracks rows
//! locked through the LockManager so we know what to release at COMMIT /
//! ROLLBACK / auto-commit boundary.

use std::collections::HashSet;
use std::sync::Arc;

use crate::page::Rid;
use crate::transaction_manager::{Snapshot, TransactionManager};
use crate::wal::Lsn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    Inactive,
    Active,
}

#[derive(Debug, Clone)]
pub enum UndoLogEntry {
    /// Inserted a tuple at `rid`. The original Insert WAL record's LSN is
    /// kept so rollback can chain CLRs (undo_next_lsn = previous record LSN).
    Insert { rid: Rid, lsn: Lsn },
    /// Tombstoned a tuple. Carries restorable bytes and the original Delete
    /// record's LSN for the same chaining reason.
    Delete { rid: Rid, data: Vec<u8>, lsn: Lsn },
}

impl UndoLogEntry {
    pub fn lsn(&self) -> Lsn {
        match self {
            UndoLogEntry::Insert { lsn, .. } | UndoLogEntry::Delete { lsn, .. } => *lsn,
        }
    }
}

#[derive(Debug)]
pub struct Transaction {
    state: TxState,
    id: u64,
    log: Vec<UndoLogEntry>,
    held_locks: HashSet<Rid>,
    last_lsn: Lsn,
    tm: Arc<TransactionManager>,
    /// Snapshot taken at BEGIN (or at every auto-commit boundary). MVCC
    /// reads filter visibility against this.
    snapshot: Option<Snapshot>,
}

impl Transaction {
    pub fn new(tm: Arc<TransactionManager>) -> Self {
        let id = tm.begin();
        let snapshot = tm.snapshot(id);
        Self {
            state: TxState::Inactive,
            id,
            log: Vec::new(),
            held_locks: HashSet::new(),
            last_lsn: 0,
            tm,
            snapshot: Some(snapshot),
        }
    }

    pub fn snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    pub fn tm(&self) -> &Arc<TransactionManager> {
        &self.tm
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn last_lsn(&self) -> Lsn {
        self.last_lsn
    }

    pub fn set_last_lsn(&mut self, lsn: Lsn) {
        self.last_lsn = lsn;
        self.tm.update_last_lsn(self.id, lsn);
    }

    pub fn is_active(&self) -> bool {
        self.state == TxState::Active
    }

    pub fn begin(&mut self) {
        // Cleanly close the previous boundary in the ATT before opening a new id.
        self.tm.commit(self.id);
        self.id = self.tm.begin();
        self.state = TxState::Active;
        self.log.clear();
        self.held_locks.clear();
        self.last_lsn = 0;
        self.snapshot = Some(self.tm.snapshot(self.id));
    }

    pub fn commit(&mut self) {
        self.log.clear();
        self.state = TxState::Inactive;
        self.last_lsn = 0;
        self.tm.commit(self.id);
        self.snapshot = None;
    }

    /// Start a fresh auto-commit boundary. No-op while explicit BEGIN is in
    /// effect.
    pub fn refresh_autocommit(&mut self) {
        if !self.is_active() {
            self.tm.commit(self.id);
            self.id = self.tm.begin();
            self.log.clear();
            self.held_locks.clear();
            self.last_lsn = 0;
            self.snapshot = Some(self.tm.snapshot(self.id));
        }
    }

    pub fn record(&mut self, entry: UndoLogEntry) {
        debug_assert!(self.is_active(), "record() called outside an active tx");
        self.log.push(entry);
    }

    pub fn add_lock(&mut self, rid: Rid) {
        self.held_locks.insert(rid);
    }

    pub fn drain_log(&mut self) -> Vec<UndoLogEntry> {
        std::mem::take(&mut self.log)
    }

    pub fn set_inactive(&mut self) {
        self.state = TxState::Inactive;
        self.last_lsn = 0;
        self.tm.abort(self.id);
        self.snapshot = None;
    }

    pub fn take_held_locks(&mut self) -> HashSet<Rid> {
        std::mem::take(&mut self.held_locks)
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // Connection ended without a clean lifecycle — make sure we don't
        // leak the txn in the ATT.
        self.tm.abort(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle() {
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
        let id_before = tx.id();
        tx.begin();
        assert!(tx.is_active());
        assert_ne!(tx.id(), id_before, "begin refreshes the id");
        tx.record(UndoLogEntry::Insert {
            rid: (1, 2),
            lsn: 100,
        });
        tx.add_lock((0, 0));
        assert_eq!(tx.drain_log().len(), 1);
        assert_eq!(tx.take_held_locks().len(), 1);
        tx.set_inactive();
        assert!(!tx.is_active());
    }

    #[test]
    fn refresh_changes_id_only_when_inactive() {
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
        let id1 = tx.id();
        tx.refresh_autocommit();
        let id2 = tx.id();
        assert_ne!(id1, id2);

        tx.begin();
        let id3 = tx.id();
        tx.refresh_autocommit(); // no-op while active
        assert_eq!(tx.id(), id3);
    }
}
