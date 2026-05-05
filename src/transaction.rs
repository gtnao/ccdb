//! Per-connection transaction state with a physical undo log and a
//! held-locks set for the LockManager.
//!
//! Each `Transaction` always has a `txn_id` (refreshed on every BEGIN and
//! every implicit auto-commit boundary). The undo log is populated during
//! DML and applied in reverse on ROLLBACK. The held_locks set tracks rows
//! locked through the LockManager so we know what to release at COMMIT /
//! ROLLBACK / auto-commit boundary.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::page::Rid;
use crate::wal::Lsn;

static NEXT_TXN_ID: AtomicU64 = AtomicU64::new(1);

fn fresh_txn_id() -> u64 {
    NEXT_TXN_ID.fetch_add(1, Ordering::SeqCst)
}

/// Recovery uses this to advance past txn_ids that already appear in the WAL.
pub fn set_next_txn_id(id: u64) {
    NEXT_TXN_ID.store(id, Ordering::SeqCst);
}

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
}

impl Transaction {
    pub fn new() -> Self {
        Self {
            state: TxState::Inactive,
            id: fresh_txn_id(),
            log: Vec::new(),
            held_locks: HashSet::new(),
            last_lsn: 0,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn last_lsn(&self) -> Lsn {
        self.last_lsn
    }

    pub fn set_last_lsn(&mut self, lsn: Lsn) {
        self.last_lsn = lsn;
    }

    pub fn is_active(&self) -> bool {
        self.state == TxState::Active
    }

    pub fn begin(&mut self) {
        self.id = fresh_txn_id();
        self.state = TxState::Active;
        self.log.clear();
        self.held_locks.clear();
        self.last_lsn = 0;
    }

    pub fn commit(&mut self) {
        self.log.clear();
        self.state = TxState::Inactive;
        self.last_lsn = 0;
    }

    /// Start a fresh auto-commit boundary. No-op while explicit BEGIN is in
    /// effect.
    pub fn refresh_autocommit(&mut self) {
        if !self.is_active() {
            self.id = fresh_txn_id();
            self.log.clear();
            self.held_locks.clear();
            self.last_lsn = 0;
        }
    }

    pub fn record(&mut self, entry: UndoLogEntry) {
        debug_assert!(self.is_active(), "record() called outside an active tx");
        self.log.push(entry);
    }

    pub fn add_lock(&mut self, rid: Rid) {
        self.held_locks.insert(rid);
    }

    /// Drains the undo log without touching `last_lsn` (rollback still needs
    /// it to chain CLRs). State remains as-is — the caller decides when to
    /// flip back to Inactive.
    pub fn drain_log(&mut self) -> Vec<UndoLogEntry> {
        std::mem::take(&mut self.log)
    }

    pub fn set_inactive(&mut self) {
        self.state = TxState::Inactive;
        self.last_lsn = 0;
    }

    pub fn take_held_locks(&mut self) -> HashSet<Rid> {
        std::mem::take(&mut self.held_locks)
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle() {
        let mut tx = Transaction::new();
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
        let mut tx = Transaction::new();
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
