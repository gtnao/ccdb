//! Per-connection transaction state with a physical undo log.
//!
//! day09 supports a single, flat transaction (no savepoints). Each DML pushes
//! one or more entries; ROLLBACK applies them in reverse. COMMIT discards the
//! log without applying. This is sufficient for atomicity of a single
//! transaction but provides no isolation between concurrent connections —
//! that's day10's lock manager territory.

use crate::page::Rid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    Inactive,
    Active,
}

#[derive(Debug, Clone)]
pub enum UndoLogEntry {
    /// Inserted a tuple at `rid`. To undo, tombstone it.
    Insert { rid: Rid },
    /// Tombstoned the tuple at `rid` whose serialized bytes were `data`.
    /// To undo, restore those bytes into the slot.
    Delete { rid: Rid, data: Vec<u8> },
}

#[derive(Debug)]
pub struct Transaction {
    state: TxState,
    log: Vec<UndoLogEntry>,
}

impl Transaction {
    pub fn new() -> Self {
        Self {
            state: TxState::Inactive,
            log: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == TxState::Active
    }

    pub fn begin(&mut self) {
        self.state = TxState::Active;
        self.log.clear();
    }

    pub fn commit(&mut self) {
        self.log.clear();
        self.state = TxState::Inactive;
    }

    pub fn record(&mut self, entry: UndoLogEntry) {
        debug_assert!(self.is_active(), "record() called outside an active tx");
        self.log.push(entry);
    }

    /// Drains the undo log and returns to Inactive. Caller is responsible for
    /// applying the entries (in reverse order).
    pub fn take_log(&mut self) -> Vec<UndoLogEntry> {
        self.state = TxState::Inactive;
        std::mem::take(&mut self.log)
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
        assert!(!tx.is_active());
        tx.begin();
        assert!(tx.is_active());
        tx.record(UndoLogEntry::Insert { rid: (1, 2) });
        let log = tx.take_log();
        assert_eq!(log.len(), 1);
        assert!(!tx.is_active());
    }

    #[test]
    fn commit_clears_log() {
        let mut tx = Transaction::new();
        tx.begin();
        tx.record(UndoLogEntry::Insert { rid: (0, 0) });
        tx.commit();
        assert!(!tx.is_active());
        assert_eq!(tx.take_log().len(), 0);
    }
}
