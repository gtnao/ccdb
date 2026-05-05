//! MVCC tuple visibility under Snapshot Isolation.
//!
//! A tuple carries `xmin` (creator) and `xmax` (deleter, 0 if not deleted).
//! Whether the tuple is visible to a given snapshot follows Postgres-style
//! rules. The transaction manager's [`crate::transaction_manager::TxnStatus`]
//! lookup decides whether a writer's effect is durable.

use crate::transaction_manager::{Snapshot, TransactionManager, TxnStatus};
use crate::tuple::{INVALID_TXN_ID, TxnId};

/// Returns true when the tuple `(xmin, xmax)` is visible to `snapshot`.
pub fn is_visible(
    xmin: TxnId,
    xmax: TxnId,
    snapshot: &Snapshot,
    tm: &TransactionManager,
) -> bool {
    if !creator_visible(xmin, snapshot, tm) {
        return false;
    }

    // No deletion recorded.
    if xmax == INVALID_TXN_ID {
        return true;
    }

    // We deleted it ourselves — invisible to ourselves.
    if xmax == snapshot.txn_id {
        return false;
    }

    // The deleting txn started after our snapshot — its delete isn't visible.
    if xmax >= snapshot.xmax {
        return true;
    }

    // The deleting txn was concurrent with us — the deletion isn't visible.
    if snapshot.active.contains(&xmax) {
        return true;
    }

    // Otherwise the deletion is visible iff the deleting txn committed.
    match tm.status(xmax) {
        TxnStatus::Committed => false,
        TxnStatus::Aborted => true,
        // In-progress is excluded above by the active-set check; defensively
        // treat as not-deleted-yet.
        TxnStatus::InProgress => true,
    }
}

fn creator_visible(xmin: TxnId, snapshot: &Snapshot, tm: &TransactionManager) -> bool {
    // We created it ourselves.
    if xmin == snapshot.txn_id {
        return true;
    }
    // Created after our snapshot — invisible.
    if xmin >= snapshot.xmax {
        return false;
    }
    // Concurrent with us — invisible.
    if snapshot.active.contains(&xmin) {
        return false;
    }
    // Otherwise visible iff committed.
    match tm.status(xmin) {
        TxnStatus::Committed => true,
        TxnStatus::Aborted => false,
        TxnStatus::InProgress => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn snap(txn_id: TxnId, xmax: TxnId, active: &[TxnId]) -> Snapshot {
        Snapshot {
            txn_id,
            xmax,
            active: active.iter().copied().collect(),
        }
    }

    #[test]
    fn own_insert_is_visible_to_self() {
        let tm = TransactionManager::new();
        let s = snap(5, 10, &[]);
        assert!(is_visible(5, 0, &s, &tm));
    }

    #[test]
    fn concurrent_insert_invisible_until_commit() {
        let tm = Arc::new(TransactionManager::new());
        let writer = tm.begin();
        let reader = tm.begin();
        // Reader's snapshot includes writer in active set.
        let s = tm.snapshot(reader);
        assert!(s.active.contains(&writer));
        // Writer's tuple — xmin = writer
        assert!(!is_visible(writer, 0, &s, &tm));
        // Writer commits, but reader's snapshot was taken before — still hidden.
        tm.commit(writer);
        assert!(!is_visible(writer, 0, &s, &tm));
    }

    #[test]
    fn future_insert_invisible() {
        let tm = TransactionManager::new();
        let s = snap(5, 10, &[]);
        assert!(!is_visible(11, 0, &s, &tm));
    }

    #[test]
    fn committed_old_insert_visible() {
        let tm = TransactionManager::new();
        // xmin=2 committed before our snapshot started.
        tm.record_status(2, TxnStatus::Committed);
        let s = snap(5, 10, &[]);
        assert!(is_visible(2, 0, &s, &tm));
    }

    #[test]
    fn aborted_xmin_invisible() {
        let tm = TransactionManager::new();
        tm.record_status(2, TxnStatus::Aborted);
        let s = snap(5, 10, &[]);
        assert!(!is_visible(2, 0, &s, &tm));
    }

    #[test]
    fn deletion_by_committed_other_hides_tuple() {
        let tm = TransactionManager::new();
        tm.record_status(2, TxnStatus::Committed); // creator
        tm.record_status(3, TxnStatus::Committed); // deleter
        let s = snap(5, 10, &[]);
        assert!(!is_visible(2, 3, &s, &tm));
    }

    #[test]
    fn deletion_by_aborted_other_keeps_visible() {
        let tm = TransactionManager::new();
        tm.record_status(2, TxnStatus::Committed);
        tm.record_status(3, TxnStatus::Aborted);
        let s = snap(5, 10, &[]);
        assert!(is_visible(2, 3, &s, &tm));
    }

    #[test]
    fn own_delete_hides_tuple_from_self() {
        let tm = TransactionManager::new();
        tm.record_status(2, TxnStatus::Committed);
        let s = snap(5, 10, &[]);
        assert!(!is_visible(2, 5, &s, &tm));
    }
}
