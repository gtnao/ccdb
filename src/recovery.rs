//! Crash recovery driven by the WAL. Three phases:
//!
//! 1. **Analyze** — scan WAL to classify each transaction as committed or
//!    uncommitted (no Commit / Abort record before the end).
//! 2. **Redo** — reapply every Insert/Delete in LSN order, but only if
//!    `record.lsn > page.page_lsn` so already-flushed pages aren't touched.
//! 3. **Undo** — for uncommitted transactions, walk that transaction's
//!    records in reverse and apply the inverse operation.
//!
//! Compared to full ARIES this skips Compensation Log Records (CLR) and the
//! prev_lsn chain. Consequence: a crash during recovery itself can leave
//! the database in an undefined state. For "kill the server, restart it,
//! everything's fine" semantics this is sufficient.

use std::collections::HashSet;

use anyhow::Result;

use crate::buffer_pool::BufferPool;
use crate::page::Rid;
use crate::wal::{Lsn, WalRecord, WalRecordType};

#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub committed_txns: usize,
    pub uncommitted_txns: usize,
    pub redo_applied: usize,
    pub undo_applied: usize,
    pub max_lsn: Lsn,
    pub max_txn_id: u64,
}

pub fn recover(bpm: &BufferPool, records: &[WalRecord]) -> Result<RecoveryStats> {
    if records.is_empty() {
        return Ok(RecoveryStats::default());
    }

    let analysis = analyze(records);

    let redo_applied = redo(bpm, records)?;
    let undo_applied = undo(bpm, records, &analysis.uncommitted)?;

    bpm.flush_all()?;

    Ok(RecoveryStats {
        committed_txns: analysis.committed.len(),
        uncommitted_txns: analysis.uncommitted.len(),
        redo_applied,
        undo_applied,
        max_lsn: analysis.max_lsn,
        max_txn_id: analysis.max_txn_id,
    })
}

struct Analysis {
    committed: HashSet<u64>,
    uncommitted: HashSet<u64>,
    max_lsn: Lsn,
    max_txn_id: u64,
}

fn analyze(records: &[WalRecord]) -> Analysis {
    let mut active: HashSet<u64> = HashSet::new();
    let mut committed: HashSet<u64> = HashSet::new();
    let mut max_lsn: Lsn = 0;
    let mut max_txn_id: u64 = 0;
    // Track txns that wrote DML — Begin alone shouldn't count as "uncommitted
    // work to undo" since there's nothing to undo.
    let mut wrote_dml: HashSet<u64> = HashSet::new();

    for r in records {
        max_lsn = max_lsn.max(r.lsn);
        max_txn_id = max_txn_id.max(r.txn_id);
        match &r.record_type {
            WalRecordType::Begin => {
                active.insert(r.txn_id);
            }
            WalRecordType::Commit => {
                active.remove(&r.txn_id);
                committed.insert(r.txn_id);
            }
            WalRecordType::Abort => {
                active.remove(&r.txn_id);
            }
            WalRecordType::Insert { .. } | WalRecordType::Delete { .. } => {
                wrote_dml.insert(r.txn_id);
            }
        }
    }

    let uncommitted: HashSet<u64> = active.intersection(&wrote_dml).copied().collect();
    Analysis {
        committed,
        uncommitted,
        max_lsn,
        max_txn_id,
    }
}

fn redo(bpm: &BufferPool, records: &[WalRecord]) -> Result<usize> {
    let mut count = 0;
    for r in records {
        match &r.record_type {
            WalRecordType::Insert { rid, data } => {
                if redo_insert(bpm, *rid, data, r.lsn)? {
                    count += 1;
                }
            }
            WalRecordType::Delete { rid, .. } => {
                if redo_delete(bpm, *rid, r.lsn)? {
                    count += 1;
                }
            }
            _ => {}
        }
    }
    Ok(count)
}

fn ensure_page_allocated(bpm: &BufferPool, page_id: u32) -> Result<()> {
    while bpm.page_count() <= page_id {
        let _ = bpm.new_page()?;
    }
    Ok(())
}

fn redo_insert(bpm: &BufferPool, rid: Rid, data: &[u8], record_lsn: Lsn) -> Result<bool> {
    let (pid, slot) = rid;
    ensure_page_allocated(bpm, pid)?;
    let g = bpm.fetch_page(pid)?;
    let mut p = g.write();
    if p.page_lsn() >= record_lsn {
        return Ok(false);
    }
    let n = p.tuple_count();
    if slot < n {
        // Slot exists. If it's empty (tombstone or never-inserted-here), restore.
        if p.get_tuple(slot).is_none() {
            p.restore(slot, data)?;
        }
    } else if slot == n {
        // Append a new slot — slot id will match.
        p.insert(data)?;
    } else {
        // There's a gap. Insert empty slots up to `slot`, then write our data.
        // For simplicity we insert dummy tuples first, then tombstone them.
        // (Recovery normally sees slots in order, so this branch is rare.)
        while p.tuple_count() < slot {
            let id = p.insert(b"")?;
            p.delete(id)?;
        }
        p.insert(data)?;
    }
    p.set_page_lsn(record_lsn);
    Ok(true)
}

fn redo_delete(bpm: &BufferPool, rid: Rid, record_lsn: Lsn) -> Result<bool> {
    let (pid, slot) = rid;
    if bpm.page_count() <= pid {
        // Page doesn't exist: nothing to delete. The Insert that paired with
        // this would also redo first (records replayed in LSN order).
        return Ok(false);
    }
    let g = bpm.fetch_page(pid)?;
    let mut p = g.write();
    if p.page_lsn() >= record_lsn {
        return Ok(false);
    }
    if p.get_tuple(slot).is_some() {
        p.delete(slot)?;
    }
    p.set_page_lsn(record_lsn);
    Ok(true)
}

fn undo(
    bpm: &BufferPool,
    records: &[WalRecord],
    uncommitted: &HashSet<u64>,
) -> Result<usize> {
    if uncommitted.is_empty() {
        return Ok(0);
    }
    // Walk records in reverse: most-recent change first, like normal rollback.
    let mut count = 0;
    for r in records.iter().rev() {
        if !uncommitted.contains(&r.txn_id) {
            continue;
        }
        match &r.record_type {
            WalRecordType::Insert { rid, .. } => {
                undo_insert(bpm, *rid)?;
                count += 1;
            }
            WalRecordType::Delete { rid, data } => {
                undo_delete(bpm, *rid, data)?;
                count += 1;
            }
            _ => {}
        }
    }
    Ok(count)
}

fn undo_insert(bpm: &BufferPool, rid: Rid) -> Result<()> {
    let (pid, slot) = rid;
    if bpm.page_count() <= pid {
        return Ok(());
    }
    let g = bpm.fetch_page(pid)?;
    let mut p = g.write();
    if p.get_tuple(slot).is_some() {
        p.delete(slot)?;
    }
    Ok(())
}

fn undo_delete(bpm: &BufferPool, rid: Rid, data: &[u8]) -> Result<()> {
    let (pid, slot) = rid;
    ensure_page_allocated(bpm, pid)?;
    let g = bpm.fetch_page(pid)?;
    let mut p = g.write();
    if p.get_tuple(slot).is_none() {
        p.restore(slot, data)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::DiskManager;
    use crate::wal::{WalManager, WalRecordType};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn paths(label: &str) -> (PathBuf, PathBuf) {
        let stamp = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let mut data = std::env::temp_dir();
        data.push(format!("ccdb-recov-{label}-{stamp}.db"));
        let mut wal = std::env::temp_dir();
        wal.push(format!("ccdb-recov-{label}-{stamp}.log"));
        let _ = std::fs::remove_file(&data);
        let _ = std::fs::remove_file(&wal);
        (data, wal)
    }

    fn make_pool(data: &PathBuf, wal_path: &PathBuf) -> (BufferPool, Arc<WalManager>) {
        let disk = DiskManager::open(data).unwrap();
        let wal = Arc::new(WalManager::open(wal_path).unwrap());
        (BufferPool::new(disk, 4, Arc::clone(&wal)), wal)
    }

    #[test]
    fn redo_replays_committed_inserts() {
        let (data, wal_path) = paths("redo-commit");
        // Construct a synthetic WAL: Begin, Insert at (0,0), Commit.
        {
            let (_pool, wal) = make_pool(&data, &wal_path);
            wal.append(1, WalRecordType::Begin).unwrap();
            wal.append(
                1,
                WalRecordType::Insert {
                    rid: (0, 0),
                    data: vec![0xAA, 0xBB], // bitmap (1 byte) + payload
                },
            )
            .unwrap();
            wal.append(1, WalRecordType::Commit).unwrap();
            wal.flush().unwrap();
        }
        // No data file yet (we never flushed pages).
        let recs = crate::wal::read_records(&wal_path).unwrap();
        let (pool, _wal) = make_pool(&data, &wal_path);
        let stats = recover(&pool, &recs).unwrap();
        assert_eq!(stats.committed_txns, 1);
        assert_eq!(stats.redo_applied, 1);
        assert_eq!(stats.undo_applied, 0);

        let g = pool.fetch_page(0).unwrap();
        let p = g.read();
        assert_eq!(p.get_tuple(0).unwrap(), &[0xAA, 0xBB]);

        std::fs::remove_file(&data).ok();
        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn undo_rolls_back_uncommitted_inserts() {
        let (data, wal_path) = paths("undo-uncommit");
        {
            let (_pool, wal) = make_pool(&data, &wal_path);
            wal.append(2, WalRecordType::Begin).unwrap();
            wal.append(
                2,
                WalRecordType::Insert {
                    rid: (0, 0),
                    data: vec![0xCC],
                },
            )
            .unwrap();
            // No Commit / Abort — simulates crash mid-tx.
            wal.flush().unwrap();
        }
        let recs = crate::wal::read_records(&wal_path).unwrap();
        let (pool, _wal) = make_pool(&data, &wal_path);
        let stats = recover(&pool, &recs).unwrap();
        assert_eq!(stats.committed_txns, 0);
        assert_eq!(stats.uncommitted_txns, 1);
        assert_eq!(stats.redo_applied, 1);
        assert_eq!(stats.undo_applied, 1);

        let g = pool.fetch_page(0).unwrap();
        let p = g.read();
        assert!(p.get_tuple(0).is_none(), "uncommitted insert should be tombstoned");

        std::fs::remove_file(&data).ok();
        std::fs::remove_file(&wal_path).ok();
    }
}
