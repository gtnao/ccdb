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

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::buffer_pool::BufferPool;
use crate::page::Rid;
use crate::wal::{ClrRedo, Lsn, WalManager, WalRecord, WalRecordType};

#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub committed_txns: usize,
    pub uncommitted_txns: usize,
    pub redo_applied: usize,
    pub undo_applied: usize,
    pub max_lsn: Lsn,
    pub max_txn_id: u64,
}

pub fn recover(
    bpm: &BufferPool,
    wal: &WalManager,
    records: &[WalRecord],
    checkpoint_lsn: Option<Lsn>,
) -> Result<RecoveryStats> {
    if records.is_empty() {
        return Ok(RecoveryStats::default());
    }

    // Find the most recent Checkpoint record at or after `checkpoint_lsn`.
    // Its ATT/DPT seed analyze; redo can start from min(rec_lsn in DPT).
    let ckpt_idx = checkpoint_lsn.and_then(|target| {
        records
            .iter()
            .position(|r| r.lsn == target && matches!(r.record_type, WalRecordType::Checkpoint { .. }))
    });

    let analysis = analyze(records);

    // For redo: start at min(rec_lsn in DPT from the checkpoint), or from
    // the beginning if no checkpoint. Records before the start LSN are
    // guaranteed to already be on disk.
    let redo_start = ckpt_idx
        .and_then(|i| match &records[i].record_type {
            WalRecordType::Checkpoint { dpt, .. } => dpt.values().copied().min(),
            _ => None,
        })
        .unwrap_or(0);

    let redo_applied = redo_from(bpm, records, redo_start)?;
    let undo_applied = undo(bpm, wal, records, &analysis)?;

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
    /// Most recent LSN seen for each txn — entry point for the prev_lsn walk
    /// during undo.
    last_lsn_per_txn: HashMap<u64, Lsn>,
    max_lsn: Lsn,
    max_txn_id: u64,
}

fn analyze(records: &[WalRecord]) -> Analysis {
    let mut active: HashSet<u64> = HashSet::new();
    let mut committed: HashSet<u64> = HashSet::new();
    let mut last_lsn_per_txn: HashMap<u64, Lsn> = HashMap::new();
    let mut max_lsn: Lsn = 0;
    let mut max_txn_id: u64 = 0;
    let mut wrote_dml: HashSet<u64> = HashSet::new();

    for r in records {
        max_lsn = max_lsn.max(r.lsn);
        max_txn_id = max_txn_id.max(r.txn_id);
        last_lsn_per_txn.insert(r.txn_id, r.lsn);
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
            WalRecordType::Clr { .. } => {
                // CLR implies prior Insert/Delete by this txn.
                wrote_dml.insert(r.txn_id);
            }
            WalRecordType::Checkpoint { .. } => {
                // Pure metadata; no per-txn effect on classification.
            }
        }
    }

    let uncommitted: HashSet<u64> = active.intersection(&wrote_dml).copied().collect();
    Analysis {
        committed,
        uncommitted,
        last_lsn_per_txn,
        max_lsn,
        max_txn_id,
    }
}

fn redo_from(bpm: &BufferPool, records: &[WalRecord], start_lsn: Lsn) -> Result<usize> {
    let mut count = 0;
    for r in records {
        if r.lsn < start_lsn {
            continue;
        }
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
            WalRecordType::Clr { redo: cr, .. } => match cr {
                // CLR for an Insert: original was tombstoned during undo.
                ClrRedo::UndoInsert { rid } => {
                    if redo_delete(bpm, *rid, r.lsn)? {
                        count += 1;
                    }
                }
                // CLR for a Delete: original tombstone was restored during undo.
                ClrRedo::UndoDelete { rid, data } => {
                    if redo_insert(bpm, *rid, data, r.lsn)? {
                        count += 1;
                    }
                }
            },
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
    wal: &WalManager,
    records: &[WalRecord],
    analysis: &Analysis,
) -> Result<usize> {
    if analysis.uncommitted.is_empty() {
        return Ok(0);
    }
    let by_lsn: HashMap<Lsn, &WalRecord> = records.iter().map(|r| (r.lsn, r)).collect();

    let mut count = 0;
    for &txn_id in &analysis.uncommitted {
        // Walk the prev_lsn chain backwards from the txn's most recent record.
        // CLRs short-circuit via undo_next_lsn (already-undone region is skipped).
        // We maintain `last_clr_lsn` so newly-written CLRs chain together via prev_lsn.
        let start = match analysis.last_lsn_per_txn.get(&txn_id) {
            Some(&l) => l,
            None => continue,
        };
        let mut cur = start;
        let mut last_clr_lsn = start;

        while cur != 0 {
            let r = match by_lsn.get(&cur) {
                Some(r) => *r,
                None => break,
            };
            match &r.record_type {
                WalRecordType::Insert { rid, .. } => {
                    let (pid, slot) = *rid;
                    ensure_page_allocated(bpm, pid)?;
                    let g = bpm.fetch_page(pid)?;
                    let mut p = g.write();
                    if p.get_tuple(slot).is_some() {
                        p.delete(slot)?;
                    }
                    let clr_lsn = wal.append(
                        txn_id,
                        last_clr_lsn,
                        WalRecordType::Clr {
                            undo_next_lsn: r.prev_lsn,
                            redo: ClrRedo::UndoInsert { rid: *rid },
                        },
                    )?;
                    p.set_page_lsn(clr_lsn);
                    last_clr_lsn = clr_lsn;
                    cur = r.prev_lsn;
                    count += 1;
                }
                WalRecordType::Delete { rid, data } => {
                    let (pid, slot) = *rid;
                    ensure_page_allocated(bpm, pid)?;
                    let g = bpm.fetch_page(pid)?;
                    let mut p = g.write();
                    if p.get_tuple(slot).is_none() {
                        p.restore(slot, data)?;
                    }
                    let clr_lsn = wal.append(
                        txn_id,
                        last_clr_lsn,
                        WalRecordType::Clr {
                            undo_next_lsn: r.prev_lsn,
                            redo: ClrRedo::UndoDelete {
                                rid: *rid,
                                data: data.clone(),
                            },
                        },
                    )?;
                    p.set_page_lsn(clr_lsn);
                    last_clr_lsn = clr_lsn;
                    cur = r.prev_lsn;
                    count += 1;
                }
                WalRecordType::Clr { undo_next_lsn, .. } => {
                    // Already-undone region; jump past it.
                    cur = *undo_next_lsn;
                }
                WalRecordType::Begin => {
                    // Reached the transaction's start — write Abort and stop.
                    wal.append(txn_id, last_clr_lsn, WalRecordType::Abort)?;
                    break;
                }
                WalRecordType::Commit | WalRecordType::Abort => {
                    // Shouldn't happen for an "uncommitted" txn, but defensively
                    // just follow the chain.
                    cur = r.prev_lsn;
                }
                WalRecordType::Checkpoint { .. } => {
                    // Checkpoint records are not part of any txn's chain;
                    // shouldn't be reached by prev_lsn following, but be defensive.
                    cur = r.prev_lsn;
                }
            }
        }
    }
    wal.flush()?;
    Ok(count)
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
        {
            let (_pool, wal) = make_pool(&data, &wal_path);
            let l1 = wal.append(1, 0, WalRecordType::Begin).unwrap();
            let l2 = wal
                .append(
                    1,
                    l1,
                    WalRecordType::Insert {
                        rid: (0, 0),
                        data: vec![0xAA, 0xBB],
                    },
                )
                .unwrap();
            wal.append(1, l2, WalRecordType::Commit).unwrap();
            wal.flush().unwrap();
        }
        let recs = crate::wal::read_records(&wal_path).unwrap();
        let (pool, wal) = make_pool(&data, &wal_path);
        wal.set_next_lsn(recs.iter().map(|r| r.lsn).max().unwrap_or(0) + 1);
        let stats = recover(&pool, &wal, &recs, None).unwrap();
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
            let l1 = wal.append(2, 0, WalRecordType::Begin).unwrap();
            wal.append(
                2,
                l1,
                WalRecordType::Insert {
                    rid: (0, 0),
                    data: vec![0xCC],
                },
            )
            .unwrap();
            wal.flush().unwrap();
        }
        let recs = crate::wal::read_records(&wal_path).unwrap();
        let (pool, wal) = make_pool(&data, &wal_path);
        wal.set_next_lsn(recs.iter().map(|r| r.lsn).max().unwrap_or(0) + 1);
        let stats = recover(&pool, &wal, &recs, None).unwrap();
        assert_eq!(stats.committed_txns, 0);
        assert_eq!(stats.uncommitted_txns, 1);
        assert_eq!(stats.redo_applied, 1);
        assert_eq!(stats.undo_applied, 1);

        let g = pool.fetch_page(0).unwrap();
        let p = g.read();
        assert!(
            p.get_tuple(0).is_none(),
            "uncommitted insert should be tombstoned"
        );

        std::fs::remove_file(&data).ok();
        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn recovery_idempotent_with_clrs_from_partial_undo() {
        // Simulate: tx wrote Insert, recovery undid it (writing CLR), then
        // crashed before Abort. Re-running recovery sees the CLR and shouldn't
        // double-undo. End state: tombstoned, no spurious extra CLRs.
        let (data, wal_path) = paths("clr-resume");
        {
            let (_pool, wal) = make_pool(&data, &wal_path);
            let l1 = wal.append(3, 0, WalRecordType::Begin).unwrap();
            let l2 = wal
                .append(
                    3,
                    l1,
                    WalRecordType::Insert {
                        rid: (0, 0),
                        data: vec![0xEE],
                    },
                )
                .unwrap();
            // CLR for the Insert above (simulates first recovery's undo)
            wal.append(
                3,
                l2,
                WalRecordType::Clr {
                    undo_next_lsn: l1, // Insert.prev_lsn = Begin's lsn
                    redo: ClrRedo::UndoInsert { rid: (0, 0) },
                },
            )
            .unwrap();
            wal.flush().unwrap();
        }
        let recs = crate::wal::read_records(&wal_path).unwrap();
        let (pool, wal) = make_pool(&data, &wal_path);
        wal.set_next_lsn(recs.iter().map(|r| r.lsn).max().unwrap_or(0) + 1);
        let stats = recover(&pool, &wal, &recs, None).unwrap();
        // The Insert was already compensated; undo phase should write 0 new
        // undo records (the CLR's undo_next_lsn jumps past the Insert).
        assert_eq!(stats.uncommitted_txns, 1);
        assert_eq!(stats.undo_applied, 0);
        std::fs::remove_file(&data).ok();
        std::fs::remove_file(&wal_path).ok();
    }
}
