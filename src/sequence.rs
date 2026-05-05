//! Sequences (`CREATE SEQUENCE` / `nextval` / `setval`).
//!
//! Each sequence owns one buffer-pool page laid out as:
//! ```text
//!   header (24 bytes, page_kind = SequenceRel)
//!   [24..32) last_value : i64 LE
//!   [32..36) log_cnt    : i32 LE   — values reserved by the most recent
//!                                     SequenceAdvance WAL record but not
//!                                     yet handed out
//!   [36)     is_called  : u8 (0 or 1) — false until the first nextval
//! ```
//!
//! ## Concurrency / non-transactional semantics
//!
//! Every `nextval` takes the page's `RwLock::write()` so two callers can't
//! hand out the same value. Sequences are **non-transactional**: nextval
//! advances even when the calling transaction aborts, intentionally,
//! because making sequences transactional would force a single shared
//! lock across the whole tx and kill concurrent inserter throughput.
//! Gaps (skipped values) are part of the contract.
//!
//! ## WAL 32-batch optimisation
//!
//! Without batching, every nextval would write a WAL record + flush — too
//! expensive for hot sequences. PostgreSQL's trick: when log_cnt hits
//! zero, log a single `SequenceAdvance(new_last_value = current + 32 *
//! increment)` and set log_cnt = 32. The next 32 nextvals then run
//! without WAL until log_cnt drops back to zero. On crash, recovery
//! replays the last logged advance — at most 31 reservations are wasted,
//! but no duplicate value is ever returned.

use anyhow::{Result, bail};

use crate::buffer_pool::BufferPool;
use crate::page::{PageId, PageKind, Page};
use crate::transaction::Transaction;
use crate::wal::{Lsn, WalManager, WalRecordType};

/// Number of values reserved per WAL advance — matches PG's `SEQ_LOG_VALS`.
pub const SEQ_LOG_VALS: i32 = 32;

const OFF_LAST_VALUE: usize = 24;
const OFF_LOG_CNT: usize = 32;
const OFF_IS_CALLED: usize = 36;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceState {
    pub last_value: i64,
    pub log_cnt: i32,
    pub is_called: bool,
}

/// Initialise a freshly allocated page as a sequence relation. After this
/// call the page contains the canonical zero state — `last_value = start - increment`
/// is the caller's responsibility to write afterwards.
pub fn init_sequence_page(page: &mut Page) {
    let pid = page.page_id();
    *page = Page::new(pid);
    page.set_page_kind(PageKind::SequenceRel);
    write_state(
        page,
        SequenceState {
            last_value: 0,
            log_cnt: 0,
            is_called: false,
        },
    );
}

pub fn read_state(page: &Page) -> SequenceState {
    let bytes = page.as_bytes();
    let last_value = i64::from_le_bytes(bytes[OFF_LAST_VALUE..OFF_LAST_VALUE + 8].try_into().unwrap());
    let log_cnt = i32::from_le_bytes(bytes[OFF_LOG_CNT..OFF_LOG_CNT + 4].try_into().unwrap());
    let is_called = bytes[OFF_IS_CALLED] != 0;
    SequenceState {
        last_value,
        log_cnt,
        is_called,
    }
}

pub fn write_state(page: &mut Page, s: SequenceState) {
    let bytes = page.as_bytes_mut();
    bytes[OFF_LAST_VALUE..OFF_LAST_VALUE + 8].copy_from_slice(&s.last_value.to_le_bytes());
    bytes[OFF_LOG_CNT..OFF_LOG_CNT + 4].copy_from_slice(&s.log_cnt.to_le_bytes());
    bytes[OFF_IS_CALLED] = if s.is_called { 1 } else { 0 };
}

/// `nextval(seq)`. Returns the next value and updates the page state.
/// `start_value` and `increment` come from the catalog row; we trust the
/// caller passed them correctly.
pub fn nextval(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    seq_page_id: PageId,
    start_value: i64,
    increment: i64,
) -> Result<i64> {
    let g = bpm.fetch_page(seq_page_id)?;
    let mut page = g.write();
    if page.page_kind() != PageKind::SequenceRel {
        bail!("page {seq_page_id} is not a sequence relation");
    }
    let mut s = read_state(&page);

    // Compute the value to return.
    let next = if !s.is_called {
        start_value
    } else {
        s.last_value
            .checked_add(increment)
            .ok_or_else(|| anyhow::anyhow!("sequence overflow"))?
    };

    // If we exhausted the WAL-reserved batch, log a fresh advance record.
    if s.log_cnt <= 0 {
        let reserved_to = next
            .checked_add(increment.checked_mul(SEQ_LOG_VALS as i64).unwrap_or(i64::MAX))
            .unwrap_or(i64::MAX);
        let lsn = log_seq_advance(wal, tx, seq_page_id, reserved_to)?;
        page.set_page_lsn(lsn);
        s.log_cnt = SEQ_LOG_VALS;
    } else {
        s.log_cnt -= 1;
    }

    s.last_value = next;
    s.is_called = true;
    write_state(&mut page, s);
    Ok(next)
}

/// `setval(seq, n, [is_called])`. Always WAL-logged because the user-set
/// value would otherwise be visible only on the page until checkpoint.
pub fn setval(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    seq_page_id: PageId,
    new_value: i64,
    is_called: bool,
) -> Result<i64> {
    let g = bpm.fetch_page(seq_page_id)?;
    let mut page = g.write();
    if page.page_kind() != PageKind::SequenceRel {
        bail!("page {seq_page_id} is not a sequence relation");
    }
    let lsn = log_seq_advance(wal, tx, seq_page_id, new_value)?;
    write_state(
        &mut page,
        SequenceState {
            last_value: new_value,
            log_cnt: 0, // force next nextval to log
            is_called,
        },
    );
    page.set_page_lsn(lsn);
    Ok(new_value)
}

fn log_seq_advance(
    wal: &WalManager,
    tx: &mut Transaction,
    seq_page_id: PageId,
    new_last_value: i64,
) -> Result<Lsn> {
    let lsn = wal.append(
        tx.id(),
        tx.last_lsn(),
        WalRecordType::SequenceAdvance {
            seq_page_id,
            new_last_value,
        },
    )?;
    tx.set_last_lsn(lsn);
    Ok(lsn)
}
