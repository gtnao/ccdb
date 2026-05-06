//! Volcano-style executor: each operator implements [`Executor::open`] and
//! [`Executor::next`], yielding tuples lazily up the tree.
//!
//! INSERT is intentionally *not* an Executor — it's a side-effecting action
//! and doesn't compose with row sources. The engine matches on statement
//! shape and dispatches separately.

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::analyzer::{
    AggArg, AggKind, AnalyzedAggregate, AnalyzedCreateIndexStatement,
    AnalyzedCreateSequenceStatement, AnalyzedDeleteStatement, AnalyzedDropIndexStatement,
    AnalyzedDropSequenceStatement, AnalyzedDropTableStatement, AnalyzedExpr, AnalyzedFrom,
    AnalyzedInsertStatement, AnalyzedLiteral, AnalyzedOrderBy, AnalyzedSelectStatement,
    AnalyzedStatement, AnalyzedTruncateStatement, AnalyzedUpdateStatement, AnalyzedVacuumStatement,
    LiteralValue, SequenceFnKind, TableSource,
};
use crate::ast::{BinaryOperator, JoinType, OrderDir, UnaryOperator};
use crate::buffer_pool::BufferPool;
use crate::catalog::Catalog;
use crate::lock_manager::{LockManager, LockMode};
use crate::page::{PageId, Rid, SlotId};
use crate::transaction::Transaction;
use crate::transaction_manager::{Snapshot, TransactionManager, TxnStatus};
use crate::tuple::{
    DataType, INVALID_TXN_ID, Schema, TxnId, Value, deserialize_tuple_mvcc, serialize_tuple_mvcc,
};
use crate::visibility;
use crate::wal::{Lsn, WalManager, WalRecordType};

/// Append a WAL record under `tx`'s id/last_lsn chain and update `last_lsn`.
fn log_record(wal: &WalManager, tx: &mut Transaction, rt: WalRecordType) -> Result<Lsn> {
    let lsn = wal.append(tx.id(), tx.last_lsn(), rt)?;
    tx.set_last_lsn(lsn);
    Ok(lsn)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tuple {
    pub values: Vec<Value>,
}

impl Tuple {
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }
}

pub trait Executor {
    fn open(&mut self) -> Result<()>;
    fn next(&mut self) -> Result<Option<Tuple>>;
}

#[derive(Debug)]
pub enum Output {
    /// SELECT: zero or more result rows.
    Rows(Vec<Tuple>),
    /// INSERT/UPDATE/DELETE: number of rows affected.
    Affected(usize),
    Begin,
    Commit,
    Rollback,
}

// -- SeqScan -----------------------------------------------------------------

/// MVCC-aware sequential scan over a single table's page chain.
pub struct SeqScan<'a> {
    bpm: &'a BufferPool,
    schema: Schema,
    first_page: PageId,
    cur_page: PageId,
    cur_slot: u16,
    snapshot: Snapshot,
    tm: &'a TransactionManager,
}

impl<'a> SeqScan<'a> {
    pub fn new(
        bpm: &'a BufferPool,
        catalog: &Catalog,
        table_id: usize,
        snapshot: Snapshot,
        tm: &'a TransactionManager,
    ) -> Result<Self> {
        let table = catalog
            .table_by_id(table_id)?
            .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?;
        Ok(Self {
            bpm,
            schema: table.to_schema(),
            first_page: table.first_page_id,
            cur_page: table.first_page_id,
            cur_slot: 0,
            snapshot,
            tm,
        })
    }
}

impl Executor for SeqScan<'_> {
    fn open(&mut self) -> Result<()> {
        self.cur_page = self.first_page;
        self.cur_slot = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        while self.cur_page != crate::page::NO_NEXT_PAGE
            && self.cur_page < self.bpm.page_count()
        {
            let (found, next): (Option<Vec<Value>>, PageId) = {
                let guard = self.bpm.fetch_page(self.cur_page)?;
                let page = guard.read();
                let next = page.next_page_id();
                let tc = page.tuple_count();
                let page_all_visible = page.all_visible();
                let mut out = None;
                while self.cur_slot < tc {
                    if let Some(raw) = page.get_tuple(self.cur_slot) {
                        let (xmin, xmax, values) = deserialize_tuple_mvcc(raw, &self.schema)?;
                        self.cur_slot += 1;
                        let vis = page_all_visible
                            || visibility::is_visible(xmin, xmax, &self.snapshot, self.tm);
                        if vis {
                            out = Some(values);
                            break;
                        }
                        continue;
                    }
                    self.cur_slot += 1;
                }
                (out, next)
            };
            if let Some(values) = found {
                return Ok(Some(Tuple::new(values)));
            }
            self.cur_page = next;
            self.cur_slot = 0;
        }
        Ok(None)
    }
}

// -- OneRowRelation ----------------------------------------------------------

/// `SELECT expr;` (no FROM) needs an executor that yields exactly one empty
/// tuple so the SELECT list is evaluated once. Postgres calls this the
/// "Result" node; we keep the descriptive name.
pub struct OneRowRelation {
    emitted: bool,
}

impl OneRowRelation {
    pub fn new() -> Self {
        Self { emitted: false }
    }
}

impl Executor for OneRowRelation {
    fn open(&mut self) -> Result<()> {
        self.emitted = false;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        Ok(Some(Tuple::new(Vec::new())))
    }
}

// -- IndexScan ---------------------------------------------------------------

/// Range over which `IndexScan` walks the leaf chain. `Eq` is just a special
/// case of a closed range over a single key, kept separate so the planner
/// can talk about it cleanly.
#[derive(Debug, Clone)]
pub enum IndexRange {
    /// Equality lookup: only entries whose key is exactly `key`.
    Eq { key: crate::btree::KeyBytes },
    /// Closed range `low ≤ key ≤ high` (sysbench's BETWEEN translates here).
    Between {
        low: crate::btree::KeyBytes,
        high: crate::btree::KeyBytes,
    },
}

/// MVCC-aware scan that walks an index leaf chain instead of the table's
/// heap pages. For each `(key, rid)` it pulls the heap tuple at `rid` and
/// runs a visibility check.
pub struct IndexScan<'a> {
    bpm: &'a BufferPool,
    schema: Schema,
    root: PageId,
    range: IndexRange,
    key_type: DataType,
    snapshot: Snapshot,
    tm: &'a TransactionManager,
    cursor: Option<crate::btree::LeafCursor>,
    initialized: bool,
}

impl<'a> IndexScan<'a> {
    pub fn new(
        bpm: &'a BufferPool,
        catalog: &Catalog,
        table_id: usize,
        root: PageId,
        range: IndexRange,
        key_type: DataType,
        snapshot: Snapshot,
        tm: &'a TransactionManager,
    ) -> Result<Self> {
        let table = catalog
            .table_by_id(table_id)?
            .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?;
        Ok(Self {
            bpm,
            schema: table.to_schema(),
            root,
            range,
            key_type,
            snapshot,
            tm,
            cursor: None,
            initialized: false,
        })
    }

    fn start_key(&self) -> &[u8] {
        match &self.range {
            IndexRange::Eq { key } => key,
            IndexRange::Between { low, .. } => low,
        }
    }

    /// True if `key` is past the upper bound of `range` (so we should stop).
    fn key_past_upper(&self, key: &[u8]) -> Result<bool> {
        match &self.range {
            IndexRange::Eq { key: target } => {
                Ok(crate::btree::compare_keys(key, target, self.key_type)?
                    != std::cmp::Ordering::Equal)
            }
            IndexRange::Between { high, .. } => Ok(crate::btree::compare_keys(
                key,
                high,
                self.key_type,
            )? == std::cmp::Ordering::Greater),
        }
    }
}

impl Executor for IndexScan<'_> {
    fn open(&mut self) -> Result<()> {
        if !self.initialized {
            self.cursor =
                crate::btree::first_ge(self.bpm, self.root, self.start_key(), self.key_type)?;
            self.initialized = true;
        }
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        loop {
            let Some(c) = self.cursor else {
                return Ok(None);
            };
            let Some((key, rid, next_cursor)) = crate::btree::read_at(self.bpm, c)? else {
                self.cursor = None;
                return Ok(None);
            };
            self.cursor = next_cursor;
            if self.key_past_upper(&key)? {
                self.cursor = None;
                return Ok(None);
            }
            // Heap visibility — with HOT chain follow. The leaf entry
            // may point at a row whose key has been "moved" to a
            // newer-still-same-key version via t_ctid. Follow the chain
            // until we find a visible tuple or the chain ends.
            match follow_chain_visible(self.bpm, &self.schema, &self.snapshot, self.tm, rid)? {
                Some((_rid, values)) => return Ok(Some(Tuple::new(values))),
                None => continue,
            }
        }
    }
}

// -- Filter ------------------------------------------------------------------

pub struct Filter<'a> {
    child: Box<dyn Executor + 'a>,
    predicate: AnalyzedExpr,
}

impl<'a> Filter<'a> {
    pub fn new(child: Box<dyn Executor + 'a>, predicate: AnalyzedExpr) -> Self {
        Self { child, predicate }
    }
}

impl Executor for Filter<'_> {
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        while let Some(t) = self.child.next()? {
            // SQL semantics: NULL predicate excludes the row.
            match evaluate_expr(&self.predicate, &t)? {
                Value::Bool(true) => return Ok(Some(t)),
                Value::Bool(false) | Value::Null => continue,
                other => bail!("WHERE predicate must be boolean, got {other:?}"),
            }
        }
        Ok(None)
    }
}

// -- Project -----------------------------------------------------------------

pub struct Project<'a> {
    child: Box<dyn Executor + 'a>,
    exprs: Vec<AnalyzedExpr>,
}

impl<'a> Project<'a> {
    pub fn new(child: Box<dyn Executor + 'a>, exprs: Vec<AnalyzedExpr>) -> Self {
        Self { child, exprs }
    }
}

impl Executor for Project<'_> {
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        match self.child.next()? {
            None => Ok(None),
            Some(t) => {
                let row: Vec<Value> = self
                    .exprs
                    .iter()
                    .map(|e| evaluate_expr(e, &t))
                    .collect::<Result<_>>()?;
                Ok(Some(Tuple::new(row)))
            }
        }
    }
}

// -- NestedLoopJoin ----------------------------------------------------------

/// Volcano-style nested-loop join. For each outer row, scan the inner from
/// the start; emit `outer ++ inner` when the ON predicate is true. LEFT
/// JOIN additionally emits `outer ++ NULL*` when no inner row matched.
///
/// `inner_width` is how many columns the inner side contributes — needed
/// to NULL-pad on a non-match for LEFT JOIN.
pub struct NestedLoopJoin<'a> {
    outer: Box<dyn Executor + 'a>,
    inner: Box<dyn Executor + 'a>,
    on: AnalyzedExpr,
    join_type: JoinType,
    inner_width: usize,
    cur_outer: Option<Tuple>,
    /// Whether the current outer row found at least one inner match. Used
    /// by LEFT JOIN to decide if a NULL-padded row needs to be emitted.
    matched_current_outer: bool,
    /// True between `next()` discovering inner is exhausted and the next
    /// `next()` actually advancing the outer. Lets us emit a single
    /// NULL-padded row for an unmatched outer before moving on.
    pending_left_null_pad: bool,
}

impl<'a> NestedLoopJoin<'a> {
    pub fn new(
        outer: Box<dyn Executor + 'a>,
        inner: Box<dyn Executor + 'a>,
        on: AnalyzedExpr,
        join_type: JoinType,
        inner_width: usize,
    ) -> Self {
        Self {
            outer,
            inner,
            on,
            join_type,
            inner_width,
            cur_outer: None,
            matched_current_outer: false,
            pending_left_null_pad: false,
        }
    }

    fn concat(outer: &Tuple, inner: &Tuple) -> Tuple {
        let mut v = Vec::with_capacity(outer.values.len() + inner.values.len());
        v.extend_from_slice(&outer.values);
        v.extend_from_slice(&inner.values);
        Tuple::new(v)
    }

    fn null_padded(outer: &Tuple, inner_width: usize) -> Tuple {
        let mut v = outer.values.clone();
        v.resize(v.len() + inner_width, Value::Null);
        Tuple::new(v)
    }
}

impl Executor for NestedLoopJoin<'_> {
    fn open(&mut self) -> Result<()> {
        self.outer.open()?;
        self.inner.open()?;
        self.cur_outer = self.outer.next()?;
        self.matched_current_outer = false;
        self.pending_left_null_pad = false;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        loop {
            // LEFT JOIN: outer with no matches gets a single NULL-padded row.
            if self.pending_left_null_pad {
                self.pending_left_null_pad = false;
                let outer = self
                    .cur_outer
                    .take()
                    .expect("pending_left_null_pad implies a current outer");
                self.cur_outer = self.outer.next()?;
                self.matched_current_outer = false;
                return Ok(Some(Self::null_padded(&outer, self.inner_width)));
            }

            let Some(outer_t) = &self.cur_outer else {
                return Ok(None);
            };

            match self.inner.next()? {
                Some(inner_t) => {
                    let joined = Self::concat(outer_t, &inner_t);
                    match evaluate_expr(&self.on, &joined)? {
                        Value::Bool(true) => {
                            self.matched_current_outer = true;
                            return Ok(Some(joined));
                        }
                        Value::Bool(false) | Value::Null => continue,
                        other => bail!("JOIN ON must be boolean, got {other:?}"),
                    }
                }
                None => {
                    // Inner exhausted for this outer row.
                    let need_pad = matches!(self.join_type, JoinType::Left)
                        && !self.matched_current_outer;
                    if need_pad {
                        self.pending_left_null_pad = true;
                        // Loop will emit the padded row on the next iteration.
                        continue;
                    }
                    // Advance outer; reset inner.
                    self.cur_outer = self.outer.next()?;
                    self.matched_current_outer = false;
                    if self.cur_outer.is_some() {
                        self.inner.open()?;
                    }
                }
            }
        }
    }
}

// -- HashAggregate -----------------------------------------------------------

/// Per-aggregate accumulator state. Numeric flavors split by input type so
/// SUM/AVG over DOUBLE doesn't lose precision through an i64 detour.
#[derive(Debug, Clone)]
enum AggState {
    Count(i64),
    /// (sum, saw_any) — INT input.
    SumInt(i64, bool),
    /// (sum, saw_any) — DOUBLE input.
    SumDouble(f64, bool),
    /// (sum, count) — INT input. AVG always finalizes to DOUBLE.
    AvgInt(i64, i64),
    /// (sum, count) — DOUBLE input.
    AvgDouble(f64, i64),
    Min(Option<Value>),
    Max(Option<Value>),
}

impl AggState {
    fn init(agg: &AnalyzedAggregate) -> Self {
        let arg_type = match &agg.arg {
            AggArg::Star => None,
            AggArg::Expr(e) => e.data_type(),
        };
        match agg.kind {
            AggKind::Count => AggState::Count(0),
            AggKind::Sum => match arg_type {
                Some(DataType::Double) => AggState::SumDouble(0.0, false),
                _ => AggState::SumInt(0, false),
            },
            AggKind::Avg => match arg_type {
                Some(DataType::Double) => AggState::AvgDouble(0.0, 0),
                _ => AggState::AvgInt(0, 0),
            },
            AggKind::Min => AggState::Min(None),
            AggKind::Max => AggState::Max(None),
        }
    }

    /// Update with one input row's evaluated argument value. `arg` is `None`
    /// for COUNT(*) — every row counts regardless of any column being NULL.
    fn update(&mut self, arg: Option<&Value>) -> Result<()> {
        match self {
            AggState::Count(n) => match arg {
                None => *n += 1, // COUNT(*)
                Some(Value::Null) => {}
                Some(_) => *n += 1,
            },
            AggState::SumInt(sum, any) => {
                if let Some(v) = arg {
                    match v {
                        Value::Null => {}
                        Value::Int(i) => {
                            *sum += *i as i64;
                            *any = true;
                        }
                        other => bail!("SUM(INT) not supported on {other:?}"),
                    }
                }
            }
            AggState::SumDouble(sum, any) => {
                if let Some(v) = arg {
                    match v {
                        Value::Null => {}
                        Value::Double(d) => {
                            *sum += *d;
                            *any = true;
                        }
                        Value::Int(i) => {
                            *sum += *i as f64;
                            *any = true;
                        }
                        other => bail!("SUM(DOUBLE) not supported on {other:?}"),
                    }
                }
            }
            AggState::AvgInt(sum, count) => {
                if let Some(v) = arg {
                    match v {
                        Value::Null => {}
                        Value::Int(i) => {
                            *sum += *i as i64;
                            *count += 1;
                        }
                        other => bail!("AVG(INT) not supported on {other:?}"),
                    }
                }
            }
            AggState::AvgDouble(sum, count) => {
                if let Some(v) = arg {
                    match v {
                        Value::Null => {}
                        Value::Double(d) => {
                            *sum += *d;
                            *count += 1;
                        }
                        Value::Int(i) => {
                            *sum += *i as f64;
                            *count += 1;
                        }
                        other => bail!("AVG(DOUBLE) not supported on {other:?}"),
                    }
                }
            }
            AggState::Min(cur) => {
                if let Some(v) = arg {
                    if !matches!(v, Value::Null)
                        && (cur.is_none() || compare_values(v, cur.as_ref().unwrap())?
                            == std::cmp::Ordering::Less)
                    {
                        *cur = Some(v.clone());
                    }
                }
            }
            AggState::Max(cur) => {
                if let Some(v) = arg {
                    if !matches!(v, Value::Null)
                        && (cur.is_none() || compare_values(v, cur.as_ref().unwrap())?
                            == std::cmp::Ordering::Greater)
                    {
                        *cur = Some(v.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn finalize(self) -> Value {
        match self {
            AggState::Count(n) => Value::Int(n as i32),
            AggState::SumInt(sum, any) => {
                if any { Value::Int(sum as i32) } else { Value::Null }
            }
            AggState::SumDouble(sum, any) => {
                if any { Value::Double(sum) } else { Value::Null }
            }
            // AVG always emits DOUBLE — the standard mathematical mean.
            AggState::AvgInt(sum, count) => {
                if count == 0 {
                    Value::Null
                } else {
                    Value::Double(sum as f64 / count as f64)
                }
            }
            AggState::AvgDouble(sum, count) => {
                if count == 0 {
                    Value::Null
                } else {
                    Value::Double(sum / count as f64)
                }
            }
            AggState::Min(v) | AggState::Max(v) => v.unwrap_or(Value::Null),
        }
    }
}

/// Apply a calendar interval to a TIMESTAMP (μs since PG epoch). Order:
/// (1) months — calendar advance, day clamped to month length
/// (2) days   — flat day count
/// (3) micros — flat duration
/// PG applies in this order; combining months and days separately is the
/// whole point of a 3-field interval.
fn apply_interval_to_timestamp(
    ts_micros: i64,
    months: i32,
    days: i32,
    micros: i64,
) -> Result<i64> {
    use chrono::{Datelike, Duration, NaiveDate};
    let pg_epoch = NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let mut dt = pg_epoch + Duration::microseconds(ts_micros);
    if months != 0 {
        // Compute target year/month and clamp the day to the month's length.
        let total_months = dt.year() as i64 * 12 + (dt.month() as i64 - 1) + months as i64;
        let new_year = total_months.div_euclid(12) as i32;
        let new_month = (total_months.rem_euclid(12) + 1) as u32;
        let max_day = days_in_month(new_year, new_month);
        let new_day = dt.day().min(max_day);
        let new_date = NaiveDate::from_ymd_opt(new_year, new_month, new_day)
            .ok_or_else(|| anyhow::anyhow!("date out of range"))?;
        dt = new_date.and_time(dt.time());
    }
    if days != 0 {
        dt += Duration::days(days as i64);
    }
    if micros != 0 {
        dt += Duration::microseconds(micros);
    }
    let delta = dt.signed_duration_since(pg_epoch);
    delta
        .num_microseconds()
        .ok_or_else(|| anyhow::anyhow!("timestamp out of range"))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    use chrono::{Datelike, NaiveDate};
    // Find the first of the *next* month, then back up one day.
    let (ny, nm) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let next = NaiveDate::from_ymd_opt(ny, nm, 1).expect("valid first-of-month");
    let last = next - chrono::Duration::days(1);
    last.day()
}

fn compare_values(a: &Value, b: &Value) -> Result<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(x.cmp(y)),
        (Value::Varchar(x), Value::Varchar(y)) => Ok(x.cmp(y)),
        (Value::Bool(x), Value::Bool(y)) => Ok(x.cmp(y)),
        (Value::Double(x), Value::Double(y)) => {
            // partial_cmp returns None for NaN; treat NaN as equal so
            // sort stays well-defined. (Real DBs collate NaN as greatest.)
            Ok(x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        }
        // Mixed numeric: promote INT → DOUBLE and recurse.
        (Value::Int(x), Value::Double(y)) => Ok((*x as f64)
            .partial_cmp(y)
            .unwrap_or(std::cmp::Ordering::Equal)),
        (Value::Double(x), Value::Int(y)) => Ok(x
            .partial_cmp(&(*y as f64))
            .unwrap_or(std::cmp::Ordering::Equal)),
        (Value::Timestamp(x), Value::Timestamp(y)) => Ok(x.cmp(y)),
        (Value::Date(x), Value::Date(y)) => Ok(x.cmp(y)),
        (Value::Time(x), Value::Time(y)) => Ok(x.cmp(y)),
        // Intervals don't have a total order in general (1 month vs 30 days
        // depends on context), but for sort stability we order by the rough
        // total micros = months*30d + days*1d + micros. This is approximate
        // — PostgreSQL does the same by convention.
        (
            Value::Interval {
                months: m1,
                days: d1,
                micros: u1,
            },
            Value::Interval {
                months: m2,
                days: d2,
                micros: u2,
            },
        ) => {
            const D_PER_MONTH: i64 = 30;
            const US_PER_DAY: i64 = 86_400_000_000;
            let total1 = (*m1 as i64) * D_PER_MONTH * US_PER_DAY
                + (*d1 as i64) * US_PER_DAY
                + u1;
            let total2 = (*m2 as i64) * D_PER_MONTH * US_PER_DAY
                + (*d2 as i64) * US_PER_DAY
                + u2;
            Ok(total1.cmp(&total2))
        }
        _ => bail!("cannot compare {a:?} and {b:?}"),
    }
}

/// Volcano-style hash aggregator. Blocking: drains the child on `open()`,
/// builds groups in a HashMap keyed by the group-key tuple, and emits one
/// row per group on `next()`. Output tuple shape is `[group_keys...,
/// agg_results...]`.
pub struct HashAggregate<'a> {
    child: Box<dyn Executor + 'a>,
    group_keys: Vec<AnalyzedExpr>,
    aggregates: Vec<AnalyzedAggregate>,
    /// After open(), contains finalized output rows in iteration order.
    rows: Vec<Tuple>,
    cursor: usize,
    initialized: bool,
}

impl<'a> HashAggregate<'a> {
    pub fn new(
        child: Box<dyn Executor + 'a>,
        group_keys: Vec<AnalyzedExpr>,
        aggregates: Vec<AnalyzedAggregate>,
    ) -> Self {
        Self {
            child,
            group_keys,
            aggregates,
            rows: Vec::new(),
            cursor: 0,
            initialized: false,
        }
    }

    fn build(&mut self) -> Result<()> {
        // Group order is preserved by remembering insertion order — using a
        // Vec as the table because group keys are Vec<Value> (no Hash impl
        // yet for Value, and small N is fine for now).
        let mut keys: Vec<Vec<Value>> = Vec::new();
        let mut states: Vec<Vec<AggState>> = Vec::new();

        self.child.open()?;
        let mut saw_any_input = false;
        while let Some(t) = self.child.next()? {
            saw_any_input = true;
            let key: Vec<Value> = self
                .group_keys
                .iter()
                .map(|gk| evaluate_expr(gk, &t))
                .collect::<Result<_>>()?;
            let group_idx = match keys.iter().position(|k| k == &key) {
                Some(i) => i,
                None => {
                    keys.push(key);
                    states.push(
                        self.aggregates.iter().map(AggState::init).collect(),
                    );
                    keys.len() - 1
                }
            };
            for (i, agg) in self.aggregates.iter().enumerate() {
                let arg_val = match &agg.arg {
                    AggArg::Star => None,
                    AggArg::Expr(e) => Some(evaluate_expr(e, &t)?),
                };
                states[group_idx][i].update(arg_val.as_ref())?;
            }
        }

        // SQL: a SELECT with aggregates and *no* GROUP BY produces exactly
        // one output row even on empty input. With GROUP BY, empty input
        // produces zero rows.
        if !saw_any_input && self.group_keys.is_empty() && !self.aggregates.is_empty() {
            keys.push(Vec::new());
            states.push(self.aggregates.iter().map(AggState::init).collect());
        }

        for (k, s) in keys.into_iter().zip(states.into_iter()) {
            let mut row: Vec<Value> = k;
            row.extend(s.into_iter().map(AggState::finalize));
            self.rows.push(Tuple::new(row));
        }
        Ok(())
    }
}

impl Executor for HashAggregate<'_> {
    fn open(&mut self) -> Result<()> {
        if !self.initialized {
            self.build()?;
            self.initialized = true;
        }
        self.cursor = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        if self.cursor >= self.rows.len() {
            return Ok(None);
        }
        let t = self.rows[self.cursor].clone();
        self.cursor += 1;
        Ok(Some(t))
    }
}

// -- Sort --------------------------------------------------------------------

/// Volcano-style Sort. Blocking: drains the child on `open()`, sorts the
/// buffered tuples by the given keys, then emits one per `next()`. NULLs
/// follow the PostgreSQL default — ASC puts them last, DESC puts them
/// first. Stable on equal keys, falling through to insertion order.
pub struct Sort<'a> {
    child: Box<dyn Executor + 'a>,
    keys: Vec<AnalyzedOrderBy>,
    rows: Vec<Tuple>,
    cursor: usize,
    initialized: bool,
}

impl<'a> Sort<'a> {
    pub fn new(child: Box<dyn Executor + 'a>, keys: Vec<AnalyzedOrderBy>) -> Self {
        Self {
            child,
            keys,
            rows: Vec::new(),
            cursor: 0,
            initialized: false,
        }
    }

    fn build(&mut self) -> Result<()> {
        self.child.open()?;
        // Pre-evaluate all sort keys for each tuple — this both avoids
        // re-evaluating on every comparison and lets us bail out cleanly
        // before the sort starts if a key fails to evaluate.
        let mut entries: Vec<(Vec<Value>, Tuple)> = Vec::new();
        while let Some(t) = self.child.next()? {
            let mut keyvals = Vec::with_capacity(self.keys.len());
            for k in &self.keys {
                keyvals.push(evaluate_expr(&k.expr, &t)?);
            }
            entries.push((keyvals, t));
        }
        let dirs: Vec<OrderDir> = self.keys.iter().map(|k| k.dir).collect();
        // sort_by is stable in std.
        entries.sort_by(|a, b| compare_keys(&a.0, &b.0, &dirs));
        self.rows = entries.into_iter().map(|(_, t)| t).collect();
        Ok(())
    }
}

impl Executor for Sort<'_> {
    fn open(&mut self) -> Result<()> {
        if !self.initialized {
            self.build()?;
            self.initialized = true;
        }
        self.cursor = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        if self.cursor >= self.rows.len() {
            return Ok(None);
        }
        let t = self.rows[self.cursor].clone();
        self.cursor += 1;
        Ok(Some(t))
    }
}

fn compare_keys(a: &[Value], b: &[Value], dirs: &[OrderDir]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (i, dir) in dirs.iter().enumerate() {
        let av = &a[i];
        let bv = &b[i];
        // NULL placement: ASC → last, DESC → first (PostgreSQL default).
        let ord = match (av, bv) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => match dir {
                OrderDir::Asc => Ordering::Greater,
                OrderDir::Desc => Ordering::Less,
            },
            (_, Value::Null) => match dir {
                OrderDir::Asc => Ordering::Less,
                OrderDir::Desc => Ordering::Greater,
            },
            _ => match compare_values(av, bv) {
                Ok(o) => match dir {
                    OrderDir::Asc => o,
                    OrderDir::Desc => o.reverse(),
                },
                // Type-mismatched key — treat as equal so we don't panic;
                // the analyzer should've caught this earlier.
                Err(_) => Ordering::Equal,
            },
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

// -- Limit -------------------------------------------------------------------

pub struct Limit<'a> {
    child: Box<dyn Executor + 'a>,
    remaining: u64,
}

impl<'a> Limit<'a> {
    pub fn new(child: Box<dyn Executor + 'a>, n: u64) -> Self {
        Self {
            child,
            remaining: n,
        }
    }
}

impl Executor for Limit<'_> {
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        match self.child.next()? {
            Some(t) => {
                self.remaining -= 1;
                Ok(Some(t))
            }
            None => Ok(None),
        }
    }
}

// -- Expression evaluation ---------------------------------------------------

fn evaluate_expr(expr: &AnalyzedExpr, tuple: &Tuple) -> Result<Value> {
    match expr {
        AnalyzedExpr::Literal(lit) => Ok(literal_to_value(lit)),
        AnalyzedExpr::ColumnRef(c) => Ok(tuple
            .values
            .get(c.column_index)
            .cloned()
            .unwrap_or(Value::Null)),
        AnalyzedExpr::BinaryOp {
            left, op, right, ..
        } => {
            let l = evaluate_expr(left, tuple)?;
            let r = evaluate_expr(right, tuple)?;
            evaluate_binary(*op, &l, &r)
        }
        AnalyzedExpr::UnaryOp { op, expr, .. } => {
            let v = evaluate_expr(expr, tuple)?;
            evaluate_unary(*op, &v)
        }
        // IS NULL is the one predicate that returns a *definite* bool when
        // its operand is NULL — that's the whole point.
        AnalyzedExpr::Now => bail!(
            "internal: AnalyzedExpr::Now reached evaluator (should be substituted at execute() entry)"
        ),
        AnalyzedExpr::SequenceCall { .. } => bail!(
            "internal: AnalyzedExpr::SequenceCall reached evaluator (must be pre-evaluated)"
        ),
        AnalyzedExpr::IsNull { expr, negated } => {
            let v = evaluate_expr(expr, tuple)?;
            let is_null = matches!(v, Value::Null);
            Ok(Value::Bool(if *negated { !is_null } else { is_null }))
        }
    }
}

/// Walk every AnalyzedExpr in a statement and replace `Now` with a
/// Literal::Timestamp(ts). The substitution is structural — we clone the
/// statement so the original analyzer-cached version stays Now-bearing and
/// can be re-executed in a different transaction with a different ts.
fn substitute_now_in_statement(stmt: &AnalyzedStatement, ts: i64) -> AnalyzedStatement {
    match stmt {
        AnalyzedStatement::Select(s) => {
            let mut s = s.clone();
            substitute_now_in_select(&mut s, ts);
            AnalyzedStatement::Select(s)
        }
        AnalyzedStatement::Insert(s) => {
            let mut s = s.clone();
            for row in &mut s.rows {
                for e in row.iter_mut() {
                    substitute_now_in_expr(e, ts);
                }
            }
            AnalyzedStatement::Insert(s)
        }
        AnalyzedStatement::Update(s) => {
            let mut s = s.clone();
            for a in &mut s.assignments {
                substitute_now_in_expr(&mut a.value, ts);
            }
            if let Some(w) = &mut s.where_clause {
                substitute_now_in_expr(w, ts);
            }
            AnalyzedStatement::Update(s)
        }
        AnalyzedStatement::Delete(s) => {
            let mut s = s.clone();
            if let Some(w) = &mut s.where_clause {
                substitute_now_in_expr(w, ts);
            }
            AnalyzedStatement::Delete(s)
        }
        // CreateTable / CreateIndex / Begin / Commit / Rollback / Checkpoint
        // don't carry user-supplied expressions that could contain now().
        other => other.clone(),
    }
}

fn substitute_now_in_select(s: &mut AnalyzedSelectStatement, ts: i64) {
    if let Some(w) = &mut s.where_clause {
        substitute_now_in_expr(w, ts);
    }
    for it in &mut s.select_items {
        substitute_now_in_expr(&mut it.expr, ts);
    }
    if let Some(h) = &mut s.having {
        substitute_now_in_expr(h, ts);
    }
    for ob in &mut s.order_by {
        substitute_now_in_expr(&mut ob.expr, ts);
    }
    if let Some(agg) = &mut s.aggregation {
        for k in &mut agg.group_keys {
            substitute_now_in_expr(k, ts);
        }
        for a in &mut agg.aggregates {
            if let crate::analyzer::AggArg::Expr(e) = &mut a.arg {
                substitute_now_in_expr(e, ts);
            }
        }
    }
    substitute_now_in_from(&mut s.from, ts);
}

fn substitute_now_in_from(f: &mut AnalyzedFrom, ts: i64) {
    if let AnalyzedFrom::Join { left, on, .. } = f {
        substitute_now_in_from(left, ts);
        substitute_now_in_expr(on, ts);
    }
}

fn substitute_now_in_expr(e: &mut AnalyzedExpr, ts: i64) {
    match e {
        AnalyzedExpr::Now => {
            *e = AnalyzedExpr::Literal(AnalyzedLiteral {
                value: LiteralValue::Timestamp(ts),
                data_type: Some(DataType::Timestamp),
            });
        }
        AnalyzedExpr::Literal(_)
        | AnalyzedExpr::ColumnRef(_)
        | AnalyzedExpr::SequenceCall { .. } => {}
        AnalyzedExpr::BinaryOp { left, right, .. } => {
            substitute_now_in_expr(left, ts);
            substitute_now_in_expr(right, ts);
        }
        AnalyzedExpr::UnaryOp { expr, .. } => substitute_now_in_expr(expr, ts),
        AnalyzedExpr::IsNull { expr, .. } => substitute_now_in_expr(expr, ts),
    }
}

/// Walk an analyzed expression and execute every SequenceCall, replacing
/// it with the resulting Literal::Int. Each call advances/reads the live
/// sequence relation, so this MUST be called per-row evaluation context
/// (once per output row in INSERT / SELECT-no-FROM).
fn evaluate_sequence_calls_in_expr(
    e: &mut AnalyzedExpr,
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
) -> Result<()> {
    match e {
        AnalyzedExpr::SequenceCall {
            kind,
            seq_page_id,
            increment,
            start_value,
            setval_arg,
            setval_is_called,
            ..
        } => {
            let v = match kind {
                SequenceFnKind::Nextval => crate::sequence::nextval(
                    bpm,
                    wal,
                    tx,
                    *seq_page_id,
                    *start_value,
                    *increment,
                )?,
                SequenceFnKind::Setval => {
                    let n = setval_arg.ok_or_else(|| anyhow::anyhow!("setval missing arg"))?;
                    crate::sequence::setval(bpm, wal, tx, *seq_page_id, n, *setval_is_called)?
                }
            };
            *e = AnalyzedExpr::Literal(AnalyzedLiteral {
                value: LiteralValue::Integer(v),
                data_type: Some(DataType::Int),
            });
            Ok(())
        }
        AnalyzedExpr::BinaryOp { left, right, .. } => {
            evaluate_sequence_calls_in_expr(left, bpm, wal, tx)?;
            evaluate_sequence_calls_in_expr(right, bpm, wal, tx)
        }
        AnalyzedExpr::UnaryOp { expr, .. } => evaluate_sequence_calls_in_expr(expr, bpm, wal, tx),
        AnalyzedExpr::IsNull { expr, .. } => evaluate_sequence_calls_in_expr(expr, bpm, wal, tx),
        _ => Ok(()),
    }
}

fn literal_to_value(lit: &AnalyzedLiteral) -> Value {
    match &lit.value {
        // Note: Literal::Integer is i64 in the AST but our runtime int is i32.
        // Truncation here is the same compromise as the storage tuple layout.
        LiteralValue::Integer(n) => Value::Int(*n as i32),
        LiteralValue::Float(f) => Value::Double(*f),
        LiteralValue::String(s) => Value::Varchar(s.clone()),
        LiteralValue::Boolean(b) => Value::Bool(*b),
        LiteralValue::Timestamp(t) => Value::Timestamp(*t),
        LiteralValue::Date(d) => Value::Date(*d),
        LiteralValue::Time(t) => Value::Time(*t),
        LiteralValue::Interval {
            months,
            days,
            micros,
        } => Value::Interval {
            months: *months,
            days: *days,
            micros: *micros,
        },
        LiteralValue::Null => Value::Null,
    }
}

fn evaluate_binary(op: BinaryOperator, l: &Value, r: &Value) -> Result<Value> {
    // SQL: any NULL operand → NULL result. (3VL refinements for AND/OR are
    // a deliberate non-goal at day06; result is conservative.)
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }

    use BinaryOperator::*;
    // Numeric promotion: any DOUBLE operand pulls the other into DOUBLE.
    let left_promote = match (l, r) {
        (Value::Int(a), Value::Double(_)) => Some(Value::Double(*a as f64)),
        _ => None,
    };
    let right_promote = match (l, r) {
        (Value::Double(_), Value::Int(b)) => Some(Value::Double(*b as f64)),
        _ => None,
    };
    let l = left_promote.as_ref().unwrap_or(l);
    let r = right_promote.as_ref().unwrap_or(r);

    match (l, r) {
        (Value::Double(a), Value::Double(b)) => Ok(match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Add => Value::Double(a + b),
            Sub => Value::Double(a - b),
            Mul => Value::Double(a * b),
            Div => {
                if *b == 0.0 {
                    bail!("division by zero");
                }
                Value::Double(a / b)
            }
            And | Or => bail!("AND/OR not supported on DOUBLE"),
        }),
        (Value::Int(a), Value::Int(b)) => Ok(match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Add => Value::Int(a + b),
            Sub => Value::Int(a - b),
            Mul => Value::Int(a * b),
            Div => {
                if *b == 0 {
                    bail!("division by zero");
                }
                Value::Int(a / b)
            }
            And | Or => bail!("AND/OR not supported on INT"),
        }),
        (Value::Bool(a), Value::Bool(b)) => Ok(match op {
            And => Value::Bool(*a && *b),
            Or => Value::Bool(*a || *b),
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            _ => bail!("unsupported op {op:?} on BOOL"),
        }),
        (Value::Varchar(a), Value::Varchar(b)) => Ok(match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            _ => bail!("unsupported op {op:?} on VARCHAR"),
        }),
        (Value::Timestamp(a), Value::Timestamp(b)) => Ok(match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Sub => Value::Interval {
                months: 0,
                days: 0,
                micros: a - b,
            },
            _ => bail!("unsupported op {op:?} on TIMESTAMP"),
        }),
        (Value::Date(a), Value::Date(b)) => Ok(match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Sub => Value::Int(a - b),
            _ => bail!("unsupported op {op:?} on DATE"),
        }),
        (Value::Time(a), Value::Time(b)) => Ok(match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            _ => bail!("unsupported op {op:?} on TIME"),
        }),
        // -- Date arithmetic --------------------------------------------------
        (Value::Date(d), Value::Int(n)) if matches!(op, Add | Sub) => {
            let delta = if matches!(op, Add) { *n } else { -*n };
            Ok(Value::Date(d.saturating_add(delta)))
        }
        (Value::Int(n), Value::Date(d)) if matches!(op, Add) => {
            Ok(Value::Date(d.saturating_add(*n)))
        }
        (Value::Date(d), Value::Interval { months, days, micros })
            if matches!(op, Add | Sub) =>
        {
            // PG: date + interval → timestamp (because interval may carry
            // a sub-day component). Convert date to timestamp at midnight,
            // then apply.
            let ts = (*d as i64) * 86_400_000_000;
            let sign = if matches!(op, Add) { 1 } else { -1 };
            apply_interval_to_timestamp(ts, sign * months, sign * days, sign as i64 * micros)
                .map(Value::Timestamp)
        }
        (Value::Interval { months, days, micros }, Value::Date(d))
            if matches!(op, Add) =>
        {
            let ts = (*d as i64) * 86_400_000_000;
            apply_interval_to_timestamp(ts, *months, *days, *micros).map(Value::Timestamp)
        }
        // -- Timestamp arithmetic ---------------------------------------------
        (Value::Timestamp(ts), Value::Interval { months, days, micros })
            if matches!(op, Add | Sub) =>
        {
            let sign = if matches!(op, Add) { 1 } else { -1 };
            apply_interval_to_timestamp(*ts, sign * months, sign * days, sign as i64 * micros)
                .map(Value::Timestamp)
        }
        (Value::Interval { months, days, micros }, Value::Timestamp(ts))
            if matches!(op, Add) =>
        {
            apply_interval_to_timestamp(*ts, *months, *days, *micros).map(Value::Timestamp)
        }
        // -- Interval arithmetic ----------------------------------------------
        (
            Value::Interval { months: m1, days: d1, micros: u1 },
            Value::Interval { months: m2, days: d2, micros: u2 },
        ) if matches!(op, Add | Sub) => {
            let sign: i64 = if matches!(op, Add) { 1 } else { -1 };
            Ok(Value::Interval {
                months: m1 + sign as i32 * m2,
                days: d1 + sign as i32 * d2,
                micros: u1 + sign * u2,
            })
        }
        (Value::Interval { months, days, micros }, Value::Int(n)) if matches!(op, Mul) => {
            Ok(Value::Interval {
                months: months * n,
                days: days * n,
                micros: micros * (*n as i64),
            })
        }
        (Value::Int(n), Value::Interval { months, days, micros }) if matches!(op, Mul) => {
            Ok(Value::Interval {
                months: months * n,
                days: days * n,
                micros: micros * (*n as i64),
            })
        }
        (Value::Interval { months, days, micros }, Value::Int(n)) if matches!(op, Div) => {
            if *n == 0 {
                bail!("division by zero");
            }
            Ok(Value::Interval {
                months: months / n,
                days: days / n,
                micros: micros / (*n as i64),
            })
        }
        // -- Time + Interval --------------------------------------------------
        (Value::Time(t), Value::Interval { micros, .. }) if matches!(op, Add | Sub) => {
            // PG: TIME + INTERVAL ignores the months/days components.
            let sign: i64 = if matches!(op, Add) { 1 } else { -1 };
            const DAY_US: i64 = 86_400_000_000;
            let new_t = (*t + sign * *micros).rem_euclid(DAY_US);
            Ok(Value::Time(new_t))
        }
        (Value::Interval { micros, .. }, Value::Time(t)) if matches!(op, Add) => {
            const DAY_US: i64 = 86_400_000_000;
            Ok(Value::Time((*t + *micros).rem_euclid(DAY_US)))
        }
        _ => bail!("type mismatch in binary op {op:?}"),
    }
}

fn evaluate_unary(op: UnaryOperator, v: &Value) -> Result<Value> {
    if matches!(v, Value::Null) {
        return Ok(Value::Null);
    }
    match (op, v) {
        (UnaryOperator::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (UnaryOperator::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
        _ => bail!("unsupported unary op {op:?} on {v:?}"),
    }
}

// -- INSERT (not an Executor) ------------------------------------------------

fn perform_create_table(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &crate::analyzer::AnalyzedCreateTableStatement,
    tx: &mut Transaction,
) -> Result<()> {
    use crate::bootstrap::{
        DT_BOOL, DT_DATE, DT_DOUBLE, DT_INT, DT_INTERVAL, DT_TIME, DT_TIMESTAMP, DT_VARCHAR,
        PG_ATTRIBUTE_PAGE_ID, PG_CLASS_PAGE_ID,
    };

    // Pick a fresh table_id (max existing + 1). Catalog scan is enough at
    // this scale; no concurrent CREATEs assumed.
    let mut max_id: i32 = -1;
    for table in catalog.user_tables()? {
        max_id = max_id.max(table.table_id as i32);
    }
    // System tables occupy 0..=4 (pg_class, pg_attribute, pg_index,
    // pg_sequence, pg_constraint). User tables start at 5.
    let new_table_id = (max_id + 1).max(5);

    // Allocate the table's first heap page.
    let new_page_id = {
        let g = bpm.new_page()?;
        g.page_id()
    };

    // Insert into pg_class.
    let pg_class_row = serialize_tuple_mvcc(
        tx.id(),
        INVALID_TXN_ID,
        &[
            Value::Int(new_table_id),
            Value::Varchar(stmt.table_name.clone()),
            Value::Int(new_page_id as i32),
        ],
    );
    insert_bytes(bpm, wal, tx, PG_CLASS_PAGE_ID, &pg_class_row)?;

    // Insert one row per column into pg_attribute.
    for (ord, col) in stmt.columns.iter().enumerate() {
        let dt = match col.data_type {
            crate::tuple::DataType::Int => DT_INT,
            crate::tuple::DataType::Varchar => DT_VARCHAR,
            crate::tuple::DataType::Bool => DT_BOOL,
            crate::tuple::DataType::Double => DT_DOUBLE,
            crate::tuple::DataType::Timestamp => DT_TIMESTAMP,
            crate::tuple::DataType::Date => DT_DATE,
            crate::tuple::DataType::Time => DT_TIME,
            crate::tuple::DataType::Interval => DT_INTERVAL,
        };
        let default_text = col.default_text.clone().unwrap_or_default();
        let row = serialize_tuple_mvcc(
            tx.id(),
            INVALID_TXN_ID,
            &[
                Value::Int(new_table_id),
                Value::Varchar(col.name.clone()),
                Value::Int(dt),
                Value::Bool(col.nullable),
                Value::Int(ord as i32),
                Value::Varchar(default_text),
            ],
        );
        insert_bytes(bpm, wal, tx, PG_ATTRIBUTE_PAGE_ID, &row)?;
    }

    // CHECK constraints → pg_constraint rows, one per predicate.
    if !stmt.check_constraints.is_empty() || !stmt.foreign_keys.is_empty() {
        use crate::bootstrap::PG_CONSTRAINT_PAGE_ID;
        let existing = catalog.all_constraints()?;
        let mut next_id: i32 = existing
            .iter()
            .map(|c| c.constraint_id as i32)
            .max()
            .unwrap_or(-1)
            + 1;
        for (i, defn) in stmt.check_constraints.iter().enumerate() {
            let cname = format!("{}_check_{}", stmt.table_name, i);
            let row = serialize_tuple_mvcc(
                tx.id(),
                INVALID_TXN_ID,
                &[
                    Value::Int(next_id),
                    Value::Varchar(cname),
                    Value::Int(new_table_id),
                    Value::Int(crate::catalog::ConstraintKind::Check as i32),
                    Value::Varchar(defn.clone()),
                ],
            );
            insert_bytes(bpm, wal, tx, PG_CONSTRAINT_PAGE_ID, &row)?;
            next_id += 1;
        }
        for (i, defn) in stmt.foreign_keys.iter().enumerate() {
            let cname = format!("{}_fkey_{}", stmt.table_name, i);
            let row = serialize_tuple_mvcc(
                tx.id(),
                INVALID_TXN_ID,
                &[
                    Value::Int(next_id),
                    Value::Varchar(cname),
                    Value::Int(new_table_id),
                    Value::Int(crate::catalog::ConstraintKind::ForeignKey as i32),
                    Value::Varchar(defn.clone()),
                ],
            );
            insert_bytes(bpm, wal, tx, PG_CONSTRAINT_PAGE_ID, &row)?;
            next_id += 1;
        }
    }

    // PRIMARY KEY → auto-create a unique index named `<table>_pkey`. The
    // table is empty at this point so the per-row scan in perform_create_index
    // is a no-op; we still go through it so pg_index/the btree root are set
    // up identically to an explicit CREATE UNIQUE INDEX.
    if let Some(col_idx) = stmt.primary_key_column {
        let col = &stmt.columns[col_idx];
        let pk_stmt = AnalyzedCreateIndexStatement {
            name: format!("{}_pkey", stmt.table_name),
            table_id: new_table_id as usize,
            table_name: stmt.table_name.clone(),
            column_index: col_idx,
            column_name: col.name.clone(),
            data_type: col.data_type,
            is_unique: true,
        };
        perform_create_index(bpm, wal, catalog, &pk_stmt, tx)?;
    }
    catalog.invalidate();
    Ok(())
}

/// Build a B+Tree index over an existing table.
///
/// Steps:
///   1. Pick a fresh `index_id` (max existing + 1).
///   2. Allocate an empty leaf as the initial root.
///   3. Scan the table's heap, inserting every committed-and-visible
///      `(key, rid)` pair into the tree. We use the system snapshot here —
///      uncommitted concurrent writes get re-indexed on their commit (the
///      DML path inserts into every index for the table).
///   4. Insert the corresponding row into `pg_index`.
///
/// Logged via per-page LSNs as part of the regular WAL machinery.
fn perform_create_index(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedCreateIndexStatement,
    tx: &mut Transaction,
) -> Result<()> {
    use crate::bootstrap::PG_INDEX_PAGE_ID;

    // Pick a fresh index_id.
    let mut max_id: i32 = -1;
    for idx in catalog.all_indexes()? {
        max_id = max_id.max(idx.index_id as i32);
    }
    let new_index_id = max_id + 1;

    // Bulk-build path: heap scan → sort by key → pack leaves → fan up
    // internal nodes. Avoids the O(n log n) per-row btree::insert with
    // its descend + split overhead, which dominates `pgbench -i` time.
    let snapshot = tx
        .snapshot()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no snapshot for tx"))?;
    let (_schema, rows) = visible_rows(bpm, catalog, stmt.table_id, &snapshot, tx.tm())?;
    let mut entries: Vec<(crate::btree::KeyBytes, Rid)> = Vec::with_capacity(rows.len());
    for (pid, slot, values) in rows {
        let key_value = &values[stmt.column_index];
        if matches!(key_value, Value::Null) {
            continue; // NULLs are not indexed (matches CREATE INDEX convention)
        }
        let key = crate::btree::encode_key(key_value);
        entries.push((key, (pid, slot)));
    }
    // Sort by key (ties broken by rid for determinism).
    entries.sort_by(|a, b| {
        match crate::btree::compare_keys(&a.0, &b.0, stmt.data_type) {
            Ok(o) if o != std::cmp::Ordering::Equal => o,
            _ => a.1.cmp(&b.1),
        }
    });

    // WAL: still log per-(key, rid) IndexInsert so recovery can rebuild
    // the tree from records if we crash before pg_index is durable.
    for (key, rid) in &entries {
        log_record(
            wal,
            tx,
            WalRecordType::IndexInsert {
                index_id: new_index_id as u64,
                key: key.clone(),
                rid: *rid,
            },
        )?;
    }

    let current_root =
        bulk_build_btree(bpm, wal, tx, &entries, stmt.data_type)?;

    // Register in pg_index.
    let row = serialize_tuple_mvcc(
        tx.id(),
        INVALID_TXN_ID,
        &[
            Value::Int(new_index_id),
            Value::Varchar(stmt.name.clone()),
            Value::Int(stmt.table_id as i32),
            Value::Int(stmt.column_index as i32),
            Value::Int(current_root as i32),
            Value::Bool(stmt.is_unique),
        ],
    );
    insert_bytes(bpm, wal, tx, PG_INDEX_PAGE_ID, &row)?;
    catalog.invalidate();
    Ok(())
}

/// Sort-then-pack B+Tree builder. Returns the new root's PageId.
/// `entries` must already be sorted by key. Logs PageInit for every
/// allocated tree page so recovery can re-tag them as
/// BTreeLeaf / BTreeInternal even if no flush reached disk.
fn bulk_build_btree(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    entries: &[(crate::btree::KeyBytes, Rid)],
    _dt: DataType,
) -> Result<PageId> {
    use crate::page::PageKind;

    if entries.is_empty() {
        return crate::btree::new_empty_root(bpm, wal, tx);
    }

    // Pack into leaves left-to-right, threading next_page_id along.
    // Each entry is `[key_len:u16][key][rid_page:u32][rid_slot:u16]`
    // plus a 4-byte slot, so we need free_space() ≥ entry_len + 4.
    struct LevelNode {
        page_id: PageId,
        first_key: crate::btree::KeyBytes,
    }

    let mut leaves: Vec<LevelNode> = Vec::new();
    let mut prev_leaf: Option<PageId> = None;
    let mut i = 0usize;
    while i < entries.len() {
        let g = bpm.new_page()?;
        let pid = g.page_id();
        let lsn = log_record(
            wal,
            tx,
            WalRecordType::PageInit {
                page_id: pid,
                kind: PageKind::BTreeLeaf as u8,
            },
        )?;
        let mut p = g.write();
        crate::btree::init_leaf(&mut p);
        p.set_page_lsn(lsn);

        let first_key = entries[i].0.clone();
        while i < entries.len() {
            let raw = crate::btree::encode_leaf_entry(&entries[i].0, entries[i].1);
            // page.insert needs raw.len() + slot (4) of free space.
            if p.free_space() < raw.len() + 4 {
                break;
            }
            p.insert(&raw)?;
            i += 1;
        }
        drop(p);

        // Link prev leaf's next_page_id forward.
        if let Some(prev) = prev_leaf {
            let prev_g = bpm.fetch_page(prev)?;
            prev_g.write().set_next_page_id(pid);
        }
        prev_leaf = Some(pid);
        leaves.push(LevelNode {
            page_id: pid,
            first_key,
        });
    }

    if leaves.len() == 1 {
        return Ok(leaves[0].page_id);
    }

    // Build internal levels until a single root remains. Each internal
    // node has a leftmost child (`p0`, stored in next_page_id) plus a
    // sequence of `(separator_key, child)` entries — the separator is
    // the first key of the child subtree.
    let mut current: Vec<LevelNode> = leaves;
    while current.len() > 1 {
        let mut next: Vec<LevelNode> = Vec::new();
        let mut j = 0;
        while j < current.len() {
            let g = bpm.new_page()?;
            let pid = g.page_id();
            let lsn = log_record(
                wal,
                tx,
                WalRecordType::PageInit {
                    page_id: pid,
                    kind: PageKind::BTreeInternal as u8,
                },
            )?;
            let mut p = g.write();
            crate::btree::init_internal(&mut p, current[j].page_id);
            p.set_page_lsn(lsn);
            let internal_first_key = current[j].first_key.clone();
            j += 1;

            while j < current.len() {
                let raw = crate::btree::encode_internal_entry(
                    &current[j].first_key,
                    current[j].page_id,
                );
                if p.free_space() < raw.len() + 4 {
                    break;
                }
                p.insert(&raw)?;
                j += 1;
            }
            drop(p);
            next.push(LevelNode {
                page_id: pid,
                first_key: internal_first_key,
            });
        }
        current = next;
    }
    Ok(current[0].page_id)
}

/// `CREATE SEQUENCE` — allocate a sequence relation page, initialise it,
/// and register the new sequence in `pg_sequence`.
fn perform_create_sequence(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedCreateSequenceStatement,
    tx: &mut Transaction,
) -> Result<()> {
    use crate::bootstrap::PG_SEQUENCE_PAGE_ID;
    // Pick a fresh seq_id (max existing + 1).
    let mut max_id: i32 = -1;
    for s in catalog.all_sequences()? {
        max_id = max_id.max(s.seq_id as i32);
    }
    let new_seq_id = max_id + 1;

    // Allocate the sequence relation page. The PageInit WAL record makes
    // the SequenceRel page_kind recoverable — redo will re-tag the page
    // even if it never made it to disk before a crash.
    let seq_page_id = {
        let g = bpm.new_page()?;
        let pid = g.page_id();
        let lsn = log_record(
            wal,
            tx,
            WalRecordType::PageInit {
                page_id: pid,
                kind: crate::page::PageKind::SequenceRel as u8,
            },
        )?;
        {
            let mut p = g.write();
            crate::sequence::init_sequence_page(&mut p);
            p.set_page_lsn(lsn);
        }
        pid
    };

    // Insert into pg_sequence.
    let row = serialize_tuple_mvcc(
        tx.id(),
        INVALID_TXN_ID,
        &[
            Value::Int(new_seq_id),
            Value::Varchar(stmt.name.clone()),
            Value::Int(seq_page_id as i32),
            Value::Int(stmt.increment as i32),
            Value::Int(stmt.start_value as i32),
            Value::Int(stmt.min_value as i32),
            Value::Int(stmt.max_value as i32),
        ],
    );
    insert_bytes(bpm, wal, tx, PG_SEQUENCE_PAGE_ID, &row)?;
    catalog.invalidate();
    Ok(())
}

/// `DROP SEQUENCE` — tombstone pg_sequence rows. Sequence relation pages
/// become orphaned; VACUUM later. (Phase 1-8.)
fn perform_drop_sequence(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedDropSequenceStatement,
    tx: &mut Transaction,
) -> Result<()> {
    use crate::bootstrap::PG_SEQUENCE_PAGE_ID;
    for (seq_id, _name, _page) in &stmt.seqs {
        let id = *seq_id;
        tombstone_catalog_rows(bpm, wal, tx, PG_SEQUENCE_PAGE_ID, |vals| {
            matches!(&vals[0], Value::Int(n) if *n as usize == id)
        })?;
    }
    catalog.invalidate();
    Ok(())
}

/// `DROP TABLE` — tombstone the catalog rows for the table and its indexes.
/// Heap pages and index pages become orphaned; VACUUM reclaims them later.
fn perform_drop_table(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedDropTableStatement,
    tx: &mut Transaction,
) -> Result<()> {
    use crate::bootstrap::{PG_ATTRIBUTE_PAGE_ID, PG_CLASS_PAGE_ID, PG_INDEX_PAGE_ID};
    for (table_id, _name) in &stmt.tables {
        // Tombstone pg_class row(s) for this table.
        tombstone_catalog_rows(bpm, wal, tx, PG_CLASS_PAGE_ID, |vals| {
            matches!(&vals[0], Value::Int(n) if *n as usize == *table_id)
        })?;
        // Tombstone every pg_attribute row for the table.
        tombstone_catalog_rows(bpm, wal, tx, PG_ATTRIBUTE_PAGE_ID, |vals| {
            matches!(&vals[0], Value::Int(n) if *n as usize == *table_id)
        })?;
        // Tombstone every pg_index row that points at the table.
        tombstone_catalog_rows(bpm, wal, tx, PG_INDEX_PAGE_ID, |vals| {
            matches!(&vals[2], Value::Int(n) if *n as usize == *table_id)
        })?;
    }
    catalog.invalidate();
    Ok(())
}

/// `DROP INDEX` — tombstone the pg_index row.
fn perform_drop_index(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedDropIndexStatement,
    tx: &mut Transaction,
) -> Result<()> {
    use crate::bootstrap::PG_INDEX_PAGE_ID;
    if stmt.index_id == usize::MAX {
        return Ok(()); // IF EXISTS on missing
    }
    let idx_id = stmt.index_id;
    tombstone_catalog_rows(bpm, wal, tx, PG_INDEX_PAGE_ID, |vals| {
        matches!(&vals[0], Value::Int(n) if *n as usize == idx_id)
    })?;
    catalog.invalidate();
    Ok(())
}

/// `TRUNCATE TABLE` — install a fresh empty heap page for each target,
/// orphaning the prior chain (cleaned by VACUUM). Indexes are also reset.
/// Returns the number of tables truncated.
fn perform_truncate(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedTruncateStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    use crate::bootstrap::{PG_CLASS_PAGE_ID, PG_INDEX_PAGE_ID};
    use crate::catalog::{pg_class_schema, pg_index_schema};
    for (table_id, _) in &stmt.tables {
        let table_id = *table_id;
        // Allocate a fresh empty heap page and patch pg_class.first_page_id.
        let new_page_id = {
            let g = bpm.new_page()?;
            g.page_id()
        };
        rewrite_catalog_row(
            bpm,
            wal,
            tx,
            PG_CLASS_PAGE_ID,
            &pg_class_schema(),
            |vals| matches!(&vals[0], Value::Int(n) if *n as usize == table_id),
            |vals| {
                let mut v = vals.to_vec();
                v[2] = Value::Int(new_page_id as i32);
                v
            },
        )?;
        // Reset every index on the table to a fresh empty leaf.
        for idx in catalog.indexes_for_table(table_id)? {
            let new_root = crate::btree::new_empty_root(bpm, wal, tx)?;
            let id = idx.index_id;
            rewrite_catalog_row(
                bpm,
                wal,
                tx,
                PG_INDEX_PAGE_ID,
                &pg_index_schema(),
                |vals| matches!(&vals[0], Value::Int(n) if *n as usize == id),
                |vals| {
                    let mut v = vals.to_vec();
                    v[4] = Value::Int(new_root as i32);
                    v
                },
            )?;
        }
    }
    catalog.invalidate();
    Ok(stmt.tables.len())
}

/// `VACUUM [tables]` — physically reclaim space from tuples that no live txn
/// can possibly see. Two pieces of work per heap page:
///   1. tombstone every dead tuple's slot (length=0) and `vacuum_compact()`
///      the page so the byte range is reusable.
///   2. for each index on the table, remove the (key, rid) entry pointing at
///      each reclaimed heap slot.
///
/// "Dead" means: the creating txn aborted, or the deleting txn committed
/// before the oldest currently-active txn started. We don't log per-page
/// vacuum records — instead we stamp every modified page's `page_lsn` with
/// a fresh marker LSN so recovery's redo skips earlier WAL records that
/// would otherwise try to set xmax on now-tombstoned slots.
fn perform_vacuum(
    bpm: &BufferPool,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    stmt: &AnalyzedVacuumStatement,
    tx: &mut Transaction,
) -> Result<()> {
    let oldest_xmin = tm.oldest_active_xmin();

    // Marker LSN to stamp every page we vacuum. Bracketed by Begin/Commit
    // so recovery's analyze pass sees a clean closed transaction (no DML
    // records, so it ends up classified as committed-no-op).
    let stamp_lsn = log_record(wal, tx, WalRecordType::Begin)?;

    for (table_id, _name) in &stmt.tables {
        let table = catalog
            .table_by_id(*table_id)?
            .ok_or_else(|| anyhow::anyhow!("table id {} not in catalog", table_id))?;
        let schema = table.to_schema();
        let indexes = catalog.indexes_for_table(*table_id)?;
        let index_meta: Vec<(usize, PageId, usize, DataType)> = indexes
            .iter()
            .map(|i| {
                let dt = table.columns[i.column_index].data_type;
                (i.index_id, i.root_page_id, i.column_index, dt)
            })
            .collect();

        // `prev` is the last page we kept in the chain; it owns the
        // next-pointer that's currently pointing at `cur`. We never unlink
        // the table's first page (`prev == None`) — pg_class.first_page_id
        // would have to be rewritten, and an always-present empty head is
        // harmless.
        let mut prev: Option<PageId> = None;
        let mut cur = table.first_page_id;
        while cur != crate::page::NO_NEXT_PAGE && cur < bpm.page_count() {
            let g = bpm.fetch_page(cur)?;
            let next;
            let page_is_heap;
            // Each dead row carries its own values + the t_ctid forward
            // pointer (sentinel-or-RID) so the index-cleanup pass below
            // can walk a HOT chain to its live successor and re-aim the
            // index entry there instead of just dropping it.
            let mut dead: Vec<(SlotId, Vec<Value>, (u32, u16))> = Vec::new();
            {
                let p = g.read();
                next = p.next_page_id();
                page_is_heap = matches!(p.page_kind(), crate::page::PageKind::Heap);
                if !page_is_heap {
                    drop(p);
                    drop(g);
                    prev = Some(cur);
                    cur = next;
                    continue;
                }
                let tc = p.tuple_count();
                for slot in 0..tc {
                    let raw = match p.get_tuple(slot) {
                        Some(r) => r,
                        None => continue,
                    };
                    let (xmin, xmax, ctid, values) =
                        crate::tuple::deserialize_tuple_mvcc_full(raw, &schema)?;
                    if is_tuple_dead(xmin, xmax, oldest_xmin, tm) {
                        dead.push((slot, values, ctid));
                    }
                }
            }
            let became_empty;
            {
                if !dead.is_empty() {
                    let mut p = g.write();
                    for (slot, _, _) in &dead {
                        p.delete(*slot)?;
                    }
                    p.vacuum_compact();
                    if p.page_lsn() < stamp_lsn {
                        p.set_page_lsn(stamp_lsn);
                    }
                }
                let p = g.read();
                became_empty = p.is_empty();
                drop(p);
            }
            // After dead-tuple cleanup, see whether every remaining tuple is
            // visible to every snapshot the VACUUM saw — i.e. xmin committed
            // before oldest_active_xmin and xmax is unset. If so, set the
            // page's all_visible bit. Future writes (insert/set_xmax/restore)
            // clear it automatically.
            if !became_empty {
                let mut all_visible = true;
                {
                    let p = g.read();
                    let n = p.tuple_count();
                    for slot in 0..n {
                        let raw = match p.get_tuple(slot) {
                            Some(r) => r,
                            None => continue,
                        };
                        let (xmin, xmax, _) = deserialize_tuple_mvcc(raw, &schema)?;
                        if xmax != INVALID_TXN_ID
                            || xmin >= oldest_xmin
                            || !matches!(tm.status(xmin), TxnStatus::Committed)
                        {
                            all_visible = false;
                            break;
                        }
                    }
                }
                if all_visible {
                    let mut p = g.write();
                    if !p.all_visible() {
                        p.set_all_visible(true);
                        if p.page_lsn() < stamp_lsn {
                            p.set_page_lsn(stamp_lsn);
                        }
                    }
                }
            }
            drop(g);

            // Index cleanup. For each dead row's index entry:
            //   - if the row was a HOT-chain head and its t_ctid leads
            //     to a still-live successor with the *same* key, rewire
            //     the index entry to point at the successor instead of
            //     deleting it (otherwise the live successor — which is
            //     reachable only via the chain — would become invisible
            //     to index lookups);
            //   - otherwise just drop the entry (this row was a plain
            //     INSERT-then-DELETE or its successor's key has changed
            //     so the same index entry no longer applies).
            for (idx_id, root, col_idx, dt) in &index_meta {
                for (slot, values, ctid) in &dead {
                    let key_value = &values[*col_idx];
                    if matches!(key_value, Value::Null) {
                        continue;
                    }
                    let key = crate::btree::encode_key(key_value);
                    let dead_rid: Rid = (cur, *slot);

                    // Walk the chain looking for a live successor.
                    let mut alive: Option<(Rid, Vec<Value>)> = None;
                    if !crate::tuple::is_no_ctid(*ctid) {
                        let mut step = (ctid.0, ctid.1);
                        for _ in 0..32 {
                            let g2 = bpm.fetch_page(step.0)?;
                            let p2 = g2.read();
                            let Some(raw) = p2.get_tuple(step.1) else {
                                break;
                            };
                            let (sxmin, sxmax, sctid, svals) =
                                crate::tuple::deserialize_tuple_mvcc_full(raw, &schema)?;
                            drop(p2);
                            drop(g2);
                            if sxmax == INVALID_TXN_ID
                                && matches!(tm.status(sxmin), TxnStatus::Committed)
                            {
                                alive = Some((step, svals));
                                break;
                            }
                            if crate::tuple::is_no_ctid(sctid) {
                                break;
                            }
                            step = (sctid.0, sctid.1);
                        }
                    }

                    let _ = crate::btree::delete(bpm, *root, &key, dead_rid, *dt)?;
                    if let Some((live_rid, live_vals)) = alive {
                        let live_key = &live_vals[*col_idx];
                        // Only rewire when the chain key didn't drift
                        // (the HOT path stamps t_ctid only when no
                        // indexed column changed, but a future relax
                        // could allow drift — be defensive).
                        if !matches!(live_key, Value::Null) && live_key == key_value {
                            let new_root = crate::btree::insert(
                                bpm, *root, &key, live_rid, *dt,
                            )?;
                            if new_root != *root {
                                update_index_root(bpm, *idx_id, new_root)?;
                            }
                        }
                    }
                }
            }

            // Unlink fully-empty pages from the chain (except the head)
            // and hand them back to the buffer pool's free list so the
            // next `new_page` reuses them instead of extending the file.
            if became_empty && prev.is_some() {
                let prev_pid = prev.unwrap();
                let g2 = bpm.fetch_page(prev_pid)?;
                let mut pp = g2.write();
                pp.set_next_page_id(next);
                if pp.page_lsn() < stamp_lsn {
                    pp.set_page_lsn(stamp_lsn);
                }
                drop(pp);
                drop(g2);
                bpm.recycle_page(cur);
                // prev stays the same — the page we just dropped is gone.
            } else {
                prev = Some(cur);
            }
            cur = next;
        }
    }

    // Sweep each index's leaf chain for empty leaves and unlink them from
    // their left neighbour. Internal nodes still point at the empty leaf,
    // but a search that descends to it just sees zero entries — correct
    // behaviour, only minor wasted space until a future merge pass takes
    // care of the parent. Range scans (which use the leaf chain) skip
    // the empty leaves entirely.
    for (table_id, _name) in &stmt.tables {
        for idx in catalog.indexes_for_table(*table_id)? {
            sweep_empty_leaves(bpm, idx.root_page_id, stamp_lsn)?;
        }
    }

    log_record(wal, tx, WalRecordType::Commit)?;
    wal.flush()?;
    Ok(())
}

/// Walk the leaf chain from the leftmost leaf, dropping every empty
/// (Page::is_empty) leaf out of the chain by patching the previous leaf's
/// next-pointer. The first leaf is always kept — its position is reachable
/// from the root's leftmost-child pointer, which we don't rewrite here.
fn sweep_empty_leaves(bpm: &BufferPool, root: PageId, stamp_lsn: Lsn) -> Result<()> {
    let first = crate::btree::leftmost_leaf(bpm, root)?;
    let mut prev = first;
    let mut cur = {
        let g = bpm.fetch_page(first)?;
        g.read().next_page_id()
    };
    while cur != crate::page::NO_NEXT_PAGE {
        let g = bpm.fetch_page(cur)?;
        let (next, empty) = {
            let p = g.read();
            (p.next_page_id(), p.is_empty())
        };
        drop(g);
        if empty {
            let g2 = bpm.fetch_page(prev)?;
            let mut pp = g2.write();
            pp.set_next_page_id(next);
            if pp.page_lsn() < stamp_lsn {
                pp.set_page_lsn(stamp_lsn);
            }
            drop(pp);
            drop(g2);
            // The leaf is unreachable from the chain. Internal nodes still
            // point at it, but a search descending there now lands on a
            // newly-Heap page and bails on the page_kind mismatch — so
            // only recycle leaves that no internal node points at. We
            // can't cheaply prove that here without parent tracking, so
            // leave btree leaf bytes in place for now (B2 already saved
            // the chain-traversal cost).
            // prev stays — the empty leaf is gone from the chain.
        } else {
            prev = cur;
        }
        cur = next;
    }
    Ok(())
}

/// A tuple is dead — physically reclaimable — when no live or future txn
/// could see it. Two cases:
///   - xmin aborted: never committed, so already invisible to everyone.
///   - xmax committed AND xmax < oldest_active_xmin: every active txn
///     started after the delete committed, so they all see the row as
///     deleted; the pre-image is unreachable.
fn is_tuple_dead(
    xmin: TxnId,
    xmax: TxnId,
    oldest_xmin: TxnId,
    tm: &TransactionManager,
) -> bool {
    if matches!(tm.status(xmin), TxnStatus::Aborted) {
        return true;
    }
    if xmax != INVALID_TXN_ID
        && xmax < oldest_xmin
        && matches!(tm.status(xmax), TxnStatus::Committed)
    {
        return true;
    }
    false
}

/// Walk a catalog page chain; for every visible row whose values satisfy
/// `pred`, mark it deleted (xmax = current tx) and write a Delete WAL record.
/// Used by DROP TABLE / DROP INDEX to remove catalog entries logically.
fn tombstone_catalog_rows(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    first_page: PageId,
    pred: impl Fn(&[Value]) -> bool,
) -> Result<()> {
    use crate::catalog::{pg_attribute_schema, pg_class_schema, pg_index_schema, pg_sequence_schema};
    use crate::bootstrap::{
        PG_ATTRIBUTE_PAGE_ID, PG_CLASS_PAGE_ID, PG_INDEX_PAGE_ID, PG_SEQUENCE_PAGE_ID,
    };
    let schema = match first_page {
        PG_CLASS_PAGE_ID => pg_class_schema(),
        PG_ATTRIBUTE_PAGE_ID => pg_attribute_schema(),
        PG_INDEX_PAGE_ID => pg_index_schema(),
        PG_SEQUENCE_PAGE_ID => pg_sequence_schema(),
        _ => bail!("tombstone_catalog_rows: unknown catalog page {first_page}"),
    };
    let mut victims: Vec<(PageId, SlotId)> = Vec::new();
    let mut cur = first_page;
    while cur != crate::page::NO_NEXT_PAGE && cur < bpm.page_count() {
        let g = bpm.fetch_page(cur)?;
        let p = g.read();
        let next = p.next_page_id();
        for slot in 0..p.tuple_count() {
            if let Some(raw) = p.get_tuple(slot) {
                let (_, xmax, vals) = deserialize_tuple_mvcc(raw, &schema)?;
                if xmax == 0 && pred(&vals) {
                    victims.push((cur, slot));
                }
            }
        }
        drop(p);
        drop(g);
        cur = next;
    }
    for (pid, slot) in victims {
        let g = bpm.fetch_page(pid)?;
        let mut p = g.write();
        p.set_tuple_xmax(slot, tx.id())?;
        let lsn = log_record(
            wal,
            tx,
            WalRecordType::Delete {
                rid: (pid, slot),
                xmax: tx.id(),
            },
        )?;
        p.set_page_lsn(lsn);
    }
    Ok(())
}

/// Replace the matching catalog row with a freshly-built one. Used by
/// TRUNCATE to swap out first_page_id and root_page_id pointers without
/// affecting the table_id / name. The old row is tombstoned so concurrent
/// readers under MVCC continue to see the prior version until commit.
fn rewrite_catalog_row(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    first_page: PageId,
    schema: &Schema,
    pred: impl Fn(&[Value]) -> bool,
    transform: impl Fn(&[Value]) -> Vec<Value>,
) -> Result<()> {
    let mut found: Option<(PageId, SlotId, Vec<Value>)> = None;
    let mut cur = first_page;
    while cur != crate::page::NO_NEXT_PAGE && cur < bpm.page_count() {
        let g = bpm.fetch_page(cur)?;
        let p = g.read();
        let next = p.next_page_id();
        for slot in 0..p.tuple_count() {
            if let Some(raw) = p.get_tuple(slot) {
                let (_, xmax, vals) = deserialize_tuple_mvcc(raw, schema)?;
                if xmax == 0 && pred(&vals) {
                    found = Some((cur, slot, vals));
                    break;
                }
            }
        }
        if found.is_some() {
            break;
        }
        drop(p);
        drop(g);
        cur = next;
    }
    let (page_id, slot, old_vals) = found
        .ok_or_else(|| anyhow::anyhow!("rewrite_catalog_row: matching row not found"))?;
    // Tombstone the old row.
    {
        let g = bpm.fetch_page(page_id)?;
        let mut p = g.write();
        p.set_tuple_xmax(slot, tx.id())?;
        let lsn = log_record(
            wal,
            tx,
            WalRecordType::Delete {
                rid: (page_id, slot),
                xmax: tx.id(),
            },
        )?;
        p.set_page_lsn(lsn);
    }
    // Insert the rewritten one (system-txn xmin to keep visible across recoveries).
    let new_vals = transform(&old_vals);
    let bytes = serialize_tuple_mvcc(crate::bootstrap::SYSTEM_TXN_ID, INVALID_TXN_ID, &new_vals);
    insert_bytes(bpm, wal, tx, first_page, &bytes)?;
    Ok(())
}

/// Walk the t_ctid chain from `start_rid` looking for a tuple visible to
/// `snapshot`. Returns the first visible (rid, values) pair, or None if
/// the chain ends without finding one. Bounded chain length so a corrupt
/// pointer can't loop forever.
fn follow_chain_visible(
    bpm: &BufferPool,
    schema: &Schema,
    snapshot: &Snapshot,
    tm: &TransactionManager,
    start_rid: Rid,
) -> Result<Option<(Rid, Vec<Value>)>> {
    const MAX_CHAIN: usize = 32;
    let mut cur = start_rid;
    for _ in 0..MAX_CHAIN {
        let g = bpm.fetch_page(cur.0)?;
        let p = g.read();
        let Some(raw) = p.get_tuple(cur.1) else {
            return Ok(None);
        };
        let (xmin, xmax, ctid, values) =
            crate::tuple::deserialize_tuple_mvcc_full(raw, schema)?;
        let page_av = p.all_visible();
        drop(p);
        drop(g);
        let visible = page_av || visibility::is_visible(xmin, xmax, snapshot, tm);
        if visible {
            return Ok(Some((cur, values)));
        }
        if crate::tuple::is_no_ctid(ctid) {
            return Ok(None);
        }
        cur = (ctid.0, ctid.1);
    }
    Ok(None)
}

/// Confirm every FOREIGN KEY on `table_id` is satisfied by the candidate
/// row. NULL FK values are allowed (PG semantics: a NULL key references
/// nothing). Otherwise a visible row with `ref_column = value` must
/// exist in `ref_table`. Bails with SQLSTATE 23503.
fn enforce_fk_on_child_write(
    bpm: &BufferPool,
    tm: &TransactionManager,
    catalog: &Catalog,
    table_id: usize,
    values: &[Value],
    tx: &Transaction,
) -> Result<()> {
    let constraints = catalog.constraints_for_table(table_id)?;
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table {table_id} missing"))?;
    let snapshot = tx
        .snapshot()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no snapshot for tx"))?;
    for c in constraints {
        if !matches!(c.kind, crate::catalog::ConstraintKind::ForeignKey) {
            continue;
        }
        let fk = match crate::catalog::ForeignKeyDef::decode(&c.definition) {
            Some(f) => f,
            None => bail!("malformed FK definition: {}", c.definition),
        };
        let col_idx = table
            .columns
            .iter()
            .position(|tc| tc.name == fk.child_column)
            .ok_or_else(|| anyhow::anyhow!("FK '{}' references missing local column", c.name))?;
        let v = &values[col_idx];
        if matches!(v, Value::Null) {
            continue;
        }
        let (ref_table_id, _) = catalog
            .find_table(&fk.ref_table)?
            .ok_or_else(|| anyhow::anyhow!("FK target '{}' missing", fk.ref_table))?;
        if !fk_target_exists(bpm, catalog, tm, &snapshot, ref_table_id, &fk.ref_column, v)? {
            bail!(
                "insert or update on \"{}\" violates foreign key constraint \"{}\" [SQLSTATE 23503]",
                table.name,
                c.name
            );
        }
    }
    Ok(())
}

fn fk_target_exists(
    bpm: &BufferPool,
    catalog: &Catalog,
    tm: &TransactionManager,
    snapshot: &Snapshot,
    ref_table_id: usize,
    ref_column: &str,
    target: &Value,
) -> Result<bool> {
    let ref_table = catalog
        .table_by_id(ref_table_id)?
        .ok_or_else(|| anyhow::anyhow!("ref table {ref_table_id} missing"))?;
    let ref_col_idx = ref_table
        .columns
        .iter()
        .position(|c| c.name == ref_column)
        .ok_or_else(|| anyhow::anyhow!("ref column {ref_column} missing"))?;
    let schema = ref_table.to_schema();

    // Fast path: a covering index on the ref column lets us answer with
    // an O(log n) btree probe instead of an O(n) heap scan. PRIMARY KEYs
    // and explicit UNIQUE indexes both qualify; secondary non-unique
    // indexes work too (we only need to find *any* visible matching row).
    let indexes = catalog.indexes_for_table(ref_table_id)?;
    if let Some(idx) = indexes.iter().find(|i| i.column_index == ref_col_idx) {
        let dt = ref_table.columns[ref_col_idx].data_type;
        let key = crate::btree::encode_key(target);
        let leaf_id = crate::btree::descend_to_leaf(bpm, idx.root_page_id, &key, dt)?;
        let candidates: Vec<Rid> = {
            let g = bpm.fetch_page(leaf_id)?;
            let p = g.read();
            crate::btree::leaf_lookup_eq(&p, &key, dt)?
        };
        for rid in candidates {
            let g = bpm.fetch_page(rid.0)?;
            let p = g.read();
            let Some(raw) = p.get_tuple(rid.1) else {
                continue;
            };
            let (xmin, xmax, vals) = deserialize_tuple_mvcc(raw, &schema)?;
            if !visibility::is_visible(xmin, xmax, snapshot, tm) {
                continue;
            }
            // Stale index entry under a different key would still get
            // surfaced by leaf_lookup_eq if memcmp succeeded, so confirm
            // the heap value really matches.
            if values_equal(&vals[ref_col_idx], target) {
                return Ok(true);
            }
        }
        return Ok(false);
    }

    // Fallback: full visible scan + linear probe.
    let (_schema, rows) = visible_rows(bpm, catalog, ref_table_id, snapshot, tm)?;
    for (_, _, vals) in rows {
        if values_equal(&vals[ref_col_idx], target) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) => false,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Varchar(x), Value::Varchar(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Double(x), Value::Double(y)) => x.to_bits() == y.to_bits(),
        (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
        (Value::Date(x), Value::Date(y)) => x == y,
        (Value::Time(x), Value::Time(y)) => x == y,
        _ => false,
    }
}

/// Process-wide cache of bound CHECK predicates keyed by `constraint_id`.
/// pg_constraint never reuses a constraint id (perform_create_table picks
/// `max+1`), so the cache is append-only — no invalidation needed even
/// across DROP / CREATE cycles. Re-parsing + re-binding every CHECK on
/// every row was the dominant cost on tables with multiple CHECK
/// constraints.
fn analyzed_check_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<usize, std::sync::Arc<crate::analyzer::AnalyzedExpr>>,
> {
    static C: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<usize, std::sync::Arc<crate::analyzer::AnalyzedExpr>>,
        >,
    > = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Evaluate every CHECK constraint registered for `table_id` against a
/// fully-coerced row. AnalyzedExpr cached by constraint_id; bails with
/// SQLSTATE 23514 on the first failing predicate.
fn enforce_check_constraints(
    catalog: &Catalog,
    table_id: usize,
    table_name: &str,
    values: &[Value],
) -> Result<()> {
    let constraints = catalog.constraints_for_table(table_id)?;
    if constraints.is_empty() {
        return Ok(());
    }
    let row = Tuple::new(values.to_vec());
    let cache = analyzed_check_cache();
    for c in constraints {
        if !matches!(c.kind, crate::catalog::ConstraintKind::Check) {
            continue;
        }
        let analyzed: std::sync::Arc<crate::analyzer::AnalyzedExpr> = {
            let mut g = cache.lock().unwrap();
            if let Some(a) = g.get(&c.constraint_id).cloned() {
                a
            } else {
                let expr = crate::parser::parse_expr_str(&c.definition).map_err(|e| {
                    anyhow::anyhow!(
                        "stored CHECK '{}' failed to parse: {e}",
                        c.definition
                    )
                })?;
                let bound =
                    crate::analyzer::analyze_expr_for_table(catalog, table_id, table_name, &expr)?;
                let arc = std::sync::Arc::new(bound);
                g.insert(c.constraint_id, arc.clone());
                arc
            }
        };
        match evaluate_expr(&analyzed, &row)? {
            Value::Bool(true) => continue,
            Value::Bool(false) => bail!(
                "new row violates check constraint \"{}\" [SQLSTATE 23514]",
                c.name
            ),
            Value::Null => continue, // PG semantics: NULL check passes
            other => bail!("CHECK '{}' returned non-boolean: {other:?}", c.name),
        }
    }
    Ok(())
}

/// Insert one COPY row. Caller has already turned the wire-format text into a
/// per-table-column `Value` vector (NULL-padded for unmapped columns). Mirrors
/// perform_insert's per-row work: heap insert + lock + index maintenance.
///
/// `tail_hint` is a page id believed to be at or near the end of the heap
/// chain — bulk callers (`run_copy_in`) thread the hint forward so we don't
/// rescan the whole chain on every row. Returns the Rid plus an updated hint.
pub fn perform_copy_row(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    catalog: &Catalog,
    table_id: usize,
    values: Vec<Value>,
    tx: &mut Transaction,
    tail_hint: Option<PageId>,
) -> Result<(Rid, PageId)> {
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?;
    let coerced: Vec<Value> = values
        .into_iter()
        .zip(table.columns.iter())
        .map(|(v, c)| coerce_for_storage(v, c.data_type))
        .collect::<Result<_>>()?;
    enforce_check_constraints(catalog, table_id, &table.name, &coerced)?;
    let tm_for_fk = Arc::clone(tx.tm());
    enforce_fk_on_child_write(bpm, &tm_for_fk, catalog, table_id, &coerced, tx)?;
    let bytes = serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &coerced);
    let (rid, _lsn, new_tail) =
        insert_bytes_hinted(bpm, wal, tx, table.first_page_id, &bytes, tail_hint)?;
    lm.lock(tx.id(), rid, LockMode::Exclusive)
        .map_err(|e| anyhow::anyhow!("X-lock on {rid:?}: {e}"))?;
    tx.add_lock(rid);
    let tm = Arc::clone(tx.tm());
    index_insert_for_row_with_lm(bpm, Some(lm), wal, Some(&tm), catalog, tx, table_id, &coerced, rid)?;
    Ok((rid, new_tail))
}

fn perform_insert(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedInsertStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    let table = catalog
        .table_by_id(stmt.table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {} not in catalog", stmt.table_id))?;
    // Empty tuple for evaluating constant expressions in VALUES (no column
    // refs allowed — those would be caught earlier).
    let empty = Tuple::new(Vec::new());
    let mut count = 0;
    for row in &stmt.rows {
        // Pre-evaluate SequenceCalls per row so each row gets a fresh
        // nextval. Other expression nodes are left intact.
        let mut row_owned = row.clone();
        for e in row_owned.iter_mut() {
            evaluate_sequence_calls_in_expr(e, bpm, wal, tx)?;
        }
        let raw: Vec<Value> = row_owned
            .iter()
            .map(|e| evaluate_expr(e, &empty))
            .collect::<Result<_>>()?;
        // Coerce values to the column's storage type. The analyzer accepts INT
        // values for DOUBLE columns; the storage layer needs the exact width.
        let values: Vec<Value> = raw
            .into_iter()
            .zip(table.columns.iter())
            .map(|(v, c)| coerce_for_storage(v, c.data_type))
            .collect::<Result<_>>()?;

        // ON CONFLICT pre-flight: detect a conflicting unique-index entry
        // before any heap write. If one exists, branch to DO NOTHING / DO
        // UPDATE; otherwise fall through to the normal INSERT path.
        if let Some(on_conflict) = &stmt.on_conflict {
            let tm = Arc::clone(tx.tm());
            if let Some((_idx, conflict_rid)) = detect_unique_conflict(
                bpm, lm, &tm, catalog, stmt.table_id, &values, tx,
            )? {
                match on_conflict {
                    crate::analyzer::AnalyzedOnConflict::DoNothing => {
                        // Skip — neither heap nor index touched.
                        continue;
                    }
                    crate::analyzer::AnalyzedOnConflict::DoUpdate {
                        assignments,
                        where_clause,
                    } => {
                        let updated = on_conflict_do_update(
                            bpm, lm, wal, catalog, stmt.table_id, &table.name,
                            conflict_rid, assignments, where_clause.as_ref(), tx,
                        )?;
                        if updated {
                            count += 1;
                        }
                        continue;
                    }
                }
            }
        }

        enforce_check_constraints(catalog, stmt.table_id, &stmt.table_name, &values)?;
        let tm_for_fk = Arc::clone(tx.tm());
        enforce_fk_on_child_write(bpm, &tm_for_fk, catalog, stmt.table_id, &values, tx)?;
        let bytes = serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &values);
        let (rid, _lsn) = insert_bytes(bpm, wal, tx, table.first_page_id, &bytes)?;
        lm.lock(tx.id(), rid, LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {rid:?}: {e}"))?;
        tx.add_lock(rid);
        // Maintain every index on this table. Index lookup of `key → rid`
        // and heap visibility check work together to keep aborts correct:
        // if this txn aborts, the index entry stays but the heap tuple has
        // an aborted xmin so visibility filters it out.
        let tm = Arc::clone(tx.tm());
        index_insert_for_row_with_lm(
            bpm, Some(lm), wal, Some(&tm), catalog, tx, stmt.table_id, &values, rid,
        )?;
        count += 1;
    }
    Ok(count)
}

/// Walk every index on `table_id` and add a `(key, rid)` entry for each.
/// Treats NULL values as not-indexed.
///
/// The index is *add-only*: DELETE/UPDATE don't touch it. Stale entries are
/// fine because every IndexScan re-checks heap visibility — an aborted or
/// deleted heap row is filtered there. This matches PostgreSQL's split
/// between index lookup and heap visibility, and avoids the abort-correctness
/// gap of physically removing entries on DELETE (a DELETE that aborts must
/// leave the heap row visible; the corresponding index entry must therefore
/// also still be there).
fn index_insert_for_row(
    bpm: &BufferPool,
    wal: &WalManager,
    catalog: &Catalog,
    tx: &mut Transaction,
    table_id: usize,
    values: &[Value],
    rid: Rid,
) -> Result<()> {
    index_insert_for_row_with_lm(bpm, None, wal, None, catalog, tx, table_id, values, rid)
}

/// Same as `index_insert_for_row` but accepts a LockManager + TransactionManager
/// so unique indexes can wait for in-progress conflicting transactions before
/// raising 23505. The non-LM variant is used by perform_create_index where the
/// catalog rows are bootstrapped under SYSTEM_TXN_ID and uniqueness is by
/// construction.
fn index_insert_for_row_with_lm(
    bpm: &BufferPool,
    lm: Option<&LockManager>,
    wal: &WalManager,
    tm: Option<&TransactionManager>,
    catalog: &Catalog,
    tx: &mut Transaction,
    table_id: usize,
    values: &[Value],
    rid: Rid,
) -> Result<()> {
    let indexes = catalog.indexes_for_table(table_id)?;
    if indexes.is_empty() {
        return Ok(());
    }
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table {table_id} missing"))?;
    let schema = table.to_schema();
    for idx in indexes {
        let key_value = &values[idx.column_index];
        if matches!(key_value, Value::Null) {
            continue;
        }
        let key = crate::btree::encode_key(key_value);
        let dt = table.columns[idx.column_index].data_type;
        if idx.is_unique {
            // Speculative uniqueness check. We have the LM/TM only when
            // called from a real INSERT; bootstrap (CREATE INDEX scan) hits
            // the non-LM branch above and skips the check entirely — that
            // path is system-txn data, no contention to resolve.
            if let (Some(lm), Some(tm)) = (lm, tm) {
                check_unique_or_wait(bpm, lm, tm, &idx, &key, dt, &schema, tx, rid)?;
            }
        }
        let new_root = crate::btree::insert(bpm, idx.root_page_id, &key, rid, dt)?;
        if new_root != idx.root_page_id {
            update_index_root(bpm, idx.index_id, new_root)?;
        }
        log_record(
            wal,
            tx,
            WalRecordType::IndexInsert {
                index_id: idx.index_id as u64,
                key,
                rid,
            },
        )?;
    }
    Ok(())
}

/// Apply an `ON CONFLICT ... DO UPDATE` clause to a single conflicting
/// row. Behaves like a one-row UPDATE: re-fetch the conflicting tuple,
/// run the analyzer-bound WHERE filter (if any), evaluate each
/// assignment against the existing row, set xmax + write a new heap
/// tuple + WAL Delete + maintain indexes. Returns true if the row was
/// actually updated (false ⇒ WHERE filtered it out, equivalent to
/// DO NOTHING for this row).
fn on_conflict_do_update(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    catalog: &Catalog,
    table_id: usize,
    table_name: &str,
    target_rid: Rid,
    assignments: &[crate::analyzer::AnalyzedAssignment],
    where_clause: Option<&AnalyzedExpr>,
    tx: &mut Transaction,
) -> Result<bool> {
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?;
    let schema = table.to_schema();
    // Read the existing tuple's values.
    let old_values: Vec<Value> = {
        let g = bpm.fetch_page(target_rid.0)?;
        let p = g.read();
        let raw = p
            .get_tuple(target_rid.1)
            .ok_or_else(|| anyhow::anyhow!("ON CONFLICT target tuple disappeared"))?;
        let (_, _, vals) = deserialize_tuple_mvcc(raw, &schema)?;
        vals
    };
    let row = Tuple::new(old_values.clone());
    if !matches(where_clause, &row)? {
        return Ok(false);
    }
    let mut new_values = old_values.clone();
    for a in assignments {
        let v = evaluate_expr(&a.value, &row)?;
        let target = table.columns[a.column_index].data_type;
        new_values[a.column_index] = coerce_for_storage(v, target)?;
    }
    enforce_check_constraints(catalog, table_id, table_name, &new_values)?;

    // Tombstone old + log Delete.
    lm.lock(tx.id(), target_rid, LockMode::Exclusive)
        .map_err(|e| anyhow::anyhow!("X-lock on {target_rid:?}: {e}"))?;
    tx.add_lock(target_rid);
    {
        let g = bpm.fetch_page(target_rid.0)?;
        let mut p = g.write();
        p.set_tuple_xmax(target_rid.1, tx.id())?;
        let lsn = log_record(
            wal,
            tx,
            WalRecordType::Delete {
                rid: target_rid,
                xmax: tx.id(),
            },
        )?;
        p.set_page_lsn(lsn);
    }

    // Insert new version + maintain indexes (uniqueness check sees our own
    // freshly-deleted row and skips it).
    let new_bytes = serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &new_values);
    let (new_rid, _lsn) = insert_bytes(bpm, wal, tx, table.first_page_id, &new_bytes)?;
    lm.lock(tx.id(), new_rid, LockMode::Exclusive)
        .map_err(|e| anyhow::anyhow!("X-lock on {new_rid:?}: {e}"))?;
    tx.add_lock(new_rid);
    let tm_arc = Arc::clone(tx.tm());
    index_insert_for_row_with_lm(
        bpm, Some(lm), wal, Some(&tm_arc), catalog, tx, table_id, &new_values, new_rid,
    )?;
    Ok(true)
}

/// Like `check_unique_or_wait` but returns the conflicting heap RID
/// instead of erroring. Used by INSERT ... ON CONFLICT to dispatch
/// before any heap write happens. None ⇒ no conflict, the caller
/// can proceed with the normal INSERT path.
fn detect_unique_conflict(
    bpm: &BufferPool,
    lm: &LockManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    table_id: usize,
    values: &[Value],
    tx: &mut Transaction,
) -> Result<Option<(crate::catalog::IndexDef, Rid)>> {
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table {table_id} missing"))?;
    let schema = table.to_schema();
    for idx in catalog.indexes_for_table(table_id)? {
        if !idx.is_unique {
            continue;
        }
        let key_value = &values[idx.column_index];
        if matches!(key_value, Value::Null) {
            continue;
        }
        let key = crate::btree::encode_key(key_value);
        let dt = table.columns[idx.column_index].data_type;
        // Re-run the visibility loop, returning the first live RID we see.
        loop {
            let leaf_id = crate::btree::descend_to_leaf(bpm, idx.root_page_id, &key, dt)?;
            let candidates: Vec<Rid> = {
                let g = bpm.fetch_page(leaf_id)?;
                let p = g.read();
                crate::btree::leaf_lookup_eq(&p, &key, dt)?
            };
            let mut to_wait: Option<u64> = None;
            let mut found: Option<Rid> = None;
            for cand in candidates {
                let g = bpm.fetch_page(cand.0)?;
                let p = g.read();
                let raw = match p.get_tuple(cand.1) {
                    Some(r) => r,
                    None => continue,
                };
                let (xmin, xmax, _) = deserialize_tuple_mvcc(raw, &schema)?;
                drop(p);
                drop(g);
                if xmin == tx.id() && xmax == INVALID_TXN_ID {
                    found = Some(cand);
                    break;
                }
                match tm.status(xmin) {
                    TxnStatus::Aborted => continue,
                    TxnStatus::InProgress => {
                        to_wait = Some(xmin);
                        break;
                    }
                    TxnStatus::Committed => {
                        if xmax == INVALID_TXN_ID {
                            found = Some(cand);
                            break;
                        }
                        if xmax == tx.id() {
                            continue;
                        }
                        match tm.status(xmax) {
                            TxnStatus::Committed => continue,
                            TxnStatus::Aborted => {
                                found = Some(cand);
                                break;
                            }
                            TxnStatus::InProgress => {
                                to_wait = Some(xmax);
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(r) = found {
                return Ok(Some((idx.clone(), r)));
            }
            match to_wait {
                None => break,
                Some(other) => {
                    lm.wait_for_txn_completion(other, tm)
                        .map_err(|e| anyhow::anyhow!("waiting for txn {other}: {e}"))?;
                    continue;
                }
            }
        }
    }
    Ok(None)
}

/// PG-style speculative uniqueness check. Looks up every existing entry for
/// `key` in `idx`, fetches the corresponding heap tuple, and decides whether
/// it represents a live row that would clash with our insert:
///
///   - heap tuple aborted-creation       → ignore
///   - heap tuple deleted-by-committed   → ignore
///   - heap tuple alive (xmax = 0)       → 23505
///   - heap tuple xmin in-progress       → wait, then re-decide
///   - heap tuple xmin == self           → ignore (our own previously-inserted row)
fn check_unique_or_wait(
    bpm: &BufferPool,
    lm: &LockManager,
    tm: &TransactionManager,
    idx: &crate::catalog::IndexDef,
    key: &[u8],
    dt: DataType,
    schema: &Schema,
    tx: &mut Transaction,
    own_rid: Rid,
) -> Result<()> {
    use crate::transaction_manager::TxnStatus;
    loop {
        let leaf_id = crate::btree::descend_to_leaf(bpm, idx.root_page_id, key, dt)?;
        let candidates: Vec<Rid> = {
            let g = bpm.fetch_page(leaf_id)?;
            let p = g.read();
            crate::btree::leaf_lookup_eq(&p, key, dt)?
        };
        let mut to_wait: Option<u64> = None;
        for cand in candidates {
            if cand == own_rid {
                continue;
            }
            let g = bpm.fetch_page(cand.0)?;
            let p = g.read();
            let raw = match p.get_tuple(cand.1) {
                Some(r) => r,
                None => continue,
            };
            let (xmin, xmax, _) = deserialize_tuple_mvcc(raw, schema)?;
            drop(p);
            drop(g);

            if xmin == tx.id() {
                // Our own earlier insert in this txn. If we deleted it
                // ourselves (xmax == self), no conflict. If still live to
                // ourselves, that's an actual self-conflict the caller wants
                // surfaced.
                if xmax == tx.id() {
                    continue;
                }
                if xmax == INVALID_TXN_ID {
                    bail!(
                        "duplicate key value violates unique constraint \"{}\" [SQLSTATE 23505]",
                        idx.name
                    );
                }
                continue;
            }
            // xmin not us. Decide based on creator status.
            match tm.status(xmin) {
                TxnStatus::Aborted => continue, // never visible
                TxnStatus::InProgress => {
                    to_wait = Some(xmin);
                    break;
                }
                TxnStatus::Committed => {
                    // Creator committed. Now look at the deleter.
                    if xmax == INVALID_TXN_ID {
                        bail!(
                            "duplicate key value violates unique constraint \"{}\" [SQLSTATE 23505]",
                            idx.name
                        );
                    }
                    if xmax == tx.id() {
                        // We deleted it earlier in this txn — taking its key
                        // back is fine.
                        continue;
                    }
                    match tm.status(xmax) {
                        TxnStatus::Committed => continue, // row gone
                        TxnStatus::Aborted => {
                            bail!(
                                "duplicate key value violates unique constraint \"{}\" [SQLSTATE 23505]",
                                idx.name
                            );
                        }
                        TxnStatus::InProgress => {
                            to_wait = Some(xmax);
                            break;
                        }
                    }
                }
            }
        }
        match to_wait {
            None => return Ok(()),
            Some(other) => {
                lm.wait_for_txn_completion(other, tm)
                    .map_err(|e| anyhow::anyhow!("waiting for txn {other}: {e}"))?;
                // Re-evaluate: that txn may have committed or aborted; the
                // next loop iteration sees the new status.
            }
        }
    }
}

/// Patch the `root_page_id` column in the `pg_index` row whose `index_id`
/// matches. Called when a tree split bubbles up to a brand-new root.
///
/// We tombstone the old row and append a new one with the updated root in
/// the same `pg_index` chain. Since `Catalog::all_indexes` reads the chain
/// linearly and skips tombstones, lookups see only the latest version.
fn update_index_root(bpm: &BufferPool, index_id: usize, new_root: PageId) -> Result<()> {
    use crate::bootstrap::{PG_INDEX_PAGE_ID, SYSTEM_TXN_ID};
    use crate::catalog::pg_index_schema;
    let schema = pg_index_schema();

    // Pass 1: find the matching row and snapshot its other columns.
    let mut found: Option<(PageId, SlotId, Vec<Value>)> = None;
    let mut cur = PG_INDEX_PAGE_ID;
    while cur != crate::page::NO_NEXT_PAGE && cur < bpm.page_count() {
        let g = bpm.fetch_page(cur)?;
        let p = g.read();
        let next = p.next_page_id();
        let n = p.tuple_count();
        for slot in 0..n {
            if let Some(raw) = p.get_tuple(slot) {
                let (_, _, vals) = deserialize_tuple_mvcc(raw, &schema)?;
                if matches!(&vals[0], Value::Int(n) if *n as usize == index_id) {
                    found = Some((cur, slot, vals));
                    break;
                }
            }
        }
        if found.is_some() {
            break;
        }
        drop(p);
        drop(g);
        cur = next;
    }
    let (page_id, slot, vals) =
        found.ok_or_else(|| anyhow::anyhow!("update_index_root: index {index_id} not found"))?;

    // Pass 2: tombstone old, append new. Skip WAL — these rewrites are rare
    // and the system-txn xmin keeps them visible across crashes.
    {
        let g = bpm.fetch_page(page_id)?;
        let mut p = g.write();
        p.delete(slot)?;
    }
    // Re-emit ALL columns the original row carried, including is_unique
    // (added in Phase 4-1b). Older codepaths only re-wrote 5 of the 6
    // columns, which left a row that decode_tuple later couldn't parse.
    let is_unique_val = vals
        .get(5)
        .cloned()
        .unwrap_or(Value::Bool(false));
    let new_bytes = serialize_tuple_mvcc(
        SYSTEM_TXN_ID,
        INVALID_TXN_ID,
        &[
            vals[0].clone(),
            vals[1].clone(),
            vals[2].clone(),
            vals[3].clone(),
            Value::Int(new_root as i32),
            is_unique_val,
        ],
    );
    let g = bpm.fetch_page(PG_INDEX_PAGE_ID)?;
    let mut head = g.write();
    head.insert(&new_bytes)?;
    Ok(())
}

/// Coerce an evaluated value into the column's declared storage type.
/// The only widening currently allowed is INT → DOUBLE, mirroring the
/// analyzer's `assignable()`. NULL passes through unchanged.
fn coerce_for_storage(v: Value, target: DataType) -> Result<Value> {
    Ok(match (&v, target) {
        (Value::Null, _) => v,
        (Value::Int(i), DataType::Double) => Value::Double(*i as f64),
        _ => v,
    })
}

// -- DELETE / UPDATE (not Executors either — both are bulk side effects) -----

/// Visit all *visible* rows of a table under the given snapshot, returning
/// (rid, values) pairs for predicate evaluation. Tuples invisible to the
/// snapshot (uncommitted others, future-tx, already-deleted) are skipped.
/// Visible rows that match a WHERE predicate. Picks an index-driven path
/// when the predicate has an `indexed_col = lit` or BETWEEN shape; falls
/// back to a full heap scan otherwise. Used by UPDATE/DELETE so they don't
/// scan the whole table on every statement when an equality predicate on
/// an indexed column is available.
fn matching_rows(
    bpm: &BufferPool,
    catalog: &Catalog,
    table_id: usize,
    where_clause: &Option<AnalyzedExpr>,
    snapshot: &Snapshot,
    tm: &TransactionManager,
) -> Result<(Schema, Vec<(PageId, SlotId, Vec<Value>)>)> {
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?;
    let schema = table.to_schema();

    if let Some((idx, range, key_type)) = find_indexable_predicate(where_clause, table_id, catalog)?
    {
        let mut out = Vec::new();
        let start_key = match &range {
            IndexRange::Eq { key } => key.clone(),
            IndexRange::Between { low, .. } => low.clone(),
        };
        let mut cursor = crate::btree::first_ge(bpm, idx.root_page_id, &start_key, key_type)?;
        while let Some(c) = cursor {
            let Some((key, rid, next)) = crate::btree::read_at(bpm, c)? else {
                break;
            };
            cursor = next;
            // Stop once we've stepped past the upper bound.
            let past = match &range {
                IndexRange::Eq { key: target } => {
                    crate::btree::compare_keys(&key, target, key_type)? != std::cmp::Ordering::Equal
                }
                IndexRange::Between { high, .. } => {
                    crate::btree::compare_keys(&key, high, key_type)? == std::cmp::Ordering::Greater
                }
            };
            if past {
                break;
            }
            // Follow the t_ctid chain to find the live successor of a
            // possibly-stale index entry. Without this, a HOT-update
            // index lookup would land on the tombstoned old row and
            // miss the new one.
            let Some((live_rid, values)) =
                follow_chain_visible(bpm, &schema, snapshot, tm, rid)?
            else {
                continue;
            };
            // Re-check the full predicate: stale index entries, AND-clauses
            // beyond the indexable one, etc.
            let t = Tuple::new(values.clone());
            if !matches(where_clause.as_ref(), &t)? {
                continue;
            }
            out.push((live_rid.0, live_rid.1, values));
        }
        return Ok((schema, out));
    }

    // No usable index — full scan, applying the predicate row-by-row so
    // the contract stays "returned rows already passed WHERE."
    let (schema, all) = visible_rows(bpm, catalog, table_id, snapshot, tm)?;
    let mut out = Vec::with_capacity(all.len());
    for (pid, slot, values) in all {
        let t = Tuple::new(values.clone());
        if matches(where_clause.as_ref(), &t)? {
            out.push((pid, slot, values));
        }
    }
    Ok((schema, out))
}

fn visible_rows(
    bpm: &BufferPool,
    catalog: &Catalog,
    table_id: usize,
    snapshot: &Snapshot,
    tm: &TransactionManager,
) -> Result<(Schema, Vec<(PageId, SlotId, Vec<Value>)>)> {
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?;
    let schema = table.to_schema();
    let mut out = Vec::new();
    let mut cur = table.first_page_id;
    while cur != crate::page::NO_NEXT_PAGE && cur < bpm.page_count() {
        let guard = bpm.fetch_page(cur)?;
        let page = guard.read();
        let next = page.next_page_id();
        let tc = page.tuple_count();
        // Page-level VM fast path: every live tuple on an all_visible
        // page is visible to every active snapshot, so the per-tuple
        // MVCC check (and its CLOG / ATT lookups) collapses to a
        // single byte read.
        let page_all_visible = page.all_visible();
        for slot in 0..tc {
            if let Some(raw) = page.get_tuple(slot) {
                let (xmin, xmax, values) = deserialize_tuple_mvcc(raw, &schema)?;
                let vis = page_all_visible
                    || visibility::is_visible(xmin, xmax, snapshot, tm);
                if vis {
                    out.push((cur, slot, values));
                }
            }
        }
        drop(page);
        drop(guard);
        cur = next;
    }
    Ok((schema, out))
}

fn matches(predicate: Option<&AnalyzedExpr>, tuple: &Tuple) -> Result<bool> {
    let Some(pred) = predicate else { return Ok(true) };
    match evaluate_expr(pred, tuple)? {
        Value::Bool(b) => Ok(b),
        Value::Null => Ok(false),
        other => bail!("WHERE predicate must be boolean, got {other:?}"),
    }
}

fn perform_delete(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    stmt: &AnalyzedDeleteStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    let table = catalog
        .table_by_id(stmt.table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {} not in catalog", stmt.table_id))?;
    let schema = table.to_schema();

    const MAX_RETRIES: u32 = 8;
    'restart: for _attempt in 0..MAX_RETRIES {
        let snapshot = tx.tm().snapshot(tx.id());
        tx.set_snapshot(snapshot.clone());
        let (_schema, rows) = matching_rows(
            bpm, catalog, stmt.table_id, &stmt.where_clause, &snapshot, tm,
        )?;
        let victims: Vec<(Rid, Vec<Value>)> = rows
            .into_iter()
            .map(|(pid, slot, values)| ((pid, slot), values))
            .collect();
        let mut count = 0usize;
        for ((pid, slot), values) in &victims {
            // FK action on referencing children must run BEFORE we tombstone
            // this row — RESTRICT needs to be able to abort the whole DELETE
            // before any heap state has changed for this victim.
            cascade_fk_on_parent_delete(
                bpm, lm, wal, tm, catalog, stmt.table_id, values, tx,
            )?;
            lm.lock(tx.id(), (*pid, *slot), LockMode::Exclusive)
                .map_err(|e| anyhow::anyhow!("X-lock on {:?}: {e}", (*pid, *slot)))?;
            tx.add_lock((*pid, *slot));

            // Post-lock recheck — same Read-Committed pattern as UPDATE.
            let current_xmax = {
                let g = bpm.fetch_page(*pid)?;
                let p = g.read();
                let raw = match p.get_tuple(*slot) {
                    Some(r) => r,
                    None => continue,
                };
                let (_, x, _) = deserialize_tuple_mvcc(raw, &schema)?;
                x
            };
            if current_xmax != INVALID_TXN_ID && current_xmax != tx.id() {
                match tm.status(current_xmax) {
                    TxnStatus::Committed => continue 'restart,
                    TxnStatus::Aborted => {}
                    TxnStatus::InProgress => bail!(
                        "could not serialize access due to concurrent update [SQLSTATE 40001]"
                    ),
                }
            }
            {
                let g = bpm.fetch_page(*pid)?;
                let mut p = g.write();
                p.set_tuple_xmax(*slot, tx.id())?;
                let lsn = log_record(
                    wal,
                    tx,
                    WalRecordType::Delete {
                        rid: (*pid, *slot),
                        xmax: tx.id(),
                    },
                )?;
                p.set_page_lsn(lsn);
            }
            count += 1;
        }
        return Ok(count);
    }
    bail!(
        "could not serialize access — DELETE retried {MAX_RETRIES} times against concurrent updates [SQLSTATE 40001]"
    );
}

/// Walk every FK that *targets* `parent_table_id` and apply its
/// `on_delete` action to rows in the child table that reference the
/// row about to be deleted. RESTRICT/NO ACTION raise SQLSTATE 23503;
/// CASCADE recursively deletes; SET NULL writes NULL into the child
/// FK column.
fn cascade_fk_on_parent_delete(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    parent_table_id: usize,
    parent_row: &[Value],
    tx: &mut Transaction,
) -> Result<()> {
    let parent = catalog
        .table_by_id(parent_table_id)?
        .ok_or_else(|| anyhow::anyhow!("parent table {parent_table_id} missing"))?;
    let snapshot = tx
        .snapshot()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no snapshot for tx"))?;

    // Collect FK constraints (across all tables) that point at this parent.
    let all = catalog.all_constraints()?;
    for c in all {
        if !matches!(c.kind, crate::catalog::ConstraintKind::ForeignKey) {
            continue;
        }
        let fk = match crate::catalog::ForeignKeyDef::decode(&c.definition) {
            Some(f) => f,
            None => continue,
        };
        if fk.ref_table != parent.name {
            continue;
        }
        let parent_col_idx = parent
            .columns
            .iter()
            .position(|col| col.name == fk.ref_column)
            .ok_or_else(|| anyhow::anyhow!("FK ref column missing"))?;
        let target_value = &parent_row[parent_col_idx];
        if matches!(target_value, Value::Null) {
            continue;
        }
        let child_table_id = c.table_id;
        let child = catalog
            .table_by_id(child_table_id)?
            .ok_or_else(|| anyhow::anyhow!("child table {child_table_id} missing"))?;
        let child_col_idx = child
            .columns
            .iter()
            .position(|col| col.name == fk.child_column)
            .ok_or_else(|| anyhow::anyhow!("FK child column missing"))?;

        // Find all visible child rows that reference target_value.
        let (_schema, rows) =
            visible_rows(bpm, catalog, child_table_id, &snapshot, tm)?;
        let matches_rows: Vec<(Rid, Vec<Value>)> = rows
            .into_iter()
            .filter(|(_, _, v)| values_equal(&v[child_col_idx], target_value))
            .map(|(p, s, v)| ((p, s), v))
            .collect();
        if matches_rows.is_empty() {
            continue;
        }

        match fk.on_delete.as_str() {
            "RESTRICT" | "NO_ACTION" => {
                bail!(
                    "update or delete on \"{}\" violates foreign key constraint \"{}\" on \"{}\" [SQLSTATE 23503]",
                    parent.name,
                    c.name,
                    child.name
                );
            }
            "CASCADE" => {
                for ((pid, slot), values) in &matches_rows {
                    // Recurse so cascading children are also handled.
                    cascade_fk_on_parent_delete(
                        bpm, lm, wal, tm, catalog, child_table_id, values, tx,
                    )?;
                    lm.lock(tx.id(), (*pid, *slot), LockMode::Exclusive).map_err(
                        |e| anyhow::anyhow!("X-lock on {:?}: {e}", (*pid, *slot)),
                    )?;
                    tx.add_lock((*pid, *slot));
                    let g = bpm.fetch_page(*pid)?;
                    let mut p = g.write();
                    p.set_tuple_xmax(*slot, tx.id())?;
                    let lsn = log_record(
                        wal,
                        tx,
                        WalRecordType::Delete {
                            rid: (*pid, *slot),
                            xmax: tx.id(),
                        },
                    )?;
                    p.set_page_lsn(lsn);
                }
            }
            "SET_NULL" => {
                for ((pid, slot), old_values) in matches_rows {
                    let mut new_values = old_values.clone();
                    new_values[child_col_idx] = Value::Null;
                    enforce_check_constraints(
                        catalog, child_table_id, &child.name, &new_values,
                    )?;
                    lm.lock(tx.id(), (pid, slot), LockMode::Exclusive).map_err(
                        |e| anyhow::anyhow!("X-lock on {:?}: {e}", (pid, slot)),
                    )?;
                    tx.add_lock((pid, slot));
                    {
                        let g = bpm.fetch_page(pid)?;
                        let mut p = g.write();
                        p.set_tuple_xmax(slot, tx.id())?;
                        let lsn = log_record(
                            wal,
                            tx,
                            WalRecordType::Delete {
                                rid: (pid, slot),
                                xmax: tx.id(),
                            },
                        )?;
                        p.set_page_lsn(lsn);
                    }
                    let new_bytes =
                        serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &new_values);
                    let (new_rid, _lsn) =
                        insert_bytes(bpm, wal, tx, child.first_page_id, &new_bytes)?;
                    lm.lock(tx.id(), new_rid, LockMode::Exclusive).map_err(
                        |e| anyhow::anyhow!("X-lock on {new_rid:?}: {e}"),
                    )?;
                    tx.add_lock(new_rid);
                    let tm_arc = Arc::clone(tx.tm());
                    index_insert_for_row_with_lm(
                        bpm,
                        Some(lm),
                        wal,
                        Some(&tm_arc),
                        catalog,
                        tx,
                        child_table_id,
                        &new_values,
                        new_rid,
                    )?;
                }
            }
            other => bail!("unknown FK action '{other}'"),
        }
    }
    Ok(())
}

fn perform_update(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    stmt: &AnalyzedUpdateStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    let table = catalog
        .table_by_id(stmt.table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {} not in catalog", stmt.table_id))?;
    let schema = table.to_schema();
    let mut count = 0usize;

    // PG-style "Read Committed" loop: take a fresh statement snapshot,
    // find matching rows, lock+update each. If a lock arrives at a row
    // that's already been updated by a concurrent committed txn, restart
    // the whole statement against the latest snapshot. Capped to a few
    // retries so a stuck conflict surfaces as 40001 instead of looping.
    const MAX_RETRIES: u32 = 8;
    'restart: for _attempt in 0..MAX_RETRIES {
        // Fresh snapshot: pulls in every txn that has committed since the
        // last attempt — that's what makes a re-evaluated WHERE see the
        // updated row instead of the gone one.
        let snapshot = tx.tm().snapshot(tx.id());
        tx.set_snapshot(snapshot.clone());
        let (_schema, rows) = matching_rows(
            bpm, catalog, stmt.table_id, &stmt.where_clause, &snapshot, tm,
        )?;
        count = 0;
        for (pid, slot, values) in rows {
            // Acquire the row X-lock before reading the latest header,
            // so any other writer either left or is now sequenced behind us.
            lm.lock(tx.id(), (pid, slot), LockMode::Exclusive)
                .map_err(|e| anyhow::anyhow!("X-lock on {:?}: {e}", (pid, slot)))?;
            tx.add_lock((pid, slot));

            // Post-lock recheck: if the row was updated/deleted by another
            // committed txn while we were finding it, restart with a fresh
            // snapshot. A self-set xmax (re-running our own UPDATE) just
            // means we've already done the work for this row.
            let (current_xmax, _row_alive) = {
                let g = bpm.fetch_page(pid)?;
                let p = g.read();
                let raw = match p.get_tuple(slot) {
                    Some(r) => r,
                    None => continue, // tombstoned by VACUUM mid-flight
                };
                let (_, x, _) = deserialize_tuple_mvcc(raw, &schema)?;
                (x, ())
            };
            if current_xmax != INVALID_TXN_ID && current_xmax != tx.id() {
                match tm.status(current_xmax) {
                    TxnStatus::Committed => {
                        // Row was replaced under our feet — let matching_rows
                        // pick up the new version on the next pass.
                        continue 'restart;
                    }
                    TxnStatus::Aborted => {
                        // Their delete didn't take effect; row is still ours.
                    }
                    TxnStatus::InProgress => {
                        // We hold the X-lock, so no one else should be in
                        // the middle of writing this row. Defensive abort.
                        bail!(
                            "could not serialize access due to concurrent update [SQLSTATE 40001]"
                        );
                    }
                }
            }

            // Now safe to compute the new tuple and write it.
            let t = Tuple::new(values);
            let mut new_values = t.values.clone();
            for a in &stmt.assignments {
                let v = evaluate_expr(&a.value, &t)?;
                let target = table.columns[a.column_index].data_type;
                new_values[a.column_index] = coerce_for_storage(v, target)?;
            }
            enforce_check_constraints(catalog, stmt.table_id, &stmt.table_name, &new_values)?;
            let tm_for_fk = Arc::clone(tx.tm());
            enforce_fk_on_child_write(
                bpm, &tm_for_fk, catalog, stmt.table_id, &new_values, tx,
            )?;

            // HOT-update gate: if no indexed column changes value, the
            // existing index entry already points (transitively) to the
            // right row. We skip the per-index re-insert and instead
            // record a t_ctid forward pointer on the old tuple so
            // IndexScan / matching_rows can step from a stale RID to
            // the live successor.
            let touched: std::collections::HashSet<usize> =
                stmt.assignments.iter().map(|a| a.column_index).collect();
            let any_index_col_touched = catalog
                .indexes_for_table(stmt.table_id)?
                .iter()
                .any(|i| touched.contains(&i.column_index));
            let do_hot = !any_index_col_touched;

            // Tombstone old + WAL Delete (always — same shape for HOT and
            // non-HOT).
            {
                let g = bpm.fetch_page(pid)?;
                let mut p = g.write();
                p.set_tuple_xmax(slot, tx.id())?;
                let lsn = log_record(
                    wal,
                    tx,
                    WalRecordType::Delete {
                        rid: (pid, slot),
                        xmax: tx.id(),
                    },
                )?;
                p.set_page_lsn(lsn);
            }
            let new_bytes = serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &new_values);
            let (new_rid, _) = insert_bytes(bpm, wal, tx, table.first_page_id, &new_bytes)?;
            lm.lock(tx.id(), new_rid, LockMode::Exclusive)
                .map_err(|e| anyhow::anyhow!("X-lock on {new_rid:?}: {e}"))?;
            tx.add_lock(new_rid);

            if do_hot {
                // Stamp the chain pointer. Readers now find the new row
                // by following old.t_ctid.
                let g = bpm.fetch_page(pid)?;
                g.write().set_tuple_ctid(slot, new_rid)?;
            } else {
                let tm_arc = Arc::clone(tx.tm());
                // Same race-handling as before: a 23505 from a concurrent
                // committed UPDATE feeds back into the outer restart.
                match index_insert_for_row_with_lm(
                    bpm,
                    Some(lm),
                    wal,
                    Some(&tm_arc),
                    catalog,
                    tx,
                    stmt.table_id,
                    &new_values,
                    new_rid,
                ) {
                    Ok(()) => {}
                    Err(e) if e.to_string().contains("SQLSTATE 23505") => {
                        continue 'restart;
                    }
                    Err(e) => return Err(e),
                }
            }
            count += 1;
        }
        return Ok(count);
    }
    bail!(
        "could not serialize access — UPDATE retried {MAX_RETRIES} times against concurrent updates [SQLSTATE 40001]"
    );
}

/// Walk the table's page chain to its tail, then try to insert. If the
/// tail is full, allocate a new page and link it into the chain.
fn insert_bytes(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    first_page: PageId,
    bytes: &[u8],
) -> Result<(Rid, Lsn)> {
    let (rid, lsn, _tail) = insert_bytes_hinted(bpm, wal, tx, first_page, bytes, None)?;
    Ok((rid, lsn))
}

/// Same as `insert_bytes` but accepts a "tail hint" — a page id believed to
/// be at or near the end of the chain. Walking from the hint avoids the
/// O(n) scan from `first_page` that turns bulk inserts (COPY, multi-row
/// INSERT) into O(rows²). Returns the page that actually received the
/// tuple so callers can thread the hint forward.
fn insert_bytes_hinted(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    first_page: PageId,
    bytes: &[u8],
    hint: Option<PageId>,
) -> Result<(Rid, Lsn, PageId)> {
    let start = hint.unwrap_or(first_page);
    let mut last = start;
    loop {
        let g = bpm.fetch_page(last)?;
        let next = g.read().next_page_id();
        if next == crate::page::NO_NEXT_PAGE {
            break;
        }
        last = next;
    }

    {
        let g = bpm.fetch_page(last)?;
        let mut p = g.write();
        if let Ok(slot) = p.insert(bytes) {
            let rid = (last, slot);
            let lsn = log_record(
                wal,
                tx,
                WalRecordType::Insert {
                    rid,
                    data: bytes.to_vec(),
                },
            )?;
            p.set_page_lsn(lsn);
            return Ok((rid, lsn, last));
        }
    }

    let new_g = bpm.new_page()?;
    let new_pid = new_g.page_id();
    {
        let mut p = new_g.write();
        let slot = p.insert(bytes).map_err(|e| {
            anyhow::anyhow!("tuple does not fit on a fresh page: {e}")
        })?;
        let rid = (new_pid, slot);
        let lsn = log_record(
            wal,
            tx,
            WalRecordType::Insert {
                rid,
                data: bytes.to_vec(),
            },
        )?;
        p.set_page_lsn(lsn);
        drop(p);
        let prev_g = bpm.fetch_page(last)?;
        prev_g.write().set_next_page_id(new_pid);
        return Ok((rid, lsn, new_pid));
    }
}

/// Roll back a transaction. With MVCC the page state doesn't need to be
/// physically reverted — visibility checks already exclude rows whose
/// xmin/xmax came from an aborted txn. We just emit the Abort record so
/// recovery sees the same outcome and mark the txn aborted in the TM
/// (which `set_inactive` does).
///
/// CLRs from day12 are no longer written: their purpose was to make
/// physical undo crash-safe, and we no longer have physical undo to
/// protect.
pub fn rollback(_bpm: &BufferPool, wal: &WalManager, tx: &mut Transaction) -> Result<()> {
    let _ = tx.drain_log(); // legacy field; MVCC doesn't fill it.
    log_record(wal, tx, WalRecordType::Abort)?;
    wal.flush()?;
    tx.set_inactive();
    Ok(())
}

// -- ExecutionEngine ---------------------------------------------------------

pub fn execute(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    stmt: &AnalyzedStatement,
    tx: &mut Transaction,
) -> Result<Output> {
    // Auto-commit boundary: a fresh tx_id per implicit-tx statement so the
    // LockManager and WAL can distinguish concurrent auto-commit statements.
    let was_inactive_at_start = !tx.is_active();
    if was_inactive_at_start {
        tx.refresh_autocommit();
    }

    // Replace `now()` / `current_timestamp` placeholders with the txn's
    // start timestamp. This makes the value stable within a single tx
    // and lets every downstream operator see only Literal nodes.
    let stmt_owned = substitute_now_in_statement(stmt, tx.start_ts());
    let stmt = &stmt_owned;

    // For DML running under auto-commit, bracket the records with Begin/Commit
    // (or Abort on error) so the WAL is self-describing for recovery.
    let needs_dml_brackets = was_inactive_at_start
        && matches!(
            stmt,
            AnalyzedStatement::Insert(_)
                | AnalyzedStatement::Delete(_)
                | AnalyzedStatement::Update(_)
        );
    if needs_dml_brackets {
        log_record(wal, tx, WalRecordType::Begin)?;
    }

    let result: Result<Output> = (|| match stmt {
        AnalyzedStatement::Select(s) => {
            // Pre-evaluate SequenceCalls in select_items / where / etc.
            // Only safe when the source is OneRowRelation (no FROM) — for
            // FROM-based SELECTs, sequence calls in projections would need
            // per-row evaluation which isn't wired in Phase 1-7.
            let mut s_owned = s.clone();
            if matches!(s_owned.from, AnalyzedFrom::Empty) {
                for it in &mut s_owned.select_items {
                    evaluate_sequence_calls_in_expr(&mut it.expr, bpm, wal, tx)?;
                }
            }
            let s = &s_owned;
            let snapshot = tx
                .snapshot()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no snapshot for tx"))?;
            let rows = {
                let mut exec = build_select_pipeline(bpm, catalog, s, snapshot, tm)?;
                exec.open()?;
                let mut rows = Vec::new();
                while let Some(t) = exec.next()? {
                    rows.push(t);
                }
                rows
            };
            Ok(Output::Rows(rows))
        }
        AnalyzedStatement::Insert(s) => Ok(Output::Affected(perform_insert(
            bpm, lm, wal, catalog, s, tx,
        )?)),
        AnalyzedStatement::Delete(s) => Ok(Output::Affected(perform_delete(
            bpm, lm, wal, tm, catalog, s, tx,
        )?)),
        AnalyzedStatement::Update(s) => Ok(Output::Affected(perform_update(
            bpm, lm, wal, tm, catalog, s, tx,
        )?)),
        AnalyzedStatement::Begin => {
            if tx.is_active() {
                bail!("there is already a transaction in progress");
            }
            tx.begin();
            log_record(wal, tx, WalRecordType::Begin)?;
            Ok(Output::Begin)
        }
        AnalyzedStatement::Commit => {
            if !tx.is_active() {
                bail!("there is no transaction in progress");
            }
            log_record(wal, tx, WalRecordType::Commit)?;
            wal.flush()?;
            let held = tx.take_held_locks();
            lm.unlock_all(tx.id(), &held);
            tx.commit();
            Ok(Output::Commit)
        }
        AnalyzedStatement::Rollback => {
            if !tx.is_active() {
                bail!("there is no transaction in progress");
            }
            // rollback() writes CLRs and the Abort record itself.
            rollback(bpm, wal, tx)?;
            let held = tx.take_held_locks();
            lm.unlock_all(tx.id(), &held);
            Ok(Output::Rollback)
        }
        AnalyzedStatement::CreateTable(s) => {
            perform_create_table(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::CreateIndex(s) => {
            perform_create_index(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::AlterTableAddIndex(s) => {
            // ALTER TABLE ... ADD PRIMARY KEY / UNIQUE — currently the same
            // as CREATE INDEX (no uniqueness enforcement until Phase 4).
            perform_create_index(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::DropTable(s) => {
            perform_drop_table(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::DropIndex(s) => {
            perform_drop_index(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::Truncate(s) => Ok(Output::Affected(perform_truncate(
            bpm, wal, catalog, s, tx,
        )?)),
        AnalyzedStatement::CreateSequence(s) => {
            perform_create_sequence(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::DropSequence(s) => {
            perform_drop_sequence(bpm, wal, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::Vacuum(s) => {
            perform_vacuum(bpm, wal, tm, catalog, s, tx)?;
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::AnalyzeNoop => {
            // Phase 9 will collect real statistics; for now it's parse-and-accept.
            Ok(Output::Affected(0))
        }
        AnalyzedStatement::Copy(_) => {
            // COPY is driven from the connection layer (instance.rs) so it
            // can shuttle CopyData messages off the wire — the executor
            // doesn't see the data stream itself.
            bail!("COPY must be handled outside the executor")
        }
        AnalyzedStatement::Checkpoint => {
            // Handled at the connection layer (instance.rs) — has access to
            // the global ATT and DPT, which the executor doesn't.
            bail!("CHECKPOINT must be handled outside the executor")
        }
    })();

    // Bracket the auto-commit DML records with Commit (or Abort on error).
    if needs_dml_brackets {
        let bracket = if result.is_ok() {
            WalRecordType::Commit
        } else {
            WalRecordType::Abort
        };
        let _ = log_record(wal, tx, bracket);
        let _ = wal.flush();
    }

    // Auto-commit boundary: release locks AND finalize TM status so
    // visibility checks for future txns see the right state. (Without this,
    // Transaction's Drop would later abort an implicit-tx that succeeded.)
    if was_inactive_at_start && !tx.is_active() {
        let held = tx.take_held_locks();
        lm.unlock_all(tx.id(), &held);
        if result.is_ok() {
            tx.tm().commit(tx.id());
        } else {
            tx.tm().abort(tx.id());
        }
    }

    result
}

fn build_select_pipeline<'a>(
    bpm: &'a BufferPool,
    catalog: &'a Catalog,
    stmt: &AnalyzedSelectStatement,
    snapshot: Snapshot,
    tm: &'a TransactionManager,
) -> Result<Box<dyn Executor + 'a>> {
    // Try to swap a SeqScan for an IndexScan when:
    //   - FROM is a single base table (no joins)
    //   - WHERE has an `indexed_col = literal` or `indexed_col BETWEEN x AND y`
    //     pattern, after BETWEEN's parse-time desugaring to >= AND <=.
    // The Filter on top stays in place: it's a no-op for the index-driven
    // predicate but still applies any other AND-clauses, and is harmless if
    // duplicated.
    let from_exec = if let AnalyzedFrom::Table { rte_index } = &stmt.from {
        let rte = &stmt.range_table[*rte_index];
        let TableSource::BaseTable { table_id, .. } = &rte.source;
        let table_id = *table_id;
        if let Some((idx, range, key_type)) =
            find_indexable_predicate(&stmt.where_clause, table_id, catalog)?
        {
            Box::new(IndexScan::new(
                bpm,
                catalog,
                table_id,
                idx.root_page_id,
                range,
                key_type,
                snapshot.clone(),
                tm,
            )?) as Box<dyn Executor + 'a>
        } else {
            build_from_pipeline(bpm, catalog, &stmt.from, &stmt.range_table, &snapshot, tm)?
        }
    } else {
        build_from_pipeline(bpm, catalog, &stmt.from, &stmt.range_table, &snapshot, tm)?
    };
    let filtered: Box<dyn Executor + 'a> = match &stmt.where_clause {
        Some(p) => Box::new(Filter::new(from_exec, p.clone())),
        None => from_exec,
    };
    // HashAggregate goes in if and only if the analyzer decided aggregation
    // was needed. After it, expressions reference post-agg positions.
    let post_agg: Box<dyn Executor + 'a> = match &stmt.aggregation {
        Some(agg) => Box::new(HashAggregate::new(
            filtered,
            agg.group_keys.clone(),
            agg.aggregates.clone(),
        )),
        None => filtered,
    };
    // HAVING is a post-agg filter.
    let post_having: Box<dyn Executor + 'a> = match &stmt.having {
        Some(p) => Box::new(Filter::new(post_agg, p.clone())),
        None => post_agg,
    };
    // Sort runs on pre-projection tuples so ORDER BY can reference columns
    // not in the SELECT list. Skipped when there are no sort keys.
    let sorted: Box<dyn Executor + 'a> = if stmt.order_by.is_empty() {
        post_having
    } else {
        Box::new(Sort::new(post_having, stmt.order_by.clone()))
    };
    // LIMIT after Sort so the cap applies to the ordered output.
    let limited: Box<dyn Executor + 'a> = match stmt.limit {
        Some(n) => Box::new(Limit::new(sorted, n)),
        None => sorted,
    };
    let exprs: Vec<AnalyzedExpr> = stmt
        .select_items
        .iter()
        .map(|i| i.expr.clone())
        .collect();
    Ok(Box::new(Project::new(limited, exprs)))
}

/// Look at the WHERE clause and decide whether an IndexScan is applicable.
/// The first matching pattern wins; any remaining predicate is left to the
/// Filter on top of the scan.
fn find_indexable_predicate(
    where_clause: &Option<AnalyzedExpr>,
    table_id: usize,
    catalog: &Catalog,
) -> Result<Option<(crate::catalog::IndexDef, IndexRange, DataType)>> {
    let Some(expr) = where_clause else {
        return Ok(None);
    };
    let table = catalog
        .table_by_id(table_id)?
        .ok_or_else(|| anyhow::anyhow!("table {table_id} missing"))?;
    let indexes = catalog.indexes_for_table(table_id)?;
    if indexes.is_empty() {
        return Ok(None);
    }

    // Try equality first.
    if let Some((col_idx, lit)) = match_eq_col_literal(expr) {
        if let Some(idx) = indexes.iter().find(|i| i.column_index == col_idx) {
            let dt = table.columns[col_idx].data_type;
            let key = crate::btree::encode_key(&literal_to_value(&lit));
            return Ok(Some((idx.clone(), IndexRange::Eq { key }, dt)));
        }
    }

    // Then closed range `col >= L AND col <= H` (the desugared form of BETWEEN).
    if let Some((col_idx, low_lit, high_lit)) = match_between(expr) {
        if let Some(idx) = indexes.iter().find(|i| i.column_index == col_idx) {
            let dt = table.columns[col_idx].data_type;
            let low = crate::btree::encode_key(&literal_to_value(&low_lit));
            let high = crate::btree::encode_key(&literal_to_value(&high_lit));
            return Ok(Some((idx.clone(), IndexRange::Between { low, high }, dt)));
        }
    }

    Ok(None)
}

/// Match `col = lit` or `lit = col` and return `(column_index, lit)`.
fn match_eq_col_literal(expr: &AnalyzedExpr) -> Option<(usize, AnalyzedLiteral)> {
    let AnalyzedExpr::BinaryOp { left, op, right, .. } = expr else {
        return None;
    };
    if !matches!(op, crate::ast::BinaryOperator::Eq) {
        return None;
    }
    if let (AnalyzedExpr::ColumnRef(c), AnalyzedExpr::Literal(l)) = (left.as_ref(), right.as_ref()) {
        return Some((c.column_index, l.clone()));
    }
    if let (AnalyzedExpr::Literal(l), AnalyzedExpr::ColumnRef(c)) = (left.as_ref(), right.as_ref()) {
        return Some((c.column_index, l.clone()));
    }
    None
}

/// Match `col >= L AND col <= H` (or with operands swapped on either side)
/// and return `(column_index, low, high)`.
fn match_between(expr: &AnalyzedExpr) -> Option<(usize, AnalyzedLiteral, AnalyzedLiteral)> {
    let AnalyzedExpr::BinaryOp { left, op, right, .. } = expr else {
        return None;
    };
    if !matches!(op, crate::ast::BinaryOperator::And) {
        return None;
    }
    let (lcol, lop, llit) = match_cmp_col_literal(left)?;
    let (rcol, rop, rlit) = match_cmp_col_literal(right)?;
    if lcol != rcol {
        return None;
    }
    use crate::ast::BinaryOperator::*;
    match (lop, rop) {
        (Ge, Le) => Some((lcol, llit, rlit)),
        (Le, Ge) => Some((lcol, rlit, llit)),
        _ => None,
    }
}

fn match_cmp_col_literal(
    expr: &AnalyzedExpr,
) -> Option<(usize, crate::ast::BinaryOperator, AnalyzedLiteral)> {
    let AnalyzedExpr::BinaryOp { left, op, right, .. } = expr else {
        return None;
    };
    use crate::ast::BinaryOperator::*;
    if !matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
        return None;
    }
    if let (AnalyzedExpr::ColumnRef(c), AnalyzedExpr::Literal(l)) = (left.as_ref(), right.as_ref()) {
        return Some((c.column_index, *op, l.clone()));
    }
    if let (AnalyzedExpr::Literal(l), AnalyzedExpr::ColumnRef(c)) = (left.as_ref(), right.as_ref()) {
        // Swap: `lit OP col` is equivalent to `col SWAP_OP lit`.
        let swapped = match op {
            Lt => Gt,
            Le => Ge,
            Gt => Lt,
            Ge => Le,
            other => *other,
        };
        return Some((c.column_index, swapped, l.clone()));
    }
    None
}

fn build_from_pipeline<'a>(
    bpm: &'a BufferPool,
    catalog: &'a Catalog,
    from: &AnalyzedFrom,
    range_table: &[crate::analyzer::RangeTableEntry],
    snapshot: &Snapshot,
    tm: &'a TransactionManager,
) -> Result<Box<dyn Executor + 'a>> {
    match from {
        AnalyzedFrom::Empty => Ok(Box::new(OneRowRelation::new())),
        AnalyzedFrom::Table { rte_index } => {
            let rte = &range_table[*rte_index];
            let TableSource::BaseTable { table_id, .. } = &rte.source;
            Ok(Box::new(SeqScan::new(
                bpm,
                catalog,
                *table_id,
                snapshot.clone(),
                tm,
            )?))
        }
        AnalyzedFrom::Join {
            left,
            right_rte_index,
            join_type,
            on,
        } => {
            let outer = build_from_pipeline(bpm, catalog, left, range_table, snapshot, tm)?;
            let rte = &range_table[*right_rte_index];
            let TableSource::BaseTable { table_id, .. } = &rte.source;
            let inner: Box<dyn Executor + 'a> = Box::new(SeqScan::new(
                bpm,
                catalog,
                *table_id,
                snapshot.clone(),
                tm,
            )?);
            let inner_width = rte.output_columns.len();
            Ok(Box::new(NestedLoopJoin::new(
                outer,
                inner,
                on.clone(),
                *join_type,
                inner_width,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::analyze;
    use crate::disk::DiskManager;
    use crate::parser::parse;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-exec-{name}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    /// Tests share a single ephemeral WAL file path per executor — recovery
    /// isn't yet implemented so contents don't matter, but `BufferPool::new`
    /// requires a real file.
    fn ephemeral_wal() -> std::sync::Arc<crate::wal::WalManager> {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-exec-wal-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&p);
        std::sync::Arc::new(crate::wal::WalManager::open(&p).unwrap())
    }

    /// Build a fresh DB env with the `users(id INT, name VARCHAR)` table
    /// already created, ready for INSERT/SELECT tests. Replaces the old
    /// hardcoded-catalog setup post-day16.
    fn setup_users() -> (
        Catalog,
        BufferPool,
        std::sync::Arc<crate::wal::WalManager>,
        std::sync::Arc<crate::transaction_manager::TransactionManager>,
    ) {
        let path = temp_path("setup");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(
            crate::wal::WalManager::open(&path.with_extension("wal")).unwrap(),
        );
        let bpm = BufferPool::new(disk, 8, wal.clone());
        let clog = std::sync::Arc::new(crate::clog::Clog::in_memory());
        let tm = std::sync::Arc::new(
            crate::transaction_manager::TransactionManager::new(clog),
        );
        crate::bootstrap::bootstrap(&bpm, &tm).unwrap();
        let cat = Catalog::new(bpm.clone(), std::sync::Arc::clone(&tm));
        // Create users(id INT, name VARCHAR).
        {
            let mut tx = Transaction::new(std::sync::Arc::clone(&tm));
            let lm = LockManager::new();
            let stmt = parse("CREATE TABLE users (id INT, name VARCHAR)").unwrap();
            let analyzed = analyze(&cat, &stmt).unwrap();
            execute(&bpm, &lm, &wal, &tm, &cat, &analyzed, &mut tx).unwrap();
        }
        (cat, bpm, wal, tm)
    }

    fn run(
        sql: &str,
        cat: &Catalog,
        bpm: &BufferPool,
        wal: &WalManager,
        tm: &std::sync::Arc<crate::transaction_manager::TransactionManager>,
    ) -> Output {
        let mut tx = Transaction::new(std::sync::Arc::clone(tm));
        let lm = LockManager::new();
        run_full(sql, cat, bpm, &lm, wal, &mut tx)
    }

    /// Like `run()` but uses the catalog's TM (so visibility checks work).
    fn run_with_tm(
        sql: &str,
        cat: &Catalog,
        bpm: &BufferPool,
        wal: &WalManager,
        tm: &std::sync::Arc<crate::transaction_manager::TransactionManager>,
    ) -> Output {
        let mut tx = Transaction::new(std::sync::Arc::clone(tm));
        let lm = LockManager::new();
        run_full(sql, cat, bpm, &lm, wal, &mut tx)
    }

    fn run_tx(
        sql: &str,
        cat: &Catalog,
        bpm: &BufferPool,
        wal: &WalManager,
        tx: &mut Transaction,
    ) -> Output {
        let lm = LockManager::new();
        run_full(sql, cat, bpm, &lm, wal, tx)
    }

    fn run_full(
        sql: &str,
        cat: &Catalog,
        bpm: &BufferPool,
        lm: &LockManager,
        wal: &WalManager,
        tx: &mut Transaction,
    ) -> Output {
        let tm = std::sync::Arc::clone(tx.tm());
        let stmt = parse(sql).unwrap();
        let analyzed = analyze(cat, &stmt).unwrap();
        execute(bpm, lm, wal, &tm, cat, &analyzed, tx).unwrap()
    }
    #[test]
    fn insert_then_select_star() {
        let (cat, bpm, wal, tm) = setup_users();

        for sql in [
            "INSERT INTO users VALUES (1, 'Alice')",
            "INSERT INTO users VALUES (2, 'Bob')",
            "INSERT INTO users VALUES (3, NULL)",
        ] {
            assert!(matches!(run(sql, &cat, &bpm, &wal, &tm), Output::Affected(1)));
        }
        let Output::Rows(rows) = run("SELECT * FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[2].values[1], Value::Null);
    }
    #[test]
    fn where_filters_rows() {
        let (cat, bpm, wal, tm) = setup_users();
        for i in 1..=5 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm,
                &wal,
                &tm,
            );
        }
        let Output::Rows(rows) = run("SELECT id FROM users WHERE id > 2", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], Value::Int(3));
    }
    #[test]
    fn projection_evaluates_arithmetic() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (10, 'a')", &cat, &bpm, &wal, &tm);
        let Output::Rows(rows) = run("SELECT id + 1 FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Int(11));
    }
    #[test]
    fn null_predicate_excludes_row() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, NULL)", &cat, &bpm, &wal, &tm);
        // name = 'Alice' on the NULL row evaluates to NULL → row excluded.
        let Output::Rows(rows) = run(
            "SELECT id FROM users WHERE name = 'Alice'",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Int(1));
    }
    #[test]
    fn delete_with_predicate() {
        let (cat, bpm, wal, tm) = setup_users();
        for i in 1..=4 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm,
                &wal,
                &tm,
            );
        }
        assert!(matches!(
            run("DELETE FROM users WHERE id > 2", &cat, &bpm, &wal, &tm),
            Output::Affected(2)
        ));
        let Output::Rows(rows) = run("SELECT id FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[1].values[0], Value::Int(2));
    }
    #[test]
    fn delete_all_rows() {
        let (cat, bpm, wal, tm) = setup_users();
        for i in 1..=3 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm,
                &wal,
                &tm,
            );
        }
        assert!(matches!(
            run("DELETE FROM users", &cat, &bpm, &wal, &tm),
            Output::Affected(3)
        ));
        let Output::Rows(rows) = run("SELECT * FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert!(rows.is_empty());
    }
    #[test]
    fn update_changes_matching_rows() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, 'Bob')", &cat, &bpm, &wal, &tm);
        assert!(matches!(
            run(
                "UPDATE users SET name = 'A2' WHERE id = 1",
                &cat,
                &bpm,
                &wal,
                &tm,
            ),
            Output::Affected(1)
        ));
        let Output::Rows(rows) = run("SELECT id, name FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        // Order may shift because UPDATE = delete + insert; check by id.
        let mut seen = std::collections::HashMap::new();
        for r in &rows {
            let Value::Int(id) = r.values[0] else { panic!() };
            let Value::Varchar(name) = &r.values[1] else { panic!() };
            seen.insert(id, name.clone());
        }
        assert_eq!(seen[&1], "A2");
        assert_eq!(seen[&2], "Bob");
    }
    #[test]
    fn update_does_not_re_match_inserted_row() {
        // Snapshot-then-apply guarantees we don't see our own writes.
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &tm);
        // SET name = 'a' WHERE name = 'a' affects exactly one row, not infinite.
        assert!(matches!(
            run(
                "UPDATE users SET name = 'a' WHERE name = 'a'",
                &cat,
                &bpm,
                &wal,
                &tm,
            ),
            Output::Affected(1)
        ));
        let Output::Rows(rows) = run("SELECT id FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
    }
    #[test]
    fn null_propagates_in_arithmetic() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, NULL)", &cat, &bpm, &wal, &tm);
        // SELECT name (which is NULL) projects through; arithmetic on NULL would
        // also produce NULL — exercised via id (non-null) for the all-OK row.
        let Output::Rows(rows) = run("SELECT name FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Null);
    }
    #[test]
    fn rollback_undoes_insert() {
        let (cat, bpm, wal, tm) = setup_users();
        let mut tx = Transaction::new(std::sync::Arc::clone(&tm));

        run_tx("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &mut tx);
        run_tx("BEGIN", &cat, &bpm, &wal, &mut tx);
        run_tx("INSERT INTO users VALUES (2, 'b')", &cat, &bpm, &wal, &mut tx);
        // Inside tx: 2 rows visible.
        let Output::Rows(rows) = run_tx("SELECT id FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        run_tx("ROLLBACK", &cat, &bpm, &wal, &mut tx);
        // After rollback: just 1.
        let Output::Rows(rows) = run_tx("SELECT id FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
    }
    #[test]
    fn rollback_undoes_delete() {
        let (cat, bpm, wal, tm) = setup_users();
        let mut tx = Transaction::new(std::sync::Arc::clone(&tm));

        run_tx("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &mut tx);
        run_tx("INSERT INTO users VALUES (2, 'b')", &cat, &bpm, &wal, &mut tx);
        run_tx("BEGIN", &cat, &bpm, &wal, &mut tx);
        run_tx("DELETE FROM users WHERE id = 1", &cat, &bpm, &wal, &mut tx);
        run_tx("ROLLBACK", &cat, &bpm, &wal, &mut tx);
        let Output::Rows(rows) = run_tx("SELECT id FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
    }
    #[test]
    fn rollback_undoes_update() {
        // UPDATE = delete + insert, so the undo log holds two entries per row.
        let (cat, bpm, wal, tm) = setup_users();
        let mut tx = Transaction::new(std::sync::Arc::clone(&tm));

        run_tx("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal, &mut tx);
        run_tx("BEGIN", &cat, &bpm, &wal, &mut tx);
        run_tx(
            "UPDATE users SET name = 'A2' WHERE id = 1",
            &cat,
            &bpm,
            &wal,
            &mut tx,
        );
        // Mid-tx the new value is visible.
        let Output::Rows(rows) = run_tx("SELECT name FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Varchar("A2".into()));
        run_tx("ROLLBACK", &cat, &bpm, &wal, &mut tx);
        let Output::Rows(rows) = run_tx("SELECT name FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Varchar("Alice".into()));
    }
    #[test]
    fn commit_persists_changes() {
        let (cat, bpm, wal, tm) = setup_users();
        let mut tx = Transaction::new(std::sync::Arc::clone(&tm));

        run_tx("BEGIN", &cat, &bpm, &wal, &mut tx);
        run_tx("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &mut tx);
        run_tx("COMMIT", &cat, &bpm, &wal, &mut tx);
        let Output::Rows(rows) = run_tx("SELECT id FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
    }
    #[test]
    fn nested_begin_errors() {
        let (cat, bpm, wal, tm) = setup_users();
        let mut tx = Transaction::new(std::sync::Arc::clone(&tm));

        let stmt = parse("BEGIN").unwrap();
        let analyzed = analyze(&cat, &stmt).unwrap();
        let lm = LockManager::new();
        execute(&bpm, &lm, &wal, &tm, &cat, &analyzed, &mut tx).unwrap();
        let err = execute(&bpm, &lm, &wal, &tm, &cat, &analyzed, &mut tx);
        assert!(err.is_err());
    }
    #[test]
    fn concurrent_update_serializes_via_x_lock() {
        // Two threads BEGIN, both UPDATE the same row, then COMMIT.
        // Without the lock manager they'd race; with X-locks the second
        // blocks until the first commits, then proceeds.
        use crate::lock_manager::LockManager;
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let (cat, bpm, wal, tm) = setup_users();
        let cat = Arc::new(cat);
        let lm = Arc::new(LockManager::new());

        // Seed one row.
        {
            let mut tx = Transaction::new(Arc::clone(&tm));
            run_full(
                "INSERT INTO users VALUES (1, 'init')",
                &cat,
                &bpm,
                &lm,
                &wal,
                &mut tx,
            );
        }

        let barrier = Arc::new(Barrier::new(2));
        let (a_started, b_can_proceed) = mpsc::channel::<()>();

        let bpm_a = bpm.clone();
        let lm_a = Arc::clone(&lm);
        let cat_a = Arc::clone(&cat);
        let wal_a = Arc::clone(&wal);
        let tm_a = Arc::clone(&tm);
        let bar_a = Arc::clone(&barrier);
        let h_a = thread::spawn(move || {
            bar_a.wait();
            let mut tx = Transaction::new(tm_a);
            run_full("BEGIN", &cat_a, &bpm_a, &lm_a, &wal_a, &mut tx);
            run_full(
                "UPDATE users SET name = 'A' WHERE id = 1",
                &cat_a,
                &bpm_a,
                &lm_a,
                &wal_a,
                &mut tx,
            );
            a_started.send(()).unwrap();
            thread::sleep(Duration::from_millis(100));
            run_full("COMMIT", &cat_a, &bpm_a, &lm_a, &wal_a, &mut tx);
        });

        let bpm_b = bpm.clone();
        let lm_b = Arc::clone(&lm);
        let cat_b = Arc::clone(&cat);
        let wal_b = Arc::clone(&wal);
        let tm_b = Arc::clone(&tm);
        let bar_b = Arc::clone(&barrier);
        let h_b = thread::spawn(move || {
            bar_b.wait();
            b_can_proceed.recv().unwrap();
            let mut tx = Transaction::new(tm_b);
            run_full("BEGIN", &cat_b, &bpm_b, &lm_b, &wal_b, &mut tx);
            run_full(
                "UPDATE users SET name = 'B' WHERE id = 1",
                &cat_b,
                &bpm_b,
                &lm_b,
                &wal_b,
                &mut tx,
            );
            run_full("COMMIT", &cat_b, &bpm_b, &lm_b, &wal_b, &mut tx);
        });

        h_a.join().unwrap();
        h_b.join().unwrap();

        // Lost-update prevention (added in Phase 4): T2 takes the row's
        // X-lock after T1 commits, sees that the snapshot's row was
        // updated by a concurrent committed txn, and restarts its
        // statement against a fresh snapshot. The restart sees T1's new
        // version and overwrites it. So exactly the second writer's
        // value survives — no anomaly.
        let mut tx = Transaction::new(Arc::clone(&tm));
        let Output::Rows(rows) = run_full(
            "SELECT name FROM users WHERE id = 1",
            &cat,
            &bpm,
            &lm,
            &wal,
            &mut tx,
        ) else {
            panic!()
        };
        let names: std::collections::HashSet<String> = rows
            .iter()
            .filter_map(|r| match &r.values[0] {
                Value::Varchar(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        // Whichever thread won the lock race goes first and gets
        // overwritten by the other. Exactly one of the two writes
        // survives.
        assert_eq!(names.len(), 1, "expected one surviving version, got {names:?}");
        assert!(names.contains("A") || names.contains("B"));
    }

    /// Same as setup_users() but also creates an `orders(id INT, user_id INT,
    /// product VARCHAR)` table — the canonical day17 fixture.
    fn setup_users_orders() -> (
        Catalog,
        BufferPool,
        std::sync::Arc<crate::wal::WalManager>,
        std::sync::Arc<crate::transaction_manager::TransactionManager>,
    ) {
        let (cat, bpm, wal, tm) = setup_users();
        run(
            "CREATE TABLE orders (id INT, user_id INT, product VARCHAR)",
            &cat,
            &bpm,
            &wal,
            &tm,
        );
        (cat, bpm, wal, tm)
    }

    #[test]
    fn inner_join_basic() {
        let (cat, bpm, wal, tm) = setup_users_orders();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, 'Bob')", &cat, &bpm, &wal, &tm);
        run(
            "INSERT INTO orders VALUES (1, 1, 'Book')",
            &cat,
            &bpm,
            &wal,
            &tm,
        );
        run(
            "INSERT INTO orders VALUES (2, 1, 'Pen')",
            &cat,
            &bpm,
            &wal,
            &tm,
        );
        run(
            "INSERT INTO orders VALUES (3, 2, 'Notebook')",
            &cat,
            &bpm,
            &wal,
            &tm,
        );
        let Output::Rows(rows) = run(
            "SELECT u.id, u.name, o.product FROM users u INNER JOIN orders o ON u.id = o.user_id",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 3);
        // Three matched products, one per row.
        let products: std::collections::HashSet<String> = rows
            .iter()
            .filter_map(|r| match &r.values[2] {
                Value::Varchar(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(products.contains("Book"));
        assert!(products.contains("Pen"));
        assert!(products.contains("Notebook"));
    }

    #[test]
    fn left_join_unmatched_emits_null() {
        let (cat, bpm, wal, tm) = setup_users_orders();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, 'Bob')", &cat, &bpm, &wal, &tm);
        // No orders for Bob.
        run(
            "INSERT INTO orders VALUES (1, 1, 'Book')",
            &cat,
            &bpm,
            &wal,
            &tm,
        );
        let Output::Rows(rows) = run(
            "SELECT u.id, u.name, o.product FROM users u LEFT JOIN orders o ON u.id = o.user_id",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        // Find Bob's row — product should be NULL.
        let bob = rows
            .iter()
            .find(|r| matches!(&r.values[1], Value::Varchar(s) if s == "Bob"))
            .expect("Bob row");
        assert_eq!(bob.values[2], Value::Null);
    }

    /// Build a fresh DB with `sales(region, product, quantity, price)` and
    /// the canonical day18 fixture data already loaded.
    fn setup_sales() -> (
        Catalog,
        BufferPool,
        std::sync::Arc<crate::wal::WalManager>,
        std::sync::Arc<crate::transaction_manager::TransactionManager>,
    ) {
        let path = temp_path("sales");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(
            crate::wal::WalManager::open(&path.with_extension("wal")).unwrap(),
        );
        let bpm = BufferPool::new(disk, 8, wal.clone());
        let clog = std::sync::Arc::new(crate::clog::Clog::in_memory());
        let tm = std::sync::Arc::new(
            crate::transaction_manager::TransactionManager::new(clog),
        );
        crate::bootstrap::bootstrap(&bpm, &tm).unwrap();
        let cat = Catalog::new(bpm.clone(), std::sync::Arc::clone(&tm));
        run(
            "CREATE TABLE sales (region VARCHAR, product VARCHAR, quantity INT, price INT)",
            &cat,
            &bpm,
            &wal,
            &tm,
        );
        for sql in [
            "INSERT INTO sales VALUES ('east', 'apple', 10, 100)",
            "INSERT INTO sales VALUES ('east', 'apple', 5, 100)",
            "INSERT INTO sales VALUES ('east', 'banana', 8, 50)",
            "INSERT INTO sales VALUES ('west', 'apple', 3, 100)",
            "INSERT INTO sales VALUES ('west', 'banana', 12, 50)",
            "INSERT INTO sales VALUES ('west', 'banana', 7, 50)",
        ] {
            run(sql, &cat, &bpm, &wal, &tm);
        }
        (cat, bpm, wal, tm)
    }

    #[test]
    fn count_star_no_group() {
        let (cat, bpm, wal, tm) = setup_sales();
        let Output::Rows(rows) = run("SELECT COUNT(*) FROM sales", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Int(6));
    }

    #[test]
    fn count_star_on_empty_returns_zero() {
        let (cat, bpm, wal, tm) = setup_users(); // empty users table
        let Output::Rows(rows) = run("SELECT COUNT(*) FROM users", &cat, &bpm, &wal, &tm) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Int(0));
    }

    #[test]
    fn sum_avg_min_max_no_group() {
        let (cat, bpm, wal, tm) = setup_sales();
        let Output::Rows(rows) = run(
            "SELECT SUM(quantity), AVG(quantity), MIN(quantity), MAX(quantity) FROM sales",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Int(45)); // SUM = 10+5+8+3+12+7
        assert_eq!(rows[0].values[1], Value::Double(7.5)); // AVG = 45/6 = 7.5 (DOUBLE)
        assert_eq!(rows[0].values[2], Value::Int(3)); // MIN
        assert_eq!(rows[0].values[3], Value::Int(12)); // MAX
    }

    #[test]
    fn group_by_single_column() {
        let (cat, bpm, wal, tm) = setup_sales();
        let Output::Rows(rows) = run(
            "SELECT product, COUNT(*), SUM(quantity) FROM sales GROUP BY product",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        // Build a {product → (count, sum)} map.
        let mut got = std::collections::HashMap::new();
        for r in &rows {
            let Value::Varchar(p) = &r.values[0] else { panic!() };
            let Value::Int(c) = &r.values[1] else { panic!() };
            let Value::Int(s) = &r.values[2] else { panic!() };
            got.insert(p.clone(), (*c, *s));
        }
        assert_eq!(got["apple"], (3, 18));
        assert_eq!(got["banana"], (3, 27));
    }

    #[test]
    fn group_by_having() {
        let (cat, bpm, wal, tm) = setup_sales();
        let Output::Rows(rows) = run(
            "SELECT product, SUM(quantity) FROM sales GROUP BY product HAVING SUM(quantity) > 20",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        let Value::Varchar(p) = &rows[0].values[0] else { panic!() };
        assert_eq!(p, "banana");
        assert_eq!(rows[0].values[1], Value::Int(27));
    }

    #[test]
    fn ungrouped_column_in_select_errors() {
        let (cat, _bpm, _wal, tm) = setup_sales();
        let stmt = parse("SELECT region, COUNT(*) FROM sales GROUP BY product").unwrap();
        let err = analyze(&cat, &stmt).unwrap_err().to_string();
        assert!(err.contains("GROUP BY"), "got: {err}");
        let _ = tm;
    }

    #[test]
    fn select_distinct_dedupes() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, 'a')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (3, 'b')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (4, NULL)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (5, NULL)", &cat, &bpm, &wal, &tm);
        let Output::Rows(rows) = run(
            "SELECT DISTINCT name FROM users ORDER BY name",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        // Distinct values: 'a', 'b', NULL → 3 rows. Order: a, b, NULL (NULL last under ASC).
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], Value::Varchar("a".into()));
        assert_eq!(rows[1].values[0], Value::Varchar("b".into()));
        assert_eq!(rows[2].values[0], Value::Null);
    }

    #[test]
    fn order_by_asc_default() {
        let (cat, bpm, wal, tm) = setup_users();
        for (i, name) in [(3, "c"), (1, "a"), (2, "b")] {
            run(
                &format!("INSERT INTO users VALUES ({i}, '{name}')"),
                &cat,
                &bpm,
                &wal,
                &tm,
            );
        }
        let Output::Rows(rows) = run("SELECT id FROM users ORDER BY id", &cat, &bpm, &wal, &tm)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[1].values[0], Value::Int(2));
        assert_eq!(rows[2].values[0], Value::Int(3));
    }

    #[test]
    fn order_by_desc_with_limit() {
        let (cat, bpm, wal, tm) = setup_users();
        for i in [1, 4, 2, 5, 3] {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm,
                &wal,
                &tm,
            );
        }
        let Output::Rows(rows) = run(
            "SELECT id FROM users ORDER BY id DESC LIMIT 2",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], Value::Int(5));
        assert_eq!(rows[1].values[0], Value::Int(4));
    }

    #[test]
    fn order_by_null_placement_asc_last() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, NULL)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (3, 'b')", &cat, &bpm, &wal, &tm);
        let Output::Rows(rows) = run(
            "SELECT id FROM users ORDER BY name",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        // NULL row sorts last under ASC.
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[1].values[0], Value::Int(3));
        assert_eq!(rows[2].values[0], Value::Int(2));
    }

    #[test]
    fn order_by_with_aggregate() {
        let (cat, bpm, wal, tm) = setup_sales();
        let Output::Rows(rows) = run(
            "SELECT product, SUM(quantity) FROM sales GROUP BY product ORDER BY SUM(quantity) DESC",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        let Value::Varchar(p0) = &rows[0].values[0] else { panic!() };
        assert_eq!(p0, "banana"); // SUM=27 > 18
    }

    #[test]
    fn double_column_round_trip_and_arithmetic() {
        let path = temp_path("dbl");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(
            crate::wal::WalManager::open(&path.with_extension("wal")).unwrap(),
        );
        let bpm = BufferPool::new(disk, 8, wal.clone());
        let clog = std::sync::Arc::new(crate::clog::Clog::in_memory());
        let tm = std::sync::Arc::new(
            crate::transaction_manager::TransactionManager::new(clog),
        );
        crate::bootstrap::bootstrap(&bpm, &tm).unwrap();
        let cat = Catalog::new(bpm.clone(), std::sync::Arc::clone(&tm));
        run("CREATE TABLE m (id INT, ratio DOUBLE)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO m VALUES (1, 1.5)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO m VALUES (2, 2.25)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO m VALUES (3, 0.75)", &cat, &bpm, &wal, &tm);

        // Round-trip: stored DOUBLE comes back as DOUBLE.
        let Output::Rows(rows) = run(
            "SELECT ratio FROM m ORDER BY id",
            &cat, &bpm, &wal, &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Double(1.5));
        assert_eq!(rows[1].values[0], Value::Double(2.25));

        // INT + DOUBLE promotes to DOUBLE.
        let Output::Rows(rows) = run(
            "SELECT id + ratio FROM m ORDER BY id",
            &cat, &bpm, &wal, &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Double(2.5));
        assert_eq!(rows[1].values[0], Value::Double(4.25));

        // SUM(DOUBLE) → DOUBLE. AVG(DOUBLE) → DOUBLE.
        let Output::Rows(rows) = run(
            "SELECT SUM(ratio), AVG(ratio) FROM m",
            &cat, &bpm, &wal, &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Double(4.5));
        assert_eq!(rows[0].values[1], Value::Double(1.5));
    }

    #[test]
    fn select_alias_carries_through() {
        // The execution layer doesn't surface the alias by itself — it's
        // attached to AnalyzedSelectItem. The instance.rs RowDescription
        // path uses it for the column name. Here we just verify the
        // analyzer wired it through.
        let (cat, _bpm, _wal, _tm) = setup_users();
        let stmt = parse("SELECT id AS user_id, name n FROM users").unwrap();
        let analyzed = analyze(&cat, &stmt).unwrap();
        let crate::analyzer::AnalyzedStatement::Select(s) = analyzed else { panic!() };
        assert_eq!(s.select_items[0].alias.as_deref(), Some("user_id"));
        assert_eq!(s.select_items[1].alias.as_deref(), Some("n"));
    }

    #[test]
    fn is_null_filters_to_null_rows() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, NULL)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (3, 'b')", &cat, &bpm, &wal, &tm);
        let Output::Rows(rows) = run(
            "SELECT id FROM users WHERE name IS NULL",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Int(2));
    }

    #[test]
    fn is_not_null_filters_out_null_rows() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (2, NULL)", &cat, &bpm, &wal, &tm);
        run("INSERT INTO users VALUES (3, 'b')", &cat, &bpm, &wal, &tm);
        let Output::Rows(rows) = run(
            "SELECT id FROM users WHERE name IS NOT NULL ORDER BY id",
            &cat,
            &bpm,
            &wal,
            &tm,
        ) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[1].values[0], Value::Int(3));
    }

    #[test]
    fn limit_zero_emits_no_rows() {
        let (cat, bpm, wal, tm) = setup_users();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &tm);
        let Output::Rows(rows) =
            run("SELECT id FROM users LIMIT 0", &cat, &bpm, &wal, &tm)
        else {
            panic!()
        };
        assert!(rows.is_empty());
    }

    #[test]
    fn aggregate_in_where_errors() {
        let (cat, _bpm, _wal, tm) = setup_sales();
        let stmt = parse("SELECT product FROM sales WHERE SUM(quantity) > 10").unwrap();
        let err = analyze(&cat, &stmt).unwrap_err().to_string();
        assert!(err.contains("aggregate"), "got: {err}");
        let _ = tm;
    }

    #[test]
    fn join_qualified_column_ambiguous_unqualified_errors() {
        // Both users.id and orders.id exist; bare `id` should be ambiguous.
        let (cat, bpm, wal, tm) = setup_users_orders();
        let lm = LockManager::new();
        let mut tx = Transaction::new(std::sync::Arc::clone(&tm));
        let stmt = parse(
            "SELECT id FROM users u INNER JOIN orders o ON u.id = o.user_id",
        )
        .unwrap();
        let err = analyze(&cat, &stmt).unwrap_err().to_string();
        assert!(err.contains("ambiguous"), "got: {err}");
        // Avoid unused-var warnings.
        let _ = (bpm, wal, lm, &mut tx);
    }
}
