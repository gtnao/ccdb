//! Volcano-style executor: each operator implements [`Executor::open`] and
//! [`Executor::next`], yielding tuples lazily up the tree.
//!
//! INSERT is intentionally *not* an Executor — it's a side-effecting action
//! and doesn't compose with row sources. The engine matches on statement
//! shape and dispatches separately.

use anyhow::{Result, bail};

use crate::analyzer::{
    AnalyzedDeleteStatement, AnalyzedExpr, AnalyzedFrom, AnalyzedInsertStatement, AnalyzedLiteral,
    AnalyzedSelectStatement, AnalyzedStatement, AnalyzedUpdateStatement, LiteralValue, TableSource,
};
use crate::ast::{BinaryOperator, JoinType, UnaryOperator};
use crate::buffer_pool::BufferPool;
use crate::catalog::Catalog;
use crate::lock_manager::{LockManager, LockMode};
use crate::page::{PageId, Rid, SlotId};
use crate::transaction::Transaction;
use crate::transaction_manager::{Snapshot, TransactionManager};
use crate::tuple::{
    INVALID_TXN_ID, Schema, Value, deserialize_tuple_mvcc, serialize_tuple_mvcc,
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
                let mut out = None;
                while self.cur_slot < tc {
                    if let Some(raw) = page.get_tuple(self.cur_slot) {
                        let (xmin, xmax, values) = deserialize_tuple_mvcc(raw, &self.schema)?;
                        self.cur_slot += 1;
                        if visibility::is_visible(xmin, xmax, &self.snapshot, self.tm) {
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
    }
}

fn literal_to_value(lit: &AnalyzedLiteral) -> Value {
    match &lit.value {
        // Note: Literal::Integer is i64 in the AST but our runtime int is i32.
        // Truncation here is the same compromise as the storage tuple layout.
        LiteralValue::Integer(n) => Value::Int(*n as i32),
        LiteralValue::String(s) => Value::Varchar(s.clone()),
        LiteralValue::Boolean(b) => Value::Bool(*b),
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
    match (l, r) {
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
        DT_BOOL, DT_INT, DT_VARCHAR, PG_ATTRIBUTE_PAGE_ID, PG_CLASS_PAGE_ID,
    };

    // Pick a fresh table_id (max existing + 1). Catalog scan is enough at
    // this scale; no concurrent CREATEs assumed.
    let mut max_id: i32 = -1;
    for table in catalog.user_tables()? {
        max_id = max_id.max(table.table_id as i32);
    }
    // System tables occupy 0 and 1; user tables start at 2.
    let new_table_id = (max_id + 1).max(2);

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
        };
        let row = serialize_tuple_mvcc(
            tx.id(),
            INVALID_TXN_ID,
            &[
                Value::Int(new_table_id),
                Value::Varchar(col.name.clone()),
                Value::Int(dt),
                Value::Bool(col.nullable),
                Value::Int(ord as i32),
            ],
        );
        insert_bytes(bpm, wal, tx, PG_ATTRIBUTE_PAGE_ID, &row)?;
    }
    Ok(())
}

fn perform_insert(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedInsertStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    let values: Vec<Value> = stmt
        .values
        .iter()
        .map(|e| match e {
            AnalyzedExpr::Literal(lit) => Ok(literal_to_value(lit)),
            _ => bail!("INSERT VALUES must be literals (no exprs yet)"),
        })
        .collect::<Result<_>>()?;
    let table = catalog
        .table_by_id(stmt.table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {} not in catalog", stmt.table_id))?;
    let bytes = serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &values);
    let (rid, _lsn) = insert_bytes(bpm, wal, tx, table.first_page_id, &bytes)?;
    lm.lock(tx.id(), rid, LockMode::Exclusive)
        .map_err(|e| anyhow::anyhow!("X-lock on {rid:?}: {e}"))?;
    tx.add_lock(rid);
    Ok(1)
}

// -- DELETE / UPDATE (not Executors either — both are bulk side effects) -----

/// Visit all *visible* rows of a table under the given snapshot, returning
/// (rid, values) pairs for predicate evaluation. Tuples invisible to the
/// snapshot (uncommitted others, future-tx, already-deleted) are skipped.
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
        for slot in 0..tc {
            if let Some(raw) = page.get_tuple(slot) {
                let (xmin, xmax, values) = deserialize_tuple_mvcc(raw, &schema)?;
                if visibility::is_visible(xmin, xmax, snapshot, tm) {
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
    let snapshot = tx
        .snapshot()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no snapshot for tx"))?;
    let (_schema, rows) = visible_rows(bpm, catalog, stmt.table_id, &snapshot, tm)?;
    let mut victims: Vec<Rid> = Vec::new();
    for (pid, slot, values) in rows {
        let t = Tuple::new(values);
        if matches(stmt.where_clause.as_ref(), &t)? {
            victims.push((pid, slot));
        }
    }
    for &(pid, slot) in &victims {
        lm.lock(tx.id(), (pid, slot), LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {:?}: {e}", (pid, slot)))?;
        tx.add_lock((pid, slot));
        // MVCC logical delete: only the xmax field changes.
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
    Ok(victims.len())
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
    let snapshot = tx
        .snapshot()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no snapshot for tx"))?;
    let (_schema, rows) = visible_rows(bpm, catalog, stmt.table_id, &snapshot, tm)?;
    let table = catalog
        .table_by_id(stmt.table_id)?
        .ok_or_else(|| anyhow::anyhow!("table id {} not in catalog", stmt.table_id))?;
    let mut work: Vec<(PageId, SlotId, Vec<u8>)> = Vec::new();

    for (pid, slot, values) in rows {
        let t = Tuple::new(values);
        if !matches(stmt.where_clause.as_ref(), &t)? {
            continue;
        }
        let mut new_values = t.values.clone();
        for a in &stmt.assignments {
            let v = evaluate_expr(&a.value, &t)?;
            new_values[a.column_index] = v;
        }
        let new_bytes = serialize_tuple_mvcc(tx.id(), INVALID_TXN_ID, &new_values);
        work.push((pid, slot, new_bytes));
    }

    let count = work.len();
    for (pid, slot, new_bytes) in work {
        lm.lock(tx.id(), (pid, slot), LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {:?}: {e}", (pid, slot)))?;
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
        let (new_rid, _) = insert_bytes(bpm, wal, tx, table.first_page_id, &new_bytes)?;
        lm.lock(tx.id(), new_rid, LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {new_rid:?}: {e}"))?;
        tx.add_lock(new_rid);
    }
    Ok(count)
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
    // 1. Find the last page in the chain.
    let mut last = first_page;
    loop {
        let g = bpm.fetch_page(last)?;
        let next = g.read().next_page_id();
        if next == crate::page::NO_NEXT_PAGE {
            break;
        }
        last = next;
    }

    // 2. Try to insert into the last page.
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
            return Ok((rid, lsn));
        }
    }

    // 3. Allocate a fresh page and link it after `last`.
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
        // Now link the previous tail to the new page.
        let prev_g = bpm.fetch_page(last)?;
        prev_g.write().set_next_page_id(new_pid);
        return Ok((rid, lsn));
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
    let from_exec = build_from_pipeline(bpm, catalog, &stmt.from, &stmt.range_table, &snapshot, tm)?;
    let filtered: Box<dyn Executor + 'a> = match &stmt.where_clause {
        Some(p) => Box::new(Filter::new(from_exec, p.clone())),
        None => from_exec,
    };
    let exprs: Vec<AnalyzedExpr> = stmt
        .select_items
        .iter()
        .map(|i| i.expr.clone())
        .collect();
    Ok(Box::new(Project::new(filtered, exprs)))
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

        // Under MVCC without lost-update prevention, both T1's and T2's
        // updates produce visible versions: T2's snapshot was taken before
        // T1 committed, so T2's UPDATE matches the *original* (xmax=T1 in
        // T2's view doesn't make it invisible because T1 was active in T2's
        // snapshot). Each writer's new tuple lives on. This is the classic
        // "lost update" anomaly that plain Snapshot Isolation allows;
        // preventing it would require a row-version check at write time
        // (Postgres' EvalPlanQual). Within day14's scope we just verify
        // that BOTH writers' values become visible.
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
        assert!(names.contains("A"));
        assert!(names.contains("B"));
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
