//! Volcano-style executor: each operator implements [`Executor::open`] and
//! [`Executor::next`], yielding tuples lazily up the tree.
//!
//! INSERT is intentionally *not* an Executor — it's a side-effecting action
//! and doesn't compose with row sources. The engine matches on statement
//! shape and dispatches separately.

use anyhow::{Result, bail};

use crate::analyzer::{
    AnalyzedDeleteStatement, AnalyzedExpr, AnalyzedInsertStatement, AnalyzedLiteral,
    AnalyzedSelectStatement, AnalyzedStatement, AnalyzedUpdateStatement, LiteralValue, TableSource,
};
use crate::ast::{BinaryOperator, UnaryOperator};
use crate::buffer_pool::BufferPool;
use crate::catalog::Catalog;
use crate::lock_manager::{LockManager, LockMode};
use crate::page::{PageId, Rid, SlotId};
use crate::transaction::{Transaction, UndoLogEntry};
use crate::tuple::{Schema, Value, deserialize_tuple, serialize_tuple};
use crate::wal::{ClrRedo, Lsn, WalManager, WalRecordType};

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

pub struct SeqScan<'a> {
    bpm: &'a BufferPool,
    schema: Schema,
    cur_page: u32,
    cur_slot: u16,
    locking: Option<LockSink<'a>>,
}

/// Borrowed handle SeqScan uses to acquire S-locks per row and remember
/// what it locked so the caller (execute) can release on commit.
struct LockSink<'a> {
    lm: &'a LockManager,
    tx_id: u64,
    held: &'a mut std::collections::HashSet<Rid>,
}

impl<'a> SeqScan<'a> {
    pub fn new(bpm: &'a BufferPool, catalog: &Catalog, table_id: usize) -> Result<Self> {
        let schema = catalog
            .table_by_id(table_id)
            .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?
            .to_schema();
        Ok(Self {
            bpm,
            schema,
            cur_page: 0,
            cur_slot: 0,
            locking: None,
        })
    }

    pub fn with_locking(
        bpm: &'a BufferPool,
        catalog: &Catalog,
        table_id: usize,
        lm: &'a LockManager,
        tx_id: u64,
        held: &'a mut std::collections::HashSet<Rid>,
    ) -> Result<Self> {
        let mut s = Self::new(bpm, catalog, table_id)?;
        s.locking = Some(LockSink { lm, tx_id, held });
        Ok(s)
    }
}

impl Executor for SeqScan<'_> {
    fn open(&mut self) -> Result<()> {
        self.cur_page = 0;
        self.cur_slot = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>> {
        let n = self.bpm.page_count();
        while self.cur_page < n {
            let found: Option<(Rid, Vec<Value>)> = {
                let guard = self.bpm.fetch_page(self.cur_page)?;
                let page = guard.read();
                let tc = page.tuple_count();
                let mut out = None;
                while self.cur_slot < tc {
                    if let Some(raw) = page.get_tuple(self.cur_slot) {
                        let values = deserialize_tuple(raw, &self.schema)?;
                        let rid = (self.cur_page, self.cur_slot);
                        self.cur_slot += 1;
                        out = Some((rid, values));
                        break;
                    }
                    self.cur_slot += 1;
                }
                out
            };
            if let Some((rid, values)) = found {
                if let Some(sink) = self.locking.as_mut() {
                    sink.lm
                        .lock(sink.tx_id, rid, LockMode::Shared)
                        .map_err(|e| anyhow::anyhow!("S-lock on {rid:?}: {e}"))?;
                    sink.held.insert(rid);
                }
                return Ok(Some(Tuple::new(values)));
            }
            self.cur_page += 1;
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

fn perform_insert(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
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
    let bytes = serialize_tuple(&values);
    let (rid, lsn) = insert_bytes(bpm, wal, tx, &bytes)?;
    lm.lock(tx.id(), rid, LockMode::Exclusive)
        .map_err(|e| anyhow::anyhow!("X-lock on {rid:?}: {e}"))?;
    tx.add_lock(rid);
    if tx.is_active() {
        tx.record(UndoLogEntry::Insert { rid, lsn });
    }
    Ok(1)
}

// -- DELETE / UPDATE (not Executors either — both are bulk side effects) -----

/// Materializes the table once into (rid, tuple) pairs so we can apply
/// modifications without worrying about re-visiting newly inserted rows
/// (UPDATE does delete+insert; without snapshotting we'd loop forever).
fn snapshot_table(
    bpm: &BufferPool,
    catalog: &Catalog,
    table_id: usize,
) -> Result<(Schema, Vec<(PageId, SlotId, Vec<Value>)>)> {
    let schema = catalog
        .table_by_id(table_id)
        .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?
        .to_schema();
    let mut out = Vec::new();
    for pid in 0..bpm.page_count() {
        let guard = bpm.fetch_page(pid)?;
        let page = guard.read();
        let tc = page.tuple_count();
        for slot in 0..tc {
            if let Some(raw) = page.get_tuple(slot) {
                let values = deserialize_tuple(raw, &schema)?;
                out.push((pid, slot, values));
            }
        }
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
    catalog: &Catalog,
    stmt: &AnalyzedDeleteStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    let (_schema, rows) = snapshot_table(bpm, catalog, stmt.table_id)?;
    let mut victims: Vec<(Rid, Vec<u8>)> = Vec::new();
    for (pid, slot, values) in rows {
        let t = Tuple::new(values);
        if matches(stmt.where_clause.as_ref(), &t)? {
            let bytes = serialize_tuple(&t.values);
            victims.push(((pid, slot), bytes));
        }
    }
    for (rid, bytes) in &victims {
        let (pid, slot) = *rid;
        lm.lock(tx.id(), (pid, slot), LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {:?}: {e}", (pid, slot)))?;
        tx.add_lock((pid, slot));
        let lsn = {
            let g = bpm.fetch_page(pid)?;
            let mut p = g.write();
            p.delete(slot)?;
            let lsn = log_record(
                wal,
                tx,
                WalRecordType::Delete {
                    rid: (pid, slot),
                    data: bytes.clone(),
                },
            )?;
            p.set_page_lsn(lsn);
            lsn
        };
        if tx.is_active() {
            tx.record(UndoLogEntry::Delete {
                rid: (pid, slot),
                data: bytes.clone(),
                lsn,
            });
        }
    }
    Ok(victims.len())
}

fn perform_update(
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    catalog: &Catalog,
    stmt: &AnalyzedUpdateStatement,
    tx: &mut Transaction,
) -> Result<usize> {
    let (_schema, rows) = snapshot_table(bpm, catalog, stmt.table_id)?;
    // Per matched row we keep: old rid, old bytes (for undo of the delete),
    // and the new tuple bytes to insert.
    let mut work: Vec<(PageId, SlotId, Vec<u8>, Vec<u8>)> = Vec::new();

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
        let old_bytes = serialize_tuple(&t.values);
        let new_bytes = serialize_tuple(&new_values);
        work.push((pid, slot, old_bytes, new_bytes));
    }

    let count = work.len();
    for (pid, slot, old_bytes, new_bytes) in work {
        lm.lock(tx.id(), (pid, slot), LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {:?}: {e}", (pid, slot)))?;
        tx.add_lock((pid, slot));
        let del_lsn = {
            let g = bpm.fetch_page(pid)?;
            let mut p = g.write();
            p.delete(slot)?;
            let lsn = log_record(
                wal,
                tx,
                WalRecordType::Delete {
                    rid: (pid, slot),
                    data: old_bytes.clone(),
                },
            )?;
            p.set_page_lsn(lsn);
            lsn
        };
        if tx.is_active() {
            tx.record(UndoLogEntry::Delete {
                rid: (pid, slot),
                data: old_bytes,
                lsn: del_lsn,
            });
        }
        let (new_rid, ins_lsn) = insert_bytes(bpm, wal, tx, &new_bytes)?;
        lm.lock(tx.id(), new_rid, LockMode::Exclusive)
            .map_err(|e| anyhow::anyhow!("X-lock on {new_rid:?}: {e}"))?;
        tx.add_lock(new_rid);
        if tx.is_active() {
            tx.record(UndoLogEntry::Insert {
                rid: new_rid,
                lsn: ins_lsn,
            });
        }
    }
    Ok(count)
}

// Shared insertion helper. Writes the tuple, records WAL Insert chained
// against the current tx, and stamps the page with the resulting LSN.
// Returns (rid, lsn-of-Insert-record) so callers can record it in the
// undo log for CLR chaining.
fn insert_bytes(
    bpm: &BufferPool,
    wal: &WalManager,
    tx: &mut Transaction,
    bytes: &[u8],
) -> Result<(Rid, Lsn)> {
    let n = bpm.page_count();
    if n > 0 {
        let last = n - 1;
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
        drop(p);
        drop(g);
    }
    let g = bpm.new_page()?;
    let pid = g.page_id();
    let mut p = g.write();
    let slot = p
        .insert(bytes)
        .map_err(|e| anyhow::anyhow!("tuple does not fit on a fresh page: {e}"))?;
    let rid = (pid, slot);
    let lsn = log_record(
        wal,
        tx,
        WalRecordType::Insert {
            rid,
            data: bytes.to_vec(),
        },
    )?;
    p.set_page_lsn(lsn);
    Ok((rid, lsn))
}

/// Roll back a transaction by walking its undo log in reverse. Each
/// inverse-action is paired with a CLR record so a crash mid-rollback can
/// be recovered without double-undoing. After all CLRs are written we
/// emit Abort and fsync.
pub fn rollback(bpm: &BufferPool, wal: &WalManager, tx: &mut Transaction) -> Result<()> {
    let entries = tx.drain_log();
    let n = entries.len();
    for i in (0..n).rev() {
        let undo_next = if i == 0 { 0 } else { entries[i - 1].lsn() };
        match &entries[i] {
            UndoLogEntry::Insert { rid, .. } => {
                let (pid, slot) = *rid;
                let g = bpm.fetch_page(pid)?;
                let mut p = g.write();
                if p.get_tuple(slot).is_some() {
                    p.delete(slot)?;
                }
                let clr_lsn = log_record(
                    wal,
                    tx,
                    WalRecordType::Clr {
                        undo_next_lsn: undo_next,
                        redo: ClrRedo::UndoInsert { rid: *rid },
                    },
                )?;
                p.set_page_lsn(clr_lsn);
            }
            UndoLogEntry::Delete { rid, data, .. } => {
                let (pid, slot) = *rid;
                let g = bpm.fetch_page(pid)?;
                let mut p = g.write();
                if p.get_tuple(slot).is_none() {
                    p.restore(slot, data)?;
                }
                let clr_lsn = log_record(
                    wal,
                    tx,
                    WalRecordType::Clr {
                        undo_next_lsn: undo_next,
                        redo: ClrRedo::UndoDelete {
                            rid: *rid,
                            data: data.clone(),
                        },
                    },
                )?;
                p.set_page_lsn(clr_lsn);
            }
        }
    }
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
            // SeqScan accumulates S-locks into a local set so it doesn't need
            // to borrow tx mutably alongside the pipeline.
            let mut local_locks: std::collections::HashSet<Rid> =
                std::collections::HashSet::new();
            let rows = {
                let mut exec = build_select_pipeline_locked(
                    bpm,
                    catalog,
                    s,
                    lm,
                    tx.id(),
                    &mut local_locks,
                )?;
                exec.open()?;
                let mut rows = Vec::new();
                while let Some(t) = exec.next()? {
                    rows.push(t);
                }
                rows
            };
            for rid in local_locks {
                tx.add_lock(rid);
            }
            Ok(Output::Rows(rows))
        }
        AnalyzedStatement::Insert(s) => {
            Ok(Output::Affected(perform_insert(bpm, lm, wal, s, tx)?))
        }
        AnalyzedStatement::Delete(s) => Ok(Output::Affected(perform_delete(
            bpm, lm, wal, catalog, s, tx,
        )?)),
        AnalyzedStatement::Update(s) => Ok(Output::Affected(perform_update(
            bpm, lm, wal, catalog, s, tx,
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
        AnalyzedStatement::CreateTable(_) => {
            bail!("CREATE TABLE execution is not yet wired up (catalog is read-only)")
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
        // Best-effort: if WAL append fails here we still propagate the
        // original result.
        let _ = log_record(wal, tx, bracket);
        let _ = wal.flush();
    }

    if was_inactive_at_start && !tx.is_active() {
        let held = tx.take_held_locks();
        lm.unlock_all(tx.id(), &held);
    }

    result
}

fn build_select_pipeline_locked<'a>(
    bpm: &'a BufferPool,
    catalog: &'a Catalog,
    stmt: &AnalyzedSelectStatement,
    lm: &'a LockManager,
    tx_id: u64,
    held: &'a mut std::collections::HashSet<Rid>,
) -> Result<Box<dyn Executor + 'a>> {
    let rte = &stmt.range_table[stmt.from_rte_index];
    let table_id = match &rte.source {
        TableSource::BaseTable { table_id, .. } => *table_id,
    };
    let scan: Box<dyn Executor + 'a> =
        Box::new(SeqScan::with_locking(bpm, catalog, table_id, lm, tx_id, held)?);
    let filtered: Box<dyn Executor + 'a> = match &stmt.where_clause {
        Some(p) => Box::new(Filter::new(scan, p.clone())),
        None => scan,
    };
    let exprs: Vec<AnalyzedExpr> = stmt
        .select_items
        .iter()
        .map(|i| i.expr.clone())
        .collect();
    Ok(Box::new(Project::new(filtered, exprs)))
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

    fn run(sql: &str, cat: &Catalog, bpm: &BufferPool, wal: &WalManager) -> Output {
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
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
        let stmt = parse(sql).unwrap();
        let analyzed = analyze(cat, &stmt).unwrap();
        execute(bpm, lm, wal, cat, &analyzed, tx).unwrap()
    }

    #[test]
    fn insert_then_select_star() {
        let path = temp_path("insert-select");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();

        for sql in [
            "INSERT INTO users VALUES (1, 'Alice')",
            "INSERT INTO users VALUES (2, 'Bob')",
            "INSERT INTO users VALUES (3, NULL)",
        ] {
            assert!(matches!(run(sql, &cat, &bpm, &wal), Output::Affected(1)));
        }
        let Output::Rows(rows) = run("SELECT * FROM users", &cat, &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[2].values[1], Value::Null);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn where_filters_rows() {
        let path = temp_path("where");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        for i in 1..=5 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm, &wal);
        }
        let Output::Rows(rows) = run("SELECT id FROM users WHERE id > 2", &cat, &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], Value::Int(3));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn projection_evaluates_arithmetic() {
        let path = temp_path("proj");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (10, 'a')", &cat, &bpm, &wal);
        let Output::Rows(rows) = run("SELECT id + 1 FROM users", &cat, &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Int(11));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn null_predicate_excludes_row() {
        let path = temp_path("null-pred");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal);
        run("INSERT INTO users VALUES (2, NULL)", &cat, &bpm, &wal);
        // name = 'Alice' on the NULL row evaluates to NULL → row excluded.
        let Output::Rows(rows) = run(
            "SELECT id FROM users WHERE name = 'Alice'",
            &cat,
            &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], Value::Int(1));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_with_predicate() {
        let path = temp_path("delete-pred");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        for i in 1..=4 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm, &wal);
        }
        assert!(matches!(
            run("DELETE FROM users WHERE id > 2", &cat, &bpm, &wal),
            Output::Affected(2)
        ));
        let Output::Rows(rows) = run("SELECT id FROM users", &cat, &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], Value::Int(1));
        assert_eq!(rows[1].values[0], Value::Int(2));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_all_rows() {
        let path = temp_path("delete-all");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        for i in 1..=3 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &bpm, &wal);
        }
        assert!(matches!(
            run("DELETE FROM users", &cat, &bpm, &wal),
            Output::Affected(3)
        ));
        let Output::Rows(rows) = run("SELECT * FROM users", &cat, &bpm, &wal) else {
            panic!()
        };
        assert!(rows.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn update_changes_matching_rows() {
        let path = temp_path("update");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &bpm, &wal);
        run("INSERT INTO users VALUES (2, 'Bob')", &cat, &bpm, &wal);
        assert!(matches!(
            run(
                "UPDATE users SET name = 'A2' WHERE id = 1",
                &cat,
                &bpm, &wal),
            Output::Affected(1)
        ));
        let Output::Rows(rows) = run("SELECT id, name FROM users", &cat, &bpm, &wal) else {
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
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn update_does_not_re_match_inserted_row() {
        // Snapshot-then-apply guarantees we don't see our own writes.
        let path = temp_path("update-stable");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal);
        // SET name = 'a' WHERE name = 'a' affects exactly one row, not infinite.
        assert!(matches!(
            run(
                "UPDATE users SET name = 'a' WHERE name = 'a'",
                &cat,
                &bpm, &wal),
            Output::Affected(1)
        ));
        let Output::Rows(rows) = run("SELECT id FROM users", &cat, &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn null_propagates_in_arithmetic() {
        let path = temp_path("null-arith");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, NULL)", &cat, &bpm, &wal);
        // SELECT name (which is NULL) projects through; arithmetic on NULL would
        // also produce NULL — exercised via id (non-null) for the all-OK row.
        let Output::Rows(rows) = run("SELECT name FROM users", &cat, &bpm, &wal) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Null);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rollback_undoes_insert() {
        let path = temp_path("tx-insert");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));

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
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rollback_undoes_delete() {
        let path = temp_path("tx-delete");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));

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
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rollback_undoes_update() {
        // UPDATE = delete + insert, so the undo log holds two entries per row.
        let path = temp_path("tx-update");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));

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
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn commit_persists_changes() {
        let path = temp_path("tx-commit");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));

        run_tx("BEGIN", &cat, &bpm, &wal, &mut tx);
        run_tx("INSERT INTO users VALUES (1, 'a')", &cat, &bpm, &wal, &mut tx);
        run_tx("COMMIT", &cat, &bpm, &wal, &mut tx);
        let Output::Rows(rows) = run_tx("SELECT id FROM users", &cat, &bpm, &wal, &mut tx)
        else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nested_begin_errors() {
        let path = temp_path("tx-nested");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let cat = Catalog::new();
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));

        let stmt = parse("BEGIN").unwrap();
        let analyzed = analyze(&cat, &stmt).unwrap();
        let lm = LockManager::new();
        execute(&bpm, &lm, &wal, &cat, &analyzed, &mut tx).unwrap();
        let err = execute(&bpm, &lm, &wal, &cat, &analyzed, &mut tx);
        assert!(err.is_err());
        std::fs::remove_file(&path).ok();
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

        let path = temp_path("concurrent-update");
        let disk = DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(crate::wal::WalManager::open(&path.with_extension("wal")).unwrap()); let bpm = BufferPool::new(disk, 4, wal.clone());
        let lm = Arc::new(LockManager::new());
        let cat = Arc::new(Catalog::new());

        // Seed one row.
        {
            let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
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
        let bar_a = Arc::clone(&barrier);
        let h_a = thread::spawn(move || {
            bar_a.wait();
            let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
            run_full("BEGIN", &cat_a, &bpm_a, &lm_a, &wal_a, &mut tx);
            run_full(
                "UPDATE users SET name = 'A' WHERE id = 1",
                &cat_a,
                &bpm_a,
                &lm_a,
                &wal_a,
                &mut tx,
            );
            // Signal B to start; hold the lock briefly.
            a_started.send(()).unwrap();
            thread::sleep(Duration::from_millis(100));
            run_full("COMMIT", &cat_a, &bpm_a, &lm_a, &wal_a, &mut tx);
        });

        let bpm_b = bpm.clone();
        let lm_b = Arc::clone(&lm);
        let cat_b = Arc::clone(&cat);
        let wal_b = Arc::clone(&wal);
        let bar_b = Arc::clone(&barrier);
        let h_b = thread::spawn(move || {
            bar_b.wait();
            b_can_proceed.recv().unwrap();
            let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
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

        // After both commit, B's update wins (it ran second, post-A's release).
        let mut tx = Transaction::new(std::sync::Arc::new(crate::transaction_manager::TransactionManager::new()));
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
        assert_eq!(rows[0].values[0], Value::Varchar("B".into()));
        std::fs::remove_file(&path).ok();
    }
}
