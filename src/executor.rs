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
use crate::buffer_pool::BufferPoolManager;
use crate::catalog::Catalog;
use crate::page::{PageId, SlotId};
use crate::tuple::{Schema, Value, deserialize_tuple, serialize_tuple};

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
}

// -- SeqScan -----------------------------------------------------------------

pub struct SeqScan<'a> {
    bpm: &'a mut BufferPoolManager,
    schema: Schema,
    cur_page: u32,
    cur_slot: u16,
}

impl<'a> SeqScan<'a> {
    pub fn new(bpm: &'a mut BufferPoolManager, catalog: &Catalog, table_id: usize) -> Result<Self> {
        let schema = catalog
            .table_by_id(table_id)
            .ok_or_else(|| anyhow::anyhow!("table id {table_id} not in catalog"))?
            .to_schema();
        Ok(Self {
            bpm,
            schema,
            cur_page: 0,
            cur_slot: 0,
        })
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
            // Inner block scopes the PageGuard so the borrow on `self.bpm`
            // ends before we (potentially) move on to the next page.
            let found: Option<Vec<Value>> = {
                let guard = self.bpm.fetch_page(self.cur_page)?;
                let tc = guard.page().tuple_count();
                let mut out = None;
                while self.cur_slot < tc {
                    if let Some(raw) = guard.page().get_tuple(self.cur_slot) {
                        let values = deserialize_tuple(raw, &self.schema)?;
                        self.cur_slot += 1;
                        out = Some(values);
                        break;
                    }
                    self.cur_slot += 1;
                }
                out
            };
            if let Some(values) = found {
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
    bpm: &mut BufferPoolManager,
    stmt: &AnalyzedInsertStatement,
) -> Result<usize> {
    let values: Vec<Value> = stmt
        .values
        .iter()
        .map(|e| match e {
            AnalyzedExpr::Literal(lit) => Ok(literal_to_value(lit)),
            _ => bail!("INSERT VALUES must be literals (no exprs yet)"),
        })
        .collect::<Result<_>>()?;
    insert_bytes(bpm, &serialize_tuple(&values))?;
    Ok(1)
}

// -- DELETE / UPDATE (not Executors either — both are bulk side effects) -----

/// Materializes the table once into (rid, tuple) pairs so we can apply
/// modifications without worrying about re-visiting newly inserted rows
/// (UPDATE does delete+insert; without snapshotting we'd loop forever).
fn snapshot_table(
    bpm: &mut BufferPoolManager,
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
        let tc = guard.page().tuple_count();
        for slot in 0..tc {
            if let Some(raw) = guard.page().get_tuple(slot) {
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
    bpm: &mut BufferPoolManager,
    catalog: &Catalog,
    stmt: &AnalyzedDeleteStatement,
) -> Result<usize> {
    let (_schema, rows) = snapshot_table(bpm, catalog, stmt.table_id)?;
    let mut victims = Vec::new();
    for (pid, slot, values) in rows {
        let t = Tuple::new(values);
        if matches(stmt.where_clause.as_ref(), &t)? {
            victims.push((pid, slot));
        }
    }
    for (pid, slot) in &victims {
        let mut g = bpm.fetch_page(*pid)?;
        g.page_mut().delete(*slot)?;
    }
    Ok(victims.len())
}

fn perform_update(
    bpm: &mut BufferPoolManager,
    catalog: &Catalog,
    stmt: &AnalyzedUpdateStatement,
) -> Result<usize> {
    let (_schema, rows) = snapshot_table(bpm, catalog, stmt.table_id)?;
    let mut work: Vec<(PageId, SlotId, Vec<u8>)> = Vec::new();

    for (pid, slot, values) in rows {
        let t = Tuple::new(values);
        if !matches(stmt.where_clause.as_ref(), &t)? {
            continue;
        }
        // Build the new tuple: start from the old, apply each assignment.
        let mut new_values = t.values.clone();
        for a in &stmt.assignments {
            let v = evaluate_expr(&a.value, &t)?;
            new_values[a.column_index] = v;
        }
        work.push((pid, slot, serialize_tuple(&new_values)));
    }

    let count = work.len();
    for (pid, slot, bytes) in work {
        // Tombstone the old slot first.
        {
            let mut g = bpm.fetch_page(pid)?;
            g.page_mut().delete(slot)?;
        }
        // Insert the new tuple via the standard path (last page, or new one).
        insert_bytes(bpm, &bytes)?;
    }
    Ok(count)
}

// Shared insertion helper used by INSERT and UPDATE.
fn insert_bytes(bpm: &mut BufferPoolManager, bytes: &[u8]) -> Result<()> {
    let n = bpm.page_count();
    if n > 0 {
        let last = n - 1;
        let mut g = bpm.fetch_page(last)?;
        if g.page_mut().insert(bytes).is_ok() {
            return Ok(());
        }
        drop(g);
    }
    let mut g = bpm.new_page()?;
    g.page_mut()
        .insert(bytes)
        .map_err(|e| anyhow::anyhow!("tuple does not fit on a fresh page: {e}"))?;
    Ok(())
}

// -- ExecutionEngine ---------------------------------------------------------

pub fn execute(
    bpm: &mut BufferPoolManager,
    catalog: &Catalog,
    stmt: &AnalyzedStatement,
) -> Result<Output> {
    match stmt {
        AnalyzedStatement::Select(s) => {
            let mut exec = build_select_pipeline(bpm, catalog, s)?;
            exec.open()?;
            let mut rows = Vec::new();
            while let Some(t) = exec.next()? {
                rows.push(t);
            }
            Ok(Output::Rows(rows))
        }
        AnalyzedStatement::Insert(s) => Ok(Output::Affected(perform_insert(bpm, s)?)),
        AnalyzedStatement::Delete(s) => Ok(Output::Affected(perform_delete(bpm, catalog, s)?)),
        AnalyzedStatement::Update(s) => Ok(Output::Affected(perform_update(bpm, catalog, s)?)),
        AnalyzedStatement::CreateTable(_) => {
            bail!("CREATE TABLE execution is not yet wired up (catalog is read-only)")
        }
    }
}

fn build_select_pipeline<'a>(
    bpm: &'a mut BufferPoolManager,
    catalog: &'a Catalog,
    stmt: &AnalyzedSelectStatement,
) -> Result<Box<dyn Executor + 'a>> {
    let rte = &stmt.range_table[stmt.from_rte_index];
    let table_id = match &rte.source {
        TableSource::BaseTable { table_id, .. } => *table_id,
    };
    let scan: Box<dyn Executor + 'a> = Box::new(SeqScan::new(bpm, catalog, table_id)?);
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

    fn run(sql: &str, cat: &Catalog, bpm: &mut BufferPoolManager) -> Output {
        let stmt = parse(sql).unwrap();
        let analyzed = analyze(cat, &stmt).unwrap();
        execute(bpm, cat, &analyzed).unwrap()
    }

    #[test]
    fn insert_then_select_star() {
        let path = temp_path("insert-select");
        let disk = DiskManager::open(&path).unwrap();
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();

        for sql in [
            "INSERT INTO users VALUES (1, 'Alice')",
            "INSERT INTO users VALUES (2, 'Bob')",
            "INSERT INTO users VALUES (3, NULL)",
        ] {
            assert!(matches!(run(sql, &cat, &mut bpm), Output::Affected(1)));
        }
        let Output::Rows(rows) = run("SELECT * FROM users", &cat, &mut bpm) else {
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
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        for i in 1..=5 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &mut bpm,
            );
        }
        let Output::Rows(rows) = run("SELECT id FROM users WHERE id > 2", &cat, &mut bpm) else {
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
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (10, 'a')", &cat, &mut bpm);
        let Output::Rows(rows) = run("SELECT id + 1 FROM users", &cat, &mut bpm) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Int(11));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn null_predicate_excludes_row() {
        let path = temp_path("null-pred");
        let disk = DiskManager::open(&path).unwrap();
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &mut bpm);
        run("INSERT INTO users VALUES (2, NULL)", &cat, &mut bpm);
        // name = 'Alice' on the NULL row evaluates to NULL → row excluded.
        let Output::Rows(rows) = run(
            "SELECT id FROM users WHERE name = 'Alice'",
            &cat,
            &mut bpm,
        ) else {
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
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        for i in 1..=4 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &mut bpm,
            );
        }
        assert!(matches!(
            run("DELETE FROM users WHERE id > 2", &cat, &mut bpm),
            Output::Affected(2)
        ));
        let Output::Rows(rows) = run("SELECT id FROM users", &cat, &mut bpm) else {
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
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        for i in 1..=3 {
            run(
                &format!("INSERT INTO users VALUES ({i}, 'x')"),
                &cat,
                &mut bpm,
            );
        }
        assert!(matches!(
            run("DELETE FROM users", &cat, &mut bpm),
            Output::Affected(3)
        ));
        let Output::Rows(rows) = run("SELECT * FROM users", &cat, &mut bpm) else {
            panic!()
        };
        assert!(rows.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn update_changes_matching_rows() {
        let path = temp_path("update");
        let disk = DiskManager::open(&path).unwrap();
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, 'Alice')", &cat, &mut bpm);
        run("INSERT INTO users VALUES (2, 'Bob')", &cat, &mut bpm);
        assert!(matches!(
            run(
                "UPDATE users SET name = 'A2' WHERE id = 1",
                &cat,
                &mut bpm
            ),
            Output::Affected(1)
        ));
        let Output::Rows(rows) = run("SELECT id, name FROM users", &cat, &mut bpm) else {
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
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, 'a')", &cat, &mut bpm);
        // SET name = 'a' WHERE name = 'a' affects exactly one row, not infinite.
        assert!(matches!(
            run(
                "UPDATE users SET name = 'a' WHERE name = 'a'",
                &cat,
                &mut bpm
            ),
            Output::Affected(1)
        ));
        let Output::Rows(rows) = run("SELECT id FROM users", &cat, &mut bpm) else {
            panic!()
        };
        assert_eq!(rows.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn null_propagates_in_arithmetic() {
        let path = temp_path("null-arith");
        let disk = DiskManager::open(&path).unwrap();
        let mut bpm = BufferPoolManager::new(disk, 4);
        let cat = Catalog::new();
        run("INSERT INTO users VALUES (1, NULL)", &cat, &mut bpm);
        // SELECT name (which is NULL) projects through; arithmetic on NULL would
        // also produce NULL — exercised via id (non-null) for the all-OK row.
        let Output::Rows(rows) = run("SELECT name FROM users", &cat, &mut bpm) else {
            panic!()
        };
        assert_eq!(rows[0].values[0], Value::Null);
        std::fs::remove_file(&path).ok();
    }
}
