//! Resolves a parsed AST against the catalog, producing a typed AnalyzedStatement.
//!
//! Uses a Postgres-style range table + scope model so that subqueries / joins
//! can be added without restructuring later. day05 only ever produces a single
//! base-table RTE per SELECT.

// Many analyzed-AST fields (column_name, rte_index, table_name, ...) are public
// surface that later days will read (wire protocol, errors, planner). The
// day06 executor doesn't yet consume all of them.
#![allow(dead_code)]

use std::mem;

use anyhow::{Result, bail};

use crate::ast::{
    self, AlterTableAction, AlterTableStatement, AnalyzeStatement, BinaryOperator,
    CopyStatement, CreateIndexStatement, CreateSequenceStatement, CreateTableStatement,
    DeleteStatement, DropIndexStatement, DropSequenceStatement, DropTableStatement, Expr,
    FromClause, FuncArgs, InsertStatement, JoinType, Literal, OrderDir, SelectColumn,
    SelectStatement, Statement, TableRef, TruncateStatement, UnaryOperator, UpdateStatement,
    VacuumStatement,
};
use crate::catalog::Catalog;
use crate::tuple::DataType;

#[derive(Debug, Clone)]
pub enum TableSource {
    BaseTable {
        table_id: usize,
        table_name: String,
    },
}

#[derive(Debug, Clone)]
pub struct OutputColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

#[derive(Debug, Clone)]
pub struct RangeTableEntry {
    pub rte_index: usize,
    pub source: TableSource,
    pub output_columns: Vec<OutputColumn>,
    /// Offset of this RTE's columns inside the joined-output tuple. The
    /// executor's flat tuple is the concat of all RTEs in FROM order.
    pub flat_offset: usize,
}

#[derive(Debug, Clone)]
pub enum AnalyzedStatement {
    Select(AnalyzedSelectStatement),
    Insert(AnalyzedInsertStatement),
    Delete(AnalyzedDeleteStatement),
    Update(AnalyzedUpdateStatement),
    CreateTable(AnalyzedCreateTableStatement),
    CreateIndex(AnalyzedCreateIndexStatement),
    DropTable(AnalyzedDropTableStatement),
    DropIndex(AnalyzedDropIndexStatement),
    Truncate(AnalyzedTruncateStatement),
    AlterTableAddIndex(AnalyzedCreateIndexStatement),
    CreateSequence(AnalyzedCreateSequenceStatement),
    DropSequence(AnalyzedDropSequenceStatement),
    Vacuum(AnalyzedVacuumStatement),
    /// ANALYZE on its own — parse it, no-op execute (real stats arrive
    /// in Phase 9).
    AnalyzeNoop,
    Copy(AnalyzedCopyStatement),
    Begin,
    Commit,
    Rollback,
    Checkpoint,
}

#[derive(Debug, Clone)]
pub struct AnalyzedDeleteStatement {
    pub table_id: usize,
    pub table_name: String,
    pub where_clause: Option<AnalyzedExpr>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedUpdateStatement {
    pub table_id: usize,
    pub table_name: String,
    pub assignments: Vec<AnalyzedAssignment>,
    pub where_clause: Option<AnalyzedExpr>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedAssignment {
    pub column_index: usize,
    pub column_name: String,
    pub value: AnalyzedExpr,
}

#[derive(Debug, Clone)]
pub struct AnalyzedSelectStatement {
    pub range_table: Vec<RangeTableEntry>,
    pub from: AnalyzedFrom,
    /// Pre-aggregate filter (evaluated against input tuples).
    pub where_clause: Option<AnalyzedExpr>,
    /// Set when the SELECT has GROUP BY or any aggregate function in
    /// SELECT/HAVING. The pipeline becomes ... → HashAggregate → HAVING → Project.
    pub aggregation: Option<AnalyzedAggregation>,
    /// Evaluated against:
    ///   - input tuples if `aggregation` is None
    ///   - post-aggregate tuples ([keys..., agg_results...]) otherwise.
    pub select_items: Vec<AnalyzedSelectItem>,
    /// Same evaluation context as `select_items` (only meaningful with aggregation).
    pub having: Option<AnalyzedExpr>,
    /// Sort keys evaluated against the same context as `select_items`.
    pub order_by: Vec<AnalyzedOrderBy>,
    /// Cap on rows after Sort. None ⇒ no cap.
    pub limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedOrderBy {
    pub expr: AnalyzedExpr,
    pub dir: OrderDir,
}

#[derive(Debug, Clone)]
pub struct AnalyzedAggregation {
    /// GROUP BY keys, evaluated against input tuples.
    pub group_keys: Vec<AnalyzedExpr>,
    /// Aggregate functions in invocation order; their args reference input tuples.
    pub aggregates: Vec<AnalyzedAggregate>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedAggregate {
    pub kind: AggKind,
    pub arg: AggArg,
    pub result_type: DataType,
}

#[derive(Debug, Clone)]
pub enum AggArg {
    /// COUNT(*) — count every input row.
    Star,
    /// Aggregate over a per-row expression (`SUM(quantity)` etc).
    Expr(Box<AnalyzedExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl AggKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "COUNT" => Some(Self::Count),
            "SUM" => Some(Self::Sum),
            "AVG" => Some(Self::Avg),
            "MIN" => Some(Self::Min),
            "MAX" => Some(Self::Max),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Count => "COUNT",
            Self::Sum => "SUM",
            Self::Avg => "AVG",
            Self::Min => "MIN",
            Self::Max => "MAX",
        }
    }
}

/// Tree of joined sources. Mirrors the AST `FromClause` but carries
/// resolved `rte_index` references and analyzed ON predicates.
#[derive(Debug, Clone)]
pub enum AnalyzedFrom {
    /// `SELECT expr;` with no FROM — yields a single all-empty input row.
    Empty,
    Table {
        rte_index: usize,
    },
    Join {
        left: Box<AnalyzedFrom>,
        right_rte_index: usize,
        join_type: JoinType,
        on: AnalyzedExpr,
    },
}

#[derive(Debug, Clone)]
pub struct AnalyzedSelectItem {
    pub expr: AnalyzedExpr,
    pub alias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedInsertStatement {
    pub table_id: usize,
    pub table_name: String,
    /// One Vec<AnalyzedExpr> per row.
    pub rows: Vec<Vec<AnalyzedExpr>>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedCreateTableStatement {
    pub table_name: String,
    pub columns: Vec<AnalyzedColumnDef>,
    /// Column index of the single-column PRIMARY KEY, if the CREATE TABLE
    /// declared one. Multi-column PKs are rejected at the analyzer for now;
    /// none ⇒ no automatic unique index.
    pub primary_key_column: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedCreateIndexStatement {
    pub name: String,
    pub table_id: usize,
    pub table_name: String,
    pub column_index: usize,
    pub column_name: String,
    pub data_type: DataType,
    /// Reject duplicate keys at insert time (PRIMARY KEY / UNIQUE / explicit
    /// `CREATE UNIQUE INDEX`). Plain `CREATE INDEX` leaves this false.
    pub is_unique: bool,
}

#[derive(Debug, Clone)]
pub struct AnalyzedDropTableStatement {
    /// Resolved (table_id, name) pairs; only includes tables that actually
    /// existed (the rest were silently skipped under IF EXISTS).
    pub tables: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedDropIndexStatement {
    pub index_id: usize,
    pub name: String,
    pub table_id: usize,
}

#[derive(Debug, Clone)]
pub struct AnalyzedTruncateStatement {
    pub tables: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedCreateSequenceStatement {
    pub name: String,
    pub increment: i64,
    pub start_value: i64,
    pub min_value: i64,
    pub max_value: i64,
}

#[derive(Debug, Clone)]
pub struct AnalyzedDropSequenceStatement {
    pub seqs: Vec<(usize, String, crate::page::PageId)>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedVacuumStatement {
    /// Resolved (table_id, name). Empty source means "all user tables".
    pub tables: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedCopyStatement {
    pub table_id: usize,
    pub table_name: String,
    /// One entry per *table* column. `Some(i)` means this column receives
    /// the i-th field of each COPY data line; `None` means it gets NULL.
    /// (Same shape as INSERT's column-list mapping.)
    pub column_to_field: Vec<Option<usize>>,
    /// Number of fields each CopyData line must contain.
    pub field_count: usize,
    /// Per-table-column data type, used by the executor to decode each
    /// text-format field into a `Value`.
    pub column_types: Vec<DataType>,
    pub column_nullable: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    /// Serialized DEFAULT expression (Expr::Display form). The executor
    /// hands it to pg_attribute.default_text; the analyzer re-parses it
    /// at INSERT time to apply.
    pub default_text: Option<String>,
}

#[derive(Debug, Clone)]
pub enum AnalyzedExpr {
    Literal(AnalyzedLiteral),
    ColumnRef(AnalyzedColumnRef),
    BinaryOp {
        left: Box<AnalyzedExpr>,
        op: BinaryOperator,
        right: Box<AnalyzedExpr>,
        result_type: DataType,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: Box<AnalyzedExpr>,
        result_type: DataType,
    },
    /// `expr IS [NOT] NULL`. Always returns Bool, even on NULL operand.
    IsNull {
        expr: Box<AnalyzedExpr>,
        negated: bool,
    },
    /// `now()` / `current_timestamp` — placeholder for the transaction
    /// start timestamp. Replaced with a Literal at the executor entry once
    /// `tx.start_ts()` is known. Always typed as TIMESTAMP.
    Now,
    /// `nextval('seq')` / `setval('seq', n)` placeholder. Resolved against
    /// the live sequence relation at evaluation time. Always returns INT.
    SequenceCall {
        kind: SequenceFnKind,
        seq_id: usize,
        seq_page_id: crate::page::PageId,
        increment: i64,
        start_value: i64,
        /// For setval(seq, n) and setval(seq, n, is_called).
        setval_arg: Option<i64>,
        setval_is_called: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceFnKind {
    Nextval,
    Setval,
}

#[derive(Debug, Clone)]
pub struct AnalyzedColumnRef {
    pub rte_index: usize,
    pub column_index: usize,
    pub column_name: String,
    pub data_type: DataType,
}

#[derive(Debug, Clone)]
pub struct AnalyzedLiteral {
    pub value: LiteralValue,
    /// `None` for SQL NULL — type is determined by context (e.g. INSERT column).
    pub data_type: Option<DataType>,
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    /// PG-epoch microseconds (already parsed by lexer/parser).
    Timestamp(i64),
    Date(i32),
    Time(i64),
    Interval { months: i32, days: i32, micros: i64 },
}

impl AnalyzedExpr {
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            AnalyzedExpr::Literal(l) => l.data_type,
            AnalyzedExpr::ColumnRef(c) => Some(c.data_type),
            AnalyzedExpr::BinaryOp { result_type, .. }
            | AnalyzedExpr::UnaryOp { result_type, .. } => Some(*result_type),
            AnalyzedExpr::IsNull { .. } => Some(DataType::Bool),
            AnalyzedExpr::Now => Some(DataType::Timestamp),
            AnalyzedExpr::SequenceCall { .. } => Some(DataType::Int),
        }
    }
}

#[derive(Debug, Clone)]
struct ScopeEntry {
    /// The name visible in qualified refs — alias if present, otherwise table name.
    visible_name: String,
    rte_index: usize,
}

struct Analyzer<'a> {
    catalog: &'a Catalog,
    range_table: Vec<RangeTableEntry>,
    scopes: Vec<Vec<ScopeEntry>>,
}

impl<'a> Analyzer<'a> {
    fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            range_table: Vec::new(),
            scopes: Vec::new(),
        }
    }

    fn add_rte(
        &mut self,
        source: TableSource,
        output_columns: Vec<OutputColumn>,
        flat_offset: usize,
    ) -> usize {
        let idx = self.range_table.len();
        self.range_table.push(RangeTableEntry {
            rte_index: idx,
            source,
            output_columns,
            flat_offset,
        });
        idx
    }

    fn resolve_column(
        &self,
        qualifier: Option<&str>,
        name: &str,
    ) -> Result<AnalyzedColumnRef> {
        // Qualified: only consider the matching scope entry.
        if let Some(q) = qualifier {
            for scope in self.scopes.iter().rev() {
                for entry in scope {
                    if entry.visible_name != q {
                        continue;
                    }
                    let rte = &self.range_table[entry.rte_index];
                    let (idx, col) = rte
                        .output_columns
                        .iter()
                        .enumerate()
                        .find(|(_, c)| c.name == name)
                        .ok_or_else(|| {
                            anyhow::anyhow!("column '{q}.{name}' not found")
                        })?;
                    return Ok(AnalyzedColumnRef {
                        rte_index: entry.rte_index,
                        column_index: rte.flat_offset + idx,
                        column_name: name.to_string(),
                        data_type: col.data_type,
                    });
                }
            }
            bail!("table or alias '{q}' not in scope");
        }

        // Unqualified: search all scopes; reject ambiguity.
        let mut hit: Option<AnalyzedColumnRef> = None;
        for scope in self.scopes.iter().rev() {
            for entry in scope {
                let rte = &self.range_table[entry.rte_index];
                if let Some((idx, col)) = rte
                    .output_columns
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.name == name)
                {
                    let cand = AnalyzedColumnRef {
                        rte_index: entry.rte_index,
                        column_index: rte.flat_offset + idx,
                        column_name: name.to_string(),
                        data_type: col.data_type,
                    };
                    if hit.is_some() {
                        bail!("column '{name}' is ambiguous");
                    }
                    hit = Some(cand);
                }
            }
        }
        hit.ok_or_else(|| anyhow::anyhow!("column '{name}' not found"))
    }

    /// Analyze an expression that runs against per-row input tuples (WHERE,
    /// GROUP BY exprs, aggregate-function arguments). Aggregate calls are
    /// rejected here — they'd be nonsensical on a single row.
    fn analyze_expr(&self, e: &Expr) -> Result<AnalyzedExpr> {
        match e {
            Expr::Literal(l) => Ok(AnalyzedExpr::Literal(literal_to_analyzed(l))),
            Expr::Column { qualifier, name } => Ok(AnalyzedExpr::ColumnRef(
                self.resolve_column(qualifier.as_deref(), name)?,
            )),
            Expr::BinaryOp { left, op, right } => {
                let l = self.analyze_expr(left)?;
                let r = self.analyze_expr(right)?;
                let result_type = infer_binary_type(*op, l.data_type(), r.data_type());
                Ok(AnalyzedExpr::BinaryOp {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                    result_type,
                })
            }
            Expr::UnaryOp { op, expr } => {
                let inner = self.analyze_expr(expr)?;
                Ok(AnalyzedExpr::UnaryOp {
                    op: *op,
                    expr: Box::new(inner),
                    result_type: infer_unary_type(*op),
                })
            }
            Expr::IsNull { expr, negated } => {
                let inner = self.analyze_expr(expr)?;
                Ok(AnalyzedExpr::IsNull {
                    expr: Box::new(inner),
                    negated: *negated,
                })
            }
            Expr::FuncCall { name, args } => {
                if let Some(builtin) = self.analyze_scalar_builtin(name, args)? {
                    return Ok(builtin);
                }
                if AggKind::from_name(name).is_some() {
                    bail!("aggregate '{name}' not allowed here");
                }
                bail!("unknown function '{name}'")
            }
        }
    }

    /// Add an RTE for `t`, register it in the current scope, and return its
    /// rte_index. `current_offset` is the next free flat-tuple position.
    fn intro_table(&mut self, t: &TableRef, current_offset: usize) -> Result<usize> {
        let (table_id, table) = self
            .catalog
            .find_table(&t.name)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", t.name))?;
        let output_columns: Vec<OutputColumn> = table
            .columns
            .iter()
            .map(|c| OutputColumn {
                name: c.name.clone(),
                data_type: c.data_type,
                nullable: c.nullable,
            })
            .collect();
        let rte_index = self.add_rte(
            TableSource::BaseTable {
                table_id,
                table_name: t.name.clone(),
            },
            output_columns,
            current_offset,
        );
        let visible_name = t.alias.clone().unwrap_or_else(|| t.name.clone());
        // Reject duplicate names within the same SELECT scope.
        let scope = self.scopes.last_mut().expect("scope pushed");
        if scope.iter().any(|e| e.visible_name == visible_name) {
            bail!("table name or alias '{visible_name}' duplicated in FROM");
        }
        scope.push(ScopeEntry {
            visible_name,
            rte_index,
        });
        Ok(rte_index)
    }

    /// Recursively analyze a FROM tree. Returns (analyzed-from, total flat width).
    fn analyze_from(
        &mut self,
        from: &FromClause,
        offset: usize,
    ) -> Result<(AnalyzedFrom, usize)> {
        match from {
            FromClause::Empty => Ok((AnalyzedFrom::Empty, offset)),
            FromClause::Table(t) => {
                let rte_index = self.intro_table(t, offset)?;
                let width = self.range_table[rte_index].output_columns.len();
                Ok((AnalyzedFrom::Table { rte_index }, offset + width))
            }
            FromClause::Join {
                left,
                right,
                join_type,
                on,
            } => {
                let (left_a, after_left) = self.analyze_from(left, offset)?;
                let right_rte = self.intro_table(right, after_left)?;
                let total = after_left + self.range_table[right_rte].output_columns.len();
                // ON predicate is analyzed in the scope that includes both sides.
                let on_a = self.analyze_expr(on)?;
                if !matches!(on_a.data_type(), Some(DataType::Bool) | None) {
                    bail!("JOIN ON must be boolean, got {:?}", on_a.data_type());
                }
                Ok((
                    AnalyzedFrom::Join {
                        left: Box::new(left_a),
                        right_rte_index: right_rte,
                        join_type: *join_type,
                        on: on_a,
                    },
                    total,
                ))
            }
        }
    }

    fn analyze_select(&mut self, s: &SelectStatement) -> Result<AnalyzedSelectStatement> {
        // Push scope BEFORE walking FROM so intro_table can register entries.
        self.scopes.push(Vec::new());
        let (from_a, _total_width) = self.analyze_from(&s.from, 0)?;

        // Build (alias, expr) map for ORDER BY alias resolution. Built from
        // the AST so it's usable before we analyze the SELECT items.
        let select_alias_map: Vec<(String, Expr)> = s
            .columns
            .iter()
            .filter_map(|c| match c {
                SelectColumn::Expr {
                    expr,
                    alias: Some(a),
                } => Some((a.clone(), expr.clone())),
                _ => None,
            })
            .collect();

        // WHERE: pre-aggregate, no aggregates allowed.
        let where_clause = match &s.where_clause {
            Some(e) => {
                let a = self.analyze_expr(e)?;
                if !matches!(a.data_type(), Some(DataType::Bool) | None) {
                    bail!("WHERE clause must be boolean, got {:?}", a.data_type());
                }
                Some(a)
            }
            None => None,
        };

        // GROUP BY: each expression analyzed against input tuples. DISTINCT
        // is desugared into "group by every projected expression" so the
        // existing HashAggregate path handles it for free.
        let mut group_by_ast: Vec<Expr> = s.group_by.clone();
        if s.distinct {
            if !s.group_by.is_empty() {
                bail!("DISTINCT combined with explicit GROUP BY is not supported");
            }
            for c in &s.columns {
                match c {
                    SelectColumn::Asterisk => {
                        bail!("`SELECT DISTINCT *` is not supported");
                    }
                    SelectColumn::Expr { expr, .. } => {
                        if contains_aggregate(expr) {
                            bail!("DISTINCT combined with aggregate functions is not supported");
                        }
                        group_by_ast.push(expr.clone());
                    }
                }
            }
        }
        let group_keys: Vec<AnalyzedExpr> = group_by_ast
            .iter()
            .map(|e| self.analyze_expr(e))
            .collect::<Result<_>>()?;

        // Detect whether aggregation is needed: GROUP BY present OR any
        // aggregate function found in SELECT/HAVING.
        let has_aggregate_call = s.columns.iter().any(|c| match c {
            SelectColumn::Asterisk => false,
            SelectColumn::Expr { expr: e, .. } => contains_aggregate(e),
        }) || s.having.as_ref().map(contains_aggregate).unwrap_or(false);
        let needs_aggregation = !group_keys.is_empty() || has_aggregate_call;

        let (select_items, having, aggregation, order_by) = if needs_aggregation {
            let mut aggs: Vec<AnalyzedAggregate> = Vec::new();
            let mut select_items = Vec::new();
            for c in &s.columns {
                match c {
                    SelectColumn::Asterisk => {
                        bail!("`SELECT *` with GROUP BY/aggregates is not supported");
                    }
                    SelectColumn::Expr { expr: e, alias } => {
                        let rewritten = self.analyze_post_agg(e, &group_keys, &mut aggs)?;
                        select_items.push(AnalyzedSelectItem {
                            expr: rewritten,
                            alias: alias.clone(),
                        });
                    }
                }
            }
            let having = match &s.having {
                Some(e) => {
                    let a = self.analyze_post_agg(e, &group_keys, &mut aggs)?;
                    if !matches!(a.data_type(), Some(DataType::Bool) | None) {
                        bail!("HAVING clause must be boolean, got {:?}", a.data_type());
                    }
                    Some(a)
                }
                None => None,
            };
            // ORDER BY uses the same post-aggregate context, but bare
            // column refs may reference SELECT aliases.
            let mut order_by = Vec::new();
            for ob in &s.order_by {
                let substituted = substitute_aliases(&ob.expr, &select_alias_map);
                let expr = self.analyze_post_agg(&substituted, &group_keys, &mut aggs)?;
                order_by.push(AnalyzedOrderBy { expr, dir: ob.dir });
            }
            (
                select_items,
                having,
                Some(AnalyzedAggregation {
                    group_keys,
                    aggregates: aggs,
                }),
                order_by,
            )
        } else {
            // No aggregation: SELECT operates on input tuples directly.
            if s.having.is_some() {
                bail!("HAVING requires GROUP BY or an aggregate function");
            }
            let mut select_items = Vec::new();
            for c in &s.columns {
                match c {
                    SelectColumn::Asterisk => {
                        let scope = self.scopes.last().expect("scope pushed").clone();
                        for entry in scope {
                            let rte = &self.range_table[entry.rte_index];
                            let off = rte.flat_offset;
                            for (i, oc) in rte.output_columns.iter().enumerate() {
                                select_items.push(AnalyzedSelectItem {
                                    expr: AnalyzedExpr::ColumnRef(AnalyzedColumnRef {
                                        rte_index: entry.rte_index,
                                        column_index: off + i,
                                        column_name: oc.name.clone(),
                                        data_type: oc.data_type,
                                    }),
                                    alias: None,
                                });
                            }
                        }
                    }
                    SelectColumn::Expr { expr: e, alias } => {
                        select_items.push(AnalyzedSelectItem {
                            expr: self.analyze_expr(e)?,
                            alias: alias.clone(),
                        });
                    }
                }
            }
            // ORDER BY against per-row tuples, with SELECT alias substitution.
            let mut order_by = Vec::new();
            for ob in &s.order_by {
                let substituted = substitute_aliases(&ob.expr, &select_alias_map);
                order_by.push(AnalyzedOrderBy {
                    expr: self.analyze_expr(&substituted)?,
                    dir: ob.dir,
                });
            }
            (select_items, None, None, order_by)
        };

        self.scopes.pop();

        Ok(AnalyzedSelectStatement {
            range_table: mem::take(&mut self.range_table),
            from: from_a,
            where_clause,
            aggregation,
            select_items,
            having,
            order_by,
            limit: s.limit,
        })
    }

    /// Analyze an expression that runs against post-aggregate tuples
    /// `[group_keys..., agg_results...]`. Aggregate calls are extracted into
    /// `aggs` and replaced with ColumnRefs into the post-agg position.
    /// Bare column refs must match a `group_keys` entry — anything else is
    /// rejected as not grouped.
    fn analyze_post_agg(
        &self,
        e: &Expr,
        group_keys: &[AnalyzedExpr],
        aggs: &mut Vec<AnalyzedAggregate>,
    ) -> Result<AnalyzedExpr> {
        match e {
            Expr::Literal(l) => Ok(AnalyzedExpr::Literal(literal_to_analyzed(l))),
            Expr::Column { qualifier, name } => {
                let resolved = self.resolve_column(qualifier.as_deref(), name)?;
                let resolved_expr = AnalyzedExpr::ColumnRef(resolved.clone());
                let pos = group_keys
                    .iter()
                    .position(|gk| same_expr(gk, &resolved_expr))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "column '{name}' must appear in GROUP BY or in an aggregate function"
                        )
                    })?;
                Ok(AnalyzedExpr::ColumnRef(AnalyzedColumnRef {
                    rte_index: usize::MAX,
                    column_index: pos,
                    column_name: name.clone(),
                    data_type: resolved.data_type,
                }))
            }
            Expr::BinaryOp { left, op, right } => {
                let l = self.analyze_post_agg(left, group_keys, aggs)?;
                let r = self.analyze_post_agg(right, group_keys, aggs)?;
                let result_type = infer_binary_type(*op, l.data_type(), r.data_type());
                Ok(AnalyzedExpr::BinaryOp {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                    result_type,
                })
            }
            Expr::UnaryOp { op, expr } => {
                let inner = self.analyze_post_agg(expr, group_keys, aggs)?;
                Ok(AnalyzedExpr::UnaryOp {
                    op: *op,
                    expr: Box::new(inner),
                    result_type: infer_unary_type(*op),
                })
            }
            Expr::IsNull { expr, negated } => {
                let inner = self.analyze_post_agg(expr, group_keys, aggs)?;
                Ok(AnalyzedExpr::IsNull {
                    expr: Box::new(inner),
                    negated: *negated,
                })
            }
            Expr::FuncCall { name, args } => {
                if let Some(builtin) = self.analyze_scalar_builtin(name, args)? {
                    return Ok(builtin);
                }
                let kind = AggKind::from_name(name)
                    .ok_or_else(|| anyhow::anyhow!("unknown function '{name}'"))?;
                let (analyzed_arg, arg_type) = match (kind, args) {
                    (AggKind::Count, FuncArgs::Star) => (AggArg::Star, None),
                    (_, FuncArgs::Star) => bail!("only COUNT supports `*`"),
                    (_, FuncArgs::Exprs(es)) => {
                        if es.len() != 1 {
                            bail!("aggregate '{name}' takes exactly one argument");
                        }
                        // Aggregate-arg cannot itself contain an aggregate.
                        if contains_aggregate(&es[0]) {
                            bail!("nested aggregate in '{name}' not allowed");
                        }
                        let a = self.analyze_expr(&es[0])?;
                        let t = a.data_type();
                        (AggArg::Expr(Box::new(a)), t)
                    }
                };
                let result_type = infer_aggregate_type(kind, arg_type)?;
                let agg_index = aggs.len();
                aggs.push(AnalyzedAggregate {
                    kind,
                    arg: analyzed_arg,
                    result_type,
                });
                // Post-agg position is group_keys.len() + agg_index.
                Ok(AnalyzedExpr::ColumnRef(AnalyzedColumnRef {
                    rte_index: usize::MAX,
                    column_index: group_keys.len() + agg_index,
                    column_name: kind.name().to_string(),
                    data_type: result_type,
                }))
            }
        }
    }

    fn analyze_insert(&self, s: &InsertStatement) -> Result<AnalyzedInsertStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;

        // Resolve the column list (or default to declaration order). For each
        // table column, `col_to_value_idx[i]` is the position in the row's
        // VALUES tuple, or None if that column should default to NULL.
        let col_to_value_idx: Vec<Option<usize>> = match &s.columns {
            None => (0..table.columns.len()).map(Some).collect(),
            Some(names) => {
                let mut indexed: Vec<Option<usize>> = vec![None; table.columns.len()];
                for (vi, want) in names.iter().enumerate() {
                    let (col_idx, _) = table
                        .columns
                        .iter()
                        .enumerate()
                        .find(|(_, c)| c.name == *want)
                        .ok_or_else(|| anyhow::anyhow!("column '{want}' not found"))?;
                    if indexed[col_idx].is_some() {
                        bail!("column '{want}' specified more than once");
                    }
                    indexed[col_idx] = Some(vi);
                }
                indexed
            }
        };
        let expected_value_count = match &s.columns {
            None => table.columns.len(),
            Some(c) => c.len(),
        };

        let mut analyzed_rows = Vec::with_capacity(s.rows.len());
        for row in &s.rows {
            if row.len() != expected_value_count {
                bail!(
                    "INSERT has {} values but {} were expected",
                    row.len(),
                    expected_value_count,
                );
            }
            let mut values = Vec::with_capacity(table.columns.len());
            for (col_idx, col) in table.columns.iter().enumerate() {
                let expr = match col_to_value_idx[col_idx] {
                    Some(vi) => self.analyze_expr(&row[vi])?,
                    None => {
                        // Apply DEFAULT if the column has one.
                        if let Some(text) = &col.default_text {
                            let parsed = crate::parser::parse_expr_str(text).map_err(|e| {
                                anyhow::anyhow!(
                                    "failed to parse stored DEFAULT for column '{}': {e}",
                                    col.name
                                )
                            })?;
                            self.analyze_expr(&parsed)?
                        } else {
                            if !col.nullable {
                                bail!(
                                    "column '{}' is not nullable and was not given a value",
                                    col.name
                                );
                            }
                            AnalyzedExpr::Literal(literal_to_analyzed(&Literal::Null))
                        }
                    }
                };
                match expr.data_type() {
                    None => {
                        if !col.nullable {
                            bail!("column '{}' is not nullable", col.name);
                        }
                    }
                    Some(t) if assignable(t, col.data_type) => {}
                    Some(t) => bail!(
                        "type mismatch for column '{}': expected {:?}, got {:?}",
                        col.name,
                        col.data_type,
                        t
                    ),
                }
                values.push(expr);
            }
            analyzed_rows.push(values);
        }

        Ok(AnalyzedInsertStatement {
            table_id,
            table_name: s.table.clone(),
            rows: analyzed_rows,
        })
    }

    fn analyze_delete(&mut self, s: &DeleteStatement) -> Result<AnalyzedDeleteStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;

        // Build a scope so WHERE expressions can reference columns.
        let output_columns: Vec<OutputColumn> = table
            .columns
            .iter()
            .map(|c| OutputColumn {
                name: c.name.clone(),
                data_type: c.data_type,
                nullable: c.nullable,
            })
            .collect();
        let rte_index = self.add_rte(
            TableSource::BaseTable {
                table_id,
                table_name: s.table.clone(),
            },
            output_columns,
            0,
        );
        self.scopes.push(vec![ScopeEntry {
            visible_name: s.table.clone(),
            rte_index,
        }]);

        let where_clause = match &s.where_clause {
            Some(e) => {
                let analyzed = self.analyze_expr(e)?;
                if !matches!(analyzed.data_type(), Some(DataType::Bool) | None) {
                    bail!("WHERE clause must be boolean, got {:?}", analyzed.data_type());
                }
                Some(analyzed)
            }
            None => None,
        };

        self.scopes.pop();

        Ok(AnalyzedDeleteStatement {
            table_id,
            table_name: s.table.clone(),
            where_clause,
        })
    }

    fn analyze_update(&mut self, s: &UpdateStatement) -> Result<AnalyzedUpdateStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;

        let output_columns: Vec<OutputColumn> = table
            .columns
            .iter()
            .map(|c| OutputColumn {
                name: c.name.clone(),
                data_type: c.data_type,
                nullable: c.nullable,
            })
            .collect();
        let rte_index = self.add_rte(
            TableSource::BaseTable {
                table_id,
                table_name: s.table.clone(),
            },
            output_columns,
            0,
        );
        self.scopes.push(vec![ScopeEntry {
            visible_name: s.table.clone(),
            rte_index,
        }]);

        let mut assignments = Vec::with_capacity(s.assignments.len());
        for a in &s.assignments {
            let (col_idx, col) = table
                .columns
                .iter()
                .enumerate()
                .find(|(_, c)| c.name == a.column)
                .ok_or_else(|| anyhow::anyhow!("column '{}' not found", a.column))?;
            let analyzed_value = self.analyze_expr(&a.value)?;
            match analyzed_value.data_type() {
                None => {
                    if !col.nullable {
                        bail!("column '{}' is not nullable", col.name);
                    }
                }
                Some(t) if assignable(t, col.data_type) => {}
                Some(t) => bail!(
                    "type mismatch in UPDATE for column '{}': expected {:?}, got {:?}",
                    col.name,
                    col.data_type,
                    t
                ),
            }
            assignments.push(AnalyzedAssignment {
                column_index: col_idx,
                column_name: a.column.clone(),
                value: analyzed_value,
            });
        }

        let where_clause = match &s.where_clause {
            Some(e) => {
                let analyzed = self.analyze_expr(e)?;
                if !matches!(analyzed.data_type(), Some(DataType::Bool) | None) {
                    bail!("WHERE clause must be boolean, got {:?}", analyzed.data_type());
                }
                Some(analyzed)
            }
            None => None,
        };

        self.scopes.pop();

        Ok(AnalyzedUpdateStatement {
            table_id,
            table_name: s.table.clone(),
            assignments,
            where_clause,
        })
    }

    fn analyze_create_sequence(
        &self,
        s: &CreateSequenceStatement,
    ) -> Result<AnalyzedCreateSequenceStatement> {
        if !s.if_not_exists && self.catalog.find_sequence(&s.name)?.is_some() {
            bail!("sequence '{}' already exists", s.name);
        }
        let increment = s.increment;
        if increment == 0 {
            bail!("sequence INCREMENT must not be zero");
        }
        let (default_min, default_max) = if increment > 0 {
            (1i64, i32::MAX as i64)
        } else {
            (i32::MIN as i64, -1i64)
        };
        let min_value = s.min_value.unwrap_or(default_min);
        let max_value = s.max_value.unwrap_or(default_max);
        let start_value = s
            .start_value
            .unwrap_or(if increment > 0 { min_value } else { max_value });
        if start_value < min_value || start_value > max_value {
            bail!("START value out of [MINVALUE, MAXVALUE] range");
        }
        Ok(AnalyzedCreateSequenceStatement {
            name: s.name.clone(),
            increment,
            start_value,
            min_value,
            max_value,
        })
    }

    fn analyze_vacuum(&self, s: &VacuumStatement) -> Result<AnalyzedVacuumStatement> {
        let mut tables = Vec::new();
        if s.tables.is_empty() {
            // `VACUUM` (no list) → every user table.
            for t in self.catalog.user_tables()? {
                tables.push((t.table_id, t.name));
            }
        } else {
            for name in &s.tables {
                let (id, _) = self
                    .catalog
                    .find_table(name)?
                    .ok_or_else(|| anyhow::anyhow!("table '{name}' not found"))?;
                tables.push((id, name.clone()));
            }
        }
        Ok(AnalyzedVacuumStatement { tables })
    }

    fn analyze_copy(&self, s: &CopyStatement) -> Result<AnalyzedCopyStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;

        let column_to_field: Vec<Option<usize>> = match &s.columns {
            None => (0..table.columns.len()).map(Some).collect(),
            Some(names) => {
                let mut indexed: Vec<Option<usize>> = vec![None; table.columns.len()];
                for (fi, want) in names.iter().enumerate() {
                    let (col_idx, _) = table
                        .columns
                        .iter()
                        .enumerate()
                        .find(|(_, c)| c.name == *want)
                        .ok_or_else(|| anyhow::anyhow!("column '{want}' not found"))?;
                    if indexed[col_idx].is_some() {
                        bail!("column '{want}' specified more than once");
                    }
                    indexed[col_idx] = Some(fi);
                }
                indexed
            }
        };
        let field_count = match &s.columns {
            None => table.columns.len(),
            Some(c) => c.len(),
        };
        // Reject targets where a NOT NULL column would receive nothing —
        // matches INSERT's column-list rule.
        for (i, col) in table.columns.iter().enumerate() {
            if column_to_field[i].is_none() && !col.nullable {
                bail!(
                    "column '{}' is not nullable and was not given a value",
                    col.name
                );
            }
        }
        let column_types: Vec<DataType> = table.columns.iter().map(|c| c.data_type).collect();
        let column_nullable: Vec<bool> = table.columns.iter().map(|c| c.nullable).collect();
        Ok(AnalyzedCopyStatement {
            table_id,
            table_name: s.table.clone(),
            column_to_field,
            field_count,
            column_types,
            column_nullable,
        })
    }

    fn analyze_analyze_noop(&self, s: &AnalyzeStatement) -> Result<()> {
        for name in &s.tables {
            if self.catalog.find_table(name)?.is_none() {
                bail!("table '{name}' not found");
            }
        }
        Ok(())
    }

    fn analyze_drop_sequence(
        &self,
        s: &DropSequenceStatement,
    ) -> Result<AnalyzedDropSequenceStatement> {
        let mut seqs = Vec::new();
        for name in &s.names {
            match self.catalog.find_sequence(name)? {
                Some(seq) => seqs.push((seq.seq_id, name.clone(), seq.seq_page_id)),
                None if s.if_exists => continue,
                None => bail!("sequence '{name}' not found"),
            }
        }
        Ok(AnalyzedDropSequenceStatement { seqs })
    }

    fn analyze_drop_table(
        &self,
        s: &DropTableStatement,
    ) -> Result<AnalyzedDropTableStatement> {
        let mut tables = Vec::new();
        for name in &s.tables {
            match self.catalog.find_table(name)? {
                Some((id, _)) => tables.push((id, name.clone())),
                None if s.if_exists => continue,
                None => bail!("table '{name}' not found"),
            }
        }
        Ok(AnalyzedDropTableStatement { tables })
    }

    fn analyze_drop_index(
        &self,
        s: &DropIndexStatement,
    ) -> Result<Option<AnalyzedDropIndexStatement>> {
        match self.catalog.find_index(&s.name)? {
            Some(idx) => Ok(Some(AnalyzedDropIndexStatement {
                index_id: idx.index_id,
                name: idx.name,
                table_id: idx.table_id,
            })),
            None if s.if_exists => Ok(None),
            None => bail!("index '{}' not found", s.name),
        }
    }

    fn analyze_truncate(&self, s: &TruncateStatement) -> Result<AnalyzedTruncateStatement> {
        let mut tables = Vec::new();
        for name in &s.tables {
            let (id, _) = self
                .catalog
                .find_table(name)?
                .ok_or_else(|| anyhow::anyhow!("table '{name}' not found"))?;
            tables.push((id, name.clone()));
        }
        Ok(AnalyzedTruncateStatement { tables })
    }

    /// `ALTER TABLE ... ADD PRIMARY KEY/UNIQUE (col)` reuses CREATE INDEX
    /// machinery (Phase 4 will add the uniqueness enforcement).
    fn analyze_alter_table(
        &self,
        s: &AlterTableStatement,
    ) -> Result<AnalyzedCreateIndexStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;
        let columns = match &s.action {
            AlterTableAction::AddPrimaryKey { columns } => columns,
            AlterTableAction::AddUnique { columns } => columns,
        };
        if columns.len() != 1 {
            bail!("multi-column constraints not supported yet");
        }
        let col_name = &columns[0];
        let (col_idx, col) = table
            .columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == *col_name)
            .ok_or_else(|| anyhow::anyhow!("column '{col_name}' not found"))?;
        // Auto-name: `<table>_<col>_pkey` mirrors PG.
        let kind = match s.action {
            AlterTableAction::AddPrimaryKey { .. } => "pkey",
            AlterTableAction::AddUnique { .. } => "key",
        };
        let auto_name = format!("{}_{}_{}", s.table, col_name, kind);
        if self.catalog.find_index(&auto_name)?.is_some() {
            bail!("index '{auto_name}' already exists");
        }
        Ok(AnalyzedCreateIndexStatement {
            name: auto_name,
            table_id,
            table_name: s.table.clone(),
            column_index: col_idx,
            column_name: col.name.clone(),
            data_type: col.data_type,
            // ALTER TABLE ADD PRIMARY KEY / UNIQUE both create unique
            // indexes. The btree's insert_unique path enforces this
            // (see 4-1c).
            is_unique: true,
        })
    }

    fn analyze_create_index(
        &self,
        s: &CreateIndexStatement,
    ) -> Result<AnalyzedCreateIndexStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)?
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;
        let (col_idx, col) = table
            .columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == s.column)
            .ok_or_else(|| anyhow::anyhow!("column '{}' not found", s.column))?;
        if self.catalog.find_index(&s.name)?.is_some() {
            bail!("index '{}' already exists", s.name);
        }
        Ok(AnalyzedCreateIndexStatement {
            name: s.name.clone(),
            table_id,
            table_name: s.table.clone(),
            column_index: col_idx,
            column_name: col.name.clone(),
            data_type: col.data_type,
            // Plain CREATE INDEX is non-unique. CREATE UNIQUE INDEX
            // would set this true; the parser doesn't distinguish yet.
            is_unique: false,
        })
    }

    fn analyze_create_table(
        &self,
        s: &CreateTableStatement,
    ) -> Result<AnalyzedCreateTableStatement> {
        if self.catalog.find_table(&s.table)?.is_some() {
            bail!("table '{}' already exists", s.table);
        }
        let columns: Vec<AnalyzedColumnDef> = s
            .columns
            .iter()
            .map(|c| AnalyzedColumnDef {
                name: c.name.clone(),
                data_type: ast_to_runtime(c.data_type),
                nullable: c.nullable,
                default_text: c.default.as_ref().map(|e| e.to_string()),
            })
            .collect();
        let primary_key_column = match s.primary_key.len() {
            0 => None,
            1 => {
                let name = &s.primary_key[0];
                let idx = columns
                    .iter()
                    .position(|c| c.name == *name)
                    .ok_or_else(|| anyhow::anyhow!("PRIMARY KEY references unknown column '{name}'"))?;
                Some(idx)
            }
            _ => bail!("multi-column PRIMARY KEY not supported yet"),
        };
        Ok(AnalyzedCreateTableStatement {
            table_name: s.table.clone(),
            columns,
            primary_key_column,
        })
    }
}

/// AST-level rewrite: replace bare (unqualified) column refs whose name
/// matches one of `aliases` with the SELECT item's expression. Used to
/// make ORDER BY see SELECT aliases — `ORDER BY r` resolves to whatever
/// `region AS r` resolves to. Qualified refs like `t.r` are left alone.
fn substitute_aliases(e: &Expr, aliases: &[(String, Expr)]) -> Expr {
    match e {
        Expr::Column { qualifier: None, name } => {
            for (alias, expr) in aliases {
                if alias == name {
                    return expr.clone();
                }
            }
            e.clone()
        }
        Expr::Column { .. } | Expr::Literal(_) => e.clone(),
        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left: Box::new(substitute_aliases(left, aliases)),
            op: *op,
            right: Box::new(substitute_aliases(right, aliases)),
        },
        Expr::UnaryOp { op, expr } => Expr::UnaryOp {
            op: *op,
            expr: Box::new(substitute_aliases(expr, aliases)),
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: Box::new(substitute_aliases(expr, aliases)),
            negated: *negated,
        },
        Expr::FuncCall { name, args } => Expr::FuncCall {
            name: name.clone(),
            args: match args {
                FuncArgs::Star => FuncArgs::Star,
                FuncArgs::Exprs(es) => FuncArgs::Exprs(
                    es.iter().map(|e| substitute_aliases(e, aliases)).collect(),
                ),
            },
        },
    }
}

impl<'a> Analyzer<'a> {
    /// Recognise a scalar built-in by name, returning its analyzed form when
    /// matched. Returns `None` for unknown names so callers can try
    /// aggregates or fall back to error.
    fn analyze_scalar_builtin(
        &self,
        name: &str,
        args: &FuncArgs,
    ) -> Result<Option<AnalyzedExpr>> {
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "now" | "current_timestamp" | "transaction_timestamp" => match args {
                FuncArgs::Star => bail!("{lower}() does not take *"),
                FuncArgs::Exprs(es) if !es.is_empty() => {
                    bail!("{lower}() takes no arguments")
                }
                _ => Ok(Some(AnalyzedExpr::Now)),
            },
            "nextval" => self.analyze_sequence_call(SequenceFnKind::Nextval, args),
            "setval" => self.analyze_sequence_call(SequenceFnKind::Setval, args),
            _ => Ok(None),
        }
    }

    fn analyze_sequence_call(
        &self,
        kind: SequenceFnKind,
        args: &FuncArgs,
    ) -> Result<Option<AnalyzedExpr>> {
        let exprs = match args {
            FuncArgs::Exprs(es) => es,
            FuncArgs::Star => bail!("nextval/setval do not take *"),
        };
        let (name, setval_arg, setval_is_called) = match (kind, exprs.len()) {
            (SequenceFnKind::Nextval, 1) => (extract_string_literal(&exprs[0])?, None, false),
            (SequenceFnKind::Setval, 2) => (
                extract_string_literal(&exprs[0])?,
                Some(extract_int_literal(&exprs[1])?),
                true,
            ),
            (SequenceFnKind::Setval, 3) => (
                extract_string_literal(&exprs[0])?,
                Some(extract_int_literal(&exprs[1])?),
                extract_bool_literal(&exprs[2])?,
            ),
            (k, n) => bail!("{:?} expects {} arg(s), got {n}", k, match k {
                SequenceFnKind::Nextval => "1",
                SequenceFnKind::Setval => "2 or 3",
            }),
        };
        let seq = self
            .catalog
            .find_sequence(&name)?
            .ok_or_else(|| anyhow::anyhow!("sequence '{name}' not found"))?;
        Ok(Some(AnalyzedExpr::SequenceCall {
            kind,
            seq_id: seq.seq_id,
            seq_page_id: seq.seq_page_id,
            increment: seq.increment,
            start_value: seq.start_value,
            setval_arg,
            setval_is_called,
        }))
    }
}

fn extract_string_literal(e: &Expr) -> Result<String> {
    match e {
        Expr::Literal(Literal::String(s)) => Ok(s.clone()),
        _ => bail!("expected string literal"),
    }
}
fn extract_int_literal(e: &Expr) -> Result<i64> {
    match e {
        Expr::Literal(Literal::Integer(n)) => Ok(*n),
        Expr::UnaryOp {
            op: ast::UnaryOperator::Neg,
            expr,
        } => match expr.as_ref() {
            Expr::Literal(Literal::Integer(n)) => Ok(-*n),
            _ => bail!("expected integer literal"),
        },
        _ => bail!("expected integer literal"),
    }
}
fn extract_bool_literal(e: &Expr) -> Result<bool> {
    match e {
        Expr::Literal(Literal::Boolean(b)) => Ok(*b),
        _ => bail!("expected boolean literal"),
    }
}

/// Walk an AST expression to see whether it contains any aggregate-named
/// function call. Used to decide whether a SELECT needs an aggregation
/// pipeline even without a GROUP BY clause.
fn contains_aggregate(e: &Expr) -> bool {
    match e {
        Expr::Literal(_) | Expr::Column { .. } => false,
        Expr::BinaryOp { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        Expr::UnaryOp { expr, .. } => contains_aggregate(expr),
        Expr::IsNull { expr, .. } => contains_aggregate(expr),
        Expr::FuncCall { name, args } => {
            if AggKind::from_name(name).is_some() {
                return true;
            }
            match args {
                FuncArgs::Star => false,
                FuncArgs::Exprs(es) => es.iter().any(contains_aggregate),
            }
        }
    }
}

/// Structural equality of analyzed expressions. Used to match SELECT/HAVING
/// column refs against GROUP BY entries.
fn same_expr(a: &AnalyzedExpr, b: &AnalyzedExpr) -> bool {
    match (a, b) {
        (AnalyzedExpr::Literal(la), AnalyzedExpr::Literal(lb)) => match (&la.value, &lb.value) {
            (LiteralValue::Integer(x), LiteralValue::Integer(y)) => x == y,
            (LiteralValue::Float(x), LiteralValue::Float(y)) => x.to_bits() == y.to_bits(),
            (LiteralValue::String(x), LiteralValue::String(y)) => x == y,
            (LiteralValue::Boolean(x), LiteralValue::Boolean(y)) => x == y,
            (LiteralValue::Timestamp(x), LiteralValue::Timestamp(y)) => x == y,
            (LiteralValue::Date(x), LiteralValue::Date(y)) => x == y,
            (LiteralValue::Time(x), LiteralValue::Time(y)) => x == y,
            (
                LiteralValue::Interval {
                    months: m1,
                    days: d1,
                    micros: u1,
                },
                LiteralValue::Interval {
                    months: m2,
                    days: d2,
                    micros: u2,
                },
            ) => m1 == m2 && d1 == d2 && u1 == u2,
            (LiteralValue::Null, LiteralValue::Null) => true,
            _ => false,
        },
        (AnalyzedExpr::ColumnRef(x), AnalyzedExpr::ColumnRef(y)) => {
            x.rte_index == y.rte_index && x.column_index == y.column_index
        }
        (
            AnalyzedExpr::BinaryOp {
                left: l1,
                op: o1,
                right: r1,
                ..
            },
            AnalyzedExpr::BinaryOp {
                left: l2,
                op: o2,
                right: r2,
                ..
            },
        ) => o1 == o2 && same_expr(l1, l2) && same_expr(r1, r2),
        (
            AnalyzedExpr::UnaryOp {
                op: o1, expr: e1, ..
            },
            AnalyzedExpr::UnaryOp {
                op: o2, expr: e2, ..
            },
        ) => o1 == o2 && same_expr(e1, e2),
        (
            AnalyzedExpr::IsNull {
                expr: e1,
                negated: n1,
            },
            AnalyzedExpr::IsNull {
                expr: e2,
                negated: n2,
            },
        ) => n1 == n2 && same_expr(e1, e2),
        (AnalyzedExpr::Now, AnalyzedExpr::Now) => true,
        (
            AnalyzedExpr::SequenceCall {
                kind: k1,
                seq_id: i1,
                setval_arg: a1,
                setval_is_called: c1,
                ..
            },
            AnalyzedExpr::SequenceCall {
                kind: k2,
                seq_id: i2,
                setval_arg: a2,
                setval_is_called: c2,
                ..
            },
        ) => k1 == k2 && i1 == i2 && a1 == a2 && c1 == c2,
        _ => false,
    }
}

/// Type of an aggregate's result given its argument type.
///   COUNT → INT
///   SUM   → DOUBLE if arg is DOUBLE, otherwise INT
///   AVG   → DOUBLE always (mathematical mean)
///   MIN/MAX → same as arg
fn infer_aggregate_type(kind: AggKind, arg_type: Option<DataType>) -> Result<DataType> {
    Ok(match kind {
        AggKind::Count => DataType::Int,
        AggKind::Sum => match arg_type {
            Some(DataType::Double) => DataType::Double,
            Some(DataType::Int) | None => DataType::Int,
            Some(t) => bail!("{:?} is not numeric for SUM", t),
        },
        AggKind::Avg => match arg_type {
            Some(DataType::Int) | Some(DataType::Double) | None => DataType::Double,
            Some(t) => bail!("{:?} is not numeric for AVG", t),
        },
        AggKind::Min | AggKind::Max => match arg_type {
            Some(t) => t,
            None => bail!("argument type required for {:?}", kind),
        },
    })
}

fn literal_to_analyzed(lit: &Literal) -> AnalyzedLiteral {
    match lit {
        Literal::Integer(n) => AnalyzedLiteral {
            value: LiteralValue::Integer(*n),
            data_type: Some(DataType::Int),
        },
        Literal::Float(f) => AnalyzedLiteral {
            value: LiteralValue::Float(*f),
            data_type: Some(DataType::Double),
        },
        Literal::String(s) => AnalyzedLiteral {
            value: LiteralValue::String(s.clone()),
            data_type: Some(DataType::Varchar),
        },
        Literal::Timestamp(t) => AnalyzedLiteral {
            value: LiteralValue::Timestamp(*t),
            data_type: Some(DataType::Timestamp),
        },
        Literal::Date(d) => AnalyzedLiteral {
            value: LiteralValue::Date(*d),
            data_type: Some(DataType::Date),
        },
        Literal::Time(t) => AnalyzedLiteral {
            value: LiteralValue::Time(*t),
            data_type: Some(DataType::Time),
        },
        Literal::Interval {
            months,
            days,
            micros,
        } => AnalyzedLiteral {
            value: LiteralValue::Interval {
                months: *months,
                days: *days,
                micros: *micros,
            },
            data_type: Some(DataType::Interval),
        },
        Literal::Boolean(b) => AnalyzedLiteral {
            value: LiteralValue::Boolean(*b),
            data_type: Some(DataType::Bool),
        },
        Literal::Null => AnalyzedLiteral {
            value: LiteralValue::Null,
            data_type: None,
        },
    }
}

fn ast_to_runtime(dt: ast::DataType) -> DataType {
    match dt {
        ast::DataType::Int => DataType::Int,
        ast::DataType::Varchar => DataType::Varchar,
        ast::DataType::Double => DataType::Double,
        ast::DataType::Timestamp => DataType::Timestamp,
        ast::DataType::Date => DataType::Date,
        ast::DataType::Time => DataType::Time,
        ast::DataType::Interval => DataType::Interval,
    }
}

fn infer_binary_type(
    op: BinaryOperator,
    left: Option<DataType>,
    right: Option<DataType>,
) -> DataType {
    use BinaryOperator::*;
    use DataType::*;
    match op {
        Eq | Ne | Lt | Le | Gt | Ge | And | Or => Bool,
        Add | Sub | Mul | Div => {
            match (left, right, op) {
                // Date arithmetic.
                (Some(Date), Some(Interval), Add | Sub) => Timestamp, // PG: date+interval is timestamp
                (Some(Interval), Some(Date), Add) => Timestamp,
                (Some(Date), Some(Int), Add | Sub) => Date,
                (Some(Int), Some(Date), Add) => Date,
                (Some(Date), Some(Date), Sub) => Int, // days
                // Timestamp arithmetic.
                (Some(Timestamp), Some(Interval), Add | Sub) => Timestamp,
                (Some(Interval), Some(Timestamp), Add) => Timestamp,
                (Some(Timestamp), Some(Timestamp), Sub) => Interval,
                // Interval arithmetic.
                (Some(Interval), Some(Interval), Add | Sub) => Interval,
                (Some(Interval), Some(Int | Double), Mul | Div) => Interval,
                (Some(Int | Double), Some(Interval), Mul) => Interval,
                // Time + Interval → Time (modulo 24h).
                (Some(Time), Some(Interval), Add | Sub) => Time,
                (Some(Interval), Some(Time), Add) => Time,
                // Numeric default.
                _ if matches!(left, Some(Double)) || matches!(right, Some(Double)) => Double,
                _ => Int,
            }
        }
    }
}

/// Whether a value of `from` can be assigned to a column of `to` (e.g. at
/// INSERT/UPDATE). Exact match always works; INT widens to DOUBLE.
fn assignable(from: DataType, to: DataType) -> bool {
    if from == to {
        return true;
    }
    matches!((from, to), (DataType::Int, DataType::Double))
}

fn infer_unary_type(op: UnaryOperator) -> DataType {
    match op {
        UnaryOperator::Not => DataType::Bool,
        UnaryOperator::Neg => DataType::Int,
    }
}

pub fn analyze(catalog: &Catalog, stmt: &Statement) -> Result<AnalyzedStatement> {
    let mut a = Analyzer::new(catalog);
    Ok(match stmt {
        Statement::Select(s) => AnalyzedStatement::Select(a.analyze_select(s)?),
        Statement::Insert(s) => AnalyzedStatement::Insert(a.analyze_insert(s)?),
        Statement::Delete(s) => AnalyzedStatement::Delete(a.analyze_delete(s)?),
        Statement::Update(s) => AnalyzedStatement::Update(a.analyze_update(s)?),
        Statement::CreateTable(s) => AnalyzedStatement::CreateTable(a.analyze_create_table(s)?),
        Statement::CreateIndex(s) => AnalyzedStatement::CreateIndex(a.analyze_create_index(s)?),
        Statement::DropTable(s) => AnalyzedStatement::DropTable(a.analyze_drop_table(s)?),
        Statement::DropIndex(s) => match a.analyze_drop_index(s)? {
            Some(d) => AnalyzedStatement::DropIndex(d),
            // IF EXISTS on a missing index → emit empty drop so executor no-ops.
            None => AnalyzedStatement::DropIndex(AnalyzedDropIndexStatement {
                index_id: usize::MAX,
                name: s.name.clone(),
                table_id: usize::MAX,
            }),
        },
        Statement::TruncateTable(s) => AnalyzedStatement::Truncate(a.analyze_truncate(s)?),
        Statement::AlterTable(s) => {
            AnalyzedStatement::AlterTableAddIndex(a.analyze_alter_table(s)?)
        }
        Statement::CreateSequence(s) => {
            AnalyzedStatement::CreateSequence(a.analyze_create_sequence(s)?)
        }
        Statement::DropSequence(s) => {
            AnalyzedStatement::DropSequence(a.analyze_drop_sequence(s)?)
        }
        Statement::Vacuum(s) => AnalyzedStatement::Vacuum(a.analyze_vacuum(s)?),
        Statement::Analyze(s) => {
            a.analyze_analyze_noop(s)?;
            AnalyzedStatement::AnalyzeNoop
        }
        Statement::Copy(s) => AnalyzedStatement::Copy(a.analyze_copy(s)?),
        Statement::Begin => AnalyzedStatement::Begin,
        Statement::Commit => AnalyzedStatement::Commit,
        Statement::Rollback => AnalyzedStatement::Rollback,
        Statement::Checkpoint => AnalyzedStatement::Checkpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    /// Build a fresh Catalog over a temp DB with `users(id INT NOT NULL, name
    /// VARCHAR)` already created. Each call uses a unique path so tests don't
    /// share state.
    fn setup() -> Catalog {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ccdb_an_{}_{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("wal"));
        let disk = crate::disk::DiskManager::open(&path).unwrap();
        let wal = std::sync::Arc::new(
            crate::wal::WalManager::open(&path.with_extension("wal")).unwrap(),
        );
        let bpm = crate::buffer_pool::BufferPool::new(disk, 8, wal.clone());
        let clog = std::sync::Arc::new(crate::clog::Clog::in_memory());
        let tm = std::sync::Arc::new(
            crate::transaction_manager::TransactionManager::new(clog),
        );
        crate::bootstrap::bootstrap(&bpm, &tm).unwrap();
        let cat = Catalog::new(bpm.clone(), std::sync::Arc::clone(&tm));
        // CREATE TABLE users (id INT NOT NULL, name VARCHAR).
        {
            let mut tx = crate::transaction::Transaction::new(
                std::sync::Arc::clone(&tm),
            );
            let lm = crate::lock_manager::LockManager::new();
            let stmt = parse(
                "CREATE TABLE users (id INT NOT NULL, name VARCHAR)",
            )
            .unwrap();
            let analyzed = analyze(&cat, &stmt).unwrap();
            crate::executor::execute(
                &bpm, &lm, &wal, &tm, &cat, &analyzed, &mut tx,
            )
            .unwrap();
        }
        cat
    }

    fn an(sql: &str) -> Result<AnalyzedStatement> {
        let cat = setup();
        let stmt = parse(sql)?;
        analyze(&cat, &stmt)
    }

    #[test]
    fn select_star_expands_to_all_columns() {
        let s = an("SELECT * FROM users").unwrap();
        let AnalyzedStatement::Select(s) = s else {
            panic!()
        };
        assert_eq!(s.select_items.len(), 2);
        assert!(matches!(
            s.select_items[0].expr,
            AnalyzedExpr::ColumnRef(_)
        ));
        assert_eq!(s.range_table.len(), 1);
    }

    #[test]
    fn select_resolves_known_column_with_type() {
        let s = an("SELECT id FROM users WHERE id > 10").unwrap();
        let AnalyzedStatement::Select(s) = s else {
            panic!()
        };
        let AnalyzedExpr::ColumnRef(r) = &s.select_items[0].expr else {
            panic!()
        };
        assert_eq!(r.column_name, "id");
        assert_eq!(r.data_type, DataType::Int);
        let w = s.where_clause.unwrap();
        assert_eq!(w.data_type(), Some(DataType::Bool));
    }

    #[test]
    fn unknown_table_errors() {
        let e = an("SELECT * FROM nope").unwrap_err().to_string();
        assert!(e.contains("'nope'"));
    }

    #[test]
    fn unknown_column_errors() {
        let e = an("SELECT zz FROM users").unwrap_err().to_string();
        assert!(e.contains("'zz'"));
    }

    #[test]
    fn insert_arity_mismatch() {
        assert!(an("INSERT INTO users VALUES (1)").is_err());
    }

    #[test]
    fn insert_type_mismatch() {
        assert!(an("INSERT INTO users VALUES ('Alice', 1)").is_err());
    }

    #[test]
    fn insert_null_into_not_null_errors() {
        // users.id is NOT NULL.
        assert!(an("INSERT INTO users VALUES (NULL, 'Alice')").is_err());
    }

    #[test]
    fn insert_null_into_nullable_ok() {
        // users.name is NULL.
        an("INSERT INTO users VALUES (1, NULL)").unwrap();
    }

    #[test]
    fn create_table_already_exists() {
        assert!(an("CREATE TABLE users (id INT)").is_err());
    }

    #[test]
    fn create_table_new_ok() {
        let s = an("CREATE TABLE foo (a INT, b VARCHAR)").unwrap();
        let AnalyzedStatement::CreateTable(c) = s else {
            panic!()
        };
        assert_eq!(c.table_name, "foo");
        assert_eq!(c.columns.len(), 2);
    }

    #[test]
    fn arithmetic_result_type_is_int() {
        let s = an("SELECT id + 1 FROM users").unwrap();
        let AnalyzedStatement::Select(s) = s else {
            panic!()
        };
        assert_eq!(s.select_items[0].expr.data_type(), Some(DataType::Int));
    }
}
