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
    self, BinaryOperator, CreateTableStatement, DeleteStatement, Expr, InsertStatement, Literal,
    SelectColumn, SelectStatement, Statement, UnaryOperator, UpdateStatement,
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
}

#[derive(Debug, Clone)]
pub enum AnalyzedStatement {
    Select(AnalyzedSelectStatement),
    Insert(AnalyzedInsertStatement),
    Delete(AnalyzedDeleteStatement),
    Update(AnalyzedUpdateStatement),
    CreateTable(AnalyzedCreateTableStatement),
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
    pub from_rte_index: usize,
    pub select_items: Vec<AnalyzedSelectItem>,
    pub where_clause: Option<AnalyzedExpr>,
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
    pub values: Vec<AnalyzedExpr>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedCreateTableStatement {
    pub table_name: String,
    pub columns: Vec<AnalyzedColumnDef>,
}

#[derive(Debug, Clone)]
pub struct AnalyzedColumnDef {
    pub name: String,
    pub data_type: DataType,
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
    String(String),
    Boolean(bool),
    Null,
}

impl AnalyzedExpr {
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            AnalyzedExpr::Literal(l) => l.data_type,
            AnalyzedExpr::ColumnRef(c) => Some(c.data_type),
            AnalyzedExpr::BinaryOp { result_type, .. }
            | AnalyzedExpr::UnaryOp { result_type, .. } => Some(*result_type),
        }
    }
}

#[derive(Debug, Clone)]
struct ScopeEntry {
    #[allow(dead_code)] // alias resolution comes when qualified refs land
    name: String,
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

    fn add_rte(&mut self, source: TableSource, output_columns: Vec<OutputColumn>) -> usize {
        let idx = self.range_table.len();
        self.range_table.push(RangeTableEntry {
            rte_index: idx,
            source,
            output_columns,
        });
        idx
    }

    fn resolve_column(&self, name: &str) -> Result<AnalyzedColumnRef> {
        for scope in self.scopes.iter().rev() {
            for entry in scope {
                let rte = &self.range_table[entry.rte_index];
                if let Some((idx, col)) = rte
                    .output_columns
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.name == name)
                {
                    return Ok(AnalyzedColumnRef {
                        rte_index: entry.rte_index,
                        column_index: idx,
                        column_name: name.to_string(),
                        data_type: col.data_type,
                    });
                }
            }
        }
        bail!("column '{name}' not found")
    }

    fn analyze_expr(&self, e: &Expr) -> Result<AnalyzedExpr> {
        match e {
            Expr::Literal(l) => Ok(AnalyzedExpr::Literal(literal_to_analyzed(l))),
            Expr::Column(name) => Ok(AnalyzedExpr::ColumnRef(self.resolve_column(name)?)),
            Expr::BinaryOp { left, op, right } => {
                let l = self.analyze_expr(left)?;
                let r = self.analyze_expr(right)?;
                Ok(AnalyzedExpr::BinaryOp {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                    result_type: infer_binary_type(*op),
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
        }
    }

    fn analyze_select(&mut self, s: &SelectStatement) -> Result<AnalyzedSelectStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.from.name)
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.from.name))?;
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
                table_name: s.from.name.clone(),
            },
            output_columns,
        );

        let scope_name = s.from.alias.clone().unwrap_or_else(|| s.from.name.clone());
        self.scopes.push(vec![ScopeEntry {
            name: scope_name,
            rte_index,
        }]);

        let mut select_items = Vec::new();
        for c in &s.columns {
            match c {
                SelectColumn::Asterisk => {
                    let rte = &self.range_table[rte_index];
                    for (i, oc) in rte.output_columns.iter().enumerate() {
                        select_items.push(AnalyzedSelectItem {
                            expr: AnalyzedExpr::ColumnRef(AnalyzedColumnRef {
                                rte_index,
                                column_index: i,
                                column_name: oc.name.clone(),
                                data_type: oc.data_type,
                            }),
                            alias: None,
                        });
                    }
                }
                SelectColumn::Expr(e) => {
                    select_items.push(AnalyzedSelectItem {
                        expr: self.analyze_expr(e)?,
                        alias: None,
                    });
                }
            }
        }

        let where_clause = match &s.where_clause {
            Some(e) => Some(self.analyze_expr(e)?),
            None => None,
        };

        self.scopes.pop();

        Ok(AnalyzedSelectStatement {
            range_table: mem::take(&mut self.range_table),
            from_rte_index: rte_index,
            select_items,
            where_clause,
        })
    }

    fn analyze_insert(&self, s: &InsertStatement) -> Result<AnalyzedInsertStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)
            .ok_or_else(|| anyhow::anyhow!("table '{}' not found", s.table))?;
        if s.values.len() != table.columns.len() {
            bail!(
                "INSERT has {} values but table has {} columns",
                s.values.len(),
                table.columns.len()
            );
        }

        let mut values = Vec::with_capacity(s.values.len());
        for (i, v) in s.values.iter().enumerate() {
            let expr = self.analyze_expr(v)?;
            let col = &table.columns[i];
            match expr.data_type() {
                None => {
                    if !col.nullable {
                        bail!("column '{}' is not nullable", col.name);
                    }
                }
                Some(t) if t == col.data_type => {}
                Some(t) => bail!(
                    "type mismatch for column '{}': expected {:?}, got {:?}",
                    col.name,
                    col.data_type,
                    t
                ),
            }
            values.push(expr);
        }

        Ok(AnalyzedInsertStatement {
            table_id,
            table_name: s.table.clone(),
            values,
        })
    }

    fn analyze_delete(&mut self, s: &DeleteStatement) -> Result<AnalyzedDeleteStatement> {
        let (table_id, table) = self
            .catalog
            .find_table(&s.table)
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
        );
        self.scopes.push(vec![ScopeEntry {
            name: s.table.clone(),
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
            .find_table(&s.table)
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
        );
        self.scopes.push(vec![ScopeEntry {
            name: s.table.clone(),
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
                Some(t) if t == col.data_type => {}
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

    fn analyze_create_table(
        &self,
        s: &CreateTableStatement,
    ) -> Result<AnalyzedCreateTableStatement> {
        if self.catalog.find_table(&s.table).is_some() {
            bail!("table '{}' already exists", s.table);
        }
        let columns = s
            .columns
            .iter()
            .map(|c| AnalyzedColumnDef {
                name: c.name.clone(),
                data_type: ast_to_runtime(c.data_type),
            })
            .collect();
        Ok(AnalyzedCreateTableStatement {
            table_name: s.table.clone(),
            columns,
        })
    }
}

fn literal_to_analyzed(lit: &Literal) -> AnalyzedLiteral {
    match lit {
        Literal::Integer(n) => AnalyzedLiteral {
            value: LiteralValue::Integer(*n),
            data_type: Some(DataType::Int),
        },
        Literal::String(s) => AnalyzedLiteral {
            value: LiteralValue::String(s.clone()),
            data_type: Some(DataType::Varchar),
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
    }
}

fn infer_binary_type(op: BinaryOperator) -> DataType {
    use BinaryOperator::*;
    match op {
        Eq | Ne | Lt | Le | Gt | Ge | And | Or => DataType::Bool,
        Add | Sub | Mul | Div => DataType::Int,
    }
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

    fn an(sql: &str) -> Result<AnalyzedStatement> {
        let stmt = parse(sql)?;
        let cat = Catalog::new();
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
