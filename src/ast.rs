//! Parsed SQL AST. Distinct from runtime types in `tuple.rs`.

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Select(SelectStatement),
    Insert(InsertStatement),
    Delete(DeleteStatement),
    Update(UpdateStatement),
    CreateTable(CreateTableStatement),
    CreateIndex(CreateIndexStatement),
    DropTable(DropTableStatement),
    DropIndex(DropIndexStatement),
    TruncateTable(TruncateStatement),
    AlterTable(AlterTableStatement),
    CreateSequence(CreateSequenceStatement),
    DropSequence(DropSequenceStatement),
    Vacuum(VacuumStatement),
    Analyze(AnalyzeStatement),
    /// `COPY t [(cols)] FROM STDIN`. Only the FROM STDIN form is supported;
    /// the data itself arrives as CopyData messages on the wire, not in the
    /// statement body.
    Copy(CopyStatement),
    Begin,
    Commit,
    Rollback,
    Checkpoint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: Vec<Assignment>,
    pub where_clause: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub column: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    pub columns: Vec<SelectColumn>,
    pub from: FromClause,
    pub where_clause: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderBy>,
    /// Cap on rows emitted to the client. Omitted ⇒ no cap.
    pub limit: Option<u64>,
    /// `SELECT DISTINCT col, ...` — analyzer treats this as adding the
    /// projected expressions to GROUP BY (semantically equivalent).
    pub distinct: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderBy {
    pub expr: Expr,
    pub dir: OrderDir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDir {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
}

/// Tree-shaped FROM clause. Joins are left-associative:
/// `A JOIN B JOIN C` parses as `Join(Join(Table(A), B), C)`.
#[derive(Debug, Clone, PartialEq)]
pub enum FromClause {
    /// No FROM at all (e.g. `SELECT 1 + 1;`). Yields one empty input row
    /// so the SELECT list is evaluated once.
    Empty,
    Table(TableRef),
    Join {
        left: Box<FromClause>,
        right: TableRef,
        join_type: JoinType,
        on: Expr,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectColumn {
    Asterisk,
    /// `expr [AS alias]`. Bare alias (no `AS` keyword) is also accepted.
    Expr {
        expr: Expr,
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct InsertStatement {
    pub table: String,
    /// `INSERT INTO t (a, b) VALUES (...)` records the column list here.
    /// `None` means VALUES targets every column in declaration order.
    /// Unlisted columns get NULL (or the analyzer rejects if NOT NULL).
    pub columns: Option<Vec<String>>,
    /// One Vec<Expr> per row. Multi-row form: VALUES (...), (...), ... .
    pub rows: Vec<Vec<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    pub table: String,
    pub columns: Vec<ColumnDef>,
    /// Column names appearing in a table-level `PRIMARY KEY (...)` clause
    /// or in a single column-level `PRIMARY KEY` qualifier. Empty when the
    /// table has no PK; multi-column PKs land here too but are not yet
    /// supported by the index machinery (analyzer rejects).
    pub primary_key: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStatement {
    pub name: String,
    pub table: String,
    pub column: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropTableStatement {
    pub tables: Vec<String>,
    pub if_exists: bool,
    /// CASCADE/RESTRICT — currently only parsed; nothing depends on tables
    /// in a way that CASCADE would need to remove. Recorded so syntactic
    /// completeness is preserved.
    pub cascade: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TruncateStatement {
    pub tables: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStatement {
    pub table: String,
    pub action: AlterTableAction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateSequenceStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub increment: i64,
    pub start_value: Option<i64>,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropSequenceStatement {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VacuumStatement {
    /// Empty = vacuum every user table. Otherwise just the listed ones.
    pub tables: Vec<String>,
    /// VACUUM ANALYZE — at the moment ANALYZE is a no-op until Phase 9
    /// adds real statistics collection. We still parse and accept it.
    pub analyze: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzeStatement {
    pub tables: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CopyStatement {
    pub table: String,
    /// `None` ⇒ every column in declaration order. Otherwise the listed
    /// columns receive the COPY data; unlisted columns get NULL.
    pub columns: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterTableAction {
    /// `ADD [CONSTRAINT name] PRIMARY KEY (col, ...)` — currently single-column.
    /// Constraint enforcement is Phase 4; this commit just creates a B+Tree
    /// index on the column.
    AddPrimaryKey { columns: Vec<String> },
    /// `ADD [CONSTRAINT name] UNIQUE (col, ...)` — same caveat.
    AddUnique { columns: Vec<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    /// `DEFAULT <expr>` from CREATE TABLE. Carried as a parsed Expr so the
    /// analyzer can bind it. Currently only literal expressions persist
    /// across the catalog round-trip (function-call defaults like
    /// CURRENT_TIMESTAMP would need a stable text encoding — TODO).
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int,
    Varchar,
    Double,
    Timestamp,
    Date,
    Time,
    Interval,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    /// Column reference: bare `id` has qualifier=None; `u.id` has
    /// qualifier=Some("u") (the alias or table name as written).
    Column {
        qualifier: Option<String>,
        name: String,
    },
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expr>,
    },
    /// `expr IS NULL` / `expr IS NOT NULL`. Distinct from `= NULL` because
    /// IS NULL is the only predicate that returns a definite bool when its
    /// operand is NULL; equality propagates NULL into NULL (3VL).
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    /// `name(args)` — function or aggregate call. Aggregate detection is
    /// the analyzer's job; the parser just records the syntactic shape.
    /// `COUNT(*)` is represented with `args = FuncArgs::Star`.
    FuncCall {
        name: String,
        args: FuncArgs,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FuncArgs {
    Star,
    Exprs(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    /// `TIMESTAMP '2024-01-01 12:34:56'` — already parsed to PG epoch microseconds.
    Timestamp(i64),
    /// `DATE 'YYYY-MM-DD'` — already parsed to days since PG epoch.
    Date(i32),
    /// `TIME 'HH:MM:SS[.f]'` — μs since 00:00:00.
    Time(i64),
    /// `INTERVAL '...'` — already parsed to (months, days, micros).
    Interval { months: i32, days: i32, micros: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Not,
    Neg,
}

impl std::fmt::Display for Expr {
    /// Round-trip-able SQL serialization for the subset that survives a
    /// `pg_attribute.default_text` save: literals, unary ±/NOT, the usual
    /// binary operators, function calls, IS NULL. Used to persist DEFAULT
    /// expressions; the analyzer re-parses them on every INSERT that
    /// omits the column.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Literal(l) => match l {
                Literal::Integer(n) => write!(f, "{n}"),
                Literal::Float(x) => write!(f, "{x}"),
                Literal::Boolean(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
                Literal::Null => write!(f, "NULL"),
                Literal::String(s) => write!(f, "'{}'", s.replace('\'', "''")),
                Literal::Timestamp(t) => write!(f, "TIMESTAMP '{t}us'"),
                Literal::Date(d) => write!(f, "DATE '{d}d'"),
                Literal::Time(t) => write!(f, "TIME '{t}us'"),
                Literal::Interval { months, days, micros } => {
                    write!(f, "INTERVAL '{months}m {days}d {micros}us'")
                }
            },
            Expr::Column { qualifier: Some(q), name } => write!(f, "{q}.{name}"),
            Expr::Column { qualifier: None, name } => write!(f, "{name}"),
            Expr::UnaryOp { op, expr } => {
                let op_s = match op {
                    UnaryOperator::Not => "NOT ",
                    UnaryOperator::Neg => "-",
                };
                write!(f, "({op_s}{expr})")
            }
            Expr::BinaryOp { left, op, right } => {
                let op_s = match op {
                    BinaryOperator::Eq => "=",
                    BinaryOperator::Ne => "<>",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Ge => ">=",
                    BinaryOperator::And => "AND",
                    BinaryOperator::Or => "OR",
                    BinaryOperator::Add => "+",
                    BinaryOperator::Sub => "-",
                    BinaryOperator::Mul => "*",
                    BinaryOperator::Div => "/",
                };
                write!(f, "({left} {op_s} {right})")
            }
            Expr::IsNull { expr, negated: false } => write!(f, "({expr} IS NULL)"),
            Expr::IsNull { expr, negated: true } => write!(f, "({expr} IS NOT NULL)"),
            Expr::FuncCall { name, args } => {
                write!(f, "{name}(")?;
                match args {
                    FuncArgs::Star => write!(f, "*")?,
                    FuncArgs::Exprs(es) => {
                        for (i, e) in es.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{e}")?;
                        }
                    }
                }
                write!(f, ")")
            }
        }
    }
}
