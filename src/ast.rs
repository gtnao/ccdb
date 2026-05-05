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
    /// One Vec<Expr> per row. Multi-row form: VALUES (...), (...), ... .
    pub rows: Vec<Vec<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    pub table: String,
    pub columns: Vec<ColumnDef>,
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
