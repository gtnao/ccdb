use anyhow::{Result, bail};

use crate::ast::*;
use crate::lexer::{Lexer, Token};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn expect(&mut self, want: &Token) -> Result<()> {
        match self.peek() {
            Some(t) if t == want => {
                self.bump();
                Ok(())
            }
            other => bail!("expected {want:?}, got {other:?}"),
        }
    }

    pub fn parse(&mut self) -> Result<Statement> {
        let stmt = match self.peek() {
            Some(Token::Select) => self.parse_select()?,
            Some(Token::Insert) => self.parse_insert()?,
            Some(Token::Delete) => self.parse_delete()?,
            Some(Token::Update) => self.parse_update()?,
            Some(Token::Create) => self.parse_create_table()?,
            Some(Token::Drop) => self.parse_drop()?,
            Some(Token::Truncate) => self.parse_truncate()?,
            Some(Token::Alter) => self.parse_alter()?,
            Some(Token::Vacuum) => self.parse_vacuum()?,
            Some(Token::Analyze) => self.parse_analyze()?,
            Some(Token::Begin) => {
                self.bump();
                Statement::Begin
            }
            // `END` is a PG-specific spelling of COMMIT (used by pgbench).
            Some(Token::Commit) | Some(Token::End) => {
                self.bump();
                Statement::Commit
            }
            Some(Token::Rollback) => {
                self.bump();
                Statement::Rollback
            }
            Some(Token::Checkpoint) => {
                self.bump();
                Statement::Checkpoint
            }
            other => bail!(
                "expected statement keyword, got {other:?}"
            ),
        };
        if let Some(Token::Semicolon) = self.peek() {
            self.bump();
        }
        if let Some(extra) = self.peek() {
            bail!("unexpected trailing token: {extra:?}");
        }
        Ok(stmt)
    }

    fn parse_select(&mut self) -> Result<Statement> {
        self.expect(&Token::Select)?;
        let distinct = if matches!(self.peek(), Some(Token::Distinct)) {
            self.bump();
            true
        } else {
            false
        };
        let mut columns = Vec::new();
        loop {
            if matches!(self.peek(), Some(Token::Asterisk)) {
                self.bump();
                columns.push(SelectColumn::Asterisk);
            } else {
                let expr = self.parse_expr()?;
                // Optional `AS alias` or bare-ident alias. Stop short of FROM
                // and other clauses so `SELECT id FROM t` doesn't try to
                // alias `id` with `FROM`.
                let alias = if matches!(self.peek(), Some(Token::As)) {
                    self.bump();
                    Some(self.parse_ident()?)
                } else if matches!(self.peek(), Some(Token::Ident(_))) {
                    Some(self.parse_ident()?)
                } else {
                    None
                };
                columns.push(SelectColumn::Expr { expr, alias });
            }
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        // FROM is optional: `SELECT 1+1;` produces a single row.
        let from = if matches!(self.peek(), Some(Token::From)) {
            self.bump();
            self.parse_from_clause()?
        } else {
            FromClause::Empty
        };
        let where_clause = if matches!(self.peek(), Some(Token::Where)) {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let group_by = if matches!(self.peek(), Some(Token::Group)) {
            self.bump();
            self.expect(&Token::By)?;
            let mut exprs = vec![self.parse_expr()?];
            while matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                exprs.push(self.parse_expr()?);
            }
            exprs
        } else {
            Vec::new()
        };
        let having = if matches!(self.peek(), Some(Token::Having)) {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let order_by = if matches!(self.peek(), Some(Token::Order)) {
            self.bump();
            self.expect(&Token::By)?;
            let mut items = vec![self.parse_order_item()?];
            while matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                items.push(self.parse_order_item()?);
            }
            items
        } else {
            Vec::new()
        };
        let limit = if matches!(self.peek(), Some(Token::Limit)) {
            self.bump();
            match self.peek() {
                Some(Token::Integer(n)) if *n >= 0 => {
                    let n = *n as u64;
                    self.bump();
                    Some(n)
                }
                other => bail!("LIMIT requires non-negative integer, got {other:?}"),
            }
        } else {
            None
        };
        Ok(Statement::Select(SelectStatement {
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            distinct,
        }))
    }

    fn parse_order_item(&mut self) -> Result<OrderBy> {
        let expr = self.parse_expr()?;
        let dir = match self.peek() {
            Some(Token::Asc) => {
                self.bump();
                OrderDir::Asc
            }
            Some(Token::Desc) => {
                self.bump();
                OrderDir::Desc
            }
            _ => OrderDir::Asc,
        };
        Ok(OrderBy { expr, dir })
    }

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect(&Token::Insert)?;
        self.expect(&Token::Into)?;
        let table = self.parse_ident()?;
        // Optional column list: `INSERT INTO t (a, b) VALUES (...)`.
        let columns = if matches!(self.peek(), Some(Token::LParen)) {
            self.bump();
            let mut cols = Vec::new();
            loop {
                cols.push(self.parse_ident()?);
                match self.peek() {
                    Some(Token::Comma) => {
                        self.bump();
                    }
                    Some(Token::RParen) => break,
                    other => bail!("expected ',' or ')' in column list, got {other:?}"),
                }
            }
            self.expect(&Token::RParen)?;
            Some(cols)
        } else {
            None
        };
        self.expect(&Token::Values)?;
        let mut rows = Vec::new();
        loop {
            self.expect(&Token::LParen)?;
            let mut values = Vec::new();
            loop {
                values.push(self.parse_expr()?);
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.bump();
                } else {
                    break;
                }
            }
            self.expect(&Token::RParen)?;
            rows.push(values);
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        Ok(Statement::Insert(InsertStatement {
            table,
            columns,
            rows,
        }))
    }

    fn parse_create_table(&mut self) -> Result<Statement> {
        self.expect(&Token::Create)?;
        // CREATE INDEX ... — recognized as a no-op so sysbench's index DDL
        // doesn't error out. Real index support comes later.
        if matches!(self.peek(), Some(Token::Index)) {
            return self.parse_create_index_noop();
        }
        if matches!(self.peek(), Some(Token::Sequence)) {
            return self.parse_create_sequence();
        }
        self.expect(&Token::Table)?;
        // Optional `IF NOT EXISTS` — accepted but not enforced. Re-creating
        // an existing table will still fail in analyze_create_table.
        if matches!(self.peek(), Some(Token::If)) {
            self.bump();
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
        }
        let table = self.parse_ident()?;
        self.expect(&Token::LParen)?;
        let mut columns = Vec::new();
        loop {
            // Table-level constraint: `PRIMARY KEY (col, ...)` — parse, ignore.
            if matches!(self.peek(), Some(Token::Primary)) {
                self.bump();
                self.expect(&Token::Key)?;
                self.expect(&Token::LParen)?;
                // Skip column list.
                while !matches!(self.peek(), Some(Token::RParen)) {
                    self.bump();
                }
                self.expect(&Token::RParen)?;
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.bump();
                    continue;
                } else {
                    break;
                }
            }
            // Table-level KEY/UNIQUE/INDEX clauses also tolerated as no-ops
            // (none enforced).
            if matches!(self.peek(), Some(Token::Key) | Some(Token::Index)) {
                self.bump();
                // Skip optional name + parenthesized column list.
                if matches!(self.peek(), Some(Token::Ident(_))) {
                    self.bump();
                }
                if matches!(self.peek(), Some(Token::LParen)) {
                    let mut depth = 1;
                    self.bump();
                    while depth > 0 {
                        match self.peek() {
                            Some(Token::LParen) => depth += 1,
                            Some(Token::RParen) => depth -= 1,
                            None => bail!("unterminated KEY/INDEX clause"),
                            _ => {}
                        }
                        self.bump();
                    }
                }
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.bump();
                    continue;
                } else {
                    break;
                }
            }
            let name = self.parse_ident()?;
            let data_type = self.parse_data_type()?;
            // Trailing column constraints in any order. We accept them and
            // mostly ignore — only NOT NULL has runtime meaning.
            let mut nullable = true;
            loop {
                match self.peek() {
                    Some(Token::Not) => {
                        self.bump();
                        self.expect(&Token::Null)?;
                        nullable = false;
                    }
                    // `DEFAULT <expr>` — value is parsed and discarded.
                    Some(Token::Default) => {
                        self.bump();
                        let _ = self.parse_expr()?;
                    }
                    // `PRIMARY KEY` inline on a column — ignored (uniqueness
                    // not enforced).
                    Some(Token::Primary) => {
                        self.bump();
                        self.expect(&Token::Key)?;
                    }
                    _ => break,
                }
            }
            columns.push(ColumnDef {
                name,
                data_type,
                nullable,
            });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        // pgbench / many tools emit `WITH (fillfactor=100, ...)` after the
        // column list. We don't honour storage parameters; just consume and
        // discard the parenthesised list.
        if matches!(self.peek(), Some(Token::Ident(s)) if s.eq_ignore_ascii_case("with")) {
            self.bump();
            self.expect(&Token::LParen)?;
            let mut depth = 1;
            while depth > 0 {
                match self.peek() {
                    Some(Token::LParen) => depth += 1,
                    Some(Token::RParen) => depth -= 1,
                    None => bail!("unterminated WITH (...) clause"),
                    _ => {}
                }
                self.bump();
            }
        }
        Ok(Statement::CreateTable(CreateTableStatement {
            table,
            columns,
        }))
    }

    /// `CREATE SEQUENCE [IF NOT EXISTS] name
    ///     [INCREMENT [BY] n] [START [WITH] n] [MINVALUE n] [MAXVALUE n]`
    /// Cycle / cache / owned-by are not parsed yet.
    fn parse_create_sequence(&mut self) -> Result<Statement> {
        self.expect(&Token::Sequence)?;
        let if_not_exists = if matches!(self.peek(), Some(Token::If)) {
            self.bump();
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        let mut increment: i64 = 1;
        let mut start_value: Option<i64> = None;
        let mut min_value: Option<i64> = None;
        let mut max_value: Option<i64> = None;
        // Loop over option keywords. They're plain idents (we don't dedicate
        // tokens to every option), so we eat them via parse_ident lookahead.
        while let Some(Token::Ident(s)) = self.peek() {
            let kw = s.to_ascii_uppercase();
            match kw.as_str() {
                "INCREMENT" => {
                    self.bump();
                    if matches!(self.peek(), Some(Token::Ident(b)) if b.eq_ignore_ascii_case("BY"))
                    {
                        self.bump();
                    }
                    increment = self.parse_signed_int()?;
                }
                "START" => {
                    self.bump();
                    if matches!(self.peek(), Some(Token::Ident(w)) if w.eq_ignore_ascii_case("WITH"))
                    {
                        self.bump();
                    }
                    start_value = Some(self.parse_signed_int()?);
                }
                "MINVALUE" => {
                    self.bump();
                    min_value = Some(self.parse_signed_int()?);
                }
                "MAXVALUE" => {
                    self.bump();
                    max_value = Some(self.parse_signed_int()?);
                }
                _ => break,
            }
        }
        Ok(Statement::CreateSequence(CreateSequenceStatement {
            name,
            if_not_exists,
            increment,
            start_value,
            min_value,
            max_value,
        }))
    }

    fn parse_signed_int(&mut self) -> Result<i64> {
        let neg = if matches!(self.peek(), Some(Token::Minus)) {
            self.bump();
            true
        } else {
            false
        };
        match self.peek() {
            Some(Token::Integer(n)) => {
                let v = *n;
                self.bump();
                Ok(if neg { -v } else { v })
            }
            other => bail!("expected integer, got {other:?}"),
        }
    }

    /// `DROP TABLE [IF EXISTS] name [, name, ...] [CASCADE | RESTRICT]`
    /// or `DROP INDEX [IF EXISTS] name`.
    fn parse_drop(&mut self) -> Result<Statement> {
        self.expect(&Token::Drop)?;
        match self.peek() {
            Some(Token::Table) => {
                self.bump();
                let if_exists = if matches!(self.peek(), Some(Token::If)) {
                    self.bump();
                    self.expect(&Token::Exists)?;
                    true
                } else {
                    false
                };
                let mut tables = vec![self.parse_ident()?];
                while matches!(self.peek(), Some(Token::Comma)) {
                    self.bump();
                    tables.push(self.parse_ident()?);
                }
                let cascade = match self.peek() {
                    Some(Token::Cascade) => {
                        self.bump();
                        true
                    }
                    Some(Token::Restrict) => {
                        self.bump();
                        false
                    }
                    _ => false,
                };
                Ok(Statement::DropTable(DropTableStatement {
                    tables,
                    if_exists,
                    cascade,
                }))
            }
            Some(Token::Index) => {
                self.bump();
                let if_exists = if matches!(self.peek(), Some(Token::If)) {
                    self.bump();
                    self.expect(&Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_ident()?;
                Ok(Statement::DropIndex(DropIndexStatement { name, if_exists }))
            }
            Some(Token::Sequence) => {
                self.bump();
                let if_exists = if matches!(self.peek(), Some(Token::If)) {
                    self.bump();
                    self.expect(&Token::Exists)?;
                    true
                } else {
                    false
                };
                let mut names = vec![self.parse_ident()?];
                while matches!(self.peek(), Some(Token::Comma)) {
                    self.bump();
                    names.push(self.parse_ident()?);
                }
                Ok(Statement::DropSequence(DropSequenceStatement {
                    names,
                    if_exists,
                }))
            }
            other => bail!("expected TABLE / INDEX / SEQUENCE after DROP, got {other:?}"),
        }
    }

    /// `VACUUM [(option [, ...])] [ANALYZE] [table_list]` —
    /// option list (FULL, FREEZE, VERBOSE, …) is parse-and-ignored.
    fn parse_vacuum(&mut self) -> Result<Statement> {
        self.expect(&Token::Vacuum)?;
        // Optional `( opt [, opt] )` block — eat until matching ).
        if matches!(self.peek(), Some(Token::LParen)) {
            self.bump();
            let mut depth = 1;
            while depth > 0 {
                match self.peek() {
                    Some(Token::LParen) => depth += 1,
                    Some(Token::RParen) => depth -= 1,
                    None => bail!("unterminated VACUUM ( ... )"),
                    _ => {}
                }
                self.bump();
            }
        }
        let analyze = if matches!(self.peek(), Some(Token::Analyze)) {
            self.bump();
            true
        } else {
            false
        };
        let mut tables = Vec::new();
        if matches!(self.peek(), Some(Token::Ident(_))) {
            tables.push(self.parse_ident()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                tables.push(self.parse_ident()?);
            }
        }
        Ok(Statement::Vacuum(VacuumStatement { tables, analyze }))
    }

    /// `ANALYZE [table_list]`.
    fn parse_analyze(&mut self) -> Result<Statement> {
        self.expect(&Token::Analyze)?;
        let mut tables = Vec::new();
        if matches!(self.peek(), Some(Token::Ident(_))) {
            tables.push(self.parse_ident()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                tables.push(self.parse_ident()?);
            }
        }
        Ok(Statement::Analyze(AnalyzeStatement { tables }))
    }

    /// `TRUNCATE [TABLE] name [, name, ...]`.
    fn parse_truncate(&mut self) -> Result<Statement> {
        self.expect(&Token::Truncate)?;
        if matches!(self.peek(), Some(Token::Table)) {
            self.bump();
        }
        let mut tables = vec![self.parse_ident()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.bump();
            tables.push(self.parse_ident()?);
        }
        Ok(Statement::TruncateTable(TruncateStatement { tables }))
    }

    /// `ALTER TABLE name <action>`.
    fn parse_alter(&mut self) -> Result<Statement> {
        self.expect(&Token::Alter)?;
        self.expect(&Token::Table)?;
        let table = self.parse_ident()?;
        // Only ADD CONSTRAINT / ADD PRIMARY KEY / ADD UNIQUE supported.
        // ADD/DROP/RENAME COLUMN is deferred (storage layout migration).
        if matches!(self.peek(), Some(Token::Ident(s)) if s.eq_ignore_ascii_case("ADD"))
            || matches!(self.peek(), Some(Token::Ident(_)))
        {
            // We use Ident("ADD") rather than a keyword for simplicity; the
            // parser already keeps ADD as Ident.
            let kw = self.parse_ident()?;
            if !kw.eq_ignore_ascii_case("ADD") {
                bail!("ALTER TABLE: expected ADD, got {kw:?}");
            }
            // Optional CONSTRAINT <name>.
            if matches!(self.peek(), Some(Token::Ident(s)) if s.eq_ignore_ascii_case("CONSTRAINT"))
            {
                self.bump();
                let _name = self.parse_ident()?; // name is recorded in pg_constraint later
            }
            match self.peek() {
                Some(Token::Primary) => {
                    self.bump();
                    self.expect(&Token::Key)?;
                    self.expect(&Token::LParen)?;
                    let columns = self.parse_ident_list()?;
                    self.expect(&Token::RParen)?;
                    Ok(Statement::AlterTable(AlterTableStatement {
                        table,
                        action: AlterTableAction::AddPrimaryKey { columns },
                    }))
                }
                Some(tok) if matches!(tok, Token::Ident(s) if s.eq_ignore_ascii_case("UNIQUE")) => {
                    self.bump();
                    self.expect(&Token::LParen)?;
                    let columns = self.parse_ident_list()?;
                    self.expect(&Token::RParen)?;
                    Ok(Statement::AlterTable(AlterTableStatement {
                        table,
                        action: AlterTableAction::AddUnique { columns },
                    }))
                }
                other => bail!(
                    "ALTER TABLE ADD: expected PRIMARY KEY or UNIQUE, got {other:?}"
                ),
            }
        } else {
            bail!("ALTER TABLE: only ADD ... is supported")
        }
    }

    fn parse_ident_list(&mut self) -> Result<Vec<String>> {
        let mut out = vec![self.parse_ident()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.bump();
            out.push(self.parse_ident()?);
        }
        Ok(out)
    }

    /// `CREATE INDEX <name> ON <table> (<col>)` — currently single-column only.
    /// Optional `IF NOT EXISTS` is accepted (sysbench uses it).
    fn parse_create_index_noop(&mut self) -> Result<Statement> {
        self.expect(&Token::Index)?;
        if matches!(self.peek(), Some(Token::If)) {
            self.bump();
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
        }
        let name = self.parse_ident()?;
        self.expect(&Token::On)?;
        let table = self.parse_ident()?;
        self.expect(&Token::LParen)?;
        let column = self.parse_ident()?;
        // Skip any extra columns or modifiers — multi-column indexes not
        // supported yet; we just take the first column and ignore the rest.
        while !matches!(self.peek(), Some(Token::RParen) | None) {
            self.bump();
        }
        self.expect(&Token::RParen)?;
        Ok(Statement::CreateIndex(CreateIndexStatement {
            name,
            table,
            column,
        }))
    }

    fn parse_data_type(&mut self) -> Result<DataType> {
        match self.peek() {
            Some(Token::Int) => {
                self.bump();
                self.skip_optional_size();
                Ok(DataType::Int)
            }
            Some(Token::Varchar) | Some(Token::Char) => {
                self.bump();
                // CHAR/VARCHAR optionally take a size like (120). We don't
                // enforce length yet, so just consume and ignore.
                self.skip_optional_size();
                Ok(DataType::Varchar)
            }
            Some(Token::Double) => {
                self.bump();
                self.skip_optional_size();
                Ok(DataType::Double)
            }
            Some(Token::Timestamp) => {
                self.bump();
                // Optional `(N)` for fractional-second precision; we always
                // store μs precision so the value is parsed and ignored.
                self.skip_optional_size();
                Ok(DataType::Timestamp)
            }
            Some(Token::Date) => {
                self.bump();
                Ok(DataType::Date)
            }
            Some(Token::Time) => {
                self.bump();
                self.skip_optional_size();
                Ok(DataType::Time)
            }
            Some(Token::Interval) => {
                self.bump();
                Ok(DataType::Interval)
            }
            other => bail!("expected data type, got {other:?}"),
        }
    }

    /// Consume `(N)` after a type if present (e.g. `CHAR(120)`). Stored size
    /// is not enforced at runtime — VARCHAR is unbounded already.
    fn skip_optional_size(&mut self) {
        if matches!(self.peek(), Some(Token::LParen)) {
            self.bump();
            // Allow integer or comma-separated digits (NUMERIC(p,s) etc).
            while !matches!(self.peek(), Some(Token::RParen) | None) {
                self.bump();
            }
            if matches!(self.peek(), Some(Token::RParen)) {
                self.bump();
            }
        }
    }

    fn parse_delete(&mut self) -> Result<Statement> {
        self.expect(&Token::Delete)?;
        self.expect(&Token::From)?;
        let table = self.parse_ident()?;
        let where_clause = if matches!(self.peek(), Some(Token::Where)) {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::Delete(DeleteStatement {
            table,
            where_clause,
        }))
    }

    fn parse_update(&mut self) -> Result<Statement> {
        self.expect(&Token::Update)?;
        let table = self.parse_ident()?;
        self.expect(&Token::Set)?;
        let mut assignments = Vec::new();
        loop {
            let column = self.parse_ident()?;
            self.expect(&Token::Eq)?;
            let value = self.parse_expr()?;
            assignments.push(Assignment { column, value });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        let where_clause = if matches!(self.peek(), Some(Token::Where)) {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::Update(UpdateStatement {
            table,
            assignments,
            where_clause,
        }))
    }

    /// Parse the FROM tree, building Joins left-associatively.
    fn parse_from_clause(&mut self) -> Result<FromClause> {
        let first = self.parse_table_ref()?;
        let mut node = FromClause::Table(first);
        loop {
            let join_type = match self.peek() {
                Some(Token::Inner) => {
                    self.bump();
                    self.expect(&Token::Join)?;
                    JoinType::Inner
                }
                // Bare JOIN means INNER JOIN.
                Some(Token::Join) => {
                    self.bump();
                    JoinType::Inner
                }
                Some(Token::Left) => {
                    self.bump();
                    self.expect(&Token::Join)?;
                    JoinType::Left
                }
                _ => break,
            };
            let right = self.parse_table_ref()?;
            self.expect(&Token::On)?;
            let on = self.parse_expr()?;
            node = FromClause::Join {
                left: Box::new(node),
                right,
                join_type,
                on,
            };
        }
        Ok(node)
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let name = self.parse_ident()?;
        // Optional alias: `AS u` or bare `u`. Bare alias must not collide
        // with any keyword that can legally follow a table ref (WHERE, JOIN,
        // INNER, LEFT, ON, semicolon, end-of-input).
        let alias = if matches!(self.peek(), Some(Token::As)) {
            self.bump();
            Some(self.parse_ident()?)
        } else if matches!(self.peek(), Some(Token::Ident(_))) {
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(TableRef { name, alias })
    }

    fn parse_ident(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::Ident(s)) => {
                let s = s.clone();
                self.bump();
                Ok(s)
            }
            other => bail!("expected identifier, got {other:?}"),
        }
    }

    // Expression precedence (lowest to highest): OR, AND, comparison, +/-, *//, unary
    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Or)) {
            self.bump();
            let right = self.parse_and()?;
            left = bin(left, BinaryOperator::Or, right);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_comparison()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.bump();
            let right = self.parse_comparison()?;
            left = bin(left, BinaryOperator::And, right);
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let left = self.parse_additive()?;
        // `expr IS [NOT] NULL` postfix. Lives at comparison precedence; no
        // chaining (e.g. `x IS NULL = TRUE` is rejected by the grammar).
        if matches!(self.peek(), Some(Token::Is)) {
            self.bump();
            let negated = if matches!(self.peek(), Some(Token::Not)) {
                self.bump();
                true
            } else {
                false
            };
            self.expect(&Token::Null)?;
            return Ok(Expr::IsNull {
                expr: Box::new(left),
                negated,
            });
        }
        // `expr IN (a, b, c)` — desugar to `expr=a OR expr=b OR expr=c`.
        // `expr NOT IN (...)` wraps the result in a NOT.
        if matches!(self.peek(), Some(Token::In))
            || (matches!(self.peek(), Some(Token::Not))
                && matches!(self.tokens.get(self.pos + 1), Some(Token::In)))
        {
            let negated = if matches!(self.peek(), Some(Token::Not)) {
                self.bump();
                true
            } else {
                false
            };
            self.bump(); // IN
            self.expect(&Token::LParen)?;
            let mut items = vec![self.parse_expr()?];
            while matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                items.push(self.parse_expr()?);
            }
            self.expect(&Token::RParen)?;
            // Build a left-folded OR chain: ((lhs=a OR lhs=b) OR lhs=c).
            let mut iter = items.into_iter();
            let first = iter.next().expect("IN list non-empty");
            let mut acc = bin(left.clone(), BinaryOperator::Eq, first);
            for v in iter {
                let eq = bin(left.clone(), BinaryOperator::Eq, v);
                acc = bin(acc, BinaryOperator::Or, eq);
            }
            return Ok(if negated {
                Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(acc),
                }
            } else {
                acc
            });
        }
        // `expr BETWEEN low AND high` — desugar to `expr >= low AND expr <= high`.
        // Cloning the left side is fine here; expressions are tree-shaped and
        // small. NOT BETWEEN gets the negation wrapped on the outside.
        if matches!(self.peek(), Some(Token::Between))
            || (matches!(self.peek(), Some(Token::Not))
                && matches!(self.tokens.get(self.pos + 1), Some(Token::Between)))
        {
            let negated = if matches!(self.peek(), Some(Token::Not)) {
                self.bump();
                true
            } else {
                false
            };
            self.bump(); // BETWEEN
            let low = self.parse_additive()?;
            self.expect(&Token::And)?;
            let high = self.parse_additive()?;
            let lhs1 = left.clone();
            let ge = bin(lhs1, BinaryOperator::Ge, low);
            let le = bin(left, BinaryOperator::Le, high);
            let combined = bin(ge, BinaryOperator::And, le);
            return Ok(if negated {
                Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(combined),
                }
            } else {
                combined
            });
        }
        let op = match self.peek() {
            Some(Token::Eq) => BinaryOperator::Eq,
            Some(Token::Ne) => BinaryOperator::Ne,
            Some(Token::Lt) => BinaryOperator::Lt,
            Some(Token::Le) => BinaryOperator::Le,
            Some(Token::Gt) => BinaryOperator::Gt,
            Some(Token::Ge) => BinaryOperator::Ge,
            _ => return Ok(left),
        };
        self.bump();
        let right = self.parse_additive()?;
        Ok(bin(left, op, right))
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinaryOperator::Add,
                Some(Token::Minus) => BinaryOperator::Sub,
                _ => return Ok(left),
            };
            self.bump();
            let right = self.parse_multiplicative()?;
            left = bin(left, op, right);
        }
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Asterisk) => BinaryOperator::Mul,
                Some(Token::Slash) => BinaryOperator::Div,
                _ => return Ok(left),
            };
            self.bump();
            let right = self.parse_unary()?;
            left = bin(left, op, right);
        }
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Token::Not) => {
                self.bump();
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            Some(Token::Minus) => {
                self.bump();
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Neg,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Token::Timestamp) => {
                // `TIMESTAMP 'YYYY-MM-DD HH:MM:SS[.fff]'` — typed literal.
                self.bump();
                let s = self.expect_string_literal_after("TIMESTAMP")?;
                let micros = parse_timestamp_literal(&s)?;
                Ok(Expr::Literal(Literal::Timestamp(micros)))
            }
            Some(Token::Date) => {
                self.bump();
                let s = self.expect_string_literal_after("DATE")?;
                let days = parse_date_literal(&s)?;
                Ok(Expr::Literal(Literal::Date(days)))
            }
            Some(Token::Time) => {
                self.bump();
                let s = self.expect_string_literal_after("TIME")?;
                let micros = parse_time_literal(&s)?;
                Ok(Expr::Literal(Literal::Time(micros)))
            }
            Some(Token::Interval) => {
                self.bump();
                let s = self.expect_string_literal_after("INTERVAL")?;
                let (months, days, micros) = parse_interval_literal(&s)?;
                Ok(Expr::Literal(Literal::Interval {
                    months,
                    days,
                    micros,
                }))
            }
            Some(Token::CurrentTimestamp) => {
                // SQL standard: `current_timestamp` (no parens). Normalised to
                // a `now()` function call so analyzer treats both forms the
                // same.
                self.bump();
                Ok(Expr::FuncCall {
                    name: "now".to_string(),
                    args: FuncArgs::Exprs(Vec::new()),
                })
            }
            Some(Token::Integer(n)) => {
                let n = *n;
                self.bump();
                Ok(Expr::Literal(Literal::Integer(n)))
            }
            Some(Token::Float(f)) => {
                let f = *f;
                self.bump();
                Ok(Expr::Literal(Literal::Float(f)))
            }
            Some(Token::String(s)) => {
                let s = s.clone();
                self.bump();
                Ok(Expr::Literal(Literal::String(s)))
            }
            Some(Token::Null) => {
                self.bump();
                Ok(Expr::Literal(Literal::Null))
            }
            Some(Token::True) => {
                self.bump();
                Ok(Expr::Literal(Literal::Boolean(true)))
            }
            Some(Token::False) => {
                self.bump();
                Ok(Expr::Literal(Literal::Boolean(false)))
            }
            Some(Token::Ident(s)) => {
                let s = s.clone();
                self.bump();
                // Function call: ident immediately followed by `(`.
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.bump();
                    let args = if matches!(self.peek(), Some(Token::Asterisk)) {
                        self.bump();
                        FuncArgs::Star
                    } else if matches!(self.peek(), Some(Token::RParen)) {
                        FuncArgs::Exprs(Vec::new())
                    } else {
                        let mut exprs = vec![self.parse_expr()?];
                        while matches!(self.peek(), Some(Token::Comma)) {
                            self.bump();
                            exprs.push(self.parse_expr()?);
                        }
                        FuncArgs::Exprs(exprs)
                    };
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::FuncCall { name: s, args });
                }
                // Optional `.ident` for qualified column references.
                if matches!(self.peek(), Some(Token::Dot)) {
                    self.bump();
                    let col = self.parse_ident()?;
                    Ok(Expr::Column {
                        qualifier: Some(s),
                        name: col,
                    })
                } else {
                    Ok(Expr::Column {
                        qualifier: None,
                        name: s,
                    })
                }
            }
            Some(Token::LParen) => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            other => bail!("unexpected token in expression: {other:?}"),
        }
    }
}

fn bin(l: Expr, op: BinaryOperator, r: Expr) -> Expr {
    Expr::BinaryOp {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

/// Parse a TIMESTAMP literal string into microseconds from the PostgreSQL
/// epoch (2000-01-01 UTC midnight). Accepted forms:
///   - `YYYY-MM-DD HH:MM:SS`
///   - `YYYY-MM-DD HH:MM:SS.ffffff` (1-6 fractional digits, μs precision)
///   - `YYYY-MM-DD` (treated as midnight)
/// Naive parsing only — no time-zone offsets here (those land with TIMESTAMPTZ).
pub fn parse_timestamp_literal(s: &str) -> Result<i64> {
    use chrono::NaiveDateTime;
    let trimmed = s.trim();
    // Try several format strings in order; first hit wins.
    let candidates: &[&str] = &[
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ];
    let dt = candidates
        .iter()
        .find_map(|fmt| NaiveDateTime::parse_from_str(trimmed, fmt).ok())
        .or_else(|| {
            chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d")
                .ok()
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        })
        .ok_or_else(|| anyhow::anyhow!("invalid timestamp literal: {trimmed:?}"))?;
    // PostgreSQL epoch: 2000-01-01 00:00:00 UTC.
    let pg_epoch = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let delta = dt.signed_duration_since(pg_epoch);
    delta
        .num_microseconds()
        .ok_or_else(|| anyhow::anyhow!("timestamp out of range"))
}

impl Parser {
    /// Read the string literal that follows a typed-literal keyword
    /// (e.g. `TIMESTAMP '...'`, `DATE '...'`). The keyword itself must
    /// already have been consumed.
    fn expect_string_literal_after(&mut self, kw: &'static str) -> Result<String> {
        match self.peek() {
            Some(Token::String(s)) => {
                let s = s.clone();
                self.bump();
                Ok(s)
            }
            other => bail!("expected string literal after {kw}, got {other:?}"),
        }
    }
}

/// `DATE 'YYYY-MM-DD'` → days since 2000-01-01.
pub fn parse_date_literal(s: &str) -> Result<i32> {
    let d = chrono::NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("invalid date literal {s:?}: {e}"))?;
    let pg_epoch = chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    let delta = d.signed_duration_since(pg_epoch).num_days();
    if delta < i32::MIN as i64 || delta > i32::MAX as i64 {
        bail!("date out of i32 range");
    }
    Ok(delta as i32)
}

/// `TIME 'HH:MM:SS[.fff]'` → microseconds since 00:00:00.
pub fn parse_time_literal(s: &str) -> Result<i64> {
    let candidates: &[&str] = &["%H:%M:%S%.f", "%H:%M:%S", "%H:%M"];
    let t = candidates
        .iter()
        .find_map(|fmt| chrono::NaiveTime::parse_from_str(s.trim(), fmt).ok())
        .ok_or_else(|| anyhow::anyhow!("invalid time literal: {s:?}"))?;
    let micros = t.signed_duration_since(chrono::NaiveTime::MIN)
        .num_microseconds()
        .ok_or_else(|| anyhow::anyhow!("time out of range"))?;
    Ok(micros)
}

/// `INTERVAL '...'` → `(months, days, micros)`. Accepts a small subset of
/// PostgreSQL's interval syntax that covers the common cases:
///   - `'<n> year[s]'` / `'<n> month[s]'` / `'<n> week[s]'`
///   - `'<n> day[s]'` / `'<n> hour[s]'` / `'<n> minute[s]'` / `'<n> second[s]'`
///   - `'<n> millisecond[s]'` / `'<n> microsecond[s]'`
///   - HH:MM:SS form (e.g. `'12:30:00'` for 12 hours 30 minutes)
///   - Combinations separated by spaces (`'1 day 12:00:00'`)
/// Negative units like `'-3 days'` and ago / mixed signs are *not* supported
/// yet; postpone to a richer interval parser when tests demand it.
pub fn parse_interval_literal(s: &str) -> Result<(i32, i32, i64)> {
    let mut months: i32 = 0;
    let mut days: i32 = 0;
    let mut micros: i64 = 0;
    let trimmed = s.trim();
    // Tokenise on whitespace, then walk pairs.
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        // HH:MM:SS form?
        if let Some(t) = parse_hms(tok) {
            micros += t;
            i += 1;
            continue;
        }
        // <number> <unit>
        if i + 1 >= tokens.len() {
            bail!("interval token without unit: {tok:?}");
        }
        let n: i64 = tok
            .parse()
            .map_err(|_| anyhow::anyhow!("bad interval number: {tok:?}"))?;
        let unit = tokens[i + 1].to_ascii_lowercase();
        let unit = unit.trim_end_matches('s'); // plural → singular
        match unit {
            "year" => months += (n * 12) as i32,
            "month" | "mon" => months += n as i32,
            "week" => days += (n * 7) as i32,
            "day" => days += n as i32,
            "hour" | "hr" | "h" => micros += n * 3_600_000_000,
            "minute" | "min" | "m" => micros += n * 60_000_000,
            "second" | "sec" | "s" => micros += n * 1_000_000,
            "millisecond" | "ms" => micros += n * 1_000,
            "microsecond" | "us" => micros += n,
            other => bail!("unknown interval unit: {other:?}"),
        }
        i += 2;
    }
    Ok((months, days, micros))
}

fn parse_hms(s: &str) -> Option<i64> {
    // Accept `HH:MM` and `HH:MM:SS[.fff]`.
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let h: i64 = parts[0].parse().ok()?;
    let m: i64 = parts[1].parse().ok()?;
    let mut micros = h * 3_600_000_000 + m * 60_000_000;
    if let Some(sec_part) = parts.get(2) {
        let sec: f64 = sec_part.parse().ok()?;
        micros += (sec * 1_000_000.0).round() as i64;
    }
    Some(micros)
}

pub fn parse(sql: &str) -> Result<Statement> {
    let tokens = Lexer::new(sql).tokenize()?;
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit_int(n: i64) -> Expr {
        Expr::Literal(Literal::Integer(n))
    }
    fn col(s: &str) -> Expr {
        Expr::Column {
            qualifier: None,
            name: s.into(),
        }
    }

    #[test]
    fn select_star() {
        let s = parse("SELECT * FROM users").unwrap();
        assert_eq!(
            s,
            Statement::Select(SelectStatement {
                columns: vec![SelectColumn::Asterisk],
                from: FromClause::Table(TableRef {
                    name: "users".into(),
                    alias: None
                }),
                where_clause: None,
                group_by: Vec::new(),
                having: None,
                order_by: Vec::new(),
                limit: None,
                distinct: false,
            })
        );
    }

    #[test]
    fn parse_select_distinct() {
        let s = parse("SELECT DISTINCT name FROM t").unwrap();
        let Statement::Select(sel) = s else { panic!() };
        assert!(sel.distinct);
    }

    #[test]
    fn parse_is_null_and_is_not_null() {
        let s = parse("SELECT id FROM t WHERE name IS NULL").unwrap();
        let Statement::Select(sel) = s else { panic!() };
        let w = sel.where_clause.unwrap();
        let Expr::IsNull { negated, .. } = w else { panic!("expected IsNull") };
        assert!(!negated);

        let s = parse("SELECT id FROM t WHERE name IS NOT NULL").unwrap();
        let Statement::Select(sel) = s else { panic!() };
        let Expr::IsNull { negated, .. } = sel.where_clause.unwrap() else { panic!() };
        assert!(negated);
    }

    #[test]
    fn parse_order_by_and_limit() {
        let s = parse("SELECT id FROM t ORDER BY name DESC, id LIMIT 5").unwrap();
        let Statement::Select(sel) = s else { panic!() };
        assert_eq!(sel.order_by.len(), 2);
        assert_eq!(sel.order_by[0].dir, OrderDir::Desc);
        assert_eq!(sel.order_by[1].dir, OrderDir::Asc);
        assert_eq!(sel.limit, Some(5));
    }

    #[test]
    fn parse_count_star() {
        let s = parse("SELECT COUNT(*) FROM t").unwrap();
        let Statement::Select(sel) = s else { panic!() };
        let SelectColumn::Expr { expr: Expr::FuncCall { name, args }, .. } = &sel.columns[0]
        else {
            panic!()
        };
        assert_eq!(name.to_uppercase(), "COUNT");
        assert!(matches!(args, FuncArgs::Star));
    }

    #[test]
    fn parse_group_by_having() {
        let s =
            parse("SELECT product, SUM(quantity) FROM sales GROUP BY product HAVING SUM(quantity) > 10")
                .unwrap();
        let Statement::Select(sel) = s else { panic!() };
        assert_eq!(sel.group_by.len(), 1);
        assert!(sel.having.is_some());
    }

    #[test]
    fn select_columns_with_where() {
        let s = parse("SELECT id, name FROM users WHERE id > 10").unwrap();
        let Statement::Select(sel) = s else {
            panic!()
        };
        let FromClause::Table(t) = &sel.from else { panic!() };
        assert_eq!(t.name, "users");
        assert_eq!(sel.columns.len(), 2);
        assert!(matches!(sel.where_clause, Some(_)));
    }

    #[test]
    fn parse_inner_join_with_aliases() {
        let s = parse(
            "SELECT u.id, o.product FROM users u INNER JOIN orders o ON u.id = o.user_id",
        )
        .unwrap();
        let Statement::Select(sel) = s else { panic!() };
        let FromClause::Join { left, right, join_type, .. } = sel.from else {
            panic!("expected join")
        };
        assert_eq!(join_type, JoinType::Inner);
        assert_eq!(right.name, "orders");
        assert_eq!(right.alias.as_deref(), Some("o"));
        let FromClause::Table(t) = *left else { panic!() };
        assert_eq!(t.name, "users");
        assert_eq!(t.alias.as_deref(), Some("u"));
    }

    #[test]
    fn parse_left_join_three_way() {
        let s = parse(
            "SELECT a.x FROM a JOIN b ON a.id = b.id LEFT JOIN c ON b.id = c.id",
        )
        .unwrap();
        let Statement::Select(sel) = s else { panic!() };
        // Outermost is LEFT JOIN with c.
        let FromClause::Join { left, right, join_type, .. } = sel.from else { panic!() };
        assert_eq!(join_type, JoinType::Left);
        assert_eq!(right.name, "c");
        // Inner is INNER JOIN(a, b).
        let FromClause::Join { join_type: inner_jt, .. } = *left else { panic!() };
        assert_eq!(inner_jt, JoinType::Inner);
    }

    #[test]
    fn qualified_column_in_expression() {
        let s = parse("SELECT u.name FROM users u").unwrap();
        let Statement::Select(sel) = s else { panic!() };
        let SelectColumn::Expr { expr: Expr::Column { qualifier, name }, .. } = &sel.columns[0]
        else {
            panic!()
        };
        assert_eq!(qualifier.as_deref(), Some("u"));
        assert_eq!(name, "name");
    }

    #[test]
    fn precedence_add_then_mul() {
        // a + b * c → Add(a, Mul(b, c))
        let s = parse("SELECT a + b * c FROM t").unwrap();
        let Statement::Select(sel) = s else {
            panic!()
        };
        let SelectColumn::Expr { expr: e, .. } = &sel.columns[0] else {
            panic!()
        };
        let Expr::BinaryOp {
            left,
            op,
            right,
        } = e
        else {
            panic!("expected binary op, got {e:?}")
        };
        assert_eq!(*op, BinaryOperator::Add);
        assert_eq!(**left, col("a"));
        // right side is Mul(b, c)
        let Expr::BinaryOp {
            left: rl,
            op: rop,
            right: rr,
        } = right.as_ref()
        else {
            panic!()
        };
        assert_eq!(*rop, BinaryOperator::Mul);
        assert_eq!(**rl, col("b"));
        assert_eq!(**rr, col("c"));
    }

    #[test]
    fn not_binds_inside_or() {
        // NOT (x = 1 OR y = 2)
        let s = parse("SELECT * FROM t WHERE NOT (x = 1 OR y = 2)").unwrap();
        let Statement::Select(sel) = s else {
            panic!()
        };
        let w = sel.where_clause.unwrap();
        // outermost should be UnaryOp::Not
        let Expr::UnaryOp { op, expr } = w else {
            panic!("expected NOT")
        };
        assert_eq!(op, UnaryOperator::Not);
        // inside parens: Or(...)
        let Expr::BinaryOp { op: inner_op, .. } = *expr else {
            panic!()
        };
        assert_eq!(inner_op, BinaryOperator::Or);
    }

    #[test]
    fn insert_values_with_null() {
        let s = parse("INSERT INTO users VALUES (1, 'Alice', NULL)").unwrap();
        let Statement::Insert(ins) = s else {
            panic!()
        };
        assert_eq!(ins.table, "users");
        assert_eq!(ins.rows.len(), 1);
        let row = &ins.rows[0];
        assert_eq!(row.len(), 3);
        assert_eq!(row[0], lit_int(1));
        assert_eq!(row[1], Expr::Literal(Literal::String("Alice".into())));
        assert_eq!(row[2], Expr::Literal(Literal::Null));
    }

    #[test]
    fn insert_multi_row() {
        let s = parse("INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
        let Statement::Insert(ins) = s else { panic!() };
        assert_eq!(ins.rows.len(), 3);
        assert_eq!(ins.rows[1][0], lit_int(2));
    }

    #[test]
    fn create_table_with_default_and_pk_and_char() {
        // sysbench-style DDL — should parse without error.
        let s = parse(
            "CREATE TABLE sbtest1 (id INTEGER NOT NULL, k INTEGER DEFAULT 0 NOT NULL, c CHAR(120) DEFAULT '' NOT NULL, pad CHAR(60) DEFAULT '' NOT NULL, PRIMARY KEY (id))"
        ).unwrap();
        let Statement::CreateTable(c) = s else { panic!() };
        assert_eq!(c.columns.len(), 4);
        assert_eq!(c.columns[0].name, "id");
        assert!(!c.columns[0].nullable);
        // CHAR maps to VARCHAR.
        assert_eq!(c.columns[2].data_type, DataType::Varchar);
    }

    #[test]
    fn create_index_parses() {
        let s = parse("CREATE INDEX k_1 ON sbtest1(k)").unwrap();
        let Statement::CreateIndex(c) = s else { panic!() };
        assert_eq!(c.name, "k_1");
        assert_eq!(c.table, "sbtest1");
        assert_eq!(c.column, "k");
    }

    #[test]
    fn create_table_two_cols() {
        let s = parse("CREATE TABLE users (id INT, name VARCHAR)").unwrap();
        let Statement::CreateTable(c) = s else {
            panic!()
        };
        assert_eq!(c.table, "users");
        assert_eq!(
            c.columns,
            vec![
                ColumnDef {
                    name: "id".into(),
                    data_type: DataType::Int,
                    nullable: true,
                },
                ColumnDef {
                    name: "name".into(),
                    data_type: DataType::Varchar,
                    nullable: true,
                },
            ]
        );
    }

    #[test]
    fn create_table_not_null() {
        let s = parse("CREATE TABLE users (id INT NOT NULL, name VARCHAR)").unwrap();
        let Statement::CreateTable(c) = s else {
            panic!()
        };
        assert!(!c.columns[0].nullable);
        assert!(c.columns[1].nullable);
    }

    #[test]
    fn trailing_garbage_errors() {
        // `FROM t blah` now parses as `FROM t AS blah` (bare-alias). Use a
        // token that can't be an alias to exercise the trailing-garbage path.
        assert!(parse("SELECT * FROM t 123").is_err());
    }

    #[test]
    fn empty_input_errors() {
        assert!(parse("").is_err());
    }

    #[test]
    fn delete_with_where() {
        let s = parse("DELETE FROM users WHERE id = 1").unwrap();
        let Statement::Delete(d) = s else { panic!() };
        assert_eq!(d.table, "users");
        assert!(d.where_clause.is_some());
    }

    #[test]
    fn delete_no_where() {
        let s = parse("DELETE FROM users").unwrap();
        let Statement::Delete(d) = s else { panic!() };
        assert!(d.where_clause.is_none());
    }

    #[test]
    fn update_two_assignments_with_where() {
        let s = parse("UPDATE users SET name = 'A', id = 5 WHERE id = 1").unwrap();
        let Statement::Update(u) = s else { panic!() };
        assert_eq!(u.assignments.len(), 2);
        assert_eq!(u.assignments[0].column, "name");
        assert_eq!(u.assignments[1].column, "id");
        assert!(u.where_clause.is_some());
    }
}
