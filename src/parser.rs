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
            Some(Token::Create) => self.parse_create_table()?,
            other => bail!("expected SELECT / INSERT / CREATE, got {other:?}"),
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
        let mut columns = Vec::new();
        loop {
            if matches!(self.peek(), Some(Token::Asterisk)) {
                self.bump();
                columns.push(SelectColumn::Asterisk);
            } else {
                columns.push(SelectColumn::Expr(self.parse_expr()?));
            }
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Token::From)?;
        let from = self.parse_table_ref()?;
        let where_clause = if matches!(self.peek(), Some(Token::Where)) {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::Select(SelectStatement {
            columns,
            from,
            where_clause,
        }))
    }

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect(&Token::Insert)?;
        self.expect(&Token::Into)?;
        let table = self.parse_ident()?;
        self.expect(&Token::Values)?;
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
        Ok(Statement::Insert(InsertStatement { table, values }))
    }

    fn parse_create_table(&mut self) -> Result<Statement> {
        self.expect(&Token::Create)?;
        self.expect(&Token::Table)?;
        let table = self.parse_ident()?;
        self.expect(&Token::LParen)?;
        let mut columns = Vec::new();
        loop {
            let name = self.parse_ident()?;
            let data_type = self.parse_data_type()?;
            columns.push(ColumnDef { name, data_type });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        Ok(Statement::CreateTable(CreateTableStatement {
            table,
            columns,
        }))
    }

    fn parse_data_type(&mut self) -> Result<DataType> {
        match self.peek() {
            Some(Token::Int) => {
                self.bump();
                Ok(DataType::Int)
            }
            Some(Token::Varchar) => {
                self.bump();
                Ok(DataType::Varchar)
            }
            other => bail!("expected data type, got {other:?}"),
        }
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let name = self.parse_ident()?;
        // Alias parsing (`AS u` / bare `u`) is deferred — qualified column refs
        // (`u.col`) aren't supported yet, so the alias would be unused.
        Ok(TableRef { name, alias: None })
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
            Some(Token::Integer(n)) => {
                let n = *n;
                self.bump();
                Ok(Expr::Literal(Literal::Integer(n)))
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
                Ok(Expr::Column(s))
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
        Expr::Column(s.into())
    }

    #[test]
    fn select_star() {
        let s = parse("SELECT * FROM users").unwrap();
        assert_eq!(
            s,
            Statement::Select(SelectStatement {
                columns: vec![SelectColumn::Asterisk],
                from: TableRef {
                    name: "users".into(),
                    alias: None
                },
                where_clause: None,
            })
        );
    }

    #[test]
    fn select_columns_with_where() {
        let s = parse("SELECT id, name FROM users WHERE id > 10").unwrap();
        let Statement::Select(sel) = s else {
            panic!()
        };
        assert_eq!(sel.from.name, "users");
        assert_eq!(sel.columns.len(), 2);
        assert!(matches!(sel.where_clause, Some(_)));
    }

    #[test]
    fn precedence_add_then_mul() {
        // a + b * c → Add(a, Mul(b, c))
        let s = parse("SELECT a + b * c FROM t").unwrap();
        let Statement::Select(sel) = s else {
            panic!()
        };
        let SelectColumn::Expr(e) = &sel.columns[0] else {
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
        assert_eq!(ins.values.len(), 3);
        assert_eq!(ins.values[0], lit_int(1));
        assert_eq!(
            ins.values[1],
            Expr::Literal(Literal::String("Alice".into()))
        );
        assert_eq!(ins.values[2], Expr::Literal(Literal::Null));
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
                    data_type: DataType::Int
                },
                ColumnDef {
                    name: "name".into(),
                    data_type: DataType::Varchar
                },
            ]
        );
    }

    #[test]
    fn trailing_garbage_errors() {
        assert!(parse("SELECT * FROM t blah").is_err());
    }

    #[test]
    fn empty_input_errors() {
        assert!(parse("").is_err());
    }
}
