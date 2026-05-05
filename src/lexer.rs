use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords (matched case-insensitively)
    Select,
    From,
    Where,
    Insert,
    Into,
    Values,
    Delete,
    Update,
    Set,
    Begin,
    Commit,
    Rollback,
    Checkpoint,
    Create,
    Table,
    Int,
    Varchar,
    Double,
    Timestamp,
    And,
    Or,
    Not,
    Null,
    True,
    False,
    Join,
    Inner,
    Left,
    On,
    As,
    Group,
    By,
    Having,
    Order,
    Asc,
    Desc,
    Limit,
    Is,
    Default,
    Char,
    Primary,
    Key,
    Index,
    Between,
    Distinct,
    In,
    If,
    Exists,

    // Identifiers and literals
    Ident(String),
    Integer(i64),
    Float(f64),
    String(String),

    // Punctuation / operators
    Asterisk,  // *
    Comma,     // ,
    Semicolon, // ;
    LParen,    // (
    RParen,    // )
    Dot,       // .
    Eq,        // =
    Ne,        // <>
    Lt,        // <
    Le,        // <=
    Gt,        // >
    Ge,        // >=
    Plus,      // +
    Minus,     // -
    Slash,     // /
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(src: &str) -> Self {
        Self {
            input: src.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.bump();
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.peek().is_none() {
                return Ok(out);
            }
            out.push(self.next_token()?);
        }
    }

    fn next_token(&mut self) -> Result<Token> {
        let c = self.peek().expect("caller already checked end-of-input");
        let tok = match c {
            '*' => {
                self.bump();
                Token::Asterisk
            }
            ',' => {
                self.bump();
                Token::Comma
            }
            '.' => {
                self.bump();
                Token::Dot
            }
            ';' => {
                self.bump();
                Token::Semicolon
            }
            '(' => {
                self.bump();
                Token::LParen
            }
            ')' => {
                self.bump();
                Token::RParen
            }
            '+' => {
                self.bump();
                Token::Plus
            }
            '-' => {
                self.bump();
                Token::Minus
            }
            '/' => {
                self.bump();
                Token::Slash
            }
            '=' => {
                self.bump();
                Token::Eq
            }
            '<' => {
                self.bump();
                match self.peek() {
                    Some('=') => {
                        self.bump();
                        Token::Le
                    }
                    Some('>') => {
                        self.bump();
                        Token::Ne
                    }
                    _ => Token::Lt,
                }
            }
            '>' => {
                self.bump();
                match self.peek() {
                    Some('=') => {
                        self.bump();
                        Token::Ge
                    }
                    _ => Token::Gt,
                }
            }
            '\'' => self.read_string()?,
            d if d.is_ascii_digit() => self.read_integer()?,
            a if a.is_ascii_alphabetic() || a == '_' => self.read_word(),
            _ => bail!("unexpected character: {c:?}"),
        };
        Ok(tok)
    }

    fn read_string(&mut self) -> Result<Token> {
        self.bump(); // opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('\'') => return Ok(Token::String(s)),
                Some(c) => s.push(c),
                None => bail!("unterminated string literal"),
            }
        }
    }

    fn read_integer(&mut self) -> Result<Token> {
        let mut s = String::new();
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            s.push(self.bump().unwrap());
        }
        // Float literal: `123.45`. We require at least one digit after `.`
        // so `t.col` (table-qualified column) is still a valid token sequence.
        if matches!(self.peek(), Some('.')) {
            // Peek one more to confirm a digit follows.
            let next = self.input.get(self.pos + 1).copied();
            if matches!(next, Some(c) if c.is_ascii_digit()) {
                s.push(self.bump().unwrap()); // consume `.`
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    s.push(self.bump().unwrap());
                }
                return Ok(Token::Float(s.parse()?));
            }
        }
        Ok(Token::Integer(s.parse()?))
    }

    fn read_word(&mut self) -> Token {
        let mut s = String::new();
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
            s.push(self.bump().unwrap());
        }
        match s.to_ascii_uppercase().as_str() {
            "SELECT" => Token::Select,
            "FROM" => Token::From,
            "WHERE" => Token::Where,
            "INSERT" => Token::Insert,
            "INTO" => Token::Into,
            "VALUES" => Token::Values,
            "DELETE" => Token::Delete,
            "UPDATE" => Token::Update,
            "SET" => Token::Set,
            "BEGIN" => Token::Begin,
            "COMMIT" => Token::Commit,
            "ROLLBACK" => Token::Rollback,
            "CHECKPOINT" => Token::Checkpoint,
            "CREATE" => Token::Create,
            "TABLE" => Token::Table,
            "INT" | "INTEGER" => Token::Int,
            "VARCHAR" => Token::Varchar,
            "DOUBLE" | "FLOAT" => Token::Double,
            "TIMESTAMP" => Token::Timestamp,
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "NULL" => Token::Null,
            "TRUE" => Token::True,
            "FALSE" => Token::False,
            "JOIN" => Token::Join,
            "INNER" => Token::Inner,
            "LEFT" => Token::Left,
            "ON" => Token::On,
            "AS" => Token::As,
            "GROUP" => Token::Group,
            "BY" => Token::By,
            "HAVING" => Token::Having,
            "ORDER" => Token::Order,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "LIMIT" => Token::Limit,
            "IS" => Token::Is,
            "DEFAULT" => Token::Default,
            "CHAR" => Token::Char,
            "PRIMARY" => Token::Primary,
            "KEY" => Token::Key,
            "INDEX" => Token::Index,
            "BETWEEN" => Token::Between,
            "DISTINCT" => Token::Distinct,
            "IN" => Token::In,
            "IF" => Token::If,
            "EXISTS" => Token::Exists,
            _ => Token::Ident(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(s: &str) -> Vec<Token> {
        Lexer::new(s).tokenize().unwrap()
    }

    #[test]
    fn keywords_are_case_insensitive() {
        assert_eq!(lex("select Select SELECT"), vec![Token::Select; 3]);
    }

    #[test]
    fn integers_strings_idents() {
        assert_eq!(
            lex("123 'foo bar' baz_qux"),
            vec![
                Token::Integer(123),
                Token::String("foo bar".into()),
                Token::Ident("baz_qux".into()),
            ]
        );
    }

    #[test]
    fn comparison_operators() {
        assert_eq!(
            lex("= <> < <= > >="),
            vec![Token::Eq, Token::Ne, Token::Lt, Token::Le, Token::Gt, Token::Ge,]
        );
    }

    #[test]
    fn arithmetic_and_punct() {
        assert_eq!(
            lex("+ - * / , ; ( )"),
            vec![
                Token::Plus,
                Token::Minus,
                Token::Asterisk,
                Token::Slash,
                Token::Comma,
                Token::Semicolon,
                Token::LParen,
                Token::RParen,
            ]
        );
    }

    #[test]
    fn unterminated_string_errors() {
        assert!(Lexer::new("'oops").tokenize().is_err());
    }
}
