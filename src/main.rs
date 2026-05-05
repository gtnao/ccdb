// Storage modules are wired in but unused at this stage; the analyzer in
// day05 will start tying them to the SQL frontend.
#![allow(dead_code)]

mod ast;
mod buffer_pool;
mod disk;
mod lexer;
mod page;
mod parser;
mod table;
mod tuple;

use anyhow::Result;

fn main() -> Result<()> {
    let queries = [
        "SELECT * FROM users",
        "SELECT id, name FROM users WHERE id > 10",
        "SELECT id FROM users WHERE id = 1 AND name = 'Alice'",
        "SELECT a + b * c FROM t",
        "SELECT * FROM t WHERE NOT (x = 1 OR y = 2)",
        "SELECT * FROM t WHERE active = TRUE",
        "INSERT INTO users VALUES (1, 'Alice', NULL)",
        "CREATE TABLE users (id INT, name VARCHAR)",
    ];

    for q in queries {
        println!("SQL: {q}");
        match parser::parse(q) {
            Ok(stmt) => println!("  OK: {stmt:?}"),
            Err(e) => println!("  ERR: {e}"),
        }
    }

    println!("\n--- error case ---");
    println!("SQL: SELECT FROM");
    if let Err(e) = parser::parse("SELECT FROM") {
        println!("  ERR: {e}");
    }
    Ok(())
}
