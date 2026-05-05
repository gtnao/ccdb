// Storage modules are wired in but not yet used by the analyzer; day06 will
// connect them via the executor.
#![allow(dead_code)]

mod analyzer;
mod ast;
mod buffer_pool;
mod catalog;
mod disk;
mod lexer;
mod page;
mod parser;
mod table;
mod tuple;

use anyhow::Result;

use analyzer::analyze;
use catalog::Catalog;
use parser::parse;

fn main() -> Result<()> {
    let cat = Catalog::new();

    println!("--- valid queries ---");
    for sql in [
        "SELECT * FROM users",
        "SELECT id, name FROM users WHERE id > 10",
        "SELECT id + 1 FROM users",
        "INSERT INTO users VALUES (1, 'Alice')",
        "INSERT INTO users VALUES (2, NULL)",
        "CREATE TABLE accounts (uid INT, kind VARCHAR)",
    ] {
        match parse(sql).and_then(|s| analyze(&cat, &s)) {
            Ok(_) => println!("OK   {sql}"),
            Err(e) => println!("ERR  {sql}\n     -> {e}"),
        }
    }

    println!("\n--- expected errors ---");
    for sql in [
        "SELECT * FROM nope",
        "SELECT zz FROM users",
        "INSERT INTO users VALUES (1)",
        "INSERT INTO users VALUES ('a', 1)",
        "INSERT INTO users VALUES (NULL, 'Alice')",
        "CREATE TABLE users (id INT)",
    ] {
        match parse(sql).and_then(|s| analyze(&cat, &s)) {
            Ok(_) => println!("UNEXPECTED OK  {sql}"),
            Err(e) => println!("ERR  {sql}\n     -> {e}"),
        }
    }
    Ok(())
}
