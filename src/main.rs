// `Catalog::find_table` returns (id, &def) but only the executor uses the def
// directly via id; reachable callers don't all flow through the same pattern,
// so suppress noise here.
#![allow(dead_code)]

mod analyzer;
mod ast;
mod buffer_pool;
mod catalog;
mod disk;
mod executor;
mod lexer;
mod page;
mod parser;
mod tuple;

use anyhow::Result;

use analyzer::analyze;
use buffer_pool::BufferPoolManager;
use catalog::Catalog;
use disk::DiskManager;
use executor::{Output, execute};
use parser::parse;

const DATA_FILE: &str = "table.db";

fn run(sql: &str, cat: &Catalog, bpm: &mut BufferPoolManager) -> Result<()> {
    println!("> {sql}");
    let stmt = parse(sql)?;
    let analyzed = analyze(cat, &stmt)?;
    match execute(bpm, cat, &analyzed)? {
        Output::Rows(rows) => {
            for r in &rows {
                println!("  {:?}", r.values);
            }
            println!("({} rows)", rows.len());
        }
        Output::Affected(n) => println!("({n} row(s) affected)"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ = std::fs::remove_file(DATA_FILE);
    let cat = Catalog::new();
    let disk = DiskManager::open(DATA_FILE)?;
    let mut bpm = BufferPoolManager::new(disk, 4);

    for sql in [
        "INSERT INTO users VALUES (1, 'Alice')",
        "INSERT INTO users VALUES (2, 'Bob')",
        "INSERT INTO users VALUES (3, 'Charlie')",
        "INSERT INTO users VALUES (10, NULL)",
        "INSERT INTO users VALUES (20, 'Eve')",
        "SELECT * FROM users",
        "SELECT name FROM users",
        "SELECT * FROM users WHERE id > 5",
        "SELECT id, name FROM users WHERE id >= 2 AND id <= 10",
        "SELECT id + 1 FROM users",
        "SELECT id FROM users WHERE name = 'Alice'", // NULL row excluded
    ] {
        run(sql, &cat, &mut bpm)?;
    }

    bpm.flush_all()?;
    println!("(flushed)");
    Ok(())
}
