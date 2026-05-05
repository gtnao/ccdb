// Catalog/analyzer fields and a few accessors are part of the public surface
// even when the immediate executor doesn't read them. See analyzer.rs.
#![allow(dead_code)]

mod analyzer;
mod ast;
mod bootstrap;
mod buffer_pool;
mod catalog;
mod checkpoint;
mod clog;
mod disk;
mod executor;
mod instance;
mod lexer;
mod lock_manager;
mod page;
mod parser;
mod protocol;
mod recovery;
mod transaction;
mod transaction_manager;
mod tuple;
mod visibility;
mod wal;

use anyhow::Result;

use instance::Instance;

fn main() -> Result<()> {
    // `--init` clears table.db and wal.log so a fresh server starts from
    // empty state. Without it, on-disk data and WAL persist across runs
    // (recovery handling lands in the next day).
    let init = std::env::args().any(|a| a == "--init");
    let instance = Instance::new(init)?;
    instance.start()
}
