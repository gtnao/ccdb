//! Server instance: TCP listener that speaks PG wire protocol.
//!
//! Multi-threaded: each accepted connection runs in its own thread. Shared
//! state (catalog, buffer pool) is reachable via cheap Arc clones; the
//! BufferPool internalizes its own locking.

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use anyhow::{Result, bail};

use crate::analyzer::{AnalyzedExpr, AnalyzedSelectItem, AnalyzedStatement, analyze};
use crate::buffer_pool::BufferPool;
use crate::catalog::Catalog;
use crate::disk::DiskManager;
use crate::executor::{self, Output, execute};
use crate::lock_manager::LockManager;
use crate::parser::parse;
use crate::protocol::{ColumnDesc, Connection, FrontendMessage};
use crate::transaction::Transaction;
use crate::tuple::{DataType, Value};

const DATA_FILE: &str = "table.db";
const DEFAULT_PORT: u16 = 5433;
const POOL_CAPACITY: usize = 64;

pub struct Instance {
    catalog: Arc<Catalog>,
    bpm: BufferPool,
    lock_manager: Arc<LockManager>,
}

impl Instance {
    pub fn new() -> Result<Self> {
        let _ = std::fs::remove_file(DATA_FILE);
        let disk = DiskManager::open(DATA_FILE)?;
        Ok(Self {
            catalog: Arc::new(Catalog::new()),
            bpm: BufferPool::new(disk, POOL_CAPACITY),
            lock_manager: Arc::new(LockManager::new()),
        })
    }

    pub fn start(&self) -> Result<()> {
        let addr = format!("127.0.0.1:{DEFAULT_PORT}");
        let listener = TcpListener::bind(&addr)?;
        eprintln!(
            "ccdb listening on {addr} (multi-threaded) — connect with: psql -h localhost -p {DEFAULT_PORT}"
        );

        for stream in listener.incoming() {
            let stream = stream?;
            eprintln!("client connected: {:?}", stream.peer_addr());
            let conn = Connection::new(stream);

            let catalog = Arc::clone(&self.catalog);
            let bpm = self.bpm.clone();
            let lock_manager = Arc::clone(&self.lock_manager);

            thread::spawn(move || {
                if let Err(e) = handle_client(conn, catalog, bpm, lock_manager) {
                    eprintln!("connection error: {e}");
                }
            });
        }
        Ok(())
    }
}

fn handle_client(
    mut conn: Connection<TcpStream>,
    catalog: Arc<Catalog>,
    bpm: BufferPool,
    lock_manager: Arc<LockManager>,
) -> Result<()> {
    let startup = conn.read_startup()?;
    eprintln!(
        "startup params: {:?} (thread {:?})",
        startup.params,
        thread::current().id()
    );

    conn.send_auth_ok()?;
    conn.send_parameter_status("server_version", "ccdb-0.0.1")?;
    conn.send_parameter_status("client_encoding", "UTF8")?;
    conn.send_backend_key_data(1, 0xC0FFEE)?;
    conn.send_ready_for_query()?;

    let mut tx = Transaction::new();

    let result = (|| -> Result<()> {
        loop {
            match conn.read_message()? {
                None => return Ok(()),
                Some(FrontendMessage::Terminate) => return Ok(()),
                Some(FrontendMessage::Unknown(t)) => {
                    eprintln!("ignoring unknown message type: 0x{t:02x}");
                    conn.send_ready_for_query()?;
                }
                Some(FrontendMessage::Query(sql)) => {
                    if sql.trim().is_empty() {
                        conn.send_empty_query()?;
                    } else if let Err(e) =
                        run_query(&sql, &mut conn, &bpm, &lock_manager, &catalog, &mut tx)
                    {
                        eprintln!("query error: {e}");
                        conn.send_error(&e.to_string())?;
                    }
                    conn.send_ready_for_query()?;
                }
            }
        }
    })();

    if tx.is_active() {
        if let Err(e) = executor::rollback(&bpm, &mut tx) {
            eprintln!("auto-rollback failed: {e}");
        }
    }
    // Release any locks left over (auto-rollback or aborted statement).
    let held = tx.take_held_locks();
    if !held.is_empty() {
        lock_manager.unlock_all(tx.id(), &held);
    }

    // Per-connection flush so writes are durable across sessions.
    bpm.flush_all()?;

    result
}

fn run_query(
    sql: &str,
    conn: &mut Connection<TcpStream>,
    bpm: &BufferPool,
    lm: &LockManager,
    catalog: &Catalog,
    tx: &mut Transaction,
) -> Result<()> {
    let stmt = parse(sql)?;
    let analyzed = analyze(catalog, &stmt)?;

    match &analyzed {
        AnalyzedStatement::Select(s) => {
            let columns: Vec<ColumnDesc> = s.select_items.iter().map(column_desc_for).collect();
            let out = execute(bpm, lm, catalog, &analyzed, tx)?;
            let rows = match out {
                Output::Rows(r) => r,
                other => bail!("SELECT yielded non-Rows output: {other:?}"),
            };
            conn.send_row_description(&columns)?;
            for row in &rows {
                let vals: Vec<Option<String>> = row.values.iter().map(value_to_text).collect();
                conn.send_data_row(&vals)?;
            }
            conn.send_command_complete(&format!("SELECT {}", rows.len()))?;
        }
        AnalyzedStatement::Insert(_) => {
            let n = expect_affected(execute(bpm, lm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("INSERT 0 {n}"))?;
        }
        AnalyzedStatement::Delete(_) => {
            let n = expect_affected(execute(bpm, lm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("DELETE {n}"))?;
        }
        AnalyzedStatement::Update(_) => {
            let n = expect_affected(execute(bpm, lm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("UPDATE {n}"))?;
        }
        AnalyzedStatement::Begin => {
            execute(bpm, lm, catalog, &analyzed, tx)?;
            conn.send_command_complete("BEGIN")?;
        }
        AnalyzedStatement::Commit => {
            execute(bpm, lm, catalog, &analyzed, tx)?;
            conn.send_command_complete("COMMIT")?;
        }
        AnalyzedStatement::Rollback => {
            execute(bpm, lm, catalog, &analyzed, tx)?;
            conn.send_command_complete("ROLLBACK")?;
        }
        AnalyzedStatement::CreateTable(_) => {
            bail!("CREATE TABLE is not yet wired up (catalog is read-only)")
        }
    }
    Ok(())
}

fn expect_affected(out: Output) -> Result<usize> {
    match out {
        Output::Affected(n) => Ok(n),
        other => bail!("expected affected-row count, got {other:?}"),
    }
}

fn column_desc_for(item: &AnalyzedSelectItem) -> ColumnDesc {
    let name = item
        .alias
        .clone()
        .unwrap_or_else(|| display_name(&item.expr));
    match item.expr.data_type() {
        Some(DataType::Int) => ColumnDesc::int(&name),
        Some(DataType::Varchar) => ColumnDesc::varchar(&name),
        Some(DataType::Bool) => ColumnDesc::bool(&name),
        // NULL literal without column context — Postgres convention is "text".
        None => ColumnDesc::varchar(&name),
    }
}

fn display_name(e: &AnalyzedExpr) -> String {
    match e {
        AnalyzedExpr::ColumnRef(c) => c.column_name.clone(),
        // Postgres reports "?column?" for unaliased computed expressions.
        _ => "?column?".to_string(),
    }
}

fn value_to_text(v: &Value) -> Option<String> {
    match v {
        Value::Int(n) => Some(n.to_string()),
        Value::Varchar(s) => Some(s.clone()),
        Value::Bool(b) => Some(if *b { "t" } else { "f" }.to_string()),
        Value::Null => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{AnalyzedColumnRef, AnalyzedLiteral, LiteralValue};

    #[test]
    fn value_to_text_handles_all_variants() {
        assert_eq!(value_to_text(&Value::Int(42)).as_deref(), Some("42"));
        assert_eq!(
            value_to_text(&Value::Varchar("x".into())).as_deref(),
            Some("x")
        );
        assert_eq!(value_to_text(&Value::Bool(true)).as_deref(), Some("t"));
        assert_eq!(value_to_text(&Value::Bool(false)).as_deref(), Some("f"));
        assert_eq!(value_to_text(&Value::Null), None);
    }

    #[test]
    fn column_desc_for_column_ref() {
        let item = AnalyzedSelectItem {
            expr: AnalyzedExpr::ColumnRef(AnalyzedColumnRef {
                rte_index: 0,
                column_index: 0,
                column_name: "id".into(),
                data_type: DataType::Int,
            }),
            alias: None,
        };
        let d = column_desc_for(&item);
        assert_eq!(d.name, "id");
        assert_eq!(d.type_oid, 23);
    }

    #[test]
    fn column_desc_for_unknown_type_falls_back_to_text() {
        let item = AnalyzedSelectItem {
            expr: AnalyzedExpr::Literal(AnalyzedLiteral {
                value: LiteralValue::Null,
                data_type: None,
            }),
            alias: None,
        };
        let d = column_desc_for(&item);
        assert_eq!(d.type_oid, 25); // TEXT
    }
}
