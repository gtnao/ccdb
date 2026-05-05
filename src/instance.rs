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
use crate::recovery;
use crate::transaction::Transaction;
use crate::transaction_manager::TransactionManager;
use crate::tuple::{DataType, Value};
use crate::wal::{self, WalManager, WalRecordType};
use crate::checkpoint::{self, CheckpointMeta};
use crate::clog::{self, Clog};

const DATA_FILE: &str = "table.db";
const WAL_FILE: &str = "wal.log";
const DATA_DIR: &str = ".";
const DEFAULT_PORT: u16 = 5433;
const POOL_CAPACITY: usize = 64;

pub struct Instance {
    catalog: Arc<Catalog>,
    bpm: BufferPool,
    lock_manager: Arc<LockManager>,
    wal: Arc<WalManager>,
    tm: Arc<TransactionManager>,
}

impl Instance {
    pub fn new(init: bool) -> Result<Self> {
        if init {
            let _ = std::fs::remove_file(DATA_FILE);
            let _ = std::fs::remove_file(WAL_FILE);
            checkpoint::delete(DATA_DIR)?;
            clog::delete(DATA_DIR)?;
        }

        let wal_records = wal::read_records(WAL_FILE)?;
        let meta = checkpoint::read(DATA_DIR)?;

        let disk = DiskManager::open(DATA_FILE)?;
        let wal = Arc::new(WalManager::open(WAL_FILE)?);
        let bpm = BufferPool::new(disk, POOL_CAPACITY, Arc::clone(&wal));
        let clog = Arc::new(Clog::open(DATA_DIR)?);
        let tm = Arc::new(TransactionManager::new(Arc::clone(&clog)));

        // Seed counter from checkpoint so we don't reuse txn_ids.
        if let Some(m) = meta {
            if m.next_txn_id > 0 {
                tm.set_next_txn_id(m.next_txn_id);
            }
        }

        if init {
            // Lay down pg_class / pg_attribute on the fresh storage.
            crate::bootstrap::bootstrap(&bpm, &tm)?;
        }

        if !wal_records.is_empty() {
            let max_lsn = wal_records.iter().map(|r| r.lsn).max().unwrap_or(0);
            wal.set_next_lsn(max_lsn + 1);

            let stats =
                recovery::recover(&bpm, &wal, &wal_records, meta.map(|m| m.lsn), &tm)?;
            eprintln!(
                "recovery: checkpoint={:?} committed={} uncommitted={} redo={} undo={}",
                meta.map(|m| m.lsn),
                stats.committed_txns,
                stats.uncommitted_txns,
                stats.redo_applied,
                stats.undo_applied,
            );
            let max_id = stats.max_txn_id.max(meta.map(|m| m.next_txn_id).unwrap_or(0));
            tm.set_next_txn_id(max_id + 1);
        }

        Ok(Self {
            catalog: Arc::new(Catalog::new(bpm.clone(), Arc::clone(&tm))),
            bpm,
            lock_manager: Arc::new(LockManager::new()),
            wal,
            tm,
        })
    }

    /// Take a fuzzy checkpoint: persist CLOG, write a Checkpoint WAL
    /// record (ATT + DPT snapshot), fsync the WAL, then update
    /// checkpoint.meta atomically.
    pub fn checkpoint(&self) -> Result<()> {
        // CLOG first so any committed status referenced by post-checkpoint
        // visibility is durable.
        self.tm.clog().flush()?;

        let att = self.tm.att_snapshot();
        let dpt = self.bpm.dpt_snapshot();
        let lsn = self
            .wal
            .append(0, 0, WalRecordType::Checkpoint { att, dpt })?;
        self.wal.flush()?;
        checkpoint::write(
            DATA_DIR,
            CheckpointMeta {
                lsn,
                next_txn_id: self.tm.next_txn_id(),
            },
        )?;
        Ok(())
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
            let wal = Arc::clone(&self.wal);
            let tm = Arc::clone(&self.tm);
            // Each connection holds an Instance handle for the CHECKPOINT path.
            let instance_handle = InstanceHandle {
                bpm: self.bpm.clone(),
                wal: Arc::clone(&self.wal),
                tm: Arc::clone(&self.tm),
            };

            thread::spawn(move || {
                if let Err(e) = handle_client(conn, catalog, bpm, lock_manager, wal, tm, instance_handle) {
                    eprintln!("connection error: {e}");
                }
            });
        }
        Ok(())
    }
}

/// Subset of `Instance` reachable from a connection thread. Lets a
/// CHECKPOINT statement trigger the same routine as the public method.
#[derive(Clone)]
struct InstanceHandle {
    bpm: BufferPool,
    wal: Arc<WalManager>,
    tm: Arc<TransactionManager>,
}

impl InstanceHandle {
    fn checkpoint(&self) -> Result<()> {
        self.tm.clog().flush()?;
        let att = self.tm.att_snapshot();
        let dpt = self.bpm.dpt_snapshot();
        let lsn = self
            .wal
            .append(0, 0, WalRecordType::Checkpoint { att, dpt })?;
        self.wal.flush()?;
        checkpoint::write(
            DATA_DIR,
            CheckpointMeta {
                lsn,
                next_txn_id: self.tm.next_txn_id(),
            },
        )?;
        Ok(())
    }
}

fn handle_client(
    mut conn: Connection<TcpStream>,
    catalog: Arc<Catalog>,
    bpm: BufferPool,
    lock_manager: Arc<LockManager>,
    wal: Arc<WalManager>,
    tm: Arc<TransactionManager>,
    instance: InstanceHandle,
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

    let mut tx = Transaction::new(Arc::clone(&tm));

    // Extended-protocol scratchpad. Statements/portals are scoped to the
    // session and survive across Sync boundaries.
    let mut statements: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut portals: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    // Tracks whether the current extended-protocol message group has hit
    // an error — once set, we skip everything until Sync (per spec).
    let mut extended_error = false;

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
                    } else if let Err(e) = run_query(
                        &sql,
                        &mut conn,
                        &bpm,
                        &lock_manager,
                        &wal,
                        &tm,
                        &catalog,
                        &mut tx,
                        &instance,
                    ) {
                        eprintln!("query error: {e}");
                        conn.send_error(&e.to_string())?;
                    }
                    conn.send_ready_for_query()?;
                }
                Some(FrontendMessage::Parse {
                    name,
                    query,
                    param_types: _,
                }) => {
                    if extended_error {
                        continue;
                    }
                    statements.insert(name, query);
                    conn.send_parse_complete()?;
                }
                Some(FrontendMessage::Bind {
                    portal,
                    statement,
                    param_formats: _,
                    params,
                    result_formats: _,
                }) => {
                    if extended_error {
                        continue;
                    }
                    let template = match statements.get(&statement) {
                        Some(s) => s.clone(),
                        None => {
                            extended_error = true;
                            conn.send_error(&format!(
                                "prepared statement '{statement}' not found"
                            ))?;
                            continue;
                        }
                    };
                    let bound = substitute_params(&template, &params);
                    portals.insert(portal, bound);
                    conn.send_bind_complete()?;
                }
                Some(FrontendMessage::Describe { kind, name }) => {
                    if extended_error {
                        continue;
                    }
                    // For a statement (S) the spec also asks us to send
                    // ParameterDescription first. For a portal (P) we
                    // skip directly to row info.
                    if kind == b'S' {
                        let pcount = statements
                            .get(&name)
                            .map(|s| count_placeholders(s))
                            .unwrap_or(0);
                        conn.send_parameter_description(pcount)?;
                    }
                    let sql = match kind {
                        b'P' => portals.get(&name).cloned(),
                        _ => statements.get(&name).cloned(),
                    };
                    let sql = sql.unwrap_or_default();
                    // For Describe-statement we don't yet know parameter
                    // values; analyzer would fail on placeholders. Send
                    // NoData and let the client figure it out from
                    // RowDescription that arrives after Bind+Describe-portal.
                    if kind == b'S' || sql.is_empty() {
                        conn.send_no_data()?;
                    } else {
                        match describe_columns(&sql, &catalog) {
                            Ok(Some(cols)) => conn.send_row_description(&cols)?,
                            Ok(None) => conn.send_no_data()?,
                            Err(e) => {
                                extended_error = true;
                                conn.send_error(&e.to_string())?;
                            }
                        }
                    }
                }
                Some(FrontendMessage::Execute { portal, max_rows: _ }) => {
                    if extended_error {
                        continue;
                    }
                    let sql = match portals.get(&portal) {
                        Some(s) => s.clone(),
                        None => {
                            extended_error = true;
                            conn.send_error(&format!(
                                "portal '{portal}' not found"
                            ))?;
                            continue;
                        }
                    };
                    if let Err(e) = run_query_with_options(
                        &sql,
                        &mut conn,
                        &bpm,
                        &lock_manager,
                        &wal,
                        &tm,
                        &catalog,
                        &mut tx,
                        &instance,
                        false, // RowDescription was sent at Describe-portal time.
                    ) {
                        eprintln!("execute error: {e}");
                        extended_error = true;
                        conn.send_error(&e.to_string())?;
                    }
                }
                Some(FrontendMessage::Close { kind, name }) => {
                    if kind == b'S' {
                        statements.remove(&name);
                    } else {
                        portals.remove(&name);
                    }
                    conn.send_close_complete()?;
                }
                Some(FrontendMessage::Sync) => {
                    extended_error = false;
                    conn.send_ready_for_query()?;
                }
                Some(FrontendMessage::Flush) => {
                    // We already flush after every send.
                }
            }
        }
    })();

    if tx.is_active() {
        if let Err(e) = executor::rollback(&bpm, &wal, &mut tx) {
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
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    tx: &mut Transaction,
    instance: &InstanceHandle,
) -> Result<()> {
    run_query_with_options(sql, conn, bpm, lm, wal, tm, catalog, tx, instance, true)
}

/// `send_row_description` controls whether to emit `T` before data rows.
/// Simple-Q always wants it; extended-protocol Execute wants it suppressed
/// because the client already received it from the prior Describe.
fn run_query_with_options(
    sql: &str,
    conn: &mut Connection<TcpStream>,
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    tx: &mut Transaction,
    instance: &InstanceHandle,
    send_row_description: bool,
) -> Result<()> {
    let stmt = parse(sql)?;
    let analyzed = analyze(catalog, &stmt)?;

    match &analyzed {
        AnalyzedStatement::Select(s) => {
            let columns: Vec<ColumnDesc> = s.select_items.iter().map(column_desc_for).collect();
            let out = execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            let rows = match out {
                Output::Rows(r) => r,
                other => bail!("SELECT yielded non-Rows output: {other:?}"),
            };
            if send_row_description {
                conn.send_row_description(&columns)?;
            }
            for row in &rows {
                let vals: Vec<Option<String>> = row.values.iter().map(value_to_text).collect();
                conn.send_data_row(&vals)?;
            }
            conn.send_command_complete(&format!("SELECT {}", rows.len()))?;
        }
        AnalyzedStatement::Insert(_) => {
            let n = expect_affected(execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("INSERT 0 {n}"))?;
        }
        AnalyzedStatement::Delete(_) => {
            let n = expect_affected(execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("DELETE {n}"))?;
        }
        AnalyzedStatement::Update(_) => {
            let n = expect_affected(execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("UPDATE {n}"))?;
        }
        AnalyzedStatement::Begin => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("BEGIN")?;
        }
        AnalyzedStatement::Commit => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("COMMIT")?;
        }
        AnalyzedStatement::Rollback => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("ROLLBACK")?;
        }
        AnalyzedStatement::Checkpoint => {
            instance.checkpoint()?;
            conn.send_command_complete("CHECKPOINT")?;
        }
        AnalyzedStatement::CreateTable(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("CREATE TABLE")?;
        }
        AnalyzedStatement::CreateIndex(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("CREATE INDEX")?;
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

/// Replace `$N` placeholders in `sql` with the textual form of the
/// corresponding `params` entry. Numeric values are inlined raw; everything
/// else is wrapped in single quotes (with `'` doubled). NULL params become
/// the SQL keyword `NULL`. Skips placeholders that occur inside quoted
/// string literals.
fn substitute_params(sql: &str, params: &[Option<Vec<u8>>]) -> String {
    let chars: Vec<char> = sql.chars().collect();
    let mut out = String::with_capacity(sql.len());
    let mut i = 0;
    let mut in_string = false;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            // Inside '...' — only single-quote toggles state, with '' escape.
            out.push(c);
            if c == '\'' {
                if i + 1 < chars.len() && chars[i + 1] == '\'' {
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '\'' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == '$' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let n: usize = chars[i + 1..j].iter().collect::<String>().parse().unwrap_or(0);
            if n >= 1 && n <= params.len() {
                out.push_str(&format_param(&params[n - 1]));
            } else {
                out.push_str("NULL");
            }
            i = j;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn format_param(p: &Option<Vec<u8>>) -> String {
    match p {
        None => "NULL".to_string(),
        Some(bytes) => {
            let s = std::str::from_utf8(bytes).unwrap_or("");
            // Numeric pattern → inline raw. Everything else gets quoted.
            if !s.is_empty() && s.parse::<f64>().is_ok() {
                s.to_string()
            } else {
                let escaped = s.replace('\'', "''");
                format!("'{escaped}'")
            }
        }
    }
}

/// Count `$N` placeholders in a SQL string (skipping those inside string
/// literals). Used for ParameterDescription.
fn count_placeholders(sql: &str) -> usize {
    let chars: Vec<char> = sql.chars().collect();
    let mut max_n = 0;
    let mut i = 0;
    let mut in_string = false;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            if c == '\'' {
                if i + 1 < chars.len() && chars[i + 1] == '\'' {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '\'' {
            in_string = true;
            i += 1;
            continue;
        }
        if c == '$' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if let Ok(n) = chars[i + 1..j].iter().collect::<String>().parse::<usize>() {
                max_n = max_n.max(n);
            }
            i = j;
            continue;
        }
        i += 1;
    }
    max_n
}

/// Parse + analyze `sql` to figure out the row description for a SELECT.
/// Returns `Some(cols)` for SELECT and `None` for everything else.
fn describe_columns(sql: &str, catalog: &Catalog) -> Result<Option<Vec<ColumnDesc>>> {
    let stmt = parse(sql)?;
    let analyzed = analyze(catalog, &stmt)?;
    Ok(match analyzed {
        AnalyzedStatement::Select(s) => {
            Some(s.select_items.iter().map(column_desc_for).collect())
        }
        _ => None,
    })
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
        Some(DataType::Double) => ColumnDesc::double(&name),
        Some(DataType::Timestamp) => ColumnDesc::timestamp(&name),
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
        Value::Double(f) => Some(format_double(*f)),
        Value::Timestamp(t) => Some(format_timestamp(*t)),
        Value::Null => None,
    }
}

/// Postgres formats doubles with up to 15 significant digits and trims
/// trailing zeros. We approximate with `{:.15}` then strip — good enough
/// for psql display parity at our current precision needs.
fn format_double(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let s = format!("{f}");
    // Rust's default {} on f64 already gives a reasonable shortest form.
    s
}

/// Format a TIMESTAMP (μs since PG epoch 2000-01-01 UTC) as ISO 8601
/// `YYYY-MM-DD HH:MM:SS[.ffffff]`. Trailing zeros on the fractional part
/// are trimmed; a value with zero microseconds omits the fractional part.
fn format_timestamp(micros: i64) -> String {
    use chrono::{Duration, NaiveDate};
    let epoch = NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let dt = epoch + Duration::microseconds(micros);
    let frac = (micros.rem_euclid(1_000_000)) as u32;
    if frac == 0 {
        dt.format("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        // Use chrono's %.f which trims trailing zeros automatically.
        dt.format("%Y-%m-%d %H:%M:%S%.f").to_string()
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
