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
/// 16384 frames × 4 KB = 64 MB. Enough to keep scale=10 pgbench
/// working set (≈40 MB) entirely in memory; was 64 frames (256 KB)
/// which forced near-every page fetch through disk I/O.
const POOL_CAPACITY: usize = 16384;

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

        // Background WAL writer: periodic fsync so commit-side flush_to
        // can hand off to a single shared syscall. 10 ms is the PG-style
        // wal_writer_delay sweet spot — short enough that commit latency
        // stays bounded, long enough that idle sweeps don't burn CPU /
        // syscall bandwidth.
        let _writer = Arc::clone(&wal).spawn_writer(std::time::Duration::from_millis(10));

        let catalog = Arc::new(Catalog::new(bpm.clone(), Arc::clone(&tm)));
        let lock_manager = Arc::new(LockManager::new());

        // Background autovacuum: 60 s is conservative — short enough
        // for HOT chain growth not to dominate, long enough that the
        // VACUUM pass doesn't fight a busy COPY / pgbench-i. Faster
        // intervals tank bulk-insert throughput by an order of magnitude.
        spawn_autovacuum(
            bpm.clone(),
            Arc::clone(&wal),
            Arc::clone(&tm),
            Arc::clone(&catalog),
            Arc::clone(&lock_manager),
            std::time::Duration::from_secs(60),
        );

        Ok(Self {
            catalog,
            bpm,
            lock_manager,
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
    //
    // Statements hold the parsed AST: one parse per Parse message, every
    // Execute on a derived portal reuses that AST. Portals carry the
    // post-Bind AST (parameters substituted in place).
    let mut statements: std::collections::HashMap<String, crate::ast::Statement> =
        std::collections::HashMap::new();
    /// Parameter count cached at Parse time so Describe-statement can
    /// emit ParameterDescription without re-parsing.
    let mut statement_param_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    /// portal name → (bound Statement, per-column wire format codes).
    /// `format_codes` is the value Bind requested:
    ///   - empty: implicit text
    ///   - len 1: same format for all columns
    ///   - len N: per-column override (0=text, 1=binary)
    let mut portals: std::collections::HashMap<String, (crate::ast::Statement, Vec<i16>)> =
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
                        // If the failure happened inside an explicit
                        // transaction, roll it back automatically. PG
                        // would put the txn into a "failed" state and
                        // require ROLLBACK; abandoning that for now lets
                        // pgbench's --max-tries retry path actually fire
                        // a clean BEGIN on the next iteration.
                        if tx.is_active() {
                            let _ = executor::rollback(&bpm, &wal, &mut tx);
                            let held = tx.take_held_locks();
                            lock_manager.unlock_all(tx.id(), &held);
                        }
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
                    match parse(&query) {
                        Ok(stmt) => {
                            let pcount = max_param_index(&stmt);
                            statement_param_counts.insert(name.clone(), pcount);
                            statements.insert(name, stmt);
                            conn.send_parse_complete()?;
                        }
                        Err(e) => {
                            extended_error = true;
                            conn.send_error(&e.to_string())?;
                        }
                    }
                }
                Some(FrontendMessage::Bind {
                    portal,
                    statement,
                    param_formats: _,
                    params,
                    result_formats,
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
                    match bind_params_into_stmt(&template, &params) {
                        Ok(bound) => {
                            portals.insert(portal, (bound, result_formats));
                            conn.send_bind_complete()?;
                        }
                        Err(e) => {
                            extended_error = true;
                            conn.send_error(&e.to_string())?;
                        }
                    }
                }
                Some(FrontendMessage::Describe { kind, name }) => {
                    if extended_error {
                        continue;
                    }
                    if kind == b'S' {
                        let pcount = statement_param_counts.get(&name).copied().unwrap_or(0);
                        conn.send_parameter_description(pcount)?;
                    }
                    if kind == b'S' {
                        // Describe-statement doesn't yet know parameter
                        // values; we'd need to bind+analyze to know the
                        // exact RowDescription. Defer to Describe-portal.
                        conn.send_no_data()?;
                    } else {
                        match portals.get(&name) {
                            None => conn.send_no_data()?,
                            Some((stmt, _fmts)) => match describe_columns_for_stmt(stmt, &catalog) {
                                Ok(Some(cols)) => conn.send_row_description(&cols)?,
                                Ok(None) => conn.send_no_data()?,
                                Err(e) => {
                                    extended_error = true;
                                    conn.send_error(&e.to_string())?;
                                }
                            },
                        }
                    }
                }
                Some(FrontendMessage::Execute { portal, max_rows: _ }) => {
                    if extended_error {
                        continue;
                    }
                    let (stmt, fmts) = match portals.get(&portal) {
                        Some((s, f)) => (s.clone(), f.clone()),
                        None => {
                            extended_error = true;
                            conn.send_error(&format!(
                                "portal '{portal}' not found"
                            ))?;
                            continue;
                        }
                    };
                    if let Err(e) = run_parsed_with_formats(
                        &stmt,
                        &fmts,
                        &mut conn,
                        &bpm,
                        &lock_manager,
                        &wal,
                        &tm,
                        &catalog,
                        &mut tx,
                        &instance,
                    ) {
                        eprintln!("execute error: {e}");
                        extended_error = true;
                        conn.send_error(&e.to_string())?;
                    }
                }
                Some(FrontendMessage::Close { kind, name }) => {
                    if kind == b'S' {
                        statements.remove(&name);
                        statement_param_counts.remove(&name);
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
                    // Per the extended-query protocol, Flush forces
                    // any queued response messages out to the client
                    // without ending the response cycle. Required so
                    // a client that sent Parse/Describe/Flush gets
                    // their replies before sending Bind.
                    conn.flush_out()?;
                }
                Some(FrontendMessage::CopyData(_))
                | Some(FrontendMessage::CopyDone)
                | Some(FrontendMessage::CopyFail(_)) => {
                    // These belong inside `run_copy_in`'s inner loop. Seeing
                    // one out here means the client is sending COPY data
                    // without an active COPY — protocol violation, drop it.
                    eprintln!("stray COPY message outside of an active COPY");
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

/// Spawn a background thread that runs `VACUUM` on every user table at
/// the given interval. Errors are logged and the loop continues —
/// autovacuum is best-effort, not a correctness requirement.
fn spawn_autovacuum(
    bpm: BufferPool,
    wal: Arc<WalManager>,
    tm: Arc<TransactionManager>,
    catalog: Arc<Catalog>,
    lock_manager: Arc<LockManager>,
    interval: std::time::Duration,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || loop {
        std::thread::sleep(interval);
        if let Err(e) =
            autovacuum_pass(&bpm, &wal, &tm, &catalog, &lock_manager)
        {
            eprintln!("autovacuum: {e}");
        }
    })
}

fn autovacuum_pass(
    bpm: &BufferPool,
    wal: &WalManager,
    tm: &Arc<TransactionManager>,
    catalog: &Catalog,
    lock_manager: &LockManager,
) -> Result<()> {
    use crate::ast::{Statement, VacuumStatement};
    if catalog.user_tables()?.is_empty() {
        return Ok(());
    }
    let mut tx = Transaction::new(Arc::clone(tm));
    let stmt = Statement::Vacuum(VacuumStatement {
        tables: Vec::new(), // empty list ⇒ every user table
        analyze: false,
    });
    let analyzed = analyze(catalog, &stmt)?;
    executor::execute(bpm, lock_manager, wal, tm, catalog, &analyzed, &mut tx)?;
    Ok(())
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
    let stmt = parse(sql)?;
    run_parsed_with_options(&stmt, conn, bpm, lm, wal, tm, catalog, tx, instance, true)
}

/// `send_row_description` controls whether to emit `T` before data rows.
/// Simple-Q always wants it; extended-protocol Execute wants it suppressed
/// because the client already received it from the prior Describe.
fn run_parsed_with_options(
    stmt: &crate::ast::Statement,
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
    run_parsed_inner(
        stmt,
        &[],
        conn,
        bpm,
        lm,
        wal,
        tm,
        catalog,
        tx,
        instance,
        send_row_description,
    )
}

/// Extended-protocol Execute path: portal-specified format codes
/// (text=0 / binary=1, optionally per-column). RowDescription is
/// suppressed because Bind+Describe-portal already issued it.
fn run_parsed_with_formats(
    stmt: &crate::ast::Statement,
    formats: &[i16],
    conn: &mut Connection<TcpStream>,
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    tx: &mut Transaction,
    instance: &InstanceHandle,
) -> Result<()> {
    run_parsed_inner(
        stmt, formats, conn, bpm, lm, wal, tm, catalog, tx, instance, false,
    )
}

fn run_parsed_inner(
    stmt: &crate::ast::Statement,
    formats: &[i16],
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
    let analyzed = analyze(catalog, stmt)?;

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
                let cols: Vec<Option<Vec<u8>>> = row
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| value_to_wire(v, format_for_col(formats, i)))
                    .collect();
                conn.send_data_row_bytes(&cols)?;
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
        AnalyzedStatement::CreateIndex(_) | AnalyzedStatement::AlterTableAddIndex(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("CREATE INDEX")?;
        }
        AnalyzedStatement::DropTable(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("DROP TABLE")?;
        }
        AnalyzedStatement::DropIndex(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("DROP INDEX")?;
        }
        AnalyzedStatement::Truncate(_) => {
            let n = expect_affected(execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?)?;
            conn.send_command_complete(&format!("TRUNCATE TABLE {n}"))?;
        }
        AnalyzedStatement::CreateSequence(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("CREATE SEQUENCE")?;
        }
        AnalyzedStatement::DropSequence(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("DROP SEQUENCE")?;
        }
        AnalyzedStatement::Vacuum(_) => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("VACUUM")?;
        }
        AnalyzedStatement::AnalyzeNoop => {
            execute(bpm, lm, wal, tm, catalog, &analyzed, tx)?;
            conn.send_command_complete("ANALYZE")?;
        }
        AnalyzedStatement::Copy(s) => {
            let n = run_copy_in(s, conn, bpm, lm, wal, tm, catalog, tx)?;
            conn.send_command_complete(&format!("COPY {n}"))?;
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

/// Drive a `COPY t FROM STDIN` over the wire. Sends CopyInResponse, reads
/// CopyData/CopyDone (or CopyFail), inserts each line, then returns the
/// row count. Wraps everything in an auto-commit-style WAL bracket so a
/// crash mid-COPY rolls back cleanly.
fn run_copy_in(
    stmt: &crate::analyzer::AnalyzedCopyStatement,
    conn: &mut Connection<TcpStream>,
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    tm: &TransactionManager,
    catalog: &Catalog,
    tx: &mut Transaction,
) -> Result<usize> {
    conn.send_copy_in_response(stmt.field_count)?;

    let was_inactive = !tx.is_active();
    if was_inactive {
        tx.refresh_autocommit();
        let lsn = wal.append(tx.id(), tx.last_lsn(), WalRecordType::Begin)?;
        tx.set_last_lsn(lsn);
    }

    let mut leftover: Vec<u8> = Vec::new();
    let mut rows_inserted: usize = 0;
    let mut tail_hint: Option<crate::page::PageId> = None;
    let result: Result<()> = (|| {
        loop {
            match conn.read_message()? {
                Some(FrontendMessage::CopyData(bytes)) => {
                    leftover.extend_from_slice(&bytes);
                    while let Some(nl) = leftover.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = leftover.drain(..=nl).take(nl).collect();
                        rows_inserted += copy_apply_line(
                            &line, stmt, bpm, lm, wal, catalog, tx, &mut tail_hint,
                        )?;
                    }
                }
                Some(FrontendMessage::CopyDone) => {
                    if !leftover.is_empty() {
                        rows_inserted += copy_apply_line(
                            &leftover, stmt, bpm, lm, wal, catalog, tx, &mut tail_hint,
                        )?;
                        leftover.clear();
                    }
                    break;
                }
                Some(FrontendMessage::CopyFail(reason)) => {
                    bail!("client cancelled COPY: {reason}");
                }
                Some(FrontendMessage::Terminate) | None => {
                    bail!("connection closed during COPY");
                }
                Some(other) => bail!("unexpected message during COPY: {other:?}"),
            }
        }
        Ok(())
    })();

    if was_inactive {
        let bracket = if result.is_ok() {
            WalRecordType::Commit
        } else {
            WalRecordType::Abort
        };
        let lsn = wal.append(tx.id(), tx.last_lsn(), bracket)?;
        tx.set_last_lsn(lsn);
        wal.flush()?;
        let held = tx.take_held_locks();
        lm.unlock_all(tx.id(), &held);
        if result.is_ok() {
            tm.commit(tx.id());
        } else {
            tm.abort(tx.id());
        }
    }

    result.map(|_| rows_inserted)
}

/// Decode one CopyData text-format line, build the row's per-table-column
/// `Value` vector (NULL-padding columns the client didn't list), and hand
/// it to the executor. Strips a single trailing CR if present (psql sends
/// CRLF on Windows; libpq strips it but we don't depend on that).
fn copy_apply_line(
    line: &[u8],
    stmt: &crate::analyzer::AnalyzedCopyStatement,
    bpm: &BufferPool,
    lm: &LockManager,
    wal: &WalManager,
    catalog: &Catalog,
    tx: &mut Transaction,
    tail_hint: &mut Option<crate::page::PageId>,
) -> Result<usize> {
    let line = if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    };
    // The text format end-of-data sentinel is a line that's exactly `\.`. PG's
    // libpq strips it before sending CopyDone, but tolerate it just in case.
    if line == b"\\." {
        return Ok(0);
    }
    let raw_fields = split_copy_fields(line);
    if raw_fields.len() != stmt.field_count {
        bail!(
            "COPY line has {} fields, expected {}",
            raw_fields.len(),
            stmt.field_count,
        );
    }
    // Decode each field once into a Value, then reorder by table-column order.
    let decoded: Vec<Value> = raw_fields
        .iter()
        .enumerate()
        .map(|(i, raw)| {
            // The data type for this *field* depends on which table column
            // it maps to — find the column whose column_to_field == Some(i).
            let col_idx = stmt
                .column_to_field
                .iter()
                .position(|m| *m == Some(i))
                .ok_or_else(|| anyhow::anyhow!("internal: COPY field {i} maps to no column"))?;
            decode_copy_field(raw, stmt.column_types[col_idx], stmt.column_nullable[col_idx])
        })
        .collect::<Result<_>>()?;
    let row: Vec<Value> = stmt
        .column_to_field
        .iter()
        .map(|m| match m {
            Some(i) => decoded[*i].clone(),
            None => Value::Null,
        })
        .collect();
    let (_rid, new_tail) = crate::executor::perform_copy_row(
        bpm, lm, wal, catalog, stmt.table_id, row, tx, *tail_hint,
    )?;
    *tail_hint = Some(new_tail);
    Ok(1)
}

/// Split a COPY text-format line on raw (un-escaped) tab bytes. Backslash
/// escapes don't change tab semantics in PG's COPY: only an unescaped tab
/// separates fields. Empty fields (back-to-back tabs) are preserved.
fn split_copy_fields(line: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    let mut i = 0;
    while i < line.len() {
        let b = line[i];
        if b == b'\\' && i + 1 < line.len() {
            // Pass the escape through; field decoding handles \\ etc.
            cur.push(b);
            cur.push(line[i + 1]);
            i += 2;
            continue;
        }
        if b == b'\t' {
            out.push(std::mem::take(&mut cur));
            i += 1;
            continue;
        }
        cur.push(b);
        i += 1;
    }
    out.push(cur);
    out
}

/// Decode one COPY text-format field. `\N` ⇒ NULL; otherwise apply backslash
/// unescaping (`\\`, `\t`, `\n`, `\r` and a few others) then parse against
/// `dt`.
fn decode_copy_field(raw: &[u8], dt: DataType, nullable: bool) -> Result<Value> {
    if raw == b"\\N" {
        if !nullable {
            bail!("NULL value in non-nullable COPY column");
        }
        return Ok(Value::Null);
    }
    let mut s = String::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let b = raw[i];
        if b == b'\\' && i + 1 < raw.len() {
            let next = raw[i + 1];
            let mapped = match next {
                b'b' => 0x08,
                b'f' => 0x0C,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'v' => 0x0B,
                b'\\' => b'\\',
                _ => {
                    s.push(b as char);
                    i += 1;
                    continue;
                }
            };
            s.push(mapped as char);
            i += 2;
            continue;
        }
        s.push(b as char);
        i += 1;
    }
    match dt {
        DataType::Int => s
            .parse::<i64>()
            .map(|n| Value::Int(n as i32))
            .map_err(|e| anyhow::anyhow!("COPY: invalid int '{s}': {e}")),
        DataType::Double => s
            .parse::<f64>()
            .map(Value::Double)
            .map_err(|e| anyhow::anyhow!("COPY: invalid double '{s}': {e}")),
        DataType::Varchar => Ok(Value::Varchar(s)),
        DataType::Bool => match s.to_ascii_lowercase().as_str() {
            "t" | "true" | "1" => Ok(Value::Bool(true)),
            "f" | "false" | "0" => Ok(Value::Bool(false)),
            _ => bail!("COPY: invalid bool '{s}'"),
        },
        DataType::Timestamp | DataType::Date | DataType::Time | DataType::Interval => {
            // Defer temporal decoding until pgbench actually exercises it —
            // history.mtime is populated via INSERT, not COPY.
            bail!("COPY of {:?} not yet supported", dt);
        }
    }
}


/// Walk a parsed Statement looking for the highest `$N` Param index.
/// Used by Describe-statement to emit ParameterDescription without
/// having to walk the original SQL text.
fn max_param_index(s: &crate::ast::Statement) -> usize {
    use crate::ast::Statement;
    let mut n = 0;
    visit_stmt_exprs(s, &mut |e| {
        n = n.max(max_param_in_expr(e));
    });
    n
}

fn max_param_in_expr(e: &crate::ast::Expr) -> usize {
    use crate::ast::{Expr, FuncArgs};
    match e {
        Expr::Param(n) => *n,
        Expr::Literal(_) | Expr::Column { .. } => 0,
        Expr::BinaryOp { left, right, .. } => {
            max_param_in_expr(left).max(max_param_in_expr(right))
        }
        Expr::UnaryOp { expr, .. } | Expr::IsNull { expr, .. } => max_param_in_expr(expr),
        Expr::FuncCall { args, .. } => match args {
            FuncArgs::Star => 0,
            FuncArgs::Exprs(es) => es.iter().map(max_param_in_expr).max().unwrap_or(0),
        },
    }
}

/// Walk every Expr inside a Statement, calling `f` on each. Used both
/// by `max_param_index` (read-only) and as a template for the param
/// substitution path below (which mutates clones).
fn visit_stmt_exprs<F: FnMut(&crate::ast::Expr)>(s: &crate::ast::Statement, f: &mut F) {
    use crate::ast::{FromClause, SelectColumn, Statement};
    fn walk_from<F: FnMut(&crate::ast::Expr)>(fc: &FromClause, f: &mut F) {
        match fc {
            FromClause::Empty | FromClause::Table(_) => {}
            FromClause::Join { left, on, .. } => {
                walk_from(left, f);
                f(on);
            }
        }
    }
    match s {
        Statement::Select(sel) => {
            for c in &sel.columns {
                if let SelectColumn::Expr { expr, .. } = c {
                    f(expr);
                }
            }
            walk_from(&sel.from, f);
            if let Some(w) = &sel.where_clause {
                f(w);
            }
            for g in &sel.group_by {
                f(g);
            }
            if let Some(h) = &sel.having {
                f(h);
            }
            for o in &sel.order_by {
                f(&o.expr);
            }
        }
        Statement::Insert(ins) => {
            for row in &ins.rows {
                for e in row {
                    f(e);
                }
            }
        }
        Statement::Delete(d) => {
            if let Some(w) = &d.where_clause {
                f(w);
            }
        }
        Statement::Update(u) => {
            for a in &u.assignments {
                f(&a.value);
            }
            if let Some(w) = &u.where_clause {
                f(w);
            }
        }
        Statement::CreateTable(_)
        | Statement::CreateIndex(_)
        | Statement::DropTable(_)
        | Statement::DropIndex(_)
        | Statement::TruncateTable(_)
        | Statement::AlterTable(_)
        | Statement::CreateSequence(_)
        | Statement::DropSequence(_)
        | Statement::Vacuum(_)
        | Statement::Analyze(_)
        | Statement::Copy(_)
        | Statement::Begin
        | Statement::Commit
        | Statement::Rollback
        | Statement::Checkpoint => {}
    }
}

/// Decode one Bind parameter (text format) into an AST literal.
fn param_to_literal(p: &Option<Vec<u8>>) -> crate::ast::Literal {
    use crate::ast::Literal;
    match p {
        None => Literal::Null,
        Some(bytes) => {
            let s = std::str::from_utf8(bytes).unwrap_or("");
            if let Ok(n) = s.parse::<i64>() {
                return Literal::Integer(n);
            }
            if let Ok(f) = s.parse::<f64>() {
                return Literal::Float(f);
            }
            Literal::String(s.to_string())
        }
    }
}

/// Substitute every `Expr::Param(N)` in `e` with the bound value's literal.
fn replace_params_in_expr(
    e: &crate::ast::Expr,
    lits: &[crate::ast::Literal],
) -> Result<crate::ast::Expr> {
    use crate::ast::{Expr, FuncArgs};
    Ok(match e {
        Expr::Param(n) => {
            let lit = lits
                .get(*n - 1)
                .ok_or_else(|| anyhow::anyhow!("missing value for parameter ${n}"))?
                .clone();
            Expr::Literal(lit)
        }
        Expr::Literal(_) | Expr::Column { .. } => e.clone(),
        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left: Box::new(replace_params_in_expr(left, lits)?),
            op: *op,
            right: Box::new(replace_params_in_expr(right, lits)?),
        },
        Expr::UnaryOp { op, expr } => Expr::UnaryOp {
            op: *op,
            expr: Box::new(replace_params_in_expr(expr, lits)?),
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: Box::new(replace_params_in_expr(expr, lits)?),
            negated: *negated,
        },
        Expr::FuncCall { name, args } => Expr::FuncCall {
            name: name.clone(),
            args: match args {
                FuncArgs::Star => FuncArgs::Star,
                FuncArgs::Exprs(es) => FuncArgs::Exprs(
                    es.iter()
                        .map(|e| replace_params_in_expr(e, lits))
                        .collect::<Result<_>>()?,
                ),
            },
        },
    })
}

/// Clone a parsed Statement and substitute every `$N` Param with the
/// corresponding bound value (decoded as an AST Literal). The returned
/// Statement is plan-able by `analyze` because no Param nodes remain.
fn bind_params_into_stmt(
    s: &crate::ast::Statement,
    params: &[Option<Vec<u8>>],
) -> Result<crate::ast::Statement> {
    use crate::ast::{FromClause, SelectColumn, Statement};
    let lits: Vec<crate::ast::Literal> = params.iter().map(param_to_literal).collect();
    fn walk_from(
        fc: &FromClause,
        lits: &[crate::ast::Literal],
    ) -> Result<FromClause> {
        Ok(match fc {
            FromClause::Empty => FromClause::Empty,
            FromClause::Table(t) => FromClause::Table(t.clone()),
            FromClause::Join {
                left,
                right,
                join_type,
                on,
            } => FromClause::Join {
                left: Box::new(walk_from(left, lits)?),
                right: right.clone(),
                join_type: *join_type,
                on: replace_params_in_expr(on, lits)?,
            },
        })
    }
    Ok(match s {
        Statement::Select(sel) => {
            let mut new_sel = sel.clone();
            new_sel.columns = sel
                .columns
                .iter()
                .map(|c| match c {
                    SelectColumn::Asterisk => Ok(SelectColumn::Asterisk),
                    SelectColumn::Expr { expr, alias } => Ok(SelectColumn::Expr {
                        expr: replace_params_in_expr(expr, &lits)?,
                        alias: alias.clone(),
                    }),
                })
                .collect::<Result<_>>()?;
            new_sel.from = walk_from(&sel.from, &lits)?;
            new_sel.where_clause = sel
                .where_clause
                .as_ref()
                .map(|w| replace_params_in_expr(w, &lits))
                .transpose()?;
            new_sel.group_by = sel
                .group_by
                .iter()
                .map(|g| replace_params_in_expr(g, &lits))
                .collect::<Result<_>>()?;
            new_sel.having = sel
                .having
                .as_ref()
                .map(|h| replace_params_in_expr(h, &lits))
                .transpose()?;
            new_sel.order_by = sel
                .order_by
                .iter()
                .map(|o| {
                    Ok(crate::ast::OrderBy {
                        expr: replace_params_in_expr(&o.expr, &lits)?,
                        dir: o.dir,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Statement::Select(new_sel)
        }
        Statement::Insert(ins) => {
            let mut new_ins = ins.clone();
            new_ins.rows = ins
                .rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| replace_params_in_expr(e, &lits))
                        .collect::<Result<_>>()
                })
                .collect::<Result<_>>()?;
            Statement::Insert(new_ins)
        }
        Statement::Delete(d) => {
            let mut nd = d.clone();
            nd.where_clause = d
                .where_clause
                .as_ref()
                .map(|w| replace_params_in_expr(w, &lits))
                .transpose()?;
            Statement::Delete(nd)
        }
        Statement::Update(u) => {
            let mut nu = u.clone();
            nu.assignments = u
                .assignments
                .iter()
                .map(|a| {
                    Ok(crate::ast::Assignment {
                        column: a.column.clone(),
                        value: replace_params_in_expr(&a.value, &lits)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            nu.where_clause = u
                .where_clause
                .as_ref()
                .map(|w| replace_params_in_expr(w, &lits))
                .transpose()?;
            Statement::Update(nu)
        }
        // Non-DML statements either have no Expr at all or carry only
        // parser-time literals (DEFAULT / CHECK / FK), which we treat
        // as already-bound — no Param substitution needed.
        other => other.clone(),
    })
}

/// AST-based RowDescription helper for Describe-portal in the extended
/// protocol. The portal carries an already-bound Statement, so we just
/// analyze + project.
fn describe_columns_for_stmt(
    stmt: &crate::ast::Statement,
    catalog: &Catalog,
) -> Result<Option<Vec<ColumnDesc>>> {
    let analyzed = analyze(catalog, stmt)?;
    Ok(match analyzed {
        AnalyzedStatement::Select(s) => Some(s.select_items.iter().map(column_desc_for).collect()),
        _ => None,
    })
}

/// Replace `$N` placeholders in `sql` with the textual form of the
/// corresponding `params` entry. Numeric values are inlined raw; everything
/// else is wrapped in single quotes (with `'` doubled). NULL params become
/// the SQL keyword `NULL`. Skips placeholders that occur inside quoted
/// string literals.
#[allow(dead_code)]
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
        Some(DataType::Date) => ColumnDesc::date(&name),
        Some(DataType::Time) => ColumnDesc::time(&name),
        Some(DataType::Interval) => ColumnDesc::interval(&name),
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

/// Pick the wire format for a particular result column. Bind sends:
///   - empty Vec      ⇒ implicit text for every column
///   - len 1          ⇒ same format for every column
///   - len = #columns ⇒ per-column override
fn format_for_col(formats: &[i16], col_idx: usize) -> i16 {
    match formats.len() {
        0 => 0,
        1 => formats[0],
        _ => formats.get(col_idx).copied().unwrap_or(0),
    }
}

/// Encode one Value for the wire. `format=0` is PG-style text (same
/// strings the simple-Q path emits); `format=1` is the on-the-wire
/// binary form. NULL becomes `None` (-1 length).
fn value_to_wire(v: &Value, format: i16) -> Option<Vec<u8>> {
    if matches!(v, Value::Null) {
        return None;
    }
    if format == 1 {
        return Some(match v {
            Value::Int(n) => n.to_be_bytes().to_vec(),
            Value::Bool(b) => vec![if *b { 1 } else { 0 }],
            Value::Double(f) => f.to_be_bytes().to_vec(),
            Value::Varchar(s) => s.as_bytes().to_vec(),
            Value::Timestamp(t) => t.to_be_bytes().to_vec(),
            Value::Date(d) => d.to_be_bytes().to_vec(),
            Value::Time(t) => t.to_be_bytes().to_vec(),
            Value::Interval { months, days, micros } => {
                // PG binary layout: int64 microseconds, int32 days,
                // int32 months — big-endian.
                let mut buf = Vec::with_capacity(16);
                buf.extend_from_slice(&micros.to_be_bytes());
                buf.extend_from_slice(&days.to_be_bytes());
                buf.extend_from_slice(&months.to_be_bytes());
                buf
            }
            Value::Null => unreachable!(),
        });
    }
    value_to_text(v).map(|s| s.into_bytes())
}

fn value_to_text(v: &Value) -> Option<String> {
    match v {
        Value::Int(n) => Some(n.to_string()),
        Value::Varchar(s) => Some(s.clone()),
        Value::Bool(b) => Some(if *b { "t" } else { "f" }.to_string()),
        Value::Double(f) => Some(format_double(*f)),
        Value::Timestamp(t) => Some(format_timestamp(*t)),
        Value::Date(d) => Some(format_date(*d)),
        Value::Time(t) => Some(format_time(*t)),
        Value::Interval { months, days, micros } => Some(format_interval(*months, *days, *micros)),
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

/// Format a DATE (days since PG epoch 2000-01-01) as `YYYY-MM-DD`.
fn format_date(days: i32) -> String {
    use chrono::{Duration, NaiveDate};
    let epoch = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    let d = epoch + Duration::days(days as i64);
    d.format("%Y-%m-%d").to_string()
}

/// Format a TIME (μs since 00:00:00) as `HH:MM:SS[.f]`.
fn format_time(micros: i64) -> String {
    use chrono::{Duration, NaiveTime};
    let t = NaiveTime::MIN + Duration::microseconds(micros);
    let frac = micros.rem_euclid(1_000_000);
    if frac == 0 {
        t.format("%H:%M:%S").to_string()
    } else {
        t.format("%H:%M:%S%.f").to_string()
    }
}

/// Format an INTERVAL using PostgreSQL's default verbose form
/// (`'1 year 2 mons 3 days 04:05:06'`). Components that are zero are
/// omitted; an all-zero interval is rendered as `'00:00:00'` to mirror PG.
fn format_interval(months: i32, days: i32, micros: i64) -> String {
    let mut parts: Vec<String> = Vec::new();
    let years = months / 12;
    let mons = months % 12;
    if years != 0 {
        parts.push(format!("{years} year{}", if years.abs() == 1 { "" } else { "s" }));
    }
    if mons != 0 {
        parts.push(format!("{mons} mon{}", if mons.abs() == 1 { "" } else { "s" }));
    }
    if days != 0 {
        parts.push(format!("{days} day{}", if days.abs() == 1 { "" } else { "s" }));
    }
    if micros != 0 || parts.is_empty() {
        let total_secs = micros / 1_000_000;
        let h = total_secs / 3600;
        let m = (total_secs % 3600) / 60;
        let s = total_secs % 60;
        let frac = micros.rem_euclid(1_000_000);
        if frac == 0 {
            parts.push(format!("{h:02}:{m:02}:{s:02}"));
        } else {
            // Trim trailing zeros from the fractional part.
            let frac_str = format!("{frac:06}");
            let trimmed = frac_str.trim_end_matches('0');
            parts.push(format!("{h:02}:{m:02}:{s:02}.{trimmed}"));
        }
    }
    parts.join(" ")
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
