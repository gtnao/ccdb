//! Catalog backed by on-disk system tables.
//!
//! `pg_class` (table_id, name, first_page_id) and `pg_attribute`
//! (table_id, column_name, data_type, nullable, ordinal_position) are
//! laid down by `bootstrap()` at fixed pages. This module reads them
//! whenever a SQL statement needs to resolve a table or its schema.

use std::sync::Arc;

use anyhow::Result;

use crate::bootstrap::{
    datatype_from_int, PG_ATTRIBUTE_PAGE_ID, PG_ATTRIBUTE_TABLE_ID, PG_CLASS_PAGE_ID,
    PG_CLASS_TABLE_ID, PG_CONSTRAINT_PAGE_ID, PG_CONSTRAINT_TABLE_ID, PG_INDEX_PAGE_ID,
    PG_INDEX_TABLE_ID, PG_SEQUENCE_PAGE_ID, PG_SEQUENCE_TABLE_ID,
};
use crate::buffer_pool::BufferPool;
use crate::page::{PageId, NO_NEXT_PAGE};
use crate::transaction_manager::TransactionManager;
use crate::tuple::{deserialize_tuple_mvcc, Column, DataType, Schema, Value};

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    /// Serialized DEFAULT expression. Lives in pg_attribute.default_text
    /// as a Value::Varchar and is re-parsed by the analyzer on every
    /// INSERT that omits this column. Empty string ⇒ no DEFAULT.
    pub default_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TableDef {
    pub table_id: usize,
    pub name: String,
    pub first_page_id: PageId,
    pub columns: Vec<ColumnDef>,
}

#[derive(Debug, Clone)]
pub struct IndexDef {
    pub index_id: usize,
    pub name: String,
    pub table_id: usize,
    /// Position of the indexed column inside the table's column list.
    pub column_index: usize,
    pub root_page_id: PageId,
    /// Reject duplicate keys (PRIMARY KEY / UNIQUE). Plain CREATE INDEX
    /// leaves this false.
    pub is_unique: bool,
}

/// Kind of constraint stored in pg_constraint. Numeric values are the
/// on-disk encoding (`pg_constraint.contype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ConstraintKind {
    Check = 1,
    ForeignKey = 2,
    // Primary key, unique, etc. join here in later rounds.
}

impl ConstraintKind {
    pub fn from_int(n: i32) -> Option<Self> {
        match n {
            1 => Some(Self::Check),
            2 => Some(Self::ForeignKey),
            _ => None,
        }
    }
}

/// Decoded `pg_constraint.definition` for a FOREIGN KEY constraint.
/// Stored on disk as a tab-delimited string; this is the parsed form
/// the executor uses to enforce / cascade.
#[derive(Debug, Clone)]
pub struct ForeignKeyDef {
    pub child_column: String,
    pub ref_table: String,
    pub ref_column: String,
    pub on_delete: String,
    pub on_update: String,
}

impl ForeignKeyDef {
    pub fn encode(&self) -> String {
        format!(
            "FK\t{}\t{}\t{}\t{}\t{}",
            self.child_column,
            self.ref_table,
            self.ref_column,
            self.on_delete,
            self.on_update,
        )
    }

    pub fn decode(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('\t').collect();
        if parts.len() != 6 || parts[0] != "FK" {
            return None;
        }
        Some(Self {
            child_column: parts[1].to_string(),
            ref_table: parts[2].to_string(),
            ref_column: parts[3].to_string(),
            on_delete: parts[4].to_string(),
            on_update: parts[5].to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintDef {
    pub constraint_id: usize,
    pub name: String,
    pub table_id: usize,
    pub kind: ConstraintKind,
    /// `Expr::Display` form of the constraint expression. The analyzer
    /// re-parses it whenever the constraint needs to be applied (CHECK
    /// against an INSERT/UPDATE row).
    pub definition: String,
}

#[derive(Debug, Clone)]
pub struct SequenceDef {
    pub seq_id: usize,
    pub name: String,
    pub seq_page_id: PageId,
    pub increment: i64,
    pub start_value: i64,
    pub min_value: i64,
    pub max_value: i64,
}

impl TableDef {
    pub fn to_schema(&self) -> Schema {
        Schema {
            columns: self
                .columns
                .iter()
                .map(|c| Column {
                    name: c.name.clone(),
                    data_type: c.data_type,
                })
                .collect(),
        }
    }
}

/// Memoised lookups for the four hot read paths: table-by-id, table-by-name,
/// indexes-for-table, constraints-for-table. Catalog rows change only on
/// DDL; readers populate on miss, and `invalidate` clears everything after
/// any catalog write.
#[derive(Default)]
struct CatalogCache {
    tables_by_id: std::collections::HashMap<usize, TableDef>,
    name_to_id: std::collections::HashMap<String, usize>,
    indexes_by_table: std::collections::HashMap<usize, std::sync::Arc<Vec<IndexDef>>>,
    constraints_by_table:
        std::collections::HashMap<usize, std::sync::Arc<Vec<ConstraintDef>>>,
}

pub struct Catalog {
    bpm: BufferPool,
    tm: Arc<TransactionManager>,
    cache: std::sync::Mutex<CatalogCache>,
}

impl Catalog {
    pub fn new(bpm: BufferPool, tm: Arc<TransactionManager>) -> Self {
        Self {
            bpm,
            tm,
            cache: std::sync::Mutex::new(CatalogCache::default()),
        }
    }

    /// Drop every cached entry. Call after any catalog write (CREATE /
    /// DROP / ALTER, index/constraint changes). Cheap — readers rebuild
    /// on next access.
    pub fn invalidate(&self) {
        let mut c = self.cache.lock().unwrap();
        c.tables_by_id.clear();
        c.name_to_id.clear();
        c.indexes_by_table.clear();
        c.constraints_by_table.clear();
    }

    /// Look up a table by name. Returns the full TableDef (including columns).
    pub fn find_table(&self, name: &str) -> Result<Option<(usize, TableDef)>> {
        if let Some((id, def)) = {
            let c = self.cache.lock().unwrap();
            c.name_to_id
                .get(name)
                .and_then(|id| c.tables_by_id.get(id).map(|t| (*id, t.clone())))
        } {
            return Ok(Some((id, def)));
        }
        let pg_class_schema = pg_class_schema();
        for (_, _, _, values) in self.scan_chain(PG_CLASS_PAGE_ID, &pg_class_schema)? {
            let table_id = match &values[0] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let tname = match &values[1] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            let first_page = match &values[2] {
                Value::Int(n) => *n as PageId,
                _ => continue,
            };
            if tname == name {
                let columns = self.columns_for(table_id as i32)?;
                let def = TableDef {
                    table_id,
                    name: tname.clone(),
                    first_page_id: first_page,
                    columns,
                };
                let mut c = self.cache.lock().unwrap();
                c.tables_by_id.insert(table_id, def.clone());
                c.name_to_id.insert(tname, table_id);
                return Ok(Some((table_id, def)));
            }
        }
        Ok(None)
    }

    pub fn table_by_id(&self, id: usize) -> Result<Option<TableDef>> {
        if let Some(def) = self.cache.lock().unwrap().tables_by_id.get(&id).cloned() {
            return Ok(Some(def));
        }
        let pg_class_schema = pg_class_schema();
        for (_, _, _, values) in self.scan_chain(PG_CLASS_PAGE_ID, &pg_class_schema)? {
            let tid = match &values[0] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            if tid == id {
                let name = match &values[1] {
                    Value::Varchar(s) => s.clone(),
                    _ => continue,
                };
                let first_page = match &values[2] {
                    Value::Int(n) => *n as PageId,
                    _ => continue,
                };
                let columns = self.columns_for(id as i32)?;
                let def = TableDef {
                    table_id: id,
                    name: name.clone(),
                    first_page_id: first_page,
                    columns,
                };
                let mut c = self.cache.lock().unwrap();
                c.tables_by_id.insert(id, def.clone());
                c.name_to_id.insert(name, id);
                return Ok(Some(def));
            }
        }
        Ok(None)
    }

    /// All non-system tables, used by demo / debugging paths.
    #[allow(dead_code)]
    pub fn user_tables(&self) -> Result<Vec<TableDef>> {
        let pg_class_schema = pg_class_schema();
        let mut out = Vec::new();
        for (_, _, _, values) in self.scan_chain(PG_CLASS_PAGE_ID, &pg_class_schema)? {
            let table_id = match &values[0] {
                Value::Int(n) => *n,
                _ => continue,
            };
            if table_id == PG_CLASS_TABLE_ID
                || table_id == PG_ATTRIBUTE_TABLE_ID
                || table_id == PG_INDEX_TABLE_ID
                || table_id == PG_SEQUENCE_TABLE_ID
                || table_id == PG_CONSTRAINT_TABLE_ID
            {
                continue;
            }
            let name = match &values[1] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            let first_page = match &values[2] {
                Value::Int(n) => *n as PageId,
                _ => continue,
            };
            let columns = self.columns_for(table_id)?;
            out.push(TableDef {
                table_id: table_id as usize,
                name,
                first_page_id: first_page,
                columns,
            });
        }
        Ok(out)
    }

    fn columns_for(&self, table_id: i32) -> Result<Vec<ColumnDef>> {
        let schema = pg_attribute_schema();
        let mut rows: Vec<(i32, ColumnDef)> = Vec::new();
        for (_, _, _, values) in self.scan_chain(PG_ATTRIBUTE_PAGE_ID, &schema)? {
            let tid = match &values[0] {
                Value::Int(n) => *n,
                _ => continue,
            };
            if tid != table_id {
                continue;
            }
            let cname = match &values[1] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            let dt = match &values[2] {
                Value::Int(n) => datatype_from_int(*n)
                    .ok_or_else(|| anyhow::anyhow!("unknown data type: {n}"))?,
                _ => continue,
            };
            let nullable = match &values[3] {
                Value::Bool(b) => *b,
                _ => continue,
            };
            let ord = match &values[4] {
                Value::Int(n) => *n,
                _ => continue,
            };
            // default_text added in Phase 4-2a. Empty string or missing
            // value means "no DEFAULT". Older bootstrap rows predate the
            // column so a missing position is tolerated.
            let default_text = match values.get(5) {
                Some(Value::Varchar(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            };
            rows.push((
                ord,
                ColumnDef {
                    name: cname,
                    data_type: dt,
                    nullable,
                    default_text,
                },
            ));
        }
        rows.sort_by_key(|(ord, _)| *ord);
        Ok(rows.into_iter().map(|(_, c)| c).collect())
    }

    /// Walk the page chain starting at `first_page_id` and yield every
    /// MVCC tuple. Note: visibility filtering is NOT applied here —
    /// callers that need it (regular SELECT) call this and filter via
    /// `visibility::is_visible`. The catalog uses bootstrapped rows whose
    /// xmin is the SYSTEM_TXN_ID (committed in CLOG), so simple
    /// non-deleted (xmax=0) filtering is enough for catalog reads.
    fn scan_chain(
        &self,
        first_page: PageId,
        schema: &Schema,
    ) -> Result<Vec<(PageId, u16, u64, Vec<Value>)>> {
        let mut out = Vec::new();
        let mut cur = first_page;
        while cur != NO_NEXT_PAGE && cur < self.bpm.page_count() {
            let g = self.bpm.fetch_page(cur)?;
            let p = g.read();
            let next = p.next_page_id();
            for slot in 0..p.tuple_count() {
                if let Some(raw) = p.get_tuple(slot) {
                    let (xmin, xmax, vals) = deserialize_tuple_mvcc(raw, schema)?;
                    if xmax == 0 {
                        out.push((cur, slot, xmin, vals));
                    }
                }
            }
            drop(p);
            drop(g);
            cur = next;
        }
        Ok(out)
    }

    pub fn bpm(&self) -> &BufferPool {
        &self.bpm
    }

    pub fn tm(&self) -> &Arc<TransactionManager> {
        &self.tm
    }

    /// Every committed-and-not-deleted index registered in `pg_index`.
    pub fn all_indexes(&self) -> Result<Vec<IndexDef>> {
        let schema = pg_index_schema();
        let mut out = Vec::new();
        for (_, _, _, values) in self.scan_chain(PG_INDEX_PAGE_ID, &schema)? {
            let index_id = match &values[0] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let name = match &values[1] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            let table_id = match &values[2] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let column_index = match &values[3] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let root_page_id = match &values[4] {
                Value::Int(n) => *n as PageId,
                _ => continue,
            };
            // is_unique was added in Phase 4. Tolerate older rows missing
            // the column (defaults to false) so the storage upgrade is
            // backward-compatible.
            let is_unique = match values.get(5) {
                Some(Value::Bool(b)) => *b,
                _ => false,
            };
            out.push(IndexDef {
                index_id,
                name,
                table_id,
                column_index,
                root_page_id,
                is_unique,
            });
        }
        Ok(out)
    }

    /// Indexes defined on a particular table (in declaration order).
    /// Cached: a per-table `Arc<Vec<IndexDef>>` is kept across calls and
    /// invalidated by `invalidate()` after every DDL.
    pub fn indexes_for_table(&self, table_id: usize) -> Result<Vec<IndexDef>> {
        if let Some(arc) = self
            .cache
            .lock()
            .unwrap()
            .indexes_by_table
            .get(&table_id)
            .cloned()
        {
            return Ok((*arc).clone());
        }
        let list: Vec<IndexDef> = self
            .all_indexes()?
            .into_iter()
            .filter(|i| i.table_id == table_id)
            .collect();
        let arc = std::sync::Arc::new(list.clone());
        self.cache
            .lock()
            .unwrap()
            .indexes_by_table
            .insert(table_id, arc);
        Ok(list)
    }

    pub fn find_index(&self, name: &str) -> Result<Option<IndexDef>> {
        Ok(self.all_indexes()?.into_iter().find(|i| i.name == name))
    }

    pub fn all_sequences(&self) -> Result<Vec<SequenceDef>> {
        let schema = pg_sequence_schema();
        let mut out = Vec::new();
        for (_, _, _, values) in self.scan_chain(PG_SEQUENCE_PAGE_ID, &schema)? {
            let seq_id = match &values[0] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let name = match &values[1] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            let seq_page_id = match &values[2] {
                Value::Int(n) => *n as PageId,
                _ => continue,
            };
            let increment = match &values[3] {
                Value::Int(n) => *n as i64,
                _ => continue,
            };
            let start_value = match &values[4] {
                Value::Int(n) => *n as i64,
                _ => continue,
            };
            let min_value = match &values[5] {
                Value::Int(n) => *n as i64,
                _ => continue,
            };
            let max_value = match &values[6] {
                Value::Int(n) => *n as i64,
                _ => continue,
            };
            out.push(SequenceDef {
                seq_id,
                name,
                seq_page_id,
                increment,
                start_value,
                min_value,
                max_value,
            });
        }
        Ok(out)
    }

    pub fn find_sequence(&self, name: &str) -> Result<Option<SequenceDef>> {
        Ok(self.all_sequences()?.into_iter().find(|s| s.name == name))
    }

    /// Every constraint in pg_constraint, regardless of kind. Callers
    /// usually filter by kind / table_id afterwards.
    pub fn all_constraints(&self) -> Result<Vec<ConstraintDef>> {
        let schema = pg_constraint_schema();
        let mut out = Vec::new();
        for (_, _, _, values) in self.scan_chain(PG_CONSTRAINT_PAGE_ID, &schema)? {
            let constraint_id = match &values[0] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let name = match &values[1] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            let table_id = match &values[2] {
                Value::Int(n) => *n as usize,
                _ => continue,
            };
            let kind = match &values[3] {
                Value::Int(n) => match ConstraintKind::from_int(*n) {
                    Some(k) => k,
                    None => continue,
                },
                _ => continue,
            };
            let definition = match &values[4] {
                Value::Varchar(s) => s.clone(),
                _ => continue,
            };
            out.push(ConstraintDef {
                constraint_id,
                name,
                table_id,
                kind,
                definition,
            });
        }
        Ok(out)
    }

    pub fn constraints_for_table(&self, table_id: usize) -> Result<Vec<ConstraintDef>> {
        if let Some(arc) = self
            .cache
            .lock()
            .unwrap()
            .constraints_by_table
            .get(&table_id)
            .cloned()
        {
            return Ok((*arc).clone());
        }
        let list: Vec<ConstraintDef> = self
            .all_constraints()?
            .into_iter()
            .filter(|c| c.table_id == table_id)
            .collect();
        let arc = std::sync::Arc::new(list.clone());
        self.cache
            .lock()
            .unwrap()
            .constraints_by_table
            .insert(table_id, arc);
        Ok(list)
    }
}

pub fn pg_constraint_schema() -> Schema {
    Schema {
        columns: vec![
            Column {
                name: "constraint_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar,
            },
            Column {
                name: "table_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "contype".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "definition".to_string(),
                data_type: DataType::Varchar,
            },
        ],
    }
}

pub fn pg_class_schema() -> Schema {
    Schema {
        columns: vec![
            Column {
                name: "table_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar,
            },
            Column {
                name: "first_page_id".to_string(),
                data_type: DataType::Int,
            },
        ],
    }
}

pub fn pg_sequence_schema() -> Schema {
    Schema {
        columns: vec![
            Column {
                name: "seq_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar,
            },
            Column {
                name: "seq_page_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "increment".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "start_value".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "min_value".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "max_value".to_string(),
                data_type: DataType::Int,
            },
        ],
    }
}

pub fn pg_index_schema() -> Schema {
    Schema {
        columns: vec![
            Column {
                name: "index_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar,
            },
            Column {
                name: "table_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "column_index".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "root_page_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "is_unique".to_string(),
                data_type: DataType::Bool,
            },
        ],
    }
}

pub fn pg_attribute_schema() -> Schema {
    Schema {
        columns: vec![
            Column {
                name: "table_id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "column_name".to_string(),
                data_type: DataType::Varchar,
            },
            Column {
                name: "data_type".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "nullable".to_string(),
                data_type: DataType::Bool,
            },
            Column {
                name: "ordinal_position".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "default_text".to_string(),
                data_type: DataType::Varchar,
            },
        ],
    }
}
