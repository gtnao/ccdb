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
    PG_CLASS_TABLE_ID,
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
}

#[derive(Debug, Clone)]
pub struct TableDef {
    pub table_id: usize,
    pub name: String,
    pub first_page_id: PageId,
    pub columns: Vec<ColumnDef>,
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

pub struct Catalog {
    bpm: BufferPool,
    tm: Arc<TransactionManager>,
}

impl Catalog {
    pub fn new(bpm: BufferPool, tm: Arc<TransactionManager>) -> Self {
        Self { bpm, tm }
    }

    /// Look up a table by name. Returns the full TableDef (including columns).
    pub fn find_table(&self, name: &str) -> Result<Option<(usize, TableDef)>> {
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
                return Ok(Some((
                    table_id,
                    TableDef {
                        table_id,
                        name: tname,
                        first_page_id: first_page,
                        columns,
                    },
                )));
            }
        }
        Ok(None)
    }

    pub fn table_by_id(&self, id: usize) -> Result<Option<TableDef>> {
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
                return Ok(Some(TableDef {
                    table_id: id,
                    name,
                    first_page_id: first_page,
                    columns,
                }));
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
            if table_id == PG_CLASS_TABLE_ID || table_id == PG_ATTRIBUTE_TABLE_ID {
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
            rows.push((
                ord,
                ColumnDef {
                    name: cname,
                    data_type: dt,
                    nullable,
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
        ],
    }
}
