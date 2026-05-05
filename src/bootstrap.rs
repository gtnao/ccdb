//! Initialise the system catalog tables (pg_class / pg_attribute / pg_index)
//! on a fresh database. All three live as ordinary heap tables — pg_class
//! itself is the first row in pg_class, identifying its own page.
//!
//! Layout:
//!   - page 0 = pg_class            (table_id, name, first_page_id)
//!   - page 1 = pg_attribute        (table_id, column_name, data_type, nullable, ordinal)
//!   - page 2 = pg_index            (index_id, name, table_id, column_index, root_page_id)
//! Rows are inserted with `xmin = SYSTEM_TXN_ID` (a reserved committed txn)
//! so visibility checks accept them without further setup.

use anyhow::Result;

use crate::buffer_pool::BufferPool;
use crate::page::PageId;
use crate::transaction_manager::{TransactionManager, TxnStatus};
use crate::tuple::{serialize_tuple_mvcc, Value};

pub const SYSTEM_TXN_ID: u64 = 1;
pub const PG_CLASS_TABLE_ID: i32 = 0;
pub const PG_ATTRIBUTE_TABLE_ID: i32 = 1;
pub const PG_INDEX_TABLE_ID: i32 = 2;
pub const PG_CLASS_PAGE_ID: PageId = 0;
pub const PG_ATTRIBUTE_PAGE_ID: PageId = 1;
pub const PG_INDEX_PAGE_ID: PageId = 2;

pub const DT_INT: i32 = 0;
pub const DT_VARCHAR: i32 = 1;
pub const DT_BOOL: i32 = 2;
pub const DT_DOUBLE: i32 = 3;
pub const DT_TIMESTAMP: i32 = 4;

pub fn bootstrap(bpm: &BufferPool, tm: &TransactionManager) -> Result<()> {
    // -- pg_class at page 0 --
    {
        let g = bpm.new_page()?;
        debug_assert_eq!(g.page_id(), PG_CLASS_PAGE_ID);
        let mut p = g.write();
        for row in [
            (PG_CLASS_TABLE_ID, "pg_class", PG_CLASS_PAGE_ID as i32),
            (PG_ATTRIBUTE_TABLE_ID, "pg_attribute", PG_ATTRIBUTE_PAGE_ID as i32),
            (PG_INDEX_TABLE_ID, "pg_index", PG_INDEX_PAGE_ID as i32),
        ] {
            let bytes = serialize_tuple_mvcc(
                SYSTEM_TXN_ID,
                0,
                &[
                    Value::Int(row.0),
                    Value::Varchar(row.1.to_string()),
                    Value::Int(row.2),
                ],
            );
            p.insert(&bytes)?;
        }
    }

    // -- pg_attribute at page 1 --
    {
        let g = bpm.new_page()?;
        debug_assert_eq!(g.page_id(), PG_ATTRIBUTE_PAGE_ID);
        let mut p = g.write();
        let cols: &[(i32, &str, i32, bool, i32)] = &[
            (PG_CLASS_TABLE_ID, "table_id", DT_INT, false, 0),
            (PG_CLASS_TABLE_ID, "name", DT_VARCHAR, false, 1),
            (PG_CLASS_TABLE_ID, "first_page_id", DT_INT, false, 2),
            (PG_ATTRIBUTE_TABLE_ID, "table_id", DT_INT, false, 0),
            (PG_ATTRIBUTE_TABLE_ID, "column_name", DT_VARCHAR, false, 1),
            (PG_ATTRIBUTE_TABLE_ID, "data_type", DT_INT, false, 2),
            (PG_ATTRIBUTE_TABLE_ID, "nullable", DT_BOOL, false, 3),
            (PG_ATTRIBUTE_TABLE_ID, "ordinal_position", DT_INT, false, 4),
            (PG_INDEX_TABLE_ID, "index_id", DT_INT, false, 0),
            (PG_INDEX_TABLE_ID, "name", DT_VARCHAR, false, 1),
            (PG_INDEX_TABLE_ID, "table_id", DT_INT, false, 2),
            (PG_INDEX_TABLE_ID, "column_index", DT_INT, false, 3),
            (PG_INDEX_TABLE_ID, "root_page_id", DT_INT, false, 4),
        ];
        for (tid, cname, dt, nul, ord) in cols {
            let bytes = serialize_tuple_mvcc(
                SYSTEM_TXN_ID,
                0,
                &[
                    Value::Int(*tid),
                    Value::Varchar(cname.to_string()),
                    Value::Int(*dt),
                    Value::Bool(*nul),
                    Value::Int(*ord),
                ],
            );
            p.insert(&bytes)?;
        }
    }

    // -- pg_index at page 2 (empty until CREATE INDEX runs) --
    {
        let g = bpm.new_page()?;
        debug_assert_eq!(g.page_id(), PG_INDEX_PAGE_ID);
        let _p = g.write();
    }

    bpm.flush_all()?;
    tm.record_status(SYSTEM_TXN_ID, TxnStatus::Committed);
    tm.set_next_txn_id(SYSTEM_TXN_ID + 1);
    tm.clog().flush()?;
    Ok(())
}

pub fn datatype_from_int(dt: i32) -> Option<crate::tuple::DataType> {
    match dt {
        x if x == DT_INT => Some(crate::tuple::DataType::Int),
        x if x == DT_VARCHAR => Some(crate::tuple::DataType::Varchar),
        x if x == DT_BOOL => Some(crate::tuple::DataType::Bool),
        x if x == DT_DOUBLE => Some(crate::tuple::DataType::Double),
        x if x == DT_TIMESTAMP => Some(crate::tuple::DataType::Timestamp),
        _ => None,
    }
}
