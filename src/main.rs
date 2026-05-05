mod disk;
mod page;
mod table;
mod tuple;

use anyhow::Result;

use disk::DiskManager;
use table::Table;
use tuple::{Column, DataType, Schema, Value};

const DATA_FILE: &str = "table.db";

fn main() -> Result<()> {
    let _ = std::fs::remove_file(DATA_FILE);

    let schema = Schema {
        columns: vec![
            Column {
                name: "id".to_string(),
                data_type: DataType::Int,
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar,
            },
        ],
    };

    let disk = DiskManager::open(DATA_FILE)?;
    let mut table = Table::new(disk, schema);

    let r1 = table.insert(&[Value::Int(1), Value::Varchar("Alice".to_string())])?;
    let r2 = table.insert(&[Value::Int(2), Value::Null])?;
    let r3 = table.insert(&[Value::Null, Value::Varchar("Charlie".to_string())])?;

    println!("inserted rids: {r1:?} {r2:?} {r3:?}");
    println!("page_count = {}", table.page_count());

    let r2_again = table.get(r2)?;
    println!("get({r2:?}) = {r2_again:?}");

    let tuples = table.scan()?;
    println!("scanned {} tuples:", tuples.len());
    for values in tuples {
        println!("  {values:?}");
    }
    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ccdb-{name}-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn schema_kv() -> Schema {
        Schema {
            columns: vec![
                Column {
                    name: "k".to_string(),
                    data_type: DataType::Int,
                },
                Column {
                    name: "v".to_string(),
                    data_type: DataType::Varchar,
                },
            ],
        }
    }

    #[test]
    fn insert_spans_multiple_pages() {
        let path = temp_path("multi-page");
        let disk = DiskManager::open(&path).unwrap();
        let mut t = Table::new(disk, schema_kv());

        // ~1KB per tuple → forces multiple 4KB pages.
        let big = "x".repeat(1024);
        let n = 20;
        let mut rids = Vec::new();
        for i in 0..n {
            rids.push(
                t.insert(&[Value::Int(i), Value::Varchar(big.clone())])
                    .unwrap(),
            );
        }
        assert!(t.page_count() > 1, "expected >1 pages, got {}", t.page_count());

        let scanned = t.scan().unwrap();
        assert_eq!(scanned.len(), n as usize);
        for (i, row) in scanned.iter().enumerate() {
            assert_eq!(row[0], Value::Int(i as i32));
            assert_eq!(row[1], Value::Varchar(big.clone()));
        }
        for (i, rid) in rids.into_iter().enumerate() {
            let row = t.get(rid).unwrap().unwrap();
            assert_eq!(row[0], Value::Int(i as i32));
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reopen_preserves_data() {
        let path = temp_path("reopen");
        {
            let disk = DiskManager::open(&path).unwrap();
            let mut t = Table::new(disk, schema_kv());
            t.insert(&[Value::Int(42), Value::Varchar("hi".to_string())])
                .unwrap();
        }
        let disk = DiskManager::open(&path).unwrap();
        let mut t = Table::new(disk, schema_kv());
        let rows = t.scan().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Int(42));
        std::fs::remove_file(&path).ok();
    }
}
