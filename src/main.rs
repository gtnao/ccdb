mod buffer_pool;
mod disk;
mod page;
mod table;
mod tuple;

use anyhow::Result;

use buffer_pool::BufferPoolManager;
use disk::DiskManager;
use table::Table;
use tuple::{Column, DataType, Schema, Value};

const DATA_FILE: &str = "table.db";
const POOL_CAPACITY: usize = 3;

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
    let bpm = BufferPoolManager::new(disk, POOL_CAPACITY);
    let mut table = Table::new(bpm, schema);

    // Insert enough rows that some pages must be evicted before flush.
    // ~1KB rows × 20 → multiple 4KB pages, exceeding capacity=3.
    let big = "x".repeat(1024);
    for i in 1..=20 {
        table.insert(&[Value::Int(i), Value::Varchar(big.clone())])?;
    }

    println!("page_count = {}", table.page_count());

    table.flush()?;

    let rows = table.scan()?;
    println!("scanned {} rows (showing first 3):", rows.len());
    for row in rows.iter().take(3) {
        match &row[0] {
            Value::Int(v) => println!("  id={v}, name=<...>"),
            _ => println!("  {row:?}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-int-{name}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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
    fn many_inserts_under_small_pool_persist() {
        let path = temp_path("many");
        let big = "x".repeat(1024);
        let n = 30i32;

        {
            let disk = DiskManager::open(&path).unwrap();
            let bpm = BufferPoolManager::new(disk, 2); // very small pool
            let mut t = Table::new(bpm, schema_kv());
            for i in 0..n {
                t.insert(&[Value::Int(i), Value::Varchar(big.clone())])
                    .unwrap();
            }
            t.flush().unwrap();
        }

        let disk = DiskManager::open(&path).unwrap();
        let bpm = BufferPoolManager::new(disk, 2);
        let mut t = Table::new(bpm, schema_kv());
        let rows = t.scan().unwrap();
        assert_eq!(rows.len(), n as usize);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row[0], Value::Int(i as i32));
            assert_eq!(row[1], Value::Varchar(big.clone()));
        }
        std::fs::remove_file(&path).ok();
    }
}
