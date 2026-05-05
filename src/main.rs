use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

use anyhow::{Result, bail};

const DATA_FILE: &str = "table.db";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataType {
    Int,
    Varchar,
}

#[derive(Debug, Clone)]
struct Column {
    #[allow(dead_code)]
    name: String,
    data_type: DataType,
}

#[derive(Debug, Clone)]
struct Schema {
    columns: Vec<Column>,
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Null,
    Int(i32),
    Varchar(String),
}

fn serialize_value(value: &Value, buf: &mut Vec<u8>) {
    match value {
        Value::Null => {}
        Value::Int(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Value::Varchar(v) => {
            let bytes = v.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
    }
}

fn deserialize_value(data: &[u8], data_type: DataType, is_null: bool) -> Result<(Value, usize)> {
    if is_null {
        return Ok((Value::Null, 0));
    }
    match data_type {
        DataType::Int => {
            if data.len() < 4 {
                bail!("not enough bytes for INT");
            }
            let v = i32::from_le_bytes(data[..4].try_into()?);
            Ok((Value::Int(v), 4))
        }
        DataType::Varchar => {
            if data.len() < 4 {
                bail!("not enough bytes for VARCHAR length prefix");
            }
            let len = u32::from_le_bytes(data[..4].try_into()?) as usize;
            if data.len() < 4 + len {
                bail!("VARCHAR payload truncated");
            }
            let v = String::from_utf8(data[4..4 + len].to_vec())?;
            Ok((Value::Varchar(v), 4 + len))
        }
    }
}

fn null_bitmap_size(num_columns: usize) -> usize {
    num_columns.div_ceil(8)
}

// Tuple layout: [null bitmap (ceil(N/8) bytes)] [non-null values, in column order]
// Bitmap convention: bit i set => column i IS NULL.
fn serialize_tuple(values: &[Value], schema: &Schema) -> Result<Vec<u8>> {
    if values.len() != schema.columns.len() {
        bail!(
            "tuple arity mismatch: got {} values, schema has {} columns",
            values.len(),
            schema.columns.len()
        );
    }

    let bitmap_len = null_bitmap_size(schema.columns.len());
    let mut buf = vec![0u8; bitmap_len];
    for (i, value) in values.iter().enumerate() {
        if matches!(value, Value::Null) {
            buf[i / 8] |= 1 << (i % 8);
        }
    }
    for value in values {
        serialize_value(value, &mut buf);
    }
    Ok(buf)
}

fn deserialize_tuple(data: &[u8], schema: &Schema) -> Result<(Vec<Value>, usize)> {
    let bitmap_len = null_bitmap_size(schema.columns.len());
    if data.len() < bitmap_len {
        bail!("tuple truncated: missing null bitmap");
    }
    let bitmap = &data[..bitmap_len];
    let mut offset = bitmap_len;

    let mut values = Vec::with_capacity(schema.columns.len());
    for (i, column) in schema.columns.iter().enumerate() {
        let is_null = bitmap[i / 8] & (1 << (i % 8)) != 0;
        let (value, len) = deserialize_value(&data[offset..], column.data_type, is_null)?;
        values.push(value);
        offset += len;
    }
    Ok((values, offset))
}

fn insert(values: &[Value], schema: &Schema) -> Result<()> {
    let bytes = serialize_tuple(values, schema)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(DATA_FILE)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn scan(schema: &Schema) -> Result<Vec<Vec<Value>>> {
    let mut file = match File::open(DATA_FILE) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut tuples = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let (values, len) = deserialize_tuple(&data[offset..], schema)?;
        tuples.push(values);
        offset += len;
    }
    Ok(tuples)
}

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

    insert(&[Value::Int(1), Value::Varchar("Alice".to_string())], &schema)?;
    insert(&[Value::Int(2), Value::Null], &schema)?;
    insert(&[Value::Null, Value::Varchar("Charlie".to_string())], &schema)?;

    let tuples = scan(&schema)?;
    println!("scanned {} tuples:", tuples.len());
    for values in tuples {
        println!("  {values:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_2col() -> Schema {
        Schema {
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
        }
    }

    #[test]
    fn round_trip_with_nulls() {
        let schema = schema_2col();
        let cases: Vec<Vec<Value>> = vec![
            vec![Value::Int(1), Value::Varchar("Alice".to_string())],
            vec![Value::Int(2), Value::Null],
            vec![Value::Null, Value::Varchar("Charlie".to_string())],
            vec![Value::Null, Value::Null],
        ];
        for original in cases {
            let bytes = serialize_tuple(&original, &schema).unwrap();
            let (decoded, consumed) = deserialize_tuple(&bytes, &schema).unwrap();
            assert_eq!(consumed, bytes.len());
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn arity_mismatch_is_rejected() {
        let schema = schema_2col();
        let err = serialize_tuple(&[Value::Int(1)], &schema);
        assert!(err.is_err());
    }

    #[test]
    fn concatenated_tuples_decode_in_order() {
        let schema = schema_2col();
        let mut buf = Vec::new();
        let inputs: Vec<Vec<Value>> = vec![
            vec![Value::Int(10), Value::Varchar("x".to_string())],
            vec![Value::Null, Value::Varchar("yy".to_string())],
            vec![Value::Int(-7), Value::Null],
        ];
        for t in &inputs {
            buf.extend(serialize_tuple(t, &schema).unwrap());
        }

        let mut decoded = Vec::new();
        let mut off = 0;
        while off < buf.len() {
            let (vals, n) = deserialize_tuple(&buf[off..], &schema).unwrap();
            decoded.push(vals);
            off += n;
        }
        assert_eq!(decoded, inputs);
    }
}
