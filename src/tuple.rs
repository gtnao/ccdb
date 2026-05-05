use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int,
    Varchar,
    Bool,
}

#[derive(Debug, Clone)]
pub struct Column {
    #[allow(dead_code)]
    pub name: String,
    pub data_type: DataType,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub columns: Vec<Column>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i32),
    Varchar(String),
    Bool(bool),
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
        Value::Bool(v) => buf.push(if *v { 1 } else { 0 }),
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
        DataType::Bool => {
            if data.is_empty() {
                bail!("not enough bytes for BOOL");
            }
            Ok((Value::Bool(data[0] != 0), 1))
        }
    }
}

fn null_bitmap_size(num_columns: usize) -> usize {
    num_columns.div_ceil(8)
}

// Tuple layout: [null bitmap (ceil(N/8) bytes)] [non-null values, in column order]
// Bitmap convention: bit i set => column i IS NULL.
//
// Arity validation is the caller's responsibility (the analyzer already
// enforces it for INSERT). This keeps the function dependency-free.
pub fn serialize_tuple(values: &[Value]) -> Vec<u8> {
    let bitmap_len = null_bitmap_size(values.len());
    let mut buf = vec![0u8; bitmap_len];
    for (i, value) in values.iter().enumerate() {
        if matches!(value, Value::Null) {
            buf[i / 8] |= 1 << (i % 8);
        }
    }
    for value in values {
        serialize_value(value, &mut buf);
    }
    buf
}

pub fn deserialize_tuple(data: &[u8], schema: &Schema) -> Result<Vec<Value>> {
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
    Ok(values)
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
            let bytes = serialize_tuple(&original);
            let decoded = deserialize_tuple(&bytes, &schema).unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn bool_round_trip() {
        let schema = Schema {
            columns: vec![
                Column {
                    name: "a".into(),
                    data_type: DataType::Bool,
                },
                Column {
                    name: "b".into(),
                    data_type: DataType::Bool,
                },
            ],
        };
        for vs in [
            vec![Value::Bool(true), Value::Bool(false)],
            vec![Value::Null, Value::Bool(true)],
        ] {
            let bytes = serialize_tuple(&vs);
            assert_eq!(deserialize_tuple(&bytes, &schema).unwrap(), vs);
        }
    }
}
