use anyhow::{Result, bail};

/// Transaction id used in MVCC tuple headers (xmin/xmax).
pub type TxnId = u64;
/// Sentinel meaning "no deletion yet" in `xmax`.
pub const INVALID_TXN_ID: TxnId = 0;
/// Bytes occupied by the per-tuple MVCC header:
///   xmin   (8)
///   xmax   (8)
///   t_ctid (6) — forward pointer for HOT update chains:
///     [u32 page_id][u16 slot]. Sentinel `(u32::MAX, u16::MAX)`
///     means "no successor; this tuple is the head/tail".
pub const MVCC_HEADER_SIZE: usize = 22;
pub const TCTID_PAGE_OFF: usize = 16;
pub const TCTID_SLOT_OFF: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int,
    Varchar,
    Bool,
    /// 64-bit IEEE 754 floating point (`DOUBLE` / `FLOAT` keyword).
    Double,
    /// `TIMESTAMP WITHOUT TIME ZONE` — i64 microseconds since
    /// 2000-01-01 UTC midnight (PostgreSQL epoch).
    Timestamp,
    /// `DATE` — i32 days since 2000-01-01 (PG epoch).
    Date,
    /// `TIME WITHOUT TIME ZONE` — i64 microseconds since 00:00:00.
    Time,
    /// `INTERVAL` — three-field calendar duration. Months and days are
    /// kept separate from microseconds because their length depends on
    /// context (a month is 28-31 days, a day is usually 24h but can be
    /// 23h or 25h around DST). PG layout: 16 bytes total.
    Interval,
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
    Double(f64),
    /// Microseconds from PostgreSQL epoch (2000-01-01 UTC midnight). Stored
    /// as i64 little-endian (8 bytes). Range: ~292000 BC to AD 294276.
    Timestamp(i64),
    /// Days from PG epoch (2000-01-01). i32 LE (4 bytes).
    Date(i32),
    /// Microseconds since 00:00:00. i64 LE (8 bytes). 0 ≤ x < 86400 * 10^6.
    Time(i64),
    /// 3-field calendar duration. Stored as 16 bytes:
    ///   `[months: i32 LE | days: i32 LE | micros: i64 LE]`
    Interval { months: i32, days: i32, micros: i64 },
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
        Value::Double(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Value::Timestamp(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Value::Date(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Value::Time(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Value::Interval { months, days, micros } => {
            buf.extend_from_slice(&months.to_le_bytes());
            buf.extend_from_slice(&days.to_le_bytes());
            buf.extend_from_slice(&micros.to_le_bytes());
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
        DataType::Bool => {
            if data.is_empty() {
                bail!("not enough bytes for BOOL");
            }
            Ok((Value::Bool(data[0] != 0), 1))
        }
        DataType::Double => {
            if data.len() < 8 {
                bail!("not enough bytes for DOUBLE");
            }
            let v = f64::from_le_bytes(data[..8].try_into()?);
            Ok((Value::Double(v), 8))
        }
        DataType::Timestamp => {
            if data.len() < 8 {
                bail!("not enough bytes for TIMESTAMP");
            }
            let v = i64::from_le_bytes(data[..8].try_into()?);
            Ok((Value::Timestamp(v), 8))
        }
        DataType::Date => {
            if data.len() < 4 {
                bail!("not enough bytes for DATE");
            }
            let v = i32::from_le_bytes(data[..4].try_into()?);
            Ok((Value::Date(v), 4))
        }
        DataType::Time => {
            if data.len() < 8 {
                bail!("not enough bytes for TIME");
            }
            let v = i64::from_le_bytes(data[..8].try_into()?);
            Ok((Value::Time(v), 8))
        }
        DataType::Interval => {
            if data.len() < 16 {
                bail!("not enough bytes for INTERVAL");
            }
            let months = i32::from_le_bytes(data[0..4].try_into()?);
            let days = i32::from_le_bytes(data[4..8].try_into()?);
            let micros = i64::from_le_bytes(data[8..16].try_into()?);
            Ok((
                Value::Interval {
                    months,
                    days,
                    micros,
                },
                16,
            ))
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

// -- MVCC tuple format -------------------------------------------------------
//
// On-disk layout: `[xmin: u64][xmax: u64][t_ctid: u32+u16][bitmap][values...]`.
// xmin is the txn that created the row; xmax is the txn that deleted it
// (0 = not deleted). t_ctid is the HOT-chain forward pointer set during
// an UPDATE that doesn't change any indexed column — the deleted version
// uses it to point at its replacement so IndexScan can follow the chain
// past stale-but-still-indexed entries. `(u32::MAX, u16::MAX)` ⇒ no
// successor.
//
// Visibility check (in `visibility.rs`) interprets xmin/xmax against a
// snapshot; `t_ctid` is consumed by the heap-walk callers when they need
// to find the live successor of an invisible-because-deleted row.

/// Sentinel "no successor" forward pointer.
pub fn no_ctid() -> (u32, u16) {
    (u32::MAX, u16::MAX)
}

pub fn is_no_ctid(c: (u32, u16)) -> bool {
    c == no_ctid()
}

pub fn serialize_tuple_mvcc(xmin: TxnId, xmax: TxnId, values: &[Value]) -> Vec<u8> {
    serialize_tuple_mvcc_with_ctid(xmin, xmax, no_ctid(), values)
}

pub fn serialize_tuple_mvcc_with_ctid(
    xmin: TxnId,
    xmax: TxnId,
    ctid: (u32, u16),
    values: &[Value],
) -> Vec<u8> {
    let payload = serialize_tuple(values);
    let mut buf = Vec::with_capacity(MVCC_HEADER_SIZE + payload.len());
    buf.extend_from_slice(&xmin.to_le_bytes());
    buf.extend_from_slice(&xmax.to_le_bytes());
    buf.extend_from_slice(&ctid.0.to_le_bytes());
    buf.extend_from_slice(&ctid.1.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

pub fn deserialize_tuple_mvcc(
    data: &[u8],
    schema: &Schema,
) -> Result<(TxnId, TxnId, Vec<Value>)> {
    if data.len() < MVCC_HEADER_SIZE {
        bail!("tuple truncated: missing MVCC header");
    }
    let xmin = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let xmax = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let values = deserialize_tuple(&data[MVCC_HEADER_SIZE..], schema)?;
    Ok((xmin, xmax, values))
}

pub fn deserialize_tuple_mvcc_full(
    data: &[u8],
    schema: &Schema,
) -> Result<(TxnId, TxnId, (u32, u16), Vec<Value>)> {
    if data.len() < MVCC_HEADER_SIZE {
        bail!("tuple truncated: missing MVCC header");
    }
    let xmin = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let xmax = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let ctid_page = u32::from_le_bytes(data[TCTID_PAGE_OFF..TCTID_PAGE_OFF + 4].try_into().unwrap());
    let ctid_slot = u16::from_le_bytes(data[TCTID_SLOT_OFF..TCTID_SLOT_OFF + 2].try_into().unwrap());
    let values = deserialize_tuple(&data[MVCC_HEADER_SIZE..], schema)?;
    Ok((xmin, xmax, (ctid_page, ctid_slot), values))
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
