//! PostgreSQL Wire Protocol v3 — simple-Q + minimal extended protocol.
//!
//! The extended path supports Parse/Bind/Describe/Execute/Sync as a thin
//! shim: Bind substitutes parameter values into the stored query text,
//! then Execute runs the result through the same machinery as a simple
//! `Query`. Good enough for libpq-based clients (psql, sysbench) that
//! issue parameterized queries through PQexecParams or PQprepare.
//!
//! Generic over the underlying byte stream so the wire framing can be
//! exercised in unit tests via in-memory pairs (rather than spinning up
//! TCP sockets).

use std::io::{Read, Write};

use anyhow::{Result, bail};

const PROTO_V3: i32 = 196_608; // 3.0
const SSL_REQUEST: i32 = 80_877_103;

// Postgres type OIDs (subset)
const OID_BOOL: i32 = 16;
const OID_INT4: i32 = 23;
const OID_TEXT: i32 = 25;
const OID_FLOAT8: i32 = 701;

#[derive(Debug, Clone)]
pub struct ColumnDesc {
    pub name: String,
    pub type_oid: i32,
    pub type_size: i16,
}

impl ColumnDesc {
    pub fn int(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_oid: OID_INT4,
            type_size: 4,
        }
    }
    pub fn varchar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_oid: OID_TEXT,
            type_size: -1,
        }
    }
    pub fn bool(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_oid: OID_BOOL,
            type_size: 1,
        }
    }
    pub fn double(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_oid: OID_FLOAT8,
            type_size: 8,
        }
    }
}

#[derive(Debug)]
pub struct StartupMessage {
    pub params: Vec<(String, String)>,
}

#[derive(Debug)]
pub enum FrontendMessage {
    Query(String),
    /// `P` — pre-register a SQL string under `name` (empty = unnamed
    /// statement). `param_types` may be all-zero (= unspecified).
    Parse {
        name: String,
        query: String,
        param_types: Vec<i32>,
    },
    /// `B` — bind values to a parsed statement, producing a portal.
    Bind {
        portal: String,
        statement: String,
        param_formats: Vec<i16>,
        params: Vec<Option<Vec<u8>>>,
        result_formats: Vec<i16>,
    },
    /// `D` — describe a statement (kind=`S`) or portal (kind=`P`).
    Describe { kind: u8, name: String },
    /// `E` — run a portal. `max_rows = 0` means no cap.
    Execute { portal: String, max_rows: i32 },
    /// `C` — close a statement or portal.
    Close { kind: u8, name: String },
    /// `S` — terminate the extended-protocol message group; reply with
    /// ReadyForQuery.
    Sync,
    /// `H` — flush; we already flush after every send, so it's a no-op.
    Flush,
    Terminate,
    Unknown(u8),
}

pub struct Connection<S: Read + Write> {
    stream: S,
}

impl<S: Read + Write> Connection<S> {
    pub fn new(stream: S) -> Self {
        Self { stream }
    }

    // --- reads ---

    pub fn read_startup(&mut self) -> Result<StartupMessage> {
        let len = self.read_i32()? as usize;
        let version = self.read_i32()?;

        if version == SSL_REQUEST {
            // Decline SSL and re-read the actual startup.
            self.stream.write_all(b"N")?;
            self.stream.flush()?;
            return self.read_startup();
        }
        if version != PROTO_V3 {
            bail!("unsupported protocol version: {version}");
        }

        let body_len = len
            .checked_sub(8)
            .ok_or_else(|| anyhow::anyhow!("startup length {len} smaller than header"))?;
        let mut buf = vec![0u8; body_len];
        self.stream.read_exact(&mut buf)?;
        Ok(StartupMessage {
            params: parse_kv_pairs(&buf)?,
        })
    }

    pub fn read_message(&mut self) -> Result<Option<FrontendMessage>> {
        let mut t = [0u8; 1];
        let n = self.stream.read(&mut t)?;
        if n == 0 {
            return Ok(None); // peer closed
        }
        let msg_type = t[0];

        let len = self.read_i32()? as usize;
        let body_len = len
            .checked_sub(4)
            .ok_or_else(|| anyhow::anyhow!("frame length {len} smaller than header"))?;
        let mut buf = vec![0u8; body_len];
        self.stream.read_exact(&mut buf)?;

        Ok(Some(match msg_type {
            b'Q' => {
                if buf.last() != Some(&0) {
                    bail!("Query message not null-terminated");
                }
                FrontendMessage::Query(std::str::from_utf8(&buf[..buf.len() - 1])?.to_string())
            }
            b'P' => parse_parse(&buf)?,
            b'B' => parse_bind(&buf)?,
            b'D' => parse_describe(&buf)?,
            b'E' => parse_execute(&buf)?,
            b'C' => parse_close(&buf)?,
            b'S' => FrontendMessage::Sync,
            b'H' => FrontendMessage::Flush,
            b'X' => FrontendMessage::Terminate,
            other => FrontendMessage::Unknown(other),
        }))
    }

    // --- writes ---

    pub fn send_auth_ok(&mut self) -> Result<()> {
        self.write_message(b'R', &0i32.to_be_bytes())
    }

    pub fn send_parameter_status(&mut self, name: &str, value: &str) -> Result<()> {
        let mut buf = Vec::with_capacity(name.len() + value.len() + 2);
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(value.as_bytes());
        buf.push(0);
        self.write_message(b'S', &buf)
    }

    pub fn send_backend_key_data(&mut self, pid: i32, secret: i32) -> Result<()> {
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&pid.to_be_bytes());
        buf[4..].copy_from_slice(&secret.to_be_bytes());
        self.write_message(b'K', &buf)
    }

    pub fn send_ready_for_query(&mut self) -> Result<()> {
        // 'I' = idle (no transaction)
        self.write_message(b'Z', b"I")
    }

    pub fn send_row_description(&mut self, columns: &[ColumnDesc]) -> Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(columns.len() as i16).to_be_bytes());
        for c in columns {
            buf.extend_from_slice(c.name.as_bytes());
            buf.push(0);
            buf.extend_from_slice(&0i32.to_be_bytes()); // table OID
            buf.extend_from_slice(&0i16.to_be_bytes()); // attr number
            buf.extend_from_slice(&c.type_oid.to_be_bytes());
            buf.extend_from_slice(&c.type_size.to_be_bytes());
            buf.extend_from_slice(&(-1i32).to_be_bytes()); // type modifier
            buf.extend_from_slice(&0i16.to_be_bytes()); // format code (text)
        }
        self.write_message(b'T', &buf)
    }

    pub fn send_data_row(&mut self, values: &[Option<String>]) -> Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(values.len() as i16).to_be_bytes());
        for v in values {
            match v {
                Some(s) => {
                    buf.extend_from_slice(&(s.len() as i32).to_be_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
                None => buf.extend_from_slice(&(-1i32).to_be_bytes()),
            }
        }
        self.write_message(b'D', &buf)
    }

    pub fn send_command_complete(&mut self, tag: &str) -> Result<()> {
        let mut buf = Vec::with_capacity(tag.len() + 1);
        buf.extend_from_slice(tag.as_bytes());
        buf.push(0);
        self.write_message(b'C', &buf)
    }

    pub fn send_error(&mut self, message: &str) -> Result<()> {
        let mut buf = Vec::new();
        buf.push(b'S'); // severity field
        buf.extend_from_slice(b"ERROR");
        buf.push(0);
        buf.push(b'M'); // message field
        buf.extend_from_slice(message.as_bytes());
        buf.push(0);
        buf.push(0); // fields terminator
        self.write_message(b'E', &buf)
    }

    pub fn send_empty_query(&mut self) -> Result<()> {
        self.write_message(b'I', &[])
    }

    /// `1` — Parse complete.
    pub fn send_parse_complete(&mut self) -> Result<()> {
        self.write_message(b'1', &[])
    }

    /// `2` — Bind complete.
    pub fn send_bind_complete(&mut self) -> Result<()> {
        self.write_message(b'2', &[])
    }

    /// `n` — Statement/portal returns no rows.
    pub fn send_no_data(&mut self) -> Result<()> {
        self.write_message(b'n', &[])
    }

    /// `t` — Parameter type list. We always report unspecified (0).
    pub fn send_parameter_description(&mut self, n: usize) -> Result<()> {
        let mut buf = Vec::with_capacity(2 + n * 4);
        buf.extend_from_slice(&(n as i16).to_be_bytes());
        for _ in 0..n {
            buf.extend_from_slice(&0i32.to_be_bytes());
        }
        self.write_message(b't', &buf)
    }

    /// `3` — Close complete.
    pub fn send_close_complete(&mut self) -> Result<()> {
        self.write_message(b'3', &[])
    }

    // --- low level ---

    fn write_message(&mut self, msg_type: u8, body: &[u8]) -> Result<()> {
        let len = (body.len() + 4) as i32;
        self.stream.write_all(&[msg_type])?;
        self.stream.write_all(&len.to_be_bytes())?;
        self.stream.write_all(body)?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.stream.read_exact(&mut b)?;
        Ok(i32::from_be_bytes(b))
    }
}

/// Read a NUL-terminated C string from `buf` starting at `*i`. Advances `*i`
/// past the terminator on success.
fn read_cstr(buf: &[u8], i: &mut usize) -> Result<String> {
    let end = buf[*i..]
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| anyhow::anyhow!("expected NUL-terminated string"))?;
    let s = std::str::from_utf8(&buf[*i..*i + end])?.to_string();
    *i += end + 1;
    Ok(s)
}

fn read_i16(buf: &[u8], i: &mut usize) -> Result<i16> {
    if buf.len() < *i + 2 {
        bail!("short read i16");
    }
    let v = i16::from_be_bytes(buf[*i..*i + 2].try_into().unwrap());
    *i += 2;
    Ok(v)
}

fn read_i32(buf: &[u8], i: &mut usize) -> Result<i32> {
    if buf.len() < *i + 4 {
        bail!("short read i32");
    }
    let v = i32::from_be_bytes(buf[*i..*i + 4].try_into().unwrap());
    *i += 4;
    Ok(v)
}

fn parse_parse(buf: &[u8]) -> Result<FrontendMessage> {
    let mut i = 0;
    let name = read_cstr(buf, &mut i)?;
    let query = read_cstr(buf, &mut i)?;
    let n = read_i16(buf, &mut i)? as usize;
    let mut param_types = Vec::with_capacity(n);
    for _ in 0..n {
        param_types.push(read_i32(buf, &mut i)?);
    }
    Ok(FrontendMessage::Parse {
        name,
        query,
        param_types,
    })
}

fn parse_bind(buf: &[u8]) -> Result<FrontendMessage> {
    let mut i = 0;
    let portal = read_cstr(buf, &mut i)?;
    let statement = read_cstr(buf, &mut i)?;
    let nf = read_i16(buf, &mut i)? as usize;
    let mut param_formats = Vec::with_capacity(nf);
    for _ in 0..nf {
        param_formats.push(read_i16(buf, &mut i)?);
    }
    let np = read_i16(buf, &mut i)? as usize;
    let mut params = Vec::with_capacity(np);
    for _ in 0..np {
        let len = read_i32(buf, &mut i)?;
        if len < 0 {
            params.push(None);
        } else {
            let len = len as usize;
            if buf.len() < i + len {
                bail!("Bind: short param value");
            }
            params.push(Some(buf[i..i + len].to_vec()));
            i += len;
        }
    }
    let nr = read_i16(buf, &mut i)? as usize;
    let mut result_formats = Vec::with_capacity(nr);
    for _ in 0..nr {
        result_formats.push(read_i16(buf, &mut i)?);
    }
    Ok(FrontendMessage::Bind {
        portal,
        statement,
        param_formats,
        params,
        result_formats,
    })
}

fn parse_describe(buf: &[u8]) -> Result<FrontendMessage> {
    if buf.is_empty() {
        bail!("Describe: empty body");
    }
    let kind = buf[0];
    let mut i = 1;
    let name = read_cstr(buf, &mut i)?;
    Ok(FrontendMessage::Describe { kind, name })
}

fn parse_execute(buf: &[u8]) -> Result<FrontendMessage> {
    let mut i = 0;
    let portal = read_cstr(buf, &mut i)?;
    let max_rows = read_i32(buf, &mut i)?;
    Ok(FrontendMessage::Execute { portal, max_rows })
}

fn parse_close(buf: &[u8]) -> Result<FrontendMessage> {
    if buf.is_empty() {
        bail!("Close: empty body");
    }
    let kind = buf[0];
    let mut i = 1;
    let name = read_cstr(buf, &mut i)?;
    Ok(FrontendMessage::Close { kind, name })
}

fn parse_kv_pairs(buf: &[u8]) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < buf.len() && buf[i] != 0 {
        let k_end = buf[i..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| anyhow::anyhow!("startup: unterminated key"))?;
        let key = std::str::from_utf8(&buf[i..i + k_end])?.to_string();
        i += k_end + 1;
        let v_end = buf[i..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| anyhow::anyhow!("startup: unterminated value"))?;
        let val = std::str::from_utf8(&buf[i..i + v_end])?.to_string();
        i += v_end + 1;
        out.push((key, val));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// In-memory paired stream: reads from `inbuf`, writes to `outbuf`.
    struct MemStream {
        inbuf: Cursor<Vec<u8>>,
        outbuf: Vec<u8>,
    }
    impl MemStream {
        fn new(input: Vec<u8>) -> Self {
            Self {
                inbuf: Cursor::new(input),
                outbuf: Vec::new(),
            }
        }
    }
    impl Read for MemStream {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            self.inbuf.read(b)
        }
    }
    impl Write for MemStream {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.outbuf.write(b)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn auth_ok_frame_bytes() {
        let mut conn = Connection::new(MemStream::new(Vec::new()));
        conn.send_auth_ok().unwrap();
        // 'R' | len=8 | i32(0)
        assert_eq!(conn.stream.outbuf, b"R\x00\x00\x00\x08\x00\x00\x00\x00");
    }

    #[test]
    fn ready_for_query_frame_bytes() {
        let mut conn = Connection::new(MemStream::new(Vec::new()));
        conn.send_ready_for_query().unwrap();
        // 'Z' | len=5 | 'I'
        assert_eq!(conn.stream.outbuf, b"Z\x00\x00\x00\x05I");
    }

    #[test]
    fn data_row_with_null_uses_minus_one_length() {
        let mut conn = Connection::new(MemStream::new(Vec::new()));
        conn.send_data_row(&[Some("hi".into()), None]).unwrap();
        let out = conn.stream.outbuf;
        // 'D' | len | i16(2) | i32(2) | "hi" | i32(-1)
        assert_eq!(out[0], b'D');
        let len = i32::from_be_bytes(out[1..5].try_into().unwrap()) as usize;
        assert_eq!(len, out.len() - 1); // len includes itself, not the type byte
        assert_eq!(&out[5..7], &2i16.to_be_bytes());
        assert_eq!(&out[7..11], &2i32.to_be_bytes());
        assert_eq!(&out[11..13], b"hi");
        assert_eq!(&out[13..17], &(-1i32).to_be_bytes());
    }

    #[test]
    fn read_query_message_roundtrip() {
        // 'Q' | len=11 | "SELECT 1\0"
        let frame = b"Q\x00\x00\x00\x0dSELECT 1\x00";
        let mut conn = Connection::new(MemStream::new(frame.to_vec()));
        let msg = conn.read_message().unwrap().unwrap();
        match msg {
            FrontendMessage::Query(s) => assert_eq!(s, "SELECT 1"),
            other => panic!("expected Query, got {other:?}"),
        }
    }

    #[test]
    fn read_terminate_message() {
        // 'X' | len=4
        let frame = b"X\x00\x00\x00\x04";
        let mut conn = Connection::new(MemStream::new(frame.to_vec()));
        assert!(matches!(
            conn.read_message().unwrap().unwrap(),
            FrontendMessage::Terminate
        ));
    }

    #[test]
    fn read_returns_none_on_eof() {
        let mut conn = Connection::new(MemStream::new(Vec::new()));
        assert!(conn.read_message().unwrap().is_none());
    }

    #[test]
    fn parse_kv_pairs_basic() {
        // user=alice\0database=ccdb\0\0 (final terminating 0 is the per-spec end)
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"user");
        buf.push(0);
        buf.extend_from_slice(b"alice");
        buf.push(0);
        buf.extend_from_slice(b"database");
        buf.push(0);
        buf.extend_from_slice(b"ccdb");
        buf.push(0);
        buf.push(0);
        let pairs = parse_kv_pairs(&buf).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("user".to_string(), "alice".to_string()),
                ("database".to_string(), "ccdb".to_string()),
            ]
        );
    }

    #[test]
    fn startup_decodes_v3() {
        // length(13) | version(196608) | "user\0a\0\0"
        let mut bytes = Vec::new();
        let body: Vec<u8> = b"user\0a\0\0".to_vec();
        let total_len = (body.len() + 8) as i32;
        bytes.extend_from_slice(&total_len.to_be_bytes());
        bytes.extend_from_slice(&PROTO_V3.to_be_bytes());
        bytes.extend_from_slice(&body);
        let mut conn = Connection::new(MemStream::new(bytes));
        let s = conn.read_startup().unwrap();
        assert_eq!(s.params, vec![("user".to_string(), "a".to_string())]);
    }
}
