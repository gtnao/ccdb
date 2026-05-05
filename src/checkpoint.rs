//! Persists checkpoint metadata: the most recent checkpoint LSN and the
//! global txn_id frontier at that moment. Recovery uses both to skip the
//! WAL prefix it doesn't need to replay AND to continue allocating
//! txn_ids without colliding with anything already on disk.
//!
//! `checkpoint.meta` layout (16 bytes, little-endian):
//!   `[checkpoint_lsn: u64][next_txn_id: u64]`

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;

use crate::wal::Lsn;

const META_FILE: &str = "checkpoint.meta";

#[derive(Debug, Default, Clone, Copy)]
pub struct CheckpointMeta {
    pub lsn: Lsn,
    pub next_txn_id: u64,
}

pub fn read<P: AsRef<Path>>(dir: P) -> Result<Option<CheckpointMeta>> {
    let path = dir.as_ref().join(META_FILE);
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut buf = [0u8; 16];
    match file.read_exact(&mut buf) {
        Ok(()) => {
            let lsn = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let next_txn_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            Ok(Some(CheckpointMeta { lsn, next_txn_id }))
        }
        Err(_) => Ok(None),
    }
}

pub fn write<P: AsRef<Path>>(dir: P, meta: CheckpointMeta) -> Result<()> {
    let path = dir.as_ref().join(META_FILE);
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)?;
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&meta.lsn.to_le_bytes());
    buf[8..16].copy_from_slice(&meta.next_txn_id.to_le_bytes());
    file.write_all(&buf)?;
    file.sync_all()?;
    Ok(())
}

pub fn delete<P: AsRef<Path>>(dir: P) -> Result<()> {
    let path = dir.as_ref().join(META_FILE);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccdb-ckpt-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn round_trip() {
        let d = temp_dir("rt");
        assert!(read(&d).unwrap().is_none());
        write(&d, CheckpointMeta { lsn: 42, next_txn_id: 100 }).unwrap();
        let m = read(&d).unwrap().unwrap();
        assert_eq!(m.lsn, 42);
        assert_eq!(m.next_txn_id, 100);
        delete(&d).unwrap();
        assert!(read(&d).unwrap().is_none());
        std::fs::remove_dir_all(&d).ok();
    }
}
