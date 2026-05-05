//! Trivial persistence of the most recent checkpoint LSN.
//!
//! `checkpoint.meta` is a tiny file containing one little-endian u64. It's
//! the entry point recovery uses to skip the WAL prefix it doesn't need to
//! replay. Updated atomically (truncate + write + fsync) at the end of a
//! successful checkpoint.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;

use crate::wal::Lsn;

const META_FILE: &str = "checkpoint.meta";

pub fn read_lsn<P: AsRef<Path>>(dir: P) -> Result<Option<Lsn>> {
    let path = dir.as_ref().join(META_FILE);
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut buf = [0u8; 8];
    match file.read_exact(&mut buf) {
        Ok(()) => Ok(Some(u64::from_le_bytes(buf))),
        // Treat a partial/empty file as "no checkpoint yet" rather than fatal.
        Err(_) => Ok(None),
    }
}

pub fn write_lsn<P: AsRef<Path>>(dir: P, lsn: Lsn) -> Result<()> {
    let path = dir.as_ref().join(META_FILE);
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)?;
    file.write_all(&lsn.to_le_bytes())?;
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
        assert_eq!(read_lsn(&d).unwrap(), None);
        write_lsn(&d, 42).unwrap();
        assert_eq!(read_lsn(&d).unwrap(), Some(42));
        delete(&d).unwrap();
        assert_eq!(read_lsn(&d).unwrap(), None);
        std::fs::remove_dir_all(&d).ok();
    }
}
