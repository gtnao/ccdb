use anyhow::{Result, bail};

use crate::wal::Lsn;

pub const PAGE_SIZE: usize = 4096;

pub type PageId = u32;
pub type SlotId = u16;
/// Record identifier: page + slot inside that page.
pub type Rid = (PageId, SlotId);

// Slotted page layout (all little-endian):
//
//   header (24 bytes):
//     [0..4)   page_id            : u32
//     [4..8)   next_page_id       : u32   // heap: next chain page; btree leaf:
//                                          // next leaf; btree internal: leftmost
//                                          // child pointer (p0).
//     [8..10)  tuple_count        : u16   // number of slot entries
//     [10..12) free_space_offset  : u16   // smallest tuple-data offset in use
//     [12..20) page_lsn           : u64
//     [20..21) page_kind          : u8    // 0=heap, 1=btree leaf, 2=btree internal
//     [21..24) reserved
//
//   slot array (grows forward from byte 24):
//     each slot is 4 bytes: u16 offset || u16 length
//
//   tuple data (grows backward from PAGE_SIZE):
//     newest tuple sits at free_space_offset

const HEADER_SIZE: usize = 24;
const SLOT_SIZE: usize = 4;
pub const NO_NEXT_PAGE: PageId = u32::MAX;

/// Distinguishes heap pages from B+Tree nodes. Stored in the page header so
/// that recovery and the buffer pool can tell which structural layout the
/// page bytes encode without consulting the catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageKind {
    Heap = 0,
    BTreeLeaf = 1,
    BTreeInternal = 2,
    /// Single-row relation holding a sequence's mutable state (last_value,
    /// log_cnt, is_called) at fixed byte offsets after the standard header.
    SequenceRel = 3,
}

impl PageKind {
    pub fn from_u8(b: u8) -> Self {
        match b {
            1 => PageKind::BTreeLeaf,
            2 => PageKind::BTreeInternal,
            3 => PageKind::SequenceRel,
            _ => PageKind::Heap,
        }
    }
}

pub struct Page {
    data: [u8; PAGE_SIZE],
}

impl Page {
    pub fn new(page_id: PageId) -> Self {
        let mut p = Page {
            data: [0u8; PAGE_SIZE],
        };
        p.set_page_id(page_id);
        p.set_next_page_id(NO_NEXT_PAGE);
        p.set_tuple_count(0);
        p.set_free_space_offset(PAGE_SIZE as u16);
        p
    }

    pub fn from_bytes(bytes: &[u8; PAGE_SIZE]) -> Self {
        Page { data: *bytes }
    }

    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    #[allow(dead_code)]
    pub fn page_id(&self) -> PageId {
        u32::from_le_bytes(self.data[0..4].try_into().unwrap())
    }

    fn set_page_id(&mut self, id: PageId) {
        self.data[0..4].copy_from_slice(&id.to_le_bytes());
    }

    pub fn next_page_id(&self) -> PageId {
        u32::from_le_bytes(self.data[4..8].try_into().unwrap())
    }

    pub fn set_next_page_id(&mut self, next: PageId) {
        self.data[4..8].copy_from_slice(&next.to_le_bytes());
    }

    pub fn tuple_count(&self) -> SlotId {
        u16::from_le_bytes(self.data[8..10].try_into().unwrap())
    }

    fn set_tuple_count(&mut self, n: SlotId) {
        self.data[8..10].copy_from_slice(&n.to_le_bytes());
    }

    pub fn free_space_offset(&self) -> u16 {
        u16::from_le_bytes(self.data[10..12].try_into().unwrap())
    }

    fn set_free_space_offset(&mut self, off: u16) {
        self.data[10..12].copy_from_slice(&off.to_le_bytes());
    }

    pub fn page_lsn(&self) -> Lsn {
        u64::from_le_bytes(self.data[12..20].try_into().unwrap())
    }

    pub fn set_page_lsn(&mut self, lsn: Lsn) {
        self.data[12..20].copy_from_slice(&lsn.to_le_bytes());
    }

    pub fn page_kind(&self) -> PageKind {
        PageKind::from_u8(self.data[20])
    }

    pub fn set_page_kind(&mut self, k: PageKind) {
        self.data[20] = k as u8;
    }

    /// VM bit. `true` means "every live tuple on this page is visible to
    /// any snapshot that could be running" — VACUUM sets it; any write
    /// (INSERT, set_tuple_xmax, restore) clears it. Phase 9 will use it
    /// to skip the heap visit during index-only scans.
    pub fn all_visible(&self) -> bool {
        self.data[22] != 0
    }

    pub fn set_all_visible(&mut self, v: bool) {
        self.data[22] = v as u8;
    }

    /// Mutable raw access to the page bytes. Used by recovery to apply a
    /// physical undo image without going through the slot interface, and by
    /// the B+Tree to write its own layout directly.
    pub fn as_bytes_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        &mut self.data
    }

    pub fn free_space(&self) -> usize {
        let slots_end = HEADER_SIZE + (self.tuple_count() as usize) * SLOT_SIZE;
        (self.free_space_offset() as usize).saturating_sub(slots_end)
    }

    fn slot_pos(slot_id: SlotId) -> usize {
        HEADER_SIZE + (slot_id as usize) * SLOT_SIZE
    }

    fn read_slot(&self, slot_id: SlotId) -> (u16, u16) {
        let p = Self::slot_pos(slot_id);
        let off = u16::from_le_bytes(self.data[p..p + 2].try_into().unwrap());
        let len = u16::from_le_bytes(self.data[p + 2..p + 4].try_into().unwrap());
        (off, len)
    }

    fn write_slot(&mut self, slot_id: SlotId, offset: u16, length: u16) {
        let p = Self::slot_pos(slot_id);
        self.data[p..p + 2].copy_from_slice(&offset.to_le_bytes());
        self.data[p + 2..p + 4].copy_from_slice(&length.to_le_bytes());
    }

    /// Returns Err("page full: ...") when the tuple does not fit.
    pub fn insert(&mut self, tuple_data: &[u8]) -> Result<SlotId> {
        let tuple_len = tuple_data.len();
        let need = tuple_len + SLOT_SIZE;
        if self.free_space() < need {
            bail!(
                "page full: need {need} bytes (tuple {tuple_len} + slot {SLOT_SIZE}), have {}",
                self.free_space()
            );
        }
        let new_off = self.free_space_offset() - tuple_len as u16;
        self.data[new_off as usize..new_off as usize + tuple_len].copy_from_slice(tuple_data);

        let slot_id = self.tuple_count();
        self.write_slot(slot_id, new_off, tuple_len as u16);
        self.set_tuple_count(slot_id + 1);
        self.set_free_space_offset(new_off);
        // A new (xmin = current txn) tuple just landed — the page is no
        // longer "every tuple visible to every snapshot."
        self.set_all_visible(false);
        Ok(slot_id)
    }

    pub fn get_tuple(&self, slot_id: SlotId) -> Option<&[u8]> {
        if slot_id >= self.tuple_count() {
            return None;
        }
        let (off, len) = self.read_slot(slot_id);
        if len == 0 {
            // tombstone — slot exists but tuple was deleted
            return None;
        }
        Some(&self.data[off as usize..(off + len) as usize])
    }

    /// Logical delete: marks the slot as a tombstone (length=0). The tuple's
    /// bytes are NOT reclaimed — vacuum/compaction is a future concern.
    pub fn delete(&mut self, slot_id: SlotId) -> Result<()> {
        if slot_id >= self.tuple_count() {
            bail!("slot {slot_id} out of range");
        }
        let (off, len) = self.read_slot(slot_id);
        if len == 0 {
            bail!("slot {slot_id} already deleted");
        }
        self.write_slot(slot_id, off, 0);
        Ok(())
    }

    /// Update the `xmax` field of an MVCC tuple in place. Used by logical
    /// DELETE / UPDATE — the row's bytes stay on the page, only the MVCC
    /// header changes so visibility checks know who deleted it.
    pub fn set_tuple_xmax(&mut self, slot_id: SlotId, xmax: u64) -> Result<()> {
        if slot_id >= self.tuple_count() {
            bail!("slot {slot_id} out of range");
        }
        let (offset, length) = self.read_slot(slot_id);
        if length == 0 {
            bail!("slot {slot_id} is tombstoned (legacy delete)");
        }
        if (length as usize) < 16 {
            bail!("slot {slot_id} too short for MVCC header");
        }
        let xmax_off = offset as usize + 8;
        self.data[xmax_off..xmax_off + 8].copy_from_slice(&xmax.to_le_bytes());
        // A delete invalidates the all-visible promise.
        self.set_all_visible(false);
        Ok(())
    }

    /// Write the HOT-chain forward pointer (`t_ctid`) into an existing
    /// MVCC tuple. Used by UPDATE when the new row replaces the old in
    /// place: IndexScan readers can follow the pointer past the deleted
    /// (xmax = my-tx) old row to its live successor without going through
    /// the index again.
    pub fn set_tuple_ctid(&mut self, slot_id: SlotId, ctid: (u32, u16)) -> Result<()> {
        if slot_id >= self.tuple_count() {
            bail!("slot {slot_id} out of range");
        }
        let (offset, length) = self.read_slot(slot_id);
        if length == 0 {
            bail!("slot {slot_id} tombstoned");
        }
        if (length as usize) < crate::tuple::MVCC_HEADER_SIZE {
            bail!("slot {slot_id} too short for MVCC header");
        }
        let p_off = offset as usize + crate::tuple::TCTID_PAGE_OFF;
        let s_off = offset as usize + crate::tuple::TCTID_SLOT_OFF;
        self.data[p_off..p_off + 4].copy_from_slice(&ctid.0.to_le_bytes());
        self.data[s_off..s_off + 2].copy_from_slice(&ctid.1.to_le_bytes());
        Ok(())
    }

    /// Compact the data area in place. Tombstoned slots (length=0) keep
    /// their slot id but free up their byte range; live slots get repacked
    /// against the back of the page. Slot ids are preserved so external
    /// references (RIDs, index entries) stay valid.
    ///
    /// Used by VACUUM after dead tuples are tombstoned. Idempotent.
    pub fn vacuum_compact(&mut self) {
        let n = self.tuple_count();
        // Snapshot every live tuple's bytes in slot order.
        let mut live: Vec<(SlotId, Vec<u8>)> = Vec::new();
        for slot in 0..n {
            let (off, len) = self.read_slot(slot);
            if len == 0 {
                continue;
            }
            let bytes = self.data[off as usize..(off + len) as usize].to_vec();
            live.push((slot, bytes));
        }
        // Reset the data area to "all free" and rewrite each live tuple
        // from the back, just like fresh inserts. Tombstoned slots keep
        // their (offset=0, length=0) entries.
        let mut free_off = PAGE_SIZE as u16;
        for slot in 0..n {
            self.write_slot(slot, 0, 0);
        }
        for (slot, bytes) in &live {
            let len = bytes.len() as u16;
            free_off -= len;
            self.data[free_off as usize..(free_off + len) as usize].copy_from_slice(bytes);
            self.write_slot(*slot, free_off, len);
        }
        self.set_free_space_offset(free_off);
    }

    /// True iff the page holds no live tuples — every slot is either
    /// tombstoned (length=0) or there are no slots at all. Used by VACUUM
    /// to decide whether to unlink the page from its heap chain.
    pub fn is_empty(&self) -> bool {
        let n = self.tuple_count();
        for slot in 0..n {
            let (_, len) = self.read_slot(slot);
            if len != 0 {
                return false;
            }
        }
        true
    }

    /// Reverse of `delete`: revives a tombstoned slot by writing the saved
    /// bytes back at the original offset and restoring the slot length.
    /// Relies on the invariant that `delete` does not reclaim space, so the
    /// region `[offset, offset + data.len())` is still untouched.
    pub fn restore(&mut self, slot_id: SlotId, data: &[u8]) -> Result<()> {
        if slot_id >= self.tuple_count() {
            bail!("slot {slot_id} out of range");
        }
        let (offset, len) = self.read_slot(slot_id);
        if len != 0 {
            bail!("slot {slot_id} is not deleted (length={len})");
        }
        let n = data.len();
        let end = offset as usize + n;
        if end > PAGE_SIZE {
            bail!("restore would overflow page");
        }
        self.data[offset as usize..end].copy_from_slice(data);
        self.write_slot(slot_id, offset, n as u16);
        // restore is the redo path for an Insert; treat it as a write.
        self.set_all_visible(false);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_page_has_full_free_space() {
        let p = Page::new(7);
        assert_eq!(p.page_id(), 7);
        assert_eq!(p.tuple_count(), 0);
        assert_eq!(p.free_space_offset() as usize, PAGE_SIZE);
        assert_eq!(p.free_space(), PAGE_SIZE - HEADER_SIZE);
    }

    #[test]
    fn round_trip_via_bytes() {
        let mut p = Page::new(3);
        let s = p.insert(b"hello").unwrap();
        assert_eq!(s, 0);
        let bytes = *p.as_bytes();
        let q = Page::from_bytes(&bytes);
        assert_eq!(q.page_id(), 3);
        assert_eq!(q.tuple_count(), 1);
        assert_eq!(q.get_tuple(0).unwrap(), b"hello");
    }

    #[test]
    fn restore_revives_tombstone() {
        let mut p = Page::new(0);
        let s = p.insert(b"hello").unwrap();
        p.delete(s).unwrap();
        assert!(p.get_tuple(s).is_none());
        p.restore(s, b"hello").unwrap();
        assert_eq!(p.get_tuple(s).unwrap(), b"hello");
        // Restoring a non-deleted slot is an error.
        assert!(p.restore(s, b"hello").is_err());
    }

    #[test]
    fn all_visible_clears_on_writes_and_can_be_set() {
        let mut p = Page::new(0);
        assert!(!p.all_visible(), "fresh page is not yet all_visible");
        p.set_all_visible(true);
        assert!(p.all_visible());
        // INSERT clears it.
        let s = p.insert(b"hello").unwrap();
        assert!(!p.all_visible());
        // VACUUM-like manual set works again.
        p.set_all_visible(true);
        assert!(p.all_visible());
        // delete (logical tombstone) doesn't go through set_tuple_xmax,
        // so it does NOT clear all_visible by itself — that matches the
        // spec: actual MVCC delete uses set_tuple_xmax.
        p.delete(s).unwrap();
        assert!(p.all_visible());
    }

    #[test]
    fn tombstone_hides_tuple() {
        let mut p = Page::new(0);
        let s0 = p.insert(b"hello").unwrap();
        let s1 = p.insert(b"world").unwrap();
        assert_eq!(p.get_tuple(s0).unwrap(), b"hello");
        p.delete(s0).unwrap();
        assert!(p.get_tuple(s0).is_none());
        assert_eq!(p.get_tuple(s1).unwrap(), b"world");
        // Double-delete is an error.
        assert!(p.delete(s0).is_err());
    }

    #[test]
    fn insert_until_full_then_errors() {
        let mut p = Page::new(0);
        let payload = [0xABu8; 16]; // 16 + 4 = 20 bytes per insert
        let mut count = 0;
        loop {
            match p.insert(&payload) {
                Ok(_) => count += 1,
                Err(_) => break,
            }
        }
        // Sanity: we inserted *some* tuples, all readable, and now it's full.
        assert!(count > 0);
        assert!(p.insert(&payload).is_err());
        for slot in 0..count {
            assert_eq!(p.get_tuple(slot).unwrap(), &payload);
        }
    }
}
