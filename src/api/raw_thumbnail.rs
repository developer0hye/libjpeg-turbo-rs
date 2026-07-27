//! Extract embedded JPEG thumbnails from TIFF-based RAW files.
//!
//! Sony ARW, Canon CR2, Nikon NEF, and similar camera-RAW formats all use
//! TIFF 6.0 as their container.  Each IFD (Image File Directory) may record
//! a JPEG thumbnail via two tags:
//!
//!   * `JPEGInterchangeFormat`       (0x0201) — byte offset of the JPEG.
//!   * `JPEGInterchangeFormatLength` (0x0202) — length in bytes.
//!
//! This module implements `extract_embedded_jpeg()` which scans IFD0 and the
//! linked next-IFD chain for that pair and returns the contained JPEG bytes
//! on the first hit.  The scan is deliberately minimal: just enough TIFF to
//! locate the thumbnail, without full metadata parsing.  The returned bytes
//! can then be fed into [`crate::decompress`] to obtain the thumbnail image.
//!
//! # Design
//!
//! * Pure Rust, no allocation beyond the returned `Vec<u8>`.
//! * Accepts both little-endian ("II") and big-endian ("MM") TIFFs —
//!   Sony/Nikon/Panasonic/Pentax write LE, Canon CR2 also writes LE, but
//!   some scientific formats and older raws use BE.
//! * Bounds-checks every offset and count, returning
//!   `JpegError::CorruptData` on any truncated or self-inconsistent TIFF.
//! * Walks at most a small number of IFDs to avoid pathological loops in
//!   hostile input.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::{JpegError, Result};
#[allow(unused_imports)]
use alloc::vec::Vec;
#[allow(unused_imports)]
use alloc::{format, vec};

/// Maximum number of IFDs to follow in the next-IFD chain.  Real cameras
/// use at most 3 (IFD0 = metadata, IFD1 = thumbnail, IFD2 = preview); we
/// allow 8 to accommodate exotic formats while still refusing cyclic input.
const MAX_IFD_CHAIN: usize = 8;

/// TIFF tag identifiers used for locating embedded JPEG thumbnails.
const TAG_JPEG_IF_FORMAT: u16 = 0x0201;
const TAG_JPEG_IF_LENGTH: u16 = 0x0202;

/// Extract the first embedded JPEG thumbnail from a TIFF-based container.
///
/// Accepts raw file bytes of a TIFF 6.0 / RAW container (ARW, CR2, NEF,
/// DNG, ORF, RW2, etc.).  Returns the thumbnail JPEG's byte slice on the
/// first IFD that carries both `JPEGInterchangeFormat` (0x0201) and
/// `JPEGInterchangeFormatLength` (0x0202).
///
/// # Errors
///
/// Returns `JpegError::CorruptData` for truncated TIFF headers, invalid
/// magic, or offsets that exceed the file bounds.  Returns
/// `JpegError::Unsupported` when no IFD carries a JPEG-interchange pair.
pub fn extract_embedded_jpeg(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        return Err(JpegError::CorruptData(
            "TIFF header requires at least 8 bytes".into(),
        ));
    }

    let is_le: bool = match (data[0], data[1]) {
        (b'I', b'I') => true,
        (b'M', b'M') => false,
        _ => {
            return Err(JpegError::CorruptData(format!(
                "unknown byte-order marker: {:?}{:?}",
                data[0] as char, data[1] as char
            )))
        }
    };

    let magic: u16 = read_u16(data, 2, is_le);
    if magic != 42 {
        return Err(JpegError::CorruptData(format!(
            "invalid TIFF magic: expected 42, got {}",
            magic
        )));
    }

    let mut ifd_offset: u32 = read_u32(data, 4, is_le);
    for _ in 0..MAX_IFD_CHAIN {
        if ifd_offset == 0 {
            break;
        }
        let (jpeg, next): (Option<Vec<u8>>, u32) = scan_ifd(data, ifd_offset as usize, is_le)?;
        if let Some(jpeg) = jpeg {
            return Ok(jpeg);
        }
        ifd_offset = next;
    }

    Err(JpegError::Unsupported(
        "no embedded JPEG thumbnail found in TIFF container".into(),
    ))
}

/// Scan a single IFD at `offset`.  Returns `(jpeg_bytes, next_ifd_offset)`
/// — the JPEG bytes (if both JPEGInterchangeFormat and length were
/// present) and the offset of the next IFD in the chain (0 terminates).
fn scan_ifd(data: &[u8], offset: usize, is_le: bool) -> Result<(Option<Vec<u8>>, u32)> {
    if offset + 2 > data.len() {
        return Err(JpegError::CorruptData(format!(
            "IFD offset {} exceeds file length {}",
            offset,
            data.len()
        )));
    }
    let entry_count: usize = read_u16(data, offset, is_le) as usize;
    let entries_start: usize = offset + 2;
    let entries_end: usize = entries_start + entry_count * 12;
    if entries_end + 4 > data.len() {
        return Err(JpegError::CorruptData(format!(
            "IFD at {} extends past file (entries={}, needs up to {})",
            offset,
            entry_count,
            entries_end + 4
        )));
    }

    let mut jpeg_offset: Option<u32> = None;
    let mut jpeg_length: Option<u32> = None;

    for i in 0..entry_count {
        let e: usize = entries_start + i * 12;
        let tag: u16 = read_u16(data, e, is_le);
        match tag {
            TAG_JPEG_IF_FORMAT => jpeg_offset = Some(read_ifd_entry_u32(data, e, is_le)?),
            TAG_JPEG_IF_LENGTH => jpeg_length = Some(read_ifd_entry_u32(data, e, is_le)?),
            _ => {}
        }
    }

    let next_ifd: u32 = read_u32(data, entries_end, is_le);

    if let (Some(off), Some(len)) = (jpeg_offset, jpeg_length) {
        let start: usize = off as usize;
        let end: usize = start
            .checked_add(len as usize)
            .ok_or_else(|| JpegError::CorruptData("JPEG length overflow".into()))?;
        if end > data.len() {
            return Err(JpegError::CorruptData(format!(
                "embedded JPEG spans {}..{} but file is {} bytes",
                start,
                end,
                data.len()
            )));
        }
        // Minimum sanity: must begin with SOI (FF D8).
        if end - start < 2 || data[start] != 0xFF || data[start + 1] != 0xD8 {
            return Err(JpegError::CorruptData(
                "embedded JPEG does not start with SOI (FF D8)".into(),
            ));
        }
        return Ok((Some(data[start..end].to_vec()), next_ifd));
    }

    Ok((None, next_ifd))
}

/// Read a LONG or SHORT value from an IFD entry, packed in the `value`
/// slot when count = 1.  Anything else is rejected — camera thumbnails
/// always pack a single scalar here.
fn read_ifd_entry_u32(data: &[u8], entry: usize, is_le: bool) -> Result<u32> {
    let value_type: u16 = read_u16(data, entry + 2, is_le);
    let count: u32 = read_u32(data, entry + 4, is_le);
    if count != 1 {
        return Err(JpegError::CorruptData(format!(
            "expected count=1 for JPEG-interchange tag, got {}",
            count
        )));
    }
    let v: u32 = match value_type {
        3 => read_u16(data, entry + 8, is_le) as u32, // SHORT
        4 => read_u32(data, entry + 8, is_le),        // LONG
        other => {
            return Err(JpegError::CorruptData(format!(
                "unsupported TIFF type {} for JPEG-interchange tag",
                other
            )))
        }
    };
    Ok(v)
}

#[inline]
fn read_u16(data: &[u8], offset: usize, is_le: bool) -> u16 {
    let b: [u8; 2] = [data[offset], data[offset + 1]];
    if is_le {
        u16::from_le_bytes(b)
    } else {
        u16::from_be_bytes(b)
    }
}

#[inline]
fn read_u32(data: &[u8], offset: usize, is_le: bool) -> u32 {
    let b: [u8; 4] = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    if is_le {
        u32::from_le_bytes(b)
    } else {
        u32::from_be_bytes(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal TIFF wrapping the given JPEG bytes under IFD0,
    /// tagged via JPEGInterchangeFormat + JPEGInterchangeFormatLength.
    /// Returns the full TIFF blob.
    fn wrap_jpeg_in_tiff(jpeg: &[u8], is_le: bool) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        // Header
        out.extend_from_slice(if is_le { b"II" } else { b"MM" });
        push_u16(&mut out, 42, is_le);
        // IFD0 starts immediately after the 4-byte offset we're about to
        // write.  We place IFD0 at byte 8, JPEG data at byte (8 + 2 + 2*12 + 4) = 38.
        push_u32(&mut out, 8, is_le);
        // IFD0: 2 entries
        push_u16(&mut out, 2, is_le);
        // Entry 1: JPEGInterchangeFormat (LONG, count=1, value=38)
        push_u16(&mut out, TAG_JPEG_IF_FORMAT, is_le);
        push_u16(&mut out, 4, is_le); // LONG
        push_u32(&mut out, 1, is_le); // count
        push_u32(&mut out, 38, is_le); // value (offset of JPEG)
                                       // Entry 2: JPEGInterchangeFormatLength (LONG, count=1, value=len)
        push_u16(&mut out, TAG_JPEG_IF_LENGTH, is_le);
        push_u16(&mut out, 4, is_le);
        push_u32(&mut out, 1, is_le);
        push_u32(&mut out, jpeg.len() as u32, is_le);
        // Next IFD offset = 0
        push_u32(&mut out, 0, is_le);
        // Payload
        assert_eq!(out.len(), 38, "header + IFD must end at offset 38");
        out.extend_from_slice(jpeg);
        out
    }

    fn push_u16(buf: &mut Vec<u8>, val: u16, is_le: bool) {
        if is_le {
            buf.extend_from_slice(&val.to_le_bytes());
        } else {
            buf.extend_from_slice(&val.to_be_bytes());
        }
    }

    fn push_u32(buf: &mut Vec<u8>, val: u32, is_le: bool) {
        if is_le {
            buf.extend_from_slice(&val.to_le_bytes());
        } else {
            buf.extend_from_slice(&val.to_be_bytes());
        }
    }

    fn tiny_jpeg() -> Vec<u8> {
        // A minimal but syntactically valid JPEG: SOI + single comment + EOI.
        vec![0xFF, 0xD8, 0xFF, 0xFE, 0x00, 0x04, b'o', b'k', 0xFF, 0xD9]
    }

    #[test]
    fn extracts_le_tiff_with_jpeg() {
        let jpeg: Vec<u8> = tiny_jpeg();
        let tiff: Vec<u8> = wrap_jpeg_in_tiff(&jpeg, true);
        let extracted: Vec<u8> = extract_embedded_jpeg(&tiff).unwrap();
        assert_eq!(extracted, jpeg);
    }

    #[test]
    fn extracts_be_tiff_with_jpeg() {
        let jpeg: Vec<u8> = tiny_jpeg();
        let tiff: Vec<u8> = wrap_jpeg_in_tiff(&jpeg, false);
        let extracted: Vec<u8> = extract_embedded_jpeg(&tiff).unwrap();
        assert_eq!(extracted, jpeg);
    }

    #[test]
    fn rejects_unknown_byte_order() {
        let buf: Vec<u8> = vec![b'X', b'X', 0, 42, 0, 0, 0, 8];
        let err = extract_embedded_jpeg(&buf).unwrap_err();
        assert!(
            matches!(err, JpegError::CorruptData(_)),
            "expected CorruptData, got {:?}",
            err
        );
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut tiff: Vec<u8> = wrap_jpeg_in_tiff(&tiny_jpeg(), true);
        // Flip the magic number.
        tiff[2] = 99;
        tiff[3] = 0;
        let err = extract_embedded_jpeg(&tiff).unwrap_err();
        assert!(matches!(err, JpegError::CorruptData(_)));
    }

    #[test]
    fn rejects_tiff_without_thumbnail_tag() {
        // IFD0 with one unrelated tag (ImageWidth = 0x0100).
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"II");
        push_u16(&mut out, 42, true);
        push_u32(&mut out, 8, true);
        push_u16(&mut out, 1, true); // 1 entry
        push_u16(&mut out, 0x0100, true); // ImageWidth
        push_u16(&mut out, 3, true); // SHORT
        push_u32(&mut out, 1, true); // count
        push_u16(&mut out, 64, true); // value
        push_u16(&mut out, 0, true); // pad
        push_u32(&mut out, 0, true); // next IFD = 0
        let err = extract_embedded_jpeg(&out).unwrap_err();
        assert!(matches!(err, JpegError::Unsupported(_)));
    }

    #[test]
    fn rejects_truncated_jpeg_span() {
        let jpeg: Vec<u8> = tiny_jpeg();
        let mut tiff: Vec<u8> = wrap_jpeg_in_tiff(&jpeg, true);
        // Chop off the JPEG payload so the declared length exceeds the file.
        tiff.truncate(tiff.len() - 3);
        let err = extract_embedded_jpeg(&tiff).unwrap_err();
        assert!(matches!(err, JpegError::CorruptData(_)));
    }

    #[test]
    fn rejects_non_soi_payload() {
        // Wrap bytes that don't begin with FF D8.
        let bogus: Vec<u8> = vec![0x00, 0x00, 0x00, 0x00];
        let tiff: Vec<u8> = wrap_jpeg_in_tiff(&bogus, true);
        let err = extract_embedded_jpeg(&tiff).unwrap_err();
        assert!(matches!(err, JpegError::CorruptData(_)));
    }
}
