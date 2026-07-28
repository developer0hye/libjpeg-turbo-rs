//! Marker-boundary scanning over partially buffered JPEG streams.
//!
//! Lifted verbatim from the C-ABI shim's P4-13 suspension core
//! (`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`) so the Rust-native
//! incremental reader (P4-58 / issue #357) and the C-ABI
//! `jpeg_consume_input` state machine share ONE boundary scanner
//! instead of growing parallel mechanisms (the P4-26 co-design
//! constraint). Pure functions over `&[u8]`; byte-exactness against
//! stock libjpeg's marker walk is pinned by the capi suspension tests.

/// Where a forward scan through entropy-coded data stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerBoundary {
    /// A complete SOS header was found; payload starts at this offset.
    Sos(usize),
    /// EOI found; the offset is one past the marker.
    Eoi(usize),
    /// Ran out of buffered bytes; resume scanning from this offset once
    /// more input arrives (always ≤ the buffered length, and positioned
    /// so no marker byte is skipped on resume).
    NeedMore(usize),
}

/// Scan from the SOI for the first SOS marker. Returns `Some(offset)` where
/// `offset` is the first byte of entropy-coded data (one past the SOS header
/// segment), or `None` if no complete SOS is buffered yet — either more bytes
/// are needed, or the stream hits EOI first (a tables-only abbreviated
/// datastream, which has no SOS).
pub fn find_first_sos(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None; // not a JPEG SOI start
    }
    let n: usize = bytes.len();
    let mut p: usize = 2; // index of the next marker's 0xFF, just past the SOI
    loop {
        if p + 1 >= n {
            return None; // need the marker code byte
        }
        if bytes[p] != 0xFF {
            return None; // misaligned — not at a marker
        }
        // Collapse any run of fill 0xFF bytes; the code is the first non-FF.
        let mut code_idx: usize = p + 1;
        while code_idx < n && bytes[code_idx] == 0xFF {
            code_idx += 1;
        }
        if code_idx >= n {
            return None;
        }
        let marker: u8 = bytes[code_idx];
        match marker {
            // Standalone markers (no length): SOI, TEM, RSTn.
            0xD8 | 0x01 | 0xD0..=0xD7 => {
                p = code_idx + 1;
            }
            // EOI before any SOS → tables-only; no SOS present.
            0xD9 => return None,
            // Every other marker (incl. SOS) carries a 2-byte length.
            _ => {
                let len_at: usize = code_idx + 1;
                if len_at + 2 > n {
                    return None; // length field not yet buffered
                }
                let seg_len: usize = ((bytes[len_at] as usize) << 8) | bytes[len_at + 1] as usize;
                if seg_len < 2 {
                    return None; // malformed length
                }
                let seg_end: usize = len_at + seg_len;
                if seg_end > n {
                    return None; // segment payload not fully buffered
                }
                if marker == 0xDA {
                    return Some(seg_end); // entropy data starts here
                }
                p = seg_end;
            }
        }
    }
}

/// Scan forward from `from` (inside entropy-coded data) for the next scan
/// boundary, skipping stuffed `FF 00`, fill `FF FF`, and restart markers
/// `FF D0..D7`, and skipping any length-bearing inter-scan segments (DHT / DQT
/// / DRI / APPn that a progressive stream may interleave between scans).
pub fn scan_next_boundary(bytes: &[u8], from: usize) -> MarkerBoundary {
    let n: usize = bytes.len();
    let mut i: usize = from;
    loop {
        while i < n && bytes[i] != 0xFF {
            i += 1;
        }
        if i + 1 >= n {
            return MarkerBoundary::NeedMore(i); // trailing 0xFF, need the code
        }
        let code: u8 = bytes[i + 1];
        match code {
            0x00 => i += 2,        // stuffed FF (literal 0xFF in entropy)
            0xFF => i += 1,        // fill byte; re-examine the next byte
            0xD0..=0xD7 => i += 2, // restart marker, still entropy
            0x01 => i += 2,        // TEM — parameterless, like RSTn
            0xD9 => return MarkerBoundary::Eoi(i + 2),
            0xDA => {
                let len_at: usize = i + 2;
                if len_at + 2 > n {
                    return MarkerBoundary::NeedMore(i);
                }
                let seg_len: usize = ((bytes[len_at] as usize) << 8) | bytes[len_at + 1] as usize;
                if seg_len < 2 {
                    return MarkerBoundary::NeedMore(i);
                }
                let seg_end: usize = len_at + seg_len;
                if seg_end > n {
                    return MarkerBoundary::NeedMore(i);
                }
                return MarkerBoundary::Sos(seg_end);
            }
            _ => {
                // Inter-scan length-bearing marker (tables/DRI/APPn).
                let len_at: usize = i + 2;
                if len_at + 2 > n {
                    return MarkerBoundary::NeedMore(i);
                }
                let seg_len: usize = ((bytes[len_at] as usize) << 8) | bytes[len_at + 1] as usize;
                if seg_len < 2 {
                    return MarkerBoundary::NeedMore(i);
                }
                let seg_end: usize = len_at + seg_len;
                if seg_end > n {
                    return MarkerBoundary::NeedMore(i);
                }
                i = seg_end;
            }
        }
    }
}
