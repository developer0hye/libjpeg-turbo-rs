//! JFIF APP0 thumbnail extraction.
//!
//! Parses the JFIF APP0 marker in a JPEG byte stream to extract an embedded
//! uncompressed RGB thumbnail, if present.

/// Extract JFIF thumbnail from JPEG data if present.
///
/// Parses the first APP0 (JFIF) marker in the JPEG stream and extracts the
/// uncompressed RGB thumbnail data. Returns `None` if no JFIF marker is found,
/// or the thumbnail dimensions are 0x0.
///
/// The JFIF APP0 format after the length field is:
/// - `"JFIF\0"` (5 bytes) identifier
/// - Version major/minor (2 bytes)
/// - Density units (1 byte)
/// - X density (2 bytes, big-endian)
/// - Y density (2 bytes, big-endian)
/// - Thumbnail width (1 byte)
/// - Thumbnail height (1 byte)
/// - Thumbnail pixel data (3 * width * height bytes, RGB)
pub fn extract_jfif_thumbnail(data: &[u8]) -> Option<Vec<u8>> {
    // Minimum JPEG: SOI (2 bytes) + marker (2 bytes) + length (2 bytes)
    if data.len() < 6 {
        return None;
    }

    // Verify SOI marker
    if data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }

    // Scan for APP0 marker (0xFF 0xE0)
    let mut pos: usize = 2;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            return None;
        }

        let marker_code: u8 = data[pos + 1];

        // Skip padding 0xFF bytes
        if marker_code == 0xFF {
            pos += 1;
            continue;
        }

        // Found APP0
        if marker_code == 0xE0 {
            return parse_jfif_app0(&data[pos + 2..]);
        }

        // Not APP0 — skip this marker segment
        if pos + 4 > data.len() {
            return None;
        }
        let seg_length: usize = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        if seg_length < 2 {
            return None;
        }
        pos += 2 + seg_length;
    }

    None
}

/// Parse JFIF APP0 payload starting at the length field.
fn parse_jfif_app0(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 2 {
        return None;
    }

    let seg_length: usize = u16::from_be_bytes([data[0], data[1]]) as usize;
    if seg_length < 16 {
        // Minimum JFIF payload: 5 (id) + 2 (ver) + 1 (units) + 4 (density) + 2 (thumb dims) = 14
        // Plus 2 for the length field itself = 16
        return None;
    }

    let payload: &[u8] = &data[2..];
    if payload.len() < seg_length - 2 {
        return None;
    }

    // Verify "JFIF\0" identifier
    if payload.len() < 14 || &payload[..5] != b"JFIF\0" {
        return None;
    }

    // Thumbnail dimensions are at offset 12 and 13 within the payload
    // (after "JFIF\0" (5) + version (2) + units (1) + density (4) = 12)
    let thumb_w: u8 = payload[12];
    let thumb_h: u8 = payload[13];

    if thumb_w == 0 || thumb_h == 0 {
        return None;
    }

    let thumb_size: usize = thumb_w as usize * thumb_h as usize * 3;
    let thumb_start: usize = 14; // offset within payload where thumbnail data begins

    if payload.len() < thumb_start + thumb_size {
        return None;
    }

    Some(payload[thumb_start..thumb_start + thumb_size].to_vec())
}
