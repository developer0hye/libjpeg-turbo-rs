/// Parse the EXIF orientation tag (0x0112) from raw TIFF data.
///
/// The input is the raw TIFF data (after stripping the "Exif\0\0" header from APP1).
/// Returns the orientation value (1-8) or `None` if not found or data is malformed.
pub fn parse_orientation(tiff: &[u8]) -> Option<u8> {
    if tiff.len() < 8 {
        return None;
    }

    // Byte order: "II" = little-endian, "MM" = big-endian
    let is_le = match (tiff[0], tiff[1]) {
        (b'I', b'I') => true,
        (b'M', b'M') => false,
        _ => return None,
    };

    // Verify TIFF magic number (42)
    let magic = read_u16(tiff, 2, is_le);
    if magic != 42 {
        return None;
    }

    // Read IFD0 offset
    let ifd_offset = read_u32(tiff, 4, is_le) as usize;
    if ifd_offset + 2 > tiff.len() {
        return None;
    }

    // Read number of IFD entries
    let entry_count = read_u16(tiff, ifd_offset, is_le) as usize;
    let entries_start = ifd_offset + 2;

    // Each IFD entry is 12 bytes: tag(2) + type(2) + count(4) + value/offset(4)
    for i in 0..entry_count {
        let entry_offset = entries_start + i * 12;
        if entry_offset + 12 > tiff.len() {
            return None;
        }

        let tag = read_u16(tiff, entry_offset, is_le);
        if tag == 0x0112 {
            // Orientation tag found — type should be SHORT (3), count 1
            let value_type = read_u16(tiff, entry_offset + 2, is_le);
            if value_type != 3 {
                return None; // unexpected type
            }
            let value = read_u16(tiff, entry_offset + 8, is_le);
            if (1..=8).contains(&value) {
                return Some(value as u8);
            }
            return None;
        }
    }

    None
}

fn read_u16(data: &[u8], offset: usize, is_le: bool) -> u16 {
    if is_le {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    }
}

fn read_u32(data: &[u8], offset: usize, is_le: bool) -> u32 {
    if is_le {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    } else {
        u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build minimal TIFF data with a single IFD entry for orientation.
    fn build_tiff_with_orientation(orientation: u16, is_le: bool) -> Vec<u8> {
        let mut data = Vec::new();

        // Byte order
        if is_le {
            data.extend_from_slice(b"II");
        } else {
            data.extend_from_slice(b"MM");
        }

        // Magic (42)
        push_u16(&mut data, 42, is_le);

        // IFD0 offset (immediately after header = 8)
        push_u32(&mut data, 8, is_le);

        // IFD0: 1 entry
        push_u16(&mut data, 1, is_le);

        // Entry: tag=0x0112 (Orientation), type=3 (SHORT), count=1, value=orientation
        push_u16(&mut data, 0x0112, is_le);
        push_u16(&mut data, 3, is_le); // SHORT
        push_u32(&mut data, 1, is_le); // count
        push_u16(&mut data, orientation, is_le);
        push_u16(&mut data, 0, is_le); // padding

        // Next IFD offset = 0 (end)
        push_u32(&mut data, 0, is_le);

        data
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

    #[test]
    fn orientation_le_normal() {
        let tiff = build_tiff_with_orientation(1, true);
        assert_eq!(parse_orientation(&tiff), Some(1));
    }

    #[test]
    fn orientation_le_rotate_90() {
        let tiff = build_tiff_with_orientation(6, true);
        assert_eq!(parse_orientation(&tiff), Some(6));
    }

    #[test]
    fn orientation_be_rotate_180() {
        let tiff = build_tiff_with_orientation(3, false);
        assert_eq!(parse_orientation(&tiff), Some(3));
    }

    #[test]
    fn orientation_be_rotate_270() {
        let tiff = build_tiff_with_orientation(8, false);
        assert_eq!(parse_orientation(&tiff), Some(8));
    }

    #[test]
    fn all_orientation_values() {
        for val in 1..=8 {
            let tiff = build_tiff_with_orientation(val, true);
            assert_eq!(parse_orientation(&tiff), Some(val as u8));
        }
    }

    #[test]
    fn invalid_orientation_value_zero() {
        let tiff = build_tiff_with_orientation(0, true);
        assert_eq!(parse_orientation(&tiff), None);
    }

    #[test]
    fn invalid_orientation_value_nine() {
        let tiff = build_tiff_with_orientation(9, true);
        assert_eq!(parse_orientation(&tiff), None);
    }

    #[test]
    fn truncated_data() {
        assert_eq!(parse_orientation(&[]), None);
        assert_eq!(parse_orientation(&[0x49, 0x49, 0x2A, 0x00]), None);
    }

    #[test]
    fn wrong_byte_order() {
        let mut tiff = build_tiff_with_orientation(1, true);
        tiff[0] = b'X';
        tiff[1] = b'X';
        assert_eq!(parse_orientation(&tiff), None);
    }

    #[test]
    fn wrong_magic() {
        let mut tiff = build_tiff_with_orientation(1, true);
        // Change magic from 42 to 99
        tiff[2] = 99;
        tiff[3] = 0;
        assert_eq!(parse_orientation(&tiff), None);
    }

    #[test]
    fn no_orientation_tag() {
        // Build TIFF with a different tag (e.g., 0x010F = Make)
        let mut data = Vec::new();
        data.extend_from_slice(b"II");
        push_u16(&mut data, 42, true);
        push_u32(&mut data, 8, true);
        push_u16(&mut data, 1, true); // 1 entry
        push_u16(&mut data, 0x010F, true); // Make tag, not orientation
        push_u16(&mut data, 2, true); // ASCII type
        push_u32(&mut data, 5, true);
        push_u32(&mut data, 0, true); // value offset
        push_u32(&mut data, 0, true); // next IFD
        assert_eq!(parse_orientation(&data), None);
    }
}
