use super::{
    compress, format, marker_writer, DctMethod, JpegError, PixelFormat, Result, SavedMarker,
    Subsampling, Vec,
};

/// Compress with optional ICC profile and EXIF metadata.
///
/// Inserts APP1 (EXIF) and APP2 (ICC) markers after the APP0 JFIF marker.
#[allow(clippy::too_many_arguments)]
pub fn compress_with_metadata(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let base = compress(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        DctMethod::IsLow,
    )?;
    inject_metadata(&base, icc_profile, exif_data)
}

/// Insert APP1 (EXIF) and APP2 (ICC) markers into an existing JPEG byte stream.
pub fn inject_metadata(
    base: &[u8],
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    inject_metadata_full(base, icc_profile, exif_data, None, None)
}

/// Like [`inject_metadata`] with XMP (APP1) and IPTC (APP13) payloads
/// (issue #358). Segment-sized payloads only; oversized ones error
/// rather than truncating.
pub fn inject_metadata_full(
    base: &[u8],
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
    xmp_data: Option<&[u8]>,
    iptc_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if icc_profile.is_none() && exif_data.is_none() && xmp_data.is_none() && iptc_data.is_none() {
        return Ok(base.to_vec());
    }

    // Find insertion point after all leading APP markers (APP0/JFIF, APP14/Adobe).
    // ICC (APP2) and EXIF (APP1) are inserted after these application markers
    // but before SOF/DHT/DRI/SOS.
    let mut insert_pos: usize = 2; // After SOI
    while insert_pos + 3 < base.len()
        && base[insert_pos] == 0xFF
        && (base[insert_pos + 1] & 0xF0) == 0xE0
    {
        let app_len = u16::from_be_bytes([base[insert_pos + 2], base[insert_pos + 3]]) as usize;
        insert_pos += 2 + app_len;
    }

    let extra_cap = icc_profile.map_or(0, |p| p.len() + 100)
        + exif_data.map_or(0, |e| e.len() + 20)
        + xmp_data.map_or(0, |x| x.len() + 40)
        + iptc_data.map_or(0, |i| i.len() + 40);
    let mut out = Vec::with_capacity(base.len() + extra_cap);
    out.extend_from_slice(&base[..insert_pos]);
    if let Some(exif) = exif_data {
        marker_writer::write_app1_exif(&mut out, exif);
    }
    if let Some(xmp) = xmp_data {
        if !marker_writer::write_app1_xmp(&mut out, xmp) {
            return Err(JpegError::Unsupported(format!(
                "XMP packet of {} bytes exceeds one APP1 segment (Extended XMP writing not implemented)",
                xmp.len()
            )));
        }
    }
    // Adobe/exiftool convention: APP0 JFIF, APP1 Exif, APP1 XMP,
    // APP2 ICC, APP13 Photoshop — ICC precedes IPTC (review LOW).
    if let Some(icc) = icc_profile {
        marker_writer::write_app2_icc(&mut out, icc);
    }
    if let Some(iptc) = iptc_data {
        if !marker_writer::write_app13_iptc(&mut out, iptc) {
            return Err(JpegError::Unsupported(format!(
                "IPTC payload of {} bytes exceeds one APP13 segment",
                iptc.len()
            )));
        }
    }
    out.extend_from_slice(&base[insert_pos..]);
    Ok(out)
}

/// Inject a COM (comment) marker into an existing JPEG byte stream, after APP0.
pub fn inject_comment(base: &[u8], text: &str) -> Vec<u8> {
    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len // SOI(2) + APP0 marker(2) + APP0 data
    } else {
        2 // After SOI only
    };

    let mut out = Vec::with_capacity(base.len() + text.len() + 6);
    out.extend_from_slice(&base[..insert_pos]);
    marker_writer::write_com(&mut out, text);
    out.extend_from_slice(&base[insert_pos..]);
    out
}

/// Inject saved markers (APP/COM) into an existing JPEG byte stream.
///
/// Markers are inserted after SOI + APP0 (and any existing metadata markers),
/// preserving the same insertion point pattern as `inject_metadata`/`inject_comment`.
pub fn inject_saved_markers(base: &[u8], markers: &[SavedMarker]) -> Vec<u8> {
    if markers.is_empty() {
        return base.to_vec();
    }

    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos: usize = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len: usize = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len
    } else {
        2
    };

    let extra: usize = markers.iter().map(|m| m.data.len() + 4).sum();
    let mut out: Vec<u8> = Vec::with_capacity(base.len() + extra);
    out.extend_from_slice(&base[..insert_pos]);
    for marker in markers {
        marker_writer::write_marker(&mut out, marker.code, &marker.data);
    }
    out.extend_from_slice(&base[insert_pos..]);
    out
}
