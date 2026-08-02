use super::{
    build_huff_table, format, marker_writer, vec, BitWriter, HuffTable, HuffmanEncoder, JpegError,
    PixelFormat, Result, ToString, Vec,
};

/// Compress as lossless JPEG (SOF3).
///
/// Uses predictor 1 (left) and no point transform.
/// Produces exact pixel-identical output when decoded.
/// Currently supports grayscale only; use `compress_lossless_extended` for color.
pub fn compress_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    compress_lossless_extended(pixels, width, height, pixel_format, 1, 0, 0)
}

/// Compress as lossless JPEG (SOF3) with configurable predictor and point transform.
///
/// # Arguments
/// * `predictor` - Predictor selection value (1-7), as defined in ITU-T T.81 Table H.1
/// * `point_transform` - Point transform value (0-15), right-shifts pixel data before encoding
///
/// Supports grayscale (1-component) and RGB (3-component interleaved).
/// For RGB, the encoder converts to YCbCr before encoding (JFIF convention).
pub fn compress_lossless_extended(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    compress_lossless_extended_precision(
        pixels,
        width,
        height,
        pixel_format,
        predictor,
        point_transform,
        restart_interval,
        8,
    )
}

/// Like `compress_lossless_extended` but with an explicit sample precision
/// (2..=8). The precision field controls the SOF3 marker and the lossless
/// predictor arithmetic; the source samples are still `u8` (8-bit values).
#[allow(clippy::too_many_arguments)]
pub fn compress_lossless_extended_precision(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }

    if !(2..=8).contains(&precision) {
        return Err(JpegError::Unsupported(format!(
            "lossless precision must be 2-8 for 8-bit samples, got {}",
            precision
        )));
    }

    if point_transform >= precision {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-{} for {}-bit precision, got {}",
            precision - 1,
            precision,
            point_transform
        )));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    match pixel_format {
        PixelFormat::Grayscale => compress_lossless_grayscale(
            pixels,
            width,
            height,
            predictor,
            point_transform,
            restart_interval,
            precision,
        ),
        PixelFormat::Rgb => compress_lossless_rgb(
            pixels,
            width,
            height,
            predictor,
            point_transform,
            restart_interval,
            precision,
        ),
        _ => Err(JpegError::Unsupported(format!(
            "lossless encoding does not support {:?}, use Grayscale or Rgb",
            pixel_format
        ))),
    }
}

/// Compute the lossless difference for a single sample.
///
/// Returns the **raw signed difference** `(sample >> Pt) - prediction`,
/// matching libjpeg-turbo `jclossls.c` (`*diff_buf++ = samp - PREDICTOR;`).
/// The lossless JPEG bitstream (ITU-T T.81 Annex H.1.2.2) classifies the
/// diff by its raw 16-bit signed magnitude, NOT by the P-bit modular value.
/// Folding to the P-bit modular range produces a bitstream that decodes to
/// the same pixels (the decoder reconstructs modulo 2^P) but is NOT
/// byte-identical to C cjpeg, because the magnitude category (and thus
/// the optimised Huffman table) differs.
///
/// For 8-bit (P=8) samples the diff is in [-255, +255]; for higher
/// precision it is in [-(2^P - 1), +(2^P - 1)]. Both fit in i16 for
/// P <= 15. The 16-bit precision path lives in `src/api/precision.rs`.
#[allow(clippy::too_many_arguments)]
fn lossless_diff(
    pixel: i32,
    x: usize,
    y: usize,
    plane: &[u8],
    width: usize,
    predictor: u8,
    point_transform: u8,
    precision: u8,
) -> i16 {
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Apply point transform: shift right before encoding
    let sample: i32 = pixel >> point_transform as i32;

    let prediction: i32 = if y == 0 && x == 0 {
        initial_pred
    } else if y == 0 {
        // First row: predictor is always "left" (ra) regardless of psv
        (plane[y * width + x - 1] as i32) >> point_transform as i32
    } else if x == 0 {
        // First column: predictor is always "above" (rb) regardless of psv
        (plane[(y - 1) * width + x] as i32) >> point_transform as i32
    } else {
        let ra: i32 = (plane[y * width + x - 1] as i32) >> point_transform as i32;
        let rb: i32 = (plane[(y - 1) * width + x] as i32) >> point_transform as i32;
        let rc: i32 = (plane[(y - 1) * width + x - 1] as i32) >> point_transform as i32;
        crate::decode::lossless::predict(predictor, ra, rb, rc)
    };

    // Raw signed difference (no modular fold). See doc comment.
    (sample - prediction) as i16
}

/// Encode a single-component (grayscale) lossless JPEG.
fn compress_lossless_grayscale(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    let num_pixels: usize = width * height;
    let ri: u32 = restart_interval as u32;
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Collect all diffs for 2-pass optimized Huffman encoding.
    let mut all_diffs: Vec<i16> = Vec::with_capacity(num_pixels);
    let mut mcu_count: u32 = 0;
    let mut in_restart_row: bool = false;

    for y in 0..height {
        for x in 0..width {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                in_restart_row = true;
            }
            let pixel: i32 = pixels[y * width + x] as i32;
            // After restart, use "first row" prediction: x=0 → initial_pred,
            // x>0 → left neighbor (PSV=1 fallback, matching decoder behavior).
            let signed_diff: i16 = if in_restart_row {
                let sample: i32 = pixel >> point_transform as i32;
                if x == 0 {
                    (sample - initial_pred) as i16
                } else {
                    let left: i32 = pixels[y * width + x - 1] as i32 >> point_transform as i32;
                    (sample - left) as i16
                }
            } else {
                lossless_diff(
                    pixel,
                    x,
                    y,
                    pixels,
                    width,
                    predictor,
                    point_transform,
                    precision,
                )
            };
            all_diffs.push(signed_diff);
            mcu_count += 1;
        }
        in_restart_row = false;
    }

    // Pass 1: gather DC symbol frequencies for optimal Huffman table.
    use crate::encode::huff_opt;
    let mut dc_freq: [u32; 257] = [0u32; 257];
    for &diff in &all_diffs {
        huff_opt::gather_dc_symbol(diff, &mut dc_freq);
    }
    dc_freq[256] = 1;
    let (opt_bits, opt_values) = huff_opt::gen_optimal_table(&dc_freq);
    let dc_table: HuffTable = build_huff_table(&opt_bits, &opt_values);

    // Pass 2: entropy encode with optimal table + restart markers.
    let mut bit_writer: BitWriter = BitWriter::new(num_pixels);
    let mut restart_idx: u8 = 0;
    mcu_count = 0;

    for &diff in &all_diffs {
        if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
            bit_writer.flush();
            bit_writer.write_restart_marker(restart_idx);
            restart_idx = (restart_idx + 1) & 7;
        }
        HuffmanEncoder::encode_dc_only(&mut bit_writer, diff, &dc_table);
        mcu_count += 1;
    }
    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 256);

    marker_writer::write_soi(&mut output);

    // JFIF APP0 marker (matching C cjpeg grayscale lossless)
    marker_writer::write_app0_jfif(&mut output);

    // SOF3 with 1 component
    let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // Optimized DC Huffman table (after SOF3, matching C)
    marker_writer::write_dht(&mut output, 0, 0, &opt_bits, &opt_values);

    // DRI (restart interval)
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    let scan_components: Vec<(u8, u8)> = vec![(1, 0)];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a 3-component RGB interleaved lossless JPEG.
///
/// Stores raw RGB component values with no color conversion, matching
/// C libjpeg-turbo behavior for lossless JPEG (JCS_RGB, no YCbCr conversion).
fn compress_lossless_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    let num_pixels: usize = width * height;
    let ri: u32 = restart_interval as u32;
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Split interleaved RGB into separate planes (no color conversion)
    let mut r_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut g_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut b_plane: Vec<u8> = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        r_plane[i] = pixels[i * 3];
        g_plane[i] = pixels[i * 3 + 1];
        b_plane[i] = pixels[i * 3 + 2];
    }

    let planes: [&[u8]; 3] = [&r_plane, &g_plane, &b_plane];

    // Collect all lossless diffs first for 2-pass optimized Huffman encoding.
    // One MCU = one pixel (all 3 interleaved components).
    let mut all_diffs: Vec<i16> = Vec::with_capacity(num_pixels * 3);
    let mut mcu_count: u32 = 0;
    let mut in_restart_row: bool = false;

    for y in 0..height {
        for x in 0..width {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                in_restart_row = true;
            }
            for plane in &planes {
                let pixel: i32 = plane[y * width + x] as i32;
                // After restart, use "first row" prediction: x=0 → initial_pred,
                // x>0 → left neighbor (PSV=1 fallback, matching decoder).
                let signed_diff: i16 = if in_restart_row {
                    let sample: i32 = pixel >> point_transform as i32;
                    if x == 0 {
                        (sample - initial_pred) as i16
                    } else {
                        let left: i32 = plane[y * width + x - 1] as i32 >> point_transform as i32;
                        (sample - left) as i16
                    }
                } else {
                    lossless_diff(
                        pixel,
                        x,
                        y,
                        plane,
                        width,
                        predictor,
                        point_transform,
                        precision,
                    )
                };
                all_diffs.push(signed_diff);
            }
            mcu_count += 1;
        }
        in_restart_row = false;
    }

    // Pass 1: gather DC symbol frequencies for optimal Huffman table.
    use crate::encode::huff_opt;
    let mut dc_freq: [u32; 257] = [0u32; 257];
    for &diff in &all_diffs {
        huff_opt::gather_dc_symbol(diff, &mut dc_freq);
    }
    dc_freq[256] = 1; // pseudo-symbol (Annex K.2)
    let (opt_bits, opt_values) = huff_opt::gen_optimal_table(&dc_freq);
    let dc_table: HuffTable = build_huff_table(&opt_bits, &opt_values);

    // Pass 2: entropy encode with optimal table + restart markers.
    // Restart markers are emitted between MCUs (1 MCU = 3 component data units).
    let mut bit_writer: BitWriter = BitWriter::new(num_pixels * 3);
    let mut restart_idx: u8 = 0;
    let mut diff_idx: usize = 0;
    mcu_count = 0;

    for _y in 0..height {
        for _x in 0..width {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                bit_writer.flush();
                bit_writer.write_restart_marker(restart_idx);
                restart_idx = (restart_idx + 1) & 7;
            }
            for _ in 0..3 {
                HuffmanEncoder::encode_dc_only(&mut bit_writer, all_diffs[diff_idx], &dc_table);
                diff_idx += 1;
            }
            mcu_count += 1;
        }
    }
    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 512);

    marker_writer::write_soi(&mut output);

    // Adobe APP14 with transform=0 to signal RGB colorspace (matching C cjpeg).
    // C cjpeg does NOT emit JFIF APP0 for RGB lossless — only APP14.
    marker_writer::write_app14_adobe(&mut output, 0);

    // SOF3 with 3 components: R(id='R'), G(id='G'), B(id='B'), all 1x1, qt=0.
    // C libjpeg-turbo uses ASCII component IDs for RGB colorspace lossless.
    let components: Vec<(u8, u8, u8, u8)> = vec![
        (b'R', 1, 1, 0), // R: id=0x52, h=1, v=1, qt=0
        (b'G', 1, 1, 0), // G: id=0x47, h=1, v=1, qt=0
        (b'B', 1, 1, 0), // B: id=0x42, h=1, v=1, qt=0
    ];
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // Optimized DC Huffman table 0 for all 3 components (after SOF3, matching C)
    marker_writer::write_dht(&mut output, 0, 0, &opt_bits, &opt_values);

    // DRI (restart interval)
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // SOS with 3 components: all use DC table 0 (matching SOF3 ASCII IDs)
    let scan_components: Vec<(u8, u8)> = vec![
        (b'R', 0), // R -> DC table 0
        (b'G', 0), // G -> DC table 0
        (b'B', 0), // B -> DC table 0
    ];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress as lossless JPEG with arithmetic entropy coding (SOF11).
///
/// Same predictor-based pipeline as SOF3 but uses ArithEncoder instead of
/// Huffman coding. Writes SOF11 (0xCB) marker and DAC conditioning parameters.
pub fn compress_lossless_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }

    if point_transform >= 8 {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-7 for 8-bit precision, got {}",
            point_transform
        )));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    match pixel_format {
        PixelFormat::Grayscale => compress_lossless_arithmetic_grayscale(
            pixels,
            width,
            height,
            predictor,
            point_transform,
        ),
        PixelFormat::Rgb => {
            compress_lossless_arithmetic_rgb(pixels, width, height, predictor, point_transform)
        }
        _ => Err(JpegError::Unsupported(format!(
            "lossless arithmetic encoding does not support {:?}, use Grayscale or Rgb",
            pixel_format
        ))),
    }
}

/// Encode a single-component (grayscale) lossless JPEG with arithmetic coding.
fn compress_lossless_arithmetic_grayscale(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let precision: u8 = 8;

    let mut arith_enc: ArithEncoder = ArithEncoder::new(width * height);

    // Encode each pixel's difference as a DC coefficient
    for y in 0..height {
        for x in 0..width {
            let pixel: i32 = pixels[y * width + x] as i32;
            let signed_diff: i16 = lossless_diff(
                pixel,
                x,
                y,
                pixels,
                width,
                predictor,
                point_transform,
                precision,
            );
            // Pack the difference into block[0] and encode as DC-only
            let mut block: [i16; 64] = [0i16; 64];
            block[0] = signed_diff.wrapping_add(arith_enc.last_dc_val[0] as i16);
            arith_enc.encode_dc_sequential(&block, 0, 0);
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 256);

    marker_writer::write_soi(&mut output);

    // SOF11 with 1 component
    let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
    marker_writer::write_sof11(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // DAC marker for DC table 0
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    marker_writer::write_dac(&mut output, 1, &dc_params, 0, &ac_params);

    // SOS for lossless scan
    let scan_components: Vec<(u8, u8)> = vec![(1, 0)];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a 3-component RGB interleaved lossless JPEG with arithmetic coding.
///
/// Stores raw RGB component values with no color conversion, matching
/// C libjpeg-turbo behavior for lossless JPEG (JCS_RGB, no YCbCr conversion).
fn compress_lossless_arithmetic_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let precision: u8 = 8;
    let num_pixels: usize = width * height;

    // Split interleaved RGB into separate planes (no color conversion)
    let mut r_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut g_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut b_plane: Vec<u8> = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        r_plane[i] = pixels[i * 3];
        g_plane[i] = pixels[i * 3 + 1];
        b_plane[i] = pixels[i * 3 + 2];
    }

    let planes: [&[u8]; 3] = [&r_plane, &g_plane, &b_plane];
    // All components use DC table 0 (no chrominance table)
    let dc_tbls: [usize; 3] = [0, 0, 0];

    let mut arith_enc: ArithEncoder = ArithEncoder::new(num_pixels * 3);

    // Interleaved encoding: for each pixel, encode diff for Y, Cb, Cr
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel: i32 = planes[c][y * width + x] as i32;
                let signed_diff: i16 = lossless_diff(
                    pixel,
                    x,
                    y,
                    planes[c],
                    width,
                    predictor,
                    point_transform,
                    precision,
                );
                // Pack the difference into block[0] and encode as DC-only
                let mut block: [i16; 64] = [0i16; 64];
                block[0] = signed_diff.wrapping_add(arith_enc.last_dc_val[c] as i16);
                arith_enc.encode_dc_sequential(&block, c, dc_tbls[c]);
            }
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 512);

    marker_writer::write_soi(&mut output);

    // SOF11 with 3 components: R(id=1), G(id=2), B(id=3), all 1x1, qt=0
    let components: Vec<(u8, u8, u8, u8)> = vec![
        (1, 1, 1, 0), // R
        (2, 1, 1, 0), // G
        (3, 1, 1, 0), // B
    ];
    marker_writer::write_sof11(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // DAC marker for DC table 0 only
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    marker_writer::write_dac(&mut output, 1, &dc_params, 0, &ac_params);

    // SOS with 3 components: all use DC table 0
    let scan_components: Vec<(u8, u8)> = vec![
        (1, 0), // R -> DC table 0
        (2, 0), // G -> DC table 0
        (3, 0), // B -> DC table 0
    ];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}
