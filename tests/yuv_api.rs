/// Integration tests for the YUV planar encode/decode API.
///
/// These tests cover the full set of `encode_yuv`, `decode_yuv`,
/// `compress_from_yuv`, `decompress_to_yuv` functions plus buffer size helpers.
use libjpeg_turbo_rs::api::yuv;
use libjpeg_turbo_rs::{
    compress, decompress, yuv_buf_size, yuv_plane_height, yuv_plane_size, yuv_plane_width,
    PixelFormat, Subsampling,
};

/// Helper: generate a simple gradient RGB image (3 bpp).
fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

// ──────────────────────────────────────────────
// 1. encode_yuv roundtrip: RGB → YUV → RGB
// ──────────────────────────────────────────────

#[test]
fn encode_decode_yuv_roundtrip_444() {
    let width: usize = 16;
    let height: usize = 16;
    let original: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> = yuv::encode_yuv(
        &original,
        width,
        height,
        PixelFormat::Rgb,
        Subsampling::S444,
    )
    .unwrap();
    let expected_size: usize = yuv_buf_size(width, height, Subsampling::S444);
    assert_eq!(yuv_packed.len(), expected_size);

    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S444,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), original.len());

    // BT.601 roundtrip has rounding error; accept +/- 2 per channel
    for i in 0..original.len() {
        let diff: i16 = original[i] as i16 - decoded[i] as i16;
        assert!(
            diff.abs() <= 2,
            "pixel byte {} differs by {}: original={}, decoded={}",
            i,
            diff,
            original[i],
            decoded[i]
        );
    }
}

#[test]
fn encode_decode_yuv_roundtrip_420() {
    let width: usize = 32;
    let height: usize = 32;
    let original: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> = yuv::encode_yuv(
        &original,
        width,
        height,
        PixelFormat::Rgb,
        Subsampling::S420,
    )
    .unwrap();
    let expected_size: usize = yuv_buf_size(width, height, Subsampling::S420);
    assert_eq!(yuv_packed.len(), expected_size);

    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), original.len());

    // With 4:2:0 subsampling, chroma is averaged so more error is expected
    let max_error: i16 = 40;
    let mut total_error: i64 = 0;
    for i in 0..original.len() {
        let diff: i16 = original[i] as i16 - decoded[i] as i16;
        total_error += diff.abs() as i64;
        assert!(
            diff.abs() <= max_error,
            "pixel byte {} differs by {}: original={}, decoded={}",
            i,
            diff,
            original[i],
            decoded[i]
        );
    }
    // Average error should be modest
    let avg_error: f64 = total_error as f64 / original.len() as f64;
    assert!(
        avg_error < 10.0,
        "average error {} too high for 4:2:0 roundtrip",
        avg_error
    );
}

// ──────────────────────────────────────────────
// 2. encode_yuv_planes produces correct plane sizes
// ──────────────────────────────────────────────

#[test]
fn encode_yuv_planes_correct_sizes_444() {
    let width: usize = 24;
    let height: usize = 16;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S444)
            .unwrap();
    assert_eq!(planes.len(), 3);

    for comp in 0..3 {
        let expected: usize = yuv_plane_size(comp, width, height, Subsampling::S444);
        assert_eq!(
            planes[comp].len(),
            expected,
            "plane {} size mismatch: got {}, expected {}",
            comp,
            planes[comp].len(),
            expected
        );
    }
}

#[test]
fn encode_yuv_planes_correct_sizes_420() {
    let width: usize = 32;
    let height: usize = 24;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420)
            .unwrap();
    assert_eq!(planes.len(), 3);

    let y_size: usize = yuv_plane_size(0, width, height, Subsampling::S420);
    let cb_size: usize = yuv_plane_size(1, width, height, Subsampling::S420);
    let cr_size: usize = yuv_plane_size(2, width, height, Subsampling::S420);

    assert_eq!(planes[0].len(), y_size);
    assert_eq!(planes[1].len(), cb_size);
    assert_eq!(planes[2].len(), cr_size);
    // For 4:2:0: chroma is 1/4 of luma
    assert_eq!(cb_size, y_size / 4);
    assert_eq!(cr_size, y_size / 4);
}

#[test]
fn encode_yuv_planes_correct_sizes_422() {
    let width: usize = 32;
    let height: usize = 16;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S422)
            .unwrap();
    assert_eq!(planes.len(), 3);

    for comp in 0..3 {
        let expected: usize = yuv_plane_size(comp, width, height, Subsampling::S422);
        assert_eq!(planes[comp].len(), expected);
    }
    // For 4:2:2: chroma width is half, height is same
    assert_eq!(planes[1].len(), planes[0].len() / 2);
}

// ──────────────────────────────────────────────
// 3. compress_from_yuv → decompress produces valid image
// ──────────────────────────────────────────────

#[test]
fn compress_from_yuv_produces_valid_jpeg() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> =
        yuv::encode_yuv(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420).unwrap();

    let jpeg_data: Vec<u8> =
        yuv::compress_from_yuv(&yuv_packed, width, height, Subsampling::S420, 90).unwrap();
    assert!(jpeg_data.len() > 2);
    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI marker

    // Decompress and check dimensions
    let image = decompress(&jpeg_data).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
}

#[test]
fn compress_from_yuv_planes_produces_valid_jpeg() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S444)
            .unwrap();
    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();

    let jpeg_data: Vec<u8> =
        yuv::compress_from_yuv_planes(&plane_refs, width, height, Subsampling::S444, 90).unwrap();

    let image = decompress(&jpeg_data).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
}

// ──────────────────────────────────────────────
// 4. decompress_to_yuv → compress_from_yuv roundtrip
// ──────────────────────────────────────────────

#[test]
fn decompress_to_yuv_roundtrip() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        Subsampling::S420,
    )
    .unwrap();

    let (yuv_buf, dec_w, dec_h, dec_sub) = yuv::decompress_to_yuv(&jpeg_data).unwrap();
    assert_eq!(dec_w, width);
    assert_eq!(dec_h, height);
    assert_eq!(dec_sub, Subsampling::S420);

    // Re-compress from YUV
    let jpeg_data2: Vec<u8> = yuv::compress_from_yuv(&yuv_buf, dec_w, dec_h, dec_sub, 95).unwrap();
    let image2 = decompress(&jpeg_data2).unwrap();
    assert_eq!(image2.width, width);
    assert_eq!(image2.height, height);
}

#[test]
fn decompress_to_yuv_planes_roundtrip() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        Subsampling::S444,
    )
    .unwrap();

    let (planes, dec_w, dec_h, dec_sub) = yuv::decompress_to_yuv_planes(&jpeg_data).unwrap();
    assert_eq!(dec_w, width);
    assert_eq!(dec_h, height);
    assert_eq!(dec_sub, Subsampling::S444);
    assert_eq!(planes.len(), 3);

    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
    let jpeg_data2: Vec<u8> =
        yuv::compress_from_yuv_planes(&plane_refs, dec_w, dec_h, dec_sub, 95).unwrap();
    let image2 = decompress(&jpeg_data2).unwrap();
    assert_eq!(image2.width, width);
    assert_eq!(image2.height, height);
}

// ──────────────────────────────────────────────
// 5. decode_yuv_planes with 4:2:0 produces correct output size
// ──────────────────────────────────────────────

#[test]
fn decode_yuv_planes_420_correct_output_size() {
    let width: usize = 48;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420)
            .unwrap();
    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();

    let decoded: Vec<u8> = yuv::decode_yuv_planes(
        &plane_refs,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), width * height * 3);

    // Also test RGBA output
    let decoded_rgba: Vec<u8> = yuv::decode_yuv_planes(
        &plane_refs,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgba,
    )
    .unwrap();
    assert_eq!(decoded_rgba.len(), width * height * 4);
}

// ──────────────────────────────────────────────
// 6. Buffer size helpers return correct values
// ──────────────────────────────────────────────

#[test]
fn buffer_size_helpers_444() {
    let width: usize = 640;
    let height: usize = 480;

    // 4:4:4: all planes same size
    let pw0: usize = yuv_plane_width(0, width, Subsampling::S444);
    let pw1: usize = yuv_plane_width(1, width, Subsampling::S444);
    let ph0: usize = yuv_plane_height(0, height, Subsampling::S444);
    let ph1: usize = yuv_plane_height(1, height, Subsampling::S444);

    assert_eq!(pw0, 640);
    assert_eq!(pw1, 640);
    assert_eq!(ph0, 480);
    assert_eq!(ph1, 480);

    let total: usize = yuv_buf_size(width, height, Subsampling::S444);
    assert_eq!(total, 640 * 480 * 3);
}

#[test]
fn buffer_size_helpers_420() {
    let width: usize = 640;
    let height: usize = 480;

    let pw0: usize = yuv_plane_width(0, width, Subsampling::S420);
    let pw1: usize = yuv_plane_width(1, width, Subsampling::S420);
    let ph0: usize = yuv_plane_height(0, height, Subsampling::S420);
    let ph1: usize = yuv_plane_height(1, height, Subsampling::S420);

    assert_eq!(pw0, 640);
    assert_eq!(pw1, 320);
    assert_eq!(ph0, 480);
    assert_eq!(ph1, 240);

    let y_size: usize = yuv_plane_size(0, width, height, Subsampling::S420);
    let cb_size: usize = yuv_plane_size(1, width, height, Subsampling::S420);
    assert_eq!(y_size, 640 * 480);
    assert_eq!(cb_size, 320 * 240);

    let total: usize = yuv_buf_size(width, height, Subsampling::S420);
    assert_eq!(total, 640 * 480 + 2 * 320 * 240);
}

#[test]
fn buffer_size_helpers_422() {
    let width: usize = 640;
    let height: usize = 480;

    let pw0: usize = yuv_plane_width(0, width, Subsampling::S422);
    let pw1: usize = yuv_plane_width(1, width, Subsampling::S422);
    let ph0: usize = yuv_plane_height(0, height, Subsampling::S422);
    let ph1: usize = yuv_plane_height(1, height, Subsampling::S422);

    assert_eq!(pw0, 640);
    assert_eq!(pw1, 320);
    assert_eq!(ph0, 480);
    assert_eq!(ph1, 480);

    let total: usize = yuv_buf_size(width, height, Subsampling::S422);
    assert_eq!(total, 640 * 480 + 2 * 320 * 480);
}

#[test]
fn buffer_size_helpers_odd_dimensions() {
    // Odd dimensions should be padded up
    let pw0: usize = yuv_plane_width(0, 641, Subsampling::S420);
    let pw1: usize = yuv_plane_width(1, 641, Subsampling::S420);
    assert_eq!(pw0, 642); // padded to multiple of 2
    assert_eq!(pw1, 321);

    let ph0: usize = yuv_plane_height(0, 481, Subsampling::S420);
    let ph1: usize = yuv_plane_height(1, 481, Subsampling::S420);
    assert_eq!(ph0, 482); // padded to multiple of 2
    assert_eq!(ph1, 241);
}

// ──────────────────────────────────────────────
// 7. Grayscale YUV (single plane)
// ──────────────────────────────────────────────

#[test]
fn grayscale_encode_yuv_single_plane() {
    let width: usize = 16;
    let height: usize = 16;
    // Grayscale input
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let yuv_packed: Vec<u8> = yuv::encode_yuv(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        Subsampling::S444,
    )
    .unwrap();
    // For grayscale, the YUV buffer is just the Y plane (no Cb/Cr)
    let y_size: usize = yuv_plane_size(0, width, height, Subsampling::S444);
    assert_eq!(yuv_packed.len(), y_size);

    // Decode back to grayscale
    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S444,
        PixelFormat::Grayscale,
    )
    .unwrap();
    assert_eq!(decoded.len(), width * height);
    // Grayscale → Y is identity, so should be exact
    assert_eq!(decoded, pixels);
}

#[test]
fn grayscale_encode_yuv_planes_single_plane() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let planes: Vec<Vec<u8>> = yuv::encode_yuv_planes(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        Subsampling::S444,
    )
    .unwrap();
    assert_eq!(planes.len(), 1); // Only Y plane for grayscale
    assert_eq!(planes[0].len(), width * height);
}

// ──────────────────────────────────────────────
// Additional: BGRA pixel format support
// ──────────────────────────────────────────────

#[test]
fn encode_decode_yuv_bgra_format() {
    let width: usize = 16;
    let height: usize = 16;
    // BGRA input
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = 128;
            pixels.push(b); // B
            pixels.push(g); // G
            pixels.push(r); // R
            pixels.push(255); // A
        }
    }

    let yuv_packed: Vec<u8> =
        yuv::encode_yuv(&pixels, width, height, PixelFormat::Bgra, Subsampling::S444).unwrap();
    let expected_size: usize = yuv_buf_size(width, height, Subsampling::S444);
    assert_eq!(yuv_packed.len(), expected_size);

    // Decode back to BGRA
    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S444,
        PixelFormat::Bgra,
    )
    .unwrap();
    assert_eq!(decoded.len(), pixels.len());
}

// ──────────────────────────────────────────────
// Edge case: non-multiple-of-MCU dimensions
// ──────────────────────────────────────────────

#[test]
fn encode_decode_yuv_non_aligned_dimensions() {
    let width: usize = 17;
    let height: usize = 13;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> =
        yuv::encode_yuv(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420).unwrap();

    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), width * height * 3);
}
