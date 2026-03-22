use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit,
};
use libjpeg_turbo_rs::quantize::{dequantize, quantize, DitherMode, QuantizeOptions};
use libjpeg_turbo_rs::{
    compress, compress_into, decompress, read_scanlines_12, read_scanlines_16, requantize,
    write_scanlines_12, write_scanlines_16, PixelFormat, ScanlineDecoder, Subsampling,
};

/// 12-bit scanline write/read roundtrip: encode via write_scanlines_12, then
/// decode via read_scanlines_12, and verify recovered samples are close to the
/// originals (DCT is lossy, so we allow small tolerance).
#[test]
fn scanline_12bit_roundtrip() {
    let width: usize = 16;
    let height: usize = 8;
    let num_components: usize = 1;
    // Build 12-bit sample rows (0-4095 range)
    let rows: Vec<Vec<i16>> = (0..height)
        .map(|y| {
            (0..width)
                .map(|x| ((y * width + x) * 256 % 4096) as i16)
                .collect()
        })
        .collect();

    let row_refs: Vec<&[i16]> = rows.iter().map(|r| r.as_slice()).collect();

    let jpeg_data: Vec<u8> = write_scanlines_12(
        &row_refs,
        width,
        height,
        num_components,
        95,
        Subsampling::S444,
    )
    .expect("write_scanlines_12 should succeed");

    // Should start with SOI marker
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]);

    let decoded_rows: Vec<Vec<i16>> =
        read_scanlines_12(&jpeg_data, height).expect("read_scanlines_12 should succeed");

    assert_eq!(decoded_rows.len(), height);
    for (y, (original, decoded)) in rows.iter().zip(decoded_rows.iter()).enumerate() {
        assert_eq!(original.len(), decoded.len(), "row {} length mismatch", y);
        for (x, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            let diff: i16 = (orig - dec).abs();
            assert!(
                diff < 200,
                "12-bit sample at ({},{}) differs too much: orig={}, decoded={}, diff={}",
                x,
                y,
                orig,
                dec,
                diff
            );
        }
    }
}

/// 16-bit scanline write/read roundtrip. 16-bit is lossless, so we expect exact
/// recovery.
#[test]
fn scanline_16bit_roundtrip() {
    let width: usize = 8;
    let height: usize = 4;
    let num_components: usize = 1;
    let rows: Vec<Vec<u16>> = (0..height)
        .map(|y| {
            (0..width)
                .map(|x| ((y * width + x) * 1000 % 65536) as u16)
                .collect()
        })
        .collect();

    let row_refs: Vec<&[u16]> = rows.iter().map(|r| r.as_slice()).collect();

    let jpeg_data: Vec<u8> = write_scanlines_16(&row_refs, width, height, num_components, 1, 0)
        .expect("write_scanlines_16 should succeed");

    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]);

    let decoded_rows: Vec<Vec<u16>> =
        read_scanlines_16(&jpeg_data, height).expect("read_scanlines_16 should succeed");

    assert_eq!(decoded_rows.len(), height);
    for (y, (original, decoded)) in rows.iter().zip(decoded_rows.iter()).enumerate() {
        assert_eq!(original.len(), decoded.len(), "row {} length mismatch", y);
        for (x, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig, dec,
                "16-bit sample at ({},{}) differs: orig={}, decoded={}",
                x, y, orig, dec
            );
        }
    }
}

/// Bottom-up decode: when enabled, the output rows should be the reverse of
/// normal top-to-bottom decode order.
#[test]
fn bottom_up_decode_produces_flipped_rows() {
    let width: usize = 8;
    let height: usize = 8;
    // Create a gradient image: row 0 is dark, row 7 is bright
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val: u8 = (y * 32).min(255) as u8;
        for _x in 0..width {
            pixels.extend_from_slice(&[val, val, val]);
        }
    }

    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("compress should succeed");

    // Normal decode
    let normal = decompress(&jpeg_data).expect("decompress should succeed");

    // Bottom-up decode
    let mut decoder = ScanlineDecoder::new(&jpeg_data).expect("decoder should succeed");
    decoder.set_bottom_up(true);
    let flipped = decoder.finish().expect("finish should succeed");

    assert_eq!(normal.width, flipped.width);
    assert_eq!(normal.height, flipped.height);

    let bpp: usize = normal.pixel_format.bytes_per_pixel();
    let row_bytes: usize = normal.width * bpp;

    // Verify row order is reversed
    for y in 0..height {
        let normal_row: &[u8] = &normal.data[y * row_bytes..(y + 1) * row_bytes];
        let flipped_row: &[u8] =
            &flipped.data[(height - 1 - y) * row_bytes..(height - y) * row_bytes];
        assert_eq!(
            normal_row,
            flipped_row,
            "row {} of normal should match row {} of flipped",
            y,
            height - 1 - y
        );
    }
}

/// compress_into with a sufficient buffer should succeed and return byte count.
#[test]
fn compress_into_sufficient_buffer() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = vec![128u8; width * height * 3];

    // Allocate a generous buffer
    let mut buf: Vec<u8> = vec![0u8; width * height * 3 + 4096];

    let written: usize = compress_into(
        &mut buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress_into should succeed");

    assert!(written > 0, "should write some bytes");
    assert!(written <= buf.len(), "should not exceed buffer");
    // Verify it starts with SOI marker
    assert_eq!(&buf[0..2], &[0xFF, 0xD8]);
    // Verify it ends with EOI marker
    assert_eq!(&buf[written - 2..written], &[0xFF, 0xD9]);
}

/// compress_into with an insufficient buffer should return an error.
#[test]
fn compress_into_insufficient_buffer() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = vec![128u8; width * height * 3];

    // Allocate a tiny buffer that cannot hold the JPEG
    let mut buf: Vec<u8> = vec![0u8; 10];

    let result = compress_into(
        &mut buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    );

    assert!(
        result.is_err(),
        "compress_into should fail with tiny buffer"
    );
}

/// requantize: take an already-quantized image and re-quantize it with a
/// different palette. The resulting image should use only colors from the new
/// palette.
#[test]
fn requantize_with_different_palette() {
    // Create a simple 4x4 RGB image with known colors
    let width: usize = 4;
    let height: usize = 4;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        pixels.extend_from_slice(&[200, 50, 50]); // reddish
    }

    // First quantize to a 4-color palette
    let opts = QuantizeOptions {
        num_colors: 4,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let quantized = quantize(&pixels, width, height, &opts).expect("quantize should succeed");

    // Now re-quantize with a completely different palette
    let new_palette: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
    let requ = requantize(&quantized, &new_palette, DitherMode::None);

    assert_eq!(requ.width, width);
    assert_eq!(requ.height, height);
    assert_eq!(requ.indices.len(), width * height);
    assert_eq!(requ.palette, new_palette);

    // All indices should be valid indices into the new palette
    for &idx in &requ.indices {
        assert!(
            (idx as usize) < new_palette.len(),
            "index {} out of range",
            idx
        );
    }

    // Since the original was reddish, the nearest color in the new palette
    // should be [255, 0, 0] (index 1)
    for &idx in &requ.indices {
        assert_eq!(idx, 1, "reddish pixel should map to red in new palette");
    }
}
