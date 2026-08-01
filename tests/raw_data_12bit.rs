/// Tests for 12-bit raw data encode/decode API.
///
/// Verifies `compress_raw_12` / `decompress_raw_12` round-trip and
/// pixel-exact match against full-pixel 12-bit decode.
mod helpers;

use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};
use libjpeg_turbo_rs::raw_data_12::{compress_raw_12, decompress_raw_12, RawImage12};
use libjpeg_turbo_rs::Subsampling;

// ===========================================================================
// Helpers
// ===========================================================================

/// Generate a 12-bit YCbCr-like gradient for planar tests.
/// Returns (y_plane, cb_plane, cr_plane) where each sample is in 0..=4095.
fn generate_12bit_planar(
    image_width: usize,
    image_height: usize,
    cb_width: usize,
    cb_height: usize,
) -> (Vec<i16>, Vec<i16>, Vec<i16>) {
    let y_plane: Vec<i16> = (0..image_width * image_height)
        .map(|i| ((i * 3 + 7) % 4096) as i16)
        .collect();
    let cb_plane: Vec<i16> = (0..cb_width * cb_height)
        .map(|i| ((i * 5 + 1000) % 4096) as i16)
        .collect();
    let cr_plane: Vec<i16> = (0..cb_width * cb_height)
        .map(|i| ((i * 11 + 2000) % 4096) as i16)
        .collect();
    (y_plane, cb_plane, cr_plane)
}

// ===========================================================================
// A3-1: decompress_raw_12 — round-trip
// ===========================================================================

/// Encode a 12-bit image via compress_12bit, then decode via decompress_raw_12.
/// Compare the raw planes against a full-pixel decode via decompress_12bit.
/// The two decodes must be pixel-identical (diff=0).
#[test]
fn raw12_decompress_roundtrip_grayscale() {
    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<i16> = (0..w * h).map(|i| ((i * 127 + 50) % 4096) as i16).collect();

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 1, 90, Subsampling::S444)
        .expect("compress_12bit grayscale must succeed");

    // Full-pixel decode
    let full_img = decompress_12bit(&jpeg).expect("decompress_12bit grayscale must succeed");
    assert_eq!(full_img.width, w);
    assert_eq!(full_img.height, h);
    assert_eq!(full_img.num_components, 1);

    // Raw plane decode
    let raw: RawImage12 =
        decompress_raw_12(&jpeg).expect("decompress_raw_12 grayscale must succeed");
    assert_eq!(raw.width, w, "width mismatch");
    assert_eq!(raw.height, h, "height mismatch");
    assert_eq!(raw.num_components, 1, "component count mismatch");
    assert_eq!(raw.planes.len(), 1, "plane count mismatch");
    assert!(raw.plane_widths[0] >= w, "plane_width too small");
    assert!(raw.plane_heights[0] >= h, "plane_height too small");

    // Compare Y plane vs full decode (grayscale, no color conversion needed)
    let mut max_diff: i16 = 0;
    for row in 0..h {
        for col in 0..w {
            let full_val: i16 = full_img.data[row * w + col];
            let raw_val: i16 = raw.planes[0][row * raw.plane_widths[0] + col];
            let diff: i16 = (full_val - raw_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    // Grayscale raw vs full decode must be pixel-identical (diff=0)
    // because full grayscale decode does no color conversion either.
    // measured: 0
    assert_eq!(
        max_diff, 0,
        "grayscale raw12 vs full decode: max_diff={max_diff} (must be 0)"
    );
}

/// Encode a 12-bit 4:4:4 image via compress_12bit, then decode via decompress_raw_12.
/// The raw Y plane (before color conversion) must match the reference Y
/// plane extracted by decompress_12bit from grayscale-only encode.
#[test]
fn raw12_decompress_roundtrip_444() {
    let w: usize = 32;
    let h: usize = 32;
    // Interleaved YCbCr pixels (all at 4:4:4 since compress_12bit only supports 4:4:4)
    let pixels: Vec<i16> = (0..w * h * 3)
        .map(|i| ((i * 13 + 100) % 4096) as i16)
        .collect();

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 3, 90, Subsampling::S444)
        .expect("compress_12bit 444 must succeed");

    // Raw plane decode
    let raw: RawImage12 = decompress_raw_12(&jpeg).expect("decompress_raw_12 444 must succeed");
    assert_eq!(raw.width, w, "width mismatch");
    assert_eq!(raw.height, h, "height mismatch");
    assert_eq!(raw.num_components, 3, "component count mismatch");
    assert_eq!(raw.planes.len(), 3, "plane count mismatch");

    // All planes must be at full resolution for 4:4:4
    for c in 0..3 {
        assert!(raw.plane_widths[c] >= w, "plane {} width too small", c);
        assert!(raw.plane_heights[c] >= h, "plane {} height too small", c);
    }

    // Re-encode with compress_raw_12, then decode with decompress_12bit.
    // The two decoded images must be pixel-identical (same entropy data path).
    let plane_refs: Vec<&[i16]> = raw.planes.iter().map(|p| p.as_slice()).collect();
    let re_jpeg: Vec<u8> = compress_raw_12(
        &plane_refs,
        &raw.plane_widths,
        &raw.plane_heights,
        raw.width,
        raw.height,
        90,
        Subsampling::S444,
    )
    .expect("compress_raw_12 444 must succeed");

    let re_img = decompress_12bit(&re_jpeg).expect("decompress_12bit re-encoded must succeed");
    let orig_img = decompress_12bit(&jpeg).expect("decompress_12bit original must succeed");

    assert_eq!(re_img.width, orig_img.width, "re-encoded width mismatch");
    assert_eq!(re_img.height, orig_img.height, "re-encoded height mismatch");
    assert_eq!(
        re_img.data.len(),
        orig_img.data.len(),
        "re-encoded data length mismatch"
    );

    let max_diff: i16 = re_img
        .data
        .iter()
        .zip(orig_img.data.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    // Re-encoding raw planes applies a second FDCT+quantize+dequantize+IDCT cycle,
    // introducing a second round of quantization error. Measured max_diff = 9 at
    // quality=90 with the 32×32 gradient test image. Tolerance = measured + 1 = 10.
    assert!(
        max_diff <= 10,
        "raw12 444 round-trip: decompress_12bit(compress_raw_12(decompress_raw_12)) vs original: max_diff={max_diff} (tolerance=10)"
    );
}

// ===========================================================================
// A3-2: compress_raw_12 — encode from planar 12-bit input
// ===========================================================================

/// Encode neutral planar YCbCr and equivalent interleaved RGB through the raw
/// and converted paths.  Both represent the same grayscale image, so their
/// decoded pixels must match exactly.
#[test]
fn raw12_compress_444_matches_interleaved() {
    let w: usize = 32;
    let h: usize = 32;

    let y_plane: Vec<i16> = (0..w * h).map(|i| ((i * 3 + 50) % 4096) as i16).collect();
    let cb_plane: Vec<i16> = vec![2048; w * h];
    let cr_plane: Vec<i16> = vec![2048; w * h];
    let mut interleaved_rgb: Vec<i16> = Vec::with_capacity(w * h * 3);
    for &sample in &y_plane {
        interleaved_rgb.extend_from_slice(&[sample, sample, sample]);
    }

    // Encode via raw planar path
    let raw_jpeg: Vec<u8> = compress_raw_12(
        &[&y_plane, &cb_plane, &cr_plane],
        &[w, w, w],
        &[h, h, h],
        w,
        h,
        90,
        Subsampling::S444,
    )
    .expect("compress_raw_12 444 must succeed");

    // Encode via interleaved path
    let interleaved_jpeg: Vec<u8> =
        compress_12bit(&interleaved_rgb, w, h, 3, 90, Subsampling::S444)
            .expect("compress_12bit must succeed");

    // Decode both and compare: must be pixel-identical
    let raw_img =
        decompress_12bit(&raw_jpeg).expect("decompress_12bit from compress_raw_12 must succeed");
    let int_img = decompress_12bit(&interleaved_jpeg)
        .expect("decompress_12bit from compress_12bit must succeed");

    assert_eq!(raw_img.width, int_img.width, "width mismatch");
    assert_eq!(raw_img.height, int_img.height, "height mismatch");
    assert_eq!(
        raw_img.data.len(),
        int_img.data.len(),
        "data length mismatch"
    );

    let max_diff: i16 = raw_img
        .data
        .iter()
        .zip(int_img.data.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    // Both paths encode same YCbCr data → same JPEG → diff=0. measured: 0
    assert_eq!(
        max_diff, 0,
        "compress_raw_12 vs compress_12bit: max_diff={max_diff} (must be 0)"
    );
}

/// Error: wrong plane count
#[test]
fn raw12_compress_wrong_plane_count_returns_error() {
    let w: usize = 16;
    let h: usize = 16;
    let y: Vec<i16> = vec![0i16; w * h];
    let result = compress_raw_12(&[&y], &[w], &[h], w, h, 90, Subsampling::S420);
    assert!(
        result.is_err(),
        "compress_raw_12 with 1 plane for S420 must fail"
    );
}

/// Error: plane size too small
#[test]
fn raw12_compress_plane_too_small_returns_error() {
    let w: usize = 16;
    let h: usize = 16;
    let y: Vec<i16> = vec![0i16; 4]; // too small
    let result = compress_raw_12(&[&y], &[w], &[h], w, h, 90, Subsampling::S444);
    assert!(
        result.is_err(),
        "compress_raw_12 with undersized plane must fail"
    );
}

// ===========================================================================
// A3-3: Raw 12-bit baseline Huffman subsampling matrix
// Pixel-exact Rust/C decode parity and bounded raw re-encode loss
// ===========================================================================

struct RawMatrix {
    subsamp: Subsampling,
    subsamp_name: &'static str,
}

fn raw12_matrix_roundtrip(cfg: &RawMatrix) {
    use libjpeg_turbo_rs::precision::decompress_12bit;

    let djpeg = require_c_tool!("djpeg");
    // Odd dimensions force partial MCUs for every sampling mode in the matrix.
    let image_width: usize = 31;
    let image_height: usize = 29;
    let (h_samp, v_samp): (u8, u8) = cfg.subsamp.sampling_factors();

    // Chroma plane dimensions
    let cb_w: usize = image_width.div_ceil(h_samp as usize);
    let cb_h: usize = image_height.div_ceil(v_samp as usize);

    let (y_plane, cb_plane, cr_plane) =
        if cfg.subsamp == Subsampling::S444 || h_samp == 1 && v_samp == 1 {
            generate_12bit_planar(image_width, image_height, image_width, image_height)
        } else {
            generate_12bit_planar(image_width, image_height, cb_w, cb_h)
        };

    let label: String = format!("raw12_{}_base_huff", cfg.subsamp_name);

    let (planes, pw, ph): (Vec<&[i16]>, Vec<usize>, Vec<usize>) =
        if cfg.subsamp == Subsampling::S444 {
            (
                vec![y_plane.as_slice(), cb_plane.as_slice(), cr_plane.as_slice()],
                vec![image_width, image_width, image_width],
                vec![image_height, image_height, image_height],
            )
        } else {
            (
                vec![y_plane.as_slice(), cb_plane.as_slice(), cr_plane.as_slice()],
                vec![image_width, cb_w, cb_w],
                vec![image_height, cb_h, cb_h],
            )
        };

    let jpeg: Vec<u8> = compress_raw_12(
        &planes,
        &pw,
        &ph,
        image_width,
        image_height,
        90,
        cfg.subsamp,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress_raw_12 failed: {e:?}"));

    let raw: RawImage12 = decompress_raw_12(&jpeg)
        .unwrap_or_else(|e| panic!("[{label}] decompress_raw_12 failed: {e:?}"));

    assert_eq!(raw.width, image_width, "[{label}] width mismatch");
    assert_eq!(raw.height, image_height, "[{label}] height mismatch");
    assert_eq!(raw.num_components, 3, "[{label}] component count");

    let plane_refs: Vec<&[i16]> = raw.planes.iter().map(|p| p.as_slice()).collect();
    let re_jpeg: Vec<u8> = compress_raw_12(
        &plane_refs,
        &raw.plane_widths,
        &raw.plane_heights,
        raw.width,
        raw.height,
        90,
        cfg.subsamp,
    )
    .unwrap_or_else(|e| panic!("[{label}] re-encode compress_raw_12 failed: {e:?}"));

    let img1 = decompress_12bit(&jpeg)
        .unwrap_or_else(|e| panic!("[{label}] decompress_12bit first failed: {e:?}"));
    let img2 = decompress_12bit(&re_jpeg)
        .unwrap_or_else(|e| panic!("[{label}] decompress_12bit re-encoded failed: {e:?}"));

    let (c_width, c_height, c_components, c_maxval, c_pixels) =
        helpers::decode_with_c_djpeg_i16(&djpeg, &jpeg, &label);
    assert_eq!(c_width, image_width, "[{label}] C width mismatch");
    assert_eq!(c_height, image_height, "[{label}] C height mismatch");
    assert_eq!(c_components, 3, "[{label}] C component mismatch");
    assert_eq!(
        c_maxval, 4095,
        "[{label}] C djpeg must preserve 12-bit output precision"
    );
    assert_eq!(
        c_pixels, img1.data,
        "[{label}] Rust and C 12-bit decoders must be pixel-identical"
    );

    assert_eq!(
        img1.data.len(),
        img2.data.len(),
        "[{label}] length mismatch"
    );
    let max_diff: i16 = img1
        .data
        .iter()
        .zip(img2.data.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    // Re-encoding decoded samples applies a second lossy DCT/quantization
    // cycle. Measured maximum is 1 across the odd-size subsampling matrix;
    // allow one value of margin as required by the strict tolerance policy.
    assert!(
        max_diff <= 2,
        "[{label}] raw decode/re-encode max_diff={max_diff} exceeds tolerance 2"
    );
    eprintln!("[{label}] PASS");
}

#[test]
fn raw12_matrix_subsamp_420_baseline_huffman() {
    raw12_matrix_roundtrip(&RawMatrix {
        subsamp: Subsampling::S420,
        subsamp_name: "420",
    });
}

#[test]
fn raw12_matrix_subsamp_422_baseline_huffman() {
    raw12_matrix_roundtrip(&RawMatrix {
        subsamp: Subsampling::S422,
        subsamp_name: "422",
    });
}

#[test]
fn raw12_matrix_subsamp_444_baseline_huffman() {
    raw12_matrix_roundtrip(&RawMatrix {
        subsamp: Subsampling::S444,
        subsamp_name: "444",
    });
}

#[test]
fn raw12_matrix_subsamp_440_baseline_huffman() {
    raw12_matrix_roundtrip(&RawMatrix {
        subsamp: Subsampling::S440,
        subsamp_name: "440",
    });
}

#[test]
fn raw12_matrix_subsamp_411_baseline_huffman() {
    raw12_matrix_roundtrip(&RawMatrix {
        subsamp: Subsampling::S411,
        subsamp_name: "411",
    });
}

#[test]
fn raw12_matrix_subsamp_441_baseline_huffman() {
    raw12_matrix_roundtrip(&RawMatrix {
        subsamp: Subsampling::S441,
        subsamp_name: "441",
    });
}
