//! P4-135 criteria 1/3/4 (#474): the public SIMD dispatch tables must not
//! hand a downstream crate a safe way to run a kernel out of bounds.
//!
//! Making the arch modules `pub(crate)` (see `simd_module_privacy.rs`) closed
//! the *by-path* route from the issue's proof-of-concept. It did not close the
//! *by-table* route: `simd::detect()` is public and returned a struct whose
//! safe `fn`-pointer fields were `pub`, so the original UB was still one line
//! away with no `unsafe` at the call site:
//!
//! ```ignore
//! let r = libjpeg_turbo_rs::simd::detect();
//! (r.ycbcr_to_rgb_row)(&[], &[], &[], &mut [0u8; 4], 4096);
//! ```
//!
//! `width` is a parameter independent of every slice length, and the AVX2/NEON
//! kernels loop on `width` with raw loads and stores. Empty inputs and a
//! 4-byte output therefore read and write far past both allocations.
//!
//! This file lives in `tests/` deliberately: it compiles as an external crate,
//! so it sees exactly the API surface a downstream user sees. An in-crate test
//! could reach `pub(crate)` items and would not observe the distinction.

/// Every kernel behind `ycbcr_to_rgb_row` writes interleaved RGB.
const RGB_BYTES_PER_PIXEL: usize = 3;

// ---------------------------------------------------------------------------
// Decode table
// ---------------------------------------------------------------------------

#[test]
fn ycbcr_to_rgb_row_accepts_exactly_sized_buffers() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let width: usize = 64;

    let y: Vec<u8> = vec![128; width];
    let cb: Vec<u8> = vec![100; width];
    let cr: Vec<u8> = vec![160; width];
    let mut rgb: Vec<u8> = vec![0; width * RGB_BYTES_PER_PIXEL];

    routines.ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, width);

    // A real conversion ran: mid-grey luma with off-centre chroma is not black.
    assert!(
        rgb.iter().any(|&b| b != 0),
        "conversion left the output untouched, so this test would pass even if \
         validation rejected everything"
    );
}

#[test]
#[should_panic(expected = "ycbcr_to_rgb_row")]
fn ycbcr_to_rgb_row_rejects_the_issue_474_proof_of_concept() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let mut out = [0u8; 4];
    // Verbatim from #474: empty planes, 4-byte output, width 4096.
    routines.ycbcr_to_rgb_row(&[], &[], &[], &mut out, 4096);
}

#[test]
#[should_panic(expected = "ycbcr_to_rgb_row")]
fn ycbcr_to_rgb_row_rejects_short_chroma() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let width: usize = 32;
    let y: Vec<u8> = vec![0; width];
    let cb: Vec<u8> = vec![0; width - 1]; // one sample short
    let cr: Vec<u8> = vec![0; width];
    let mut rgb: Vec<u8> = vec![0; width * RGB_BYTES_PER_PIXEL];

    routines.ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, width);
}

#[test]
#[should_panic(expected = "ycbcr_to_rgb_row")]
fn ycbcr_to_rgb_row_rejects_short_output() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let width: usize = 32;
    let y: Vec<u8> = vec![0; width];
    let cb: Vec<u8> = vec![0; width];
    let cr: Vec<u8> = vec![0; width];
    // Enough for `width` bytes but not for `width * 3`.
    let mut rgb: Vec<u8> = vec![0; width];

    routines.ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, width);
}

/// `width * 3` must not be allowed to wrap into a small, satisfiable bound.
#[test]
#[should_panic(expected = "ycbcr_to_rgb_row")]
fn ycbcr_to_rgb_row_rejects_a_width_whose_byte_count_overflows() {
    let routines = libjpeg_turbo_rs::simd::detect();
    // usize::MAX / 3 + 1 overflows when multiplied by 3 bytes per pixel.
    let width: usize = usize::MAX / RGB_BYTES_PER_PIXEL + 1;
    let mut rgb: Vec<u8> = vec![0; 8];

    routines.ycbcr_to_rgb_row(&[], &[], &[], &mut rgb, width);
}

#[test]
fn fancy_upsample_h2v1_accepts_exactly_sized_buffers() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let in_width: usize = 32;
    let input: Vec<u8> = (0..in_width).map(|i| (i * 7) as u8).collect();
    let mut output: Vec<u8> = vec![0; in_width * 2];

    routines.fancy_upsample_h2v1(&input, in_width, &mut output);

    assert!(
        output.iter().any(|&b| b != 0),
        "upsample left the output untouched"
    );
}

#[test]
#[should_panic(expected = "fancy_upsample_h2v1")]
fn fancy_upsample_h2v1_rejects_short_output() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let in_width: usize = 32;
    let input: Vec<u8> = vec![0; in_width];
    // The contract is `in_width * 2`; one short is out of bounds.
    let mut output: Vec<u8> = vec![0; in_width * 2 - 1];

    routines.fancy_upsample_h2v1(&input, in_width, &mut output);
}

#[test]
#[should_panic(expected = "fancy_upsample_h2v1")]
fn fancy_upsample_h2v1_rejects_short_input() {
    let routines = libjpeg_turbo_rs::simd::detect();
    let in_width: usize = 32;
    let input: Vec<u8> = vec![0; in_width - 1];
    let mut output: Vec<u8> = vec![0; in_width * 2];

    routines.fancy_upsample_h2v1(&input, in_width, &mut output);
}

// ---------------------------------------------------------------------------
// Encode table
// ---------------------------------------------------------------------------

#[test]
fn rgb_to_ycbcr_row_accepts_exactly_sized_buffers() {
    let encoder = libjpeg_turbo_rs::simd::detect_encoder();
    let width: usize = 64;
    let rgb: Vec<u8> = (0..width * RGB_BYTES_PER_PIXEL).map(|i| i as u8).collect();
    let mut y: Vec<u8> = vec![0; width];
    let mut cb: Vec<u8> = vec![0; width];
    let mut cr: Vec<u8> = vec![0; width];

    encoder.rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, width);

    assert!(
        y.iter().any(|&b| b != 0),
        "conversion left the luma plane untouched"
    );
}

#[test]
#[should_panic(expected = "rgb_to_ycbcr_row")]
fn rgb_to_ycbcr_row_rejects_short_input() {
    let encoder = libjpeg_turbo_rs::simd::detect_encoder();
    let width: usize = 64;
    // Holds `width` bytes, not `width * 3`.
    let rgb: Vec<u8> = vec![0; width];
    let mut y: Vec<u8> = vec![0; width];
    let mut cb: Vec<u8> = vec![0; width];
    let mut cr: Vec<u8> = vec![0; width];

    encoder.rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, width);
}

#[test]
#[should_panic(expected = "rgb_to_ycbcr_row")]
fn rgb_to_ycbcr_row_rejects_short_chroma_output() {
    let encoder = libjpeg_turbo_rs::simd::detect_encoder();
    let width: usize = 64;
    let rgb: Vec<u8> = vec![0; width * RGB_BYTES_PER_PIXEL];
    let mut y: Vec<u8> = vec![0; width];
    let mut cb: Vec<u8> = vec![0; width - 1]; // one sample short
    let mut cr: Vec<u8> = vec![0; width];

    encoder.rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, width);
}

#[test]
#[should_panic(expected = "rgb_to_ycbcr_row")]
fn rgb_to_ycbcr_row_rejects_a_width_whose_byte_count_overflows() {
    let encoder = libjpeg_turbo_rs::simd::detect_encoder();
    let width: usize = usize::MAX / RGB_BYTES_PER_PIXEL + 1;
    let mut y: Vec<u8> = vec![0; 8];
    let mut cb: Vec<u8> = vec![0; 8];
    let mut cr: Vec<u8> = vec![0; 8];

    encoder.rgb_to_ycbcr_row(&[], &mut y, &mut cb, &mut cr, width);
}

// ---------------------------------------------------------------------------
// The raw pointers must stay unreachable
// ---------------------------------------------------------------------------

/// The validated methods above are only a guarantee if the raw `fn` pointers
/// they wrap cannot be pulled out of the struct and called directly.
///
/// This asserts the source rather than using `trybuild`, for the reasons
/// written up in `simd_module_privacy.rs`: a stderr snapshot pins exact rustc
/// diagnostics, which drift between the MSRV and stable toolchains this repo
/// both builds on.
#[test]
#[cfg(not(target_family = "wasm"))]
fn hazardous_dispatch_fields_are_crate_private() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("simd")
        .join("mod.rs");
    let source: String =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    // Fields whose safety depends on a `width` argument agreeing with slice
    // lengths. The three IDCT fields and `fdct_quantize` are deliberately
    // absent: their parameters are fixed-size arrays, so every length is in
    // the type and no caller can get them wrong.
    for field in [
        "ycbcr_to_rgb_row",
        "fancy_upsample_h2v1",
        "rgb_to_ycbcr_row",
    ] {
        assert!(
            source.contains(&format!("pub(crate) {field}:")),
            "`{field}` is not a `pub(crate)` field of its dispatch table. If it \
             is `pub`, a downstream crate can bypass the validating wrapper and \
             call the kernel with mismatched lengths — the #474 hole, reopened."
        );
    }
}
