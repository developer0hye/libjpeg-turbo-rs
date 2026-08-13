//! Relocated from `tests/simd_x86.rs` for P4-135 criterion 2 (#474).
//!
//! This suite reaches SIMD kernels directly, which is why the arch
//! modules had to stay `pub` and were therefore callable from any
//! downstream crate. As an in-crate test it uses `crate::`, so they
//! can be private. Moved verbatim apart from the path rewrite.

//! x86_64 SSE2 SIMD tests -- verify byte-exact match with scalar implementation.
#![cfg(all(target_arch = "x86_64", feature = "simd"))]

use crate::simd;

fn scalar_routines() -> simd::SimdRoutines {
    std::env::set_var("JSIMD_FORCENONE", "1");
    let r = simd::detect();
    std::env::remove_var("JSIMD_FORCENONE");
    r
}

fn sse2_routines() -> simd::SimdRoutines {
    simd::x86_64::routines()
}

#[test]
fn sse2_idct_all_zeros() {
    let coeffs = [0i16; 64];
    let quant = [1u16; 64];
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    let (mut expected, mut actual) = ([0u8; 64], [0u8; 64]);
    (scalar.idct_islow)(&coeffs, &quant, &mut expected);
    (sse2.idct_islow)(&coeffs, &quant, &mut actual);
    assert_eq!(actual, expected);
}

#[test]
fn sse2_idct_dc_only() {
    let quant = [1u16; 64];
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    for dc in [-2000i16, -1000, -500, -100, -1, 0, 1, 100, 500, 1000, 2000] {
        let mut coeffs = [0i16; 64];
        coeffs[0] = dc;
        let (mut expected, mut actual) = ([0u8; 64], [0u8; 64]);
        (scalar.idct_islow)(&coeffs, &quant, &mut expected);
        (sse2.idct_islow)(&coeffs, &quant, &mut actual);
        assert_eq!(actual, expected, "DC-only mismatch for dc={dc}");
    }
}

#[test]
fn sse2_idct_with_ac() {
    let mut coeffs = [0i16; 64];
    coeffs[0] = 200;
    coeffs[1] = -30;
    coeffs[8] = 15;
    coeffs[9] = -10;
    coeffs[16] = -5;
    let mut quant = [1u16; 64];
    quant[0] = 16;
    quant[1] = 11;
    quant[8] = 12;
    quant[9] = 10;
    quant[16] = 14;
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    let (mut expected, mut actual) = ([0u8; 64], [0u8; 64]);
    (scalar.idct_islow)(&coeffs, &quant, &mut expected);
    (sse2.idct_islow)(&coeffs, &quant, &mut actual);
    assert_eq!(actual, expected);
}

#[test]
fn sse2_idct_clamping() {
    let quant = [1u16; 64];
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    for dc in [4000i16, -4000] {
        let mut coeffs = [0i16; 64];
        coeffs[0] = dc;
        let (mut expected, mut actual) = ([0u8; 64], [0u8; 64]);
        (scalar.idct_islow)(&coeffs, &quant, &mut expected);
        (sse2.idct_islow)(&coeffs, &quant, &mut actual);
        assert_eq!(actual, expected, "clamping mismatch for dc={dc}");
    }
}

#[test]
fn sse2_color_gray() {
    let width = 32;
    let (y, cb, cr) = (vec![128u8; width], vec![128u8; width], vec![128u8; width]);
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    let (mut expected, mut actual) = (vec![0u8; width * 3], vec![0u8; width * 3]);
    (scalar.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut expected, width);
    (sse2.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut actual, width);
    assert_eq!(actual, expected);
}

#[test]
fn sse2_color_gradient() {
    let width = 64;
    let y: Vec<u8> = (0..width).map(|i| (i * 4) as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| (128 + (i as i32 - 32)) as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| (128 - (i as i32 - 32)) as u8).collect();
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    let (mut expected, mut actual) = (vec![0u8; width * 3], vec![0u8; width * 3]);
    (scalar.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut expected, width);
    (sse2.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut actual, width);
    assert_eq!(actual, expected);
}

#[test]
fn sse2_color_non_aligned() {
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    for width in [1, 3, 7, 9, 15, 17, 31, 33] {
        let y: Vec<u8> = (0..width).map(|i| ((i * 7) % 256) as u8).collect();
        let cb: Vec<u8> = (0..width).map(|i| ((i * 11 + 64) % 256) as u8).collect();
        let cr: Vec<u8> = (0..width).map(|i| ((i * 13 + 128) % 256) as u8).collect();
        let (mut expected, mut actual) = (vec![0u8; width * 3], vec![0u8; width * 3]);
        (scalar.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut expected, width);
        (sse2.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut actual, width);
        assert_eq!(actual, expected, "width={width}");
    }
}

#[test]
fn sse2_upsample_edge_cases() {
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    // Single sample
    let (mut expected, mut actual) = ([0u8; 2], [0u8; 2]);
    (scalar.fancy_upsample_h2v1)(&[42], 1, &mut expected);
    (sse2.fancy_upsample_h2v1)(&[42], 1, &mut actual);
    assert_eq!(actual, expected);
    // Two-sample case (width=2): both scalar and SSE2 emit box filter
    // [i0, i0, i1, i1] (matches C merged path when downsampled_width=2).
    for input in [[0u8, 0], [50, 200], [255, 128], [7, 13]] {
        let (mut expected, mut actual) = ([0u8; 4], [0u8; 4]);
        (scalar.fancy_upsample_h2v1)(&input, 2, &mut expected);
        (sse2.fancy_upsample_h2v1)(&input, 2, &mut actual);
        assert_eq!(
            actual, expected,
            "SSE2 width=2 mismatch for input={input:?}"
        );
    }
}

#[test]
fn sse2_upsample_various_widths() {
    let (scalar, sse2) = (scalar_routines(), sse2_routines());
    for in_width in [3, 4, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 64, 100] {
        let input: Vec<u8> = (0..in_width).map(|i| ((i * 13 + 7) % 256) as u8).collect();
        let out_width = in_width * 2;
        let (mut expected, mut actual) = (vec![0u8; out_width], vec![0u8; out_width]);
        (scalar.fancy_upsample_h2v1)(&input, in_width, &mut expected);
        (sse2.fancy_upsample_h2v1)(&input, in_width, &mut actual);
        assert_eq!(actual, expected, "width={in_width}");
    }
}

#[test]
fn sse2_dispatch_integration() {
    let routines = simd::detect();
    let scalar = scalar_routines();
    // IDCT
    let mut coeffs = [0i16; 64];
    coeffs[0] = 800;
    let quant = [1u16; 64];
    let (mut output, mut expected) = ([0u8; 64], [0u8; 64]);
    (routines.idct_islow)(&coeffs, &quant, &mut output);
    (scalar.idct_islow)(&coeffs, &quant, &mut expected);
    assert_eq!(output, expected);
    // Color
    let width = 32;
    let y: Vec<u8> = (0..width).map(|i| (i * 8) as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| (128 + i) as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| (128 - i) as u8).collect();
    let (mut er, mut ar) = (vec![0u8; width * 3], vec![0u8; width * 3]);
    (scalar.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut er, width);
    (routines.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut ar, width);
    assert_eq!(ar, er);
    // Upsample
    let input: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
    let (mut eu, mut au) = (vec![0u8; 64], vec![0u8; 64]);
    (scalar.fancy_upsample_h2v1)(&input, 32, &mut eu);
    (routines.fancy_upsample_h2v1)(&input, 32, &mut au);
    assert_eq!(au, eu);
}

/// P4-135 (#474): the safe fixed-array wrappers check their own CPU
/// feature and fall back to scalar — "verified at dispatch time" is not a
/// contract a safe function may impose on its caller. Deliberately
/// UNGUARDED by any feature probe: whichever arm the executing host takes
/// must equal the scalar reference, so an AVX2 host exercises the SIMD
/// arms and a host without AVX2 the new fallback arms. Both arms execute
/// in CI: the AVX2 leg's `cargo test --tests` includes this lib binary,
/// and the CPUID-masked no-AVX2 leg (#320,
/// `.github/workflows/cross-arch.yml`) carries `--lib` so the fallback
/// arms run under an emulated Nehalem where
/// `is_x86_feature_detected!("avx2")` is genuinely false — the x86 half
/// of P4-141 criterion 2 for the lib suite.
///
/// Inputs come from the parity suite's generators: coefficients in
/// [-128, 127] and quant in [1, 8], the documented envelope where every
/// IDCT variant is bit-exact against scalar. Outside it they legitimately
/// diverge — scalar truncates its pass-1 workspace to i16, SSE2 keeps
/// i32, AVX2 saturates (the P4-19/P4-20 family) — so a wider generator
/// here would assert a property the kernels never promised.
#[test]
fn feature_checked_wrappers_match_scalar_on_any_cpu() {
    let mut rng = super::simd_parity_tests::Mulberry32::new(0x8676_2d5e);
    let coeffs: [i16; 64] = super::simd_parity_tests::random_coeffs(&mut rng);
    let quant: [u16; 64] = super::simd_parity_tests::random_quant(&mut rng);

    let (mut expected, mut got) = ([0u8; 64], [0u8; 64]);
    crate::simd::scalar::scalar_idct_islow(&coeffs, &quant, &mut expected);
    crate::simd::x86_64::avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut got);
    assert_eq!(got, expected, "avx2_idct_islow diverges from scalar");
    let mut got_sse = [0u8; 64];
    crate::simd::x86_64::idct::sse2_idct_islow(&coeffs, &quant, &mut got_sse);
    assert_eq!(got_sse, expected, "sse2_idct_islow diverges from scalar");

    let divisors = super::simd_parity_tests::build_quant_divisors(quant);
    let mut block = [0i16; 64];
    for (i, slot) in block.iter_mut().enumerate() {
        *slot = coeffs[i] / 2;
    }
    let (mut fdct_expected, mut fdct_got) = ([0i16; 64], [0i16; 64]);
    let mut scalar_in = block;
    crate::simd::scalar::scalar_fdct_quantize(&mut scalar_in, &divisors, &mut fdct_expected);
    let mut simd_in = block;
    crate::simd::x86_64::avx2_fdct_quantize(&mut simd_in, &divisors, &mut fdct_got);
    assert_eq!(
        fdct_got, fdct_expected,
        "avx2_fdct_quantize diverges from scalar"
    );
}
