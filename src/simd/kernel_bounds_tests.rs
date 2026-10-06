//! P4-191 (#609): the precondition edges of SIMD kernels that take a raw
//! destination or that trust a caller-sized slice, driven *directly* so a
//! native sanitizer sees the exact footprint.
//!
//! Miri never interprets anything under `src/simd/` (its `--lib` step skips
//! `simd::` and builds with the `simd` feature off), so for these kernels the
//! only memory checker is the `asan` / `ubsan` pair in
//! `.github/workflows/sanitizers.yml`, which runs every lib unit test on an
//! x86_64 AVX2 runner. Production callers reach these kernels only with
//! comfortably sized planes, which is why the dispatch-level suites never put
//! a footprint flush against the end of an allocation. These tests do: each
//! destination is a heap allocation of *exactly* the documented footprint, so
//! one byte too many is a heap-buffer-overflow under ASan, and a canary-padded
//! twin catches the same defect on legs without a sanitizer (and catches a
//! short-stride write *inside* the footprint, which ASan cannot see).
//!
//! Every module in the codebase whose name ends `kernel_bounds_tests` is
//! selected by the `sanitizers.yml` coverage step; keep the suffix when moving
//! a test.

#![cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]

use crate::simd::scalar;
use crate::simd::simd_parity_tests::{random_coeffs, random_quant, Mulberry32};

/// Row strides for the strided 8x8 IDCT writes: the contiguous minimum, odd
/// strides that misalign every row after the first, a power of two, and a
/// plane-width stride.
const STRIDES: [usize; 7] = [8, 9, 13, 16, 17, 64, 641];

/// Canary bytes. Two of them, because a stray write that happens to store the
/// canary's own value is invisible against that canary.
const CANARIES: [u8; 2] = [0xA5, 0x5A];

/// Bytes of canary in front of and behind the footprint in the padded buffer.
const MARGIN: usize = 32;

/// The signature every strided 8x8 islow IDCT kernel shares.
type StridedIdct = unsafe fn(&[i16; 64], &[u16; 64], *mut u8, usize);

/// Bytes a strided 8x8 write may touch: rows 0..7 start `stride` apart and the
/// last one ends 8 bytes in — the `7 * stride + 8` every kernel documents.
fn strided_footprint(stride: usize) -> usize {
    7 * stride + 8
}

/// Run one strided kernel twice — into an allocation of exactly the footprint
/// and into a canary-padded buffer — and assert the 64 pixels equal the scalar
/// reference while every byte outside them is untouched.
///
/// The exact allocation goes first so that, under ASan, a store past the last
/// row is reported as the heap-buffer-overflow it is rather than pre-empted by
/// the padded buffer's canary assertion; without ASan such a store is invisible
/// there, and the padded buffer's trailing canaries report it instead.
fn assert_strided_idct_matches_scalar(
    kernel_name: &str,
    kernel: StridedIdct,
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    stride: usize,
    case: &str,
) {
    let mut expected: [u8; 64] = [0u8; 64];
    scalar::scalar_idct_islow(coeffs, quant, &mut expected);
    let footprint: usize = strided_footprint(stride);
    // The pixel a footprint offset must hold, or `None` for an inter-row gap.
    let pixel_at = |offset: usize| -> Option<u8> {
        (offset % stride < 8).then(|| expected[(offset / stride) * 8 + offset % stride])
    };

    for canary in CANARIES {
        let mut exact: Vec<u8> = vec![canary; footprint];
        // SAFETY: exactly the documented `7 * stride + 8` writable bytes; the
        // CPU feature was checked by the caller.
        unsafe {
            kernel(coeffs, quant, exact.as_mut_ptr(), stride);
        }
        for (offset, &byte) in exact.iter().enumerate() {
            assert_eq!(
                byte,
                pixel_at(offset).unwrap_or(canary),
                "{kernel_name} stride={stride} {case} canary={canary:#04x}: exact byte {offset}"
            );
        }

        let mut padded: Vec<u8> = vec![canary; MARGIN + footprint + MARGIN];
        // SAFETY: the pointer at `MARGIN` has `footprint + MARGIN` writable
        // bytes behind it, more than the `7 * stride + 8` required.
        unsafe {
            kernel(coeffs, quant, padded.as_mut_ptr().add(MARGIN), stride);
        }
        for (index, &byte) in padded.iter().enumerate() {
            let want: u8 = index
                .checked_sub(MARGIN)
                .filter(|&offset| offset < footprint)
                .and_then(pixel_at)
                .unwrap_or(canary);
            assert_eq!(
                byte, want,
                "{kernel_name} stride={stride} {case} canary={canary:#04x}: padded byte {index}"
            );
        }
    }
}

/// Drive one strided kernel across every stride with random blocks, DC-only
/// blocks (the AVX2 core's pure-DC fill shortcut stores through the stride on
/// its own code path) and the all-zero block.
fn sweep_strided_idct(kernel_name: &str, kernel: StridedIdct, seed: u32) {
    let mut rng: Mulberry32 = Mulberry32::new(seed);
    for stride in STRIDES {
        for iteration in 0..48 {
            let coeffs: [i16; 64] = random_coeffs(&mut rng);
            let quant: [u16; 64] = random_quant(&mut rng);
            let case: String = format!("random#{iteration}");
            assert_strided_idct_matches_scalar(kernel_name, kernel, &coeffs, &quant, stride, &case);
        }
        for dc in [-128i16, -37, 0, 1, 64, 127] {
            let mut coeffs: [i16; 64] = [0i16; 64];
            coeffs[0] = dc;
            let quant: [u16; 64] = [8u16; 64];
            let case: String = format!("dc-only {dc}");
            assert_strided_idct_matches_scalar(kernel_name, kernel, &coeffs, &quant, stride, &case);
        }
    }
}

/// P4-191 criterion 4: the strided islow IDCT writes land in place for every
/// backend this host has, with nothing written outside `7 * stride + 8`.
///
/// `sse2_idct_islow_strided` is otherwise reached only on the emulated Nehalem
/// leg (every other x86_64 runner dispatches to AVX2), and no test passed
/// `neon_` / `avx2_idct_islow_strided` a stride other than the decoder's.
///
/// Discrimination, measured 2026-10-07 on an aarch64 host by mutating the NEON
/// kernel and restoring it:
///
/// * passing `8` instead of `stride` to `neon_idct_islow_core` fails the exact
///   allocation's check at `stride=9 random#0`, byte 8 (an inter-row gap);
/// * widening the DC-only fill's row store from `vst1_u8` to a 16-byte
///   `vst1q_u8` is, under `-Z sanitizer=address`, a heap-buffer-overflow
///   ("WRITE of size 16 … 0 bytes after 64-byte region", `stride=8`), and
///   without a sanitizer fails the padded check at the first trailing canary
///   (`stride=8 dc-only -128`, padded byte 96).
///
/// The SSE2 and AVX2 arms run the same body on the x86_64 sanitizer legs; this
/// host cannot execute x86_64 code, so they were not mutated locally.
#[test]
fn strided_islow_idct_writes_exactly_its_footprint() {
    #[cfg(target_arch = "aarch64")]
    {
        sweep_strided_idct(
            "neon_idct_islow_strided",
            crate::simd::aarch64::idct::neon_idct_islow_strided,
            0xB0B0_0001,
        );
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is part of the x86_64 baseline, so this arm always runs here.
        assert!(
            is_x86_feature_detected!("sse2"),
            "x86_64 without SSE2 is not a target this crate supports"
        );
        sweep_strided_idct(
            "sse2_idct_islow_strided",
            crate::simd::x86_64::idct::sse2_idct_islow_strided,
            0xB0B0_0002,
        );
        if is_x86_feature_detected!("avx2") {
            sweep_strided_idct(
                "avx2_idct_islow_strided",
                crate::simd::x86_64::avx2_idct::avx2_idct_islow_strided,
                0xB0B0_0003,
            );
        } else {
            eprintln!("NOTE: no AVX2 on this host; avx2_idct_islow_strided not exercised");
        }
    }
}

/// Luma widths for the merged H2V2 checks: one full 32-pixel SIMD step, a step
/// plus the scalar tail, odd widths (the extra right-hand column), and a
/// width whose last step ends flush with the row.
#[cfg(target_arch = "x86_64")]
const MERGED_WIDTHS: [usize; 6] = [32, 33, 47, 63, 64, 97];

/// Random rows sized *exactly* for `width`, so the last SIMD load or store
/// ends at the allocation's end.
#[cfg(target_arch = "x86_64")]
fn merged_h2v2_rows(rng: &mut Mulberry32, width: usize) -> [Vec<u8>; 4] {
    use crate::simd::simd_parity_tests::random_plane_u8;

    let chroma: usize = width.div_ceil(2);
    [
        random_plane_u8(rng, width),
        random_plane_u8(rng, width),
        random_plane_u8(rng, chroma),
        random_plane_u8(rng, chroma),
    ]
}

/// P4-191 criterion 5: `avx2_merged_h2v2_ycbcr_to_rgb` with exactly sized
/// rows, including the odd widths `parity_merged_upsample_h2v2` (even widths
/// only) never feeds, matches the scalar reference byte for byte.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_merged_h2v2_exact_rows_match_scalar() {
    use crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb;
    use crate::simd::x86_64::avx2_merged::avx2_merged_h2v2_ycbcr_to_rgb;

    if !is_x86_feature_detected!("avx2") {
        eprintln!("NOTE: no AVX2 on this host; the wrapper takes the scalar arm");
    }
    let mut rng: Mulberry32 = Mulberry32::new(0xB0B0_0010);
    for width in MERGED_WIDTHS {
        for _ in 0..8 {
            let [y0, y1, cb, cr] = merged_h2v2_rows(&mut rng, width);
            let mut want0: Vec<u8> = vec![0u8; width * 3];
            let mut want1: Vec<u8> = vec![0u8; width * 3];
            merged_h2v2_ycbcr_to_rgb(&y0, &y1, &cb, &cr, &mut want0, &mut want1, width);
            let mut got0: Vec<u8> = vec![0u8; width * 3];
            let mut got1: Vec<u8> = vec![0u8; width * 3];
            avx2_merged_h2v2_ycbcr_to_rgb(&y0, &y1, &cb, &cr, &mut got0, &mut got1, width);
            assert_eq!(got0, want0, "row 0 at width {width}");
            assert_eq!(got1, want1, "row 1 at width {width}");
        }
    }
}

/// P4-191 criterion 5: a second luma row shorter than `width` must not reach
/// the AVX2 kernel, which reads 32 luma bytes per step from it by raw pointer.
///
/// The wrapper used to check `y_row0`, `cb_row`, `cr_row` and `rgb_out0` only.
/// With `width = 32` and a 16-byte `y_row1`, that sends one full SIMD step
/// reading 16 bytes past the allocation and leaves no scalar tail to index, so
/// the unchecked call returns normally — this test fails for want of a panic,
/// and under ASan the read is a heap-buffer-overflow. (Derived from the kernel's
/// loop bounds; the x86_64 legs are where it executes, not the aarch64 host
/// this was written on.) Checked, the call takes the scalar arm, whose
/// bounds-checked indexing panics: the safe outcome pinned here.
#[cfg(target_arch = "x86_64")]
#[test]
#[should_panic(expected = "index out of bounds")]
fn avx2_merged_h2v2_short_second_luma_row_is_refused() {
    use crate::simd::x86_64::avx2_merged::avx2_merged_h2v2_ycbcr_to_rgb;

    if !is_x86_feature_detected!("avx2") {
        eprintln!("NOTE: no AVX2 on this host; the panic below proves only the scalar arm");
    }
    let width: usize = 32;
    let y0: Vec<u8> = vec![120u8; width];
    let y1_short: Vec<u8> = vec![120u8; width / 2];
    let cb: Vec<u8> = vec![90u8; width / 2];
    let cr: Vec<u8> = vec![170u8; width / 2];
    let mut rgb0: Vec<u8> = vec![0u8; width * 3];
    let mut rgb1: Vec<u8> = vec![0u8; width * 3];
    avx2_merged_h2v2_ycbcr_to_rgb(&y0, &y1_short, &cb, &cr, &mut rgb0, &mut rgb1, width);
}

/// P4-191 criterion 5, the write side: a second output row shorter than
/// `width * 3` must not reach the kernel, which stores 96 bytes per step into
/// it by raw pointer. Unchecked, `width = 32` with a 48-byte `rgb_out1` writes
/// 48 bytes past the allocation and returns normally — a heap corruption ASan
/// reports and an unsanitized run may not notice at all.
#[cfg(target_arch = "x86_64")]
#[test]
#[should_panic(expected = "index out of bounds")]
fn avx2_merged_h2v2_short_second_output_row_is_refused() {
    use crate::simd::x86_64::avx2_merged::avx2_merged_h2v2_ycbcr_to_rgb;

    if !is_x86_feature_detected!("avx2") {
        eprintln!("NOTE: no AVX2 on this host; the panic below proves only the scalar arm");
    }
    let width: usize = 32;
    let y0: Vec<u8> = vec![120u8; width];
    let y1: Vec<u8> = vec![120u8; width];
    let cb: Vec<u8> = vec![90u8; width / 2];
    let cr: Vec<u8> = vec![170u8; width / 2];
    let mut rgb0: Vec<u8> = vec![0u8; width * 3];
    let mut rgb1_short: Vec<u8> = vec![0u8; width * 3 / 2];
    avx2_merged_h2v2_ycbcr_to_rgb(&y0, &y1, &cb, &cr, &mut rgb0, &mut rgb1_short, width);
}
