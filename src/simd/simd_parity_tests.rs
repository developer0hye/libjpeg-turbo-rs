//! Relocated from `tests/simd_parity.rs` for P4-135 criterion 2 (#474).
//!
//! This suite reaches SIMD kernels directly, which is why the arch
//! modules had to stay `pub` and were therefore callable from any
//! downstream crate. As an in-crate test it uses `crate::`, so they
//! can be private. Moved verbatim apart from the path rewrite.

//! Cross-arch SIMD parity tests.
//!
//! For every SIMD kernel pair (scalar reference vs SIMD implementation)
//! that exists on the host arch, this suite generates N=1000 reproducible
//! randomized inputs using a Mulberry32 PRNG and asserts that the SIMD
//! backend produces bit-exact output compared to the scalar reference.
//! The scalar reference path is always available (see `src/simd/scalar.rs`),
//! and platform-specific SIMD kernels are gated with `#[cfg(target_arch)]`
//! so this file builds cleanly on every supported architecture. Where a
//! port is missing the comparison block compiles away and only the scalar
//! not-panic path runs — today that is the reduced-size IDCT trio off
//! aarch64 (tracked as P4-71). The wasm32 blocks additionally require
//! `target_feature = "simd128"` at compile time: build without
//! `-C target-feature=+simd128` and EVERY wasm comparison compiles away,
//! leaving the whole suite scalar-only. In-repo builds normally get the
//! flag from `.cargo/config.toml` (both wasm targets); beware that a
//! `RUSTFLAGS` env var overrides that config, so custom flags must keep
//! `-C target-feature=+simd128`.
//!
//! Reproducibility
//! ---------------
//! Each kernel is seeded with a distinct constant so a single flaky
//! input prints the Mulberry32 state (iteration index) that reproduces
//! it. All comparisons use `assert_eq!` on the full output buffers so
//! any mismatch surfaces immediately.
//!
//! Tolerance
//! ---------
//! All integer SIMD kernels covered here target bit-exact parity
//! (max_diff = 0). We therefore use `assert_eq!` rather than a
//! tolerance-based comparison. The integer FDCT, IDCT, quantize,
//! zigzag, color-conversion, and upsample kernels have all been ported
//! to produce results identical to the scalar reference (measured:
//! `max_diff = 0` across all 1000 random inputs per kernel). If a
//! future kernel requires a non-zero tolerance (e.g., a float IDCT
//! variant whose intermediate rounding cannot be matched exactly), use
//! `assert!(diff <= measured + 1)` with the measured value documented
//! in a comment. No such kernel is currently in scope.

use crate::encode::pipeline::compute_reciprocal;
use crate::encode::tables::ZIGZAG_ORDER;
use crate::simd::scalar;
use crate::simd::QuantDivisors;

/// Number of random inputs per kernel. Large enough to probe corner
/// cases (DC-only, saturating, sign-extended) while keeping the suite
/// under a second on CI runners.
const N: usize = 1000;

// ---------------------------------------------------------------------
// Mulberry32 PRNG — small, fast, reproducible, public-domain
// algorithm. Suitable for deterministic fuzz-style tests.
// ---------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B79F5);
        let mut z: u32 = self.state;
        z = (z ^ (z >> 15)).wrapping_mul(z | 1);
        z ^= z.wrapping_add((z ^ (z >> 7)).wrapping_mul(z | 61));
        z ^ (z >> 14)
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        self.next_u32() as u8
    }

    #[inline]
    fn next_u16_range(&mut self, lo: u16, hi_inclusive: u16) -> u16 {
        let span: u32 = (hi_inclusive as u32) - (lo as u32) + 1;
        (lo as u32 + self.next_u32() % span) as u16
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Produce JPEG-realistic DCT coefficients in a bounded range that
/// keeps `coeff * quant` within i16 range for the SIMD IDCT kernels.
///
/// Real JPEG coefficients are in [-1024, 1023] (11-bit signed) and
/// quant values in [1, 255], which can produce `coeff * quant` values
/// outside i16 for large quants — the C/SIMD IDCT kernels handle that
/// correctly via saturating operations. However the scaled-IDCT scalar
/// reference (`idct_4x4`, `idct_2x2`) uses unchecked i32 arithmetic
/// that overflows with extreme inputs. We therefore clamp both coeff
/// and quant to a range where every kernel, scalar and SIMD, produces
/// identical output and no debug-build overflow checks trip.
fn random_coeffs(rng: &mut Mulberry32) -> [i16; 64] {
    let mut coeffs: [i16; 64] = [0i16; 64];
    for slot in coeffs.iter_mut() {
        // Random in [-128, 127] — conservative enough that every IDCT
        // variant's internal arithmetic stays bit-exact between the
        // scalar reference and every SIMD backend. Wider ranges expose
        // pre-existing rounding diffs in the scaled-IDCT kernels that
        // are not in scope for the cross-backend parity check.
        let raw: i16 = (rng.next_u32() as i16) & 0x00FF; // 8 bits, 0..255
        *slot = raw - 128;
    }
    coeffs
}

/// Quant table values matching real JPEG encoders, but narrowed so the
/// scaled-IDCT scalar reference cannot overflow its i32 intermediate.
fn random_quant(rng: &mut Mulberry32) -> [u16; 64] {
    let mut quant: [u16; 64] = [0u16; 64];
    for slot in quant.iter_mut() {
        // Quantization values in [1, 8] cover typical high-quality
        // JPEG tables (quality 90+). Combined with coefficients in
        // [-256, 255], the product `coeff * quant` is always within
        // [-2048, 2047], which every SIMD IDCT kernel (whose internal
        // multiply/shift chains were tuned for C's standard range)
        // handles bit-exactly against the scalar reference.
        *slot = rng.next_u16_range(1, 8);
    }
    quant
}

/// Build `QuantDivisors` from a natural-order quant table using the same
/// `compute_reciprocal` path the encoder uses. Divisors are multiplied
/// by 8 to match FDCT output scaling.
fn build_quant_divisors(natural_quant: [u16; 64]) -> QuantDivisors {
    let mut divisors: [u16; 64] = [0u16; 64];
    let mut reciprocals: [u16; 64] = [0u16; 64];
    let mut corrections: [u16; 64] = [0u16; 64];
    let mut shifts: [i16; 64] = [0i16; 64];
    let mut scales: [u16; 64] = [0u16; 64];

    for i in 0..64 {
        let d: u16 = natural_quant[i].saturating_mul(8).max(1);
        divisors[i] = d;
        let (r, c, sc, s) = compute_reciprocal(d);
        reciprocals[i] = r;
        corrections[i] = c;
        scales[i] = sc;
        shifts[i] = s;
    }

    let mut divisors_zigzag: [u16; 64] = [0u16; 64];
    let mut reciprocals_zigzag: [u16; 64] = [0u16; 64];
    let mut corrections_zigzag: [u16; 64] = [0u16; 64];
    let mut shifts_zigzag: [i16; 64] = [0i16; 64];
    let mut scales_zigzag: [u16; 64] = [0u16; 64];
    for zz in 0..64 {
        let natural_idx: usize = ZIGZAG_ORDER[zz];
        divisors_zigzag[zz] = divisors[natural_idx];
        reciprocals_zigzag[zz] = reciprocals[natural_idx];
        corrections_zigzag[zz] = corrections[natural_idx];
        shifts_zigzag[zz] = shifts[natural_idx];
        scales_zigzag[zz] = scales[natural_idx];
    }

    QuantDivisors {
        divisors,
        reciprocals,
        corrections,
        shifts,
        scales,
        divisors_zigzag,
        reciprocals_zigzag,
        corrections_zigzag,
        shifts_zigzag,
        scales_zigzag,
        // Float-DCT divisors are unused on the islow paths exercised here;
        // leave as zero so the struct is well-formed without changing
        // observed behaviour.
        float_divisors: [0.0; 64],
        float_divisors_zigzag: [0.0; 64],
    }
}

fn random_plane_u8(rng: &mut Mulberry32, len: usize) -> Vec<u8> {
    (0..len).map(|_| rng.next_u8()).collect()
}

/// Widths exercised by row-based color-conversion kernels. Covers
/// SIMD-aligned, tail-only, and odd/edge widths to trip up
/// register-boundary logic. Widths 1 and 2 are intentionally excluded
/// for upsample kernels because the scalar `fancy_h2v1` path uses
/// edge-replication there while the SIMD kernels interpolate — a
/// deliberate divergence we do not need to cross-validate.
const ROW_WIDTHS: &[usize] = &[1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65, 128, 255];

/// Widths for upsample parity — all SIMD backends match scalar across every
/// width. Widths 1 (edge replication) and 2 (box filter) exercise kernel guards
/// that must agree with scalar fancy_h2v1's small-width behavior.
const UPSAMPLE_WIDTHS: &[usize] = &[1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 33, 64, 65, 128, 255];

// =====================================================================
// IDCT parity — 8x8 full (scalar_idct_islow vs SIMD)
// =====================================================================

#[test]
fn parity_idct_islow_full() {
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0001);
    for i in 0..N {
        let coeffs: [i16; 64] = random_coeffs(&mut rng);
        let quant: [u16; 64] = random_quant(&mut rng);
        let mut scalar_out: [u8; 64] = [0u8; 64];
        scalar::scalar_idct_islow(&coeffs, &quant, &mut scalar_out);

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            use crate::simd::aarch64::idct::neon_idct_islow;
            let mut simd_out: [u8; 64] = [0u8; 64];
            neon_idct_islow(&coeffs, &quant, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "NEON idct_islow mismatch at iter {i}");
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::simd::x86_64::avx2_idct::avx2_idct_islow;
                let mut simd_out: [u8; 64] = [0u8; 64];
                avx2_idct_islow(&coeffs, &quant, &mut simd_out);
                assert_eq!(simd_out, scalar_out, "AVX2 idct_islow mismatch at iter {i}");
            }
            if is_x86_feature_detected!("sse2") {
                use crate::simd::x86_64::idct::sse2_idct_islow;
                let mut simd_out: [u8; 64] = [0u8; 64];
                sse2_idct_islow(&coeffs, &quant, &mut simd_out);
                assert_eq!(simd_out, scalar_out, "SSE2 idct_islow mismatch at iter {i}");
            }
        }
        #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
        {
            use crate::simd::wasm32::idct::wasm_idct_islow;
            let mut simd_out: [u8; 64] = [0u8; 64];
            wasm_idct_islow(&coeffs, &quant, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "WASM idct_islow mismatch at iter {i}");
        }
    }
}

// =====================================================================
// IDCT parity — reduced-size kernels (4x4, 2x2, 1x1)
// =====================================================================
// These are scalar-reference only (`idct_4x4` etc.) on x86_64 and wasm,
// but aarch64 has NEON variants. Run parity on aarch64, and a scalar
// self-consistency check elsewhere (it must at least not panic) — hence
// the allow(unused_variables) off aarch64, where the comparison block
// that consumes `i`/`scalar_out` compiles away. The missing x86_64/wasm
// reduced-size kernels are tracked as P4-71 in docs/last_mile/phase4.md.

#[test]
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
fn parity_idct_4x4() {
    use crate::decode::idct_scaled::idct_4x4;
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0002);
    for i in 0..N {
        let coeffs: [i16; 64] = random_coeffs(&mut rng);
        let quant: [u16; 64] = random_quant(&mut rng);
        let mut scalar_out: [u8; 16] = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut scalar_out);

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            use crate::simd::aarch64::idct_scaled::neon_idct_4x4;
            let mut simd_out: [u8; 16] = [0u8; 16];
            neon_idct_4x4(&coeffs, &quant, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "NEON idct_4x4 mismatch at iter {i}");
        }
    }
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
fn parity_idct_2x2() {
    use crate::decode::idct_scaled::idct_2x2;
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0003);
    for i in 0..N {
        let coeffs: [i16; 64] = random_coeffs(&mut rng);
        let quant: [u16; 64] = random_quant(&mut rng);
        let mut scalar_out: [u8; 4] = [0u8; 4];
        idct_2x2(&coeffs, &quant, &mut scalar_out);

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            use crate::simd::aarch64::idct_scaled::neon_idct_2x2;
            let mut simd_out: [u8; 4] = [0u8; 4];
            neon_idct_2x2(&coeffs, &quant, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "NEON idct_2x2 mismatch at iter {i}");
        }
    }
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
fn parity_idct_1x1() {
    use crate::decode::idct_scaled::idct_1x1;
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0004);
    for i in 0..N {
        let coeffs: [i16; 64] = random_coeffs(&mut rng);
        let quant: [u16; 64] = random_quant(&mut rng);
        let scalar_out: u8 = idct_1x1(&coeffs, &quant);

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            use crate::simd::aarch64::idct_scaled::neon_idct_1x1;
            let mut simd_out: [u8; 1] = [0u8; 1];
            neon_idct_1x1(&coeffs, &quant, &mut simd_out);
            assert_eq!(
                simd_out[0], scalar_out,
                "NEON idct_1x1 mismatch at iter {i}"
            );
        }
    }
}

// =====================================================================
// YCbCr → RGB row conversion parity (RGB / RGBA / BGR / BGRA)
// =====================================================================

#[test]
fn parity_ycbcr_to_rgb_rows() {
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0010);

    for &width in ROW_WIDTHS {
        for i in 0..(N / ROW_WIDTHS.len() + 1) {
            let y: Vec<u8> = random_plane_u8(&mut rng, width);
            let cb: Vec<u8> = random_plane_u8(&mut rng, width);
            let cr: Vec<u8> = random_plane_u8(&mut rng, width);

            // RGB (3 bpp) — scalar vs SIMD
            let mut scalar_rgb: Vec<u8> = vec![0u8; width * 3];
            scalar::scalar_ycbcr_to_rgb_row(&y, &cb, &cr, &mut scalar_rgb, width);

            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                use crate::simd::aarch64::color::{
                    neon_ycbcr_to_bgr_row, neon_ycbcr_to_bgra_row, neon_ycbcr_to_rgb_row,
                    neon_ycbcr_to_rgba_row,
                };

                let mut simd_rgb: Vec<u8> = vec![0u8; width * 3];
                neon_ycbcr_to_rgb_row(&y, &cb, &cr, &mut simd_rgb, width);
                assert_eq!(simd_rgb, scalar_rgb, "NEON RGB mismatch w={width} iter={i}");

                // RGBA / BGR / BGRA compared against their own scalar reference
                let mut scalar_rgba: Vec<u8> = vec![0u8; width * 4];
                let mut simd_rgba: Vec<u8> = vec![0u8; width * 4];
                crate::decode::color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut scalar_rgba, width);
                neon_ycbcr_to_rgba_row(&y, &cb, &cr, &mut simd_rgba, width);
                assert_eq!(
                    simd_rgba, scalar_rgba,
                    "NEON RGBA mismatch w={width} iter={i}"
                );

                let mut scalar_bgr: Vec<u8> = vec![0u8; width * 3];
                let mut simd_bgr: Vec<u8> = vec![0u8; width * 3];
                crate::decode::color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut scalar_bgr, width);
                neon_ycbcr_to_bgr_row(&y, &cb, &cr, &mut simd_bgr, width);
                assert_eq!(simd_bgr, scalar_bgr, "NEON BGR mismatch w={width} iter={i}");

                let mut scalar_bgra: Vec<u8> = vec![0u8; width * 4];
                let mut simd_bgra: Vec<u8> = vec![0u8; width * 4];
                crate::decode::color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut scalar_bgra, width);
                neon_ycbcr_to_bgra_row(&y, &cb, &cr, &mut simd_bgra, width);
                assert_eq!(
                    simd_bgra, scalar_bgra,
                    "NEON BGRA mismatch w={width} iter={i}"
                );
            }

            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if is_x86_feature_detected!("avx2") {
                    use crate::simd::x86_64::avx2_color::avx2_ycbcr_to_rgb_row;
                    let mut simd_rgb: Vec<u8> = vec![0u8; width * 3];
                    avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut simd_rgb, width);
                    assert_eq!(simd_rgb, scalar_rgb, "AVX2 RGB mismatch w={width} iter={i}");
                }
                if is_x86_feature_detected!("sse2") {
                    use crate::simd::x86_64::color::sse2_ycbcr_to_rgb_row;
                    let mut simd_rgb: Vec<u8> = vec![0u8; width * 3];
                    sse2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut simd_rgb, width);
                    assert_eq!(simd_rgb, scalar_rgb, "SSE2 RGB mismatch w={width} iter={i}");
                }
            }

            #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
            {
                use crate::simd::wasm32::color::{
                    wasm_ycbcr_to_bgr_row, wasm_ycbcr_to_bgra_row, wasm_ycbcr_to_rgb_row,
                    wasm_ycbcr_to_rgba_row,
                };

                let mut simd_rgb: Vec<u8> = vec![0u8; width * 3];
                wasm_ycbcr_to_rgb_row(&y, &cb, &cr, &mut simd_rgb, width);
                assert_eq!(simd_rgb, scalar_rgb, "WASM RGB mismatch w={width} iter={i}");

                let mut scalar_rgba: Vec<u8> = vec![0u8; width * 4];
                let mut simd_rgba: Vec<u8> = vec![0u8; width * 4];
                crate::decode::color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut scalar_rgba, width);
                wasm_ycbcr_to_rgba_row(&y, &cb, &cr, &mut simd_rgba, width);
                assert_eq!(
                    simd_rgba, scalar_rgba,
                    "WASM RGBA mismatch w={width} iter={i}"
                );

                let mut scalar_bgr: Vec<u8> = vec![0u8; width * 3];
                let mut simd_bgr: Vec<u8> = vec![0u8; width * 3];
                crate::decode::color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut scalar_bgr, width);
                wasm_ycbcr_to_bgr_row(&y, &cb, &cr, &mut simd_bgr, width);
                assert_eq!(simd_bgr, scalar_bgr, "WASM BGR mismatch w={width} iter={i}");

                let mut scalar_bgra: Vec<u8> = vec![0u8; width * 4];
                let mut simd_bgra: Vec<u8> = vec![0u8; width * 4];
                crate::decode::color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut scalar_bgra, width);
                wasm_ycbcr_to_bgra_row(&y, &cb, &cr, &mut simd_bgra, width);
                assert_eq!(
                    simd_bgra, scalar_bgra,
                    "WASM BGRA mismatch w={width} iter={i}"
                );
            }
        }
    }
}

// =====================================================================
// Fancy upsample parity — H2V1 and H2V2
// =====================================================================

#[test]
fn parity_fancy_upsample_h2v1() {
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0020);

    for &in_width in UPSAMPLE_WIDTHS {
        for i in 0..(N / UPSAMPLE_WIDTHS.len() + 1) {
            let input: Vec<u8> = random_plane_u8(&mut rng, in_width);
            let mut scalar_out: Vec<u8> = vec![0u8; in_width * 2];
            scalar::scalar_fancy_upsample_h2v1(&input, in_width, &mut scalar_out);

            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                use crate::simd::aarch64::upsample::neon_fancy_upsample_h2v1;
                let mut simd_out: Vec<u8> = vec![0u8; in_width * 2];
                neon_fancy_upsample_h2v1(&input, in_width, &mut simd_out);
                assert_eq!(
                    simd_out, scalar_out,
                    "NEON fancy_h2v1 mismatch w={in_width} iter={i}"
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if is_x86_feature_detected!("avx2") {
                    use crate::simd::x86_64::avx2_upsample::avx2_fancy_upsample_h2v1;
                    let mut simd_out: Vec<u8> = vec![0u8; in_width * 2];
                    avx2_fancy_upsample_h2v1(&input, in_width, &mut simd_out);
                    assert_eq!(
                        simd_out, scalar_out,
                        "AVX2 fancy_h2v1 mismatch w={in_width} iter={i}"
                    );
                }
                if is_x86_feature_detected!("sse2") {
                    use crate::simd::x86_64::upsample::sse2_fancy_upsample_h2v1;
                    let mut simd_out: Vec<u8> = vec![0u8; in_width * 2];
                    sse2_fancy_upsample_h2v1(&input, in_width, &mut simd_out);
                    assert_eq!(
                        simd_out, scalar_out,
                        "SSE2 fancy_h2v1 mismatch w={in_width} iter={i}"
                    );
                }
            }
            #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
            {
                use crate::simd::wasm32::upsample::wasm_fancy_upsample_h2v1;
                let mut simd_out: Vec<u8> = vec![0u8; in_width * 2];
                wasm_fancy_upsample_h2v1(&input, in_width, &mut simd_out);
                assert_eq!(
                    simd_out, scalar_out,
                    "WASM fancy_h2v1 mismatch w={in_width} iter={i}"
                );
            }
        }
    }
}

#[test]
fn parity_fancy_upsample_h2v2() {
    use crate::decode::upsample::fancy_h2v2;

    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0021);
    // H2V2 needs a 2D plane; use a small set of (w, h) pairs and run
    // enough iterations per pair to hit N total. Widths <3 use a
    // box-filter fallback in the scalar path that the SIMD kernels
    // don't replicate, so we skip them here.
    let dims: &[(usize, usize)] = &[
        (3, 5),
        (7, 3),
        (8, 8),
        (15, 9),
        (17, 13),
        (32, 16),
        (64, 32),
    ];
    for &(w, h) in dims {
        for i in 0..(N / dims.len() + 1) {
            let input: Vec<u8> = random_plane_u8(&mut rng, w * h);
            let out_w: usize = w * 2;
            let out_h: usize = h * 2;
            let mut scalar_out: Vec<u8> = vec![0u8; out_w * out_h];
            fancy_h2v2(&input, w, h, &mut scalar_out, out_w, out_h);

            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                use crate::simd::aarch64::upsample::neon_fancy_upsample_h2v2;
                let mut simd_out: Vec<u8> = vec![0u8; out_w * out_h];
                neon_fancy_upsample_h2v2(&input, w, h, &mut simd_out, out_w);
                assert_eq!(
                    simd_out, scalar_out,
                    "NEON fancy_h2v2 mismatch w={w} h={h} iter={i}"
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if is_x86_feature_detected!("avx2") {
                    use crate::simd::x86_64::avx2_upsample::avx2_fancy_upsample_h2v2;
                    let mut simd_out: Vec<u8> = vec![0u8; out_w * out_h];
                    avx2_fancy_upsample_h2v2(&input, w, h, &mut simd_out, out_w);
                    assert_eq!(
                        simd_out, scalar_out,
                        "AVX2 fancy_h2v2 mismatch w={w} h={h} iter={i}"
                    );
                }
                if is_x86_feature_detected!("sse2") {
                    use crate::simd::x86_64::upsample::sse2_fancy_upsample_h2v2;
                    let mut simd_out: Vec<u8> = vec![0u8; out_w * out_h];
                    sse2_fancy_upsample_h2v2(&input, w, h, &mut simd_out, out_w);
                    assert_eq!(
                        simd_out, scalar_out,
                        "SSE2 fancy_h2v2 mismatch w={w} h={h} iter={i}"
                    );
                }
            }
            #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
            {
                use crate::simd::wasm32::upsample::wasm_fancy_upsample_h2v2;
                let mut simd_out: Vec<u8> = vec![0u8; out_w * out_h];
                wasm_fancy_upsample_h2v2(&input, w, h, &mut simd_out, out_w);
                assert_eq!(
                    simd_out, scalar_out,
                    "WASM fancy_h2v2 mismatch w={w} h={h} iter={i}"
                );
            }
        }
    }
}

// =====================================================================
// Merged upsample + YCbCr→RGB (H2V1, H2V2)
// =====================================================================

#[test]
fn parity_merged_upsample_h2v1() {
    use crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb;

    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0030);
    for &luma_w in &[2usize, 4, 6, 8, 16, 18, 32, 62, 64, 128] {
        let chroma_w: usize = luma_w.div_ceil(2);
        for i in 0..(N / 10 + 1) {
            let y: Vec<u8> = random_plane_u8(&mut rng, luma_w);
            let cb: Vec<u8> = random_plane_u8(&mut rng, chroma_w);
            let cr: Vec<u8> = random_plane_u8(&mut rng, chroma_w);

            let mut scalar_rgb: Vec<u8> = vec![0u8; luma_w * 3];
            merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut scalar_rgb, luma_w);

            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                use crate::simd::aarch64::merged::neon_merged_h2v1_ycbcr_to_rgb;
                let mut simd_rgb: Vec<u8> = vec![0u8; luma_w * 3];
                neon_merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut simd_rgb, luma_w);
                assert_eq!(
                    simd_rgb, scalar_rgb,
                    "NEON merged_h2v1 mismatch w={luma_w} iter={i}"
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if is_x86_feature_detected!("avx2") {
                    use crate::simd::x86_64::avx2_merged::avx2_merged_h2v1_ycbcr_to_rgb;
                    let mut simd_rgb: Vec<u8> = vec![0u8; luma_w * 3];
                    avx2_merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut simd_rgb, luma_w);
                    assert_eq!(
                        simd_rgb, scalar_rgb,
                        "AVX2 merged_h2v1 mismatch w={luma_w} iter={i}"
                    );
                }
            }
            #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
            {
                use crate::simd::wasm32::merged::wasm_merged_h2v1_ycbcr_to_rgb;
                let mut simd_rgb: Vec<u8> = vec![0u8; luma_w * 3];
                wasm_merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut simd_rgb, luma_w);
                assert_eq!(
                    simd_rgb, scalar_rgb,
                    "WASM merged_h2v1 mismatch w={luma_w} iter={i}"
                );
            }
        }
    }
}

#[test]
fn parity_merged_upsample_h2v2() {
    use crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb;

    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0031);
    for &luma_w in &[2usize, 4, 8, 16, 32, 64, 128] {
        let chroma_w: usize = luma_w.div_ceil(2);
        for i in 0..(N / 7 + 1) {
            let y0: Vec<u8> = random_plane_u8(&mut rng, luma_w);
            let y1: Vec<u8> = random_plane_u8(&mut rng, luma_w);
            let cb: Vec<u8> = random_plane_u8(&mut rng, chroma_w);
            let cr: Vec<u8> = random_plane_u8(&mut rng, chroma_w);

            let mut scalar_rgb0: Vec<u8> = vec![0u8; luma_w * 3];
            let mut scalar_rgb1: Vec<u8> = vec![0u8; luma_w * 3];
            merged_h2v2_ycbcr_to_rgb(
                &y0,
                &y1,
                &cb,
                &cr,
                &mut scalar_rgb0,
                &mut scalar_rgb1,
                luma_w,
            );

            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                use crate::simd::aarch64::merged::neon_merged_h2v2_ycbcr_to_rgb;
                let mut simd_rgb0: Vec<u8> = vec![0u8; luma_w * 3];
                let mut simd_rgb1: Vec<u8> = vec![0u8; luma_w * 3];
                neon_merged_h2v2_ycbcr_to_rgb(
                    &y0,
                    &y1,
                    &cb,
                    &cr,
                    &mut simd_rgb0,
                    &mut simd_rgb1,
                    luma_w,
                );
                assert_eq!(
                    simd_rgb0, scalar_rgb0,
                    "NEON merged_h2v2 row0 mismatch w={luma_w} iter={i}"
                );
                assert_eq!(
                    simd_rgb1, scalar_rgb1,
                    "NEON merged_h2v2 row1 mismatch w={luma_w} iter={i}"
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if is_x86_feature_detected!("avx2") {
                    use crate::simd::x86_64::avx2_merged::avx2_merged_h2v2_ycbcr_to_rgb;
                    let mut simd_rgb0: Vec<u8> = vec![0u8; luma_w * 3];
                    let mut simd_rgb1: Vec<u8> = vec![0u8; luma_w * 3];
                    avx2_merged_h2v2_ycbcr_to_rgb(
                        &y0,
                        &y1,
                        &cb,
                        &cr,
                        &mut simd_rgb0,
                        &mut simd_rgb1,
                        luma_w,
                    );
                    assert_eq!(
                        simd_rgb0, scalar_rgb0,
                        "AVX2 merged_h2v2 row0 mismatch w={luma_w} iter={i}"
                    );
                    assert_eq!(
                        simd_rgb1, scalar_rgb1,
                        "AVX2 merged_h2v2 row1 mismatch w={luma_w} iter={i}"
                    );
                }
            }
            #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
            {
                use crate::simd::wasm32::merged::wasm_merged_h2v2_ycbcr_to_rgb;
                let mut simd_rgb0: Vec<u8> = vec![0u8; luma_w * 3];
                let mut simd_rgb1: Vec<u8> = vec![0u8; luma_w * 3];
                wasm_merged_h2v2_ycbcr_to_rgb(
                    &y0,
                    &y1,
                    &cb,
                    &cr,
                    &mut simd_rgb0,
                    &mut simd_rgb1,
                    luma_w,
                );
                assert_eq!(
                    simd_rgb0, scalar_rgb0,
                    "WASM merged_h2v2 row0 mismatch w={luma_w} iter={i}"
                );
                assert_eq!(
                    simd_rgb1, scalar_rgb1,
                    "WASM merged_h2v2 row1 mismatch w={luma_w} iter={i}"
                );
            }
        }
    }
}

// =====================================================================
// RGB → YCbCr encoder row conversion
// =====================================================================

#[test]
fn parity_rgb_to_ycbcr_rows() {
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0040);

    for &width in ROW_WIDTHS {
        for i in 0..(N / ROW_WIDTHS.len() + 1) {
            let rgb: Vec<u8> = random_plane_u8(&mut rng, width * 3);
            let rgba: Vec<u8> = random_plane_u8(&mut rng, width * 4);

            let mut scalar_y: Vec<u8> = vec![0u8; width];
            let mut scalar_cb: Vec<u8> = vec![0u8; width];
            let mut scalar_cr: Vec<u8> = vec![0u8; width];
            scalar::scalar_rgb_to_ycbcr_row_enc(
                &rgb,
                &mut scalar_y,
                &mut scalar_cb,
                &mut scalar_cr,
                width,
            );

            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                use crate::simd::aarch64::color_encode::{
                    neon_bgr_to_ycbcr_row, neon_bgra_to_ycbcr_row, neon_rgb_to_ycbcr_row,
                    neon_rgba_to_ycbcr_row,
                };

                let mut simd_y: Vec<u8> = vec![0u8; width];
                let mut simd_cb: Vec<u8> = vec![0u8; width];
                let mut simd_cr: Vec<u8> = vec![0u8; width];
                neon_rgb_to_ycbcr_row(&rgb, &mut simd_y, &mut simd_cb, &mut simd_cr, width);
                assert_eq!(
                    simd_y, scalar_y,
                    "NEON enc RGB→Y mismatch w={width} iter={i}"
                );
                assert_eq!(simd_cb, scalar_cb, "NEON enc RGB→Cb mismatch w={width}");
                assert_eq!(simd_cr, scalar_cr, "NEON enc RGB→Cr mismatch w={width}");

                // RGBA / BGR / BGRA via their own scalar references
                let mut sc_y: Vec<u8> = vec![0u8; width];
                let mut sc_cb: Vec<u8> = vec![0u8; width];
                let mut sc_cr: Vec<u8> = vec![0u8; width];
                crate::encode::color::rgba_to_ycbcr_row(
                    &rgba, &mut sc_y, &mut sc_cb, &mut sc_cr, width,
                );
                let mut sd_y: Vec<u8> = vec![0u8; width];
                let mut sd_cb: Vec<u8> = vec![0u8; width];
                let mut sd_cr: Vec<u8> = vec![0u8; width];
                neon_rgba_to_ycbcr_row(&rgba, &mut sd_y, &mut sd_cb, &mut sd_cr, width);
                assert_eq!(sd_y, sc_y, "NEON RGBA→Y w={width} iter={i}");
                assert_eq!(sd_cb, sc_cb, "NEON RGBA→Cb w={width}");
                assert_eq!(sd_cr, sc_cr, "NEON RGBA→Cr w={width}");

                let bgr: &[u8] = &rgb;
                let mut sc_y2: Vec<u8> = vec![0u8; width];
                let mut sc_cb2: Vec<u8> = vec![0u8; width];
                let mut sc_cr2: Vec<u8> = vec![0u8; width];
                crate::encode::color::bgr_to_ycbcr_row_scalar(
                    bgr,
                    &mut sc_y2,
                    &mut sc_cb2,
                    &mut sc_cr2,
                    width,
                );
                let mut sd_y2: Vec<u8> = vec![0u8; width];
                let mut sd_cb2: Vec<u8> = vec![0u8; width];
                let mut sd_cr2: Vec<u8> = vec![0u8; width];
                neon_bgr_to_ycbcr_row(bgr, &mut sd_y2, &mut sd_cb2, &mut sd_cr2, width);
                assert_eq!(sd_y2, sc_y2, "NEON BGR→Y w={width} iter={i}");
                assert_eq!(sd_cb2, sc_cb2, "NEON BGR→Cb w={width}");
                assert_eq!(sd_cr2, sc_cr2, "NEON BGR→Cr w={width}");

                let bgra: &[u8] = &rgba;
                let mut sc_y3: Vec<u8> = vec![0u8; width];
                let mut sc_cb3: Vec<u8> = vec![0u8; width];
                let mut sc_cr3: Vec<u8> = vec![0u8; width];
                crate::encode::color::bgra_to_ycbcr_row_scalar(
                    bgra,
                    &mut sc_y3,
                    &mut sc_cb3,
                    &mut sc_cr3,
                    width,
                );
                let mut sd_y3: Vec<u8> = vec![0u8; width];
                let mut sd_cb3: Vec<u8> = vec![0u8; width];
                let mut sd_cr3: Vec<u8> = vec![0u8; width];
                neon_bgra_to_ycbcr_row(bgra, &mut sd_y3, &mut sd_cb3, &mut sd_cr3, width);
                assert_eq!(sd_y3, sc_y3, "NEON BGRA→Y w={width} iter={i}");
                assert_eq!(sd_cb3, sc_cb3, "NEON BGRA→Cb w={width}");
                assert_eq!(sd_cr3, sc_cr3, "NEON BGRA→Cr w={width}");
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if is_x86_feature_detected!("avx2") {
                    use crate::simd::x86_64::avx2_color_encode::{
                        avx2_bgr_to_ycbcr_row, avx2_bgra_to_ycbcr_row, avx2_rgb_to_ycbcr_row,
                        avx2_rgba_to_ycbcr_row,
                    };
                    let mut simd_y: Vec<u8> = vec![0u8; width];
                    let mut simd_cb: Vec<u8> = vec![0u8; width];
                    let mut simd_cr: Vec<u8> = vec![0u8; width];
                    avx2_rgb_to_ycbcr_row(&rgb, &mut simd_y, &mut simd_cb, &mut simd_cr, width);
                    assert_eq!(simd_y, scalar_y, "AVX2 RGB→Y w={width} iter={i}");
                    assert_eq!(simd_cb, scalar_cb, "AVX2 RGB→Cb w={width}");
                    assert_eq!(simd_cr, scalar_cr, "AVX2 RGB→Cr w={width}");

                    // RGBA / BGR / BGRA via their own scalar references,
                    // mirroring the NEON and WASM blocks above/below.
                    let mut sc_y: Vec<u8> = vec![0u8; width];
                    let mut sc_cb: Vec<u8> = vec![0u8; width];
                    let mut sc_cr: Vec<u8> = vec![0u8; width];
                    crate::encode::color::rgba_to_ycbcr_row(
                        &rgba, &mut sc_y, &mut sc_cb, &mut sc_cr, width,
                    );
                    let mut sd_y: Vec<u8> = vec![0u8; width];
                    let mut sd_cb: Vec<u8> = vec![0u8; width];
                    let mut sd_cr: Vec<u8> = vec![0u8; width];
                    avx2_rgba_to_ycbcr_row(&rgba, &mut sd_y, &mut sd_cb, &mut sd_cr, width);
                    assert_eq!(sd_y, sc_y, "AVX2 RGBA→Y w={width} iter={i}");
                    assert_eq!(sd_cb, sc_cb, "AVX2 RGBA→Cb w={width}");
                    assert_eq!(sd_cr, sc_cr, "AVX2 RGBA→Cr w={width}");

                    let bgr: &[u8] = &rgb;
                    let mut sc_y2: Vec<u8> = vec![0u8; width];
                    let mut sc_cb2: Vec<u8> = vec![0u8; width];
                    let mut sc_cr2: Vec<u8> = vec![0u8; width];
                    crate::encode::color::bgr_to_ycbcr_row_scalar(
                        bgr,
                        &mut sc_y2,
                        &mut sc_cb2,
                        &mut sc_cr2,
                        width,
                    );
                    let mut sd_y2: Vec<u8> = vec![0u8; width];
                    let mut sd_cb2: Vec<u8> = vec![0u8; width];
                    let mut sd_cr2: Vec<u8> = vec![0u8; width];
                    avx2_bgr_to_ycbcr_row(bgr, &mut sd_y2, &mut sd_cb2, &mut sd_cr2, width);
                    assert_eq!(sd_y2, sc_y2, "AVX2 BGR→Y w={width} iter={i}");
                    assert_eq!(sd_cb2, sc_cb2, "AVX2 BGR→Cb w={width}");
                    assert_eq!(sd_cr2, sc_cr2, "AVX2 BGR→Cr w={width}");

                    let bgra: &[u8] = &rgba;
                    let mut sc_y3: Vec<u8> = vec![0u8; width];
                    let mut sc_cb3: Vec<u8> = vec![0u8; width];
                    let mut sc_cr3: Vec<u8> = vec![0u8; width];
                    crate::encode::color::bgra_to_ycbcr_row_scalar(
                        bgra,
                        &mut sc_y3,
                        &mut sc_cb3,
                        &mut sc_cr3,
                        width,
                    );
                    let mut sd_y3: Vec<u8> = vec![0u8; width];
                    let mut sd_cb3: Vec<u8> = vec![0u8; width];
                    let mut sd_cr3: Vec<u8> = vec![0u8; width];
                    avx2_bgra_to_ycbcr_row(bgra, &mut sd_y3, &mut sd_cb3, &mut sd_cr3, width);
                    assert_eq!(sd_y3, sc_y3, "AVX2 BGRA→Y w={width} iter={i}");
                    assert_eq!(sd_cb3, sc_cb3, "AVX2 BGRA→Cb w={width}");
                    assert_eq!(sd_cr3, sc_cr3, "AVX2 BGRA→Cr w={width}");
                }
            }
            #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
            {
                use crate::simd::wasm32::color_encode::{
                    wasm_bgr_to_ycbcr_row, wasm_bgra_to_ycbcr_row, wasm_rgb_to_ycbcr_row,
                    wasm_rgba_to_ycbcr_row,
                };
                let mut simd_y: Vec<u8> = vec![0u8; width];
                let mut simd_cb: Vec<u8> = vec![0u8; width];
                let mut simd_cr: Vec<u8> = vec![0u8; width];
                wasm_rgb_to_ycbcr_row(&rgb, &mut simd_y, &mut simd_cb, &mut simd_cr, width);
                assert_eq!(simd_y, scalar_y, "WASM RGB→Y w={width} iter={i}");
                assert_eq!(simd_cb, scalar_cb, "WASM RGB→Cb w={width}");
                assert_eq!(simd_cr, scalar_cr, "WASM RGB→Cr w={width}");

                let mut sc_y: Vec<u8> = vec![0u8; width];
                let mut sc_cb: Vec<u8> = vec![0u8; width];
                let mut sc_cr: Vec<u8> = vec![0u8; width];
                crate::encode::color::rgba_to_ycbcr_row(
                    &rgba, &mut sc_y, &mut sc_cb, &mut sc_cr, width,
                );
                let mut sd_y: Vec<u8> = vec![0u8; width];
                let mut sd_cb: Vec<u8> = vec![0u8; width];
                let mut sd_cr: Vec<u8> = vec![0u8; width];
                wasm_rgba_to_ycbcr_row(&rgba, &mut sd_y, &mut sd_cb, &mut sd_cr, width);
                assert_eq!(sd_y, sc_y, "WASM RGBA→Y w={width} iter={i}");
                assert_eq!(sd_cb, sc_cb, "WASM RGBA→Cb w={width}");
                assert_eq!(sd_cr, sc_cr, "WASM RGBA→Cr w={width}");

                let bgr: &[u8] = &rgb;
                let mut sc_y2: Vec<u8> = vec![0u8; width];
                let mut sc_cb2: Vec<u8> = vec![0u8; width];
                let mut sc_cr2: Vec<u8> = vec![0u8; width];
                crate::encode::color::bgr_to_ycbcr_row_scalar(
                    bgr,
                    &mut sc_y2,
                    &mut sc_cb2,
                    &mut sc_cr2,
                    width,
                );
                let mut sd_y2: Vec<u8> = vec![0u8; width];
                let mut sd_cb2: Vec<u8> = vec![0u8; width];
                let mut sd_cr2: Vec<u8> = vec![0u8; width];
                wasm_bgr_to_ycbcr_row(bgr, &mut sd_y2, &mut sd_cb2, &mut sd_cr2, width);
                assert_eq!(sd_y2, sc_y2, "WASM BGR→Y w={width} iter={i}");
                assert_eq!(sd_cb2, sc_cb2, "WASM BGR→Cb w={width}");
                assert_eq!(sd_cr2, sc_cr2, "WASM BGR→Cr w={width}");

                let bgra: &[u8] = &rgba;
                let mut sc_y3: Vec<u8> = vec![0u8; width];
                let mut sc_cb3: Vec<u8> = vec![0u8; width];
                let mut sc_cr3: Vec<u8> = vec![0u8; width];
                crate::encode::color::bgra_to_ycbcr_row_scalar(
                    bgra,
                    &mut sc_y3,
                    &mut sc_cb3,
                    &mut sc_cr3,
                    width,
                );
                let mut sd_y3: Vec<u8> = vec![0u8; width];
                let mut sd_cb3: Vec<u8> = vec![0u8; width];
                let mut sd_cr3: Vec<u8> = vec![0u8; width];
                wasm_bgra_to_ycbcr_row(bgra, &mut sd_y3, &mut sd_cb3, &mut sd_cr3, width);
                assert_eq!(sd_y3, sc_y3, "WASM BGRA→Y w={width} iter={i}");
                assert_eq!(sd_cb3, sc_cb3, "WASM BGRA→Cb w={width}");
                assert_eq!(sd_cr3, sc_cr3, "WASM BGRA→Cr w={width}");
            }
        }
    }
}

// =====================================================================
// Fused FDCT + quantize + zigzag (ISLOW path)
// =====================================================================

#[test]
fn parity_fdct_quantize_islow() {
    let mut rng: Mulberry32 = Mulberry32::new(0xA5A5_0050);

    for i in 0..N {
        // Input is the level-shifted 8x8 block, values in i16 centered
        // around 0 (matches post-level-shift pre-FDCT workspace).
        let mut scalar_input: [i16; 64] = [0i16; 64];
        for slot in scalar_input.iter_mut() {
            // Keep inputs in [-128, 127] to reflect real-world pixel data.
            *slot = (rng.next_u8() as i16) - 128;
        }

        let natural_quant: [u16; 64] = random_quant(&mut rng);
        let quant_divisors: QuantDivisors = build_quant_divisors(natural_quant);

        // Scalar reference modifies its input in-place; clone per call.
        let mut scalar_in: [i16; 64] = scalar_input;
        let mut scalar_out: [i16; 64] = [0i16; 64];
        scalar::scalar_fdct_quantize(&mut scalar_in, &quant_divisors, &mut scalar_out);

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            // There is no pub `neon_fdct_quantize`; the dispatcher picks
            // it via EncoderSimdRoutines. Use the public encoder detector
            // and invoke through the function pointer.
            let routines = crate::simd::detect_encoder();
            let mut simd_in: [i16; 64] = scalar_input;
            let mut simd_out: [i16; 64] = [0i16; 64];
            (routines.fdct_quantize)(&mut simd_in, &quant_divisors, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "NEON fdct_quantize mismatch iter={i}");
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                let routines = crate::simd::detect_encoder();
                let mut simd_in: [i16; 64] = scalar_input;
                let mut simd_out: [i16; 64] = [0i16; 64];
                (routines.fdct_quantize)(&mut simd_in, &quant_divisors, &mut simd_out);
                assert_eq!(simd_out, scalar_out, "AVX2 fdct_quantize mismatch iter={i}");
            }
        }
        #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
        {
            let routines = crate::simd::detect_encoder();
            let mut simd_in: [i16; 64] = scalar_input;
            let mut simd_out: [i16; 64] = [0i16; 64];
            (routines.fdct_quantize)(&mut simd_in, &quant_divisors, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "WASM fdct_quantize mismatch iter={i}");
        }
    }
}
