//! SIMD dispatch layer for hot-path JPEG decode and encode operations.
//!
//! Resolves function pointers once at init time via `detect()` / `detect_encoder()`.
//! On aarch64, NEON is always available (ARMv8 mandatory).
//! Set `JSIMD_FORCENONE=1` to force scalar fallback.

// P4-135 criterion 2 (#474): the arch kernel modules are crate-private. They
// were `pub` only because the SIMD suites lived in `tests/` and reached them
// through `libjpeg_turbo_rs::simd::*`; those suites now live in this module,
// so the kernels are no longer callable from a downstream crate.
//
// `detect`, `detect_encoder`, `SimdRoutines` and `QuantDivisors` stay public:
// `benches/{decode,encode}.rs` use them to select a routine set, and they
// reach kernels only through the dispatch table, never by path. There is no
// trade-off between benchmark coverage and reachability here.
//
// Module privacy alone was not enough, though. `detect()` is public and used
// to return a table whose safe `fn`-pointer fields were `pub`, which put the
// same unvalidated kernels one field access away from a downstream crate —
// P4-135 criterion 4. The fields carrying a `width`-versus-length precondition
// are now `pub(crate)`, fronted by validating methods on the two tables; see
// `SimdRoutines` for which fields that covers and why the rest are exempt.
pub(crate) mod scalar;

#[cfg(test)]
mod neon_color_tests;
#[cfg(test)]
mod neon_idct_tests;
#[cfg(test)]
mod neon_upsample_tests;
#[cfg(test)]
mod simd_avx2_encode_tests;
#[cfg(test)]
mod simd_avx2_tests;
#[cfg(test)]
mod simd_dispatch_tests;
#[cfg(test)]
mod simd_neon_encode_tests;
#[cfg(test)]
mod simd_neon_scaled_tests;
#[cfg(test)]
mod simd_parity_tests;
#[cfg(test)]
mod simd_x86_tests;

// P4-135 criterion 5 (#474): arch backends compile only under the Cargo
// `simd` feature. The wasm32 backend additionally requires the `simd128`
// *target* feature at compile time — its kernels are `core::arch::wasm32`
// intrinsics, wasm has no runtime feature detection, and an engine without
// SIMD support rejects a module containing SIMD instructions at validation,
// so baseline `wasm32` builds must not contain them at all. Every call site
// uses the same predicate as its module gate (P4-143 aligned them); the
// `not(...)` complements select the scalar fallback documented in
// `crates/libjpeg-turbo-rs-wasm/README.md`.
//
// The first attempt at this narrowing introduced a compile break for baseline
// `wasm32` that the whole matrix missed, because `.cargo/config.toml` forces
// `+simd128` on every in-tree build (P4-143); review caught it before it
// shipped. The wasm workflow now carries a baseline leg that builds both wasm
// targets with the config's rustflags overridden and warnings denied, so a
// call site that drifts from these predicates fails CI instead of a
// downstream consumer's build.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub(crate) mod aarch64;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub(crate) mod x86_64;

#[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
pub(crate) mod wasm32;

/// Bytes per pixel written by every kernel behind `ycbcr_to_rgb_row` and read
/// by every kernel behind `rgb_to_ycbcr_row`.
const RGB_BYTES_PER_PIXEL: usize = 3;

/// Shared by the validating wrappers: `width` samples must fit in `slice`.
///
/// Split out so the panic message names the entry point rather than the
/// helper, which is what a caller needs in order to fix the call.
#[inline]
#[track_caller]
fn require_samples(entry: &str, plane: &str, slice_len: usize, width: usize) {
    assert!(
        slice_len >= width,
        "{entry}: `{plane}` holds {slice_len} samples but width is {width}; the \
         SIMD kernels index up to `width` with unchecked loads and stores"
    );
}

/// Shared by the validating wrappers: `width * bytes_per_pixel` bytes must fit.
///
/// The multiplication is `checked_mul` so an attacker-influenced `width` cannot
/// wrap into a small product that a short buffer appears to satisfy.
#[inline]
#[track_caller]
fn require_bytes(entry: &str, plane: &str, slice_len: usize, width: usize, bpp: usize) {
    let needed: usize = width.checked_mul(bpp).unwrap_or_else(|| {
        panic!("{entry}: width {width} × {bpp} bytes per pixel overflows `usize`")
    });
    assert!(
        slice_len >= needed,
        "{entry}: `{plane}` holds {slice_len} bytes but width {width} needs \
         {needed} ({bpp} bytes per pixel)"
    );
}

/// Function-pointer dispatch table for SIMD-accelerated decode operations.
///
/// # Why some fields are public and others are not
///
/// The kernels behind this table are `target_feature` functions that index on a
/// `width` argument using unchecked loads and stores. A field is `pub` only
/// where the type alone makes that impossible to get wrong (P4-135, #474):
///
/// * `idct_islow` / `idct_ifast` / `idct_float` — **`pub`**. Every parameter is
///   a fixed-size array reference, so the lengths are in the type and no
///   argument choice can drive the kernel out of bounds.
/// * `ycbcr_to_rgb_row` / `fancy_upsample_h2v1` — **`pub(crate)`**. Both carry a
///   length precondition that the type does not express, so calling them is
///   only safe once that precondition is checked. External callers go through
///   the identically named methods below, which check it first.
pub struct SimdRoutines {
    /// Combined dequant + IDCT (ISLOW) + level-shift + clamp → u8 output.
    /// `coeffs` and `quant` are both in natural (row-major) order.
    pub idct_islow: fn(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]),

    /// Combined dequant + IDCT (IFAST) + level-shift + clamp → u8 output.
    pub idct_ifast: fn(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]),

    /// Combined dequant + IDCT (Float) + level-shift + clamp → u8 output.
    pub idct_float: fn(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]),

    /// YCbCr → interleaved RGB, one row.
    ///
    /// Crate-private: `y`, `cb` and `cr` must each hold at least `width`
    /// samples and `rgb` at least `width * 3` bytes. Call
    /// [`SimdRoutines::ycbcr_to_rgb_row()`] to have that checked.
    #[allow(clippy::type_complexity)]
    pub(crate) ycbcr_to_rgb_row: fn(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize),

    /// Fancy horizontal 2x upsample, one row.
    ///
    /// Crate-private: `input` must hold at least `in_width` samples and
    /// `output` at least `in_width * 2`. Call
    /// [`SimdRoutines::fancy_upsample_h2v1()`] to have that checked.
    pub(crate) fancy_upsample_h2v1: fn(input: &[u8], in_width: usize, output: &mut [u8]),
}

impl SimdRoutines {
    /// YCbCr → interleaved RGB, one row, with the kernel's length precondition
    /// checked first.
    ///
    /// The CPU-feature half of the precondition is discharged by construction:
    /// the only way to obtain a `SimdRoutines` is [`detect`], which installs an
    /// arch kernel exclusively on a CPU where runtime detection confirmed it.
    ///
    /// # Panics
    ///
    /// If `y`, `cb` or `cr` holds fewer than `width` samples, if `rgb` holds
    /// fewer than `width * 3` bytes, or if `width * 3` overflows `usize`.
    ///
    /// Panicking rather than returning a `Result` matches what the scalar
    /// kernel already does on the same inputs, where the slice indexing panics.
    #[inline]
    #[track_caller]
    pub fn ycbcr_to_rgb_row(&self, y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
        const ENTRY: &str = "ycbcr_to_rgb_row";
        require_samples(ENTRY, "y", y.len(), width);
        require_samples(ENTRY, "cb", cb.len(), width);
        require_samples(ENTRY, "cr", cr.len(), width);
        require_bytes(ENTRY, "rgb", rgb.len(), width, RGB_BYTES_PER_PIXEL);

        (self.ycbcr_to_rgb_row)(y, cb, cr, rgb, width)
    }

    /// Fancy horizontal 2x upsample, one row, with the kernel's length
    /// precondition checked first.
    ///
    /// # Panics
    ///
    /// If `input` holds fewer than `in_width` samples, if `output` holds fewer
    /// than `in_width * 2` bytes, or if `in_width * 2` overflows `usize`.
    #[inline]
    #[track_caller]
    pub fn fancy_upsample_h2v1(&self, input: &[u8], in_width: usize, output: &mut [u8]) {
        const ENTRY: &str = "fancy_upsample_h2v1";
        require_samples(ENTRY, "input", input.len(), in_width);
        require_bytes(ENTRY, "output", output.len(), in_width, 2);

        (self.fancy_upsample_h2v1)(input, in_width, output)
    }
}

/// Pre-computed quantization divisor table with adaptive-precision reciprocals.
///
/// Uses C libjpeg-turbo's `compute_reciprocal` algorithm for exact results:
/// reciprocal precision adapts per-element, with a correction factor and
/// per-element variable shift to match true integer division.
///
/// The NEON path uses `reciprocals`, `corrections`, `shifts` to avoid scalar division.
/// The AVX2 path uses `reciprocals`, `corrections`, `scales` (two `mulhi` ops, matching C).
/// The scalar path ignores these and divides directly using `divisors`.
pub struct QuantDivisors {
    /// Divisor values (quant × 8, matching FDCT output scaling).
    pub divisors: [u16; 64],
    /// Adaptive-precision reciprocals (see `compute_reciprocal`).
    pub reciprocals: [u16; 64],
    /// Correction factors: divisor/2, adjusted +1 when reciprocal was rounded down.
    pub corrections: [u16; 64],
    /// Per-element right-shift amounts: `r - 16` where `r = 16 + flss(divisor) - 1`.
    pub shifts: [i16; 64],
    /// Scale factors for AVX2: `1 << (32 - r)`, replacing per-element shift with a
    /// second `pmulhuw`. Matches C libjpeg-turbo's SCALE table in jcdctmgr.c.
    pub scales: [u16; 64],
    /// Divisors re-arranged in zigzag scan order for fused quantize+reorder.
    pub divisors_zigzag: [u16; 64],
    /// Reciprocals re-arranged in zigzag scan order.
    pub reciprocals_zigzag: [u16; 64],
    /// Corrections re-arranged in zigzag scan order.
    pub corrections_zigzag: [u16; 64],
    /// Shifts re-arranged in zigzag scan order.
    pub shifts_zigzag: [i16; 64],
    /// Scales re-arranged in zigzag scan order.
    pub scales_zigzag: [u16; 64],
    /// Float divisors matching C `jcdctmgr.c` `forward_DCT_float`:
    /// `1 / (quant[i] * aanscale[row] * aanscale[col] * 8)`. Paired with the
    /// raw float FDCT (no AA&N rescale) and `quantize_float` to reproduce
    /// `cjpeg -dct float` byte-for-byte.
    pub float_divisors: [f32; 64],
    /// Float divisors re-arranged in zigzag scan order so the float quant
    /// step can fuse zigzag reorder.
    pub float_divisors_zigzag: [f32; 64],
}

/// Function-pointer dispatch table for SIMD-accelerated encode operations.
///
/// Field visibility follows the same rule as [`SimdRoutines`]: `fdct_quantize`
/// is `pub` because its parameters are fixed-size arrays, `rgb_to_ycbcr_row` is
/// `pub(crate)` because its safety depends on `width` agreeing with four slice
/// lengths (P4-135, #474).
pub struct EncoderSimdRoutines {
    /// RGB → YCbCr color conversion, one row.
    /// Only handles interleaved RGB (3 bytes/pixel).
    ///
    /// Crate-private: `rgb` must hold at least `width * 3` bytes and `y`, `cb`,
    /// `cr` at least `width` samples each. Call
    /// [`EncoderSimdRoutines::rgb_to_ycbcr_row()`] to have that checked.
    #[allow(clippy::type_complexity)]
    pub(crate) rgb_to_ycbcr_row:
        fn(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize),

    /// Combined FDCT (islow) + quantize + zigzag reorder for one 8×8 block.
    /// `input` is modified in-place by FDCT (caller must not read after call).
    /// `quant` contains pre-scaled divisors and reciprocals.
    /// Output is in zigzag scan order, ready for Huffman encoding.
    pub fdct_quantize: fn(input: &mut [i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]),
}

impl EncoderSimdRoutines {
    /// RGB → YCbCr color conversion, one row, with the kernel's length
    /// precondition checked first.
    ///
    /// As with [`SimdRoutines::ycbcr_to_rgb_row()`], the CPU-feature half of the
    /// precondition is discharged by construction — [`detect_encoder`] is the
    /// only constructor and it installs an arch kernel only where runtime
    /// detection confirmed the feature.
    ///
    /// # Panics
    ///
    /// If `rgb` holds fewer than `width * 3` bytes, if `y`, `cb` or `cr` holds
    /// fewer than `width` samples, or if `width * 3` overflows `usize`.
    #[inline]
    #[track_caller]
    pub fn rgb_to_ycbcr_row(
        &self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        width: usize,
    ) {
        const ENTRY: &str = "rgb_to_ycbcr_row";
        require_bytes(ENTRY, "rgb", rgb.len(), width, RGB_BYTES_PER_PIXEL);
        require_samples(ENTRY, "y", y.len(), width);
        require_samples(ENTRY, "cb", cb.len(), width);
        require_samples(ENTRY, "cr", cr.len(), width);

        (self.rgb_to_ycbcr_row)(rgb, y, cb, cr, width)
    }
}

/// Detect available SIMD features and return the best dispatch table.
///
/// Checks `JSIMD_FORCENONE` env var first. If set to "1", returns scalar.
/// Otherwise selects NEON on aarch64, scalar elsewhere.
pub fn detect() -> SimdRoutines {
    // Env-var override is a std-only debugging aid; a no_std build has
    // no environment to read (issue #356).
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    if std::env::var("JSIMD_FORCENONE").ok().as_deref() == Some("1") {
        return scalar::routines();
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return aarch64::routines();
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        return x86_64::routines();
    }

    #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
    {
        return wasm32::routines();
    }

    #[allow(unreachable_code)]
    scalar::routines()
}

/// Detect available SIMD features and return the best encoder dispatch table.
pub fn detect_encoder() -> EncoderSimdRoutines {
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    if std::env::var("JSIMD_FORCENONE").ok().as_deref() == Some("1") {
        return scalar::encoder_routines();
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return aarch64::encoder_routines();
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        return x86_64::encoder_routines();
    }

    #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
    {
        return wasm32::encoder_routines();
    }

    #[allow(unreachable_code)]
    scalar::encoder_routines()
}
