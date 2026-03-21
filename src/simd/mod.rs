//! SIMD dispatch layer for hot-path JPEG decode operations.
//!
//! Resolves function pointers once at init time via `detect()`.
//! On aarch64, NEON is always available (ARMv8 mandatory).
//! Set `JSIMD_FORCENONE=1` to force scalar fallback.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

/// Function-pointer dispatch table for SIMD-accelerated decode operations.
pub struct SimdRoutines {
    /// Combined dequant + IDCT + level-shift + clamp → u8 output.
    /// `coeffs` and `quant` are both in natural (row-major) order.
    pub idct_islow: fn(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]),

    /// YCbCr → interleaved RGB, one row.
    pub ycbcr_to_rgb_row: fn(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize),

    /// Fancy horizontal 2x upsample, one row.
    /// Output length must be `in_width * 2`.
    pub fancy_upsample_h2v1: fn(input: &[u8], in_width: usize, output: &mut [u8]),
}

/// Detect available SIMD features and return the best dispatch table.
///
/// Checks `JSIMD_FORCENONE` env var first. If set to "1", returns scalar.
/// Otherwise selects NEON on aarch64, scalar elsewhere.
pub fn detect() -> SimdRoutines {
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

    #[allow(unreachable_code)]
    scalar::routines()
}
