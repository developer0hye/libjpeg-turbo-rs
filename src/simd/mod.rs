//! SIMD dispatch layer for hot-path JPEG decode and encode operations.
//!
//! Resolves function pointers once at init time via `detect()` / `detect_encoder()`.
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
    #[allow(clippy::type_complexity)]
    pub ycbcr_to_rgb_row: fn(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize),

    /// Fancy horizontal 2x upsample, one row.
    /// Output length must be `in_width * 2`.
    pub fancy_upsample_h2v1: fn(input: &[u8], in_width: usize, output: &mut [u8]),
}

/// Pre-computed quantization divisor table with reciprocals for fast multiply-shift.
///
/// The NEON path uses `reciprocals` to avoid scalar division.
/// The scalar path ignores reciprocals and divides directly using `divisors`.
pub struct QuantDivisors {
    /// Divisor values (quant × 8, matching FDCT output scaling).
    pub divisors: [u16; 64],
    /// Fixed-point reciprocals: `((1u32 << 16) + divisor - 1) / divisor` (ceiling).
    pub reciprocals: [u16; 64],
    /// Divisors re-arranged in zigzag scan order for fused quantize+reorder.
    /// `divisors_zigzag[zz] = divisors[ZIGZAG_ORDER[zz]]`
    pub divisors_zigzag: [u16; 64],
    /// Reciprocals re-arranged in zigzag scan order.
    pub reciprocals_zigzag: [u16; 64],
}

/// Function-pointer dispatch table for SIMD-accelerated encode operations.
pub struct EncoderSimdRoutines {
    /// RGB → YCbCr color conversion, one row.
    /// Only handles interleaved RGB (3 bytes/pixel).
    #[allow(clippy::type_complexity)]
    pub rgb_to_ycbcr_row: fn(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize),

    /// Combined FDCT (islow) + quantize + zigzag reorder for one 8×8 block.
    /// `quant` contains pre-scaled divisors and reciprocals.
    /// Output is in zigzag scan order, ready for Huffman encoding.
    pub fdct_quantize: fn(input: &[i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]),
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

/// Detect available SIMD features and return the best encoder dispatch table.
pub fn detect_encoder() -> EncoderSimdRoutines {
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

    #[allow(unreachable_code)]
    scalar::encoder_routines()
}
