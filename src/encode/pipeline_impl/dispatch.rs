// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
//! Shared encoder dispatch and option resolution.

use super::{color, tables, QuantDivisors};

/// Resolves the luma/chroma quantization tables for a component pair.
///
/// A custom slot wins; otherwise Annex K scaled by quality. Slot 0 is luma,
/// slot 1 chroma — the convention every entry point here shares.
pub(super) fn resolve_quant_tables(
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
    quality: u8,
) -> ([u16; 64], [u16; 64]) {
    let luma: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma: [u16; 64] = match custom_quant.and_then(|tables| tables[1]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };
    (luma, chroma)
}

/// Whether the fused SIMD extract+FDCT+quantize kernels may be used.
///
/// Those kernels hardcode the **islow** transform. The `ifast` and `float`
/// methods come with divisor tables scaled for their own transforms, so
/// feeding islow coefficients to them mis-scales every output by the AA&N
/// factor — which is how `-dct fast` ended up both lower quality and larger
/// than C's (#330). Callers that hold a `fdct_quantize_fn` must therefore ask
/// this before taking a SIMD shortcut.
#[inline]
pub(super) fn may_use_islow_simd_kernel(
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> bool {
    let is_ifast: bool = core::ptr::eq(
        fdct_quantize_fn as *const (),
        crate::simd::scalar::scalar_fdct_ifast_quantize as *const (),
    );
    let is_float: bool = core::ptr::eq(
        fdct_quantize_fn as *const (),
        crate::simd::scalar::scalar_fdct_float_quantize as *const (),
    );
    !is_ifast && !is_float
}

/// Color conversion function: (pixels, y, cb, cr, width).
pub(super) type ColorConvertRowFn = fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize);

/// Select the best available RGBA→YCbCr row conversion function.
pub(super) fn select_rgba_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_rgba_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    {
        return crate::simd::wasm32::color_encode::wasm_rgba_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if crate::cpu_has!("avx2") {
            return crate::simd::x86_64::avx2_color_encode::avx2_rgba_to_ycbcr_row;
        }
    }
    #[allow(unreachable_code)]
    color::rgba_to_ycbcr_row
}

/// Select the best available BGR→YCbCr row conversion function.
pub(super) fn select_bgr_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_bgr_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    {
        return crate::simd::wasm32::color_encode::wasm_bgr_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if crate::cpu_has!("avx2") {
            return crate::simd::x86_64::avx2_color_encode::avx2_bgr_to_ycbcr_row;
        }
    }
    #[allow(unreachable_code)]
    color::bgr_to_ycbcr_row_scalar
}

/// Select the best available BGRA→YCbCr row conversion function.
pub(super) fn select_bgra_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_bgra_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    {
        return crate::simd::wasm32::color_encode::wasm_bgra_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if crate::cpu_has!("avx2") {
            return crate::simd::x86_64::avx2_color_encode::avx2_bgra_to_ycbcr_row;
        }
    }
    #[allow(unreachable_code)]
    color::bgra_to_ycbcr_row_scalar
}
