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
///
/// The float method has more than one kernel — the scalar reference and, on
/// x86_64 with FMA, its `target_feature` twin (P4-133) — so this is the one
/// place that enumerates them.
#[inline]
pub(super) fn may_use_islow_simd_kernel(
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> bool {
    let kernel: *const () = fdct_quantize_fn as *const ();
    let is_ifast: bool = core::ptr::eq(
        kernel,
        crate::simd::scalar::scalar_fdct_ifast_quantize as *const (),
    );
    let is_float: bool = core::ptr::eq(
        kernel,
        crate::simd::scalar::scalar_fdct_float_quantize as *const (),
    ) || is_fma_float_twin(kernel);
    !is_ifast && !is_float
}

/// Whether `kernel` is the x86_64 FMA twin of the float FDCT kernel.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn is_fma_float_twin(kernel: *const ()) -> bool {
    core::ptr::eq(
        kernel,
        crate::simd::x86_64::fma_fdct::fma_fdct_float_quantize as *const (),
    )
}

/// No FMA twin exists off x86_64; the scalar kernel is the only float kernel.
#[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
#[inline]
fn is_fma_float_twin(_kernel: *const ()) -> bool {
    false
}

/// Color conversion function: (pixels, y, cb, cr, width).
pub(super) type ColorConvertRowFn = fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize);

/// Select the best available RGBA→YCbCr row conversion function.
pub(super) fn select_rgba_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_rgba_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
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
    #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
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
    #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// P4-133 (#464): the float-DCT kernel now has an FMA twin. The one guard
    /// that decides "may the fused islow SIMD kernel replace the requested
    /// transform?" compares function pointers, so the twin must be rejected
    /// exactly like the scalar float kernel — otherwise `-dct float` on an
    /// FMA CPU would silently get islow coefficients fed to float divisors,
    /// the #330 defect all over again.
    #[test]
    fn float_and_ifast_kernels_never_unlock_the_islow_simd_shortcut() {
        let enc_simd = crate::simd::detect_encoder();
        assert!(
            may_use_islow_simd_kernel(enc_simd.fdct_quantize),
            "the islow kernel the plan installs must keep the SIMD shortcut"
        );
        assert!(!may_use_islow_simd_kernel(
            crate::simd::scalar::scalar_fdct_ifast_quantize
        ));
        assert!(!may_use_islow_simd_kernel(
            crate::simd::scalar::scalar_fdct_float_quantize
        ));
        assert!(
            !may_use_islow_simd_kernel(enc_simd.fdct_float_quantize),
            "whatever float kernel the plan installed on this CPU must not \
             unlock the islow shortcut"
        );

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            let fma_twin = crate::simd::x86_64::fma_fdct::fma_fdct_float_quantize;
            assert!(!may_use_islow_simd_kernel(fma_twin));
        }
    }

    /// P4-133 (#464) criterion 2: the FMA twin of the float FDCT+quantise
    /// kernel must be bit-identical to the scalar reference on every block,
    /// because `f32::mul_add` is already a fused single-rounding operation —
    /// the twin only changes how it is *emitted* (one `vfmadd` instead of a
    /// libm `fmaf` call). On a CPU with FMA the FMA-compiled body is called
    /// directly, so the comparison cannot quietly degrade into
    /// scalar-versus-scalar through the safe wrapper's fallback; without FMA
    /// the wrapper is compared and the run says so.
    ///
    /// Uses the encoder's real divisor builder over Annex K tables at five
    /// qualities, so the compared values are the ones `cjpeg -dct float`
    /// parity rests on.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[test]
    fn fma_float_fdct_twin_is_bit_identical_to_the_scalar_kernel() {
        use crate::simd::scalar::scalar_fdct_float_quantize;
        use crate::simd::x86_64::fma_fdct::{
            fma_fdct_float_quantize, fma_fdct_float_quantize_inner,
        };

        let cpu_has_fma: bool = crate::cpu_has!("fma");
        if !cpu_has_fma {
            eprintln!("NOTE: no FMA on this CPU; comparing the twin's fallback path only");
        }

        let mut state: u32 = 0x9E37_79B9;
        let mut next_u32 = move || -> u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for quality in [25u8, 50, 75, 90, 100] {
            for table in [
                &tables::STD_LUMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
            ] {
                let scaled: [u16; 64] = tables::quality_scale_quant_table(table, quality);
                let divisors: QuantDivisors =
                    super::super::quant_divisors::scale_quant_for_fdct(&scaled);

                for block_index in 0..400usize {
                    let mut block: [i16; 64] = [0i16; 64];
                    match block_index % 4 {
                        // Full-range level-shifted samples.
                        0 => {
                            for sample in block.iter_mut() {
                                *sample = (next_u32() & 0xff) as i16 - 128;
                            }
                        }
                        // Flat extremes: the largest DC the FDCT can see.
                        1 => {
                            let flat: i16 = if block_index % 8 == 1 { -128 } else { 127 };
                            block = [flat; 64];
                        }
                        // Low-amplitude noise: coefficients that land right at
                        // the `(int)(temp + 16384.5)` rounding boundary.
                        2 => {
                            for sample in block.iter_mut() {
                                *sample = (next_u32() % 7) as i16 - 3;
                            }
                        }
                        // Checkerboard: maximum high-frequency energy.
                        _ => {
                            for (i, sample) in block.iter_mut().enumerate() {
                                let parity: usize = (i / 8 + i % 8) % 2;
                                *sample = if parity == 0 { 127 } else { -128 };
                            }
                        }
                    }

                    let mut scalar_input: [i16; 64] = block;
                    let mut fma_input: [i16; 64] = block;
                    let mut scalar_output: [i16; 64] = [0i16; 64];
                    let mut fma_output: [i16; 64] = [0i16; 64];
                    scalar_fdct_float_quantize(&mut scalar_input, &divisors, &mut scalar_output);
                    if cpu_has_fma {
                        // SAFETY: FMA confirmed above; fixed-size arrays.
                        unsafe {
                            fma_fdct_float_quantize_inner(
                                &mut fma_input,
                                &divisors,
                                &mut fma_output,
                            );
                        }
                    } else {
                        fma_fdct_float_quantize(&mut fma_input, &divisors, &mut fma_output);
                    }
                    assert_eq!(
                        scalar_output, fma_output,
                        "quality {quality} block {block_index}: FMA twin diverged from scalar"
                    );
                }
            }
        }
    }
}
