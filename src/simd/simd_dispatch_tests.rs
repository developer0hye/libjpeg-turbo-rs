//! Relocated from `tests/simd_dispatch.rs` for P4-135 criterion 2 (#474).
//!
//! This suite reaches SIMD kernels directly, which is why the arch
//! modules had to stay `pub` and were therefore callable from any
//! downstream crate. As an in-crate test it uses `crate::`, so they
//! can be private. Moved verbatim apart from the path rewrite.

use crate::simd::{self, SimdRoutines};

#[test]
fn detect_returns_valid_routines() {
    let routines: SimdRoutines = simd::detect();
    let coeffs = [0i16; 64];
    let quant = [1u16; 64];
    let mut output = [0u8; 64];
    (routines.idct_islow)(&coeffs, &quant, &mut output);
    // DC=0, quant=1 → all zeros after IDCT → level shift +128 → all 128
    assert!(
        output.iter().all(|&v| v == 128),
        "DC-only zero block should produce all 128s, got {:?}",
        &output[..8]
    );
}

#[test]
fn forcenone_forces_scalar() {
    std::env::set_var("JSIMD_FORCENONE", "1");
    let routines = simd::detect();
    std::env::remove_var("JSIMD_FORCENONE");

    // DC=800 with quant=1 in natural order position 0
    let mut coeffs = [0i16; 64];
    coeffs[0] = 800;
    let quant = [1u16; 64];
    let mut output = [0u8; 64];
    (routines.idct_islow)(&coeffs, &quant, &mut output);
    assert!(
        output.iter().all(|&v| v == 228),
        "DC=800 block should produce all 228s, got {:?}",
        &output[..8]
    );
}

#[test]
fn scalar_idct_matches_existing_functions() {
    use crate::decode::idct;

    // Create a block with known coefficients in natural (row-major) order.
    // Position [0] = DC, [1] = (0,1), [8] = (1,0), [16] = (2,0)
    let mut natural_coeffs = [0i16; 64];
    natural_coeffs[0] = 200; // DC
    natural_coeffs[1] = -30; // (0,1)
    natural_coeffs[8] = 15; // (1,0)
    natural_coeffs[16] = -5; // (2,0)

    let mut quant_values = [1u16; 64];
    quant_values[0] = 16;
    quant_values[1] = 11;
    quant_values[8] = 12;
    quant_values[16] = 14;

    // Compute expected: dequantize then IDCT then level-shift
    let mut dequantized = [0i16; 64];
    for i in 0..64 {
        dequantized[i] = natural_coeffs[i].wrapping_mul(quant_values[i] as i16);
    }
    let spatial = idct::idct_8x8(&dequantized);
    let mut expected = [0u8; 64];
    for i in 0..64 {
        expected[i] = (spatial[i] as i32 + 128).clamp(0, 255) as u8;
    }

    // Compute using SIMD wrapper (coeffs now in natural order)
    let routines = simd::detect();
    let mut actual = [0u8; 64];
    (routines.idct_islow)(&natural_coeffs, &quant_values, &mut actual);

    assert_eq!(
        actual, expected,
        "SIMD wrapper should match dequant+idct+level-shift"
    );
}

#[test]
fn scalar_ycbcr_to_rgb_matches_existing() {
    use crate::decode::color;

    let routines = simd::detect();

    let width = 32;
    let y: Vec<u8> = (0..width).map(|i| (i * 8) as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| (128 + i) as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| (128 - i) as u8).collect();

    let mut expected = vec![0u8; width * 3];
    color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

    let mut actual = vec![0u8; width * 3];
    (routines.ycbcr_to_rgb_row)(&y, &cb, &cr, &mut actual, width);

    assert_eq!(actual, expected);
}

#[test]
fn scalar_fancy_upsample_matches_existing() {
    use crate::decode::upsample;

    let routines = simd::detect();

    let input: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
    let in_width = input.len();
    let out_width = in_width * 2;

    let mut expected = vec![0u8; out_width];
    upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

    let mut actual = vec![0u8; out_width];
    (routines.fancy_upsample_h2v1)(&input, in_width, &mut actual);

    assert_eq!(actual, expected);
}

#[test]
fn no_simd_feature_compiles_scalar() {
    let routines = simd::detect();
    let coeffs = [0i16; 64];
    let quant = [1u16; 64];
    let mut output = [0u8; 64];
    (routines.idct_islow)(&coeffs, &quant, &mut output);
    assert!(output.iter().all(|&v| v == 128));
}

/// P4-133 (#464) criteria 2 and 3: the float FDCT's FMA twin is reached from
/// a baseline build by runtime detection, and the choice is made once, where
/// the encoder's kernel set is built — not per block.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod p4_133_float_fdct_dispatch {
    use crate::simd::{self, EncoderSimdRoutines};

    fn as_ptr(kernel: fn(&mut [i16; 64], &simd::QuantDivisors, &mut [i16; 64])) -> *const () {
        kernel as *const ()
    }

    #[test]
    fn encoder_kernel_set_installs_the_fma_float_fdct_only_when_the_cpu_has_it() {
        let routines: EncoderSimdRoutines = simd::x86_64::encoder_routines();
        let selected: *const () = as_ptr(routines.fdct_float_quantize);
        let fma_twin: *const () = as_ptr(simd::x86_64::fma_fdct::fma_fdct_float_quantize);
        let scalar: *const () = as_ptr(simd::scalar::scalar_fdct_float_quantize);
        let cpu_has_fma: bool = crate::cpu_has!("fma");
        eprintln!("runtime dispatch: fma={cpu_has_fma}");

        if cpu_has_fma {
            assert!(
                core::ptr::eq(selected, fma_twin),
                "CPU reports FMA but the plan's float FDCT kernel is not the FMA twin"
            );
        } else {
            assert!(
                core::ptr::eq(selected, scalar),
                "CPU has no FMA but the plan's float FDCT kernel is not the scalar reference"
            );
        }
    }

    /// The scalar kernel set (what `JSIMD_FORCENONE=1` and a `no_std` build
    /// without `target_feature = "fma"` get) must stay scalar for the float
    /// kernel too.
    #[test]
    fn scalar_kernel_set_keeps_the_float_fdct_scalar() {
        let routines: EncoderSimdRoutines = simd::scalar::encoder_routines();
        assert!(core::ptr::eq(
            as_ptr(routines.fdct_float_quantize),
            as_ptr(simd::scalar::scalar_fdct_float_quantize)
        ));
    }
}
