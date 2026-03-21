use libjpeg_turbo_rs::simd::{self, SimdRoutines};

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
    use libjpeg_turbo_rs::decode::idct;

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
    use libjpeg_turbo_rs::decode::color;

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
    use libjpeg_turbo_rs::decode::upsample;

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
