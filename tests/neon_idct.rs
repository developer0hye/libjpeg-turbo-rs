//! NEON IDCT tests — verify byte-exact match with scalar implementation.
//!
//! Note: The NEON IDCT uses i16 workspace internally (matching libjpeg-turbo's
//! NEON implementation), while the scalar path uses i32. For very large
//! dequantized values (|coeff * quant| >= ~8192), intermediate narrowing
//! can overflow i16. All tests use realistic JPEG value ranges where both
//! paths agree.
#![cfg(target_arch = "aarch64")]

use libjpeg_turbo_rs::simd;

/// Helper: compute scalar IDCT result for comparison.
fn scalar_idct(coeffs: &[i16; 64], quant: &[u16; 64]) -> [u8; 64] {
    let routines = {
        std::env::set_var("JSIMD_FORCENONE", "1");
        let r = simd::detect();
        std::env::remove_var("JSIMD_FORCENONE");
        r
    };
    let mut output = [0u8; 64];
    (routines.idct_islow)(coeffs, quant, &mut output);
    output
}

/// Helper: compute NEON IDCT result.
fn neon_idct(coeffs: &[i16; 64], quant: &[u16; 64]) -> [u8; 64] {
    use libjpeg_turbo_rs::simd::aarch64;
    let routines = aarch64::routines();
    let mut output = [0u8; 64];
    (routines.idct_islow)(coeffs, quant, &mut output);
    output
}

#[test]
fn neon_idct_all_zeros() {
    let coeffs = [0i16; 64];
    let quant = [1u16; 64];
    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "all-zeros block mismatch");
}

#[test]
fn neon_idct_dc_only() {
    // quant=1 keeps dequantized values small enough for i16 workspace
    let quant = [1u16; 64];
    for dc in [-2000i16, -1000, -500, -100, -1, 0, 1, 100, 500, 1000, 2000] {
        let mut coeffs = [0i16; 64];
        coeffs[0] = dc;
        let scalar = scalar_idct(&coeffs, &quant);
        let neon = neon_idct(&coeffs, &quant);
        assert_eq!(neon, scalar, "DC-only block mismatch for dc={dc}");
    }
}

#[test]
fn neon_idct_dc_only_with_quant() {
    // With larger quant, constrain DC so |DC*quant| < 8192
    let quant = [16u16; 64];
    for dc in [-500i16, -200, -50, 0, 50, 200, 500] {
        let mut coeffs = [0i16; 64];
        coeffs[0] = dc;
        let scalar = scalar_idct(&coeffs, &quant);
        let neon = neon_idct(&coeffs, &quant);
        assert_eq!(neon, scalar, "DC-only (quant=16) mismatch for dc={dc}");
    }
}

#[test]
fn neon_idct_known_ac() {
    // Realistic coefficients: moderate DC, small AC
    let mut coeffs = [0i16; 64];
    coeffs[0] = 200; // DC (zigzag 0)
    coeffs[1] = -30; // zigzag pos 1
    coeffs[2] = 15; // zigzag pos 2
    coeffs[3] = -10; // zigzag pos 3
    coeffs[5] = 8; // zigzag pos 5
    coeffs[10] = -5; // zigzag pos 10

    let mut quant = [1u16; 64];
    quant[0] = 16;
    quant[1] = 11;
    quant[8] = 12;
    quant[16] = 14;
    quant[9] = 12;
    quant[2] = 10;

    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "known AC block mismatch");
}

#[test]
fn neon_idct_negative_dc() {
    let mut coeffs = [0i16; 64];
    coeffs[0] = -500;
    let quant = [1u16; 64];
    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "negative DC block mismatch");
}

#[test]
fn neon_idct_moderate_values() {
    // Non-trivial block with moderate coefficients and quant values
    let mut coeffs = [0i16; 64];
    coeffs[0] = 300;
    coeffs[1] = -80;
    coeffs[2] = 40;
    coeffs[3] = -20;
    coeffs[4] = 10;
    coeffs[5] = -5;
    coeffs[8] = 50;
    coeffs[9] = -25;

    let mut quant = [2u16; 64];
    quant[0] = 8;
    quant[1] = 6;

    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "moderate values mismatch");
}

#[test]
fn neon_idct_random_blocks() {
    // Pseudorandom deterministic blocks with realistic JPEG value ranges.
    let mut seed: u32 = 0xDEAD_BEEF;
    let quant_table: [u16; 64] = {
        let mut q = [0u16; 64];
        for v in q.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed >> 16) % 32 + 1) as u16; // 1..32
        }
        q
    };

    for block_num in 0..100 {
        let mut coeffs = [0i16; 64];
        for (i, c) in coeffs.iter_mut().enumerate() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            // Constrain so |coeff * quant| < ~4000 to avoid i16 overflow
            let range = if i == 0 { 100 } else { 30 / (1 + i / 8) };
            let range = range.max(1) as i32;
            *c = ((seed >> 16) as i32 % (2 * range) - range) as i16;
        }

        let scalar = scalar_idct(&coeffs, &quant_table);
        let neon = neon_idct(&coeffs, &quant_table);
        assert_eq!(
            neon,
            scalar,
            "random block {block_num} mismatch\ncoeffs[0..8]: {:?}",
            &coeffs[..8]
        );
    }
}

#[test]
fn neon_idct_sparse_rows4to7_zero() {
    // Only first few zigzag positions non-zero (rows 4-7 zero → sparse path)
    let mut coeffs = [0i16; 64];
    coeffs[0] = 400; // DC
    coeffs[1] = -20; // AC
    coeffs[2] = 10;
    coeffs[3] = -5;
    let quant = [4u16; 64];
    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "sparse (rows 4-7 zero) block mismatch");
}

#[test]
fn neon_idct_right_half_zero() {
    // Only left columns have non-zero coefficients
    let mut coeffs = [0i16; 64];
    coeffs[0] = 300;
    coeffs[1] = 20;
    coeffs[2] = -15;
    let quant = [2u16; 64];
    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "right-half-zero block mismatch");
}

#[test]
fn neon_idct_full_block() {
    // All 64 coefficients non-zero (exercises the regular path fully)
    let mut seed: u32 = 0xCAFE_BABE;
    let mut coeffs = [0i16; 64];
    for (i, c) in coeffs.iter_mut().enumerate() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let range = if i == 0 { 80 } else { 20 / (1 + i / 16) };
        let range = range.max(1) as i32;
        *c = ((seed >> 16) as i32 % (2 * range) - range) as i16;
    }
    let quant = [4u16; 64];
    let scalar = scalar_idct(&coeffs, &quant);
    let neon = neon_idct(&coeffs, &quant);
    assert_eq!(neon, scalar, "full block mismatch");
}
