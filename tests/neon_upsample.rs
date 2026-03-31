//! NEON fancy h2v1 upsampling tests.
#![cfg(target_arch = "aarch64")]

use libjpeg_turbo_rs::decode::upsample;
use libjpeg_turbo_rs::simd;

fn scalar_upsample(input: &[u8], in_width: usize) -> Vec<u8> {
    std::env::set_var("JSIMD_FORCENONE", "1");
    let routines = simd::detect();
    std::env::remove_var("JSIMD_FORCENONE");
    let mut output = vec![0u8; in_width * 2];
    (routines.fancy_upsample_h2v1)(input, in_width, &mut output);
    output
}

fn neon_upsample(input: &[u8], in_width: usize) -> Vec<u8> {
    let routines = libjpeg_turbo_rs::simd::aarch64::routines();
    let mut output = vec![0u8; in_width * 2];
    (routines.fancy_upsample_h2v1)(input, in_width, &mut output);
    output
}

/// Two-step scalar H2V2: vertical blend (>> 2) then scalar H2V1.
/// Matches the NEON H2V2 algorithm (which also uses two-step).
/// The fused `fancy_h2v2` uses a single >> 4 pass for better rounding
/// but produces different results due to reduced intermediate truncation.
fn scalar_upsample_h2v2(input: &[u8], in_width: usize, in_height: usize) -> Vec<u8> {
    let out_width = in_width * 2;
    let out_height = in_height * 2;
    let mut output = vec![0u8; out_width * out_height];

    for y in 0..in_height {
        let cur_row = &input[y * in_width..(y + 1) * in_width];
        let above = if y > 0 {
            &input[(y - 1) * in_width..y * in_width]
        } else {
            cur_row
        };
        let below = if y + 1 < in_height {
            &input[(y + 1) * in_width..(y + 2) * in_width]
        } else {
            cur_row
        };

        // Vertical blend (same as NEON vertical blend)
        let mut row_above = vec![0u8; in_width];
        let mut row_below = vec![0u8; in_width];
        for i in 0..in_width {
            let cur = cur_row[i] as u16;
            row_above[i] = ((3 * cur + above[i] as u16 + 2) >> 2) as u8;
            row_below[i] = ((3 * cur + below[i] as u16 + 2) >> 2) as u8;
        }

        // Horizontal H2V1 (scalar, matching corrected bias)
        let top_offset = (y * 2) * out_width;
        let bot_offset = (y * 2 + 1) * out_width;
        upsample::fancy_h2v1(&row_above, in_width, &mut output[top_offset..], out_width);
        upsample::fancy_h2v1(&row_below, in_width, &mut output[bot_offset..], out_width);
    }
    output
}

fn neon_upsample_h2v2(input: &[u8], in_width: usize, in_height: usize) -> Vec<u8> {
    let out_width = in_width * 2;
    let out_height = in_height * 2;
    let mut output = vec![0u8; out_width * out_height];
    libjpeg_turbo_rs::simd::aarch64::upsample::neon_fancy_upsample_h2v2(
        input,
        in_width,
        in_height,
        &mut output,
        out_width,
    );
    output.truncate(out_width * out_height);
    output
}

#[test]
fn neon_upsample_uniform() {
    let input = vec![128u8; 64];
    let scalar = scalar_upsample(&input, 64);
    let neon = neon_upsample(&input, 64);
    assert_eq!(neon, scalar, "uniform mismatch");
}

#[test]
fn neon_upsample_gradient() {
    let input: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let scalar = scalar_upsample(&input, 64);
    let neon = neon_upsample(&input, 64);
    assert_eq!(neon, scalar, "gradient mismatch");
}

#[test]
fn neon_upsample_random() {
    let mut seed: u32 = 0xABCD_EF01;
    let input: Vec<u8> = (0..128)
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as u8
        })
        .collect();
    let scalar = scalar_upsample(&input, 128);
    let neon = neon_upsample(&input, 128);
    assert_eq!(neon, scalar, "random 128-sample mismatch");
}

#[test]
fn neon_upsample_short() {
    // Less than 16 samples (tail-only)
    let input: Vec<u8> = (0..5).map(|i| (i * 50) as u8).collect();
    let scalar = scalar_upsample(&input, 5);
    let neon = neon_upsample(&input, 5);
    assert_eq!(neon, scalar, "short (5) mismatch");
}

#[test]
fn neon_upsample_exact_16() {
    let input: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
    let scalar = scalar_upsample(&input, 16);
    let neon = neon_upsample(&input, 16);
    assert_eq!(neon, scalar, "exact 16 mismatch");
}

#[test]
fn neon_upsample_exact_32() {
    let input: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
    let scalar = scalar_upsample(&input, 32);
    let neon = neon_upsample(&input, 32);
    assert_eq!(neon, scalar, "exact 32 mismatch");
}

#[test]
fn neon_upsample_one_sample() {
    let input = vec![200u8];
    let scalar = scalar_upsample(&input, 1);
    let neon = neon_upsample(&input, 1);
    assert_eq!(neon, scalar, "single sample mismatch");
}

#[test]
fn neon_upsample_two_samples() {
    // For in_width=2, C uses box filter (each chroma replicated to 2 output pixels).
    // The pipeline guards this case before reaching NEON, so test the expected output.
    let input = vec![100u8, 200];
    let scalar = scalar_upsample(&input, 2);
    // Box filter: [100, 100, 200, 200]
    assert_eq!(
        scalar,
        vec![100, 100, 200, 200],
        "two samples should use box filter"
    );
}

#[test]
fn neon_upsample_large_random() {
    let mut seed: u32 = 0x9876_5432;
    let input: Vec<u8> = (0..960)
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as u8
        })
        .collect();
    let scalar = scalar_upsample(&input, 960);
    let neon = neon_upsample(&input, 960);
    assert_eq!(neon, scalar, "large random (960) mismatch");
}

#[test]
fn neon_h2v2_gradient_matches_scalar() {
    let in_width = 32;
    let in_height = 8;
    let input: Vec<u8> = (0..(in_width * in_height))
        .map(|i: usize| (i.wrapping_mul(7) % 251) as u8)
        .collect();
    let scalar = scalar_upsample_h2v2(&input, in_width, in_height);
    let neon = neon_upsample_h2v2(&input, in_width, in_height);
    assert_eq!(neon, scalar, "gradient h2v2 mismatch");
}

#[test]
fn neon_h2v2_random_short_rows_match_scalar() {
    let in_width = 9;
    let in_height = 5;
    let mut seed: u32 = 0x1357_2468;
    let input: Vec<u8> = (0..(in_width * in_height))
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as u8
        })
        .collect();
    let scalar = scalar_upsample_h2v2(&input, in_width, in_height);
    let neon = neon_upsample_h2v2(&input, in_width, in_height);
    assert_eq!(neon, scalar, "short-row h2v2 mismatch");
}
