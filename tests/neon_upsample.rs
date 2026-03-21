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

fn scalar_upsample_h2v2(input: &[u8], in_width: usize, in_height: usize) -> Vec<u8> {
    let out_width = in_width * 2;
    let out_height = in_height * 2;
    let mut output = vec![0u8; out_width * out_height];
    upsample::fancy_h2v2(
        input,
        in_width,
        in_height,
        &mut output,
        out_width,
        out_height,
    );
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
    let input = vec![100u8, 200];
    let scalar = scalar_upsample(&input, 2);
    let neon = neon_upsample(&input, 2);
    assert_eq!(neon, scalar, "two samples mismatch");
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
