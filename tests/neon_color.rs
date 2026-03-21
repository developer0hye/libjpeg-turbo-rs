//! NEON YCbCr→pixel color conversion tests.
#![cfg(target_arch = "aarch64")]

use libjpeg_turbo_rs::decode::color;
use libjpeg_turbo_rs::simd;
use libjpeg_turbo_rs::simd::aarch64::color as neon_color;

fn scalar_rgb(y: &[u8], cb: &[u8], cr: &[u8], width: usize) -> Vec<u8> {
    std::env::set_var("JSIMD_FORCENONE", "1");
    let routines = simd::detect();
    std::env::remove_var("JSIMD_FORCENONE");
    let mut rgb = vec![0u8; width * 3];
    (routines.ycbcr_to_rgb_row)(y, cb, cr, &mut rgb, width);
    rgb
}

fn neon_rgb(y: &[u8], cb: &[u8], cr: &[u8], width: usize) -> Vec<u8> {
    let routines = libjpeg_turbo_rs::simd::aarch64::routines();
    let mut rgb = vec![0u8; width * 3];
    (routines.ycbcr_to_rgb_row)(y, cb, cr, &mut rgb, width);
    rgb
}

#[test]
fn neon_color_white() {
    let y = vec![255u8; 32];
    let cb = vec![128u8; 32];
    let cr = vec![128u8; 32];
    let scalar = scalar_rgb(&y, &cb, &cr, 32);
    let neon = neon_rgb(&y, &cb, &cr, 32);
    assert_eq!(neon, scalar, "white mismatch");
}

#[test]
fn neon_color_black() {
    let y = vec![0u8; 32];
    let cb = vec![128u8; 32];
    let cr = vec![128u8; 32];
    let scalar = scalar_rgb(&y, &cb, &cr, 32);
    let neon = neon_rgb(&y, &cb, &cr, 32);
    assert_eq!(neon, scalar, "black mismatch");
}

#[test]
fn neon_color_red() {
    // Pure red in YCbCr: Y≈76, Cb≈84, Cr≈255
    let y = vec![76u8; 32];
    let cb = vec![84u8; 32];
    let cr = vec![255u8; 32];
    let scalar = scalar_rgb(&y, &cb, &cr, 32);
    let neon = neon_rgb(&y, &cb, &cr, 32);
    assert_eq!(neon, scalar, "red mismatch");
}

#[test]
fn neon_color_gradient_sweep() {
    let width = 256;
    let y: Vec<u8> = (0..width).map(|i| i as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| i as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| (255 - i) as u8).collect();
    let scalar = scalar_rgb(&y, &cb, &cr, width);
    let neon = neon_rgb(&y, &cb, &cr, width);
    assert_eq!(neon, scalar, "gradient sweep mismatch");
}

#[test]
fn neon_color_random_1920() {
    let width = 1920;
    let mut seed: u32 = 0x1234_5678;
    let mut gen = || -> u8 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) as u8
    };
    let y: Vec<u8> = (0..width).map(|_| gen()).collect();
    let cb: Vec<u8> = (0..width).map(|_| gen()).collect();
    let cr: Vec<u8> = (0..width).map(|_| gen()).collect();
    let scalar = scalar_rgb(&y, &cb, &cr, width);
    let neon = neon_rgb(&y, &cb, &cr, width);
    assert_eq!(neon, scalar, "random 1920px row mismatch");
}

#[test]
fn neon_color_short_row() {
    // Less than 16 pixels (tail-only)
    let width = 7;
    let y = vec![180u8; width];
    let cb = vec![100u8; width];
    let cr = vec![200u8; width];
    let scalar = scalar_rgb(&y, &cb, &cr, width);
    let neon = neon_rgb(&y, &cb, &cr, width);
    assert_eq!(neon, scalar, "short row (7px) mismatch");
}

#[test]
fn neon_color_exact_16() {
    // Exactly 16 pixels (one full NEON iteration, no tail)
    let width = 16;
    let y: Vec<u8> = (0..width).map(|i| (i * 16) as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| (64 + i * 8) as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| (192 - i * 4) as u8).collect();
    let scalar = scalar_rgb(&y, &cb, &cr, width);
    let neon = neon_rgb(&y, &cb, &cr, width);
    assert_eq!(neon, scalar, "exact 16px mismatch");
}

#[test]
fn neon_color_exact_multiple() {
    // 48 pixels: exact multiple of 16
    let width = 48;
    let y: Vec<u8> = (0..width).map(|i| ((i * 5) % 256) as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| ((i * 3 + 64) % 256) as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| ((i * 7 + 128) % 256) as u8).collect();
    let scalar = scalar_rgb(&y, &cb, &cr, width);
    let neon = neon_rgb(&y, &cb, &cr, width);
    assert_eq!(neon, scalar, "exact multiple (48px) mismatch");
}

#[test]
fn neon_color_single_pixel() {
    let y = vec![128u8];
    let cb = vec![128u8];
    let cr = vec![128u8];
    let scalar = scalar_rgb(&y, &cb, &cr, 1);
    let neon = neon_rgb(&y, &cb, &cr, 1);
    assert_eq!(neon, scalar, "single pixel mismatch");
}

// --- Helper for format-specific NEON vs scalar comparison ---

fn random_ycbcr(width: usize, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut s = seed;
    let mut gen = || -> u8 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (s >> 16) as u8
    };
    let y: Vec<u8> = (0..width).map(|_| gen()).collect();
    let cb: Vec<u8> = (0..width).map(|_| gen()).collect();
    let cr: Vec<u8> = (0..width).map(|_| gen()).collect();
    (y, cb, cr)
}

// --- NEON RGBA tests ---

#[test]
fn neon_rgba_random_1920() {
    let width = 1920;
    let (y, cb, cr) = random_ycbcr(width, 0xAAAA_BBBB);
    let mut scalar = vec![0u8; width * 4];
    let mut neon = vec![0u8; width * 4];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut scalar, width);
    neon_color::neon_ycbcr_to_rgba_row(&y, &cb, &cr, &mut neon, width);
    assert_eq!(neon, scalar, "RGBA random 1920px mismatch");
}

#[test]
fn neon_rgba_short_tail() {
    let width = 7;
    let (y, cb, cr) = random_ycbcr(width, 0x1111);
    let mut scalar = vec![0u8; width * 4];
    let mut neon = vec![0u8; width * 4];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut scalar, width);
    neon_color::neon_ycbcr_to_rgba_row(&y, &cb, &cr, &mut neon, width);
    assert_eq!(neon, scalar, "RGBA short tail mismatch");
}

// --- NEON BGR tests ---

#[test]
fn neon_bgr_random_1920() {
    let width = 1920;
    let (y, cb, cr) = random_ycbcr(width, 0xCCCC_DDDD);
    let mut scalar = vec![0u8; width * 3];
    let mut neon = vec![0u8; width * 3];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut scalar, width);
    neon_color::neon_ycbcr_to_bgr_row(&y, &cb, &cr, &mut neon, width);
    assert_eq!(neon, scalar, "BGR random 1920px mismatch");
}

#[test]
fn neon_bgr_short_tail() {
    let width = 13;
    let (y, cb, cr) = random_ycbcr(width, 0x2222);
    let mut scalar = vec![0u8; width * 3];
    let mut neon = vec![0u8; width * 3];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut scalar, width);
    neon_color::neon_ycbcr_to_bgr_row(&y, &cb, &cr, &mut neon, width);
    assert_eq!(neon, scalar, "BGR short tail mismatch");
}

// --- NEON BGRA tests ---

#[test]
fn neon_bgra_random_1920() {
    let width = 1920;
    let (y, cb, cr) = random_ycbcr(width, 0xEEEE_FFFF);
    let mut scalar = vec![0u8; width * 4];
    let mut neon = vec![0u8; width * 4];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut scalar, width);
    neon_color::neon_ycbcr_to_bgra_row(&y, &cb, &cr, &mut neon, width);
    assert_eq!(neon, scalar, "BGRA random 1920px mismatch");
}

#[test]
fn neon_bgra_short_tail() {
    let width = 3;
    let (y, cb, cr) = random_ycbcr(width, 0x3333);
    let mut scalar = vec![0u8; width * 4];
    let mut neon = vec![0u8; width * 4];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut scalar, width);
    neon_color::neon_ycbcr_to_bgra_row(&y, &cb, &cr, &mut neon, width);
    assert_eq!(neon, scalar, "BGRA short tail mismatch");
}
