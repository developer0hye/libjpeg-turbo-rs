//! P4-191 (#609): the AVX2 arms of the per-block encode helpers in `mcu.rs`,
//! driven directly at their interior-bound edges.
//!
//! `fdct_quantize_block` and `fdct_quantize_chroma_h2v1` each guard a raw-pointer
//! AVX2 kernel with an interior test (`block_x + 8 <= plane_width`, or `+ 16`
//! for the H2V1 chroma window). In the encoder those arms never run: the MCU
//! loops pad their strips so `interior` is always true and the helpers are
//! entered only when AVX2 is absent, which also switches their own AVX2 arm
//! off. So no CI leg executed either arm, and no sanitizer saw their reads.
//!
//! Here each plane is a heap allocation of exactly `plane_width * plane_height`
//! bytes — the size the helpers' callers promise and nothing more — and each
//! block sits on or just beyond the interior edge. On the last interior block
//! the kernel's final row load ends on the allocation's last byte, so a bound
//! loosened by one column reads one byte past it (ASan: heap-buffer-overflow)
//! and, on a leg without ASan, takes the next row's first pixel where the
//! border path would replicate the edge, which the comparison below reports.
//!
//! The reference is written out here rather than borrowed from `sampling.rs`,
//! whose interior paths are themselves SIMD (`extract_block_sse2`,
//! `downsample_chroma_block_h2v1_ssse3`): a scalar statement of the same
//! arithmetic — clamp-to-edge sampling, level shift, the alternating H2V1
//! bias of `jcsample.c` — followed by the scalar islow FDCT.
//!
//! Selected by the `kernel_bounds_tests` filter in
//! `.github/workflows/sanitizers.yml`; keep the module name when moving it.

use super::mcu::{fdct_quantize_block, fdct_quantize_chroma_h2v1};
use super::quant_divisors::scale_quant_for_fdct;
use super::QuantDivisors;
use crate::encode::tables::{
    quality_scale_quant_table, STD_CHROMINANCE_QUANT_TABLE, STD_LUMINANCE_QUANT_TABLE,
};
use crate::simd::scalar::scalar_fdct_quantize;

/// A deterministic textured plane: a diagonal gradient with an XOR pattern, so
/// neighbouring columns and rows differ and a one-pixel misread changes the
/// coefficients.
fn textured_plane(width: usize, height: usize) -> Vec<u8> {
    let mut plane: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            plane.push(((x * 7 + y * 13) ^ (x * y)) as u8);
        }
    }
    assert_eq!(plane.capacity(), width * height, "exact-size allocation");
    plane
}

/// Clamp-to-edge 8x8 extraction with the -128 level shift, then scalar islow
/// FDCT + quantize: what `fdct_quantize_block` must produce on either arm.
fn reference_luma_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
) -> [i16; 64] {
    let mut block: [i16; 64] = [0i16; 64];
    for row in 0..8 {
        let source_y: usize = (block_y + row).min(plane_height - 1);
        for col in 0..8 {
            let source_x: usize = (block_x + col).min(plane_width - 1);
            block[row * 8 + col] = plane[source_y * plane_width + source_x] as i16 - 128;
        }
    }
    let mut out: [i16; 64] = [0i16; 64];
    scalar_fdct_quantize(&mut block, quant, &mut out);
    out
}

/// H2V1 downsample of a 16x8 source window — pairs averaged with a bias that
/// alternates 0, 1, 0, 1 across each output row (`jcsample.c`
/// `h2v1_downsample`), source pixels clamped to the edge — then level shift and
/// scalar islow FDCT + quantize.
fn reference_chroma_h2v1_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
) -> [i16; 64] {
    let mut block: [i16; 64] = [0i16; 64];
    for row in 0..8 {
        let source_y: usize = (block_y + row).min(plane_height - 1);
        let mut bias: u32 = 0;
        for col in 0..8 {
            let mut sum: u32 = 0;
            for dx in 0..2 {
                let source_x: usize = (block_x + col * 2 + dx).min(plane_width - 1);
                sum += plane[source_y * plane_width + source_x] as u32;
            }
            block[row * 8 + col] = ((sum + bias) >> 1) as i16 - 128;
            bias ^= 1;
        }
    }
    let mut out: [i16; 64] = [0i16; 64];
    scalar_fdct_quantize(&mut block, quant, &mut out);
    out
}

fn luma_divisors() -> QuantDivisors {
    scale_quant_for_fdct(&quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 90))
}

fn chroma_divisors() -> QuantDivisors {
    scale_quant_for_fdct(&quality_scale_quant_table(&STD_CHROMINANCE_QUANT_TABLE, 90))
}

fn note_when_avx2_absent() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("NOTE: no AVX2 on this host; only the border arms are compared");
    }
}

/// P4-191: `fdct_quantize_block`'s AVX2 arm (`avx2_extract_fdct_quantize`,
/// eight 8-byte loads at `stride = plane_width`) on exactly sized planes.
///
/// Geometries: the minimum plane, 8x8, whose only block is interior and whose
/// last load ends on the last byte; a 24x16 plane walked block by block to
/// (16, 8), the last interior block; and planes one column (23x16) or one row
/// (24x15) short of that, where the final block is *not* interior and must take
/// the border arm. Loosening either half of the interior test to `+ 7` would
/// send that final block into the kernel, which then reads past the allocation.
#[test]
fn fdct_quantize_block_avx2_arm_matches_scalar_at_the_interior_edge() {
    note_when_avx2_absent();
    let quant: QuantDivisors = luma_divisors();
    for (plane_width, plane_height) in [(8usize, 8usize), (24, 16), (23, 16), (24, 15), (9, 9)] {
        let plane: Vec<u8> = textured_plane(plane_width, plane_height);
        let mut block_y: usize = 0;
        while block_y < plane_height {
            let mut block_x: usize = 0;
            while block_x < plane_width {
                let want: [i16; 64] = reference_luma_block(
                    &plane,
                    plane_width,
                    plane_height,
                    block_x,
                    block_y,
                    &quant,
                );
                let mut got: [i16; 64] = [0i16; 64];
                fdct_quantize_block(
                    &plane,
                    plane_width,
                    plane_height,
                    block_x,
                    block_y,
                    &quant,
                    scalar_fdct_quantize,
                    &mut got,
                );
                assert_eq!(
                    got, want,
                    "{plane_width}x{plane_height} plane, block at ({block_x}, {block_y})"
                );
                block_x += 8;
            }
            block_y += 8;
        }
    }
}

/// P4-191: `fdct_quantize_chroma_h2v1`'s site 1 (`avx2_downsample_h2v1_fdct_quantize`,
/// eight 16-byte loads) on exactly sized planes.
///
/// Geometries: 16x8, the minimum interior window, whose last load ends on the
/// last byte; 48x16 walked to (32, 8), the last interior window; and 47x16 /
/// 48x15, where the final window is one column or one row short and must take
/// the border arm. 4:2:2 chroma windows are 16 source columns wide, so the
/// bound is `block_x + 16 <= plane_width`.
#[test]
fn fdct_quantize_chroma_h2v1_avx2_arm_matches_scalar_at_the_interior_edge() {
    note_when_avx2_absent();
    let quant: QuantDivisors = chroma_divisors();
    for (plane_width, plane_height) in [(16usize, 8usize), (48, 16), (47, 16), (48, 15), (17, 9)] {
        let plane: Vec<u8> = textured_plane(plane_width, plane_height);
        let mut block_y: usize = 0;
        while block_y < plane_height {
            let mut block_x: usize = 0;
            while block_x < plane_width {
                let want: [i16; 64] = reference_chroma_h2v1_block(
                    &plane,
                    plane_width,
                    plane_height,
                    block_x,
                    block_y,
                    &quant,
                );
                let mut got: [i16; 64] = [0i16; 64];
                fdct_quantize_chroma_h2v1(
                    &plane,
                    plane_width,
                    plane_height,
                    block_x,
                    block_y,
                    &quant,
                    scalar_fdct_quantize,
                    &mut got,
                );
                assert_eq!(
                    got, want,
                    "{plane_width}x{plane_height} plane, window at ({block_x}, {block_y})"
                );
                block_x += 16;
            }
            block_y += 8;
        }
    }
}
