//! WASM simd128-accelerated merged upsample + YCbCr → RGB color conversion.
//!
//! For H2V1 (4:2:2) and H2V2 (4:2:0), computes chroma deltas once per Cb/Cr
//! sample and applies to 2 (H2V1) or 4 (H2V2) luma pixels, eliminating
//! intermediate upsample buffers.
//!
//! Uses the same SSE2-style fixed-point constants as `color.rs`:
//!   PW_F0402  = 26345  (FIX(0.40200))
//!   PW_MF0228 = -14942 (-FIX(0.22800))
//!   PW_MF0344 = -22554 (-FIX(0.34414))
//!   PW_F0285  = 18734  (FIX(0.28586))

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

const PW_F0402: i16 = 26345;
const PW_MF0228: i16 = -14942;
const PW_MF0344: i16 = -22554;
const PW_F0285: i16 = 18734;

/// Emulate `_mm_mulhi_epi16`: signed multiply, return high 16 bits.
#[inline(always)]
fn mulhi_epi16(a: v128, b: v128) -> v128 {
    let lo: v128 = i32x4_extmul_low_i16x8(a, b);
    let hi: v128 = i32x4_extmul_high_i16x8(a, b);
    let lo_shifted: v128 = i32x4_shr(lo, 16);
    let hi_shifted: v128 = i32x4_shr(hi, 16);
    i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(
        lo_shifted, hi_shifted,
    )
}

/// Compute chroma deltas (r_delta, g_delta, b_delta) from 8 Cb/Cr samples.
/// Returns (r_sub_y, g_sub_y, b_sub_y) as i16x8 vectors.
#[inline(always)]
fn compute_chroma_deltas(cb_centered: v128, cr_centered: v128) -> (v128, v128, v128) {
    let one: v128 = i16x8_splat(1);

    // R-Y = Cr + round(mulhi(2*Cr, F_0_402))
    let cr2: v128 = i16x8_add(cr_centered, cr_centered);
    let r_mul: v128 = mulhi_epi16(cr2, i16x8_splat(PW_F0402));
    let r_sub_y: v128 = i16x8_add(cr_centered, i16x8_shr(i16x8_add(r_mul, one), 1));

    // G-Y using i32x4_dot_i16x8 for fused multiply-add
    let coeff: v128 =
        i32x4_splat(((PW_F0285 as u16 as u32) << 16 | (PW_MF0344 as u16 as u32)) as i32);
    let cb_cr_lo: v128 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
        cb_centered,
        cr_centered,
    );
    let cb_cr_hi: v128 =
        i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
            cb_centered,
            cr_centered,
        );
    let g_lo_32: v128 = i32x4_dot_i16x8(cb_cr_lo, coeff);
    let g_hi_32: v128 = i32x4_dot_i16x8(cb_cr_hi, coeff);
    let one_half: v128 = i32x4_splat(1 << 15);
    let g_lo_shifted: v128 = i32x4_shr(i32x4_add(g_lo_32, one_half), 16);
    let g_hi_shifted: v128 = i32x4_shr(i32x4_add(g_hi_32, one_half), 16);
    let g_packed: v128 = i16x8_narrow_i32x4(g_lo_shifted, g_hi_shifted);
    let g_sub_y: v128 = i16x8_sub(g_packed, cr_centered);

    // B-Y = 2*Cb + round(mulhi(2*Cb, MF_0_228))
    let cb2: v128 = i16x8_add(cb_centered, cb_centered);
    let b_mul: v128 = mulhi_epi16(cb2, i16x8_splat(PW_MF0228));
    let b_sub_y: v128 = i16x8_add(cb2, i16x8_shr(i16x8_add(b_mul, one), 1));

    (r_sub_y, g_sub_y, b_sub_y)
}

/// Apply chroma deltas to 8 Y samples, producing clamped u8 R, G, B vectors.
#[inline(always)]
fn apply_chroma_to_y(
    y_u8: v128,
    r_sub_y: v128,
    g_sub_y: v128,
    b_sub_y: v128,
) -> (v128, v128, v128) {
    let y16: v128 = u16x8_extend_low_u8x16(y_u8);
    let zero: v128 = i32x4_splat(0);

    let r16: v128 = i16x8_add(y16, r_sub_y);
    let g16: v128 = i16x8_add(y16, g_sub_y);
    let b16: v128 = i16x8_add(y16, b_sub_y);

    // Saturating narrow i16 → u8 (clamps to 0..255)
    (
        u8x16_narrow_i16x8(r16, zero),
        u8x16_narrow_i16x8(g16, zero),
        u8x16_narrow_i16x8(b16, zero),
    )
}

/// Store 8 interleaved RGB pixels (24 bytes) from R, G, B u8 vectors.
/// Uses temp arrays + scalar interleave (3-byte-per-pixel doesn't align to v128).
#[inline(always)]
unsafe fn store_rgb_8pixels(r_u8: v128, g_u8: v128, b_u8: v128, out: *mut u8) {
    let mut r_buf = [0u8; 16];
    let mut g_buf = [0u8; 16];
    let mut b_buf = [0u8; 16];
    v128_store(r_buf.as_mut_ptr() as *mut v128, r_u8);
    v128_store(g_buf.as_mut_ptr() as *mut v128, g_u8);
    v128_store(b_buf.as_mut_ptr() as *mut v128, b_u8);
    for i in 0..8 {
        *out.add(i * 3) = r_buf[i];
        *out.add(i * 3 + 1) = g_buf[i];
        *out.add(i * 3 + 2) = b_buf[i];
    }
}

/// WASM simd128 merged H2V1 upsample + YCbCr→RGB.
///
/// Y is full width, Cb/Cr are half width. Each Cb/Cr sample covers
/// 2 horizontal Y pixels (box-filter replication).
pub fn wasm_merged_h2v1_ycbcr_to_rgb(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out: &mut [u8],
    width: usize,
) {
    // SAFETY: Caller guarantees y_row.len() >= width, cb_row.len() >= width/2,
    // cr_row.len() >= width/2, rgb_out.len() >= width * 3. The inner function
    // processes 16 pixels per SIMD iteration with a scalar tail, preventing
    // out-of-bounds access. simd128 target_feature is enabled on the callee.
    unsafe {
        wasm_merged_h2v1_inner(y_row, cb_row, cr_row, rgb_out, width);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_merged_h2v1_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out: &mut [u8],
    width: usize,
) {
    let offset_128: v128 = i16x8_splat(128);

    let y_ptr: *const u8 = y_row.as_ptr();
    let cb_ptr: *const u8 = cb_row.as_ptr();
    let cr_ptr: *const u8 = cr_row.as_ptr();
    let out_ptr: *mut u8 = rgb_out.as_mut_ptr();

    let mut cols_remaining: usize = width;
    let mut y_offset: usize = 0;
    let mut c_offset: usize = 0;
    let mut out_offset: usize = 0;

    // Main loop: 16 output pixels (8 chroma samples) per iteration
    while cols_remaining >= 16 {
        // Load 8 Cb/Cr samples, widen and center
        let cb_raw: v128 =
            u16x8_extend_low_u8x16(v128_load64_zero(cb_ptr.add(c_offset) as *const u64));
        let cr_raw: v128 =
            u16x8_extend_low_u8x16(v128_load64_zero(cr_ptr.add(c_offset) as *const u64));
        let cb_c: v128 = i16x8_sub(cb_raw, offset_128);
        let cr_c: v128 = i16x8_sub(cr_raw, offset_128);

        // Compute chroma deltas once for 8 chroma samples
        let (r_sub_y, g_sub_y, b_sub_y) = compute_chroma_deltas(cb_c, cr_c);

        // Load 16 Y pixels as two groups of 8 (even positions, odd positions)
        // Y layout: Y0 Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y8 Y9 Y10 Y11 Y12 Y13 Y14 Y15
        // Chroma:   C0  C0  C1  C1  C2  C2  C3  C3  C4   C4   C5   C5   C6   C6   C7   C7
        // Even Y (indices 0,2,4,...,14) gets chroma samples 0,1,2,...,7
        // Odd  Y (indices 1,3,5,...,15) gets chroma samples 0,1,2,...,7
        let y_full: v128 = v128_load(y_ptr.add(y_offset) as *const v128);

        // De-interleave even/odd: even = Y[0],Y[2],Y[4],...,Y[14], odd = Y[1],Y[3],...,Y[15]
        let y_even: v128 =
            i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15>(y_full, y_full);
        // y_even low 8 bytes = even positions, high 8 bytes = odd positions
        // We need: y_even_8 = low 8 bytes as v128 (zero-extended), y_odd_8 = high 8 bytes

        // Apply chroma to even Y pixels (positions 0,2,4,...,14)
        let (r_even, g_even, b_even) = apply_chroma_to_y(y_even, r_sub_y, g_sub_y, b_sub_y);

        // For odd Y, shift the odd bytes into the low position
        let y_odd: v128 =
            i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(y_even, y_even);
        let (r_odd, g_odd, b_odd) = apply_chroma_to_y(y_odd, r_sub_y, g_sub_y, b_sub_y);

        // Re-interleave even and odd pixels: R[0],R[1],R[2],R[3],...
        let r_interleaved: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r_even, r_odd);
        let g_interleaved: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(g_even, g_odd);
        let b_interleaved: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(b_even, b_odd);

        // Store 16 RGB pixels (48 bytes) — use two 8-pixel stores
        let out_base: *mut u8 = out_ptr.add(out_offset);

        // First 8 pixels (bytes 0..23): low 8 of each interleaved vector
        store_rgb_8pixels(r_interleaved, g_interleaved, b_interleaved, out_base);

        // Second 8 pixels (bytes 24..47): high 8 of each interleaved vector
        let r_hi: v128 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(
            r_interleaved,
            r_interleaved,
        );
        let g_hi: v128 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(
            g_interleaved,
            g_interleaved,
        );
        let b_hi: v128 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(
            b_interleaved,
            b_interleaved,
        );
        store_rgb_8pixels(r_hi, g_hi, b_hi, out_base.add(24));

        y_offset += 16;
        c_offset += 8;
        out_offset += 48;
        cols_remaining -= 16;
    }

    // Scalar tail
    if cols_remaining > 0 {
        crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb(
            &y_row[y_offset..y_offset + cols_remaining],
            &cb_row[c_offset..c_offset + cols_remaining.div_ceil(2)],
            &cr_row[c_offset..c_offset + cols_remaining.div_ceil(2)],
            &mut rgb_out[out_offset..out_offset + cols_remaining * 3],
            cols_remaining,
        );
    }
}

/// WASM simd128 merged H2V2 upsample + YCbCr→RGB.
///
/// Processes two output rows at once. Each Cb/Cr sample covers a 2x2
/// block of luma pixels. Computes chroma deltas once per 2x2 block.
pub fn wasm_merged_h2v2_ycbcr_to_rgb(
    y_row0: &[u8],
    y_row1: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out0: &mut [u8],
    rgb_out1: &mut [u8],
    width: usize,
) {
    // SAFETY: Caller guarantees y_row0/y_row1.len() >= width, cb_row/cr_row.len() >= width/2,
    // rgb_out0/rgb_out1.len() >= width * 3. The inner function processes 16 pixels per SIMD
    // iteration with a scalar tail, preventing out-of-bounds access.
    unsafe {
        wasm_merged_h2v2_inner(y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width);
    }
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn wasm_merged_h2v2_inner(
    y_row0: &[u8],
    y_row1: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out0: &mut [u8],
    rgb_out1: &mut [u8],
    width: usize,
) {
    let offset_128: v128 = i16x8_splat(128);

    let y0_ptr: *const u8 = y_row0.as_ptr();
    let y1_ptr: *const u8 = y_row1.as_ptr();
    let cb_ptr: *const u8 = cb_row.as_ptr();
    let cr_ptr: *const u8 = cr_row.as_ptr();
    let out0_ptr: *mut u8 = rgb_out0.as_mut_ptr();
    let out1_ptr: *mut u8 = rgb_out1.as_mut_ptr();

    let mut cols_remaining: usize = width;
    let mut y_offset: usize = 0;
    let mut c_offset: usize = 0;
    let mut out_offset: usize = 0;

    while cols_remaining >= 16 {
        // Load and center 8 Cb/Cr samples
        let cb_raw: v128 =
            u16x8_extend_low_u8x16(v128_load64_zero(cb_ptr.add(c_offset) as *const u64));
        let cr_raw: v128 =
            u16x8_extend_low_u8x16(v128_load64_zero(cr_ptr.add(c_offset) as *const u64));
        let cb_c: v128 = i16x8_sub(cb_raw, offset_128);
        let cr_c: v128 = i16x8_sub(cr_raw, offset_128);

        let (r_sub_y, g_sub_y, b_sub_y) = compute_chroma_deltas(cb_c, cr_c);

        // Process row 0
        {
            let y_full: v128 = v128_load(y0_ptr.add(y_offset) as *const v128);
            let y_deinterleaved: v128 =
                i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15>(
                    y_full, y_full,
                );
            let (r_even, g_even, b_even) =
                apply_chroma_to_y(y_deinterleaved, r_sub_y, g_sub_y, b_sub_y);
            let y_odd: v128 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(
                y_deinterleaved,
                y_deinterleaved,
            );
            let (r_odd, g_odd, b_odd) = apply_chroma_to_y(y_odd, r_sub_y, g_sub_y, b_sub_y);

            let r_int: v128 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                r_even, r_odd,
            );
            let g_int: v128 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                g_even, g_odd,
            );
            let b_int: v128 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                b_even, b_odd,
            );

            let out_base: *mut u8 = out0_ptr.add(out_offset);
            store_rgb_8pixels(r_int, g_int, b_int, out_base);
            let r_hi: v128 =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(r_int, r_int);
            let g_hi: v128 =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(g_int, g_int);
            let b_hi: v128 =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(b_int, b_int);
            store_rgb_8pixels(r_hi, g_hi, b_hi, out_base.add(24));
        }

        // Process row 1 (same chroma deltas)
        {
            let y_full: v128 = v128_load(y1_ptr.add(y_offset) as *const v128);
            let y_deinterleaved: v128 =
                i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15>(
                    y_full, y_full,
                );
            let (r_even, g_even, b_even) =
                apply_chroma_to_y(y_deinterleaved, r_sub_y, g_sub_y, b_sub_y);
            let y_odd: v128 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(
                y_deinterleaved,
                y_deinterleaved,
            );
            let (r_odd, g_odd, b_odd) = apply_chroma_to_y(y_odd, r_sub_y, g_sub_y, b_sub_y);

            let r_int: v128 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                r_even, r_odd,
            );
            let g_int: v128 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                g_even, g_odd,
            );
            let b_int: v128 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                b_even, b_odd,
            );

            let out_base: *mut u8 = out1_ptr.add(out_offset);
            store_rgb_8pixels(r_int, g_int, b_int, out_base);
            let r_hi: v128 =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(r_int, r_int);
            let g_hi: v128 =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(g_int, g_int);
            let b_hi: v128 =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(b_int, b_int);
            store_rgb_8pixels(r_hi, g_hi, b_hi, out_base.add(24));
        }

        y_offset += 16;
        c_offset += 8;
        out_offset += 48;
        cols_remaining -= 16;
    }

    // Scalar tail
    if cols_remaining > 0 {
        let tail_chroma_w: usize = cols_remaining.div_ceil(2);
        crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb(
            &y_row0[y_offset..y_offset + cols_remaining],
            &y_row1[y_offset..y_offset + cols_remaining],
            &cb_row[c_offset..c_offset + tail_chroma_w],
            &cr_row[c_offset..c_offset + tail_chroma_w],
            &mut rgb_out0[out_offset..out_offset + cols_remaining * 3],
            &mut rgb_out1[out_offset..out_offset + cols_remaining * 3],
            cols_remaining,
        );
    }
}
