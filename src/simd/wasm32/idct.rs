//! WASM simd128-accelerated 8x8 IDCT (accurate integer, "islow").
//!
//! Port of the SSE2 IDCT to WASM simd128 intrinsics.
//! Key difference: WASM has native `i32x4_mul` (SSE2 lacks this),
//! eliminating the `mullo_epi32_sse2` emulation.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;

const F_0_298: i32 = 2446;
const F_0_390: i32 = 3196;
const F_0_541: i32 = 4433;
const F_0_765: i32 = 6270;
const F_0_899: i32 = 7373;
const F_1_175: i32 = 9633;
const F_1_501: i32 = 12299;
const F_1_847: i32 = 15137;
const F_1_961: i32 = 16069;
const F_2_053: i32 = 16819;
const F_2_562: i32 = 20995;
const F_3_072: i32 = 25172;

/// WASM simd128 combined dequant + IDCT + level-shift + clamp.
pub fn wasm_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // SAFETY: simd128 target feature is enabled on the callee via #[target_feature].
    // Input arrays are fixed-size [i16; 64]/[u16; 64], guaranteeing correct length.
    // Output is [u8; 64] with stride=8, satisfying the 64-byte write requirement.
    unsafe {
        wasm_idct_islow_core(coeffs, quant, output.as_mut_ptr(), 8);
    }
}

/// 1-D IDCT on 4 lanes in parallel (i32x4).
#[inline(always)]
fn idct_1d_pass(
    s0: v128,
    s1: v128,
    s2: v128,
    s3: v128,
    s4: v128,
    s5: v128,
    s6: v128,
    s7: v128,
) -> [v128; 8] {
    // Even part
    let s2_plus_s6: v128 = i32x4_add(s2, s6);
    let z1: v128 = i32x4_mul(s2_plus_s6, i32x4_splat(F_0_541));
    let tmp2: v128 = i32x4_add(z1, i32x4_mul(s6, i32x4_splat(-F_1_847)));
    let tmp3: v128 = i32x4_add(z1, i32x4_mul(s2, i32x4_splat(F_0_765)));

    let tmp0: v128 = i32x4_shl(i32x4_add(s0, s4), CONST_BITS);
    let tmp1: v128 = i32x4_shl(i32x4_sub(s0, s4), CONST_BITS);

    let tmp10: v128 = i32x4_add(tmp0, tmp3);
    let tmp13: v128 = i32x4_sub(tmp0, tmp3);
    let tmp11: v128 = i32x4_add(tmp1, tmp2);
    let tmp12: v128 = i32x4_sub(tmp1, tmp2);

    // Odd part
    let z1: v128 = i32x4_add(s7, s1);
    let z2: v128 = i32x4_add(s5, s3);
    let z3: v128 = i32x4_add(s7, s3);
    let z4: v128 = i32x4_add(s5, s1);
    let z5: v128 = i32x4_mul(i32x4_add(z3, z4), i32x4_splat(F_1_175));

    let o0: v128 = i32x4_mul(s7, i32x4_splat(F_0_298));
    let o1: v128 = i32x4_mul(s5, i32x4_splat(F_2_053));
    let o2: v128 = i32x4_mul(s3, i32x4_splat(F_3_072));
    let o3: v128 = i32x4_mul(s1, i32x4_splat(F_1_501));
    let z1: v128 = i32x4_mul(z1, i32x4_splat(-F_0_899));
    let z2: v128 = i32x4_mul(z2, i32x4_splat(-F_2_562));
    let z3: v128 = i32x4_add(i32x4_mul(z3, i32x4_splat(-F_1_961)), z5);
    let z4: v128 = i32x4_add(i32x4_mul(z4, i32x4_splat(-F_0_390)), z5);

    let o0: v128 = i32x4_add(i32x4_add(o0, z1), z3);
    let o1: v128 = i32x4_add(i32x4_add(o1, z2), z4);
    let o2: v128 = i32x4_add(i32x4_add(o2, z2), z3);
    let o3: v128 = i32x4_add(i32x4_add(o3, z1), z4);

    [
        i32x4_add(tmp10, o3),
        i32x4_add(tmp11, o2),
        i32x4_add(tmp12, o1),
        i32x4_add(tmp13, o0),
        i32x4_sub(tmp13, o0),
        i32x4_sub(tmp12, o1),
        i32x4_sub(tmp11, o2),
        i32x4_sub(tmp10, o3),
    ]
}

/// Descale for pass 1: shift right by (CONST_BITS - PASS1_BITS).
#[inline(always)]
fn descale_p1(val: v128) -> v128 {
    let round: v128 = i32x4_splat(1 << (CONST_BITS - PASS1_BITS - 1));
    i32x4_shr(i32x4_add(val, round), CONST_BITS - PASS1_BITS)
}

/// Descale for pass 2: shift right by (CONST_BITS + PASS1_BITS + 3).
#[inline(always)]
fn descale_p2(val: v128) -> v128 {
    let round: v128 = i32x4_splat(1 << (CONST_BITS + PASS1_BITS + 3 - 1));
    i32x4_shr(i32x4_add(val, round), CONST_BITS + PASS1_BITS + 3)
}

/// Transpose helper: unpack low 32-bit elements.
#[inline(always)]
fn unpacklo_epi32(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(a, b)
}

/// Transpose helper: unpack high 32-bit elements.
#[inline(always)]
fn unpackhi_epi32(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(a, b)
}

/// Transpose helper: unpack low 64-bit elements.
#[inline(always)]
fn unpacklo_epi64(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(a, b)
}

/// Transpose helper: unpack high 64-bit elements.
#[inline(always)]
fn unpackhi_epi64(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(a, b)
}

/// Core WASM simd128 IDCT.
#[target_feature(enable = "simd128")]
unsafe fn wasm_idct_islow_core(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    let cptr: *const i16 = coeffs.as_ptr();

    // --- DC-only sparsity check ---
    let row1: v128 = v128_load(cptr.add(8) as *const v128);
    let row2: v128 = v128_load(cptr.add(16) as *const v128);
    let row3: v128 = v128_load(cptr.add(24) as *const v128);
    let row4: v128 = v128_load(cptr.add(32) as *const v128);
    let row5: v128 = v128_load(cptr.add(40) as *const v128);
    let row6: v128 = v128_load(cptr.add(48) as *const v128);
    let row7: v128 = v128_load(cptr.add(56) as *const v128);

    let ac_or: v128 = v128_or(
        v128_or(v128_or(row1, row2), v128_or(row3, row4)),
        v128_or(v128_or(row5, row6), row7),
    );

    let zero: v128 = i32x4_splat(0);
    if u8x16_bitmask(u8x16_eq(ac_or, zero)) == 0xFFFF {
        let row0: v128 = v128_load(cptr as *const v128);
        let ac_mask: v128 = i16x8(0, -1, -1, -1, -1, -1, -1, -1);
        let row0_ac: v128 = v128_and(row0, ac_mask);

        if u8x16_bitmask(u8x16_eq(row0_ac, zero)) == 0xFFFF {
            let dc: i32 = *cptr as i32 * *quant.as_ptr() as i32;
            let pv: u8 = (((dc + 4) >> 3) + 128).clamp(0, 255) as u8;
            for r in 0..8 {
                let row_ptr: *mut u8 = output.add(r * stride);
                for c in 0..8 {
                    *row_ptr.add(c) = pv;
                }
            }
            return;
        }
    }

    // --- Full IDCT path ---
    let mut ws = [0i32; 64];

    // ========== Pass 1: columns ==========
    for col_base in (0..8).step_by(4) {
        let mut rows = [zero; 8];
        for row in 0..8 {
            let idx: usize = row * 8 + col_base;
            let c0: i32 = coeffs[idx] as i32;
            let c1: i32 = coeffs[idx + 1] as i32;
            let c2: i32 = coeffs[idx + 2] as i32;
            let c3: i32 = coeffs[idx + 3] as i32;
            let q0: i32 = quant[idx] as i32;
            let q1: i32 = quant[idx + 1] as i32;
            let q2: i32 = quant[idx + 2] as i32;
            let q3: i32 = quant[idx + 3] as i32;
            rows[row] = i32x4(c0 * q0, c1 * q1, c2 * q2, c3 * q3);
        }

        let result: [v128; 8] = idct_1d_pass(
            rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
        );

        for (row, &res) in result.iter().enumerate() {
            let descaled: v128 = descale_p1(res);
            let idx: usize = row * 8 + col_base;
            v128_store(ws.as_mut_ptr().add(idx) as *mut v128, descaled);
        }
    }

    // ========== Pass 2: rows ==========
    let center: v128 = i32x4_splat(128);

    for row_base in (0..8).step_by(4) {
        let r0: v128 = v128_load(ws.as_ptr().add(row_base * 8) as *const v128);
        let r0h: v128 = v128_load(ws.as_ptr().add(row_base * 8 + 4) as *const v128);
        let r1: v128 = v128_load(ws.as_ptr().add((row_base + 1) * 8) as *const v128);
        let r1h: v128 = v128_load(ws.as_ptr().add((row_base + 1) * 8 + 4) as *const v128);
        let r2: v128 = v128_load(ws.as_ptr().add((row_base + 2) * 8) as *const v128);
        let r2h: v128 = v128_load(ws.as_ptr().add((row_base + 2) * 8 + 4) as *const v128);
        let r3: v128 = v128_load(ws.as_ptr().add((row_base + 3) * 8) as *const v128);
        let r3h: v128 = v128_load(ws.as_ptr().add((row_base + 3) * 8 + 4) as *const v128);

        // Transpose low halves (columns 0-3)
        let t0: v128 = unpacklo_epi32(r0, r1);
        let t1: v128 = unpackhi_epi32(r0, r1);
        let t2: v128 = unpacklo_epi32(r2, r3);
        let t3: v128 = unpackhi_epi32(r2, r3);

        let col0: v128 = unpacklo_epi64(t0, t2);
        let col1: v128 = unpackhi_epi64(t0, t2);
        let col2: v128 = unpacklo_epi64(t1, t3);
        let col3: v128 = unpackhi_epi64(t1, t3);

        // Transpose high halves (columns 4-7)
        let t0h: v128 = unpacklo_epi32(r0h, r1h);
        let t1h: v128 = unpackhi_epi32(r0h, r1h);
        let t2h: v128 = unpacklo_epi32(r2h, r3h);
        let t3h: v128 = unpackhi_epi32(r2h, r3h);

        let col4: v128 = unpacklo_epi64(t0h, t2h);
        let col5: v128 = unpackhi_epi64(t0h, t2h);
        let col6: v128 = unpacklo_epi64(t1h, t3h);
        let col7: v128 = unpackhi_epi64(t1h, t3h);

        let result: [v128; 8] = idct_1d_pass(col0, col1, col2, col3, col4, col5, col6, col7);

        for (c, &res) in result.iter().enumerate() {
            let descaled: v128 = descale_p2(res);
            let shifted: v128 = i32x4_add(descaled, center);

            let packed_i16: v128 = i16x8_narrow_i32x4(shifted, zero);
            let packed_u8: v128 = u8x16_narrow_i16x8(packed_i16, zero);

            *output.add(row_base * stride + c) = u8x16_extract_lane::<0>(packed_u8);
            *output.add((row_base + 1) * stride + c) = u8x16_extract_lane::<1>(packed_u8);
            *output.add((row_base + 2) * stride + c) = u8x16_extract_lane::<2>(packed_u8);
            *output.add((row_base + 3) * stride + c) = u8x16_extract_lane::<3>(packed_u8);
        }
    }
}
