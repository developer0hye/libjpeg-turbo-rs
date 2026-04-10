//! WASM simd128-accelerated forward DCT (accurate integer, "islow").
//!
//! Port of the NEON FDCT to WASM simd128. Processes all 8 rows/columns
//! in parallel using i16x8 vectors, with widening to i32x4 for cosine
//! multiplications.
//!
//! Input: 8x8 block of i16 (level-shifted pixel values, i.e. pixel - 128)
//! Output: 8x8 block of i16 (DCT coefficients in natural row-major order)

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
const DESCALE_P1: u32 = CONST_BITS - PASS1_BITS; // 11
const DESCALE_P2: u32 = CONST_BITS + PASS1_BITS; // 15

// DCT constants (same as IDCT)
const F_0_298: i16 = 2446;
const F_N0_390: i16 = -3196;
const F_0_541: i16 = 4433;
const F_0_765: i16 = 6270;
const F_N0_899: i16 = -7373;
const F_1_175: i16 = 9633;
const F_1_501: i16 = 12299;
const F_N1_847: i16 = -15137;
const F_N1_961: i16 = -16069;
const F_2_053: i16 = 16819;
const F_N2_562: i16 = -20995;
const F_3_072: i16 = 25172;

/// WASM simd128 forward DCT on one 8x8 block.
pub fn wasm_fdct(input: &[i16; 64], output: &mut [i16; 64]) {
    unsafe {
        wasm_fdct_core(input.as_ptr(), output.as_mut_ptr());
    }
}

/// Widening multiply i16×const→i32 for low 4 lanes.
#[inline(always)]
fn wmul_lo(a: v128, c: i16) -> v128 {
    i32x4_extmul_low_i16x8(a, i16x8_splat(c))
}

/// Widening multiply i16×const→i32 for high 4 lanes.
#[inline(always)]
fn wmul_hi(a: v128, c: i16) -> v128 {
    i32x4_extmul_high_i16x8(a, i16x8_splat(c))
}

/// Rounding shift right by N on i32x4, then narrow to i16 (pack low+high halves).
#[inline(always)]
fn rshrn(lo: v128, hi: v128, n: u32) -> v128 {
    let rnd: v128 = i32x4_splat(1 << (n - 1));
    let lo_shifted: v128 = i32x4_shr(i32x4_add(lo, rnd), n);
    let hi_shifted: v128 = i32x4_shr(i32x4_add(hi, rnd), n);
    i16x8_narrow_i32x4(lo_shifted, hi_shifted)
}

// ===== 8x8 i16 transpose helpers =====

#[inline(always)]
fn unpacklo_epi16(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(a, b)
}

#[inline(always)]
fn unpackhi_epi16(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(a, b)
}

#[inline(always)]
fn unpacklo_epi32(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(a, b)
}

#[inline(always)]
fn unpackhi_epi32(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(a, b)
}

#[inline(always)]
fn unpacklo_epi64(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(a, b)
}

#[inline(always)]
fn unpackhi_epi64(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(a, b)
}

/// Transpose 8 i16x8 vectors (8x8 matrix transpose).
#[inline(always)]
fn transpose_8x8(
    r0: v128,
    r1: v128,
    r2: v128,
    r3: v128,
    r4: v128,
    r5: v128,
    r6: v128,
    r7: v128,
) -> [v128; 8] {
    // Stage 1: interleave 16-bit pairs
    let a0: v128 = unpacklo_epi16(r0, r1);
    let a1: v128 = unpackhi_epi16(r0, r1);
    let a2: v128 = unpacklo_epi16(r2, r3);
    let a3: v128 = unpackhi_epi16(r2, r3);
    let a4: v128 = unpacklo_epi16(r4, r5);
    let a5: v128 = unpackhi_epi16(r4, r5);
    let a6: v128 = unpacklo_epi16(r6, r7);
    let a7: v128 = unpackhi_epi16(r6, r7);

    // Stage 2: interleave 32-bit pairs
    let b0: v128 = unpacklo_epi32(a0, a2);
    let b1: v128 = unpackhi_epi32(a0, a2);
    let b2: v128 = unpacklo_epi32(a1, a3);
    let b3: v128 = unpackhi_epi32(a1, a3);
    let b4: v128 = unpacklo_epi32(a4, a6);
    let b5: v128 = unpackhi_epi32(a4, a6);
    let b6: v128 = unpacklo_epi32(a5, a7);
    let b7: v128 = unpackhi_epi32(a5, a7);

    // Stage 3: interleave 64-bit pairs
    [
        unpacklo_epi64(b0, b4),
        unpackhi_epi64(b0, b4),
        unpacklo_epi64(b1, b5),
        unpackhi_epi64(b1, b5),
        unpacklo_epi64(b2, b6),
        unpackhi_epi64(b2, b6),
        unpacklo_epi64(b3, b7),
        unpackhi_epi64(b3, b7),
    ]
}

/// Forward DCT butterfly for one pass (8-wide i16, widens to i32 for cosine multiply).
/// Returns 8 output vectors: data[0], data[1], ..., data[7].
#[inline(always)]
fn fdct_pass(
    s0: v128,
    s1: v128,
    s2: v128,
    s3: v128,
    s4: v128,
    s5: v128,
    s6: v128,
    s7: v128,
    descale: u32,
) -> [v128; 8] {
    let tmp0: v128 = i16x8_add(s0, s7);
    let tmp7: v128 = i16x8_sub(s0, s7);
    let tmp1: v128 = i16x8_add(s1, s6);
    let tmp6: v128 = i16x8_sub(s1, s6);
    let tmp2: v128 = i16x8_add(s2, s5);
    let tmp5: v128 = i16x8_sub(s2, s5);
    let tmp3: v128 = i16x8_add(s3, s4);
    let tmp4: v128 = i16x8_sub(s3, s4);

    // Even part
    let tmp10: v128 = i16x8_add(tmp0, tmp3);
    let tmp13: v128 = i16x8_sub(tmp0, tmp3);
    let tmp11: v128 = i16x8_add(tmp1, tmp2);
    let tmp12: v128 = i16x8_sub(tmp1, tmp2);

    // data[0] and data[4]: pure add/sub, shift left by PASS1_BITS
    let out0: v128 = if descale == DESCALE_P1 {
        i16x8_shl(i16x8_add(tmp10, tmp11), PASS1_BITS)
    } else {
        // Pass 2: no shift needed for output 0 and 4 (the shift is part of descale)
        rshrn(
            i32x4_shl(
                i32x4_extmul_low_i16x8(i16x8_add(tmp10, tmp11), i16x8_splat(1)),
                CONST_BITS,
            ),
            i32x4_shl(
                i32x4_extmul_high_i16x8(i16x8_add(tmp10, tmp11), i16x8_splat(1)),
                CONST_BITS,
            ),
            descale,
        )
    };
    let out4: v128 = if descale == DESCALE_P1 {
        i16x8_shl(i16x8_sub(tmp10, tmp11), PASS1_BITS)
    } else {
        rshrn(
            i32x4_shl(
                i32x4_extmul_low_i16x8(i16x8_sub(tmp10, tmp11), i16x8_splat(1)),
                CONST_BITS,
            ),
            i32x4_shl(
                i32x4_extmul_high_i16x8(i16x8_sub(tmp10, tmp11), i16x8_splat(1)),
                CONST_BITS,
            ),
            descale,
        )
    };

    // data[2] and data[6]: z1 = F_0_541 * (tmp12 + tmp13)
    let t12_add_t13: v128 = i16x8_add(tmp12, tmp13);
    let z1_lo: v128 = wmul_lo(t12_add_t13, F_0_541);
    let z1_hi: v128 = wmul_hi(t12_add_t13, F_0_541);

    let out2: v128 = rshrn(
        i32x4_add(z1_lo, wmul_lo(tmp13, F_0_765)),
        i32x4_add(z1_hi, wmul_hi(tmp13, F_0_765)),
        descale,
    );
    let out6: v128 = rshrn(
        i32x4_add(z1_lo, wmul_lo(tmp12, F_N1_847)),
        i32x4_add(z1_hi, wmul_hi(tmp12, F_N1_847)),
        descale,
    );

    // Odd part
    let z1: v128 = i16x8_add(tmp4, tmp7);
    let z2: v128 = i16x8_add(tmp5, tmp6);
    let z3: v128 = i16x8_add(tmp4, tmp6);
    let z4: v128 = i16x8_add(tmp5, tmp7);

    // z5 = F_1_175 * (z3 + z4)
    let z5_lo: v128 = i32x4_add(wmul_lo(z3, F_1_175), wmul_lo(z4, F_1_175));
    let z5_hi: v128 = i32x4_add(wmul_hi(z3, F_1_175), wmul_hi(z4, F_1_175));

    let z1_lo: v128 = wmul_lo(z1, F_N0_899);
    let z1_hi: v128 = wmul_hi(z1, F_N0_899);
    let z2_lo: v128 = wmul_lo(z2, F_N2_562);
    let z2_hi: v128 = wmul_hi(z2, F_N2_562);
    let z3_lo: v128 = i32x4_add(wmul_lo(z3, F_N1_961), z5_lo);
    let z3_hi: v128 = i32x4_add(wmul_hi(z3, F_N1_961), z5_hi);
    let z4_lo: v128 = i32x4_add(wmul_lo(z4, F_N0_390), z5_lo);
    let z4_hi: v128 = i32x4_add(wmul_hi(z4, F_N0_390), z5_hi);

    let out7: v128 = rshrn(
        i32x4_add(i32x4_add(wmul_lo(tmp4, F_0_298), z1_lo), z3_lo),
        i32x4_add(i32x4_add(wmul_hi(tmp4, F_0_298), z1_hi), z3_hi),
        descale,
    );
    let out5: v128 = rshrn(
        i32x4_add(i32x4_add(wmul_lo(tmp5, F_2_053), z2_lo), z4_lo),
        i32x4_add(i32x4_add(wmul_hi(tmp5, F_2_053), z2_hi), z4_hi),
        descale,
    );
    let out3: v128 = rshrn(
        i32x4_add(i32x4_add(wmul_lo(tmp6, F_3_072), z2_lo), z3_lo),
        i32x4_add(i32x4_add(wmul_hi(tmp6, F_3_072), z2_hi), z3_hi),
        descale,
    );
    let out1: v128 = rshrn(
        i32x4_add(i32x4_add(wmul_lo(tmp7, F_1_501), z1_lo), z4_lo),
        i32x4_add(i32x4_add(wmul_hi(tmp7, F_1_501), z1_hi), z4_hi),
        descale,
    );

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_fdct_core(input: *const i16, output: *mut i16) {
    // Load 8 rows
    let r0: v128 = v128_load(input as *const v128);
    let r1: v128 = v128_load(input.add(8) as *const v128);
    let r2: v128 = v128_load(input.add(16) as *const v128);
    let r3: v128 = v128_load(input.add(24) as *const v128);
    let r4: v128 = v128_load(input.add(32) as *const v128);
    let r5: v128 = v128_load(input.add(40) as *const v128);
    let r6: v128 = v128_load(input.add(48) as *const v128);
    let r7: v128 = v128_load(input.add(56) as *const v128);

    // Transpose: rows → columns
    let cols: [v128; 8] = transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

    // Pass 1: process rows (descale by CONST_BITS - PASS1_BITS)
    let pass1: [v128; 8] = fdct_pass(
        cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], DESCALE_P1,
    );

    // Transpose: columns → rows
    let rows: [v128; 8] = transpose_8x8(
        pass1[0], pass1[1], pass1[2], pass1[3], pass1[4], pass1[5], pass1[6], pass1[7],
    );

    // Pass 2: process columns (descale by CONST_BITS + PASS1_BITS)
    let pass2: [v128; 8] = fdct_pass(
        rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], DESCALE_P2,
    );

    // Store 8 rows
    v128_store(output as *mut v128, pass2[0]);
    v128_store(output.add(8) as *mut v128, pass2[1]);
    v128_store(output.add(16) as *mut v128, pass2[2]);
    v128_store(output.add(24) as *mut v128, pass2[3]);
    v128_store(output.add(32) as *mut v128, pass2[4]);
    v128_store(output.add(40) as *mut v128, pass2[5]);
    v128_store(output.add(48) as *mut v128, pass2[6]);
    v128_store(output.add(56) as *mut v128, pass2[7]);
}
