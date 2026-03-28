//! AVX2-accelerated 8x8 IDCT (accurate integer, "islow").
//!
//! Port of the libjpeg-turbo IDCT algorithm using 256-bit AVX2 intrinsics.
//! Combines dequantization, IDCT, level-shift (+128), and clamping [0,255]
//! into a single fused operation.
//!
//! Strategy: load 8x8 block as 8 x __m128i rows, widen to i32 using AVX2
//! `_mm256_cvtepi16_epi32` for arithmetic, then narrow back to i16.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// IDCT constants matching libjpeg-turbo's jidctint.c
const CONST_BITS: i32 = 13;
#[allow(dead_code)]
const PASS1_BITS: i32 = 2;

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

/// AVX2-accelerated combined dequant + IDCT + level-shift + clamp.
///
/// `coeffs`: 64 i16 coefficients in natural (row-major) order.
/// `quant`: 64 u16 quantization values in natural (row-major) order.
/// `output`: 64 u8 samples in natural (row-major) order.
///
/// # Safety contract
/// Caller must ensure AVX2 is available (dispatch in `x86_64/mod.rs` verifies this).
pub fn avx2_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // SAFETY: AVX2 availability guaranteed by dispatch in x86_64::routines().
    unsafe {
        avx2_idct_islow_inner(coeffs, quant, output);
    }
}

/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
unsafe fn avx2_idct_islow_inner(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // Step 1: Dequantize -- multiply coefficients by quantization table
    let mut rows: [__m128i; 8] = [_mm_setzero_si128(); 8];

    for (i, row) in rows.iter_mut().enumerate() {
        let coeff_row = _mm_loadu_si128(coeffs.as_ptr().add(i * 8) as *const __m128i);
        let quant_row = _mm_loadu_si128(quant.as_ptr().add(i * 8) as *const __m128i);
        *row = _mm_mullo_epi16(coeff_row, quant_row);
    }

    // Step 2: Column pass -- transpose to get columns, run 1-D IDCT
    let transposed = transpose_8x8_i16(rows);
    let col_results = idct_pass_columns(transposed);

    // Step 3: Transpose back for row pass
    let transposed_back = transpose_8x8_i16(col_results);
    let row_results = idct_pass_rows(transposed_back);

    // Step 4: Level-shift (+128) and clamp [0,255]
    let offset = _mm_set1_epi16(128);
    let mut final_rows: [__m128i; 8] = [_mm_setzero_si128(); 8];

    for i in 0..8 {
        final_rows[i] = _mm_add_epi16(row_results[i], offset);
    }

    // Pack pairs of rows from i16 to u8 using _mm_packus_epi16 (saturating)
    for i in (0..8).step_by(2) {
        let packed = _mm_packus_epi16(final_rows[i], final_rows[i + 1]);
        _mm_storeu_si128(output.as_mut_ptr().add(i * 8) as *mut __m128i, packed);
    }
}

/// Transpose an 8x8 matrix of i16 values stored in 8 __m128i registers.
///
/// # Safety
/// Requires SSE2 (available under AVX2 target feature).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8_i16(rows: [__m128i; 8]) -> [__m128i; 8] {
    // Step 1: Interleave 16-bit values from pairs of rows
    let t0 = _mm_unpacklo_epi16(rows[0], rows[1]);
    let t1 = _mm_unpackhi_epi16(rows[0], rows[1]);
    let t2 = _mm_unpacklo_epi16(rows[2], rows[3]);
    let t3 = _mm_unpackhi_epi16(rows[2], rows[3]);
    let t4 = _mm_unpacklo_epi16(rows[4], rows[5]);
    let t5 = _mm_unpackhi_epi16(rows[4], rows[5]);
    let t6 = _mm_unpacklo_epi16(rows[6], rows[7]);
    let t7 = _mm_unpackhi_epi16(rows[6], rows[7]);

    // Step 2: Interleave 32-bit pairs
    let u0 = _mm_unpacklo_epi32(t0, t2);
    let u1 = _mm_unpackhi_epi32(t0, t2);
    let u2 = _mm_unpacklo_epi32(t1, t3);
    let u3 = _mm_unpackhi_epi32(t1, t3);
    let u4 = _mm_unpacklo_epi32(t4, t6);
    let u5 = _mm_unpackhi_epi32(t4, t6);
    let u6 = _mm_unpacklo_epi32(t5, t7);
    let u7 = _mm_unpackhi_epi32(t5, t7);

    // Step 3: Interleave 64-bit pairs
    [
        _mm_unpacklo_epi64(u0, u4),
        _mm_unpackhi_epi64(u0, u4),
        _mm_unpacklo_epi64(u1, u5),
        _mm_unpackhi_epi64(u1, u5),
        _mm_unpacklo_epi64(u2, u6),
        _mm_unpackhi_epi64(u2, u6),
        _mm_unpacklo_epi64(u3, u7),
        _mm_unpackhi_epi64(u3, u7),
    ]
}

/// Core 1-D IDCT arithmetic using AVX2 i32 operations.
///
/// `_mm256_srai_epi32` requires a compile-time constant for the immediate operand,
/// so we use a macro to generate code with the shift baked in.
macro_rules! idct_1d_avx2 {
    ($s:expr, $shift:literal) => {{
        let s = $s;

        // Widen all inputs from i16 to i32
        let s0 = _mm256_cvtepi16_epi32(s[0]);
        let s1 = _mm256_cvtepi16_epi32(s[1]);
        let s2 = _mm256_cvtepi16_epi32(s[2]);
        let s3 = _mm256_cvtepi16_epi32(s[3]);
        let s4 = _mm256_cvtepi16_epi32(s[4]);
        let s5 = _mm256_cvtepi16_epi32(s[5]);
        let s6 = _mm256_cvtepi16_epi32(s[6]);
        let s7 = _mm256_cvtepi16_epi32(s[7]);

        // Even part: z1 = (s2 + s6) * F_0_541
        let z1 = _mm256_mullo_epi32(_mm256_add_epi32(s2, s6), _mm256_set1_epi32(F_0_541));
        let tmp2a = _mm256_add_epi32(z1, _mm256_mullo_epi32(s6, _mm256_set1_epi32(-F_1_847)));
        let tmp3a = _mm256_add_epi32(z1, _mm256_mullo_epi32(s2, _mm256_set1_epi32(F_0_765)));

        let tmp0a = _mm256_slli_epi32::<CONST_BITS>(_mm256_add_epi32(s0, s4));
        let tmp1a = _mm256_slli_epi32::<CONST_BITS>(_mm256_sub_epi32(s0, s4));

        let tmp10 = _mm256_add_epi32(tmp0a, tmp3a);
        let tmp13 = _mm256_sub_epi32(tmp0a, tmp3a);
        let tmp11 = _mm256_add_epi32(tmp1a, tmp2a);
        let tmp12 = _mm256_sub_epi32(tmp1a, tmp2a);

        // Odd part
        let z1o = _mm256_add_epi32(s7, s1);
        let z2o = _mm256_add_epi32(s5, s3);
        let z3o = _mm256_add_epi32(s7, s3);
        let z4o = _mm256_add_epi32(s5, s1);
        let z5 = _mm256_mullo_epi32(_mm256_add_epi32(z3o, z4o), _mm256_set1_epi32(F_1_175));

        let ot0 = _mm256_mullo_epi32(s7, _mm256_set1_epi32(F_0_298));
        let ot1 = _mm256_mullo_epi32(s5, _mm256_set1_epi32(F_2_053));
        let ot2 = _mm256_mullo_epi32(s3, _mm256_set1_epi32(F_3_072));
        let ot3 = _mm256_mullo_epi32(s1, _mm256_set1_epi32(F_1_501));
        let z1f = _mm256_mullo_epi32(z1o, _mm256_set1_epi32(-F_0_899));
        let z2f = _mm256_mullo_epi32(z2o, _mm256_set1_epi32(-F_2_562));
        let z3f = _mm256_add_epi32(_mm256_mullo_epi32(z3o, _mm256_set1_epi32(-F_1_961)), z5);
        let z4f = _mm256_add_epi32(_mm256_mullo_epi32(z4o, _mm256_set1_epi32(-F_0_390)), z5);

        let ot0 = _mm256_add_epi32(_mm256_add_epi32(ot0, z1f), z3f);
        let ot1 = _mm256_add_epi32(_mm256_add_epi32(ot1, z2f), z4f);
        let ot2 = _mm256_add_epi32(_mm256_add_epi32(ot2, z2f), z3f);
        let ot3 = _mm256_add_epi32(_mm256_add_epi32(ot3, z1f), z4f);

        // Combine and descale with rounding
        let round = _mm256_set1_epi32(1i32 << ($shift - 1));

        let r0 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_add_epi32(tmp10, ot3), round));
        let r1 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_add_epi32(tmp11, ot2), round));
        let r2 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_add_epi32(tmp12, ot1), round));
        let r3 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_add_epi32(tmp13, ot0), round));
        let r4 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_sub_epi32(tmp13, ot0), round));
        let r5 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_sub_epi32(tmp12, ot1), round));
        let r6 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_sub_epi32(tmp11, ot2), round));
        let r7 = _mm256_srai_epi32::<$shift>(_mm256_add_epi32(_mm256_sub_epi32(tmp10, ot3), round));

        // Narrow i32 -> i16
        [
            narrow_i32_to_i16(r0),
            narrow_i32_to_i16(r1),
            narrow_i32_to_i16(r2),
            narrow_i32_to_i16(r3),
            narrow_i32_to_i16(r4),
            narrow_i32_to_i16(r5),
            narrow_i32_to_i16(r6),
            narrow_i32_to_i16(r7),
        ]
    }};
}

/// Column pass: IDCT with descale shift = CONST_BITS - PASS1_BITS = 11.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct_pass_columns(s: [__m128i; 8]) -> [__m128i; 8] {
    idct_1d_avx2!(s, 11)
}

/// Row pass: IDCT with descale shift = CONST_BITS + PASS1_BITS + 3 = 18.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct_pass_rows(s: [__m128i; 8]) -> [__m128i; 8] {
    idct_1d_avx2!(s, 18)
}

/// Narrow 8 x i32 in a __m256i to 8 x i16 in a __m128i (with saturation).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow_i32_to_i16(v: __m256i) -> __m128i {
    // _mm256_packs_epi32 operates on 128-bit lanes independently:
    //   low128:  packs lo128(a), lo128(b) -> 8 x i16
    //   high128: packs hi128(a), hi128(b) -> 8 x i16
    //
    // With packs_epi32(v, zero):
    //   lane0: [v[0] v[1] v[2] v[3] | 0 0 0 0]  (i16)
    //   lane1: [v[4] v[5] v[6] v[7] | 0 0 0 0]  (i16)
    //
    // Quadwords: [q0=v[0..3], q1=zeros, q2=v[4..7], q3=zeros]
    // We want [v[0..3] v[4..7]] in a __m128i -> bring q0 and q2 together.
    let zero = _mm256_setzero_si256();
    let packed = _mm256_packs_epi32(v, zero);
    // permute4x64: position 0=q0, position 1=q2 -> 0b_11_01_10_00
    let shuffled = _mm256_permute4x64_epi64::<0b_11_01_10_00>(packed);
    _mm256_castsi256_si128(shuffled)
}
