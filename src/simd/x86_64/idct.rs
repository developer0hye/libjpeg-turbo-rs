//! SSE2-accelerated 8x8 IDCT (accurate integer, "islow").
//!
//! Port of the libjpeg-turbo integer IDCT algorithm using SSE2 intrinsics.
//! Combines dequantization, IDCT, level-shift (+128), and clamping.
//!
//! The input coefficients and quantization table are both in natural
//! (row-major) order. We dequantize during load, perform the 2-pass IDCT
//! with 4 columns/rows processed in parallel per __m128i, level-shift by
//! +128, and clamp to [0, 255].
//!
//! Strategy:
//! - Pass 1 (columns): process columns 0-3 then 4-7 as 4-wide i32 SIMD
//! - Pass 2 (rows): process rows using the same 4-wide approach after
//!   transposing the workspace
//! - Final: level-shift, pack to u8

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const CONST_BITS: i32 = 13;
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

/// SSE2-accelerated combined dequant + IDCT + level-shift + clamp.
///
/// Safe wrapper matching the `SimdRoutines::idct_islow` signature.
pub fn sse2_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // SAFETY: SSE2 is verified at dispatch time via `is_x86_feature_detected!`.
    unsafe {
        sse2_idct_islow_inner(coeffs, quant, output);
    }
}

/// SSE2 does not have `_mm_mullo_epi32` (SSE4.1). Emulate by extracting
/// the low 32 bits from 64-bit unsigned products. The low 32 bits are
/// identical for signed and unsigned multiplication.
#[inline(always)]
unsafe fn mullo_epi32_sse2(a: __m128i, b: __m128i) -> __m128i {
    let mul02: __m128i = _mm_mul_epu32(a, b);
    let a_odd: __m128i = _mm_srli_si128(a, 4);
    let b_odd: __m128i = _mm_srli_si128(b, 4);
    let mul13: __m128i = _mm_mul_epu32(a_odd, b_odd);
    let lo02: __m128i = _mm_shuffle_epi32(mul02, 0b00_00_10_00);
    let lo13: __m128i = _mm_shuffle_epi32(mul13, 0b00_00_10_00);
    _mm_unpacklo_epi32(lo02, lo13)
}

/// Perform 1-D IDCT on 4 lanes in parallel (i32x4).
///
/// `s0..s7` are the 8 frequency-domain inputs, each an __m128i with 4 parallel values.
/// Returns 8 outputs (spatial domain), still needing descale.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn idct_1d_pass(
    s0: __m128i,
    s1: __m128i,
    s2: __m128i,
    s3: __m128i,
    s4: __m128i,
    s5: __m128i,
    s6: __m128i,
    s7: __m128i,
) -> [__m128i; 8] {
    // Even part
    let s2_plus_s6: __m128i = _mm_add_epi32(s2, s6);
    let z1: __m128i = mullo_epi32_sse2(s2_plus_s6, _mm_set1_epi32(F_0_541));
    let tmp2: __m128i = _mm_add_epi32(z1, mullo_epi32_sse2(s6, _mm_set1_epi32(-F_1_847)));
    let tmp3: __m128i = _mm_add_epi32(z1, mullo_epi32_sse2(s2, _mm_set1_epi32(F_0_765)));

    let tmp0: __m128i = _mm_slli_epi32(_mm_add_epi32(s0, s4), CONST_BITS);
    let tmp1: __m128i = _mm_slli_epi32(_mm_sub_epi32(s0, s4), CONST_BITS);

    let tmp10: __m128i = _mm_add_epi32(tmp0, tmp3);
    let tmp13: __m128i = _mm_sub_epi32(tmp0, tmp3);
    let tmp11: __m128i = _mm_add_epi32(tmp1, tmp2);
    let tmp12: __m128i = _mm_sub_epi32(tmp1, tmp2);

    // Odd part
    let z1: __m128i = _mm_add_epi32(s7, s1);
    let z2: __m128i = _mm_add_epi32(s5, s3);
    let z3: __m128i = _mm_add_epi32(s7, s3);
    let z4: __m128i = _mm_add_epi32(s5, s1);
    let z5: __m128i = mullo_epi32_sse2(_mm_add_epi32(z3, z4), _mm_set1_epi32(F_1_175));

    let o0: __m128i = mullo_epi32_sse2(s7, _mm_set1_epi32(F_0_298));
    let o1: __m128i = mullo_epi32_sse2(s5, _mm_set1_epi32(F_2_053));
    let o2: __m128i = mullo_epi32_sse2(s3, _mm_set1_epi32(F_3_072));
    let o3: __m128i = mullo_epi32_sse2(s1, _mm_set1_epi32(F_1_501));
    let z1: __m128i = mullo_epi32_sse2(z1, _mm_set1_epi32(-F_0_899));
    let z2: __m128i = mullo_epi32_sse2(z2, _mm_set1_epi32(-F_2_562));
    let z3: __m128i = _mm_add_epi32(mullo_epi32_sse2(z3, _mm_set1_epi32(-F_1_961)), z5);
    let z4: __m128i = _mm_add_epi32(mullo_epi32_sse2(z4, _mm_set1_epi32(-F_0_390)), z5);

    let o0: __m128i = _mm_add_epi32(_mm_add_epi32(o0, z1), z3);
    let o1: __m128i = _mm_add_epi32(_mm_add_epi32(o1, z2), z4);
    let o2: __m128i = _mm_add_epi32(_mm_add_epi32(o2, z2), z3);
    let o3: __m128i = _mm_add_epi32(_mm_add_epi32(o3, z1), z4);

    [
        _mm_add_epi32(tmp10, o3),
        _mm_add_epi32(tmp11, o2),
        _mm_add_epi32(tmp12, o1),
        _mm_add_epi32(tmp13, o0),
        _mm_sub_epi32(tmp13, o0),
        _mm_sub_epi32(tmp12, o1),
        _mm_sub_epi32(tmp11, o2),
        _mm_sub_epi32(tmp10, o3),
    ]
}

/// Descale (round-towards-nearest) for pass 1: shift right by (CONST_BITS - PASS1_BITS).
#[inline(always)]
unsafe fn descale_p1(val: __m128i) -> __m128i {
    let round: __m128i = _mm_set1_epi32(1 << (CONST_BITS - PASS1_BITS - 1));
    _mm_srai_epi32(_mm_add_epi32(val, round), CONST_BITS - PASS1_BITS)
}

/// Descale for pass 2: shift right by (CONST_BITS + PASS1_BITS + 3).
#[inline(always)]
unsafe fn descale_p2(val: __m128i) -> __m128i {
    let round: __m128i = _mm_set1_epi32(1 << (CONST_BITS + PASS1_BITS + 3 - 1));
    _mm_srai_epi32(_mm_add_epi32(val, round), CONST_BITS + PASS1_BITS + 3)
}

/// Core SSE2 IDCT: dequant + 2-pass IDCT + level-shift + clamp.
///
/// # Safety
/// Requires x86_64 SSE2 support.
#[target_feature(enable = "sse2")]
unsafe fn sse2_idct_islow_inner(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // Workspace: 8x8 i32 values in row-major order.
    let mut ws = [0i32; 64];

    // ========== Pass 1: columns ==========
    // Process 4 columns at a time. Each __m128i holds [col+0, col+1, col+2, col+3].
    for col_base in (0..8).step_by(4) {
        let mut rows = [_mm_setzero_si128(); 8];
        for (row, row_val) in rows.iter_mut().enumerate() {
            let idx: usize = row * 8 + col_base;
            let c0: i32 = coeffs[idx] as i32;
            let c1: i32 = coeffs[idx + 1] as i32;
            let c2: i32 = coeffs[idx + 2] as i32;
            let c3: i32 = coeffs[idx + 3] as i32;
            let q0: i32 = quant[idx] as i32;
            let q1: i32 = quant[idx + 1] as i32;
            let q2: i32 = quant[idx + 2] as i32;
            let q3: i32 = quant[idx + 3] as i32;
            *row_val = _mm_set_epi32(c3 * q3, c2 * q2, c1 * q1, c0 * q0);
        }

        let result: [__m128i; 8] = idct_1d_pass(
            rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
        );

        for (row, &res) in result.iter().enumerate() {
            let descaled: __m128i = descale_p1(res);
            let idx: usize = row * 8 + col_base;
            _mm_storeu_si128(ws.as_mut_ptr().add(idx) as *mut __m128i, descaled);
        }
    }

    // ========== Pass 2: rows ==========
    let center: __m128i = _mm_set1_epi32(128);

    for row_base in (0..8).step_by(4) {
        let r0: __m128i = _mm_loadu_si128(ws.as_ptr().add(row_base * 8) as *const __m128i);
        let r0h: __m128i = _mm_loadu_si128(ws.as_ptr().add(row_base * 8 + 4) as *const __m128i);
        let r1: __m128i = _mm_loadu_si128(ws.as_ptr().add((row_base + 1) * 8) as *const __m128i);
        let r1h: __m128i =
            _mm_loadu_si128(ws.as_ptr().add((row_base + 1) * 8 + 4) as *const __m128i);
        let r2: __m128i = _mm_loadu_si128(ws.as_ptr().add((row_base + 2) * 8) as *const __m128i);
        let r2h: __m128i =
            _mm_loadu_si128(ws.as_ptr().add((row_base + 2) * 8 + 4) as *const __m128i);
        let r3: __m128i = _mm_loadu_si128(ws.as_ptr().add((row_base + 3) * 8) as *const __m128i);
        let r3h: __m128i =
            _mm_loadu_si128(ws.as_ptr().add((row_base + 3) * 8 + 4) as *const __m128i);

        // Transpose low halves (columns 0-3)
        let t0: __m128i = _mm_unpacklo_epi32(r0, r1);
        let t1: __m128i = _mm_unpackhi_epi32(r0, r1);
        let t2: __m128i = _mm_unpacklo_epi32(r2, r3);
        let t3: __m128i = _mm_unpackhi_epi32(r2, r3);

        let col0: __m128i = _mm_unpacklo_epi64(t0, t2);
        let col1: __m128i = _mm_unpackhi_epi64(t0, t2);
        let col2: __m128i = _mm_unpacklo_epi64(t1, t3);
        let col3: __m128i = _mm_unpackhi_epi64(t1, t3);

        // Transpose high halves (columns 4-7)
        let t0h: __m128i = _mm_unpacklo_epi32(r0h, r1h);
        let t1h: __m128i = _mm_unpackhi_epi32(r0h, r1h);
        let t2h: __m128i = _mm_unpacklo_epi32(r2h, r3h);
        let t3h: __m128i = _mm_unpackhi_epi32(r2h, r3h);

        let col4: __m128i = _mm_unpacklo_epi64(t0h, t2h);
        let col5: __m128i = _mm_unpackhi_epi64(t0h, t2h);
        let col6: __m128i = _mm_unpacklo_epi64(t1h, t3h);
        let col7: __m128i = _mm_unpackhi_epi64(t1h, t3h);

        let result: [__m128i; 8] = idct_1d_pass(col0, col1, col2, col3, col4, col5, col6, col7);

        for c in 0..8 {
            let descaled: __m128i = descale_p2(result[c]);
            let shifted: __m128i = _mm_add_epi32(descaled, center);

            let zero: __m128i = _mm_setzero_si128();
            let packed_i16: __m128i = _mm_packs_epi32(shifted, zero);
            let packed_u8: __m128i = _mm_packus_epi16(packed_i16, zero);

            let mut bytes = [0u8; 16];
            _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, packed_u8);

            output[(row_base) * 8 + c] = bytes[0];
            output[(row_base + 1) * 8 + c] = bytes[1];
            output[(row_base + 2) * 8 + c] = bytes[2];
            output[(row_base + 3) * 8 + c] = bytes[3];
        }
    }
}
