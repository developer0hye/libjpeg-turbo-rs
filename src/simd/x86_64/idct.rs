//! SSE2-accelerated 8x8 IDCT (accurate integer, "islow").
//!
//! Port of the libjpeg-turbo integer IDCT algorithm using SSE2 intrinsics.
//! Combines dequantization, IDCT, level-shift (+128), and clamping.
//! Includes DC-only sparsity fast path and strided output support.
//!
//! The input coefficients and quantization table are both in natural
//! (row-major) order. We dequantize during load, perform the 2-pass IDCT
//! with 4 columns/rows processed in parallel per __m128i, level-shift by
//! +128, and clamp to [0, 255].
//!
//! Strategy:
//! - DC-only fast path: if all AC coefficients are zero, fill 8x8 block
//!   with a single value (common in flat/smooth areas and low-quality JPEGs)
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
        sse2_idct_islow_core(coeffs, quant, output.as_mut_ptr(), 8);
    }
}

/// # Safety
/// Requires SSE2. `output` must point to at least `stride * 7 + 8` writable bytes.
pub unsafe fn sse2_idct_islow_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    sse2_idct_islow_core(coeffs, quant, output, stride);
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
/// Includes DC-only sparsity fast path and strided output.
///
/// # Safety
/// Requires x86_64 SSE2 support. `output` must point to at least
/// `stride * 7 + 8` writable bytes.
#[target_feature(enable = "sse2")]
unsafe fn sse2_idct_islow_core(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    let cptr: *const i16 = coeffs.as_ptr();

    // --- DC-only sparsity check (SSE2-compatible) ---
    // OR rows 1-7 together. If all zero, only row 0 may have non-zero coefficients.
    let row1: __m128i = _mm_loadu_si128(cptr.add(8) as *const __m128i);
    let row2: __m128i = _mm_loadu_si128(cptr.add(16) as *const __m128i);
    let row3: __m128i = _mm_loadu_si128(cptr.add(24) as *const __m128i);
    let row4: __m128i = _mm_loadu_si128(cptr.add(32) as *const __m128i);
    let row5: __m128i = _mm_loadu_si128(cptr.add(40) as *const __m128i);
    let row6: __m128i = _mm_loadu_si128(cptr.add(48) as *const __m128i);
    let row7: __m128i = _mm_loadu_si128(cptr.add(56) as *const __m128i);

    let ac_or: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_or_si128(row1, row2), _mm_or_si128(row3, row4)),
        _mm_or_si128(_mm_or_si128(row5, row6), row7),
    );

    // SSE2 zero test: cmpeq against zero, then movemask. 0xFFFF means all zero.
    let zero: __m128i = _mm_setzero_si128();
    if _mm_movemask_epi8(_mm_cmpeq_epi8(ac_or, zero)) == 0xFFFF {
        // Rows 1-7 are all zero. Check if row 0 AC coefficients are also zero.
        let row0: __m128i = _mm_loadu_si128(cptr as *const __m128i);
        // Mask out DC (position 0), keep AC (positions 1-7).
        let ac_mask: __m128i = _mm_setr_epi16(0, -1, -1, -1, -1, -1, -1, -1);
        let row0_ac: __m128i = _mm_and_si128(row0, ac_mask);

        if _mm_movemask_epi8(_mm_cmpeq_epi8(row0_ac, zero)) == 0xFFFF {
            // True DC-only: compute fill value and broadcast.
            //
            // Mirror the full-pipeline i16 wrap exactly. The non-shortcut
            // path uses `_mm_mullo_epi16` (i16 truncating multiply) for
            // dequant, then a column shortcut at line 1487 etc. of
            // `pipeline.rs` propagates an i16 dcval through pass 2's i16
            // arithmetic and final i16→i8 saturating narrow + center add.
            // An adversarial DC like 2032 * 85 = 172720 wraps in i16 to
            // -23888, and the rest of the pipeline produces a saturated-low
            // sample. Using the un-wrapped i32 product here returned a
            // saturated-high sample, diverging from the full pipeline +
            // libjpeg-turbo on such inputs (fuzz_decode_diff_c finding
            // 2026-05-09).
            let dq_i16: i16 = (*cptr).wrapping_mul(*(quant.as_ptr() as *const i16));
            let pass1_i32: i32 = (dq_i16 as i32) << PASS1_BITS;
            let pass2_i32: i32 = (pass1_i32 + (1 << (PASS1_BITS + 3 - 1))) >> (PASS1_BITS + 3);
            let pv: u8 = (pass2_i32 + 128).clamp(0, 255) as u8;
            let fill: __m128i = _mm_set1_epi8(pv as i8);
            for r in 0..8 {
                _mm_storel_epi64(output.add(r * stride) as *mut __m128i, fill);
            }
            return;
        }
    }

    // --- Full IDCT path ---

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

        for (c, &res) in result.iter().enumerate() {
            let descaled: __m128i = descale_p2(res);
            let shifted: __m128i = _mm_add_epi32(descaled, center);

            let packed_i16: __m128i = _mm_packs_epi32(shifted, zero);
            let packed_u8: __m128i = _mm_packus_epi16(packed_i16, zero);

            let mut bytes = [0u8; 16];
            _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, packed_u8);

            *output.add((row_base) * stride + c) = bytes[0];
            *output.add((row_base + 1) * stride + c) = bytes[1];
            *output.add((row_base + 2) * stride + c) = bytes[2];
            *output.add((row_base + 3) * stride + c) = bytes[3];
        }
    }
}
