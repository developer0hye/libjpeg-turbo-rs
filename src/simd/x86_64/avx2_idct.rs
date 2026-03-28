//! AVX2-accelerated 8x8 IDCT (accurate integer, "islow").
//!
//! Port of the libjpeg-turbo IDCT algorithm using AVX2 intrinsics.
//! Combines dequantization, IDCT, level-shift (+128), and clamping [0,255]
//! into a single fused operation.
//!
//! Includes sparsity detection (DC-only fast path) matching the NEON
//! implementation for significant speedup on typical JPEG blocks.

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
/// # Safety contract
/// Caller must ensure AVX2 is available (dispatch in `x86_64/mod.rs` verifies this).
pub fn avx2_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // SAFETY: AVX2 availability guaranteed by dispatch in x86_64::routines().
    unsafe {
        avx2_idct_islow_core(coeffs, quant, output.as_mut_ptr(), 8);
    }
}

/// Strided variant: writes 8x8 block directly to `output` with row stride `stride`.
///
/// # Safety
/// `output` must point to at least `7 * stride + 8` writable bytes.
/// Caller must ensure AVX2 is available.
pub unsafe fn avx2_idct_islow_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    avx2_idct_islow_core(coeffs, quant, output, stride);
}

/// Core IDCT with configurable output stride.
///
/// # Safety
/// Requires AVX2. `output` must point to at least `7 * stride + 8` writable bytes.
#[target_feature(enable = "avx2")]
unsafe fn avx2_idct_islow_core(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    // Step 1: Dequantize -- multiply coefficients by quantization table
    let mut rows: [__m128i; 8] = [_mm_setzero_si128(); 8];

    for (i, row) in rows.iter_mut().enumerate() {
        let coeff_row = _mm_loadu_si128(coeffs.as_ptr().add(i * 8) as *const __m128i);
        let quant_row = _mm_loadu_si128(quant.as_ptr().add(i * 8) as *const __m128i);
        *row = _mm_mullo_epi16(coeff_row, quant_row);
    }

    // Sparsity check: DC-only fast path
    // OR all AC rows together, then OR with AC coefficients of row 0
    let ac_bitmap = _mm_or_si128(
        _mm_or_si128(
            _mm_or_si128(rows[1], rows[2]),
            _mm_or_si128(rows[3], rows[4]),
        ),
        _mm_or_si128(_mm_or_si128(rows[5], rows[6]), rows[7]),
    );

    // Check if all AC rows are zero
    if _mm_testz_si128(ac_bitmap, ac_bitmap) != 0 {
        // Also check if row0 has only DC (position 0) non-zero
        // Mask out position 0 to check AC positions 1-7 of row 0
        let ac_mask = _mm_setr_epi16(0, -1, -1, -1, -1, -1, -1, -1);
        let row0_ac = _mm_and_si128(rows[0], ac_mask);

        if _mm_testz_si128(row0_ac, row0_ac) != 0 {
            // Pure DC block: compute DC pixel value and fill
            let dc_coeff = *coeffs.as_ptr() as i32;
            let dc_quant = *quant.as_ptr() as i32;
            let dc_dequant = dc_coeff * dc_quant;
            // IDCT of DC-only: value = (dequant + 4) >> 3, then level-shift +128
            let pixel_val = (((dc_dequant + 4) >> 3) + 128).clamp(0, 255) as u8;

            // Fill all 8 rows with the DC value
            let fill = _mm_set1_epi8(pixel_val as i8);
            for row in 0..8 {
                _mm_storel_epi64(output.add(row * stride) as *mut __m128i, fill);
            }
            return;
        }
    }

    // Step 2: Column pass -- transpose to get columns, run 1-D IDCT
    let transposed = transpose_8x8_i16(rows);
    let col_results = idct_pass_columns(transposed);

    // Step 3: Transpose back for row pass
    let transposed_back = transpose_8x8_i16(col_results);
    let row_results = idct_pass_rows(transposed_back);

    // Step 4: Level-shift (+128) and clamp [0,255], write with stride
    let offset = _mm_set1_epi16(128);

    for i in (0..8).step_by(2) {
        let r0 = _mm_add_epi16(row_results[i], offset);
        let r1 = _mm_add_epi16(row_results[i + 1], offset);
        let packed = _mm_packus_epi16(r0, r1);
        // Store each row separately at stride offsets
        _mm_storel_epi64(output.add(i * stride) as *mut __m128i, packed);
        _mm_storel_epi64(
            output.add((i + 1) * stride) as *mut __m128i,
            _mm_srli_si128(packed, 8),
        );
    }
}

/// Transpose an 8x8 matrix of i16 values stored in 8 __m128i registers.
///
/// # Safety
/// Requires SSE2 (available under AVX2 target feature).
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn transpose_8x8_i16(rows: [__m128i; 8]) -> [__m128i; 8] {
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
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct_pass_columns(s: [__m128i; 8]) -> [__m128i; 8] {
    idct_1d_avx2!(s, 11)
}

/// Row pass: IDCT with descale shift = CONST_BITS + PASS1_BITS + 3 = 18.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct_pass_rows(s: [__m128i; 8]) -> [__m128i; 8] {
    idct_1d_avx2!(s, 18)
}

/// Narrow 8 x i32 in a __m256i to 8 x i16 in a __m128i (with saturation).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow_i32_to_i16(v: __m256i) -> __m128i {
    let zero = _mm256_setzero_si256();
    let packed = _mm256_packs_epi32(v, zero);
    let shuffled = _mm256_permute4x64_epi64::<0b_11_01_10_00>(packed);
    _mm256_castsi256_si128(shuffled)
}
