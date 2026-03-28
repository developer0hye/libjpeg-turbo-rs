//! AVX2-accelerated 8x8 IDCT (accurate integer, "islow").
//!
//! Direct port of libjpeg-turbo's `jidctint-avx2.asm`.
//! Processes all 8 columns in 4 ymm registers using `vpmaddwd`.
//! Includes DC-only sparsity fast path.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// IDCT constants (CONST_BITS=13)
const F_0_298: i16 = 2446;
const F_0_390: i16 = 3196;
const F_0_541: i16 = 4433;
const F_0_765: i16 = 6270;
const F_0_899: i16 = 7373;
const F_1_175: i16 = 9633;
const F_1_501: i16 = 12299;
const F_1_847: i16 = 15137;
const F_1_961: i16 = 16069;
const F_2_053: i16 = 16819;
const F_2_562: i16 = 20995;
const F_3_072: i16 = 25172;

// Pre-combined vpmaddwd constant tables — exact layout matching C NASM.
// Each `times 4 dw A, B` in NASM = 4 repeats of (A, B) as i16 pairs in 128-bit lane.
// ymm = [lower 128 bits | upper 128 bits], each lane has 4 pairs.
//
// PW_F130_F054_MF130_F054:
//   lower 128: times 4 dw (F_0_541+F_0_765), F_0_541  => tmp3 constants
//   upper 128: times 4 dw (F_0_541-F_1_847), F_0_541  => tmp2 constants
#[repr(align(32))]
struct AlignedI16x16([i16; 16]);

static PW_F130_F054_MF130_F054: AlignedI16x16 = AlignedI16x16([
    // lower 128: (F_0_541+F_0_765), F_0_541 repeated 4 times
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    // upper 128: (F_0_541-F_1_847), F_0_541 repeated 4 times
    (F_0_541 - F_1_847) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
]);

// PW_MF078_F117_F078_F117:
//   lower 128: times 4 dw (F_1_175-F_1_961), F_1_175  => z3 constants
//   upper 128: times 4 dw (F_1_175-F_0_390), F_1_175  => z4 constants
static PW_MF078_F117_F078_F117: AlignedI16x16 = AlignedI16x16([
    (F_1_175 - F_1_961) as i16,
    F_1_175,
    (F_1_175 - F_1_961) as i16,
    F_1_175,
    (F_1_175 - F_1_961) as i16,
    F_1_175,
    (F_1_175 - F_1_961) as i16,
    F_1_175,
    (F_1_175 - F_0_390) as i16,
    F_1_175,
    (F_1_175 - F_0_390) as i16,
    F_1_175,
    (F_1_175 - F_0_390) as i16,
    F_1_175,
    (F_1_175 - F_0_390) as i16,
    F_1_175,
]);

// PW_MF060_MF089_MF050_MF256:
//   lower 128: times 4 dw (F_0_298-F_0_899), -F_0_899  => tmp0 constants
//   upper 128: times 4 dw (F_2_053-F_2_562), -F_2_562  => tmp1 constants
static PW_MF060_MF089_MF050_MF256: AlignedI16x16 = AlignedI16x16([
    (F_0_298 - F_0_899) as i16,
    -F_0_899,
    (F_0_298 - F_0_899) as i16,
    -F_0_899,
    (F_0_298 - F_0_899) as i16,
    -F_0_899,
    (F_0_298 - F_0_899) as i16,
    -F_0_899,
    (F_2_053 - F_2_562) as i16,
    -F_2_562,
    (F_2_053 - F_2_562) as i16,
    -F_2_562,
    (F_2_053 - F_2_562) as i16,
    -F_2_562,
    (F_2_053 - F_2_562) as i16,
    -F_2_562,
]);

// PW_MF089_F060_MF256_F050:
//   lower 128: times 4 dw -F_0_899, (F_1_501-F_0_899)  => tmp3 constants
//   upper 128: times 4 dw -F_2_562, (F_3_072-F_2_562)  => tmp2 constants
static PW_MF089_F060_MF256_F050: AlignedI16x16 = AlignedI16x16([
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
]);

// PW_1_NEG1: lower 128 = 1,1,1,...  upper 128 = -1,-1,-1,...
static PW_1_NEG1: AlignedI16x16 =
    AlignedI16x16([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]);

/// DODCT macro: 1-D IDCT with compile-time descale constant.
macro_rules! dodct {
    ($in04:expr, $in31:expr, $in26:expr, $in75:expr, $descale:literal) => {{
        dodct_inner::<$descale>($in04, $in31, $in26, $in75)
    }};
}

pub fn avx2_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    unsafe { avx2_idct_islow_core(coeffs, quant, output.as_mut_ptr(), 8) }
}

pub unsafe fn avx2_idct_islow_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    avx2_idct_islow_core(coeffs, quant, output, stride);
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_idct_islow_core(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    let cptr = coeffs.as_ptr();

    // --- DC-only sparsity check ---
    let row1 = _mm_loadu_si128(cptr.add(8) as *const __m128i);
    let row2 = _mm_loadu_si128(cptr.add(16) as *const __m128i);
    let row3 = _mm_loadu_si128(cptr.add(24) as *const __m128i);
    let row4 = _mm_loadu_si128(cptr.add(32) as *const __m128i);
    let row5 = _mm_loadu_si128(cptr.add(40) as *const __m128i);
    let row6 = _mm_loadu_si128(cptr.add(48) as *const __m128i);
    let row7 = _mm_loadu_si128(cptr.add(56) as *const __m128i);

    let ac_or = _mm_or_si128(
        _mm_or_si128(_mm_or_si128(row1, row2), _mm_or_si128(row3, row4)),
        _mm_or_si128(_mm_or_si128(row5, row6), row7),
    );

    if _mm_testz_si128(ac_or, ac_or) != 0 {
        let row0 = _mm_loadu_si128(cptr as *const __m128i);
        let ac_mask = _mm_setr_epi16(0, -1, -1, -1, -1, -1, -1, -1);
        let row0_ac = _mm_and_si128(row0, ac_mask);

        if _mm_testz_si128(row0_ac, row0_ac) != 0 {
            let dc = *cptr as i32 * *quant.as_ptr() as i32;
            let pv = (((dc + 4) >> 3) + 128).clamp(0, 255) as u8;
            let fill = _mm_set1_epi8(pv as i8);
            for r in 0..8 {
                _mm_storel_epi64(output.add(r * stride) as *mut __m128i, fill);
            }
            return;
        }
    }

    // --- Load & dequantize: 2 rows per ymm ---
    let qptr = quant.as_ptr() as *const __m256i;
    let ymm4 = _mm256_mullo_epi16(
        _mm256_loadu_si256(cptr as *const __m256i),
        _mm256_loadu_si256(qptr),
    );
    let ymm5 = _mm256_mullo_epi16(
        _mm256_loadu_si256(cptr.add(16) as *const __m256i),
        _mm256_loadu_si256(qptr.cast::<i16>().add(16) as *const __m256i),
    );
    let ymm6 = _mm256_mullo_epi16(
        _mm256_loadu_si256(cptr.add(32) as *const __m256i),
        _mm256_loadu_si256(qptr.cast::<i16>().add(32) as *const __m256i),
    );
    let ymm7 = _mm256_mullo_epi16(
        _mm256_loadu_si256(cptr.add(48) as *const __m256i),
        _mm256_loadu_si256(qptr.cast::<i16>().add(48) as *const __m256i),
    );

    // Rearrange: ymm4=in0_1, ymm5=in2_3, ymm6=in4_5, ymm7=in6_7
    // → ymm0=in0_4, ymm1=in3_1, ymm2=in2_6, ymm3=in7_5
    let ymm0 = _mm256_permute2x128_si256(ymm4, ymm6, 0x20); // lo(in0_1), lo(in4_5) = in0_4
    let ymm1 = _mm256_permute2x128_si256(ymm5, ymm4, 0x31); // hi(in2_3), hi(in0_1) = in3_1
    let ymm2 = _mm256_permute2x128_si256(ymm5, ymm7, 0x20); // lo(in2_3), lo(in6_7) = in2_6
    let ymm3 = _mm256_permute2x128_si256(ymm7, ymm6, 0x31); // hi(in6_7), hi(in4_5) = in7_5

    // --- Pass 1: columns (descale = 11) ---
    let (d0, d1, d2, d3) = dodct!(ymm0, ymm1, ymm2, ymm3, 11);

    // --- Transpose ---
    let (t0, t1, t2, t3) = dotranspose(d0, d1, d2, d3);

    // --- Pass 2: rows (descale = 18) ---
    // t0=data0_4, t1=data1_5, t2=data2_6, t3=data3_7
    // DODCT wants: in0_4, in3_1, in2_6, in7_5
    let p2_in1 = _mm256_permute2x128_si256(t3, t1, 0x20); // lo(data3_7), lo(data1_5) = in3_1
    let p2_in3 = _mm256_permute2x128_si256(t3, t1, 0x31); // hi(data3_7), hi(data1_5) = in7_5
    let (r0, r1, r2, r3) = dodct!(t0, p2_in1, t2, p2_in3, 18);

    // --- Transpose again ---
    let (f0, f1, f2, f3) = dotranspose(r0, r1, r2, r3);

    // --- Pack i16→i8 + level-shift + strided store ---
    let center = _mm256_set1_epi8(-128i8); // 0x80

    let packed01 = _mm256_add_epi8(_mm256_packs_epi16(f0, f1), center);
    let packed23 = _mm256_add_epi8(_mm256_packs_epi16(f2, f3), center);

    // Extract 128-bit halves
    let lo01 = _mm256_castsi256_si128(packed01);
    let hi01 = _mm256_extracti128_si256::<1>(packed01);
    let lo23 = _mm256_castsi256_si128(packed23);
    let hi23 = _mm256_extracti128_si256::<1>(packed23);

    // Each xmm has 2 packed rows: [rowA(8B) | rowB(8B)]
    // pshufd 0x4E swaps the two 64-bit halves to extract rowB
    _mm_storel_epi64(output as *mut __m128i, lo01);
    _mm_storel_epi64(
        output.add(stride) as *mut __m128i,
        _mm_shuffle_epi32(lo01, 0x4E),
    );
    _mm_storel_epi64(output.add(2 * stride) as *mut __m128i, lo23);
    _mm_storel_epi64(
        output.add(3 * stride) as *mut __m128i,
        _mm_shuffle_epi32(lo23, 0x4E),
    );
    _mm_storel_epi64(output.add(4 * stride) as *mut __m128i, hi01);
    _mm_storel_epi64(
        output.add(5 * stride) as *mut __m128i,
        _mm_shuffle_epi32(hi01, 0x4E),
    );
    _mm_storel_epi64(output.add(6 * stride) as *mut __m128i, hi23);
    _mm_storel_epi64(
        output.add(7 * stride) as *mut __m128i,
        _mm_shuffle_epi32(hi23, 0x4E),
    );
}

/// 1-D IDCT matching libjpeg-turbo DODCT macro.
/// Input: in0_4, in3_1, in2_6, in7_5 (each ymm holds 2 column-pairs).
/// Output: data0_1, data3_2, data4_5, data7_6.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dodct_inner<const DESCALE: i32>(
    in04: __m256i, // %1
    in31: __m256i, // %2
    in26: __m256i, // %3
    in75: __m256i, // %4
) -> (__m256i, __m256i, __m256i, __m256i) {
    // --- Even part ---
    // %6 = vperm2i128(%3, %3, 0x01) => in6_2
    let in62 = _mm256_permute2x128_si256(in26, in26, 0x01);
    // %5 = vpunpcklwd(%3, %6) => in26_62L
    let in26_62l = _mm256_unpacklo_epi16(in26, in62);
    // %6 = vpunpckhwd(%3, %6) => in26_62H
    let in26_62h = _mm256_unpackhi_epi16(in26, in62);
    // %5 = vpmaddwd(%5, PW_F130_F054_MF130_F054) => tmp3_2L
    let tmp32_l = _mm256_madd_epi16(
        in26_62l,
        _mm256_load_si256(PW_F130_F054_MF130_F054.0.as_ptr() as *const __m256i),
    );
    // %6 = vpmaddwd(%6, PW_F130_F054_MF130_F054) => tmp3_2H
    let tmp32_h = _mm256_madd_epi16(
        in26_62h,
        _mm256_load_si256(PW_F130_F054_MF130_F054.0.as_ptr() as *const __m256i),
    );

    // %7 = vperm2i128(%1, %1, 0x01) => in4_0
    let in40 = _mm256_permute2x128_si256(in04, in04, 0x01);
    // %1 = vpsignw(%1, PW_1_NEG1) => negate upper half
    let in04_signed = _mm256_sign_epi16(
        in04,
        _mm256_load_si256(PW_1_NEG1.0.as_ptr() as *const __m256i),
    );
    // %7 = vpaddw(%7, %1) => (in0+in4)_(in0-in4)
    let sum_diff = _mm256_add_epi16(in40, in04_signed);

    // Widen to i32 and shift left by CONST_BITS (=13)
    // vpunpcklwd(zero, %7) then vpsrad by (16-13)=3
    let zero = _mm256_setzero_si256();
    let tmp01_l = _mm256_srai_epi32::<3>(_mm256_unpacklo_epi16(zero, sum_diff));
    let tmp01_h = _mm256_srai_epi32::<3>(_mm256_unpackhi_epi16(zero, sum_diff));

    // tmp10_11 = tmp0_1 + tmp3_2; tmp13_12 = tmp0_1 - tmp3_2
    let tmp1011_l = _mm256_add_epi32(tmp01_l, tmp32_l);
    let tmp1011_h = _mm256_add_epi32(tmp01_h, tmp32_h);
    let tmp1312_l = _mm256_sub_epi32(tmp01_l, tmp32_l);
    let tmp1312_h = _mm256_sub_epi32(tmp01_h, tmp32_h);

    // --- Odd part ---
    // %1 = vpaddw(%4, %2) = in7_5 + in3_1 = z3_4
    let z34 = _mm256_add_epi16(in75, in31);

    // z4_3 = vperm2i128(z3_4, z3_4, 0x01)
    let z43 = _mm256_permute2x128_si256(z34, z34, 0x01);
    let z3443_l = _mm256_unpacklo_epi16(z34, z43);
    let z3443_h = _mm256_unpackhi_epi16(z34, z43);
    let z34_l = _mm256_madd_epi16(
        z3443_l,
        _mm256_load_si256(PW_MF078_F117_F078_F117.0.as_ptr() as *const __m256i),
    );
    let z34_h = _mm256_madd_epi16(
        z3443_h,
        _mm256_load_si256(PW_MF078_F117_F078_F117.0.as_ptr() as *const __m256i),
    );

    // in1_3 = vperm2i128(in3_1, in3_1, 0x01)
    let in13 = _mm256_permute2x128_si256(in31, in31, 0x01);
    // in71_53L = vpunpcklwd(in7_5, in1_3)
    let in7153_l = _mm256_unpacklo_epi16(in75, in13);
    let in7153_h = _mm256_unpackhi_epi16(in75, in13);

    // tmp0_1 = vpmaddwd(in7153, PW_MF060...) + z3_4
    let t01_l = _mm256_add_epi32(
        _mm256_madd_epi16(
            in7153_l,
            _mm256_load_si256(PW_MF060_MF089_MF050_MF256.0.as_ptr() as *const __m256i),
        ),
        z34_l,
    );
    let t01_h = _mm256_add_epi32(
        _mm256_madd_epi16(
            in7153_h,
            _mm256_load_si256(PW_MF060_MF089_MF050_MF256.0.as_ptr() as *const __m256i),
        ),
        z34_h,
    );

    // tmp3_2 = vpmaddwd(in7153, PW_MF089...) + z4_3
    let z43_l = _mm256_permute2x128_si256(z34_l, z34_l, 0x01);
    let z43_h = _mm256_permute2x128_si256(z34_h, z34_h, 0x01);
    let t32_l = _mm256_add_epi32(
        _mm256_madd_epi16(
            in7153_l,
            _mm256_load_si256(PW_MF089_F060_MF256_F050.0.as_ptr() as *const __m256i),
        ),
        z43_l,
    );
    let t32_h = _mm256_add_epi32(
        _mm256_madd_epi16(
            in7153_h,
            _mm256_load_si256(PW_MF089_F060_MF256_F050.0.as_ptr() as *const __m256i),
        ),
        z43_h,
    );

    // --- Final: combine and descale ---
    let round = _mm256_set1_epi32(1 << (DESCALE - 1));

    let d01_l =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_add_epi32(tmp1011_l, t32_l), round));
    let d01_h =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_add_epi32(tmp1011_h, t32_h), round));
    let data01 = _mm256_packs_epi32(d01_l, d01_h); // data0_1

    let d76_l =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_sub_epi32(tmp1011_l, t32_l), round));
    let d76_h =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_sub_epi32(tmp1011_h, t32_h), round));
    let data76 = _mm256_packs_epi32(d76_l, d76_h); // data7_6

    let d32_l =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_add_epi32(tmp1312_l, t01_l), round));
    let d32_h =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_add_epi32(tmp1312_h, t01_h), round));
    let data32 = _mm256_packs_epi32(d32_l, d32_h); // data3_2

    let d45_l =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_sub_epi32(tmp1312_l, t01_l), round));
    let d45_h =
        _mm256_srai_epi32::<DESCALE>(_mm256_add_epi32(_mm256_sub_epi32(tmp1312_h, t01_h), round));
    let data45 = _mm256_packs_epi32(d45_l, d45_h); // data4_5

    (data01, data32, data45, data76)
}

/// 256-bit 8x8 transpose matching libjpeg-turbo DOTRANSPOSE.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dotranspose(
    r0: __m256i,
    r1: __m256i,
    r2: __m256i,
    r3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let t0 = _mm256_permute4x64_epi64(r0, 0xD8);
    let t1 = _mm256_permute4x64_epi64(r1, 0x72);
    let t2 = _mm256_permute4x64_epi64(r2, 0xD8);
    let t3 = _mm256_permute4x64_epi64(r3, 0x72);

    let u0 = _mm256_unpacklo_epi16(t0, t1);
    let u1 = _mm256_unpackhi_epi16(t0, t1);
    let u2 = _mm256_unpacklo_epi16(t2, t3);
    let u3 = _mm256_unpackhi_epi16(t2, t3);

    let v0 = _mm256_unpacklo_epi16(u0, u1);
    let v1 = _mm256_unpacklo_epi16(u2, u3);
    let v2 = _mm256_unpackhi_epi16(u0, u1);
    let v3 = _mm256_unpackhi_epi16(u2, u3);

    (
        _mm256_unpacklo_epi64(v0, v1),
        _mm256_unpackhi_epi64(v0, v1),
        _mm256_unpacklo_epi64(v2, v3),
        _mm256_unpackhi_epi64(v2, v3),
    )
}

/// SSE2 transpose helper (kept for SSE2 IDCT path).
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn transpose_8x8_i16(rows: [__m128i; 8]) -> [__m128i; 8] {
    let t0 = _mm_unpacklo_epi16(rows[0], rows[1]);
    let t1 = _mm_unpackhi_epi16(rows[0], rows[1]);
    let t2 = _mm_unpacklo_epi16(rows[2], rows[3]);
    let t3 = _mm_unpackhi_epi16(rows[2], rows[3]);
    let t4 = _mm_unpacklo_epi16(rows[4], rows[5]);
    let t5 = _mm_unpackhi_epi16(rows[4], rows[5]);
    let t6 = _mm_unpacklo_epi16(rows[6], rows[7]);
    let t7 = _mm_unpackhi_epi16(rows[6], rows[7]);
    let u0 = _mm_unpacklo_epi32(t0, t2);
    let u1 = _mm_unpackhi_epi32(t0, t2);
    let u2 = _mm_unpacklo_epi32(t1, t3);
    let u3 = _mm_unpackhi_epi32(t1, t3);
    let u4 = _mm_unpacklo_epi32(t4, t6);
    let u5 = _mm_unpackhi_epi32(t4, t6);
    let u6 = _mm_unpacklo_epi32(t5, t7);
    let u7 = _mm_unpackhi_epi32(t5, t7);
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
