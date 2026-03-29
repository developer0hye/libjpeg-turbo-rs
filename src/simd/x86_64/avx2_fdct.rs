//! AVX2-accelerated 8x8 forward DCT (accurate integer, "islow").
//!
//! Direct port of libjpeg-turbo's `jfdctint-avx2.asm`.
//! Processes all 8 rows/columns in 4 ymm registers using `vpmaddwd`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Constants identical to IDCT (same LL&M butterfly)
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

const PASS1_BITS: i32 = 2;
const DESCALE_P1: i32 = 13 - PASS1_BITS; // 11
const DESCALE_P2: i32 = 13 + PASS1_BITS; // 15

#[repr(align(32))]
struct A16([i16; 16]);
#[repr(align(32))]
struct A32([i32; 8]);

// Even-part constants (same as IDCT)
static PW_F130_F054_MF130_F054: A16 = A16([
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 + F_0_765) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
    (F_0_541 - F_1_847) as i16,
    F_0_541,
]);

// Odd-part z3/z4 constants (same as IDCT)
static PW_MF078_F117_F078_F117: A16 = A16([
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

// Odd-part tmp4/tmp5 constants (same as IDCT)
static PW_MF060_MF089_MF050_MF256: A16 = A16([
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

// Odd-part tmp6/tmp7 constants (FDCT-specific order, different from IDCT)
static PW_F050_MF256_F060_MF089: A16 = A16([
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_3_072 - F_2_562) as i16,
    -F_2_562,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
    (F_1_501 - F_0_899) as i16,
    -F_0_899,
]);

static PW_1_NEG1: A16 = A16([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]);

static PD_DESCALE_P1: A32 = A32([1 << (DESCALE_P1 - 1); 8]);
static PD_DESCALE_P2: A32 = A32([1 << (DESCALE_P2 - 1); 8]);

/// DODCT macro dispatch with compile-time pass number.
macro_rules! dofdct {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr, 1) => {
        dofdct_pass1($r0, $r1, $r2, $r3)
    };
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr, 2) => {
        dofdct_pass2($r0, $r1, $r2, $r3)
    };
}

/// AVX2 forward DCT on 64 i16 coefficients (in-place, natural order).
///
/// # Safety contract
/// Caller must ensure AVX2 is available.
pub fn avx2_fdct_islow(data: &mut [i16; 64]) {
    unsafe { avx2_fdct_islow_inner(data) }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_fdct_islow_inner(data: &mut [i16; 64]) {
    let ptr = data.as_mut_ptr() as *mut __m256i;

    // Load 4 ymm (2 rows each)
    let ymm4 = _mm256_loadu_si256(ptr); // rows 0,1
    let ymm5 = _mm256_loadu_si256(ptr.add(1)); // rows 2,3
    let ymm6 = _mm256_loadu_si256(ptr.add(2)); // rows 4,5
    let ymm7 = _mm256_loadu_si256(ptr.add(3)); // rows 6,7

    // Rearrange: group row pairs for butterfly
    // ymm0 = row0_row4, ymm1 = row1_row5, ymm2 = row2_row6, ymm3 = row3_row7
    let ymm0 = _mm256_permute2x128_si256(ymm4, ymm6, 0x20);
    let ymm1 = _mm256_permute2x128_si256(ymm4, ymm6, 0x31);
    let ymm2 = _mm256_permute2x128_si256(ymm5, ymm7, 0x20);
    let ymm3 = _mm256_permute2x128_si256(ymm5, ymm7, 0x31);

    // --- Pass 1: rows (transpose first, then DCT) ---
    let (t0, t1, t2, t3) = dotranspose_fdct(ymm0, ymm1, ymm2, ymm3);
    let (d0, d1, d2, d3) = dofdct!(t0, t1, t2, t3, 1);
    // d0=data0_4, d1=data3_1, d2=data2_6, d3=data7_5

    // --- Pass 2: columns (rearrange, transpose, then DCT) ---
    let c4 = _mm256_permute2x128_si256(d1, d3, 0x20); // data3_7
    let c1 = _mm256_permute2x128_si256(d1, d3, 0x31); // data1_5

    let (t0, t1, t2, t3) = dotranspose_fdct(d0, c1, d2, c4);
    let (r0, r1, r2, r3) = dofdct!(t0, t1, t2, t3, 2);
    // r0=data0_4, r1=data3_1, r2=data2_6, r3=data7_5

    // Rearrange back to sequential row order and store
    let out0 = _mm256_permute2x128_si256(r0, r1, 0x30); // data0_1
    let out1 = _mm256_permute2x128_si256(r2, r1, 0x20); // data2_3
    let out2 = _mm256_permute2x128_si256(r0, r3, 0x31); // data4_5
    let out3 = _mm256_permute2x128_si256(r2, r3, 0x21); // data6_7

    _mm256_storeu_si256(ptr, out0);
    _mm256_storeu_si256(ptr.add(1), out1);
    _mm256_storeu_si256(ptr.add(2), out2);
    _mm256_storeu_si256(ptr.add(3), out3);
}

/// FDCT transpose (different from IDCT transpose).
/// Uses vpunpcklwd/vpunpckhdq/vpermq pattern from C's jfdctint-avx2.asm.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dotranspose_fdct(
    r0: __m256i,
    r1: __m256i,
    r2: __m256i,
    r3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    // Phase 1: word interleave
    let t0 = _mm256_unpacklo_epi16(r0, r1);
    let t1 = _mm256_unpackhi_epi16(r0, r1);
    let t2 = _mm256_unpacklo_epi16(r2, r3);
    let t3 = _mm256_unpackhi_epi16(r2, r3);

    // Phase 2: dword interleave
    let u0 = _mm256_unpacklo_epi32(t0, t2);
    let u1 = _mm256_unpackhi_epi32(t0, t2);
    let u2 = _mm256_unpacklo_epi32(t1, t3);
    let u3 = _mm256_unpackhi_epi32(t1, t3);

    // Phase 3: lane permutation
    let out0 = _mm256_permute4x64_epi64(u0, 0x8D); // swap halves
    let out1 = _mm256_permute4x64_epi64(u1, 0x8D);
    let out2 = _mm256_permute4x64_epi64(u2, 0xD8);
    let out3 = _mm256_permute4x64_epi64(u3, 0xD8);

    (out0, out1, out2, out3)
}

/// FDCT Pass 1: left-shift output by PASS1_BITS for data0_4.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dofdct_pass1(
    in04: __m256i,
    in15: __m256i,
    in26: __m256i,
    in37: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    // Input butterfly: add/sub pairs
    let tmp67 = _mm256_sub_epi16(in04, in37); // data1_0 - data6_7 = tmp6_7
    let tmp10 = _mm256_add_epi16(in04, in37); // data1_0 + data6_7 = tmp1_0
    let tmp32 = _mm256_add_epi16(in15, in26); // data3_2 + data4_5 = tmp3_2
    let tmp45 = _mm256_sub_epi16(in15, in26); // data3_2 - data4_5 = tmp4_5

    // -- Even part --
    let tmp01 = _mm256_permute2x128_si256(tmp10, tmp10, 0x01); // tmp0_1
    let tmp1011 = _mm256_add_epi16(tmp01, tmp32); // tmp10_11
    let tmp1312 = _mm256_sub_epi16(tmp01, tmp32); // tmp13_12

    let tmp1110 = _mm256_permute2x128_si256(tmp1011, tmp1011, 0x01);
    let signed = _mm256_sign_epi16(
        tmp1011,
        _mm256_load_si256(PW_1_NEG1.0.as_ptr() as *const __m256i),
    );
    let sum = _mm256_add_epi16(tmp1110, signed); // (tmp10+tmp11)_(tmp10-tmp11)
                                                 // Pass 1: left-shift by PASS1_BITS
    let data04 = _mm256_slli_epi16::<PASS1_BITS>(sum);

    // data2_6 via vpmaddwd
    let tmp1213 = _mm256_permute2x128_si256(tmp1312, tmp1312, 0x01);
    let pairs_lo = _mm256_unpacklo_epi16(tmp1312, tmp1213);
    let pairs_hi = _mm256_unpackhi_epi16(tmp1312, tmp1213);
    let pw = _mm256_load_si256(PW_F130_F054_MF130_F054.0.as_ptr() as *const __m256i);
    let d26_lo = _mm256_madd_epi16(pairs_lo, pw);
    let d26_hi = _mm256_madd_epi16(pairs_hi, pw);

    let round = _mm256_load_si256(PD_DESCALE_P1.0.as_ptr() as *const __m256i);
    let d26_lo = _mm256_srai_epi32::<DESCALE_P1>(_mm256_add_epi32(d26_lo, round));
    let d26_hi = _mm256_srai_epi32::<DESCALE_P1>(_mm256_add_epi32(d26_hi, round));
    let data26 = _mm256_packs_epi32(d26_lo, d26_hi);

    // -- Odd part --
    let z34 = _mm256_add_epi16(tmp45, tmp67); // z3_4
    let z43 = _mm256_permute2x128_si256(z34, z34, 0x01);
    let z_lo = _mm256_unpacklo_epi16(z34, z43);
    let z_hi = _mm256_unpackhi_epi16(z34, z43);
    let pw_z = _mm256_load_si256(PW_MF078_F117_F078_F117.0.as_ptr() as *const __m256i);
    let z34_lo = _mm256_madd_epi16(z_lo, pw_z);
    let z34_hi = _mm256_madd_epi16(z_hi, pw_z);

    // tmp4_5 (data7_5)
    let tmp76 = _mm256_permute2x128_si256(tmp67, tmp67, 0x01);
    let p45_lo = _mm256_unpacklo_epi16(tmp45, tmp76);
    let p45_hi = _mm256_unpackhi_epi16(tmp45, tmp76);
    let pw_45 = _mm256_load_si256(PW_MF060_MF089_MF050_MF256.0.as_ptr() as *const __m256i);
    let d75_lo = _mm256_add_epi32(_mm256_madd_epi16(p45_lo, pw_45), z34_lo);
    let d75_hi = _mm256_add_epi32(_mm256_madd_epi16(p45_hi, pw_45), z34_hi);
    let d75_lo = _mm256_srai_epi32::<DESCALE_P1>(_mm256_add_epi32(d75_lo, round));
    let d75_hi = _mm256_srai_epi32::<DESCALE_P1>(_mm256_add_epi32(d75_hi, round));
    let data75 = _mm256_packs_epi32(d75_lo, d75_hi);

    // tmp6_7 (data3_1)
    let tmp54 = _mm256_permute2x128_si256(tmp45, tmp45, 0x01);
    let p67_lo = _mm256_unpacklo_epi16(tmp67, tmp54);
    let p67_hi = _mm256_unpackhi_epi16(tmp67, tmp54);
    let pw_67 = _mm256_load_si256(PW_F050_MF256_F060_MF089.0.as_ptr() as *const __m256i);
    let d31_lo = _mm256_add_epi32(_mm256_madd_epi16(p67_lo, pw_67), z34_lo);
    let d31_hi = _mm256_add_epi32(_mm256_madd_epi16(p67_hi, pw_67), z34_hi);
    let d31_lo = _mm256_srai_epi32::<DESCALE_P1>(_mm256_add_epi32(d31_lo, round));
    let d31_hi = _mm256_srai_epi32::<DESCALE_P1>(_mm256_add_epi32(d31_hi, round));
    let data31 = _mm256_packs_epi32(d31_lo, d31_hi);

    (data04, data31, data26, data75)
}

/// FDCT Pass 2: right-shift+round for data0_4, descale by DESCALE_P2 for others.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dofdct_pass2(
    in04: __m256i,
    in15: __m256i,
    in26: __m256i,
    in37: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let tmp67 = _mm256_sub_epi16(in04, in37);
    let tmp10 = _mm256_add_epi16(in04, in37);
    let tmp32 = _mm256_add_epi16(in15, in26);
    let tmp45 = _mm256_sub_epi16(in15, in26);

    // -- Even part --
    let tmp01 = _mm256_permute2x128_si256(tmp10, tmp10, 0x01);
    let tmp1011 = _mm256_add_epi16(tmp01, tmp32);
    let tmp1312 = _mm256_sub_epi16(tmp01, tmp32);

    let tmp1110 = _mm256_permute2x128_si256(tmp1011, tmp1011, 0x01);
    let signed = _mm256_sign_epi16(
        tmp1011,
        _mm256_load_si256(PW_1_NEG1.0.as_ptr() as *const __m256i),
    );
    let sum = _mm256_add_epi16(tmp1110, signed);
    // Pass 2: right-shift with rounding
    let p2_round = _mm256_set1_epi16(1 << (PASS1_BITS - 1));
    let data04 = _mm256_srai_epi16::<PASS1_BITS>(_mm256_add_epi16(sum, p2_round));

    // data2_6
    let tmp1213 = _mm256_permute2x128_si256(tmp1312, tmp1312, 0x01);
    let pairs_lo = _mm256_unpacklo_epi16(tmp1312, tmp1213);
    let pairs_hi = _mm256_unpackhi_epi16(tmp1312, tmp1213);
    let pw = _mm256_load_si256(PW_F130_F054_MF130_F054.0.as_ptr() as *const __m256i);
    let d26_lo = _mm256_madd_epi16(pairs_lo, pw);
    let d26_hi = _mm256_madd_epi16(pairs_hi, pw);

    let round = _mm256_load_si256(PD_DESCALE_P2.0.as_ptr() as *const __m256i);
    let d26_lo = _mm256_srai_epi32::<DESCALE_P2>(_mm256_add_epi32(d26_lo, round));
    let d26_hi = _mm256_srai_epi32::<DESCALE_P2>(_mm256_add_epi32(d26_hi, round));
    let data26 = _mm256_packs_epi32(d26_lo, d26_hi);

    // -- Odd part --
    let z34 = _mm256_add_epi16(tmp45, tmp67);
    let z43 = _mm256_permute2x128_si256(z34, z34, 0x01);
    let z_lo = _mm256_unpacklo_epi16(z34, z43);
    let z_hi = _mm256_unpackhi_epi16(z34, z43);
    let pw_z = _mm256_load_si256(PW_MF078_F117_F078_F117.0.as_ptr() as *const __m256i);
    let z34_lo = _mm256_madd_epi16(z_lo, pw_z);
    let z34_hi = _mm256_madd_epi16(z_hi, pw_z);

    let tmp76 = _mm256_permute2x128_si256(tmp67, tmp67, 0x01);
    let p45_lo = _mm256_unpacklo_epi16(tmp45, tmp76);
    let p45_hi = _mm256_unpackhi_epi16(tmp45, tmp76);
    let pw_45 = _mm256_load_si256(PW_MF060_MF089_MF050_MF256.0.as_ptr() as *const __m256i);
    let d75_lo = _mm256_add_epi32(_mm256_madd_epi16(p45_lo, pw_45), z34_lo);
    let d75_hi = _mm256_add_epi32(_mm256_madd_epi16(p45_hi, pw_45), z34_hi);
    let d75_lo = _mm256_srai_epi32::<DESCALE_P2>(_mm256_add_epi32(d75_lo, round));
    let d75_hi = _mm256_srai_epi32::<DESCALE_P2>(_mm256_add_epi32(d75_hi, round));
    let data75 = _mm256_packs_epi32(d75_lo, d75_hi);

    let tmp54 = _mm256_permute2x128_si256(tmp45, tmp45, 0x01);
    let p67_lo = _mm256_unpacklo_epi16(tmp67, tmp54);
    let p67_hi = _mm256_unpackhi_epi16(tmp67, tmp54);
    let pw_67 = _mm256_load_si256(PW_F050_MF256_F060_MF089.0.as_ptr() as *const __m256i);
    let d31_lo = _mm256_add_epi32(_mm256_madd_epi16(p67_lo, pw_67), z34_lo);
    let d31_hi = _mm256_add_epi32(_mm256_madd_epi16(p67_hi, pw_67), z34_hi);
    let d31_lo = _mm256_srai_epi32::<DESCALE_P2>(_mm256_add_epi32(d31_lo, round));
    let d31_hi = _mm256_srai_epi32::<DESCALE_P2>(_mm256_add_epi32(d31_hi, round));
    let data31 = _mm256_packs_epi32(d31_lo, d31_hi);

    (data04, data31, data26, data75)
}
