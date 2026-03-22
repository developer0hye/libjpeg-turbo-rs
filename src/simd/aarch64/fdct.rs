//! NEON-accelerated forward DCT (accurate integer, "islow").
//!
//! Port of libjpeg-turbo's `jfdctint-neon.c`. Performs an 8x8 forward DCT
//! using the LL&M algorithm with NEON SIMD intrinsics. Processes all 8 rows
//! simultaneously in pass 1, then transposes and processes all 8 columns
//! simultaneously in pass 2.
//!
//! Input: 8x8 block of i16 (level-shifted pixel values, i.e. pixel - 128)
//! Output: 8x8 block of i16 (DCT coefficients in natural row-major order)
//!
//! The output is scaled up by a factor of 8 compared to the true DCT,
//! matching `fdct_islow` output. This scaling factor is removed during
//! quantization.

use std::arch::aarch64::*;

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const DESCALE_P1: i32 = CONST_BITS - PASS1_BITS;
const DESCALE_P2: i32 = CONST_BITS + PASS1_BITS;

// Constants packed for lane-indexed multiply, matching libjpeg-turbo layout.
// consts[0]: F_0_298, -F_0_390,  F_0_541,  F_0_765
// consts[1]: -F_0_899,  F_1_175,  F_1_501, -F_1_847
// consts[2]: -F_1_961,  F_2_053, -F_2_562,  F_3_072
#[repr(align(16))]
struct FdctConsts {
    data: [i16; 12],
}

const FDCT_CONSTS: FdctConsts = FdctConsts {
    data: [
        2446, -3196, 4433, 6270, // F_0_298, -F_0_390, F_0_541, F_0_765
        -7373, 9633, 12299, -15137, // -F_0_899, F_1_175, F_1_501, -F_1_847
        -16069, 16819, -20995, 25172, // -F_1_961, F_2_053, -F_2_562, F_3_072
    ],
};

/// Perform NEON-accelerated forward DCT on one 8x8 block.
///
/// `input` contains level-shifted pixel values (pixel - 128) in row-major order.
/// `output` receives DCT coefficients in row-major order (i16).
///
/// The output matches `fdct_islow` (scaled up by factor of 8).
pub fn neon_fdct(input: &[i16; 64], output: &mut [i16; 64]) {
    // SAFETY: NEON is mandatory on aarch64 (ARMv8).
    unsafe {
        neon_fdct_core(input.as_ptr(), output.as_mut_ptr());
    }
}

/// Core NEON FDCT implementation.
///
/// Closely follows libjpeg-turbo's jfdctint-neon.c, processing all 8 rows/columns
/// in parallel using int16x8_t vectors.
#[target_feature(enable = "neon")]
unsafe fn neon_fdct_core(input: *const i16, output: *mut i16) {
    // Load constants
    let consts0: int16x4_t = vld1_s16(FDCT_CONSTS.data.as_ptr());
    let consts1: int16x4_t = vld1_s16(FDCT_CONSTS.data.as_ptr().add(4));
    let consts2: int16x4_t = vld1_s16(FDCT_CONSTS.data.as_ptr().add(8));

    // Load 8x8 block and transpose so each vector holds one column.
    // libjpeg-turbo uses vld4q_s16 + vuzpq_s16 for the transpose.
    let s_rows_0123: int16x8x4_t = vld4q_s16(input);
    let s_rows_4567: int16x8x4_t = vld4q_s16(input.add(32));

    let cols_04: int16x8x2_t = vuzpq_s16(s_rows_0123.0, s_rows_4567.0);
    let cols_15: int16x8x2_t = vuzpq_s16(s_rows_0123.1, s_rows_4567.1);
    let cols_26: int16x8x2_t = vuzpq_s16(s_rows_0123.2, s_rows_4567.2);
    let cols_37: int16x8x2_t = vuzpq_s16(s_rows_0123.3, s_rows_4567.3);

    let mut col0: int16x8_t = cols_04.0;
    let mut col1: int16x8_t = cols_15.0;
    let mut col2: int16x8_t = cols_26.0;
    let mut col3: int16x8_t = cols_37.0;
    let mut col4: int16x8_t = cols_04.1;
    let mut col5: int16x8_t = cols_15.1;
    let mut col6: int16x8_t = cols_26.1;
    let mut col7: int16x8_t = cols_37.1;

    // ------ Pass 1: process rows ------

    let tmp0: int16x8_t = vaddq_s16(col0, col7);
    let tmp7: int16x8_t = vsubq_s16(col0, col7);
    let tmp1: int16x8_t = vaddq_s16(col1, col6);
    let tmp6: int16x8_t = vsubq_s16(col1, col6);
    let tmp2: int16x8_t = vaddq_s16(col2, col5);
    let tmp5: int16x8_t = vsubq_s16(col2, col5);
    let tmp3: int16x8_t = vaddq_s16(col3, col4);
    let tmp4: int16x8_t = vsubq_s16(col3, col4);

    // Even part
    let tmp10: int16x8_t = vaddq_s16(tmp0, tmp3);
    let tmp13: int16x8_t = vsubq_s16(tmp0, tmp3);
    let tmp11: int16x8_t = vaddq_s16(tmp1, tmp2);
    let tmp12: int16x8_t = vsubq_s16(tmp1, tmp2);

    col0 = vshlq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS as _);
    col4 = vshlq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS as _);

    let tmp12_add_tmp13: int16x8_t = vaddq_s16(tmp12, tmp13);
    let z1_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp12_add_tmp13), consts0, 2);
    let z1_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp12_add_tmp13), consts0, 2);

    let col2_l: int32x4_t = vmlal_lane_s16(z1_l, vget_low_s16(tmp13), consts0, 3);
    let col2_h: int32x4_t = vmlal_lane_s16(z1_h, vget_high_s16(tmp13), consts0, 3);
    col2 = vcombine_s16(
        vrshrn_n_s32(col2_l, DESCALE_P1 as _),
        vrshrn_n_s32(col2_h, DESCALE_P1 as _),
    );

    let col6_l: int32x4_t = vmlal_lane_s16(z1_l, vget_low_s16(tmp12), consts1, 3);
    let col6_h: int32x4_t = vmlal_lane_s16(z1_h, vget_high_s16(tmp12), consts1, 3);
    col6 = vcombine_s16(
        vrshrn_n_s32(col6_l, DESCALE_P1 as _),
        vrshrn_n_s32(col6_h, DESCALE_P1 as _),
    );

    // Odd part
    let z1: int16x8_t = vaddq_s16(tmp4, tmp7);
    let z2: int16x8_t = vaddq_s16(tmp5, tmp6);
    let z3: int16x8_t = vaddq_s16(tmp4, tmp6);
    let z4: int16x8_t = vaddq_s16(tmp5, tmp7);

    // sqrt(2) * c3: z5 = F_1_175 * (z3 + z4)
    let mut z5_l: int32x4_t = vmull_lane_s16(vget_low_s16(z3), consts1, 1);
    let mut z5_h: int32x4_t = vmull_lane_s16(vget_high_s16(z3), consts1, 1);
    z5_l = vmlal_lane_s16(z5_l, vget_low_s16(z4), consts1, 1);
    z5_h = vmlal_lane_s16(z5_h, vget_high_s16(z4), consts1, 1);

    // sqrt(2) * (-c1 + c3 + c5 - c7) = F_0_298
    let mut tmp4_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp4), consts0, 0);
    let mut tmp4_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp4), consts0, 0);
    // sqrt(2) * ( c1 + c3 - c5 + c7) = F_2_053
    let mut tmp5_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp5), consts2, 1);
    let mut tmp5_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp5), consts2, 1);
    // sqrt(2) * ( c1 + c3 + c5 - c7) = F_3_072
    let mut tmp6_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp6), consts2, 3);
    let mut tmp6_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp6), consts2, 3);
    // sqrt(2) * ( c1 + c3 - c5 - c7) = F_1_501
    let mut tmp7_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp7), consts1, 2);
    let mut tmp7_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp7), consts1, 2);

    // sqrt(2) * (c7 - c3) = -F_0_899
    let z1_l: int32x4_t = vmull_lane_s16(vget_low_s16(z1), consts1, 0);
    let z1_h: int32x4_t = vmull_lane_s16(vget_high_s16(z1), consts1, 0);
    // sqrt(2) * (-c1 - c3) = -F_2_562
    let z2_l: int32x4_t = vmull_lane_s16(vget_low_s16(z2), consts2, 2);
    let z2_h: int32x4_t = vmull_lane_s16(vget_high_s16(z2), consts2, 2);
    // sqrt(2) * (-c3 - c5) = -F_1_961
    let mut z3_l: int32x4_t = vmull_lane_s16(vget_low_s16(z3), consts2, 0);
    let mut z3_h: int32x4_t = vmull_lane_s16(vget_high_s16(z3), consts2, 0);
    // sqrt(2) * (c5 - c3) = -F_0_390
    let mut z4_l: int32x4_t = vmull_lane_s16(vget_low_s16(z4), consts0, 1);
    let mut z4_h: int32x4_t = vmull_lane_s16(vget_high_s16(z4), consts0, 1);

    z3_l = vaddq_s32(z3_l, z5_l);
    z3_h = vaddq_s32(z3_h, z5_h);
    z4_l = vaddq_s32(z4_l, z5_l);
    z4_h = vaddq_s32(z4_h, z5_h);

    tmp4_l = vaddq_s32(tmp4_l, z1_l);
    tmp4_h = vaddq_s32(tmp4_h, z1_h);
    tmp4_l = vaddq_s32(tmp4_l, z3_l);
    tmp4_h = vaddq_s32(tmp4_h, z3_h);
    col7 = vcombine_s16(
        vrshrn_n_s32(tmp4_l, DESCALE_P1 as _),
        vrshrn_n_s32(tmp4_h, DESCALE_P1 as _),
    );

    tmp5_l = vaddq_s32(tmp5_l, z2_l);
    tmp5_h = vaddq_s32(tmp5_h, z2_h);
    tmp5_l = vaddq_s32(tmp5_l, z4_l);
    tmp5_h = vaddq_s32(tmp5_h, z4_h);
    col5 = vcombine_s16(
        vrshrn_n_s32(tmp5_l, DESCALE_P1 as _),
        vrshrn_n_s32(tmp5_h, DESCALE_P1 as _),
    );

    tmp6_l = vaddq_s32(tmp6_l, z2_l);
    tmp6_h = vaddq_s32(tmp6_h, z2_h);
    tmp6_l = vaddq_s32(tmp6_l, z3_l);
    tmp6_h = vaddq_s32(tmp6_h, z3_h);
    col3 = vcombine_s16(
        vrshrn_n_s32(tmp6_l, DESCALE_P1 as _),
        vrshrn_n_s32(tmp6_h, DESCALE_P1 as _),
    );

    tmp7_l = vaddq_s32(tmp7_l, z1_l);
    tmp7_h = vaddq_s32(tmp7_h, z1_h);
    tmp7_l = vaddq_s32(tmp7_l, z4_l);
    tmp7_h = vaddq_s32(tmp7_h, z4_h);
    col1 = vcombine_s16(
        vrshrn_n_s32(tmp7_l, DESCALE_P1 as _),
        vrshrn_n_s32(tmp7_h, DESCALE_P1 as _),
    );

    // ------ Transpose for pass 2 ------
    // 8x8 matrix transpose using trn/zip operations (same as libjpeg-turbo)
    let cols_01: int16x8x2_t = vtrnq_s16(col0, col1);
    let cols_23: int16x8x2_t = vtrnq_s16(col2, col3);
    let cols_45: int16x8x2_t = vtrnq_s16(col4, col5);
    let cols_67: int16x8x2_t = vtrnq_s16(col6, col7);

    let cols_0145_l: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(cols_01.0),
        vreinterpretq_s32_s16(cols_45.0),
    );
    let cols_0145_h: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(cols_01.1),
        vreinterpretq_s32_s16(cols_45.1),
    );
    let cols_2367_l: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(cols_23.0),
        vreinterpretq_s32_s16(cols_67.0),
    );
    let cols_2367_h: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(cols_23.1),
        vreinterpretq_s32_s16(cols_67.1),
    );

    let rows_04: int32x4x2_t = vzipq_s32(cols_0145_l.0, cols_2367_l.0);
    let rows_15: int32x4x2_t = vzipq_s32(cols_0145_h.0, cols_2367_h.0);
    let rows_26: int32x4x2_t = vzipq_s32(cols_0145_l.1, cols_2367_l.1);
    let rows_37: int32x4x2_t = vzipq_s32(cols_0145_h.1, cols_2367_h.1);

    let row0: int16x8_t = vreinterpretq_s16_s32(rows_04.0);
    let row1: int16x8_t = vreinterpretq_s16_s32(rows_15.0);
    let row2: int16x8_t = vreinterpretq_s16_s32(rows_26.0);
    let row3: int16x8_t = vreinterpretq_s16_s32(rows_37.0);
    let row4: int16x8_t = vreinterpretq_s16_s32(rows_04.1);
    let row5: int16x8_t = vreinterpretq_s16_s32(rows_15.1);
    let row6: int16x8_t = vreinterpretq_s16_s32(rows_26.1);
    let row7: int16x8_t = vreinterpretq_s16_s32(rows_37.1);

    // ------ Pass 2: process columns ------

    let tmp0: int16x8_t = vaddq_s16(row0, row7);
    let tmp7: int16x8_t = vsubq_s16(row0, row7);
    let tmp1: int16x8_t = vaddq_s16(row1, row6);
    let tmp6: int16x8_t = vsubq_s16(row1, row6);
    let tmp2: int16x8_t = vaddq_s16(row2, row5);
    let tmp5: int16x8_t = vsubq_s16(row2, row5);
    let tmp3: int16x8_t = vaddq_s16(row3, row4);
    let tmp4: int16x8_t = vsubq_s16(row3, row4);

    // Even part
    let tmp10: int16x8_t = vaddq_s16(tmp0, tmp3);
    let tmp13: int16x8_t = vsubq_s16(tmp0, tmp3);
    let tmp11: int16x8_t = vaddq_s16(tmp1, tmp2);
    let tmp12: int16x8_t = vsubq_s16(tmp1, tmp2);

    let out_row0: int16x8_t = vrshrq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS as _);
    let out_row4: int16x8_t = vrshrq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS as _);

    let tmp12_add_tmp13: int16x8_t = vaddq_s16(tmp12, tmp13);
    let z1_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp12_add_tmp13), consts0, 2);
    let z1_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp12_add_tmp13), consts0, 2);

    let row2_l: int32x4_t = vmlal_lane_s16(z1_l, vget_low_s16(tmp13), consts0, 3);
    let row2_h: int32x4_t = vmlal_lane_s16(z1_h, vget_high_s16(tmp13), consts0, 3);
    let out_row2: int16x8_t = vcombine_s16(
        vrshrn_n_s32(row2_l, DESCALE_P2 as _),
        vrshrn_n_s32(row2_h, DESCALE_P2 as _),
    );

    let row6_l: int32x4_t = vmlal_lane_s16(z1_l, vget_low_s16(tmp12), consts1, 3);
    let row6_h: int32x4_t = vmlal_lane_s16(z1_h, vget_high_s16(tmp12), consts1, 3);
    let out_row6: int16x8_t = vcombine_s16(
        vrshrn_n_s32(row6_l, DESCALE_P2 as _),
        vrshrn_n_s32(row6_h, DESCALE_P2 as _),
    );

    // Odd part
    let z1: int16x8_t = vaddq_s16(tmp4, tmp7);
    let z2: int16x8_t = vaddq_s16(tmp5, tmp6);
    let z3: int16x8_t = vaddq_s16(tmp4, tmp6);
    let z4: int16x8_t = vaddq_s16(tmp5, tmp7);

    let mut z5_l: int32x4_t = vmull_lane_s16(vget_low_s16(z3), consts1, 1);
    let mut z5_h: int32x4_t = vmull_lane_s16(vget_high_s16(z3), consts1, 1);
    z5_l = vmlal_lane_s16(z5_l, vget_low_s16(z4), consts1, 1);
    z5_h = vmlal_lane_s16(z5_h, vget_high_s16(z4), consts1, 1);

    let mut tmp4_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp4), consts0, 0);
    let mut tmp4_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp4), consts0, 0);
    let mut tmp5_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp5), consts2, 1);
    let mut tmp5_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp5), consts2, 1);
    let mut tmp6_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp6), consts2, 3);
    let mut tmp6_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp6), consts2, 3);
    let mut tmp7_l: int32x4_t = vmull_lane_s16(vget_low_s16(tmp7), consts1, 2);
    let mut tmp7_h: int32x4_t = vmull_lane_s16(vget_high_s16(tmp7), consts1, 2);

    let z1_l: int32x4_t = vmull_lane_s16(vget_low_s16(z1), consts1, 0);
    let z1_h: int32x4_t = vmull_lane_s16(vget_high_s16(z1), consts1, 0);
    let z2_l: int32x4_t = vmull_lane_s16(vget_low_s16(z2), consts2, 2);
    let z2_h: int32x4_t = vmull_lane_s16(vget_high_s16(z2), consts2, 2);
    let mut z3_l: int32x4_t = vmull_lane_s16(vget_low_s16(z3), consts2, 0);
    let mut z3_h: int32x4_t = vmull_lane_s16(vget_high_s16(z3), consts2, 0);
    let mut z4_l: int32x4_t = vmull_lane_s16(vget_low_s16(z4), consts0, 1);
    let mut z4_h: int32x4_t = vmull_lane_s16(vget_high_s16(z4), consts0, 1);

    z3_l = vaddq_s32(z3_l, z5_l);
    z3_h = vaddq_s32(z3_h, z5_h);
    z4_l = vaddq_s32(z4_l, z5_l);
    z4_h = vaddq_s32(z4_h, z5_h);

    tmp4_l = vaddq_s32(tmp4_l, z1_l);
    tmp4_h = vaddq_s32(tmp4_h, z1_h);
    tmp4_l = vaddq_s32(tmp4_l, z3_l);
    tmp4_h = vaddq_s32(tmp4_h, z3_h);
    let out_row7: int16x8_t = vcombine_s16(
        vrshrn_n_s32(tmp4_l, DESCALE_P2 as _),
        vrshrn_n_s32(tmp4_h, DESCALE_P2 as _),
    );

    tmp5_l = vaddq_s32(tmp5_l, z2_l);
    tmp5_h = vaddq_s32(tmp5_h, z2_h);
    tmp5_l = vaddq_s32(tmp5_l, z4_l);
    tmp5_h = vaddq_s32(tmp5_h, z4_h);
    let out_row5: int16x8_t = vcombine_s16(
        vrshrn_n_s32(tmp5_l, DESCALE_P2 as _),
        vrshrn_n_s32(tmp5_h, DESCALE_P2 as _),
    );

    tmp6_l = vaddq_s32(tmp6_l, z2_l);
    tmp6_h = vaddq_s32(tmp6_h, z2_h);
    tmp6_l = vaddq_s32(tmp6_l, z3_l);
    tmp6_h = vaddq_s32(tmp6_h, z3_h);
    let out_row3: int16x8_t = vcombine_s16(
        vrshrn_n_s32(tmp6_l, DESCALE_P2 as _),
        vrshrn_n_s32(tmp6_h, DESCALE_P2 as _),
    );

    tmp7_l = vaddq_s32(tmp7_l, z1_l);
    tmp7_h = vaddq_s32(tmp7_h, z1_h);
    tmp7_l = vaddq_s32(tmp7_l, z4_l);
    tmp7_h = vaddq_s32(tmp7_h, z4_h);
    let out_row1: int16x8_t = vcombine_s16(
        vrshrn_n_s32(tmp7_l, DESCALE_P2 as _),
        vrshrn_n_s32(tmp7_h, DESCALE_P2 as _),
    );

    // Store results
    vst1q_s16(output, out_row0);
    vst1q_s16(output.add(8), out_row1);
    vst1q_s16(output.add(16), out_row2);
    vst1q_s16(output.add(24), out_row3);
    vst1q_s16(output.add(32), out_row4);
    vst1q_s16(output.add(40), out_row5);
    vst1q_s16(output.add(48), out_row6);
    vst1q_s16(output.add(56), out_row7);
}
