//! NEON-accelerated reduced-size IDCT variants for scaled JPEG decoding.
//!
//! Port of libjpeg-turbo's `jidctred-neon.c`. Produces smaller output blocks
//! from 8x8 DCT coefficients:
//! - `neon_idct_4x4`: 4x4 output (1/2 scale)
//! - `neon_idct_2x2`: 2x2 output (1/4 scale)
//! - `neon_idct_1x1`: 1x1 output (1/8 scale)
//!
//! Each function combines dequantization, IDCT, level-shift (+128), and clamping.

use std::arch::aarch64::*;

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const CENTERJSAMPLE: i16 = 128;

// 4x4 IDCT constants (from jidctred-neon.c)
const F_0_211: i16 = 1730;
const F_0_509: i16 = 4176;
const F_0_601: i16 = 4926;
const F_0_765: i16 = 6270;
const F_0_899: i16 = 7373;
const F_1_061: i16 = 8697;
const F_1_451: i16 = 11893;
const F_1_847: i16 = 15137;
const F_2_172: i16 = 17799;
const F_2_562: i16 = 20995;

// 2x2 IDCT constants (from jidctred-neon.c)
const F_0_720: i16 = 5906;
const F_0_850: i16 = 6967;
const F_1_272: i16 = 10426;
const F_3_624: i16 = 29692;

/// 4x4 IDCT constants packed for NEON lane-indexed multiply.
/// Layout: [F_1_847, -F_0_765, -F_0_211, F_1_451, -F_2_172, F_1_061, -F_0_509, -F_0_601,
///          F_0_899, F_2_562, 0, 0]
#[repr(align(16))]
struct Idct4x4Consts {
    data: [i16; 12],
}

const IDCT_4X4_CONSTS: Idct4x4Consts = Idct4x4Consts {
    data: [
        F_1_847,
        -(F_0_765 as i32) as i16,
        -(F_0_211 as i32) as i16,
        F_1_451,
        -(F_2_172 as i32) as i16,
        F_1_061,
        -(F_0_509 as i32) as i16,
        -(F_0_601 as i32) as i16,
        F_0_899,
        F_2_562,
        0,
        0,
    ],
};

/// 2x2 IDCT constants packed for NEON lane-indexed multiply.
/// Layout: [-F_0_720, F_0_850, -F_1_272, F_3_624]
#[repr(align(16))]
struct Idct2x2Consts {
    data: [i16; 4],
}

const IDCT_2X2_CONSTS: Idct2x2Consts = Idct2x2Consts {
    data: [
        -(F_0_720 as i32) as i16,
        F_0_850,
        -(F_1_272 as i32) as i16,
        F_3_624,
    ],
};

/// NEON-accelerated 4x4 IDCT for 1/2 scale decode.
///
/// Takes 8x8 coefficients (natural row-major order) and quantization table,
/// producing a 4x4 output block with dequantization, IDCT, level-shift, and clamping.
pub fn neon_idct_4x4(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 16]) {
    // SAFETY: NEON is mandatory on aarch64 (ARMv8).
    unsafe {
        neon_idct_4x4_core(coeffs.as_ptr(), quant.as_ptr() as *const i16, output);
    }
}

/// NEON-accelerated 2x2 IDCT for 1/4 scale decode.
///
/// Takes 8x8 coefficients and quantization table, producing a 2x2 output block.
pub fn neon_idct_2x2(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 4]) {
    // SAFETY: NEON is mandatory on aarch64 (ARMv8).
    unsafe {
        neon_idct_2x2_core(coeffs.as_ptr(), quant.as_ptr() as *const i16, output);
    }
}

/// NEON-accelerated 1x1 IDCT for 1/8 scale decode.
///
/// Uses only the DC coefficient to produce a single pixel value.
pub fn neon_idct_1x1(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 1]) {
    // 1x1 is trivially DC / 8 + 128, clamped. NEON adds no benefit for a single value,
    // but we implement it for API consistency and to avoid branching in callers.
    let dc_coeff: i32 = coeffs[0] as i32;
    let dc_quant: i32 = quant[0] as i32;
    let dequantized: i32 = dc_coeff * dc_quant;
    // DESCALE(dequantized, 3) = (dequantized + (1 << 2)) >> 3
    let descaled: i32 = (dequantized + (1 << 2)) >> 3;
    let shifted: i32 = descaled + 128;
    output[0] = shifted.clamp(0, 255) as u8;
}

/// Core 4x4 IDCT implementation using NEON intrinsics.
///
/// Closely follows `jsimd_idct_4x4_neon` from libjpeg-turbo.
///
/// # Safety
/// Requires aarch64 NEON. `cptr`/`qptr` must point to 64-element arrays.
#[target_feature(enable = "neon")]
unsafe fn neon_idct_4x4_core(cptr: *const i16, qptr: *const i16, output: &mut [u8; 16]) {
    // Load DCT coefficients (rows 0,1,2,3,5,6,7 -- row 4 not needed for 4x4)
    let row0: int16x8_t = vld1q_s16(cptr);
    let row1: int16x8_t = vld1q_s16(cptr.add(8));
    let row2: int16x8_t = vld1q_s16(cptr.add(16));
    let row3: int16x8_t = vld1q_s16(cptr.add(24));
    let row5: int16x8_t = vld1q_s16(cptr.add(40));
    let row6: int16x8_t = vld1q_s16(cptr.add(48));
    let row7: int16x8_t = vld1q_s16(cptr.add(56));

    // Load quantization table values
    let quant_row0: int16x8_t = vld1q_s16(qptr);

    // Dequantize DC coefficients (row 0)
    let row0: int16x8_t = vmulq_s16(row0, quant_row0);

    // Construct bitmap to test if all AC coefficients are zero
    let bitmap: int16x8_t = vorrq_s16(
        vorrq_s16(vorrq_s16(row1, row2), row3),
        vorrq_s16(vorrq_s16(row5, row6), row7),
    );

    let left_ac_bitmap: i64 = vgetq_lane_s64(vreinterpretq_s64_s16(bitmap), 0);
    let right_ac_bitmap: i64 = vgetq_lane_s64(vreinterpretq_s64_s16(bitmap), 1);

    // Load IDCT constants
    let consts0: int16x4_t = vld1_s16(IDCT_4X4_CONSTS.data.as_ptr());
    let consts1: int16x4_t = vld1_s16(IDCT_4X4_CONSTS.data.as_ptr().add(4));
    let consts2: int16x4_t = vld1_s16(IDCT_4X4_CONSTS.data.as_ptr().add(8));

    let (row0, row1, row2, row3) = if left_ac_bitmap == 0 && right_ac_bitmap == 0 {
        // All AC coefficients zero: just duplicate DC values
        let dcval: int16x8_t = vshlq_n_s16(row0, PASS1_BITS);
        (dcval, dcval, dcval, dcval)
    } else {
        // Full IDCT computation - load remaining quant rows and dequantize
        let quant_row1: int16x8_t = vld1q_s16(qptr.add(8));
        let quant_row2: int16x8_t = vld1q_s16(qptr.add(16));
        let quant_row3: int16x8_t = vld1q_s16(qptr.add(24));
        let quant_row5: int16x8_t = vld1q_s16(qptr.add(40));
        let quant_row6: int16x8_t = vld1q_s16(qptr.add(48));
        let quant_row7: int16x8_t = vld1q_s16(qptr.add(56));

        // Even part
        let tmp0_l: int32x4_t = vshll_n_s16(vget_low_s16(row0), CONST_BITS + 1);
        let tmp0_h: int32x4_t = vshll_n_s16(vget_high_s16(row0), CONST_BITS + 1);

        let z2: int16x8_t = vmulq_s16(row2, quant_row2);
        let z3: int16x8_t = vmulq_s16(row6, quant_row6);

        let mut tmp2_l: int32x4_t = vmull_lane_s16(vget_low_s16(z2), consts0, 0);
        let mut tmp2_h: int32x4_t = vmull_lane_s16(vget_high_s16(z2), consts0, 0);
        tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z3), consts0, 1);
        tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z3), consts0, 1);

        let tmp10_l: int32x4_t = vaddq_s32(tmp0_l, tmp2_l);
        let tmp10_h: int32x4_t = vaddq_s32(tmp0_h, tmp2_h);
        let tmp12_l: int32x4_t = vsubq_s32(tmp0_l, tmp2_l);
        let tmp12_h: int32x4_t = vsubq_s32(tmp0_h, tmp2_h);

        // Odd part
        let z1: int16x8_t = vmulq_s16(row7, quant_row7);
        let z2: int16x8_t = vmulq_s16(row5, quant_row5);
        let z3: int16x8_t = vmulq_s16(row3, quant_row3);
        let z4: int16x8_t = vmulq_s16(row1, quant_row1);

        let mut tmp0_l: int32x4_t = vmull_lane_s16(vget_low_s16(z1), consts0, 2);
        tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(z2), consts0, 3);
        tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(z3), consts1, 0);
        tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(z4), consts1, 1);
        let mut tmp0_h: int32x4_t = vmull_lane_s16(vget_high_s16(z1), consts0, 2);
        tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(z2), consts0, 3);
        tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(z3), consts1, 0);
        tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(z4), consts1, 1);

        let mut tmp2_l: int32x4_t = vmull_lane_s16(vget_low_s16(z1), consts1, 2);
        tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z2), consts1, 3);
        tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z3), consts2, 0);
        tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z4), consts2, 1);
        let mut tmp2_h: int32x4_t = vmull_lane_s16(vget_high_s16(z1), consts1, 2);
        tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z2), consts1, 3);
        tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z3), consts2, 0);
        tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z4), consts2, 1);

        // Final output stage: descale and narrow to 16-bit
        // Shift amount: CONST_BITS - PASS1_BITS + 1 = 13 - 2 + 1 = 12
        let r0: int16x8_t = vcombine_s16(
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vaddq_s32(tmp10_l, tmp2_l)),
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vaddq_s32(tmp10_h, tmp2_h)),
        );
        let r3: int16x8_t = vcombine_s16(
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vsubq_s32(tmp10_l, tmp2_l)),
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vsubq_s32(tmp10_h, tmp2_h)),
        );
        let r1: int16x8_t = vcombine_s16(
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vaddq_s32(tmp12_l, tmp0_l)),
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vaddq_s32(tmp12_h, tmp0_h)),
        );
        let r2: int16x8_t = vcombine_s16(
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vsubq_s32(tmp12_l, tmp0_l)),
            vrshrn_n_s32::<{ 13 - 2 + 1 }>(vsubq_s32(tmp12_h, tmp0_h)),
        );
        (r0, r1, r2, r3)
    };

    // Transpose 8x4 block for second pass
    let row_01: int16x8x2_t = vtrnq_s16(row0, row1);
    let row_23: int16x8x2_t = vtrnq_s16(row2, row3);

    let cols_0426: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(row_01.0),
        vreinterpretq_s32_s16(row_23.0),
    );
    let cols_1537: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(row_01.1),
        vreinterpretq_s32_s16(row_23.1),
    );

    let col0: int16x4_t = vreinterpret_s16_s32(vget_low_s32(cols_0426.0));
    let col1: int16x4_t = vreinterpret_s16_s32(vget_low_s32(cols_1537.0));
    let col2: int16x4_t = vreinterpret_s16_s32(vget_low_s32(cols_0426.1));
    let col3: int16x4_t = vreinterpret_s16_s32(vget_low_s32(cols_1537.1));
    let col5: int16x4_t = vreinterpret_s16_s32(vget_high_s32(cols_1537.0));
    let col6: int16x4_t = vreinterpret_s16_s32(vget_high_s32(cols_0426.1));
    let col7: int16x4_t = vreinterpret_s16_s32(vget_high_s32(cols_1537.1));

    // Second pass of IDCT

    // Even part
    let tmp0: int32x4_t = vshll_n_s16(col0, CONST_BITS + 1);
    let mut tmp2: int32x4_t = vmull_lane_s16(col2, consts0, 0);
    tmp2 = vmlal_lane_s16(tmp2, col6, consts0, 1);

    let tmp10: int32x4_t = vaddq_s32(tmp0, tmp2);
    let tmp12: int32x4_t = vsubq_s32(tmp0, tmp2);

    // Odd part
    let mut tmp0: int32x4_t = vmull_lane_s16(col7, consts0, 2);
    tmp0 = vmlal_lane_s16(tmp0, col5, consts0, 3);
    tmp0 = vmlal_lane_s16(tmp0, col3, consts1, 0);
    tmp0 = vmlal_lane_s16(tmp0, col1, consts1, 1);

    let mut tmp2: int32x4_t = vmull_lane_s16(col7, consts1, 2);
    tmp2 = vmlal_lane_s16(tmp2, col5, consts1, 3);
    tmp2 = vmlal_lane_s16(tmp2, col3, consts2, 0);
    tmp2 = vmlal_lane_s16(tmp2, col1, consts2, 1);

    // Final output stage: descale and clamp to [0, 255]
    // The C code uses vaddhn/vsubhn which does a narrowing shift of 16 bits,
    // then vrsraq for the remaining shift. Combined shift is:
    // CONST_BITS + PASS1_BITS + 3 + 1 = 13 + 2 + 3 + 1 = 19
    let output_cols_02: int16x8_t = vcombine_s16(vaddhn_s32(tmp10, tmp2), vsubhn_s32(tmp12, tmp0));
    let output_cols_13: int16x8_t = vcombine_s16(vaddhn_s32(tmp12, tmp0), vsubhn_s32(tmp10, tmp2));

    // Level shift and remaining descale
    let center: int16x8_t = vdupq_n_s16(CENTERJSAMPLE);
    let output_cols_02: int16x8_t = vrsraq_n_s16::<{ 13 + 2 + 3 + 1 - 16 }>(center, output_cols_02);
    let output_cols_13: int16x8_t = vrsraq_n_s16::<{ 13 + 2 + 3 + 1 - 16 }>(center, output_cols_13);

    // Narrow to 8-bit unsigned with saturation
    let u8_cols_02: uint8x8_t = vqmovun_s16(output_cols_02);
    let u8_cols_13: uint8x8_t = vqmovun_s16(output_cols_13);

    // Interleave columns and store
    let interleaved: uint8x8x2_t = vzip_u8(u8_cols_02, u8_cols_13);
    let output_01_23: uint16x4x2_t = uint16x4x2_t(
        vreinterpret_u16_u8(interleaved.0),
        vreinterpret_u16_u8(interleaved.1),
    );

    // Store 4x4 block row by row
    let out_ptr: *mut u8 = output.as_mut_ptr();
    vst2_lane_u16::<0>(out_ptr as *mut u16, output_01_23);
    vst2_lane_u16::<1>(out_ptr.add(4) as *mut u16, output_01_23);
    vst2_lane_u16::<2>(out_ptr.add(8) as *mut u16, output_01_23);
    vst2_lane_u16::<3>(out_ptr.add(12) as *mut u16, output_01_23);
}

/// Core 2x2 IDCT implementation using NEON intrinsics.
///
/// Closely follows `jsimd_idct_2x2_neon` from libjpeg-turbo.
///
/// # Safety
/// Requires aarch64 NEON. `cptr`/`qptr` must point to 64-element arrays.
#[target_feature(enable = "neon")]
unsafe fn neon_idct_2x2_core(cptr: *const i16, qptr: *const i16, output: &mut [u8; 4]) {
    // Load DCT coefficients (rows 0, 1, 3, 5, 7)
    let row0: int16x8_t = vld1q_s16(cptr);
    let row1: int16x8_t = vld1q_s16(cptr.add(8));
    let row3: int16x8_t = vld1q_s16(cptr.add(24));
    let row5: int16x8_t = vld1q_s16(cptr.add(40));
    let row7: int16x8_t = vld1q_s16(cptr.add(56));

    // Load quantization table values
    let quant_row0: int16x8_t = vld1q_s16(qptr);
    let quant_row1: int16x8_t = vld1q_s16(qptr.add(8));
    let quant_row3: int16x8_t = vld1q_s16(qptr.add(24));
    let quant_row5: int16x8_t = vld1q_s16(qptr.add(40));
    let quant_row7: int16x8_t = vld1q_s16(qptr.add(56));

    // Dequantize DCT coefficients
    let row0: int16x8_t = vmulq_s16(row0, quant_row0);
    let row1: int16x8_t = vmulq_s16(row1, quant_row1);
    let row3: int16x8_t = vmulq_s16(row3, quant_row3);
    let row5: int16x8_t = vmulq_s16(row5, quant_row5);
    let row7: int16x8_t = vmulq_s16(row7, quant_row7);

    // Load IDCT conversion constants
    let consts: int16x4_t = vld1_s16(IDCT_2X2_CONSTS.data.as_ptr());

    // Pass 1: process columns

    // Even part
    let tmp10_l: int32x4_t = vshll_n_s16(vget_low_s16(row0), CONST_BITS + 2);
    let tmp10_h: int32x4_t = vshll_n_s16(vget_high_s16(row0), CONST_BITS + 2);

    // Odd part
    let mut tmp0_l: int32x4_t = vmull_lane_s16(vget_low_s16(row1), consts, 3);
    tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(row3), consts, 2);
    tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(row5), consts, 1);
    tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(row7), consts, 0);
    let mut tmp0_h: int32x4_t = vmull_lane_s16(vget_high_s16(row1), consts, 3);
    tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(row3), consts, 2);
    tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(row5), consts, 1);
    tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(row7), consts, 0);

    // Final output stage: descale and narrow to 16-bit
    // Descale by CONST_BITS (13)
    let row0: int16x8_t = vcombine_s16(
        vrshrn_n_s32::<13>(vaddq_s32(tmp10_l, tmp0_l)),
        vrshrn_n_s32::<13>(vaddq_s32(tmp10_h, tmp0_h)),
    );
    let row1: int16x8_t = vcombine_s16(
        vrshrn_n_s32::<13>(vsubq_s32(tmp10_l, tmp0_l)),
        vrshrn_n_s32::<13>(vsubq_s32(tmp10_h, tmp0_h)),
    );

    // Transpose two rows for second pass
    let cols_0246_1357: int16x8x2_t = vtrnq_s16(row0, row1);
    let cols_0246: int16x8_t = cols_0246_1357.0;
    let cols_1357: int16x8_t = cols_0246_1357.1;

    // Duplicate columns so each is accessible in its own vector
    let cols_1155_3377: int32x4x2_t = vtrnq_s32(
        vreinterpretq_s32_s16(cols_1357),
        vreinterpretq_s32_s16(cols_1357),
    );
    let cols_1155: int16x8_t = vreinterpretq_s16_s32(cols_1155_3377.0);
    let cols_3377: int16x8_t = vreinterpretq_s16_s32(cols_1155_3377.1);

    // Pass 2: process two rows

    // Even part: only col0 matters
    let tmp10: int32x4_t = vshll_n_s16(vget_low_s16(cols_0246), CONST_BITS + 2);

    // Odd part
    let mut tmp0: int32x4_t = vmull_lane_s16(vget_low_s16(cols_1155), consts, 3);
    tmp0 = vmlal_lane_s16(tmp0, vget_low_s16(cols_3377), consts, 2);
    tmp0 = vmlal_lane_s16(tmp0, vget_high_s16(cols_1155), consts, 1);
    tmp0 = vmlal_lane_s16(tmp0, vget_high_s16(cols_3377), consts, 0);

    // Final output stage: descale and clamp to [0, 255]
    // C code uses vaddhn (shift 16) then vrsraq for remaining bits
    let output_s16: int16x8_t = vcombine_s16(vaddhn_s32(tmp10, tmp0), vsubhn_s32(tmp10, tmp0));
    let center: int16x8_t = vdupq_n_s16(CENTERJSAMPLE);
    // Remaining shift: CONST_BITS + PASS1_BITS + 3 + 2 - 16 = 13 + 2 + 3 + 2 - 16 = 4
    let output_s16: int16x8_t = vrsraq_n_s16::<{ 13 + 2 + 3 + 2 - 16 }>(center, output_s16);
    let output_u8: uint8x8_t = vqmovun_s16(output_s16);

    // Store 2x2 block
    // The output layout from the C code stores as:
    // output[0][0] = lane 0, output[1][0] = lane 1
    // output[0][1] = lane 4, output[1][1] = lane 5
    output[0] = vget_lane_u8(output_u8, 0);
    output[2] = vget_lane_u8(output_u8, 1);
    output[1] = vget_lane_u8(output_u8, 4);
    output[3] = vget_lane_u8(output_u8, 5);
}
