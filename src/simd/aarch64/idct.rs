//! NEON-accelerated 8×8 IDCT (accurate integer, "islow").
//!
//! Port of libjpeg-turbo's `jidctint-neon.c`. Combines dequantization,
//! IDCT, level-shift (+128), and clamping into a single fused operation.
//!
//! The input coefficients are in zigzag order; the quantization table is
//! in natural (row-major) order. We first reorder to natural order during
//! the dequantization step, then perform the 2-pass IDCT.

use std::arch::aarch64::*;

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

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

/// Pre-combined constants matching libjpeg-turbo's layout for lane-indexed multiply.
#[repr(align(16))]
struct IdctConsts {
    data: [i16; 16],
}

const IDCT_CONSTS: IdctConsts = IdctConsts {
    data: [
        F_0_899,                                  // [0].0
        F_0_541,                                  // [0].1
        F_2_562,                                  // [0].2
        (F_0_298 as i32 - F_0_899 as i32) as i16, // [0].3 = -4927
        (F_1_501 as i32 - F_0_899 as i32) as i16, // [1].0 = 4926
        (F_2_053 as i32 - F_2_562 as i32) as i16, // [1].1 = -4176
        (F_0_541 as i32 + F_0_765 as i32) as i16, // [1].2 = 10703
        F_1_175,                                  // [1].3
        (F_1_175 as i32 - F_0_390 as i32) as i16, // [2].0 = 6437
        (F_0_541 as i32 - F_1_847 as i32) as i16, // [2].1 = -10704
        (F_3_072 as i32 - F_2_562 as i32) as i16, // [2].2 = 4177
        (F_1_175 as i32 - F_1_961 as i32) as i16, // [2].3 = -6436
        0,
        0,
        0,
        0,
    ],
};

/// Combined dequant + IDCT + level-shift + clamp, NEON-accelerated.
///
/// `coeffs`: 64 i16 coefficients in zigzag order.
/// `quant`: 64 u16 quantization values in natural (row-major) order.
/// `output`: 64 u8 samples in natural (row-major) order.
pub fn neon_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // SAFETY: NEON is mandatory on aarch64 (ARMv8).
    unsafe {
        neon_idct_islow_core(
            coeffs.as_ptr(),
            quant.as_ptr() as *const i16,
            output.as_mut_ptr(),
            8,
        );
    }
}

/// Strided variant: writes 8×8 block directly to `output` with row stride `stride`.
///
/// # Safety
/// `output` must point to at least `7 * stride + 8` writable bytes.
pub unsafe fn neon_idct_islow_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    neon_idct_islow_core(
        coeffs.as_ptr(),
        quant.as_ptr() as *const i16,
        output,
        stride,
    );
}

/// Fused dequant + IDCT: loads raw coefficients and quant table, multiplies
/// during the load phase (like libjpeg-turbo's C NEON), eliminating the
/// intermediate dequantized buffer entirely.
///
/// # Safety
/// Requires aarch64 NEON. `cptr`/`qptr` must point to 64-element i16 arrays.
/// `output` must have at least `7 * stride + 8` writable bytes.
#[target_feature(enable = "neon")]
unsafe fn neon_idct_islow_core(cptr: *const i16, qptr: *const i16, output: *mut u8, stride: usize) {
    let consts0: int16x4_t = vld1_s16(IDCT_CONSTS.data.as_ptr());
    let consts1: int16x4_t = vld1_s16(IDCT_CONSTS.data.as_ptr().add(4));
    let consts2: int16x4_t = vld1_s16(IDCT_CONSTS.data.as_ptr().add(8));

    // --- Pass 1: columns, left 4×8 half ---
    // Fused load+dequant: vmul_s16(coeff, quant) during load
    let mut left_dc_only = false;
    let mut ws_l = [0i16; 32];
    let mut ws_r = [0i16; 32];

    {
        let row0 = vmul_s16(vld1_s16(cptr), vld1_s16(qptr));
        let row1 = vmul_s16(vld1_s16(cptr.add(8)), vld1_s16(qptr.add(8)));
        let row2 = vmul_s16(vld1_s16(cptr.add(16)), vld1_s16(qptr.add(16)));
        let row3 = vmul_s16(vld1_s16(cptr.add(24)), vld1_s16(qptr.add(24)));
        let row4 = vmul_s16(vld1_s16(cptr.add(32)), vld1_s16(qptr.add(32)));
        let row5 = vmul_s16(vld1_s16(cptr.add(40)), vld1_s16(qptr.add(40)));
        let row6 = vmul_s16(vld1_s16(cptr.add(48)), vld1_s16(qptr.add(48)));
        let row7 = vmul_s16(vld1_s16(cptr.add(56)), vld1_s16(qptr.add(56)));

        // Check sparsity
        let bitmap = vorr_s16(vorr_s16(vorr_s16(row7, row6), row5), row4);
        let bitmap_4567: i64 = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

        if bitmap_4567 == 0 {
            let bitmap_ac = vorr_s16(vorr_s16(vorr_s16(bitmap, row3), row2), row1);
            let ac_bitmap: i64 = vget_lane_s64(vreinterpret_s64_s16(bitmap_ac), 0);

            if ac_bitmap == 0 {
                // Check if positions 1-3 of row0 are also zero (true DC-only).
                // row0 = [pos0(DC), pos1, pos2, pos3] — mask out DC.
                let row0_ac_mask = vcreate_s16(0xFFFF_FFFF_FFFF_0000u64);
                let row0_ac = vand_s16(row0, row0_ac_mask);
                let row0_ac_bits: i64 = vget_lane_s64(vreinterpret_s64_s16(row0_ac), 0);
                left_dc_only = row0_ac_bits == 0;

                let dcval = vshl_n_s16::<PASS1_BITS>(row0);
                let quad: int16x4x4_t = int16x4x4_t(dcval, dcval, dcval, dcval);
                vst4_s16(ws_l.as_mut_ptr(), quad);
                vst4_s16(ws_r.as_mut_ptr(), quad);
            } else {
                // Sparse: rows 4-7 are zero
                idct_pass1_sparse(
                    row0,
                    row1,
                    row2,
                    row3,
                    consts0,
                    consts1,
                    consts2,
                    ws_l.as_mut_ptr(),
                    ws_r.as_mut_ptr(),
                );
            }
        } else {
            idct_pass1_regular(
                row0,
                row1,
                row2,
                row3,
                row4,
                row5,
                row6,
                row7,
                consts0,
                consts1,
                consts2,
                ws_l.as_mut_ptr(),
                ws_r.as_mut_ptr(),
            );
        }
    }

    // --- Pass 1: columns, right 4×8 half ---
    let mut right_all_zero = false;
    {
        let row0 = vmul_s16(vld1_s16(cptr.add(4)), vld1_s16(qptr.add(4)));
        let row1 = vmul_s16(vld1_s16(cptr.add(12)), vld1_s16(qptr.add(12)));
        let row2 = vmul_s16(vld1_s16(cptr.add(20)), vld1_s16(qptr.add(20)));
        let row3 = vmul_s16(vld1_s16(cptr.add(28)), vld1_s16(qptr.add(28)));
        let row4 = vmul_s16(vld1_s16(cptr.add(36)), vld1_s16(qptr.add(36)));
        let row5 = vmul_s16(vld1_s16(cptr.add(44)), vld1_s16(qptr.add(44)));
        let row6 = vmul_s16(vld1_s16(cptr.add(52)), vld1_s16(qptr.add(52)));
        let row7 = vmul_s16(vld1_s16(cptr.add(60)), vld1_s16(qptr.add(60)));

        let bitmap = vorr_s16(vorr_s16(vorr_s16(row7, row6), row5), row4);
        let bitmap_4567: i64 = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);
        let bitmap_ac = vorr_s16(vorr_s16(vorr_s16(bitmap, row3), row2), row1);
        let right_ac_bitmap: i64 = vget_lane_s64(vreinterpret_s64_s16(bitmap_ac), 0);

        if right_ac_bitmap == 0 {
            let bitmap_all = vorr_s16(bitmap_ac, row0);
            let right_all_bitmap: i64 = vget_lane_s64(vreinterpret_s64_s16(bitmap_all), 0);

            if right_all_bitmap == 0 {
                if left_dc_only {
                    // Pure DC block: entire right half zero + left DC-only.
                    // Skip pass2 — fill output with the DC pixel value directly.
                    let dc_dequant = (*cptr as i32) * (*qptr as i32);
                    let pixel_val = (((dc_dequant + 4) >> 3) + 128).clamp(0, 255) as u8;
                    let fill: uint8x8_t = vdup_n_u8(pixel_val);
                    for row in 0..8 {
                        vst1_u8(output.add(row * stride), fill);
                    }
                    return;
                }
                // Entire right half is zero — skip, use sparse pass 2
                right_all_zero = true;
            } else {
                // DC-only in right half
                let dcval = vshl_n_s16::<PASS1_BITS>(row0);
                let quad: int16x4x4_t = int16x4x4_t(dcval, dcval, dcval, dcval);
                vst4_s16(ws_l.as_mut_ptr().add(16), quad);
                vst4_s16(ws_r.as_mut_ptr().add(16), quad);
            }
        } else if bitmap_4567 == 0 {
            idct_pass1_sparse(
                row0,
                row1,
                row2,
                row3,
                consts0,
                consts1,
                consts2,
                ws_l.as_mut_ptr().add(16),
                ws_r.as_mut_ptr().add(16),
            );
        } else {
            idct_pass1_regular(
                row0,
                row1,
                row2,
                row3,
                row4,
                row5,
                row6,
                row7,
                consts0,
                consts1,
                consts2,
                ws_l.as_mut_ptr().add(16),
                ws_r.as_mut_ptr().add(16),
            );
        }
    }

    // --- Pass 2: rows ---
    // After pass 1, transposition happened via vst4. The workspace layout is:
    // ws_l[0..16]: rows 0-3 of left columns (transposed from left 4×8)
    // ws_l[16..32]: rows 0-3 from right columns → these become rows 4-7
    // ws_r[0..16]: rows 4-7 of left columns
    // ws_r[16..32]: rows 4-7 from right columns
    //
    // Pass 2 processes: ws_l (left workspace = rows 0-3 output to buf rows 0-3)
    //                   ws_r (right workspace = rows 4-7 output to buf rows 4-7)
    // But we swap workspace pointers to achieve the transposition:
    // ws_l processes rows 0,1,2,3 → output rows 0,1,2,3
    // ws_r processes rows 4,5,6,7 → output rows 4,5,6,7

    if right_all_zero {
        idct_pass2_sparse(ws_l.as_ptr(), output, 0, stride);
        idct_pass2_sparse(ws_r.as_ptr(), output, 4, stride);
    } else {
        idct_pass2_regular(ws_l.as_ptr(), output, 0, stride, consts0, consts1, consts2);
        idct_pass2_regular(ws_r.as_ptr(), output, 4, stride, consts0, consts1, consts2);
    }
}

/// Pass 1 regular: full 1-D IDCT on 4 columns in parallel.
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn idct_pass1_regular(
    row0: int16x4_t,
    row1: int16x4_t,
    row2: int16x4_t,
    row3: int16x4_t,
    row4: int16x4_t,
    row5: int16x4_t,
    row6: int16x4_t,
    row7: int16x4_t,
    consts0: int16x4_t,
    consts1: int16x4_t,
    consts2: int16x4_t,
    workspace_1: *mut i16,
    workspace_2: *mut i16,
) {
    // Even part
    let z2_s16 = row2;
    let z3_s16 = row6;

    let mut tmp2: int32x4_t = vmull_lane_s16(z2_s16, consts0, 1); // z2 * F_0_541
    let mut tmp3: int32x4_t = vmull_lane_s16(z2_s16, consts1, 2); // z2 * (F_0_541+F_0_765)
    tmp2 = vmlal_lane_s16(tmp2, z3_s16, consts2, 1); // + z3 * (F_0_541-F_1_847)
    tmp3 = vmlal_lane_s16(tmp3, z3_s16, consts0, 1); // + z3 * F_0_541

    let z2_s16 = row0;
    let z3_s16 = row4;

    let tmp0: int32x4_t = vshll_n_s16(vadd_s16(z2_s16, z3_s16), CONST_BITS);
    let tmp1: int32x4_t = vshll_n_s16(vsub_s16(z2_s16, z3_s16), CONST_BITS);

    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);

    // Odd part
    let tmp0_s16 = row7;
    let tmp1_s16 = row5;
    let tmp2_s16 = row3;
    let tmp3_s16 = row1;

    let z3_s16 = vadd_s16(tmp0_s16, tmp2_s16);
    let z4_s16 = vadd_s16(tmp1_s16, tmp3_s16);

    let mut z3: int32x4_t = vmull_lane_s16(z3_s16, consts2, 3); // z3 * (F_1_175-F_1_961)
    let mut z4: int32x4_t = vmull_lane_s16(z3_s16, consts1, 3); // z3 * F_1_175
    z3 = vmlal_lane_s16(z3, z4_s16, consts1, 3); // + z4 * F_1_175
    z4 = vmlal_lane_s16(z4, z4_s16, consts2, 0); // + z4 * (F_1_175-F_0_390)

    let mut tmp0 = vmull_lane_s16(tmp0_s16, consts0, 3); // tmp0 * (F_0_298-F_0_899)
    let mut tmp1 = vmull_lane_s16(tmp1_s16, consts1, 1); // tmp1 * (F_2_053-F_2_562)
    let mut tmp2 = vmull_lane_s16(tmp2_s16, consts2, 2); // tmp2 * (F_3_072-F_2_562)
    let mut tmp3 = vmull_lane_s16(tmp3_s16, consts1, 0); // tmp3 * (F_1_501-F_0_899)

    tmp0 = vmlsl_lane_s16(tmp0, tmp3_s16, consts0, 0); // - tmp3 * F_0_899
    tmp1 = vmlsl_lane_s16(tmp1, tmp2_s16, consts0, 2); // - tmp2 * F_2_562
    tmp2 = vmlsl_lane_s16(tmp2, tmp1_s16, consts0, 2); // - tmp1 * F_2_562
    tmp3 = vmlsl_lane_s16(tmp3, tmp0_s16, consts0, 0); // - tmp0 * F_0_899

    tmp0 = vaddq_s32(tmp0, z3);
    tmp1 = vaddq_s32(tmp1, z4);
    tmp2 = vaddq_s32(tmp2, z3);
    tmp3 = vaddq_s32(tmp3, z4);

    // Descale and narrow to 16-bit, then store transposed via vst4
    let descale_p1 = CONST_BITS - PASS1_BITS;

    let rows_0123 = int16x4x4_t(
        vrshrn_n_s32::<11>(vaddq_s32(tmp10, tmp3)),
        vrshrn_n_s32::<11>(vaddq_s32(tmp11, tmp2)),
        vrshrn_n_s32::<11>(vaddq_s32(tmp12, tmp1)),
        vrshrn_n_s32::<11>(vaddq_s32(tmp13, tmp0)),
    );
    let rows_4567 = int16x4x4_t(
        vrshrn_n_s32::<11>(vsubq_s32(tmp13, tmp0)),
        vrshrn_n_s32::<11>(vsubq_s32(tmp12, tmp1)),
        vrshrn_n_s32::<11>(vsubq_s32(tmp11, tmp2)),
        vrshrn_n_s32::<11>(vsubq_s32(tmp10, tmp3)),
    );

    // VST4 transposes the 4×4 block
    vst4_s16(workspace_1, rows_0123);
    vst4_s16(workspace_2, rows_4567);

    let _ = descale_p1; // Silence warning; value is compile-time 11
}

/// Pass 1 sparse: rows 4-7 are all zero.
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn idct_pass1_sparse(
    row0: int16x4_t,
    row1: int16x4_t,
    row2: int16x4_t,
    row3: int16x4_t,
    consts0: int16x4_t,
    consts1: int16x4_t,
    consts2: int16x4_t,
    workspace_1: *mut i16,
    workspace_2: *mut i16,
) {
    // Even part (z3 = row6 is zero)
    let z2_s16 = row2;

    let tmp2: int32x4_t = vmull_lane_s16(z2_s16, consts0, 1); // z2 * F_0_541
    let tmp3: int32x4_t = vmull_lane_s16(z2_s16, consts1, 2); // z2 * (F_0_541+F_0_765)

    let z2_s16 = row0;
    let tmp0: int32x4_t = vshll_n_s16(z2_s16, CONST_BITS);
    let tmp1: int32x4_t = vshll_n_s16(z2_s16, CONST_BITS);

    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);

    // Odd part (tmp0_s16 = row7 = 0, tmp1_s16 = row5 = 0)
    let tmp2_s16 = row3;
    let tmp3_s16 = row1;

    let z3_s16 = tmp2_s16;
    let z4_s16 = tmp3_s16;

    let mut z3: int32x4_t = vmull_lane_s16(z3_s16, consts2, 3);
    let mut z4: int32x4_t = vmull_lane_s16(z3_s16, consts1, 3);
    z3 = vmlal_lane_s16(z3, z4_s16, consts1, 3);
    z4 = vmlal_lane_s16(z4, z4_s16, consts2, 0);

    let tmp0 = vmlsl_lane_s16(z3, tmp3_s16, consts0, 0);
    let tmp1 = vmlsl_lane_s16(z4, tmp2_s16, consts0, 2);
    let tmp2 = vmlal_lane_s16(z3, tmp2_s16, consts2, 2);
    let tmp3 = vmlal_lane_s16(z4, tmp3_s16, consts1, 0);

    let rows_0123 = int16x4x4_t(
        vrshrn_n_s32::<11>(vaddq_s32(tmp10, tmp3)),
        vrshrn_n_s32::<11>(vaddq_s32(tmp11, tmp2)),
        vrshrn_n_s32::<11>(vaddq_s32(tmp12, tmp1)),
        vrshrn_n_s32::<11>(vaddq_s32(tmp13, tmp0)),
    );
    let rows_4567 = int16x4x4_t(
        vrshrn_n_s32::<11>(vsubq_s32(tmp13, tmp0)),
        vrshrn_n_s32::<11>(vsubq_s32(tmp12, tmp1)),
        vrshrn_n_s32::<11>(vsubq_s32(tmp11, tmp2)),
        vrshrn_n_s32::<11>(vsubq_s32(tmp10, tmp3)),
    );

    vst4_s16(workspace_1, rows_0123);
    vst4_s16(workspace_2, rows_4567);
}

/// Pass 2 regular: process 4 rows from workspace, output u8.
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
unsafe fn idct_pass2_regular(
    workspace: *const i16,
    output: *mut u8,
    buf_offset: usize,
    stride: usize,
    consts0: int16x4_t,
    consts1: int16x4_t,
    consts2: int16x4_t,
) {
    // Even part
    let z2_s16 = vld1_s16(workspace.add(2 * 4));
    let z3_s16 = vld1_s16(workspace.add(6 * 4));

    let mut tmp2: int32x4_t = vmull_lane_s16(z2_s16, consts0, 1);
    let mut tmp3: int32x4_t = vmull_lane_s16(z2_s16, consts1, 2);
    tmp2 = vmlal_lane_s16(tmp2, z3_s16, consts2, 1);
    tmp3 = vmlal_lane_s16(tmp3, z3_s16, consts0, 1);

    #[allow(clippy::erasing_op)]
    let z2_s16 = vld1_s16(workspace.add(0 * 4));
    let z3_s16 = vld1_s16(workspace.add(4 * 4));

    let tmp0: int32x4_t = vshll_n_s16(vadd_s16(z2_s16, z3_s16), CONST_BITS);
    let tmp1: int32x4_t = vshll_n_s16(vsub_s16(z2_s16, z3_s16), CONST_BITS);

    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);

    // Odd part
    let tmp0_s16 = vld1_s16(workspace.add(7 * 4));
    let tmp1_s16 = vld1_s16(workspace.add(5 * 4));
    let tmp2_s16 = vld1_s16(workspace.add(3 * 4));
    let tmp3_s16 = vld1_s16(workspace.add(4));

    let z3_s16 = vadd_s16(tmp0_s16, tmp2_s16);
    let z4_s16 = vadd_s16(tmp1_s16, tmp3_s16);

    let mut z3: int32x4_t = vmull_lane_s16(z3_s16, consts2, 3);
    let mut z4: int32x4_t = vmull_lane_s16(z3_s16, consts1, 3);
    z3 = vmlal_lane_s16(z3, z4_s16, consts1, 3);
    z4 = vmlal_lane_s16(z4, z4_s16, consts2, 0);

    let mut tmp0 = vmull_lane_s16(tmp0_s16, consts0, 3);
    let mut tmp1 = vmull_lane_s16(tmp1_s16, consts1, 1);
    let mut tmp2 = vmull_lane_s16(tmp2_s16, consts2, 2);
    let mut tmp3 = vmull_lane_s16(tmp3_s16, consts1, 0);

    tmp0 = vmlsl_lane_s16(tmp0, tmp3_s16, consts0, 0);
    tmp1 = vmlsl_lane_s16(tmp1, tmp2_s16, consts0, 2);
    tmp2 = vmlsl_lane_s16(tmp2, tmp1_s16, consts0, 2);
    tmp3 = vmlsl_lane_s16(tmp3, tmp0_s16, consts0, 0);

    tmp0 = vaddq_s32(tmp0, z3);
    tmp1 = vaddq_s32(tmp1, z4);
    tmp2 = vaddq_s32(tmp2, z3);
    tmp3 = vaddq_s32(tmp3, z4);

    // Final output: descale to 16-bit, then narrow to 8-bit with level shift.
    // libjpeg-turbo uses vaddhn (add halving narrow = shift right 16) first,
    // then vqrshrn to finish the descale. DESCALE_P2 = 13 + 2 + 3 = 18.
    // Total shift = 16 (from vaddhn) + 2 (from vqrshrn) = 18.
    let cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3), vaddhn_s32(tmp12, tmp1));
    let cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2), vaddhn_s32(tmp13, tmp0));
    let cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0), vsubhn_s32(tmp11, tmp2));
    let cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1), vsubhn_s32(tmp10, tmp3));

    // Descale remaining 2 bits and narrow to 8-bit signed
    let cols_02_s8: int8x8_t = vqrshrn_n_s16(cols_02_s16, 2);
    let cols_13_s8: int8x8_t = vqrshrn_n_s16(cols_13_s16, 2);
    let cols_46_s8: int8x8_t = vqrshrn_n_s16(cols_46_s16, 2);
    let cols_57_s8: int8x8_t = vqrshrn_n_s16(cols_57_s16, 2);

    // Level shift: add 128 (CENTERJSAMPLE)
    let center = vdup_n_u8(128);
    let cols_02_u8: uint8x8_t = vadd_u8(vreinterpret_u8_s8(cols_02_s8), center);
    let cols_13_u8: uint8x8_t = vadd_u8(vreinterpret_u8_s8(cols_13_s8), center);
    let cols_46_u8: uint8x8_t = vadd_u8(vreinterpret_u8_s8(cols_46_s8), center);
    let cols_57_u8: uint8x8_t = vadd_u8(vreinterpret_u8_s8(cols_57_s8), center);

    // Transpose 4×8 block and store to output rows.
    // Zip adjacent columns, then use vst4_lane_u16 to complete transpose.
    let cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
    let cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
    let cols_all = uint16x4x4_t(
        vreinterpret_u16_u8(cols_01_23.0),
        vreinterpret_u16_u8(cols_01_23.1),
        vreinterpret_u16_u8(cols_45_67.0),
        vreinterpret_u16_u8(cols_45_67.1),
    );

    // Store 4 rows, each row gets 8 bytes (4 from left half + 4 from right half)
    // buf_offset: 0 for rows 0-3, 4 for rows 4-7
    let out_row0 = output.add(buf_offset * stride);
    let out_row1 = output.add((buf_offset + 1) * stride);
    let out_row2 = output.add((buf_offset + 2) * stride);
    let out_row3 = output.add((buf_offset + 3) * stride);

    vst4_lane_u16::<0>(out_row0 as *mut u16, cols_all);
    vst4_lane_u16::<1>(out_row1 as *mut u16, cols_all);
    vst4_lane_u16::<2>(out_row2 as *mut u16, cols_all);
    vst4_lane_u16::<3>(out_row3 as *mut u16, cols_all);
}

/// Pass 2 sparse: rows 4-7 in workspace are all zero.
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
unsafe fn idct_pass2_sparse(
    workspace: *const i16,
    output: *mut u8,
    buf_offset: usize,
    stride: usize,
) {
    let consts0: int16x4_t = vld1_s16(IDCT_CONSTS.data.as_ptr());
    let consts1: int16x4_t = vld1_s16(IDCT_CONSTS.data.as_ptr().add(4));
    let consts2: int16x4_t = vld1_s16(IDCT_CONSTS.data.as_ptr().add(8));

    // Even part (z3 = row6 is zero)
    let z2_s16 = vld1_s16(workspace.add(2 * 4));

    let tmp2: int32x4_t = vmull_lane_s16(z2_s16, consts0, 1);
    let tmp3: int32x4_t = vmull_lane_s16(z2_s16, consts1, 2);

    #[allow(clippy::erasing_op)]
    let z2_s16 = vld1_s16(workspace.add(0 * 4));
    let tmp0: int32x4_t = vshll_n_s16(z2_s16, CONST_BITS);
    let tmp1: int32x4_t = vshll_n_s16(z2_s16, CONST_BITS);

    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);

    // Odd part (tmp0_s16 = row7 = 0, tmp1_s16 = row5 = 0)
    let tmp2_s16 = vld1_s16(workspace.add(3 * 4));
    let tmp3_s16 = vld1_s16(workspace.add(4));

    let z3_s16 = tmp2_s16;
    let z4_s16 = tmp3_s16;

    let mut z3: int32x4_t = vmull_lane_s16(z3_s16, consts2, 3);
    z3 = vmlal_lane_s16(z3, z4_s16, consts1, 3);
    let mut z4: int32x4_t = vmull_lane_s16(z3_s16, consts1, 3);
    z4 = vmlal_lane_s16(z4, z4_s16, consts2, 0);

    let tmp0 = vmlsl_lane_s16(z3, tmp3_s16, consts0, 0);
    let tmp1 = vmlsl_lane_s16(z4, tmp2_s16, consts0, 2);
    let tmp2 = vmlal_lane_s16(z3, tmp2_s16, consts2, 2);
    let tmp3 = vmlal_lane_s16(z4, tmp3_s16, consts1, 0);

    // Final output
    let cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3), vaddhn_s32(tmp12, tmp1));
    let cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2), vaddhn_s32(tmp13, tmp0));
    let cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0), vsubhn_s32(tmp11, tmp2));
    let cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1), vsubhn_s32(tmp10, tmp3));

    let cols_02_s8: int8x8_t = vqrshrn_n_s16(cols_02_s16, 2);
    let cols_13_s8: int8x8_t = vqrshrn_n_s16(cols_13_s16, 2);
    let cols_46_s8: int8x8_t = vqrshrn_n_s16(cols_46_s16, 2);
    let cols_57_s8: int8x8_t = vqrshrn_n_s16(cols_57_s16, 2);

    let center = vdup_n_u8(128);
    let cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8), center);
    let cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8), center);
    let cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8), center);
    let cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8), center);

    let cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
    let cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
    let cols_all = uint16x4x4_t(
        vreinterpret_u16_u8(cols_01_23.0),
        vreinterpret_u16_u8(cols_01_23.1),
        vreinterpret_u16_u8(cols_45_67.0),
        vreinterpret_u16_u8(cols_45_67.1),
    );

    let out_row0 = output.add(buf_offset * stride);
    let out_row1 = output.add((buf_offset + 1) * stride);
    let out_row2 = output.add((buf_offset + 2) * stride);
    let out_row3 = output.add((buf_offset + 3) * stride);

    vst4_lane_u16::<0>(out_row0 as *mut u16, cols_all);
    vst4_lane_u16::<1>(out_row1 as *mut u16, cols_all);
    vst4_lane_u16::<2>(out_row2 as *mut u16, cols_all);
    vst4_lane_u16::<3>(out_row3 as *mut u16, cols_all);
}
