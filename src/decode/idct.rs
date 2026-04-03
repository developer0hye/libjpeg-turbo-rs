/// Scalar 8x8 inverse DCT using the AAN (Arai, Agui, Nakajima) algorithm
/// with fixed-point integer arithmetic. Matches libjpeg-turbo's jidctint.c.
///
/// Input: 64 dequantized coefficients in natural (row-major) order.
/// Output: 64 spatial-domain sample values (not yet level-shifted or clamped).
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

#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    x.wrapping_add(1 << (n - 1)) >> n
}

/// Perform one pass of the 1-D IDCT on 8 values.
/// Used for both column pass (pass 1) and row pass (pass 2).
/// Uses wrapping arithmetic to match C's implicit i32 overflow behavior,
/// which is safe and correct for the IDCT algorithm (especially with 12-bit input).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn idct_1d(s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, s7: i32) -> [i32; 8] {
    // Even part
    let tmp0 = s0;
    let tmp1 = s2;
    let tmp2 = s4;
    let tmp3 = s6;

    let z1 = (tmp1.wrapping_add(tmp3)).wrapping_mul(F_0_541);
    let tmp2a = z1.wrapping_add(tmp3.wrapping_mul(-F_1_847));
    let tmp3a = z1.wrapping_add(tmp1.wrapping_mul(F_0_765));

    let tmp0a = (tmp0.wrapping_add(tmp2)) << CONST_BITS;
    let tmp1a = (tmp0.wrapping_sub(tmp2)) << CONST_BITS;

    let tmp10 = tmp0a.wrapping_add(tmp3a);
    let tmp13 = tmp0a.wrapping_sub(tmp3a);
    let tmp11 = tmp1a.wrapping_add(tmp2a);
    let tmp12 = tmp1a.wrapping_sub(tmp2a);

    // Odd part
    let z1 = s7.wrapping_add(s1);
    let z2 = s5.wrapping_add(s3);
    let z3 = s7.wrapping_add(s3);
    let z4 = s5.wrapping_add(s1);
    let z5 = (z3.wrapping_add(z4)).wrapping_mul(F_1_175);

    let tmp0 = s7.wrapping_mul(F_0_298);
    let tmp1 = s5.wrapping_mul(F_2_053);
    let tmp2 = s3.wrapping_mul(F_3_072);
    let tmp3 = s1.wrapping_mul(F_1_501);
    let z1 = z1.wrapping_mul(-F_0_899);
    let z2 = z2.wrapping_mul(-F_2_562);
    let z3 = z3.wrapping_mul(-F_1_961).wrapping_add(z5);
    let z4 = z4.wrapping_mul(-F_0_390).wrapping_add(z5);

    let tmp0 = tmp0.wrapping_add(z1).wrapping_add(z3);
    let tmp1 = tmp1.wrapping_add(z2).wrapping_add(z4);
    let tmp2 = tmp2.wrapping_add(z2).wrapping_add(z3);
    let tmp3 = tmp3.wrapping_add(z1).wrapping_add(z4);

    [
        tmp10.wrapping_add(tmp3),
        tmp11.wrapping_add(tmp2),
        tmp12.wrapping_add(tmp1),
        tmp13.wrapping_add(tmp0),
        tmp13.wrapping_sub(tmp0),
        tmp12.wrapping_sub(tmp1),
        tmp11.wrapping_sub(tmp2),
        tmp10.wrapping_sub(tmp3),
    ]
}

/// 12-bit IDCT: uses PASS1_BITS=1 to avoid overflow with 12-bit sample range.
/// Matches C jidctint.c with BITS_IN_JSAMPLE=12.
pub fn idct_8x8_12bit(coeffs: &[i16; 64]) -> [i16; 64] {
    const PASS1_BITS_12: i32 = 1;
    let mut workspace = [0i32; 64];

    // Pass 1: process columns
    for col in 0..8 {
        let s = |row: usize| coeffs[row * 8 + col] as i32;

        if s(1) == 0 && s(2) == 0 && s(3) == 0 && s(4) == 0 && s(5) == 0 && s(6) == 0 && s(7) == 0 {
            let dcval: i32 = s(0) << PASS1_BITS_12;
            for row in 0..8 {
                workspace[row * 8 + col] = dcval;
            }
            continue;
        }

        let result: [i32; 8] = idct_1d(s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7));
        for (row, &val) in result.iter().enumerate() {
            workspace[row * 8 + col] = descale(val, CONST_BITS - PASS1_BITS_12);
        }
    }

    // Pass 2: process rows
    let mut output = [0i16; 64];
    let descale_bits: i32 = CONST_BITS + PASS1_BITS_12 + 3;

    for row in 0..8 {
        let w = |col: usize| workspace[row * 8 + col];

        if w(1) == 0 && w(2) == 0 && w(3) == 0 && w(4) == 0 && w(5) == 0 && w(6) == 0 && w(7) == 0 {
            let dcval: i16 = descale(w(0), PASS1_BITS_12 + 3) as i16;
            for col in 0..8 {
                output[row * 8 + col] = dcval;
            }
            continue;
        }

        let result: [i32; 8] = idct_1d(w(0), w(1), w(2), w(3), w(4), w(5), w(6), w(7));
        for (col, &val) in result.iter().enumerate() {
            output[row * 8 + col] = descale(val, descale_bits) as i16;
        }
    }

    output
}

// =========================================================================
// IFAST IDCT — matches jidctfst.c (CONST_BITS=8, no rounding in descale)
// =========================================================================
/// AA&N scaling factors (CONST_BITS=14) from jddctmgr.c, used to build
/// IFAST-scaled quantization tables.
const AANSCALES: [i32; 64] = [
    16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520, 22725, 31521, 29692, 26722, 22725, 17855,
    12299, 6270, 21407, 29692, 27969, 25172, 21407, 16819, 11585, 5906, 19266, 26722, 25172, 22654,
    19266, 15137, 10426, 5315, 16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520, 12873, 17855,
    16819, 15137, 12873, 10114, 6967, 3552, 8867, 12299, 11585, 10426, 8867, 6967, 4799, 2446,
    4520, 6270, 5906, 5315, 4520, 3552, 2446, 1247,
];

/// IFAST constants at 8-bit precision.
const IFAST_FIX_1_082: i32 = 277;
const IFAST_FIX_1_414: i32 = 362;
const IFAST_FIX_1_848: i32 = 473;
const IFAST_FIX_2_613: i32 = 669;

/// Combined dequant + IFAST IDCT.
///
/// Matches C libjpeg-turbo's `jidctfst.c` exactly, including 16-bit DCTELEM
/// truncation at each intermediate step (C uses `short` for all temporaries).
#[allow(clippy::identity_op, clippy::erasing_op)]
pub fn idct_ifast_8x8(coeffs: &[i16; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut ifast_quant = [0i16; 64];
    for i in 0..64 {
        ifast_quant[i] = ((quant[i] as i32 * AANSCALES[i] + (1 << 11)) >> 12) as i16;
    }

    // Truncate to i16 then widen back, matching C's DCTELEM (short) truncation.
    #[inline(always)]
    fn d(v: i32) -> i32 {
        v as i16 as i32
    }

    #[inline(always)]
    fn ifast_mul(var: i32, constant: i32) -> i32 {
        d((var * constant) >> 8)
    }

    let mut workspace = [0i32; 64];

    // Pass 1: process columns — dequantize truncates to i16 like C's DCTELEM
    for col in 0..8 {
        let dequant = |row: usize| -> i32 {
            d((coeffs[row * 8 + col] as i32) * (ifast_quant[row * 8 + col] as i32))
        };

        if coeffs[1 * 8 + col] == 0
            && coeffs[2 * 8 + col] == 0
            && coeffs[3 * 8 + col] == 0
            && coeffs[4 * 8 + col] == 0
            && coeffs[5 * 8 + col] == 0
            && coeffs[6 * 8 + col] == 0
            && coeffs[7 * 8 + col] == 0
        {
            let dcval: i32 = dequant(0);
            for row in 0..8 {
                workspace[row * 8 + col] = dcval;
            }
            continue;
        }

        let tmp0 = dequant(0);
        let tmp1 = dequant(2);
        let tmp2 = dequant(4);
        let tmp3 = dequant(6);

        let tmp10 = d(tmp0 + tmp2);
        let tmp11 = d(tmp0 - tmp2);
        let tmp13 = d(tmp1 + tmp3);
        let tmp12 = d(ifast_mul(d(tmp1 - tmp3), IFAST_FIX_1_414) - tmp13);
        let e0 = d(tmp10 + tmp13);
        let e3 = d(tmp10 - tmp13);
        let e1 = d(tmp11 + tmp12);
        let e2 = d(tmp11 - tmp12);

        let tmp4 = dequant(1);
        let tmp5 = dequant(3);
        let tmp6 = dequant(5);
        let tmp7 = dequant(7);

        let z13 = d(tmp6 + tmp5);
        let z10 = d(tmp6 - tmp5);
        let z11 = d(tmp4 + tmp7);
        let z12 = d(tmp4 - tmp7);

        let o7 = d(z11 + z13);
        let o11 = ifast_mul(d(z11 - z13), IFAST_FIX_1_414);
        let z5 = ifast_mul(d(z10 + z12), IFAST_FIX_1_848);
        let o10 = d(ifast_mul(z12, IFAST_FIX_1_082) - z5);
        let o12 = d(ifast_mul(z10, -IFAST_FIX_2_613) + z5);

        let o6 = d(o12 - o7);
        let o5 = d(o11 - o6);
        let o4 = d(o10 + o5);

        // Final writes: C does (int)(tmp0 + tmp7) — no truncation
        workspace[0 * 8 + col] = e0 + o7;
        workspace[7 * 8 + col] = e0 - o7;
        workspace[1 * 8 + col] = e1 + o6;
        workspace[6 * 8 + col] = e1 - o6;
        workspace[2 * 8 + col] = e2 + o5;
        workspace[5 * 8 + col] = e2 - o5;
        workspace[4 * 8 + col] = e3 + o4;
        workspace[3 * 8 + col] = e3 - o4;
    }

    // Pass 2: process rows — C casts workspace ints to DCTELEM (short)
    let mut output = [0i16; 64];

    for row in 0..8 {
        let w = |c: usize| d(workspace[row * 8 + c]);

        if workspace[row * 8 + 1] == 0
            && workspace[row * 8 + 2] == 0
            && workspace[row * 8 + 3] == 0
            && workspace[row * 8 + 4] == 0
            && workspace[row * 8 + 5] == 0
            && workspace[row * 8 + 6] == 0
            && workspace[row * 8 + 7] == 0
        {
            let dcval: i16 = (d(workspace[row * 8]) >> 5) as i16;
            for c in 0..8 {
                output[row * 8 + c] = dcval;
            }
            continue;
        }

        let tmp0 = w(0);
        let tmp1 = w(2);
        let tmp2 = w(4);
        let tmp3 = w(6);

        let tmp10 = d(tmp0 + tmp2);
        let tmp11 = d(tmp0 - tmp2);
        let tmp13 = d(tmp1 + tmp3);
        let tmp12 = d(ifast_mul(d(tmp1 - tmp3), IFAST_FIX_1_414) - tmp13);
        let e0 = d(tmp10 + tmp13);
        let e3 = d(tmp10 - tmp13);
        let e1 = d(tmp11 + tmp12);
        let e2 = d(tmp11 - tmp12);

        let z13 = d(w(5) + w(3));
        let z10 = d(w(5) - w(3));
        let z11 = d(w(1) + w(7));
        let z12 = d(w(1) - w(7));

        let o7 = d(z11 + z13);
        let o11 = ifast_mul(d(z11 - z13), IFAST_FIX_1_414);
        let z5 = ifast_mul(d(z10 + z12), IFAST_FIX_1_848);
        let o10 = d(ifast_mul(z12, IFAST_FIX_1_082) - z5);
        let o12 = d(ifast_mul(z10, -IFAST_FIX_2_613) + z5);

        let o6 = d(o12 - o7);
        let o5 = d(o11 - o6);
        let o4 = d(o10 + o5);

        output[row * 8 + 0] = ((e0 + o7) >> 5) as i16;
        output[row * 8 + 7] = ((e0 - o7) >> 5) as i16;
        output[row * 8 + 1] = ((e1 + o6) >> 5) as i16;
        output[row * 8 + 6] = ((e1 - o6) >> 5) as i16;
        output[row * 8 + 2] = ((e2 + o5) >> 5) as i16;
        output[row * 8 + 5] = ((e2 - o5) >> 5) as i16;
        output[row * 8 + 4] = ((e3 + o4) >> 5) as i16;
        output[row * 8 + 3] = ((e3 - o4) >> 5) as i16;
    }

    output
}

// =========================================================================
// Float IDCT — matches jidctflt.c
// =========================================================================

/// AA&N scale factors (floating-point) from jddctmgr.c.
const AAN_SCALE_FACTOR: [f64; 8] = [
    1.0,
    1.387039845,
    1.306562965,
    1.175875602,
    1.0,
    0.785694958,
    0.541196100,
    0.275899379,
];

/// Combined dequant + Float IDCT.
///
/// `coeffs`: raw quantized coefficients in natural order.
/// `quant`: standard quantization table.
/// Returns spatial-domain values (before level-shift/clamp).
#[allow(
    clippy::identity_op,
    clippy::erasing_op,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop
)]
pub fn idct_float_8x8(coeffs: &[i16; 64], quant: &[u16; 64]) -> [i16; 64] {
    // Build float quant table: quant[i] * aanscale[row] * aanscale[col] * 0.125
    let mut fquant = [0.0f32; 64];
    for row in 0..8 {
        for col in 0..8 {
            let i: usize = row * 8 + col;
            fquant[i] = (quant[i] as f32)
                * (AAN_SCALE_FACTOR[row] as f32)
                * (AAN_SCALE_FACTOR[col] as f32)
                * 0.125;
        }
    }

    let mut workspace = [0.0f32; 64];

    // Pass 1: process columns
    for col in 0..8 {
        let dequant =
            |row: usize| -> f32 { (coeffs[row * 8 + col] as f32) * fquant[row * 8 + col] };

        if coeffs[1 * 8 + col] == 0
            && coeffs[2 * 8 + col] == 0
            && coeffs[3 * 8 + col] == 0
            && coeffs[4 * 8 + col] == 0
            && coeffs[5 * 8 + col] == 0
            && coeffs[6 * 8 + col] == 0
            && coeffs[7 * 8 + col] == 0
        {
            let dcval: f32 = dequant(0);
            for row in 0..8 {
                workspace[row * 8 + col] = dcval;
            }
            continue;
        }

        let tmp0 = dequant(0);
        let tmp1 = dequant(2);
        let tmp2 = dequant(4);
        let tmp3 = dequant(6);

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;
        let tmp13 = tmp1 + tmp3;
        let tmp12 = (tmp1 - tmp3) * 1.414213562f32 - tmp13;
        let e0 = tmp10 + tmp13;
        let e3 = tmp10 - tmp13;
        let e1 = tmp11 + tmp12;
        let e2 = tmp11 - tmp12;

        let tmp4 = dequant(1);
        let tmp5 = dequant(3);
        let tmp6 = dequant(5);
        let tmp7 = dequant(7);

        let z13 = tmp6 + tmp5;
        let z10 = tmp6 - tmp5;
        let z11 = tmp4 + tmp7;
        let z12 = tmp4 - tmp7;

        let o7 = z11 + z13;
        let o11 = (z11 - z13) * 1.414213562f32;
        let z5 = (z10 + z12) * 1.847759065f32;
        let o10 = z5 - z12 * 1.082392200f32;
        let o12 = z5 - z10 * 2.613125930f32;

        let o6 = o12 - o7;
        let o5 = o11 - o6;
        let o4 = o10 - o5;

        workspace[0 * 8 + col] = e0 + o7;
        workspace[7 * 8 + col] = e0 - o7;
        workspace[1 * 8 + col] = e1 + o6;
        workspace[6 * 8 + col] = e1 - o6;
        workspace[2 * 8 + col] = e2 + o5;
        workspace[5 * 8 + col] = e2 - o5;
        workspace[3 * 8 + col] = e3 + o4;
        workspace[4 * 8 + col] = e3 - o4;
    }

    // Pass 2: process rows, apply level-shift bias (128.5) for float->int truncation
    let mut output = [0i16; 64];

    for row in 0..8 {
        let w = |c: usize| workspace[row * 8 + c];

        // Add CENTERJSAMPLE + 0.5 to DC for level-shift + truncation rounding
        let z5 = w(0) + 128.5f32;
        let tmp10 = z5 + w(4);
        let tmp11 = z5 - w(4);
        let tmp13 = w(2) + w(6);
        let tmp12 = (w(2) - w(6)) * 1.414213562f32 - tmp13;
        let e0 = tmp10 + tmp13;
        let e3 = tmp10 - tmp13;
        let e1 = tmp11 + tmp12;
        let e2 = tmp11 - tmp12;

        let z13 = w(5) + w(3);
        let z10 = w(5) - w(3);
        let z11 = w(1) + w(7);
        let z12 = w(1) - w(7);

        let o7 = z11 + z13;
        let o11 = (z11 - z13) * 1.414213562f32;
        let z5 = (z10 + z12) * 1.847759065f32;
        let o10 = z5 - z12 * 1.082392200f32;
        let o12 = z5 - z10 * 2.613125930f32;

        let o6 = o12 - o7;
        let o5 = o11 - o6;
        let o4 = o10 - o5;

        // Float->int truncation and range-limit (clamped by caller).
        // Output includes level-shift (128) already applied via the +128.5 bias.
        // Subtract 128 to return pre-level-shift values (matching idct_8x8 convention).
        output[row * 8 + 0] = ((e0 + o7) as i32 - 128) as i16;
        output[row * 8 + 7] = ((e0 - o7) as i32 - 128) as i16;
        output[row * 8 + 1] = ((e1 + o6) as i32 - 128) as i16;
        output[row * 8 + 6] = ((e1 - o6) as i32 - 128) as i16;
        output[row * 8 + 2] = ((e2 + o5) as i32 - 128) as i16;
        output[row * 8 + 5] = ((e2 - o5) as i32 - 128) as i16;
        output[row * 8 + 3] = ((e3 + o4) as i32 - 128) as i16;
        output[row * 8 + 4] = ((e3 - o4) as i32 - 128) as i16;
    }

    output
}

pub fn idct_8x8(coeffs: &[i16; 64]) -> [i16; 64] {
    let mut workspace = [0i32; 64];

    // Pass 1: process columns
    for col in 0..8 {
        let s = |row: usize| coeffs[row * 8 + col] as i32;

        if s(1) == 0 && s(2) == 0 && s(3) == 0 && s(4) == 0 && s(5) == 0 && s(6) == 0 && s(7) == 0 {
            let dcval = s(0) << PASS1_BITS;
            for row in 0..8 {
                workspace[row * 8 + col] = dcval;
            }
            continue;
        }

        let result = idct_1d(s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7));
        for (row, &val) in result.iter().enumerate() {
            workspace[row * 8 + col] = descale(val, CONST_BITS - PASS1_BITS);
        }
    }

    // Pass 2: process rows
    let mut output = [0i16; 64];
    let descale_bits = CONST_BITS + PASS1_BITS + 3;

    for row in 0..8 {
        let w = |col: usize| workspace[row * 8 + col];

        if w(1) == 0 && w(2) == 0 && w(3) == 0 && w(4) == 0 && w(5) == 0 && w(6) == 0 && w(7) == 0 {
            let dcval = descale(w(0), PASS1_BITS + 3) as i16;
            for col in 0..8 {
                output[row * 8 + col] = dcval;
            }
            continue;
        }

        let result = idct_1d(w(0), w(1), w(2), w(3), w(4), w(5), w(6), w(7));
        for (col, &val) in result.iter().enumerate() {
            output[row * 8 + col] = descale(val, descale_bits) as i16;
        }
    }

    output
}
