/// Forward DCT (integer, accurate) — LL&M algorithm.
///
/// This is a direct port of libjpeg-turbo's `jfdctint.c` (the "islow" forward DCT).
/// It uses fixed-point arithmetic with CONST_BITS=13 and PASS1_BITS=2 for 8-bit samples.
// Fixed-point constants for CONST_BITS = 13.
// These match the pre-calculated values in libjpeg-turbo's jfdctint.c.
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// Right-shift with rounding (DESCALE macro from libjpeg-turbo).
#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Perform the forward DCT on one 8x8 block.
///
/// `input` contains level-shifted pixel values (pixel - 128) in row-major order.
/// `output` receives DCT coefficients in row-major order.
///
/// The output is scaled up by a factor of 8 compared to the true DCT;
/// this factor is removed during quantization (matching libjpeg-turbo behavior).
pub fn fdct_islow(input: &[i16; 64], output: &mut [i32; 64]) {
    // Copy input to workspace (i32 for intermediate precision)
    let mut workspace = [0i32; 64];
    for i in 0..64 {
        workspace[i] = input[i] as i32;
    }

    // Pass 1: process rows.
    // Results are scaled up by sqrt(8) compared to true DCT,
    // and further scaled by 2^PASS1_BITS.
    for row in 0..8 {
        let base = row * 8;

        let tmp0 = workspace[base] + workspace[base + 7];
        let tmp7 = workspace[base] - workspace[base + 7];
        let tmp1 = workspace[base + 1] + workspace[base + 6];
        let tmp6 = workspace[base + 1] - workspace[base + 6];
        let tmp2 = workspace[base + 2] + workspace[base + 5];
        let tmp5 = workspace[base + 2] - workspace[base + 5];
        let tmp3 = workspace[base + 3] + workspace[base + 4];
        let tmp4 = workspace[base + 3] - workspace[base + 4];

        // Even part per LL&M figure 1
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        workspace[base] = (tmp10 + tmp11) << PASS1_BITS;
        workspace[base + 4] = (tmp10 - tmp11) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        workspace[base + 2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        workspace[base + 6] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);

        // Odd part per figure 8
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;

        workspace[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        workspace[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        workspace[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        workspace[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns.
    // Remove PASS1_BITS scaling but leave results scaled up by factor of 8.
    for col in 0..8 {
        let tmp0 = workspace[col] + workspace[col + 56];
        let tmp7 = workspace[col] - workspace[col + 56];
        let tmp1 = workspace[col + 8] + workspace[col + 48];
        let tmp6 = workspace[col + 8] - workspace[col + 48];
        let tmp2 = workspace[col + 16] + workspace[col + 40];
        let tmp5 = workspace[col + 16] - workspace[col + 40];
        let tmp3 = workspace[col + 24] + workspace[col + 32];
        let tmp4 = workspace[col + 24] - workspace[col + 32];

        // Even part
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        output[col] = descale(tmp10 + tmp11, PASS1_BITS);
        output[col + 32] = descale(tmp10 - tmp11, PASS1_BITS);

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        output[col + 16] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        output[col + 48] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS + PASS1_BITS);

        // Odd part
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;

        output[col + 56] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        output[col + 40] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        output[col + 24] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        output[col + 8] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }
}

/// Forward DCT (fast integer, reduced accuracy) — AA&N algorithm.
///
/// Port of libjpeg-turbo's `jfdctfst.c`. Uses 8-bit fixed-point precision
/// (CONST_BITS=8) instead of 13-bit. Fewer bits means fewer shift operations
/// but reduced accuracy, especially at high quality settings.
///
/// The AA&N algorithm (Arai, Agui, Nakajima) uses only 5 multiplies and 29 adds
/// per 1-D DCT pass. In libjpeg-turbo, the remaining scale factors are folded
/// into the quantization tables, but here we apply them explicitly so the output
/// matches `fdct_islow`'s scale (factor of 8) for use with the same quantization path.
pub fn fdct_ifast(input: &[i16; 64], output: &mut [i32; 64]) {
    // AA&N scale factors to convert from AAN-scaled output to islow-equivalent output.
    // These are: aanscale[i] = cos(i*PI/16) * sqrt(2)  (for i=0, aanscale=1.0)
    // Expressed as fixed-point with 12 bits of fraction.
    const AAN_SCALE_BITS: i32 = 12;
    #[rustfmt::skip]
    const AAN_SCALES: [i32; 8] = [
        4096, // 1.0 * 4096
        5681, // 1.387039845 * 4096
        5352, // 1.306562965 * 4096
        4816, // 1.175875602 * 4096
        4096, // 1.0 * 4096
        3218, // 0.785694958 * 4096
        2217, // 0.541196100 * 4096
        1130, // 0.275899379 * 4096
    ];

    // IFAST constants for CONST_BITS = 8 (from jfdctfst.c)
    const IFAST_CONST_BITS: i32 = 8;
    const IFAST_FIX_0_382683433: i32 = 98;
    const IFAST_FIX_0_541196100: i32 = 139;
    const IFAST_FIX_0_707106781: i32 = 181;
    const IFAST_FIX_1_306562965: i32 = 334;

    // DESCALE without rounding for ifast (matches libjpeg-turbo's ifast behavior)
    #[inline(always)]
    fn ifast_descale(x: i32, n: i32) -> i32 {
        x >> n
    }

    #[inline(always)]
    fn multiply(var: i32, constant: i32) -> i32 {
        ifast_descale(var * constant, IFAST_CONST_BITS)
    }

    // Work in i32 to avoid overflow
    let mut workspace = [0i32; 64];
    for i in 0..64 {
        workspace[i] = input[i] as i32;
    }

    // Pass 1: process rows (AA&N algorithm from jfdctfst.c)
    for row in 0..8 {
        let base: usize = row * 8;

        let tmp0: i32 = workspace[base] + workspace[base + 7];
        let tmp7: i32 = workspace[base] - workspace[base + 7];
        let tmp1: i32 = workspace[base + 1] + workspace[base + 6];
        let tmp6: i32 = workspace[base + 1] - workspace[base + 6];
        let tmp2: i32 = workspace[base + 2] + workspace[base + 5];
        let tmp5: i32 = workspace[base + 2] - workspace[base + 5];
        let tmp3: i32 = workspace[base + 3] + workspace[base + 4];
        let tmp4: i32 = workspace[base + 3] - workspace[base + 4];

        // Even part
        let tmp10: i32 = tmp0 + tmp3;
        let tmp13: i32 = tmp0 - tmp3;
        let tmp11: i32 = tmp1 + tmp2;
        let tmp12: i32 = tmp1 - tmp2;

        workspace[base] = tmp10 + tmp11;
        workspace[base + 4] = tmp10 - tmp11;

        let z1: i32 = multiply(tmp12 + tmp13, IFAST_FIX_0_707106781);
        workspace[base + 2] = tmp13 + z1;
        workspace[base + 6] = tmp13 - z1;

        // Odd part
        let tmp10: i32 = tmp4 + tmp5;
        let tmp11: i32 = tmp5 + tmp6;
        let tmp12: i32 = tmp6 + tmp7;

        let z5: i32 = multiply(tmp10 - tmp12, IFAST_FIX_0_382683433);
        let z2: i32 = multiply(tmp10, IFAST_FIX_0_541196100) + z5;
        let z4: i32 = multiply(tmp12, IFAST_FIX_1_306562965) + z5;
        let z3: i32 = multiply(tmp11, IFAST_FIX_0_707106781);

        let z11: i32 = tmp7 + z3;
        let z13: i32 = tmp7 - z3;

        workspace[base + 5] = z13 + z2;
        workspace[base + 3] = z13 - z2;
        workspace[base + 1] = z11 + z4;
        workspace[base + 7] = z11 - z4;
    }

    // Pass 2: process columns
    for col in 0..8 {
        let tmp0: i32 = workspace[col] + workspace[col + 56];
        let tmp7: i32 = workspace[col] - workspace[col + 56];
        let tmp1: i32 = workspace[col + 8] + workspace[col + 48];
        let tmp6: i32 = workspace[col + 8] - workspace[col + 48];
        let tmp2: i32 = workspace[col + 16] + workspace[col + 40];
        let tmp5: i32 = workspace[col + 16] - workspace[col + 40];
        let tmp3: i32 = workspace[col + 24] + workspace[col + 32];
        let tmp4: i32 = workspace[col + 24] - workspace[col + 32];

        // Even part
        let tmp10: i32 = tmp0 + tmp3;
        let tmp13: i32 = tmp0 - tmp3;
        let tmp11: i32 = tmp1 + tmp2;
        let tmp12: i32 = tmp1 - tmp2;

        workspace[col] = tmp10 + tmp11;
        workspace[col + 32] = tmp10 - tmp11;

        let z1: i32 = multiply(tmp12 + tmp13, IFAST_FIX_0_707106781);
        workspace[col + 16] = tmp13 + z1;
        workspace[col + 48] = tmp13 - z1;

        // Odd part
        let tmp10: i32 = tmp4 + tmp5;
        let tmp11: i32 = tmp5 + tmp6;
        let tmp12: i32 = tmp6 + tmp7;

        let z5: i32 = multiply(tmp10 - tmp12, IFAST_FIX_0_382683433);
        let z2: i32 = multiply(tmp10, IFAST_FIX_0_541196100) + z5;
        let z4: i32 = multiply(tmp12, IFAST_FIX_1_306562965) + z5;
        let z3: i32 = multiply(tmp11, IFAST_FIX_0_707106781);

        let z11: i32 = tmp7 + z3;
        let z13: i32 = tmp7 - z3;

        workspace[col + 40] = z13 + z2;
        workspace[col + 24] = z13 - z2;
        workspace[col + 8] = z11 + z4;
        workspace[col + 56] = z11 - z4;
    }

    // Apply AA&N scale factors so output matches fdct_islow's scale.
    // The raw AA&N output for coefficient (row, col) is:
    //   islow_val * aanscale[row] * aanscale[col]
    // So we divide by the 2-D AA&N scale factor to get islow-equivalent output.
    for (row, &row_scale) in AAN_SCALES.iter().enumerate() {
        for (col, &col_scale) in AAN_SCALES.iter().enumerate() {
            let idx: usize = row * 8 + col;
            let scale: i64 = (row_scale as i64) * (col_scale as i64);
            let raw: i64 = workspace[idx] as i64;
            // output = raw * 4096^2 / scale (fixed-point division)
            output[idx] = ((raw * (1i64 << (AAN_SCALE_BITS * 2))) / scale) as i32;
        }
    }
}

/// Forward DCT (floating-point) — AA&N algorithm.
///
/// Port of libjpeg-turbo's `jfdctflt.c`. Uses f64 arithmetic for maximum
/// accuracy, then converts to i32 output matching `fdct_islow`'s output
/// format for use with the same quantization path.
pub fn fdct_float(input: &[i16; 64], output: &mut [i32; 64]) {
    // AA&N scale factors: aanscale[k] = cos(k*PI/16) * sqrt(2), aanscale[0] = 1.0
    #[rustfmt::skip]
    const AAN_SCALES: [f64; 8] = [
        1.0,
        1.387039845,
        1.306562965,
        1.175875602,
        1.0,
        0.785694958,
        0.541196100,
        0.275899379,
    ];

    let mut workspace = [0.0f64; 64];
    for i in 0..64 {
        workspace[i] = input[i] as f64;
    }

    // Pass 1: process rows (AA&N algorithm from jfdctflt.c)
    for row in 0..8 {
        let base: usize = row * 8;

        let tmp0: f64 = workspace[base] + workspace[base + 7];
        let tmp7: f64 = workspace[base] - workspace[base + 7];
        let tmp1: f64 = workspace[base + 1] + workspace[base + 6];
        let tmp6: f64 = workspace[base + 1] - workspace[base + 6];
        let tmp2: f64 = workspace[base + 2] + workspace[base + 5];
        let tmp5: f64 = workspace[base + 2] - workspace[base + 5];
        let tmp3: f64 = workspace[base + 3] + workspace[base + 4];
        let tmp4: f64 = workspace[base + 3] - workspace[base + 4];

        // Even part
        let tmp10: f64 = tmp0 + tmp3;
        let tmp13: f64 = tmp0 - tmp3;
        let tmp11: f64 = tmp1 + tmp2;
        let tmp12: f64 = tmp1 - tmp2;

        workspace[base] = tmp10 + tmp11;
        workspace[base + 4] = tmp10 - tmp11;

        let z1: f64 = (tmp12 + tmp13) * std::f64::consts::FRAC_1_SQRT_2;
        workspace[base + 2] = tmp13 + z1;
        workspace[base + 6] = tmp13 - z1;

        // Odd part
        let tmp10: f64 = tmp4 + tmp5;
        let tmp11: f64 = tmp5 + tmp6;
        let tmp12: f64 = tmp6 + tmp7;

        let z5: f64 = (tmp10 - tmp12) * 0.382683433;
        let z2: f64 = 0.541196100 * tmp10 + z5;
        let z4: f64 = 1.306562965 * tmp12 + z5;
        let z3: f64 = tmp11 * std::f64::consts::FRAC_1_SQRT_2;

        let z11: f64 = tmp7 + z3;
        let z13: f64 = tmp7 - z3;

        workspace[base + 5] = z13 + z2;
        workspace[base + 3] = z13 - z2;
        workspace[base + 1] = z11 + z4;
        workspace[base + 7] = z11 - z4;
    }

    // Pass 2: process columns
    for col in 0..8 {
        let tmp0: f64 = workspace[col] + workspace[col + 56];
        let tmp7: f64 = workspace[col] - workspace[col + 56];
        let tmp1: f64 = workspace[col + 8] + workspace[col + 48];
        let tmp6: f64 = workspace[col + 8] - workspace[col + 48];
        let tmp2: f64 = workspace[col + 16] + workspace[col + 40];
        let tmp5: f64 = workspace[col + 16] - workspace[col + 40];
        let tmp3: f64 = workspace[col + 24] + workspace[col + 32];
        let tmp4: f64 = workspace[col + 24] - workspace[col + 32];

        // Even part
        let tmp10: f64 = tmp0 + tmp3;
        let tmp13: f64 = tmp0 - tmp3;
        let tmp11: f64 = tmp1 + tmp2;
        let tmp12: f64 = tmp1 - tmp2;

        workspace[col] = tmp10 + tmp11;
        workspace[col + 32] = tmp10 - tmp11;

        let z1: f64 = (tmp12 + tmp13) * std::f64::consts::FRAC_1_SQRT_2;
        workspace[col + 16] = tmp13 + z1;
        workspace[col + 48] = tmp13 - z1;

        // Odd part
        let tmp10: f64 = tmp4 + tmp5;
        let tmp11: f64 = tmp5 + tmp6;
        let tmp12: f64 = tmp6 + tmp7;

        let z5: f64 = (tmp10 - tmp12) * 0.382683433;
        let z2: f64 = 0.541196100 * tmp10 + z5;
        let z4: f64 = 1.306562965 * tmp12 + z5;
        let z3: f64 = tmp11 * std::f64::consts::FRAC_1_SQRT_2;

        let z11: f64 = tmp7 + z3;
        let z13: f64 = tmp7 - z3;

        workspace[col + 40] = z13 + z2;
        workspace[col + 24] = z13 - z2;
        workspace[col + 8] = z11 + z4;
        workspace[col + 56] = z11 - z4;
    }

    // Convert to i32 output, dividing by AA&N scale factors to match
    // fdct_islow's output format.
    for (row, &row_scale) in AAN_SCALES.iter().enumerate() {
        for (col, &col_scale) in AAN_SCALES.iter().enumerate() {
            let idx: usize = row * 8 + col;
            let scale: f64 = row_scale * col_scale;
            // Divide by the 2-D AA&N scale factor to get islow-equivalent output.
            let val: f64 = workspace[idx] / scale;
            // Round to nearest integer (matching libjpeg behavior)
            output[idx] = if val >= 0.0 {
                (val + 0.5) as i32
            } else {
                (val - 0.5) as i32
            };
        }
    }
}

/// Returns the appropriate FDCT function for the given DCT method.
pub fn select_fdct(method: crate::common::types::DctMethod) -> fn(&[i16; 64], &mut [i32; 64]) {
    match method {
        crate::common::types::DctMethod::IsLow => fdct_islow,
        crate::common::types::DctMethod::IsFast => fdct_ifast,
        crate::common::types::DctMethod::Float => fdct_float,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdct_all_zeros_produces_zeros() {
        let input = [0i16; 64];
        let mut output = [0i32; 64];
        fdct_islow(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0, "expected 0 at index {i}, got {v}");
        }
    }

    #[test]
    fn fdct_dc_only_block() {
        // A uniform block of value 100 (level-shifted to 100-128 = -28)
        let input = [-28i16; 64];
        let mut output = [0i32; 64];
        fdct_islow(&input, &mut output);
        // The FDCT output is scaled up by a factor of 8 (one factor of sqrt(8)
        // per pass). For a uniform block, DC = value * 64 / 8 * 8 = value * 64.
        // But with the LL&M scaling: DC = value * 8 * 8 / (factor removed at
        // quantization). Actually: sum_of_inputs * scaling.
        // In practice: -28 * 8 (first pass) * 8 (second pass) / 4 (PASS1_BITS
        // removed in pass 2) = -28 * 64 / ... Let's just check the computed value.
        // For v = -28 uniform: DC = descale(256 * v, PASS1_BITS) = (-7168+2)>>2 = -1792
        assert_eq!(output[0], -1792);
        // All AC coefficients should be zero for a uniform block
        for i in 1..64 {
            assert_eq!(
                output[i], 0,
                "AC coefficient at {i} should be 0, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn fdct_known_pattern() {
        // Create a simple gradient pattern and verify the DCT produces
        // expected non-zero AC coefficients
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = (row * 8 + col) as i16 - 32;
            }
        }
        let mut output = [0i32; 64];
        fdct_islow(&input, &mut output);

        // DC should be the sum of all values * 8 / 64 = average * 8
        // Sum = sum of (i - 32) for i=0..63 = (0+1+...+63) - 32*64 = 2016 - 2048 = -32
        // DC = -32 * 8 / 8 = ... actually DC = sum / 8 * 8 = sum
        // The FDCT DC = sum of all input values (due to the factor-of-8 scaling)
        // Actually just check it's non-zero and reasonable
        assert_ne!(output[0], 0);

        // Horizontal gradient should produce non-zero AC coefficients
        let has_nonzero_ac = output[1..].iter().any(|&v| v != 0);
        assert!(
            has_nonzero_ac,
            "gradient should produce non-zero AC coefficients"
        );
    }

    #[test]
    fn fdct_symmetry_check() {
        // A horizontally symmetric block should produce zero odd-column coefficients
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..4 {
                let val = (col as i16 + 1) * 10;
                input[row * 8 + col] = val;
                input[row * 8 + (7 - col)] = val;
            }
        }
        let mut output = [0i32; 64];
        fdct_islow(&input, &mut output);

        // Odd-indexed columns (1, 3, 5, 7) in DCT output should be zero
        // for a horizontally symmetric input
        for row in 0..8 {
            for &col in &[1, 3, 5, 7] {
                assert_eq!(
                    output[row * 8 + col],
                    0,
                    "symmetry: DCT[{},{}] should be 0, got {}",
                    row,
                    col,
                    output[row * 8 + col]
                );
            }
        }
    }

    #[test]
    fn fdct_ifast_all_zeros_produces_zeros() {
        let input = [0i16; 64];
        let mut output = [0i32; 64];
        fdct_ifast(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0, "ifast: expected 0 at index {i}, got {v}");
        }
    }

    #[test]
    fn fdct_ifast_dc_only_block() {
        // Uniform block: all AC should be zero (or very close), DC should be close to islow
        let input = [-28i16; 64];
        let mut output_ifast = [0i32; 64];
        let mut output_islow = [0i32; 64];
        fdct_ifast(&input, &mut output_ifast);
        fdct_islow(&input, &mut output_islow);

        // DC should be within 5% of islow's DC
        let dc_diff: i32 = (output_ifast[0] - output_islow[0]).abs();
        assert!(
            dc_diff < (output_islow[0].abs() / 20).max(2),
            "ifast DC {} too far from islow DC {} (diff={})",
            output_ifast[0],
            output_islow[0],
            dc_diff
        );

        // AC coefficients should be zero for a uniform block
        for i in 1..64 {
            assert!(
                output_ifast[i].abs() <= 1,
                "ifast AC[{i}] should be ~0, got {}",
                output_ifast[i]
            );
        }
    }

    #[test]
    fn fdct_ifast_produces_nonzero_ac_for_gradient() {
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = (row * 8 + col) as i16 - 32;
            }
        }
        let mut output = [0i32; 64];
        fdct_ifast(&input, &mut output);
        assert_ne!(output[0], 0, "DC should be non-zero for gradient");
        let has_ac = output[1..].iter().any(|&v| v != 0);
        assert!(has_ac, "gradient should have non-zero AC in ifast");
    }

    #[test]
    fn fdct_float_all_zeros_produces_zeros() {
        let input = [0i16; 64];
        let mut output = [0i32; 64];
        fdct_float(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0, "float: expected 0 at index {i}, got {v}");
        }
    }

    #[test]
    fn fdct_float_dc_only_block() {
        // Uniform block: DC should be close to islow, all AC should be ~0
        let input = [-28i16; 64];
        let mut output_float = [0i32; 64];
        let mut output_islow = [0i32; 64];
        fdct_float(&input, &mut output_float);
        fdct_islow(&input, &mut output_islow);

        // Float should be very close to islow for DC
        let dc_diff: i32 = (output_float[0] - output_islow[0]).abs();
        assert!(
            dc_diff <= 1,
            "float DC {} should match islow DC {} (diff={})",
            output_float[0],
            output_islow[0],
            dc_diff
        );

        // AC should be zero for a uniform block
        for i in 1..64 {
            assert!(
                output_float[i].abs() <= 1,
                "float AC[{i}] should be ~0, got {}",
                output_float[i]
            );
        }
    }

    #[test]
    fn fdct_float_produces_nonzero_ac_for_gradient() {
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = (row * 8 + col) as i16 - 32;
            }
        }
        let mut output = [0i32; 64];
        fdct_float(&input, &mut output);
        assert_ne!(output[0], 0, "DC should be non-zero for gradient");
        let has_ac = output[1..].iter().any(|&v| v != 0);
        assert!(has_ac, "gradient should have non-zero AC in float");
    }

    #[test]
    fn fdct_float_close_to_islow_for_gradient() {
        // Float should generally produce values close to islow
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = (row * 8 + col) as i16 - 32;
            }
        }
        let mut output_float = [0i32; 64];
        let mut output_islow = [0i32; 64];
        fdct_float(&input, &mut output_float);
        fdct_islow(&input, &mut output_islow);

        // Allow up to 10% relative error or absolute error of 2
        for i in 0..64 {
            let diff: i32 = (output_float[i] - output_islow[i]).abs();
            let tolerance: i32 = (output_islow[i].abs() / 10).max(2);
            assert!(
                diff <= tolerance,
                "float[{i}]={} vs islow[{i}]={} differ by {} (tolerance={})",
                output_float[i],
                output_islow[i],
                diff,
                tolerance
            );
        }
    }

    #[test]
    fn select_fdct_returns_correct_variant() {
        use crate::common::types::DctMethod;
        let input = [-28i16; 64];

        let mut out_islow = [0i32; 64];
        let mut out_selected = [0i32; 64];

        fdct_islow(&input, &mut out_islow);
        let f = select_fdct(DctMethod::IsLow);
        f(&input, &mut out_selected);
        assert_eq!(
            out_islow, out_selected,
            "select_fdct(IsLow) should return fdct_islow"
        );

        let mut out_ifast = [0i32; 64];
        let mut out_selected2 = [0i32; 64];
        fdct_ifast(&input, &mut out_ifast);
        let f = select_fdct(DctMethod::IsFast);
        f(&input, &mut out_selected2);
        assert_eq!(
            out_ifast, out_selected2,
            "select_fdct(IsFast) should return fdct_ifast"
        );

        let mut out_float = [0i32; 64];
        let mut out_selected3 = [0i32; 64];
        fdct_float(&input, &mut out_float);
        let f = select_fdct(DctMethod::Float);
        f(&input, &mut out_selected3);
        assert_eq!(
            out_float, out_selected3,
            "select_fdct(Float) should return fdct_float"
        );
    }
}
