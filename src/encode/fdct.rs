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
}
