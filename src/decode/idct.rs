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
    (x + (1 << (n - 1))) >> n
}

/// Perform one pass of the 1-D IDCT on 8 values.
/// Used for both column pass (pass 1) and row pass (pass 2).
#[inline(always)]
fn idct_1d(s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, s7: i32) -> [i32; 8] {
    // Even part
    let tmp0 = s0;
    let tmp1 = s2;
    let tmp2 = s4;
    let tmp3 = s6;

    let z1 = (tmp1 + tmp3) * F_0_541;
    let tmp2a = z1 + tmp3 * (-F_1_847);
    let tmp3a = z1 + tmp1 * F_0_765;

    let tmp0a = (tmp0 + tmp2) << CONST_BITS;
    let tmp1a = (tmp0 - tmp2) << CONST_BITS;

    let tmp10 = tmp0a + tmp3a;
    let tmp13 = tmp0a - tmp3a;
    let tmp11 = tmp1a + tmp2a;
    let tmp12 = tmp1a - tmp2a;

    // Odd part
    let z1 = s7 + s1;
    let z2 = s5 + s3;
    let z3 = s7 + s3;
    let z4 = s5 + s1;
    let z5 = (z3 + z4) * F_1_175;

    let tmp0 = s7 * F_0_298;
    let tmp1 = s5 * F_2_053;
    let tmp2 = s3 * F_3_072;
    let tmp3 = s1 * F_1_501;
    let z1 = z1 * (-F_0_899);
    let z2 = z2 * (-F_2_562);
    let z3 = z3 * (-F_1_961) + z5;
    let z4 = z4 * (-F_0_390) + z5;

    let tmp0 = tmp0 + z1 + z3;
    let tmp1 = tmp1 + z2 + z4;
    let tmp2 = tmp2 + z2 + z3;
    let tmp3 = tmp3 + z1 + z4;

    [
        tmp10 + tmp3,
        tmp11 + tmp2,
        tmp12 + tmp1,
        tmp13 + tmp0,
        tmp13 - tmp0,
        tmp12 - tmp1,
        tmp11 - tmp2,
        tmp10 - tmp3,
    ]
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
