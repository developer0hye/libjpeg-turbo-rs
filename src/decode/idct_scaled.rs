/// Reduced-size IDCT implementations for scaled JPEG decoding.
///
/// These produce smaller output blocks from 8x8 DCT coefficients:
/// - idct_4x4: 4x4 output (1/2 scale)
/// - idct_2x2: 2x2 output (1/4 scale)
/// - idct_1x1: 1x1 output (1/8 scale)
///
/// Ported from libjpeg-turbo's jidctred.c.

// Fixed-point constants (CONST_BITS = 13)
const FIX_0_211164243: i32 = 1730;
const FIX_0_509795579: i32 = 4176;
const FIX_0_601344887: i32 = 4926;
const FIX_0_720959822: i32 = 5906;
const FIX_0_765366865: i32 = 6270;
const FIX_0_850430095: i32 = 6967;
const FIX_0_899976223: i32 = 7373;
const FIX_1_061594337: i32 = 8697;
const FIX_1_272758580: i32 = 10426;
const FIX_1_451774981: i32 = 11893;
const FIX_1_847759065: i32 = 15137;
const FIX_2_172734803: i32 = 17799;
const FIX_2_562915447: i32 = 20995;
const FIX_3_624509785: i32 = 29692;

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

#[inline(always)]
fn clamp_to_u8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// 4x4 reduced IDCT: produces 4x4 output from 8x8 DCT coefficients.
///
/// `coeffs` are raw DCT coefficients (not dequantized).
/// `quant` is the quantization table in natural order.
/// Output is 16 bytes in row-major order.
pub fn idct_4x4(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 16]) {
    let mut workspace = [0i32; 4 * 8]; // 4 columns × 8 rows

    // Pass 1: columns. Process columns 0-7, skip column 4.
    for col in 0..8 {
        if col == 4 {
            continue;
        }

        let c = |row: usize| -> i32 { coeffs[row * 8 + col] as i32 * quant[row * 8 + col] as i32 };

        // DC-only shortcut
        if c(1) == 0 && c(2) == 0 && c(3) == 0 && c(5) == 0 && c(6) == 0 && c(7) == 0 {
            let dcval = c(0) << PASS1_BITS;
            let ws_col = if col < 4 { col } else { col - 1 };
            workspace[0 * 7 + ws_col] = dcval;
            workspace[1 * 7 + ws_col] = dcval;
            workspace[2 * 7 + ws_col] = dcval;
            workspace[3 * 7 + ws_col] = dcval;
            continue;
        }

        // Even part
        let tmp0 = c(0) << (CONST_BITS + 1);
        let z2 = c(2);
        let z3 = c(6);
        let tmp2 = z2 * FIX_1_847759065 + z3 * (-FIX_0_765366865);
        let tmp10 = tmp0 + tmp2;
        let tmp12 = tmp0 - tmp2;

        // Odd part
        let z1 = c(7);
        let z2 = c(5);
        let z3 = c(3);
        let z4 = c(1);
        let tmp0 = z1 * (-FIX_0_211164243)
            + z2 * FIX_1_451774981
            + z3 * (-FIX_2_172734803)
            + z4 * FIX_1_061594337;
        let tmp2 = z1 * (-FIX_0_509795579)
            + z2 * (-FIX_0_601344887)
            + z3 * FIX_0_899976223
            + z4 * FIX_2_562915447;

        let ws_col = if col < 4 { col } else { col - 1 };
        workspace[0 * 7 + ws_col] = descale(tmp10 + tmp2, CONST_BITS - PASS1_BITS + 1);
        workspace[3 * 7 + ws_col] = descale(tmp10 - tmp2, CONST_BITS - PASS1_BITS + 1);
        workspace[1 * 7 + ws_col] = descale(tmp12 + tmp0, CONST_BITS - PASS1_BITS + 1);
        workspace[2 * 7 + ws_col] = descale(tmp12 - tmp0, CONST_BITS - PASS1_BITS + 1);
    }

    // Pass 2: rows. Process 4 rows, producing 4 output pixels each.
    for row in 0..4 {
        let w = |col: usize| -> i32 { workspace[row * 7 + col] };

        // DC-only shortcut
        if w(1) == 0 && w(2) == 0 && w(3) == 0 && w(4) == 0 && w(5) == 0 && w(6) == 0 {
            let dcval = clamp_to_u8(descale(w(0), PASS1_BITS + 3) + 128);
            output[row * 4] = dcval;
            output[row * 4 + 1] = dcval;
            output[row * 4 + 2] = dcval;
            output[row * 4 + 3] = dcval;
            continue;
        }

        // Even part
        let tmp0 = w(0) << (CONST_BITS + 1);
        let z2 = w(2);
        let z3 = w(5); // note: ws col 5 maps to original col 6 (skipping col 4)
        let tmp2 = z2 * FIX_1_847759065 + z3 * (-FIX_0_765366865);
        let tmp10 = tmp0 + tmp2;
        let tmp12 = tmp0 - tmp2;

        // Odd part
        let z1 = w(6); // original col 7
        let z2 = w(4); // original col 5
        let z3 = w(3); // original col 3
        let z4 = w(1); // original col 1
        let tmp0 = z1 * (-FIX_0_211164243)
            + z2 * FIX_1_451774981
            + z3 * (-FIX_2_172734803)
            + z4 * FIX_1_061594337;
        let tmp2 = z1 * (-FIX_0_509795579)
            + z2 * (-FIX_0_601344887)
            + z3 * FIX_0_899976223
            + z4 * FIX_2_562915447;

        let shift = CONST_BITS + PASS1_BITS + 3 + 1;
        output[row * 4] = clamp_to_u8(descale(tmp10 + tmp2, shift) + 128);
        output[row * 4 + 3] = clamp_to_u8(descale(tmp10 - tmp2, shift) + 128);
        output[row * 4 + 1] = clamp_to_u8(descale(tmp12 + tmp0, shift) + 128);
        output[row * 4 + 2] = clamp_to_u8(descale(tmp12 - tmp0, shift) + 128);
    }
}

/// 2x2 reduced IDCT: produces 2x2 output from 8x8 DCT coefficients.
pub fn idct_2x2(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 4]) {
    let mut workspace = [0i32; 2 * 8]; // 2 columns × 8 rows

    // Pass 1: columns. Only odd-indexed original columns contribute (0,1,3,5,7).
    // Skip columns 2, 4, 6.
    for col in 0..8 {
        if col == 2 || col == 4 || col == 6 {
            continue;
        }

        let c = |row: usize| -> i32 { coeffs[row * 8 + col] as i32 * quant[row * 8 + col] as i32 };

        // DC-only shortcut
        if c(1) == 0 && c(3) == 0 && c(5) == 0 && c(7) == 0 {
            let dcval = c(0) << PASS1_BITS;
            let ws_col = match col {
                0 => 0,
                1 => 1,
                3 => 2,
                5 => 3,
                7 => 4,
                _ => unreachable!(),
            };
            workspace[0 * 5 + ws_col] = dcval;
            workspace[1 * 5 + ws_col] = dcval;
            continue;
        }

        // Even part
        let tmp10 = c(0) << (CONST_BITS + 2);

        // Odd part
        let tmp0 = c(7) * (-FIX_0_720959822)
            + c(5) * FIX_0_850430095
            + c(3) * (-FIX_1_272758580)
            + c(1) * FIX_3_624509785;

        let ws_col = match col {
            0 => 0,
            1 => 1,
            3 => 2,
            5 => 3,
            7 => 4,
            _ => unreachable!(),
        };
        workspace[0 * 5 + ws_col] = descale(tmp10 + tmp0, CONST_BITS - PASS1_BITS + 2);
        workspace[1 * 5 + ws_col] = descale(tmp10 - tmp0, CONST_BITS - PASS1_BITS + 2);
    }

    // Pass 2: rows
    for row in 0..2 {
        let w = |col: usize| -> i32 { workspace[row * 5 + col] };

        // DC-only shortcut
        if w(1) == 0 && w(2) == 0 && w(3) == 0 && w(4) == 0 {
            let dcval = clamp_to_u8(descale(w(0), PASS1_BITS + 3) + 128);
            output[row * 2] = dcval;
            output[row * 2 + 1] = dcval;
            continue;
        }

        // Even part
        let tmp10 = w(0) << (CONST_BITS + 2);

        // Odd part: ws cols 1,2,3,4 map to original cols 1,3,5,7
        let tmp0 = w(4) * (-FIX_0_720959822)
            + w(3) * FIX_0_850430095
            + w(2) * (-FIX_1_272758580)
            + w(1) * FIX_3_624509785;

        let shift = CONST_BITS + PASS1_BITS + 3 + 2;
        output[row * 2] = clamp_to_u8(descale(tmp10 + tmp0, shift) + 128);
        output[row * 2 + 1] = clamp_to_u8(descale(tmp10 - tmp0, shift) + 128);
    }
}

/// 1x1 reduced IDCT: produces single pixel from DC coefficient.
pub fn idct_1x1(coeffs: &[i16; 64], quant: &[u16; 64]) -> u8 {
    let dcval = coeffs[0] as i32 * quant[0] as i32;
    clamp_to_u8(descale(dcval, 3) + 128)
}

/// 4x4 reduced IDCT writing directly to a strided buffer.
///
/// # Safety
/// `output` must point to at least `3 * stride + 4` writable bytes.
pub unsafe fn idct_4x4_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    let mut tmp = [0u8; 16];
    idct_4x4(coeffs, quant, &mut tmp);
    for row in 0..4 {
        std::ptr::copy_nonoverlapping(tmp.as_ptr().add(row * 4), output.add(row * stride), 4);
    }
}

/// 2x2 reduced IDCT writing directly to a strided buffer.
///
/// # Safety
/// `output` must point to at least `stride + 2` writable bytes.
pub unsafe fn idct_2x2_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    stride: usize,
) {
    let mut tmp = [0u8; 4];
    idct_2x2(coeffs, quant, &mut tmp);
    for row in 0..2 {
        std::ptr::copy_nonoverlapping(tmp.as_ptr().add(row * 2), output.add(row * stride), 2);
    }
}

/// 1x1 reduced IDCT writing to a strided buffer.
///
/// # Safety
/// `output` must point to at least 1 writable byte.
pub unsafe fn idct_1x1_strided(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: *mut u8,
    _stride: usize,
) {
    *output = idct_1x1(coeffs, quant);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_1x1_dc_only() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        // DC = 800, quant = 1 → dcval = 800, descale(800, 3) = 100, + 128 = 228
        coeffs[0] = 800;
        quant[0] = 1;
        assert_eq!(idct_1x1(&coeffs, &quant), 228);
    }

    #[test]
    fn idct_1x1_with_quant() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        // DC = 10, quant = 16 → dcval = 160, descale(160, 3) = 20, + 128 = 148
        coeffs[0] = 10;
        quant[0] = 16;
        assert_eq!(idct_1x1(&coeffs, &quant), 148);
    }

    #[test]
    fn idct_1x1_clamps_high() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 2000;
        quant[0] = 1;
        // descale(2000, 3) = 250, + 128 = 378 → clamped to 255
        assert_eq!(idct_1x1(&coeffs, &quant), 255);
    }

    #[test]
    fn idct_1x1_clamps_low() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = -2000;
        quant[0] = 1;
        // descale(-2000, 3) = -250, + 128 = -122 → clamped to 0
        assert_eq!(idct_1x1(&coeffs, &quant), 0);
    }

    #[test]
    fn idct_1x1_zero_dc() {
        let coeffs = [0i16; 64];
        let quant = [1u16; 64];
        // DC = 0 → descale(0, 3) = 0, + 128 = 128
        assert_eq!(idct_1x1(&coeffs, &quant), 128);
    }

    #[test]
    fn idct_4x4_dc_only_uniform() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 800;
        quant[0] = 1;
        let mut output = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut output);
        // All pixels should be the same value (DC-only → uniform block)
        let first = output[0];
        assert!(
            first > 100,
            "DC-only should produce reasonable value, got {}",
            first
        );
        for &v in &output {
            assert_eq!(v, first, "DC-only 4x4 block should be uniform");
        }
    }

    #[test]
    fn idct_2x2_dc_only_uniform() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 800;
        quant[0] = 1;
        let mut output = [0u8; 4];
        idct_2x2(&coeffs, &quant, &mut output);
        let first = output[0];
        assert!(first > 100);
        for &v in &output {
            assert_eq!(v, first, "DC-only 2x2 block should be uniform");
        }
    }

    #[test]
    fn idct_4x4_matches_1x1_dc() {
        // For DC-only input, all reduced IDCTs should agree on the pixel value
        let mut coeffs = [0i16; 64];
        let quant = [1u16; 64];
        coeffs[0] = 400;

        let val_1x1 = idct_1x1(&coeffs, &quant);

        let mut out_4x4 = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut out_4x4);

        let mut out_2x2 = [0u8; 4];
        idct_2x2(&coeffs, &quant, &mut out_2x2);

        // DC-only: all should produce the same value (or very close due to rounding)
        assert!(
            (out_4x4[0] as i16 - val_1x1 as i16).unsigned_abs() <= 1,
            "4x4 DC={} vs 1x1 DC={}",
            out_4x4[0],
            val_1x1
        );
        assert!(
            (out_2x2[0] as i16 - val_1x1 as i16).unsigned_abs() <= 1,
            "2x2 DC={} vs 1x1 DC={}",
            out_2x2[0],
            val_1x1
        );
    }

    #[test]
    fn idct_4x4_strided_writes_correctly() {
        let mut coeffs = [0i16; 64];
        let quant = [1u16; 64];
        coeffs[0] = 400;

        let stride = 16;
        let mut buf = vec![0xFFu8; stride * 4];
        unsafe {
            idct_4x4_strided(&coeffs, &quant, buf.as_mut_ptr(), stride);
        }

        // Check that 4 pixels are written per row at correct stride
        let mut expected = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut expected);

        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(buf[row * stride + col], expected[row * 4 + col]);
            }
            // Bytes beyond col 3 should be untouched
            assert_eq!(buf[row * stride + 4], 0xFF);
        }
    }

    #[test]
    fn idct_2x2_strided_writes_correctly() {
        let mut coeffs = [0i16; 64];
        let quant = [1u16; 64];
        coeffs[0] = 400;

        let stride = 16;
        let mut buf = vec![0xFFu8; stride * 2];
        unsafe {
            idct_2x2_strided(&coeffs, &quant, buf.as_mut_ptr(), stride);
        }

        let mut expected = [0u8; 4];
        idct_2x2(&coeffs, &quant, &mut expected);

        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(buf[row * stride + col], expected[row * 2 + col]);
            }
            assert_eq!(buf[row * stride + 2], 0xFF);
        }
    }
}
