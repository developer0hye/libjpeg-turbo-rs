/// Extended IDCT kernels for all 16 JPEG scaling factors.
/// Ported from libjpeg-turbo's jidctint.c.
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const fn fix(val: f64) -> i32 { (val * (1i64 << CONST_BITS) as f64 + 0.5) as i32 }
#[inline(always)]
fn clamp_to_u8(val: i32) -> u8 { val.clamp(0, 255) as u8 }
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_847759065: i32 = 15137;
const FIX_2_562915447: i32 = 20995;

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_3x3(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 9]) {
    let mut ws = [0i32; 9];
    for col in 0..3 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp0 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let tmp2 = c(2); let tmp12 = tmp2 * fix(0.707106781);
        let tmp10 = tmp0 + tmp12; let tmp2 = tmp0 - tmp12 - tmp12;
        let tmp0 = c(1) * fix(1.224744871);
        ws[0 + col] = (tmp10 + tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[6 + col] = (tmp10 - tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[3 + col] = tmp2 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..3 {
        let w = |c: usize| -> i32 { ws[row * 3 + c] };
        let tmp0 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let tmp2 = w(2); let tmp12 = tmp2 * fix(0.707106781);
        let tmp10 = tmp0 + tmp12; let tmp2 = tmp0 - tmp12 - tmp12;
        let tmp0 = w(1) * fix(1.224744871);
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 3] = clamp_to_u8(((tmp10 + tmp0) >> s) + 128);
        output[row * 3 + 2] = clamp_to_u8(((tmp10 - tmp0) >> s) + 128);
        output[row * 3 + 1] = clamp_to_u8((tmp2 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_5x5(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 25]) {
    let mut ws = [0i32; 25];
    for col in 0..5 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp12 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let z1 = (c(2) + c(4)) * fix(0.790569415);
        let z2 = (c(2) - c(4)) * fix(0.353553391);
        let z3 = tmp12 + z2; let tmp10 = z3 + z1; let tmp11 = z3 - z1;
        let tmp12 = tmp12 - (z2 << 2);
        let z1 = (c(1) + c(3)) * fix(0.831253876);
        let tmp0 = z1 + c(1) * fix(0.513743148);
        let tmp1 = z1 - c(3) * fix(2.176250899);
        ws[col] = (tmp10 + tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[20 + col] = (tmp10 - tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[5 + col] = (tmp11 + tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[15 + col] = (tmp11 - tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[10 + col] = tmp12 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..5 {
        let w = |c: usize| -> i32 { ws[row * 5 + c] };
        let tmp12 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let z1 = (w(2) + w(4)) * fix(0.790569415);
        let z2 = (w(2) - w(4)) * fix(0.353553391);
        let z3 = tmp12 + z2; let tmp10 = z3 + z1; let tmp11 = z3 - z1;
        let tmp12 = tmp12 - (z2 << 2);
        let z1 = (w(1) + w(3)) * fix(0.831253876);
        let tmp0 = z1 + w(1) * fix(0.513743148);
        let tmp1 = z1 - w(3) * fix(2.176250899);
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 5] = clamp_to_u8(((tmp10 + tmp0) >> s) + 128);
        output[row * 5 + 4] = clamp_to_u8(((tmp10 - tmp0) >> s) + 128);
        output[row * 5 + 1] = clamp_to_u8(((tmp11 + tmp1) >> s) + 128);
        output[row * 5 + 3] = clamp_to_u8(((tmp11 - tmp1) >> s) + 128);
        output[row * 5 + 2] = clamp_to_u8((tmp12 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_6x6(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 36]) {
    let mut ws = [0i32; 36];
    for col in 0..6 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp0 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let tmp10m = c(4) * fix(0.707106781);
        let tmp1 = tmp0 + tmp10m;
        let tmp11 = (tmp0 - tmp10m - tmp10m) >> (CONST_BITS - PASS1_BITS);
        let tmp0m = c(2) * fix(1.224744871);
        let tmp10 = tmp1 + tmp0m; let tmp12 = tmp1 - tmp0m;
        let (z1, z2, z3) = (c(1), c(3), c(5));
        let tmp1o = (z1 + z3) * fix(0.366025404);
        let tmp0o = tmp1o + ((z1 + z2) << CONST_BITS);
        let tmp2o = tmp1o + ((z3 - z2) << CONST_BITS);
        let tmp1f = (z1 - z2 - z3) << PASS1_BITS;
        ws[col] = (tmp10 + tmp0o) >> (CONST_BITS - PASS1_BITS);
        ws[30 + col] = (tmp10 - tmp0o) >> (CONST_BITS - PASS1_BITS);
        ws[6 + col] = tmp11 + tmp1f; ws[24 + col] = tmp11 - tmp1f;
        ws[12 + col] = (tmp12 + tmp2o) >> (CONST_BITS - PASS1_BITS);
        ws[18 + col] = (tmp12 - tmp2o) >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..6 {
        let w = |c: usize| -> i32 { ws[row * 6 + c] };
        let tmp0 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let tmp10m = w(4) * fix(0.707106781);
        let tmp1 = tmp0 + tmp10m; let tmp11 = tmp0 - tmp10m - tmp10m;
        let tmp0m = w(2) * fix(1.224744871);
        let tmp10 = tmp1 + tmp0m; let tmp12 = tmp1 - tmp0m;
        let (z1, z2, z3) = (w(1), w(3), w(5));
        let tmp1o = (z1 + z3) * fix(0.366025404);
        let tmp0o = tmp1o + ((z1 + z2) << CONST_BITS);
        let tmp2o = tmp1o + ((z3 - z2) << CONST_BITS);
        let tmp1f = (z1 - z2 - z3) << CONST_BITS;
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 6] = clamp_to_u8(((tmp10 + tmp0o) >> s) + 128);
        output[row * 6 + 5] = clamp_to_u8(((tmp10 - tmp0o) >> s) + 128);
        output[row * 6 + 1] = clamp_to_u8(((tmp11 + tmp1f) >> s) + 128);
        output[row * 6 + 4] = clamp_to_u8(((tmp11 - tmp1f) >> s) + 128);
        output[row * 6 + 2] = clamp_to_u8(((tmp12 + tmp2o) >> s) + 128);
        output[row * 6 + 3] = clamp_to_u8(((tmp12 - tmp2o) >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_7x7(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 49]) {
    let mut ws = [0i32; 49];
    for col in 0..7 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp13 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let (z1, z2, z3) = (c(2), c(4), c(6));
        let tmp10 = (z2 - z3) * fix(0.881747734);
        let tmp12 = (z1 - z2) * fix(0.314692123);
        let tmp11 = tmp10 + tmp12 + tmp13 - z2 * fix(1.841218003);
        let tmp0e = z1 + z3; let z2m = z2 - tmp0e;
        let tmp0b = tmp0e * fix(1.274162392) + tmp13;
        let tmp10 = tmp10 + tmp0b - z3 * fix(0.077722536);
        let tmp12 = tmp12 + tmp0b - z1 * fix(2.470602249);
        let tmp13 = tmp13 + z2m * fix(1.414213562);
        let (z1, z2, z3) = (c(1), c(3), c(5));
        let tmp1 = (z1 + z2) * fix(0.935414347);
        let tmp2i = (z1 - z2) * fix(0.170262339);
        let tmp0 = tmp1 - tmp2i; let tmp1 = tmp1 + tmp2i;
        let tmp2 = (z2 + z3) * (-fix(1.378756276)); let tmp1 = tmp1 + tmp2;
        let z2v = (z1 + z3) * fix(0.613604268);
        let tmp0 = tmp0 + z2v; let tmp2 = tmp2 + z2v + z3 * fix(1.870828693);
        ws[col] = (tmp10 + tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[42 + col] = (tmp10 - tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[7 + col] = (tmp11 + tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[35 + col] = (tmp11 - tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[14 + col] = (tmp12 + tmp2) >> (CONST_BITS - PASS1_BITS);
        ws[28 + col] = (tmp12 - tmp2) >> (CONST_BITS - PASS1_BITS);
        ws[21 + col] = tmp13 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..7 {
        let w = |c: usize| -> i32 { ws[row * 7 + c] };
        let tmp13 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let (z1, z2, z3) = (w(2), w(4), w(6));
        let tmp10 = (z2 - z3) * fix(0.881747734);
        let tmp12 = (z1 - z2) * fix(0.314692123);
        let tmp11 = tmp10 + tmp12 + tmp13 - z2 * fix(1.841218003);
        let tmp0e = z1 + z3; let z2m = z2 - tmp0e;
        let tmp0b = tmp0e * fix(1.274162392) + tmp13;
        let tmp10 = tmp10 + tmp0b - z3 * fix(0.077722536);
        let tmp12 = tmp12 + tmp0b - z1 * fix(2.470602249);
        let tmp13 = tmp13 + z2m * fix(1.414213562);
        let (z1, z2, z3) = (w(1), w(3), w(5));
        let tmp1 = (z1 + z2) * fix(0.935414347);
        let tmp2i = (z1 - z2) * fix(0.170262339);
        let tmp0 = tmp1 - tmp2i; let tmp1 = tmp1 + tmp2i;
        let tmp2 = (z2 + z3) * (-fix(1.378756276)); let tmp1 = tmp1 + tmp2;
        let z2v = (z1 + z3) * fix(0.613604268);
        let tmp0 = tmp0 + z2v; let tmp2 = tmp2 + z2v + z3 * fix(1.870828693);
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 7] = clamp_to_u8(((tmp10 + tmp0) >> s) + 128);
        output[row * 7 + 6] = clamp_to_u8(((tmp10 - tmp0) >> s) + 128);
        output[row * 7 + 1] = clamp_to_u8(((tmp11 + tmp1) >> s) + 128);
        output[row * 7 + 5] = clamp_to_u8(((tmp11 - tmp1) >> s) + 128);
        output[row * 7 + 2] = clamp_to_u8(((tmp12 + tmp2) >> s) + 128);
        output[row * 7 + 4] = clamp_to_u8(((tmp12 - tmp2) >> s) + 128);
        output[row * 7 + 3] = clamp_to_u8((tmp13 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_9x9(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 81]) {
    let mut ws = [0i32; 72];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp0 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let (z1, z2, z3) = (c(2), c(4), c(6));
        let tmp3 = z3 * fix(0.707106781); let tmp1 = tmp0 + tmp3; let tmp2 = tmp0 - tmp3 - tmp3;
        let tmp0 = (z1 - z2) * fix(0.707106781); let tmp11 = tmp2 + tmp0; let tmp14 = tmp2 - tmp0 - tmp0;
        let tmp0 = (z1 + z2) * fix(1.328926049); let tmp2 = z1 * fix(1.083350441); let tmp3 = z2 * fix(0.245575608);
        let tmp10 = tmp1 + tmp0 - tmp3; let tmp12 = tmp1 - tmp0 + tmp2; let tmp13 = tmp1 - tmp2 + tmp3;
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let z2 = z2 * (-fix(1.224744871));
        let tmp2 = (z1 + z3) * fix(0.909038955); let tmp3 = (z1 + z4) * fix(0.483689525);
        let tmp0 = tmp2 + tmp3 - z2; let tmp1 = (z3 - z4) * fix(1.392728481);
        let tmp2 = tmp2 + z2 - tmp1; let tmp3 = tmp3 + z2 + tmp1;
        let tmp1 = (z1 - z3 - z4) * fix(1.224744871);
        ws[col] = (tmp10 + tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp10 - tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp11 + tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = (tmp11 - tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp12 + tmp2) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp12 - tmp2) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = (tmp13 + tmp3) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp13 - tmp3) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = tmp14 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..9 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let tmp0 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let (z1, z2, z3) = (w(2), w(4), w(6));
        let tmp3 = z3 * fix(0.707106781); let tmp1 = tmp0 + tmp3; let tmp2 = tmp0 - tmp3 - tmp3;
        let tmp0 = (z1 - z2) * fix(0.707106781); let tmp11 = tmp2 + tmp0; let tmp14 = tmp2 - tmp0 - tmp0;
        let tmp0 = (z1 + z2) * fix(1.328926049); let tmp2 = z1 * fix(1.083350441); let tmp3 = z2 * fix(0.245575608);
        let tmp10 = tmp1 + tmp0 - tmp3; let tmp12 = tmp1 - tmp0 + tmp2; let tmp13 = tmp1 - tmp2 + tmp3;
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let z2 = z2 * (-fix(1.224744871));
        let tmp2 = (z1 + z3) * fix(0.909038955); let tmp3 = (z1 + z4) * fix(0.483689525);
        let tmp0 = tmp2 + tmp3 - z2; let tmp1 = (z3 - z4) * fix(1.392728481);
        let tmp2 = tmp2 + z2 - tmp1; let tmp3 = tmp3 + z2 + tmp1;
        let tmp1 = (z1 - z3 - z4) * fix(1.224744871);
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 9] = clamp_to_u8(((tmp10 + tmp0) >> s) + 128);
        output[row * 9 + 8] = clamp_to_u8(((tmp10 - tmp0) >> s) + 128);
        output[row * 9 + 1] = clamp_to_u8(((tmp11 + tmp1) >> s) + 128);
        output[row * 9 + 7] = clamp_to_u8(((tmp11 - tmp1) >> s) + 128);
        output[row * 9 + 2] = clamp_to_u8(((tmp12 + tmp2) >> s) + 128);
        output[row * 9 + 6] = clamp_to_u8(((tmp12 - tmp2) >> s) + 128);
        output[row * 9 + 3] = clamp_to_u8(((tmp13 + tmp3) >> s) + 128);
        output[row * 9 + 5] = clamp_to_u8(((tmp13 - tmp3) >> s) + 128);
        output[row * 9 + 4] = clamp_to_u8((tmp14 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_10x10(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 100]) {
    let mut ws = [0i32; 80];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let z3 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let z4 = c(4); let z1 = z4 * fix(1.144122806); let z2 = z4 * fix(0.437016024);
        let tmp10 = z3 + z1; let tmp11 = z3 - z2;
        let tmp22 = (z3 - ((z1 - z2) << 1)) >> (CONST_BITS - PASS1_BITS);
        let (z2, z3) = (c(2), c(6));
        let z1 = (z2 + z3) * fix(0.831253876);
        let tmp12 = z1 + z2 * fix(0.513743148); let tmp13 = z1 - z3 * fix(2.176250899);
        let tmp20 = tmp10 + tmp12; let tmp24 = tmp10 - tmp12;
        let tmp21 = tmp11 + tmp13; let tmp23 = tmp11 - tmp13;
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let tmp11 = z2 + z4; let tmp13 = z2 - z4;
        let tmp12 = tmp13 * fix(0.309016994); let z5 = z3 << CONST_BITS;
        let z2 = tmp11 * fix(0.951056516); let z4 = z5 + tmp12;
        let tmp10 = z1 * fix(1.396802247) + z2 + z4;
        let tmp14 = z1 * fix(0.221231742) - z2 + z4;
        let z2 = tmp11 * fix(0.587785252);
        let z4 = z5 - tmp12 - (tmp13 << (CONST_BITS - 1));
        let tmp12 = (z1 - tmp13 - z3) << PASS1_BITS;
        let tmp11 = z1 * fix(1.260073511) - z2 - z4;
        let tmp13 = z1 * fix(0.642039522) - z2 + z4;
        ws[col] = (tmp20 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp20 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp21 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = tmp22 + tmp12; ws[56 + col] = tmp22 - tmp12;
        ws[24 + col] = (tmp23 + tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp23 - tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = (tmp24 + tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp24 - tmp14) >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..10 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let z3 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let z4 = w(4); let z1 = z4 * fix(1.144122806); let z2 = z4 * fix(0.437016024);
        let tmp10 = z3 + z1; let tmp11 = z3 - z2;
        let tmp22 = z3 - ((z1 - z2) << 1);
        let (z2, z3) = (w(2), w(6));
        let z1 = (z2 + z3) * fix(0.831253876);
        let tmp12 = z1 + z2 * fix(0.513743148); let tmp13 = z1 - z3 * fix(2.176250899);
        let tmp20 = tmp10 + tmp12; let tmp24 = tmp10 - tmp12;
        let tmp21 = tmp11 + tmp13; let tmp23 = tmp11 - tmp13;
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let tmp11 = z2 + z4; let tmp13 = z2 - z4;
        let tmp12 = tmp13 * fix(0.309016994); let z3s = z3 << CONST_BITS;
        let z2 = tmp11 * fix(0.951056516); let z4 = z3s + tmp12;
        let tmp10 = z1 * fix(1.396802247) + z2 + z4;
        let tmp14 = z1 * fix(0.221231742) - z2 + z4;
        let z2 = tmp11 * fix(0.587785252);
        let z4 = z3s - tmp12 - (tmp13 << (CONST_BITS - 1));
        let tmp12 = ((z1 - tmp13) << CONST_BITS) - z3s;
        let tmp11 = z1 * fix(1.260073511) - z2 - z4;
        let tmp13 = z1 * fix(0.642039522) - z2 + z4;
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 10] = clamp_to_u8(((tmp20 + tmp10) >> s) + 128);
        output[row * 10 + 9] = clamp_to_u8(((tmp20 - tmp10) >> s) + 128);
        output[row * 10 + 1] = clamp_to_u8(((tmp21 + tmp11) >> s) + 128);
        output[row * 10 + 8] = clamp_to_u8(((tmp21 - tmp11) >> s) + 128);
        output[row * 10 + 2] = clamp_to_u8(((tmp22 + tmp12) >> s) + 128);
        output[row * 10 + 7] = clamp_to_u8(((tmp22 - tmp12) >> s) + 128);
        output[row * 10 + 3] = clamp_to_u8(((tmp23 + tmp13) >> s) + 128);
        output[row * 10 + 6] = clamp_to_u8(((tmp23 - tmp13) >> s) + 128);
        output[row * 10 + 4] = clamp_to_u8(((tmp24 + tmp14) >> s) + 128);
        output[row * 10 + 5] = clamp_to_u8(((tmp24 - tmp14) >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_11x11(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 121]) {
    let mut ws = [0i32; 88];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp10 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let (z1, z2, z3) = (c(2), c(4), c(6));
        let tmp20 = (z2 - z3) * fix(2.546640132); let tmp23 = (z2 - z1) * fix(0.430815045);
        let z4 = z1 + z3; let tmp24 = z4 * (-fix(1.155664402)); let z4 = z4 - z2;
        let tmp25 = tmp10 + z4 * fix(1.356927976);
        let tmp21 = tmp20 + tmp23 + tmp25 - z2 * fix(1.821790775);
        let tmp20 = tmp20 + tmp25 + z3 * fix(2.115825087);
        let tmp23 = tmp23 + tmp25 - z1 * fix(1.513598477);
        let tmp24 = tmp24 + tmp25;
        let tmp22 = tmp24 - z3 * fix(0.788749120);
        let tmp24 = tmp24 + z2 * fix(1.944413522) - z1 * fix(1.390975730);
        let tmp25 = tmp10 - z4 * fix(1.414213562);
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let tmp11 = z1 + z2;
        let tmp14 = (tmp11 + z3 + z4) * fix(0.398430003);
        let tmp11 = tmp11 * fix(0.887983902);
        let tmp12 = (z1 + z3) * fix(0.670361295);
        let tmp13 = tmp14 + (z1 + z4) * fix(0.366151574);
        let tmp10 = tmp11 + tmp12 + tmp13 - z1 * fix(0.923107866);
        let z1t = tmp14 - (z2 + z3) * fix(1.163011579);
        let tmp11 = tmp11 + z1t + z2 * fix(2.073276588);
        let tmp12 = tmp12 + z1t - z3 * fix(1.192193623);
        let z1t = (z2 + z4) * (-fix(1.798248910));
        let tmp11 = tmp11 + z1t; let tmp13 = tmp13 + z1t + z4 * fix(2.102458632);
        let tmp14 = tmp14 + z2 * (-fix(1.467221301)) + z3 * fix(1.001388905) - z4 * fix(1.684843907);
        ws[col] = (tmp20 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[80 + col] = (tmp20 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp21 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp22 + tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp22 - tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = (tmp23 + tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = (tmp23 - tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = (tmp24 + tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp24 - tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = tmp25 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..11 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let tmp10 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let (z1, z2, z3) = (w(2), w(4), w(6));
        let tmp20 = (z2 - z3) * fix(2.546640132); let tmp23 = (z2 - z1) * fix(0.430815045);
        let z4 = z1 + z3; let tmp24 = z4 * (-fix(1.155664402)); let z4 = z4 - z2;
        let tmp25 = tmp10 + z4 * fix(1.356927976);
        let tmp21 = tmp20 + tmp23 + tmp25 - z2 * fix(1.821790775);
        let tmp20 = tmp20 + tmp25 + z3 * fix(2.115825087);
        let tmp23 = tmp23 + tmp25 - z1 * fix(1.513598477);
        let tmp24 = tmp24 + tmp25;
        let tmp22 = tmp24 - z3 * fix(0.788749120);
        let tmp24 = tmp24 + z2 * fix(1.944413522) - z1 * fix(1.390975730);
        let tmp25 = tmp10 - z4 * fix(1.414213562);
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let tmp11 = z1 + z2;
        let tmp14 = (tmp11 + z3 + z4) * fix(0.398430003);
        let tmp11 = tmp11 * fix(0.887983902);
        let tmp12 = (z1 + z3) * fix(0.670361295);
        let tmp13 = tmp14 + (z1 + z4) * fix(0.366151574);
        let tmp10 = tmp11 + tmp12 + tmp13 - z1 * fix(0.923107866);
        let z1t = tmp14 - (z2 + z3) * fix(1.163011579);
        let tmp11 = tmp11 + z1t + z2 * fix(2.073276588);
        let tmp12 = tmp12 + z1t - z3 * fix(1.192193623);
        let z1t = (z2 + z4) * (-fix(1.798248910));
        let tmp11 = tmp11 + z1t; let tmp13 = tmp13 + z1t + z4 * fix(2.102458632);
        let tmp14 = tmp14 + z2 * (-fix(1.467221301)) + z3 * fix(1.001388905) - z4 * fix(1.684843907);
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 11] = clamp_to_u8(((tmp20 + tmp10) >> s) + 128);
        output[row * 11 + 10] = clamp_to_u8(((tmp20 - tmp10) >> s) + 128);
        output[row * 11 + 1] = clamp_to_u8(((tmp21 + tmp11) >> s) + 128);
        output[row * 11 + 9] = clamp_to_u8(((tmp21 - tmp11) >> s) + 128);
        output[row * 11 + 2] = clamp_to_u8(((tmp22 + tmp12) >> s) + 128);
        output[row * 11 + 8] = clamp_to_u8(((tmp22 - tmp12) >> s) + 128);
        output[row * 11 + 3] = clamp_to_u8(((tmp23 + tmp13) >> s) + 128);
        output[row * 11 + 7] = clamp_to_u8(((tmp23 - tmp13) >> s) + 128);
        output[row * 11 + 4] = clamp_to_u8(((tmp24 + tmp14) >> s) + 128);
        output[row * 11 + 6] = clamp_to_u8(((tmp24 - tmp14) >> s) + 128);
        output[row * 11 + 5] = clamp_to_u8((tmp25 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_12x12(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 144]) {
    let mut ws = [0i32; 96];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let z3 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let z4 = c(4) * fix(1.224744871); let tmp10 = z3 + z4; let tmp11 = z3 - z4;
        let z1 = c(2); let z4 = z1 * fix(1.366025404);
        let z1s = z1 << CONST_BITS; let z2 = c(6); let z2s = z2 << CONST_BITS;
        let tmp12 = z1s - z2s; let tmp21 = z3 + tmp12; let tmp24 = z3 - tmp12;
        let tmp12 = z4 + z2s; let tmp20 = tmp10 + tmp12; let tmp25 = tmp10 - tmp12;
        let tmp12 = z4 - z1s - z2s; let tmp22 = tmp11 + tmp12; let tmp23 = tmp11 - tmp12;
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let tmp11 = z2 * fix(1.306562965); let tmp14 = z2 * (-FIX_0_541196100);
        let tmp10 = z1 + z3; let tmp15 = (tmp10 + z4) * fix(0.860918669);
        let tmp12 = tmp15 + tmp10 * fix(0.261052384);
        let tmp10 = tmp12 + tmp11 + z1 * fix(0.280143716);
        let tmp13 = (z3 + z4) * (-fix(1.045510580));
        let tmp12 = tmp12 + tmp13 + tmp14 - z3 * fix(1.478575242);
        let tmp13 = tmp13 + tmp15 - tmp11 + z4 * fix(1.586706681);
        let tmp15 = tmp15 + tmp14 - z1 * fix(0.676326758) - z4 * fix(1.982889723);
        let z1 = z1 - z4; let z2 = z2 - z3;
        let z3 = (z1 + z2) * FIX_0_541196100;
        let tmp11 = z3 + z1 * FIX_0_765366865; let tmp14 = z3 - z2 * FIX_1_847759065;
        ws[col] = (tmp20 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[88 + col] = (tmp20 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[80 + col] = (tmp21 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp22 + tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp22 - tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = (tmp23 + tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp23 - tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = (tmp24 + tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = (tmp24 - tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp25 + tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp25 - tmp15) >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..12 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let z3 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let z4 = w(4) * fix(1.224744871); let tmp10 = z3 + z4; let tmp11 = z3 - z4;
        let z1 = w(2); let z4 = z1 * fix(1.366025404);
        let z1s = z1 << CONST_BITS; let z2 = w(6); let z2s = z2 << CONST_BITS;
        let tmp12 = z1s - z2s; let tmp21 = z3 + tmp12; let tmp24 = z3 - tmp12;
        let tmp12 = z4 + z2s; let tmp20 = tmp10 + tmp12; let tmp25 = tmp10 - tmp12;
        let tmp12 = z4 - z1s - z2s; let tmp22 = tmp11 + tmp12; let tmp23 = tmp11 - tmp12;
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let tmp11 = z2 * fix(1.306562965); let tmp14 = z2 * (-FIX_0_541196100);
        let tmp10 = z1 + z3; let tmp15 = (tmp10 + z4) * fix(0.860918669);
        let tmp12 = tmp15 + tmp10 * fix(0.261052384);
        let tmp10 = tmp12 + tmp11 + z1 * fix(0.280143716);
        let tmp13 = (z3 + z4) * (-fix(1.045510580));
        let tmp12 = tmp12 + tmp13 + tmp14 - z3 * fix(1.478575242);
        let tmp13 = tmp13 + tmp15 - tmp11 + z4 * fix(1.586706681);
        let tmp15 = tmp15 + tmp14 - z1 * fix(0.676326758) - z4 * fix(1.982889723);
        let z1 = z1 - z4; let z2 = z2 - z3;
        let z3 = (z1 + z2) * FIX_0_541196100;
        let tmp11 = z3 + z1 * FIX_0_765366865; let tmp14 = z3 - z2 * FIX_1_847759065;
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 12] = clamp_to_u8(((tmp20 + tmp10) >> s) + 128);
        output[row * 12 + 11] = clamp_to_u8(((tmp20 - tmp10) >> s) + 128);
        output[row * 12 + 1] = clamp_to_u8(((tmp21 + tmp11) >> s) + 128);
        output[row * 12 + 10] = clamp_to_u8(((tmp21 - tmp11) >> s) + 128);
        output[row * 12 + 2] = clamp_to_u8(((tmp22 + tmp12) >> s) + 128);
        output[row * 12 + 9] = clamp_to_u8(((tmp22 - tmp12) >> s) + 128);
        output[row * 12 + 3] = clamp_to_u8(((tmp23 + tmp13) >> s) + 128);
        output[row * 12 + 8] = clamp_to_u8(((tmp23 - tmp13) >> s) + 128);
        output[row * 12 + 4] = clamp_to_u8(((tmp24 + tmp14) >> s) + 128);
        output[row * 12 + 7] = clamp_to_u8(((tmp24 - tmp14) >> s) + 128);
        output[row * 12 + 5] = clamp_to_u8(((tmp25 + tmp15) >> s) + 128);
        output[row * 12 + 6] = clamp_to_u8(((tmp25 - tmp15) >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_13x13(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 169]) {
    let mut ws = [0i32; 104];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let z1 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let (z2, z3, z4) = (c(2), c(4), c(6));
        let tmp10 = z3 + z4; let tmp11 = z3 - z4;
        let tmp12 = tmp10 * fix(1.155388986); let tmp13 = tmp11 * fix(0.096834934) + z1;
        let tmp20 = z2 * fix(1.373119086) + tmp12 + tmp13;
        let tmp22 = z2 * fix(0.501487041) - tmp12 + tmp13;
        let tmp12 = tmp10 * fix(0.316450131); let tmp13 = tmp11 * fix(0.486914739) + z1;
        let tmp21 = z2 * fix(1.058554052) - tmp12 + tmp13;
        let tmp25 = z2 * (-fix(1.252223920)) + tmp12 + tmp13;
        let tmp12 = tmp10 * fix(0.435816023); let tmp13 = tmp11 * fix(0.937303064) - z1;
        let tmp23 = z2 * (-fix(0.170464608)) - tmp12 - tmp13;
        let tmp24 = z2 * (-fix(0.803364869)) + tmp12 - tmp13;
        let tmp26 = (tmp11 - z2) * fix(1.414213562) + z1;
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let tmp11 = (z1 + z2) * fix(1.322312651); let tmp12 = (z1 + z3) * fix(1.163874945);
        let tmp15 = z1 + z4; let tmp13 = tmp15 * fix(0.937797057);
        let tmp10 = tmp11 + tmp12 + tmp13 - z1 * fix(2.020082300);
        let tmp14 = (z2 + z3) * (-fix(0.338443458));
        let tmp11 = tmp11 + tmp14 + z2 * fix(0.837223564);
        let tmp12 = tmp12 + tmp14 - z3 * fix(1.572116027);
        let tmp14 = (z2 + z4) * (-fix(1.163874945));
        let tmp11 = tmp11 + tmp14; let tmp13 = tmp13 + tmp14 + z4 * fix(2.205608352);
        let tmp14 = (z3 + z4) * (-fix(0.657217813)); let tmp12 = tmp12 + tmp14; let tmp13 = tmp13 + tmp14;
        let tmp15 = tmp15 * fix(0.338443458);
        let tmp14 = tmp15 + z1 * fix(0.318774355) - z2 * fix(0.466105296);
        let z1t = (z3 - z2) * fix(0.937797057); let tmp14 = tmp14 + z1t;
        let tmp15 = tmp15 + z1t + z3 * fix(0.384515595) - z4 * fix(1.742345811);
        ws[col] = (tmp20 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[96 + col] = (tmp20 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[88 + col] = (tmp21 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp22 + tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[80 + col] = (tmp22 - tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = (tmp23 + tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp23 - tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = (tmp24 + tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp24 - tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp25 + tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = (tmp25 - tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = tmp26 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..13 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let z1 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let (z2, z3, z4) = (w(2), w(4), w(6));
        let tmp10 = z3 + z4; let tmp11 = z3 - z4;
        let tmp12 = tmp10 * fix(1.155388986); let tmp13 = tmp11 * fix(0.096834934) + z1;
        let tmp20 = z2 * fix(1.373119086) + tmp12 + tmp13;
        let tmp22 = z2 * fix(0.501487041) - tmp12 + tmp13;
        let tmp12 = tmp10 * fix(0.316450131); let tmp13 = tmp11 * fix(0.486914739) + z1;
        let tmp21 = z2 * fix(1.058554052) - tmp12 + tmp13;
        let tmp25 = z2 * (-fix(1.252223920)) + tmp12 + tmp13;
        let tmp12 = tmp10 * fix(0.435816023); let tmp13 = tmp11 * fix(0.937303064) - z1;
        let tmp23 = z2 * (-fix(0.170464608)) - tmp12 - tmp13;
        let tmp24 = z2 * (-fix(0.803364869)) + tmp12 - tmp13;
        let tmp26 = (tmp11 - z2) * fix(1.414213562) + z1;
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let tmp11 = (z1 + z2) * fix(1.322312651); let tmp12 = (z1 + z3) * fix(1.163874945);
        let tmp15 = z1 + z4; let tmp13 = tmp15 * fix(0.937797057);
        let tmp10 = tmp11 + tmp12 + tmp13 - z1 * fix(2.020082300);
        let tmp14 = (z2 + z3) * (-fix(0.338443458));
        let tmp11 = tmp11 + tmp14 + z2 * fix(0.837223564);
        let tmp12 = tmp12 + tmp14 - z3 * fix(1.572116027);
        let tmp14 = (z2 + z4) * (-fix(1.163874945));
        let tmp11 = tmp11 + tmp14; let tmp13 = tmp13 + tmp14 + z4 * fix(2.205608352);
        let tmp14 = (z3 + z4) * (-fix(0.657217813)); let tmp12 = tmp12 + tmp14; let tmp13 = tmp13 + tmp14;
        let tmp15 = tmp15 * fix(0.338443458);
        let tmp14 = tmp15 + z1 * fix(0.318774355) - z2 * fix(0.466105296);
        let z1t = (z3 - z2) * fix(0.937797057); let tmp14 = tmp14 + z1t;
        let tmp15 = tmp15 + z1t + z3 * fix(0.384515595) - z4 * fix(1.742345811);
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 13] = clamp_to_u8(((tmp20 + tmp10) >> s) + 128);
        output[row * 13 + 12] = clamp_to_u8(((tmp20 - tmp10) >> s) + 128);
        output[row * 13 + 1] = clamp_to_u8(((tmp21 + tmp11) >> s) + 128);
        output[row * 13 + 11] = clamp_to_u8(((tmp21 - tmp11) >> s) + 128);
        output[row * 13 + 2] = clamp_to_u8(((tmp22 + tmp12) >> s) + 128);
        output[row * 13 + 10] = clamp_to_u8(((tmp22 - tmp12) >> s) + 128);
        output[row * 13 + 3] = clamp_to_u8(((tmp23 + tmp13) >> s) + 128);
        output[row * 13 + 9] = clamp_to_u8(((tmp23 - tmp13) >> s) + 128);
        output[row * 13 + 4] = clamp_to_u8(((tmp24 + tmp14) >> s) + 128);
        output[row * 13 + 8] = clamp_to_u8(((tmp24 - tmp14) >> s) + 128);
        output[row * 13 + 5] = clamp_to_u8(((tmp25 + tmp15) >> s) + 128);
        output[row * 13 + 7] = clamp_to_u8(((tmp25 - tmp15) >> s) + 128);
        output[row * 13 + 6] = clamp_to_u8((tmp26 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_14x14(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 196]) {
    let mut ws = [0i32; 112];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let z1 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let z4 = c(4);
        let z2 = z4 * fix(1.274162392); let z3 = z4 * fix(0.314692123); let z4 = z4 * fix(0.881747734);
        let tmp10 = z1 + z2; let tmp11 = z1 + z3; let tmp12 = z1 - z4;
        let tmp23 = (z1 - ((z2 + z3 - z4) << 1)) >> (CONST_BITS - PASS1_BITS);
        let (z1, z2) = (c(2), c(6));
        let z3 = (z1 + z2) * fix(1.105676686);
        let tmp13 = z3 + z1 * fix(0.273079590); let tmp14 = z3 - z2 * fix(1.719280954);
        let tmp15 = z1 * fix(0.613604268) - z2 * fix(1.378756276);
        let tmp20 = tmp10 + tmp13; let tmp26 = tmp10 - tmp13;
        let tmp21 = tmp11 + tmp14; let tmp25 = tmp11 - tmp14;
        let tmp22 = tmp12 + tmp15; let tmp24 = tmp12 - tmp15;
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let tmp13 = z4 << CONST_BITS; let tmp14 = z1 + z3;
        let tmp11 = (z1 + z2) * fix(1.334852607); let tmp12 = tmp14 * fix(1.197448846);
        let tmp10 = tmp11 + tmp12 + tmp13 - z1 * fix(1.126980169);
        let tmp14 = tmp14 * fix(0.752406978); let tmp16 = tmp14 - z1 * fix(1.061150426);
        let z1 = z1 - z2; let tmp15 = z1 * fix(0.467085129) - tmp13; let tmp16 = tmp16 + tmp15;
        let z1 = z1 + z4;
        let z4t = (z2 + z3) * (-fix(0.158341681)) - tmp13;
        let tmp11 = tmp11 + z4t - z2 * fix(0.424103948);
        let tmp12 = tmp12 + z4t - z3 * fix(2.373959773);
        let z4t = (z3 - z2) * fix(1.405321284);
        let tmp14 = tmp14 + z4t + tmp13 - z3 * fix(1.6906431334);
        let tmp15 = tmp15 + z4t + z2 * fix(0.674957567);
        let tmp13 = (z1 - z3) << PASS1_BITS;
        ws[col] = (tmp20 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[104 + col] = (tmp20 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[96 + col] = (tmp21 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp22 + tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[88 + col] = (tmp22 - tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = tmp23 + tmp13; ws[80 + col] = tmp23 - tmp13;
        ws[32 + col] = (tmp24 + tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp24 - tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp25 + tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp25 - tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp26 + tmp16) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = (tmp26 - tmp16) >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..14 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let z1 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let z4 = w(4);
        let z2 = z4 * fix(1.274162392); let z3 = z4 * fix(0.314692123); let z4 = z4 * fix(0.881747734);
        let tmp10 = z1 + z2; let tmp11 = z1 + z3; let tmp12 = z1 - z4;
        let tmp23 = z1 - ((z2 + z3 - z4) << 1);
        let (z1, z2) = (w(2), w(6));
        let z3 = (z1 + z2) * fix(1.105676686);
        let tmp13 = z3 + z1 * fix(0.273079590); let tmp14 = z3 - z2 * fix(1.719280954);
        let tmp15 = z1 * fix(0.613604268) - z2 * fix(1.378756276);
        let tmp20 = tmp10 + tmp13; let tmp26 = tmp10 - tmp13;
        let tmp21 = tmp11 + tmp14; let tmp25 = tmp11 - tmp14;
        let tmp22 = tmp12 + tmp15; let tmp24 = tmp12 - tmp15;
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let z4s = z4 << CONST_BITS; let tmp14 = z1 + z3;
        let tmp11 = (z1 + z2) * fix(1.334852607); let tmp12 = tmp14 * fix(1.197448846);
        let tmp10 = tmp11 + tmp12 + z4s - z1 * fix(1.126980169);
        let tmp14 = tmp14 * fix(0.752406978); let tmp16 = tmp14 - z1 * fix(1.061150426);
        let z1 = z1 - z2; let tmp15 = z1 * fix(0.467085129) - z4s; let tmp16 = tmp16 + tmp15;
        let tmp13 = (z2 + z3) * (-fix(0.158341681)) - z4s;
        let tmp11 = tmp11 + tmp13 - z2 * fix(0.424103948);
        let tmp12 = tmp12 + tmp13 - z3 * fix(2.373959773);
        let tmp13 = (z3 - z2) * fix(1.405321284);
        let tmp14 = tmp14 + tmp13 + z4s - z3 * fix(1.6906431334);
        let tmp15 = tmp15 + tmp13 + z2 * fix(0.674957567);
        let tmp13 = ((z1 - z3) << CONST_BITS) + z4s;
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 14] = clamp_to_u8(((tmp20 + tmp10) >> s) + 128);
        output[row * 14 + 13] = clamp_to_u8(((tmp20 - tmp10) >> s) + 128);
        output[row * 14 + 1] = clamp_to_u8(((tmp21 + tmp11) >> s) + 128);
        output[row * 14 + 12] = clamp_to_u8(((tmp21 - tmp11) >> s) + 128);
        output[row * 14 + 2] = clamp_to_u8(((tmp22 + tmp12) >> s) + 128);
        output[row * 14 + 11] = clamp_to_u8(((tmp22 - tmp12) >> s) + 128);
        output[row * 14 + 3] = clamp_to_u8(((tmp23 + tmp13) >> s) + 128);
        output[row * 14 + 10] = clamp_to_u8(((tmp23 - tmp13) >> s) + 128);
        output[row * 14 + 4] = clamp_to_u8(((tmp24 + tmp14) >> s) + 128);
        output[row * 14 + 9] = clamp_to_u8(((tmp24 - tmp14) >> s) + 128);
        output[row * 14 + 5] = clamp_to_u8(((tmp25 + tmp15) >> s) + 128);
        output[row * 14 + 8] = clamp_to_u8(((tmp25 - tmp15) >> s) + 128);
        output[row * 14 + 6] = clamp_to_u8(((tmp26 + tmp16) >> s) + 128);
        output[row * 14 + 7] = clamp_to_u8(((tmp26 - tmp16) >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_15x15(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 225]) {
    let mut ws = [0i32; 120];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let z1 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let (z2, z3, z4) = (c(2), c(4), c(6));
        let tmp10 = z4 * fix(0.437016024); let tmp11 = z4 * fix(1.144122806);
        let tmp12 = z1 - tmp10; let tmp13 = z1 + tmp11;
        let z1 = z1 - ((tmp11 - tmp10) << 1);
        let z4 = z2 - z3; let z3 = z3 + z2;
        let tmp10 = z3 * fix(1.337628990); let tmp11 = z4 * fix(0.045680613);
        let z2 = z2 * fix(1.439773946);
        let tmp20 = tmp13 + tmp10 + tmp11; let tmp23 = tmp12 - tmp10 + tmp11 + z2;
        let tmp10 = z3 * fix(0.547059574); let tmp11 = z4 * fix(0.399234004);
        let tmp25 = tmp13 - tmp10 - tmp11; let tmp26 = tmp12 + tmp10 - tmp11 - z2;
        let tmp10 = z3 * fix(0.790569415); let tmp11 = z4 * fix(0.353553391);
        let tmp21 = tmp12 + tmp10 + tmp11; let tmp24 = tmp13 - tmp10 + tmp11;
        let tmp11 = tmp11 + tmp11; let tmp22 = z1 + tmp11; let tmp27 = z1 - tmp11 - tmp11;
        let (z1, z2) = (c(1), c(3));
        let z4 = c(5); let z3 = z4 * fix(1.224744871); let z4 = c(7);
        let tmp13 = z2 - z4;
        let tmp15 = (z1 + tmp13) * fix(0.831253876);
        let tmp11 = tmp15 + z1 * fix(0.513743148); let tmp14 = tmp15 - tmp13 * fix(2.176250899);
        let tmp13 = z2 * (-fix(0.831253876)); let tmp15 = z2 * (-fix(1.344997024));
        let z2 = z1 - z4; let tmp12 = z3 + z2 * fix(1.406466353);
        let tmp10 = tmp12 + z4 * fix(2.457431844) - tmp15;
        let tmp16 = tmp12 - z1 * fix(1.112434820) + tmp13;
        let tmp12 = z2 * fix(1.224744871) - z3;
        let z2 = (z1 + z4) * fix(0.575212477);
        let tmp13 = tmp13 + z2 + z1 * fix(0.475753014) - z3;
        let tmp15 = tmp15 + z2 - z4 * fix(0.869244010) + z3;
        ws[col] = (tmp20 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[112 + col] = (tmp20 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[104 + col] = (tmp21 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp22 + tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[96 + col] = (tmp22 - tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = (tmp23 + tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[88 + col] = (tmp23 - tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = (tmp24 + tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[80 + col] = (tmp24 - tmp14) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp25 + tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp25 - tmp15) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp26 + tmp16) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp26 - tmp16) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = tmp27 >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..15 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let z1 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let (z2, z3, z4) = (w(2), w(4), w(6));
        let tmp10 = z4 * fix(0.437016024); let tmp11 = z4 * fix(1.144122806);
        let tmp12 = z1 - tmp10; let tmp13 = z1 + tmp11;
        let z1 = z1 - ((tmp11 - tmp10) << 1);
        let z4 = z2 - z3; let z3 = z3 + z2;
        let tmp10 = z3 * fix(1.337628990); let tmp11 = z4 * fix(0.045680613);
        let z2 = z2 * fix(1.439773946);
        let tmp20 = tmp13 + tmp10 + tmp11; let tmp23 = tmp12 - tmp10 + tmp11 + z2;
        let tmp10 = z3 * fix(0.547059574); let tmp11 = z4 * fix(0.399234004);
        let tmp25 = tmp13 - tmp10 - tmp11; let tmp26 = tmp12 + tmp10 - tmp11 - z2;
        let tmp10 = z3 * fix(0.790569415); let tmp11 = z4 * fix(0.353553391);
        let tmp21 = tmp12 + tmp10 + tmp11; let tmp24 = tmp13 - tmp10 + tmp11;
        let tmp11 = tmp11 + tmp11; let tmp22 = z1 + tmp11; let tmp27 = z1 - tmp11 - tmp11;
        let (z1, z2) = (w(1), w(3));
        let z4 = w(5); let z3 = z4 * fix(1.224744871); let z4 = w(7);
        let tmp13 = z2 - z4;
        let tmp15 = (z1 + tmp13) * fix(0.831253876);
        let tmp11 = tmp15 + z1 * fix(0.513743148); let tmp14 = tmp15 - tmp13 * fix(2.176250899);
        let tmp13 = z2 * (-fix(0.831253876)); let tmp15 = z2 * (-fix(1.344997024));
        let z2 = z1 - z4; let tmp12 = z3 + z2 * fix(1.406466353);
        let tmp10 = tmp12 + z4 * fix(2.457431844) - tmp15;
        let tmp16 = tmp12 - z1 * fix(1.112434820) + tmp13;
        let tmp12 = z2 * fix(1.224744871) - z3;
        let z2 = (z1 + z4) * fix(0.575212477);
        let tmp13 = tmp13 + z2 + z1 * fix(0.475753014) - z3;
        let tmp15 = tmp15 + z2 - z4 * fix(0.869244010) + z3;
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 15] = clamp_to_u8(((tmp20 + tmp10) >> s) + 128);
        output[row * 15 + 14] = clamp_to_u8(((tmp20 - tmp10) >> s) + 128);
        output[row * 15 + 1] = clamp_to_u8(((tmp21 + tmp11) >> s) + 128);
        output[row * 15 + 13] = clamp_to_u8(((tmp21 - tmp11) >> s) + 128);
        output[row * 15 + 2] = clamp_to_u8(((tmp22 + tmp12) >> s) + 128);
        output[row * 15 + 12] = clamp_to_u8(((tmp22 - tmp12) >> s) + 128);
        output[row * 15 + 3] = clamp_to_u8(((tmp23 + tmp13) >> s) + 128);
        output[row * 15 + 11] = clamp_to_u8(((tmp23 - tmp13) >> s) + 128);
        output[row * 15 + 4] = clamp_to_u8(((tmp24 + tmp14) >> s) + 128);
        output[row * 15 + 10] = clamp_to_u8(((tmp24 - tmp14) >> s) + 128);
        output[row * 15 + 5] = clamp_to_u8(((tmp25 + tmp15) >> s) + 128);
        output[row * 15 + 9] = clamp_to_u8(((tmp25 - tmp15) >> s) + 128);
        output[row * 15 + 6] = clamp_to_u8(((tmp26 + tmp16) >> s) + 128);
        output[row * 15 + 8] = clamp_to_u8(((tmp26 - tmp16) >> s) + 128);
        output[row * 15 + 7] = clamp_to_u8((tmp27 >> s) + 128);
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn idct_16x16(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 256]) {
    let mut ws = [0i32; 128];
    for col in 0..8 {
        let c = |r: usize| -> i32 { coeffs[r * 8 + col] as i32 * quant[r * 8 + col] as i32 };
        let tmp0 = (c(0) << CONST_BITS) + (1 << (CONST_BITS - PASS1_BITS - 1));
        let z1 = c(4); let tmp1 = z1 * fix(1.306562965); let tmp2 = z1 * FIX_0_541196100;
        let tmp10 = tmp0 + tmp1; let tmp11 = tmp0 - tmp1; let tmp12 = tmp0 + tmp2; let tmp13 = tmp0 - tmp2;
        let (z1, z2) = (c(2), c(6)); let z3 = z1 - z2;
        let z4 = z3 * fix(0.275899379); let z3 = z3 * fix(1.387039845);
        let tmp0 = z3 + z2 * FIX_2_562915447; let tmp1 = z4 + z1 * FIX_0_899976223;
        let tmp2 = z3 - z1 * fix(0.601344887); let tmp3 = z4 - z2 * fix(0.509795579);
        let tmp20 = tmp10 + tmp0; let tmp27 = tmp10 - tmp0;
        let tmp21 = tmp12 + tmp1; let tmp26 = tmp12 - tmp1;
        let tmp22 = tmp13 + tmp2; let tmp25 = tmp13 - tmp2;
        let tmp23 = tmp11 + tmp3; let tmp24 = tmp11 - tmp3;
        let (z1, z2, z3, z4) = (c(1), c(3), c(5), c(7));
        let tmp11 = z1 + z3;
        let tmp1 = (z1 + z2) * fix(1.353318001); let tmp2 = tmp11 * fix(1.247225013);
        let tmp3 = (z1 + z4) * fix(1.093201867); let tmp10 = (z1 - z4) * fix(0.897167586);
        let tmp11 = tmp11 * fix(0.666655658); let tmp12 = (z1 - z2) * fix(0.410524528);
        let tmp0 = tmp1 + tmp2 + tmp3 - z1 * fix(2.286341144);
        let tmp13 = tmp10 + tmp11 + tmp12 - z1 * fix(1.835730603);
        let z1t = (z2 + z3) * fix(0.138617169);
        let tmp1 = tmp1 + z1t + z2 * fix(0.071888074); let tmp2 = tmp2 + z1t - z3 * fix(1.125726048);
        let z1t = (z3 - z2) * fix(1.407403738);
        let tmp11 = tmp11 + z1t - z3 * fix(0.766367282); let tmp12 = tmp12 + z1t + z2 * fix(1.971951411);
        let z2 = z2 + z4; let z1t = z2 * (-fix(0.666655658));
        let tmp1 = tmp1 + z1t; let tmp3 = tmp3 + z1t + z4 * fix(1.065388962);
        let z2 = z2 * (-fix(1.247225013));
        let tmp10 = tmp10 + z2 + z4 * fix(3.141271809); let tmp12 = tmp12 + z2;
        let z2 = (z3 + z4) * (-fix(1.353318001)); let tmp2 = tmp2 + z2; let tmp3 = tmp3 + z2;
        let z2 = (z4 - z3) * fix(0.410524528); let tmp10 = tmp10 + z2; let tmp11 = tmp11 + z2;
        ws[col] = (tmp20 + tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[120 + col] = (tmp20 - tmp0) >> (CONST_BITS - PASS1_BITS);
        ws[8 + col] = (tmp21 + tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[112 + col] = (tmp21 - tmp1) >> (CONST_BITS - PASS1_BITS);
        ws[16 + col] = (tmp22 + tmp2) >> (CONST_BITS - PASS1_BITS);
        ws[104 + col] = (tmp22 - tmp2) >> (CONST_BITS - PASS1_BITS);
        ws[24 + col] = (tmp23 + tmp3) >> (CONST_BITS - PASS1_BITS);
        ws[96 + col] = (tmp23 - tmp3) >> (CONST_BITS - PASS1_BITS);
        ws[32 + col] = (tmp24 + tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[88 + col] = (tmp24 - tmp10) >> (CONST_BITS - PASS1_BITS);
        ws[40 + col] = (tmp25 + tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[80 + col] = (tmp25 - tmp11) >> (CONST_BITS - PASS1_BITS);
        ws[48 + col] = (tmp26 + tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[72 + col] = (tmp26 - tmp12) >> (CONST_BITS - PASS1_BITS);
        ws[56 + col] = (tmp27 + tmp13) >> (CONST_BITS - PASS1_BITS);
        ws[64 + col] = (tmp27 - tmp13) >> (CONST_BITS - PASS1_BITS);
    }
    for row in 0..16 {
        let w = |c: usize| -> i32 { ws[row * 8 + c] };
        let tmp0 = (w(0) + (1 << (PASS1_BITS + 2))) << CONST_BITS;
        let z1 = w(4); let tmp1 = z1 * fix(1.306562965); let tmp2 = z1 * FIX_0_541196100;
        let tmp10 = tmp0 + tmp1; let tmp11 = tmp0 - tmp1; let tmp12 = tmp0 + tmp2; let tmp13 = tmp0 - tmp2;
        let (z1, z2) = (w(2), w(6)); let z3 = z1 - z2;
        let z4 = z3 * fix(0.275899379); let z3 = z3 * fix(1.387039845);
        let tmp0 = z3 + z2 * FIX_2_562915447; let tmp1 = z4 + z1 * FIX_0_899976223;
        let tmp2 = z3 - z1 * fix(0.601344887); let tmp3 = z4 - z2 * fix(0.509795579);
        let tmp20 = tmp10 + tmp0; let tmp27 = tmp10 - tmp0;
        let tmp21 = tmp12 + tmp1; let tmp26 = tmp12 - tmp1;
        let tmp22 = tmp13 + tmp2; let tmp25 = tmp13 - tmp2;
        let tmp23 = tmp11 + tmp3; let tmp24 = tmp11 - tmp3;
        let (z1, z2, z3, z4) = (w(1), w(3), w(5), w(7));
        let tmp11 = z1 + z3;
        let tmp1 = (z1 + z2) * fix(1.353318001); let tmp2 = tmp11 * fix(1.247225013);
        let tmp3 = (z1 + z4) * fix(1.093201867); let tmp10 = (z1 - z4) * fix(0.897167586);
        let tmp11 = tmp11 * fix(0.666655658); let tmp12 = (z1 - z2) * fix(0.410524528);
        let tmp0 = tmp1 + tmp2 + tmp3 - z1 * fix(2.286341144);
        let tmp13 = tmp10 + tmp11 + tmp12 - z1 * fix(1.835730603);
        let z1t = (z2 + z3) * fix(0.138617169);
        let tmp1 = tmp1 + z1t + z2 * fix(0.071888074); let tmp2 = tmp2 + z1t - z3 * fix(1.125726048);
        let z1t = (z3 - z2) * fix(1.407403738);
        let tmp11 = tmp11 + z1t - z3 * fix(0.766367282); let tmp12 = tmp12 + z1t + z2 * fix(1.971951411);
        let z2 = z2 + z4; let z1t = z2 * (-fix(0.666655658));
        let tmp1 = tmp1 + z1t; let tmp3 = tmp3 + z1t + z4 * fix(1.065388962);
        let z2 = z2 * (-fix(1.247225013));
        let tmp10 = tmp10 + z2 + z4 * fix(3.141271809); let tmp12 = tmp12 + z2;
        let z2 = (z3 + z4) * (-fix(1.353318001)); let tmp2 = tmp2 + z2; let tmp3 = tmp3 + z2;
        let z2 = (z4 - z3) * fix(0.410524528); let tmp10 = tmp10 + z2; let tmp11 = tmp11 + z2;
        let s = CONST_BITS + PASS1_BITS + 3;
        output[row * 16] = clamp_to_u8(((tmp20 + tmp0) >> s) + 128);
        output[row * 16 + 15] = clamp_to_u8(((tmp20 - tmp0) >> s) + 128);
        output[row * 16 + 1] = clamp_to_u8(((tmp21 + tmp1) >> s) + 128);
        output[row * 16 + 14] = clamp_to_u8(((tmp21 - tmp1) >> s) + 128);
        output[row * 16 + 2] = clamp_to_u8(((tmp22 + tmp2) >> s) + 128);
        output[row * 16 + 13] = clamp_to_u8(((tmp22 - tmp2) >> s) + 128);
        output[row * 16 + 3] = clamp_to_u8(((tmp23 + tmp3) >> s) + 128);
        output[row * 16 + 12] = clamp_to_u8(((tmp23 - tmp3) >> s) + 128);
        output[row * 16 + 4] = clamp_to_u8(((tmp24 + tmp10) >> s) + 128);
        output[row * 16 + 11] = clamp_to_u8(((tmp24 - tmp10) >> s) + 128);
        output[row * 16 + 5] = clamp_to_u8(((tmp25 + tmp11) >> s) + 128);
        output[row * 16 + 10] = clamp_to_u8(((tmp25 - tmp11) >> s) + 128);
        output[row * 16 + 6] = clamp_to_u8(((tmp26 + tmp12) >> s) + 128);
        output[row * 16 + 9] = clamp_to_u8(((tmp26 - tmp12) >> s) + 128);
        output[row * 16 + 7] = clamp_to_u8(((tmp27 + tmp13) >> s) + 128);
        output[row * 16 + 8] = clamp_to_u8(((tmp27 - tmp13) >> s) + 128);
    }
}

macro_rules! strided_wrapper {
    ($name:ident, $inner:ident, $n:expr) => {
        pub unsafe fn $name(coeffs: &[i16; 64], quant: &[u16; 64], output: *mut u8, stride: usize) {
            let mut tmp = [0u8; $n * $n];
            $inner(coeffs, quant, &mut tmp);
            for row in 0..$n {
                std::ptr::copy_nonoverlapping(tmp.as_ptr().add(row * $n), output.add(row * stride), $n);
            }
        }
    };
}
strided_wrapper!(idct_3x3_strided, idct_3x3, 3);
strided_wrapper!(idct_5x5_strided, idct_5x5, 5);
strided_wrapper!(idct_6x6_strided, idct_6x6, 6);
strided_wrapper!(idct_7x7_strided, idct_7x7, 7);
strided_wrapper!(idct_9x9_strided, idct_9x9, 9);
strided_wrapper!(idct_10x10_strided, idct_10x10, 10);
strided_wrapper!(idct_11x11_strided, idct_11x11, 11);
strided_wrapper!(idct_12x12_strided, idct_12x12, 12);
strided_wrapper!(idct_13x13_strided, idct_13x13, 13);
strided_wrapper!(idct_14x14_strided, idct_14x14, 14);
strided_wrapper!(idct_15x15_strided, idct_15x15, 15);
strided_wrapper!(idct_16x16_strided, idct_16x16, 16);
