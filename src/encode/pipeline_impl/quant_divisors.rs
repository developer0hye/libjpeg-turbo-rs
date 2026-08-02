use super::QuantDivisors;

/// Find highest set bit position (1-indexed). Returns 0 for val=0.
/// Port of C libjpeg-turbo's `flss` from jcdctmgr.c.
pub(super) fn flss(val: u16) -> i32 {
    if val == 0 {
        return 0;
    }
    16 - val.leading_zeros() as i32
}

/// Compute adaptive-precision reciprocal for exact SIMD quantization.
/// Port of C libjpeg-turbo's `compute_reciprocal` from jcdctmgr.c.
///
/// Returns (reciprocal, correction, scale, shift).
/// - NEON uses (reciprocal, correction, shift) with per-element variable shift.
/// - AVX2 uses (reciprocal, correction, scale) with two `pmulhuw` ops (matching C).
pub fn compute_reciprocal(divisor: u16) -> (u16, u16, u16, i16) {
    if divisor <= 1 {
        // scale=1 for the identity case (matches C: dtbl[DCTSIZE2*2] = 1)
        return (1, 0, 1, -(core::mem::size_of::<i16>() as i16 * 8));
    }

    let b: i32 = flss(divisor) - 1;
    let r: i32 = 16 + b; // adaptive precision

    let fq: u32 = (1u32 << r) / divisor as u32;
    let fr: u32 = (1u32 << r) % divisor as u32;

    let mut recip: u32 = fq;
    let mut corr: u16 = divisor / 2;
    let mut r: i32 = r;

    if fr == 0 {
        // Divisor is power of two: fq is one bit too large, adjust
        recip >>= 1;
        r -= 1;
    } else if fr <= (divisor as u32 / 2) {
        // Fractional part < 0.5: round down, bump correction
        corr += 1;
    } else {
        // Fractional part > 0.5: round up
        recip += 1;
    }

    let shift: i16 = (r - 16) as i16;
    // Scale for AVX2: replaces per-element variable shift with a second mulhi.
    // scale = 1 << (32 - r), so mulhi(x, scale) == x >> (r - 16) == x >> shift.
    // Matches C: dtbl[DCTSIZE2 * 2] = (DCTELEM)(1 << (sizeof(DCTELEM)*8*2 - r))
    let scale: u16 = (1u32 << (32 - r)) as u16;
    (recip as u16, corr, scale, shift)
}

/// Scale quantization table for the IFAST FDCT using AA&N scale factors.
///
/// Computes `DESCALE(quant[i] * aanscales[i], CONST_BITS - 3)` where
/// `CONST_BITS = 14`, matching C libjpeg-turbo's `jcdctmgr.c` ifast divisor
/// computation exactly. Paired with `fdct_ifast_raw` (no AA&N rescaling).
pub(super) fn scale_quant_for_ifast(quant_table: &[u16; 64]) -> QuantDivisors {
    use crate::encode::fdct::AANSCALES;
    let mut divisors = [0u16; 64];
    let mut reciprocals = [0u16; 64];
    let mut corrections = [0u16; 64];
    let mut shifts = [0i16; 64];
    let mut scales = [0u16; 64];
    for i in 0..64 {
        // DESCALE(quant * aanscale, 14 - 3) = (quant * aanscale + 1024) >> 11
        let product: i64 = quant_table[i] as i64 * AANSCALES[i] as i64;
        let d: u16 = ((product + (1i64 << 10)) >> 11) as u16;
        divisors[i] = d;
        let (recip, corr, scale, shift) = compute_reciprocal(d);
        reciprocals[i] = recip;
        corrections[i] = corr;
        scales[i] = scale;
        shifts[i] = shift;
    }
    let float_divisors = compute_float_divisors(quant_table);
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    let mut divisors_zigzag = [0u16; 64];
    let mut reciprocals_zigzag = [0u16; 64];
    let mut corrections_zigzag = [0u16; 64];
    let mut shifts_zigzag = [0i16; 64];
    let mut scales_zigzag = [0u16; 64];
    let mut float_divisors_zigzag = [0.0f32; 64];
    for zz in 0..64 {
        divisors_zigzag[zz] = divisors[zigzag[zz]];
        reciprocals_zigzag[zz] = reciprocals[zigzag[zz]];
        corrections_zigzag[zz] = corrections[zigzag[zz]];
        shifts_zigzag[zz] = shifts[zigzag[zz]];
        scales_zigzag[zz] = scales[zigzag[zz]];
        float_divisors_zigzag[zz] = float_divisors[zigzag[zz]];
    }
    QuantDivisors {
        divisors,
        reciprocals,
        corrections,
        shifts,
        scales,
        divisors_zigzag,
        reciprocals_zigzag,
        corrections_zigzag,
        shifts_zigzag,
        scales_zigzag,
        float_divisors,
        float_divisors_zigzag,
    }
}

/// C `jcdctmgr.c` lines 346–365: float divisor =
/// `1 / (quant[i] * aanscalefactor[row] * aanscalefactor[col] * 8)`.
pub(super) fn compute_float_divisors(quant_table: &[u16; 64]) -> [f32; 64] {
    const AANSCALEFACTOR: [f64; 8] = [
        1.0,
        1.387039845,
        1.306562965,
        1.175875602,
        1.0,
        0.785694958,
        0.541196100,
        0.275899379,
    ];
    let mut out = [0.0f32; 64];
    #[allow(clippy::needless_range_loop)]
    for row in 0..8 {
        for col in 0..8 {
            let i: usize = row * 8 + col;
            let denom: f64 =
                quant_table[i] as f64 * AANSCALEFACTOR[row] * AANSCALEFACTOR[col] * 8.0;
            out[i] = (1.0 / denom) as f32;
        }
    }
    out
}

/// Scale quantization table values by 8 to create divisor table for the islow FDCT.
///
/// Uses C libjpeg-turbo's adaptive-precision reciprocal algorithm for exact
/// SIMD quantization (no rounding errors vs true integer division).
pub(super) fn scale_quant_for_fdct(quant_table: &[u16; 64]) -> QuantDivisors {
    let mut divisors = [0u16; 64];
    let mut reciprocals = [0u16; 64];
    let mut corrections = [0u16; 64];
    let mut shifts = [0i16; 64];
    let mut scales = [0u16; 64];
    for i in 0..64 {
        let d: u16 = (quant_table[i] as u32 * 8) as u16;
        divisors[i] = d;
        let (recip, corr, scale, shift) = compute_reciprocal(d);
        reciprocals[i] = recip;
        corrections[i] = corr;
        scales[i] = scale;
        shifts[i] = shift;
    }
    let float_divisors = compute_float_divisors(quant_table);
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    let mut divisors_zigzag = [0u16; 64];
    let mut reciprocals_zigzag = [0u16; 64];
    let mut corrections_zigzag = [0u16; 64];
    let mut shifts_zigzag = [0i16; 64];
    let mut scales_zigzag = [0u16; 64];
    let mut float_divisors_zigzag = [0.0f32; 64];
    for zz in 0..64 {
        divisors_zigzag[zz] = divisors[zigzag[zz]];
        reciprocals_zigzag[zz] = reciprocals[zigzag[zz]];
        corrections_zigzag[zz] = corrections[zigzag[zz]];
        shifts_zigzag[zz] = shifts[zigzag[zz]];
        scales_zigzag[zz] = scales[zigzag[zz]];
        float_divisors_zigzag[zz] = float_divisors[zigzag[zz]];
    }
    QuantDivisors {
        divisors,
        reciprocals,
        corrections,
        shifts,
        scales,
        divisors_zigzag,
        reciprocals_zigzag,
        corrections_zigzag,
        shifts_zigzag,
        scales_zigzag,
        float_divisors,
        float_divisors_zigzag,
    }
}
