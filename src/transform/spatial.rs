/// Spatial transforms on 8x8 DCT coefficient blocks.
///
/// Each transform manipulates DCT coefficients directly in the frequency domain,
/// avoiding decode/re-encode quality loss.
/// No transform — copy block unchanged.
pub fn do_nothing(src: &[i16; 64], dst: &mut [i16; 64]) {
    dst.copy_from_slice(src);
}

/// Horizontal flip: negate odd-column coefficients.
///
/// Flipping horizontally in spatial domain corresponds to negating
/// coefficients at odd column positions (1, 3, 5, 7) in the DCT domain.
pub fn do_flip_h(src: &[i16; 64], dst: &mut [i16; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            let idx = row * 8 + col;
            // wrapping_neg: see do_rot_180 for the i16::MIN rationale.
            dst[idx] = if col % 2 == 1 {
                src[idx].wrapping_neg()
            } else {
                src[idx]
            };
        }
    }
}

/// Vertical flip: negate odd-row coefficients.
pub fn do_flip_v(src: &[i16; 64], dst: &mut [i16; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            let idx = row * 8 + col;
            // wrapping_neg: see do_rot_180 for the i16::MIN rationale.
            dst[idx] = if row % 2 == 1 {
                src[idx].wrapping_neg()
            } else {
                src[idx]
            };
        }
    }
}

/// Transpose: swap row and column indices within the block.
pub fn do_transpose(src: &[i16; 64], dst: &mut [i16; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            dst[col * 8 + row] = src[row * 8 + col];
        }
    }
}

/// Transverse transpose: rotate 180° + transpose.
/// Equivalent to negate odd-(row+col) coefficients, then transpose.
pub fn do_transverse(src: &[i16; 64], dst: &mut [i16; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            // wrapping_neg: see do_rot_180 for the i16::MIN rationale.
            let v = src[row * 8 + col];
            dst[col * 8 + row] = if (row + col) % 2 == 1 {
                v.wrapping_neg()
            } else {
                v
            };
        }
    }
}

/// Rotate 90° clockwise: transpose + horizontal flip.
pub fn do_rot_90(src: &[i16; 64], dst: &mut [i16; 64]) {
    let mut tmp = [0i16; 64];
    do_transpose(src, &mut tmp);
    do_flip_h(&tmp, dst);
}

/// Rotate 180°: horizontal flip + vertical flip.
pub fn do_rot_180(src: &[i16; 64], dst: &mut [i16; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            let idx = row * 8 + col;
            // Use wrapping_neg: adversarial inputs can carry quantized
            // coefficients of i16::MIN, and `i16::MIN * -1` panics under
            // overflow checks (cargo-fuzz default). The wrap is safe — a
            // single quantized coefficient at the extreme is malformed
            // and will be reclassified by Huffman categorisation anyway.
            // Found via fuzz_transform_options round-3 (CI run 25215431132).
            dst[idx] = if (row + col) % 2 == 1 {
                src[idx].wrapping_neg()
            } else {
                src[idx]
            };
        }
    }
}

/// Rotate 270° clockwise: transpose + vertical flip.
pub fn do_rot_270(src: &[i16; 64], dst: &mut [i16; 64]) {
    let mut tmp = [0i16; 64];
    do_transpose(src, &mut tmp);
    do_flip_v(&tmp, dst);
}

/// A spatial block op composed with the zigzag↔natural reorder, so it can
/// be applied directly to blocks stored in zigzag order (the storage
/// format of `JpegCoefficients`). Every `do_*` op above is a pure
/// per-coefficient permutation with optional negation, so the composition
/// `natural_to_zigzag ∘ do_op ∘ zigzag_to_natural` is again a
/// permutation+negation — one gather per coefficient instead of three
/// full-block passes (issue #308: the transform path used to convert the
/// whole coefficient corpus zigzag→natural, permute, then convert back).
///
/// `src[d]` is the source zigzag index feeding destination zigzag index
/// `d`; `neg[d]` applies the DCT mirror sign flip.
///
/// The maps are `const` so each `MAP_*.apply(..)` call site unrolls with
/// immediate indices and compile-time signs, matching the codegen of the
/// natural-domain `do_*` kernels.
pub struct ZigzagMap {
    src: [u8; 64],
    neg: [bool; 64],
}

impl ZigzagMap {
    #[inline]
    pub fn apply(&self, src: &[i16; 64], dst: &mut [i16; 64]) {
        for d in 0..64 {
            let v: i16 = src[self.src[d] as usize];
            // wrapping_neg: see do_rot_180 for the i16::MIN rationale.
            dst[d] = if self.neg[d] { v.wrapping_neg() } else { v };
        }
    }
}

/// Build the zigzag-domain map for a natural-domain op of the form
/// `dst(r,c) = ±src(c,r 𝗂𝖿 transpose 𝖾𝗅𝗌𝖾 r,c)` with the sign given by
/// `(-1)^(r·neg_odd_row) · (-1)^(c·neg_odd_col)` in DESTINATION (r,c)
/// coordinates. All eight jpegtran ops factor into this form:
///
/// | op         | transpose | neg_odd_row | neg_odd_col |
/// |------------|-----------|-------------|-------------|
/// | None       | no        | no          | no          |
/// | HFlip      | no        | no          | yes         |
/// | VFlip      | no        | yes         | no          |
/// | Transpose  | yes       | no          | no          |
/// | Transverse | yes       | yes         | yes         |
/// | Rot90      | yes       | no          | yes         |
/// | Rot180     | no        | yes         | yes         |
/// | Rot270     | yes       | yes         | no          |
///
/// `tests` pins each const map against the basis-vector composition of
/// the `do_*` kernels, so the two representations cannot drift.
const fn build_zigzag_map(transpose: bool, neg_odd_row: bool, neg_odd_col: bool) -> ZigzagMap {
    use crate::common::quant_table::NATURAL_ORDER;

    // Invert NATURAL_ORDER (natural idx → zigzag pos) to get
    // zigzag pos → natural idx.
    let mut nat_of_zz: [usize; 64] = [0; 64];
    let mut n: usize = 0;
    while n < 64 {
        nat_of_zz[NATURAL_ORDER[n]] = n;
        n += 1;
    }

    let mut src: [u8; 64] = [0; 64];
    let mut neg: [bool; 64] = [false; 64];
    let mut d: usize = 0;
    while d < 64 {
        let nd: usize = nat_of_zz[d];
        let r: usize = nd / 8;
        let c: usize = nd % 8;
        let (sr, sc): (usize, usize) = if transpose { (c, r) } else { (r, c) };
        src[d] = NATURAL_ORDER[sr * 8 + sc] as u8;
        neg[d] = (neg_odd_row && r % 2 == 1) != (neg_odd_col && c % 2 == 1);
        d += 1;
    }
    ZigzagMap { src, neg }
}

pub const MAP_NONE: ZigzagMap = build_zigzag_map(false, false, false);
pub const MAP_HFLIP: ZigzagMap = build_zigzag_map(false, false, true);
pub const MAP_VFLIP: ZigzagMap = build_zigzag_map(false, true, false);
pub const MAP_TRANSPOSE: ZigzagMap = build_zigzag_map(true, false, false);
pub const MAP_TRANSVERSE: ZigzagMap = build_zigzag_map(true, true, true);
pub const MAP_ROT90: ZigzagMap = build_zigzag_map(true, false, true);
pub const MAP_ROT180: ZigzagMap = build_zigzag_map(false, true, true);
pub const MAP_ROT270: ZigzagMap = build_zigzag_map(true, true, false);

/// Zigzag-domain map for `op`. Prefer the `MAP_*` consts directly at call
/// sites where `op` is statically known — that lets the permutation
/// indices constant-fold.
pub fn zigzag_map(op: crate::transform::TransformOp) -> &'static ZigzagMap {
    use crate::transform::TransformOp;
    match op {
        TransformOp::None => &MAP_NONE,
        TransformOp::HFlip => &MAP_HFLIP,
        TransformOp::VFlip => &MAP_VFLIP,
        TransformOp::Transpose => &MAP_TRANSPOSE,
        TransformOp::Transverse => &MAP_TRANSVERSE,
        TransformOp::Rot90 => &MAP_ROT90,
        TransformOp::Rot180 => &MAP_ROT180,
        TransformOp::Rot270 => &MAP_ROT270,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_block() -> [i16; 64] {
        let mut block = [0i16; 64];
        for i in 0..64 {
            block[i] = (i + 1) as i16;
        }
        block
    }

    #[test]
    fn nothing_is_identity() {
        let src = make_test_block();
        let mut dst = [0i16; 64];
        do_nothing(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn double_hflip_is_identity() {
        let src = make_test_block();
        let mut tmp = [0i16; 64];
        let mut dst = [0i16; 64];
        do_flip_h(&src, &mut tmp);
        do_flip_h(&tmp, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn double_vflip_is_identity() {
        let src = make_test_block();
        let mut tmp = [0i16; 64];
        let mut dst = [0i16; 64];
        do_flip_v(&src, &mut tmp);
        do_flip_v(&tmp, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn double_transpose_is_identity() {
        let src = make_test_block();
        let mut tmp = [0i16; 64];
        let mut dst = [0i16; 64];
        do_transpose(&src, &mut tmp);
        do_transpose(&tmp, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn rot180_is_double_identity() {
        let src = make_test_block();
        let mut tmp = [0i16; 64];
        let mut dst = [0i16; 64];
        do_rot_180(&src, &mut tmp);
        do_rot_180(&tmp, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn four_rot90_is_identity() {
        let src = make_test_block();
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        do_rot_90(&src, &mut a);
        do_rot_90(&a, &mut b);
        do_rot_90(&b, &mut a);
        do_rot_90(&a, &mut b);
        assert_eq!(src, b);
    }

    /// Issue #308: the transform path applies ops directly in zigzag
    /// order through the const `MAP_*` tables. Pin every table against
    /// the basis-vector composition
    /// `natural_to_zigzag ∘ do_op ∘ zigzag_to_natural` of the
    /// natural-domain kernels above, so the two representations cannot
    /// silently drift.
    #[test]
    fn zigzag_maps_match_natural_kernels() {
        use crate::common::quant_table::NATURAL_ORDER;

        let cases: [(&ZigzagMap, fn(&[i16; 64], &mut [i16; 64]), &str); 8] = [
            (&MAP_NONE, do_nothing as _, "None"),
            (&MAP_HFLIP, do_flip_h as _, "HFlip"),
            (&MAP_VFLIP, do_flip_v as _, "VFlip"),
            (&MAP_TRANSPOSE, do_transpose as _, "Transpose"),
            (&MAP_TRANSVERSE, do_transverse as _, "Transverse"),
            (&MAP_ROT90, do_rot_90 as _, "Rot90"),
            (&MAP_ROT180, do_rot_180 as _, "Rot180"),
            (&MAP_ROT270, do_rot_270 as _, "Rot270"),
        ];

        // Distinct nonzero magnitudes per position so any permutation or
        // sign mistake is detectable.
        let mut zigzag_in = [0i16; 64];
        for (i, v) in zigzag_in.iter_mut().enumerate() {
            *v = (i as i16 + 1) * if i % 3 == 0 { -1 } else { 1 };
        }

        for (map, kernel, label) in cases {
            // Reference: convert to natural, run kernel, convert back.
            let mut natural_in = [0i16; 64];
            for n in 0..64 {
                natural_in[n] = zigzag_in[NATURAL_ORDER[n]];
            }
            let mut natural_out = [0i16; 64];
            kernel(&natural_in, &mut natural_out);
            let mut expected = [0i16; 64];
            for n in 0..64 {
                expected[NATURAL_ORDER[n]] = natural_out[n];
            }

            let mut actual = [0i16; 64];
            map.apply(&zigzag_in, &mut actual);
            assert_eq!(actual, expected, "zigzag map for {} drifted", label);
        }
    }

    #[test]
    fn hflip_negates_odd_columns() {
        let src = make_test_block();
        let mut dst = [0i16; 64];
        do_flip_h(&src, &mut dst);
        // DC (0,0) should be unchanged
        assert_eq!(dst[0], src[0]);
        // (0,1) should be negated
        assert_eq!(dst[1], -src[1]);
        // (0,2) should be unchanged
        assert_eq!(dst[2], src[2]);
    }
}
