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
            dst[idx] = if col % 2 == 1 { -src[idx] } else { src[idx] };
        }
    }
}

/// Vertical flip: negate odd-row coefficients.
pub fn do_flip_v(src: &[i16; 64], dst: &mut [i16; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            let idx = row * 8 + col;
            dst[idx] = if row % 2 == 1 { -src[idx] } else { src[idx] };
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
            let sign = if (row + col) % 2 == 1 { -1 } else { 1 };
            dst[col * 8 + row] = src[row * 8 + col] * sign;
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
            let sign = if (row + col) % 2 == 1 { -1 } else { 1 };
            dst[idx] = src[idx] * sign;
        }
    }
}

/// Rotate 270° clockwise: transpose + vertical flip.
pub fn do_rot_270(src: &[i16; 64], dst: &mut [i16; 64]) {
    let mut tmp = [0i16; 64];
    do_transpose(src, &mut tmp);
    do_flip_v(&tmp, dst);
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
