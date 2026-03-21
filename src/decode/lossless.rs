/// Lossless JPEG decoding (SOF3).
///
/// Implements prediction and undifferencing for lossless JPEG
/// as specified in ITU-T T.81 sections 14-15.

/// Apply one of 7 lossless predictors.
///
/// ra = left, rb = above, rc = upper-left (diagonal).
pub fn predict(psv: u8, ra: i32, rb: i32, rc: i32) -> i32 {
    match psv {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => 0,
    }
}

/// Undifference a row of lossless-coded samples.
///
/// Converts difference values (from Huffman decoding) back to pixel values
/// using the selected predictor.
pub fn undifference_row(
    diffs: &[i16],
    prev_row: Option<&[u16]>,
    output: &mut [u16],
    psv: u8,
    precision: u8,
    point_transform: u8,
    is_first_row: bool,
) {
    let mask = ((1u32 << precision) - 1) as i32;
    let initial = 1i32 << (precision as i32 - point_transform as i32 - 1);

    for x in 0..diffs.len() {
        let prediction = if is_first_row && x == 0 {
            initial
        } else if is_first_row {
            output[x - 1] as i32
        } else if x == 0 {
            prev_row.unwrap()[0] as i32
        } else {
            let ra = output[x - 1] as i32;
            let rb = prev_row.unwrap()[x] as i32;
            let rc = prev_row.unwrap()[x - 1] as i32;
            predict(psv, ra, rb, rc)
        };

        output[x] = ((diffs[x] as i32 + prediction) & mask) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_7_predictors_produce_valid_output() {
        for psv in 1..=7 {
            let ra = 100i32;
            let rb = 110;
            let rc = 105;
            let pred = predict(psv, ra, rb, rc);
            assert!(pred >= 0 && pred < 512, "predictor {psv} gave {pred}");
        }
    }

    #[test]
    fn predictor_1_is_left() {
        assert_eq!(predict(1, 100, 200, 150), 100);
    }

    #[test]
    fn predictor_2_is_above() {
        assert_eq!(predict(2, 100, 200, 150), 200);
    }

    #[test]
    fn predictor_4_is_ra_plus_rb_minus_rc() {
        assert_eq!(predict(4, 100, 200, 150), 150);
    }

    #[test]
    fn predictor_7_is_average() {
        assert_eq!(predict(7, 100, 200, 150), 150);
    }

    #[test]
    fn undifference_first_row() {
        let diffs = [10i16, 5, -3, 2];
        let mut output = [0u16; 4];
        undifference_row(&diffs, None, &mut output, 1, 8, 0, true);
        // initial = 1 << (8 - 0 - 1) = 128
        assert_eq!(output[0], (10 + 128) as u16); // 138
        assert_eq!(output[1], (5 + 138) as u16); // 143
        assert_eq!(output[2], ((-3i32 + 143) & 255) as u16); // 140
        assert_eq!(output[3], ((2 + 140) & 255) as u16); // 142
    }

    #[test]
    fn undifference_second_row_predictor_2() {
        let prev = [100u16, 110, 120, 130];
        let diffs = [5i16, -10, 3, 0];
        let mut output = [0u16; 4];
        undifference_row(&diffs, Some(&prev), &mut output, 2, 8, 0, false);
        // First pixel: pred = prev[0] = 100
        assert_eq!(output[0], 105);
        // Remaining: pred = prev[x] (predictor 2 = above)
        assert_eq!(output[1], 100); // -10 + 110
        assert_eq!(output[2], 123); // 3 + 120
        assert_eq!(output[3], 130); // 0 + 130
    }
}
