/// Simple nearest-neighbor horizontal 2x upsampling.
pub fn simple_h2v1(input: &[u8], in_width: usize, output: &mut [u8], _out_width: usize) {
    for x in 0..in_width {
        let val = input[x];
        output[x * 2] = val;
        output[x * 2 + 1] = val;
    }
}

/// Simple nearest-neighbor 2x2 upsampling.
pub fn simple_h2v2(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
    _out_height: usize,
) {
    for y in 0..in_height {
        for x in 0..in_width {
            let val = input[y * in_width + x];
            let out_y = y * 2;
            let out_x = x * 2;
            output[out_y * out_width + out_x] = val;
            output[out_y * out_width + out_x + 1] = val;
            output[(out_y + 1) * out_width + out_x] = val;
            output[(out_y + 1) * out_width + out_x + 1] = val;
        }
    }
}

/// Fancy horizontal 2x upsampling using triangle filter.
/// Formula: output[2i] = (3*input[i] + input[i-1] + 2) >> 2
///          output[2i+1] = (3*input[i] + input[i+1] + 2) >> 2
pub fn fancy_h2v1(input: &[u8], in_width: usize, output: &mut [u8], _out_width: usize) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    for x in 1..in_width - 1 {
        let left = input[x - 1] as u16;
        let cur = input[x] as u16;
        let right = input[x + 1] as u16;
        output[x * 2] = ((3 * cur + left + 2) >> 2) as u8;
        output[x * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
    }

    let last = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 2) >> 2) as u8;
    output[last * 2 + 1] = input[last];
}

/// Fancy 2x2 upsampling using triangle filter in both dimensions.
pub fn fancy_h2v2(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
    _out_height: usize,
) {
    let mut row_above = vec![0u8; in_width];
    let mut row_below = vec![0u8; in_width];

    for y in 0..in_height {
        let cur_row = &input[y * in_width..(y + 1) * in_width];
        let above = if y > 0 {
            &input[(y - 1) * in_width..y * in_width]
        } else {
            cur_row
        };
        let below = if y + 1 < in_height {
            &input[(y + 1) * in_width..(y + 2) * in_width]
        } else {
            cur_row
        };

        for x in 0..in_width {
            row_above[x] = ((3 * cur_row[x] as u16 + above[x] as u16 + 2) >> 2) as u8;
            row_below[x] = ((3 * cur_row[x] as u16 + below[x] as u16 + 2) >> 2) as u8;
        }

        let out_y_top = y * 2;
        let out_y_bot = y * 2 + 1;
        fancy_h2v1(
            &row_above,
            in_width,
            &mut output[out_y_top * out_width..],
            out_width,
        );
        fancy_h2v1(
            &row_below,
            in_width,
            &mut output[out_y_bot * out_width..],
            out_width,
        );
    }
}
