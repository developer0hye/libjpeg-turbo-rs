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

/// Fancy horizontal 2x upsampling using triangle filter with alternating bias.
///
/// Matches libjpeg-turbo `h2v1_fancy_upsample` exactly:
///   output\[2i\]   = (3*input\[i\] + input\[i-1\] + 1) >> 2  (even: bias +1)
///   output\[2i+1\] = (3*input\[i\] + input\[i+1\] + 2) >> 2  (odd:  bias +2)
///
/// The alternating bias avoids systematic rounding towards larger values
/// (ordered dither pattern as described in the C source).
pub fn fancy_h2v1(input: &[u8], in_width: usize, output: &mut [u8], _out_width: usize) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    // First column: left edge
    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    // Interior columns: alternating bias +1 (even) / +2 (odd)
    for x in 1..in_width - 1 {
        let left = input[x - 1] as u16;
        let cur = input[x] as u16;
        let right = input[x + 1] as u16;
        output[x * 2] = ((3 * cur + left + 1) >> 2) as u8;
        output[x * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
    }

    // Last column: right edge
    let last = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 1) >> 2) as u8;
    output[last * 2 + 1] = input[last];
}

/// Fancy 2x2 upsampling using fused 2D triangle filter.
///
/// Matches libjpeg-turbo `h2v2_fancy_upsample` exactly:
///   colsum = cur_row\[i\] * 3 + neighbor_row\[i\]   (vertical blend, 16-bit)
///   output_even = (thiscolsum * 3 + lastcolsum + 8) >> 4
///   output_odd  = (thiscolsum * 3 + nextcolsum + 7) >> 4
///
/// This fused approach avoids double-rounding by keeping the vertical
/// intermediate as a 16-bit column sum, then doing horizontal + final
/// rounding in a single `>> 4` operation.
pub fn fancy_h2v2(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
    _out_height: usize,
) {
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

        // Two output rows per input row: top (blend with above), bottom (blend with below)
        for (v, neighbor) in [(0, above), (1, below)] {
            let out_y = y * 2 + v;
            let out_row = &mut output[out_y * out_width..];

            fancy_h2v2_row(cur_row, neighbor, out_row, in_width);
        }
    }
}

/// Fused H2V2 fancy upsample for one output row.
/// `cur` is the nearest input row, `neighbor` is the next-nearest.
///
/// Matches C libjpeg-turbo `h2v2_fancy_upsample` inner loop:
///   colsum = cur * 3 + neighbor  (no rounding yet)
///   even pixel: (thiscolsum * 3 + lastcolsum + 8) >> 4
///   odd pixel:  (thiscolsum * 3 + nextcolsum + 7) >> 4
pub fn fancy_h2v2_row(cur: &[u8], neighbor: &[u8], output: &mut [u8], in_width: usize) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        let colsum = cur[0] as i32 * 3 + neighbor[0] as i32;
        output[0] = ((colsum * 4 + 8) >> 4) as u8;
        output[1] = ((colsum * 4 + 7) >> 4) as u8;
        return;
    }

    // Column sums: colsum[i] = cur[i] * 3 + neighbor[i]
    let colsum = |i: usize| -> i32 { cur[i] as i32 * 3 + neighbor[i] as i32 };

    let mut this_cs = colsum(0);
    let mut next_cs = colsum(1);

    // First column
    output[0] = ((this_cs * 4 + 8) >> 4) as u8;
    output[1] = ((this_cs * 3 + next_cs + 7) >> 4) as u8;

    let mut last_cs = this_cs;
    this_cs = next_cs;

    // Interior columns
    for x in 1..in_width - 1 {
        next_cs = colsum(x + 1);
        output[x * 2] = ((this_cs * 3 + last_cs + 8) >> 4) as u8;
        output[x * 2 + 1] = ((this_cs * 3 + next_cs + 7) >> 4) as u8;
        last_cs = this_cs;
        this_cs = next_cs;
    }

    // Last column
    let last = in_width - 1;
    output[last * 2] = ((this_cs * 3 + last_cs + 8) >> 4) as u8;
    output[last * 2 + 1] = ((this_cs * 4 + 7) >> 4) as u8;
}
