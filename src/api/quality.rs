/// Quality scaling utilities matching libjpeg-turbo's `jpeg_quality_scaling()`.
///
/// These functions convert between user-facing quality ratings (1-100)
/// and the internal linear scale factors used to scale quantization tables.
/// Convert a quality rating (0-100) to a linear scale factor.
///
/// Matches libjpeg-turbo's `jpeg_quality_scaling()`. The returned scale factor
/// is a percentage used to scale the standard quantization tables:
/// - Quality 50 maps to scale factor 100 (tables used as-is)
/// - Quality below 50 increases the scale factor (coarser quantization)
/// - Quality above 50 decreases the scale factor (finer quantization)
///
/// # Arguments
/// * `quality` - Quality rating from 0 to 100. Values <= 0 are treated as 1.
///
/// # Returns
/// Linear scale factor as a percentage (e.g., 100 means use tables unscaled).
pub fn quality_scaling(quality: u8) -> u32 {
    // Match libjpeg behavior: clamp to 1-100 range
    let q: u32 = if quality == 0 { 1 } else { quality as u32 };

    if q < 50 {
        5000 / q
    } else {
        200 - q * 2
    }
}

/// Scale a quantization table using a linear scale factor.
///
/// This matches libjpeg-turbo's `jpeg_add_quant_table()` when called with
/// the `force_baseline` parameter.
///
/// # Arguments
/// * `table` - Base quantization table (64 values in natural order)
/// * `scale_factor` - Linear scale factor (from `quality_scaling()`)
/// * `force_baseline` - When true, clamp values to 1-255 for baseline compatibility.
///   When false, clamp to 1-32767 (12-bit max).
pub fn scale_quant_table_linear(
    table: &[u8; 64],
    scale_factor: u32,
    force_baseline: bool,
) -> [u16; 64] {
    let max_val: i32 = if force_baseline { 255 } else { 32767 };
    let mut output: [u16; 64] = [0u16; 64];
    for i in 0..64 {
        let temp: i32 = (table[i] as i32 * scale_factor as i32 + 50) / 100;
        output[i] = temp.clamp(1, max_val) as u16;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_1_gives_5000() {
        assert_eq!(quality_scaling(1), 5000);
    }

    #[test]
    fn quality_50_gives_100() {
        assert_eq!(quality_scaling(50), 100);
    }

    #[test]
    fn quality_75_gives_50() {
        assert_eq!(quality_scaling(75), 50);
    }

    #[test]
    fn quality_100_gives_0() {
        assert_eq!(quality_scaling(100), 0);
    }

    #[test]
    fn quality_0_treated_as_1() {
        assert_eq!(quality_scaling(0), 5000);
    }

    #[test]
    fn force_baseline_clamps_to_255() {
        let table: [u8; 64] = [99; 64];
        let result: [u16; 64] = scale_quant_table_linear(&table, 5000, true);
        // 99 * 5000 / 100 = 4950, clamped to 255
        assert_eq!(result[0], 255);
    }

    #[test]
    fn no_force_baseline_allows_higher() {
        let table: [u8; 64] = [99; 64];
        let result: [u16; 64] = scale_quant_table_linear(&table, 5000, false);
        // 99 * 5000 / 100 = 4950, clamped to 32767 (so 4950)
        assert_eq!(result[0], 4950);
    }
}
