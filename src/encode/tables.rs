//! Encoder-side quantization-table policy.
//!
//! The Annex K specification data these functions start from — the two base
//! quantization tables, the four standard Huffman tables, and the zigzag order
//! — now lives in [`crate::common::tables`]. That data is direction-neutral:
//! the decoder builds `std_huffman_tables()` from the same Huffman constants
//! and the SIMD quantization kernels use the same zigzag order, so keeping it
//! here forced `common` and `simd` to depend *upward* on `encode` (issue #442).

// Re-exported rather than simply moved: `encode::tables` is reachable from
// outside the crate, so dropping these names would be a breaking change.
pub use crate::common::tables::{
    AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES,
    DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
    STD_CHROMINANCE_QUANT_TABLE, STD_LUMINANCE_QUANT_TABLE, ZIGZAG_ORDER,
};

/// Scale a quantization table by quality factor.
///
/// Quality ranges from 1 (worst) to 100 (best).
/// Quality 50 uses the table as-is. Below 50, values increase (coarser quantization).
/// Above 50, values decrease (finer quantization). Matching libjpeg-turbo's
/// `jpeg_quality_scaling` + `jpeg_add_quant_table`.
///
/// When `force_baseline` is true, values are clamped to [1, 255] for baseline JPEG
/// compatibility. When false, values are clamped to [1, 32767] to support extended
/// (12-bit) JPEG, matching C libjpeg-turbo's `jpeg_add_quant_table`.
pub fn quality_scale_quant_table(table: &[u8; 64], quality: u8) -> [u16; 64] {
    quality_scale_quant_table_ext(table, quality, true)
}

/// Scale a quantization table by quality factor with explicit baseline control.
///
/// See [`quality_scale_quant_table`] for details. When `force_baseline` is false,
/// quantization values up to 32767 are permitted (required for 12-bit precision).
pub fn quality_scale_quant_table_ext(
    table: &[u8; 64],
    quality: u8,
    force_baseline: bool,
) -> [u16; 64] {
    let quality = quality.clamp(1, 100) as i32;
    let scale_factor: i32 = if quality < 50 {
        5000 / quality
    } else {
        200 - 2 * quality
    };

    let max_val: i32 = if force_baseline { 255 } else { 32767 };

    let mut output = [0u16; 64];
    for i in 0..64 {
        let temp = (table[i] as i32 * scale_factor + 50) / 100;
        output[i] = temp.clamp(1, max_val) as u16;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_50_returns_original_table() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 50);
        for i in 0..64 {
            assert_eq!(
                scaled[i], STD_LUMINANCE_QUANT_TABLE[i] as u16,
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn quality_100_returns_all_ones() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 100);
        // At quality 100, scale_factor = 0, so (val * 0 + 50) / 100 = 0,
        // clamped to 1.
        for i in 0..64 {
            assert_eq!(scaled[i], 1, "expected 1 at index {i}, got {}", scaled[i]);
        }
    }

    #[test]
    fn quality_1_produces_max_quantization() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 1);
        // scale_factor = 5000. Most values will be clamped to 255.
        for i in 0..64 {
            assert!(scaled[i] >= 1 && scaled[i] <= 255);
        }
        // The smallest table entry (10 at index 2) * 5000 / 100 = 500 -> clamped to 255
        assert_eq!(scaled[2], 255);
    }

    #[test]
    fn quality_75_produces_lower_values() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 75);
        // scale_factor = 200 - 150 = 50
        // First entry: (16 * 50 + 50) / 100 = 8
        assert_eq!(scaled[0], 8);
    }

    #[test]
    fn quality_25_produces_higher_values() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 25);
        // scale_factor = 5000 / 25 = 200
        // First entry: (16 * 200 + 50) / 100 = 32
        assert_eq!(scaled[0], 32);
    }

    #[test]
    fn quality_1_extended_allows_values_above_255() {
        // With force_baseline=false, values can exceed 255 (up to 32767)
        let scaled = quality_scale_quant_table_ext(&STD_LUMINANCE_QUANT_TABLE, 1, false);
        // scale_factor = 5000. Entry at index 2 (value=10): 10*5000/100 = 500
        assert_eq!(scaled[2], 500);
        // Entry at index 7 (value=61): 61*5000/100 = 3050
        assert_eq!(scaled[7], 3050);
        for i in 0..64 {
            assert!(scaled[i] >= 1 && scaled[i] <= 32767);
        }
    }

    #[test]
    fn quality_scale_baseline_clamps_at_255() {
        // With force_baseline=true (default), values are clamped to 255
        let baseline = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 1);
        let extended = quality_scale_quant_table_ext(&STD_LUMINANCE_QUANT_TABLE, 1, false);
        // Baseline should clamp, extended should not
        assert_eq!(baseline[2], 255);
        assert_eq!(extended[2], 500);
    }
}
