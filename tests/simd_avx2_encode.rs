//! Integration tests for x86_64 AVX2 encode-side SIMD operations.
//!
//! Verifies that AVX2-accelerated RGB->YCbCr color conversion produces
//! identical results to the scalar equivalent.

#[cfg(target_arch = "x86_64")]
mod tests {
    use libjpeg_turbo_rs::encode::color::rgb_to_ycbcr_row;
    use libjpeg_turbo_rs::simd::x86_64::avx2_color_encode::avx2_rgb_to_ycbcr_row;

    fn skip_if_no_avx2() -> bool {
        !is_x86_feature_detected!("avx2")
    }

    #[test]
    fn avx2_rgb_to_ycbcr_matches_scalar_white() {
        if skip_if_no_avx2() {
            return;
        }
        let rgb = [255u8; 48]; // 16 white pixels
        let width: usize = 16;
        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];
        let mut y_avx2 = vec![0u8; width];
        let mut cb_avx2 = vec![0u8; width];
        let mut cr_avx2 = vec![0u8; width];

        rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
        avx2_rgb_to_ycbcr_row(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, width);

        for i in 0..width {
            assert_eq!(
                y_scalar[i], y_avx2[i],
                "Y mismatch at pixel {i}: scalar={} avx2={}",
                y_scalar[i], y_avx2[i]
            );
            assert_eq!(
                cb_scalar[i], cb_avx2[i],
                "Cb mismatch at pixel {i}: scalar={} avx2={}",
                cb_scalar[i], cb_avx2[i]
            );
            assert_eq!(
                cr_scalar[i], cr_avx2[i],
                "Cr mismatch at pixel {i}: scalar={} avx2={}",
                cr_scalar[i], cr_avx2[i]
            );
        }
    }

    #[test]
    fn avx2_rgb_to_ycbcr_matches_scalar_black() {
        if skip_if_no_avx2() {
            return;
        }
        let rgb = [0u8; 48]; // 16 black pixels
        let width: usize = 16;
        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];
        let mut y_avx2 = vec![0u8; width];
        let mut cb_avx2 = vec![0u8; width];
        let mut cr_avx2 = vec![0u8; width];

        rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
        avx2_rgb_to_ycbcr_row(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, width);

        for i in 0..width {
            assert_eq!(y_scalar[i], y_avx2[i], "Y mismatch at {i}");
            assert_eq!(cb_scalar[i], cb_avx2[i], "Cb mismatch at {i}");
            assert_eq!(cr_scalar[i], cr_avx2[i], "Cr mismatch at {i}");
        }
    }

    #[test]
    fn avx2_rgb_to_ycbcr_matches_scalar_primary_colors() {
        if skip_if_no_avx2() {
            return;
        }
        // Red, Green, Blue repeated to fill 16 pixels
        let mut rgb = Vec::with_capacity(48);
        for _ in 0..6 {
            rgb.extend_from_slice(&[255, 0, 0]); // Red
        }
        for _ in 0..5 {
            rgb.extend_from_slice(&[0, 255, 0]); // Green
        }
        for _ in 0..5 {
            rgb.extend_from_slice(&[0, 0, 255]); // Blue
        }
        let width: usize = 16;
        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];
        let mut y_avx2 = vec![0u8; width];
        let mut cb_avx2 = vec![0u8; width];
        let mut cr_avx2 = vec![0u8; width];

        rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
        avx2_rgb_to_ycbcr_row(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, width);

        for i in 0..width {
            let y_diff = (y_scalar[i] as i16 - y_avx2[i] as i16).unsigned_abs();
            let cb_diff = (cb_scalar[i] as i16 - cb_avx2[i] as i16).unsigned_abs();
            let cr_diff = (cr_scalar[i] as i16 - cr_avx2[i] as i16).unsigned_abs();
            assert!(
                y_diff <= 1,
                "Y mismatch at pixel {i}: scalar={} avx2={} diff={y_diff}",
                y_scalar[i],
                y_avx2[i]
            );
            assert!(
                cb_diff <= 1,
                "Cb mismatch at pixel {i}: scalar={} avx2={} diff={cb_diff}",
                cb_scalar[i],
                cb_avx2[i]
            );
            assert!(
                cr_diff <= 1,
                "Cr mismatch at pixel {i}: scalar={} avx2={} diff={cr_diff}",
                cr_scalar[i],
                cr_avx2[i]
            );
        }
    }

    #[test]
    fn avx2_rgb_to_ycbcr_various_widths() {
        if skip_if_no_avx2() {
            return;
        }
        // Widths that exercise AVX2 main loop (>=16), scalar tail, and small sizes
        for &width in &[1, 7, 8, 15, 16, 17, 31, 32, 33, 48, 100] {
            let mut rgb = vec![0u8; width * 3];
            for i in 0..width {
                rgb[i * 3] = (i * 37 % 256) as u8;
                rgb[i * 3 + 1] = (i * 73 % 256) as u8;
                rgb[i * 3 + 2] = (i * 113 % 256) as u8;
            }

            let mut y_scalar = vec![0u8; width];
            let mut cb_scalar = vec![0u8; width];
            let mut cr_scalar = vec![0u8; width];
            let mut y_avx2 = vec![0u8; width];
            let mut cb_avx2 = vec![0u8; width];
            let mut cr_avx2 = vec![0u8; width];

            rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
            avx2_rgb_to_ycbcr_row(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, width);

            for i in 0..width {
                let y_diff = (y_scalar[i] as i16 - y_avx2[i] as i16).unsigned_abs();
                let cb_diff = (cb_scalar[i] as i16 - cb_avx2[i] as i16).unsigned_abs();
                let cr_diff = (cr_scalar[i] as i16 - cr_avx2[i] as i16).unsigned_abs();
                assert!(
                    y_diff <= 1,
                    "width={width} Y mismatch at pixel {i}: scalar={} avx2={} diff={y_diff}",
                    y_scalar[i],
                    y_avx2[i]
                );
                assert!(
                    cb_diff <= 1,
                    "width={width} Cb mismatch at pixel {i}: scalar={} avx2={} diff={cb_diff}",
                    cb_scalar[i],
                    cb_avx2[i]
                );
                assert!(
                    cr_diff <= 1,
                    "width={width} Cr mismatch at pixel {i}: scalar={} avx2={} diff={cr_diff}",
                    cr_scalar[i],
                    cr_avx2[i]
                );
            }
        }
    }

    #[test]
    fn avx2_rgb_to_ycbcr_extreme_values() {
        if skip_if_no_avx2() {
            return;
        }
        // All channels at max/min boundaries
        let width: usize = 16;
        let mut rgb = vec![0u8; width * 3];

        // First 8 pixels: all 255
        for i in 0..8 {
            rgb[i * 3] = 255;
            rgb[i * 3 + 1] = 255;
            rgb[i * 3 + 2] = 255;
        }
        // Next 8 pixels: all 0
        // (already zero)

        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];
        let mut y_avx2 = vec![0u8; width];
        let mut cb_avx2 = vec![0u8; width];
        let mut cr_avx2 = vec![0u8; width];

        rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
        avx2_rgb_to_ycbcr_row(&rgb, &mut y_avx2, &mut cb_avx2, &mut cr_avx2, width);

        for i in 0..width {
            let y_diff = (y_scalar[i] as i16 - y_avx2[i] as i16).unsigned_abs();
            let cb_diff = (cb_scalar[i] as i16 - cb_avx2[i] as i16).unsigned_abs();
            let cr_diff = (cr_scalar[i] as i16 - cr_avx2[i] as i16).unsigned_abs();
            assert!(
                y_diff <= 1,
                "extreme Y mismatch at {i}: scalar={} avx2={}",
                y_scalar[i],
                y_avx2[i]
            );
            assert!(
                cb_diff <= 1,
                "extreme Cb mismatch at {i}: scalar={} avx2={}",
                cb_scalar[i],
                cb_avx2[i]
            );
            assert!(
                cr_diff <= 1,
                "extreme Cr mismatch at {i}: scalar={} avx2={}",
                cr_scalar[i],
                cr_avx2[i]
            );
        }
    }

    #[test]
    fn avx2_rgb_to_ycbcr_width_zero() {
        if skip_if_no_avx2() {
            return;
        }
        let rgb: [u8; 0] = [];
        let mut y: [u8; 0] = [];
        let mut cb: [u8; 0] = [];
        let mut cr: [u8; 0] = [];
        // Should not panic
        avx2_rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, 0);
    }
}
