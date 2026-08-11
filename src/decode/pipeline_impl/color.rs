use super::{Decoder, Image};
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::try_alloc::{try_clone_opt, try_clone_opt_string, try_clone_saved_markers};
use crate::common::types::{ColorSpace, DctMethod, FrameHeader, PixelFormat};
use crate::decode::{idct_extended, idct_scaled};
use alloc::{borrow::Cow, format, vec, vec::Vec};

/// Generic nearest-neighbor upsampling for arbitrary h/v factor combinations.
///
/// Handles non-standard sampling factors like 3x2, 3x1, 1x3, 4x2 that lack
/// dedicated optimized paths. Each input sample is replicated h_factor times
/// horizontally and v_factor times vertically.
pub(crate) fn upsample_generic_nearest(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_stride: usize,
    h_factor: usize,
    v_factor: usize,
) {
    for y in 0..in_height {
        let in_row: &[u8] = &input[y * in_width..y * in_width + in_width];
        // Build one upsampled row (horizontal replication)
        let out_y_base: usize = y * v_factor;
        let first_out_row: usize = out_y_base * out_stride;
        for (x, &val) in in_row.iter().enumerate() {
            let out_x: usize = x * h_factor;
            for dx in 0..h_factor {
                output[first_out_row + out_x + dx] = val;
            }
        }
        // Replicate the row vertically
        for dy in 1..v_factor {
            let src_start: usize = first_out_row;
            let dst_start: usize = (out_y_base + dy) * out_stride;
            let copy_len: usize = in_width * h_factor;
            output.copy_within(src_start..src_start + copy_len, dst_start);
        }
    }
}

/// Dispatch fancy_h2v2_row to AVX2, SSE2, or scalar based on CPU features.
#[inline]
pub(super) fn fancy_h2v2_row_dispatch(
    cur: &[u8],
    neighbor: &[u8],
    output: &mut [u8],
    in_width: usize,
) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if crate::cpu_has!("avx2") {
            return crate::simd::x86_64::avx2_upsample::avx2_fancy_h2v2_row(
                cur, neighbor, output, in_width,
            );
        }
        if crate::cpu_has!("sse2") {
            return crate::simd::x86_64::upsample::sse2_fancy_h2v2_row(
                cur, neighbor, output, in_width,
            );
        }
    }

    crate::decode::upsample::fancy_h2v2_row(cur, neighbor, output, in_width);
}

/// Dispatch fancy_h2v2_strided to AVX2 row function or scalar.
#[inline]
pub(super) fn fancy_h2v2_strided_dispatch(
    input: &[u8],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
) {
    for y in 0..in_height {
        let cur_row: &[u8] = &input[y * in_stride..y * in_stride + in_width];
        let above: &[u8] = if y > 0 {
            &input[(y - 1) * in_stride..(y - 1) * in_stride + in_width]
        } else {
            cur_row
        };
        let below: &[u8] = if y + 1 < in_height {
            &input[(y + 1) * in_stride..(y + 1) * in_stride + in_width]
        } else {
            cur_row
        };

        for (v, neighbor) in [(0, above), (1, below)] {
            let out_y: usize = y * 2 + v;
            let out_row: &mut [u8] = &mut output[out_y * out_width..];
            fancy_h2v2_row_dispatch(cur_row, neighbor, out_row, in_width);
        }
    }
}

impl<'a> Decoder<'a> {
    /// Select the IDCT function based on the configured DCT method.
    #[inline(always)]
    pub(super) fn idct_fn(&self) -> fn(&[i16; 64], &[u16; 64], &mut [u8; 64]) {
        match self.dct_method {
            DctMethod::IsFast => self.routines.idct_ifast,
            DctMethod::Float => self.routines.idct_float,
            DctMethod::IsLow => self.routines.idct_islow,
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(super) fn idct_islow(&self, coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::idct::neon_idct_islow(coeffs, quant, output);
        }

        #[allow(unreachable_code)]
        (self.routines.idct_islow)(coeffs, quant, output)
    }

    /// IDCT writing directly to a strided destination buffer (no intermediate copy).
    /// Dispatches to ISLOW/IFAST/Float based on `self.dct_method`.
    ///
    /// # Safety
    /// `output` must point to at least `7 * stride + 8` writable bytes.
    #[inline(always)]
    pub(super) unsafe fn idct_islow_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
    ) {
        unsafe {
            // For ISLOW, use optimized strided SIMD paths when available.
            if matches!(self.dct_method, DctMethod::IsLow) {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::idct::neon_idct_islow_strided(
                        coeffs, quant, output, stride,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_idct::avx2_idct_islow_strided(
                            coeffs, quant, output, stride,
                        );
                    }
                    if crate::cpu_has!("sse2") {
                        return crate::simd::x86_64::idct::sse2_idct_islow_strided(
                            coeffs, quant, output, stride,
                        );
                    }
                }
            }

            // Generic path: IDCT into temp buffer, then copy row-by-row.
            #[allow(unreachable_code)]
            {
                let idct = self.idct_fn();
                let mut tmp = [0u8; 64];
                idct(coeffs, quant, &mut tmp);
                for row in 0..8 {
                    core::ptr::copy_nonoverlapping(
                        tmp.as_ptr().add(row * 8),
                        output.add(row * stride),
                        8,
                    );
                }
            }
        }
    }

    /// Scale-aware IDCT dispatch: picks 8x8, 4x4, 2x2, or 1x1 based on block_size.
    ///
    /// # Safety
    /// `output` must point to sufficient writable bytes for the chosen block_size × stride.
    #[inline(always)]
    pub(super) unsafe fn idct_scaled_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
        block_size: usize,
    ) {
        unsafe {
            match block_size {
                16 => idct_extended::idct_16x16_strided(coeffs, quant, output, stride),
                15 => idct_extended::idct_15x15_strided(coeffs, quant, output, stride),
                14 => idct_extended::idct_14x14_strided(coeffs, quant, output, stride),
                13 => idct_extended::idct_13x13_strided(coeffs, quant, output, stride),
                12 => idct_extended::idct_12x12_strided(coeffs, quant, output, stride),
                11 => idct_extended::idct_11x11_strided(coeffs, quant, output, stride),
                10 => idct_extended::idct_10x10_strided(coeffs, quant, output, stride),
                9 => idct_extended::idct_9x9_strided(coeffs, quant, output, stride),
                8 => self.idct_islow_strided(coeffs, quant, output, stride),
                7 => idct_extended::idct_7x7_strided(coeffs, quant, output, stride),
                6 => idct_extended::idct_6x6_strided(coeffs, quant, output, stride),
                5 => idct_extended::idct_5x5_strided(coeffs, quant, output, stride),
                4 => idct_scaled::idct_4x4_strided(coeffs, quant, output, stride),
                3 => idct_extended::idct_3x3_strided(coeffs, quant, output, stride),
                2 => idct_scaled::idct_2x2_strided(coeffs, quant, output, stride),
                1 => idct_scaled::idct_1x1_strided(coeffs, quant, output, stride),
                _ => unreachable!("invalid block_size: {}", block_size),
            }
        }
    }

    /// Compute per-component IDCT block size for scaled decode.
    ///
    /// Matches C libjpeg-turbo's `jpeg_calc_output_dimensions` (jdmaster.c):
    /// chroma components get a larger IDCT to absorb subsampling factors,
    /// eliminating spatial upsampling. For example, 4:2:0 at 1/2 scale uses
    /// 4x4 IDCT for Y but 8x8 IDCT for Cb/Cr, so all planes end up the same
    /// pixel dimensions — no upsample needed.
    pub(super) fn compute_comp_block_size(
        min_block_size: usize,
        max_h: usize,
        max_v: usize,
        h_samp: usize,
        v_samp: usize,
    ) -> usize {
        let mut ssize: usize = min_block_size;
        while ssize < 8
            && (max_h * min_block_size).is_multiple_of(h_samp * ssize * 2)
            && (max_v * min_block_size).is_multiple_of(v_samp * ssize * 2)
        {
            ssize *= 2;
        }
        ssize
    }

    /// Compute per-component block sizes for all components in a frame.
    /// Returns a fixed-size array (frame components are capped at 4 by
    /// `read_sof`); only the first `components.len()` entries are
    /// meaningful. Fixed-size to keep small-image decode allocation-free
    /// here (issue #351).
    pub(super) fn compute_all_comp_block_sizes(
        min_block_size: usize,
        max_h: usize,
        max_v: usize,
        frame: &FrameHeader,
    ) -> [usize; 4] {
        let mut sizes = [min_block_size; 4];
        for (size, comp) in sizes.iter_mut().zip(frame.components.iter()) {
            *size = Self::compute_comp_block_size(
                min_block_size,
                max_h,
                max_v,
                comp.horizontal_sampling as usize,
                comp.vertical_sampling as usize,
            );
        }
        sizes
    }

    #[inline(always)]
    pub(super) fn ycbcr_to_rgb_row(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgb_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        (self.routines.ycbcr_to_rgb_row)(y, cb, cr, out, width)
    }

    #[inline(always)]
    pub(super) fn ycbcr_to_rgba_row(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgba_row(y, cb, cr, out, width);
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_rgba_row(
                    y, cb, cr, out, width,
                );
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            return crate::simd::wasm32::color::wasm_ycbcr_to_rgba_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_rgba_row(y, cb, cr, out, width)
    }

    #[inline(always)]
    pub(super) fn ycbcr_to_bgr_row(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_bgr_row(y, cb, cr, out, width);
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_bgr_row(
                    y, cb, cr, out, width,
                );
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            return crate::simd::wasm32::color::wasm_ycbcr_to_bgr_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_bgr_row(y, cb, cr, out, width)
    }

    #[inline(always)]
    pub(super) fn ycbcr_to_bgra_row(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_bgra_row(y, cb, cr, out, width);
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_bgra_row(
                    y, cb, cr, out, width,
                );
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            return crate::simd::wasm32::color::wasm_ycbcr_to_bgra_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_bgra_row(y, cb, cr, out, width)
    }

    /// Dispatch color conversion for one row based on the target pixel format.
    /// `row_index` is the output row number, used for ordered dithering in RGB565 mode.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn color_convert_row(
        &self,
        format: PixelFormat,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
        row_index: usize,
    ) {
        match format {
            PixelFormat::Rgb => self.ycbcr_to_rgb_row(y, cb, cr, out, width),
            PixelFormat::Rgba => self.ycbcr_to_rgba_row(y, cb, cr, out, width),
            PixelFormat::Bgr => self.ycbcr_to_bgr_row(y, cb, cr, out, width),
            PixelFormat::Bgra => self.ycbcr_to_bgra_row(y, cb, cr, out, width),
            #[allow(unreachable_code)]
            PixelFormat::Rgbx => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_rgbx_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_rgbx_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 0, 1, 2, 3)
            }
            #[allow(unreachable_code)]
            PixelFormat::Bgrx => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_bgrx_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_bgrx_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 2, 1, 0, 3)
            }
            #[allow(unreachable_code)]
            PixelFormat::Xrgb => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_xrgb_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_xrgb_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0)
            }
            #[allow(unreachable_code)]
            PixelFormat::Xbgr => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_xbgr_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_xbgr_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 3, 2, 1, 0)
            }
            #[allow(unreachable_code)]
            PixelFormat::Argb => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_argb_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_argb_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0)
            }
            #[allow(unreachable_code)]
            PixelFormat::Abgr => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_abgr_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_abgr_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 3, 2, 1, 0)
            }
            PixelFormat::Rgb565 => {
                if self.dither_565 {
                    crate::decode::color::ycbcr_to_rgb565_dithered_row(
                        y, cb, cr, out, width, row_index,
                    )
                } else {
                    crate::decode::color::ycbcr_to_rgb565_row(y, cb, cr, out, width)
                }
            }
            PixelFormat::Grayscale | PixelFormat::Cmyk => {
                unreachable!("grayscale/cmyk handled separately")
            }
        }
    }

    /// Merged H2V1 upsample + color convert dispatch.
    #[inline(always)]
    pub(super) fn merged_h2v1(
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        rgb_out: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                crate::simd::x86_64::avx2_merged::avx2_merged_h2v1_ycbcr_to_rgb(
                    y_row, cb_row, cr_row, rgb_out, width,
                );
                return;
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            crate::simd::aarch64::merged::neon_merged_h2v1_ycbcr_to_rgb(
                y_row, cb_row, cr_row, rgb_out, width,
            );
            return;
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            return crate::simd::wasm32::merged::wasm_merged_h2v1_ycbcr_to_rgb(
                y_row, cb_row, cr_row, rgb_out, width,
            );
        }

        #[allow(unreachable_code)]
        crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb(
            y_row, cb_row, cr_row, rgb_out, width,
        );
    }

    /// Merged H2V2 upsample + color convert dispatch.
    #[inline(always)]
    pub(super) fn merged_h2v2(
        y_row0: &[u8],
        y_row1: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        rgb_out0: &mut [u8],
        rgb_out1: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                crate::simd::x86_64::avx2_merged::avx2_merged_h2v2_ycbcr_to_rgb(
                    y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width,
                );
                return;
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            crate::simd::aarch64::merged::neon_merged_h2v2_ycbcr_to_rgb(
                y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width,
            );
            return;
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            return crate::simd::wasm32::merged::wasm_merged_h2v2_ycbcr_to_rgb(
                y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width,
            );
        }

        #[allow(unreachable_code)]
        crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb(
            y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width,
        );
    }

    #[inline(always)]
    pub(super) fn fancy_upsample_h2v1(&self, input: &[u8], in_width: usize, output: &mut [u8]) {
        // For in_width <= 2, C's merged path uses box filter (no interpolation).
        // NEON/SIMD paths may not handle this edge case correctly, so use scalar.
        if in_width <= 2 {
            crate::decode::upsample::fancy_h2v1(input, in_width, output, 0);
            return;
        }

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::upsample::neon_fancy_upsample_h2v1(
                input, in_width, output,
            );
        }

        #[allow(unreachable_code)]
        (self.routines.fancy_upsample_h2v1)(input, in_width, output)
    }

    /// Fancy h2v2 upsample. On aarch64 this uses a dedicated helper that
    /// fuses the two vertical blends into one pass before the h2v1 stage.
    pub(super) fn fancy_h2v2(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            crate::simd::aarch64::upsample::neon_fancy_upsample_h2v2(
                input, in_width, in_height, output, out_width,
            )
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd"))]
        {
            return crate::simd::wasm32::upsample::wasm_fancy_upsample_h2v2(
                input, in_width, in_height, output, out_width,
            );
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                return crate::simd::x86_64::avx2_upsample::avx2_fancy_upsample_h2v2(
                    input, in_width, in_height, output, out_width,
                );
            }
            if crate::cpu_has!("sse2") {
                return crate::simd::x86_64::upsample::sse2_fancy_upsample_h2v2(
                    input, in_width, in_height, output, out_width,
                );
            }
        }

        // Fused H2V2: vertical + horizontal in one pass using >> 4 arithmetic.
        // Matches C libjpeg-turbo h2v2_fancy_upsample exactly, avoiding
        // double-rounding from the previous two-pass approach.
        #[allow(unreachable_code)]
        {
            crate::decode::upsample::fancy_h2v2(
                input,
                in_width,
                in_height,
                output,
                out_width,
                in_height * 2,
            );
        }
    }

    /// One output row of the vertical-only 2x triangle filter (S440).
    ///
    /// Matches C jdsample.c `h1v2_fancy_upsample`:
    /// `(3*cur + adj + bias) >> 2` with bias 1 for the top row
    /// (adj = above) and bias 2 for the bottom row (adj = below); the
    /// alternating bias avoids a systematic rounding drift. Shared by
    /// the whole-plane `fancy_h1v2` and the row-streaming H1V2 decode
    /// path so the rounding behaviour cannot silently diverge.
    #[inline(always)]
    pub(super) fn fancy_h1v2_row(cur: &[u8], adj: &[u8], out: &mut [u8], width: usize, bias: u16) {
        for i in 0..width {
            out[i] = ((3 * cur[i] as u16 + adj[i] as u16 + bias) >> 2) as u8;
        }
    }

    /// Fancy h1v2 upsample: vertical-only 2x (for S440).
    /// Each input row produces two output rows using triangle filter vertically.
    /// Horizontal samples are copied 1:1.
    pub(super) fn fancy_h1v2(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
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

            let out_y_top = y * 2;
            let out_y_bot = y * 2 + 1;
            // split_at_mut to get non-overlapping mutable slices
            let (top_half, bot_half) = output.split_at_mut(out_y_bot * out_width);
            let out_top = &mut top_half[out_y_top * out_width..out_y_top * out_width + in_width];
            let out_bot = &mut bot_half[..in_width];
            Self::fancy_h1v2_row(cur_row, above, out_top, in_width, 1);
            Self::fancy_h1v2_row(cur_row, below, out_bot, in_width, 2);
        }
    }

    /// Return one decoded component at full output resolution.
    ///
    /// This mirrors `jdsample.c`'s per-component selection: full-size planes
    /// are borrowed, 2:1 ratios use the fancy triangle filters when enabled,
    /// and all other integral ratios use box replication.  The active source
    /// area is repacked first so right-edge MCU padding is never treated as an
    /// image sample by a fancy filter.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn upsample_component_plane<'plane>(
        &self,
        component_plane: &'plane [u8],
        component_index: usize,
        frame: &FrameHeader,
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
        full_width: usize,
        full_height: usize,
        output_width: usize,
        output_height: usize,
        block_size: usize,
    ) -> Result<(Cow<'plane, [u8]>, usize)> {
        let component = frame.components.get(component_index).ok_or_else(|| {
            JpegError::CorruptData(format!(
                "missing component {component_index} for colorspace conversion"
            ))
        })?;
        let component_block_size = *comp_block_sizes.get(component_index).ok_or_else(|| {
            JpegError::CorruptData(format!(
                "missing IDCT block size for component {component_index}"
            ))
        })?;
        let component_width =
            mcus_x * component.horizontal_sampling as usize * component_block_size;
        let component_height = mcus_y * component.vertical_sampling as usize * component_block_size;
        let required_len = component_width
            .checked_mul(component_height)
            .ok_or_else(|| {
                JpegError::CorruptData(format!(
                    "component {component_index} plane dimensions overflow"
                ))
            })?;
        if component_width == 0 || component_height == 0 || component_plane.len() < required_len {
            return Err(JpegError::CorruptData(format!(
                "invalid component {component_index} plane: {} bytes for {component_width}x{component_height}",
                component_plane.len()
            )));
        }
        if !full_width.is_multiple_of(component_width)
            || !full_height.is_multiple_of(component_height)
        {
            return Err(JpegError::CorruptData(format!(
                "non-integral upsample ratio for component {component_index}: \
                 {component_width}x{component_height} -> {full_width}x{full_height}"
            )));
        }
        let horizontal_factor = full_width / component_width;
        let vertical_factor = full_height / component_height;
        if horizontal_factor == 0 || vertical_factor == 0 {
            return Err(JpegError::CorruptData(format!(
                "zero upsample factor for component {component_index}"
            )));
        }
        if horizontal_factor == 1 && vertical_factor == 1 {
            return Ok((Cow::Borrowed(component_plane), component_width));
        }

        let active_width = output_width.div_ceil(horizontal_factor);
        let active_height = output_height.div_ceil(vertical_factor);
        if active_width > component_width || active_height > component_height {
            return Err(JpegError::CorruptData(format!(
                "component {component_index} active area exceeds its plane: \
                 active={active_width}x{active_height}, plane={component_width}x{component_height}"
            )));
        }
        let mut active = Vec::with_capacity(active_width * active_height);
        for row in 0..active_height {
            let start = row * component_width;
            active.extend_from_slice(&component_plane[start..start + active_width]);
        }

        let mut full = vec![0u8; full_width * full_height];
        // C disables fancy upsampling for a 1x1 scaled IDCT, and for the
        // horizontal 2:1 kernels when the active input is at most two pixels.
        let horizontal_fancy_too_narrow = horizontal_factor == 2 && active_width <= 2;
        let use_box_filter = self.fast_upsample || block_size == 1 || horizontal_fancy_too_narrow;
        if use_box_filter {
            upsample_generic_nearest(
                &active,
                active_width,
                active_height,
                &mut full,
                full_width,
                horizontal_factor,
                vertical_factor,
            );
        } else if horizontal_factor == 2 && vertical_factor == 1 {
            for row in 0..active_height {
                self.fancy_upsample_h2v1(
                    &active[row * active_width..(row + 1) * active_width],
                    active_width,
                    &mut full[row * full_width..(row + 1) * full_width],
                );
            }
        } else if horizontal_factor == 2 && vertical_factor == 2 {
            fancy_h2v2_strided_dispatch(
                &active,
                active_width,
                active_width,
                active_height,
                &mut full,
                full_width,
            );
        } else if horizontal_factor == 1 && vertical_factor == 2 {
            self.fancy_h1v2(&active, active_width, active_height, &mut full, full_width);
        } else {
            upsample_generic_nearest(
                &active,
                active_width,
                active_height,
                &mut full,
                full_width,
                horizontal_factor,
                vertical_factor,
            );
        }

        Ok((Cow::Owned(full), full_width))
    }

    /// Produce grayscale output through the same two conversions as
    /// libjpeg-turbo's `jdcolor.c`: YCbCr/gray copies the fully upsampled
    /// component 0, while JCS_RGB applies the fixed-point RGB→Y matrix after
    /// all three component planes have been upsampled.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_grayscale_override(
        &self,
        component_planes: &[Vec<u8>],
        frame: &FrameHeader,
        out_width: usize,
        out_height: usize,
        uncropped_out_width: usize,
        full_width: usize,
        full_height: usize,
        mcus_x: usize,
        mcus_y: usize,
        block_size: usize,
        comp_block_sizes: &[usize],
        crop_x: Option<usize>,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        let jpeg_color_space = self.detect_color_space();
        let (required_components, geometry_components): (usize, usize) = match jpeg_color_space {
            ColorSpace::Grayscale if frame.components.len() == 1 => (1, 1),
            ColorSpace::YCbCr if frame.components.len() == 3 => (1, 3),
            ColorSpace::Rgb if frame.components.len() == 3 => (3, 3),
            other => {
                return Err(JpegError::Unsupported(format!(
                    "cannot convert {other:?} JPEG with {} components to grayscale (C: JERR_CONVERSION_NOTIMPL)",
                    frame.components.len()
                )));
            }
        };
        if comp_block_sizes.len() < geometry_components {
            return Err(JpegError::CorruptData(format!(
                "{jpeg_color_space:?} grayscale conversion needs {geometry_components} component geometries, got {}",
                comp_block_sizes.len()
            )));
        }
        if component_planes.len() < required_components {
            return Err(JpegError::CorruptData(format!(
                "{jpeg_color_space:?} grayscale conversion needs {required_components} components, got {}",
                component_planes.len()
            )));
        }
        // Preserve P4-21's strict contract.  Grayscale needs only component
        // 0, but a strict YCbCr decode still rejects streams where chroma
        // out-samples luma; lenient mode is the opt-in that accepts them.
        if !self.lenient && jpeg_color_space == ColorSpace::YCbCr {
            let component_width = |index: usize| {
                mcus_x
                    * frame.components[index].horizontal_sampling as usize
                    * comp_block_sizes[index]
            };
            let component_height = |index: usize| {
                mcus_y
                    * frame.components[index].vertical_sampling as usize
                    * comp_block_sizes[index]
            };
            let luma_width = component_width(0);
            let luma_height = component_height(0);
            let cb_width = component_width(1);
            let cb_height = component_height(1);
            let cr_width = component_width(2);
            let cr_height = component_height(2);
            if [
                luma_width,
                luma_height,
                cb_width,
                cb_height,
                cr_width,
                cr_height,
            ]
            .contains(&0)
            {
                return Err(JpegError::CorruptData(
                    "cannot validate grayscale sampling factors with zero component geometry"
                        .into(),
                ));
            }
            let cb_h_factor = luma_width / cb_width;
            let cb_v_factor = luma_height / cb_height;
            let cr_h_factor = luma_width / cr_width;
            let cr_v_factor = luma_height / cr_height;
            if cb_h_factor == 0 || cb_v_factor == 0 || cr_h_factor == 0 || cr_v_factor == 0 {
                return Err(JpegError::CorruptData(format!(
                    "chroma upsample factor zero (a chroma component out-samples luma): \
                     cb={cb_h_factor}x{cb_v_factor} cr={cr_h_factor}x{cr_v_factor}"
                )));
            }
        }

        let mut full_planes: Vec<Cow<'_, [u8]>> = Vec::with_capacity(required_components);
        let mut strides: Vec<usize> = Vec::with_capacity(required_components);
        for (component_index, component_plane) in component_planes
            .iter()
            .take(required_components)
            .enumerate()
        {
            let (full_plane, stride) = self.upsample_component_plane(
                component_plane,
                component_index,
                frame,
                mcus_x,
                mcus_y,
                comp_block_sizes,
                full_width,
                full_height,
                uncropped_out_width,
                out_height,
                block_size,
            )?;
            full_planes.push(full_plane);
            strides.push(stride);
        }

        let output_x_offset = crop_x.unwrap_or(0);
        if output_x_offset.saturating_add(out_width) > full_width || out_height > full_height {
            return Err(JpegError::CorruptData(format!(
                "grayscale output region {output_x_offset}+{out_width}x{out_height} exceeds {full_width}x{full_height}"
            )));
        }
        let mut data = Vec::with_capacity(out_width * out_height);
        if jpeg_color_space == ColorSpace::Rgb {
            for y in 0..out_height {
                let red_row = &full_planes[0][y * strides[0]..];
                let green_row = &full_planes[1][y * strides[1]..];
                let blue_row = &full_planes[2][y * strides[2]..];
                for x in output_x_offset..output_x_offset + out_width {
                    // `jdcolor.c::rgb_gray_convert`: FIX(0.299) * R +
                    // FIX(0.587) * G + FIX(0.114) * B + ONE_HALF, >> 16.
                    let y_sample = (19595 * red_row[x] as u32
                        + 38470 * green_row[x] as u32
                        + 7471 * blue_row[x] as u32
                        + 32768)
                        >> 16;
                    data.push(y_sample as u8);
                }
            }
        } else {
            for y in 0..out_height {
                let row = &full_planes[0][y * strides[0] + output_x_offset
                    ..y * strides[0] + output_x_offset + out_width];
                data.extend_from_slice(row);
            }
        }

        Ok(Image {
            xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
            iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
            width: out_width,
            height: out_height,
            pixel_format: PixelFormat::Grayscale,
            precision: 8,
            data,
            icc_profile,
            exif_data,
            comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
            density: self.metadata.density,
            saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
            warnings,
        })
    }
}
