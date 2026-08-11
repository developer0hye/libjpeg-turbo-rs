use super::color::{fancy_h2v2_row_dispatch, fancy_h2v2_strided_dispatch};
use super::{upsample_generic_nearest, Decoder, Image, ImageInfo};
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::quant_table::QuantTable;
use crate::common::try_alloc::{try_clone_opt, try_clone_opt_string, try_clone_saved_markers};
use crate::common::types::{ColorSpace, ComponentInfo, PixelFormat};
use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Output destination for one decode: an owned vector or caller buffer.
enum OutBuf<'a> {
    Owned(Vec<u8>),
    Borrowed(&'a mut [u8]),
}

impl core::ops::Deref for OutBuf<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            OutBuf::Owned(v) => v,
            OutBuf::Borrowed(b) => b,
        }
    }
}

impl core::ops::DerefMut for OutBuf<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            OutBuf::Owned(v) => v,
            OutBuf::Borrowed(b) => b,
        }
    }
}

impl OutBuf<'_> {
    fn into_vec(self) -> Vec<u8> {
        match self {
            OutBuf::Owned(v) => v,
            OutBuf::Borrowed(_) => Vec::new(),
        }
    }
}

/// Pad/alpha byte position within a 4bpp pixel: the one offset in `0..=3` not
/// taken by R, G, or B (the four offsets sum to 6). Deriving it from the
/// format — like the CMYK/YCCK conversion arms — makes a wrong alpha-first vs
/// alpha-last grouping unrepresentable (issue #369). `None` for everything
/// else, including the 4bpp formats without R/G/B offsets (`Cmyk`).
pub(super) fn pad_alpha_offset(format: PixelFormat) -> Option<usize> {
    match (
        format.red_offset(),
        format.green_offset(),
        format.blue_offset(),
    ) {
        (Some(r), Some(g), Some(b)) if format.bytes_per_pixel() == 4 => Some(6 - r - g - b),
        _ => None,
    }
}

/// Claim and size the caller buffer, or allocate an owned output buffer.
fn take_out_buf<'a>(sink: &mut Option<&'a mut [u8]>, size: usize) -> Result<OutBuf<'a>> {
    match sink.take() {
        Some(buf) => {
            if buf.len() < size {
                return Err(JpegError::BufferTooSmall {
                    need: size,
                    got: buf.len(),
                });
            }
            Ok(OutBuf::Borrowed(&mut buf[..size]))
        }
        None => Ok(OutBuf::Owned(vec![0u8; size])),
    }
}

impl<'a> Decoder<'a> {
    /// Decode the full image into an owned [`Image`] using the
    /// configuration set on this decoder (output format, scaling, crop,
    /// leniency, ...). To decode into a caller-owned buffer, see
    /// [`Decoder::decode_image_into`] (which avoids the per-frame output
    /// allocation on the standard direct-output paths; its docs list the
    /// staged exceptions).
    pub fn decode_image(&self) -> Result<Image> {
        self.decode_image_with_sink(&mut None)
    }

    /// Header-derived limit checks shared by every decode entry point
    /// (decode_image_inner, decode_raw, output_buffer_size): dimensions,
    /// pixel product, and scan count, all validated before any
    /// output-scale allocation (issue #355; the sizing-path and raw-path
    /// coverage came from the codex review).
    pub(super) fn check_header_limits(&self) -> Result<()> {
        let frame = &self.metadata.frame;
        let width: usize = frame.width as usize;
        let height: usize = frame.height as usize;
        self.limits.check_frame(width, height)?;
        let total_pixels: u64 = (width as u64) * (height as u64);
        if self.metadata.scans.len() > self.limits.max_scans {
            return Err(JpegError::LimitExceeded {
                what: "scan count",
                actual: self.metadata.scans.len() as u64,
                limit: self.limits.max_scans as u64,
            });
        }
        // Estimated decode memory: output buffer + component planes.
        // Enforced here so the sizing and raw paths share it (stop-gate
        // review on #355); the estimate intentionally includes the
        // packed-output term even for decode_raw — one conservative
        // model, one enforcement point.
        if let Some(max_mem) = self.limits.max_memory {
            let nc = frame.components.len();
            let out_bpp = self
                .output_format
                .unwrap_or(if nc == 1 {
                    PixelFormat::Grayscale
                } else {
                    PixelFormat::Rgb
                })
                .bytes_per_pixel();
            let upsample_expansion_planes: usize = if nc == 3 {
                let max_h = frame
                    .components
                    .iter()
                    .map(|component| component.horizontal_sampling)
                    .max()
                    .unwrap_or(1);
                let max_v = frame
                    .components
                    .iter()
                    .map(|component| component.vertical_sampling)
                    .max()
                    .unwrap_or(1);
                let is_subsampled = |component: &ComponentInfo| {
                    component.horizontal_sampling != max_h || component.vertical_sampling != max_v
                };
                match self.detect_color_space() {
                    // Direct RGB output and RGB→gray both need all three
                    // component planes at full resolution.
                    ColorSpace::Rgb => frame
                        .components
                        .iter()
                        .filter(|component| is_subsampled(component))
                        .count(),
                    // YCbCr→gray only needs component 0, but P4-72 now
                    // expands it when an unusual legal stream samples it
                    // below another component.
                    ColorSpace::YCbCr
                        if self.effective_output_colorspace() == Some(ColorSpace::Grayscale) =>
                    {
                        usize::from(is_subsampled(&frame.components[0]))
                    }
                    _ => 0,
                }
            } else {
                0
            };
            // Progressive adds one [i16; 64] per 8x8 block per component
            // (~2 B/pixel/component) plus the ac_max_k byte per block —
            // without this term the ceiling under-enforced by ~1.5x on
            // exactly the hostile input class (#355 review).
            let coeff_bytes: u64 = if frame.is_progressive {
                total_pixels * (2 * nc as u64) + total_pixels * nc as u64 / 64
            } else {
                0
            };
            let total_estimated: u64 = total_pixels
                * (out_bpp as u64 + nc as u64 + upsample_expansion_planes as u64)
                + coeff_bytes;
            if total_estimated > max_mem {
                return Err(JpegError::LimitExceeded {
                    what: "estimated decode memory",
                    actual: total_estimated,
                    limit: max_mem,
                });
            }
        }
        Ok(())
    }

    /// Bytes `decode_image_into` needs for this stream with the current
    /// decoder options. Exact for the standard paths; a safe upper
    /// bound when an output-colourspace override is active (sized at 4
    /// bytes/pixel) or when cropping trims the image below the
    /// MCU-aligned estimate.
    #[must_use = "the returned size is the whole point of this query"]
    pub fn output_buffer_size(&self) -> Result<usize> {
        // Untrusted-input workflow is size -> allocate -> decode, so the
        // limits must fire here too or the caller OOMs before decode can
        // reject (codex P1 on #355).
        self.check_header_limits()?;
        let frame = &self.metadata.frame;
        let num_components: usize = frame.components.len();
        let bpp: usize = if self.output_colorspace.is_some() {
            4
        } else {
            self.output_format
                .unwrap_or(match num_components {
                    1 => PixelFormat::Grayscale,
                    4 => PixelFormat::Cmyk,
                    _ => PixelFormat::Rgb,
                })
                .bytes_per_pixel()
        };

        // The 12-bit and lossless paths return through
        // decode_12bit_as_8bit / decode_lossless_image, which apply
        // neither scaled decode nor horizontal crop — size them at full
        // frame geometry or the advertised size under-reports (codex P2
        // on #354). Vertical crop still applies to every path: it is a
        // post-slice in decode_image.
        let scale_and_hcrop_apply: bool = frame.precision != 12 && !frame.is_lossless;
        let (base_w, base_h): (usize, usize) = if scale_and_hcrop_apply {
            (
                self.scale.scale_dim(frame.width as usize),
                self.scale.scale_dim(frame.height as usize),
            )
        } else {
            (frame.width as usize, frame.height as usize)
        };

        // Horizontal crop mirrors decode_image_inner: X aligns down to
        // the scaled iMCU boundary and the width expands to compensate,
        // then clamps to the image.
        let out_w: usize = if !scale_and_hcrop_apply {
            base_w
        } else if let (Some(cx), Some(cw)) = (self.crop_x, self.crop_width) {
            let max_h_samp: usize = frame
                .components
                .iter()
                .map(|c| c.horizontal_sampling as usize)
                .max()
                .unwrap_or(1);
            let scaled_imcu_w: usize = max_h_samp * self.scale.block_size();
            let aligned_x: usize = (cx / scaled_imcu_w) * scaled_imcu_w;
            let expanded_w: usize = cw + (cx - aligned_x);
            expanded_w.min(base_w.saturating_sub(aligned_x))
        } else {
            base_w
        };
        // Vertical crop is applied as a post-slice in decode_image.
        let out_h: usize = if let (Some(cy), Some(ch)) = (self.crop_y, self.crop_height) {
            let offset: usize = cy.min(base_h);
            ch.min(base_h.saturating_sub(offset))
        } else {
            base_h
        };

        out_w
            .checked_mul(out_h)
            .and_then(|px| px.checked_mul(bpp))
            .ok_or_else(|| {
                JpegError::Unsupported(format!("output size {out_w}x{out_h}x{bpp} exceeds usize"))
            })
    }

    /// Decode into a caller-provided buffer, returning the metadata
    /// [`Image`] would carry minus the pixel `Vec` (issue #354).
    ///
    /// `out` must hold at least [`Self::output_buffer_size`] bytes; a
    /// short buffer is a typed [`JpegError::BufferTooSmall`], never a
    /// panic or truncation. The standard baseline/progressive output
    /// paths (grayscale, 4:4:4, and every streamed subsampling mode)
    /// write pixels directly into `out` with no output-sized heap
    /// allocation; the remaining paths (CMYK, 12-bit, lossless, output
    /// colourspace overrides, vertical crop) stage through an internal
    /// buffer and copy, which is still allocation-neutral versus
    /// `decode_image` and byte-identical to it.
    ///
    /// On error the contents of `out` are unspecified: a decode may
    /// have written part of the frame before failing.
    pub fn decode_image_into(&self, out: &mut [u8]) -> Result<ImageInfo> {
        // Vertical crop re-slices the finished image below, which cannot
        // be expressed in a caller buffer written at full height — stage
        // those decodes.
        let use_sink: bool = self.crop_y.is_none() && self.crop_height.is_none();
        let mut sink: Option<&mut [u8]> = if use_sink { Some(&mut *out) } else { None };
        let image: Image = self.decode_image_with_sink(&mut sink)?;
        let sink_consumed: bool = use_sink && sink.is_none();

        // Structural, not asserted: the staged path reports exactly what
        // it copied, so a hypothetical path producing data shorter than
        // w*h*bpp cannot over-report and expose stale caller bytes.
        let bytes_written: usize = if sink_consumed {
            debug_assert!(image.data.is_empty(), "sink branches return empty data");
            image.width * image.height * image.pixel_format.bytes_per_pixel()
        } else {
            if out.len() < image.data.len() {
                return Err(JpegError::BufferTooSmall {
                    need: image.data.len(),
                    got: out.len(),
                });
            }
            out[..image.data.len()].copy_from_slice(&image.data);
            image.data.len()
        };
        Ok(ImageInfo {
            width: image.width,
            height: image.height,
            pixel_format: image.pixel_format,
            precision: image.precision,
            bytes_written,
            icc_profile: image.icc_profile,
            exif_data: image.exif_data,
            xmp_data: image.xmp_data,
            iptc_data: image.iptc_data,
            comment: image.comment,
            density: image.density,
            saved_markers: image.saved_markers,
            warnings: image.warnings,
        })
    }

    pub(super) fn decode_image_with_sink(&self, sink: &mut Option<&mut [u8]>) -> Result<Image> {
        let mut image: Image = self.decode_image_inner(sink)?;
        if !self.marker_processors.is_empty() {
            for marker in &image.saved_markers {
                if let Some(processor) = self.marker_processors.get(&marker.code) {
                    processor(&marker.data);
                }
            }
        }
        // When stop_on_warning is enabled, any accumulated warning becomes fatal.
        if self.stop_on_warning && !image.warnings.is_empty() {
            let first_warning = &image.warnings[0];
            let detail = match first_warning {
                DecodeWarning::HuffmanError {
                    mcu_x,
                    mcu_y,
                    message,
                } => format!("Huffman error at MCU ({}, {}): {}", mcu_x, mcu_y, message),
                DecodeWarning::TruncatedData {
                    decoded_mcus,
                    total_mcus,
                } => format!("truncated: decoded {} of {} MCUs", decoded_mcus, total_mcus),
                DecodeWarning::UnsupportedRecovered { detail } => {
                    format!("unsupported feature: {}", detail)
                }
            };
            return Err(JpegError::CorruptData(format!(
                "stop_on_warning: {}",
                detail
            )));
        }
        // Apply vertical crop: slice the output to exactly the requested
        // crop_y..crop_y+crop_height region. Horizontal crop is handled
        // during decode; vertical crop is applied here to avoid threading
        // the offset through every output path.
        if let (Some(cy), Some(ch)) = (self.crop_y, self.crop_height) {
            // Crop coordinates are in the output (post-scale) space
            let offset: usize = cy.min(image.height);
            let height: usize = ch.min(image.height.saturating_sub(offset));
            if offset > 0 || height < image.height {
                // A sink-claimed decode returns empty data; vertical crop
                // is excluded from sink mode in decode_image_into, so this
                // is unreachable — keep it panic-free if that invariant
                // ever breaks rather than indexing out of range.
                debug_assert!(
                    !image.data.is_empty(),
                    "vertical crop cannot apply to a sink-claimed decode"
                );
                if !image.data.is_empty() {
                    let bpp: usize = image.pixel_format.bytes_per_pixel();
                    let row_bytes: usize = image.width * bpp;
                    let start: usize = offset * row_bytes;
                    let end: usize = start + height * row_bytes;
                    image.data = image.data[start..end].to_vec();
                    image.height = height;
                }
            }
        }
        Ok(image)
    }

    /// Decode a 12-bit JPEG by delegating to `decompress_12bit`, then scaling
    /// the 12-bit samples (0-4095) down to 8-bit (0-255). Converts to the
    /// requested output pixel format if one was set.
    pub(super) fn decode_12bit_as_8bit(
        &self,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        // Defense in depth: decode_image_inner rejects Cmyk-for-non-CMYK
        // before dispatching here (P4-68), and a 12-bit source never
        // converts to Cmyk in either arm (codex review on issue #394).
        if self.output_format == Some(PixelFormat::Cmyk) {
            return Err(JpegError::Unsupported(
                "cannot convert 12-bit JPEG to Cmyk".to_string(),
            ));
        }
        let img12 = crate::api::precision::decompress_12bit(self.raw_data)?;
        let num_components: usize = img12.num_components;

        // Determine output format: default to Grayscale for 1-component,
        // RGB for 3-component, same as the 8-bit path.
        let default_format: PixelFormat = if num_components == 1 {
            PixelFormat::Grayscale
        } else {
            PixelFormat::Rgb
        };
        let out_format: PixelFormat = self.output_format.unwrap_or(default_format);

        // Scale 12-bit i16 samples to 8-bit u8: val * 255 / 4095.
        // This matches C djpeg's 12-to-8 bit downscaling.
        let width: usize = img12.width;
        let height: usize = img12.height;

        if num_components == 1 {
            // Scale to 8-bit gray first, then honour the requested output
            // format with the same expansion family as the 8-bit grayscale
            // path — this arm used to hard-code Grayscale and silently
            // ignore `output_format` while `output_buffer_size()`
            // advertised the requested bpp (issue #394 / P4-65; C's
            // `djpeg -rgb` expands instead of ignoring).
            // Scale each 12-bit sample inline (v * 255 / 4095, matching C
            // djpeg's 12->8 downscale) straight into the output buffer:
            // no intermediate 8-bit plane, so peak memory stays within
            // what check_header_limits estimated (codex P1 on #394 —
            // a staging plane pushed RGBA to 7 bytes/pixel against a
            // 5 bytes/pixel estimate under max_memory).
            let scale = |v: i16| (v.clamp(0, 4095) as u32 * 255 / 4095) as u8;
            let pad_off: Option<usize> = pad_alpha_offset(out_format);
            let bpp: usize = out_format.bytes_per_pixel();
            let data: Vec<u8> = match out_format {
                PixelFormat::Grayscale => img12.data.iter().map(|&v| scale(v)).collect(),
                PixelFormat::Rgb | PixelFormat::Bgr => {
                    let mut out: Vec<u8> = Vec::with_capacity(width * height * bpp);
                    for &v12 in &img12.data {
                        let v: u8 = scale(v12);
                        out.extend_from_slice(&[v, v, v]);
                    }
                    out
                }
                PixelFormat::Rgba
                | PixelFormat::Bgra
                | PixelFormat::Rgbx
                | PixelFormat::Bgrx
                | PixelFormat::Xrgb
                | PixelFormat::Xbgr
                | PixelFormat::Argb
                | PixelFormat::Abgr => {
                    let pad: usize = pad_off.expect("4bpp format has a pad offset");
                    let mut out: Vec<u8> = Vec::with_capacity(width * height * bpp);
                    for &v12 in &img12.data {
                        let mut px: [u8; 4] = [scale(v12); 4];
                        px[pad] = 255;
                        out.extend_from_slice(&px);
                    }
                    out
                }
                PixelFormat::Rgb565 => {
                    if self.dither_565 {
                        // Same row-aware ordered dither as the 8-bit
                        // grayscale path (codex review on issue #394:
                        // set_dither_565 must not be silently ignored).
                        // One row of 8-bit scratch, not a full plane.
                        let mut out: Vec<u8> = vec![0u8; width * height * 2];
                        let mut row8: Vec<u8> = vec![0u8; width];
                        for y in 0..height {
                            for (dst, &v12) in
                                row8.iter_mut().zip(&img12.data[y * width..(y + 1) * width])
                            {
                                *dst = scale(v12);
                            }
                            crate::decode::color::gray_to_rgb565_dithered_row(
                                &row8,
                                &mut out[y * width * 2..(y + 1) * width * 2],
                                width,
                                y,
                            );
                        }
                        out
                    } else {
                        let mut out: Vec<u8> = Vec::with_capacity(width * height * 2);
                        for &v12 in &img12.data {
                            let v: u8 = scale(v12);
                            let packed: u16 = (((v as u16) >> 3) << 11)
                                | (((v as u16) >> 2) << 5)
                                | ((v as u16) >> 3);
                            out.extend_from_slice(&packed.to_ne_bytes());
                        }
                        out
                    }
                }
                PixelFormat::Cmyk => unreachable!("rejected above"),
            };
            Ok(Image {
                xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                width,
                height,
                pixel_format: out_format,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                density: self.metadata.density,
                saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                warnings: Vec::new(),
            })
        } else {
            // Color image: img12.data is interleaved RGB (3 values per pixel).
            // Convert to the requested output format.
            let bpp: usize = out_format.bytes_per_pixel();
            let mut data: Vec<u8> = vec![0u8; width * height * bpp];

            let r_off: Option<usize> = out_format.red_offset();
            let g_off: Option<usize> = out_format.green_offset();
            let b_off: Option<usize> = out_format.blue_offset();

            for i in 0..(width * height) {
                let src_idx: usize = i * 3;
                let r: u8 = (img12.data[src_idx].clamp(0, 4095) as u32 * 255 / 4095) as u8;
                let g: u8 = (img12.data[src_idx + 1].clamp(0, 4095) as u32 * 255 / 4095) as u8;
                let b: u8 = (img12.data[src_idx + 2].clamp(0, 4095) as u32 * 255 / 4095) as u8;
                let dst_idx: usize = i * bpp;

                match out_format {
                    PixelFormat::Rgb => {
                        data[dst_idx] = r;
                        data[dst_idx + 1] = g;
                        data[dst_idx + 2] = b;
                    }
                    PixelFormat::Grayscale => {
                        // Approximate luminance from RGB.
                        data[dst_idx] =
                            ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8) as u8;
                    }
                    _ => {
                        // Use offset-based mapping for all other RGB-derived formats.
                        if let (Some(ro), Some(go), Some(bo)) = (r_off, g_off, b_off) {
                            data[dst_idx + ro] = r;
                            data[dst_idx + go] = g;
                            data[dst_idx + bo] = b;
                            // Fill alpha/padding byte to 0xFF for 4-bpp formats.
                            if bpp == 4 {
                                let alpha_off: usize = 6 - ro - go - bo;
                                data[dst_idx + alpha_off] = 0xFF;
                            }
                        } else {
                            return Err(JpegError::Unsupported(format!(
                                "cannot convert 12-bit color JPEG to {:?}",
                                out_format
                            )));
                        }
                    }
                }
            }
            Ok(Image {
                xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                width,
                height,
                pixel_format: out_format,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                density: self.metadata.density,
                saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                warnings: Vec::new(),
            })
        }
    }

    /// `sink`: caller-provided output buffer for `decode_image_into`.
    /// Branches with native support claim it via `take_out_buf` (which
    /// leaves it `None`); branches without leave it untouched and the
    /// wrapper stages through the returned owned `data` instead.
    // The split changed LTO code generation enough to cost about 2% on the
    // measured x86_64 SIMD path. Keep this single-caller body in its wrapper
    // there; unmeasured targets retain the compiler's own inlining choice.
    #[cfg_attr(all(target_arch = "x86_64", feature = "simd"), inline(always))]
    fn decode_image_inner(&self, sink: &mut Option<&mut [u8]>) -> Result<Image> {
        let frame = &self.metadata.frame;
        let width = frame.width as usize;
        let height = frame.height as usize;

        // Dimension / pixel / scan-count limits (issue #355).
        self.check_header_limits()?;

        // Cmyk output exists only for CMYK/YCCK sources; C raises
        // JERR_CONVERSION_NOTIMPL for everything else (jdcolor.c). The
        // Rust paths used to panic on this well-formed request (P4-68:
        // three separate unreachable!/match panics for baseline colour,
        // baseline grayscale, and lossless grayscale sources).
        if self.output_format == Some(PixelFormat::Cmyk)
            && !matches!(
                self.detect_color_space(),
                ColorSpace::Cmyk | ColorSpace::Ycck
            )
        {
            return Err(JpegError::Unsupported(format!(
                "cannot convert {:?} JPEG to Cmyk (C: JERR_CONVERSION_NOTIMPL)",
                self.detect_color_space()
            )));
        }

        // P4-144: all four are input-sized allocations that used to abort the
        // process when the allocator refused. This function already returns
        // `Result`, so making them fallible costs a helper call rather than the
        // API churn the item anticipated.
        let icc_profile = self.icc_profile()?;
        let exif_data = try_clone_opt(&self.metadata.exif_data, "EXIF metadata")?;
        let xmp_data = try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?;
        let iptc_data = try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?;

        // Handle 12-bit JPEG transparently: decode via the 12-bit path, then
        // scale samples from 0-4095 to 0-255 so callers get standard 8-bit output.
        // This matches C djpeg behavior which handles 12-bit JPEGs automatically.
        if frame.precision == 12 {
            return self.decode_12bit_as_8bit(icc_profile, exif_data);
        }

        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported)",
                frame.precision
            )));
        }

        let num_components = frame.components.len();
        let max_h = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);

        let block_size = self.scale.block_size();
        // Per-component IDCT block sizes: chroma components may use a larger
        // IDCT to absorb subsampling factors (matches C libjpeg-turbo).
        let comp_block_sizes: [usize; 4] =
            Self::compute_all_comp_block_sizes(block_size, max_h, max_v, frame);
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let mcus_x = width.div_ceil(mcu_width);
        let mcus_y = height.div_ceil(mcu_height);
        // Scaled output dimensions
        let scaled_mcu_w = max_h * block_size;
        let scaled_mcu_h = max_v * block_size;
        let full_width = mcus_x * scaled_mcu_w;
        let full_height = mcus_y * scaled_mcu_h;
        // Final output dimensions (may be smaller than full due to MCU alignment)
        let out_width = self.scale.scale_dim(width);
        let out_height = self.scale.scale_dim(height);

        // Lossless JPEG (SOF3/SOF11) — different pipeline, no IDCT/quant
        if frame.is_lossless {
            return self.decode_lossless_image(frame, width, height, icc_profile, exif_data);
        }

        // Pre-resolve quant tables per component (once, not per-block).
        // Fixed-size backing array (≤ 4 components) — the slice passed
        // downstream is length `num_components`, so the tail padding with
        // the first table is never read.
        let mut quant_table_refs: [Option<&QuantTable>; 4] = [None; 4];
        for (slot, comp) in quant_table_refs.iter_mut().zip(frame.components.iter()) {
            *slot = Some(
                self.metadata.quant_tables[comp.quant_table_index as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "missing quant table {}",
                            comp.quant_table_index
                        ))
                    })?,
            );
        }
        let first_quant: &QuantTable = quant_table_refs[0].ok_or_else(|| {
            // Unreachable for streams parsed by read_sof (component count
            // validated 1..=MAX_COMPONENTS), but the input is
            // attacker-controlled — error, don't panic.
            JpegError::CorruptData("frame has no components".into())
        })?;
        let quant_table_arr: [&QuantTable; 4] =
            quant_table_refs.map(|slot| slot.unwrap_or(first_quant));
        let quant_tables: &[&QuantTable] = &quant_table_arr[..num_components];

        // Decode component planes — different paths for baseline vs progressive vs arithmetic
        let (mut component_planes, warnings) =
            if self.metadata.is_arithmetic && frame.is_progressive {
                self.decode_arithmetic_progressive_planes(
                    frame,
                    quant_tables,
                    num_components,
                    mcus_x,
                    mcus_y,
                    max_h,
                    max_v,
                    &comp_block_sizes,
                )?
            } else if self.metadata.is_arithmetic {
                self.decode_arithmetic_planes(
                    frame,
                    quant_tables,
                    num_components,
                    mcus_x,
                    mcus_y,
                    &comp_block_sizes,
                )?
            } else if frame.is_progressive {
                self.decode_progressive_planes(
                    frame,
                    quant_tables,
                    num_components,
                    mcus_x,
                    mcus_y,
                    max_h,
                    max_v,
                    &comp_block_sizes,
                    self.block_smoothing,
                )?
            } else {
                self.decode_baseline_planes(
                    frame,
                    quant_tables,
                    num_components,
                    mcus_x,
                    mcus_y,
                    &comp_block_sizes,
                )?
            };

        // Crop-aware output: when crop_x/crop_width are set, the output
        // narrows to the crop width. Matches C jpeg_crop_scanline behavior:
        // X is aligned down to iMCU boundary, width is expanded accordingly.
        // Crop coordinates are in the original image space; align then scale
        // to output space so they index correctly into scaled component planes.
        // Crop coordinates are in the output (post-scale) space, matching C
        // djpeg -crop behavior. Align X down to the scaled iMCU boundary and
        // expand width to compensate.
        let scaled_imcu_w: usize = max_h * block_size; // iMCU width in scaled output pixels
        let (scaled_crop_x, scaled_crop_w): (Option<usize>, Option<usize>) =
            if let (Some(cx), Some(cw)) = (self.crop_x, self.crop_width) {
                // Align X down to scaled iMCU boundary, expand width
                let aligned_x: usize = (cx / scaled_imcu_w) * scaled_imcu_w;
                let expanded_w: usize = cw + (cx - aligned_x);
                let clamped_w: usize = expanded_w.min(out_width.saturating_sub(aligned_x));
                (Some(aligned_x), Some(clamped_w))
            } else {
                (None, None)
            };
        // Preserve the uncropped scaled width for filters that need context
        // outside the requested crop (notably fancy direct-RGB upsampling).
        let uncropped_out_width: usize = out_width;
        // Apply horizontal crop to out_width so all downstream paths use it.
        let out_width: usize = if let Some(cw) = scaled_crop_w {
            let cx: usize = scaled_crop_x.unwrap_or(0).min(out_width);
            cw.min(out_width.saturating_sub(cx))
        } else {
            out_width
        };

        // Per-component X offsets for horizontal crop.
        // scaled_crop_x is an offset into the full-width component planes,
        // NOT clamped to out_width (which is the crop width).
        let mut comp_x_offsets: [usize; 4] = [0; 4];
        if let Some(cx) = scaled_crop_x {
            for (off, (ci, comp)) in comp_x_offsets
                .iter_mut()
                .zip(frame.components.iter().enumerate())
            {
                let comp_w: usize =
                    mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                // Scaled IDCT can absorb subsampling by using a larger block
                // size for chroma components, so derive the crop offset from
                // the decoded plane width rather than sampling factors alone.
                *off = (cx * comp_w / full_width).min(comp_w.saturating_sub(1));
            }
        }

        // Handle output colorspace override (with crop offsets applied)
        if let Some(cs) = self.effective_output_colorspace() {
            if cs == ColorSpace::Grayscale {
                return self.decode_grayscale_override(
                    &component_planes,
                    frame,
                    out_width,
                    out_height,
                    uncropped_out_width,
                    full_width,
                    full_height,
                    mcus_x,
                    mcus_y,
                    block_size,
                    &comp_block_sizes,
                    scaled_crop_x,
                    icc_profile,
                    exif_data,
                    warnings,
                );
            }
            if cs != ColorSpace::YCbCr {
                return Err(JpegError::Unsupported(format!(
                    "output colorspace {cs:?} not supported"
                )));
            }
            // No crop-X shift needed: hand the planes over as-is. The
            // clone below would double plane memory and blow through a
            // `set_max_memory` cap the limit check just approved
            // (codex P1 on #386) — only pay it when an offset exists.
            if comp_x_offsets.iter().all(|&off| off == 0) {
                return crate::decode::toggles::decode_with_colorspace_override(
                    ColorSpace::YCbCr,
                    &component_planes,
                    frame,
                    out_width,
                    out_height,
                    mcus_x,
                    &comp_block_sizes,
                    icc_profile,
                    exif_data,
                    try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                    try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                    try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                    self.metadata.density,
                    try_clone_saved_markers(&self.metadata.saved_markers)?,
                    warnings,
                );
            }
            // Shift component planes by the crop X offset so the override
            // function reads from the correct horizontal position.
            let cropped_planes: Vec<Vec<u8>> = component_planes
                .iter()
                .enumerate()
                .map(|(ci, plane)| {
                    let comp_w: usize = mcus_x
                        * frame.components[ci].horizontal_sampling as usize
                        * comp_block_sizes[ci];
                    let comp_h: usize = mcus_y
                        * frame.components[ci].vertical_sampling as usize
                        * comp_block_sizes[ci];
                    let off: usize = comp_x_offsets[ci];
                    if off == 0 {
                        return plane.clone();
                    }
                    // Re-pack rows shifted by `off` pixels
                    let mut shifted: Vec<u8> = Vec::with_capacity(plane.len());
                    for row in 0..comp_h {
                        shifted.extend_from_slice(&plane[row * comp_w + off..(row + 1) * comp_w]);
                        // Pad to maintain stride
                        shifted.extend(core::iter::repeat_n(0, off));
                    }
                    shifted
                })
                .collect();
            return crate::decode::toggles::decode_with_colorspace_override(
                ColorSpace::YCbCr,
                &cropped_planes,
                frame,
                out_width,
                out_height,
                mcus_x,
                &comp_block_sizes,
                icc_profile,
                exif_data,
                try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                self.metadata.density,
                try_clone_saved_markers(&self.metadata.saved_markers)?,
                warnings,
            );
        }
        // Cap output height to the extended MCU range when vertical crop is set.
        // IDCT is skipped outside this range, so component planes contain
        // uninitialized data beyond it. The upsampler needs this cap to produce
        // correct edge behavior. decode_image() then slices relative to this
        // capped range using crop_y offset from the MCU-range start.
        let out_height: usize = if let (Some(cy), Some(ch)) = (self.crop_y, self.crop_height) {
            let mcu_h: usize = max_v * block_size;
            let mcu_end: usize = (cy + ch).div_ceil(mcu_h);
            let extended_end: usize = (mcu_end + 1).min(mcus_y);
            (extended_end * mcu_h).min(out_height)
        } else {
            out_height
        };

        // Upsample and color convert
        if num_components == 1 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Grayscale);
            let comp_w =
                mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];

            if out_format == PixelFormat::Grayscale {
                let off: usize = comp_x_offsets[0];
                // When the decoded plane already has exactly the output
                // geometry (MCU-aligned dimensions, no crop/scale slack),
                // hand it over instead of row-copying into a fresh buffer
                // (owned mode) or bulk-copy it (caller-buffer mode).
                let exact_geometry: bool = off == 0
                    && comp_w == out_width
                    && component_planes[0].len() == comp_w * out_height;
                let data: Vec<u8> = if sink.is_some() {
                    let mut buf = take_out_buf(sink, out_width * out_height)?;
                    if exact_geometry {
                        buf.copy_from_slice(&component_planes[0]);
                    } else {
                        for y in 0..out_height {
                            buf[y * out_width..(y + 1) * out_width].copy_from_slice(
                                &component_planes[0]
                                    [y * comp_w + off..y * comp_w + off + out_width],
                            );
                        }
                    }
                    buf.into_vec()
                } else if exact_geometry {
                    core::mem::take(&mut component_planes[0])
                } else {
                    let mut data = Vec::with_capacity(out_width * out_height);
                    for y in 0..out_height {
                        data.extend_from_slice(
                            &component_planes[0][y * comp_w + off..y * comp_w + off + out_width],
                        );
                    }
                    data
                };
                Ok(Image {
                    xmp_data: try_clone_opt(&xmp_data, "XMP metadata")?,
                    iptc_data: try_clone_opt(&iptc_data, "IPTC metadata")?,
                    width: out_width,
                    height: out_height,
                    pixel_format: PixelFormat::Grayscale,
                    precision: 8,
                    data,
                    icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                    exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                    comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                    density: self.metadata.density,
                    saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                    warnings: warnings.clone(),
                })
            } else {
                // Expand grayscale to requested color format
                let bpp = out_format.bytes_per_pixel();
                let data_size = out_width * out_height * bpp;
                let pad_off: Option<usize> = pad_alpha_offset(out_format);
                let mut data = take_out_buf(sink, data_size)?;
                for y in 0..out_height {
                    let row = &component_planes[0][y * comp_w + comp_x_offsets[0]
                        ..y * comp_w + comp_x_offsets[0] + out_width];
                    let out_row = &mut data[y * out_width * bpp..(y + 1) * out_width * bpp];
                    // For dithered RGB565, use the dedicated row-level function.
                    if out_format == PixelFormat::Rgb565 && self.dither_565 {
                        crate::decode::color::gray_to_rgb565_dithered_row(
                            row, out_row, out_width, y,
                        );
                        continue;
                    }
                    for x in 0..out_width {
                        let v = row[x];
                        match out_format {
                            PixelFormat::Rgb | PixelFormat::Bgr => {
                                out_row[x * 3] = v;
                                out_row[x * 3 + 1] = v;
                                out_row[x * 3 + 2] = v;
                            }
                            // Offset-derived like the CMYK/YCCK arms: the
                            // pad/alpha byte position comes from the format,
                            // matching the 3-component colour path and C
                            // JCS_EXT_* (issue #369 had Argb/Abgr grouped
                            // alpha-last).
                            PixelFormat::Rgba
                            | PixelFormat::Bgra
                            | PixelFormat::Rgbx
                            | PixelFormat::Bgrx
                            | PixelFormat::Xrgb
                            | PixelFormat::Xbgr
                            | PixelFormat::Argb
                            | PixelFormat::Abgr => {
                                let mut px: [u8; 4] = [v; 4];
                                px[pad_off.expect("4bpp format has a pad offset")] = 255;
                                out_row[x * 4..x * 4 + 4].copy_from_slice(&px);
                            }
                            PixelFormat::Rgb565 => {
                                // Grayscale v → pack as R=G=B=v (no dither)
                                let packed: u16 = ((v as u16 >> 3) << 11)
                                    | ((v as u16 >> 2) << 5)
                                    | (v as u16 >> 3);
                                let bytes: [u8; 2] = packed.to_ne_bytes();
                                out_row[x * 2] = bytes[0];
                                out_row[x * 2 + 1] = bytes[1];
                            }
                            PixelFormat::Grayscale | PixelFormat::Cmyk => unreachable!(),
                        }
                    }
                }
                Ok(Image {
                    xmp_data: try_clone_opt(&xmp_data, "XMP metadata")?,
                    iptc_data: try_clone_opt(&iptc_data, "IPTC metadata")?,
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    precision: 8,
                    data: data.into_vec(),
                    icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                    exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                    comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                    density: self.metadata.density,
                    saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                    warnings: warnings.clone(),
                })
            }
        } else if num_components == 3 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Rgb);
            let jpeg_color_space: ColorSpace = self.detect_color_space();

            // For RGB-colorspace JPEGs (e.g., cjpeg -rgb with Adobe APP14
            // transform=0), component planes store raw R,G,B — no YCbCr→RGB
            // conversion needed.  Interleave planes directly.
            if jpeg_color_space == ColorSpace::Rgb {
                if out_format == PixelFormat::Cmyk {
                    return Err(JpegError::Unsupported(
                        "cannot convert direct-RGB JPEG to CMYK".to_string(),
                    ));
                }
                // Direct-RGB JPEGs can still use unequal component sampling
                // (`cjpeg -rgb -sample HxV`).  Upsample the lower-resolution
                // component planes before interleaving them; treating G/B as
                // full-size rows either truncates the planes or reads across
                // row boundaries for every non-1x1 sampling mode.
                let mut rgb_full: Vec<Cow<'_, [u8]>> = Vec::with_capacity(3);
                let mut rgb_strides: [usize; 3] = [0; 3];
                for ci in 0..3 {
                    let (full_plane, stride) = self.upsample_component_plane(
                        &component_planes[ci],
                        ci,
                        frame,
                        mcus_x,
                        mcus_y,
                        &comp_block_sizes,
                        full_width,
                        full_height,
                        uncropped_out_width,
                        out_height,
                        block_size,
                    )?;
                    rgb_full.push(full_plane);
                    rgb_strides[ci] = stride;
                }

                let bytes_per_pixel: usize = out_format.bytes_per_pixel();
                let data_size: usize = out_width * out_height * bytes_per_pixel;
                let mut data = take_out_buf(sink, data_size)?;
                let output_x_offset: usize = scaled_crop_x.unwrap_or(0);
                for y in 0..out_height {
                    let r_plane: &[u8] = &rgb_full[0];
                    let g_plane: &[u8] = &rgb_full[1];
                    let b_plane: &[u8] = &rgb_full[2];
                    let r_row: &[u8] = &r_plane[y * rgb_strides[0]..];
                    let g_row: &[u8] = &g_plane[y * rgb_strides[1]..];
                    let b_row: &[u8] = &b_plane[y * rgb_strides[2]..];
                    let out_row: &mut [u8] =
                        &mut data[y * out_width * bytes_per_pixel..][..out_width * bytes_per_pixel];
                    for x in 0..out_width {
                        let source_x: usize = output_x_offset + x;
                        let (r, g, b): (u8, u8, u8) =
                            (r_row[source_x], g_row[source_x], b_row[source_x]);
                        match out_format {
                            PixelFormat::Rgb => {
                                out_row[x * 3] = r;
                                out_row[x * 3 + 1] = g;
                                out_row[x * 3 + 2] = b;
                            }
                            PixelFormat::Bgr => {
                                out_row[x * 3] = b;
                                out_row[x * 3 + 1] = g;
                                out_row[x * 3 + 2] = r;
                            }
                            PixelFormat::Rgba | PixelFormat::Rgbx => {
                                out_row[x * 4] = r;
                                out_row[x * 4 + 1] = g;
                                out_row[x * 4 + 2] = b;
                                out_row[x * 4 + 3] = 255;
                            }
                            PixelFormat::Bgra | PixelFormat::Bgrx => {
                                out_row[x * 4] = b;
                                out_row[x * 4 + 1] = g;
                                out_row[x * 4 + 2] = r;
                                out_row[x * 4 + 3] = 255;
                            }
                            PixelFormat::Xrgb | PixelFormat::Argb => {
                                out_row[x * 4] = 255;
                                out_row[x * 4 + 1] = r;
                                out_row[x * 4 + 2] = g;
                                out_row[x * 4 + 3] = b;
                            }
                            PixelFormat::Xbgr | PixelFormat::Abgr => {
                                out_row[x * 4] = 255;
                                out_row[x * 4 + 1] = b;
                                out_row[x * 4 + 2] = g;
                                out_row[x * 4 + 3] = r;
                            }
                            PixelFormat::Rgb565 => {
                                let packed: u16 = ((r as u16 >> 3) << 11)
                                    | ((g as u16 >> 2) << 5)
                                    | (b as u16 >> 3);
                                let bytes: [u8; 2] = packed.to_ne_bytes();
                                out_row[x * 2] = bytes[0];
                                out_row[x * 2 + 1] = bytes[1];
                            }
                            PixelFormat::Grayscale | PixelFormat::Cmyk => unreachable!(),
                        }
                    }
                }

                return Ok(Image {
                    xmp_data: try_clone_opt(&xmp_data, "XMP metadata")?,
                    iptc_data: try_clone_opt(&iptc_data, "IPTC metadata")?,
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    precision: 8,
                    data: data.into_vec(),
                    icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                    exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                    comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                    density: self.metadata.density,
                    saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                    warnings: warnings.clone(),
                });
            }

            let bpp = out_format.bytes_per_pixel();

            let y_plane = &component_planes[0];
            let y_width =
                mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];

            let cb_comp = &frame.components[1];
            let cr_comp = &frame.components[2];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * comp_block_sizes[1];
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * comp_block_sizes[1];
            let cr_w = mcus_x * cr_comp.horizontal_sampling as usize * comp_block_sizes[2];
            let cr_h = mcus_y * cr_comp.vertical_sampling as usize * comp_block_sizes[2];

            let y_height =
                mcus_y * frame.components[0].vertical_sampling as usize * comp_block_sizes[0];

            // Reject degenerate chroma dimensions. Valid SOF guarantees each
            // sampling factor is 1..=4 and mcus_* >= 1, so cb_w / cb_h etc.
            // are always positive for a well-formed JPEG — zero here means
            // the header's component table is inconsistent with the scan.
            if cb_w == 0 || cb_h == 0 || cr_w == 0 || cr_h == 0 {
                return Err(JpegError::CorruptData(format!(
                    "zero chroma plane dimensions: cb={}x{} cr={}x{}",
                    cb_w, cb_h, cr_w, cr_h
                )));
            }

            // Per-component effective upsample factors.
            // For scaled decode, chroma may use a larger IDCT that absorbs subsampling,
            // making the effective factor 1 (no upsample needed).
            let cb_h_factor: usize = y_width / cb_w;
            let cb_v_factor: usize = y_height / cb_h;
            let cr_h_factor: usize = y_width / cr_w;
            let cr_v_factor: usize = y_height / cr_h;
            // Valid JPEG has chroma <= luma, so every factor >= 1. A zero
            // here means the luma plane is smaller than its chroma plane
            // (e.g. a crafted scan with inverted sampling factors), which
            // would feed a div-by-zero into the `out_*.div_ceil(*_factor)`
            // calls below and into downstream upsample math.
            if cb_h_factor == 0 || cb_v_factor == 0 || cr_h_factor == 0 || cr_v_factor == 0 {
                let detail: String = format!(
                    "chroma upsample factor zero (a chroma component out-samples luma): \
                     cb={}x{} cr={}x{}",
                    cb_h_factor, cb_v_factor, cr_h_factor, cr_v_factor
                );
                if !self.lenient {
                    return Err(JpegError::CorruptData(detail));
                }
                // Lenient recovery (LAST_MILE P4-21): the colour/upsample
                // pipeline assumes luma is the maximally-sampled component, an
                // invariant this spec-valid-but-unusual sampling violates
                // (e.g. Cr=h3v1 while Y=h1v1). djpeg decodes it; rather than
                // reject and be *less* accepting than the reference, emit a
                // best-effort neutral raster plus a warning so the lenient
                // contract holds. Pixel-correct decoding is tracked as P4-21.
                let mut recovered_warnings: Vec<DecodeWarning> = warnings.clone();
                recovered_warnings.push(DecodeWarning::UnsupportedRecovered { detail });
                let data: Vec<u8> = vec![128u8; out_width * out_height * bpp];
                return Ok(Image {
                    xmp_data: try_clone_opt(&xmp_data, "XMP metadata")?,
                    iptc_data: try_clone_opt(&iptc_data, "IPTC metadata")?,
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    precision: 8,
                    data,
                    icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                    exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                    comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                    density: self.metadata.density,
                    saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                    warnings: recovered_warnings,
                });
            }

            // Non-integer luma/chroma ratios (e.g. Y=4x1 with Cb=3x1 → 4/3)
            // truncate to a wrong integer factor above; the copy/upsample
            // paths below would then read luma-sized rows out of a smaller
            // chroma plane (Fuzz Smoke run 29977722126, P4-36). C rejects
            // every fractional ratio up front with
            // ERREXIT(JERR_FRACT_SAMPLE_NOTIMPL) ("Fractional sampling not
            // implemented yet") in jdsample.c — match it in both strict and
            // lenient modes (djpeg fails fatally, so there is nothing more
            // lenient to offer).
            if !y_width.is_multiple_of(cb_w)
                || !y_height.is_multiple_of(cb_h)
                || !y_width.is_multiple_of(cr_w)
                || !y_height.is_multiple_of(cr_h)
            {
                return Err(JpegError::Unsupported(format!(
                    "fractional chroma sampling ratio: luma {}x{} vs cb {}x{} cr {}x{}",
                    y_width, y_height, cb_w, cb_h, cr_w, cr_h
                )));
            }

            // When both chroma components have the same factors, use the shared
            // factor variables that the existing optimized paths expect.
            let uniform_chroma: bool = cb_h_factor == cr_h_factor && cb_v_factor == cr_v_factor;
            let h_factor: usize = cb_h_factor;
            let v_factor: usize = cb_v_factor;

            // Actual chroma dimensions (may be smaller than MCU-aligned cb_w/cb_h).
            // C libjpeg-turbo uses downsampled_width/height for upsample, not
            // MCU-padded dimensions. Using MCU-padded values causes the upsample
            // to interpolate padding data, producing wrong edge pixels.
            let actual_cb_w: usize = out_width.div_ceil(cb_h_factor);
            let actual_cb_h: usize = out_height.div_ceil(cb_v_factor);
            let actual_cr_w: usize = out_width.div_ceil(cr_h_factor);
            let actual_cr_h: usize = out_height.div_ceil(cr_v_factor);

            // For 4:4:4, use component planes directly without clone.
            // For subsampled modes, upsample into separate buffers.
            let (cb_data, cr_data, cb_stride, cr_stride): (&[u8], &[u8], usize, usize);

            if cb_h_factor == 1 && cb_v_factor == 1 && cr_h_factor == 1 && cr_v_factor == 1 {
                // 4:4:4: no upsampling needed — reference planes directly
                cb_data = &component_planes[1];
                cr_data = &component_planes[2];
                cb_stride = cb_w;
                cr_stride = cr_w;
            } else {
                // Merged upsample path: combine upsample + color convert in one pass
                // for H2V1 (4:2:2) and H2V2 (4:2:0), avoiding intermediate chroma buffers.
                // Only available when both chroma components have the same sampling factors.
                // RGB565 routes through this merged branch only when
                // dithering is OFF — upstream has a separate `*_565D`
                // merged path for ordered-dither RGB565, which the shim
                // doesn't yet implement; falling through to the slow
                // path preserves the pre-fix behavior for that combo
                // (ordered dither honored via the non-merged dithered
                // RGB565 writer). Dither + merged is tracked as a
                // Phase 4 perf follow-up.
                let merged_rgb565_ok: bool = out_format == PixelFormat::Rgb565 && !self.dither_565;
                if self.merged_upsample
                    && uniform_chroma
                    && (out_format == PixelFormat::Rgb || merged_rgb565_ok)
                    && h_factor == 2
                    && (v_factor == 1 || v_factor == 2)
                {
                    // The merged kernels produce RGB at bpp=3. RGB565 output
                    // routes through an intermediate RGB buffer + 5-6-5
                    // truncation; this preserves the merged-upsample SIMD
                    // hot path and matches upstream's `_565` jdmerge.c
                    // semantics (truncation only, no dither). The
                    // dedicated `_565` and `_565D` SIMD kernels
                    // (jdmrgext-*-565*) are deferred to a Phase 4 perf
                    // task.
                    let merged_bpp: usize = 3;
                    let merged_size: usize = out_width * out_height * merged_bpp;
                    let mut merged_rgb: Vec<u8> = vec![0u8; merged_size];

                    if v_factor == 1 {
                        // H2V1 (4:2:2): one chroma row per Y row
                        for y in 0..out_height {
                            Self::merged_h2v1(
                                &y_plane[y * y_width + comp_x_offsets[0]..],
                                &component_planes[1][y * cb_w + comp_x_offsets[1]..],
                                &component_planes[2][y * cb_w + comp_x_offsets[2]..],
                                &mut merged_rgb[y * out_width * merged_bpp..],
                                out_width,
                            );
                        }
                    } else {
                        // H2V2 (4:2:0): one chroma row per 2 Y rows
                        let row_pairs: usize = out_height / 2;
                        for pair in 0..row_pairs {
                            let y0: usize = pair * 2;
                            let y1: usize = pair * 2 + 1;
                            let chroma_row: usize = pair;
                            let out0_start: usize = y0 * out_width * merged_bpp;
                            let out1_start: usize = y1 * out_width * merged_bpp;
                            let (top, bottom) = merged_rgb.split_at_mut(out1_start);
                            Self::merged_h2v2(
                                &y_plane[y0 * y_width + comp_x_offsets[0]..],
                                &y_plane[y1 * y_width + comp_x_offsets[0]..],
                                &component_planes[1][chroma_row * cb_w + comp_x_offsets[1]..],
                                &component_planes[2][chroma_row * cb_w + comp_x_offsets[2]..],
                                &mut top[out0_start..],
                                bottom,
                                out_width,
                            );
                        }
                        if out_height & 1 != 0 {
                            let last_y: usize = out_height - 1;
                            let chroma_row: usize = last_y / 2;
                            Self::merged_h2v1(
                                &y_plane[last_y * y_width + comp_x_offsets[0]..],
                                &component_planes[1][chroma_row * cb_w + comp_x_offsets[1]..],
                                &component_planes[2][chroma_row * cb_w + comp_x_offsets[2]..],
                                &mut merged_rgb[last_y * out_width * merged_bpp..],
                                out_width,
                            );
                        }
                    }

                    let data: Vec<u8> = if out_format == PixelFormat::Rgb {
                        merged_rgb
                    } else {
                        // RGB565 little-endian: word = (R5 << 11) | (G6 << 5) | B5,
                        // with 5-6-5 truncation matching upstream.
                        let mut packed: Vec<u8> = vec![0u8; out_width * out_height * bpp];
                        for i in 0..(out_width * out_height) {
                            let r: u16 = merged_rgb[i * 3] as u16;
                            let g: u16 = merged_rgb[i * 3 + 1] as u16;
                            let b: u16 = merged_rgb[i * 3 + 2] as u16;
                            let word: u16 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
                            let bytes = word.to_le_bytes();
                            packed[i * 2] = bytes[0];
                            packed[i * 2 + 1] = bytes[1];
                        }
                        packed
                    };

                    return Ok(Image {
                        xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                        iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                        width: out_width,
                        height: out_height,
                        pixel_format: out_format,
                        precision: 8,
                        data,
                        icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                        exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                        comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                        density: self.metadata.density,
                        saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                        warnings: warnings.clone(),
                    });
                }

                // Row-streaming H2V2: skip full-plane allocation, process 2 rows at a time.
                // When actual_cb_w <= 2, C's merged upsample uses box filter for the
                // entire image (the NEON/SIMD fancy path doesn't kick in). Use box
                // filter (fast_upsample equivalent) to match C exactly.
                // Only available when both chroma components have the same sampling factors.
                if !self.fast_upsample
                    && uniform_chroma
                    && h_factor == 2
                    && v_factor == 2
                    && actual_cb_w > 2
                    && block_size == 8
                {
                    // Row-streaming H2V2: fuse upsample + color convert to avoid
                    // allocating full-size cb_full/cr_full buffers (~4MB for 1080p).
                    // Process 2 output rows at a time, keeping data in L1/L2 cache.
                    let data_size = out_width * out_height * bpp;
                    let mut data = take_out_buf(sink, data_size)?;

                    // Small per-row upsample buffers (2 rows × full_width per component)
                    let mut cb_row_top = vec![0u8; full_width];
                    let mut cb_row_bot = vec![0u8; full_width];
                    let mut cr_row_top = vec![0u8; full_width];
                    let mut cr_row_bot = vec![0u8; full_width];

                    // Use actual chroma dimensions for upsample (not MCU-padded).
                    let cb_off: usize = comp_x_offsets[1];
                    let cr_off: usize = comp_x_offsets[2];
                    for cy in 0..actual_cb_h {
                        let cb_cur = &component_planes[1]
                            [cy * cb_w + cb_off..cy * cb_w + cb_off + actual_cb_w];
                        let cr_cur = &component_planes[2]
                            [cy * cb_w + cr_off..cy * cb_w + cr_off + actual_cb_w];
                        let cb_above = if cy > 0 {
                            &component_planes[1]
                                [(cy - 1) * cb_w + cb_off..(cy - 1) * cb_w + cb_off + actual_cb_w]
                        } else {
                            cb_cur
                        };
                        let cb_below = if cy + 1 < actual_cb_h {
                            &component_planes[1]
                                [(cy + 1) * cb_w + cb_off..(cy + 1) * cb_w + cb_off + actual_cb_w]
                        } else {
                            cb_cur
                        };
                        let cr_above = if cy > 0 {
                            &component_planes[2]
                                [(cy - 1) * cb_w + cr_off..(cy - 1) * cb_w + cr_off + actual_cb_w]
                        } else {
                            cr_cur
                        };
                        let cr_below = if cy + 1 < actual_cb_h {
                            &component_planes[2]
                                [(cy + 1) * cb_w + cr_off..(cy + 1) * cb_w + cr_off + actual_cb_w]
                        } else {
                            cr_cur
                        };

                        // Fused vertical+horizontal upsample for top output row
                        fancy_h2v2_row_dispatch(cb_cur, cb_above, &mut cb_row_top, actual_cb_w);
                        fancy_h2v2_row_dispatch(cr_cur, cr_above, &mut cr_row_top, actual_cb_w);

                        // Fused vertical+horizontal upsample for bottom output row
                        fancy_h2v2_row_dispatch(cb_cur, cb_below, &mut cb_row_bot, actual_cb_w);
                        fancy_h2v2_row_dispatch(cr_cur, cr_below, &mut cr_row_bot, actual_cb_w);

                        // Color convert both output rows immediately
                        let out_y_top = cy * 2;
                        let out_y_bot = cy * 2 + 1;
                        let y_off: usize = comp_x_offsets[0];
                        if out_y_top < out_height {
                            self.color_convert_row(
                                out_format,
                                &y_plane[out_y_top * y_width + y_off..],
                                &cb_row_top,
                                &cr_row_top,
                                &mut data[out_y_top * out_width * bpp..],
                                out_width,
                                out_y_top,
                            );
                        }
                        if out_y_bot < out_height {
                            self.color_convert_row(
                                out_format,
                                &y_plane[out_y_bot * y_width + y_off..],
                                &cb_row_bot,
                                &cr_row_bot,
                                &mut data[out_y_bot * out_width * bpp..],
                                out_width,
                                out_y_bot,
                            );
                        }
                    }

                    return Ok(Image {
                        xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                        iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                        width: out_width,
                        height: out_height,
                        pixel_format: out_format,
                        precision: 8,
                        data: data.into_vec(),
                        icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                        exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                        comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                        density: self.metadata.density,
                        saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                        warnings: warnings.clone(),
                    });
                }

                // Row-streaming H2V1 (4:2:2): one chroma row per output row.
                // Mirrors the H2V2 block above — fuse the fancy horizontal
                // upsample + colour convert per row so the two
                // full-resolution cb_full/cr_full planes (~4.2 MB extra at
                // 1080p) are never materialised (issue #350). Same gate
                // shape as H2V2: the non-streamed conditions
                // (fast_upsample, actual_cb_w <= 2, scaled IDCT) fall
                // through to the generic path, which reproduces C's
                // filter selection for each of them (jdsample.c:478/506
                // — box filter only for !do_fancy_upsampling,
                // downsampled_width <= 2 on 2h1v, or
                // _min_DCT_scaled_size == 1).
                if !self.fast_upsample
                    && uniform_chroma
                    && h_factor == 2
                    && v_factor == 1
                    && actual_cb_w > 2
                    && block_size == 8
                {
                    let data_size: usize = out_width * out_height * bpp;
                    let mut data = take_out_buf(sink, data_size)?;

                    // Per-row upsample scratch (full_width per component).
                    let mut cb_row: Vec<u8> = vec![0u8; full_width];
                    let mut cr_row: Vec<u8> = vec![0u8; full_width];

                    let cb_off: usize = comp_x_offsets[1];
                    let cr_off: usize = comp_x_offsets[2];
                    let y_off: usize = comp_x_offsets[0];
                    // Exact-length row slices are safe: cb_off ==
                    // aligned_x / h_factor exactly (aligned_x is a
                    // multiple of max_h*block_size and cb_w*h_factor ==
                    // full_width under the block_size==8 gate) and
                    // out_width <= pre-crop out_width - aligned_x, so
                    // off + actual_cb_w <= cb_w for every crop. The
                    // upsample kernels read exactly in_width samples
                    // (audited: scalar/NEON/SSE2/AVX2/WASM stop at
                    // in_width-1). Cr shares cb_w: uniform_chroma pins
                    // identical sampling factors.
                    debug_assert_eq!(cb_w, cr_w, "uniform_chroma implies equal plane strides");
                    for y in 0..out_height {
                        let cb_src = &component_planes[1]
                            [y * cb_w + cb_off..y * cb_w + cb_off + actual_cb_w];
                        let cr_src = &component_planes[2]
                            [y * cb_w + cr_off..y * cb_w + cr_off + actual_cb_w];
                        self.fancy_upsample_h2v1(cb_src, actual_cb_w, &mut cb_row);
                        self.fancy_upsample_h2v1(cr_src, actual_cb_w, &mut cr_row);
                        self.color_convert_row(
                            out_format,
                            &y_plane[y * y_width + y_off..],
                            &cb_row,
                            &cr_row,
                            &mut data[y * out_width * bpp..],
                            out_width,
                            y,
                        );
                    }

                    return Ok(Image {
                        xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                        iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                        width: out_width,
                        height: out_height,
                        pixel_format: out_format,
                        precision: 8,
                        data: data.into_vec(),
                        icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                        exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                        comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                        density: self.metadata.density,
                        saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                        warnings: warnings.clone(),
                    });
                }

                // Row-streaming H1V2 (4:4:0): one chroma row per two output
                // rows, vertical-only triangle filter (chroma width already
                // matches the output). Same structural fix as H2V1 above.
                // The blend biases (top +1, bottom +2) match C jdsample.c
                // h1v2_fancy_upsample and the whole-plane fancy_h1v2 helper.
                if !self.fast_upsample
                    && uniform_chroma
                    && h_factor == 1
                    && v_factor == 2
                    && block_size == 8
                {
                    let data_size: usize = out_width * out_height * bpp;
                    let mut data = take_out_buf(sink, data_size)?;

                    let mut cb_row: Vec<u8> = vec![0u8; full_width];
                    let mut cr_row: Vec<u8> = vec![0u8; full_width];

                    let cb_off: usize = comp_x_offsets[1];
                    let cr_off: usize = comp_x_offsets[2];
                    let y_off: usize = comp_x_offsets[0];
                    // Exact-length row slices are safe: cb_off ==
                    // aligned_x / h_factor exactly (aligned_x is a
                    // multiple of max_h*block_size and cb_w*h_factor ==
                    // full_width under the block_size==8 gate) and
                    // out_width <= pre-crop out_width - aligned_x, so
                    // off + actual_cb_w <= cb_w for every crop. Cr shares
                    // cb_w: uniform_chroma pins identical sampling factors.
                    debug_assert_eq!(cb_w, cr_w, "uniform_chroma implies equal plane strides");
                    let row_range = |row: usize, off: usize| -> core::ops::Range<usize> {
                        let start = row * cb_w + off;
                        start..start + actual_cb_w
                    };
                    // Loop bound: actual_cb_h >= 1 whenever the loop body
                    // runs (out_height >= 1), so last_cy cannot underflow
                    // into a wrap.
                    let last_cy: usize = actual_cb_h.saturating_sub(1);
                    for cy in 0..actual_cb_h {
                        let above_idx: usize = cy.saturating_sub(1);
                        let below_idx: usize = (cy + 1).min(last_cy);
                        let cb_cur = &component_planes[1][row_range(cy, cb_off)];
                        let cr_cur = &component_planes[2][row_range(cy, cr_off)];
                        for (out_y, adj_idx, bias) in
                            [(cy * 2, above_idx, 1u16), (cy * 2 + 1, below_idx, 2u16)]
                        {
                            if out_y >= out_height {
                                continue;
                            }
                            let cb_adj = &component_planes[1][row_range(adj_idx, cb_off)];
                            let cr_adj = &component_planes[2][row_range(adj_idx, cr_off)];
                            Self::fancy_h1v2_row(cb_cur, cb_adj, &mut cb_row, actual_cb_w, bias);
                            Self::fancy_h1v2_row(cr_cur, cr_adj, &mut cr_row, actual_cb_w, bias);
                            self.color_convert_row(
                                out_format,
                                &y_plane[out_y * y_width + y_off..],
                                &cb_row,
                                &cr_row,
                                &mut data[out_y * out_width * bpp..],
                                out_width,
                                out_y,
                            );
                        }
                    }

                    return Ok(Image {
                        xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                        iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                        width: out_width,
                        height: out_height,
                        pixel_format: out_format,
                        precision: 8,
                        data: data.into_vec(),
                        icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                        exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                        comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                        density: self.metadata.density,
                        saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                        warnings: warnings.clone(),
                    });
                }

                // Generic row-streaming for nearest/box-filtered chroma
                // (C's int_upsample): covers fast_upsample (djpeg
                // -nosmooth) in every mode, 4:1:1 (H4V1), 4:4:1 (H1V4),
                // and non-uniform factor combinations — any case where
                // BOTH chroma components resolve to replication rather
                // than a fancy filter (issue #353). Only fancy-filter
                // modes (streamed above when uniform, full-plane below
                // when mixed) and scaled IDCT still materialise
                // full-resolution chroma.
                //
                // Filter selection mirrors the full-plane path exactly:
                // a component is nearest-eligible when it is 1:1, or
                // `use_box_filter` would fire for it, or its factors
                // have no fancy kernel ((2,1)/(2,2)/(1,2) are the only
                // fancy shapes, matching C jdsample.c).
                // block_size == 1 always selects box in the full-plane
                // path, but the gate below is 8-only so it cannot reach
                // here; scaled-IDCT streaming is deferred to #353 step 2.
                let nearest_eligible = |hf: usize, vf: usize, actual_w: usize| -> bool {
                    let use_box: bool = self.fast_upsample || (actual_w <= 2 && hf >= 2);
                    (hf == 1 && vf == 1) || use_box || !matches!((hf, vf), (2, 1) | (2, 2) | (1, 2))
                };
                if block_size == 8
                    && nearest_eligible(cb_h_factor, cb_v_factor, actual_cb_w)
                    && nearest_eligible(cr_h_factor, cr_v_factor, actual_cr_w)
                {
                    let data_size: usize = out_width * out_height * bpp;
                    let mut data = take_out_buf(sink, data_size)?;

                    let mut cb_row: Vec<u8> = vec![0u8; full_width];
                    let mut cr_row: Vec<u8> = vec![0u8; full_width];

                    let y_off: usize = comp_x_offsets[0];
                    // Exact-length slices are safe for every admitted
                    // (hf, vf): comp_w * hf == full_width exactly (luma
                    // is max-sampled, so full_width is a multiple of
                    // hf), hence actual_w * hf <= full_width (no scratch
                    // overrun) and comp_x_offsets[i] == aligned_crop_x /
                    // hf exactly, giving off + actual_w <= comp_w for
                    // every crop.
                    //
                    // Expand one chroma source row by horizontal
                    // replication (dst x -> src x/hf), the exact
                    // per-row slice of `upsample_nearest`'s mapping.
                    let expand_row = |plane: &[u8],
                                      stride: usize,
                                      off: usize,
                                      src_y: usize,
                                      actual_w: usize,
                                      hf: usize,
                                      out: &mut [u8]| {
                        let src = &plane[src_y * stride + off..src_y * stride + off + actual_w];
                        if hf == 1 {
                            out[..actual_w].copy_from_slice(src);
                        } else {
                            for (sx, &v) in src.iter().enumerate() {
                                let start = sx * hf;
                                out[start..start + hf].fill(v);
                            }
                        }
                    };

                    // Track the last expanded source row per component so
                    // vf > 1 rows reuse the scratch instead of re-expanding.
                    let mut last_cb_src: usize = usize::MAX;
                    let mut last_cr_src: usize = usize::MAX;
                    for y in 0..out_height {
                        let cb_src_y: usize = y / cb_v_factor;
                        let cr_src_y: usize = y / cr_v_factor;
                        // actual_h == ceil(out_height / vf), so the map
                        // stays inside the decoded rows; the full-plane
                        // path has no clamp either.
                        debug_assert!(cb_src_y < actual_cb_h && cr_src_y < actual_cr_h);
                        if cb_src_y != last_cb_src {
                            expand_row(
                                &component_planes[1],
                                cb_w,
                                comp_x_offsets[1],
                                cb_src_y,
                                actual_cb_w,
                                cb_h_factor,
                                &mut cb_row,
                            );
                            last_cb_src = cb_src_y;
                        }
                        if cr_src_y != last_cr_src {
                            expand_row(
                                &component_planes[2],
                                cr_w,
                                comp_x_offsets[2],
                                cr_src_y,
                                actual_cr_w,
                                cr_h_factor,
                                &mut cr_row,
                            );
                            last_cr_src = cr_src_y;
                        }
                        self.color_convert_row(
                            out_format,
                            &y_plane[y * y_width + y_off..],
                            &cb_row,
                            &cr_row,
                            &mut data[y * out_width * bpp..],
                            out_width,
                            y,
                        );
                    }

                    return Ok(Image {
                        xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                        iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                        width: out_width,
                        height: out_height,
                        pixel_format: out_format,
                        precision: 8,
                        data: data.into_vec(),
                        icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                        exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                        comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                        density: self.metadata.density,
                        saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                        warnings: warnings.clone(),
                    });
                }

                // All remaining paths need full-plane cb_full/cr_full buffers.
                let alloc_size = full_width * full_height;
                let mut cb_full = vec![0u8; alloc_size];
                let mut cr_full = vec![0u8; alloc_size];

                // Upsample each chroma component independently using its own factors.
                // This handles non-uniform chroma sampling (e.g. Cb=2x1, Cr=1x1)
                // where each component needs a different upsample strategy.
                for (
                    comp_plane,
                    comp_full,
                    comp_w,
                    comp_h,
                    comp_hf,
                    comp_vf,
                    actual_w,
                    actual_h,
                    comp_off,
                ) in [
                    (
                        &component_planes[1],
                        &mut cb_full,
                        cb_w,
                        cb_h,
                        cb_h_factor,
                        cb_v_factor,
                        actual_cb_w,
                        actual_cb_h,
                        comp_x_offsets[1],
                    ),
                    (
                        &component_planes[2],
                        &mut cr_full,
                        cr_w,
                        cr_h,
                        cr_h_factor,
                        cr_v_factor,
                        actual_cr_w,
                        actual_cr_h,
                        comp_x_offsets[2],
                    ),
                ] {
                    // C libjpeg-turbo uses box filter when:
                    // - fast_upsample requested, OR
                    // - actual chroma width <= 2 AND horizontal upsampling is needed
                    //   (fancy horizontal filter needs >= 3 columns; vertical-only
                    //   H1V2 works fine with any width), OR
                    // - min_DCT_scaled_size == 1 (jdsample.c line 478: jdmainct.c
                    //   doesn't support context rows at this size)
                    let use_box_filter: bool =
                        self.fast_upsample || (actual_w <= 2 && comp_hf >= 2) || block_size == 1;

                    if comp_hf == 1 && comp_vf == 1 {
                        // No upsampling needed for this component — copy directly.
                        let copy_len: usize = actual_w.min(full_width);
                        for row in 0..full_height.min(comp_h) {
                            let src_start: usize = row * comp_w + comp_off;
                            let dst_start: usize = row * full_width;
                            comp_full[dst_start..dst_start + copy_len]
                                .copy_from_slice(&comp_plane[src_start..src_start + copy_len]);
                        }
                    } else if use_box_filter {
                        if comp_off > 0 {
                            let mut cropped: Vec<u8> = Vec::with_capacity(actual_w * actual_h);
                            for row in 0..comp_h.min(actual_h) {
                                let s: usize = row * comp_w + comp_off;
                                cropped.extend_from_slice(&comp_plane[s..s + actual_w]);
                            }
                            crate::decode::toggles::upsample_nearest(
                                &cropped, actual_w, actual_h, comp_full, full_width, comp_hf,
                                comp_vf,
                            );
                        } else {
                            crate::decode::toggles::upsample_nearest(
                                comp_plane, comp_w, comp_h, comp_full, full_width, comp_hf, comp_vf,
                            );
                        }
                    } else if comp_hf == 2 && comp_vf == 1 {
                        // H2V1: horizontal-only 2x fancy upsample.
                        for row in 0..actual_h {
                            self.fancy_upsample_h2v1(
                                &comp_plane[row * comp_w + comp_off..],
                                actual_w,
                                &mut comp_full[row * full_width..],
                            );
                        }
                    } else if comp_hf == 2 && comp_vf == 2 {
                        // H2V2: fused 2D triangle filter fancy upsample.
                        fancy_h2v2_strided_dispatch(
                            &comp_plane[comp_off..],
                            actual_w,
                            comp_w,
                            actual_h,
                            comp_full,
                            full_width,
                        );
                    } else if comp_hf == 1 && comp_vf == 2 {
                        // H1V2: vertical-only 2x fancy upsample.
                        if comp_off > 0 {
                            let mut cropped: Vec<u8> = Vec::with_capacity(actual_w * actual_h);
                            for row in 0..actual_h {
                                let s: usize = row * comp_w + comp_off;
                                cropped.extend_from_slice(&comp_plane[s..s + actual_w]);
                            }
                            self.fancy_h1v2(&cropped, actual_w, actual_h, comp_full, full_width);
                        } else {
                            self.fancy_h1v2(comp_plane, comp_w, actual_h, comp_full, full_width);
                        }
                    } else {
                        // Generic fallback: nearest-neighbor for any factor combination.
                        if comp_off > 0 {
                            let mut cropped: Vec<u8> = Vec::with_capacity(actual_w * actual_h);
                            for row in 0..comp_h.min(actual_h) {
                                let s: usize = row * comp_w + comp_off;
                                cropped.extend_from_slice(&comp_plane[s..s + actual_w]);
                            }
                            upsample_generic_nearest(
                                &cropped, actual_w, actual_h, comp_full, full_width, comp_hf,
                                comp_vf,
                            );
                        } else {
                            upsample_generic_nearest(
                                comp_plane, comp_w, comp_h, comp_full, full_width, comp_hf, comp_vf,
                            );
                        }
                    }
                }

                // Rebind as immutable references for color conversion below.
                // We use a trick: leak the Vecs temporarily, do the conversion,
                // then reconstruct and drop them. But simpler: just use a nested scope.
                // Actually, let's just do the color conversion here and return.
                let data_size = out_width * out_height * bpp;
                let mut data = take_out_buf(sink, data_size)?;
                for y in 0..out_height {
                    self.color_convert_row(
                        out_format,
                        &y_plane[y * y_width + comp_x_offsets[0]..],
                        &cb_full[y * full_width..],
                        &cr_full[y * full_width..],
                        &mut data[y * out_width * bpp..],
                        out_width,
                        y,
                    );
                }

                return Ok(Image {
                    xmp_data: try_clone_opt(&xmp_data, "XMP metadata")?,
                    iptc_data: try_clone_opt(&iptc_data, "IPTC metadata")?,
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    precision: 8,
                    data: data.into_vec(),
                    icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                    exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                    comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                    density: self.metadata.density,
                    saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                    warnings: warnings.clone(),
                });
            }

            // 4:4:4 path (no upsampling)
            let data_size = out_width * out_height * bpp;
            let mut data = take_out_buf(sink, data_size)?;
            for y in 0..out_height {
                self.color_convert_row(
                    out_format,
                    &y_plane[y * y_width + comp_x_offsets[0]..],
                    &cb_data[y * cb_stride + comp_x_offsets[1]..],
                    &cr_data[y * cr_stride + comp_x_offsets[2]..],
                    &mut data[y * out_width * bpp..],
                    out_width,
                    y,
                );
            }

            Ok(Image {
                xmp_data: try_clone_opt(&xmp_data, "XMP metadata")?,
                iptc_data: try_clone_opt(&iptc_data, "IPTC metadata")?,
                width: out_width,
                height: out_height,
                pixel_format: out_format,
                precision: 8,
                data: data.into_vec(),
                icc_profile: try_clone_opt(&icc_profile, "ICC profile")?,
                exif_data: try_clone_opt(&exif_data, "EXIF metadata")?,
                comment: try_clone_opt_string(&self.metadata.comment, "COM comment")?,
                density: self.metadata.density,
                saved_markers: try_clone_saved_markers(&self.metadata.saved_markers)?,
                warnings: warnings.clone(),
            })
        } else if num_components == 4 {
            self.decode_4_component(
                &component_planes,
                frame,
                out_width,
                out_height,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                full_width,
                full_height,
                &comp_block_sizes,
                icc_profile,
                exif_data,
                warnings,
            )
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported",
                num_components
            )))
        }
    }
}
