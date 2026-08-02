use super::{upsample_generic_nearest, Decoder, Image};
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::types::{ColorSpace, FrameHeader, PixelFormat};
use alloc::{format, string::ToString, vec, vec::Vec};

impl<'a> Decoder<'a> {
    /// Determine the JPEG color space from component count and Adobe marker.
    /// Follows the same heuristic as libjpeg-turbo (jdapimin.c).
    pub(super) fn detect_color_space(&self) -> ColorSpace {
        let num_components = self.metadata.frame.components.len();
        match num_components {
            1 => ColorSpace::Grayscale,
            3 => {
                if self.metadata.saw_jfif_marker {
                    // JFIF takes precedence over Adobe and component IDs in
                    // libjpeg-turbo: a three-component JFIF stream is YCbCr.
                    ColorSpace::YCbCr
                } else if self.metadata.saw_adobe_marker {
                    if self.metadata.adobe_transform == 0 {
                        ColorSpace::Rgb
                    } else {
                        ColorSpace::YCbCr
                    }
                } else {
                    let components = &self.metadata.frame.components;
                    let ids = [components[0].id, components[1].id, components[2].id];
                    if ids == *b"RGB" || self.metadata.frame.is_lossless {
                        ColorSpace::Rgb
                    } else {
                        ColorSpace::YCbCr
                    }
                }
            }
            4 => {
                if self.metadata.saw_adobe_marker {
                    match self.metadata.adobe_transform {
                        0 => ColorSpace::Cmyk,
                        2 => ColorSpace::Ycck,
                        _ => ColorSpace::Ycck, // default for unknown Adobe transforms
                    }
                } else {
                    ColorSpace::Cmyk // no Adobe marker → assume CMYK
                }
            }
            _ => ColorSpace::YCbCr, // fallback
        }
    }

    /// Decode a 4-component (CMYK/YCCK) image.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_4_component(
        &self,
        component_planes: &[Vec<u8>],
        frame: &FrameHeader,
        width: usize,
        height: usize,
        mcus_x: usize,
        mcus_y: usize,
        _max_h: usize,
        _max_v: usize,
        full_width: usize,
        full_height: usize,
        comp_block_sizes: &[usize],
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        let color_space = self.detect_color_space();
        let out_format = self.output_format.unwrap_or(PixelFormat::Cmyk);

        if out_format == PixelFormat::Grayscale {
            return Err(JpegError::Unsupported(
                "cannot convert CMYK/YCCK to grayscale".to_string(),
            ));
        }

        // Component 0 is always full-resolution (Y or C).
        let comp0_w =
            mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];

        // For YCCK, components 1-2 may be subsampled (chroma), component 3 (K) is full.
        // For CMYK, all components are typically the same resolution.
        let comp1 = &frame.components[1];
        let comp1_w = mcus_x * comp1.horizontal_sampling as usize * comp_block_sizes[1];
        let comp1_h = mcus_y * comp1.vertical_sampling as usize * comp_block_sizes[1];
        let comp3_w =
            mcus_x * frame.components[3].horizontal_sampling as usize * comp_block_sizes[3];

        let h_factor = comp0_w / comp1_w;
        let v_factor =
            (mcus_y * frame.components[0].vertical_sampling as usize * comp_block_sizes[0])
                / (mcus_y * comp1.vertical_sampling as usize * comp_block_sizes[1]);

        // Upsample chroma if needed (for YCCK subsampled images)
        let (plane1, plane2, p1_stride, p2_stride): (&[u8], &[u8], usize, usize);

        if h_factor == 1 && v_factor == 1 {
            plane1 = &component_planes[1];
            plane2 = &component_planes[2];
            p1_stride = comp1_w;
            p2_stride = comp1_w;
        } else {
            let alloc_size = full_width * full_height;
            let mut p1_full = vec![0u8; alloc_size];
            let mut p2_full = vec![0u8; alloc_size];

            // Honor TJPARAM_FASTUPSAMPLE: when set, use the box-filter
            // (nearest-neighbor) upsample instead of the fancy triangle
            // filter. The 3-component path already checks self.fast_upsample;
            // the 4-component (CMYK/YCCK) path was not, so tjunittest CMYK
            // 4:2:2/4:2:0/4:4:0 subtests at scaled output saw the fancy
            // filter blend chroma values across checker boundaries (191
            // instead of 255 / 64 instead of 0), failing the per-pixel
            // CHECKVAL with tolerance=1.
            if self.fast_upsample {
                upsample_generic_nearest(
                    &component_planes[1],
                    comp1_w,
                    comp1_h,
                    &mut p1_full,
                    full_width,
                    h_factor,
                    v_factor,
                );
                upsample_generic_nearest(
                    &component_planes[2],
                    comp1_w,
                    comp1_h,
                    &mut p2_full,
                    full_width,
                    h_factor,
                    v_factor,
                );
            } else if h_factor == 2 && v_factor == 1 {
                for row in 0..comp1_h {
                    self.fancy_upsample_h2v1(
                        &component_planes[1][row * comp1_w..],
                        comp1_w,
                        &mut p1_full[row * full_width..],
                    );
                    self.fancy_upsample_h2v1(
                        &component_planes[2][row * comp1_w..],
                        comp1_w,
                        &mut p2_full[row * full_width..],
                    );
                }
            } else if h_factor == 2 && v_factor == 2 {
                self.fancy_h2v2(
                    &component_planes[1],
                    comp1_w,
                    comp1_h,
                    &mut p1_full,
                    full_width,
                );
                self.fancy_h2v2(
                    &component_planes[2],
                    comp1_w,
                    comp1_h,
                    &mut p2_full,
                    full_width,
                );
            } else {
                // Generic fallback for non-standard 4-component sampling factors.
                upsample_generic_nearest(
                    &component_planes[1],
                    comp1_w,
                    comp1_h,
                    &mut p1_full,
                    full_width,
                    h_factor,
                    v_factor,
                );
                upsample_generic_nearest(
                    &component_planes[2],
                    comp1_w,
                    comp1_h,
                    &mut p2_full,
                    full_width,
                    h_factor,
                    v_factor,
                );
            }

            return self.convert_4comp_output(
                color_space,
                out_format,
                &component_planes[0],
                comp0_w,
                &p1_full,
                full_width,
                &p2_full,
                full_width,
                &component_planes[3],
                comp3_w,
                width,
                height,
                icc_profile,
                exif_data,
                warnings,
            );
        }

        self.convert_4comp_output(
            color_space,
            out_format,
            &component_planes[0],
            comp0_w,
            plane1,
            p1_stride,
            plane2,
            p2_stride,
            &component_planes[3],
            comp3_w,
            width,
            height,
            icc_profile,
            exif_data,
            warnings,
        )
    }

    /// Color-convert 4 component planes to the output format.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_4comp_output(
        &self,
        color_space: ColorSpace,
        out_format: PixelFormat,
        plane0: &[u8],
        p0_stride: usize,
        plane1: &[u8],
        p1_stride: usize,
        plane2: &[u8],
        p2_stride: usize,
        plane3: &[u8],
        p3_stride: usize,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        use crate::decode::color;

        let bpp = out_format.bytes_per_pixel();
        let data_size = width * height * bpp;
        let mut data = vec![0u8; data_size];

        for y in 0..height {
            let p0 = &plane0[y * p0_stride..];
            let p1 = &plane1[y * p1_stride..];
            let p2 = &plane2[y * p2_stride..];
            let p3 = &plane3[y * p3_stride..];
            let out = &mut data[y * width * bpp..];

            match (color_space, out_format) {
                // CMYK → CMYK: passthrough
                (ColorSpace::Cmyk, PixelFormat::Cmyk) => {
                    color::cmyk_passthrough_row(p0, p1, p2, p3, out, width);
                }
                // CMYK → RGB/RGBA/BGR/BGRA: direct conversion
                (ColorSpace::Cmyk, PixelFormat::Rgb) => {
                    color::cmyk_to_rgb_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Rgba) => {
                    color::cmyk_to_rgba_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Bgr) => {
                    color::cmyk_to_bgr_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Bgra) => {
                    color::cmyk_to_bgra_row(p0, p1, p2, p3, out, width);
                }
                // YCCK → CMYK: YCbCr→RGB→invert→CMYK, K passthrough
                (ColorSpace::Ycck, PixelFormat::Cmyk) => {
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, out, width);
                }
                // YCCK → RGB: convert YCCK→CMYK first (into temp), then CMYK→RGB
                (ColorSpace::Ycck, PixelFormat::Rgb) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        out[x * 3] = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        out[x * 3 + 1] = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        out[x * 3 + 2] = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Rgba) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        out[x * 4] = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + 1] = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + 2] = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + 3] = 255;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgr) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        let r = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        let g = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        let b = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                        out[x * 3] = b;
                        out[x * 3 + 1] = g;
                        out[x * 3 + 2] = r;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgra) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        let r = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        let g = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        let b = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                        out[x * 4] = b;
                        out[x * 4 + 1] = g;
                        out[x * 4 + 2] = r;
                        out[x * 4 + 3] = 255;
                    }
                }
                // CMYK → 4bpp offset-based formats
                (
                    ColorSpace::Cmyk,
                    PixelFormat::Rgbx
                    | PixelFormat::Bgrx
                    | PixelFormat::Xrgb
                    | PixelFormat::Xbgr
                    | PixelFormat::Argb
                    | PixelFormat::Abgr,
                ) => {
                    let r_off: usize = out_format.red_offset().unwrap();
                    let g_off: usize = out_format.green_offset().unwrap();
                    let b_off: usize = out_format.blue_offset().unwrap();
                    // The remaining offset is 0+1+2+3=6 minus the other three
                    let pad_off: usize = 6 - r_off - g_off - b_off;
                    for x in 0..width {
                        let kv = p3[x] as u16;
                        let r = ((p0[x] as u16 * kv + 127) / 255) as u8;
                        let g = ((p1[x] as u16 * kv + 127) / 255) as u8;
                        let b = ((p2[x] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + r_off] = r;
                        out[x * 4 + g_off] = g;
                        out[x * 4 + b_off] = b;
                        out[x * 4 + pad_off] = 255;
                    }
                }
                // YCCK → 4bpp offset-based formats
                (
                    ColorSpace::Ycck,
                    PixelFormat::Rgbx
                    | PixelFormat::Bgrx
                    | PixelFormat::Xrgb
                    | PixelFormat::Xbgr
                    | PixelFormat::Argb
                    | PixelFormat::Abgr,
                ) => {
                    let r_off: usize = out_format.red_offset().unwrap();
                    let g_off: usize = out_format.green_offset().unwrap();
                    let b_off: usize = out_format.blue_offset().unwrap();
                    let pad_off: usize = 6 - r_off - g_off - b_off;
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        let r = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        let g = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        let b = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + r_off] = r;
                        out[x * 4 + g_off] = g;
                        out[x * 4 + b_off] = b;
                        out[x * 4 + pad_off] = 255;
                    }
                }
                _ => {
                    return Err(JpegError::Unsupported(format!(
                        "unsupported conversion: {:?} → {:?}",
                        color_space, out_format
                    )));
                }
            }
        }

        Ok(Image {
            xmp_data: self.metadata.xmp_data.clone(),
            iptc_data: self.metadata.iptc_data.clone(),
            width,
            height,
            pixel_format: out_format,
            precision: 8,
            data,
            icc_profile,
            exif_data,
            comment: self.metadata.comment.clone(),
            density: self.metadata.density,
            saved_markers: self.metadata.saved_markers.clone(),
            warnings,
        })
    }
}
