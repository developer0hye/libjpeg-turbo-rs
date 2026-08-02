use super::output::pad_alpha_offset;
use super::{Decoder, Image};
use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::icc;
use crate::common::types::{FrameHeader, PixelFormat};
use crate::decode::bitstream::BitReader;
use crate::decode::{huffman, lossless as lossless_codec};
use alloc::{format, string::ToString, vec, vec::Vec};

impl<'a> Decoder<'a> {
    /// Reassemble ICC profile from parsed APP2 chunks.
    pub(super) fn icc_profile(&self) -> Option<Vec<u8>> {
        icc::reassemble_icc_profile(&self.metadata.icc_chunks)
    }

    /// Decode a lossless JPEG (SOF3).
    ///
    /// Lossless JPEG uses Huffman-coded differences + prediction instead of DCT.
    /// No quantization or IDCT is involved.
    pub(super) fn decode_lossless_image(
        &self,
        frame: &FrameHeader,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        if self.metadata.is_arithmetic {
            self.decode_lossless_arithmetic(frame, width, height, icc_profile, exif_data)
        } else {
            self.decode_lossless_huffman(frame, width, height, icc_profile, exif_data)
        }
    }

    /// Decode lossless JPEG with Huffman entropy coding (SOF3).
    pub(super) fn decode_lossless_huffman(
        &self,
        frame: &FrameHeader,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let scan = &self.metadata.scan;
        let precision = frame.precision;
        let psv = scan.spec_start; // Predictor selection value (Ss field)
        let pt = scan.succ_low; // Point transform (Al field)

        if !(1..=7).contains(&psv) {
            return Err(JpegError::Unsupported(format!(
                "lossless predictor {} (must be 1-7)",
                psv
            )));
        }

        let num_components = frame.components.len();

        // Resolve DC Huffman tables for each scan component
        let mut dc_tables: Vec<&HuffmanTable> = Vec::with_capacity(num_components);
        for i in 0..scan.components.len().min(num_components) {
            let dc_tbl_idx = scan.components[i].dc_table_index as usize;
            let dc_table = self.metadata.dc_huffman_tables[dc_tbl_idx]
                .as_ref()
                .ok_or_else(|| {
                    JpegError::CorruptData(format!("missing DC Huffman table {}", dc_tbl_idx))
                })?;
            dc_tables.push(dc_table);
        }

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut reader = BitReader::new(entropy_data);

        if num_components == 1 {
            // Single-component (grayscale) lossless decode
            let dc_table = dc_tables[0];
            let mut output = vec![0u16; width * height];
            let mut prev_row: Option<Vec<u16>> = None;
            let ri: u32 = self.metadata.restart_interval as u32;
            let mut mcu_count: u32 = 0;

            for y in 0..height {
                let row_start = y * width;
                let mut diffs = Vec::with_capacity(width);
                for _x in 0..width {
                    if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                        reader.reset();
                        prev_row = None;
                    }
                    let diff = huffman::decode_dc_coefficient(&mut reader, dc_table)?;
                    diffs.push(diff);
                    mcu_count += 1;
                }
                lossless_codec::undifference_row(
                    &diffs,
                    prev_row.as_deref(),
                    &mut output[row_start..row_start + width],
                    psv,
                    precision,
                    pt,
                    prev_row.is_none(),
                );
                prev_row = Some(output[row_start..row_start + width].to_vec());
            }

            self.lossless_output_grayscale(&output, width, height, pt, icc_profile, exif_data)
        } else if num_components == 3 {
            // Multi-component (color) lossless decode — interleaved scan.
            // The interleaved loop indexes `dc_tables[0..3]`, so a malformed
            // SOS that lists fewer than 3 components (or omits any DC table
            // entry) must be rejected up front instead of panicking on the
            // first MCU. Discovered via fuzz_decompress_lenient on a 3-comp
            // SOF3 with a 1-component SOS.
            if dc_tables.len() < 3 {
                return Err(JpegError::CorruptData(format!(
                    "lossless 3-component SOS lists {} DC table(s); 3 required",
                    dc_tables.len()
                )));
            }
            let mut comp_planes: Vec<Vec<u16>> =
                (0..3).map(|_| vec![0u16; width * height]).collect();
            let mut prev_rows: Vec<Option<Vec<u16>>> = vec![None; 3];
            let ri: u32 = self.metadata.restart_interval as u32;
            let mut mcu_count: u32 = 0;

            for y in 0..height {
                let row_start = y * width;
                let mut comp_diffs: Vec<Vec<i16>> =
                    (0..3).map(|_| Vec::with_capacity(width)).collect();

                // Interleaved: for each pixel, decode diff for each component
                for _ in 0..width {
                    if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                        reader.reset();
                        for pr in prev_rows.iter_mut() {
                            *pr = None;
                        }
                    }
                    for c in 0..3 {
                        let diff = huffman::decode_dc_coefficient(&mut reader, dc_tables[c])?;
                        comp_diffs[c].push(diff);
                    }
                    mcu_count += 1;
                }

                // Undifference each component
                for c in 0..3 {
                    lossless_codec::undifference_row(
                        &comp_diffs[c],
                        prev_rows[c].as_deref(),
                        &mut comp_planes[c][row_start..row_start + width],
                        psv,
                        precision,
                        pt,
                        prev_rows[c].is_none(),
                    );
                    prev_rows[c] = Some(comp_planes[c][row_start..row_start + width].to_vec());
                }
            }

            self.lossless_output_color(&comp_planes, width, height, pt, icc_profile, exif_data)
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported for lossless",
                num_components
            )))
        }
    }

    /// Decode lossless JPEG with arithmetic entropy coding (SOF11).
    pub(super) fn decode_lossless_arithmetic(
        &self,
        frame: &FrameHeader,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        use crate::decode::arithmetic::ArithDecoder;

        let scan = &self.metadata.scan;
        let precision = frame.precision;
        let psv = scan.spec_start;
        let pt = scan.succ_low;

        if !(1..=7).contains(&psv) {
            return Err(JpegError::Unsupported(format!(
                "lossless predictor {} (must be 1-7)",
                psv
            )));
        }

        let num_components = frame.components.len();

        // Resolve DC table indices for each scan component
        let dc_tbl_indices: Vec<usize> = scan
            .components
            .iter()
            .take(num_components)
            .map(|sc| sc.dc_table_index as usize)
            .collect();
        // Lossless multi-component decode below indexes `dc_tbl_indices[c]`
        // for every frame component; malformed SOS with fewer scan
        // components than frame.components would trip an OOB panic.
        if dc_tbl_indices.len() < num_components {
            return Err(JpegError::CorruptData(format!(
                "lossless SOS has {} components but frame has {}",
                dc_tbl_indices.len(),
                num_components
            )));
        }

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut arith = ArithDecoder::new(entropy_data, 0);

        // Set conditioning parameters from DAC marker (16 slots).
        for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
            let (l, u) = self.metadata.arith_dc_params[i];
            arith.set_dc_conditioning(i, l, u);
            arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
        }

        if num_components == 1 {
            let dc_tbl = dc_tbl_indices[0];
            let mut output = vec![0u16; width * height];
            let mut prev_row: Option<Vec<u16>> = None;

            for y in 0..height {
                let row_start = y * width;
                let mut diffs = Vec::with_capacity(width);
                for _ in 0..width {
                    // Save previous accumulated DC to extract the raw difference
                    let prev_dc: i32 = arith.last_dc_val[0];
                    let mut block: [i16; 64] = [0i16; 64];
                    arith.decode_dc_sequential(&mut block, 0, dc_tbl)?;
                    let diff: i16 = (arith.last_dc_val[0] - prev_dc) as i16;
                    diffs.push(diff);
                }
                lossless_codec::undifference_row(
                    &diffs,
                    prev_row.as_deref(),
                    &mut output[row_start..row_start + width],
                    psv,
                    precision,
                    pt,
                    y == 0,
                );
                prev_row = Some(output[row_start..row_start + width].to_vec());
            }

            self.lossless_output_grayscale(&output, width, height, pt, icc_profile, exif_data)
        } else if num_components == 3 {
            let mut comp_planes: Vec<Vec<u16>> =
                (0..3).map(|_| vec![0u16; width * height]).collect();
            let mut prev_rows: Vec<Option<Vec<u16>>> = vec![None; 3];

            for y in 0..height {
                let row_start = y * width;
                let mut comp_diffs: Vec<Vec<i16>> =
                    (0..3).map(|_| Vec::with_capacity(width)).collect();

                // Interleaved: for each pixel, decode diff for each component
                for _ in 0..width {
                    for c in 0..3 {
                        let prev_dc: i32 = arith.last_dc_val[c];
                        let mut block: [i16; 64] = [0i16; 64];
                        arith.decode_dc_sequential(&mut block, c, dc_tbl_indices[c])?;
                        let diff: i16 = (arith.last_dc_val[c] - prev_dc) as i16;
                        comp_diffs[c].push(diff);
                    }
                }

                // Undifference each component
                for c in 0..3 {
                    lossless_codec::undifference_row(
                        &comp_diffs[c],
                        prev_rows[c].as_deref(),
                        &mut comp_planes[c][row_start..row_start + width],
                        psv,
                        precision,
                        pt,
                        y == 0,
                    );
                    prev_rows[c] = Some(comp_planes[c][row_start..row_start + width].to_vec());
                }
            }

            self.lossless_output_color(&comp_planes, width, height, pt, icc_profile, exif_data)
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported for lossless",
                num_components
            )))
        }
    }

    /// Convert decoded lossless grayscale samples to output Image.
    pub(super) fn lossless_output_grayscale(
        &self,
        output: &[u16],
        width: usize,
        height: usize,
        pt: u8,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let out_format = self.output_format.unwrap_or(PixelFormat::Grayscale);
        let bpp = out_format.bytes_per_pixel();

        if out_format == PixelFormat::Grayscale {
            let mut data = Vec::with_capacity(width * height);
            for &sample in output {
                let val = if pt > 0 {
                    ((sample as u32) << pt) as u8
                } else {
                    sample as u8
                };
                data.push(val);
            }
            Ok(Image {
                xmp_data: self.metadata.xmp_data.clone(),
                iptc_data: self.metadata.iptc_data.clone(),
                width,
                height,
                pixel_format: PixelFormat::Grayscale,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment: self.metadata.comment.clone(),
                density: self.metadata.density,
                saved_markers: self.metadata.saved_markers.clone(),
                warnings: Vec::new(),
            })
        } else {
            // Note: C refuses lossless non-RGB→extended-RGB conversion
            // outright (jdcolor.c JERR_CONVERSION_NOTIMPL when
            // master->lossless && jpeg_color_space != JCS_RGB, see P4-64);
            // we expand with the same channel layout as the baseline
            // grayscale path below (issue #369 had Argb/Abgr grouped
            // alpha-last here).
            let pad_off: Option<usize> = pad_alpha_offset(out_format);
            let mut data = Vec::with_capacity(width * height * bpp);
            for &sample in output {
                let val = if pt > 0 {
                    ((sample as u32) << pt) as u8
                } else {
                    sample as u8
                };
                match out_format {
                    PixelFormat::Rgb | PixelFormat::Bgr => {
                        data.push(val);
                        data.push(val);
                        data.push(val);
                    }
                    PixelFormat::Rgba
                    | PixelFormat::Bgra
                    | PixelFormat::Rgbx
                    | PixelFormat::Bgrx
                    | PixelFormat::Xrgb
                    | PixelFormat::Xbgr
                    | PixelFormat::Argb
                    | PixelFormat::Abgr => {
                        let mut px: [u8; 4] = [val; 4];
                        px[pad_off.expect("4bpp format has a pad offset")] = 255;
                        data.extend_from_slice(&px);
                    }
                    PixelFormat::Rgb565 => {
                        let packed: u16 = ((val as u16 >> 3) << 11)
                            | ((val as u16 >> 2) << 5)
                            | (val as u16 >> 3);
                        let bytes: [u8; 2] = packed.to_ne_bytes();
                        data.push(bytes[0]);
                        data.push(bytes[1]);
                    }
                    _ => unreachable!(),
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
                warnings: Vec::new(),
            })
        }
    }

    /// Convert decoded lossless RGB component planes to output Image.
    ///
    /// Lossless JPEG stores raw RGB values with no color conversion (matching
    /// C libjpeg-turbo JCS_RGB behavior), so we output component values directly.
    pub(super) fn lossless_output_color(
        &self,
        comp_planes: &[Vec<u16>],
        width: usize,
        height: usize,
        pt: u8,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let out_format = self.output_format.unwrap_or(PixelFormat::Rgb);
        let bpp = out_format.bytes_per_pixel();
        let mut data = Vec::with_capacity(width * height * bpp);

        // C jdlossls.c simple_upscale/noscale: every component sample is
        // scaled by `<< Al` and truncated to the sample type
        // (`(_JSAMPLE)(x << Al)`, i.e. `& 0xFF` for 8-bit) — not
        // saturated. Skipping the shift here left color output in the
        // point-transformed domain (Fuzz Smoke run 29689718301, P4-38).
        let shift: u32 = (pt as u32).min(15);

        for ((&r_pix, &g_pix), &b_pix) in comp_planes[0]
            .iter()
            .zip(comp_planes[1].iter())
            .zip(comp_planes[2].iter())
        {
            // Raw RGB: output component values directly (no color conversion)
            let r: u8 = ((r_pix as u32) << shift) as u8;
            let g: u8 = ((g_pix as u32) << shift) as u8;
            let b: u8 = ((b_pix as u32) << shift) as u8;

            match out_format {
                PixelFormat::Rgb => {
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
                PixelFormat::Bgr => {
                    data.push(b);
                    data.push(g);
                    data.push(r);
                }
                PixelFormat::Rgba | PixelFormat::Rgbx => {
                    data.push(r);
                    data.push(g);
                    data.push(b);
                    data.push(255);
                }
                PixelFormat::Bgra | PixelFormat::Bgrx => {
                    data.push(b);
                    data.push(g);
                    data.push(r);
                    data.push(255);
                }
                PixelFormat::Xrgb | PixelFormat::Argb => {
                    data.push(255);
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
                PixelFormat::Xbgr | PixelFormat::Abgr => {
                    data.push(255);
                    data.push(b);
                    data.push(g);
                    data.push(r);
                }
                PixelFormat::Rgb565 => {
                    let packed: u16 =
                        ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3);
                    let bytes: [u8; 2] = packed.to_ne_bytes();
                    data.push(bytes[0]);
                    data.push(bytes[1]);
                }
                _ => {
                    return Err(JpegError::Unsupported(
                        "cannot convert lossless 3-component to requested format".to_string(),
                    ));
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
            warnings: Vec::new(),
        })
    }
}
