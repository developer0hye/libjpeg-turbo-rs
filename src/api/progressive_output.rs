/// Progressive buffered output / scan-by-scan decode.
///
/// Matches libjpeg-turbo's `buffered_image` mode: `jpeg_has_multiple_scans()`,
/// `jpeg_start_output()` / `jpeg_finish_output()`, `jpeg_consume_input()`,
/// `jpeg_input_complete()`.
///
/// Progressive JPEGs are encoded in multiple scans. This decoder allows you to
/// output the image after each scan, progressively refining the quality.
use crate::common::error::{JpegError, Result};
use crate::common::icc;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::marker::{JpegMetadata, MarkerReader, ScanInfo};
use crate::decode::pipeline::Image;
use crate::decode::progressive;
use crate::simd::{self, SimdRoutines};

/// Per-component layout info for progressive coefficient management.
struct CompInfo {
    blocks_x: usize,
    blocks_y: usize,
    h_samp: usize,
    v_samp: usize,
    comp_w: usize,
}

/// Decoder that supports scan-by-scan progressive output.
///
/// Progressive JPEGs encode image data in multiple scans, each refining
/// the image quality. This decoder lets you consume scans one at a time
/// and output the best available reconstruction at any point.
pub struct ProgressiveDecoder {
    /// Raw JPEG data (borrowed lifetime replaced with owned for simplicity).
    raw_data: Vec<u8>,
    /// Parsed metadata from JPEG headers.
    metadata: JpegMetadata,
    /// SIMD dispatch routines.
    routines: SimdRoutines,
    /// Per-component coefficient buffers, accumulated across scans.
    coeff_bufs: Vec<Vec<[i16; 64]>>,
    /// Per-component layout info.
    comp_infos: Vec<CompInfo>,
    /// MCUs in horizontal direction.
    mcus_x: usize,
    /// MCUs in vertical direction.
    mcus_y: usize,
    /// Max horizontal sampling factor.
    max_h: usize,
    /// Max vertical sampling factor.
    max_v: usize,
    /// Number of scans consumed so far.
    scans_consumed: usize,
}

impl ProgressiveDecoder {
    /// Create from JPEG data. Returns error if not a progressive JPEG.
    pub fn new(data: &[u8]) -> Result<Self> {
        let mut reader: MarkerReader<'_> = MarkerReader::new(data);
        let metadata: JpegMetadata = reader.read_markers()?;

        if !metadata.frame.is_progressive {
            return Err(JpegError::Unsupported(
                "ProgressiveDecoder requires a progressive JPEG (SOF2)".into(),
            ));
        }

        let frame = &metadata.frame;
        let max_h: usize = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v: usize = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);

        let mcu_w: usize = max_h * 8;
        let mcu_h: usize = max_v * 8;
        let mcus_x: usize = (frame.width as usize).div_ceil(mcu_w);
        let mcus_y: usize = (frame.height as usize).div_ceil(mcu_h);

        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .map(|comp| {
                let h_samp: usize = comp.horizontal_sampling as usize;
                let v_samp: usize = comp.vertical_sampling as usize;
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    // Full IDCT block size = 8
                    comp_w: mcus_x * h_samp * 8,
                }
            })
            .collect();

        // Allocate coefficient buffers (zero-initialized for progressive accumulation)
        let coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| vec![[0i16; 64]; ci.blocks_x * ci.blocks_y])
            .collect();

        let routines: SimdRoutines = simd::detect();

        Ok(Self {
            raw_data: data.to_vec(),
            metadata,
            routines,
            coeff_bufs,
            comp_infos,
            mcus_x,
            mcus_y,
            max_h,
            max_v,
            scans_consumed: 0,
        })
    }

    /// Check if the JPEG has multiple scans (i.e., is progressive).
    pub fn has_multiple_scans(&self) -> bool {
        self.metadata.scans.len() > 1
    }

    /// Get total number of scans in the image.
    pub fn num_scans(&self) -> usize {
        self.metadata.scans.len()
    }

    /// Get image width in pixels.
    pub fn width(&self) -> usize {
        self.metadata.frame.width as usize
    }

    /// Get image height in pixels.
    pub fn height(&self) -> usize {
        self.metadata.frame.height as usize
    }

    /// Consume the next scan from input.
    /// Returns true if a scan was consumed, false if all scans are done.
    pub fn consume_input(&mut self) -> Result<bool> {
        let scan_idx: usize = self.scans_consumed;
        if scan_idx >= self.metadata.scans.len() {
            return Ok(false);
        }

        self.decode_one_scan(scan_idx)?;
        self.scans_consumed += 1;
        Ok(true)
    }

    /// Check if all input scans have been consumed.
    pub fn input_complete(&self) -> bool {
        self.scans_consumed >= self.metadata.scans.len()
    }

    /// Get the number of scans consumed so far.
    pub fn scans_consumed(&self) -> usize {
        self.scans_consumed
    }

    /// Output the current image state (after consuming some scans).
    /// Returns the best available reconstruction from scans consumed so far.
    /// Each call to `consume_input()` followed by `output()` gives a
    /// progressively better image.
    pub fn output(&self) -> Result<Image> {
        let frame = &self.metadata.frame;
        let block_size: usize = 8;
        let num_components: usize = frame.components.len();
        let out_width: usize = frame.width as usize;
        let out_height: usize = frame.height as usize;
        let full_width: usize = self.mcus_x * self.max_h * block_size;
        let full_height: usize = self.mcus_y * self.max_v * block_size;

        // Resolve quant tables
        let quant_tables: Vec<&QuantTable> = frame
            .components
            .iter()
            .map(|comp| {
                self.metadata.quant_tables[comp.quant_table_index as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "missing quant table {}",
                            comp.quant_table_index
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // IDCT all blocks into component planes
        let mut component_planes: Vec<Vec<u8>> = self
            .comp_infos
            .iter()
            .map(|ci| {
                let size: usize = ci.comp_w * ci.blocks_y * block_size;
                let mut v: Vec<u8> = Vec::with_capacity(size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    v.set_len(size)
                };
                v
            })
            .collect();

        for (comp_idx, ci) in self.comp_infos.iter().enumerate() {
            let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx: usize = by * ci.blocks_x + bx;
                    let coeffs: &[i16; 64] = &self.coeff_bufs[comp_idx][block_idx];

                    let px_x: usize = bx * block_size;
                    let px_y: usize = by * block_size;
                    let dst_offset: usize = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst: *mut u8 = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_islow_strided(coeffs, qt_values, dst, ci.comp_w);
                    }
                }
            }
        }

        // Assemble into final Image with color conversion
        let icc_profile: Option<Vec<u8>> = icc::reassemble_icc_profile(&self.metadata.icc_chunks);
        let exif_data: Option<Vec<u8>> = self.metadata.exif_data.clone();

        if num_components == 1 {
            self.assemble_grayscale(
                &component_planes,
                out_width,
                out_height,
                icc_profile,
                exif_data,
            )
        } else if num_components == 3 {
            self.assemble_ycbcr(
                &component_planes,
                frame,
                out_width,
                out_height,
                full_width,
                full_height,
                icc_profile,
                exif_data,
            )
        } else if num_components == 4 {
            self.assemble_4_component(
                &component_planes,
                frame,
                out_width,
                out_height,
                full_width,
                full_height,
                icc_profile,
                exif_data,
            )
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not supported in progressive output",
                num_components
            )))
        }
    }

    /// Consume all remaining scans and output the final image.
    /// Equivalent to calling `consume_input()` in a loop then `output()`.
    pub fn finish(mut self) -> Result<Image> {
        while self.consume_input()? {}
        self.output()
    }

    // ---- Private helpers ----

    /// IDCT writing directly to a strided destination buffer.
    ///
    /// # Safety
    /// `output` must point to at least `7 * stride + 8` writable bytes.
    #[inline(always)]
    unsafe fn idct_islow_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::idct::neon_idct_islow_strided(
                coeffs, quant, output, stride,
            );
        }

        #[allow(unreachable_code)]
        {
            let mut tmp = [0u8; 64];
            (self.routines.idct_islow)(coeffs, quant, &mut tmp);
            for row in 0..8 {
                std::ptr::copy_nonoverlapping(
                    tmp.as_ptr().add(row * 8),
                    output.add(row * stride),
                    8,
                );
            }
        }
    }

    /// Decode a single progressive scan's entropy data into coefficient buffers.
    fn decode_one_scan(&mut self, scan_idx: usize) -> Result<()> {
        // Extract all needed scan parameters before mutably borrowing coeff_bufs.
        let scan_info: &ScanInfo = &self.metadata.scans[scan_idx];
        let ss: u8 = scan_info.header.spec_start;
        let se: u8 = scan_info.header.spec_end;
        let ah: u8 = scan_info.header.succ_high;
        let al: u8 = scan_info.header.succ_low;
        let is_dc: bool = ss == 0 && se == 0;
        let data_offset: usize = scan_info.data_offset;
        let restart_interval: u16 = scan_info.restart_interval;
        let num_scan_components: usize = scan_info.header.components.len();

        // Clone scan component selectors to avoid holding borrow on metadata
        let scan_components: Vec<ScanComponentSelector> = scan_info.header.components.clone();

        // Clone Huffman tables needed for this scan
        let dc_tables: [Option<crate::common::huffman_table::HuffmanTable>; 4] =
            scan_info.dc_huffman_tables.clone();
        let ac_tables: [Option<crate::common::huffman_table::HuffmanTable>; 4] =
            scan_info.ac_huffman_tables.clone();

        let entropy_data: &[u8] = &self.raw_data[data_offset..];
        let mut bit_reader: BitReader = BitReader::new(entropy_data);

        // Resolve component indices for this scan
        let scan_comp_indices: Vec<usize> = scan_components
            .iter()
            .map(|sc| {
                self.metadata
                    .frame
                    .components
                    .iter()
                    .position(|fc| fc.id == sc.component_id)
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "scan references unknown component {}",
                            sc.component_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        if num_scan_components > 1 {
            // Interleaved scan (DC only in progressive)
            let mut dc_preds = [0i16; 4];
            let mut mcu_count: u16 = 0;

            for mcu_y in 0..self.mcus_y {
                for mcu_x in 0..self.mcus_x {
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        bit_reader.reset();
                        dc_preds = [0i16; 4];
                    }

                    for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                        let blocks_x: usize = self.comp_infos[comp_idx].blocks_x;
                        let h_samp: usize = self.comp_infos[comp_idx].h_samp;
                        let v_samp: usize = self.comp_infos[comp_idx].v_samp;
                        let sc = &scan_components[si];

                        let dc_table =
                            dc_tables[sc.dc_table_index as usize]
                                .as_ref()
                                .ok_or_else(|| {
                                    JpegError::CorruptData(format!(
                                        "missing DC table {}",
                                        sc.dc_table_index
                                    ))
                                })?;

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let bx: usize = mcu_x * h_samp + h;
                                let by: usize = mcu_y * v_samp + v;
                                let block_idx: usize = by * blocks_x + bx;
                                let coeffs: &mut [i16; 64] =
                                    &mut self.coeff_bufs[comp_idx][block_idx];

                                if is_dc {
                                    if ah == 0 {
                                        progressive::decode_dc_first(
                                            &mut bit_reader,
                                            dc_table,
                                            &mut dc_preds[comp_idx],
                                            coeffs,
                                            al,
                                        )?;
                                    } else {
                                        progressive::decode_dc_refine(&mut bit_reader, coeffs, al)?;
                                    }
                                }
                            }
                        }
                    }

                    mcu_count += 1;
                }
            }
            Ok(())
        } else {
            // Non-interleaved scan (single component)
            let comp_idx: usize = scan_comp_indices[0];
            let sc = &scan_components[0];
            let blocks_x: usize = self.comp_infos[comp_idx].blocks_x;
            let blocks_y: usize = self.comp_infos[comp_idx].blocks_y;
            let mut dc_pred: i16 = 0;
            let mut eob_run: u16 = 0;
            let mut mcu_count: u16 = 0;

            let dc_table_ref = if is_dc {
                Some(
                    dc_tables[sc.dc_table_index as usize]
                        .as_ref()
                        .ok_or_else(|| {
                            JpegError::CorruptData(format!(
                                "missing DC table {}",
                                sc.dc_table_index
                            ))
                        })?,
                )
            } else {
                None
            };

            let ac_table_ref = if !is_dc || se > 0 {
                Some(
                    ac_tables[sc.ac_table_index as usize]
                        .as_ref()
                        .ok_or_else(|| {
                            JpegError::CorruptData(format!(
                                "missing AC table {}",
                                sc.ac_table_index
                            ))
                        })?,
                )
            } else {
                None
            };

            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        bit_reader.reset();
                        dc_pred = 0;
                        eob_run = 0;
                    }

                    let block_idx: usize = by * blocks_x + bx;
                    let coeffs: &mut [i16; 64] = &mut self.coeff_bufs[comp_idx][block_idx];

                    if is_dc {
                        if ah == 0 {
                            progressive::decode_dc_first(
                                &mut bit_reader,
                                dc_table_ref.unwrap(),
                                &mut dc_pred,
                                coeffs,
                                al,
                            )?;
                        } else {
                            progressive::decode_dc_refine(&mut bit_reader, coeffs, al)?;
                        }
                    } else if ah == 0 {
                        progressive::decode_ac_first(
                            &mut bit_reader,
                            ac_table_ref.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                        )?;
                    } else {
                        progressive::decode_ac_refine(
                            &mut bit_reader,
                            ac_table_ref.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                        )?;
                    }

                    mcu_count += 1;
                }
            }
            Ok(())
        }
    }

    /// Assemble a grayscale image from component planes.
    fn assemble_grayscale(
        &self,
        component_planes: &[Vec<u8>],
        out_width: usize,
        out_height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let comp_w: usize = self.comp_infos[0].comp_w;
        let mut data: Vec<u8> = Vec::with_capacity(out_width * out_height);
        for y in 0..out_height {
            data.extend_from_slice(&component_planes[0][y * comp_w..y * comp_w + out_width]);
        }
        Ok(Image {
            width: out_width,
            height: out_height,
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
    }

    /// Assemble a 3-component YCbCr image with upsampling and color conversion.
    #[allow(clippy::too_many_arguments)]
    fn assemble_ycbcr(
        &self,
        component_planes: &[Vec<u8>],
        frame: &FrameHeader,
        out_width: usize,
        out_height: usize,
        full_width: usize,
        full_height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let out_format: PixelFormat = PixelFormat::Rgb;
        let bpp: usize = out_format.bytes_per_pixel();

        let y_plane: &[u8] = &component_planes[0];
        let y_width: usize = self.comp_infos[0].comp_w;

        let cb_comp = &frame.components[1];
        let cb_w: usize = self.comp_infos[1].comp_w;
        let cb_h: usize = self.comp_infos[1].blocks_y * 8;

        let h_factor: usize = self.max_h / cb_comp.horizontal_sampling as usize;
        let v_factor: usize = self.max_v / cb_comp.vertical_sampling as usize;

        if h_factor == 1 && v_factor == 1 {
            // 4:4:4: no upsampling needed
            let data_size: usize = out_width * out_height * bpp;
            let mut data: Vec<u8> = Vec::with_capacity(data_size);
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(data_size)
            };
            for y in 0..out_height {
                self.ycbcr_to_rgb_row(
                    &y_plane[y * y_width..],
                    &component_planes[1][y * cb_w..],
                    &component_planes[2][y * cb_w..],
                    &mut data[y * out_width * bpp..],
                    out_width,
                );
            }
            Ok(Image {
                width: out_width,
                height: out_height,
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
        } else {
            // Upsample chroma
            let alloc_size: usize = full_width * full_height;
            let mut cb_full: Vec<u8> = Vec::with_capacity(alloc_size);
            let mut cr_full: Vec<u8> = Vec::with_capacity(alloc_size);
            unsafe {
                cb_full.set_len(alloc_size);
                cr_full.set_len(alloc_size);
            }

            if h_factor == 2 && v_factor == 1 {
                for row in 0..cb_h {
                    self.fancy_upsample_h2v1(
                        &component_planes[1][row * cb_w..],
                        cb_w,
                        &mut cb_full[row * full_width..],
                    );
                    self.fancy_upsample_h2v1(
                        &component_planes[2][row * cb_w..],
                        cb_w,
                        &mut cr_full[row * full_width..],
                    );
                }
            } else if h_factor == 2 && v_factor == 2 {
                self.fancy_h2v2(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                self.fancy_h2v2(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
            } else if h_factor == 1 && v_factor == 2 {
                self.fancy_h1v2(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                self.fancy_h1v2(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
            } else if h_factor == 4 && v_factor == 1 {
                for row in 0..cb_h {
                    Self::fancy_upsample_h4v1(
                        &component_planes[1][row * cb_w..row * cb_w + cb_w],
                        cb_w,
                        &mut cb_full[row * full_width..],
                    );
                    Self::fancy_upsample_h4v1(
                        &component_planes[2][row * cb_w..row * cb_w + cb_w],
                        cb_w,
                        &mut cr_full[row * full_width..],
                    );
                }
            } else if h_factor == 1 && v_factor == 4 {
                self.fancy_h1v4(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                self.fancy_h1v4(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
            } else {
                return Err(JpegError::Unsupported(format!(
                    "subsampling {}x{} not supported in progressive output",
                    h_factor, v_factor
                )));
            }

            let data_size: usize = out_width * out_height * bpp;
            let mut data: Vec<u8> = Vec::with_capacity(data_size);
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(data_size)
            };
            for y in 0..out_height {
                self.ycbcr_to_rgb_row(
                    &y_plane[y * y_width..],
                    &cb_full[y * full_width..],
                    &cr_full[y * full_width..],
                    &mut data[y * out_width * bpp..],
                    out_width,
                );
            }

            Ok(Image {
                width: out_width,
                height: out_height,
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

    /// Assemble a 4-component (CMYK/YCCK) image.
    #[allow(clippy::too_many_arguments)]
    fn assemble_4_component(
        &self,
        component_planes: &[Vec<u8>],
        _frame: &FrameHeader,
        out_width: usize,
        out_height: usize,
        _full_width: usize,
        _full_height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        // For 4-component, output as CMYK (no color conversion)
        let bpp: usize = 4;
        let data_size: usize = out_width * out_height * bpp;
        let mut data: Vec<u8> = Vec::with_capacity(data_size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(data_size)
        };

        for y in 0..out_height {
            for x in 0..out_width {
                for c in 0..4 {
                    let comp_w: usize = self.comp_infos[c].comp_w;
                    data[y * out_width * bpp + x * bpp + c] = component_planes[c][y * comp_w + x];
                }
            }
        }

        Ok(Image {
            width: out_width,
            height: out_height,
            pixel_format: PixelFormat::Cmyk,
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

    // ---- Color conversion and upsampling delegates ----
    // These mirror the Decoder methods but operate on &self.

    #[inline(always)]
    fn ycbcr_to_rgb_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgb_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        (self.routines.ycbcr_to_rgb_row)(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn fancy_upsample_h2v1(&self, input: &[u8], in_width: usize, output: &mut [u8]) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::upsample::neon_fancy_upsample_h2v1(
                input, in_width, output,
            );
        }

        #[allow(unreachable_code)]
        (self.routines.fancy_upsample_h2v1)(input, in_width, output)
    }

    fn fancy_h2v2(
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

        // Fused H2V2: vertical + horizontal in one pass using >> 4 arithmetic.
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

    fn fancy_h1v2(
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

            let out_y_top: usize = y * 2;
            let out_y_bot: usize = y * 2 + 1;

            for i in 0..in_width {
                output[out_y_top * out_width + i] =
                    ((3 * cur_row[i] as u16 + above[i] as u16 + 2) >> 2) as u8;
                output[out_y_bot * out_width + i] =
                    ((3 * cur_row[i] as u16 + below[i] as u16 + 2) >> 2) as u8;
            }
        }
    }

    fn fancy_h1v4(
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

            // 4x vertical upsample with triangle filter
            for out_row in 0..4 {
                let out_y: usize = y * 4 + out_row;
                let (weight_cur, weight_nbr, nbr) = match out_row {
                    0 => (7u16, 1u16, above),
                    1 => (5u16, 3u16, above),
                    2 => (5u16, 3u16, below),
                    3 => (7u16, 1u16, below),
                    _ => unreachable!(),
                };
                for i in 0..in_width {
                    output[out_y * out_width + i] =
                        ((weight_cur * cur_row[i] as u16 + weight_nbr * nbr[i] as u16 + 4) >> 3)
                            as u8;
                }
            }
        }
    }

    fn fancy_upsample_h4v1(input: &[u8], in_width: usize, output: &mut [u8]) {
        for x in 0..in_width {
            let left = if x > 0 { input[x - 1] } else { input[x] };
            let center = input[x];
            let right = if x + 1 < in_width {
                input[x + 1]
            } else {
                input[x]
            };

            // 4x horizontal upsample with triangle filter
            let l = left as u16;
            let c = center as u16;
            let r = right as u16;
            output[x * 4] = ((7 * c + l + 4) >> 3) as u8;
            output[x * 4 + 1] = ((5 * c + 3 * l + 4) >> 3) as u8;
            output[x * 4 + 2] = ((5 * c + 3 * r + 4) >> 3) as u8;
            output[x * 4 + 3] = ((7 * c + r + 4) >> 3) as u8;
        }
    }
}
