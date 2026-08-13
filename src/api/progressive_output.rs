// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::{JpegError, Result};
use crate::common::icc;
use crate::common::quant_table::QuantTable;
use crate::common::try_alloc::try_clone_saved_markers;
/// Progressive buffered output / scan-by-scan decode.
///
/// Matches libjpeg-turbo's `buffered_image` mode: `jpeg_has_multiple_scans()`,
/// `jpeg_start_output()` / `jpeg_finish_output()`, `jpeg_consume_input()`,
/// `jpeg_input_complete()`.
///
/// Progressive JPEGs are encoded in multiple scans. This decoder allows you to
/// output the image after each scan, progressively refining the quality.
use crate::common::try_alloc::{
    try_clone_opt, try_clone_opt_string, try_copy_of, try_filled_vec, try_reserved_vec,
};
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::marker::{JpegMetadata, MarkerReader, ScanInfo};
use crate::decode::pipeline::{upsample_generic_nearest, Image};
use crate::decode::progressive;
use crate::simd::{self, SimdRoutines};
#[allow(unused_imports)]
use alloc::vec::Vec;
#[allow(unused_imports)]
use alloc::{format, vec};

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
    /// Per-block highest nonzero AC zigzag index (see issue #352:
    /// bounds refinement EOB-run walks to the block's spectral extent).
    ac_max_k_bufs: Vec<Vec<u8>>,
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

/// Byte size of a plane, refusing geometry that would wrap `usize`.
///
/// The factors come from header geometry an attacker controls, so an unchecked
/// product wraps in release and yields a short allocation that the IDCT then
/// writes past — a memory-safety bug independent of how the buffer is
/// initialized (P4-136). `isize::MAX` is the ceiling because a single
/// allocation larger than that violates the allocator contract regardless of
/// available memory.
fn checked_plane_size(factors: &[usize], what: &'static str) -> Result<usize> {
    let mut total: usize = 1;
    for factor in factors {
        total = total.checked_mul(*factor).ok_or(JpegError::LimitExceeded {
            what,
            actual: u64::MAX,
            limit: isize::MAX as u64,
        })?;
    }
    if total > isize::MAX as usize {
        return Err(JpegError::LimitExceeded {
            what,
            actual: total as u64,
            limit: isize::MAX as u64,
        });
    }
    Ok(total)
}

impl ProgressiveDecoder {
    /// Create from JPEG data. Returns error if not a progressive JPEG.
    /// Applies [`crate::common::types::DecodeLimits::default`]; use
    /// [`Self::with_limits`] to tighten (e.g. a `max_memory` ceiling for
    /// the coefficient buffers this decoder holds across scans).
    pub fn new(data: &[u8]) -> Result<Self> {
        Self::with_limits(data, crate::common::types::DecodeLimits::default())
    }

    /// Like [`Self::new`] with caller-chosen resource limits, applied
    /// from marker parsing onward (issue #355).
    pub fn with_limits(data: &[u8], limits: crate::common::types::DecodeLimits) -> Result<Self> {
        let mut reader: MarkerReader<'_> = MarkerReader::new(data);
        reader.set_scan_cap(limits.max_scans);
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

        // Resource guards (issue #355): coefficient buffers are
        // allocated below from header-declared dimensions, so bound them
        // before any allocation.
        limits.check_frame(frame.width as usize, frame.height as usize)?;
        // Defence-in-depth: with default limits this is subsumed by the
        // parse-time cap in read_markers; it stays so the two cannot
        // drift and so tighter caller limits apply here too.
        if metadata.scans.len() > limits.max_scans {
            return Err(JpegError::LimitExceeded {
                what: "progressive scan count",
                actual: metadata.scans.len() as u64,
                limit: limits.max_scans as u64,
            });
        }
        // Coefficient memory ceiling (progressive holds ~2 B/px/component
        // of i16 coefficients plus the raw stream copy).
        if let Some(max_mem) = limits.max_memory {
            let total_pixels: u64 = (frame.width as u64) * (frame.height as u64);
            let nc: u64 = frame.components.len() as u64;
            let estimated: u64 =
                total_pixels * (2 * nc) + total_pixels * nc / 64 + data.len() as u64;
            if estimated > max_mem {
                return Err(JpegError::LimitExceeded {
                    what: "estimated decode memory",
                    actual: estimated,
                    limit: max_mem,
                });
            }
        }

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

        // Allocate coefficient buffers (zero-initialized for progressive
        // accumulation). `blocks_x * blocks_y` is header-derived, so both the
        // product and the allocation itself are fallible: at 128 bytes per
        // block this is the largest buffer the decoder holds.
        let coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| {
                let blocks: usize = checked_plane_size(
                    &[ci.blocks_x, ci.blocks_y],
                    "progressive coefficient buffer",
                )?;
                try_filled_vec(blocks, [0i16; 64], "progressive coefficient buffer")
            })
            .collect::<Result<Vec<Vec<[i16; 64]>>>>()?;
        let ac_max_k_bufs: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let blocks: usize =
                    checked_plane_size(&[ci.blocks_x, ci.blocks_y], "progressive AC-max buffer")?;
                try_filled_vec(blocks, 0u8, "progressive AC-max buffer")
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;

        let routines: SimdRoutines = simd::detect();

        Ok(Self {
            raw_data: try_copy_of(data, "progressive input copy")?,
            metadata,
            routines,
            coeff_bufs,
            ac_max_k_bufs,
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
            .map(|ci| -> Result<Vec<u8>> {
                let size: usize = checked_plane_size(
                    &[ci.comp_w, ci.blocks_y, block_size],
                    "progressive component plane",
                )?;
                try_filled_vec(size, 0u8, "progressive component plane")
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;

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
        // P4-144: both are input-sized and used to abort on refusal. This
        // function already returns `Result`, so propagating costs nothing.
        let icc_profile: Option<Vec<u8>> =
            icc::try_reassemble_icc_profile(&self.metadata.icc_chunks)?;
        let exif_data: Option<Vec<u8>> = try_clone_opt(&self.metadata.exif_data, "EXIF metadata")?;

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
        unsafe {
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

            #[allow(unreachable_code)]
            {
                let mut tmp = [0u8; 64];
                (self.routines.idct_islow)(coeffs, quant, &mut tmp);
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

        // Clone Huffman table handles needed for this scan (Arc refcount
        // bumps — the tables themselves are shared, not copied).
        let dc_tables: [Option<alloc::sync::Arc<crate::common::huffman_table::HuffmanTable>>; 4] =
            scan_info.dc_huffman_tables.clone();
        let ac_tables: [Option<alloc::sync::Arc<crate::common::huffman_table::HuffmanTable>>; 4] =
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
            let mut mcu_count: u32 = 0;

            for mcu_y in 0..self.mcus_y {
                for mcu_x in 0..self.mcus_x {
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval as u32)
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
            let mut mcu_count: u32 = 0;

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
                        && mcu_count.is_multiple_of(restart_interval as u32)
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
                        progressive::decode_ac_first_tracked(
                            &mut bit_reader,
                            ac_table_ref.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                            &mut self.ac_max_k_bufs[comp_idx][block_idx],
                        )?;
                    } else {
                        progressive::decode_ac_refine_tracked(
                            &mut bit_reader,
                            ac_table_ref.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                            &mut self.ac_max_k_bufs[comp_idx][block_idx],
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
        let data_size: usize = checked_plane_size(
            &[out_width, out_height],
            "progressive grayscale output image",
        )?;
        let mut data: Vec<u8> = try_reserved_vec(data_size, "progressive grayscale output image")?;
        for y in 0..out_height {
            data.extend_from_slice(&component_planes[0][y * comp_w..y * comp_w + out_width]);
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

        // The progressive YCbCr assembly paths below assume the standard
        // JPEG sampling layout: Y at the frame's max sampling, Cb/Cr at
        // sampling factors ≤ max with matching ratios. Adversarial /
        // malformed streams can pick e.g. Y=h1v2, Cb=h4v1, Cr=h1v1 — all
        // legal per ITU-T T.81 §B.2.2 syntax — which yields component
        // planes whose row strides and counts disagree, and the SIMD
        // upsamplers (`fancy_h1v2`, …) read past the chroma plane. ASan
        // caught this as a heap-buffer-overflow READ of size 16 in
        // fuzz_progressive_decoder (CI run 25215431132). C `libjpeg`
        // accepts only the standard layouts; reject the rest as
        // Unsupported up front rather than tripping unsafe SIMD loads.
        let y_comp = &frame.components[0];
        let cb_comp = &frame.components[1];
        let cr_comp = &frame.components[2];
        if y_comp.horizontal_sampling as usize != self.max_h
            || y_comp.vertical_sampling as usize != self.max_v
            || cb_comp.horizontal_sampling != cr_comp.horizontal_sampling
            || cb_comp.vertical_sampling != cr_comp.vertical_sampling
        {
            return Err(JpegError::Unsupported(format!(
                "non-standard YCbCr sampling Y={}x{}, Cb={}x{}, Cr={}x{} \
                 (max={}x{}); progressive assembly requires Y at full \
                 sampling and Cb/Cr matching",
                y_comp.horizontal_sampling,
                y_comp.vertical_sampling,
                cb_comp.horizontal_sampling,
                cb_comp.vertical_sampling,
                cr_comp.horizontal_sampling,
                cr_comp.vertical_sampling,
                self.max_h,
                self.max_v
            )));
        }

        let y_plane: &[u8] = &component_planes[0];
        let y_width: usize = self.comp_infos[0].comp_w;

        let cb_w: usize = self.comp_infos[1].comp_w;
        let cb_h: usize = self.comp_infos[1].blocks_y * 8;

        let h_factor: usize = self.max_h / cb_comp.horizontal_sampling as usize;
        let v_factor: usize = self.max_v / cb_comp.vertical_sampling as usize;

        // The 4:4:4 fast path below assumes all three components share the
        // same row stride and have at least `out_height` rows of decoded
        // data. h_factor / v_factor are derived from Cb only, so a stream
        // with Cb at max sampling but Y or Cr undersampled (e.g.
        // Y=h1v1, Cb=h1v3, Cr=h1v1 — max_v=3 dominated by Cb) would
        // satisfy `h_factor == 1 && v_factor == 1` while the Y plane is
        // shorter than the image raster. The previous code then
        // panicked at the slice index in `y_plane[y * y_width..]` when
        // `y` exceeded the actual Y plane height. Found via
        // fuzz_progressive_decoder on a 16x16 SOF2 stream with these
        // factors (Fuzz Smoke run 25213799463). Demand all three
        // components be at full sampling for the fast path.
        let y_comp = &frame.components[0];
        let cr_comp = &frame.components[2];
        let all_full_sampling = y_comp.horizontal_sampling as usize == self.max_h
            && y_comp.vertical_sampling as usize == self.max_v
            && cb_comp.horizontal_sampling as usize == self.max_h
            && cb_comp.vertical_sampling as usize == self.max_v
            && cr_comp.horizontal_sampling as usize == self.max_h
            && cr_comp.vertical_sampling as usize == self.max_v;

        if h_factor == 1 && v_factor == 1 && all_full_sampling {
            // 4:4:4: no upsampling needed
            let data_size: usize =
                checked_plane_size(&[out_width, out_height, bpp], "progressive output image")?;
            let mut data: Vec<u8> = try_filled_vec(data_size, 0u8, "progressive output image")?;
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
                xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                width: out_width,
                height: out_height,
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
            // Upsample chroma
            let alloc_size: usize = checked_plane_size(
                &[full_width, full_height],
                "progressive upsampled chroma plane",
            )?;
            let mut cb_full: Vec<u8> =
                try_filled_vec(alloc_size, 0u8, "progressive upsampled chroma plane")?;
            let mut cr_full: Vec<u8> =
                try_filled_vec(alloc_size, 0u8, "progressive upsampled chroma plane")?;

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
                // S411: C uses int_upsample (box filter), not fancy interpolation.
                upsample_generic_nearest(
                    &component_planes[1],
                    cb_w,
                    cb_h,
                    &mut cb_full,
                    full_width,
                    h_factor,
                    1,
                );
                upsample_generic_nearest(
                    &component_planes[2],
                    cb_w,
                    cb_h,
                    &mut cr_full,
                    full_width,
                    h_factor,
                    1,
                );
            } else if h_factor == 1 && v_factor == 4 {
                // S441: C uses int_upsample (box filter), not fancy interpolation.
                upsample_generic_nearest(
                    &component_planes[1],
                    cb_w,
                    cb_h,
                    &mut cb_full,
                    full_width,
                    1,
                    v_factor,
                );
                upsample_generic_nearest(
                    &component_planes[2],
                    cb_w,
                    cb_h,
                    &mut cr_full,
                    full_width,
                    1,
                    v_factor,
                );
            } else {
                return Err(JpegError::Unsupported(format!(
                    "subsampling {}x{} not supported in progressive output",
                    h_factor, v_factor
                )));
            }

            let data_size: usize =
                checked_plane_size(&[out_width, out_height, bpp], "progressive output image")?;
            let mut data: Vec<u8> = try_filled_vec(data_size, 0u8, "progressive output image")?;
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
                xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
                iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
                width: out_width,
                height: out_height,
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
        let data_size: usize = checked_plane_size(
            &[out_width, out_height, bpp],
            "progressive CMYK output image",
        )?;
        let mut data: Vec<u8> = try_filled_vec(data_size, 0u8, "progressive CMYK output image")?;

        for y in 0..out_height {
            for x in 0..out_width {
                for c in 0..4 {
                    let comp_w: usize = self.comp_infos[c].comp_w;
                    data[y * out_width * bpp + x * bpp + c] = component_planes[c][y * comp_w + x];
                }
            }
        }

        Ok(Image {
            xmp_data: try_clone_opt(&self.metadata.xmp_data, "XMP metadata")?,
            iptc_data: try_clone_opt(&self.metadata.iptc_data, "IPTC metadata")?,
            width: out_width,
            height: out_height,
            pixel_format: PixelFormat::Cmyk,
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

        #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
        {
            return crate::simd::wasm32::upsample::wasm_fancy_upsample_h2v2(
                input, in_width, in_height, output, out_width,
            );
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
        // Defensive bounds: malformed progressive JPEGs can produce a chroma
        // plane shorter than `in_height * in_width` if blocks_y was clobbered
        // mid-stream. Clamp every per-row slice to the actual buffer length so
        // the upsample falls back to repeating the current row at boundaries
        // instead of panicking. Discovered via fuzz_progressive_decoder on a
        // double-SOF input that left coeff_bufs sized to the first SOF and
        // in_height/in_width sized to the second.
        let actual_rows: usize = input.len().checked_div(in_width).unwrap_or(0);
        let safe_height: usize = in_height.min(actual_rows);
        for y in 0..safe_height {
            let cur_row = &input[y * in_width..(y + 1) * in_width];
            let above = if y > 0 {
                &input[(y - 1) * in_width..y * in_width]
            } else {
                cur_row
            };
            let below = if y + 1 < safe_height {
                &input[(y + 1) * in_width..(y + 2) * in_width]
            } else {
                cur_row
            };

            let out_y_top: usize = y * 2;
            let out_y_bot: usize = y * 2 + 1;

            // Ordered dither bias: top=1, bottom=2 (matches C jdsample.c)
            for i in 0..in_width {
                output[out_y_top * out_width + i] =
                    ((3 * cur_row[i] as u16 + above[i] as u16 + 1) >> 2) as u8;
                output[out_y_bot * out_width + i] =
                    ((3 * cur_row[i] as u16 + below[i] as u16 + 2) >> 2) as u8;
            }
        }
        // When safe_height < in_height the tail rows were skipped. Since
        // P4-136 the caller hands over a zero-initialized buffer, so those rows
        // already read as zero and this fill is redundant for that caller. It
        // stays because it states the guarantee locally rather than relying on
        // one: a caller that ever passes a reused buffer gets zeros here, not
        // the previous image's rows.
        let written: usize = safe_height * 2 * out_width;
        let cap: usize = in_height * 2 * out_width;
        if written < cap && cap <= output.len() {
            output[written..cap].fill(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Smallest progressive fixture in the tree (522 bytes, 4:2:0). Embedded
    /// rather than read at runtime so the test needs no filesystem access —
    /// Miri only reaches files with `-Zmiri-disable-isolation`, and a test
    /// that silently depends on a CI flag is one refactor away from being
    /// skipped without anyone noticing.
    const PROGRESSIVE_16X16_420: &[u8] =
        include_bytes!("../../tests/fixtures/blue_16x16_420_prog.jpg");

    /// P4-136 criterion 6: put the progressive output path under Miri.
    ///
    /// CI runs `cargo miri test --lib` (`ci.yml`), so a unit test *in this
    /// module* is what gets Miri onto this code — the `tests/` progressive
    /// suites are integration targets Miri never builds. That gap is why the
    /// `set_len`-on-uninitialised-`Vec` pattern this item removed survived a
    /// green Miri job.
    ///
    /// Decode only, from the smallest progressive fixture available: Miri
    /// interprets every operation, so anything this test does that is not the
    /// path under audit is paid for at interpreter speed. 16x16 4:2:0 still
    /// walks all of it — multi-scan coefficient accumulation, IDCT into the
    /// plane buffers, chroma upsampling (4:2:0 makes `full_width`/`full_height`
    /// differ from the output geometry), and colour conversion.
    #[test]
    fn progressive_output_path_is_miri_covered() {
        const WIDTH: usize = 16;
        const HEIGHT: usize = 16;

        let mut decoder: ProgressiveDecoder =
            ProgressiveDecoder::new(PROGRESSIVE_16X16_420).expect("progressive decode");
        assert!(decoder.has_multiple_scans());

        // Output after every scan, not just at the end: the per-scan
        // reconstruction is what allocates and fills the plane buffers, and
        // it is the path an early return would leave partly written.
        let mut outputs: usize = 0;
        loop {
            let image: Image = decoder.output().expect("scan output");
            assert_eq!(image.width, WIDTH);
            assert_eq!(image.height, HEIGHT);
            assert_eq!(image.data.len(), WIDTH * HEIGHT * 3);
            outputs += 1;
            if !decoder.consume_input().expect("consume scan") {
                break;
            }
        }
        assert!(outputs > 1, "expected a multi-scan walk, got {outputs}");

        let final_image: Image = decoder.finish().expect("finish");
        assert_eq!(final_image.data.len(), WIDTH * HEIGHT * 3);
    }

    /// P4-136 criterion 4: a size the allocator cannot serve must surface as
    /// a recoverable error. Before this, `vec![0u8; n]` aborted the process,
    /// so a header declaring more memory than the machine has was an
    /// uncatchable denial of service.
    ///
    /// `isize::MAX` elements is 8 EiB — no allocator on any 64-bit target can
    /// serve it, so this exercises the refusal path deterministically without
    /// depending on how much memory the test host happens to have.
    ///
    /// 64-bit only, and not for convenience: at 32-bit width `isize::MAX` is
    /// 2 GiB, which a host *may* actually serve. Asserting refusal there would
    /// be flaky, and letting it succeed would make the armv7 and wasm legs
    /// zero 2 GiB under emulation. Those legs cover the arithmetic rejection
    /// paths below instead, which are deterministic at 32-bit width.
    ///
    /// Ignored under Miri: Miri's allocator reports an unservable request as
    /// "resource exhaustion" and aborts the interpreter instead of returning
    /// the null that `try_reserve_exact` turns into `Err`, so the refusal path
    /// is not observable there. The rest of this module still runs under Miri
    /// — see `progressive_output_path_is_miri_covered`.
    #[cfg(target_pointer_width = "64")]
    #[cfg_attr(
        miri,
        ignore = "Miri aborts on unservable allocations instead of returning null"
    )]
    #[test]
    fn allocator_refusal_is_an_error_not_an_abort() {
        let err: JpegError =
            try_filled_vec(isize::MAX as usize, 0u8, "test plane").expect_err("must refuse");
        assert!(
            matches!(err, JpegError::AllocationFailed { what, bytes }
                if what == "test plane" && bytes == isize::MAX as u64),
            "expected AllocationFailed, got {err:?}"
        );
    }

    /// A `len` whose *byte* count cannot be expressed is a geometry limit,
    /// not a failed allocation: the machine was never asked to allocate.
    /// Keeping them distinct is what makes the error actionable.
    #[test]
    fn byte_count_overflow_reports_the_geometry_limit() {
        // 128 bytes per element, so half of `isize::MAX` elements already
        // overflows the byte product on every pointer width.
        let len: usize = (isize::MAX as usize) / 2 + 1;
        let err: JpegError = try_filled_vec::<[i16; 64]>(len, [0i16; 64], "test coefficients")
            .expect_err("must refuse");
        assert!(
            matches!(err, JpegError::LimitExceeded { what, .. } if what == "test coefficients"),
            "expected LimitExceeded, got {err:?}"
        );
    }

    /// A size the allocator *can* serve still yields a zero-filled buffer of
    /// exactly the requested length — the fallible path must not change the
    /// contract the IDCT relies on.
    #[test]
    fn successful_allocation_is_zero_filled_and_exact() {
        let buf: Vec<u8> = try_filled_vec(1024, 0u8, "test plane").expect("1 KiB must succeed");
        assert_eq!(buf.len(), 1024);
        assert!(buf.iter().all(|&b| b == 0));

        let blocks: Vec<[i16; 64]> =
            try_filled_vec(16, [0i16; 64], "test coefficients").expect("2 KiB must succeed");
        assert_eq!(blocks.len(), 16);
        assert!(blocks.iter().all(|block| block.iter().all(|&c| c == 0)));
    }

    #[test]
    fn copy_of_input_preserves_contents() {
        let src: [u8; 5] = [0xFF, 0xD8, 0xFF, 0xC2, 0x00];
        assert_eq!(
            try_copy_of(&src, "test copy").expect("small copy must succeed"),
            src
        );
        assert!(try_copy_of(&[], "test copy")
            .expect("empty copy must succeed")
            .is_empty());
    }

    /// P4-136 criterion 5: geometry that fits `usize` on a 64-bit target and
    /// overflows it on a 32-bit one must be *rejected*, not wrapped.
    ///
    /// The shape is real: a 65535x65535 progressive frame yields per-component
    /// block counts whose product times the 128-byte block stride exceeds
    /// 32-bit `usize`. On 64-bit the same product is representable, so the
    /// assertion has to be pointer-width aware to say anything true on both.
    ///
    /// Only the 32-bit arm calls `try_filled_vec`, and only because there the
    /// answer is decided by arithmetic *before* any allocation is attempted.
    /// The 64-bit arm stops at the arithmetic deliberately: calling it would
    /// ask the allocator for 8 GiB, which is a memory hog natively and hangs
    /// the Miri leg, and it would prove nothing this test is about.
    #[test]
    fn thirty_two_bit_geometry_overflow_is_rejected() {
        // 8192 x 8192 blocks = 2^26 blocks; x 128 bytes/block = 2^33 bytes.
        let blocks: usize =
            checked_plane_size(&[8192, 8192], "test blocks").expect("block count fits both widths");
        assert_eq!(blocks, 1 << 26);

        let bytes: Option<usize> = blocks.checked_mul(core::mem::size_of::<[i16; 64]>());

        #[cfg(target_pointer_width = "32")]
        {
            assert!(bytes.is_none(), "2^33 bytes must not fit a 32-bit usize");
            assert!(
                matches!(
                    try_filled_vec::<[i16; 64]>(blocks, [0i16; 64], "test coefficients"),
                    Err(JpegError::LimitExceeded { .. })
                ),
                "rejection must come from the arithmetic, before any allocation"
            );
        }

        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            bytes,
            Some(1usize << 33),
            "the same geometry is expressible at 64-bit width"
        );
    }

    /// Width-independent companion to the above: a product that overflows
    /// `usize` itself is rejected on *every* target.
    #[test]
    fn plane_size_overflow_is_rejected_on_every_pointer_width() {
        let half: usize = (isize::MAX as usize) / 2 + 1;
        assert!(matches!(
            checked_plane_size(&[half, 4], "test plane"),
            Err(JpegError::LimitExceeded { .. })
        ));
        // Exactly `isize::MAX` is the last accepted value; one more is not.
        assert_eq!(
            checked_plane_size(&[isize::MAX as usize, 1], "test plane").expect("boundary is legal"),
            isize::MAX as usize
        );
        assert!(matches!(
            checked_plane_size(&[isize::MAX as usize / 2 + 1, 2], "test plane"),
            Err(JpegError::LimitExceeded { .. })
        ));
    }
}
