use super::{Decoder, Image, ImageInfo, JpegInfo};
use crate::common::error::{JpegError, Result};
use crate::common::types::{
    ColorSpace, DctMethod, DecodeLimits, DensityInfo, FrameHeader, MarkerSaveConfig, PixelFormat,
    SavedMarker, ScalingFactor, Subsampling,
};
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::simd;
use alloc::{boxed::Box, format, vec, vec::Vec};

impl ImageInfo {
    /// EXIF orientation (1-8) parsed from `exif_data`, for parity with
    /// [`Image::exif_orientation`] and [`Decoder::exif_orientation`] on
    /// the caller-buffer decode path (issue #391).
    pub fn exif_orientation(&self) -> Option<u8> {
        self.exif_data
            .as_deref()
            .and_then(crate::common::exif::parse_orientation)
    }
}

/// One-call header probe: dimensions, coding mode, subsampling,
/// colorspace, and metadata presence without decoding any pixels
/// (issue #386).
///
/// Wraps `Decoder::new` (marker parse only), so it never allocates
/// pixel buffers; cost is linear in the compressed size for multi-scan
/// streams (scan boundaries are walked byte-wise), far below a decode. For decoding afterwards, create a
/// [`Decoder`] — the header work is repeated, but it is negligible next
/// to the pixel decode.
///
/// ```
/// # let jpeg = libjpeg_turbo_rs::compress(&[128; 48], 4, 4,
/// #     libjpeg_turbo_rs::PixelFormat::Rgb, 90,
/// #     libjpeg_turbo_rs::Subsampling::S444).unwrap();
/// let info = libjpeg_turbo_rs::probe(&jpeg)?;
/// assert_eq!((info.width, info.height), (4, 4));
/// assert!(!info.progressive);
/// # Ok::<(), libjpeg_turbo_rs::JpegError>(())
/// ```
pub fn probe(jpeg: &[u8]) -> Result<JpegInfo> {
    let decoder: Decoder = Decoder::new(jpeg)?;
    let metadata: &crate::decode::marker::JpegMetadata = &decoder.metadata;
    let frame: &FrameHeader = &metadata.frame;
    Ok(JpegInfo {
        width: frame.width(),
        height: frame.height(),
        components: frame.components.len(),
        precision: frame.precision,
        progressive: frame.is_progressive,
        lossless: frame.is_lossless,
        arithmetic: metadata.is_arithmetic,
        subsampling: decoder.jpeg_subsampling(),
        color_space: decoder.jpeg_color_space(),
        exif_orientation: decoder.exif_orientation(),
        has_exif: metadata.exif_data.is_some(),
        has_icc: !metadata.icc_chunks.is_empty(),
        has_xmp: metadata.xmp_data.is_some(),
        has_iptc: metadata.iptc_data.is_some(),
        comment: metadata.comment.clone(),
        density: metadata.density,
    })
}

impl Image {
    /// Returns the ICC color profile embedded in this JPEG, if any.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Returns the raw EXIF TIFF data, if present.
    pub fn exif_data(&self) -> Option<&[u8]> {
        self.exif_data.as_deref()
    }

    /// Raw XMP packet (APP1 `http://ns.adobe.com/xap/1.0/`), with
    /// Extended XMP chunks reassembled in offset order and appended.
    /// Returns the bytes, not a parsed tree (issue #358).
    ///
    /// Reassembly requires exact contiguous coverage of the declared
    /// length; a gapped, overlapping, or partial chunk set degrades to
    /// the standard packet alone. Extension chunks arriving without a
    /// standard packet are dropped. The extension set is selected by
    /// the first extension chunk's GUID rather than the standard
    /// packet's `xmpNote:HasExtendedXMP` — parsing the RDF to read that
    /// belongs in a consumer crate, not the codec.
    /// TODO(#358): revisit if a consumer needs GUID-authoritative
    /// selection.
    pub fn xmp_data(&self) -> Option<&[u8]> {
        self.xmp_data.as_deref()
    }

    /// Raw IPTC IIM payload extracted from the APP13 Photoshop IRB
    /// (resource 0x0404). Returns the bytes, not parsed datasets.
    pub fn iptc_data(&self) -> Option<&[u8]> {
        self.iptc_data.as_deref()
    }

    /// Parses and returns the EXIF orientation tag (1-8), if present.
    pub fn exif_orientation(&self) -> Option<u8> {
        self.exif_data
            .as_ref()
            .and_then(|d| crate::common::exif::parse_orientation(d))
    }

    /// Returns all saved markers (APP and COM) collected during decoding.
    ///
    /// Only populated when the decoder was configured with `save_markers()`.
    pub fn markers(&self) -> &[SavedMarker] {
        &self.saved_markers
    }

    /// The raw pixel bytes — row-major, `width * height *
    /// pixel_format.bytes_per_pixel()` long (issue #386). Prefer this
    /// over reaching into `.data` directly.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the image and hand back the owned pixel buffer, dropping
    /// the metadata (issue #386).
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    /// Reorient the pixels so the image displays upright according to its
    /// own EXIF orientation tag (issue #391). Identity when there is no
    /// EXIF, no orientation tag, or orientation 1. Convenience over
    /// [`Image::apply_orientation_value`] using [`Image::exif_orientation`].
    ///
    /// **Not idempotent**: the tag in `exif_data` is left at its original
    /// value, so calling this twice reorients twice, and re-encoding with
    /// the returned `exif_data` re-applies the rotation on display —
    /// strip or rewrite the metadata when re-encoding.
    ///
    /// For a lossless alternative that never leaves the DCT domain, map
    /// the tag with [`crate::TransformOp::from_exif_orientation`] and run
    /// [`crate::transform()`] on the compressed bytes instead.
    #[must_use]
    pub fn apply_orientation(self) -> Self {
        let orientation: u8 = self.exif_orientation().unwrap_or(1);
        self.apply_orientation_value(orientation)
    }

    /// Reorient the pixels for an explicitly supplied EXIF orientation
    /// value (1-8) — the primitive behind [`Image::apply_orientation`],
    /// for callers whose orientation comes from elsewhere (XMP
    /// `tiff:Orientation`, a container, a user override). Identity for
    /// 1 and out-of-range values.
    ///
    /// Orientations 5-8 swap `width`/`height`. Works for every pixel
    /// format: the remap moves whole `bytes_per_pixel` units, including
    /// packed `Rgb565`.
    #[must_use]
    pub fn apply_orientation_value(mut self, orientation: u8) -> Self {
        if !(2..=8).contains(&orientation) {
            return self;
        }
        let bpp: usize = self.pixel_format.bytes_per_pixel();
        let (w, h): (usize, usize) = (self.width, self.height);
        // Structural invariant, not asserted elsewhere: an all-pub-fields
        // Image can be hand-built undersized (or with dims whose product
        // overflows); bail rather than index OOB or panic on the multiply.
        let Some(expected) = w.checked_mul(h).and_then(|n| n.checked_mul(bpp)) else {
            return self;
        };
        if self.data.len() < expected {
            return self;
        }
        let swaps_dims: bool = (5..=8).contains(&orientation);
        let (dst_w, dst_h): (usize, usize) = if swaps_dims { (h, w) } else { (w, h) };
        let mut out: Vec<u8> = vec![0u8; self.data.len()];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                // EXIF tag 0x0112: source coordinates that land at
                // (dx, dy) of the upright image.
                let (sx, sy): (usize, usize) = match orientation {
                    2 => (w - 1 - dx, dy),         // mirrored horizontally
                    3 => (w - 1 - dx, h - 1 - dy), // rotated 180
                    4 => (dx, h - 1 - dy),         // mirrored vertically
                    5 => (dy, dx),                 // transposed
                    6 => (dy, h - 1 - dx),         // rotate 90 CW to fix
                    7 => (w - 1 - dy, h - 1 - dx), // transverse
                    _ => (w - 1 - dy, dx),         // 8: rotate 270 CW to fix
                };
                let src: usize = (sy * w + sx) * bpp;
                let dst: usize = (dy * dst_w + dx) * bpp;
                out[dst..dst + bpp].copy_from_slice(&self.data[src..src + bpp]);
            }
        }
        self.data = out;
        self.width = dst_w;
        self.height = dst_h;
        self
    }
}

impl<'a> Decoder<'a> {
    /// Parse the JPEG headers in `data` and return a configurable
    /// decoder. No entropy decoding and no pixel work happen here, but
    /// for progressive and non-interleaved streams locating the scan
    /// boundaries walks the entropy bytes, so worst-case probe time
    /// scales with the compressed input length (still far cheaper than
    /// a decode). Suitable as a probe (dimensions,
    /// [`Decoder::exif_orientation`], ICC/EXIF/XMP metadata) even when
    /// no decode follows. Uses [`DecodeLimits::default`]; use
    /// [`Decoder::new_with_limits`] to bound resource use differently.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Self::new_with_limits(data, DecodeLimits::default())
    }

    /// Like [`Self::new`], but the limits apply from marker parsing
    /// onward: a `max_scans` tighter than the parse default bounds
    /// ScanInfo buffering during the header walk itself, not just at
    /// decode time (issue #355).
    pub fn new_with_limits(data: &'a [u8], limits: DecodeLimits) -> Result<Self> {
        let mut reader = MarkerReader::new(data);
        reader.set_scan_cap(limits.max_scans);
        let mut metadata = reader.read_markers()?;
        // MJPEG frames may omit Huffman tables; provide standard defaults
        // (JPEG spec section K.3), matching C libjpeg-turbo's std_huff_tables().
        Self::fill_default_huffman_tables(&mut metadata);
        let routines = simd::detect();
        Ok(Self {
            metadata,
            raw_data: data,
            routines,
            output_format: None,
            scale: ScalingFactor::default(),
            lenient: false,
            crop_x: None,
            crop_width: None,
            crop_y: None,
            crop_height: None,
            stop_on_warning: false,
            limits,
            fast_upsample: false,
            fast_dct: false,
            dct_method: DctMethod::IsLow,
            block_smoothing: false,
            output_colorspace: None,
            dither_565: false,
            merged_upsample: false,
            marker_processors: alloc::collections::BTreeMap::new(),
            resync_strategy: core::cell::RefCell::new(None),
            prefilled_baseline_planes: core::cell::RefCell::new(None),
        })
    }

    /// Fill in standard JPEG Huffman tables (K.3) for any unset table slot.
    ///
    /// Mirrors libjpeg-turbo's `jinit_huff_decoder` → `std_huff_tables`: every
    /// DC/AC slot left NULL after marker parsing gets the standard table. This
    /// covers MJPEG frames (no DHT at all), partially-defined streams that
    /// reference a never-emitted slot in SOS (real-world C-decodable inputs
    /// found by `fuzz_decode_diff_c`), and standard JFIF inputs (a no-op
    /// because the per-slot fill below only writes `None` slots).
    ///
    /// The four Annex K tables are process-global (`std_huffman_tables`,
    /// built once behind a `OnceLock`) and shared by `Arc` clone — filling
    /// a slot is a refcount bump, not a 4 KB table build (issue #351).
    pub(super) fn fill_default_huffman_tables(metadata: &mut JpegMetadata) {
        use crate::common::huffman_table::std_huffman_tables;

        // libjpeg-turbo only auto-fills standard tables for the
        // baseline (sequential Huffman) decoder via `jinit_huff_decoder`
        // → `std_huff_tables`. The progressive entropy decoder
        // (`jinit_phuff_decoder`) does **not** auto-fill — a progressive
        // SOS that references an unset table slot must keep returning a
        // "missing Huffman table" error to match djpeg's behaviour
        // (codex P2: removing this gate would have Rust accept inputs
        // djpeg rejects, the opposite of the drop-in regression we just
        // fixed).
        if metadata.frame.is_progressive {
            return;
        }

        // Fill missing slots in the *final metadata snapshot* (baseline
        // single-scan path reads `metadata.dc_huffman_tables` directly)
        // AND in each per-scan snapshot. The per-scan fill must use the
        // standard table — never the final-metadata table — because a
        // later DHT can redefine the same slot mid-stream
        // (non-interleaved baseline emits one DHT per scan); copying a
        // late definition back into an earlier scan silently alters the
        // bytes that scan was supposed to decode against.
        let [std_dc_lum, std_dc_chr, std_ac_lum, std_ac_chr] = std_huffman_tables();
        let std_dc = [std_dc_lum, std_dc_chr];
        let std_ac = [std_ac_lum, std_ac_chr];

        for i in 0..2 {
            if metadata.dc_huffman_tables[i].is_none() {
                metadata.dc_huffman_tables[i] = Some(std_dc[i].clone());
            }
            if metadata.ac_huffman_tables[i].is_none() {
                metadata.ac_huffman_tables[i] = Some(std_ac[i].clone());
            }
        }

        for scan in &mut metadata.scans {
            for i in 0..2 {
                if scan.dc_huffman_tables[i].is_none() {
                    scan.dc_huffman_tables[i] = Some(std_dc[i].clone());
                }
                if scan.ac_huffman_tables[i].is_none() {
                    scan.ac_huffman_tables[i] = Some(std_ac[i].clone());
                }
            }
        }
    }

    /// Create a decoder for a body-only abbreviated stream using preloaded tables.
    ///
    /// The `body_data` must contain SOF and SOS markers but may omit DQT and DHT.
    /// Tables from `tables` are injected into the decoder's internal state before decoding.
    ///
    /// Matches libjpeg-turbo's abbreviated compressed data datastream handling:
    /// use `read_header()` to get a `TablesOnlyState`, then this function to decode
    /// the body-only stream.
    pub fn new_with_tables(
        body_data: &'a [u8],
        tables: &crate::api::abbreviated::TablesOnlyState,
    ) -> Result<Self> {
        let mut reader = MarkerReader::new(body_data);
        let mut metadata = reader.read_markers()?;

        // Inject preloaded quant tables for any slot the body didn't define
        for i in 0..4 {
            if metadata.quant_tables[i].is_none() {
                if let Some(ref qt) = tables.quant_tables[i] {
                    metadata.quant_tables[i] = Some(qt.clone());
                }
            }
        }

        // Inject preloaded DC Huffman tables
        for i in 0..4 {
            if metadata.dc_huffman_tables[i].is_none() {
                if let Some(ref ht) = tables.dc_huffman_tables[i] {
                    metadata.dc_huffman_tables[i] = Some(ht.clone());
                }
            }
        }

        // Inject preloaded AC Huffman tables
        for i in 0..4 {
            if metadata.ac_huffman_tables[i].is_none() {
                if let Some(ref ht) = tables.ac_huffman_tables[i] {
                    metadata.ac_huffman_tables[i] = Some(ht.clone());
                }
            }
        }

        // Propagate injected tables into scan snapshots
        for scan in &mut metadata.scans {
            for i in 0..4 {
                if scan.dc_huffman_tables[i].is_none() && metadata.dc_huffman_tables[i].is_some() {
                    scan.dc_huffman_tables[i] = metadata.dc_huffman_tables[i].clone();
                }
                if scan.ac_huffman_tables[i].is_none() && metadata.ac_huffman_tables[i].is_some() {
                    scan.ac_huffman_tables[i] = metadata.ac_huffman_tables[i].clone();
                }
            }
        }

        // Carry arithmetic coding state from tables
        if tables.is_arithmetic {
            metadata.is_arithmetic = true;
            metadata.arith_dc_params = tables.arith_dc_params;
            metadata.arith_ac_params = tables.arith_ac_params;
        }

        let routines = simd::detect();
        Ok(Self {
            metadata,
            raw_data: body_data,
            routines,
            output_format: None,
            scale: ScalingFactor::default(),
            lenient: false,
            crop_x: None,
            crop_width: None,
            crop_y: None,
            crop_height: None,
            stop_on_warning: false,
            limits: DecodeLimits::default(),
            fast_upsample: false,
            fast_dct: false,
            dct_method: DctMethod::IsLow,
            block_smoothing: false,
            output_colorspace: None,
            dither_565: false,
            merged_upsample: false,
            marker_processors: alloc::collections::BTreeMap::new(),
            resync_strategy: core::cell::RefCell::new(None),
            prefilled_baseline_planes: core::cell::RefCell::new(None),
        })
    }

    /// EXIF orientation (1-8) from the already-parsed headers, without
    /// any pixel decode (issue #391). `Decoder::new` is a header parse
    /// only, so probing a camera JPEG's orientation costs no decoding:
    ///
    /// ```
    /// # let jpeg = libjpeg_turbo_rs::compress(&[128; 48], 4, 4,
    /// #     libjpeg_turbo_rs::PixelFormat::Rgb, 90,
    /// #     libjpeg_turbo_rs::Subsampling::S444).unwrap();
    /// let decoder = libjpeg_turbo_rs::Decoder::new(&jpeg)?;
    /// // This fixture carries no EXIF; a camera JPEG returns Some(1..=8).
    /// // Map 2-8 with TransformOp::from_exif_orientation (lossless, DCT
    /// // domain) or Image::apply_orientation (pixels).
    /// assert_eq!(decoder.exif_orientation(), None);
    /// # Ok::<(), libjpeg_turbo_rs::JpegError>(())
    /// ```
    pub fn exif_orientation(&self) -> Option<u8> {
        self.metadata
            .exif_data
            .as_deref()
            .and_then(crate::common::exif::parse_orientation)
    }

    /// The parsed frame header: dimensions, per-component sampling,
    /// precision, progressive/lossless flags. Available immediately
    /// after [`Decoder::new`], before any pixel decode.
    pub fn header(&self) -> &FrameHeader {
        &self.metadata.frame
    }

    /// Pixel density parsed from the JFIF APP0 marker (or
    /// `DensityInfo::default()` if no JFIF was present).
    /// Mirrors stock libjpeg's `cinfo.density_unit / X_density /
    /// Y_density` exposed after `jpeg_read_header`.
    pub fn density(&self) -> &DensityInfo {
        &self.metadata.density
    }

    /// Whether the source carried a JFIF APP0 marker (regardless of
    /// the density values it contained). Mirrors stock libjpeg's
    /// `cinfo.saw_JFIF_marker`.
    pub fn saw_jfif_marker(&self) -> bool {
        self.metadata.saw_jfif_marker
    }

    /// JFIF version bytes from the APP0 marker. Returns `(0, 0)` if
    /// `saw_jfif_marker()` is false.
    pub fn jfif_version(&self) -> (u8, u8) {
        (
            self.metadata.jfif_major_version,
            self.metadata.jfif_minor_version,
        )
    }

    /// Whether the source uses arithmetic entropy coding (SOF9 / SOF10 / SOF11).
    ///
    /// Returns `true` for arithmetic-coded streams and `false` for
    /// Huffman-coded streams (SOF0 / SOF1 / SOF2 / SOF3).  Mirrors stock
    /// libjpeg's `cinfo.arith_code` populated by `jpeg_read_header`.
    pub fn is_arithmetic(&self) -> bool {
        self.metadata.is_arithmetic
    }

    /// Set the desired output pixel format.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.output_format = Some(format);
    }

    /// Set the decompression scaling factor (e.g., 1/2, 1/4, 1/8).
    pub fn set_scale(&mut self, scale: ScalingFactor) {
        self.scale = scale;
    }

    /// Enable lenient mode: continue decoding on errors, filling corrupt areas with gray.
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
    }

    /// Set horizontal crop region. Offsets are auto-aligned to iMCU boundaries.
    pub fn set_crop(&mut self, x: usize, width: usize) {
        self.crop_x = Some(x);
        self.crop_width = Some(width);
    }

    /// Set only the vertical crop range, leaving any horizontal crop
    /// untouched. MCU rows fully outside the range skip IDCT during
    /// decoding (issue #383: backs `StreamingDecoder::skip_scanlines`).
    pub fn set_crop_y(&mut self, y: usize, height: usize) {
        self.crop_y = Some(y);
        self.crop_height = Some(height);
    }

    /// Rows a decode will actually emit: the scaled frame height, except
    /// on the 12-bit and lossless paths, which bypass scaled decode (see
    /// `output_buffer_size`). `crop_y`/`crop_height` are interpreted in
    /// these output rows, matching C, where `jpeg_skip_scanlines`
    /// validates against `cinfo->output_height` (jdapistd.c).
    pub fn output_height(&self) -> usize {
        let frame = self.header();
        let h: usize = frame.height as usize;
        if frame.precision != 12 && !frame.is_lossless {
            self.scale.scale_dim(h)
        } else {
            h
        }
    }

    /// Set full crop region (horizontal + vertical).
    /// MCU rows outside the vertical range will skip IDCT during decoding.
    pub fn set_crop_region(&mut self, x: usize, y: usize, width: usize, height: usize) {
        self.crop_x = Some(x);
        self.crop_width = Some(width);
        self.crop_y = Some(y);
        self.crop_height = Some(height);
    }

    /// Treat warnings as fatal errors.
    pub fn set_stop_on_warning(&mut self, stop: bool) {
        self.stop_on_warning = stop;
    }

    /// Set maximum allowed image size in pixels. Reject images exceeding this.
    pub fn set_max_pixels(&mut self, limit: usize) {
        self.limits.max_pixels = limit as u64;
    }

    /// Set maximum memory usage in bytes.
    pub fn set_max_memory(&mut self, limit: usize) {
        self.limits.max_memory = Some(limit as u64);
    }

    /// Set maximum number of progressive scans before error.
    pub fn set_scan_limit(&mut self, limit: u32) {
        self.limits.max_scans = limit as usize;
    }

    /// Configure all decoder resource limits at once (issue #355).
    ///
    /// Defaults ([`DecodeLimits::default`]) are permissive — they accept
    /// everything djpeg accepts in the corpus gates while rejecting the
    /// pathological corner (a header-only 65535x65535 SOF exceeds the
    /// default `max_pixels` before any plane allocation). Use
    /// [`DecodeLimits::strict`] for zune-like tight bounds. Exceeding a
    /// limit is a typed [`JpegError::LimitExceeded`], never a panic.
    /// Note on `max_scans`: marker parsing happens in the constructor,
    /// bounded by the construction-time cap (`new` uses the 8192
    /// default; `new_with_limits` threads the caller's value, higher or
    /// lower). Setting a different `max_scans` here affects only the
    /// decode-time check — a stream needing a larger parse cap must be
    /// constructed with `new_with_limits`.
    pub fn set_limits(&mut self, limits: DecodeLimits) {
        self.limits = limits;
    }

    /// The currently configured resource limits.
    #[must_use]
    pub fn limits(&self) -> &DecodeLimits {
        &self.limits
    }

    /// Enable or disable fast (nearest-neighbor) upsampling.
    pub fn set_fast_upsample(&mut self, fast: bool) {
        self.fast_upsample = fast;
    }

    /// Get the JPEG color space detected from the file header.
    ///
    /// Maps component count and Adobe APP14 marker to a `ColorSpace` value,
    /// matching libjpeg-turbo's `jpeg_color_space` behavior.
    pub fn jpeg_color_space(&self) -> ColorSpace {
        self.detect_color_space()
    }

    /// Get the chroma subsampling detected from the SOF component sampling factors.
    ///
    /// Compares luma vs chroma sampling factors to determine the standard
    /// subsampling mode. Returns `Subsampling::Unknown` for grayscale
    /// (caller should check component count and map to TJSAMP_GRAY=3).
    pub fn jpeg_subsampling(&self) -> Subsampling {
        let frame = &self.metadata.frame;
        let num_components = frame.components.len();
        if num_components == 1 {
            // Grayscale: no chroma subsampling concept. Return Unknown so
            // TjHandle can map to TJSAMP_GRAY=3 explicitly.
            return Subsampling::Unknown;
        }
        if num_components < 3 {
            return Subsampling::Unknown;
        }
        let luma_h: u8 = frame.components[0].horizontal_sampling;
        let luma_v: u8 = frame.components[0].vertical_sampling;
        let chroma_h: u8 = frame.components[1].horizontal_sampling;
        let chroma_v: u8 = frame.components[1].vertical_sampling;
        if chroma_h == 0 || chroma_v == 0 {
            return Subsampling::Unknown;
        }
        let h_ratio: u8 = luma_h / chroma_h;
        let v_ratio: u8 = luma_v / chroma_v;
        match (h_ratio, v_ratio) {
            (1, 1) => Subsampling::S444,
            (2, 1) => Subsampling::S422,
            (2, 2) => Subsampling::S420,
            (1, 2) => Subsampling::S440,
            (4, 1) => Subsampling::S411,
            (1, 4) => Subsampling::S441,
            (4, 2) => Subsampling::S410,
            (2, 4) => Subsampling::S24,
            _ => Subsampling::Unknown,
        }
    }

    /// Enable or disable fast DCT for decoding.
    pub fn set_fast_dct(&mut self, fast: bool) {
        self.fast_dct = fast;
        if fast {
            self.dct_method = DctMethod::IsFast;
        } else if self.dct_method == DctMethod::IsFast {
            self.dct_method = DctMethod::IsLow;
        }
    }

    /// Set the DCT/IDCT method for decoding.
    pub fn set_dct_method(&mut self, method: DctMethod) {
        self.dct_method = method;
    }

    /// Enable or disable inter-block smoothing.
    pub fn set_block_smoothing(&mut self, smooth: bool) {
        self.block_smoothing = smooth;
    }

    /// Override the output color space.
    pub fn set_output_colorspace(&mut self, cs: ColorSpace) {
        self.output_colorspace = Some(cs);
    }

    /// The colorspace override the decode stage should honour: the
    /// explicit `set_output_colorspace` value, or — issue #386 — the
    /// grayscale conversion implied by `set_output_format(Grayscale)`
    /// on a 3-component source. TurboJPEG decodes TJPF_GRAY from colour
    /// JPEGs by exactly this mapping (TJPF_GRAY -> JCS_GRAYSCALE), so a
    /// Grayscale *pixel format* request must not error where the
    /// *colorspace* request succeeds.
    pub(super) fn effective_output_colorspace(&self) -> Option<ColorSpace> {
        if self.output_colorspace.is_some() {
            return self.output_colorspace;
        }
        if self.output_format == Some(PixelFormat::Grayscale)
            && self.metadata.frame.components.len() == 3
        {
            return Some(ColorSpace::Grayscale);
        }
        None
    }

    /// Enable or disable ordered dithering for RGB565 output.
    ///
    /// When enabled, applies a 4x4 ordered dither pattern before truncating
    /// 8-bit RGB to 5-6-5, reducing visible banding in smooth gradients.
    /// Matches libjpeg-turbo's dithered RGB565 output mode.
    pub fn set_dither_565(&mut self, dither: bool) {
        self.dither_565 = dither;
    }

    /// Enable merged upsampling optimization (combines upsample + color convert).
    ///
    /// When enabled and subsampling is 4:2:0 or 4:2:2, uses a merged path that
    /// performs chroma upsampling and YCbCr->RGB conversion in a single pass.
    /// This avoids writing upsampled chroma to intermediate buffers, improving
    /// cache behavior. Slightly less accurate than separate fancy upsample
    /// because merged uses box-filter (nearest-neighbor) chroma replication.
    pub fn set_merged_upsample(&mut self, enabled: bool) {
        self.merged_upsample = enabled;
    }

    /// Configure which markers to save during decoding.
    ///
    /// By default, the decoder only parses known markers (JFIF, ICC, EXIF, Adobe, COM)
    /// and discards unknown APP markers. Call this to preserve arbitrary APP/COM markers
    /// in the decoded `Image.saved_markers` field.
    ///
    /// This re-parses the JPEG header with the new configuration.
    pub fn save_markers(&mut self, config: MarkerSaveConfig) {
        let mut reader: MarkerReader<'_> = MarkerReader::new(self.raw_data);
        reader.set_marker_save_config(config);
        if let Ok(metadata) = reader.read_markers() {
            self.metadata = metadata;
        }
    }

    /// Saved APP/COM markers from the source JPEG, populated by the most
    /// recent header parse. Empty unless `save_markers()` has been called
    /// with a non-`None` config (or the underlying parser was constructed
    /// with one). Used by C-ABI consumers (`crates/libjpeg-turbo-rs-capi`)
    /// that need to re-emit source markers verbatim during transcode.
    #[inline]
    pub fn saved_markers(&self) -> &[SavedMarker] {
        &self.metadata.saved_markers
    }

    // -----------------------------------------------------------------
    // Chainable configuration (issue #386). Each `with_*` is a by-value
    // wrapper over the matching `set_*` so construction reads as one
    // expression. The `set_*` forms stay for imperative call sites;
    // both routes hit the same field, so mixing them is fine.
    // -----------------------------------------------------------------

    /// Chainable [`Decoder::set_output_format`].
    ///
    /// ```
    /// # let jpeg = libjpeg_turbo_rs::compress(&[128; 48], 4, 4,
    /// #     libjpeg_turbo_rs::PixelFormat::Rgb, 90,
    /// #     libjpeg_turbo_rs::Subsampling::S444).unwrap();
    /// use libjpeg_turbo_rs::{Decoder, PixelFormat};
    /// let image = Decoder::new(&jpeg)?
    ///     .with_output_format(PixelFormat::Bgra)
    ///     .with_block_smoothing(false)
    ///     .decode_image()?;
    /// assert_eq!(image.pixel_format, PixelFormat::Bgra);
    /// # Ok::<(), libjpeg_turbo_rs::JpegError>(())
    /// ```
    #[must_use]
    pub fn with_output_format(mut self, format: PixelFormat) -> Self {
        self.set_output_format(format);
        self
    }

    /// Chainable [`Decoder::set_scale`].
    #[must_use]
    pub fn with_scale(mut self, scale: ScalingFactor) -> Self {
        self.set_scale(scale);
        self
    }

    /// Chainable [`Decoder::set_lenient`].
    #[must_use]
    pub fn with_lenient(mut self, lenient: bool) -> Self {
        self.set_lenient(lenient);
        self
    }

    /// Chainable [`Decoder::set_crop`].
    #[must_use]
    pub fn with_crop(mut self, x: usize, width: usize) -> Self {
        self.set_crop(x, width);
        self
    }

    /// Chainable [`Decoder::set_crop_y`].
    #[must_use]
    pub fn with_crop_y(mut self, y: usize, height: usize) -> Self {
        self.set_crop_y(y, height);
        self
    }

    /// Chainable [`Decoder::set_crop_region`].
    #[must_use]
    pub fn with_crop_region(mut self, x: usize, y: usize, width: usize, height: usize) -> Self {
        self.set_crop_region(x, y, width, height);
        self
    }

    /// Chainable [`Decoder::set_stop_on_warning`].
    #[must_use]
    pub fn with_stop_on_warning(mut self, stop: bool) -> Self {
        self.set_stop_on_warning(stop);
        self
    }

    /// Chainable [`Decoder::set_max_pixels`].
    #[must_use]
    pub fn with_max_pixels(mut self, limit: usize) -> Self {
        self.set_max_pixels(limit);
        self
    }

    /// Chainable [`Decoder::set_max_memory`].
    #[must_use]
    pub fn with_max_memory(mut self, limit: usize) -> Self {
        self.set_max_memory(limit);
        self
    }

    /// Chainable [`Decoder::set_scan_limit`].
    #[must_use]
    pub fn with_scan_limit(mut self, limit: u32) -> Self {
        self.set_scan_limit(limit);
        self
    }

    /// Chainable [`Decoder::set_limits`].
    #[must_use]
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.set_limits(limits);
        self
    }

    /// Chainable [`Decoder::set_fast_upsample`].
    #[must_use]
    pub fn with_fast_upsample(mut self, fast: bool) -> Self {
        self.set_fast_upsample(fast);
        self
    }

    /// Chainable [`Decoder::set_fast_dct`].
    #[must_use]
    pub fn with_fast_dct(mut self, fast: bool) -> Self {
        self.set_fast_dct(fast);
        self
    }

    /// Chainable [`Decoder::set_dct_method`].
    #[must_use]
    pub fn with_dct_method(mut self, method: DctMethod) -> Self {
        self.set_dct_method(method);
        self
    }

    /// Chainable [`Decoder::set_block_smoothing`].
    #[must_use]
    pub fn with_block_smoothing(mut self, smooth: bool) -> Self {
        self.set_block_smoothing(smooth);
        self
    }

    /// Chainable [`Decoder::set_output_colorspace`].
    #[must_use]
    pub fn with_output_colorspace(mut self, cs: ColorSpace) -> Self {
        self.set_output_colorspace(cs);
        self
    }

    /// Chainable [`Decoder::set_dither_565`].
    #[must_use]
    pub fn with_dither_565(mut self, dither: bool) -> Self {
        self.set_dither_565(dither);
        self
    }

    /// Chainable [`Decoder::set_merged_upsample`].
    #[must_use]
    pub fn with_merged_upsample(mut self, enabled: bool) -> Self {
        self.set_merged_upsample(enabled);
        self
    }

    /// Chainable [`Decoder::save_markers`].
    ///
    /// Like `save_markers`, a marker re-parse failure is swallowed and
    /// previously parsed metadata stays in effect — check
    /// [`Decoder::saved_markers`] afterwards if the distinction matters.
    #[must_use]
    pub fn with_save_markers(mut self, config: MarkerSaveConfig) -> Self {
        self.save_markers(config);
        self
    }

    /// Chainable [`Decoder::set_marker_processor`].
    #[must_use]
    pub fn with_marker_processor<F>(mut self, marker_type: u8, processor: F) -> Self
    where
        F: Fn(&[u8]) -> Option<Vec<u8>> + Send + 'static,
    {
        self.set_marker_processor(marker_type, processor);
        self
    }

    /// Chainable [`Decoder::set_resync_strategy`].
    #[must_use]
    pub fn with_resync_strategy<S>(mut self, strategy: S) -> Self
    where
        S: crate::decode::resync::RestartResyncStrategy + Send + 'static,
    {
        self.set_resync_strategy(strategy);
        self
    }

    /// Register a custom marker processor callback for a specific marker type.
    ///
    /// The callback must be `Send`: the decoder itself is `Send`
    /// (issue #384), so everything installed into it travels across
    /// threads with it.
    pub fn set_marker_processor<F>(&mut self, marker_type: u8, processor: F)
    where
        F: Fn(&[u8]) -> Option<Vec<u8>> + Send + 'static,
    {
        let has_marker: bool = self
            .metadata
            .saved_markers
            .iter()
            .any(|m| m.code == marker_type);
        if !has_marker {
            let mut reader: MarkerReader<'_> = MarkerReader::new(self.raw_data);
            reader.set_marker_save_config(MarkerSaveConfig::Specific(vec![marker_type]));
            if let Ok(metadata) = reader.read_markers() {
                self.metadata = metadata;
            }
        }
        self.marker_processors
            .insert(marker_type, Box::new(processor));
    }

    /// Install a custom `RestartResyncStrategy` to handle RST-marker desync
    /// events (mirrors C libjpeg-turbo's `jpeg_resync_to_restart` hook).
    ///
    /// When the decoder encounters a restart marker whose RST number does
    /// not match the expected counter — or when no RST marker is found at
    /// the expected position — the strategy's `on_desync` method is
    /// consulted. The returned `ResyncAction` tells the decoder whether to
    /// continue (accept the observed marker), skip to the next RST in the
    /// stream, or abort with a `CorruptData` error.
    ///
    /// If no strategy is installed, the decoder defaults to `Continue` —
    /// the historical Rust behavior of unconditionally accepting whatever
    /// RST marker it finds.
    pub fn set_resync_strategy<S>(&mut self, strategy: S)
    where
        S: crate::decode::resync::RestartResyncStrategy + Send + 'static,
    {
        *self.resync_strategy.borrow_mut() = Some(Box::new(strategy));
    }

    /// Resync logic shared between the fast-path decode loop and any
    /// future callers. Consults the installed `RestartResyncStrategy` when
    /// the observed RST number does not match `expected_rst`, applies the
    /// returned `ResyncAction`, and updates `expected_rst` to reflect the
    /// post-resync synchronization point.
    pub(super) fn apply_resync(
        bit_reader: &mut crate::decode::bitstream::BitReader,
        expected_rst: &mut u8,
        strategy: &mut Option<Box<dyn crate::decode::resync::RestartResyncStrategy + Send>>,
    ) -> Result<()> {
        use crate::decode::resync::ResyncAction;
        let found: Option<u8> = bit_reader.reset_and_consume_rst();
        let expected_val: u8 = *expected_rst & 0x07;
        let is_match: bool = matches!(found, Some(n) if n == expected_val);
        if is_match || strategy.is_none() {
            // Matched (or no strategy: historical Continue behavior).
            *expected_rst = expected_val.wrapping_add(1) & 0x07;
            return Ok(());
        }
        let strategy = strategy.as_mut().expect("handled above").as_mut();
        match strategy.on_desync(expected_val, found) {
            ResyncAction::Continue => {
                // Accept the observed RST (or lack thereof) as the new
                // sync point. If we saw an RST number, realign the counter
                // so subsequent expectations follow it.
                if let Some(n) = found {
                    *expected_rst = n.wrapping_add(1) & 0x07;
                } else {
                    *expected_rst = expected_val.wrapping_add(1) & 0x07;
                }
                Ok(())
            }
            ResyncAction::Skip => {
                // Advance past the bad marker (if any) and scan for the
                // next RST in the stream. Re-align the counter to that
                // marker's number + 1.
                if let Some(n) = bit_reader.scan_to_next_rst() {
                    *expected_rst = n.wrapping_add(1) & 0x07;
                    Ok(())
                } else {
                    Err(JpegError::CorruptData(
                        "no further RST marker found after desync".into(),
                    ))
                }
            }
            ResyncAction::Abort => Err(JpegError::CorruptData(format!(
                "RST marker desync: expected RST{expected_val}, found {found:?}"
            ))),
        }
    }

    pub fn decode(data: &'a [u8]) -> Result<Image> {
        let decoder = Self::new(data)?;
        decoder.decode_image()
    }

    pub fn decode_to(data: &'a [u8], format: PixelFormat) -> Result<Image> {
        let mut decoder = Self::new(data)?;
        decoder.set_output_format(format);
        decoder.decode_image()
    }
}
