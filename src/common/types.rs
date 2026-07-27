/// JPEG color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Grayscale,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
    /// Unknown / pass-through colorspace (no color conversion).
    /// Matches libjpeg's `JCS_UNKNOWN`.
    Unknown,
}

impl ColorSpace {
    pub fn num_components(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::YCbCr | Self::Rgb => 3,
            Self::Cmyk | Self::Ycck => 4,
            // Warning: returns 3 for `Unknown`. Callers processing unknown
            // colorspaces should use `FrameHeader::components.len()` instead.
            Self::Unknown => 3,
        }
    }
}

/// Chroma subsampling modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:4:4 — no subsampling
    S444,
    /// 4:2:2 — horizontal 2x, vertical 1x
    S422,
    /// 4:2:0 — horizontal 2x, vertical 2x
    S420,
    /// 4:4:0 — horizontal 1x, vertical 2x
    S440,
    /// 4:1:1 — horizontal 4x, vertical 1x
    S411,
    /// 4:4:1 — horizontal 1x, vertical 4x
    S441,
    /// 4:1:0 — horizontal 4x, vertical 2x (TJSAMP_410, libjpeg-turbo 3.x).
    S410,
    /// 2:4 — horizontal 2x, vertical 4x (TJSAMP_24, libjpeg-turbo 3.x).
    S24,
    /// Unknown / non-standard subsampling factors.
    /// Matches libjpeg's `TJSAMP_UNKNOWN`.
    Unknown,
}

impl Subsampling {
    /// Max horizontal sampling factor (luma blocks per MCU row).
    pub fn mcu_width_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S440 | Self::S441 | Self::Unknown => 1,
            Self::S422 | Self::S420 | Self::S24 => 2,
            Self::S411 | Self::S410 => 4,
        }
    }

    /// Max vertical sampling factor (luma blocks per MCU column).
    pub fn mcu_height_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S422 | Self::S411 | Self::Unknown => 1,
            Self::S420 | Self::S440 | Self::S410 => 2,
            Self::S441 | Self::S24 => 4,
        }
    }

    /// Returns (h_sampling_factor, v_sampling_factor) for SOF component definitions.
    pub fn sampling_factors(self) -> (u8, u8) {
        match self {
            Self::S444 | Self::Unknown => (1, 1),
            Self::S422 => (2, 1),
            Self::S420 => (2, 2),
            Self::S440 => (1, 2),
            Self::S411 => (4, 1),
            Self::S441 => (1, 4),
            Self::S410 => (4, 2),
            Self::S24 => (2, 4),
        }
    }
}

/// DCT/IDCT algorithm selection.
///
/// Controls which forward DCT algorithm the encoder uses. All three methods
/// produce valid JPEG output that any decoder can read. They differ in speed
/// and accuracy trade-offs, matching libjpeg-turbo's `JDCT_ISLOW`, `JDCT_IFAST`,
/// and `JDCT_FLOAT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DctMethod {
    /// Accurate integer DCT (default). Uses 13-bit fixed-point arithmetic.
    /// Matches libjpeg-turbo's `JDCT_ISLOW`.
    #[default]
    IsLow,
    /// Fast integer DCT with reduced accuracy. Uses 8-bit fixed-point arithmetic
    /// and the AA&N (Arai, Agui, Nakajima) algorithm with only 5 multiplies.
    /// Matches libjpeg-turbo's `JDCT_IFAST`.
    IsFast,
    /// Floating-point DCT. Uses f64 arithmetic and the AA&N algorithm.
    /// Matches libjpeg-turbo's `JDCT_FLOAT`.
    Float,
}

/// Output pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Grayscale,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
    /// Raw CMYK output (4 bytes per pixel: C, M, Y, K).
    Cmyk,
    /// RGB + padding byte (4bpp, padding ignored).
    Rgbx,
    /// BGR + padding byte (4bpp, padding ignored).
    Bgrx,
    /// Padding + RGB (4bpp, padding byte first).
    Xrgb,
    /// Padding + BGR (4bpp, padding byte first).
    Xbgr,
    /// Alpha + RGB (4bpp, alpha byte first).
    Argb,
    /// Alpha + BGR (4bpp, alpha byte first).
    Abgr,
    /// 5-6-5 packed RGB (2bpp, decode output only).
    Rgb565,
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::Rgb565 => 2,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba
            | Self::Bgra
            | Self::Cmyk
            | Self::Rgbx
            | Self::Bgrx
            | Self::Xrgb
            | Self::Xbgr
            | Self::Argb
            | Self::Abgr => 4,
        }
    }

    /// Channel byte offset for red within one pixel.
    /// Returns `None` for Grayscale, Cmyk, and Rgb565.
    pub fn red_offset(self) -> Option<usize> {
        match self {
            Self::Rgb | Self::Rgba | Self::Rgbx => Some(0),
            Self::Bgr | Self::Bgra | Self::Bgrx => Some(2),
            Self::Xrgb | Self::Argb => Some(1),
            Self::Xbgr | Self::Abgr => Some(3),
            _ => None,
        }
    }

    /// Channel byte offset for green within one pixel.
    /// Returns `None` for Grayscale, Cmyk, and Rgb565.
    pub fn green_offset(self) -> Option<usize> {
        match self {
            Self::Rgb | Self::Rgba | Self::Rgbx => Some(1),
            Self::Bgr | Self::Bgra | Self::Bgrx => Some(1),
            Self::Xrgb | Self::Argb => Some(2),
            Self::Xbgr | Self::Abgr => Some(2),
            _ => None,
        }
    }

    /// Channel byte offset for blue within one pixel.
    /// Returns `None` for Grayscale, Cmyk, and Rgb565.
    pub fn blue_offset(self) -> Option<usize> {
        match self {
            Self::Rgb | Self::Rgba | Self::Rgbx => Some(2),
            Self::Bgr | Self::Bgra | Self::Bgrx => Some(0),
            Self::Xrgb | Self::Argb => Some(3),
            Self::Xbgr | Self::Abgr => Some(1),
            _ => None,
        }
    }
}

/// Information about a single image component (Y, Cb, or Cr).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentInfo {
    /// Component identifier (1=Y, 2=Cb, 3=Cr per JFIF).
    pub id: u8,
    /// Horizontal sampling factor (1-4).
    pub horizontal_sampling: u8,
    /// Vertical sampling factor (1-4).
    pub vertical_sampling: u8,
    /// Index into the quantization table array.
    pub quant_table_index: u8,
}

/// Maximum number of frame/scan components the decoder supports.
///
/// `read_sof` / `read_sos` reject anything above this, which is what
/// lets the decode path use fixed `[_; MAX_COMPONENTS]` arrays instead
/// of per-decode Vecs (issue #351).
///
/// For **scans** this is exactly the spec limit: ISO 10918-1 B.2.3 caps
/// `Ns ≤ 4`, and C's `MAX_COMPS_IN_SCAN` (`jpeglib.h:71`) is likewise 4.
/// For **frames** it is deliberately stricter than C: ISO 10918-1 B.2.2
/// allows `Nf ≤ 255` and C caps frame components at `MAX_COMPONENTS`
/// = 10 (`jmorecfg.h:30`, enforced in `jdinput.c:74` with
/// `JERR_COMPONENT_COUNT`). No real-world colour model needs more than
/// 4 (YCbCr / YCCK / CMYK), and a 5+-component SOF that C would accept
/// is rejected here — a known, intentional divergence, not spec parity.
pub const MAX_COMPONENTS: usize = 4;

/// Parsed from the SOF marker — describes the image frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// Sample precision in bits (8 for Baseline).
    pub precision: u8,
    /// Image height in pixels.
    pub height: u16,
    /// Image width in pixels.
    pub width: u16,
    /// Per-component info.
    pub components: Vec<ComponentInfo>,
    /// True for SOF2 (progressive DCT).
    pub is_progressive: bool,
    /// True for SOF3 (lossless Huffman-coded).
    pub is_lossless: bool,
}

/// Parsed from the SOS marker — describes one scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanHeader {
    /// Component selectors for this scan.
    pub components: Vec<ScanComponentSelector>,
    /// Spectral selection start (0 for DC, 1..63 for AC).
    pub spec_start: u8,
    /// Spectral selection end (0 for DC-only, up to 63).
    pub spec_end: u8,
    /// Successive approximation high bit position (0 = first scan for this band).
    pub succ_high: u8,
    /// Successive approximation low bit position.
    pub succ_low: u8,
}

/// Region of interest for cropped decompression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Per-component selector within a scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanComponentSelector {
    /// Component identifier (matches ComponentInfo::id).
    pub component_id: u8,
    /// DC Huffman table index (0-3).
    pub dc_table_index: u8,
    /// AC Huffman table index (0-3).
    pub ac_table_index: u8,
}

/// Decompression scaling factor.
///
/// Controls the output size via scaled IDCT. All 16 libjpeg-turbo factors are
/// supported: 2/1, 15/8, 7/4, 13/8, 3/2, 11/8, 5/4, 9/8, 1/1, 7/8, 3/4,
/// 5/8, 1/2, 3/8, 1/4, 1/8. Each factor maps to an IDCT block output size
/// from 16×16 (2/1) down to 1×1 (1/8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalingFactor {
    pub num: u32,
    pub denom: u32,
}

impl ScalingFactor {
    pub fn new(num: u32, denom: u32) -> Self {
        Self { num, denom }
    }

    /// The IDCT block output size for this scaling factor.
    /// Ranges from 16 (for 2/1) through 8 (for 1/1) down to 1 (for 1/8).
    pub fn block_size(self) -> usize {
        assert!(
            self.denom != 0,
            "ScalingFactor denominator must not be zero"
        );
        let ratio_x8 = (self.num * 8).div_ceil(self.denom);
        (ratio_x8 as usize).clamp(1, 16)
    }

    /// Compute scaled output dimension: ceil(input_dim * num / denom).
    pub fn scale_dim(self, input_dim: usize) -> usize {
        assert!(
            self.denom != 0,
            "ScalingFactor denominator must not be zero"
        );
        (input_dim * self.num as usize).div_ceil(self.denom as usize)
    }
}

impl Default for ScalingFactor {
    fn default() -> Self {
        Self { num: 1, denom: 1 }
    }
}

/// Pixel density information from JFIF marker.
/// Configurable decoder resource limits (issue #355, the Rust-side twin
/// of P4-14's C-ABI `max_memory_to_use`).
///
/// Defaults are permissive — they accept everything `djpeg` accepts in
/// the corpus gates (the drop-in contract comes first) while still
/// bounding the pathological corner: a header-only 65535x65535 SOF
/// (4.29 gigapixels from a few dozen bytes) exceeds the default
/// `max_pixels` and is rejected before any plane allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum image width in pixels (default: 65_500, matching C's
    /// `JPEG_MAX_DIMENSION`). Mostly shadowed by `max_pixels`; matters
    /// for extreme aspect ratios.
    pub max_width: usize,
    /// Maximum image height in pixels (default: 65_500).
    pub max_height: usize,
    /// Maximum total pixels, guarding the product the individual
    /// dimension caps cannot (default: 2_147_483_647 ≈ 2.1 gigapixels).
    pub max_pixels: u64,
    /// Maximum number of scans (default: 8_192 — far above any real
    /// progressive script, low enough to stop scan bombs).
    pub max_scans: usize,
    /// Estimated decode-memory ceiling in bytes (default: `None`).
    /// Shares the estimation model of `Decoder::set_max_memory`.
    ///
    /// Note: the default `max_pixels` bounds the *header bomb*, not
    /// decode memory — a 2-gigapixel progressive image legitimately
    /// needs multi-GB coefficient buffers. Set `max_memory` to bound
    /// memory.
    pub max_memory: Option<u64>,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_width: 65_500,
            max_height: 65_500,
            max_pixels: 2_147_483_647,
            max_scans: 8_192,
            max_memory: None,
        }
    }
}

impl DecodeLimits {
    /// Frame-dimension checks shared by every decode entry point,
    /// including the ones without a limits API (`read_coefficients`,
    /// `decompress_12bit`/`_16bit`), which apply the defaults.
    pub(crate) fn check_frame(
        &self,
        width: usize,
        height: usize,
    ) -> crate::common::error::Result<()> {
        use crate::common::error::JpegError;
        if width > self.max_width {
            return Err(JpegError::LimitExceeded {
                what: "image width",
                actual: width as u64,
                limit: self.max_width as u64,
            });
        }
        if height > self.max_height {
            return Err(JpegError::LimitExceeded {
                what: "image height",
                actual: height as u64,
                limit: self.max_height as u64,
            });
        }
        let total_pixels: u64 = (width as u64) * (height as u64);
        if total_pixels > self.max_pixels {
            return Err(JpegError::LimitExceeded {
                what: "total pixels",
                actual: total_pixels,
                limit: self.max_pixels,
            });
        }
        Ok(())
    }

    /// zune-jpeg `new_safe`-like values for callers that want tight
    /// bounds: 16_384x16_384, 100 scans. Memory stays unbounded — set
    /// `max_memory` if you need a byte ceiling.
    pub fn strict() -> Self {
        Self {
            max_width: 16_384,
            max_height: 16_384,
            max_pixels: 16_384 * 16_384,
            max_scans: 100,
            max_memory: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensityInfo {
    pub unit: DensityUnit,
    pub x: u16,
    pub y: u16,
}

impl Default for DensityInfo {
    fn default() -> Self {
        Self {
            unit: DensityUnit::Unknown,
            x: 1,
            y: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityUnit {
    Unknown,
    Dpi,
    Dpcm,
}

/// A saved JPEG marker (APP or COM).
#[derive(Debug, Clone)]
pub struct SavedMarker {
    /// Marker code (e.g., 0xE0 for APP0, 0xFE for COM).
    pub code: u8,
    /// Raw marker data (after the 2-byte length field).
    pub data: Vec<u8>,
}

/// Configuration for which markers the decoder should save.
///
/// Controls which APP and COM markers are preserved during decoding,
/// matching libjpeg-turbo's `jpeg_save_markers()` / `TJPARAM_SAVEMARKERS`.
#[derive(Debug, Clone, Default)]
pub enum MarkerSaveConfig {
    /// Do not save any markers (default).
    #[default]
    None,
    /// Save all APP (0xE0-0xEF) and COM (0xFE) markers.
    All,
    /// Save only APP markers (0xE0-0xEF), not COM.
    AppOnly,
    /// Save only the specified marker codes.
    Specific(Vec<u8>),
    /// Save only the specified marker codes, truncating each marker's body
    /// to at most the associated byte limit.
    ///
    /// This matches libjpeg's `jpeg_save_markers(cinfo, code, length_limit)`
    /// per-code truncation: the saved `data` slice is `min(full_len, limit)`
    /// bytes long.  A missing entry for a code is treated as "no limit"
    /// (`usize::MAX`).
    WithLimits(std::collections::HashMap<u8, usize>),
}

/// Progressive scan script entry.
///
/// Defines one scan in a custom progressive scan script. Users can build
/// a `Vec<ScanScript>` to control the exact ordering and spectral/successive
/// approximation parameters of each progressive scan pass.
#[derive(Debug, Clone)]
pub struct ScanScript {
    /// Component indices (0-based) included in this scan.
    pub components: Vec<u8>,
    /// Spectral selection start (0 for DC).
    pub ss: u8,
    /// Spectral selection end (0 for DC-only, 63 for full AC).
    pub se: u8,
    /// Successive approximation high bit (0 for first pass).
    pub ah: u8,
    /// Successive approximation low bit.
    pub al: u8,
}

/// One chunk of an ICC profile stored in an APP2 marker.
///
/// ICC profiles larger than 65519 bytes are split across multiple APP2 markers,
/// each carrying a sequence number and total count.
#[derive(Debug, Clone)]
pub struct IccChunk {
    /// 1-based sequence number of this chunk.
    pub seq_no: u8,
    /// Total number of chunks for the complete profile.
    pub num_markers: u8,
    /// Raw profile data for this chunk.
    pub data: Vec<u8>,
}
