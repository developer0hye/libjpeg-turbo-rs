/// JPEG color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Grayscale,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
}

impl ColorSpace {
    pub fn num_components(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::YCbCr | Self::Rgb => 3,
            Self::Cmyk | Self::Ycck => 4,
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
}

impl Subsampling {
    /// Max horizontal sampling factor (luma blocks per MCU row).
    pub fn mcu_width_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S440 | Self::S441 => 1,
            Self::S422 | Self::S420 => 2,
            Self::S411 => 4,
        }
    }

    /// Max vertical sampling factor (luma blocks per MCU column).
    pub fn mcu_height_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S422 | Self::S411 => 1,
            Self::S420 | Self::S440 => 2,
            Self::S441 => 4,
        }
    }

    /// Returns (h_sampling_factor, v_sampling_factor) for SOF component definitions.
    pub fn sampling_factors(self) -> (u8, u8) {
        match self {
            Self::S444 => (1, 1),
            Self::S422 => (2, 1),
            Self::S420 => (2, 2),
            Self::S440 => (1, 2),
            Self::S411 => (4, 1),
            Self::S441 => (1, 4),
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
#[derive(Debug, Clone, Copy)]
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

/// Parsed from the SOF marker — describes the image frame.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Copy)]
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
/// Controls the output size via reduced IDCT. Supported ratios: 1/1, 1/2, 1/4, 1/8.
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
    /// 8 for full, 4 for 1/2, 2 for 1/4, 1 for 1/8.
    pub fn block_size(self) -> usize {
        let ratio_x8 = (self.num * 8 + self.denom - 1) / self.denom;
        match ratio_x8 {
            0 => 1,
            1 => 1,
            2 => 2,
            3..=4 => 4,
            _ => 8,
        }
    }

    /// Compute scaled output dimension: ceil(input_dim * num / denom).
    pub fn scale_dim(self, input_dim: usize) -> usize {
        (input_dim * self.num as usize + self.denom as usize - 1) / self.denom as usize
    }
}

impl Default for ScalingFactor {
    fn default() -> Self {
        Self { num: 1, denom: 1 }
    }
}

/// Pixel density information from JFIF marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensityInfo {
    pub unit: DensityUnit,
    pub x: u16,
    pub y: u16,
}

impl Default for DensityInfo {
    fn default() -> Self {
        Self {
            unit: DensityUnit::Dpi,
            x: 72,
            y: 72,
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
#[derive(Debug, Clone)]
pub enum MarkerSaveConfig {
    /// Do not save any markers (default).
    None,
    /// Save all APP (0xE0-0xEF) and COM (0xFE) markers.
    All,
    /// Save only APP markers (0xE0-0xEF), not COM.
    AppOnly,
    /// Save only the specified marker codes.
    Specific(Vec<u8>),
}

impl Default for MarkerSaveConfig {
    fn default() -> Self {
        MarkerSaveConfig::None
    }
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
