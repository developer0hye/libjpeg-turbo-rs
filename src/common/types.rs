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
}

impl Subsampling {
    /// Max horizontal sampling factor (luma blocks per MCU row).
    pub fn mcu_width_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S440 => 1,
            Self::S422 | Self::S420 => 2,
            Self::S411 => 4,
        }
    }

    /// Max vertical sampling factor (luma blocks per MCU column).
    pub fn mcu_height_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S422 | Self::S411 => 1,
            Self::S420 | Self::S440 => 2,
        }
    }
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
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra | Self::Cmyk => 4,
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
