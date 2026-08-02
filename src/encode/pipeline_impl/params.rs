use super::{DctMethod, HuffmanTableDef, PixelFormat, Subsampling};

/// The full option set for a single-pass baseline encode.
///
/// This exists so that the baseline `compress_*` entry points are thin shims
/// over one implementation instead of near-copies of it. Historically each
/// variant carried only the options it named — `compress_with_restart` could
/// not express custom tables, `compress_custom_quant` could not express a
/// restart interval — so a fix or an optimization landed in whichever copy the
/// author happened to be editing. That produced real divergence: the
/// dummy-block contract was implemented in one branch and not the others
/// (#316), and CMYK silently discarded every option a variant could not pass
/// on (#313).
///
/// Options that are `None` / zero mean "not requested" and select the JPEG
/// default, so adding a field does not change any existing caller's output.
pub struct CompressParams<'a> {
    /// Raw pixel data in the format given by `pixel_format`.
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    /// Quality factor 1-100. Ignored for components whose quantization table
    /// is supplied through `custom_quant`.
    pub quality: u8,
    pub subsampling: Subsampling,
    pub dct_method: DctMethod,
    /// MCUs between RST markers; 0 emits no DRI marker and no restarts.
    pub restart_interval: u16,
    /// Per-slot quantization tables. Slot 0 overrides luma, slot 1 chroma;
    /// unset slots fall back to the quality-scaled Annex K tables.
    pub custom_quant: Option<&'a [Option<[u16; 64]>; 4]>,
    /// Per-slot DC Huffman tables, same slot convention as `custom_quant`.
    pub custom_dc_huffman: Option<&'a [Option<HuffmanTableDef>; 4]>,
    /// Per-slot AC Huffman tables, same slot convention as `custom_quant`.
    pub custom_ac_huffman: Option<&'a [Option<HuffmanTableDef>; 4]>,
    /// Two-pass optimized Huffman coding. Computes tables from the actual
    /// symbol statistics, so any `custom_*_huffman` tables are superseded —
    /// matching libjpeg's `optimize_coding` semantics.
    pub optimize_huffman: bool,
    /// Input smoothing strength 0-100, as C's `smoothing_factor`.
    pub smoothing_factor: u8,
}

impl<'a> CompressParams<'a> {
    /// Construct with every optional knob at its JPEG default.
    pub fn new(
        pixels: &'a [u8],
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        quality: u8,
        subsampling: Subsampling,
    ) -> Self {
        Self {
            pixels,
            width,
            height,
            pixel_format,
            quality,
            subsampling,
            dct_method: DctMethod::IsLow,
            restart_interval: 0,
            custom_quant: None,
            custom_dc_huffman: None,
            custom_ac_huffman: None,
            optimize_huffman: false,
            smoothing_factor: 0,
        }
    }

    pub fn dct_method(mut self, dct_method: DctMethod) -> Self {
        self.dct_method = dct_method;
        self
    }

    pub fn restart_interval(mut self, restart_interval: u16) -> Self {
        self.restart_interval = restart_interval;
        self
    }

    pub fn custom_quant(mut self, custom_quant: &'a [Option<[u16; 64]>; 4]) -> Self {
        self.custom_quant = Some(custom_quant);
        self
    }

    pub fn custom_huffman(
        mut self,
        dc: &'a [Option<HuffmanTableDef>; 4],
        ac: &'a [Option<HuffmanTableDef>; 4],
    ) -> Self {
        self.custom_dc_huffman = Some(dc);
        self.custom_ac_huffman = Some(ac);
        self
    }

    pub fn optimize_huffman(mut self, optimize: bool) -> Self {
        self.optimize_huffman = optimize;
        self
    }

    pub fn smoothing_factor(mut self, factor: u8) -> Self {
        self.smoothing_factor = factor.min(100);
        self
    }
}
