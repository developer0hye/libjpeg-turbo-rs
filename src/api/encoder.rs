use crate::common::error::Result;
use crate::common::types::{PixelFormat, Subsampling};
use crate::encode::pipeline as encoder;

/// JPEG encoder with builder-pattern configuration.
pub struct Encoder<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    optimize_huffman: bool,
    progressive: bool,
    arithmetic: bool,
    lossless: bool,
    icc_profile: Option<&'a [u8]>,
    exif_data: Option<&'a [u8]>,
}

impl<'a> Encoder<'a> {
    /// Create a new encoder for the given pixel data.
    pub fn new(pixels: &'a [u8], width: usize, height: usize, pixel_format: PixelFormat) -> Self {
        Self {
            pixels,
            width,
            height,
            pixel_format,
            quality: 75,
            subsampling: Subsampling::S420,
            optimize_huffman: false,
            progressive: false,
            arithmetic: false,
            lossless: false,
            icc_profile: None,
            exif_data: None,
        }
    }

    /// Set JPEG quality (1-100, default 75).
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Set chroma subsampling (default S420).
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Enable 2-pass optimized Huffman tables.
    pub fn optimize_huffman(mut self, optimize: bool) -> Self {
        self.optimize_huffman = optimize;
        self
    }

    /// Enable progressive JPEG mode.
    pub fn progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }

    /// Enable arithmetic entropy coding.
    pub fn arithmetic(mut self, arithmetic: bool) -> Self {
        self.arithmetic = arithmetic;
        self
    }

    /// Enable lossless JPEG mode (SOF3).
    pub fn lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Embed an ICC color profile.
    pub fn icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(data);
        self
    }

    /// Embed EXIF metadata (raw TIFF data).
    pub fn exif_data(mut self, data: &'a [u8]) -> Self {
        self.exif_data = Some(data);
        self
    }

    /// Encode and return the JPEG byte stream.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let base = if self.lossless {
            encoder::compress_lossless(self.pixels, self.width, self.height, self.pixel_format)?
        } else if self.arithmetic {
            encoder::compress_arithmetic(
                self.pixels,
                self.width,
                self.height,
                self.pixel_format,
                self.quality,
                self.subsampling,
            )?
        } else if self.progressive {
            encoder::compress_progressive(
                self.pixels,
                self.width,
                self.height,
                self.pixel_format,
                self.quality,
                self.subsampling,
            )?
        } else if self.optimize_huffman {
            encoder::compress_optimized(
                self.pixels,
                self.width,
                self.height,
                self.pixel_format,
                self.quality,
                self.subsampling,
            )?
        } else {
            encoder::compress(
                self.pixels,
                self.width,
                self.height,
                self.pixel_format,
                self.quality,
                self.subsampling,
            )?
        };

        if self.icc_profile.is_some() || self.exif_data.is_some() {
            encoder::inject_metadata(&base, self.icc_profile, self.exif_data)
        } else {
            Ok(base)
        }
    }
}
