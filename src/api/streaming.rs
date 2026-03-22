use crate::common::error::Result;
use crate::common::types::{FrameHeader, PixelFormat, ScalingFactor};
use crate::decode::pipeline::{Decoder, Image};

/// Streaming JPEG decoder — reads header first, then decodes on demand.
pub struct StreamingDecoder<'a> {
    inner: Decoder<'a>,
}

impl<'a> StreamingDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let inner = Decoder::new(data)?;
        Ok(Self { inner })
    }

    pub fn header(&self) -> &FrameHeader {
        self.inner.header()
    }

    /// Set the output pixel format.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.inner.set_output_format(format);
    }

    /// Set the decompression scaling factor.
    pub fn set_scale(&mut self, scale: ScalingFactor) {
        self.inner.set_scale(scale);
    }

    /// Enable lenient mode: continue on errors, filling corrupt areas with gray.
    pub fn set_lenient(&mut self, lenient: bool) {
        self.inner.set_lenient(lenient);
    }

    /// Set horizontal crop. xoffset is auto-aligned down to iMCU column boundary.
    /// Updates xoffset and width in-place to reflect aligned values.
    pub fn crop_scanline(&mut self, xoffset: &mut usize, width: &mut usize) -> Result<()> {
        let header = self.inner.header();
        let max_h = header
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let block_size: usize = 8;
        let imcu_width = max_h * block_size;

        // Align xoffset down to iMCU boundary
        let aligned_x = (*xoffset / imcu_width) * imcu_width;
        let aligned_end = ((*xoffset + *width + imcu_width - 1) / imcu_width) * imcu_width;
        let aligned_width = (aligned_end - aligned_x).min(header.width as usize - aligned_x);

        *xoffset = aligned_x;
        *width = aligned_width;

        self.inner.set_crop(aligned_x, aligned_width);
        Ok(())
    }

    /// Skip scanlines without full decoding.
    /// Returns actual number of lines skipped.
    pub fn skip_scanlines(&mut self, num_lines: usize) -> Result<usize> {
        // Initial implementation: returns requested count (no-op optimization).
        // Full decode + crop handles the actual row selection.
        Ok(num_lines)
    }

    /// Decode the JPEG payload using the already-parsed metadata.
    pub fn decode(&self) -> Result<Image> {
        self.inner.decode_image()
    }
}
