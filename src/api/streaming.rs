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

    /// Decode the JPEG payload using the already-parsed metadata.
    pub fn decode(&self) -> Result<Image> {
        self.inner.decode_image()
    }
}
