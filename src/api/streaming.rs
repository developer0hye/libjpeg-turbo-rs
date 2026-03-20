use crate::common::error::Result;
use crate::common::types::FrameHeader;
use crate::decode::pipeline::Decoder;

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
}
