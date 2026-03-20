use crate::common::error::Result;
use crate::decode::pipeline::{Decoder, Image};

/// Decompress a JPEG byte slice into an Image.
pub fn decompress(data: &[u8]) -> Result<Image> {
    Decoder::decode(data)
}
