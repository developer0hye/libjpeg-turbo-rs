pub mod api;
pub mod common;
pub mod decode;
pub mod encode;
pub mod simd;
pub mod transform;

pub use api::high_level::{
    compress, compress_optimized, compress_progressive, decompress, decompress_cropped,
    decompress_lenient, decompress_to,
};
pub use common::error::{DecodeWarning, JpegError, Result};
pub use common::types::*;
pub use decode::pipeline::Image;
