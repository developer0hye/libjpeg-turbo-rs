pub mod api;
pub mod common;
pub mod decode;
pub mod encode;
pub mod simd;
pub mod transform;

pub use api::coefficient::{
    copy_critical_parameters, read_coefficients, transform_jpeg as transform,
    transform_jpeg_with_options, write_coefficients, EncoderComponentInfo, EncoderConfig,
    JpegCoefficients,
};
pub use api::encoder::{Encoder, HuffmanTableDef};
pub use api::high_level::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless,
    compress_lossless_arithmetic, compress_lossless_extended, compress_optimized,
    compress_progressive, compress_with_metadata, decompress, decompress_cropped,
    decompress_lenient, decompress_to,
};
pub use api::image_io::{load_image, load_image_from_bytes, save_bmp, save_ppm, LoadedImage};
pub use api::quality::quality_scaling;
pub use api::raw_data::{compress_raw, decompress_raw, RawImage};
pub use encode::marker_writer::MarkerStreamWriter;
/// Color quantization for 8-bit indexed/palette output.
pub mod quantize {
    pub use crate::api::quantize::{
        dequantize, quantize, DitherMode, QuantizeOptions, QuantizedImage,
    };
}
pub use api::progressive_output::ProgressiveDecoder;
pub use api::scanline::{ScanlineDecoder, ScanlineEncoder};
/// Streaming I/O functions for reading/writing JPEG via `std::io` traits and file paths.
pub use api::stream;
pub use common::bufsize::{
    calc_jpeg_dimensions, calc_output_dimensions, jpeg_buf_size, transform_buf_size, yuv_buf_size,
    yuv_plane_height, yuv_plane_size, yuv_plane_width,
};
pub use common::error::{DecodeWarning, JpegError, Result};
pub use common::jfif::extract_jfif_thumbnail;
pub use common::sample::Sample;
pub use common::traits::{DefaultErrorHandler, ErrorHandler, ProgressInfo, ProgressListener};
pub use common::types::*;
pub use decode::pipeline::Image;
pub use transform::{TransformOp, TransformOptions};
/// 12-bit and 16-bit sample precision support.
pub mod precision {
    pub use crate::api::precision::{
        compress_12bit, compress_16bit, decompress_12bit, decompress_16bit, Image12, Image16,
    };
}
/// TJ3-compatible handle/parameter API.
pub mod tj3 {
    pub use crate::api::tj3::{TjHandle, TjParam};
}
