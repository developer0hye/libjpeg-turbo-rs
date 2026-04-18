pub mod api;
pub mod common;
pub mod decode;
pub mod encode;
pub mod simd;
pub mod transform;

pub use api::abbreviated::{read_header, HeaderResult, TablesOnlyState};
pub use api::coefficient::{
    copy_critical_parameters, read_coefficients, transform_jpeg as transform,
    transform_jpeg_with_options, write_coefficients, EncoderComponentInfo, EncoderConfig,
    JpegCoefficients,
};
pub use api::encoder::{Encoder, HuffmanTableDef};

/// Produce a tables-only abbreviated JPEG datastream for the given encoder configuration.
///
/// Returns `SOI + DQT(s) + DHT(s) + EOI` with no image data. Equivalent to
/// libjpeg-turbo's `jpeg_write_tables()`. The stream can be parsed by `read_header()`
/// to preload tables for subsequent decoding of body-only streams.
pub fn jpeg_write_tables(encoder: &Encoder<'_>) -> Vec<u8> {
    encoder.write_tables()
}
pub use api::high_level::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_into,
    compress_lossless, compress_lossless_arithmetic, compress_lossless_extended,
    compress_optimized, compress_progressive, compress_with_metadata, decompress,
    decompress_cropped, decompress_lenient, decompress_to,
};
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
pub use api::image_io::{
    load_image, load_ppm_12bit, load_ppm_16bit, save_bmp, save_ppm, save_ppm_12bit, save_ppm_16bit,
};
pub use api::image_io::{
    load_image_from_bytes, load_ppm_12bit_from_bytes, load_ppm_16bit_from_bytes, LoadedImage,
    LoadedImage12, LoadedImage16,
};
pub use api::precision::{
    read_scanlines_12, read_scanlines_16, write_scanlines_12, write_scanlines_16,
};
pub use api::quality::quality_scaling;
pub use api::quantize::requantize;
pub use api::raw_data::{compress_raw, decompress_raw, RawImage};
/// 12-bit raw planar encode/decode (YCbCr component planes at native resolution).
pub mod raw_data_12 {
    pub use crate::api::raw_data_12::{compress_raw_12, decompress_raw_12, RawImage12};
}
pub use api::raw_thumbnail::extract_embedded_jpeg;
pub use encode::marker_writer::MarkerStreamWriter;
/// Color quantization for 8-bit indexed/palette output.
pub mod quantize {
    pub use crate::api::quantize::{
        dequantize, quantize, requantize, DitherMode, QuantizeOptions, QuantizedImage,
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
pub use decode::pipeline::{Decoder, Image};
pub use transform::{MarkerCopyMode, TransformOp, TransformOptions};
/// 12-bit and 16-bit sample precision support.
pub mod precision {
    pub use crate::api::precision::{
        compress_12bit, compress_16bit, compress_lossless_arbitrary, decompress_12bit,
        decompress_16bit, decompress_lossless_arbitrary, read_scanlines_12, read_scanlines_16,
        write_scanlines_12, write_scanlines_16, Image12, Image16,
    };
}
/// TJ3-compatible handle/parameter API.
pub mod tj3 {
    pub use crate::api::tj3::{TjHandle, TjParam};
}
