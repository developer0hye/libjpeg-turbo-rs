//! Stable public facade for the JPEG encoder pipeline.
//!
//! Implementations are split by encoding mode under the private
//! `pipeline_impl` module.  Re-exporting them here preserves the established
//! `libjpeg_turbo_rs::encode::pipeline::*` API paths and signatures.

pub use super::pipeline_impl::{
    compress, compress_arithmetic, compress_arithmetic_progressive,
    compress_arithmetic_progressive_rgb_direct, compress_arithmetic_rgb_direct,
    compress_custom_huffman, compress_custom_quant, compress_custom_sampling, compress_lossless,
    compress_lossless_arithmetic, compress_lossless_extended, compress_lossless_extended_precision,
    compress_optimized, compress_optimized_with_params, compress_progressive,
    compress_progressive_custom, compress_progressive_custom_with_restart,
    compress_progressive_rgb_direct, compress_progressive_with_restart, compress_raw,
    compress_rgb_direct, compress_rgb_direct_with_params, compress_with_metadata,
    compress_with_params, compress_with_restart, compute_reciprocal, inject_comment,
    inject_metadata, inject_metadata_full, inject_saved_markers, CompressParams,
};

pub(crate) use super::pipeline_impl::{
    emit_eobrun, emit_eobrun_with_corr, encode_ac_first_block, encode_ac_refine_block,
    MAX_CORR_BITS,
};
