//! Internal encoder pipeline implementation, split by encoding mode.
//!
//! The public compatibility surface remains in [`super::pipeline`].  Shared
//! kernels live at this module root so each mode can depend on them without
//! importing through the public facade.

use crate::api::encoder::HuffmanTableDef;
use crate::common::error::{JpegError, Result};
// Every mode's input check sizes the same `width x height x bytes_per_pixel`
// rectangle, so they all size it through one implementation (P4-139 chunk 2).
use crate::common::layout::{checked_span, ImageLayout};
use crate::common::types::{DctMethod, PixelFormat, SavedMarker, ScanScript, Subsampling};
use crate::encode::color;
use crate::encode::huffman_encode::{
    build_huff_table, local_drain_bits, local_put_bits, BitWriter, HuffTable, HuffmanEncoder,
};
use crate::encode::marker_writer;
use crate::encode::progressive::ProgressiveScan;
use crate::encode::tables;
use crate::simd::QuantDivisors;
#[allow(unused_imports)]
use alloc::{format, vec};
#[allow(unused_imports)]
use alloc::{string::ToString, vec::Vec};

mod arithmetic;
mod baseline;
mod custom_sampling;
mod dispatch;
mod huffman_tables;
mod lossless;
mod mcu;
mod metadata;
mod optimized;
mod params;
mod progressive;
mod progressive_entropy;
mod quant_divisors;
mod raw;
mod sampling;
#[cfg(test)]
mod tests;

use dispatch::{
    may_use_islow_simd_kernel, resolve_quant_tables, select_bgr_to_ycbcr_fn,
    select_bgra_to_ycbcr_fn, select_rgba_to_ycbcr_fn, ColorConvertRowFn,
};
use huffman_tables::ResolvedHuffman;
use mcu::{
    encode_color_mcu, encode_color_mcu_with_dummies, encode_downsampled_chroma_block,
    encode_dummy_block, encode_single_block, is_y_dummy,
};
pub use params::CompressParams;
pub use quant_divisors::compute_reciprocal;
use quant_divisors::{scale_quant_for_fdct, scale_quant_for_ifast};
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use sampling::downsample_chroma_block_h2v1_ssse3;
use sampling::{
    convert_to_ycbcr, convert_to_ycbcr_padded, downsample_chroma_block, extract_block,
    fullsize_smooth_plane, h2v2_smooth_downsample_plane, pad_plane_to_mcu_grid,
};

use baseline::compress_cmyk;
use optimized::{gather_block, gather_downsampled_block};
use progressive::CompLayout;
use progressive_entropy::{
    emit_buffered_bits, encode_progressive_dc_scan, progressive_fdct_chroma_block,
    progressive_fdct_y_block,
};

pub use arithmetic::{
    compress_arithmetic, compress_arithmetic_progressive,
    compress_arithmetic_progressive_rgb_direct, compress_arithmetic_rgb_direct,
};
pub use baseline::{
    compress, compress_custom_huffman, compress_custom_quant, compress_rgb_direct,
    compress_rgb_direct_with_params, compress_with_params, compress_with_restart,
};
pub use custom_sampling::compress_custom_sampling;
pub use lossless::{
    compress_lossless, compress_lossless_arithmetic, compress_lossless_extended,
    compress_lossless_extended_precision,
};
pub use metadata::{
    compress_with_metadata, inject_comment, inject_metadata, inject_metadata_full,
    inject_saved_markers,
};
pub use optimized::{compress_optimized, compress_optimized_with_params};
pub use progressive::{
    compress_progressive, compress_progressive_custom, compress_progressive_custom_with_restart,
    compress_progressive_rgb_direct, compress_progressive_with_restart,
};
pub use raw::compress_raw;

pub(crate) use progressive_entropy::{
    emit_eobrun, emit_eobrun_with_corr, encode_ac_first_block, encode_ac_refine_block,
    MAX_CORR_BITS,
};
