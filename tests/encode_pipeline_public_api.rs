//! Compile-time characterization of the established encoder pipeline API.
//!
//! P4-40 is an internal module split. Downstream callers must continue to
//! resolve every existing public item through `encode::pipeline`, with the
//! same function signatures and `CompressParams` surface.

use libjpeg_turbo_rs::encode::pipeline::{
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
use libjpeg_turbo_rs::{
    DctMethod, HuffmanTableDef, PixelFormat, Result, SavedMarker, ScanScript, Subsampling,
};
use std::marker::PhantomData;

type QuantTables = [Option<[u16; 64]>; 4];
type HuffmanTables = [Option<HuffmanTableDef>; 4];
type JpegResult = Result<Vec<u8>>;

type CompressFn = fn(&[u8], usize, usize, PixelFormat, u8, Subsampling, DctMethod) -> JpegResult;
type ParamsFn = for<'a, 'b> fn(&'a CompressParams<'b>) -> JpegResult;
type ParamsIccFn = for<'a, 'b, 'c> fn(&'a CompressParams<'b>, Option<&'c [u8]>) -> JpegResult;

fn assert_compress_params_builder_signatures<'a>(_: PhantomData<&'a ()>) {
    let _: fn(&'a [u8], usize, usize, PixelFormat, u8, Subsampling) -> CompressParams<'a> =
        CompressParams::<'a>::new;
    let _: fn(CompressParams<'a>, DctMethod) -> CompressParams<'a> =
        CompressParams::<'a>::dct_method;
    let _: fn(CompressParams<'a>, u16) -> CompressParams<'a> =
        CompressParams::<'a>::restart_interval;
    let _: fn(CompressParams<'a>, &'a QuantTables) -> CompressParams<'a> =
        CompressParams::<'a>::custom_quant;
    let _: fn(CompressParams<'a>, &'a HuffmanTables, &'a HuffmanTables) -> CompressParams<'a> =
        CompressParams::<'a>::custom_huffman;
    let _: fn(CompressParams<'a>, bool) -> CompressParams<'a> =
        CompressParams::<'a>::optimize_huffman;
    let _: fn(CompressParams<'a>, u8) -> CompressParams<'a> =
        CompressParams::<'a>::smoothing_factor;
}

#[test]
fn established_pipeline_function_signatures_are_unchanged() {
    let _: CompressFn = compress;
    let _: ParamsFn = compress_with_params;
    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        &HuffmanTables,
        &HuffmanTables,
    ) -> JpegResult = compress_custom_huffman;
    let _: fn(&[u8], usize, usize, PixelFormat, u8, Subsampling, &QuantTables) -> JpegResult =
        compress_custom_quant;
    let _: fn(&[u8], usize, usize, PixelFormat, u8, Subsampling, u16, DctMethod) -> JpegResult =
        compress_with_restart;
    let _: fn(&[u8], usize, usize, u8, DctMethod, Option<&[u8]>) -> JpegResult =
        compress_rgb_direct;
    let _: ParamsIccFn = compress_rgb_direct_with_params;

    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        DctMethod,
        u16,
        Option<&QuantTables>,
    ) -> JpegResult = compress_arithmetic;
    let _: ParamsIccFn = compress_arithmetic_rgb_direct;
    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        DctMethod,
        u16,
        u16,
        Option<&QuantTables>,
    ) -> JpegResult = compress_arithmetic_progressive;
    let _: for<'a, 'b, 'c> fn(&'a CompressParams<'b>, Option<&'c [u8]>, u16) -> JpegResult =
        compress_arithmetic_progressive_rgb_direct;

    let _: CompressFn = compress_progressive;
    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        DctMethod,
        u16,
        u16,
        Option<&QuantTables>,
    ) -> JpegResult = compress_progressive_with_restart;
    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        &[ScanScript],
        DctMethod,
    ) -> JpegResult = compress_progressive_custom;
    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        &[ScanScript],
        DctMethod,
        u16,
        u16,
        Option<&QuantTables>,
    ) -> JpegResult = compress_progressive_custom_with_restart;
    let _: for<'a, 'b, 'c> fn(&'a CompressParams<'b>, Option<&'c [u8]>, u16) -> JpegResult =
        compress_progressive_rgb_direct;

    let _: fn(&[u8], usize, usize, PixelFormat) -> JpegResult = compress_lossless;
    let _: fn(&[u8], usize, usize, PixelFormat, u8, u8, u16) -> JpegResult =
        compress_lossless_extended;
    let _: fn(&[u8], usize, usize, PixelFormat, u8, u8, u16, u8) -> JpegResult =
        compress_lossless_extended_precision;
    let _: fn(&[u8], usize, usize, PixelFormat, u8, u8) -> JpegResult =
        compress_lossless_arithmetic;

    let _: fn(&[u8], usize, usize, PixelFormat, u8, Subsampling, u8, DctMethod, u16) -> JpegResult =
        compress_optimized;
    let _: ParamsFn = compress_optimized_with_params;
    let _: fn(&[u8], usize, usize, PixelFormat, u8, &[(u8, u8)]) -> JpegResult =
        compress_custom_sampling;
    let _: fn(&[&[u8]], &[usize], &[usize], usize, usize, u8, Subsampling) -> JpegResult =
        compress_raw;

    let _: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
        Option<&[u8]>,
        Option<&[u8]>,
    ) -> JpegResult = compress_with_metadata;
    let _: fn(&[u8], Option<&[u8]>, Option<&[u8]>) -> JpegResult = inject_metadata;
    let _: fn(&[u8], Option<&[u8]>, Option<&[u8]>, Option<&[u8]>, Option<&[u8]>) -> JpegResult =
        inject_metadata_full;
    let _: fn(&[u8], &str) -> Vec<u8> = inject_comment;
    let _: fn(&[u8], &[SavedMarker]) -> Vec<u8> = inject_saved_markers;
    let _: fn(u16) -> (u16, u16, u16, i16) = compute_reciprocal;
}

#[test]
fn compress_params_fields_and_builders_are_unchanged() {
    let params = CompressParams {
        pixels: &[],
        width: 0,
        height: 0,
        pixel_format: PixelFormat::Rgb,
        quality: 75,
        subsampling: Subsampling::S444,
        dct_method: DctMethod::IsLow,
        restart_interval: 0,
        custom_quant: None,
        custom_dc_huffman: None,
        custom_ac_huffman: None,
        optimize_huffman: false,
        smoothing_factor: 0,
    };
    let CompressParams {
        pixels: _,
        width: _,
        height: _,
        pixel_format: _,
        quality: _,
        subsampling: _,
        dct_method: _,
        restart_interval: _,
        custom_quant: _,
        custom_dc_huffman: _,
        custom_ac_huffman: _,
        optimize_huffman: _,
        smoothing_factor: _,
    } = params;

    assert_compress_params_builder_signatures(PhantomData::<&'static ()>);
}
