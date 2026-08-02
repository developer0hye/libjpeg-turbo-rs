//! Compile-time characterization of the established decoder pipeline API.
//!
//! P4-119 is an internal module split. Downstream callers must continue to
//! resolve the public surface through both `decode::pipeline` and the crate
//! root with the same signatures, fields, lifetimes, and callback bounds.

use libjpeg_turbo_rs::api::abbreviated::TablesOnlyState;
use libjpeg_turbo_rs::api::raw_data::RawImage;
use libjpeg_turbo_rs::decode::pipeline::{probe, Decoder, Image, ImageInfo, JpegInfo};
use libjpeg_turbo_rs::decode::resync::{
    DefaultResyncStrategy, RestartResyncStrategy, ResyncAction,
};
use libjpeg_turbo_rs::{
    ColorSpace, DctMethod, DecodeLimits, DecodeWarning, DensityInfo, FrameHeader, MarkerSaveConfig,
    PixelFormat, Result, SavedMarker, ScalingFactor, Subsampling,
};
use std::{cell::Cell, fmt::Debug, marker::PhantomData};

type MarkerProcessor = fn(&[u8]) -> Option<Vec<u8>>;

fn assert_send<T: Send>() {}
fn assert_debug<T: Debug>() {}
fn assert_debug_clone_eq<T: Debug + Clone + Eq>() {}

struct SendNotSyncStrategy(Cell<u8>);

impl RestartResyncStrategy for SendNotSyncStrategy {
    fn on_desync(&mut self, _expected: u8, _found: Option<u8>) -> ResyncAction {
        self.0.set(self.0.get().wrapping_add(1));
        ResyncAction::Continue
    }
}

fn assert_image_field_types(image: &Image) {
    let _: usize = image.width;
    let _: usize = image.height;
    let _: PixelFormat = image.pixel_format;
    let _: u8 = image.precision;
    let _: &Vec<u8> = &image.data;
    let _: &Option<Vec<u8>> = &image.icc_profile;
    let _: &Option<Vec<u8>> = &image.exif_data;
    let _: &Option<Vec<u8>> = &image.xmp_data;
    let _: &Option<Vec<u8>> = &image.iptc_data;
    let _: &Option<String> = &image.comment;
    let _: &DensityInfo = &image.density;
    let _: &Vec<SavedMarker> = &image.saved_markers;
    let _: &Vec<DecodeWarning> = &image.warnings;
}

fn assert_image_info_field_types(info: &ImageInfo) {
    let _: usize = info.width;
    let _: usize = info.height;
    let _: PixelFormat = info.pixel_format;
    let _: u8 = info.precision;
    let _: usize = info.bytes_written;
    let _: &Option<Vec<u8>> = &info.icc_profile;
    let _: &Option<Vec<u8>> = &info.exif_data;
    let _: &Option<Vec<u8>> = &info.xmp_data;
    let _: &Option<Vec<u8>> = &info.iptc_data;
    let _: &Option<String> = &info.comment;
    let _: &DensityInfo = &info.density;
    let _: &Vec<SavedMarker> = &info.saved_markers;
    let _: &Vec<DecodeWarning> = &info.warnings;
}

fn assert_jpeg_info_field_types(info: &JpegInfo) {
    let _: usize = info.width;
    let _: usize = info.height;
    let _: usize = info.components;
    let _: u8 = info.precision;
    let _: bool = info.progressive;
    let _: bool = info.lossless;
    let _: bool = info.arithmetic;
    let _: Subsampling = info.subsampling;
    let _: ColorSpace = info.color_space;
    let _: &Option<u8> = &info.exif_orientation;
    let _: bool = info.has_exif;
    let _: bool = info.has_icc;
    let _: bool = info.has_xmp;
    let _: bool = info.has_iptc;
    let _: &Option<String> = &info.comment;
    let _: &DensityInfo = &info.density;
}

fn assert_decoder_signatures<'a>(_: PhantomData<&'a ()>) {
    let _: fn(&'a [u8]) -> Result<Decoder<'a>> = Decoder::<'a>::new;
    let _: fn(&'a [u8], DecodeLimits) -> Result<Decoder<'a>> = Decoder::<'a>::new_with_limits;
    let _: fn(&'a [u8], &TablesOnlyState) -> Result<Decoder<'a>> = Decoder::<'a>::new_with_tables;

    let _: fn(&Decoder<'a>) -> Option<u8> = Decoder::<'a>::exif_orientation;
    let _: for<'b> fn(&'b Decoder<'a>) -> &'b FrameHeader = Decoder::<'a>::header;
    let _: for<'b> fn(&'b Decoder<'a>) -> &'b DensityInfo = Decoder::<'a>::density;
    let _: fn(&Decoder<'a>) -> bool = Decoder::<'a>::saw_jfif_marker;
    let _: fn(&Decoder<'a>) -> (u8, u8) = Decoder::<'a>::jfif_version;
    let _: fn(&Decoder<'a>) -> bool = Decoder::<'a>::is_arithmetic;
    let _: fn(&Decoder<'a>) -> usize = Decoder::<'a>::output_height;
    let _: for<'b> fn(&'b Decoder<'a>) -> &'b DecodeLimits = Decoder::<'a>::limits;
    let _: fn(&Decoder<'a>) -> ColorSpace = Decoder::<'a>::jpeg_color_space;
    let _: fn(&Decoder<'a>) -> Subsampling = Decoder::<'a>::jpeg_subsampling;
    let _: for<'b> fn(&'b Decoder<'a>) -> &'b [SavedMarker] = Decoder::<'a>::saved_markers;

    let _: fn(&mut Decoder<'a>, PixelFormat) = Decoder::<'a>::set_output_format;
    let _: fn(&mut Decoder<'a>, ScalingFactor) = Decoder::<'a>::set_scale;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_lenient;
    let _: fn(&mut Decoder<'a>, usize, usize) = Decoder::<'a>::set_crop;
    let _: fn(&mut Decoder<'a>, usize, usize) = Decoder::<'a>::set_crop_y;
    let _: fn(&mut Decoder<'a>, usize, usize, usize, usize) = Decoder::<'a>::set_crop_region;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_stop_on_warning;
    let _: fn(&mut Decoder<'a>, usize) = Decoder::<'a>::set_max_pixels;
    let _: fn(&mut Decoder<'a>, usize) = Decoder::<'a>::set_max_memory;
    let _: fn(&mut Decoder<'a>, u32) = Decoder::<'a>::set_scan_limit;
    let _: fn(&mut Decoder<'a>, DecodeLimits) = Decoder::<'a>::set_limits;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_fast_upsample;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_fast_dct;
    let _: fn(&mut Decoder<'a>, DctMethod) = Decoder::<'a>::set_dct_method;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_block_smoothing;
    let _: fn(&mut Decoder<'a>, ColorSpace) = Decoder::<'a>::set_output_colorspace;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_dither_565;
    let _: fn(&mut Decoder<'a>, bool) = Decoder::<'a>::set_merged_upsample;
    let _: fn(&mut Decoder<'a>, MarkerSaveConfig) = Decoder::<'a>::save_markers;

    let _: fn(Decoder<'a>, PixelFormat) -> Decoder<'a> = Decoder::<'a>::with_output_format;
    let _: fn(Decoder<'a>, ScalingFactor) -> Decoder<'a> = Decoder::<'a>::with_scale;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_lenient;
    let _: fn(Decoder<'a>, usize, usize) -> Decoder<'a> = Decoder::<'a>::with_crop;
    let _: fn(Decoder<'a>, usize, usize) -> Decoder<'a> = Decoder::<'a>::with_crop_y;
    let _: fn(Decoder<'a>, usize, usize, usize, usize) -> Decoder<'a> =
        Decoder::<'a>::with_crop_region;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_stop_on_warning;
    let _: fn(Decoder<'a>, usize) -> Decoder<'a> = Decoder::<'a>::with_max_pixels;
    let _: fn(Decoder<'a>, usize) -> Decoder<'a> = Decoder::<'a>::with_max_memory;
    let _: fn(Decoder<'a>, u32) -> Decoder<'a> = Decoder::<'a>::with_scan_limit;
    let _: fn(Decoder<'a>, DecodeLimits) -> Decoder<'a> = Decoder::<'a>::with_limits;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_fast_upsample;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_fast_dct;
    let _: fn(Decoder<'a>, DctMethod) -> Decoder<'a> = Decoder::<'a>::with_dct_method;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_block_smoothing;
    let _: fn(Decoder<'a>, ColorSpace) -> Decoder<'a> = Decoder::<'a>::with_output_colorspace;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_dither_565;
    let _: fn(Decoder<'a>, bool) -> Decoder<'a> = Decoder::<'a>::with_merged_upsample;
    let _: fn(Decoder<'a>, MarkerSaveConfig) -> Decoder<'a> = Decoder::<'a>::with_save_markers;

    let _: fn(Decoder<'a>, u8, MarkerProcessor) -> Decoder<'a> =
        Decoder::<'a>::with_marker_processor::<MarkerProcessor>;
    let _: fn(&mut Decoder<'a>, u8, MarkerProcessor) =
        Decoder::<'a>::set_marker_processor::<MarkerProcessor>;
    let _: fn(Decoder<'a>, DefaultResyncStrategy) -> Decoder<'a> =
        Decoder::<'a>::with_resync_strategy::<DefaultResyncStrategy>;
    let _: fn(&mut Decoder<'a>, DefaultResyncStrategy) =
        Decoder::<'a>::set_resync_strategy::<DefaultResyncStrategy>;

    let _: fn(&'a [u8]) -> Result<Image> = Decoder::<'a>::decode;
    let _: fn(&'a [u8], PixelFormat) -> Result<Image> = Decoder::<'a>::decode_to;
    let _: fn(&Decoder<'a>) -> Result<Image> = Decoder::<'a>::decode_image;
    let _: fn(&Decoder<'a>) -> Result<usize> = Decoder::<'a>::output_buffer_size;
    let _: fn(&Decoder<'a>, &mut [u8]) -> Result<ImageInfo> = Decoder::<'a>::decode_image_into;
    let _: fn(Decoder<'a>) -> Result<RawImage> = Decoder::<'a>::decode_raw;
}

#[test]
fn established_pipeline_paths_and_signatures_are_unchanged() {
    let _: fn(&[u8]) -> Result<JpegInfo> = probe;
    let _: fn(&[u8]) -> Result<libjpeg_turbo_rs::JpegInfo> = libjpeg_turbo_rs::probe;

    let _: fn(&ImageInfo) -> Option<u8> = ImageInfo::exif_orientation;
    let _: fn(&Image) -> Option<&[u8]> = Image::icc_profile;
    let _: fn(&Image) -> Option<&[u8]> = Image::exif_data;
    let _: fn(&Image) -> Option<&[u8]> = Image::xmp_data;
    let _: fn(&Image) -> Option<&[u8]> = Image::iptc_data;
    let _: fn(&Image) -> Option<u8> = Image::exif_orientation;
    let _: fn(&Image) -> &[SavedMarker] = Image::markers;
    let _: fn(&Image) -> &[u8] = Image::as_bytes;
    let _: fn(Image) -> Vec<u8> = Image::into_vec;
    let _: fn(Image) -> Image = Image::apply_orientation;
    let _: fn(Image, u8) -> Image = Image::apply_orientation_value;

    fn pipeline_to_root<'a>(decoder: Decoder<'a>) -> libjpeg_turbo_rs::Decoder<'a> {
        decoder
    }
    fn image_to_root(image: Image) -> libjpeg_turbo_rs::Image {
        image
    }
    fn image_info_to_root(info: ImageInfo) -> libjpeg_turbo_rs::ImageInfo {
        info
    }
    fn jpeg_info_to_root(info: JpegInfo) -> libjpeg_turbo_rs::JpegInfo {
        info
    }
    let _: for<'a> fn(Decoder<'a>) -> libjpeg_turbo_rs::Decoder<'a> = pipeline_to_root;
    let _: fn(Image) -> libjpeg_turbo_rs::Image = image_to_root;
    let _: fn(ImageInfo) -> libjpeg_turbo_rs::ImageInfo = image_info_to_root;
    let _: fn(JpegInfo) -> libjpeg_turbo_rs::JpegInfo = jpeg_info_to_root;

    assert_send::<Decoder<'static>>();
    assert_debug_clone_eq::<Image>();
    assert_debug::<ImageInfo>();
    assert_debug_clone_eq::<JpegInfo>();

    assert_decoder_signatures(PhantomData::<&'static ()>);
}

#[test]
fn callbacks_accept_send_but_not_sync_implementations() {
    let jpeg = include_bytes!("fixtures/photo_64x64_420.jpg");
    let mut decoder = Decoder::new(jpeg).expect("fixture header");

    let calls = Cell::new(0u8);
    decoder.set_marker_processor(0xE1, move |_| {
        calls.set(calls.get().wrapping_add(1));
        None
    });
    decoder.set_resync_strategy(SendNotSyncStrategy(Cell::new(0)));
}

#[test]
fn public_pipeline_types_keep_their_canonical_paths() {
    assert_eq!(
        std::any::type_name::<Decoder<'static>>(),
        "libjpeg_turbo_rs::decode::pipeline::Decoder<'_>"
    );
    assert_eq!(
        std::any::type_name::<Image>(),
        "libjpeg_turbo_rs::decode::pipeline::Image"
    );
    assert_eq!(
        std::any::type_name::<ImageInfo>(),
        "libjpeg_turbo_rs::decode::pipeline::ImageInfo"
    );
    assert_eq!(
        std::any::type_name::<JpegInfo>(),
        "libjpeg_turbo_rs::decode::pipeline::JpegInfo"
    );
}

#[test]
fn public_image_and_probe_fields_are_unchanged() {
    let image = Image {
        width: 0,
        height: 0,
        pixel_format: PixelFormat::Rgb,
        precision: 8,
        data: Vec::new(),
        icc_profile: None,
        exif_data: None,
        xmp_data: None,
        iptc_data: None,
        comment: None,
        density: DensityInfo::default(),
        saved_markers: Vec::new(),
        warnings: Vec::<DecodeWarning>::new(),
    };
    assert_image_field_types(&image);
    let Image {
        width: _,
        height: _,
        pixel_format: _,
        precision: _,
        data: _,
        icc_profile: _,
        exif_data: _,
        xmp_data: _,
        iptc_data: _,
        comment: _,
        density: _,
        saved_markers: _,
        warnings: _,
    } = image;

    let image_info = ImageInfo {
        width: 0,
        height: 0,
        pixel_format: PixelFormat::Rgb,
        precision: 8,
        bytes_written: 0,
        icc_profile: None,
        exif_data: None,
        xmp_data: None,
        iptc_data: None,
        comment: None,
        density: DensityInfo::default(),
        saved_markers: Vec::new(),
        warnings: Vec::new(),
    };
    assert_image_info_field_types(&image_info);
    let ImageInfo {
        width: _,
        height: _,
        pixel_format: _,
        precision: _,
        bytes_written: _,
        icc_profile: _,
        exif_data: _,
        xmp_data: _,
        iptc_data: _,
        comment: _,
        density: _,
        saved_markers: _,
        warnings: _,
    } = image_info;

    let jpeg_info = JpegInfo {
        width: 0,
        height: 0,
        components: 0,
        precision: 8,
        progressive: false,
        lossless: false,
        arithmetic: false,
        subsampling: Subsampling::Unknown,
        color_space: ColorSpace::Unknown,
        exif_orientation: None,
        has_exif: false,
        has_icc: false,
        has_xmp: false,
        has_iptc: false,
        comment: None,
        density: DensityInfo::default(),
    };
    assert_jpeg_info_field_types(&jpeg_info);
    let JpegInfo {
        width: _,
        height: _,
        components: _,
        precision: _,
        progressive: _,
        lossless: _,
        arithmetic: _,
        subsampling: _,
        color_space: _,
        exif_orientation: _,
        has_exif: _,
        has_icc: _,
        has_xmp: _,
        has_iptc: _,
        comment: _,
        density: _,
    } = jpeg_info;
}
