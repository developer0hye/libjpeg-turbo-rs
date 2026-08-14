//! P4-139 chunk 2: the root crate's span arithmetic, and the five defects
//! adopting [`ImageLayout`] uncovered.
//!
//! The C-ABI crate's half of this adoption is pinned by
//! `crates/libjpeg-turbo-rs-capi/tests/capi_layout_adoption.rs` (chunk 1).
//! This file is its root-crate counterpart: every case below is a *caller
//! error* — a crop past the image, a short pixel buffer, a plane geometry
//! whose product does not exist — that the library answered with a panic or
//! with silent wrapping instead of the typed error its signature already
//! promised.
//!
//! Each test names what was observed before the fix, because "returns an
//! error" is the same assertion whether the old behaviour was a panic, a
//! wrap, or an unrelated error raised later by accident.

use libjpeg_turbo_rs::api::encoder::Encoder;
use libjpeg_turbo_rs::api::scanline::ScanlineDecoder;
use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{compress_raw, JpegError, PixelFormat, ScalingFactor};

/// 640x480 4:2:0, so the iMCU width is 16 and crop alignment is visible.
const GRADIENT_640X480: &[u8] = include_bytes!("fixtures/gradient_640x480.jpg");

// ---------------------------------------------------------------------------
// Defect 1: StreamingDecoder::crop_scanline underflows past the right edge
// ---------------------------------------------------------------------------

/// `crop_scanline` computed `header.width - aligned_x` with no proof that
/// `aligned_x <= width`. Observed before the fix on a 640-wide image with
/// `x = 700`: `aligned_x` is 688, and `640 - 688` panicked with "attempt to
/// subtract with overflow" in debug — in release it wraps to
/// 18446744073709551568, which `min` then discards, reporting `xoffset = 688`
/// and `width = 32`: a window wholly outside a 640-pixel image, handed back as
/// success. The function already returns `Result`, so the caller error has
/// somewhere honest to go.
#[test]
fn crop_scanline_past_the_right_edge_is_an_error_not_an_underflow() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    assert_eq!(decoder.header().width, 640);

    let mut xoffset: usize = 700;
    let mut width: usize = 16;
    let err: JpegError = decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect_err("x=700 lies past the 640-pixel image");
    assert!(
        matches!(err, JpegError::Unsupported(_)),
        "want Unsupported for an out-of-range crop origin, got {err:?}"
    );
}

/// The other end of the same arithmetic: `xoffset + width` was an unchecked
/// sum. Observed before the fix with `xoffset = 16, width = usize::MAX`: the
/// sum panicked with "attempt to add with overflow" in debug, and in release
/// wrapped to 15, which `div_ceil` turned into a *small* aligned end — a
/// window of 16 pixels reported for a request that asked for the whole
/// address space.
///
/// The offset has to be small and the width huge, not the other way round:
/// the preceding origin check already rejects an offset past the image, so an
/// `xoffset` near `usize::MAX` never reaches this addition and would leave the
/// `checked_add` untested.
#[test]
fn crop_scanline_refuses_an_origin_plus_width_that_cannot_be_represented() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");

    let mut xoffset: usize = 16;
    let mut width: usize = usize::MAX;
    let err: JpegError = decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect_err("xoffset + width is not representable");
    match err {
        JpegError::Unsupported(ref message) => assert!(
            message.contains("not representable"),
            "want the extent error, not the origin error; got {message:?}"
        ),
        other => panic!("want Unsupported, got {other:?}"),
    }
}

/// The fix must not narrow what already worked: an in-range crop still aligns
/// down to the iMCU column boundary and reports the aligned window.
#[test]
fn crop_scanline_still_aligns_an_in_range_window_down_to_the_imcu_boundary() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");

    let mut xoffset: usize = 100;
    let mut width: usize = 200;
    decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect("an in-range crop is accepted");
    // 100 aligns down to 96; 100+200=300 aligns up to 304; 304-96 = 208.
    assert_eq!(xoffset, 96);
    assert_eq!(width, 208);
}

/// A window ending exactly at the right edge is in range, and the reported
/// width is clamped by the image rather than by the aligned end.
#[test]
fn crop_scanline_clamps_the_aligned_window_at_the_image_edge() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");

    let mut xoffset: usize = 632;
    let mut width: usize = 8;
    decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect("a window ending at the right edge is accepted");
    assert_eq!(xoffset, 624);
    assert_eq!(width, 16);
}

/// The bound the new guard measures against has to be the one `set_crop`
/// consumes, and that is the **output** width, not `header.width`.
///
/// `crop_scanline` used the unscaled width and a hard-coded block size of 8.
/// At 1/1 those agree with the output stage, which is why nothing noticed —
/// but under an upscaled decode a perfectly valid origin sits past
/// `header.width`, and the guard added for the underflow would have rejected
/// it. At 2/1 a 640-wide image emits 1280 columns, so `xoffset = 800` is in
/// range and must be accepted, aligned to the *scaled* 32-pixel iMCU.
#[test]
fn crop_scanline_bounds_and_aligns_in_output_space_not_header_space() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    decoder.set_scale(ScalingFactor::new(2, 1));

    let mut xoffset: usize = 800;
    let mut width: usize = 16;
    decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect("800 is inside the 1280-column output of a 2/1 decode");
    // Scaled iMCU width is max_h(2) * block_size(16) = 32.
    assert_eq!(xoffset, 800, "800 is already on a 32-pixel boundary");
    assert_eq!(width, 32);
}

/// The downscaled direction moves the alignment grid too, and there the two
/// formulations give genuinely different windows.
#[test]
fn crop_scanline_aligns_to_the_scaled_imcu_grid_when_downscaling() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    decoder.set_scale(ScalingFactor::new(1, 2));

    let mut xoffset: usize = 300;
    let mut width: usize = 16;
    decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect("300 is inside the 320-column output of a 1/2 decode");
    // Scaled iMCU width is max_h(2) * block_size(4) = 8, and the window is
    // clamped at the 320-column output edge. Against the unscaled grid this
    // reported (288, 32) — a different window on a different grid.
    assert_eq!(xoffset, 296);
    assert_eq!(width, 24);
}

/// And an origin past the *output* edge is still refused under scaling.
#[test]
fn crop_scanline_refuses_an_origin_past_the_scaled_output_edge() {
    let mut decoder: StreamingDecoder<'_> =
        StreamingDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    decoder.set_scale(ScalingFactor::new(1, 2));

    let mut xoffset: usize = 400;
    let mut width: usize = 16;
    let err: JpegError = decoder
        .crop_scanline(&mut xoffset, &mut width)
        .expect_err("400 lies past the 320-column output");
    assert!(
        matches!(err, JpegError::Unsupported(ref m) if m.contains("output width")),
        "want the origin error naming the output width, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Defect 2: ScanlineDecoder's crop bounds check wraps
// ---------------------------------------------------------------------------

/// `apply_horizontal_crop` guarded with `x + width > img.width` on an
/// unchecked sum. Observed before the fix with `x = usize::MAX - 7` and
/// `width = 8`: the sum wraps to 0, `0 > 640` is false, the guard passes —
/// and the very next statement, `y * src_row_bytes + x * bpp`, indexed the
/// decoded image at a wrapped offset and panicked with a slice range error.
/// The guard exists precisely to turn this into `Unsupported`.
#[test]
fn scanline_crop_origin_that_wraps_the_bounds_check_is_still_refused() {
    let mut decoder: ScanlineDecoder<'_> =
        ScanlineDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    decoder.set_crop_x(usize::MAX - 7, 8);

    let err: JpegError = decoder
        .finish()
        .expect_err("a crop origin near usize::MAX cannot lie inside a 640-pixel row");
    assert!(
        matches!(err, JpegError::Unsupported(_)),
        "want Unsupported for a wrapping crop window, got {err:?}"
    );
}

/// The ordinary out-of-range crop keeps its existing error, so the checked
/// rewrite did not move the boundary for well-behaved inputs.
#[test]
fn scanline_crop_past_the_right_edge_keeps_its_error() {
    let mut decoder: ScanlineDecoder<'_> =
        ScanlineDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    decoder.set_crop_x(600, 100);

    let err: JpegError = decoder.finish().expect_err("600+100 exceeds width 640");
    assert!(
        matches!(err, JpegError::Unsupported(_)),
        "want Unsupported, got {err:?}"
    );
}

/// And an in-range crop still produces the cropped image.
#[test]
fn scanline_crop_in_range_still_crops() {
    let mut decoder: ScanlineDecoder<'_> =
        ScanlineDecoder::new(GRADIENT_640X480).expect("fixture decodes");
    decoder.set_crop_x(64, 128);

    let image = decoder.finish().expect("an in-range crop decodes");
    assert_eq!(image.width, 128);
    assert_eq!(image.height, 480);
    assert_eq!(
        image.data.len(),
        128 * 480 * image.pixel_format.bytes_per_pixel()
    );
}

// ---------------------------------------------------------------------------
// Defect 3: Encoder reads rows before the per-path size check runs
// ---------------------------------------------------------------------------

/// Every encode path validates `pixels.len()` against `width * height * bpp`
/// — but only *after* `Encoder::encode` has already rearranged the caller's
/// buffer. Observed before the fix: `bottom_up(true)` on a buffer one row
/// short panicked inside `flip_rows` with a slice range error, because
/// `flip_rows` walks `height` rows of `width * bpp` bytes with no check of
/// its own. The pipeline's `BufferTooSmall` was unreachable for this path.
#[test]
fn bottom_up_encode_of_a_short_buffer_reports_buffer_too_small() {
    let width: usize = 16;
    let height: usize = 16;
    let bpp: usize = 3;
    let short: Vec<u8> = vec![0u8; width * (height - 1) * bpp];

    let err: JpegError = Encoder::new(&short, width, height, PixelFormat::Rgb)
        .bottom_up(true)
        .encode()
        .expect_err("one row short");
    assert!(
        matches!(err, JpegError::BufferTooSmall { need, got }
            if need == width * height * bpp && got == short.len()),
        "want BufferTooSmall{{need: {}, got: {}}}, got {err:?}",
        width * height * bpp,
        short.len()
    );
}

/// The same hole on the colour-to-grayscale path: the RGB arm slices
/// `after_fancy[src_off..src_off + width * 3]` per row before any path has
/// checked the length. Observed before the fix: a slice range panic.
#[test]
fn grayscale_from_color_encode_of_a_short_buffer_reports_buffer_too_small() {
    let width: usize = 16;
    let height: usize = 16;
    let bpp: usize = 3;
    let short: Vec<u8> = vec![0u8; width * height * bpp - 1];

    let err: JpegError = Encoder::new(&short, width, height, PixelFormat::Rgb)
        .grayscale_from_color(true)
        .encode()
        .expect_err("one byte short");
    assert!(
        matches!(err, JpegError::BufferTooSmall { need, got }
            if need == width * height * bpp && got == short.len()),
        "want BufferTooSmall, got {err:?}"
    );
}

/// The fancy-downsampling prefilter reads the whole buffer too, and it runs
/// before every mode dispatch.
#[test]
fn fancy_downsampling_encode_of_a_short_buffer_reports_buffer_too_small() {
    let width: usize = 32;
    let height: usize = 32;
    let bpp: usize = 3;
    let short: Vec<u8> = vec![0u8; width * height * bpp / 2];

    let err: JpegError = Encoder::new(&short, width, height, PixelFormat::Rgb)
        .fancy_downsampling(true)
        .encode()
        .expect_err("half a buffer");
    assert!(
        matches!(err, JpegError::BufferTooSmall { .. }),
        "want BufferTooSmall, got {err:?}"
    );
}

/// The buffer check must reject only what the pipeline would have rejected.
/// An exactly-sized buffer still encodes on the same paths.
#[test]
fn an_exactly_sized_buffer_still_encodes_on_every_rearranging_path() {
    let width: usize = 16;
    let height: usize = 16;
    let exact: Vec<u8> = vec![128u8; width * height * 3];

    for (label, jpeg) in [
        (
            "bottom_up",
            Encoder::new(&exact, width, height, PixelFormat::Rgb)
                .bottom_up(true)
                .encode(),
        ),
        (
            "grayscale_from_color",
            Encoder::new(&exact, width, height, PixelFormat::Rgb)
                .grayscale_from_color(true)
                .encode(),
        ),
        (
            "fancy_downsampling",
            Encoder::new(&exact, width, height, PixelFormat::Rgb)
                .fancy_downsampling(true)
                .encode(),
        ),
        (
            "plain",
            Encoder::new(&exact, width, height, PixelFormat::Rgb).encode(),
        ),
    ] {
        let bytes: Vec<u8> = jpeg.unwrap_or_else(|e| panic!("{label} must encode: {e:?}"));
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "{label} must emit SOI");
    }
}

/// The buffer check must not become the *first* thing `encode` reports.
///
/// A new gate at the top of a function does not only add an error — it takes
/// precedence over every error that used to come first, and a caller who was
/// told "your lossless predictor is 9" would now be told "your buffer is
/// short" about a buffer they had not got to yet. Each row below pairs an
/// option error with a buffer that is *also* wrong, and asserts the option
/// error still wins. This is what forces the length check to live in the three
/// helpers that read rows rather than in one up-front gate.
#[test]
fn an_option_error_still_precedes_the_buffer_check() {
    let short: Vec<u8> = vec![0u8; 10];

    // An out-of-range lossless predictor is validated inside the lossless
    // encoder, ahead of its own dimension and buffer checks.
    let err: JpegError = Encoder::new(&short, 16, 16, PixelFormat::Rgb)
        .lossless(true)
        .lossless_predictor(9)
        .encode()
        .expect_err("predictor 9 is out of range");
    assert!(
        matches!(err, JpegError::Unsupported(ref m) if m.contains("predictor")),
        "the predictor error must still win over BufferTooSmall, got {err:?}"
    );

    // The same, with a zero dimension rather than a short buffer.
    let err: JpegError = Encoder::new(&short, 0, 0, PixelFormat::Rgb)
        .lossless(true)
        .lossless_predictor(9)
        .encode()
        .expect_err("predictor 9 is out of range");
    assert!(
        matches!(err, JpegError::Unsupported(ref m) if m.contains("predictor")),
        "the predictor error must still win over the dimension error, got {err:?}"
    );

    // An option combination `encode` itself refuses, ahead of any dispatch.
    let err: JpegError = Encoder::new(&short, 16, 16, PixelFormat::Rgb)
        .smoothing_factor(50)
        .progressive(true)
        .encode()
        .expect_err("smoothing is unsupported with progressive");
    assert!(
        matches!(err, JpegError::Unsupported(ref m) if m.contains("smoothing_factor")),
        "the smoothing error must still win over BufferTooSmall, got {err:?}"
    );

    // And the dimension errors keep beating the buffer error, because both
    // are reported by the same pipeline prologue in the same order.
    let err: JpegError = Encoder::new(&short, 0, 16, PixelFormat::Rgb)
        .encode()
        .expect_err("zero width");
    assert!(
        matches!(err, JpegError::CorruptData(ref m) if m.contains("non-zero")),
        "want the dimension error, got {err:?}"
    );
    let err: JpegError = Encoder::new(&short, 65536, 16, PixelFormat::Rgb)
        .encode()
        .expect_err("width past the SOF limit");
    assert!(
        matches!(err, JpegError::CorruptData(ref m) if m.contains("65535")),
        "want the dimension error, got {err:?}"
    );
}

/// A buffer *larger* than the geometry needs is legal — callers pass strided
/// or over-allocated buffers — so the check is a floor, not an equality.
#[test]
fn an_oversized_buffer_is_still_accepted() {
    let width: usize = 16;
    let height: usize = 16;
    let oversized: Vec<u8> = vec![64u8; width * height * 3 + 4096];

    let bytes: Vec<u8> = Encoder::new(&oversized, width, height, PixelFormat::Rgb)
        .bottom_up(true)
        .encode()
        .expect("an oversized buffer encodes");
    assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
}

// ---------------------------------------------------------------------------
// Defects 4 and 5: size products that do not exist — the raw-plane loops, and
// the YUV conversion entry gate
// ---------------------------------------------------------------------------

/// `compress_raw` takes the plane dimensions from the caller and applied no
/// cap at all, then sized the guard with an unchecked
/// `plane_widths[i] * plane_heights[i]`. Observed before the fix with
/// `usize::MAX / 4 + 1` by `4`: the product wraps to exactly 0, so
/// `plane.len() < 0` is false and an **empty** plane passed the buffer check
/// that exists to stop exactly that. The wrap is exact on 32-bit and 64-bit
/// alike, so the shape is not a pointer-width curiosity.
#[test]
fn compress_raw_refuses_a_plane_geometry_whose_product_wraps() {
    let empty: &[u8] = &[];
    let planes: [&[u8]; 1] = [empty];
    let widths: [usize; 1] = [usize::MAX / 4 + 1];
    let heights: [usize; 1] = [4];
    assert_eq!(
        widths[0].wrapping_mul(heights[0]),
        0,
        "the test's premise: this product wraps to zero"
    );

    let err: JpegError = compress_raw(
        &planes,
        &widths,
        &heights,
        8,
        8,
        75,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect_err("a plane whose size is not representable cannot be encoded");
    assert!(
        matches!(err, JpegError::LimitExceeded { .. }),
        "want LimitExceeded for an unrepresentable plane span, got {err:?}"
    );
}

/// The 12-bit twin has the same signature and the same unchecked product.
/// Observed before the fix: "attempt to multiply with overflow" at
/// `raw_data_12.rs:485` in debug; a wrap to zero in release.
///
/// The factors differ from the 8-bit case only because this path *does*
/// cross-check the Y plane against the image dimensions, so the second factor
/// has to stay above the image height while the product still wraps:
/// `2^60 x 16` is exactly `2^64`.
#[test]
fn compress_raw_12_refuses_a_plane_geometry_whose_product_wraps() {
    let empty: &[i16] = &[];
    let planes: [&[i16]; 1] = [empty];
    let widths: [usize; 1] = [usize::MAX / 16 + 1];
    let heights: [usize; 1] = [16];
    assert_eq!(
        widths[0].wrapping_mul(heights[0]),
        0,
        "the test's premise: this product wraps to zero"
    );

    let err: JpegError = libjpeg_turbo_rs::raw_data_12::compress_raw_12(
        &planes,
        &widths,
        &heights,
        8,
        8,
        75,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect_err("a plane whose size is not representable cannot be encoded");
    assert!(
        matches!(err, JpegError::LimitExceeded { .. }),
        "want LimitExceeded, got {err:?}"
    );
}

/// A short-but-representable plane keeps reporting `BufferTooSmall`, which is
/// the diagnosis the caller can act on. Routing the product through the
/// checked helper must not collapse the two failures into one code.
#[test]
fn compress_raw_still_reports_buffer_too_small_for_a_representable_short_plane() {
    let plane: Vec<u8> = vec![0u8; 8 * 8 - 1];
    let planes: [&[u8]; 1] = [&plane];

    let err: JpegError = compress_raw(
        &planes,
        &[8],
        &[8],
        8,
        8,
        75,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect_err("one sample short");
    assert!(
        matches!(err, JpegError::BufferTooSmall { need, got } if need == 64 && got == 63),
        "want BufferTooSmall{{need: 64, got: 63}}, got {err:?}"
    );
}

/// The YUV conversion entry gate has the same shape, and the same defect:
/// `validate_pixel_buffer` sized `width * height * bpp` unchecked and used it
/// as the sole bound on the caller's buffer. Observed before the fix with
/// `(usize::MAX / 4 + 1)` by `4` at `Rgb`: "attempt to multiply with overflow"
/// in debug; in release the product wraps to 0, so an **empty** buffer passed
/// and the per-row loops indexed it.
#[test]
fn yuv_encode_refuses_a_geometry_whose_pixel_span_is_not_representable() {
    let width: usize = usize::MAX / 4 + 1;
    let empty: &[u8] = &[];

    let err: JpegError = libjpeg_turbo_rs::api::yuv::encode_yuv_planes(
        empty,
        width,
        4,
        PixelFormat::Rgb,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect_err("a pixel span that is not representable cannot be converted");
    assert!(
        matches!(err, JpegError::LimitExceeded { .. }),
        "want LimitExceeded for an unrepresentable source span, got {err:?}"
    );
}

/// The gate keeps reporting the ordinary short buffer as `BufferTooSmall`, so
/// routing it through the layout did not collapse two diagnoses into one.
#[test]
fn yuv_encode_still_reports_buffer_too_small_for_a_representable_short_buffer() {
    let short: Vec<u8> = vec![0u8; 16 * 16 * 3 - 1];

    let err: JpegError = libjpeg_turbo_rs::api::yuv::encode_yuv_planes(
        &short,
        16,
        16,
        PixelFormat::Rgb,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect_err("one byte short");
    assert!(
        matches!(err, JpegError::BufferTooSmall { need, got } if need == 768 && got == 767),
        "want BufferTooSmall{{need: 768, got: 767}}, got {err:?}"
    );
}

/// And a well-formed grayscale plane still encodes, so the new refusal did
/// not swallow the legitimate path.
#[test]
fn compress_raw_still_encodes_a_well_formed_grayscale_plane() {
    let plane: Vec<u8> = vec![96u8; 16 * 16];
    let planes: [&[u8]; 1] = [&plane];

    let jpeg: Vec<u8> = compress_raw(
        &planes,
        &[16],
        &[16],
        16,
        16,
        75,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("a well-formed plane encodes");
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
}
