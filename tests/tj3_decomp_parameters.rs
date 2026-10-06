//! P4-199 (#620): what a `TjHandle` decompress publishes, and which limits the
//! 12/16-bit entry points honour. P4-200 (#621), P4-203 (#625) and P4-142
//! close with it: they are the *values* of the same thirteen parameters and
//! the header-only route to them.
//!
//! Every expected tuple below is stock libjpeg-turbo 3.2.0's, measured by
//! `crates/libjpeg-turbo-rs-capi/examples/decomp_parameters_oracle.c` (which
//! `crates/libjpeg-turbo-rs-capi/tests/capi_decomp_parameters.rs` re-runs
//! against stock on every C-ABI pass). The tuples are written out here so the
//! contract also holds on a machine — and a target, `wasm32-wasip1` — with no
//! TurboJPEG install; fixtures are embedded for the same reason.

use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit_with_limits, decompress_16bit_with_limits,
};
use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
use libjpeg_turbo_rs::{CropRegion, DecodeLimits, JpegError, Subsampling};

/// `setDecompParameters`' thirteen (`turbojpeg.c:514-536`), in its order.
const PUBLISHED: [TjParam; 13] = [
    TjParam::Subsampling,
    TjParam::Width,
    TjParam::Height,
    TjParam::Precision,
    TjParam::ColorSpace,
    TjParam::Progressive,
    TjParam::Arithmetic,
    TjParam::Lossless,
    TjParam::LosslessPsv,
    TjParam::LosslessPt,
    TjParam::XDensity,
    TjParam::YDensity,
    TjParam::DensityUnits,
];

const GRAY: &[u8] = include_bytes!("fixtures/gray_8x8.jpg");
const DENSE: &[u8] = include_bytes!("fixtures/api_sequence_color_16x16_422_dense.jpg");
const PROG: &[u8] = include_bytes!("fixtures/photo_64x64_420_prog.jpg");
const ARITH: &[u8] =
    include_bytes!("fixtures/real_world/libjpeg_testimgari_227x149_arithmetic.jpg");
const PROG_ARITH: &[u8] = include_bytes!("fixtures/decomp_params_prog_arith_24x16_420.jpg");
const LOSSLESS8: &[u8] = include_bytes!("fixtures/decomp_params_lossless8_psv4_pt1_24x16.jpg");
const LOSSY12: &[u8] = include_bytes!("fixtures/real_world/libjpeg_testorig12_227x149_12bit.jpg");
const LOSSLESS16: &[u8] = include_bytes!("inputs/api_sequence_lossless16_gray_8x8.jpg");
const CMYK: &[u8] = include_bytes!("fixtures/real_world/pil_cmyk.jpg");

/// `(label, jpeg, stock 3.2.0's thirteen after tj3DecompressHeader and after
/// the matching tj3Decompress*)`.
const EXPECTED: [(&str, &[u8], [i32; 13]); 9] = [
    ("gray", GRAY, [3, 8, 8, 8, 2, 0, 0, 0, 0, 0, 1, 1, 0]),
    ("dense", DENSE, [1, 16, 16, 8, 1, 0, 0, 0, 0, 0, 72, 71, 1]),
    // LOSSLESSPT is the first scan's `Al` — the DC scan's point transform.
    ("prog", PROG, [2, 64, 64, 8, 1, 1, 0, 0, 0, 1, 1, 1, 0]),
    ("arith", ARITH, [2, 227, 149, 8, 1, 0, 1, 0, 0, 0, 1, 1, 0]),
    (
        "progarith",
        PROG_ARITH,
        [2, 24, 16, 8, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    ),
    (
        "lossless8",
        LOSSLESS8,
        [0, 24, 16, 8, 0, 0, 0, 1, 4, 1, 1, 1, 0],
    ),
    (
        "lossy12",
        LOSSY12,
        [2, 227, 149, 12, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    ),
    (
        "lossless16",
        LOSSLESS16,
        [3, 8, 8, 16, 2, 0, 0, 1, 1, 0, 1, 1, 0],
    ),
    ("cmyk", CMYK, [0, 100, 100, 8, 4, 0, 0, 0, 0, 0, 1, 1, 0]),
];

fn published(handle: &TjHandle) -> [i32; 13] {
    PUBLISHED.map(|param| handle.get(param))
}

/// The documented routing: branch on `TJPARAM_PRECISION`.
fn routed_decompress(handle: &mut TjHandle, jpeg: &[u8], precision: i32) -> Result<(), JpegError> {
    if precision <= 8 {
        handle.decompress(jpeg).map(|_| ())
    } else if precision <= 12 {
        handle.decompress_12bit(jpeg).map(|_| ())
    } else {
        handle.decompress_16bit(jpeg).map(|_| ())
    }
}

fn expected_precision(expected: &[i32; 13]) -> i32 {
    expected[3]
}

/// Criteria 1 and 2: every entry point publishes all thirteen, with the
/// frame's values.
#[test]
fn every_decompress_entry_point_publishes_the_thirteen_set_decomp_parameters_writes() {
    for (label, jpeg, expected) in EXPECTED {
        let mut handle: TjHandle = TjHandle::new();
        routed_decompress(&mut handle, jpeg, expected_precision(&expected))
            .unwrap_or_else(|e| panic!("{label}: the matching entry point must decode: {e}"));
        assert_eq!(published(&handle), expected, "{label}: after decompress");
    }
}

/// P4-203 criterion 3 and P4-142: `decompress_header` publishes the same
/// thirteen at every precision, and the precision it publishes routes to an
/// entry point that accepts the frame.
#[test]
fn the_header_routes_every_precision_to_an_entry_point_that_accepts_it() {
    for (label, jpeg, expected) in EXPECTED {
        let mut handle: TjHandle = TjHandle::new();
        handle
            .decompress_header(jpeg)
            .unwrap_or_else(|e| panic!("{label}: header: {e}"));
        assert_eq!(published(&handle), expected, "{label}: after the header");
        let precision: i32 = handle.get(TjParam::Precision);
        routed_decompress(&mut handle, jpeg, precision)
            .unwrap_or_else(|e| panic!("{label}: routed by PRECISION={precision}: {e}"));
        assert_eq!(
            published(&handle),
            expected,
            "{label}: after the routed decode"
        );
    }
}

/// A decode replaces every one of the thirteen; none survives from the image
/// before. Upstream's `setDecompParameters` writes all of them unconditionally.
#[test]
fn each_decode_replaces_what_the_previous_one_published() {
    let mut handle: TjHandle = TjHandle::new();
    // Twice round, so every fixture follows one that differs from it.
    for _ in 0..2 {
        for (label, jpeg, expected) in EXPECTED {
            routed_decompress(&mut handle, jpeg, expected_precision(&expected))
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(published(&handle), expected, "{label}: on a reused handle");
        }
    }
}

/// P4-200: scaling and cropping change the output, not `JPEGWIDTH` /
/// `JPEGHEIGHT` (`this->jpegWidth = this->dinfo.image_width`).
#[test]
fn scaled_and_cropped_decodes_publish_the_frame_dimensions() {
    let mut scaled: TjHandle = TjHandle::new();
    scaled
        .set_scaling_factor(1, 2)
        .expect("1/2 is a supported factor");
    let image = scaled.decompress(PROG).expect("scaled decode");
    assert_eq!(
        (image.width, image.height),
        (32, 32),
        "the output is scaled"
    );
    assert_eq!(published(&scaled), EXPECTED[2].2);

    let mut cropped: TjHandle = TjHandle::new();
    cropped.decompress_header(PROG).expect("header");
    cropped.set_cropping_region(Some(CropRegion {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    }));
    let image = cropped.decompress(PROG).expect("cropped decode");
    assert_eq!((image.width, image.height), (8, 8), "the output is cropped");
    assert_eq!(published(&cropped), EXPECTED[2].2);
}

/// P4-200 criterion 6: a handle that has read nothing reports the "not read
/// yet" sentinel, as `tj3InitVersion` seeds it (`turbojpeg.c:600-601`).
#[test]
fn a_fresh_handle_reports_no_frame_dimensions() {
    let handle: TjHandle = TjHandle::new();
    assert_eq!(handle.get(TjParam::Width), -1);
    assert_eq!(handle.get(TjParam::Height), -1);
}

/// Issue #620: `decompress_16bit` and `decompress_12bit` refuse a frame over
/// the handle's `TJPARAM_MAXPIXELS`, having published first — upstream's
/// shared body calls `setDecompParameters` at `turbojpeg-mp.c:190` and refuses
/// at `:195-199`. Before the fix both read nothing from the handle and decoded.
#[test]
fn the_12_and_16_bit_entry_points_refuse_a_frame_over_maxpixels() {
    let mut handle16: TjHandle = TjHandle::new();
    handle16.set(TjParam::MaxPixels, 63).expect("MAXPIXELS");
    let refused = handle16.decompress_16bit(LOSSLESS16);
    assert!(
        matches!(
            refused,
            Err(JpegError::LimitExceeded {
                what: "total pixels",
                actual: 64,
                limit: 63
            })
        ),
        "an 8x8 16-bit frame over a 63-pixel ceiling must be refused: {refused:?}"
    );
    assert_eq!(
        published(&handle16),
        EXPECTED[7].2,
        "published before refusing"
    );
    handle16.set(TjParam::MaxPixels, 64).expect("MAXPIXELS");
    handle16
        .decompress_16bit(LOSSLESS16)
        .expect("at the ceiling it decodes");

    let mut handle12: TjHandle = TjHandle::new();
    handle12
        .set(TjParam::MaxPixels, 227 * 149 - 1)
        .expect("MAXPIXELS");
    let refused = handle12.decompress_12bit(LOSSY12);
    assert!(
        matches!(
            refused,
            Err(JpegError::LimitExceeded {
                what: "total pixels",
                ..
            })
        ),
        "a 227x149 12-bit frame over the ceiling must be refused: {refused:?}"
    );
    assert_eq!(
        published(&handle12),
        EXPECTED[6].2,
        "published before refusing"
    );
    handle12
        .set(TjParam::MaxPixels, 227 * 149)
        .expect("MAXPIXELS");
    handle12
        .decompress_12bit(LOSSY12)
        .expect("at the ceiling it decodes");
}

/// A 16-bit lossless gray frame whose output alone is `width * height * 2`
/// bytes: 1024 x 1024 is 2 MiB, between a 1 MiB and an 8 MiB budget.
fn lossless16_1024() -> Vec<u8> {
    let (width, height): (usize, usize) = (1024, 1024);
    let pixels: Vec<u16> = (0..width * height)
        .map(|i| ((i % width) * 61 + (i / width) * 17) as u16)
        .collect();
    compress_16bit(&pixels, width, height, 1, 1, 0).expect("16-bit lossless encode")
}

/// A 12-bit 4:4:4 colour frame: 512 x 512 x 3 components, staged three times
/// at two bytes a sample (component planes, upsampled planes, interleaved
/// result) — 4.5 MiB, between a 1 MiB and an 8 MiB budget.
fn lossy12_512() -> Vec<u8> {
    let (width, height): (usize, usize) = (512, 512);
    let pixels: Vec<i16> = (0..width * height * 3)
        .map(|i| ((i * 7) % 4096) as i16)
        .collect();
    compress_12bit(&pixels, width, height, 3, 90, Subsampling::S444).expect("12-bit encode")
}

/// Issue #620: `TJPARAM_MAXMEMORY` (megabytes) reaches the 12/16-bit paths.
#[test]
fn the_12_and_16_bit_entry_points_honour_maxmemory() {
    let jpeg16: Vec<u8> = lossless16_1024();
    let mut handle: TjHandle = TjHandle::new();
    handle.set(TjParam::MaxMemory, 1).expect("MAXMEMORY");
    let refused = handle.decompress_16bit(&jpeg16);
    assert!(
        matches!(
            refused,
            Err(JpegError::LimitExceeded {
                what: "estimated decode memory",
                actual: 2_097_152,
                limit: 1_048_576
            })
        ),
        "a 2 MiB 16-bit decode must be refused under a 1 MiB budget: {refused:?}"
    );
    handle.set(TjParam::MaxMemory, 8).expect("MAXMEMORY");
    handle.decompress_16bit(&jpeg16).expect("8 MiB is enough");

    let jpeg12: Vec<u8> = lossy12_512();
    let mut handle: TjHandle = TjHandle::new();
    handle.set(TjParam::MaxMemory, 1).expect("MAXMEMORY");
    let refused = handle.decompress_12bit(&jpeg12);
    assert!(
        matches!(
            refused,
            Err(JpegError::LimitExceeded {
                what: "estimated decode memory",
                actual: 4_718_592,
                limit: 1_048_576
            })
        ),
        "a 4.5 MiB 12-bit decode must be refused under a 1 MiB budget: {refused:?}"
    );
    handle.set(TjParam::MaxMemory, 8).expect("MAXMEMORY");
    handle.decompress_12bit(&jpeg12).expect("8 MiB is enough");
}

/// A 12-bit stream of three single-component scans, written by stock
/// `cjpeg -precision 12 -sample 1x1 -scans` with the script `0; 1; 2;`. A
/// single interleaved scan — the only shape the 16-bit decoder accepts — ends
/// the header walk at its SOS, so its scan count is always one; a
/// non-interleaved stream is what makes the walk count scans at all.
const NONINTERLEAVED12: &[u8] = include_bytes!("inputs/p4199_noninterleaved12_16x16_444.jpg");

/// `jpeg`'s first, interleaved SOS split into one single-component SOS per
/// component, each followed by a few bytes of entropy data — enough for the
/// header walk to count them, which is all a scan limit reads.
fn split_into_single_component_scans(jpeg: &[u8]) -> Vec<u8> {
    let sos: usize = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("an SOS marker");
    let count: usize = usize::from(jpeg[sos + 4]);
    let selectors: &[u8] = &jpeg[sos + 5..sos + 5 + 2 * count];
    let spectral: &[u8] = &jpeg[sos + 5 + 2 * count..sos + 5 + 2 * count + 3];
    let mut stream: Vec<u8> = jpeg[..sos].to_vec();
    for component in 0..count {
        // Ls = 2 + 1 + 2 + 3 for one component.
        stream.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01]);
        stream.extend_from_slice(&selectors[2 * component..2 * component + 2]);
        stream.extend_from_slice(spectral);
        stream.extend_from_slice(&[0x00; 4]);
    }
    stream.extend_from_slice(&[0xFF, 0xD9]);
    stream
}

/// Issue #620: `TJPARAM_SCANLIMIT` reaches the 12/16-bit paths. Both streams
/// carry three scans; under a limit of two the header walk refuses them, and
/// without one the refusal (if any) is something else — so `LimitExceeded`
/// below is the limit's and nothing else's.
#[test]
fn the_12_and_16_bit_entry_points_honour_scanlimit() {
    let pixels: Vec<u16> = (0..8usize * 8 * 3).map(|i| (i * 977) as u16).collect();
    let interleaved16: Vec<u8> = compress_16bit(&pixels, 8, 8, 3, 1, 0).expect("16-bit encode");
    let stream16: Vec<u8> = split_into_single_component_scans(&interleaved16);

    let mut unlimited: TjHandle = TjHandle::new();
    for (label, outcome) in [
        (
            "12-bit",
            unlimited.decompress_12bit(NONINTERLEAVED12).map(|_| ()),
        ),
        ("16-bit", unlimited.decompress_16bit(&stream16).map(|_| ())),
    ] {
        assert!(
            !matches!(outcome, Err(JpegError::LimitExceeded { .. })),
            "{label}: with no scan limit nothing may refuse for a limit: {outcome:?}"
        );
    }

    let mut limited: TjHandle = TjHandle::new();
    limited.set(TjParam::ScanLimit, 2).expect("SCANLIMIT");
    for (label, refused) in [
        (
            "12-bit",
            limited.decompress_12bit(NONINTERLEAVED12).map(|_| ()),
        ),
        ("16-bit", limited.decompress_16bit(&stream16).map(|_| ())),
    ] {
        assert!(
            matches!(
                refused,
                Err(JpegError::LimitExceeded {
                    actual: 3,
                    limit: 2,
                    ..
                })
            ),
            "{label}: three scans over a limit of two must be refused: {refused:?}"
        );
    }
}

/// The Rust-native entry points take the same budget a `Decoder` does.
#[test]
fn the_precision_entry_points_apply_decode_limits() {
    let narrow: DecodeLimits = DecodeLimits {
        max_width: 7,
        ..DecodeLimits::default()
    };
    assert!(matches!(
        decompress_16bit_with_limits(LOSSLESS16, &narrow),
        Err(JpegError::LimitExceeded {
            what: "image width",
            actual: 8,
            limit: 7
        })
    ));
    assert!(matches!(
        decompress_12bit_with_limits(LOSSY12, &narrow),
        Err(JpegError::LimitExceeded {
            what: "image width",
            actual: 227,
            limit: 7
        })
    ));
    let short: DecodeLimits = DecodeLimits {
        max_height: 148,
        ..DecodeLimits::default()
    };
    assert!(matches!(
        decompress_12bit_with_limits(LOSSY12, &short),
        Err(JpegError::LimitExceeded {
            what: "image height",
            actual: 149,
            limit: 148
        })
    ));
    let defaults: DecodeLimits = DecodeLimits::default();
    let image16 = decompress_16bit_with_limits(LOSSLESS16, &defaults).expect("defaults accept");
    assert_eq!(
        image16.data,
        libjpeg_turbo_rs::decompress_16bit(LOSSLESS16)
            .expect("the limit-free spelling")
            .data,
        "the limits change what is refused, not what is decoded"
    );
}

/// `decompress_header` follows `tj3DecompressHeader`: no `TJPARAM_MAXPIXELS`
/// test there (`turbojpeg.c:1872-1927` has none), only in the decompress that
/// follows.
#[test]
fn the_header_is_read_under_maxpixels_and_the_decode_is_refused() {
    let mut handle: TjHandle = TjHandle::new();
    handle.set(TjParam::MaxPixels, 1).expect("MAXPIXELS");
    handle
        .decompress_header(PROG)
        .expect("tj3DecompressHeader has no pixel ceiling");
    assert_eq!(published(&handle), EXPECTED[2].2);
    assert!(matches!(
        handle.decompress(PROG),
        Err(JpegError::LimitExceeded {
            what: "total pixels",
            ..
        })
    ));
}

/// P4-142: the header is read without decoding. A stream whose entropy data
/// is garbage after a valid header fails to decode and still has a header —
/// `jpeg_read_header` stops at the first SOS.
#[test]
fn the_header_is_read_without_decoding_the_entropy_data() {
    let corrupt: Vec<u8> = corrupt_entropy(include_bytes!("fixtures/photo_64x64_420.jpg"));
    let mut handle: TjHandle = TjHandle::new();
    handle
        .decompress_header(&corrupt)
        .expect("the header is intact");
    assert_eq!(
        published(&handle),
        [2, 64, 64, 8, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    );
    assert!(
        TjHandle::new().decompress(&corrupt).is_err(),
        "the fixture must be one a decode refuses, or this proves nothing"
    );
}

/// Replace the first scan's entropy data with stuffed all-ones bytes
/// (`FF 00` pairs): every Huffman lookup then reads an all-ones code, which
/// no JPEG table assigns (ITU T.81 C.2), so the first block fails.
fn corrupt_entropy(jpeg: &[u8]) -> Vec<u8> {
    let first_sos: usize = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("an SOS marker");
    let segment_length: usize =
        usize::from(jpeg[first_sos + 2]) << 8 | usize::from(jpeg[first_sos + 3]);
    let entropy_start: usize = first_sos + 2 + segment_length;
    let entropy_len: usize = jpeg.len() - 2 - entropy_start;
    let mut stream: Vec<u8> = jpeg[..entropy_start].to_vec();
    for _ in 0..entropy_len / 2 {
        stream.extend_from_slice(&[0xFF, 0x00]);
    }
    stream.extend_from_slice(&[0xFF, 0xD9]);
    stream
}
