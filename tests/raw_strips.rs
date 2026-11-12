//! `Decoder::into_raw_strips`: raw component planes delivered one iMCU row at
//! a time, borrowed from the reader, so a caller collects them into whatever
//! buffer layout it owns.
//!
//! The reference is libjpeg's own `jpeg_read_raw_data`, which returns the
//! same iMCU rows (`examples/raw_decode_c_oracle.c`). Every sample of each
//! component's active area must be identical; the MCU padding beyond it is
//! compared against `decode_raw`, whose padding the strips document as theirs.
//! The oracle runs with `do_block_smoothing = FALSE`: the reader, like
//! `decode_raw`, never smooths (P4-249).

mod helpers;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use helpers::c_oracle::{decode_with_raw_c_oracle, raw_decode_c_oracle, CRawPlane};
use libjpeg_turbo_rs::yuv::{decompress_to_yuv_planes, decompress_to_yuv_planes_into, YuvPlaneMut};
use libjpeg_turbo_rs::{
    compress_12bit, decompress_raw, yuv_plane_height, yuv_plane_width, DecodeWarning, Decoder,
    Encoder, JpegError, PixelFormat, RawComponentGeometry, RawGeometry, RawImage, RawStrip,
    RawStripReader, RestartResyncStrategy, ResyncAction, ScalingFactor, Subsampling,
};

/// Smooth gradients plus a fine texture, so every block carries AC energy.
fn textured_rgb(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .flat_map(|index| {
            let x: usize = index % width;
            let y: usize = index / width;
            [
                ((x * 255 / width.max(2) + (x * y) % 7 * 9) % 256) as u8,
                ((y * 255 / height.max(2) + (x + y) % 5 * 11) % 256) as u8,
                (((x + 2 * y) * 3 + (x * 3) % 13 * 7) % 256) as u8,
            ]
        })
        .collect()
}

fn encode_with_cjpeg(cjpeg: &Path, width: usize, height: usize, args: &[&str]) -> Vec<u8> {
    let ppm: Vec<u8> = helpers::build_ppm(&textured_rgb(width, height), width, height);
    let mut all_args: Vec<&str> = vec!["-quality", "90"];
    all_args.extend_from_slice(args);
    let label: String =
        format!("raw_strips_{width}x{height}_{}", args.join("_")).replace([' ', ',', '/'], "-");
    helpers::encode_with_c_cjpeg(cjpeg, &ppm, &all_args, &label)
}

/// The raw oracle, or `None` (after a local SKIP) when no libjpeg development
/// install can build it. CI provisions one, so there a missing oracle fails.
fn raw_oracle() -> Option<PathBuf> {
    let oracle: Option<PathBuf> = raw_decode_c_oracle();
    if oracle.is_none() {
        assert!(
            !helpers::is_ci(),
            "CI provisions a libjpeg development install, so the raw oracle must be buildable"
        );
        eprintln!("SKIP: no libjpeg development install found; cannot build the raw oracle");
    }
    oracle
}

/// Every strip a reader yields, collected into tight per-component planes of
/// the active area, checking each strip's geometry on the way.
struct Collected {
    geometry: RawGeometry,
    /// Active area, `width * height`, per component.
    active: Vec<Vec<u8>>,
    /// Padded rows, `padded_width * strip_count * rows_per_strip`, per component.
    padded: Vec<Vec<u8>>,
    strips: usize,
    warnings: Vec<DecodeWarning>,
}

fn collect(mut reader: RawStripReader<'_>) -> Collected {
    let geometry: RawGeometry = reader.geometry().clone();
    let components: usize = geometry.components.len();
    let mut active: Vec<Vec<u8>> = vec![Vec::new(); components];
    let mut padded: Vec<Vec<u8>> = vec![Vec::new(); components];
    let mut strips: usize = 0;
    while let Some(strip) = reader.next_strip().expect("strip decodes") {
        let strip: RawStrip<'_> = strip;
        assert_eq!(strip.index(), strips, "strips arrive in order");
        assert_eq!(strip.planes().len(), components);
        for (ci, plane) in strip.planes().enumerate() {
            let comp: &RawComponentGeometry = &geometry.components[ci];
            assert_eq!(plane.component(), ci);
            assert_eq!(plane.first_row(), strips * comp.rows_per_strip);
            assert_eq!(plane.width(), comp.width);
            assert_eq!(plane.padded_width(), comp.padded_width);
            let expected_rows: usize = comp.rows_per_strip.min(comp.height - plane.first_row());
            assert_eq!(
                plane.rows().len(),
                expected_rows,
                "component {ci} strip {strips}"
            );
            for row in plane.rows() {
                assert_eq!(row.len(), comp.width);
                active[ci].extend_from_slice(row);
            }
            assert_eq!(plane.padded_rows().len(), comp.rows_per_strip);
            for row in plane.padded_rows() {
                assert_eq!(row.len(), comp.padded_width);
                padded[ci].extend_from_slice(row);
            }
        }
        strips += 1;
    }
    assert_eq!(strips, geometry.strip_count);
    // A drained reader stays drained.
    assert!(reader.next_strip().expect("drained reader").is_none());
    let warnings: Vec<DecodeWarning> = reader.finish().expect("finish");
    Collected {
        geometry,
        active,
        padded,
        strips,
        warnings,
    }
}

fn assert_matches_libjpeg(collected: &Collected, c_planes: &[CRawPlane], label: &str) {
    assert_eq!(
        collected.active.len(),
        c_planes.len(),
        "[{label}] component count"
    );
    for (ci, (rust, c)) in collected.active.iter().zip(c_planes).enumerate() {
        let comp: &RawComponentGeometry = &collected.geometry.components[ci];
        assert_eq!(
            (comp.width, comp.height),
            (c.width, c.height),
            "[{label}] component {ci} active area"
        );
        let mismatches: usize = rust.iter().zip(&c.samples).filter(|(a, b)| a != b).count();
        assert_eq!(rust.len(), c.samples.len(), "[{label}] component {ci} size");
        assert_eq!(
            mismatches, 0,
            "[{label}] component {ci}: {mismatches} samples differ"
        );
    }
}

fn assert_matches_decode_raw(collected: &Collected, raw: &RawImage, label: &str) {
    for ci in 0..raw.num_components {
        assert_eq!(
            collected.geometry.components[ci].padded_width,
            raw.plane_widths[ci]
        );
        assert_eq!(
            collected.padded[ci].len(),
            raw.plane_widths[ci] * raw.plane_heights[ci],
            "[{label}] component {ci} padded size"
        );
        assert!(
            collected.padded[ci] == raw.planes[ci],
            "[{label}] component {ci} padded samples differ from decode_raw"
        );
    }
}

/// The layouts and coding options the strip reader must agree with libjpeg
/// on: every standard subsampling, an unusual 3:1 ratio, restart intervals in
/// rows and in MCUs, optimized tables, grayscale (a non-interleaved scan),
/// and the progressive and arithmetic-coded streams that the reader serves
/// from a whole-frame decode.
const CASES: &[&[&str]] = &[
    &["-sample", "1x1"],
    &["-sample", "2x1"],
    &["-sample", "2x2"],
    &["-sample", "1x2"],
    &["-sample", "4x1"],
    &["-sample", "4x2"],
    &["-sample", "3x1"],
    &["-sample", "2x2,2x1,1x1"],
    &["-sample", "2x2", "-restart", "1"],
    &["-sample", "2x1", "-restart", "3B"],
    &["-sample", "2x2", "-optimize"],
    &["-grayscale"],
    &["-grayscale", "-restart", "2B"],
    // A single component sampled 2x2: still one non-interleaved scan, but
    // its iMCU row is two block rows, and the MCU grid pads past the last
    // coded block row and column.
    &["-grayscale", "-sample", "2x2"],
    &["-sample", "2x2", "-progressive"],
    &["-grayscale", "-progressive"],
    &["-sample", "2x2", "-arithmetic"],
    &["-grayscale", "-arithmetic", "-restart", "2B"],
];

/// Odd sizes put a partial MCU at the right and bottom edges; 1x1 is a single
/// padded MCU; 16x16 and 64x48 are MCU-aligned for most samplings (not 3x1,
/// nor 4x1 / 4x2 at 16x16).
const SIZES: &[(usize, usize)] = &[(101, 101), (61, 45), (17, 33), (1, 1), (16, 16), (64, 48)];

#[test]
fn strips_match_libjpeg_raw_output() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let Some(oracle) = raw_oracle() else {
        return;
    };
    for args in CASES {
        for &(width, height) in SIZES {
            let label: String = format!("{width}x{height} {}", args.join(" "));
            let jpeg: Vec<u8> = encode_with_cjpeg(&cjpeg, width, height, args);
            let c_planes: Vec<CRawPlane> = decode_with_raw_c_oracle(&oracle, &jpeg, false)
                .unwrap_or_else(|stderr| panic!("[{label}] libjpeg refused: {stderr}"));
            let reader: RawStripReader<'_> = Decoder::new(&jpeg)
                .expect("header")
                .into_raw_strips()
                .unwrap_or_else(|e| panic!("[{label}] into_raw_strips: {e}"));
            let collected: Collected = collect(reader);
            assert!(
                collected.warnings.is_empty(),
                "[{label}] clean stream warned"
            );
            assert_matches_libjpeg(&collected, &c_planes, &label);
            let raw: RawImage = decompress_raw(&jpeg).expect("decode_raw");
            assert_matches_decode_raw(&collected, &raw, &label);
        }
    }
}

#[test]
fn geometry_follows_the_frame_for_an_odd_420_image() {
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(101, 101),
        101,
        101,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .expect("encode");
    let reader: RawStripReader<'_> = Decoder::new(&jpeg).unwrap().into_raw_strips().unwrap();
    let geometry: &RawGeometry = reader.geometry();
    assert_eq!((geometry.width, geometry.height), (101, 101));
    // ceil(101 / 16) iMCU rows, each 16 luma rows tall.
    assert_eq!(geometry.strip_count, 7);
    assert_eq!(geometry.strip_height, 16);
    let luma: &RawComponentGeometry = &geometry.components[0];
    assert_eq!((luma.horizontal_sampling, luma.vertical_sampling), (2, 2));
    assert_eq!((luma.width, luma.height), (101, 101));
    assert_eq!((luma.padded_width, luma.rows_per_strip), (112, 16));
    for chroma in &geometry.components[1..] {
        assert_eq!(
            (chroma.horizontal_sampling, chroma.vertical_sampling),
            (1, 1)
        );
        // libjpeg's downsampled_width: ceil(101 * 1 / 2).
        assert_eq!((chroma.width, chroma.height), (51, 51));
        assert_eq!((chroma.padded_width, chroma.rows_per_strip), (56, 8));
    }

    let collected: Collected = collect(reader);
    assert_eq!(collected.strips, 7);
}

/// Offsets of `jpeg`'s RST markers.
fn rst_offsets(jpeg: &[u8]) -> Vec<usize> {
    jpeg.windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
        .map(|(i, _)| i)
        .collect()
}

/// `jpeg` with four entropy bytes, `skip` bytes after its `nth` (0-based)
/// RST marker, overwritten by stuffed `FF 00` pairs: sixteen 1-bits, which
/// no Huffman code can be (the all-ones code is reserved), so every decoder
/// must notice. At `skip` 0 the interval's first DC symbol fails before any
/// coefficient is written; further in, a symbol fails mid-block.
fn corrupt_after_rst(mut jpeg: Vec<u8>, nth: usize, skip: usize) -> Vec<u8> {
    let rst: usize = *rst_offsets(&jpeg)
        .get(nth)
        .expect("fixture needs restart markers");
    let start: usize = rst + 2 + skip;
    // A preceding FF would turn the first written FF into its stuffing.
    assert!(
        skip == 0 || jpeg[start - 1] != 0xFF,
        "fixture offset splits a stuffed byte"
    );
    jpeg[start..start + 4].copy_from_slice(&[0xFF, 0x00, 0xFF, 0x00]);
    jpeg
}

fn encode_420_96(restart: impl FnOnce(Encoder<'_>) -> Encoder<'_>, pixels: &[u8]) -> Vec<u8> {
    restart(
        Encoder::new(pixels, 96, 96, PixelFormat::Rgb)
            .quality(85)
            .subsampling(Subsampling::S420),
    )
    .encode()
    .expect("encode")
}

/// A 96x96 4:2:0 stream with one restart interval per iMCU row, so every
/// RST sits on a strip boundary.
fn restart_rows_stream() -> Vec<u8> {
    encode_420_96(|e| e.restart_rows(1), &textured_rgb(96, 96))
}

/// [`restart_rows_stream`] with entropy bytes overwritten in its third
/// restart interval (the third strip).
fn corrupt_restart_stream() -> Vec<u8> {
    corrupt_after_rst(restart_rows_stream(), 1, 0)
}

/// A 96x96 4:2:0 stream (six MCUs per row) restarting every four MCUs, with
/// entropy bytes overwritten after the RST that precedes MCU 8: the third
/// MCU of the second iMCU row.
fn corrupt_mid_row_restart_stream() -> Vec<u8> {
    corrupt_after_rst(
        encode_420_96(|e| e.restart_blocks(4), &textured_rgb(96, 96)),
        1,
        0,
    )
}

/// 64x64 grayscale (a single-component block raster) restarting every four
/// blocks, half a block row.
fn grayscale_restart_stream() -> Vec<u8> {
    let gray: Vec<u8> = textured_rgb(64, 64).iter().step_by(3).copied().collect();
    Encoder::new(&gray, 64, 64, PixelFormat::Grayscale)
        .quality(90)
        .restart_blocks(4)
        .encode()
        .expect("encode")
}

/// [`grayscale_restart_stream`] with entropy bytes overwritten after its
/// second RST.
fn corrupt_grayscale_restart_stream() -> Vec<u8> {
    corrupt_after_rst(grayscale_restart_stream(), 1, 0)
}

/// `jpeg` cut inside its entropy data, three fifths of the way in.
fn truncated(mut jpeg: Vec<u8>) -> Vec<u8> {
    let sos: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS");
    let cut: usize = jpeg.len() * 3 / 5;
    assert!(cut > sos + 16, "the cut must fall in the entropy data");
    // A trailing FF would leave half a stuffed byte or marker: corruption,
    // not a clean truncation.
    assert_ne!(
        jpeg[cut - 1],
        0xFF,
        "the cut must fall between entropy bytes"
    );
    jpeg.truncate(cut);
    jpeg
}

/// [`restart_rows_stream`] with the RST at its second strip boundary
/// renumbered (RST1 becomes RST5), so the decoder meets the wrong marker.
fn renumbered_rst_stream() -> Vec<u8> {
    let mut jpeg: Vec<u8> = restart_rows_stream();
    let rst: usize = rst_offsets(&jpeg)[1];
    assert_eq!(jpeg[rst + 1], 0xD1);
    jpeg[rst + 1] = 0xD5;
    jpeg
}

/// [`restart_rows_stream`] with the RST at its second strip boundary
/// removed, so the next interval starts with no marker before it.
fn dropped_rst_stream() -> Vec<u8> {
    let mut jpeg: Vec<u8> = restart_rows_stream();
    let rst: usize = rst_offsets(&jpeg)[1];
    jpeg.drain(rst..rst + 2);
    jpeg
}

#[test]
fn strict_reader_yields_the_strips_before_the_corruption_then_fails() {
    let jpeg: Vec<u8> = corrupt_restart_stream();
    let strict_error: String = match decompress_raw(&jpeg) {
        Err(e) => e.to_string(),
        Ok(_) => {
            // The corruption decoded as valid (if wrong) Huffman codes; this
            // fixture then has nothing to say about strict failure.
            panic!("fixture must make a strict decode fail");
        }
    };
    let mut reader: RawStripReader<'_> = Decoder::new(&jpeg).unwrap().into_raw_strips().unwrap();
    let mut yielded: usize = 0;
    let error: JpegError = loop {
        match reader.next_strip() {
            Ok(Some(_)) => yielded += 1,
            Ok(None) => panic!("strict reader reached the end of a corrupt stream"),
            Err(e) => break e,
        }
    };
    assert_eq!(error.to_string(), strict_error);
    // The corruption sits in the third restart interval (one iMCU row each).
    assert_eq!(yielded, 2, "the two clean strips are delivered first");
    // A failed reader keeps failing rather than resuming mid-stream.
    assert!(reader.next_strip().is_err());
}

#[test]
fn for_each_strip_stops_at_the_callers_error() {
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(64, 64),
        64,
        64,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let reader: RawStripReader<'_> = Decoder::new(&jpeg).unwrap().into_raw_strips().unwrap();
    assert_eq!(reader.geometry().strip_count, 4);
    let mut seen: usize = 0;
    let result: Result<Vec<DecodeWarning>, JpegError> = reader.for_each_strip(|strip| {
        seen += 1;
        if strip.index() == 1 {
            return Err(JpegError::Unsupported("caller stopped".into()));
        }
        Ok(())
    });
    assert!(matches!(result, Err(JpegError::Unsupported(ref m)) if m == "caller stopped"));
    assert_eq!(seen, 2);
}

#[test]
fn reader_refuses_requests_it_cannot_honor_before_decoding() {
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(64, 64),
        64,
        64,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();

    let mut cropped: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    cropped.set_crop_region(0, 16, 64, 16);
    assert!(matches!(
        cropped.into_raw_strips(),
        Err(JpegError::Unsupported(_))
    ));

    let mut scaled: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    scaled.set_scale(ScalingFactor::try_new(1, 2).unwrap());
    assert!(matches!(
        scaled.into_raw_strips(),
        Err(JpegError::Unsupported(_))
    ));

    let pixels: Vec<i16> = (0..32 * 32 * 3).map(|i| (i % 4096) as i16).collect();
    let jpeg12: Vec<u8> = compress_12bit(&pixels, 32, 32, 3, 90, Subsampling::S420).unwrap();
    assert!(matches!(
        Decoder::new(&jpeg12).unwrap().into_raw_strips(),
        Err(JpegError::Unsupported(_))
    ));
}

#[test]
fn reader_applies_the_decoders_limits() {
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(64, 64),
        64,
        64,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let mut decoder: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(64 * 63);
    assert!(matches!(
        decoder.into_raw_strips(),
        Err(JpegError::LimitExceeded { .. })
    ));
}

// ---------------------------------------------------------------------------
// decompress_to_yuv_planes_into: the strips collected into caller planes
// ---------------------------------------------------------------------------

const CANARY: u8 = 0xA5;
const STRIDE_SLACK: usize = 7;

fn yuv_cases(cjpeg: &Path) -> Vec<(String, Vec<u8>)> {
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    for sampling in ["1x1", "2x1", "2x2", "1x2", "4x1"] {
        for &(width, height) in &[(101, 101), (61, 45), (1, 1), (64, 48)] {
            cases.push((
                format!("{width}x{height} {sampling}"),
                encode_with_cjpeg(cjpeg, width, height, &["-sample", sampling]),
            ));
        }
    }
    cases.push((
        "61x45 grayscale".into(),
        encode_with_cjpeg(cjpeg, 61, 45, &["-grayscale"]),
    ));
    cases.push((
        "61x45 2x2 progressive".into(),
        encode_with_cjpeg(cjpeg, 61, 45, &["-sample", "2x2", "-progressive"]),
    ));
    cases
}

#[test]
fn yuv_planes_into_matches_the_owned_planes_and_libjpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    // Without the oracle the owned-plane equality and the stride canary are
    // still checked; only the libjpeg comparison is skipped.
    let oracle: Option<PathBuf> = raw_oracle();
    for (label, jpeg) in yuv_cases(&cjpeg) {
        let (owned, width, height, subsampling): (Vec<Vec<u8>>, usize, usize, Subsampling) =
            decompress_to_yuv_planes(&jpeg).expect("owned planes");
        let strides: Vec<usize> = (0..owned.len())
            .map(|ci| yuv_plane_width(ci, width, subsampling) + STRIDE_SLACK)
            .collect();
        let mut buffers: Vec<Vec<u8>> = (0..owned.len())
            .map(|ci| vec![CANARY; strides[ci] * yuv_plane_height(ci, height, subsampling)])
            .collect();
        let mut planes: Vec<YuvPlaneMut<'_>> = buffers
            .iter_mut()
            .zip(&strides)
            .map(|(data, &stride)| YuvPlaneMut::new(data, stride))
            .collect();
        let info: (usize, usize, Subsampling) = decompress_to_yuv_planes_into(&jpeg, &mut planes)
            .unwrap_or_else(|e| panic!("[{label}] into: {e}"));
        assert_eq!(info, (width, height, subsampling), "[{label}]");

        let c_planes: Option<Vec<CRawPlane>> = oracle.as_ref().map(|oracle| {
            decode_with_raw_c_oracle(oracle, &jpeg, false)
                .unwrap_or_else(|stderr| panic!("[{label}] libjpeg refused: {stderr}"))
        });
        for ci in 0..owned.len() {
            let plane_width: usize = yuv_plane_width(ci, width, subsampling);
            let plane_height: usize = yuv_plane_height(ci, height, subsampling);
            for row in 0..plane_height {
                let line: &[u8] = &buffers[ci][row * strides[ci]..(row + 1) * strides[ci]];
                assert_eq!(
                    &line[..plane_width],
                    &owned[ci][row * plane_width..(row + 1) * plane_width],
                    "[{label}] plane {ci} row {row} differs from decompress_to_yuv_planes"
                );
                assert!(
                    line[plane_width..].iter().all(|&b| b == CANARY),
                    "[{label}] plane {ci} row {row}: stride padding was written"
                );
            }
            let Some(c_planes) = &c_planes else {
                continue;
            };
            let c: &CRawPlane = &c_planes[ci];
            for row in 0..c.height {
                assert_eq!(
                    &buffers[ci][row * strides[ci]..row * strides[ci] + c.width],
                    &c.samples[row * c.width..(row + 1) * c.width],
                    "[{label}] plane {ci} row {row} differs from libjpeg's raw output"
                );
            }
        }
    }
}

#[test]
fn yuv_planes_into_checks_every_destination_before_decoding() {
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(61, 45),
        61,
        45,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let widths: [usize; 3] = [62, 31, 31];
    let heights: [usize; 3] = [46, 23, 23];

    // Last plane one byte short: nothing may be written to any plane.
    let mut buffers: Vec<Vec<u8>> = (0..3)
        .map(|ci| vec![CANARY; widths[ci] * heights[ci] - usize::from(ci == 2)])
        .collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .enumerate()
        .map(|(ci, data)| YuvPlaneMut::new(data, widths[ci]))
        .collect();
    assert!(matches!(
        decompress_to_yuv_planes_into(&jpeg, &mut planes),
        Err(JpegError::BufferTooSmall { .. })
    ));
    assert!(buffers.iter().flatten().all(|&b| b == CANARY));

    // A stride narrower than its plane.
    let mut buffers: Vec<Vec<u8>> = (0..3).map(|ci| vec![0; widths[ci] * heights[ci]]).collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .enumerate()
        .map(|(ci, data)| YuvPlaneMut::new(data, widths[ci] - usize::from(ci == 1)))
        .collect();
    assert!(matches!(
        decompress_to_yuv_planes_into(&jpeg, &mut planes),
        Err(JpegError::Unsupported(_))
    ));
    assert!(buffers.iter().flatten().all(|&b| b == 0));

    // Wrong plane count.
    let mut buffers: Vec<Vec<u8>> = (0..2).map(|ci| vec![0; widths[ci] * heights[ci]]).collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .enumerate()
        .map(|(ci, data)| YuvPlaneMut::new(data, widths[ci]))
        .collect();
    assert!(matches!(
        decompress_to_yuv_planes_into(&jpeg, &mut planes),
        Err(JpegError::Unsupported(_))
    ));
    assert!(buffers.iter().flatten().all(|&b| b == 0));
}

#[test]
fn yuv_planes_into_accepts_packed_stride_zero_and_four_component_frames() {
    // Stride 0 is the TurboJPEG "tightly packed" convention.
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(61, 45),
        61,
        45,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let (owned, _, _, _): (Vec<Vec<u8>>, usize, usize, Subsampling) =
        decompress_to_yuv_planes(&jpeg).unwrap();
    let mut buffers: Vec<Vec<u8>> = owned.iter().map(|p| vec![0; p.len()]).collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .map(|data| YuvPlaneMut::new(data, 0))
        .collect();
    decompress_to_yuv_planes_into(&jpeg, &mut planes).expect("packed planes");
    assert_eq!(buffers, owned);

    // CMYK: four planes, the fourth (K) sized like luma.
    let cmyk: Vec<u8> = textured_rgb(37, 29)
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], p[0] ^ p[2]])
        .collect();
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &cmyk,
        37,
        29,
        libjpeg_turbo_rs::PixelFormat::Cmyk,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let (owned, _, _, _): (Vec<Vec<u8>>, usize, usize, Subsampling) =
        decompress_to_yuv_planes(&jpeg).unwrap();
    assert_eq!(owned.len(), 4);
    let mut buffers: Vec<Vec<u8>> = owned.iter().map(|p| vec![0; p.len()]).collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .map(|data| YuvPlaneMut::new(data, 0))
        .collect();
    decompress_to_yuv_planes_into(&jpeg, &mut planes).expect("CMYK planes");
    assert_eq!(buffers, owned);
}

/// A 4:2:0 baseline frame whose SOF height is patched to 0: a DNL-defined
/// height, which libjpeg refuses (`JERR_EMPTY_IMAGE`) and which divided by
/// zero in `decompress_to_yuv_planes`' subsampling detection.
fn zero_height_stream() -> Vec<u8> {
    let mut jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(16, 16),
        16,
        16,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let sof: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("SOF0");
    // FFC0, length (2), precision (1), height (2).
    jpeg[sof + 5] = 0;
    jpeg[sof + 6] = 0;
    jpeg
}

#[test]
fn a_zero_height_frame_is_refused_like_libjpeg_not_a_panic() {
    let jpeg: Vec<u8> = zero_height_stream();
    assert!(matches!(
        Decoder::new(&jpeg).and_then(|d| d.into_raw_strips()),
        Err(JpegError::CorruptData(_))
    ));
    assert!(decompress_to_yuv_planes(&jpeg).is_err());
    let mut buffers: Vec<Vec<u8>> = vec![vec![0; 16]; 3];
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .map(|data| YuvPlaneMut::new(data, 16))
        .collect();
    assert!(decompress_to_yuv_planes_into(&jpeg, &mut planes).is_err());

    // The C reference: libjpeg refuses the same stream (JERR_EMPTY_IMAGE).
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let dir: tempfile::TempDir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("zero_height.jpg");
    std::fs::write(&path, &jpeg).unwrap();
    let c: std::process::Output = std::process::Command::new(&djpeg)
        .arg(&path)
        .output()
        .unwrap();
    assert!(!c.status.success(), "libjpeg refuses a zero-height frame");
}

#[test]
fn first_error_keeps_its_variant_and_later_calls_still_fail() {
    let mut jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(64, 64),
        64,
        64,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    // Cut the entropy data in the third iMCU row.
    jpeg.truncate(jpeg.len() * 3 / 5);
    let mut reader: RawStripReader<'_> = Decoder::new(&jpeg).unwrap().into_raw_strips().unwrap();
    let first: JpegError = loop {
        match reader.next_strip() {
            Ok(Some(_)) => {}
            Ok(None) => panic!("a truncated stream decoded to the end"),
            Err(e) => break e,
        }
    };
    let raw_error: JpegError = decompress_raw(&jpeg).expect_err("decode_raw fails too");
    assert_eq!(first.to_string(), raw_error.to_string());
    assert!(matches!(first, JpegError::UnexpectedEof), "got {first:?}");
    assert!(reader.next_strip().is_err());
    assert!(reader.finish().is_err());
}

#[test]
fn finish_before_any_strip_drains_and_reports_the_same_warnings() {
    let jpeg: Vec<u8> = corrupt_restart_stream();
    let mut decoder: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    decoder.set_lenient(true);
    let (_, expected): (RawImage, Vec<DecodeWarning>) = decoder.decode_raw_with_warnings().unwrap();
    let mut decoder: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    decoder.set_lenient(true);
    let warnings: Vec<DecodeWarning> = decoder.into_raw_strips().unwrap().finish().unwrap();
    assert_eq!(warnings, expected);
}

/// What a recovery fixture must do under `decode_raw`, so a fixture whose
/// damage goes unnoticed fails instead of comparing two clean decodes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Damage {
    /// Strict decodes fail; lenient ones succeed and warn.
    Corrupt,
    /// Strict decodes fail with `UnexpectedEof`; lenient ones succeed and
    /// report `TruncatedData`.
    Truncated,
    /// The RST markers are damaged, not the entropy data: the outcome depends
    /// on how a restart is handled, so only agreement is required.
    Restart,
}

/// Padded planes and the warnings of the decode that produced them.
type PlanesAndWarnings = (Vec<Vec<u8>>, Vec<DecodeWarning>);
/// The `(expected, found)` pairs a [`RecordingStrategy`] was asked.
type DesyncCalls = Arc<Mutex<Vec<(u8, Option<u8>)>>>;

/// The strips of `decoder`, concatenated per component, and its warnings,
/// or the first error.
fn drain_padded(decoder: Decoder<'_>) -> Result<PlanesAndWarnings, JpegError> {
    let mut reader: RawStripReader<'_> = decoder.into_raw_strips()?;
    let mut padded: Vec<Vec<u8>> = vec![Vec::new(); reader.geometry().components.len()];
    while let Some(strip) = reader.next_strip()? {
        for plane in strip.planes() {
            padded[plane.component()].extend_from_slice(plane.padded_data());
        }
    }
    Ok((padded, reader.finish()?))
}

/// A resync strategy that answers every desync with `action` and records
/// what it was asked, so two decodes can be compared call for call.
struct RecordingStrategy {
    action: ResyncAction,
    calls: DesyncCalls,
}

impl RestartResyncStrategy for RecordingStrategy {
    fn on_desync(&mut self, expected: u8, found: Option<u8>) -> ResyncAction {
        self.calls.lock().unwrap().push((expected, found));
        self.action
    }
}

/// Every recovery path the strip loops copy from `decode_raw` (concealment,
/// the gray tail of a short segment, restart resets and the resync strategy,
/// with the reader's state carried across strips) must give `decode_raw`'s
/// planes, warnings or error, and ask the strategy the same questions.
#[test]
fn corrupt_streams_match_decode_raw_under_every_recovery_setting() {
    let streams: [(&str, Vec<u8>, Damage); 9] = [
        (
            "4:2:0 restart rows",
            corrupt_restart_stream(),
            Damage::Corrupt,
        ),
        (
            "4:2:0 restart mid-row",
            corrupt_mid_row_restart_stream(),
            Damage::Corrupt,
        ),
        (
            "grayscale restart 4 blocks",
            corrupt_grayscale_restart_stream(),
            Damage::Corrupt,
        ),
        // Eight bytes into the interval a symbol fails mid-block, after
        // coefficients were written, which concealment must discard.
        (
            "4:2:0 mid-block",
            corrupt_after_rst(restart_rows_stream(), 1, 8),
            Damage::Corrupt,
        ),
        (
            "grayscale mid-block",
            corrupt_after_rst(grayscale_restart_stream(), 1, 8),
            Damage::Corrupt,
        ),
        (
            "4:2:0 truncated",
            truncated(restart_rows_stream()),
            Damage::Truncated,
        ),
        (
            "grayscale truncated",
            truncated(grayscale_restart_stream()),
            Damage::Truncated,
        ),
        (
            "4:2:0 renumbered RST",
            renumbered_rst_stream(),
            Damage::Restart,
        ),
        ("4:2:0 dropped RST", dropped_rst_stream(), Damage::Restart),
    ];
    for (label, jpeg, damage) in &streams {
        let is_single_component: bool = Decoder::new(jpeg).unwrap().header().components.len() == 1;
        for lenient in [false, true] {
            for resync in [None, Some(ResyncAction::Continue), Some(ResyncAction::Skip)] {
                let case: String = format!("{label} lenient={lenient} resync={resync:?}");
                // A lenient single-component decode still consults the
                // strategy (lenient interleaved ones do not, P4-270), and a
                // `Skip` that finds no marker left to skip to fails it.
                let is_lenient_skip_failure = |e: &JpegError| -> bool {
                    lenient
                        && resync == Some(ResyncAction::Skip)
                        && is_single_component
                        && matches!(e, JpegError::CorruptData(msg)
                            if msg.starts_with("no further RST marker"))
                };
                let decoder = |calls: &DesyncCalls| {
                    let mut decoder: Decoder<'_> = Decoder::new(jpeg).unwrap();
                    decoder.set_lenient(lenient);
                    if let Some(action) = resync {
                        decoder.set_resync_strategy(RecordingStrategy {
                            action,
                            calls: Arc::clone(calls),
                        });
                    }
                    decoder
                };
                let raw_calls: DesyncCalls = Arc::default();
                let expected: Result<(RawImage, Vec<DecodeWarning>), JpegError> =
                    decoder(&raw_calls).decode_raw_with_warnings();
                let strip_calls: DesyncCalls = Arc::default();
                let outcome: Result<PlanesAndWarnings, JpegError> =
                    drain_padded(decoder(&strip_calls));
                match damage {
                    // A lenient decode recovers and warns.
                    Damage::Corrupt => match &expected {
                        Ok((_, warnings)) => assert!(
                            lenient && !warnings.is_empty(),
                            "[{case}] the damage went unnoticed"
                        ),
                        Err(e) => assert!(!lenient, "[{case}] lenient decode failed: {e}"),
                    },
                    Damage::Truncated => match &expected {
                        Ok((_, warnings)) => assert!(
                            lenient
                                && warnings
                                    .iter()
                                    .any(|w| matches!(w, DecodeWarning::TruncatedData { .. })),
                            "[{case}] the truncation went unnoticed: {warnings:?}"
                        ),
                        Err(e) => assert!(
                            if lenient {
                                is_lenient_skip_failure(e)
                            } else {
                                matches!(e, JpegError::UnexpectedEof)
                            },
                            "[{case}] {e:?}"
                        ),
                    },
                    // Strict decodes consult the strategy (lenient interleaved
                    // ones, as both 4:2:0 fixtures are, do not, P4-270), and it
                    // must have met the damaged marker.
                    Damage::Restart => assert_eq!(
                        resync.is_some() && !lenient,
                        !raw_calls.lock().unwrap().is_empty(),
                        "[{case}] strategy calls {:?}",
                        raw_calls.lock().unwrap()
                    ),
                }
                assert_eq!(
                    *strip_calls.lock().unwrap(),
                    *raw_calls.lock().unwrap(),
                    "[{case}] strategy calls"
                );
                match (expected, outcome) {
                    (Ok((raw, raw_warnings)), Ok((padded, warnings))) => {
                        assert_eq!(padded, raw.planes, "[{case}] planes");
                        assert_eq!(warnings, raw_warnings, "[{case}] warnings");
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "[{case}]"),
                    (a, b) => panic!(
                        "[{case}] decode_raw {:?} but strips {:?}",
                        a.map(|_| ()),
                        b.map(|_| ())
                    ),
                }
            }
        }
    }
}

#[test]
fn decoder_yuv_planes_into_applies_the_decoders_settings() {
    // Limits: refused before any destination is touched.
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(64, 64),
        64,
        64,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let mut buffers: Vec<Vec<u8>> = vec![
        vec![CANARY; 64 * 64],
        vec![CANARY; 32 * 32],
        vec![CANARY; 32 * 32],
    ];
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .zip([64, 32, 32])
        .map(|(data, stride)| YuvPlaneMut::new(data, stride))
        .collect();
    let mut limited: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    limited.set_max_pixels(64 * 63);
    assert!(matches!(
        limited.decode_yuv_planes_into(&mut planes),
        Err(JpegError::LimitExceeded { .. })
    ));
    assert!(buffers.iter().flatten().all(|&b| b == CANARY));

    // Leniency: a corrupt stream the strict form refuses decodes, with the
    // concealed samples `decode_raw` produces.
    let jpeg: Vec<u8> = corrupt_restart_stream();
    let widths: [usize; 3] = [96, 48, 48];
    let mut buffers: Vec<Vec<u8>> = widths.iter().map(|&w| vec![0; w * w]).collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .zip(widths)
        .map(|(data, stride)| YuvPlaneMut::new(data, stride))
        .collect();
    assert!(decompress_to_yuv_planes_into(&jpeg, &mut planes).is_err());
    // The warnings are `decode_raw_with_warnings`', and `stop_on_warning`
    // escalates none of them: like `decode_raw`, the YUV decode applies it to
    // the marker walk only (P4-273).
    let mut lenient: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    lenient.set_lenient(true);
    lenient.set_stop_on_warning(true);
    let (info, warnings): ((usize, usize, Subsampling), Vec<DecodeWarning>) = lenient
        .decode_yuv_planes_into_with_warnings(&mut planes)
        .unwrap();
    assert_eq!(info, (96, 96, Subsampling::S420));
    let mut decoder: Decoder<'_> = Decoder::new(&jpeg).unwrap();
    decoder.set_lenient(true);
    let (raw, raw_warnings): (RawImage, Vec<DecodeWarning>) =
        decoder.decode_raw_with_warnings().unwrap();
    assert!(!warnings.is_empty());
    assert_eq!(warnings, raw_warnings);
    for (ci, &width) in widths.iter().enumerate() {
        for row in 0..width {
            assert_eq!(
                &buffers[ci][row * width..(row + 1) * width],
                &raw.planes[ci][row * raw.plane_widths[ci]..row * raw.plane_widths[ci] + width],
                "plane {ci} row {row}"
            );
        }
    }
}

/// The whole-frame fallback (here a multi-scan baseline stream) must report
/// the warnings its decode recovered from, as `decode_raw_with_warnings`
/// does. A refactor once cleared them on every strip after the first.
#[test]
fn fallback_strips_keep_the_whole_frame_decodes_warnings() {
    // Embedded, not read at run time: the wasm32-wasip1 job runs this file under
    // wasmtime, which cannot open the absolute fixture path.
    let jpeg: &[u8] = include_bytes!("fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg");
    let probe: Decoder<'_> = Decoder::new(jpeg).unwrap();
    assert!(
        probe.has_multiple_scans(),
        "fixture must take the fallback path"
    );

    let mut decoder: Decoder<'_> = Decoder::new(jpeg).unwrap();
    decoder.set_lenient(true);
    let (raw, expected): (RawImage, Vec<DecodeWarning>) =
        decoder.decode_raw_with_warnings().unwrap();
    assert!(!expected.is_empty(), "fixture must be corrupt");

    let mut decoder: Decoder<'_> = Decoder::new(jpeg).unwrap();
    decoder.set_lenient(true);
    let collected: Collected = collect(decoder.into_raw_strips().unwrap());
    assert!(collected.strips > 1, "the bug needed more than one strip");
    assert_matches_decode_raw(&collected, &raw, "multi-scan lenient");
    assert_eq!(collected.warnings, expected);
}

/// A strict decode error leaves the rows of the strips decoded before it in
/// the caller's planes and nothing after them, as
/// `decompress_to_yuv_planes_into` documents (P4-273), whether the stream is
/// corrupt or cut short.
#[test]
fn yuv_planes_into_keeps_the_strips_before_a_decode_error() {
    for (label, jpeg) in [
        ("corrupt", corrupt_restart_stream()),
        ("truncated", truncated(restart_rows_stream())),
    ] {
        // The rows a strict reader delivers before it fails, per component.
        let mut reader: RawStripReader<'_> =
            Decoder::new(&jpeg).unwrap().into_raw_strips().unwrap();
        let mut delivered: usize = 0;
        let mut delivered_rows: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 3];
        let error: JpegError = loop {
            match reader.next_strip() {
                Ok(Some(strip)) => {
                    delivered += 1;
                    for plane in strip.planes() {
                        delivered_rows[plane.component()].extend(plane.rows().map(<[u8]>::to_vec));
                    }
                }
                Ok(None) => panic!("[{label}] strict strips decoded a damaged stream"),
                Err(e) => break e,
            }
        };
        assert!(
            (1..reader.geometry().strip_count).contains(&delivered),
            "[{label}] the error must fall after the first strip, got {delivered}"
        );

        let widths: [usize; 3] = [96, 48, 48];
        let heights: [usize; 3] = [96, 48, 48];
        let mut buffers: Vec<Vec<u8>> = widths
            .iter()
            .zip(heights)
            .map(|(&w, h)| vec![CANARY; w * h])
            .collect();
        let mut planes: Vec<YuvPlaneMut<'_>> = buffers
            .iter_mut()
            .zip(widths)
            .map(|(data, stride)| YuvPlaneMut::new(data, stride))
            .collect();
        let yuv_error: JpegError = decompress_to_yuv_planes_into(&jpeg, &mut planes)
            .expect_err("strict YUV decode of a damaged stream");
        assert_eq!(yuv_error.to_string(), error.to_string(), "[{label}]");

        for (ci, (&width, height)) in widths.iter().zip(heights).enumerate() {
            let written_rows: &[Vec<u8>] = &delivered_rows[ci];
            for row in 0..height {
                let line: &[u8] = &buffers[ci][row * width..(row + 1) * width];
                if let Some(expected) = written_rows.get(row) {
                    assert_eq!(line, &expected[..], "[{label}] plane {ci} row {row}");
                } else {
                    assert!(
                        line.iter().all(|&b| b == CANARY),
                        "[{label}] plane {ci} row {row} written after the error"
                    );
                }
            }
        }
    }
}

/// A plane at a stride wider than its width needs `(height - 1) * stride +
/// width` bytes, the last row unpadded: that length is accepted and one byte
/// less is refused.
#[test]
fn yuv_planes_into_minimum_length_at_a_padded_stride() {
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &textured_rgb(61, 45),
        61,
        45,
        PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .unwrap();
    let (owned, width, height, subsampling): (Vec<Vec<u8>>, usize, usize, Subsampling) =
        decompress_to_yuv_planes(&jpeg).unwrap();
    let widths: Vec<usize> = (0..3)
        .map(|ci| yuv_plane_width(ci, width, subsampling))
        .collect();
    let heights: Vec<usize> = (0..3)
        .map(|ci| yuv_plane_height(ci, height, subsampling))
        .collect();
    let strides: Vec<usize> = widths.iter().map(|w| w + STRIDE_SLACK).collect();
    let minimum = |ci: usize| (heights[ci] - 1) * strides[ci] + widths[ci];

    for short in 0..3 {
        let mut buffers: Vec<Vec<u8>> = (0..3)
            .map(|ci| vec![CANARY; minimum(ci) - usize::from(ci == short)])
            .collect();
        let mut planes: Vec<YuvPlaneMut<'_>> = buffers
            .iter_mut()
            .zip(&strides)
            .map(|(data, &stride)| YuvPlaneMut::new(data, stride))
            .collect();
        assert!(
            matches!(
                decompress_to_yuv_planes_into(&jpeg, &mut planes),
                Err(JpegError::BufferTooSmall { .. })
            ),
            "plane {short} one byte short"
        );
        assert!(buffers.iter().flatten().all(|&b| b == CANARY));
    }

    let mut buffers: Vec<Vec<u8>> = (0..3).map(|ci| vec![CANARY; minimum(ci)]).collect();
    let mut planes: Vec<YuvPlaneMut<'_>> = buffers
        .iter_mut()
        .zip(&strides)
        .map(|(data, &stride)| YuvPlaneMut::new(data, stride))
        .collect();
    decompress_to_yuv_planes_into(&jpeg, &mut planes).expect("minimum-length planes");
    for ci in 0..3 {
        for row in 0..heights[ci] {
            let start: usize = row * strides[ci];
            assert_eq!(
                &buffers[ci][start..start + widths[ci]],
                &owned[ci][row * widths[ci]..(row + 1) * widths[ci]],
                "plane {ci} row {row}"
            );
        }
    }
}

/// Four-component frames (CMYK, full size and with two subsampled
/// components) against libjpeg's raw output, which `cjpeg` cannot encode from
/// a PPM, so the Rust encoder writes them.
#[test]
fn four_component_strips_match_libjpeg_raw_output() {
    // Without the oracle the `decode_raw` comparison still runs.
    let oracle: Option<PathBuf> = raw_oracle();
    for (label, factors) in [
        ("CMYK 1x1", vec![(1, 1), (1, 1), (1, 1), (1, 1)]),
        ("CMYK 2x2,1x1,1x1,2x2", vec![(2, 2), (1, 1), (1, 1), (2, 2)]),
    ] {
        for &(width, height) in &[(61usize, 45usize), (37, 29)] {
            let label: String = format!("{width}x{height} {label}");
            let cmyk: Vec<u8> = textured_rgb(width, height)
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], p[0] ^ p[2]])
                .collect();
            let jpeg: Vec<u8> = Encoder::new(&cmyk, width, height, PixelFormat::Cmyk)
                .quality(90)
                .sampling_factors(factors.clone())
                .encode()
                .unwrap_or_else(|e| panic!("[{label}] encode: {e}"));
            let collected: Collected =
                collect(Decoder::new(&jpeg).unwrap().into_raw_strips().unwrap());
            assert_eq!(collected.geometry.components.len(), 4, "[{label}]");
            assert!(
                collected.warnings.is_empty(),
                "[{label}] clean stream warned"
            );
            if let Some(oracle) = &oracle {
                let c_planes: Vec<CRawPlane> = decode_with_raw_c_oracle(oracle, &jpeg, false)
                    .unwrap_or_else(|stderr| panic!("[{label}] libjpeg refused: {stderr}"));
                assert_matches_libjpeg(&collected, &c_planes, &label);
            }
            let raw: RawImage = decompress_raw(&jpeg).expect("decode_raw");
            assert_matches_decode_raw(&collected, &raw, &label);
        }
    }
}
