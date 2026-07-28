//! Issue #357 / P4-58: bounded-input-memory incremental decode from a
//! `Read` source. The interleaved-baseline path must decode from a
//! sliding window without ever buffering the whole compressed stream;
//! progressive and non-interleaved streams fall back to buffering
//! (documented — their entropy layout requires the full stream).

mod helpers;

use libjpeg_turbo_rs::{decompress, decompress_from_reader_incremental, Image};
use std::io::Read;

const BASELINE_420: &[u8] = include_bytes!("fixtures/photo_640x480_420.jpg");
const BASELINE_444: &[u8] = include_bytes!("fixtures/photo_640x480_444.jpg");
const BASELINE_RST: &[u8] = include_bytes!("fixtures/photo_640x480_420_rst.jpg");
const PROGRESSIVE: &[u8] = include_bytes!("fixtures/graphic_640x480_420_prog.jpg");
const GRAY_BASELINE: &[u8] = include_bytes!("fixtures/gray_8x8.jpg");

/// A reader that yields at most `chunk` bytes per read call and counts
/// the total handed out — lets tests drive arbitrarily small feeds.
struct ChunkedReader<'a> {
    data: &'a [u8],
    pos: usize,
    chunk: usize,
}

impl<'a> ChunkedReader<'a> {
    fn new(data: &'a [u8], chunk: usize) -> Self {
        Self {
            data,
            pos: 0,
            chunk,
        }
    }
}

impl Read for ChunkedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n: usize = buf.len().min(self.chunk).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn assert_identical(a: &Image, b: &Image, label: &str) {
    assert_eq!(a.width, b.width, "{label}: width");
    assert_eq!(a.height, b.height, "{label}: height");
    assert_eq!(a.pixel_format, b.pixel_format, "{label}: format");
    assert_eq!(a.data, b.data, "{label}: pixels must be byte-identical");
}

/// The incremental path must match the slice path bit-for-bit across
/// chunk sizes that stress every window-boundary case. The slice path
/// is djpeg-pinned by the existing suites, so equality here is
/// transitive C cross-validation.
///
/// A short read makes the driver retry the starved row immediately
/// (it cannot block waiting for a source that may already have
/// delivered the whole image), so a 1-byte drip is retry-per-byte —
/// quadratic in row size by design. The drip cases therefore run on a
/// small interleaved fixture; the 640x480 trio uses realistic chunks.
#[test]
fn incremental_matches_slice_for_baseline_across_chunk_sizes() {
    let small: &[u8] = include_bytes!("fixtures/blue_16x16_420.jpg");
    let small_ref: Image = decompress(small).expect("slice decode small");
    for chunk in [1usize, 3, 7] {
        let streamed: Image = decompress_from_reader_incremental(ChunkedReader::new(small, chunk))
            .unwrap_or_else(|e| panic!("blue_16x16 chunk={chunk}: {e:?}"));
        assert_identical(&small_ref, &streamed, &format!("blue_16x16 chunk={chunk}"));
    }

    for (fixture, name) in [
        (BASELINE_420, "photo_640x480_420"),
        (BASELINE_444, "photo_640x480_444"),
        (BASELINE_RST, "photo_640x480_420_rst"),
    ] {
        let reference: Image = decompress(fixture).expect("slice decode");
        for chunk in [1024usize, 4096, 65536, usize::MAX] {
            let reader = ChunkedReader::new(fixture, chunk);
            let streamed: Image = decompress_from_reader_incremental(reader)
                .unwrap_or_else(|e| panic!("{name} chunk={chunk}: {e:?}"));
            assert_identical(&reference, &streamed, &format!("{name} chunk={chunk}"));
        }
    }
}

/// Progressive input requires the whole entropy stream by construction;
/// the incremental entry point must still decode it correctly (internal
/// buffering fallback), not error.
#[test]
fn incremental_falls_back_correctly_for_progressive() {
    let reference: Image = decompress(PROGRESSIVE).expect("slice decode");
    let streamed: Image = decompress_from_reader_incremental(ChunkedReader::new(PROGRESSIVE, 4096))
        .expect("progressive fallback decode");
    assert_identical(&reference, &streamed, "progressive fallback");

    // Tiny single-MCU baseline gray through a 3-byte drip feed.
    let gray_ref: Image = decompress(GRAY_BASELINE).expect("slice gray");
    let gray_streamed: Image =
        decompress_from_reader_incremental(ChunkedReader::new(GRAY_BASELINE, 3))
            .expect("gray decode");
    assert_identical(&gray_ref, &gray_streamed, "gray");
}

/// Truncated input must surface an error, not hang waiting for bytes
/// or silently return garbage.
#[test]
fn incremental_errors_on_truncated_input() {
    let truncated: &[u8] = &BASELINE_420[..BASELINE_420.len() / 2];
    let result = decompress_from_reader_incremental(ChunkedReader::new(truncated, 4096));
    assert!(result.is_err(), "truncated stream must error, got Ok");

    let result = decompress_from_reader_incremental(ChunkedReader::new(&[], 16));
    assert!(result.is_err(), "empty stream must error");
}

/// I/O errors from the reader must propagate as errors.
#[test]
fn incremental_propagates_reader_errors() {
    struct FailingReader {
        fed: usize,
    }
    impl Read for FailingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.fed >= 1024 {
                return Err(std::io::Error::other("simulated network drop"));
            }
            let n: usize = buf.len().min(1024 - self.fed);
            buf[..n].copy_from_slice(&BASELINE_420[self.fed..self.fed + n]);
            self.fed += n;
            Ok(n)
        }
    }
    let result = decompress_from_reader_incremental(FailingReader { fed: 0 });
    assert!(result.is_err(), "reader error must propagate");
}

/// The bounded-memory acceptance criterion (P4-58): the input window a
/// baseline decode holds must stay far below the compressed size. The
/// instrumented entry point reports the peak window; the cap asserted
/// here is the measured value + margin, not a guess — re-measure before
/// tightening.
#[test]
fn incremental_input_window_stays_bounded() {
    // 1.25 MB 1080p baseline: large enough that the fixed refill chunk
    // (64 KiB) is a small fraction of the stream. Measured peak on this
    // fixture: 195,985 bytes of allocation capacity (header + entropy
    // window + the 64 KiB read-staging buffer — the metric counts
    // capacities, not just live bytes). The assertion allows up to
    // 256 KiB — still 6x below the compressed size, independent of it.
    let fixture: &[u8] = include_bytes!("fixtures/photo_1920x1080_420.jpg");
    let compressed_len: usize = fixture.len();
    let (image, peak_window): (Image, usize) =
        libjpeg_turbo_rs::decompress_from_reader_incremental_instrumented(ChunkedReader::new(
            fixture, 8192,
        ))
        .expect("instrumented decode");
    let reference: Image = decompress(fixture).expect("slice decode");
    assert_identical(&reference, &image, "instrumented");

    assert!(
        peak_window <= 4 * 64 * 1024,
        "peak input window {peak_window} exceeded 256 KiB"
    );
    assert!(
        peak_window < compressed_len / 4,
        "peak input window {peak_window} must stay well below the \
         compressed size {compressed_len} — the whole point of P4-58"
    );
}
