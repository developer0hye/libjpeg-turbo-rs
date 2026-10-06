//! P4-141 criterion 1: Miri over the root crate's **public API**, not its unit
//! tests.
//!
//! The Miri job before this suite ran `cargo miri test --lib`, so everything it
//! interpreted was reached from inside the crate — a `#[cfg(test)]` module
//! calling a function with arguments the module chose. Neither of the two
//! surfaces below was untouched there, and the honest delta is narrower and
//! sharper than "uncovered" (the first draft of this comment claimed
//! `ProgressiveDecoder` was unreachable from a `--lib` test, which
//! `progressive_output::tests::progressive_output_path_is_miri_covered` refutes;
//! `docs-drift-auditor` measured it):
//!
//! * **Progressive output (P4-136).** `ProgressiveDecoder::output()` allocates
//!   one buffer per intermediate reconstruction and fills it from the
//!   coefficients decoded so far. Two things happen when
//!   [`progressive_intermediate_outputs_are_fully_initialised`] walks that path
//!   under the interpreter, and they are worth separating because one of them
//!   is live and the other is a tripwire:
//!
//!   1. **Live.** Reconstructing a scan runs `decode_progressive_planes`
//!      (`src/decode/pipeline_impl/progressive.rs:186`), which writes every
//!      block through `component_planes[c].as_mut_ptr().add(dst_offset)` into
//!      `idct_scaled_strided` — a raw destination pointer plus a stride, not
//!      feature-gated, and one of the sites `docs/UNSAFE_INVENTORY.md` owns.
//!      Miri checks each of those writes for bounds and provenance. The
//!      baseline decodes this suite performs reach the three equivalents in
//!      `pipeline_impl/baseline.rs`.
//!   2. **A tripwire.** *Reading* every byte does not, today, test
//!      initialisation: both destinations are initialised by construction —
//!      the colour path allocates with `try_filled_vec(.., 0u8, ..)` and the
//!      grayscale path reserves and `extend_from_slice`es — so Miri's uninit
//!      tracking cannot fire. It is a guard against the shape P4-136 removed:
//!      a `set_len` over spare capacity, whose bytes the lib test's
//!      `image.data.len()` assertion would pass over exactly as happily as
//!      over written memory. Stating that plainly rather than claiming the
//!      stronger thing is the point — the claim was the stronger one until
//!      `rust-code-reviewer` read the allocator (2026-09-09).
//!
//!   [`progressive_grayscale_intermediate_outputs_are_fully_initialised`]
//!   covers the single-component branch of `output()`, a separate allocation
//!   and fill with no lib test at all, and both tests assert the final
//!   reconstruction equals the one-shot `decompress`.
//! * **`BitWriter` (P4-138).** The writer treats a `Vec<u8>`'s *spare capacity*
//!   as an arena: the hot path writes through `as_mut_ptr().add(pos)` and
//!   `set_len(pos)` is a synchronisation point before every reallocation. Miri
//!   is the only tool in this repository that checks such a write is inside the
//!   allocation. The lib tests drive the writer directly with sizes they choose;
//!   what nothing established is that a *real encode* reaches the interesting
//!   states — a buffer that has grown at least once, and an entropy stream
//!   containing `0xFF` so the byte-stuffing slow path runs. Both are pinned by
//!   assertion here rather than hoped for: see
//!   [`baseline_encode_grows_the_bitwriter_and_takes_the_ff_stuffing_path`].
//!
//! And both, unlike the lib tests, are reached through the API a consumer
//! actually has.
//!
//! **What this suite does not do is prove byte parity with C.** It runs under
//! an interpreter that cannot spawn a process, so the C oracle has to live in a
//! test the Miri legs skip. `djpeg_agrees_with_every_fixture_this_suite_builds`
//! is that test: it decodes *every fixture this suite builds* with the pinned
//! `djpeg` and requires diff = 0, and it carries `#[cfg_attr(miri, ignore)]`
//! because `std::process::Command` is unsupported under Miri. The native
//! Integration Tests leg runs it on every pull request.
//!
//! Kept off SIMD deliberately: the Miri legs build `--no-default-features
//! --features std`, so every kernel these paths reach is the scalar one. Miri
//! cannot interpret vendor intrinsics, which is why the job excludes them; the
//! SIMD arms are covered by the sanitizer legs instead.
//!
//! No `#![cfg(not(target_arch = "wasm32"))]`, unlike the other three suites in
//! this landing: nothing here reads a file, spawns a process or starts a
//! thread, so `wasm.yml`'s `cargo test --target wasm32-wasip1` runs these four
//! tests under `wasmtime` (verified, 2026-09-09) and the C-oracle test below
//! carries its own `cfg`. That leg builds `panic = "abort"`, which is why the
//! reservation figures [`writer_reservation`] re-derives are pinned in an
//! ungated `#[cfg(test)]` module rather than beside the unwinding `BitWriter`
//! tests.

use libjpeg_turbo_rs::{
    compress, compress_progressive, decompress, Image, PixelFormat, ProgressiveDecoder, Subsampling,
};

mod helpers;

/// Side of the fixtures. Two MCU rows of 4:2:0 at 32×32, which is the smallest
/// size that still produces a multi-scan progressive stream *and* more than one
/// MCU row to reconstruct — while staying inside a Miri run measured in
/// seconds.
const SIDE: usize = 32;

/// Side of the fixture in [`progressive_encode_grows_the_bitwriter_across_a_reset`].
/// Larger than [`SIDE`] because the progressive writer is reset per scan: see
/// that test for the measurement that fixes this number.
const PROGRESSIVE_GROWTH_SIDE: usize = 48;

/// Deterministic high-entropy bytes.
///
/// A xorshift rather than a gradient: the `BitWriter` assertions below need an
/// entropy stream that overflows the writer's initial capacity and contains
/// `0xFF` bytes, and smooth content produces neither.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state: u32 = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as u8
        })
        .collect()
}

/// The same pixel pattern `tests/progressive_output.rs` builds its fixtures
/// from, so the progressive bytes this suite interprets under Miri are the
/// bytes that suite already proved equal to `djpeg`'s output, diff = 0
/// (`c_djpeg_progressive_intermediate_diff_zero`).
fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8);
            pixels.push(((x * 3 + y * 7 + 50) % 256) as u8);
            pixels.push(((x * 5 + y * 5 + 100) % 256) as u8);
        }
    }
    pixels
}

fn progressive_rgb_fixture() -> Vec<u8> {
    compress_progressive(
        &gradient_rgb(SIDE, SIDE),
        SIDE,
        SIDE,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("progressive RGB encode")
}

fn progressive_gray_fixture() -> Vec<u8> {
    let gray: Vec<u8> = (0..SIDE * SIDE)
        .map(|i| ((i * 11) % 256) as u8)
        .collect::<Vec<u8>>();
    compress_progressive(
        &gray,
        SIDE,
        SIDE,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .expect("progressive grayscale encode")
}

fn noisy_progressive_fixture() -> Vec<u8> {
    compress_progressive(
        &noise(
            PROGRESSIVE_GROWTH_SIDE * PROGRESSIVE_GROWTH_SIDE * 3,
            0x1234_5678,
        ),
        PROGRESSIVE_GROWTH_SIDE,
        PROGRESSIVE_GROWTH_SIDE,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("progressive encode of high-entropy pixels")
}

fn noisy_baseline_fixture() -> Vec<u8> {
    compress(
        &noise(SIDE * SIDE * 3, 0x9E37_79B9),
        SIDE,
        SIDE,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("baseline encode of high-entropy pixels")
}

/// Sum of every byte, forcing a read of the whole buffer.
///
/// The read is the tripwire described in the module documentation: it is what a
/// return to `set_len`-over-spare-capacity would trip under the interpreter.
/// Against today's zero-filled destination it cannot fail on its own, which is
/// why the callers assert on the *content* as well.
fn sum_every_byte(image: &Image) -> u64 {
    image.data.iter().map(|&b| u64::from(b)).sum()
}

/// Every row of a reconstruction must carry at least one non-zero byte.
///
/// `sum > 0` alone is satisfied by a single written byte, so a fill that
/// stopped after one row — or one component, or one MCU row — would pass it.
/// These fixtures are gradients and noise with no black row in them, so this is
/// content-agnostic and still catches a partial fill (rust-code-reviewer,
/// 2026-09-09).
fn every_row_was_written(image: &Image, bytes_per_pixel: usize, label: &str) {
    let stride: usize = image.width * bytes_per_pixel;
    for (row, bytes) in image.data.chunks(stride).enumerate() {
        assert!(
            bytes.iter().any(|&b| b != 0),
            "{label}: row {row} of {} is entirely zero — the fill stopped short",
            image.height
        );
    }
}

/// P4-136, from outside the crate: every intermediate reconstruction a caller
/// can observe is fully written before it is handed over.
#[test]
fn progressive_intermediate_outputs_are_fully_initialised() {
    let jpeg: Vec<u8> = progressive_rgb_fixture();
    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).expect("open progressive");
    let scans: usize = decoder.num_scans();
    assert!(
        scans > 1,
        "fixture must be multi-scan for this to test anything, got {scans}"
    );

    let expected_len: usize = SIDE * SIDE * 3;
    let mut observed: usize = 0;
    while decoder.consume_input().expect("consume scan") {
        let image: Image = decoder.output().expect("intermediate output");
        assert_eq!(image.width, SIDE);
        assert_eq!(image.height, SIDE);
        assert_eq!(
            image.data.len(),
            expected_len,
            "intermediate output {} is {} bytes, want {}",
            observed,
            image.data.len(),
            expected_len
        );
        // The read is the point; the sum only keeps the optimiser from
        // discarding it in the non-Miri legs.
        let sum: u64 = sum_every_byte(&image);
        assert!(
            sum > 0,
            "intermediate output {observed} decoded to all-zero"
        );
        every_row_was_written(&image, 3, &format!("intermediate output {observed}"));
        observed += 1;
    }

    assert_eq!(observed, scans, "every scan must produce an output");
    assert!(decoder.input_complete());

    // The last reconstruction is the whole image, and the one-shot decoder
    // agrees with it byte for byte — an assertion about the *values*, which no
    // amount of initialisation checking gives you.
    let final_image: Image = decoder.output().expect("final output");
    let reference: Image = decompress(&jpeg).expect("one-shot decode");
    assert_eq!(final_image.data, reference.data);
}

/// The grayscale branch of `output()` allocates its own single-component
/// buffer; it is a different `try_reserved_vec` call site from the colour path
/// above and shares none of its fill code.
#[test]
fn progressive_grayscale_intermediate_outputs_are_fully_initialised() {
    let jpeg: Vec<u8> = progressive_gray_fixture();
    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).expect("open progressive");
    let scans: usize = decoder.num_scans();
    assert!(
        scans > 1,
        "grayscale fixture must be multi-scan, got {scans}"
    );

    let mut observed: usize = 0;
    while decoder.consume_input().expect("consume scan") {
        let image: Image = decoder.output().expect("intermediate output");
        assert_eq!(image.data.len(), SIDE * SIDE);
        assert!(sum_every_byte(&image) > 0);
        every_row_was_written(&image, 1, &format!("grayscale intermediate {observed}"));
        observed += 1;
    }
    assert_eq!(observed, scans);

    let final_image: Image = decoder.output().expect("final output");
    let reference: Image = decompress(&jpeg).expect("one-shot decode");
    assert_eq!(final_image.data, reference.data);
}

/// Per-scan statistics of an encoded stream: the longest entropy-coded scan
/// payload in bytes, how many stuffed `0xFF 0x00` pairs it contains, and how
/// many scans there are.
///
/// The *longest scan* is the figure the `BitWriter` assertions need, not the
/// file size. The writer is reset — not reallocated — between the scans of a
/// progressive encode (`pipeline_impl/progressive.rs`), so what has to exceed
/// its reservation for `ensure_capacity` to grow the arena is one scan's
/// payload. Asserting on `jpeg.len()` instead would have passed on a
/// progressive fixture whose largest scan was 849 bytes against a 1024-byte
/// reservation: a comparison that cannot fail is what this repository keeps
/// finding at the bottom of a green harness.
fn scan_stats(jpeg: &[u8]) -> ScanStats {
    let mut stats: ScanStats = ScanStats::default();
    let mut i: usize = 2; // past SOI
    while i + 1 < jpeg.len() {
        if jpeg[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker: u8 = jpeg[i + 1];
        // A run of 0xFF fill bytes may precede any marker, so consume one
        // byte and re-read — `i += 2` would eat the marker *behind* the fill
        // byte and skip the segment it introduces (rust-code-reviewer,
        // 2026-09-09). Our encoder emits no fill bytes today; a walker that
        // silently loses a scan is how the growth assertion below would come
        // to compare the wrong number.
        if marker == 0xFF {
            i += 1;
            continue;
        }
        // Standalone markers carry no length field.
        if marker == 0xD8 || (0xD0..=0xD9).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        // `get`, not indexing: a truncated stream should end the walk, not panic
        // with an index message that says nothing about the truncation.
        let Some(length_bytes) = jpeg.get(i + 2..i + 4) else {
            break;
        };
        let header_len: usize = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
        if marker != 0xDA {
            i += 2 + header_len;
            continue;
        }

        // Entropy-coded segment: everything up to the next marker that is not
        // a stuffed zero or a restart.
        stats.scans += 1;
        let start: usize = i + 2 + header_len;
        let mut j: usize = start;
        while j + 1 < jpeg.len() {
            if jpeg[j] != 0xFF {
                j += 1;
                continue;
            }
            match jpeg[j + 1] {
                0x00 => {
                    stats.stuffed_pairs += 1;
                    j += 2;
                }
                0xD0..=0xD7 => j += 2,
                // Fill bytes again: the scan ends at the marker they precede,
                // not at the first of them.
                0xFF => j += 1,
                _ => break,
            }
        }
        stats.longest_scan = stats.longest_scan.max(j - start);
        i = j;
    }
    stats
}

#[derive(Debug, Default)]
struct ScanStats {
    longest_scan: usize,
    stuffed_pairs: usize,
    scans: usize,
}

/// What `BitWriter::new(capacity)` actually reserves: `2 * capacity`, floored at
/// 1024 bytes.
///
/// A copy of a private policy, and the inference below has *two* such copies —
/// this multiplier, and the capacity each encoder starts from. Both are pinned
/// on the other side, in `huffman_encode::tests`:
/// `new_reserves_double_the_request_floored_at_1024` asserts these three
/// figures, and `frame_and_progressive_reservations_are_what_the_miri_suite_models`
/// asserts what `BitWriter::for_frame` and `BitWriter::for_progressive_scan`
/// hand it — the named constructors the encoders now call, which exist so that
/// argument has one definition instead of five. Without both pins, raising
/// either number would leave this function computing the old, smaller one: the
/// growth inference would hold trivially and prove nothing
/// (rust-code-reviewer, 2026-09-09).
fn writer_reservation(requested: usize) -> usize {
    (requested * 2).max(1024)
}

/// P4-138, from outside the crate: the arena writes stay inside the allocation
/// across a reallocation, and the stuffing path writes inside it too.
///
/// The two `assert!`s are what make this a test of the writer rather than of
/// `compress`. The baseline encoder builds its writer with `width * height`, so
/// a single-scan payload longer than the resulting reservation cannot have been
/// produced without at least one `ensure_capacity` growth — the `set_len` +
/// `reserve` sequence P4-138 is about. And a stuffed pair cannot appear without
/// `emit_byte_unchecked`, the one-raw-`write`-per-byte slow path, having run.
/// Measured on this fixture: 4218 bytes of scan against a 2048-byte
/// reservation, 21 stuffed pairs.
#[test]
fn baseline_encode_grows_the_bitwriter_and_takes_the_ff_stuffing_path() {
    let jpeg: Vec<u8> = noisy_baseline_fixture();
    let stats: ScanStats = scan_stats(&jpeg);
    let reservation: usize = writer_reservation(SIDE * SIDE);

    assert_eq!(stats.scans, 1, "baseline fixture must be single-scan");
    assert!(
        stats.longest_scan > reservation,
        "scan payload is {} bytes and the writer reserved {} — it fits, so nothing proves a growth",
        stats.longest_scan,
        reservation
    );
    assert!(
        stats.stuffed_pairs > 0,
        "fixture contains no 0xFF00 pair, so the stuffing path never ran"
    );

    let image: Image = decompress(&jpeg).expect("decode what we just encoded");
    assert_eq!((image.width, image.height), (SIDE, SIDE));
    assert_eq!(image.data.len(), SIDE * SIDE * 3);
    assert!(sum_every_byte(&image) > 0);
}

/// The progressive encoder reuses one writer across every scan and sizes it at
/// `width * height / 4`, so growth here happens inside a `reset` cycle — the
/// state where `pos` is 0 but the buffer already holds a previous scan's bytes.
///
/// 48×48 rather than 32×32 on measurement: at 32×32 the longest scan is 849
/// bytes against a 1024-byte reservation, so no growth occurs and the test
/// would have proved nothing. At 48×48 it is 1911 against 1152.
#[test]
fn progressive_encode_grows_the_bitwriter_across_a_reset() {
    let side: usize = PROGRESSIVE_GROWTH_SIDE;
    let jpeg: Vec<u8> = noisy_progressive_fixture();

    let stats: ScanStats = scan_stats(&jpeg);
    let reservation: usize = writer_reservation(side * side / 4);
    assert!(stats.scans > 1, "progressive fixture must be multi-scan");
    // The growth assertion below rests on this walk finding every scan, so the
    // walk answers to the decoder rather than to itself: a mis-parse that lost
    // a scan would quietly shrink `longest_scan` (rust-code-reviewer,
    // 2026-09-09).
    let scans_seen_by_the_decoder: usize = ProgressiveDecoder::new(&jpeg)
        .expect("open the progressive fixture")
        .num_scans();
    assert_eq!(
        stats.scans, scans_seen_by_the_decoder,
        "the marker walk found {} scans and the decoder {} — the walk is wrong",
        stats.scans, scans_seen_by_the_decoder
    );
    assert!(
        stats.longest_scan > reservation,
        "longest scan is {} bytes and the writer reserved {} — no growth is proved",
        stats.longest_scan,
        reservation
    );

    let image: Image = decompress(&jpeg).expect("decode what we just encoded");
    assert_eq!((image.width, image.height), (side, side));
    assert_eq!(image.data.len(), side * side * 3);
}

/// The C oracle for all four fixtures above.
///
/// Ignored under Miri (no `std::process::Command`) and absent on wasm32 (no
/// process at all), which is exactly why it is a separate test rather than an
/// assertion inside the others: the initialisation and aliasing checks must run
/// where the interpreter is, and byte parity must run where `djpeg` is.
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(miri, ignore = "Miri cannot spawn a process")]
#[test]
fn djpeg_agrees_with_every_fixture_this_suite_builds() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");

    let colour: [(&str, Vec<u8>); 3] = [
        ("miri_progressive_rgb", progressive_rgb_fixture()),
        ("miri_noisy_baseline", noisy_baseline_fixture()),
        // The growth fixture too: a writer that reallocated mid-scan is exactly
        // the one whose bytes are worth comparing against C, and asserting only
        // that our own decoder round-trips it would compare this port with
        // itself.
        ("miri_noisy_progressive", noisy_progressive_fixture()),
    ];
    for (label, jpeg) in colour {
        let (width, height, c_pixels) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, label);
        let ours: Image = decompress(&jpeg).expect("decode fixture");
        assert_eq!((width, height), (ours.width, ours.height), "{label}: size");
        helpers::assert_pixels_identical(&c_pixels, &ours.data, width, height, 3, label);
    }

    let gray_jpeg: Vec<u8> = progressive_gray_fixture();
    let (width, height, c_gray) =
        helpers::decode_gray_with_c_djpeg(&djpeg, &gray_jpeg, "miri_progressive_gray");
    let ours: Image = decompress(&gray_jpeg).expect("decode grayscale fixture");
    assert_eq!((width, height), (ours.width, ours.height), "gray: size");
    helpers::assert_pixels_identical(
        &c_gray,
        &ours.data,
        width,
        height,
        1,
        "miri_progressive_gray",
    );
}
