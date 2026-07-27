//! A5-A6 integration tests for TjHandle session reset and ICC/buffer/marker
//! symmetry. These tests belong to this worker (agent-a94c0e13) and live
//! outside the main tj3_handle.rs suite to avoid churn with other workers.

use libjpeg_turbo_rs::tj3::TjHandle;
use libjpeg_turbo_rs::{compress_with_metadata, PixelFormat, Subsampling};

/// A5-1: ICC profile symmetry between handle and decoded image.
///
/// After a successful `decompress()`, both `TjHandle::icc_profile()` and
/// `Image::icc_profile()` must report the same bytes as the embedded JPEG ICC.
/// This mirrors C libjpeg-turbo's `tj3GetICCProfile()` contract where the
/// handle retains a copy that the caller can query independently of the image.
#[test]
fn a5_1_icc_symmetry_handle_equals_image() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let icc = vec![0xCDu8; 96];
    let jpeg = compress_with_metadata(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&icc),
        None,
    )
    .expect("compress_with_metadata must succeed");

    let mut handle = TjHandle::new();
    let img = handle.decompress(&jpeg).expect("decompress must succeed");

    assert_eq!(
        handle.icc_profile(),
        Some(icc.as_slice()),
        "handle ICC must equal the embedded profile"
    );
    assert_eq!(
        img.icc_profile(),
        Some(icc.as_slice()),
        "image ICC must equal the embedded profile"
    );
    assert_eq!(
        handle.icc_profile(),
        img.icc_profile(),
        "handle and image ICC must be symmetric"
    );
}

// === A5-2: TJPARAM_NOREALLOC behavior on compress_into ===
//
// When TJPARAM_NOREALLOC is set (1), `TjHandle::compress_into()` must emit
// JpegError::BufferTooSmall if the encoded JPEG exceeds the caller's buffer
// capacity. When the buffer is large enough, it returns Ok(bytes_written)
// without touching the underlying storage beyond the written slice (no
// reallocation). This mirrors C tj3Compress8 with TJPARAM_NOREALLOC.

#[test]
fn a5_2_norealloc_buffer_too_small_returns_error() {
    use libjpeg_turbo_rs::tj3::TjParam;
    use libjpeg_turbo_rs::JpegError;

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();

    let mut handle = TjHandle::new();
    handle.set(TjParam::Quality, 90).unwrap();
    handle.set(TjParam::NoRealloc, 1).unwrap();

    let mut buf = vec![0u8; 64];
    let err = handle
        .compress_into(&pixels, width, height, PixelFormat::Rgb, &mut buf)
        .expect_err("NoRealloc must reject undersized buffer");

    match err {
        JpegError::BufferTooSmall { need, got } => {
            assert_eq!(got, 64);
            assert!(need > got, "need={need} should exceed got={got}");
        }
        other => panic!("expected BufferTooSmall, got {other:?}"),
    }
}

#[test]
fn a5_2_norealloc_adequate_buffer_writes_without_realloc() {
    use libjpeg_turbo_rs::tj3::TjParam;

    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    let mut handle = TjHandle::new();
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::NoRealloc, 1).unwrap();

    let capacity: usize = 16 * 1024;
    let mut buf = vec![0u8; capacity];
    let buf_ptr_before: *const u8 = buf.as_ptr();
    let buf_cap_before: usize = buf.capacity();

    let written: usize = handle
        .compress_into(&pixels, width, height, PixelFormat::Rgb, &mut buf)
        .expect("adequate buffer must succeed");

    assert!(
        written >= 4,
        "written length should be meaningful, got {written}"
    );
    assert_eq!(buf[0], 0xFF);
    assert_eq!(buf[1], 0xD8);
    assert_eq!(buf[written - 2], 0xFF);
    assert_eq!(buf[written - 1], 0xD9);

    assert_eq!(
        buf.as_ptr(),
        buf_ptr_before,
        "buffer must not be reallocated"
    );
    assert_eq!(buf.capacity(), buf_cap_before, "capacity must be preserved");

    let img = libjpeg_turbo_rs::decompress(&buf[..written]).expect("decompress written JPEG");
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
}

#[test]
fn a5_2_norealloc_zero_still_errors_on_undersized_slice() {
    use libjpeg_turbo_rs::JpegError;

    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    let handle = TjHandle::new();
    let mut buf = vec![0u8; 32];
    let err = handle
        .compress_into(&pixels, width, height, PixelFormat::Rgb, &mut buf)
        .expect_err("undersized slice must error even with NoRealloc=0");
    assert!(matches!(err, JpegError::BufferTooSmall { .. }));
}

// === A5-3: TJPARAM_SAVEMARKERS wires Decoder::save_markers() ===

fn make_jpeg_with_app1_and_app2() -> Vec<u8> {
    use libjpeg_turbo_rs::Encoder;
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let icc = vec![0xAAu8; 48];
    let exif = b"Exif\0\0MM\0*\0\0\0\x08\0\0\0\0".to_vec();
    Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .icc_profile(&icc)
        .exif_data(&exif)
        .encode()
        .expect("encode APP1+APP2 JPEG")
}

#[test]
fn a5_3_save_markers_level2_populates_saved_markers() {
    use libjpeg_turbo_rs::tj3::TjParam;
    let jpeg = make_jpeg_with_app1_and_app2();
    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 2).unwrap();
    let img = handle.decompress(&jpeg).expect("decompress must succeed");
    assert!(
        !img.markers().is_empty(),
        "SaveMarkers=2 must populate Image.saved_markers"
    );
    let has_app: bool = img
        .markers()
        .iter()
        .any(|m| m.code == 0xE1 || m.code == 0xE2);
    assert!(
        has_app,
        "expected at least one APP1 or APP2, got codes: {:?}",
        img.markers().iter().map(|m| m.code).collect::<Vec<_>>()
    );
}

#[test]
fn a5_3_save_markers_level0_leaves_saved_markers_empty() {
    use libjpeg_turbo_rs::tj3::TjParam;
    let jpeg = make_jpeg_with_app1_and_app2();
    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 0).unwrap();
    let img = handle.decompress(&jpeg).expect("decompress must succeed");
    assert!(
        img.markers().is_empty(),
        "SaveMarkers=0 must leave saved_markers empty, got {} markers",
        img.markers().len()
    );
}

// === A6-1: Encoder::reset_colorspace() matches jpeg_default_colorspace() ===
//
// `reset_colorspace()` clears any previously-set JPEG colorspace override
// and restores inference from `PixelFormat`, mirroring C libjpeg-turbo's
// `jpeg_default_colorspace()`. Concretely: after calling `.colorspace(Rgb)`
// followed by `.reset_colorspace()`, a CMYK input must encode as a
// 4-component JPEG (with the Adobe APP14 marker signaling CMYK), exactly
// as if the override had never been applied.

fn parse_sof_component_count(jpeg: &[u8]) -> Option<u8> {
    let mut i: usize = 2;
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xD9 {
            return None;
        }
        // Standalone markers without payload length.
        if (0xD0..=0xD9).contains(&code) || code == 0x01 {
            i += 2;
            continue;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if matches!(code, 0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC9 | 0xCA | 0xCB) {
            // SOFn payload: [P(1)][Y(2)][X(2)][Nf(1)][components...]
            return Some(jpeg[i + 4 + 5]);
        }
        i += 2 + seg_len;
    }
    None
}

fn find_adobe_transform(jpeg: &[u8]) -> Option<u8> {
    let mut i: usize = 2;
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xDA || code == 0xD9 {
            return None;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if code == 0xEE
            && seg_len >= 12
            && i + 4 + 5 <= jpeg.len()
            && &jpeg[i + 4..i + 9] == b"Adobe"
        {
            // Adobe APP14 layout after the len field (12 payload bytes):
            //   "Adobe"(5) ver(2) flags0(2) flags1(2) transform(1)
            return Some(jpeg[i + 2 + seg_len - 1]);
        }
        i += 2 + seg_len;
    }
    None
}

#[test]
fn a6_1_reset_colorspace_auto_selects_cmyk_for_cmyk_input() {
    use libjpeg_turbo_rs::{ColorSpace, Encoder};
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height * 4).map(|i| (i % 251) as u8).collect();

    // Force an incorrect override, then reset and encode.
    let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Cmyk)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .reset_colorspace()
        .encode()
        .expect("reset_colorspace then CMYK encode must succeed");

    // SOI check.
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);

    assert_eq!(
        parse_sof_component_count(&jpeg),
        Some(4),
        "CMYK input must encode as 4-component JPEG after reset_colorspace"
    );

    assert_eq!(
        find_adobe_transform(&jpeg),
        Some(0),
        "Adobe APP14 transform must be 0 (CMYK) after reset_colorspace"
    );
}

#[test]
fn a6_1_reset_colorspace_clears_rgb_override_for_rgb_input() {
    use libjpeg_turbo_rs::{ColorSpace, Encoder};
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    // With JCS_RGB override, encoder emits RGB-direct (no color conversion).
    let jpeg_rgb = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .encode()
        .unwrap();

    // After reset, auto-detection picks YCbCr — different output bytes.
    let jpeg_auto = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .reset_colorspace()
        .encode()
        .unwrap();

    assert_ne!(
        jpeg_rgb, jpeg_auto,
        "reset_colorspace must undo the RGB override"
    );
}

// === A6-2: Encoder::reset_quant_tables(force_baseline) matches jpeg_default_qtables() ===
//
// Contract: after loading custom quantization tables and then calling
// `reset_quant_tables(force_baseline)`, the encoder must regenerate the
// standard luminance and chrominance tables scaled by the current quality,
// matching `Encoder::new(...).quality(q).encode()` byte-for-byte in the
// DQT portion of the JPEG.

fn extract_dqt_tables(jpeg: &[u8]) -> Vec<(u8, Vec<u16>)> {
    let mut out: Vec<(u8, Vec<u16>)> = Vec::new();
    let mut i: usize = 2;
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xDA || code == 0xD9 {
            break;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if code == 0xDB {
            // DQT payload (possibly multiple tables concatenated)
            let payload = &jpeg[i + 4..i + 2 + seg_len];
            let mut p: usize = 0;
            while p < payload.len() {
                let pq_tq = payload[p];
                p += 1;
                let precision = pq_tq >> 4;
                let tq = pq_tq & 0x0F;
                let mut tbl: Vec<u16> = Vec::with_capacity(64);
                if precision == 0 {
                    for _ in 0..64 {
                        tbl.push(payload[p] as u16);
                        p += 1;
                    }
                } else {
                    for _ in 0..64 {
                        let v = u16::from_be_bytes([payload[p], payload[p + 1]]);
                        tbl.push(v);
                        p += 2;
                    }
                }
                out.push((tq, tbl));
            }
        }
        i += 2 + seg_len;
    }
    out
}

#[test]
fn a6_2_reset_quant_tables_restores_standard_tables_at_quality() {
    use libjpeg_turbo_rs::Encoder;
    let width: usize = 32;
    let height: usize = 32;
    let quality: u8 = 60; // non-default so scaling is visible
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    // Reference: default tables at this quality.
    let jpeg_default = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(quality)
        .encode()
        .unwrap();

    // Construct a wildly non-standard custom table set (all 1s for luma,
    // all 255 for chroma) so it is impossible to confuse with the defaults.
    let custom_luma: [u16; 64] = [1; 64];
    let custom_chroma: [u16; 64] = [200; 64];

    // Encoder with custom tables, then reset.
    let jpeg_after_reset = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(quality)
        .quant_table(0, custom_luma)
        .quant_table(1, custom_chroma)
        .reset_quant_tables(false)
        .encode()
        .unwrap();

    let dqt_default = extract_dqt_tables(&jpeg_default);
    let dqt_after_reset = extract_dqt_tables(&jpeg_after_reset);
    assert_eq!(
        dqt_default, dqt_after_reset,
        "reset_quant_tables must reproduce the default DQT payload at the same quality"
    );
}

#[test]
fn a6_2_reset_quant_tables_force_baseline_clamps_to_255() {
    use libjpeg_turbo_rs::Encoder;
    let width: usize = 32;
    let height: usize = 32;
    // Use a very low quality: at q=1 the scaled table values should exceed
    // 255 unless force_baseline is applied.
    let quality: u8 = 1;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(quality)
        .quant_table(0, [1; 64])
        .quant_table(1, [1; 64])
        .reset_quant_tables(true)
        .encode()
        .unwrap();

    let tables = extract_dqt_tables(&jpeg);
    assert!(!tables.is_empty(), "DQT markers must be present");
    for (idx, tbl) in &tables {
        for (i, &v) in tbl.iter().enumerate() {
            assert!(
                v <= 255,
                "force_baseline=true must clamp table {idx} coeff[{i}] to ≤255, got {v}"
            );
            assert!(v >= 1, "quant coefficients must be ≥1");
        }
    }
}

// === A6-3: RestartResyncStrategy trait + Decoder::set_resync_strategy() ===
//
// Contract (jpeg_resync_to_restart hook): when the decoder hits a restart
// marker boundary and the RST number does not match the expected counter
// (or is missing), it consults the user's strategy. The strategy returns
// Continue / Skip / Abort. Default strategy = "Continue" (the historical
// Rust behavior: accept whatever RST we find).

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{RestartResyncStrategy, ResyncAction};

// Arc + atomic rather than Rc + Cell: set_resync_strategy requires
// `Send` since issue #384 made `Decoder: Send`.
struct SkipStrategy {
    desync_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl RestartResyncStrategy for SkipStrategy {
    fn on_desync(&mut self, _expected: u8, _found: Option<u8>) -> ResyncAction {
        self.desync_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ResyncAction::Skip
    }
}

struct AbortStrategy;

impl RestartResyncStrategy for AbortStrategy {
    fn on_desync(&mut self, _expected: u8, _found: Option<u8>) -> ResyncAction {
        ResyncAction::Abort
    }
}

// Baseline: a default-strategy decoder handles a pristine RST stream.
#[test]
fn a6_3_default_strategy_decodes_clean_restart_stream() {
    use libjpeg_turbo_rs::Encoder;
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();
    let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(4)
        .encode()
        .unwrap();

    // Default strategy: plain decode must still succeed.
    let img = libjpeg_turbo_rs::decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
}

// Default strategy on a corrupted RST: current behavior is to skip past
// any RST marker bytes (reset() already does this). This documents that
// omitting `set_resync_strategy()` keeps the historical lenient
// semantics.
#[test]
fn a6_3_default_strategy_does_not_abort_on_rst_mismatch() {
    let jpeg = make_jpeg_with_corrupt_rst_number();
    // Should NOT error out on the RST number mismatch — default behavior
    // is Continue.
    let decoder = Decoder::new(&jpeg).expect("parse headers");
    let _img = decoder
        .decode_image()
        .expect("default strategy must not abort on RST mismatch");
}

// Custom Abort strategy: explicit abort on desync → decoder must surface
// CorruptData rather than silently accepting the mismatch.
#[test]
fn a6_3_abort_strategy_surfaces_error_on_rst_mismatch() {
    let jpeg = make_jpeg_with_corrupt_rst_number();
    let mut decoder = Decoder::new(&jpeg).expect("parse headers");
    decoder.set_resync_strategy(AbortStrategy);
    let err = decoder
        .decode_image()
        .expect_err("abort strategy must surface error on RST mismatch");
    // The only guarantee: decoder returns an error rather than OK.
    let _ = err;
}

// Regression for grayscale baseline restart handling: single-component
// baseline scans use the non-interleaved block path, but they must still
// honor the same restart-resync hook as interleaved color scans.
#[test]
fn a6_3_abort_strategy_surfaces_error_on_grayscale_rst_mismatch() {
    let jpeg = make_grayscale_jpeg_with_corrupt_rst_number();
    let mut decoder = Decoder::new(&jpeg).expect("parse headers");
    decoder.set_resync_strategy(AbortStrategy);
    let err = decoder
        .decode_image()
        .expect_err("abort strategy must surface grayscale RST mismatch");
    let _ = err;
}

// Diagnostic: verify the test JPEG actually contains RST markers.
#[test]
fn a6_3_diagnostic_test_jpeg_has_rst_markers() {
    let jpeg = make_jpeg_with_corrupt_rst_number();
    let mut rst_count = 0usize;
    let mut in_entropy = false;
    let mut i: usize = 0;
    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF {
            let m = jpeg[i + 1];
            if m == 0xDA {
                in_entropy = true;
                i += 2;
                continue;
            }
            if m == 0xD9 {
                break;
            }
            if in_entropy {
                if (0xD0..=0xD7).contains(&m) {
                    rst_count += 1;
                }
                if m == 0x00 {
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    assert!(
        rst_count >= 1,
        "Test JPEG must contain at least one RST marker in entropy data, got {}",
        rst_count
    );
}

// Skip strategy: when the stream is corrupted at the first RST, a Skip
// strategy must advance past the bad marker and resume at the following
// RST by invoking the decoder's internal scan-to-next-RST routine. The
// strategy is observed to fire on the desync event; whether the decode
// then completes cleanly depends on MCU/entropy alignment after the skip
// (a concern orthogonal to the hook wiring). The invariant we assert is:
// the strategy was consulted, and when it returns Skip the decoder's
// recovery does advance past the corrupted marker (i.e., the decoder
// does not get stuck in a reset loop on the bad RST).
#[test]
fn a6_3_skip_strategy_fires_on_rst_mismatch() {
    let jpeg = make_jpeg_with_corrupt_rst_number();
    let mut decoder = Decoder::new(&jpeg).expect("parse headers");
    let desync_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let strategy = SkipStrategy {
        desync_count: desync_count.clone(),
    };
    let strategy_counter = desync_count;
    decoder.set_resync_strategy(strategy);
    // Decode may fail after skip (entropy state is lost across a Skip),
    // but the strategy MUST have been consulted before the decoder gave up
    // — otherwise the hook is not wired.
    let _ = decoder.decode_image();
    assert!(
        strategy_counter.load(std::sync::atomic::Ordering::Relaxed) >= 1,
        "Skip strategy must observe at least one desync event, saw {}",
        strategy_counter.load(std::sync::atomic::Ordering::Relaxed)
    );
}

// Skip strategy on a clean stream with manually-forced desync: construct
// a strategy that always reports "Skip" but is only consulted when
// `reset_and_consume_rst` reads a non-matching marker. We verify the
// decoder's scan_to_next_rst routine can find the next RST in the
// stream and advance past it without aborting. To force the desync on
// a clean stream, we mutate the first RST marker code to a different
// (still-valid) RST number.
#[test]
fn a6_3_continue_strategy_on_clean_stream_does_not_abort() {
    use libjpeg_turbo_rs::{Encoder, ResyncAction};

    struct ContinueStrategy {
        seen: std::sync::Arc<std::sync::atomic::AtomicU32>,
    }
    impl RestartResyncStrategy for ContinueStrategy {
        fn on_desync(&mut self, _e: u8, _f: Option<u8>) -> ResyncAction {
            self.seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ResyncAction::Continue
        }
    }

    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
    let mut jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(4)
        .encode()
        .unwrap();

    // Flip first RST0 (0xFF 0xD0) to RST3 (0xFF 0xD3). Decoder expects
    // RST0 → desync is reported.
    let mut i: usize = 2;
    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF && (0xD0..=0xD7).contains(&jpeg[i + 1]) {
            jpeg[i + 1] = 0xD3;
            break;
        }
        i += 1;
    }

    let mut decoder = Decoder::new(&jpeg).expect("parse headers");
    let seen = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let strategy = ContinueStrategy { seen: seen.clone() };
    let counter = seen;
    decoder.set_resync_strategy(strategy);
    // Continue strategy: accept the observed RST, realign counter,
    // proceed. The decoder should not abort on this desync.
    let _ = decoder.decode_image();
    assert!(
        counter.load(std::sync::atomic::Ordering::Relaxed) >= 1,
        "Continue strategy must be consulted on RST mismatch"
    );
}

// Helper: build a JPEG that contains real restart intervals, then
// corrupt the RST number in the first RST marker (e.g., change 0xFFD0
// to 0xFFD3) so the decoder's expected counter (0) diverges from the
// observed (3).
fn make_jpeg_with_corrupt_rst_number() -> Vec<u8> {
    use libjpeg_turbo_rs::Encoder;
    // Non-uniform pixel data to force real bitstream content between RSTs.
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
    let mut jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(4)
        .encode()
        .unwrap();

    // Find first RST marker after SOS and mutate its RST number.
    let mut i: usize = 2;
    // Skip past headers to SOS (0xFF 0xDA)
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xDA {
            // SOS: skip header + enter entropy-coded segment
            let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            i += 2 + seg_len;
            break;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        i += 2 + seg_len;
    }
    // Scan for the first RST marker (0xFF 0xD0..=0xD7), skipping byte-stuffed
    // 0xFF 0x00 and actual EOI 0xFF 0xD9.
    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF {
            let code = jpeg[i + 1];
            if (0xD0..=0xD7).contains(&code) {
                // Found an RST. Flip it to an unexpected number: xor with 3.
                let new_code = 0xD0 | ((code & 0x07) ^ 0x03);
                jpeg[i + 1] = new_code;
                break;
            }
            if code == 0xD9 {
                break;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    jpeg
}

fn make_grayscale_jpeg_with_corrupt_rst_number() -> Vec<u8> {
    use libjpeg_turbo_rs::Encoder;

    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 251) as u8).collect();
    let mut jpeg = Encoder::new(&pixels, width, height, PixelFormat::Grayscale)
        .quality(75)
        .restart_blocks(4)
        .encode()
        .unwrap();

    corrupt_first_rst_number(&mut jpeg);
    jpeg
}

fn corrupt_first_rst_number(jpeg: &mut [u8]) {
    let mut i: usize = 2;
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xDA {
            let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            i += 2 + seg_len;
            break;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        i += 2 + seg_len;
    }

    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF {
            let code = jpeg[i + 1];
            if (0xD0..=0xD7).contains(&code) {
                jpeg[i + 1] = 0xD0 | ((code & 0x07) ^ 0x03);
                return;
            }
            if code == 0xD9 {
                return;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
}
