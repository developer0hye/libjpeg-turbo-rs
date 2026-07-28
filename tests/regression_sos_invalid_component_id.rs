//! Regression tests for SOS component-id validation.
//!
//! Found by scheduled Fuzz Smoke `fuzz_read_coefficients` run 29815394302
//! (2026-07-21, timeout-7a780449): a 16400x48 arithmetic-progressive stream
//! with 1371 scans, where scan 8 (and many later scans) references
//! component id 2 while the frame declares only id 1. C libjpeg-turbo
//! rejects the stream at that scan with ERREXIT(JERR_BAD_COMPONENT_ID,
//! "Invalid component ID %d in SOS") (jdmarker.c get_sos, verified: both
//! jpegtran and jpeg_read_coefficients fail in ~3 ms). Our marker reader
//! accepted the scan header and kept decoding all 1371 scans (~670 ms
//! native, a 30 s+ libFuzzer timeout under instrumentation).
//!
//! The mechanism pinned here: an SOS whose component id does not bind to
//! a distinct frame component must fail marker parsing — that ends the
//! pathological stream at the first bad scan exactly like C.
//!
//! Tracked as P4-37 in docs/last_mile/phase4.md.

mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{read_coefficients, Decoder};

/// Minimal baseline JPEG: frame declares a single component with id 1,
/// but the SOS references component id 2.
fn fixture_sos_wrong_component_id() -> Vec<u8> {
    let mut jpeg: Vec<u8> = Vec::new();
    jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
    jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]); // DQT slot 0
    jpeg.extend_from_slice(&[0x10; 64]);
    // SOF0: 8x8, 1 component, id=1, 1x1 sampling, Tq0
    jpeg.extend_from_slice(&[
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
    ]);
    // DHT DC0: one 1-bit code for symbol 0
    jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01]);
    jpeg.extend_from_slice(&[0x00; 15]);
    jpeg.push(0x00);
    // DHT AC0: one 1-bit code for symbol 0 (EOB)
    jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01]);
    jpeg.extend_from_slice(&[0x00; 15]);
    jpeg.push(0x00);
    // SOS: 1 component, id=2 (frame only has id=1)
    jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x02, 0x00, 0x00, 0x3F, 0x00]);
    jpeg.extend_from_slice(&[0x00, 0x00]); // entropy filler
    jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI
    jpeg
}

/// Same frame, but the SOS lists component id 1 twice — C rejects a
/// repeated CSi with the same JERR_BAD_COMPONENT_ID error.
fn fixture_sos_duplicate_component_id() -> Vec<u8> {
    let mut jpeg: Vec<u8> = Vec::new();
    jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
    jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]); // DQT slot 0
    jpeg.extend_from_slice(&[0x10; 64]);
    // SOF0: 8x8, 2 components (id 1, id 2), 1x1 sampling, Tq0
    jpeg.extend_from_slice(&[
        0xFF, 0xC0, 0x00, 0x0E, 0x08, 0x00, 0x08, 0x00, 0x08, 0x02, 0x01, 0x11, 0x00, 0x02, 0x11,
        0x00,
    ]);
    // DHT DC0 + AC0 as above
    jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01]);
    jpeg.extend_from_slice(&[0x00; 15]);
    jpeg.push(0x00);
    jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01]);
    jpeg.extend_from_slice(&[0x00; 15]);
    jpeg.push(0x00);
    // SOS: 2 components, both id=1
    jpeg.extend_from_slice(&[
        0xFF, 0xDA, 0x00, 0x0A, 0x02, 0x01, 0x00, 0x01, 0x00, 0x00, 0x3F, 0x00,
    ]);
    jpeg.extend_from_slice(&[0x00, 0x00]); // entropy filler
    jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI
    jpeg
}

fn assert_all_paths_reject(source: &[u8], label: &str) {
    // Whichever layer surfaces it (header parse or coefficient read),
    // no path may accept the stream, and the surfaced error must be the
    // component-binding rejection — not some unrelated failure.
    let coeff_err = read_coefficients(source)
        .err()
        .unwrap_or_else(|| panic!("{label}: read_coefficients must reject the stream"));
    assert!(
        matches!(&coeff_err, libjpeg_turbo_rs::JpegError::CorruptData(msg)
            if msg.contains("component ID")),
        "{label}: expected the C JERR_BAD_COMPONENT_ID-parity rejection, got {coeff_err:?}"
    );
    let decode_err = Decoder::new(source)
        .and_then(|d| d.decode_image())
        .err()
        .unwrap_or_else(|| panic!("{label}: decode must reject the stream"));
    assert!(
        matches!(&decode_err, libjpeg_turbo_rs::JpegError::CorruptData(msg)
            if msg.contains("component ID")),
        "{label}: expected the C JERR_BAD_COMPONENT_ID-parity rejection, got {decode_err:?}"
    );
}

/// Progressive frame (id 1 only) whose second scan references component
/// id 2 — the exact shape of the fuzz timeout stream, where the scan was
/// silently skipped instead of rejected.
fn fixture_progressive_sos_wrong_component_id() -> Vec<u8> {
    let mut jpeg: Vec<u8> = Vec::new();
    jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
    jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]); // DQT slot 0
    jpeg.extend_from_slice(&[0x10; 64]);
    // SOF2 (progressive): 8x8, 1 component, id=1, 1x1 sampling, Tq0
    jpeg.extend_from_slice(&[
        0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
    ]);
    // DHT DC0: one 1-bit code for symbol 0
    jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01]);
    jpeg.extend_from_slice(&[0x00; 15]);
    jpeg.push(0x00);
    // Scan 1: DC first (id=1, Ss=0 Se=0 Ah=0 Al=1) — decodes one zero block.
    jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01]);
    jpeg.push(0x00); // entropy: DC category 0
                     // Scan 2: AC first for component id 2, which the frame never declares.
    jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x02, 0x00, 0x01, 0x05, 0x02]);
    jpeg.push(0x00); // entropy filler
    jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI
    jpeg
}

#[test]
fn sos_with_unknown_component_id_is_rejected() {
    assert_all_paths_reject(&fixture_sos_wrong_component_id(), "unknown-id");
}

#[test]
fn progressive_sos_with_unknown_component_id_is_rejected() {
    assert_all_paths_reject(
        &fixture_progressive_sos_wrong_component_id(),
        "progressive-unknown-id",
    );
}

#[test]
fn sos_with_duplicate_component_id_is_rejected() {
    assert_all_paths_reject(&fixture_sos_duplicate_component_id(), "duplicate-id");
}

/// Same two-component frame, but the interleaved SOS lists the components
/// in the opposite order of the frame header ([id2, id1] vs frame
/// [id1, id2]). C's get_sos rejects this through its `cur_comp_info`
/// index-aliasing (the search for scan position i only considers frame
/// slots >= i), so we must too.
fn fixture_sos_swapped_component_order() -> Vec<u8> {
    let mut jpeg: Vec<u8> = fixture_sos_duplicate_component_id();
    // Rewrite the SOS component list [id1, id1] -> [id2, id1]. The SOS
    // payload starts 14 bytes from the end (built by the shared fixture):
    // marker(2) len(2) ns(1) [id,tables]x2 Ss Se AhAl + entropy(2) + EOI(2).
    let sos_first_id: usize = jpeg.len() - 11;
    assert_eq!(jpeg[sos_first_id], 0x01);
    jpeg[sos_first_id] = 0x02;
    jpeg
}

#[test]
fn sos_with_swapped_component_order_is_rejected() {
    assert_all_paths_reject(&fixture_sos_swapped_component_order(), "swapped-order");
}

/// C cross-validation: stock djpeg rejects both fixtures with
/// "Invalid component ID ... in SOS".
#[test]
fn sos_component_id_fixtures_rejected_by_c_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    for (source, label) in [
        (fixture_sos_wrong_component_id(), "unknown-id"),
        (fixture_sos_duplicate_component_id(), "duplicate-id"),
        (
            fixture_progressive_sos_wrong_component_id(),
            "progressive-unknown-id",
        ),
        (fixture_sos_swapped_component_order(), "swapped-order"),
    ] {
        let jpeg_file = helpers::TempFile::new(&format!("sos_bad_component_{label}.jpg"));
        jpeg_file.write_bytes(&source);
        let output = std::process::Command::new(&djpeg)
            .arg("-pnm")
            .arg(jpeg_file.path())
            .output()
            .expect("djpeg should spawn");
        assert!(
            !output.status.success(),
            "{label}: expected djpeg rejection, got exit {:?} (stderr: {})",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
