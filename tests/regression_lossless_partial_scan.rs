//! Fuzz Smoke `fuzz_decompress_precision` regression: a lossless SOS
//! listing fewer components than the frame indexed past `dc_tables`.
//!
//! Scheduled runs 30461194331 (2026-07-29 14:30) and 30485530878 (19:41)
//! both aborted at `src/api/precision.rs:1815` with `index out of bounds:
//! the len is 2 but the index is 2`. The minimized seed is a 3-component
//! SOF3 (`Nf=3`) followed by a SOS with `Ns=2`.
//!
//! `decompress_lossless_arbitrary` builds `dc_tables` from
//! `scan.components.len().min(nc)` — so a short SOS yields a short table
//! list — then decodes with `for c in 0..nc`, indexing past the end.
//! `Ns < Nf` is legal at parse time (C splits the remaining components
//! across further scans, `jdmarker.c get_sos`), but this entry point
//! only ever decodes the single fully-interleaved scan in
//! `metadata.scan`, so it must reject the stream rather than index out of
//! bounds.
//!
//! The 8-bit twin in `decode/pipeline.rs` (`decode_lossless_huffman`)
//! already carried this guard, added from a `fuzz_decompress_lenient`
//! finding; the arbitrary-precision path never received it.
//!
//! `decompress_16bit` had the *same* defect at `precision.rs:1316`.
//! Fixing only the arbitrary-precision site let the dispatched Fuzz
//! Smoke run 30504332488 walk straight into the twin, so both now share
//! one `lossless_dc_tables` helper and both are covered here.

use libjpeg_turbo_rs::precision::{decompress_16bit, decompress_lossless_arbitrary};

/// Minimal lossless (SOF3) stream, 4x4, precision 8, predictor 1.
///
/// `frame_components` sets `Nf`; `scan_components` sets `Ns`. Both draw
/// component ids from the same list so the SOS ids always resolve
/// against the frame — the defect under test is the *count* mismatch,
/// not an unknown component id (that path is covered by
/// `regression_sos_invalid_component_id.rs`).
fn lossless_stream(frame_components: usize, scan_components: usize) -> Vec<u8> {
    lossless_stream_with_precision(frame_components, scan_components, 8)
}

/// As above, with an explicit sample precision so the `decompress_16bit`
/// entry point (which requires `P=16`) can be driven by the same fixture.
fn lossless_stream_with_precision(
    frame_components: usize,
    scan_components: usize,
    precision: u8,
) -> Vec<u8> {
    assert!(scan_components <= frame_components);
    let ids: [u8; 4] = [1, 2, 3, 4];

    let mut s: Vec<u8> = vec![0xFF, 0xD8]; // SOI

    // SOF3: len = 8 + 3*Nf
    let sof_len: usize = 8 + 3 * frame_components;
    s.extend_from_slice(&[0xFF, 0xC3]);
    s.extend_from_slice(&[(sof_len >> 8) as u8, sof_len as u8]);
    s.push(precision);
    s.extend_from_slice(&[0x00, 0x04]); // height
    s.extend_from_slice(&[0x00, 0x04]); // width
    s.push(frame_components as u8);
    for &id in ids.iter().take(frame_components) {
        s.extend_from_slice(&[id, 0x11, 0x00]); // 1x1 sampling, quant slot 0
    }

    // DHT: DC table 0, a single 1-bit code for category 0 (diff = 0)
    s.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01]);
    s.extend_from_slice(&[0x00; 15]);
    s.push(0x00);

    // SOS: len = 6 + 2*Ns, Ss=1 (predictor 1), Se=0, Ah/Al=0
    let sos_len: usize = 6 + 2 * scan_components;
    s.extend_from_slice(&[0xFF, 0xDA]);
    s.extend_from_slice(&[(sos_len >> 8) as u8, sos_len as u8]);
    s.push(scan_components as u8);
    for &id in ids.iter().take(scan_components) {
        s.extend_from_slice(&[id, 0x00]); // DC table 0
    }
    s.extend_from_slice(&[0x01, 0x00, 0x00]);

    // Entropy: one '0' bit per decoded sample (the single category-0
    // code). 4x4 pixels x Nf components needs 16*Nf bits; 16 zero bytes
    // covers the largest fixture (Nf=4 -> 64 bits) with slack.
    s.extend_from_slice(&[0x00; 16]);
    s.extend_from_slice(&[0xFF, 0xD9]); // EOI
    s
}

/// The exact shape the fuzzer minimized to: `Nf=3`, `Ns=2`.
#[test]
fn lossless_sos_shorter_than_frame_is_rejected_not_panicking() {
    let stream: Vec<u8> = lossless_stream(3, 2);
    let result = decompress_lossless_arbitrary(&stream);
    assert!(
        result.is_err(),
        "Nf=3/Ns=2 must be a typed error (single-scan entry point), got Ok"
    );
}

/// Generalize past the one minimized seed: every `Ns < Nf` combination
/// must be rejected, not just 3/2. Guards against a fix that only
/// special-cases the reported shape.
#[test]
fn every_short_scan_component_count_is_rejected() {
    for frame_components in 2..=4usize {
        for scan_components in 1..frame_components {
            let stream: Vec<u8> = lossless_stream(frame_components, scan_components);
            let result = decompress_lossless_arbitrary(&stream);
            assert!(
                result.is_err(),
                "Nf={frame_components}/Ns={scan_components} must be a typed error, got Ok"
            );
        }
    }
}

/// `decompress_16bit` is the twin that Fuzz Smoke 30504332488 found once
/// the arbitrary-precision site alone was fixed. Same contract, `P=16`.
#[test]
fn decompress_16bit_short_scan_is_rejected_not_panicking() {
    for frame_components in 2..=4usize {
        for scan_components in 1..frame_components {
            let stream: Vec<u8> =
                lossless_stream_with_precision(frame_components, scan_components, 16);
            let result = decompress_16bit(&stream);
            assert!(
                result.is_err(),
                "16-bit Nf={frame_components}/Ns={scan_components} must be a typed error, got Ok"
            );
        }
    }
}

/// Control for the 16-bit twin: `Ns == Nf` still decodes.
#[test]
fn decompress_16bit_matching_scan_component_count_still_decodes() {
    for n in 1..=4usize {
        let stream: Vec<u8> = lossless_stream_with_precision(n, n, 16);
        let image = decompress_16bit(&stream)
            .unwrap_or_else(|e| panic!("16-bit Nf=Ns={n} fixture must decode: {e}"));
        assert_eq!((image.width, image.height), (4, 4));
        assert_eq!(image.num_components, n);
        // Initial prediction 1 << (16 - 0 - 1) = 32768, every diff zero.
        assert!(
            image.data.iter().all(|&v| v == 32768),
            "16-bit Nf=Ns={n}: expected a uniform 32768 plane"
        );
    }
}

/// Control: `Ns == Nf` must still decode for every component count, so
/// the rejections above are driven by the count mismatch and not by a
/// malformed fixture or an over-broad guard.
#[test]
fn matching_scan_component_count_still_decodes() {
    for n in 1..=4usize {
        let stream: Vec<u8> = lossless_stream(n, n);
        let image = decompress_lossless_arbitrary(&stream)
            .unwrap_or_else(|e| panic!("Nf=Ns={n} fixture must decode: {e}"));
        assert_eq!((image.width, image.height), (4, 4));
        assert_eq!(image.num_components, n);
        assert_eq!(image.data.len(), 4 * 4 * n);
        // Every diff is category 0, so every sample is the initial
        // prediction 1 << (8 - 0 - 1) = 128 propagated across the plane.
        assert!(
            image.data.iter().all(|&v| v == 128),
            "Nf=Ns={n}: expected a uniform 128 plane, got {:?}",
            &image.data[..image.data.len().min(8)]
        );
    }
}
