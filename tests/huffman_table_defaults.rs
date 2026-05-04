//! Regression tests for Huffman-table default-table handling, mirroring
//! libjpeg-turbo's M-JPEG / refinement-scan compatibility behaviour.
//!
//! Three drop-in agreement contracts are exercised:
//!
//! 1. **Baseline** — when SOS references a DC/AC table slot that no DHT
//!    defined, the decoder fills the slot with the standard Annex K table
//!    (`jdhuff.c::jinit_huff_decoder` → `std_huff_tables`). This is the
//!    M-JPEG "tables omitted from every frame" path. Surfaced by
//!    `fuzz_decode_diff_c` on a 428-byte fixture: djpeg accepted, Rust was
//!    rejecting with `CorruptData("missing DC Huffman table 1")`.
//!
//! 2. **Progressive** — `jinit_phuff_decoder` does *not* auto-fill, so a
//!    progressive scan that names an undefined slot must keep returning a
//!    "missing Huffman table" error. The auto-fill must be gated to
//!    baseline; otherwise Rust would accept inputs djpeg rejects (the
//!    inverse drop-in regression).
//!
//! 3. **Progressive DC refinement** — `decode_dc_refine` reads only one
//!    bit per block and never consumes a Huffman symbol, so libjpeg-turbo
//!    explicitly skips DC table validation for refinement scans
//!    (`start_pass_phuff_decoder`: "DC refinement needs no table"). A SOS
//!    in a refinement scan that names an undefined Td must still decode.
//!
//! The contract this guards is **drop-in agreement with djpeg** on these
//! three input shapes — silently swallowing or unilaterally rejecting any
//! of them is the regression we're locking down.

use libjpeg_turbo_rs::{compress_progressive, Decoder, JpegError, PixelFormat, Subsampling};

/// 428-byte baseline fixture from `fuzz_decode_diff_c`'s 2026-05-04 crash:
/// SOS for a 16×16 RGB JPEG references DC/AC table 1, but only DHTs for
/// table 0 were emitted. djpeg auto-fills table 1 with the Annex K
/// standard tables and decodes; Rust used to reject before the fix.
const BASELINE_MISSING_DHT_FIXTURE: &[u8] = &[
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0,
    80, 55, 60, 70, 60, 50, 80, 79, 65, 70, 90, 85, 80, 95, 120, 200, 130, 120, 110, 110, 120, 245,
    175, 185, 145, 200, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 219, 0, 67, 1, 85, 90, 90, 120, 105, 120, 235, 130, 130, 235,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    192, 0, 17, 8, 0, 16, 0, 16, 3, 1, 17, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0, 31, 0, 0, 1, 5, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16,
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19,
    81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98,
    114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67,
    68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105,
    106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147,
    148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244,
    245, 246, 247, 248, 249, 250, 255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0, 129, 82, 128,
    76, 145, 82, 153, 178, 100, 138, 149, 39, 26, 100, 170, 148, 205, 147, 63, 255, 217,
];

#[test]
fn baseline_sos_referencing_undefined_dht_decodes() {
    let mut d = Decoder::new(BASELINE_MISSING_DHT_FIXTURE).expect("header parse");
    d.set_lenient(true);
    let img = d
        .decode_image()
        .expect("baseline must auto-fill missing DHT slots — djpeg accepts this");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);
    assert_eq!(img.data.len(), 16 * 16 * 3);
}

#[test]
fn baseline_sos_referencing_undefined_dht_strict_mode_decodes() {
    let mut d = Decoder::new(BASELINE_MISSING_DHT_FIXTURE).expect("header parse");
    // Strict mode must also accept — the auto-fill is a libjpeg-turbo
    // compatibility feature, not a recovery heuristic.
    d.set_lenient(false);
    let img = d.decode_image().expect("strict path also auto-fills");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

/// Build a tiny progressive grayscale JPEG, then locate a SOS marker
/// matching the predicate and rewrite its first scan-component selector's
/// Td/Ta byte to point at slot `td` (DC) and `ta` (AC).
fn mutate_progressive_sos<F>(predicate: F, td: u8, ta: u8) -> Vec<u8>
where
    F: Fn(u8, u8, u8) -> bool, // (Ss, Se, Ah) → match?
{
    // Smooth gradient gives every progressive scan content to encode.
    let mut pixels = vec![0u8; 16 * 16];
    for (idx, p) in pixels.iter_mut().enumerate() {
        let x = idx % 16;
        let y = idx / 16;
        *p = ((x * 8 + y * 4) & 0xff) as u8;
    }
    let mut data = compress_progressive(
        &pixels,
        16,
        16,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .expect("compress_progressive");

    // Marker walker that correctly skips entropy-coded data (0xFF 0x00
    // stuffed bytes pass through; any other 0xFF xx ends the segment).
    // Markers without a length field (SOI, EOI, RSTn, TEM) advance by 2.
    let mut i: usize = 2; // skip SOI
    while i + 3 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        // Skip fill bytes (FF FF ...).
        let mut j = i;
        while j < data.len() && data[j] == 0xFF {
            j += 1;
        }
        if j >= data.len() {
            break;
        }
        let code = data[j];
        let after = j + 1;
        // Standalone markers (no length / no entropy follow-on).
        if code == 0xD9 || code == 0x01 || (0xD0..=0xD7).contains(&code) {
            i = after;
            continue;
        }
        // SOS: parse body, then walk entropy data until the next non-stuffed
        // marker (FF xx where xx != 0x00 and xx not in 0xD0..=0xD7).
        if code == 0xDA {
            let length = ((data[after] as usize) << 8) | data[after + 1] as usize;
            let body_start = after + 2;
            let ns = data[body_start];
            assert_eq!(ns, 1, "grayscale → Ns=1 in every SOS");
            let td_ta_off = body_start + 2; // skip Ns + Cs
            let ss_off = body_start + 1 + 2 * ns as usize;
            let ss = data[ss_off];
            let se = data[ss_off + 1];
            let ah = data[ss_off + 2] >> 4;
            if predicate(ss, se, ah) {
                data[td_ta_off] = (td << 4) | (ta & 0x0f);
                return data;
            }
            // Advance past SOS header.
            let mut k = after + length;
            // Walk entropy data.
            while k + 1 < data.len() {
                if data[k] == 0xFF {
                    let nxt = data[k + 1];
                    if nxt == 0x00 || (0xD0..=0xD7).contains(&nxt) {
                        k += 2;
                    } else {
                        break;
                    }
                } else {
                    k += 1;
                }
            }
            i = k;
            continue;
        }
        // Length-prefixed marker.
        if after + 1 >= data.len() {
            break;
        }
        let length = ((data[after] as usize) << 8) | data[after + 1] as usize;
        i = after + length;
    }
    panic!("no matching SOS found in progressive JPEG");
}

#[test]
fn progressive_initial_scan_referencing_undefined_dht_errors() {
    // First DC-initial scan: Ss=0, Se=0, Ah=0. Re-target Td to slot 1
    // (no DHT defined yet for slot 1). libjpeg-turbo's `jinit_phuff_decoder`
    // does *not* auto-fill, so this must keep returning a "missing
    // Huffman table" error — diverging from djpeg in either direction is
    // a drop-in regression.
    let mutated = mutate_progressive_sos(|ss, se, ah| ss == 0 && se == 0 && ah == 0, 1, 0);
    let mut d = Decoder::new(&mutated).expect("header parse");
    d.set_lenient(true);
    match d.decode_image() {
        Err(JpegError::CorruptData(msg)) => {
            assert!(
                msg.contains("Huffman table"),
                "expected missing-Huffman-table error, got: {msg}"
            );
        }
        Ok(_) => panic!(
            "progressive initial scan with undefined DC table must error \
             (matches djpeg's `Huffman table 0x01 was not defined`)"
        ),
        Err(e) => panic!("expected CorruptData(missing Huffman table), got: {e:?}"),
    }
}

#[test]
fn progressive_dc_refinement_with_undefined_td_still_decodes() {
    // DC refinement scan: Ss=0, Se=0, Ah>0. `decode_dc_refine` reads
    // one bit per block — never a Huffman symbol — so the SOS Td
    // selector is unused. libjpeg-turbo (`start_pass_phuff_decoder`:
    // "DC refinement needs no table") skips validation here.
    let mutated = mutate_progressive_sos(|ss, se, ah| ss == 0 && se == 0 && ah > 0, 1, 0);
    let mut d = Decoder::new(&mutated).expect("header parse");
    d.set_lenient(true);
    let img = d
        .decode_image()
        .expect("DC refinement must skip table validation — djpeg accepts");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn dht_with_invalid_table_class_is_rejected() {
    // Tc must be 0 (DC) or 1 (AC). libjpeg-turbo rejects Tc>=2 with
    // `JERR_DHT_INDEX` ("Bogus DHT index %d"). Without this check, a
    // malformed DHT silently routes its data into ac_tables, corrupting
    // unrelated tables — so the fix is paired with the auto-fill above.
    let pixels = vec![128u8; 8 * 8];
    let mut jpeg =
        libjpeg_turbo_rs::compress(&pixels, 8, 8, PixelFormat::Grayscale, 75, Subsampling::S444)
            .expect("baseline compress");
    let pos = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC4])
        .expect("DHT marker present");
    // Set Tc to 2 (invalid). Class byte sits at marker+4.
    jpeg[pos + 4] = 0x20;
    let result = Decoder::new(&jpeg).and_then(|mut d| {
        d.set_lenient(true);
        d.decode_image()
    });
    assert!(
        matches!(result, Err(JpegError::CorruptData(_))),
        "DHT with Tc=2 must error (matches djpeg's `Bogus DHT index`), got: {result:?}"
    );
}
