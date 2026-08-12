//! Worker-B8 Huffman bomb: crafted JPEG whose Huffman tables use only the
//! maximum 16-bit code lengths, forcing the decoder onto the slow bit-by-bit
//! path for every symbol.
//!
//! # Why this is a real DoS vector
//!
//! A naive decoder that tries to canonicalize the full Huffman tree may
//! allocate `2^16` entries or loop bit-by-bit without a cap — an attacker
//! supplying such a table combined with a dense 256x256 image can inflate
//! per-symbol decode cost by ~1000x relative to a well-formed table whose
//! most-common codes are 2-3 bits.
//!
//! libjpeg-turbo bounds this via a fixed-depth fast-lookup table (9 bits) and
//! a slow path whose cost is O(16 bit-reads / symbol) — so the worst-case
//! decode is still O(pixels). This test fixes a documented upper bound for
//! that worst case. References: libjpeg-turbo's `jdhuff.c` slow_decode and
//! `scanlimit` mitigation.
//!
//! # What we assert
//!
//! - The JPEG header itself parses (fixture must be well-formed, not a
//!   truncation).
//! - Decode of a 256x256 bomb terminates, with a correct-size image or a
//!   structured error. (P4-152: the 1 s wall-clock bound this used to assert is
//!   now a fallback for platforms with no RSS reporting — measured 0.020 ms, so
//!   as a primary bound it was 50 000x above the workload.)
//! - Peak RSS delta stays under 100 MiB (measured baseline ~1-2 MiB).
//! - A decode **without** any limits still terminates — lenient mode may
//!   either succeed or return an error, but never hang.

#[path = "worker_b8_measure.rs"]
mod measure;

use measure::{measure, rss_supported};

use libjpeg_turbo_rs::decode::pipeline::Decoder;

const BOMB_WIDTH: usize = 256;
const BOMB_HEIGHT: usize = 256;

// P4-152: the wall-clock bound that used to sit here is deleted, not converted
// to a ratio, and the measurement is why. On darwin arm64 release the bomb
// decodes in **0.020 ms** (min-of-9, stable to the third decimal over four
// rounds) — the bound was 1000 ms, fifty thousand times above it. Nothing short
// of a hang crosses that, and a hang is what the CI job timeout is for.
//
// A ratio against a control was measured too, and rejected: an ordinary
// 256x256 decode takes 0.071 ms, so the bomb runs at 0.29x an *ordinary* image.
// That is not a slow path being kept honest, it is a fixture with almost no
// entropy data — the bomb is a pathological Huffman *table*, not a large
// payload. Comparing the two would pin the ratio of two unrelated workloads.
//
// What this fixture actually guards is memory, and that assertion stays:
// `BOMB_PEAK_RSS_DELTA_LIMIT` catches the "decoder mistakenly allocates a
// 2^16-entry lookup per symbol" regression, which is the failure a pathological
// table produces. The test also still requires the decode to terminate with a
// correct-size image or a structured error.
//
// The clock survives as a *fallback* on platforms where RSS cannot be read —
// `worker_b8_measure` reports none outside Linux and macOS, and its contract
// says callers "should skip RSS assertions but still run wall-clock bounds so
// the test remains useful". Deleting it everywhere would have left Windows with
// no resource bound at all.
const BOMB_WALL_CLOCK_MS: u128 = 1_000;
/// Peak RSS delta upper bound. Measured <2 MiB. 100 MiB catches the "decoder
/// mistakenly allocates 2^16-entry lookup per symbol" regression.
const BOMB_PEAK_RSS_DELTA_LIMIT: u64 = 100 * 1024 * 1024;

// -----------------------------------------------------------------------------
// JPEG bytestream construction
// -----------------------------------------------------------------------------

/// Build a minimal grayscale JPEG with pathological Huffman tables: all codes
/// are 16 bits long so every symbol hits the slow decode path.
fn build_huffman_bomb_jpeg() -> Vec<u8> {
    let mut jpeg: Vec<u8> = Vec::with_capacity(64 * 1024);

    // SOI
    jpeg.extend_from_slice(&[0xFF, 0xD8]);

    // DQT — single table, all 1s so every AC coefficient is encoded raw.
    // Marker + length (2+65) + table-id (0x00 = 8-bit precision, table 0).
    jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    jpeg.extend_from_slice(&[1u8; 64]);

    // SOF0 (baseline) — 1 component, 8-bit precision.
    jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x0B, 0x08]);
    jpeg.push((BOMB_HEIGHT >> 8) as u8);
    jpeg.push((BOMB_HEIGHT & 0xFF) as u8);
    jpeg.push((BOMB_WIDTH >> 8) as u8);
    jpeg.push((BOMB_WIDTH & 0xFF) as u8);
    jpeg.push(0x01); // 1 component
    jpeg.extend_from_slice(&[0x01, 0x11, 0x00]); // id=1, H=1 V=1, quant table 0

    // DHT DC table 0 — all 16-bit codes.
    // We need enough symbols to cover every possible DC magnitude category.
    // DC symbols are categories 0..=11, so 12 symbols total.
    // bits[16] = 12 (12 symbols all at length 16). Values: 0..11.
    jpeg.extend_from_slice(&[0xFF, 0xC4]);
    let dc_table_symbols: Vec<u8> = (0..12).collect();
    let dht_dc_len: u16 = (2 + 1 + 16 + dc_table_symbols.len()) as u16;
    jpeg.push((dht_dc_len >> 8) as u8);
    jpeg.push((dht_dc_len & 0xFF) as u8);
    jpeg.push(0x00); // table class 0 (DC), id 0
    let mut dc_bits: [u8; 16] = [0; 16];
    dc_bits[15] = dc_table_symbols.len() as u8;
    jpeg.extend_from_slice(&dc_bits);
    jpeg.extend_from_slice(&dc_table_symbols);

    // DHT AC table 0 — all 16-bit codes. Must include 0x00 (EOB) and 0xF0
    // (ZRL) plus every (run << 4) | size we might emit. To keep the table
    // simple we enumerate all 162 standard AC symbols (per JPEG spec Table K.5)
    // and assign them all length 16.
    jpeg.extend_from_slice(&[0xFF, 0xC4]);
    let mut ac_symbols: Vec<u8> = Vec::with_capacity(162);
    ac_symbols.push(0x00); // EOB
    ac_symbols.push(0xF0); // ZRL
    for run in 0u8..=0xF {
        for size in 1u8..=10 {
            ac_symbols.push((run << 4) | size);
        }
    }
    // ac_symbols now has 2 + 16*10 = 162 entries.
    assert_eq!(ac_symbols.len(), 162);
    let dht_ac_len: u16 = (2 + 1 + 16 + ac_symbols.len()) as u16;
    jpeg.push((dht_ac_len >> 8) as u8);
    jpeg.push((dht_ac_len & 0xFF) as u8);
    jpeg.push(0x10); // table class 1 (AC), id 0
                     // 162 symbols exceeds 255, cannot fit in a single length byte — use
                     // multiple lengths: 128 at length 15, 34 at length 16.
    let mut ac_bits: [u8; 16] = [0; 16];
    ac_bits[14] = 128;
    ac_bits[15] = (ac_symbols.len() - 128) as u8;
    jpeg.extend_from_slice(&ac_bits);
    jpeg.extend_from_slice(&ac_symbols);

    // SOS — 1 component, DC table 0, AC table 0. Ss=0, Se=63, Ah=0, Al=0.
    jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);

    // Entropy data: for every MCU emit a single EOB symbol (DC category 0 +
    // AC EOB). We need to know the exact code bits produced by our canonical
    // Huffman builder so the decoder accepts the stream.
    //
    // Canonical Huffman (JPEG Annex C): codes assigned in symbol order, left
    // to right, within each length. With our bits arrays:
    //   DC table: 12 symbols all at length 16. Code values: 0..11, bit-packed
    //             as 16-bit MSB-first.
    //   AC table: 128 symbols at length 15 (codes 0..127), then 34 symbols at
    //             length 16 (codes 256..289).
    let mcus: usize = BOMB_WIDTH.div_ceil(8) * BOMB_HEIGHT.div_ceil(8);
    let dc_code_0: u32 = 0; // first symbol at length 16 -> binary 0000_0000_0000_0000
    let dc_len_0: u8 = 16;
    // AC EOB is ac_symbols[0] == 0x00, first symbol at length 15 -> code 0, 15 bits.
    let ac_eob_code: u32 = 0;
    let ac_eob_len: u8 = 15;

    // Bit buffer with byte-stuffing (0xFF -> 0xFF 0x00).
    let mut bitbuf: u32 = 0;
    let mut bitcnt: u32 = 0;
    let push_bits = |bits: u32, len: u8, out: &mut Vec<u8>, bitbuf: &mut u32, bitcnt: &mut u32| {
        // Append `len` low bits of `bits` MSB-first.
        *bitbuf = (*bitbuf << len) | (bits & ((1u32 << len) - 1));
        *bitcnt += len as u32;
        while *bitcnt >= 8 {
            *bitcnt -= 8;
            let byte: u8 = (*bitbuf >> *bitcnt) as u8;
            out.push(byte);
            if byte == 0xFF {
                out.push(0x00);
            }
        }
    };

    for _ in 0..mcus {
        // DC: symbol 0 (category 0 => difference = 0, no extra bits).
        push_bits(dc_code_0, dc_len_0, &mut jpeg, &mut bitbuf, &mut bitcnt);
        // AC: EOB (ends the block).
        push_bits(ac_eob_code, ac_eob_len, &mut jpeg, &mut bitbuf, &mut bitcnt);
    }
    // Flush remaining bits padded with 1s (JPEG convention).
    if bitcnt > 0 {
        let pad: u32 = 8 - bitcnt;
        push_bits(
            (1u32 << pad) - 1,
            pad as u8,
            &mut jpeg,
            &mut bitbuf,
            &mut bitcnt,
        );
    }

    // EOI
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    jpeg
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn huffman_bomb_256x256_decodes_within_bounds() {
    let jpeg: Vec<u8> = build_huffman_bomb_jpeg();
    // Sanity: the fixture must at least parse headers.
    let header_check: Decoder = Decoder::new(&jpeg).unwrap_or_else(|e| {
        panic!(
            "bomb fixture failed to parse headers — indicates a build bug, \
             NOT a decoder issue (fake-pass guard): {}",
            e
        )
    });
    drop(header_check);

    let (result, m) = measure("huffman_bomb_decode", || {
        let mut decoder: Decoder = Decoder::new(&jpeg).unwrap();
        decoder.set_max_pixels(BOMB_WIDTH * BOMB_HEIGHT);
        decoder.set_max_memory(64 * 1024 * 1024);
        decoder.decode_image()
    });

    // The bomb must either decode successfully or surface an error — NEVER
    // hang, NEVER exceed the RSS bound.
    match &result {
        Ok(img) => {
            assert_eq!(img.width, BOMB_WIDTH);
            assert_eq!(img.height, BOMB_HEIGHT);
        }
        Err(e) => {
            // Accept a decode error (e.g., decoder rejects the table), but
            // the failure must be quick and low-memory — which is what the
            // bounds below verify.
            eprintln!("huffman_bomb_decode returned error (accepted): {}", e);
        }
    }

    if !rss_supported() {
        assert!(
            m.wall_clock.as_millis() < BOMB_WALL_CLOCK_MS,
            "huffman bomb decode wall_clock={:?} exceeds fallback bound {}ms \
             (no RSS on this platform)",
            m.wall_clock,
            BOMB_WALL_CLOCK_MS
        );
    }
    if rss_supported() {
        assert!(
            m.peak_rss_delta_bytes < BOMB_PEAK_RSS_DELTA_LIMIT,
            "huffman bomb decode peak_rss_delta={:.2}MiB exceeds {:.2}MiB \
             — possible regression in Huffman table construction",
            m.peak_rss_delta_mib(),
            BOMB_PEAK_RSS_DELTA_LIMIT as f64 / (1024.0 * 1024.0),
        );
    }
}

#[test]
fn huffman_bomb_fixture_is_well_formed() {
    // Guard against silent fake-pass: the fixture must be byte-stable and
    // parse correctly. If the builder breaks, this fails loudly before the
    // main bound test.
    let jpeg: Vec<u8> = build_huffman_bomb_jpeg();
    assert!(
        jpeg.len() > 512,
        "bomb JPEG suspiciously small: {}",
        jpeg.len()
    );
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "missing SOI");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "missing EOI");
    // Header parse must succeed.
    let _ = Decoder::new(&jpeg)
        .unwrap_or_else(|e| panic!("bomb JPEG fails header parse — fixture bug: {}", e));
}
