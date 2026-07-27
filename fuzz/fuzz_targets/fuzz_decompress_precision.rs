#![no_main]
//! Fuzz the 12/16-bit and arbitrary-precision decode entry points
//! (`src/api/precision.rs`).
//!
//! Issue #382: this ~1.9k-LOC module had no fuzz coverage at all, which
//! is exactly why the smoke sweeps that caught the 8-bit lossless
//! shift-overflow (P4-38) could not catch its 12/16-bit twin — a
//! crafted SOF3 `P=2`/`Al=15` stream made
//! `1 << (precision - pt - 1)` negative: panic in debug, wrong pixels
//! in release. Every entry point here must return `Err` on malformed
//! input, never panic.

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::precision::{decompress_12bit, decompress_16bit, decompress_lossless_arbitrary};

// Same rationale as fuzz_decompress: bound decoded size so the fuzzer
// spends time in parsing/entropy/prediction code, not in huge allocs.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fuzz_target!(|data: &[u8]| {
    // Cheap pre-parse for a dimension cap; malformed headers fall through
    // to the decoders, which must reject them with typed errors.
    if let Ok(decoder) = libjpeg_turbo_rs::Decoder::new(data) {
        let header = decoder.header();
        let pixels = (header.width as u64).saturating_mul(header.height as u64);
        if pixels > MAX_FUZZ_PIXELS {
            return;
        }
    }
    let _ = decompress_12bit(data);
    let _ = decompress_16bit(data);
    let _ = decompress_lossless_arbitrary(data);
});
