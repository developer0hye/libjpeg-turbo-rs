#![no_main]
use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{Decoder, ProgressiveDecoder};

// See fuzz_decompress_lenient.rs for the rationale behind this cap.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fuzz_target!(|data: &[u8]| {
    // Peek dimensions cheaply via Decoder::new (metadata-only, no per-pixel
    // allocation). ProgressiveDecoder::new pre-allocates coefficient buffers
    // sized by SOF dimensions, so the cap must apply before constructing it.
    let Ok(decoder) = Decoder::new(data) else {
        return;
    };
    let header = decoder.header();
    let pixels = (header.width as u64).saturating_mul(header.height as u64);
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    drop(decoder);

    if let Ok(mut progressive) = ProgressiveDecoder::new(data) {
        while progressive.consume_input().unwrap_or(false) {}
        let _ = progressive.output();
    }
});
