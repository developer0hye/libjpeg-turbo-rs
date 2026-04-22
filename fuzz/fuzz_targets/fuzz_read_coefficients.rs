#![no_main]
use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{read_coefficients, Decoder};

// See fuzz_decompress_lenient.rs for the rationale behind this cap.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fuzz_target!(|data: &[u8]| {
    // read_coefficients allocates `mcus_x * mcus_y * components * 64 * sizeof(i16)`
    // up front; a 65535x65528 SOF pushes that to multi-GB. Peek dims first.
    let Ok(decoder) = Decoder::new(data) else {
        return;
    };
    let header = decoder.header();
    let pixels = (header.width as u64).saturating_mul(header.height as u64);
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    drop(decoder);

    let _ = read_coefficients(data);
});
