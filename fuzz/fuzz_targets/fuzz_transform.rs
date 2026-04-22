#![no_main]
use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{read_coefficients, write_coefficients, Decoder};

// See fuzz_decompress_lenient.rs for the rationale behind this cap.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fuzz_target!(|data: &[u8]| {
    // Both read_coefficients and write_coefficients scale memory by image
    // dimensions; cap up front so we exercise the algorithms, not the OOM killer.
    let Ok(decoder) = Decoder::new(data) else {
        return;
    };
    let header = decoder.header();
    let pixels = (header.width as u64).saturating_mul(header.height as u64);
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    drop(decoder);

    if let Ok(coefficients) = read_coefficients(data) {
        let _ = write_coefficients(&coefficients);
    }
});
