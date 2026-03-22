#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Coefficient reading must handle arbitrary input without panicking.
    let _ = libjpeg_turbo_rs::read_coefficients(data);
});
