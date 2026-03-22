#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Lenient mode should tolerate even more malformed input without panicking.
    let _ = libjpeg_turbo_rs::decompress_lenient(data);
});
