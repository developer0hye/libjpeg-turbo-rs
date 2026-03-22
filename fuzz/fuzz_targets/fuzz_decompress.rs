#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to decompress arbitrary data — must never panic or abort.
    // All malformed inputs should produce a clean Err result.
    let _ = libjpeg_turbo_rs::decompress(data);
});
