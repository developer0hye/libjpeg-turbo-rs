#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Progressive decoder must handle arbitrary input without panicking.
    if let Ok(mut decoder) = libjpeg_turbo_rs::ProgressiveDecoder::new(data) {
        // Consume all available scans
        while decoder.consume_input().unwrap_or(false) {}
        // Attempt to produce output from whatever scans were consumed
        let _ = decoder.output();
    }
});
