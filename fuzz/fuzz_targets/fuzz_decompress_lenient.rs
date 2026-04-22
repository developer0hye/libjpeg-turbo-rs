#![no_main]
use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::Decoder;

// Megapixel cap mirrors upstream libjpeg-turbo's fuzz/decompress.cc:
// dimension-bomb SOFs (e.g. 65535x65528) cause >GB allocations long before
// hitting any decoder bug, masking real defects with libfuzzer OOMs.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fuzz_target!(|data: &[u8]| {
    let Ok(mut decoder) = Decoder::new(data) else {
        return;
    };
    let header = decoder.header();
    let pixels = (header.width as u64).saturating_mul(header.height as u64);
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    decoder.set_lenient(true);
    decoder.set_scan_limit(100);
    let _ = decoder.decode_image();
});
