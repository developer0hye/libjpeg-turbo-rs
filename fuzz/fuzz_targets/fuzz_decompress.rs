#![no_main]
use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::Decoder;

// See fuzz_decompress_lenient.rs for the rationale behind this cap.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fuzz_target!(|data: &[u8]| {
    let Ok(mut decoder) = Decoder::new(data) else {
        return;
    };
    let (frame_w, frame_h) = {
        let header = decoder.header();
        (header.width, header.height)
    };
    let pixels = (frame_w as u64).saturating_mul(frame_h as u64);
    if frame_w == 0 || frame_h == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    decoder.set_scan_limit(100);
    // Drive the riskiest sink branches too: derive options from the
    // input, applied identically to both decode paths so parity holds.
    // Keyed off the LENGTH, not data[0]: every decodable input starts
    // with SOI's 0xFF, so a data[0]-keyed match always took the same arm
    // (0xFF % 6 == 3) and the other options got zero coverage (codex P2
    // on #386 — a pre-existing harness defect this fixes for all arms).
    match data.len() % 7 {
        1 => decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgba),
        2 => {
            decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb565);
            decoder.set_dither_565(true);
        }
        3 => decoder.set_scale(libjpeg_turbo_rs::ScalingFactor::new(1, 2)),
        4 => decoder.set_crop_region(3, 0, 40, frame_h as usize),
        5 => decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Xrgb),
        // The implied gray route slices component plane 0 directly; a
        // comp0 sampled below max used to panic (#386 review P1).
        6 => decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Grayscale),
        _ => {}
    }
    let owned = decoder.decode_image();

    // Caller-buffer path (#354): must never panic or write past `out`,
    // for any buffer size — including deliberately short ones (typed
    // BufferTooSmall) and oversized ones. When both paths succeed the
    // pixels must be identical.
    // Two sizes per input (each is a full decode — keep fuzz throughput):
    // the advertised size (success + parity) and an input-derived
    // adversarial size (short buffers must yield BufferTooSmall).
    let byte0 = data.first().copied().unwrap_or(0) as usize;
    let advertised = decoder.output_buffer_size().unwrap_or(0);
    for size in [advertised, byte0.min(advertised.saturating_sub(1))] {
        // Cap so a fuzz input claiming huge dimensions cannot OOM the
        // harness through our own test allocation.
        if size as u64 > MAX_FUZZ_PIXELS * 4 {
            continue;
        }
        // Non-zero sentinel fill: the borrowed sink is deliberately not
        // zeroed, so a skipped byte must differ from a written zero for
        // the parity assert to catch it.
        let fill = data.get(1).copied().unwrap_or(0xA5) | 1;
        let mut out = vec![fill; size];
        if let (Ok(info), Ok(img)) = (decoder.decode_image_into(&mut out), owned.as_ref()) {
            assert_eq!(info.bytes_written, img.data.len());
            assert_eq!(&out[..info.bytes_written], &img.data[..]);
        }
    }
});
