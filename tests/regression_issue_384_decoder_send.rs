//! Issue #384: `Decoder` was neither `Send` nor `Sync` — compile-verified
//! — and nothing documented it. Handing a configured `Decoder` to rayon or
//! `tokio::task::spawn_blocking` (the thumbnail-farm case) produced an
//! opaque trait-bound error, and only for users of the configurable API:
//! the one-shot helpers construct their decoder internally.
//!
//! Contract pinned here: `Decoder` (and the wrappers that embed it) are
//! `Send`. `Sync` stays off — the decoder uses interior mutability
//! (`RefCell`) for in-decode state and holds `Send`-only callback boxes
//! — but that half is documented on `Decoder`, not asserted here; it
//! matches upstream libjpeg-turbo's one-thread-at-a-time rule per
//! `cinfo`, and our C ABI shim is stricter still (a `cinfo` may not
//! leave its creating thread — docs/ABI_COMPATIBILITY.md).

fn assert_send<T: Send>() {}

/// Issue #384: a configured decoder must move across threads.
#[test]
fn issue_384_decoder_is_send() {
    assert_send::<libjpeg_turbo_rs::Decoder<'static>>();
}

/// The wrappers embedding a Decoder inherit Send.
#[test]
fn issue_384_streaming_and_scanline_decoders_are_send() {
    assert_send::<libjpeg_turbo_rs::api::streaming::StreamingDecoder<'static>>();
    assert_send::<libjpeg_turbo_rs::ScanlineDecoder<'static>>();
}

/// Not just the bound: a real configured decoder must decode correctly
/// on another thread. Gated off wasm32: wasm32-wasip1 has no threads, so
/// `std::thread::spawn` traps (`unreachable`) under wasmtime — the Send
/// bound itself is still compile-asserted above on every target.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn issue_384_decoder_decodes_on_another_thread() {
    let (width, height): (usize, usize) = (32, 24);
    let mut rgb: Vec<u8> = Vec::with_capacity(width * height * 3);
    for i in 0..width * height {
        rgb.extend_from_slice(&[(i % 256) as u8, (i * 3 % 256) as u8, (i * 7 % 256) as u8]);
    }
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &rgb,
        width,
        height,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        90,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("encode");

    let baseline = {
        let mut dec = libjpeg_turbo_rs::Decoder::new(&jpeg).expect("header");
        dec.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgba);
        dec.decode_image().expect("decode")
    };

    let jpeg_arc: std::sync::Arc<Vec<u8>> = std::sync::Arc::new(jpeg);
    let handle = std::thread::spawn({
        let jpeg = std::sync::Arc::clone(&jpeg_arc);
        move || {
            let mut dec = libjpeg_turbo_rs::Decoder::new(&jpeg).expect("header");
            dec.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgba);
            // Move the *configured* decoder to yet another scope to make
            // the Send requirement do real work.
            dec.decode_image().expect("decode on worker thread")
        }
    });
    let threaded = handle.join().expect("worker thread");
    assert_eq!(
        threaded.data, baseline.data,
        "cross-thread decode must match"
    );
}
