use libjpeg_turbo_rs::{compress, decompress, Encoder, Image, PixelFormat, Subsampling};
use std::sync::Arc;
use std::thread;

/// Generate a deterministic pixel buffer for testing.
fn make_pixels(width: usize, height: usize, seed: u8) -> Vec<u8> {
    (0..width * height * 3)
        .map(|i| ((i as u32 * 37 + seed as u32 * 53 + 13) % 256) as u8)
        .collect()
}

/// Create a small test JPEG from given seed.
fn make_test_jpeg(seed: u8) -> Vec<u8> {
    let pixels = make_pixels(32, 32, seed);
    compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap()
}

/// Shared JPEG data for concurrent decode tests.
fn shared_jpeg() -> Arc<Vec<u8>> {
    Arc::new(make_test_jpeg(42))
}

// --- Concurrent decode tests ---

#[test]
fn ten_threads_decode_same_jpeg_all_correct() {
    let jpeg_data = shared_jpeg();
    // Decode once as reference
    let reference: Image = decompress(&jpeg_data).unwrap();

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let data = Arc::clone(&jpeg_data);
            thread::spawn(move || decompress(&data).unwrap())
        })
        .collect();

    for handle in handles {
        let image: Image = handle.join().expect("thread should not panic");
        assert_eq!(image.width, reference.width);
        assert_eq!(image.height, reference.height);
        assert_eq!(
            image.data, reference.data,
            "decoded pixels should match reference"
        );
    }
}

// --- Concurrent encode tests ---

#[test]
fn ten_threads_encode_different_images() {
    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let pixels = make_pixels(16, 16, i as u8);
                let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
                    .quality(75)
                    .subsampling(Subsampling::S444)
                    .encode()
                    .unwrap();
                // Verify the produced JPEG is valid by decoding it
                let image = decompress(&jpeg).unwrap();
                assert_eq!(image.width, 16);
                assert_eq!(image.height, 16);
                jpeg
            })
        })
        .collect();

    let results: Vec<Vec<u8>> = handles
        .into_iter()
        .map(|h| h.join().expect("thread should not panic"))
        .collect();

    // Each thread used a different seed, so outputs should generally differ
    // (at minimum, they all succeeded)
    assert_eq!(results.len(), 10);
    for jpeg in &results {
        assert!(!jpeg.is_empty(), "encoded JPEG should not be empty");
        // Verify SOI marker
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
    }
}

// --- Mixed encode/decode ---

#[test]
fn mixed_encode_and_decode_threads() {
    let jpeg_data = shared_jpeg();

    let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

    // 5 decoder threads
    for _ in 0..5 {
        let data = Arc::clone(&jpeg_data);
        handles.push(thread::spawn(move || {
            let image: Image = decompress(&data).unwrap();
            assert_eq!(image.width, 32);
            assert_eq!(image.height, 32);
            assert!(!image.data.is_empty());
        }));
    }

    // 5 encoder threads
    for i in 0..5 {
        handles.push(thread::spawn(move || {
            let pixels = make_pixels(16, 16, i as u8 + 100);
            let jpeg = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
            let image = decompress(&jpeg).unwrap();
            assert_eq!(image.width, 16);
            assert_eq!(image.height, 16);
        }));
    }

    for handle in handles {
        handle.join().expect("thread should not panic");
    }
}

// --- SIMD dispatch consistency ---

#[test]
fn simd_detect_consistent_across_threads() {
    // Decode the same image in multiple threads and verify identical output,
    // which proves the SIMD dispatch produces consistent results regardless of thread.
    let jpeg_data = shared_jpeg();
    let reference = decompress(&jpeg_data).unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let data = Arc::clone(&jpeg_data);
            thread::spawn(move || decompress(&data).unwrap())
        })
        .collect();

    for handle in handles {
        let image = handle.join().expect("thread should not panic");
        assert_eq!(
            image.data, reference.data,
            "SIMD dispatch should produce identical results across threads"
        );
    }
}

// --- Pipeline: decode in one thread, encode in another ---

#[test]
fn decode_then_encode_across_threads() {
    let jpeg_data = shared_jpeg();

    // Decode in thread 1
    let decoded = thread::spawn(move || decompress(&jpeg_data).unwrap())
        .join()
        .expect("decode thread should not panic");

    // Encode the decoded pixels in thread 2
    let width = decoded.width;
    let height = decoded.height;
    let pixel_format = decoded.pixel_format;
    let data = decoded.data;

    let re_encoded = thread::spawn(move || {
        Encoder::new(&data, width, height, pixel_format)
            .quality(80)
            .subsampling(Subsampling::S444)
            .encode()
            .unwrap()
    })
    .join()
    .expect("encode thread should not panic");

    // Verify the re-encoded JPEG is valid
    let final_image = decompress(&re_encoded).unwrap();
    assert_eq!(final_image.width, width);
    assert_eq!(final_image.height, height);
}

// --- Stress test ---

#[test]
fn stress_100_concurrent_decodes() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg = Arc::new(compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap());

    let handles: Vec<_> = (0..100)
        .map(|_| {
            let data = Arc::clone(&jpeg);
            thread::spawn(move || {
                let image = decompress(&data).unwrap();
                assert_eq!(image.width, 8);
                assert_eq!(image.height, 8);
                assert_eq!(image.data.len(), 8 * 8 * 3);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread should not panic");
    }
}

// --- Send + Sync compile-time checks ---

/// Compile-time verification that Image is Send.
/// This test passes if it compiles; the function is never called at runtime.
fn _assert_image_is_send() {
    fn require_send<T: Send>() {}
    require_send::<Image>();
}

/// Compile-time verification that Encoder is Send.
fn _assert_encoder_is_send() {
    fn require_send<T: Send>() {}
    require_send::<Encoder<'static>>();
}

#[test]
fn image_is_send_verified() {
    // This test just exercises the compile-time check above.
    // If Image were not Send, this file would fail to compile.
    let jpeg = make_test_jpeg(1);
    let image = decompress(&jpeg).unwrap();
    // Move image to another thread (proves Send)
    let handle = thread::spawn(move || {
        assert_eq!(image.width, 32);
        image
    });
    let _returned = handle.join().unwrap();
}

#[test]
fn encoder_is_send_verified() {
    // Prove Encoder can be moved to another thread
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let handle = thread::spawn(move || {
        let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
            .quality(75)
            .encode()
            .unwrap();
        assert!(!jpeg.is_empty());
    });
    handle.join().unwrap();
}
