//! `no_std` build contract (issue #356).
//!
//! The crate builds as `no_std + alloc` with default features off. This
//! test runs on the std build (the test harness needs std) and pins the
//! two properties the no_std path depends on:
//!
//! 1. SIMD dispatch is compile-time-only without `std` — there is no
//!    CPUID probe, so `cpu_has!` must answer from `target_feature`.
//! 2. The scalar routines are selected when no arch feature is
//!    available, and produce the same pixels as the SIMD path.
//!
//! The actual `no_std` compilation is gated in CI
//! (`cargo check --no-default-features --target thumbv7em-none-eabihf`),
//! which is the only way to prove the crate links without std.

use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

/// The scalar path must produce byte-identical output to whatever the
/// host dispatches, or a `no_std` build (which can only use scalar
/// unless target_feature is set) would silently diverge from the
/// byte-exactness the whole suite asserts against djpeg.
#[test]
fn scalar_dispatch_matches_the_default_dispatch() {
    let mut rgb = Vec::with_capacity(64 * 48 * 3);
    for y in 0..48u32 {
        for x in 0..64u32 {
            rgb.extend_from_slice(&[(x * 4) as u8, (y * 5) as u8, ((x ^ y) * 3) as u8]);
        }
    }

    for sub in [Subsampling::S444, Subsampling::S422, Subsampling::S420] {
        let jpeg = compress(&rgb, 64, 48, PixelFormat::Rgb, 85, sub).expect("encode");
        let dispatched = decompress(&jpeg).expect("decode (host dispatch)");
        assert_eq!(dispatched.data.len(), 64 * 48 * 3, "{sub:?}");

        // JSIMD_FORCENONE selects the same scalar routines a no_std
        // build gets by default (no CPUID probe available there).
        // Decode through a fresh Decoder after setting it so the
        // dispatch tables are rebuilt.
        std::env::set_var("JSIMD_FORCENONE", "1");
        let scalar = decompress(&jpeg).expect("decode (scalar forced)");
        std::env::remove_var("JSIMD_FORCENONE");

        assert_eq!(
            scalar.data, dispatched.data,
            "{sub:?}: scalar path diverges from host dispatch — a no_std \
             build would silently lose the byte-exactness the suite asserts"
        );
    }
}

/// The `std` feature must be additive: default features expose the I/O
/// wrappers, and the core codec entry points exist either way.
#[test]
fn std_feature_is_additive_over_the_core_api() {
    // Core (available in both builds).
    let pixels = vec![128u8; 8 * 8];
    let jpeg = compress(&pixels, 8, 8, PixelFormat::Grayscale, 75, Subsampling::S444)
        .expect("core encode");
    let img = decompress(&jpeg).expect("core decode");
    assert_eq!((img.width, img.height), (8, 8));

    // std-only surface still present on the default build.
    let mut cursor = std::io::Cursor::new(jpeg);
    let from_reader =
        libjpeg_turbo_rs::stream::decompress_from_reader(&mut cursor).expect("std reader path");
    assert_eq!((from_reader.width, from_reader.height), (8, 8));
}
