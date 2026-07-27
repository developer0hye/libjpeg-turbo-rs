//! Issue #381: the bridge depended on the codec with
//! `default-features = false, features = ["simd"]`, dropping `std` and
//! with it runtime CPU detection — a stock x86-64-baseline build of the
//! bridge silently ran SSE2-only, never taking the AVX2 paths.
//!
//! This pins the mechanism (feature wiring), not a timing: the codec
//! linked into this bridge must have runtime SIMD detection compiled in.
//!
//! Note on feature unification: this assertion is meaningful when the
//! bridge's dependency graph is resolved on its own (`cargo test -p
//! libjpeg-turbo-rs-image`, or a downstream `cargo add` from crates.io).
//! In a whole-workspace build other members may re-enable `std` for the
//! shared codec build, masking a bad manifest — so CI must run this
//! crate's tests with `-p`.

/// Issue #381: runtime dispatch must be compiled into the codec this
/// bridge links.
#[test]
fn issue_381_codec_has_runtime_simd_detection() {
    assert!(
        libjpeg_turbo_rs::simd_and_std_features_enabled(),
        "the bridge built libjpeg-turbo-rs without the `std` feature: \
         runtime CPU detection is off, so x86-64 builds silently lose \
         AVX2 dispatch (issue #381 regressed)"
    );
}
