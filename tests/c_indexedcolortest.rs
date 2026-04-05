//! C indexedcolortest.in parity tests — color quantization cross-validation.
//!
//! C reference: references/libjpeg-turbo/test/indexedcolortest.in
//!
//! Color quantization (indexed color, dithering) is NOT yet implemented in
//! libjpeg-turbo-rs.  All tests are `#[ignore]` and will activate when the
//! feature is added.
//!
//! The C test uses MD5 hash comparison against pre-computed baselines (not
//! binary cmp).  When implementing, use MD5 validation against the same
//! reference hashes from the C script.

// ---------------------------------------------------------------------------
// Output format × color depth × conversion matrix
// ---------------------------------------------------------------------------

/// Indexed color test for 8-bit precision images.
///
/// C script iterates: ext × colors × conversion
///   ext:        png, ppm, bmp, os2(→bmp), gif, gif0(→gif), tga  (7 formats)
///   colors:     128, 256
///   conversion: RGB→RGB, RGB→GRAY, GRAY→GRAY, GRAY→RGB
///
/// Total: 7 × 2 × 4 = 56 per source image, × 2 images = 112 scenarios.
#[test]
#[ignore = "not yet implemented: indexed color quantization (djpeg -colors)"]
fn c_indexedcolortest_8bit() {
    let _formats: [&str; 7] = ["png", "ppm", "bmp", "os2", "gif", "gif0", "tga"];
    let _color_depths: [u16; 2] = [128, 256];

    // When implemented:
    // for ext in formats:
    //   for colors in color_depths:
    //     for (src, dst) in [(RGB,RGB), (RGB,GRAY), (GRAY,GRAY), (GRAY,RGB)]:
    //       1. Encode source to JPEG at precision 8
    //       2. Decode JPEG → quantized image (colors) in format ext
    //       3. Re-encode quantized image to lossless JPEG
    //       4. Compare MD5 hash against C reference value
    //
    // Reference MD5 hashes are in the C script (lines 33-95).
    todo!("Implement indexed color quantization tests");
}

/// Indexed color test for 12-bit precision images.
///
/// C script tests only png and ppm formats for 12-bit (skip bmp, gif, os2, tga).
/// Total: 2 × 2 × 4 = 16 per source image, × 2 images = 32 scenarios.
#[test]
#[ignore = "not yet implemented: indexed color quantization (12-bit precision)"]
fn c_indexedcolortest_12bit() {
    let _formats: [&str; 2] = ["png", "ppm"];
    let _color_depths: [u16; 2] = [128, 256];

    // When implemented:
    // Same as 8-bit but with monkey16 source images and only png/ppm output.
    // C script lines 100-160.
    todo!("Implement 12-bit indexed color quantization tests");
}

/// Cross-precision test: 8-bit quantized → 12-bit lossless re-encode.
///
/// C script encodes 8-bit quantized output to 12-bit lossless JPEG and
/// validates deterministic round-trip via MD5.  ~32 scenarios.
#[test]
#[ignore = "not yet implemented: indexed color cross-precision encode"]
fn c_indexedcolortest_cross_precision() {
    // When implemented:
    // for non-binary formats (png, ppm):
    //   Encode 8-bit quantized output → 12-bit lossless JPEG
    //   Compare MD5 hash
    todo!("Implement cross-precision indexed color tests");
}
