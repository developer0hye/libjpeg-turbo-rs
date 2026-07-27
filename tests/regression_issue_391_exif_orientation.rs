//! Issue #391: EXIF orientation required a full pixel decode to read
//! (`Image::exif_orientation()` was the only accessor), and there was no
//! helper to apply it — despite a complete, `jpegtran`-cross-validated
//! lossless `transform/` module in the same crate. Every photo-displaying
//! application had to hand-roll the 8-case EXIF mapping.
//!
//! Closed by: `Decoder::exif_orientation()` (header probe, no pixel
//! decode), `TransformOp::from_exif_orientation` (DCT domain), and
//! `Image::apply_orientation` (pixel domain), with the pixel path
//! cross-validated case-by-case against the DCT-domain transforms.

use libjpeg_turbo_rs::{
    decompress, transform, Decoder, Encoder, PixelFormat, Subsampling, TransformOp,
};

/// Little-endian TIFF blob with a single IFD0 entry: orientation tag
/// (0x0112, SHORT) set to `orientation`.
fn tiff_with_orientation(orientation: u16) -> Vec<u8> {
    let mut t: Vec<u8> = Vec::new();
    t.extend_from_slice(b"II"); // little-endian
    t.extend_from_slice(&42u16.to_le_bytes()); // TIFF magic
    t.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    t.extend_from_slice(&1u16.to_le_bytes()); // entry count
    t.extend_from_slice(&0x0112u16.to_le_bytes()); // orientation tag
    t.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
    t.extend_from_slice(&1u32.to_le_bytes()); // count
    t.extend_from_slice(&orientation.to_le_bytes()); // value
    t.extend_from_slice(&0u16.to_le_bytes()); // value padding
    t.extend_from_slice(&0u32.to_le_bytes()); // next IFD offset
    t
}

fn photo_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            px.push((255.0 * (0.5 + 0.5 * (fx * 4.9).sin() * (fy * 3.3).cos())) as u8);
            px.push((255.0 * (0.5 + 0.5 * (fx * 2.7 + fy * 5.1).sin())) as u8);
            px.push((255.0 * fx) as u8);
        }
    }
    px
}

/// Encode a photo with the given EXIF orientation embedded.
fn fixture_with_orientation(width: usize, height: usize, orientation: u16) -> Vec<u8> {
    let tiff: Vec<u8> = tiff_with_orientation(orientation);
    Encoder::new(&photo_rgb(width, height), width, height, PixelFormat::Rgb)
        .quality(92)
        .subsampling(Subsampling::S444)
        .exif_data(&tiff)
        .encode()
        .expect("encode with EXIF")
}

/// Issue #391 core: orientation must be readable from the header parse
/// alone — `Decoder::new` runs no pixel decode.
#[test]
fn issue_391_orientation_readable_from_header_probe() {
    for orientation in 1u16..=8 {
        let jpeg: Vec<u8> = fixture_with_orientation(32, 16, orientation);
        let decoder = Decoder::new(&jpeg).expect("header parse");
        assert_eq!(
            decoder.exif_orientation(),
            Some(orientation as u8),
            "orientation {orientation} must be visible pre-decode"
        );
    }
    // No EXIF → None.
    let plain: Vec<u8> = Encoder::new(&photo_rgb(16, 16), 16, 16, PixelFormat::Rgb)
        .encode()
        .expect("encode");
    assert_eq!(
        Decoder::new(&plain).expect("header").exif_orientation(),
        None,
        "no EXIF must probe as None"
    );

    // The caller-buffer path's ImageInfo carries the same accessor.
    let tagged: Vec<u8> = fixture_with_orientation(16, 16, 6);
    let mut buf: Vec<u8> = vec![0u8; 16 * 16 * 3];
    let info = libjpeg_turbo_rs::decompress_into(&tagged, PixelFormat::Rgb, &mut buf)
        .expect("decompress_into");
    assert_eq!(info.exif_orientation(), Some(6), "ImageInfo accessor");
}

/// The DCT-domain mapping follows the standard EXIF table; values
/// outside 1..=8 are rejected rather than guessed.
#[test]
fn issue_391_transform_op_mapping() {
    let expected: [(u8, TransformOp); 8] = [
        (1, TransformOp::None),
        (2, TransformOp::HFlip),
        (3, TransformOp::Rot180),
        (4, TransformOp::VFlip),
        (5, TransformOp::Transpose),
        (6, TransformOp::Rot90),
        (7, TransformOp::Transverse),
        (8, TransformOp::Rot270),
    ];
    for (orientation, op) in expected {
        assert_eq!(
            TransformOp::from_exif_orientation(orientation),
            Some(op),
            "orientation {orientation}"
        );
    }
    assert_eq!(TransformOp::from_exif_orientation(0), None);
    assert_eq!(TransformOp::from_exif_orientation(9), None);
}

/// Pixel-domain apply cross-validated against the DCT-domain transforms
/// (which are themselves `jpegtran`-cross-checked). Byte-exactness is NOT
/// the contract: the integer IDCT rounds its column pass and row pass
/// asymmetrically (`jidctint.c:307` descales pass 1 by `CONST_BITS -
/// PASS1_BITS`, `:410` descales pass 2 by `CONST_BITS + PASS1_BITS + 3`),
/// so decode-then-transpose and transpose-then-decode legitimately differ
/// by one code on a few samples, which YCbCr->RGB widens to at most 3 per
/// channel. C behaves the same way: on this content `djpeg` vs `jpegtran
/// -transpose | djpeg` measures 1 in the Y samples and 3 in RGB.
/// A placement error, by contrast, moves whole gradient values: measured
/// max per-channel diff on this fixture is 3 (orientations 5-8,
/// 2026-07-28; only the axis-swapping cases round at all, 2-4 measure 0),
/// while a one-pixel placement slip measures 13 (horizontal) / 21
/// (vertical) on this fixture's gradient and the tightest of the 56
/// wrong-case mappings measures 251. Assert measured + 1.
#[test]
fn issue_391_apply_orientation_matches_dct_domain_transforms() {
    let (width, height): (usize, usize) = (48, 32);
    for orientation in 1u8..=8 {
        let jpeg: Vec<u8> = fixture_with_orientation(width, height, orientation as u16);

        // Path A: decode, then reorient pixels.
        let upright_a = decompress(&jpeg).expect("decode").apply_orientation();

        // Path B: lossless DCT-domain transform, then decode.
        let op: TransformOp =
            TransformOp::from_exif_orientation(orientation).expect("valid orientation");
        let transformed: Vec<u8> = transform(&jpeg, op).expect("lossless transform");
        let upright_b = decompress(&transformed).expect("decode transformed");

        assert_eq!(
            (upright_a.width, upright_a.height),
            (upright_b.width, upright_b.height),
            "orientation {orientation}: dims"
        );
        let max_diff: u8 = upright_a
            .data
            .iter()
            .zip(upright_b.data.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 4,
            "orientation {orientation}: max per-channel diff {max_diff} vs the \
             jpegtran-validated DCT-domain transform (allowed 4 = measured 3 + 1; \
             a one-pixel placement slip measures >= 13 on this gradient fixture)"
        );
    }
}

/// Placement geometry pinned independently of the implementation: for a
/// lossless (exact) grayscale fixture with distinct values, all four
/// corners of every orientation's output are hand-derived from the EXIF
/// tag definition (tag 0x0112: which source corner is row-0/col-0).
#[test]
fn issue_391_all_orientations_corner_geometry() {
    let (w, h): (usize, usize) = (5, 3);
    let gray: Vec<u8> = (0..w * h).map(|i| (i * 16) as u8).collect();
    let src = |x: usize, y: usize| gray[y * w + x];
    // (orientation, out_w, out_h, [TL, TR, BL, BR] as source coords)
    #[allow(clippy::type_complexity)]
    let cases: [(u16, usize, usize, [(usize, usize); 4]); 8] = [
        (1, w, h, [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]),
        (2, w, h, [(w - 1, 0), (0, 0), (w - 1, h - 1), (0, h - 1)]),
        (3, w, h, [(w - 1, h - 1), (0, h - 1), (w - 1, 0), (0, 0)]),
        (4, w, h, [(0, h - 1), (w - 1, h - 1), (0, 0), (w - 1, 0)]),
        (5, h, w, [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]),
        (6, h, w, [(0, h - 1), (0, 0), (w - 1, h - 1), (w - 1, 0)]),
        (7, h, w, [(w - 1, h - 1), (w - 1, 0), (0, h - 1), (0, 0)]),
        (8, h, w, [(w - 1, 0), (w - 1, h - 1), (0, 0), (0, h - 1)]),
    ];
    for (orientation, out_w, out_h, corners) in cases {
        let jpeg: Vec<u8> = Encoder::new(&gray, w, h, PixelFormat::Grayscale)
            .lossless(true)
            .exif_data(&tiff_with_orientation(orientation))
            .encode()
            .expect("lossless gray encode");
        let img = decompress(&jpeg).expect("decode").apply_orientation();
        assert_eq!(
            (img.width, img.height),
            (out_w, out_h),
            "orientation {orientation}: dims"
        );
        let dst = |x: usize, y: usize| img.data[y * out_w + x];
        let got = [
            dst(0, 0),
            dst(out_w - 1, 0),
            dst(0, out_h - 1),
            dst(out_w - 1, out_h - 1),
        ];
        let want = [
            src(corners[0].0, corners[0].1),
            src(corners[1].0, corners[1].1),
            src(corners[2].0, corners[2].1),
            src(corners[3].0, corners[3].1),
        ];
        assert_eq!(got, want, "orientation {orientation}: corner placement");
    }
}

/// Portrait (h > w) fixture: a dimension-swap bug that only manifests
/// when height exceeds width would pass every landscape case.
#[test]
fn issue_391_portrait_corner_geometry_rot90() {
    let (w, h): (usize, usize) = (3, 5);
    let gray: Vec<u8> = (0..w * h).map(|i| (i * 17) as u8).collect();
    let jpeg: Vec<u8> = Encoder::new(&gray, w, h, PixelFormat::Grayscale)
        .lossless(true)
        .exif_data(&tiff_with_orientation(6))
        .encode()
        .expect("lossless gray encode");
    let img = decompress(&jpeg).expect("decode").apply_orientation();
    assert_eq!((img.width, img.height), (h, w));
    let src = |x: usize, y: usize| gray[y * w + x];
    let dst = |x: usize, y: usize| img.data[y * h + x];
    assert_eq!(dst(0, 0), src(0, h - 1), "top-left");
    assert_eq!(dst(h - 1, 0), src(0, 0), "top-right");
    assert_eq!(dst(0, w - 1), src(w - 1, h - 1), "bottom-left");
    assert_eq!(dst(h - 1, w - 1), src(w - 1, 0), "bottom-right");
}

/// The "works for every pixel format" doc claim, exercised: 2bpp packed
/// Rgb565 and 4bpp Rgba corners on the lossless rot90 fixture.
#[test]
fn issue_391_apply_orientation_2bpp_and_4bpp() {
    use libjpeg_turbo_rs::decompress_to;
    let (w, h): (usize, usize) = (5, 3);
    let gray: Vec<u8> = (0..w * h).map(|i| (i * 16) as u8).collect();
    let jpeg: Vec<u8> = Encoder::new(&gray, w, h, PixelFormat::Grayscale)
        .lossless(true)
        .exif_data(&tiff_with_orientation(6))
        .encode()
        .expect("lossless gray encode");
    let src = |x: usize, y: usize| gray[y * w + x];

    // 4bpp Rgba: gray expands to [v, v, v, 255].
    let rgba = decompress_to(&jpeg, PixelFormat::Rgba)
        .expect("decode rgba")
        .apply_orientation();
    assert_eq!((rgba.width, rgba.height), (h, w));
    let px = |x: usize, y: usize| &rgba.data[(y * h + x) * 4..(y * h + x) * 4 + 4];
    let v = src(0, h - 1);
    assert_eq!(px(0, 0), &[v, v, v, 255], "rgba top-left");
    let v = src(w - 1, 0);
    assert_eq!(px(h - 1, w - 1), &[v, v, v, 255], "rgba bottom-right");

    // 2bpp packed Rgb565: gray v packs to ((v>>3)<<11)|((v>>2)<<5)|(v>>3).
    let px565 = |v: u8| -> [u8; 2] {
        let packed: u16 = (((v as u16) >> 3) << 11) | (((v as u16) >> 2) << 5) | ((v as u16) >> 3);
        packed.to_ne_bytes()
    };
    let rgb565 = decompress_to(&jpeg, PixelFormat::Rgb565)
        .expect("decode rgb565")
        .apply_orientation();
    assert_eq!((rgb565.width, rgb565.height), (h, w));
    let at = |x: usize, y: usize| &rgb565.data[(y * h + x) * 2..(y * h + x) * 2 + 2];
    assert_eq!(at(0, 0), &px565(src(0, h - 1)), "565 top-left");
    assert_eq!(at(h - 1, w - 1), &px565(src(w - 1, 0)), "565 bottom-right");
}

/// Big-endian ("MM") EXIF and the repo's curated real-layout fixtures:
/// the header probe must agree with the fixture set that
/// worker_b4_exif_orientation.rs djpeg-cross-validates.
#[test]
fn issue_391_probe_mm_endian_and_curated_fixtures() {
    // MM variant of the synthetic helper.
    let mut t: Vec<u8> = Vec::new();
    t.extend_from_slice(b"MM");
    t.extend_from_slice(&42u16.to_be_bytes());
    t.extend_from_slice(&8u32.to_be_bytes());
    t.extend_from_slice(&1u16.to_be_bytes());
    t.extend_from_slice(&0x0112u16.to_be_bytes());
    t.extend_from_slice(&3u16.to_be_bytes());
    t.extend_from_slice(&1u32.to_be_bytes());
    t.extend_from_slice(&6u16.to_be_bytes());
    t.extend_from_slice(&0u16.to_be_bytes());
    t.extend_from_slice(&0u32.to_be_bytes());
    let jpeg: Vec<u8> = Encoder::new(&[128u8; 64], 8, 8, PixelFormat::Grayscale)
        .exif_data(&t)
        .encode()
        .expect("encode MM");
    assert_eq!(
        Decoder::new(&jpeg).expect("header").exif_orientation(),
        Some(6),
        "big-endian (MM) EXIF"
    );

    // Curated fixtures with the real APP1-after-SOI layout.
    for orientation in 1u8..=8 {
        let path: String = format!("tests/fixtures/exif_orientation/orient_{orientation}_16x8.jpg");
        let bytes: Vec<u8> = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("curated fixture {path} must exist: {e}"));
        assert_eq!(
            Decoder::new(&bytes).expect("header").exif_orientation(),
            Some(orientation),
            "curated fixture {path}"
        );
    }
}

/// apply_orientation on an image without EXIF (or orientation 1) is the
/// identity.
#[test]
fn issue_391_apply_orientation_identity_cases() {
    let plain: Vec<u8> = Encoder::new(&photo_rgb(32, 16), 32, 16, PixelFormat::Rgb)
        .quality(92)
        .encode()
        .expect("encode");
    let img = decompress(&plain).expect("decode");
    let before = img.data.clone();
    let after = img.apply_orientation();
    assert_eq!(after.data, before, "no EXIF: identity");
    assert_eq!((after.width, after.height), (32, 16));

    // Explicit orientation 1 must also be the identity.
    let tagged: Vec<u8> = fixture_with_orientation(32, 16, 1);
    let img1 = decompress(&tagged).expect("decode");
    let before1 = img1.data.clone();
    let after1 = img1.apply_orientation();
    assert_eq!(after1.data, before1, "orientation 1: identity");

    // And the parameterized primitive rejects out-of-range values.
    let img9 = decompress(&plain).expect("decode");
    let before9 = img9.data.clone();
    let after9 = img9.apply_orientation_value(9);
    assert_eq!(after9.data, before9, "orientation 9: identity");
}

/// Odd (non-iMCU) dimensions still reorient correctly in the pixel
/// domain — checked against a hand-computed corner map for rot90.
#[test]
fn issue_391_apply_orientation_odd_dimensions_rot90() {
    let (width, height): (usize, usize) = (5, 3);
    // Distinct grayscale values 0..15.
    let gray: Vec<u8> = (0..width * height).map(|i| (i * 16) as u8).collect();
    let jpeg: Vec<u8> = Encoder::new(&gray, width, height, PixelFormat::Grayscale)
        .lossless(true)
        .exif_data(&tiff_with_orientation(6))
        .encode()
        .expect("lossless gray encode");
    let img = decompress(&jpeg).expect("decode");
    assert_eq!(img.data, gray, "lossless round-trip");
    let rotated = img.apply_orientation();
    assert_eq!((rotated.width, rotated.height), (height, width));
    // Rot90 CW: dst(dx, dy) = src(dy, h-1-dx). Check all four corners.
    let src = |x: usize, y: usize| gray[y * width + x];
    let dst = |x: usize, y: usize| rotated.data[y * height + x];
    assert_eq!(dst(0, 0), src(0, height - 1), "top-left");
    assert_eq!(dst(height - 1, 0), src(0, 0), "top-right");
    assert_eq!(dst(0, width - 1), src(width - 1, height - 1), "bottom-left");
    assert_eq!(
        dst(height - 1, width - 1),
        src(width - 1, 0),
        "bottom-right"
    );
}
