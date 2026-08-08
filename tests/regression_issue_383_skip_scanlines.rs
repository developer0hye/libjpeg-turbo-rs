//! Issue #383: `StreamingDecoder::skip_scanlines` was a no-op that
//! reported success — documented "Returns actual number of lines
//! skipped", implemented as `Ok(num_lines)` with no state change, so
//! `decode()` still returned every row.
//!
//! The fixed contract: skipped rows are dropped from the front of every
//! subsequent `decode()` output (C equivalent: `djpeg -skip 0,N-1`,
//! backed by `jpeg_skip_scanlines` in jdapistd.c), calls accumulate,
//! `num_lines` counts output (post-scale) rows, the return value clamps
//! at the image end exactly as C does (skipping everything is allowed
//! and decodes to a zero-height image), and decoded work for
//! fully-skipped iMCU rows is bounded (the pipeline's crop_y machinery
//! skips their IDCT).

mod helpers;

use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

/// djpeg's `-skip` flag arrived with libjpeg-turbo 1.5's partial decode
/// support; probe the usage text rather than assuming.
fn djpeg_supports_skip(djpeg: &PathBuf) -> bool {
    let output = Command::new(djpeg).arg("-?").output();
    match output {
        Ok(o) => {
            let text: String = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            text.contains("-skip")
        }
        Err(_) => false,
    }
}

fn photo_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            px.push((255.0 * (0.5 + 0.5 * (fx * 5.3).sin() * (fy * 2.9).cos())) as u8);
            px.push((255.0 * (0.5 + 0.5 * (fx * 3.1 + fy * 4.7).sin())) as u8);
            px.push((255.0 * fy) as u8);
        }
    }
    px
}

fn fixture_jpeg(width: usize, height: usize) -> Vec<u8> {
    compress(
        &photo_rgb(width, height),
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("encode fixture")
}

fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut pos: usize = 2;
    let read_number = |data: &[u8], pos: &mut usize| -> usize {
        while data[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        let start: usize = *pos;
        while data[*pos].is_ascii_digit() {
            *pos += 1;
        }
        std::str::from_utf8(&data[start..*pos])
            .unwrap()
            .parse()
            .unwrap()
    };
    let w: usize = read_number(data, &mut pos);
    let h: usize = read_number(data, &mut pos);
    let _maxval: usize = read_number(data, &mut pos);
    pos += 1;
    (w, h, data[pos..pos + w * h * 3].to_vec())
}

/// Issue #383 core: skipping N rows must actually remove them — the
/// decode output equals the tail of a full decode. C-tool-free, so it
/// runs on every platform.
#[test]
fn issue_383_skip_drops_rows_from_decode_output() {
    let (width, height): (usize, usize) = (48, 41);
    let jpeg: Vec<u8> = fixture_jpeg(width, height);

    let full = StreamingDecoder::new(&jpeg)
        .expect("header")
        .decode()
        .expect("full decode");
    assert_eq!((full.width, full.height), (width, height));

    for skip in [1usize, 8, 13, 40] {
        let mut dec = StreamingDecoder::new(&jpeg).expect("header");
        let skipped: usize = dec.skip_scanlines(skip).expect("skip");
        assert_eq!(skipped, skip, "skip={skip} fully within the image");
        let img = dec
            .decode()
            .unwrap_or_else(|e| panic!("decode after skip={skip}: {e}"));
        assert_eq!(
            (img.width, img.height),
            (width, height - skip),
            "skip={skip}: decode must return the remaining rows only"
        );
        assert_eq!(
            img.data,
            full.data[skip * width * 3..],
            "skip={skip}: rows must be identical to the tail of a full decode"
        );
    }
}

/// Consecutive calls accumulate; the clamp matches C exactly
/// (`jpeg_skip_scanlines` allows skipping every remaining row and
/// returns the true count).
#[test]
fn issue_383_skip_accumulates_and_clamps() {
    let (width, height): (usize, usize) = (32, 25);
    let jpeg: Vec<u8> = fixture_jpeg(width, height);

    // Accumulation: 2 + 3 == 5.
    let mut split = StreamingDecoder::new(&jpeg).expect("header");
    assert_eq!(split.skip_scanlines(2).expect("skip 2"), 2);
    assert_eq!(split.skip_scanlines(3).expect("skip 3"), 3);
    let split_img = split.decode().expect("decode after 2+3");

    let mut single = StreamingDecoder::new(&jpeg).expect("header");
    assert_eq!(single.skip_scanlines(5).expect("skip 5"), 5);
    let single_img = single.decode().expect("decode after 5");
    assert_eq!(split_img.data, single_img.data, "2+3 must equal 5");

    // Clamp matches C (jdapistd.c: num_lines = output_height -
    // output_scanline): the whole image may be skipped, the return value
    // is the true remaining count, and decode() then yields a
    // zero-height image — djpeg -skip 0,H-1 likewise writes an empty PPM.
    let mut over = StreamingDecoder::new(&jpeg).expect("header");
    let skipped: usize = over.skip_scanlines(height + 100).expect("overskip");
    assert_eq!(
        skipped, height,
        "clamp must return the full remaining count"
    );
    let img = over.decode().expect("decode after full skip");
    assert_eq!(img.height, 0, "full skip must decode to zero rows");
    assert!(img.data.is_empty(), "full skip must decode to empty data");

    // skip(0) is a no-op that must not disturb anything.
    let mut zero = StreamingDecoder::new(&jpeg).expect("header");
    assert_eq!(zero.skip_scanlines(0).expect("skip 0"), 0);
    let zero_img = zero.decode().expect("decode after skip 0");
    assert_eq!(zero_img.height, height);
}

/// Skips landing exactly on 4:2:0 iMCU boundaries (16-row iMCUs): the
/// first kept row is the top of a decoded iMCU row, and the fancy
/// upsampler needs context from a *skipped* row — the pipeline's
/// mcu_row_range extends one iMCU row above for exactly this. Pins that
/// extension against being "optimized away".
#[test]
fn issue_383_imcu_aligned_skip_stays_byte_identical() {
    let (width, height): (usize, usize) = (48, 57);
    let jpeg: Vec<u8> = fixture_jpeg(width, height);
    let full = StreamingDecoder::new(&jpeg)
        .expect("header")
        .decode()
        .expect("full decode");
    for skip in [16usize, 32] {
        let mut dec = StreamingDecoder::new(&jpeg).expect("header");
        assert_eq!(dec.skip_scanlines(skip).expect("skip"), skip);
        let img = dec.decode().expect("decode");
        assert_eq!(
            img.data,
            full.data[skip * width * 3..],
            "skip={skip} (iMCU-aligned) must equal the full-decode tail"
        );
    }
}

/// Progressive streams go through a different entropy path; the skip
/// contract must hold there too.
#[test]
fn issue_383_progressive_skip_matches_full_decode_tail() {
    let (width, height): (usize, usize) = (40, 33);
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress_progressive(
        &photo_rgb(width, height),
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("encode progressive fixture");
    let full = StreamingDecoder::new(&jpeg)
        .expect("header")
        .decode()
        .expect("full decode");
    let mut dec = StreamingDecoder::new(&jpeg).expect("header");
    assert_eq!(dec.skip_scanlines(9).expect("skip"), 9);
    let img = dec.decode().expect("decode");
    assert_eq!(img.data, full.data[9 * width * 3..]);
}

/// `num_lines` counts output rows: under `set_scale(1/2)` the clamp and
/// the skip interpretation follow the scaled height, byte-identical to
/// `djpeg -scale 1/2 -skip`. Pre-review, the clamp used the full-res
/// height, over-reporting the skipped count and silently returning
/// zero-height images.
#[test]
fn issue_383_scaled_skip_clamps_and_matches_c() {
    let (width, height): (usize, usize) = (64, 96);
    let jpeg: Vec<u8> = fixture_jpeg(width, height);
    let scaled_h: usize = height / 2;

    // Clamp is in scaled rows: requesting more than the scaled height
    // returns the scaled remaining count, not the requested count.
    let mut over = StreamingDecoder::new(&jpeg).expect("header");
    over.set_scale(libjpeg_turbo_rs::ScalingFactor::new(1, 2));
    assert_eq!(
        over.skip_scanlines(height).expect("overskip"),
        scaled_h,
        "clamp must count scaled output rows"
    );

    // Byte-identity vs djpeg -scale 1/2 -skip 0,39.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    if !djpeg_supports_skip(&djpeg) {
        eprintln!("SKIP(C xval): djpeg lacks -skip");
        return;
    }
    let skip: usize = 40;
    let tmp_jpg: PathBuf =
        std::env::temp_dir().join(format!("issue383s_{}.jpg", std::process::id()));
    let tmp_ppm: PathBuf =
        std::env::temp_dir().join(format!("issue383s_{}.ppm", std::process::id()));
    std::fs::write(&tmp_jpg, &jpeg).expect("write tmp jpg");
    let output = Command::new(&djpeg)
        .arg("-scale")
        .arg("1/2")
        .arg("-skip")
        .arg(format!("0,{}", skip - 1))
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm)
        .arg(&tmp_jpg)
        .output()
        .expect("run djpeg");
    assert!(
        output.status.success(),
        "djpeg -scale 1/2 -skip failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (cw, ch, c_px) = parse_ppm(&std::fs::read(&tmp_ppm).expect("read ppm"));
    assert_eq!((cw, ch), (width / 2, scaled_h - skip), "djpeg dims");

    let mut dec = StreamingDecoder::new(&jpeg).expect("header");
    dec.set_scale(libjpeg_turbo_rs::ScalingFactor::new(1, 2));
    assert_eq!(dec.skip_scanlines(skip).expect("skip"), skip);
    let img = dec.decode().expect("decode");
    assert_eq!((img.width, img.height), (cw, ch));
    assert_eq!(img.data, c_px, "scaled skip must match djpeg byte-exactly");
    let _ = std::fs::remove_file(&tmp_jpg);
    let _ = std::fs::remove_file(&tmp_ppm);
}

/// A zero-height SOF (legal per DNL) must not panic the clamp
/// arithmetic — pre-review this underflowed `height - 1` in debug.
#[test]
fn issue_383_zero_height_stream_does_not_panic() {
    let (width, height): (usize, usize) = (16, 16);
    let mut jpeg: Vec<u8> = fixture_jpeg(width, height);
    // Patch the SOF0 height field (2 bytes at marker offset +5) to 0.
    let sof: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("SOF0 present");
    jpeg[sof + 5] = 0;
    jpeg[sof + 6] = 0;
    let mut dec = match StreamingDecoder::new(&jpeg) {
        Ok(d) => d,
        // Rejecting the DNL-style header outright is also acceptable.
        Err(_) => return,
    };
    assert_eq!(
        dec.skip_scanlines(5).expect("skip on zero-height header"),
        0,
        "nothing to skip in a zero-height frame"
    );
}

/// C cross-validation: our post-skip output must be byte-identical to
/// `djpeg -skip 0,N-1` on the same stream.
#[test]
fn issue_383_skip_matches_c_djpeg_skip() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    if !djpeg_supports_skip(&djpeg) {
        eprintln!("SKIP: djpeg lacks -skip (need libjpeg-turbo >= 1.5)");
        return;
    }

    let (width, height): (usize, usize) = (64, 53);
    let jpeg: Vec<u8> = fixture_jpeg(width, height);
    let tmp_jpg: PathBuf =
        std::env::temp_dir().join(format!("issue383_{}.jpg", std::process::id()));
    let tmp_ppm: PathBuf =
        std::env::temp_dir().join(format!("issue383_{}.ppm", std::process::id()));
    std::fs::write(&tmp_jpg, &jpeg).expect("write tmp jpg");

    for skip in [1usize, 8, 19] {
        let output = Command::new(&djpeg)
            .arg("-skip")
            .arg(format!("0,{}", skip - 1))
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .expect("run djpeg");
        assert!(
            output.status.success(),
            "djpeg -skip 0,{} failed: {}",
            skip - 1,
            String::from_utf8_lossy(&output.stderr)
        );
        let (cw, ch, c_px) = parse_ppm(&std::fs::read(&tmp_ppm).expect("read ppm"));
        assert_eq!((cw, ch), (width, height - skip), "djpeg output dims");

        let mut dec = StreamingDecoder::new(&jpeg).expect("header");
        assert_eq!(dec.skip_scanlines(skip).expect("skip"), skip);
        let img = dec.decode().expect("decode");
        assert_eq!(
            img.data, c_px,
            "skip={skip}: must be byte-identical to djpeg -skip"
        );
    }
    let _ = std::fs::remove_file(&tmp_jpg);
    let _ = std::fs::remove_file(&tmp_ppm);
}
