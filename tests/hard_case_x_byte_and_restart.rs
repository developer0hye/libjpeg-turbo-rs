//! P4-12 phase-2: hard-case decode/encode coverage for two distinct
//! divergence-prone areas that the C-API replacement claim depends on.
//!
//! ## Coverage today
//!
//! 1. **`JCS_EXT_RGBX` / `JCS_EXT_BGRX` X-byte semantics.** Upstream
//!    libjpeg-turbo documents the X (padding) byte in these "extra"
//!    pixel formats as *undefined* on decode and *ignored* on encode.
//!    Consumers downstream (Pillow, OpenCV, GD, ImageMagick) routinely
//!    set X to `0xFF` to repurpose RGBX/BGRX as RGBA/BGRA storage with a
//!    constant opaque alpha.  Our Rust decoder must match the upstream
//!    contract: write the documented constant 0xFF in the X slot on
//!    decode, and silently consume any value in the X slot on encode
//!    (so an RGBA buffer with alpha values can be passed verbatim to an
//!    RGBX encoder without corrupting the JPEG).
//!
//!    Failure modes the test catches:
//!    - X-byte left uninitialized on decode (sanitizer would flag, but
//!      callers without sanitizer enabled would silently read garbage).
//!    - Encoder treating the X byte as a 4th color channel (would
//!      change the encoded Y/Cb/Cr coefficients vs. the 3-channel path).
//!
//! 2. **4096² restart-every-MCU DoS bomb.** A pathological progressive
//!    JPEG with `restart_in_rows=1` at 4096×4096 has 256k restart
//!    markers, each forcing the decoder to re-init the Huffman / DC
//!    state.  Upstream documents this as a slowdown vector but bounds
//!    output size to the declared image dimensions.  Our test asserts:
//!    (a) decode terminates within a reasonable wall-clock budget, (b)
//!    output dimensions match the declared header dimensions (no
//!    runaway allocation past the declared pixel count), (c) the
//!    decoder does not panic.
//!
//! ## Why phase-2 (not phase-1)
//!
//! Phase-1 covered q∈{98,99,100} encode parity in
//! `tests/hard_case_high_quality_parity.rs`.  The two patterns here
//! were broken out because each requires its own fixture-generation
//! path:
//!   - X-byte: needs a 4-channel RGBA buffer and the
//!     `JCS_EXT_*` pixel formats from the C API + Rust mirror.
//!   - Restart bomb: needs an upstream encoder run with
//!     `-restart 1B` at 4096² so the restart density is correct.
//!
//! Tracked under P4-12 in `docs/last_mile/phase4.md`.

mod helpers;

use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

// ---------- helpers ----------

/// Synthesize a 32×32 RGB gradient — small enough to round-trip fast,
/// large enough that any per-channel divergence shows up in 1-byte
/// resolution diffs.
fn gradient_rgb_32x32() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(32 * 32 * 3);
    for y in 0..32u32 {
        for x in 0..32u32 {
            buf.push((x * 8) as u8);
            buf.push((y * 8) as u8);
            buf.push(((x + y) * 4) as u8);
        }
    }
    buf
}

/// Pack a 3-channel RGB buffer into RGBX with `x_byte` in slot 3.
fn pack_rgbx(rgb: &[u8], x_byte: u8) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity((rgb.len() / 3) * 4);
    for chunk in rgb.chunks_exact(3) {
        out.push(chunk[0]);
        out.push(chunk[1]);
        out.push(chunk[2]);
        out.push(x_byte);
    }
    out
}

/// Pack a 3-channel RGB buffer into BGRX with `x_byte` in slot 3.
fn pack_bgrx(rgb: &[u8], x_byte: u8) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity((rgb.len() / 3) * 4);
    for chunk in rgb.chunks_exact(3) {
        out.push(chunk[2]);
        out.push(chunk[1]);
        out.push(chunk[0]);
        out.push(x_byte);
    }
    out
}

// ---------- pattern A: JCS_EXT_RGBX / JCS_EXT_BGRX X-byte semantics ----------

/// Encode an RGBX buffer through the Rust encoder and verify the X byte
/// (slot 3) is ignored — the encoded JPEG must be byte-identical to the
/// encode that drops the X byte and supplies RGB only.
#[test]
fn rgbx_x_byte_ignored_on_encode() {
    let rgb: Vec<u8> = gradient_rgb_32x32();

    // Two RGBX inputs that share the same RGB payload but differ only
    // in the X slot. After encoding through the X-aware pixel format,
    // the resulting JPEG bytes MUST match — proving the X slot was
    // dropped on the ingest side.
    let rgbx_alpha_opaque: Vec<u8> = pack_rgbx(&rgb, 0xFF);
    let rgbx_alpha_zero: Vec<u8> = pack_rgbx(&rgb, 0x00);

    let jpeg_opaque: Vec<u8> = compress(
        &rgbx_alpha_opaque,
        32,
        32,
        PixelFormat::Rgbx,
        85,
        Subsampling::S444,
    )
    .expect("rust encode RGBX(0xFF)");
    let jpeg_zero: Vec<u8> = compress(
        &rgbx_alpha_zero,
        32,
        32,
        PixelFormat::Rgbx,
        85,
        Subsampling::S444,
    )
    .expect("rust encode RGBX(0x00)");

    assert_eq!(
        jpeg_opaque, jpeg_zero,
        "X-byte must not affect the encoded JPEG (RGBX 0xFF vs 0x00 diverged)"
    );

    // And the X-aware encode must match a separate RGB encode — proves
    // the X slot was dropped on ingest, not merely "consistent across X
    // values" (e.g. a bug that mixed both X values into Y identically
    // would also pass the first assertion).
    let jpeg_rgb_only: Vec<u8> =
        compress(&rgb, 32, 32, PixelFormat::Rgb, 85, Subsampling::S444).expect("rust encode RGB");
    assert_eq!(
        jpeg_opaque, jpeg_rgb_only,
        "RGBX encode (with any X) must match RGB-only encode"
    );
}

#[test]
fn bgrx_x_byte_ignored_on_encode() {
    let rgb: Vec<u8> = gradient_rgb_32x32();
    let bgrx_a: Vec<u8> = pack_bgrx(&rgb, 0xFF);
    let bgrx_b: Vec<u8> = pack_bgrx(&rgb, 0x33);

    let jpeg_a: Vec<u8> = compress(&bgrx_a, 32, 32, PixelFormat::Bgrx, 85, Subsampling::S444)
        .expect("rust encode BGRX(0xFF)");
    let jpeg_b: Vec<u8> = compress(&bgrx_b, 32, 32, PixelFormat::Bgrx, 85, Subsampling::S444)
        .expect("rust encode BGRX(0x33)");

    assert_eq!(
        jpeg_a, jpeg_b,
        "X-byte must not affect the encoded JPEG (BGRX 0xFF vs 0x33 diverged)"
    );
}

/// Decode through the Rust decoder asking for RGBX output and verify
/// the X byte (slot 3) is the documented constant `0xFF`. The upstream
/// contract says X is "padding"; in practice libjpeg-turbo writes 0xFF
/// because downstream consumers rely on RGBX-as-opaque-RGBA semantics.
#[test]
fn rgbx_x_byte_is_ff_on_decode() {
    // Encode an RGB checker via Rust → JPEG, then decode through
    // PixelFormat::Rgbx and inspect the X slot of every pixel.
    let rgb: Vec<u8> = gradient_rgb_32x32();
    let jpeg: Vec<u8> =
        compress(&rgb, 32, 32, PixelFormat::Rgb, 95, Subsampling::S444).expect("rust encode RGB");

    let decoded =
        libjpeg_turbo_rs::decompress_to(&jpeg, PixelFormat::Rgbx).expect("rust decode RGBX");
    assert_eq!(decoded.data.len(), 32 * 32 * 4, "RGBX decode size mismatch");

    // The X byte (slot 3 of every 4-byte pixel) must be 0xFF.
    for (i, px) in decoded.data.chunks_exact(4).enumerate() {
        assert_eq!(
            px[3], 0xFF,
            "pixel #{i}: X byte must be 0xFF on RGBX decode, got 0x{:02x}",
            px[3]
        );
    }
}

#[test]
fn bgrx_x_byte_is_ff_on_decode() {
    let rgb: Vec<u8> = gradient_rgb_32x32();
    let jpeg: Vec<u8> =
        compress(&rgb, 32, 32, PixelFormat::Rgb, 95, Subsampling::S444).expect("rust encode RGB");

    let decoded =
        libjpeg_turbo_rs::decompress_to(&jpeg, PixelFormat::Bgrx).expect("rust decode BGRX");
    assert_eq!(decoded.data.len(), 32 * 32 * 4, "BGRX decode size mismatch");

    for (i, px) in decoded.data.chunks_exact(4).enumerate() {
        assert_eq!(
            px[3], 0xFF,
            "pixel #{i}: X byte must be 0xFF on BGRX decode, got 0x{:02x}",
            px[3]
        );
    }
}

// ---------- pattern B: 4096² restart-every-MCU DoS bomb ----------

/// Build a 4096² grayscale JPEG with the maximum-density restart-marker
/// configuration (`-restart 1B` ≡ a restart marker after every MCU).
/// Use C `cjpeg` to construct the fixture — it's the canonical encoder
/// for this exact knob and matches what an attacker would produce.
fn build_restart_bomb(cjpeg: &Path, dst: &Path) -> bool {
    // Synthesize a 4096×4096 grayscale PPM (P5). A flat-value image is
    // adversarial in the right way: tiny encoded MCU bytes, maximum
    // restart-marker overhead per byte of actual data.
    let w: usize = 4096;
    let h: usize = 4096;
    let mut pgm: Vec<u8> = Vec::with_capacity(w * h + 32);
    pgm.extend_from_slice(format!("P5\n{w} {h}\n255\n").as_bytes());
    pgm.resize(pgm.len() + w * h, 0x80); // mid-gray

    let tmp: tempfile::TempDir = match tempfile::tempdir() {
        Ok(t) => t,
        Err(_) => return false,
    };
    let pgm_path: PathBuf = tmp.path().join("flat.pgm");
    if std::fs::write(&pgm_path, &pgm).is_err() {
        return false;
    }

    // -restart 1B: a restart marker after every MCU, the worst case.
    // For 4096² grayscale at 4:4:4 the MCU is 8×8 — 512×512 = 262144
    // MCUs and therefore (262144 - 1) restart markers in the stream.
    let out = Command::new(cjpeg)
        .args([
            "-grayscale",
            "-quality",
            "75",
            "-restart",
            "1B",
            "-outfile",
            dst.to_str().expect("dst path utf-8"),
            pgm_path.to_str().expect("pgm path utf-8"),
        ])
        .output();
    matches!(out, Ok(o) if o.status.success())
}

#[test]
fn restart_bomb_4096_terminates_within_budget() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let jpeg_path: PathBuf = tmp.path().join("bomb.jpg");
    if !build_restart_bomb(&cjpeg, &jpeg_path) {
        // P4-116: CI provisions libjpeg-turbo 3.x, so a missing
        // capability there is a provisioning defect, not a skip.
        assert!(
            !std::env::var("CI")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false),
            "CI must provide a cjpeg restart-bomb build-capable cjpeg"
        );
        eprintln!("SKIP restart_bomb_4096: cjpeg build failed");
        return;
    }
    let jpeg: Vec<u8> = std::fs::read(&jpeg_path).expect("read bomb.jpg");

    // 60s upper bound is generous: a Rust decoder matching C should
    // finish in well under 10s on the CI runners, but we want a clear
    // "this is a DoS bound, not a perf benchmark" signal.
    let budget: Duration = Duration::from_secs(60);
    let start: Instant = Instant::now();
    let decoded = decompress(&jpeg).expect("decode restart-every-MCU 4096²");
    let elapsed: Duration = start.elapsed();
    assert!(
        elapsed < budget,
        "restart-every-MCU 4096² took {:?}, exceeded budget {:?}",
        elapsed,
        budget
    );

    // Dimensions must match the declared header — no runaway allocation.
    assert_eq!(decoded.width as usize, 4096, "width mismatch");
    assert_eq!(decoded.height as usize, 4096, "height mismatch");
    // Grayscale → 1 channel, but the high-level decompress() expands to
    // RGB. Either output is fine; assert against actual length.
    let expected_rgb: usize = 4096 * 4096 * 3;
    let expected_gray: usize = 4096 * 4096;
    assert!(
        decoded.data.len() == expected_rgb || decoded.data.len() == expected_gray,
        "decoded length {} not RGB ({expected_rgb}) or grayscale ({expected_gray})",
        decoded.data.len()
    );
}

/// Cross-check: a 4096² restart-bomb decoded by C `djpeg` must produce
/// the *same* dimensions our decoder reports. This catches any
/// dimension-truncation regression at MCU-walk boundaries.
#[test]
fn restart_bomb_4096_dimensions_match_djpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let jpeg_path: PathBuf = tmp.path().join("bomb.jpg");
    if !build_restart_bomb(&cjpeg, &jpeg_path) {
        // P4-116: CI provisions libjpeg-turbo 3.x, so a missing
        // capability there is a provisioning defect, not a skip.
        assert!(
            !std::env::var("CI")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false),
            "CI must provide a cjpeg restart-bomb build-capable cjpeg"
        );
        eprintln!("SKIP restart_bomb_dim_match: cjpeg build failed");
        return;
    }

    // Ask djpeg for header-only info (`-fast`); the PPM/PGM size line
    // is the canonical dimensions.
    let c_pgm: PathBuf = tmp.path().join("c.pgm");
    let out = Command::new(&djpeg)
        .args([
            "-grayscale",
            "-fast",
            "-outfile",
            c_pgm.to_str().expect("c.pgm path utf-8"),
            jpeg_path.to_str().expect("bomb.jpg path utf-8"),
        ])
        .output()
        .expect("run djpeg");
    assert!(
        out.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Parse the PGM header — first three whitespace-separated tokens
    // after "P5\n" are width, height, maxval.
    let bytes: Vec<u8> = std::fs::read(&c_pgm).expect("read c.pgm");
    let text_prefix: &[u8] = &bytes[..bytes.len().min(64)];
    let header_str: String = String::from_utf8_lossy(text_prefix).into_owned();
    let mut tokens = header_str.split_ascii_whitespace();
    let magic: &str = tokens.next().unwrap_or("");
    let w: usize = tokens.next().unwrap_or("0").parse().unwrap_or(0);
    let h: usize = tokens.next().unwrap_or("0").parse().unwrap_or(0);
    assert_eq!(magic, "P5", "expected djpeg PGM magic");
    assert_eq!(w, 4096, "djpeg width mismatch");
    assert_eq!(h, 4096, "djpeg height mismatch");

    // Our decoder should match.
    let jpeg: Vec<u8> = std::fs::read(&jpeg_path).expect("read bomb.jpg");
    let decoded = decompress(&jpeg).expect("rust decode restart-bomb 4096²");
    assert_eq!(decoded.width as usize, w, "rust width != djpeg width");
    assert_eq!(decoded.height as usize, h, "rust height != djpeg height");
}
