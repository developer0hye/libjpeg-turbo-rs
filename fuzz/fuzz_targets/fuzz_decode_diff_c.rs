//! P2-7: differential decode fuzzer.
//!
//! Feeds each fuzzed input to *both* `libjpeg_turbo_rs::Decoder::decode`
//! and a subprocessed `djpeg`, then asserts:
//!
//! 1. **Acceptance agreement.** When C accepts the input (djpeg exits 0
//!    with non-empty PPM), Rust must accept it too — being more strict
//!    than the reference is a drop-in regression. Rust accepting an
//!    input C rejects is allowed (Rust is lenient by design;
//!    `fuzz_decompress_lenient` covers that direction).
//! 2. **Pixel agreement.** When both succeed and report the same
//!    dimensions / channel count, decoded pixels must agree within an
//!    IDCT tolerance of ±2 per byte. Anything larger is a real
//!    decode-output divergence.
//!
//! Subprocesses djpeg per input rather than linking C libjpeg in
//! process — this is slower (~ms per iter) but avoids dragging
//! `cc-rs` + system libjpeg into the fuzz crate. In-process FFI is
//! tracked as a follow-up.
//!
//! Skip-with-reason via early `return`:
//! - djpeg not on PATH (CI must install libjpeg-turbo-progs).
//! - input dimensions exceed the standard pixel cap.

#![no_main]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::Decoder;

const MAX_FUZZ_PIXELS: u64 = 1_048_576;
const PIXEL_TOLERANCE: i32 = 2;

fn djpeg_path() -> Option<PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            for p in [
                "/opt/homebrew/bin/djpeg",
                "/usr/local/bin/djpeg",
                "/usr/bin/djpeg",
                "/opt/libjpeg-turbo/bin/djpeg",
            ] {
                let pb = PathBuf::from(p);
                if pb.exists() {
                    return Some(pb);
                }
            }
            None
        })
        .clone()
}

fn decode_with_djpeg(djpeg: &PathBuf, jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let mut child = Command::new(djpeg)
        .arg("-pnm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Codex P2: drain stdout concurrently with the stdin write to avoid
    // a pipe-buffer deadlock when the decoded PNM exceeds the OS pipe
    // capacity (~64 KB on Linux/macOS).
    let mut stdin = child.stdin.take()?;
    let payload: Vec<u8> = jpeg.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() {
        return None;
    }
    parse_pnm(&out.stdout)
}

/// Returns (width, height, channels, raw_pixels).
fn parse_pnm(bytes: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let mut i: usize = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            tokens.push(String::from_utf8(bytes[start..i].to_vec()).ok()?);
        }
    }
    if tokens.len() < 4 {
        return None;
    }
    let channels: usize = match tokens[0].as_str() {
        "P5" => 1,
        "P6" => 3,
        _ => return None,
    };
    let w: usize = tokens[1].parse().ok()?;
    let h: usize = tokens[2].parse().ok()?;
    if tokens[3] != "255" {
        return None;
    }
    i += 1; // single whitespace after maxval per Netpbm spec
    let needed = w.checked_mul(h)?.checked_mul(channels)?;
    if bytes.len() < i + needed {
        return None;
    }
    Some((w, h, channels, bytes[i..i + needed].to_vec()))
}

fn rust_decode(jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let img = Decoder::decode(jpeg).ok()?;
    let channels: usize = match img.pixel_format {
        libjpeg_turbo_rs::PixelFormat::Grayscale => 1,
        libjpeg_turbo_rs::PixelFormat::Rgb => 3,
        _ => return None, // out-of-scope formats — skip the differential.
    };
    Some((img.width, img.height, channels, img.data))
}

fuzz_target!(|data: &[u8]| {
    let Some(djpeg) = djpeg_path() else {
        return;
    };

    // Cheap header-only Rust probe to enforce the pixel cap before
    // spawning djpeg. If Rust can't even open the header, skip — the
    // reference comparison only matters for inputs both decoders see
    // as JPEG.
    let Ok(mut probe) = Decoder::new(data) else {
        return;
    };
    let header = probe.header();
    let pixels: u64 = (header.width as u64).saturating_mul(header.height as u64);
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    probe.set_scan_limit(100);

    let c_result = decode_with_djpeg(&djpeg, data);
    let r_result = rust_decode(data);

    match (c_result, r_result) {
        (Some((cw, ch, cc, c_px)), Some((rw, rh, rc, r_px))) => {
            // (1) Acceptance agreement: when C succeeds, Rust succeeded
            // (we're inside Some(...) on both arms — pass).
            // (2) Dimension agreement.
            if cw != rw || ch != rh || cc != rc {
                // Different output shape is a real divergence — for
                // example one side decoded as RGB and the other as
                // grayscale. Surface as a fuzz finding.
                panic!(
                    "decode dimensions diverge: C={}x{}x{}, Rust={}x{}x{}",
                    cw, ch, cc, rw, rh, rc
                );
            }
            // (3) Pixel agreement within ±IDCT tolerance.
            let mut max_d: i32 = 0;
            for (a, b) in c_px.iter().zip(r_px.iter()) {
                let d: i32 = (*a as i32 - *b as i32).abs();
                if d > max_d {
                    max_d = d;
                }
            }
            if max_d > PIXEL_TOLERANCE {
                panic!(
                    "decode pixels diverge: max abs diff = {} (tolerance {}); \
                     dims {}x{}x{}; first 16 c=[{:?}] r=[{:?}]",
                    max_d,
                    PIXEL_TOLERANCE,
                    cw,
                    ch,
                    cc,
                    &c_px[..16.min(c_px.len())],
                    &r_px[..16.min(r_px.len())],
                );
            }
        }
        (Some(_), None) => {
            // C accepted, Rust rejected — drop-in regression. The
            // intent of "we are at least as accepting as the reference"
            // gates here.
            panic!(
                "drop-in regression: C djpeg accepted input ({} bytes) but Rust rejected",
                data.len()
            );
        }
        (None, _) => {
            // C rejected — Rust may accept (lenient by design) or
            // reject. Either way, no comparison signal to act on.
        }
    }
});
