//! P2-7 follow-up: differential encode-roundtrip-vs-C fuzzer.
//!
//! Encodes a fuzz-supplied pixel buffer via Rust, then verifies that
//! the resulting JPEG bytes decode equivalently through:
//!   1. our own Rust decoder, and
//!   2. a subprocessed C `djpeg`.
//!
//! Asserts:
//! 1. **C accepts our encoder output.** djpeg must exit 0 with non-empty
//!    PPM output; otherwise our encoder produced a JPEG no C consumer
//!    would accept (drop-in regression on the encode side).
//! 2. **Both decoders agree.** Pixel buffers must match within ±2 per
//!    byte (IDCT tolerance — the same threshold `fuzz_decode_diff_c`
//!    uses for the decode-side differential).
//!
//! The complementary `fuzz_decode_diff_c` target catches "C decoder
//! accepts but Rust decoder rejects/diverges". This target catches
//! "Rust encoder produces output C decoder rejects/diverges" — the
//! mirror image.
//!
//! Subprocesses djpeg per input, same rationale as fuzz_decode_diff_c
//! (slower but avoids dragging cc-rs + system libjpeg into the fuzz
//! crate). In-process FFI is tracked as a follow-up.

#![no_main]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_progressive, Decoder,
    PixelFormat, Subsampling,
};

// Smaller than fuzz_decode_diff_c's cap because we encode-then-decode-twice
// per input. 256 × 256 × 3 = 192 KB pixel budget keeps each iteration well
// under the libfuzzer per-iteration timeout.
const MAX_DIM: usize = 96;
const PIXEL_TOLERANCE: i32 = 2;
const HEADER_LEN: usize = 4;

/// One-shot SIGPIPE masking — see fuzz_decode_diff_c::ensure_sigpipe_ignored
/// for rationale.
fn ensure_sigpipe_ignored() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    });
}

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
    // Codex P2: drain stdout via a writer thread to avoid pipe-buffer
    // deadlock when the decoded PNM exceeds the OS pipe capacity.
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

/// Returns (width, height, channels, raw_pixels) for P5/P6 PNM output.
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
        _ => return None,
    };
    Some((img.width, img.height, channels, img.data))
}

fn subsampling_for(idx: u8) -> Subsampling {
    match idx % 4 {
        0 => Subsampling::S420,
        1 => Subsampling::S422,
        2 => Subsampling::S444,
        _ => Subsampling::S440,
    }
}

fuzz_target!(|data: &[u8]| {
    ensure_sigpipe_ignored();
    let Some(djpeg) = djpeg_path() else {
        return;
    };

    if data.len() < HEADER_LEN {
        return;
    }

    let width: usize = (data[0] as usize % MAX_DIM).max(1);
    let height: usize = (data[1] as usize % MAX_DIM).max(1);
    let quality: u8 = ((data[2] as u32 % 100) as u8).max(1);
    let flags: u8 = data[3];

    let sub_idx: u8 = flags & 0b0000_0011;
    let entropy: u8 = (flags >> 2) & 0b11;
    let grayscale: bool = (flags & 0b1000_0000) != 0;

    let subsampling: Subsampling = subsampling_for(sub_idx);
    let pf: PixelFormat = if grayscale {
        PixelFormat::Grayscale
    } else {
        PixelFormat::Rgb
    };
    let bpp: usize = pf.bytes_per_pixel();

    let required: usize = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(bpp))
        .unwrap_or(usize::MAX);
    if data.len() < HEADER_LEN + required || required == 0 {
        return;
    }
    let pixels: &[u8] = &data[HEADER_LEN..HEADER_LEN + required];

    // Encode via Rust. Err → invalid configuration combo, just bail.
    let encoded: Vec<u8> = match entropy {
        0 => compress(pixels, width, height, pf, quality, subsampling),
        1 => compress_progressive(pixels, width, height, pf, quality, subsampling),
        2 => compress_arithmetic(pixels, width, height, pf, quality, subsampling),
        _ => compress_arithmetic_progressive(pixels, width, height, pf, quality, subsampling),
    }
    .ok()
    .filter(|b| !b.is_empty())
    .unwrap_or_default();
    if encoded.is_empty() {
        return;
    }

    let r_result = rust_decode(&encoded);
    let c_result = decode_with_djpeg(&djpeg, &encoded);

    match (r_result, c_result) {
        (Some((rw, rh, rc, r_px)), Some((cw, ch, cc, c_px))) => {
            // Dimension agreement.
            if cw != rw || ch != rh || cc != rc {
                panic!(
                    "encode-roundtrip: decoded dims diverge: \
                     C={}x{}x{}, Rust={}x{}x{}, encoded len={}",
                    cw,
                    ch,
                    cc,
                    rw,
                    rh,
                    rc,
                    encoded.len()
                );
            }
            // Pixel agreement within IDCT tolerance.
            let mut max_d: i32 = 0;
            for (a, b) in c_px.iter().zip(r_px.iter()) {
                let d: i32 = (*a as i32 - *b as i32).abs();
                if d > max_d {
                    max_d = d;
                }
            }
            if max_d > PIXEL_TOLERANCE {
                panic!(
                    "encode-roundtrip: decoded pixels diverge: max abs diff = {} \
                     (tolerance {}); dims {}x{}x{}; first 16 c=[{:?}] r=[{:?}]",
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
        (None, _) => {
            // Rust failed to decode its own encoder output — this is a
            // self-consistency bug in the encode/decode pipeline.
            panic!(
                "encode-roundtrip: Rust decoder rejected our own encoded JPEG \
                 (w={width} h={height} q={quality} entropy={entropy} \
                 subsampling={subsampling:?} encoded_len={})",
                encoded.len()
            );
        }
        (Some(_), None) => {
            // Rust accepted but C rejected — our encoder produced output
            // outside the libjpeg-turbo acceptance envelope. Drop-in
            // regression on the encode side.
            panic!(
                "encode-roundtrip: C djpeg rejected our encoded JPEG \
                 (w={width} h={height} q={quality} entropy={entropy} \
                 subsampling={subsampling:?} encoded_len={})",
                encoded.len()
            );
        }
    }
});
