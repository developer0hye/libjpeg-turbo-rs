//! P2-7 follow-up: differential encode-roundtrip-vs-C fuzzer.
//!
//! Encodes a fuzz-supplied pixel buffer via Rust, then verifies that
//! the resulting JPEG bytes decode equivalently through:
//!   1. our own Rust decoder, and
//!   2. a subprocessed C `djpeg`.
//!
//! Asserts:
//! 1. **We produce the bytes C produces.** The same pixels encoded through
//!    stock `cjpeg`, with flags matching our configuration, must be
//!    byte-identical. This is the *reference* oracle — it is the only one
//!    of the three that can see a conformance defect.
//! 2. **C accepts our encoder output.** djpeg must exit 0 with non-empty
//!    PPM output; otherwise our encoder produced a JPEG no C consumer
//!    would accept (drop-in regression on the encode side).
//! 3. **Both decoders agree.** Pixel buffers must match within ±2 per
//!    byte (IDCT tolerance — the same threshold `fuzz_decode_diff_c`
//!    uses for the decode-side differential).
//!
//! Assertions 2 and 3 are *validity* oracles and were, for a long time, the
//! only ones here. They cannot distinguish "correct output" from "a different
//! but equally valid JPEG", so they stayed green through #314 — 4:2:0 encoding
//! wrong for every width with `ceil(width/8)` odd — even though this target's
//! own geometry range covers 48 such widths. Assertion 1 was added for exactly
//! that class.
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

/// Locates a stock libjpeg-turbo tool.
///
/// `LIBJPEG_TURBO_BIN` overrides the search when the C tools live outside the
/// standard prefixes — without it a developer box with a source-built
/// toolchain silently runs this target with **no C oracle at all**, which is
/// the same "green because it checked nothing" failure the reference oracle
/// exists to prevent.
fn c_tool_path(name: &str) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("LIBJPEG_TURBO_BIN") {
        let pb = PathBuf::from(dir).join(name);
        if pb.exists() {
            return Some(pb);
        }
    }
    for prefix in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/libjpeg-turbo/bin",
        "/usr/bin",
    ] {
        let pb = PathBuf::from(prefix).join(name);
        if pb.exists() {
            return Some(pb);
        }
    }
    None
}

fn djpeg_path() -> Option<PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE.get_or_init(|| c_tool_path("djpeg")).clone()
}

fn cjpeg_path() -> Option<PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE.get_or_init(|| c_tool_path("cjpeg")).clone()
}

/// Encodes the same pixels through stock `cjpeg` with flags chosen to match
/// our encoder's configuration exactly, returning the JPEG bytes.
///
/// `-baseline` is passed for **every** mode, not just baseline: in cjpeg it
/// controls `force_baseline` (clamping scaled quantization values to 255),
/// which is orthogonal to the entropy mode and is what our
/// `quality_scale_quant_table` does unconditionally. Omitting it makes C use
/// unclamped 16-bit quant values at quality <= 20 and the comparison diverges
/// for reasons that are a flag mismatch, not an encoder defect.
fn encode_with_cjpeg(
    cjpeg: &PathBuf,
    pixels: &[u8],
    width: usize,
    height: usize,
    grayscale: bool,
    quality: u8,
    sample: &str,
    entropy: u8,
) -> Option<Vec<u8>> {
    let magic: &str = if grayscale { "P5" } else { "P6" };
    let mut pnm: Vec<u8> = format!("{magic}\n{width} {height}\n255\n").into_bytes();
    pnm.extend_from_slice(pixels);

    let quality_arg: String = quality.to_string();
    let mut args: Vec<&str> = vec!["-quality", &quality_arg, "-dct", "int", "-baseline"];
    match entropy {
        0 => {}
        1 => args.push("-progressive"),
        2 => args.push("-arithmetic"),
        _ => {
            args.push("-arithmetic");
            args.push("-progressive");
        }
    }
    if grayscale {
        args.push("-grayscale");
    } else {
        args.push("-sample");
        args.push(sample);
    }

    let mut child = Command::new(cjpeg)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&pnm);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    Some(out.stdout)
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
    // Both C tools are required: without cjpeg this target degrades to a
    // roundtrip check, which `fuzz_encode_roundtrip` already provides.
    let Some(cjpeg) = cjpeg_path() else {
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

    // --- Reference oracle: are we producing the bytes C would produce? ---
    //
    // The roundtrip assertions below are *validity* oracles: they check that C
    // accepts our output and that both decoders read it the same way. Both
    // stay green when our encoder emits a perfectly valid JPEG that simply is
    // not the one cjpeg would have emitted, which is how #314 (4:2:0 wrong for
    // every width with ceil(width/8) odd) and #316 survived in a target whose
    // geometry range covered them 48 times over. Byte-equality against cjpeg is
    // measured to be the contract across all four entropy modes, both
    // colourspaces, all subsamplings and qualities 1-100, so it is checked
    // unconditionally rather than gated.
    let sample: &str = match subsampling {
        Subsampling::S444 => "1x1",
        Subsampling::S422 => "2x1",
        Subsampling::S420 => "2x2",
        _ => "1x2",
    };
    if let Some(c_encoded) = encode_with_cjpeg(
        &cjpeg,
        pixels,
        width,
        height,
        grayscale,
        quality,
        sample,
        entropy,
    ) {
        if c_encoded != encoded {
            let first_diff: usize = encoded
                .iter()
                .zip(c_encoded.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(encoded.len().min(c_encoded.len()));
            panic!(
                "encode-vs-cjpeg: byte divergence at offset {first_diff} \
                 (rust={} bytes, c={} bytes); w={width} h={height} q={quality} \
                 entropy={entropy} subsampling={subsampling:?} grayscale={grayscale} \
                 sample={sample}",
                encoded.len(),
                c_encoded.len(),
            );
        }
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
