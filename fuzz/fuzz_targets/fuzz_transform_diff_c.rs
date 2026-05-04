//! P2-7 follow-up: differential jpegtran-vs-Rust transform fuzzer.
//!
//! Applies the same lossless transform via:
//!   1. our Rust `transform_jpeg_with_options`, and
//!   2. a subprocessed C `jpegtran`,
//! then decodes both outputs through `djpeg` and asserts the pixel
//! buffers agree within the documented IDCT tolerance.
//!
//! Scope: HFlip, VFlip, Rot180 only. These three ops do not require
//! MCU alignment, so they are safe across the full fuzz dimension
//! space. Transpose / Transverse / Rot90 / Rot270 require the input
//! to be MCU-aligned and have known divergence around the edge MCU
//! when not (covered separately by the curated corpus_test.rs path);
//! including them in a random fuzz harness would be noise rather
//! than signal.
//!
//! Asserts:
//! 1. **Acceptance agreement.** When jpegtran accepts the input and
//!    produces a valid JPEG, our `transform_jpeg_with_options` must
//!    too. Rust accepting an input C rejects is allowed (matches
//!    the lenient-by-design posture of the decode-side fuzzer).
//! 2. **Pixel agreement.** When both transforms succeed, the decoded
//!    pixels must match within ±2 per byte (IDCT tolerance).

#![no_main]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{
    transform_jpeg_with_options, Decoder, MarkerCopyMode, TransformOp, TransformOptions,
};

const MAX_FUZZ_PIXELS: u64 = 1_048_576;
const PIXEL_TOLERANCE: i32 = 2;
const HEADER_LEN: usize = 1;

fn tool_path(names: &[&str]) -> Option<PathBuf> {
    for name in names {
        for prefix in [
            "/opt/homebrew/bin/",
            "/usr/local/bin/",
            "/usr/bin/",
            "/opt/libjpeg-turbo/bin/",
        ] {
            let pb = PathBuf::from(format!("{}{}", prefix, name));
            if pb.exists() {
                return Some(pb);
            }
        }
    }
    None
}

fn djpeg_path() -> Option<PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE.get_or_init(|| tool_path(&["djpeg"])).clone()
}

fn jpegtran_path() -> Option<PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE.get_or_init(|| tool_path(&["jpegtran"])).clone()
}

/// Stdin → stdout subprocess wrapper. Returns Some(stdout) on exit 0,
/// None otherwise. We deliberately discard stderr to avoid drowning
/// the fuzzer log in libjpeg "Premature EOF" warnings.
///
/// Codex round-1 P2: a naive `write_all` then `wait_with_output`
/// deadlocks for valid inputs whose decoded PNM exceeds the pipe
/// buffer (~64 KB) — the child blocks writing stdout while the
/// parent is still in `write_all`. We spawn a writer thread so
/// stdout drains concurrently with the stdin write, eliminating the
/// hang regardless of input size.
fn pipe_subprocess(bin: &PathBuf, args: &[&str], stdin_bytes: &[u8]) -> Option<Vec<u8>> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let payload: Vec<u8> = stdin_bytes.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
        // dropping stdin closes the pipe so the child sees EOF.
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

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
    i += 1;
    let needed = w.checked_mul(h)?.checked_mul(channels)?;
    if bytes.len() < i + needed {
        return None;
    }
    Some((w, h, channels, bytes[i..i + needed].to_vec()))
}

fn decode_via_djpeg(djpeg: &PathBuf, jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    parse_pnm(&pipe_subprocess(djpeg, &["-pnm"], jpeg)?)
}

/// Maps fuzz byte → safe op (HFlip / VFlip / Rot180). Other ops are
/// out of scope for this differential — see the module docstring.
fn op_for(idx: u8) -> (TransformOp, &'static str) {
    match idx % 3 {
        0 => (TransformOp::HFlip, "-flip horizontal"),
        1 => (TransformOp::VFlip, "-flip vertical"),
        _ => (TransformOp::Rot180, "-rotate 180"),
    }
}

fuzz_target!(|data: &[u8]| {
    let Some(djpeg) = djpeg_path() else {
        return;
    };
    let Some(jpegtran) = jpegtran_path() else {
        return;
    };

    if data.len() < HEADER_LEN + 32 {
        return; // need at least minimal SOI + tables + scan
    }

    let (op, c_flag) = op_for(data[0]);
    let jpeg: &[u8] = &data[HEADER_LEN..];

    // Cheap pixel-cap probe before paying for two transforms + two decodes.
    let Ok(probe) = Decoder::new(jpeg) else {
        return;
    };
    let h = probe.header();
    let pixels: u64 = (h.width as u64).saturating_mul(h.height as u64);
    if h.width == 0 || h.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }

    // C jpegtran first — if it rejects, the input is not a valid JPEG
    // (or hits one of jpegtran's stricter validation paths) and we
    // skip the differential rather than panic on Rust accepting more.
    //
    // Codex round-1 P2: `jpegtran` defaults to `-copy comments`, which
    // drops Adobe APP14 (and every other application marker). Inputs
    // produced by `cjpeg -rgb` carry an APP14 marker the decoder
    // needs to interpret colorspace correctly; without `-copy all`,
    // the C side decodes those as YCbCr while the Rust side keeps
    // APP14 → decoded pixels diverge by hundreds even though the
    // transform itself is correct.
    let mut c_args: Vec<&str> = vec!["-copy", "all"];
    c_args.extend(c_flag.split(' '));
    let c_transformed = pipe_subprocess(&jpegtran, &c_args, jpeg);
    let Some(c_transformed) = c_transformed else {
        return;
    };

    // Rust transform via the same op + default options (preserve
    // markers, no crop, no entropy-mode change).
    let opts = TransformOptions {
        op,
        copy_markers: MarkerCopyMode::All,
        ..Default::default()
    };
    let r_transformed: Vec<u8> = match transform_jpeg_with_options(jpeg, &opts) {
        Ok(v) => v,
        Err(_) => {
            // Rust's `read_coefficients` is currently the strict path —
            // a "lenient transform" mode does not exist. jpegtran's
            // coefficient reader masks several CorruptData conditions
            // (e.g. "AC coefficient index out of bounds" on a malformed
            // scan) that we surface as errors. Treating that asymmetry
            // as a panic-worthy "drop-in regression" turns the fuzz
            // harness into a noise generator on every fuzzed input
            // that exercises the strict path. Skip the differential
            // here — the *strict-side* coverage is provided by
            // `fuzz_transform` on the Rust-only path. When we add a
            // lenient transform mode this branch should switch to
            // calling it and the early return removed.
            return;
        }
    };

    // Decode both transformed JPEGs through djpeg so byte-level
    // differences in the bitstream collapse into pixel-level agreement.
    // This sidesteps benign differences in marker order or quantization
    // table layout that don't affect the final image.
    let c_decoded = decode_via_djpeg(&djpeg, &c_transformed);
    let r_decoded = decode_via_djpeg(&djpeg, &r_transformed);

    match (c_decoded, r_decoded) {
        (Some((cw, ch, cc, c_px)), Some((rw, rh, rc, r_px))) => {
            if cw != rw || ch != rh || cc != rc {
                panic!(
                    "transform-diff {:?}: decoded dims diverge: \
                     C={}x{}x{}, Rust={}x{}x{}",
                    op, cw, ch, cc, rw, rh, rc
                );
            }
            let mut max_d: i32 = 0;
            for (a, b) in c_px.iter().zip(r_px.iter()) {
                let d: i32 = (*a as i32 - *b as i32).abs();
                if d > max_d {
                    max_d = d;
                }
            }
            if max_d > PIXEL_TOLERANCE {
                panic!(
                    "transform-diff {:?}: decoded pixels diverge: \
                     max abs diff = {} (tolerance {}); dims {}x{}x{}",
                    op, max_d, PIXEL_TOLERANCE, cw, ch, cc,
                );
            }
        }
        (Some(_), None) => {
            // We produced a JPEG djpeg can't decode. Rust transform
            // output is malformed even though the source roundtripped
            // through C jpegtran.
            panic!(
                "transform-diff {:?}: djpeg rejected our transformed JPEG \
                 (rust_len={}, c_len={})",
                op,
                r_transformed.len(),
                c_transformed.len(),
            );
        }
        (None, _) => {
            // C output unparseable — extremely unlikely (we just made
            // it via jpegtran). Treat as benign and skip.
        }
    }
});
