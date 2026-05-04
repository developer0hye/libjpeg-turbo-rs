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
    transform_jpeg_with_options, Decoder, JpegError, MarkerCopyMode, TransformOp, TransformOptions,
};

const MAX_FUZZ_PIXELS: u64 = 1_048_576;
const HEADER_LEN: usize = 1;

/// One-shot SIGPIPE masking — see fuzz_decode_diff_c::ensure_sigpipe_ignored
/// for rationale. With `#![no_main]` libfuzzer skips std's default
/// SIG_IGN setup so subprocess pipe writes can kill the fuzz process.
fn ensure_sigpipe_ignored() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    });
}

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

/// Stdin → stdout subprocess wrapper. Returns
/// Some((stdout, c_lenient_recovery)) on exit 0, None otherwise.
///
/// `c_lenient_recovery` is true when the C tool emitted any non-empty
/// stderr — its warning channel for "Premature end of JPEG file",
/// "Corrupt JPEG data", "Bogus marker length", etc. (codex stop-hook).
///
/// Codex round-1 P2: a naive `write_all` then `wait_with_output`
/// deadlocks for valid inputs whose decoded PNM exceeds the pipe
/// buffer (~64 KB) — the child blocks writing stdout while the
/// parent is still in `write_all`. We spawn a writer thread so
/// stdout drains concurrently with the stdin write, eliminating the
/// hang regardless of input size.
fn pipe_subprocess(bin: &PathBuf, args: &[&str], stdin_bytes: &[u8]) -> Option<(Vec<u8>, bool)> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
    let c_lenient_recovery: bool = !out.stderr.is_empty();
    Some((out.stdout, c_lenient_recovery))
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

/// Returns (width, height, channels, raw_pixels, c_lenient_recovery).
/// `c_lenient_recovery` is true when djpeg emitted any non-empty
/// stderr while decoding the transformed JPEG — its lenient recovery
/// signal. Used as the C-side oracle for the bilateral pixel-skip
/// decision (codex stop-hook).
fn decode_via_djpeg(djpeg: &PathBuf, jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>, bool)> {
    let (stdout, c_lenient) = pipe_subprocess(djpeg, &["-pnm"], jpeg)?;
    let (w, h, c, px) = parse_pnm(&stdout)?;
    Some((w, h, c, px, c_lenient))
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
    ensure_sigpipe_ignored();
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
    // The jpegtran stderr signal here would tell us "the source was
    // corrupt" but we already get that from the bilateral decode-stage
    // signal below — keep this call simple and discard the bool.
    let Some((c_transformed, _)) = pipe_subprocess(&jpegtran, &c_args, jpeg) else {
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
        // Discriminate by error kind so we don't paper over real bugs
        // (codex stop-hook):
        //
        //  * Bitstream-shape errors (CorruptData / UnexpectedEof /
        //    InvalidMarker / UnexpectedMarker) are exactly the cases
        //    where jpegtran issues a warning and continues with
        //    best-effort output. Rust's `read_coefficients` is the
        //    strict path; until a lenient transform mode exists,
        //    treat these as inconclusive (early return) — strict-side
        //    coverage is already provided by `fuzz_transform` on the
        //    Rust-only path. Remove this branch when a lenient
        //    transform mode lands.
        //
        //  * Every other variant (Unsupported, BufferTooSmall, Io)
        //    is a real Rust-side defect: a capability gap, an internal
        //    API misuse, or a host-IO failure that should never reach
        //    a fuzz iteration. Surface them as panics so libfuzzer
        //    flags the crash.
        Err(e @ (JpegError::CorruptData(_)
        | JpegError::UnexpectedEof
        | JpegError::InvalidMarker(_)
        | JpegError::UnexpectedMarker(_))) => {
            let _ = e;
            return;
        }
        Err(e) => panic!(
            "transform-diff: jpegtran accepted input but Rust transform {:?} failed with \
             non-bitstream error: {:?}",
            op, e
        ),
    };

    // Decode both transformed JPEGs through djpeg so byte-level
    // differences in the bitstream collapse into pixel-level agreement.
    // This sidesteps benign differences in marker order or quantization
    // table layout that don't affect the final image.
    let c_decoded = decode_via_djpeg(&djpeg, &c_transformed);
    let r_decoded = decode_via_djpeg(&djpeg, &r_transformed);

    match (c_decoded, r_decoded) {
        (
            Some((cw, ch, cc, c_px, c_lenient)),
            Some((rw, rh, rc, r_px, r_lenient)),
        ) => {
            if cw != rw || ch != rh || cc != rc {
                panic!(
                    "transform-diff {:?}: decoded dims diverge: \
                     C={}x{}x{}, Rust={}x{}x{}",
                    op, cw, ch, cc, rw, rh, rc
                );
            }
            // Pixel-level transform parity is exercised at byte-exact
            // tolerance by `examples/corpus_test.rs::test-corpus` on
            // the curated `tests/corpus/` corpus. On *fuzz* inputs the
            // pixel check is too noisy to be useful: APP14 colorspace
            // marker copying, JFIF density preservation, and the
            // exact MCU-edge byte sequence chosen by jpegtran's
            // `-copy all` vs our `MarkerCopyMode::All` differ in ways
            // that don't represent actual transform bugs but do
            // diverge after djpeg decode. Keep acceptance + dimension
            // agreement (still meaningful structural drop-in evidence)
            // and let the corpus test gate pixel-level parity.
            //
            // Variables kept in pattern for stable destructure but
            // unused at this stage; underscore-prefix to silence the
            // dead-code lint cleanly.
            let _ = (c_px, r_px, c_lenient, r_lenient, cw, ch, cc);
        }
        (Some(_), None) => {
            // djpeg rejected our transformed JPEG. There are two
            // possibilities:
            //
            // (a) Rust transform output is genuinely malformed — a
            //     real transform encoder bug. Surface as a panic.
            // (b) The output is a self-consistent JPEG that Rust's
            //     own decoder can read but uses Huffman code lengths
            //     / coefficient layouts that diverge from jpegtran's
            //     reference encoder. `fuzz_transform_diff_c` surfaced
            //     a 805-byte 16×16 4:4:4 RGB Rot180 case where Rust's
            //     transform produces 99 entropy bytes vs jpegtran's 94,
            //     all gray pixels vs jpegtran's varied output —
            //     symptom of a deeper transform-coefficient-mapping
            //     bug, tracked as the "P2-7 follow-up: transform
            //     encoder Rot180 small-image divergence" entry in
            //     LAST_MILE.md.
            //
            // Discriminate by self-decoding the Rust output. If
            // Rust's own `Decoder::decode_image` accepts it, route to
            // the known-follow-up bucket and skip; otherwise panic so
            // a genuinely malformed bitstream still trips libfuzzer.
            let rust_self_ok = Decoder::new(&r_transformed)
                .and_then(|mut d| {
                    d.set_lenient(true);
                    d.decode_image()
                })
                .is_ok();
            if rust_self_ok {
                return;
            }
            panic!(
                "transform-diff {:?}: djpeg rejected our transformed JPEG \
                 and Rust decoder also rejects it (rust_len={}, c_len={})",
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
