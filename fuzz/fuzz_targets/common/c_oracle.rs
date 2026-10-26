//! Shared plumbing for the differential targets that compare against a C
//! oracle program — `examples/*_c_oracle.c`, built against a libjpeg-turbo
//! 3.x development install by `fuzz/build_c_oracles.sh`.
//!
//! Each oracle reads the JPEG from `/dev/stdin`, exits non-zero when libjpeg
//! refuses it, and writes libjpeg's first warning (if any) to stderr.

#![allow(dead_code)] // each target uses a subset

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// One-shot SIGPIPE masking. With `#![no_main]` libfuzzer skips std's
/// default SIG_IGN setup, so a pipe write to an oracle that exited early
/// would kill the fuzz process with signal 13 instead of returning
/// `Err(BrokenPipe)` (same reason as `fuzz_decode_diff_c`).
pub fn ensure_sigpipe_ignored() {
    static ONCE: OnceLock<()> = OnceLock::new();
    // SAFETY: installing SIG_IGN for SIGPIPE has no handler code to run
    // and is done once, before any oracle pipe is written.
    ONCE.get_or_init(|| unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    });
}

/// The oracle binary named by the environment variable `env_var`.
///
/// A differential target without its reference compares nothing, so a
/// missing oracle stops the run with an explanation rather than letting it
/// finish "clean". (`fuzz_decode_diff_c` skips silently when `djpeg` is
/// absent; these oracles are never on a system path, so a skip would only
/// ever mean a misconfigured run.)
pub fn oracle_path(env_var: &'static str) -> PathBuf {
    match std::env::var_os(env_var).map(PathBuf::from) {
        Some(path) if path.is_file() => path,
        _ => {
            eprintln!(
                "{env_var} must name the C oracle built by fuzz/build_c_oracles.sh \
                 (this target compares against it and cannot run without it)"
            );
            std::process::exit(2);
        }
    }
}

/// What libjpeg produced for one input.
pub struct OracleRun {
    pub stdout: Vec<u8>,
    /// libjpeg emitted a warning: it recovered from corrupt data, so its
    /// output is one recovery strategy among several, not a reference.
    pub warned: bool,
}

/// Run `oracle` with `args` (the first is normally `/dev/stdin`), feeding
/// `jpeg` on stdin. `None` when libjpeg refused the input: exit status 1,
/// which is what libjpeg's `error_exit` and the oracles' own failure paths
/// return. An oracle that cannot be started, exits with any other status
/// (2 is a usage error) or dies on a signal panics instead: treating that as
/// a refusal would let a broken oracle compare nothing.
pub fn run_oracle(oracle: &Path, args: &[String], jpeg: &[u8]) -> Option<OracleRun> {
    let mut child = Command::new(oracle)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("cannot run the C oracle {oracle:?}: {error}"));
    // Write on a thread while wait_with_output drains stdout and stderr, so
    // a decoded image larger than the pipe buffer cannot deadlock the pair.
    let mut stdin = child.stdin.take().expect("oracle stdin is piped");
    let payload: Vec<u8> = jpeg.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let output = child
        .wait_with_output()
        .unwrap_or_else(|error| panic!("cannot wait for the C oracle {oracle:?}: {error}"));
    let _ = writer.join();
    match output.status.code() {
        Some(0) => {}
        Some(1) => return None,
        _ => panic!(
            "the C oracle {oracle:?} failed abnormally ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ),
    }
    Some(OracleRun {
        stdout: output.stdout,
        warned: !output.stderr.is_empty(),
    })
}

/// Split a `<text header>\n<binary body>` oracle output.
pub fn split_header(stdout: &[u8]) -> Option<(Vec<&str>, &[u8])> {
    let newline: usize = stdout.iter().position(|&b| b == b'\n')?;
    let header: &str = std::str::from_utf8(&stdout[..newline]).ok()?;
    Some((header.split_whitespace().collect(), &stdout[newline + 1..]))
}

/// `LTRS_FUZZ_INCLUDE_KNOWN=1` stops a target skipping the divergences
/// already filed (each skip names its LAST_MILE item). Use it to confirm a
/// target still detects them, or that a fix closed them.
pub fn include_known_gaps() -> bool {
    static INCLUDE: OnceLock<bool> = OnceLock::new();
    *INCLUDE.get_or_init(|| std::env::var("LTRS_FUZZ_INCLUDE_KNOWN").as_deref() == Ok("1"))
}

/// Position and channel of the first `limit` differing samples, for a panic
/// message.
pub fn first_differences(ours: &[u8], theirs: &[u8], channels: usize, limit: usize) -> String {
    ours.iter()
        .zip(theirs)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(limit)
        .map(|(i, (a, b))| {
            format!(
                "[sample {} ch {}] rust {a} c {b}",
                i / channels,
                i % channels
            )
        })
        .collect::<Vec<String>>()
        .join(", ")
}

/// P4-248: a sequential Huffman frame (SOF0/SOF1) whose entropy-coded data
/// holds a run of two or more 0xFF before a 0x00. libjpeg skips the extra
/// 0xFF as fill and reads one stuffed 0xFF; the baseline decoder here reads
/// it differently, with no warning on either side. Progressive streams are
/// not excused: P4-242 fixed that path, and an excuse there would hide a
/// regression of it. Only entropy-coded bytes are scanned — the walk skips
/// marker segments, so a run inside an APPn or DHT payload does not count.
pub fn has_fill_before_stuffed_zero(jpeg: &[u8]) -> bool {
    let be16 = |at: usize| -> Option<usize> {
        Some(usize::from(u16::from_be_bytes([
            *jpeg.get(at)?,
            *jpeg.get(at + 1)?,
        ])))
    };
    let mut is_sequential_huffman: bool = false;
    let mut pos: usize = 2; // after SOI
    while pos + 1 < jpeg.len() {
        if jpeg[pos] != 0xFF {
            return false;
        }
        let marker: u8 = jpeg[pos + 1];
        match marker {
            0xFF => {
                pos += 1; // fill before a marker
                continue;
            }
            0xD9 => return false,
            0xC0 | 0xC1 => is_sequential_huffman = true,
            0xC2 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => is_sequential_huffman = false,
            _ => {}
        }
        let Some(length) = be16(pos + 2) else {
            return false;
        };
        pos += 2 + length;
        if marker != 0xDA {
            continue;
        }
        if !is_sequential_huffman {
            return false;
        }
        // Entropy-coded data runs to the first marker other than RSTn.
        loop {
            let Some(&byte) = jpeg.get(pos) else {
                return false;
            };
            if byte != 0xFF {
                pos += 1;
                continue;
            }
            let run_start: usize = pos;
            while jpeg.get(pos) == Some(&0xFF) {
                pos += 1;
            }
            match jpeg.get(pos) {
                None => return false,
                Some(0x00) if pos - run_start >= 2 => return true,
                Some(0x00) | Some(0xD0..=0xD7) => pos += 1,
                Some(_) => {
                    pos -= 1; // the marker's own 0xFF
                    break;
                }
            }
        }
    }
    false
}

/// The largest |coefficient × quantizer| in the stream, or `None` when the
/// coefficients cannot be read. A DCT of 8-bit samples never exceeds 2047;
/// beyond that, the scaled IDCTs saturate where libjpeg's scalar
/// `range_limit[x & RANGE_MASK]` wraps — a deliberate choice documented in
/// `src/decode/idct_scaled.rs`, matching libjpeg's own NEON kernels, not a
/// defect to report.
///
/// `ComponentCoefficients::blocks` are in zigzag order and
/// `JpegCoefficients::quant_tables` in natural order, so each coefficient is
/// paired with the quantizer at its natural position.
pub fn max_dequantized_coefficient(jpeg: &[u8]) -> Option<i64> {
    let coefficients = libjpeg_turbo_rs::read_coefficients(jpeg).ok()?;
    coefficients
        .components
        .iter()
        .flat_map(|component| {
            let quant = coefficients
                .quant_tables
                .get(usize::from(component.quant_table_index));
            component.blocks.iter().flat_map(move |block| {
                block.iter().enumerate().map(move |(k, &v)| {
                    i64::from(v).abs() * quant.map_or(1, |q| i64::from(q[ZIGZAG_TO_NATURAL[k]]))
                })
            })
        })
        .max()
}

/// Natural (row-major) position of each zigzag index: libjpeg's
/// `jpeg_natural_order`, `ZIGZAG_ORDER` in `src/common/quant_table.rs`.
#[rustfmt::skip]
const ZIGZAG_TO_NATURAL: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Above this a dequantized coefficient cannot come from 8-bit samples.
pub const MAX_LEGAL_DEQUANTIZED: i64 = 2047;
