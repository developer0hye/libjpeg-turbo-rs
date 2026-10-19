//! Differential fuzzing of `Decoder::decode_raw` — component planes before
//! upsampling and colour conversion — against libjpeg's
//! `jpeg_read_raw_data`, through `examples/raw_decode_c_oracle.c`.
//!
//! `decode_raw` documents itself as that C call, and downstream decoders
//! build their own upsampling on it, but no target compared it with libjpeg.
//! The input is a JPEG.
//!
//! Asserts, when libjpeg decodes the input without a warning:
//! 1. Rust accepts it too (drop-in floor);
//! 2. the component count and each component's `downsampled_width` ×
//!    `downsampled_height` agree;
//! 3. every sample inside those areas is identical. The MCU padding beyond
//!    them is not compared: it is not part of the image.
//!
//! Skips: arithmetic-coded streams (the open divergence
//! `fuzz_decode_diff_c` also skips), and precisions other than 8 and
//! lossless frames, which both decoders refuse for raw output. A mismatch is
//! excused, unless `LTRS_FUZZ_INCLUDE_KNOWN=1`, when libjpeg without block
//! smoothing agrees with Rust (P4-249) or the entropy data holds 0xFF fill
//! before a stuffed byte (P4-248).

#![no_main]

#[path = "common/c_oracle.rs"]
mod c_oracle;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{Decoder, JpegError};

const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fn oracle() -> &'static PathBuf {
    static ORACLE: OnceLock<PathBuf> = OnceLock::new();
    ORACLE.get_or_init(|| c_oracle::oracle_path("RAW_C_ORACLE"))
}

/// The oracle's planes as `(width, height, samples)`.
fn parse_raw(stdout: &[u8]) -> Vec<(usize, usize, Vec<u8>)> {
    let (header, mut rest) = c_oracle::split_header(stdout).expect("raw oracle header");
    let components: usize = match header[..] {
        ["RAW", n] => n.parse().expect("component count"),
        _ => panic!("malformed raw oracle header {header:?}"),
    };
    (0..components)
        .map(|_| {
            let (dims, body) = c_oracle::split_header(rest).expect("raw plane header");
            let (w, h): (usize, usize) = match dims[..] {
                [w, h] => (w.parse().unwrap(), h.parse().unwrap()),
                _ => panic!("malformed raw plane header {dims:?}"),
            };
            let plane: Vec<u8> = body[..w * h].to_vec();
            rest = &body[w * h..];
            (w, h, plane)
        })
        .collect()
}

/// A raw mismatch with a filed explanation: libjpeg smooths progressive
/// scans not refined to the last bit and `decode_raw` does not (P4-249, so
/// Rust equals libjpeg with `nosmooth`), or 0xFF fill before a stuffed byte
/// (P4-248).
fn known_raw_divergence(
    oracle: &Path,
    data: &[u8],
    raw: &libjpeg_turbo_rs::RawImage,
    active: &[(usize, usize)],
) -> bool {
    if c_oracle::has_fill_before_stuffed_zero(data) {
        return true;
    }
    let args: [String; 2] = ["/dev/stdin".to_string(), "nosmooth".to_string()];
    let Some(unsmoothed) = c_oracle::run_oracle(oracle, &args, data) else {
        return false;
    };
    parse_raw(&unsmoothed.stdout)
        .iter()
        .zip(active)
        .enumerate()
        .all(|(ci, ((_, _, plane), &(w, h)))| {
            let stride: usize = raw.plane_widths[ci];
            (0..h).all(|y| raw.planes[ci][y * stride..y * stride + w] == plane[y * w..(y + 1) * w])
        })
}

fuzz_target!(|data: &[u8]| {
    c_oracle::ensure_sigpipe_ignored();
    let oracle: &PathBuf = oracle();

    let Ok(probe) = Decoder::new(data) else {
        return;
    };
    let header = probe.header();
    let pixels: u64 = u64::from(header.width).saturating_mul(u64::from(header.height));
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    if probe.is_arithmetic() && !c_oracle::include_known_gaps() {
        return;
    }
    if header.precision != 8 || header.is_lossless {
        return;
    }
    let (width, height): (usize, usize) = (header.width(), header.height());
    let max_h: usize = header
        .components
        .iter()
        .map(|c| usize::from(c.horizontal_sampling))
        .max()
        .unwrap_or(1);
    let max_v: usize = header
        .components
        .iter()
        .map(|c| usize::from(c.vertical_sampling))
        .max()
        .unwrap_or(1);
    let active: Vec<(usize, usize)> = header
        .components
        .iter()
        .map(|c| {
            (
                (width * usize::from(c.horizontal_sampling)).div_ceil(max_h),
                (height * usize::from(c.vertical_sampling)).div_ceil(max_v),
            )
        })
        .collect();

    let c_result = c_oracle::run_oracle(oracle, &["/dev/stdin".to_string()], data);
    let rust_result = Decoder::new(data).and_then(|decoder| decoder.decode_raw());

    let c = match c_result {
        None => return,
        Some(c) if c.warned => return,
        Some(c) => c,
    };
    let raw = match rust_result {
        Ok(raw) => raw,
        // Deliberate divergence (#355 scan cap, P4-55).
        Err(JpegError::LimitExceeded { .. }) => return,
        Err(error) => {
            panic!(
                "drop-in regression: libjpeg read raw data cleanly but Rust rejected it: {error}"
            )
        }
    };
    let c_planes = parse_raw(&c.stdout);
    assert_eq!(
        raw.num_components,
        c_planes.len(),
        "raw component count diverges"
    );
    for (ci, ((cw, ch, c_plane), &(rw, rh))) in c_planes.iter().zip(&active).enumerate() {
        assert_eq!(
            (rw, rh),
            (*cw, *ch),
            "component {ci} active area diverges from libjpeg's downsampled size"
        );
        let stride: usize = raw.plane_widths[ci];
        assert!(
            stride >= rw && raw.planes[ci].len() >= stride * rh,
            "component {ci} plane {}x{} ({} bytes) cannot hold its {rw}x{rh} active area",
            stride,
            raw.plane_heights[ci],
            raw.planes[ci].len()
        );
        let ours: Vec<u8> = (0..rh)
            .flat_map(|y| raw.planes[ci][y * stride..y * stride + rw].iter().copied())
            .collect();
        if ours != *c_plane {
            if !c_oracle::include_known_gaps() && known_raw_divergence(oracle, data, &raw, &active)
            {
                return;
            }
            let differing: usize = ours.iter().zip(c_plane).filter(|(a, b)| a != b).count();
            panic!(
                "raw component {ci} ({rw}x{rh}) diverges from libjpeg: {differing} samples differ; first: {}",
                c_oracle::first_differences(&ours, c_plane, 1, 6)
            );
        }
    }
});
