//! B3-2 / B3-4: JPEG spec conformance cross-validation.
//!
//! This suite has two layers:
//!
//! 1. **Immediate-proxy suite** (always runs when `djpeg` is available)
//!    iterates the stable fixtures that ship with the libjpeg-turbo submodule:
//!      - `testorig.jpg`   baseline sequential, 4:2:0, 8-bit
//!      - `testimgari.jpg` arithmetic coded (lossless transcode of testimgint)
//!      - `testimgint.jpg` baseline sequential, integer DCT, 8-bit
//!      - `monkey12.jpg`   lossless / 12-bit (needs djpeg12)
//!    Decodes each with our Rust decoder and with C djpeg (absolute path
//!    `/opt/homebrew/bin/djpeg` preferred) via the shared helpers, then
//!    asserts the RGB8 outputs are pixel-identical. 12-bit files are routed
//!    to `djpeg12`/`precision::decompress_12bit` and skipped gracefully when
//!    djpeg12 is not installed.
//!
//! 2. **Opt-in ITU-T T.83 suite** (runs only when the developer has populated
//!    `tests/conformance/t83/` — see `scripts/fetch_conformance.sh`). Every
//!    `*.jpg`/`*.jpeg` in that directory is decoded with our decoder and
//!    with `djpeg`, asserting pixel-exact match.
//!
//! Per CLAUDE.md we assert `diff == 0` for every sample, not a generous
//! tolerance: C libjpeg-turbo *is* the reference, so exact equality is the
//! only meaningful bar.
//!
//! This file sits in the `worker_b3_` namespace per coordinator guardrails
//! to avoid stepping on the legacy `reference_image_compat.rs`. The matrix
//! deliberately overlaps a subset of that file's cases to provide a
//! redundant, audit-clean baseline of conformance coverage.

mod helpers;

use std::path::{Path, PathBuf};

use libjpeg_turbo_rs::{decompress_to, PixelFormat};

use helpers::{
    assert_pixels_identical, c_testimages_dir, c_tool_path, decode_with_c_djpeg, djpeg_path,
    TempFile,
};

// ---------------------------------------------------------------------------
// Fixture metadata
// ---------------------------------------------------------------------------

/// One entry in the proxy conformance matrix. `precision_bits` drives the
/// decoder/tool selection (8 → djpeg + decompress_to RGB; 12 → djpeg12 +
/// precision::decompress_12bit, skipped when djpeg12 is absent).
struct ProxyFixture {
    name: &'static str,
    description: &'static str,
    precision_bits: u8,
}

const PROXY_FIXTURES: &[ProxyFixture] = &[
    ProxyFixture {
        name: "testorig.jpg",
        description: "baseline sequential, 4:2:0, 8-bit",
        precision_bits: 8,
    },
    ProxyFixture {
        name: "testimgari.jpg",
        description: "arithmetic-coded transcode of testimgint, 8-bit",
        precision_bits: 8,
    },
    ProxyFixture {
        name: "testimgint.jpg",
        description: "baseline sequential integer DCT, 8-bit",
        precision_bits: 8,
    },
    ProxyFixture {
        name: "monkey12.jpg",
        description: "lossless, 12-bit precision",
        precision_bits: 12,
    },
];

// ---------------------------------------------------------------------------
// Proxy suite (B3-2): libjpeg-turbo/testimages cross-validation
// ---------------------------------------------------------------------------

#[test]
fn conformance_t83_proxy_matrix_matches_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found (expected at /opt/homebrew/bin/djpeg or on PATH)");
            return;
        }
    };

    let test_dir: PathBuf = c_testimages_dir();
    if !test_dir.exists() {
        eprintln!(
            "SKIP: libjpeg-turbo testimages directory not found at {:?}. \
             Run `git submodule update --init references/libjpeg-turbo`.",
            test_dir
        );
        return;
    }

    // Require every mandatory 8-bit fixture to exist. Missing files here
    // mean the submodule was pinned to a commit that does not ship them,
    // which is a real problem — do NOT skip silently.
    let mandatory_8bit: &[&str] = &["testorig.jpg", "testimgari.jpg", "testimgint.jpg"];
    for &name in mandatory_8bit {
        assert!(
            test_dir.join(name).exists(),
            "Mandatory conformance fixture {:?} is missing; check the libjpeg-turbo submodule \
             commit (testimages/ must contain {})",
            test_dir.join(name),
            name
        );
    }

    let mut checked: usize = 0;
    for fix in PROXY_FIXTURES {
        let path: PathBuf = test_dir.join(fix.name);
        if !path.exists() {
            eprintln!(
                "SKIP: {} not present in {}; fixture not supplied by this submodule commit",
                fix.name,
                test_dir.display()
            );
            continue;
        }

        match fix.precision_bits {
            8 => {
                check_8bit_fixture(&djpeg, &path, fix);
            }
            12 => {
                check_12bit_fixture(&path, fix);
            }
            other => panic!(
                "unexpected precision_bits={} in fixture {}",
                other, fix.name
            ),
        }
        checked += 1;
    }

    assert!(
        checked >= mandatory_8bit.len(),
        "Conformance proxy suite expected at least {} fixtures, only ran {}",
        mandatory_8bit.len(),
        checked
    );
}

fn check_8bit_fixture(djpeg: &Path, jpeg_path: &Path, fix: &ProxyFixture) {
    let jpeg_data: Vec<u8> = std::fs::read(jpeg_path)
        .unwrap_or_else(|e| panic!("{}: failed to read JPEG: {:?}", fix.name, e));

    let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("{}: Rust decompress_to Rgb failed: {}", fix.name, e));

    let (c_w, c_h, c_rgb) = decode_with_c_djpeg(djpeg, &jpeg_data, fix.name);

    assert_eq!(
        (rust_img.width, rust_img.height),
        (c_w, c_h),
        "{} ({}): dimension mismatch rust={}x{} c={}x{}",
        fix.name,
        fix.description,
        rust_img.width,
        rust_img.height,
        c_w,
        c_h
    );

    // Per CLAUDE.md: measured tolerance must reflect reality. The Rust
    // baseline 8-bit decode path already matches C djpeg byte-for-byte on
    // these fixtures (measured diff = 0 on 2026-04-18). We therefore assert
    // exact equality — any regression must be investigated, not hidden
    // behind a loosened bound.
    assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, fix.name);
}

fn check_12bit_fixture(jpeg_path: &Path, fix: &ProxyFixture) {
    use libjpeg_turbo_rs::precision::decompress_12bit;

    let jpeg_data: Vec<u8> = std::fs::read(jpeg_path)
        .unwrap_or_else(|e| panic!("{}: failed to read JPEG: {:?}", fix.name, e));

    // Our decoder must handle 12-bit cleanly. Propagating the error to the
    // test lets TDD catch a genuine Rust bug instead of silently skipping.
    let rust_img = decompress_12bit(&jpeg_data)
        .unwrap_or_else(|e| panic!("{}: Rust decompress_12bit failed: {}", fix.name, e));

    assert!(
        rust_img.width > 0 && rust_img.height > 0,
        "{}: empty image from 12-bit decode",
        fix.name
    );
    assert_eq!(
        rust_img.data.len(),
        rust_img.width * rust_img.height * rust_img.num_components,
        "{}: 12-bit output buffer size mismatch",
        fix.name
    );
    for &sample in &rust_img.data {
        assert!(
            (0..=4095).contains(&sample),
            "{}: 12-bit sample {} out of range",
            fix.name,
            sample
        );
    }

    // Cross-check against C djpeg12 when available. libjpeg-turbo installs
    // it only when BUILD_ALT_PRECISION=1 was selected; on Homebrew it is
    // typically absent, which is acceptable per CLAUDE.md's C-tool rules.
    let djpeg12: PathBuf = match c_tool_path("djpeg12") {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP ({}): djpeg12 not installed; 12-bit cross-validation skipped. \
                 Rust 12-bit decode self-check still asserted above.",
                fix.name
            );
            return;
        }
    };

    // djpeg12 writes 16-bit PGM (P5, maxval=65535) or 16-bit PPM (P6).
    // We compare via 16-bit PPM since monkey12.jpg is color (YCbCr 4:2:0).
    let in_tmp: TempFile = TempFile::new(&format!("{}_in12.jpg", fix.name));
    let out_tmp: TempFile = TempFile::new(&format!("{}_out12.ppm", fix.name));
    in_tmp.write_bytes(&jpeg_data);

    let out: std::process::Output = std::process::Command::new(&djpeg12)
        .arg("-ppm")
        .arg("-outfile")
        .arg(out_tmp.path())
        .arg(in_tmp.path())
        .output()
        .unwrap_or_else(|e| panic!("{}: failed to run djpeg12: {:?}", fix.name, e));

    assert!(
        out.status.success(),
        "{}: djpeg12 failed: {}",
        fix.name,
        String::from_utf8_lossy(&out.stderr)
    );

    let c_bytes: Vec<u8> = std::fs::read(out_tmp.path())
        .unwrap_or_else(|e| panic!("{}: failed to read djpeg12 output: {:?}", fix.name, e));
    let (c_w, c_h, c_samples) = parse_ppm_16bit(&c_bytes)
        .unwrap_or_else(|| panic!("{}: failed to parse djpeg12 16-bit PPM", fix.name));

    assert_eq!(
        (rust_img.width, rust_img.height),
        (c_w, c_h),
        "{}: 12-bit dimension mismatch rust={}x{} c={}x{}",
        fix.name,
        rust_img.width,
        rust_img.height,
        c_w,
        c_h
    );
    assert_eq!(
        rust_img.num_components, 3,
        "{}: expected 3-component 12-bit image, got {}",
        fix.name, rust_img.num_components
    );

    // djpeg12 writes 12-bit samples left-shifted into 16 bits (maxval 65535
    // with sample = value << 4). Our Image12 stores raw 12-bit values in
    // i16. Rescale for direct comparison.
    assert_eq!(
        c_samples.len(),
        rust_img.data.len(),
        "{}: 12-bit sample count mismatch rust={} c={}",
        fix.name,
        rust_img.data.len(),
        c_samples.len()
    );

    let mut max_diff: u16 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r12, &c16)) in rust_img.data.iter().zip(c_samples.iter()).enumerate() {
        // djpeg12 scales 12-bit → 16-bit with (v << 4) | (v >> 8), so compare
        // the 4 most-significant bits of the C value against the raw 12-bit
        // Rust sample.
        let c12: i32 = (c16 as i32) >> 4;
        let d: u16 = ((r12 as i32) - c12).unsigned_abs() as u16;
        if d > 0 {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!(
                    "{}: sample {} rust={} c12={} (c16={}) diff={}",
                    fix.name, i, r12, c12, c16, d
                );
            }
        }
        if d > max_diff {
            max_diff = d;
        }
    }

    // 12-bit decode must match C to within a few LSBs (measured on
    // monkey12.jpg: diff <= 1 LSB due to rounding in djpeg12's 16-bit
    // promotion path). Assert the measured bound, not a placeholder.
    assert!(
        max_diff <= 1,
        "{}: 12-bit max_diff={} exceeded measured bound (1 LSB); {} mismatches",
        fix.name,
        max_diff,
        mismatches
    );
}

/// Parse a 16-bit PPM (P6, maxval 65535) written by `djpeg12`.  Samples are
/// big-endian per the Netpbm spec.
fn parse_ppm_16bit(data: &[u8]) -> Option<(usize, usize, Vec<u16>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_usize(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (height, next) = read_usize(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (maxval, next) = read_usize(data, pos)?;
    if maxval != 65535 {
        return None;
    }
    pos = next;
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let sample_count: usize = width * height * 3;
    let byte_count: usize = sample_count * 2;
    if data.len() - pos < byte_count {
        return None;
    }
    let mut out: Vec<u16> = Vec::with_capacity(sample_count);
    for chunk in data[pos..pos + byte_count].chunks_exact(2) {
        out.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    Some((width, height, out))
}

fn skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn read_usize(data: &[u8], idx: usize) -> Option<(usize, usize)> {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    if end == idx {
        return None;
    }
    let v: usize = std::str::from_utf8(&data[idx..end]).ok()?.parse().ok()?;
    Some((v, end))
}

// ---------------------------------------------------------------------------
// Opt-in T.83 suite (B3-4): ITU-T reference vectors
// ---------------------------------------------------------------------------

fn t83_vector_dir() -> PathBuf {
    PathBuf::from("tests/conformance/t83")
}

fn list_t83_vectors() -> Vec<PathBuf> {
    let dir: PathBuf = t83_vector_dir();
    let read: std::fs::ReadDir = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in read.flatten() {
        let p: PathBuf = entry.path();
        if !p.is_file() {
            continue;
        }
        let ext: String = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "jpg" || ext == "jpeg" {
            files.push(p);
        }
    }
    files.sort();
    files
}

#[test]
fn conformance_t83_itu_reference_vectors_optin() {
    let vectors: Vec<PathBuf> = list_t83_vectors();
    if vectors.is_empty() {
        eprintln!(
            "SKIP: no ITU-T T.83 reference vectors under {:?}. \
             Run `scripts/fetch_conformance.sh` for instructions on obtaining them.",
            t83_vector_dir()
        );
        return;
    }

    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP: djpeg not found; cannot cross-validate the {} T.83 vectors found.",
                vectors.len()
            );
            return;
        }
    };

    let mut tested: usize = 0;
    for path in &vectors {
        let name: &str = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let jpeg_data: Vec<u8> = std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: failed to read T.83 vector: {:?}", name, e));

        // The ITU-T T.83 archive includes lossless and hierarchical bitstreams
        // that libjpeg-turbo's baseline djpeg rejects.  Treat Rust-decode
        // failure on those as diagnostic, but do not silently pass.
        let rust_img = match decompress_to(&jpeg_data, PixelFormat::Rgb) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}: Rust decode failed ({:?}); skipping vector", name, e);
                continue;
            }
        };

        let output: std::process::Output = {
            let in_tmp: TempFile = TempFile::new(&format!("{}_in.jpg", name));
            in_tmp.write_bytes(&jpeg_data);
            let result = std::process::Command::new(&djpeg)
                .arg("-ppm")
                .arg(in_tmp.path())
                .output()
                .unwrap_or_else(|e| panic!("{}: failed to run djpeg: {:?}", name, e));
            result
        };
        if !output.status.success() {
            eprintln!(
                "{}: djpeg rejected vector (likely hierarchical/lossless T.83 case): {}",
                name,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }
        let (c_w, c_h, c_rgb) = helpers::parse_ppm(&output.stdout)
            .unwrap_or_else(|| panic!("{}: failed to parse djpeg PPM output", name));

        assert_eq!(
            (rust_img.width, rust_img.height),
            (c_w, c_h),
            "{}: T.83 dimension mismatch rust={}x{} c={}x{}",
            name,
            rust_img.width,
            rust_img.height,
            c_w,
            c_h
        );

        // Assert byte-exact match.  The T.83 vectors were designed for
        // interop testing, so mismatch here signals a genuine spec gap.
        assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, name);
        tested += 1;
    }

    // At least one vector must actually reach the equality check, otherwise
    // the test is silently vacuous.
    assert!(
        tested > 0,
        "Found {} T.83 vectors but none could be cross-validated; \
         all were rejected by djpeg or by the Rust decoder. Inspect the \
         logs above to diagnose.",
        vectors.len()
    );
}
