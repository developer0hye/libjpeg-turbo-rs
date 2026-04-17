//! B3-4: opt-in ITU-T T.83 / ISO 10918-2 reference-vector conformance.
//!
//! This file ADDS to the lightweight T.83 scanner introduced in
//! `worker_b3_conformance_t83.rs` with three extra guarantees required by
//! the B3 mission:
//!
//! 1. **Canonical-vector checklist.** The ITU-T T.83 CD-ROM ships a set of
//!    reference bitstreams whose filenames have been stable for three
//!    decades (A1.JPG, A2.JPG, F-1.JPG, F-4.JPG, ...). When the opt-in
//!    directory contains any vectors, we also emit warnings for the
//!    canonical files still missing — so a half-populated suite does not
//!    silently understate its coverage.
//!
//! 2. **Byte-exact djpeg cross-validation** for every vector that both
//!    decoders can process. Failure at this layer is a genuine spec gap,
//!    not rounding: we do not loosen tolerance.
//!
//! 3. **Fetch-script contract test.** Verifies that
//!    `scripts/fetch_conformance.sh` exists, is readable, and correctly
//!    reports "missing" via exit code 2 when the directory is empty. This
//!    keeps the fetch workflow from silently breaking.
//!
//! When `tests/conformance/t83/` is empty (the default for fresh clones),
//! the conformance test skips gracefully and prints the fetch instructions.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use libjpeg_turbo_rs::{decompress_to, PixelFormat};

use helpers::{assert_pixels_identical, djpeg_path, parse_ppm, TempFile};

const T83_DIR: &str = "tests/conformance/t83";
const FETCH_SCRIPT: &str = "scripts/fetch_conformance.sh";

/// Canonical filenames found on the ITU-T T.83 CD-ROM. Case-insensitive.
/// Missing files do not fail the test (the archive is licensed and often
/// incomplete in local copies) — but we print an explicit warning so the
/// developer notices partial coverage.
const T83_CANONICAL_VECTORS: &[&str] = &[
    "A1.JPG", "A2.JPG", "F-1.JPG", "F-4.JPG", "F-7.JPG", "F-14.JPG", "F-18.JPG", "F-23.JPG",
];

fn list_t83_vectors(dir: &Path) -> Vec<PathBuf> {
    let rd: std::fs::ReadDir = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let p: PathBuf = e.path();
        if !p.is_file() {
            continue;
        }
        let ext: String = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "jpg" || ext == "jpeg" {
            out.push(p);
        }
    }
    out.sort();
    out
}

fn uppercase_filename(p: &Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_default()
}

#[test]
fn conformance_t83_itu_vectors_strict() {
    let dir: PathBuf = PathBuf::from(T83_DIR);
    let vectors: Vec<PathBuf> = list_t83_vectors(&dir);

    if vectors.is_empty() {
        eprintln!(
            "SKIP: no ITU-T T.83 reference vectors under {:?}.\n\
             Obtain them via `bash {}` and re-run this test.",
            dir, FETCH_SCRIPT
        );
        return;
    }

    // Warn (do not fail) about canonical vectors the developer has not
    // placed.  The archive is frequently partial in practice.
    let present_upper: std::collections::HashSet<String> =
        vectors.iter().map(|p| uppercase_filename(p)).collect();
    let mut missing_canonical: Vec<&str> = Vec::new();
    for &expected in T83_CANONICAL_VECTORS {
        if !present_upper.contains(expected) {
            missing_canonical.push(expected);
        }
    }
    if !missing_canonical.is_empty() {
        eprintln!(
            "[conformance_t83_strict] partial T.83 coverage — canonical vectors \
             missing from {}: {}",
            dir.display(),
            missing_canonical.join(", ")
        );
    }

    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP: djpeg not found; cannot cross-validate the {} T.83 vectors present.",
                vectors.len()
            );
            return;
        }
    };

    let mut validated: usize = 0;
    let mut rust_errors: Vec<(String, String)> = Vec::new();
    let mut djpeg_errors: Vec<(String, String)> = Vec::new();

    for path in &vectors {
        let name: String = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        let jpeg_data: Vec<u8> = std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: failed to read T.83 vector: {:?}", name, e));

        let rust_result = decompress_to(&jpeg_data, PixelFormat::Rgb);
        let output: std::process::Output = {
            let in_tmp: TempFile = TempFile::new(&format!("b3-4_{}.jpg", name));
            in_tmp.write_bytes(&jpeg_data);
            Command::new(&djpeg)
                .arg("-ppm")
                .arg(in_tmp.path())
                .output()
                .unwrap_or_else(|e| panic!("{}: failed to run djpeg: {:?}", name, e))
        };

        let djpeg_ok: bool = output.status.success();

        match (rust_result, djpeg_ok) {
            (Ok(rust_img), true) => {
                let (c_w, c_h, c_rgb) = parse_ppm(&output.stdout).unwrap_or_else(|| {
                    panic!("{}: djpeg produced non-PPM output on a T.83 vector", name)
                });
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
                // Byte-exact equality is the only meaningful T.83 bar.
                assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, &name);
                validated += 1;
            }
            (Err(e), true) => {
                // Rust rejects a vector djpeg accepts: real gap, not a
                // format djpeg also skips. Collect and fail at the end.
                rust_errors.push((name, format!("{}", e)));
            }
            (Ok(_), false) => {
                // djpeg rejects but Rust accepts — likely a T.83 mode
                // (lossless SOF3, hierarchical SOF5) djpeg never supported.
                // Treat as diagnostic, not a regression.
                let err_msg: String = String::from_utf8_lossy(&output.stderr).trim().to_string();
                djpeg_errors.push((name, err_msg));
            }
            (Err(rust_e), false) => {
                // Both reject: likely hierarchical / out-of-scope. Skip
                // with a diagnostic.
                let err_msg: String = String::from_utf8_lossy(&output.stderr).trim().to_string();
                eprintln!(
                    "{}: both Rust and djpeg reject vector (rust={}; c={})",
                    name, rust_e, err_msg
                );
            }
        }
    }

    for (name, err) in &djpeg_errors {
        eprintln!(
            "{}: djpeg rejected vector (Rust accepted): {}",
            name,
            err.lines().next().unwrap_or("")
        );
    }

    if !rust_errors.is_empty() {
        for (name, err) in &rust_errors {
            eprintln!(
                "REGRESSION: {}: Rust rejected djpeg-accepted vector: {}",
                name, err
            );
        }
        panic!(
            "{} T.83 vector(s) accepted by djpeg but rejected by the Rust decoder. \
             Each represents a genuine conformance gap.",
            rust_errors.len()
        );
    }

    assert!(
        validated > 0,
        "{} T.83 vector(s) found but none were byte-exact validated. \
         Inspect the log above for djpeg/Rust rejections — at least one \
         shared-mode vector must round-trip exactly.",
        vectors.len()
    );
}

#[test]
fn fetch_conformance_script_exists_and_reports_missing() {
    let script: PathBuf = PathBuf::from(FETCH_SCRIPT);
    assert!(
        script.exists(),
        "{} is missing — B3-1 provides this fetcher and it is part of the contract.",
        FETCH_SCRIPT
    );

    // Run the script with --check and an explicitly redirected output.
    // We do NOT run it in a fresh temp dir: the script discovers the repo
    // via its own path, so we just rely on the working state of the
    // current checkout. If tests/conformance/t83 happens to be populated
    // locally, the script exits 0 — both 0 and 2 are contractually valid.
    let out = match Command::new("bash")
        .arg(&script)
        .arg("--check")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("SKIP: unable to execute bash on {}: {:?}", FETCH_SCRIPT, e);
            return;
        }
    };

    let code: Option<i32> = out.status.code();
    let stdout_text: String = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr_text: String = String::from_utf8_lossy(&out.stderr).to_string();

    // Script must terminate with 0 (vectors present) or 2 (missing — needs
    // manual acquisition). Any other exit code is a real bug.
    assert!(
        matches!(code, Some(0) | Some(2)),
        "fetch_conformance.sh exited {:?}; expected 0 or 2.\nstdout:\n{}\nstderr:\n{}",
        code,
        stdout_text,
        stderr_text
    );

    // When the directory is empty, the script must print a pointer to the
    // target path and to the manual-acquisition instructions.
    let t83_dir: PathBuf = PathBuf::from(T83_DIR);
    let has_jpegs: bool = list_t83_vectors(&t83_dir).iter().any(|_| true);
    if !has_jpegs {
        assert_eq!(
            code,
            Some(2),
            "fetch_conformance.sh should return exit code 2 when T.83 vectors are missing.\n\
             stdout:\n{}\nstderr:\n{}",
            stdout_text,
            stderr_text
        );
        let expected_hint: &str = "tests/conformance/t83";
        assert!(
            stdout_text.contains(expected_hint) || stderr_text.contains(expected_hint),
            "fetch_conformance.sh output does not reference the expected target path \
             `{}`.\nstdout:\n{}\nstderr:\n{}",
            expected_hint,
            stdout_text,
            stderr_text
        );
    }
}
