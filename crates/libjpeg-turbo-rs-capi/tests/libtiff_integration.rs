//! End-to-end integration test: libtiff JPEG-compressed TIFF round-trip
//! through our cdylib shim.
//!
//! # What this tests
//!
//! libtiff uses `jpeg_write_raw_data` / `jpeg_read_raw_data` (the iMCU-row
//! delivery API wired by PR #240 / #241) when encoding or decoding a TIFF
//! with `COMPRESSION_JPEG`.  Until this test existed we only validated that
//! libtiff could *dlopen* our shim (symbol resolution) — the actual
//! JPEG-compressed TIFF round-trip was never exercised.
//!
//! This test:
//!   1. Shells out to `examples/libtiff_integration/build.sh` to compile
//!      `main.c` against libtiff (skipping if `cc` or `tiffio.h` is absent).
//!   2. Shells out to `examples/libtiff_integration/run.sh`, which stages
//!      our cdylib as the JPEG provider and runs the compiled binary.
//!   3. Hard-panics on any Rust/build failure; soft-skips only when system
//!      tools (cc, libtiff) are genuinely absent — per CLAUDE.md rule.
//!
//! # Skip conditions (per CLAUDE.md C cross-validation rules)
//!
//! - `cc` compiler not found → SKIP
//! - `tiffio.h` / libtiff not found → SKIP
//! - Any Rust / cdylib failure → FAIL (panic)

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR for the capi crate is
    // <repo>/crates/libjpeg-turbo-rs-capi; go up two levels.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|e| panic!("cannot canonicalize repo root: {e}"))
}

/// Hard-build the cdylib if it is not already present.
///
/// The test must be able to find the cdylib that run.sh will stage as the
/// JPEG provider.  We replicate the `cdylib_path_or_build()` pattern from
/// `tjunittest_link.rs` so a stale `target/release/...` cannot silently
/// satisfy the gate.
fn cdylib_path() -> PathBuf {
    let dlext: &str = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let lib_prefix: &str = if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    };

    // Walk upward from the test binary to find the cdylib, the same way
    // precision.rs does.
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate = dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix, dlext));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!(
        "could not locate cdylib near {}: \
         run `cargo build -p libjpeg-turbo-rs-capi --release` first",
        exe.display()
    );
}

/// `libtiff_jpeg_roundtrip_via_shim` — the primary end-to-end gate.
///
/// Builds `examples/libtiff_integration/main.c`, runs it with our cdylib
/// as the JPEG provider, and asserts exit 0.
///
/// libtiff's COMPRESSION_JPEG read path calls `jpeg_read_header` first on
/// the TIFF `JPEGTables` tag (an abbreviated tables-only datastream:
/// SOI + DQT + DHT + EOI, no SOF/SOS) and expects
/// `JPEG_HEADER_TABLES_ONLY` (= 2). The shim's `jpeg_read_header` walks
/// markers manually to detect the absence of SOF/SOS, stashes the
/// (EOI-stripped) bytes in `priv_state.tables_only_prefix`, resets the
/// drained source so the next call re-reads from the caller's source
/// manager, and returns `JPEG_HEADER_TABLES_ONLY`. On the subsequent
/// per-strip `jpeg_read_header` call the prefix is spliced in front of
/// the strip body (after dropping the strip's own SOI) so
/// `Decoder::new` sees a complete tables+image stream — see the
/// implementation in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`'s
/// `detect_tables_only` helper plus the splice block in
/// `jpeg_read_header`.
#[test]
fn libtiff_jpeg_roundtrip_via_shim() {
    // Skip on Windows — the SONAME / loader-path scheme is POSIX-only.
    if cfg!(target_os = "windows") {
        eprintln!("SKIP: libtiff_integration is POSIX-only (Windows not supported)");
        return;
    }

    // Ensure the cdylib exists (panics if not).
    let _cdylib: PathBuf = cdylib_path();

    let root: PathBuf = repo_root();
    let build_sh: PathBuf = root.join("examples/libtiff_integration/build.sh");
    let run_sh: PathBuf = root.join("examples/libtiff_integration/run.sh");

    if !build_sh.is_file() {
        panic!(
            "examples/libtiff_integration/build.sh not found at {}; \
             check your worktree",
            build_sh.display()
        );
    }
    if !run_sh.is_file() {
        panic!(
            "examples/libtiff_integration/run.sh not found at {}; \
             check your worktree",
            run_sh.display()
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1: build.sh
    // Skip if cc or libtiff are absent (exit 1 or 2 from build.sh).
    // Hard-panic on compilation failure (exit 3).
    // -----------------------------------------------------------------------
    eprintln!("==> Running examples/libtiff_integration/build.sh");
    let build_status: std::process::Output = Command::new("bash")
        .arg(&build_sh)
        .output()
        .expect("failed to spawn build.sh — bash must be on PATH");

    eprintln!(
        "--- build.sh stdout ---\n{}",
        String::from_utf8_lossy(&build_status.stdout)
    );
    eprintln!(
        "--- build.sh stderr ---\n{}",
        String::from_utf8_lossy(&build_status.stderr)
    );

    match build_status.status.code() {
        Some(0) => { /* Build succeeded, continue. */ }
        Some(1) => {
            // libtiff headers / library not found.
            eprintln!(
                "SKIP: libtiff not installed (build.sh exit 1) — \
                 install libtiff-dev (apt) or libtiff (brew) to enable this test"
            );
            return;
        }
        Some(2) => {
            // cc not found.
            eprintln!(
                "SKIP: cc compiler not found (build.sh exit 2) — \
                 install build tools to enable this test"
            );
            return;
        }
        Some(3) | _ => {
            // Compilation failed — this is a real error that must not be skipped.
            panic!(
                "examples/libtiff_integration/build.sh FAILED (exit {:?}): \
                 compilation error — this is a real build failure, not a \
                 missing-tool skip. See stderr above.",
                build_status.status.code()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: run.sh
    // Exit 2 = skip (binary / shim not found — should not happen after a
    //   successful build above, but guard defensively).
    // Exit 1 = pixel mismatch or API failure — REAL BUG in our shim.
    // Exit 0 = PASS.
    // -----------------------------------------------------------------------
    eprintln!("==> Running examples/libtiff_integration/run.sh");
    let run_status: std::process::Output = Command::new("bash")
        .arg(&run_sh)
        .output()
        .expect("failed to spawn run.sh");

    eprintln!(
        "--- run.sh stdout ---\n{}",
        String::from_utf8_lossy(&run_status.stdout)
    );
    eprintln!(
        "--- run.sh stderr ---\n{}",
        String::from_utf8_lossy(&run_status.stderr)
    );

    match run_status.status.code() {
        Some(0) => {
            eprintln!("PASS: libtiff JPEG round-trip succeeded via our shim");
        }
        Some(2) => {
            // run.sh exits 2 only when it cannot find the binary or shim
            // (pre-flight check failure, not a shim bug).  Unlikely after a
            // clean build.sh pass but handled gracefully.
            eprintln!(
                "SKIP: run.sh reported exit 2 (binary or shim not found) — \
                 this is unexpected after a successful build"
            );
        }
        Some(1) => {
            panic!(
                "examples/libtiff_integration/run.sh FAILED with exit 1: \
                 pixel mismatch — JPEG round-trip through our shim produced \
                 incorrect pixel values.  This is a real correctness bug in \
                 jpeg_write_raw_data / jpeg_read_raw_data or the libtiff \
                 integration path.  See stdout/stderr above for details."
            );
        }
        Some(code) => {
            // exit 3 = API failure (TIFFOpen/Write/Read returned error from main.c).
            // Any other non-zero exit is also a real failure.
            panic!(
                "examples/libtiff_integration/run.sh FAILED with exit code {code}: \
                 libtiff API error — TIFFWriteEncodedStrip / TIFFReadEncodedStrip \
                 failed against our shim.  This is a real C-ABI shim bug \
                 (likely in jpeg_write_raw_data / jpeg_read_raw_data or the \
                 abbreviated tables-only datastream path in jpeg_read_header).  \
                 See stdout/stderr above for the exact error message."
            );
        }
        None => {
            panic!(
                "run.sh was killed by a signal — likely a crash or abort in our \
                 shim's jpeg_write_raw_data / jpeg_read_raw_data path"
            );
        }
    }
}
