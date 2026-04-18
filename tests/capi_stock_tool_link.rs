//! FFI B9-4: Verify stock djpeg/cjpeg/jpegtran can link against our cdylib.
//!
//! This test drives `examples/stock_djpeg_cjpeg/build.sh` (build stock C
//! tools linked against our shim) and — if that succeeds — `run.sh` (invoke
//! each built tool over `references/libjpeg-turbo/testimages/*.jpg` and
//! diff-compare against the system stock binaries).
//!
//! Current status: the build step fails at the linker with 20+ missing
//! `jpeg_*` symbols per tool because our shim currently exports only the
//! TurboJPEG (`tj3*` / `tj*`) API, not the classic libjpeg API that stock
//! `djpeg`/`cjpeg`/`jpegtran` require. This test documents that gap as a
//! hard assertion so the status is machine-checkable and regression-safe.
//!
//! When the classic API layer is added to the shim, this test should flip
//! to "pass" automatically: the link will succeed, `run.sh` will execute,
//! and the byte-exact assertions below will start firing on real output.
//!
//! Reference: examples/stock_djpeg_cjpeg/COORDINATOR_NOTES.md

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script_dir() -> PathBuf {
    repo_root().join("examples/stock_djpeg_cjpeg")
}

fn build_dir() -> PathBuf {
    script_dir().join("build")
}

/// Guarantee the shim cdylib exists. Running the test harness in release
/// is the supported mode because `build.sh` reads from `target/release`.
fn shim_lib_path() -> Option<PathBuf> {
    let release: PathBuf = repo_root()
        .join("target/release")
        .join(if cfg!(target_os = "macos") {
            "liblibjpeg_turbo_rs_capi.dylib"
        } else if cfg!(target_os = "linux") {
            "liblibjpeg_turbo_rs_capi.so"
        } else {
            return None;
        });
    release.exists().then_some(release)
}

/// Does the host have the C tool chain required to drive `build.sh`?
fn host_has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o: std::process::Output| o.status.success())
        .unwrap_or(false)
}

/// Does the host have the stock libjpeg-turbo CLI tools for the diff half
/// of the pipeline? Needed only if `build.sh` succeeds, but probed early
/// so we can print actionable SKIP messages for local developers.
fn host_has_stock_tools() -> bool {
    helpers::djpeg_path().is_some()
        && helpers::cjpeg_path().is_some()
        && helpers::jpegtran_path().is_some()
}

/// Does `references/libjpeg-turbo/src/` contain the stock C source tree?
/// The submodule must be populated before this test can compile stock
/// tool sources.
fn submodule_populated() -> bool {
    repo_root()
        .join("references/libjpeg-turbo/src/djpeg.c")
        .exists()
}

/// Parse `build.sh`'s link_errors.txt to count the distinct missing
/// symbols reported by the linker. Each `Undefined symbols` block lists
/// symbols on their own line after `"_name", referenced from:`.
fn count_missing_symbols(log_path: &Path) -> usize {
    let Ok(content) = std::fs::read_to_string(log_path) else {
        return 0;
    };
    content
        .lines()
        .filter(|l: &&str| {
            let t: &str = l.trim();
            t.starts_with('"') && t.contains("\", referenced from:")
        })
        .count()
}

/// Core harness: run `build.sh`, then `run.sh` if the build succeeded.
/// Returns `Ok(())` on byte-exact pass, `Err(reason)` otherwise.
fn drive_pipeline() -> Result<(), String> {
    let shim: PathBuf = shim_lib_path().ok_or_else(|| {
        "shim cdylib missing — run `cargo build -p libjpeg-turbo-rs-capi --release`".to_string()
    })?;

    let build_sh: PathBuf = script_dir().join("build.sh");
    if !build_sh.exists() {
        return Err(format!("build.sh not found at {}", build_sh.display()));
    }

    let status: std::process::ExitStatus = Command::new("bash")
        .arg(&build_sh)
        .env("OUT_DIR", build_dir())
        .env("CAPI_TARGET_DIR", shim.parent().unwrap())
        .status()
        .map_err(|e: std::io::Error| format!("failed to spawn build.sh: {e}"))?;

    if !status.success() {
        // Build failed — aggregate the three per-tool link logs into a
        // single actionable error. Each log is populated by build.sh on
        // failure; on success they remain empty / missing.
        // `build.sh` emits `${name}_build.log` on link failure (stderr
        // redirect) alongside an aggregated `link_errors.txt`. Count
        // distinct undefined-symbol lines from the per-tool log.
        let djpeg_log: PathBuf = build_dir().join("djpeg_build.log");
        let cjpeg_log: PathBuf = build_dir().join("cjpeg_build.log");
        let jpegtran_log: PathBuf = build_dir().join("jpegtran_build.log");
        let total_missing: usize = count_missing_symbols(&djpeg_log)
            + count_missing_symbols(&cjpeg_log)
            + count_missing_symbols(&jpegtran_log);
        return Err(format!(
            "stock tool link FAILED (exit={}). Missing-symbol count: {} \
             (djpeg={}, cjpeg={}, jpegtran={}). See \
             examples/stock_djpeg_cjpeg/COORDINATOR_NOTES.md for the \
             missing-symbol inventory and remediation path.",
            status.code().unwrap_or(-1),
            total_missing,
            count_missing_symbols(&djpeg_log),
            count_missing_symbols(&cjpeg_log),
            count_missing_symbols(&jpegtran_log),
        ));
    }

    // Build succeeded — proceed to byte-diff comparison.
    let run_sh: PathBuf = script_dir().join("run.sh");
    let output: std::process::Output = Command::new("bash")
        .arg(&run_sh)
        .env("OUT_DIR", build_dir())
        .output()
        .map_err(|e: std::io::Error| format!("failed to spawn run.sh: {e}"))?;

    let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        return Err(format!(
            "run.sh reported byte-diff failures:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Top-level harness: exercises build + run, asserts byte-exact success.
///
/// Today this test FAILS (by assertion, not by panic) because the shim
/// lacks the classic libjpeg API. The assertion message includes the
/// missing-symbol count so coordinators see the exact gap size without
/// opening the log files.
///
/// Skip behavior:
/// * Non-macOS/Linux host → skip (build.sh is POSIX-only for now).
/// * `cc` missing → skip (cannot build stock tools).
/// * Submodule unpopulated → skip (no stock tool sources to compile).
/// * Shim cdylib missing → skip (run `cargo build --release` first).
#[test]
fn stock_tools_link_against_our_shim() {
    // Platform gate.
    if !(cfg!(target_os = "macos") || cfg!(target_os = "linux")) {
        eprintln!("SKIP: B9-4 link test supports macOS/Linux only");
        return;
    }
    if !host_has_cc() {
        eprintln!("SKIP: no `cc` on host");
        return;
    }
    if !submodule_populated() {
        eprintln!("SKIP: references/libjpeg-turbo submodule unpopulated");
        return;
    }
    if shim_lib_path().is_none() {
        eprintln!(
            "SKIP: shim cdylib missing — run \
             `cargo build -p libjpeg-turbo-rs-capi --release` first"
        );
        return;
    }
    // CI *must* have the stock tools for this to be meaningful.
    if !host_has_stock_tools() {
        if helpers::is_ci() {
            panic!(
                "CI is expected to have stock djpeg/cjpeg/jpegtran but they \
                 are missing from PATH/homebrew — install libjpeg-turbo on \
                 the runner"
            );
        }
        eprintln!("SKIP: stock djpeg/cjpeg/jpegtran not found on host");
        return;
    }

    match drive_pipeline() {
        Ok(()) => {
            // Successful path: build linked + run matched byte-for-byte.
            // No further asserts needed; run.sh returns non-zero on any
            // mismatch, which we already propagated into the Err arm.
        }
        Err(reason) => {
            // Documented failure: surface the missing-symbol count and
            // point at COORDINATOR_NOTES.md for the remediation plan.
            panic!("B9-4 stock-tool link test failed: {reason}");
        }
    }
}

/// Secondary test: assert that our shim, as currently built, does **not**
/// yet export the classic libjpeg API. This is a regression guard that
/// pins today's observed state so any future work that adds `jpeg_*`
/// exports has to affirmatively flip this test.
///
/// It uses `nm` (available on both Linux and macOS) to list exported
/// symbols of the shim cdylib and greps for `jpeg_`-prefixed entries.
#[test]
fn shim_currently_lacks_classic_jpeg_api() {
    let Some(shim) = shim_lib_path() else {
        eprintln!("SKIP: shim cdylib missing");
        return;
    };

    let nm_out: std::process::Output = match Command::new("nm").arg(&shim).output() {
        Ok(o) => o,
        Err(_) => {
            eprintln!("SKIP: `nm` not available on host");
            return;
        }
    };
    if !nm_out.status.success() {
        eprintln!("SKIP: `nm {}` failed", shim.display());
        return;
    }

    let text: String = String::from_utf8_lossy(&nm_out.stdout).into_owned();
    // Count defined (T/D/S = text/data/rodata) symbols that start with
    // `jpeg_` (ignoring macOS underscore prefix). `nm` column 2 is the
    // type letter; uppercase = exported, lowercase = local. Example line:
    //   0000000000012abc T _jpeg_std_error
    let exported_jpeg_syms: Vec<&str> = text
        .lines()
        .filter_map(|l: &str| {
            let mut parts: std::str::SplitWhitespace<'_> = l.split_whitespace();
            let _addr: &str = parts.next()?;
            let ty: &str = parts.next()?;
            let name: &str = parts.next()?;
            let is_exported: bool = matches!(ty, "T" | "D" | "S" | "B" | "R");
            let bare: &str = name.strip_prefix('_').unwrap_or(name);
            (is_exported && bare.starts_with("jpeg_")).then_some(name)
        })
        .collect();

    // Today we expect zero. If this ever becomes non-zero, the other
    // test in this file will simultaneously start passing on its own.
    assert_eq!(
        exported_jpeg_syms.len(),
        0,
        "Shim has started exporting classic libjpeg symbols: {:?}. \
         Update `stock_tools_link_against_our_shim` to require a \
         passing end-to-end link now that the API surface exists.",
        exported_jpeg_syms,
    );
}
