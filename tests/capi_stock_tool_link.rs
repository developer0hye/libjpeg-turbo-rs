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

/// Resolve the cargo target directory. Honors `CARGO_TARGET_DIR`
/// (relative paths are resolved against `repo_root()`) and falls back
/// to `<repo>/target` otherwise. This matches what `cargo build`
/// itself does, so a nested `cargo build` invoked from
/// `shim_lib_path_or_build` writes its artifact to the same place we
/// then look for it.
fn cargo_target_dir() -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(v) => {
            let p: PathBuf = PathBuf::from(v);
            if p.is_absolute() {
                p
            } else {
                repo_root().join(p)
            }
        }
        None => repo_root().join("target"),
    }
}

/// Platform-specific filename for the release cdylib emitted by
/// `cargo build -p libjpeg-turbo-rs-capi --release`. Returns `None`
/// for hosts we don't ship a cdylib for, which lets callers skip
/// rather than panic.
fn shim_release_filename() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("liblibjpeg_turbo_rs_capi.dylib")
    } else if cfg!(target_os = "linux") {
        Some("liblibjpeg_turbo_rs_capi.so")
    } else {
        None
    }
}

/// Guarantee the shim cdylib exists. Running the test harness in release
/// is the supported mode because `build.sh` reads from `target/release`.
/// Honors `CARGO_TARGET_DIR` so a redirected target tree is found.
fn shim_lib_path() -> Option<PathBuf> {
    let release: PathBuf = cargo_target_dir()
        .join("release")
        .join(shim_release_filename()?);
    release.exists().then_some(release)
}

/// Like `shim_lib_path`, but **always** runs `cargo build -p
/// libjpeg-turbo-rs-capi --release` first so the export-guard test
/// validates the *current* source. Without the unconditional rebuild,
/// a stale or restored `target/release/liblibjpeg_turbo_rs_capi.*`
/// (e.g. from a CI cache) could satisfy the existence check and let
/// `nm` inspect bits that no longer reflect the working tree.
fn shim_lib_path_or_build() -> Option<PathBuf> {
    if !cfg!(any(target_os = "macos", target_os = "linux")) {
        return None;
    }
    eprintln!(
        "INFO: rebuilding shim cdylib via `cargo build -p libjpeg-turbo-rs-capi --release` so the export guard inspects current source"
    );
    let status: std::process::ExitStatus = Command::new(env!("CARGO"))
        .args([
            "build",
            "-p",
            "libjpeg-turbo-rs-capi",
            "--release",
            "--quiet",
        ])
        .current_dir(repo_root())
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    shim_lib_path()
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
    // Propagate the resolved target directory to run.sh so the canonical-
    // soname symlinks it stages in $WORK/loader point at the same cdylib
    // build.sh just linked against. Without this, run.sh defaults to
    // `$REPO_ROOT/target/release` and a `CARGO_TARGET_DIR` redirected
    // run would either fail (missing artifact) or quietly load a stale
    // cdylib instead of the freshly rebuilt one from
    // `shim_lib_path_or_build`.
    let run_sh: PathBuf = script_dir().join("run.sh");
    let output: std::process::Output = Command::new("bash")
        .arg(&run_sh)
        .env("OUT_DIR", build_dir())
        .env("SHIM_DIR", shim.parent().unwrap())
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

/// Top-level harness: exercises build + run, asserts byte-exact
/// success across djpeg, cjpeg, AND jpegtran on the
/// `references/libjpeg-turbo/testimages/` corpus.
///
/// All three arms — djpeg, cjpeg, and jpegtran (`-copy all -rotate 90`)
/// — are byte-exact against stock libjpeg-turbo for every fixture in
/// `references/libjpeg-turbo/testimages/`, including the 12-bit
/// `monkey12.jpg`. The 12-bit transcode byte-exactness gate landed
/// once `jpeg_read_header` started populating `cinfo->marker_list`
/// so `transupp::jcopy_markers_execute` could forward the source
/// APP2/ICC chunks verbatim (LAST_MILE.md → Suggested Order item 5b).
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

/// Hard gate: the shim cdylib must export a meaningful set of classic
/// `jpeg_*` symbols. Originally this test pinned the opposite state
/// (zero exported `jpeg_*`) as a regression guard while the shim was
/// being built up; now those symbols exist (decode side A1-11 + encode
/// side C2 batch) and the gate flipped to a positive assertion so any
/// regression that strips them out fails CI immediately.
///
/// Uses `nm` (available on macOS and Linux) and counts defined,
/// exported `jpeg_*` symbols (excluding shim-private `jpeg_capi_test_*`
/// helpers, which are testing affordances rather than the classic API).
#[test]
fn shim_exports_classic_jpeg_api() {
    // Skip on platforms we don't ship a cdylib for; on Linux/macOS,
    // build the shim if missing so the guard always inspects current
    // source. Anything else is a hard failure (we don't allow
    // soft-skipping a regression that strips the classic API).
    if !cfg!(any(target_os = "macos", target_os = "linux")) {
        eprintln!("SKIP: classic-jpeg-api guard supports macOS/Linux only");
        return;
    }
    let shim: PathBuf = shim_lib_path_or_build().expect(
        "shim cdylib could not be built — run \
             `cargo build -p libjpeg-turbo-rs-capi --release` and re-run this test",
    );

    let nm_out: std::process::Output = match Command::new("nm").arg(&shim).output() {
        Ok(o) => o,
        Err(_) => {
            eprintln!("SKIP: `nm` not available on host");
            return;
        }
    };
    assert!(
        nm_out.status.success(),
        "`nm {}` failed — cannot inspect classic API exports",
        shim.display()
    );

    let text: String = String::from_utf8_lossy(&nm_out.stdout).into_owned();
    // Count defined (T/D/S/B/R) symbols that start with `jpeg_`,
    // `jpeg12_`, or `jpeg16_` (the high-precision family that real
    // libjpeg-turbo also ships), ignoring the shim-private
    // `jpeg_capi_test_*` helpers and the platform-specific leading
    // underscore. Example line:
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
            let is_classic_family: bool = (bare.starts_with("jpeg_")
                || bare.starts_with("jpeg12_")
                || bare.starts_with("jpeg16_"))
                && !bare.starts_with("jpeg_capi_test_");
            (is_exported && is_classic_family).then_some(name)
        })
        .collect();

    // The decode-side A1-11 batch (9 symbols) + encode-side C2 batch
    // (24 symbols) + decode extensions C1 batch (12 symbols) gives 45
    // classic entry points today. 30 is a generous floor that catches
    // a regression which strips a meaningful chunk without being
    // brittle against tiny renames.
    assert!(
        exported_jpeg_syms.len() >= 30,
        "Shim is missing classic libjpeg API surface: only {} jpeg_* \
         symbols are exported, expected ≥30. Stock djpeg/cjpeg/jpegtran \
         link will break. Symbols seen: {:?}",
        exported_jpeg_syms.len(),
        exported_jpeg_syms,
    );
    // jpeg_std_error is the canonical entry point any classic libjpeg
    // caller hits first — call it out by name for fast diagnosis.
    let has_std_error: bool = exported_jpeg_syms
        .iter()
        .any(|s| s.strip_prefix('_').unwrap_or(s) == "jpeg_std_error");
    assert!(
        has_std_error,
        "shim must export jpeg_std_error (the canonical libjpeg entry point); \
         got {exported_jpeg_syms:?}"
    );

    // P0-3: explicit symbol-presence guard for the names Pillow, libtiff,
    // and other downstream wrappers resolve via dlsym at load time.
    // Without this, a future refactor that drops a symbol could pass the
    // ≥30 floor while silently re-introducing the loader blocker that
    // motivated `LAST_MILE.md`'s P0-3 ticket. We assert presence (each
    // bound exists in the export table) — behavior tests live in
    // `tests/capi_pillow_compat.rs`.
    let required_p0_3: &[&str] = &[
        // Raw-data entry points (the original libtiff blocker).
        "jpeg_read_raw_data",
        "jpeg12_read_raw_data",
        "jpeg_write_raw_data",
        "jpeg12_write_raw_data",
        // Buffered-image / streaming entry points.
        "jpeg_consume_input",
        "jpeg_input_complete",
        "jpeg_has_multiple_scans",
        "jpeg_start_output",
        "jpeg_finish_output",
        "jpeg_new_colormap",
        // Abort / generic destroy entry points.
        "jpeg_abort",
        "jpeg_abort_compress",
        "jpeg_abort_decompress",
        "jpeg_destroy",
        // Allocation helpers.
        "jpeg_alloc_huff_table",
        "jpeg_alloc_quant_table",
        // Linear-quality compress entry point used by `cjpeg -baseline`.
        "jpeg_set_linear_quality",
    ];
    let missing: Vec<&str> = required_p0_3
        .iter()
        .copied()
        .filter(|want: &&str| {
            !exported_jpeg_syms
                .iter()
                .any(|got: &&str| got.strip_prefix('_').unwrap_or(got) == *want)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "Shim is missing P0-3 classic-API symbols required by \
         downstream wrappers (Pillow / libtiff / etc.): {missing:?}. \
         See docs/LAST_MILE.md → P0-3 for the rationale."
    );
}
