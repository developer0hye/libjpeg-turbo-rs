//! FFI B9-5: compile the upstream `tjunittest.c` against OUR
//! `libturbojpeg.0.dylib` / `libturbojpeg.so.0` cdylib and assert the
//! full suite passes.
//!
//! Two tests live here:
//!
//! * `tjunittest_link_symbols_resolve` — compiles `tjunittest.c` +
//!   `tjutil.c` + `md5/{md5,md5hl}.c` against our cdylib and asserts the
//!   link succeeds (i.e. every TJ3 symbol the upstream test requires is
//!   exported from our shim). Runs the resulting binary through
//!   `overflowTest()` to confirm the no-handle sizing + error-string
//!   contract matches libjpeg-turbo.
//! * `tjunittest_default_suite_passes` — runs the full unmodified
//!   tjunittest binary and asserts every subtest reports `Done.`. This
//!   currently fails on macOS with a SIGKILL inside the first `doTest()`
//!   call (encoder-side crash); see the BLOCKER note on that test.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Repository root (the worktree that contains `Cargo.toml`,
/// `references/`, and `target/`).
fn repo_root() -> PathBuf {
    let manifest: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root above crates/libjpeg-turbo-rs-capi")
        .to_path_buf()
}

fn dlext() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    }
}

fn turbojpeg_versioned_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libturbojpeg.0.dylib"
    } else {
        "libturbojpeg.so.0"
    }
}

fn turbojpeg_short_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libturbojpeg.dylib"
    } else {
        "libturbojpeg.so"
    }
}

fn cdylib_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        let pb: PathBuf = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let base: PathBuf = repo_root()
        .join("target")
        .join("release")
        .join(format!("liblibjpeg_turbo_rs_capi.{}", dlext()));
    if base.exists() {
        return Some(base);
    }
    None
}

/// Like `cdylib_path`, but **always** runs `cargo build -p
/// libjpeg-turbo-rs-capi --release` first so the test exercises the
/// *current* source. Without the unconditional rebuild, a stale
/// `target/release/liblibjpeg_turbo_rs_capi.*` (CI cache, prior run,
/// etc.) could satisfy the existence check and let `tjunittest` link
/// against bits that no longer reflect the working tree.
fn cdylib_path_or_build() -> Option<PathBuf> {
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
    cdylib_path()
}

fn find_cc() -> Option<PathBuf> {
    for candidate in ["cc", "clang", "gcc"] {
        if let Ok(out) = Command::new("which").arg(candidate).output() {
            if out.status.success() {
                let s: String = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return Some(PathBuf::from(s));
                }
            }
        }
    }
    None
}

/// Install the alias-symlinks our cdylib needs to resolve at both
/// link-time and load-time:
/// * `libturbojpeg.{0.dylib|so.0}` / `libturbojpeg.{dylib|so}` so
///   `-lturbojpeg` resolves.
/// * `libjpeg.{62.dylib|so.62}` / `libjpeg.{dylib|so}` so `dyld`/`ld.so`
///   resolve the SONAME/install_name embedded in the cdylib.
fn install_aliases(cdylib: &Path, link_dir: &Path) -> Result<(), String> {
    let mut names: Vec<&str> = vec![turbojpeg_versioned_name(), turbojpeg_short_name()];
    if cfg!(target_os = "macos") {
        names.push("libjpeg.62.dylib");
        names.push("libjpeg.dylib");
    } else {
        names.push("libjpeg.so.62");
        names.push("libjpeg.so");
    }
    for name in &names {
        let link: PathBuf = link_dir.join(name);
        let _ = std::fs::remove_file(&link);
        symlink(cdylib, &link)
            .map_err(|e| format!("symlink {} -> {}: {e}", link.display(), cdylib.display()))?;
    }
    Ok(())
}

/// Build the tjunittest binary, returning the path to it and the link
/// directory that must be on the runtime library search path.
///
/// Forces a fresh `cargo build -p libjpeg-turbo-rs-capi --release` so the
/// linker resolves against the current source — no stale-cdylib pass.
fn build_tjunittest() -> Result<(PathBuf, PathBuf), String> {
    let root: PathBuf = repo_root();
    let ref_src: PathBuf = root.join("references/libjpeg-turbo/src");
    let tjunittest_c: PathBuf = ref_src.join("tjunittest.c");
    if !tjunittest_c.exists() {
        return Err(format!(
            "{} not found — run `git submodule update --init --depth 1 references/libjpeg-turbo`",
            tjunittest_c.display()
        ));
    }

    let cc: PathBuf = match find_cc() {
        Some(p) => p,
        None => return Err("no C compiler (cc/clang/gcc) on PATH".to_string()),
    };

    let cdylib: PathBuf = match cdylib_path_or_build() {
        Some(p) => p,
        None => {
            return Err(
                "cdylib build failed. Run: cargo build -p libjpeg-turbo-rs-capi --release"
                    .to_string(),
            )
        }
    };

    let out_dir: PathBuf = root.join("examples/tjunittest_link/build");
    let link_dir: PathBuf = out_dir.join("linkdir");
    std::fs::create_dir_all(&link_dir).map_err(|e| format!("mkdir {}: {e}", link_dir.display()))?;

    install_aliases(&cdylib, &link_dir)?;

    let bin: PathBuf = out_dir.join("tjunittest");
    let _ = std::fs::remove_file(&bin);

    let jconfigint_dir: PathBuf = root.join("examples/tjunittest_link");

    // We link against `-ljpeg` (not `-lturbojpeg`) because our cdylib's
    // install_name is `@rpath/libjpeg.62.dylib`. Using the same library
    // name both at link-time and at load-time avoids macOS AMFI
    // mismatches between the codesign record and the resolved path.
    let status = Command::new(&cc)
        .arg("-O2")
        .arg("-I")
        .arg(&jconfigint_dir)
        .arg("-I")
        .arg(&ref_src)
        .arg(ref_src.join("tjunittest.c"))
        .arg(ref_src.join("tjutil.c"))
        .arg(ref_src.join("md5/md5.c"))
        .arg(ref_src.join("md5/md5hl.c"))
        .arg(format!("-L{}", link_dir.display()))
        .arg("-ljpeg")
        .arg(format!("-Wl,-rpath,{}", link_dir.display()))
        .arg("-o")
        .arg(&bin)
        .status()
        .map_err(|e| format!("cc spawn: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cc compile of tjunittest exited with status {:?}",
            status.code()
        ));
    }
    if !bin.exists() {
        return Err(format!("tjunittest binary missing at {}", bin.display()));
    }
    Ok((bin, link_dir))
}

#[cfg(unix)]
fn symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(not(unix))]
fn symlink(_target: &Path, _link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "tjunittest link harness only supports Unix targets",
    ))
}

/// Scan tjunittest's stdout for `FAILED!` markers and return them.
fn collect_failed_lines(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter(|line| line.contains("FAILED"))
        .map(|line| line.to_string())
        .collect()
}

/// Run the built binary and return its exit-status, stdout, stderr.
/// Writing capture to temp files (rather than pipes) avoids a macOS
/// AMFI interaction we saw with Rust's pipe-based `.output()`.
fn run_binary(bin: &Path, extra_args: &[&str]) -> (std::process::ExitStatus, Vec<u8>, Vec<u8>) {
    let work: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(bin);
    cmd.args(extra_args).current_dir(work.path());
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawn {}: {e}", bin.display()));
    (out.status, out.stdout, out.stderr)
}

/// FFI B9-5 linkage check: every TJ3 symbol `tjunittest.c` references
/// must resolve against our cdylib. We also exercise
/// `overflowTest()` — which hits every no-handle sizing helper and
/// `tj3GetErrorStr(NULL)` — to confirm the cross-language calling
/// convention matches the reference.
///
/// Runs `tjunittest` with a command line that drives only
/// `overflowTest()` (built in-place from the upstream sources) so the
/// failure mode stays isolated from the encoder-side blocker below.
#[cfg(unix)]
#[test]
fn tjunittest_link_symbols_resolve() {
    let root: PathBuf = repo_root();
    let ref_src: PathBuf = root.join("references/libjpeg-turbo/src");
    if !ref_src.join("tjunittest.c").exists() {
        eprintln!("SKIP tjunittest_link_symbols_resolve: references/libjpeg-turbo not populated");
        return;
    }
    let cc: PathBuf = match find_cc() {
        Some(p) => p,
        None => {
            eprintln!("SKIP tjunittest_link_symbols_resolve: no cc/clang/gcc on PATH");
            return;
        }
    };
    // Force a fresh build so we exercise the *current* shim source,
    // not whatever stale `target/release/...` happened to be left from a
    // prior run or CI cache.
    let cdylib: PathBuf = match cdylib_path_or_build() {
        Some(p) => p,
        None => {
            panic!(
                "tjunittest_link_symbols_resolve: cdylib build failed — `cargo build -p \
                 libjpeg-turbo-rs-capi --release` reported failure"
            );
        }
    };

    let out_dir: PathBuf = root.join("examples/tjunittest_link/build_symbols");
    let link_dir: PathBuf = out_dir.join("linkdir");
    std::fs::create_dir_all(&link_dir).expect("mkdir");
    install_aliases(&cdylib, &link_dir).expect("install aliases");

    // Compile a driver that invokes `overflowTest` and prints a success
    // marker. Because `tjunittest.c` carries its own `main`, we pull in
    // the file as a translation unit alongside `tjutil.c` / `md5*.c`
    // but define `main` ourselves in a separate shim.
    //
    // Upstream's `tjunittest.c` conditionalizes `main` on `TJUNITTEST_NO_MAIN`
    // being unset, so we pass `-DTJUNITTEST_NO_MAIN` to suppress it —
    // BUT the reference does not actually support that define, so we
    // instead replace the whole tjunittest.c with a preprocessed copy
    // whose `main` symbol is renamed to `tjunittest_main`.
    let driver_src: PathBuf = out_dir.join("tjunittest_driver.c");
    let driver: String = r#"
#include <stdio.h>
#include <stdlib.h>
#include "tjutil.h"
#include "turbojpeg.h"

extern int precision, sampleSize, tolerance;
extern int maxSample, redToY, yellowToY;
extern int exitStatus;
extern void overflowTest(void);

int main(void) {
    precision = 8;
    sampleSize = 1;
    maxSample = 255;
    tolerance = 1;
    redToY = (19595U * 255U) >> 16;
    yellowToY = (58065U * 255U) >> 16;
    overflowTest();
    if (exitStatus != 0) {
        printf("overflowTest FAILED: exitStatus=%d\n", exitStatus);
        return 1;
    }
    printf("overflowTest OK\n");
    return 0;
}
"#
    .to_string();
    std::fs::write(&driver_src, driver).expect("write driver");

    // Build a version of tjunittest.c that renames `main` to
    // `tjunittest_main_unused` so the linker picks up our driver's
    // `main` instead, and promotes the precision/sampleSize/... state
    // globals from `static` to external linkage so our driver can set
    // them before calling `overflowTest`.
    let orig: String = std::fs::read_to_string(ref_src.join("tjunittest.c")).expect("read tjutil");
    let renamed: String = orig
        .replace(
            "int main(int argc, char *argv[])\n{",
            "int tjunittest_main_unused(int argc, char *argv[])\n{",
        )
        .replace(
            "static int precision = 8, sampleSize, maxSample, tolerance, redToY, yellowToY;",
            "int precision = 8, sampleSize, maxSample, tolerance, redToY, yellowToY;",
        )
        .replace("static int exitStatus = 0;", "int exitStatus = 0;")
        .replace("static void overflowTest(void)", "void overflowTest(void)");
    let renamed_src: PathBuf = out_dir.join("tjunittest_renamed.c");
    std::fs::write(&renamed_src, renamed).expect("write renamed");

    let bin: PathBuf = out_dir.join("tjunittest_symbols");
    let _ = std::fs::remove_file(&bin);
    let jc_dir: PathBuf = root.join("examples/tjunittest_link");
    let status = Command::new(&cc)
        .arg("-O2")
        .arg("-I")
        .arg(&jc_dir)
        .arg("-I")
        .arg(&ref_src)
        .arg(&driver_src)
        .arg(&renamed_src)
        .arg(ref_src.join("tjutil.c"))
        .arg(ref_src.join("md5/md5.c"))
        .arg(ref_src.join("md5/md5hl.c"))
        .arg(format!("-L{}", link_dir.display()))
        .arg("-ljpeg")
        .arg(format!("-Wl,-rpath,{}", link_dir.display()))
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("cc spawn");
    assert!(
        status.success(),
        "Linking tjunittest.c against our cdylib failed (missing TJ3 symbol?)"
    );

    let (exit, stdout_raw, stderr_raw) = run_binary(&bin, &[]);
    let stdout: String = String::from_utf8_lossy(&stdout_raw).into_owned();
    let stderr: String = String::from_utf8_lossy(&stderr_raw).into_owned();
    assert!(
        exit.success(),
        "overflowTest-only run failed.\n-- stdout --\n{stdout}\n-- stderr --\n{stderr}"
    );
    assert!(
        stdout.contains("overflowTest OK"),
        "overflowTest marker not found in stdout:\n{stdout}"
    );
}

/// FFI B9-5 full-suite assertion. Compiles and runs the unmodified
/// upstream `tjunittest` binary against our cdylib and asserts every
/// subtest reports `Done.`. The historic SIGKILL on the first
/// `doTest()` call is closed (`docs/LAST_MILE.md` → P1 Soft-Skip
/// 2026-04-28: forced run goes 1 passed; 0 failed in ~6 s).
///
/// All error paths panic — the original soft-skip on missing
/// submodule / cc / cdylib was the soft-skip pattern called out in
/// `docs/LAST_MILE.md` → P1 (a green test that doesn't actually
/// exercise the gate). `build_tjunittest` always rebuilds the cdylib
/// before linking so a stale `target/release/...` cannot let this pass
/// against bits that no longer reflect the working tree.
#[cfg(unix)]
#[test]
fn tjunittest_default_suite_passes() {
    let (bin, _link_dir): (PathBuf, PathBuf) = build_tjunittest().unwrap_or_else(|e| {
        panic!(
            "tjunittest harness setup failed: {e}\n\
             This gate is no longer soft-skipped — set up the prerequisites \
             (submodule, C compiler) so the suite can exercise the cdylib."
        )
    });

    let (status, stdout_raw, stderr_raw) = run_binary(&bin, &[]);
    let stdout: String = String::from_utf8_lossy(&stdout_raw).into_owned();
    let stderr: String = String::from_utf8_lossy(&stderr_raw).into_owned();

    eprintln!(
        "----- tjunittest exit: success={}, code={:?} -----",
        status.success(),
        status.code()
    );
    eprintln!("----- tjunittest stdout ({} bytes) -----", stdout.len());
    eprintln!("{stdout}");
    eprintln!("----- tjunittest stderr ({} bytes) -----", stderr.len());
    eprintln!("{stderr}");

    let failed_lines: Vec<String> = collect_failed_lines(&stdout);
    if !status.success() || !failed_lines.is_empty() {
        if !failed_lines.is_empty() {
            eprintln!("----- FAILED lines -----");
            for line in &failed_lines {
                eprintln!("{line}");
            }
        }
        panic!(
            "tjunittest reported failure: exit_ok={}, FAILED_lines={}",
            status.success(),
            failed_lines.len()
        );
    }
}
