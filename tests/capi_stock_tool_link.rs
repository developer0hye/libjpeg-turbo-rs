//! FFI B9-4: Verify stock djpeg/cjpeg/jpegtran can link against our cdylib.
//!
//! This test drives `examples/stock_djpeg_cjpeg/build.sh` (build stock C
//! tools linked against our shim) and — if that succeeds — `run.sh` (invoke
//! each built tool over `references/libjpeg-turbo/testimages/*.jpg` and
//! diff-compare against the system stock binaries).
//!
//! The current shim exports the required classic and TurboJPEG APIs. The
//! pipeline builds into a fresh output directory and requires byte-exact
//! decode/transform parity plus equivalent decoded output for encode, while
//! companion tests guard the export inventory and compiler failure propagation.
//!
//! Reference: examples/stock_djpeg_cjpeg/COORDINATOR_NOTES.md

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

static RELEASE_SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
const EXECUTABLE_BUSY_RAW_OS_ERROR: i32 = 26;
const EXECUTABLE_BUSY_RETRY_LIMIT: usize = 20;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script_dir() -> PathBuf {
    repo_root().join("examples/stock_djpeg_cjpeg")
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

/// Return the shim from the resolved release directory if it exists.
fn shim_lib_path_in(target_dir: &Path, target: &str) -> Option<PathBuf> {
    let target_component: &std::ffi::OsStr = if target.ends_with(".json") {
        Path::new(target).file_stem()?
    } else {
        std::ffi::OsStr::new(target)
    };
    let release: PathBuf = target_dir
        .join(target_component)
        .join("release")
        .join(shim_release_filename()?);
    release.is_file().then_some(release)
}

fn cargo_host_target(cargo: &Path) -> Result<String, String> {
    let output: std::process::Output = Command::new(cargo)
        .arg("-vV")
        .output()
        .map_err(|error| format!("failed to query Cargo host target: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo -vV failed while resolving the host target with {}",
            output.status
        ));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line: &str| line.strip_prefix("host: "))
        .map(str::to_owned)
        .ok_or_else(|| "cargo -vV did not report a host target".to_owned())
}

fn retry_executable_busy<T, F, W>(mut operation: F, mut wait: W) -> std::io::Result<T>
where
    F: FnMut() -> std::io::Result<T>,
    W: FnMut(),
{
    let mut retry_count: usize = 0;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error)
                if cfg!(unix)
                    && error.raw_os_error() == Some(EXECUTABLE_BUSY_RAW_OS_ERROR)
                    && retry_count < EXECUTABLE_BUSY_RETRY_LIMIT =>
            {
                retry_count += 1;
                wait();
            }
            Err(error) => return Err(error),
        }
    }
}

fn build_release_shim_with(
    cargo: &Path,
    target_dir: &Path,
    target: &str,
) -> Result<PathBuf, String> {
    eprintln!(
        "INFO: rebuilding shim cdylib for target {target} so stock-tool tests inspect current source"
    );
    let run_build = || -> std::io::Result<std::process::ExitStatus> {
        Command::new(cargo)
            .args([
                "build",
                "-p",
                "libjpeg-turbo-rs-capi",
                "--release",
                "--quiet",
                "--target",
                target,
            ])
            .current_dir(repo_root())
            .env("CARGO_TARGET_DIR", target_dir)
            .status()
    };
    // Linux CI can briefly retain a writable handle to a freshly materialized
    // executable. Retry only ETXTBSY; every other spawn error remains an
    // immediate hard failure.
    let status: std::process::ExitStatus = retry_executable_busy(run_build, || {
        std::thread::sleep(std::time::Duration::from_millis(10));
    })
    .map_err(|error| format!("failed to start release shim build: {error}"))?;
    if !status.success() {
        return Err(format!("release shim build failed with {status}"));
    }
    shim_lib_path_in(target_dir, target).ok_or_else(|| {
        format!(
            "release shim build succeeded but no cdylib was found under {}",
            target_dir.join(target).join("release").display()
        )
    })
}

/// Build once per test process so the stock-tool tests inspect the current
/// source without racing two release builds or accepting a cached artifact.
fn build_release_shim() -> Result<PathBuf, String> {
    let cargo: &Path = Path::new(env!("CARGO"));
    let target_dir: PathBuf = cargo_target_dir();
    let target: String = cargo_host_target(cargo)?;
    build_release_shim_with(cargo, &target_dir, &target)
}

fn shim_lib_path_or_build() -> Result<PathBuf, String> {
    RELEASE_SHIM.get_or_init(build_release_shim).clone()
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
/// Returns `Ok(())` when every operation-specific parity check passes.
fn drive_pipeline() -> Result<(), String> {
    let shim: PathBuf = shim_lib_path_or_build()?;

    let build_sh: PathBuf = script_dir().join("build.sh");
    if !build_sh.exists() {
        return Err(format!("build.sh not found at {}", build_sh.display()));
    }
    let build_temp: tempfile::TempDir = tempfile::tempdir()
        .map_err(|error| format!("failed to create fresh stock-tool output directory: {error}"))?;
    let build_dir: &Path = build_temp.path();

    let status: std::process::ExitStatus = Command::new("bash")
        .arg(&build_sh)
        .env("OUT_DIR", build_dir)
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
        let djpeg_log: PathBuf = build_dir.join("djpeg_build.log");
        let cjpeg_log: PathBuf = build_dir.join("cjpeg_build.log");
        let jpegtran_log: PathBuf = build_dir.join("jpegtran_build.log");
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
        .env("OUT_DIR", build_dir)
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

/// Top-level harness: exercises build + run across djpeg, cjpeg, jpegtran,
/// the comment tools, and tjbench on the upstream test-image corpus.
///
/// djpeg and jpegtran (`-copy all -rotate 90`) are byte-exact against stock
/// libjpeg-turbo for every fixture, including 12-bit `monkey12.jpg`. cjpeg is
/// accepted when its bytes match or when stock djpeg decodes both outputs to
/// identical bytes. The runner also requires one COM round-trip and one
/// tjbench smoke result per fixture.
///
/// Skip behavior:
/// * Non-macOS/Linux host → skip (build.sh is POSIX-only for now).
/// * `cc` missing → skip (cannot build stock tools).
/// * Submodule unpopulated → skip (no stock tool sources to compile).
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
            // Successful path: the build linked and run.sh accepted every
            // comparison under its operation-specific parity contract.
        }
        Err(reason) => {
            panic!("B9-4 stock-tool link test failed: {reason}");
        }
    }
}

#[cfg(unix)]
#[test]
fn stock_tool_build_rejects_a_failed_wrapper_rebuild() {
    if !cfg!(any(target_os = "macos", target_os = "linux")) || !submodule_populated() {
        return;
    }

    let shim: PathBuf = shim_lib_path_or_build().expect("build current shim");
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let fake_cc: PathBuf = temp.path().join("fake-cc");
    std::fs::write(
        &fake_cc,
        r#"#!/bin/sh
set -eu
output=
previous=
for argument in "$@"; do
    case "$argument" in
        */wrapper/*.c) exit 19 ;;
    esac
    if [ "$previous" = "-o" ]; then
        output="$argument"
    fi
    previous="$argument"
done
test -n "$output"
: > "$output"
"#,
    )
    .expect("write fake compiler");
    let mut permissions: std::fs::Permissions = std::fs::metadata(&fake_cc)
        .expect("stat fake compiler")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&fake_cc, permissions).expect("make fake compiler executable");

    let output: std::process::Output = Command::new("bash")
        .arg(script_dir().join("build.sh"))
        .env("OUT_DIR", temp.path().join("build"))
        .env(
            "CAPI_TARGET_DIR",
            shim.parent().expect("shim release directory"),
        )
        .env("CC", &fake_cc)
        .output()
        .expect("run stock-tool build with failing wrapper compiler");

    assert!(
        !output.status.success(),
        "a wrapper compile failure must not fall through to stale objects"
    );
}

#[cfg(unix)]
#[test]
fn nested_shim_build_and_probe_share_the_target_qualified_tree() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let target_dir: PathBuf = temp.path().join("target");
    let target: &str = "x86_64-unknown-linux-gnu";
    let fake_cargo: PathBuf = temp.path().join("cargo");
    let captured_args: PathBuf = temp.path().join("cargo-args");
    let Some(artifact_name): Option<&str> = shim_release_filename() else {
        eprintln!("SKIP: stock-tool cdylib regression supports macOS/Linux only");
        return;
    };
    let script: String = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > '{}'\nmkdir -p \"$CARGO_TARGET_DIR/{target}/release\"\n: > \"$CARGO_TARGET_DIR/{target}/release/{artifact_name}\"\n",
        captured_args.display()
    );
    std::fs::write(&fake_cargo, script).expect("write fake cargo");
    let mut permissions: std::fs::Permissions = std::fs::metadata(&fake_cargo)
        .expect("stat fake cargo")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&fake_cargo, permissions).expect("make fake cargo executable");

    let artifact: PathBuf =
        build_release_shim_with(&fake_cargo, &target_dir, target).expect("fake shim build");

    assert_eq!(
        artifact,
        target_dir.join(target).join("release").join(artifact_name)
    );
    let args: String = std::fs::read_to_string(captured_args).expect("read cargo args");
    assert!(
        args.lines()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|pair| pair == ["--target", target]),
        "the nested build must pin the target used by its artifact probe"
    );
}

#[cfg(unix)]
#[test]
fn executable_busy_retry_is_selective_and_bounded() {
    let mut transient_attempts: usize = 0;
    let mut transient_waits: usize = 0;
    let value: usize = retry_executable_busy(
        || {
            transient_attempts += 1;
            if transient_attempts <= 3 {
                Err(std::io::Error::from_raw_os_error(
                    EXECUTABLE_BUSY_RAW_OS_ERROR,
                ))
            } else {
                Ok(17)
            }
        },
        || transient_waits += 1,
    )
    .expect("transient ETXTBSY must be retried");
    assert_eq!(value, 17);
    assert_eq!(transient_attempts, 4);
    assert_eq!(transient_waits, 3);

    let mut other_attempts: usize = 0;
    let other_error: std::io::Error = retry_executable_busy(
        || -> std::io::Result<()> {
            other_attempts += 1;
            Err(std::io::Error::from_raw_os_error(2))
        },
        || panic!("non-ETXTBSY errors must not wait or retry"),
    )
    .expect_err("non-ETXTBSY spawn errors must remain hard failures");
    assert_eq!(other_error.raw_os_error(), Some(2));
    assert_eq!(other_attempts, 1);

    let mut bounded_attempts: usize = 0;
    let mut bounded_waits: usize = 0;
    let bounded_error: std::io::Error = retry_executable_busy(
        || -> std::io::Result<()> {
            bounded_attempts += 1;
            Err(std::io::Error::from_raw_os_error(
                EXECUTABLE_BUSY_RAW_OS_ERROR,
            ))
        },
        || bounded_waits += 1,
    )
    .expect_err("persistent ETXTBSY must fail after the retry budget");
    assert_eq!(
        bounded_error.raw_os_error(),
        Some(EXECUTABLE_BUSY_RAW_OS_ERROR)
    );
    assert_eq!(bounded_attempts, EXECUTABLE_BUSY_RETRY_LIMIT + 1);
    assert_eq!(bounded_waits, EXECUTABLE_BUSY_RETRY_LIMIT);
}

#[cfg(unix)]
fn make_executable(path: &Path, contents: &str) {
    std::fs::write(path, contents).expect("write fake executable");
    let mut permissions: std::fs::Permissions = std::fs::metadata(path)
        .expect("stat fake executable")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).expect("make fake executable executable");
}

#[cfg(unix)]
fn compile_loader_probe_copy_tool(temp: &tempfile::TempDir) -> PathBuf {
    let source: PathBuf = temp.path().join("loader_probe_copy.c");
    let executable: PathBuf = temp.path().join("loader-probe-copy");
    std::fs::write(
        &source,
        r#"#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *ld = getenv("LD_LIBRARY_PATH");
    const char *dyld = getenv("DYLD_LIBRARY_PATH");
    const char *soname = getenv("EXPECTED_SONAME");
    const char *expected_shim = getenv("EXPECTED_SHIM");
    if (getenv("LD_PRELOAD") || getenv("DYLD_INSERT_LIBRARIES")) return 90;
    if (!ld || !dyld || strcmp(ld, dyld) != 0) return 91;
    size_t loader_len = strlen(ld);
    if (loader_len < 7 || strcmp(ld + loader_len - 7, "/loader") != 0) return 92;
    if (!soname || !expected_shim) return 93;

    char link_path[PATH_MAX];
    if (snprintf(link_path, sizeof(link_path), "%s/%s", ld, soname) >= (int)sizeof(link_path)) return 94;
    char target[PATH_MAX];
    ssize_t target_len = readlink(link_path, target, sizeof(target) - 1);
    if (target_len < 0) return 95;
    target[target_len] = '\0';
    if (strcmp(target, expected_shim) != 0) return 96;

    const char *output = NULL;
    const char *input = argc > 1 ? argv[argc - 1] : NULL;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-outfile") == 0) output = argv[i + 1];
    }
    if (!input || !output) return 97;
    FILE *src = fopen(input, "rb");
    FILE *dst = fopen(output, "wb");
    if (!src || !dst) return 98;
    char buffer[4096];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), src)) != 0) {
        if (fwrite(buffer, 1, count, dst) != count) return 99;
    }
    int src_status = fclose(src);
    int dst_status = fclose(dst);
    return src_status == 0 && dst_status == 0 ? 0 : 100;
}
"#,
    )
    .expect("write native loader probe source");
    let compiler: std::ffi::OsString =
        std::env::var_os("CC").unwrap_or_else(|| std::ffi::OsString::from("cc"));
    let output: std::process::Output = Command::new(compiler)
        .arg(&source)
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("compile native loader probe");
    assert!(
        output.status.success(),
        "native loader probe must compile:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    executable
}

#[cfg(unix)]
fn loader_contamination_library(temp: &tempfile::TempDir) -> PathBuf {
    if cfg!(target_os = "macos") {
        // Apple Silicon system utilities use the arm64e ABI while test-built
        // artifacts use arm64.  libSystem is present in dyld's shared cache
        // for both, so it can contaminate setup commands without an ABI trap.
        return PathBuf::from("/usr/lib/libSystem.B.dylib");
    }
    let source: PathBuf = temp.path().join("loader_contamination.c");
    let library: PathBuf = temp.path().join("libloader_contamination.so");
    std::fs::write(&source, "void loader_contamination_marker(void) {}\n")
        .expect("write loader contamination source");
    let compiler: std::ffi::OsString =
        std::env::var_os("CC").unwrap_or_else(|| std::ffi::OsString::from("cc"));
    let mut command: Command = Command::new(compiler);
    command.args(["-shared", "-fPIC"]);
    let output: std::process::Output = command
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .expect("compile loader contamination library");
    assert!(
        output.status.success(),
        "loader contamination library must compile:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    library
}

#[cfg(unix)]
fn fake_stock_runner(temp: &tempfile::TempDir, testimages: &Path) -> std::process::Output {
    let our_build: PathBuf = temp.path().join("our-build");
    let stock_bin: PathBuf = temp.path().join("stock-bin");
    let shim_dir: PathBuf = temp.path().join("shim");
    std::fs::create_dir_all(&our_build).expect("create fake our-build");
    std::fs::create_dir_all(&stock_bin).expect("create fake stock-bin");
    std::fs::create_dir_all(&shim_dir).expect("create fake shim directory");

    let copy_tool: &str = r#"#!/bin/sh
set -eu
output=
previous=
input=
for argument in "$@"; do
    if [ "$previous" = "-outfile" ]; then
        output="$argument"
    fi
    previous="$argument"
    input="$argument"
done
test -n "$output"
cp "$input" "$output"
"#;
    for name in ["djpeg", "cjpeg", "jpegtran"] {
        make_executable(&our_build.join(name), copy_tool);
        make_executable(&stock_bin.join(name), copy_tool);
    }
    make_executable(&our_build.join("tjbench"), "#!/bin/sh\nexit 0\n");
    make_executable(
        &our_build.join("wrjpgcom"),
        "#!/bin/sh\nset -eu\nfor argument in \"$@\"; do input=\"$argument\"; done\ncat \"$input\"\n",
    );
    make_executable(
        &our_build.join("rdjpgcom"),
        "#!/bin/sh\nset -eu\nname=$(basename \"$1\" .ours.com.jpg)\nprintf 'ljt-test-%s\\n' \"$name\"\n",
    );

    let shim_name: &str = if cfg!(target_os = "macos") {
        "liblibjpeg_turbo_rs_capi.dylib"
    } else {
        "liblibjpeg_turbo_rs_capi.so"
    };
    std::fs::write(shim_dir.join(shim_name), []).expect("write fake shim");

    Command::new("bash")
        .arg(script_dir().join("run.sh"))
        .env("OUT_DIR", our_build)
        .env("STOCK_BIN", stock_bin)
        .env("SHIM_DIR", &shim_dir)
        .env("TESTIMAGES", testimages)
        .output()
        .expect("run stock-tool harness with fake tools")
}

#[cfg(unix)]
#[test]
fn stock_tool_runner_rejects_an_empty_fixture_directory() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let empty_images: PathBuf = temp.path().join("empty-images");
    std::fs::create_dir(&empty_images).expect("create empty fixture directory");

    let output: std::process::Output = fake_stock_runner(&temp, &empty_images);

    assert!(
        !output.status.success(),
        "zero fixtures must not report green"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no JPEG fixtures"),
        "failure must identify the vacuous fixture matrix: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn stock_tool_runner_rejects_a_failed_stock_oracle() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let testimages: PathBuf = temp.path().join("testimages");
    std::fs::create_dir(&testimages).expect("create fixture directory");
    std::fs::write(testimages.join("fixture.jpg"), b"fake-jpeg").expect("write fake fixture");

    let stock_djpeg: PathBuf = temp.path().join("stock-djpeg-fails");
    make_executable(&stock_djpeg, "#!/bin/sh\nexit 29\n");

    let our_build: PathBuf = temp.path().join("our-build");
    let stock_bin: PathBuf = temp.path().join("stock-bin");
    let shim_dir: PathBuf = temp.path().join("shim");
    let _ = fake_stock_runner(&temp, &testimages);
    let output: std::process::Output = Command::new("bash")
        .arg(script_dir().join("run.sh"))
        .env("OUT_DIR", our_build)
        .env("STOCK_BIN", stock_bin)
        .env("STOCK_DJPEG", stock_djpeg)
        .env("SHIM_DIR", shim_dir)
        .env("TESTIMAGES", testimages)
        .output()
        .expect("run stock-tool harness with failed stock oracle");

    assert!(
        !output.status.success(),
        "a committed fixture whose stock oracle fails must not be skipped green"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("stock_failed"),
        "runner must identify the failed stock oracle"
    );
}

#[cfg(unix)]
#[test]
fn stock_tool_runner_rejects_failed_partial_fallback_decodes() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let testimages: PathBuf = temp.path().join("testimages");
    std::fs::create_dir(&testimages).expect("create fixture directory");
    std::fs::write(testimages.join("fixture.jpg"), b"fake-jpeg").expect("write fake fixture");

    let _ = fake_stock_runner(&temp, &testimages);
    let our_build: PathBuf = temp.path().join("our-build");
    let stock_bin: PathBuf = temp.path().join("stock-bin");
    let shim_dir: PathBuf = temp.path().join("shim");
    make_executable(
        &our_build.join("cjpeg"),
        "#!/bin/sh\nset -eu\noutput=\nprevious=\nfor argument in \"$@\"; do\n  if [ \"$previous\" = -outfile ]; then output=\"$argument\"; fi\n  previous=\"$argument\"\ndone\nprintf ours > \"$output\"\n",
    );
    make_executable(
        &stock_bin.join("cjpeg"),
        "#!/bin/sh\nset -eu\noutput=\nprevious=\nfor argument in \"$@\"; do\n  if [ \"$previous\" = -outfile ]; then output=\"$argument\"; fi\n  previous=\"$argument\"\ndone\nprintf stock > \"$output\"\n",
    );
    make_executable(
        &stock_bin.join("djpeg"),
        r#"#!/bin/sh
set -eu
output=
previous=
input=
for argument in "$@"; do
    if [ "$previous" = -outfile ]; then output="$argument"; fi
    previous="$argument"
    input="$argument"
done
case "$input" in
    */testimages/*) cp "$input" "$output" ;;
    *) : > "$output"; exit 29 ;;
esac
"#,
    );

    let output: std::process::Output = Command::new("bash")
        .arg(script_dir().join("run.sh"))
        .env("OUT_DIR", our_build)
        .env("STOCK_BIN", stock_bin)
        .env("SHIM_DIR", &shim_dir)
        .env("TESTIMAGES", testimages)
        .output()
        .expect("run stock-tool harness with partial fallback output");

    assert!(
        !output.status.success(),
        "failed fallback decoders must not pass because their partial files compare equal"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("stock_roundtrip_failed"),
        "runner must report the failed decoded-output oracle: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[cfg(unix)]
#[test]
fn stock_oracles_do_not_inherit_loader_path_overrides() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let testimages: PathBuf = temp.path().join("testimages");
    std::fs::create_dir(&testimages).expect("create fixture directory");
    std::fs::write(testimages.join("fixture.jpg"), b"fake-jpeg").expect("write fake fixture");

    let _ = fake_stock_runner(&temp, &testimages);
    let our_build: PathBuf = temp.path().join("our-build");
    let stock_bin: PathBuf = temp.path().join("stock-bin");
    let shim_dir: PathBuf = temp.path().join("shim");
    let clean_loader_copy_tool: &str = r#"#!/bin/sh
set -eu
test -z "${LD_LIBRARY_PATH+x}"
test -z "${DYLD_LIBRARY_PATH+x}"
test -z "${LD_PRELOAD+x}"
test -z "${DYLD_INSERT_LIBRARIES+x}"
output=
previous=
input=
for argument in "$@"; do
    if [ "$previous" = -outfile ]; then output="$argument"; fi
    previous="$argument"
    input="$argument"
done
cp "$input" "$output"
"#;
    for name in ["djpeg", "cjpeg", "jpegtran"] {
        make_executable(&stock_bin.join(name), clean_loader_copy_tool);
    }
    // A script interpreter under /bin is SIP-protected on macOS and strips
    // DYLD_* before the script can inspect it. A temporary native executable
    // observes the environment received from `run_ours` on both macOS/Linux.
    let isolated_ours_copy_tool: PathBuf = compile_loader_probe_copy_tool(&temp);
    for name in ["djpeg", "cjpeg", "jpegtran"] {
        std::fs::copy(&isolated_ours_copy_tool, our_build.join(name))
            .expect("install native loader probe as fake tool");
    }
    make_executable(
        &our_build.join("tjbench"),
        "#!/bin/sh\nset -eu\ntest -z \"${LD_PRELOAD+x}\"\ntest -z \"${DYLD_INSERT_LIBRARIES+x}\"\n",
    );
    let loader_contamination_library: PathBuf = loader_contamination_library(&temp);

    // Start Bash before injecting loader overrides.  On macOS, launching Bash
    // with DYLD_INSERT_LIBRARIES terminates before run.sh can exercise its
    // per-command isolation unless the injected library is loadable.
    let output: std::process::Output = Command::new("bash")
        .arg("-c")
        .arg(
            r#"export LD_LIBRARY_PATH="$2"
export DYLD_LIBRARY_PATH="$3"
export LD_PRELOAD="$4"
export DYLD_INSERT_LIBRARIES="$5"
source "$1""#,
        )
        .arg("loader-isolation-test")
        .arg(script_dir().join("run.sh"))
        .arg("/ambient/rust-shim")
        .arg("/ambient/rust-shim")
        .arg(&loader_contamination_library)
        .arg(&loader_contamination_library)
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("DYLD_LIBRARY_PATH")
        .env_remove("LD_PRELOAD")
        .env_remove("DYLD_INSERT_LIBRARIES")
        .env("OUT_DIR", our_build)
        .env("STOCK_BIN", stock_bin)
        .env("SHIM_DIR", &shim_dir)
        .env("TESTIMAGES", testimages)
        .env(
            "EXPECTED_SONAME",
            if cfg!(target_os = "macos") {
                "libjpeg.8.dylib"
            } else {
                "libjpeg.so.8"
            },
        )
        .env(
            "EXPECTED_SHIM",
            shim_dir.join(if cfg!(target_os = "macos") {
                "liblibjpeg_turbo_rs_capi.dylib"
            } else {
                "liblibjpeg_turbo_rs_capi.so"
            }),
        )
        .output()
        .expect("run stock-tool harness with ambient loader overrides");

    assert!(
        output.status.success(),
        "stock oracles must run with loader overrides removed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn stock_tool_runner_records_all_cases_after_our_djpeg_fails() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let testimages: PathBuf = temp.path().join("testimages");
    std::fs::create_dir(&testimages).expect("create fixture directory");
    std::fs::write(testimages.join("fixture.jpg"), b"fake-jpeg").expect("write fake fixture");

    let _ = fake_stock_runner(&temp, &testimages);
    let our_build: PathBuf = temp.path().join("our-build");
    let stock_bin: PathBuf = temp.path().join("stock-bin");
    let shim_dir: PathBuf = temp.path().join("shim");
    make_executable(&our_build.join("djpeg"), "#!/bin/sh\nexit 19\n");

    let output: std::process::Output = Command::new("bash")
        .arg(script_dir().join("run.sh"))
        .env("OUT_DIR", our_build)
        .env("STOCK_BIN", stock_bin)
        .env("SHIM_DIR", shim_dir)
        .env("TESTIMAGES", testimages)
        .output()
        .expect("run stock-tool harness with failed Rust djpeg");

    assert!(!output.status.success(), "failed Rust djpeg must fail");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("cases=5"),
        "one failed operation must not abort the rest of the matrix:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("comtools\tfixture\tfail\tencode_input_missing"),
        "a missing encode input must be recorded, not reused from another fixture"
    );
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
