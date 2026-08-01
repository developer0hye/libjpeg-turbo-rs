//! FFI B9-2: Pillow-against-libjpeg-turbo-rs-capi compatibility check.
//!
//! Shells out to `examples/pillow_smoke/run.sh`, which:
//!   1. rebuilds the shim in the target-qualified `CARGO_TARGET_DIR` release tree
//!   2. symlinks it as `libjpeg.so.62` / `libjpeg.62.dylib`
//!   3. spins up a Pillow venv and runs `test_pillow.py` against our shim
//!
//! Exit-code contract (see `test_pillow.py`):
//!   0 → PASS
//!   2 → SKIP (python/Pillow/fixture/network not available)
//!   3 → BLOCKER (shim build/load or Pillow round-trip failed)
//!   1 → FAIL  (shim loaded but output was wrong)
//!
//! A missing optional Python/Pillow environment is a SKIP. Any build, loader,
//! symbol, or round-trip blocker after that gate is a hard test failure.

use std::path::PathBuf;
use std::process::Command;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = this worktree root (the Cargo.toml with `tests/`).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn pillow_links_against_rust_shim_and_roundtrips_jpeg() {
    let run_sh: PathBuf = repo_root().join("examples/pillow_smoke/run.sh");
    if !run_sh.is_file() {
        eprintln!(
            "SKIP: runner script not found at {} (is this the right worktree?)",
            run_sh.display()
        );
        return;
    }

    // bash is present on every target platform we care about (macOS, Linux,
    // WSL). On Windows we skip outright — the shim SONAME scheme is POSIX-only.
    let bash: &str = "bash";
    if which_exists(bash).is_none() {
        eprintln!("SKIP: bash not on PATH; cannot drive pillow_smoke/run.sh");
        return;
    }

    let status = Command::new(bash)
        .arg(&run_sh)
        .status()
        .expect("failed to spawn pillow_smoke/run.sh");

    match status.code() {
        Some(0) => { /* PASS */ }
        Some(2) => {
            eprintln!(
                "SKIP: pillow_smoke/run.sh reported code 2 (python/Pillow/fixture \
                 unavailable — see stderr for details)"
            );
        }
        Some(3) => {
            // BLOCKER. Target resolution, the shim build/probe, dynamic
            // loading, symbol resolution, or the Pillow round-trip failed.
            // These are hard failures after the optional-environment gates;
            // treating them as SKIPs would hide stale-artifact and C-API bugs.
            panic!(
                "pillow_smoke/run.sh reported BLOCKER (exit 3): shim target \
                 resolution, build/probe, load, symbol resolution, or the \
                 Pillow round-trip failed. Fix the runner or shim, not the test."
            );
        }
        Some(code) => {
            panic!(
                "pillow_smoke/run.sh failed with exit code {code}: \
                 Pillow loaded the shim but decode/encode round-trip failed. \
                 This is a REAL bug in the C-API shim and must be investigated."
            );
        }
        None => {
            panic!("pillow_smoke/run.sh terminated by signal");
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn pillow_runner_ignores_an_ambient_cross_target() {
    let temp: tempfile::TempDir = tempfile::tempdir().expect("create tempdir");
    let fake_cargo: PathBuf = temp.path().join("cargo");
    let captured_target: PathBuf = temp.path().join("captured-target");
    let target_dir: PathBuf = temp.path().join("target");
    let artifact_name: &str = if cfg!(target_os = "macos") {
        "liblibjpeg_turbo_rs_capi.dylib"
    } else {
        "liblibjpeg_turbo_rs_capi.so"
    };
    let script: String = format!(
        r#"#!/bin/sh
set -eu
if [ "${{1:-}}" = "-vV" ]; then
    printf 'cargo 1.93.1\nhost: x86_64-unknown-linux-gnu\n'
    exit 0
fi
previous=
target=
for argument in "$@"; do
    if [ "$previous" = "--target" ]; then
        target="$argument"
    fi
    previous="$argument"
done
printf '%s' "$target" > '{}'
mkdir -p "$CARGO_TARGET_DIR/$target/release"
: > "$CARGO_TARGET_DIR/$target/release/{artifact_name}"
"#,
        captured_target.display()
    );
    std::fs::write(&fake_cargo, script).expect("write fake cargo");
    let mut permissions: std::fs::Permissions = std::fs::metadata(&fake_cargo)
        .expect("stat fake cargo")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&fake_cargo, permissions).expect("make fake cargo executable");

    let output: std::process::Output = Command::new("bash")
        .arg(repo_root().join("examples/pillow_smoke/run.sh"))
        .env("CARGO", &fake_cargo)
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("CARGO_BUILD_TARGET", "aarch64-unknown-linux-gnu")
        .env_remove("CAPI_BUILD_TARGET")
        .env(
            "PILLOW_SMOKE_FIXTURE",
            temp.path().join("intentionally-missing.jpg"),
        )
        .output()
        .expect("run Pillow harness with ambient cross target");

    assert_eq!(
        output.status.code(),
        Some(2),
        "the fake host build should reach the deliberate missing-fixture skip: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        std::fs::read_to_string(captured_target).expect("read captured target"),
        "x86_64-unknown-linux-gnu",
        "host-execution runner must ignore ambient CARGO_BUILD_TARGET"
    );
}

fn which_exists(cmd: &str) -> Option<PathBuf> {
    let path_var: String = std::env::var("PATH").ok()?;
    for dir in path_var.split(':') {
        let candidate: PathBuf = PathBuf::from(dir).join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
