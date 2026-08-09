//! Shared TurboJPEG-oracle discovery for this crate's C-parity tests.
//!
//! Extracted from `yuv_four_component_c_parity.rs`, whose own doc comment
//! recorded the duplication as deliberate ("not reachable from this workspace
//! member"). That reasoning covers the *root crate's* `tests/helpers/c_oracle.rs`
//! — it does not justify a second copy inside this crate, which is what a
//! third C-parity suite would have created.
//!
//! The root-crate helper stays separate: cross-crate test-helper sharing would
//! mean publishing it from the library, and this copy is deliberately stricter
//! anyway (an explicit prefix is exclusive rather than first-tried, and the
//! header must declare `tj3*`).

// Each integration-test binary compiles this module separately, so anything a
// given binary does not call reads as dead there. Suppressing per-item would
// mean re-annotating on every new consumer.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// A TurboJPEG development install: `turbojpeg.h` plus a linkable
/// `libturbojpeg`.
pub struct TurboJpegDev {
    pub include_dir: PathBuf,
    pub lib_dir: PathBuf,
}

/// Locate a TurboJPEG 3 development install, or `None` on a machine without
/// one.
pub fn find_turbojpeg_dev() -> Option<TurboJpegDev> {
    // An explicit LIBJPEG_TURBO_PREFIX is exclusive: falling through to another
    // install when it turns out to be TurboJPEG 2.x or unreadable would let the
    // gate validate a different setup than the one CI provisioned, and pass
    // while the configured prefix is broken.
    let prefixes: Vec<PathBuf> = match std::env::var_os("LIBJPEG_TURBO_PREFIX") {
        Some(prefix) => vec![PathBuf::from(prefix)],
        None => {
            let mut prefixes: Vec<PathBuf> = Vec::new();
            if let Some(prefix) = std::env::var_os("CONDA_PREFIX") {
                prefixes.push(PathBuf::from(prefix));
            }
            prefixes.extend(
                [
                    "/opt/libjpeg-turbo",
                    "/opt/homebrew/opt/jpeg-turbo",
                    "/opt/homebrew",
                    "/usr/local",
                    "/usr",
                ]
                .iter()
                .map(PathBuf::from),
            );
            prefixes
        }
    };

    // Debian/Ubuntu put the header in `<prefix>/include` but the linker stub in
    // `<prefix>/lib/<triplet>`, so a plain lib64/lib scan reports a perfectly
    // good install as absent — which fails the run outright once
    // LIBJPEG_TURBO_PREFIX has made the oracle mandatory, and skips the C
    // comparison when it has not.
    let mut lib_subdirs: Vec<PathBuf> = vec![PathBuf::from("lib64"), PathBuf::from("lib")];
    if let Some(triplet) = host_multiarch_triplet() {
        lib_subdirs.push(PathBuf::from("lib").join(&triplet));
        lib_subdirs.push(PathBuf::from("lib64").join(&triplet));
    }

    for prefix in prefixes {
        let include_dir: PathBuf = prefix.join("include");
        // The oracles are TJ3-only. Ubuntu's stock `libturbojpeg0-dev` is still
        // TurboJPEG 2.1.x, whose header declares no `tj3*` entry point, so
        // accepting it here would turn a skip into a compile failure on a
        // perfectly ordinary developer machine.
        let header: PathBuf = include_dir.join("turbojpeg.h");
        let Ok(header_text) = std::fs::read_to_string(&header) else {
            continue;
        };
        if !header_text.contains("tj3DecompressToYUVPlanes8") {
            continue;
        }
        for lib_subdir in &lib_subdirs {
            let lib_dir: PathBuf = prefix.join(lib_subdir);
            let has_library: bool = ["libturbojpeg.so", "libturbojpeg.dylib", "libturbojpeg.a"]
                .iter()
                .any(|file| lib_dir.join(file).exists());
            if has_library {
                return Some(TurboJpegDev {
                    include_dir,
                    lib_dir,
                });
            }
        }
    }
    None
}

/// The compiler's multiarch triplet (e.g. `x86_64-linux-gnu`), or `None` where
/// the toolchain does not use one (macOS clang prints an empty line).
fn host_multiarch_triplet() -> Option<String> {
    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(compiler)
        .arg("-print-multiarch")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let triplet: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!triplet.is_empty()).then_some(triplet)
}

/// True when a missing TurboJPEG 3 install must fail rather than skip: an
/// explicit `LIBJPEG_TURBO_PREFIX` is a statement that one is provisioned, and
/// skipping would leave the parity gate green while checking nothing. The CI
/// step that runs these tests sets it. Deliberately *not* keyed on a bare `CI`
/// variable — a CI service running `cargo test --workspace` without TurboJPEG
/// development files is the missing-tool case CLAUDE.md lets us skip.
pub fn oracle_is_required() -> bool {
    std::env::var_os("LIBJPEG_TURBO_PREFIX").is_some()
}

/// Build `examples/<source_stem>.c` against a real `libturbojpeg`.
///
/// `None` means no TurboJPEG development install was found on a developer
/// machine — the one legitimate reason to skip, and not one
/// [`oracle_is_required`] allows. A compile failure once an install *is*
/// present is fatal: it would otherwise hide the parity check behind a green
/// run.
pub fn build_oracle(source_stem: &str) -> Option<PathBuf> {
    let install: TurboJpegDev = match find_turbojpeg_dev() {
        Some(install) => install,
        None => {
            assert!(
                !oracle_is_required(),
                "no TurboJPEG 3 development install (a turbojpeg.h declaring tj3* plus a linkable \
                 libturbojpeg) found under LIBJPEG_TURBO_PREFIX={:?} — that variable says one is \
                 provisioned, and skipping here would pass the C parity gate without checking \
                 anything",
                std::env::var_os("LIBJPEG_TURBO_PREFIX")
            );
            return None;
        }
    };

    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(format!("{source_stem}.c"));
    assert!(
        source.exists(),
        "oracle source missing: {}",
        source.display()
    );

    let out_dir: PathBuf = std::env::temp_dir().join(format!(
        "libjpeg_turbo_rs_oracle_{}_{source_stem}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).expect("create oracle build dir");
    let oracle: PathBuf = out_dir.join(source_stem);

    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(&compiler)
        .arg("-O2")
        .arg("-o")
        .arg(&oracle)
        .arg(&source)
        .arg(format!("-I{}", install.include_dir.display()))
        .arg(format!("-L{}", install.lib_dir.display()))
        .arg("-lturbojpeg")
        .arg(format!("-Wl,-rpath,{}", install.lib_dir.display()))
        .output()
        .unwrap_or_else(|e| panic!("failed to run {compiler}: {e}"));
    assert!(
        output.status.success(),
        "failed to build the {source_stem} C oracle with {compiler} against {}:\n{}",
        install.include_dir.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    Some(oracle)
}

/// Run an oracle binary and return its stdout, failing loudly on a non-zero
/// exit so a broken oracle cannot read as "no differences found".
pub fn run_oracle(oracle: &Path, args: &[&str]) -> String {
    let output = Command::new(oracle)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run oracle {}: {e}", oracle.display()));
    assert!(
        output.status.success(),
        "oracle {} exited {:?}:\n{}",
        oracle.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("oracle stdout is not UTF-8")
}
