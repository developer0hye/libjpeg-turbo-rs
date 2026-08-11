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

/// A stock libjpeg development install: `jpeglib.h` plus a linkable
/// `libjpeg`. Distinct from [`TurboJpegDev`] — the classic `jpeg_*` API and
/// the TurboJPEG `tj3*` API ship in different libraries.
pub struct LibJpegDev {
    /// Every directory that must go on the include path. Multiarch installs
    /// split the headers: `jpeglib.h` in `include/`, the ABI-specific
    /// `jconfig.h` in `include/<triplet>/`. Passing only one of them either
    /// fails to compile or, worse, silently picks up the *host's* `jpeglib.h`
    /// while linking the prefixed library — a version mismatch that would
    /// quietly invalidate the comparison the oracle exists to make.
    pub include_dirs: Vec<PathBuf>,
    pub lib_dir: PathBuf,
    /// The static archive, when the install ships one.
    ///
    /// Preferred over `-ljpeg`, and not as an optimisation: a shared link
    /// leaves the choice of library to the runtime loader, and
    /// `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` outrank the `-rpath` this
    /// helper sets. An environment pointing at an installed copy of *our own*
    /// shim would therefore have the oracle load the implementation under
    /// test and agree with it perfectly. Linking the archive by absolute path
    /// takes the loader out of the picture: the bytes checked by
    /// [`is_our_own_shim`] are the bytes that end up in the binary.
    pub static_lib: Option<PathBuf>,
}

/// Locate a stock libjpeg(-turbo) development install for the classic API.
///
/// **This must not find our own shim.** The capi crate installs a
/// libjpeg-compatible `libjpeg` of its own — `scripts/install_capi.sh` ships a
/// `jconfig.h`, a `jpeglib.h` and a `libjpeg` under whatever prefix it is
/// given — and an oracle linked against that would compare the implementation
/// with itself, agreeing perfectly while both were wrong. Header presence
/// therefore proves nothing; the candidate library itself is inspected for our
/// crate name (see [`is_our_own_shim`]), which no C build can contain.
pub fn find_libjpeg_dev() -> Option<LibJpegDev> {
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

    let mut lib_subdirs: Vec<PathBuf> = vec![PathBuf::from("lib64"), PathBuf::from("lib")];
    if let Some(triplet) = host_multiarch_triplet() {
        lib_subdirs.push(PathBuf::from("lib").join(&triplet));
        lib_subdirs.push(PathBuf::from("lib64").join(&triplet));
    }

    for prefix in prefixes {
        let base_include: PathBuf = prefix.join("include");
        let Ok(header_text) = std::fs::read_to_string(base_include.join("jpeglib.h")) else {
            continue;
        };
        if !header_text.contains("jpeg_consume_input") {
            continue;
        }
        // Debian/Ubuntu keep the architecture-specific `jconfig.h` under
        // `/usr/include/<triplet>` while `jpeglib.h` stays in `/usr/include`,
        // so a plain `include/` test rejects a perfectly good distro install
        // and silently drops the C comparison. Accept either spelling.
        //
        // Reachable only where the compiler reports a multiarch triplet, so it
        // is untested by this repository's own runs: macOS clang reports none,
        // and the Linux job points `LIBJPEG_TURBO_PREFIX` at a cmake install
        // that puts `jconfig.h` in plain `include/`. Treat it as convenience
        // for a developer's distro box, not as a gate that something checks.
        let mut candidates: Vec<PathBuf> = vec![base_include.clone()];
        if let Some(triplet) = host_multiarch_triplet() {
            candidates.push(base_include.join(&triplet));
        }
        let Some(config_dir) = candidates.into_iter().find(|dir| {
            // Must be a *v8* install. The traces these oracles print are
            // version- and layout-specific — `JPEG_LIB_VERSION`, both struct
            // sizes — so comparing against an ordinary v6b development install
            // (still the default on several distributions) reports a
            // divergence that is really an ABI mismatch: version 62 and a
            // 632-byte struct against our 80 and 656. That is a false failure,
            // and the kind that gets a real gate deleted.
            std::fs::read_to_string(dir.join("jconfig.h")).is_ok_and(|text| {
                text.lines().any(|line| {
                    let mut parts = line.split_whitespace();
                    parts.next() == Some("#define")
                        && parts.next() == Some("JPEG_LIB_VERSION")
                        && parts.next() == Some("80")
                })
            })
        }) else {
            continue;
        };
        let mut include_dirs: Vec<PathBuf> = vec![base_include.clone()];
        if config_dir != base_include {
            include_dirs.push(config_dir);
        }
        for lib_subdir in &lib_subdirs {
            let lib_dir: PathBuf = prefix.join(lib_subdir);
            let static_lib: Option<PathBuf> =
                Some(lib_dir.join("libjpeg.a")).filter(|path| path.exists());
            // Check provenance on whichever file the link will actually use.
            let library: Option<PathBuf> = static_lib.clone().or_else(|| {
                ["libjpeg.so", "libjpeg.dylib"]
                    .iter()
                    .map(|file| lib_dir.join(file))
                    .find(|path| path.exists())
            });
            // Our own installed shim is the one candidate that would make this
            // gate compare the implementation with itself.
            let usable: bool = library
                .as_deref()
                .is_some_and(|path| !is_our_own_shim(path));
            if usable {
                return Some(LibJpegDev {
                    include_dirs,
                    lib_dir,
                    static_lib,
                });
            }
        }
    }
    None
}

/// True when `library` is this crate's own C-ABI shim rather than a C libjpeg.
///
/// The Rust build embeds the crate name in the binary — symbol names, panic
/// location strings, the metadata section — and no C build of libjpeg contains
/// it. This is provenance rather than a naming or layout convention, which is
/// what makes it safe to rely on: our shim installs the same header set under
/// the same library name as the real thing, so nothing about the file system
/// distinguishes the two.
fn is_our_own_shim(library: &Path) -> bool {
    let Ok(bytes) = std::fs::read(library) else {
        // Unreadable: treat as suspect rather than trusting it.
        return true;
    };
    const CRATE_MARKER: &[u8] = b"libjpeg_turbo_rs";
    bytes
        .windows(CRATE_MARKER.len())
        .any(|window| window == CRATE_MARKER)
}

/// Build `examples/<source_stem>.c` against stock libjpeg's classic API.
///
/// `None` means no install was found, which [`oracle_is_required`] still
/// forbids when `LIBJPEG_TURBO_PREFIX` says one is provisioned. A compile
/// failure once an install *is* present is fatal, for the same reason as in
/// [`build_oracle`]: a skipped comparison that reads as a pass is the failure
/// mode these gates exist to prevent.
pub fn build_classic_oracle(source_stem: &str) -> Option<PathBuf> {
    let install: LibJpegDev = match find_libjpeg_dev() {
        Some(install) => install,
        None => {
            assert!(
                !oracle_is_required(),
                "no stock *v8* libjpeg development install found under LIBJPEG_TURBO_PREFIX={:?} — \
                 that variable says one is provisioned, and skipping here would pass the C parity \
                 gate without checking anything. A candidate needs a jpeglib.h declaring \
                 jpeg_consume_input, a jconfig.h declaring JPEG_LIB_VERSION 80 (in include/ or \
                 include/<triplet>/), and a \
                 libjpeg that is not this crate's own shim — an installed shim is rejected on \
                 purpose, since linking it would compare the implementation with itself",
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
        .args(
            install
                .include_dirs
                .iter()
                .map(|dir| format!("-I{}", dir.display())),
        )
        .args(match &install.static_lib {
            // Absolute path, no `-l` search and no loader involvement.
            Some(archive) => vec![archive.display().to_string()],
            None => vec![
                format!("-L{}", install.lib_dir.display()),
                "-ljpeg".to_string(),
                format!("-Wl,-rpath,{}", install.lib_dir.display()),
            ],
        })
        .output()
        .unwrap_or_else(|e| panic!("failed to run {compiler}: {e}"));
    assert!(
        output.status.success(),
        "failed to build the {source_stem} C oracle with {compiler} against {:?}:\n{}",
        install.include_dirs,
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
