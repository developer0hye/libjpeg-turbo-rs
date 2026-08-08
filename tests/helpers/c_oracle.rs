//! Builds and runs the C reference oracles that the stock tools cannot
//! stand in for: the four-component CMYK oracle, and the gray→4bpp
//! TurboJPEG oracle added for issue #369.
//!
//! Issues #313 / #339. Every other encode cross-check shells out to `cjpeg`,
//! but cjpeg reads only PNM/BMP/GIF/Targa — it cannot ingest CMYK at all. The
//! four-component path therefore had no C reference and its options were only
//! ever compared against themselves, which is how an 18-byte spurious JFIF
//! marker survived in every CMYK file we ever wrote.
//!
//! `examples/cmyk_encode_c_oracle.c` drives libjpeg directly to close that
//! gap. It needs libjpeg's headers and library, which a runtime-only install
//! does not provide, so this module locates a development install and compiles
//! the harness on demand, caching the binary next to the test executables.
//!
//! Set `CMYK_C_ORACLE` to a prebuilt binary to skip discovery and compilation
//! entirely — useful where libjpeg is built from source in a non-standard
//! layout.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Serializes builds within a process. Tests run concurrently and several will
/// ask for the oracle at once; without this they race to compile into the same
/// artifact directory, and the loser sees its staging file renamed out from
/// under it. That failure surfaces as "no libjpeg install found", which is a
/// thoroughly misleading way to say "you lost a race".
static BUILD_LOCK: Mutex<()> = Mutex::new(());

/// A libjpeg development install: headers plus a linkable library.
struct LibjpegDevInstall {
    include_dir: PathBuf,
    lib_dir: PathBuf,
}

fn find_libjpeg_dev() -> Option<LibjpegDevInstall> {
    let mut prefixes: Vec<PathBuf> = Vec::new();
    if let Ok(prefix) = std::env::var("LIBJPEG_TURBO_PREFIX") {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
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

    for prefix in prefixes {
        let include_dir: PathBuf = prefix.join("include");
        if !include_dir.join("jpeglib.h").exists() {
            continue;
        }
        for lib_name in ["lib64", "lib"] {
            let lib_dir: PathBuf = prefix.join(lib_name);
            let has_library: bool = ["libjpeg.a", "libjpeg.so", "libjpeg.dylib"]
                .iter()
                .any(|file| lib_dir.join(file).exists());
            if has_library {
                return Some(LibjpegDevInstall {
                    include_dir,
                    lib_dir,
                });
            }
        }
    }
    None
}

/// Directory holding the test executables — a writable, per-profile location
/// that is already gitignored, so the compiled oracle lands beside the binaries
/// that use it rather than in the source tree.
fn artifact_dir() -> Option<PathBuf> {
    let mut path: PathBuf = std::env::current_exe().ok()?;
    // .../target/<profile>/deps/<test-binary>
    path.pop();
    path.pop();
    Some(path)
}

/// Locate — building if necessary — the CMYK reference oracle.
///
/// Returns `None` when no libjpeg development install can be found, which is
/// the ordinary state of a machine with only the runtime tools installed.
/// Callers apply the project's skip-locally / fail-in-CI policy.
pub fn cmyk_c_oracle() -> Option<PathBuf> {
    if let Ok(prebuilt) = std::env::var("CMYK_C_ORACLE") {
        let path: PathBuf = PathBuf::from(prebuilt);
        if path.exists() {
            return Some(path);
        }
        // An explicit path that does not exist is a misconfiguration, not a
        // reason to silently fall back to a different binary.
        panic!("CMYK_C_ORACLE points at {path:?}, which does not exist");
    }

    let artifact_dir: PathBuf = artifact_dir()?;
    let oracle: PathBuf = artifact_dir.join("cmyk_encode_c_oracle");
    let _guard = BUILD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("cmyk_encode_c_oracle.c");
    if !source.exists() {
        return None;
    }
    if is_newer(&oracle, &source) {
        return Some(oracle);
    }

    let install: LibjpegDevInstall = find_libjpeg_dev()?;

    // Build to a name unique across processes and rename, so a concurrently
    // running test binary cannot observe a half-written executable or compile
    // into the same staging path.
    let staging: PathBuf =
        artifact_dir.join(format!("cmyk_encode_c_oracle.{}.tmp", std::process::id()));
    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(&compiler)
        .arg("-O2")
        .arg("-o")
        .arg(&staging)
        .arg(&source)
        .arg(format!("-I{}", install.include_dir.display()))
        .arg(format!("-L{}", install.lib_dir.display()))
        .arg("-ljpeg")
        .arg(format!("-Wl,-rpath,{}", install.lib_dir.display()))
        .output()
        .ok()?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&staging);
        panic!(
            "failed to build the CMYK C oracle with {compiler} against {:?}:\n{}",
            install.include_dir,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if std::fs::rename(&staging, &oracle).is_err() {
        // Another test binary finished first and its build is equally valid.
        let _ = std::fs::remove_file(&staging);
        return oracle.exists().then_some(oracle);
    }
    Some(oracle)
}

/// Locate — building if necessary — the tables-only reference oracle
/// (`examples/tables_only_c_oracle.c`).
///
/// P4-116: `jpeg_write_tables()` has no `cjpeg` switch, so the abbreviated
/// tables-only stream can only be cross-validated through the library API.
/// Same build-and-cache and `None`-means-skip contract as [`cmyk_c_oracle`].
pub fn tables_only_c_oracle() -> Option<PathBuf> {
    build_libjpeg_oracle("tables_only_c_oracle")
}

/// Shared build-and-cache for the `libjpeg`-linked oracles under `examples/`.
fn build_libjpeg_oracle(stem: &str) -> Option<PathBuf> {
    let artifact_dir: PathBuf = artifact_dir()?;
    let oracle: PathBuf = artifact_dir.join(stem);
    let _guard = BUILD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(format!("{stem}.c"));
    if !source.exists() {
        return None;
    }
    if is_newer(&oracle, &source) {
        return Some(oracle);
    }

    let install: LibjpegDevInstall = find_libjpeg_dev()?;
    let staging: PathBuf = artifact_dir.join(format!("{stem}.{}.tmp", std::process::id()));
    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(&compiler)
        .arg("-O2")
        .arg("-o")
        .arg(&staging)
        .arg(&source)
        .arg(format!("-I{}", install.include_dir.display()))
        .arg(format!("-L{}", install.lib_dir.display()))
        .arg("-ljpeg")
        .arg(format!("-Wl,-rpath,{}", install.lib_dir.display()))
        .output()
        .ok()?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&staging);
        panic!(
            "failed to build the {stem} C oracle with {compiler} against {:?}:\n{}",
            install.include_dir,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if std::fs::rename(&staging, &oracle).is_err() {
        let _ = std::fs::remove_file(&staging);
        return oracle.exists().then_some(oracle);
    }
    Some(oracle)
}

/// A TurboJPEG development install: `turbojpeg.h` plus a linkable
/// `libturbojpeg`. Discovered separately from libjpeg because runtime
/// packages (e.g. `libjpeg-turbo-progs`) ship neither, and some prefixes
/// carry one library but not the other.
fn find_turbojpeg_dev() -> Option<LibjpegDevInstall> {
    let mut prefixes: Vec<PathBuf> = Vec::new();
    if let Ok(prefix) = std::env::var("LIBJPEG_TURBO_PREFIX") {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
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

    for prefix in prefixes {
        let include_dir: PathBuf = prefix.join("include");
        if !include_dir.join("turbojpeg.h").exists() {
            continue;
        }
        for lib_name in ["lib64", "lib"] {
            let lib_dir: PathBuf = prefix.join(lib_name);
            let has_library: bool = ["libturbojpeg.a", "libturbojpeg.so", "libturbojpeg.dylib"]
                .iter()
                .any(|file| lib_dir.join(file).exists());
            if has_library {
                return Some(LibjpegDevInstall {
                    include_dir,
                    lib_dir,
                });
            }
        }
    }
    None
}

/// Locate — building if necessary — the issue #369 gray→4bpp TurboJPEG
/// oracle (`examples/gray_argb_c_oracle.c`). Same build-and-cache and
/// `None`-means-skip contract as [`cmyk_c_oracle`], but linked against
/// `libturbojpeg` (tj3 API) rather than `libjpeg`, and with no prebuilt-binary
/// environment override.
pub fn gray_argb_c_oracle() -> Option<PathBuf> {
    let artifact_dir: PathBuf = artifact_dir()?;
    let oracle: PathBuf = artifact_dir.join("gray_argb_c_oracle");
    let _guard = BUILD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("gray_argb_c_oracle.c");
    if !source.exists() {
        return None;
    }
    if is_newer(&oracle, &source) {
        return Some(oracle);
    }

    let install: LibjpegDevInstall = find_turbojpeg_dev()?;

    let staging: PathBuf =
        artifact_dir.join(format!("gray_argb_c_oracle.{}.tmp", std::process::id()));
    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(&compiler)
        .arg("-O2")
        .arg("-o")
        .arg(&staging)
        .arg(&source)
        .arg(format!("-I{}", install.include_dir.display()))
        .arg(format!("-L{}", install.lib_dir.display()))
        .arg("-lturbojpeg")
        .arg(format!("-Wl,-rpath,{}", install.lib_dir.display()))
        .output()
        .ok()?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&staging);
        panic!(
            "failed to build the gray-ARGB C oracle with {compiler} against {:?}:\n{}",
            install.include_dir,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if std::fs::rename(&staging, &oracle).is_err() {
        let _ = std::fs::remove_file(&staging);
        return oracle.exists().then_some(oracle);
    }
    Some(oracle)
}

/// Decode `jpeg_path` through the gray-ARGB oracle to `format_name`
/// (e.g. "ARGB"), returning the raw 4bpp pixel buffer.
///
/// Panics on oracle failure: once the harness exists, a non-zero exit is a
/// real problem with the case under test, not a reason to skip.
pub fn decode_with_gray_argb_c_oracle(
    oracle: &Path,
    jpeg_path: &Path,
    format_name: &str,
) -> Vec<u8> {
    let output = Command::new(oracle)
        .arg(jpeg_path)
        .arg(format_name)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {oracle:?}: {error:?}"));
    assert!(
        output.status.success() && !output.stdout.is_empty(),
        "gray-ARGB C oracle failed for {format_name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn is_newer(candidate: &Path, source: &Path) -> bool {
    let (Ok(candidate_meta), Ok(source_meta)) = (candidate.metadata(), source.metadata()) else {
        return false;
    };
    match (candidate_meta.modified(), source_meta.modified()) {
        (Ok(built), Ok(edited)) => built >= edited,
        _ => false,
    }
}

/// Encode raw interleaved CMYK through the C oracle, returning the JPEG bytes.
///
/// Panics on oracle failure: once the harness exists, a non-zero exit is a real
/// problem with the case under test, not a reason to skip.
pub fn encode_with_cmyk_c_oracle(oracle: &Path, pixels: &[u8], args: &[String]) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new(oracle)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn {oracle:?}: {error:?}"));
    let mut stdin = child.stdin.take().expect("oracle stdin");
    let payload: Vec<u8> = pixels.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let output = child
        .wait_with_output()
        .unwrap_or_else(|error| panic!("failed to run {oracle:?}: {error:?}"));
    let _ = writer.join();
    assert!(
        output.status.success() && !output.stdout.is_empty(),
        "CMYK C oracle failed for args {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}
