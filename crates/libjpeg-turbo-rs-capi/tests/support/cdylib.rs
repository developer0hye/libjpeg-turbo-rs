#![allow(dead_code)] // Each integration-test binary uses a different subset of this shared helper.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

/// An explicit library to test instead of the one Cargo built. The path must
/// exist: an override that names nothing is a misconfiguration, not a reason
/// to fall back to some other library.
pub const CDYLIB_OVERRIDE_ENV: &str = "CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI";

/// The crate name the test's Cargo fingerprint lists the library under.
const LIBRARY_CRATE: &str = "libjpeg_turbo_rs_capi";

pub fn cdylib_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "libjpeg_turbo_rs_capi.dll"
    } else if cfg!(target_os = "macos") {
        "liblibjpeg_turbo_rs_capi.dylib"
    } else {
        "liblibjpeg_turbo_rs_capi.so"
    }
}

/// Where Cargo up to 1.99 put the cdylib: in `deps/`, beside the
/// integration-test executable it built in the same invocation.
pub fn cargo_built_cdylib_path_for_executable(executable: &Path) -> Result<PathBuf, String> {
    let deps_dir: &Path = executable.parent().ok_or_else(|| {
        format!(
            "integration-test executable has no parent directory: {}",
            executable.display()
        )
    })?;
    Ok(deps_dir.join(cdylib_filename()))
}

/// Resolve the cdylib of the build the running integration-test executable
/// was linked against, so a test does not pass against a stale library or
/// one built with other features (P4-73).
///
/// 1. [`CDYLIB_OVERRIDE_ENV`], if set.
/// 2. Cargo up to 1.99: the sibling in the profile's `deps/` directory. That
///    file carries no hash, so one profile directory holds a single library
///    whatever features built it last; the outer invocation has just rebuilt
///    it for this configuration.
/// 3. Cargo 1.100 gives every unit its own `build/<pkg>/<hash>/out/`
///    directory (P4-276). The test unit's fingerprint
///    (`fingerprint/test-integration-test-<name>.json`) records the
///    fingerprint of the library unit it was compiled against, and each
///    library unit records its own (`fingerprint/lib-libjpeg_turbo_rs_capi`).
///    The library is the cdylib of the one sibling unit whose fingerprint is
///    the one the test recorded, provided it is no newer than the test
///    executable: rebuilding a library in place after a source edit keeps
///    its unit and its fingerprint, and the test was linked against the
///    earlier one.
///
/// Step 3 reads Cargo's internal files, which Cargo may change without
/// notice; CI builds with the current stable toolchain, so a change shows up
/// there as a refusal. Every other outcome (no match, several, an unreadable
/// file, a match without a cdylib) is an error naming
/// [`CDYLIB_OVERRIDE_ENV`]; the helper never chooses among libraries by
/// itself. Asking Cargo instead (`cargo test --no-run`) needs the outer
/// invocation's exact target selection, which the test cannot see: an
/// unfiltered `cargo test --workspace` builds a different unit graph from
/// any per-target query, and a query that does not match compiles a second
/// build in the middle of the run.
///
/// Step 3 builds the fingerprint file name from `CARGO_CRATE_NAME`, which
/// equals the test target's name only while test file names have no hyphen;
/// a `tests/foo-bar.rs` would always be refused.
///
/// The answer is computed once per test process: the running executable
/// cannot be relinked under it.
pub fn cargo_built_cdylib_path() -> Result<PathBuf, String> {
    static RESOLVED: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    RESOLVED.get_or_init(resolve_cargo_built_cdylib).clone()
}

fn resolve_cargo_built_cdylib() -> Result<PathBuf, String> {
    if let Some(explicit) = std::env::var_os(CDYLIB_OVERRIDE_ENV) {
        let path: PathBuf = PathBuf::from(explicit);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "{CDYLIB_OVERRIDE_ENV} points at {}, which is not a file",
            path.display()
        ));
    }
    let located = || -> Result<PathBuf, String> {
        let executable: PathBuf = std::env::current_exe()
            .map_err(|error| format!("could not resolve current test executable: {error}"))?;
        let sibling: PathBuf = cargo_built_cdylib_path_for_executable(&executable)?;
        if sibling.is_file() {
            return Ok(sibling);
        }
        cdylib_by_fingerprint(&executable, env!("CARGO_CRATE_NAME"))
    };
    located().map_err(|error| format!("{error}; set {CDYLIB_OVERRIDE_ENV} to the library to test"))
}

/// Step 3 of [`cargo_built_cdylib_path`] for the integration test
/// `test_name` whose executable is `executable`.
pub fn cdylib_by_fingerprint(executable: &Path, test_name: &str) -> Result<PathBuf, String> {
    let out_dir: &Path = executable
        .parent()
        .filter(|directory| directory.file_name() == Some(OsStr::new("out")))
        .ok_or_else(|| {
            format!(
                "{} is neither beside the cdylib (Cargo <= 1.99) nor in a unit's out/ \
                 directory (Cargo 1.100)",
                executable.display()
            )
        })?;
    let test_unit: &Path = out_dir
        .parent()
        .ok_or_else(|| format!("{} has no unit directory above it", out_dir.display()))?;
    let package: &Path = test_unit
        .parent()
        .ok_or_else(|| format!("{} has no package directory above it", test_unit.display()))?;
    let test_fingerprint: PathBuf = test_unit
        .join("fingerprint")
        .join(format!("test-integration-test-{test_name}.json"));
    let json: String = std::fs::read_to_string(&test_fingerprint)
        .map_err(|error| format!("could not read {}: {error}", test_fingerprint.display()))?;
    let wanted: String = library_fingerprint_in(&json)
        .map_err(|error| format!("{}: {error}", test_fingerprint.display()))?;

    let entries: std::fs::ReadDir = std::fs::read_dir(package)
        .map_err(|error| format!("could not list {}: {error}", package.display()))?;
    let mut matches: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let candidate: PathBuf = entry
            .map_err(|error| format!("could not list {}: {error}", package.display()))?
            .path();
        let recorded: PathBuf = candidate
            .join("fingerprint")
            .join(format!("lib-{LIBRARY_CRATE}"));
        if let Ok(text) = std::fs::read_to_string(&recorded) {
            if text.trim() == wanted {
                matches.push(candidate);
            }
        }
    }
    let unit: PathBuf = match matches.as_slice() {
        [unit] => unit.clone(),
        [] => {
            return Err(format!(
                "no library unit in {} has the fingerprint {wanted} that {} was built against",
                package.display(),
                test_fingerprint.display()
            ))
        }
        several => {
            return Err(format!(
                "several library units have the fingerprint {wanted}: {several:?}"
            ))
        }
    };
    let cdylib: PathBuf = unit.join("out").join(cdylib_filename());
    if !cdylib.is_file() {
        return Err(format!(
            "the library unit {} that {} was built against holds no {} (a check-only build?)",
            unit.display(),
            executable.display(),
            cdylib_filename()
        ));
    }
    if modified(&cdylib)? > modified(executable)? {
        return Err(format!(
            "{} was rebuilt after {} was linked, so it is not the library the test was built \
             against",
            cdylib.display(),
            executable.display()
        ));
    }
    Ok(cdylib)
}

/// The fingerprint, as the library unit's `lib-libjpeg_turbo_rs_capi` file
/// spells it, of the library a test unit's fingerprint JSON depends on.
///
/// The JSON lists dependencies as `[<id>,"<crate>",<public>,<fingerprint>]`
/// with the fingerprint a `u64`; the library unit's file holds the same
/// number as the hex of its little-endian bytes. Exactly one dependency
/// entry must name the library.
pub fn library_fingerprint_in(json: &str) -> Result<String, String> {
    let deps: &str = json
        .split_once("\"deps\":[")
        .map(|(_, rest)| rest)
        .ok_or("no \"deps\" list")?;
    let needle: String = format!(",\"{LIBRARY_CRATE}\",");
    let mut found: Vec<u64> = Vec::new();
    let mut rest: &str = deps;
    while let Some(entry) = rest.strip_prefix('[') {
        let end: usize = entry.find(']').ok_or("unterminated dependency entry")?;
        let fields: &str = &entry[..end];
        if fields.contains(&needle) {
            let fingerprint: &str = fields.rsplit(',').next().ok_or("empty entry")?;
            found.push(
                fingerprint
                    .trim()
                    .parse::<u64>()
                    .map_err(|error| format!("dependency fingerprint {fingerprint:?}: {error}"))?,
            );
        }
        rest = entry[end + 1..].trim_start_matches(',');
    }
    match found.as_slice() {
        [fingerprint] => Ok(fingerprint
            .to_le_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()),
        [] => Err(format!("no dependency on {LIBRARY_CRATE}")),
        _ => Err(format!("{} dependencies on {LIBRARY_CRATE}", found.len())),
    }
}

fn modified(path: &Path) -> Result<SystemTime, String> {
    std::fs::metadata(path)
        .and_then(|metadata: std::fs::Metadata| metadata.modified())
        .map_err(|error| format!("could not read the time of {}: {error}", path.display()))
}

/// The GNU version script this build's build script wrote, as the variable
/// `scripts/install_capi.sh` reads it. The script cannot find the file from
/// `CAPI_TARGET_DIR` on the Cargo 1.100 layout, where the library and the
/// build script's output are in different unit directories (P4-276); the
/// path Cargo compiled into this test is the one of this very build.
pub fn version_map_environment() -> Option<(&'static str, &'static str)> {
    option_env!("CAPI_VERSION_SCRIPT").map(|map| ("CAPI_VERSION_MAP", map))
}

fn target_component(target: &OsStr) -> Option<&OsStr> {
    let target_path: &Path = Path::new(target);
    if target_path.extension().is_some_and(|ext| ext == "json") {
        target_path.file_stem()
    } else {
        Some(target)
    }
}

pub fn release_cdylib_path_in(target_dir: &Path) -> PathBuf {
    target_dir.join("release").join(cdylib_filename())
}

pub fn release_cdylib_path_for_target_in(target_dir: &Path, target: &OsStr) -> PathBuf {
    let target_component: &OsStr =
        target_component(target).expect("custom target JSON path has a file stem");
    release_cdylib_path_in(&target_dir.join(target_component))
}

pub fn cdylib_path() -> PathBuf {
    cargo_built_cdylib_path().unwrap_or_else(|error: String| panic!("{error}"))
}

/// The staged or bundled library must be *the* library: loadable, and
/// exporting one entry point from each of the two APIs it is shipped under.
/// A file that merely has the right name and magic — truncated, or the wrong
/// DLL copied under this one — passes every shape check and fails here.
pub fn assert_library_exports(library: &Path, symbols: &[&str]) {
    let loaded = unsafe { libloading::Library::new(library) }
        .unwrap_or_else(|e| panic!("the shipped library {library:?} does not load: {e}"));
    for symbol in symbols {
        let mut name: Vec<u8> = symbol.as_bytes().to_vec();
        name.push(0);
        let resolved: Result<libloading::Symbol<*const ()>, _> =
            unsafe { loaded.get(name.as_slice()) };
        assert!(
            resolved.is_ok(),
            "{library:?} loads but does not export {symbol}: {:?}",
            resolved.err()
        );
    }
}

/// The entry points every shipped library must resolve: one from the classic
/// libjpeg API and one from TurboJPEG 3.
pub const REQUIRED_EXPORTS: &[&str] = &["jpeg_std_error", "tj3Init"];
