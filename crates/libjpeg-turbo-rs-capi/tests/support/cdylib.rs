#![allow(dead_code)] // Each integration-test binary uses a different subset of this shared helper.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

fn cdylib_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "libjpeg_turbo_rs_capi.dll"
    } else if cfg!(target_os = "macos") {
        "liblibjpeg_turbo_rs_capi.dylib"
    } else {
        "liblibjpeg_turbo_rs_capi.so"
    }
}

/// Resolve the cdylib emitted by the same outer Cargo invocation that built
/// the integration-test executable.
///
/// Cargo places both files in the profile's `deps/` directory. Using that
/// sibling avoids a nested Cargo process, so CLI-only target configuration
/// (`--config`, `-Z build-std`, custom linkers, and JSON target paths) remains
/// exactly the configuration that produced the artifact under test.
pub fn cargo_built_cdylib_path_for_executable(executable: &Path) -> Result<PathBuf, String> {
    let deps_dir: &Path = executable.parent().ok_or_else(|| {
        format!(
            "integration-test executable has no parent directory: {}",
            executable.display()
        )
    })?;
    Ok(deps_dir.join(cdylib_filename()))
}

pub fn cargo_built_cdylib_path() -> Result<PathBuf, String> {
    let executable: PathBuf = std::env::current_exe()
        .map_err(|error| format!("could not resolve current test executable: {error}"))?;
    let cdylib: PathBuf = cargo_built_cdylib_path_for_executable(&executable)?;
    if !cdylib.is_file() {
        return Err(format!(
            "Cargo did not emit the C-ABI cdylib next to the integration test: {}",
            cdylib.display()
        ));
    }
    Ok(cdylib)
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
