#[path = "support/cdylib.rs"]
mod cdylib;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[test]
fn cargo_built_cdylib_path_is_next_to_the_outer_test_executable() {
    let executable: &Path = Path::new("/cargo-target/custom-board/debug/deps/cdylib_support-hash");

    let cdylib: PathBuf =
        cdylib::cargo_built_cdylib_path_for_executable(executable).expect("test executable path");

    assert_eq!(cdylib.parent(), executable.parent());
    assert!(cdylib
        .file_name()
        .and_then(|name: &OsStr| name.to_str())
        .is_some_and(|name: &str| name.contains("libjpeg_turbo_rs_capi")));
}

#[test]
fn outer_cargo_invocation_emits_the_sibling_cdylib() {
    let cdylib: PathBuf = cdylib::cargo_built_cdylib_path()
        .expect("Cargo must emit the current C-ABI artifact beside this integration test");

    assert!(cdylib.is_file(), "missing cdylib: {}", cdylib.display());
}

#[test]
fn target_qualified_release_cdylib_path_uses_cargo_layout() {
    let target_dir: PathBuf = PathBuf::from("/workspace/libjpeg-turbo-rs/target");
    let target: &OsStr = OsStr::new("x86_64-unknown-linux-gnu");

    let path: PathBuf = cdylib::release_cdylib_path_for_target_in(&target_dir, target);

    assert_eq!(
        path.parent(),
        Some(target_dir.join(target).join("release").as_path())
    );
}

#[test]
fn custom_target_release_path_uses_the_json_file_stem() {
    let target_dir: PathBuf = PathBuf::from("/workspace/libjpeg-turbo-rs/target");
    let custom_target: &OsStr = OsStr::new("targets/custom-board.json");

    let path: PathBuf = cdylib::release_cdylib_path_for_target_in(&target_dir, custom_target);

    assert_eq!(
        path.parent(),
        Some(target_dir.join("custom-board").join("release").as_path())
    );
}
