#[path = "support/cdylib.rs"]
mod cdylib;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Cargo up to 1.99: the library is the sibling in `deps/`.
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

/// P4-276: on whichever layout the running Cargo uses, the helper resolves a
/// loadable cdylib (up to 1.99 the `deps/` sibling; from 1.100 the unit
/// this test's Cargo fingerprint names). That it is *this* build's library is
/// enforced by the refusals pinned below, not asserted here. CI runs this
/// test under an unfiltered `cargo test --workspace` as well as under `-p`,
/// which select different library units.
#[test]
fn outer_cargo_invocation_resolves_its_own_cdylib() {
    let cdylib: PathBuf = cdylib::cargo_built_cdylib_path()
        .expect("the helper must resolve the C-ABI artifact this Cargo invocation built");

    assert!(cdylib.is_file(), "missing cdylib: {}", cdylib.display());
    assert_eq!(
        cdylib.file_name().and_then(|name: &OsStr| name.to_str()),
        Some(cdylib::cdylib_filename())
    );
    cdylib::assert_library_exports(&cdylib, cdylib::REQUIRED_EXPORTS);
}

/// One library unit of a Cargo 1.100 package directory: its recorded
/// fingerprint and, unless `has_cdylib` is false (a check-only build), the
/// cdylib in its `out/`.
fn library_unit(package: &Path, unit: &str, fingerprint_hex: &str, has_cdylib: bool) -> PathBuf {
    let directory: PathBuf = package.join(unit);
    std::fs::create_dir_all(directory.join("fingerprint")).expect("fingerprint directory");
    std::fs::create_dir_all(directory.join("out")).expect("out directory");
    std::fs::write(
        directory.join("fingerprint/lib-libjpeg_turbo_rs_capi"),
        fingerprint_hex,
    )
    .expect("library fingerprint");
    let cdylib: PathBuf = directory.join("out").join(cdylib::cdylib_filename());
    if has_cdylib {
        std::fs::write(&cdylib, b"library").expect("cdylib");
    }
    cdylib
}

/// A test unit whose fingerprint depends on a library with fingerprint
/// `library_fingerprint`, laid out as Cargo 1.100 writes it (the JSON is cut
/// from `symbol_inventory`'s, 2026-11-13). Returns the test executable.
fn test_unit(package: &Path, library_fingerprint: u64) -> PathBuf {
    let directory: PathBuf = package.join("9e788ac3b2a830e4");
    std::fs::create_dir_all(directory.join("fingerprint")).expect("fingerprint directory");
    std::fs::create_dir_all(directory.join("out")).expect("out directory");
    let json: String = format!(
        concat!(
            r#"{{"rustc":4106920547260086991,"features":"[]","declared_features":"[\"png\"]","#,
            r#""target":10547447312741522014,"profile":11751664268911691724,"#,
            r#""path":12184988007399145868,"deps":[[7350845333435588724,"libloading",false,9639548321222082767],"#,
            r#"[11932089159807619702,"libjpeg_turbo_rs",false,7064387275689026485],"#,
            r#"[12812997960710357285,"libjpeg_turbo_rs_capi",false,{library}],"#,
            r#"[12812997960710357285,"build_script_build",false,4392344171590827181]],"#,
            r#""local":[{{"CheckDepInfo":{{"dep_info":"x","fingerprint":"mtime"}}}}],"rustflags":[]}}"#
        ),
        library = library_fingerprint,
    );
    std::fs::write(
        directory.join("fingerprint/test-integration-test-symbol_inventory.json"),
        json,
    )
    .expect("test fingerprint");
    let executable: PathBuf = directory.join("out/symbol_inventory-9e788ac3b2a830e4");
    std::fs::write(&executable, b"test").expect("test executable");
    executable
}

/// The fingerprint Cargo 1.100 recorded for `symbol_inventory`'s library
/// (17174420862757387360) and the same number as the library unit's file
/// spells it, the hex of its little-endian bytes.
const LIBRARY_FINGERPRINT: u64 = 17174420862757387360;
const LIBRARY_FINGERPRINT_HEX: &str = "603cb6dbc7cc57ee";

fn set_time(path: &Path, seconds_after_epoch: u64) {
    let time: std::time::SystemTime =
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds_after_epoch);
    std::fs::File::options()
        .write(true)
        .open(path)
        .and_then(|file: std::fs::File| file.set_modified(time))
        .expect("set the modification time");
}

/// P4-276: the test's own fingerprint names the library it was built
/// against, and only that unit's cdylib is taken. The other unit, built
/// later for another feature set, is newer than the test executable too:
/// choosing by modification time would take it, and the test would load the
/// wrong library and could pass.
#[test]
fn the_library_is_the_unit_the_test_fingerprint_names() {
    let scratch: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    let package: &Path = scratch.path();
    let ours: PathBuf = library_unit(package, "701e8b7399e923b8", LIBRARY_FINGERPRINT_HEX, true);
    let other: PathBuf = library_unit(package, "137613d0c220edb4", "217dd677f4072be4", true);
    let executable: PathBuf = test_unit(package, LIBRARY_FINGERPRINT);
    set_time(&ours, 1_000);
    set_time(&other, 2_000);
    set_time(&executable, 3_000);

    assert_eq!(
        cdylib::cdylib_by_fingerprint(&executable, "symbol_inventory").expect("one match"),
        ours
    );
}

/// P4-276: the fingerprint is read from the dependency entry that names the
/// library, not from a neighbour, and converted to the library file's
/// spelling.
#[test]
fn the_test_fingerprint_is_read_and_spelled_as_the_library_file_spells_it() {
    let json: &str = concat!(
        r#"{"rustc":1,"deps":[[1,"libjpeg_turbo_rs",false,2],"#,
        r#"[12812997960710357285,"libjpeg_turbo_rs_capi",false,17174420862757387360],"#,
        r#"[12812997960710357285,"build_script_build",false,4]],"local":[]}"#,
    );
    assert_eq!(
        cdylib::library_fingerprint_in(json).expect("one dependency"),
        LIBRARY_FINGERPRINT_HEX
    );
    assert!(cdylib::library_fingerprint_in(r#"{"deps":[[1,"libloading",false,2]]}"#).is_err());
    assert!(cdylib::library_fingerprint_in(r#"{"rustc":1}"#).is_err());
    assert!(
        cdylib::library_fingerprint_in(r#"{"deps":[[1,"libjpeg_turbo_rs_capi",false,x]]}"#)
            .is_err()
    );
}

/// P4-276: every outcome other than one unit with a cdylib that predates the
/// test is a refusal, never a guess.
#[test]
fn anything_but_one_matching_library_is_refused() {
    // No unit has the fingerprint.
    let none: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    library_unit(none.path(), "137613d0c220edb4", "217dd677f4072be4", true);
    let executable: PathBuf = test_unit(none.path(), LIBRARY_FINGERPRINT);
    let error: String = cdylib::cdylib_by_fingerprint(&executable, "symbol_inventory")
        .expect_err("no unit matches");
    assert!(error.contains("no library unit"), "wrong refusal: {error}");

    // Two units claim it.
    let two: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    library_unit(
        two.path(),
        "701e8b7399e923b8",
        LIBRARY_FINGERPRINT_HEX,
        true,
    );
    library_unit(
        two.path(),
        "1111111111111111",
        LIBRARY_FINGERPRINT_HEX,
        true,
    );
    let executable: PathBuf = test_unit(two.path(), LIBRARY_FINGERPRINT);
    let error: String = cdylib::cdylib_by_fingerprint(&executable, "symbol_inventory")
        .expect_err("two units match");
    assert!(error.contains("several"), "wrong refusal: {error}");

    // The matching unit is a check-only build: metadata, no cdylib.
    let check: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    library_unit(
        check.path(),
        "d867a76d8cc141da",
        LIBRARY_FINGERPRINT_HEX,
        false,
    );
    let executable: PathBuf = test_unit(check.path(), LIBRARY_FINGERPRINT);
    let error: String = cdylib::cdylib_by_fingerprint(&executable, "symbol_inventory")
        .expect_err("no cdylib in the match");
    assert!(error.contains("holds no"), "wrong refusal: {error}");

    // A test executable outside a unit's out/ directory.
    let loose: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    let elsewhere: PathBuf = loose.path().join("symbol_inventory");
    std::fs::write(&elsewhere, b"test").expect("executable");
    assert!(cdylib::cdylib_by_fingerprint(&elsewhere, "symbol_inventory").is_err());
}

/// P4-276 (review): a library rebuilt in place after a source edit keeps its
/// unit and its fingerprint, but the running test was linked against the
/// earlier build. A cdylib newer than the test executable is therefore
/// refused.
#[test]
fn a_library_rebuilt_after_the_test_was_linked_is_refused() {
    let scratch: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    let ours: PathBuf = library_unit(
        scratch.path(),
        "701e8b7399e923b8",
        LIBRARY_FINGERPRINT_HEX,
        true,
    );
    let executable: PathBuf = test_unit(scratch.path(), LIBRARY_FINGERPRINT);
    set_time(&executable, 1_000);
    set_time(&ours, 2_000);

    let error: String = cdylib::cdylib_by_fingerprint(&executable, "symbol_inventory")
        .expect_err("rebuilt after the test");
    assert!(error.contains("rebuilt after"), "wrong refusal: {error}");
}

/// P4-276 (review): `scripts/install_capi.sh` relinks the Linux library with
/// the version script the build script wrote. On the 1.100 layout that file
/// is no longer below or beside the library's directory, so the staging
/// tests hand the script the path Cargo compiled into them. It must be a
/// real file wherever the build script ran, which is every platform. This
/// pins the pair the staging tests pass; that `install_capi.sh` uses it is
/// checked on Linux by `capi_symbol_versions`.
#[test]
fn the_version_script_of_this_build_is_handed_to_the_install_script() {
    let (variable, map): (&str, &str) = cdylib::version_map_environment()
        .expect("build.rs publishes CAPI_VERSION_SCRIPT for every target of the package");

    assert_eq!(variable, "CAPI_VERSION_MAP");
    assert!(Path::new(map).is_file(), "no version script at {map}");
    assert_eq!(
        Path::new(map)
            .file_name()
            .and_then(|name: &OsStr| name.to_str()),
        Some("libjpeg.map")
    );
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
