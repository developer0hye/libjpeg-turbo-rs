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

/// P4-276: whatever layout the running Cargo uses, the helper must hand back
/// the cdylib of *this* invocation. Up to Cargo 1.99 that is the sibling in
/// `deps/`; from 1.100 the test executable and the library live in different
/// unit directories and the helper asks Cargo instead.
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

/// `compiler-artifact` lines as Cargo 1.100.0 printed them for
/// `cargo test --no-run --message-format=json` on macOS (2026-11-13), cut
/// down to the fields the helper reads plus enough neighbours to show it does
/// not depend on their order. The library's file name is the one of the
/// platform running the test.
fn cargo_1_100_messages(lib_unit: &str, test_executable: &str) -> String {
    format!(
        concat!(
            r#"{{"reason":"compiler-artifact","package_id":"path+file:///w/crates/libjpeg-turbo-rs-capi#0.2.0","#,
            r#""target":{{"kind":["rlib","cdylib","staticlib"],"name":"libjpeg_turbo_rs_capi"}},"features":[],"#,
            r#""filenames":["{unit}/out/liblibjpeg_turbo_rs_capi.rlib","{unit}/out/{lib}","{unit}/out/liblibjpeg_turbo_rs_capi.a"],"#,
            r#""executable":null,"fresh":true}}"#,
            "\n",
            r#"{{"reason":"compiler-artifact","target":{{"kind":["test"],"name":"symbol_inventory"}},"#,
            r#""filenames":["{exe}"],"executable":"{exe}","fresh":true}}"#,
            "\n",
            r#"{{"reason":"build-finished","success":true}}"#,
            "\n",
        ),
        unit = lib_unit,
        lib = cdylib::cdylib_filename(),
        exe = test_executable,
    )
}

const RUNNING_TEST: &str =
    "/t/debug/build/libjpeg-turbo-rs-capi/9e788ac3b2a830e4/out/symbol_inventory-9e788ac3b2a830e4";

/// P4-276: the library is taken from Cargo's own artifact messages, and only
/// when the same messages name the running test executable, which is what
/// ties the answer to this invocation's unit graph.
#[test]
fn cdylib_is_read_from_cargo_messages_that_name_this_test_executable() {
    let unit: &str = "/t/debug/build/libjpeg-turbo-rs-capi/701e8b7399e923b8";
    let messages: String = cargo_1_100_messages(unit, RUNNING_TEST);

    let cdylib: PathBuf = cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST))
        .expect("the messages name both the library and this executable");

    assert_eq!(
        cdylib,
        Path::new(unit).join("out").join(cdylib::cdylib_filename())
    );
}

/// P4-276: a query that resolved another configuration (other features,
/// profile or target) reports a different test executable. Its library is
/// then a different build and must be refused, not used: `ci.yml` builds the
/// default and the `--features png` library into one profile directory in a
/// single job, so a guess would load the wrong one and still pass.
#[test]
fn messages_for_another_configuration_are_refused() {
    let other_test: &str =
        "/t/debug/build/libjpeg-turbo-rs-capi/2222222222222222/out/symbol_inventory-2222222222222222";
    let messages: String = cargo_1_100_messages(
        "/t/debug/build/libjpeg-turbo-rs-capi/1111111111111111",
        other_test,
    );

    let error: String = cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST))
        .expect_err("a different test executable is a different build");

    assert!(
        error.contains(cdylib::CDYLIB_OVERRIDE_ENV),
        "the error must name the override: {error}"
    );
}

/// P4-276: no library line (a failed build, or a message format this helper
/// does not know) is an error, never a fall back to scanning the build
/// directory.
#[test]
fn messages_without_the_library_are_refused() {
    let messages: String = format!(
        r#"{{"reason":"compiler-artifact","target":{{"kind":["test"],"name":"symbol_inventory"}},"filenames":["{exe}"],"executable":"{exe}","fresh":true}}"#,
        exe = RUNNING_TEST,
    );

    assert!(cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST)).is_err());
}

/// P4-276: two different libraries in one answer cannot both be this
/// invocation's.
#[test]
fn messages_with_two_libraries_are_refused() {
    let mut messages: String = cargo_1_100_messages(
        "/t/debug/build/libjpeg-turbo-rs-capi/701e8b7399e923b8",
        RUNNING_TEST,
    );
    messages.push_str(&cargo_1_100_messages(
        "/t/debug/build/libjpeg-turbo-rs-capi/1111111111111111",
        RUNNING_TEST,
    ));

    assert!(cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST)).is_err());
}

/// P4-276: Cargo escapes Windows separators in its JSON and may print a
/// character outside ASCII raw or as `\u` escapes (a surrogate pair above the
/// basic plane); each must decode to the path Cargo meant.
#[test]
fn json_string_escapes_decode_to_the_path_cargo_meant() {
    let line: &str = concat!(
        r#"{"executable":"D:\\a\\target\\out\\t-9e78.exe","#,
        r#""filenames":["C:\\d\u00e9p\\\"q\"\\lib.dll","/tmp/caf\u00e9/\ud83d\ude00/é"],"other":null}"#,
    );

    assert_eq!(
        cdylib::json_string_field(line, "executable").expect("well-formed"),
        Some(String::from(r"D:\a\target\out\t-9e78.exe"))
    );
    assert_eq!(
        cdylib::json_string_array_field(line, "filenames").expect("well-formed"),
        vec![
            String::from("C:\\d\u{e9}p\\\"q\"\\lib.dll"),
            String::from("/tmp/caf\u{e9}/\u{1f600}/\u{e9}"),
        ]
    );
    assert_eq!(
        cdylib::json_string_field(line, "other").expect("well-formed"),
        None,
        "null is not a string"
    );
    assert_eq!(
        cdylib::json_string_field(line, "absent").expect("well-formed"),
        None
    );
    assert!(
        cdylib::json_string_field(r#"{"executable":"unterminated"#, "executable").is_err(),
        "a cut-off string is an error, not an empty path"
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
