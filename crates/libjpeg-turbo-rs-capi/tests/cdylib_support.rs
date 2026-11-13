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

/// P4-276 (codex review): `cargo test --workspace`, the replacement gate's
/// first item, unifies features across the workspace and so builds a
/// different library unit from `cargo test -p libjpeg-turbo-rs-capi`. The
/// helper must be able to ask for either; both name this test target.
#[test]
fn the_query_can_name_the_package_or_the_whole_workspace() {
    let package: Vec<&str> = cdylib::cargo_query_arguments(cdylib::PackageSelection::Package, None);
    let workspace: Vec<&str> =
        cdylib::cargo_query_arguments(cdylib::PackageSelection::Workspace, None);

    assert!(package
        .windows(2)
        .any(|pair| pair == ["-p", "libjpeg-turbo-rs-capi"]));
    assert!(!package.contains(&"--workspace"));
    assert!(workspace.contains(&"--workspace"));
    assert!(!workspace.contains(&"-p"));
    for arguments in [&package, &workspace] {
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--test", "cdylib_support"]));
        assert!(arguments.contains(&"--no-run") && arguments.contains(&"--locked"));
        assert_eq!(
            arguments.contains(&"--release"),
            !cfg!(debug_assertions),
            "the query must ask for the profile this test was built with"
        );
        assert!(!arguments.contains(&"--target"), "no target was named");
    }

    let cross: Vec<&str> = cdylib::cargo_query_arguments(
        cdylib::PackageSelection::Package,
        Some("aarch64-unknown-linux-gnu"),
    );
    assert!(cross
        .windows(2)
        .any(|pair| pair == ["--target", "aarch64-unknown-linux-gnu"]));
}

/// P4-276 (codex review): a relative `CARGO_TARGET_DIR` is resolved against
/// the outer invocation's working directory, which the test cannot see, so
/// the query is given the build directory itself. It is recognised by the
/// tag Cargo writes there, at any depth above the executable.
#[test]
fn the_build_root_is_the_directory_cargo_tagged() {
    const TAG: &str = "Signature: 8a477f597d28d172789f06886806bc55\n";
    let scratch: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    let root: PathBuf = scratch.path().join("outer").join("target");
    let unit: PathBuf = root.join("debug/build/libjpeg-turbo-rs-capi/9e788ac3b2a830e4/out");
    std::fs::create_dir_all(&unit).expect("unit directory");
    let executable: PathBuf = unit.join("symbol_inventory-9e788ac3b2a830e4");

    assert_eq!(
        cdylib::cargo_build_root_of(&executable),
        None,
        "no tag, no claim about where the build directory is"
    );

    std::fs::write(root.join("CACHEDIR.TAG"), TAG).expect("tag");
    assert_eq!(
        cdylib::cargo_build_root_of(&executable),
        Some(cdylib::CargoBuildRoot {
            root: root.clone(),
            target: None,
        })
    );
}

/// P4-276 (review): a build for a named target has the triple's directory
/// between the build directory and the profile, and Cargo tags that one too.
/// The nearest tag is then not the build directory: handing it to the query
/// would nest a second tree inside the first. The helper must give the outer
/// tagged directory and report the triple, so the query can name it.
#[test]
fn a_named_target_is_reported_and_is_not_mistaken_for_the_build_root() {
    const TAG: &str = "Signature: 8a477f597d28d172789f06886806bc55\n";
    let scratch: tempfile::TempDir = tempfile::tempdir().expect("temporary directory");
    let root: PathBuf = scratch.path().join("target");
    let triple_dir: PathBuf = root.join("aarch64-unknown-linux-gnu");
    let unit: PathBuf = triple_dir.join("release/build/libjpeg-turbo-rs-capi/1111111111111111/out");
    std::fs::create_dir_all(&unit).expect("unit directory");
    std::fs::write(root.join("CACHEDIR.TAG"), TAG).expect("root tag");
    std::fs::write(triple_dir.join("CACHEDIR.TAG"), TAG).expect("target tag");

    assert_eq!(
        cdylib::cargo_build_root_of(&unit.join("soname-1111111111111111")),
        Some(cdylib::CargoBuildRoot {
            root,
            target: Some(String::from("aarch64-unknown-linux-gnu")),
        })
    );
}

/// `compiler-artifact` lines as Cargo 1.100.0 printed them for
/// `cargo test --no-run --message-format=json` on macOS (2026-11-13), cut
/// down to the fields the helper reads plus enough neighbours to show it does
/// not depend on their order. The library's file name is the one of the
/// platform running the test.
fn cargo_1_100_messages(lib_unit: &str, test_executable: &str) -> String {
    cargo_1_100_messages_with_freshness(lib_unit, test_executable, true)
}

/// The same lines, with the library reported as just compiled when
/// `is_library_fresh` is false. Around them are the other messages a real
/// query prints: a build script's result, whose `out_dir` holds the library's
/// name only as a directory, and a diagnostic whose rendered text quotes the
/// very keys the reader looks for.
fn cargo_1_100_messages_with_freshness(
    lib_unit: &str,
    test_executable: &str,
    is_library_fresh: bool,
) -> String {
    format!(
        concat!(
            r#"{{"reason":"build-script-executed","package_id":"path+file:///w/crates/libjpeg-turbo-rs-capi#0.2.0","#,
            r#""linked_libs":[],"env":[["CAPI_VERSION_SCRIPT","/t/debug/build/libjpeg-turbo-rs-capi/de0225842e89a4b0/out/libjpeg.map"]],"#,
            r#""out_dir":"/t/debug/build/libjpeg-turbo-rs-capi/de0225842e89a4b0/out"}}"#,
            "\n",
            r#"{{"reason":"compiler-message","message":{{"rendered":"warning: \"executable\":\"/elsewhere/t\" and \"filenames\":[\"/elsewhere/{lib}\"]\n","level":"warning"}}}}"#,
            "\n",
            r#"{{"reason":"compiler-artifact","package_id":"path+file:///w/crates/libjpeg-turbo-rs-capi#0.2.0","#,
            r#""target":{{"kind":["rlib","cdylib","staticlib"],"name":"libjpeg_turbo_rs_capi"}},"features":[],"#,
            r#""filenames":["{unit}/out/liblibjpeg_turbo_rs_capi.rlib","{unit}/out/{lib}","{unit}/out/liblibjpeg_turbo_rs_capi.a"],"#,
            r#""executable":null,"fresh":{fresh}}}"#,
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
        fresh = is_library_fresh,
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

    let error: String = cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST))
        .expect_err("no library was named");
    assert!(error.contains("name no"), "wrong refusal: {error}");
}

/// P4-276 (review): a query that had to compile the library did not find the
/// outer invocation's: either the outer build was another configuration
/// (`cargo test --workspace` against the package query), or a source changed
/// while the tests were running and Cargo rebuilt the unit in place, under
/// the same path. Both must be refused although the executable matches.
#[test]
fn a_library_the_query_had_to_compile_is_refused() {
    let messages: String = cargo_1_100_messages_with_freshness(
        "/t/debug/build/libjpeg-turbo-rs-capi/701e8b7399e923b8",
        RUNNING_TEST,
        false,
    );

    let error: String = cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST))
        .expect_err("a library compiled by the query is not the one under test");

    assert!(error.contains("had to compile"), "wrong refusal: {error}");
    assert!(error.contains(cdylib::CDYLIB_OVERRIDE_ENV));
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

    let error: String = cdylib::cdylib_from_cargo_messages(&messages, Path::new(RUNNING_TEST))
        .expect_err("two libraries were named");
    assert!(error.contains("more than one"), "wrong refusal: {error}");
}

/// P4-276 (review): `scripts/install_capi.sh` relinks the Linux library with
/// the version script the build script wrote. On the 1.100 layout that file
/// is no longer below or beside the library's directory, so the staging
/// tests hand the script the path Cargo compiled into them. It must be a
/// real file wherever the build script ran, which is every platform.
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
