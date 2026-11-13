#![allow(dead_code)] // Each integration-test binary uses a different subset of this shared helper.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// An explicit library to test instead of the one Cargo built. The path must
/// exist: an override that names nothing is a misconfiguration, not a reason
/// to fall back to some other library.
pub const CDYLIB_OVERRIDE_ENV: &str = "CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI";

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

/// Resolve the cdylib built by the same Cargo invocation that built the
/// running integration-test executable, so a test can never pass against a
/// stale library or one built with other features (P4-73).
///
/// - Cargo up to 1.99 puts both files in the profile's `deps/` directory, and
///   the sibling is the answer.
/// - Cargo 1.100 gives every unit its own directory and declares the layout
///   internal (P4-276). No stable variable names a package's own cdylib, so
///   the helper asks Cargo: `cargo test --no-run --message-format=json` for
///   this test target, with this build's profile and features. The answer is
///   used only if it names the running executable, which proves the query
///   resolved the unit graph this test was built in and makes its library
///   the one that graph holds. Everything is already built, so the query
///   compiles nothing.
///
/// A query that resolves anything else is an error naming
/// [`CDYLIB_OVERRIDE_ENV`]: flags that exist only on the outer command line
/// (`--target`, `--config`, a custom profile) are not visible from inside
/// the test, and guessing among the libraries in the build directory is how
/// a wrong one gets tested.
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
    let executable: PathBuf = std::env::current_exe()
        .map_err(|error| format!("could not resolve current test executable: {error}"))?;
    let sibling: PathBuf = cargo_built_cdylib_path_for_executable(&executable)?;
    if sibling.is_file() {
        return Ok(sibling);
    }
    let cargo: OsString = std::env::var_os("CARGO").ok_or_else(|| {
        format!(
            "no C-ABI cdylib beside {} and CARGO is not set, so Cargo cannot be asked where it \
             is; run the test through `cargo test` or set {CDYLIB_OVERRIDE_ENV}",
            executable.display()
        )
    })?;
    let arguments: Vec<&str> = cargo_query_arguments();
    let output: Output = Command::new(&cargo)
        .args(&arguments)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|error| format!("could not run {cargo:?} {arguments:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo {}` failed ({}); set {CDYLIB_OVERRIDE_ENV} to test a library directly.\n{}",
            arguments.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    cdylib_from_cargo_messages(&String::from_utf8_lossy(&output.stdout), &executable)
        .map_err(|error| format!("`cargo {}`: {error}", arguments.join(" ")))
}

/// The query that rebuilds nothing when it matches the outer invocation:
/// same package, this test target, this build's profile and features.
/// `--locked` because a query must never be what re-resolves dependencies.
fn cargo_query_arguments() -> Vec<&'static str> {
    let mut arguments: Vec<&'static str> = vec![
        "test",
        "--no-run",
        "--locked",
        "--message-format=json",
        "-p",
        "libjpeg-turbo-rs-capi",
        "--test",
        env!("CARGO_CRATE_NAME"),
    ];
    if !cfg!(debug_assertions) {
        arguments.push("--release");
    }
    // Every feature of the package, so the query enables exactly the set this
    // test was compiled with. A feature added to `Cargo.toml` but not here
    // makes the query resolve another build, which the caller refuses.
    if cfg!(feature = "png") {
        arguments.extend(["--features", "png"]);
    }
    arguments
}

/// Pick the cdylib out of Cargo's `compiler-artifact` messages, provided the
/// same messages name `executable` as a test executable.
pub fn cdylib_from_cargo_messages(messages: &str, executable: &Path) -> Result<PathBuf, String> {
    let running: PathBuf = canonical(executable);
    let mut names_running_executable: bool = false;
    let mut reported_executables: Vec<String> = Vec::new();
    let mut libraries: Vec<PathBuf> = Vec::new();
    for line in messages.lines() {
        if json_string_field(line, "reason")?.as_deref() != Some("compiler-artifact") {
            continue;
        }
        if let Some(reported) = json_string_field(line, "executable")? {
            if canonical(Path::new(&reported)) == running {
                names_running_executable = true;
            }
            reported_executables.push(reported);
        }
        for file in json_string_array_field(line, "filenames")? {
            let path: PathBuf = PathBuf::from(file);
            let is_library: bool = path.file_name() == Some(OsStr::new(cdylib_filename()));
            if is_library && !libraries.contains(&path) {
                libraries.push(path);
            }
        }
    }
    if !names_running_executable {
        return Err(format!(
            "Cargo resolved a different build than the one running: this test is {}, the query \
             reported {reported_executables:?}. The outer invocation used a profile, target or \
             configuration the query cannot see; set {CDYLIB_OVERRIDE_ENV} to the library to test",
            executable.display()
        ));
    }
    match libraries.as_slice() {
        [library] => Ok(library.clone()),
        [] => Err(format!(
            "Cargo's messages name no {}; set {CDYLIB_OVERRIDE_ENV} to the library to test",
            cdylib_filename()
        )),
        several => Err(format!(
            "Cargo's messages name more than one {}: {several:?}",
            cdylib_filename()
        )),
    }
}

/// Symlinks (`/tmp` on macOS) and Windows path prefixes must not make one
/// file look like two; a path that does not exist compares as written.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The string value of `"key":"…"` in one line of Cargo's JSON, `None` when
/// the key is absent or its value is not a string (`null`).
///
/// A JSON parser would be a new dev-dependency for two fields; Cargo prints
/// each message on one line and these keys occur once in it.
pub fn json_string_field(line: &str, key: &str) -> Result<Option<String>, String> {
    let Some(value) = json_value_after_key(line, key) else {
        return Ok(None);
    };
    if !value.starts_with('"') {
        return Ok(None);
    }
    json_string_at(value).map(|(string, _)| Some(string))
}

/// The strings of `"key":["…","…"]`; empty when the key is absent.
pub fn json_string_array_field(line: &str, key: &str) -> Result<Vec<String>, String> {
    let Some(value) = json_value_after_key(line, key) else {
        return Ok(Vec::new());
    };
    let Some(mut rest) = value.strip_prefix('[') else {
        return Ok(Vec::new());
    };
    let mut strings: Vec<String> = Vec::new();
    loop {
        rest = rest.trim_start_matches([',', ' ']);
        if rest.starts_with(']') {
            return Ok(strings);
        }
        let (string, after) = json_string_at(rest)?;
        strings.push(string);
        rest = after;
    }
}

fn json_value_after_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let needle: String = format!("\"{key}\":");
    line.find(&needle)
        .map(|start| line[start + needle.len()..].trim_start())
}

/// The four hex digits of a `\\u` escape.
fn json_hex4(chars: &mut std::str::CharIndices<'_>) -> Option<u32> {
    let digits: String = chars.take(4).map(|(_, digit)| digit).collect();
    if digits.len() != 4 {
        return None;
    }
    u32::from_str_radix(&digits, 16).ok()
}

/// Decode the JSON string that `text` starts with; returns it and the text
/// after its closing quote.
fn json_string_at(text: &str) -> Result<(String, &str), String> {
    let malformed = || format!("malformed JSON string in Cargo's output: {text:.80}");
    let mut chars = text.char_indices();
    if chars.next().map(|(_, c)| c) != Some('"') {
        return Err(malformed());
    }
    let mut decoded: String = String::new();
    while let Some((index, c)) = chars.next() {
        match c {
            '"' => return Ok((decoded, &text[index + 1..])),
            '\\' => match chars.next().map(|(_, escaped)| escaped) {
                Some('n') => decoded.push('\n'),
                Some('t') => decoded.push('\t'),
                Some('r') => decoded.push('\r'),
                Some('b') => decoded.push('\u{8}'),
                Some('f') => decoded.push('\u{c}'),
                Some('u') => {
                    let first: u32 = json_hex4(&mut chars).ok_or_else(malformed)?;
                    let scalar: u32 = if (0xD800..0xDC00).contains(&first) {
                        // A character above the basic plane: a surrogate pair.
                        let starts_low: bool = chars.next().map(|(_, c)| c) == Some('\\')
                            && chars.next().map(|(_, c)| c) == Some('u');
                        let second: u32 = json_hex4(&mut chars).ok_or_else(malformed)?;
                        if !starts_low || !(0xDC00..0xE000).contains(&second) {
                            return Err(malformed());
                        }
                        0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                    } else {
                        first
                    };
                    decoded.push(char::from_u32(scalar).ok_or_else(malformed)?);
                }
                // `\\`, `\"` and `\/` stand for themselves.
                Some(literal) => decoded.push(literal),
                None => return Err(malformed()),
            },
            plain => decoded.push(plain),
        }
    }
    Err(malformed())
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
