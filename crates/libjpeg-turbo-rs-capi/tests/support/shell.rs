#![allow(dead_code)] // Each integration-test binary uses a different subset of this shared helper.

//! The shell and archive tools the staging tests drive, resolved per host.
//!
//! `scripts/install_capi.sh` and `scripts/package_capi_release.sh` are bash.
//! On Linux and macOS that is the `bash` on PATH. On Windows it is Git for
//! Windows' bash — which the CI image installs, but which can sit behind
//! `C:\Windows\System32\bash.exe` (the WSL launcher, present whether or not a
//! distribution is) in PATH order, so a bare `bash` may resolve to a shell
//! that cannot run anything. Resolving it by install location first is what
//! keeps the Windows leg (P4-131) from silently exercising the wrong shell.

use std::path::{Path, PathBuf};
use std::process::Command;

fn runs(program: &Path, arg: &str) -> bool {
    Command::new(program)
        .arg(arg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The bash that runs the staging scripts, or the reason there is none.
pub fn bash() -> Result<PathBuf, &'static str> {
    if cfg!(windows) {
        for base in ["ProgramFiles", "ProgramW6432"] {
            if let Some(dir) = std::env::var_os(base) {
                let candidate: PathBuf =
                    PathBuf::from(dir).join("Git").join("bin").join("bash.exe");
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
    }
    if runs(Path::new("bash"), "--version") {
        Ok(PathBuf::from("bash"))
    } else {
        Err("bash not on PATH (Windows: Git for Windows is required)")
    }
}

/// The `tar` that unpacks a bundle, or the reason there is none.
///
/// Windows ships bsdtar as `System32\tar.exe`, which takes Windows paths; the
/// GNU tar inside Git for Windows reads `C:\...` as `host:file` and tries to
/// open a remote archive. Prefer the one that understands the paths the test
/// hands it.
pub fn tar() -> Result<PathBuf, &'static str> {
    if cfg!(windows) {
        if let Some(root) = std::env::var_os("SystemRoot") {
            let candidate: PathBuf = PathBuf::from(root).join("System32").join("tar.exe");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    if runs(Path::new("tar"), "--version") {
        Ok(PathBuf::from("tar"))
    } else {
        Err("tar not on PATH")
    }
}

/// A path as an argument to the bash scripts.
///
/// Forward slashes on Windows: inside a double-quoted bash expansion a
/// backslash is literal, and `dirname` does not split on it. The scripts also
/// normalise what they receive with `cygpath`, so this is belt and braces —
/// but it keeps the argument readable in failure output either way.
pub fn script_arg(path: &Path) -> String {
    let text: String = path.to_string_lossy().into_owned();
    if cfg!(windows) {
        text.replace('\\', "/")
    } else {
        text
    }
}

/// Whether `haystack` contains `needle` as a contiguous byte sequence.
pub fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}
