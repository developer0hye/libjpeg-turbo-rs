//! CI-aware discovery helpers for C libjpeg-turbo tools.
//!
//! Integration tests in this crate shell out to `djpeg`, `cjpeg`,
//! `jpegtran`, and `rdjpgcom` for cross-validation.  Historically every
//! test handled a missing tool with `eprintln!("SKIP: ..."); return;`,
//! which silently degraded cross-validation coverage to zero when the
//! tools were not installed — including on CI runners where the tools
//! *should* always be present.
//!
//! This module centralises the discovery policy:
//!
//! * [`require_c_tool`] returns `Ok(PathBuf)` when the binary is
//!   discoverable and an `io::ErrorKind::NotFound` error otherwise.
//!   It is a thin, library-style wrapper around [`super::c_tool_path`].
//! * [`is_ci`] reads the `CI` environment variable and returns whether
//!   the process is running on a hosted CI runner.  It treats any
//!   non-empty value other than `0` / `false` as CI.
//!
//! The `require_c_tool!` macro in `helpers/mod.rs` composes both into
//! the CI-vs-local policy that callers actually want at `#[test]` sites.

use std::path::PathBuf;

/// Returns `true` when running under CI (any truthy value of the `CI`
/// environment variable).
///
/// GitHub Actions, GitLab CI, CircleCI, Travis, and most hosted runners
/// set `CI=true`.  Any non-empty value that is not `0` / `false` counts
/// as CI so we stay conservative and fail closed — a misconfigured
/// runner that leaves `CI` set to something unusual still triggers the
/// hard-fail path rather than the silent-skip path.
pub fn is_ci() -> bool {
    std::env::var("CI")
        .map(|v: String| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

/// Locate a C libjpeg-turbo tool (`djpeg`, `cjpeg`, `jpegtran`,
/// `rdjpgcom`, ...) for use in cross-validation tests.
///
/// Returns `Ok(path)` when the binary is discoverable via
/// `/opt/homebrew/bin/` (macOS Homebrew install) or the system `which`.
/// Returns `Err(io::Error)` with `ErrorKind::NotFound` when the binary
/// cannot be located on either path.
///
/// Prefer the `require_c_tool!` macro (defined in `helpers/mod.rs`)
/// over calling this function directly: the macro enforces the
/// CI-vs-local policy (CI panics, local prints `SKIP` and early
/// `return`s) at the `#[test]` call site so no test can silently pass
/// in CI with missing cross-validation coverage.
pub fn require_c_tool(name: &str) -> Result<PathBuf, std::io::Error> {
    match super::c_tool_path(name) {
        Some(path) => Ok(path),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "C libjpeg-turbo tool '{}' not found on PATH or in /opt/homebrew/bin/",
                name
            ),
        )),
    }
}
