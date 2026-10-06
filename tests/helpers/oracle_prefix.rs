//! Which C libjpeg-turbo install this repository means — one implementation.
//!
//! Two kinds of consumer resolve `djpeg`/`cjpeg`/`jpegtran`: the differential
//! `#[test]` suites, through `helpers::c_tool_path`, and the corpus harness in
//! `examples/`, which is a binary and cannot use a test-only module the way the
//! suites do. Both include *this* file, so a leg that names its oracle with
//! `LIBJPEG_TURBO_PREFIX` selects the same install whichever of them runs, and
//! the rule has exactly one place to be wrong.
//!
//! It is shared rather than copied because a copy is how a leg labelled with
//! one release measures another. `capi_classic_lifecycle_pathological` carried
//! a private lookup that read `/opt/homebrew/bin` first, and the review of
//! [#569] found a step named "oracle 3.2.0" comparing against homebrew's
//! 3.1.4.1 through it. Both corpus examples carried the same shape (P4-130).
//!
//! [#569]: https://github.com/developer0hye/libjpeg-turbo-rs/pull/569

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// Generic C tool discovery.
///
/// `LIBJPEG_TURBO_PREFIX` names the C install to compare against, exactly as it
/// does for the C-ABI crate's oracle helpers: when it is set, the tool is taken
/// from `<prefix>/bin` and from nowhere else. Otherwise discovery is the
/// historical `/opt/homebrew/bin` then `which` order.
///
/// The variable is what lets the P4-130 dual-oracle matrix run the same suites
/// against two libjpeg-turbo releases and *know* which one answered. PATH alone
/// cannot express it — `/opt/homebrew/bin` is read first, so on macOS the
/// homebrew build wins regardless of PATH.
pub fn c_tool_path(name: &str) -> Option<PathBuf> {
    c_tool_path_under(
        name,
        std::env::var_os("LIBJPEG_TURBO_PREFIX")
            .map(PathBuf::from)
            .as_deref(),
    )
}

/// [`c_tool_path`] with the oracle prefix passed in rather than read from the
/// environment.
///
/// `cargo` runs `#[test]`s as parallel threads of one process, so a test that
/// set `LIBJPEG_TURBO_PREFIX` to exercise the override would race every other
/// test in the binary. Taking the prefix as an argument makes both branches
/// deterministically reachable.
///
/// An explicit prefix is **exclusive**: if the tool is not under it, the answer
/// is `None`. Falling back would let a leg that claims to measure one release
/// silently measure another and report green.
pub fn c_tool_path_under(name: &str, oracle_prefix: Option<&Path>) -> Option<PathBuf> {
    if let Some(prefix) = oracle_prefix {
        let pinned: PathBuf = prefix.join("bin").join(name);
        return pinned.exists().then_some(pinned);
    }
    let homebrew: PathBuf = PathBuf::from(format!("/opt/homebrew/bin/{}", name));
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}
