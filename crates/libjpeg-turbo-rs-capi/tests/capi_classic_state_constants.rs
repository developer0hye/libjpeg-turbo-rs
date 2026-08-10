//! Issue #468 (P4-104): the classic state constants must match upstream's
//! numbering, and must be *proved* to, not assumed.
//!
//! The issue records "`STOPPING` is misnumbered relative to upstream" as an
//! open problem. Measured against the pinned submodule that is no longer true
//! — all fifteen agree. This suite exists so the agreement is checked rather
//! than rediscovered, and so a future edit cannot quietly reintroduce the
//! divergence the issue was filed for.
//!
//! Upstream declares them in `jpegint.h` (not `jpeglib.h`, which the
//! acceptance criterion names): `CSTATE_*` at 100.. and `DSTATE_*` at 200...
//! Both sides are parsed from source rather than restated here, so this cannot
//! agree with a stale copy of either.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// `#define CSTATE_START 100` / `#define DSTATE_START 200` from `jpegint.h`.
fn upstream_states() -> BTreeMap<String, i32> {
    let header: PathBuf = workspace_root().join("references/libjpeg-turbo/src/jpegint.h");
    let text: String = match std::fs::read_to_string(&header) {
        Ok(t) => t,
        Err(e) => panic!(
            "cannot read {} ({e}). The reference submodule must be checked out; \
             skipping would leave this gate green while comparing nothing.",
            header.display()
        ),
    };
    let mut out: BTreeMap<String, i32> = BTreeMap::new();
    for line in text.lines() {
        let t: &str = line.trim();
        let Some(rest) = t.strip_prefix("#define ") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let (Some(name), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        if !(name.starts_with("CSTATE_") || name.starts_with("DSTATE_")) {
            continue;
        }
        if let Ok(v) = value.parse::<i32>() {
            out.insert(name.to_string(), v);
        }
    }
    out
}

/// `const CSTATE_START: c_int = 100;` from the shim, read from source so the
/// test cannot drift from a private constant it has no other way to see.
fn shim_states() -> BTreeMap<String, i32> {
    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("jpeglib.rs");
    let text: String = std::fs::read_to_string(&source)
        .unwrap_or_else(|e| panic!("read {}: {e}", source.display()));
    let mut out: BTreeMap<String, i32> = BTreeMap::new();
    for line in text.lines() {
        let t: &str = line.trim();
        let Some(rest) = t.strip_prefix("const ") else {
            continue;
        };
        let Some((name, tail)) = rest.split_once(':') else {
            continue;
        };
        if !(name.starts_with("CSTATE_") || name.starts_with("DSTATE_")) {
            continue;
        }
        let Some((_, value)) = tail.split_once('=') else {
            continue;
        };
        let value: &str = value
            .trim()
            .trim_end_matches(';')
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        if let Ok(v) = value.parse::<i32>() {
            out.insert(name.to_string(), v);
        }
    }
    out
}

#[test]
fn classic_state_constants_match_upstream_numbering() {
    let upstream: BTreeMap<String, i32> = upstream_states();
    let shim: BTreeMap<String, i32> = shim_states();

    assert!(
        upstream.len() >= 15,
        "parsed only {} upstream state constants; the parser is broken and a \
         vacuous comparison would pass",
        upstream.len()
    );
    assert!(
        !shim.is_empty(),
        "parsed no shim state constants; see the note above"
    );

    let mut wrong: Vec<String> = Vec::new();
    for (name, ours) in &shim {
        match upstream.get(name) {
            Some(theirs) if theirs == ours => {}
            Some(theirs) => wrong.push(format!("  {name}: shim={ours} upstream={theirs}")),
            None => wrong.push(format!(
                "  {name}: shim={ours} but upstream has no such state"
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "state constants diverge from references/libjpeg-turbo/src/jpegint.h:\n{}",
        wrong.join("\n")
    );

    // DSTATE_STOPPING is named in the issue specifically; assert it by name so
    // a regression there is unmistakable rather than one line in a diff.
    assert_eq!(
        shim.get("DSTATE_STOPPING").copied(),
        upstream.get("DSTATE_STOPPING").copied(),
        "DSTATE_STOPPING is the constant P4-104 recorded as misnumbered"
    );
}
