//! P4-209 (#632) criterion 3: no infallible, input-sized allocation under
//! `src/decode/` — enforced, not merely asserted in prose.
//!
//! `vec![…]`, `Vec::with_capacity(…)` and `.to_vec()` abort the process when
//! the allocator refuses. `src/common/try_alloc.rs` states the rule this gate
//! holds the decoder to: a size that came out of a JPEG stream goes through
//! `try_reserve_exact` and surfaces refusal as `JpegError::AllocationFailed`.
//! P4-136 and P4-144 applied that rule site by site, and the primary decode
//! destination still slipped through (P4-209) — which is why the criterion asks
//! for a *mechanism* rather than another sweep.
//!
//! **The mechanism is this test plus `docs/decode_alloc_inventory.tsv`**, in
//! the shape `tests/sizing_arithmetic_gate.rs` set for saturating arithmetic.
//! Every remaining infallible allocation in decoder library code must appear in
//! the inventory with a classification saying why its size is bounded by
//! something other than the input's geometry. A new one fails here until a
//! human either routes it through `common::try_alloc` or classifies it, and a
//! removed one fails until the row is deleted, so the inventory cannot drift.
//!
//! **What is scanned, and where this departs from the model gate.** Lines of
//! `.rs` files under `src/decode/` containing one of [`SCANNED_PATTERNS`],
//! keyed by (file, trimmed line) with a count per key. Skipped: `//` comment
//! lines, `*_tests.rs` files, and — unlike `sizing_arithmetic_gate.rs` —
//! inline `#[cfg(test)] mod … { … }` blocks. The decoder keeps its unit tests
//! inline (`merged_upsample.rs` alone has two dozen `vec!` fixtures), and an
//! inventory that had to track test fixtures would break on every test edit
//! and stop meaning anything. The block is found by brace depth, counted
//! outside string literals, single-character literals (`'{'`) and `//`
//! comments — not `/* */` comments, which `src/decode/` does not use. Because
//! a miscount would end the skip late and hide library code silently,
//! `every_scanned_file_ends_with_balanced_braces` requires every file to end
//! at depth zero with no string or skip left open.
//!
//! **What it cannot see.** The patterns are textual. An input-sized allocation
//! spelled another way — `.clone()` of a `Vec`, `.collect()` into a `Vec`,
//! `extend` past a reservation — is invisible here by construction. P4-209
//! converted the two such sites it found on the paths it touched (the 12-bit
//! grayscale `collect()` and the crop-shift `plane.clone()`); the gate does not
//! claim the class is closed.
//!
//! **Environment:** this reads the repository source tree, so it is skipped
//! where that tree is not reachable — `wasm32-wasip1` under wasmtime (which
//! preopens only `.` and `/tmp`) and a packaged crate, which ships no `docs/`.
//! It runs on every native leg, which is where a developer adds the allocation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One classified line: how many identical copies, and why it is allowed.
#[derive(Debug, PartialEq, Eq)]
struct Entry {
    count: usize,
    classification: String,
}

const INVENTORY: &str = "docs/decode_alloc_inventory.tsv";

/// The allocation forms that abort on refusal and are greppable.
const SCANNED_PATTERNS: [&str; 3] = ["vec![", "Vec::with_capacity(", ".to_vec()"];

/// Classifications the inventory may use. A typo becomes a failure rather than
/// an unnoticed blank cheque. Each one names a bound that is *not* the input's
/// geometry; "the input is probably small" is not on the list.
const KNOWN_CLASSIFICATIONS: [&str; 4] = [
    "fixed-size",
    "component-count",
    "row-scratch",
    "infallible-public-signature",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn scanned_root() -> PathBuf {
    repo_root().join("src").join("decode")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            // `*_tests.rs` are unit-test modules that live in `src/` for
            // visibility reasons; they are test code by any other measure.
            let is_test_module: bool = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("_tests.rs"));
            if !is_test_module {
                out.push(path);
            }
        }
    }
}

fn relative(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Net `{` minus `}` on one line, ignoring braces inside string literals,
/// character literals and a trailing `//` comment. `in_string` carries an
/// unterminated (multi-line) string literal across lines.
///
/// Raw strings are treated as ordinary strings, which is exact for every raw
/// string without an embedded `"` — the only kind `src/decode/` contains.
fn brace_delta(line: &str, in_string: &mut bool) -> i64 {
    let chars: Vec<char> = line.chars().collect();
    let mut delta: i64 = 0;
    let mut i: usize = 0;
    while i < chars.len() {
        let c: char = chars[i];
        if *in_string {
            if c == '\\' {
                i += 2;
                continue;
            }
            if c == '"' {
                *in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            '"' => *in_string = true,
            '/' if chars.get(i + 1) == Some(&'/') => break,
            // `'{'`, `'}'` and an escaped `'\''`; a lifetime (`'a`) has no
            // closing quote two characters on and falls through harmlessly.
            '\'' if chars.get(i + 2) == Some(&'\'') => {
                i += 3;
                continue;
            }
            '{' => delta += 1,
            '}' => delta -= 1,
            _ => {}
        }
        i += 1;
    }
    delta
}

/// The result of splitting one source file: its library lines, and the
/// scanner state at end of file, which must be closed for the split to be
/// trusted.
struct LibrarySplit<'a> {
    lines: Vec<&'a str>,
    final_depth: i64,
    ends_in_string: bool,
    ends_in_skip: bool,
}

/// The lines of `text` that are library code: everything outside
/// `#[cfg(test)] mod … { … }` blocks.
fn library_lines(text: &str) -> Vec<&str> {
    split_library(text).lines
}

fn split_library(text: &str) -> LibrarySplit<'_> {
    let mut kept: Vec<&str> = Vec::new();
    let mut depth: i64 = 0;
    let mut in_string: bool = false;
    // Depth at which the test module being skipped was opened.
    let mut skipping_from: Option<i64> = None;
    let mut cfg_test_pending: bool = false;

    for line in text.lines() {
        let trimmed: &str = line.trim();
        let delta: i64 = brace_delta(line, &mut in_string);

        if let Some(open_depth) = skipping_from {
            depth += delta;
            if depth <= open_depth {
                skipping_from = None;
            }
            continue;
        }

        if trimmed.starts_with("#[cfg(test)]") {
            cfg_test_pending = true;
            depth += delta;
            continue;
        }
        if cfg_test_pending && !trimmed.starts_with("#[") && !trimmed.starts_with("//") {
            cfg_test_pending = false;
            let opens_inline_module: bool = (trimmed.starts_with("mod ")
                || trimmed.starts_with("pub mod ")
                || trimmed.starts_with("pub(crate) mod ")
                || trimmed.starts_with("pub(super) mod "))
                && trimmed.ends_with('{');
            if opens_inline_module {
                skipping_from = Some(depth);
                depth += delta;
                continue;
            }
        }

        depth += delta;
        kept.push(line);
    }
    LibrarySplit {
        lines: kept,
        final_depth: depth,
        ends_in_string: in_string,
        ends_in_skip: skipping_from.is_some(),
    }
}

/// Every infallible allocation in decoder library code, keyed by (file,
/// trimmed source line) so the gate survives line-number drift but still
/// notices a genuinely new allocation — or a second copy of an existing one.
fn occurrences_in_sources() -> BTreeMap<(String, String), usize> {
    let mut files: Vec<PathBuf> = Vec::new();
    walk(&scanned_root(), &mut files);

    let mut found: BTreeMap<(String, String), usize> = BTreeMap::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in library_lines(&text) {
            if !SCANNED_PATTERNS.iter().any(|p| line.contains(p)) {
                continue;
            }
            let trimmed: &str = line.trim();
            // Prose about the rule is not an instance of breaking it.
            if trimmed.starts_with("//") {
                continue;
            }
            *found
                .entry((relative(&path), trimmed.to_string()))
                .or_insert(0) += 1;
        }
    }
    found
}

fn inventory() -> BTreeMap<(String, String), Entry> {
    let path: PathBuf = repo_root().join(INVENTORY);
    let text: String =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let mut map: BTreeMap<(String, String), Entry> = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.splitn(5, '\t').collect();
        assert_eq!(
            fields.len(),
            5,
            "{INVENTORY}:{} must have 5 tab-separated columns \
             (path, count, classification, justification, line), got {}: {line:?}",
            lineno + 1,
            fields.len()
        );
        let count: usize = fields[1].parse().unwrap_or_else(|e| {
            panic!("{INVENTORY}:{}: bad count {:?}: {e}", lineno + 1, fields[1])
        });
        let classification: String = fields[2].to_string();
        assert!(
            KNOWN_CLASSIFICATIONS.contains(&classification.as_str()),
            "{INVENTORY}:{}: unknown classification {classification:?}; \
             allowed: {KNOWN_CLASSIFICATIONS:?}",
            lineno + 1
        );
        assert!(
            !fields[3].trim().is_empty(),
            "{INVENTORY}:{}: the justification column is empty — say what bounds \
             this allocation, since the classification alone does not",
            lineno + 1
        );
        let key: (String, String) = (fields[0].to_string(), fields[4].to_string());
        let previous: Option<Entry> = map.insert(
            key,
            Entry {
                count,
                classification,
            },
        );
        assert!(
            previous.is_none(),
            "{INVENTORY}:{}: duplicate row for {:?}; fold identical lines into \
             one row with a count",
            lineno + 1,
            (fields[0], fields[4])
        );
    }
    map
}

/// `false` when the repository source tree is not reachable — a packaged crate,
/// or a sandboxed target such as `wasm32-wasip1`. Reported explicitly rather
/// than silently, so a green run always means the gate either ran or said why
/// it did not.
fn repository_tree_is_readable() -> bool {
    let root: PathBuf = repo_root();
    root.join(INVENTORY).is_file() && scanned_root().is_dir()
}

/// The gate. Any disagreement between the sources and the inventory fails,
/// with the exact rows to add or remove.
#[test]
fn decode_allocations_match_the_classified_inventory() {
    if !repository_tree_is_readable() {
        eprintln!(
            "SKIP: {INVENTORY} / src/decode is not readable from {}. \
             This gate inspects repository sources, which a packaged crate and \
             a sandboxed target (wasm32-wasip1) do not provide. It runs on \
             every native leg.",
            repo_root().display()
        );
        return;
    }
    let found: BTreeMap<(String, String), usize> = occurrences_in_sources();
    let listed: BTreeMap<(String, String), Entry> = inventory();

    let unlisted: Vec<String> = found
        .iter()
        .filter(|(key, _)| !listed.contains_key(*key))
        .map(|((path, line), n)| {
            format!("  + {path}\t{n}\t<CLASSIFY ME>\t<WHAT BOUNDS IT>\t{line}")
        })
        .collect();
    assert!(
        unlisted.is_empty(),
        "new infallible allocation under src/decode/ is not classified in \
         {INVENTORY}.\n\n\
         If its size comes from the JPEG stream's geometry (width, height, \
         blocks, planes, output size — scaled or cropped), it is a bug: an \
         allocator refusal aborts the process. Route it through \
         `common::try_alloc` (`try_filled_vec`, `try_reserved_vec`, \
         `try_with_capacity`, `try_copy_of`) and propagate with `?` (P4-209).\n\n\
         If it genuinely is bounded by something else, add the row with a \
         classification from {KNOWN_CLASSIFICATIONS:?} and a justification:\n\n{}\n",
        unlisted.join("\n")
    );

    let stale: Vec<String> = listed
        .keys()
        .filter(|key| !found.contains_key(*key))
        .map(|(path, line)| format!("  - {path}\t{line}"))
        .collect();
    assert!(
        stale.is_empty(),
        "{INVENTORY} lists allocations that no longer exist. Delete these rows \
         (or update the source line if it was only reformatted) so the inventory \
         keeps meaning something:\n\n{}\n",
        stale.join("\n")
    );

    let miscounted: Vec<String> = found
        .iter()
        .filter_map(|(key, n)| {
            let entry: &Entry = listed.get(key)?;
            (entry.count != *n).then(|| {
                format!(
                    "  {}\t{} listed ({}), {n} found\t{}",
                    key.0, entry.count, entry.classification, key.1
                )
            })
        })
        .collect();
    assert!(
        miscounted.is_empty(),
        "{INVENTORY} counts disagree with the sources — a copy of a classified \
         line was added or removed. A new copy needs the same scrutiny as a new \
         line:\n\n{}\n",
        miscounted.join("\n")
    );
}

/// The gate is worthless if it scans nothing, or if the test-module skip eats
/// library code. A refactor that moves sources, or a brace-counting bug, would
/// otherwise turn it green forever.
#[test]
fn the_gate_actually_scans_the_library_sources() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository sources not readable; see the sibling test.");
        return;
    }
    let mut files: Vec<PathBuf> = Vec::new();
    walk(&scanned_root(), &mut files);
    assert!(
        files.len() >= 25,
        "expected src/decode/ to hold dozens of sources, found {}",
        files.len()
    );

    let found: BTreeMap<(String, String), usize> = occurrences_in_sources();
    assert!(
        !found.is_empty(),
        "found zero allocations, which means the scan is broken — the \
         inventory is not empty"
    );
    // Both the top-level decoder modules and the pipeline split must be
    // reachable: the defect P4-209 fixed lived in the latter.
    assert!(
        found
            .keys()
            .any(|(p, _)| p.starts_with("src/decode/pipeline_impl/")),
        "src/decode/pipeline_impl/ not scanned"
    );
    assert!(
        found
            .keys()
            .any(|(p, _)| p.starts_with("src/decode/") && !p.contains("pipeline_impl")),
        "top-level src/decode/ modules not scanned"
    );
}

/// A brace miscount — a `/* { */`, a raw string with an embedded quote —
/// would leave a test-module skip open past its module and hide every later
/// line of that file from the gate while it stayed green. Every scanned file
/// must therefore end where it started.
#[test]
fn every_scanned_file_ends_with_balanced_braces() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository sources not readable; see the sibling test.");
        return;
    }
    let mut files: Vec<PathBuf> = Vec::new();
    walk(&scanned_root(), &mut files);
    let unbalanced: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let text: String = std::fs::read_to_string(path).ok()?;
            let split: LibrarySplit<'_> = split_library(&text);
            (split.final_depth != 0 || split.ends_in_string || split.ends_in_skip).then(|| {
                format!(
                    "  {}: depth {}, open string {}, open test-module skip {}",
                    relative(path),
                    split.final_depth,
                    split.ends_in_string,
                    split.ends_in_skip
                )
            })
        })
        .collect();
    assert!(
        unbalanced.is_empty(),
        "the brace scanner lost track in these files, so its test-module skip \
         cannot be trusted there; teach `brace_delta` the construct:\n\n{}\n",
        unbalanced.join("\n")
    );
}

/// The test-module skip, pinned on synthetic input so it is proved to both
/// skip and stop skipping — a skip that never ends would hide every later
/// line in the file, and the inventory would look complete.
#[test]
fn the_test_module_skip_resumes_after_the_module() {
    let source: &str = r#"
fn library() {
    let a: Vec<u8> = vec![0u8; n];
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    fn fixture() {
        let s: &str = "}}} not a brace";
        let c: char = '}';
        let b: Vec<u8> = vec![1u8; 8]; // }
    }
}

fn after() {
    let d: Vec<u8> = slice.to_vec();
    let e: Vec<u16> = Vec::with_capacity(n);
}
"#;
    let kept: Vec<&str> = library_lines(source);
    let allocations: Vec<&str> = kept
        .iter()
        .copied()
        .filter(|line| SCANNED_PATTERNS.iter().any(|p| line.contains(p)))
        .map(str::trim)
        .collect();
    assert_eq!(
        allocations,
        vec![
            "let a: Vec<u8> = vec![0u8; n];",
            "let d: Vec<u8> = slice.to_vec();",
            "let e: Vec<u16> = Vec::with_capacity(n);"
        ],
        "the skip must drop the test module's allocation and only that, and \
         every SCANNED_PATTERNS entry must match its own form"
    );
    let split: LibrarySplit<'_> = split_library(source);
    assert!(
        split.final_depth == 0 && !split.ends_in_string && !split.ends_in_skip,
        "the synthetic source is balanced; the scanner must say so"
    );

    // A `#[cfg(test)]` item that is not a module is library-adjacent code the
    // gate has no business hiding, and must not start a skip.
    let not_a_module: &str = "#[cfg(test)]\nfn helper() {}\nfn real() { let v = vec![0u8; n]; }\n";
    assert_eq!(
        library_lines(not_a_module)
            .iter()
            .filter(|l| l.contains("vec!["))
            .count(),
        1
    );
}
