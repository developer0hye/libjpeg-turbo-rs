//! P4-139 criterion 3: no `saturating_*` in an expression that sizes or bounds
//! a memory region — enforced, not merely asserted in prose.
//!
//! Saturating a memory span converts an overflow into `usize::MAX`, which is
//! the single worst value to then hand an allocator or `slice::from_raw_parts`.
//! Three separate live defects had this exact shape (P4-136's plane sizes,
//! P4-137's packed-YUV and `tj3SaveImage8` slices), which is why the criterion
//! asks for a *mechanism* rather than a one-time sweep.
//!
//! **The mechanism is this test plus `docs/sizing_arithmetic_inventory.tsv`.**
//! Every `saturating_mul`/`saturating_add` in library sources must appear in
//! that file with a classification saying why it is not a span. A new one fails
//! here until a human classifies it, and a removed one fails until the file is
//! updated — so the inventory cannot quietly drift out of date the way a
//! comment would.
//!
//! Scope note: `wrapping_*` is deliberately **not** gated, which narrows the
//! criterion's literal wording ("no `saturating_*` or `wrapping_*`"). It is the
//! IDCT's C-parity idiom at 244 sites, where wrapping is the *specified*
//! behaviour being matched rather than an accident, and those kernels are under
//! active optimisation. An inventory that had to be regenerated on every IDCT
//! edit would be deleted within a month, and a gate nobody keeps is worse than
//! a narrower one that holds. The narrowing is recorded in P4-139 rather than
//! left implicit. `saturating_sub` *is* covered — it sizes `repeat_n` lengths
//! and slice indices in this codebase.
//!
//! **Environment:** this reads the repository source tree, so it is skipped
//! where that tree is not reachable — `wasm32-wasip1` under wasmtime (which
//! preopens only `.` and `/tmp`) and a packaged crate, which ships neither
//! `docs/` nor sibling crates. It runs on every native leg, which is where a
//! developer introduces the arithmetic in the first place.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One classified occurrence: how many identical lines, and why it is allowed.
#[derive(Debug, PartialEq, Eq)]
struct Entry {
    count: usize,
    classification: String,
}

const INVENTORY: &str = "docs/sizing_arithmetic_inventory.tsv";

/// Classifications the inventory may use. A typo becomes a failure rather than
/// an unnoticed blank cheque.
/// The operations this gate covers. See the module note on `wrapping_*`.
const SCANNED_OPS: [&str; 3] = ["saturating_mul", "saturating_add", "saturating_sub"];

const KNOWN_CLASSIFICATIONS: [&str; 6] = [
    "value-saturation",
    "counter",
    "bounds-check",
    "geometry-count",
    "capacity-growth",
    "clamped-difference",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Library sources only: `src/` and each crate's `src/`. Test code is out of
/// scope — a saturating span in a test cannot reach a user.
fn scanned_roots() -> Vec<PathBuf> {
    let root: PathBuf = repo_root();
    let mut roots: Vec<PathBuf> = vec![root.join("src")];
    let crates_dir: PathBuf = root.join("crates");
    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        let mut found: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path().join("src"))
            .filter(|p| p.is_dir())
            .collect();
        found.sort();
        roots.extend(found);
    }
    roots
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

/// Every `saturating_mul`/`saturating_add` in library sources, keyed by
/// (file, trimmed source line) so the gate survives line-number drift but still
/// notices a genuinely new expression — or a second copy of an existing one.
fn occurrences_in_sources() -> BTreeMap<(String, String), usize> {
    let mut files: Vec<PathBuf> = Vec::new();
    for root in scanned_roots() {
        walk(&root, &mut files);
    }

    let mut found: BTreeMap<(String, String), usize> = BTreeMap::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            if !SCANNED_OPS.iter().any(|op| line.contains(op)) {
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
        let fields: Vec<&str> = line.splitn(4, '\t').collect();
        assert_eq!(
            fields.len(),
            4,
            "{INVENTORY}:{} must have 4 tab-separated columns, got {}: {line:?}",
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
        map.insert(
            (fields[0].to_string(), fields[3].to_string()),
            Entry {
                count,
                classification,
            },
        );
    }
    map
}

/// The gate. Any disagreement between the sources and the inventory fails,
/// with the exact rows to add or remove.
/// `false` when the repository source tree is not reachable — a packaged crate,
/// or a sandboxed target such as `wasm32-wasip1`. Reported explicitly rather
/// than silently, so a green run always means the gate either ran or said why
/// it did not.
fn repository_tree_is_readable() -> bool {
    let root: PathBuf = repo_root();
    root.join(INVENTORY).is_file() && root.join("crates").is_dir() && root.join("src").is_dir()
}

#[test]
fn saturating_arithmetic_matches_the_classified_inventory() {
    if !repository_tree_is_readable() {
        eprintln!(
            "SKIP: {INVENTORY} / the source tree is not readable from {}. \
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
        .map(|((path, line), n)| format!("  + {path}\t{n}\t<CLASSIFY ME>\t{line}"))
        .collect();
    assert!(
        unlisted.is_empty(),
        "new saturating arithmetic is not classified in {INVENTORY}.\n\n\
         If any of these SIZES OR BOUNDS A MEMORY REGION, it is a bug — convert \
         it to checked arithmetic with a typed error (P4-139). Saturating a span \
         yields usize::MAX, which is the worst value to hand an allocator or \
         slice constructor.\n\n\
         If it genuinely is not a span, add the row with a classification from \
         {KNOWN_CLASSIFICATIONS:?}:\n\n{}\n",
        unlisted.join("\n")
    );

    let stale: Vec<String> = listed
        .keys()
        .filter(|key| !found.contains_key(*key))
        .map(|(path, line)| format!("  - {path}\t{line}"))
        .collect();
    assert!(
        stale.is_empty(),
        "{INVENTORY} lists occurrences that no longer exist. Delete these rows \
         so the inventory keeps meaning something:\n\n{}\n",
        stale.join("\n")
    );

    let miscounted: Vec<String> = found
        .iter()
        .filter_map(|(key, n)| {
            let entry: &Entry = listed.get(key)?;
            (entry.count != *n)
                .then(|| format!("  {}\t{} listed, {n} found\t{}", key.0, entry.count, key.1))
        })
        .collect();
    assert!(
        miscounted.is_empty(),
        "{INVENTORY} counts disagree with the sources — a duplicate was added or \
         removed:\n\n{}\n",
        miscounted.join("\n")
    );
}

/// The gate is worthless if it scans nothing. A refactor that moves sources,
/// or a path bug on another platform, would otherwise turn it green forever.
#[test]
fn the_gate_actually_scans_the_library_sources() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository sources not readable; see the sibling test.");
        return;
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for root in scanned_roots() {
        assert!(root.is_dir(), "scanned root missing: {}", root.display());
        walk(&root, &mut files);
    }
    assert!(
        files.len() > 100,
        "expected the library sources to be hundreds of files, found {}",
        files.len()
    );

    let found: BTreeMap<(String, String), usize> = occurrences_in_sources();
    assert!(
        !found.is_empty(),
        "found zero saturating expressions, which means the scan is broken — \
         the inventory is not empty"
    );
    // Both crates must be reachable, since the capi crate is where the
    // C-ABI spans live.
    assert!(
        found.keys().any(|(p, _)| p.starts_with("src/")),
        "core crate not scanned"
    );
    assert!(
        found.keys().any(|(p, _)| p.starts_with("crates/")),
        "capi crate not scanned"
    );
}
