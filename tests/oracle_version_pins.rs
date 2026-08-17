//! P4-130: every upstream libjpeg-turbo version this repository provisions is
//! declared in `docs/oracle_versions.tsv`, and every declared version is
//! actually provisioned.
//!
//! The gap this closes is currency drift. Upstream released 3.2.0 on
//! 2026-06-30 and it took until 2026-08-09 for anyone to notice that all
//! thirteen oracle pins still said 3.1.4.1 — because the version lived in
//! thirteen shell fragments across five workflow files and nowhere else. There
//! was nothing to read to answer "which upstream release do our differential
//! gates prove parity with?", so nobody read it.
//!
//! The manifest is that answer, and this gate keeps it true in both
//! directions:
//!
//! * a pin added to a workflow without a manifest row fails here, so a
//!   fourteenth site cannot appear unannounced;
//! * a manifest row no version pin uses fails here, so the file cannot claim a
//!   leg that does not exist — which matters because P4-130's first acceptance
//!   criterion is a *second running leg*, not a documented intention;
//! * the submodule row is cross-checked against the submodule actually checked
//!   out, which is how the split this manifest first recorded was found at all:
//!   `references/libjpeg-turbo` is 3.1.90 (3.2 beta1), not the 3.1.4.1 the
//!   workflows install, so the classic-ABI trace oracles were already running
//!   against a different release than the tool oracles.
//!
//! Currency against *upstream* is a network question and cannot be asked here;
//! `scripts/check_oracle_currency.sh` asks it on a schedule.
//!
//! **Environment:** this reads the repository tree, so it reports an explicit
//! SKIP where that tree is absent — a packaged crate, or a sandboxed target
//! such as `wasm32-wasip1`, which preopens only `.` and `/tmp`. It runs on
//! every native leg.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const MANIFEST: &str = "docs/oracle_versions.tsv";
const WORKFLOW_DIR: &str = ".github/workflows";
const SUBMODULE_CMAKE: &str = "references/libjpeg-turbo/CMakeLists.txt";

/// Roles a manifest row may declare. A typo becomes a failure rather than an
/// unnoticed row nothing checks.
const KNOWN_ROLES: [&str; 3] = ["tool-baseline", "tool-current", "submodule"];

/// Roles provisioned by a workflow pin (as opposed to by a git submodule).
const PROVISIONED_ROLES: [&str; 2] = ["tool-baseline", "tool-current"];

/// Dotted-numeric tokens in a workflow whose first component is this are
/// libjpeg-turbo versions. The 3.x line is the only major upstream ships, and
/// the only dotted tokens in these files today are libjpeg-turbo's 13 pins plus
/// two crate/tool versions (`0.8.0`, `0.36.5`) that this filter excludes.
const ORACLE_MAJOR: &str = "3";

/// One manifest row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Declared {
    role: String,
    version: String,
    provisioned_as: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `false` when the repository tree is not reachable — a packaged crate, or a
/// sandboxed target such as `wasm32-wasip1`. Reported explicitly rather than
/// silently, so a green run always means the gate either ran or said why not.
fn repository_tree_is_readable() -> bool {
    let root: PathBuf = repo_root();
    root.join(MANIFEST).is_file() && root.join(WORKFLOW_DIR).is_dir()
}

fn manifest_rows() -> Vec<Declared> {
    let text: String = std::fs::read_to_string(repo_root().join(MANIFEST))
        .unwrap_or_else(|e| panic!("{MANIFEST} must be readable: {e}"));
    let mut rows: Vec<Declared> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        assert!(
            fields.len() >= 3,
            "{MANIFEST} rows are TAB-separated \
             `role<TAB>version<TAB>provisioned_as<TAB>released<TAB>purpose`; got {line:?}"
        );
        let role: String = fields[0].trim().to_string();
        assert!(
            KNOWN_ROLES.contains(&role.as_str()),
            "unknown role {role:?} in {MANIFEST}; known roles are {KNOWN_ROLES:?}"
        );
        rows.push(Declared {
            role,
            version: fields[1].trim().to_string(),
            provisioned_as: fields[2].trim().to_string(),
        });
    }
    rows
}

fn workflow_files() -> Vec<PathBuf> {
    let dir: PathBuf = repo_root().join(WORKFLOW_DIR);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{WORKFLOW_DIR} must be readable: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "yml" || e == "yaml"))
        .collect();
    files.sort();
    files
}

/// Every dotted-numeric token on the 3.x line, keyed by version, with the
/// `file:line` sites that pin it.
fn version_pins_in_workflows() -> BTreeMap<String, Vec<String>> {
    let mut pins: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in workflow_files() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let name: String = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for (index, line) in text.lines().enumerate() {
            for token in dotted_numeric_tokens(line) {
                if token.split('.').next() == Some(ORACLE_MAJOR) {
                    pins.entry(token)
                        .or_default()
                        .push(format!("{name}:{}", index + 1));
                }
            }
        }
    }
    pins
}

/// Maximal runs of digits and `.` that contain at least one `.`, with any
/// leading/trailing dot trimmed — `3.1.4.1`, `0.36.5`, and the `3.2.0` inside
/// `--branch 3.2.0` or `libjpeg-turbo-official_3.2.0_amd64.deb`.
fn dotted_numeric_tokens(line: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current: String = String::new();
    for ch in line.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            current.push(ch);
            continue;
        }
        push_token(&mut tokens, std::mem::take(&mut current));
    }
    push_token(&mut tokens, current);
    tokens
}

fn push_token(tokens: &mut Vec<String>, token: String) {
    let trimmed: &str = token.trim_matches('.');
    if trimmed.contains('.') && trimmed.split('.').all(|part| !part.is_empty()) {
        tokens.push(trimmed.to_string());
    }
}

#[test]
fn every_workflow_version_pin_is_declared_in_the_manifest() {
    if !repository_tree_is_readable() {
        eprintln!(
            "SKIP: {MANIFEST} / {WORKFLOW_DIR} not readable from {}. This gate \
             inspects the repository tree, which a packaged crate and a \
             sandboxed target (wasm32-wasip1) do not provide.",
            repo_root().display()
        );
        return;
    }

    let declared: BTreeSet<String> = manifest_rows().into_iter().map(|row| row.version).collect();
    let pinned: BTreeMap<String, Vec<String>> = version_pins_in_workflows();

    assert!(
        !pinned.is_empty(),
        "no libjpeg-turbo version pin found in {WORKFLOW_DIR} — the scanner \
         has stopped matching, so this gate would pass no matter what the \
         workflows install"
    );

    let undeclared: Vec<String> = pinned
        .iter()
        .filter(|(version, _)| !declared.contains(*version))
        .map(|(version, sites)| format!("  {version}  pinned at {}", sites.join(", ")))
        .collect();
    assert!(
        undeclared.is_empty(),
        "workflow pins an undeclared libjpeg-turbo version:\n{}\n\n\
         Add a row to {MANIFEST} saying which oracle role it plays and why, or \
         change the pin. A version nothing declares is how 3.1.4.1 stayed \
         current-looking for two months after 3.2.0 shipped.",
        undeclared.join("\n")
    );
}

#[test]
fn every_declared_tool_version_is_actually_provisioned() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }

    let pinned: BTreeMap<String, Vec<String>> = version_pins_in_workflows();
    let unused: Vec<String> = manifest_rows()
        .into_iter()
        .filter(|row| PROVISIONED_ROLES.contains(&row.role.as_str()))
        .filter(|row| !pinned.contains_key(&row.version))
        .map(|row| format!("  {} {} ({})", row.role, row.version, row.provisioned_as))
        .collect();

    assert!(
        unused.is_empty(),
        "{MANIFEST} declares an oracle version no workflow installs:\n{}\n\n\
         P4-130's first criterion is a second *running* leg, not a documented \
         intention: an oracle that is only declared proves nothing, and a \
         divergence it would have caught is indistinguishable from a pass.",
        unused.join("\n")
    );
}

#[test]
fn the_submodule_row_matches_the_checked_out_submodule() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let cmake: PathBuf = repo_root().join(SUBMODULE_CMAKE);
    if !cmake.is_file() {
        eprintln!(
            "SKIP: {SUBMODULE_CMAKE} is absent — the submodule is not checked \
             out. Jobs that use the submodule oracle check out with \
             `submodules: recursive`, and this cross-check runs there."
        );
        return;
    }

    let declared: Vec<Declared> = manifest_rows()
        .into_iter()
        .filter(|row| row.role == "submodule")
        .collect();
    assert_eq!(
        declared.len(),
        1,
        "{MANIFEST} must declare exactly one submodule row; got {declared:?}"
    );

    let actual: String = submodule_version(&cmake);
    assert_eq!(
        declared[0].version, actual,
        "{MANIFEST} says the pinned submodule is {}, but \
         {SUBMODULE_CMAKE} says {actual}. The submodule is the oracle for the \
         classic-ABI trace suites *and* the source every `j*.c:NNN` citation \
         quotes, so a stale row here mislabels both.",
        declared[0].version
    );
}

/// The `set(VERSION x.y.z)` line of the pinned submodule's top-level
/// `CMakeLists.txt` — upstream's own statement of which release the tree is.
fn submodule_version(cmake: &Path) -> String {
    let text: String = std::fs::read_to_string(cmake)
        .unwrap_or_else(|e| panic!("{SUBMODULE_CMAKE} must be readable: {e}"));
    for line in text.lines() {
        let trimmed: &str = line.trim();
        if let Some(rest) = trimmed.strip_prefix("set(VERSION ") {
            if let Some(version) = rest.strip_suffix(')') {
                return version.trim().to_string();
            }
        }
    }
    panic!("no `set(VERSION ...)` line in {SUBMODULE_CMAKE}");
}
