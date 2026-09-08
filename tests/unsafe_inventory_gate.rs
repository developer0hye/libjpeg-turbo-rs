//! P4-141 criterion 4: the workspace's `unsafe` inventories are committed
//! and gated — CI diffs them against the source tree, so an added, removed
//! or moved `unsafe` fails the build until the owning document says why the
//! site exists, what invariant it relies on, whether a safe caller can reach
//! it, which test would catch a broken invariant, which tools exercise it,
//! and who last reviewed it.
//!
//! **Why an inventory and not a count.** "780 unsafe operations" says
//! nothing about risk — one precondition-free safe wrapper (P4-135)
//! outweighs hundreds of intrinsic calls. The unit here is therefore the
//! *item* an `unsafe` token belongs to (the enclosing `fn`, or the
//! `impl`/`trait` header for `unsafe impl` / `unsafe trait`), and each row
//! carries the seven fields the criterion names. The per-item site count is
//! what makes the diff mechanical: a new `unsafe` inside an already-listed
//! function changes that row's count, so it is a visible inventory edit in
//! the pull request and not a silent addition behind an existing
//! justification.
//!
//! **What is gated.** Two trees, each with the document that owns it (see
//! [`SCOPES`]): every `.rs` under `src/` — the whole root crate, `simd/`
//! included — against `docs/UNSAFE_INVENTORY.md`, and every `.rs` under
//! `crates/libjpeg-turbo-rs-capi/src/` against
//! `docs/UNSAFE_INVENTORY_CAPI.md`. Sites in `#[cfg(test)]` code count too:
//! they say "test-only" in their *Safe caller* cell rather than being
//! exempt, because a test that dereferences a raw pointer is still code Miri
//! and ASan need to see. Nothing shipped may sit outside those two trees:
//! [`the_scanned_roots_are_the_whole_workspace`] enumerates `crates/` from
//! the tree — not from a list — and fails on any `unsafe` in a member's
//! `src/` or `build.rs` that no inventory covers, so a *new* crate is
//! gated the day it lands rather than the day someone remembers.
//!
//! **What is deferred, in code rather than in prose.** [`DEFERRED`] names
//! the sources that hold `unsafe` and are not inventoried yet — today only
//! `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`, the classic `jpeg_*`
//! surface, 501 sites in one 12.9k-line file. The list is checked both ways:
//! a deferred path must exist and must still hold `unsafe` (so a stale entry
//! fails once the file is inventoried or cleaned), and a file that is
//! neither inventoried nor deferred fails as before. A new source therefore
//! cannot slip past by being forgotten — only by an explicit, reviewable
//! edit to this list, which `.github/CODEOWNERS` routes to the maintainer.
//!
//! **How the diff works.** [`unsafe_scan`] lexes each source (comments,
//! strings and literals cannot hide or fake a site — see that module) and
//! attributes each `unsafe` to its enclosing item with a deterministic rule.
//! [`parse_inventory`] reads the markdown into the same (file, item, count)
//! shape and checks every row is complete. [`diff`] then reports, with the
//! exact rows to add or fix: files that hold `unsafe` and are not
//! inventoried, inventory sections whose file no longer holds any, and
//! items whose count moved. The failure message is the patch.
//!
//! **Requiring review.** The gate makes the inventory edit mandatory and
//! therefore visible in the pull-request diff; `.github/CODEOWNERS` names
//! the inventory and this gate so that, once the repository requires
//! code-owner review, an addition cannot merge on CI alone.
//!
//! **Environment:** this reads the repository tree, so it is skipped where
//! that tree is not reachable — `wasm32-wasip1` under wasmtime and a
//! packaged crate. It runs wherever a workflow runs `cargo test --tests`:
//! `ci.yml`'s Integration Tests (ubuntu x86_64) and C Interop (macos
//! aarch64), and all six `cross-arch.yml` legs. Not the Windows job, which
//! builds the workspace and runs two named suites, and not ARMv7.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "helpers/unsafe_scan.rs"]
mod unsafe_scan;
use unsafe_scan::UnsafeSite;

/// One scanned tree and the inventory document that must account for it,
/// both repository-relative.
struct Scope {
    /// Directory to walk for `.rs` sources.
    scan_root: &'static str,
    /// Markdown inventory whose sections must match that walk exactly.
    inventory: &'static str,
}

/// Every tree under the gate. Adding a workspace member with `unsafe` means
/// adding it here *and* writing its inventory; leaving it out is caught by
/// [`the_scanned_roots_are_the_whole_workspace`], which enumerates `crates/`
/// from the tree and fails on any `unsafe` outside these roots and
/// [`DEFERRED`].
const SCOPES: [Scope; 2] = [
    Scope {
        scan_root: "src",
        inventory: "docs/UNSAFE_INVENTORY.md",
    },
    Scope {
        scan_root: "crates/libjpeg-turbo-rs-capi/src",
        inventory: "docs/UNSAFE_INVENTORY_CAPI.md",
    },
];

/// Sources inside a scanned root that hold `unsafe` and are **not** yet
/// inventoried. Each entry is a promise with a deadline, not an exemption:
/// the gate requires the path to exist and to still hold at least one
/// `unsafe`, so an entry that has been inventoried or cleaned up fails until
/// it is removed. Shrinking this list is the remaining work of P4-141
/// criterion 4.
const DEFERRED: [&str; 1] = ["crates/libjpeg-turbo-rs-capi/src/jpeglib.rs"];

/// The table columns every section must declare, in this order. Renaming
/// one is an inventory-format change and fails here on purpose.
const COLUMNS: [&str; 8] = [
    "Item",
    "Sites",
    "Why not safe Rust",
    "Invariant",
    "Safe caller",
    "Regression test",
    "Tool coverage",
    "Reviewer",
];

/// One inventoried item: `| `name` | n | … |`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InventoryRow {
    item: String,
    sites: usize,
    line: usize,
}

/// One `## `path` — N sites` section and its rows.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InventorySection {
    path: String,
    declared_total: usize,
    rows: Vec<InventoryRow>,
    line: usize,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repository_tree_is_readable() -> bool {
    // Deliberately *not* every scope root: this answers "is this a
    // repository checkout or a packaged crate?", and a scope root that has
    // moved is a bug in SCOPES, asserted where the scope is used, rather
    // than a reason to skip the gate for the tree that is still here.
    repo_root().join("src").is_dir() && repo_root().join("docs").is_dir()
}

fn skip_reason() -> String {
    format!(
        "SKIP: the source tree is not readable from {}. This gate inspects \
         repository sources and docs, which a packaged crate and a sandboxed \
         target (wasm32-wasip1) do not provide. It runs wherever a workflow \
         runs `cargo test --tests`.",
        repo_root().display()
    )
}

/// The document path the parser and diff tests below phrase their
/// diagnostics with. Those two functions take the path as an argument, so
/// the fixtures need one; the root crate's is used because its messages are
/// the ones the assertions quote verbatim.
const INVENTORY_FIXTURE: &str = SCOPES[0].inventory;

/// The six prose cells [`skeleton`] emits for a new row, in `COLUMNS[2..]`
/// order. They are the *only* source of those strings: `skeleton` writes
/// them and [`is_placeholder`] rejects them, so a generated section pasted
/// unchanged cannot pass the gate and the two cannot drift apart.
const CELL_TEMPLATES: [&str; 6] = [
    "<why safe Rust cannot express it>",
    "<the invariant>",
    "<yes: entry point / no: pub(crate) only / test-only>",
    "<test name>",
    "<CI jobs and tools>",
    "<who, when>",
];

/// A cell that says nothing. Empty cells, the usual "fill me in later"
/// markers, and anything still written as an angle-bracket template are
/// rejected so a row cannot be committed half-written.
fn is_placeholder(cell: &str) -> bool {
    let trimmed: &str = cell.trim();
    let lower: String = trimmed.to_ascii_lowercase();
    // Any whole cell of the form `<...>` is a template, not prose: it covers
    // the generated ones above and any hand-written `<fill in>` beside them.
    let is_template: bool = trimmed.starts_with('<') && trimmed.ends_with('>') && trimmed.len() > 1;
    is_template
        || lower.is_empty()
        || lower == "-"
        || lower == "—"
        || lower == "?"
        || lower == "tbd"
        || lower.starts_with("todo")
        || lower.starts_with("fixme")
}

/// `## `src/x.rs` — 3 sites` → `("src/x.rs", 3)`.
fn parse_heading(line: &str) -> Option<(String, usize)> {
    let rest: &str = line.strip_prefix("## `")?;
    let close: usize = rest.find('`')?;
    let path: &str = &rest[..close];
    let tail: &str = rest[close + 1..].trim();
    let tail: &str = tail.strip_prefix("—")?.trim();
    let mut words = tail.split_whitespace();
    let count: usize = words.next()?.parse().ok()?;
    let unit: &str = words.next()?;
    let unit_ok: bool = if count == 1 {
        unit == "site"
    } else {
        unit == "sites"
    };
    (unit_ok && words.next().is_none()).then(|| (path.to_string(), count))
}

/// The cells of a `| a | b | … |` table line, trimmed.
fn table_cells(line: &str) -> Option<Vec<String>> {
    let trimmed: &str = line.trim();
    let inner: &str = trimmed.strip_prefix('|')?.strip_suffix('|')?;
    Some(inner.split('|').map(|c| c.trim().to_string()).collect())
}

/// The line that ends the prose preamble and opens the gated region: a
/// markdown horizontal rule on a line of its own. Above it the document
/// explains its own format — headings like `## Columns`, and a table of
/// column descriptions — and none of that is inventory data. Below it every
/// `## ` heading must be a file section, so a typo (a missing backtick, `1
/// sites`) is an error rather than a silently ignored heading.
const GATED_REGION_RULE: &str = "---";

/// Read an inventory document into sections. `inventory` is only the
/// repository-relative path used to phrase the diagnostics, so a failure
/// names the file the reader has to open. Every structural or
/// completeness problem is reported with its line number; a parse that
/// returns `Err` lists all of them rather than the first.
fn parse_inventory(inventory: &str, text: &str) -> Result<Vec<InventorySection>, Vec<String>> {
    let mut sections: Vec<InventorySection> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut saw_header_for_current: bool = false;
    let mut in_gated_region: bool = false;
    let mut in_code_fence: bool = false;

    for (index, raw) in text.lines().enumerate() {
        let lineno: usize = index + 1;
        let line: &str = raw.trim_end_matches('\r');

        if !in_gated_region {
            in_gated_region = line.trim() == GATED_REGION_RULE;
            continue;
        }

        // A fenced block is an example, not data. Without this a documented
        // sample row — which `skeleton` tells authors to paste — would be
        // read as a real one.
        if line.trim_start().starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }

        if line.starts_with("## ") {
            if let Some(previous) = sections.last() {
                if !saw_header_for_current {
                    errors.push(format!(
                        "{inventory}:{}: section `{}` has no column header row",
                        previous.line, previous.path
                    ));
                }
            }
            match parse_heading(line) {
                Some((path, declared_total)) => {
                    if sections.iter().any(|s| s.path == path) {
                        errors.push(format!(
                            "{inventory}:{lineno}: `{path}` is inventoried twice"
                        ));
                    }
                    sections.push(InventorySection {
                        path,
                        declared_total,
                        rows: Vec::new(),
                        line: lineno,
                    });
                    saw_header_for_current = false;
                }
                None => errors.push(format!(
                    "{inventory}:{lineno}: a `## ` heading must read \
                     `## `<path>` — <N> sites` (or `1 site`): {line}"
                )),
            }
            continue;
        }

        let Some(cells) = table_cells(line) else {
            continue;
        };
        let Some(section) = sections.last_mut() else {
            errors.push(format!(
                "{inventory}:{lineno}: table row before any `## ` file section"
            ));
            continue;
        };
        if cells
            .iter()
            .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue; // the `|---|---|` separator
        }
        if cells.len() == COLUMNS.len() && cells.iter().zip(COLUMNS).all(|(c, h)| c == h) {
            saw_header_for_current = true;
            continue;
        }
        if cells.len() != COLUMNS.len() {
            errors.push(format!(
                "{inventory}:{lineno}: expected {} cells ({}), found {}",
                COLUMNS.len(),
                COLUMNS.join(" | "),
                cells.len()
            ));
            continue;
        }
        let item: &str = match cells[0].strip_prefix('`').and_then(|s| s.strip_suffix('`')) {
            Some(item) if !item.is_empty() => item,
            _ => {
                errors.push(format!(
                    "{inventory}:{lineno}: the Item cell must be the enclosing item in \
                     backticks, e.g. `` `decode_baseline_planes` ``: {}",
                    cells[0]
                ));
                continue;
            }
        };
        let sites: usize = match cells[1].parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                errors.push(format!(
                    "{inventory}:{lineno}: Sites must be a positive integer: {}",
                    cells[1]
                ));
                continue;
            }
        };
        for (cell, column) in cells[2..].iter().zip(&COLUMNS[2..]) {
            if is_placeholder(cell) {
                errors.push(format!(
                    "{inventory}:{lineno}: `{item}` has an empty or placeholder \
                     `{column}` cell — every row must be complete"
                ));
            }
        }
        if section.rows.iter().any(|r| r.item == item) {
            errors.push(format!(
                "{inventory}:{lineno}: `{item}` is listed twice under `{}`",
                section.path
            ));
        }
        section.rows.push(InventoryRow {
            item: item.to_string(),
            sites,
            line: lineno,
        });
    }

    if !in_gated_region {
        errors.push(format!(
            "{inventory}: no `{GATED_REGION_RULE}` rule — the file must separate its \
             prose preamble from the gated file sections with a horizontal rule on a \
             line of its own; everything after the first one is inventory data"
        ));
    }
    if let Some(last) = sections.last() {
        if !saw_header_for_current {
            errors.push(format!(
                "{inventory}:{}: section `{}` has no column header row",
                last.line, last.path
            ));
        }
    }
    for section in &sections {
        let sum: usize = section.rows.iter().map(|r| r.sites).sum();
        if sum != section.declared_total {
            errors.push(format!(
                "{inventory}:{}: `{}` declares {} sites but its rows sum to {sum}",
                section.line, section.path, section.declared_total
            ));
        }
    }
    if errors.is_empty() {
        Ok(sections)
    } else {
        Err(errors)
    }
}

/// Every `unsafe` site under `<root>/<scan_root>`, keyed by
/// repository-relative path, with the [`DEFERRED`] sources removed —
/// those are accounted for by [`deferrals_are_live_and_named`] instead.
fn scan_sources(root: &Path, scan_root: &str) -> BTreeMap<String, Vec<UnsafeSite>> {
    let mut files: Vec<PathBuf> = Vec::new();
    unsafe_scan::rust_sources(&root.join(scan_root), &mut files);
    files
        .into_iter()
        .map(|path| {
            (
                unsafe_scan::relative(root, &path),
                unsafe_scan::unsafe_sites(&path),
            )
        })
        .filter(|(path, sites)| !sites.is_empty() && !DEFERRED.contains(&path.as_str()))
        .collect()
}

/// (item → count) for one scanned file, in first-seen order.
fn counts_by_item(sites: &[UnsafeSite]) -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for site in sites {
        match counts.iter_mut().find(|(item, _)| *item == site.item) {
            Some((_, n)) => *n += 1,
            None => counts.push((site.item.clone(), 1)),
        }
    }
    counts
}

/// The markdown a missing section should contain, ready to paste.
fn skeleton(path: &str, sites: &[UnsafeSite]) -> String {
    let unit: &str = if sites.len() == 1 { "site" } else { "sites" };
    let mut out: String = format!("## `{path}` — {} {unit}\n\n", sites.len());
    out.push_str(&format!("| {} |\n", COLUMNS.join(" | ")));
    out.push_str(&format!("|{}\n", "---|".repeat(COLUMNS.len())));
    for (item, n) in counts_by_item(sites) {
        out.push_str(&format!(
            "| `{item}` | {n} | {} |\n",
            CELL_TEMPLATES.join(" | ")
        ));
    }
    out
}

/// Everything that differs between the inventory and the tree. Empty means
/// the inventory is exact.
fn diff(
    inventory: &str,
    sections: &[InventorySection],
    scanned: &BTreeMap<String, Vec<UnsafeSite>>,
) -> Vec<String> {
    let mut problems: Vec<String> = Vec::new();

    for (path, sites) in scanned {
        let Some(section) = sections.iter().find(|s| &s.path == path) else {
            problems.push(format!(
                "`{path}` holds {} `unsafe` site(s) and is not in {inventory}. \
                 Add this section and fill every cell:\n\n{}",
                sites.len(),
                skeleton(path, sites)
            ));
            continue;
        };
        let found: Vec<(String, usize)> = counts_by_item(sites);
        for (item, n) in &found {
            match section.rows.iter().find(|r| &r.item == item) {
                None => problems.push(format!(
                    "`{path}`: `{item}` holds {n} `unsafe` site(s) but has no row in \
                     {inventory} (section at line {}). Sites:\n{}",
                    section.line,
                    site_list(sites, item)
                )),
                Some(row) if row.sites != *n => problems.push(format!(
                    "`{path}`: `{item}` holds {n} `unsafe` site(s) but {inventory}:{} \
                     says {}. Re-justify the row and update its count. Sites:\n{}",
                    row.line,
                    row.sites,
                    site_list(sites, item)
                )),
                Some(_) => {}
            }
        }
        for row in &section.rows {
            if !found.iter().any(|(item, _)| item == &row.item) {
                problems.push(format!(
                    "`{path}`: {inventory}:{} lists `{}` but no `unsafe` is attributed \
                     to it any more — remove the row (or fix the item name; the \
                     attributed items are: {})",
                    row.line,
                    row.item,
                    found
                        .iter()
                        .map(|(item, _)| format!("`{item}`"))
                        .collect::<Vec<String>>()
                        .join(", ")
                ));
            }
        }
    }

    for section in sections {
        if !scanned.contains_key(&section.path) {
            problems.push(format!(
                "{inventory}:{}: `{}` is inventoried but holds no `unsafe` (or does not \
                 exist) — remove the section, or move it to a \"retired\" note outside \
                 the gated format",
                section.line, section.path
            ));
        }
    }
    problems
}

fn site_list(sites: &[UnsafeSite], item: &str) -> String {
    sites
        .iter()
        .filter(|s| s.item == item)
        .map(|s| format!("    {}: {}", s.line, s.text))
        .collect::<Vec<String>>()
        .join("\n")
}

/// The gate. Reads each scanned tree and the inventory that owns it and
/// requires them to agree item by item.
#[test]
fn every_inventory_matches_every_unsafe_site_in_its_tree() {
    if !repository_tree_is_readable() {
        eprintln!("{}", skip_reason());
        return;
    }
    let root: PathBuf = repo_root();
    for scope in &SCOPES {
        let inventory: &str = scope.inventory;
        assert!(
            root.join(scope.scan_root).is_dir(),
            "SCOPES names `{}`, which is not a directory — a scanned tree that moved \
             must move in SCOPES too, not silently stop being gated",
            scope.scan_root
        );
        let inventory_path: PathBuf = root.join(inventory);
        let text: String = std::fs::read_to_string(&inventory_path).unwrap_or_else(|e| {
            panic!(
                "{inventory} is missing ({e}). P4-141 criterion 4 requires the \
                 `unsafe` inventory for `{}` to be committed; see the doc comment \
                 of tests/unsafe_inventory_gate.rs for the format.",
                scope.scan_root
            )
        });
        let sections: Vec<InventorySection> =
            parse_inventory(inventory, &text).unwrap_or_else(|errors| {
                panic!(
                    "{inventory} is malformed or incomplete ({} problem(s)):\n\n  {}\n",
                    errors.len(),
                    errors.join("\n  ")
                )
            });
        for section in &sections {
            assert!(
                !DEFERRED.contains(&section.path.as_str()),
                "{inventory}:{}: `{}` has an inventory section now — remove it from \
                 DEFERRED in tests/unsafe_inventory_gate.rs so the gate diffs it",
                section.line,
                section.path
            );
            assert!(
                root.join(&section.path).is_file(),
                "{inventory}:{}: `{}` does not exist; if the file was renamed or \
                 removed, update the section",
                section.line,
                section.path
            );
            assert!(
                section.path.starts_with(&format!("{}/", scope.scan_root)),
                "{inventory}:{}: `{}` is outside `{}`, the tree this document owns — \
                 a section in the wrong inventory is invisible to the other one's \
                 diff",
                section.line,
                section.path,
                scope.scan_root
            );
        }

        let scanned: BTreeMap<String, Vec<UnsafeSite>> = scan_sources(&root, scope.scan_root);
        assert!(
            !scanned.is_empty(),
            "no `unsafe` found under {} — the scanner is broken, the crate did not \
             become safe Rust overnight",
            scope.scan_root
        );

        let problems: Vec<String> = diff(inventory, &sections, &scanned);
        assert!(
            problems.is_empty(),
            "{inventory} does not match `{}` (P4-141 criterion 4): {} problem(s).\n\n\
             Every `unsafe` in that tree must have a complete inventory row, and the \
             row's site count must match. Fix the inventory, not the gate:\n\n{}\n",
            scope.scan_root,
            problems.len(),
            problems.join("\n\n")
        );
    }
}

/// A deferral is a promise with a deadline. Each [`DEFERRED`] path must
/// exist, must sit inside a scanned root — otherwise it silences nothing and
/// only reads as though it does — and must still hold at least one `unsafe`,
/// so the entry has to be deleted the moment the file is inventoried or made
/// safe. Without this the list would be a place for stale exemptions to
/// accumulate, which is the failure mode the inventory exists to prevent.
#[test]
fn deferrals_are_live_and_named() {
    if !repository_tree_is_readable() {
        eprintln!("{}", skip_reason());
        return;
    }
    let root: PathBuf = repo_root();
    for deferred in DEFERRED {
        let path: PathBuf = root.join(deferred);
        assert!(
            path.is_file(),
            "DEFERRED lists `{deferred}`, which does not exist — remove the entry \
             (tests/unsafe_inventory_gate.rs)"
        );
        assert!(
            SCOPES
                .iter()
                .any(|scope| deferred.starts_with(&format!("{}/", scope.scan_root))),
            "DEFERRED lists `{deferred}`, which is outside every scanned root, so it \
             defers nothing — remove the entry or add its tree to SCOPES"
        );
        assert!(
            !unsafe_scan::unsafe_sites(&path).is_empty(),
            "DEFERRED lists `{deferred}`, which no longer holds any `unsafe` — it is \
             either inventoried or safe now, so remove the entry and let the gate \
             cover it"
        );
    }
}

/// The scanned roots plus [`DEFERRED`] are the *whole* workspace: no
/// `unsafe` may live in a crate this gate never walks. `crates/` is
/// **enumerated from the tree**, not listed here, because a hard-coded list
/// makes the completeness claim true only of today's members — a new crate
/// with an `unsafe` would be neither inventoried, nor deferred, nor
/// detected. `known` is checked afterwards so a rename is still loud.
///
/// Each member's `src/` and its `build.rs`: `tests/`, `benches/`,
/// `examples/` and `fuzz/` are out of scope for the same reason the root
/// crate's inventory stops at `src/` — they are not shipped. A `build.rs`
/// is: it runs on the consumer's machine.
#[test]
fn the_scanned_roots_are_the_whole_workspace() {
    if !repository_tree_is_readable() {
        eprintln!("{}", skip_reason());
        return;
    }
    let root: PathBuf = repo_root();
    let known: [&str; 3] = [
        "libjpeg-turbo-rs-capi",
        "libjpeg-turbo-rs-image",
        "libjpeg-turbo-rs-wasm",
    ];
    let mut seen: Vec<String> = Vec::new();
    let entries = std::fs::read_dir(root.join("crates")).expect("crates/ must be readable");
    for entry in entries {
        let member: PathBuf = entry.expect("a readable directory entry").path();
        if !member.join("Cargo.toml").is_file() {
            continue;
        }
        seen.push(
            member
                .file_name()
                .expect("a named directory")
                .to_string_lossy()
                .into_owned(),
        );
        let mut files: Vec<PathBuf> = Vec::new();
        unsafe_scan::rust_sources(&member.join("src"), &mut files);
        let build_script: PathBuf = member.join("build.rs");
        if build_script.is_file() {
            files.push(build_script);
        }
        for file in files {
            let relative: String = unsafe_scan::relative(&root, &file);
            // A file its own inventory already accounts for, or one this
            // criterion has explicitly deferred, is covered elsewhere.
            let covered: bool = SCOPES
                .iter()
                .any(|scope| relative.starts_with(&format!("{}/", scope.scan_root)))
                || DEFERRED.contains(&relative.as_str());
            if covered {
                continue;
            }
            let sites: Vec<UnsafeSite> = unsafe_scan::unsafe_sites(&file);
            assert!(
                sites.is_empty(),
                "`{relative}` holds {} `unsafe` site(s) and no inventory covers it. \
                 Add its tree to SCOPES with a document of its own — do not delete \
                 this assertion.",
                sites.len()
            );
        }
    }
    for member in known {
        assert!(
            seen.iter().any(|name| name == member),
            "`crates/{member}` is gone or renamed; update this test and SCOPES together"
        );
    }
}

/// The attribution rule, on the shapes the crate contains and on the ones
/// that could fool it: plain and qualified `fn`s, `extern "C"` and `default`
/// qualifiers, macro-template names (qualified by their `macro_rules!`, so
/// two macros in one file do not share a row), `unsafe impl` and
/// `unsafe trait`, an `unsafe fn` *type* in a body, prose in `//` and
/// `/* */` comments — including a commented-out `fn` and the phrase
/// `unsafe impl` in a trailing comment, both of which named the row before
/// attribution moved to the comment-masked copy — and a module-level site.
///
/// Each line here kills a mutant: deleting the qualifier check, the
/// comment masking, the `trait` arm or the macro qualification makes this
/// test fail. That matters more than usual: this is a gate, so a vacuous
/// test here is a gate that only appears to hold.
#[test]
fn attribution_handles_the_shapes_the_crate_uses() {
    let text: &str = "\
static X: usize = unsafe { init() };
pub(crate) unsafe fn a() {
    unsafe { core::ptr::read(p) }
}
fn b() {
    // unsafe fn not_a_decl() — prose, not code
    let k: unsafe fn (&mut u64) = h;
    /*
    pub fn ghost() {}
    */
    unsafe { c() } // mirrors the unsafe impl Send for Ghost above
}
pub extern \"C\" fn c_abi() {
    unsafe { d() }
}
default unsafe fn defaulted() {
    unsafe { e() }
}
macro_rules! first_macro {
    () => {
        #[target_feature(enable = \"neon\")]
        unsafe fn $inner(y: &[u8]) {
            let v = unsafe { vld1q_u8(y.as_ptr()) };
        }
    };
}
macro_rules! second_macro {
    () => {
        unsafe fn $inner(y: &[u8]) {
            let v = unsafe { vld1q_u8(y.as_ptr()) };
        }
    };
}
unsafe impl<T: Send + Sync + 'static> Sync for OnceBox<T> {}
unsafe trait Contiguous {}
impl<T> OnceBox<T> {
    pub fn get(&self) -> &T { unsafe { &*self.0 } }
}
";
    let sites: Vec<UnsafeSite> = unsafe_scan::unsafe_sites_in(text);
    let keyed: Vec<(usize, &str)> = sites.iter().map(|s| (s.line, s.item.as_str())).collect();
    assert_eq!(
        keyed,
        vec![
            (1, "(module)"),
            (2, "a"),
            (3, "a"),
            // `unsafe fn (…)` as a *type* is not a declaration, and neither
            // the commented-out `fn ghost` nor the `unsafe impl` phrase in
            // the trailing comment takes the row.
            (7, "b"),
            (11, "b"),
            (14, "c_abi"),
            (16, "defaulted"),
            (17, "defaulted"),
            // Same template name in two macros, two keys.
            (22, "first_macro::$inner"),
            (23, "first_macro::$inner"),
            (29, "second_macro::$inner"),
            (30, "second_macro::$inner"),
            (34, "impl Sync for OnceBox"),
            (35, "trait Contiguous"),
            (37, "get"),
        ]
    );
    assert_eq!(
        counts_by_item(&sites),
        vec![
            ("(module)".to_string(), 1),
            ("a".to_string(), 2),
            ("b".to_string(), 2),
            ("c_abi".to_string(), 1),
            ("defaulted".to_string(), 2),
            ("first_macro::$inner".to_string(), 2),
            ("second_macro::$inner".to_string(), 2),
            ("impl Sync for OnceBox".to_string(), 1),
            ("trait Contiguous".to_string(), 1),
            ("get".to_string(), 1),
        ]
    );
}

/// A `fn` declaration written inside a literal must not take the row. This
/// is the same hazard as the commented-out `fn` above, one layer down: a
/// multi-line raw string is code as far as the file is concerned, and a
/// fixture or an error message quoting Rust source is a normal thing for
/// this crate to contain. Removing the literal masking fails this test.
#[test]
fn a_fn_inside_a_literal_does_not_take_the_row() {
    let text: &str = "\
fn real_one() {
    let fixture = r#\"
        pub fn ghost_in_a_raw_string() {}
    \"#;
    let quoted = \"fn ghost_in_a_string() {}\";
    unsafe { g() }
}
";
    let sites: Vec<UnsafeSite> = unsafe_scan::unsafe_sites_in(text);
    assert_eq!(
        sites.iter().map(|s| s.item.as_str()).collect::<Vec<&str>>(),
        vec!["real_one"]
    );
}

/// The shapes attribution does **not** resolve, pinned so a change to them
/// is visible. `declared_fn_name` accepts a declaration only when every
/// token before `fn` is a qualifier, so a whole `impl` on one line and an
/// attribute sharing the declaration's line both fall through to the
/// preceding declaration. No source under `src/` is written either way —
/// the gate reports zero `(module)` sites and `cargo fmt` produces neither —
/// and widening the rule would risk matching `fn` inside an expression, so
/// this is a documented limitation rather than a bug. Deleting the
/// qualifier check fails this test.
#[test]
fn declarations_sharing_a_line_with_other_code_fall_through() {
    let text: &str = "\
fn earlier() {}
impl T { fn inline_one_liner() { unsafe { g() } } }
#[inline] fn same_line_attribute() { unsafe { h() } }
";
    let sites: Vec<UnsafeSite> = unsafe_scan::unsafe_sites_in(text);
    assert_eq!(
        sites.iter().map(|s| s.item.as_str()).collect::<Vec<&str>>(),
        vec!["earlier", "earlier"]
    );
}

/// The documented shape parses to the expected sections and rows — and the
/// prose preamble above the `---` rule, which explains the format with its
/// own `## ` headings and its own table, contributes nothing.
#[test]
fn parser_accepts_the_documented_shape() {
    let text: &str = "\
# Inventory

prose

## Columns

| Column | What goes in it |
|---|---|
| Item | The enclosing item. |

## `src/not_a_section.rs` — 9 sites

---

## `src/a.rs` — 3 sites

| Item | Sites | Why not safe Rust | Invariant | Safe caller | Regression test | Tool coverage | Reviewer |
|---|---|---|---|---|---|---|---|
| `f` | 2 | raw pointer write | `len >= 8` checked by caller | no: pub(crate) | `f_matches_scalar` | ASan, Miri | x, 2026-09-08 |
| `impl Sync for T` | 1 | interior mutability | published once via CAS | yes | `t_is_sync` | Miri | x, 2026-09-08 |

## `src/b.rs` — 1 site

| Item | Sites | Why not safe Rust | Invariant | Safe caller | Regression test | Tool coverage | Reviewer |
|---|---|---|---|---|---|---|---|
| `g` | 1 | intrinsics | AVX2 checked at dispatch | yes: `decode()` | `g_matches_scalar` | ASan (AVX2 leg) | x, 2026-09-08 |
";
    let sections: Vec<InventorySection> =
        parse_inventory(INVENTORY_FIXTURE, text).expect("well-formed inventory");
    assert_eq!(sections.len(), 2);
    assert_eq!(sections[0].path, "src/a.rs");
    assert_eq!(sections[0].declared_total, 3);
    assert_eq!(
        sections[0]
            .rows
            .iter()
            .map(|r| (r.item.as_str(), r.sites))
            .collect::<Vec<(&str, usize)>>(),
        vec![("f", 2), ("impl Sync for T", 1)]
    );
    assert_eq!(sections[1].path, "src/b.rs");
    assert_eq!(sections[1].declared_total, 1);
    assert_eq!(sections[1].rows[0].item, "g");
}

/// Each way a row can be incomplete or inconsistent is a named error, so a
/// half-written inventory cannot pass. Line numbers are reported so the
/// message points at the row to fix.
#[test]
fn parser_rejects_malformed_and_placeholder_rows() {
    let header: &str = "\
| Item | Sites | Why not safe Rust | Invariant | Safe caller | Regression test | Tool coverage | Reviewer |
|---|---|---|---|---|---|---|---|
";
    // Every case is written inside the gated region: `parse_inventory` reads
    // nothing until the `---` rule, so the fixture has to open it.
    let check = |body: &str, expect: &str| {
        let gated: String = format!("---\n{body}");
        let errors: Vec<String> = parse_inventory(INVENTORY_FIXTURE, &gated)
            .expect_err(&format!("must be rejected: {body}"));
        assert!(
            errors.iter().any(|e| e.contains(expect)),
            "expected an error containing {expect:?}, got {errors:#?}"
        );
    };

    // Placeholders and empties in the prose cells.
    check(
        &format!(
            "## `src/a.rs` — 1 site\n\n{header}| `f` | 1 | TODO | inv | yes | t | asan | me |\n"
        ),
        "placeholder `Why not safe Rust`",
    );
    check(
        &format!("## `src/a.rs` — 1 site\n\n{header}| `f` | 1 | why | inv | yes | t |  | me |\n"),
        "placeholder `Tool coverage`",
    );
    check(
        &format!(
            "## `src/a.rs` — 1 site\n\n{header}| `f` | 1 | why | inv | yes | t | asan | ? |\n"
        ),
        "placeholder `Reviewer`",
    );
    // Structural problems.
    check(
        &format!(
            "## `src/a.rs` — 2 sites\n\n{header}| `f` | 1 | why | inv | yes | t | asan | me |\n"
        ),
        "declares 2 sites but its rows sum to 1",
    );
    check(
        &format!("## `src/a.rs` — 1 site\n\n{header}| f | 1 | why | inv | yes | t | asan | me |\n"),
        "must be the enclosing item in backticks",
    );
    check(
        &format!(
            "## `src/a.rs` — 1 site\n\n{header}| `f` | 0 | why | inv | yes | t | asan | me |\n"
        ),
        "positive integer",
    );
    check(
        &format!("## `src/a.rs` — 1 site\n\n{header}| `f` | 1 | why | inv | yes | t | asan |\n"),
        "expected 8 cells",
    );
    check(
        &format!(
            "## `src/a.rs` — 2 sites\n\n{header}| `f` | 1 | why | inv | yes | t | asan | me |\n\
             | `f` | 1 | why | inv | yes | t | asan | me |\n"
        ),
        "listed twice",
    );
    check(
        &format!(
            "## `src/a.rs` — 1 site\n\n{header}| `f` | 1 | why | inv | yes | t | asan | me |\n\n\
             ## `src/a.rs` — 1 site\n\n{header}| `g` | 1 | why | inv | yes | t | asan | me |\n"
        ),
        "inventoried twice",
    );
    check(
        "## `src/a.rs` — 1 site\n\n| `f` | 1 | why | inv | yes | t | asan | me |\n",
        "no column header row",
    );
    check(
        &format!("## src/a.rs (1 site)\n\n{header}"),
        "heading must read",
    );
    check(
        &format!("## `src/a.rs` — 2 site\n\n{header}"),
        "heading must read",
    );
    check(
        "| `f` | 1 | why | inv | yes | t | asan | me |\n",
        "before any `## ` file section",
    );

    // A file with no `---` rule is all preamble, so every section would be
    // silently invisible to the gate. That is the one failure this format
    // cannot report per line, so it is reported for the file.
    let no_rule: String = format!(
        "# Inventory\n\n## `src/a.rs` — 1 site\n\n{header}\
         | `f` | 1 | why | inv | yes | t | asan | me |\n"
    );
    let errors: Vec<String> =
        parse_inventory(INVENTORY_FIXTURE, &no_rule).expect_err("a file with no rule");
    assert!(
        errors.iter().any(|e| e.contains("no `---` rule")),
        "{errors:#?}"
    );
}

/// The rows the failure message hands you are a *template*, not an answer:
/// pasting `skeleton` output unchanged must still fail, or the gate would
/// accept a new `unsafe` site carrying no justification at all. Generated
/// from the real `skeleton`, so a template string added there without a
/// matching rejection fails here.
#[test]
fn a_pasted_skeleton_is_still_rejected() {
    let sites: Vec<UnsafeSite> = vec![
        UnsafeSite {
            line: 3,
            item: "f".to_string(),
            text: "unsafe { read(p) }".to_string(),
        },
        UnsafeSite {
            line: 9,
            item: "f".to_string(),
            text: "unsafe fn f()".to_string(),
        },
    ];
    let pasted: String = format!("---\n\n{}", skeleton("src/a.rs", &sites));
    let errors: Vec<String> =
        parse_inventory(INVENTORY_FIXTURE, &pasted).expect_err("an unfilled skeleton");
    for column in &COLUMNS[2..] {
        assert!(
            errors
                .iter()
                .any(|e| e.contains(&format!("placeholder `{column}` cell"))),
            "`{column}` was accepted from the skeleton: {errors:#?}"
        );
    }

    // The section shape itself is right, so the *only* complaints are the
    // six empty cells — the skeleton is paste-ready, just not fill-free.
    assert_eq!(errors.len(), COLUMNS.len() - 2, "{errors:#?}");
}

/// The diff names each kind of drift and carries the rows to paste, so the
/// failure message is actionable and the report path is not vacuous.
#[test]
fn diff_reports_added_removed_moved_and_stale_sites() {
    let site = |line: usize, item: &str| UnsafeSite {
        line,
        item: item.to_string(),
        text: format!("unsafe {{ line {line} }}"),
    };
    let row = |item: &str, sites: usize| InventoryRow {
        item: item.to_string(),
        sites,
        line: 10,
    };
    let inventory: Vec<InventorySection> = vec![
        InventorySection {
            path: "src/a.rs".into(),
            declared_total: 3,
            rows: vec![row("f", 2), row("gone", 1)],
            line: 5,
        },
        InventorySection {
            path: "src/stale.rs".into(),
            declared_total: 1,
            rows: vec![row("h", 1)],
            line: 20,
        },
    ];
    let mut scanned: BTreeMap<String, Vec<UnsafeSite>> = BTreeMap::new();
    scanned.insert(
        "src/a.rs".into(),
        vec![
            site(3, "f"),
            site(4, "f"),
            site(9, "f"),
            site(30, "moved_here"),
        ],
    );
    scanned.insert("src/new.rs".into(), vec![site(7, "k"), site(8, "k")]);

    let problems: Vec<String> = diff(INVENTORY_FIXTURE, &inventory, &scanned);
    let joined: String = problems.join("\n");
    assert_eq!(problems.len(), 5, "{joined}");
    assert!(
        joined.contains(
            "`src/a.rs`: `f` holds 3 `unsafe` site(s) but docs/UNSAFE_INVENTORY.md:10 says 2"
        ),
        "{joined}"
    );
    assert!(
        joined.contains("`src/a.rs`: `moved_here` holds 1 `unsafe` site(s) but has no row"),
        "{joined}"
    );
    assert!(
        joined.contains("lists `gone` but no `unsafe` is attributed to it"),
        "{joined}"
    );
    assert!(
        joined.contains("`src/new.rs` holds 2 `unsafe` site(s) and is not in"),
        "{joined}"
    );
    assert!(joined.contains("## `src/new.rs` — 2 sites"), "{joined}");
    assert!(joined.contains("| `k` | 2 |"), "{joined}");
    assert!(
        joined.contains(
            "docs/UNSAFE_INVENTORY.md:20: `src/stale.rs` is inventoried but holds no `unsafe`"
        ),
        "{joined}"
    );

    // And nothing to report when they agree.
    let exact: Vec<InventorySection> = vec![
        InventorySection {
            path: "src/a.rs".into(),
            declared_total: 4,
            rows: vec![row("f", 3), row("moved_here", 1)],
            line: 5,
        },
        InventorySection {
            path: "src/new.rs".into(),
            declared_total: 2,
            rows: vec![row("k", 2)],
            line: 9,
        },
    ];
    assert!(diff(INVENTORY_FIXTURE, &exact, &scanned).is_empty());
}
