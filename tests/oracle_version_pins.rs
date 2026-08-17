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
//! A second, independent dimension lives in the back half of this file: the
//! two tool legs have to run the *same* C-ABI oracle suites. Versions being
//! declared says nothing about which suites actually meet them, and the C-ABI
//! crate's suites are selected by name — so a suite added to one leg is
//! invisible to the other. That gate, its classifier and the job-block parse
//! it rests on are documented at their own banner below.
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
const KNOWN_ROLES: [&str; 4] = [
    "tool-baseline",
    "tool-current",
    "trace-current",
    "submodule",
];

/// How a row's version has to appear in a workflow before the row counts as
/// backed by a leg that runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteShape {
    /// Any provisioning site will do — upstream's official deb, a source
    /// clone, whatever the role's own step chooses.
    AnyProvisioning,
    /// A source clone specifically. Upstream's official deb ships
    /// `JPEG_LIB_VERSION 62`, so no deb can back a v8-ABI oracle however
    /// current its version is; only a `WITH_JPEG8=1` build of the sources can.
    /// Without this distinction the deb installed for `tool-current` would
    /// satisfy a `trace-current` row of the same version, and deleting the
    /// build step would leave the manifest still claiming the leg.
    SourceClone,
}

/// Roles provisioned by a workflow pin (as opposed to by a git submodule),
/// with the site shape each one requires.
const PROVISIONED_ROLES: [(&str, SiteShape); 3] = [
    ("tool-baseline", SiteShape::AnyProvisioning),
    ("tool-current", SiteShape::AnyProvisioning),
    ("trace-current", SiteShape::SourceClone),
];

/// Dotted-numeric tokens in a workflow whose first component is this are
/// libjpeg-turbo versions.
///
/// The invariant, rather than an inventory that goes stale as pins are added:
/// upstream ships one major line, and no dotted token in these files that is
/// *not* a libjpeg-turbo version has major 3. The others are toolchain and
/// action versions (`1.87`, `1.88`, `1.90`, `24.04`, `0.36.5`, …), none of
/// which collide. A future `3.x` of something else would have to be excluded
/// here explicitly — which is the right failure, since it would otherwise be
/// read as an undeclared oracle pin.
const ORACLE_MAJOR: &str = "3";

/// The workflow carrying both tool legs.
const CI_WORKFLOW: &str = ".github/workflows/ci.yml";

/// The `tool-baseline` leg (3.1.4.1) and the `tool-current` leg (3.2.0).
const BASELINE_LEG_JOB: &str = "test-integration";
const CURRENT_LEG_JOB: &str = "test-integration-current-oracle";

/// The workflow carrying the exhaustive matrices: the 12,230-case transform
/// cross-product, the crop grid, and the tj comp/decomp matrices. They run
/// weekly rather than per pull request, which is why the release they measure
/// is easy to leave behind — nothing red ever points at them.
const FULL_PARITY_WORKFLOW: &str = ".github/workflows/full-c-parity.yml";

/// `(baseline leg, current-parity leg)`, one pair per host architecture the
/// exhaustive matrices run on. The pairing gate below reads the pairs rather
/// than a single job name because these matrices compare *our SIMD output*
/// against C's, so x86_64 and aarch64 are different measurements, not two runs
/// of the same one.
const FULL_PARITY_LEG_PAIRS: [(&str, &str); 2] = [
    ("full-c-parity-x86", "full-c-parity-x86-current-oracle"),
    ("full-c-parity-arm64", "full-c-parity-arm64-current-oracle"),
];

/// A `cargo test` invocation carrying this selects the C-ABI crate.
const CAPI_PACKAGE: &str = "-p libjpeg-turbo-rs-capi";
const CAPI_TEST_DIR: &str = "crates/libjpeg-turbo-rs-capi/tests";

/// A `cargo test` invocation carrying this selects *some* package explicitly,
/// so an invocation without it runs the workspace default member — the root
/// crate, whose integration suites live in [`ROOT_TEST_DIR`].
const PACKAGE_SELECTOR: &str = "-p ";
const ROOT_TEST_DIR: &str = "tests";

/// Source markers that mean "what this suite asserts depends on the C
/// libjpeg-turbo it ran against": the environment variables that name an
/// oracle prefix, the helpers that compile and run one, and the stock tools a
/// suite shells out to.
///
/// Classifying from each suite's own source, rather than from a list kept
/// here, is what stops the pairing gate below from rotting: a suite that
/// *gains* a C comparison is reclassified by the same commit that gives it
/// one, with nothing to remember to update.
const ORACLE_MARKERS: [&str; 10] = [
    "LIBJPEG_TURBO_PREFIX",
    "LIBJPEG_TURBO_REFERENCE_DIR",
    "build_oracle",
    "build_classic_oracle",
    "run_oracle",
    "find_turbojpeg_dev",
    "find_libjpeg_dev",
    "\"cjpeg\"",
    "\"djpeg\"",
    "\"jpegtran\"",
];

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

/// Which workflow lines a version scan counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteFilter {
    /// Every line, including comments and step titles. Loose on purpose: a
    /// stale version named in prose is drift worth catching.
    AnyMention,
    /// Only lines that install or build an oracle.
    Provisioning,
    /// Only lines that clone a source tree at a tag.
    SourceClone,
}

impl SiteFilter {
    fn accepts(self, line: &str) -> bool {
        match self {
            SiteFilter::AnyMention => true,
            SiteFilter::Provisioning => is_provisioning_line(line),
            SiteFilter::SourceClone => is_source_clone_line(line),
        }
    }
}

/// Every dotted-numeric token on the 3.x line, keyed by version, with the
/// `file:line` sites that carry it, restricted to the lines `filter` accepts.
fn version_sites_in_workflows(filter: SiteFilter) -> BTreeMap<String, Vec<String>> {
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
            if !filter.accepts(line) {
                continue;
            }
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

/// Does this line *install or build* an oracle, as opposed to merely naming a
/// version?
///
/// The distinction is the whole strength of the "a declared leg really runs"
/// direction. Counting every line that contains a version string would let a
/// comment satisfy it — and the comments in `upstream-currency.yml` alone name
/// both declared versions, so the 3.2.0 job could be deleted with the gate
/// still green. Only three shapes provision: a `VERSION=` assignment, a
/// `--branch` clone, and upstream's `libjpeg-turbo-official` package name.
/// A YAML comment and an `echo`'d instruction are documentation about a pin,
/// not a pin.
fn is_provisioning_line(line: &str) -> bool {
    let trimmed: &str = line.trim_start();
    if trimmed.starts_with('#') || trimmed.contains("echo ") {
        return false;
    }
    trimmed.contains("VERSION=")
        || trimmed.contains("--branch ")
        || trimmed.contains("libjpeg-turbo-official")
}

/// Does this line clone an upstream source tree at a tag?
///
/// The narrow half of [`is_provisioning_line`], for the roles a packaged
/// release cannot serve — see [`SiteShape::SourceClone`].
fn is_source_clone_line(line: &str) -> bool {
    is_provisioning_line(line) && line.contains("--branch ")
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
    // Loose on purpose: a stale version named in a comment or a step title is
    // exactly the kind of drift this direction exists to catch.
    let pinned: BTreeMap<String, Vec<String>> = version_sites_in_workflows(SiteFilter::AnyMention);

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

    // Every provisioned role is present, exactly once, before any row is
    // checked against a workflow. Rows are examined one at a time below, so a
    // *deleted* row is examined zero times and passes — and `trace-current`
    // shares its version with `tool-current`, so deleting it would leave 3.2.0
    // still declared and every direction of this gate still green while the
    // v8-ABI role it names had silently ceased to exist.
    let rows: Vec<Declared> = manifest_rows();
    for (role, _) in PROVISIONED_ROLES {
        let matching: usize = rows.iter().filter(|row| row.role == role).count();
        assert_eq!(
            matching, 1,
            "{MANIFEST} must declare exactly one {role:?} row; found {matching}. \
             Every role in this list backs a CI leg, so a missing row is a leg \
             nothing describes and a duplicate is two answers to \"which \
             release does this gate prove parity with?\""
        );
    }

    // Strict: only lines that install or build an oracle count. A comment
    // naming the version is not a leg.
    let provisioned: BTreeMap<String, Vec<String>> =
        version_sites_in_workflows(SiteFilter::Provisioning);
    let cloned: BTreeMap<String, Vec<String>> = version_sites_in_workflows(SiteFilter::SourceClone);

    let unused: Vec<String> = rows
        .into_iter()
        .filter_map(|row| {
            let shape: SiteShape = PROVISIONED_ROLES
                .iter()
                .find(|(role, _)| *role == row.role)
                .map(|(_, shape)| *shape)?;
            let sites: &BTreeMap<String, Vec<String>> = match shape {
                SiteShape::AnyProvisioning => &provisioned,
                SiteShape::SourceClone => &cloned,
            };
            if sites.contains_key(&row.version) {
                return None;
            }
            Some(format!(
                "  {} {} ({}) — needs {}",
                row.role,
                row.version,
                row.provisioned_as,
                match shape {
                    SiteShape::AnyProvisioning => "a step that installs or builds it",
                    SiteShape::SourceClone =>
                        "a `--branch <version>` source clone; upstream's deb is \
                         JPEG_LIB_VERSION 62 and cannot serve a v8-ABI oracle",
                }
            ))
        })
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
fn a_version_named_in_a_comment_is_not_a_provisioning_site() {
    // The classifier this rests on, pinned directly: the "declared leg really
    // runs" direction is only as strong as its ability to tell an install from
    // a mention, and every line below appears in these workflows today.
    assert!(is_provisioning_line("          VERSION=3.2.0"));
    assert!(is_provisioning_line(
        "          git clone --depth 1 --branch 3.1.4.1 https://github.com/libjpeg-turbo/libjpeg-turbo.git /tmp/ljt"
    ));
    assert!(is_provisioning_line(
        "            \"https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${VERSION}/libjpeg-turbo-official_${VERSION}_${ARCH}.deb\""
    ));

    assert!(!is_provisioning_line(
        "# 3.2.0 shipped 2026-06-30 and every oracle pin still said 3.1.4.1"
    ));
    assert!(!is_provisioning_line(
        "      - name: Build libjpeg-turbo 3.1.4.1 from source"
    ));
    assert!(!is_provisioning_line(
        "                echo \"  VERSION=3.1.4.1\""
    ));
}

#[test]
fn the_official_deb_does_not_back_a_v8_abi_row() {
    // `trace-current` asks for a source build because upstream's packaged
    // release is JPEG_LIB_VERSION 62. Both shapes below install the same
    // release, so without this distinction the deb would satisfy the row and
    // the v8 build could be deleted with the manifest still claiming it.
    assert!(is_source_clone_line(
        "          git clone --depth 1 --branch 3.2.0 \\"
    ));
    assert!(!is_source_clone_line("          VERSION=3.2.0"));
    assert!(!is_source_clone_line(
        "            \"https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${VERSION}/libjpeg-turbo-official_${VERSION}_${ARCH}.deb\""
    ));
    assert!(!is_source_clone_line(
        "# built from --branch 3.2.0 in the current-parity leg"
    ));
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

// ---------------------------------------------------------------------------
// P4-130 criterion 1, the C-ABI half: both tool legs run the same oracle
// suites.
//
// `cargo test --tests` selects the root crate, so the current-parity leg
// measured the root differential matrix and none of the classic-`jpeg_*` /
// TurboJPEG shim — the half of this repository whose entire contract is "what
// stock libjpeg does". The C-ABI crate's suites are selected by name (see the
// P4-61 notes in ci.yml), which is what let the two legs diverge silently: a
// step added to one leg is invisible to the other.
// ---------------------------------------------------------------------------

fn workflow_text(workflow: &str) -> String {
    let path: PathBuf = repo_root().join(workflow);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()))
}

/// The lines of one job in a workflow: from its `  <name>:` header to the next
/// header at the same indent.
fn job_block(text: &str, job: &str, workflow: &str) -> String {
    let header: String = format!("  {job}:");
    let mut block: String = String::new();
    let mut inside: bool = false;
    for line in text.lines() {
        if !inside {
            inside = line.trim_end() == header;
            continue;
        }
        // Job headers are the only content at exactly two spaces of indent.
        let starts_a_new_job: bool = line.starts_with("  ")
            && !line.starts_with("   ")
            && !line.trim_start().starts_with('#')
            && line.trim_end().ends_with(':');
        if starts_a_new_job {
            break;
        }
        block.push_str(line);
        block.push('\n');
    }
    assert!(
        inside,
        "no job named {job:?} in {workflow} — this gate pairs named jobs, so a \
         rename has to reach it rather than silently leaving it comparing \
         nothing"
    );
    block
}

/// YAML keys that end a step's `run:` script. A `run:` block is shell until
/// one of these appears; without the boundary, a following `- name:` would be
/// read as more shell, and a step *named* after a test — every step in
/// `full-c-parity.yml` is — would be indistinguishable from a libtest filter
/// selecting it.
const STEP_KEYS: [&str; 9] = [
    "- name:",
    "- run:",
    "- uses:",
    "run:",
    "env:",
    "with:",
    "if:",
    "id:",
    "timeout-minutes:",
];

/// The shell scripts a job block runs, one entry per `run:` step.
///
/// Comment lines are dropped first: these workflows discuss suites by name in
/// prose, and a mention is not a run — the same distinction
/// [`is_provisioning_line`] draws for version pins.
fn shell_scripts_in(job_block: &str) -> Vec<String> {
    let mut scripts: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for line in job_block.lines() {
        let trimmed: &str = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let starts_a_step: Option<&str> = ["- run:", "run:"]
            .iter()
            .find(|key| trimmed.starts_with(**key))
            .copied();
        if let Some(key) = starts_a_step {
            if let Some(script) = current.take() {
                scripts.push(script);
            }
            let rest: &str = trimmed[key.len()..].trim().trim_start_matches('|').trim();
            current = Some(rest.to_string());
            continue;
        }
        if STEP_KEYS.iter().any(|key| trimmed.starts_with(key)) {
            if let Some(script) = current.take() {
                scripts.push(script);
            }
            continue;
        }
        if let Some(script) = current.as_mut() {
            script.push(' ');
            script.push_str(trimmed);
        }
    }
    scripts.extend(current);
    scripts
}

/// Harness arguments that do **not** change which tests execute. Anything else
/// after `--` is either a filter or something this scanner does not model, and
/// the difference has to be visible: `--ignored` and `--include-ignored` change
/// the selected set as surely as a positional filter does, so silently reading
/// them as "no filter" would report the whole binary where a leg runs a
/// different part of it.
const COVERAGE_NEUTRAL_HARNESS_FLAGS: [&str; 8] = [
    "--nocapture",
    "--show-output",
    "--quiet",
    "-q",
    "--test-threads",
    "--color",
    "--format",
    "--report-time",
];

/// What one `cargo test` invocation selects out of a test binary.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Selection {
    /// The whole binary.
    All,
    /// Only the tests matching these libtest filters.
    Filters(BTreeSet<String>),
    /// Post-`--` syntax this scanner does not model — an unknown harness flag,
    /// a shell-quoted token. Never silently treated as "everything": that is
    /// the direction that turns an unread argument into a coverage claim.
    Unrecognised(String),
}

impl Selection {
    /// Two runs of the same binary in one leg select the union of what each
    /// selects, and "the whole binary" absorbs any filtered run.
    fn merge(self, other: Selection) -> Selection {
        match (self, other) {
            (Selection::Unrecognised(what), _) | (_, Selection::Unrecognised(what)) => {
                Selection::Unrecognised(what)
            }
            (Selection::All, _) | (_, Selection::All) => Selection::All,
            (Selection::Filters(mut a), Selection::Filters(b)) => {
                a.extend(b);
                Selection::Filters(a)
            }
        }
    }

    /// Does this selection cover everything `baseline` selects?
    ///
    /// Naming the same binary is not enough. A filter that selects one test —
    /// or one that has been mistyped, and therefore selects none — leaves the
    /// two legs measuring different things while a name-only comparison stays
    /// green; that vacuous shape is P4-61's whole finding, and it happened in
    /// this repository twice. Anything unmodelled fails closed on either side,
    /// because a gate that cannot read an argument cannot vouch for it.
    fn covers(&self, baseline: &Selection) -> bool {
        match (baseline, self) {
            (Selection::Unrecognised(_), _) | (_, Selection::Unrecognised(_)) => false,
            (_, Selection::All) => true,
            (Selection::All, Selection::Filters(_)) => false,
            (Selection::Filters(wanted), Selection::Filters(have)) => wanted.is_subset(have),
        }
    }
}

/// The libtest selection an invocation's post-`--` arguments describe.
fn selection_after_the_double_dash<'a>(args: impl Iterator<Item = &'a str>) -> Selection {
    let mut filters: BTreeSet<String> = BTreeSet::new();
    for arg in args {
        if COVERAGE_NEUTRAL_HARNESS_FLAGS.contains(&arg) {
            continue;
        }
        if arg.starts_with('-') {
            return Selection::Unrecognised(arg.to_string());
        }
        // Shell plumbing — `2>&1`, `|`, `&&`, a redirect target — is the shell
        // taking the line back, so the invocation ends here rather than being
        // unreadable.
        let is_shell_plumbing: bool = arg.contains(['|', '>', '<', ';', '&', '$', '`']);
        if is_shell_plumbing {
            break;
        }
        let is_test_name: bool = arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
            && arg.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
        if !is_test_name {
            // A quoted filter, a glob, a shell variable: readable to the
            // shell, not to this scanner.
            return Selection::Unrecognised(arg.to_string());
        }
        filters.insert(arg.to_string());
    }
    if filters.is_empty() {
        Selection::All
    } else {
        Selection::Filters(filters)
    }
}

/// The test binaries a job selects with `--test`, each mapped to what the
/// invocations select out of it, restricted to the `cargo test` invocations
/// `accepts_invocation` recognises as belonging to one crate.
fn suites_selected_by(
    job_block: &str,
    accepts_invocation: fn(&str) -> bool,
) -> BTreeMap<String, Selection> {
    let mut suites: BTreeMap<String, Selection> = BTreeMap::new();
    for script in shell_scripts_in(job_block) {
        // Each chunk holds one invocation's arguments; a following
        // invocation's `--test` flags land in the next chunk, not this one.
        for invocation in script.split("cargo test").skip(1) {
            if !accepts_invocation(invocation) {
                continue;
            }
            let mut selected: Vec<String> = Vec::new();
            let mut selection: Selection = Selection::All;
            let mut tokens = invocation.split_whitespace();
            while let Some(token) = tokens.next() {
                if token == "--test" {
                    if let Some(suite) = tokens.next() {
                        selected.push(suite.to_string());
                    }
                    continue;
                }
                if token == "--" {
                    selection = selection_after_the_double_dash(tokens.by_ref());
                    break;
                }
            }
            for suite in selected {
                let merged: Selection = match suites.remove(&suite) {
                    Some(existing) => existing.merge(selection.clone()),
                    None => selection.clone(),
                };
                suites.insert(suite, merged);
            }
        }
    }
    suites
}

/// The C-ABI test binaries a job selects.
///
/// Only invocations that name the C-ABI package count, so the root-crate
/// suites a job also runs (the serial timing step, for one) are not mistaken
/// for capi coverage.
fn capi_suites_selected_by(job_block: &str) -> BTreeMap<String, Selection> {
    suites_selected_by(job_block, |invocation| invocation.contains(CAPI_PACKAGE))
}

/// The root-crate test binaries a job selects.
///
/// An invocation that names no package runs the workspace default member,
/// which is the root crate; one that names any package is somebody else's
/// coverage and must not be counted here, or a capi suite would be looked up
/// in the root `tests/` directory and the lookup would panic.
fn root_suites_selected_by(job_block: &str) -> BTreeMap<String, Selection> {
    suites_selected_by(job_block, |invocation| {
        !invocation.contains(PACKAGE_SELECTOR)
    })
}

/// Does this suite compare against a C libjpeg-turbo, and therefore answer
/// differently depending on which release it ran against?
///
/// `test_dir` is where the suite's source lives — the root crate's `tests/`
/// or the C-ABI crate's. The classification rule is the same for both: read
/// the suite's own source, never a list kept here.
fn suite_consumes_a_c_oracle_in(test_dir: &str, suite: &str, workflow: &str) -> bool {
    let path: PathBuf = repo_root().join(test_dir).join(format!("{suite}.rs"));
    let source: String = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is named by {workflow} but could not be read: {e}",
            path.display()
        )
    });
    ORACLE_MARKERS.iter().any(|marker| source.contains(marker))
}

#[test]
fn a_job_block_stops_at_the_next_job() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    // `test-integration` is a prefix of `test-integration-current-oracle`. If
    // the block for the first ran on into the second, the pairing gate would
    // compare a superset with its own subset and could never fail.
    let baseline: String = job_block(&workflow_text(CI_WORKFLOW), BASELINE_LEG_JOB, CI_WORKFLOW);
    assert!(
        !baseline.contains(CURRENT_LEG_JOB),
        "the {BASELINE_LEG_JOB} block swallowed {CURRENT_LEG_JOB}"
    );

    // Every full-parity leg name is a prefix of its own current-oracle twin,
    // so the same trap is waiting once per architecture.
    let full_parity: String = workflow_text(FULL_PARITY_WORKFLOW);
    for (baseline_job, current_job) in FULL_PARITY_LEG_PAIRS {
        let block: String = job_block(&full_parity, baseline_job, FULL_PARITY_WORKFLOW);
        assert!(
            !block.contains(current_job),
            "the {baseline_job} block swallowed {current_job}"
        );
    }
}

#[test]
fn the_oracle_classifier_separates_c_comparisons_from_self_contained_suites() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    // Both directions pinned from suites that exist today, because the pairing
    // gate is only as strong as this classifier: an over-broad marker makes it
    // demand duplicate runs that measure nothing twice, and a missing one lets
    // a real C comparison stay on the baseline leg alone.
    for suite in [
        "capi_classic_lifecycle_state", // compiles a classic oracle
        "capi_classic_dest_ownership",  // LIBJPEG_TURBO_REFERENCE_DIR
        "capi_jpeglib_encode",          // shells out to stock cjpeg/djpeg
        "norealloc_all_entry_points",   // compares against real TurboJPEG
    ] {
        assert!(
            suite_consumes_a_c_oracle_in(CAPI_TEST_DIR, suite, CI_WORKFLOW),
            "{suite} compares against a C libjpeg-turbo but was classified as \
             self-contained, so the current-parity leg would never be asked to \
             run it"
        );
    }
    for suite in [
        "capi_span_overflow_guards", // arithmetic on our own spans
        "capi_output_message",       // our error manager's rendering
        "capi_max_memory_budget",    // our budget accounting
        "capi_symbol_versions",      // our generated version script and ELF
    ] {
        assert!(
            !suite_consumes_a_c_oracle_in(CAPI_TEST_DIR, suite, CI_WORKFLOW),
            "{suite} was classified as an oracle comparison; running it a \
             second time at another upstream release measures the same thing \
             twice and costs a leg's wall clock"
        );
    }
}

#[test]
fn every_oracle_backed_capi_suite_on_the_baseline_leg_also_runs_on_the_current_leg() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let ci: String = workflow_text(CI_WORKFLOW);
    let baseline: BTreeMap<String, Selection> =
        capi_suites_selected_by(&job_block(&ci, BASELINE_LEG_JOB, CI_WORKFLOW));
    let current: BTreeMap<String, Selection> =
        capi_suites_selected_by(&job_block(&ci, CURRENT_LEG_JOB, CI_WORKFLOW));

    assert!(
        !baseline.is_empty(),
        "no `cargo test {CAPI_PACKAGE} --test ...` invocation found in the \
         {BASELINE_LEG_JOB} job — the scanner has stopped matching, so this \
         gate would pass no matter which leg runs what"
    );

    let missing: Vec<String> = baseline
        .iter()
        .filter(|(suite, _)| suite_consumes_a_c_oracle_in(CAPI_TEST_DIR, suite, CI_WORKFLOW))
        .filter_map(|(suite, wanted)| match current.get(suite) {
            None => Some(format!("  {suite} — not run at all")),
            Some(run) if !run.covers(wanted) => {
                Some(format!("  {suite} — runs {run:?}, wanted {wanted:?}"))
            }
            Some(_) => None,
        })
        .collect();

    assert!(
        missing.is_empty(),
        "the {BASELINE_LEG_JOB} leg compares these C-ABI suites against a C \
         libjpeg-turbo, but the {CURRENT_LEG_JOB} leg does not run them — or \
         runs them under a narrower libtest filter:\n{}\n\n\
         Each one asserts what stock libjpeg does, so its answer is only as \
         current as the release it ran against. Add it to the current-parity \
         leg with the oracle prefix it needs (docs/oracle_versions.tsv names \
         which install serves which role), or drop it from the baseline leg.",
        missing.join("\n")
    );
}

// ---------------------------------------------------------------------------
// P4-130 criterion 1, the exhaustive-matrix half: the weekly Full C Parity
// legs run on both oracles too.
//
// `ci.yml`'s pair covers what a pull request runs. The largest differential
// surface in this repository is not there: the `full-c-parity` feature gates
// 12,230 transform cases, a 10,880-cell crop grid and the tj comp/decomp
// matrices behind a weekly workflow, and every one of those cases is a
// byte-comparison against a C tool. A matrix that big proving parity with a
// superseded release is the same gap this item was filed for, one workflow
// over.
// ---------------------------------------------------------------------------

/// The manifest version for one role, for gates that compare a workflow
/// against what the manifest says that leg is supposed to be.
fn declared_version(rows: &[Declared], role: &str) -> String {
    rows.iter()
        .find(|row| row.role == role)
        .map(|row| row.version.clone())
        .unwrap_or_else(|| panic!("{MANIFEST} declares no {role:?} row"))
}

/// The oracle versions a job block actually installs or builds, as opposed to
/// mentions. Empty means the job takes whatever the host happens to offer.
fn provisioned_versions_in(job_block: &str) -> BTreeSet<String> {
    let mut versions: BTreeSet<String> = BTreeSet::new();
    for line in job_block.lines().filter(|line| is_provisioning_line(line)) {
        for token in dotted_numeric_tokens(line) {
            if token.split('.').next() == Some(ORACLE_MAJOR) {
                versions.insert(token);
            }
        }
    }
    versions
}

#[test]
fn each_full_c_parity_leg_provisions_the_release_its_role_names() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let rows: Vec<Declared> = manifest_rows();
    let baseline_version: String = declared_version(&rows, "tool-baseline");
    let current_version: String = declared_version(&rows, "tool-current");
    let text: String = workflow_text(FULL_PARITY_WORKFLOW);

    for (baseline_job, current_job) in FULL_PARITY_LEG_PAIRS {
        for (job, role, expected) in [
            (baseline_job, "tool-baseline", &baseline_version),
            (current_job, "tool-current", &current_version),
        ] {
            let provisioned: BTreeSet<String> =
                provisioned_versions_in(&job_block(&text, job, FULL_PARITY_WORKFLOW));
            assert!(
                provisioned.contains(expected),
                "{FULL_PARITY_WORKFLOW}'s {job} leg plays the {role} role, so \
                 it must install libjpeg-turbo {expected}; it provisions {}.\n\n\
                 A leg that installs no version at all — `brew install \
                 jpeg-turbo`, an apt package, whatever is already on the \
                 runner — answers against a release nothing in this repository \
                 names, and the answer changes under you when the packager \
                 moves. That is the drift {MANIFEST} exists to make \
                 unrepresentable.",
                if provisioned.is_empty() {
                    "nothing".to_string()
                } else {
                    provisioned.into_iter().collect::<Vec<String>>().join(", ")
                }
            );
        }
    }
}

/// Indentation of a job's own keys inside a workflow: jobs sit at two spaces,
/// their keys at four, and a job-level `env:` mapping's entries at six.
const JOB_KEY_INDENT: usize = 4;

/// The path a job assigns to `LIBJPEG_TURBO_PREFIX` **at job level**, which is
/// what every step in that job inherits and therefore what actually selects the
/// oracle for the `cargo test` steps.
///
/// Scope is the point, not the spelling. A comment mentioning the variable
/// selects nothing, and an assignment inside one step's `env:` selects the
/// oracle for that step alone — so a leg could verify a prefix in the step that
/// names it and then run its matrices against whatever lookup order finds. Both
/// shapes read identically once indentation is trimmed, which is why this
/// reads indentation instead.
fn oracle_prefix_assigned_by(job_block: &str) -> Option<String> {
    let mut assigned: Option<String> = None;
    let mut inside_job_env: bool = false;
    for line in job_block.lines() {
        let trimmed: &str = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent: usize = line.len() - line.trim_start().len();
        if indent <= JOB_KEY_INDENT {
            inside_job_env = indent == JOB_KEY_INDENT && trimmed == "env:";
            continue;
        }
        if !inside_job_env {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("LIBJPEG_TURBO_PREFIX:") else {
            continue;
        };
        let value: String = rest.trim().trim_matches(['"', '\'']).to_string();
        assert!(
            assigned.is_none(),
            "two job-level LIBJPEG_TURBO_PREFIX assignments in one job — which \
             one selects the oracle depends on YAML scoping, so this gate \
             cannot say what the leg measures"
        );
        assigned = Some(value);
    }
    assigned
}

/// Does this line *check* that a command's output carries `needle`, as opposed
/// to printing `needle` itself?
///
/// `echo "version 3.2.0"` satisfies any substring search and asserts nothing —
/// the same distinction [`is_provisioning_line`] draws between installing a
/// release and mentioning one.
fn is_assertion_over_output(line: &str, needle: &str) -> bool {
    let trimmed: &str = line.trim_start();
    !trimmed.starts_with('#')
        && !trimmed.contains("echo ")
        && trimmed.contains("grep")
        && trimmed.contains(needle)
}

#[test]
fn every_full_c_parity_leg_verifies_the_prefix_it_measures() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    // Installing the right release is not the same as measuring it.
    // `helpers::c_tool_path` reads `/opt/homebrew/bin` before PATH and falls
    // through to `which`, so a leg can install 3.2.0, verify 3.2.0, and still
    // compare against whatever else is on the runner — the false green #569
    // found inside its own change. So the gate is not "the variable appears
    // somewhere": the assigned path has to be the path whose `djpeg` the leg
    // checks, against the version its role declares.
    let rows: Vec<Declared> = manifest_rows();
    let text: String = workflow_text(FULL_PARITY_WORKFLOW);
    for (baseline_job, current_job) in FULL_PARITY_LEG_PAIRS {
        for (job, role) in [
            (baseline_job, "tool-baseline"),
            (current_job, "tool-current"),
        ] {
            let block: String = job_block(&text, job, FULL_PARITY_WORKFLOW);
            let prefix: String = oracle_prefix_assigned_by(&block).unwrap_or_else(|| {
                panic!(
                    "{FULL_PARITY_WORKFLOW}'s {job} leg assigns no job-level \
                     LIBJPEG_TURBO_PREFIX, so which install its `cargo test` \
                     steps measure is decided by lookup order rather than by \
                     the job. A step-level assignment is not enough: it selects \
                     the oracle for that step alone."
                )
            });
            let version: String = declared_version(&rows, role);
            let checks_that_prefix: bool = block
                .lines()
                .filter(|line| !line.trim_start().starts_with('#'))
                .any(|line| {
                    line.contains(&format!("{prefix}/bin/djpeg")) && line.contains("-version")
                });
            assert!(
                checks_that_prefix,
                "{job} measures {prefix} but never runs {prefix}/bin/djpeg \
                 -version, so nothing ties the install it verified to the \
                 install the tests resolve"
            );
            let checks_the_version: bool = block
                .lines()
                .any(|line| is_assertion_over_output(line, &format!("version {version}")));
            assert!(
                checks_the_version,
                "{job} plays the {role} role but never *asserts* that its \
                 oracle reports version {version} — printing the string is not \
                 checking it"
            );
        }
    }
}

#[test]
fn the_suite_selector_tells_the_root_crate_from_the_c_abi_crate() {
    // The two pairing gates read different crates out of the same workflows,
    // and a selector that confused them would look up a capi suite under
    // `tests/` and panic, or silently drop a root suite and pass vacuously.
    let block: &str = "      - run: cargo test --features full-c-parity --test c_croptest\n\
                       \x20     - run: cargo test -p libjpeg-turbo-rs-capi --test capi_jpeglib_encode\n";
    assert_eq!(
        root_suites_selected_by(block)
            .into_keys()
            .collect::<Vec<String>>(),
        vec!["c_croptest".to_string()]
    );
    assert_eq!(
        capi_suites_selected_by(block)
            .into_keys()
            .collect::<Vec<String>>(),
        vec!["capi_jpeglib_encode".to_string()]
    );
}

#[test]
fn a_step_named_after_a_test_is_not_a_libtest_filter() {
    // Every step in full-c-parity.yml is *named* after the matrix it runs, and
    // the scanner joins a step's shell lines together. Without the step-key
    // boundary the next step's name reads as another filter on the previous
    // invocation, and two legs whose filters actually differ would compare
    // equal.
    let block: &str = "      - name: c_tjdecomptest_full\n\
                       \x20       run: cargo test --features full-c-parity --test c_tjdecomptest -- c_tjdecomptest_full\n\
                       \x20       timeout-minutes: 10\n\
                       \x20     - name: c_tjcomptest_full\n\
                       \x20       run: cargo test --features full-c-parity --test c_tjcomptest\n";
    let selected: BTreeMap<String, Selection> = root_suites_selected_by(block);
    assert_eq!(
        selected.get("c_tjdecomptest"),
        Some(&Selection::Filters(BTreeSet::from([
            "c_tjdecomptest_full".to_string()
        ]))),
        "the filter this step really passes, and nothing the next step is named"
    );
    assert_eq!(
        selected.get("c_tjcomptest"),
        Some(&Selection::All),
        "an unfiltered invocation runs the whole binary"
    );
}

#[test]
fn a_harness_flag_is_not_a_filter_and_a_redirect_ends_the_invocation() {
    // `-- --nocapture` changes nothing about which tests run, so requiring the
    // two legs to match on it would demand agreement about output formatting.
    // `2>&1 | tee …` is the shell taking the line back, and every token after
    // it belongs to a pipeline, not to libtest — the P4-108 step is written
    // exactly this way.
    let block: &str = "      - run: cargo test -p libjpeg-turbo-rs-capi --test capi_classic_dest_ownership -- --nocapture 2>&1 | tee p4108.log\n";
    assert_eq!(
        capi_suites_selected_by(block).get("capi_classic_dest_ownership"),
        Some(&Selection::All)
    );
}

#[test]
fn an_argument_this_scanner_cannot_read_fails_the_comparison_closed() {
    // Both shapes below *do* change which tests run, and both used to leave an
    // empty filter set behind — which the comparison then read as "the whole
    // binary", the most generous answer available. A gate that cannot read an
    // argument must not vouch for it.
    let ignored: &str = "      - run: cargo test --test c_croptest -- --ignored\n";
    assert!(matches!(
        root_suites_selected_by(ignored).get("c_croptest"),
        Some(Selection::Unrecognised(_))
    ));

    let quoted: &str = "      - run: cargo test --test c_croptest -- 'c_croptest_full'\n";
    assert!(matches!(
        root_suites_selected_by(quoted).get("c_croptest"),
        Some(Selection::Unrecognised(_))
    ));

    let unreadable: Selection = Selection::Unrecognised("--ignored".to_string());
    assert!(!unreadable.covers(&Selection::All));
    assert!(!Selection::All.covers(&unreadable));
}

#[test]
fn a_narrower_selection_on_the_current_leg_does_not_cover_the_baseline() {
    // The pairing gates' load-bearing comparison. Naming the same binary while
    // selecting fewer of its tests — or, after a typo, none of them — is the
    // vacuous-green shape P4-61 documents, and a name-only comparison cannot
    // see it.
    let full: Selection = Selection::Filters(BTreeSet::from(["c_tjdecomptest_full".to_string()]));
    let typo: Selection = Selection::Filters(BTreeSet::from(["c_tjdecomptest_fll".to_string()]));

    assert!(full.covers(&full));
    assert!(Selection::All.covers(&full));
    assert!(!typo.covers(&full));
    assert!(!full.covers(&Selection::All));
}

#[test]
fn running_a_suite_twice_in_one_leg_selects_the_union() {
    // A leg that runs one binary unfiltered *and* filtered runs all of it, so
    // merging the two must not shrink to the filter — that would let a current
    // leg running only the filtered form compare equal to a baseline running
    // both.
    let block: &str = "      - run: cargo test --test c_croptest -- c_croptest_full\n\
                       \x20     - run: cargo test --test c_croptest\n";
    assert_eq!(
        root_suites_selected_by(block).get("c_croptest"),
        Some(&Selection::All)
    );
}

#[test]
fn only_a_job_level_prefix_assignment_selects_the_oracle_for_a_whole_leg() {
    // A comment mentioning the variable selects nothing, and a step-level
    // assignment selects the oracle for that step alone — so a leg could verify
    // one install in the step that names it and run its matrices against
    // another. Both read identically once indentation is trimmed.
    assert_eq!(
        oracle_prefix_assigned_by("    env:\n      # LIBJPEG_TURBO_PREFIX: /opt/libjpeg-turbo\n"),
        None
    );
    assert_eq!(
        oracle_prefix_assigned_by(
            "    steps:\n      - name: Verify\n        run: djpeg -version\n        env:\n          LIBJPEG_TURBO_PREFIX: /tmp/ljt320/prefix\n"
        ),
        None,
        "a step-level assignment does not reach the leg's other steps"
    );
    assert_eq!(
        oracle_prefix_assigned_by("    env:\n      LIBJPEG_TURBO_PREFIX: /tmp/ljt320/prefix\n"),
        Some("/tmp/ljt320/prefix".to_string())
    );
}

#[test]
fn an_echoed_version_string_is_not_a_version_check() {
    // The version assertion is the only thing standing between "the leg
    // installed something" and "the leg installed what it claims", so it has to
    // be an assertion over djpeg's output rather than any line carrying the
    // number.
    assert!(is_assertion_over_output(
        "          grep -q \"version 3.2.0\" /tmp/oracle-version.txt",
        "version 3.2.0"
    ));
    assert!(!is_assertion_over_output(
        "          echo \"version 3.2.0\"",
        "version 3.2.0"
    ));
    assert!(!is_assertion_over_output(
        "          # asserts version 3.2.0 below",
        "version 3.2.0"
    ));
}

#[test]
fn the_oracle_classifier_reads_root_crate_suites_too() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    // Same both-directions pin as the capi classifier, against root-crate
    // suites: every exhaustive matrix shells out to a stock tool, and a suite
    // that computes buffer sizes does not.
    for suite in [
        "c_tjdecomptest",
        "c_tjcomptest",
        "c_tjtrantest",
        "c_croptest",
    ] {
        assert!(
            suite_consumes_a_c_oracle_in(ROOT_TEST_DIR, suite, FULL_PARITY_WORKFLOW),
            "{suite} compares its output against a stock C tool but was \
             classified as self-contained, so the current-parity leg would \
             never be asked to run it"
        );
    }
    for suite in ["bufsize", "common_types"] {
        assert!(
            !suite_consumes_a_c_oracle_in(ROOT_TEST_DIR, suite, FULL_PARITY_WORKFLOW),
            "{suite} was classified as an oracle comparison; running it a \
             second time at another upstream release measures the same thing \
             twice"
        );
    }
}

#[test]
fn every_oracle_backed_full_parity_suite_on_the_baseline_leg_also_runs_on_the_current_leg() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let text: String = workflow_text(FULL_PARITY_WORKFLOW);

    for (baseline_job, current_job) in FULL_PARITY_LEG_PAIRS {
        let baseline: BTreeMap<String, Selection> =
            root_suites_selected_by(&job_block(&text, baseline_job, FULL_PARITY_WORKFLOW));
        let current: BTreeMap<String, Selection> =
            root_suites_selected_by(&job_block(&text, current_job, FULL_PARITY_WORKFLOW));

        assert!(
            !baseline.is_empty(),
            "no `cargo test --test ...` invocation found in {baseline_job} — \
             the scanner has stopped matching, so this gate would pass no \
             matter which leg runs what"
        );

        let missing: Vec<String> = baseline
            .iter()
            .filter(|(suite, _)| {
                suite_consumes_a_c_oracle_in(ROOT_TEST_DIR, suite, FULL_PARITY_WORKFLOW)
            })
            .filter_map(|(suite, wanted)| match current.get(suite) {
                None => Some(format!("  {suite} — not run at all")),
                Some(run) if !run.covers(wanted) => Some(format!(
                    "  {suite} — runs {run:?}, which does not cover the \
                     baseline leg's {wanted:?}"
                )),
                Some(_) => None,
            })
            .collect();

        assert!(
            missing.is_empty(),
            "{baseline_job} compares these matrices against a C \
             libjpeg-turbo, but {current_job} does not:\n{}\n\n\
             These are the exhaustive matrices — the widest differential \
             surface in this repository — so a release they have never been \
             measured against is precisely the unmeasured delta P4-130 was \
             filed for.",
            missing.join("\n")
        );
    }
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
