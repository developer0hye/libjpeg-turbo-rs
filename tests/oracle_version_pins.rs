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
//! Two further dimensions live in the back half of this file, each under its
//! own banner:
//!
//! * **the two tool legs run the same oracle-backed suites** — declaring
//!   versions says nothing about which suites actually meet them, and suites
//!   are selected by name, so one added to one leg is invisible to the other.
//!   One pairing gate per crate: the C-ABI crate's suites, and the root
//!   crate's exhaustive `full-c-parity` matrices.
//! * **every oracle-provisioning job is pinned, checked and measured** — read
//!   from all the jobs in all the workflows rather than from a list of
//!   workflow files, which is what let one leg keep an unpinned
//!   `brew install jpeg-turbo` while the legs a gate happened to name lost it.
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
/// the difference has to be visible: `--ignored`, `--include-ignored` and
/// `--skip` change the selected set as surely as a positional filter does, so
/// silently reading them as "no filter" would report the whole binary where a
/// leg runs a different part of it.
const COVERAGE_NEUTRAL_FLAGS: [&str; 5] = [
    "--nocapture",
    "--show-output",
    "--quiet",
    "-q",
    "--report-time",
];

/// Coverage-neutral flags that take a value. The value is theirs, not a
/// filter — `--test-threads 1` selects every test, and reading the `1` as
/// unmodelled syntax would fail the pairing gate on two legs that run
/// identical sets.
const COVERAGE_NEUTRAL_FLAGS_WITH_VALUE: [&str; 4] =
    ["--test-threads", "--color", "--format", "--logfile"];

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
    let mut expecting_a_flag_value: bool = false;
    for arg in args {
        if expecting_a_flag_value {
            expecting_a_flag_value = false;
            continue;
        }
        if COVERAGE_NEUTRAL_FLAGS.contains(&arg) {
            continue;
        }
        // `--test-threads 1` and `--test-threads=1` are the same flag; the
        // value belongs to it either way and is not a filter.
        let flag_name: &str = arg.split('=').next().unwrap_or(arg);
        if COVERAGE_NEUTRAL_FLAGS_WITH_VALUE.contains(&flag_name) {
            expecting_a_flag_value = !arg.contains('=');
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

/// Every `LIBJPEG_TURBO_PREFIX` a job assigns, split by scope.
///
/// Scope is the point, not the spelling, and both directions matter. The
/// job-level assignment is what every step inherits, so it is what selects the
/// oracle for the `cargo test` steps. A *step-level* assignment overrides it
/// for that one step — so a leg could verify one install in the step that names
/// it and measure another in the steps that matter. Both read identically once
/// indentation is trimmed, which is why this reads indentation.
#[derive(Debug, Default, PartialEq, Eq)]
struct PrefixAssignments {
    job_level: Option<String>,
    step_level: Vec<String>,
}

fn oracle_prefixes_assigned_by(job_block: &str) -> PrefixAssignments {
    let mut found: PrefixAssignments = PrefixAssignments::default();
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
        let Some(rest) = trimmed.strip_prefix("LIBJPEG_TURBO_PREFIX:") else {
            continue;
        };
        let value: String = rest.trim().trim_matches(['"', '\'']).to_string();
        // A job-level `env:` mapping's entries sit one level in from the key.
        if inside_job_env && indent <= JOB_KEY_INDENT + 2 {
            assert!(
                found.job_level.is_none(),
                "two job-level LIBJPEG_TURBO_PREFIX assignments in one job — \
                 which one selects the oracle depends on YAML scoping, so this \
                 gate cannot say what the leg measures"
            );
            found.job_level = Some(value);
        } else {
            found.step_level.push(value);
        }
    }
    found
}

/// Does this line *assert* that a command's output carries `needle`?
///
/// Three ways to look like a check without being one, all of them accepted by
/// an earlier draft of this predicate: printing the string (`echo "version
/// 3.2.0"`), writing the check in a comment, and running one that cannot fail
/// (`grep … || true`, or a `true` with the grep commented out). A step's shell
/// runs under `set -e`, so an *uncommented* grep at the head of a line or at
/// the end of a pipeline is the thing that stops the job.
fn is_assertion_over_output(line: &str, needle: &str) -> bool {
    let trimmed: &str = line.trim_start();
    if trimmed.starts_with('#') || trimmed.contains("echo ") || !trimmed.contains(needle) {
        return false;
    }
    // `|| true`, `|| :`, `|| echo …`: the exit status is swallowed, so a
    // mismatch does not fail the step.
    if trimmed.contains("||") {
        return false;
    }
    // The grep has to be the command being run, not something quoted inside a
    // comment further along the line.
    let code: &str = trimmed.split('#').next().unwrap_or(trimmed);
    code.trim_start().starts_with("grep ") || code.contains("| grep ")
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
            let assignments: PrefixAssignments = oracle_prefixes_assigned_by(&block);
            let prefix: String = assignments.job_level.clone().unwrap_or_else(|| {
                panic!(
                    "{FULL_PARITY_WORKFLOW}'s {job} leg assigns no job-level \
                     LIBJPEG_TURBO_PREFIX, so which install its `cargo test` \
                     steps measure is decided by lookup order rather than by \
                     the job. A step-level assignment is not enough: it selects \
                     the oracle for that step alone."
                )
            });
            let overrides: Vec<&String> = assignments
                .step_level
                .iter()
                .filter(|value| **value != prefix)
                .collect();
            assert!(
                overrides.is_empty(),
                "{job} inherits {prefix} but one of its steps overrides \
                 LIBJPEG_TURBO_PREFIX with {overrides:?}. A step-level value \
                 wins for that step, so the leg would verify one install and \
                 measure another — the split this pair of legs exists to make \
                 impossible."
            );
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
fn a_value_taking_harness_flag_does_not_look_like_a_filter() {
    // `--test-threads 1` runs every test. Reading the `1` as unmodelled syntax
    // would fail the pairing gate on two legs that select identical sets — the
    // opposite error from the one the fail-closed rule exists for, and just as
    // useless. `--skip` is not in that list on purpose: it *does* change the
    // set.
    let neutral: &str =
        "      - run: cargo test --test c_croptest -- --test-threads 1 --color=always\n";
    assert_eq!(
        root_suites_selected_by(neutral).get("c_croptest"),
        Some(&Selection::All)
    );

    let skipping: &str = "      - run: cargo test --test c_croptest -- --skip c_croptest_full\n";
    assert!(matches!(
        root_suites_selected_by(skipping).get("c_croptest"),
        Some(Selection::Unrecognised(_))
    ));
}

#[test]
fn only_a_job_level_prefix_assignment_selects_the_oracle_for_a_whole_leg() {
    // A comment mentioning the variable selects nothing; a step-level
    // assignment selects the oracle for that step alone and overrides the
    // inherited one — so both have to be visible to the gate rather than
    // skipped. Both read identically once indentation is trimmed.
    assert_eq!(
        oracle_prefixes_assigned_by("    env:\n      # LIBJPEG_TURBO_PREFIX: /opt/libjpeg-turbo\n"),
        PrefixAssignments::default()
    );
    assert_eq!(
        oracle_prefixes_assigned_by(
            "    steps:\n      - name: Verify\n        run: djpeg -version\n        env:\n          LIBJPEG_TURBO_PREFIX: /opt/homebrew\n"
        ),
        PrefixAssignments {
            job_level: None,
            step_level: vec!["/opt/homebrew".to_string()],
        },
        "a step-level assignment is reported as one, not silently ignored"
    );
    assert_eq!(
        oracle_prefixes_assigned_by(
            "    env:\n      LIBJPEG_TURBO_PREFIX: /tmp/ljt320/prefix\n    steps:\n      - run: cargo test\n        env:\n          LIBJPEG_TURBO_PREFIX: /opt/homebrew\n"
        ),
        PrefixAssignments {
            job_level: Some("/tmp/ljt320/prefix".to_string()),
            step_level: vec!["/opt/homebrew".to_string()],
        },
        "an override wins for its own step, so the gate has to see both"
    );
}

#[test]
fn a_version_check_that_cannot_fail_is_not_a_version_check() {
    // The version assertion is the only thing standing between "the leg
    // installed something" and "the leg installed what it claims". Three ways
    // to look like one without being one: print the string, write it in a
    // comment, or run a check whose exit status is swallowed.
    assert!(is_assertion_over_output(
        "          grep -q \"version 3.2.0\" /tmp/oracle-version.txt",
        "version 3.2.0"
    ));
    assert!(is_assertion_over_output(
        "          /opt/libjpeg-turbo/bin/cjpeg -version 2>&1 | grep -q \"version 3.2.0\"",
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
    assert!(!is_assertion_over_output(
        "          true # grep -q \"version 3.2.0\" /tmp/oracle-version.txt",
        "version 3.2.0"
    ));
    assert!(!is_assertion_over_output(
        "          grep -q \"version 3.2.0\" /tmp/oracle-version.txt || true",
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

// ---------------------------------------------------------------------------
// P4-130 criterion 1, generalised: the pin-and-name rule holds for every job
// that installs an oracle, in every workflow.
//
// Every gate above names the workflow it reads — `ci.yml`'s two legs,
// `full-c-parity.yml`'s four. That is how `test-cross-encode` still ran
// `brew install jpeg-turbo` the day after the aarch64 full-parity legs lost
// exactly that shape: homebrew has no per-version formula for jpeg-turbo, so
// the release that leg measures is whatever the packager shipped that week,
// named nowhere in this repository and free to move without a commit to
// review. Eight legs across three workflows were in the same position, and a
// gate keyed to a list of workflow files cannot see one of them.
//
// So this half is keyed to *jobs*, enumerated by reading every workflow. Three
// requirements, each the general form of one the full-parity legs already
// meet:
//
// * **pinned** — an install names the release it installs. A package manager's
//   own name for the package does not name a release, so that shape is
//   rejected rather than discouraged. A shape this scanner cannot read fails
//   closed for the same reason.
// * **checked** — the job *asserts* that release. Running `djpeg -version` and
//   printing the output is what five of these legs did, and it fails nothing.
// * **measured** — the tests resolve to the install the job checked. On a
//   macOS runner that requires `LIBJPEG_TURBO_PREFIX`: `helpers::c_tool_path`
//   reads `/opt/homebrew/bin` before PATH, so a PATH entry does not select an
//   oracle there. That is the false green #569 found inside its own change.
// ---------------------------------------------------------------------------

/// Package managers whose install names a *package*, leaving the release to
/// the packager.
const PACKAGE_MANAGER_INSTALLS: [&str; 6] = [
    "brew install",
    "apt-get install",
    "apt install",
    "dnf install",
    "yum install",
    "port install",
];

/// Commands that fetch a tree or an archive. A fetch that names upstream but
/// matches none of the pinned shapes is unreadable, not absent.
const FETCH_COMMANDS: [&str; 3] = ["curl ", "wget ", "git clone"];

/// This repository's own crate names, which contain upstream's as a prefix.
/// Stripped before asking whether a line names upstream, so `cargo publish -p
/// libjpeg-turbo-rs-capi` is not read as provisioning a C oracle.
const OUR_CRATE_NAME: &str = "libjpeg-turbo-rs";

/// How a line provisions a C libjpeg-turbo, if it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OracleInstall {
    /// Pinned by a release the line names: upstream's
    /// `libjpeg-turbo-official_<version>` package, or a `--branch <tag>` clone.
    /// Which release is read at job scope by [`provisioned_versions_in`],
    /// because these workflows put the version in a `VERSION=` assignment one
    /// line above the URL that interpolates it.
    Pinned,
    /// Built from `references/libjpeg-turbo`, a git submodule — pinned by
    /// commit rather than by a version token, and cross-checked against the
    /// manifest by [`the_submodule_row_matches_the_checked_out_submodule`].
    Submodule,
    /// Installed under a name whose release the packager chooses.
    Unpinned,
    /// Names upstream and fetches something, in a shape this scanner cannot
    /// resolve to a release.
    Unreadable,
}

/// Shell lines with backslash continuations joined, so a command split across
/// lines is classified as the one command it is.
///
/// `ci.yml` splits both provisioning shapes: `curl -fL -o /tmp/ljt.deb \` puts
/// the URL carrying `libjpeg-turbo-official` on the next line, and
/// `git clone --depth 1 --branch 3.2.0 \` puts the repository on the next.
/// Read line by line, the half naming upstream and the half naming the release
/// are different lines and neither is an install.
fn logical_lines(block: &str) -> Vec<String> {
    let mut joined: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;
    for line in block.lines() {
        let trimmed: &str = line.trim();
        let continues: bool = trimmed.ends_with('\\');
        let body: &str = trimmed.strip_suffix('\\').unwrap_or(trimmed).trim_end();
        match pending.as_mut() {
            Some(current) => {
                current.push(' ');
                current.push_str(body);
            }
            None => pending = Some(body.to_string()),
        }
        if !continues {
            joined.extend(pending.take());
        }
    }
    joined.extend(pending);
    joined
}

fn oracle_install_on(line: &str) -> Option<OracleInstall> {
    let trimmed: &str = line.trim_start();
    // Documentation about an install is not one — the distinction
    // [`is_provisioning_line`] already draws, and `fuzz-smoke.yml` needs it:
    // the reproduction instructions it prints on failure name the deb, the
    // release and the PATH entry in full.
    if trimmed.starts_with('#') || trimmed.contains("echo ") {
        return None;
    }
    if trimmed.contains("cmake -S references/libjpeg-turbo") {
        return Some(OracleInstall::Submodule);
    }
    let without_our_crate: String = trimmed.replace(OUR_CRATE_NAME, "");
    if !without_our_crate.contains("jpeg-turbo") && !without_our_crate.contains("libjpeg") {
        return None;
    }
    // Upstream's own release assets: the official package carries the release
    // in its filename, a `--branch` clone in its tag.
    if without_our_crate.contains("libjpeg-turbo-official")
        || without_our_crate.contains("--branch ")
    {
        return Some(OracleInstall::Pinned);
    }
    if PACKAGE_MANAGER_INSTALLS
        .iter()
        .any(|command| without_our_crate.contains(command))
    {
        return Some(OracleInstall::Unpinned);
    }
    if FETCH_COMMANDS
        .iter()
        .any(|command| without_our_crate.contains(command))
    {
        return Some(OracleInstall::Unreadable);
    }
    None
}

/// Every job in a workflow, in file order.
///
/// Only the mapping under the top-level `jobs:` key: `on:` carries two-space
/// keys of its own (`push:`, `schedule:`, `workflow_dispatch:`), and reading
/// those as jobs would ask [`job_block`] for a block that is not one.
///
/// Checked once against a real YAML parser while this gate was written: the
/// scanner's list is identical to PyYAML's for all 41 jobs in the nine
/// workflows, name for name. The standing protection is the sibling test —
/// a workflow this returns nothing for is a workflow every gate here passes
/// vacuously.
fn job_names_in(text: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut inside_jobs: bool = false;
    for line in text.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if !line.starts_with(' ') {
            inside_jobs = line.trim_end() == "jobs:";
            continue;
        }
        if !inside_jobs {
            continue;
        }
        let is_job_header: bool =
            line.starts_with("  ") && !line.starts_with("   ") && line.trim_end().ends_with(':');
        if is_job_header {
            names.push(line.trim().trim_end_matches(':').to_string());
        }
    }
    names
}

/// `(workflow file name, job name, job block)` for every job in the repository.
fn every_job() -> Vec<(String, String, String)> {
    let mut jobs: Vec<(String, String, String)> = Vec::new();
    for path in workflow_files() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let workflow: String = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        for job in job_names_in(&text) {
            let block: String = job_block(&text, &job, &workflow);
            jobs.push((workflow.clone(), job, block));
        }
    }
    jobs
}

/// How this job installs a C oracle, over all of its lines.
fn oracle_installs_in(block: &str) -> Vec<(OracleInstall, String)> {
    logical_lines(block)
        .into_iter()
        .filter_map(|line| oracle_install_on(&line).map(|kind| (kind, line)))
        .collect()
}

/// Does this line assert that a release really is the one installed?
///
/// Two spellings, because the two things a job can check are the built tool
/// and the tree it was built from: `djpeg -version` prints `version 3.2.0`,
/// while upstream's `CMakeLists.txt` states `set(VERSION 3.2.0)` and the grep
/// that reads it escapes the dots. Both are assertions in the sense
/// [`is_assertion_over_output`] means — a grep whose failure stops the step.
fn asserts_release(line: &str, version: &str) -> bool {
    let escaped: String = version.replace('.', "\\.");
    is_assertion_over_output(line, &format!("version {version}"))
        || is_assertion_over_output(line, &format!("VERSION {escaped}"))
}

/// The absolute path a `<prefix>/bin/djpeg -version` line invokes, and the
/// file it tees the output into, if any.
///
/// An absolute path, because a bare `djpeg -version` is answered by PATH and
/// says nothing about the install the job made — the shape `test-integration`
/// carried while reading as a version check.
fn djpeg_version_invocation(line: &str) -> Option<(String, Option<String>)> {
    let trimmed: &str = line.trim_start();
    if trimmed.starts_with('#') || !trimmed.contains("-version") {
        return None;
    }
    let at: usize = trimmed.find("/bin/djpeg")?;
    let head: &str = &trimmed[..at];
    let start: usize = head
        .rfind([' ', '"', '\'', '\t', '('])
        .map(|index| index + 1)
        .unwrap_or(0);
    let prefix: &str = &head[start..];
    if !prefix.starts_with('/') {
        return None;
    }
    let teed: Option<String> = trimmed.split("tee ").nth(1).map(|rest| {
        rest.split_whitespace()
            .next()
            .unwrap_or_default()
            .trim_matches(['"', '\''])
            .to_string()
    });
    Some((prefix.to_string(), teed))
}

/// Which release each prefix in this job has been *checked* to be.
///
/// A prefix is checked at a release when the job both invokes that prefix's
/// `djpeg -version` and asserts the release over the output — either on the
/// one line, or through the file the invocation tees into. Loose pairing
/// ("somewhere in the job a version is asserted, somewhere a prefix is run")
/// is not enough once a job carries more than one oracle, and
/// `test-integration` carries three.
fn prefix_releases_checked_in(block: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut checked: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let lines: Vec<&str> = block.lines().collect();
    for line in &lines {
        let Some((prefix, teed)) = djpeg_version_invocation(line) else {
            continue;
        };
        for version in versions_asserted_over(line, teed.as_deref(), &lines) {
            checked.entry(prefix.clone()).or_default().insert(version);
        }
    }
    checked
}

/// The releases asserted over one `djpeg -version` invocation's output.
fn versions_asserted_over(line: &str, teed: Option<&str>, lines: &[&str]) -> BTreeSet<String> {
    let mut versions: BTreeSet<String> = BTreeSet::new();
    for candidate in dotted_numeric_tokens_in_job(lines) {
        let on_this_line: bool = asserts_release(line, &candidate);
        let through_the_file: bool = teed.is_some_and(|file| {
            lines
                .iter()
                .any(|other| other.contains(file) && asserts_release(other, &candidate))
        });
        if on_this_line || through_the_file {
            versions.insert(candidate);
        }
    }
    versions
}

/// Every 3.x token a job names anywhere — the candidate releases its checks
/// could be asserting.
fn dotted_numeric_tokens_in_job(lines: &[&str]) -> BTreeSet<String> {
    let mut tokens: BTreeSet<String> = BTreeSet::new();
    for line in lines {
        for token in dotted_numeric_tokens(line) {
            if token.split('.').next() == Some(ORACLE_MAJOR) {
                tokens.insert(token);
            }
        }
    }
    tokens
}

/// Environment variables that name an oracle prefix for the step they are set
/// on. `LIBJPEG_TURBO_REFERENCE_DIR` is P4-108's spelling of the same thing.
const ORACLE_PREFIX_VARS: [&str; 2] = ["LIBJPEG_TURBO_PREFIX", "LIBJPEG_TURBO_REFERENCE_DIR"];

/// One step of a job: what it runs, and the oracle prefixes it names.
#[derive(Debug, Default)]
struct Step {
    script: String,
    prefixes: BTreeSet<String>,
}

/// The steps of a job, each with its own `env:` scope.
///
/// Per step rather than per job, because a step-level assignment overrides the
/// job's for that step alone. A union over the job would report a leg green
/// whenever *any* of its steps named the checked install, while another
/// `cargo test` step resolved a different oracle — on macOS, homebrew's. That
/// is #569's false green one step over, and it is what a job-scope union
/// cannot see.
fn steps_in(job_block: &str) -> Vec<Step> {
    // From the `steps:` key, not from the first `- ` in the job: a matrix
    // entry (`- os: macos-latest`) comes first and sits deeper, and taking its
    // indent as the step indent merged every real step into one — which is a
    // job-scope union again, arriving through the parser, on the matrix jobs
    // where the macOS rule matters most.
    let after_steps_key: Option<&str> = job_block
        .find("\n    steps:")
        .map(|at| &job_block[at + "\n    steps:".len()..]);
    let Some(body) = after_steps_key else {
        return Vec::new();
    };
    let step_indent: Option<usize> = body
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && !line.trim_start().starts_with("- #"))
        .map(|line| line.len() - line.trim_start().len());
    let Some(step_indent) = step_indent else {
        return Vec::new();
    };
    let mut steps: Vec<Step> = Vec::new();
    let mut current: Option<Step> = None;
    let mut in_run: bool = false;
    for line in body.lines() {
        let trimmed: &str = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent: usize = line.len() - line.trim_start().len();
        if indent == step_indent && trimmed.starts_with("- ") {
            steps.extend(current.take());
            current = Some(Step::default());
            in_run = false;
        }
        let Some(step) = current.as_mut() else {
            continue;
        };
        for variable in ORACLE_PREFIX_VARS {
            if let Some(rest) = trimmed.strip_prefix(&format!("{variable}:")) {
                step.prefixes
                    .insert(rest.trim().trim_matches(['"', '\'']).to_string());
            }
        }
        let starts_run: Option<&str> = ["- run:", "run:"]
            .iter()
            .find(|key| trimmed.starts_with(**key))
            .copied();
        if let Some(key) = starts_run {
            in_run = true;
            step.script
                .push_str(trimmed[key.len()..].trim().trim_start_matches(['|', '>']));
            continue;
        }
        if STEP_KEYS.iter().any(|key| trimmed.starts_with(key)) {
            in_run = false;
            continue;
        }
        if in_run {
            step.script.push(' ');
            step.script.push_str(trimmed);
        }
    }
    steps.extend(current);
    steps
}

/// The prefixes a job prepends to PATH, read from its `$GITHUB_PATH` writes.
fn path_entry_prefixes_in(block: &str) -> BTreeSet<String> {
    let mut prefixes: BTreeSet<String> = BTreeSet::new();
    for line in block.lines() {
        let trimmed: &str = line.trim_start();
        if trimmed.starts_with('#') || !trimmed.contains("GITHUB_PATH") {
            continue;
        }
        let mut quoted = trimmed.split('"');
        let _before: Option<&str> = quoted.next();
        let Some(entry) = quoted.next() else {
            continue;
        };
        if let Some(prefix) = entry.strip_suffix("/bin") {
            if prefix.starts_with('/') {
                prefixes.insert(prefix.to_string());
            }
        }
    }
    prefixes
}

/// Does this job run on macOS, where a PATH entry does not select the oracle?
///
/// Reads the whole block rather than `runs-on:` alone: `test-cross-encode`
/// runs on `${{ matrix.os }}` and names macOS only in its matrix.
fn job_runs_on_macos(block: &str) -> bool {
    block
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .any(|line| line.contains("macos"))
}

#[test]
fn every_oracle_install_in_every_workflow_names_the_release_it_installs() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let mut offenders: Vec<String> = Vec::new();
    for (workflow, job, block) in every_job() {
        for (kind, line) in oracle_installs_in(&block) {
            let complaint: &str = match kind {
                OracleInstall::Unpinned => {
                    "installs a C libjpeg-turbo by package name, so the release \
                     it measures is the packager's choice and changes without a \
                     commit here"
                }
                OracleInstall::Unreadable => {
                    "fetches a C libjpeg-turbo in a shape this gate cannot \
                     resolve to a release; failing closed, because an \
                     unreadable pin and no pin are the same thing to a reader"
                }
                OracleInstall::Pinned | OracleInstall::Submodule => continue,
            };
            offenders.push(format!("  {workflow} / {job} — {complaint}\n      {line}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "every oracle install must name the release it installs:\n{}\n\n\
         Pin it the way the other legs are pinned — upstream's \
         `libjpeg-turbo-official_<version>` package, or a `--branch <tag>` \
         source build — and declare the release in {MANIFEST}. An oracle whose \
         release is named nowhere re-baselines every expectation it backs the \
         week the packager moves, which is the single global bump P4-130's \
         first criterion forbids, arriving without a commit to review.",
        offenders.join("\n")
    );
}

#[test]
fn every_job_that_installs_an_oracle_asserts_the_release_it_installed() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let declared: BTreeSet<String> = manifest_rows().into_iter().map(|row| row.version).collect();
    let mut offenders: Vec<String> = Vec::new();
    for (workflow, job, block) in every_job() {
        let installs_a_release: bool = oracle_installs_in(&block)
            .iter()
            .any(|(kind, _)| *kind == OracleInstall::Pinned);
        if !installs_a_release {
            continue;
        }
        let versions: BTreeSet<String> = provisioned_versions_in(&block);
        assert!(
            !versions.is_empty(),
            "{workflow} / {job} installs an upstream release asset but names no \
             version this gate can read"
        );
        let checked: BTreeMap<String, BTreeSet<String>> = prefix_releases_checked_in(&block);
        for version in versions {
            assert!(
                declared.contains(&version),
                "{workflow} / {job} installs libjpeg-turbo {version}, which \
                 {MANIFEST} does not declare"
            );
            // Checked *at a prefix*, not merely somewhere in the job: a job
            // carrying two oracles could otherwise satisfy both rows with one
            // check on one of them.
            if !checked.values().any(|releases| releases.contains(&version)) {
                offenders.push(format!("  {workflow} / {job} — installs {version}"));
            }
        }
    }
    // Every offender, not the first: these legs are spread over three
    // workflows, and a gate that names one per run turns one review into as
    // many rounds as there are legs.
    assert!(
        offenders.is_empty(),
        "these jobs install a release they never *assert*:\n{}\n\n\
         Printing `djpeg -version` fails nothing, so a deb that installed \
         something else, a tag moved to another release, or a runner image \
         carrying its own libjpeg would run the whole leg under a release it \
         is not measuring — and report green. The other legs check it with \
         `<prefix>/bin/djpeg -version 2>&1 | grep -q \"version <release>\"`.",
        offenders.join("\n")
    );
}

#[test]
fn every_job_that_installs_an_oracle_measures_the_install_it_checked() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let mut offenders: Vec<String> = Vec::new();
    for (workflow, job, block) in every_job() {
        let installs_a_release: bool = oracle_installs_in(&block)
            .iter()
            .any(|(kind, _)| *kind == OracleInstall::Pinned);
        if !installs_a_release {
            continue;
        }
        let checked: BTreeMap<String, BTreeSet<String>> = prefix_releases_checked_in(&block);
        if checked.is_empty() {
            offenders.push(format!(
                "  {workflow} / {job} — runs no `<prefix>/bin/djpeg -version` \
                 at all, so nothing ties the release it installed to a place \
                 on disk; a bare `djpeg` is answered by PATH and can be any \
                 djpeg on the runner"
            ));
            continue;
        }
        let job_level: Option<String> = oracle_prefixes_assigned_by(&block).job_level;
        let on_macos: bool = job_runs_on_macos(&block);
        let path_prefixes: BTreeSet<String> = path_entry_prefixes_in(&block);
        for (index, step) in steps_in(&block).into_iter().enumerate() {
            let consumes_an_oracle: bool = TEST_INVOCATIONS
                .iter()
                .any(|command| step.script.contains(command));
            // A step's own assignment wins over the job's for that step alone,
            // which is the whole reason this is read per step.
            let effective: BTreeSet<String> = if !step.prefixes.is_empty() {
                step.prefixes
            } else if let Some(prefix) = job_level.clone() {
                BTreeSet::from([prefix])
            } else {
                BTreeSet::new()
            };
            for prefix in &effective {
                if !checked.contains_key(prefix) {
                    offenders.push(format!(
                        "  {workflow} / {job} step {index} — selects {prefix}, \
                         which this job never checks the release of; it checks \
                         {:?}",
                        checked.keys().collect::<Vec<&String>>()
                    ));
                }
            }
            if consumes_an_oracle && effective.is_empty() {
                // Nothing names an oracle for this step, so it takes whatever
                // lookup order finds.
                let resolvable: bool = !on_macos
                    && path_prefixes
                        .iter()
                        .any(|prefix| checked.contains_key(prefix));
                if !resolvable {
                    offenders.push(format!(
                        "  {workflow} / {job} step {index} — runs tests against \
                         whatever lookup order finds{}",
                        if on_macos {
                            ", and this is a macOS runner, where \
                             `helpers::c_tool_path` reads /opt/homebrew/bin \
                             before PATH: only LIBJPEG_TURBO_PREFIX names an \
                             oracle there"
                        } else {
                            ", and no prefix this job checked is on its PATH"
                        }
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these steps measure an oracle the job never checked:\n{}\n\n\
         Installing the right release is not measuring it, and the scope that \
         matters is the *step*: a job-level prefix is overridden by a \
         step-level one for that step alone, so a leg can verify one install \
         and run its tests against another. Give the step a \
         LIBJPEG_TURBO_PREFIX the job checked, or — off macOS, where lookup \
         order can express it — put that prefix's bin first on PATH.",
        offenders.join("\n")
    );
}

/// Cargo invocations that reach the C oracle: the test binaries, the corpus
/// example, the mutation run whose oracle decides whether a mutant is caught,
/// and the differential fuzz targets that subprocess `djpeg`/`cjpeg`.
const TEST_INVOCATIONS: [&str; 4] = ["cargo test", "cargo run", "cargo mutants", "fuzz run"];

#[test]
fn an_unpinned_package_manager_install_is_an_oracle_install() {
    // The shape this half of the gate exists for. Both spellings: homebrew's
    // formula and a distribution package, neither of which names a release.
    for line in [
        "            install: brew install jpeg-turbo",
        "          sudo apt-get install -y libjpeg-turbo-progs",
        "          brew install libjpeg-turbo",
    ] {
        assert_eq!(
            oracle_install_on(line),
            Some(OracleInstall::Unpinned),
            "{line:?} installs a C oracle under a name the packager controls"
        );
    }
}

#[test]
fn installing_a_pinned_asset_is_not_an_unpinned_install() {
    // The pinned shapes must not be swept up by the package-manager rule —
    // upstream's deb *is* installed with `apt-get install`, by file path.
    for line in [
        "          sudo apt-get install -y /tmp/ljt.deb",
        "            \"https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/\
         ${VERSION}/libjpeg-turbo-official_${VERSION}_${ARCH}.deb\"",
        "          git clone --depth 1 --branch 3.1.4.1 \
         https://github.com/libjpeg-turbo/libjpeg-turbo.git /tmp/ljt",
    ] {
        assert_ne!(
            oracle_install_on(line),
            Some(OracleInstall::Unpinned),
            "{line:?} names the release it installs"
        );
    }
    // Tooling installs are not oracle installs, however they are spelled.
    for line in [
        "          command -v cmake >/dev/null || brew install cmake",
        "          sudo apt-get install -y cmake nasm",
        "        run: cargo publish -p libjpeg-turbo-rs-capi",
    ] {
        assert_eq!(
            oracle_install_on(line),
            None,
            "{line:?} installs no C libjpeg-turbo"
        );
    }
}

#[test]
fn a_documented_install_is_not_an_install() {
    // `fuzz-smoke.yml` prints reproduction instructions on failure: the deb
    // URL, the release, the PATH entry. Read as actions they would satisfy
    // this gate from inside an error message, and `full-c-parity.yml`'s
    // comments discuss `brew install jpeg-turbo` by name in order to explain
    // why it is gone.
    for line in [
        "  # jpeg-turbo, so `brew install jpeg-turbo` is an oracle whose release is",
        "                echo \"  VERSION=3.1.4.1\"",
        "                echo \"  Install libjpeg-turbo 3.1.4.1 C tools and put \
         /opt/libjpeg-turbo/bin first on PATH.\"",
        "          echo \"/opt/libjpeg-turbo/bin\" >> $GITHUB_PATH",
        "      - name: Install libjpeg-turbo 3.2.0 from official release",
    ] {
        assert_eq!(
            oracle_install_on(line),
            None,
            "{line:?} describes an install rather than performing one"
        );
    }
}

#[test]
fn a_clone_with_no_tag_is_unreadable_rather_than_absent() {
    // A clone of the default branch is the worst pin of all: it moves every
    // time upstream commits. Reporting it as "no install here" would let it
    // pass; reporting it as unreadable fails the gate and asks for a tag.
    assert_eq!(
        oracle_install_on(
            "          git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git /tmp/ljt"
        ),
        Some(OracleInstall::Unreadable)
    );
    assert_eq!(
        oracle_install_on(
            "          git clone --depth 1 --branch 3.2.0 \
             https://github.com/libjpeg-turbo/libjpeg-turbo.git /tmp/ljt320src"
        ),
        Some(OracleInstall::Pinned)
    );
}

#[test]
fn a_submodule_build_is_pinned_by_commit_rather_than_by_a_version_token() {
    assert_eq!(
        oracle_install_on("          cmake -S references/libjpeg-turbo -B /tmp/ljt8/build \\"),
        Some(OracleInstall::Submodule),
        "the submodule is pinned by commit; its release is cross-checked \
         against the manifest from the checked-out tree"
    );
    // Reading a file out of the submodule is not building it.
    assert_eq!(
        oracle_install_on("            references/libjpeg-turbo/testimages/testorig.jpg \\"),
        None
    );
}

#[test]
fn a_continued_command_is_classified_as_the_one_command_it_is() {
    // Both provisioning shapes in `ci.yml` are split across lines, and each
    // half alone is invisible: the first names no release, the second no
    // command.
    let split_clone: &str = "          git clone --depth 1 --branch 3.2.0 \\\n\
                             \x20           https://github.com/libjpeg-turbo/libjpeg-turbo.git \
                             /tmp/ljt320src";
    let joined: Vec<String> = logical_lines(split_clone);
    assert_eq!(joined.len(), 1, "a continuation is one command: {joined:?}");
    assert_eq!(oracle_install_on(&joined[0]), Some(OracleInstall::Pinned));

    let split_curl: &str = "          curl -fL -o /tmp/ljt.deb \\\n\
                            \x20           \"https://github.com/libjpeg-turbo/libjpeg-turbo/\
                            releases/download/${VERSION}/libjpeg-turbo-official_${VERSION}_\
                            ${ARCH}.deb\"";
    let joined: Vec<String> = logical_lines(split_curl);
    assert_eq!(joined.len(), 1);
    assert_eq!(oracle_install_on(&joined[0]), Some(OracleInstall::Pinned));
}

#[test]
fn printing_the_release_is_not_asserting_it() {
    // The shape five of these legs carried: an absolute-path `-version` call
    // whose output nothing reads.
    assert!(!asserts_release(
        "          /opt/libjpeg-turbo/bin/djpeg -version",
        "3.1.4.1"
    ));
    assert!(asserts_release(
        "          /opt/libjpeg-turbo/bin/djpeg -version 2>&1 | grep -q \"version 3.1.4.1\"",
        "3.1.4.1"
    ));
    // The source-tree spelling, for a leg that checks the tag it cloned rather
    // than the tool it built.
    assert!(asserts_release(
        "          grep -Eq '^set\\(VERSION 3\\.2\\.0\\)$' /tmp/ljt320src/CMakeLists.txt",
        "3.2.0"
    ));
    // …and a check on the wrong release is not a check on this one.
    assert!(!asserts_release(
        "          /opt/libjpeg-turbo/bin/djpeg -version 2>&1 | grep -q \"version 3.2.0\"",
        "3.1.4.1"
    ));
}

#[test]
fn a_bare_tool_name_does_not_check_an_install() {
    // `test-integration` ran `djpeg -version` after exporting PATH. It reads
    // as a version check and names no install: which djpeg answered is decided
    // by lookup order, which is the ambiguity this item is about.
    assert!(
        prefix_releases_checked_in("          djpeg -version | grep -q \"version 3.1.4.1\"")
            .is_empty()
    );
    // The one-line spelling and the `tee`-then-`grep` spelling both check the
    // prefix they name; the second is what the older legs carry.
    assert_eq!(
        prefix_releases_checked_in(
            "          /tmp/ljt3141/prefix/bin/djpeg -version 2>&1 | grep -q \"version 3.1.4.1\""
        ),
        BTreeMap::from([(
            "/tmp/ljt3141/prefix".to_string(),
            BTreeSet::from(["3.1.4.1".to_string()])
        )])
    );
    assert_eq!(
        prefix_releases_checked_in(
            "          /usr/local/bin/djpeg -version 2>&1 | tee /tmp/oracle-version.txt\n\
             \x20         grep -q \"version 3.1.4.1\" /tmp/oracle-version.txt"
        ),
        BTreeMap::from([(
            "/usr/local".to_string(),
            BTreeSet::from(["3.1.4.1".to_string()])
        )])
    );
}

#[test]
fn one_prefixs_check_does_not_vouch_for_another() {
    // The shape a job-scope pairing accepts and this one must not: two oracles
    // in one job, one release asserted. `test-integration` carries three
    // prefixes, so "somewhere in this job a version is asserted" would let a
    // second oracle ride in unchecked — the finding that turned this from a
    // set of prefixes into a map.
    let two_oracles: &str = "          /opt/libjpeg-turbo/bin/djpeg -version 2>&1 | \
                             grep -q \"version 3.1.4.1\"\n\
                             \x20         /tmp/ljt8/prefix/bin/djpeg -version";
    let checked: BTreeMap<String, BTreeSet<String>> = prefix_releases_checked_in(two_oracles);
    assert!(checked.contains_key("/opt/libjpeg-turbo"));
    assert!(
        !checked.contains_key("/tmp/ljt8/prefix"),
        "the second prefix is run, not checked: {checked:?}"
    );
    // A `tee` file belongs to the invocation that wrote it. Asserting one
    // release over one file does not check the *other* prefix, however
    // adjacent the lines are.
    let crossed: &str = "          /opt/libjpeg-turbo/bin/djpeg -version 2>&1 | tee /tmp/a.txt\n\
                         \x20         /tmp/ljt8/prefix/bin/djpeg -version 2>&1 | tee /tmp/b.txt\n\
                         \x20         grep -q \"version 3.1.4.1\" /tmp/a.txt";
    let checked: BTreeMap<String, BTreeSet<String>> = prefix_releases_checked_in(crossed);
    assert_eq!(
        checked.keys().collect::<Vec<&String>>(),
        vec!["/opt/libjpeg-turbo"],
        "{checked:?}"
    );
}

#[test]
fn a_step_level_prefix_is_read_at_step_scope() {
    // Why this is parsed per step at all: a job-level value and a step-level
    // override read identically once indentation is gone, and the difference
    // decides which oracle a `cargo test` step measures.
    let job: &str = "\n    steps:\n\
                     \x20     - name: Install\n\
                     \x20       run: echo install\n\
                     \x20     - name: Tests against the deb\n\
                     \x20       run: cargo test --tests\n\
                     \x20       env:\n\
                     \x20         LIBJPEG_TURBO_PREFIX: /opt/libjpeg-turbo\n\
                     \x20     - name: Traces against the v8 build\n\
                     \x20       run: cargo test -p libjpeg-turbo-rs-capi --test capi_x\n\
                     \x20       env:\n\
                     \x20         LIBJPEG_TURBO_REFERENCE_DIR: /tmp/ljt8/prefix\n";
    let steps: Vec<Step> = steps_in(job);
    assert_eq!(steps.len(), 3, "{steps:?}");
    assert!(steps[0].prefixes.is_empty());
    assert!(!steps[0].script.contains("cargo test"));
    assert_eq!(
        steps[1].prefixes,
        BTreeSet::from(["/opt/libjpeg-turbo".to_string()])
    );
    assert!(steps[1].script.contains("cargo test"));
    // P4-108's spelling of the same idea names the oracle just as surely.
    assert_eq!(
        steps[2].prefixes,
        BTreeSet::from(["/tmp/ljt8/prefix".to_string()])
    );
}

#[test]
fn a_matrix_entry_is_not_the_first_step() {
    // `test-cross-encode` declares its runner in a matrix, and a matrix entry
    // is a `- ` line that comes before `steps:` and sits deeper. Reading the
    // step indent off the first `- ` in the job took *that* line, so no real
    // step boundary matched and every step — with every prefix any of them
    // set — merged into one. That is the job-scope union this parser exists to
    // replace, arriving through the parser instead of the rule.
    let job: &str = "    runs-on: ${{ matrix.os }}\n\
                     \x20   strategy:\n\
                     \x20     matrix:\n\
                     \x20       include:\n\
                     \x20         - os: macos-latest\n\
                     \x20   steps:\n\
                     \x20     - uses: actions/checkout@v7\n\
                     \x20     - name: Tests\n\
                     \x20       run: cargo test --tests\n";
    let steps: Vec<Step> = steps_in(job);
    assert_eq!(steps.len(), 2, "{steps:?}");
    assert!(
        !steps[0].script.contains("cargo test"),
        "the checkout step runs no tests: {steps:?}"
    );
    assert!(steps[1].script.contains("cargo test"));
}

#[test]
fn every_cargo_invocation_that_reaches_the_oracle_is_recognised() {
    // A step this list does not match is a step the gate never asks about, so
    // an unchecked oracle would ride in under a spelling nobody added here.
    // These are the four shapes the workflows use today.
    for script in [
        "cargo test --tests",
        "cargo run --release --example corpus_test -- --corpus-dir tests/corpus/",
        "cargo mutants --in-diff /tmp/pr.diff",
        "cargo +nightly fuzz run --target x86_64-unknown-linux-gnu \"${FUZZ_TARGET}\"",
    ] {
        assert!(
            TEST_INVOCATIONS
                .iter()
                .any(|command| script.contains(command)),
            "{script:?} reaches the C oracle and no entry recognises it"
        );
    }
    // …and a step that only installs does not.
    for script in [
        "sudo apt-get install -y /tmp/ljt.deb",
        "cargo build --release",
    ] {
        assert!(
            !TEST_INVOCATIONS
                .iter()
                .any(|command| script.contains(command)),
            "{script:?} runs no tests"
        );
    }
}

#[test]
fn a_path_entry_selects_an_oracle_everywhere_except_macos() {
    let linux_leg: &str = "    runs-on: ubuntu-latest\n\
                           \x20         echo \"/opt/libjpeg-turbo/bin\" >> $GITHUB_PATH\n";
    assert!(!job_runs_on_macos(linux_leg));
    assert_eq!(
        path_entry_prefixes_in(linux_leg),
        BTreeSet::from(["/opt/libjpeg-turbo".to_string()])
    );
    // The matrix spelling: `runs-on: ${{ matrix.os }}` names macOS nowhere but
    // in the matrix, and that is the leg where PATH does not select.
    let macos_leg: &str = "    runs-on: ${{ matrix.os }}\n\
                           \x20         - os: macos-latest  # aarch64\n";
    assert!(job_runs_on_macos(macos_leg));
    // A comment recalling that some other leg runs on macOS must not make this
    // one a macOS leg, or the rule would be applied where it does not hold.
    assert!(!job_runs_on_macos(
        "    runs-on: ubuntu-latest\n      # matches the macos leg's oracle\n"
    ));
}

#[test]
fn the_job_scanner_reads_jobs_and_not_trigger_keys() {
    if !repository_tree_is_readable() {
        eprintln!("SKIP: repository tree not readable; see the sibling test.");
        return;
    }
    let text: String = workflow_text(CI_WORKFLOW);
    let jobs: Vec<String> = job_names_in(&text);
    for expected in [BASELINE_LEG_JOB, CURRENT_LEG_JOB, "test-cross-encode"] {
        assert!(
            jobs.iter().any(|job| job == expected),
            "{CI_WORKFLOW} has a {expected} job, but the scanner found {jobs:?}"
        );
    }
    // `on:` carries two-space keys of its own; reading them as jobs would ask
    // `job_block` for a block that is not one.
    for trigger in ["push", "pull_request", "schedule", "workflow_dispatch"] {
        assert!(
            !jobs.iter().any(|job| job == trigger),
            "{trigger:?} is a trigger, not a job"
        );
    }
    // Every workflow in the repository parses, and the enumeration is not
    // silently empty for one of them — an empty list passes every gate above.
    for path in workflow_files() {
        let text: String = std::fs::read_to_string(&path).expect("workflow must be readable");
        assert!(
            !job_names_in(&text).is_empty(),
            "no jobs found in {} — the scanner has stopped matching, and a \
             workflow it cannot read is a workflow these gates do not check",
            path.display()
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
