//! P4-141 criterion 1: the Miri job really selects the suites that cover the
//! surfaces the criterion names.
//!
//! Every suite in this repository compiles on every native leg, and three of
//! them — `miri_public_api`, `miri_alloc_failure`, `miri_once_init` — are only
//! worth having because an *interpreter* runs them. Nothing in their source says
//! so. The selection lives in one `run:` line each in `.github/workflows/ci.yml`,
//! and a suite dropped from that line keeps compiling, keeps passing on the
//! native legs, and stops being interpreted — with no failure anywhere. That is
//! the shape #320 named: a job whose name claims a mechanism it does not run.
//!
//! So this gate reads the workflow and requires, on every pull request:
//!
//! * each surface's suite is selected by a `cargo miri test` step, and the file
//!   holds the tests that cover it, at least one of them runnable under Miri;
//! * the concurrency suite's step passes `-Zmiri-many-seeds`, because a single
//!   interleaving is not a search and the loser branch of a `compare_exchange`
//!   is what that suite exists to reach;
//! * the doctests are selected, which `--lib` does not build;
//! * the pre-existing selections are still there — the `--lib` run with
//!   `--skip simd::` and the two C-ABI suites — so this gate covers the whole
//!   job rather than only what its author added;
//! * **every** `tests/miri_*.rs` file is selected by some step, so a fourth
//!   suite cannot be added and left uninterpreted;
//! * every `#[ignore]` **anywhere in those suites** cites an issue — not only
//!   the ones a surface names — so a surface cannot be quietly switched off. One
//!   exemption exists and is named in code, the C-oracle tests Miri cannot spawn
//!   a process for; `miri_alloc_failure`'s cited ignore is the mainline decode
//!   contract that P4-209 (#632) breaks;
//! * no interpreting step is *non-executing*: `--no-run` and `--list` compile the
//!   suite while satisfying every selection above, and an `if:` — on the step or
//!   on the **job**, which switches all of them off at once — is coverage that can
//!   be withdrawn without touching a command;
//! * no interpreting step is *forgiven*: `continue-on-error:` on the step or on
//!   the job lets it run, fail and report green — the same withdrawal from the
//!   other side, and one line either way;
//! * no step selecting a `miri_*` suite passes a libtest filter after `--`,
//!   which would narrow the suite to a subset while every selection above still
//!   sees its name (the `--lib` step's `--skip simd::` is required, so filters
//!   are judged per step);
//! * no `miri_*` suite carries a `cfg` on `miri`. `#![cfg(not(miri))]` at the
//!   top of a file leaves every named test defined, keeps the native legs
//!   green, and makes the interpreting step run zero tests and exit 0 — the
//!   shortest withdrawal there is. Only the per-test
//!   `cfg_attr(miri, ignore = ...)` is allowed, and the ignore rules below
//!   hold that to citing an issue.
//!
//! Isolation is required only of the steps whose suites read files (the `--lib`
//! run and the two C-ABI suites). Requiring it everywhere would report a step
//! that runs *with* isolation — a stronger configuration — as a problem.
//!
//! What it cannot check is that a selected suite *asserts* anything; that is a
//! review question, and `.github/CODEOWNERS` routes these four suites and
//! `ci.yml` to the maintainer for it — lines this landing added, because the
//! clause was here first and the routing was not (`docs-drift-auditor`,
//! 2026-09-09). The rules
//! themselves are pure functions over (workflow text, suite sources), which is
//! what lets `the_rules_reject_each_way_the_job_can_go_stale` drive all of them
//! over fixtures and
//! `dropping_a_selection_from_the_real_workflow_is_reported` drive two of them
//! over the shape this repository actually has — the gate's own mutation check,
//! in the shape the criterion-4 gate settled on. The fixture alone was not
//! enough: it is the author's idea of the job, and the review that asked for the
//! real-workflow pass is the same one that found the two non-executing forms.

#![cfg(not(target_arch = "wasm32"))]

use std::collections::BTreeSet;
use std::path::PathBuf;

const WORKFLOW: &str = ".github/workflows/ci.yml";
const JOB: &str = "miri";

/// One surface acceptance criterion 1 names, and the suite that covers it.
struct Surface {
    /// The criterion's own words, so a reader can check the mapping against the
    /// item rather than against this file.
    named: &'static str,
    /// Path relative to the crate root.
    suite: &'static str,
    /// `--test` argument that selects it.
    selection: &'static str,
    /// Tests that must exist in the suite. At least one must be runnable under
    /// Miri; an `#[ignore]`d one still has to be listed, so deleting it is a
    /// failure rather than a silent narrowing.
    tests: &'static [&'static str],
}

const SURFACES: [Surface; 4] = [
    Surface {
        named: "non-SIMD integration tests",
        suite: "tests/miri_public_api.rs",
        selection: "--test miri_public_api",
        // The whole suite is integration-level: it reaches the crate only
        // through its public API, which is what `--lib` cannot do.
        tests: &[
            "progressive_intermediate_outputs_are_fully_initialised",
            "progressive_grayscale_intermediate_outputs_are_fully_initialised",
            "baseline_encode_grows_the_bitwriter_and_takes_the_ff_stuffing_path",
            "progressive_encode_grows_the_bitwriter_across_a_reset",
        ],
    },
    Surface {
        named: "progressive output (P4-136)",
        suite: "tests/miri_public_api.rs",
        selection: "--test miri_public_api",
        tests: &[
            "progressive_intermediate_outputs_are_fully_initialised",
            "progressive_grayscale_intermediate_outputs_are_fully_initialised",
        ],
    },
    Surface {
        named: "BitWriter (P4-138)",
        suite: "tests/miri_public_api.rs",
        selection: "--test miri_public_api",
        tests: &[
            "baseline_encode_grows_the_bitwriter_and_takes_the_ff_stuffing_path",
            "progressive_encode_grows_the_bitwriter_across_a_reset",
        ],
    },
    Surface {
        named: "post-allocation-failure state",
        suite: "tests/miri_alloc_failure.rs",
        selection: "--test miri_alloc_failure",
        tests: &[
            "selftest_the_injector_refuses_exactly_what_it_is_armed_for",
            "a_refused_progressive_output_leaves_the_decoder_usable",
            "a_refused_icc_reassembly_leaves_the_decoder_usable",
            "the_mainline_decode_reports_refusal_instead_of_aborting",
        ],
    },
];

/// Concurrent one-time initialisation is a fifth surface with a requirement the
/// others do not have, so it is stated apart rather than folded into
/// [`SURFACES`]: its step must search interleavings.
const CONCURRENCY: Surface = Surface {
    named: "concurrent one-time initialisation",
    suite: "tests/miri_once_init.rs",
    selection: "--test miri_once_init",
    tests: &["racing_initialisers_publish_exactly_one_set_of_standard_tables"],
};

const MANY_SEEDS: &str = "-Zmiri-many-seeds";

/// Arguments that turn `cargo miri test` into a compile: the suite is built, no
/// test is interpreted, and every selection check still sees its name.
const NON_EXECUTING_FORMS: [&str; 2] = ["--no-run", "--list"];

/// Selections the job carried before criterion 1 and must keep carrying. A gate
/// that only knows about its author's additions lets the surface it was built on
/// disappear.
const PRE_EXISTING: [&str; 4] = [
    "--lib",
    "--skip simd::",
    "--test capi_create_abi_guards",
    "--test capi_thread_affinity",
];

/// The doctest selection, which has no suite file to point at.
const DOCTESTS: &str = "--doc";

/// Selections whose suites read files, and therefore need Miri's isolation off.
/// The root `--lib` run stats fixtures; the two C-ABI suites read a JPEG from
/// disk. The three `miri_*` suites synthesise every byte they use, so they do
/// not appear here — and are not required to disable isolation.
const NEEDS_ISOLATION_OFF: [&str; 3] = [
    "--lib",
    "--test capi_create_abi_guards",
    "--test capi_thread_affinity",
];

// ---------------------------------------------------------------------------
// Workflow reading
// ---------------------------------------------------------------------------

/// One step of the Miri job: its shell script, and the `MIRIFLAGS` it runs with.
#[derive(Debug, Default, Clone)]
struct Step {
    script: String,
    miriflags: String,
    /// The step's `if:` expression, if it has one. A step can be switched off
    /// without touching its command, so a selection that only reads the command
    /// is satisfied by a step that never runs (codex review, 2026-09-09).
    condition: Option<String>,
    /// The step's `continue-on-error:` value, if it has one. Its sibling of the
    /// `if:` hole: the step runs, fails, and the job reports green — one line,
    /// no command touched (rust-code-reviewer, 2026-09-09).
    continue_on_error: Option<String>,
}

/// The steps of a named job.
///
/// Hand-rolled rather than pulled from a YAML crate: this repository has no YAML
/// dependency, and the shapes that appear here are the three GitHub allows for a
/// script — a quoted scalar, a folded `>` block and a literal `|` block. Each is
/// covered by a fixture in the self-tests below.
fn steps_of(workflow: &str, job: &str) -> Job {
    let header: String = format!("  {job}:");
    let mut inside_job: bool = false;
    let mut steps: Vec<Step> = Vec::new();
    let mut collecting: Option<Collecting> = None;
    let mut job_condition: Option<String> = None;
    let mut job_continue_on_error: Option<String> = None;
    let mut job_miriflags: String = String::new();

    for line in workflow.lines() {
        if !inside_job {
            inside_job = line.trim_end() == header;
            continue;
        }
        let indent: usize = line.len() - line.trim_start().len();
        let trimmed: &str = line.trim();
        if !trimmed.is_empty() && indent <= 2 && !trimmed.starts_with('#') {
            break; // the next job's header
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // A continuation of the script or env block we are inside.
        match collecting {
            Some(Collecting::Script { indent: key_indent }) if indent > key_indent => {
                let step: &mut Step = steps.last_mut().expect("a script needs a step");
                step.script.push(' ');
                step.script.push_str(trimmed);
                continue;
            }
            Some(Collecting::Env { indent: key_indent }) if indent > key_indent => {
                if let Some(value) = trimmed.strip_prefix("MIRIFLAGS:") {
                    // Before the first `- `, this is the *job's* `env:`, which
                    // every step inherits. Panicking there (the first version's
                    // `expect("an env needs a step")`) would have rejected an
                    // ordinary simplification of this job as a parse failure
                    // (rust-code-reviewer, 2026-09-09).
                    match steps.last_mut() {
                        Some(step) => step.miriflags = value.trim().to_string(),
                        None => job_miriflags = value.trim().to_string(),
                    }
                }
                continue;
            }
            _ => collecting = None,
        }

        let (is_new_step, body): (bool, &str) = match trimmed.strip_prefix("- ") {
            Some(rest) => (true, rest),
            None => (false, trimmed),
        };
        if is_new_step {
            steps.push(Step::default());
        }
        let step_indent: usize = if is_new_step { indent + 2 } else { indent };

        if let Some(rest) = body.strip_prefix("run:") {
            let rest: &str = rest.trim();
            if rest == ">" || rest == "|" || rest == ">-" || rest == "|-" {
                collecting = Some(Collecting::Script {
                    indent: step_indent,
                });
            } else {
                let step: &mut Step = steps.last_mut().expect("a run needs a step");
                step.script = unquote(rest).to_string();
            }
        } else if body.trim_end() == "env:" {
            collecting = Some(Collecting::Env {
                indent: step_indent,
            });
        } else if let Some(rest) = body.strip_prefix("if:") {
            let condition: String = unquote(rest.trim()).to_string();
            // A job-level `if:` switches off every step at once, which is the
            // same #320 shape as a step-level one and was the larger of the two
            // holes: the first version reached for `steps.last_mut()` and
            // panicked instead of reporting it.
            match steps.last_mut() {
                Some(step) => step.condition = Some(condition),
                None => job_condition = Some(condition),
            }
        } else if let Some(rest) = body.strip_prefix("continue-on-error:") {
            let value: String = unquote(rest.trim()).to_string();
            match steps.last_mut() {
                Some(step) => step.continue_on_error = Some(value),
                None => job_continue_on_error = Some(value),
            }
        }
    }

    assert!(
        inside_job,
        "{WORKFLOW} has no job named {job:?} — a renamed job must reach this \
         gate rather than leave it comparing nothing"
    );
    if !job_miriflags.is_empty() {
        for step in &mut steps {
            if step.miriflags.is_empty() {
                step.miriflags = job_miriflags.clone();
            }
        }
    }
    Job {
        steps,
        condition: job_condition,
        continue_on_error: job_continue_on_error,
    }
}

/// A job: its steps, and the two keys that can withdraw all of them at once —
/// an `if:` that stops them running and a `continue-on-error:` that keeps them
/// running and stops their failure meaning anything.
#[derive(Debug, Default, Clone)]
struct Job {
    steps: Vec<Step>,
    condition: Option<String>,
    continue_on_error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum Collecting {
    Script { indent: usize },
    Env { indent: usize },
}

fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|rest| rest.strip_suffix('\''))
        })
        .unwrap_or(value)
}

/// Whether a YAML scalar reads as true. GitHub accepts `true` and the string
/// `"true"`; anything else (including `${{ ... }}`, which cannot be evaluated
/// here) is reported by the caller as unknown rather than silently accepted —
/// so this returns true for those too.
fn is_truthy(value: &str) -> bool {
    !matches!(value.trim(), "false" | "'false'" | "\"false\"" | "")
}

/// Cargo options that take a separate value, so the word after them is not a
/// test-name filter.
const OPTIONS_TAKING_A_VALUE: [&str; 12] = [
    "--test",
    "--features",
    "-p",
    "--package",
    "--target",
    "--manifest-path",
    "--profile",
    "--bin",
    "--example",
    "--bench",
    "--jobs",
    "-j",
];

/// Arguments that narrow what a step *runs* on a step selecting a `miri_*`
/// suite: everything after a bare `--`, and any bare positional word before it.
///
/// Both forms reach libtest. `cargo miri test --test miri_once_init racing`
/// passes `racing` straight through as a name filter, so a step can be reduced
/// to one test — or, with a name that matches nothing, to none — while every
/// selection check above still sees `--test miri_once_init` (codex review,
/// 2026-09-09).
///
/// Returns nothing for a step that selects no `miri_*` suite, because the
/// `--lib` run's `--skip simd::` is a filter this job requires.
fn suite_filters(script: &str) -> Vec<String> {
    if !script.contains("--test miri_") {
        return Vec::new();
    }
    let mut filters: Vec<String> = Vec::new();
    let mut words = script.split_whitespace();
    // `cargo miri test` itself, and anything before it (`caffeinate`, an env
    // assignment), is not a filter.
    if words.by_ref().find(|word| *word == "test").is_none() {
        return Vec::new();
    }
    while let Some(word) = words.next() {
        if word == "--" {
            filters.extend(words.map(str::to_string));
            break;
        }
        if OPTIONS_TAKING_A_VALUE.contains(&word) {
            words.next();
            continue;
        }
        if word.starts_with('-') {
            continue;
        }
        // A `|` block writes the command over several lines; the continuation
        // backslash is shell syntax, not an argument.
        if word == "\\" {
            continue;
        }
        filters.push(word.to_string());
    }
    filters
}

/// The steps that interpret something, i.e. run `cargo miri test`.
fn interpreting_steps(steps: &[Step]) -> Vec<&Step> {
    steps
        .iter()
        .filter(|step| step.script.contains("cargo miri test"))
        .collect()
}

// ---------------------------------------------------------------------------
// The rules, as functions so the self-tests can drive them
// ---------------------------------------------------------------------------

/// Problems with what the job selects. Empty means the job is complete.
fn selection_problems(workflow: &str, surfaces: &[&Surface]) -> Vec<String> {
    let job: Job = steps_of(workflow, JOB);
    let interpreting: Vec<&Step> = interpreting_steps(&job.steps);
    let mut problems: Vec<String> = Vec::new();

    if let Some(condition) = &job.condition {
        problems.push(format!(
            "the {JOB} job is conditional on {condition:?}, which switches every \
             step below off at once without touching a command"
        ));
    }
    if let Some(value) = &job.continue_on_error {
        if is_truthy(value) {
            problems.push(format!(
                "the {JOB} job carries continue-on-error: {value:?}, so every step \
                 below may fail and the job still reports green — the same \
                 withdrawal as an `if:`, from the other side"
            ));
        }
    }

    if interpreting.is_empty() {
        problems.push(format!(
            "job {JOB:?} runs no `cargo miri test` step at all, so every check \
             below would pass over nothing"
        ));
        return problems;
    }

    for surface in surfaces {
        let selected: bool = interpreting
            .iter()
            .any(|step| step.script.contains(surface.selection));
        if !selected {
            problems.push(format!(
                "no `cargo miri test` step passes {} — the {} surface would be \
                 compiled by the native legs and interpreted by nobody",
                surface.selection, surface.named
            ));
        }
    }

    for required in PRE_EXISTING.iter().chain([DOCTESTS].iter()) {
        if !interpreting
            .iter()
            .any(|step| step.script.contains(*required))
        {
            problems.push(format!(
                "no `cargo miri test` step passes {required:?}, which the job \
                 carried before this gate"
            ));
        }
    }

    for step in &interpreting {
        // A command that compiles and does not interpret satisfies every
        // selection above while running nothing.
        for form in NON_EXECUTING_FORMS {
            if step.script.contains(form) {
                problems.push(format!(
                    "step {:?} passes {form:?}, which compiles the suite without \
                     interpreting it",
                    step.script
                ));
            }
        }
        if let Some(condition) = &step.condition {
            problems.push(format!(
                "step {:?} is conditional on {condition:?}; a step that can be \
                 switched off without touching its command is not coverage. \
                 Justify it here if the condition is deliberate",
                step.script
            ));
        }
        // A step that runs, fails and is forgiven is coverage on paper only,
        // and it is one line that touches no command (rust-code-reviewer,
        // 2026-09-09).
        if let Some(value) = &step.continue_on_error {
            if is_truthy(value) {
                problems.push(format!(
                    "step {:?} carries continue-on-error: {value:?}; its failure \
                     would not fail the job",
                    step.script
                ));
            }
        }
        // A libtest filter narrows what a selected suite runs while every
        // selection check above still sees its name. The `--lib` step's
        // `--skip simd::` is the one this job wants, and it is required
        // elsewhere; on a `--test miri_*` step there is nothing to allow.
        for filter in suite_filters(&step.script) {
            problems.push(format!(
                "step {:?} passes {filter:?} to libtest, which narrows a \
                 miri_* suite to a subset of its tests",
                step.script
            ));
        }
        // Only where it is *needed*. Requiring it of every step would report a
        // step that runs with isolation — a strictly stronger configuration — as
        // a problem (rust-code-reviewer, 2026-09-09).
        let needs_isolation_off: bool = NEEDS_ISOLATION_OFF
            .iter()
            .any(|selection| step.script.contains(selection));
        if needs_isolation_off && !step.miriflags.contains("-Zmiri-disable-isolation") {
            problems.push(format!(
                "step {:?} runs without -Zmiri-disable-isolation, and the suites \
                 it selects read fixtures from disk — they would fail on the \
                 first stat",
                step.script
            ));
        }
    }

    match interpreting
        .iter()
        .find(|step| step.script.contains(CONCURRENCY.selection))
    {
        None => problems.push(format!(
            "no step passes {} — the {} surface is unselected",
            CONCURRENCY.selection, CONCURRENCY.named
        )),
        Some(step) if !step.miriflags.contains(MANY_SEEDS) => problems.push(format!(
            "the step selecting {} runs with MIRIFLAGS {:?}, which lacks \
             {MANY_SEEDS}: one interleaving is not a search, and the \
             compare_exchange loser branch is what that suite exists to reach",
            CONCURRENCY.selection, step.miriflags
        )),
        Some(_) => {}
    }

    problems
}

/// Problems with a suite's own source: a missing test, or an `#[ignore]` that
/// cites nothing.
fn suite_problems(source: &str, surface: &Surface) -> Vec<String> {
    let mut problems: Vec<String> = Vec::new();
    let mut runnable: usize = 0;

    for test in surface.tests {
        match test_attributes(source, test) {
            None => problems.push(format!(
                "{} does not define {test} — the {} surface names it",
                surface.suite, surface.named
            )),
            Some(attributes) => {
                let registered: bool = attributes
                    .iter()
                    .any(|line| line.starts_with("#[test]") || line.contains("test]"));
                let ignored: bool = attributes.iter().any(|line| line.contains("ignore"));
                if !registered {
                    // libtest runs what carries `#[test]`. Without the attribute
                    // the function is dead code the compiler may not even warn
                    // about in a test binary, and the suite executes nothing
                    // (codex review, 2026-09-09).
                    problems.push(format!(
                        "{}::{test} is not a `#[test]`, so nothing executes it",
                        surface.suite
                    ));
                } else if !ignored {
                    runnable += 1;
                } else if !attributes
                    .iter()
                    .filter(|line| line.contains("ignore"))
                    .all(|line| cites_an_issue(line))
                {
                    problems.push(format!(
                        "{}::{test} is ignored without citing an issue; an \
                         ignore is how a surface switches itself off quietly",
                        surface.suite
                    ));
                }
            }
        }
    }

    if runnable == 0 {
        problems.push(format!(
            "every test the {} surface names in {} is ignored, so the suite \
             runs nothing for it",
            surface.named, surface.suite
        ));
    }
    problems
}

/// The attributes immediately above `fn <name>`, each joined onto one line, or
/// `None` if the function is not defined at all.
///
/// Driven by [`attribute_spans`] rather than by a line heuristic, so a
/// `rustfmt`-split attribute is one entry here instead of a `)]` fragment the
/// walk stops at (rust-code-reviewer, 2026-09-09).
fn test_attributes(source: &str, name: &str) -> Option<Vec<String>> {
    let lines: Vec<&str> = source.lines().collect();
    let at: usize = lines
        .iter()
        .position(|line| line.trim_start().starts_with(&format!("fn {name}(")))?;
    let spans: Vec<(String, usize, usize)> = attribute_spans(source);

    let mut collected: Vec<String> = Vec::new();
    let mut cursor: usize = at;
    while cursor > 0 {
        let previous: usize = cursor - 1;
        if let Some((attribute, start, _)) = spans.iter().find(|(_, _, end)| *end == previous) {
            collected.push(attribute.clone());
            cursor = *start;
            continue;
        }
        let trimmed: &str = lines[previous].trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            cursor = previous;
            continue;
        }
        break;
    }
    Some(collected)
}

/// The one ignore reason allowed without an issue: the C-oracle tests, which
/// cannot run under an interpreter that has no `posix_spawn`. Named here so the
/// exemption is visible rather than accidental — the rule below rejects every
/// other uncited ignore anywhere in a Miri suite.
const EXEMPT_IGNORE_REASONS: [&str; 1] = ["Miri cannot spawn a process"];

/// Every attribute in a source, joined onto one line each, with the line span
/// it occupied.
///
/// `rustfmt` splits a long attribute across lines, and a hand-written
/// `#[cfg_attr(\n    miri,\n    ignore = "flaky"\n)]` has no *line* that both
/// opens an attribute and mentions `ignore` — so a line-at-a-time rule reads
/// past it (rust-code-reviewer, 2026-09-09). Brackets are counted rather than
/// parsed, which is sound for the attributes this repository writes; a `[` or
/// `]` inside a reason string would end the join early, and the rules below
/// would then report the fragment rather than pass over it.
fn attribute_spans(source: &str) -> Vec<(String, usize, usize)> {
    let lines: Vec<&str> = source.lines().map(str::trim).collect();
    let mut spans: Vec<(String, usize, usize)> = Vec::new();
    let mut at: usize = 0;
    while at < lines.len() {
        if !(lines[at].starts_with("#[") || lines[at].starts_with("#![")) {
            at += 1;
            continue;
        }
        let start: usize = at;
        let mut attribute: String = String::new();
        let mut depth: isize = 0;
        while at < lines.len() {
            if !attribute.is_empty() {
                attribute.push(' ');
            }
            attribute.push_str(lines[at]);
            depth += lines[at].matches('[').count() as isize;
            depth -= lines[at].matches(']').count() as isize;
            at += 1;
            if depth <= 0 {
                break;
            }
        }
        spans.push((attribute, start, at - 1));
    }
    spans
}

/// The attributes of a source, without their spans.
fn attributes(source: &str) -> Vec<String> {
    attribute_spans(source)
        .into_iter()
        .map(|(attribute, _, _)| attribute)
        .collect()
}

/// Every `#[ignore]` in a suite must cite an issue or be exempt, not only the
/// ones a surface happens to name.
///
/// The first version audited `surface.tests` only, while the module doc claimed
/// it audited the suites — so an `#[ignore = "flaky"]` on an unlisted test passed
/// (rust-code-reviewer, 2026-09-09).
fn ignore_problems(source: &str, suite: &str) -> Vec<String> {
    attributes(source)
        .into_iter()
        .filter(|attribute| attribute.contains("ignore"))
        .filter(|attribute| !cites_an_issue(attribute) && !is_exempt_ignore(attribute))
        .map(|attribute| {
            format!(
                "{suite} carries {attribute} — an ignore must cite an issue, or a \
                 surface can be switched off with a word"
            )
        })
        .collect()
}

/// An attribute with every whitespace character removed.
///
/// Joining a split attribute leaves `#[cfg_attr( miri, ignore = ...)]`, which
/// no literal written the way a human writes it can match — so a *permitted*
/// `cfg_attr(miri, ignore)` formatted across lines was rejected by the rule
/// that exists to allow it (codex review, 2026-09-09). Both sides of every
/// comparison below are compacted, so spacing decides nothing.
fn compact(attribute: &str) -> String {
    attribute.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Whether an attribute's ignore reason is *exactly* one of the listed
/// exemptions.
///
/// Anchored to the whole reason: a substring test exempts
/// `#[ignore = "flaky; Miri cannot spawn a process"]` (rust-code-reviewer,
/// 2026-09-09).
fn is_exempt_ignore(attribute: &str) -> bool {
    let attribute: String = compact(attribute);
    EXEMPT_IGNORE_REASONS
        .iter()
        .any(|reason| attribute.contains(&compact(&format!("ignore = \"{reason}\""))))
}

/// A `cfg` on `miri` withdraws a whole suite from the interpreter while every
/// selection in the workflow still names it.
///
/// `#![cfg(not(miri))]` at the top of a file — the shape three of these suites
/// already use for `target_arch = "wasm32"` — leaves every named function
/// defined, keeps the native legs green, and makes `cargo miri test --test
/// <suite>` run zero tests and exit 0. No other rule here sees it
/// (rust-code-reviewer, 2026-09-09). The one form allowed is the per-test
/// `cfg_attr(miri, ignore = ...)`, which the ignore rules above already hold to
/// citing an issue or being listed as exempt.
fn miri_cfg_problems(source: &str, suite: &str) -> Vec<String> {
    attributes(source)
        .into_iter()
        .filter(|attribute| attribute.contains("miri"))
        .filter(|attribute| !compact(attribute).contains("cfg_attr(miri,ignore"))
        .map(|attribute| {
            format!(
                "{suite} carries {attribute} — a cfg on `miri` withdraws the suite \
                 from the interpreter while the workflow still selects it; only \
                 `cfg_attr(miri, ignore = ...)` on a single test is allowed"
            )
        })
        .collect()
}

/// Whether an `#[ignore]` attribute's reason names a GitHub issue: a `#` and at
/// least three digits.
///
/// This function knows nothing about exemptions — prose is not a tracked item, so
/// a reason that merely *sounds* principled does not satisfy it. The one
/// exemption that exists is [`EXEMPT_IGNORE_REASONS`], applied by the caller, so
/// it is a listed decision rather than a rule with a soft edge.
fn cites_an_issue(attribute: &str) -> bool {
    attribute
        .split('#')
        .skip(1)
        .any(|rest| rest.chars().take_while(char::is_ascii_digit).count().ge(&3))
}

/// Suites named `tests/miri_*.rs` that no step selects.
fn unselected_miri_suites(workflow: &str, suites: &BTreeSet<String>) -> Vec<String> {
    let job: Job = steps_of(workflow, JOB);
    let interpreting: Vec<&Step> = interpreting_steps(&job.steps);
    suites
        .iter()
        .filter(|suite| {
            // This gate itself is not interpreted: it reads the workflow and
            // the sources, which is a stable, native question. Named from the
            // build rather than spelled out, so renaming the file cannot leave
            // the rule demanding that the gate be run under Miri.
            *suite != env!("CARGO_CRATE_NAME")
                && !interpreting
                    .iter()
                    .any(|step| step.script.contains(&format!("--test {suite}")))
        })
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repository_tree_is_readable() -> bool {
    repo_root().join("tests").is_dir() && repo_root().join(WORKFLOW).is_file()
}

/// An absent tree is an environment, not a pass — but on CI it is a defect: every
/// leg that runs `cargo test --tests` has the repository checked out, so a skip
/// there means the gate silently stopped running.
fn refuse_to_skip_on_ci() {
    let on_ci: bool = std::env::var("CI")
        .map(|value| !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    assert!(
        !on_ci,
        "CI has no readable repository tree at {} — this gate must never be \
         skipped on a leg that checks the repository out",
        repo_root().display()
    );
}

fn skip_reason() -> String {
    format!(
        "SKIP: {} is not readable from {}. This gate inspects the workflow and \
         the test sources, which a packaged crate does not provide. It runs \
         wherever a workflow runs `cargo test --tests`.",
        WORKFLOW,
        repo_root().display()
    )
}

fn workflow_text() -> String {
    let path: PathBuf = repo_root().join(WORKFLOW);
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    // A CRLF checkout would defeat the mutation checks below, which splice on
    // `"\n"`-terminated fragments: the replacement would find nothing, the
    // "mutated" workflow would equal the real one, and the assertion that the
    // mutation is *reported* would fail for a reason that has nothing to do
    // with coverage (rust-code-reviewer, 2026-09-09). `.gitattributes`
    // normalises `*.sh` but not `*.yml`.
    text.replace("\r\n", "\n")
}

fn suite_text(relative: &str) -> String {
    let path: PathBuf = repo_root().join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_miri_job_selects_every_surface_criterion_1_names() {
    if !repository_tree_is_readable() {
        refuse_to_skip_on_ci();
        eprintln!("{}", skip_reason());
        return;
    }
    let workflow: String = workflow_text();
    let surfaces: Vec<&Surface> = SURFACES.iter().chain([&CONCURRENCY]).collect();
    let problems: Vec<String> = selection_problems(&workflow, &surfaces);
    assert!(
        problems.is_empty(),
        "the Miri job in {WORKFLOW} no longer covers what P4-141 criterion 1 \
         requires:\n  - {}",
        problems.join("\n  - ")
    );
}

#[test]
fn every_named_test_exists_and_at_least_one_runs_per_surface() {
    if !repository_tree_is_readable() {
        refuse_to_skip_on_ci();
        eprintln!("{}", skip_reason());
        return;
    }
    let mut problems: Vec<String> = Vec::new();
    let mut audited: BTreeSet<&str> = BTreeSet::new();
    for surface in SURFACES.iter().chain([&CONCURRENCY]) {
        let source: String = suite_text(surface.suite);
        problems.extend(suite_problems(&source, surface));
        if audited.insert(surface.suite) {
            problems.extend(ignore_problems(&source, surface.suite));
            problems.extend(miri_cfg_problems(&source, surface.suite));
        }
    }
    assert!(
        problems.is_empty(),
        "the Miri suites no longer hold what P4-141 criterion 1 maps to \
         them:\n  - {}",
        problems.join("\n  - ")
    );
}

#[test]
fn every_miri_suite_in_the_tree_is_selected_by_a_step() {
    if !repository_tree_is_readable() {
        refuse_to_skip_on_ci();
        eprintln!("{}", skip_reason());
        return;
    }
    let mut suites: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(repo_root().join("tests")).expect("tests/ must be readable") {
        let path: PathBuf = entry.expect("directory entry").path();
        let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if name.starts_with("miri_") && path.extension().is_some_and(|ext| ext == "rs") {
            suites.insert(name.to_string());
        }
    }
    assert!(
        suites.len() >= 3,
        "found {} miri_* suites in tests/, expected at least the three \
         criterion 1 landed — this check would otherwise pass over an empty set",
        suites.len()
    );

    let unselected: Vec<String> = unselected_miri_suites(&workflow_text(), &suites);
    assert!(
        unselected.is_empty(),
        "these tests/miri_*.rs suites are interpreted by no step in the {JOB} \
         job, so they compile on the native legs and prove nothing under Miri: \
         {unselected:?}"
    );
}

// ---------------------------------------------------------------------------
// The gate's own mutation check
// ---------------------------------------------------------------------------

/// A job in each of the three script shapes, with everything the rules want.
fn healthy_job() -> String {
    format!(
        r#"jobs:
  check:
    steps:
      - run: cargo check
  {JOB}:
    name: Miri
    steps:
      - uses: actions/checkout@v7
      - run: "cargo miri test --no-default-features --features std --lib -- --skip simd::"
        env:
          MIRIFLAGS: -Zmiri-disable-isolation
        timeout-minutes: 25
      - run: >
          cargo miri test -p libjpeg-turbo-rs-capi
          --test capi_create_abi_guards
          --test capi_thread_affinity
        env:
          MIRIFLAGS: -Zmiri-disable-isolation
      - run: |
          cargo miri test --no-default-features --features std \
            --test miri_public_api --test miri_alloc_failure
        env:
          MIRIFLAGS: -Zmiri-disable-isolation
      - run: >
          cargo miri test --no-default-features --features std
          --test miri_once_init
        env:
          MIRIFLAGS: -Zmiri-disable-isolation -Zmiri-many-seeds=0..8
      - run: "cargo miri test --no-default-features --features std --doc"
        env:
          MIRIFLAGS: -Zmiri-disable-isolation
  other:
    steps:
      - run: cargo test
"#
    )
}

#[test]
fn the_rules_reject_each_way_the_job_can_go_stale() {
    let surfaces: Vec<&Surface> = SURFACES.iter().chain([&CONCURRENCY]).collect();
    assert_eq!(
        selection_problems(&healthy_job(), &surfaces),
        Vec::<String>::new(),
        "the fixture is supposed to satisfy every rule; if it does not, the \
         negative cases below prove nothing"
    );

    // A dropped suite.
    let without_public_api: String = healthy_job().replace("--test miri_public_api ", "");
    assert!(
        selection_problems(&without_public_api, &surfaces)
            .iter()
            .any(|problem| problem.contains("--test miri_public_api")),
        "dropping a suite from the job must be reported"
    );

    // A dropped doctest step.
    let without_doctests: String = healthy_job().replace(
        r#"      - run: "cargo miri test --no-default-features --features std --doc""#,
        "",
    );
    assert!(
        selection_problems(&without_doctests, &surfaces)
            .iter()
            .any(|problem| problem.contains("--doc")),
        "dropping the doctest step must be reported"
    );

    // Many-seeds removed from the concurrency step.
    let single_seed: String = healthy_job().replace(" -Zmiri-many-seeds=0..8", "");
    assert!(
        selection_problems(&single_seed, &surfaces)
            .iter()
            .any(|problem| problem.contains(MANY_SEEDS)),
        "losing -Zmiri-many-seeds must be reported"
    );

    // Isolation re-enabled.
    let isolated: String = healthy_job().replace(
        "MIRIFLAGS: -Zmiri-disable-isolation\n",
        "MIRIFLAGS: -Zmiri-track-raw-pointers\n",
    );
    assert!(
        selection_problems(&isolated, &surfaces)
            .iter()
            .any(|problem| problem.contains("isolation")),
        "a step without -Zmiri-disable-isolation must be reported"
    );

    // Compiled but not interpreted: every selection name still appears.
    let compile_only: String = healthy_job().replace("cargo miri test", "cargo miri test --no-run");
    let problems: Vec<String> = selection_problems(&compile_only, &surfaces);
    assert!(
        problems.iter().any(|problem| problem.contains("--no-run")),
        "a step that compiles without interpreting must be reported, got {problems:?}"
    );

    // Switched off by a condition, command untouched.
    let disabled: String = healthy_job().replace(
        "      - run: \"cargo miri test --no-default-features --features std --doc\"",
        "      - if: false\n        run: \"cargo miri test --no-default-features --features std --doc\"",
    );
    let problems: Vec<String> = selection_problems(&disabled, &surfaces);
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("conditional on")),
        "a conditional interpreting step must be reported, got {problems:?}"
    );

    // Forgiven failure, on the step and on the job. Neither touches a command.
    let forgiven_step: String = healthy_job().replace(
        "      - run: \"cargo miri test --no-default-features --features std --doc\"",
        "      - continue-on-error: true\n        run: \"cargo miri test --no-default-features --features std --doc\"",
    );
    let problems: Vec<String> = selection_problems(&forgiven_step, &surfaces);
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("continue-on-error")),
        "a step whose failure is forgiven must be reported, got {problems:?}"
    );
    let forgiven_job: String = healthy_job().replace(
        &format!("  {JOB}:\n"),
        &format!("  {JOB}:\n    continue-on-error: true\n"),
    );
    let problems: Vec<String> = selection_problems(&forgiven_job, &surfaces);
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("job carries continue-on-error")),
        "a job whose failures are forgiven must be reported, got {problems:?}"
    );
    // `continue-on-error: false` is the default spelled out, not a withdrawal.
    let explicit_default: String = healthy_job().replace(
        "      - run: \"cargo miri test --no-default-features --features std --doc\"",
        "      - continue-on-error: false\n        run: \"cargo miri test --no-default-features --features std --doc\"",
    );
    assert_eq!(
        selection_problems(&explicit_default, &surfaces),
        Vec::<String>::new(),
        "continue-on-error: false must not be reported as a withdrawal"
    );

    // A libtest filter on a miri_* step: every selection name is still there.
    let filtered: String = healthy_job().replace(
        "          --test miri_once_init",
        "          --test miri_once_init -- --skip racing",
    );
    let problems: Vec<String> = selection_problems(&filtered, &surfaces);
    assert!(
        problems.iter().any(|problem| problem.contains("--skip")),
        "narrowing a miri_* suite with a libtest filter must be reported, got {problems:?}"
    );
    // The same withdrawal without a `--`: cargo forwards a bare word to libtest
    // as a name filter, and a name that matches nothing runs zero tests and
    // exits 0 (codex review, 2026-09-09).
    let positional: String = healthy_job().replace(
        "          --test miri_once_init",
        "          --test miri_once_init racing_but_renamed",
    );
    let problems: Vec<String> = selection_problems(&positional, &surfaces);
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("racing_but_renamed")),
        "a positional test-name filter must be reported, got {problems:?}"
    );
    // ...while the `--lib` step's own filter is required and must not be.
    assert!(
        !selection_problems(&healthy_job(), &surfaces)
            .iter()
            .any(|problem| problem.contains("libtest")),
        "the --lib step's `--skip simd::` is a required filter, not a problem"
    );

    // The whole job stripped of its interpreting steps.
    let no_miri: String = healthy_job().replace("cargo miri test", "cargo test");
    let problems: Vec<String> = selection_problems(&no_miri, &surfaces);
    assert_eq!(
        problems.len(),
        1,
        "a job with no `cargo miri test` step must report exactly the one \
         problem that subsumes the rest, got {problems:?}"
    );

    // A new suite nobody wired up.
    let mut suites: BTreeSet<String> = BTreeSet::new();
    suites.insert("miri_public_api".to_string());
    suites.insert("miri_alloc_failure".to_string());
    suites.insert("miri_once_init".to_string());
    suites.insert("miri_coverage_gate".to_string());
    assert_eq!(
        unselected_miri_suites(&healthy_job(), &suites),
        Vec::<String>::new(),
        "the four suites in the tree today are all accounted for"
    );
    suites.insert("miri_something_new".to_string());
    assert_eq!(
        unselected_miri_suites(&healthy_job(), &suites),
        vec!["miri_something_new".to_string()],
        "a suite no step selects must be reported"
    );
}

/// The fixture above is the author's idea of the job's shape; this drives the
/// same rules over the shape the repository actually has.
///
/// `phase4.md` claimed the gate was "mutation-checked against the real
/// workflow" when only the synthetic fixture was (rust-code-reviewer,
/// 2026-09-09). Three lines make the claim true.
#[test]
fn dropping_a_selection_from_the_real_workflow_is_reported() {
    if !repository_tree_is_readable() {
        refuse_to_skip_on_ci();
        eprintln!("{}", skip_reason());
        return;
    }
    let surfaces: Vec<&Surface> = SURFACES.iter().chain([&CONCURRENCY]).collect();
    let real: String = workflow_text();
    assert_eq!(
        selection_problems(&real, &surfaces),
        Vec::<String>::new(),
        "the real workflow must satisfy the rules before a mutation of it proves \
         anything"
    );

    let without_concurrency: String = real.replace("--test miri_once_init", "");
    assert!(
        selection_problems(&without_concurrency, &surfaces)
            .iter()
            .any(|problem| problem.contains("miri_once_init")),
        "dropping a selection from the real workflow must be reported"
    );

    let disabled_job: String = real.replace("  miri:\n", "  miri:\n    if: false\n");
    assert!(
        selection_problems(&disabled_job, &surfaces)
            .iter()
            .any(|problem| problem.contains("job is conditional")),
        "a job-level `if:` on the real workflow must be reported"
    );

    let forgiven_job: String = real.replace("  miri:\n", "  miri:\n    continue-on-error: true\n");
    assert!(
        selection_problems(&forgiven_job, &surfaces)
            .iter()
            .any(|problem| problem.contains("continue-on-error")),
        "forgiving the real job's failures must be reported"
    );

    let filtered: String = real.replace(
        "--test miri_once_init",
        "--test miri_once_init -- --skip racing",
    );
    assert!(
        selection_problems(&filtered, &surfaces)
            .iter()
            .any(|problem| problem.contains("libtest")),
        "narrowing a real miri_* step with a libtest filter must be reported"
    );

    let positional: String = real.replace(
        "--test miri_once_init",
        "--test miri_once_init no_such_test",
    );
    assert!(
        selection_problems(&positional, &surfaces)
            .iter()
            .any(|problem| problem.contains("no_such_test")),
        "a positional filter on the real workflow must be reported"
    );
}

#[test]
fn the_suite_rules_reject_a_missing_or_silently_ignored_test() {
    let surface: Surface = Surface {
        named: "fixture",
        suite: "tests/fixture.rs",
        selection: "--test fixture",
        tests: &["alpha", "beta"],
    };

    let healthy: &str = "#[test]\nfn alpha() {}\n\n#[test]\nfn beta() {}\n";
    assert_eq!(
        suite_problems(healthy, &surface),
        Vec::<String>::new(),
        "two runnable tests satisfy the rule"
    );

    let renamed: &str = "#[test]\nfn alpha() {}\n\n#[test]\nfn gamma() {}\n";
    assert!(
        suite_problems(renamed, &surface)
            .iter()
            .any(|problem| problem.contains("does not define beta")),
        "a renamed test must be reported, since the workflow selects by name"
    );

    let cited: &str =
        "#[test]\n#[ignore = \"P4-209 (#632): pending\"]\nfn alpha() {}\n\n#[test]\nfn beta() {}\n";
    assert_eq!(
        suite_problems(cited, &surface),
        Vec::<String>::new(),
        "an ignore that cites an issue is allowed while another test still runs"
    );

    let uncited: &str = "#[test]\n#[ignore = \"flaky\"]\nfn alpha() {}\n\n#[test]\nfn beta() {}\n";
    assert!(
        suite_problems(uncited, &surface)
            .iter()
            .any(|problem| problem.contains("without citing an issue")),
        "an ignore with no issue must be reported"
    );

    let unregistered: &str = "fn alpha() {}\n\n#[test]\nfn beta() {}\n";
    assert!(
        suite_problems(unregistered, &surface)
            .iter()
            .any(|problem| problem.contains("is not a `#[test]`")),
        "a named function that libtest does not run must be reported"
    );

    // The whole-file ignore audit, which is a different rule from the per-surface
    // one above: it sees tests no surface names.
    assert_eq!(
        ignore_problems(
            "#[test]\n#[ignore = \"P4-209 (#632)\"]\nfn cited() {}\n",
            "tests/fixture.rs"
        ),
        Vec::<String>::new(),
        "a cited ignore is allowed"
    );
    assert_eq!(
        ignore_problems(
            "#[cfg_attr(miri, ignore = \"Miri cannot spawn a process\")]\nfn oracle() {}\n",
            "tests/fixture.rs"
        ),
        Vec::<String>::new(),
        "the named C-oracle exemption is allowed"
    );
    assert_eq!(
        ignore_problems("#[ignore = \"flaky\"]\nfn quiet() {}\n", "tests/fixture.rs").len(),
        1,
        "an uncited ignore on a test no surface names must still be reported"
    );

    let all_ignored: &str =
        "#[test]\n#[ignore = \"P4-209 (#632)\"]\nfn alpha() {}\n\n#[test]\n#[ignore = \"P4-209 (#632)\"]\nfn beta() {}\n";
    assert!(
        suite_problems(all_ignored, &surface)
            .iter()
            .any(|problem| problem.contains("runs nothing for it")),
        "a surface whose every test is ignored must be reported"
    );

    // A split attribute is one attribute, not three unreadable fragments.
    let split: &str =
        "#[test]\n#[cfg_attr(\n    miri,\n    ignore = \"flaky\"\n)]\nfn alpha() {}\n\n#[test]\nfn beta() {}\n";
    assert_eq!(
        ignore_problems(split, "tests/fixture.rs").len(),
        1,
        "a rustfmt-split ignore with no issue must still be reported"
    );
    assert!(
        suite_problems(split, &surface)
            .iter()
            .any(|problem| problem.contains("without citing an issue")),
        "the per-surface rule must see a split attribute too"
    );

    // The exemption is the whole reason, not a substring of it.
    assert_eq!(
        ignore_problems(
            "#[ignore = \"flaky; Miri cannot spawn a process\"]\nfn smuggled() {}\n",
            "tests/fixture.rs"
        )
        .len(),
        1,
        "an exempt reason quoted inside a longer one must not exempt it"
    );
}

/// The `cfg`-on-`miri` rule: the one-line way to withdraw a whole suite from
/// the interpreter while the workflow still selects it.
#[test]
fn the_cfg_rule_rejects_every_way_a_suite_can_leave_the_interpreter() {
    let allowed: &str = concat!(
        "#![cfg(not(target_arch = \"wasm32\"))]\n",
        "#[cfg_attr(miri, ignore = \"Miri cannot spawn a process\")]\n",
        "#[test]\nfn oracle() {}\n",
    );
    assert_eq!(
        miri_cfg_problems(allowed, "tests/fixture.rs"),
        Vec::<String>::new(),
        "a per-test cfg_attr(miri, ignore) and a wasm guard are both allowed"
    );

    let split_but_permitted: &str = concat!(
        "#[cfg_attr(\n",
        "    miri,\n",
        "    ignore = \"P4-209 (#632): pending\"\n",
        ")]\n#[test]\nfn alpha() {}\n",
    );
    assert_eq!(
        miri_cfg_problems(split_but_permitted, "tests/fixture.rs"),
        Vec::<String>::new(),
        "a permitted cfg_attr(miri, ignore) must stay permitted when rustfmt \
         splits it (codex review, 2026-09-09)"
    );
    assert_eq!(
        ignore_problems(split_but_permitted, "tests/fixture.rs"),
        Vec::<String>::new(),
        "and its citation must still be read across the split"
    );
    let split_exempt_reason: &str = concat!(
        "#[cfg_attr(\n",
        "    miri,\n",
        "    ignore = \"Miri cannot spawn a process\"\n",
        ")]\n#[test]\nfn oracle() {}\n",
    );
    assert_eq!(
        ignore_problems(split_exempt_reason, "tests/fixture.rs"),
        Vec::<String>::new(),
        "the listed exemption must survive the split too"
    );

    for withdrawal in [
        "#![cfg(not(miri))]\n#[test]\nfn alpha() {}\n",
        "#[cfg(not(miri))]\n#[test]\nfn alpha() {}\n",
        "#[cfg(miri)]\n#[test]\nfn alpha() {}\n",
        "#[cfg_attr(\n    not(miri),\n    test\n)]\nfn alpha() {}\n",
    ] {
        assert_eq!(
            miri_cfg_problems(withdrawal, "tests/fixture.rs").len(),
            1,
            "this withdraws the suite and must be reported: {withdrawal:?}"
        );
    }

    // And the real suites are clean, which is what makes the rule live rather
    // than a fixture exercise.
    if !repository_tree_is_readable() {
        refuse_to_skip_on_ci();
        eprintln!("{}", skip_reason());
        return;
    }
    let mut audited: BTreeSet<&str> = BTreeSet::new();
    for surface in SURFACES.iter().chain([&CONCURRENCY]) {
        if audited.insert(surface.suite) {
            let source: String = suite_text(surface.suite);
            assert_eq!(
                miri_cfg_problems(&source, surface.suite),
                Vec::<String>::new(),
                "{} carries a cfg on miri",
                surface.suite
            );
        }
    }
}
