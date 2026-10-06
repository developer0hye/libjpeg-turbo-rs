//! P4-191 (#609): the sanitizer jobs really run the `kernel_bounds_tests`.
//!
//! The SIMD kernels Miri never interprets have direct tests in modules named
//! `kernel_bounds_tests` (`src/simd/kernel_bounds_tests.rs`,
//! `src/encode/pipeline_impl/kernel_bounds_tests.rs`). Each puts a kernel's
//! footprint flush against the end of an exact-size allocation, which is worth
//! something only where AddressSanitizer or `-Z ub-checks` is watching — the
//! `asan` and `ubsan` jobs of `.github/workflows/sanitizers.yml`. And the AVX2
//! arms run only on a CPU that reports AVX2; elsewhere they print a NOTE and
//! compare a fallback.
//!
//! None of that is visible in the test sources, so the ways it can silently
//! stop holding all live in the workflow: the `pull_request` trigger dropped
//! or narrowed by a `paths` filter, the job's instrumentation flags dropped or
//! overridden by a step-level `RUSTFLAGS`, the unfiltered `--lib` run narrowed by a filter or `--skip`, the
//! AVX2 requirement dropped, the confirmation step's count no longer asserted
//! (`cargo test` exits 0 when a filter selects nothing) or left behind when a
//! test is added or removed, or a job switched off by `if:` or forgiven by
//! `continue-on-error:`. This gate reads
//! the workflow and the test modules and fails on each of those, the same
//! shape as `tests/miri_coverage_gate.rs` for the Miri job.
//!
//! What it cannot check is that a test *asserts* anything; that is review.

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};

const WORKFLOW: &str = ".github/workflows/sanitizers.yml";
const JOBS: [&str; 2] = ["asan", "ubsan"];
/// The instrumentation each job exists for, as its `env:` line spells it.
/// Without it the job runs ordinary tests and every other rule still holds.
const INSTRUMENTATION: [(&str, &str); 2] = [
    ("asan", "RUSTFLAGS: \"-Z sanitizer=address\""),
    ("ubsan", "RUSTFLAGS: \"-Z ub-checks=yes\""),
];
/// Every token the unfiltered `--lib` run may carry before `--`.
const FULL_RUN_CARGO_ARGS: [&str; 9] = [
    "cargo",
    "+nightly",
    "test",
    "--locked",
    "--workspace",
    "--lib",
    "--no-fail-fast",
    "--target",
    "x86_64-unknown-linux-gnu",
];
/// The lines that turn the confirmation run into an assertion. `cargo test`
/// exits 0 when a filter selects nothing, so the run alone proves nothing;
/// `pipefail` keeps a failing test from hiding behind `tee`.
const COUNT_ENFORCEMENT: [&str; 4] = [
    "set -o pipefail",
    "| tee kernel_bounds.log",
    "passed=$(grep -E '^test result: ok\\.' kernel_bounds.log | awk '{s += $4} END {print s + 0}')",
    "if [ \"$passed\" -ne \"$EXPECTED_KERNEL_BOUNDS_TESTS\" ]; then",
];
const MODULE: &str = "kernel_bounds_tests";
const EXPECTED_KEY: &str = "EXPECTED_KERNEL_BOUNDS_TESTS:";
const AVX2_REQUIREMENT: &str = "grep -qw avx2 /proc/cpuinfo";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The text of one job: from its `  <job>:` line to the next two-space key.
fn job_block(workflow: &str, job: &str) -> Option<String> {
    let header: String = format!("  {job}:");
    let mut lines = workflow
        .lines()
        .skip_while(|line| line.trim_end() != header);
    let first: &str = lines.next()?;
    let mut block: String = format!("{first}\n");
    for line in lines {
        let is_next_job: bool = line.starts_with("  ")
            && !line.starts_with("   ")
            && !line.trim().is_empty()
            && !line.trim_start().starts_with('#');
        if is_next_job {
            break;
        }
        block.push_str(line);
        block.push('\n');
    }
    Some(block)
}

/// Every `cargo +nightly test` invocation in a block, continuation lines
/// joined, comments dropped, as whitespace-separated tokens.
fn cargo_test_invocations(block: &str) -> Vec<Vec<String>> {
    let mut joined: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for raw in block.lines() {
        let line: &str = raw.trim();
        if line.starts_with('#') {
            continue;
        }
        let (body, continues): (&str, bool) = match line.strip_suffix('\\') {
            Some(body) => (body, true),
            None => (line, false),
        };
        match current.as_mut() {
            Some(text) => {
                text.push(' ');
                text.push_str(body);
            }
            None if body.contains("cargo +nightly test") => {
                current = Some(body.to_string());
            }
            None => continue,
        }
        if !continues {
            joined.extend(current.take());
        }
    }
    joined.extend(current);
    joined
        .iter()
        .map(|text| {
            let start: usize = text.find("cargo +nightly test").unwrap_or(0);
            text[start..]
                .split_whitespace()
                .take_while(|token| !matches!(*token, "|" | "2>&1" | ">" | "&&" | ";"))
                .map(str::to_string)
                .collect()
        })
        .collect()
}

/// Split an invocation into (cargo arguments, libtest arguments after `--`).
fn split_at_double_dash(tokens: &[String]) -> (&[String], &[String]) {
    match tokens.iter().position(|token| token == "--") {
        Some(at) => (&tokens[..at], &tokens[at + 1..]),
        None => (tokens, &[]),
    }
}

/// Every `src/**/kernel_bounds_tests.rs` under `root`.
fn kernel_bounds_files(root: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut pending: Vec<PathBuf> = vec![root.join("src")];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for entry in entries {
            let path: PathBuf = entry.expect("directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("kernel_bounds_tests.rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// The `#[test]` functions a module declares, counted by attribute line.
fn declared_tests(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.trim() == "#[test]")
        .count()
}

/// Every way the two jobs can stop running the kernel-bounds tests under a
/// sanitizer. Pure over (workflow text, declared test count) so the rule set
/// can be driven over mutations of the real workflow below.
fn problems(workflow: &str, declared: usize) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    // The jobs gate pull requests only if the workflow triggers on them, and
    // a `paths` filter would let a pull request skip both jobs entirely.
    let trigger: &str = workflow
        .split("\non:\n")
        .nth(1)
        .and_then(|after| after.split("\n\n").next())
        .unwrap_or("");
    if !trigger.lines().any(|line| line.trim() == "pull_request:") {
        found.push(format!("{WORKFLOW} no longer triggers on `pull_request`"));
    }
    if trigger.contains("paths") {
        found.push(format!(
            "{WORKFLOW}: a `paths` / `paths-ignore` filter lets a pull request skip the sanitizers"
        ));
    }
    for job in JOBS {
        let Some(block) = job_block(workflow, job) else {
            found.push(format!("{job}: job not found in {WORKFLOW}"));
            continue;
        };
        for forbidden in ["continue-on-error:", "if:"] {
            if block
                .lines()
                .any(|line| line.trim_start().starts_with(forbidden))
            {
                found.push(format!(
                    "{job}: `{forbidden}` can switch the job off or forgive it"
                ));
            }
        }
        for (instrumented_job, flags) in INSTRUMENTATION {
            if instrumented_job == job && !block.contains(flags) {
                found.push(format!(
                    "{job}: `{flags}` is missing, so its tests run uninstrumented"
                ));
            }
        }
        // A step-level `RUSTFLAGS` (or an inline `RUSTFLAGS=` on a `run:`
        // line) overrides the job-level one for that step, so the job's line
        // must be the only place the variable is set.
        let rustflags_settings: usize = block
            .lines()
            .map(str::trim_start)
            .filter(|line| !line.starts_with('#'))
            .filter(|line| line.contains("RUSTFLAGS") && !line.contains("RUSTDOCFLAGS"))
            .count();
        if rustflags_settings != 1 {
            found.push(format!(
                "{job}: `RUSTFLAGS` is set {rustflags_settings} times; any setting but \
                 the job-level one can strip the instrumentation from a step"
            ));
        }
        for line in COUNT_ENFORCEMENT {
            if !block.contains(line) {
                found.push(format!(
                    "{job}: the confirmation step no longer asserts the count (`{line}` missing)"
                ));
            }
        }
        let exits_on_mismatch: bool = block
            .split(COUNT_ENFORCEMENT[3])
            .nth(1)
            .and_then(|after| after.split("fi").next())
            .is_some_and(|branch| branch.contains("exit 1"));
        if !exits_on_mismatch {
            found.push(format!("{job}: a count mismatch does not `exit 1`"));
        }
        if !block.contains(AVX2_REQUIREMENT) {
            found.push(format!(
                "{job}: no `{AVX2_REQUIREMENT}` step, so a runner without AVX2 \
                 compares fallbacks and passes"
            ));
        }
        let invocations: Vec<Vec<String>> = cargo_test_invocations(&block);
        if invocations
            .iter()
            .any(|tokens| tokens.iter().any(|t| t == "--no-run" || t == "--list"))
        {
            found.push(format!(
                "{job}: an invocation compiles tests without running them"
            ));
        }
        let full_lib_run: bool = invocations.iter().any(|tokens| {
            let (cargo_args, libtest_args) = split_at_double_dash(tokens);
            // An allowlist, not a denylist: a positional `TESTNAME` or any
            // other narrowing option before `--` filters as surely as one after.
            cargo_args.iter().any(|t| t == "--workspace")
                && cargo_args.iter().any(|t| t == "--lib")
                && cargo_args
                    .iter()
                    .all(|t| FULL_RUN_CARGO_ARGS.contains(&t.as_str()))
                && libtest_args.iter().all(|t| t == "--test-threads=1")
        });
        if !full_lib_run {
            found.push(format!(
                "{job}: no unfiltered `--workspace --lib` run (a filter or \
                 `--skip` after `--` narrows what the sanitizer sees)"
            ));
        }
        let confirmation: bool = invocations.iter().any(|tokens| {
            let (cargo_args, libtest_args) = split_at_double_dash(tokens);
            cargo_args.iter().any(|t| t == "--lib")
                && libtest_args.iter().any(|t| t == MODULE)
                && !libtest_args
                    .iter()
                    .any(|t| t == "--skip" || t == "--ignored")
        });
        if !confirmation {
            found.push(format!(
                "{job}: no step runs `-- {MODULE}` to confirm the tests were selected"
            ));
        }
        let expected: Vec<usize> = block
            .lines()
            .filter_map(|line| line.trim().strip_prefix(EXPECTED_KEY))
            .filter_map(|value| value.trim().trim_matches('"').parse::<usize>().ok())
            .collect();
        match expected.as_slice() {
            [count] if *count == declared => {}
            [count] => found.push(format!(
                "{job}: {EXPECTED_KEY} {count}, but the {MODULE} modules declare \
                 {declared} tests"
            )),
            _ => found.push(format!(
                "{job}: expected exactly one numeric {EXPECTED_KEY} entry, found {expected:?}"
            )),
        }
    }
    found
}

#[test]
fn the_sanitizer_jobs_run_every_kernel_bounds_test() {
    let root: PathBuf = repo_root();
    let workflow: String =
        std::fs::read_to_string(root.join(WORKFLOW)).unwrap_or_else(|e| panic!("{WORKFLOW}: {e}"));
    let files: Vec<PathBuf> = kernel_bounds_files(&root);
    assert!(
        files.len() >= 2,
        "expected the simd and encode {MODULE} modules, found {files:?}"
    );
    let mut declared: usize = 0;
    for file in &files {
        let source: String = std::fs::read_to_string(file).expect("readable test module");
        assert!(
            !source.contains("#[ignore"),
            "{}: an ignored kernel-bounds test still counts as selected",
            file.display()
        );
        declared += declared_tests(&source);
        // A module file nobody declares is never compiled, and its tests
        // never run anywhere.
        let directory: &Path = file.parent().expect("module file has a parent");
        let parent_file: PathBuf = directory.with_extension("rs");
        let parent_mod: PathBuf = directory.join("mod.rs");
        let declares: bool = [parent_file, parent_mod].iter().any(|parent| {
            std::fs::read_to_string(parent)
                .map(|text| text.contains(&format!("mod {MODULE};")))
                .unwrap_or(false)
        });
        assert!(declares, "{}: no parent module declares it", file.display());
    }
    let found: Vec<String> = problems(&workflow, declared);
    assert!(
        found.is_empty(),
        "{WORKFLOW} no longer runs the {MODULE} tests under a sanitizer:\n{}",
        found.join("\n")
    );
}

/// The gate's own mutation check: each edit below is one way the jobs can stop
/// sanitizing the kernel-bounds tests, applied to the real workflow, and each
/// must be reported.
#[test]
fn each_way_the_jobs_can_go_stale_is_reported() {
    let workflow: String =
        std::fs::read_to_string(repo_root().join(WORKFLOW)).expect("workflow readable");
    let declared: usize = kernel_bounds_files(&repo_root())
        .iter()
        .map(|file| declared_tests(&std::fs::read_to_string(file).expect("readable")))
        .sum();
    assert!(
        problems(&workflow, declared).is_empty(),
        "real workflow must pass"
    );

    let mutations: [(&str, String); 14] = [
        (
            "full run narrowed by a positional cargo filter",
            workflow.replace(
                "--no-fail-fast \\\n            -- --test-threads=1",
                "--no-fail-fast \\\n            kernel_bounds_tests -- --test-threads=1",
            ),
        ),
        (
            "pull_request trigger removed",
            workflow.replacen(
                "\non:\n  pull_request:\n",
                "\non:\n  workflow_dispatch:\n",
                1,
            ),
        ),
        (
            "paths filter added",
            workflow.replacen(
                "    branches: [main]\n",
                "    branches: [main]\n    paths: ['src/simd/**']\n",
                1,
            ),
        ),
        (
            "step-level RUSTFLAGS override",
            workflow.replacen(
                "        env:\n          EXPECTED_KERNEL_BOUNDS_TESTS",
                "        env:\n          RUSTFLAGS: \"\"\n          EXPECTED_KERNEL_BOUNDS_TESTS",
                1,
            ),
        ),
        (
            "ASan instrumentation dropped",
            workflow.replacen("RUSTFLAGS: \"-Z sanitizer=address\"", "RUSTFLAGS: \"\"", 1),
        ),
        (
            "ub-checks dropped",
            workflow.replace("RUSTFLAGS: \"-Z ub-checks=yes\"", "RUSTFLAGS: \"\""),
        ),
        (
            "count comparison removed",
            workflow.replacen(COUNT_ENFORCEMENT[3], "if false; then", 1),
        ),
        (
            "mismatch no longer fails",
            workflow.replacen(
                "expected $EXPECTED_KERNEL_BOUNDS_TESTS\" >&2\n            exit 1",
                "expected $EXPECTED_KERNEL_BOUNDS_TESTS\" >&2\n            true",
                1,
            ),
        ),
        (
            "AVX2 requirement dropped",
            workflow.replace(AVX2_REQUIREMENT, "true"),
        ),
        (
            "confirmation step's filter dropped",
            workflow.replace(
                &format!("-- {MODULE} --test-threads=1"),
                "-- --test-threads=1",
            ),
        ),
        (
            "full run narrowed by --skip",
            workflow.replace(
                "-- --test-threads=1\n",
                "-- --skip simd:: --test-threads=1\n",
            ),
        ),
        (
            "job switched off",
            workflow.replace(
                "    name: AddressSanitizer\n",
                "    name: AddressSanitizer\n    if: false\n",
            ),
        ),
        (
            "job forgiven",
            workflow.replace(
                "    name: UndefinedBehaviorSanitizer\n",
                "    name: UndefinedBehaviorSanitizer\n    continue-on-error: true\n",
            ),
        ),
        (
            "tests compiled, not run",
            workflow.replacen("--lib \\", "--lib --no-run \\", 1),
        ),
    ];
    for (what, mutated) in &mutations {
        assert_ne!(mutated, &workflow, "{what}: the mutation did not apply");
        assert!(
            !problems(mutated, declared).is_empty(),
            "{what}: not reported"
        );
    }
    assert!(
        !problems(&workflow, declared + 1).is_empty(),
        "a test added without raising {EXPECTED_KEY}: not reported"
    );
}
