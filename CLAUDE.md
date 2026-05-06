# Project Goal

Rust port of libjpeg-turbo with equivalent or better performance.

# Reference Source

- `references/libjpeg-turbo/` contains the original libjpeg-turbo C source (git submodule). Read and reference it during implementation for algorithm details, edge cases, and correctness verification.
- `references/zune-image/crates/zune-jpeg/` contains zune-jpeg, the fastest pure-Rust JPEG decoder. Reference its optimization techniques, but our goal is to outperform it.
- **x86_64 SIMD reference files** (read these before optimizing any hot path):
  - IDCT: `references/libjpeg-turbo/simd/x86_64/jidctint-sse2.asm`, `jidctint-avx2.asm`
  - Color conversion: `references/libjpeg-turbo/simd/x86_64/jdcolext-sse2.asm`, `jdcolext-avx2.asm` (core kernel, included by `jdcolor-*.asm` wrapper with per-format defines)
  - Upsample: `references/libjpeg-turbo/simd/x86_64/jdsample-sse2.asm`, `jdsample-avx2.asm`
  - Merged upsample+color: `references/libjpeg-turbo/simd/x86_64/jdmrgext-sse2.asm`, `jdmrgext-avx2.asm` (fused H2V1/H2V2 upsample + YCbCr→RGB, eliminates intermediate buffers)
  - Huffman encode: `references/libjpeg-turbo/simd/x86_64/jchuff-sse2.asm`
  - Dispatch: `references/libjpeg-turbo/simd/jsimd.h`, `simd/x86_64/jsimd.c`
  - **Pattern**: C uses wrapper+core include pattern. Wrapper (`jdcolor-avx2.asm`) includes core (`jdcolext-avx2.asm`) multiple times with different `RGB_*` defines to generate per-format variants. Replicate this in Rust using macros.

# Feature Parity Tracking

- **Read `docs/FEATURE_PARITY.md` before starting any feature work.** It is the checklist of every feature with `[x]`/`[ ]` status.
- **Read `docs/LAST_MILE.md` for the replacement-readiness gate** — slim index with current status, OPEN items, and the active Suggested Order. Phase detail lives in `docs/last_mile/phase{1,2,3}.md`; open the phase file only for the gap you are working on.
- **Use `docs/C_API_REFERENCE.md` as the definitive mapping** of every C function → Rust equivalent (✅/❌/🔶).
- After implementing a feature, update **all three** docs: checkbox in FEATURE_PARITY.md, status entry in the relevant `docs/last_mile/phaseN.md` when it closes a tracked gap (and the index in `docs/LAST_MILE.md` if the OPEN list changes), and status in C_API_REFERENCE.md.
- Follow the Suggested Order in the relevant phase file (release gate) over FEATURE_PARITY.md (feature checklist) when choosing what to work on next. Phase 3 has the live OPEN items and its order lives in the index.

## LAST_MILE Management

- **File layout:**
  - `docs/LAST_MILE.md` — slim index (~100 lines): Cold Assessment, 7-item Replacement Gate, OPEN items table, Phase Map, Phase 3 Suggested Order. Always read first.
  - `docs/last_mile/phase1.md` — historical P0-1..4, P1, Phase-1 P2, Tasks 1-7, Definition of Done. All CLOSED.
  - `docs/last_mile/phase2.md` — system-library hardening P2-1..11. All CLOSED.
  - `docs/last_mile/phase3.md` — long-tail C compatibility P3-1..6. Has live OPEN items.
  - `docs/last_mile/reference_commands.md` — common verification commands.
- **Read rule:** open the index first; open exactly one phase file for the gap you are touching. Do not load all phases by default.
- **Adding a new gap:** append the entry to the appropriate phase file (existing phase if it fits the theme; new phase only if scope is genuinely different). Then add a row to the OPEN Items table in `docs/LAST_MILE.md` with a deep link to the new section anchor. Use kebab-case anchors GitHub generates from the heading.
- **Closing a gap:**
  1. In the phase file: change the heading suffix to `— **CLOSED YYYY-MM-DD**` (or `**PARTIAL: …**` when scope shrank but didn't fully close). Add a `**Status (YYYY-MM-DD): closed.**` line citing the test/command that proves it. Strike-through the matching Suggested Order entry with `~~…~~` and append the same date.
  2. In `docs/LAST_MILE.md`: remove the row from OPEN Items if fully closed; refresh the live-checks table if the gate output changed.
  3. Update `docs/FEATURE_PARITY.md` checkbox and `docs/C_API_REFERENCE.md` status if a canonical mapping changed.
- **No archive churn:** CLOSED entries stay in their phase file as institutional memory (root cause + fix + verification). Don't move them to a separate archive — searching `git log` for the SHA that closed an entry is more useful than a flat archive.
- **Index discipline:** the slim index must stay under ~150 lines. If it grows, that's a signal to push detail back into the phase files, not to expand the index.

# Project Rules

- Always communicate and work in English.
- Before starting development, check if `PRD.md` exists in the project root. If it does, read and follow the requirements defined in it throughout the development process.
- **IMPORTANT: Follow Test-Driven Development (TDD).** See the **Testing (TDD)** section below for detailed rules.
- **IMPORTANT: Read and follow `METHODOLOGY.md`** before starting any task.
- When editing `CLAUDE.md`, use the minimum words and sentences needed to convey 100% of the meaning.
- Before each commit, run auto-formatting. Follow `FORMATTING.md` first; if it has no command, use the project's existing formatter; if none exists, use the language-default formatter and record the exact command in `FORMATTING.md`.
- After completing each planned task, run `cargo test` and commit before moving to the next task. **Skip tests if the change has no impact on runtime behavior** (e.g., docs, comments, CI config). Changes to runtime config files (YAML, JSON, etc. read by code) must still trigger tests.
- **A task is not "done" until CI/CD is green.** Local `cargo test` / `cargo clippy` / `cargo fmt --check` are necessary but not sufficient — they only cover the host platform (typically aarch64-darwin) while CI runs on `ubuntu-latest`, `macos-latest`, and `windows-latest`. Push the commits, then verify all jobs in `.github/workflows/*.yml` pass with `gh pr checks --watch <PR#>` (or `gh run watch <run-id>` for a direct push). Only after every required job is green may you tell the user the task is complete. If CI fails, fix the failure and re-verify; do **not** report completion based on local results alone.
- **After any code change (feature addition, bug fix, refactoring, PR merge), check if `README.md` needs updating.** If project description, usage, setup, architecture, or API changed, update `README.md` with clear, concise language. Keep it minimal — only document what users need to know.

## Testing (TDD)

- Write tests first. Follow Red-Green-Refactor: (1) failing test, (2) minimal code to pass, (3) refactor.
- Use real-world scenarios and realistic data in tests. Prefer actual use cases over trivial/contrived examples.
- **Never overfit to tests.** Implementation must solve the general problem, not just the specific test cases. No hardcoded returns, no input-matching conditionals, no logic that only handles test values. Use triangulation — when a fake/hardcoded implementation passes, add tests with different inputs to force generalization.
- Test behavior, not implementation. Assert on observable outcomes, not internal details — tests must survive refactoring.
- Every new feature or bug fix must have corresponding tests.
- **Optimize test execution speed.** Use `cargo test` with parallel execution (default behavior). Keep each test isolated — no shared mutable state.
- **Skip tests when no runtime impact.** Non-runtime changes (docs, README, `.md`, CI pipeline config) should not trigger test runs.

### C Cross-Validation (Mandatory)

- **Every decode/encode/transform test MUST cross-validate against C libjpeg-turbo.** Target: diff=0 (pixel-identical to C djpeg/cjpeg/jpegtran output).
- Use `djpeg` for decode comparison, `cjpeg` for encode comparison, `jpegtran` for transform comparison. Tool paths: check `/opt/homebrew/bin/` first, then `which`.
- If C tools are not available, `eprintln!("SKIP: ... not found"); return;` is acceptable — but **never** for Rust-internal failures.

### Strict Assertion Rules

- **Never log diffs without asserting.** Every computed `max_diff`/`mean_diff` MUST have a corresponding `assert!`/`assert_eq!`. Logging without asserting is a meaningless test.
- **Never silently skip on Rust errors.** `match result { Err(e) => { eprintln!("SKIP"); continue/return; } }` is **forbidden** for Rust library failures. Use `.expect()` or `.unwrap_or_else(|e| panic!(...))`. Silent skips hide bugs.
- **Never use generous tolerances as placeholders.** If the implementation is not ready, use `#[ignore = "reason with issue number"]` instead of inflating tolerance (e.g., `mean_diff < 2048` for a 0–4095 range is meaningless).
- **Tolerance must reflect measured reality + small margin**, not a guess. Measure the actual diff, then set tolerance to `actual + 1` or tighter. Document the measured value in a comment.

## Logging

- Add structured logs at key decision points, state transitions, and external calls — not every line.
- Include context: request/correlation IDs, input parameters, elapsed time, and outcome (success/failure with reason).
- Use appropriate log levels: `error!` for failures requiring action, `warn!` for recoverable issues, `info!` for business events, `debug!` for development diagnostics.
- Never log sensitive data (credentials, tokens, PII).

## Naming

- Follow Rust conventions: `snake_case` for functions, variables, modules; `CamelCase` for types and traits; `SCREAMING_SNAKE_CASE` for constants.
- Names must be self-descriptive. Avoid cryptic abbreviations (`proc`, `mgr`, `tmp`).
- Prefer clarity over brevity: `user_email` > `e`, `calculate_shipping_cost` > `calc`.
- Booleans should read as yes/no questions: `is_valid`, `has_permission`, `should_retry`.

## Types

- Prefer explicit type annotations over type inference.
- At minimum, annotate function signatures (parameters and return types).
- Use `Result<T, E>` for fallible operations. Avoid `.unwrap()` in library code.

## Comments

- Explain **why**, not what. Code already shows what it does.
- Comment business rules, workarounds, and non-obvious decisions.
- Use `///` doc comments for public API items. Use `//` for internal notes.
- Mark known limitations with `TODO(reason)` or `FIXME(reason)` — always include why.
- Delete comments when the code changes — outdated comments are worse than no comments.

## Post-Implementation Review

- **After completing any implementation task** (feature, optimization, bug fix), spawn a `code-reviewer` agent to review the changes before committing.
- The reviewer should check: missed optimizations, logic defects, SIMD correctness, unnecessary overhead, and SOLID principles.
- Apply reviewer suggestions if they are clearly beneficial, then commit.
- **For non-trivial commits** (≥ ~50 changed lines, OR touching `decode/`, `encode/`, `simd/`, public API, or fuzz harness) also run codex review before pushing — independent second opinion. Invoke the CLI directly: `codex review --commit <SHA>` for the just-created commit, or `codex review --uncommitted` for staged/unstaged changes (`--commit` and a custom prompt are mutually exclusive — pass one or the other). Trivial fixes (typos, comments, single-line tweaks) are exempt. The codex stop-review-gate hook also fires automatically at end-of-turn as a safety net, but proactively running it before push surfaces issues earlier.

## Performance Optimization (Experiment Tracking)

When optimizing performance, follow the experiment-driven workflow in `experiments/README.md`.

**Key rules:**

- **Every change must pass all tests.** Run `cargo test` after each optimization attempt. If tests fail, fix the issue or revert — never commit broken code for a benchmark win.
- **Record every attempt** in `experiments/<target>.tsv` — successes, failures, and crashes. Failures are data.
- **Per-target logs**: each optimization area (idct, huffman, color, upsample, pipeline) has its own TSV. **Only read the relevant TSV** when starting work on a target — do NOT read all experiment logs. This prevents context pollution and keeps focus.
- **Benchmark harness**: use `cargo bench` full matrix for latency comparison. Quick iteration uses `cargo bench -- decode_640x480`. Always compare against C libjpeg-turbo using `examples/bench_c_matrix` (see below).
- **Full benchmark matrix**: when reporting performance, run the full test matrix covering resolutions (64×64 → 3840×2160), subsampling (4:2:0, 4:2:2, 4:4:4), and content types (photo, graphic, checker, restart). Compare Rust vs C side-by-side for all cases.
- **Keep/discard protocol**: if benchmark improves → commit + append `keep`. If regresses → `git checkout --` to revert + append `discard` with explanation of WHY it failed. If crash → append `crash` with error summary.
- **Description must explain causality**: not "tried X" but "tried X because profiling showed Y; failed because Z" or "tried X because Y; saved N us because Z".
- **Profile before optimizing**: always `samply record` or `perf record` to identify the actual hotspot before changing code. Don't guess.
- **Stable CPU frequency for benchmarks**: before benchmarking, run `sudo bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'` and `sudo bash -c 'echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo'` to disable turbo boost and fix at base clock. This prevents measurement variance from CPU frequency scaling.
- **Run benchmarks sequentially, never in parallel.** Parallel benchmark processes compete for CPU, cache, and memory bandwidth, producing unreliable results. Run C benchmark first, then Rust benchmark (or vice versa), one at a time.
- **One change at a time**: isolate each experiment to a single variable. If you change two things and perf improves, you don't know which one helped.
- **Always compare against C libjpeg-turbo**: after every performance change, run `./bench_c_decode_linux` alongside `cargo bench` and report Rust/C ratio. The goal is to match or beat C libjpeg-turbo on all benchmarks. Use `examples/bench_c_decode_linux.c` (compile: `cc -O2 -o bench_c_decode_linux examples/bench_c_decode_linux.c -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -ljpeg -Wl,-rpath,$CONDA_PREFIX/lib`).
- **Study C SIMD before porting**: when optimizing a hot path, first download and study the corresponding libjpeg-turbo C/ASM SIMD implementation. Understand the algorithm, register allocation, and data flow before writing Rust. Port the design, not just the intrinsics.
- **WASM benchmark**: use Playwright MCP to measure real browser performance. Steps:
  1. Build: `cd crates/libjpeg-turbo-rs-wasm && wasm-pack build --release --target web`
  2. Serve: `python3 -m http.server 8079 --bind 127.0.0.1` from the WASM crate directory
  3. Open `http://127.0.0.1:8079/bench/index.html` via Playwright, click "Run Benchmark", wait for "Done!"
  4. Capture results from the page snapshot (table rows: Operation, Image, WASM ms, Native ms, Ratio, Throughput)
  5. C libjpeg-turbo comparison: Chrome's native JPEG codec is C libjpeg-turbo-based, so `createImageBitmap(blob)` time = C decode time, `canvas.toBlob('image/jpeg', quality)` time = C encode time. This is the real Rust WASM vs C libjpeg-turbo benchmark.
  6. Save results to `experiments/wasm_*.md`.

## Git Configuration

- All commits must use the local git config `user.name` and `user.email`.
- All commits must include `Signed-off-by` line (always use `git commit -s`).
- **Pre-commit hook**: `.githooks/pre-commit` runs `cargo fmt --check` and `cargo clippy --lib -- -D warnings` before every commit. Activate after clone: `git config core.hooksPath .githooks`. CI runs the same checks — local clippy (aarch64) and CI clippy (x86_64) can differ due to `#[cfg(target_arch)]` blocks, so fix warnings for all platforms.

## Branching & PR Workflow

- All changes go through pull requests. No direct commits to `main`.
- Branch naming: `<type>/<short-description>` (e.g., `feat/add-parser`, `fix/table-bug`).
- One branch = one focused unit of work.

## PR Merge Procedure

1. Rewrite PR description if empty/unclear via `gh pr edit`.
2. Cross-reference related issues. Use "Related: #N".
3. Check for conflicts. Rebase/merge if `main` has advanced.
4. Wait for CI to pass: `gh pr checks <number> --watch`.
5. Final code review via `gh pr diff <number>`.
6. Merge: `gh pr merge <number> --merge`.
7. Sync main: `git pull`.
8. Clean up branches.
