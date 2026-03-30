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
- **Use `docs/C_API_REFERENCE.md` as the definitive mapping** of every C function → Rust equivalent (✅/❌/🔶).
- After implementing a feature, update **both** docs: checkbox in FEATURE_PARITY.md and status in C_API_REFERENCE.md.
- Follow the priority roadmap at the bottom of FEATURE_PARITY.md when choosing what to work on next.

# Project Rules

- Always communicate and work in English.
- Before starting development, check if `PRD.md` exists in the project root. If it does, read and follow the requirements defined in it throughout the development process.
- **IMPORTANT: Follow Test-Driven Development (TDD).** See the **Testing (TDD)** section below for detailed rules.
- **IMPORTANT: Read and follow `METHODOLOGY.md`** before starting any task.
- When editing `CLAUDE.md`, use the minimum words and sentences needed to convey 100% of the meaning.
- Before each commit, run auto-formatting. Follow `FORMATTING.md` first; if it has no command, use the project's existing formatter; if none exists, use the language-default formatter and record the exact command in `FORMATTING.md`.
- After completing each planned task, run `cargo test` and commit before moving to the next task. **Skip tests if the change has no impact on runtime behavior** (e.g., docs, comments, CI config). Changes to runtime config files (YAML, JSON, etc. read by code) must still trigger tests.
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
- **One change at a time**: isolate each experiment to a single variable. If you change two things and perf improves, you don't know which one helped.
- **Always compare against C libjpeg-turbo**: after every performance change, run `./bench_c_decode_linux` alongside `cargo bench` and report Rust/C ratio. The goal is to match or beat C libjpeg-turbo on all benchmarks. Use `examples/bench_c_decode_linux.c` (compile: `cc -O2 -o bench_c_decode_linux examples/bench_c_decode_linux.c -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -ljpeg -Wl,-rpath,$CONDA_PREFIX/lib`).
- **Study C SIMD before porting**: when optimizing a hot path, first download and study the corresponding libjpeg-turbo C/ASM SIMD implementation. Understand the algorithm, register allocation, and data flow before writing Rust. Port the design, not just the intrinsics.

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
