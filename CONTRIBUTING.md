# Contributing to libjpeg-turbo-rs

Thank you for helping build a JPEG codec that developers can adopt on evidence,
not only on feature claims.

Before changing code or public documentation, read:

1. [`PRD.md`](PRD.md) — product direction, priorities, milestones, and the
   `1.0` definition of done;
2. [`docs/README.md`](docs/README.md) — documentation map and source-of-truth
   order;
3. [`docs/LAST_MILE.md`](docs/LAST_MILE.md) — live T1-T4 release gates;
4. [`CLAUDE.md`](CLAUDE.md) — repository-specific implementation, test, and
   review rules;
5. the relevant issue and phase item under [`docs/last_mile`](docs/last_mile).

The project has broad feature coverage. The highest-value contribution is not
always another feature: release-mode correctness, safe boundary enforcement,
portable performance, packaged-artifact validation, supply-chain evidence, and
maintained ecosystem integrations have direct adoption impact.

## Ground rules

- All changes go through pull requests.
- Follow test-driven development where a behavior can be specified before its
  implementation.
- Cross-validate against pinned C libjpeg-turbo tools or libraries whenever a
  corresponding C contract exists.
- Do not convert code presence, a debug-only pass, an emulated benchmark, or an
  intermediate build artifact into a readiness claim.
- Do not promote a compatibility tier by changing a README. Promotion requires
  the canonical live gate, acceptance criteria, executable evidence, and
  user-facing summary to change together.
- Keep the scope focused. Necessary integration and documentation changes
  belong in the same pull request; unrelated refactors do not.
- New dependencies need a maintenance, security, Minimum Supported Rust Version
  (MSRV), binary-size, and `no_std` impact justification.
- New public APIs need documentation, examples, tests, an error contract, and a
  compatibility rationale.
- New architecture optimizations need scalar equivalence tests, feature
  detection, unsupported-CPU behavior, and hardware measurements.
- Record what was verified and what remains unverified.

## Choose work by adoption impact

Use the priorities in [`PRD.md`](PRD.md). In brief:

### P0 — trustworthy adoption

Examples include:

- required release-mode gate failures;
- safe-API soundness and checked layout arithmetic;
- automated unsafe-dispatch and boundary regression detection;
- validation of the exact packaged library users download;
- stale or contradictory readiness documentation;
- private security reporting and release evidence.

### P1 — lower migration and deployment cost

Examples include:

- portable performance that does not require `target-cpu=native`;
- Windows native bundles;
- signatures or attestations and a Software Bill of Materials (SBOM);
- removal of the classic C API cross-thread ownership divergence;
- reproducible benchmark results and maintained downstream integrations.

### P2 — expand the addressable platform

Examples include AArch32 NEON, RISC-V Vector (RVV), additional package-manager
support, and separately built ABI variants. These require hardware or adopter
evidence rather than speculative implementation.

## Development setup

```bash
git clone --recurse-submodules https://github.com/developer0hye/libjpeg-turbo-rs.git
cd libjpeg-turbo-rs

git config core.hooksPath .githooks
cargo build --workspace
```

The submodules under `references/` pin upstream sources used by differential
validation. Do not silently update an oracle. An oracle update requires its own
reviewable version/hash change and a report of behavioral differences.

## Local checks

Run the checks relevant to your change before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --lib -- -D warnings
cargo test --doc
```

For code changes, run the narrowest relevant test first, then the workspace
release gate when the environment supports it:

```bash
cargo test --workspace --release --no-fail-fast
```

The canonical current result is recorded in
[`docs/LAST_MILE.md`](docs/LAST_MILE.md). If the base branch has a known
failure, reproduce it on an unmodified base commit, state that clearly, and
show that your change adds no unexplained failures. A green debug run does not
replace a required release-mode run.

Useful repository entry points include:

```bash
# Root library tests
cargo test -p libjpeg-turbo-rs --release

# C ABI crate and integration tests
cargo test -p libjpeg-turbo-rs-capi --release

# image adapter
cargo test -p libjpeg-turbo-rs-image --release

# no_std core build
cargo build -p libjpeg-turbo-rs --no-default-features \
  --target thumbv7em-none-eabihf

# Compile-checked public documentation
cargo test --doc
```

Use the exact commands in the relevant live-gate item for full C parity,
architecture, Wasm, downstream, or timing-sensitive matrices.

## Evidence required by change type

### Codec behavior or feature

Include:

- a regression test through the public path;
- differential evidence against the pinned C oracle when a C contract exists;
- success/failure accounting for every generated case;
- malformed and boundary inputs relevant to the behavior;
- updates to [`docs/FEATURE_PARITY.md`](docs/FEATURE_PARITY.md),
  [`docs/TEST_PARITY.md`](docs/TEST_PARITY.md), or the live gate when their
  claims change.

A checkbox may be marked complete only when the documented public surface is
wired end to end. An internal helper, stored parameter, or nearby API does not
count as parity.

### Unsafe code, SIMD, or buffer arithmetic

Include:

- the safety contract and caller obligations;
- proof that safe callers cannot violate the contract;
- scalar equivalence and short/misaligned/edge-length tests;
- architecture feature-detection behavior;
- Miri, sanitizer, fuzz, or manual-review evidence appropriate to the risk;
- checked overflow, allocation, alignment, and slice-span handling;
- an update to the unsafe/layout inventory or live gate when applicable.

Do not describe a sanitizer pass as a proof of memory safety. The verification
program is cumulative.

### Performance

Include correctness checks before timing. Record:

- processor, operating system, and power/governor state;
- Rust/LLVM and C compiler versions;
- C oracle version;
- portable or native build flags;
- corpus hashes or deterministic generator parameters;
- output format, quality, subsampling, precision, and thread count;
- warmup, sample count, statistic, noise threshold, and raw output;
- before/after results on the same machine.

Keep portable `cargo build --release` results separate from
`-C target-cpu=native`. Emulation can prove functional execution but not target
hardware performance.

Store dated evidence under [`experiments/`](experiments) and update a
user-facing performance summary only when the evidence supports it.

### Rust API or Cargo feature

Include:

- API documentation and a compile-checked example;
- success, error, allocation/ownership, and threading semantics;
- compatibility or migration notes for a breaking change;
- MSRV, `no_std`, Wasm, and dependency-graph impact;
- changelog entry when user-visible behavior changes.

### C API or ABI

Include:

- exact symbol/signature and the targeted TurboJPEG or libjpeg version;
- create/destroy, ownership, allocation, error, callback, suspension, reuse,
  and threading behavior as applicable;
- differential tests through the produced shared library, not only a Rust
  helper;
- symbol/SONAME/install-name/version checks;
- the packaged-artifact path when release behavior is affected;
- updates to [`docs/C_API_REFERENCE.md`](docs/C_API_REFERENCE.md),
  [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md), and the live gate.

A TurboJPEG 3 success does not imply classic libjpeg readiness. A v8 result
must not be generalized to v6b or v7.

### Platform, package, or release artifact

Include:

- clean-machine or isolated-environment installation;
- artifact inventory, hashes, linkage metadata, and sample compile/run;
- native hardware evidence for performance claims;
- fallback behavior on unsupported CPUs;
- reproducible packaging path;
- updates to [`docs/RELEASE_ARTIFACTS.md`](docs/RELEASE_ARTIFACTS.md) and the
  support matrix.

Test the artifact users receive. Passing against `target/release` is not enough
when the release script relinks, renames, signs, packages, or otherwise changes
the library.

### Documentation-only change

Verify:

- links and anchors;
- code examples and commands where practical;
- consistency with the canonical live gate and ABI policy;
- terminology and status dates;
- that no compatibility or safety claim was promoted without new evidence.

## Pull request expectations

Use [`.github/pull_request_template.md`](.github/pull_request_template.md) and
include:

- the user or adopter problem;
- the selected PRD priority and live-gate item;
- the exact public behavior before and after;
- tests and raw evidence;
- supported and unverified environments;
- safety, performance, API/ABI, and documentation impact;
- rollback or compatibility notes for user-visible changes.

Keep commits reviewable. A commit should explain *why* the change exists, not
only repeat the file name.

### Developer Certificate of Origin

This repository uses a Developer Certificate of Origin (DCO) check. Sign off
each commit:

```bash
git commit -s -m "component: explain the change"
```

## Documentation source-of-truth rules

When a public claim changes, update the corresponding canonical document:

| Change | Required source of truth |
| --- | --- |
| Implemented JPEG capability | `docs/FEATURE_PARITY.md` |
| C function status | `docs/C_API_REFERENCE.md` |
| ABI, SONAME, layout, lifecycle, allocator, or threading policy | `docs/ABI_COMPATIBILITY.md` |
| Replacement readiness or open blocker | `docs/LAST_MILE.md` and the relevant phase file |
| C behavior/test mapping | `docs/TEST_PARITY.md` |
| Corpus result | `docs/CORPUS_TEST_REPORT.md` |
| Benchmark claim | dated `experiments/` evidence and the relevant summary |
| Release contents or verification | `docs/RELEASE_ARTIFACTS.md` |
| Product priority or milestone | `PRD.md` |
| Onboarding or migration | README and `docs/ADOPTION_GUIDE.md` |

Read [`docs/README.md`](docs/README.md) for the complete hierarchy. A concise
README is a summary, not the authority for a contested compatibility claim.

## Issues worked on unattended

`scripts/issue_loop.sh` can work through open issues without supervision, one
fresh agent per issue. [`CLAUDE.md`](CLAUDE.md) documents how to run it.

Two labels are also useful for human contributors:

- **`autofix-skip`** — an umbrella or tracker issue that closes only when its
  child issues close; an unattended worker must not try to "fix" the tracker.
- **`autofix-blocked`** — two attempts produced no merged pull request, or the
  issue requires a human decision, private data, hardware, signing authority,
  or another unavailable resource. Read the comments before removing it.

Run reports land in `target/issue-loop-logs/` on the machine that ran the loop.
An unattended worker must satisfy the same acceptance and evidence criteria as
a human contributor.

## Running sanitizers locally

Install a nightly toolchain and `rust-src`:

```bash
rustup install nightly
rustup component add rust-src --toolchain nightly
```

### AddressSanitizer

Detects heap overflows, use-after-free, and stack overflows:

```bash
RUSTFLAGS="-Z sanitizer=address" \
LSAN_OPTIONS="suppressions=$(pwd)/lsan_suppressions.txt:detect_leaks=1" \
cargo +nightly test --workspace --lib \
  --target x86_64-unknown-linux-gnu \
  --no-fail-fast -- --test-threads=1
```

### Rust Undefined Behavior checks

Detects supported classes such as signed integer overflow, invalid enum
discriminants, and misaligned pointer dereferences:

```bash
RUSTFLAGS="-Z ub-checks=yes" \
cargo +nightly test --workspace --lib \
  --no-fail-fast -- --test-threads=1
```

`rustc` does not implement `sanitizer=undefined`; `-Z ub-checks=yes` is the
correct nightly runtime check used here.

The sanitizer workflow also includes the C-boundary AddressSanitizer harness.
macOS is excluded from the sanitizer jobs because the NEON SIMD paths produce
spurious cross-thread AddressSanitizer shadow-map failures under parallel test
execution; that exclusion is a recorded tool limitation, not evidence that the
platform needs no safety testing.

## Review standard

A review should answer:

1. Does the public contract match the implementation?
2. Can safe Rust reach an unsafe operation without satisfying its contract?
3. Does the C behavior match the targeted upstream API/ABI version?
4. Are all generated attempts accounted for rather than silently skipped?
5. Are size, stride, pitch, alignment, multiplication, and addition checked?
6. Does runtime feature detection protect unsupported CPUs?
7. Is the performance comparison fair, reproducible, and correctness-checked?
8. Is the shipped artifact the artifact that was tested?
9. Do the documentation and live gate state exactly what remains unverified?
10. Does this change lower adoption risk or add a justified maintenance burden?

A pull request is complete when its acceptance criteria and evidence are
satisfied, not when an implementation file exists.
