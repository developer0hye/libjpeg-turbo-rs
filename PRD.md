# Product Requirements Document: Adoption Strategy

**Status:** Active  
**Last reviewed:** 2026-08-27  
**Scope:** `libjpeg-turbo-rs` workspace, from the current `0.8.x` line through a trustworthy `1.0` release  
**Primary audience:** maintainers, contributors, downstream evaluators, package maintainers, and agentic coding tools

## 1. Executive summary

`libjpeg-turbo-rs` is not competing with an ordinary codec. The C implementation of
[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) is a mature,
widely packaged, industry-standard implementation with long-lived Application
Programming Interface (API) and Application Binary Interface (ABI) contracts,
official binaries, broad Single Instruction, Multiple Data (SIMD) coverage, and
a large installed base. Version 3.2.0, released on 2026-06-30, is the upstream
comparison baseline at the time of this document.

The product strategy is therefore not "rewrite the same code in Rust and wait
for adoption." It is to win where a Rust-native implementation can deliver a
materially better total developer experience while preserving measurable
correctness and performance:

1. become the default high-performance JPEG codec for new Rust-native projects;
2. remove C build-system and Foreign Function Interface (FFI) friction for
   cross-platform, embedded, and WebAssembly (Wasm) deployments;
3. provide a well-evidenced migration path for TurboJPEG 3 consumers;
4. expand classic libjpeg replacement only when the shipped artifact, ABI,
   lifecycle, threading, and error contracts are proved rather than assumed;
5. make every adoption claim reproducible from public tests, benchmarks, and
   release artifacts.

The long-term ambition is broader adoption than the C implementation in the
segments where Rust-native delivery is a meaningful advantage. It is not a
near-term claim that this project can replace the C implementation in every
existing operating system, application, architecture, or ABI configuration.

## 2. Source-of-truth hierarchy

Documents have different jobs. When wording conflicts, use this order:

1. [`docs/LAST_MILE.md`](docs/LAST_MILE.md) and its phase files: live
   replacement gates and exact T1-T4 readiness.
2. [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md): C ABI and SONAME
   policy.
3. [`docs/FEATURE_PARITY.md`](docs/FEATURE_PARITY.md): implemented feature
   inventory.
4. [`docs/TEST_PARITY.md`](docs/TEST_PARITY.md),
   [`docs/CORPUS_TEST_REPORT.md`](docs/CORPUS_TEST_REPORT.md), and dated
   benchmark evidence: validation status.
5. This Product Requirements Document (PRD): product direction, priorities,
   success criteria, and release requirements.
6. [`README.md`](README.md) and crate READMEs: concise user-facing summaries.

A compatibility tier may only be promoted by changing the live gate with the
required evidence. Marketing or onboarding documentation must never promote a
tier by itself.

## 3. Current product baseline

The current workspace version is `0.8.0`. The following is a concise product
snapshot, not a replacement for the live gate.

| Surface | Current position | Appropriate use now | Principal remaining risk |
| --- | --- | --- | --- |
| **T1: Rust API** | Feature-rich, C-cross-validated, competitive on measured x86_64 and aarch64 workloads | Evaluation and production use after the adopter reviews the documented limitations | No known live Undefined Behavior (UB) is recorded in the safe Rust API, but a formal memory-safety guarantee still requires the remaining layout-centralization and verification work in P4-139/P4-141. The full release-mode workspace gate is currently red because of P4-170. |
| **T2: TurboJPEG 3 C ABI** | Primary C ABI target; opaque-handle API avoids classic struct-layout risk | TurboJPEG 3 consumers whose required symbols and behavior are covered by the reference matrix | Legacy TurboJPEG 1.x/2.x aliases are partial; the release artifact and exact downstream workload still need to be evaluated by each adopter. |
| **T3: classic libjpeg v8 C ABI** | Experimental and partial | Controlled pilots with an explicit compatibility test plan and rollback | Open lifecycle, ownership, option, error, threading, artifact-validation, and downstream-coverage gaps. It is not a general system-library replacement. |
| **T4: libjpeg v6b/v7 ABI** | Explicit non-goal without separately built ABI variants | None as a drop-in replacement | Struct layouts differ. Renaming a v8-layout library to a v6b/v7 SONAME does not make it binary-compatible. |
| **Distribution** | crates.io, npm for Wasm, and Linux/macOS x86_64/aarch64 native bundles | Rust-native use and supported native pilots | No Windows native bundle; bundles are checksummed but not yet signed and do not publish a Software Bill of Materials (SBOM). |
| **Architecture performance** | Strongest on x86_64 and aarch64; Wasm SIMD128 supported | Measured target/platform combinations | AArch32/armv7 lacks a production SIMD backend; RISC-V Vector (RVV) is not implemented; scalar-only architectures need current hardware measurements. |

## 4. Problem statement

### 4.1 Why developers stay on C libjpeg-turbo

The C implementation has advantages that cannot be erased by codec feature
parity alone:

- decades of downstream testing and operational familiarity;
- broad operating-system and package-manager availability;
- stable TurboJPEG and classic libjpeg contracts;
- official, signed binaries for common platforms;
- wide SIMD coverage and platform-specific tuning;
- existing integrations in image libraries, frameworks, browsers, computer
  vision systems, and language bindings;
- established security reporting and maintenance practices.

### 4.2 Why developers would choose this project

A Rust-native implementation can remove costs that the C implementation cannot
remove without changing its architecture:

- one Cargo dependency instead of a C compiler, CMake/NASM/Yasm setup, system
  package discovery, and FFI bindings;
- Rust ownership and error types at the public API;
- `no_std + alloc` and Wasm delivery from the same codec codebase;
- caller-owned reusable output buffers and idiomatic streaming APIs;
- a direct path to Rust ecosystem integrations;
- easier cross-compilation and hermetic builds;
- the ability to validate every unsafe boundary and expose the remaining risk
  rather than hiding it behind a "pure Rust" label.

### 4.3 Adoption barriers in the current project

The implementation is ahead of its adoption experience. The main barriers are:

1. **Trust is difficult to summarize.** Detailed evidence exists, but status is
   distributed across large documents and dated experiment files.
2. **The safe choice is not obvious.** Rust API, image bridge, Wasm, TurboJPEG
   3, classic libjpeg v8, and unsupported v6b/v7 paths need a visible decision
   tree.
3. **Compatibility language has drifted.** Some user-facing text still refers
   to already-closed safety defects, while the live release-mode test failure
   is less visible.
4. **Portable performance is not always the quoted performance.** Native
   `target-cpu` results and portable release results must remain clearly
   separated.
5. **Distribution trails the C project.** Windows bundles, signatures,
   provenance, SBOMs, and package-manager integrations are incomplete.
6. **Ecosystem integration is early.** A bridge crate exists for the `image`
   crate, but adoption requires downstream pull requests, stable adapter APIs,
   and public case studies.
7. **Classic ABI replacement is expensive.** Every unproved lifecycle or
   threading behavior can invalidate a nominally correct symbol surface.

## 5. Product vision

> A production-grade JPEG codec that Rust developers can adopt with one
> dependency, evaluate with reproducible evidence, deploy across native,
> embedded, and Wasm targets, and migrate to without accepting ambiguous
> compatibility claims.

## 6. Goals

### G1. Win new Rust-native JPEG workloads

For projects that do not require an existing classic libjpeg ABI, the default
recommendation should be the Rust API rather than a binding to the C codec.

### G2. Make evaluation fast and honest

A technically competent developer should be able to determine the correct
integration path in under ten minutes and complete a representative pilot in
one working day without reading the entire replacement gate.

### G3. Match or beat the C implementation on total adoption cost

Performance is necessary but not sufficient. The product must reduce build,
cross-compilation, packaging, deployment, and maintenance complexity while
remaining competitive on throughput, latency, and memory for supported
workloads.

### G4. Establish release-level trust

Every release must publish an evidence bundle covering correctness, safety,
performance, supported platforms, exact oracle versions, artifact integrity,
and known limitations.

### G5. Create a credible migration path for C consumers

TurboJPEG 3 should be the first-class C migration target. Classic libjpeg v8
must remain explicitly experimental until its complete release gate is green.

### G6. Grow through ecosystem integrations

Adoption should come from maintained integrations in widely used Rust and
cross-platform libraries, not only direct dependencies on the core crate.

## 7. Non-goals

The following are not required to declare the Rust-native product successful:

- replacing every system `libjpeg.so.62` installation;
- claiming v6b or v7 binary compatibility from a renamed v8-layout library;
- matching upstream performance on every architecture before `1.0`;
- preserving every pre-1.0 Rust API shape if a change materially improves
  soundness or usability and is documented in the changelog;
- adding unrelated image formats to the core crate;
- optimizing obscure paths without a measured workload or adopter signal;
- publishing benchmark wins that cannot be reproduced from a documented
  environment.

## 8. Target users and jobs to be done

### 8.1 Rust application developers

**Job:** decode, encode, transform, inspect, or stream JPEG data without a C
build dependency.

**Must have:** simple one-shot API, reusable-buffer API, predictable errors,
examples, SemVer policy, documentation on memory and threading behavior.

### 8.2 Rust library and framework maintainers

**Job:** expose JPEG functionality to many downstream users without imposing a
fragile native dependency.

**Must have:** stable adapter interfaces, controlled feature flags, Minimum
Supported Rust Version (MSRV) policy, low dependency footprint, benchmark and
fuzz evidence, and a clear maintenance commitment.

### 8.3 Cross-platform, embedded, and Wasm teams

**Job:** use the same codec implementation across native, browser, WebAssembly
System Interface (WASI), and constrained `no_std + alloc` environments.

**Must have:** explicit target matrix, compile-time SIMD instructions, memory
limits, no hidden runtime dependency, and reproducible builds.

### 8.4 C/C++ TurboJPEG users

**Job:** replace or pilot `libturbojpeg` without rewriting the application.

**Must have:** symbol inventory, versioned headers, prebuilt artifacts,
behavioral parity tests, installation instructions, and rollback guidance.

### 8.5 Classic libjpeg and distribution maintainers

**Job:** determine whether a specific downstream binary can safely use the v8
shim.

**Must have:** exact ABI identity, symbol versions, lifecycle and threading
contracts, tested shipped artifacts, package metadata, and an explicit list of
unsupported configurations.

## 9. Positioning

### 9.1 Primary positioning

**Fast, evidence-driven, pure-Rust JPEG for applications that want
libjpeg-turbo-class capability without depending on a C codec.**

"Pure Rust" means the Rust-native codec does not call a C JPEG implementation.
It does not mean the repository contains no `unsafe` code. SIMD kernels and C
ABI boundaries necessarily require narrowly reviewed unsafe code.

### 9.2 Differentiators

- Cargo-native installation and cross-compilation;
- Rust-first one-shot, builder, streaming, coefficient, metadata, and
  caller-owned-buffer APIs;
- `no_std + alloc` and Wasm support;
- broad JPEG feature support, including progressive, arithmetic, lossless,
  high precision, transforms, metadata, and YUV paths;
- byte-level and behavior-level cross-validation against pinned C oracles;
- competitive measured performance on x86_64 and aarch64;
- tiered compatibility language that separates Rust API readiness from C ABI
  replacement readiness.

### 9.3 Competitive rule

A benchmark win does not compensate for a correctness, soundness, packaging,
or compatibility regression. A feature win does not compensate for an
unusable migration path. Product decisions should optimize the entire adoption
funnel.

## 10. Product principles

1. **Evidence before claims.** Every compatibility or performance statement
   links to a test, benchmark, or release artifact.
2. **Safe Rust path first.** Work reachable from the safe API has higher
   priority than expanding an experimental classic C ABI surface.
3. **One obvious default.** Common decode and encode workflows require minimal
   configuration; advanced controls remain available through builders.
4. **Portable numbers are the default numbers.** Native-CPU measurements may
   be reported, but never presented as portable release performance.
5. **Compatibility is tiered.** A green TurboJPEG 3 path does not imply a green
   classic libjpeg path.
6. **Shipped artifacts are the product.** Testing an intermediate `cargo`
   artifact is insufficient when users download a relinked or packaged binary.
7. **Unsupported is better than silently unsafe.** Reject mismatched ABI
   versions and invalid geometry instead of attempting best-effort behavior.
8. **Maintenance cost is a feature.** New APIs, architectures, and packages
   require an owner, tests, and a lifecycle plan.

## 11. Adoption funnel requirements

### 11.1 Discover

A visitor must understand within the first README screen:

- what the project is;
- why it exists despite C libjpeg-turbo;
- the current Rust API, TurboJPEG 3, classic v8, and v6b/v7 statuses;
- whether their target platform is optimized or scalar;
- where to find a quick start, migration guide, benchmark evidence, and live
  limitations.

### 11.2 Evaluate

The repository must provide:

- one representative decode and encode example;
- a decision table for each integration path;
- a reproducible benchmark command and methodology;
- dated C-oracle version pins;
- a correctness corpus summary;
- memory and reusable-buffer guidance;
- a checklist for unsupported formats, architectures, and ABIs.

### 11.3 Integrate

The project must support these explicit paths:

1. direct Rust API;
2. `image`-crate-compatible adapter;
3. Wasm/npm package;
4. TurboJPEG 3 C ABI bundle;
5. controlled classic libjpeg v8 pilot.

Each path needs installation, minimum example, error model, supported feature
matrix, platform notes, and rollback instructions where replacement is
involved.

### 11.4 Validate

A downstream pilot must be able to compare:

- decoded dimensions, color format, and pixel output;
- encoded decode-equivalence and, where relevant, byte identity;
- metadata preservation;
- latency, throughput, peak memory, and allocation behavior;
- malformed-input handling;
- concurrency behavior;
- artifact loading and symbol resolution.

### 11.5 Operate

Production adopters need:

- SemVer and MSRV policies;
- security reporting instructions;
- release notes and known limitations;
- signed or attestable artifacts and SBOMs;
- supported-version policy;
- a stable issue-triage path;
- migration instructions for breaking pre-1.0 changes.

## 12. Functional requirements

### FR1. Rust-native onboarding

- `cargo add libjpeg-turbo-rs` must lead to a compiling decode example.
- One-shot decode/encode, caller-owned-buffer, builder, streaming, transform,
  metadata, and `no_std` examples must remain compile-checked.
- Public examples must use stable APIs and explicit quality/subsampling where
  required.

**Acceptance:** documentation examples run in Continuous Integration (CI) as
`cargo test --doc` or dedicated example smoke tests.

### FR2. Integration-path decision support

- README and adoption guide must expose a consistent T1-T4 decision table.
- Unsupported v6b/v7 drop-in use must be visible before installation steps.
- Each C ABI path must link to the symbol and behavioral reference.

**Acceptance:** a documentation-drift check fails when readiness wording
contradicts the live gate's canonical status block.

### FR3. Rust ecosystem bridge

- The `libjpeg-turbo-rs-image` crate must implement the relevant `image` traits
  with documented color mappings and limitations.
- Adapter performance and compatibility must be measured separately from the
  core codec so bridge overhead is visible.
- The project should pursue an accepted optional integration in at least one
  widely used downstream crate rather than relying only on a standalone
  adapter.

**Acceptance:** at least one maintained downstream integration has automated
compatibility tests against its upstream project.

### FR4. Wasm and embedded delivery

- Wasm SIMD128 requirements must be explicit because repository-local Cargo
  configuration does not travel with a published crate.
- `no_std + alloc` builds must remain in CI.
- Memory growth and maximum input/output sizing behavior must be documented.

**Acceptance:** browser/WASI smoke tests and `thumbv7em` builds pass on every
release candidate.

### FR5. TurboJPEG 3 C ABI

- The supported TJ3 symbol set must be complete and versioned in the reference
  document.
- Missing legacy aliases must have a migration mapping or a deliberate
  compatibility decision.
- Release bundles must include headers, shared/static linkage metadata, and
  reproducible install instructions.

**Acceptance:** the shipped bundle, not only the raw Cargo cdylib, passes stock
TJ3 tools and representative downstream harnesses.

### FR6. Classic libjpeg v8 pilot path

- The project must preserve the v8 identity and reject incompatible create-time
  version/size pairs.
- Ownership, lifecycle, suspension, error, source/destination manager,
  threading, option, precision, and symbol-version contracts must each have
  executable differential coverage.
- The classic path remains experimental until every P0/P1 release gate for T3
  is closed.

**Acceptance:** T3 promotion requires a dedicated release-gate change and
public downstream evidence; documentation changes alone cannot promote it.

### FR7. Release artifacts

- Linux and macOS x86_64/aarch64 bundles remain reproducible.
- Windows x64 and Arm64 bundles, including import libraries, are required for
  broad C ABI adoption.
- Artifacts must publish provenance/signatures, checksums, and an SBOM.

**Acceptance:** a clean machine can verify, install, compile a sample, and run
it without a Rust toolchain.

## 13. Quality requirements

### QR1. Correctness

- Pinned libjpeg-turbo 3.1.x and 3.2.x oracles remain in CI where behavior may
  differ across versions.
- Every claimed feature has an end-to-end public-path test.
- Differential tests must account for every attempted case; silent skip or
  catch-all success branches are prohibited.

### QR2. Memory safety and unsafe-code discipline

- No known UB may remain reachable from the safe Rust API for `1.0`.
- All public-size and layout arithmetic must flow through centralized checked
  abstractions or an enforced equivalent.
- Every unsafe function and block must have a documented safety contract and a
  test strategy.
- Miri, AddressSanitizer, fuzzing, and code review are complementary; no single
  tool is a proof of safety.

### QR3. Performance

- Benchmarks report portable and native builds separately.
- Every result records processor, operating system, compiler, Rust toolchain,
  C oracle version, build flags, image corpus, warmup, sample count, and noise
  threshold.
- Regressions greater than 5% on a supported representative workload require
  investigation; exceptions need a documented correctness or maintainability
  justification.
- Small-image latency, large-image throughput, progressive workloads,
  transforms, reusable-buffer paths, and adapter overhead are measured
  separately.

### QR4. Portability

- x86_64 and aarch64 Linux/macOS/Windows are first-class Rust API targets.
- Wasm is first-class for browser/WASI decode/encode within documented limits.
- armv7 and RVV are not described as performance-competitive until measured on
  hardware with production SIMD paths.

### QR5. Reliability and security

- Fuzz targets cover parser, entropy, transform, metadata, high-precision, and
  C ABI entry points.
- Release candidates pass release-mode tests, sanitizers, fuzz smoke, corpus
  tests, and cross-architecture jobs.
- A private vulnerability-reporting path and supported-version policy are
  required before `1.0`.

### QR6. API stability

- `1.0` requires a documented SemVer policy for public Rust APIs and feature
  flags.
- MSRV changes occur only in minor versions and are called out in the
  changelog.
- C ABI behavior is versioned independently from Rust API SemVer where needed.

## 14. Success metrics

### 14.1 North-star metric

**Verified production workloads using a supported `libjpeg-turbo-rs`
integration path.**

A workload is verified when the adopter, integration repository, public case
study, or reproducible downstream test identifies the path and version in use.
Download counts alone do not prove adoption.

### 14.2 Trust and readiness metrics

For a `1.0` release candidate:

- zero known safe-API UB defects;
- zero red required release-mode gates for 30 consecutive days;
- 100% of public unsafe boundaries listed in an auditable inventory;
- all supported artifacts built, verified, and tested from their packaged
  form;
- portable benchmark report available for every first-class architecture;
- signed/attested artifacts plus SBOM;
- no stale readiness statement detected by the documentation gate.

### 14.3 Adoption targets

Within twelve months of `1.0`:

- at least 10 publicly identifiable production adopters across at least three
  product categories;
- at least 3 accepted or maintained downstream ecosystem integrations;
- at least 2 regular non-maintainer contributors owning a tested subsystem;
- at least 2 public migration case studies, including one C/TurboJPEG pilot;
- measurable growth in 90-day crate downloads and reverse dependencies from
  the baseline captured at the `1.0` tag.

These are product targets, not compatibility evidence. Missing an adoption
target does not justify weakening a release gate.

### 14.4 Developer-experience targets

- first successful decode from the README in under 10 minutes;
- representative Rust-native evaluation in under 1 hour;
- C ABI bundle verification and sample link in under 30 minutes on a supported
  platform;
- no more than one documentation hop from a readiness warning to the exact
  live gate item.

## 15. Milestones

### M0. Adoption documentation baseline

Deliver:

- this PRD;
- an adoption and migration guide;
- a shorter, decision-oriented README;
- a documentation index;
- contribution and pull-request evidence requirements;
- corrected safety/readiness wording across crate READMEs.

Exit criterion: a new evaluator can identify the correct integration path and
known blockers without reading every engineering tracker.

### M1. Trustworthy `0.9`

Priority work:

- close P4-170 and make the full release-mode workspace gate green;
- finish or explicitly bound P4-139 layout centralization;
- implement P4-141-style automated unsafe-boundary regression detection;
- test the exact shipped C ABI artifact in all downstream harnesses (P4-124);
- publish a reproducible portable-vs-native benchmark report;
- add documentation-drift CI.

Exit criterion: T1 and T2 have green, repeatable release evidence, with no
known safe-API UB and no required red gate.

### M2. Migration-ready `1.0`

Priority work:

- freeze and document the public Rust API and feature-flag policy;
- publish a security policy and private reporting route;
- ship Windows native artifacts;
- add signed build provenance and SBOMs;
- complete clean-machine artifact install tests;
- validate at least three external pilot workloads;
- publish the `1.0` evidence bundle and limitations.

Exit criterion: Rust-native adoption is low-friction and release artifacts are
suitable for a production dependency review.

### M3. Ecosystem expansion

Priority work:

- upstream or co-maintain integrations in major Rust image/computer-vision
  crates;
- publish framework adapters only where they have maintainers and tests;
- provide benchmark dashboards generated from reproducible result files;
- add migration case studies and adopter references;
- pursue distribution packages after T3 risk is appropriately scoped.

Exit criterion: most new users arrive through maintained ecosystem paths, not
only the root crate page.

### M4. Classic C replacement expansion

Priority work:

- eliminate the stricter cross-thread `cinfo` ownership divergence (P4-132);
- close remaining classic lifecycle, option, error, precision, and suspension
  gaps;
- test packaged libraries against representative prebuilt consumers;
- decide whether per-ABI v6b/v7 variants are worth the maintenance cost.

Exit criterion: T3 is promoted only if its complete live gate is green. T4
remains a separate product decision.

## 16. Prioritized backlog

### P0: blocks trustworthy adoption

1. Fix P4-170: release-only classic source-manager parity failures.
2. Keep the required release-mode workspace gate green on every pull request.
3. Complete the remaining checked-layout centralization and enforcement in
   P4-139.
4. Add automated detection for unsafe dispatch/boundary regressions as tracked
   by P4-141.
5. Run downstream compatibility harnesses against the exact release bundle
   (P4-124), not an intermediate library.
6. Remove stale or contradictory readiness wording automatically.
7. Establish a private security-reporting route before `1.0`.

### P1: directly increases adoption

1. Close the portable x86_64 performance gap without requiring
   `target-cpu=native` where practical (#464).
2. Produce Windows x64/Arm64 C ABI bundles.
3. Add signatures/provenance and SBOMs to every native release.
4. Remove the classic C API thread-ownership divergence (P4-132/#463).
5. Maintain a public, reproducible benchmark result format and dashboard.
6. Pursue at least one upstream `image` ecosystem integration and one
   computer-vision/media integration.
7. Publish migration case studies with rollback and benchmark results.

### P2: expands addressable platforms and workloads

1. Implement and measure AArch32 NEON if adopter demand justifies it.
2. Implement and measure RISC-V Vector support (#465).
3. Evaluate POWER and s390x optimization only with hardware and adopter
   sponsorship.
4. Add package-manager integration after artifact and compatibility gates are
   mature.
5. Consider per-ABI v6b/v7 libraries only as separate tested artifacts, never
   as SONAME aliases of the v8 layout.

## 17. Definition of done for `1.0`

`1.0` is ready only when all of the following are true:

- [ ] Full required workspace tests pass in release mode on the first-class
      matrix.
- [ ] No known UB is reachable through the safe Rust API.
- [ ] Unsafe-code inventory, contracts, Miri/sanitizer/fuzz coverage, and
      manual audit status are published.
- [ ] Public layout/size arithmetic is centralized or equivalently enforced.
- [ ] T1 and T2 claims are tested against the exact packaged artifacts.
- [ ] T3 remains explicitly experimental unless its separate gate is green.
- [ ] Portable and native performance reports are clearly separated and
      reproducible.
- [ ] Linux, macOS, and Windows first-class artifacts are available where
      applicable.
- [ ] Release artifacts have checksums, signatures or attestations, and SBOMs.
- [ ] Security policy, supported-version policy, and disclosure path exist.
- [ ] SemVer, MSRV, feature-flag, deprecation, and compatibility policies are
      documented.
- [ ] At least three independent pilot workloads have passed their own
      correctness and performance acceptance criteria.
- [ ] README, adoption guide, API docs, ABI docs, and live gate agree.

## 18. Release evidence bundle

Every release candidate should publish or link to a machine-readable and human
readable bundle containing:

- source commit and dirty-tree status;
- Rust, LLVM, C compiler, and linker versions;
- pinned C oracle versions and hashes;
- supported target matrix and feature flags;
- release-mode test summary with exact pass/fail/ignore accounting;
- fuzz and sanitizer summary;
- corpus and differential-test summary;
- portable and native benchmark results with methodology;
- artifact hashes, signatures/attestations, and SBOM;
- API/ABI compatibility changes;
- current T1-T4 status;
- known limitations and rollback instructions.

## 19. Contribution and governance rules

- Work is prioritized by P0/P1/P2 impact in this PRD and the live release gate.
- A pull request that changes readiness, performance, feature parity, or ABI
  behavior must update the corresponding source-of-truth document.
- New dependencies require a documented maintenance, security, MSRV, and
  binary-size justification.
- New architecture backends require scalar equivalence tests, feature
  detection, unsupported-CPU behavior, and hardware benchmarks.
- New public APIs require documentation, examples, tests, and a compatibility
  rationale.
- Agents and unattended issue loops must not close tracker items based only on
  code presence; acceptance criteria and evidence must be satisfied.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`.github/pull_request_template.md`](.github/pull_request_template.md).

## 20. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Overclaiming safety or drop-in compatibility | Production failures and loss of trust | Tiered status, live-gate links, documentation-drift CI, explicit unsupported paths |
| Benchmark cherry-picking | Adopters choose the codec on misleading evidence | Portable/native separation, dated matrices, raw results, representative workload categories |
| Unsafe SIMD regression | Memory corruption reachable from safe Rust | Centralized dispatch contracts, unsafe inventory, Miri/sanitizers/fuzzing, manual review |
| Maintainer overload | Stalled issues and fragile integrations | Narrow first-class matrix, subsystem ownership, contribution evidence template, reject unsupported surfaces |
| Upstream C advances faster | Performance or feature gap widens | Track current stable and previous stable oracle, currency CI, adoption-priority backlog |
| C ABI scope consumes Rust API roadmap | Core Rust experience stagnates | T1/T2 first; T3 work gated by adopter evidence and separate milestones |
| Packaging increases support burden | Broken releases across toolchains | Reproducible scripts, clean-machine tests, artifact attestations, explicit supported environments |
| Low ecosystem visibility | Strong implementation remains unused | Downstream integrations, case studies, concise README, docs index, stable bridge crates |

## 21. Open product decisions

The maintainer must eventually decide:

1. whether T3 classic libjpeg replacement is a core `1.x` commitment or a
   separately versioned compatibility product;
2. whether first-party deb/rpm/Homebrew/Windows package-manager distribution is
   sustainable;
3. which downstream integrations have committed co-maintainers;
4. whether v6b/v7 per-ABI builds justify their test and release matrix cost;
5. what support window applies to pre-1.0 and post-1.0 releases;
6. which performance architectures are first-class after x86_64/aarch64/Wasm;
7. which private security-reporting channel will be maintained long term.

## 22. References

- [libjpeg-turbo upstream repository](https://github.com/libjpeg-turbo/libjpeg-turbo)
- [libjpeg-turbo 3.2.0 release](https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/3.2.0)
- [Official libjpeg-turbo documentation](https://libjpeg-turbo.org/Documentation/Documentation)
- [Official libjpeg-turbo binary policy](https://libjpeg-turbo.org/Documentation/OfficialBinaries)
- [`docs/LAST_MILE.md`](docs/LAST_MILE.md)
- [`docs/FEATURE_PARITY.md`](docs/FEATURE_PARITY.md)
- [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md)
- [`docs/TEST_PARITY.md`](docs/TEST_PARITY.md)
- [`docs/CORPUS_TEST_REPORT.md`](docs/CORPUS_TEST_REPORT.md)
- [`docs/RELEASE_ARTIFACTS.md`](docs/RELEASE_ARTIFACTS.md)
