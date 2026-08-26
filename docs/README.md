# Documentation Map

Start with the document that matches the decision you are making. The files in
this directory include user guides, compatibility contracts, release gates,
validation evidence, and maintainer work plans; they are not interchangeable.

## I want to adopt the codec

1. Read the root [`README.md`](../README.md) for the project overview and quick
   start.
2. Use [`ADOPTION_GUIDE.md`](ADOPTION_GUIDE.md) to choose the Rust API,
   `image` bridge, WebAssembly (Wasm), TurboJPEG 3, or classic libjpeg path.
3. Check the live readiness in [`LAST_MILE.md`](LAST_MILE.md) before pinning a
   release.
4. Use the evidence documents below to validate the exact feature, platform,
   artifact, and workload you plan to ship.

## I am using Rust

| Question | Document |
| --- | --- |
| How do I decode, encode, stream, transform, or reuse buffers? | Root [`README.md`](../README.md), [`examples/`](../examples), and [docs.rs](https://docs.rs/libjpeg-turbo-rs) |
| Is a JPEG feature implemented? | [`FEATURE_PARITY.md`](FEATURE_PARITY.md) |
| How is behavior checked against C libjpeg-turbo? | [`TEST_PARITY.md`](TEST_PARITY.md) and [`CORPUS_TEST_REPORT.md`](CORPUS_TEST_REPORT.md) |
| What are the current safety and release blockers? | [`LAST_MILE.md`](LAST_MILE.md) |
| How does encoding performance compare? | [`ENCODING_PERFORMANCE.md`](ENCODING_PERFORMANCE.md) and dated files under [`../experiments`](../experiments) |
| How do I use the `image` crate traits? | [`../crates/libjpeg-turbo-rs-image/README.md`](../crates/libjpeg-turbo-rs-image/README.md) |
| How do I build for Wasm? | [`../crates/libjpeg-turbo-rs-wasm/README.md`](../crates/libjpeg-turbo-rs-wasm/README.md) |

## I am using C or C++

| Question | Document |
| --- | --- |
| Should I use TurboJPEG 3 or classic libjpeg? | [`ADOPTION_GUIDE.md`](ADOPTION_GUIDE.md) |
| Which C functions are implemented, partial, or missing? | [`C_API_REFERENCE.md`](C_API_REFERENCE.md) |
| Which ABI and SONAME are safe for my consumer? | [`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md) |
| Can this replace a system library today? | [`LAST_MILE.md`](LAST_MILE.md), T1-T4 status |
| What do release bundles contain and how are they verified? | [`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md) |
| How do I build the C ABI crate? | [`../crates/libjpeg-turbo-rs-capi/README.md`](../crates/libjpeg-turbo-rs-capi/README.md) |

## I am evaluating correctness and performance

| Evidence | Purpose |
| --- | --- |
| [`FEATURE_PARITY.md`](FEATURE_PARITY.md) | Public feature inventory; a checked item must be wired end to end |
| [`TEST_PARITY.md`](TEST_PARITY.md) | Mapping between upstream C behavior/tests and this implementation |
| [`CORPUS_TEST_REPORT.md`](CORPUS_TEST_REPORT.md) | Real and generated corpus results |
| [`oracle_versions.tsv`](oracle_versions.tsv) | Pinned oracle identities used by validation jobs |
| [`ENCODING_PERFORMANCE.md`](ENCODING_PERFORMANCE.md) | Detailed encoding measurements and methodology links |
| [`../experiments`](../experiments) | Dated raw or summarized benchmark and downstream-integration evidence |
| [`LAST_MILE.md`](LAST_MILE.md) | Release-gate result and unresolved replacement gaps |

Benchmark evidence must distinguish a portable `cargo build --release` from a
machine-specific `-C target-cpu=native` build. Results are valid only for the
recorded processor, operating system, toolchain, build flags, codec versions,
and corpus.

## I want to contribute

Read these in order:

1. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — development workflow and the
   evidence expected in a pull request.
2. [`../PRD.md`](../PRD.md) — adoption goals, priorities, milestones, and the
   `1.0` definition of done.
3. [`LAST_MILE.md`](LAST_MILE.md) — live release blockers and the T1-T4 gate.
4. [`../CLAUDE.md`](../CLAUDE.md) — repository-specific implementation and
   testing rules for human and agentic contributors.
5. The relevant phase file under [`last_mile/`](last_mile/) for the issue you
   are changing.

Use [`NEXT_SESSION_PLAN.md`](NEXT_SESSION_PLAN.md) only as a current work-plan
artifact. It does not override the product requirements or compatibility gate.
Historical design and implementation plans live under [`plans/`](plans/) and
[`superpowers/`](superpowers/); they explain how earlier work was approached,
not necessarily the current public contract.

## Source-of-truth hierarchy

When documents disagree, use this order:

1. [`LAST_MILE.md`](LAST_MILE.md) and its phase files for readiness and open
   replacement gates.
2. [`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md) for C ABI and SONAME policy.
3. [`FEATURE_PARITY.md`](FEATURE_PARITY.md) for feature implementation status.
4. [`TEST_PARITY.md`](TEST_PARITY.md), [`CORPUS_TEST_REPORT.md`](CORPUS_TEST_REPORT.md),
   and dated benchmark evidence for validation claims.
5. [`../PRD.md`](../PRD.md) for product direction and priority.
6. User-facing READMEs for concise summaries.

A README or Product Requirements Document (PRD) cannot promote a compatibility
tier. Promotion requires the canonical gate, its acceptance criteria, and the
corresponding executable evidence to change together.

## Documentation maintenance rules

When a change affects a public claim:

- implementation or feature status → update `FEATURE_PARITY.md`;
- C symbol or behavior → update `C_API_REFERENCE.md`;
- ABI, SONAME, struct layout, allocator, lifecycle, or threading contract →
  update `ABI_COMPATIBILITY.md`;
- release readiness or a P0/P1 replacement blocker → update `LAST_MILE.md` and
  the relevant phase file;
- benchmark claim → add dated raw evidence and update the user-facing summary;
- release contents or verification → update `RELEASE_ARTIFACTS.md`;
- adoption priority, milestone, or success criterion → update `PRD.md`;
- public onboarding language → update the relevant README and
  `ADOPTION_GUIDE.md`.

State what was verified and what remains unverified. Do not turn code presence,
a debug-only pass, an emulated performance result, or an intermediate build
artifact into a production-readiness claim.
