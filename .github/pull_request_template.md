## Summary

<!-- What user, adopter, or maintainer problem does this solve? -->

## Product priority and gate

- PRD priority: <!-- P0 / P1 / P2 / maintenance -->
- Live-gate item or issue: <!-- P4-..., #...; write N/A only with a reason -->
- Integration path affected: <!-- Rust API / image bridge / Wasm / TurboJPEG 3 / classic libjpeg v8 / release tooling / docs -->

## Public behavior

### Before

<!-- Exact observable behavior, limitation, failure, or adoption cost. -->

### After

<!-- Exact observable behavior after this change. Do not overstate readiness. -->

## Scope

### Included

-

### Deliberately not included

-

## Evidence

| Command or artifact | Environment | Result |
| --- | --- | --- |
|  |  |  |

<!--
For benchmarks, include CPU, operating system, compiler/toolchain, C oracle
version, portable/native flags, corpus, warmup, sample count, statistic, noise
threshold, and raw output.

For C compatibility, identify the exact API/ABI version, shared library,
SONAME/install name, symbol version, and whether the packaged artifact or only
an intermediate Cargo artifact was tested.
-->

## Correctness

- [ ] A regression test exercises the documented public path.
- [ ] The result is cross-validated against a pinned C oracle when a C contract
      exists.
- [ ] Every generated/differential attempt is accounted for as pass, refusal,
      expected failure, or unexpected failure.
- [ ] Relevant malformed, truncated, oversized, short-buffer, stride, pitch,
      alignment, and lifecycle cases are covered.
- [ ] Output pixels/bytes/metadata are compared using a rule defined before the
      test.
- [ ] I have stated anything that remains unverified.

## Safety and resource handling

- [ ] No unsafe code or unsafe contract changes.
- [ ] New/changed unsafe code documents its safety contract and why safe callers
      cannot violate it.
- [ ] Size, offset, stride, allocation, and slice-span arithmetic use checked
      operations or an enforced checked abstraction.
- [ ] Scalar/SIMD equivalence and runtime feature-detection behavior are tested.
- [ ] Relevant Miri, AddressSanitizer, Undefined Behavior checks, fuzz, or
      manual-audit evidence is included.
- [ ] Memory limits, allocation behavior, and termination behavior were
      considered.

## Performance

- [ ] No performance-sensitive path changes.
- [ ] Correctness is checked before timing.
- [ ] Portable `cargo build --release` and `-C target-cpu=native` results are
      reported separately.
- [ ] Before/after measurements use the same hardware, build class, input,
      output format, and concurrency.
- [ ] Raw dated evidence is added under `experiments/`.
- [ ] Any regression above the project threshold is explained and justified.
- [ ] Emulation is used only for functional evidence, not hardware performance
      claims.

## Rust API and dependencies

- [ ] No public Rust API or Cargo feature changes.
- [ ] Public items have documentation, examples, tests, and explicit error and
      ownership behavior.
- [ ] A breaking change has migration and changelog notes.
- [ ] Minimum Supported Rust Version (MSRV), `no_std`, WebAssembly (Wasm), and
      dependency-graph impact were checked.
- [ ] A new dependency includes maintenance, security, MSRV, binary-size, and
      feature justification.

## C API and ABI

- [ ] No C Application Programming Interface (API) or Application Binary
      Interface (ABI) changes.
- [ ] Targeted TurboJPEG/libjpeg version and every changed symbol are identified.
- [ ] Ownership, allocation, callbacks, errors, suspension, abort/reuse,
      precision, and threading contracts are tested as applicable.
- [ ] SONAME/install name, symbol versions, and create-time version/size guards
      are verified.
- [ ] The exact packaged artifact users receive is tested where release behavior
      changes.
- [ ] This change does not generalize a TurboJPEG 3 result to classic libjpeg or
      a v8 result to v6b/v7.

## Platform and release artifacts

- [ ] No platform, packaging, or release-artifact changes.
- [ ] Clean-machine or isolated installation and sample compile/run are shown.
- [ ] Unsupported-CPU fallback behavior is tested.
- [ ] Native hardware evidence is included for performance claims.
- [ ] Artifact inventory, hashes, signatures/attestations, and Software Bill of
      Materials (SBOM) impact are documented.
- [ ] Rollback to the previous artifact or implementation was considered.

## Documentation

- [ ] README/onboarding remains consistent with the canonical live gate.
- [ ] `docs/FEATURE_PARITY.md` is updated for feature-status changes.
- [ ] `docs/TEST_PARITY.md` or `docs/CORPUS_TEST_REPORT.md` is updated for
      validation changes.
- [ ] `docs/C_API_REFERENCE.md` is updated for C function changes.
- [ ] `docs/ABI_COMPATIBILITY.md` is updated for ABI, SONAME, layout,
      lifecycle, allocator, or threading changes.
- [ ] `docs/LAST_MILE.md` and the relevant phase item are updated for readiness
      or blocker changes.
- [ ] `docs/RELEASE_ARTIFACTS.md` is updated for distribution changes.
- [ ] `PRD.md` is updated only when product priority, milestone, metric, or
      release requirement changes.
- [ ] Links, anchors, commands, examples, terminology, and status dates were
      checked.

## Adoption impact

<!--
How does this change lower adoption risk or cost? Examples: fixes a red gate,
reduces build friction, proves a packaged artifact, improves portable
performance, adds a maintained integration, clarifies an unsupported path.
-->

## Compatibility and rollback

<!-- What can break, how will users migrate, and how can they revert? -->

## Reviewer focus

<!-- Point reviewers to the highest-risk files, contracts, and evidence. -->
