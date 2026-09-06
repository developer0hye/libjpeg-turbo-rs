# Agent Rules

Before starting any task:

1. Read and follow [`CLAUDE.md`](CLAUDE.md).
2. Read [`PRD.md`](PRD.md) for product priority and acceptance criteria.
3. Use [`docs/README.md`](docs/README.md) to identify the canonical document
   for the claim you are changing.
4. Read the relevant issue, [`docs/LAST_MILE.md`](docs/LAST_MILE.md), and the
   corresponding phase item before changing replacement readiness.

## Non-negotiable rules

- Do not mark a feature complete because an internal helper exists. Verify the
  documented public path end to end.
- Do not promote T1, T2, T3, or T4 compatibility by editing a README. Update
  the canonical gate, executable evidence, and user-facing summary together.
- Do not call a release gate green from a debug-only run when release mode is
  required.
- Do not use emulation as performance evidence for physical hardware.
- Do not generalize a TurboJPEG 3 result to classic libjpeg, or a v8 result to
  v6b/v7.
- Test the artifact users receive when packaging, relinking, SONAME, symbol
  versioning, signing, or installation is involved.
- Account for every generated or differential test case. Silent catch-all
  success and unexplained skips are prohibited.
- State exactly what was verified, on which environment, and what remains
  unverified.
- Update feature, test, ABI, release, performance, and adoption documentation
  whenever the corresponding public claim changes.
- Preserve a focused scope. Do not add speculative abstraction or unrelated
  refactors.

A task is complete only when its acceptance criteria, tests, evidence, and
source-of-truth documentation agree.
