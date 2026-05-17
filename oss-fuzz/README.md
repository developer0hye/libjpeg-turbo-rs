# OSS-Fuzz integration

This directory holds the build files needed to enroll `libjpeg-turbo-rs` in
[OSS-Fuzz](https://github.com/google/oss-fuzz). It is **not** itself part of
the OSS-Fuzz repository — the expectation is to copy
`projects/libjpeg-turbo-rs/` into `google/oss-fuzz/projects/` when submitting.

## Layout

```
oss-fuzz/
  README.md                               # this file
  projects/
    libjpeg-turbo-rs/
      Dockerfile        # base-builder-rust + git clone
      build.sh          # `cargo fuzz build` per target + copy to $OUT
      project.yaml      # OSS-Fuzz metadata (language, sanitizers, contacts)
```

## Local smoke test (from an OSS-Fuzz checkout)

```bash
# 1. Copy the project definition into an OSS-Fuzz checkout.
cp -R oss-fuzz/projects/libjpeg-turbo-rs <oss-fuzz-checkout>/projects/

# 2. Build the container and fuzzers.
cd <oss-fuzz-checkout>
python infra/helper.py build_image libjpeg-turbo-rs
python infra/helper.py build_fuzzers --sanitizer address libjpeg-turbo-rs

# 3. Run one fuzzer briefly to confirm startup.
python infra/helper.py run_fuzzer libjpeg-turbo-rs fuzz_decompress -- -max_total_time=30
```

## Status — ready for upstream submission (P4-11, 2026-05-17)

Pre-submission checklist (all green):

- [x] `primary_contact` / `auto_ccs` set to a stable maintainer email in
      `project.yaml`.
- [x] `cargo-fuzz` version pinned in `build.sh` (`0.12.0`) to match the one
      shipped in `gcr.io/oss-fuzz-base/base-builder-rust`.
- [x] Sanitizers enabled: `address`, `undefined`, `memory`.
- [x] Fuzz target set covers decode, encode round-trip, transform,
      progressive, and coefficient surfaces.

Submission steps (manual — performed by a maintainer with a `google/oss-fuzz`
clone):

1. Fork `google/oss-fuzz` and create a branch `add-libjpeg-turbo-rs`.
2. `cp -R oss-fuzz/projects/libjpeg-turbo-rs <fork>/projects/`.
3. Open a PR to `google/oss-fuzz` titled "Add libjpeg-turbo-rs project".
4. Once merged, the project appears on
   [OSS-Fuzz's introspector](https://introspector.oss-fuzz.com/) and the
   continuous fuzzing pipeline picks it up automatically.

## Complementary local coverage

The on-tree workflows below cover the FFI surface that OSS-Fuzz cannot:

- `.github/workflows/sanitizers.yml` — Rust crate exercised under ASan +
  UB-checks on every PR (Linux + macOS subset).
- `.github/workflows/fuzz-smoke.yml` — nightly 5-minute fuzz smoke over each
  target with an upstream-pinned libjpeg-turbo C oracle for the
  `fuzz_*_diff_c` differential targets (those are intentionally *not*
  enrolled in OSS-Fuzz because they need C libjpeg-turbo binaries at run
  time, which is awkward inside the OSS-Fuzz container).
