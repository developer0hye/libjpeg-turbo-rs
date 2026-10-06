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

## Status — not enrolled (re-checked 2026-10-07, P4-217)

Files in this directory are **not** evidence of enrollment: no
`projects/libjpeg-turbo-rs` exists in `google/oss-fuzz` until a maintainer
submits it. Re-checked against the current
[Rust integration guide](https://google.github.io/oss-fuzz/getting-started/new-project-guide/rust-lang/):

- [x] `FROM gcr.io/oss-fuzz-base/base-builder-rust`; `cargo fuzz build -O`.
- [x] `sanitizers: [address]`, `fuzzing_engines: [libfuzzer]` — the only
      combination OSS-Fuzz supports for Rust. (`undefined` and `memory` were
      listed until 2026-10-07.)
- [x] No build-time tool install: `cargo-fuzz` is preinstalled in the base
      image. (`build.sh` used to `cargo install` a pinned copy with `|| true`.)
- [x] Fuzz target set covers decode, encode round-trip, transform,
      progressive, and coefficient surfaces.
- [ ] **Maintainer decision:** `primary_contact` / `auto_ccs` in
      `project.yaml` name a work address; OSS-Fuzz sends crash reports there,
      so it must be the address the maintainer wants for security reports.
- [ ] Submission itself (below), and a local `helper.py build_fuzzers` run
      against the current base image, which has not been done since the files
      were written.

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
