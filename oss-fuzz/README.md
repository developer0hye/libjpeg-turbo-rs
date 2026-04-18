# OSS-Fuzz integration (draft)

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

## Status

Draft. Not yet submitted upstream. Before submitting:

- [ ] Replace `primary_contact` / `auto_ccs` with a stable maintainer email.
- [ ] Confirm the `cargo-fuzz` version pinned in `build.sh` matches the one
      shipped in `gcr.io/oss-fuzz-base/base-builder-rust`.
- [ ] Add a follow-up PR to OSS-Fuzz once the project is accepted for
      continuous fuzzing.
