# Contributing

Development workflow notes for `libjpeg-turbo-rs`. User-facing
documentation lives in [README.md](README.md) and on
[docs.rs](https://docs.rs/libjpeg-turbo-rs).

## Ground rules

- All changes go through pull requests; CI must be green (the workspace
  gate cross-validates against C libjpeg-turbo byte-for-byte).
- `cargo fmt --all` and `cargo clippy --lib -- -D warnings` before each
  commit (`git config core.hooksPath .githooks` installs the pre-commit
  hook).
- Tests follow TDD and must cross-validate against C `djpeg`/`cjpeg`/
  `jpegtran` where a C contract exists — see `CLAUDE.md` for the full
  testing rules, and `docs/LAST_MILE.md` for the live release gate.

## Issues worked on unattended

`scripts/issue_loop.sh` can work through the open issues without supervision,
one fresh agent per issue; `CLAUDE.md` documents how to run it. Two label
conventions come out of that, and both are meant for humans to use:

- **`autofix-skip`** — put it on an umbrella or tracker issue that closes only
  when its children do, so no agent tries to "fix" the tracker itself.
- **`autofix-blocked`** — the loop applies this after two attempts produce no
  merged pull request, and an agent applies it itself when the issue needs a
  human decision or hardware it does not have. Either way it means *read the
  issue comments*; removing the label puts the issue back in the queue.

Run reports land in `target/issue-loop-logs/` on the machine that ran the loop.

## Running sanitizers locally

Requires a nightly toolchain (`rustup install nightly`) and the `rust-src` component:

```bash
rustup component add rust-src --toolchain nightly
```

**AddressSanitizer** (detects heap overflows, use-after-free, stack overflows):

```bash
RUSTFLAGS="-Z sanitizer=address" \
LSAN_OPTIONS="suppressions=$(pwd)/lsan_suppressions.txt:detect_leaks=1" \
cargo +nightly test --workspace --lib \
  --target x86_64-unknown-linux-gnu \
  --no-fail-fast -- --test-threads=1
```

**UB checks** (detects signed integer overflow, invalid enum discriminant, misaligned pointer dereference):

```bash
RUSTFLAGS="-Z ub-checks=yes" \
cargo +nightly test --workspace --lib \
  --no-fail-fast -- --test-threads=1
```

Note: `rustc` does not implement `sanitizer=undefined`; `-Z ub-checks=yes` is the correct nightly knob for runtime UB detection.

All three sanitizer jobs (asan, ubsan, and the P4-11 C-boundary asan harness) run on every PR via `.github/workflows/sanitizers.yml`. macOS is excluded because the NEON SIMD paths produce spurious cross-thread ASan shadow-map false positives under parallel test execution.

