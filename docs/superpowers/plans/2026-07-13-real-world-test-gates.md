# Real-World JPEG Test Gate Hardening Plan

> **For Codex:** Execute this plan test-first and keep the `v0.6.3` tag blocked until the hardening PR and its post-merge CI are green.

**Goal:** Make the existing real-world JPEG coverage regression-resistant and ensure the CI corpus actually includes nested real-world fixture sets.

**Architecture:** Preserve the current C libjpeg-turbo differential oracle, but remove Rust-failure soft skips, add minimum fixture-category assertions, copy fixture trees recursively into the corpus, and turn corpus summary counts into enforceable CI gates. Do not modify the existing fuzz corpus.

**Tech Stack:** Rust integration tests/examples, Cargo, GitHub Actions, C `djpeg`/`cjpeg`/`jpegtran`.

---

### Task 1: File the discovered gap before implementation

**Files:**
- Modify: `docs/LAST_MILE.md`
- Modify: `docs/last_mile/phase4.md`

- [ ] File P4-31 for soft-skipped Rust failures, nested real-world corpus omission, extensionless JPEG seed omission, and fail-open summary parsing.
- [ ] Commit the documentation-only filing with Signed-off-by; runtime tests are not required for this task.

### Task 2: Pin recursive and extensionless fixture inclusion with failing tests

**Files:**
- Modify: `examples/generate_corpus.rs`

- [ ] Add unit tests proving nested `.jpg`/`.jpeg` fixtures are copied while relative paths are preserved.
- [ ] Add a unit test proving extensionless files with JPEG SOI bytes are copied from the fuzz corpus, while non-JPEG files are not.
- [ ] Run the example test and observe failure with the current top-level-only copier.
- [ ] Implement the minimal recursive copier and rerun the test green.
- [ ] Assert each generated source bucket has a nonzero/minimum count.
- [ ] Run full `cargo test`, obtain independent review, then commit this task separately.

### Task 3: Make real-world decoder failures fail loudly

**Files:**
- Modify: `tests/real_world_images.rs`

- [ ] First add failing policy tests proving a Rust panic/error cannot become a successful skip.
- [ ] Add fixture inventory/category minimums (total, progressive, high-resolution, arithmetic, 12-bit, four-component/non-interleaved families).
- [ ] Remove the path that converts a Rust decoder panic into a successful skip and the path that skips a Rust arithmetic decode error after C succeeds.
- [ ] Keep skips only for an unavailable external C capability and assert the committed fixture inventory cannot silently shrink.
- [ ] Run all real-world image tests against local `djpeg`; require 61/61 pixel-identical and zero skips on the current toolchain.
- [ ] Run full `cargo test`, obtain independent review, then commit this task separately.

### Task 4: Normalize the corpus oracle and classify expected rejects

**Files:**
- Modify: `examples/corpus_test.rs`

- [ ] Add failing tests for RGB-normalized comparison of CMYK/YCCK and 12-bit inputs.
- [ ] Decode valid non-grayscale inputs to RGB on both Rust and C sides; require the six currently exposed valid fixtures to become diff-zero passes.
- [ ] Add `ExpectedReject` and an exact path + operation + reason allowlist, initially only `fuzz_repro/corrupt_huffman_65x65_422.jpg` decode with Rust `InvalidHuffmanCode` while C succeeds.
- [ ] Assert every allowlist entry is exercised and any unlisted reject/skip/failure is fatal.
- [ ] Measure encode and transform outcomes on the recursively included fixtures before defining any additional exact allowlist entries.
- [ ] Run full `cargo test`, obtain independent review, then commit this task separately.

### Task 5: Make Corpus Test fail closed in CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `examples/corpus_test.rs`

- [ ] Require zero encode failures instead of emitting a warning.
- [ ] Make the runner return nonzero for any unallowlisted failure/crash/skip or an unexercised allowlist entry; do not rely on shell fields that default missing values to zero.
- [ ] Add source-bucket minimums so missing generated, fuzz-seed, real-world, Kodak, or USC-SIPI coverage cannot remain green.
- [ ] Verify the recursive corpus contains representative `real_world/`, `kodak/`, and `usc_sipi/` paths.
- [ ] Run focused CI-script-equivalent checks plus full `cargo test`, obtain independent review, then commit this task separately.

### Task 6: PR, evidence-backed closure, and release gate

- [ ] Run `cargo fmt --all` and the complete focused suite after the task-level Red-Green commits.
- [ ] Run `cargo test --test real_world_images`, generator unit tests, corpus-runner unit tests, and `cargo test`.
- [ ] Run `codex review --commit <SHA>` for every non-trivial implementation commit before pushing.
- [ ] Commit with Signed-off-by, push, and open a focused PR.
- [ ] Keep P4-31 OPEN until the PR's full Corpus Test passes.
- [ ] After the full PR gate passes, add an evidence-backed P4-31 closure commit, rerun all required PR checks, and obtain final review.
- [ ] Wait for all PR checks, merge, sync main, and wait for post-merge CI including Corpus Test.
- [ ] Only then resume the `v0.6.3` tag/publish plan.
