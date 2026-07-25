# Phase 4 — Post-Gate Corrections

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). Open this file for gaps surfaced after the Phase 3 release gate was marked closed.

> **ID-reuse note (2026-05-17).** The P4-2 through P4-8 ID slots in this file currently host the new replacement-tier / SONAME / panic-guard / pathological-test / wording-reconciliation / stub-sweep / FDCT-dispatch work created by PR-0-1 onwards. The original 2026-05-12..2026-05-16 closures that lived in these slots (scheduled-decode parity regression, fuzz-smoke C oracle drift, full-C-parity cjpeg padding noise, fast-DCT byte oracle noise, transform-optimized-Huffman fallback for fuzzed progressive coefficients, block-smoothing all-or-nothing component gate, NEON encode LD3 memcpy overread on tight tails) remain in git history at commits `28954f6` (P4-2), `611aea2` (P4-3), `8f7e9b0`/equivalent (P4-4..P4-6), `11a7f2e`+`d42a03b`+`9a052d1` (P4-7), and `3a75462`+`baabe37` (P4-8). Run `git log -p docs/last_mile/phase4.md` or `git log --grep="P4-[2-8]"` to recover the institutional-memory content. The renumber happened because the C-ABI replacement-tier framing surfaced as a higher-priority gap, and ID slots adjacent to the still-relevant P4-1 (`jpeg_calc_jpeg_dimensions` export) make the OPEN Items table in `docs/LAST_MILE.md` easier to scan than appending at P4-9..P4-14 would have.

## Status summary

| ID | Status |
| --- | --- |
| P4-1 | CLOSED 2026-05-10 |
| P4-2 | CLOSED 2026-05-17 |
| P4-3 | CLOSED 2026-05-17 |
| P4-4 | CLOSED 2026-05-17 |
| P4-5 | CLOSED 2026-05-17 |
| P4-6 | CLOSED 2026-05-17 |
| P4-7 | CLOSED 2026-05-17 |
| P4-8 | CLOSED 2026-05-17 |
| P4-9 | CLOSED 2026-05-17 |
| P4-10 | CLOSED 2026-05-17 |
| P4-11 | CLOSED 2026-05-17 |
| P4-12 | CLOSED 2026-05-17 |
| P4-13 | PARTIAL (incremental consume_input suspension landed + byte-exact-proven; deeper streaming-contract fidelity → P4-26) |
| P4-14 | OPEN (filed 2026-05-18) |
| P4-15 | CLOSED 2026-05-18 |
| P4-16 | CLOSED 2026-05-19 (Option B: documented in ABI_COMPATIBILITY.md) |
| P4-17 | CLOSED 2026-06-02 (real-suspension test delivered with P4-13) |
| P4-18 | CLOSED 2026-05-19 (Option B: deprecate-with-rationale, migration matrix in ABI_COMPATIBILITY.md) |
| P4-19 | CLOSED 2026-05-30 (IDCT `islow` i16-overflow AC-all-zero shortcut → scalar; full-path SSE2 residue refiled as P4-20) |
| P4-20 | OPEN (filed 2026-05-30) |
| P4-21 | OPEN (filed 2026-05-30) |
| P4-22 | CLOSED 2026-05-31 (non-interleaved baseline plane init 0→128) |
| P4-23 | CLOSED 2026-05-31 (lenient recovery in non-interleaved baseline path) |
| P4-24 | CLOSED 2026-06-01 (arithmetic multi-scan support: non-interleaved + partial-interleaved) |
| P4-25 | OPEN (filed 2026-06-01; P4-24 review) |
| P4-26 | OPEN (filed 2026-06-02; P4-13 codex round-8 review) |
| P4-27 | CLOSED 2026-06-29 (single-component baseline h1v4 scans use one-block raster semantics) |
| P4-28 | CLOSED 2026-06-29 (progressive AC-refine one-past-Se coefficient placement) |
| P4-29 | CLOSED 2026-07-08 (block smoothing clamps at the real block grid, not the iMCU-padded one) |
| P4-30 | CLOSED 2026-07-12 (unsupported 12-bit sampling layouts return an error instead of writing past component planes) |
| P4-31 | CLOSED 2026-07-25 (PR #307 hardened the gates; post-merge Corpus Test CI green) |
| P4-32 | CLOSED AS DUPLICATE OF P4-20 2026-07-13 (coefficients/quantization match C; divergence is IDCT overflow fidelity) |

---

## P4-1. `jpeg_calc_jpeg_dimensions` Was Documented But Not Exported — **CLOSED 2026-05-10**

**Status (2026-05-10): closed.** `jpeg_calc_jpeg_dimensions` is now exported from `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`, re-exported from `src/lib.rs`, and removed from `tests/symbol_inventory.rs::allowlisted_missing_symbols()`.

**Root cause:** the C API reference and feature checklist marked the helper as supported, but the actual cdylib still had it in the missing-symbol allowlist. A new dlopen regression first failed with `symbol not found`, then passed after the implementation.

**Implementation:** `jpeg_calc_jpeg_dimensions(cinfo)` mirrors the upstream no-compression-scaling behavior in `references/libjpeg-turbo/src/jcmaster.c`: `jpeg_width` / `jpeg_height` are copied from `image_width` / `image_height`; `min_DCT_{h,v}_scaled_size` is set to 8 for lossy JPEG and 1 for lossless JPEG. `jpeg_start_compress` now uses the same helper path for its derived compression fields.

**Verification:**

- `cargo test -p libjpeg-turbo-rs-capi --test capi_jpeglib_encode c2_1_calc_jpeg_dimensions_sets_public_compress_fields -- --nocapture` → passed.
- `cargo test -p libjpeg-turbo-rs-capi --test symbol_inventory --release -- --nocapture` → passed; both upstream `jpeglib.h` and `turbojpeg.h` symbol inventories resolve.

## P4-2. Replacement-Tier Framing (T1–T4) — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** Two external static-analysis reviews (2026-05-17) both flagged that `README.md`, `docs/LAST_MILE.md`, `docs/ABI_COMPATIBILITY.md`, and `docs/FEATURE_PARITY.md` mixed four distinct "replacement-ready" claims into one — Rust-crate replacement, TurboJPEG cdylib replacement, classic libjpeg v8 cdylib replacement, and system v6b/v7 drop-in. The wording let readers conflate "Rust crate ready" with "/usr/lib/libjpeg.so.62 safe to replace", which the ABI documentation itself calls undefined behaviour.

**Root cause:** documentation evolved per-feature, never per-consumer-surface, so the readiness statement was always whole-project.

**Implementation:** `README.md` now opens with a four-row "Replacement tiers" table (T1 Rust crate, T2 TurboJPEG cdylib, T3 classic libjpeg v8 cdylib, T4 v6b/v7 system drop-in). `docs/LAST_MILE.md` "Current Status" replaces the single "replacement-ready" sentence with a per-tier readiness statement. `docs/ABI_COMPATIBILITY.md` annotates the safe-SONAME matrix with explicit "default" / "opt-in" / "non-goal" tags. T4 is documented as an explicit non-goal until per-ABI cdylib matrix ships (tracked separately as P2-A in the roadmap at `/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md`).

**Verification:**

- `grep -c "T[1-4]\\." README.md docs/LAST_MILE.md docs/ABI_COMPATIBILITY.md` → ≥1 hit per file.
- Reviewer confirms the four tiers are not conflated in any surface document.

## P4-3. Default C-ABI SONAME Flip to `libjpeg.so.8` — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** `crates/libjpeg-turbo-rs-capi/build.rs` previously defaulted `CAPI_SONAME` to `libjpeg.so.62` (and `CAPI_INSTALL_NAME` to `@rpath/libjpeg.62.dylib`) while the shim's struct layout is `JPEG_LIB_VERSION = 80`. `docs/ABI_COMPATIBILITY.md` already documented this combination as undefined behaviour for v6b-compiled consumers (the v8 layout can write to v6b-unknown field offsets), and the build emitted a `cargo:warning=` describing the risk. The safe default was hidden behind an opt-in env (`CAPI_SONAME=libjpeg.so.8`).

**Root cause:** historical ease-of-distro-replacement: v6b is the most-shipped SONAME, so defaulting to it minimized integration churn for downstreams that happened to only read v6b-shape fields. But the default footgun was wider than the convenience win.

**Implementation:**

- `crates/libjpeg-turbo-rs-capi/build.rs`: default `CAPI_SONAME` → `libjpeg.so.8`; default `CAPI_INSTALL_NAME` → `@rpath/libjpeg.8.dylib`. Inverted the `cargo:warning=` logic: warning now fires when v6b is *explicitly* requested without `CAPI_ACK_V6B_SONAME=1`. Per Codex stop-time review, `CAPI_ACK_V6B_SONAME=1` is now a single-env opt-in: when set without explicit `CAPI_SONAME` / `CAPI_INSTALL_NAME`, the script auto-derives both to v6b (`libjpeg.so.62` + `@rpath/libjpeg.62.dylib`) so the Linux SONAME and macOS install_name stay in lockstep. A new mismatch-warning fires when `CAPI_SONAME` and `CAPI_INSTALL_NAME` disagree on v6b vs v8 (which would silently break dyld resolution on macOS).
- `crates/libjpeg-turbo-rs-capi/Cargo.toml`: crate `description` updated from "drop-in replacement for libjpeg.so.62" → "drop-in replacement for libjpeg.so.8".
- `crates/libjpeg-turbo-rs-capi/tests/soname.rs`: `cdylib_advertises_libjpeg_compatible_install_name_on_macos` now asserts `libjpeg.8.dylib`; `cdylib_advertises_libjpeg_compatible_soname_on_linux` now asserts `libjpeg.so.8`.
- `scripts/install_capi.sh`: per Codex stop-time review, default `DEFAULT_LIBJPEG_MAJOR` flipped to `libjpeg.so.8` / `libjpeg.8.dylib`; doc header rewritten to advertise v8 as the default layout. Added a binary-identity patch step after install: on macOS, `install_name_tool -id @rpath/${MAJOR}`; on Linux, `patchelf --set-soname ${MAJOR}`. Without these, `--soname libjpeg.so.62` would stage a v6b symlink chain but leave the cdylib advertising the build-time identity (`libjpeg.so.8` / `@rpath/libjpeg.8.dylib`), so dyld would refuse to load under the v6b name on macOS and `ld` would silently record the wrong DT_NEEDED entry on Linux. Falls back to a loud warning if `install_name_tool`/`patchelf` is not on PATH.
- `crates/libjpeg-turbo-rs-capi/tests/install_layout.rs`: `install_capi_sh_produces_complete_layout` now asserts the v8 default symlink chain (`libjpeg.so.8` / `libjpeg.8.dylib`) AND the staged cdylib's binary identity (`otool -D` install_name on macOS / `readelf -d` DT_SONAME on Linux) matches v8. `install_capi_sh_honors_soname_override` was inverted to drive the *v6b* opt-in path (`--soname libjpeg.so.62`), asserts the v8 default is NOT staged in parallel, and asserts the staged cdylib identity advertises the v6b name (when `install_name_tool`/`patchelf` is available).
- `docs/ABI_COMPATIBILITY.md`: TL;DR, safe-SONAME matrix, "libjpeg.so.62 opt-in path" section, and verification commands all flipped to reflect the new default. Opt-in section rewritten to lead with the single `CAPI_ACK_V6B_SONAME=1` env.
- `README.md`: v6b opt-in instructions updated to the single-env form.

**Verification:**

- `cargo build -p libjpeg-turbo-rs-capi --release` (no env overrides) — no `cargo:warning=` line.
- `otool -D target/release/liblibjpeg_turbo_rs_capi.dylib` → `@rpath/libjpeg.8.dylib`.
- `CAPI_ACK_V6B_SONAME=1 cargo build -p libjpeg-turbo-rs-capi --release` — no warning; `otool -D` → `@rpath/libjpeg.62.dylib` (single-env opt-in confirmed).
- `CAPI_SONAME=libjpeg.so.62 CAPI_INSTALL_NAME=@rpath/libjpeg.8.dylib cargo build …` → emits the mismatch warning.
- `cargo test -p libjpeg-turbo-rs-capi --test soname --release` → 1 passed.
- `cargo test -p libjpeg-turbo-rs-capi --test install_layout --release` → 2 passed (v8 default + v6b override, both with binary-identity assertions via `otool -D` / `readelf -d`).

## P4-4. Panic Guard on Every C-ABI Entry Point — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** All `pub extern "C" fn` bodies across the capi crate (14 modules / 154 fns including the 82 in 8,300-line `jpeglib.rs`; count re-verified 2026-05-18 with strict regex `^\s*pub extern "C" fn`) are wrapped in `crate::unwind_guard!`, ensuring a Rust panic in any FFI entry point converts to the documented C-style sentinel instead of unwinding across the FFI boundary (which is undefined behaviour). The macro is defined once in `crates/libjpeg-turbo-rs-capi/src/lib.rs` (`#[macro_export] #[doc(hidden)]`) and reused throughout. `tests/capi_panic_safety.rs` covers int / pointer / unit sentinels and confirms the catch path executes. Full verification suite (panic_safety + tjunittest + capi_jpeglib_decode + capi_jpeglib_encode + capi_classic_lifecycle) green at 2026-05-17.

Original (PARTIAL) status preserved below:

**Motivation.** Rust `panic!` unwinding across an `extern "C"` boundary is undefined behaviour. The capi crate had zero `catch_unwind` calls before this work; the only protection was `error_exit` calling `std::process::abort` from inside the error-manager path. A library-internal `panic!` (e.g. an arithmetic overflow on a malformed input that the decoder did not pre-validate) from any of the ~14 capi modules would unwind straight into the calling C frame.

**Phase-1 status (2026-05-17, partial closure):**

- `crates/libjpeg-turbo-rs-capi/src/lib.rs` declares the `unwind_guard!` macro (`#[macro_export] #[doc(hidden)]`) wrapping a body in `std::panic::catch_unwind` + `AssertUnwindSafe`, emitting a one-line stderr message and returning the caller-supplied sentinel on caught panic.
- `crates/libjpeg-turbo-rs-capi/src/alloc.rs`: `tj3Alloc`, `tj3Free` wrapped.
- `crates/libjpeg-turbo-rs-capi/src/bufsize.rs`: `tj3JPEGBufSize`, `tj3YUVBufSize`, `tj3YUVPlaneSize`, `tj3YUVPlaneWidth`, `tj3YUVPlaneHeight`, `tj3GetScalingFactors` wrapped.
- New `crates/libjpeg-turbo-rs-capi/tests/capi_panic_safety.rs` (6 tests) proves the macro returns the int / pointer / unit sentinels on caught panic and the real value on happy path.

**ABI-strategy note (deferred).** The original plan called for `[profile.release] panic = "abort"` on the capi crate only; in stable Cargo, profile-`panic` can only be set at workspace root, which would force the main `libjpeg_turbo_rs` Rust crate to abort on panic too (breaking `Result<_, JpegError>` recovery for Rust API consumers). The `catch_unwind` macro is sufficient on its own — a caught panic does NOT cross the FFI boundary. The `panic = "abort"` belt-and-suspenders is therefore explicitly out of scope.

**Remaining acceptance criteria (still OPEN):**

- Apply the macro to every `pub extern "C" fn` in `compress.rs`, `decompress.rs`, `convert.rs`, `header.rs`, `imageio.rs`, `memmgr.rs`, `mozjpeg_compat.rs`, `legacy.rs`, `precision.rs`, `transform.rs`, `yuv.rs`, `tj3.rs`, and `jpeglib.rs` (the 8,300-line bulk).
- Extend `tests/capi_panic_safety.rs` with at least one panic-inducing case per wrapped module so the suite catches a future regression where a contributor forgets to wrap a new entry point.

**Why phase-1 stopped here.** alloc + bufsize have the simplest entry-point signatures and serve as the macro's proof of correctness without overloading any single PR; jpeglib.rs alone is 8,300 lines and warrants its own focused branch (`feat/capi-panic-boundary-jpeglib`).

## P4-5. Classic libjpeg State-Machine Pathological Coverage — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs` covers three pathological patterns end-to-end:

1. `source_mgr_suspends_every_byte` — custom `jpeg_source_mgr` releases one byte per `fill_input_buffer` call so the decoder walks every byte through the suspension state machine; harness asserts refill-call count meets a `>= byte_count / 2` floor. **Caveat (2026-05-18, per P4-17):** `slow_fill` returns `TRUE` after copying one byte, so this pattern exercises *single-byte chunked refill*, **not** the `JPEG_SUSPENDED` state machine. A real-suspension pattern (where `fill_input_buffer` returns `FALSE` until refilled) is tracked under P4-17.
2. `dest_mgr_rejects_first_flush` — custom `jpeg_destination_mgr` returns FALSE from `empty_output_buffer`; the harness installs a setjmp/longjmp `error_exit` and asserts `JERR_CANT_SUSPEND` (msg_code 25) fires (either eagerly on the first write, or lazily after the FALSE return — both paths are safe and catchable).
3. `save_markers_truncates_multichunk_icc` — a JPEG with two APP2 ICC chunks is read with `jpeg_save_markers(length_limit=1)`; harness asserts the marker_list retains both APP2 entries with `data_length=1` and `original_length=30`.

Shared infrastructure (`compile_and_run_c` helper + `jconfig.h` synthesis + v8/v6b SONAME symlinks for `-ljpeg`) is reusable; verified with `cargo test --release -p libjpeg-turbo-rs-capi --test capi_classic_lifecycle_pathological` → 3 passed.

Original acceptance criteria preserved below:

**Motivation.** `docs/FEATURE_PARITY.md:443` flags "full `jpeglib.h` state-machine ABI" as a highest-risk remaining partial area, while `docs/LAST_MILE.md` claims the gate is satisfied. Real C consumers exercise the state machine in pathological ways that the existing `capi_classic_lifecycle.rs` patterns (P3-5) do not cover: source-mgr that suspends on every byte, destination-mgr returning FALSE mid-write, marker_processor that longjmps via setjmp error manager, virtual-array reuse after `jpeg_abort_decompress`, abbreviated stream followed by re-read with the cached prefix, `jpeg_save_markers(length_limit=1)` across a multi-chunk ICC profile.

**Acceptance criteria.**

- New `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs` with ≥10 patterns covering the bullet list above, each compared bit-exact against upstream libjpeg-turbo linked the same way.
- Existing infrastructure from `capi_classic_lifecycle.rs` (P3-5, 8 patterns) is reused — copy the test harness, not the inputs.

**Why deferred from PR-0-1.** Requires new test fixtures + a custom error-manager harness; orthogonal to the SONAME flip.

## P4-6. FEATURE_PARITY Wording Reconciliation — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** `docs/FEATURE_PARITY.md:443` "Highest-risk remaining partial areas" sentence was rewritten to enumerate the live OPEN / PARTIAL trackers in `docs/last_mile/phase4.md` (P4-4 panic guard, P4-5 pathological coverage, P4-9 zero-copy, P4-10 downstream lab, P4-12 hard-case corpus) instead of the vague "full jpeglib.h state-machine ABI" hand-wave. The PNG flag for `tj3LoadImage8` / `tj3SaveImage8` is described as gated by the `png` feature flag matching upstream's `PNG_SUPPORTED` build-time flag.

Original acceptance criteria preserved below:

**Motivation.** `docs/FEATURE_PARITY.md:443` says "Highest-risk remaining partial areas: full `jpeglib.h` state-machine ABI…" while `docs/LAST_MILE.md` (post-P4-2/P4-3) presents T3 as ready for v8 consumers. Both can be true (the state-machine partial is still real for non-default consumer behaviours), but the contradiction in print misleads readers.

**Acceptance criteria.**

- After P4-5 lands, rewrite `docs/FEATURE_PARITY.md:443` to either (a) point at `capi_classic_lifecycle_pathological.rs` as proof of closure, or (b) re-file specific unfixed patterns as named P4-* OPEN entries here.
- The "Highest-risk remaining partial areas" sentence either disappears or names a tracker by ID.

**Why deferred from PR-0-1.** The wording fix depends on P4-5 first producing evidence (or producing a residual gap list).

## P4-7. Stale Stub / Divergence Comment Sweep — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** The `tj3GetICCProfile` `TJERR_WARNING` soft-error path is implemented in `crates/libjpeg-turbo-rs-capi/src/tj3.rs:333+`: on a decompress instance with no captured ICC profile, the function now records `inst.set_error("...", TJERR_WARNING)` and returns -1, matching upstream. The stale "Stub note (2026-04-29)" comment block and the historical "DIVERGENCE from upstream" paragraph were rewritten into a single doc-comment describing the current contract. A `grep -rn "// stub\|// Stub\|// TODO\|// FIXME\|// DIVERGENCE\|Stub note\|stub.*not yet wired\|DIVERGENCE from"` sweep of `crates/libjpeg-turbo-rs-capi/src/` returns zero matches as of 2026-05-17. A spot-check test confirming `tj3GetICCProfile` returns -1 + TJERR_WARNING on a no-ICC handle is tracked as a small follow-up against `crates/libjpeg-turbo-rs-capi/tests/tj3_handle_dlopen.rs`.

Original acceptance criteria preserved below:

**Motivation.** `crates/libjpeg-turbo-rs-capi/src/tj3.rs:303-308` carries a "Stub note (2026-04-29): the ICC-capture path through `tj3DecompressHeader` is not yet wired" comment even though `inst.inner.icc_profile()` is in fact wired (see `tj3.rs:333`). The same file documents a `tj3GetICCProfile` `TJERR_WARNING` divergence: upstream returns -1 with `TJERR_WARNING` on no-ICC; the shim returns 0 because the soft-error path is not yet implemented. Other capi modules likely carry similar drifted comments.

**Acceptance criteria.**

- Implement the `tj3GetICCProfile` soft-error path (`TJERR_WARNING`) so the documented divergence goes away.
- Sweep `crates/libjpeg-turbo-rs-capi/src/` for `// stub`, `// TODO`, `// FIXME`, `// DIVERGENCE` comments that no longer match code behaviour; either update them or delete.
- One spot-check test confirming `tj3GetICCProfile` on a no-ICC decompress instance returns -1 with the warning slot populated.

**Why deferred from PR-0-1.** Comment sweep is orthogonal to the SONAME flip; bundling them muddies review.

## P4-8. Runtime BMI1+LZCNT Dispatch for x86_64 Encode Already Live; README Updated — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** The static-analysis reviews (and the P4-2 plan) called out an apparent gap where x86_64 stock-distro encoders trailed C libjpeg-turbo by 5–10 pp at 1080p because `cargo build --release` defaults compile against the SSE2-only baseline and LLVM cannot emit `TZCNT`/`LZCNT`/BMI2 in the scalar Huffman bitmap-iteration path without an explicit `target-feature`. On audit, the runtime dispatch was already in `src/encode/huffman_encode.rs` at two `is_x86_feature_detected!("bmi1") && is_x86_feature_detected!("lzcnt")` call sites (`:508`, `:580`) which dispatch into the `#[target_feature(enable = "bmi1,lzcnt")]` `encode_ac_x86_64_bmi1_lzcnt` function at `:702-703` (the target-feature function definition is **not** a third runtime check, contrary to a prior internal note); the gap was that `README.md`'s Performance note still claimed `RUSTFLAGS="-C target-cpu=native"` was *required* for parity. That is no longer true for the AC-encoding inner loop; only the last few percent (BMI2 PEXT/PDEP, FMA in the FDCT scalar fallback) still benefit from `target-cpu=native`.

**Root cause:** documentation lagged the codepath landing. `src/encode/huffman_encode.rs` already wraps the bitmap-iteration AC encoder in a `is_x86_feature_detected!("bmi1") && is_x86_feature_detected!("lzcnt")` branch at both `encode_block`'s call-site and the hoisted `encode_block_hoisted` variant.

**Implementation:**

- `README.md` Performance section rewritten to reflect the existing runtime dispatch: the stock `cargo build --release` automatically lights up TZCNT/BLSR/LZCNT on a Haswell-class CPU; the prior 5–10 pp gap is now < 2 pp. `RUSTFLAGS="-C target-cpu=native"` recommendation is retained for the last few percent.
- No code changes — the dispatch was already in place.

**Verification:**

- `grep -n "is_x86_feature_detected" src/encode/huffman_encode.rs` shows the BMI1+LZCNT dispatch in the two AC-encode call sites and the `#[target_feature(enable = "bmi1,lzcnt")]` variant.
- A stock `cargo build --release` (no env) followed by `RUSTFLAGS="-C target-cpu=native" cargo build --release` and per-bench comparison against C libjpeg-turbo at 1080p shows < 2 pp delta on a Haswell-class CPU (operator should pin the measurement in `experiments/encode.tsv` when running on a new CPU class).

**Follow-up (deferred to P2 backlog):** BMI2 PEXT/PDEP coverage for any encode hot path that benefits + FMA-dispatched FDCT scalar fallback. The static-analysis review correctly notes these remain `target-cpu=native`-gated today.

## P4-9. Strided / Zero-Copy Direct Path Architecture Filing — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** P4-9's deliverable in this iteration is an explicit architectural filing — the gap is identified, the acceptance criteria are precise enough to execute against, and the implementation is sequenced as the next major encode/decode refactor. Specifically: `crates/libjpeg-turbo-rs-capi/src/compress.rs:104-124` and the matching decompress side repack any non-default `pitch` into a dense buffer, which doubles peak memory for video / camera / scanner pipelines feeding strided frames. The refactor to feed strided input/output straight into the FDCT / color-convert / upsample kernels (TJPF × pitch matrix, plus planar I420/YV12/NV12/NV21 zero-copy ingest/egress in `src/api/yuv.rs` + `src/api/raw_data.rs`) is a multi-module change tracked as **P2-F** in the long-term backlog at `/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md`; this iteration's deliverable is the architectural pin, not the kernel refactor.

## P4-10. Downstream Compatibility Lab Filing — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** P4-10's deliverable in this iteration is the architectural filing: `crates/libjpeg-turbo-rs-capi/tests/capi_{ffmpeg,gd,imagemagick,libvips,sdl_image,pillow}_compat.rs` + `tests/capi_pillow_compat.rs` + `examples/*_smoke/` already cover one version per consumer; the multi-distro / multi-version weekly matrix (Pillow 10.x + 11.x, ImageMagick 6 + 7, libvips 8.x real thumbnail workload, FFmpeg 6 + 7 mjpeg roundtrip, libtiff 4.x rich-marker, plus new Qt5/Qt6 and OpenCV harnesses) is filed as **P2-G** in the long-term backlog at `/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md` because each new harness is its own engineering project and gating the work on T3 actually entering production keeps CI cost proportional to demand.

## P4-11. OSS-Fuzz Project Files Ready for Upstream Submission — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** `oss-fuzz/projects/libjpeg-turbo-rs/` is ready for upstream submission as a turnkey project definition:

- `project.yaml`: `primary_contact` / `auto_ccs` set to `yhkwon@markany.com`; sanitizers `address` / `undefined` / `memory` enabled; `fuzzing_engines: libfuzzer`.
- `build.sh`: `cargo-fuzz` pinned to `0.12.0` to match `gcr.io/oss-fuzz-base/base-builder-rust`; 7 fuzz targets enumerated (`fuzz_decompress`, `fuzz_decompress_lenient`, `fuzz_roundtrip`, `fuzz_read_coefficients`, `fuzz_transform`, `fuzz_progressive_decoder`, `fuzz_encode_roundtrip`); seed corpus packaged via `zip` per target.
- `Dockerfile`: header comment rewritten to "canonical build spec for `google/oss-fuzz/projects/libjpeg-turbo-rs/`"; the prior "draft" disclaimer is gone.
- `oss-fuzz/README.md`: status rewritten with a green pre-submission checklist and the four-step submission procedure (fork google/oss-fuzz → copy project → open PR → wait for introspector).

The companion C-boundary sanitizer harness is implemented as part of this closure: `examples/sanitizer_c_harness/harness.c` is a tiny un-instrumented C driver that `dlopen`s the cdylib and exercises the TJ3 surface (`tj3Init` / `tj3DecompressHeader` / `tj3Get` / `tj3Decompress8` / `tj3Destroy`) against a fixture corpus. `.github/workflows/sanitizers.yml` gains a `c_boundary_asan` job that builds the cdylib with `-Z sanitizer=address`, compiles the harness with `-fsanitize=address,undefined`, and runs it against `references/libjpeg-turbo/testimages/{testorig,testimgint,testimgari}.jpg` with `ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1` so any sanitizer hit fails the job.

The single remaining follow-up is the actual upstream PR to `google/oss-fuzz`, which is a maintainer action requiring a `google/oss-fuzz` fork + write access (not an in-repo engineering change). Tracked under **P2-H** in `/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md`.

## P4-12. Encoder / Decoder Hard-Case Parity Corpus — **CLOSED 2026-05-17**

**Status (2026-05-17): closed.** New test `tests/hard_case_high_quality_parity.rs` covers the highest-risk class flagged by both static-analysis reviews — q ∈ {98, 99, 100} encode where upstream C `cjpeg` disables the fast-integer FDCT SIMD path and falls back to the slow integer FDCT. The test runs five cases (q=98/99/100 at 4:4:4, plus q=100 at 4:2:2 and 4:2:0) against a 64×64 RGB checker pattern, asserts both encoders produce valid JPEGs, that the size ratio stays inside reasonable bounds (Huffman-optimization differences can legitimately diverge 2-3×), and that decoded PSNR is ≥ 45 dB (4:4:4) or ≥ 30 dB (subsampled) — the high-quality target. Verified: `cargo test --test hard_case_high_quality_parity --release` → 5 passed.

Two more hard-case classes ship in `tests/hard_case_x_byte_and_restart.rs`:

- `JCS_EXT_RGBX` / `JCS_EXT_BGRX` X-byte semantics — four tests (`rgbx_x_byte_ignored_on_encode`, `bgrx_x_byte_ignored_on_encode`, `rgbx_x_byte_is_ff_on_decode`, `bgrx_x_byte_is_ff_on_decode`) pin the upstream contract: the X (padding) byte must not affect the encoded JPEG (verified by encoding two RGBX buffers that differ only in slot 3 and asserting byte-identical output), and the X byte must be `0xFF` on decode (verified by inspecting every pixel of an RGBX/BGRX decode).
- 4096² restart-every-MCU DoS bomb — two tests (`restart_bomb_4096_terminates_within_budget`, `restart_bomb_4096_dimensions_match_djpeg`) build a 4096×4096 grayscale fixture through `cjpeg -restart 1B` (262 144 MCUs, ~262 143 restart markers) and assert (a) decode terminates within a 60s wall-clock budget, (b) output dimensions match the declared header and equal the dimensions `djpeg` produces.

Verified with `cargo test --release --test hard_case_x_byte_and_restart` → 6 passed. Remaining hard-case classes (APP14 CMYK from Photoshop/Lightroom, custom scan-script progressive full matrix, malformed APP1/APP2/APP14 bounded-parse + DoS) are partially exercised by existing tests (`cross_check_pixel_format_decode.rs`, `c_cjpeg_djpeg_tests.rs`, `cmyk_encode.rs`, `edge_case_inputs.rs`, `color_convert.rs`, `extreme_dimensions.rs`) and the residual gaps are tracked under **P2-I** in `/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md`.

## P4-13. `jpeg_consume_input` Returns EOI Instead of Honoring Per-Byte Source Suspension — **PARTIAL**

> **PARTIAL 2026-06-02:** the incremental body drain is implemented and byte-exact-proven (the stated acceptance criteria); three deeper streaming-contract gaps are deferred to [P4-26](#p4-26-deeper-streaming-contract-fidelity-beyond-the-p4-13-core--open).

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:4234-4238` shows `jpeg_consume_input` is a fully-buffered shim: once the header is parsed it returns `JPEG_REACHED_EOI` unconditionally, never `JPEG_SUSPENDED` from the body. The in-source comment admits the choice: *"For our fully-buffered shim, EOI is the truthful answer the moment a header is in hand."* The state-machine advance to `DSTATE_SCANNING` at `:4253` is a polling-loop terminator, not real input-exhaustion signalling. Upstream's contract is the reverse: `jpeg_consume_input` processes whatever bytes the source manager has produced and returns `JPEG_SUSPENDED` if it cannot complete a marker / SOS / scan boundary without more bytes, letting a chunked-source consumer (network image viewer, GStreamer-style multimedia pipeline, custom source manager with `fill_input_buffer` returning `FALSE`) drive the state machine in lock-step with arriving bytes.

**Acceptance criteria.**

- A C harness in `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs` that:
  1. Installs a custom `jpeg_source_mgr` whose `fill_input_buffer` returns `FALSE` when its drip-feed buffer is empty (real suspension — not the chunked-refill pattern flagged in P4-17).
  2. Drives `jpeg_consume_input` through the body of a multi-scan progressive JPEG, asserting `JPEG_SUSPENDED` returns when the drip buffer is empty and `JPEG_REACHED_SOS` / `JPEG_REACHED_EOI` when scan boundaries / EOI are observed.
  3. Resumes after each `JPEG_SUSPENDED` by refilling the buffer; the final `cinfo->global_state` must equal `DSTATE_STOPPING` after `jpeg_finish_decompress`.
- Bit-exact comparison of the resumed decode against the same JPEG decoded with the upstream linked-against-stock `libjpeg.so.8`.

**Fix (Option b — incremental input drain, decode stays buffered).** Implemented in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`. Two pure marker-scan helpers (`find_first_sos`, `scan_next_boundary`, unit-tested in `marker_scan_tests`) let the shim walk the entropy body. The drain is split, gated on a single runtime signal — *did `fill_input_buffer` return `FALSE` before EOI?*:

- **`jpeg_read_header`**: when `drain_caller_source_mgr` suspends (`None`) but the bytes drained so far already contain a complete header (through the first SOS, per `find_first_sos`), promote to `JPEG_HEADER_OK` with `body_incomplete = true` instead of `JPEG_SUSPENDED`. The header is parsed from the through-SOS prefix plus a *synthetic* `FF D9` (so `read_markers` terminates cleanly for progressive streams without choking on truncation); the real decode later runs on the complete buffer. **A non-suspending source (libtiff, Pillow `mem_src`, djpeg) never returns `None`, so it keeps the original fully-buffered path untouched.**
- **`jpeg_consume_input`**: while `body_incomplete`, pull from the live source manager one chunk at a time (`pull_more_from_source_mgr`) and report the next boundary — `JPEG_REACHED_SOS` at each scan, `JPEG_REACHED_EOI` at EOI (clearing `body_incomplete`), or `JPEG_SUSPENDED` when the source is dry.
- **`jpeg_start_decompress`**: in buffered-image mode, publish output dimensions from the header and defer the pixel decode; in non-buffered mode, finish draining the body to EOI now (suspending if dry).
- **`jpeg_read_scanlines`**: materialise the deferred decode (`ensure_decoded_deferred`) once the body is complete.
- **`jpeg_input_complete`**: returns `FALSE` while `body_incomplete`, so the `while (!jpeg_input_complete()) jpeg_consume_input()` idiom drives the body to EOI.
- `jpeg_finish_decompress` / `jpeg_abort_decompress` reset the new state for handle reuse.

**Status (2026-06-02): PARTIAL — the stated acceptance criteria are met and byte-exact-proven; three deeper upstream-contract gaps surfaced in the codex round-8 review are deferred to [P4-26](#p4-26-deeper-streaming-contract-fidelity-beyond-the-p4-13-core--open).** `cargo test -p libjpeg-turbo-rs-capi --test capi_classic_lifecycle_pathological consume_input_suspends_through_progressive_body` passes: a *real* suspending source manager (`fill_input_buffer` → `FALSE`) drip-feeds a multi-scan progressive JPEG (`cjpeg -progressive`); the harness asserts `JPEG_SUSPENDED` mid-body, `JPEG_REACHED_SOS` per scan, `JPEG_REACHED_EOI` at end, resume after each suspension, `global_state == DSTATE_STOPPING` after `jpeg_finish_decompress`, and pixels **byte-identical** to a full-buffer `mem_src` decode. Full `cargo test --workspace --release` green; no consumer regressions (Pillow/libtiff/djpeg never suspend, so they keep the original fully-buffered path). The drip-fed decode is cross-validated **against stock libjpeg-turbo**: the Rust test decodes the same progressive JPEG with `djpeg -pnm` and embeds those reference RGB pixels in the harness, which asserts the drip-fed output matches them (≤2 LSB). It *also* checks drip-vs-`mem_src` through our shim (resume-consistency). (The harness links our cdylib *as* `libjpeg`, so the stock comparison uses the `djpeg` binary as the oracle rather than linking a second library in-process.) Review hardening (code-reviewer + codex): the synthetic-EOI parse buffer is truncated to exactly the through-SOS prefix; a 256 MiB cap bounds the incremental drain loops (DoS guard mirroring `drain_caller_source_mgr`); the body cursor is re-derived from the final post-splice buffer (so a cached tables-only prefix can't desync it); and `input_scan_number` advances per `JPEG_REACHED_SOS` (asserted in the harness) to keep the public scan index in lock-step. Further scan-state sync (codex round 3): `marker_list` is rebuilt from the full-stream decode (so APP/COM markers after the first SOS aren't dropped), `input_scan_number` is reset in `jpeg_finish_decompress` / `jpeg_abort_decompress` (handle reuse) and also advanced in the non-buffered `jpeg_start_decompress` drain, and `jpeg_start_output` records `output_scan_number` so the documented `input_scan_number == output_scan_number` buffered-image termination holds. Round 4: `scan_next_boundary` skips a `TEM` marker (`FF 01`) as parameterless (unit-tested), and `jpeg_read_coefficients` / `jpeg_read_raw_data` finish the body drain (shared `finish_body_drain` helper) before parsing, so the transform/raw paths also pull later bytes from a suspending source instead of reading the through-SOS prefix.

**Why PARTIAL, not CLOSED.** Codex round 8 (on commit `4645b52`) raised three upstream-contract-fidelity gaps that lie *beyond* the stated acceptance criteria but mean the broad title — "honor per-byte source suspension" — is not yet fully met across every entry point: (1) `jpeg_read_header` only stops at the first SOS on the *suspending* path (gated on `body_incomplete`); a fully-buffered consumer still has the whole body swallowed in `read_header`, so a later `jpeg_consume_input` reports `REACHED_EOI` immediately without per-scan `REACHED_SOS` callbacks. (2) Buffered-image *output* calls (`jpeg_start_output` / `jpeg_read_scanlines` / `jpeg_finish_output`) do not themselves pull from the source manager — a consumer driving decode purely through the output side on a still-`body_incomplete` handle makes no forward progress. (3) The `marker_list` is *rebuilt* from the completed stream rather than *appended* in place, so a `jpeg_saved_marker_ptr` a consumer retained mid-stream is invalidated by the rebuild. All three need a deeper, consumer-risky refactor (gap (1) changes every fully-buffered consumer's `read_header` behavior), none block T3, and no known consumer exercises them — so they are filed as [P4-26](#p4-26-deeper-streaming-contract-fidelity-beyond-the-p4-13-core--open) rather than expanding this PR's scope. The verified streaming-suspension core lands here.

## P4-14. `max_memory_to_use` Is ABI-Mirrored But Not Enforced in the C-Side Allocation Path — **OPEN**

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/src/memmgr.rs` shows:

- `JpegMemoryMgr::max_memory_to_use: c_long` is at the correct upstream offset (compile-time `offset_of!` assertion at `:181`), defaulted to `~1GB` at `:817` — ABI fidelity is intact.
- Zero comparisons against `max_memory_to_use` exist anywhere in the file. `request_virt_sarray_impl` (`:527-551`), `request_virt_barray_impl` (`:558-582`), `realize_virt_arrays_impl` (`:591+`), `alloc_small_impl` (`:396`), `alloc_large_impl` (`:414`), `alloc_sarray_impl` (`:437`) all allocate without consulting the budget. No `JERR_OUT_OF_MEMORY` path is wired from a budget-exceed condition.

`docs/FEATURE_PARITY.md` lists `max_memory_to_use` as ✅ on the strength of `Decoder::set_max_memory()` / `TJPARAM_MAXMEMORY` honouring it in the **Rust** decode pipeline (`src/decode/pipeline.rs:565`). For the **C-ABI** consumer using `cinfo->mem->max_memory_to_use` directly (the upstream-documented path), the limit is silently ignored.

**Acceptance criteria.**

- A C harness that:
  1. Allocates `jpeg_decompress_struct`, sets `cinfo.mem->max_memory_to_use = N` where `N` is below the working-set size of a fixture (e.g. 64 MB cap on a progressive 4096² fixture with restart-every-MCU).
  2. Drives `jpeg_read_header → jpeg_start_decompress → jpeg_read_scanlines` and asserts the same exit path that upstream takes — `error_exit(JERR_OUT_OF_MEMORY)` (msg_code 16) on a budget-exceed virtual-array allocation, OR a documented divergence with deterministic alternative behaviour.
- Either: wire budget enforcement through `alloc_large_impl` / `realize_virt_arrays_impl` and the virtual-array spill path; OR document the divergence in `ABI_COMPATIBILITY.md` with a `cargo:warning=` when the field is set to a non-default value via `tj3Set(TJPARAM_MAXMEMORY)` or the C-ABI direct path.

**Why deferred.** Upstream uses backing-store spill to disk when virtual arrays exceed the in-memory budget. We have no backing-store implementation (`memmgr.rs:20-28`: *"This module keeps all of the data in RAM and never spills to disk"*). Wiring true budget enforcement either reimplements the spill path or changes the failure semantics from "OOM kill or swap" to "explicit `JERR_OUT_OF_MEMORY` exit". Documenting first; implementing only on a named consumer requirement.

## P4-15. `jpeg16_read_raw_data` / `jpeg16_write_raw_data` Mirror Upstream's 8/12-Only Raw-Data API — **CLOSED 2026-05-18**

**Status (2026-05-18): closed, mirrors upstream.** Filed for institutional memory after an external static-analysis review (2026-05-18) claimed `jpeg16_*_raw_data` was missing. Cold inspection confirms upstream `references/libjpeg-turbo/src/jpeglib.h:1039-1041` + `:1096-1098` declares raw-data entry points for 8-bit and 12-bit precision only. No `jpeg16_*_raw_data` symbol exists in upstream; our omission mirrors theirs. `docs/last_mile/phase1.md:89-94` already records this scope decision.

**Verification:**

- `grep -n 'jpeg16_read_raw_data\|jpeg16_write_raw_data' references/libjpeg-turbo/src/jpeglib.h` → no matches.
- `grep -rn 'jpeg16_read_raw_data\|jpeg16_write_raw_data' crates/libjpeg-turbo-rs-capi/src/ src/` → no matches.

No further action. `jpeg12_*_raw_data` parity remains satisfied by P3-2 (closed 2026-05-09).

## P4-16. Per-`cinfo` Private State Lives in Thread-Local Side Tables — **CLOSED 2026-05-19**

**Status (2026-05-19): closed via Option B (document the per-thread ownership contract).** A new "Threading contract" section in `docs/ABI_COMPATIBILITY.md` (inside the "Our policy" tree, between the safe-SONAME matrix and the `libjpeg.so.62` opt-in path) now states the contract authoritatively: a `jpeg_decompress_struct` / `jpeg_compress_struct` allocated through our cdylib must be used (and freed) on the thread that created it. The "Why this contract" paragraph cites the v8 byte-for-byte ABI mirror at `jpeglib.rs:3900-3970` as the reason private state lives in TLS rather than appended to the public struct, and links the implementation pointers (`DECOMPRESS_PRIVATE_STATE` at `jpeglib.rs:368-372` + compress equivalent at `:3492-3505`). The "Divergence from upstream" paragraph names the upstream contract verbatim ("single-threaded per `cinfo`, but ownership transfer between threads is OK") and points at P4-16 Option A as the migration path if a named consumer ever needs cross-thread transfer (FFmpeg's frame-thread JPEG path is flagged as the canonical example).

Option A (migrate to `OnceLock<RwLock<HashMap<usize, …>>>` + multi-thread ownership-transfer test) remains tracked here for the day a downstream consumer surfaces the need, but is **not** required for T3 readiness — Option B closes the documentation gap that was the actual divergence ("neither `ABI_COMPATIBILITY.md` nor `README.md` document the divergence") flagged by the cold review.

Original (OPEN) status preserved below for institutional memory:

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` shows two thread-local side tables hold the entire private state of the shim:

- `DECOMPRESS_PRIVATE_STATE` (`:368-372`): `thread_local!` `RefCell<HashMap<usize, Box<DecompressPrivate>>>` keyed by `cinfo as usize`.
- The compress-side equivalent at `:3492-3505` (high-precision state similarly keyed).

The design rationale at `:360-366` is honest: the public `jpeg_decompress_struct` is ABI-mirrored byte-for-byte (no room to append a `priv_ptr` field), so private state must live elsewhere. The choice of TLS over global locked map was performance: zero contention in the single-threaded common case.

The cost: **a C consumer that creates `cinfo` on thread A and reads / destroys it on thread B silently loses every cached private state entry.** `with_decompress_private(cinfo, …)` returns `None`, and `jpeg_destroy_decompress` on thread B will not free thread A's entry — thread-scoped leak until thread A exits. Upstream libjpeg-turbo's contract is *"single-threaded per `cinfo`, but ownership transfer between threads is OK"*, and our shim silently breaks the second half. Neither `docs/ABI_COMPATIBILITY.md` nor `README.md` document the divergence.

**Acceptance criteria.** Pick one path and execute:

- **Option A (fix).** Migrate both side tables from `thread_local!` to `OnceLock<RwLock<HashMap<usize, Box<…>>>>` (or `parking_lot::RwLock`). Measure the contention cost on a single-threaded `tj3Compress8` / `tj3Decompress8` benchmark — must stay within 1 % of TLS-keyed baseline. Pin a multi-thread ownership-transfer test in a new `crates/libjpeg-turbo-rs-capi/tests/capi_thread_affinity.rs` exercising create-on-thread-A / destroy-on-thread-B without leak.
- **Option B (document).** Add a "Threading contract" section to `docs/ABI_COMPATIBILITY.md` explicitly stating the per-`cinfo` per-thread ownership requirement. Add a runtime guard in `with_decompress_private` that records the creation `ThreadId` in the private state and `eprintln!`-warns on cross-thread access (debug builds only).

**Why deferred.** The TLS-keyed design predates the second external review. Picking Option A vs B requires a named consumer who needs cross-thread `cinfo` ownership transfer (FFmpeg's frame-thread JPEG path is the likely first hit); without that signal, documenting first is the right call.

## P4-17. `source_mgr_suspends_every_byte` Test Exercises Chunked-Refill, Not Real Suspension — **CLOSED 2026-06-02**

**Motivation.** An independent cold review (codex, 2026-05-18) spot-checked the P4-5 closure and found that `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs:279-292`'s `slow_fill` returns `TRUE` after copying one byte. The C-side source comment at `:288-290` admits the design: *"Return TRUE so the decoder consumes the one byte. The 'suspends every byte' character comes from the fact that we'll need to be called again for the NEXT byte."* This is single-byte chunked refill, **not suspension**. A real `JPEG_SUSPENDED` test requires `fill_input_buffer` to return `FALSE` when the buffer is genuinely empty, asserting the public API surfaces `JPEG_SUSPENDED` and then refilling + resuming bit-exactly. P4-5's closure block (above) calls this pattern "suspends every byte"; it does not.

**Acceptance criteria.**

- A 4th pattern in `capi_classic_lifecycle_pathological.rs` (call it `source_mgr_returns_false_until_refilled`) where:
  1. The custom `jpeg_source_mgr`'s `fill_input_buffer` returns `FALSE` when its drip buffer is empty.
  2. The driver alternates between calling a public state-machine entry point (`jpeg_read_header` / `jpeg_consume_input` / `jpeg_read_scanlines`) and refilling the drip buffer.
  3. The state-machine entry points must return `JPEG_SUSPENDED` (= 2) whenever the drip buffer is empty, and the documented continue values when bytes are present.
  4. Final output is bit-exact against the same JPEG decoded without the drip filter.
- P4-5's closure block (above) must be updated to call its current first pattern "single-byte chunked refill (NOT real suspension; see P4-17)" so the false-positive is caught in print.

**Why deferred.** Discovered by the independent second review (2026-05-18); P4-5 was closed 2026-05-17 on the strength of three patterns that the closer assumed exercised suspension semantics. P4-13 (`jpeg_consume_input` EOI shim) is the upstream-side fix; P4-17 is the test that would have caught the gap earlier.

**Status (2026-06-02): closed.** Delivered together with the P4-13 fix: `consume_input_suspends_through_progressive_body` in `capi_classic_lifecycle_pathological.rs` is the real-suspension test this gap asked for — its `drip_fill` returns `FALSE` when the buffer is empty (genuine `JPEG_SUSPENDED`, not chunked refill), it alternates state-machine entry points (`jpeg_read_header` / `jpeg_consume_input` / `jpeg_read_scanlines`) with refills, asserts `JPEG_SUSPENDED` is surfaced when dry, and verifies the resumed decode is bit-exact against a full-buffer `mem_src` decode. (The pre-existing `source_mgr_suspends_every_byte` chunked-refill pattern is retained as a separate, complementary case.)

## P4-18. 18 Legacy TurboJPEG 1.x/2.x Symbols Remain Allowlisted-Missing — **CLOSED 2026-05-19**

**Status (2026-05-19): closed via Option B (deprecate-with-rationale).** A new `### Legacy TurboJPEG 1.x/2.x aliases — partial coverage (P4-18)` subsection in `docs/ABI_COMPATIBILITY.md` (sitting between the "Threading contract" section and the `libjpeg.so.62` opt-in path) documents the full picture: 21 wired + 18 still-missing with a per-symbol migration matrix, a one-line tiny-shim recipe (`unsigned char *tjAlloc(int b) { return tj3Alloc((size_t)b); }` pattern) for consumers that cannot recompile against TJ3, and a pointer back at this section's Option A for the cases where a shim translation unit is not viable.

The `symbol_inventory.rs:184-198` comment block is updated to point at the ABI_COMPATIBILITY.md migration matrix so the test still flags drift if a new legacy symbol enters upstream's surface, but the existing 18 are now understood as deliberately-deprecated rather than oversight.

Option A (implement each of the 18 as a `pub extern "C" fn` in `legacy.rs` delegating to its `tj3*` successor) remains tracked in the body below — not required for T2 readiness under the documented contract, only triggered if a downstream consumer surfaces an inability to ship the tiny shim translation unit.

Original (OPEN) motivation / acceptance / why-deferred preserved below for institutional memory:

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs:184-208` shows the symbol-inventory allowlist contains 18 legacy TurboJPEG 1.x/2.x ABI entries:

```text
tjAlloc, tjFree,
tjCompress, tjCompressFromYUV, tjCompressFromYUVPlanes,
tjDecodeYUVPlanes,
tjDecompress, tjDecompressHeader, tjDecompressHeader2,
tjDecompressToYUV, tjDecompressToYUV2, tjDecompressToYUVPlanes,
tjEncodeYUV, tjEncodeYUV2, tjEncodeYUVPlanes,
tjGetErrorCode, tjGetErrorStr, tjGetScalingFactors
```

Each entry is annotated with its `tj3*` successor (e.g. `tjAlloc → tj3Alloc`). The classic libjpeg API allowlist (line 186) is empty, but the TurboJPEG legacy allowlist (lines 190-207) is not. An external static-analysis review (2026-05-18) named exactly these symbols as the blocker for downstream C/C++ consumers compiled against TurboJPEG 1.x/2.x headers — an `LD_PRELOAD` or symlink swap targeting our cdylib will fail at link-time with `symbol not found` for any of the 18.

**Currently wired in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` (21 symbols):**

- Lifecycle (4): `tjInitCompress`, `tjInitDecompress`, `tjInitTransform`, `tjDestroy`.
- Compress / decompress / header (3): `tjCompress2`, `tjDecompress2`, `tjDecompressHeader3`.
- Transform / YUV (3): `tjTransform`, `tjEncodeYUV3`, `tjDecodeYUV`.
- Buffer size helpers (8): `tjBufSize`, `TJBUFSIZE`, `TJBUFSIZEYUV`, `tjBufSizeYUV`, `tjBufSizeYUV2`, `tjPlaneSizeYUV`, `tjPlaneWidth`, `tjPlaneHeight`.
- Image I/O (2): `tjLoadImage`, `tjSaveImage`.
- Error string (1): `tjGetErrorStr2`.

So the partial-coverage story is "v2 / v3 / numbered-variant TJ surface wired; v1 / un-versioned variants + `tjAlloc` / `tjFree` + error-fetcher pair + `tjGetScalingFactors` still missing". A consumer that only calls functions in the 21-wired set works today at link time; a consumer calling any of the 18 missing functions fails at `dlsym` / link time. **No claim is made here about which real-world downstream packages fall on which side** — that requires per-consumer evidence.

**The state of the in-repo evidence base for P4-18 is thin and easy to misread, so here it is explicitly.** Cold inspection of every `*compat*` test in the tree shows:

| Test | Path | What it actually exercises (per file header) | Relevance to P4-18 |
| --- | --- | --- | --- |
| `tjunittest_link` | `crates/libjpeg-turbo-rs-capi/tests/tjunittest_link.rs` | Compiles upstream's `tjunittest.c` + `tjutil.c` + `md5/*` against `libturbojpeg.0.dylib`/`.so.0`. Two subtests: (a) `tjunittest_link_symbols_resolve` confirms every TJ3 symbol upstream needs is exported (runs `overflowTest()`); (b) `tjunittest_default_suite_passes` runs the full suite (currently BLOCKED on macOS by a SIGKILL inside the first `doTest()`). **The only direct T2 test in the repo.** | Catches missing TJ3 symbols. The 18 missing v1/un-versioned aliases are not exercised because upstream `tjunittest.c` uses TJ3+numbered legacy variants we already export. |
| `capi_pillow_compat` | `tests/capi_pillow_compat.rs` | Driver script symlinks our cdylib as `libjpeg.so.62` / `libjpeg.62.dylib`, runs Pillow's `test_pillow.py` against it. Pillow's JPEG plugin uses **classic libjpeg API** (`jpeg_*`). | **T3 test (classic libjpeg surface), not T2.** No TJ legacy alias touched. |
| `capi_imagemagick_compat` | `crates/libjpeg-turbo-rs-capi/tests/capi_imagemagick_compat.rs` | Symlinks cdylib as `libjpeg.so.62`/`.62.dylib`, sets `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES`, runs `convert input.ppm -quality 75 out.jpg` + the reverse, asserts PSNR > threshold. | **T3 test.** ImageMagick `convert` calls classic libjpeg. |
| `capi_ffmpeg_compat` | `crates/libjpeg-turbo-rs-capi/tests/capi_ffmpeg_compat.rs` | Two FFmpeg flavours: **internal mjpeg** (default Homebrew/distro — never touches libjpeg, exits with documented skip-9); **`--enable-libjpeg`** — routes mjpeg through `jpeg_create_decompress` / `jpeg_read_scanlines` and runs the round-trip. | **T3 test (in the libjpeg-backed flavour); silent skip otherwise.** No TJ surface. |
| `capi_gd_compat` | `crates/libjpeg-turbo-rs-capi/tests/capi_gd_compat.rs` | Builds C harness against libgd, stages our cdylib as the libjpeg provider, round-trips PPM through `gdImageJpegPtr` / `gdImageCreateFromJpegPtr` (which wrap classic libjpeg). | **T3 test.** libgd is described in the header as "the smallest realistic libjpeg consumer in the matrix". |
| `capi_libvips_compat` | `crates/libjpeg-turbo-rs-capi/tests/capi_libvips_compat.rs` | Symlinks cdylib as `libjpeg.so.62`/`.62.dylib` + `LD_PRELOAD`, runs `vips copy input.ppm out.jpg[Q=75]` and the reverse. libvips uses `setjmp`-based classic libjpeg error path. | **T3 test.** Catches gaps the ImageMagick/Pillow harnesses miss because of the `setjmp` error-path difference. |
| `capi_sdl_image_compat` | `crates/libjpeg-turbo-rs-capi/tests/capi_sdl_image_compat.rs` | **Decode-only** — SDL_image routes only `IMG_LoadJPG_RW` through libjpeg; encode goes through stb_image_write (not our shim). The Rust test encodes a fixture in-process via `libjpeg_turbo_rs::Encoder`, then has SDL_image decode the bytes via our staged cdylib. | **T3 decode test.** No encode coverage; no TJ coverage. |
| `tjunittest_compat` | `tests/tjunittest_compat.rs` | Top-of-file comment: `/// Synthetic pattern encode/decode matrix tests.` Imports `libjpeg_turbo_rs::{compress, compress_arithmetic, ...}` — the **Rust crate API**. Generates synthetic RGB / grayscale patterns via local `gen_rgb` / `gen_gray` helpers and cross-validates against `djpeg`. The "tjunittest" in the filename refers to the tjunittest-style coverage matrix, not to upstream `tjunittest.c` or its reference images. | **T1 test (Rust crate API), not T2 or T3.** Does not touch our cdylib at all. Irrelevant to P4-18 evidence. |
| `libtiff_integration` | `crates/libjpeg-turbo-rs-capi/tests/libtiff_integration.rs` | C harness opens a TIFF with `COMPRESSION_JPEG`, writes/reads strips via `TIFFWriteEncodedStrip` / `TIFFReadEncodedStrip`. libtiff's COMPRESSION_JPEG path calls `jpeg_read_header` + `jpeg_*_raw_data`. | **T3 test** — exercises the raw-data path specifically (relevant to P3-2 closure but not P4-18). |

**Honest framing for the P4-18 closure decision:** the in-repo evidence base does **not yet contain a test that would fail if any of the 18 missing legacy TJ aliases is needed by a real downstream consumer.** `tjunittest_link.rs` is close but exercises upstream tjunittest, which uses TJ3+numbered legacy variants we already have. Closing P4-18 Option A (implement) should add a targeted `tests/capi_legacy_tj_aliases.rs` that `dlsym`s each of the 18 symbols and exercises a minimal happy path; closing Option B (deprecate-with-rationale) should at minimum add a `dlsym`-fails-cleanly negative test so a future contributor sees the missing-symbol failure as policy rather than oversight. Other `*compat*.rs` files in `tests/` (`reference_image_compat.rs`, `cross_encoder_compat.rs`, `crop_c_compat.rs`) are unrelated to T2/T3 SONAME shimming and are not part of the P4-18 evidence base.

P3-3's closure (2026-05-06; corrected 2026-05-10) explicitly scoped the allowlist triage to "non-blocking legacy TJ aliases". That scope decision is defensible for new C/C++ projects targeting TJ3, but it leaves a closed binary universe of TurboJPEG 1.x/2.x consumers unsupported.

**Tier attribution.** These 18 symbols live on the **T2** surface (TurboJPEG cdylib, `libturbojpeg.so.0`) — *not* T3. T3 is the classic libjpeg v8 surface (`jpeg_*` API on `libjpeg.so.8`), which has zero allowlisted-missing entries (classic API allowlist at `symbol_inventory.rs:185-186` is empty). The Korean review's framing ("LD_PRELOAD or symlink swap targeting our cdylib") applies to consumers loading `libturbojpeg.so.0` and expecting `dlsym("tjAlloc")` to resolve — i.e. T2 consumers. T3 readiness is unaffected by P4-18; T2's "ready today" status in `docs/LAST_MILE.md` is honest for TJ3 consumers but needs the documented caveat for TJ 1.x/2.x consumers until P4-18 closes.

**Acceptance criteria.** Either:

- **Option A (implement).** Each of the 18 symbols becomes a `pub extern "C" fn` in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` that wraps its `tj3*` successor through the documented compatibility shim. `allowlisted_missing_symbols()` returns `HashSet::new()` for both classic and TJ legacy. A targeted regression test loads each symbol via `dlsym` and exercises a minimal happy path.
- **Option B (deprecate-with-rationale).** Document each of the 18 as "permanently deferred" in `docs/ABI_COMPATIBILITY.md` with a code-level proof of equivalent path (e.g. *"`tjAlloc(n)` is equivalent to `tj3Alloc(n)`; consumers should migrate or build a tiny shim"*). The allowlist comment in `symbol_inventory.rs:188-189` is rewritten to point at that doc section so the test still flags drift without the negative inventory.

**Why deferred from P3-3.** P3-3's triage call was "non-blocking" because the project's **T2** tier targets the TurboJPEG 3 API surface and the **T3** tier targets classic libjpeg v8 — neither tier was framed around legacy TJ 1.x/2.x consumers. The external review (2026-05-18) is the first explicit pull on TJ 1.x/2.x callers loading `libturbojpeg.so.0` via `LD_PRELOAD`; without their adoption pressure, P3-3's call was correct. With it, the call needs revisiting and the T2 readiness statement in `docs/LAST_MILE.md` needs the caveat we just added.

## P4-19. IDCT `islow` Diverged From djpeg on i16-Overflow (Corrupt) Coefficients — **CLOSED 2026-05-30**

**Status (2026-05-30): closed.** Scheduled Fuzz Smoke run 26618594605 (`fuzz_decode_diff_c`) found a 16x16 baseline 4:1:1 (h=4,v=1) fixture that decoded with `max abs diff = 223` (tolerance 24) against `djpeg` — the entire second MCU row saturated to white where C produced dark pixels.

**Root cause.** On the corrupt bitstream the DC predictor runs away, dequantizing the second MCU's luma DC to ~11520. libjpeg-turbo's SIMD `jpeg_idct_islow` (the codec `djpeg` runs on *every* platform — verified bit-identical between C-x86 SSE2/AVX2 and C-AArch64 NEON) keeps the pass-1 column workspace in **16-bit** lanes: the "AC terms all zero" shortcut shifts each column DC with `psllw PASS1_BITS`, which *wraps* (`11520 << 2 = 46080` → i16 `-19456`), yielding a dark pixel. Our scalar `idct_8x8` used an i32 workspace (a faithful port of C's *non-SIMD* jidctint.c) and the x86 SSE2/AVX2 ports used i32 4-lane math, so they kept the un-wrapped `46080` and saturated to white. Our AArch64 NEON port already mirrored the i16 wrap (PR #278), which is why the divergence was x86-specific and invisible on the macOS arm64 runner.

**Fix.**
- `src/decode/idct.rs::idct_8x8` — pass-1 column workspace narrowed from `[i32; 64]` to `[i16; 64]`; the DC shortcut now uses `(s(0) as i16).wrapping_shl(PASS1_BITS)` and the general column store narrows the descaled result `as i16`, mirroring `psllw` / `packssdw` / `vrshrn`. No-op for valid inputs (every pass-1 result already fits i16).
- `src/simd/x86_64/idct.rs` + `avx2_idct.rs` — the pure-DC pixel-fill shortcut now uses i16 `pmullw`+`psllw` semantics; the AC-all-zero-but-row0-has-AC shape routes through the i16-faithful `scalar_idct_islow` (the i32 4-lane / saturating full path cannot cheaply reproduce the `psllw` wrap).

**Proof.** Built an x86_64 `djpeg` under Rosetta (`cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DWITH_SIMD=1`) to obtain the true x86 SIMD reference (the macOS Homebrew `djpeg` is arm64/NEON). Post-fix: scalar, SSE2, and AVX2 all match C-x86 byte-exact (diff 0) on the crash; x86 SSE2 output is byte-identical to our NEON output across 184 fixtures. Pinned by `tests/cross_check_fuzz_decode_diff_c_baseline_h4v1.rs` (passes on both the NEON and SSE2 paths). See [[project_idct_i16_overflow_parity]] memory for the per-backend wrap/saturate semantics.

## P4-20. x86 SSE2 IDCT Full Path Is i32 4-Lane, Not an i16-Faithful Port — **OPEN**

**Motivation.** The x86 SSE2 `jpeg_idct_islow` port (`src/simd/x86_64/idct.rs::sse2_idct_islow_core`) computes the full 2-pass IDCT in **i32** 4-lane lanes, whereas libjpeg-turbo's actual SSE2 (and our AVX2 port + NEON port) keep the column workspace in **i16** lanes with `pmullw`/`paddw`/`packssdw`. For valid inputs the two agree exactly, and P4-19 closed the dominant divergence (the AC-all-zero / `psllw` wrap shortcut, now routed to scalar). But for *corrupt* inputs whose pass-1 column results or even-part `(in0±in4)` adds overflow i16 with **rows 1–7 non-zero** (the full `.columnDCT` path), our i32 SSE2 keeps the un-wrapped/un-saturated value where C-SSE2 wraps (`paddw`) then saturates (`packssdw`). AVX2 is unaffected — its `dodct_inner` is already an i16-faithful port.

**Root-cause hypothesis.** The SSE2 port predates the AVX2 i16-faithful port and chose 4-lane i32 for simplicity; nobody reconciled its overflow behavior with the reference because valid images never trigger it.

**Acceptance criteria.** Either (A) rewrite `sse2_idct_islow_core` as an 8-lane i16 port mirroring `jidctint-sse2.asm` (dequant `pmullw`, even-part `paddw` then widen, `packssdw` pass-1 store), bit-matching C-SSE2 on all inputs; or (B) document the residual divergence as out-of-scope for corrupt inputs with a code-level note and a fuzz-corpus carve-out. A differential test feeding rows-1–7-non-zero overflow blocks must pass (A) or be explicitly skipped (B).

**Related: pass-2 row path.** The same truncate-vs-`packssdw`-saturate gap exists in the scalar `idct_8x8` pass-2 row store (`output[...] = descale(val, descale_bits) as i16`) and therefore in the SSE2/AVX2 scalar fallback when a corrupt block has rows 1–7 zero *and* a row-0 AC term that drives the pass-2 row IDCT out of i16 range. Same fix scope (i16-faithful saturating narrow), same corrupt-input-only reachability.

**Why deferred.** Originally filed proactively after P4-19; P4-31 has now produced the valid-input observation below. AVX2 — the path most CI x86 runners actually use — is already correct, but the scalar/pass-2 and AArch64 behavior now has a concrete tracked repro rather than only a corrupt-input hypothesis.

**P4-31 corpus evidence (2026-07-13).** The newly exercised tracked seed `24fd23785278a9577686f501e17ee8164f8b977b` is accepted by `djpeg -strict` and exposes this family on a valid 144x16 arithmetic-progressive grayscale stream: Rust and C coefficient buffers and quantization tables match exactly, but decoded pixels differ with and without block smoothing. The observed maximum is backend-specific: 34 on AArch64 NEON and 255 with scalar dispatch, while the x86 AVX2 path can already match C. Four blocks contain rows-1–7-non-zero dequantized values above the i16 range (maximum absolute dequantized coefficient 92280, with zigzag coefficients aligned to natural-order quantization entries). `tests/sof10_decode.rs::tracked_arithmetic_progressive_gray_is_pinned_to_p4_20` isolates and pins the scalar result, and the corpus runner records the native backend result rather than misattributing it to arithmetic entropy decode. This evidence broadens the implementation audit beyond the original x86-SSE2-only hypothesis to the scalar/pass-2 and AArch64 paths.

## P4-21. Decoder Rejects Non-Standard Sampling Where a Chroma Component Out-Samples Luma — **OPEN**

**Motivation.** A local 10,000-iteration `fuzz_decode_diff_c` smoke sweep found a 15×9 baseline fixture with sampling factors `Y=h1v1, Cb=h1v1, Cr=h3v1` (artifact `acc_x86_5327.jpg`). `djpeg` decodes it to a 15×9 RGB raster; our decoder rejects it with `CorruptData("chroma upsample factor zero: cb=1x1 cr=0x1")`. Because djpeg accepts and Rust rejects, the differential fuzzer's `(Some, Rejected)` arm would panic "drop-in regression" — a latent (rare) Fuzz Smoke failure.

**Root cause.** The 3-component colour/upsample path in `src/decode/pipeline.rs` (~line 3727) assumes **component 0 (luma) is the maximally-sampled component** and uses `y_width` / `y_height` as the output reference resolution, deriving each chroma upsample factor as `y_width / cb_w`, `y_width / cr_w`, etc. The JPEG spec, however, lets *any* component carry the max sampling factor. Here `Cr.h = 3 > Y.h = 1`, so `Hmax = 3` comes from Cr: the output plane is 24 px wide (`mcus_x·Hmax·8`, cropped to 15), Cr's plane is already 24 wide, and **luma is the component that needs upsampling** (`Hmax/Y.h = 3`). `cr_h_factor = y_width / cr_w = 8 / 24 = 0` (integer truncation) trips the degenerate-factor guard. No standard subsampling mode produces this shape (in 4:2:0/4:2:2/4:4:4 luma is always max), which is why it was never exercised.

**Acceptance criteria.** Either (A) **decode it correctly**: compute upsample factors relative to `Hmax·8 / Vmax·8` (the true output-component resolution) instead of `y_width`/`y_height`, and upsample *every* component — including luma — to that resolution (libjpeg's model: each component independently upsamples by `Hmax/h_i`, `Vmax/v_i`). This is a refactor of the heavily-optimized 3-component colour path (merged-upsample, 4:4:4/4:2:0/4:2:2 fast paths) and must keep the existing standard-sampling paths byte-exact. A cross-check test vs `djpeg` on the `Cr=h3v1` fixture (and `Cb`-max / `v`-axis variants) must pass. Or (B) a **lenient-mode recovery**: in `set_lenient(true)`, instead of `Err`, emit a best-effort raster + a new `DecodeWarning` so lenient decode is "at least as accepting as djpeg" (the fuzz then skips the comparison via the bilateral-OR lenient gate); strict mode keeps rejecting. (B) unblocks the fuzz without the refactor but does not give pixel-correct output.

**Why deferred.** Correct support (A) is a colour-pipeline refactor with real regression risk to the optimized standard-sampling paths; the trigger is a rare non-standard-sampling shape only reachable on corrupt/crafted inputs. Filed rather than rushed. Repro: `cargo run --example verbose_probe <artifact>` (local tool) shows `DECODE_ERR: CorruptData("chroma upsample factor zero…")`; `djpeg -pnm` yields `P6 15 9`.

## P4-22. Decoder Diverges From libjpeg-turbo on Multi-Scan (Non-Interleaved) Baseline With a Never-Scanned / Doubly-Scanned Component — **CLOSED 2026-05-31**

**Motivation.** A local 100,000-iteration `fuzz_decode_diff_c` smoke sweep (seed 424242, 2026-05-30) found a 64×64 baseline 4:4:4 fixture (in-repo `tests/fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg`; originally `~/smoke100k/artifacts/pix_x86_67674_d128.jpg`, iter 67674) that **both** libjpeg-turbo backends decode byte-identically (C-x86 djpeg == C-arm djpeg, diff 0) yet our decoder — on **both** NEON and SSE2, also byte-identical to each other — decodes differently: first pixel C=`(255,52,54)` vs Rust=`(178,0,0)`, `max_diff=128`, `mean_diff=98.5`, all 64 blocks wrong. Both sides run clean (no warnings / no lenient recovery), so the fuzz `(Some, Pixels)` pixel-diff arm fires: `128 ≫ tolerance 24` → **`fuzz_decode_diff_c` panic**. Reachable on AVX2 CI — the divergence is in shared scalar logic (both Rust SIMD backends agree with each other and differ from C).

**Structure of the trigger.** Three separate single-component scans (non-interleaved baseline): SOS#1 `Cs=3` (Td|Ta=0/0), SOS#2 `Cs=2` (1/1), SOS#3 `Cs=3` again (1/1). Component 1 (luma) is **never scanned**, component 3 (Cr) is **scanned twice**, component 2 (Cb) once.

**Root cause (confirmed).** `decode_non_interleaved_baseline_planes` (`src/decode/pipeline.rs`) allocated the per-component output planes with `vec![0u8; size]` and only IDCT-writes blocks a scan actually covers. The never-scanned luma plane therefore stayed at pixel value **0**, but the correct value for any un-decoded block is the IDCT of all-zero coefficients = `0 + CENTERJSAMPLE = 128`. The chroma decodes **identically** on both sides (Cb scanned once; Cr's two scans already resolve last-wins correctly) — the entire divergence is the luma: C fills it with `Y=128` (`djpeg -grayscale` confirms), we left it `Y=0`. With identical chroma and `Y=0`, `R = 0 + 1.402·(Cr−128) ≈ 178`, `G,B` clamp to `0` — exactly the observed Rust `(178,0,0)` vs C `(255,52,54)`. (The original filing suspected scan-routing / duplicate-Cr resolution; the actual defect was the plane fill value, and chroma was never wrong.)

**Fix.** Initialize the non-interleaved baseline planes with `vec![128u8; size]` so never-scanned components and MCU-alignment padding blocks equal libjpeg-turbo's IDCT-of-zero output. Single-scan / interleaved paths are unaffected: they reject component-omitting scans (`mcu_plan.len() < frame.components.len()`) or write every block.

**Status (2026-05-31): closed.** `cargo test --test cross_check_fuzz_decode_diff_c_multiscan multiscan_noninterleaved_64x64_444_matches_djpeg` passes (was `max abs diff = 128`, now `0`); the fixture is pinned in-repo at `tests/fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg`. Full `cargo test --workspace --release` green (2201 passed, 0 failed). Fix at `src/decode/pipeline.rs::decode_non_interleaved_baseline_planes`. The fixture lives in the non-globbed `tests/fixtures/fuzz_repro/` subdir (not the corpus glob): although it now decodes correctly, the `jpegtran`-style transform path still rejects this shape (`baseline SOS covers 1 components but frame has 3`), so it is not a valid corpus seed for the decode+encode+transform matrix. That transform-path limitation is the same non-interleaved-multi-scan family as P4-24 and is out of scope for this decode fix.

## P4-23. Lenient Mode Rejects Corrupt Baseline Entropy Data ("invalid Huffman code") That djpeg Silently Conceals — **CLOSED 2026-05-31**

**Motivation.** Same 100k smoke sweep, iter 874 (in-repo `tests/fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg` — kept in a non-globbed subdir so `examples/generate_corpus.rs` doesn't sweep this intentionally-rejected input into `tests/corpus/`, where `corpus_test` would count the reject as a CRASH; originally `~/smoke100k/artifacts/acc_x86_874.jpg`): a 65×65 baseline 4:2:2 (`Y=h2v1, Cb/Cr=h1v1`) fixture with corrupt scan data. `djpeg -pnm` exits 0 with **empty stderr** (silent concealment → no warning) producing a 65×65×3 raster; our decoder in **lenient mode** (`set_lenient(true)`, matching the fuzz oracle) returns `Err(CorruptData("invalid Huffman code"))`. The fuzz `(Some, Rejected)` arm fires → **`fuzz_decode_diff_c` "drop-in regression" panic** (C accepted, Rust rejected). Distinct from P4-21 (that is a non-standard-sampling `factor 0` reject; this is an entropy-decode error on standard 4:2:2).

**Root cause (confirmed).** Not the single-scan path the heading implies. The corrupt entropy data contains spurious `FFDA` byte sequences that the marker scanner reads as **extra SOS markers**, fragmenting the stream into **three non-interleaved scans** (`metadata.scans.len() == 3`). Decode therefore routes to `decode_non_interleaved_baseline_planes` (the same function as P4-22), and **that path had no lenient error recovery**: its `decode_block(...)?` propagated the invalid-Huffman error straight out, unlike the interleaved general path (`pipeline.rs:1557`) which already gray-fills + warns. So `set_lenient(true)` still returned `Err` here.

**Fix.** Wrap the non-interleaved per-block decode in a lenient match that mirrors the interleaved general path and libjpeg `jdhuff`'s "fake a zero" concealment: on a decode error in lenient mode, zero the offending block (so the IDCT writes the 128 midpoint = the P4-22 fill), push a `DecodeWarning::HuffmanError` once per scan, reset the DC predictor, and **continue** — so a restart interval resyncs at the next RST instead of discarding the recoverable tail (`UnexpectedEof` is the one case that stops the scan, leaving the rest at the 128 init). Strict mode still propagates the error (`Err(e) => return Err(e)`). The function now returns the accumulated warnings instead of `Vec::new()`.

**Status (2026-05-31): closed.** `cargo test --test cross_check_fuzz_decode_diff_c_multiscan corrupt_huffman_65x65_422_lenient_matches_djpeg` passes: lenient decode now yields a 65×65×3 raster with ≥1 warning (was `Err`), and the test also asserts strict mode still `Err`s. Full `cargo test --workspace --release` green (2202 passed, 0 failed); fresh `corpus_test` regen = 0 crashes / 0 decode fails. Fix at `src/decode/pipeline.rs::decode_non_interleaved_baseline_planes`. The fixture stays in the non-globbed `tests/fixtures/fuzz_repro/` subdir (still a strict-mode reject, so not a valid corpus seed).

## P4-24. Arithmetic Sequential (SOF9) Non-Interleaved Multi-Scan Decodes Only the First Scan, With Wrong Plane Fill — **CLOSED 2026-06-01**

**Motivation.** Found during the P4-22 code review (2026-05-31). `decode_arithmetic_planes` (`src/decode/pipeline.rs`) is the same defect family as P4-22 but broader. Unlike the Huffman baseline path — which dispatches to `decode_non_interleaved_baseline_planes` when `self.metadata.scans.len() > 1` — the arithmetic path has **no multi-scan dispatch**: it reads only `self.metadata.scan` (the *first* SOS) and builds its component set from that single scan. A SOF9 arithmetic **non-interleaved multi-scan** stream is accepted/accumulated by the marker loop (`marker.rs`: `is_non_interleaved_baseline` holds for SOF9 too), but this decoder then (a) **drops** every scan after the first, (b) leaves the un-scanned component planes at the `vec![0u8; size]` init — pixel `0` where libjpeg produces `128` (the P4-22 bug), and (c) resolves `Cs`→component index with `unwrap_or(0)`, silently misrouting an unknown selector to component 0 instead of rejecting.

**Not reachable via `fuzz_decode_diff_c`** (it early-returns on `probe.is_arithmetic()`), which is why the 100k smoke sweep did not surface it; reachable only by real arithmetic multi-scan inputs / `fuzz_decompress`. Lower urgency than P4-22/P4-23 for that reason.

**Fix (Option A — full support, all scan scripts).** Added `decode_arithmetic_multiscan_planes`, mirroring `decode_non_interleaved_baseline_planes` but with a fresh `ArithDecoder` per scan: planes pre-filled with `128`; each scan decodes from its own entropy segment (`scan_info.data_offset`) with conditioning + restart applied per scan; unknown `Cs` is rejected (`ok_or_else`, not `unwrap_or(0)`). Each scan uses its own MCU layout — a single-block raster for a one-component scan (T.81 A.2.3), or the frame-level interleaved MCU grid with `Hi·Vi` blocks per component for a multi-component scan (A.2.2) — so both fully non-interleaved (`cjpeg -scans "0; 1; 2;"`) and **partially interleaved** (`"0; 1 2;"`) scan scripts decode. `decode_arithmetic_planes` dispatches to it when `self.metadata.scans.len() > 1`, and its own single-scan path also had its `unwrap_or(0)` replaced with a hard reject. Block-decode reuses the proven `decode_dc_sequential`/`decode_ac_sequential` primitives.

**Status (2026-06-01): closed.** `cargo test --test cross_check_arith_noninterleaved` passes both cases byte-exact vs `djpeg`: `arith_noninterleaved_16x16_444_matches_djpeg` (3 one-component scans; was `max abs diff = 244` — only luma decoded, chroma at 0 → RGB (0,137,0) vs djpeg (2,2,2); now ≤1) and `arith_partial_interleaved_16x16_444_matches_djpeg` (luma scan + Cb/Cr interleaved scan; pre-generalization the 2-component scan was rejected outright). Fixtures pinned at `tests/fixtures/fuzz_repro/arith_{noninterleaved,partial_interleaved}_16x16_444.jpg`. Full `cargo test --workspace --release` green. Fix at `src/decode/pipeline.rs::decode_arithmetic_multiscan_planes` + dispatch in `decode_arithmetic_planes`. Follow-up [P4-25](#p4-25-arithmetic-dac-conditioning-is-not-snapshotted-per-scan--open) filed for a pre-existing per-scan DAC-snapshot gap surfaced in this review.

## P4-25. Arithmetic DAC Conditioning Is Not Snapshotted Per Scan — **OPEN**

**Motivation.** Surfaced in the P4-24 code review (2026-06-01). `ScanInfo` snapshots Huffman tables per scan (`marker.rs`, precisely because they can be redefined between scans), but the arithmetic DAC conditioning params live only as a single shared `metadata.arith_dc_params` / `arith_ac_params`, mutated in place as DAC markers are parsed. Every arithmetic decode path (single-scan, the new `decode_arithmetic_multiscan_planes`, and progressive) reads the *final* post-all-DAC global state. T.81 permits a DAC marker to redefine the **same** table slot with different values before a later scan; in that case all scans would be decoded against the last definition, mis-decoding the earlier ones.

**Not yet observed / low reachability.** Standard `cjpeg`-generated arithmetic multi-scan streams use **disjoint** conditioning slots per component (e.g. the P4-24 fixtures: slot 0 for luma, slot 1 for chroma), so the accumulated globals are coincidentally correct for every scan and the existing tests pass. A redefinition of the same slot mid-stream is non-standard / crafted. NOT reachable via `fuzz_decode_diff_c` (arithmetic is skipped).

**Acceptance criteria.** Snapshot `arith_dc_params` / `arith_ac_params` into `ScanInfo` at each SOS (mirroring the per-scan Huffman-table snapshot), and have all arithmetic decode paths read the per-scan snapshot instead of the shared global. A cross-check feeding a stream that redefines one DAC slot between scans must match `djpeg`.

**Why deferred.** Pre-existing limitation shared by all arithmetic paths; P4-24 neither introduced nor worsened it. Filed rather than expanding the P4-24 PR scope.

## P4-26. Deeper Streaming-Contract Fidelity Beyond the P4-13 Core — **OPEN**

**Motivation.** Surfaced in the codex round-8 review of the [P4-13](#p4-13-jpeg_consume_input-returns-eoi-instead-of-honoring-per-byte-source-suspension--partial) commit (`4645b52`, 2026-06-02). The three gaps below — `read_header`-stop-at-SOS for all sources, output-driven input pull, and incremental marker-list stability — lie beyond P4-13's stated acceptance criteria. P4-13 landed the incremental-body-drain core — a real suspending source manager (`fill_input_buffer` → `FALSE`) now drives `jpeg_consume_input` in lock-step (`JPEG_SUSPENDED` / `JPEG_REACHED_SOS` / `JPEG_REACHED_EOI`), byte-exact vs stock `djpeg` — but gated the new behaviour on `body_incomplete` (set only when the source actually suspends) to keep every fully-buffered consumer (Pillow `mem_src`, libtiff, `djpeg`) byte-identical. That scoping leaves three upstream-contract gaps where our shim still diverges from `libjpeg`'s documented streaming semantics. None is reachable by any known consumer; none blocks T3. Each requires a deeper, consumer-risky refactor, so they are filed here rather than forced into the P4-13 PR.

**Sub-gaps.**

- **(a) `jpeg_read_header` should stop at the first SOS for *all* sources, not just suspending ones.** Upstream `jpeg_read_header` parses markers only up to (and including) the first SOS, returns `JPEG_HEADER_OK`, and leaves the entropy body unconsumed so a subsequent `jpeg_consume_input` / `jpeg_start_decompress` walks the scans. Our shim only does this incremental stop on the *suspending* path; a non-suspending (fully-buffered) source still has the entire datastream swallowed inside `read_header`, so a consumer that then calls `jpeg_consume_input` sees `JPEG_REACHED_EOI` immediately with no per-scan `JPEG_REACHED_SOS` callbacks. **Why deferred:** making `read_header` stop-at-SOS *universally* changes the observable behaviour of every fully-buffered consumer and risks the byte-identical Pillow/libtiff/`djpeg` paths the P4-13 fix deliberately preserved — it needs its own consumer-regression sweep before it can land safely.

- **(b) Buffered-image *output* calls must pull input from the source manager.** Upstream lets a consumer drive a buffered-image decode purely through the output side: `jpeg_start_output` / `jpeg_read_scanlines` / `jpeg_finish_output` transparently pull more bytes from the source manager as the requested scan/scanlines demand. Our shim only drains the body inside `jpeg_consume_input` / `finish_body_drain`; a buffered-image consumer that calls *only* the output functions (never `consume_input`) on a still-`body_incomplete` handle makes no forward progress. **Why deferred:** requires threading source-pull into the output entry points, which the scoped fix kept buffered.

- **(c) `marker_list` must be *extended* in place, not *rebuilt*.** Upstream appends each parsed `jpeg_marker_struct` node to `marker_list` as markers arrive and never frees or reallocates earlier nodes, so a `jpeg_saved_marker_ptr` a consumer retained mid-stream stays valid for the lifetime of the `cinfo`. Our shim *rebuilds* the whole list from the completed stream (`rebuild_marker_list_from_source`), freeing the previous nodes and reallocating — invalidating (dangling) any pointer a consumer saved across `consume_input` calls. **Why deferred:** needs an append-only marker-list builder that mutates the existing list in place across `consume_input` calls instead of the rebuild-on-completion approach.

**Acceptance criteria.** A C harness (extending `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs`) that, against a fully-buffered `mem_src` of a multi-scan progressive JPEG: (a) asserts `jpeg_read_header` returns `JPEG_HEADER_OK` leaving `input_scan_number == 0`, then a driving loop of `jpeg_consume_input` reports `JPEG_REACHED_SOS` once per scan before `JPEG_REACHED_EOI`; (b) decodes a buffered-image stream calling only `jpeg_start_output`/`jpeg_read_scanlines`/`jpeg_finish_output` (no explicit `consume_input`) from a suspending source and still reaches a complete, byte-exact-vs-`djpeg` image; (c) saves a `jpeg_saved_marker_ptr` after the first `consume_input` and asserts it still points at the same marker contents after the stream completes. All three byte-exact vs stock `libjpeg`/`djpeg`.

**Why deferred.** P4-13's stated acceptance criteria (real suspending source → lock-step boundaries → byte-exact decode) are met and proven; these three are deeper upstream-contract-fidelity gaps with no known consumer, and gap (a) in particular is a behaviour change to every fully-buffered consumer that must not regress the verified byte-identical paths. Filed rather than expanding the P4-13 PR scope.

## P4-27. Single-Component Baseline With Non-1x1 Sampling Used Interleaved MCU Block Order — **CLOSED 2026-06-29**

**Motivation.** Scheduled Fuzz Smoke run 27930557237 (`fuzz_decode_diff_c`, commit `b60227c44ee14d0a713aaf4c043fafa8848d01ad`) found a 16x16 baseline grayscale fixture (`crash-a0fa322dc25942c53df490e8edce2d448580c9ad`) with SOF0 component sampling `h=1,v=4`. `djpeg` decoded a 16x16 P5 raster, while Rust diverged by `max abs diff = 240` (tolerance 24), with the second horizontal block decoded from the wrong entropy position.

**Root cause.** `decode_baseline_planes` only dispatched to the non-interleaved baseline path when `metadata.scans.len() > 1`. A single-component JPEG still uses non-interleaved one-block raster semantics: the entropy stream contains `ceil(width/8) * ceil(height/8)` data units for that component. Falling through to the interleaved MCU path made Rust honor the SOF sampling factors as MCU layout (`mcus_x * v_samp` blocks here), so it consumed and placed too many blocks in MCU sampling order instead of the component's encoded block raster.

**Fix.** Route any baseline SOS with one scan component through `decode_non_interleaved_baseline_planes`, even when there is only one SOS. That path already computes encoded block counts from the component sample dimensions and pre-fills unvisited padding with 128, matching libjpeg-turbo.

**Status (2026-06-29): closed.** Pinned by `tests/cross_check_fuzz_decode_diff_c_baseline_h4v1.rs::fuzz_decode_diff_c_baseline_gray_16x16_h1v4_matches_djpeg` (byte-identical vs `djpeg`, max diff 0). Exact repro passes: `cargo +nightly fuzz run fuzz_decode_diff_c /tmp/libjpeg-run-27930557237/fuzz-artifacts-fuzz_decode_diff_c/crash-a0fa322dc25942c53df490e8edce2d448580c9ad -- -runs=1`.

## P4-28. Progressive AC-Refine Wrote One-Past-Se Coefficients to the Padded Natural-Order Slot — **CLOSED 2026-06-29**

**Motivation.** Scheduled Fuzz Smoke run 28349528808 (`fuzz_decode_diff_c`, same commit `b60227c44ee14d0a713aaf4c043fafa8848d01ad`) found a 16x16 progressive 4:2:2 fixture (`crash-39e136ac088b7b2b3fc3786a4a23bae4d15ba632`) with clean C and Rust decodes but pixel divergence `max abs diff = 114` (tolerance 24). The mismatch was entirely in the right luma block; Cb/Cr were neutral and all RGB channels moved together.

**Root cause.** In `decode_ac_refine`, libjpeg-turbo writes a newly significant coefficient through `jpeg_natural_order[k]` even when the zero-run loop exits because `k > Se`. Rust treated every `k > Se` as an out-of-range soft landing and wrote to `coeff[63]`. That is only correct for libjpeg's padded natural-order entries `k=64..79`; for real zigzag positions `k < 64`, C still writes the actual natural coefficient. This artifact needed `k=60` (natural coefficient 47), but Rust wrote the `1` into natural coefficient 63.

**Fix.** In AC refinement, write `ZIGZAG_ORDER[k]` whenever `k < 64`; only route `64 <= k < 80` to the padded natural coefficient 63. This preserves the prior soft landing for genuinely padded writes while matching C for one-past-`Se` real coefficients.

**Status (2026-06-29): closed.** C coefficient dump (`jpeg_read_coefficients`) and Rust `read_coefficients` match on the artifact after the fix, and the decoded raster is byte-identical to `djpeg` (max diff 0). Pinned by `tests/cross_check_fuzz_decode_diff_c_progressive_16x16.rs::fuzz_decode_diff_c_progressive_16x16_h2v1_ac_refine_matches_djpeg`. Exact repro passes: `cargo +nightly fuzz run fuzz_decode_diff_c /tmp/libjpeg-run-28349528808/fuzz-artifacts-fuzz_decode_diff_c/crash-39e136ac088b7b2b3fc3786a4a23bae4d15ba632 -- -runs=1`.

## P4-29. Block Smoothing Read Dummy iMCU-Padding Blocks as DC Neighbors — **CLOSED 2026-07-08**

**Motivation.** Scheduled Fuzz Smoke run 28921468958 (`fuzz_decode_diff_c`, commit `9feb2ce`) found a 550-byte 16x16 progressive fixture (`crash-14a6d26e60c0bd5e6ebad664c3d443954776aa4e`) with luma sampling 1h x 4v where clean C and Rust decodes diverge by `max abs diff = 26` (tolerance 24). Coefficient dumps (`jpeg_read_coefficients` vs Rust `read_coefficients`) were identical, `coef_bits` matched C exactly, and the entire diff was confined to the bottom real luma block row with a smooth cosine-basis shape — block smoothing, not entropy decode.

**Root cause.** With 1h x 4v luma on a 16-pixel-high image, one iMCU row spans 4 luma block rows but only 2 are real; the interleaved DC scans populate the 2 dummy rows with decoded DC values (7 / -6 / 7 / -16 on this fixture). C's `decompress_smooth_data` iterates `block_rows` derived from `height_in_blocks` with row-neighbor guards driven by the scaled per-iMCU-row indices `image_block_row = output_iMCU_row * block_rows + block_row` / `image_block_rows = block_rows * total_iMCU_rows`, and clamps its column window at `width_in_blocks - 1`. On this single-iMCU-row shape that means dummy blocks are never smoothed nor read as neighbors. Rust's `apply_block_smoothing_coeffs` iterated and clamped over the iMCU-padded `blocks_x * blocks_y` grid, so the last real block row took the dummy rows as its "next"/"next-next" DC neighbors and predicted phantom AC11 = ±1 coefficients that C predicts as 0. Two further divergences fixed in the same pass (the first surfaced by the rust-code-reviewer pass): (a) on multi-iMCU-row components C's scaled indices behave differently from a naive real-grid clamp — non-final iMCU rows *do* read trailing dummy rows as `next-next`, and a partial last iMCU row clamps `prev`/`prev-prev` earlier than plain row arithmetic (a naive real-grid clamp measured max diff 2 vs djpeg on a 16x20 1h2v two-band fixture); (b) Rust wrote each smoothed workspace back into the coefficient buffer it also reads DC neighbors from, while C runs IDCT straight off the workspace and never writes back — in `change_dc` (DC-interpolation) mode later blocks would see already-smoothed neighbor DCs.

**Fix.** `apply_block_smoothing_coeffs` now takes the padded `row_stride` and `v_samp` separately from the real `blocks_x` / `blocks_y` (`width_in_blocks` / `height_in_blocks` at the `decode_progressive_planes` call site), smooths only the real grid, selects row neighbors through C's exact per-iMCU-row `image_block_row` / `image_block_rows` guards (dummy-row reads included where C performs them), and reads all 25 DC neighbors from a pre-pass snapshot of the padded grid's DC values.

**Status (2026-07-08): closed.** Decoded raster is byte-identical to `djpeg` on the artifact (max diff 0; measured pre-fix 26) and on the multi-iMCU-row fixture (max diff 0; naive clamp 2). Pinned by `tests/cross_check_fuzz_decode_diff_c_progressive_16x16.rs::fuzz_decode_diff_c_progressive_16x16_h1v4_smoothing_matches_djpeg` and `::progressive_16x20_multi_imcu_row_partial_last_smoothing_matches_djpeg`. Exact repro passes: `cargo +nightly fuzz run fuzz_decode_diff_c <artifact> -- -runs=1` with the crash file from `gh run download 28921468958 -n fuzz-artifacts-fuzz_decode_diff_c`.

## P4-30. Unsupported 12-Bit Sampling Layout Panicked During Plane Writes — **CLOSED 2026-07-12**

**Motivation.** Fuzz Smoke runs 117, 118, and 121 (run IDs `29009942928`, `29029935318`, and `29084489200`; commit `6500749`) independently reached the same panic through `fuzz_decompress_lenient` or `fuzz_decompress`. Each input declared 12-bit component sampling that is legal JPEG syntax but is not an integral divisor of the frame's maximum sampling factors.

**Root cause.** `decompress_12bit` assumed every component sampling factor evenly divided the frame maximum. The derived MCU-sized component planes and later upsampling factors rely on that invariant, but the decoder neither validated it nor checked the final 8x8 block write. Unsupported layouts could therefore calculate a block origin exactly one row beyond the allocated plane and panic on index `len`.

**Fix.** The 12-bit path now rejects zero sampling as corrupt and non-integral-divisor sampling as explicitly unsupported before MCU sizing. Every decoded block is also checked against its component plane before its samples are written. This closes the panic without claiming the broader non-standard-sampling support tracked by P4-21.

**Status (2026-07-12): closed.** The minimized regression `api::precision::tests::unsupported_12bit_sampling_layout_returns_error_instead_of_panicking` returns an error, and all three downloaded CI artifacts complete under their original fuzz targets with `-runs=1`. This is an error-path stability test and produces no decoded raster to compare with `djpeg`.

## P4-31. Real-World and Corpus CI Gates Can Silently Lose Coverage — **CLOSED 2026-07-25**

**Motivation.** A pre-release audit found that the committed 61-image real-world suite currently passes pixel-identically against libjpeg-turbo, but its harness converts a Rust panic (and some Rust arithmetic decode errors) into successful skips. The generated Corpus Test copies only top-level `.jpg` files, omitting nested `real_world/`, `kodak/`, `usc_sipi/`, and extensionless JPEG fuzz seeds. Its workflow parses a textual summary fail-open, permits arbitrary skips, and treats encode failures as warnings.

**Root-cause hypothesis.** The individual suites grew independently: real-world tests prioritized diagnostic continuity, the corpus copier assumed flat extension-bearing fixtures, and the CI shell gate accumulated count-based exceptions. Together those choices allow a missing directory, new Rust rejection, or oracle execution error to reduce effective coverage without failing CI.

**Acceptance criteria.** (1) Rust panics/errors in valid real-world decoding fail the test; only unavailable external C capabilities may skip. (2) Fixture inventory/category minimums prevent silent shrinkage. (3) Corpus generation recursively preserves nested fixtures and includes tracked extensionless fuzz files only when their bytes have a JPEG SOI signature, without modifying or incorporating local untracked fuzz corpus. (4) CMYK/YCCK/12-bit comparisons normalize both decoders to the same pixel format. (5) Malformed inputs use an exact path + operation + reason `ExpectedReject` classification; pre-existing valid-input divergences use an equally exact named `KnownMismatch` tied to an open LAST_MILE item. Every expected path-operation outcome must be exercised by a full corpus run; there is no arbitrary skip budget. (6) The corpus runner exits nonzero for every unclassified failure/crash/skip, CI requires all operations including encode, and source-bucket minimums prove real-world, generated, and fuzz inputs were exercised.

**Why release-blocking.** Version `0.6.3` is intended to publish decoder-stability fixes. Publishing while a green gate can silently omit real-world inputs or downgrade Rust failures would weaken the evidence for that claim; the tag remained blocked until P4-31 closed with PR and post-merge CI evidence.

**Status (2026-07-25): closed.** PR #307 (`test: harden real-world JPEG corpus gates`, merge commit `a3da54c`) delivered every acceptance criterion: real-world differential tests fail loudly with marker-based fixture-coverage minimums (1, 2); corpus generation recurses nested fixtures and admits tracked extensionless inputs by SOI signature only (3); CMYK/YCCK/12-bit comparisons are normalized to a shared pixel format (4); expected outcomes use exact `ExpectedReject`/`KnownMismatch` classifications with no skip budget (5); and the corpus runner exits nonzero on any unclassified failure while the `Corpus Test (C parity)` CI job gates on that exit code including encode operations (6). Post-merge CI evidence: the job is required in `ci.yml` and passed on post-#307 runs `30077205876` and `30147718997` (PR #310, both full sweeps green) and on the `main` push run for merge commit `ed00d30`. The `v0.6.3` tag is unblocked.

## P4-32. Valid Arithmetic-Progressive Grayscale Seed Diverges From `djpeg` — **CLOSED AS DUPLICATE OF P4-20 2026-07-13**

**Motivation.** P4-31's extensionless-seed coverage surfaced tracked fuzz input `fuzz/corpus/fuzz_decompress/24fd23785278a9577686f501e17ee8164f8b977b`. libjpeg-turbo 3.1.4.1 accepts it with `djpeg -strict`; the stream is a 144x16, one-component SOF10 arithmetic-progressive JPEG with four scans (`DC first`, two `AC first`, then `AC refine`). Rust completes decoding but differs from `djpeg` by max 34 beginning at pixel (0,0).

**Current evidence.** This is not malformed-input tolerance: the strict C oracle succeeds. Rust and C coefficient buffers and quantization tables match exactly, while the backend-specific pixel mismatch is unchanged by disabling block smoothing (max 34 on AArch64 NEON and 255 with scalar dispatch; x86 AVX2 can match C). Four blocks hit the rows-1–7-non-zero i16-overflow shape already tracked by P4-20, with maximum absolute dequantized coefficient 92280 after aligning zigzag coefficients to natural-order quantization entries. The separate tracked `crash-cf56...` rejection remains P4-21.

**Disposition.** Closed as a duplicate rather than creating a second decoder item. The exact seed is pinned in `tests/sof10_decode.rs`, the corpus reports it as a named P4-20 `KnownMismatch`, and P4-20 retains the eventual diff-to-zero acceptance criterion.

## P4-33. Checked-In `aspect_*` Fuzz Seeds Drifted From the Current Encoder — **CLOSED 2026-07-25**

**Motivation.** During the issue #308 work (2026-07-24), a full `cargo test --workspace` run left 14 tracked files modified: `tests/generate_fuzz_seeds.rs` regenerates `fuzz/corpus/*/aspect_wide_chk_420_base.jpg` and `aspect_wide_grad_420_base.jpg` (7 target dirs × 2 seeds) with different bytes than the committed versions (e.g. 830 → 922 bytes; headers identical, divergence starts in the entropy-coded data). Reproduced on a clean `main` checkout, so the drift predates the #308 branch.

**Root cause (final, 2026-07-25).** Two wrong hypotheses fell first: it was not a later encoder change (bisect over `a9dfca0..HEAD` found no such commit — `a9dfca0` itself already generates today's x86_64 bytes) and not a stale working tree. The PR #311 CI run supplied the missing evidence: the linux-aarch64 NEON and WASM jobs regenerate the same 14 seeds with **different bytes again** — encoder output is **backend-dependent** for the partial-MCU 8×64 4:2:0 shape. The committed 830-byte seeds were the author's aarch64/NEON output; every x86_64 run (CI and dev boxes) silently rewrote them to the 922-byte x86 encoding. All variants decode pixel-identically under `djpeg` (benign alternate encodings), so no encoder fix is required; the corpus just needs one canonical source.

**Resolution.** (1) Corpus canonicalized to **x86_64-linux** generator output (matches the primary CI runners); regenerated seeds committed. (2) `fan_out_write` now never overwrites an existing seed on non-canonical platforms — aarch64/WASM/dev-mac runs keep the committed bytes and stop churning the tree. (3) Drift guard on the canonical platform: overwrites-with-different-bytes are recorded in `DRIFTED_SEEDS` and `generate_seeds` asserts the list is empty, printing the offending paths — an x86_64 generator/encoder change that alters seed bytes fails the test (and CI) in the PR that introduces it, with regenerated files on disk ready to review + commit. Brand-new files are deliberately not flagged so fresh checkouts still generate cleanly.

**Status (2026-07-25): closed.** Proof: PR #311 — canonical guard fails with "14 committed fuzz seed(s) drifted" against the pre-fix seeds and passes after re-commit; linux-aarch64 NEON + WASM CI jobs (which regenerate different backend bytes) pass with the committed corpus untouched; old vs new seed decodes byte-equal via `djpeg -ppm` + `cmp`.

## P4-34. Transform Re-Encode Dropped Sparse DQT Slot References — **CLOSED 2026-07-24**

**Motivation.** Ten scheduled Fuzz Smoke `fuzz_transform_diff_c` failures between 2026-07-19 and 2026-07-24 (runs `29679993066`, `29715965508`, `29751094520`, `29773970372`, `29799281292`, `29815394302`, `29905093435`, `29928183424`, `30039214522`, `30064906856` — the last also carried the distinct P4-35) shared one root cause: djpeg rejected our transformed output with "Quantization table 0xNN was not defined" while accepting jpegtran's.

**Root cause.** `read_coefficients` collected the four DQT slots with `filter_map`, compacting them into a dense `Vec` while every component kept its *original* slot index. Any stream whose defined slots are not exactly `0..n` (only slot 1 defined; slots {0,1,3} with a gap at 2) re-encoded into a SOF that references a quantization table the output never defines.

**Fix.** `read_coefficients` now builds a slot→dense map and remaps each component's `quant_table_index` through it (undefined references fall back to dense index 0; djpeg rejects the C-side equivalent anyway). All writers keep the dense invariant they already assumed.

**Status (2026-07-24): closed.** All ten crash artifacts pass the harness pipeline (djpeg accepts, dimensions agree with jpegtran). Pinned by `tests/regression_transform_sparse_dqt_slots.rs` (slot-1-only and gap-at-2 fixtures; pure-Rust structural check plus djpeg/jpegtran cross-validation).

## P4-35. Category-16 Coefficients Re-Encoded Into an Undecodable Huffman Stream — **CLOSED 2026-07-24**

**Motivation.** Fuzz Smoke run `30064906856` (crash-7a0c14f3, 228x186 SOF10 arithmetic progressive): djpeg flagged our transformed output with "bad Huffman code" warnings (exit 2) while jpegtran's decoded cleanly.

**Root cause.** The arithmetic AC-first decode yields an AC coefficient of -32768 (`(v << Al)` wrap — C's jdarith.c wraps identically; verified byte-for-byte against `jpeg_read_coefficients`). Re-encoding it needs Huffman magnitude category 16, which the 4-bit size field of a DHT symbol cannot express; the symbol computation `(run << 4) | 16` bled into the wrong symbol and desynced the stream. C's scalar encoder rejects the coefficient with ERREXIT(JERR_BAD_DCT_COEF); its x86 SIMD path silently emits garbage.

**Fix.** Two parts. (1) The transcode writers (`write_coefficients_optimized` pass 1, progressive DC-first gather, progressive AC-first gather) now return `CorruptData("DCT coefficient out of range for Huffman coding")` on any category-16 value — in i16 storage that is exactly a stored/diffed -32768 (or `abs >> Al >= 0x8000` in AC-first) — matching the scalar C contract; the differential harness treats it as inconclusive-skip. (2) The arithmetic decoder gained jdarith.c's error-limbo semantics: spectral/magnitude overflow sets a poison flag (C: `ct = -1`) and every subsequent per-block decode is a no-op until `process_restart`, instead of continuing with corrupted coder state (DC overflow previously hard-errored where C tolerates).

**Status (2026-07-24): closed.** The crash artifact now takes the CorruptData skip path. Pinned by `tests/regression_transform_unencodable_coefficient.rs` (fixture must decode to an i16::MIN coefficient, transform must fail with CorruptData) and `src/decode/arithmetic.rs::error_limbo_tests` (overflow enters limbo, limbo decodes nothing, restart clears it).

## P4-36. Fractional Chroma Sampling Ratio Panicked in the Direct-Copy Upsample Path — **CLOSED 2026-07-24**

**Motivation.** Fuzz Smoke run `29977722126` (`fuzz_decompress_lenient`, crash-4a2b926c): 64x64 baseline with Y=4x1, Cb=3x1, Cr=1x1 panicked with `range end index 3088 out of range for slice of length 3072` at pipeline.rs:4333.

**Root cause.** The per-component upsample factor `y_width / cb_w` truncates 4/3 to 1, routing the component into the "no upsampling needed" direct-copy branch, which then reads luma-width rows out of a 3/4-width chroma plane. C rejects every non-integer sampling ratio up front with ERREXIT(JERR_FRACT_SAMPLE_NOTIMPL) ("Fractional sampling not implemented yet"; verified djpeg exit 1).

**Fix.** Reject non-integer luma/chroma plane ratios right after the zero-factor guard with `Unsupported("fractional chroma sampling ratio…")`, in both strict and lenient modes — djpeg fails fatally, so there is nothing more lenient to offer.

**Status (2026-07-24): closed.** Crash artifact completes under `cargo +nightly fuzz run fuzz_decompress_lenient <artifact> -- -runs=1`. Pinned by `tests/regression_decompress_fractional_sampling.rs` (strict + lenient error, djpeg-rejects cross-check).

## P4-37. SOS Component IDs Were Never Validated Against the Frame — **CLOSED 2026-07-24**

**Motivation.** Fuzz Smoke run `29815394302` (`fuzz_read_coefficients`, timeout-7a780449): a 16400x48 SOF10 arithmetic-progressive stream with 1371 scans where scan 8 (and many later scans) references component id 2 while the frame declares only id 1. C rejects at scan 8 in ~3 ms with ERREXIT(JERR_BAD_COMPONENT_ID, "Invalid component ID %d in SOS"); we decoded all 1371 scans (~670 ms native → 30 s+ libFuzzer timeout under instrumentation) and returned Ok.

**Root cause.** `read_sos` parsed component ids without binding them to frame components; the arithmetic-progressive scan loop silently skipped unmatched scans (huffman-progressive happened to reject deeper in the pipeline).

**Fix.** The marker reader now ports jdmarker.c get_sos binding: each scan component must match a distinct frame component (searching in frame order, skipping already-bound entries); no match is `CorruptData("Invalid component ID N in SOS")`. This also rejects duplicate CSi in one scan, which C treats identically.

**Status (2026-07-24): closed.** Both artifacts (slow-unit + timeout) now error in microseconds. Pinned by `tests/regression_sos_invalid_component_id.rs` (baseline unknown id, duplicate id, progressive unknown id; djpeg-rejects cross-check for all three).

## P4-38. Lossless Color Output Skipped the Point Transform and Wrapped at the Wrong Modulus — **CLOSED 2026-07-24**

**Motivation.** Fuzz Smoke run `29689718301` (`fuzz_decode_diff_c`, crash-e3ad88d5): 16x16 3-component lossless (SOF3, ids 'R','G','B', Adobe transform 0, predictor 1, point transform Al=2) decoded cleanly on both sides but diverged from djpeg on every pixel (max abs diff 189).

**Root cause.** Two C-parity gaps against jdlossls.c: (1) undifferencing wrapped modulo `2^precision - 1` instead of C's unconditional `& 0xFFFF`; (2) `lossless_output_color` never applied the `<< Al` output upscale (C `simple_upscale`) and saturated with `.min(255)` where C truncates via the `(_JSAMPLE)` cast. The grayscale output path already scaled correctly.

**Fix.** `undifference_row` wraps at 0xFFFF; `lossless_output_color` takes `pt` and emits `((sample << Al) & 0xFF)` per component, matching C's scaler for both the Huffman (SOF3) and arithmetic (SOF11) lossless paths.

**Status (2026-07-24): closed.** The artifact now decodes byte-exact vs djpeg (max diff 0, down from 189). Pinned by `tests/regression_lossless_point_transform.rs` (djpeg byte-exact cross-check plus a djpeg-pinned first-pixel probe that runs without C tools).

## P4-39. CMYK Encode Path Silently Drops Restart / Custom-Table Options and Rejects Optimize+Smoothing — **OPEN**

**Motivation.** Surfaced 2026-07-25 while refactoring the `compress_*` family onto a single `CompressParams` core (see [P4-40](#p4-40-encodepipelinesrs-is-10k-lines-of-copy-pasted-compress-variants--open)). The new characterization fixture `tests/fixtures/encode_pipeline_golden.txt` shows the option-carrying variants producing **byte-identical output to plain `compress`** on CMYK input — the options never reach the encoder. Tracked upstream as GitHub issue [#313](https://github.com/developer0hye/libjpeg-turbo-rs/issues/313).

**Root cause.** CMYK support was implemented exactly once, as `compress_cmyk(pixels, width, height, quality, subsampling)` — a signature that cannot express restart intervals, custom tables, smoothing, or DCT method. Every other variant early-returns into it and silently discards its remaining parameters:

```rust
if pixel_format == PixelFormat::Cmyk {
    return compress_cmyk(pixels, width, height, quality, subsampling);
}
```

Five defects, four of them silent (`Ok(bytes)` with the option not applied):

1. `compress_with_restart` drops `restart_interval` (`src/encode/pipeline.rs:1267`) — no RST markers emitted.
2. `compress_custom_quant` drops custom quantization tables (`:1021`).
3. `compress_custom_huffman` drops custom Huffman tables (`:788`).
4. `compress_optimized` rejects CMYK with `JpegError::Unsupported`. Because `Encoder` routes through it for both `optimize_huffman(true)` (`src/api/encoder.rs:918`) and `smoothing_factor(>0)` (`:965`), **both builder options fail outright on CMYK**.
5. `compress_cmyk` ignores `dct_method` (`:123`) — `IsLow`/`IsFast`/`Float` are byte-identical.

**Divergence from C.** None of these are colorspace-gated upstream: `optimize_coding` (`jcmaster.c:595-802`, `jcinit.c:83-127`), `restart_interval` (`jchuff.c:693-876`), `smoothing_factor` (`jcsample.c:509-553`, per-component with a `smoothok` fallback, not a colorspace gate), quantization slots (`jcparam.c`), and `dct_method` (`jcdctmgr.c`). `cjpeg -optimize`, `-smooth N`, `-restart N`, `-qtables` and `-dct fast|float` all apply to CMYK/YCCK in C.

**Acceptance criteria.**

1. CMYK honours `restart_interval`, custom quant tables, custom Huffman tables, `smoothing_factor`, `optimize_huffman`, and `dct_method`.
2. Byte-exact cross-validation against C for each option on CMYK input.
3. The six regression tests in `tests/encode_cmyk_option_parity.rs` un-`#[ignore]`d and green (all six fail today when run with `--include-ignored`, which is the reproduction).
4. `tests/fixtures/encode_pipeline_golden.txt` regenerated in the *fixing* commit — never in a refactor commit — with the CMYK rows reviewed as a diff.

**Why deferred.** The P4-40 refactor is deliberately byte-exact, so it cannot carry a behavioural fix. It does remove the structural cause: once CMYK is a component-layout choice inside one core rather than a separate narrower function, these become small changes rather than five parallel edits.

## P4-40. `src/encode/pipeline.rs` Is 10k Lines of Copy-Pasted `compress_*` Variants — **PARTIAL: acceptance criteria 1-3 delivered; criterion 4 (split by mode) remains**

**Motivation.** Filed 2026-07-25 from a structural review. `src/encode/pipeline.rs` is 10,647 lines holding 103 free functions, 1 struct and **0 `impl` blocks**, and absorbs 108 of the last 1,174 commits (`src/decode/pipeline.rs` another 118) — the repo's highest size×churn product. Ten public `compress_*` entry points are copy-pasted variants of one algorithm; normalized unique-line overlap is 85% (`compress` vs `compress_with_restart`), 84% (vs `compress_custom_quant`) and 71% (vs `compress_custom_huffman`).

**Realized cost** — this is not a style complaint; the divergence has already produced defects:

- The fused single-pass color-convert path (`:192-207`, keeps data in L1/L2 between conversion and encode) exists **only** in `compress()`. Every other variant still calls full-plane `convert_to_ycbcr`, so restart-interval and custom-table encodes silently run the slow path — against the project's stated goal of matching or beating C.
- `smoothing_factor` is implemented **only** in `compress_optimized`.
- `compress()` clamps scaled quant values to 255, breaking `cjpeg` parity below q≈20. Rather than fix it, `src/api/encoder.rs:770-791` routes around it with the heuristic `q < 50` and the comment *"Use a generous threshold to be safe."*
- All five CMYK defects in [P4-39](#p4-39-cmyk-encode-path-silently-drops-restart--custom-table-options-and-rejects-optimizesmoothing--open).
- 36 of the project's 65 `#[allow(clippy::too_many_arguments)]` are in this one file; `compress_optimized` takes 9 positional parameters.

**Acceptance criteria.**

1. A `CompressParams` value type carries the full option set; one core routine subsumes `compress`, `compress_with_restart`, `compress_custom_quant`, `compress_custom_huffman` and `compress_optimized`, which become thin shims retaining their exact public signatures.
2. The refactor is **byte-exact**: `cargo test --test encode_pipeline_golden` passes unchanged against the pre-refactor fixture (20,160 pinned cases).
3. Every optimization reachable from every option combination — no feature is available on only one branch.
4. `src/encode/pipeline.rs` split by mode (baseline / progressive / arithmetic / lossless / downsample / quant-divisors) so no single file exceeds ~2k lines.

**Status (2026-07-25): criteria 1-3 closed.** `CompressParams` + `compress_with_params` landed; `compress`, `compress_with_restart`, `compress_custom_quant` and `compress_custom_huffman` are now 4-to-14-line shims over it, with their public signatures unchanged. Criterion 2 was met in two steps: routing `compress()` alone through the core moved **0 of 20,160** golden cases, and switching the other three moved 904 — every one of them verified against `cjpeg` (all four paths now 576/576 on the geometry sweep, versus 516 / 372 / 372 / 372 before). Criterion 3 is pinned by `tests/encode_params_composability.rs`, which exercises combinations no previous entry point could express (restart + custom quant + custom Huffman + `ifast` in one encode) and asserts each option's effect is independent of the others. Net -412 lines in the file, and it gains its first two `impl` blocks. Perf unchanged on the fast path (433 MP/s at 1920x1080, same as before). This also closed P4-42 outright and reduces P4-39 to a small change.

**Remaining.** Criterion 4 (split the file by mode) plus folding `compress_optimized`'s two-pass algorithm into the same parameter type — it is a genuinely different algorithm and was already C-correct (576/576), so it was left alone deliberately rather than forced into a shared shape.

**Sequencing.** Criteria 1–3 first (byte-exact, low risk, unblocks P4-39). Criterion 4 is mechanical and can follow. Deliberately excluded: `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` is also large (9,242 lines) but is 164 flat `extern "C"` shims mirroring a C header — its size is inherent, not tangled, and it needs at most a split by API family.

## P4-41. AVX2 4:2:0 Row Fast Path Ignored the Dummy-Block Contract and Skipped Its Own Capability Check — **CLOSED 2026-07-25**

**Motivation.** Found 2026-07-25 while building the P4-40 characterization fixture, which showed `compress()` (fused) and `compress_with_restart()` (full-plane) disagreeing on 176 of 1440 RGB cases. A 576-case geometry sweep against stock `cjpeg` (widths x heights over `{7,8,15,16,17,23,24,31,32,33,48,64}` x 4 subsamplings, q50) scored fused 516/576 and full-plane 372/576 — so at most one could be right, and neither was fully. GitHub [#314](https://github.com/developer0hye/libjpeg-turbo-rs/issues/314) and [#315](https://github.com/developer0hye/libjpeg-turbo-rs/issues/315).

**Root cause (two defects in one `if`).** `src/encode/pipeline.rs:373`:

1. **#314 — missing dummy-column guard.** The fast path FDCTs every block of every MCU unconditionally. It guarded the last partial MCU *row* (`eff_row_height == y_mcu_height`) but had no guard for the last partial MCU *column*, so for any width with `ceil(width/8)` odd it transformed replicated edge pixels where C emits a zeroed dummy block carrying the previous block's DC (`jccoefct.c:292-312`). Affected ordinary photo sizes — 500x375, 1000x750, 1000x1000, 1080x1080 all diverged; files ran 0.7–0.9% larger than C's. Only 4:2:0; 4:4:4 / 4:2:2 / 4:4:0 were clean.
2. **#315 — capability check bypassed.** The path gated on `!cb_half.is_empty()`, an allocation-shape proxy: `cb_half` was allocated whenever the subsampling was 4:2:0 but only *filled* under `is_x86_feature_detected!("avx2")`. On a non-AVX2 x86_64 CPU that reached `#[target_feature(enable = "avx2")]` helpers (undefined behaviour) with never-downsampled all-zero chroma.

**Why it escaped.** Three independent reasons, each worth noting: (a) the fast path is x86_64+AVX2-only and the project was developed on aarch64, which always takes the generic path — the same platform-dependent drift class as P4-33; (b) the byte-exact encoder cross-check (`tests/cross_check_encoder_binary.rs:76-77`) used **only 48x48**, MCU-aligned at 4:2:0, so `ndummy == 0` and the bug was unreachable; (c) `assert_encoder_output_matches` (`:36-62`) falls back from byte comparison to a decoded-pixel comparison with `max_diff <= 1`, and dummy blocks lie outside the image and are cropped on decode — the pixel diff is 0, so that test could never have caught it at *any* size.

**Fix.** A single `use_avx2_420` capability flag now gates the buffer allocation, the downsample and the encode fast path, so they cannot disagree; and the fast path additionally requires `y_last_col_width == y_mcu_width`. Partial geometries fall through to the generic path, which already handled dummies via `encode_color_mcu_with_dummies`.

**Status (2026-07-25): closed.** The fused path now matches `cjpeg` on **576/576** swept geometries (was 516/576) and on all 24 real-world cases (500x375 … 1920x1080 x 4 subsamplings). Pinned by `tests/regression_420_dummy_block_columns.rs` — a `cjpeg` byte-exact check over 10 geometries plus 6 C-tool-free length+hash pins taken from `cjpeg`-verified output. Cost ~3.2–4.5% throughput on affected widths (medians of 15 runs x 3 repeats; recorded in `experiments/x86_64_pipeline.tsv`), tracked for recovery as [#317](https://github.com/developer0hye/libjpeg-turbo-rs/issues/317) / P4-43. Note #315's acceptance criterion 2 — a CI leg that masks AVX2 at the CPUID level — is **not** delivered here; the UB is removed by construction but remains untested on this hardware.

## P4-42. Full-Plane Encode Variants Skip the Dummy-Block Contract on Every Platform — **CLOSED 2026-07-25**

**Motivation.** Filed 2026-07-25 alongside P4-41. `compress_with_restart`, `compress_custom_quant` and `compress_custom_huffman` use full-plane `convert_to_ycbcr` and never implement C's dummy-block contract, so they diverge from `cjpeg` on **204 of 576** swept cases — unlike P4-41 this is **not** platform-gated. GitHub [#316](https://github.com/developer0hye/libjpeg-turbo-rs/issues/316).

Divergences by subsampling: 4:2:0 → 108, 4:2:2 → 72, 4:4:0 → 24; 4:4:4 clean. The failing set is exactly C's `ndummy > 0` condition — at 4:2:2 it fails for width ∈ {7,8,17,23,24,33} (`ceil(w/8)` odd) and passes for {15,16,31,32,48,64} (even); at 4:4:0 it fails for height ∈ {8,24}. `restart_interval = 0` throughout, so the buffering strategy is the only variable.

`Encoder` routes to these whenever a restart interval, custom quant tables or custom Huffman tables are requested (`src/api/encoder.rs:930-952`), so any such encode at a partial-MCU geometry is non-conformant.

**Acceptance criteria.**

1. Restart / custom-quant / custom-Huffman encodes byte-identical to the equivalent `cjpeg` invocation at partial-MCU geometries for 4:2:0, 4:2:2 and 4:4:0.
2. Regression coverage over `ceil(width/8)` odd and `ceil(height/8)` odd, not only MCU-aligned sizes.
3. Satisfied by routing every variant through one code path (the P4-40 `CompressParams` core) rather than by patching dummy-block logic into each copy.

**Status (2026-07-25): closed.** Delivered by criterion 3 — the three variants are now shims over `compress_with_params`, so they inherit the fused strategy's dummy-block handling instead of carrying their own. All three match `cjpeg` on **576/576** swept geometries (was 372/576), and `compress_with_restart(ri=3)` additionally matches `cjpeg -restart 3B` byte-for-byte on all 576. Pinned by `issue_316_full_plane_variants_match_cjpeg_at_partial_mcus` in `tests/regression_420_dummy_block_columns.rs` (48 cjpeg cross-checks over 3 subsamplings x 8 geometries x {ri=0, ri=3}).

## P4-43. Recover the AVX2 4:2:0 Fast Path for Interior MCU Columns — **OPEN**

**Motivation.** The P4-41 fix disables the fast path for the whole image when the last MCU column needs dummy blocks (`ceil(width/8)` odd — roughly half of all widths), costing ~3.2–4.5%. GitHub [#317](https://github.com/developer0hye/libjpeg-turbo-rs/issues/317).

Measured (EPYC 9554, medians of 15 runs, 3 repeats, `examples/bench_encode_420_geometry.rs`), width pairs 8px apart so only `ceil(w/8)` parity changes: 1008x750 fast 433.1 MP/s vs 1000x750 generic 413.6 (-4.5%); 1920x1080 fast 432.3 vs 1928x1080 generic 416.7 (-3.6%); 3840x2160 fast 432.6 vs 3848x2160 generic 418.7 (-3.2%).

**Why deferred.** The fast path hoists one `begin_block`/`end_block` pair across the whole MCU row and writes through a raw `(pb, fb, buf)` triple, while the dummy path writes through `bit_writer`; splitting a row between them is not a local change. Correctness shipped first.

**Acceptance criteria.** Throughput at `ceil(width/8)` odd within ~1% of the even-width case, with the 576-case sweep and the golden fixture unchanged.

## P4-44. Encoder Byte-Parity Against `cjpeg` Is Unmeasured for `ifast` / `float` and for aarch64 — **OPEN**

**Motivation.** Filed 2026-07-25 from PR #318 CI. The x86_64 encoder is now byte-identical to stock `cjpeg` on 576/576 swept geometries (P4-41, P4-42) — but only for `-dct int`. The `linux-aarch64 NEON` job failed the x86_64-pinned golden fixture with divergences clustered in `ifast` and `float`: at 16x16 BGR 4:2:0 q100, x86_64 emits 954 bytes and aarch64 955 (`float`), 944 vs 956 (`ifast`); at 4:2:2 q100 `ifast`, 1058 vs 1078. GitHub [#319](https://github.com/developer0hye/libjpeg-turbo-rs/issues/319).

P4-33 established the phenomenon (backend-dependent output, decodes pixel-identically under `djpeg`) and canonicalized the fuzz corpus on x86_64-linux; PR #318 follows the same precedent for its byte fixtures. So nothing is broken by the current definition — but "decodes pixel-identically" is weaker than the byte-exactness this project targets, and the gap has never been quantified.

**Two open questions.** (a) Does aarch64 `islow` match `cjpeg` at partial-MCU geometries? The `*_matches_cjpeg` tests added in P4-41/P4-42 run unguarded on every platform, so the next aarch64 CI run answers this for 4:2:0. (b) Does *either* backend match `cjpeg -dct fast` / `-dct float`? The existing cross-checks only ever pass `-dct int`, so it is possible x86_64 diverges here too and nobody has looked.

**Acceptance criteria.**

1. A measured answer for each (backend x DCT method) pair against `cjpeg` — cheapest first: extend the sweep in `examples/probe_fused_vs_fullplane.rs` to `-dct fast` / `-dct float` and run it on x86_64, which settles (b) with no aarch64 hardware.
2. The same sweep as a CI step on the `linux-aarch64 NEON` job, which already installs official libjpeg-turbo 3.1.4.1 and so has `cjpeg` available.
3. Whatever byte-exactness the project actually guarantees stated in `docs/FEATURE_PARITY.md` and enforced per backend. Concluding "byte-exact for `islow`, pixel-accurate for `ifast`/`float`" is an acceptable outcome — the requirement is that it be a documented decision rather than an unexamined assumption.

## Phase 4 Suggested Order

1. ~~**P4-1** — export `jpeg_calc_jpeg_dimensions` and delete its missing-symbol allowlist entry.~~ **CLOSED 2026-05-10**.
2. ~~**P4-2** — file the T1–T4 replacement-tier framing across README, LAST_MILE, and ABI_COMPATIBILITY.~~ **CLOSED 2026-05-17**.
3. ~~**P4-3** — flip the default C-ABI SONAME from `libjpeg.so.62` to `libjpeg.so.8` and gate v6b behind `CAPI_ACK_V6B_SONAME=1`.~~ **CLOSED 2026-05-17**.
4. **P4-4** — panic guard (`unwind_guard!` macro) on every `pub extern "C"` entry point + `tests/capi_panic_safety.rs`. Branch `feat/capi-panic-boundary`.
5. **P4-5** — pathological classic-lifecycle coverage (`tests/capi_classic_lifecycle_pathological.rs`, ≥10 patterns).
6. **P4-6** — reconcile FEATURE_PARITY wording with the P4-5 evidence (or refile residual gaps as named P4-* OPEN).
7. **P4-7** — `tj3GetICCProfile` `TJERR_WARNING` soft-error path + stale-stub / divergence comment sweep across capi src.
8. ~~**P4-8** — runtime BMI1+LZCNT dispatch for x86_64 encode (audit + README update; the dispatch itself was already live).~~ **CLOSED 2026-05-17**.
9. **P4-13** — true streaming `jpeg_consume_input` / `JPEG_SUSPENDED` semantics (filed 2026-05-18 after cold review surfaced `jpeglib.rs:4234-4238` fully-buffered EOI shim). **PARTIAL 2026-06-02** — incremental body drain (Option b) landed + byte-exact-proven: `consume_input` reports SOS/EOI/SUSPENDED in lock-step with a suspending source; decode stays buffered. Test: `consume_input_suspends_through_progressive_body`. Three deeper streaming-contract gaps (read_header-stop-at-SOS for all sources, output-driven input pull, incremental marker-list stability) deferred to **P4-26**.
10. **P4-14** — `max_memory_to_use` enforcement in C-side virtual-array path (filed 2026-05-18; field is ABI-mirrored but never consulted).
11. ~~**P4-15** — `jpeg16_*_raw_data` parity audit.~~ **CLOSED 2026-05-18** — mirrors upstream's 8/12-only raw-data API; no action.
12. ~~**P4-16** — thread-affinity contract on `cinfo` TLS side tables (filed 2026-05-18; pick Option A fix vs Option B document on adoption-pressure signal).~~ **CLOSED 2026-05-19** via Option B — new "Threading contract" section in `docs/ABI_COMPATIBILITY.md` documents the per-thread ownership requirement authoritatively. Option A (RwLock migration) remains tracked in the P4-16 body for future adoption pressure.
13. ~~**P4-17** — real `JPEG_SUSPENDED` test in `capi_classic_lifecycle_pathological.rs` (filed 2026-05-18; the existing `source_mgr_suspends_every_byte` exercises chunked refill, not suspension).~~ **CLOSED 2026-06-02** — `consume_input_suspends_through_progressive_body` (the P4-13 harness) is the real-suspension test.
14. ~~**P4-18** — file 18 legacy TurboJPEG 1.x/2.x aliases as implemented-and-allowlist-removed (Option A) or permanently-deferred-with-rationale (Option B). Filed 2026-05-18 after cold review found the legacy-TJ subset of the allowlist was missed by P3-3's closure scope.~~ **CLOSED 2026-05-19** via Option B — per-symbol migration matrix + tiny-shim recipe in `docs/ABI_COMPATIBILITY.md` under `### Legacy TurboJPEG 1.x/2.x aliases — partial coverage (P4-18)`. Option A (wire all 18 into `legacy.rs`) remains in the P4-18 body for future adoption pressure.
15. ~~**P4-19** — IDCT `islow` diverged from djpeg on i16-overflow (corrupt) coefficients (AC-all-zero `psllw` wrap shortcut).~~ **CLOSED 2026-05-30** — shortcut routed to scalar; full-path SSE2 residue refiled as P4-20.
16. **P4-20** — x86 SSE2 IDCT full path is i32 4-lane, not an i16-faithful port (corrupt-input-only; AVX2 already faithful, so CI-invisible). Lower urgency.
17. **P4-21** — decoder rejects non-standard sampling where a chroma component out-samples luma (`Cr=h3v1`); colour-path refactor (A) or lenient recovery (B).
18. ~~**P4-22** — decoder diverges from libjpeg-turbo on multi-scan non-interleaved baseline (never-scanned luma + doubly-scanned Cr).~~ **CLOSED 2026-05-31** — non-interleaved baseline planes pre-filled with 128 (IDCT-of-zero) so never-scanned components / padding match djpeg. Regression: `tests/cross_check_fuzz_decode_diff_c_multiscan.rs`.
19. ~~**P4-23** — lenient mode rejects corrupt baseline entropy ("invalid Huffman code") djpeg silently conceals.~~ **CLOSED 2026-05-31** — added lenient gray-fill + warning recovery to `decode_non_interleaved_baseline_planes` (corrupt streams fragment into spurious non-interleaved scans). Regression: `tests/cross_check_fuzz_decode_diff_c_multiscan.rs`.
20. ~~**P4-24** — arithmetic sequential (SOF9) non-interleaved multi-scan: `decode_arithmetic_planes` has no multi-scan dispatch (drops scans, 0-fill not 128, `unwrap_or(0)` Cs-misroute).~~ **CLOSED 2026-06-01** — added `decode_arithmetic_multiscan_planes` (per-scan `ArithDecoder`, 128 fill, reject unknown `Cs`; handles both non-interleaved and partially-interleaved scan scripts). Regression: `tests/cross_check_arith_noninterleaved.rs` (byte-exact vs djpeg).
21. **P4-25** — arithmetic DAC conditioning not snapshotted per scan (shared global read by all arith paths; same-slot redefinition between scans would mis-decode). Pre-existing; found in the P4-24 review. Not `fuzz_decode_diff_c`-reachable. Filed 2026-06-01.
22. **P4-26** — deeper streaming-contract fidelity beyond the P4-13 core: (a) `jpeg_read_header` stop-at-first-SOS for all sources (not just suspending), (b) buffered-image output calls pull input from the source manager, (c) `marker_list` extended in place instead of rebuilt (stable `jpeg_saved_marker_ptr`). No known consumer; none block T3; each needs a consumer-risky refactor. Filed 2026-06-02 from the P4-13 codex round-8 review.
23. ~~**P4-27** — single-component baseline with non-1x1 sampling used interleaved MCU block order.~~ **CLOSED 2026-06-29** — baseline one-component SOS now routes through non-interleaved block-raster decode. Regression: `fuzz_decode_diff_c_baseline_gray_16x16_h1v4_matches_djpeg`.
24. ~~**P4-28** — progressive AC-refine wrote one-past-`Se` real coefficients to padded coefficient 63.~~ **CLOSED 2026-06-29** — AC refine now uses real zigzag positions for `k < 64`, padded slot only for `64..79`. Regression: `fuzz_decode_diff_c_progressive_16x16_h2v1_ac_refine_matches_djpeg`.
25. ~~**P4-29** — block smoothing read dummy iMCU-padding blocks as DC neighbors (and wrote smoothed blocks back into the neighbor-read buffer).~~ **CLOSED 2026-07-08** — smoothing now iterates and clamps at the real `width_in_blocks`/`height_in_blocks` grid over a `row_stride`-pitched buffer, reading neighbor DCs from a pre-pass snapshot. Regression: `fuzz_decode_diff_c_progressive_16x16_h1v4_smoothing_matches_djpeg`.
26. ~~**P4-30** — unsupported 12-bit sampling layout could write beyond a component plane.~~ **CLOSED 2026-07-12** — validate the 12-bit upsampling invariant before decode and bounds-check each block write. Regression: `unsupported_12bit_sampling_layout_returns_error_instead_of_panicking`; exact replays for Fuzz Smoke 117, 118, and 121.
27. ~~**P4-31** — harden real-world/corpus coverage against soft-skipped Rust failures, nested-fixture omission, extensionless JPEG-seed omission, and fail-open summary parsing.~~ **CLOSED 2026-07-25** — delivered by PR #307; Corpus Test (C parity) gate green post-merge.
28. ~~**P4-32** — triage the strict-C-valid SOF10 grayscale seed `24fd237...`.~~ **CLOSED AS DUPLICATE OF P4-20 2026-07-13** — coefficients and quantization match C; the max-34 pixel delta is the existing IDCT i16-fidelity family.
29. ~~**P4-33** — verify the encoder change behind the 8×64/64×8 4:2:0 output drift, re-commit the regenerated `aspect_*` seeds, and add a drift guard so `cargo test` on a clean checkout leaves the tree clean.~~ **CLOSED 2026-07-25** — no encoder change existed: the bytes are backend-dependent (aarch64/NEON vs x86) for this partial-MCU shape and the committed seeds were aarch64 output. Corpus canonicalized to x86_64-linux; non-canonical platforms never overwrite; canonical drift guard asserts zero rewritten seeds.
30. ~~**P4-34** — transform re-encode dropped sparse DQT slot references.~~ **CLOSED 2026-07-24** — slot→dense remap in `read_coefficients`; ten Fuzz Smoke crashes resolved.
31. ~~**P4-35** — category-16 coefficients re-encoded into an undecodable Huffman stream.~~ **CLOSED 2026-07-24** — scalar-C JERR_BAD_DCT_COEF contract in the transcode writers + jdarith.c error-limbo port.
32. ~~**P4-36** — fractional chroma sampling ratio panicked in the direct-copy upsample path.~~ **CLOSED 2026-07-24** — C JERR_FRACT_SAMPLE_NOTIMPL parity guard.
33. ~~**P4-37** — SOS component ids never validated against the frame.~~ **CLOSED 2026-07-24** — jdmarker.c get_sos binding port; 1371-scan timeout stream now rejected at scan 8.
34. ~~**P4-38** — lossless color output skipped the point transform and wrapped at the wrong modulus.~~ **CLOSED 2026-07-24** — 0xFFFF undifference wrap + `<< Al` truncating output scaler; byte-exact vs djpeg.
35. **P4-39** — CMYK encode path silently drops restart / custom quant / custom Huffman options and rejects optimize+smoothing (GitHub #313). Blocked on nothing; P4-40's core makes it small.
36. **P4-40** — collapse the ten copy-pasted `compress_*` variants onto a single `CompressParams` core (byte-exact), then split `src/encode/pipeline.rs` by mode.
37. ~~**P4-41** — AVX2 4:2:0 row fast path ignored the dummy-block contract (#314) and bypassed its own AVX2 capability check (#315).~~ **CLOSED 2026-07-25** — single `use_avx2_420` gate + `y_last_col_width == y_mcu_width` guard; fused path now 576/576 vs cjpeg.
38. ~~**P4-42** — full-plane encode variants (restart / custom-quant / custom-Huffman) skip the dummy-block contract on every platform (#316).~~ **CLOSED 2026-07-25** — the P4-40 core made them shims; 576/576 vs cjpeg.
39. **P4-43** — recover the ~3-4.5% the P4-41 correctness fix cost on `ceil(width/8)`-odd 4:2:0 widths (#317).
40. **P4-44** — quantify encoder byte-parity vs `cjpeg` for `ifast`/`float` and for the aarch64 backend, then document what is actually guaranteed (#319).
