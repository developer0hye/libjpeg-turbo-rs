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
| P4-13 | OPEN (filed 2026-05-18) |
| P4-14 | OPEN (filed 2026-05-18) |
| P4-15 | CLOSED 2026-05-18 |
| P4-16 | CLOSED 2026-05-19 (Option B: documented in ABI_COMPATIBILITY.md) |
| P4-17 | OPEN (filed 2026-05-18) |
| P4-18 | CLOSED 2026-05-19 (Option B: deprecate-with-rationale, migration matrix in ABI_COMPATIBILITY.md) |
| P4-19 | CLOSED 2026-05-30 (IDCT `islow` i16-overflow AC-all-zero shortcut → scalar; full-path SSE2 residue refiled as P4-20) |
| P4-20 | OPEN (filed 2026-05-30) |
| P4-21 | OPEN (filed 2026-05-30) |
| P4-22 | CLOSED 2026-05-31 (non-interleaved baseline plane init 0→128) |
| P4-23 | CLOSED 2026-05-31 (lenient recovery in non-interleaved baseline path) |
| P4-24 | CLOSED 2026-06-01 (arithmetic multi-scan support: non-interleaved + partial-interleaved) |
| P4-25 | OPEN (filed 2026-06-01; P4-24 review) |

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

## P4-13. `jpeg_consume_input` Returns EOI Instead of Honoring Per-Byte Source Suspension — **OPEN**

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:4234-4238` shows `jpeg_consume_input` is a fully-buffered shim: once the header is parsed it returns `JPEG_REACHED_EOI` unconditionally, never `JPEG_SUSPENDED` from the body. The in-source comment admits the choice: *"For our fully-buffered shim, EOI is the truthful answer the moment a header is in hand."* The state-machine advance to `DSTATE_SCANNING` at `:4253` is a polling-loop terminator, not real input-exhaustion signalling. Upstream's contract is the reverse: `jpeg_consume_input` processes whatever bytes the source manager has produced and returns `JPEG_SUSPENDED` if it cannot complete a marker / SOS / scan boundary without more bytes, letting a chunked-source consumer (network image viewer, GStreamer-style multimedia pipeline, custom source manager with `fill_input_buffer` returning `FALSE`) drive the state machine in lock-step with arriving bytes.

**Acceptance criteria.**

- A C harness in `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs` that:
  1. Installs a custom `jpeg_source_mgr` whose `fill_input_buffer` returns `FALSE` when its drip-feed buffer is empty (real suspension — not the chunked-refill pattern flagged in P4-17).
  2. Drives `jpeg_consume_input` through the body of a multi-scan progressive JPEG, asserting `JPEG_SUSPENDED` returns when the drip buffer is empty and `JPEG_REACHED_SOS` / `JPEG_REACHED_EOI` when scan boundaries / EOI are observed.
  3. Resumes after each `JPEG_SUSPENDED` by refilling the buffer; the final `cinfo->global_state` must equal `DSTATE_STOPPING` after `jpeg_finish_decompress`.
- Bit-exact comparison of the resumed decode against the same JPEG decoded with the upstream linked-against-stock `libjpeg.so.8`.

**Why deferred.** P3-5 (2026-05-08) chose the "fully-buffered shim with EOI sentinel" path to make the buffered-image polling idiom (`while !jpeg_input_complete() consume_input()`) terminate without rewriting the decode driver to be re-entrant. Closing P4-13 requires either (a) lifting the internal decoder to a resumable state-machine, or (b) buffering the source-mgr output into our own internal buffer until the existing decoder accepts it. Both are non-trivial; documenting and gating the divergence in `ABI_COMPATIBILITY.md` until the larger refactor is also an acceptable interim.

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

## P4-17. `source_mgr_suspends_every_byte` Test Exercises Chunked-Refill, Not Real Suspension — **OPEN**

**Motivation.** An independent cold review (codex, 2026-05-18) spot-checked the P4-5 closure and found that `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle_pathological.rs:279-292`'s `slow_fill` returns `TRUE` after copying one byte. The C-side source comment at `:288-290` admits the design: *"Return TRUE so the decoder consumes the one byte. The 'suspends every byte' character comes from the fact that we'll need to be called again for the NEXT byte."* This is single-byte chunked refill, **not suspension**. A real `JPEG_SUSPENDED` test requires `fill_input_buffer` to return `FALSE` when the buffer is genuinely empty, asserting the public API surfaces `JPEG_SUSPENDED` and then refilling + resuming bit-exactly. P4-5's closure block (above) calls this pattern "suspends every byte"; it does not.

**Acceptance criteria.**

- A 4th pattern in `capi_classic_lifecycle_pathological.rs` (call it `source_mgr_returns_false_until_refilled`) where:
  1. The custom `jpeg_source_mgr`'s `fill_input_buffer` returns `FALSE` when its drip buffer is empty.
  2. The driver alternates between calling a public state-machine entry point (`jpeg_read_header` / `jpeg_consume_input` / `jpeg_read_scanlines`) and refilling the drip buffer.
  3. The state-machine entry points must return `JPEG_SUSPENDED` (= 2) whenever the drip buffer is empty, and the documented continue values when bytes are present.
  4. Final output is bit-exact against the same JPEG decoded without the drip filter.
- P4-5's closure block (above) must be updated to call its current first pattern "single-byte chunked refill (NOT real suspension; see P4-17)" so the false-positive is caught in print.

**Why deferred.** Discovered by the independent second review (2026-05-18); P4-5 was closed 2026-05-17 on the strength of three patterns that the closer assumed exercised suspension semantics. P4-13 (`jpeg_consume_input` EOI shim) is the upstream-side fix; P4-17 is the test that would have caught the gap earlier.

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

**Why deferred.** Not yet observed in a fuzz finding (P4-19's crash was the AC-all-zero shortcut, now fixed). Filed proactively because the differential fuzzer may eventually reach the full-path overflow. AVX2 — the path most CI x86 runners actually use — is already correct, lowering urgency.

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

## Phase 4 Suggested Order

1. ~~**P4-1** — export `jpeg_calc_jpeg_dimensions` and delete its missing-symbol allowlist entry.~~ **CLOSED 2026-05-10**.
2. ~~**P4-2** — file the T1–T4 replacement-tier framing across README, LAST_MILE, and ABI_COMPATIBILITY.~~ **CLOSED 2026-05-17**.
3. ~~**P4-3** — flip the default C-ABI SONAME from `libjpeg.so.62` to `libjpeg.so.8` and gate v6b behind `CAPI_ACK_V6B_SONAME=1`.~~ **CLOSED 2026-05-17**.
4. **P4-4** — panic guard (`unwind_guard!` macro) on every `pub extern "C"` entry point + `tests/capi_panic_safety.rs`. Branch `feat/capi-panic-boundary`.
5. **P4-5** — pathological classic-lifecycle coverage (`tests/capi_classic_lifecycle_pathological.rs`, ≥10 patterns).
6. **P4-6** — reconcile FEATURE_PARITY wording with the P4-5 evidence (or refile residual gaps as named P4-* OPEN).
7. **P4-7** — `tj3GetICCProfile` `TJERR_WARNING` soft-error path + stale-stub / divergence comment sweep across capi src.
8. ~~**P4-8** — runtime BMI1+LZCNT dispatch for x86_64 encode (audit + README update; the dispatch itself was already live).~~ **CLOSED 2026-05-17**.
9. **P4-13** — true streaming `jpeg_consume_input` / `JPEG_SUSPENDED` semantics (filed 2026-05-18 after cold review surfaced `jpeglib.rs:4234-4238` fully-buffered EOI shim).
10. **P4-14** — `max_memory_to_use` enforcement in C-side virtual-array path (filed 2026-05-18; field is ABI-mirrored but never consulted).
11. ~~**P4-15** — `jpeg16_*_raw_data` parity audit.~~ **CLOSED 2026-05-18** — mirrors upstream's 8/12-only raw-data API; no action.
12. ~~**P4-16** — thread-affinity contract on `cinfo` TLS side tables (filed 2026-05-18; pick Option A fix vs Option B document on adoption-pressure signal).~~ **CLOSED 2026-05-19** via Option B — new "Threading contract" section in `docs/ABI_COMPATIBILITY.md` documents the per-thread ownership requirement authoritatively. Option A (RwLock migration) remains tracked in the P4-16 body for future adoption pressure.
13. **P4-17** — real `JPEG_SUSPENDED` test in `capi_classic_lifecycle_pathological.rs` (filed 2026-05-18; the existing `source_mgr_suspends_every_byte` exercises chunked refill, not suspension).
14. ~~**P4-18** — file 18 legacy TurboJPEG 1.x/2.x aliases as implemented-and-allowlist-removed (Option A) or permanently-deferred-with-rationale (Option B). Filed 2026-05-18 after cold review found the legacy-TJ subset of the allowlist was missed by P3-3's closure scope.~~ **CLOSED 2026-05-19** via Option B — per-symbol migration matrix + tiny-shim recipe in `docs/ABI_COMPATIBILITY.md` under `### Legacy TurboJPEG 1.x/2.x aliases — partial coverage (P4-18)`. Option A (wire all 18 into `legacy.rs`) remains in the P4-18 body for future adoption pressure.
15. ~~**P4-19** — IDCT `islow` diverged from djpeg on i16-overflow (corrupt) coefficients (AC-all-zero `psllw` wrap shortcut).~~ **CLOSED 2026-05-30** — shortcut routed to scalar; full-path SSE2 residue refiled as P4-20.
16. **P4-20** — x86 SSE2 IDCT full path is i32 4-lane, not an i16-faithful port (corrupt-input-only; AVX2 already faithful, so CI-invisible). Lower urgency.
17. **P4-21** — decoder rejects non-standard sampling where a chroma component out-samples luma (`Cr=h3v1`); colour-path refactor (A) or lenient recovery (B).
18. ~~**P4-22** — decoder diverges from libjpeg-turbo on multi-scan non-interleaved baseline (never-scanned luma + doubly-scanned Cr).~~ **CLOSED 2026-05-31** — non-interleaved baseline planes pre-filled with 128 (IDCT-of-zero) so never-scanned components / padding match djpeg. Regression: `tests/cross_check_fuzz_decode_diff_c_multiscan.rs`.
19. ~~**P4-23** — lenient mode rejects corrupt baseline entropy ("invalid Huffman code") djpeg silently conceals.~~ **CLOSED 2026-05-31** — added lenient gray-fill + warning recovery to `decode_non_interleaved_baseline_planes` (corrupt streams fragment into spurious non-interleaved scans). Regression: `tests/cross_check_fuzz_decode_diff_c_multiscan.rs`.
20. ~~**P4-24** — arithmetic sequential (SOF9) non-interleaved multi-scan: `decode_arithmetic_planes` has no multi-scan dispatch (drops scans, 0-fill not 128, `unwrap_or(0)` Cs-misroute).~~ **CLOSED 2026-06-01** — added `decode_arithmetic_multiscan_planes` (per-scan `ArithDecoder`, 128 fill, reject unknown `Cs`; handles both non-interleaved and partially-interleaved scan scripts). Regression: `tests/cross_check_arith_noninterleaved.rs` (byte-exact vs djpeg).
21. **P4-25** — arithmetic DAC conditioning not snapshotted per scan (shared global read by all arith paths; same-slot redefinition between scans would mis-decode). Pre-existing; found in the P4-24 review. Not `fuzz_decode_diff_c`-reachable. Filed 2026-06-01.
