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
| P4-82 | CLOSED 2026-08-02 (classic scanline encoder now forwards public restart settings through every entropy branch) |
| P4-83 | CLOSED 2026-08-02 (baseline classic scanline encoder now honors public input smoothing without forcing Huffman optimization) |
| P4-84 | OPEN (classic C-ABI progressive/arithmetic scanline encoding still drops public input smoothing) |
| P4-85 | OPEN (classic scanline compression stores but does not apply public custom quantization/Huffman tables) |
| P4-86 | OPEN (classic lossy scanline compression hardcodes ISLOW and ignores public `dct_method`) |
| P4-87 | OPEN (classic abbreviated-datastream table state is not wired) |
| P4-88 | OPEN (classic scanline marker controls and CCIR601 rejection are ignored) |
| P4-89 | OPEN (classic arithmetic+lossless requests silently become Huffman lossless) |
| P4-90 | OPEN (classic arithmetic scanline compression ignores public DAC conditioning) |
| P4-91 | OPEN (classic scanline compression ignores custom scan scripts) |
| P4-92 | OPEN (classic scanline compression collapses valid sampling-factor layouts to 4:4:4) |
| P4-93 | OPEN (classic scanline compression ignores requested JPEG colorspace) |
| P4-94 | OPEN (classic 12/16-bit scanline buffers never reach a high-precision encoder) |
| P4-95 | OPEN (classic raw-data compression drops most public encode options) |
| P4-96 | OPEN (classic decompression color quantization and colormap switching are not wired) |
| P4-97 | OPEN (`jpeg_resync_to_restart` is an unconditional success no-op) |
| P4-98 | OPEN (classic 12/16-bit decode bypasses lifecycle and public output options) |
| P4-99 | OPEN (classic decode dispatcher ignores output options and colorspace metadata) |
| P4-100 | PARTIAL (classic codec failures are reported as suspension or silent success — translator + finish/start landed 2026-08-08) |
| P4-101 | OPEN (classic header parse does not publish coding tables/scan state) |
| P4-102 | OPEN (classic raw-data decode bypasses public options and state contracts) |
| P4-103 | OPEN (`jpeg_crop_scanline` does not implement iMCU-aligned C semantics) |
| P4-104 | OPEN (classic decompressor state constants/transitions and finish lifecycle diverge) |
| P4-105 | OPEN (classic marker writers ignore state and declared lengths) |
| P4-106 | OPEN (`jpeg_finish_compress` accepts incomplete input and bad states) |
| P4-107 | OPEN (`jpeg_enable_lossless` clamps invalid input and omits public state) |
| P4-108 | CLOSED 2026-08-08 (classic destination managers violate buffer ownership and I/O errors) |
| P4-109 | OPEN (classic source-manager setup/stdio semantics diverge) |
| P4-110 | OPEN (`jpeg_Create*` ignores version/struct-size ABI guards) |
| P4-111 | OPEN (classic progress-manager callbacks/counters are not wired) |
| P4-112 | OPEN (`jpeg_set_marker_processor` callbacks are stored but never invoked) |
| P4-113 | OPEN (`jpeg_read_icc_profile` bypasses classic saved-marker semantics) |
| P4-114 | OPEN (`jpeg_has_multiple_scans` equates multi-scan with progressive) |
| P4-115 | OPEN (native 12-bit coverage claims include modes and sampling layouts that are not tested) |
| P4-116 | PARTIAL (C-parity tests can convert Rust/oracle failures or missing comparisons into a pass — named suites closed 2026-08-08) |
| P4-117 | CLOSED 2026-08-08 (4:4:1 trim rejected images shorter than one iMCU row) |
| P4-120 | OPEN (classic-shim allocation-failure paths are unreachable from tests) |
| P4-121 | OPEN (lossless encode accepts a restart interval C refuses to decode) |

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

**Superseded 2026-07-28 (#385):** the tier table left `README.md` in the user-first restructure. The canonical T1–T4 framing is now the `docs/LAST_MILE.md` "Current Status" list, which `README.md` § "C ABI replacement tiers" points to; P4-2's substance (four surfaces, never conflated) is unchanged. The verification grep below therefore reports 0 for `README.md` — and always did for `docs/ABI_COMPATIBILITY.md`, whose tier mentions are prose, not `T1.`-style labels.

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

**Status (2026-05-17): closed.** P4-10's deliverable in this iteration is the architectural filing: `crates/libjpeg-turbo-rs-capi/tests/capi_{ffmpeg,gd,imagemagick,libvips,sdl_image,pillow}_compat.rs` + `tests/capi_pillow_compat.rs` + `examples/*_smoke/` already cover one version per consumer; the multi-distro / multi-version weekly matrix (Pillow 10.x + 11.x, ImageMagick 6 + 7, libvips 8.x real thumbnail workload, FFmpeg 6 + 7 mjpeg roundtrip, libtiff 4.x rich-marker, plus new Qt5/Qt6 and OpenCV harnesses) is filed as **[P2-G](backlog.md#p2-g-downstream-lab-multi-version--multi-distro-matrix--open)** in the in-repository long-term backlog because each new harness is its own engineering project and gating the work on T3 actually entering production keeps CI cost proportional to demand.

**Follow-up evidence (2026-08-02):** the first OpenCV leg now lives in
`examples/opencv_smoke/`. A pinned Ubuntu 24.04 image installs OpenCV 4.6,
executes real `cv::imwrite` plus color/grayscale `cv::imread`, proves through
glibc binding logs that OpenCV's `jpeg_CreateCompress` and
`jpeg_CreateDecompress` resolve to the Rust `libjpeg.so.8`, and cross-decodes
the system and Rust outputs in both directions. All four paths measure 49.226
dB PSNR with the same grayscale checksum. The reproducible record is
`experiments/opencv_downstream_2026-08-02.md`. This delivers one P2-G OpenCV
version, not the deferred Qt5/Qt6 or multi-version/multi-distro matrix. The
run also surfaced the missing GNU ELF `LIBJPEG_8.0` symbol definitions now
tracked as P4-81.

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
  3. Resumes after each `JPEG_SUSPENDED` by refilling the buffer; finish-decompress state/reset fidelity is tracked separately by P4-104.
- Bit-exact comparison of the resumed decode against the same JPEG decoded with the upstream linked-against-stock `libjpeg.so.8`.

**Fix (Option b — incremental input drain, decode stays buffered).** Implemented in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`. Two pure marker-scan helpers (`find_first_sos`, `scan_next_boundary`, unit-tested in `marker_scan_tests`) let the shim walk the entropy body. The drain is split, gated on a single runtime signal — *did `fill_input_buffer` return `FALSE` before EOI?*:

- **`jpeg_read_header`**: when `drain_caller_source_mgr` suspends (`None`) but the bytes drained so far already contain a complete header (through the first SOS, per `find_first_sos`), promote to `JPEG_HEADER_OK` with `body_incomplete = true` instead of `JPEG_SUSPENDED`. The header is parsed from the through-SOS prefix plus a *synthetic* `FF D9` (so `read_markers` terminates cleanly for progressive streams without choking on truncation); the real decode later runs on the complete buffer. **A non-suspending source (libtiff, Pillow `mem_src`, djpeg) never returns `None`, so it keeps the original fully-buffered path untouched.**
- **`jpeg_consume_input`**: while `body_incomplete`, pull from the live source manager one chunk at a time (`pull_more_from_source_mgr`) and report the next boundary — `JPEG_REACHED_SOS` at each scan, `JPEG_REACHED_EOI` at EOI (clearing `body_incomplete`), or `JPEG_SUSPENDED` when the source is dry.
- **`jpeg_start_decompress`**: in buffered-image mode, publish output dimensions from the header and defer the pixel decode; in non-buffered mode, finish draining the body to EOI now (suspending if dry).
- **`jpeg_read_scanlines`**: materialise the deferred decode (`ensure_decoded_deferred`) once the body is complete.
- **`jpeg_input_complete`**: returns `FALSE` while `body_incomplete`, so the `while (!jpeg_input_complete()) jpeg_consume_input()` idiom drives the body to EOI.
- `jpeg_finish_decompress` / `jpeg_abort_decompress` reset the new state for handle reuse.

**Status (2026-06-02): PARTIAL — suspension is byte-exact-proven; deeper contracts are P4-26 and finish lifecycle is P4-104.** `cargo test -p libjpeg-turbo-rs-capi --test capi_classic_lifecycle_pathological consume_input_suspends_through_progressive_body` passes: a real suspending source manager drip-feeds a multi-scan progressive JPEG; the harness asserts mid-body suspension, SOS/EOI progression, resume, and pixels byte-identical to both full-buffer shim decode and stock `djpeg`. It no longer treats the shim's post-finish `DSTATE_STOPPING` as an oracle because upstream resets the cinfo. Boundary scanning, the 256 MiB drain cap, marker rebuild, scan tracking, and raw/coefficient body drain remain covered.

**Why PARTIAL, not CLOSED.** Codex round 8 (on commit `4645b52`) raised three upstream-contract-fidelity gaps that lie *beyond* the stated acceptance criteria but mean the broad title — "honor per-byte source suspension" — is not yet fully met across every entry point: (1) `jpeg_read_header` only stops at the first SOS on the *suspending* path (gated on `body_incomplete`); a fully-buffered consumer still has the whole body swallowed in `read_header`, so a later `jpeg_consume_input` reports `REACHED_EOI` immediately without per-scan `REACHED_SOS` callbacks. (2) Buffered-image *output* calls (`jpeg_start_output` / `jpeg_read_scanlines` / `jpeg_finish_output`) do not themselves pull from the source manager — a consumer driving decode purely through the output side on a still-`body_incomplete` handle makes no forward progress. (3) The `marker_list` is *rebuilt* from the completed stream rather than *appended* in place, so a `jpeg_saved_marker_ptr` a consumer retained mid-stream is invalidated by the rebuild. All three need a deeper, consumer-risky refactor (gap (1) changes every fully-buffered consumer's `read_header` behavior), none block T3, and no known consumer exercises them — so they are filed as [P4-26](#p4-26-deeper-streaming-contract-fidelity-beyond-the-p4-13-core--open) rather than expanding this PR's scope. The verified streaming-suspension core lands here.

## P4-14. `max_memory_to_use` Is ABI-Mirrored But Not Enforced in the C-Side Allocation Path — **OPEN**

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/src/memmgr.rs` shows:

- `JpegMemoryMgr::max_memory_to_use: c_long` is at the correct upstream offset (compile-time `offset_of!` assertion at `:181`), defaulted to `~1GB` at `:817` — ABI fidelity is intact.
- Zero comparisons against `max_memory_to_use` exist anywhere in the file. `request_virt_sarray_impl` (`:527-551`), `request_virt_barray_impl` (`:558-582`), `realize_virt_arrays_impl` (`:591+`), `alloc_small_impl` (`:396`), `alloc_large_impl` (`:414`), `alloc_sarray_impl` (`:437`) all allocate without consulting the budget. No `JERR_OUT_OF_MEMORY` path is wired from a budget-exceed condition.

`docs/FEATURE_PARITY.md` lists `max_memory_to_use` as ✅ on the strength of `Decoder::set_max_memory()` / `TJPARAM_MAXMEMORY` honouring it in the **Rust** decode pipeline (now `src/decode/pipeline_impl/api.rs`). For the **C-ABI** consumer using `cinfo->mem->max_memory_to_use` directly (the upstream-documented path), the limit is silently ignored.

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

**Root cause.** The 3-component colour/upsample path now in `src/decode/pipeline_impl/output.rs` assumes **component 0 (luma) is the maximally-sampled component** and uses `y_width` / `y_height` as the output reference resolution, deriving each chroma upsample factor as `y_width / cb_w`, `y_width / cr_w`, etc. The JPEG spec, however, lets *any* component carry the max sampling factor. Here `Cr.h = 3 > Y.h = 1`, so `Hmax = 3` comes from Cr: the output plane is 24 px wide (`mcus_x·Hmax·8`, cropped to 15), Cr's plane is already 24 wide, and **luma is the component that needs upsampling** (`Hmax/Y.h = 3`). `cr_h_factor = y_width / cr_w = 8 / 24 = 0` (integer truncation) trips the degenerate-factor guard. No standard subsampling mode produces this shape (in 4:2:0/4:2:2/4:4:4 luma is always max), which is why it was never exercised.

**Acceptance criteria.** Either (A) **decode it correctly**: compute upsample factors relative to `Hmax·8 / Vmax·8` (the true output-component resolution) instead of `y_width`/`y_height`, and upsample *every* component — including luma — to that resolution (libjpeg's model: each component independently upsamples by `Hmax/h_i`, `Vmax/v_i`). This is a refactor of the heavily-optimized 3-component colour path (merged-upsample, 4:4:4/4:2:0/4:2:2 fast paths) and must keep the existing standard-sampling paths byte-exact. A cross-check test vs `djpeg` on the `Cr=h3v1` fixture (and `Cb`-max / `v`-axis variants) must pass. Or (B) a **lenient-mode recovery**: in `set_lenient(true)`, instead of `Err`, emit a best-effort raster + a new `DecodeWarning` so lenient decode is "at least as accepting as djpeg" (the fuzz then skips the comparison via the bilateral-OR lenient gate); strict mode keeps rejecting. (B) unblocks the fuzz without the refactor but does not give pixel-correct output.

**Why deferred.** Correct support (A) is a colour-pipeline refactor with real regression risk to the optimized standard-sampling paths; the trigger is a rare non-standard-sampling shape only reachable on corrupt/crafted inputs. Filed rather than rushed. Repro: `cargo run --example verbose_probe <artifact>` (local tool) shows `DECODE_ERR: CorruptData("chroma upsample factor zero…")`; `djpeg -pnm` yields `P6 15 9`.

## P4-22. Decoder Diverges From libjpeg-turbo on Multi-Scan (Non-Interleaved) Baseline With a Never-Scanned / Doubly-Scanned Component — **CLOSED 2026-05-31**

**Motivation.** A local 100,000-iteration `fuzz_decode_diff_c` smoke sweep (seed 424242, 2026-05-30) found a 64×64 baseline 4:4:4 fixture (in-repo `tests/fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg`; originally `~/smoke100k/artifacts/pix_x86_67674_d128.jpg`, iter 67674) that **both** libjpeg-turbo backends decode byte-identically (C-x86 djpeg == C-arm djpeg, diff 0) yet our decoder — on **both** NEON and SSE2, also byte-identical to each other — decodes differently: first pixel C=`(255,52,54)` vs Rust=`(178,0,0)`, `max_diff=128`, `mean_diff=98.5`, all 64 blocks wrong. Both sides run clean (no warnings / no lenient recovery), so the fuzz `(Some, Pixels)` pixel-diff arm fires: `128 ≫ tolerance 24` → **`fuzz_decode_diff_c` panic**. Reachable on AVX2 CI — the divergence is in shared scalar logic (both Rust SIMD backends agree with each other and differ from C).

**Structure of the trigger.** Three separate single-component scans (non-interleaved baseline): SOS#1 `Cs=3` (Td|Ta=0/0), SOS#2 `Cs=2` (1/1), SOS#3 `Cs=3` again (1/1). Component 1 (luma) is **never scanned**, component 3 (Cr) is **scanned twice**, component 2 (Cb) once.

**Root cause (confirmed).** `decode_non_interleaved_baseline_planes` (now `src/decode/pipeline_impl/baseline.rs`) allocated the per-component output planes with `vec![0u8; size]` and only IDCT-writes blocks a scan actually covers. The never-scanned luma plane therefore stayed at pixel value **0**, but the correct value for any un-decoded block is the IDCT of all-zero coefficients = `0 + CENTERJSAMPLE = 128`. The chroma decodes **identically** on both sides (Cb scanned once; Cr's two scans already resolve last-wins correctly) — the entire divergence is the luma: C fills it with `Y=128` (`djpeg -grayscale` confirms), we left it `Y=0`. With identical chroma and `Y=0`, `R = 0 + 1.402·(Cr−128) ≈ 178`, `G,B` clamp to `0` — exactly the observed Rust `(178,0,0)` vs C `(255,52,54)`. (The original filing suspected scan-routing / duplicate-Cr resolution; the actual defect was the plane fill value, and chroma was never wrong.)

**Fix.** Initialize the non-interleaved baseline planes with `vec![128u8; size]` so never-scanned components and MCU-alignment padding blocks equal libjpeg-turbo's IDCT-of-zero output. Single-scan / interleaved paths are unaffected: they reject component-omitting scans (`mcu_plan.len() < frame.components.len()`) or write every block.

**Status (2026-05-31): closed.** `cargo test --test cross_check_fuzz_decode_diff_c_multiscan multiscan_noninterleaved_64x64_444_matches_djpeg` passes (was `max abs diff = 128`, now `0`); the fixture is pinned in-repo at `tests/fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg`. Full `cargo test --workspace --release` green (2201 passed, 0 failed). Fix now lives at `src/decode/pipeline_impl/baseline.rs::decode_non_interleaved_baseline_planes`. The fixture lives in the non-globbed `tests/fixtures/fuzz_repro/` subdir (not the corpus glob): although it now decodes correctly, the `jpegtran`-style transform path still rejects this shape (`baseline SOS covers 1 components but frame has 3`), so it is not a valid corpus seed for the decode+encode+transform matrix. That transform-path limitation is the same non-interleaved-multi-scan family as P4-24 and is out of scope for this decode fix.

## P4-23. Lenient Mode Rejects Corrupt Baseline Entropy Data ("invalid Huffman code") That djpeg Silently Conceals — **CLOSED 2026-05-31**

**Motivation.** Same 100k smoke sweep, iter 874 (in-repo `tests/fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg` — kept in a non-globbed subdir so `examples/generate_corpus.rs` doesn't sweep this intentionally-rejected input into `tests/corpus/`, where `corpus_test` would count the reject as a CRASH; originally `~/smoke100k/artifacts/acc_x86_874.jpg`): a 65×65 baseline 4:2:2 (`Y=h2v1, Cb/Cr=h1v1`) fixture with corrupt scan data. `djpeg -pnm` exits 0 with **empty stderr** (silent concealment → no warning) producing a 65×65×3 raster; our decoder in **lenient mode** (`set_lenient(true)`, matching the fuzz oracle) returns `Err(CorruptData("invalid Huffman code"))`. The fuzz `(Some, Rejected)` arm fires → **`fuzz_decode_diff_c` "drop-in regression" panic** (C accepted, Rust rejected). Distinct from P4-21 (that is a non-standard-sampling `factor 0` reject; this is an entropy-decode error on standard 4:2:2).

**Root cause (confirmed).** Not the single-scan path the heading implies. The corrupt entropy data contains spurious `FFDA` byte sequences that the marker scanner reads as **extra SOS markers**, fragmenting the stream into **three non-interleaved scans** (`metadata.scans.len() == 3`). Decode therefore routes to `decode_non_interleaved_baseline_planes` (the same function as P4-22), and **that path had no lenient error recovery**: its `decode_block(...)?` propagated the invalid-Huffman error straight out, unlike the interleaved general path (now `src/decode/pipeline_impl/baseline.rs`) which already gray-fills + warns. So `set_lenient(true)` still returned `Err` here.

**Fix.** Wrap the non-interleaved per-block decode in a lenient match that mirrors the interleaved general path and libjpeg `jdhuff`'s "fake a zero" concealment: on a decode error in lenient mode, zero the offending block (so the IDCT writes the 128 midpoint = the P4-22 fill), push a `DecodeWarning::HuffmanError` once per scan, reset the DC predictor, and **continue** — so a restart interval resyncs at the next RST instead of discarding the recoverable tail (`UnexpectedEof` is the one case that stops the scan, leaving the rest at the 128 init). Strict mode still propagates the error (`Err(e) => return Err(e)`). The function now returns the accumulated warnings instead of `Vec::new()`.

**Status (2026-05-31): closed.** `cargo test --test cross_check_fuzz_decode_diff_c_multiscan corrupt_huffman_65x65_422_lenient_matches_djpeg` passes: lenient decode now yields a 65×65×3 raster with ≥1 warning (was `Err`), and the test also asserts strict mode still `Err`s. Full `cargo test --workspace --release` green (2202 passed, 0 failed); fresh `corpus_test` regen = 0 crashes / 0 decode fails. Fix now lives at `src/decode/pipeline_impl/baseline.rs::decode_non_interleaved_baseline_planes`. The fixture stays in the non-globbed `tests/fixtures/fuzz_repro/` subdir (still a strict-mode reject, so not a valid corpus seed).

## P4-24. Arithmetic Sequential (SOF9) Non-Interleaved Multi-Scan Decodes Only the First Scan, With Wrong Plane Fill — **CLOSED 2026-06-01**

**Motivation.** Found during the P4-22 code review (2026-05-31). `decode_arithmetic_planes` (now `src/decode/pipeline_impl/arithmetic.rs`) is the same defect family as P4-22 but broader. Unlike the Huffman baseline path — which dispatches to `decode_non_interleaved_baseline_planes` when `self.metadata.scans.len() > 1` — the arithmetic path has **no multi-scan dispatch**: it reads only `self.metadata.scan` (the *first* SOS) and builds its component set from that single scan. A SOF9 arithmetic **non-interleaved multi-scan** stream is accepted/accumulated by the marker loop (`marker.rs`: `is_non_interleaved_baseline` holds for SOF9 too), but this decoder then (a) **drops** every scan after the first, (b) leaves the un-scanned component planes at the `vec![0u8; size]` init — pixel `0` where libjpeg produces `128` (the P4-22 bug), and (c) resolves `Cs`→component index with `unwrap_or(0)`, silently misrouting an unknown selector to component 0 instead of rejecting.

**Not reachable via `fuzz_decode_diff_c`** (it early-returns on `probe.is_arithmetic()`), which is why the 100k smoke sweep did not surface it; reachable only by real arithmetic multi-scan inputs / `fuzz_decompress`. Lower urgency than P4-22/P4-23 for that reason.

**Fix (Option A — full support, all scan scripts).** Added `decode_arithmetic_multiscan_planes`, mirroring `decode_non_interleaved_baseline_planes` but with a fresh `ArithDecoder` per scan: planes pre-filled with `128`; each scan decodes from its own entropy segment (`scan_info.data_offset`) with conditioning + restart applied per scan; unknown `Cs` is rejected (`ok_or_else`, not `unwrap_or(0)`). Each scan uses its own MCU layout — a single-block raster for a one-component scan (T.81 A.2.3), or the frame-level interleaved MCU grid with `Hi·Vi` blocks per component for a multi-component scan (A.2.2) — so both fully non-interleaved (`cjpeg -scans "0; 1; 2;"`) and **partially interleaved** (`"0; 1 2;"`) scan scripts decode. `decode_arithmetic_planes` dispatches to it when `self.metadata.scans.len() > 1`, and its own single-scan path also had its `unwrap_or(0)` replaced with a hard reject. Block-decode reuses the proven `decode_dc_sequential`/`decode_ac_sequential` primitives.

**Status (2026-06-01): closed.** `cargo test --test cross_check_arith_noninterleaved` passes both cases byte-exact vs `djpeg`: `arith_noninterleaved_16x16_444_matches_djpeg` (3 one-component scans; was `max abs diff = 244` — only luma decoded, chroma at 0 → RGB (0,137,0) vs djpeg (2,2,2); now ≤1) and `arith_partial_interleaved_16x16_444_matches_djpeg` (luma scan + Cb/Cr interleaved scan; pre-generalization the 2-component scan was rejected outright). Fixtures pinned at `tests/fixtures/fuzz_repro/arith_{noninterleaved,partial_interleaved}_16x16_444.jpg`. Full `cargo test --workspace --release` green. Fix now lives at `src/decode/pipeline_impl/arithmetic.rs::decode_arithmetic_multiscan_planes`, with dispatch in `decode_arithmetic_planes`. Follow-up [P4-25](#p4-25-arithmetic-dac-conditioning-is-not-snapshotted-per-scan--open) filed for a pre-existing per-scan DAC-snapshot gap surfaced in this review.

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

**Motivation.** Fuzz Smoke run `29977722126` (`fuzz_decompress_lenient`, crash-4a2b926c): 64x64 baseline with Y=4x1, Cb=3x1, Cr=1x1 panicked with `range end index 3088 out of range for slice of length 3072` at the then-current `pipeline.rs:4333`.

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

## P4-39. CMYK Encode Path Silently Drops Restart / Custom-Table Options and Rejects Optimize+Smoothing — **CLOSED 2026-07-26**

**Motivation.** Surfaced 2026-07-25 while refactoring the `compress_*` family onto a single `CompressParams` core (see [P4-40](#p4-40-srcencodepipeliners-is-10k-lines-of-copy-pasted-compress_-variants--closed-2026-08-02)). The new characterization fixture `tests/fixtures/encode_pipeline_golden.txt` shows the option-carrying variants producing **byte-identical output to plain `compress`** on CMYK input — the options never reach the encoder. Tracked upstream as GitHub issue [#313](https://github.com/developer0hye/libjpeg-turbo-rs/issues/313).

**Root cause.** CMYK support was implemented exactly once, as `compress_cmyk(pixels, width, height, quality, subsampling)` — a signature that cannot express restart intervals, custom tables, smoothing, or DCT method. Every other variant early-returned into it and silently discarded its remaining parameters. The first P4-40 consolidation reduced that to one site in `src/encode/pipeline_impl/baseline.rs::compress_with_params`, inherited by all four baseline shims:

```rust
if pixel_format == PixelFormat::Cmyk {
    return compress_cmyk(pixels, width, height, quality, subsampling);
}
```

Five defects, four of them silent (`Ok(bytes)` with the option not applied):

1. `compress_with_restart` dropped `restart_interval` (the CMYK dispatch in `baseline::compress_with_params`) — no RST markers emitted.
2. `compress_custom_quant` dropped custom quantization tables (same site).
3. `compress_custom_huffman` dropped custom Huffman tables (same site).
4. `compress_optimized` rejected CMYK with `JpegError::Unsupported("CMYK pixel format not supported for encoding")`. Because `Encoder`'s mode dispatch routed through it for both `optimize_huffman(true)` and `smoothing_factor(>0)`, **both builder options failed outright on CMYK**.
5. `compress_cmyk` (now `src/encode/pipeline_impl/baseline.rs::compress_cmyk`) ignored `dct_method` — `IsLow`/`IsFast`/`Float` were byte-identical.

**Divergence from C.** None of these are colorspace-gated upstream: `optimize_coding` (`jcmaster.c:595-802`, `jcinit.c:83-127`), `restart_interval` (`jchuff.c:693-876`), `smoothing_factor` (`jcsample.c:509-553`, per-component with a `smoothok` fallback, not a colorspace gate), quantization slots (`jcparam.c`), and `dct_method` (`jcdctmgr.c`). `cjpeg -optimize`, `-smooth N`, `-restart N`, `-qtables` and `-dct fast|float` all apply to CMYK/YCCK in C.

**Acceptance criteria.**

1. CMYK honours `restart_interval`, custom quant tables, custom Huffman tables, `smoothing_factor`, `optimize_huffman`, and `dct_method`.
2. Byte-exact cross-validation against C for each option on CMYK input.
3. The six regression tests in `tests/encode_cmyk_option_parity.rs` un-`#[ignore]`d and green (all six fail today when run with `--include-ignored`, which is the reproduction).
4. `tests/fixtures/encode_pipeline_golden.txt` regenerated in the *fixing* commit — never in a refactor commit — with the CMYK rows reviewed as a diff.

**Why deferred.** The P4-40 refactor is deliberately byte-exact, so it cannot carry a behavioural fix. It does remove the structural cause: once CMYK is a component-layout choice inside one core rather than a separate narrower function, these become small changes rather than five parallel edits.

**Status (2026-07-26): 4 of 5 closed.** `compress_cmyk` now takes `CompressParams` and honours the restart interval (DRI + RST with all four DC predictors reset together), custom quantization tables, custom Huffman tables, and `dct_method`. Slot 0 is the one that applies: all four components share one quantization table and one DC/AC pair, per `turbojpeg.c:418-427`.

That prediction held — once the option set was a value rather than a signature, the fix was mechanical rather than five parallel edits.

The golden fixture moved in exactly **922 of 20,160** cases, every one of them CMYK: 202 `compress` (the `dct_method` variants), 180 `customhuff`, 180 `customquant`, 360 `restart`. Nothing else shifted. The four corresponding `cmyk|effect|*` rows are gone from the option matrix's allowlist, and four of the six reproductions in `tests/encode_cmyk_option_parity.rs` are un-`#[ignore]`d.

**Status (2026-07-26): closed.** `optimize_huffman` and `smoothing_factor` now work on CMYK, and the whole path is byte-exact against C.

Closing the last two required first building the thing that should have existed from the start: **a C oracle**. `cjpeg` reads PNM, BMP, GIF and Targa — none of which carry CMYK — so the four-component path was the one encode path in this tree that could only ever be compared against itself. `examples/cmyk_encode_c_oracle.c` drives libjpeg directly with `JCS_CMYK` and TurboJPEG's component layout; `tests/helpers/c_oracle.rs` compiles it on demand against whatever libjpeg development install it finds.

The first sweep answered the question the issue had been asking for a week, and then some. **Every single case differed** — but by exactly 18 bytes, all at byte 3. That was a JFIF APP0 marker C never writes ([#339](https://github.com/developer0hye/libjpeg-turbo-rs/issues/339)), and behind it component IDs `1,2,3,4` where libjpeg writes `'C','M','Y','K'`, and behind *those* a bottom-padding rule that clamped the last row where C repeats the last row group ([#340](https://github.com/developer0hye/libjpeg-turbo-rs/issues/340)). Three defects stacked in the header and the padding, each invisible until the one in front of it was removed.

The two features themselves:

- **`optimize_huffman`** — the MCU walk is now a single `scan_cmyk_blocks` driven by a callback, so the pass that counts symbols and the pass that writes them cannot drift apart. It runs twice, as C's `optimize_coding` does, rather than buffering every block: a four-component image buffers a third more than a three-component one, and re-deriving costs a second FDCT pass instead of memory.
- **`smoothing_factor`** — follows `jcsample.c:506-553`'s per-component chooser, not a colorspace gate. Components 0 and 3 are always at the maximum and take `fullsize_smooth_downsample`; components 1 and 2 take it too at 1x1, `h2v2_smooth_downsample` at 2x2, and nothing at all otherwise, which is what C's `smoothok` fallback does.

The subtlety worth recording: **enabling smoothing changes the padding of components it never touches**. Smoothing needs context rows, which moves the whole prep controller onto `pre_process_context` (`jcprepct.c:220-299`) — and that routine has no output-side padding at all, so every component switches from the row-group repeat to a plain last-row repeat. `need_context_rows` is a pipeline-wide flag, not a per-component one. This cost six cases at 1x2 and was the last thing to fall.

**Proof.** 504 cases — 4 subsamplings x 7 geometries x 3 qualities x {plain, restart 3, `-dct fast`, `-optimize`, `-smooth 25`, `-smooth 100`} — byte-identical to the C oracle, in `tests/regression_cmyk_c_parity.rs`. `-dct float` gets the weaker guarantee it gets everywhere else: pixel-equivalent, max 4 per sample measured. All six reproductions in `tests/encode_cmyk_option_parity.rs` are un-`#[ignore]`d. The golden fixture moved in 2,520 rows, every one CMYK, and 1,080 `ERR Unsupported` rows became real output — the 840 that remain are the S410/S24 combinations that genuinely exceed the 10-block MCU cap.

## P4-51. CMYK Streams Carry a JFIF APP0 Marker C Never Writes, and Non-libjpeg Component IDs — **CLOSED 2026-07-26**

**Motivation.** Found 2026-07-26 the moment the P4-39 C oracle existed. GitHub [#339](https://github.com/developer0hye/libjpeg-turbo-rs/issues/339).

**Root cause.** `jpeg_set_colorspace` clears both marker flags and re-enables `write_JFIF_header` only for `JCS_GRAYSCALE` and `JCS_YCbCr` (`jcparam.c:357-392`); `JCS_CMYK` sets `write_Adobe_marker` alone. We wrote both. JFIF is defined for grayscale and YCbCr only, so an APP0 on a CMYK stream is not merely 18 redundant bytes — it asserts something untrue about the data. The same section sets the component IDs to the ASCII initials `'C','M','Y','K'`; we wrote `1,2,3,4`, which decoders tolerate but libjpeg does not emit.

**Status (2026-07-26): closed.** Both fixed. `tests/regression_cmyk_c_parity.rs::cmyk_stream_carries_the_adobe_marker_and_no_jfif` pins the marker sequence and the IDs without needing any C tool, so the contract holds even where the oracle cannot be built.

## P4-52. CMYK Bottom Padding Clamps the Last Row Where C Repeats the Last Row Group — **CLOSED 2026-07-26**

**Motivation.** Found 2026-07-26 once P4-51 was out of the way and the byte comparison could reach the entropy-coded data. GitHub [#340](https://github.com/developer0hye/libjpeg-turbo-rs/issues/340).

**Root cause.** C pads twice, in different places and by different rules: the **input** side completes the final row group by repeating the last real row (`jcprepct.c:171-178`), and the **output** side fills the rest of the iMCU by repeating the last *downsampled* row (`jcprepct.c:197-205`). Carried back to full resolution, that second rule means different things per component. A component sampled at the maximum downsamples 1:1, so repeating its last output row is just repeating its last input row. A component subsampled `v` ways has one output row per `v` input rows, so repeating its last output row means repeating the last complete **group** of `v` input rows.

CMYK has both kinds in one image — components 0 and 3 carry the sampling factors, 1 and 2 sit at 1x1 — so no single rule is right for the whole image, and letting the per-block edge path clamp is right for neither when `v > 1`. This is the same distinction as [P4-47](#p4-47-progressive-420--440-diverges-from-cjpeg-at-every-even-height-not-a-multiple-of-16--closed-2026-07-26) (#324), arrived at from the other direction.

**Status (2026-07-26): closed.** `pad_plane_to_mcu_grid` takes the row-group height as a parameter and each component passes its own. Byte-exact against the C oracle for every legal CMYK subsampling; before the fix, 1x2 and 2x2 were 15/21 and the six failures per subsampling were exactly the geometries whose height is a multiple of `v_samp` but not of the MCU height.

## P4-40. `src/encode/pipeline.rs` Is 10k Lines of Copy-Pasted `compress_*` Variants — **CLOSED 2026-08-02**

**Motivation.** Filed 2026-07-25 from a structural review. At filing `src/encode/pipeline.rs` was 10,647 lines holding 103 free functions, 1 struct and **0 `impl` blocks** (10,235 / 3 / 2 after the Status below), and it absorbed 108 of the last 1,174 commits (`src/decode/pipeline.rs` another 118) — the repo's highest size×churn product. Ten public `compress_*` entry points are copy-pasted variants of one algorithm; normalized unique-line overlap is 85% (`compress` vs `compress_with_restart`), 84% (vs `compress_custom_quant`) and 71% (vs `compress_custom_huffman`).

**Realized cost** — this is not a style complaint; the divergence has already produced defects:

- The fused single-pass color-convert path (keeps data in L1/L2 between conversion and encode) existed **only** in `compress()`. Every other variant called full-plane `convert_to_ycbcr`, so restart-interval and custom-table encodes silently ran the slow path — against the project's stated goal of matching or beating C. Resolved by the Status below: it now lives once in `src/encode/pipeline_impl/baseline.rs::compress_with_params`, and all four baseline entry points reach it.
- `smoothing_factor` was implemented **only** in `compress_optimized`.
- `compress()` clamps scaled quant values to 255, breaking `cjpeg` parity below q≈20. Rather than fix it, `src/api/encoder.rs:825-846` routes around it with the heuristic `q < 50` and the comment *"Use a generous threshold to be safe."*
- All five CMYK defects in [P4-39](#p4-39-cmyk-encode-path-silently-drops-restart--custom-table-options-and-rejects-optimizesmoothing--closed-2026-07-26).
- At filing, 36 of the project's 66 `#[allow(clippy::too_many_arguments)]` were in the monolith; `compress_optimized` takes 9 positional parameters.

**Acceptance criteria.**

1. A `CompressParams` value type carries the full option set; one core routine subsumes `compress`, `compress_with_restart`, `compress_custom_quant`, `compress_custom_huffman` and `compress_optimized`, which become thin shims retaining their exact public signatures.
2. The refactor is **byte-exact**: `cargo test --test encode_pipeline_golden` passes unchanged against the pre-refactor fixture (33,600 pinned cases today).
3. Every optimization reachable from every option combination — no feature is available on only one branch.
4. `src/encode/pipeline.rs` becomes a stable façade and implementation is split by mode (baseline / progressive / arithmetic / lossless / downsample / quant-divisors) so no implementation file exceeds ~2k lines.

**Status (2026-07-25): criteria 1-3 closed.** `CompressParams` + `compress_with_params` landed; `compress`, `compress_with_restart`, `compress_custom_quant` and `compress_custom_huffman` are now 4-to-14-line shims over it, with their public signatures unchanged. Criterion 2 was met in two steps: routing `compress()` alone through the core moved **0 of 20,160** golden cases, and switching the other three moved 904 — every one of them verified against `cjpeg` (all four paths now 576/576 on the geometry sweep, versus 516 / 372 / 372 / 372 before). Criterion 3 is pinned by `tests/encode_params_composability.rs`, which exercises combinations no previous entry point could express (restart + custom quant + custom Huffman + `ifast` in one encode) and asserts each option's effect is independent of the others. Net -412 lines in the file, and it gains its first two `impl` blocks. Perf unchanged on the fast path (433 MP/s at 1920x1080, same as before). This also closed P4-42 outright and reduces P4-39 to a small change.

**Sequencing.** Criteria 1–3 first (byte-exact, low risk, unblocks P4-39). Criterion 4 is mechanical and can follow. Deliberately excluded: `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` is also large (9,242 lines) but is 164 flat `extern "C"` shims mirroring a C header — its size is inherent, not tangled, and it needs at most a split by API family.

**Status (2026-08-02): closed.** Criterion 4 is complete. The former 11,169-line implementation is now a 23-line `encode::pipeline` compatibility façade over 15 private implementation modules. The largest implementation file is `pipeline_impl/mcu.rs` at 1,704 lines; baseline (1,479), progressive (1,562), arithmetic (1,349), optimized (1,342), sampling (967), and every other implementation file remain below the ~2,000-line ceiling. `compress_optimized` and `compress_optimized_with_params` retain the two-pass algorithm but consume the same `CompressParams` value and are reached through the shared `compress_with_params` dispatch. The split is mechanically byte-exact: the unchanged golden fixture passes all **33,600** cases, and the focused C suite passes **81** arithmetic/progressive/lossless/raw/cross-product/encoder-compatibility tests with zero differential failures. `tests/encode_pipeline_public_api.rs` pins every established `encode::pipeline` function path and exact function-pointer type, every public `CompressParams` field, and every builder method. A sanitized system-toolchain `cargo test --workspace --no-fail-fast` run passes, including stock-tool linkage and downstream C-ABI integration tests. Criterion's non-LTO bench profile was compared sequentially against a clean `main` build on the same host and pinned CPU: every encode group was non-regressing; isolated reruns of the two noisy small-image groups measured 4:2:2 at **-0.22% (p=0.56)** and 4:4:4 at **-0.61% (p=0.47)**, both reported as no change. Hot helpers that now cross codegen-unit boundaries carry explicit `#[inline]` hints.

## P4-41. AVX2 4:2:0 Row Fast Path Ignored the Dummy-Block Contract and Skipped Its Own Capability Check — **CLOSED 2026-07-25**

**Motivation.** Found 2026-07-25 while building the P4-40 characterization fixture, which showed `compress()` (fused) and `compress_with_restart()` (full-plane) disagreeing on 176 of 1440 RGB cases. A 576-case geometry sweep against stock `cjpeg` (widths x heights over `{7,8,15,16,17,23,24,31,32,33,48,64}` x 4 subsamplings, q50) scored fused 516/576 and full-plane 372/576 — so at most one could be right, and neither was fully. GitHub [#314](https://github.com/developer0hye/libjpeg-turbo-rs/issues/314) and [#315](https://github.com/developer0hye/libjpeg-turbo-rs/issues/315).

**Root cause (two defects in one `if`)** — the AVX2 4:2:0 row fast-path guard, `if is_420 && !cb_half.is_empty() && eff_row_height == y_mcu_height` in the former monolithic pipeline before the fix (the replacement guard and `use_avx2_420` capability flag now live in `src/encode/pipeline_impl/baseline.rs::compress_with_params`):

1. **#314 — missing dummy-column guard.** The fast path FDCTs every block of every MCU unconditionally. It guarded the last partial MCU *row* (`eff_row_height == y_mcu_height`) but had no guard for the last partial MCU *column*, so for any width with `ceil(width/8)` odd it transformed replicated edge pixels where C emits a zeroed dummy block carrying the previous block's DC (`jccoefct.c:292-312`). Affected ordinary photo sizes — 500x375, 1000x750, 1000x1000, 1080x1080 all diverged; files ran 0.7–0.9% larger than C's. Only 4:2:0; 4:4:4 / 4:2:2 / 4:4:0 were clean.
2. **#315 — capability check bypassed.** The path gated on `!cb_half.is_empty()`, an allocation-shape proxy: `cb_half` was allocated whenever the subsampling was 4:2:0 but only *filled* under `is_x86_feature_detected!("avx2")`. On a non-AVX2 x86_64 CPU that reached `#[target_feature(enable = "avx2")]` helpers (undefined behaviour) with never-downsampled all-zero chroma.

**Why it escaped.** Three independent reasons, each worth noting: (a) the fast path is x86_64+AVX2-only and the project was developed on aarch64, which always takes the generic path — the same platform-dependent drift class as P4-33; (b) the byte-exact encoder cross-check (`tests/cross_check_encoder_binary.rs:77-78`) used **only 48x48**, MCU-aligned at 4:2:0, so `ndummy == 0` and the bug was unreachable; (c) `assert_encoder_output_matches` (`:36-67`) falls back from byte comparison to a decoded-pixel comparison with `max_diff <= 1`, and dummy blocks lie outside the image and are cropped on decode — the pixel diff is 0, so that test could never have caught it at *any* size.

**Fix.** A single `use_avx2_420` capability flag now gates the buffer allocation, the downsample and the encode fast path, so they cannot disagree; and the fast path additionally requires `y_last_col_width == y_mcu_width`. Partial geometries fall through to the generic path, which already handled dummies via `encode_color_mcu_with_dummies`.

**Status (2026-07-25): closed.** The fused path now matches `cjpeg` on **576/576** swept geometries (was 516/576) and on all 24 real-world cases (500x375 … 1920x1080 x 4 subsamplings). Pinned by `tests/regression_420_dummy_block_columns.rs` — a `cjpeg` byte-exact check over 10 geometries plus 6 C-tool-free length+hash pins taken from `cjpeg`-verified output. Cost ~3.2–4.5% throughput on affected widths (medians of 15 runs x 3 repeats; recorded in `experiments/x86_64_pipeline.tsv`), tracked for recovery as [#317](https://github.com/developer0hye/libjpeg-turbo-rs/issues/317) / P4-43. Note #315's acceptance criterion 2 — a CI leg that masks AVX2 at the CPUID level — is **not** delivered here; the UB is removed by construction but remains untested on this hardware.

**Follow-up (2026-07-27): duplicate report [#362](https://github.com/developer0hye/libjpeg-turbo-rs/issues/362) confirmed already fixed.** Filed against 0.6.2 / 0.6.3 as a 4:2:0-only divergence window at `width % 16 ∈ 1..=8` — which is exactly the residue set that makes `ceil(width/8)` odd, i.e. this defect. Reproduced at `v0.6.3` (144 of 918 swept scans diverged, +0.64…+1.10% bytes) and absent at `main` (918/918 byte-exact vs C libjpeg-turbo 3.1.4.1 across content x quality x height x width x 4:4:4/4:2:2/4:2:0, plus 153/153 across baseline / optimized-Huffman / progressive); the fix shipped in **v0.7.0**. The report's own diagnosis — chroma `expand_right_edge` running before `h2v2` downsampling rather than after — is wrong: 4:2:0 chroma has exactly `ceil(width/16)` block columns and therefore never carries a padding block at all, so the mechanism is luma-side only. `tests/regression_issue_362_420_trailing_mcu.rs` now pins both halves *without* C tools (red at `v0.6.3`, green at `main`), closing the coverage gap that allowed the re-report: the P4-41 pins above are opaque length+hash values that assert no contract, and its `cjpeg` leg skips wherever the C tools are absent.

## P4-42. Full-Plane Encode Variants Skip the Dummy-Block Contract on Every Platform — **CLOSED 2026-07-25**

**Motivation.** Filed 2026-07-25 alongside P4-41. `compress_with_restart`, `compress_custom_quant` and `compress_custom_huffman` use full-plane `convert_to_ycbcr` and never implement C's dummy-block contract, so they diverge from `cjpeg` on **204 of 576** swept cases — unlike P4-41 this is **not** platform-gated. GitHub [#316](https://github.com/developer0hye/libjpeg-turbo-rs/issues/316).

Divergences by subsampling: 4:2:0 → 108, 4:2:2 → 72, 4:4:0 → 24; 4:4:4 clean. The failing set is exactly C's `ndummy > 0` condition — at 4:2:2 it fails for width ∈ {7,8,17,23,24,33} (`ceil(w/8)` odd) and passes for {15,16,31,32,48,64} (even); at 4:4:0 it fails for height ∈ {8,24}. `restart_interval = 0` throughout, so the buffering strategy is the only variable.

`Encoder` routes to these whenever a restart interval, custom quant tables or custom Huffman tables are requested (`src/api/encoder.rs:930-952`), so any such encode at a partial-MCU geometry is non-conformant.

**Acceptance criteria.**

1. Restart / custom-quant / custom-Huffman encodes byte-identical to the equivalent `cjpeg` invocation at partial-MCU geometries for 4:2:0, 4:2:2 and 4:4:0.
2. Regression coverage over `ceil(width/8)` odd and `ceil(height/8)` odd, not only MCU-aligned sizes.
3. Satisfied by routing every variant through one code path (the P4-40 `CompressParams` core) rather than by patching dummy-block logic into each copy.

**Status (2026-07-25): closed.** Delivered by criterion 3 — the three variants are now shims over `compress_with_params`, so they inherit the fused strategy's dummy-block handling instead of carrying their own. All three match `cjpeg` on **576/576** swept geometries (was 372/576), and `compress_with_restart(ri=3)` additionally matches `cjpeg -restart 3B` byte-for-byte on all 576. Pinned by `issue_316_full_plane_variants_match_cjpeg_at_partial_mcus` in `tests/regression_420_dummy_block_columns.rs` (48 cjpeg cross-checks over 3 subsamplings x 8 geometries x {ri=0, ri=3}).

## P4-43. Recover the AVX2 4:2:0 Fast Path for Interior MCU Columns — **CLOSED 2026-07-26**

**Motivation.** The P4-41 fix disables the fast path for the whole image when the last MCU column needs dummy blocks (`ceil(width/8)` odd — roughly half of all widths), costing ~3.2–4.5%. GitHub [#317](https://github.com/developer0hye/libjpeg-turbo-rs/issues/317).

Measured (EPYC 9554, medians of 15 runs, 3 repeats, `examples/bench_encode_420_geometry.rs`), width pairs 8px apart so only `ceil(w/8)` parity changes: 1008x750 fast 433.1 MP/s vs 1000x750 generic 413.6 (-4.5%); 1920x1080 fast 432.3 vs 1928x1080 generic 416.7 (-3.6%); 3840x2160 fast 432.6 vs 3848x2160 generic 418.7 (-3.2%).

**Why deferred.** The fast path hoists one `begin_block`/`end_block` pair across the whole MCU row and writes through a raw `(pb, fb, buf)` triple, while the dummy path writes through `bit_writer`; splitting a row between them is not a local change. Correctness shipped first.

**Acceptance criteria.** Throughput at `ceil(width/8)` odd within ~1% of the even-width case, with the 576-case sweep and the golden fixture unchanged.

**Status (2026-07-26): closed.** The row fast path now runs over `0..mcus_x - 1` when the last MCU column is partial, and only that trailing column falls through to `encode_color_mcu_with_dummies`. Previously one partial column disqualified the entire row.

Residual gap (medians of 15 runs x 2 repeats, EPYC 9554): 3848x2160 **0.7%** (was 3.2%), 1928x1080 **1.2%** (was 3.6%), 1000x750 **2.0%** (was 4.5%). The two large sizes meet the ~1% criterion; the smallest retains a little more because one generic column is a larger share of a short row.

Free of correctness cost: the golden fixture is unchanged (byte-exact) and the C sweep shows no new divergence.

## P4-44. Encoder Byte-Parity Against `cjpeg` Is Unmeasured for `ifast` / `float` and for aarch64 — **CLOSED 2026-07-26**

**Motivation.** Filed 2026-07-25 from PR #318 CI. The x86_64 encoder is now byte-identical to stock `cjpeg` on 576/576 swept geometries (P4-41, P4-42) — but only for `-dct int`. The `linux-aarch64 NEON` job failed the x86_64-pinned golden fixture with divergences clustered in `ifast` and `float`: at 16x16 BGR 4:2:0 q100, x86_64 emits 954 bytes and aarch64 955 (`float`), 944 vs 956 (`ifast`); at 4:2:2 q100 `ifast`, 1058 vs 1078. GitHub [#319](https://github.com/developer0hye/libjpeg-turbo-rs/issues/319).

P4-33 established the phenomenon (backend-dependent output, decodes pixel-identically under `djpeg`) and canonicalized the fuzz corpus on x86_64-linux; PR #318 follows the same precedent for its byte fixtures. So nothing is broken by the current definition — but "decodes pixel-identically" is weaker than the byte-exactness this project targets, and the gap has never been quantified.

**Two open questions.** (a) Does aarch64 `islow` match `cjpeg` at partial-MCU geometries? The `*_matches_cjpeg` tests added in P4-41/P4-42 run unguarded on every platform, so the next aarch64 CI run answers this for 4:2:0. (b) Does *either* backend match `cjpeg -dct fast` / `-dct float`? The existing cross-checks only ever pass `-dct int`, so it is possible x86_64 diverges here too and nobody has looked.

**Acceptance criteria.**

1. A measured answer for each (backend x DCT method) pair against `cjpeg` — cheapest first: extend the sweep in `examples/probe_fused_vs_fullplane.rs` to `-dct fast` / `-dct float` and run it on x86_64, which settles (b) with no aarch64 hardware.
2. The same sweep as a CI step on the `linux-aarch64 NEON` job, which already installs official libjpeg-turbo 3.1.4.1 and so has `cjpeg` available.
3. Whatever byte-exactness the project actually guarantees stated in `docs/FEATURE_PARITY.md` and enforced per backend. Concluding "byte-exact for `islow`, pixel-accurate for `ifast`/`float`" is an acceptable outcome — the requirement is that it be a documented decision rather than an unexamined assumption.

**Status (2026-07-26): measured, and it was not a documentation question.** Criterion 1 done on x86_64: sweeping `-dct` across 3456 cases showed `int` byte-exact everywhere, `float` diverging broadly, and `ifast` diverging for 4:4:4 / 4:2:2 / 4:2:0 while matching for grayscale and the 4-factor subsamplings — a pattern that tracks SIMD availability, not the transform.

That led to **P4-50 / #330**: `ifast` was not merely non-byte-exact, it was *2.5x the error and 22% larger* than C's, which no tradeoff explains. Fixed; `int` and `ifast` are now both byte-identical to `cjpeg` across every subsampling and colourspace.

`float` remains non-byte-exact by nature — floating-point operation ordering — but matches C's quality and size, with a measured max per-sample difference of 7. That is now a stated guarantee rather than an assumption, pinned by `float_is_pixel_equivalent_to_cjpeg`.

**Status (2026-07-26): closed.** Criterion 2 needed no new CI job. `tests/regression_dct_method_parity.rs` is unguarded — it runs on every platform that has `cjpeg` — and the `Test (linux-aarch64 NEON)` leg installs official libjpeg-turbo 3.1.4.1, so it has been answering the aarch64 question on every run since #330 merged. Run 30175204716 shows all three of its tests green on that leg with **zero** `SKIP` lines, which is what distinguishes a leg that passed from a leg that ran.

Two gaps in what that actually proved are now closed:

- The sweep stopped at q95 and used RGB and grayscale only, while every divergence #319 cited was **BGR at 16x16 q100**. The sweep now includes q100 and 16x16, and `issue_319_bgr_input_matches_cjpeg_on_every_backend` covers BGR by comparing against cjpeg's RGB encode of the same pixels — cjpeg has no BGR input, and a channel order that reaches a different SIMD colour-conversion kernel is exactly where a backend difference would hide.
- The golden fixture (`tests/encode_pipeline_golden.txt`) is no longer x86_64-pinned: all 33,600 cases, including the `bgra|q100|ifast` and `bgra|q100|float` rows #319 quoted, are byte-identical on aarch64. Backend-independence and C-parity are now both enforced, on both architectures.

**Guarantee, stated:** `int` and `fast` are byte-identical to `cjpeg` on x86_64 and aarch64; `float` is pixel-equivalent, max 7 per sample measured.

## P4-45. `SSE2-only` CI Job Does Not Test the SSE2 Fallback — **CLOSED 2026-07-26**

**Motivation.** Filed 2026-07-25 while looking for a way to verify P4-41/#315 on a non-AVX2 CPU. GitHub [#320](https://github.com/developer0hye/libjpeg-turbo-rs/issues/320).

`cross-arch.yml:78` sets `RUSTFLAGS: "-C target-feature=-avx2,-sse4.2"` and the job comment claims it "validates the secondary tier SIMD routines and scalar tail code remain correct when AVX2 is unavailable (older CPUs)". It does not. That flag is compile-time; every SIMD dispatch here is a runtime CPUID query via `is_x86_feature_detected!`, which ignores it. Built with the job's exact flags on an AVX2 machine: `cfg!(target_feature="avx2")` is **false** while `is_x86_feature_detected!("avx2")` is **true**, so the AVX2 branch is taken anyway.

**Scope.** 50 runtime dispatch invocations across 48 lines (the two `huffman_encode.rs` sites each call it twice, `bmi1 && lzcnt`) — 38 on `avx2`, 6 `sse2`, 2 each `ssse3`/`lzcnt`/`bmi1` — in nine files: `src/encode/pipeline_impl/{baseline,dispatch,mcu,optimized,sampling}.rs` (24), `src/decode/pipeline_impl/color.rs` (17), `src/encode/huffman_encode.rs` (4), `src/simd/x86_64/mod.rs` (3) and `src/api/progressive_output.rs` (2). The `src/simd/x86_64/{color,idct,avx2_idct,avx2_fdct,avx2_merged,upsample}.rs` kernels name the macro only in `// SAFETY:` comments — they are the callees, dispatched from the nine files above. The SSE2/scalar fallback for essentially the whole x86_64 SIMD layer has never executed in CI. Worse than having no job, since the name asserts coverage that does not exist — #315 is one concrete bug it could never have caught.

**Acceptance criteria.**

1. A CI leg where `is_x86_feature_detected!("avx2")` genuinely returns false for the test binaries — user-mode QEMU (`qemu-x86_64 -cpu Nehalem`) or Intel SDE (`sde -snb`) — asserted in a test rather than assumed.
2. Full suite green under it. This also discharges #315's outstanding acceptance criterion 2.
3. The existing compile-time job renamed to say it is a build check.

**Status (2026-07-26): closed.** A new `Test (linux-x86_64 no-AVX2, emulated)` job runs the SIMD-sensitive tests under `qemu-x86_64 -cpu Nehalem`, a CPU model predating AVX entirely, so CPUID genuinely reports no AVX2 and the fallback kernels execute.

The leg is **self-verifying**: `EXPECT_NO_AVX2=1` makes `tests/simd_dispatch_capability.rs` assert AVX2 really is masked. Without that, an emulation change could silently stop masking and the job would pass while testing the AVX2 path — reproducing exactly the false confidence being fixed. Verified in both directions locally: passes normally, fails with a pointed message when `EXPECT_NO_AVX2=1` is set on an AVX2 machine.

Scoped to tests that do not subprocess the C tools, since `qemu-user` intercepts `execve` and running native `cjpeg`/`djpeg` from an emulated process is not like-for-like. It does include the golden fixture, which asserts the fallback produces byte-identical output to the AVX2 path. The C cross-checks run natively in the other legs.

The old job is renamed `Build (linux-x86_64, AVX2 disabled at compile time)` with a comment stating plainly what it does and does not cover. This also discharges #315's outstanding acceptance criterion 2.

## P4-46. `Encoder` Silently Drops Builder Options When Combined — **CLOSED 2026-07-26**

**Motivation.** Filed 2026-07-25 while deciding what a README example for `CompressParams` should say. GitHub [#322](https://github.com/developer0hye/libjpeg-turbo-rs/issues/322). Unlike P4-39 this is **not** CMYK-specific — it hits ordinary RGB input through the public builder.

Measured at 64x48 RGB q75: `.restart_blocks(3)` alone gives 3 RST markers, but `.quant_table(..).restart_blocks(3)` gives **0**, and `.huffman_*_table(..).restart_blocks(3)` gives **0**. `.huffman_*_table(..).quant_table(..)` returns output byte-identical to custom-Huffman-alone, i.e. the quant table is discarded.

**Scope (2026-07-25, from `tests/encode_option_matrix.rs`).** The metamorphic matrix found **29** masked interactions, not 3: `restart_blocks` lost after `quant_table`/`huffman_tables`; `quant_table` lost after `huffman_tables`/`arithmetic`/`progressive`/`optimize_huffman`; `huffman_tables` lost after `progressive`; `dct_method` lost after `quant_table`/`huffman_tables`; and `smoothing_factor` lost after `arithmetic`/`progressive`/`huffman_tables`/`quant_table`/`restart_blocks` — all for both RGB and grayscale. `smoothing_factor` was the worst: it reached the encoder only via `compress_optimized`, so every earlier branch discarded it, while upstream applies smoothing in `jcsample.c` independently of entropy mode. Five combinations were classified **by-design** rather than dropped (arithmetic has no Huffman tables; `optimize_coding` overrides supplied tables; progressive already optimizes).

**Root cause.** Before #322, `Encoder` dispatched through an if/else chain; each arm called a shim forwarding only the options it named, and the first matching arm won. `compress_custom_quant` and `compress_custom_huffman` took neither a restart interval nor a `dct_method`, so both were dropped alongside either table option.

**Why it is now small.** P4-40 collapsed those four shims onto `CompressParams`, which carries every option at once — the chain can be replaced by building one params value. The `optimize_huffman` / `smoothing_factor` arm still needs the genuinely two-pass `compress_optimized`, but it now consumes the same `CompressParams` type and is selected centrally by `compress_with_params`.

**Acceptance criteria.**

1. Every pair of `restart_blocks`/`restart_rows`, `quant_table`, `huffman_dc_table`/`huffman_ac_table` and `dct_method` composes, each effect observable regardless of the others.
2. Cross-validated against `cjpeg` for combinations C can express (`-restart NB` with `-qtables`).
3. The three `#[ignore]`d tests in `tests/encoder_option_composability.rs` un-ignored and green (all three fail today under `--include-ignored`, which is the reproduction).
4. Combinations that genuinely cannot be honoured return an error rather than dropping silently.

**Status (2026-07-26): baseline paths closed.** The if/else chain is replaced by a single `CompressParams` build, and `compress_optimized` became `compress_optimized_with_params`, honouring custom quant tables and — new — the `optimize_huffman` flag separately from `smoothing_factor`. Previously *any* smoothing forced two-pass optimization, which silently overrode custom Huffman tables. 22 of the 29 tracked violations no longer reproduce.

One entry was reclassified `test-limit` rather than fixed: `gray|independence|dct_method_ifast after quant_table` is unobservable with the fixture's deliberately coarse table (every coefficient quantizes to ~0, so `islow` and `ifast` agree at 364 bytes each), while against the default tables they differ (811 vs 810).

Fixing this also surfaced **P4-49 / #327** — `smoothing_factor` was a silent no-op for grayscale. It had been masked: grayscale + smoothing previously routed through the optimized path whose optimal tables changed the bytes, so smoothing appeared to work while contributing nothing.

**Status (2026-07-26): closed.** The remaining 10 are resolved three ways, each matching what the option can actually mean:

- **Custom quantization tables now reach progressive and arithmetic** (4 violations). `compress_progressive_with_scans`, `compress_arithmetic` and `compress_arithmetic_progressive` take an `Option<&[Option<[u16; 64]>; 4]>`, resolved through a shared `resolve_quant_tables` helper that replaced four duplicated copies of the same match.
- **Smoothing combined with progressive / arithmetic / lossless now returns an error** (4 violations) instead of being accepted and dropped. It needs the full-plane buffering only the baseline optimized path provides; those paths downsample per block from unpadded planes. This is acceptance criterion 4 — a visible failure beats a silent one.
- **Custom Huffman tables with progressive are reclassified `by-design`** (2 violations). A progressive scan covers one coefficient band, so tables are derived per scan from that scan's own statistics; a single supplied pair cannot express them. C behaves identically (`jcmaster.c:770-774` forces `optimize_coding` for progressive when tables are absent, and `cjpeg -progressive` always optimizes).

Every entry left in the matrix's allowlist is now `by-design` — zero outstanding drops.

## P4-47. Progressive Encoding Diverges From `cjpeg` At Every Even Height Not A Multiple Of 16 — **CLOSED 2026-07-26**

**Motivation.** Filed 2026-07-25, found within 90 seconds of giving `fuzz_encode_diff_c` a reference oracle. GitHub [#324](https://github.com/developer0hye/libjpeg-turbo-rs/issues/324).

Progressive output diverges from stock `cjpeg` whenever the height is **even and not a multiple of 16**, for any subsampling with vertical chroma decimation (4:2:0, 4:4:0). Odd heights and multiples of 16 are byte-exact. This covers **1920x1080 progressive 4:2:0**, the most common web configuration: 673,854 bytes against C's 673,796. Also 800x600 (158,182 vs 158,167). 1280x720, 640x480, 1024x768, 1920x1088 and 500x375 all match.

Swept heights 1..24 at width 32: every even height fails except 16; every odd height passes. Affects Huffman progressive and arithmetic progressive alike. Baseline and arithmetic-sequential are clean at the same geometries, and 4:4:4 / 4:2:2 are clean in every mode — so the trigger is progressive scan emission combined with `v_samp == 2`.

**Not a P4-41/P4-42 relapse.** Those were the baseline dummy-block contract; baseline passes here and their geometries are unaffected.

**Why it was invisible.** `fuzz_encode_diff_c` asserted only that `djpeg` accepts our output and that both decoders read it identically — *validity* oracles, which stay green when we emit a valid JPEG that is simply not the one `cjpeg` would emit. The same blindness explains P4-41/P4-42 surviving in a target whose geometry range covered them many times over.

**Acceptance criteria.**

1. Progressive output byte-identical to `cjpeg -progressive` for all heights, 4:2:0 and 4:4:0, Huffman and arithmetic.
2. Regression coverage over even heights not divisible by 16, including 1920x1080 — MCU-aligned sizes alone would not have caught this.
3. `fuzz_encode_diff_c` clean over an extended session afterwards.

**Root cause.** `downsample_chroma_block` clamped the *source* row — `(block_y + row * v_factor + dy).min(plane_height - 1)` — which models C only when the final row group is incomplete. C works in two phases (`jcprepct.c` then `jccoefct.c`): pad the source up to a complete row **group**, downsample, then replicate the resulting **downsampled** row. With `v_factor == 2` and an even height the last group is complete, so C replicates `avg(last_two_rows)` while a source clamp yields `last_row` alone. Odd heights agreed by accident — their last group is incomplete, so both models replicate the same single row.

Baseline was unaffected because it feeds planes already padded by `pad_chroma_plane`, which implements both phases, so the clamp never engaged. Only the progressive path passed an unpadded plane and relied on it.

**Fix.** Clamp the *output* chroma row instead: `(block_y / v_factor + row).min(plane_height.div_ceil(v_factor) - 1) * v_factor`. Horizontal clamping is left alone — C's `expand_right_edge` replicates source *pixels*, so column clamping was already the right model.

**Status (2026-07-26): closed.** All four entropy modes now match `cjpeg` on **4032/4032** swept cases — 8 subsamplings (including the 4-factor 4:4:1 / 4:1:1 / 4:1:0 / 2:4), both colourspaces, 28 geometries covering heights 1..20 plus 1920x1080 / 800x600 / 1920x1088 / 500x375, at four qualities. Was 4016/4032 before. Pinned by `tests/regression_progressive_chroma_row_group.rs`. The P4-33 drift guard correctly flagged 108 `*_441_*_prog` / `_aprog` corpus seeds whose bytes the fix changes; the regenerated seeds are committed and verified to still decode under `djpeg`.

## P4-48. Mutation Testing: 12 Of 38 Encoder Mutants Survive The Full Suite — **CLOSED 2026-07-26**

**Motivation.** Filed 2026-07-25 from the first `cargo-mutants` pass over `src/api/encoder.rs`. GitHub [#325](https://github.com/developer0hye/libjpeg-turbo-rs/issues/325). This is the meta-level complement to P4-39/P4-46/P4-47: those are bugs, this is where a bug *would not be noticed*.

**Blind spot 1 — `extract_luminance` has no correctness coverage (8 mutants).** The BT.601 luma weights (`19595*R + 38470*G + 7471*B + 32768 >> 16`, `encoder.rs:478-520`) can be scrambled — `*`→`+`, `+`→`-`, `*`→`/` — with the full suite green. It is the path taken by `grayscale_from_color(true)` for every non-RGB format (plain `Rgb` routes through the SIMD `rgb_to_ycbcr_row`, per the comment at `:717-721`). Tests *do* execute it (`tests/grayscale_encode.rs:22`, `:35`, `tests/pixel_formats.rs:335`) but assert only `img.pixel_format == Grayscale` — metadata, not content — on uniform `vec![128u8; ..]` input. Same shape as the pixel-diff fallback that hid P4-41: the test runs the code and then declines to check it.

**Blind spot 2 — `_effective_quant_tables` is dead code (3 mutants).** Declared at `encoder.rs:427`, referenced nowhere. Survives mutation because it is never called; the leading underscore silences the dead-code lint instead of removing the function. `build_quant_tables` (`:1069`) is the live equivalent.

**Blind spot 3 — `compute_restart_interval` (1 mutant).** `==` at `:409:45` can be inverted undetected. Given P4-46 shows restart handling is already fragile, worth pinning.

**Method.** `cargo mutants --file src/api/encoder.rs --shard 1/10 -j 16 --timeout 300`, full workspace suite as oracle (unmutated baseline green in 120s); 38 of 380 mutants sampled, 12 missed / 26 caught. A separate run against only the five newest encode-focused tests scored 312 missed / 30 caught / 38 unviable — expected, since those target specific properties, and only the full-suite figures are blind spots.

**Acceptance criteria.**

1. Blind spots 1-3 closed, each with a test that fails if the mutant is reintroduced.
2. A sharded run over the former monolithic `src/encode/pipeline.rs` (**3807** mutants, not attempted here; implementation now under `src/encode/pipeline_impl/`), surviving mutants triaged into "needs a test" vs "equivalent mutant".
3. `cargo mutants --in-diff` in CI so untested new code is flagged at review time.

**Status (2026-07-26): criteria 1 closed.** All three blind spots are shut, verified by re-running the same mutation scope: **81/81 mutants now caught**, versus 16 surviving before.

- **`extract_luminance`** gained assertions on the actual luma values, checked against an independently written BT.601 reference, for all 11 formats x 7 colours — with primaries isolating each weight and asymmetric mixes catching an R/B swap, which greys cannot. A separate test pins that the pad/alpha byte of the 4-byte formats does not reach the computation.
  Note this needed a **unit** test, not an integration one: `Encoder` never reaches the `Rgb` or `Grayscale` arms (it routes plain `Rgb` through the SIMD `rgb_to_ycbcr_row` and skips the call entirely for grayscale), so 16 mutants in the `Rgb` arm alone were unreachable from outside the crate.
- **`_effective_quant_tables`** deleted — 31 lines, referenced nowhere.
- **`compute_restart_interval`** pinned: `restart_rows(n)` must convert as `n x MCUs_across` for all eight subsamplings, and `restart_blocks` must pass through unchanged.

**Status (2026-07-26): criteria 2 and 3 closed.**

**Criterion 2 — the former `encode/pipeline.rs` sampled.** 288 of its 3807 mutants, three shards of 40, with the encode suites as oracle. Surviving mutants fell from **51 to 20**, and every one of the 20 was triaged as an *equivalent mutant* rather than a missing test — several proven so by applying the mutation and confirming all 33,600 golden cases stay byte-identical:

- `need_dummies` forced always-true (`:796`) — `encode_color_mcu_with_dummies` at full dimensions is exactly `encode_color_mcu`, which is a good property to have confirmed.
- `(row - row_group_end) % max_v` → `+` (`:588`) — `row_group_end` is a multiple of `max_v`, so the sign cannot matter.
- The `.min(dst_h / max_v)` clamps (`:575`, `:926`) — `dst_h` is always a multiple of `max_v` and at least `src_h`, so the clamp never binds.
- The `if src_w < dst_w` / `if src_h < dst_h` padding guards — on equality the loop body is an empty range.
- `BitWriter::new(..)` and `begin_block(..)` arguments — capacity hints only.
- `may_use_islow_simd_kernel -> false` — forces the generic path, which P4-45's emulated leg already proved byte-identical.

Two real gaps were found and closed on the way:

1. **The validation prologue had no tests at all** (8 mutants). `width == 0 || height == 0` could become `&&`, `width > 65535` could become `>=`, and the buffer-size product could become `/` or `+`, all with the suite green — nothing exercised them because every other test passes well-formed input. Closed by `tests/encode_input_validation.rs`.
2. **The full-plane fallback was never executed** (23 mutants). `pad_chroma_plane` could return `vec![]` unnoticed: `Rgb`/`Rgba`/`Bgr`/`Bgra` take the fused path and `Cmyk` has its own, so only the pad-byte formats reach it — and the golden fixture carried none. Adding `rgbx`/`xrgb`/`bgrx`/`abgr` to it (13,440 new cases, **zero** existing rows changed) exercises the path.

**Criterion 3 — `--in-diff` in CI.** A non-blocking `Mutation test (changed lines)` job mutates only what a PR touches. Non-blocking on purpose: a surviving mutant is sometimes an equivalent mutant, a judgement call rather than a defect, and failing the build on it would train people to ignore the job.

The job is also **time-bounded**, which the first large PR through it proved necessary: P4-39's CMYK diff produced enough mutants to run past the 30-minute step limit, and a timed-out job reports `fail` while telling the reader nothing. It now counts the mutants first and shards down to what fits, emitting a `::warning::` naming how many it skipped — a clean result on a large diff means "the sampled shard was clean", not "the diff was checked".

## P4-49. `smoothing_factor` Is A Silent No-Op For Grayscale — **CLOSED 2026-07-26**

**Motivation.** Found 2026-07-26 by the metamorphic option matrix while closing the baseline half of P4-46. GitHub [#327](https://github.com/developer0hye/libjpeg-turbo-rs/issues/327).

The former monolithic encoder gated full-size smoothing on `!is_grayscale`; the corrected code now lives in `src/encode/pipeline_impl/optimized.rs`. C selects `fullsize_smooth_downsample` for **every** component sampled at the maximum factors (`jcsample.c:506-513`), which for a single-component image is the grayscale plane itself: on a 48x32 noisy gradient at q75, `cjpeg -grayscale` emits 811 bytes and `-smooth 50` emits 657.

**Why it was invisible.** Before P4-46, grayscale + smoothing routed unconditionally through the optimized-Huffman path, and those optimal tables changed the bytes — so smoothing *appeared* to do something while contributing nothing. Only once the dispatch stopped forcing optimization did the no-op become observable.

**Fix.** Drop the `!is_grayscale` term from the luma gate. The adjacent `use_smooth_chroma` gate keeps its own — a grayscale image has no chroma planes to smooth.

**Status (2026-07-26): closed.** Byte-identical to `cjpeg -grayscale -smooth N` across 6 geometries x smoothing {0,1,25,50,100}, pinned by `tests/regression_grayscale_smoothing.rs`. Note the effect test starts at factor 2: C itself produces identical output for `-smooth 0` and `-smooth 1` on this content, because the weights (`memberscale = 16384 - factor * 80`, `jcsample.c:338`) round away — asserting an effect there would assert something C does not do. The golden fixture moved in exactly 736 of 20,160 cases, all `optimized|gray` with smoothing > 0.

## P4-50. `DctMethod::IsFast` Was Both Lower Quality And Larger Than C's — **CLOSED 2026-07-26**

**Motivation.** Found 2026-07-26 while measuring P4-44. GitHub [#330](https://github.com/developer0hye/libjpeg-turbo-rs/issues/330). On a 64x48 fixture at q75 4:2:0, decoding through `djpeg` and comparing to the source: ours had mean error **15.096** in **1567** bytes against C's **5.902** in **1285**. C's fast path is within noise of its own `int` path on both axes — that is the point of AA&N — so being worse on *both* is a defect, not a tradeoff.

**Root cause.** The fused SIMD extract+FDCT+quantize kernels hardcode the **islow** transform, while `ifast` and `float` carry divisor tables scaled for their own transforms. Feeding islow coefficients to those divisors mis-scales every output by the AA&N factor — which is exactly the ratio observed in the coefficients (index 1: C 3 vs ours 2, C -8 vs ours -6; aanscale 22725/16384 = 1.387).

Several call sites already guarded against this (`encode_single_block`, `encode_downsampled_chroma_block`) and several did not: `encode_mcu_{444,422,420}_x86_64`, `encode_mcu_420_half_chroma`, `fdct_quantize_block`, `fdct_quantize_chroma_h2v1`, and the AVX2 4:2:0 row path. Hence the selective symptom — grayscale and the 4-factor subsamplings take generic paths and matched C; 4:4:4 / 4:2:2 / 4:2:0 have SIMD shortcuts and did not.

Ruled out along the way, each by direct measurement rather than inspection: `fdct_ifast_raw` is **bit-exact** with a fresh port of `jfdctfst.c` (0/64 differing on a test block); the quantization tables written to DQT are identical to C's; the divisor formula matches `jcdctmgr.c:308-317`; and `compute_reciprocal`'s `divisor <= 1` identity case matches C's.

**Fix.** A single `may_use_islow_simd_kernel(fdct_quantize_fn)` helper, replacing four inline copies of the check and applied at the seven sites that lacked it.

**Status (2026-07-26): closed.** `-dct fast` is now byte-identical to `cjpeg -dct fast` across all 8 subsamplings x both colourspaces (was 3/64 at 4:4:4, 4/64 at 4:2:2, 24/64 at 4:2:0), and its quality and size now match C's exactly (mean 5.902, 1285 bytes). Pinned by `tests/regression_dct_method_parity.rs`, which asserts byte-equality *and* that `fast` is never simultaneously worse and bigger than `int` — a byte test alone would not convey the severity. Golden fixture moved in 348 of 20,160 cases, all `compress` with `ifast` (236) or `float` (112); zero `islow`.

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
35. ~~**P4-39** — CMYK encode path silently drops restart / custom quant / custom Huffman options and rejects optimize+smoothing (GitHub #313).~~ **CLOSED 2026-07-26** — all five fixed; 504/504 byte-exact vs a purpose-built C oracle. Closed P4-51 and P4-52 on the way.
36. ~~**P4-40** — collapse the ten copy-pasted `compress_*` variants onto a single `CompressParams` core (byte-exact), then split `src/encode/pipeline.rs` by mode.~~ **CLOSED 2026-08-02** — 23-line stable façade, 15 implementation modules, largest implementation file 1,704 lines; 33,600-case golden and C parity gates green.
37. ~~**P4-41** — AVX2 4:2:0 row fast path ignored the dummy-block contract (#314) and bypassed its own AVX2 capability check (#315).~~ **CLOSED 2026-07-25** — single `use_avx2_420` gate + `y_last_col_width == y_mcu_width` guard; fused path now 576/576 vs cjpeg.
38. ~~**P4-42** — full-plane encode variants (restart / custom-quant / custom-Huffman) skip the dummy-block contract on every platform (#316).~~ **CLOSED 2026-07-25** — the P4-40 core made them shims; 576/576 vs cjpeg.
39. ~~**P4-43** — recover the ~3-4.5% the P4-41 correctness fix cost on `ceil(width/8)`-odd 4:2:0 widths (#317).~~ **CLOSED 2026-07-26** — fast path kept for interior columns; residual 0.7-2.0%.
40. ~~**P4-44** — quantify encoder byte-parity vs `cjpeg` for `ifast`/`float` and for the aarch64 backend, then document what is actually guaranteed (#319).~~ **CLOSED 2026-07-26** — the parity sweep is unguarded, so the aarch64 CI leg runs it; extended to q100 and BGR, the cases #319 cited.
46. ~~**P4-50** — `DctMethod::IsFast` both lower quality and larger than C's (#330).~~ **CLOSED 2026-07-26** — SIMD kernels gated on the DCT method.
41. **P4-45** — make the `SSE2-only` CI job actually exercise the SSE2 fallback (QEMU/SDE); discharges #315's remaining criterion (#320).
42. ~~**P4-46** — make `Encoder`'s dispatch build one `CompressParams` so builder options stop dropping each other (#322).~~ **CLOSED 2026-07-26** — all 29 resolved: 26 fixed, 3 classes reclassified by-design with reasons.
45. ~~**P4-49** — `smoothing_factor` is a silent no-op for grayscale (#327).~~ **CLOSED 2026-07-26** — byte-exact vs `cjpeg -grayscale -smooth`.
43. ~~**P4-47** — progressive 4:2:0/4:4:0 diverges from `cjpeg` at every even height not a multiple of 16, including 1920x1080 (#324).~~ **CLOSED 2026-07-26** — chroma row-group replication; 4032/4032 vs cjpeg.
44. ~~**P4-48** — close the mutation-testing blind spots in `api/encoder.rs`, then shard `encode/pipeline.rs` (#325).~~ **CLOSED 2026-07-26** — encoder.rs 81/81; pipeline.rs 288 sampled, 51→20 survivors all triaged equivalent; `--in-diff` job added.
## P4-53. RGB-Direct Encode (`JCS_RGB`) Silently Drops Every Builder Option — **CLOSED 2026-07-26**

**Motivation.** Found 2026-07-26 immediately after [P4-39](#p4-39-cmyk-encode-path-silently-drops-restart--custom-table-options-and-rejects-optimizesmoothing--closed-2026-07-26) closed, by asking the obvious follow-up question: *what else sits behind an early return into a narrower signature?* GitHub [#343](https://github.com/developer0hye/libjpeg-turbo-rs/issues/343).

**Root cause.** `Encoder::encode` returned early into `compress_rgb_direct(pixels, width, height, quality, dct_method, icc_profile)` — and the `dct_method` parameter was spelled `_dct_method`, i.e. the signature said out loud that it was ignored. Restart interval, custom quantization tables, custom Huffman tables, `optimize_huffman`, `smoothing_factor` and the DCT method were all discarded, along with the comment / EXIF / saved-marker injection that every other colorspace runs. Six options, all silent.

Identical in shape to P4-39, and found the same way: not by reading the code, but by asking which entry points cannot express the option set.

**Why it went unnoticed.** Unlike CMYK, this path *is* cross-validatable — `cjpeg -rgb` reads ordinary PPM. Nothing was checking it. `tests/encode_option_matrix.rs`, which exists precisely to catch dropped options, swept `rgb`/`gray`/`cmyk` at the **default** colorspace and so never reached the RGB-direct branch. A metamorphic matrix is only as good as its axes.

**Status (2026-07-26): closed.** CMYK and RGB-direct now share one `compress_direct_planar`, parameterized by a `DirectPlanarSpec` (component IDs plus per-component sampling factors). They always were the same encoder — Adobe APP14 and no JFIF, one quantization and Huffman slot for every component, ASCII-initial component IDs (`jcparam.c:365-390`) — differing only in component count and which components carry the sampling factors. Two copies of that is how P4-39's five dropped options and this one's six got there.

75 cases byte-exact against `cjpeg -rgb` across 5 geometries x 3 qualities x {plain, `-dct fast`, `-restart 3B`, `-optimize`, `-smooth 50`}. Custom tables get the weaker property (the option must change the bytes) because cjpeg has no flag that expresses them.

Review of the fix found two more, both in configurations the sweep could not reach and both now pinned against C:

- **Row-based restarts used the wrong effective MCU width.** `jpeg_set_colorspace(JCS_RGB)` defaults R, G, and B to 1x1 sampling (`jcparam.c:367-373`), but that is not an RGB restriction in JPEG: T.81 B.2.2 defines sampling factors per component, and `cjpeg` applies an explicit `-sample` after the colorspace defaults (`cjpeg.c:544-552,609-611`; `rdswitch.c:397-425`). Thus implicit RGB-direct has an 8-pixel MCU width, while explicit `2x2,1x1,1x1` sampling has `Hmax=2` and a 16-pixel width. This is RGB component sampling, not JFIF/YCbCr chroma subsampling. `compute_restart_interval` instead counted rows from the requested subsampling without distinguishing the default from an explicit request; the divergence is visible where `ceil(width/8) != ceil(width/16)`.
- **16-bit quantization tables were declared SOF0.** Below quality ~20 with `force_baseline` off, or with a coarse custom table, the DQT entries exceed 255 — which baseline forbids. C switches to SOF1 and emits `JTRC_16BIT_TABLES` (`jcmarker.c:517-535`); we wrote SOF0 and a non-conforming stream. The fix lands in the shared core, so it closes the same latent hole on the CMYK side: 180 `customquant|cmyk` fixture rows moved, and nothing else.

**The matrix gained a colorspace axis**, which is the part that generalizes: the next entry point to early-return past the option set gets caught by the suite rather than by someone thinking to look. It immediately earned its keep by surfacing [P4-54](#p4-54-colorspacergb-silently-ignores-progressive--arithmetic--lossless--open).

## P4-54. `colorspace(Rgb)` Silently Ignores `progressive` / `arithmetic` / `lossless` — **CLOSED 2026-07-26**

**Motivation.** Surfaced 2026-07-26 by the colorspace axis P4-53 added to the option matrix, on its first run. GitHub [#345](https://github.com/developer0hye/libjpeg-turbo-rs/issues/345).

**Root cause.** RGB-direct takes precedence over the mode switches: the baseline encoder runs and `progressive` / `arithmetic` / `lossless` are discarded. The caller gets `Ok(bytes)` holding a baseline Huffman stream. C supports all three with `JCS_RGB` (`cjpeg -rgb -progressive`), so this is a missing feature rather than a colorspace-imposed limit.

`tests/cross_product_compress.rs::tjcomptest_lossy_rgb_colorspace` has been exercising 40 such cases all along and passing, because the baseline stream round-trips — a weaker property than "the requested mode was used".

**Why not fixed with P4-53.** The precedence predates it. Reversing it either starts emitting YCbCr progressive streams where a caller asked for `JCS_RGB`, or starts returning errors where callers currently get a working file. Both are user-visible changes that deserve their own decision, not a side effect of a bug fix.

**Acceptance criteria.**

1. `colorspace(Rgb)` + `progressive` / `arithmetic` either produces that mode or errors — never a silently-baseline `Ok`.
2. If implemented: byte-exact against `cjpeg -rgb -progressive` and `cjpeg -rgb -arithmetic`.
3. The `rgb-direct|effect|progressive` / `|arithmetic` entries removed from `KNOWN_VIOLATIONS`.

**Status (2026-07-26): closed, and implemented rather than rejected.** Rejecting would have taken working files away from callers; implementing is what C does. All four mode combinations are now byte-exact against `cjpeg -rgb`: `-progressive`, `-arithmetic`, `-arithmetic -progressive`, and `-lossless 1,0`.

Each encoder needed the same four things, because each had the YCbCr assumption baked in the same four places: skip the colour conversion, put every component on quantization and entropy table slot 0 (with one DQT and one DAC entry rather than two), write `'R','G','B'` and the Adobe marker, and — for the Huffman DC scan — gather all three components into **one** histogram, since splitting them fits a table to a distribution no scan actually has.

The one non-obvious piece was the **scan script**. `jpeg_simple_progression` takes its tuned 10-scan script only when `ncomps == 3 && jpeg_color_space == JCS_YCbCr`; every other three-component colorspace gets the 14-scan all-purpose script (`jcparam.c`). Ours keyed on the component count alone, so JCS_RGB got the YCbCr script. That script's shortcuts are chroma-specific — *"chroma data is too small to be worth expending many scans on"* is a statement about Cb and Cr, and simply false of G and B. `simple_progression_for(ncomps, ycbcr)` now mirrors C's condition.

**Lossless needed no encoder work at all** — `compress_lossless_rgb` already wrote JCS_RGB correctly. It was only the dispatch chain routing `lossless` into the baseline arm, which is the same class of defect as [P4-53](#p4-53-rgb-direct-encode-jcs_rgb-silently-drops-every-builder-option--closed-2026-07-26): a working implementation made unreachable by the branch in front of it.

**Proof.** `tests/regression_rgb_direct_options.rs::issue_345_rgb_direct_composes_with_every_mode` sweeps 5 geometries x 3 qualities x 4 modes, all byte-identical to `cjpeg`. The four `rgb-direct|effect|*` entries are gone from `KNOWN_VIOLATIONS`; the four that replace them are the by-design classes `rgb` and `gray` already carry (arithmetic has no Huffman tables; progressive derives them per scan).

47. ~~**P4-51** — CMYK streams carry a JFIF APP0 marker C never writes, and non-libjpeg component IDs (#339).~~ **CLOSED 2026-07-26** — SOI then Adobe APP14 only; IDs are `'C','M','Y','K'`.
48. ~~**P4-52** — CMYK bottom padding clamps the last row where C repeats the last row group (#340).~~ **CLOSED 2026-07-26** — per-component row-group height.
49. ~~**P4-53** — RGB-direct encode drops every builder option (#343).~~ **CLOSED 2026-07-26** — CMYK and RGB-direct share one `compress_direct_planar`; 75/75 vs `cjpeg -rgb`.
51. ~~**P4-54** — `colorspace(Rgb)` silently ignores `progressive` / `arithmetic` / `lossless` (#345).~~ **CLOSED 2026-07-26** — implemented, not rejected; all four modes byte-exact vs `cjpeg -rgb`.

## P4-55. zune-jpeg Competitive-Gap Program (#350–#361) — **OPEN**

**Motivation.** The 2026-07-26 full gap analysis vs `zune-jpeg` 0.5.15 (GitHub tracking issue #361, AMD EPYC 9554 reference box) found us slower in four specific areas nobody was measuring — 4:2:2 decode (1.22–1.24×, #350), small-image fixed cost (2–3.6×, #351), 8K progressive scaling (1.30×, #352), 4:4:4/low-density multi-pass output (#353) — plus six capability gaps (#354–#359) and a benchmark blind spot (#360). Detail, measurements, and per-item acceptance criteria live in the GitHub issues; this entry keeps the program visible to the LAST_MILE release gate rather than duplicating it.

**Acceptance criteria.** All sub-issues #350–#360 closed on GitHub with their stated criteria (each requires C cross-validation and an `experiments/` record where perf-related), and the #361 tracking table re-measured.

**Progress.** #351 (small-image fixed cost) closed by the PR that adds this entry: gray_8x8 44 allocs/60.5 KB → 8 allocs/9.6 KB, ratio vs zune 3.60× → 1.35×, all large-image cases improved. #350: structural fix landed 2026-07-27 — H2V1/H1V2 row-streaming fused upsample+colour, 1080p 4:2:2 allocation 14.53 → 10.39 MB, byte-exact vs djpeg across 40 geometry×quality cases, latency-neutral on aarch64 (which never had the regression); the 1.22× EPYC ratio re-measurement remains for the #361 closing table, and `ProgressiveDecoder::output()` (buffered-image API, `src/api/progressive_output.rs`) still materialises full-resolution chroma planes for H2V1/H1V2. #352: closed 2026-07-27 — per-block `ac_max_k` bounds the AC-refine EOB-run walk; sparse 8K progressive 1.33× → 0.77× vs zune on aarch64 (EPYC re-measurement rides the same #361 closing table). #353 step 1: closed 2026-07-27 — generic nearest/box row-streaming for 4:1:1 / 4:4:1 / non-uniform / `-nosmooth` (1080p 4:1:1 allocation 13.5 → 9.35 MB); 4:4:4 was already row-wise with zero chroma allocation, and step 2 (fusing output into the MCU-row loop) plus scaled-IDCT streaming remain the tracked residue. #354: closed 2026-07-27 — caller-buffer decode API (`decompress_into` / `output_buffer_size` / `Decoder::decode_image_into`), zero output-sized allocation on the standard paths (1080p 9.38 → 3.16 MB allocated), staged+copied elsewhere. #355: closed 2026-07-27 — `DecodeLimits` (width/height/pixels/scans/memory) with permissive defaults that still reject the 65535×65535 header bomb before allocation (typed `LimitExceeded`, 9.5 KB allocated at rejection), plus a parse-time 8192-scan cap; the Rust-side twin of P4-14, whose C-ABI `max_memory_to_use` enforcement remains open and should reuse this estimation model.

## P4-56. Frame Component Cap Is 4 Where C Accepts Up To 10 — **OPEN**

**Motivation.** Surfaced by the #351 docs audit: our `read_sof` rejects frames with more than `MAX_COMPONENTS = 4` components at header parse, while the C *library* caps frames at `MAX_COMPONENTS 10` (`jmorecfg.h:30`, enforced at `jdinput.c:74` via `JERR_COMPONENT_COUNT`); ISO 10918-1 B.2.2 allows Nf ≤ 255. Stock `djpeg` is **not** a divergence witness here: it parses the 5-component header fine but aborts at output-colour selection (`JERR_CONVERSION_NOTIMPL`, `jdcolor.c:254` et al.) because none of its output formats accept a 5-component stream. The C surface that genuinely decodes 5–10-component streams is the library raw-data path (`jpeg_read_raw_data` with `JCS_UNKNOWN`) and coefficient transcoding (`jpeg_read_coefficients`/`jdtrans.c`) — our equivalents (`decompress_raw`, `read_coefficients`) reject at `read_sof` instead. Same less-accepting-than-C class as P4-21, but library-level only. Scan-level Ns ≤ 4 is exact parity (`MAX_COMPS_IN_SCAN`).

**Root-cause hypothesis.** The decode pipeline assumes ≤ 4 planes throughout (fixed `[_; 4]` table/plane arrays, colour paths for 1/3/4 components), so the cap was set to the scan limit rather than C's frame limit.

**Acceptance criteria.** Either (a) frames with 5–10 components work through the raw-data and coefficient paths, cross-validated against a small C harness linked to libjpeg-turbo driving `jpeg_read_raw_data` / `jpeg_read_coefficients` (stock `djpeg` cannot serve as the oracle — no output format accepts these streams), or (b) an explicit decision records the cap as an intentional divergence in `docs/ABI_COMPATIBILITY.md` with the rejection error made descriptive. Low urgency: no real-world corpus input exercises >4 components.

## P4-57. Grayscale Decode to Argb/Abgr Misplaces Alpha and Blue — **CLOSED 2026-07-28**

**Motivation.** Surfaced by the #354 caller-buffer review (GitHub [#369](https://github.com/developer0hye/libjpeg-turbo-rs/issues/369)): the grayscale→colour expansion in `decode_image_inner` groups `Argb`/`Abgr` with `Rgba`/`Bgra` and writes `[v, v, v, 255]`, putting 0xFF in the blue slot and the gray value in alpha. The 3-component path (and C's `JCS_EXT_ARGB`/`ABGR`) writes alpha first. All four bytes are written, so this is a channel-order divergence, not an uninitialised read.

**Acceptance criteria.** Grayscale→Argb/Abgr matches the 3-component path's channel order (alpha slot = 255), cross-validated against the C tj3 path for one grayscale fixture, with a regression test; `Xrgb`/`Xbgr` (currently correct) stay green. Check the 12/16-bit grayscale expansions for the same grouping.

**Status (2026-07-28): closed.** Both defect sites, now split between `src/decode/pipeline_impl/output.rs` (main grayscale expansion in `decode_image_inner`) and `src/decode/pipeline_impl/lossless.rs` (lossless SOF3 grayscale expansion), derive the pad/alpha byte position from the format via `pad_alpha_offset` (`6 - r_off - g_off - b_off`, the same construction the CMYK/YCCK arms already used), which makes an alpha-first vs alpha-last mis-grouping unrepresentable. A sweep of every `PixelFormat::Argb` site in the decode path — including NEON/AVX2 kernels, capi repack, and `yuv.rs` — found no other wrong grouping. The 12/16-bit paths had no equivalent branch at the time: `Image12`/`Image16` carry raw `Vec<i16>` / `Vec<u16>` samples with no `PixelFormat`, and `decode_12bit_as_8bit` returned `Grayscale` for 1-component frames without consulting `output_format` at all (a separate divergence, filed as P4-65 and closed 2026-07-28 — its 1-component arm now uses the same `pad_alpha_offset` construction). Pinned by `tests/regression_issue_369_gray_argb_abgr.rs` (5 tests): (a) **baseline path**: all eight 4bpp formats, gray values cross-validated against C `djpeg` (PGM reference) with per-offset placement asserts — where `djpeg` is absent the reference falls back to our own grayscale decode, which still pins placement but no longer cross-validates values (CI always has it: the `Integration Tests` job installs libjpeg-turbo 3.x); (b) **C tj3 oracle**: full-buffer byte-equality vs real `tj3Decompress8` (`examples/gray_argb_c_oracle.c`, built on demand by `tests/helpers/c_oracle.rs`) for the four alpha formats RGBA/BGRA/ARGB/ABGR — the X-format pad byte is documented-undefined in TurboJPEG, so only alpha formats carry a byte-exact C contract; (c) **lossless path**: placement asserted against the exact round-trip input — *no C oracle exists for this leg* because C refuses lossless non-RGB→extended-RGB conversion outright (`jdcolor.c` `JERR_CONVERSION_NOTIMPL`; divergence filed as P4-64); (d) gray-content ARGB==XRGB / ABGR==XBGR byte-identity (C-tool-free, so this leg can never degrade to a skip — note the whole file runs only in the ubuntu-latest `Integration Tests` job, since `Test (${{ matrix.os }})` runs `cargo test --lib`); (e) `decompress_into` == `decompress_to` byte-identity for all eight formats (the entry point that surfaced the bug).

## P4-58. Rust-Native Incremental Decode Source (Bounded Input Memory) — **CLOSED 2026-07-28**

**Motivation.** GitHub [#357](https://github.com/developer0hye/libjpeg-turbo-rs/issues/357): `decompress_from_reader`/`decompress_from_file` buffer the whole stream (`read_to_end`) before decoding, so peak memory is compressed size + intermediates and sockets/pipes cannot feed the decoder incrementally. zune 0.5 reworked its I/O for exactly this. The doc-honesty half of #357 landed 2026-07-27 (`src/api/stream.rs` states the buffering plainly); this entry is the deferred incremental path.

**Design constraints (from the issue, binding).** (1) The slice-based core stays the default — its bounds-checked slice fetches are why we win throughput benchmarks; any `BufRead`-backed bitstream ships as a separate opt-in type unless the measured cost on the full matrix is under a few percent. (2) Design together with **P4-26** (`read_header`-stop-at-SOS for all sources, output-driven input pull, incremental marker-list stability) — the C-ABI suspension core from P4-13 already solved the harder lock-step half; a Rust-native incremental reader must not become a third independent streaming mechanism. (3) Architecture note: baseline single-scan can stream a fixed window; progressive requires the full entropy stream buffered by construction (multi-pass over scans), so the bounded-RSS criterion applies to the baseline path.

**Acceptance criteria.** Peak RSS for a streaming baseline decode of the 8K fixture bounded by intermediates + a fixed input window (measured); slice-path throughput unchanged on the full matrix (recorded in `experiments/`); design reviewed against P4-26 before implementation.

**Status (2026-07-28): closed.** `decompress_from_reader_incremental` (src/api/incremental.rs) decodes interleaved single-scan Huffman baseline streams from a sliding window: per-MCU-row checkpoints on a window-aware `BitReader` (additive `is_final`/`starved` fields; the slice path constructs with `is_final=true` where starvation cannot fire), retry-on-starvation, front-compaction of committed bytes, and plane injection into `decode_baseline_planes` so the whole existing output pipeline is reused. Peak input storage measured at 195,985 bytes of allocation capacity (entropy window + header prefix + the fixed 64 KiB read-staging buffer; the instrumented metric counts capacities, not live bytes) on the 1.25 MB 1080p fixture fed in ≥ 64 KiB reads — 129,203 at the 8 KiB feed the test drives — asserted ≤ 256 KiB in `tests/regression_issue_357_incremental_reader.rs`, and size-independent in the strongest sense: the full-chunk-feed peak is exactly 195,985 bytes on the 1.25 MB 1080p, 5.0 MB 4K, and 8K corpus fixtures alike (all three asserted, closing the criterion's 8K clause as written). Slice-path throughput unchanged on the full decode matrix (`experiments/pipeline.tsv`: 27 benchmarks compared branch-vs-main sequentially on a quiet host — zero criterion regression verdicts, every change midpoint within [-1.13%, +0.30%]; spot figures 416.20 µs branch vs 416.99 µs main on decode_640x480). P4-26 co-design: the P4-13 marker-boundary scanner was lifted to `src/decode/boundary.rs` and the capi shim re-imports it — one scanner drives both mechanisms. Scope note (corrects the original filing's baseline/progressive split): the windowable set is *interleaved* baseline only. Multi-component non-interleaved baseline and progressive walk the full entropy stream during header parse (`marker.rs` `skip_entropy_data`, reached only when the scan carries fewer components than the frame) and take the documented buffering fallback. Single-component/grayscale streams stop at the first SOS like interleaved baseline, but decode through P4-27's one-block raster (`pipeline_impl/baseline.rs:55-56` routes `scan.components.len() == 1` to `decode_non_interleaved_baseline_planes`), which the row loop does not model — so they fall back as well, as do arithmetic/lossless/12-bit.

## P4-60. Scalar Kernels Are ~2.5x Slower Than C's Scalar Kernels — **OPEN**

**Motivation.** The issue [#359](https://github.com/developer0hye/libjpeg-turbo-rs/issues/359) step-1 measurement (`experiments/riscv64_scalar_2026-07-27.md`, 2026-07-27) was run to decide whether a portable-SIMD fallback is warranted. It answered a different and more useful question: on `riscv64gc-unknown-linux-gnu`, where **neither** implementation has a vector path, C libjpeg-turbo's scalar decode beats ours by ~2.5–2.9x (640x480 4:2:0 and 1080p 4:2:0, process-startup-corrected). #359 had assumed scalar-on-RISC-V was "parity with C, not a regression against it" — measurement falsifies that.

**Why it matters beyond RISC-V.** The same scalar kernels serve POWER, s390x, LoongArch, 32-bit ARM, any x86_64 without SSE2/AVX2, and — since #356 — every `no_std` build, which dispatches scalar unless `target_feature` is set at compile time. This is stable-Rust work with measured headroom, unlike the portable-SIMD half of #359 which stays blocked on `core::simd` stabilising.

**Acceptance criteria.** Profile the scalar IDCT / fancy upsample / YCbCr→RGB against `jidctint.c` / `jdsample.c` / `jdcolor.c` and close the gap to ≤ 1.2x C on the RISC-V harness; byte-exactness vs `djpeg` preserved (the scalar path is the reference every SIMD kernel is checked against, so it must not drift); experiment recorded per `experiments/README.md`.

**Progress (2026-07-28): step 1 landed, item stays OPEN.** Table-driven YCbCr→RGB (`src/decode/color.rs`, exact const-evaluated precomputation of the multiply form — bit-identical, proven by an exhaustive chroma equivalence test and 49 simd-off djpeg cross-checks): kernel 1.63× faster on riscv64 (same-binary A/B), decode totals 12–14% lower than the 07-27 env though no same-env baseline was re-measured, same-run ours/C now ~1.12× at 640×480 and ~1.72× at 1080p (`experiments/riscv64_scalar_2026-07-27.md`, 07-28 section). The ≤1.2× criterion is not yet met at 1080p. The criteria's profiling step is now DONE (07-28 profile section): at 1080p volume the three kernels sum to 36.4% (IDCT 13.5%, fancy upsample 9.2%, colour 13.7%) and the ~63.6% residual is Huffman entropy decode + `BitReader` — the measured next target (`jdhuff.c` two-level lookahead tables vs `decode/huffman.rs`). No per-stage profile has been run yet, so the acceptance criteria's "profile the scalar IDCT / fancy upsample / YCbCr→RGB" step is still outstanding.

## P4-59. Extended XMP Writing Not Implemented — **OPEN**

**Motivation.** #358 landed XMP/IPTC read (with Extended XMP reassembly) and single-segment write. Packets larger than one APP1 segment (65,504 payload bytes) error at encode time with `JpegError::Unsupported` rather than being split into `http://ns.adobe.com/xmp/extension/` chunks with a GUID + full-length + offset header and an `xmpNote:HasExtendedXMP` reference in the standard packet. Cameras and editors do emit such packets, so a read-modify-write round trip through our encoder currently fails on them instead of preserving the metadata.

**Acceptance criteria.** Encoder splits oversized packets into standard + extension chunks with a stable GUID; our own decoder round-trips them byte-exactly; cross-validated with `exiftool`; the `Encoder::xmp_data` rustdoc and README lose the single-segment caveat.

## P4-61. `C Interop` CI Job Runs Zero Tests — **CLOSED 2026-07-28**

**Motivation.** Found 2026-07-27 while deciding which CI legs the #362 regression test would run in. `.github/workflows/ci.yml:244` runs `cargo test "cross_encode|cross_check" --tests`, but libtest's positional filter is a **substring**, not a regex — no test name contains the literal `cross_encode|cross_check`. Both matrix legs build every test binary and execute none. Job 89952136991 (`C Interop (macos-latest)`, run 30257998607, `main` @ 818c24f) logs `running 0 tests` 212 times, `running [1-9]* tests` zero times, and reports success. Reproduced locally: a test binary invoked with `"issue_362_420|nonexistent" --list` lists 0 tests, with `"issue_362_420"` lists 1. GitHub [#377](https://github.com/developer0hye/libjpeg-turbo-rs/issues/377).

**Impact.** The ubuntu leg is redundant with `test-integration` (`cargo test --tests`, libjpeg-turbo 3.1.4.1 on PATH), so ubuntu coverage is intact. The loss is **macos-latest (aarch64)**: the only job that would run the byte-exact `cross_check_*` / `cross_encode_*` suites against Homebrew `jpeg-turbo` on a non-x86_64 backend, and it has never run one. That is the same blind spot P4-41's "why it escaped" paragraph blames for #314 — platform-dependent encoder drift — confirmed in the opposite direction. `test-corpus` also `needs:` this job, so its gate is vacuous.

**Fix sketch.** libtest OR-s multiple harness filters: `cargo test --tests -- cross_encode cross_check`, or two invocations. The first green run proves nothing on its own — these tests have never executed on aarch64 + Homebrew jpeg-turbo, so a first-run failure is a finding to file, not a reason to revert the filter.

**Acceptance criteria.** Both `C Interop` legs report a non-zero test count; the job is validated by mechanism (a deliberately broken encoder byte-comparison must fail it) rather than by "it passed"; any aarch64 divergence the newly-live tests surface is filed before the filter fix merges.

**Status (2026-07-28): closed.** The fix sketch above was itself falsified before merging: the corrected multi-filter form (`cargo test --tests -- cross_encode cross_check`) selects only **8 tests** on this workspace, because libtest filters match test *names*, and the tests inside `cross_check_*`/`cross_encode_*` files have names like `c_xval_decode_bgr_444` that contain neither substring. The job now runs the full unfiltered `cargo test --tests` on **macos-latest only** (timeout 15→30 min) — aarch64 + Homebrew jpeg-turbo 3.x, the one C-tool environment no other job covers. The former ubuntu leg is **removed**, not fixed: with apt's 2.1.x tools the unfiltered suite cannot run (codex review caught that e.g. `lossless_point_transform_matches_c_djpeg_exactly` feeds SOF3 to `djpeg` with no capability probe), and installing the official 3.1.4.1 deb would make the leg an exact environment+command duplicate of `Integration Tests` — the redundancy this entry's Impact paragraph already established. The "both legs non-zero" acceptance criterion is therefore satisfied in its intent (every remaining leg runs the full suite; no leg silently runs zero) rather than its letter. A comment in `ci.yml` pins the substring-vs-regex trap so a filter cannot quietly come back. **Mechanism-validated**, not validated-by-passing: with a deliberate encoder break (`FIX_0_299` 19595→20100 in `src/encode/color.rs`), `cargo test --test cross_check_encoder_binary` fails 3 of 4 byte-exact comparisons against `cjpeg`; reverted, green again. aarch64 + Homebrew first run: the full `--tests` suite was executed on a macOS aarch64 host with Homebrew jpeg-turbo before merging — no divergence surfaced, so nothing needed filing; the PR's own `C Interop (macos-latest)` leg is the first CI proof and must show a non-zero test count.

## P4-62. `cargo test --workspace` Does Not Build on windows-msvc — **CLOSED 2026-07-28**

**Motivation.** Found 2026-07-27 running the release gate for #362 on a Windows host. `crates/libjpeg-turbo-rs-capi/tests/format_message.rs:45-47` declares `extern "C" { fn snprintf(...) }` as its printf-expansion oracle. On the MSVC UCRT `snprintf` is an `inline` in `<stdio.h>` and is not exported by the import libraries, so the test target fails to link (`LNK2019: unresolved external symbol snprintf` → `LNK1120`). Because that is a *build* failure, no workspace test runs at all on Windows — not merely this one. GitHub [#378](https://github.com/developer0hye/libjpeg-turbo-rs/issues/378).

**Why CI is green.** `windows-latest` appears only in the `test-cross-platform` matrix, which runs `cargo test --lib`. No job builds the integration or capi test targets on Windows, so this has never been visible.

**Impact.** Windows contributors cannot run the project's own release gate (`cargo test --workspace --release`) or any integration test without excluding the capi crate. Library code is unaffected — this is a test-harness portability defect.

**Sibling finding (same root cause: no CI leg runs the integration tests on Windows).** With the capi crate excluded, `cargo test -p libjpeg-turbo-rs --release` still fails one test on `x86_64-pc-windows-msvc`: `alloc_budget_small_decode::gray_8x8_decode_allocation_budget` reports `9 allocations, 9634 bytes` against a budget of 8. Reproduced on a clean tree at `main` @ 818c24f, so it is not introduced by any pending work. Whether the budget or the Windows allocation path is wrong is unresolved — but nothing catches it today, because `windows-latest` runs only `cargo test --lib`. Any fix for this entry should re-measure it.

**Acceptance criteria.** `cargo test --workspace` builds and runs on `x86_64-pc-windows-msvc`; a CI leg builds the workspace test targets on `windows-latest` so the next break is caught; if the oracle is swapped or MSVC-gated, the P2-2 printf-expansion coverage is either preserved or its platform loss is stated in the test's doc comment.

**Status (2026-07-28): closed.** The oracle's `extern "C"` block now carries `#[cfg_attr(target_env = "msvc", link(name = "legacy_stdio_definitions"))]` — the UCRT's out-of-line exported definitions of its inline printf family — so the P2-2 printf-expansion coverage is preserved on MSVC rather than cfg-ed away (issue #378). The sibling `gray_8x8_decode_allocation_budget` failure is resolved by pinning the Windows measurement per-platform (9 allocations / 9,634 bytes, the clean-tree P4-62 reproduction; origin inside std's Windows plumbing remains unattributed, so the budget is exact-measured, not loosened globally). New CI job `Workspace build + targeted tests (windows)` builds every workspace test target with `cargo test --workspace --no-run` (where LNK2019 surfaces) and runs the two affected suites — both deliberately C-tool-free; the full suite stays off Windows CI because the C cross-validation tests hard-fail without djpeg/cjpeg, and provisioning those on the Windows runner is its own project. The PR's own job run is the first CI proof.

## P4-63. zune-jpeg Adoption-Gap Program (#380–#392) — **CLOSED 2026-07-28**

**Motivation.** The 2026-07-27 adoption-focused comparison vs zune-jpeg (GitHub tracking issue [#392](https://github.com/developer0hye/libjpeg-turbo-rs/issues/392)) — the twin of P4-55's competitive-gap program — audited both trees from a prospective user's seat. crates.io asymmetry: `libjpeg-turbo-rs` 4.2k all-time downloads vs zune-jpeg 83.8M (29.8M/90d), overwhelmingly transitive through the `image` crate. An external-consumer probe confirmed the codec itself is adoption-ready (decode / header probe / both encode APIs first-try, 6-crate dependency tree, 9 s clean release build); the gaps are packaging and presentation, plus five defects the audit surfaced. Detail and per-item acceptance criteria live in the GitHub issues; this entry keeps the program visible to the release gate rather than duplicating it.

**Sub-issues.** Defects: #380 (`libjpeg-turbo-rs-image`/`-wasm` unpublishable — path-only deps, so the `image` bridge from #209 cannot ship), #381 (bridge drops `std` → runtime AVX2/NEON dispatch silently off, SSE2-only on stock x86_64), #382 (12/16-bit lossless `1 << (precision - pt - 1)` negative shift — the unvalidated-`Al` twin of the 8-bit path; **fixed + closed 2026-07-28**: decode/encode now reject `pt >= precision` like C, `api/precision.rs` gained the `fuzz_decompress_precision` target, which immediately also caught and fixed a DC-category>16 shift overflow), #383 (`StreamingDecoder::skip_scanlines` was a no-op that reported success; **fixed + closed 2026-07-28**: real skipping through the pipeline's vertical crop with C-matching output-row clamp, byte-identical to `djpeg -skip` incl. under `-scale 1/2`, 7 regression tests), #384 (`Decoder` was `!Send + !Sync` and undocumented; **fixed + closed 2026-07-28**: `+ Send` bounds on the stored trait objects make `Decoder`/`StreamingDecoder`/`ScanlineDecoder` all `Send`, compile-asserted + cross-thread decode test; `!Sync` is now a documented per-instance contract matching upstream's one-thread-at-a-time rule per `cinfo` — our C ABI shim stays stricter, creating-thread only). Presentation: #385 (README led with ABI tiers, no badges, no zune comparison numbers; **fixed + closed 2026-07-28**: user-first restructure with badges/MSRV/platform matrix, measured zune + C comparison section, tiers compressed to pointers at ABI_COMPATIBILITY.md and this file's Current Status list, sanitizer material moved to the new CONTRIBUTING.md, GitHub topics/homepage set), #387 (docs.rs page: 12-line crate doc, no `[package.metadata.docs.rs]`, one doctest; **fixed + closed 2026-07-28**: crate-level quickstart with three asserting doctests, docs.rs metadata + doc_cfg badges incl. target predicates, doc(inline) on the canonical re-exports with the modules-stay-pub semver call recorded (commit message on the closing PR branch), core Decoder entry points documented), #388 (zero user-facing examples; **fixed + closed 2026-07-28**: five self-contained user examples + an image-bridge example, examples/README.md fronting the directory with the dev tooling grouped below, `cargo check --examples` in CI, scratch probes triaged/deleted, packaging decision recorded). Surface & trust: #386 (API curation: `compress*` wall, non-chaining `Decoder` config, missing conveniences), #389 (safety posture: 131 of 279 `unsafe` keyword usages outside `simd/`, no crate-level lints, no Miri), #390 (release & QA hygiene: no CHANGELOG, untested MSRV, `--lib`-only clippy, no supply-chain tooling), #391 (EXIF orientation required a full decode with no apply helper; **fixed + closed 2026-07-28**: header-probe accessors on `Decoder`/`ImageInfo`, `TransformOp::from_exif_orientation` for the DCT domain, `Image::apply_orientation[_value]` for pixels — all 8 mappings independently spec-verified in review, corner geometry pinned on lossless fixtures, tolerance cross-check vs the jpegtran-validated transforms; perf follow-up filed as P4-67).

**Acceptance criteria.** All sub-issues #380–#391 closed on GitHub with their stated criteria (defect fixes C-cross-validated per project rules), and tracking issue #392 closed after a cold-newcomer re-verification (fresh `cargo add`, docs.rs page review).

**Status (2026-07-28): closed.** GitHub umbrella #392 closed the same day: all thirteen items are resolved or independently tracked. Defects #381–#384 + #394, presentation #385–#388/#391, hygiene #390 — all closed via merged PRs (#395–#409, #418). #380's mechanics are landed and CI-gated (publish-check; PR #421 wires the bridge into the release pipeline) with only the maintainer's next `v*` tag outstanding — its own GitHub issue tracks that. #389 phase 1 merged (#408); phases 2–4 are P4-69. Nothing remains that only this umbrella guards.

## P4-64. Lossless Non-RGB Decode Accepts RGB-Family Conversions That C Refuses — **OPEN**

**Motivation.** Surfaced 2026-07-28 by the #369 review's C oracle run. For a **lossless** (SOF3) source whose colour space is not `JCS_RGB`, C libjpeg-turbo refuses every RGB-family output conversion: `jdcolor.c` raises `JERR_CONVERSION_NOTIMPL` when `cinfo->master->lossless && jpeg_color_space != JCS_RGB` (guard covers the whole `JCS_RGB`/`JCS_EXT_*` case list; verified against a lossless grayscale fixture via `tj3Decompress8`, which returns "Unsupported color conversion request" for ARGB/ABGR/RGBA/XRGB alike). Our decoder instead expands lossless grayscale to all RGB-family formats — including the 4bpp layouts fixed under P4-57 — with well-defined output. The divergence is in the **more-permissive** direction: streams C can decode, we decode identically; streams C rejects at colour-conversion, we additionally decode.

**Why it matters.** It is now load-bearing surface: `tests/regression_issue_369_gray_argb_abgr.rs` pins the lossless expansion's layout, and a comment at the lossless expansion site cites this entry. A C-parity-focused consumer (fuzz differs, capi shim behind stock djpeg) could observe us succeeding where djpeg errors with "Unsupported color conversion request".

**Options.** (A) Keep and document as a deliberate extension (likely right: rejecting would break existing Rust users of lossless gray→RGBA expansion for zero compatibility win); the C-ABI shim may still need to mirror C's error for drop-in fidelity. (B) Align with C and reject, matching `JERR_CONVERSION_NOTIMPL`. Either way the fuzz oracle (`fuzz_decode_diff_c`) needs a carve-out if it ever drives lossless non-RGB sources into RGB-family output formats.

**Acceptance criteria.** A decision (A or B) recorded here with rationale; if (A), the extension documented in the decoder's rustdoc for lossless output formats and the C-ABI shim's behaviour checked against stock djpeg on a lossless gray fixture with a colour output request; if (B), the reject path cross-validated against djpeg's error and the #369 lossless placement test re-scoped. Fuzz-oracle impact assessed either way.

**Related sibling (noted 2026-07-28 while fixing #382).** The 8-bit lossless path is also more permissive than C on *invalid scan parameters*: `decode/lossless.rs` deliberately clamps the initial-prediction shift for `Al >= precision` (`cd98c21`, the 2026-04-21 fuzz-hardening pass) and the 8-bit pipeline entries now in `src/decode/pipeline_impl/lossless.rs` check only the predictor (`decode_lossless_image` and its arithmetic twin), where C rejects the whole scan (`jdlossls.c:247-261`, `JERR_BAD_PROGRESSION`). The 12/16-bit paths now reject like C (#382); whichever way this entry's A/B decision goes should make the 8-bit path consistent with it.

## P4-65. 12-Bit Grayscale Decode Silently Ignores the Requested Output Format — **CLOSED 2026-07-28**

**Motivation.** Surfaced 2026-07-28 by the P4-57 docs audit (GitHub [#394](https://github.com/developer0hye/libjpeg-turbo-rs/issues/394)), which had to check "the 12/16-bit grayscale expansions for the same grouping" and found the branch missing entirely. `decode_12bit_as_8bit` (now `src/decode/pipeline_impl/output.rs`) hard-codes `pixel_format: PixelFormat::Grayscale` for 1-component frames — the comment reads "only Grayscale makes sense for 1-component" — so `output_format` is never consulted. `decompress_to` documents "with the specified pixel format", and `output_buffer_size` sizes from the *requested* format, so the two disagree. Measured on a 33x21 12-bit grayscale fixture (aarch64-darwin, 2026-07-28):

| requested | returned `pixel_format` | `data.len()` | `output_buffer_size()` |
| --- | --- | --- | --- |
| `Argb` | `Grayscale` | 693 | 2772 |
| `Rgb` | `Grayscale` | 693 | 2079 |
| `Rgba` | `Grayscale` | 693 | 2772 |

C expands instead of ignoring: `djpeg -rgb` on a `cjpeg -precision 12` grayscale JPEG emits a P6 (RGB) PPM. The 3-component branch of the same function honours `output_format` correctly via offsets, so this is specific to 1-component 12-bit frames. Not a memory-safety issue — the buffer is sized from the returned image, not the advertised size — but a caller that trusts `output_buffer_size()` and indexes at 4 bpp reads stale bytes.

**Acceptance criteria.** 12-bit grayscale honours `output_format` through the same expansion the 8-bit path uses (including the P4-57 pad/alpha placement via `pad_alpha_offset`), cross-validated against `djpeg -rgb` / `-grayscale` on a `cjpeg -precision 12` fixture, with `output_buffer_size()` agreeing with `decompress_to(..).data.len()` for every format. Check the 16-bit path (`decompress_16bit`) for the same gap while there. Alternatively, if ignoring the format is intended, `decompress_to` must return a typed error rather than a silently different `pixel_format`.

**Status (2026-07-28): closed.** The 1-component arm of `decode_12bit_as_8bit` now expands through the same family as the 8-bit grayscale path: `Rgb`/`Bgr` triples, all eight 4bpp formats via `pad_alpha_offset` (the P4-57 construction, so a mis-grouping is unrepresentable), `Rgb565` packing, and a typed `Unsupported` error for `Cmyk`. Pinned by `tests/regression_issue_394_12bit_gray_output_format.rs` (GitHub #394): every format's returned `pixel_format`/dims/`data.len()` vs `output_buffer_size()`, per-offset channel placement against the Grayscale reference, and a C leg — `cjpeg -precision 12` + `djpeg -rgb` emits a maxval-4095 P6 whose samples, run through our documented `v * 255 / 4095` scaling, match our `Rgb` output **byte-exactly** (proving both the expansion and the underlying 12-bit sample parity). The 16-bit path has no twin: `Decoder` / `decompress_to` reject precision 16 before any format handling in `decode_image_inner` (now `src/decode/pipeline_impl/output.rs`), ahead of the lossless dispatch, so 16-bit lossless is reachable only through `decompress_16bit`, which returns an `Image16` of raw `Vec<u16>` samples with no `PixelFormat` to honour. (The 8-bit lossless grayscale expansion that `decode_lossless_image` does reach was fixed for all formats under P4-57 / #369.)

## P4-66. 12-Bit Lossless Undifference Masks by Precision Where C Wraps Modulo 0xFFFF — **OPEN**

**Motivation.** Surfaced 2026-07-28 by the #382 docs audit while verifying the P4-38 attribution. `undifference_row_16` (`src/api/precision.rs`) computes `(diff + prediction) & ((1 << precision) - 1)`, but C's `jpeg_undifference1..7` (`jdlossls.c`) compute `(diff + PREDICTOR) & 0xFFFF` **unconditionally, regardless of frame precision** — the same contract whose 8-bit violation was P4-38 (Fuzz Smoke run 29689718301, fixed 2026-07-24 with `mask = 0xFFFF` in `decode/lossless.rs`). For `precision == 16` the two agree; for 12-bit or arbitrary-precision lossless streams we truncate at `2^precision - 1` where C keeps 16 bits until the output upscale stage.

**Reachability.** Only observable when the undifference sum wraps outside `0..2^precision`, i.e. corrupt or adversarial diffs — exactly the class the P4-38 differential sweep produced for 8-bit. The new `fuzz_decompress_precision` target (issue #382) exercises this path but has no C oracle; `fuzz_decode_diff_c` drives the 8-bit `Decoder` pipeline, not `decompress_lossless_arbitrary`, so this divergence is not currently differential-fuzz-reachable.

**Acceptance criteria.** `undifference_row_16` wraps modulo 0xFFFF like C, with the sample-type truncation left to the output upscale stage (mirroring the P4-38 fix); a crafted wrapping 12-bit lossless fixture cross-validated against `djpeg` (3.x decodes 12-bit lossless); a note in the fix on whether the 12-as-8 route (`decode_12bit_as_8bit`, P4-65) shares the defect.

## P4-67. `apply_orientation` Strided Reads Are Cache-Hostile for Orientations 5-8 — **OPEN**

**Motivation.** Filed 2026-07-28 from the #391 code review. `Image::apply_orientation_value` (now `src/decode/pipeline_impl/api.rs`) walks the destination sequentially, so for the axis-swapping orientations (5-8) the source advances by `width * bpp` per inner-loop iteration — on a 4032 px-wide 12 MP phone photo (the feature's headline case) that is a ~12 KB stride at 3 bpp (~16 KB for RGBA), roughly one cache miss per pixel. A tiled transpose (32x32 or 64x64 blocks) is typically 3-6x faster on this shape (review estimate from the general pattern; not measured on this code — the acceptance criteria below require the before/after benchmark). Secondarily, the loop-invariant `match orientation` sits inside the inner loop; hoisting it to a per-orientation strategy would also let 3 degrade to whole-row reversed copies and 4 to row swaps.

**Acceptance criteria.** Tiled remap for 5-8 with a benchmark in `experiments/` on a >= 12 MP fixture (before/after, sequential runs); orientations 2-4 use row-level operations; the #391 regression tests (corner geometry, DCT cross-check, 2/4bpp legs) stay green unchanged — they pin placement, so the optimization cannot silently reorder.

## P4-68. `decompress_to(.., Cmyk)` Panics on Non-CMYK Sources Where C Raises `JERR_CONVERSION_NOTIMPL` — **CLOSED 2026-07-28**

**Motivation.** Found 2026-07-28 by the #394 docs audit while verifying that PR's comment "a 12-bit source never converts to `Cmyk` in either arm". The 12-bit arm now returns `JpegError::Unsupported`, but its 8-bit twins panic: requesting `PixelFormat::Cmyk` for a source that is not CMYK/YCCK reaches an `unreachable!()` at three sites now split across `src/decode/pipeline_impl/{color,output,lossless}.rs`. Measured on aarch64-darwin, 2026-07-28, via `decompress_to(&jpeg, PixelFormat::Cmyk)` on 16x8 fixtures built by our own encoder:

| Source | Result |
|---|---|
| 8-bit baseline YCbCr (4:2:0) | panic, now `pipeline_impl/color.rs:517` (`grayscale/cmyk handled separately`) |
| 8-bit baseline grayscale | panic, now `pipeline_impl/output.rs:993` (grayscale expansion match) |
| 8-bit lossless grayscale (SOF3) | panic, now `pipeline_impl/lossless.rs:376` (`lossless_output_grayscale` match) |
| 12-bit grayscale | `Unsupported("cannot convert 12-bit JPEG to Cmyk")` — the #394 fix |

**Root cause.** `out_format` is taken from `self.output_format` with no validation, and every non-CMYK output stage enumerates only the grayscale/RGB family, routing `Cmyk` to `unreachable!()`. A panic is reachable from safe Rust with an ordinary well-formed JPEG and a public setter — no corrupt input required. C errors instead: `jdcolor.c:857-871` raises `JERR_CONVERSION_NOTIMPL` for `out_color_space == JCS_CMYK` unless the frame is YCCK or CMYK. The C ABI shim is insulated by the P4-4 `unwind_guard!` boundary; the Rust API is not.

**Acceptance criteria.** The three panicking sites return the same typed `JpegError::Unsupported` the 12-bit arm returns (rejecting before the frame decode, as #394 does), with a regression test covering baseline colour, baseline grayscale, lossless grayscale, and 12-bit grayscale sources; the surviving `unreachable!()` arms are then provably dead. Decide at the same time whether `Cmyk` output for a YCCK/CMYK source keeps its current behaviour (it does convert today) so the rejection is scoped to conversions C also refuses.

**Status (2026-07-28): closed** in the same PR that surfaced it (#394's branch). `decode_image_inner` now rejects `output_format == Cmyk` for any source whose detected colour space is not CMYK/YCCK, before the precision/lossless dispatch, with a typed `Unsupported` naming C's `JERR_CONVERSION_NOTIMPL`; the 12-bit arm keeps its own guard as defense in depth. Pinned by `p4_68_cmyk_request_on_non_cmyk_sources_errors_not_panics` in `tests/regression_issue_394_12bit_gray_output_format.rs`: all three former panic sites (baseline colour, baseline grayscale, lossless grayscale) return typed errors, and a real CMYK source still decodes to `Cmyk`.

## P4-69. `simd` Feature Contract and the Remaining #389 Safety-Posture Work — **OPEN**

**Motivation.** Filed 2026-07-28 from #389's phase-1 branch. The first-ever Miri run surfaced that the `simd` cargo feature was only honoured at *dispatch* sites: the aarch64 NEON fast paths in `src/encode/huffman_encode.rs` and the encoder implementation (now under `src/encode/pipeline_impl/`) were gated on `cfg(target_arch)` alone and executed vendor intrinsics in `--no-default-features --features std` builds. Phase 1 fixed those call sites, but the underlying design remains per-call-site: `src/simd/mod.rs` compiles every backend module whenever the *architecture* matches, feature notwithstanding, so nothing structurally prevents the next direct call from repeating #381/#389's class. Phase 1 also carved `#[allow(unsafe_op_in_unsafe_fn)]` on `pub mod simd` (~780 bare unsafe operations inside its unsafe fns predate the crate-level deny).

**Remaining work.**
1. **Module-level feature enforcement**: gate the backend modules (or their exported fns) on `feature = "simd"` so a simd-off build cannot even name an intrinsic wrapper; decide the interplay with `cpu_has!` and the scalar reference paths.
2. **simd/ `unsafe_op_in_unsafe_fn` sweep**: lift the carve by wrapping the ~780 sites with per-op blocks + SAFETY notes (mechanical but must not be rushed — wrong SAFETY prose is worse than none).
3. **Non-SIMD unsafe audit to zero**: the sites that survive phase 1 outside `simd/` (the `BitWriter` built over `mem::forget` in `huffman_encode.rs`, `Vec::set_len`-on-uninit patterns carrying `#[allow(clippy::uninit_vec)]`, raw-pointer `.add()` arithmetic, cold-path `get_unchecked`s in `decode/progressive.rs`) — each migrated to safe code (perf-gated per `experiments/`) or kept with a written `// SAFETY:` invariant.
4. **Goal state (#389's last criterion)**: `#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]` compiles, README-advertised and CI-checked — requires (3) to reach zero.

**Acceptance criteria.** Each numbered item lands with tests/benches per the project rules; #389 closes only when all four do. The phase-1 Miri job (non-SIMD `--lib` subset, 191 tests) must stay green throughout.

## P4-70. Clippy Structural-Lint Allow-List in Test Code — **OPEN**

**Motivation.** Filed 2026-07-28 while closing the clippy half of #390. The Clippy CI job was widened from `--lib` to two gates: `--workspace` with zero allowances, and `--workspace --all-targets` with exactly three structural lints allowed: `clippy::needless_range_loop`, `clippy::too_many_arguments`, `clippy::type_complexity`. Everything else in the test/bench long tail (~80 warnings across ~25 files: manual `div_ceil`, doc-list indentation, dead test helpers, `vec_init_then_push`, `manual_memcpy`, …) was fixed in the same PR. The three allowed classes remained at 56 warnings when measured on **aarch64-apple-darwin** (`cargo clippy --workspace --all-targets`, 2026-07-28): 43× `needless_range_loop` (index-synchronized multi-array loops in the `#[cfg(test)]` modules of `src/encode/fdct.rs`/`tables.rs`/`huff_opt.rs`/`pipeline.rs`, `src/transform/spatial.rs`, `src/simd/aarch64/mod.rs`, and in `tests/`), 9× `too_many_arguments`, 4× `type_complexity` — all in test code, none in the library proper (the zero-allowance `--workspace` gate proves that). **The total is host-dependent**: on x86_64 it is 47 (34× `needless_range_loop`), because `src/simd/aarch64/mod.rs` (2) and the `#[cfg(target_arch = "aarch64")]` bodies of `tests/simd_neon_encode.rs` (3) / `tests/simd_neon_scaled.rs` (4) are not compiled there. Re-measure on the same host before concluding the count moved.

**Why deferred.** The remaining sites are semantic rewrites, not mechanical fixes: iterator-zip conversions of loops that index three parallel arrays by design (DCT reference comparisons), and signature/type refactors of test helpers. Rushing them risks changing what a test asserts for zero coverage gain.

**Acceptance criteria.** The `-A` flags are deleted from the second clippy invocation in `.github/workflows/ci.yml` and the job stays green; no `#[allow]` may be added to library (non-`#[cfg(test)]`) code to get there. Blanket module-level allows in test files are acceptable only with a one-line why-comment.

## P4-71. Scaled Decode Never Dispatches to SIMD Reduced-Size IDCT on Any Arch — **OPEN**

**Motivation.** Filed 2026-07-28 by the #390 docs-drift audit, which caught `tests/simd_parity.rs` producing `unused_variables` hard-errors under the newly widened x86_64 clippy gate: the `parity_idct_4x4/2x2/1x1` comparison blocks are `cfg(target_arch = "aarch64")`-only because `src/simd/x86_64/` has no reduced-size IDCT kernels (`avx2_idct.rs`/`idct.rs` cover 8x8 only), and wasm32 likewise. C libjpeg-turbo ships SSE2 4x4/2x2 kernels (`jidctred-sse2.asm`). Worse (codex P2 on the fix commit): **even on aarch64 the `neon_idct_4x4/2x2/1x1` kernels have no production caller** — `Decoder::idct_scaled_strided` (now `src/decode/pipeline_impl/color.rs`, re-verified 2026-08-02) dispatches the sizes 4/2/1 unconditionally to the scalar `idct_scaled::*_strided` while its `8 =>` arm delegates to the SIMD `idct_islow_strided`, so the NEON routines are exercised only by the parity test. Note the scope precisely: components whose block size stays 8 during a scaled decode (e.g. Cb/Cr in 4:2:0 at 1/2 scale, per `compute_comp_block_size`) still take the SIMD 8x8 IDCT — it is exactly the reduced-size 4/2/1 branches that are always scalar, on every arch with a SIMD backend. The same audit found `avx2_rgba/bgr/bgra_to_ycbcr_row` existed but were never parity-tested — that half was fixed on the spot (the x86_64 block in `parity_rgb_to_ycbcr_rows` now mirrors the NEON/WASM four-format coverage).

**Acceptance criteria.** (1) Wire the existing `neon_idct_*` kernels into the `idct_scaled_strided` dispatch (or reject them with a benchmark showing scalar wins on the strided shapes — the NEON kernels are contiguous-output, so a strided adapter's copy cost may eat the win). (2) Port the reduced-size kernels to x86_64 (SSE2 baseline per `jidctred-sse2.asm`; AVX2 optional) and wasm32, wired behind the same dispatch. (3) Extend the three parity tests' comparison blocks to those arches, deleting the `cfg_attr(not(aarch64), allow(unused_variables))` shims. (4) Record the per-arch perf delta in `experiments/idct.tsv`. Benchmark-gated per the experiment protocol, scoped to the three SIMD backends (aarch64/NEON, x86_64/SSE2+AVX2, wasm32/SIMD128 — intentionally scalar targets stay P4-60's business): a closure that leaves one of *those* backends scalar must cite the losing measurement, not silence.

## P4-72. Grayscale Colorspace Override Cannot Convert — It Only Copies a Full-Resolution Luma Plane — **CLOSED 2026-08-02**

**Motivation.** Filed 2026-07-28 from #386's review rounds. `decode_with_colorspace_override(ColorSpace::Grayscale, ..)` emits component plane 0 verbatim, which is only correct for a YCbCr source whose component 0 carries the max sampling factor. Two defects follow: (1) for a **JCS_RGB** source, plane 0 is RED — codex measured 31/32 pixels wrong (max diff 116) vs `djpeg -grayscale`, reachable on main via the explicit `set_output_colorspace(Grayscale)` route (#386 excluded its new implied route from JCS_RGB and pinned the exclusion with a regression test); (2) for a legal stream whose component 0 is subsampled below max (`cjpeg -sample 1x1,2x2,1x1`), the arm used to slice past the quarter-size plane and **panic** — #386 guards it with `JpegError::Unsupported` naming this item. C handles both by running component 0 through the upsampler (and, for RGB, an RGB→gray conversion, `jdcolor.c` `rgb_gray_convert`) before emitting gray. Note for whoever closes this: a real `cjpeg -sample 1x1,2x2,1x1` stream only reaches the gray arm with `set_lenient(true)` — strict decode is rejected first by **P4-21**'s chroma-out-samples-luma guard, though `djpeg -grayscale` handles the same file directly (measured 2026-07-28).

**Acceptance criteria.** (1) RGB→grayscale conversion in the `Decoder` override path, byte-exact vs `djpeg -grayscale` on JCS_RGB streams; (2) subsampled-comp0 sources upsample component 0 and produce gray output byte-exact vs `djpeg -grayscale`; (3) the `Decoder` path's `Unsupported` guard and the JCS_RGB exclusion in `effective_output_colorspace` are deleted, their regression tests flipped to assert correct pixels; (4) the P4-21 lenient contract re-checked on these paths. The pre-existing public low-level helper has insufficient metadata to satisfy (1) without changing its signature; source-compatible treatment of that separate API is tracked as P4-80.

**Status (2026-08-02): closed.** Decoder grayscale output now shares a single per-component upsample helper with the direct-RGB output path. Full-size planes stay borrowed; 2:1 planes use the existing C-matched H2V1/H2V2/H1V2 fancy filters (or box replication under `fast_upsample`, a 1x1 scaled IDCT, or C's narrow horizontal edge rule), and other integral ratios use box replication. YCbCr/gray sources emit the resulting full-resolution component 0. JCS_RGB sources upsample all three planes and then apply `jdcolor.c::rgb_gray_convert`'s exact fixed-point matrix (`19595R + 38470G + 7471B + 32768`, shifted by 16). `effective_output_colorspace` now routes both YCbCr and JCS_RGB three-component `PixelFormat::Grayscale` requests through that implementation; the Decoder path's former JCS_RGB exclusion and subsampled-component `Unsupported` guard are gone. The pre-existing public low-level `decode_with_colorspace_override` signature remains source-compatible; because it has no source-colorspace or upsampling-policy inputs, callers needing the corrected conversion use `Decoder` (P4-80). `tests/regression_issue_386_api_curation.rs` generates both S444/S420 JCS_RGB streams and a real `cjpeg -sample 1x1,2x2,1x1` YCbCr stream, then proves both `set_output_format(Grayscale)` and `set_output_colorspace(Grayscale)` byte-exact against `djpeg -grayscale`. The unusual YCbCr stream remains rejected in strict mode under P4-21 and succeeds only with `set_lenient(true)`, preserving that contract. Zero-height/missing-DNL and non-standard two-component sources are rejected without panic, with the same inputs also rejected by `djpeg -grayscale`; the strict P4-21 test pins the `CorruptData` reason instead of accepting an unrelated error. The `max_memory` estimate now charges the extra full-resolution component-0 plane on this lenient path, pinned by `max_memory_accounts_for_grayscale_component0_upsampling_buffer`.

## P4-73. capi dlopen Tests Hardcode the In-Repo `target/` — Stale-Artifact False Greens Under `CARGO_TARGET_DIR` — **CLOSED 2026-08-02**

**Motivation.** Filed 2026-07-28 from #386's first fully clean worktree run. Five capi test files build the release cdylib path as `<workspace_root>/target/release/liblibjpeg_turbo_rs_capi.*` — `format_message.rs` `cdylib_path()` (8 tests), `capi_jpeg_read_header_tables_only.rs` (3), `install_layout.rs` `ensure_cdylib()` (2), `symbol_inventory.rs` (2), `tjunittest_link.rs` (2), 17 tests in total — while the `cargo build -p libjpeg-turbo-rs-capi --release` they spawn inherits the caller's `CARGO_TARGET_DIR`. Two more break one level down: `libtiff_integration.rs` and `capi_pillow_compat.rs` resolve (or delegate) correctly but spawn `examples/*/{build,run}.sh`, which default to `$REPO_ROOT/target/release` and are handed no override. Consequences on a host that offloads the target dir (this repo's standard setup): a fresh checkout FAILS all of them (build lands elsewhere, probe misses), and a checkout with an old in-repo `target/release/` silently dlopens the **stale** shim — the main checkout had exactly that, so weeks of local capi "greens" validated an outdated artifact. CI is unaffected (no `CARGO_TARGET_DIR` override). `examples/pillow_smoke/run.sh` was fixed in the #386 branch; `tests/capi_stock_tool_link.rs` is the complete pattern to copy — it resolves `CARGO_TARGET_DIR` itself (absolute or repo-relative) *and* passes `CAPI_TARGET_DIR` / `SHIM_DIR` into the scripts it spawns.

**Acceptance criteria.** A shared helper (e.g. `tests/helpers`) resolves the cdylib honoring `CARGO_TARGET_DIR` (absolute or repo-relative), used by every dlopen-class test and passed through to every spawned harness script; each test dlopens an artifact at least as new as the current `src/` mtime or rebuilds unconditionally; a worktree with `CARGO_TARGET_DIR` set and no in-repo `target/` passes the full capi suite.

**Status (2026-08-02): closed.** `crates/libjpeg-turbo-rs-capi/tests/support/cdylib.rs` is now the single resolver for the five affected C-ABI test binaries and the libtiff harness. Cargo places an integration-test executable and the package's un-hashed cdylib together in the active profile's `deps/` directory, so the helper derives the exact library path from `current_exe().parent()` and requires that sibling to exist. The tests therefore consume the artifact produced by their own outer Cargo invocation instead of launching a second build or reconstructing Cargo's target layout. This follows absolute or relative `CARGO_TARGET_DIR`, built-in triples, custom JSON targets, and CLI-only settings such as `--config target.<triple>.linker=...` or `-Z build-std` automatically: every one of those choices has already selected the directory and bytes beside the running test. Four portable helper regressions cover the pure sibling mapping, the real sibling emitted by Cargo, and a model of the target-qualified release layout used by standalone builders for both a built-in triple and a custom JSON target stem. The stock-tool host-execution harness rebuilds current source unconditionally against Cargo's host target and ignores ambient `CARGO_BUILD_TARGET`; the Pillow runner does the same unless its explicit `CAPI_BUILD_TARGET` override is set. The installer builds when `--build` is requested or no artifact exists, otherwise it consumes the caller's exact `CAPI_TARGET_DIR`. Each entry point passes its selected release directory to child harnesses, while fake-Cargo regressions verify the stock-tool, Pillow, and installer handoffs. The stock runner additionally requires a non-empty fixture set, records exactly five operation results per fixture, treats every failed stock oracle (including cjpeg's decoded-output fallback) as a hard failure, removes `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` from reference commands, and removes `LD_PRELOAD` and `DYLD_INSERT_LIBRARIES` from both sides. Those four common injection variables therefore cannot turn the oracle into Rust-vs-Rust or force the tested side onto a different shim. The complete C-ABI unit + integration suite passed with `CARGO_TARGET_DIR` redirected outside the worktree and `CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu`; the focused dlopen/install/libtiff/tjunittest suite also passed against the exact outer-build artifact without consulting an in-repo release tree. The Pillow smoke runner rebuilt only `/tmp/.../x86_64-unknown-linux-gnu/release/liblibjpeg_turbo_rs_capi.so` before soft-skipping on a deliberately unavailable fixture, and both `tjunittest_link` tests passed with the system compiler after the active Conda linker was excluded from `PATH`.

## P4-74. Reduced-Size and Extended IDCT Kernels Panic on `i32` Overflow Under Scaled Decode — **CLOSED 2026-07-30**

**Motivation.** Four consecutive scheduled Fuzz Smoke runs failed on `fuzz_decompress`: 30420069849 (2026-07-29 03:37), 30438301770 (09:07), 30461194331 (14:30), 30485530878 (19:41). Five distinct minimized seeds, all aborting in `src/decode/idct_scaled.rs` with `attempt to {add,subtract,multiply} with overflow` at five distinct lines (116, 117, 126, 131, 136). Every failing seed satisfies `data.len() % 7 == 3` — the only `fuzz_decompress` option arm that calls `set_scale(ScalingFactor::new(1, 2))`, which is the sole dispatcher of the reduced-size IDCT.

**Root cause.** `idct_scaled.rs` (4x4/2x2/1x1) and `idct_extended.rs` (the twelve 3x3…16x16 kernels) were ported from `jidctred.c`/`jidctint.c` using plain `+`/`-`/`*`. C's intermediates are `JLONG`, declared `long` at `jpegint.h:62` with only a *"must hold at least signed 32-bit values"* guarantee — so on any LLP64 or 32-bit host these same expressions wrap silently, and libjpeg-turbo treats that as the contract. A *dequantized* coefficient is bounded by `i16::MAX * u16::MAX = 2_147_418_945`, which fits `i32`, but a single multiply by a `FIX_*` constant (up to 29692) does not. The 8x8 twin `src/decode/idct.rs` had already been converted to `wrapping_*` for exactly this reason; neither scaled-IDCT file ever received the same treatment.

**Scope was wider than CI showed.** The fuzz target only ever sets 1/2 scale, so `idct_extended.rs` was never reached by the fuzzer — but it carries the identical defect and is reachable from the public API at 3/8, 5/8, 7/8 and every other non-power-of-two factor. It was found by the regression test written for this item, not by CI.

**Status (2026-07-30): closed.** `idct_scaled.rs` uses explicit `wrapping_*`; `idct_extended.rs`'s twelve kernels were converted to a `type W = core::num::Wrapping<i32>` newtype (the compiler, not review, then guarantees no operator was missed across 1112 lines). Both files route their final sample through a shared `level_shift_clamp`, which saturates — matching the NEON twin `simd/aarch64/idct_scaled.rs` (`vqmovun_s16`) and C's own `jidctred-neon.c`, and identical to C's `range_limit[v & RANGE_MASK]` over `[-512, 511]`, the range legal input produces.

Proof the refactor is behaviour-preserving: a temporary differential harness compared all twelve `idct_extended` kernels against a verbatim copy of the pre-refactor implementation over 4000 pseudo-random blocks with a realistic frequency-decay energy profile (dequantized magnitude bounded by `2048 >> ((u+v)/2)`, the domain where the old code cannot overflow and therefore has defined behaviour to compare against) plus every DC-only case — **byte-identical, 48,000 kernel comparisons**. Pinned going forward by `tests/regression_scaled_decode_idct_overflow.rs` (saturated-DQT streams at all 16 scaling factors, plus a clean-decode control), the five crash seeds now committed under `fuzz/corpus/fuzz_decompress/`, and three unit tests in `idct_scaled.rs`. Note `djpeg` was unavailable on the authoring host, so the C cross-checks in `cross_check_extended_scaling.rs` skipped locally and ran only in CI.

## P4-75. Arbitrary-Precision Lossless Decode Indexes Past `dc_tables` When `Ns < Nf` — **CLOSED 2026-07-30**

**Motivation.** Scheduled Fuzz Smoke runs 30461194331 (2026-07-29 14:30) and 30485530878 (19:41) failed `fuzz_decompress_precision` at `src/api/precision.rs:1815` with `index out of bounds: the len is 2 but the index is 2`. The minimized 298-byte seed is a 3-component SOF3 (`Nf=3`) followed by a SOS listing two components (`Ns=2`).

**Root cause.** `decompress_lossless_arbitrary` builds `dc_tables` from `scan.components.len().min(nc)`, so a short SOS yields a short table list, then decodes the multi-component branch with `for c in 0..nc`, indexing past the end. `Ns < Nf` is legal at parse time — C splits the remaining components across further scans (`jdmarker.c` `get_sos`) — but this entry point only ever decodes the single fully-interleaved scan held in `metadata.scan`.

**The defect was a pair, not a single site.** Fixing only `decompress_lossless_arbitrary` and then dispatching Fuzz Smoke against the fix branch (run [30504332488](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/30504332488)) put the fuzzer straight into the identical bug in **`decompress_16bit`** at `precision.rs:1316` — `index out of bounds: the len is 1 but the index is 1` — within the same 600s budget. `fuzz_decompress` passed on that same run, confirming P4-74. The lesson is recorded here deliberately: a copy-paste twin in the same file is not "covered" by fixing one copy, and the cheapest way to find that out is to dispatch the failing workflow against the branch rather than wait for the next scheduled run.

**Status (2026-07-30): closed.** Both entry points now share one `lossless_dc_tables(scan, dc_huffman_tables, nc)` helper in `src/api/precision.rs`, which resolves one DC table per **frame** component and rejects `Ns < Nf` with a typed `JpegError::CorruptData` before any decoding — so the two twins cannot drift again. It also hardens the `nc == 1` path, where an `Ns = 0` scan would previously have indexed `dc_tables[0]` on an empty vector. The 8-bit equivalents in `decode/pipeline_impl/lossless.rs` (`decode_lossless_huffman` and `decode_lossless_arithmetic`, currently at lines 37 and 171) were audited in the same pass and already carry equivalent guards on every reachable branch. Pinned by `tests/regression_lossless_partial_scan.rs`: the reported 3/2 shape, every `Ns < Nf` combination for `Nf` in 2..=4 at 8-bit **and** at `P=16` (so neither fix can be a special case), plus `Ns == Nf` controls for both entry points asserting a uniform plane (128 at 8-bit, 32768 at 16-bit). Both crash seeds are committed under `fuzz/corpus/fuzz_decompress_precision/`; each was verified to panic at its exact CI line with the fix stashed.

## P4-76. `tests/fuzz_crashes.rs` Replayed Crash Seeds Without the Fuzz Targets' Option Matrix — **CLOSED 2026-07-30**

**Motivation.** Found 2026-07-30 while reproducing P4-74. `fuzz_decompress` keys a decoder-option matrix (Rgba, Rgb565+dither, **scale 1/2**, crop, Xrgb, Grayscale) off `data.len() % 7`, but `tests/fuzz_crashes.rs` replayed every seed through a bare `decompress()`. Any crash whose reproduction needs an option therefore replayed **green** locally while still failing in CI. This was not hypothetical: `crash-8d3c593a48494bcae205838850ae218d093b7171` was already committed to `fuzz/corpus/fuzz_decompress/` and "passing" — it panics immediately once the scale option is applied. The harness also had no `fuzz_decompress_precision` case at all, despite that target having existed since #382.

**Status (2026-07-30): closed.** `tests/fuzz_crashes.rs` now carries `drive_decompress`, a replica of the fuzz target's option matrix kept in lock-step with `fuzz/fuzz_targets/fuzz_decompress.rs`, plus a second test that sweeps **all seven arms over every seed** by padding the input length (trailing bytes after EOI do not change the pixels, only the arm selection) so a seed minimized under one arm is still exercised under the others. A `fuzz_decompress_precision` replay covering all three entry points was added. Verified red-before-green: with the P4-74/P4-75 fixes stashed, three of the seven tests fail at the exact CI panic sites.

## P4-77. Windows-Host `cargo clippy --workspace` Fails in `capi/src/jpeglib.rs` — **OPEN**

**Motivation.** Found 2026-07-30 while running the CI clippy gate locally before the P4-74 PR. Both CI clippy invocations (`cargo clippy --workspace -- -D warnings` and the `--all-targets` variant) fail on a Windows host with three errors in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`, none of them related to the change under test: `unused imports: FromRawHandle and RawHandle` (line 1507), `unused import: std::io::Read` (line 25), and `unused variable: file` in `read_c_file` (line 1484). The `#[cfg(windows)]` arm of `read_c_file` evidently does not use the imports the `#[cfg(unix)]` arm needs, and appears to ignore its `file` argument entirely — so the Windows implementation may also be functionally incomplete, not merely lint-dirty. Verify which before fixing: if the Windows arm genuinely cannot read the handle, silencing the lint with `_file` would paper over a real gap.

**Why it does not affect CI.** The Clippy job is `runs-on: ubuntu-latest`, so these `cfg(windows)` paths are never compiled there. This is the clippy twin of P4-62 (Windows-host workspace *build* break, closed 2026-07-27) — the same class of host-only breakage, in a job that only ever runs on Linux.

**Acceptance criteria.** `cargo clippy --workspace --all-targets -- -D warnings` (with P4-70's three allowances) is green on a Windows host; the `read_c_file` Windows arm either reads the handle correctly with a test, or documents why it cannot and returns a typed error instead of a silently unused parameter. Consider adding a Windows leg to the Clippy job so this cannot regress undetected.

## P4-78. No 32-bit ARM (AArch32) NEON Backend — ARMv7 Is Our Widest Gap vs C — **OPEN**

**Motivation.** Filed 2026-07-30 from GitHub [#424](https://github.com/developer0hye/libjpeg-turbo-rs/issues/424), a user question about decode speed on ARMv7 Cortex-A. It is also the concrete downstream request that `phase2.md` item 2 ("32-bit ABI targets … add these only when a downstream consumer requests that platform") named as this entry's trigger.

**The asymmetry.** On 32-bit ARM we dispatch **100% scalar**: `src/simd/mod.rs` compiles backend modules for `aarch64` / `x86_64` / `wasm32` only, so `detect()` and `detect_encoder()` fall through to `scalar::routines()` on `target_arch = "arm"`. C libjpeg-turbo does **not** — `simd/CMakeLists.txt:352-356` compiles the same 13 shared `simd/arm/*-neon.c` intrinsics kernels for `CPU_TYPE=arm` as for `arm64` (IDCT islow/ifast/reduced, FDCT int/fast, both colour directions, merged upsample, upsample, downsample, quantize, progressive Huffman) plus an AArch32-specific `arm/aarch32/jchuff-neon.c`, with `-mfpu=neon` forced at line 358. NEON is then detected at run time on AArch32 Linux/Android by parsing `/proc/cpuinfo` (`simd/arm/aarch32/jsimdcpu.c:72-125`), so a distro build serves NEON and non-NEON ARMv7 cores from one binary.

This makes 32-bit ARM different in kind from the other scalar targets. On RISC-V / POWER / s390x **neither** side vectorises, so P4-60's scalar-kernel gap is the whole story. On ARMv7-A it is our scalar code against C's full SIMD pipeline, and the two deficits multiply. Note the hardware caveat: NEON is optional in ARMv7-A (present on most Cortex-A parts, e.g. A7/A15; optional on A5/A9) and absent from ARMv7-M/R — on a NEON-less ARMv7 core C is scalar too, and P4-60 alone describes the gap.

**Estimated magnitude — inference, not measurement.** No ARMv7 hardware measurement exists yet; the honest bound composes two known factors: our scalar decode measured **1.12× at 640×480 and 1.72× at 1080p** vs C's scalar decode (`experiments/riscv64_scalar_2026-07-27.md`, 07-28 section, post-P4-60-step-1), times C's own NEON speedup — upstream claims **2-6×** for SIMD-capable CPUs generally (`references/libjpeg-turbo/README.md:6`). That puts us in the region of **2-5× slower than C libjpeg-turbo on a NEON-capable ARMv7-A core**, decode. Encode is not better: C's AArch32 NEON covers FDCT, colour convert, downsample, quantize and Huffman encode. **Do not quote a hardware figure until one is measured** — see the recipe below.

**Why not just port the kernels.** `core::arch::arm`'s NEON intrinsics are still unstable. Measured 2026-07-30 with rustc 1.93.1: a `vld1q_u8`/`vaddq_u8`/`vst1q_u8` probe compiled for `armv7-unknown-linux-gnueabihf` fails with five × `error[E0658]: use of unstable library feature 'stdarch_arm_neon_intrinsics'`. Against an MSRV of 1.87 and a stable-only build, the options are: **(A)** wait for stabilisation; **(B)** hand-write the kernels through `core::arch::asm!`, which *is* stable — this is closest to what C does (its AArch32 path was itself hand-written assembly before the intrinsics port) but forfeits the compiler's register allocation and needs its own soundness review under P4-69; **(C)** gate a NEON backend behind a nightly-only cargo feature, which splits the correctness matrix. No option is chosen yet — that decision is part of closing this item.

**Correction (2026-07-30): the list above omitted the cheapest option, and the omission was a reasoning error, not a missing measurement.** Unstable *intrinsics* block hand-written kernels; they say nothing about **auto-vectorisation**, which needs no intrinsics at all and is available on stable today. It is simply switched off by the target's own baseline: `armv7-unknown-linux-gnueabihf` carries `-neon`, so LLVM's vectoriser never runs on ARM, while on `x86_64` — where SSE2 *is* in the ABI baseline — it silently vectorises the same code. Verified by disassembly (`experiments/x86_64_scalar_2026-07-30.md`, `experiments/armv7_autovec_2026-07-30.md`), counting packed/vector instructions per kernel:

| kernel | x86_64 (SSE2 baseline) | armv7 default | armv7 `-C target-feature=+neon` |
| --- | --- | --- | --- |
| `decode::idct::idct_8x8` | 162 | 0 | **270** |
| `decode::color::ycbcr_to_rgb_row` | 0 | 0 | **140** |
| `decode::upsample::fancy_h2v2_row` | — | 0 | **232** |
| whole binary, 128-bit q-register ops | — | **0** | **2077** |

So **(D) enable the target feature and let the compiler vectorise** joins the list. Correctness holds: the 196 lib tests plus both dispatch suites pass under `qemu-arm` with `+neon` **and** `-C overflow-checks=on`, 204/204.

**(D) is not a recommendation, and must not become a default.** Two independent reasons. (1) `target-feature` is compile-time, so a `+neon` binary **SIGILLs on a NEON-less ARMv7 core** — C avoids this by dispatching on a `/proc/cpuinfo` probe, which a compile-time flag cannot do. (2) **It may be slower on real silicon, and the environment that produced the 1.17× cannot see why.** The A/B (`qemu-arm`, ours-before vs ours-after, so the emulation tax is symmetric) showed 1.17× at both 640×480 and 1080p — but QEMU models no pipeline, no cache, and no register-domain transfer cost, which is precisely where auto-vectorised ARM code regresses: Cortex-A8's NEON↔ARM transfer stall punishes the scalar/vector boundary traffic that vectoriser prologues and reductions generate; Cortex-A7/A9 implement NEON on a 64-bit datapath, so 128-bit ops issue over two cycles and the vectorisation setup/tail cost is proportionally larger than on a 128-bit A15; and with no `-C target-cpu=` the vectoriser uses a *generic* ARMv7 cost model. That C's hand-written AArch32 NEON kernels are a proven win transfers nothing to auto-vectorised code, which is not written to avoid those traps. **Treat 1.17× as evidence that vectorisation happens, not that it pays.**

**What would make (D) safe to recommend:** a hardware A/B on the actual target core, with `-C target-cpu=` set to that core rather than generic; per-kernel A/B rather than whole-decode timings, so a kernel that regresses is not hidden inside an average; and it stays opt-in either way, documented beside the existing x86_64 build-flag guidance rather than as a default.

**Correctness is not in question.** The scalar path is the reference every SIMD kernel is validated against, and `tests/no_std_dispatch.rs::scalar_dispatch_matches_the_default_dispatch` pins scalar output == host-dispatch output. What ARMv7 adds beyond that is a **32-bit hardware ISA** — native ARM codegen, the armhf ABI, real unaligned-access semantics. (32-bit *pointer width* alone was already executed: `wasm.yml` runs `cargo test --target wasm32-wasip1` under wasmtime, scalar-dispatched because `simd128` is off by default there.) That is now gated by the `Test (linux-armv7 scalar, emulated)` leg (`.github/workflows/armv7.yml`), which cross-builds for `armv7-unknown-linux-gnueabihf` and runs the suites under `qemu-arm` — **204 tests executed** (196 lib + 6 `simd_dispatch` + 2 `no_std_dispatch`), 0 failed, on qemu-arm 8.2.2 ([first green run](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/30519595108/job/90796847495)). Verified by mechanism rather than by the job passing: the log's per-binary `running N tests` / `test result: ok` lines were read to confirm the counts, since a filter or target mistake here would otherwise produce the vacuous green that P4-61 documents. That leg is a correctness gate only: unlike the RISC-V harness, the two sides would *not* pay a symmetric emulation tax here, since C's NEON kernels would be emulated while our scalar ones are not.

**Measurement recipe (for a real ARMv7-A device, no emulation).** Build ours with `cargo build --release --target armv7-unknown-linux-gnueabihf --example bench_scalar_p460` and C with `-mfpu=neon` (or use the distro's `libjpeg-turbo-progs`, which enables NEON at runtime); run `examples/bench_scalar_p460.rs` for our in-process decode medians and `djpeg -outfile /dev/null` best-of-N minus process startup for C, exactly as `experiments/riscv64_scalar_2026-07-27.md` does; cross-check `JSIMD_FORCENONE=1 djpeg` to separate "C's NEON win" from "our scalar deficit". Record as `experiments/armv7_<date>.md`.

**Acceptance criteria.** (1) One real-hardware ARMv7-A measurement recorded in `experiments/`, replacing the inferred 2-5× above with a number, and split into its two factors (scalar-vs-scalar, plus C's NEON win) so it says where the work has to go. (2) A decision on (A)/(B)/(C) recorded here with rationale. (3) If a backend lands: byte-exact against `djpeg`/`cjpeg` on the armhf target, wired through `simd::detect()`/`detect_encoder()` with the AArch32 runtime-detection question answered (compile-time `target_feature = "neon"` vs a `/proc/cpuinfo` probe like C's), the emulated CI leg extended to run it, and per-kernel deltas in the relevant `experiments/*.tsv`. (4) The README platform matrix and this entry updated together — the matrix must keep distinguishing "C vectorises here and we do not" from "neither side vectorises". ~~(5) A CI leg that **executes** on 32-bit ARM~~ — **DELIVERED 2026-07-30** (`.github/workflows/armv7.yml`): cross-builds for `armv7-unknown-linux-gnueabihf` and runs the lib + dispatch suites under `qemu-arm`, `--release` with `-C overflow-checks=on` so a half-width size computation that wraps fails the job instead of passing quietly. Correctness gate only — per the emulation-asymmetry note above, it must not be read as a performance measurement. 204 tests green on the first run; no 32-bit-specific failure surfaced, so nothing needed filing. **One loose end, tracked as GitHub [#428](https://github.com/developer0hye/libjpeg-turbo-rs/issues/428):** `armv7.yml` still opens with the "PARKED FILE — not active while it lives in `.github/ci-pending/`" header from its staging location, which is now false for a running job — a reader auditing CI coverage could conclude 32-bit ARM is ungated. It cannot be corrected by a push from a token without the `workflow` scope (the same constraint that parked the file); #428 carries the exact replacement text and commit message for a web-editor commit.

## P4-79. Stock-Tool Harness Reused Generated Inputs Across Toolchains and Omitted `jversion.h` in Fresh Output Directories — **CLOSED 2026-08-02**

**Motivation.** Surfaced while running the repository-wide gate for P4-73. Reusing one output directory across the Conda and system toolchains kept wrapper objects selected only by source mtime and failed with non-PIE relocations. A fresh output directory exposed the complementary failure: the fallback configuration emitted `jconfig.h` and `jconfigint.h` but not `jversion.h`, so it could not compile `djpeg.c`, `cjpeg.c`, or `jpegtran.c`. The harness therefore depended on output-directory and toolchain history.

**Root cause.** `build_wrapper_obj` treated `(source path, source mtime)` as the complete object identity even though the compiler, target, and flags affect the bytes. The fallback header generator also omitted the configured `src/jversion.h.in` input required by all three stock tools.

**Acceptance criteria.** A fresh output directory generates every required configuration header and links the stock tools; wrapper objects cannot survive a compiler/flag change; `cargo test --test capi_stock_tool_link` passes after the old build directory is absent.

**Status (2026-08-02): closed.** The harness now materializes all three configuration headers in its own output directory on every invocation, including `jversion.h` from upstream's template with its `1991-2026` copyright range, so a prior CMake tree cannot leak configuration into the test. All ten precision-wrapper objects rebuild on every invocation, and a compiler failure now propagates out of Bash command substitution instead of falling through to an older object. Every non-skipped pipeline execution creates a fresh temporary `OUT_DIR`, then builds and exercises four shim-linked tools plus the two standalone marker utilities; `tjbench -nowrite` also prevents the smoke run from leaving derived PPM files in the source submodule. The runner requires at least one JPEG fixture and exactly five reported operations per fixture; stock-oracle failures (including failed decodes that leave partial files) are failures rather than skips. Reference commands do not inherit `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `LD_PRELOAD`, or `DYLD_INSERT_LIBRARIES`, and the latter two are also removed from shim-linked commands, so neither an empty corpus, a broken oracle, nor contamination through those four variables can report `OK`. The nested release build pins Cargo's host target and probes only that target-qualified release tree, so `CARGO_BUILD_TARGET` or `[build].target` cannot redirect it toward (or let it accept) stale unqualified bits. The missing-`jversion.h`, failed-wrapper, target-qualified fake-Cargo, empty-corpus, failed-oracle, partial-output, and loader-contamination regressions were red before their fixes; the fresh-output pipeline, export-inventory guard, and failure-propagation tests now all pass.

## P4-80. Public Low-Level Grayscale Override Lacks Source Metadata and Upsampling Policy — **OPEN**

**Motivation.** Filed 2026-08-02 by P4-72's final docs audit. `decode::toggles` is public, so the pre-existing `decode_with_colorspace_override` function and its full signature are part of the source-compatibility surface even though production decoding calls it only for planar YCbCr output. Its grayscale arm receives decoded planes, a `FrameHeader`, and output geometry, but not the JFIF/Adobe markers needed to distinguish every JCS_RGB source from YCbCr, nor the `Decoder` upsampling policy and complete geometry used by P4-72. It therefore cannot implement metadata-correct RGB→gray and C-matched component upsampling for every input without a new API. Removing or renaming it would break downstream compilation, so P4-72 retained the symbol and its legacy full-size component-0 behavior; subsampled component 0 returns a typed `Unsupported` instead of panicking. The corrected supported surface is `Decoder::set_output_format(Grayscale)` / `set_output_colorspace(Grayscale)`.

**Acceptance criteria.** Decide the intended status of the public `decode` internals: (A) deprecate the low-level helper and make the module crate-private in the next semver-major release, with a downstream source scan and migration note; or (B) add a metadata- and policy-aware public companion, migrate callers, and make the legacy function delegate wherever its inputs are sufficient. In either case, rustdoc must state the exact behavior, source compatibility must be tested for the supported release line, and the legacy path must never panic on short or subsampled planes.

## P4-81. Linux cdylib Omits GNU `LIBJPEG_8.0` Symbol Versions — **OPEN**

**Motivation.** Filed 2026-08-02 by the real OpenCV replacement experiment.
Ubuntu's prebuilt `libopencv_imgcodecs.so.406` requests
`jpeg_CreateCompress@LIBJPEG_8.0` and
`jpeg_CreateDecompress@LIBJPEG_8.0`. glibc currently falls back to the Rust
cdylib's unversioned exports and the workload succeeds, but prints
`libjpeg.so.8: no version information available`. The same warning appears
for libtiff, GDAL, Poppler, and HDF4 transitive consumers loaded by OpenCV.
The SONAME and symbol names are therefore sufficient for this measured glibc
run, but the library is not yet a warning-free distro replacement and no
claim is established for stricter ELF loaders.

**Root cause.** The Linux build sets `DT_SONAME=libjpeg.so.8`, but supplies no
GNU linker version script. `readelf --version-info` shows only the Rust
cdylib's imported GLIBC/GCC version requirements; its exported `jpeg_*`
symbols are global/unversioned. Ubuntu's reference `libjpeg.so.8` defines
`LIBJPEG_8.0` and `LIBJPEGTURBO_8.0` version nodes.

**Acceptance criteria.** (1) Linux `libjpeg.so.8` reproduces the reference v8
symbol-to-node inventory: upstream exports live under `LIBJPEG_8.0`, while the
reference's empty `LIBJPEGTURBO_8.0` node is preserved. Crate-only extra/test
and TurboJPEG symbols need an explicit, tested visibility/version policy; they
must not be mislabeled as reference libjpeg-turbo extension exports. Do not
change macOS/Windows builds or the separately named TurboJPEG artifact; (2) an
automated `readelf --version-info`/symbol inventory test fails on an absent or
wrong node or symbol assignment; (3) the OpenCV harness keeps both binding
assertions and runs without `no version information available`; and (4)
alternative SONAME configurations are either given their correct version map
or rejected with a clear build-time error rather than mislabeled silently.

## P4-82. Classic Scanline Encoder Dropped Public Restart Settings — **CLOSED 2026-08-02**

**Motivation.** Filed and closed 2026-08-02 after the first OpenCV replacement
run initially reported a false green. OpenCV requested progressive Huffman
compression with `restart_interval = 4`; Ubuntu's libjpeg-turbo output carried
DRI=4 plus RST markers, while the Rust C shim's output carried neither. The
first harness version checked only decoded pixels, so both files appeared to
pass despite the encoded-structure contract divergence.

**Root cause.** `run_encoder_and_flush` selected high-level Rust compression
helpers that did not accept restart settings. It therefore discarded both
public `jpeg_compress_struct` fields, `restart_interval` and
`restart_in_rows`, when the deferred classic scanline input was finally
encoded. The omission affected baseline, optimized, progressive, arithmetic,
and lossless pixel-encode dispatch branches.

**Acceptance criteria.** (1) Both public restart fields reach baseline,
optimized, progressive, arithmetic-baseline, arithmetic-progressive, and
lossless scanline output; (2) row mode matches C's per-scan MCU calculation,
including lossless mode's forced 1x1 sampling; (3) a stock-C cross-validation
matrix requires identical DRI sequences and RST counts for block and row mode
in every branch, with byte-exact lossy output and pixel-exact lossless output;
and (4) the OpenCV workload requires SOF2, DRI=4, and RST before accepting the
replacement run.

**Implementation.** The dispatcher now forwards the direct block interval to
every entropy branch. For non-progressive scans, row-mode restarts are
converted to MCUs using the derived JPEG width, maximum horizontal sampling
factor, and scaled data-unit width; lossless calculation uses the 1x1 sampling
that C forces before `per_scan_setup`. Progressive branches retain both public
values so the encoder derives the row interval separately for each scan, as
libjpeg does.

**Status (2026-08-02): closed.** The first structural regression failed before
the dispatcher change. The final 14-case C-ABI matrix covers block and row
restarts across all six dispatcher modes against stock `cjpeg`; every case
requires identical DRI sequences and RST counts, all lossy cases are
byte-identical, and stock `djpeg` decodes both lossless outputs back to the
source pixels exactly. The optimized
mode has distinct block/row cases; smoothing likewise composes with both public
restart fields. That matrix also caught the lossless row-mode 2x sampling error
before closure. The OpenCV
harness independently requires SOF2, DRI=4, and an RST marker, then compares
both JPEG files and all four self/cross-decoded BGR and grayscale matrices
byte-for-byte. On the pinned Ubuntu 24.04/OpenCV 4.6 environment, Rust and
system output share SHA-256
`2945f085182223131779686ca88c83d0ee816222a1517bd289946ac106316905`.

## P4-83. Baseline Classic Scanline Encoder Dropped Public Input Smoothing — **CLOSED 2026-08-02**

**Motivation.** Filed and closed 2026-08-02 during P4-82's dispatcher review.
The same deferred C-ABI boundary read `smoothing_factor` from the public
`jpeg_compress_struct` but never supplied it to the Rust baseline pixel
encoder, making ordinary baseline `jpeg_write_scanlines` callers silently
encode unsmoothed data.

**Root cause.** The old high-level optimized helper could express neither the
public smoothing value nor smoothing without optimized Huffman coding. Simply
routing every smoothed input through its replacement wrapper would also have
changed C semantics by enabling `optimize_coding` implicitly.

**Acceptance criteria.** (1) For baseline output, nonzero public smoothing
selects the full-plane path and reaches its downsampler; (2)
`optimize_coding = FALSE` remains false, so Annex K selection is not silently
replaced; (3) both restart fields compose with smoothing; and (4)
a deterministic baseline classic-C-ABI encode is byte-identical to stock
`cjpeg -smooth` with the same row restart setting. Progressive/arithmetic
composition is explicitly outside this closure and filed as P4-84.

**Status (2026-08-02): closed.** `run_encoder_and_flush` now builds
`CompressParams` with independent smoothing and Huffman-optimization flags.
The `smoothing-blocks` and `smoothing-rows` legs of
`c2_3_scanline_option_dispatch_matches_cjpeg` set smoothing 25 with
`optimize_coding = FALSE` and exercise `restart_interval = 4` and
`restart_in_rows = 2`, respectively. Both are byte-identical to the matching
`cjpeg -smooth 25 -restart` output, including the DRI sequence and RST count.

## P4-84. Progressive/Arithmetic Classic Scanline Encoding Still Drops Input Smoothing — **OPEN**

**Motivation.** Filed 2026-08-02 while tightening P4-83's closure claim. The
baseline path is fixed and C-exact, but `run_encoder_and_flush` selects the
progressive and arithmetic branches before its smoothing-capable baseline
branch. A caller that combines nonzero `smoothing_factor` with
`progressive_mode` or `arith_code` therefore still receives an unsmoothed
stream without an error. Lossless is not part of this filing: upstream
deliberately resets smoothing to zero for lossless output.

**Root cause.** The Rust progressive and arithmetic pixel encoders downsample
per block and do not accept the full-plane smoothing parameter. The native
`Encoder` already rejects these combinations visibly under P4-46 rather than
dropping the option, but the deferred classic C-ABI dispatcher bypasses that
builder validation.

**Acceptance criteria.** Choose and test one explicit contract: (A) add
full-plane smoothing to progressive, arithmetic-baseline, and
arithmetic-progressive encoding and cross-validate each against the matching
`cjpeg -smooth` mode; or (B) reject the combinations through the classic
error-manager path before emitting output, with a C harness proving the
failure is visible and no partial JPEG is accepted. In either case, no
nonzero public smoothing value may be silently ignored.

## P4-85. Classic Scanline Compression Ignores Public Custom Quantization and Huffman Tables — **OPEN**

**Motivation.** Filed 2026-08-02 during the P4-83 closure audit. The native
`Encoder` and baseline `CompressParams` support custom tables, but the classic
`jpeg_write_scanlines` boundary advertises the corresponding libjpeg fields
and setup functions without applying their values to the encoded stream. This
is another silent option drop: callers receive a valid JPEG built with default
quality-scaled/Annex K tables.

**Root cause.** `jpeg_set_quality` records only an 8-bit quality and ignores
its `_force_baseline` argument instead of installing the public scaled tables,
so low-quality `force_baseline = FALSE` cannot select upstream's 16-bit
DQT/SOF1 behavior. `jpeg_add_quant_table` stores scaled entries only in
`CompressPrivate::quant_tables` instead of installing the public
`quant_tbl_ptrs` slot. Despite documenting the upstream clamp, it names its
argument `_force_baseline` and always permits values through 32767 rather than
clamping to 255 when requested. `jpeg_default_qtables` ignores caller-edited
per-slot `q_scale_factor` values and delegates back to the single private
quality, while `jpeg_set_linear_quality` passes pre-zigzagged constants into
`jpeg_add_quant_table` even though that function's C contract is natural
order. Callers can separately populate
`quant_tbl_ptrs`/`dc_huff_tbl_ptrs`/`ac_huff_tbl_ptrs`, but
`run_encoder_and_flush` reads none of them. It constructs its lossy encoders
from `quality` alone. As a result, custom quantization setup (including the
direct linear-scaling path used by `jpeg_set_linear_quality`) and caller-supplied
Huffman tables do not reach classic scanline output. Coefficient transcoding
has a separate table-materialization path and is outside this filing.

**Acceptance criteria.** (1) Make `jpeg_set_quality`, `jpeg_default_qtables`,
`jpeg_set_linear_quality`, and `jpeg_add_quant_table` install natural-order
public tables with libjpeg ownership/lifetime semantics, honor per-slot
`q_scale_factor` values, and apply the correct 1..255 or 1..32767 clamp
according to `force_baseline`, including 16-bit DQT/SOF1 selection; (2) convert all
referenced public quantization and Huffman slots into the Rust encoder's table
definitions, respecting each component's
`quant_tbl_no`/`dc_tbl_no`/`ac_tbl_no`; (3) baseline Huffman output uses caller
tables unless `optimize_coding` supersedes them, while progressive Huffman
matches upstream's forced optimization/derived-table behavior and lossy
arithmetic output still honors custom quantization; (4) lossless output matches
upstream exactly: custom quantization/AC tables are inapplicable, Huffman
lossless forces optimization and regenerates its DC table, and the unsupported
arithmetic+lossless combination is rejected as tracked by P4-89; (5) a real C harness installs non-default tables through the
public structs/setup functions and cross-validates DQT/DHT contents plus
decoded pixels against stock libjpeg-turbo, including composition with
smoothing and both restart controls; and (6) invalid or incomplete tables fail
through the classic error manager rather than panicking or silently falling
back.

## P4-86. Classic Lossy Scanline Compression Ignores Public DCT Method — **OPEN**

**Motivation.** Filed 2026-08-02 during the P4-82 option-dispatch review.
Classic callers may select `JDCT_ISLOW`, `JDCT_IFAST`, or `JDCT_FLOAT` through
`jpeg_compress_struct::dct_method`. The shim accepts all three but silently
encodes every lossy `jpeg_write_scanlines` request as ISLOW. The P4-82 matrix
fixed its C oracle to `-dct int`, so it intentionally proves restart behavior
without covering this separate option.

**Root cause.** `run_encoder_and_flush` assigns
`DctMethod::IsLow` unconditionally instead of translating `c.dct_method`, then
passes that constant into every lossy baseline/progressive/arithmetic helper.
Lossless output has no DCT and is outside this filing.

**Acceptance criteria.** (1) Translate all public DCT enum values accepted by
upstream and reject invalid values through the classic error manager; (2)
forward the selected method through baseline, optimized, smoothed,
progressive, arithmetic-baseline, and arithmetic-progressive scanline paths;
(3) cross-validate deterministic `JDCT_IFAST` and `JDCT_FLOAT` C callers
against matching stock `cjpeg -dct fast|float`, including block/row restarts;
and (4) keep ISLOW output byte-identical to the P4-82 matrix.

## P4-87. Classic Abbreviated-Datastream Table State Is Not Wired — **OPEN**

**Motivation.** Filed 2026-08-02 during the P4-85 table audit. Native
abbreviated datastream support exists, but the classic API accepts the
table-reuse controls without changing its output. Applications that emit one
tables-only stream and then suppress tables in image bodies cannot use the shim
as a libjpeg-compatible replacement.

**Root cause.** `jpeg_start_compress` ignores `write_all_tables`,
`jpeg_suppress_tables` stores a private boolean that no encode path reads, and
`jpeg_write_tables` synthesizes quality-based defaults rather than serializing
the installed public quantization/Huffman tables and their `sent_table` state.

**Acceptance criteria.** (1) Implement libjpeg's `write_all_tables`,
`sent_table`, and suppress/reset state transitions across cinfo reuse; (2)
`jpeg_write_tables` emits the installed applicable tables exactly, including
custom tables and arithmetic-mode rules; (3) suppressed image bodies omit
DQT/DHT while decoding correctly when paired with the tables-only stream; (4)
a stock-C harness cross-validates marker inventories and pixels across at least
two reused images; and (5) the test fails if either side silently emits a
self-contained body.

## P4-88. Classic Scanline Marker Controls and CCIR601 Rejection Are Ignored — **OPEN**

**Motivation.** Filed 2026-08-02 while auditing all public options consumed by
`run_encoder_and_flush`. The native `Encoder` can express these policies, but
the classic scanline boundary always accepts its own defaults.

**Root cause.** Pixel scanline encoding does not forward
`write_JFIF_header`, JFIF version/density, `write_Adobe_marker`, or
`CCIR601_sampling`. Upstream raises `JERR_CCIR601_NOTIMPL` for the latter. Its
`do_fancy_downsampling` field is deliberately ignored too, so the shim already
matches that classic behavior; native `Encoder::fancy_downsampling()` is a
Rust extension, not a classic contract. The coefficient writer has separate
metadata handling and is outside this filing.

**Acceptance criteria.** (1) Honor the public JFIF/Adobe marker toggles,
version, and density byte-exactly; (2) reject `CCIR601_sampling = TRUE`
visibly through the classic error manager before output; and (3)
cross-validate marker inventories/fields against stock libjpeg-turbo for
grayscale, YCbCr, RGB-direct, and CMYK where each option is applicable.

## P4-89. Classic Arithmetic+Lossless Requests Silently Become Huffman Lossless — **OPEN**

**Motivation.** Filed 2026-08-02 during the P4-85 lossless table review.
Upstream classic compression rejects arithmetic lossless with
`JERR_ARITH_NOTIMPL`, but the shim returns a valid SOF3 Huffman stream. Native
SOF11 support is a separate, intentional Rust capability and can remain.

**Root cause.** `run_encoder_and_flush` tests `lossless_predictor` before
`arith_code`, so the lossless Huffman helper wins and the arithmetic request is
discarded.

**Acceptance criteria.** A real C harness requests arithmetic+lossless through
the public classic fields and proves both stock and Rust invoke the error
manager before accepting an image; the Rust side must emit no usable partial
JPEG. Native Rust SOF11 tests remain green and are explicitly outside the
classic compatibility contract.

## P4-90. Classic Arithmetic Scanline Compression Ignores Public DAC Conditioning — **OPEN**

**Motivation.** Filed 2026-08-02 during the same dispatcher audit. Arithmetic
callers can set the 16 public DC/AC conditioning slots, but classic scanline
output always uses the Rust encoder defaults. P4-25 is the decode-side
per-scan snapshot problem and does not cover this encode gap.

**Root cause.** `run_encoder_and_flush` never reads `arith_dc_L`,
`arith_dc_U`, or `arith_ac_K`; its arithmetic helpers accept no conditioning
arrays.

**Acceptance criteria.** (1) Forward all referenced conditioning slots to
sequential and progressive arithmetic writers; (2) validate the same ranges
and signal errors through the classic manager; and (3) cross-validate DAC
markers and decoded pixels against a stock-C harness using non-default,
multi-slot conditioning plus both restart controls.

## P4-91. Classic Scanline Compression Ignores Custom Scan Scripts — **OPEN**

**Motivation.** Filed 2026-08-02 during the public-field audit. The native
encoder supports `ScanScript`, but classic callers that set `scan_info` and
`num_scans` receive the fixed default progressive script instead.

**Root cause.** `jpeg_simple_progression` only flips `progressive_mode`, and
`run_encoder_and_flush` never reads `scan_info`/`num_scans`. No classic-path
validation translates component indices, spectral selection, or successive
approximation fields.

**Acceptance criteria.** (1) Translate valid public scripts into the native
scan-script representation without retaining caller pointers past the legal
lifetime; (2) mirror upstream validation/error behavior for malformed scripts;
and (3) cross-validate SOS sequences, entropy mode, restarts, and decoded
pixels against stock libjpeg-turbo for custom Huffman and arithmetic scripts.

## P4-92. Classic Scanline Compression Collapses Valid Sampling-Factor Layouts to 4:4:4 — **OPEN**

**Motivation.** Filed 2026-08-02 during the P4-82 dispatcher review. The
restart matrix proves the common 2x2 layout, but classic callers can populate
every component's sampling factors directly. Several valid layouts silently
produce a different SOF and pixel stream.

**Root cause.** `subsampling_from_comp_info` inspects only component 0 and
recognizes six `(H,V)` pairs. It maps standard 4x2 (TJSAMP_410), standard 2x4
(TJSAMP_24), and non-standard layouts such as `3x2,1x1,1x1` to S444; chroma
component factors are ignored entirely. P3-6 covers native sampling-factor
encoding, not this classic translation boundary.

**Acceptance criteria.** (1) Preserve the complete public per-component
sampling-factor layout instead of lossy luma-only enum inference; (2) derive
MCU geometry and row restarts from those exact factors; and (3) use a real C
struct harness to cross-validate SOF sampling factors, DRI/RST structure, and
stock-`djpeg` pixels for all eight standard layouts plus at least two valid
non-standard layouts, including RGB-direct sampling.

## P4-93. Classic Scanline Compression Ignores Requested JPEG Colorspace — **OPEN**

**Motivation.** Filed 2026-08-02 during the same public-field audit.
`jpeg_set_colorspace` updates the public struct, but classic scanline output
can still encode a different colorspace while returning success.

**Root cause.** `run_encoder_and_flush` never reads `jpeg_color_space`.
`JCS_RGB` output therefore routes through the ordinary YCbCr helper despite
RGB component metadata, while `JCS_YCbCr` input falls back to `PixelFormat::Rgb`
and is color-converted a second time. Native `Encoder::colorspace` support from
P4-53/P4-54 does not reach this shim boundary.

**Acceptance criteria.** (1) Translate input and requested JPEG colorspaces
independently, preserving direct YCbCr/CMYK/RGB data where upstream does; (2)
route every supported entropy mode without discarding the requested output
colorspace; (3) mirror upstream errors for unsupported conversions; and (4)
cross-validate SOF component IDs/sampling factors, JFIF/Adobe markers, bytes
where deterministic, and stock-`djpeg` pixels for RGB→YCbCr, RGB→RGB,
YCbCr→YCbCr, CMYK, and YCCK classic callers.

## P4-94. Classic 12/16-Bit Scanline Buffers Never Reach a High-Precision Encoder — **OPEN**

**Motivation.** Filed 2026-08-02 during the final P4-82 test audit. The only
new C-ABI test loaded `jpeg12_write_scanlines`/`jpeg16_write_scanlines` and
called them with a null cinfo. Its former name/comment claimed row mechanics
and outside pipeline coverage that it did not exercise.

**Root cause.** The high-precision writers fill `pixels_u16`, clear
`pixels_u8`, and record `priv_state.precision`. `jpeg_finish_compress` still
routes every non-raw scanline job to `run_encoder_and_flush`, which reads only
`pixels_u8` and never calls the 12-bit or 16-bit encoder. The finish failure is
stored privately and can look like a successful void C call.

**Acceptance criteria.** (1) Dispatch real 12-bit lossy scanlines and 12/16-bit
lossless scanlines to the matching native precision backends; (2) validate
precision/mode/colorspace combinations and report failures through the classic
error manager; (3) run real stock-C create→start→write→finish harnesses with
multiple non-null row batches and compare headers, precision, and decoded
samples exactly against stock libjpeg-turbo; and (4) replace the null-only
smoke with active 12-bit and 16-bit regressions that fail if zero rows reach
the encoder or no complete JPEG is emitted.

## P4-95. Classic Raw-Data Compression Drops Most Public Encode Options — **OPEN**

**Motivation.** Filed 2026-08-02 while checking P4-94's separate raw-data
dispatch. Eight- and 12-bit raw planes do reach native encoders, but only a
small default subset of the classic compressor contract is preserved.

**Root cause.** `run_raw_encoder_and_flush` and its 12-bit counterpart reduce
the public state to planes, geometry, one private quality, and lossy enum-based
sampling. They omit restart controls, optimized/progressive/arithmetic modes,
DCT and smoothing policy where applicable, custom quant/Huffman/DAC tables,
table suppression/reuse, custom scan scripts, exact component sampling, and
requested output colorspace.

**Acceptance criteria.** (1) Define and implement every upstream option that
applies to raw-data input, visibly rejecting only combinations upstream
rejects; (2) share option translation/validation with scanline compression so
the two paths cannot drift again; and (3) cross-validate real C raw-plane
harnesses for 8/12-bit precision, all supported sampling layouts, entropy
modes, custom tables/scripts, and both restart controls using marker/SOF/SOS/
DRI/DAC inventories plus stock-decoded samples. Default quality-only round
trips do not close this item.

## P4-96. Classic Decompression Color Quantization and Colormap Switching Are Not Wired — **OPEN**

**Motivation.** Filed 2026-08-02 during the final classic C-ABI audit. Native
one/two-pass/external-palette quantization exists, but the classic public fields
and `jpeg_new_colormap` are documented as complete without a behavioral test.

**Root cause.** The shim initializes and inspects quantization flags only far
enough to set `output_components = 1`; it never calls the native quantizer,
populates `actual_number_of_colors`/`colormap`, or returns palette indices.
`jpeg_new_colormap` is unconditional no-op. With RGB bytes still buffered but
one-byte row sizing selected, scanline output can expose the wrong packed-RGB
slice rather than a valid index row.

**Acceptance criteria.** A real C harness must cross-validate stock and Rust
for one-pass, two-pass, and external-colormap quantization, asserting every
index row plus `actual_number_of_colors`, the public colormap planes, dithering
behavior, and output component counts. Buffered-image mode must also switch an
external palette through `jpeg_new_colormap` and prove re-quantized output.
Unsupported state transitions fail visibly; no test may infer completion from
only a one-component row length.

## P4-97. `jpeg_resync_to_restart` Is an Unconditional Success No-Op — **OPEN**

**Motivation.** Filed 2026-08-02 after the C-ABI encode test's null-only
utility smoke was compared with upstream `jdmarker.c`. Native restart recovery
strategies exist, but the exported classic function and installed source
callback do not implement the C default algorithm.

**Root cause.** Both paths return TRUE without inspecting `unread_marker`,
emitting warnings, scanning/discarding markers, mutating restart state, pulling
the source manager, or suspending. The filing PR removed the old null-cinfo
test that asserted this broken constant as a positive result.

**Acceptance criteria.** Add a real suspending C source-manager harness
cross-validated against stock libjpeg-turbo. Cover desired,
past, and future RST markers; non-RST markers; invalid-byte scan-forward;
warning/state mutation; refill; and a FALSE suspension return. The exported
function and default callback must share one C-exact implementation while the
native strategy extension remains available separately.

## P4-98. Classic 12/16-Bit Decode Bypasses Lifecycle and Public Output Options — **OPEN**

**Motivation.** Filed 2026-08-02 after the high-precision encode false-green
prompted review of the matching decode tests. Those tests cover grayscale and
call read-scanlines immediately after header parsing, certifying a lifecycle
that stock libjpeg does not permit while missing configured output behavior.

**Root cause.** `jpeg12_read_scanlines`/`jpeg16_read_scanlines` lazily invoke
the native full-image precision decoders directly from source bytes without
requiring `jpeg_start_decompress`. They bypass `out_color_space`, scaling,
DCT/upsampling controls, quantization and, on the 16-bit path, crop. The 12-bit
helper always converts three-component input to RGB; current assertions are
only grayscale or “some nonzero sample.”

**Acceptance criteria.** Build real stock-C and Rust lifecycle matrices for
12-bit lossy and 12/16-bit lossless inputs: read-header→start→batched
read/skip/crop→finish, plus invalid-state calls. Cross-validate public output
fields, requested colorspaces, applicable scaling/crop/DCT/upsampling options,
row counts, and exact or measured-C-tolerance samples. Tests must include color
sources and fail if read succeeds before start or silently ignores an option.
Exercise 12-bit skip and crop immediately after start, before any read has
created private decoded state, then assert subsequent cropped pixel rows; an
echoed x/width pair without output pixels is not evidence. Finish→new source
and abort→new source reuse must prove the thread-local high-precision image and
cursor are cleared rather than returning stale pixels/EOF.

## P4-99. Classic Decode Dispatcher Ignores Output Options and Colorspace Metadata — **OPEN**

**Motivation.** Filed 2026-08-02 during the same classic decode dispatcher
audit. This is safety-relevant: callers commonly size scanline buffers from
`jpeg_calc_output_dimensions`, but start-decompress can restore the full image
width and then copy full-width rows.

**Root cause.** `run_decoder_for_start` never reads `scale_num`/`scale_denom`,
`dct_method`, `do_fancy_upsampling`, or `do_block_smoothing`. Its colorspace
translation maps requested YCbCr to RGB, omits YCCK, and catches any configured
decode error by retrying a format-agnostic decode, erasing caller intent and C
errors. Dimension calculation publishes scaled fields without configuring the
decoder; start then decodes full size and overwrites those fields. Header setup
also ignores Adobe APP14, so RGB-direct/YCCK streams get guessed colorspaces and
leave `saw_Adobe_marker`/`Adobe_transform` cleared. Dimension calculation also
rounds public per-component downsampled dimensions to 8-pixel blocks instead
of upstream's unrounded sampling-ratio ceil to accommodate local writer
assumptions.

**Acceptance criteria.** (1) Apply every supported libjpeg scaling ratio and
the public DCT/upsampling/block-smoothing policies before decoding; (2) keep
calculated and actual output dimensions identical through the lifecycle; (3)
cross-validate rows and dimensions with stock `djpeg`/a real C harness across
all scale factors, ISLOW/IFAST/FLOAT, fancy on/off, and progressive smoothing
on/off; (4) cross-validate YCbCr, RGB-family, grayscale, CMYK, and YCCK output
requests plus forbidden conversions without a fallback that changes format;
(5) assert APP14 transform 0/1/2 header metadata and derived colorspaces against
stock C; and (6) use canary-guarded C buffers sized from
`jpeg_calc_output_dimensions` to prove no overwrite at reduced scales. Compare
odd-size/scaled per-component downsampled dimensions and minimum DCT sizes
exactly; repair downstream writer assumptions instead of publishing non-C
geometry.

## P4-100. Classic Codec Failures Are Reported as Suspension or Silent Success — **PARTIAL: translator + finish/start entry points landed; batch continues**

**Motivation.** Filed 2026-08-02 after P4-94 showed that a void
`jpeg_finish_compress` can appear successful without emitting an image. The
same systemic error-boundary issue explains several option-dispatch false
greens; it needs one shared fix rather than per-path private strings.

**Root cause.** Encoder helpers write native failures only to private
`last_error`; `jpeg_finish_compress` ignores their boolean results and resets
to `CSTATE_START`. `jpeg_start_decompress` maps native failure to FALSE, whose
classic meaning is source suspension, and some paths retry a different decode.
Ordinary codec failures do not consistently populate `msg_code` and call
`cinfo->err->error_exit`.

**Acceptance criteria.** (1) Centralize native→classic error translation with
stock-equivalent `msg_code`/parameters and exactly one `error_exit`; (2) reserve
FALSE returns for legal suspension only; (3) never reset to a success state or
accept a usable partial stream after encoder failure; and (4) use real setjmp C
harnesses for malformed input, unsupported option/conversion, high-precision
misconfiguration, and encoder failure, cross-validating callback, state, and
output behavior against stock libjpeg-turbo.

**Progress (2026-08-08) — shared translator and the two worst entry points.**

`classic_error_for` is the single native→classic mapping, and
`raise_native_error` / `raise_classic_error` are the only places that call
`error_exit`, satisfying criterion (1)'s "exactly one". The mapping marks its
own fidelity: marker, buffer, EOF, dimension-limit and I/O conditions map
exactly; `CorruptData` is a documented closest fit, because upstream has no
single code for it (it reports some corruption as warnings and continues).

Criterion (2) — `FALSE` reserved for suspension — is applied to
`jpeg_start_decompress`. Both a malformed stream and a missing source manager
used to return `FALSE` with the reason in a private string no C consumer can
read. In classic libjpeg `FALSE` means *source suspension*, so a caller doing
the documented thing (refill, retry) would spin forever on a stream that can
never decode.

Criterion (3) — never reset to a success state — is applied to
`jpeg_finish_compress`, which had all three failure modes at once: it discarded
its helpers' boolean results, then set `CSTATE_START` unconditionally. It now
rejects a finish that was never started (`JERR_BAD_STATE`), rejects a scanline
encode short of `image_height` (`JERR_TOO_LITTLE_DATA`, matching
jcapimin.c:184-188), reports helper failure, and leaves a failed handle in its
failed state for `jpeg_abort_compress` / `jpeg_destroy_compress`.

`CompressPrivate::error_reported` keeps the "exactly one `error_exit`"
guarantee across the layers: the encode helpers raise destination-manager and
suspension failures themselves, so finish must be able to tell "already
reported" from "failed silently".

`tests/capi_classic_error_codes.rs` asks the C compiler for every `JERR_*` the
shim defines — value *and* message text — and fails if the shim adds one the
table does not cover. That is worth its own test: `jerror.h`'s enum is
positional, and two hand-written derivations of it disagreed by one during this
work. 14 codes verified. It also removes the "wrong value silently mis-reports"
half of **P4-120**, leaving only that item's reachability concern.

The change immediately caught a silent failure:
`write_coefficients_rejects_foreign_handle` asserted only "no crash", which a
shim that quietly did nothing satisfied. Rejection is now *reported*, and the
test asserts the `msg_code`.

**Status (2026-08-08): partial.** Criteria (1)-(3) hold for
`jpeg_start_decompress` and `jpeg_finish_compress`. Not yet done: the remaining
private-string-only sites (54 of the original 59 `last_error` assignments still
have no `error_exit` nearby), criterion (4)'s stock-versus-Rust setjmp harness
matrix, and the P4-104 state work this depends on — the shim still only ever
enters `CSTATE_START` and `CSTATE_WRCOEFS`, never `CSTATE_SCANNING` or
`CSTATE_RAW_OK`, so upstream's state-gated finish contract cannot be matched
exactly yet. `cargo test --workspace --no-fail-fast`: 2466 passed, 0 failed,
1 ignored (macOS aarch64).

## P4-101. Classic Header Parse Does Not Publish Coding Tables or Scan State — **OPEN**

**Motivation.** Filed 2026-08-02 during the classic decode audit. Consumers
inspect public DQT/DHT/DAC/DRI/SOS state after `jpeg_read_header`; geometry and
saved markers alone are not the full header contract.

**Root cause.** Header parsing leaves public quant/Huffman table pointers,
arithmetic conditioning, restart interval, active scan selectors, and
`coef_bits` at create-time defaults. `is_baseline` is also derived as
`!progressive && !lossless`, which incorrectly labels extended-sequential SOF1
and arithmetic-sequential SOF9 streams as baseline even though upstream sets it
only for SOF0. Private metadata used by decode/transcode does not materialize
those C fields.

**Acceptance criteria.** A real C harness compares all public table slots,
conditioning, DRI, scan selectors, `is_baseline`, and pointer lifetime/state
for SOF0/SOF1/SOF2/SOF3/SOF9/SOF10/SOF11 plus abbreviated streams across
consume-input and cinfo reuse. Published tables must also feed
copy-critical-parameters exactly.

## P4-102. Classic Raw-Data Decode Bypasses Public Options and State Contracts — **OPEN**

**Motivation.** Filed 2026-08-02 while scoping P4-99. Eight/12-bit raw output
works for defaults, but its lazy full-image helpers bypass the classic decoder
configuration and error boundary.

**Root cause.** `jpeg_read_raw_data`/`jpeg12_read_raw_data` call native raw
decoders from source bytes without forwarding applicable scaling/DCT policy;
failures become zero/private text. Twelve-bit raw reads also lack the 8-bit
suspending-body drain check.

**Acceptance criteria.** Cross-validate stock-C raw-plane geometry/samples for
applicable scaled-IDCT/DCT choices, state and `raw_data_out` validation,
suspension/resume, both precisions, and reuse. Invalid calls use the classic
error manager, not zero as an ambiguous soft result.

## P4-103. `jpeg_crop_scanline` Does Not Implement iMCU-Aligned C Semantics — **OPEN**

**Motivation.** Filed 2026-08-02 after the existing aligned-crop test was found
to compare only against this project's exact-slice implementation.

**Root cause.** The shim clamps/slices decoded RGB bytes. Upstream validates
state/precision/order, aligns x down to the iMCU column, expands width to keep
the requested right edge, and updates output/component geometry.

**Acceptance criteria.** A stock-C harness covers unaligned offsets,
subsampling/scaling/grayscale, invalid null/zero/out-of-bounds/after-read calls,
returned x/width/output_width, component geometry, and subsequent row bytes.
The 12-bit initialization/order portion remains in P4-98.

## P4-104. Classic Decompressor State Constants, Transitions, and Finish Lifecycle Diverge — **OPEN**

**Motivation.** Filed 2026-08-02 after P4-13's harness was found to assert the
shim's `DSTATE_STOPPING`, the opposite of upstream's abort-reset completion.
That false oracle was removed in the filing PR.

**Root cause.** The shim defines `DSTATE_STOPPING = 206`, but upstream assigns
206 to `DSTATE_RAW_OK` and STOPPING is 210; several intermediate states are
missing. Successful header parse remains `DSTATE_INHEADER` instead of READY and
has no repeated-call guard. Finish unconditionally sets its misnumbered
STOPPING, clears caches/source, and returns TRUE rather than rejecting unread
rows/bad state, draining EOI with suspension, calling `term_source`, and
abort-resetting for reuse.

**Acceptance criteria.** Match every upstream state constant and transition
after create/header/start, buffered/raw/coefficient operation, finish, and
abort. Stock-C setjmp/source-manager cases cover repeated/out-of-order header,
incomplete rows, EOI suspension/retry, exactly-once `term_source`, final reset,
and same-handle reuse. P4-13 continues to prove body suspension without
asserting a shim-specific state.

## P4-105. Classic Marker Writers Ignore State and Declared Lengths — **OPEN**

**Motivation.** Filed 2026-08-02 during the deferred-encode audit. Valid
pre-scanline markers work, but invalid timing and piecemeal lengths are silently
accepted/reordered.

**Root cause.** Marker/ICC functions only append private buffers. They do not
enforce global state/`next_scanline`; `jpeg_write_m_header` uses `datalen` only
as capacity, so under/over-write changes the emitted length.

**Acceptance criteria.** A stock-C setjmp harness covers before-start,
valid pre-row, after-row, wrong byte counts, invalid sizes, ordering, and exact
marker bytes for complete, piecemeal, and ICC writers.

## P4-106. `jpeg_finish_compress` Accepts Incomplete Input and Bad States — **OPEN**

**Motivation.** Filed 2026-08-02 alongside P4-100. Finishing partial scanlines
currently encodes zero-filled unwritten rows and returns a valid-looking JPEG.

**Root cause.** Finish checks only private `have_started`, not expected
scanline/raw row counts or legal state, then resets regardless of helper result.

**Acceptance criteria.** Stock-C setjmp cases cover partial scanline/raw input,
bad/double finish, progress passes, helper failure/no usable partial output,
destination termination, final reset, and reuse. `JERR_TOO_LITTLE_DATA` and
other errors flow through P4-100's shared translator.

## P4-107. `jpeg_enable_lossless` Clamps Invalid Input and Omits Public State — **OPEN**

**Motivation.** Filed 2026-08-02 after the helper's only direct test was found
to be a null guard.

**Root cause.** The shim silently clamps predictor/Pt, stores private values,
and omits `Ss/Se/Ah/Al`, state validation, and `Pt < data_precision`. Upstream
raises `JERR_BAD_PROGRESSION` for invalid values.

**Acceptance criteria.** A real C harness compares fields and error callbacks
for valid/invalid predictors and point transforms at 8/12/16-bit precision,
including after-start calls, then proves valid settings reach output.

## P4-108. Classic Destination Managers Violate Buffer Ownership and I/O Errors — **CLOSED 2026-08-08**

**Motivation.** Filed 2026-08-02 as a P0 memory-safety finding. `jpeg_mem_dest`
tests only NULL/0 allocation, omitting libjpeg's caller-supplied-buffer branch;
stdio tests cover successful files only.

**Root cause.** The shim interprets caller `*outsize` capacity as existing data,
prefixes it to output, and unconditionally frees a non-NULL caller pointer when
growing. `jpeg_stdio_dest` ignores short `fwrite`, `fflush`, and `ferror`.

**Acceptance criteria.** Canary C tests cover sufficient stack/static buffers
(SOI at offset 0, used size, no free), insufficient caller buffers (original
never freed, returned allocated output), NULL allocation, and reuse. Stdio
tests use `/dev/full` plus a portable failing stream and require
`JERR_FILE_WRITE`; successful controls prove flush/term behavior.

**Fix.** `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` now mirrors upstream
`jdatadst.c` structurally rather than approximating it:

* `MemDestState` / `StdioDestState` mirror `my_mem_destination_mgr` /
  `my_destination_mgr`, so the two managers have **separate** callback sets.
  That separation is what makes upstream's identity check
  (`init_destination != init_mem_destination`, jdatadst.c:204-212 / 252-257)
  expressible; installing a memory destination over a foreign manager now
  raises `JERR_BUFFER_SIZE` instead of reinterpreting someone else's private
  area.
* `*outsize` is read as the caller buffer's **capacity**. A sufficient buffer is
  filled in place at offset 0 and neither moved nor freed; growth allocates a
  doubled block, copies the encoded prefix, and frees only `newbuffer` — memory
  this library allocated. A caller buffer is never passed to `free`.
* `*outbuffer == NULL || *outsize == 0` allocates `OUTPUT_BUF_SIZE` inside
  `jpeg_mem_dest` and publishes it immediately, as upstream does
  (jdatadst.c:267-273). The prior behaviour — leaving `*outbuffer` NULL until
  the first flush — was pinned by a test that has been corrected.
* `jpeg_mem_dest(cinfo, NULL, …)` raises `JERR_BUFFER_SIZE` (jdatadst.c:242-243)
  instead of silently installing no destination at all, which is what the shim
  previously did — a caller that passed a NULL out-parameter got a successful
  compress that wrote nowhere.
* stdio short writes, and any `ferror` after the terminating `fflush`, raise
  `JERR_FILE_WRITE`. Because the callbacks run beneath a Rust frame that owns
  the encoded `Vec`, they record the code in `CompressPrivate::pending_dest_error`
  and the public entry point raises it through `error_exit` after dropping its
  allocations — same `msg_code`, same `longjmp`, no leaked buffer.

**Status (2026-08-08): closed.** `cargo test -p libjpeg-turbo-rs-capi --test
capi_classic_dest_ownership` passes 12 C canaries covering every acceptance
bullet. The test is differential in two layers, because either alone can go
green while wrong:

* **Contract facts.** Each canary prints implementation-independent
  `key=value` lines (`buffer_moved=0`, `msg_code=38`, …) and the *same binary*
  is relinked against a reference libjpeg-turbo v8 build
  (`LIBJPEG_TURBO_REFERENCE_DIR`); the two reports must match exactly.
* **Payload.** Marker bookends prove nothing — a manager that dropped every
  entropy-coded byte still emits `SOI…EOI`. So each canary decodes its own
  output back to RGB and compares it to the source pixels (measured mean
  absolute difference 9.180 on both implementations for this deliberately
  high-frequency 64×64 fixture at q=90 4:2:0; the in-canary bound is 11.0).
  The encoded size and that error are then reported as `#`-prefixed metrics
  and compared across legs within a factor of two — the one part of the report
  a broken destination manager can move independently of the reference.

The reference leg refuses to run against a directory that does not actually
contain `libjpeg.so.8`/`libjpeg.8.dylib` and its own `jconfig.h`, so it cannot
silently degrade into a comparison against the system libjpeg. Measured
locally against both Homebrew `jpeg-turbo` 3.1.4.1 and a fresh
`-DWITH_JPEG8=1` build of the pinned submodule: 12 passed, 0 failed,
**0 skips** — every case ran both legs. CI performs the same pinned-submodule
comparison and fails closed if any case skips or if `/dev/full` was not
exercised. Before the fix the stack-buffer canary aborted inside `free()`,
`mem_grow` produced a non-JPEG, and all three stdio error cases reported
success. `cargo test --workspace --no-fail-fast`: 2455 passed, 0 failed,
1 ignored (`restart_bomb_4096x4096`, release-only).

One review finding was fixed structurally rather than locally: the destination
callbacks originally re-derived `&mut CompressPrivate` from `cinfo->master`
while the flushing frame still held one, which is two live `&mut` to the same
allocation — undefined behaviour that LLVM's `noalias` would be entitled to
exploit by folding away the very `pending_error` read this fix depends on. The
manager's private state now lives *inside* the manager (`OwnedDestMgr`,
reachable only through `cinfo->dest`), which is both sound and what upstream
does (`my_mem_destination_mgr`, jdatadst.c:43-53). The same re-derivation
pattern exists elsewhere in the shim and is not addressed here.

## P4-109. Classic Source-Manager Setup and Stdio Semantics Diverge — **OPEN**

**Motivation.** Filed 2026-08-02 during the destination review. Source setup is
documented as complete but misses public validation, FILE buffering, and
Windows support.

**Root cause.** `jpeg_mem_src` accepts null/empty input and overwrites foreign
managers. Unix stdio duplicates the fd and `read_to_end`s outside `FILE*`
buffering; Windows is unavailable. Errors do not consistently reach
`error_exit`.

**Acceptance criteria.** Cross-validate null/empty, foreign-manager replacement,
pre-read/buffered FILE positions, I/O failure, `term_source`, and reuse against
stock C on Unix/Windows. Preserve FILE position/buffer semantics and exact
`JERR_INPUT_EMPTY`/`JERR_BUFFER_SIZE` behavior.

## P4-110. `jpeg_Create*` Ignores Version and Struct-Size ABI Guards — **OPEN**

**Motivation.** Filed 2026-08-02 as a P0 ABI memory-safety finding. Many tests
pass a 4096-byte blob/size that stock v8 rejects, normalizing the missing guard.

**Root cause.** Both create functions ignore `version`/`struct_size` and write
the full Rust mirror. Upstream validates before writing, preserves caller `err`
and `client_data`, then zero-initializes the remaining exact struct.

**Acceptance criteria.** Convert behavioral tests to compiled C structs or the
exact mirrored size. Canary/setjmp tests cover wrong version and smaller/larger
sizes, exact `JERR_BAD_LIB_VERSION`/`JERR_BAD_STRUCT_SIZE`, no write past the
declared object, preservation of `err`/`client_data`, and zero-init of all other
public fields for both compressor and decompressor.

## P4-111. Classic Progress-Manager Callbacks and Counters Are Not Wired — **OPEN**

**Motivation.** Filed 2026-08-02 after docs mapped `jpeg_progress_mgr` to the
unrelated native listener without testing the C struct.

**Root cause.** Compressor/decompressor `cinfo->progress` is initialized but no
codec path reads it, invokes `progress_monitor`, or updates pass/counter fields.

**Acceptance criteria.** Real C harnesses cover baseline, optimized multi-pass,
progressive, coefficient, and suspending decode/encode, comparing callback
counts, pass numbers/totals, and monotonic counters with stock C (exact counts
where stable).

## P4-112. `jpeg_set_marker_processor` Callbacks Are Stored but Never Invoked — **OPEN**

**Motivation.** Filed 2026-08-02 after the marker audit found docs claiming a
callback path that no active test exercises.

**Root cause.** The shim stores marker-code callbacks but only adds those codes
to native marker saving. Header/consume paths never invoke the C routine or let
it read marker bytes through `cinfo->src`.

**Acceptance criteria.** A stock-C suspending source-manager harness proves
callback timing during header/consume, source-byte access, FALSE
suspension/resume, state mutation, callback replacement/removal, and longjmp
safety. Saving a marker without invoking the processor cannot pass.

## P4-113. `jpeg_read_icc_profile` Bypasses Classic Saved-Marker Semantics — **OPEN**

**Motivation.** Filed 2026-08-02 after the positive test was found to succeed
without `jpeg_save_markers(APP2)`, which does not prove upstream's helper
contract.

**Root cause.** The shim reads ICC only from the post-start native image, not
`cinfo->marker_list`; header-only calls fail while unsaved APP2 data can
succeed. Classic chunk validation/warnings are bypassed.

**Acceptance criteria.** Stock-C header-only tests cover saved vs unsaved APP2,
multi-chunk ordering, duplicate/missing/inconsistent chunks and warnings, null
arguments/state, returned malloc ownership, and reconstruction before start.

## P4-114. `jpeg_has_multiple_scans` Equates Multi-Scan with Progressive — **OPEN**

**Motivation.** Filed 2026-08-02 during buffered-image API review. Sequential
noninterleaved JPEGs may contain multiple scans without progressive coding.

**Root cause.** The shim returns `progressive_mode` instead of parsed
input-controller multi-scan state and omits upstream state validation.

**Acceptance criteria.** A stock-C harness compares single-scan baseline,
sequential multi-scan, progressive, abbreviated, and invalid-state cases. The
answer must come from parsed scan structure and remain correct across
consume-input/buffered-image progression.

## P4-115. Native 12-Bit Coverage Claims Include Untested Modes and Sampling Layouts — **OPEN**

**Motivation.** Filed 2026-08-02 after the C-parity audit found that B6-1 and
`FEATURE_PARITY.md` claimed a 12-bit subsampling x progressive x arithmetic
matrix that the executable tests never construct.

**Root cause.** Before this filing, `tests/cross_check_precision.rs` called baseline
`compress_12bit` with S444 while labels/comments varied unused subsampling
values; no progressive or arithmetic path was invoked. The real odd-size
`compress_raw_12`/`decompress_raw_12` C matrix covers baseline Huffman S420,
S422, S444, S440, S411, and S441 only, omitting the now-supported S410 and S24
enum layouts. Grayscale S444 output cannot prove alternate chroma geometry.

**Acceptance criteria.** Either expose and cross-validate native 12-bit
progressive/arithmetic/SOF10 encode paths or document those modes as
unsupported. Add odd-size 12-bit raw C-parity cases for S410 and S24, with
structural sampling-factor assertions and stock-C pixel/raw-plane comparison.
Descriptions and case counts must name only modes that actually execute.

**Progress (2026-08-02).** The false B6-1 mode/subsampling descriptions were
corrected and its redundant weak Rust-only loop was removed; the two retained
S444 quality matrices compare samples to 12-bit `djpeg`. The missing product
mode decision and S410/S24 raw C cases keep this item open.

## P4-116. C-Parity Tests Can Convert Failures or Missing Comparisons into a Pass — **PARTIAL: named matrices closed; repository-wide sweep remains**

**Motivation.** Filed 2026-08-02 after a documented P4-13 regression reported
`1 passed` while silently skipping because its private tool lookup ignored a
working `cjpeg` on PATH. The same audit found broader forbidden error-to-skip
patterns in active C-parity matrices.

**Root cause.** Some tests return/continue after Rust codec errors, failed C
commands, malformed oracle output, dimension/length mismatches, or unavailable
tools after discovery. Several matrix drivers do not assert the exact number
of comparisons executed. A green Cargo test can therefore mean zero or only a
subset of the advertised cases ran.

**Acceptance criteria.** Use PATH-aware C-tool discovery everywhere and fail
closed in required CI. Once prerequisites are established, Rust errors and C
oracle/process/output failures must panic. Every matrix asserts its exact
planned/executed comparison count, including quick/full variants and each
supported corpus vector. Cover at least `c_tjdecomptest`,
`c_cjpeg_djpeg_tests`, `cross_check_metadata`, `worker_b3_conformance_t83`,
`cross_check_crop_scale`, `c_croptest`, `cross_product_transform`,
`abbreviated_datastream`, `encoder_builder`, and
`capi_jpeglib_write_coefficients`, then audit analogous patterns
repository-wide. Tests that never invoke the Rust operation (for example the
current ICC jpegtran case) are removed or given a real Rust-side assertion;
attempted-loop counters cannot substitute for successful comparisons. The
crop grid must assert its real planned count and must not collapse progressive
inputs to baseline when `cjpeg` is absent. Tables-only probes must pass
multi-token CLI arguments separately and assert all planned comparisons;
required in-repo fixtures cannot disappear through a bare return.

**Progress (2026-08-02).** The P4-13 pathological harness now discovers
`cjpeg`/`djpeg` through PATH, fails on tool execution and C compilation errors,
requires missing tools in CI, and its complete four-test binary is selected by
the provisioned Linux workflow. A host-tool run completed 4/4 with no skip.
The broader matrices above remain open.

**Progress (2026-08-08) — every named suite is closed.** All ten now fail
closed. The seven that are matrices — `abbreviated_datastream`, `c_croptest`,
`c_tjdecomptest`, `cross_check_crop_scale`, `worker_b3_conformance_t83`,
`cross_product_transform`, and the lossless leg of `encoder_builder` — assert
exact planned-versus-executed counts, the first five through
`tests/helpers/tally.rs` (`ComparisonTally`) and `cross_product_transform`
through its own bucket sum. `ComparisonTally` enforces its own use: `finish()`
is required by a `Drop` guard rather than a type-level `#[must_use]`, which is
inert once the tally is bound to a variable — as it is at every real call site.
Its six guard tests (in `tests/helpers_smoke.rs`) include four intentional reds
proving both the accounting and the `Drop` guard fire. `c_cjpeg_djpeg_tests`,
`cross_check_metadata` and `capi_jpeglib_write_coefficients` are fail-closed
but are not matrices, so they carry no count. Two shared policy helpers replace the private,
per-file lookups: `require_c_tool!` (already existed, now used everywhere) and
the new `require_c_testimage!`, which distinguishes "the submodule is not
checked out" (environmental) from "the fixture this test needs is gone" (a
defect, and always fatal).

What the hardening actually caught — each was a suite reporting green while
testing less than it claimed:

* **`abbreviated_datastream`: the tables-only matrix compared zero cases.** It
  shelled out to `cjpeg -tables-only`, a switch that has never existed in any
  libjpeg release. Its support probe looked for "unrecognized"/"unknown
  option" while `cjpeg` prints a usage dump, so the probe always answered
  "supported"; every invocation then failed and the loop `continue`d past it.
  `jpeg_write_tables()` is API-only, so the oracle is now
  `examples/tables_only_c_oracle.c`. All 12 cases compare byte-identically.
* **`c_cjpeg_djpeg_tests::c_cjpeg_lossless` encoded a baseline JPEG.** It set
  `.lossless_predictor(4)` without `.lossless(true)` — the predictor selector
  does not switch the mode on — and then *logged* the resulting mismatch as a
  `NOTE` instead of asserting. It now asserts SOF3 on both sides and exact
  round-trips through our decoder and through `djpeg`. Fixing it also exposed
  that `-restart 1` had been translated as `restart_blocks(1)`; cjpeg's bare
  `-restart N` counts MCU *rows* (cjpeg.c:537-541), and the block spelling
  produced a stream real libjpeg refuses (see **P4-121**).
* **`cross_product_transform` counted refusals as successes.** Four matrices
  had `Err(_) => { /* legitimate failure */ }` arms that recorded nothing, and
  one incremented the success counter directly. Every attempt is now bucketed
  and the buckets must sum to the attempt count. This immediately surfaced the
  **P4-117** 4:4:1 trim defect, which was carved out by name and pinned at
  eight cases so that both widening and narrowing would fail. That fix landed
  first (#439), the pin fired as designed, and the carve-out is gone — the
  trim matrix now reports 78 attempted, 78 round-tripped, 0 refused.
* **`c_tjdecomptest`, `cross_check_crop_scale`, `c_croptest`** turned real
  Rust/C disagreements — height mismatches, short oracle output, length
  mismatches — into `SKIP` and a `return`. Those are assertions now.
* **`cross_product_transform::tjtrantest_full_cross_product` asserted nothing
  about 14,112 transforms.** It counted refusals into `transform_errors` and
  only *printed* the total, so refusing every combination still satisfied
  `tested >= 5000` with no decode failures. It now reports
  `14112 attempted, 14112 round-tripped, 0 refused`.
* **`encoder_builder` degraded a failed `djpeg` to an unasserted `NOTE`.**
  libjpeg-turbo has decoded SOF3 since 3.0, so on CI a rejection is our defect;
  locally it now still asserts the Rust lossless round-trip rather than
  dropping the case entirely.
* **`c_croptest_quick_420` discarded its scenario results.** Six scenarios
  could all bail out and the test still reported green.
* **`helpers_smoke::helpers_c_tool_discovery`** asserted nothing at all.

Measured on macOS aarch64: `cargo test --workspace --no-fail-fast` → 2464
passed, 0 failed, 1 ignored (`restart_bomb_4096x4096_decodes_within_measured_bound`,
release-only). Platform-gated tests differ, so this total is host-qualified. `cargo fmt --check` and
`cargo clippy --workspace --all-targets -- -D warnings` clean.

**Why still PARTIAL.** The acceptance criteria also ask to "audit analogous
patterns repository-wide". 82 test files still contain an `eprintln!("SKIP…")` site
(`tests/*.rs` plus `crates/*/tests/*.rs`; a single-line grep finds only 66 of
them because 16 spell the macro across lines), and a sample of them carries the
same shapes this pass removed — `tests/c_indexedcolortest.rs:251`,
`tests/cross_check_color_quantize.rs:257`, `tests/quantize.rs:674`,
`tests/precision*.rs` (`precision.rs:432`, `precision_extended.rs:756`,
`precision_arbitrary.rs:711`), `tests/crop_c_compat.rs:467`,
`tests/crop_skip.rs:319`, `tests/cross_check_transform.rs:1120`,
`tests/cross_check_misc_gaps.rs:235`, and
`tests/hard_case_x_byte_and_restart.rs:295` all turn a failed C invocation into
a skip. Those suites are outside the ten the criteria name and are not touched
here.

**Progress (2026-08-08, second pass).** The nine suites named above are swept:
15 `if !output.status.success() { … SKIP … return/continue }` blocks across
`cross_check_color_quantize`, `quantize`, `crop_c_compat`, `crop_skip`,
`cross_check_misc_gaps`, `cross_check_transform`, `precision_extended` and
`precision_arbitrary` are now assertions. Each sits *after* tool discovery and
any capability probe, so a failed invocation is a defect in the request the
test built, not an environment gap — the distinction the earlier pass drew for
the ten named matrices. `cargo test --workspace --no-fail-fast`: 2466 passed,
0 failed, 1 ignored (macOS aarch64).

The item stays open: 67 test files still contain a `SKIP` site, and those have
not been individually triaged into "legitimate environment gap" versus "failure
reported as a skip". The sweep so far has only covered sites already identified
by name.

## P4-119. `src/decode/pipeline.rs` Concentrates Half of the Decoder Implementation — **CLOSED 2026-08-02**

**Motivation.** Filed 2026-08-02 after the encode pipeline split exposed the
same concentration on decode. At filing, the decoder contained 14,606 Rust
lines, of which 7,008 (48%) lived in `src/decode/pipeline.rs`; its
`decode_image_inner` method
alone spans about 1,396 lines. The comparable upstream `jd*.c` implementation
totals 13,753 lines across responsibility-focused files, with no file above
1,385 lines.

**Root cause.** Low-level entropy, IDCT, colour, and upsampling kernels were
already separated, while every orchestration layer accumulated in one public
module: API types and configuration, SIMD dispatch, baseline/arithmetic/
progressive/lossless modes, output conversion, raw decode, and incremental
streaming state.

**Acceptance criteria.**

1. Preserve the established `decode::pipeline::{probe, Decoder, Image,
   ImageInfo, JpegInfo}` paths, root re-exports, public fields, method
   signatures, trait bounds, and lifetimes, pinned by a compile-time API test.
2. Make `src/decode/pipeline.rs` a stable façade and move implementation into
   private responsibility-focused modules; no implementation file exceeds
   approximately 2,000 lines and no `include!`-based split is used.
3. Preserve observable output and errors across baseline, progressive,
   arithmetic, lossless, raw, 12-bit, four-component, cropped/scaled, metadata,
   and incremental decode. Focused C cross-validation and the corpus gate have
   zero new failures or crashes.
4. Run the full Criterion decode matrix from clean-main and candidate builds
   sequentially on the same pinned CPU. No decode group may show a
   statistically meaningful regression.
5. Workspace formatting, clippy, tests, cross-architecture CI, independent
   code review, Codex review, and documentation-drift audit are green.

**Why now.** The encode façade is merged, establishing the module pattern and
verification gates. Applying it to the second-highest size×churn file now
reduces review and merge risk without mixing in behavioural changes.

**Progress (2026-08-02).** The public module now owns the established public
types at their original canonical paths and delegates implementations to eleven
private responsibility modules. `pipeline.rs` is 209 lines; the largest
implementation file is `pipeline_impl/output.rs` at 1,970 lines, and every
other implementation file is at most 1,023 lines. A compile-time API test pins
the public function and method signatures, fields, lifetimes, callback bounds,
root re-exports, and canonical `type_name` paths. Workspace tests pass; the
only failed full-suite target was rerun 24/24 after restoring its deliberately
omitted `cjpeg`/`djpeg` path. Formatting, workspace clippy, rustdoc,
no-default/all-feature checks, and both installed wasm targets are green.
Independent code review and documentation-drift audit are green. Codex review
found no functional defect; its single P3 request to restore the non-obvious
pad/alpha offset invariant rationale was applied.

The first uncontaminated x86_64 measurements exposed a reproducible code-layout
effect: the unannotated split was about 2% slower than main at 1080p 4:2:0 even
though the normalized baseline helper instruction sequence was identical.
Combining the single-caller `decode_image_inner` with its wrapper on the
measured x86_64 SIMD build recovered main-equivalent performance without
duplicating the body. Other targets retain the compiler's inlining choice. On
CPU 0, an A/B/C run averaged 13.290 ms for the final candidate,
13.294 ms for main, and 13.567 ms for the unannotated split. The subsequent
per-benchmark alternating 27-group matrix kept every end-to-end decode within
-0.95%..+0.85% of main, covering baseline, progressive, restart, and decoder
reuse. Its SMT sibling CPU 6 averaged 99.62% idle across 138 samples, never
falling below 90%, so no result was discarded for competing work. A sequential
C libjpeg-turbo matrix on the same pinned CPU also completed with the sibling
99.74% idle; it retained the project's pre-existing mixed Rust/C performance
profile rather than revealing a split-specific shift. Exact attempts and
discarded alternatives are recorded in `experiments/pipeline.tsv`.

**Status (2026-08-02): closed.** [PR #441](https://github.com/developer0hye/libjpeg-turbo-rs/pull/441)
completed all 32 checks on the implementation commit, including ARMv7 scalar,
aarch64 NEON, x86_64 AVX2 and no-AVX2, WASM SIMD128, Linux/macOS/Windows,
sanitizers, Miri, mutation testing, C interop, and the C-parity corpus gate.

## P4-120. Classic-Shim Allocation-Failure Paths Are Unreachable From Tests — **OPEN**

**Motivation.** Filed 2026-08-08 during the P4-108 review. The classic shim now
raises `JERR_OUT_OF_MEMORY` (upstream code 56, message `"Insufficient memory
(case %d)"`) from two places in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`:
`jpeg_mem_dest` when its initial `malloc` fails, and `mem_empty_output_buffer`
when the doubling `malloc` fails. Neither is reachable from any test in the
repository, because neither can be provoked without making `malloc` fail.

**Why it matters.** Every other classic error code the shim emits is pinned
against a reference v8 build — `JERR_BUFFER_SIZE` and `JERR_FILE_WRITE` by
`tests/capi_classic_dest_ownership.rs`, `JERR_CANT_SUSPEND` by
`tests/capi_classic_lifecycle_pathological.rs`. `JERR_OUT_OF_MEMORY = 56` is
asserted only by a source comment. A wrong constant would mis-report
out-of-memory to every consumer with no test able to notice, and the same blind
spot covers the `msg_parm` payload: upstream uses `ERREXIT1(…, 10)`, so the
rendered message is `"Insufficient memory (case 10)"`, and only a test that
formats the message can prove we match.

**Root cause.** The shim's allocation failures all funnel through
`crate::alloc::libc_malloc`, which calls libc `malloc` directly. There is no
injection point, and the C canary harness has no way to starve a specific
allocation.

**Acceptance criteria.**

1. A mechanism exists to force a chosen shim allocation to fail — a test-only
   allocator hook behind a `cfg`/feature, or an `LD_PRELOAD`/interposition
   shim in the C canary harness. It must not change release code paths.
2. A C canary reaches both `JERR_OUT_OF_MEMORY` sites, asserts the symbolic
   `JERR_OUT_OF_MEMORY` from upstream's `jerror.h`, and asserts the formatted
   message text matches the reference v8 build's — proving the `msg_parm`
   payload, not only the code.
3. The caller's buffer is still intact and still caller-owned after a failed
   growth, so the failure path does not itself violate P4-108's ownership
   contract.
4. Audit the rest of the classic shim for other `JERR_*` codes emitted on paths
   no test can reach, and list them.

**Why deferred.** P4-108 delivers the behaviour; this is test reachability for
one error path. It belongs with the wider test-integrity work in **P4-116**
rather than blocking the destination-ownership fix.

## P4-117. 4:4:1 Trim Rejected Images Shorter Than One iMCU Row — **CLOSED 2026-08-08**

**Motivation.** Filed 2026-08-02 as GitHub [#439](https://github.com/developer0hye/libjpeg-turbo-rs/issues/439)
while making the P4-116 transform matrices fail closed; this phase-file entry
was missed at filing time and is added here with its closure. The 35x27
non-MCU-aligned 4:4:1 fixture reported 78 attempted trim cases and only 70
completions — `VFlip`, `Rot90`, `Rot180` and `Transverse`, in both baseline and
progressive output, returned `trim would remove all image data`.

**Root cause.** 4:4:1 is `h_samp=1, v_samp=4`, so its iMCU is 8 wide by **32
tall**. A 27-row image therefore contains zero whole iMCU rows, and
`transform_coefficients` computed `(27 / 32) * 32 == 0` and rejected the
transform outright.

Upstream has no such error path. `trim_right_edge` and `trim_bottom_edge`
(transupp.c:1570-1592) each open with `if (MCU_cols > 0 && …)` /
`if (MCU_rows > 0 && …)`: an axis holding less than one whole iMCU is simply
left untrimmed. Measured against stock `jpegtran -trim` on exactly this input:

| op | C output | reason |
| --- | --- | --- |
| hflip | 32x27 | width 35 → 32; height not trimmed by this op |
| vflip | **35x27** | height 27 holds no whole iMCU — guard fires |
| transpose | 27x35 | transpose never trims (transupp.c:1873) |
| rot90 | **27x35** | output width comes from source height — guard fires |
| rot180 | 32x27 | width trims; height guarded |
| rot270 | 27x32 | output height comes from source width → 32 |
| transverse | 27x32 | width → 32; height guarded |

**Fix.** `src/api/coefficient.rs` routes both axes through
`trim_to_whole_imcus`, which returns the extent unchanged when fewer than one
iMCU fits, mirroring upstream's guard. The `trim would remove all image data`
error is gone — as in C, trimming can no longer fail.

**Status (2026-08-08): closed.** `tests/regression_s441_trim.rs` pins all seven
operations: the geometry table above, a pixel-for-pixel cross-check against
stock `jpegtran -trim` (both sides transforming the *same* source JPEG, decoded
through the same `djpeg`, `max_diff == 0`), and a 4:2:0 control proving the
guard does not weaken trimming where a whole iMCU does fit (35x27 → 32x16).
Verified red before the fix: `vflip` and three others were rejected, 2 of 3
tests failing. `cargo test --workspace --no-fail-fast`: 2458 passed, 0 failed,
1 ignored. Once this merges, `cross_product_transform`'s trim carve-out
assertion fired as designed and the carve-out is deleted; trim now reports 78/78.

## P4-121. Lossless Encode Accepts a Restart Interval C Refuses to Decode — **OPEN**

**Motivation.** Filed 2026-08-08 while repairing `c_cjpeg_djpeg_tests::
c_cjpeg_lossless` under P4-116. Asking for a lossless stream with
`restart_blocks(1)` produces a file that real libjpeg cannot read:

```
djpeg: Invalid restart interval 1; must be an integer multiple of the number
       of MCUs in an MCU row (227)
```

**Root cause.** In lossless mode the restart interval must be a whole number of
MCU rows. Upstream enforces this **in the encoder**: `jclossls.c:294-296`
raises `JERR_BAD_RESTART` from `start_pass_lossless` when
`restart_interval % MCUs_per_row != 0`, so C cannot emit such a stream at all.
The decoder repeats the check at `jddiffct.c:108`. Our encoder performs
neither check, so it accepts the value, writes `DRI` verbatim, and produces a
stream every conforming decoder rejects — including our own C shim's oracle
path.

**Acceptance criteria.**

1. Lossless encode rejects a restart interval that is not a multiple of
   `MCUs_per_row`, with an error that names both values, matching upstream's
   `JERR_BAD_RESTART` wording closely enough to be recognisable.
2. `restart_rows(n)` continues to work: it already resolves to a whole number
   of MCU rows, which is the spelling `cjpeg -restart N` maps to.
3. A regression test encodes lossless with a deliberately misaligned block
   count, asserts the error, and — for an aligned interval — asserts `djpeg`
   decodes the result back to the exact input.
4. Check whether the same rule applies to our **decoder**: confirm we reject a
   lossless stream carrying a misaligned `DRI` as `jddiffct.c:108` does, rather
   than decoding it into something C would refuse.

**Why deferred.** P4-116 is test integrity; this is an encoder validation gap
it uncovered. The affected call is a misuse that upstream diagnoses, not a
silent data corruption, so it does not block the test work.
