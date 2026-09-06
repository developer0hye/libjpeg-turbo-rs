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
| P4-14 | PARTIAL (vtable enforces 2026-08-11, classic decode sequence 2026-08-13; strip-wise realization, allocated-overhead accounting and the suspending buffered path remain) |
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
| P4-97 | CLOSED 2026-08-14 (C algorithm shared by export + default callback; suspending-source trace vs stock, `discarded_bytes` survives suspension) |
| P4-98 | OPEN (classic 12/16-bit decode bypasses lifecycle and public output options) |
| P4-99 | OPEN (classic decode dispatcher ignores output options and colorspace metadata) |
| P4-100 | PARTIAL (classic codec failures are reported as suspension or silent success — translator + finish/start landed 2026-08-08) |
| P4-101 | OPEN (classic header parse does not publish coding tables/scan state) |
| P4-102 | OPEN (classic raw-data decode bypasses public options and state contracts) |
| P4-103 | OPEN (`jpeg_crop_scanline` does not implement iMCU-aligned C semantics) |
| P4-104 | CLOSED 2026-08-14 (state constants/transitions cross-validated; 17-row lifecycle trace vs stock) |
| P4-105 | OPEN (classic marker writers ignore state and declared lengths) |
| P4-106 | CLOSED 2026-08-14 (finish/abort lifecycle guards match stock's trace) |
| P4-107 | OPEN (`jpeg_enable_lossless` clamps invalid input and omits public state) |
| P4-108 | CLOSED 2026-08-08 (classic destination managers violate buffer ownership and I/O errors) |
| P4-109 | CLOSED 2026-08-14 (setup guards + chunked `FILE*` stdio reader, 10-case trace vs stock) |
| P4-110 | CLOSED 2026-08-11 (`jpeg_Create*` version/struct-size ABI guards, compared against a real libjpeg) |
| P4-111 | OPEN (classic progress-manager callbacks/counters are not wired) |
| P4-112 | OPEN (`jpeg_set_marker_processor` callbacks are stored but never invoked) |
| P4-113 | OPEN (`jpeg_read_icc_profile` bypasses classic saved-marker semantics) |
| P4-114 | OPEN (the reported bit matches upstream since 2026-08-13; `jpeg_has_multiple_scans` state validation remains) |
| P4-115 | OPEN (native 12-bit coverage claims include modes and sampling layouts that are not tested) |
| P4-116 | CLOSED 2026-08-08 (C-parity tests can convert Rust/oracle failures or missing comparisons into a pass) |
| P4-117 | CLOSED 2026-08-08 (4:4:1 trim rejected images shorter than one iMCU row) |
| P4-120 | CLOSED 2026-08-13 (`fail_nth_allocation_for_tests` makes both `jpeg_mem_dest` OOM paths reachable, code 56 + `msg_parm.i[0] == 10` asserted) |
| P4-121 | OPEN (lossless encode accepts a restart interval C refuses to decode) |
| P4-122 | OPEN (the Pillow smoke harness substitutes for a v6b library, which its own policy forbids) |
| P4-125 | CLOSED 2026-08-08 (TurboJPEG YUV decompress entry points emitted one plane per SOF component) |
| P4-126 | CLOSED 2026-08-09 (`yuv_plane_width`/`yuv_plane_height` accept any component index) |
| P4-127 | CLOSED 2026-08-09 (C-ABI YUV decompress entry points validate after decoding, not before) |
| P4-128 | CLOSED 2026-08-09 (YUV plane dimensions padded to the MCU size in pixels, not the subsampling ratio) |
| P4-142 | OPEN (`tj3DecompressHeader` decodes the entire image to read the header) |

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
- **`jpeg_consume_input`**: while `body_incomplete`, pull from the live source manager one chunk at a time (`pull_more_from_source_mgr`) and report the next boundary — `JPEG_REACHED_SOS` at each scan, `JPEG_REACHED_EOI` at EOI (clearing `body_incomplete`), or `JPEG_SUSPENDED` when the source is dry. **Since 2026-08-11 this drain runs only from a state upstream also drains from**: P4-104's `DSTATE_READY` guard returns `REACHED_SOS` ahead of it, because upstream consumes nothing until `jpeg_start_decompress` is called. P4-13's own harness enters from `INHEADER` and is unaffected; a caller that reaches SOS and then wants the body drained calls `jpeg_start_decompress`, which publishes `DSTATE_PRELOAD` and drains from there.
- **`jpeg_start_decompress`**: in buffered-image mode, publish output dimensions from the header and defer the pixel decode; in non-buffered mode, finish draining the body to EOI now (suspending if dry).
- **`jpeg_read_scanlines`**: materialise the deferred decode (`ensure_decoded_deferred`) once the body is complete.
- **`jpeg_input_complete`**: returns `FALSE` while `body_incomplete`, so the `while (!jpeg_input_complete()) jpeg_consume_input()` idiom drives the body to EOI. **Since 2026-08-14 (P4-104's closure) the answer is `eoi_seen`, upstream's `inputctl->eoi_reached`, and `body_incomplete` only suppresses it.**
- `jpeg_finish_decompress` / `jpeg_abort_decompress` reset the new state for handle reuse.

**Handed over 2026-08-14 by the [P4-104](#p4-104-classic-decompressor-state-constants-transitions-and-finish-lifecycle-diverge--closed-2026-08-14) closure — still open here.** Buffered-image startup publishes `DSTATE_SCANNING` where upstream publishes `DSTATE_BUFIMAGE` (`jdapistd.c:60-63`), and this shim never reaches `PRESCAN` or `BUFPOST` either. The buffered arms of `jpeg_finish_decompress` / `jpeg_start_output` / `jpeg_finish_output` therefore key on `SCANNING`/`RAW_OK` **plus** `buffered_image` rather than on the state alone, and a consumer that switches on `global_state` during a buffered-image decode reads a state upstream never publishes there. It lands here rather than in P4-104 because the branch that publishes it is this item's deferred-decode path. Acceptance: buffered-image startup publishes `DSTATE_BUFIMAGE`, the output-pass entry points walk `BUFIMAGE`/`BUFPOST` as upstream does, and the `buffered_image`-keyed arms above become plain state checks — oracle-compared like the P4-104 trace.

**Status (2026-06-02): PARTIAL — suspension is byte-exact-proven; deeper contracts are P4-26 and finish lifecycle is P4-104.** `cargo test -p libjpeg-turbo-rs-capi --test capi_classic_lifecycle_pathological consume_input_suspends_through_progressive_body` passes: a real suspending source manager drip-feeds a multi-scan progressive JPEG; the harness asserts mid-body suspension, SOS/EOI progression, resume, and pixels byte-identical to both full-buffer shim decode and stock `djpeg`. It no longer treats the shim's post-finish `DSTATE_STOPPING` as an oracle because upstream resets the cinfo. Boundary scanning, the 256 MiB drain cap, marker rebuild, scan tracking, and raw/coefficient body drain remain covered.

**Why PARTIAL, not CLOSED.** Codex round 8 (on commit `4645b52`) raised three upstream-contract-fidelity gaps that lie *beyond* the stated acceptance criteria but mean the broad title — "honor per-byte source suspension" — is not yet fully met across every entry point: (1) `jpeg_read_header` only stops at the first SOS on the *suspending* path (gated on `body_incomplete`); a fully-buffered consumer still has the whole body swallowed in `read_header`, so a later `jpeg_consume_input` reports `REACHED_EOI` immediately without per-scan `REACHED_SOS` callbacks. (2) Buffered-image *output* calls (`jpeg_start_output` / `jpeg_read_scanlines` / `jpeg_finish_output`) do not themselves pull from the source manager — a consumer driving decode purely through the output side on a still-`body_incomplete` handle makes no forward progress. (3) The `marker_list` is *rebuilt* from the completed stream rather than *appended* in place, so a `jpeg_saved_marker_ptr` a consumer retained mid-stream is invalidated by the rebuild. All three need a deeper, consumer-risky refactor (gap (1) changes every fully-buffered consumer's `read_header` behavior), none block T3, and no known consumer exercises them — so they are filed as [P4-26](#p4-26-deeper-streaming-contract-fidelity-beyond-the-p4-13-core--open) rather than expanding this PR's scope. The verified streaming-suspension core lands here.

## P4-14. `max_memory_to_use` Is ABI-Mirrored But Not Enforced in the C-Side Allocation Path — **PARTIAL: vtable + classic decode sequence enforce it with upstream's coefficient-only accounting; strip-wise realization, allocated-overhead accounting, and the suspending buffered path remain**

**Correction (2026-08-11): the error contract this item and [#467](https://github.com/developer0hye/libjpeg-turbo-rs/issues/467)
specify is wrong, and the "no spill path" constraint dissolves.** Recorded
before implementation so the work is not done twice.

Upstream does **not** raise `JERR_OUT_OF_MEMORY` when `max_memory_to_use` is
exceeded. The budget is consulted by `jpeg_mem_available`
(`jmemnobs.c:66-78`), which returns the remaining allowance;
`realize_virt_arrays` parcels the shortfall into strips
(`jmemmgr.c:745-760`) and only fails when it must spill — at which point
`jmemnobs.c:87-92` raises **`JERR_NO_BACKING_STORE` (51)**.

And the constraint this item records — that we have no backing store, so true
enforcement means reimplementing the spill path — **does not apply**:
`references/libjpeg-turbo/CMakeLists.txt:678` compiles `src/jmemnobs.c`
unconditionally. The library we are replacing has no backing store either. Our
"all data in RAM, never spills" is not a divergence; it is the same design, and
matching upstream is a ~60-line change in `realize_virt_arrays_impl` rather than
a new subsystem.

**Status (2026-08-11): partial.** Enforcement landed in
`realize_virt_arrays_impl`, which is the only place upstream consults the
budget:

* `virt_array_maximum_space` mirrors upstream's `maximum_space` accumulation
  (`jmemmgr.c:712-740`) with `checked_mul`/`checked_add` — including upstream's
  own overflow guard, which is `checked_add` written in C. P4-139's rule
  applies: these bound a real allocation.
* `budget_available` mirrors `jpeg_mem_available` (`jmemnobs.c:66-78`),
  including that `max_memory_to_use <= 0` means *unlimited*. The field is a
  signed `long`, so a negative value must not read as an enormous budget.
* `total_space_allocated` now exists, tracked in `push_block` and released in
  `free_pool` as upstream does (`jmemmgr.c:1140-1157`). Without it the check
  would compare each request against the whole allowance and let a sequence of
  small ones through.
* `invoke_error_exit` is `pub(crate)`, closing the "we lack the error-mgr
  handle" gap `memmgr.rs:732` recorded.

**Deliberate simplification, recorded rather than hidden.** Upstream parcels a
shortfall into strips and errors only for the arrays that still do not fit; we
allocate full height, so any shortfall is fatal. That is more conservative in *this*
dimension — it can reject a geometry upstream would have squeezed into strips.
It is not uniformly more conservative, and the claim that it "never accepts one
upstream rejects" would be false: the untracked manager and virtual-control
allocations recorded below mean we can accept, near the limit, what upstream
refuses. Strip-wise realization is a separate piece of work.

Proof: `capi_max_memory_budget.rs` (3 tests — budget below the working set
raises `JERR_NO_BACKING_STORE`, an ample budget still realizes, and a
non-positive budget means unlimited), verified to fail when the guard is
removed. `capi_classic_error_codes.rs` cross-validates code 51 and the message
"Memory limit exceeded" against the pinned v8 headers (18 codes now) — note
this pinned the *constant*, not what a C consumer saw. When this was written,
`format_message` rendered "bogus message code" for every error — filed and
since fixed as
[P4-146](#p4-146-jpeg_std_error-leaves-jpeg_message_table-null-so-every-classic-error-formats-as-bogus-message-code--closed-2026-08-13),
which also made that test render each code through our own formatter. The
message check is still how the first draft was caught guessing "Backing store
not supported" from the macro name; the real text is "Memory limit exceeded".

**Status (2026-08-13): the classic decode sequence is now bounded, shim-side
at `jpeg_start_decompress`.** `classic_budget_refuses_start`
(`jpeglib.rs`) mirrors upstream's single enforcement point: the budget
applies exactly when whole-image coefficient arrays would exist —
`has_multiple_scans` (progressive *or* non-interleaved sequential,
`jdinput.c:153-156`) or buffered-image mode (`jdmaster.c:709`) — and the
quantity weighed is **only** the coefficient-array bytes (summed from
`coef_array_geometries`, upstream's `realize_virt_arrays` accounting),
raising `JERR_NO_BACKING_STORE` (51) at start. The check sits before the
precision dispatch, so 12/16-bit streams are bounded the same way.

Two wrong shapes of the first draft (e492da9), both caught by adversarial
review against stock 3.1.4.1 and now pinned by the oracle matrix:

* *Gate was `is_progressive`.* Stock refuses multi-scan **sequential**
  streams (what `cjpeg -scans` emits) and **any** stream in buffered-image
  mode under a tiny budget; the draft accepted all of them. The
  `mss_tiny` / `buffered_baseline_tiny` oracle rows and the committed
  `tests/fixtures/mss_64x64.jpg` fixture pin this, and
  `jpeg_has_multiple_scans` now reports upstream's bit (it returned bare
  `progressive_mode` before, wrong for non-interleaved sequential — the
  `hms_*` oracle rows pin that too).
* *Quantity was the native whole-pipeline estimate* (output buffer +
  planes + coefficients, ~4.7× upstream's number on 4:2:0). Measured
  fallout: `djpeg -maxmemory 8192` on a 1024×1024 progressive image
  aborted against our dylib while stock accepts — working→broken for a
  shipped upstream flag. The `big_progressive_midband` oracle row (2 MiB
  of coefficients, 4 MiB budget, must accept) is the regression guard.

Proof: `capi_classic_decode_budget.rs` — a 13-case + 3-`hms`-row trace
compared verbatim against `examples/classic_budget_oracle.c` on stock,
plus oracle-independent standalone assertions; verified red with the
start-time check disabled.

**Second verification round (2026-08-13), all by measurement against
stock:** the coefficient footprint equals upstream's `maximum_space` to
the byte on 13 probed geometries (odd dimensions, 4:2:2 / 4:4:0 /
4:4:4, 4-component CMYK both samplings, 1×1 px, 2048×8); 12-bit is
bounded identically (probed through `jpeg12_read_scanlines` against
stock's 12-bit entry points — `prog12_tiny start 51`, `base12_tiny ok
0`); and upstream's `max_minheights >= 1` clamp (`jmemmgr.c:753-762`)
is ported: a stream whose every array fits one `maxaccess` window
(`v_samp`, ×5 progressive — `jdcoefct.c:897-901`) is accepted at ANY
positive budget, as stock is (7 of the 13 geometries; the
`short_progressive_tiny` oracle row pins the class — the pure
footprint-vs-budget first draft refused those below footprint).

**Still open within this item:** (1) strip-wise realization: on the
vtable path, full-height allocation refuses budgets upstream would
squeeze into strips; on the shim path the accept-any-budget window class
now matches stock, but for multi-window streams stock's refusal
threshold sits *above* `maximum_space` (strip mechanics plus the
`already_allocated` deduction of `jpeg_mem_available`,
`jmemnobs.c:66-78`) — measured at 1.5×–4.1× the coefficient bytes on
small images, 1.7% above at 1024×1024 — so between our threshold and
stock's we accept where stock refuses, the safe direction for a coarser
model but not parity. (2) A *suspending* source in buffered-image mode
defers the pixel decode past `jpeg_start_decompress` (P4-13/P4-26 path),
and the deferred materialization does not re-run the start-time check —
that corner is unbounded. (3) `jpeg_read_coefficients` needs no shim
check — it registers arrays through the vtable and refuses tiny budgets
exactly where stock does — but its bisected thresholds are looser in the
same direction (ours 8224 vs stock 26454 on a 64×64; 3145760 vs 3164182
at 1024×1024), the vtable half of residue (1).

Also outstanding: **the enforcement is stricter than upstream in one direction
and laxer in the other**, and both need recording rather than a single "matches
upstream" claim.

*Stricter:* upstream parcels a shortfall into strips of `maxaccess` rows and
fails only when even that minimum will not fit. We compare against the full
footprint, so a budget landing between the minimum and the full size succeeds
upstream and fails here. `capi_max_memory_budget.rs` pins *our* behaviour and
says so; it deliberately does not claim parity for that case. Closing it means
strip-wise realization.

*Laxer:* `total_space_allocated` starts at zero
and is updated only in `push_block`, so it excludes the `Combined` manager and
every `Box<JVirt*Control>` — upstream seeds the counter with the manager size
and allocates controls through tracked `alloc_small`. Near the limit we accept
what upstream rejects.

**Two regressions this work would have shipped, caught in review:**

1. `DEFAULT_MAX_MEMORY_TO_USE` was `1_000_000_000` with a comment calling it
   upstream's default and "advisory only; we never enforce a ceiling". Both
   halves were wrong the moment the field went live — upstream's
   `jpeg_mem_init` returns **0** (`jmemnobs.c:101-104`) — and enforcing the
   dormant value would have turned a harmless wrong constant into a live 1 GB
   cap rejecting workloads upstream accepts. Now 0.
2. The estimator assumed 1 byte per sample and counted already-realized arrays.
   The first let a 12/16-bit array pass a cap it then doubled; the second
   charged realized bytes twice, so a second no-op `realize_virt_arrays` failed
   where the first succeeded.

**P4-120** (forcing a shim *allocation* to fail had no injection point, and
the `msg_parm` payload of `JERR_OUT_OF_MEMORY` was unproven) was closed
2026-08-13 by the thread-local `fail_nth_allocation_for_tests` countdown in
`alloc.rs` — see its own section.

**Motivation.** Cold inspection of `crates/libjpeg-turbo-rs-capi/src/memmgr.rs` shows:

- `JpegMemoryMgr::max_memory_to_use: c_long` is at the correct upstream offset (compile-time `offset_of!` assertion at `:181`), defaulted to `~1GB` at `:817` — ABI fidelity is intact.
- Zero comparisons against `max_memory_to_use` exist anywhere in the file. `request_virt_sarray_impl` (`:527-551`), `request_virt_barray_impl` (`:558-582`), `realize_virt_arrays_impl` (`:591+`), `alloc_small_impl` (`:396`), `alloc_large_impl` (`:414`), `alloc_sarray_impl` (`:437`) all allocate without consulting the budget. No `JERR_OUT_OF_MEMORY` path is wired from a budget-exceed condition.

`docs/FEATURE_PARITY.md` lists `max_memory_to_use` as ✅ on the strength of `Decoder::set_max_memory()` / `TJPARAM_MAXMEMORY` honouring it in the **Rust** decode pipeline (now `src/decode/pipeline_impl/api.rs`). For the **C-ABI** consumer using `cinfo->mem->max_memory_to_use` directly (the upstream-documented path), the limit is silently ignored.

**Acceptance criteria — SUPERSEDED, retained for history.**

> The criteria below were written against a mistaken model of upstream: they
> ask for `JERR_OUT_OF_MEMORY` "msg_code 16" and assume upstream spills virtual
> arrays to a backing store. Neither holds. The budget is consulted by
> `jpeg_mem_available` and a shortfall raises **`JERR_NO_BACKING_STORE` (51)**;
> upstream's shipped build has no backing store either
> (`CMakeLists.txt:678` compiles `src/jmemnobs.c`). The live contract is the
> Status sections above (2026-08-11 vtable, 2026-08-13 decode sequence). Do
> not implement to the text below.

- A C harness that:
  1. Allocates `jpeg_decompress_struct`, sets `cinfo.mem->max_memory_to_use = N` where `N` is below the working-set size of a fixture (e.g. 64 MB cap on a progressive 4096² fixture with restart-every-MCU).
  2. Drives `jpeg_read_header → jpeg_start_decompress → jpeg_read_scanlines` and asserts the same exit path that upstream takes — `error_exit(JERR_OUT_OF_MEMORY)` (msg_code 16) on a budget-exceed virtual-array allocation, OR a documented divergence with deterministic alternative behaviour.
- Either: wire budget enforcement through `alloc_large_impl` / `realize_virt_arrays_impl` and the virtual-array spill path; OR document the divergence in `ABI_COMPATIBILITY.md` with a `cargo:warning=` when the field is set to a non-default value via `tj3Set(TJPARAM_MAXMEMORY)` or the C-ABI direct path.

**Why deferred.** Upstream uses backing-store spill to disk when virtual arrays exceed the in-memory budget. We have no backing-store implementation (`memmgr.rs:20-28`: *"This module keeps all virtual arrays in memory"*). Wiring true budget enforcement either reimplements the spill path or changes the failure semantics from "OOM kill or swap" to "explicit `JERR_OUT_OF_MEMORY` exit". Documenting first; implementing only on a named consumer requirement.

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

CMYK has both kinds in one image — components 0 and 3 carry the sampling factors, 1 and 2 sit at 1x1 — so no single rule is right for the whole image, and letting the per-block edge path clamp is right for neither when `v > 1`. This is the same distinction as [P4-47](#p4-47-progressive-encoding-diverges-from-cjpeg-at-every-even-height-not-a-multiple-of-16--closed-2026-07-26) (#324), arrived at from the other direction.

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

**The matrix gained a colorspace axis**, which is the part that generalizes: the next entry point to early-return past the option set gets caught by the suite rather than by someone thinking to look. It immediately earned its keep by surfacing [P4-54](#p4-54-colorspacergb-silently-ignores-progressive--arithmetic--lossless--closed-2026-07-26).

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

**Status (2026-07-28): closed.** The fix sketch above was itself falsified before merging: the corrected multi-filter form (`cargo test --tests -- cross_encode cross_check`) selects only **8 tests** on this workspace, because libtest filters match test *names*, and the tests inside `cross_check_*`/`cross_encode_*` files have names like `c_xval_decode_bgr_444` that contain neither substring. The job now runs the full unfiltered `cargo test --tests` on **macos-latest only** (timeout 15→30 min) — aarch64 + Homebrew jpeg-turbo 3.x, the one C-tool environment no other job covers. The former ubuntu leg is **removed**, not fixed: with apt's 2.1.x tools the unfiltered suite cannot run (codex review caught that e.g. `lossless_point_transform_matches_c_djpeg_exactly` feeds SOF3 to `djpeg` with no capability probe), and installing the official 3.1.4.1 deb would make the leg an exact environment+command duplicate of `Integration Tests` — the redundancy this entry's Impact paragraph already established. The "both legs non-zero" acceptance criterion is therefore satisfied in its intent (every remaining leg runs the full suite; no leg silently runs zero) rather than its letter. A comment in `ci.yml` pins the substring-vs-regex trap so a filter cannot quietly come back. **Mechanism-validated**, not validated-by-passing: with a deliberate encoder break (`FIX_0_299` 19595→20100 in `src/encode/color.rs`), `cargo test --test cross_check_encoder_binary` fails 3 of 4 byte-exact comparisons against `cjpeg`; reverted, green again. aarch64 + Homebrew first run: the full `--tests` suite was executed on a macOS aarch64 host with Homebrew jpeg-turbo before merging — no divergence surfaced, so nothing needed filing; the PR's own `C Interop (macos-latest)` leg is the first CI proof and must show a non-zero test count. **Amended 2026-08-18 by [P4-130](#p4-130-c-parity-oracle-is-pinned-to-3141-upstream-stable-is-320--partial-every-oracle-provisioning-job-is-now-pinned-checked-and-measured-the-legs-still-on-one-release-the-submodule-bump-and-the-four-filed-gaps-remain):** the environment is still aarch64 macOS, but the oracle is no longer `brew install jpeg-turbo` — the leg builds 3.1.4.1 from source at `/tmp/ljt3141/prefix`, asserts it, and selects it with `LIBJPEG_TURBO_PREFIX`.

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

**Scope note (2026-08-13, from the P4-143 review):** the `src/simd/*_tests.rs`
in-module suites pass vacuously when the `simd` feature is off — with `simd`
disabled on aarch64 all 11 `simd_parity_tests` still run and report `ok`
while every assertion sits inside a now-excluded `cfg` block (lib test count
drops 313 → 232). No CI job runs tests with `--no-default-features`, so
nothing is fooled today, but it is the "don't assert around an unrunnable
path" shape: when this item restructures the SIMD tests, gate the *test
functions* on the same predicate as the kernels they assert against, so a
config with no kernels has no green tests claiming parity.

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

**Update (2026-08-14), from the P4-109 closure.** All three cited sites are gone: `read_c_file` was deleted with the fd-dup slurp, taking the `FromRawHandle`/`RawHandle` imports and the unused `file` parameter with it, and `use std::io::Read;` was removed from the same file. `jpeg_stdio_src` now reads through the caller's `FILE *` with C-ABI `fread` and has no platform gating, which also discharges the "functionally incomplete Windows arm" half of the criteria. The item stays OPEN because the criteria are a *measurement* — `cargo clippy --workspace --all-targets -- -D warnings` on a Windows host — that has not been re-run since; re-measure before closing rather than closing on this note.

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

**Correctness is not in question.** The scalar path is the reference every SIMD kernel is validated against, and `tests/no_std_dispatch.rs::scalar_dispatch_matches_the_default_dispatch` pins scalar output == host-dispatch output. What ARMv7 adds beyond that is a **32-bit hardware ISA** — native ARM codegen, the armhf ABI, real unaligned-access semantics. (32-bit *pointer width* alone was already executed: `wasm.yml` runs `cargo test --target wasm32-wasip1` under wasmtime — SIMD128-dispatched, since `.cargo/config.toml` forces `+simd128` for in-tree wasm builds.) That is now gated by the `Test (linux-armv7 scalar, emulated)` leg (`.github/workflows/armv7.yml`), which cross-builds for `armv7-unknown-linux-gnueabihf` and runs the suites under `qemu-arm` — **204 tests executed** (196 lib + 6 `simd_dispatch` + 2 `no_std_dispatch`), 0 failed, on qemu-arm 8.2.2 ([first green run](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/30519595108/job/90796847495)). Verified by mechanism rather than by the job passing: the log's per-binary `running N tests` / `test result: ok` lines were read to confirm the counts, since a filter or target mistake here would otherwise produce the vacuous green that P4-61 documents. That leg is a correctness gate only: unlike the RISC-V harness, the two sides would *not* pay a symmetric emulation tax here, since C's NEON kernels would be emulated while our scalar ones are not.

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

## P4-81. Linux cdylib Omits GNU `LIBJPEG_8.0` Symbol Versions — **PARTIAL: nodes emitted and tested; downstream re-verification pending**

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

**Progress (2026-08-08) — the nodes exist.** `build.rs` generates a GNU
version script and passes it to the linker on ELF targets, gated on the v8
SONAME: an artifact built with `CAPI_SONAME` set to something else gets a
`cargo:warning` instead of a silently wrong v8 label.

The map mirrors upstream `src/libjpeg.map.in` — `LIBJPEGTURBO_8.0` owning
`jpeg_mem_dest` / `jpeg_mem_src` and localising `jsimd_*` / `jconst_*`,
`LIBJPEG_8.0` owning the reference API — with **one deliberate deviation that
the issue's scope did not anticipate**.

Upstream's `LIBJPEG_8.0` node is `global: *`. It can afford a catch-all because
it builds *two* libraries: `libjpeg.so.8` from that map and
`libturbojpeg.so.0` from `src/turbojpeg-mapfile`, which assigns each `tj*`
symbol to the `TURBOJPEG_1.0`…`TURBOJPEG_3.0` node it was introduced in. We
ship **one** artifact carrying both surfaces (92 `jpeg_*` exports and 63
`tj*`/`TJ*`), so applying upstream's map verbatim would stamp every TurboJPEG
export as reference libjpeg API — precisely the mislabelling this item forbids.
Re-versioning them under a node of our own would be worse: a consumer linked
against a real libturbojpeg requests `tjInitCompress@TURBOJPEG_1.0`, and the
loader fails outright when the library offers that name under a different
version.

So the map has **no catch-all**. Symbols matched by no node keep default,
unversioned visibility — exactly their status today — so the TurboJPEG and
crate-only surfaces are unchanged while the classic API gains the nodes
prebuilt consumers look for. The `local:` clause hides nothing we ship: the
crate exports no `jsimd_*` or `jconst_*` symbol.

`tests/capi_symbol_versions.rs` splits verification by what each layer needs.
The script *content* — node names, membership, and the absence of a catch-all —
is asserted on every platform from the generated file, because that is the part
most likely to regress and it needs no Linux. The *ELF result* is asserted with
`readelf --version-info` and `--dyn-syms` where ELF versioning exists, checking
both that the nodes exist and that `jpeg_CreateDecompress` / `jpeg_read_header`
land in `LIBJPEG_8.0`, `jpeg_mem_*` in `LIBJPEGTURBO_8.0`, and `tj3Init` in
neither — a map that defined the nodes but matched nothing would otherwise pass
a nodes-exist check while leaving every symbol unversioned.

**Hard blocker found by CI (PR #447): rustc's own version script.** Adding
`-Wl,--version-script` cannot work as designed. The link fails with:

```
/usr/bin/ld: anonymous version tag cannot be combined with other version tags
```

because rustc already passes a version script for every cdylib:

```
-Wl,--version-script=.../deps/rustc*/list      <- rustc's
-Wl,--version-script,.../out/libjpeg.map       <- ours
```

rustc's uses an *anonymous* version tag — `{ global: …; local: *; };`, no name —
to export `#[no_mangle]` items and hide the rest. GNU ld forbids mixing an
anonymous tag with named ones, and rustc's script is not suppressible on
stable. The map's content is correct and its content tests pass; the two
scripts are simply mutually exclusive.

P4-81 therefore needs a different mechanism. Candidates, none free: a post-link
rewrite that synthesises `.gnu.version_d` / `.gnu.version` (patchelf cannot do
this today); replacing rather than augmenting rustc's script (no stable knob);
requiring `lld`, which pushes a toolchain constraint onto packagers; or linking
the shared object ourselves from a staticlib, a large build-system change.

**Settled by experiment (2026-08-08), binutils 2.47 via `x86_64-elf-binutils`.**
The candidate list above was written from the CI failure alone. Linking the
cases directly narrows it to one answer, and kills the cheap options
individually. `a.o` exports `jpeg_read_header`, `jpeg_mem_dest`, `tj3Init`;
`rustc_like.map` is `{ global: …; local: *; };` — an anonymous tag, exactly
what rustc emits.

| # | configuration | result |
| --- | --- | --- |
| 1 | rustc's anonymous script **+** our named script | `anonymous version tag cannot be combined with other version tags` — reproduces CI exactly |
| 2 | `.symver` in the object **+** rustc's script only | `version node not found for symbol jpeg_mem_dest@@LIBJPEGTURBO_8.0` |
| 3 | rustc's anonymous script **+** `VERSION { … }` in a linker-script *input file* | same anonymous-tag error as #1 |
| 4 | our named script **alone**, `local: *` inside the first node | **works** — emits `.gnu.version_d` with `LIBJPEG_8.0` and `LIBJPEGTURBO_8.0`, and binds the symbols to them |

Case 2 matters because `.symver` looks like a way to sidestep version scripts
entirely, and it is not: the directive *attaches* a symbol to a node, it does
not *define* the node, so a script is still required. Case 3 matters because
`VERSION { … }` inside a linker script is a different surface from
`--version-script` but merges through the same code path, so it fails
identically. Neither is a workaround.

Case 4 is the whole finding: the target ELF is reachable with **exactly one**
version script — ours, carrying `local: *` so it also does the job rustc's was
doing. And rustc 1.94.1 exposes no way to stop emitting its own: `-C help` and
`-Z help` list nothing for version scripts or symbol visibility.

So the remaining question is not *which script mechanism* — all of them are the
same mechanism — but **who runs the link**. The bounded version of that is to
produce the versioned `libjpeg.so.8` from the `staticlib` (already in
`crate-type`) at *install* time in `scripts/install_capi.sh`, rather than
restructuring the cargo build: the acceptance criterion is about the installed
library prebuilt consumers bind to, and about the OpenCV `no version
information available` warning, both of which are properties of the staged
artifact. The cdylib cargo emits would stay unversioned, which would need
saying plainly in `docs/ABI_COMPATIBILITY.md` so nobody reads a `cargo build`
artifact as the shippable one.

**A consequence #437's scope does not mention, found by CI (PR #447).** Adding
version nodes *removes glibc's unversioned-fallback path*, and the Pillow smoke
leg immediately failed with:

```
version `LIBJPEG_6.2' not found (required by .../PIL/_imaging...so)
```

That `_imaging.so` was built against a **v6b** libjpeg. With no version nodes
the loader bound it to our v8 shim anyway; with nodes present it correctly
refuses. The failure is the feature working — a v6b consumer had been binding
to a v8 struct layout, which is the silent ABI mismatch T4's non-goal status
and the "v6b substitution is not valid T3 evidence" rule both exist to prevent.

Adding a `LIBJPEG_6.2` node would "fix" CI by re-asserting a v6b ABI we do not
implement, and is rejected. The correct resolution is to make
`examples/pillow_smoke/run.sh` take its documented v8-rebuild path in that leg
rather than overwriting Pillow's bundled `libjpeg-*.so.62.4.0` in place.

**Status (2026-08-08): implemented and CI-verified; open for its downstream
criterion.** The nodes now exist on the shipped library and are asserted on a
Linux runner.

*What ships them.* Not the cdylib -- it cannot carry them, per the experiment
above. `scripts/install_capi.sh` relinks `libjpeg.so.8` from the `staticlib`
with the generated map as the only version script (`--whole-archive`, because
nothing references the `#[no_mangle]` exports; `--allow-multiple-definition`
for compiler_builtins symbols that also arrive from libgcc). A missing
prerequisite warns and degrades; a relink that is attempted and fails stops the
install, so an unversioned library cannot be shipped as though it were
versioned. `CAPI_SKIP_SYMBOL_VERSIONS=1` opts out.

*What proves it.* `Classic C-ABI GNU symbol versions (P4-81)` in `Integration
Tests` runs `capi_symbol_versions` with `--nocapture` and fails closed on any
`^SKIP`. The test stages the library through the install script itself and
asserts the script reported `P4-81: relinked` before reading the ELF, so a
degraded install fails rather than skipping. Green with
`installed_library_exports_the_reference_version_nodes ... ok`.

*Three false greens preceded that, all found by asking whether the leg ran
rather than whether it passed*, and each is worth recording because they are
distinct failure modes:

1. **The PR was `CONFLICTING`.** GitHub cannot build a merge ref for a
   conflicting PR, so `pull_request` workflows never start at all. Two pushes
   produced zero Actions runs while the PR showed a green DCO check. Rebasing
   fixed it.
2. **The test skipped.** Its first draft looked for a leftover `*.versioned`
   artifact and skipped when it found none; nothing in CI stages one. It now
   stages one itself.
3. **CI never invoked the suite.** This workflow selects the C-ABI crate's
   tests by name, and `capi_symbol_versions` was not among them -- a test
   nothing calls cannot fail. Now wired in.

A fourth defect surfaced only once the leg genuinely ran: the assignment
spot-check used `line.contains(symbol)`, which matched the Rust-mangled
`_RNvXsl_...jpeg_CreateDecompress0E...` and asserted against it instead of the
C symbol. Symbol names are compared exactly now.

*Still open, and it is what closes this item.* The acceptance criteria ask for
the OpenCV replacement harness to be re-run and the `no version information
available` warning confirmed gone, plus libtiff/GDAL/Poppler/HDF4 still
loading. Those need the downstream lab (P2-G). The nodes existing is necessary,
not sufficient -- the warning disappearing is the criterion.

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

## P4-97. `jpeg_resync_to_restart` Is an Unconditional Success No-Op — **CLOSED 2026-08-14**

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

**Status (2026-08-10): partial — the algorithm is implemented and shared.**

`jpeg_resync_to_restart` and the source manager's default callback (formerly
`noop_resync_to_restart`, now `default_resync_to_restart`) both forward to one
`resync_to_restart_impl`, so they cannot drift — the "one C-exact
implementation" half of the criteria. It is a direct port of `jdmarker.c`'s
decision table:

| Found marker | Action | Effect |
| --- | --- | --- |
| `< M_SOF0` (0xC0) | 2 | scan forward via `next_marker` |
| valid non-restart | 3 | leave `unread_marker` set |
| `RST(desired+1)` / `RST(desired+2)` | 3 | leave `unread_marker` set |
| `RST(desired-1)` / `RST(desired-2)` | 2 | scan forward |
| desired restart, or > 2 away | 1 | clear `unread_marker`, resume |

`next_marker` is ported too: it pulls bytes through `src->fill_input_buffer`,
skips non-`FF` bytes, swallows `FF` padding runs, discards stuffed `FF 00`
sequences, counts discarded bytes for `JWRN_EXTRANEOUS_DATA`, and returns
`FALSE` on suspension. `JWRN_MUST_RESYNC` is emitted unconditionally before the
decision, as upstream does.

Warning codes came from a compile-time probe against the installed C headers,
not from counting `JMESSAGE` lines in `jerror.h` — that file has
`#ifdef`-duplicated entries, so a naive scan is off by one. `JWRN_MUST_RESYNC
= 124`, `JWRN_EXTRANEOUS_DATA = 119`; the same probe reproduced this crate's
existing `JERR_UNKNOWN_MARKER = 70` and `JERR_INPUT_EMPTY = 43`, which is what
calibrates it.

Proof: `cargo test -p libjpeg-turbo-rs-capi --test capi_resync_to_restart` — 10
tests (12 with the review additions below) over the decision table, the warning,
the recovery-action trace, scan-forward against a real memory source, both
suspension paths and NULL. **Falsification measured**: patching the
implementation back to `return 1` fails 6 of the original 10. The remaining 4
are action-3 cases, where a constant `TRUE` that never touches `unread_marker`
yields the same observable; they are kept to pin the other half of the table,
and the test file records that they are *not* evidence against the old bug.

**Three fidelity defects were found by review of the first cut, not by tests**
— worth recording because each passed the 10 tests above:

1. **Aliasing UB.** The first version held a live `&mut JpegDecompressPublic`
   across `emit_message`, and a live `&mut JpegSourceMgr` across
   `fill_input_buffer`. Both callbacks receive `cinfo` and may legitimately
   touch it — `fill_input_buffer` is *required* to mutate the manager — so the
   exclusive references were aliased and every later use was UB. Now only raw
   pointers cross a callback, with fresh short-lived references either side.
2. **Suspension lost the restart point.** Upstream keeps a *local* cursor and
   commits it via `INPUT_SYNC` only at restart boundaries (`jdmarker.c:118-121`,
   explicitly: *"we update them only when we have reached a suitable place to
   restart if a suspension occurs"*). The first version committed every byte
   eagerly, so a chunk ending on `0xFF`, or between `0xFF` and a stuffed `0x00`,
   consumed a speculative prefix the retry needed. `ResyncInput` now mirrors
   `INPUT_VARS`/`INPUT_SYNC` with C's exact sync points.
3. **`TRACEMS2` was skipped.** Upstream traces `JTRC_RECOVERY_ACTION` before
   applying each action, and `TRACEMS2` publishes `msg_code`/`msg_parm`
   regardless of `trace_level`. Without it a C consumer inspecting
   `err->msg_code` after the call saw `JWRN_MUST_RESYNC` (124) where stock
   leaves `JTRC_RECOVERY_ACTION` (99).

A fourth finding fixed the **built-in memory source** rather than resync itself:
`fill_input_buffer` returned `TRUE` while supplying no bytes. Stock's
`fill_mem_input_buffer` warns `JWRN_JPEG_EOF` and inserts a fake `FF D9`
(`jdatasrc.c:125-137`). The old shape read as "more data arrived" to every
caller and made scan-forward report suspension on a source that can never
resume; it also let `drain_caller_source_mgr` spin on a truncated stream. This
is really P4-109's territory and is noted there.

**What remained** was the harness the criteria actually name: a real
**suspending C source manager** cross-validated against stock libjpeg-turbo,
exercising refill mid-scan and a `FALSE` return under genuine suspension. The
2026-08-10 tests drove the algorithm directly; they reached a real source at
end-of-buffer, but not a manager that genuinely suspends and resumes.

One known divergence for that harness to close: upstream keeps `discarded_bytes`
in the private `cinfo->marker`, so the count survives a suspension mid-scan;
the port threaded it as a local, so a scan split by suspension under-reported
the byte count in `JWRN_EXTRANEOUS_DATA`. The chosen action and the return
value were unaffected.

**Status (2026-08-14): closed.** `LIBJPEG_TURBO_PREFIX=… cargo test -p
libjpeg-turbo-rs-capi --test capi_resync_suspend` compares the full
suspending-source trace verbatim against
`examples/classic_resync_suspend_oracle.c` linked to stock: a source manager
implementing libjpeg.txt's suspension contract (the driver authorizes bytes in
stages; fill re-presents the unconsumed restart-point tail or returns `FALSE`),
driven over seven scenarios — desired RST, both next-expected RSTs, a non-RST
marker, a prior RST scanning garbage that spans a suspension, an invalid byte
over a stuffed-zero pair, and a suspension inside an FF pad run. The trace pins
return values, `unread_marker`, the full `emit_message` sequence
(`JWRN_MUST_RESYNC` / `JTRC_RECOVERY_ACTION` / `JWRN_EXTRANEOUS_DATA` with both
parameters), and per-fill `bytes_in_buffer` — the last being the observable
form of `INPUT_SYNC` discipline (scenario s6's refill sees the two uncommitted
pad FFs re-presented, `bib 2`).

The `discarded_bytes` divergence is fixed the way upstream stores it: the count
now lives in `DecompressPrivate::resync_discarded_bytes` (the shim's stand-in
for `cinfo->marker->discarded_bytes`, `jpegint.h:395`), loaded into a working
copy at `resync_to_restart_impl` entry, written back at every exit, and zeroed
where upstream's `reset_marker_reader` runs — the fresh-datastream parse.
**Falsification measured**: reverting only the entry load to a constant `0`
flips exactly scenario s4's warning from `p0 8` to `p0 3` (5 bytes discarded
before the suspension + 3 after; stock reports 8) and nothing else. CI runs the
suite against the pinned-submodule build in the "Classic C-ABI state
transitions (P4-104)" step, where the prefix makes the comparison mandatory
rather than skippable.

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

**Correction (2026-08-08).** An earlier revision of this entry claimed the shim
"only ever enters `CSTATE_START` and `CSTATE_WRCOEFS`, never `CSTATE_SCANNING`
or `CSTATE_RAW_OK`", and cited that as the blocker for matching upstream's
state-gated finish contract. That was wrong. `jpeg_start_compress` does set
them (`jpeglib.rs`, `c.global_state = if c.raw_data_in != 0 { CSTATE_RAW_OK }
else { CSTATE_SCANNING }`); the claim came from a grep for the literal
`global_state = CSTATE_`, which does not match that computed form.

`jpeg_finish_compress` now implements jcapimin.c:184-190 verbatim: the
`next_scanline < image_height` check runs for `CSTATE_SCANNING` **and**
`CSTATE_RAW_OK`, `CSTATE_WRCOEFS` passes through, and any other state raises
`JERR_BAD_STATE` with the state value. Both raw-data writers advance
`next_scanline`, so the raw arm is real — the previous revision exempted it and
would have let a short raw encode produce a file.

**Status (2026-08-08): partial.** Criteria (1)-(3) hold for
`jpeg_start_decompress` and `jpeg_finish_compress`, and the finish state gate
now matches upstream exactly. Nine more sites now raise: every
raw-data validation failure in `jpeg_read_raw_data`, `jpeg12_read_raw_data`,
`jpeg_write_raw_data` and `jpeg12_write_raw_data` already *named* its
upstream code in the message (`JERR_BUFFER_SIZE`, `JERR_BAD_PRECISION`,
`JERR_BAD_STATE`) but never reached `error_exit`, so a C consumer saw only a
`0` return. `JERR_BAD_PRECISION` joins the cross-checked constant set.

Not yet done: the remaining private-string-only
sites (45 of the original 59 `last_error` assignments still have no
`error_exit` nearby), criterion (4)'s stock-versus-Rust setjmp harness matrix,
and P4-104's *decompressor* state work, which is separate from the compressor
states corrected here. `cargo test --workspace --no-fail-fast`: 2466 passed, 0 failed,
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

**Extended 2026-08-17 by the [P4-130](#p4-130-c-parity-oracle-is-pinned-to-3141-upstream-stable-is-320--partial-every-oracle-provisioning-job-is-now-pinned-checked-and-measured-the-legs-still-on-one-release-the-submodule-bump-and-the-four-filed-gaps-remain)
3.2 delta triage.** 3.2.0 note 3 hardened this entry point, and the delta is
exactly one condition: 3.2.0 `src/jdapistd.c:203` reads
`if (cinfo->master->lossless || cinfo->raw_data_out)` where 3.1.90 reads
`if (cinfo->master->lossless)`. `jpeg_crop_scanline()` has never worked with
raw-data output; before 3.2.0 it simply did not say so, and the combination of
buffered-image mode and raw-data output could reach freed memory when a plane
was cropped to one sample. So the harness above must also assert
`JERR_NOTIMPL` for `raw_data_out`, not merely the aligned-crop geometry.

## P4-104. Classic Decompressor State Constants, Transitions, and Finish Lifecycle Diverge — **CLOSED 2026-08-14**

**Motivation.** Filed 2026-08-02 after P4-13's harness was found to assert the
shim's `DSTATE_STOPPING`, the opposite of upstream's abort-reset completion.
That false oracle was removed in the filing PR.

**Root cause (partially corrected 2026-08-11; see the Progress note below).**
A *direct* `jpeg_read_header` still leaves `DSTATE_INHEADER` instead of READY.
The consume-driven path now reaches READY and has upstream's repeated-call
guard, so the two entry points disagree; closing that is the call-direction
inversion. Finish clears caches/source and returns TRUE rather than
rejecting unread rows or a bad state, draining EOI with suspension, and calling
`term_source`.

Two pieces of the original root cause are now fixed and are called out so they
are not re-implemented: the state **numbering** (all 15 constants match), and
finish's **final state** — it ends at `DSTATE_START` as upstream's
`jpeg_abort`-terminated finish does, rather than stopping at STOPPING.

> **The state *numbering* is correct, and the paragraph above used to say
> otherwise.** It opened by claiming the shim defines `DSTATE_STOPPING = 206`
> against upstream's 210 with intermediate states missing. That was fixed —
> the Status section below documents it, including the red-test proof — but
> this paragraph was never updated, so the item's own root cause contradicted
> its own status.
>
> **A smaller finding about where that guard runs.** It is a `--lib` unit test
> in the C-ABI crate, and CI's `Unit Tests` job runs `cargo test --lib`, which
> selects the default workspace member — the root crate — only. So it does not
> run *there*. It does run on every PR, under `sanitizers.yml`, whose ASan and
> UBSan jobs both invoke `cargo test --workspace --lib`.
>
> An earlier draft of this note claimed the C-ABI crate's 24 unit tests "never
> executed in CI". **That was wrong**, and is corrected here rather than
> quietly dropped — the sanitizer jobs had them covered. What the new
> `cargo test -p libjpeg-turbo-rs-capi --lib` step adds is coverage on the
> *stable, uninstrumented* toolchain, so a failure in these tests is not
> confounded with sanitizer instrumentation and does not depend on the
> sanitizer workflow being reached.
>
> Found by writing a *new* integration test for this criterion, which turned
> out to be a weaker duplicate of the existing one — it enumerated a hardcoded
> list, so a sixteenth upstream state would not have failed it — and was
> deleted rather than merged.
>
> **What remains under P4-104 is the transitions**, which the paragraph above
> now describes on its own.

**Progress (2026-08-11): `jpeg_input_complete`'s state guard adopted; its
*answer* deliberately not, and the reason is the interesting part.**

Upstream is three lines (`jdapimin.c`):

```c
if (cinfo->global_state < DSTATE_START || cinfo->global_state > DSTATE_STOPPING)
  ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);
return cinfo->inputctl->eoi_reached;
```

**Adopted:** the range guard. A state outside `DSTATE_START..=DSTATE_STOPPING`
now raises `JERR_BAD_STATE` where this shim returned a quiet `FALSE` — telling a
caller "keep polling" about a corrupt `cinfo` is the opposite of what it needs.

**Also fixed:** `jpeg_finish_decompress` left `DSTATE_STOPPING`. Upstream's
finish ends with `jpeg_abort` — "We can use jpeg_abort to release memory and
reset global_state" — so a caller observes `DSTATE_START`; STOPPING is the state
finish passes *through* while draining to EOI.

**Not adopted, and this is the finding:** returning `eoi_reached`. It requires
`eoi_seen` to mean what `inputctl->eoi_reached` means, and it cannot while
`jpeg_consume_input` diverges — for a fully-buffered stream ours returns
`JPEG_REACHED_EOI` as soon as the header is parsed, where upstream returns
`JPEG_REACHED_SOS` and reaches EOI only on a later call.

Four successive attempts to maintain the flag around that divergence each got a
*different* shape wrong, and each was caught by review rather than by the suite:

1. it was never **set** on the fully-buffered or tables-only EOI returns;
2. nor after the eager decode in `jpeg_start_decompress`, so a non-buffered
   multi-scan image reported "keep polling" on a finished stream;
3. it was **cleared too eagerly**, in finish and abort — upstream clears
   `eoi_reached` only when the next datastream read begins, so its
   `jpeg_input_complete` still answers TRUE after a successful finish;
4. gating the startup mark on `progressive_mode` was **too narrow** (a
   sequential multi-scan SOF0 stream preloads upstream but is not
   progressive) after being **too broad** (baseline and buffered-image do not
   preload at all).

That is a model that was wrong four times, not code that was wrong once. The
flag stays out of the answer until `jpeg_consume_input` is restructured to
upstream's shape — which is the same work `DSTATE_READY` needs, so the two land
together.

`capi_input_complete_contract.rs`: **5 passing, 3 `#[ignore]`d**. One of the five, `finish_decompress_resets_state_to_start`, exists because the other two finish-related tests are ignored — without it the `DSTATE_START` change would have had no enabled gate at all, and reverting it would have left every enabled test green. The ignored
three encode upstream's contract for the cases above and unignore with the
restructure — executable specification rather than prose, per this repo's rule
that an unready implementation gets `#[ignore]` with a reason rather than a
loosened assertion. They are the first product-path ignored tests in the suite;
the live-gate line in `LAST_MILE.md` records them.

**Next in this item:** `DSTATE_READY` and the repeated-call guard landed
2026-08-11 for the consume-driven path (step 2 below). What remains is the
call-direction inversion — restructuring to upstream's shape (`jdapimin.c`),
where `jpeg_read_header` calls `jpeg_consume_input` rather than the reverse, so
a *direct* header read also lands on READY — and then `jpeg_start_decompress`
accepting READY. The three ignored tests are the acceptance criteria for that
work.

### Measured divergence map (2026-08-11)

Written before implementing, because the four failed attempts above all came
from *not* having it: each patched a symptom without a model of what upstream
actually does. Our side is measured by driving `jpeg_consume_input` on a
16x16 fixture; upstream's is read from `jdapimin.c`. That probe left no
artifact, and reading upstream is exactly the step that can go wrong — both
are now re-derived on every run instead: `capi_consume_input_states.rs`
(32x32, baseline and progressive) and `capi_preload_resume.rs` (96x96
progressive) compare our trace against a C oracle rather than against this
table. Use those; the table stays as the record of what the divergence was.

| Step | Upstream | Ours (measured) |
| --- | --- | --- |
| after `jpeg_mem_src` | `DSTATE_START` | `DSTATE_START` (200) ✓ |
| 1st `consume_input` | reset input ctl, init source, → `INHEADER`, consume → SOS → `default_decompress_parms` → **`READY` (202)**, returns `REACHED_SOS` | → **`SCANNING` (205)**, returns `REACHED_SOS` |
| 2nd `consume_input` | at READY: returns `REACHED_SOS`, **state unchanged** — "can't advance past first SOS until start_decompress" | returns **`REACHED_EOI`**, state 205 |
| 3rd `consume_input` | still `REACHED_SOS` at `READY`, indefinitely | `REACHED_EOI`, state 205 |

Identical for baseline and progressive, which is itself a divergence: upstream
does not distinguish them here either, but *we* reach EOI on call 2 in both
cases because the whole datastream is already buffered.

So the shape of the fix is not "add a READY state". It is:

1. `jpeg_read_header` becomes the thin wrapper (state guard → `consume_input`
   → map `REACHED_SOS`→`HEADER_OK`, `REACHED_EOI`→abort + `TABLES_ONLY`,
   `SUSPENDED` passthrough), and `consume_input` owns the state machine —
   the inverse of today's call direction.
2. `consume_input` gains upstream's `DSTATE_READY` arm, which returns
   `REACHED_SOS` **without advancing**. That is the repeated-call guard, and
   it is what stops call 2 reporting EOI.
3. Only then can `jpeg_input_complete` return `eoi_seen`, because only then
   does `eoi_seen` become reachable at the same moments upstream reaches it.
4. `jpeg_start_decompress` accepts `READY` and routes
   `buffered_image ? BUFIMAGE : PRELOAD`, which is also what P4-104's
   remaining `DSTATE_BUFIMAGE` note needs.

Steps 1-2 are one change; 3 and 4 follow from it. The three ignored tests in
`capi_input_complete_contract.rs` go green at step 3.

**Step 2 landed 2026-08-11 — partially.** `jpeg_consume_input` now has
upstream's `DSTATE_READY` arm, and the header-parse arm lands on READY instead
of jumping to SCANNING. Re-measuring the table above on the same fixture:

| Step | Upstream | Ours, before | Ours, now |
| --- | --- | --- | --- |
| 1st `consume_input` | → `READY`, `REACHED_SOS` | → `SCANNING`, SOS | → **`READY`, SOS** ✓ |
| 2nd | `READY`, `REACHED_SOS` | `SCANNING`, **EOI** | **`READY`, SOS** ✓ |
| 3rd | `READY`, `REACHED_SOS` | `SCANNING`, EOI | **`READY`, SOS** ✓ |

The READY check had to be hoisted *above* the "header already parsed → body is
buffered → EOI" short-circuit. Adding the match arm alone was not enough — the
short-circuit ran first, so the second poll still reported EOI. That is only
visible by measuring; the arm looked correct in isolation.

**Step 4 landed in part, as a consequence.** The READY guard is only safe if
nothing that can suspend runs while the state is still READY, so
`jpeg_start_decompress` now publishes `DSTATE_PRELOAD` before the non-buffered
body drain — `jdapistd.c:65`, which leaves READY before the absorb loop that
may return FALSE. Without it the guard *introduced* a deadlock: a custom source
reaches SOS, startup suspends mid-body, the application feeds more bytes and
resumes by polling `jpeg_consume_input`, which answers `REACHED_SOS` forever
without reading a byte. Verified red both ways in
`capi_preload_resume.rs` — the state assertion fails 202≠203, and with that
assertion removed all 512 polls still return SOS and never EOI. What step 4
still owes is READY→`DSTATE_BUFIMAGE` for `buffered_image` (that branch keeps
publishing `SCANNING`, the pre-existing divergence noted at the top of this
item).

The same reasoning applies to every entry point that absorbs the datastream,
and review found a second one: `jpeg_read_coefficients`. A `jpegtran`-shaped
caller that reaches SOS by polling `jpeg_consume_input` and then transcodes
finished the coefficient read still at READY, so `jpeg_input_complete` reported
FALSE and further polls repeated `REACHED_SOS` with nothing left to consume. It
now walks upstream's READY → `DSTATE_RDCOEFS` → `DSTATE_STOPPING`
(`jdtrans.c:54-82`) — RDCOEFS published before the drain that can suspend,
STOPPING once the whole stream is in the coefficient buffer, which is the state
`jpeg_finish_decompress` expects. Both transitions are conditioned on arriving
from READY, so callers that enter from `INHEADER` or `SCANNING` are unchanged;
upstream would `ERREXIT` on those, and that strictness is transition work this
item still owes rather than something to fold in here.

A third instance came out of the same review: with PRELOAD now reachable, the
drain's EOI exits promoted it to `SCANNING` along with every other sub-SCANNING
state, so a resume-by-polling caller was told it was scanline-ready before the
`jpeg_start_decompress` retry that runs the startup pass. Upstream returns EOI
from PRELOAD leaving the state alone. The promotion (now
`promote_to_scanning_after_eoi`) exempts PRELOAD, and `jpeg_input_complete`
admits that one state past its sub-SCANNING gate, answering it from
`body_incomplete` as it does every other state. Keying PRELOAD on `eoi_seen`
instead — the first attempt — left the one drain exit that clears
`body_incomplete` *without* reaching EOI, the `P4_13_MAX_BODY_BYTES` cap,
reporting incomplete forever while `jpeg_consume_input` reported
`REACHED_EOI`. The general move to `eoi_seen` still waits on the
`consume_input` restructuring.

**The model is now checked against C, not against a reading of C.**
`examples/consume_input_states_oracle.c` links stock libjpeg and prints the
real `(return code, global_state)` trace for the same four sequences —
baseline polling, progressive polling, suspend/resume through a chunked source,
and the coefficient read. `capi_consume_input_states.rs` and
`capi_preload_resume.rs` compare their own traces against it line for line.
This matters because every other assertion in those files was transcribed from
`jdapimin.c` / `jdapistd.c` / `jdtrans.c` by hand, and a misreading would have
been copied into the implementation and the expectation together, passing. The
oracle refuses to build (rather than skip) when `LIBJPEG_TURBO_PREFIX` says an
install is provisioned. It also refuses to link *this crate's own* `libjpeg`
and compare the shim with itself — but not by any file-layout test, because
`scripts/install_capi.sh` ships the same header set (`jconfig.h` included)
under the same library name as the real thing, so nothing about the file
system tells them apart. The guard is provenance: `is_our_own_shim` reads the
candidate library and rejects it if the bytes contain our crate name, which a
C build cannot. Verified by pointing the mandatory prefix at a directory
assembled to look exactly like a stock install with our own dylib as
`libjpeg.dylib` — it fails loudly instead of comparing the shim with itself.
The `jconfig.h` requirement is only an ABI-header check.

The link is static (`libjpeg.a` by absolute path) wherever the install ships an
archive, which closes a second way to end up self-comparing: a shared link
leaves the final choice to the runtime loader, and `DYLD_LIBRARY_PATH` /
`LD_LIBRARY_PATH` outrank the `-rpath` the helper sets, so an environment
pointing at an installed shim would load it in place of the library whose bytes
were checked. Verified with `otool -L`: the built oracle depends on
`libSystem` alone and names no libjpeg image. In CI the prefix is
the *pinned submodule* built at the v8 ABI, so the C being compared against is
the same source tree these tests' citations quote. Measured C output for the suspend/resume sequence, now pinned:
`create 200 → sos SOS/202 → startup FALSE/203 → drained EOI/203 →
input_complete TRUE/203 → retry TRUE/205`.

**A fourth instance, one layer down.** Reviewing the PRELOAD answer surfaced
that P4-13's 256 MiB buffering cap reports a *resource limit* as ordinary
suspension — `None` from the header drain, `FALSE` from `finish_body_drain`,
`JPEG_SUSPENDED` from the polling drain. All three tell a classic caller
"refill and retry", and no refill can ever satisfy a stream that has already
exceeded the cap; the header-drain site additionally re-accumulated a fresh
256 MiB on every retry. Whether `jpeg_input_complete` should then answer TRUE
(a truncated stream reported as consumed) or FALSE (a polling loop that never
terminates) has no good answer, because the question was wrong: the event is an
error. All three sites now raise `JERR_OUT_OF_MEMORY` with case 100 —
deliberately outside upstream's `jmemmgr.c` range of 1..=10, since the cap is
this shim's own bound and not one of libjpeg's allocation sites.
`capi_suspended_body_cap.rs` covers all three, each verified red on its own:
disabling one raise fails exactly one test or phase and leaves the others
green.

Raising an error is not free of its own hazard: a conforming caller's
`error_exit` ends in `longjmp`, which runs no Rust destructor on the frame it
jumps out of. The header-drain site held the whole 256 MiB accumulator in a
local `Vec` that had already been moved out of `bridge_partial`, so raising
there would have stranded a quarter gigabyte per rejected stream, unreachable
even by `jpeg_destroy_decompress`. It is dropped before the raise. **Any raise
site with a live local allocation has this problem** — the other two sites are
safe only because their bytes live in `priv_state.source`, which `destroy`
owns.

**The generalisation worth keeping:** a state that refuses to consume is only
safe if *every* path that can consume leaves it first. The READY guard is
upstream-faithful in isolation and still introduced two deadlocks, one per
entry point that had no transition, and then mis-reported readiness on a third
path once the new state existed. All three are pinned in
`capi_preload_resume.rs` and each was verified red without its fix.

**What was still partial (superseded 2026-08-14 by the closing status
below):** a *direct* `jpeg_read_header` call still leaves
`DSTATE_INHEADER`. READY is reached only through `consume_input`, so the two
entry points disagree about the post-header state where upstream has one
answer. Making `read_header` land on READY is the remaining half of step 1 —
the call-direction inversion — and it has a wider blast radius, since several
sites still test for `INHEADER`. The three ignored tests stay ignored until
then.

**Acceptance criteria.** Match every upstream state constant and transition
after create/header/start, buffered/raw/coefficient operation, finish, and
abort. Stock-C setjmp/source-manager cases cover repeated/out-of-order header,
incomplete rows, EOI suspension/retry, exactly-once `term_source`, final reset,
and same-handle reuse. P4-13 continues to prove body suspension without
asserting a shim-specific state.

**Progress (2026-08-08) — the constants half is done; transitions remain.**
`cinfo.global_state` is a *public* field, so these numbers are ABI: a consumer
compares them against the values in its own `jpegint.h`. All eleven `DSTATE_*`
values are now mirrored with upstream's numbering, which corrects
`DSTATE_STOPPING` from **206** — upstream's `DSTATE_RAW_OK` — to **210**. A
consumer inspecting `global_state` during `jpeg_finish_decompress` had been
reading "start_decompress done, read_raw_data OK" from a decompressor that was
looking for EOI. The six states the shim did not transition through when this
note was written (`PRELOAD`, `PRESCAN`, `RAW_OK`, `BUFIMAGE`, `BUFPOST`,
`RDCOEFS`) are declared anyway, because their numbering is what makes the rest
correct. `RAW_OK` is published below; `PRELOAD` and `RDCOEFS` followed on
2026-08-11, leaving `PRESCAN`, `BUFIMAGE` and `BUFPOST`. The old value
was only ever assigned, never compared, so no shim logic depended on it.

`jpeg_start_decompress` now also publishes `DSTATE_RAW_OK` when `raw_data_out`
is set, matching `jdapistd.c:170`
(`cinfo->global_state = cinfo->raw_data_out ? DSTATE_RAW_OK : DSTATE_SCANNING`).
A caller that had explicitly opted into raw-data output was previously told the
decompressor was in scanline mode. This applies to the two sites that
correspond to upstream's line 170 — the normal path and the 12-bit deferred
path. It deliberately does **not** apply to the buffered-image early return:
upstream returns from that branch with `DSTATE_BUFIMAGE` (`jdapistd.c:60-63`)
and never reaches line 170, while this shim publishes `SCANNING` there so
`jpeg_input_complete` (gated on `>= DSTATE_SCANNING`) reports TRUE. That
divergence predates this work and belongs to the transitions half, where
`DSTATE_BUFIMAGE` gets wired; routing the site through the raw-data helper
would have published a third value that is neither upstream's nor the intended
one. The same `BUFIMAGE` gap remains for a `buffered_image` request whose body
is already complete, which falls through to the normal path.

Guarded by a unit test that parses `references/libjpeg-turbo/src/jpegint.h` and
compares against the real Rust constants, with exact accounting so mirroring 15
of upstream's 16 constants fails rather than passing quietly. Confirmed red
against the original bug: restoring `DSTATE_STOPPING = 206` fails with
`left: 206, right: 210`. Parsing suffices here, unlike
`tests/capi_classic_error_codes.rs`'s C probe, because these are plain
`#define NAME <int>` lines rather than a positional version-gated enum — and
comparing against the constants themselves removes the transcription step that
produced the 206.

Still open as of that date (all but `DSTATE_BUFIMAGE` closed 2026-08-14; see
the status below): the transition work — `DSTATE_BUFIMAGE` in buffered-image
mode, `DSTATE_READY` after a *direct* `jpeg_read_header` (the shim stays at
`INHEADER`; the consume-driven path and the repeated-call guard landed
2026-08-11), and finish's
unread-row rejection, EOI draining with suspension, exactly-once `term_source`
and abort-reset for reuse, together with the stock-C setjmp harness the criteria
above require.


**Status (2026-08-14): closed.** The remaining root-cause pieces are
delivered, all measured against stock 3.1.4.1 by
`examples/classic_lifecycle_state_oracle.c` /
`tests/capi_classic_lifecycle_state.rs` (13 cases, 17-line trace, verbatim;
red-check: disabling the finish guard flips four rows (d2/d4/d5/d6, measured)):

* A *direct* `jpeg_read_header` lands on `DSTATE_READY` (202), matching the
  consume-driven path (oracle d1). **Delivered as an entry guard plus
  explicit state/flag sites, not as the filing's structural inversion**
  (read_header as a thin wrapper over `consume_input`): the review re-traced
  every divergence originally attributed to the inversion's absence
  (probes u1/u2/p1/s5) and all match stock verbatim, so the inversion is
  now a maintainability argument rather than a correctness one. The
  residual risk that argument rests on, recorded for the session that
  takes it up: upstream maintains `eoi_reached` at 5 sites inside one
  input controller, while this shim mirrors it at 16 write sites across 9
  entry points — every future set-site, upstream's or ours, must be
  mirrored by hand, and only the oracle trace catches a miss.
* `jpeg_finish_decompress` implements upstream's contract
  (`jdapimin.c:404-426`): `JERR_TOO_LITTLE_DATA` on unread rows (d2),
  `JERR_BAD_STATE` with the state as parameter from READY or after a
  completed finish (d3/d5), `term_source` called exactly once — observable
  via a counting source manager (d4) — and the abort-reset to
  `DSTATE_START` (d4/d6). Buffered-image mode takes the BUFIMAGE-equivalent
  arm keyed on `buffered_image` (this shim publishes SCANNING for that
  mode — the P4-13 note; full `DSTATE_BUFIMAGE` publication remains that
  item's residue, not this one's).
* The `jpeg_consume_input` / `jpeg_input_complete` restructure the item's
  filing demanded: `eoi_seen` now means what upstream's
  `inputctl->eoi_reached` means — FALSE after a bare header parse or a
  single-scan startup, TRUE after a multi-scan startup (upstream's PRELOAD
  loop absorbs the whole datastream), post-start consume, a tables-only
  parse (direct and consume-driven), a full-height `jpeg_skip_scanlines`
  (upstream's clamp branch), `jpeg_finish_output`, `jpeg_read_coefficients`,
  or a successful finish; surviving finish and abort; cleared when the next
  datastream's parse begins. The three `#[ignore]`d
  executable-specification tests in `capi_input_complete_contract.rs` are
  enabled and pass, and the two tests that pinned the old shim-only
  promotion (a bare pre-start drain loop terminating) are rewritten to
  upstream's shape — pre-start polls report `JPEG_REACHED_SOS` forever,
  and the terminating idiom starts decompression first.
* The state guards the review round added, each measured against stock
  (probes s3/r4/u2/p3 → oracle rows d7a/d7b/d8/d9): `jpeg_read_header`
  itself is legal only from START/INHEADER (`jdapimin.c:278-281`) — a
  re-read from READY or STOPPING raises `JERR_BAD_STATE(state)`, which is
  also what makes the START-gated `eoi_seen` clear sound; mid-header
  suspension leaves `DSTATE_INHEADER`; `jpeg_finish_output` and
  `jpeg_start_output` refuse non-buffered/non-pass states
  (`jdapistd.c:747-749`, `:771-780`); `jpeg_finish_decompress` suspends
  (returns FALSE, no `term_source`, no cleanup) while a P4-13 body is
  still draining; and `term_source` runs after STOPPING is published,
  with the callback pointer derived from the live borrow (the bridge sites' child-tag pattern), with the d5 oracle row pinning that a
  refused second finish does not call it again.
* **Recorded divergence, not parity:** on a *multi-scan* stream in
  buffered-image mode, stock's `jpeg_finish_output` absorbs to the next
  scan boundary and `eoi_reached` stays FALSE until the last pass (the
  canonical libjpeg.txt loop runs one pass per scan — measured, 10 passes
  on a 10-scan progressive); the eager model reports EOI after the first
  pass, so that loop runs once. Pixels are identical (the decode is
  complete); the pass-walking observability belongs to the buffered-image
  machinery tracked in P4-13/P4-26.

Proof: `capi_classic_lifecycle_state` (trace verbatim vs stock),
`capi_input_complete_contract` 8/8 with zero ignores,
`capi_preload_resume` including the differential `complete` token, and the
full C-ABI suite at 73 suites / 304 / 0.

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

## P4-106. `jpeg_finish_compress` Accepts Incomplete Input and Bad States — **CLOSED 2026-08-14**

**Motivation.** Filed 2026-08-02 alongside P4-100. Finishing partial scanlines
currently encodes zero-filled unwritten rows and returns a valid-looking JPEG.

**Root cause.** Finish checks only private `have_started`, not expected
scanline/raw row counts or legal state, then resets regardless of helper result.

**Acceptance criteria.** Stock-C setjmp cases cover partial scanline/raw input,
bad/double finish, progress passes, helper failure/no usable partial output,
destination termination, final reset, and reuse. `JERR_TOO_LITTLE_DATA` and
other errors flow through P4-100's shared translator.


**Status (2026-08-14): closed by measurement.** The filing's claim — partial
scanlines encode zero-filled rows and return a valid-looking JPEG — no
longer reproduces: `examples/classic_lifecycle_state_oracle.c` c1/c2/c3/c5
show `jpeg_finish_compress` raising `JERR_TOO_LITTLE_DATA` (69) on missing
rows, `JERR_BAD_STATE(100)` without a start and after a completed finish,
and an error-trapped object reusing cleanly after `jpeg_abort` +
destination reinstall — line-for-line identical to stock 3.1.4.1 in
`tests/capi_classic_lifecycle_state.rs`. The row check and state guard were
delivered by the P4-100/P4-154 batches; this closure contributes the
differential proof the acceptance criteria asked for. Not separately
tested: the progress-monitor pass counters during finish's remaining
passes — that observability belongs with the option batch (P4-84..P4-115)
where progress reporting is treated as a whole.

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
1 ignored (`restart_markers_do_not_multiply_decode_cost`, release-only).

One review finding was fixed structurally rather than locally: the destination
callbacks originally re-derived `&mut CompressPrivate` from `cinfo->master`
while the flushing frame still held one, which is two live `&mut` to the same
allocation — undefined behaviour that LLVM's `noalias` would be entitled to
exploit by folding away the very `pending_error` read this fix depends on. The
manager's private state now lives *inside* the manager (`OwnedDestMgr`,
reachable only through `cinfo->dest`), which is both sound and what upstream
does (`my_mem_destination_mgr`, jdatadst.c:43-53). The same re-derivation
pattern exists elsewhere in the shim and is not addressed here.

## P4-109. Classic Source-Manager Setup and Stdio Semantics Diverge — **CLOSED 2026-08-14**

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

**Partial progress (2026-08-10), from the P4-97 review.** `fill_input_buffer`
for the built-in memory source returned `TRUE` while supplying no bytes. Stock's
`fill_mem_input_buffer` (`jdatasrc.c:125-142`) instead warns `JWRN_JPEG_EOF` and
inserts a fake `FF D9`.

The old shape reads as "more data arrived" to every caller. Two concrete
consequences: `jpeg_resync_to_restart`'s scan-forward reported *suspension* on a
source that can never resume, where stock returns `TRUE` with an unread EOI; and
`drain_caller_source_mgr` could spin on a truncated stream, since each iteration
saw `bytes_in_buffer == 0`, no EOI, and a growing-by-nothing accumulator.

Now `default_fill_input_buffer` mirrors stock. This closed none of the criteria
above on its own — validation, FILE buffering and Windows all remained — but it
removed a divergence those cross-validations would otherwise have to encode.
Pinned by `capi_resync_to_restart.rs::scan_forward_past_end_of_memory_source_yields_fake_eoi`.

**Status (2026-08-14): closed.** `LIBJPEG_TURBO_PREFIX=… cargo test -p
libjpeg-turbo-rs-capi --test capi_classic_source_mgr` compares a 10-case,
14-line trace verbatim against `examples/classic_source_mgr_oracle.c` linked
to stock (f4 alone emits five lines — three fills and the two warnings between
them), covering every criterion:

- **null/empty** (m1/m2): `jpeg_mem_src` raises `JERR_INPUT_EMPTY` (43)
  before touching state.
- **foreign-manager replacement** (m3, m5, m6): a manager the same family did
  not install is refused with `JERR_BUFFER_SIZE` (24) — including
  cross-installing stdio over mem and mem over stdio, which stock treats as
  foreign because each setup keys on its own `init_source` identity
  (`jdatasrc.c:270-279` for mem, `230-238` for stdio). Rust fn-pointer
  comparison is not ICF-safe, so the shim keys on `ShimSourceKind` +
  pointer-eq against its own manager box —
  same observable, different mechanism. **Falsification measured**: disabling
  the guard flips m3/m6 to `ok`.
- **reuse** (m4): `jpeg_mem_src` twice on one object decodes twice.
- **pre-read/buffered FILE positions** (f1/f2): `jpeg_stdio_src` now reads
  through the caller's `FILE *` with C-ABI `fread` in 4096-byte chunks — the
  fd-dup/`read_to_end` slurp is gone — so a stream pre-positioned with
  `fseek` (fd offset) or `fgetc` (stdio buffer) decodes from the current
  position, and after a completed decode the position rests strictly before
  EOF when a > read-ahead trailer exists (f1 `pos_class before_eof`, the
  djpeg-concatenated-streams observable; a slurp lands at EOF). One
  intentional difference `pos_class` coarsens away: the shim drains to the
  datastream's EOI at `jpeg_read_header` where stock reads lazily, so the
  *intermediate* position after read-header is further along than stock's
  one-chunk read-ahead; measured resting positions after `finish` are
  identical across single-image, two-image and trailer files. The chunked
  drain stops at the datastream's EOI by marker walk (`find_first_sos` +
  `scan_next_boundary`, safe against embedded-thumbnail EOIs), and bytes read
  past it are retained for handle reuse, as stock keeps them in
  `bytes_in_buffer`.
- **I/O failure** (f3/f4): an empty `FILE` raises `JERR_INPUT_EMPTY` (43)
  when `jpeg_read_header` pulls the first marker, stock's `start_of_file`
  branch; a later dry read warns `JWRN_JPEG_EOF` (123) and fabricates
  `FF D9` (`jdatasrc.c:106-118`, which folds read errors into EOF) — f4
  drives the installed `fill_input_buffer` directly through the ABI and
  pins the whole serve/warn/fake-EOI sequence against stock. The
  drift audit caught the first cut fabricating the EOI *without* the
  warning; `stdio_fill_after_decode_serves_trailing_bytes_first`
  additionally pins (Rust-side) that a fill after a drained decode serves
  the retained post-EOI remainder before touching the stream — the drain
  already read those bytes, so a fresh `fread` would silently skip them.
- **`term_source` and reuse across decodes**: the P4-104 lifecycle trace
  (`capi_classic_lifecycle_state`, d5 `term 1`) pins term through finish, and
  m4 exercises the full cycle twice.

**Windows disposition.** The root cause's "Windows is unavailable" is
resolved by mechanism: the stdio path has no platform gating left (the only
`cfg(unix)`/`cfg(windows)` in the shim is the stderr line writer) and reads
through the caller's `FILE *` via C-ABI `fread`, which is exactly stock's
`jdatasrc.c` reader on every platform — no fd duplication, no Unix-only
syscalls. The stock-trace comparison itself runs where a stock install is
provisioned (macOS local against 3.1.4.1; ubuntu CI against the pinned
submodule build in the "Classic C-ABI state transitions (P4-104)" step); on
the Windows CI leg the suite compiles against the MSVC CRT and self-skips the
comparison for lack of a stock install, the same soft-skip every classic
oracle suite has there.

## P4-110. `jpeg_Create*` Ignores Version and Struct-Size ABI Guards — **CLOSED 2026-08-11**

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

**Status (2026-08-11): closed.** `cargo test -p libjpeg-turbo-rs-capi --test
capi_create_abi_guards` passes, comparing the shim against a real libjpeg
(`examples/create_abi_guards_oracle.c`) across six cases per entry point — ok,
version low/high, size small/large, and both wrong — on every observable the
criteria name: raised code, both `ERREXIT2` parameters, canary integrity past
the declared object, `mem`, `err`/`client_data` preservation, and zeroing.

What the C oracle settled, none of which was safe to assume:

- **The version check wins** when version *and* size are both wrong.
- `JERR_BAD_LIB_VERSION` is **13**, `JERR_BAD_STRUCT_SIZE` **22**; parameter
  order is (library's value, caller's) for both. Independently confirmed by
  `capi_classic_error_codes.rs`, whose gate cross-checks every shim constant
  against upstream's `jerror.h` — it failed until the two new codes were
  registered.
- `mem` is nulled **before** either check, which is what makes
  `jpeg_destroy_*` safe on a rejected object.

That last point found a second defect. `jpeg_destroy_compress` read `master`
straight out of the caller's struct and handed it to `Box::from_raw`; on a
rejected create the struct still holds whatever the caller allocated, so the
new tests turned it into a `SIGSEGV` on the first run. Both destroys now gate
on `mem` through a raw field read (`common_mem_is_null`), never forming a
reference to a mirror the caller may not have allocated.

**The guards are raw-pointer-only, and that is checked rather than argued.**
A caller declaring a smaller struct means the mirror does not fit its
allocation, so even *forming* `&mut JpegDecompressPublic` is undefined — before
any write. The guards therefore read and write `mem` through a field offset,
and `jpeg_destroy_*` tests `mem` the same way. Review found the default error
handler still built a full mirror reference to read `err` at offset 0; it now
reads through the common prefix too.

`rejection_path_touches_only_the_declared_object` allocates **exactly** the
declared size — no canary, nothing past it — and runs create, the error
handler and destroy over it under Miri, where an oversized borrow is a hard
error. It passes. (Its first Miri run failed on a defect in the *test*: the
error manager pointer was derived from the `pub_mgr` field rather than the
enclosing struct, so the callback's write to `fired` landed outside that
pointer's provenance. The C idiom has the same shape and the same trap.)

Review then found a third defect in the fix itself, on the *accepted* path.
Upstream saves `err` and `client_data` into locals, `memset`s, and restores
them. Transcribed literally that *loads* `client_data` as an initialized
pointer — and the standard idiom leaves it indeterminate: stock `djpeg` puts
the struct on the stack and sets only `err` (`djpeg.c:573-589`), which is why
upstream's own comment says the application "may have set client_data" and
warns that "tools like Purify may complain here". C tolerates copying an
indeterminate value; Rust does not. Create now zeroes *around* the two slots
and reads neither. `create_never_loads_an_uninitialized_client_data` leaves the
allocation exactly as `alloc` returned it and sets only `err`; under Miri the
transcribed version fails with *"reading memory at alloc[0x18..0x20], but
memory is uninitialized"* — 0x18 being `client_data`. Every other test in the
file pre-fills its allocation, so none of them could have caught it.

Review found one more door into the same room: the *generic* `jpeg_destroy` /
`jpeg_abort` choose their teardown path by reading `is_decompressor` — which a
rejected create leaves indeterminate, since only `err` (the caller) and `mem`
(the guard) are initialized. Both now test `mem` through the shared prefix
first, at a named `COMMON_MEM_OFFSET` whose equality across the two mirrors is
asserted by a unit test rather than assumed. The Miri case calls
`jpeg_abort` and `jpeg_destroy` as well as the typed destroy; without the guard
it fails with *"constructing invalid value of type &mut JpegDecompressPublic:
encountered a dangling reference (going beyond the bounds of its
allocation)"* — the P0 itself, one function further along.

**One gap, stated rather than papered over:** that Miri run installs its own
`error_exit`, so it does not cover `default_error_exit`, which aborts the
process and therefore cannot be observed under Miri at all.
`default_error_handler_reports_both_parameters_and_aborts` covers it from a
child process for the observable half — the rejection reaches the default
handler and reports *both* `ERREXIT2` parameters, verified red by dropping
`parm1` — but the "stays inside the caller's object" half rests on that handler
reading only offset 0, not on a checker.

**The measured P0, before the fix:** `undersized fired=0 … canary=0
client_kept=0` — a caller declaring even 8 bytes fewer than the real struct had
the full mirror written over the end of its allocation, and its `client_data`
silently dropped. After: `fired=1 code=22 … canary=1 mem_null=1`.

14 test files declared a 4096-byte blob (or `vec![0u8; 1024]`) and passed that
as `structsize` — the normalisation the item predicted. All 58 create call
sites now derive the size from the mirrored struct, and zero remain passing a
literal. Naming the struct also fixes a second latent problem the item did not
mention: `MaybeUninit<[u8; N]>` has alignment 1, so those tests were handing the
C ABI under-aligned `j_decompress_ptr`s — including nine sites review caught
that the first sweep missed, because they spell the buffer `[u8; 4096]` or
`Box<[u8; 4096]>` rather than `MaybeUninit<[u8; N]>`. Every cinfo buffer in the
suite now names its struct, except the one that cannot: `capi_jpeglib_encode.rs`'s
red-zone test needs raw bytes on both sides of the struct, so it over-allocates
by one alignment and offsets to an aligned base instead — and its red-zone
indices carry that offset, which they did not before, so an unaligned
allocation used to be checked against the wrong bytes. **42 error-manager blobs still have that alignment
bug** and are filed separately as P4-148.

## P4-149. Preserved `client_data` May Be Uninitialized While Create Holds `&mut` — **CLOSED 2026-08-13**

**Motivation.** Raised in review while closing P4-110 (2026-08-11) and
deliberately not resolved there; filed as issue #527. `jpeg_Create*` must preserve `client_data`
across its zeroing — that is upstream's contract and one of P4-110's acceptance
criteria — but the standard idiom leaves the slot uninitialized (stock `djpeg`
sets only `err`). Create then holds `&mut JpegDecompressPublic` to write the
remaining defaults, so under a *deep* reading of reference validity the
reference points at an invalid value.

**Why it was not fixed with P4-110.** The two obvious resolutions are both
worse:

- *Null the slot.* Defined, but it is the bug P4-110 fixed — it drops the
  pointer an application attached before calling, which its own callbacks then
  read back.
- *Make create raw-pointer-only.* Correct under any reading, but it converts a
  ~60-field initializer into raw writes inside a change that was already a P0
  memory-safety fix. Reviewing that as one commit is worse for safety, not
  better.

**Status at filing.** Miri accepted the code as it then stood — retagging does
not descend into plain-data fields — and
`create_never_loads_an_uninitialized_client_data` runs exactly this shape under
it in CI. Whether deep validity applies on reference creation is unsettled in
the UCG.

**Acceptance criteria.** Either the UCG settles it permissively and this item
closes with a citation, or `jpeg_CreateCompress` / `jpeg_CreateDecompress` stop
forming `&mut` over the caller's struct entirely and write every field through
raw pointers, with the existing Miri case still passing and no `client_data`
behaviour change.

**Status (2026-08-13): closed** via the second route (issue #527). Both create
functions now write every default through `&raw mut (*p).field` projections;
`cinfo_mut` / `cinfo_compress_mut` are no longer called before the struct is
fully initialized, so no reference ever spans the possibly-uninitialized
`client_data`. The zeroing already went through raw bytes
(`zero_public_struct_preserving_err_and_client_data`), and the P4-110 guard
path was already reference-free, so create is raw-pointer-only end to end.
Field set and values are byte-identical to the `&mut` version — pinned by
`capi_create_abi_guards` (7/7, and 4 passed / 2 process-spawning ignored under
`cargo miri test -p libjpeg-turbo-rs-capi --test capi_create_abi_guards`,
including `create_never_loads_an_uninitialized_client_data`),
`capi_classic_lifecycle` 8/8, `capi_classic_lifecycle_pathological` 4/4, and
`capi_decompress_struct_abi`. No *runtime* observable distinguishes the two
implementations — Miri accepts both, which is how the defect survived — so the
#527 regression is a source gate:
`create_functions_form_no_reference_over_the_callers_struct` extracts both
function bodies and fails on `cinfo_mut` / `cinfo_compress_mut` / `&mut `,
verified to fail on the parent implementation. Entry points *after* create
still form mirror references while `client_data` may remain uninitialized;
that residue belongs to P4-69's module sweep, noted in the helper's doc
comment.

## P4-150. `tj3Compress16` Accepts Lossy 16-bit Where Upstream Refuses — **CLOSED 2026-08-12**

**Motivation.** Found 2026-08-12 (issue #531) while extending P4-145's C
oracle to every compressing entry point. 16-bit samples exist for *lossless* JPEG in TurboJPEG;
a lossy 16-bit compress is refused. Measured with the P4-145 oracle against
TurboJPEG 3:

```
tj3Compress16, quality 80 / 4:4:4, TJPARAM_LOSSLESS unset
  C     -1   (refused)
  ours   0   (encodes)
```

Setting `TJPARAM_LOSSLESS = 1` makes both succeed, which is why the P4-145
trace configures it — that comparison is about buffer ownership, and carrying
this divergence would have made it fail for an unrelated reason.

**Why it matters.** Accepting input upstream rejects is the same class of
defect as P4-39 (CMYK options silently dropped): a caller gets output where the
library it replaced gave a documented error, so the disagreement only surfaces
downstream. Silent acceptance is worse than a wrong error code.

**Acceptance criteria.** `tj3Compress16` (and `tj3Compress12`, if the same rule
applies to it — verify rather than assume) rejects the configurations upstream
rejects, with upstream's error, cross-validated by an oracle case rather than a
transcription. The P4-145 oracle's `compress16_*` lines then no longer need
`TJPARAM_LOSSLESS` to agree, which is the observable proof.

**Root cause.** TurboJPEG imposes no precision rule of its own. It sets
`cinfo->data_precision = 16` (`turbojpeg-mp.c:107`) and lets
`jpeg_start_compress` decide, where `jcmaster.c:199-208` admits 2..=16 for a
lossless compress and only 8 or 12 for a lossy one. Reading `turbojpeg-mp.c`
alone — which is what the original port did — correctly concludes that
TurboJPEG has no such check, and misses that libjpeg does two calls down. This
is the second time a "transcribe the layer the function lives in" reading has
produced a divergence in this family (the first was P4-145's NULL-slot case),
which is why the fix is pinned by an oracle rather than by assertions.

**Status (2026-08-12): closed.** `tj3Compress16` now refuses a compress with
`TJPARAM_LOSSLESS` unset, reporting libjpeg's own `JERR_BAD_PRECISION` text —
`Unsupported JPEG data precision 16`, with no `function():` prefix, because an
error raised inside libjpeg reaches `errStr` through `CATCH_LIBJPEG` verbatim.
The gate is the lossless flag, not `TJPARAM_PRECISION`: that parameter is read
only when the flag is set (`turbojpeg-mp.c:111-115`), so requesting 12 bits does
not make a lossy 16-bit call legal — traced as `c16_lossy_prec12`.

Verified by `crates/libjpeg-turbo-rs-capi/tests/capi_compress_precision.rs`
(5 tests), whose `precision_rules_match_upstream_turbojpeg` compares a
sixteen-case matrix line-for-line against
`examples/compress_precision_oracle.c` linked to real TurboJPEG 3. Before the
fix exactly two lines diverged (`c16_lossy`, `c16_lossy_prec12`), both
`0 kind=none` against C's `-1 kind=precision`.

**Where the refusal sits in the chain.** Review found that a gate placed
naively last gets the *precedence* wrong. Upstream installs the destination
before `jpeg_start_compress` (`turbojpeg-mp.c:118-120`), so a
`TJPARAM_NOREALLOC` slot that cannot be used at all — empty, or present with
zero capacity — is refused by `jdatadst-tj.c:184-192` first, and the buffer
error wins. The rule is narrower than "destination before precision", though: a
slot that is merely *too small* still reports the precision error, because its
capacity is only tested when output overflows it, which never happens once the
compress is refused. Both orderings are traced
(`c16_lossy_norealloc_null` vs `c16_lossy_norealloc_cramped`); with the
ordering check removed exactly the first line flips, so a fix that checked the
destination unconditionally would have been caught too.

Review then found a *third* stage above both. `setCompDefaults` calls
`jpeg_enable_lossless` before `jpeg_mem_dest_tj` (`turbojpeg-mp.c:117-120`), so
an out-of-range point transform beats the buffer error too: with
`PRECISION=13`, `LOSSLESSPT=13`, `NOREALLOC` and an empty slot, TurboJPEG 3
reports the lossless-parameter error. The port's Pt check had to move above the
destination preflight; `c16_pt_ge_prec_norealloc_null`,
`c16_pt_ge_prec_roomy` and `c16_pt_lt_prec_norealloc_null` trace it, and with
that check disabled the first two flip to `buffer`/`other`.

Three groups that disagree with each other, so no single ordering satisfies
them by accident — which is the point. Each was measured, not reasoned: two of
the three orderings contradict the reading a careful person would take from
`turbojpeg-mp.c`.

The trace compares `rc` and an error *kind* for every line, plus the exact
message for the precision refusal — the one string the port owes byte for byte.
Pinning the ordering by raw text would have dragged in TurboJPEG's own
`function():` messages, which carry per-call detail this port words differently
on purpose (the same reason `norealloc_oracle.c` does not compare byte counts).

`tests/precision.rs::tj3_compress16_decompress16_is_lossless` had to change: it
set `TJPARAM_LOSSLESSPSV`/`PT` but never `TJPARAM_LOSSLESS`, and passed only
because 16-bit was unconditionally lossless. Upstream reads `TJPARAM_LOSSLESS`
back as 0 after `PSV` alone and refuses, so the test now sets the flag a real
caller sets.

Steps 2-3 of that chain — `TJPARAM_QUALITY`/`TJPARAM_SUBSAMP` "must be
specified" — remain unpinnable here, because this port's defaults make the
branch unreachable. Filed as P4-155 (#539).

The rule does **not** generalise: `c12_lossy` and `c8_lossy` succeed in both,
since 8 and 12 are the two precisions `jcmaster.c:206` admits — answering the
"verify rather than assume" criterion. `lossy_12bit_and_8bit_compress_still_succeed`
guards against a fix written as "wide samples imply lossless".

The observable proof the criterion asked for: `norealloc_oracle.c` gained
`compress16_lossy_roomy`, a lossy 16-bit case with **no** `TJPARAM_LOSSLESS`
set. It agrees now and would not have before, and it pins the refusal path's
ownership behaviour as well — a caller's buffer must survive a rejected call
untouched.

Run with `LIBJPEG_TURBO_PREFIX=/opt/homebrew/opt/jpeg-turbo cargo test -p
libjpeg-turbo-rs-capi --test capi_compress_precision --test
norealloc_all_entry_points --test precision` (29 passing). The full workspace
release gate is 2597 passing across 294 suites, 0 failures — 2592/293 plus this
item's one new suite of 5.

The classic C API has the same acceptance gap, one layer over — filed
separately as P4-154 (#538), since it needs a different fix and a different
oracle. Closed 2026-08-13.

## P4-151. Legacy `tjTransform` Does Not Bridge `dstSizes` Output-vs-Capacity Semantics — **CLOSED 2026-08-12**

**Motivation.** Split out of P4-145 (2026-08-12, issue #529) after two attempts
at it were rejected in review. Legacy `dstSizes[i]` are **outputs**: a caller that sized
its destinations with `tjTransformBufSize()` may leave them at zero. TJ3 reads
the same slot as an input capacity, so under `TJFLAG_NOREALLOC` such a call now
fails as "buffer too small". Upstream bridges it by filling a temporary array
with each transformed image's worst case (`turbojpeg.c:3118-3132`).

`tjCompress2`'s equivalent bridge *did* land with P4-145: there the geometry is
in the parameters, so `tj3JPEGBufSize(width, height, subsamp)` is a direct
port of upstream's line.

**Why the transform side is harder — both failures are recorded because they
are the acceptance criteria in disguise:**

1. **The capacity must come from geometry alone.** Upstream uses bare
   `tj3JPEGBufSize` on the transformed specs. This port's
   `tj3TransformBufSize` additionally adds the extracted ICC length
   (`transform.rs:428-436`), so using it as the caller's capacity overruns a
   `tjTransformBufSize()`-sized buffer — measured at a 32x32 source with a
   128 KiB ICC profile: an 8192-byte bound against a 139264-byte copy
   (8192 + the 131072 profile bytes).
2. **Deriving the geometry must not mutate the handle.** A plain
   `tj3DecompressHeader` overwrites shared compression state — subsampling,
   colour space, density, ICC. Setting a `TJINIT_TRANSFORM` handle to S420 and
   transforming an S444 source left the handle reporting S444, where upstream's
   wrapper leaves S420, so a later compression silently used different
   settings.

**Current behaviour, and why it is acceptable meanwhile.** The flag *is* mapped
to `TJPARAM_NOREALLOC`, which is the memory-safety half: the caller's
destination buffers are no longer passed to `free()`. Only the convenience half
is missing, and it fails loudly.

**Acceptance criteria.** A legacy `tjTransform` with `TJFLAG_NOREALLOC` and
`dstSizes[i] = 0` succeeds for a `tjTransformBufSize()`-sized destination;
capacities derive from transformed geometry without metadata; the handle's
compression parameters are unchanged across the call (assert one that differs
from the source, e.g. S420 handle over an S444 source); and the P4-145 oracle
gains a legacy-wrapper case so the comparison is against C rather than against
this description.

**Status (2026-08-12): closed.** A legacy `tjTransform` with `TJFLAG_NOREALLOC`
and `dstSizes[i] = 0` now transforms, filling a temporary capacity array from
the transformed geometry and copying the produced sizes back, as upstream does
(`turbojpeg.c:3118-3132`).

Both rejected attempts were the acceptance criteria in disguise, and both
constraints are met by construction rather than by care:

**Capacity from geometry alone.** `transformed_specs` was factored out of
`tj3TransformBufSize` so the two callers share one definition of the transform
rules while differing where they must: that entry point adds the handle's ICC
length to what it returns, and the bridge does not. Using it would hand
`tj3Transform` a capacity larger than the buffer the caller sized — the measured
case is an 8192-byte destination against a 139264-byte capacity at a 32x32
source with a 128 KiB profile.

**No handle mutation.** The bridge reads the source through
`libjpeg_turbo_rs::probe`, which parses the header into its own decoder. A
`tj3DecompressHeader` here would write shared state a later `tj3Compress*`
reads.

`Subsampling::to_tjsamp` now carries the single `TJSAMP_*` mapping, because the
bridge lives in a different crate from the TJ3 parameter accessor that also
needs it and two copies would drift.

**Which parameter proves "no mutation" was measured, not assumed.** The obvious
guess — that `TJPARAM_SUBSAMP` would come back holding the source's value — is
wrong in this port: a header parse leaves it alone. The first version of the
test asserted exactly that and **passed with the rejected approach injected**.
What a parse actually moves, measured on this fixture, is `TJPARAM_JPEGHEIGHT`
(0 -> 32) and `TJPARAM_COLORSPACE` (-1 -> 1). `TJPARAM_JPEGHEIGHT` is the sharp
one: nothing but a header parse sets it, so zero afterwards says the handle was
never used to read the source.

**Two P1s the first version introduced, both found in review.**

*Grayscale sized as 4:4:4.* `probe` reports `Subsampling::Unknown` for a
single-component image — there are no chroma planes to describe — and the
generic mapping turns `Unknown` into `TJSAMP_444`. A legacy caller allocates
`tjBufSize(w, h, TJSAMP_GRAY)`: 4096 bytes at 32x32 against 8192 for 4:4:4.
Handing `tj3Transform` the larger figure as a *capacity* means output between
the two bounds is written past the end of the caller's allocation. Measured with
the gate removed: **5619 bytes reported written into a 4096-byte buffer**. The
direction is what makes it a defect rather than waste — over-stating a bound you
allocate costs memory, over-stating a capacity you trust is an overrun, and this
is the only place the value is used as the latter. `Subsampling::to_tjsamp`'s
doc now says so.

*Slice built from a NULL pointer.* The bridge sliced `jpeg_buf` before
`tj3Transform` validated it, and `from_raw_parts` requires a non-null pointer
even for a zero-length slice — a non-unwinding abort in debug, UB in release,
where the API documents -1. Verified: with the guard removed the test panics on
`unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to
be aligned and non-null`.

**The grayscale test also nearly shipped vacuous**, for the third time in this
item's history. Its first version encoded a plain grayscale source, whose output
is far below both bounds, so it passed with the gate removed. It needed a 5 KiB
ICC profile to push the transformed output into the window between the two
bounds — and its assertion had to become the invariant rather than a success
code, since refusing is a *correct* outcome when the caller's buffer genuinely
cannot hold the result. What must never happen is `rc == 0` with a produced size
past the bound.

**The Red check itself was unsafe, and review caught that too.** Allocating
exactly `gray_bound` meant that whenever the regression returned, the test
*committed* the 5619-byte overflow before its own assertion could run —
corrupting the allocator, or aborting under a sanitizer. It now over-allocates
and canaries the bytes past the bound, so the overrun is **observed** rather
than performed, and the canary is direct evidence independent of the reported
size.

**A divergence surfaced while adding the oracle coverage, and is filed rather
than papered over.** Upstream's grayscale case succeeds within the 4096-byte
grayscale bound — on *this path only*, its capacity pre-read skips marker
registration, so the transform drops the source's 5000-byte ICC profile (with
every other marker) and emits 601 bytes. This port then copied the profile on
every path — which upstream also does on legacy `flags=0` and `tj3Transform`,
both probed identical at 5619 bytes — so here it produced 5619 and refused. The
refusal is *correct for the size produced*; the divergence is confined to the
legacy NOREALLOC ordering quirk. Filed as P4-156 (#544). At this item's
closure the gray line was compared narrowed to the no-overrun invariant —
never report success having written past the bound a compliant caller
allocated — since tracing `rc` would have failed for the divergence rather
than for anything this item governs. P4-156's closure (2026-08-13) replaced
that narrowed line with the shared-fixture `fx_*` family, which compares `rc`
and exact byte size on all three call shapes, and re-pinned the capacity
derivation with the `legacy_norealloc_capacities` unit tests.

Verified by `norealloc_all_entry_points` (18 passing, up from 14) with four new
tests and three new oracle cases, each Red-checked by reintroducing the defect it
targets. The full workspace release gate is 2604 passing across 295 suites, 0
failures, 7 ignored:

- `legacy_tj_transform_fills_a_zero_dst_size_from_geometry` — fails with the
  bridge disabled.
- `legacy_tj_transform_does_not_parse_the_source_into_the_handle` — fails with
  `tj3DecompressHeader` injected, reporting `left: 32, right: 0`.
- The oracle gained `legacy_transform_zero_size`, so criterion 4's comparison is
  against real TurboJPEG rather than against this description. C reports
  `0 1 1`; with the bridge disabled this port reports `-1 1 0`.

## P4-153. Marker-Parse Metadata Copies Are Still Infallible — **CLOSED 2026-08-12**

**Motivation.** Found 2026-08-12 (issue #536) in review while closing P4-144,
and deliberately not folded into it. P4-144 made the metadata *copies* fallible —
the reassembly buffers and every clone into an `Image`. The **originals** are
still built infallibly one layer earlier, in the marker parser:
`read_app1`, `read_app2`, `read_app13`, `read_com` and `peek_marker_data` use
`.to_vec()`, `into_owned()` and infallible vector growth
(`src/decode/marker.rs`).

So a JPEG carrying EXIF, XMP, IPTC, ICC or COM data can still abort the process
during `read_markers`, before any of P4-144's work is reached.

**Why it was not folded into P4-144.** That item names its instances — the
reassembly buffers and the `Image` clones — and the parse stage is a different
layer with a different contract. The copies had a clear answer to "what happens
on refusal?" because they run inside functions already returning `Result`. The
parser needs that question answered per segment, and its existing contract is
that malformed metadata *degrades* rather than failing an otherwise-valid
decode — the same tension P4-144 resolved for Extended XMP by degrading, and
the opposite of what it did for ICC. Deciding that segment by segment is the
work, not the mechanical `try_reserve_exact` conversion.

**Acceptance criteria.** Every metadata copy in `decode/marker.rs` allocates
fallibly. For each segment kind, the refusal behaviour is *chosen and recorded*:
either `JpegError::AllocationFailed`, or degrade-and-continue with the reason
stated — matching P4-144's precedent, where the choice followed the local
contract rather than a blanket rule. A test proves at least one degrade path
and one error path, in the shape of P4-136's
`allocator_refusal_is_an_error_not_an_abort`.

**Status (2026-08-12): closed.** All seven input-sized copies in
`decode/marker.rs` now allocate through `common::try_alloc`, and all three
input-derived *lists* grow through `try_reserve` before pushing —
`icc_chunks`, `xmp_ext_chunks`, and `saved_markers`.

**`saved_markers` was the one that mattered, and review found it.** The first
version guarded the two chunk lists and missed it, while the closure text
claimed the lists were the unbounded exposure — so the item would have closed
with its own stated risk still live. With marker saving on, and TJ3 defaults to
`TJSM_ALL`, every APP/COM segment in a stream lands in that list through seven
`push` sites. It now goes through `Self::push_saved_marker`, which reserves
first and errors as `saved marker list`: `saved_markers` is read back through
the C API, so a silently dropped entry is indistinguishable from a file that
never carried the marker.

**Two smaller defects from the same review.** The `IccChunk` size used
`std::mem::size_of` in a crate that is `#![no_std]` without the `std` feature,
breaking `cargo check --no-default-features` and its CI leg; it is
`core::mem::size_of` now, verified by running that check. And the COM path
made its fallible copy and then *another* infallible one, because
`from_utf8_lossy(&text).into_owned()` copies again for valid UTF-8 — the fix
doubled peak storage for every well-formed comment. `String::from_utf8` takes
the buffer instead, so the common case is one allocation. The invalid-UTF-8
branch still allocates infallibly inside `from_utf8_lossy`, which has no
fallible form; bounded at ~192 KiB (a 64 KiB segment, 3x replacement-character
expansion) and reachable only by a malformed comment.

**The per-segment choice, and the rule behind it.** The decisive fact is that
there is no warning channel at this layer: `exif_data`, `xmp_data`,
`iptc_data`, `icc_chunks` and `comment` are fields the caller reads, so a
silently dropped one is indistinguishable from a file that never carried the
data — the failure mode with nothing to notice, the same class as P4-39 and
P4-150. Degrading is therefore reserved for the case where the caller still
holds the data:

| Segment | On refusal | Why |
| --- | --- | --- |
| EXIF (APP1) | error | caller-visible field; a silent drop is undetectable |
| Standard XMP (APP1) | error | same |
| IPTC (APP13) | error | same |
| COM | error | same; `from_utf8_lossy` copies either way |
| ICC chunk (APP2) | error | worse than the others — reassembly requires sequence numbers 1..=N, so one dropped chunk turns a valid profile into a missing one |
| **Extended XMP chunk (APP1)** | **degrade** | an optional *enlargement* of a packet the caller already holds; dropping it leaves the standard packet, which erroring would discard too. P4-144 made this same call for the reassembly buffers |
| `peek_marker_data` | degrade | returns `Option`, and `None` already means "not readable" — every caller handles it |

**How each is verified, including what is not verifiable.** The degrade path is
covered end to end by
`tests/xmp_iptc_metadata.rs::incomplete_extended_xmp_falls_back_to_the_standard_packet`:
a chunk dropped at parse time leaves exactly the state that test pins — an
extension that cannot be assembled, with the standard packet surviving.

The error path is **not** deterministically reachable, and the criterion's
suggested shape does not fit. Every parse copy is bounded by its segment's
`u16` length — at most 65 533 bytes — so no host refuses one and no JPEG can be
built that makes it; the refusal exists for a memory-constrained allocator,
which is the caller this item is for. A test calling `try_copy_of` with an
unservable length would prove the *helper* works, which
`api::progressive_output::allocator_refusal_is_an_error_not_an_abort` already
does, and would pass unchanged if this file reverted to `.to_vec()` — the drift
it is meant to catch. The first version of this test did exactly that and was
discarded as vacuous.

What can regress is the wiring, so that is what is gated:
`no_metadata_copy_bypasses_the_fallible_allocator` fails if any *statement* in
the production half of `marker.rs` slices `self.data` into a `to_vec()`,
`to_owned()`, `into_owned()` or `from_utf8_lossy` without routing through
`try_alloc`.

Statements, not lines, and that distinction was earned: the first version
scanned lines and false-greened the COM path, because rustfmt had put
`from_utf8_lossy` and its argument on different lines. Review caught it. Three
evasion forms are now Red-checked by planting each in the IPTC site — a
single-line `.to_vec()`, a multi-line `.to_owned()`, and a multi-line
`from_utf8_lossy` — and all three fail the gate. It scans only above the
`#[cfg(test)]` module, since the predicate is itself source.

**Residual limit, stated rather than implied.** A source scan cannot be
exhaustive; an infallible helper called from here would still evade it. Fault
injection would be the exhaustive answer and is not available: forcing a 64 KiB
allocation to fail needs a failing global allocator, which would apply to the
whole test binary. The gate covers the realistic regression, which is someone
reverting one of these statements to the idiom that was there before.

Note the copies were never the unbounded exposure at this layer: at 64 KiB
apiece they are defensive uniformity. The unbounded growth is the *lists* — a
stream can carry arbitrarily many APP/COM segments — which is why missing
`saved_markers` in the first version was the real defect and the copies were
the easy part.

Verified by `cargo test --lib decode::marker` (7 passing) plus the metadata
suites: `xmp_iptc_metadata` (9), `cross_check_metadata` (10),
`cross_check_metadata_edge` (11), `icc_exif_edge_cases` (21), `metadata_write`
(5). The full workspace release gate is 2600 passing across 295 suites, 0
failures — 2599 plus this item's single test, which joined an existing lib
module rather than adding a suite.

## P4-147. `worker_b8_restart_bomb` Asserts a Wall-Clock Bound and Flakes Under Parallel Load — **CLOSED 2026-08-12**

**Motivation.** Filed 2026-08-11 (issue #523) during the P4-104 work.
`tests/worker_b8_restart_bomb.rs::restart_bomb_4096x4096_decodes_within_measured_bound` (renamed to `restart_markers_do_not_multiply_decode_cost` when this closed)
asserts `m.wall_clock.as_millis() < BOMB_WALL_CLOCK_MS`. It failed once during a
full `cargo test --workspace --release` run with a parallel build competing for
CPU, and passes 3/3 in isolation.

**Why it matters.** The default suite runs with parallel test threads, so a
wall-clock assertion under contention is a coin flip — and its failure message
blames a *"RST parsing regression"* that did not happen, which costs a debugging
session before anyone thinks to re-run. `CLAUDE.md` already forbids parallel
benchmarking for exactly this reason; the same reasoning applies to a clock
assertion inside a test.

**What the test is really guarding** is the *complexity* claim: RI=1 restart
parsing must not be quadratic. That is measurable without a clock.

**Acceptance criteria.** The regression it was written for is still caught, by
a deterministic measure — a restart-marker scan counter with an O(n) bound, or
equivalent work-based assertion — and no wall-clock comparison remains in the
default suite. Timing-based checks, if kept at all, move to `experiments/` or a
serial-only harness.

**Status (2026-08-12): closed.** `cargo test --release --test
worker_b8_restart_bomb -- --include-ignored --test-threads=1` passes 5 tests, and
the default parallel run reports the timing test as **ignored**.

**The deterministic half lives in a unit test**, which is what the criterion
above actually asks for. `decode::bitstream::tests::reset_consumes_exactly_the_marker_it_is_positioned_at`
asserts the *mechanism*: on a well-formed stream `BitReader::reset()` consumes
exactly the two marker bytes it is positioned at, scanning nothing, across 64
consecutive intervals. That is what makes a 65 536-restart decode linear, and it
cannot flake — it reads no clock.

It asserts **examined bytes**, not just cursor movement, via a `#[cfg(test)]`
counter in the scan loop. Review found why that distinction matters: a scan
rewritten as `data.windows(2).enumerate().skip(pos)` walks the entire prefix
and *then* advances exactly two bytes, so a cursor-only assertion passes while
65 536 intervals become O(n²). Both regressions were injected and both fail:

```
rescan from position 0   interval 1: reset() consumed 0 bytes, not the 2 …
walk the prefix first    reset() examined 4096 bytes across 64 intervals
```

4096 is 64² — the quadratic signature, in a test that runs in microseconds.

The timing ratio below is now **supplemental evidence**, not the guard. It
measures the consequence end-to-end, which is worth having, but a duration ratio
calibrated on one machine can only ever be evidence — a point review made after
the ratio had already been tightened from 10x to 1.5x, which is exactly the
tension that says timing was the wrong instrument for the primary assertion.

Two further things had to change about that ratio, and the first attempt only
did one of them.

**The assertion.** The absolute bound is gone; the test decodes the *same image
twice* — once with RI=1, once with no restart markers — and asserts the ratio,
which is what actually carries the complexity claim:

- a loaded machine slows both halves together, so the ratio holds where an
  absolute bound flakes;
- an `O(MCUs * RST)` scan multiplies only the RI=1 half, by orders of magnitude.

Each variant is decoded three times, **alternating** between them, and the
minimum of each is taken: contention can only add time, so the fastest run is
the closest available estimate of the work required. Alternating is not
decoration — the first version ran every RI=1 decode and then every control
decode, so a load spike during the first group inflated the numerator alone and
the "ratio cancels load" argument did not hold. Review caught that.

**Where it runs.** The criterion above says *no wall-clock comparison remains in
the default suite*, and a ratio is still a wall-clock comparison. The test is
therefore `#[ignore]`d out of the default run and executed by a named CI step
with `--test-threads=1` — the "serial-only harness" this item's own options
list. That is what makes the parallel-contention failure mode impossible rather
than merely unlikely.

Measured on darwin arm64 release, ten rounds of min-of-three: **min 0.988,
median 0.997, max 1.007**. Restart parsing costs nothing measurable today, and
the minimum-of-three makes the ratio tight enough to bound closely. The bound is
**1.5** — the measured worst case plus ~50%.

An earlier draft used 10x on flake-avoidance grounds. Review pointed out that
this was a guess rather than a measurement and would have accepted a 5x
restart-handling regression — *worse* than the absolute bound it replaced. The
tolerance is now measured reality plus a small margin, as `CLAUDE.md` requires.

Verified live rather than assumed: with the bound temporarily set to 0.5 the
assertion fires with real numbers (`0.98x … 21.402083ms vs 21.796125ms`), so it
compares measurements and not two zeros. A bound of 1.0 does *not* reliably
fail, which is itself the finding — the two decodes are within noise of each
other.

**The fixtures are checked, not assumed.** A ratio near 1.0 proves nothing if
the "control" also carries restart markers, so the control is asserted to have
no DRI marker and zero `FF D0`..`FF D7` sequences, while the bomb must carry
DRI=1 and >60 000 of them. Both decode to identical pixels — restarts are
framing, not content — and both are compared byte-for-byte against stock
`djpeg`, because two Rust-encoded streams checked with the Rust decoder cannot
reveal a shared restart-marker bug.

Ordering matters there too: the measured decodes run **first**. `measure()`
reports a delta against the process high-water mark, so a content decode
performed earlier would fold its transient peak into that mark and leave the
RSS assertion measuring only what the later run adds — the per-RST allocation
guard would stop guarding. Review caught that.

**What this does not cover**, stated because the previous version implied
otherwise: a regression that slows *both* paths equally leaves the ratio
unchanged. General decode performance belongs in `experiments/`, not in a
correctness suite.

## P4-152. Five Absolute Wall-Clock Assertions Remain in the Parallel Default Suite — **CLOSED 2026-08-12**

**Motivation.** Found 2026-08-12 (issue #534) while closing P4-147, which
fixed exactly one of them. The same contention failure mode is still live at:

| File | Line | Bound |
| --- | --- | --- |
| `tests/worker_b8_huffman_bomb.rs` | 213 | `BOMB_WALL_CLOCK_MS` |
| `tests/worker_b8_memory_bounds.rs` | 82 | `SMALL_DECODE_WALL_CLOCK_MS` |
| `tests/worker_b8_memory_bounds.rs` | 99 | `MEDIUM_PROG_DECODE_WALL_CLOCK_MS` |
| `tests/worker_b8_progressive_bomb.rs` | 188 | `LIMITED_DECODE_WALL_CLOCK_MS` |
| `tests/worker_b8_progressive_bomb.rs` | 234 | `UNLIMITED_PARSE_WALL_CLOCK_MS` |

Each asserts an absolute millisecond bound while `cargo test` runs binaries and
threads in parallel. P4-147's failure showed what that costs: a green-on-rerun
failure whose message names a regression that did not happen, which burns a
debugging session before anyone re-runs.

**Why not fixed with P4-147.** That item is scoped to the test that actually
flaked, and the fix was not mechanical — it needed a *control* to compare
against (the same image without restart markers) before a ratio meant anything.
Each of these needs its own answer to "what is the deterministic measure?", and
the answer differs: a Huffman bomb has no natural control, while the
memory-bounds cases may be better served by asserting the bound they actually
care about (peak RSS) and dropping the clock entirely.

**Acceptance criteria.** No absolute wall-clock comparison remains in the
default parallel suite. Each site either (a) becomes a ratio against a control
and moves to the serial CI step P4-147 added, (b) asserts a non-timing
property that carries the same regression, or (c) is deleted with the reason
recorded — a bound nothing can trip is not worth its flake risk.

**Status (2026-08-12): closed.** No absolute wall-clock comparison remains in
the default parallel suite. Each of the five sites got its own answer, and the
answer differed because the *reason* the bound existed differed. Every decision
below rests on a measurement taken for this closure, not on the margins the
original comments claimed.

**Two qualifications, both found by review after a first draft overclaimed
"none remain in any test in this repository".** That grep was too narrow.

A sixth site existed outside the item's table:
`hard_case_x_byte_and_restart.rs::restart_bomb_4096_terminates_within_budget`,
a 60 s bound in the default parallel run. It is now `#[ignore]`d into the same
serial step. It stays an absolute bound rather than becoming a ratio because it
is a *liveness* assertion — its own comment calls it "a DoS bound, not a perf
benchmark" — and there is nothing to compare a hang against. At ~6x its expected
runtime its margin is far tighter than the bounds deleted here, so contention
could plausibly cross it; serial execution is the fix, not a control.

And the wall-clock assertions are not gone from the three files, they are
*demoted*: `worker_b8_measure`'s contract says callers "should skip RSS
assertions but still run wall-clock bounds so the test remains useful" on
platforms that cannot report RSS, which is everything outside Linux and macOS.
Deleting them outright left Windows with no resource bound at all — trading a
flaky assertion for no assertion. They now run only under `!rss_supported()`.
`worker_b8_measure.rs`'s own harness self-test, which asserts a 5 ms sleep takes
under 5 s, is left alone: it tests the clock, not a decode.

**Two demoted to fallbacks, because peak RSS already carries the regression**
(`worker_b8_memory_bounds.rs`, criterion **(b)**). Both assertion helpers
already asserted `peak_rss_delta` — the bound the file exists for, and a
deterministic one. The clock beside it added nothing: its own comment described
500 ms as "~1000x to tolerate contended CI runners" and 2000 ms as "~180x for CI
jitter". A margin that large cannot fire on a regression the RSS bound would
miss; it can only fire on contention, which is precisely the false positive. It
now runs only under `!rss_supported()`, where it is the *only* bound available —
see the qualification above. `wall_clock_is_the_only_bound` makes that policy a
pure function with its own test, because CI runs these binaries on Ubuntu only
and would otherwise never execute the fallback arm at all.

**One demoted, because the bound was 50 000x the measurement**
(`worker_b8_huffman_bomb.rs`, criterion **(b)**). Measured min-of-9 on darwin
arm64 release, stable to three decimals over four rounds: the bomb decodes in
**0.020 ms** against a 1000 ms bound. A ratio against a control was measured and
*rejected*: an ordinary 256x256 decode takes 0.071 ms, so the bomb runs at 0.29x
an ordinary image. It is a pathological Huffman *table*, not a large payload, so
comparing the two would pin the ratio of two unrelated workloads. The memory
assertion — which catches the "2^16-entry lookup per symbol" regression a
pathological table actually produces — stays, as does the requirement that the
decode terminate with a correct-size image or a structured error.

**Two converted to ratios against a control** (`worker_b8_progressive_bomb.rs`,
criterion **(a)**), `#[ignore]`d out of the default parallel run and executed by
the serial CI step P4-147 added, now renamed `Timing ratios, serial (P4-147,
P4-152)`.

`scan_loop_cost_scales_linearly_with_scan_count` quadruples the scan count and
requires the work to roughly quadruple. This is what the deleted
`UNLIMITED_PARSE_WALL_CLOCK_MS` was *for* — its comment named an O(N^2) scan
loop — but a fixed ceiling 1000x above the measurement could only catch a
catastrophic regression while failing on a loaded runner for no reason. Measured
min-of-9 over five rounds: 3.87, 3.91, 3.91, 3.91, 3.87 against a linear
expectation of 4.0; quadratic is ~16. Bound **5.0** — the measured worst case
plus ~28 %, per the tolerance rule. A first draft used 8.0, reasoning that it
sat between linear and quadratic; review pointed out that accepting nearly 8x
work for a 4x input lets a substantial superlinear regression pass, and that the
rule asks for measured reality plus a small margin, not a midpoint.

The correction then failed to land: the edit that changed the constant was in a
script whose *next* substitution raised, so the file was never written, and a
follow-up commit claimed runs "at the tightened bound" that had in fact executed
against 8.0. Review caught that too. The constant is 5.0 now, verified by
reading it back and re-running the ratio five times serially against it. The
2000-to-8000 span is the widest 4x window available, capped by the decoder's own
8192-scan parse limit.

`scan_limit_stops_early_rather_than_walking_every_scan` compares the limited
decode against the *same bomb with no limit*, which is what makes "early"
measurable. Measured 0.278, 0.269, 0.270, 0.269, 0.269 — near the 0.2 the scan
ratio implies, plus fixed header cost. Bound 0.6. The deleted
`LIMITED_DECODE_WALL_CLOCK_MS` asserted a millisecond ceiling that a mitigation
which had **stopped firing entirely** would still have satisfied, since the
unlimited decode of this fixture also finishes in about a millisecond — so the
ratio does not merely reduce flakiness here, it tests something the old bound
could not.

**How the samples are combined, and a reasoning error review caught.** The
first draft measured each workload as a min-of-9 and then took the *minimum
ratio* over three rounds, justified as "noise only ever adds time, so an unlucky
pairing can inflate a ratio but not deflate it". That is wrong. Noise landing on
the **denominator** inflates the denominator and therefore *deflates* the ratio,
so minimising across pairs selects for the most-deflated one — which can hide a
superlinear scan loop or a scan limit that has stopped firing, exactly the
regressions the tests exist to catch. Each workload is now minimised over 27
samples on its own, and the ratio is formed once from the two minima; the
minimum is still right *per workload*, where noise genuinely only adds time.

Running them serially is load-bearing — a ratio cancels machine speed but not
contention, and two workloads timed while other test binaries compete for the
same cores are not comparable to each other either.

Verified by `cargo test --release --test worker_b8_progressive_bomb --
--include-ignored --test-threads=1` — 8 tests, of which the two new ratios were
run four consecutive times at the tightened bound without variation — plus
`worker_b8_memory_bounds` (19, including the fallback check), `worker_b8_huffman_bomb` (5) and
`hard_case_x_byte_and_restart` (5 passing, 1 now ignored in the default run).
The full workspace release gate is 2600 passing across 295 suites, 0 failures,
7 ignored: ignored rises by 3, one previously-passing test becomes one of them,
and one test is added — the fallback check, which drives `check_decode_bounds`
with RSS availability injected. Red-checked by inverting the availability test
and by disabling the fallback assertion; both fail it.


## P4-148. Test Error-Manager Blobs Are Under-Aligned `[u8; N]` Buffers — **CLOSED 2026-08-12**

**Motivation.** Discovered while closing P4-110; filed as issue #526. Across the C-ABI test suite
the error manager is allocated as `MaybeUninit<[u8; ERR_BYTES]>` and cast to
`*mut JpegErrorMgr`. `[u8; N]` has alignment 1, so nothing guarantees the
pointer meets `align_of::<JpegErrorMgr>()`; `jpeg_std_error` then writes a
struct through it. It works today because stack slots happen to be
over-aligned, which is precisely the kind of accident that changes with a
compiler version or target.

**Why not fixed with P4-110.** That change converted the *cinfo* blobs, which
it had to — the new `structsize` guard rejects them. The error blobs are a
separate, pre-existing defect with no forcing function, and folding ~40 more
edits into a P0 fix would have made it harder to review, not safer.

**Acceptance criteria.**

1. No `MaybeUninit<[u8; N]>` in this crate's tests is cast to a libjpeg struct
   pointer; each names the mirrored struct instead.
2. ~~A Miri run over the affected suites passes (Miri rejects misaligned
   references, so it is the mechanism rather than a convention).~~
   **Superseded 2026-08-12 — unachievable as written**, see the status below:
   every affected suite `dlopen`s the cdylib, which Miri cannot execute at all,
   so this criterion could never be met by any amount of work on the defect.
   Replaced by, and closed against, criterion 2′.
2. **(2′, the replacement)** The absence of the idiom is enforced by a
   mechanism rather than a convention: the storage names the mirrored struct, so
   alignment is a compiler guarantee; and a source gate fails on
   reintroduction, proven by reintroducing it. The gate must match the *bug*
   rather than one spelling of it, and must carry a self-check, since a scanner
   with a broken path passes silently forever.

Criterion 2 is rewritten rather than dropped, and rather than left standing
while the item is called closed. The defect itself is fully fixed — there is no
partial code state here — so `PARTIAL` would misreport it and leave an OPEN row
implying work remains. What changed is the verification method, and that change
is recorded loudly enough for the next reader to audit it.

**Status (2026-08-12): closed.** All **43** sites now name the mirrored struct:
`MaybeUninit<JpegErrorMgr>` in place of `MaybeUninit<[u8; ERR_BYTES]>` across
ten suites, and the 33 now-dead `const ERR_BYTES` declarations are gone.

**43, not the 42 the item counted.** `capi_classic_decode_ext.rs` held a
`Box<[u8; 512]>` — the same defect in a different container, and it sat
directly beneath a comment reading *"a `[u8; N]` is align-1, so casting it to a
`j_decompress_ptr` was undefined however large it was … Boxing the mirrored
struct fixes both."* The comment described the fix applied to the `cinfo` on the
line above while the `err` on the line below still had the bug. Being on the
heap made it *likely* to be suitably aligned, which is the kind of accident an
allocator is free to stop providing. The item's stated shape was the search
term, not the boundary.

**Criterion 2 could not be met as written, and this is why** — hence 2′ above.
Every one of the eleven affected suites locates and `dlopen`s the cdylib
through `libloading`, so none of them runs under Miri at all:

```
$ cargo +nightly miri test -p libjpeg-turbo-rs-capi --test arith_code_flag
panicked at tests/arith_code_flag.rs:55:5:
could not locate cdylib near .../nightly-aarch64-apple-darwin/bin/miri
```

That is not a gap this fix can close — Miri has no FFI, and these tests exist
precisely to exercise the shared object a C caller links against. Reporting a
Miri pass here would have meant running something else and calling it this.

What the criterion actually wanted was a *mechanism* rather than a convention.
Naming the mirrored struct is a stronger one than Miri: alignment stops being a
property anything checks at run time and becomes one the compiler guarantees,
because the storage now **is** the struct. Miri could only ever have sampled the
executions it ran. What no compiler prevents is someone reintroducing the
byte-array idiom later, so that risk is carried by a source gate,
`tests/err_mgr_alignment_gate.rs` (2 tests): it fails on `MaybeUninit<[u8;`,
`Box<[u8;` or `as *mut [u8;` anywhere in this crate's tests, naming file and
line. Verified by reintroducing the pattern in `arith_code_flag.rs` — the gate
reports `arith_code_flag.rs:66` and fails.

The gate matches the defect rather than one spelling of it. Review found the
first version pinned literals: `MaybeUninit::<[u8; 512]>::zeroed()` — the more
common turbofish construction — matched nothing, and a `type ErrBlob = [u8;
512];` alias would have laundered any form past a substring scan. Lines are now
normalised (whitespace stripped, turbofish `::<` folded to `<`) and byte-array
aliases are banned outright. Both spellings were re-checked by planting them in
`arith_code_flag.rs`; both are reported.

The gate carries a self-check, because a scanner with a broken path or an
over-eager comment filter passes silently forever: six real spellings of the
bug must be flagged, five legitimate lines must not — including the bare
`let src: [u8; 12]` byte buffers several suites genuinely declare — and the walk
must still reach two named suites P4-148 converted. It skips its own file, which
necessarily contains the shapes as data.

**Residual limit, stated rather than implied.** A determined author can still
defeat any source scan. The gate is not the guarantee; the type is. Its job is
to catch the accidental copy-paste of the old idiom, which is the realistic
regression, and it is why criterion 2′ names the compiler guarantee first.

Verified by `cargo test -p libjpeg-turbo-rs-capi` over the eleven converted
suites (54 passing) plus the gate. The full workspace release gate is 2599
passing across 295 suites, 0 failures — 2597/294 plus this item's one new suite
of 2.

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

**Progress (2026-08-13, P4-14 delivery).** The reported *bit* is no longer
`progressive_mode`: `jpeg_read_header` records upstream's
`(comps_in_scan < num_components) || progressive_mode`
(`jdinput.c:153-156`) and `jpeg_has_multiple_scans` returns it, so the
sequential-multi-scan answer is now correct and oracle-pinned by the
`hms_baseline` / `hms_progressive` / `hms_mss` rows of
`capi_classic_decode_budget.rs`. Still open: the abbreviated and
invalid-state cases, and correctness across consume-input/buffered-image
progression — this item stays OPEN on the state semantics alone.

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

## P4-116. C-Parity Tests Can Convert Failures or Missing Comparisons into a Pass — **CLOSED 2026-08-08**

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

**Progress (2026-08-08, third pass) — triage rather than bulk conversion.** The
forbidden shape CLAUDE.md names explicitly, a *Rust* failure turned into a
skip, is now absent from the suite: a repo-wide search for
`Err(_) => { eprintln!("SKIP…"); return/continue }` finds no remaining
instance. What is left are oracle-side skips, which split two ways and must not
be converted uniformly:

* **defects** — the tool was discovered and its capability probed, so a failure
  is in the request the test built. Making `c_indexedcolortest` assert
  immediately exposed a *parallelism race* the skip had been hiding: `run_djpeg`
  named its temp input from `process::id()` alone, but cargo runs `#[test]`s on
  parallel threads of one process, so concurrent cases deleted each other's
  input mid-invocation. It surfaced as `djpeg: can't open …` and was swallowed.
  The path is now unique per call. `tests/c_indexedcolortest.rs`'s five
  `-colors` sites are now panics; the `-colors` capability is probed once up
  front, so nothing downstream may treat its absence as news.
* **capability gaps** — `cjpeg -precision 12`, `-precision 16 -lossless`, and
  `-arithmetic -progressive` genuinely do not exist on older toolchains. Those
  four sites keep their local skip but now assert `!helpers::is_ci()` first, so
  CI — which provisions libjpeg-turbo 3.x — fails instead of skipping.

**Progress (2026-08-08, fourth pass) — the triage is complete.** Every
failure-shaped `SKIP` in the suite has now been classified. `rgb565_dither`
joined the defect column: `djpeg_supports_rgb565` probes the capability before
the comparison runs, so a failure there is a defect and is now an assertion.
`sof10_encode`'s `-arithmetic -progressive` case and both `restart_bomb`
fixture builds in `hard_case_x_byte_and_restart` joined the capability column
and are CI-guarded.

Nine failure-shaped `SKIP` strings remain across six files, and **all of them
are now capability gaps that fail in CI** — libjpeg-turbo 3.x is provisioned
there, so a missing capability is a provisioning defect rather than a skip.
None of them can any longer turn a real failure into a green run on a
provisioned machine.

The item stays open for its other criteria: `capi_jpeglib_write_coefficients`
and the non-matrix suites still carry no planned-vs-executed count, and the
remaining `SKIP` sites across the wider suite (tool missing, submodule absent,
platform unsupported) have not been individually re-verified as genuinely
environmental.

**Progress (2026-08-08, fifth pass) — the repository-wide sweep is done.**

*Discovery.* Fourteen suites still rolled their own C-tool lookup, and two
could not fail on a provisioned runner at all:
`cross_check_fuzz_decode_diff_c_progressive_16x16` (5 sites) and
`..._baseline_h4v1` (2) had no CI guard whatsoever, so a runner missing `djpeg`
reported `5 passed` having compared nothing. Several lookups scanned only
hard-coded directories, so a `djpeg` on PATH was invisible — the exact
regression that opened this item. `cross_check_transform` derived `cjpeg` from
`djpeg`'s parent directory. The worst was `regression_issue_369_gray_argb_abgr`:
with `djpeg` absent it substituted our own grayscale decode as the "reference",
making the C cross-validation a tautology that could not fail. All now route
through `require_c_tool!`, or through the new `helpers::optional_c_tool` for
cross-checks that are one part of a larger test, where the macro's early
`return` would drop the Rust-side assertions that follow.

*Capabilities.* 47 capability probes across 20 suites answered a missing switch
with a bare `SKIP`/`return` on CI as much as locally. libjpeg-turbo 3.1.4 was
verified on the development host to carry every capability this repository
probes for (`-colors`, `-dither ordered`, `-crop`, `-skip`, `-icc`, `-rgb565`,
`-dct fast|float`, `-lossless`, `-precision`, `-arithmetic`, `-smooth`,
`-copy icc`), so a miss on a provisioned runner is a provisioning defect.
`helpers::skip_missing_c_capability` states that rule once. The single genuine
exception is arithmetic-coded lossless (SOF11): upstream omits it at compile
time even in 3.1.4 (`cjpeg -lossless 1 -arithmetic` answers "Requested feature
was omitted at compile time"), so `sof11.rs` keeps unconditional skips and now
records why, to stop a later sweep converting them.

*A measurement error, corrected.* The second and third passes above reported
their skip inventory from a plain `cargo test`, which captures stderr for
*passing* tests — so it could only ever show skips from failing ones. Measured
correctly with `--nocapture`, 26 skip lines fire on this host, not one.

*The defect that hid behind a skip.* `lossless_encode`'s
`djpeg_supports_lossless` named its probe file from `process::id()` alone.
Cargo runs `#[test]`s as parallel threads of one process and both callers live
in that binary, so the two probes shared one filename and deleted each other's
input mid-`djpeg`. The probe then answered "djpeg does not support SOF3" about
a djpeg that decodes SOF3 fine, and the case skipped. It reproduced only under
full-workspace load, which is why four passes missed it. `lossless_decode` had
the same shape. Both use `helpers::TempFile` now — the fix
`c_indexedcolortest::run_djpeg` already needed. This is the second instance of
that race found by making a skip fail closed.

*A skip that misnamed its own cause.* `subsamp_410` reported "djpeg cannot
decode 4:1:0". It decodes 4:1:0 fine; `cjpeg -sample 4x2 | djpeg` round-trips
on the same binary. `make_jpeg_with_410_sampling` patches the SOF sampling
factors to 4x2 over entropy data coded for 2x2, so djpeg correctly rejects the
result — which is the C behaviour the case measures our leniency against. It is
an explained expected refusal now, deliberately *not* a CI-fatal capability
assertion.

*Counts.* `precision_arbitrary` gains a `ComparisonTally` — 30 planned across
two 15-precision legs, reporting `30 comparisons completed out of 30 planned`;
its sub-byte-precision drop is a named exclusion rather than a silent
`continue`, and its two capability `return`s record exclusions before finishing
instead of discarding the first leg's work.
`capi_jpeglib_write_coefficients` is deliberately left without a tally, and
this is a finding rather than an omission: it is ten independent `#[test]`
scenarios with zero `SKIP`s, zero early `return`s and zero `continue`s, so
every one always reaches its assertions. A tally there would plan one case per
test and duplicate what cargo's own `10 passed` already states. The same holds
for the other non-matrix suites the earlier pass named.

*Every remaining skip, classified.* 26 fire on this host, none able to hide a
failure on a provisioned runner: 3 permanent upstream gaps (arithmetic lossless
SOF11), 2 deliberate guard-test outputs in `helpers_smoke`, 12
reference-leg-only skips that still run and pass their shim-side assertions
(`LIBJPEG_TURBO_REFERENCE_DIR` unset), and 9 environmental — platform
(`/dev/full` absent on macOS, x86_64 dispatch on aarch64), the opt-in licensed
ITU-T T.83 corpus, `exiftool`, `djpeg12`, a mozjpeg-bound `libvips`, an
`ffmpeg` built without libjpeg, and the documented Pillow v6b case.

Five guard tests pin the two new helpers, including two `should_panic`
intentional reds for their CI branches.

**Status (2026-08-08): closed.** `cargo test --workspace --no-fail-fast` on
macOS aarch64 → 2471 passed, 0 failed, 1 ignored
(`restart_bomb_4096x4096_decodes_within_measured_bound`, release-only), run
twice consecutively to confirm the parallelism race is gone. `cargo fmt
--check` and `cargo clippy --workspace --all-targets -- -D warnings` with the
three CI-allowed test-code lints are clean. GitHub
[#435](https://github.com/developer0hye/libjpeg-turbo-rs/issues/435).

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

## P4-120. Classic-Shim Allocation-Failure Paths Are Unreachable From Tests — **CLOSED 2026-08-13**

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

**Status (2026-08-13): closed** (issue #467). `libc_malloc` — the funnel every
classic dest-manager allocation passes through — carries a thread-local
failure countdown armed only by `fail_nth_allocation_for_tests` (not
`extern "C"`, not exported from the cdylib; one thread-local read on the
production path). `capi_alloc_failure_injection.rs` forces both
previously-unreachable `jpeg_mem_dest` OOM paths — the empty-slot initial
allocation (`jdatadst.c:271`) and the doubling growth (`:132`) — and asserts
code 56 **with** `msg_parm.i[0] == 10`, the `ERREXIT1` payload this item
called unproven; the growth path's `pending_error` was widened to carry the
parm through the deferred flush. A disarmed-hook control test pins that the
injection, not the sequence, causes the failures. No C oracle exists for
these lines (stock's `malloc` cannot be portably failed on demand); the
contract constants are read from `jdatadst.c` and the rendering is covered
by the P4-146 whole-table gate.

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

## P4-122. The Pillow Smoke Harness Performs the v6b Substitution Its Own Policy Forbids — **OPEN**

**Motivation.** Filed 2026-08-08 from the P4-81 CI run (PR #447). Declaring GNU
symbol versions made the Pillow leg fail with:

```
version `LIBJPEG_6.2' not found (required by .../PIL/_imaging...so)
```

**Root cause.** `examples/pillow_smoke/run.sh` symlinks the shim as
`libjpeg.so.62` (line 62) and overwrites Pillow's bundled
`libjpeg-*.so.62.*` with it (lines 150-163). The `_imaging.so` in that wheel is
built against **v6b** and requests `jpeg_*@LIBJPEG_6.2`. Until P4-81 the shim
declared no version nodes at all, so glibc's unversioned-fallback path bound a
v6b consumer to our **v8** struct layout and the harness reported success.

This contradicts the policy the same evidence chain states. `docs/LAST_MILE.md`
says the Pillow runner "rebuilds a v6b wheel against a discoverable v8 SDK" and
that "Direct v6b substitution is forbidden because T4 is a non-goal", and T4
(`libjpeg.so.62`) is an explicit non-goal. The CI leg does the forbidden thing.

**Why it matters.** P0-3 and the `capi_pillow_compat` row in the live-gate
table are cited as T3 downstream evidence. If the binding under test was
v6b-consumer-to-v8-library, that evidence is weaker than documented — it
demonstrated that a mismatch *loads*, not that the ABI matches. P4-81's version
nodes turn the mismatch from silent UB into a clean load-time refusal, which is
why the failure surfaced now rather than being introduced now.

**Acceptance criteria.**

1. The Linux Pillow leg obtains a Pillow whose `_imaging.so` links a **v8**
   libjpeg — the documented rebuild path — and never overwrites a bundled
   `*.so.62.*` in place.
2. Adding a `LIBJPEG_6.2` node to satisfy the old wheel is explicitly rejected:
   it would assert a v6b ABI this project does not implement and restore the
   struct-layout mismatch.
3. The harness fails closed when it cannot obtain a v8-ABI Pillow, rather than
   falling back to substitution.
4. `docs/LAST_MILE.md`'s `capi_pillow_compat` row is re-measured and re-worded
   to describe what the leg actually proves.
5. P4-81's version nodes stay in place while this is fixed; CI going green
   again by weakening them is not an acceptable resolution. (Written when the
   nodes were expected on the cargo cdylib. They now land on the *installed*
   library `scripts/install_capi.sh` relinks, because rustc's own anonymous
   version script makes the cdylib incapable of carrying them. The substance is
   unchanged: the harness must stop substituting for a v6b library, not lose
   the nodes.)

**Why deferred.** The fix touches the project's headline downstream-compat
evidence, so it needs its own review rather than being folded into P4-81.

## P4-123. Architecture Umbrella: Codec Plans, C-ABI State, Public Boundaries, SIMD Dispatch — **OPEN**

**Motivation.** Filed 2026-08-08 to give GitHub
[#442](https://github.com/developer0hye/libjpeg-turbo-rs/issues/442) a
LAST_MILE home. #438 and #441 split the encoder and decoder monoliths into
stable facades plus private responsibility modules, so the remaining
architectural risk is semantic rather than physical: the Rust free functions,
`Encoder`/`Decoder`, TJ3, and the classic `jpeg_*` ABI still normalise options,
state, and errors through different paths. That is the shape that produced the
option-drop family (P4-39/P4-40) and the classic option/state gaps
(P4-84..P4-115).

This is an **umbrella**, structured like [P4-55](#p4-55-zune-jpeg-competitive-gap-program-350361--open):
per-defect behaviour stays with the individual entries, and the detail of the
workstreams lives in #442. It exists here so the next session can find the
programme at all — #442 had no LAST_MILE entry, which by this repository's own
rule means it did not exist for anyone reading the index.

**Workstreams (detail in #442).** (1) classic C-ABI context, lifecycle and
error boundary — executed alongside P4-100/P4-104/P4-106 and then used by
P4-84..P4-115; (2) canonical `EncodeRequest`/`DecodeRequest` →
`EncodePlan`/`DecodePlan` models with explicit mode enums instead of
interacting booleans; (3) public API and dependency boundaries; (4) SIMD
dispatch containment — ~129 direct architecture-specific references still sit
outside the SIMD layer; (5) one parser/limits/geometry/output model across
8/12/16-bit.

**Progress (2026-08-08) — workstream 3, the spec-data dependency inversion.**
#442's evidence list names two places where lower-level modules reach *upward*
into `encode`. Both were the same cause: the JPEG Annex K specification
tables lived in `encode::tables`, although they are direction-neutral data.
`common::huffman_table::std_huffman_tables()` imported the four standard
Huffman tables from `encode` to serve the *decoder*, and five `simd/*` sites
imported `ZIGZAG_ORDER` from `encode` to serve quantisation kernels on both
sides. They now live in `common::tables`, and `src/common/` has zero
`crate::encode` references. `encode::tables` re-exports every moved name
because it is a public path, so this is not a breaking change; the encoder
*policy* (`quality_scale_quant_table{,_ext}`) stays there, and the three tests
that validate the Annex K data moved with the data. The `simd → encode`
references that remain are scalar *kernel* fallbacks for encode-direction SIMD,
which is the correct direction, not spec data.

**Acceptance criteria.** Each workstream lands incrementally under the existing
C-parity and golden matrices — explicitly not a big-bang rewrite — with the
public facades staying compatible. The umbrella closes when #442's five
workstreams are done or individually re-filed; it does not close by closing any
single defect it coordinates.

## P4-124. The OpenCV Harness Tests the Cargo cdylib, Not the Library We Ship — **OPEN**

**Motivation.** Found 2026-08-08 while checking whether P4-81's remaining
acceptance criterion — "the OpenCV replacement harness emits no
`no version information available` warning" — was reachable. It is not, and the
reason is not the lab: the harness does not test the artifact P4-81 fixed.

**Root cause.** `examples/opencv_smoke/container_run.sh:35` stages the raw
Cargo output directly:

```sh
ln -sf /input/liblibjpeg_turbo_rs_capi.so /tmp/libjpeg-rs/libjpeg.so.8
```

and `run.sh` takes that path from the caller as `--lib <release-cdylib>`. So the
library OpenCV binds to in this harness is the cdylib, while the library
`scripts/install_capi.sh` stages — and therefore the one a distro or a packager
actually installs — is a *different binary*: since P4-81 it is relinked from the
`staticlib` so it can carry `LIBJPEG_8.0` / `LIBJPEGTURBO_8.0`, which the cdylib
provably cannot (rustc's own anonymous version script; see P4-81).

Two consequences, and the second is the one that matters:

1. P4-81's criterion cannot pass as written. The harness will keep reporting the
   loader warning no matter how correct the shipped library is, because it is
   not looking at it.
2. **The project's headline T3 downstream evidence has been measuring an
   artifact that is not the shipped one.** That was harmless while the two
   binaries were byte-identical in every respect a consumer sees. It is not
   harmless now, and it was never *verified* to be harmless — nothing asserted
   the equivalence.

This is a P4-116-shaped defect one level out from the test suite: not a test
that fails to run, but a test that runs against the wrong subject.

**Acceptance criteria.** The harness stages through `scripts/install_capi.sh`
(or is given the staged tree) so the library under test is the one that ships;
`container_run.sh` asserts the absence of `no version information available`
rather than leaving it to a human reading the log; and the equivalence the old
arrangement assumed is either asserted or abandoned. P4-81's OpenCV criterion is
re-evaluated only after this lands — until then a green OpenCV run says nothing
about symbol versions either way.

**Why filed rather than fixed.** The harness is Docker-based and this host could
not start Docker, so a change to it cannot be verified here; and its result is
the project's primary T3 claim, so it should not be edited blind. Wiring it into
CI (where Docker is available) is the natural vehicle, which makes it P2-G's
neighbour rather than a drive-by fix.

## P4-125. TurboJPEG YUV decompress entry points emit one plane per SOF component, overrunning the 3-plane ABI contract — **CLOSED 2026-08-08**

**Motivation.** Filed 2026-08-08 from a scoped security scan of `src/` and
`crates/` (report `CLAUDE-SECURITY-20260807-214723`, findings F1 and F2, both
HIGH). Both TurboJPEG decompress-to-YUV entry points sized their output from the
attacker-supplied JPEG's SOF component count while the TurboJPEG YUV ABI models
only 1 (grayscale) or 3 (Y/Cb/Cr) planes, so a 4-component Adobe CMYK/YCCK frame
wrote a whole extra plane past the caller's allocation:

- `tj3DecompressToYUV8` (`crates/libjpeg-turbo-rs-capi/src/yuv.rs`) packed all
  four planes and `copy_nonoverlapping`'d them into a destination the caller
  sized with `tj3YUVBufSize`, which never reports more than 3 planes
  (`crates/libjpeg-turbo-rs-capi/src/bufsize.rs`). Heap overflow of roughly
  `PAD(width) * PAD(height)` attacker-influenced sample bytes (CWE-787).
- `tj3DecompressToYUVPlanes8` looped over all four planes against the caller's
  documented 3-entry `dstPlanes`/`strides` arrays, so iteration `i == 3` read
  `*dst_planes.add(3)` past the array and used the result as a write
  destination. The per-plane NULL check only rejects an exactly-zero word, so a
  non-NULL adjacent value becomes an arbitrary-address heap write (CWE-125 into
  CWE-787).

**Root cause.** A missing upstream guard, not a novel defect. Upstream
libjpeg-turbo rejects these frames in `tj3DecompressToYUVPlanes8`
(`references/libjpeg-turbo/src/turbojpeg.c:2229-2230`):

```c
  if (dinfo->num_components > 3)
    THROW("JPEG image must have 3 or fewer components");
```

Upstream's `tj3DecompressToYUV8` (`turbojpeg.c:2383`) builds a 3-entry
`dstPlanes[3]`/`strides[3]` and inherits that guard by *delegating* to
`tj3DecompressToYUVPlanes8`, so upstream needs only one guard site. **This port
does not delegate** — both entry points call `decompress_to_yuv_planes`
independently — so the single upstream guard corresponds to two guard sites
here, and porting it to one entry point alone leaves the other live.

The subsampling check cannot substitute for it: upstream `getSubsamp()`
(`turbojpeg.c:431-511`, esp. 446-449) deliberately returns a valid `TJSAMP_*`
for 4-component CMYK/YCCK frames, and this port's `detect_subsampling`
(`src/api/yuv.rs`) behaves the same way — an all-1x1 CMYK frame classifies as
4:4:4. Historically the guard entered upstream in commit `cd7c3e66` ("Add CMYK
support to the TurboJPEG C API", 2013-08-23), the same commit that taught
`getSubsamp` about 4-component frames; the port picked up the CMYK-aware
subsampling detection without its paired guard.

**Acceptance criteria.**

1. Both C-ABI entry points reject frames with more than 3 components before any
   write, using the crate's `inst.set_error(..., TJERR_FATAL)` + `-1` idiom —
   no panic, and no clamping that writes fewer planes while reporting success.
2. Grayscale (1-plane) and 3-component frames are unaffected.
3. Regression coverage for both entry points, each failing at the pre-fix
   revision and passing after.
4. The guard stays in the C-ABI crate: `decompress_to_yuv_planes`
   (`src/api/yuv.rs`) is a public Rust API with an unrelated third caller
   (`decompress_to_yuv`), so guarding the shared helper would change behaviour
   outside this gap.

**Status (2026-08-08): closed.** `crates/libjpeg-turbo-rs-capi/src/yuv.rs`
carries `MAX_YUV_PLANES = 3` and rejects `planes.len() > MAX_YUV_PLANES` at both
entry points, ahead of `pack_yuv_planes` and ahead of the `unsafe` plane loop
respectively. Pinned by `crates/libjpeg-turbo-rs-capi/tests/yuv_four_component_guard.rs`
(both entry points) and `crates/libjpeg-turbo-rs-capi/tests/yuv_decompress_planes_component_guard.rs`
(a canary in `dstPlanes[3]` that observes the pre-fix out-of-bounds write
directly). Both tests were measured red at `e63c46d` — the packed sink wrote
256 bytes past the 768-byte `tj3YUVBufSize` region on the 16x16 CMYK frame and
still returned 0, and the planar sink wrote a full plane through `dstPlanes[3]`
— and green with the guard. C cross-validation is
`crates/libjpeg-turbo-rs-capi/tests/yuv_four_component_c_parity.rs`, which drives
`examples/yuv_component_count_c_oracle.c` against stock libjpeg-turbo 3.1.4.1 and
requires both entry points to match C's accept/reject decision (C returns -1/-1
for the CMYK frame and 0/0 for a 3-component control). The Integration Tests job
runs all three binaries with `LIBJPEG_TURBO_PREFIX=/opt/libjpeg-turbo`, which
makes the oracle fatal rather than skippable there. Reachable plane
counts are `{1, 3, 4}` because `detect_subsampling` already rejects 2, so the
only newly-rejected input is the 4-component frame itself. The second-layer gap
this work surfaced is tracked as P4-126, and the validation-ordering gap the same
review surfaced is P4-127.

**Follow-up (2026-08-09): canonical mapping rows corrected.** Closing this item
made the C entry points diverge from the root-crate functions they are mapped to,
but `docs/C_API_REFERENCE.md` (Decompression to YUV) and
`docs/FEATURE_PARITY.md` (JPEG → YUV) still presented
`tj3DecompressToYUV8` ≡ `yuv::decompress_to_yuv()` and
`tj3DecompressToYUVPlanes8` ≡ `yuv::decompress_to_yuv_planes()` without
qualification. The C entry points now reject 4-component CMYK/YCCK frames while
the Rust functions still return one plane per SOF component — the divergence this
patch recorded in the `src/api/yuv.rs` doc comments but not in the two canonical
mapping documents. Both rows now state it. The `✅` status is unchanged and
correct: the exported C functions are complete and match C, which is what this
item established; it is the Rust-native equivalents that differ.

## P4-126. `yuv_plane_width`/`yuv_plane_height` accept any component index where C rejects `componentID >= nc` — **CLOSED 2026-08-09**

**Status (2026-08-09): closed.** The root-crate half landed with issue #466.
`yuv_plane_width` / `yuv_plane_height` now return 0 — C's documented
invalid-argument return for `tj3YUVPlaneWidth` — for any component index at or
above `YUV_PLANE_COUNT` (3). The grayscale term stays in the C-ABI layer, which
still has the raw `subsamp`; `Subsampling` has no grayscale variant and so
cannot carry it, and that constraint is now stated on the constant rather than
left for the next reader to rediscover.

**Criterion 4, decided: keep four planes and size the fourth correctly.** The
alternative — rejecting CMYK/YCCK in the public Rust API to match the C ABI —
was refused because it would remove working functionality from Rust callers to
paper over a sizing bug. And it *was* a sizing bug, not merely an unchecked
index: `decompress_to_yuv_planes` sized the K plane through the chroma rule, so
a 4:2:0 CMYK frame came back with a K plane of **1024 bytes where the channel
holds 4096** — three quarters of it silently discarded. In CMYK and YCCK the
fourth component carries full resolution, like luma, and it is now sized that
way explicitly.

That measurement is the correction P4-126 needed: this section originally
called the gap "defence in depth plus an API-contract divergence", which
understated it. No C caller could reach it — the C ABI rejects 4-component
frames at the entry points (P4-125) — but every Rust caller decoding a
subsampled CMYK frame to planes was losing data.

Pinned by `tests/regression_issue_466_cmyk_plane_sizing.rs`, red at `97421a1`
on both counts. `cargo test --workspace`: 2487 passed / 0 failed, including the
four-component fixtures from the zune-image corpus.


**Status (2026-08-09): partial.** The C-visible half is closed. `tjPlaneWidth` /
`tjPlaneHeight` now delegate to `tj3YUVPlaneWidth` / `tj3YUVPlaneHeight` and map
0 to -1, the way upstream does, so the `componentID >= nc` bound reaches them
instead of being re-derived from a root-crate helper that cannot express it; and
the grayscale branch of `tj3YUVPlaneWidth` / `tj3YUVPlaneHeight` now fills the
no-handle error slot upstream's `THROWG("Invalid argument", 0)` fills. Pinned by
`crates/libjpeg-turbo-rs-capi/tests/yuv_plane_index_c_parity.rs`, which drives
`examples/yuv_plane_index_c_oracle.c` over every (`TJSAMP_*`, `componentID` in
-1..=4) cell — 42 cells, all required to match stock libjpeg-turbo 3.1.4.1.
Measured red at `bad5493`: 21 of the 42 disagreed.

**Criterion 1 is only half-implementable as written, and this is why.**
`Subsampling` (`src/common/types.rs:32`) has **no grayscale variant** — the C-ABI
`subsamp_from_c` maps `TJSAMP_GRAY` to `S444` — so the root-crate
`yuv_plane_width(component, width, subsampling)` cannot compute
`nc = (subsamp == TJSAMP_GRAY ? 1 : 3)`; grayscale is simply not representable in
the argument it receives. The bound was therefore placed where the information
exists (the C-ABI layer, which still has the raw `subsamp` and `is_gray`) rather
than pushed into a type that cannot carry it. Closing the remaining half needs a
decision recorded here first: add a `Gray` variant to `Subsampling` (touches every
match in the codebase), or change the helpers' signature.

**Remaining.** (a) the root-crate helpers still accept any component index —
unreachable from C now that both wrapper pairs bound it, so this is defence in
depth plus a Rust-API contract gap; (b) criterion 4's decision for
`decompress_to_yuv_planes` on a 4-component frame is still open, and note that a
naive "return 0 for an out-of-range index" would make its trim loop emit an
**empty** fourth plane rather than reject — a worse failure than today's
mis-sized one. Criteria 2 and 5 are done.


**Motivation.** Filed 2026-08-08 while closing P4-125, which ported the
first-layer defence but left upstream's second layer unported in the root-crate
plane-size helpers. Upstream defends the YUV plane model twice: `tj3YUVBufSize`
(`references/libjpeg-turbo/src/turbojpeg.c:1029`, line 1038) fixes
`nc = (subsamp == TJSAMP_GRAY ? 1 : 3)`, and `tj3YUVPlaneWidth` /
`tj3YUVPlaneHeight` (`turbojpeg.c:1115`, lines 1124-1125) additionally reject an
out-of-range component with `THROWG("Invalid argument", 0)`:

```c
  nc = (subsamp == TJSAMP_GRAY ? 1 : 3);
  if (componentID < 0 || componentID >= nc)
    THROWG("Invalid argument", 0);
```

**Root cause hypothesis.** `src/common/bufsize.rs` treats every
`component != 0` as chroma with no upper bound, so `yuv_plane_width(3, ..)`
silently returns a chroma-sized plane instead of signalling an invalid argument.
The C-ABI wrappers `tj3YUVPlaneWidth` / `tj3YUVPlaneHeight`
(`crates/libjpeg-turbo-rs-capi/src/bufsize.rs`) do not inherit it — they already
bound `componentID` to `0..=2` against their own `plane_width` / `plane_height`,
so the gap is confined to the root-crate helpers, except for grayscale: there the
wrappers return 0 for components 1-2 without setting the no-handle error slot
upstream's `THROWG("Invalid argument", 0)` fills.

**Acceptance criteria.**

1. `yuv_plane_width` / `yuv_plane_height` reject a component index at or above
   the subsampling's plane count (1 for grayscale, otherwise 3), and the C-ABI
   wrappers report the grayscale case the way upstream does — return 0 *and*
   set the process-global no-handle error slot, since `tj3YUVPlaneWidth` takes no
   handle and raises `THROWG("Invalid argument", 0)`.
2. A cross-check against C `tj3YUVPlaneWidth` / `tj3YUVPlaneHeight` for
   `componentID` in `-1..=4` across every `TJSAMP_*`, including grayscale, where
   only component 0 is valid.
3. Callers inside the crate keep their current behaviour for valid indices; the
   change must not alter any packed or planar buffer size.
4. Decide, and record, what `decompress_to_yuv_planes` does for a 4-component
   frame once the helpers reject index 3. It calls them with 3 today, so
   criterion 1 forces a choice: trim the fourth plane by some other rule, or
   reject CMYK/YCCK in the public Rust API too. The two options differ in
   whether `decompress_to_yuv` keeps returning a 4-plane packed buffer.
5. While here, correct the mapping rows for these two functions in
   `docs/C_API_REFERENCE.md` (lines 72-73) and `docs/FEATURE_PARITY.md`
   (lines 338-339): they name the root-crate `yuv_plane_width()` /
   `yuv_plane_height()` as the equivalents, but the exported `tj3YUVPlaneWidth`
   / `tj3YUVPlaneHeight` run the capi-local `plane_width` / `plane_height`
   instead. Pre-existing inaccuracy, surfaced by this item's audit; the ✅
   status itself is right, because the exported functions do match C's return
   values.

**Why deferred.** Not a memory-safety issue on its own. `decompress_to_yuv_planes`
does still call `yuv_plane_width(3, ..)` / `yuv_plane_height(3, ..)` for every
4-component frame — P4-125's guard sits in the C-ABI layer and runs *after* that
helper returns — but the value only sizes the fourth plane's own trim, and the
guard then discards that plane, so nothing is written out of bounds. No C caller
reaches the permissive path either, because the wrappers already bound
`componentID`. It is defence in depth plus an API-contract divergence, so it was
deliberately kept out of the P4-125 patch to keep that change minimal and
reviewable.

## P4-127. C-ABI YUV Decompress Entry Points Validate After Decoding, Not Before — **CLOSED 2026-08-09**

**Status (2026-08-09): closed.** `TjHandle::inspect_header`
(`src/api/tj3.rs`) reads the frame from its markers alone —
`Decoder::new_with_limits` stops after marker parsing — applies the handle's
`DecodeLimits` including `check_frame`, and returns a `FrameInfo`
(width/height/num_components/subsampling). Both C-ABI entry points now call it
before `decompress_to_yuv_planes`, which resolves all three consequences:

1. `TJPARAM_MAXPIXELS` bounds them, enforced at header time as upstream does
   (`turbojpeg.c:2219-2222`) rather than left to a decode that may never run.
2. `align` is validated at function entry (`turbojpeg.c:2395-2397`), so it
   outranks the component guard again, matching C's precedence.
3. `dstPlanes[0..n]` are NULL-checked up front (`turbojpeg.c:2226-2227`), so a
   rejected call leaves every caller buffer untouched instead of writing planes
   0 and 1 before noticing a NULL plane 2.

**How "did not decode" is proven without timing.** The regressions in
`crates/libjpeg-turbo-rs-capi/tests/yuv_validate_before_decode.rs` use fixtures
with a **valid header and a truncated entropy segment**: decode-first reports
`"unexpected end of data"`, header-first reports the specific rejection, so the
message discriminates the two orders deterministically. All four were measured
red at `2304bf2` with exactly that decode error, and green after.
`cargo test --workspace`: 2485 passed / 0 failed.

The remaining `decompress_to_yuv_planes` limitation — a handle-less free
function that cannot see any of this — is unchanged and still tracked by
P4-126's criterion 4. What this item removes is the C ABI's dependence on it for
validation.


**Motivation.** Filed 2026-08-09 during the P4-125 review. That item ported
upstream's `num_components > 3` rejection correctly, but placed it at the only
point the current structure allows: *after* `decompress_to_yuv_planes` has
already decoded the whole frame and allocated every plane. Upstream validates
the same frame from the header, before any decompression
(`references/libjpeg-turbo/src/turbojpeg.c:2214-2230`):

```c
    jpeg_read_header(dinfo, TRUE);          /* header only */
  setDecompParameters(this);
  if (this->maxPixels && ... > this->maxPixels)  THROW("Image is too large");
  if (this->subsamp == TJSAMP_UNKNOWN)           THROW("Could not determine ...");
  if (this->subsamp != TJSAMP_GRAY && (!dstPlanes[1] || !dstPlanes[2]))
                                                 THROW("Invalid argument");
  if (dinfo->num_components > 3)                 THROW("JPEG image must have 3 ...");
```

Ours decodes first (`crates/libjpeg-turbo-rs-capi/src/yuv.rs:610` and `:658`)
and only then checks (`:617`, `:665`). The rejection is correct; the *work done
before* rejecting is not. Three consequences, one root cause.

**Root cause hypothesis.** Both entry points route through the root-crate free
function `decompress_to_yuv_planes(data)`, which takes no handle. It calls
`decompress_raw` → `Decoder::new(data)` (`src/api/raw_data.rs:87-90`), so no
handle-scoped configuration and no header-only inspection is reachable from
here. Any check that upstream performs between "header parsed" and "decode
begins" therefore has nowhere to live in this port.

1. **`TJPARAM_MAXPIXELS` is not enforced on these two entry points.** Upstream
   checks it at `turbojpeg.c:2219-2222`, *before* the component guard. The
   handle stores the value (`src/api/tj3.rs:697-699` maps it into `Limits`, and
   `src/common/types.rs:441` enforces it "before any plane allocation"), but
   that plumbing is bypassed: `Decoder::new` uses `Limits::default`. A caller
   setting `TJPARAM_MAXPIXELS` as a DoS bound gets no effect here. Not
   unbounded — the 2,147,483,647-pixel default still applies — but not the
   caller's bound either.
2. **Error precedence diverges for `align`.** Upstream rejects a bad `align` at
   function entry (`turbojpeg.c:2395-2397`, `"Invalid argument"`). Ours only
   discovers it inside `pack_yuv_planes` (`yuv.rs:627`), which the P4-125 guard
   now precedes, so `tj3DecompressToYUV8(h, cmyk, .., align = 0)` reports
   `"must have 3 or fewer components"` where C reports `"Invalid argument"`.
   Both return -1, which is why `yuv_four_component_c_parity` cannot see it: it
   compares accept-vs-reject, not message or precedence.
3. **Partial writes on a NULL plane pointer.** Upstream checks `dstPlanes[1]`
   and `dstPlanes[2]` up front (`turbojpeg.c:2226-2227`) and writes nothing on
   failure. Ours checks each pointer inside the copy loop (`yuv.rs:677`), so a
   NULL `dstPlanes[2]` returns -1 only after planes 0 and 1 have been written
   into caller memory.

**Acceptance criteria.**

1. Both entry points reject 4-component frames, an over-`TJPARAM_MAXPIXELS`
   frame, a bad `align`, and a NULL `dstPlanes[1]`/`[2]` *before* decoding —
   verified by an observation that decode did not run (e.g. timing on a large
   frame, or an allocation/callback counter), not merely by the -1.
2. A C cross-check that pins *precedence*, not just the verdict: for inputs that
   violate two rules at once (CMYK + `align = 0`; CMYK + over-maxPixels), the
   reported error string must match upstream's. This is the gap that let (2)
   through the P4-125 parity test.
3. `tj3Set(handle, TJPARAM_MAXPIXELS, n)` bounds these two entry points, cross-
   checked against C for a frame just over and just under `n`.
4. On any rejection, caller-supplied plane buffers are byte-for-byte unmodified
   — a sentinel-fill assertion over all three, extending the canary already in
   `yuv_decompress_planes_component_guard.rs`.
5. Whatever mechanism lands must reach `decompress_to_yuv_planes`' callers
   without duplicating the decode: either a header-only inspection entry point,
   or a handle-aware variant that carries `Limits`. Record which, since P4-126
   also has to decide what this function does with CMYK.

**Why deferred.** No memory-safety consequence: P4-125 closed the overflow, and
(1)-(3) are wasted work, an error-string mismatch, and a partial write into
buffers the caller supplied and still owns. (1) and (3) predate P4-125; only
(2)'s precedence flip was introduced by it, and only because the guard had to go
where the structure permitted. Fixing this properly means giving the C-ABI layer
a header-only decode path, which is a structural change well beyond the 12-line
guard P4-125 deliberately scoped itself to — and it overlaps the C-ABI-state
workstream P4-123 coordinates.

## P4-128. `tj3YUVPlaneWidth`/`Height` pad plane dimensions to the MCU size in pixels where C pads to the subsampling ratio — **CLOSED 2026-08-09**

**Status (2026-08-09): closed.** Found while building P4-126's C-parity matrix,
which is the only reason it surfaced: the pre-existing legacy coverage
(`tests/legacy_aliases.rs`) probes width 640, which is divisible by every
subsampling factor, so the two formulas agree there and the bug was invisible.
The new matrix uses 100 deliberately.

**The defect.** `crates/libjpeg-turbo-rs-capi/src/bufsize.rs` padded with
`pad_up(width, mcuw)` — the MCU width in *pixels* (8/16/32) — where C pads with
`PAD(width, tjMCUWidth[subsamp] / 8)` — the horizontal subsampling ratio
(1/2/2/4) (`references/libjpeg-turbo/src/turbojpeg.c:1127`, and :1150 for
height). That is 8x too coarse, so every plane whose dimension was not already
MCU-aligned came back over-sized: `tj3YUVPlaneWidth(0, 100, TJSAMP_411)`
returned 128 where C returns 100, and `TJSAMP_444` returned 104 where C returns
100. 21 of 42 matrix cells disagreed.

**Why it mattered beyond the reported number.** `plane_width` / `plane_height`
back `tj3YUVBufSize` and `tj3YUVPlaneSize` as well, so those over-reported too —
safe for allocation, but they disagreed with the *decompress* path, which sizes
its rows through the root-crate `yuv_plane_width` (`src/common/bufsize.rs`) and
pads correctly. A caller that allocated from `tj3YUVBufSize` and then walked the
result using `tj3YUVPlaneWidth` as its stride read misaligned rows: the data was
written at the narrower C-correct stride. The fix makes the C ABI, the root
crate, and C agree on one number.

**Proof.** `crates/libjpeg-turbo-rs-capi/tests/yuv_plane_index_c_parity.rs`
requires all 42 cells to match libjpeg-turbo 3.1.4.1; red at `bad5493`, green
after. `cargo test --workspace` is 2481 passed / 0 failed, so no test anywhere
had pinned the over-padded values — the formula had no coverage at all.
## P4-129. Test-Only `jpeg_capi_test_*` Symbols Ship in the Installed Library and Are Stamped `LIBJPEG_8.0` — **CLOSED 2026-08-09**

**GitHub:** [#460](https://github.com/developer0hye/libjpeg-turbo-rs/issues/460) — under the [#470](https://github.com/developer0hye/libjpeg-turbo-rs/issues/470) umbrella.

**Motivation.** Filed 2026-08-09 by the external drop-in readiness review, which
asked whether the export surface of the shipped library is an exact allowlist.
It is not. `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` defines **16**
`#[no_mangle] pub extern "C"` test accessors — `jpeg_capi_test_arith_code`,
`jpeg_capi_test_density_unit`, `jpeg_capi_test_dimensions`,
`jpeg_capi_test_get_compress_state`, `jpeg_capi_test_marker_list`,
`jpeg_capi_test_output_dims`, `jpeg_capi_test_set_arith_code`,
`jpeg_capi_test_set_compress_dims`, `jpeg_capi_test_set_optimize_coding`,
`jpeg_capi_test_set_out_cs`, `jpeg_capi_test_set_progressive`,
`jpeg_capi_test_set_restart_in_rows`, `jpeg_capi_test_set_restart_interval`,
`jpeg_capi_test_set_smoothing_factor`, `jpeg_capi_test_x_density`,
`jpeg_capi_test_y_density` — with **zero `cfg` gates**. They are compiled into
every build, including the `libjpeg.so.8` that `scripts/install_capi.sh` stages
as a system replacement.

This is a live violation of **[P4-81](#p4-81-linux-cdylib-omits-gnu-libjpeg_80-symbol-versions--partial-nodes-emitted-and-tested-downstream-re-verification-pending)**'s
own acceptance criterion (1), which requires that "crate-only extra/test and
TurboJPEG symbols need an explicit, tested visibility/version policy; they must
not be mislabeled as reference libjpeg-turbo extension exports."

**Root cause.** `gnu_version_script()` in `crates/libjpeg-turbo-rs-capi/build.rs`
(line 270) emits the reference node as a glob:

```
LIBJPEG_8.0 {
  global:
    jpeg_*;
    jcopy_block_row;
    jdiv_round_up;
};
```

`jpeg_capi_test_*` matches `jpeg_*`. P4-81 deliberately omitted a catch-all so
the TurboJPEG surface would stay unversioned, and that reasoning is sound — but
it left the *classic* node a wildcard, so the 16 test accessors are not merely
exported, they are actively stamped as reference libjpeg v8 API in the artifact
`install_capi.sh` relinks. A consumer running `readelf --dyn-syms` on our
`libjpeg.so.8` sees 16 symbols at `@@LIBJPEG_8.0` that no real libjpeg has.

The in-code comment above the block (`jpeglib.rs:3212-3215`) asserts the
opposite: *"These are intentionally NOT `jpeg_*` — they are internal helpers."*
Every one of the 16 begins with `jpeg_`, so as a statement about glob matching
it is false, and it is the sentence that would have prevented this had it been
true. Fix the comment with the code, in the same change.

**Acceptance criteria.**

1. The installed `libjpeg.so.8` exports no `jpeg_capi_test_*` symbol, or exports
   them only under an explicitly non-reference version node that is documented
   as crate-private. Prefer removing them from the shipped artifact outright.
2. `tests/capi_symbol_versions.rs` gains an assertion that fails on **any**
   symbol in `LIBJPEG_8.0` that is not on an exact, enumerated allowlist of the
   reference v8 API. A wildcard that happens to match only good symbols today
   must not pass; the test asserts the allowlist, not the current output.
3. The dlopen-based tests that consume these accessors keep working — via a
   `cfg(feature = …)`-gated build, a separate test-only cdylib, or by reading
   the public struct offsets directly. Record which, and why.
4. The `jpeglib.rs:3212` comment states what is actually true of the naming, and
   names the version-script interaction as the reason the prefix matters.
5. While here, decide and record whether `libjpeg.so.8` and `libturbojpeg.so.0`
   should be relinked as two artifacts with disjoint allowlists rather than the
   single cdylib hardlinked under both names (`install_capi.sh:41-46`). P4-81's
   "we ship one artifact carrying both surfaces" note is the constraint that
   forced the no-catch-all design; splitting the relink is the alternative that
   makes an exact allowlist per SONAME possible. This criterion is a recorded
   decision, not necessarily an implementation.

**Why deferred / severity.** Not memory-unsafe and not a binding failure: the
extra symbols are additive, so no downstream consumer breaks by their presence.
The cost is export-surface integrity — we advertise 16 non-existent entry points
as reference libjpeg v8 API, which is exactly the mislabelling P4-81 set out to
prevent, and it undermines any claim that the shipped surface is audited.

**Status (2026-08-09): closed.** Landed in #486; `crates/libjpeg-turbo-rs-capi/build.rs` now routes the 16 `jpeg_capi_test_*` accessors to a `LIBJPEGTURBORS_PRIVATE_1.0` node via an exact-name list, and `tests/soname.rs` asserts no `jpeg_capi_test_*` symbol carries `LIBJPEG_8.0`.

## P4-130. C-Parity Oracle Is Pinned to 3.1.4.1; Upstream Stable Is 3.2.0 — **PARTIAL: every oracle-provisioning job is now pinned, checked and measured; the legs still on one release, the submodule bump and the four filed gaps remain**

**GitHub:** [#461](https://github.com/developer0hye/libjpeg-turbo-rs/issues/461) — under the [#470](https://github.com/developer0hye/libjpeg-turbo-rs/issues/470) umbrella.

**Motivation.** Filed 2026-08-09 by the external drop-in readiness review.
Upstream released **3.2.0 on 2026-06-30** (verified via the GitHub releases API);
every CI oracle in this repository still installs **3.1.4.1** (2026-03-27) —
13 pin sites across five workflows: `ci.yml:62,287,508,523,526`,
`cross-arch.yml:28,59,95`, `full-c-parity.yml:22,25`,
`fuzz-smoke.yml:93,198,200`. `docs/FEATURE_PARITY.md:483` documents the pin and
its original reason (apt ships 2.1.x, which lacks `-lossless`/`-precision`),
which is still valid — but the pin has not been re-evaluated since 3.2 shipped.

Consequence: our differential gates prove parity with a release that is now one
minor version behind, and the 3.2 delta is entirely unmeasured. Some of it is
directly in scope for open items here.

**Upstream 3.2 changes that touch tracked gaps** (from the 3.2 beta1 and 3.2.0
release notes):

- **Per-instance SIMD dispatch replaces thread-local storage** (beta1 note 2):
  upstream explicitly "eliminat[ed] the need for thread-local storage in the
  libjpeg API library." Our shim's private state is still TLS-keyed — see
  **[P4-132](#p4-132-classic-c-abi-per-cinfo-state-is-thread-affine-p4-16-option-a--open)**.
  Upstream moving off TLS weakens the "upstream is single-threaded too" framing.
- **RISC-V Vector (RVV) SIMD** (beta1 note 6): +149-246% compress, +48-180%
  decompress vs 3.1.x on RVV hardware. See
  **[P4-134](#p4-134-no-risc-v-rvv-simd-backend--upstream-32-ships-one--open)**;
  it also moves the goalposts for **[P4-60](#p4-60-scalar-kernels-are-25x-slower-than-cs-scalar-kernels--open)**,
  whose riscv64 measurement assumed *neither* side had SIMD. That assumption
  expires with 3.2.
- **8-bit lossy JPEG → 12-bit output** (beta1 note 8): new capability via
  `cinfo->data_precision = 12` after `jpeg_read_header()`, and
  `tj3Decompress12()` after `tj3DecompressHeader()`. Not implemented here.
- **`jpeg_crop_scanline()` hardening** (3.2.0 note 3): errors when
  buffered-image mode and raw-data output are both enabled. Relevant to
  **[P4-103](#p4-103-jpeg_crop_scanline-does-not-implement-imcu-aligned-c-semantics--open)**.
- **TurboJPEG additions** (beta1 note 10): repeated `tj3GetICCProfile()`,
  ICC retrieval from a *compression* instance, `TJCS_DEFAULT`, 4:1:0 and 2:4
  subsampling. The last two are already implemented here, so this is a
  differential-semantics check rather than new work.
- **jpegtran `-crop` + `-trim`/`-perfect`** behaviour change (beta1 note 4) and
  the `-crop`/`-trim` overflow fix (3.2.0 note 4).
- **8/16-bit PNG in cjpeg/djpeg and `tj3LoadImage*`/`tj3SaveImage*`** with ICC
  transfer (beta1 note 9).

**Acceptance criteria.**

1. The oracle matrix runs against **both** 3.1.4.1 (behaviour-regression leg,
   keeping the existing expectations honest) and **3.2.0** (current parity
   target). A single global version bump is not acceptable: it would silently
   re-baseline every existing expectation, and any divergence it papers over
   would be indistinguishable from a pass.
2. Each of the seven 3.2 deltas above is triaged to exactly one of: already at
   parity (with the differential test that proves it), a new OPEN LAST_MILE
   entry, or an explicitly recorded non-goal. No delta is left untriaged.
3. `docs/FEATURE_PARITY.md:483` states which upstream version each gate runs
   against and why, rather than naming 3.1.4.1 as if it were current.
4. A stated policy for how upstream releases are tracked going forward, so the
   next minor does not sit unnoticed for two months.

**Why deferred.** Nothing regresses today — 3.1.4.1 is a real, supported
release and the gates it backs are genuine. This is currency, not correctness.
It is sequenced after the Stage A safety items because re-baselining oracles
while the classic-ABI error and state contracts are still in flux would mix two
sources of diff into one signal.

**Status (2026-08-18): partial.** Criteria 1, 3 and 4 are delivered — criterion
1 over the root matrix on 2026-08-17, over the C-ABI crate's oracle suites on
2026-08-18, and over the exhaustive `full-c-parity` matrices the same day — and
criterion 2's triage is complete. Every job that provisions an oracle is now
pinned, checked and measured, whichever workflow it lives in. What remains is
the *work* the triage filed, the legs still measuring one release, and the
submodule bump.

*The pin-and-name rule, generalised to every job (2026-08-18).* The three gates
above name the workflows they read. That is not a detail of their
implementation — it is the reason `ci.yml`'s `test-cross-encode` still ran
`brew install jpeg-turbo` the day after the aarch64 full-parity legs lost
exactly that shape for exactly that reason. It runs on **every pull request**,
on the only macOS leg that runs the whole root suite, and the release it
measured was whatever homebrew shipped that week.

So the rule moved off the list of workflow files and onto the **job**.
`tests/oracle_version_pins.rs` enumerates every job in `.github/workflows`
(45 today, in nine files; the enumeration is checked against a real YAML
parser, name for name, whenever a job is added — 42 when this rule landed, 45
once the cross-arch pairs below joined) and holds each one that provisions a C
libjpeg-turbo to three things:

- **pinned** — the install names the release it installs. Upstream's
  `libjpeg-turbo-official_<version>` package and a `--branch <tag>` clone do;
  a package manager's own name for the package does not, and is now
  unrepresentable rather than discouraged. A submodule build is pinned by
  commit, and a fetch shape the scanner cannot read fails *closed* — a clone
  with no tag is the worst pin of all, and reporting it as "no install here"
  would let it pass.
- **checked** — the job asserts that release. Seven did not: five ran a
  `djpeg -version` and read the output nowhere, which is a print, not a check,
  and two ran no version step at all. A deb that installed something else, a
  tag repointed at another release, or a runner image carrying its own libjpeg
  would each have run the whole leg green under a release it was not measuring.
- **measured** — the prefix it checked is the prefix its tests resolve.
  `test-integration` checked `djpeg -version` *by PATH*, which names no install
  at all. On a macOS runner a PATH entry does not even select an oracle:
  `helpers::c_tool_path` reads `/opt/homebrew/bin` first, so only
  `LIBJPEG_TURBO_PREFIX` counts there — the false green #569 found inside its
  own change, now a rule instead of a memory.

Written first and red on unmodified `main` in all three directions, naming
**eight jobs**: `test-cross-encode` unpinned; `mutants-in-diff`,
`test-integration`, `test-corpus`, the three oracle-installing `cross-arch.yml`
jobs and `fuzz-smoke.yml`'s `fuzz` never asserting; and `mutants-in-diff`,
`test-integration` and `test-corpus` checking no install path at all. Two of
those — `mutants-in-diff` and `test-integration` — are *not* in the inventory
this item's remainder listed, because that inventory counted legs measuring one
release and these were correctly pinned. Being pinned and being checked are
different properties, and only the second survives a mis-provisioned runner.
One of the eight is weaker than the rest by construction and says so in the
workflow: `mutants-in-diff` is `continue-on-error`, so its assertion stops that
job and leaves CI green. It protects the meaning of the mutation result — a
mutant that survives because the oracle was a different libjpeg reads as MISSED
for the wrong reason — not the merge.

*The scope that matters is the step, not the job (codex round).* The first
draft of the "measured" rule compared a job's *union* of prefix assignments
against the prefixes it checked, and a review pointed out that the union is
precisely the wrong shape: a step-level assignment overrides the job's for that
step alone, so a leg could name the checked install on one step and run
`cargo test` against another — `/opt/homebrew`, on macOS — with the gate green.
The rule is now per step: each step's *effective* prefix (its own, else the
job's) must be one the job checked, and a step that names none relies on lookup
order, which is accepted only off macOS and only when a checked prefix is on
PATH. Two things followed. First, checking became per prefix at a release
(`prefix_releases_checked_in`) rather than "somewhere in this job a version is
asserted", since `test-integration` carries three oracles and one check would
otherwise vouch for all of them; a `tee` file belongs to the invocation that
wrote it. Second, the two v8 source builds — `/tmp/ljt8/prefix` from the
submodule and `/tmp/ljt320v8/prefix` from the 3.2.0 clone — now assert their
own releases, which is also what makes a submodule bump come through a workflow
line rather than re-baselining the classic-ABI trace oracles silently. Verified
by removing each check in turn: dropping the 3.1.90 assertion turns the four
steps that select that prefix red, dropping `test-cross-encode`'s job-level
prefix turns its `cargo test` step red as a macOS lookup-order step, and
pointing one `test-integration` step at `/opt/homebrew` turns exactly that step
red. The step parser had the same bug in miniature and is pinned against it: it
read the step indent off the first `- ` line in the job, which in a matrix job
is a matrix entry, so every step merged into one — a job-scope union arriving
through the parser rather than the rule.

A second review round found the same class once more, in *which steps get
asked*. "Does this step run the tests?" was a substring search for
`cargo test`, `cargo run`, `cargo mutants` and `fuzz run` — and
`cargo +nightly test` contains none of them while running the whole suite, a
spelling these workflows already use. A step the predicate misses is a step the
gate never asks about, so the answer is not another substring: the cargo
invocation is parsed, the optional `+toolchain` skipped, and the subcommand
compared against a **deny** list of the ones that cannot reach an oracle
(`build`, `clippy`, `install`, `fmt`, …). The directions are not symmetric —
an unlisted subcommand that *does* reach the oracle is a silent hole, while an
unlisted one that does not costs a line here the first time it appears in an
oracle-installing job. Verified against the reproduction: pointing the macOS
leg at `cargo +nightly test` with no prefix is red.

A third round found the cost of that parse. An exact-token comparison for
`cargo` no longer recognised `"cargo test"` in a quoted YAML scalar or
`out=$(cargo test …)` in a command substitution — both of which the substring
match it replaced had handled by accident. The token is now read past the shell
punctuation that can precede a command word, `--cargo-test-arg` still being a
flag that names cargo rather than an invocation of it; that shape is red
against the real workflow too.

A fourth round found the mirror — *false failures*, the direction that costs a
valid CI change rather than a silent hole. Stripping punctuation off every
token promoted an argument to a command, so `echo "cargo test --tests"` read as
a test run, and `CARGO=/usr/bin/cargo cargo build` read as one too; while a
wrapper closing right after a deny-listed subcommand left `build)` and `fmt"`,
which match no deny-list entry. Tokens are now read in *command* position only
— after separators, after the keywords these workflows wrap commands in
(`test-corpus` runs `if ! cargo run …`), and past an environment prefix — and a
word is trimmed of its closers before its openers, since stripping only openers
turns `"cargo fmt"` into an empty subcommand that is in no deny list.

A fifth round found what the command-position rule had *cost*, including one
case live in `ci.yml`. The step reader joined a `run: |` block's lines with
spaces, so a block was one long command: P4-81's step is `set -o pipefail`
followed by `cargo test …`, and flattened it reads as a single command named
`set` — invisible to a rule that only looks at command position. Scripts are
now read per logical line, each of which starts a command. Three narrower
misses came with it: a command substitution runs its contents wherever it
appears (`echo "$(cargo test)"` runs the tests), a wrapper takes options of its
own before the command it wraps (`sudo -E cargo test`, `env -u FOO cargo test`),
and an inline assignment whose value contains a space is two tokens of which
the second is not an assignment (`RUSTFLAGS="-C target-cpu=native" cargo test`).

Five rounds, eight findings, both polarities, and every one of them in the
*scanner* rather than the rule: shapes it could not see, and shapes it saw
where there was nothing. Each fix bought its own next finding — the substring
match handled quoting by accident, the parse that fixed toolchain qualifiers
lost it, the command-position rule that fixed the false positives lost the
multi-line block — which is the argument for pinning every helper against the
spelling that would make it lie rather than trusting a gate because it is
green, and for keeping *both* polarities in the pins, since a gate that cannot
be wrong in the second direction is usually one that has stopped matching.
Verified against the real workflows at the end of it: with the macOS leg's
prefix removed, all six spellings of its test step are red, and the multi-line
P4-81 step is now one of the steps the rule reads. A sixth round found three
more shapes — an escaped quote inside an inline assignment, folded scalars and
heredocs, and substitution syntax inside single quotes — and none of them is in
`.github/workflows`, so they are filed as
**[P4-177](#p4-177-the-workflow-scanner-does-not-model-heredocs-folded-scalars-or-quoted-substitution-syntax--partial-folded-scalars-are-modelled-heredocs-and-quoteescape-state-remain)**
rather than fixed here. The gate fails closed, so what they cost is precision,
not coverage; and this file's own history is the argument for giving the
scanner its own change and its own review rather than a seventh pass inside
one about oracle legs.

`test-cross-encode` now builds 3.1.4.1 from source at `/tmp/ljt3141/prefix` and
selects it with `LIBJPEG_TURBO_PREFIX`, the same shape the aarch64 full-parity
baseline leg took. It is the same measurement under a name: homebrew's formula
was 3.1.4.1 on the day it was replaced. Both per-job gates report *every*
offender rather than the first, since these legs span three workflows and a
gate that names one per run turns one review into as many rounds as there are
legs.

The helpers are pinned against the shapes that would make them lie, as the
gates above are: an unpinned package-manager install is recognised while
`apt-get install -y /tmp/ljt.deb` and `brew install cmake` are not; a comment,
an `echo`'d reproduction instruction and a step *title* naming a release are
documentation rather than installs; a backslash continuation is classified as
the one command it is (`ci.yml` splits both provisioning shapes across lines,
and each half alone names either the release or the command, never both); a
bare `djpeg -version` checks no install; and a PATH entry selects the oracle
everywhere **except** macOS, read from the whole job block because
`test-cross-encode` names its runner only in its matrix.

*Criterion 1, the exhaustive matrices (2026-08-18).* The widest differential
surface in this repository is not in `ci.yml` at all. The `full-c-parity`
feature gates 12,230 transform cases, a 10,880-cell crop grid and the tj
comp/decomp matrices behind `full-c-parity.yml`, which runs weekly — and
`cargo test --tests` does not build them, so neither leg of the pair above ever
touched them. They now run on both, per architecture: `full-c-parity-x86` /
`full-c-parity-x86-current-oracle` and `full-c-parity-arm64` /
`full-c-parity-arm64-current-oracle`. Per architecture rather than once,
because every case here compares *our SIMD backend's* bytes against C's, so
x86_64 and aarch64 are two measurements and not two runs of one.

*The aarch64 legs had no pin at all.* They ran `brew install jpeg-turbo`, which
has no per-version formula: the release those matrices proved parity with was
whatever homebrew shipped that week, named nowhere in this repository. It
happened to be 3.1.4.1 on 2026-08-18, so the results were read as a baseline
measurement by luck. The week homebrew moves to 3.2.0, the matrices would have
re-baselined *silently* — the single global bump criterion 1 forbids, arriving
without a commit to review. Both aarch64 legs now build from source at a pinned
tag (`/tmp/ljt3141/prefix`, `/tmp/ljt320/prefix`); upstream ships no macOS
package, which is why the deb that serves the x86_64 current leg cannot serve
this one.

Three gates hold the shape, all written before the workflow was touched and all
red first:

- `each_full_c_parity_leg_provisions_the_release_its_role_names` reads the
  version out of `docs/oracle_versions.tsv` by role and requires the matching
  leg to install exactly it. An unpinned install provisions *nothing* the
  scanner can see, so this is what makes the `brew install` shape
  unrepresentable rather than merely discouraged.
- `every_full_c_parity_leg_verifies_the_prefix_it_measures` — installing a
  release is not measuring it. `helpers::c_tool_path` reads `/opt/homebrew/bin`
  before PATH, which is exactly how the false green inside #569 happened. So
  the gate parses the `LIBJPEG_TURBO_PREFIX` assignment **at job level** — a
  mention in a comment selects nothing, and a step-level assignment selects the
  oracle for that step alone, which is the shape where a leg verifies one
  install and measures another — and requires the leg to run `-version` on
  **that prefix's** `djpeg` and to *assert* over the output that it reports the
  release the leg's role declares. `echo "version 3.2.0"` satisfies a substring
  search and checks nothing, so the assertion has to be one.
- `every_oracle_backed_full_parity_suite_on_the_baseline_leg_also_runs_on_the_current_leg`
  is the capi pairing gate's twin over root-crate suites, run once per
  architecture pair. It classifies from each suite's own source, so a matrix
  that *gains* a C comparison is reclassified by the commit that gives it one,
  and it compares **what each invocation selects** and not merely binary names:
  a filter that selects one test, or after a typo none, leaves the legs
  measuring different things while a name-only comparison stays green —
  P4-61's finding, twice over. Both pairing gates now do this, over a
  three-valued selection (the whole binary / these filters / *unreadable*) that
  fails closed: `-- --ignored` and a shell-quoted filter both change which
  tests run, and reading either as "no filter" would have turned an unparsed
  argument into a claim of full coverage.

The helpers those rest on are pinned too, each against the shape that would
make it lie: `the_suite_selector_tells_the_root_crate_from_the_c_abi_crate`
(an invocation naming no package is the root crate; one naming any package is
somebody else's coverage, and confusing them would look up a capi suite under
`tests/` and panic), `a_step_named_after_a_test_is_not_a_libtest_filter`
(every step in this workflow is *named* after the matrix it runs, so without a
step-key boundary the next step's name reads as another filter on the previous
invocation), `a_harness_flag_is_not_a_filter_and_a_redirect_ends_the_invocation`
(`--nocapture` selects nothing; `2>&1 | tee` is the shell taking the line
back), `an_argument_this_scanner_cannot_read_fails_the_comparison_closed`,
`a_value_taking_harness_flag_does_not_look_like_a_filter` (`--test-threads 1`
runs everything, `--skip` does not — failing closed is only useful if it fires
on the arguments that really change the set),
`a_narrower_selection_on_the_current_leg_does_not_cover_the_baseline`,
`running_a_suite_twice_in_one_leg_selects_the_union` (an unfiltered run absorbs
a filtered one, or a leg running both would compare equal to one running only
the filter), `only_a_job_level_prefix_assignment_selects_the_oracle_for_a_whole_leg`,
`a_version_check_that_cannot_fail_is_not_a_version_check`,
`the_oracle_classifier_reads_root_crate_suites_too` in both directions, and
`a_job_block_stops_at_the_next_job` extended to the new pairs — every
full-parity leg name is a prefix of its own current-oracle twin, so the
block-parse trap that gate exists for is waiting once per architecture.

The selection comparison and the prefix parse are both **later-round work**,
and the sequence is the interesting part. The first draft compared binary names
and grepped for the variable; a codex review pointed out that each would stay
green through exactly the substitution it was written to catch. The second
carried filters and parsed the assignment, and a second review found four more
shapes that still passed — an unmodelled `--ignored`, a shell-quoted filter, a
suite run twice whose unfiltered run was merged away, and a step-scoped
assignment. The third fixed those, and a third review found the version check
accepted `true # grep …` and `grep … || true`, that a step-level override was
skipped rather than rejected, and that `--test-threads 1` was misread as a
filter — a *false* failure, the mirror of the rest. Every hole was in the
**gate**, not in the workflow, which is the argument for pinning each helper
against the shape that would make it lie rather than trusting a gate because it
is green.

Measured on macOS aarch64 against locally built prefixes at both releases,
running the four binaries unfiltered: **15 tests, 0 failed on each leg**, with
printed tallies summing to **19,439 compared cases** (crop 5,504; tjdecomp
1,689; tjtran 12,246 — `c_tjcomptest` prints no tally). A CI leg runs 13 of
those 15, because the `c_tjdecomptest` step keeps the pre-existing
`-- c_tjdecomptest_full` filter and leaves that binary's two quick tiers to the
per-pull-request matrix, which runs them without the feature. Both legs carry
the same filter, and since 2026-08-18 the pairing gate compares filters rather
than binary names, so they cannot drift apart. The two legs' tallies are identical, including the
pre-existing exclusions (5,440 crop cells for unimplemented colour
quantization, 36 tjdecomp cases for a subsampling with no Rust equivalent), so
the matrices are already at parity with 3.2.0 and this leg is what keeps that
true. The source-built 3.1.4.1 prefix reproduces the homebrew leg's numbers
exactly, so replacing `brew install` changes what that leg *names*, not what it
measures. The oracle is mandatory rather than assumed: pointed at a
non-existent prefix under `CI=1`, `c_croptest` fails 5/5 instead of skipping.

*Criterion 1, the C-ABI half (2026-08-18).* The 3.2.0 leg first covered the
root differential matrix only, because `cargo test --tests` selects the root
crate: the classic-`jpeg_*` and TurboJPEG shim — the half of this repository
whose whole contract is "what stock libjpeg does" — was still measured against
3.1.4.1 alone. It no longer is. Fourteen C-ABI suites now run on both legs:

- six against the 3.2.0 deb (`capi_jpeglib_encode`,
  `capi_classic_lifecycle_pathological`, `capi_compress_precision`,
  `norealloc_all_entry_points`, `yuv_four_component_c_parity`,
  `yuv_plane_index_c_parity`);
- eight against a **new** `trace-current` oracle — 3.2.0 built from source with
  `WITH_JPEG8=1` at `/tmp/ljt320v8/prefix`, because the classic-ABI trace
  suites compare compiled-oracle traces line by line at the v8 ABI and
  upstream's deb ships `JPEG_LIB_VERSION 62`. Building 3.2.0 beside the
  submodule moves that comparison to current stable **without** touching a
  single `j*.c:NNN` citation, which is what keeps the submodule bump a separate
  change.

Which suites belong there is decided mechanically, not in prose:
`every_oracle_backed_capi_suite_on_the_baseline_leg_also_runs_on_the_current_leg`
reads each baseline-leg suite's own source for a C-oracle marker
(`LIBJPEG_TURBO_PREFIX`, `build_classic_oracle`, a stock tool name, …) and
fails if one of them runs on the baseline leg alone. Classifying from the
suite's source rather than from a hand-kept list is what stops the pair from
drifting: a suite that *gains* a C comparison is reclassified by the commit
that gives it one. The classifier is pinned in both directions by
`the_oracle_classifier_separates_c_comparisons_from_self_contained_suites`, and
`a_job_block_stops_at_the_next_job` pins the parse — `test-integration` is a
prefix of `test-integration-current-oracle`, and a block that ran on into the
next job would compare a set with itself and never fail.

Everything self-contained stays on the baseline leg alone on purpose: the four
capi steps duplicated nowhere on the new leg — the span-overflow guards, the
error-code and message-rendering gates, the crate's unit tests, the ELF
symbol-version leg — plus the seven self-contained suites riding inside the
mixed steps (`capi_input_complete_contract`, `capi_suspended_body_cap`,
`norealloc_buffer_capacity`, `yuv_packed_length_overflow`,
`yuv_four_component_guard`, `yuv_decompress_planes_component_guard`,
`yuv_validate_before_decode`). All of them compare against constants, our own
generated files, or the pinned headers, so a second run at another upstream
release measures the same thing twice.

One of the fourteen did not honour the prefix, and the review caught it before
this landed. `capi_classic_lifecycle_pathological` carried a **private**
`find_c_tool` — a survivor of the P4-116 sweep of residual private helpers —
that ignored `LIBJPEG_TURBO_PREFIX` and searched `/opt/homebrew/bin`,
`/usr/local/bin`, `/usr/bin` and only then `/opt/libjpeg-turbo/bin`. So the
first measurement of that suite compared against homebrew's 3.1.4.1 while
reporting under a step named 3.2.0 — the exact false green the two legs exist
to prevent, reappearing inside the change that adds them. It now mirrors
`tests/helpers::c_tool_path`: an explicit prefix is exclusive, with the
lookup split into an injectable `find_c_tool_under` so
`an_explicit_oracle_prefix_is_exclusive` can assert both branches on every
platform rather than only where a second install happens to exist. The
baseline leg's step now names its prefix too — it had been landing on the
right install by absence rather than by choice.

Measured on macOS aarch64 after that fix, against locally built 3.2.0 prefixes
(default and `WITH_JPEG8=1`): the three new steps are **14 suite sections, 78
passed, 0 failed**. The one skip is `stdio_dev_full` — `/dev/full` does not
exist on macOS; on the Ubuntu runner it is exercised, and the step fails closed
on any skip line exactly as its baseline twin does. The oracle was verified
mandatory rather than assumed, in both shapes: pointed at a non-existent
prefix, `capi_classic_lifecycle_state` fails with `no stock *v8* libjpeg
development install found`, and `capi_classic_lifecycle_pathological` fails
with its own missing-oracle panic instead of falling back to the host's.

*Criterion 1 — two legs.* `ci.yml`'s `test-integration-current-oracle`
("Integration Tests (oracle 3.2.0)") installs the official 3.2.0 deb, asserts
`djpeg`/`cjpeg`/`jpegtran` all report `version 3.2.0` before running anything,
and runs the same `cargo test --tests` command as the 3.1.4.1 leg. The oracle
is named by `LIBJPEG_TURBO_PREFIX`, not by PATH order, because
`tests/helpers::c_tool_path` reads `/opt/homebrew/bin` first and would
otherwise let a leg labelled 3.2.0 measure something else and report green; the
prefix is exclusive, so a missing tool under it is an error rather than a quiet
fallback. Measured before landing against a locally built 3.2.0 on macOS
aarch64, `cargo test --tests --no-fail-fast` with `LIBJPEG_TURBO_PREFIX` at that
build: **223 suite sections, 2351 passed, 0 failed, 4 ignored** — the root
differential matrix is already at parity with 3.2.0, which is the result the leg
now keeps true. The same command with the variable unset (homebrew 3.1.4.1) is
green on this branch too, so the second leg adds coverage rather than replacing
it.

*The oracle the write-up missed.* Writing the manifest surfaced that this
repository was **already running two upstream versions, undocumented**:
`references/libjpeg-turbo` is pinned at **3.1.90 (3.2 beta1)**, not 3.1.4.1, so
the classic-ABI trace oracles built from it (`/tmp/ljt8/prefix`, `WITH_JPEG8=1`)
and every `j*.c:NNN` citation in this repository already quote the 3.2 line
while the tool oracles quote 3.1.4.1. `docs/oracle_versions.tsv` records the
split and `tests/oracle_version_pins.rs` cross-checks that row against the
submodule's own `CMakeLists.txt`, so it cannot drift again.

*Criterion 3 — the manifest.* `docs/oracle_versions.tsv` names the release each
oracle role runs against and why — four rows since the C-ABI half landed, the
new one being `trace-current`. The gate holds it in both directions: a workflow
pin with no row fails, and a declared leg no workflow installs fails — because
criterion 1 asks for a second *running* leg, not a documented intention. For
`trace-current` "installs" is narrowed to a `--branch` source clone, since the
deb that satisfies `tool-current` is the same version and would otherwise back
a v8-ABI row it cannot serve; `the_official_deb_does_not_back_a_v8_abi_row`
pins that distinction. `docs/FEATURE_PARITY.md` now states what each leg proves
instead of naming 3.1.4.1 as if it were current.

*Criterion 4 — the policy, as a job.* `scripts/check_oracle_currency.sh`
compares the manifest's `tool-current` row against upstream's latest stable
release and fails when they differ; `.github/workflows/upstream-currency.yml`
runs it weekly, on demand, and on any pull request touching the manifest or the
script. This is the mechanical form of the policy `FEATURE_PARITY.md` had
already written down: the failure mode has nothing a test can see — 3.1.4.1
never went red — so only something that watches upstream can report it.

*Criterion 2 — the triage.* Every 3.2 delta, from the 3.2.0 and 3.1.90
(3.2 beta1) release notes read in full rather than from the summary this item
was filed with:

| # | 3.2 change | Disposition |
| --- | --- | --- |
| beta1-2 | Per-instance SIMD dispatch replaces thread-local storage | **Tracked — [P4-132](#p4-132-classic-c-abi-per-cinfo-state-is-thread-affine-p4-16-option-a--open).** Upstream removing TLS from its libjpeg API is that item's central evidence. |
| beta1-6 | RISC-V Vector (RVV) SIMD | **Tracked — [P4-134](#p4-134-no-risc-v-rvv-simd-backend--upstream-32-ships-one--open)**, and it expires [P4-60](#p4-60-scalar-kernels-are-25x-slower-than-cs-scalar-kernels--open)'s premise that riscv64 was scalar-vs-scalar. |
| beta1-8 | 8-bit lossy JPEG decompressed to 12-bit output | **New — [P4-171](#p4-171-8-bit-lossy-jpeg-cannot-be-decompressed-to-12-bit-output-32-beta1-note-8--open).** Measured: `src/api/precision.rs:865` refuses any stream whose precision is not 12, so we reject where 3.2 decodes. |
| beta1-10 | TurboJPEG: `TJCS_DEFAULT`, repeated `tj3GetICCProfile`, ICC from a compression instance, 4:1:0 and 2:4 subsampling | **Split.** 4:1:0 and 2:4 are implemented (`TJSAMP_410`/`TJSAMP_24`) and covered by the subsampling matrices; the ICC and `TJCS_DEFAULT` additions are **new — [P4-172](#p4-172-turbojpeg-32-icc-and-tjcs_default-additions-are-unimplemented-32-beta1-note-10--open)**. |
| beta1-4, beta1-12, 3.2.0-4 | jpegtran `-crop` expansion honouring `-trim`/`-perfect`; new `-roll`; the `-crop`/`-trim` overflow fix and its flatten/reflect error | **New — [P4-173](#p4-173-jpegtran-32-crop-expansion--roll-and-the-flattenreflect-refusal-are-unported--open).** These are `transupp.c` semantics our transform API mirrors, so "it is app code" does not exempt us. |
| beta1-9, 3.2.0-2 | 8/16-bit PNG in cjpeg/djpeg and `tj3LoadImage*`/`tj3SaveImage*`, ICC transfer, PNG-writer hardening | **New — [P4-174](#p4-174-png-interchange-parity-for-tj3loadimagetj3saveimage-is-narrower-than-32--open).** PNG exists here behind a cargo feature; the 3.2 additions (16-bit, ICC transfer under `TJPARAM_SAVEMARKERS`, reversible upscaling of non-standard precisions) are not implemented. |
| 3.2.0-3 | `jpeg_crop_scanline()` errors when buffered-image mode and raw-data output are both enabled | **Tracked — [P4-103](#p4-103-jpeg_crop_scanline-does-not-implement-imcu-aligned-c-semantics--open).** The delta is exactly one condition: 3.2.0 `src/jdapistd.c:203` reads `if (cinfo->master->lossless \|\| cinfo->raw_data_out)` where 3.1.90 reads `if (cinfo->master->lossless)`. Recorded in P4-103 as an acceptance line. |
| beta1-1, beta1-5 | GAS Neon implementation removed; MIPS DSPr2 SIMD removed | **Non-goal.** We have neither an assembler Neon path (ours is `core::arch` intrinsics) nor a MIPS backend, so there is nothing to follow. |
| beta1-3 | `WITH_PROFILE` throughput reporting | **Non-goal.** An upstream build-time diagnostic with no ABI or output surface; our measurement lives in `experiments/`. |
| beta1-7 | TurboJPEG Java API moved to its own repository | **Non-goal.** No Java binding is in scope here. |
| beta1-11 | `-nooverwrite` in cjpeg/djpeg/jpegtran | **Non-goal.** Pure CLI file handling in upstream's application code; we ship a library and link *stock* tools against it, so the option is upstream's to implement and ours to inherit. |
| 3.2.0-1 | Arm64EC Windows build regression fixed | **Non-goal.** An upstream build-system fix with no behavioural surface. |

*Criterion 1, the cross-arch backends — and pairing as a property of the job
(2026-08-18).* The three legs in `cross-arch.yml` are now paired:
`test-linux-aarch64-neon-current-oracle`,
`test-linux-x86_64-avx2-current-oracle` and
`test-linux-x86_64-no-avx2-build-current-oracle`, each running the same
`cargo test --tests` command as its baseline twin, on the same runner, under
the same job-level environment.

Three legs rather than one, for the reason the exhaustive matrices are paired
per architecture: every case in that root matrix compares *our SIMD bytes*
against C's, and this workflow exists precisely because the backend differs.
`ci.yml`'s pair answers for one x86_64 runner at default codegen; these answer
for Linux NEON, for AVX2-enabled codegen, and for codegen with AVX2 and SSE4.2
switched off. Three backends against one release is three unmeasured 3.2.0
answers. The third is the weakest of the three and its job comment says so:
SIMD *dispatch* is a runtime CPUID query that ignores `-C target-feature`, so
the kernels are the AVX2 ones on both legs of that pair — what the flags change
is how LLVM compiles everything that is not a hand-written kernel, which is
most of what the differential suites compare.

**The membership rule moved onto the job, exactly as the pin-and-name rule
did.** The three pairing gates that existed each *named* the legs they
compared — `CI_WORKFLOW`/`BASELINE_LEG_JOB`/`CURRENT_LEG_JOB` and
`FULL_PARITY_LEG_PAIRS` — which is the shape whose failure this entry already
records once: a workflow that grows a new oracle leg is covered by nothing
until someone remembers to add it to a constant, and "someone remembers" is
what let thirteen pins sit at a superseded release for two months.
`every_oracle_installing_job_is_paired_or_on_the_recorded_remainder` reads all
45 jobs in the nine workflows and requires each one that installs a C
libjpeg-turbo to be a baseline with a `-current-oracle` twin, that twin, or a
row in `UNPAIRED_ORACLE_JOBS`. Written first and red on the unmodified tree,
naming exactly the three cross-arch jobs.

That inventory is the item's remainder, moved from this prose into the one
place a gate can read it, and checked **both ways**: a row naming a job that is
paired, that installs no oracle, or that does not exist fails as loudly as an
unpaired job missing from the list. Prose could go stale in either direction —
a paired leg staying listed, a leg added tomorrow listed nowhere — and this one
had, in a small way: the paragraph it replaces counted *legs* and said eight,
while the job scanner counts *jobs* and finds seven, because `fuzz-smoke.yml`'s
three differential targets are three entries of one matrix job and
`mutants-in-diff` was left out of the count as "correctly pinned" (a different
property, as this entry notes above).

`every_leg_pair_is_compared_and_a_twin_runs_what_its_baseline_runs` is the
second half, because naming a twin is not running one. Every discovered pair
goes down exactly one of two comparisons and neither branch may empty: a leg
that selects at least one **oracle-backed suite** by name is compared by
*selection*, since `test-integration` deliberately keeps its self-contained
suites off its twin; a leg that selects none runs whole crates, and then the
`cargo test` command — with the environment it runs under and the runner it
runs on — is the only thing there is to compare.

*Five review rounds, seventeen findings, all of them on the gate again.* The
first draft was green through each of the substitutions it existed to catch:

- **An echo is not a run.** The command scanner was a substring split on
  `cargo test`, so replacing a twin's real command with `echo cargo test
  --tests` left the echoed text standing in for the invocation the pair
  requires — 45 gates green over a leg executing nothing. It now shares
  `cargo_invocations_in` with the "measured" rule's `runs_cargo_in`: command
  position, `+toolchain` qualifiers, wrappers, environment prefixes and
  substitutions, one implementation rather than two.
- **A step's environment overrides its job's.** The comparison read job-level
  `env:` only, so a twin could set `RUSTFLAGS: -C target-feature=-avx2` on its
  `cargo test` step and compile the opposite backend with the gate green — the
  step-versus-job scope error this entry already records once, in the "measured"
  rule. Commands and environments now travel together as a `TestRun`, the
  environment being each step's *effective* one, job overlaid by step, minus
  the oracle prefix a twin is supposed to differ in.
- **A pair that starts naming suites escaped both comparisons.** The
  whole-suite branch simply skipped a pair that named any suite, on the
  assumption that a suite-level gate would take it — and those gates read
  `ci.yml` and `full-c-parity.yml` by name, so a cross-arch leg narrowed to
  `--test <something>` was compared by nothing. Every pair is now compared, the
  branch is chosen by whether *oracle-backed* suites are named (narrowing to a
  self-contained suite is the same escape one step further on), and the two
  branch counters must both stay non-zero.

A second round found that the first two fixes had each stopped one step short,
and that the third had invented a gap:

- **A mixed leg's whole-crate run was compared by nothing.** `test-integration`
  runs `cargo test --tests` *and* names capi suites, so it took the selection
  path — and the selection path never looks at commands. Its twin's
  `cargo test --tests` could become an `echo`, or gain a RUSTFLAGS, with the
  capi selections still matching. The root crate's whole integration matrix is
  now compared for **every** pair, before the branch, because that command is
  what makes a leg a measurement at all.
- **The selection reader was still a substring split.** Fixing the command
  scanner left `suites_selected_by` reading `echo cargo test --test c_croptest`
  as a selection, so the same substitution survived one path over. Both now
  read through `cargo_invocations_in`; bounding each invocation at the shell
  separator also stopped a following invocation's `-p` from being carried into
  this one's arguments, which had let a root-crate run read as another crate's
  coverage.
- **The gap the generalisation "found" was not one.** Comparing root
  selections reported `hard_case_x_byte_and_restart` as running on the baseline
  leg alone, and it was filed as an item and excepted in a constant. It is
  wrong: only `restart_bomb_4096_terminates_within_budget` is `#[ignore]`d, and
  both legs run `cargo test --tests`, so the suite's
  `restart_bomb_4096_dimensions_match_djpeg` cross-check already answers at
  **both** releases. The scanner's model — "a suite is covered when a leg names
  it with `--test`" — could not see a whole-crate run covering a suite neither
  leg names. `a_shared_whole_root_run_covers_both_legs` models it now, credit
  exactly as wide as the identical command the two legs share, and the item and
  its exception are retracted rather than left standing as a gap nobody has.
  What such a run does not cover is a selection widening past the default set:
  the serial step's `--include-ignored` adds three timing assertions the twin
  never runs, and a 60 s liveness bound does not answer differently at another
  upstream release.

A third round found the credit and the tokeniser each one step short again,
and each miss was the same shape: something that *looks* like the thing being
credited.

- **`cargo test --lib` looks whole-crate.** It names no suite and no package,
  so the classifier read it as the root matrix — and putting it on *both* legs
  of a pair credited every root oracle suite to a pair running no integration
  test at all. The classification is now cargo's own: a package selector
  disqualifies, `--tests`/`--all-targets` qualifies, and a command with no
  target selector qualifies because that is what cargo does with one.
- **The credit was unconditional.** It has to be exactly as wide as a *default*
  build of that command: a `--features full-c-parity` selection is not in one
  (that flag gates 12,230 transform cases), and neither is a selection widening
  past the default set. `--include-ignored` stays credited and says why — the
  default half runs on both legs and the ignored half is this repository's
  serial timing assertions — while bare `--ignored` selects only tests the
  shared run never executes.
- **A control operator glued to an argument ate it.** `if cargo test --test
  c_croptest; then` leaves `c_croptest;` as one whitespace token, and ending
  the invocation *at* that token dropped the suite name with the separator, so
  a leg naming an oracle suite read as naming none. The argument in front of
  the operator is kept now — but not in front of a redirect, where what
  precedes `2>&1` is a file descriptor, and keeping it would turn `2` into a
  libtest filter.

A fourth round found the same class once more, and the pattern by then was
plain: **every miss is a command that looks like the one being credited.**
`cargo test --tests --no-run` compiles the binaries and runs nothing;
`cargo test --tests c_crop` runs the tests whose names match, which is the
zero-test positional-filter shape `ci.yml` carries a comment about; and a twin
naming the same suite *without* `--features full-c-parity` compiles none of the
12,230 transform cases the flag gates while comparing equal on selection alone.
Whole-matrix credit now rejects a compile-only or filtered invocation, and the
feature set is part of what the two legs are compared on rather than a
yes/no beside it. The same round found the tokeniser dropping
`--test c_croptest>/dev/null` entirely: the argument in front of a glued
operator is kept now unless it is a bare file descriptor, which is the only
thing `2>&1` has there.

A fifth round split the last conflation. The feature set and the libtest
selection were two independent unions, so a baseline running
`--features full-c-parity -- c_croptest_full` was covered by a twin that ran
`full-c-parity` under some *other* filter and `c_croptest_full` without the
feature — neither of which compiled the exhaustive case. They are one key now:
what a leg runs of a suite is a selection *per build*. The same round found
three spellings the parser rejected rather than missed — `--all-features` read
as a literal feature name instead of a top value, cargo's attached `-Ffeature`
form, and an operator glued to the command word (`cargo test; echo done`, where
scanning past it took `echo` for a positional filter). Those cost valid
workflow changes rather than coverage, which is the polarity worth keeping in
the pins.

Mechanism-validated in twenty-five directions rather than by passing: on the
workflow side, dropping the twin's job-level RUSTFLAGS, overriding RUSTFLAGS on
its test step, replacing its command with an `echo`, narrowing either leg to a
single suite (oracle-backed or not), moving the twin to another runner,
pointing it at 3.1.4.1, renaming it so the pair dissolves, and renaming its
baseline out from under it each turn the intended gate red; so do the three
mixed-leg shapes on `ci.yml`'s pair — echoing its twin's whole-root command,
adding a RUSTFLAGS to it, and echoing its twin's capi step — and swapping both
its legs' `--tests` for `--lib`, adding `--no-run` to both, adding a positional
filter to both, widening its baseline's selection with `--features`, and
dropping `--features` from a `full-c-parity` twin; and on the inventory side,
deleting a remainder row while its leg stays single, keeping one after the leg
is paired, and naming a job that does not exist or that installs no oracle.

The scanner work also discharges criterion 1 of
**[P4-177](#p4-177-the-workflow-scanner-does-not-model-heredocs-folded-scalars-or-quoted-substitution-syntax--partial-folded-scalars-are-modelled-heredocs-and-quoteescape-state-remain)**,
which that item was holding open: comparing two legs' commands is impossible
while a folded `>` block reads as one command per physical line, because every
argument past the first drops out of the comparison.

**What remains.**

1. The four filed gaps (P4-171..P4-174) are triaged, not fixed.
2. **Four jobs still measure one release**, and the inventory now lives in
   `UNPAIRED_ORACLE_JOBS` in `tests/oracle_version_pins.rs`, where a gate reads
   it, with the reason on each row:
   - `fuzz-smoke.yml`'s `fuzz` — the three differential fuzz targets
     (`fuzz_decode_diff_c`, `fuzz_encode_diff_c`, `fuzz_transform_diff_c`) are
     matrix entries taking the 3.1.4.1 deb, and the reproduction instructions
     the failure path prints name that release too, so pairing has to reach
     those as well;
   - `ci.yml`'s `test-corpus` — a 3.1.4.1 source build at `/usr/local`;
   - `ci.yml`'s `test-cross-encode` — a 3.1.4.1 source build at
     `/tmp/ljt3141/prefix`, on the only macOS leg that runs the whole root
     suite; upstream ships no macOS package, so its twin is a *second* source
     build on every pull request;
   - `ci.yml`'s `mutants-in-diff` — a mutation run rather than a parity
     measurement, and `continue-on-error` besides. Recorded as a decision, not
     a backlog entry: pairing it would double a job's cost to answer the same
     question twice.

   Every one of them is **pinned, checked and measured** — the release each
   installs is named in `docs/oracle_versions.tsv`, asserted at the path it was
   installed to, and selected for the tests that read it. What none has is a
   *second* leg, so a 3.2.0 divergence in the differential fuzz targets or in
   the corpus comparison is still unmeasured. Pairing the remaining three is a
   question of runner cost rather than of mechanism.
3. `references/libjpeg-turbo` stays at 3.1.90. Bumping it to 3.2.0 moves every
   `j*.c:NNN` citation in this repository and re-baselines the classic-ABI
   trace oracles at the same time, which is its own change with its own
   drift audit — not a line in this one.
4. Retiring the 3.1.4.1 leg is deliberately **not** scheduled: it is the
   behaviour-regression half of the pair, and it retires only when its
   expectations are known to hold on the newer leg.

## P4-131. No Native Binary Distribution — Releases Ship crates.io and npm Only — **PARTIAL: Unix bundles ship and are gated; Windows, signing/SBOM and the deb/rpm decision remain**

**GitHub:** [#462](https://github.com/developer0hye/libjpeg-turbo-rs/issues/462) — under the [#470](https://github.com/developer0hye/libjpeg-turbo-rs/issues/470) umbrella.

**Motivation.** Filed 2026-08-09 by the external drop-in readiness review.
`.github/workflows/release.yml` had six jobs: `changelog-check`, `publish`
(crates.io), `publish-capi` (crates.io), `publish-image` (crates.io),
`publish-wasm` (npm), and `github-release` (notes from CHANGELOG). **No job
produced a native binary artifact.** A user who wanted to replace their system
`libjpeg.so.8` had to clone the repository, install a Rust toolchain, build, and
run `scripts/install_capi.sh` themselves.

The install *layout* was not the gap — `scripts/install_capi.sh` already stages a
correct prefix: `lib/libjpeg.so.8*`, `lib/libturbojpeg.so.0*`,
`lib/pkgconfig/libjpeg.pc`, `lib/pkgconfig/libturbojpeg.pc`,
`lib/cmake/JPEG/JPEGConfig.cmake`, and `include/{jpeglib,jerror,jmorecfg,jconfig,turbojpeg}.h`.
The gap was that nothing ran it in CI to produce a downloadable artifact, so
the staged prefix was only ever built on a developer's machine.

Upstream, by contrast, ships signed source tarballs plus official binary
packages per platform, with published signature-verification instructions.
A project asking distributions to swap out their JPEG library is asking for a
higher bar than "build it yourself."

**Acceptance criteria.**

1. A release job builds and attaches, per tag: Linux `libjpeg.so.8` +
   `libturbojpeg.so.0` (x86_64 + aarch64), macOS dylibs, and Windows DLL +
   import libraries — each bundled with the headers, `.pc` files, and CMake
   config that `install_capi.sh` already stages.
2. Every attached artifact is checksummed, and the checksum manifest is part of
   the release.
3. The artifact is produced by the same `install_capi.sh` path that
   **[P4-124](#p4-124-the-opencv-harness-tests-the-cargo-cdylib-not-the-library-we-ship--open)**
   requires the downstream harnesses to test — one staging path, not two. This
   item and P4-124 must not produce divergent "shipped" artifacts.
4. A recorded decision on signing and SBOM: either implemented, or documented as
   a known gap with the reason. Do not leave it unstated.
5. Distro packaging (deb/rpm) is explicitly either in scope with a target
   release, or a recorded non-goal. It is currently neither.

**Why deferred.** Pure distribution work with no correctness content, and it is
wasted effort while the T3 classic-ABI gaps are open — shipping a convenient
binary of a library that is not yet a general drop-in increases the blast
radius of the gaps rather than reducing it. Sequenced in Stage C, after the
export surface (P4-129) and the shipped-artifact test path (P4-124) are settled,
since both change what a release artifact should contain.

**Status (2026-08-18): PARTIAL** — criteria 2 and 3 are met, criterion 1 is met
for Unix and open for Windows, criteria 4 and 5 remain.

*What landed.* `scripts/package_capi_release.sh` and a `native-artifacts`
matrix job in `release.yml` build and attach
`libjpeg-turbo-rs-capi-<version>-<target>.tar.gz` for
`x86_64`/`aarch64-unknown-linux-gnu` and `x86_64`/`aarch64-apple-darwin` on
every `v*` tag, and `github-release` folds the per-bundle sums into one
`SHA256SUMS` and attaches it with them (criterion 2). It runs *ahead* of every
publish job, which now need it: a crates.io upload cannot be withdrawn, so a
bundle that fails to build must fail before the irreversible step. The workflow
also gained `workflow_dispatch`, which runs that job alone — every publish job
is now additionally gated on `github.event_name == 'push'`, because
`publish-capi` and `publish-wasm` both accept a *skipped* upstream job and
would otherwise have published to crates.io and npm off a dispatch. A tag-shape
test would not have closed that: `gh workflow run --ref v1.2.3` dispatches with
`github.ref` still on the tag.

*Criterion 3, mechanically.* The packaging script stages nothing: it calls
`scripts/install_capi.sh` and archives its output, adding only a `BUNDLE.txt`.
`crates/libjpeg-turbo-rs-capi/tests/release_bundle.rs`
(`release_bundle_is_exactly_what_install_capi_sh_stages`) unpacks a bundle and
compares it against a direct install run entry by entry — paths, symlink
targets, permission bits and file bytes — so a second staging path fails rather
than diverges quietly. Its siblings assert the SONAME chains survive the
archive as symlinks resolving to a real ELF/Mach-O inside the bundle, that the
`.pc` files carry the requested prefix, that `sha256sum -c` accepts the archive
and *rejects* a byte-flipped one, and that `--target` reaches the nested build
— the release's cross-built legs would otherwise package the host library under
a cross target's name and every shape assertion would still pass.

*Three defects the pre-merge review caught, recorded because each was invisible
to the tests as first written.* (a) `install_capi.sh` relinked to a **fixed**
`…​.versioned` path under `RELEASE_DIR`, so two concurrent installs against one
target directory — which is what `cargo test` does the moment a second suite
stages the prefix — had `ld` truncate each other's output, and every existence
check still passed on the short result. It now links to a `mktemp` name and
removes it on exit. (b) The archive recorded the build runner's uid/gid, and
GNU tar extracting as root honours that: the documented `sudo cp -a` install
would have left `/usr/local/lib` owned by whoever holds uid 1001 on the target
host. It is now written `0:0` with blank user/group names. (c) MIT and
Apache-2.0 both require the notice to travel with a binary redistribution, and
nothing staged one — `install_capi.sh` now installs both texts to
`share/doc/libjpeg-turbo-rs-capi/`, which is where upstream libjpeg-turbo puts
its own, so every install gets them and not only the archive.

*Where it runs.* `capi-abi-checks` in `ci.yml` gained a step naming
`--test install_layout --test release_bundle`, so the gate runs on
`ubuntu-latest` and `macos-latest` (and skips with a reason on
`windows-latest`) for every pull request. That step is also the first CI
coverage `install_layout` has ever had: it has existed since P2-8 and no
workflow named it, so the layout gate that closed P2-8 ran on no pull request —
the same "a suite nothing names never runs" shape P4-81 and P4-61 each hit
before. The Linux release legs additionally install `patchelf` and fail if the
packaging log lacks `P4-81: relinked`, so neither of `install_capi.sh`'s two
warn-and-continue degradations can ship silently in a published bundle.

*What remains.*

1. **Windows (criterion 1).** No DLL or import library. `install_capi.sh` is
   Linux/macOS-only and the packaging script refuses to run elsewhere rather
   than emit an unverified shape. Windows needs its own layout decision — no
   SONAME chain, an import library, a toolchain-dependent `.pc` convention —
   so it is separate work, not another matrix row.
2. **Signing and SBOM (criterion 4).** Still a recorded gap, but the recorded
   *reason* changed: "nothing to sign yet" is spent. A checksum published
   beside the file it covers proves integrity, not origin. Closing it means
   Sigstore provenance (`actions/attest-build-provenance`) or detached
   signatures, and neither is observable from a pull request — it needs a
   dispatch run and then a real tag to verify, which is why it was not wired
   into the release path blind. `docs/RELEASE_ARTIFACTS.md` states the residual
   risk to a downloader.
3. **deb/rpm (criterion 5).** Unchanged and deliberately so: `ABI_COMPATIBILITY.md`
   already records it as a maintainer decision rather than a technical one, and
   an unattended session is not the place to make it. The tarballs do remove
   the technical obstacle — a packager now has everything `debian/rules` needs.
4. **A `capi-v*` tag builds bundles and attaches them to nothing.**
   `native-artifacts` runs there — that tag ships the very crate the bundle
   contains — but `github-release` is `refs/tags/v`-only, so the four archives
   expire as workflow artifacts. An ABI-shim release is exactly the one whose
   consumers want a binary. Either `github-release` learns the `capi-v*` shape
   or the bundle job stops running on it; `release.yml`'s `on:` comment points
   here. Left as it is rather than decided unattended, because the first option
   creates a release channel this project does not currently have and the
   second removes a gate on an irreversible publish.
5. **The archive name does not distinguish two releases carrying one capi
   version.** The stem is the *capi* crate's version, and the bundle is
   attached on *root* `v*` tags, so two root releases that do not bump the capi
   crate produce identically named archives — with different bytes, since the
   capi crate compiles against the root crate. `BUNDLE.txt` records the commit,
   so an unpacked bundle is identifiable; a downloads directory holding both is
   not. Fixing it means putting the release tag in the name, which changes the
   name a packager scripts against, so it belongs with (4).

## P4-132. Classic C-ABI Per-`cinfo` State Is Thread-Affine (P4-16 Option A) — **OPEN**

**GitHub:** [#463](https://github.com/developer0hye/libjpeg-turbo-rs/issues/463) — under the [#470](https://github.com/developer0hye/libjpeg-turbo-rs/issues/470) umbrella.

**Motivation.** Filed 2026-08-09 by the external drop-in readiness review, which
names this an adoption blocker for consumers that move codec-context ownership
between threads. **[P4-16](#p4-16-per-cinfo-private-state-lives-in-thread-local-side-tables--closed-2026-05-19)**
closed 2026-05-19 via Option B — document the constraint — and recorded its own
reopen trigger: *"file an issue with the use case … we will prioritise based on
adoption signal."* This entry is that reopen. P4-16 stays closed; the decision
it recorded was correct for the evidence available then, and this is a new
entry because the evidence changed.

**What changed since P4-16 chose Option B.**

1. **Upstream moved off TLS.** libjpeg-turbo 3.2 beta1 (note 2) overhauled its
   SIMD dispatchers to initialise per-instance rather than per-thread,
   explicitly "eliminating the need for thread-local storage in the libjpeg API
   library." P4-16's divergence paragraph is written against upstream's older
   contract; the gap is now wider than when it was measured.
2. **The named consumer exists.** `docs/ABI_COMPATIBILITY.md:79` names FFmpeg's
   frame-thread JPEG path as the canonical case that would force Option A, and
   the repository already carries `capi_ffmpeg_compat.rs` as a downstream
   harness. The prerequisite for prioritising is met.
3. **T3 is the goal.** A general system drop-in cannot ship a threading contract
   stricter than the library it replaces. Every prebuilt consumer on the system
   was compiled against upstream's rules, not ours.

**Current state.** Two `thread_local!` side tables in
`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` (`:396` decompress, `:4114`
compress) key private state by `cinfo as usize`. Transferring a `cinfo` to
another thread silently misses the lookup and leaks the originating thread's
entry. `docs/ABI_COMPATIBILITY.md:70,79` documents this as a deliberate,
stricter-than-upstream contract.

**Acceptance criteria.** P4-16 Option A already specifies the shape; this entry
adds what a pointer-keyed global map needs to be correct:

1. Both side tables migrate off `thread_local!` to a process-global map behind
   a lock. A `Mutex<HashMap>` alone is not sufficient — see (3).
2. A single-threaded `tj3Compress8` / `tj3Decompress8` benchmark stays within
   **1%** of the TLS-keyed baseline (P4-16's original bar), recorded in
   `experiments/`.
3. **Pointer reuse is defended.** Keying by `cinfo as usize` means a freed and
   reallocated `cinfo` can collide with a stale entry. Store a generation
   counter and a magic value in the private state, and make destroy/abort the
   single place ownership is released. A test must show that allocating a
   `cinfo`, destroying it, and allocating a second one that lands at the same
   address does not surface the first one's state.
4. `crates/libjpeg-turbo-rs-capi/tests/capi_thread_affinity.rs` proves
   create-on-thread-A / use-and-destroy-on-thread-B with no leak, and proves
   that *concurrent* use of one `cinfo` from two threads is still rejected or
   documented — ownership transfer is the goal, shared concurrent access is not.
5. `docs/ABI_COMPATIBILITY.md`'s "Threading contract" and "Divergence from
   upstream" sections are rewritten to the new contract. If any strictness
   remains, it is stated as strictness, not omitted.

**Why deferred.** Behind the Stage A memory-safety and error-contract items: no
current test or downstream harness in this repository transfers a `cinfo` across
threads, so nothing is broken today for what we actually measure. It is a
correctness-of-contract gap that blocks the T3 claim, not a live defect.

## P4-133. BMI2/FMA Paths Are Reachable Only via `target-cpu=native`, So Portable Builds Leave Them Off — **OPEN**

**GitHub:** [#464](https://github.com/developer0hye/libjpeg-turbo-rs/issues/464) — under the [#470](https://github.com/developer0hye/libjpeg-turbo-rs/issues/470) umbrella.

**Motivation.** Filed 2026-08-09 by the external drop-in readiness review.
**[P4-8](#p4-8-runtime-bmi1lzcnt-dispatch-for-x86_64-encode-already-live-readme-updated--closed-2026-05-17)** closed 2026-05-17 after establishing that the BMI1/LZCNT AC
encoding loop already dispatches at runtime
(`src/encode/huffman_encode.rs:524,598`), so a stock `cargo build --release` is
within ~2 pp of C. That closure recorded an explicit follow-up
(`phase4.md:233`): *"BMI2 PEXT/PDEP coverage for any encode hot path that
benefits + FMA-dispatched FDCT scalar fallback. The static-analysis review
correctly notes these remain `target-cpu=native`-gated today."*

**That follow-up was never filed anywhere.** It says "deferred to P2 backlog",
but `docs/last_mile/backlog.md` contains exactly one section, P2-G, which is the
downstream lab. By this repository's own rule — if it is not in LAST_MILE, it
does not exist for the next session — the deferral was a silent drop. This entry
restores it.

**Why it matters for T3 specifically.** `README.md:94,113` still recommends
`RUSTFLAGS="-C target-cpu=native"` for the last few percent. That is sound advice
for an application built on the target machine, and unusable for a *system
library*: a distribution package is built once and runs on every CPU of that
architecture, so it must be compiled to the baseline and light up wider
instruction sets at runtime. Every percent that only `target-cpu=native` unlocks
is a percent a packaged `libjpeg.so.8` cannot have. C libjpeg-turbo has no such
constraint — its hot loops are hand-written NASM with the instructions embedded,
dispatched at runtime, which `docs/last_mile/phase1.md:217` already identifies as
the root of the original gap.

**Acceptance criteria.**

1. The `target-cpu=native`-only wins are enumerated and measured against a
   baseline build: BMI2 PEXT/PDEP in the encode hot paths, FMA in the FDCT
   scalar fallback, and anything else the A/B surfaces. State the per-benchmark
   delta; do not carry "the last few percent" forward as an unmeasured claim.
2. Each one that pays is reached by runtime detection from a **baseline**
   build, following the existing `cpu_has!` dispatch pattern.
3. Feature detection is resolved **once per operation**, not per block or per
   row — resolve it where the encode/decode plan is built and store the chosen
   kernel set. (This is workstream 2 of
   **[P4-123](#p4-123-architecture-umbrella-codec-plans-c-abi-state-public-boundaries-simd-dispatch--open)**;
   coordinate rather than building a parallel dispatch mechanism. Note that
   upstream 3.2 made exactly this move — per-instance SIMD dispatchers, beta1
   note 2.)
4. A benchmark run comparing **stock `cargo build --release`** against C
   libjpeg-turbo, recorded in `experiments/encode.tsv` per the keep/discard
   protocol. The portable build is the number that matters for T3; the
   `target-cpu=native` figure is supplementary.
5. `README.md`'s performance section separates portable-build numbers from
   native-tuned numbers, so a packager reads the one that applies to them.

**Why deferred.** Performance, and gate item 7 puts correctness first. It is
filed now because the deferral was previously untracked, not because it
outranks the Stage A items.

## P4-134. No RISC-V RVV SIMD Backend — Upstream 3.2 Ships One — **OPEN**

**GitHub:** [#465](https://github.com/developer0hye/libjpeg-turbo-rs/issues/465) — under the [#470](https://github.com/developer0hye/libjpeg-turbo-rs/issues/470) umbrella.

**Motivation.** Filed 2026-08-09 by the external drop-in readiness review.
`src/simd/` contains `aarch64/`, `wasm32/`, `x86_64/`, and `scalar.rs` — there is
no RISC-V backend, and `grep -ri "riscv\|rvv" src/` matches only two comment
lines in `src/decode/color.rs` referencing the riscv64 scalar experiment.

libjpeg-turbo 3.2 beta1 (note 6) added RVV implementations of colorspace
conversion, chroma up/downsampling, integer quantization and sample conversion,
and the integer DCT/IDCT — reporting **149-246% faster compression** and
**48-180% faster decompression** relative to 3.1.x on a Ky X1.

**This invalidates a premise of an existing item.**
**[P4-60](#p4-60-scalar-kernels-are-25x-slower-than-cs-scalar-kernels--open)**
measured our scalar deficit on riscv64 precisely because *neither side* had SIMD
there, making it a clean scalar-vs-scalar comparison. Against 3.2 that is no
longer true: on RVV hardware we would be scalar against vectorised, which is the
same structural position as
**[P4-78](#p4-78-no-32-bit-arm-aarch32-neon-backend--armv7-is-our-widest-gap-vs-c--open)**
describes for ARMv7 — the scalar deficit multiplying with C's full vector win.
P4-60's riscv64 measurements stay valid as *scalar-kernel* data; they stop being
a statement about our position versus current upstream on that architecture.

**Acceptance criteria.**

1. A measurement, on RVV hardware or a vector-capable emulator, of our decode
   and encode against libjpeg-turbo **3.2.0** on riscv64 — establishing the real
   gap rather than inferring it. This is the gating criterion; the rest depends
   on what it shows.
2. P4-60's entry is annotated with the premise change so its riscv64 numbers are
   not later read as a current-upstream comparison.
3. A recorded scope decision. `core::arch::riscv64` vector intrinsics are
   unstable, which is the same constraint that made P4-78 an options list rather
   than a plan — check the current status rather than assuming it, and if it
   still holds, say what that implies (stable-Rust autovectorisation with an
   explicit target-feature, `asm!`, or defer).
4. If deferred after measurement, the deferral names the trigger that would
   reopen it, in the shape P4-78 uses.

**Why deferred.** Lowest-urgency of the platform items: RISC-V has the smallest
installed base of the architectures we target, and unlike P4-78 (where ARMv7
hardware is everywhere and the gap is inferred from a real user question) there
is no downstream request. It is filed so the P4-60 premise change is on record.

## P4-135. Public Safe SIMD Wrappers Let Safe Rust Reach `target_feature` Kernels With Unvalidated Slices — **CLOSED 2026-08-13**

**GitHub:** [#474](https://github.com/developer0hye/libjpeg-turbo-rs/issues/474) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** This is the first *confirmed unsound safe API* in the crate:
safe Rust, with no `unsafe` block anywhere in the caller, can invoke an AVX2
kernel with empty slices and an arbitrary `width`. It is not a
theoretical reachability argument — it was **compiler-verified**. This probe
builds clean against `x86_64-apple-darwin`:

```rust
fn main() {
    let mut out = [0u8; 4];
    // No `unsafe` in this function.
    libjpeg_turbo_rs::simd::x86_64::avx2_color::avx2_ycbcr_to_rgb_row(
        &[], &[], &[], &mut out, 4096,
    );
}
```

*(Compile-checked only. It was deliberately never executed.)*

**Root cause.** The wrapper generated by `avx2_color_convert_fn!`
(`src/simd/x86_64/avx2_color.rs:192-240`) is **safe**, and its safety comment
states a precondition it does not check:

```rust
/// # Safety contract
/// Caller must ensure AVX2 is available (dispatch verifies this).
pub fn $pub_name(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
    // SAFETY: AVX2 availability guaranteed by dispatch.
    unsafe { $inner_name(y, cb, cr, out, width); }
}
```

"Dispatch verifies this" is true of *our* call sites and false of the function's
actual contract. A safe `pub fn` may not assume anything about its caller. Two
independent UB routes follow:

1. **Out-of-bounds.** `width` is a parameter separate from the slice lengths.
   The inner loop runs `while x + 16 <= width` doing
   `_mm_loadu_si128(y.as_ptr().add(x) as *const __m128i)` and storing through
   `out.as_mut_ptr().add(x * $bpp)` — no length check against `y`/`cb`/`cr`/`out`.
   (The scalar tail *does* slice, so it would panic — but only after the SIMD
   loop already read and wrote out of bounds.)
2. **Missing CPU feature.** Calling a `#[target_feature(enable = "avx2")]`
   function without AVX2 actually being available is UB per the Rust reference.
   The safe wrapper performs no `is_x86_feature_detected!` check, so a safe
   caller on a pre-Haswell CPU violates it.

**Scope — this is a pattern, not one function.** `avx2_color_convert_fn!` is
invoked 10 times (`:243-339`). The same "safe `pub fn` in front of a
`target_feature` kernel" shape appears across the SIMD tree, including
`avx2_color_encode.rs` (4), `wasm32/color.rs` (4), `wasm32/color_encode.rs` (4),
`x86_64/upsample.rs` (3), `x86_64/avx2_upsample.rs` (3), `wasm32/upsample.rs` (3),
`aarch64/idct_scaled.rs` (3), and further sites in `avx2_merged`, `merged`,
`downsample`, `idct`, `avx2_idct`, `avx2_fdct`, and the `mod.rs` dispatchers.

Reachability comes from the module tree being public at every level:
`src/lib.rs:128` `pub mod simd;` → `src/simd/mod.rs:13` `pub mod x86_64;`
(gated on `target_arch` only, **not** on `feature = "simd"`) →
`src/simd/x86_64/mod.rs:6` `pub mod avx2_color;`.

**`SimdRoutines` has the same hazard, but not uniformly.** Its fields are public
safe `fn` pointers (`src/simd/mod.rs:19-37`). The three IDCT fields are sound by
construction — `fn(&[i16; 64], &[u16; 64], &mut [u8; 64])` encodes every length
in the type. The other two do not: `ycbcr_to_rgb_row` takes `width` alongside
the slices, and `fancy_upsample_h2v1` documents "Output length must be
`in_width * 2`" in prose. Those two carry hidden preconditions on a safe type.

**Acceptance criteria.**

1. No safe function reachable from outside the crate can cause UB through
   argument choice alone. Kernels become `pub(crate) unsafe fn` or private
   `unsafe fn`; the safe entry points validate first.
2. `libjpeg_turbo_rs::simd::*` no longer resolves from an external crate. Add a
   `trybuild`/compile-fail regression so re-publishing the path fails CI.
3. Every safe entry point validates, before dispatch: each input slice holds at
   least `width` samples; the output holds at least `width * bytes_per_pixel`
   computed with **checked** arithmetic; and the CPU feature is confirmed by
   runtime detection in the same function that performs the `unsafe` call.
4. Length preconditions are removed from `SimdRoutines`' safe fn-pointer types —
   by fixed-size array parameters where the shape allows, by making the fields
   `pub(crate)`, or by making the pointers `unsafe fn`. Record which, per field;
   the IDCT fields need no change.
5. Architecture backends compile only under `feature = "simd"`, not on
   `target_arch` alone.
6. The blanket `#[allow(unsafe_op_in_unsafe_fn)]` on `pub mod simd`
   (`src/lib.rs:127`) is removed. **This is a consequence of the work, not the
   work itself** — see the note below.
7. Every remaining `SAFETY` comment on a SIMD kernel states CPU feature, input
   bounds, output bounds, pointer-range/`isize::MAX`, alignment, and
   aliasing — not "guaranteed by dispatch".

**Relationship to [P4-69](#p4-69-simd-feature-contract-and-the-remaining-389-safety-posture-work--open).**
P4-69 tracks the lint posture: module-level feature enforcement, the ~780-site
`unsafe {}` sweep, and the `forbid(unsafe_code)` goal. This item is **not** that
sweep and outranks it. Annotating 780 operations with `unsafe {}` and a comment
changes no behaviour and would leave this hole exactly as it is; closing the
public safe-to-UB path is what matters. P4-69's criteria 1 and the module-gating
work overlap and should be executed together, with this item leading.

**Why P0.** It is the difference between "we use `unsafe` carefully" and "our
safe API is sound". Until it is closed, no memory-safety claim about the Rust
API is defensible, and the README's framing (P4-140) is unsupportable.

**Status (2026-08-10): partial — the soundness hole is closed, the hygiene
sweep is not.** No safe path from outside the crate reaches a SIMD kernel with
unchecked lengths any more. Two separate routes had to be shut:

* *By path* — closed earlier: the arch modules became `pub(crate)`, guarded by
  `tests/simd_module_privacy.rs`. This is what the issue's opening
  proof-of-concept used.
* *By table* — closed here. Module privacy alone left `simd::detect()` public
  and returning a struct whose safe `fn`-pointer fields were `pub`, so the
  identical UB was still reachable with no `unsafe` at the call site:

  ```rust
  let r = libjpeg_turbo_rs::simd::detect();
  (r.ycbcr_to_rgb_row)(&[], &[], &[], &mut [0u8; 4], 4096);
  ```

  That this compiled was confirmed, not inferred: the first run of the new test
  failed with `error[E0599]: no method named 'ycbcr_to_rgb_row' … field, not a
  method`, i.e. rustc resolving the `pub` field from an external test crate.

Criteria 1–5 are met (5 as of 2026-08-13, with P4-143):

| # | Criterion | Disposition |
| --- | --- | --- |
| 1 | No externally-reachable safe fn can cause UB by argument choice | Met for the public surface. The *sub-clause* "kernels become `unsafe fn`" is not done — see below |
| 2 | `simd::*` no longer resolves externally | Met earlier; `tests/simd_module_privacy.rs` |
| 3 | Entry points validate lengths + `checked` byte counts before dispatch | Met — `require_samples` / `require_bytes` in `src/simd/mod.rs` |
| 4 | Length preconditions removed from `SimdRoutines` fn-pointer types | Met, recorded per field below |
| 5 | Arch backends compile only under `feature = "simd"` | Met 2026-08-13 with P4-143 — see below |

**Criterion 5's history.** The first attempt was backed out: adding
`feature = "simd"` to the module `cfg`s compiled clean everywhere — but only
because `.cargo/config.toml` forces `+simd128` on both wasm targets, and
narrowing the gates without aligning the disagreeing call sites was a hidden
`E0433` for anyone building baseline `wasm32`. That regression was caught by
review, not by any build or CI job, and the masking was filed as P4-143.

The second attempt (2026-08-13) landed it in the order the back-out
prescribed: every call site aligned to its arch's canonical predicate first
(13 pipeline sites and 9 `src/simd/*_tests.rs` files were missing a
condition; `detect`/`detect_encoder`'s wasm arms too), then the module gates
narrowed — wasm32's requiring `simd128` as well, since its kernels cannot
exist in a baseline module — with P4-143's `Check baseline wasm32` CI leg
proving the scalar fallback compiles warning-free without `+simd128`
(verified red against the pre-alignment tree: 67 dead-code errors). See the
P4-143 closure for the full delivery.

Per-field disposition for criterion 4:

| Field | Disposition | Why |
| --- | --- | --- |
| `idct_islow`, `idct_ifast`, `idct_float` | `pub`, unchanged | `fn(&[i16; 64], &[u16; 64], &mut [u8; 64])` — every length is in the type |
| `fdct_quantize` | `pub`, unchanged | Same: fixed-size arrays plus `&QuantDivisors` |
| `ycbcr_to_rgb_row` | `pub(crate)` + validating method | `width` is independent of four slice lengths |
| `fancy_upsample_h2v1` | `pub(crate)` + validating method | Prose-only "output must be `in_width * 2`" |
| `rgb_to_ycbcr_row` | `pub(crate)` + validating method | `width` independent of four slice lengths |

Proof: `cargo test --test simd_dispatch_bounds` — 13 tests, including the
issue's verbatim proof-of-concept now panicking instead of reading out of
bounds, short-input/short-output cases for all three entry points, and
`width * 3` overflow rejected by `checked_mul` rather than wrapping into a
bound a short buffer satisfies.

**What remains.** In-crate hygiene, not reachability:

* The kernel wrappers inside the arch modules are still safe `pub(crate) fn`
  fronting `target_feature` bodies. A *crate-local* caller can therefore still
  misuse one without writing `unsafe`. No such call site exists today, and none
  is reachable from outside the crate, so this is defence-in-depth rather than
  a live defect — but criterion 1's "kernels become `pub(crate) unsafe fn`"
  sub-clause is genuinely not done. The direct-call surface that sweep must
  annotate was measured 2026-08-13 (review recount): 129 pipeline call sites
  — `decode/pipeline_impl/color.rs` 39, `encode/pipeline_impl/mcu.rs` 40,
  `optimized.rs` 12, `sampling.rs` 9, `dispatch.rs` 9, `baseline.rs` 8,
  `progressive_entropy.rs` 5, `api/progressive_output.rs` 7 — plus the
  `src/simd/*_tests.rs` callers.
* Criterion 6 (`#[allow(unsafe_op_in_unsafe_fn)]` on `pub mod simd`) and
  criterion 7 (SAFETY-comment content) are the ~630–679-site sweep measured in
  `src/lib.rs`. The issue itself calls criterion 6 "a consequence of the work,
  not the work itself".

This residue belongs with
[P4-69](#p4-69-simd-feature-contract-and-the-remaining-389-safety-posture-work--open),
which already tracks the sweep and the `forbid(unsafe_code)` goal. P4-135 keeps
the row until the kernel signatures change; it is no longer P0, because the
property it was P0 for — "our safe API is sound" against this defect — now
holds.

**Status (2026-08-13): closed.** The "kernel signatures" residue above was
re-audited by classifier (every safe wrapper in the three arch modules,
checked for *executable* validation rather than comment claims) and turned
out to describe an architecture that had already moved on: the inner
`target_feature` kernels are `unsafe fn`, and nearly every safe wrapper
already validates lengths and CPU feature itself with a scalar fallback —
which is precisely criterion 1's "the safe entry points validate first".
The "~129 direct call sites need `unsafe` annotation" framing was wrong:
those sites call the validating wrappers, and annotating them would have
re-implemented the wrappers' job at every caller.

What the audit *did* find — nine wrappers whose stated contracts were
comment-only (eight by the classifier, the ninth — `neon_fancy_h2v2_row`
— by the review's independent sweep), all fixed here. A tenth,
`avx2_fdct_islow`, carried the same "Caller must ensure AVX2" contract on
a safe `pub fn` and is fixed as a knock-on of hardening its only caller
(the composite below) rather than counted with the nine:

* **x86_64, feature precondition unchecked (UB hazard):**
  `avx2_idct_islow`, `sse2_idct_islow`, and the `avx2_fdct_quantize`
  composite carried "verified at dispatch time" SAFETY comments — the
  filing's own root-cause premise ("a safe `pub fn` may not assume
  anything about its caller") one module deeper. Each now checks
  `cpu_has!` itself and falls back to the scalar reference.
  `avx2_fdct_islow` has no same-shape scalar (the scalar FDCT emits
  `i32`), so it became `pub(crate) unsafe fn` with the precondition in
  its signature; its one caller sits inside the composite's checked arm.
* **wasm32, length preconditions unchecked (OOB hazard):** the four
  `wasm_*_to_ycbcr_row` encode wrappers said "Caller guarantees …" and
  checked nothing; they now validate with `checked_mul` and fall back to
  scalar, mirroring their AVX2 twins. `wasm_fancy_upsample_h2v1` was
  sound only because its edge-pixel indexing happened to probe both
  bounds; it now asserts them explicitly. (simd128 needs no runtime
  check — the module only compiles where the target feature is
  statically enabled, per the P4-143 gating.)
* **aarch64, one length-unvalidated row kernel (OOB hazard):**
  `neon_fancy_h2v2_row` probed only `output[0]`/`output[1]` before its
  inner wrote `output[..in_width * 2]` by raw pointer, and its caller
  never checks `out_width >= in_width * 2` — found by the review, since
  the classifier's incidental-indexing heuristic scored it validated. It
  now runs the same `fits`-then-fallback shape as its SSE2/AVX2/wasm
  twins.
* **The rest need no change:** NEON is mandatory and simd128 is
  compile-time, so the fixed-array wrappers (lengths in the types) have
  no unchecked precondition, and the slice-taking aarch64
  downsample/upsample-h2v1 wrappers already carry explicit `assert!`s —
  sound by executable check, not by type.

Proof: `simd_x86_tests::feature_checked_wrappers_match_scalar_on_any_cpu`
— deliberately unguarded by any feature probe, so whichever arm the
executing host takes must equal scalar. Both arms execute in CI, and the
`--lib` that makes that true was added here: the AVX2 leg's
`cargo test --tests` runs the SIMD arms, and the CPUID-masked no-AVX2 leg
(#320) — which built four *integration* binaries and not `--lib` — now
carries the lib binary too, so the AVX2 wrappers take their new fallback
arms on a CPU that genuinely lacks the feature. Read that arm for what it
proves: the fallback calls the same `scalar_idct_islow` /
`scalar_fdct_quantize` the expectation uses, so its equality holds by
construction, and what the leg pins is that the wrapper *takes* the
fallback instead of entering an AVX2 kernel without AVX2.
`sse2_idct_islow` still runs its SIMD arm there, SSE2 being x86_64
baseline. That is the x86 half of P4-141 criterion 2 for the lib suite.
Inputs stay inside the parity
suite's documented bit-exactness envelope (coeffs [-128, 127] × quant
[1, 8]) — the first draft generated ±256 × [1, 31], where scalar
truncates its pass-1 workspace to i16, SSE2 keeps i32 and AVX2
saturates, and the review *ran* it under Rosetta and watched the SSE2
assertion fail 32/64 bytes; parity outside the envelope is a property
the kernels never promised (the P4-19/P4-20 family). The wasip1 suite
runs the hardened wasm wrappers through the existing parity tests
(226/0) — exact-fit slices only, so the new `fits == false` fallback
arm has no executing coverage there, and the short-slice panic arm is
not assertable under wasip1 (that target is `panic = "abort"`, which is
why `avx2_color.rs`'s `p4135_soundness_tests` is gated on
`panic = "unwind"`); for those, the checks' code is the pin.

Criteria 6–7 (the `unsafe_op_in_unsafe_fn` lift and the SAFETY-prose
sweep) remain with
[P4-69](#p4-69-simd-feature-contract-and-the-remaining-389-safety-posture-work--open),
exactly as the paragraph above records — the issue's own text calls
criterion 6 a consequence of that sweep, not of this item.

## P4-136. Progressive Output Calls `set_len()` on Uninitialized `Vec` After an Unchecked Size Multiplication — **CLOSED 2026-08-10**

**GitHub:** [#475](https://github.com/developer0hye/libjpeg-turbo-rs/issues/475) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** `src/api/progressive_output.rs:256-262` allocates each component
plane by declaring uninitialized memory initialized:

```rust
let size: usize = ci.comp_w * ci.blocks_y * block_size;
let mut v: Vec<u8> = Vec::with_capacity(size);
#[allow(clippy::uninit_vec)]
unsafe { v.set_len(size) };
v
```

`Vec::set_len`'s documented contract is that the elements up to the new length
are already initialized. This does the opposite: it sets the length *first* and
relies on the IDCT to fill every byte afterwards.

**Two defects, not one.**

1. **Uninitialized `Vec<u8>` exposed to safe code.** Between `set_len` and the
   last IDCT store, a `Vec<u8>` whose contents are undefined is reachable. The
   "every byte gets written exactly once" argument is a *global* invariant over
   block iteration, component geometry, and every early-return path — it is not
   checked, and any future `?`, `break`, or panic between the two points leaks
   uninitialized bytes into safe code. LLVM is also entitled to optimise on the
   assumption the contract held.
2. **The size is computed with unchecked multiplication.**
   `ci.comp_w * ci.blocks_y * block_size` comes from attacker-influenced header
   geometry. In release builds this wraps; a wrapped-small `size` then produces a
   short allocation that the IDCT writes past. This half is a plain
   memory-safety bug independent of the `set_len` question.

**Scope.** Six `set_len` sites in this file — `:260`, `:703`, `:735`, `:736`,
`:809`, `:858` — plus three in `src/encode/pipeline_impl/progressive_entropy.rs`
(`:65`, `:96`, `:145`) that write into `Vec` spare capacity through raw pointers
and then re-`set_len`. Audit all of them; they are the same shape.

**Acceptance criteria.**

1. Plane sizes are computed with `checked_mul` and rejected as a typed error
   (`DimensionOverflow`) rather than wrapping. The result is also checked against
   `isize::MAX`.
2. The default allocation is zero-initialized (`vec![0u8; size]`). Any retained
   uninitialized path must use `MaybeUninit` with a written invariant, not
   `set_len` on a `u8` `Vec` — and only after (3) shows it is needed.
3. A benchmark comparing zero-init against the current code on the progressive
   decode path, recorded in `experiments/` per the keep/discard protocol.
   **Measure before optimising**: `calloc`-backed zero pages are frequently free
   for large planes, so the current pattern may be buying nothing.
4. Allocation is fallible where the size is input-derived — `try_reserve_exact`
   with a typed `AllocationFailed` error rather than an abort.
5. A 32-bit-target regression test (`i686`) covering the geometry that overflows
   `usize` there but not on 64-bit.
6. Miri covers the progressive output path. It currently does not — see P4-141.

**Status (2026-08-10): partial — criteria 1 and 2 met; the memory-safety content
is closed.** Audited against `main` @ 93b74c4.

`grep -c set_len src/api/progressive_output.rs` returns **0**. All six sites the
item names are now `vec![0u8; size]`, sized by a checked helper:

```rust
fn checked_plane_size(factors: &[usize], what: &'static str) -> Result<usize> {
    let mut total: usize = 1;
    for factor in factors {
        total = total.checked_mul(*factor).ok_or(JpegError::LimitExceeded { .. })?;
    }
    if total > isize::MAX as usize { return Err(JpegError::LimitExceeded { .. }); }
    ...
}
```

Both described defects — an uninitialised `Vec<u8>` reachable from safe code,
and the wrapped-small allocation the IDCT then writes past — are gone.

This item is no longer a soundness blocker for the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Status (2026-08-10): CLOSED — criteria 3–6 delivered.** The three `set_len`
sites in `src/encode/pipeline_impl/progressive_entropy.rs` (`:65`, `:96`,
`:145`) stay out of scope and move to
[P4-139](#p4-139-memory-layout-arithmetic-is-decentralised-and-uses-saturatingunchecked-multiplication--partial-every-span-is-checked-and-the-rule-is-enforced-centralisation-and-scalingfactor-outstanding)
as recorded below.

* **Criterion 3 — met.** `experiments/progressive.tsv` records the zero-init
  comparison (8K progressive fixture, best-of-7, `examples/p4136_prog_bench.rs`).
  Two rows now: the original uninit-vs-zero-init pair, and this change's
  calloc-vs-`try_reserve_exact` pair. **The originally recorded +4.7% zero-init
  cost did not reproduce**: `main` measures 52.30 / 52.05 ms today, which is that
  row's own *uninit* figure. Treat the +4.7% as unconfirmed — on this host
  zero-init is free, exactly as the criterion's `calloc` hint predicted.
* **Criterion 4 — met for the geometry-sized allocations; metadata copies split
  out to [P4-144](#p4-144-metadata-copies-are-input-sized-but-still-allocate-infallibly--closed-2026-08-12).**
  Every allocation in `src/api/progressive_output.rs` whose size comes from
  *header geometry* goes through `try_filled_vec`, `try_reserved_vec` or
  `try_copy_of`, which use `try_reserve_exact` and report refusal as
  `JpegError::AllocationFailed { what, bytes }` instead of aborting the process.

  That boundary is the point of the item, not a convenience: geometry-sized
  allocations are **amplifying** — a 300-byte SOF can demand 8 GiB — whereas the
  metadata copies (`icc::reassemble_icc_profile`, the `saved_markers` / `xmp` /
  `exif` clones) are bounded by an input the caller already holds in memory.
  Making those fallible means threading `Result` through
  `reassemble_icc_profile`'s 11 call sites and `Image` construction in *both*
  decoders, which is a different change from this item's title; P4-144 owns it.
  Found by the codex review of this fix.

  The geometry set covers the six plane/output sites the item names plus three
  the original scope missed:
  * `coeff_bufs` and `ac_max_k_bufs`, whose `blocks_x * blocks_y` was *also* an
    unchecked product, and which at 128 bytes per block are the largest buffers
    this decoder holds;
  * `assemble_grayscale`'s output raster, which was
    `Vec::with_capacity(out_width * out_height)` — unchecked *and* infallible.
    Found by the codex review of the first version of this fix, which correctly
    refused the "every allocation is fallible" claim while one was not.

  Measured neutral (52.24 / 52.23 ms). The verification cost one CI change:
  ASan aborts an unservable request with `allocation-size-too-big` before
  `try_reserve_exact` can return, so `sanitizers.yml` now sets
  `ASAN_OPTIONS=allocator_may_return_null=1` — which makes ASan's allocator
  behave like a real one. This does not weaken the job: an *infallible*
  allocation handed a null still aborts through `handle_alloc_error`.
  Reproduced both ways locally (SIGABRT without the option, 7/7 with it).
* **Criterion 5 — met.** `thirty_two_bit_geometry_overflow_is_rejected` and
  `plane_size_overflow_is_rejected_on_every_pointer_width` pin the width-split.
  The named target is `i686`, which this repo has no leg for; the coverage comes
  from `armv7.yml` (a real 32-bit ISA, `--lib` under qemu with
  `-C overflow-checks=on`) and `wasm.yml`'s `wasm32-wasip1` — both run `--lib`,
  which is where these tests live. Only the 32-bit arm calls `try_filled_vec`,
  because only there is the answer decided by arithmetic *before* any
  allocation; the 64-bit arm stops at the arithmetic deliberately, since calling
  through would request 8 GiB.
* **Criterion 6 — met.** `progressive_output_path_is_miri_covered` decodes the
  522-byte `blue_16x16_420_prog.jpg` (embedded via `include_bytes!`, so no
  filesystem access is needed) scan by scan under `cargo miri test --lib`.
  Measured at **4.8 s** inside the job's 25-minute budget. This is the criterion
  P4-141 was pointed at: Miri ran `--lib` only, the progressive suites are
  `tests/` integration targets, so the `set_len`-on-uninitialised-`Vec` pattern
  survived a green Miri job. One sibling test is `#[cfg_attr(miri, ignore)]`:
  Miri reports an unservable allocation as *resource exhaustion* and aborts the
  interpreter rather than returning the null `try_reserve_exact` turns into
  `Err`, so the refusal path is not observable there.

**Why P0 (historical).** (2) was exploitable from a crafted header on its own,
and (1) was a documented `unsafe` contract violation sitting in the crate's own
code. Both were fixed by the criteria 1–2 work above.

## P4-137. C-ABI Raw-Pointer Exports Are Safe Rust Functions — **CLOSED 2026-08-11**

**GitHub:** [#476](https://github.com/developer0hye/libjpeg-turbo-rs/issues/476) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** `crates/libjpeg-turbo-rs-capi` builds as
`crate-type = ["rlib", "cdylib", "staticlib"]` (`Cargo.toml:16`), so its Rust
signatures are a real Rust API, not only a C symbol table. Those signatures
declare raw-pointer entry points **safe**:

- `pub extern "C" fn tj3Free(ptr: *mut c_void)` (`src/alloc.rs:72`) passes an
  arbitrary caller-supplied pointer to `free()`. Safe Rust can hand it a
  dangling pointer, a stack address, or a pointer already freed.
- `pub extern "C" fn tj3Destroy(handle: *mut c_void)` (`src/tj3.rs:187`)
  reconstructs ownership with `Box::from_raw`. Two safe calls with the same
  handle are a double free.

Both are reachable from safe Rust with no `unsafe` block. The crate suppresses
the lint that would flag exactly this, crate-wide:
`#![allow(clippy::not_unsafe_ptr_arg_deref)]` (`src/lib.rs:17`).

**The stated reason for the suppression does not hold.** The comment justifies
it on the grounds that making the exports `unsafe fn` would change the
ABI-visible symbol. It would not: `extern "C"` fixes the calling convention and
`#[no_mangle]` fixes the symbol name, while `unsafe` only adds an obligation for
*Rust* callers. `pub unsafe extern "C" fn` emits a byte-identical C symbol.

**`handle_as_mut` additionally forges an unbounded lifetime**
(`src/tj3.rs:127-133`):

```rust
pub(crate) unsafe fn handle_as_mut<'a>(handle: *mut c_void) -> Option<&'a mut TjInstance> {
    if handle.is_null() { None } else { unsafe { Some(&mut *(handle as *mut TjInstance)) } }
}
```

`'a` is chosen by the caller, so the borrow checker will not constrain the
reference to the call. Its doc names validity and non-destruction but omits
exclusivity, cross-thread non-concurrency, reentrancy (a C callback re-entering
on the same handle), and alignment — so two live `&mut TjInstance` to one
instance are constructible without tripping any check.

**Memory spans are computed with saturating arithmetic.**
`decompress.rs:148` does `effective_pitch.saturating_mul(h)` to size an output
slice, and `precision.rs` repeats the shape at `:103`, `:274`, `:351`, `:492`.
Saturation converts an overflow into `usize::MAX` instead of an error, which is
the worst option for a value about to bound a `from_raw_parts_mut` — it must be
a typed error. Tracked jointly with P4-139.

**Acceptance criteria.**

1. Every export that dereferences a raw pointer or builds a slice from one is
   `pub unsafe extern "C" fn`. Confirm with a symbol diff that the exported C
   names and the ABI are unchanged.
2. `#![allow(clippy::not_unsafe_ptr_arg_deref)]` is deleted; any remaining
   suppression is per-function with a justification.
3. Each such export carries a `# Safety` section stating pointer validity,
   minimum buffer size, alignment, ownership transfer, aliasing, and threading.
4. `handle_as_mut` is replaced by a helper that confines the borrow to a
   closure, so no caller can name the lifetime:
   `unsafe fn with_handle<R>(h: *mut c_void, f: impl FnOnce(&mut TjInstance) -> R) -> Option<R>`.
5. Slice construction happens only after `checked_mul`/`checked_add` and an
   `isize::MAX` bound.
6. A **recorded decision** on handle hardening: generation-tagged handles in a
   registry to detect double-destroy and stale use, and a busy flag or lock so a
   concurrent same-handle call returns an error rather than aliasing `&mut`.
   Weigh against P4-131's threading work — these touch the same state.
7. Documentation states the boundary plainly: arbitrary invalid pointers from C
   are **not** defended against, and cannot be. The guarantee is that a
   *malformed JPEG* cannot corrupt memory when the caller honours the pointer
   contract.

**Why P0.** Not because C callers are endangered — a C caller was always
responsible for its pointers — but because the Rust-visible signature currently
tells the compiler these are safe, which is false, and `rlib` consumers get no
warning.

**Status (2026-08-10): partial — criterion 4 done.** `handle_as_mut` is gone.

The defect it carried was that its lifetime was caller-chosen:

```rust
pub(crate) unsafe fn handle_as_mut<'a>(handle: *mut c_void) -> Option<&'a mut TjInstance>
```

`'a` appeared in no input, so the borrow checker had nothing to tie the
reference to. Two calls on one handle produced two simultaneously-live
`&mut TjInstance` — aliasing UB, with no diagnostic and nothing in the type
system to flag it.

The replacement owns the lifetime instead of exporting it:

```rust
pub(crate) unsafe fn with_handle<R>(
    handle: *mut c_void,
    f: impl FnOnce(&mut TjInstance) -> R,
) -> Option<R>
```

All **30 call sites across nine modules** (`tj3`, `header`, `compress`,
`decompress`, `transform`, `precision`, `imageio`, `yuv`, `legacy`) now go
through it. Each entry point keeps its own sentinel via `.unwrap_or(-1)` /
`.unwrap_or(0)` / `.unwrap_or(null_mut())`, so NULL-handle behaviour is
unchanged.

One structural detail worth keeping: each body is bound to a `let body = |inst:
&mut TjInstance| -> T { … }` *outside* the `unsafe` block, rather than written
inline as `unsafe { with_handle(handle, |inst| …) }`. Wrapping the whole body in
one `unsafe` block would nest every pre-existing inner `unsafe` inside it, which
both silences `unused_unsafe` as a signal and defeats the point of
`deny(unsafe_op_in_unsafe_fn)`. The `unsafe` block now covers exactly the
`with_handle` call.

Proof: `cargo test -p libjpeg-turbo-rs-capi --test handle_borrow_scope`
(4 tests — the accessor's shape, the absence of `handle_as_mut` in all nine
modules, a `tj3Set`/`tj3Get` round trip, and NULL-handle sentinels), plus the
full capi suite at 54 blocks / 0 failures and both CI clippy legs clean.

**Status (2026-08-11): CLOSED — criteria 1, 2, 3, 5, 6, 7 delivered.**

* **Criterion 1 — done.** Every export whose parameter list contains a raw
  pointer is `pub unsafe extern "C" fn`. The inventory, with the two numbers
  kept apart because they answer different questions:

  | | Count |
  | --- | --- |
  | Pointer-taking exports, now `unsafe` — the **soundness surface** | **136** |
  | …of which this change converted (4 were already `unsafe`: `tj3Free`, `tj3Destroy`, `tjDestroy`, `jpeg_resync_to_restart`) | 132 |
  | Pointer-free exports, still safe — nothing to dereference | 21 |
  | Total exported symbols | 157 |

  The pointer-free 21 (`tj3Alloc`, the `*BufSize`/`*PlaneSize` family,
  `tjInit*`, `tj3Init*`, `jpeg_quality_scaling`, `jdiv_round_up`) stay safe by
  design, not by omission.

  **Symbol diff, as the criterion requires:** `nm -gU` on the release cdylib
  before and after is **byte-identical at 157 symbols**. This is now the second
  measurement refuting the "unsafe changes the ABI symbol" claim the crate-wide
  suppression rested on — the first covered only `tj3Free`/`tj3Destroy`, which
  is also why those two already counted as `unsafe` going in.

* **Criterion 2 — done.** `#![allow(clippy::not_unsafe_ptr_arg_deref)]` is
  deleted. No per-function suppression replaces it. The lint is now load-bearing:
  a new export that dereferences a raw pointer from a safe `fn` fails the build.

* **Criterion 3 — done.** Every converted export carries a `# Safety` section
  naming its own pointer parameters. The obligations common to all of them —
  validity, minimum size, alignment, ownership transfer, aliasing, threading —
  are stated once at the crate root under **Pointer contract**, which each
  section links. Stating them once is deliberate: 132 hand-written copies would
  drift, and `clippy::missing_safety_doc` enforces the per-function section
  regardless.

  One obligation was missing from the first draft and is worth naming, because
  it is the kind a shared contract loses: the **reusable output slot**. A
  non-null `*jpeg_buf` handed to `tj3Compress12`, `tj3Compress16`,
  `tj3CompressFromYUV8`, `tj3CompressFromYUVPlanes8` or `tj3Transform` is
  `free()`d by this library when the output does not fit, so it must come from
  `tj3Alloc`/`malloc`. A stack array or a `Vec`'s buffer satisfies every
  validity, size and alignment rule and is still UB to pass, because it is freed
  with the wrong allocator. Both the crate contract and those five functions now
  say so. Upstream TurboJPEG makes the same demand; we simply had not written it
  down.

* **Criterion 5 — done, but not where the first pass looked.** The criterion is
  about *slice construction*, and claiming it from the P4-139 work on
  `decompress.rs`/`precision.rs` was wrong: that grep covered the two files this
  item happens to name, not the crate.

  Review found the real instances — two rounds of them, and the first fix was
  not the whole story. `tj3CompressFromYUV8` and `tj3DecodeYUV8` sized their
  packed-YUV slice with `total.saturating_add(stride * ph)` — an *unchecked*
  multiply inside a saturating add — and fed the result straight to
  `slice::from_raw_parts`. A later round found `tj3SaveImage8` doing the same
  shape with `row_dense * h` (`imageio.rs`), where `c_int::MAX`-square RGBA
  exceeds `isize::MAX` on 64-bit and `w * bpp` wraps on 32-bit; it is now
  `checked_mul` under the same bound. Saturation is the worst available failure mode there:
  it yields `usize::MAX`, so an overflowing geometry produced a slice claiming
  the whole address space, not a short one. Both now go through
  `packed_yuv_len`, which is `checked_mul`/`checked_add` bounded by `isize::MAX`
  (`from_raw_parts`'s own precondition) and returns the documented `-1` with a
  `TJERR_FATAL` message.

  **Still open, and deliberately not claimed here:** `jpeglib.rs` retains
  memory-sizing `saturating_mul` at `:6697`, `:7382-7383`, `:7444`,
  `:9429/:9432` and `:10467`. Those size `Vec` allocations rather than raw
  slices, so they are outside criterion 5's wording but squarely inside
  [P4-139](#p4-139-memory-layout-arithmetic-is-decentralised-and-uses-saturatingunchecked-multiplication--partial-every-span-is-checked-and-the-rule-is-enforced-centralisation-and-scalingfactor-outstanding)
  criterion 3, which is where they are recorded.

* **Criterion 7 — done.** The crate root states the boundary plainly: invalid
  pointers from C are **not** defended against and cannot be — that is inherent
  to the C ABI, and upstream makes the same bargain. What is guaranteed, and
  what the fuzz and sanitizer gates test, is that a malformed JPEG cannot
  corrupt memory when the caller honours the pointer contract. Bad *data* is our
  problem; bad *pointers* are the caller's.

**Criterion 6 — the recorded decision: do not build a handle registry.**

The criterion asks for generation-tagged handles in a registry plus a busy flag,
and for the decision to be recorded either way. **Declined**, for three reasons:

1. **It defends against something criterion 7 says we do not defend against.**
   A stale or double-destroyed `tjhandle` is an invalid pointer from C. Adding a
   registry for that one case while every other pointer parameter remains the
   caller's responsibility buys inconsistent, partial safety, and invites the
   belief that the rest is checked too.
2. **It costs a process-global lock on every entry point.** A registry lookup
   per call serialises otherwise-independent instances — the exact property
   [P4-132](#p4-132-classic-c-abi-per-cinfo-state-is-thread-affine-p4-16-option-a--open)
   exists to *improve*. Upstream 3.2 moved the other way, removing TLS from its
   libjpeg API. (The criterion says "weigh against P4-131's threading work";
   P4-131 is native binary distribution — the threading item is P4-132.)
3. **Upstream does not do it,** and a drop-in replacement that rejects a handle
   C libjpeg-turbo would have accepted is a compatibility difference, not a
   safety improvement.

The *concurrency* half has merit and is not dropped: "a concurrent same-handle
call returns an error rather than aliasing `&mut`" is a real hazard, and it is
P4-132's subject, where the per-`cinfo` threading contract is being decided as a
whole. It is recorded there rather than solved twice.

**A note on the tripwire that fired.** The previous status predicted that when
criterion 1 landed, the NULL-handle call sites in `handle_borrow_scope.rs` would
stop compiling until wrapped. They did — along with 20 more sites across six
test files, and 40 internal delegation calls in `jpeglib.rs`/`legacy.rs`. Every
one of those was safe Rust calling a raw-pointer entry point with no `unsafe`
anywhere, which is precisely the condition this item was filed for. The
comments predicting the tripwire have been replaced with `SAFETY` notes stating
why each call is sound.

## P4-138. `BitWriter` Hand-Rolls Allocation Ownership and Can Double-Free on an Unwinding `reserve` — **CLOSED 2026-08-10**

**GitHub:** [#477](https://github.com/developer0hye/libjpeg-turbo-rs/issues/477) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** `BitWriter` (`src/encode/huffman_encode.rs`) manages its buffer
as a raw `*mut u8` + `pos` + `cap` triple with a manual `Drop`, a manual
`unsafe impl Send` (`:92`), and `Vec::from_raw_parts`/`mem::forget` round-trips
(`:100`, `:113`, `:130`, `:134`, `:318`).

`ensure_capacity` (`:124-137`) has a window where the allocation is owned twice:

```rust
unsafe {
    let mut v: Vec<u8> = Vec::from_raw_parts(self.buf, self.pos, self.cap);
    v.reserve(new_cap - self.pos);      // <-- may unwind
    self.buf = v.as_mut_ptr();
    self.cap = v.capacity();
    core::mem::forget(v);
}
```

If `reserve` unwinds — capacity overflow, or the allocation-error hook being
configured to unwind — the temporary `v` is dropped and frees the buffer, while
`self.buf`/`self.cap` still name it. `BitWriter::drop` (`:100`) then reconstructs
`Vec::from_raw_parts(self.buf, 0, self.cap)` over the freed block: **double
free**. Reported from static audit; not reproduced under fault injection, which
criterion 4 below exists to settle.

The size arithmetic in the same function is unchecked: `self.pos + additional`
and `self.cap * 2` can both overflow.

`begin_block`/`end_block` add a second hazard class — a pointer-cursor protocol
enforced only by prose (do not call other methods in between; do not exceed the
per-block reserve estimate; do not reuse a pointer across a reallocation).

**Acceptance criteria.**

1. `BitWriter` owns a plain `Vec<u8>`. No `from_raw_parts`, no `mem::forget`, no
   manual `Drop`, no manual `unsafe impl Send` (it becomes automatic).
2. Growth uses `try_reserve` with a typed error, or infallible `Vec` growth —
   never a raw-pointer round-trip.
3. If a raw cursor is retained for the block hot path, it lives in **one**
   private audited helper, and a guard type ensures that on unwind the length
   reflects only bytes actually written.
4. A fault-injection test that forces the growth path to fail/unwind and shows
   no double free, under both Miri and ASan. This is the criterion that
   confirms or refutes the double-free hypothesis; record the outcome either way.
5. `BitWriter` is `pub(crate)`, not public API.
6. An encode benchmark before/after in `experiments/encode.tsv` per the
   keep/discard protocol. The raw-pointer design presumably bought throughput;
   the replacement must show what it costs. If it costs materially, (3) is the
   fallback rather than reverting to raw ownership.

**Why P0.** A double free is memory corruption, and this one is reachable from
ordinary encoding if allocation ever fails or the size arithmetic overflows.

**Status (2026-08-10): partial — the hypothesis is confirmed and the defect is
fixed.** Audited against `main` @ 93b74c4. Criterion 4 gated the rest, so take
it first: the static finding was **real**, and `ensure_capacity` now closes the
ownership window before it opens.

```rust
let old_cap: usize = self.cap;
self.cap = 0;                    // `drop` is gated on `cap > 0`

unsafe {
    let mut v: Vec<u8> = Vec::from_raw_parts(self.buf, self.pos, old_cap);
    v.reserve(new_cap - self.pos);   // may unwind — `v` is now the sole owner
    self.buf = v.as_mut_ptr();
    self.cap = v.capacity();
    core::mem::forget(v);
}
```

Two `--lib` tests pin it: `growth_unwinding_leaves_exactly_one_owner` (renamed
from `reserve_unwinding_leaves_exactly_one_owner` when the ownership refactor
below removed the `reserve` window it was named for) and
`capacity_overflow_panics_before_the_ownership_window`. The first requests
`isize::MAX + 1` specifically so the *capacity check* raises a catchable panic
from inside the window — a merely huge request goes to the allocator, which
aborts rather than unwinds, and an abort never reaches `drop` at all. That
distinction is recorded in the test comment and is easy to get wrong on a
re-write.

Both sanitizer legs cover these, satisfying criterion 4's "under both Miri and
ASan": `cargo miri test --no-default-features --features std --lib -- --skip
simd::` (ci.yml) and the AddressSanitizer `--lib` job (sanitizers.yml).

The separately-noted unchecked arithmetic is also fixed:
`pos.checked_add(additional)` and `cap.checked_mul(2)`.

| # | Criterion | State |
| --- | --- | --- |
| 1 | Owns a plain `Vec<u8>` | **Done** 2026-08-10 |
| 2 | `try_reserve`, no raw round-trip | **Done** 2026-08-10 — infallible `Vec::reserve`, which the criterion permits |
| 3 | Raw cursor in one audited helper with an unwind guard | **Moot** — with a `Vec` there is no unwind-ownership hazard left to guard |
| 4 | Fault-injection under Miri + ASan | **Done** |
| 5 | `pub(crate)` | **Done** |
| 6 | Encode benchmark | **Done** 2026-08-10 — neutral |

**Status (2026-08-10): CLOSED — criteria 1, 2 and 6 delivered.**

`BitWriter` now holds a `Vec<u8>`. The manual `Drop`, the manual
`unsafe impl Send` and both `Vec::from_raw_parts`/`mem::forget` round-trips are
gone, so the double-free surface is **removed rather than merely closed**: the
`cap = 0` idiom that guarded the window no longer has a window to guard. `Send`
is automatic. `BitWriter::new`'s `saturating_mul(2)` became `checked_mul` —
saturating turns an oversized request into `usize::MAX`, the worst value to hand
an allocator (P4-139's rule, applied here because the expression sizes memory).

The hot path is untouched: it still writes into the `Vec`'s spare capacity
through a raw pointer and tracks `pos` itself, so no `Vec` bookkeeping happens
per flush. What that costs is an invariant — `buf.len()` is a *synchronisation
point*, not the write cursor, and must be raised to `pos` before anything that
can reallocate.

**That invariant is where the interesting failure lives, and the first test
written for it was worthless.** `Vec::reserve(additional)` is defined relative
to `len`, so growing with a stale `len` of 0 asks for `additional` bytes total
instead of `pos + additional` — an under-allocation the following writes run
past. A "did the bytes survive?" test does **not** catch it: the underlying
`realloc` preserves the whole block whatever `len` says, and `Vec`'s amortised
doubling usually covers the shortfall. Removing the `set_len` line left that
test green.

`growth_reserves_room_for_everything_already_written` asserts the postcondition
directly — capacity ≥ `pos + additional`, with numbers chosen to defeat
amortised doubling — and fails with `capacity 10000 is short of pos 1000 +
10000` when the sync is removed. A test-only `capacity()` accessor exists
because the postcondition is otherwise unobservable from outside; the only other
way to check it is to write past the end, which is the bug rather than a test
for it.

**Criterion 6 — neutral, and a methodology note worth more than the number.**
1080p 420/422/444 on aarch64-darwin: `Vec` 5.4248 / 6.7569 / 10.176 ms against
`main`'s raw pointer at 5.4416 / 6.7829 / 10.219 ms — 0.3% favourable, i.e.
noise. The criterion's premise ("the raw-pointer design presumably bought
throughput") does not hold, because the raw pointer was only ever about
*ownership*; the hot path already bypassed `Vec` and still does.

The first measurement pass read +5.8% and was **contaminated** by a background
build. A/B/A/B on the 420 case settled it: 5.4226 / 5.4438 / 5.4247 / 5.4417. A
single A/B pair would have recorded a regression that does not exist and sent
this item to the criterion-3 fallback design for no reason.

## P4-139. Memory-Layout Arithmetic Is Decentralised and Uses Saturating/Unchecked Multiplication — **PARTIAL: every span is checked and the rule is enforced; centralisation and `ScalingFactor` outstanding**

**GitHub:** [#478](https://github.com/developer0hye/libjpeg-turbo-rs/issues/478) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** Width × height × bytes-per-pixel, stride × height, padded plane
sizes, and crop offsets are recomputed independently along many paths, with
inconsistent overflow behaviour. Each of P4-136, P4-137 and P4-138 contains an
instance; this entry is the common cause.

**Confirmed instances.**

- **Saturating spans.** `crates/libjpeg-turbo-rs-capi/src/decompress.rs:148`
  (`effective_pitch.saturating_mul(h)`) and `precision.rs:103,274,351,492`.
  Saturation turns overflow into `usize::MAX` — precisely the wrong value to
  then bound a raw slice. It must be a typed error.
- **Unchecked products.** `src/api/progressive_output.rs:257`
  (`ci.comp_w * ci.blocks_y * block_size`) wraps in release. **Fixed by P4-136**
  (closed 2026-08-10) — every span in that file is now `checked_span` +
  `try_filled_vec`. Listed here because it is the shape, not because it is open.
- **Inherited from P4-137 (recorded 2026-08-11):** memory-sizing
  `saturating_mul` in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` at `:6697`
  (`old_bufsize * 2`), `:7382-7383` (`width * input_components`, then
  `* height`), `:7444`, `:9429`/`:9432` (`row_samples * height`, sizing a
  `vec![0u16; …]`) and `:10467` (`num_blocks * 64`). These size `Vec`
  allocations rather than raw slices, so P4-137 criterion 5 did not cover them —
  but they are exactly this item's criterion 3. The *slice*-sizing instances in
  `yuv.rs` were live UB (`saturating_add` reaching `usize::MAX` and then being
  handed to `from_raw_parts`) and were fixed under P4-137; these remaining ones
  produce an over-large allocation request rather than an over-large span, so
  they abort or error rather than corrupt.
- **Inherited from P4-136:** the three `set_len` sites in
  `src/encode/pipeline_impl/progressive_entropy.rs` (`:65`, `:96`, `:145`). The
  `set_len` calls themselves are *correct* — they publish a span a raw cursor
  already wrote, with `written` from `offset_from`. What belongs here is the
  reserve arithmetic feeding them: `total_blocks * 4 + restart_overhead + 64`,
  unchecked, with the in-loop guards comparing against the requested `reserve`
  rather than actual capacity. The margins hold for legitimate geometry (`>= 68`
  covers the 64-byte per-block worst case, `>= 84` the 80-byte restart case) and
  no reachable path was found — it needs `total_blocks > usize::MAX / 4` to
  wrap — so this is hardening, not a live defect.
- **`ScalingFactor` is a public panic/overflow surface**
  (`src/common/types.rs:335-365`): `num` and `denom` are `pub`, and `new()`
  accepts `denom == 0`, so `block_size()` and `scale_dim()` defend with
  `assert!` — a panic on a public API, not a `Result`. Both then multiply
  unchecked (`self.num * 8`, `input_dim * self.num as usize`). Because the
  fields are public, a validating constructor alone cannot fix it: struct-literal
  construction bypasses any check.

**Acceptance criteria.**

1. One `ImageLayout`-style abstraction owns every span computation: `width`,
   `height`, `bytes_per_pixel`, optional `stride` (rejecting `stride < row_bytes`),
   producing `total_bytes` via checked arithmetic with an `isize::MAX` bound and a
   typed error.
2. Every output allocation and raw-slice construction goes through it: baseline
   decode, progressive decode, scaling, crop, 12/16-bit, encode input validation,
   TJ3, classic `jpeg_*`, YUV planes, transform output.
3. **No `saturating_*` or `wrapping_*` in any expression that sizes or bounds a
   memory region.** Enforce with a lint, a CI grep, or a review checklist item —
   and say which.
4. `ScalingFactor` fields become private with a validated constructor returning
   `Result`, or the type becomes an enum over the 16 supported factors. Public
   methods stop panicking on public input. **Note this is a breaking API change**
   — sequence it with the next major, and record the migration.
5. Property tests over adversarial geometry (huge dimensions, `stride` just under
   `row_bytes`, dimensions whose product overflows on 32-bit) on both 64- and
   32-bit targets.

**Status (2026-08-11): partial — criterion 3 done, and with it every
*confirmed instance*. What remains is the refactor, not a live defect.**

**Status (2026-08-14, chunk 1): still partial — criterion 1 done, criterion 2
half done, criterion 5's property-test half done.** `ImageLayout` exists
(`src/common/layout.rs`) and the C-ABI crate adopted it; the root crate's
decode/encode sites are chunk 2. Criterion 4 (`ScalingFactor`) is untouched and
still waits on 0.9.0. Details in **What remains** below.

**Status (2026-08-14, chunk 2): criterion 2 now covers the root crate's
encode input, crop, YUV, plane-size and coefficient families, and adoption
found five live defects.** Pinned by `tests/layout_adoption.rs` (22 tests), the
root-crate counterpart to chunk 1's `capi_layout_adoption.rs`. Each was
verified red first — the observation is recorded in the test's doc comment,
because "returns an error" is the same assertion whether the old behaviour was
a panic, a wrap, or an unrelated error raised by accident. All `file:line`
coordinates in the middle column are **pre-fix**; the lines have since moved.

| Defect | Observed before the fix | Now |
| --- | --- | --- |
| `Encoder::encode` reads rows before any path's size check (`bottom_up`, `fancy_downsampling`, `grayscale_from_color`) | slice-range panic at `encoder.rs:471` / `:789`, index-out-of-bounds at `:577` | `BufferTooSmall`, from a `checked_input_layout` inside each of the three steps |
| `StreamingDecoder::crop_scanline` computes `image_width - aligned_x` and `xoffset + width` unchecked | "attempt to subtract with overflow" / "attempt to add with overflow" at `streaming.rs:58` / `:57` | `Unsupported`, with the end still clamped to the image for in-range windows |
| `ScanlineDecoder`'s crop guard is `x + width > img.width` on an unchecked sum | "attempt to add with overflow" at `scanline.rs:163`; in release the sum wraps *below* the width and the guard passes | `Unsupported`, from `checked_add` |
| `compress_raw` / `compress_raw_12` size the plane check with an unchecked product and cap nothing | "attempt to multiply with overflow" at `raw.rs:70` / `raw_data_12.rs:485`; in release `(usize::MAX / 4 + 1) x 4` wraps to 0 and an **empty** plane passes | `LimitExceeded`; a representable short plane still reports `BufferTooSmall` |
| `api/yuv.rs`'s `validate_pixel_buffer` — the entry gate for the pixels-in direction, `encode_yuv_planes` and `encode_yuv` — sizes `width * height * bpp` unchecked | "attempt to multiply with overflow" at `yuv.rs:56`; in release `(usize::MAX / 4 + 1) x 4` at `Rgb` wraps to 0 and an **empty** buffer passes, after which the row loops index it | `LimitExceeded`; the two packed-plane totals are summed through `checked_sum_of_planes` for the same reason |

Two are worth naming. The third and fifth are the same mistake: **a bounds
check written on a wrapping product is not a weaker check, it is an *inverted*
one** — the larger the input, the smaller the value it compares, so the check
admits exactly the geometry it exists to reject.

And the first taught the sharper lesson, in review rather than in the original
fix. The obvious shape — one validation gate at the top of `Encoder::encode` —
*regressed* four error-precedence cases, because a new first gate does not only
add an error, it takes precedence over every error that used to come first. A
caller passing `lossless_predictor(9)` with a short buffer was told
"BufferTooSmall" about a buffer the lossless encoder had not reached, where it
used to be told the predictor was out of range. The check therefore lives in
the three helpers that read rows (`flip_rows`, `apply_triangle_prefilter`, the
grayscale extraction), each calling `checked_input_layout` first, so it cannot
fire unless the buffer is about to be indexed.
`an_option_error_still_precedes_the_buffer_check` pins every error that must
still come first: the lossless predictor at both a short buffer and a zero
dimension, the smoothing/progressive combination, and the two dimension errors
as controls.

The surviving claim is narrower than "precedence is preserved", and worth
stating exactly: an encode that rearranges nothing never reaches the check, so
nothing moves. On the three paths that *do* rearrange, a short buffer now
reports `BufferTooSmall` where some other error might have come first — but
there it previously reported nothing at all, because the step panicked.

Families converted in chunk 2, all behaviour-neutral except where noted above:

* **Encode input (10 sites + the 12-bit twin).** The `width * height * bpp`
  products in `encode/pipeline_impl/{baseline,optimized,progressive,arithmetic,custom_sampling,lossless}.rs`
  (nine sites) and `api/encoder.rs`'s `checked_input_layout` (the tenth) now go
  through `ImageLayout::packed`, and the two caller-supplied plane loops
  (`raw.rs`, `api/raw_data_12.rs`) through `checked_span`. On 64-bit the nine
  pipeline sites are unreachable behind the 65535 dimension cap and this is
  hardening; on 32-bit `65535 x 65535 x bpp` does not fit `usize` and the wrap
  was live. One consequence worth stating, since the 32-bit legs run the lib
  tests: on ILP32 those sites now answer `LimitExceeded` where they answered
  `BufferTooSmall`, because `65535^2` exceeds `isize::MAX` before any buffer is
  compared.
* **`common/bufsize.rs`, the unchecked twins of the C-ABI crate's checked
  forms.** `pad` became `checked_pad`, and `jpeg_buf_size`, `yuv_plane_size`,
  `yuv_buf_size`, `yuv_plane_width`/`_height`, `calc_jpeg_dimensions` and
  `calc_output_dimensions` now report **0** for unrepresentable geometry
  rather than a wrapped product.
  Saturating was not an option (criterion 3 bars it, and `tests/sizing_arithmetic_gate.rs`
  enforces that), and these signatures are infallible `usize` — so the refusal
  is 0, which is what the C twins already answer (`tj3JPEGBufSize` returns 0
  and records "Image is too large"; `tj3YUVPlaneWidth` returns 0 for an
  argument it rejects). The rule is stated once in the module header rather
  than per function. Three sites needed more than a mechanical swap:
  `yuv_buf_size` must distinguish a plane that *refused* (reporting 0) from a
  genuinely empty one, since summing the refusal would under-report the total;
  `jpeg_buf_size` puts its `+ 2048` header allowance inside the `isize::MAX`
  ceiling rather than outside it; and the chroma arms of
  `yuv_plane_width`/`_height` dropped an algebraic no-op, `padded * 8 /
  (factor * 8)`, whose only effect was that the `* 8` could wrap — at 4:2:0 a
  width of `2^62 + 2` reported a chroma width of **1** instead of `2^60 + 1`,
  under-sizing by nine orders of magnitude. Review caught that one; the
  original conversion had left it and the new module header would have been
  false the day it was written.
* **Decode coefficient and plane allocations.** `decode/pipeline_impl/progressive.rs`
  (coefficient buffers, AC-max buffers, component planes),
  `decode/pipeline_impl/arithmetic.rs` (both the sequential and the progressive
  allocations) and `api/coefficient.rs`'s whole-image coefficient array route
  their counts through `checked_span` and their allocations through
  `try_filled_vec`, so a hostile SOF gets `LimitExceeded`/`AllocationFailed`
  rather than an abort.
* **The `checked_plane_size` twin is gone.** `api/progressive_output.rs`'s
  private copy was contract-identical to `checked_span`; it is deleted and its
  twelve call sites repointed, which is what chunk 1's rustdoc said had not
  happened yet. That rustdoc now says what is true.

* **The two YUV entry gates.** `api/yuv.rs`'s `validate_pixel_buffer` routes
  through `ImageLayout::packed`, and the packed-plane totals in `decode_yuv`
  and `compress_from_yuv` through a new `checked_sum_of_planes` — three plane
  sizes each bounded by `isize::MAX` can still exceed it together, and both
  totals bound a caller-supplied buffer.
* **`crop_scanline` now measures in output space.** Its bounds and its iMCU
  alignment used `header.width` and a hard-coded block size 8, but `set_crop`'s
  coordinates are post-scale (`output.rs`'s `scaled_imcu_w`). At 1/1 the two
  agree, which is why nothing noticed; under an upscaled decode the old code
  aligned to the wrong grid and compared against the wrong edge, and the new
  refusal would have rejected valid offsets. `Decoder` gained `output_width()`
  (the twin of the existing `output_height()`) and `output_block_size()`.

**Not in chunk 2, and why.** `decode/pipeline_impl/output.rs` and `color.rs`
hold 17 more allocations sized by an unchecked geometry product (14 and 3;
the other 16 allocation sites in those files are row buffers, fixed
capacities, or sized by a length rather than by a geometry product). They are
the same shape, but they are the *scaling and transform
output* families the final chunk owns, and converting them touches the
merged-upsample and RGB565 kernels that are under active optimisation. The
same goes for `api/yuv.rs`'s *internal* plane allocations (`y_w * y_h` and
friends): the entry gates are checked now, so a wrapping geometry is refused
before it reaches them, but the allocations themselves are still written out.
Also left, from the final drift audit: `checked_staging_span`'s `isize::MAX`
arm is exercised by no leg at any pointer width — the width test returns at
the row check first and the companion geometry is 64x64; documented in
`capi_span_overflow_guards.rs` and here so the gap is tracked, not hidden.

Also left: `ScanlineEncoder::new`'s `vec![0u8; width * height * bpp]`, which is
an infallible constructor and so needs an API decision (panic, or a fallible
`try_new`) rather than a mechanical conversion — file it with the final chunk.

* **Criterion 3 — done, and it is the load-bearing one.** The rule is enforced
  by `tests/sizing_arithmetic_gate.rs` against
  `docs/sizing_arithmetic_inventory.tsv`, which classifies all 86 remaining
  `saturating_*` occurrences (73 distinct lines) in library sources. A new one fails the build until
  a human classifies it; a stale row fails until it is removed. **Naming the
  mechanism was part of the criterion, so: it is a test, not a grep or a review
  checklist** — a test runs on every CI leg and cannot be skipped by a reviewer
  in a hurry.

  Scope covers `saturating_mul`/`add`/`sub` — 86 occurrences across 73 lines.
  **`wrapping_*` is deliberately excluded, which narrows the criterion's literal
  wording** ("no `saturating_*` or `wrapping_*`"), so it is recorded here rather
  than left implicit. It is the IDCT's C-parity idiom at 244 sites where
  wrapping is the *specified* behaviour being matched, and those kernels are
  under active optimisation; an inventory regenerated on every IDCT edit would
  be deleted within a month, and a gate nobody maintains is worse than a
  narrower one that holds.

  The cost of that narrowing, assessed rather than assumed: the excluded set
  includes index arithmetic such as `src/common/exif.rs:62-75`, where
  `data[offset.wrapping_add(1)]` would wrap instead of panicking. It is **not
  reachable** — `data[offset]` is evaluated first and its bounds check panics,
  and a valid `offset` cannot be within 1 of `usize::MAX` since it is below
  `data.len()`. Latent style, not a defect; if `wrapping_*` is ever gated, that
  is the first site to fix.

  `saturating_sub` *is* covered, and adding it found two unclassified sites in
  `sampling.rs`. It can only floor at 0, never over-size a span, so every
  instance classifies as `clamped-difference` — but the gate seeing them is the
  point.

  A second test asserts the gate actually scans hundreds of files across both
  crates, because a path bug would otherwise make it pass forever.

* **Every confirmed instance is fixed.** The `saturating_mul` spans in
  `decompress.rs`/`precision.rs` went earlier; `progressive_output.rs` went with
  P4-136; the packed-YUV and `tj3SaveImage8` slice spans went with P4-137. This
  change takes the last five, in `jpeglib.rs`: the destination-buffer doubling,
  the 8-bit and 12/16-bit staging buffers, the per-row copy length in
  `jpeg_write_scanlines`, and `jcopy_block_row`'s `copy_nonoverlapping` length.
  The last two are the ones worth naming — both were feeding a *memcpy length*,
  where saturating to `usize::MAX` copies the address space rather than
  reporting the geometry.

  The error each raises follows upstream rather than collapsing to one code,
  because upstream distinguishes three different failures:

  | Failure | Code | Upstream |
  | --- | --- | --- |
  | `image_width * input_components` not representable as `JDIMENSION` | `JERR_WIDTH_OVERFLOW` (parameterless) | `jcmaster.c:190-194` |
  | An individual axis over `JPEG_MAX_DIMENSION` | `JERR_IMAGE_TOO_BIG` (`%u`) | `jcmaster.c:186-189` |
  | A staging span that cannot be allocated | `JERR_OUT_OF_MEMORY` case 8 (`%d`) | `jmemmgr.c:364-380` |

  Two further details the first attempt got wrong. The width test is
  representability as `JDIMENSION`, **not** as `usize`: on a 64-bit host a 4 GiB
  row fits `usize` and would have sailed past a `checked_mul` while the C
  library errors. And the staging span needs an `isize::MAX` bound on top of
  `checked_mul`, because a 32-bit product can fit `usize`, exceed `isize::MAX`,
  and make `Vec::resize` panic *after* the cinfo has already changed state —
  where `unwind_guard!` swallows it.

  Two paths deliberately do not raise: the destination-buffer doubling records
  `JERR_OUT_OF_MEMORY` through its existing `pending_error` channel, and
  `jcopy_block_row` returns silently because it has no error manager to raise
  through — matching its existing contract for a null or empty request.

  The three `progressive_entropy.rs` `reserve` expressions inherited from P4-136
  are unchanged and remain listed above; they are unchecked but not saturating,
  and no reachable path was found.

**What remains.**

* **Criteria 1–2 — the `ImageLayout` abstraction and its adoption. Chunks 1
  and 2 landed; the transform-output, scaling and classic `jpeg_*` staging
  families remain.** Criterion 1 is done: `src/common/layout.rs` owns
  `width`/`height`/`bytes_per_pixel`/optional `stride`, rejects
  `stride < row_bytes`, and produces `total_bytes` through checked arithmetic
  bounded by `isize::MAX` with a typed error, plus a `checked_span` free
  function for spans that are not 2-D. Criterion 2 is partial: the C-ABI crate
  adopted it in chunk 1 (`compress.rs`, `precision.rs`, `imageio.rs`,
  `yuv.rs` — TJ3 compress, 12/16-bit, YUV plane sizing, the image-file entry
  points), and the root crate in chunk 2 (encode input across all six modes,
  both raw-plane loops, `common/bufsize.rs`, the crop paths, the progressive
  and arithmetic decode allocations, `api/coefficient.rs`, and the deletion of
  `progressive_output`'s `checked_plane_size` twin). Still written out per
  site: the baseline decode output and upsample buffers
  (`decode/pipeline_impl/output.rs`, `color.rs`), scaling, transform output and
  the classic `jpeg_*` staging buffers — the final chunk. Chunk 2 was *not*
  purely a consistency refactor: it surfaced five live defects, tabulated in
  the chunk-2 status above.

  Adoption was not behaviour-neutral, and the three caller-visible changes are
  pinned by `crates/libjpeg-turbo-rs-capi/tests/capi_layout_adoption.rs`:
  `tj3SaveImage8` now refuses a negative `pitch` instead of reading it as
  "dense" (`turbojpeg-mp.c:511-513`), `tj3LoadImage8` requires `align` to be a
  positive power of two instead of clamping with `align.max(1)`
  (`turbojpeg-mp.c:317-321`), and `tj3Compress12`/`tj3Compress16` bound their
  source span in *bytes*, putting the ×2 element size inside the checked chain
  where `from_raw_parts`' precondition needs it.
* **Criterion 4 — `ScalingFactor`. Decision recorded: do it, in 0.9.0, as
  private fields plus `try_new`.** Not the 16-variant enum: the type is
  constructed from caller-supplied `num`/`denom` at the C ABI boundary
  (`tj3SetScalingFactor`), so an enum would need a fallible lookup anyway and
  would lose the ability to represent what upstream accepts. The migration is
  `num`/`denom` → `num()`/`denom()` accessors plus `try_new(num, denom) ->
  Result<Self>`, touching ~30 read sites (mostly `tests/cross_product_*`) and 2
  struct-literal sites. **It is deferred, not dismissed** — it is the one
  criterion here that cannot be done without a breaking change, and this crate
  is 0.8.0, so it belongs in a version bump rather than smuggled into a fix.

  Worth recording so the next session does not re-derive it: the *overflow* half
  of that criterion is not reachable. `scale_dim`'s `input_dim * self.num`
  cannot overflow — `input_dim` is bounded by a JPEG's 65535 dimension limit and
  `num` by 16 — so what is left is the `assert!` on `denom == 0`, a panic on
  public input. That is an API-quality defect, not a memory-safety one.
* **Criterion 5 — a 32-bit C-ABI leg. Done 2026-08-14; kept here because the
  rest of this list is not.** The compile blocker went first: chunk 1
  gated the encode ABI-offset assertion block on `target_pointer_width = "64"`
  (it was the one ungated LP64 block left, and it failed the *build* on ILP32
  rather than flagging a real mismatch), so `cargo check -p
  libjpeg-turbo-rs-capi --tests --target armv7-unknown-linux-gnueabihf` now
  succeeds, warning-free (the 64-bit-only tests live in one
  `cfg(target_pointer_width = "64")` module rather than carrying three separate
  gates, so nothing is left unused on ILP32). **The CI leg landed 2026-08-14:**
  `armv7.yml`'s job gained a second qemu-arm step building and running
  `-p libjpeg-turbo-rs-capi --lib --test capi_layout_adoption --test
  capi_span_overflow_guards` under the job's existing `-C overflow-checks=on`,
  which turns a 32-bit wrap into a loud failure. Suites are selected by name
  because a blanket capi test build would drag in C-compiling harnesses;
  neither suite named here links a C oracle at all.

  What the leg adds, stated exactly, because the suite names invite a stronger
  reading than is true. It executes one guard arm that no 64-bit leg reaches:
  `capi_span_overflow_guards`' 65500 x 65573 is 4,295,031,500, past `u32::MAX`,
  so on ILP32 the refusal comes from `checked_samples_per_row`'s `usize`
  overflow rather than from the `JDIMENSION` bound that fires on a 64-bit host.
  Everything else it adds is the crate's ordinary span arithmetic run at half
  pointer width. It does **not** execute the TJ3 `isize::MAX` arm:
  `capi_layout_adoption`'s `source_span_bounds` module is
  `cfg(target_pointer_width = "64")`, so 3 of that suite's 10 tests compile out
  here — by design, per the correction below. The classic guard's own
  `isize::MAX` arm (`checked_staging_span`) is still unexercised on either
  width; the two tests in that suite return at the row-width check or use a
  64x64 geometry.

  **Correction (chunk 1):** this entry previously said the `usize`-overflow and
  `isize::MAX` arms of *the* span guards were 32-bit-only. That is true only of
  the classic-ABI guards, whose `image_width` is a `u32`. The TJ3 source-span
  guards take caller-supplied `c_int` width and height, so their `isize::MAX`
  arm is 64-bit-reachable. `capi_layout_adoption.rs`'s `source_span_bounds`
  module now exercises it for all three: `tj3Compress12`/`tj3Compress16`
  (`c_int::MAX` x 800_000_000 at `TJPF_RGB`, 1.03e19 bytes) and `tj3Compress8`
  (`c_int::MAX` square at `TJPF_RGBA`, 18446744056529682436 bytes) — each
  inside `usize` and past `isize::MAX`. `tj3Compress8`'s guard predates this
  work (P4-137 wrote it); what it lacked was any test, because the file that
  looked like its home said the arm was unreachable.

* **Criterion 5 (original) — property tests over adversarial geometry. Done for
  `ImageLayout`.** `common::layout::tests::adversarial_geometry_matches_the_u128_model`
  sweeps a hand-picked edge matrix plus 4096 deterministic Mulberry32 cases
  against a `u128` reference model that cannot overflow, asserting both the
  accepted totals and the refusals. It is a `--lib` test, so it runs on the
  32-bit legs (`armv7.yml` under qemu, `wasm32-wasip1`) as well as the 64-bit
  ones, where the interesting boundary moves to ~2 GiB. Hand-rolled rather than
  `proptest`, for reproducibility. What it does *not* cover is the call sites:
  the per-entry-point guards still rely on targeted regressions
  (`yuv_packed_length_overflow.rs`, `norealloc_buffer_capacity.rs`,
  `capi_layout_adoption.rs`, the P4-136 pointer-width pair).

  The sweep earned its keep during chunk 1's review: extending it to validate
  `(height - 1) * stride` *before* the zero-area early-out caught a real hole
  in `ImageLayout` itself. A layout like `strided(0, 1000, 3, usize::MAX / 2)`
  was accepted — `row_bytes` is 0, so the under-stride test is vacuous and the
  total is honestly 0 — while `row_offset(999)` wrapped, and `compress.rs`
  turns that offset into a `ptr::add`. Construction now proves the head
  unconditionally.

* **Chunk 1 gap: no sanitizer leg runs `capi_layout_adoption.rs`.**
  `tj3_decompress12_writes_only_the_pitched_extent` hands the library a
  destination sized at exactly upstream's minimum, so the pre-chunk-1
  `pitch * height` span built a `from_raw_parts_mut` past the caller's `Vec`.
  Its assertions (rows match a dense decode, inter-row padding untouched) pass
  either way; only Miri or ASAN sees the out-of-bounds slice.
  `sanitizers.yml` excludes integration tests and the capi Miri step names
  `capi_create_abi_guards` alone. Adding a leg is not a one-liner — the test
  performs a real encode and decode, so Miri stops at the first SIMD intrinsic
  the dispatcher picks (measured locally: `llvm.aarch64.neon.ushl.v8i16`),
  which is why the library's own Miri run passes `--skip simd::`. A
  scalar-only capi build under Miri belongs to
  [P4-141](#p4-141-soundness-verification-program-mirisanitizerfuzz-coverage-gaps-and-an-unsafe-inventory-gate--open),
  not here.

**Also recorded: resource-limit defaults.** `DecodeLimits` currently defaults to
roughly 2.1 billion pixels with `max_memory = None`. That is a compatibility
default, not a safe-for-untrusted-input default. Consider splitting
`DecodeLimits::untrusted()` (bounded pixels/memory/scans/metadata) from
`DecodeLimits::compatibility()` (current behaviour, for libjpeg parity), with the
one-shot convenience APIs defaulting to `untrusted`. This is a robustness/DoS
posture decision, distinct from the soundness items — decide it here, or split it
out, but do not leave it unstated.

**Decision (2026-08-11): keep one permissive default; do not add an
`untrusted()`/`compatibility()` split.** Reasons, in order of weight:

1. The stated goal of this project is to *replace* C libjpeg-turbo. A default
   that rejects images `djpeg` accepts is a drop-in failure, and it would fail
   in the worst way — on the user's real input, after they had already switched.
2. The split invites the wrong mental model. `untrusted()` reads as "safe
   against hostile input", which memory safety must be **unconditionally**, not
   as an opt-in profile. Naming a profile that way implies the other one is
   unsafe, which would be a bug rather than a setting.
3. `DecodeLimits` is already public and caller-settable, so anyone with a DoS
   posture to enforce can express it today, in the units their deployment cares
   about, rather than accepting a ceiling we guessed.

What is worth doing instead, and is *not* claimed as done here: document
recommended bounds for untrusted input in the crate docs next to
`DecodeLimits`, as guidance rather than a second default. Carried with the
final chunk's remaining work above — it was parked against criterion 5, which
closed 2026-08-14.

## P4-140. Public Documentation Claims Safety and Drop-In Status the Code Does Not Support — **CLOSED 2026-08-09**

**GitHub:** [#479](https://github.com/developer0hye/libjpeg-turbo-rs/issues/479) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** Two documentation claims are currently unsupportable, and one of
them can cause memory corruption if a reader acts on it.

1. **`crates/libjpeg-turbo-rs-capi/src/lib.rs:5`** tells consumers they can
   "link against this crate in place of the stock `libjpeg.so.62`". `LAST_MILE.md`
   classifies v6b/v7 as an **explicit non-goal** (T4), and the build and install
   paths reject those identities. Acting on the crate doc means loading a v8
   struct layout into a consumer compiled against v6b — different
   `jpeg_decompress_struct` offsets, i.e. reads and writes at wrong offsets. This
   is a memory-safety claim, not a marketing one.
2. **`README.md:9`** — "Pure-Rust ... No C dependencies, no unsafe FFI". Each
   clause is literally true, but together, and next to "Pure-Rust", they read as
   "there is no unsafe code / memory safety is established". With P4-135 open,
   that reading is false.

**Acceptance criteria.**

1. The capi crate doc names the actual tier: v8 (`libjpeg.so.8`) is the
   experimental/partial target; v6b/v7 are non-goals. It must not describe
   `.so.62` substitution as supported.
2. Tier language is consistent across `README.md`, the capi crate docs,
   `ABI_COMPATIBILITY.md`, `install_capi.sh`, and the crates.io description — one
   wording, one source of truth.
3. Until P4-135..P4-139 close, no "memory-safe replacement" or unqualified
   "drop-in replacement" claim appears in published material. Interim wording
   should say the safe-API/unsafe-SIMD boundary is under audit, and name TJ3 as
   the primary C compatibility target.
4. Once those close **and** an external audit (P4-141) reports, a scoped
   guarantee may be published — separating (a) the safe Rust API being sound,
   (b) unsafe being confined to validated private SIMD kernels, and (c) the C ABI
   preserving caller pointer obligations, so that malformed *JPEG input* cannot
   corrupt memory when those obligations are met. It must not claim to defend
   against arbitrary invalid pointers from C.

**Why this is filed as a defect.** (1) is a doc bug that produces memory
corruption if believed. It is also the cheapest item in this batch — fix it
immediately, ahead of the code work.

**Status (2026-08-09): closed.** Landed in #485; `crates/libjpeg-turbo-rs-capi/src/lib.rs` no longer offers the crate as a `libjpeg.so.62` replacement, and `README.md` states the safety scope rather than a guarantee.

## P4-141. Soundness Verification Program: Miri/Sanitizer/Fuzz Coverage Gaps and an `unsafe` Inventory Gate — **OPEN**

**GitHub:** [#480](https://github.com/developer0hye/libjpeg-turbo-rs/issues/480) — under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** The existing CI is genuinely strong — Miri on non-SIMD unit
tests, ASan and UBSan, a C-boundary sanitizer harness, 12 fuzz targets on a
6-hourly schedule, `cargo-deny`, and a `no_std` matrix. But P4-135 was found by
*reading code*, not by any of it, and that is the diagnostic finding: the
tooling's coverage does not intersect the crate's highest-risk surface.

**Current gaps.** The Miri job excludes SIMD, integration tests, most of the C
ABI, and every architecture-specific direct call. No job constructs a safe-Rust
misuse of a public SIMD entry point; none injects allocation failure; none runs
32-bit.

**Acceptance criteria.**

1. **Miri** additionally covers non-SIMD integration tests, doctests, progressive
   output (P4-136), `BitWriter` (P4-138), post-allocation-failure state, and
   concurrent one-time initialisation.
2. **Sanitizers** run with SIMD on *and* off; on an AVX2 machine and a
   non-AVX2 one; on 32-bit `i686`; and on AArch64 NEON. Add guard pages either
   side of C destination buffers, short-stride and canary buffers, and repeated
   init/destroy sequences. **Partially delivered 2026-08-13 by the P4-135
   closure:** `cross-arch.yml`'s `test-linux-x86_64-no-avx2-emulated` built
   four *integration* binaries and no `--lib`, so nothing ran the lib suite
   under a masked CPUID; it now carries `--lib`, and the scalar-fallback arms
   P4-135 added to `avx2_idct_islow` / `avx2_fdct_quantize` execute there.
   That is the x86 non-AVX2 half for *tests*; the sanitizer legs, `i686` and
   AArch64 NEON remain. Also outstanding from the same closure: the wasm
   wrappers' `fits == false` fallback arms have no executing coverage
   anywhere (wasip1 parity uses exact-fit slices, and the panic arm cannot
   be asserted under `panic = "abort"`) — the "pinned by the checks' code,
   not by a gate" shape this item exists to retire.
3. **An API-sequence fuzzer** exists alongside the byte fuzzers — driving
   `new → configure → probe → decode → reset → decode → transform → destroy`
   orderings — plus a process-isolated C-ABI harness covering
   `init→destroy→destroy`, undersized output buffers, pitch boundaries, maximum
   dimensions, same-handle concurrent calls, and callback reentry.
4. **An `unsafe` inventory is committed and gated.** Per site: location,
   why safe Rust cannot express it, the invariant (bounds/lifetime/aliasing/CPU
   feature), whether a safe caller can reach it, the regression test that would
   catch a broken invariant, tool coverage, and last reviewer. CI diffs the
   inventory and requires review for additions. **A raw count is not the
   deliverable** — "780 unsafe operations" says nothing about risk; one
   precondition-free safe wrapper (P4-135) outweighs hundreds of intrinsic calls.
5. **Parser and control-plane `unsafe` goes to zero.** Malformed-input handling —
   progressive scan state, restart markers, EOB runs, spectral ranges,
   coefficient indexing, marker length parsing, custom scan scripts — is the
   largest attack surface and should be entirely safe Rust. `decode/huffman.rs`'s
   `get_unchecked`/`get_unchecked_mut` on `ZIGZAG_ORDER` and coefficients is the
   known instance: replace with safe indexing and show the generated code is
   unchanged, or justify with a benchmark.
6. **`#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]` compiles.** This
   is P4-69's goal; it becomes reachable once (5) and P4-138 land. `OnceBox` in
   `HuffmanTable` is the remaining blocker — the standard Annex K tables are
   fixed data, so `const` construction is the cleanest route.
7. **An independent external audit** of public API soundness, SIMD dispatch,
   progressive decode, `BitWriter`, layout arithmetic, C handle lifecycle,
   classic ABI struct boundaries, allocator ownership, and panic/concurrency —
   published with its commit hash and unresolved findings — before any release
   carrying a memory-safety claim.

**Why P2.** It is the evidence layer. Sequenced after the P0 fixes because
several criteria here exist specifically to prove those fixes hold, and writing
the harness first would only pin current behaviour.

## P4-142. `tj3DecompressHeader` Decodes the Entire Image to Read the Header — **OPEN**

**Motivation.** Found 2026-08-09 while building P4-127's header-only path. Filed as P4-129 and renumbered to P4-142 the same day: P4-129 through P4-141 were already claimed by the 2026-08-09 architecture-audit issues (#460, #461, #474-#480) and by `docs/expert-audit-2026-08-09`, which had not merged yet, so `origin/main`'s phase file did not show them. Checking only the phase file is not enough — open issues and unmerged branches claim IDs too.
`TjHandle::decompress_header` (`src/api/tj3.rs`) does not parse a header — it
calls `self.decompress(data)` and throws the `Image` away:

```rust
let result: Result<Image> = self.decompress(data);
result.map(|_| ())
```

Upstream's `tj3DecompressHeader` performs `jpeg_read_header` and stops. Reading
the dimensions of an image is one of the most common things a caller does before
deciding whether to decode it at all — a thumbnail service checking size, a
validator screening uploads — and this port charges a full decode for it. The
cost is the entire image, and it is attacker-controlled, so it is the same
resource-amplification shape P4-127 just removed from the YUV entry points, on a
more frequently used call.

**Root cause.** No header-only path existed when `decompress_header` was
written. One does now: `TjHandle::inspect_header` (added by P4-127) parses
markers via `Decoder::new_with_limits` and applies the handle's limits without
touching pixel data.

**Acceptance criteria.**

1. `tj3DecompressHeader` reads the header without decoding, proven the way
   P4-127's regressions prove it: a fixture with a valid header and a corrupt
   entropy segment must succeed, since upstream's `jpeg_read_header` succeeds on
   one.
2. The handle params it publishes (width, height, subsampling, colorspace,
   precision, progressive/lossless flags) are unchanged for well-formed input —
   cross-checked against C for a matrix of subsamplings and colorspaces.
3. Decide what happens to errors that only a decode can find. Today a corrupt
   scan makes `tj3DecompressHeader` fail; upstream's succeeds and defers the
   failure to `tj3Decompress*`. Criterion 1 changes that observable behaviour,
   so the C cross-check must pin it rather than assume it.

**Why deferred.** Not a correctness bug for well-formed input, and P4-127's
patch is already at the size where a second behavioural change to a different
entry point belongs in its own review. Criterion 3 in particular needs its own
C comparison: it flips a currently-failing call into a succeeding one, which is
exactly the kind of change that should not ride along in a patch about YUV
validation order.

## P4-143. `.cargo/config.toml` Forces `+simd128`, Hiding wasm Target-Feature Regressions From the Whole Matrix — **CLOSED 2026-08-13**

**GitHub:** filed from the [#474](https://github.com/developer0hye/libjpeg-turbo-rs/issues/474) (P4-135) review.

**Motivation.** `.cargo/config.toml` sets
`rustflags = ["-C", "target-feature=+simd128"]` for both `wasm32-unknown-unknown`
and `wasm32-wasip1`. Every local build, every `wasm.yml` job and every developer
therefore compiles wasm **with** simd128 — and nothing in the matrix ever
compiles it without.

A downstream crate inherits none of that: `.cargo/config.toml` applies to builds
*run from this directory tree*, not to consumers who depend on the crate. So the
configuration CI exercises is not the configuration most wasm consumers get.

**How it was found.** P4-135 criterion 5 narrowed the arch module `cfg`s to
require `feature = "simd"`. For `wasm32` the natural second condition looked
like `target_feature = "simd128"`. That built clean locally, passed
`cargo clippy --workspace --all-targets`, and passed the full workspace suite —
because `+simd128` was forced the whole time. It is nonetheless a compile break
for any consumer building baseline `wasm32`: about six call sites in
`encode/pipeline_impl/{dispatch,sampling}.rs` are guarded by the *Cargo* `simd`
feature alone and would reference a module that no longer exists (`E0433`),
turning the scalar fallback documented in `crates/libjpeg-turbo-rs-wasm/README.md`
into a hard error. Review caught it; no automated gate did.

**Root cause.** Two independent conditions — the Cargo `simd` feature and the
`simd128` target feature — are used interchangeably across ~30 sites:

* `encode/pipeline_impl/{dispatch,sampling}.rs` — Cargo `feature = "simd"` only.
* `encode/pipeline_impl/mcu.rs` — both.
* `simd/{neon_color,neon_idct,neon_upsample,simd_neon_encode,simd_neon_scaled,simd_parity}_tests.rs`
  — `target_arch` alone (11 aarch64 sites and 8 wasm `simd128` sites in
  `simd_parity_tests.rs` alone).
* `simd/mod.rs` `detect`/`detect_encoder` — aarch64/x86_64 on the Cargo feature,
  wasm32 on `simd128`.

**Acceptance criteria.**

1. A CI leg builds `wasm32-unknown-unknown` **without** `+simd128` — i.e. not
   inheriting `.cargo/config.toml`, i.e. with an explicit empty `RUSTFLAGS` or
   from outside the tree — and fails if the scalar fallback does not compile.
2. Every SIMD call site states which condition it depends on, and the module
   `cfg`s are narrowed to match. This is the prerequisite for P4-135 criterion 5.
3. A note in `.cargo/config.toml` itself recording that it changes what the test
   matrix covers, so the next person narrowing a `cfg` does not repeat this.
4. Audit whether the same masking applies elsewhere: `aarch64` NEON is mandatory
   so it has no equivalent, but the x86_64 `target-cpu=native` question in
   [P4-133](#p4-133-bmi2fma-paths-are-reachable-only-via-target-cpunative-so-portable-builds-leave-them-off--open)
   is the same class of "CI tests a configuration consumers do not get".

**Why it matters beyond P4-135.** It is a *coverage* defect, not a code defect:
the tests are green on a build nobody ships. Any future change to a wasm `cfg`
is equally invisible until a user reports it.

**Status (2026-08-13): closed.** All four criteria delivered alongside P4-135
criterion 5:

1. The `Check baseline wasm32` leg in `wasm.yml` compiles
   `wasm32-unknown-unknown` and `wasm32-wasip1` with `RUSTFLAGS: -D warnings` —
   the env var overrides the config's target rustflags, removing `+simd128`,
   and denying warnings makes compiled-but-gated-out SIMD code fail the leg.
   Verified discriminating: against the pre-alignment tree the leg fails with
   67 dead-code errors; against the aligned tree it passes.
2. Every SIMD call site now states the canonical predicate for its arch —
   `all(target_arch, feature = "simd")` for aarch64/x86_64,
   `all(target_arch = "wasm32", feature = "simd", target_feature = "simd128")`
   for wasm — and the module gates in `src/simd/mod.rs` match. 13 pipeline
   sites gained the missing `simd128` condition (including one in
   `src/api/progressive_output.rs` outside the ~30 the filing counted), 9
   `src/simd/*_tests.rs` files gained the missing Cargo-feature condition,
   and `detect`/`detect_encoder`'s wasm arms gained it too. Wasm SIMD test
   coverage is unchanged: 17 `simd::` tests run under wasip1+simd128 before
   and after the alignment.
3. `.cargo/config.toml` opens with a note recording that its rustflags change
   what the matrix covers and naming the CI leg that compensates.
4. Audit of the same masking elsewhere: aarch64 NEON is mandatory (no
   equivalent gap); the x86_64 `target-cpu=native` case is the same
   "CI tests a configuration consumers do not get" class and is already
   tracked as its own item —
   [P4-133](#p4-133-bmi2fma-paths-are-reachable-only-via-target-cpunative-so-portable-builds-leave-them-off--open)
   — so it stays there rather than being duplicated. `.cargo/config.toml`
   sets no rustflags beyond the two wasm targets, so nothing else is masked
   repo-wide; the per-job `RUSTFLAGS` in `cross-arch.yml`/`armv7.yml`/
   `sanitizers.yml` add configurations rather than hiding one.

## P4-163. `jpeg_read_coefficients` Does Not Port Upstream's Improper-Usage Refusal — **OPEN**

**Motivation.** Surfaced by the P4-104 closure's drift audit (2026-08-14):
the state-machine item that tracked classic decompressor guards closed, and
this one guard has no other home. Upstream's `jpeg_read_coefficients`
refuses improper usage after its absorb loop: unless it lands in
`DSTATE_STOPPING`, or is invoked mid-buffered-image (`DSTATE_BUFIMAGE` with
`buffered_image` set) to expose the coefficient arrays, it raises
`ERREXIT1(JERR_BAD_STATE, global_state)` and returns NULL
(`jdtrans.c:84-94`). This shim's `jpeg_read_coefficients` walks
READY→RDCOEFS→STOPPING for the standalone flow but has no equivalent
refusal for out-of-order entry (e.g. from SCANNING mid-scanline-decode),
where upstream errors and we return arrays or NULL without `error_exit`.
Related: the P4-104-family guards for `read_header`/`start_output`/
`finish_output`/`finish_decompress` all landed with #468; this is the one
classic decompress entry point still without its upstream state guard.

**Acceptance criteria.** A differential oracle row (the
`classic_lifecycle_state_oracle.c` family) calling
`jpeg_read_coefficients` from a mid-decode state shows the same
`JERR_BAD_STATE(state)` on stock and shim; the standalone and
buffered-image-access flows keep working (existing `capi_preload_resume`
coefficient trace stays green).


## P4-144. Metadata Copies Are Input-Sized But Still Allocate Infallibly — **CLOSED 2026-08-12**

**GitHub:** [#512](https://github.com/developer0hye/libjpeg-turbo-rs/issues/512) — filed 2026-08-10 while closing
[P4-136](#p4-136-progressive-output-calls-set_len-on-uninitialized-vec-after-an-unchecked-size-multiplication--closed-2026-08-10);
under the [#481](https://github.com/developer0hye/libjpeg-turbo-rs/issues/481) umbrella.

**Motivation.** P4-136 made every *geometry-sized* allocation in the progressive
decoder fallible, so a hostile SOF can no longer abort the process by asking for
more memory than the machine has. The metadata path was deliberately left out of
that change and is now the remaining infallible input-sized allocation on the
decode path.

**Confirmed instances.** In `src/api/progressive_output.rs`:
`icc::reassemble_icc_profile(&self.metadata.icc_chunks)` (`:391`) and the
`exif_data` / `xmp_data` / `iptc_data` / `saved_markers` / `comment` clones that
build each `Image` (20 `.clone()` sites in the file). The same shape exists in
the baseline decoder — `reassemble_icc_profile` alone has 11 call sites.

**Why it is P2, not P0.** These allocations are *bounded by the input the caller
already holds*: a 100 MB JPEG carries at most ~100 MB of ICC/XMP. They do not
amplify. The geometry-sized allocations P4-136 fixed did — a 300-byte SOF can
demand 8 GiB — which is why they came first. Extended XMP reassembly is the
closest thing to an amplifier here (a multi-segment packet is reassembled into
one buffer) and is the instance to measure first.

**Acceptance criteria.**

1. `icc::reassemble_icc_profile` and the Extended XMP reassembly path return
   `Result`, allocating with `try_reserve_exact` and reporting
   `JpegError::AllocationFailed`.

   **Revised twice during implementation.** The ICC half landed as a *new*
   `try_reassemble_icc_profile`; the existing `reassemble_icc_profile` is
   public API and keeps its `Option` return, folding refusal into `None`
   rather than breaking every downstream caller. See the Status note.

   **Revised 2026-08-12 during implementation, for Extended XMP only.** That
   path allocates with `try_reserve_exact` as required, but *degrades* on
   refusal — skipping the extension and keeping the standard packet — instead
   of reporting the error. The criterion was written before its local contract
   was noticed: `decode/marker.rs` states three lines above that a broken
   extension must not fail an otherwise-valid decode. "The allocator said no"
   is a worse reason to break that than "the chunks were malformed", and an
   error return there would also abandon the rest of the metadata the function
   has already parsed. ICC is unchanged and still reports, because it is
   reassembled at `Image` construction where refusal means the decode genuinely
   cannot complete. Recorded here rather than left as an unmet criterion under a
   CLOSED heading.
2. `Image` construction in both the progressive and baseline decoders propagates
   that error rather than cloning infallibly.
3. A test proving refusal is recoverable, in the shape of P4-136's
   `allocator_refusal_is_an_error_not_an_abort` — including its ASan caveat
   (`ASAN_OPTIONS=allocator_may_return_null=1`, already set in
   `sanitizers.yml`) and its Miri caveat (Miri aborts instead of returning
   null, so the probe is `cfg_attr(miri, ignore)`).
4. Record whether the `Result` threading is worth it for the non-amplifying
   clones, or whether criterion 2 should stop at the reassembly buffers. Decide
   it here; do not leave it implicit.

**Status (2026-08-12): closed.** `cargo test --workspace --release` passed 2592
tests at this closure; the live figure is the one in `LAST_MILE.md`, which later
closures move. The three helpers P4-136 left in `api::progressive_output` moved to
`common::try_alloc` — copying them into `common::icc` and `decode::marker`
would have been the `compress_*` family P4-40 was filed for, one file later.

**Every metadata copy on the decode path, not just the first.** Review found
the first version protected only the four copies at the top of
`decode_image_inner`, while downstream constructors and the progressive, 12-bit
and lossless paths still cloned straight from `self.metadata`. **85 sites**
across six files now go through the helpers — and every one compiled on the
first attempt, because each was already inside a function returning `Result`.
That is the same evidence criterion 4 turns on.

`comment: Option<String>` and `saved_markers: Vec<SavedMarker>` are included:
smaller than an ICC profile, but "small" is not "zero", and a deep clone of the
marker list allocates once per marker. A second review pass found the *owned
locals* were still cloned infallibly into each `Image` — 34 more sites in
`output.rs` alone, plus `probe()` and `tj3.rs`. Grepping the five field names
across the decode path now returns **zero** infallible clones, which is the
check the closure rests on rather than a count of what was edited.

**The public signature is unchanged.** `common::icc::reassemble_icc_profile` is
exported from the crate root, so turning it into a `Result` would break every
downstream caller for a change they did not ask for. It keeps returning
`Option<Vec<u8>>` and folds refusal into `None` — still an improvement, since it
used to abort — while a new `try_reassemble_icc_profile` carries the error for
the internal decode path. Review caught that; the first version was a silent
SemVer break.

**Criterion 4, decided rather than left implicit: the clones are threaded too.**
The item anticipated that making `Image` construction fallible would cost API
churn and asked whether it was worth it for allocations that do not amplify. It
does not cost churn: `decode_image_inner`, `output()` and `finish()` already
return `Result<Image>`, so `try_clone_opt` is a helper call at four sites, not a
signature change. The reasoning for *not* doing it — the caller already holds
the bytes — is still true, and still does not make an abort acceptable: the
clone doubles the peak for that blob, and a refusal there was previously
unrecoverable.

**Sorts allocate too.** `sort_by_key` is stable, so it reserves scratch — which
would abort after both explicit buffers had been reserved fallibly. Keys are
unique in both reassembly paths (duplicate sequence numbers and non-contiguous
offsets are rejected before the sort), so `sort_unstable_by_key` is equivalent
and allocates nothing. The ICC chunk list was likewise still an infallible
`collect`.

**One deliberate asymmetry.** Extended XMP reassembly (`decode/marker.rs`)
*degrades* on refusal rather than erroring: it skips the extension and keeps the
standard packet. All **three** of its allocations are covered — review caught
that the first version guarded only the largest: the `ext` buffer, the chunk
reference list, and growing `std_packet` by up to the 64 MiB ceiling. The contract three lines above it is that a broken extension
must not fail an otherwise-valid decode, and "the allocator said no" is a worse
reason to break that than "the chunks were malformed". ICC does the opposite and
propagates, because it is reassembled at `Image` construction, where a refusal
means the decode genuinely cannot complete. The guard order there was also
preserved deliberately: the 64 MiB ceiling and the bytes-actually-present test
run *before* the allocation, so a tiny file declaring a huge `full_len` is
rejected without allocating — an intermediate version of this fix had inverted
that.

`try_reassemble_icc_profile` returns `Result<Option<Vec<u8>>>` — the *public*
`reassemble_icc_profile` keeps its `Option` signature, see above — keeping the
two outcomes distinct: `Ok(None)` is a malformed or absent profile, which must stay
soft, and `Err` is the allocator refusing. Collapsing them would turn "this
file's ICC is broken" into "this decode failed".

**Where this stops, explicitly.** P4-144 covers the metadata *copies*. The
**originals** are built one layer earlier by the marker parser, which still uses
`.to_vec()` and infallible growth in `read_app1`/`read_app2`/`read_app13`/
`read_com`/`peek_marker_data` — so a file carrying EXIF, XMP, IPTC, ICC or COM
can still abort during `read_markers`, before any of this is reached. Filed as
**P4-153** rather than folded in: the parser needs the degrade-versus-error
question answered per segment kind, which is the work, and this item's instance
list does not name it.

**Not an amplifier, and the measurement is why.** The item asked for Extended
XMP to be measured first as the closest candidate. It is already bounded by
both a 64 MiB ceiling and `available >= full_len` from an earlier review, so a
declared length cannot conjure an allocation. Every allocation this item touches
is bounded by input the caller already holds — which is exactly why it was P2
and why P4-136's geometry-derived buffers came first.

## P4-145. `TJPARAM_NOREALLOC` Is Honoured by `tj3Compress8` Only — **CLOSED 2026-08-12**

**GitHub:** [#514](https://github.com/developer0hye/libjpeg-turbo-rs/issues/514) — filed 2026-08-11 while closing
[P4-137](#p4-137-c-abi-raw-pointer-exports-are-safe-rust-functions--closed-2026-08-11);
found by that PR's codex review.

**Motivation.** Upstream TurboJPEG lets a caller pre-allocate the output buffer
and set `TJPARAM_NOREALLOC` to promise the library will not resize it. That
promise is what makes a caller-owned buffer usable at all: with it, the pointer
the caller passed is the pointer it gets back.

We honour it in **`tj3Compress8` only**. `tj3Compress12`, `tj3Compress16`,
`tj3CompressFromYUV8`, `tj3CompressFromYUVPlanes8` and `tj3Transform` never read
the parameter — each allocates a fresh buffer and `libc_free`s the previous
pointee unconditionally, even when the caller's buffer was large enough.

**One instance was a live heap overflow and is already fixed.** `tj3Compress8`
*does* read the flag, but its in-place path ignored `*jpeg_size` — the input
capacity — and `copy_nonoverlapping`'d the encoded output on the assumption the
buffer was at least `tj3JPEGBufSize(...)`. Upstream instead raises
`JERR_BUFFER_SIZE` (`jdatadst-tj.c:92`), so a caller doing exactly what upstream
permits — a smaller buffer, its size declared — got a heap overflow.
Fixed 2026-08-11 under P4-137: the capacity is now compared before the copy.
`norealloc_buffer_capacity.rs` pins both directions, and removing the check
reproduces `AddressSanitizer: heap-buffer-overflow` on the too-small case.

**Two consequences of the remaining five, and the second is the serious one.**

1. A caller that pre-sized its buffer still gets a different pointer back, so
   any C code holding the original is left with a dangling pointer.
2. `NOREALLOC` is precisely the flag a caller sets when the buffer is *not*
   `malloc`-owned — a stack array, or a `Vec` on the Rust side. Upstream's
   contract makes that safe. Ours frees it, with the wrong allocator.

Documented at the crate root and on all five entry points as of 2026-08-11, so
the contract is at least honest; but documentation is the workaround, not the
fix.

**Acceptance criteria.**

1. All five entry points read `TJPARAM_NOREALLOC`, following `tj3Compress8`'s
   shape — which as of 2026-08-11 includes the capacity check: write in place
   when set, and raise the documented "buffer too small" error rather than
   reallocating or overrunning.
2. The previous pointee is freed only on the reallocating path.
3. A test per entry point proving the caller's pointer is unchanged when the
   flag is set and the buffer is large enough — the observable difference, and
   the one a doc fix cannot deliver.
4. The crate-root **Ownership transfer** note and the five `# Safety` sections
   drop the P4-145 caveat once (1)-(3) land.

**Status (2026-08-12): closed.** `cargo test -p libjpeg-turbo-rs-capi --test
norealloc_all_entry_points` passes 14 tests: one per compressing entry point
asserting the caller's pointer comes back **unchanged** when the flag is set
and the buffer fits, the refusal case, the upstream-oracle comparison, two
legacy-wrapper cases (flag propagation, output-slot semantics, fresh-handle
sizing, validation ordering, NULL size), and `TJXOPT_NOOUTPUT`. Each was verified red by restoring
the old path for that function: the assertion that fires is pointer identity,
which is exactly what a documentation fix could not deliver and what a
"did the encode succeed?" test would have missed throughout.

All six now share one implementation, `alloc::deliver_compressed_output`. That
matters as much as the fix: six copies of an ownership rule is how the
`compress_*` family P4-40 was filed for came about, and this bug is what the
sixth copy diverging looks like. The helper frees the previous pointee **only**
on the reallocating path (criterion 2), which also removed a leak in
`tj3Compress8` — it had been dropping the prior pointer on that path rather
than freeing it, on the grounds that its allocator was unknown. Upstream
reaches the same state by `realloc`, which consumes it, so freeing is what the
contract already required of callers.

The crate-root *Ownership transfer* note and all five `# Safety` sections drop
the caveat (criterion 4) and now state the rule positively: the slot is freed
only when the flag is unset.

**Review found the first version got a case wrong that no self-consistent test
could have caught.** With the flag set and the output slot **NULL**, it
allocated. Upstream refuses: `jdatadst-tj.c:184-192` takes the
`*outbuffer == NULL` branch and, with `alloc` false, raises `JERR_BUFFER_SIZE`.
The flag is a request *not to allocate*, so honouring it half-way — refusing to
grow a buffer but conjuring one when none was given — is the one behaviour no
caller asked for. All six now return `NoBufferSupplied` there.

That is why `examples/norealloc_oracle.c` exists. It runs the same cases
against real TurboJPEG and prints `label rc kept produced`; the suite compares
line for line. It covers **all six** entry points — roomy, cramped and NULL
each — rather than the two that happened to be written first: the NULL-slot
divergence is proof that a suite of self-consistency assertions stays green
while one call diverges. Extending it also surfaced P4-150 (`tj3Compress16`
accepts lossy 16-bit where upstream refuses), which is why those lines
configure `TJPARAM_LOSSLESS`. Without the refusal, the diff is exact:

```
ours  compress8_null 0 0 1     (succeeded, swapped the pointer, produced output)
C     compress8_null -1 1 0    (refused, pointer untouched)
```

The byte count is deliberately excluded from the comparison: two independent
encoders do not agree on entropy-coded output size, and a trace carrying it
would fail for a reason unrelated to the ownership contract. `rc` and pointer
identity are what the contract actually specifies.

Two further paths, also from review:

- **The legacy flag never reached the parameter.** `tjCompress2` and
  `tjTransform` take `TJFLAG_NOREALLOC` and discarded it, so `tj3Compress8` saw
  `TJPARAM_NOREALLOC` unset. That was survivable only while the reallocating
  path *leaked* the previous pointer; making it `free()` — which is what this
  change did, and what upstream's `realloc` does — turned it into an invalid
  free of caller-owned storage. **A regression introduced by the fix itself**,
  and the reason that leak had been load-bearing. Both now map the flag, as
  upstream's `processFlags` does for every operation (`turbojpeg.c:552`).
- **And mapping the flag alone was still wrong — for `tjCompress2`.** The legacy size slots are
  *outputs*, not capacities: a caller that sized its buffer with `tjBufSize()`
  has no reason to write `*jpegSize`, so forwarding it to TJ3 — where the same
  field *is* an input capacity — turned a valid call into "buffer too small".
  Upstream substitutes the worst case instead: `tj3JPEGBufSize(...)` in
  `tjCompress2` (`turbojpeg.c:1282-1284`) and a per-image temporary array in
  `tjTransform` (`turbojpeg.c:3118-3132`). **Only the `tjCompress2` adapter is
  ported** — see the next point for why the transform one is not. The
  distinguishing input is `size = 0`, which the first legacy test could not
  catch because it passed a real capacity.
- **The transform side of that bridge is not shipped**, and the reason is
  worth recording: two attempts were rejected in review, first for computing a
  capacity of 0 on a fresh handle, then — after adding a header parse — for
  deriving the capacity from `tj3TransformBufSize`, which adds the extracted
  ICC length and so overruns a `tjTransformBufSize()`-sized buffer, and for
  mutating the handle's compression state along the way. Split out as
  **P4-151**, with those two failures as its acceptance criteria. The flag
  mapping — the memory-safety half — did land.
- **Ordering matters too.** Mapping the flag before validating left a call that
  returned `-1` with the handle's ownership behaviour changed, so a later call
  could free caller-owned storage because of a failure. And routing the legacy
  call through a local `size_t` hid a NULL `jpegSize` from `tj3Compress8`,
  turning a call upstream rejects into a success that allocated a buffer whose
  size the caller could never learn. Both now validate first
  (`turbojpeg.c:1274-1280`).

Four review rounds produced five defects *in the fix*, every one on the legacy
wrappers rather than the TJ3 entry points the item named. The pattern is worth
keeping: adapting an API whose field *semantics* differ — output slot versus
input capacity — is where the errors were, not in the ownership rule itself.
- **`TJXOPT_NOOUTPUT` needs no destination.** Upstream skips destination setup
  entirely for it (`turbojpeg.c:3007`), so a NULL slot succeeds and a non-NULL
  slot is left alone. Delivery now returns early for that option instead of
  demanding a buffer for output that was never produced.

**Found in passing:** `tj3.rs`'s module doc said `TJINIT_COMPRESS = 1,
TJINIT_DECOMPRESS = 2, TJINIT_TRANSFORM = 4` "(bit flags; callers may OR them
together)", contradicting `tj3Init`'s own doc twelve lines below, the
`0..TJ_NUMINIT` range the code accepts, and `turbojpeg.h:91-105`, where they
are a plain enum. Corrected. The first draft of the new test believed the
module doc and failed at `tj3Init(4)`.

## P4-146. `jpeg_std_error` Leaves `jpeg_message_table` Null, So Every Classic Error Formats as "bogus message code" — **CLOSED 2026-08-13**

**GitHub:** [#518](https://github.com/developer0hye/libjpeg-turbo-rs/issues/518) — filed 2026-08-11 from the P4-14 review. Affects every classic
`JERR_*`, not one code.

**Motivation.** `jpeg_std_error` sets `jpeg_message_table = std::ptr::null()`
(`jpeglib.rs:1479` as of filing; that assignment is now the table install at
`jpeglib.rs:1684`). `default_format_message` therefore always takes its
fallback and writes `"libjpeg-turbo-rs: bogus message code"` into the caller's
buffer — for *every* error, whatever `msg_code` says.

A C consumer's normal error-reporting path is `err->output_message(cinfo)` or
`err->format_message(cinfo, buf)`. Both go through that table. So a caller that
hits, say, the new `max_memory_to_use` guard gets `msg_code = 51` (correct) and
the string `"libjpeg-turbo-rs: bogus message code"` (useless) where stock
libjpeg-turbo prints `"Memory limit exceeded"`.

**Why it stayed invisible.** `capi_classic_error_codes.rs` verifies each code's
number *and* message against the pinned upstream headers — but it reads the
message from `jerror.h`, never from our formatter. The table is correct and the
shim cannot produce it. The test is a genuine parity check of the *constants*
and a false-green on the *rendering*.

That is the same shape as P4-120's complaint one level up: the code is pinned,
the payload is not.

**Acceptance criteria.**

1. `jpeg_std_error` populates `jpeg_message_table` with the standard message
   list and sets `last_jpeg_message`, so `format_message` renders upstream's
   text for every code the shim can raise.
2. Parameter substitution works for both shapes: `%d`/`%u` from `msg_parm.i`
   (`JERR_OUT_OF_MEMORY` case %d, `JERR_IMAGE_TOO_BIG` %u) and `%s` from
   `msg_parm.s`.
3. An end-to-end test drives a real failure through `format_message` and
   asserts the rendered string, for at least one parameterless code and one of
   each parameter shape. Extend `capi_classic_error_codes.rs` so its message
   column is checked against *our formatter*, not only against `jerror.h` —
   otherwise the same false-green returns.
4. `output_message` writes to `stderr` in upstream's format, and
   `trace_level`-gated `emit_message` calls stay silent by default.

**Status (2026-08-11): partial — the rendering defect is fixed.**

* **Criteria 1–2 — done.** `jpeg_std_error` installs a 129-entry table and sets
  `last_jpeg_message = 128`. Parameter substitution already worked — the
  formatter was complete, it simply had no table to read — so `%d`, `%u` and
  `%s` all render.

* **The table is generated, not transcribed.** `message_table.rs` comes from a C
  program that `#include`s `jerror.h` twice at `JPEG_LIB_VERSION 80`, exactly as
  the pre-implementation note on #518 recommended. That mattered: the header has
  **134** `JMESSAGE` lines but **129** entries at v8, because several are
  version-conditional and a few appear twice under opposite `#if` guards. A
  line-order parse counts both and misaligns everything after the first
  divergence — and in a code-indexed table a shifted entry is a *wrong* message,
  not a missing one.

* **Criterion 3 — done, and it closed the false-green.**
  `capi_classic_error_codes.rs` now renders every code **through our
  `format_message`** rather than only comparing `jerror.h` to a literal, and a
  new `the_whole_message_table_matches_upstream` re-runs the C probe over all
  129 entries. The old test reported "18 codes verified" throughout the period
  when every one of them rendered as `"bogus message code"`; it was a real
  parity check of the *constants* and a false-green on the *rendering*.
  `capi_error_message_rendering.rs` adds 6 cases: parameterless, `%d`, `%u`,
  out-of-range, boundary — verified to fail 4/5 against the null table — plus
  `a_real_failure_renders_its_message`, which is the one the criterion actually
  asks for. The other five write `msg_code` by hand, which proves the formatter
  and the table agree but nothing about whether a *failure* populates them; the
  sixth triggers P4-14's budget guard and formats from the same error manager
  the failure used.

  Both suites, and `capi_classic_error_codes` itself, are now named in
  `ci.yml`. That last one had never been named: it compiled on every PR and
  executed on none, which is exactly how it reported "18 codes verified"
  throughout the period when all 18 rendered as the fallback.

* **A second divergence, found by the new whole-table check.** Our fallback for
  an unknown code was a fixed `"libjpeg-turbo-rs: bogus message code"`.
  Upstream's is `msg_parm.i[0] = msg_code; msgtext = table[0]`
  (`jerror.c:173-175`), and entry 0 is `"Bogus message code %d"` — so upstream
  *names the code* and we dropped the one piece of information the message
  exists to carry. Now ported exactly; `render(100_000)` gives
  `"Bogus message code 100000"`. The old string survives only for a caller that
  built a `jpeg_error_mgr` without `jpeg_std_error`, where upstream would
  dereference a null table.

* **Criterion 4 — done 2026-08-13, closing the item.** `default_output_message`
  was a documented no-op; it now renders through the *installed*
  `format_message` and prints `%s\n` to stderr, exactly `jerror.c:95-110` —
  allocation-free (it runs on the paths that report allocation failure), with
  the assembled line issued as one raw `write(2)` to fd 2 rather than through
  `std::io::stderr()`, whose Rust-side lock cannot exclude a host's own
  `fprintf(stderr, …)`. `default_emit_message`'s display
  policy already matched `jerror.c:113-143`; it now also holds no Rust
  reference across the `output_message` callback (raw field reads/writes
  only), since that callback may be a caller's hook inspecting the same
  `jpeg_error_mgr`. Pinned by `capi_output_message.rs` (3 child-process tests,
  Red-verified against the no-op): formatted-text-plus-newline on stderr with
  `%02x` parameter substitution, routing through an overridden formatter, and
  the emit policy — trace above `trace_level` silent, level-0 advisory shown,
  first warning only, both warnings counted. The suite is named in `ci.yml`
  beside the other two, for the reason recorded above.

  An earlier draft closed this item and said criterion 4 was "tracked under
  P4-100's error-reporting scope". **That was wrong and was corrected here:**
  P4-100 is about failures surfacing as suspension or silent success — error
  *propagation* — and its acceptance criteria say nothing about `output_message`
  formatting or trace gating. Criterion 4 stayed here, measurable, until it was
  done.

**Status (2026-08-13): closed.** Criteria 1–3 delivered 2026-08-11 (rendering,
generated 129-entry table, whole-table gate); criterion 4 delivered 2026-08-13
by `capi_output_message.rs` as above.

## P4-154. Classic `jpeg_write_scanlines` / `jpeg_start_compress` Ignore `data_precision` Entirely — **CLOSED 2026-08-13**

**Motivation.** Found 2026-08-12 while closing P4-150. That item fixed the
TurboJPEG entry point; the same acceptance rule is missing one layer over, on
the classic C API, where it is broader and easier for a real caller to hit.

Measured against this shim, encoding a 16x16 RGB image at quality 80 through
`jpeg_set_defaults` → `cinfo.data_precision = N` → `jpeg_start_compress` →
`jpeg_write_scanlines`:

```
data_precision  8    629 bytes, no error
data_precision  9    629 bytes, no error
data_precision 12    629 bytes, no error
data_precision 16    629 bytes, no error
```

Byte-identical output for every value: the field is not read at all on this
path. Upstream has two gates, and the caller above trips the second one first:

- `jcapistd.c:92-105` — `jpeg_write_scanlines`, the *8-bit* entry point,
  raises `JERR_BAD_PRECISION` unless `data_precision == BITS_IN_JSAMPLE` (8),
  or, for a lossless compress, unless it is in `2..=8`.
- `jcmaster.c:199-208` — reached from `jpeg_start_compress`, admits 2..=16 for
  a lossless compress and only 8 or 12 for a lossy one.

So upstream rejects 9, 12 and 16 here; only 8 survives. We accept all four and
silently produce the same 8-bit stream.

**Why it matters.** Same class as P4-150 and P4-39: a caller gets output where
the library it replaced gave a documented error. Two things make the classic
surface worse than the TurboJPEG one. `data_precision` is a *public struct
field* a caller sets directly, with no setter to funnel validation through, so
the mistake is easy to make and invisible. And the 12-bit case is not merely a
missing error — a caller that sets `data_precision = 12` and calls the 8-bit
`jpeg_write_scanlines` is asking for something upstream refuses, while we
answer with an 8-bit stream that looks like success; a caller comparing file
sizes or bit depth downstream sees plausible, wrong output.

Note this is *not* the 12-bit raw-data encode path, which works and is tested:
`jpeg12_write_raw_data` already gates on `data_precision == 12`
(`jpeglib.rs:10059`), and `jpeg_write_raw_data` already gates on `== 8`
(`jpeglib.rs:9590`). It is specifically the 8-bit *scanline* entry point and
`jpeg_start_compress` that have no gate — the two upstream added them to.
`jpeg12_write_scanlines` / `jpeg16_write_scanlines` have none either: both
delegate to `write_scanlines_highprec` (`jpeglib.rs:10264`), which takes the
precision from the entry point and never reads `data_precision`, where
upstream's is the same `jcapistd.c:92-105` check compiled with
`BITS_IN_JSAMPLE` 12 and 16. Those two entry points are
[P4-94](#p4-94-classic-1216-bit-scanline-buffers-never-reach-a-high-precision-encoder--open)'s
subject; this item does not gate them.

**Acceptance criteria.**

1. `jpeg_write_scanlines` raises `JERR_BAD_PRECISION` with the offending
   precision as `msg_parm.i[0]` when `data_precision != 8` on a lossy
   compress, mirroring `jcapistd.c:102-103`; and accepts `2..=8` when the
   compress is lossless, mirroring `jcapistd.c:93-98`.
2. `jpeg_start_compress` raises `JERR_BAD_PRECISION` for a lossy compress whose
   `data_precision` is neither 8 nor 12, and for a lossless compress outside
   `2..=16`, mirroring `jcmaster.c:199-208`.
3. Which of the two fires first, for a caller that trips both, is decided by a
   C oracle rather than by reading — the error a caller sees is the first one,
   and the two gates disagree about 12.
4. Cross-validated by an oracle binary linked against real libjpeg (the
   `build_classic_oracle` path already used by P4-104 and P4-110), not by
   assertions transcribed from the C source. P4-145 and P4-150 both had their
   first version pass a self-consistent suite while diverging.

**Why deferred.** P4-150's PR is a TurboJPEG-scoped fix with its own oracle;
folding a second gate into a different API surface would put two unrelated
acceptance rules behind one review. Splitting is the same call made for P4-151.

**Status (2026-08-13): closed** (issue #538). Both gates are ported to where
upstream raises them: `jpeg_start_compress` admits {8, 12} for lossy and
2..=16 for lossless (`jcmaster.c:196-208`), and the 8-bit
`jpeg_write_scanlines` entry admits exactly 8 for lossy and 2..=8 for
lossless ahead of even its argument checks
(`jcapistd.c:92-105`) — both `ERREXIT1`-shaped, `JERR_BAD_PRECISION` with the
offending value in `msg_parm.i[0]`. Criteria 3-4's ordering question is
decided by `examples/classic_precision_oracle.c` (stock-libjpeg-linked via
`build_classic_oracle`), whose full 10-case (precision × lossless) trace —
stage, code, parm — is compared verbatim by
`capi_classic_compress_precision.rs`. Measured: lossy 2/9/16 and nothing else
fail at `start`; lossy 12 and lossless 9/12/16 fail at `write` (the two gates
disagree about lossy 12, which is why transcription was not an option);
lossy 8 and lossless 2/8 encode. Red-verified: before the gates every case
read `ok 0 0`.

The #538 review hardened three things. The accepted lines print **exact
output size and byte checksum**, and the accepted precision now reaches the
encoder (`compress_lossless_extended_precision`): the first version accepted
a 2-bit lossless request and silently emitted an 8-bit SOF3 stream, which an
`ok`-only trace cannot see — with the routing in place the checksummed trace
matches stock libjpeg byte-for-byte on every accepted case. The gate order
inside `jpeg_start_compress` is pinned by `mixed_width_overflow_precision_9`
(row wider than `JDIMENSION` *and* precision 9): upstream raises
`JERR_WIDTH_OVERFLOW` first (`jcmaster.c:190-208`), so the precision gate
sits after the row check. And the suite is named in `ci.yml` (P4-154 step) —
a suite nothing names never runs.

## P4-155. `TJPARAM_QUALITY` / `TJPARAM_SUBSAMP` Default to Set Values, So Upstream's "must be specified" Errors Can Never Fire — **CLOSED 2026-08-13**

**Motivation.** Found 2026-08-12 (issue #539) while closing P4-150. Upstream
initialises a TurboJPEG instance with `TJPARAM_QUALITY = -1` and
`TJPARAM_SUBSAMP = TJSAMP_UNKNOWN`, both meaning *unset*, and every lossy
compress entry point refuses until the caller supplies them
(`turbojpeg-mp.c:95-97`). This port initialises them to `quality = 75` and
`subsampling = 2` (S420) — `src/api/tj3.rs:141-142` — so neither error can fire.

Measured against TurboJPEG 3, calling `tj3Compress16` after setting only
`TJPARAM_LOSSLESSPSV` and `TJPARAM_LOSSLESSPT`:

```
TJPARAM_LOSSLESS reads back as 0
compress16 rc=-1 err="tj3Compress16(): TJPARAM_QUALITY must be specified"
```

**Why it matters.** Two consequences. *Silent substitution*: same class as
P4-150 and P4-39 — a caller who forgot to set quality gets 75 where upstream
gives a documented error. *Error precedence*: these checks sit at the top of
the refusal chain, above the destination setup and `jpeg_start_compress`.
Upstream's full order for a compress entry point is

1. argument validation (NULL, dims, pitch, pixel format)
2. `TJPARAM_QUALITY must be specified` (lossy only)
3. `TJPARAM_SUBSAMP must be specified` (lossy only)
4. destination setup — `Buffer passed to JPEG library is too small` under
   `TJPARAM_NOREALLOC` with a NULL or zero-capacity slot (`jdatadst-tj.c:184-192`)
5. `jpeg_start_compress` — e.g. `Unsupported JPEG data precision 16`
   (`jcmaster.c:199-208`)

P4-150 pinned steps 4 and 5 against a C oracle. Steps 2 and 3 cannot be pinned
while the defaults differ, because the branch is unreachable.

**Why deferred.** The defaults are read by every compress path in the crate, and
`tj3Get(TJPARAM_QUALITY)` returns 75 on a fresh handle today — a value callers
may already read back. Changing them flips behaviour for every entry point at
once, so it needs its own oracle matrix rather than riding along with a
precision fix. Whether `tj3Get` on a fresh handle should report `-1` is part of
the same question.

**Acceptance criteria.**

1. A fresh `TJINIT_COMPRESS` handle reports `TJPARAM_QUALITY = -1` and
   `TJPARAM_SUBSAMP = TJSAMP_UNKNOWN` through `tj3Get`.
2. Every lossy compress entry point refuses with upstream's message when either
   is unset; a lossless compress does not consult them.
3. The refusal order above is verified end to end by a C oracle covering all
   five steps, extending `examples/compress_precision_oracle.c` rather than
   duplicating it.
4. Legacy `tjCompress2` and friends, which set quality on every call, keep
   working and must not become sensitive to the default.

**Status (2026-08-13): closed** (issue #539). A fresh handle now carries
`quality = -1` and `subsampling = -1` (TJSAMP_UNKNOWN), reported verbatim by
`tj3Get` and cross-validated by the oracle's `p4155_fresh_get` line. The
gates are ported at both layers: the TJ3 entry points refuse with upstream's
message shape after argument validation — the lossy compress entries skip
them under `TJPARAM_LOSSLESS` (`turbojpeg-mp.c:95-98`), the YUV compress
entries gate quality unconditionally — `tj3CompressFromYUVPlanes8`
quality-then-subsampling (`turbojpeg.c:1347-1350`), while the packed
`tj3CompressFromYUV8` gates the subsampling itself first, because it needs it
to size the planes (`:1497-1498`), and reaches the quality gate only through
the delegate — and the YUV encode/decode entries need only the
subsampling, with *unset* distinguished from *out-of-range* — and the native
`TjHandle` compress methods carry the same refusal as a backstop for Rust
callers. Pinned by the `p4155_*` block in `compress_precision_oracle.c` /
`capi_compress_precision.rs` (17 lines: the fresh-handle readback, the c8
unset matrix, argument-error precedence, c12, the lossless bypass, and eight
YUV lines — the six entry-point shapes plus the packed wrappers' `align` and
pixel-format precedence, and the legacy `tjEncodeYUV3`/`tjDecodeYUV`
`align=0` rows — the re-review caught those wrappers clamping `align.max(1)`
and thereby masking the new `align < 1` validation, a silent accept where
upstream refuses; the clamp is gone and the raw value reaches the TJ3
entry), classified
on upstream's documented "must be specified" substrings and Red-verified:
before the change every unset case encoded with silent 75 / 4:2:0
substitutes and the fresh handle reported `quality=75 subsamp=2`.

The #539 review round (adversarial, standing in for the quota-blocked codex
pass) reordered three gates the first version got wrong, each measured
against stock TurboJPEG 3.1.4.1 before and after: packed `tj3CompressFromYUV8`
gates the subsampling in the entry itself — it needs it to size the planes
(`turbojpeg.c:1497-1498`) — and reaches the quality gate only through the
`…Planes8` delegate; and the packed `tj3EncodeYUV8` / `tj3DecodeYUV8`
wrappers gate the subsampling *before* the pixel-format range check, which
upstream performs in the delegates (`:1745-1750`, `:2721-2726`). The packed
entries also now validate `align` (power of two) in argument validation,
where it beats every gate. The discriminating oracle lines use an
out-of-range pixel format on purpose — a valid-format line passes with the
checks in either order. The round also added standalone `#[test]`s at both
layers so the gates stay covered without a TurboJPEG install, a CHANGELOG
entry for the breaking Rust-API default change, and filed **P4-158** for the
pre-existing `compress_12bit` lossless-flag gap the sentinel comment had
papered over.

## P4-156. Legacy NOREALLOC Transform Copies the Source ICC Profile Where Upstream Drops Every Marker — **CLOSED 2026-08-13**

**Motivation.** Found 2026-08-12 (issue #544) while adding C-oracle coverage for
P4-151; scoped by the review's probe of real TurboJPEG 3.1.4.1. **The divergence
is confined to legacy `tjTransform` under `TJFLAG_NOREALLOC`.** Measured with a
32x32 grayscale source carrying a 5000-byte profile, identity transform:

```
                              C 3.1.4.1   ours
  legacy tjTransform, NOREALLOC   601     5619   <- the only divergent path
  legacy tjTransform, flags=0    5619     5619   matches
  tj3Transform                   5619     5619   matches
```

Under NOREALLOC we refuse (5619 does not fit the 4096-byte
`tjBufSize(32,32,TJSAMP_GRAY)` capacity) where C succeeds at 601 bytes. The
refusal is **correct for the size produced** — P4-151's bridge computes capacity
from geometry alone, as upstream does. The defect is upstream of that: on this
path we should not have produced 5619.

**Root cause in upstream, which parity must mimic.** Upstream's marker copying
is gated by `jcopy_markers_setup(dinfo, saveMarkers)` — registration that must
run *before* the header is parsed (`turbojpeg.c:2976-2979`, default
`saveMarkers = 2` = `JCOPYOPT_ALL`). The legacy NOREALLOC wrapper, uniquely,
pre-reads the header to derive per-transform capacities
(`turbojpeg.c:3112-3134`) — so when `tj3Transform` later calls
`jcopy_markers_setup` and finds `global_state > DSTATE_INHEADER`, the guarded
re-read is skipped, nothing was registered, and `jcopy_markers_execute` copies
*no* markers at all (not just ICC: COM and every APPn die too). On the other
two paths registration precedes the read and everything is copied — which this
port already matches. The C behaviour to replicate is an emergent ordering
quirk, not a marker policy.

**Why it matters.** A legacy NOREALLOC caller that allocates
`tjTransformBufSize()`, as the API documents, is refused for any source
carrying a profile larger than the slack in that bound — upstream succeeds.
The other paths need no change: a fix that strips the ICC generally would
*create* divergence on `flags=0` and `tj3Transform`, where copying is correct.

**Acceptance criteria.**

1. On legacy `tjTransform` + `TJFLAG_NOREALLOC`, an identity transform of an
   ICC-carrying source drops markers exactly as upstream's ordering quirk does,
   cross-validated against the C library's trace rather than against a reading
   of this description.
2. Legacy `flags=0` and direct `tj3Transform` keep copying markers byte-
   identically, and `TJXOPT_COPYNONE` is traced on both shapes — the fix must
   be provably scoped to the one divergent path.
3. `examples/norealloc_oracle.c`'s `legacy_transform_gray_no_overrun` line
   compares exact size and return code, not just the no-overrun invariant it is
   narrowed to today. The helper is already written and compiled; the narrowing
   exists only because this divergence would have failed the line for a reason
   unrelated to P4-151.
4. Whether the grayscale case then succeeds or refuses must match C, not merely
   be self-consistent.
5. **Re-pin P4-151's capacity-derivation regression by mechanism.** Both
   `legacy_tj_transform_sizes_a_grayscale_source_as_gray` and the oracle's
   gray line currently force the payload across the grayscale bound with a
   5000-byte ICC profile. Once this item lands, no marker survives that path
   and no legal Huffman payload of a 32x32 grayscale image can cross
   `tjBufSize`'s 2-bytes/px + 2048 slack — the fixtures pass vacuously whether
   or not the Unknown→S444 sizing bug returns. Whoever closes this item must
   replace the ICC-inflation mechanism with a direct pin of the derived
   subsampling (or an equivalent observable), not merely delete the fixtures.

**Status (2026-08-13): closed.** The bridge reproduces the ordering quirk by
forcing `TJXOPT_COPYNONE` on a local copy of the caller's transforms — only on
the legacy NOREALLOC path, never on the caller's array (`legacy.rs`,
`tjTransform`). Our `MarkerCopyMode::None` matches the quirk's whole effect
because this port's transform writes no handle-level ICC either, exactly as
upstream's `copyOption == JCOPYOPT_ALL` guard skips `jpeg_write_icc_profile`.

**The quirk is per-handle state, not per-call** — the #548 review's
differential probe caught the first version treating it as unconditional.
Upstream's `jcopy_markers_setup` registration is permanent for the handle's
life (there is no unregister API and it outlives `jpeg_abort_decompress`), so
the pre-read starves marker saving only on a *cold* handle; on a *warm* one
the processors are already registered, markers survive, and the ICC-carrying
copy exceeds the grayscale bound — upstream refuses where the unconditional
version succeeded markerless. Modeled as
`TjInstance::transform_markers_registered`, set by any batch that would have
registered processors upstream (a non-COPYNONE transform with
`TJPARAM_SAVEMARKERS` nonzero) in both `tj3Transform` and the bridge — the
starved read still registers for *later* calls, so the first cold NOREALLOC
call warms the handle for the second. Trace-verified transitions
(`fixture_state_cases`, both sides): `fx_warm_after_flags0 -1 1 0`,
`fx_norealloc_first 0 1 601` then `fx_norealloc_second -1 1 0`,
`fx_cold_after_copynone 0 1 601`. Red-checked by making the quirk
unconditional again: the two warm lines read `0 1 601` against C's `-1 1 0`.
The same review also made both bridge allocations fallible
(`try_reserve_exact` → `"Memory allocation failure"`, matching upstream's
THROW) since `n` is caller-controlled and Rust's infallible path aborts a C
host.

Criteria 1–4 are delivered by the oracle's `fx_*` family, which supersedes the
narrowed `legacy_transform_gray_no_overrun` line criterion 3 named: the Rust
harness now generates one ICC-carrying grayscale fixture, hands it to
`norealloc_oracle` as `argv[1]`, and both sides transform *identical bytes*,
so the six lines compare `rc` and **exact byte size** (transforms of identical
input are byte-exact between the implementations; the stock-tool gate pins that
for `jpegtran -copy all -rotate 90` over the upstream corpus).
Measured against TurboJPEG 3.1.4.1: `fx_legacy_norealloc` 601 bytes both
sides (quirk parity — criterion 1/4), `fx_legacy_flags0` and `fx_tj3_realloc`
5619 both sides (the correct paths stayed correct — criterion 2), and the
three `*_copynone` variants 601 each (COPYNONE traced on every shape —
criterion 2). Red-checked by disabling the injection: ours reads `-1 1 0`
against C's `0 1 601` on the first line only. Four state-transition lines
join them per the warm-handle model above.

Criterion 5 is the `legacy_norealloc_capacities` unit tests in `legacy.rs`
(gray-vs-4:4:4 derivation, probe-subsampling passthrough, per-transform
geometry), extracted from the bridge so the derivation is pinned without any
payload crossing a bound; the standalone
`legacy_tj_transform_sizes_a_grayscale_source_as_gray` now asserts the
post-quirk contract (success within the grayscale bound, canary untouched).
`norealloc_all_entry_points`: 18/18.

## P4-158. Native `compress_12bit` Ignores `TJPARAM_LOSSLESS` and Encodes a Lossy Stream — **OPEN**

**Motivation.** Found by the #539 review round (2026-08-13). `TjHandle`'s
`compress_12bit` / `compress_12bit_with_precision` dispatch on component
count only (`src/api/precision.rs`) and never read `self.lossless`: a caller
who sets `TJPARAM_LOSSLESS` and calls them gets `Ok(jpeg)` holding a **lossy**
SOF1 stream — neither the requested format nor an error. The P4-150 / P4-39
shape at the native layer.

P4-155 makes it observable one step earlier: `require_lossy_params` waves the
lossless flag through with quality unset, and the sentinel placeholder (75)
then feeds `quality_scale_quant_table_ext` in a stream the caller never asked
for. The placeholder becomes unobservable only once this item routes the
flag; until then the comment at those call sites names this item instead of
claiming the property.

**Acceptance criteria.**

1. With `TJPARAM_LOSSLESS` set, `compress_12bit` produces an SOF3 lossless
   stream honouring PSV/Pt (or refuses with a documented error if 12-bit
   lossless is out of scope for it), cross-validated against C.
2. The unset-quality sentinel provably cannot reach quantization on any path
   (the `configure_encoder` shape, where `.lossless(...)` is passed through,
   is the model).
3. The capi `tj3Compress12` route is traced for the same configuration.

## P4-159. `jpeg_read_coefficients` Returns NULL on Multi-Scan Sequential Streams — **OPEN**

**Motivation.** Found by the P4-14 adversarial review (2026-08-13), measured
against stock 3.1.4.1: on a non-interleaved *sequential* JPEG (SOF0, one
full-spectral scan per component — what `cjpeg -scans` produces, committed
as `tests/fixtures/mss_64x64.jpg`), stock `jpeg_read_coefficients` returns
the virtual-array set while ours returns NULL (`coef_mss_generous`: stock
`ok`, ours `coefnull`). The classic *pixel* decode of the same stream works
(the P4-14 oracle's `mss_generous` row passes), so this is a parity gap
confined to coefficient reading — likely the native `read_coefficients`
path rejecting or mishandling first-scan `comps_in_scan < num_components`.
Unrelated to the memory budget; `jpegtran`-class consumers transcoding such
streams hit it.

**Acceptance criteria.** `jpeg_read_coefficients` on a multi-scan sequential
stream returns coefficient arrays whose dump matches stock's byte-for-byte
(the P4-34 dump-comparison harness shape), pinned by a test using the
committed fixture; a red-check shows the test fails on today's NULL.

## P4-160. Default `error_exit` Presents as `abort()` + Raw `msg_code`, Not Stock's Rendered Message + `exit` — **OPEN**

**Motivation.** Found by the P4-14 adversarial review (2026-08-13), running
real stock `djpeg` against our dylib via `DYLD_LIBRARY_PATH`. On any fatal
error, our default `error_exit` prints
`libjpeg-turbo-rs: fatal JPEG error (msg_code=NN, parm0=..., parm1=...)`
and `abort()`s (SIGABRT, exit 134). Stock's default `error_exit`
(`jerror.c:79-92`) calls `output_message` — rendering the human text, e.g.
"Premature end of JPEG file" — then `jpeg_destroy` + `exit(EXIT_FAILURE)`.
Measured: a truncated file gives ours `msg_code=44` exit 134 vs stock
"Premature end of JPEG file" exit 2 (djpeg's own wrapper). P4-146 closed
message-*table* rendering (`format_message` works); this is the default
*handler* not using it and killing the process with the wrong mechanism —
a CLI consumer that never installs its own `error_exit` sees a crash where
stock sees a clean error line.

**Acceptance criteria.** The default `error_exit` renders through
`format_message`/`output_message` and terminates via `exit(EXIT_FAILURE)`
after `jpeg_destroy`, matching `jerror.c`; verified end-to-end with stock
`djpeg` on a truncated file (message text and exit status compared against
stock's dylib), with the abort-on-unwind safety rationale for the current
behaviour either preserved behind it or explicitly retired.

## P4-161. CMYK Progressive With 2×2 Luma Panics in Color Conversion, Then Surfaces as Silent Zero-Size Success — **OPEN**

**Motivation.** Found by the P4-14 verification round (2026-08-13),
reproduced with `max_memory_to_use` never set, on `src/decode/color.rs`
byte-identical across `0a4c052`/`e492da9`/the P4-14 working tree — a
pre-existing native bug, not budget work. A 97×53 4-component progressive
JPEG (CMYK, 2×2 luma sampling, built with stock cjpeg) panics at
`src/decode/color.rs:327` — `index out of bounds: the len is 56 but the
index is 56`. Stock decodes all 53 rows.

Two defects stack: the out-of-bounds itself, and how it presents through
the classic shim — the panic is caught at the FFI boundary
(`unwind_guard`), `jpeg_start_decompress` returns FALSE, and
`output_width`/`output_height`/`output_components` stay 0 **without
`error_exit` ever firing**. A C caller that ignores the boolean sees a
silent zero-size decode; upstream contract says a fatal error must reach
`error_exit` exactly once (the P4-100/P4-106 rule).

**Acceptance criteria.** (1) The 97×53 CMYK 2×2-luma progressive repro
(`cmyk_repro.c` shape, rebuilt as a fixture or generated in-test) decodes
with rows matching stock `djpeg` byte-for-byte. (2) Any panic that does
cross `unwind_guard` on the decompress path raises `error_exit` with a
real `msg_code` rather than returning FALSE with zeroed outputs —
verified by a test. Repro source:
`bisect.c`/`cmyk_repro.c` in the P4-14 review scratchpad (regenerate with
stock cjpeg: 97×53, `-sample 2x2`, 4-component, `-progressive`).

## P4-162. `max_h_samp_factor` / `max_v_samp_factor` Stay 0 After `jpeg_read_header` — **OPEN**

**Motivation.** Found by the P4-14 verification round (2026-08-13). Stock
populates the public aggregates during header parse (`jdinput.c`
`initial_setup`); we populate `comp_info[].h_samp_factor` /
`v_samp_factor` correctly but leave `cinfo->max_h_samp_factor` /
`max_v_samp_factor` at 0 after `jpeg_read_header` on every stream probed
(stock: 2/2 for 4:2:0, 1/1 for grayscale). They are set on the later
`jpeg_calc_output_dimensions` path (`jpeglib.rs:4030-4043`) but not at
header time. These are public ABI fields consumers read for MCU geometry
between header and start — and they are why `coef_array_geometries`
recomputes the maxima from `comp_info` instead of reading them.

**Scope extension (2026-08-14, from the #468 review):** the same family
one call later — the 8-bit `jpeg_start_decompress` never calls
`jpeg_calc_output_dimensions` (only the 12/16-bit path does), so
`max_v_samp_factor` / `min_DCT_v_scaled_size` / `min_DCT_h_scaled_size`
are 0 after startup where stock reports 2/8/8. A stock-shaped raw-data
consumer (libtiff, ImageMagick, ffmpeg all compute `lines_per_iMCU =
max_v_samp_factor * min_DCT_v_scaled_size`) reads zero rows per
iteration and stalls; with the P4-104 finish guard the stall now
surfaces loudly as `JERR_TOO_LITTLE_DATA` instead of silently returning
TRUE (measured: #468 review probes s1/s2 — stock `max_v 2 min_DCT_v 8`,
`scanline 64`, `finish ret 1`; shim `max_v 0`, `stalled`, `err 69`).
`capi_jpeg_read_raw_data.rs:192` masks it by calling
`jpeg_calc_output_dimensions` explicitly before asserting.

**Acceptance criteria.** After `jpeg_read_header`, both aggregates equal
stock's for a probe matrix covering 4:2:0, 4:2:2, 4:4:0, 4:4:4,
grayscale, and 4-component CMYK (oracle-compared), and
`coef_array_geometries` can assert its recomputation against them. After
the 8-bit `jpeg_start_decompress`, `max_v_samp_factor` and the
`min_DCT_*_scaled_size` pair equal stock's without the caller invoking
`jpeg_calc_output_dimensions`, and the stock-shaped raw loop (probe s2)
reads all rows and finishes TRUE; the explicit `calc` call is removed
from `capi_jpeg_read_raw_data.rs` so the test exercises the consumer's
real sequence.


## P4-164. Classic Source-Manager Residuals: Dangling Post-Decode Window, Pre-Parse Fill Continuity, Stream-Error Codes — **OPEN**

**Motivation.** Filed 2026-08-14 from the P4-109 closure's adversarial
review. The 10-case oracle closed P4-109's named criteria; the review's
differential probes (C programs linked alternately against stock 3.1.4.1
and this shim's cdylib) surfaced three adjacent divergences that are
outside those criteria but sit on the same path the item declares parity
for. None is a regression of the closure — the first reproduces
identically under the previous slurp implementation, and the third
reproduces through `jpeg_mem_src` too.

**Gaps.**

1. **The published window dangles after the `Owned` source drops.**
   `jpeg_read_header` publishes `next_input_byte`/`bytes_in_buffer` into
   the drained `JpegSource::Owned` Vec; `jpeg_finish_decompress`,
   `jpeg_abort_decompress`, and `jpeg_read_header`'s tables-only and
   splice paths drop that Vec without republishing. Measured: after
   finish on a two-image file the shim advertises `bytes_in_buffer =
   29886` over freed memory where stock reads `2882` (the unconsumed
   tail), and `29886` where stock reads `0` on a single image. The
   canonical consumer idiom `if (src->bytes_in_buffer == 0) fill();`
   then reads freed memory instead of calling fill. Fix shape: on every
   source-drop site republish an honest window — empty, or the
   still-owned `stdio_remainder` (which is exactly stock's post-finish
   leftover for the stdio family).
2. **A fill served before `jpeg_read_header` is not part of the parse.**
   Stock parses starting from the currently-published window, so
   `jpeg_stdio_src` → ABI-level `fill_input_buffer` → `jpeg_read_header`
   loses nothing; the shim's drain reads from the `FILE *` offset and
   ignores served-but-unconsumed window bytes, so the stream head served
   by that fill is skipped. Fix shape: the drain prepends the unconsumed
   window when `next_input_byte` points into `stdio_fill_backing` (the
   discriminator excludes dangling `Owned` windows, gap 1).
3. **Stream-error code and warning gaps.** A no-SOI stream raises
   `JERR_BAD_LENGTH` (12) where stock raises `JERR_NO_SOI` (55), and a
   truncated stream decodes without stock's `JWRN_JPEG_EOF` (123) +
   `JWRN_HIT_MARKER` (120) warnings. Both reproduce through
   `jpeg_mem_src` as well, so this is error-translation work in the
   decode path (P4-100's theme), recorded here because the measurements
   came from this path's probes.

**Acceptance criteria.** Extend the P4-109 oracle (or a sibling) with
rows that pin: post-finish/abort window contents against stock on
single-image, two-image and trailer files; the fill-then-read-header
sequence; no-SOI and truncated-stream error/warning traces. Fix the
shim until the traces match. Gap 1's fix must also be exercised by a
consumer-idiom probe (`bytes_in_buffer == 0 ? fill : read window`) that
never touches freed memory under ASan.

## P4-165. Packed-YUV and Plane-Array Paths Size Three Planes for `TJSAMP_GRAY` — **CLOSED 2026-08-14**

**Motivation.** Filed 2026-08-14 by the P4-139 span-computation survey,
before the `ImageLayout` refactor touches these lines. `tj3YUVBufSize`
correctly sizes **one** plane for `TJSAMP_GRAY` (`bufsize.rs:189`,
`is_gray → n_planes = 1`, matching stock), but the YUV worker paths in
`crates/libjpeg-turbo-rs-capi/src/yuv.rs` hardcode three (`yuv.rs` line
numbers as of filing, before the fix below moved them):

- `subsamp_from_tj` maps `3 => Subsampling::S444 /* TJSAMP_GRAY */`
  (`yuv.rs:79`), so `packed_yuv_len` (`:63`, `for c in 0..3`) and
  `split_packed_yuv` (`:162`) compute a three-full-resolution-plane
  span for GRAY.
- `tj3CompressFromYUV8` (`:522`) and `tj3DecodeYUV8` (`:1014`) hand that
  length to `slice::from_raw_parts` over the caller's packed buffer — a
  caller who allocated `tj3YUVBufSize(w, align, h, TJSAMP_GRAY)` bytes
  (the documented contract) is read ~3× past the allocation.
- `tj3CompressFromYUVPlanes8` (`:673`) and `tj3DecodeYUVPlanes8`
  (`:1135`) loop `0..3` over the caller's `planes`/`strides` arrays;
  stock touches only `planes[0]` for GRAY, so a legal one-element array
  is read past its end.

**Root cause.** The GRAY→S444 mapping conflates "no chroma subsampling"
with "chroma present"; every consumer then trusts `Subsampling` alone to
imply the plane count.

**Acceptance criteria.** All six YUV entry points agree with
`tj3YUVBufSize`'s plane count for `TJSAMP_GRAY`, cross-validated against
stock TurboJPEG (round-trip byte comparisons and, for the OOB shape, a
buffer sized exactly to `tj3YUVBufSize` under ASan or with guard
allocations). A capi test covers `TJSAMP_GRAY` in the packed and planar
suites — today `tests/yuv.rs` has none.

**Status (2026-08-14): closed.** The plane count now travels separately
from the geometry. `plane_count_from_tj` in `yuv.rs` is upstream's
`nc = (subsamp == TJSAMP_GRAY ? 1 : 3)` (`turbojpeg.c:1038`) built on the
*same* `bufsize::is_gray` predicate `tj3YUVBufSize` sizes by, so the two
cannot drift; `packed_yuv_len` and `split_packed_yuv` take it as a
parameter, `pack_yuv_planes` packs exactly the planes it is handed, and
all six affected entry points thread it. The GRAY→`S444` mapping stays —
it is right for plane 0's dimensions, which is all a `Subsampling` can
say. The two packed entry points also stopped routing through
`compress_from_yuv` / `decode_yuv`, which infer the plane count from the
buffer *length*: a one-plane GRAY buffer and one plane of a three-plane
4:4:4 image are the same length, so that inference cannot be trusted with
a length this layer already knows.

Proved by `crates/libjpeg-turbo-rs-capi/tests/capi_yuv_gray.rs`
(10 tests), whose full trace is compared verbatim against
`examples/tj3_yuv_gray_oracle.c` linked against real TurboJPEG, over five
geometries — 64×64 and 33×17 at `align` 1 and 4, 17×33 at `align` 1.
Destinations are over-allocated and guard-filled past the one-plane
contract length and sources sentinel-filled past it, with the chroma
slots of every plane array pointing at owned buffers, so the overrun is
observable without the test ever performing one. Before the fix, six of
the eight entry points diverged: `encodeyuv8` left `guard=DIRTY` and
`encodeplanes8` `guard1`/`guard2=DIRTY`, `decodeyuv8`/`decodeplanes8`
returned `rgbgray=no` (sentinel bytes folded in as chroma), and
`fromyuv8`/`fromyuvplanes8` produced `jsubsamp=0` —
a three-component JPEG for a grayscale request.
`toyuv8`/`toyuvplanes8` were already correct, taking their count from the
JPEG's own SOF as upstream does, and are now pinned. Falsified by forcing
`plane_count_from_tj` to 3, which reproduces exactly those six failures.
Five of the tests state the contract without reference to the oracle, so
the gate keeps its teeth where no TurboJPEG development install exists.

The two packed entry points changed route for *every* subsampling, not
only for GRAY, so the trace carries three-plane controls that the GRAY
cases cannot see: `decodeyuv8_420rgb` and `decodeyuv8_420gray` compare a
4:2:0 packed decode against stock — `TJPF_GRAY` because that output
format is the one whose internal branch moved — and
`packed_and_planar_three_plane_compress_agree` pins the compress half as
an internal equivalence. It is stated that way rather than as an oracle
line because `tj3CompressFromYUV8` diverges from stock at 4:2:0 for two
reasons that predate this change and are now filed as **P4-167**: it
refuses non-MCU-aligned dimensions, and its output cannot be decompressed
to `TJPF_GRAY`. A `fromyuv8_420` oracle case was written during this work
and removed when those two divergences turned out to be pre-existing, so
it is **not** in the tree waiting to be un-commented — writing it again is
part of P4-167.

The audit also found one deliberate non-divergence, excluded from the
trace rather than filed: this port zero-fills the inter-row alignment
padding of a packed buffer while upstream leaves those bytes untouched
(its row copies are `pw` wide though its row pointers advance by the
stride). Both stay inside the caller's own allocation, and upstream
documents nothing about their contents, so the digests compare `pw` bytes
per row at the stride. A second divergence *was* filed: the encode entry
points accept a grayscale source under a non-grayscale subsampling, which
upstream refuses — pre-existing, and the reason the plane-count clamp
here is a `min` rather than an equality check (**P4-166**).


## P4-166. `tj3EncodeYUV*8` Silently Accepts a Grayscale Source Under a Non-Grayscale Subsampling — **OPEN**

**Motivation.** Filed 2026-08-14 from the P4-165 review. With
`TJPARAM_SUBSAMP` set to anything but `TJSAMP_GRAY` and `pixelFormat`
`TJPF_GRAY`, `tj3EncodeYUV8` / `tj3EncodeYUVPlanes8` return 0 having
filled only the luma plane. The caller sized `dstBuf` from
`tj3YUVBufSize(w, align, h, TJSAMP_420)` and gets its chroma third left
at whatever was there before; feeding that buffer to
`tj3CompressFromYUV8` compresses uninitialised memory as chroma.

Upstream refuses the call. `setCompDefaults(this, pixelFormat, TRUE)`
reaches its `default:` arm (`turbojpeg.c:402-409`) and, with a non-GRAY
subsampling and a non-CMYK format, calls
`jpeg_set_colorspace(cinfo, JCS_YCbCr)`. `pf2cs[TJPF_GRAY]` is
`JCS_GRAYSCALE`, so `jinit_color_converter` hits `case JCS_YCbCr:` with
an `in_color_space` that is neither ExtRGB nor YCbCr and raises
`JERR_CONVERSION_NOTIMPL` (`jccolor.c:624-628`). The entry point returns
-1 with "Unsupported color conversion request".

**Root cause.** `encode_yuv_planes` derives its plane count from the
*pixel format* (`src/api/yuv.rs:143`, grayscale ⇒ one plane) while the
destination is sized from the *subsampling*. The C-ABI layer reconciles
the two by clamping — `current_plane_count(inst).min(planes.len())` at
`yuv.rs:391` and `:505` — which turns the mismatch into a short write.
This is pre-existing: before P4-165 the same shape produced the same
one-plane output, because `pack_yuv_planes` iterated whatever it was
handed. P4-165 made the clamp explicit without changing what it does, and
its `min` is still needed in the other direction (a three-plane RGB
source under `TJSAMP_GRAY` must truncate).

This is the P4-39 / P4-150 shape — a silent substitute where upstream
raises a documented error — and the substituted value is uninitialised
caller memory, which makes it worse than the usual case.

**Acceptance criteria.** Both encode entry points refuse a grayscale
source under a non-grayscale subsampling with an error, and the
`TJSAMP_GRAY` truncation direction keeps working. Cross-validated against
stock in `tests/capi_yuv_gray.rs`'s oracle trace, which must also pin the
refusal's *position* in the argument-validation chain — P4-155 showed the
YUV entry points disagree about whether the subsampling gate or the
pixel-format check runs first, so where this refusal lands relative to
`TJPARAM_SUBSAMP`-unset and a bad `align` is part of the contract, not an
implementation detail. Audit `tj3CompressFromYUV*8` for the same shape
while there: it takes its planes from the caller rather than from a pixel
format, so the mismatch cannot arise the same way, but the plane-count
validation is worth stating rather than assuming.

## P4-167. `tj3CompressFromYUV*8` Refuses MCU-Padded Planes, and `tj3Decompress8` Cannot Emit `TJPF_GRAY` From a Colour JPEG — **OPEN**

**Motivation.** Filed 2026-08-14 by the P4-165 three-plane control cases,
which compare `tj3CompressFromYUV8` at 4:2:0 against stock. Two
independent divergences, both pre-existing (confirmed by re-running the
control against the pre-P4-165 call path, which fails identically):

1. **Non-MCU-aligned dimensions are refused.** At 33×17 and 17×33 with
   `TJSAMP_420`, `tj3CompressFromYUV8` returns -1 with
   `"corrupt data: Y plane dimensions 34x18 do not match image dimensions
   33x17"`, where stock returns 0. The YUV plane contract *is* MCU-padded
   — `tj3YUVPlaneWidth(0, 33, TJSAMP_420)` is `PAD(33, 2) = 34` — so the
   planes are the size the caller was told to allocate, and
   `api::raw_data::compress_raw` rejects them for being larger than the
   image. Upstream handles the same mismatch by copying rows into
   MCU-sized scratch and replicating the last sample
   (`turbojpeg.c:1372-1423`, the `usetmpbuf` path — detected at `:1376`,
   copied and replicated at `:1412-1423`). Only 4:4:4 and
   already-aligned geometries work today; a caller doing
   `tj3DecompressToYUV8` → `tj3CompressFromYUV8` on any odd-sized 4:2:0
   image gets a hard failure where stock round-trips.
2. **`TJPF_GRAY` output from a three-component JPEG fails.**
   `tj3Decompress8(.., TJPF_GRAY)` on a 4:2:0 JPEG errors out:
   `repack_into_pitched`
   (`crates/libjpeg-turbo-rs-capi/src/decompress.rs:189`)
   handles grayscale *source* → RGB destination but has no
   RGB source → grayscale destination arm, and `PixelFormat::Grayscale`
   has no `red_offset`, so the generic path refuses. Upstream sets
   `out_color_space = JCS_GRAYSCALE` and lets `jdcolor` return the luma
   directly. This is a common TurboJPEG call — `tjbench` and any
   luma-only consumer make it.

**Acceptance criteria.** (1) `tj3CompressFromYUV8` /
`tj3CompressFromYUVPlanes8` accept MCU-padded planes for every
subsampling at arbitrary dimensions, matching upstream's `usetmpbuf`
handling; (2) `tj3Decompress8` and the 12/16-bit siblings emit `TJPF_GRAY`
from a colour JPEG. Cross-validated by adding a `fromyuv8_420` case to
`tests/capi_yuv_gray.rs`'s oracle trace and the matching case to
`examples/tj3_yuv_gray_oracle.c`, both of which must report
`rc=0 … jsubsamp=2 roundtrip=ok` against stock at all five geometries.
Such a pair existed while P4-165 was being written and was **removed**
once these divergences proved pre-existing — it must be written afresh,
not un-commented. The internal-equivalence test
`packed_and_planar_three_plane_compress_agree` stays either way. Note
that criterion 1 is about the *root crate's* `compress_raw` contract, so
check whether `api::raw_data` should pad internally or whether the C-ABI
layer should, and whether `P4-95`'s classic raw-data path has the same
hole.

---

## P4-168. `c_croptest` Anchored C Columns at the Requested Crop x, Not the iMCU-Aligned One — **CLOSED 2026-08-17**

**Motivation.** Filed and closed 2026-08-17 from the scheduled `Full C Parity`
run [32001568347](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/32001568347),
which failed identically on both legs (aarch64 and x86_64) at
`c_croptest_full`:

```
[full_prog0_ns0_y1_h1_GRAY] C buffer too short at row 0 x 89 (idx=315, len=315)
```

**Root cause — the harness, not the codec.** `run_crop_scenario` computed the
iMCU-aligned origin itself and handed *that* to the Rust decoder
(`aligned_x`/`aligned_w`, mirroring `jpeg_crop_scanline`), but then indexed
djpeg's PPM as if it began at image column 0, re-applying `crop_x` as a column
offset. It does not: `jpeg_crop_scanline` snaps the requested x down to an iMCU
boundary and widens the region so the right edge stays put
(`references/libjpeg-turbo/src/jdapistd.c:245-255`), and djpeg emits exactly
that aligned region. Both buffers therefore already start at the same image
column, so the offset counted the alignment twice and ran off the end of the C
buffer. Verified directly: `djpeg -rgb -crop 105x1+16+1` on a 128x95 grayscale
JPEG emits `P6 105 1` — width equal to the Rust crop width, and column 0
carrying image column 16's value.

**Why it surfaced only now, and only in the nightly leg.** Two masks stacked.
`60d7337` (2026-04-14) "fixed" the same overrun by turning it into
`eprintln!("SKIP: ...")` + `return false`, so the scenarios silently vanished
from the count — exactly the pattern [P4-116](#p4-116-c-parity-tests-can-convert-failures-or-missing-comparisons-into-a-pass--closed-2026-08-08)
was filed to abolish. P4-116 converted that skip into an assert, which is what
turned the mask into a red run. The second mask is coverage: `compute_crop_spec`
derives x from `(y_iter * 16) % 128`, and every `y_iter` in the three quick
tiers (`0`, `8`, `16`) yields `crop_x == 0`, where a mis-anchored offset is
indistinguishable from a correct one. Only the full grid reaches a non-zero
crop x, so only the scheduled job could fail. Roughly 896 GRAY scenarios
(14 of 17 `y_iter` values x 16 `h_iter` x 2 prog x 2 nosmooth) had never
actually compared pixels since 2026-04-14.

**Status (2026-08-17): closed.** The C columns are read at offset 0 against an
asserted `c_w == effective_w`, matching the existing height assertion — a
geometry disagreement is now a failure rather than something the indexing
papers over. The dead "djpeg output may be wider" branches are gone.
`cargo test --features full-c-parity --test c_croptest c_croptest_full` passes
with **5440 comparisons completed out of 10880 planned**, the only exclusion
being the deliberate colour-quantization half; every one of those 5440 asserts
`max_diff == 0` against `djpeg -crop`, so the Rust crop decoder was correct
throughout and the defect was confined to the harness. `y_iter = 1` is now in
the three quick tiers, putting a non-zero crop x in the default `cargo test`
run so this class of regression no longer waits for the nightly leg.

**Two coverage holes the fix exposed, both closed here.** The review pass that
checked the diagnosis found that the harness was not exercising what its own
comments claimed:

- **The alignment path was never taken, and our own alignment was never under
  test.** `croptest.in` derives x from `(y_iter * 16) % 128` while `align` is 8
  or 16, so `aligned_x == crop_x` in all 5,440 scenarios — the snap-and-widen
  behaviour the fix reasons about never actually ran. Worse, the harness
  pre-aligned the request with a local copy of `jpeg_crop_scanline`'s formula
  before calling `set_crop_region`, so the matrix compared C against a second
  implementation of C living in the test file, and a regression in the decoder's
  own alignment would have passed. Both sides now receive the raw request and
  align it themselves, which makes the geometry assertions a real differential
  check, and `c_croptest_unaligned_x` adds four deliberately unaligned specs
  (x = 20, 12, 4, 44) across all five layouts and both `nosmooth` values:
  **40 comparisons, 40 compared, `max_diff == 0`**. `djpeg -rgb -crop 101x4+20+1`
  returns `P6 105 4` — C snapped the origin to 16 and widened to 105, and
  `Decoder::set_crop_region(20, 1, 101, 4)` independently produces the same
  105x4, which is now asserted rather than assumed.
- **Half the grid compared two different pipelines.** `set_merged_upsample`
  was pinned to `false`, but C's `use_merged_upsample` returns TRUE exactly when
  fancy upsampling is off and the layout is 2hNv YCbCr→RGB (`jdmaster.c:43-71`),
  which `-nosmooth` selects for 420/422. So the entire `nosmooth = true` half of
  the grid compared C's *merged* output against our *non-merged* path, leaving
  our merged crop path structurally invisible to the matrix. It now tracks C's
  own condition (`set_merged_upsample(nosmooth)`), and the full grid stays at
  5440/5440 with `max_diff == 0` — the merged crop path was correct, it had
  simply never been cross-validated.

---

## P4-169. Three Sibling Crop Harnesses Carry the Same C-Column Mis-Anchoring, Two of Them Unguarded — **OPEN**

**Motivation.** Filed 2026-08-17 while fixing [P4-168](#p4-168-c_croptest-anchored-c-columns-at-the-requested-crop-x-not-the-imcu-aligned-one--closed-2026-08-17).
`c_croptest` was not the only harness that reads `djpeg -crop` output as though
it began at image column 0. The same `row * c_w * 3 + crop_x * 3` expression
appears in three more places:

- `tests/cross_check_crop_scale.rs:232` — reached when `c_w > rust_img.width`.
- `tests/crop_c_compat.rs:491` — reached whenever `c_w != crop_w`, with **no
  width guard at all**.
- `tests/crop_skip.rs:345` — same expression, but guarded by
  `assert_eq!(c_w, full_width)` immediately above it, so a snapped C origin
  fails loudly instead of comparing shifted columns.

**Root-cause hypothesis.** Per `jdapistd.c:245-255` djpeg's output starts at the
iMCU-aligned x, not at the requested one, so the correct column offset is
`crop_x - aligned_x`. These three use `crop_x`. The reason they are green today
appears to be reachability rather than correctness: every crop x they use
(0, 4 in `crop_skip`; 0, 32, 40 in `cross_check_crop_scale`; 16 in
`crop_c_compat`) is a multiple of the `align` value its fixture produces, so
`c_w == crop_w` and the offending branch is never entered. That makes the bug
latent — one subsampling change or one new crop case away from live. Note
`crop_c_compat.rs` is the dangerous shape: with no width assertion it would
silently compare the wrong columns rather than fail.

**Why this is a P4-116-family item, not a cleanup.** The failure mode is a
harness converting a real Rust-vs-C divergence into a pass (or into a confusing
OOB panic), which is exactly what P4-116 exists to abolish. P4-168 shows the
cost of leaving one of these in place: the masked scenarios sat green from
2026-04-14 until a scheduled job finally reached a non-zero crop x.

**Acceptance criteria.**

1. Each of the three sites either drops the offset and asserts
   `c_w == effective_w` (the P4-168 shape), or computes `crop_x - aligned_x`
   explicitly and proves the branch is reachable with a test that enters it.
2. Reachability is settled by evidence, not by inspection: add at least one
   non-iMCU-aligned crop x per harness, or record why that harness cannot
   produce one.
3. `crop_c_compat.rs` gains a width assertion regardless of which option is
   taken, so a geometry disagreement can never be silently re-indexed.
4. A repository-wide grep for `crop_x *` / `+ crop_x` against parsed C output
   confirms no fourth site.

**Why deferred.** P4-168's PR is scoped to the failing scheduled job and its
proof; these three are latent, currently green, and each needs its own
reachability analysis and fixture work. Filing rather than folding in keeps that
verification honest.

---

## P4-170. Classic Source-Manager Parity Fails in `--release` and Passes in Debug, So CI Never Sees It — **OPEN**

**Motivation.** Filed 2026-08-17 while running the live gate's own command
(`cargo test --workspace --release`) to re-measure test counts for the
[P4-168](#p4-168-c_croptest-anchored-c-columns-at-the-requested-crop-x-not-the-imcu-aligned-one--closed-2026-08-17)
closure. Two tests in `crates/libjpeg-turbo-rs-capi/tests/capi_classic_source_mgr.rs`
fail — the P4-109 suite that trace-compares the classic source manager against
stock libjpeg:

```
classic_source_mgr_matches_stock_libjpeg      FAILED
stdio_fill_after_decode_serves_trailing_bytes_first  FAILED  (decode failed with code 69)
```

The trace diff is ours-vs-stock on three rows; stock decodes where we raise 69:

```
left  (ours):  m4 err 69   f1 err 69                    f2 err 69
right (stock): m4 ok rows 64   f1 rows 64 pos_class before_eof   f2 rows 64
```

**Root-cause hypothesis — profile-dependent, not platform-dependent.** The same
commit, machine and oracle pass in debug and fail in release:

| Command (origin/main `d76f57c`, macOS aarch64, unmodified tree) | Result |
| --- | --- |
| `cargo test -p libjpeg-turbo-rs-capi --test capi_classic_source_mgr` | **2 passed** |
| `cargo test --release -p libjpeg-turbo-rs-capi --test capi_classic_source_mgr` | **2 failed** |

Reproduced in two independent worktrees with separate `CARGO_TARGET_DIR`s, so it
is not a stale artifact. The oracle is a stock Homebrew libjpeg-turbo 3.1.4.1
development install selected by `helpers::find_libjpeg_dev`, which rejects our
own shim by design, so this is not the self-comparison hazard.

**Why CI is green.** `.github/workflows/ci.yml`'s Integration Tests step runs
`cargo test -p libjpeg-turbo-rs-capi --test capi_input_complete_contract … --test
capi_classic_source_mgr -- --nocapture` — **debug, no `--release`**. Run
31783899509 (main, 2026-08-14) shows `test classic_source_mgr_matches_stock_libjpeg
... ok` on that leg. So the failing configuration is the one nothing runs, while
`docs/LAST_MILE.md`'s live gate asserts `cargo test --workspace --release`
passes. That combination is how a release-only divergence stays invisible.

**Acceptance criteria.**

1. Identify why the release profile changes classic source-manager behaviour.
   Error 69 arriving where stock returns 64 rows points at the fill/drain
   bookkeeping P4-109 and [P4-164](#p4-164-classic-source-manager-residuals-dangling-post-decode-window-pre-parse-fill-continuity-stream-error-codes--open)
   own; determine whether this is UB the optimiser exposes (in which case it
   outranks everything in Stage 0), a debug-only assertion masking a real error
   path, or a genuine profile-conditional branch.
2. The suite passes in both profiles on macOS aarch64 and Linux x86_64.
3. CI runs at least one C-ABI parity leg in `--release`, so a profile-conditional
   divergence cannot pass again.
4. `docs/LAST_MILE.md`'s live-gate row reflects a re-measured release run.

**Why deferred.** It reproduces on unmodified `origin/main` and is unrelated to
the crop harness work that surfaced it (different crate, different subsystem).
Diagnosing a release-only divergence in the classic source manager needs its own
change with its own oracle traces.

## P4-171. 8-Bit Lossy JPEG Cannot Be Decompressed to 12-Bit Output (3.2 beta1 note 8) — **OPEN**

**GitHub:** [#561](https://github.com/developer0hye/libjpeg-turbo-rs/issues/561) — filed 2026-08-17 by the [P4-130](#p4-130-c-parity-oracle-is-pinned-to-3141-upstream-stable-is-320--partial-every-oracle-provisioning-job-is-now-pinned-checked-and-measured-the-legs-still-on-one-release-the-submodule-bump-and-the-four-filed-gaps-remain) 3.2 delta triage.

**Motivation.** 3.2 beta1 note 8 added a capability, not a fix: an 8-bit-per-sample
*lossy* JPEG can now be decompressed to a 12-bit-per-sample output image, to
give shadow recovery in underexposed images somewhere to put the extra range.
Upstream exposes it two ways — `cinfo->data_precision = 12` after
`jpeg_read_header()` in the libjpeg API, and `tj3Decompress12()` after
`tj3DecompressHeader()` in TurboJPEG.

Neither works here. `src/api/precision.rs:865-870` refuses any stream whose SOF
precision is not 12:

```rust
if frame.precision != 12 {
    return Err(JpegError::Unsupported(format!(
        "decompress_12bit requires precision=12, got {}", frame.precision)));
}
```

`tj3Decompress12` (`crates/libjpeg-turbo-rs-capi/src/precision.rs:409`) calls
straight through to it, so a consumer following the 3.2 documentation gets
`-1` and an error string where upstream returns a 12-bit image. The classic
side is unmeasured: `data_precision` is a public ABI field a caller can simply
assign, and what our `jpeg_start_decompress` does with a value the header did
not put there is not covered by [P4-154](#p4-154-classic-jpeg_write_scanlines--jpeg_start_compress-ignore-data_precision-entirely--closed-2026-08-13)'s compression-side matrix.

The failure direction is the safe one — we refuse where upstream accepts, not
the reverse — so this is a feature gap rather than a defect.

**Acceptance criteria.**

1. A differential test decompresses the same 8-bit lossy JPEG through
   `tj3Decompress12()` on the shim and on stock 3.2.0, and the 12-bit outputs
   match. Byte-exact if upstream's upscale is exactly `<< 4`-shaped; the test
   states the measured relationship rather than choosing a tolerance first.
2. The classic route (`cinfo->data_precision = 12` after `jpeg_read_header()`)
   is covered by the same comparison, including the error upstream raises when
   the assignment is made at a state that does not allow it.
3. If the capability is declined rather than implemented, the refusal matches
   upstream's error code and message rather than our current
   `JpegError::Unsupported` prose — a consumer must be able to tell "not
   supported" from "malformed".
4. `docs/C_API_REFERENCE.md`'s `tj3Decompress12` row states which source
   precisions it accepts.

**Why deferred.** It is new upstream capability rather than a divergence in
existing behaviour, and it needs the 12-bit output pipeline to accept an 8-bit
front end — a decode-path change, not a shim wiring change.

## P4-172. TurboJPEG 3.2 ICC and `TJCS_DEFAULT` Additions Are Unimplemented (3.2 beta1 note 10) — **OPEN**

**GitHub:** [#562](https://github.com/developer0hye/libjpeg-turbo-rs/issues/562) — filed 2026-08-17 by the P4-130 3.2 delta triage.

**Motivation.** 3.2 beta1 note 10 lists four TurboJPEG additions. Two are
already here — 4:1:0 and 2:4 subsampling exist as `TJSAMP_410` / `TJSAMP_24`
and run in the subsampling matrices — and two are not:

- **`tj3GetICCProfile()` repeatability and compression instances.** Upstream 3.2
  allows the call to be repeated, and allows it against a *compression*
  instance (including a profile `tj3LoadImage*()` extracted from a PNG).
  Ours (`crates/libjpeg-turbo-rs-capi/src/tj3.rs:477`) serves whatever
  `inst.inner.icc_profile()` holds, which is populated by `decompress()`; the
  repeated-call and compression-instance behaviours are untested in either
  direction, so this entry is "unverified", not "known broken".
- **`TJCS_DEFAULT`.** A new `TJPARAM_COLORSPACE` value that resets the JPEG
  colorspace to the default. No occurrence of the name exists in this
  repository, so a caller passing it hits our unknown-value path.

`TJPARAM_SAVEMARKERS` itself is **not** part of this gap, and an earlier draft
of this entry wrongly said it was: `src/api/tj3.rs:303` accepts levels 0-4 and
`:841` wires 2 to all markers and 4 to ICC-only extraction from the JPEG. What
note 9 adds on top is the *PNG* side of that transfer, which is tracked with
the PNG work in P4-174 — a marker level with no PNG to transfer to or from has
no observable behaviour to compare.

**Acceptance criteria.**

1. `TJCS_DEFAULT` is accepted by `tj3Set(TJPARAM_COLORSPACE)` with upstream's
   semantics, cross-checked against stock 3.2.0 for at least the "set, then
   reset to default" sequence on a colour and a grayscale source.
2. `tj3GetICCProfile()` called twice returns the same profile both times, and
   called on a compression instance returns the profile associated with it —
   each compared against stock 3.2.0 rather than against our own previous
   output.
3. Whatever is declined is declined the way upstream declines an unsupported
   parameter value, not by silently accepting it. The P4-39 / P4-150 silent-
   substitute shape is the failure mode to avoid.

**Why deferred.** Small but genuinely new API surface, and the ICC half is
entangled with P4-174's PNG interchange work — the compression-instance profile
upstream describes is the one `tj3LoadImage*()` extracts from a PNG.

## P4-173. jpegtran 3.2 Crop Expansion, `-roll`, and the Flatten/Reflect Refusal Are Unported — **OPEN**

**GitHub:** [#563](https://github.com/developer0hye/libjpeg-turbo-rs/issues/563) — filed 2026-08-17 by the P4-130 3.2 delta triage.

**Motivation.** Three 3.2 changes move `transupp.c`, which is the code our
transform API mirrors:

- **beta1 note 4** — jpegtran now honours `-trim` and `-perfect` when `-crop`
  *expands* the image. With `-trim`, partial iMCUs from the source are
  discarded in the expanded image (the previous behaviour); without it they are
  left in place, which is the new default; with `-perfect`, expansion fails
  outright if the source has any partial iMCU. The stated purpose is reversible
  composition with `-drop`.
- **3.2.0 note 4** — a buffer overrun and segfault, plus an infinite loop, when
  `-crop` + `-trim` expand the width of an image narrower than one iMCU under
  the "flatten" and "reflect" extensions. 3.2 now raises an error for that
  geometry instead.
- **beta1 note 12** — a new `-roll` transform (lossless shift with wraparound).

None of these behaviours exist here. "It is application code, not library code"
does not exempt them: this repository implements the transform semantics
directly, and `tests/c_tjtrantest.rs` / `tests/cross_check_transform_*.rs`
compare against jpegtran's output, so upstream's new default *is* the oracle
those suites will meet the moment the 3.2.0 tools become their oracle.

**Acceptance criteria.**

1. Expansion semantics: for a source with partial iMCUs, our transform matches
   3.2.0 `jpegtran` byte-exactly for `-crop` expansion with `-trim`, without
   `-trim`, and with `-perfect` (including the refusal).
2. The narrower-than-one-iMCU flatten/reflect geometry raises an error rather
   than looping or overrunning, matching 3.2.0.
3. `-roll` is implemented with a byte-exact comparison against 3.2.0
   `jpegtran -roll`, or recorded as a scoped non-goal with the reason.
4. The transform cross-checks state which oracle version defines their
   expectations, since the default changed between 3.1 and 3.2.

**Why deferred.** It needs the 3.2.0 tools as the transform oracle, which is
exactly what P4-130's second leg introduces; sequencing it after that leg is
what keeps the diff readable.

## P4-174. PNG Interchange Parity for `tj3LoadImage*`/`tj3SaveImage*` Is Narrower Than 3.2 — **OPEN**

**GitHub:** [#564](https://github.com/developer0hye/libjpeg-turbo-rs/issues/564) — filed 2026-08-17 by the P4-130 3.2 delta triage.

**Motivation.** 3.2 beta1 note 9 makes PNG a first-class interchange format for
cjpeg, djpeg, `tj3LoadImage*()` and `tj3SaveImage*()`: 8- and 16-bit-per-channel
images, ICC profile transfer in both directions when `TJPARAM_SAVEMARKERS` is 2
or 4, a `-noicc` opt-out, and a documented *reversible* upscale of 2-7 and 9-15
bit precisions to 8- and 16-bit PNG so a non-standard-precision lossless JPEG
round-trips losslessly. 3.2.0 note 2 then hardened that writer against
out-of-range sample values for precisions other than 8 and 16.

We have PNG, but narrower: `crates/libjpeg-turbo-rs-capi/src/imageio.rs:212`
detects the signature and reports "PNG support not enabled in this build"
unless the `png` cargo feature is on, and neither the ICC transfer, the 16-bit
path, nor the reversible non-standard-precision upscale is implemented. The
feature gate itself is a parity question: upstream's `tj3LoadImage*()` either
supports PNG or does not, and a consumer cannot see a cargo feature.

**Acceptance criteria.**

1. `tj3LoadImage*()` / `tj3SaveImage*()` round-trip 8- and 16-bit PNG against
   stock 3.2.0, pixel-exact.
2. ICC transfer under `TJPARAM_SAVEMARKERS` 2 and 4 matches upstream in both
   directions, including the profile being absent when the parameter is 0.
3. A lossless JPEG with a non-standard precision (2-7, 9-15) round-trips
   JPEG → PNG → JPEG losslessly, as upstream's reversible upscale promises.
4. Out-of-range samples at a precision other than 8 or 16 are refused rather
   than overrunning the rescale array (3.2.0 note 2's hardening).
5. A recorded decision on the `png` cargo feature: default-on, or documented as
   a build-configuration divergence a consumer of the C ABI can discover.

**Why deferred.** It is the largest of the four 3.2 gaps and the least
load-bearing for the replacement gate — PNG interchange is a convenience
surface around the codec, not the codec.

## P4-175. `capi_classic_decode_budget` Is Never Named by a Workflow, So It Has Never Run in CI — **OPEN**

**GitHub:** [#565](https://github.com/developer0hye/libjpeg-turbo-rs/issues/565) — found 2026-08-17 while auditing which oracle each gate uses for P4-130.

**Motivation.** Integration suites in the C-ABI crate run only when a workflow
names them: `cargo test --lib` selects the default workspace member, and the
Integration Tests job's `cargo test --tests` is the root crate. `grep -rn
"capi_classic_decode_budget" .github/workflows/` returns nothing, so P4-14's
decode-sequence budget enforcement — cited in `docs/LAST_MILE.md`'s live-gate
row as "the 3-test `capi_classic_decode_budget` suite" — compiles on every pull
request and executes on none.

**A second unnamed suite (2026-08-18).** Pairing the two tool legs for P4-130
required enumerating which capi suites compare against C, and that enumeration
found `capi_yuv_gray` in the same state: `grep -rn "capi_yuv_gray"
.github/workflows/` returns nothing, so P4-165's 10-test GRAY-YUV suite — the
one that closed a heap **write** out of bounds in `tj3EncodeYUV8`, and whose
oracle test compares against real TurboJPEG — has also never run in CI. Two
suites, found by two unrelated pieces of work, is the argument for criterion 3:
the fix is the enumeration, not the two names. Neither is fixed here; P4-130's
pairing gate compares the two *legs* and cannot see a suite that is on neither.

**A third, later the same day.** P4-131's release-bundle work found
`install_layout` in the same state — `git log --all -G"install_layout" --
.github/` returns only the commit that added the step. It is P2-8's own closing
gate, so the install-layout assertion that closed that item had never run on a
pull request in the repository's history. That one *is* fixed, in the commit
that found it, because P4-131's own criterion depends on the staging path it
guards. It does not weaken criterion 3: three suites found by three unrelated
pieces of work is the same argument, one instance stronger.

This is the third instance of one defect class. P4-61 recorded it first (a test
filter that matched nothing), `capi_classic_error_codes` was caught the same way
("it compiled on every PR and executed on none, which is precisely how it
reported '18 codes verified' through the entire period when all 18 rendered as
'bogus message code'"), and the comments around the named steps in `ci.yml`
warn about it in prose. Prose has now failed to prevent it twice.

**Acceptance criteria.**

1. `capi_classic_decode_budget` and `capi_yuv_gray` are named in a CI step,
   each with the `LIBJPEG_TURBO_PREFIX` its oracle needs so it fails rather
   than soft-skips.
2. Each passes there, or the failure it surfaces is filed.
3. A mechanism, not another comment: an enumeration test comparing
   `crates/libjpeg-turbo-rs-capi/tests/*.rs` against the suites named in
   `.github/workflows/*.yml`, with an explicit opt-out list for suites that are
   deliberately local-only. Criterion 3 is the valuable half.

**Why deferred.** Unrelated to the oracle-currency work that found it, and
criterion 3 is a gate of its own — it needs the opt-out list triaged across
every capi suite, not just this one.

## P4-176. Every C Oracle Is Fetched by a Moveable Name, With Nothing Verifying What Arrived — **OPEN**

**GitHub:** [#568](https://github.com/developer0hye/libjpeg-turbo-rs/issues/568) — found 2026-08-18 by the codex review on the P4-130 C-ABI oracle-leg change.

**Motivation.** Every C libjpeg-turbo oracle here is fetched by a name upstream
can repoint, and nothing checks what arrived:

- **eleven** deb downloads —
  `curl -fL .../releases/download/${VERSION}/libjpeg-turbo-official_${VERSION}_${ARCH}.deb`,
  installed with no digest: `ci.yml:64,330,639`,
  `cross-arch.yml:48,88,131,169,214,256`, `fuzz-smoke.yml:95`,
  `full-c-parity.yml:99`. (`fuzz-smoke.yml:206` prints the same command as
  reproduction instructions and does not fetch.) Three of the `cross-arch.yml`
  sites arrived on 2026-08-18 with P4-130's `-current-oracle` twins, which is
  the point: every leg this repository pairs adds a fetch, so the inventory
  grows with the coverage rather than with this gap.
- **six** source clones — `full-c-parity.yml:58,152` and `ci.yml:925,972` at
  `--branch 3.1.4.1`, and `full-c-parity.yml:191` and `ci.yml:725` at
  `--branch 3.2.0`, the last of them for the `trace-current` v8-ABI oracle.
  `ci.yml:925` is `test-cross-encode`, which became a source clone on
  2026-08-18 when P4-130 replaced its `brew install jpeg-turbo`;
- `references/libjpeg-turbo` is the exception. A submodule is pinned by commit,
  which is why it is not part of this gap.

The `trace-current` step greps `set(VERSION 3.2.0)` from the cloned tree, so a
tag repointed at a *different release* fails there; since 2026-08-17 the
`tool-current` leg, and since 2026-08-18 every job that provisions an oracle,
checks its installed tools' `-version` output the same way. Every one of those
checks answers the same question — *is this the release it claims to be* — and
none answers the integrity one. A tag repointed at a modified tree of the same
version, or a replaced release asset, is indistinguishable from the real thing — and these oracles are
what every differential gate in this repository compares against, so a
substituted oracle does not fail: it silently redefines "correct".

The risk is small (release assets and tags on a widely mirrored project), which
is why it is filed rather than fixed inline. It is recorded because the
alternative is that the next reader takes the version pins to imply integrity.
They pin *which* release is requested, not *what* is delivered.

**Acceptance criteria.**

1. Each provisioning site verifies what it fetched: a recorded `sha256` per deb
   (per architecture) and a recorded commit SHA per source clone, checked
   against `HEAD` after checkout.
2. The digests live beside the versions they belong to.
   `docs/oracle_versions.tsv` already answers "which release" and is the
   natural place to answer "which bytes".
3. `tests/oracle_version_pins.rs` extends over the new columns in the
   both-directions style it already uses: a provisioning site whose digest is
   undeclared fails, and a declared digest nothing uses fails.
4. Or an explicitly recorded decision that upstream's assets are trusted
   unverified, with the reasoning. A recorded non-goal closes this as
   legitimately as an implementation does — what it may not do is stay
   unanswered.

**Why deferred.** Unrelated to the oracle-currency work that surfaced it, and
it touches all five workflows rather than the one step under review. Bundling
it into that change would have mixed a supply-chain change into a coverage
change.

## P4-177. The Workflow Scanner Does Not Model Heredocs, Folded Scalars or Quoted Substitution Syntax — **PARTIAL: folded scalars are modelled; heredocs and quote/escape state remain**

**GitHub:** [#572](https://github.com/developer0hye/libjpeg-turbo-rs/issues/572) — filed 2026-08-18 from the sixth codex round on the
[P4-130](#p4-130-c-parity-oracle-is-pinned-to-3141-upstream-stable-is-320--partial-every-oracle-provisioning-job-is-now-pinned-checked-and-measured-the-legs-still-on-one-release-the-submodule-bump-and-the-four-filed-gaps-remain)
per-job pin-and-name gates.

**Motivation.** `tests/oracle_version_pins.rs` decides which workflow steps
reach a C oracle by reading their shell, and five review rounds moved that
reader from a substring match to a per-line command-position walk. Each round
found a shape the previous one could not see or saw where there was nothing,
and the sixth found three more. Unlike the first five, none of these exists in
`.github/workflows` today — which is why they are filed rather than fixed
inside a change whose subject is the oracle legs:

1. **Escaped quotes in an inline assignment.** `leaves_a_quote_open` counts
   `"` characters, so `FOO="\"" cargo test` reads as an unterminated value and
   every following token is skipped — a *missed* invocation.
2. **Folded scalars and heredocs.** The step reader inserts a newline between a
   `run:` block's physical lines, which is right for a literal `|` block and
   wrong for a folded `>` one, where YAML joins with spaces; and a heredoc body
   is data, not commands. Both directions are reachable: a folded `echo` /
   `cargo test` pair would read as two commands, and a heredoc containing
   `cargo test` would read as an invocation — *false failures* in an
   oracle-provisioning job with no prefix.
3. **Quoted substitution syntax.** `$(` and a backtick are treated as an active
   command substitution wherever they appear, so `echo '$(cargo test)'` — which
   prints literal text — reads as a test run.

The residual is bounded by what the gate is for: it decides which steps must
name an oracle prefix, and it fails closed, so the live risk is a rejected
valid workflow rather than an unchecked oracle. What it costs is *precision*,
and precision is what keeps a gate from being edited away the first time it
blocks a legitimate change.

**Acceptance criteria.**

1. The scalar style is retained through the step reader, so a folded `>` block
   joins with spaces and a literal `|` block with newlines, each pinned by a
   test using the real shape from `ci.yml`.
2. Heredoc bodies are recognised and excluded from command scanning.
3. Quote and escape state is tracked well enough that a single-quoted or
   escaped `$(` is not an active substitution, and an escaped `"` inside an
   assignment value does not open one.
4. Each of the three shapes above is pinned in both directions — the shape that
   must be seen, and the neighbouring shape that must not be.
5. A `cargo test` in a **conditional** position keeps that context. `if cargo
   test; then …` and a `while` condition are exempt from `errexit`, so a
   failing test leaves the step green — and the scanner normalises the command
   to the same `TestRun` as a plain one, so a pair where only one leg wraps its
   run compares equal. Either the context is part of what is compared, or the
   shape is rejected. Added 2026-08-18 from the sixth round on the cross-arch
   pairing gate; not in `.github/workflows` today (`test-corpus`'s `if ! cargo
   run …` is a run, not a test).
6. A baseline selection merged from two runs under one feature set is covered
   when the twin splits those filters across *several* builds that each cover
   it (`--features F,png` and `--all-features`, say). The comparison asks one
   twin build to cover the whole merged selection, which rejects a pair whose
   coverage is genuinely equal — a false failure, the polarity that costs a
   valid change rather than hiding a gap. Added 2026-08-18 from the same round.

**Why deferred.** None of these shapes is in the workflows, and the same file's
history is the argument for not fixing them inline: each of the five rounds
that preceded this one bought its next finding, so a further pass at the
scanner belongs in a change whose subject *is* the scanner, with its own review.

**Status (2026-08-18): partial — criterion 1 delivered.** The cross-arch
pairing work did not set out to touch this item, but it could not avoid half of
criterion 2: comparing two legs' `cargo test` commands means reading a folded
`>` block as the one command it is, or every argument past its first line drops
out of the comparison — which is where a twin's selection would differ if it
did. `steps_in` now keeps the scalar style and joins `>` with spaces and `|`
with newlines, pinned by
`a_folded_block_is_one_command_and_a_literal_block_is_many` using `ci.yml`'s
real shape in both directions (a folded `--test` list stays one command; a
literal `set -o pipefail` + `cargo test` stays two). Criterion 2's heredoc
half, criterion 3 (quote and escape state, including the escaped `"` inside an
inline assignment) and criterion 4's remaining both-direction pins are
untouched.

## P4-179. An `apt`-Provisioning Step Can Stall Indefinitely Because Nothing Bounds Its Wall Clock — **OPEN**

**GitHub:** [#576](https://github.com/developer0hye/libjpeg-turbo-rs/issues/576) — found 2026-08-18 while landing the cross-arch oracle pair (#575).

**Motivation.** Every oracle-provisioning step here fetches the deb and then
runs `sudo apt-get update -qq` before installing it. Nothing bounds that
command, and nothing bounds the step: `timeout-minutes` sits on the `cargo
test` step, never on the install, so an `apt-get update` that stalls holds the
runner until the job default of **360 minutes**. There are **twelve**
`apt-get update` sites across `ci.yml`, `cross-arch.yml`, `armv7.yml` and
`release.yml`, and none of them is bounded.

It stalls. Measured on #575, four times across three attempts at the same head,
all on `ubuntu-24.04` in `cross-arch.yml`, on baseline and current legs alike:
`Test (linux-x86_64 AVX2)` for 24m37s (run 32084516493), then
`Build (linux-x86_64, AVX2 disabled at compile time)` and
`Test (linux-x86_64 AVX2, oracle 3.2.0)` for ~31m each (run 32086139577), then
that second one again for 22m on its rerun. Every one ended by cancellation,
never by failing.

The log places it precisely. In job 95554147447 the download finished in
**0.15 s** — `100 608k 100 608k 0 0 4818k` at `00:27:59.4` — and the step
printed nothing further until `##[error]The operation was canceled` at
`00:52:36`. `curl` is not the stall; the package-index fetch is. A healthy run
of the identical step takes **17 seconds** (job 95568671681,
`01:48:00Z → 01:48:17Z`), so the distribution is 17 s or hours with nothing in
between — which is what makes it diagnosable and what makes a bound safe.

The cost is 20–30 runner-minutes per stall and a pull request blocked until a
human cancels. The reason it is filed rather than tolerated is the second-order
one: **a step that never returns does not fail.** The legs P4-130 added exist
to make a 3.2.0 divergence visible; a leg that hangs produces neither a
divergence nor a red check, and every reader — the merge button, `gh pr checks`,
the next agent — sees "still running", which is the one state that asserts
nothing. `apt` is configured here with no `Acquire::Retries` and no
`Acquire::http::Timeout`, so a stalled mirror connection has nothing to time it
out.

**It is not the oracle steps — it is `apt`.** The PR filing this entry (#577,
documentation only) stalled twice more while it waited for CI, and one of them
installs no oracle at all: `Test (linux-armv7 scalar, emulated)` sat 29 minutes
in `Install armhf cross toolchain + qemu-user` (`armv7.yml:66`), whose first
line is the same unbounded `sudo apt-get update -qq`. `Test (linux-x86_64
AVX2)` stalled beside it for 25 minutes on the oracle install. Six stalls in
one evening, across two workflows and two step kinds, share exactly one
command — which is why the criteria below are written against every `apt`
provisioning step rather than the oracle ones that surfaced it.

Distinct from [P4-176](#p4-176-every-c-oracle-is-fetched-by-a-moveable-name-with-nothing-verifying-what-arrived--open),
which asks whether the bytes that arrive are the ones we asked for. This one
asks whether the fetch terminates.

**Acceptance criteria.**

1. Every provisioning step that runs `apt` bounds its own wall clock —
   `timeout-minutes` sized from the measured healthy duration rather than
   guessed — so a stalled index fetch turns the job red instead of holding a
   runner. All twelve sites, not only the oracle ones: the armv7 toolchain step
   stalled the same way and installs no oracle.
2. The index fetch is bounded and retried at the `apt` level
   (`-o Acquire::Retries=…`, `-o Acquire::http::Timeout=…`), or `apt-get
   update` is dropped where installing a local `.deb` does not need a fresh
   index — with the dependency-resolution argument recorded either way, since
   the deb's declared dependencies decide whether the image's existing index
   suffices.
3. `tests/oracle_version_pins.rs` extends over the new property in the
   both-directions style its `pinned` / `checked` / `measured` rules already
   use: a provisioning step with no bound fails, and the pin is asserted
   against a step that has one. That file already enumerates every job in every
   workflow, so the rule has somewhere to live — but its subject is oracles,
   and a bound on `armv7.yml`'s toolchain step is not an oracle property, so
   where the gate goes is part of the work rather than settled here.
4. The measurement is recorded — normal duration per step and runner class — so
   each bound is evidence rather than a round number.

**Why deferred.** Found while landing #575, whose subject is which *release*
each leg measures. The stall is in a step shape that predates that change and
appears in four workflows, so fixing it there would have mixed a
CI-robustness change into a coverage one — and the fix wants its own
measurement of what a healthy run costs per step and runner class before it
picks a bound.

## P4-181. Differential HFlip Fuzzing Emits a JPEG That Stock djpeg Rejects — **CLOSED 2026-09-07**

**GitHub:** [#581](https://github.com/developer0hye/libjpeg-turbo-rs/issues/581) — filed 2026-09-06 from scheduled Fuzz Smoke [run 34042331788](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/34042331788) (`main` at `118a9f9`; recorded oracle Linux x86_64, libjpeg-turbo 3.1.4.1, Rust nightly 2026-09-05).

**Evidence.** Artifact `crash-3732c8eae6ee71b7e90ba334fcb94b2e6d8de878` (876 bytes: one op-selector byte, then an 875-byte JPEG) trips the acceptance-agreement panic in `fuzz_transform_diff_c.rs`: `transform-diff HFlip: djpeg rejected our transformed JPEG (input=16x16, rust_len=925, c_len=1053)`. Re-running `djpeg` on the 925-byte output gives the actual verdict, which the panic did not carry: `Unknown Adobe color transform code 255`, exit 2 — djpeg *decoded* the file, but with a warning, and the harness (correctly) counts any non-zero exit as a rejection because jpegtran's 1053-byte output decodes with exit 0.

**Root cause.** The source is a 16x16 4:2:0 progressive JPEG whose only scan is DC-first, followed by a stray DHT, six APP0 segments (none carrying a `JFIF\0` identifier), two Exif APP1 segments and nine APP14 segments — five of them identified as `Adobe`, the last of those carrying transform byte 255. Two things about it matter to libjpeg: the leading APP0's identifier is `JFIF\x02`, not `JFIF\0`, so `examine_app0` (`jdmarker.c:606`) never sets `saw_JFIF_marker`; and every Adobe marker sits *after* the first SOS, so at `jpeg_read_header` time `default_decompress_parms` (`jdapimin.c:137`) sees no marker at all and classifies the stream as YCbCr from component IDs 1/2/3. `jpegtran` then re-encodes through `jpeg_copy_critical_parameters` (`jctrans.c:71`) → `jpeg_set_colorspace(JCS_YCbCr)` (`jcparam.c:333`) → `write_file_header` (`jcmarker.c:475`): a JFIF APP0, no Adobe marker, and — under `-copy all` — every saved APP segment copied verbatim, including the non-JFIF APP0s and all nine APP14 segments.

`write_coefficient_colorspace_marker` in `src/api/coefficient.rs` did something else: it re-emitted `JpegCoefficients::adobe_transform` *verbatim* as a synthesized Adobe APP14 whenever the source had one, and wrote JFIF only when the source had a `JFIF\0` marker or had neither an Adobe marker nor `R`/`G`/`B` component IDs. So the transcode carried `Adobe … transform=255` and no JFIF, and every libjpeg consumer that opens it warns. The verbatim rule (introduced by `abcfb1a`/`0b83648` to keep RGB-vs-YCbCr classification stable across a transcode) had three further consequences the fuzz target never reached: a JFIF+Adobe source transcoded under `-copy all` came out with **two** Adobe segments (one synthesized, one copied); `MarkerCopyMode::All` dropped **every** APP0 from the copied set where `jcopy_markers_execute` (`transupp.c:2487`) drops only a `JFIF\0` duplicate of the header the encoder wrote; and the capi's `jpeg_write_coefficients` prepended a second Adobe segment to 4-component outputs on top of the one the core writer had started emitting.

**Fix.** The marker reader now snapshots the JFIF/Adobe state at the first SOS (`JpegMetadata::saw_jfif_marker_at_first_sos` / `adobe_transform_at_first_sos`), because libjpeg classifies exactly once — `jpeg_consume_input` calls `default_decompress_parms` at `JPEG_REACHED_SOS` — while `examine_app0`/`examine_app14` keep updating `saw_*` between scans without ever re-classifying; both the decoder's `detect_color_space` and `read_coefficients` read the snapshot (a whole-stream reading would have turned this seed's post-SOS Adobe marker into an RGB header had its byte been 0, and decoded it without the YCbCr conversion djpeg applies — caught in review). `classify_coefficient_colorspace` ports `default_decompress_parms` (JFIF outranks Adobe; Adobe 0 = RGB/CMYK, 2 = YCCK, anything else = YCbCr/YCCK with libjpeg's "assume" fallback; `R`/`G`/`B` IDs = RGB), and `coefficient_header_markers` ports `jpeg_set_colorspace` + `write_file_header` (JFIF for grayscale/YCbCr, Adobe 0 for RGB/CMYK, Adobe 2 for YCCK, transform byte derived from the output colorspace per `jcmarker.c:423-429`). `transform_jpeg_with_options` keeps every saved APP/COM segment and applies `jcopy_markers_execute`'s two duplicate rules against the header the *final* coefficient set produces (a `grayscale` request changes it); `inject_saved_markers` places copied markers after the writer's own JFIF/Adobe header, where `jpegtran` puts them. The capi shim drops its `swap_jfif_for_adobe_app14` / `inject_adobe_app14_after_jfif` post-processing and instead feeds the core writer `write_JFIF_header` / `write_Adobe_marker` / `jpeg_color_space` for foreign coefficient arrays. The fuzz target's panic now carries djpeg's exit status and stderr.

**Status (2026-09-07): closed.** `tests/regression_transform_fuzz_progressive.rs::progressive_source_with_bogus_adobe_transform_after_sos_matches_jpegtran` inlines the 875-byte source and holds HFlip, VFlip and Rot180 **byte-exact** against `jpegtran -copy all` (all three were 925 → 1053 bytes, identical to C), then decodes both through `djpeg` with exit 0 asserted; its sibling `adobe_transform_after_first_sos_does_not_change_classification` flips the seed's last Adobe transform byte to 0 and holds the HFlip transcode byte-exact with `jpegtran` and the block-smoothed decode pixel-exact with `djpeg`. `tests/transform.rs::coefficient_transform_header_markers_match_jpegtran_for_adobe_sources` crafts JFIF+Adobe(1), JFIF+Adobe(255), Adobe(0), Adobe(1) and Adobe(255) sources and holds `-copy all` and `-copy none` byte-exact (the RGB-classified Adobe(0) case to header segments + pixels, see P4-182). `crates/libjpeg-turbo-rs-capi/tests/capi_jpeglib_write_coefficients.rs::write_coefficients_preserves_source_adobe_app14` now asserts exactly one Adobe segment. The seed is committed under `fuzz/corpus/fuzz_transform_diff_c/` and replayed by `tests/fuzz_crashes.rs::fuzz_transform_diff_c_crashes_are_panic_safe`. Verified red-before-green: with every `src/` change stashed, the seed test fails at its byte-exact assertion against the old 925-byte output (the stream djpeg exits 2 on), and the post-SOS test fails at its byte-exact assertion on the Adobe-0 header.

## P4-182. Transcode Huffman-Slot Assignment Keys on Component IDs Instead of the Colorspace Classification — **OPEN**

**GitHub:** [#584](https://github.com/developer0hye/libjpeg-turbo-rs/issues/584) — found 2026-09-07 while holding P4-181's crafted sources byte-exact against `jpegtran`.

**Motivation.** `jpeg_copy_critical_parameters` explicitly does **not** copy the source's Huffman table assignments (`jctrans.c:143-144`: *"instead we rely on jpeg_set_colorspace to have made a suitable choice"*). `jpeg_set_colorspace` (`jcparam.c:333`) puts every component of an RGB, CMYK, grayscale or unknown-colorspace stream on DC/AC slot 0; YCbCr uses 0/1/1; YCCK uses 0/1/1/0. `jpeg_simple_progression` likewise reserves the 10-scan luma/chroma script for 3-component YCbCr and uses the all-purpose `2 + 4·n` script for everything else. Our `coding_table_for_component` / `uses_single_rgb_coding_table` put component 0 on slot 0 and every other component on slot 1, except when the IDs are literally `R`/`G`/`B` *and* every component uses quant table 0 — a partial port keyed on the wrong input, now that `classify_coefficient_colorspace` (P4-181) exists.

**Measured** against `jpegtran -copy all -flip horizontal` (libjpeg-turbo 3.1.4.1), pixels identical in every case:

| source | C slots | ours | bytes C / ours |
| --- | --- | --- | --- |
| `tests/fixtures/cmyk_scanner/scanner_64x64.jpg` (CMYK, Adobe 0) | 0/0/0/0, two DHTs | 0/1/1/1, four DHTs | 970 / 1132 |
| `tests/fixtures/real_world/pil_cmyk.jpg` (YCCK, Adobe 2) | 0/1/1/0 | 0/1/1/1 | 31158 / 31167 |
| 32x24 4:2:0 source, Adobe transform 0, IDs 1/2/3 (RGB by classification) | 0/0/0 | 0/1/1 | differ |

**Acceptance criteria.**

1. Slot assignment (and the progressive scan-script choice) derive from `classify_coefficient_colorspace` per the table above, across every writer variant (`write_coefficients`, `_optimized`, `_progressive`, `_arithmetic`, `_progressive_arithmetic`) — the same two helpers feed all of them, so the change is one site.
2. The `adobe0` case in `tests/transform.rs::coefficient_transform_header_markers_match_jpegtran_for_adobe_sources` is promoted to byte-exact, and CMYK/YCCK `-copy all` byte-exact cases against `jpegtran` are added for both baseline and `-progressive`.
3. `examples/stock_djpeg_cjpeg/run.sh` and the corpus transform gate stay green.
4. **JFIF minor version.** `write_app0_jfif_with_density` always writes 1.01; `jpeg_copy_critical_parameters` (`jctrans.c:155-158`) copies the source's minor version when the major is 1 and `emit_jfif_app0` writes it, so a JFIF 1.02 source transcodes byte-inexactly. `MarkerReader` already parses both bytes; `JpegCoefficients` needs to carry the minor version (an additive field — note the struct is constructed literally by tests) and the writers must emit it. Add a 1.02 source to the byte-exact matrix.
5. **capi header override.** `materialize_foreign_coef_arrays` folds the destination cinfo's `write_JFIF_header` / `write_Adobe_marker` / `jpeg_color_space` into the classifier's two inputs, which round-trips only the four states `jpeg_set_colorspace` produces. An application that clears both flags (`JCS_UNKNOWN`, or `write_JFIF_header = FALSE` by hand) gets a JFIF or Adobe 0 header where `write_file_header` writes none, and `write_Adobe_marker` on a YCbCr destination gets JFIF alone where C writes JFIF + Adobe 1. Give the core writers an explicit header override (an internal `write_coefficients_with_header(coeffs, CoefficientHeaderMarkers)`) so the shim passes the decision through instead of reconstructing it, and pin both hand-set states against C `jpeg_write_coefficients`. Pre-existing — the old shim had its own divergences here — surfaced by the P4-181 review.

**Why deferred.** P4-181 is a correctness fix for a warning-free drop-in; this is a byte-level fidelity gap on streams that already decode identically, and it touches the scan-script selection for 4-component progressive output, which deserves its own oracle run.
