# Phase 3 — Long-Tail C Compatibility (0 OPEN items; P3-1/P3-2 PARTIAL)

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). This phase has no fully-OPEN items left — P3-1 and P3-2 retain narrow PARTIAL scope-tracking notes for follow-ups gated on downstream demand.

External review on 2026-05-06 (`libjpeg_turbo_rs_replacement_analysis.md`) graded the project's *Rust-application replacement* and *stock-tool drop-in* posture as ready, but flagged a long-tail of C-compatibility gaps that block the stronger claim "**any existing C binary linked against `libjpeg.so` / `libturbojpeg.so` runs unchanged**." Six of the seven gaps reproduce in this repository today. The seventh — `libjpeg.so.62` SONAME policy — is already closed under [P2-9](phase2.md#p2-9-v6b--v7--v8-abi-compatibility-matrix--closed) and is not re-listed here.

For each item below, the **agreement** line states whether the gap is reproducible in this repo right now (verified via grep / file inspection), and the **why** line states the consumer pattern that breaks if it stays open.

**Status summary:**

| ID | Status |
| --- | --- |
| P3-1 | PARTIAL (4 structs cross-checked; marker / virt_barray deferred) |
| P3-2 | PARTIAL (silent-zero stub eliminated; full backend deferred) |
| P3-3 | CLOSED 2026-05-06 |
| P3-4 | CLOSED 2026-05-07 |
| P3-5 | CLOSED 2026-05-08 |
| P3-6 | CLOSED 2026-05-08 |

---

## P3-1. ABI Offset Cross-Check Was Decompress-Only — **PARTIAL: 4 structs closed; marker / virt_barray deferred**

**Status (2026-05-06): partially closed.** `crates/libjpeg-turbo-rs-capi/tests/abi_offsets.rs` now cross-checks four structs against upstream `jpeglib.h` at `JPEG_LIB_VERSION=80`. The five tests in the file (one per struct) all pass on macOS aarch64 (LP64) and run in CI on Linux x86_64 (LP64) + Windows MSVC (LLP64) via the `abi-offsets` matrix job:

- `rust_offsets_match_upstream_jpeglib_h_at_lib_version_80` — `jpeg_decompress_struct` (27 fields, P2-4 baseline).
- `rust_offsets_match_jpeg_compress_struct_at_lib_version_80` — **NEW**, 75 fields covering jpeg_common_fields, image description, JPEG_LIB_VERSION ≥ 70 scale fields, primary compression parameters, marker emission, scan-state derived fields, JPEG_LIB_VERSION ≥ 80 extensions, *and* the trailing 11 opaque libjpeg-internal pointers (`master`, `main` ↔ Rust's `main_ctrl`, `prep`, `coef`, `marker`, `cconvert`, `downsample`, `fdct`, `entropy`, `script_space`, `script_space_size`).
- `rust_offsets_match_jpeg_error_mgr_at_lib_version_80` — **NEW**, 14 fields (`error_exit`/`emit_message`/`output_message`/`format_message`/`reset_error_mgr` callbacks, `msg_code`, `msg_parm` union slot, `trace_level`, `num_warnings`, message-table pointers + last-message indices).
- `rust_offsets_match_jpeg_source_mgr_at_lib_version_80` — **NEW**, 7 fields (`next_input_byte`, `bytes_in_buffer`, `init_source`, `fill_input_buffer`, `skip_input_data`, `resync_to_restart`, `term_source`).
- `rust_offsets_match_jpeg_destination_mgr_at_lib_version_80` — **NEW**, 5 fields (`next_output_byte`, `free_in_buffer`, `init_destination`, `empty_output_buffer`, `term_destination`).

**`sizeof` cross-check (every struct).** Each test additionally probes `sizeof(struct …)` via the same C harness and asserts it equals `mem::size_of::<MirrorStruct>()`. This closes the *false-coverage* gap a prior iteration of this work briefly opened: a per-field check alone would still pass if the Rust mirror truncated the struct's tail (so trailing fields drift silently); the size delta catches that. The error message reads `sizeof: Rust mirror is N bytes, C 'sizeof(struct X)' is M bytes — trailing field(s) are unmirrored or padding diverges`.

**Windows runs the test, not just LP64 hosts.** The platform gate is `host_is_64bit()` — *not* `64-bit AND non-Windows`. An earlier iteration mistakenly excluded Windows, which would have made the LAST_MILE.md "Windows MSVC (LLP64)" claim a false-coverage paper trail. The fix: `offset_of!` reads whatever struct layout the Rust compiler chose for the running host (LP64 on Linux/macOS, LLP64 on Windows MSVC) and the C harness compiled on the same host reports the same ABI, so per-platform divergence between our Rust mirror and upstream `jpeglib.h` shows up as a real test failure on Windows even though the *compile-time* `const_assert!` block in `jpeglib.rs` is gated on `not(windows)`. The runtime cross-check is the only gate that catches a Windows-specific layout drift; it must actually run there.

**CI cannot green-skip.** The `cc_offsetof_for_struct` helper has legitimate skip-with-reason paths for "no `cc` on PATH", "submodule not initialized", and "compile failed for missing-headers reasons". On a developer's local box those paths print a `SKIP: …` line and the test reports "ok". In CI (`CI=true` or `GITHUB_ACTIONS=true` set in env) the same paths route through `handle_environmental_skip(...)` and **panic** with a message that names the LAST_MILE.md coverage claim:

```text
ABI cross-check for `jpeg_compress_struct` cannot run in CI: <reason>
A green skip here would falsify the LAST_MILE.md claim that the
abi-offsets matrix gates Linux x86_64 / macOS aarch64 / Windows MSVC.
Either install the missing C compiler / submodule on the runner, or
remove that platform from the CI matrix and the LAST_MILE.md claim
simultaneously.
```

**MSVC support on Windows CI.** The first iteration of the CI hard-fail would have guaranteed-red the Windows-MSVC matrix leg because stock GitHub Actions `windows-latest` does not expose `cl.exe` (or any `cc`) on PATH until `vcvars*.bat` has run. Two changes close that loop:

1. `cc_offsetof_for_struct` detects MSVC by binary name (`cl` / `clang-cl`) and dispatches MSVC-style flags (`/I<dir>`, `/Fe:<path>`, `/Fo:<path>.obj`, `/nologo`, `/?` for the liveness probe instead of `--version`). Default compiler on Windows is `cl`; on Linux/macOS it stays `cc`. The `bin_path` includes `.exe` on Windows so the post-compile run finds the actual file MSVC produced.
2. `.github/workflows/ci.yml` adds an `ilammy/msvc-dev-cmd@v1` step on the `windows-latest` leg of `capi-abi-checks`. That action sets up the MSVC environment (PATH, INCLUDE, LIB) so `cl.exe` is reachable, exactly the way Visual Studio's own developer prompt would.

The MSVC error patterns recognised as environmental (vs hard-failure) are extended too: `Cannot open include file` and `fatal error C1083` route to skip-with-reason on local dev machines without MSVC, while panic-in-CI is preserved for actually unconfigured runners.

**TDD-verified on macOS aarch64 (2026-05-06):**

- `cargo test … abi_offsets` (default) — green, exercises the gcc/clang path with the host's `cc`.
- `CI=true CC=/nonexistent/cc-binary cargo test … rust_offsets_match_jpeg_compress_struct_at_lib_version_80` — hard-fail with the panic message above (gcc-style probe).
- `CI=true CC=/nonexistent/cl-binary cargo test …` — hard-fail with the same message but the MSVC code path (binary name ends in `cl`).
- `env -u CI -u GITHUB_ACTIONS CC=cl cargo test …` — soft `SKIP: C compiler cl not found or not runnable`, test reports green (legitimate dev skip when MSVC isn't installed locally).

The asymmetry forces the Windows-MSVC matrix runner to actually have MSVC reachable; otherwise the gate becomes a real CI failure instead of paper documentation. With the `ilammy/msvc-dev-cmd@v1` step in place, MSVC *is* reachable, so the test runs for real and surfaces any per-platform Rust↔C drift on Windows LLP64.

The shared helper `cc_offsetof_for_struct` builds a one-shot C harness against the submodule's `jpeglib.h` per struct, runs it, and parses `name=offset` lines back into a `CcProbeResult { offsets, sizeof }`. Failure messages name the field, the Rust offset, and the C offset side-by-side. The harness pulls in `setjmp.h` so callbacks that take a `j_common_ptr` (which contains a `jmp_buf` field via `error_exit`) compile cleanly.

**TDD verification (2026-05-06):** two mutation tests confirm the gate has real teeth, not just structural appearance.

1. Field-offset drift: temporarily rewrote `("main", offset_of!(JpegCompressPublic, main_ctrl))` to add `+ 8` to the Rust offset. Result: `field 'main': Rust says offset 512, C says 504`. Reverted.
2. Tail truncation: temporarily rewrote `let rust_sizeof = …;` to subtract 8 bytes. Result: `sizeof: Rust mirror is 576 bytes, C 'sizeof(struct jpeg_compress_struct)' is 584 bytes — trailing field(s) are unmirrored or padding diverges`. Reverted.

**Closure verification:** `cargo test -p libjpeg-turbo-rs-capi --test abi_offsets --release` reports `5 passed; 0 failed; 0 ignored` on the host (macOS aarch64).

**Still open (deferred — narrower consumer impact than the four above):**

- `jpeg_marker_struct` — `JpegMarkerStructPublic` mirrors it (`jpeglib.rs:177-183`) but is not yet covered by the cross-check. Stock `jpegtran` with `-copy all` already byte-matches upstream (P0-4 closure), which exercises this struct at runtime, so the layout is implicitly verified for the working path.
- `jvirt_barray_control` / `jvirt_sarray_control` — these are opaque types in upstream `jpeglib.h` (forward-declared, definition in `jmemmgr.c` only). Consumers don't pin field offsets; they treat the `jvirt_*_ptr` as an opaque handle. No cross-check needed.

The four cross-checked structs are the ones classic C consumers (cjpeg / Pillow / ImageMagick / libgd) read by field name, so the closure scope matches the analysis-document acceptance bar.

---

## P3-2. `jpeg12_write_raw_data` / `jpeg12_read_raw_data` Stub Semantics — **PARTIAL: error-exit semantics fixed; full 12-bit raw-data backend deferred**

**Status (2026-05-06): partial closure — silent-zero-return stub eliminated.** Both symbols still acknowledge that 12-bit raw-data is not implemented, but they no longer return `0` silently (which mimicked "no rows ready, retry later" and could spin a caller forever). Instead they invoke `cinfo->err->error_exit(cinfo)` with `msg_code = JERR_NOTIMPL` (upstream code 19), so:

- A caller with a `setjmp`-installed handler longjmps out of the call cleanly.
- A caller relying on the default `error_exit` aborts the process with a diagnostic on stderr.
- A caller that resolves the symbol only at dyld-load time (e.g. Pillow's libtiff dependency) is unaffected — symbol presence is preserved.

**Implementation** (`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs::trigger_error_exit_notimpl`):

```rust
fn trigger_error_exit_notimpl(cinfo: *mut c_void, _api_name: &str) {
    if cinfo.is_null() { return; }
    unsafe {
        let err_pp: *mut *mut JpegErrorMgr = cinfo as *mut *mut JpegErrorMgr;
        let err_ptr: *mut JpegErrorMgr = err_pp.read();
        if err_ptr.is_null() { return; }
        let err: &mut JpegErrorMgr = &mut *err_ptr;
        err.msg_code = JERR_NOTIMPL_CODE;  // upstream `JERR_NOTIMPL` = 19
        if let Some(exit) = err.error_exit { exit(cinfo); }
    }
}
```

Both `jpeg12_read_raw_data` and `jpeg12_write_raw_data` call this helper after populating `priv_state.last_error` (preserved for diagnostics), then return `0` only on the *unreachable* fall-through where a custom handler returns from `error_exit` (which violates the libjpeg contract — but defensive code is cheap).

**Why "partial" not "closed":** the symbols still don't *do* 12-bit raw-data. A consumer that wants real 12-bit raw-data encode/decode now gets a clean error path (good) but no working implementation (acceptable, but not the full P3-2 acceptance bar). The full bar — wire `cinfo.raw_data_in = TRUE` through the existing 12-bit encode backend (`compress_12bit_with_precision`), and mirror for decode — is deferred to Phase 4 work and gated on a downstream consumer surfacing demand.

**Verification:**
- `cargo test -p libjpeg-turbo-rs-capi --release --tests --no-fail-fast` → 35+ test binaries green; the only failure is the pre-existing `imagemagick_roundtrips_through_our_cdylib` (PSNR=22.6 dB, confirmed via `git stash` to be a pre-existing regression on `docs/fix-arith-contradiction`, not caused by P3-2).
- `cargo test -p libjpeg-turbo-rs-capi --test capi_jpeg_read_raw_data --test capi_jpeg_write_raw_data --release` → `2 passed` + `3 passed` (the existing 8-bit raw-data tests are unaffected).
- `cargo build -p libjpeg-turbo-rs-capi --release` clean.

The closure title reflects the actual delta: stub *semantics* moved from "silent zero return" to "loud `error_exit`-driven failure." Symbol-presence and dyld-load compatibility are preserved.

---

## P3-3. 19 Legacy TurboJPEG Symbols Are Allowlisted, Not Implemented — **CLOSED 2026-05-06**

**Status (2026-05-06): closed.** Every previously-allowlisted symbol is now implemented as a forwarding wrapper in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` and re-exported through `lib.rs`. `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs::allowlisted_missing_symbols()` returns an empty `HashSet`. Both `cdylib_exports_every_upstream_jpeglib_h_symbol` and `cdylib_exports_every_upstream_turbojpeg_h_symbol` pass without exemptions.

**What landed (19 wrappers):**

- `tjAlloc(int) → tj3Alloc(usize)` — rejects negative sizes (NULL return), matching upstream behaviour.
- `tjFree(*mut u8) → tj3Free(*mut c_void)`.
- `tjGetErrorStr() → tj3GetErrorStr(NULL)` — no-handle form, returns `*mut c_char` (the legacy ABI's mut-vs-const distinction is purely declaration; the buffer is library-owned).
- `tjGetErrorCode(handle) → tj3GetErrorCode(handle)`.
- `tjGetScalingFactors(*mut int) → tj3GetScalingFactors(...)` — static-table forwarding.
- `tjCompress(handle, src, w, pitch, h, **pixSize**, dst, *compSize, jpegSubsamp, jpegQual, flags) → tjCompress2`. Widens `pixelSize` to `pixelFormat` (3→TJPF_RGB, 4→TJPF_RGBX); sets `TJPARAM_NOREALLOC=1` so the caller's preallocated `dstBuf` is honoured; threads the unsigned-long `*compSize` through a usize stash.
- `tjDecompress(handle, jpeg, jpegSize, dst, w, pitch, h, **pixSize**, flags) → tjDecompress2`. Same `pixelSize` widening + legacy flag translation.
- `tjDecompressHeader(handle, jpeg, jpegSize, *w, *h) → tjDecompressHeader3` — drops subsamp/colorspace outs.
- `tjDecompressHeader2(handle, jpeg, jpegSize, *w, *h, *subsamp) → tjDecompressHeader3` — drops colorspace out.
- `tjDecompressToYUV(handle, jpeg, jpegSize, dst, flags) → tj3DecompressToYUV8` with `align=4` per upstream default.
- `tjDecompressToYUV2(handle, jpeg, jpegSize, dst, w, **align**, h, flags) → tj3DecompressToYUV8` with caller-specified `align`.
- `tjDecompressToYUVPlanes(handle, jpeg, jpegSize, **dst, w, *strides, h, flags) → tj3DecompressToYUVPlanes8` + flag translation.
- `tjEncodeYUV(handle, src, w, pitch, h, **pixSize**, dst, subsamp, flags) → tjEncodeYUV3` with default `align=4`.
- `tjEncodeYUV2(handle, src, w, pitch, h, pixFmt, dst, subsamp, flags) → tjEncodeYUV3` with default `align=4`.
- `tjEncodeYUVPlanes(handle, src, w, pitch, h, pixFmt, **dst, *strides, subsamp, flags) → tj3EncodeYUVPlanes8`.
- `tjCompressFromYUV(handle, src, w, **align**, h, subsamp, **jpeg, *jpegSize, qual, flags) → tj3CompressFromYUV8` with usize-↔-c_ulong stash for `*jpegSize`.
- `tjCompressFromYUVPlanes(handle, **srcPlanes, w, *strides, h, subsamp, **jpeg, *jpegSize, qual, flags) → tj3CompressFromYUVPlanes8`.
- `tjDecodeYUVPlanes(handle, **srcPlanes, *strides, subsamp, dst, w, pitch, h, pixFmt, flags) → tj3DecodeYUVPlanes8`.
- `jpeg_calc_jpeg_dimensions(j_compress_ptr) → void` — populates `cinfo.jpeg_width` / `cinfo.jpeg_height` from `scale_num` / `scale_denom` / `image_width` / `image_height`. Mirrors `jcparam.c::jpeg_calc_jpeg_dimensions`. Uses `JpegCompressPublic` (the public ABI mirror) for field access.

**Verification:** `cargo test -p libjpeg-turbo-rs-capi --test symbol_inventory --release` → `2 passed`. Full capi test suite (35+ test binaries) regresses zero existing tests; the only failure remaining (`imagemagick_roundtrips_through_our_cdylib`, PSNR=22.6 dB) is **pre-existing** — confirmed via `git stash`/retry on the baseline branch — and tracked separately under `docs/fix-arith-contradiction`. `cargo fmt --all -- --check` clean. `cargo clippy --lib -- -D warnings` clean.

**Future hardening (not blocking closure):**
- Add per-family dlopen smoke tests (`tjAlloc/tjFree` round-trip, `tjCompress` legacy-ABI round-trip, `tjEncodeYUV` legacy-ABI round-trip, `jpeg_calc_jpeg_dimensions` math) to `legacy_aliases.rs`. The symbol-inventory test gates *presence*, not *behaviour*; smoke tests gate behaviour. Tracked as a fast-follow.

---

## P3-4. 4-Pixel Chroma Progressive Transform Writer Gate — **CLOSED 2026-05-07**

**Status (2026-05-07): closed.** The `max_h <= 2 && max_v <= 2` clamp in the `progressive_safe` predicate (`src/api/coefficient.rs::transform_jpeg_with_options`) was widened to `max_h ∈ {1,2,4} && max_v ∈ {1,2,4}` — the eight standard TJSAMP factors (444/422/440/420/411/441/410/24). Non-standard 3x sampling (`max_h = 3` or `max_v = 3`) stays gated to baseline pending [P3-6](#p3-6-non-standard-sampling--rgb565-merged-upsample--open-p2-priority); P3-4 only verified standard factors against `jpegtran -progressive`, so widening further would claim coverage that does not yet exist. The underlying data-block sanity check (`data_blocks_{x,y} ≤ comp.blocks_{x,y}`) is preserved as a malformed-coefficient guard. Empirical verification on the host (macOS aarch64) confirms zero divergence between `transform_jpeg_with_options(..., progressive = true, copy_markers = All)` and `jpegtran -progressive -copy all <op>` across all four 4-pixel chroma sampling factors and all eight transform ops, on both iMCU-aligned and non-aligned image sizes, for both Rust-encoded and cjpeg-encoded source baselines.

**What turned out to be wrong with the 2026-05-07 hypothesis.** The hypothesis described a "1-LSB chroma layout drift" that lifting the gate would surface as CI failure. In practice — replayed via `examples/diag_4pixel_chroma_transform_diff.rs`, then via the dedicated regression test, then via the full `c_tjtrantest_full` matrix (12,230 cases tested, up from 11,718 before the skip removal) — there is no divergence to surface. The earlier skip almost certainly tracked a defect that was incidentally fixed by the [P2-11](phase2.md#p2-11-tjsamp_411--tjsamp_441--tjsamp_410--tjsamp_24-progressive-encode--closed) encoder-side clamp removal in `progressive_fdct_chroma_block`. The source-level evidence: the existing regression `progressive_baseline_byte_match` exercises identical chroma layouts on the encode side, and the transform writer reads coefficients shaped by the same downsampling rules — once the encoder produces the correct chroma layout, the transform writer that round-trips through `read_coefficients` inherits it. The `c_tjcomptest.rs:732` comment block (encoder-side companion) had already noted the 1-LSB diagnosis was a misread of pixel-level diffs at max=140-161, not 1; the transform-side skip was an artefact of the same misread, kept defensive after P2-11 closed encode parity.

**Verification chain:**
- `cargo run --release --example diag_4pixel_chroma_transform_diff` — 384 cases (6 sizes × 4 samplings × 2 origins × 8 ops) all byte-equal vs `jpegtran -progressive -copy all <op>`.
- `cargo test --release --test regression_progressive_4pixel_chroma_transform` — pinned regression with 256 cases (4 sizes × 4 samplings × 2 origins × 8 ops); panics if the gate is reintroduced.
- `cargo test --release --features full-c-parity --test c_tjtrantest c_tjtrantest_full` — full upstream `tjtrantest.in` matrix without the `tests/c_tjtrantest.rs:537-543` skip; reports `12230 tested, 18498 skipped` (vs `~11700 tested, ~19000 skipped` before).

**Closure delta:**
- `src/api/coefficient.rs::transform_jpeg_with_options` — widen the `progressive_safe` sampling-factor gate from `max_h ≤ 2 && max_v ≤ 2` to `max_h ∈ {1,2,4} && max_v ∈ {1,2,4}` (`is_power_of_two` + bound). Non-standard 3x sampling stays on the baseline path until P3-6.
- `tests/c_tjtrantest.rs:529-543` — remove the source-level skip block.
- `tests/regression_progressive_4pixel_chroma_transform.rs` — new pinned regression.
- `examples/diag_4pixel_chroma_transform_diff.rs` — new diagnostic, mirrors the P2-11 `diag_4pixel_chroma_diff.rs` style.

---

## P3-5. Classic `jpeglib.h` Lifecycle / Custom-I/O / Suspension C Harness — **CLOSED 2026-05-08**

**Status (2026-05-08): closed — 8 of 8 patterns landed.** `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle.rs` exercises every pattern #1..#8 against the cdylib via real C harnesses; `cargo test --release -p libjpeg-turbo-rs-capi --test capi_classic_lifecycle` reports `8 passed, 0 ignored, 0 failed`.

**Shim fix landed alongside pattern #4 (2026-05-08).** Pattern #4 surfaced a real shim defect: `push_bytes_through_dest_mgr` in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` ignored the boolean return of `empty_output_buffer` and unconditionally called `term_destination` even when the consumer signalled I/O suspension. A custom destination manager that returned `FALSE` (libjpeg.txt §5.5) saw the stream silently terminated and the post-suspension bytes dropped on the floor. Architectural reality: the shim's deferred-encode model buffers all pixels in `jpeg_write_scanlines` and runs the entire encoder synchronously inside `jpeg_finish_compress`, so upstream's per-MCU streaming-suspension contract — which fires at the entropy coder and unwinds back through `jpeg_write_scanlines` to return a short row count — cannot be honored at the right API boundary. Inventing a non-upstream resume contract at `jpeg_finish_compress` would create a callable surface no real consumer expects. The honest closure: capture the `FALSE` return, stash a diagnostic in `CompressPrivate::last_error`, skip `term_destination`, and invoke `cinfo->err->error_exit` with `msg_code = JERR_CANT_SUSPEND` (upstream code 25, "Suspension not allowed here" — exactly the message upstream uses for this contract violation). A consumer using the documented `setjmp`/`longjmp` pattern recovers cleanly; one without the longjmp hits the default abort. Either way the silent-data-loss path is gone.

**Shim fix landed alongside pattern #8 (2026-05-07).** Pattern #8 surfaced a real shim defect: `jpeg_read_header` in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` returned `JPEG_SUSPENDED` (= 0) on every `Decoder::new` rejection, conflating "input is incomplete (need more data)" with "input is syntactically complete but corrupt." The libjpeg.txt §3 contract is to invoke `cinfo->err->error_exit` for the second case so a `setjmp`/`longjmp` consumer can recover. Fix: a new `invoke_error_exit` helper that walks `cinfo->err` and dispatches `error_exit`, plus a guarded call site in `jpeg_read_header` that fires it only when the input bytes terminate in `FF D9` (a heuristic for "syntactically complete"). Truncated input still returns `JPEG_SUSPENDED`, preserving pattern #3's suspension semantics.

**Agreement (historical, original gap):** verified open. The harnesses in `crates/libjpeg-turbo-rs-capi/tests/` cover Pillow / ImageMagick / libvips / FFmpeg / GD / SDL_image consumers and the raw-data symbol exports, but not the *classic state-machine* edge cases an arbitrary C consumer can construct. Specifically missing:

1. Custom `jpeg_source_mgr` (callback-driven, small buffers).
2. Custom `jpeg_destination_mgr` with `empty_output_buffer` flush stress.
3. Source suspension (`fill_input_buffer` returns `FALSE`) and consumer-driven resume.
4. Destination suspension / partial flush.
5. `jpeg_abort_decompress` followed by re-use of the same `jpeg_decompress_struct`.
6. `jpeg_abort_compress` followed by re-use.
7. Buffered-image multi-pass progressive scan loop (`jpeg_consume_input` + `jpeg_start_output` + `jpeg_finish_output` per scan).
8. `setjmp`/`longjmp` error-cleanup pattern with a custom `error_exit`.

**Why this matters:** These eight patterns are the *raison d'être* of the classic `jpeglib.h` API. A C consumer that streams JPEGs over a network socket builds a custom source manager with suspension. A consumer that writes to a memory ring buffer builds a custom destination manager. A long-running daemon reuses one `jpeg_decompress_struct` across thousands of inputs and must `jpeg_abort_decompress` cleanly between them. The Pillow / ImageMagick harnesses don't exercise any of this — those wrappers use the canned `jpeg_mem_src` / `jpeg_mem_dest` paths.

**Acceptance:**
- Add `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle.rs` with one `#[test]` per pattern above (≥ 8 tests). Each test compiles a small C harness via `cc::Build` (or shells out to `cc` like `abi_offsets.rs`), links against the cdylib, and asserts the lifecycle behaviour matches what the same C harness produces against upstream `libjpeg.so` (when available; skip-with-reason otherwise).
- Per-test acceptance: byte-exact output (encode/decode) or pixel-exact (decode), and clean exit (no leaked file descriptors, no `error_exit` paths beyond the ones the test deliberately drives).
- Suspension tests must verify that the shim returns control to the consumer — not loop forever, not abort. The current `JpegSource::None` handling has historically swallowed suspension; this gap holds the regression.

---

## P3-6. Non-Standard Sampling / RGB565 Merged-Upsample — **CLOSED 2026-05-08**

**Status (2026-05-08): closed.** All four minimum-coverage fixtures land in `tests/cross_check_p3_6_nonstandard_rgb565.rs`:

- **3x2 decode** — `cjpeg -sample 3x2,1x1,1x1` produces a JPEG; Rust decode pixel-identical to `djpeg`.
- **3x2 encode** — Rust encodes at `(3,2)/(1,1)/(1,1)`; `djpeg` decode of Rust's output is within `max_diff ≤ 8` of the `cjpeg`+`djpeg` reference pipeline (lossy-encode quantization tolerance).
- **3x1 decode** — `cjpeg -sample 3x1,1x1,1x1` produces a JPEG; Rust decode pixel-identical to `djpeg`.
- **RGB565 merged-upsample** — Rust `merged_upsample=true + RGB565` for S420/S422 byte-identical to `djpeg -nosmooth RGB → 5-6-5 truncate` chain.

**Shim fix landed alongside the RGB565 fixture (2026-05-08).** The merged-upsample gate at `src/decode/pipeline.rs::Decoder::decode` previously bound `out_format == PixelFormat::Rgb`, so `set_merged_upsample(true) + Rgb565` silently fell through to the slow path. Fix: lift the gate to also accept `Rgb565` and route the merged kernel's RGB output through a 5-6-5 truncation pass (matches upstream's `jdmrgext-*-565` semantics for the no-dither case). The dedicated SIMD `_565` kernels remain a Phase 4 perf task — the current path keeps the SIMD merged conversion to RGB and packs to RGB565 in scalar; pixel-correctness is unaffected.

**`cjpeg -sample 3x2,1x1,1x1` baseline.** The non-standard sampling factor matrix at the C cross-check now covers (3,2)/(1,1)/(1,1) and (3,1)/(1,1)/(1,1) — the two configurations the upstream documentation explicitly calls out as "non-standard" while still being valid per ITU-T T.81 Annex B (max sampling factor ≤ 4, sum of products ≤ 10).

**Closure delta:**
- `src/decode/pipeline.rs::Decoder::decode` — widen the merged-upsample gate from `out_format == PixelFormat::Rgb` to `Rgb || Rgb565`; pack RGB → RGB565 LE after the merged kernel runs.
- `tests/cross_check_p3_6_nonstandard_rgb565.rs` — new file with 4 fixtures matching the acceptance bar.
- `docs/FEATURE_PARITY.md` — update the "Merged upsampling" entry to reflect RGB + RGB565 wiring and call out the deferred SIMD `_565` kernels.

**Acceptance (historical):**
- ~~One fixture per: `3x2` decode, `3x2` encode, `3x1` decode, RGB565 merged-upsample (encode → decode through merged path).~~ **Done** — see test file above.
- ~~Cross-validate against upstream `cjpeg -sample 3x2,1x1,1x1` and `djpeg -rgb565`.~~ **Done** — `cjpeg -sample` for the 3x cases; `djpeg -nosmooth` for the merged-RGB565 chain (djpeg's `-rgb565` flag emits 24-bpp BMP, not a comparable raw 16-bpp file, so the chain via RGB → 5-6-5 truncation is the byte-comparable form).
- ~~If gaps remain after that minimum is in, document them in `docs/FEATURE_PARITY.md`.~~ **Done** — the deferred dedicated `_565` SIMD kernels are recorded there as a Phase 4 perf task.

---

## Phase 3 Suggested Order

1. ~~**P3-1** — Extend `tests/abi_offsets.rs` to `jpeg_compress_struct`, `jpeg_error_mgr`, `jpeg_source_mgr`, `jpeg_destination_mgr`.~~ **PARTIAL 2026-05-06** — five struct cross-checks now pass (`jpeg_decompress_struct` + four new ones); marker / virt_barray are deferred per their consumer-impact rationale.
2. ~~**P3-3** — Implement the 19 legacy TurboJPEG aliases as forwarding wrappers and delete them from the allowlist.~~ **CLOSED 2026-05-06** — `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs::allowlisted_missing_symbols()` returns an empty `HashSet`; both `cdylib_exports_every_upstream_*` tests pass without exemptions. The wrappers live in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` (~390 lines for the new section).
3. ~~**P3-2** — Either implement `jpeg12_write_raw_data` / `jpeg12_read_raw_data` against the existing 12-bit encode/decode backend, or downgrade them to feature-gated absence. The "stub returning 0" middle ground must end.~~ **PARTIAL 2026-05-06** — middle ground eliminated: stubs now invoke `cinfo->err->error_exit(cinfo)` with `msg_code = JERR_NOTIMPL`, so callers either longjmp out cleanly or hit a default abort. Full 12-bit raw-data backend deferred to Phase 4 (gated on downstream demand).
4. ~~**P3-5** — Classic `jpeglib.h` lifecycle / custom-I/O / suspension C harness (≥ 8 tests).~~ **CLOSED 2026-05-08** — all 8 patterns active in `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle.rs` (custom src/dst mgr, source suspension, destination-suspension `JERR_CANT_SUSPEND` contract, abort+reuse for both decompress/compress, buffered-image multi-pass, setjmp/longjmp). Pattern #4 surfaced + fixed a real shim defect: `push_bytes_through_dest_mgr` previously ignored `empty_output_buffer`'s `FALSE` return and called `term_destination` anyway, silently dropping the post-suspension bytes. Architectural reality is that the deferred-encode shim cannot honor upstream's per-MCU streaming-suspension contract at `jpeg_write_scanlines` (no encoding happens there), so the closure invokes `cinfo->err->error_exit` with `JERR_CANT_SUSPEND` (upstream code 25, "Suspension not allowed here") rather than inventing a non-upstream resume contract; a `setjmp`/`longjmp` consumer recovers cleanly. The earlier P3-5 pattern #8 fix (2026-05-07) wired `jpeg_read_header` to invoke `error_exit` on EOI-terminated malformed input.
5. ~~**P3-4** — Lift the 4-pixel chroma transform writer gate; close the P2-12 follow-up.~~ **CLOSED 2026-05-07** — gate at `transform_jpeg_with_options::progressive_safe` widened from `max_{h,v} ≤ 2` to `max_{h,v} ∈ {1,2,4}` (the eight standard TJSAMP factors verified by `c_tjtrantest_full`); `tests/c_tjtrantest.rs` skip removed; regression pinned in `tests/regression_progressive_4pixel_chroma_transform.rs` (256 cases). Full matrix runs 12,230 cases without divergence. The 2026-05-07 "1-LSB drift" hypothesis turned out to be an artefact of the encoder-side clamp that P2-11 had already removed; the transform writer inherits the corrected chroma layout via `read_coefficients`. Non-standard 3x sampling stays gated to baseline pending P3-6.
6. ~~**P3-6** — Non-standard sampling / RGB565 merged-upsample minimum fixture set.~~ **CLOSED 2026-05-08** — 4 fixtures (3x2 decode, 3x2 encode, 3x1 decode, RGB565 merged-upsample) all green in `tests/cross_check_p3_6_nonstandard_rgb565.rs`. Shim fix: merged-upsample gate widened from `Rgb` to `Rgb || Rgb565` with a 5-6-5 truncation pass after the merged kernel; dedicated `_565` SIMD kernels deferred as a Phase 4 perf task.

The order is intentional: P3-1 is the cheapest blast-radius reduction (one test file expansion catches a whole class of encode-side ABI drift); P3-3 is the most valuable gate-removal (19 symbols disappear from "trust me" status); P3-2 fixes a specific stub; P3-5 is structural but expensive; P3-4 / P3-6 are correctness gaps with narrower consumer impact.
