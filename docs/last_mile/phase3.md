# Phase 3 — Long-Tail C Compatibility (3 OPEN items)

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). This phase has open work — keep this file in mind when planning new gaps.

External review on 2026-05-06 (`libjpeg_turbo_rs_replacement_analysis.md`) graded the project's *Rust-application replacement* and *stock-tool drop-in* posture as ready, but flagged a long-tail of C-compatibility gaps that block the stronger claim "**any existing C binary linked against `libjpeg.so` / `libturbojpeg.so` runs unchanged**." Six of the seven gaps reproduce in this repository today. The seventh — `libjpeg.so.62` SONAME policy — is already closed under [P2-9](phase2.md#p2-9-v6b--v7--v8-abi-compatibility-matrix--closed) and is not re-listed here.

For each item below, the **agreement** line states whether the gap is reproducible in this repo right now (verified via grep / file inspection), and the **why** line states the consumer pattern that breaks if it stays open.

**Status summary:**

| ID | Status |
| --- | --- |
| P3-1 | PARTIAL (4 structs cross-checked; marker / virt_barray deferred) |
| P3-2 | PARTIAL (silent-zero stub eliminated; full backend deferred) |
| P3-3 | CLOSED 2026-05-06 |
| **P3-4** | **OPEN** (root cause scoped 2026-05-07; full fix deferred) |
| **P3-5** | **OPEN** |
| **P3-6** | **OPEN** (P2 priority — narrow consumer impact) |

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

## P3-4. 4-Pixel Chroma Progressive Transform Writer Gate — **OPEN (root cause scoped 2026-05-07; full fix deferred)**

**Status (2026-05-07):** verified open with a concrete root-cause hypothesis. The `max_h <= 2 && max_v <= 2` gate at `src/api/coefficient.rs:1047` is *not* over-conservative: the source-level skip in `tests/c_tjtrantest.rs:529-543` documents the underlying defect — "our chroma layout for these factors differs from cjpeg by 1 LSB and cascades through per-scan optimised tables." Lifting the gate without fixing the chroma layout would expose the 1-LSB divergence as a real CI failure.

**Why this matters:** the encoder pipeline writes every progressive 4-pixel-chroma sampling factor (S411/441/410/24) byte-exactly against C now ([P2-11](phase2.md#p2-11-tjsamp_411--tjsamp_441--tjsamp_410--tjsamp_24-progressive-encode--closed) closure removed a `if h_samp > 1 { 2 } else { 1 }` clamp in `progressive_fdct_chroma_block`). The same sampling factors going through the *transform* writer (`write_coefficients_progressive` in `src/api/coefficient.rs`) still produce a 1-LSB chroma layout drift. `jpegtran -progressive` against a 4-pixel-chroma source therefore falls back to baseline silently when the gate fires, breaking `jpegtran -progressive` parity for those sampling factors.

**Root-cause hypothesis (next-session investigation start):** the encoder-side clamp removal (P2-11) only covers `progressive_fdct_chroma_block` plus the matching arithmetic-progressive Cb/Cr branches at `pipeline.rs:4761` / `:4781`. The *transform* writer takes a different path (`write_coefficients_progressive` works from already-decoded `JpegCoefficients`, not raw chroma planes). The 1-LSB drift therefore is **not** in the chroma-sampling-factor clamp — that fix is encoder-only — but in either:

1. The block-rotate / block-flip math for 4-pixel-sampling chroma layouts in the transform pipeline (`src/transform/*`), where the iMCU bookkeeping for `max_h > 2` may not match cjpeg's `transupp.c` row/column ordering.
2. The progressive coefficient writer's per-scan iMCU traversal at 4-pixel-sampling factors (`src/api/coefficient.rs::write_coefficients_progressive`), which may emit blocks in a different order than cjpeg's scan-by-scan progressive writer when `max_h_samp_factor > 2`.

**Acceptance (closure bar — unchanged):**
- Diagnose whether the drift originates in the transform block-reorder math or the progressive writer's iMCU traversal. Reuse `examples/diag_4pixel_chroma_diff.rs` as a template — adapt it for the transform path (input: 4-pixel-chroma JPEG; both Rust `transform_jpeg_with_options(..., progressive=true)` and stock `jpegtran -progressive -copy all <op>`; compare bytes + decoded pixels).
- Fix the divergence at the source.
- Lift the gate at `src/api/coefficient.rs:1047`.
- Pin a regression test (`tests/regression_progressive_4pixel_chroma_transform.rs`) that round-trips each of S411 / S441 / S410 / S24 through every transform op with `progressive_output = true`, asserting byte-equality against stock `jpegtran -progressive -copy all <op>`.
- Remove the source-level skip in `tests/c_tjtrantest.rs:537-543`.

**Why deferred (rule-compliance note):** the CLAUDE.md rule forbids fallback workarounds. The current gate *is* a fallback, but lifting it without the chroma-layout fix would silently produce non-byte-exact output that fails CI — the worse of two failure modes. The honest action is to leave the gate in place with this precise scope-tracking entry so the next investigation session can start from a concrete root-cause hypothesis instead of from "something is wrong."

---

## P3-5. Classic `jpeglib.h` Lifecycle / Custom-I/O / Suspension C Harness — **OPEN**

**Agreement:** verified open. The harnesses in `crates/libjpeg-turbo-rs-capi/tests/` cover Pillow / ImageMagick / libvips / FFmpeg / GD / SDL_image consumers and the raw-data symbol exports, but not the *classic state-machine* edge cases an arbitrary C consumer can construct. Specifically missing:

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

## P3-6. Non-Standard Sampling / RGB565 Merged-Upsample — **OPEN (P2 priority)**

**Agreement:** verified partially open. `src/encode/pipeline.rs:9952` claims support for "non-standard sampling configurations such as 3x2, 3x1, 1x3"; `src/decode/pipeline.rs:18` and `src/decode/color.rs:98+98` cover RGB565 + dithered RGB565 *output*, but the **merged-upsample** SIMD path (`jdmrgext-sse2.asm` / `jdmrgext-avx2.asm` in upstream) is not wired against the RGB565 output type, and there is no fixture matrix verifying 3x2 sampling decode/encode round-trip against C.

**Why this matters:** lower priority than P3-1..P3-5 because the consumer demand is narrower (classic `jpeglib.h` callers that override `comp_info[i].h_samp_factor` / `v_samp_factor` to non-power-of-two values, plus embedded-display callers using RGB565). Still a real parity gap if "anything `cjpeg` / `djpeg` accepts must round-trip through us" is the goal.

**Acceptance (minimum, not a full sweep):**
- One fixture per: `3x2` decode, `3x2` encode, `3x1` decode, RGB565 merged-upsample (encode → decode through merged path).
- Cross-validate against upstream `cjpeg -sample 3x2,1x1,1x1` and `djpeg -rgb565`.
- If gaps remain after that minimum is in, document them in `docs/FEATURE_PARITY.md` rather than silently passing.

---

## Phase 3 Suggested Order

1. ~~**P3-1** — Extend `tests/abi_offsets.rs` to `jpeg_compress_struct`, `jpeg_error_mgr`, `jpeg_source_mgr`, `jpeg_destination_mgr`.~~ **PARTIAL 2026-05-06** — five struct cross-checks now pass (`jpeg_decompress_struct` + four new ones); marker / virt_barray are deferred per their consumer-impact rationale.
2. ~~**P3-3** — Implement the 19 legacy TurboJPEG aliases as forwarding wrappers and delete them from the allowlist.~~ **CLOSED 2026-05-06** — `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs::allowlisted_missing_symbols()` returns an empty `HashSet`; both `cdylib_exports_every_upstream_*` tests pass without exemptions. The wrappers live in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` (~390 lines for the new section).
3. ~~**P3-2** — Either implement `jpeg12_write_raw_data` / `jpeg12_read_raw_data` against the existing 12-bit encode/decode backend, or downgrade them to feature-gated absence. The "stub returning 0" middle ground must end.~~ **PARTIAL 2026-05-06** — middle ground eliminated: stubs now invoke `cinfo->err->error_exit(cinfo)` with `msg_code = JERR_NOTIMPL`, so callers either longjmp out cleanly or hit a default abort. Full 12-bit raw-data backend deferred to Phase 4 (gated on downstream demand).
4. **P3-5** — Classic `jpeglib.h` lifecycle / custom-I/O / suspension C harness (≥ 8 tests). The most expensive item; defer until 1–3 are clean.
5. **P3-4** — Lift the 4-pixel chroma transform writer gate; close the P2-12 follow-up. **Open (root cause scoped 2026-05-07)** — `tests/c_tjtrantest.rs:529-543` confirms the gate is masking a 1-LSB chroma layout drift in the transform path, not a redundant guard. The gate stays in place until the underlying drift in `src/transform/*` (block-reorder math) or `src/api/coefficient.rs::write_coefficients_progressive` (per-scan iMCU traversal) is fixed; the entry above documents next-session investigation start points.
6. **P3-6** — Non-standard sampling / RGB565 merged-upsample minimum fixture set. P2 priority; do only if a downstream consumer requests it or after 1–5 are clean.

The order is intentional: P3-1 is the cheapest blast-radius reduction (one test file expansion catches a whole class of encode-side ABI drift); P3-3 is the most valuable gate-removal (19 symbols disappear from "trust me" status); P3-2 fixes a specific stub; P3-5 is structural but expensive; P3-4 / P3-6 are correctness gaps with narrower consumer impact.
