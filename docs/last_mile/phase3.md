# Phase 3 — Long-Tail C Compatibility (0 OPEN items; all CLOSED)

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). All six originally-tracked Phase 3 items are CLOSED. P3-2 closed 2026-05-09 when the 12-bit raw-data backend was wired through `libjpeg_turbo_rs::raw_data_12::{compress,decompress}_raw_12`.

External review on 2026-05-06 (`libjpeg_turbo_rs_replacement_analysis.md`) graded the project's *Rust-application replacement* and *stock-tool drop-in* posture as ready, but flagged a long-tail of C-compatibility gaps that block the stronger claim "**any existing C binary linked against `libjpeg.so` / `libturbojpeg.so` runs unchanged**." Six of the seven gaps reproduce in this repository today. The seventh — `libjpeg.so.62` SONAME policy — is already closed under [P2-9](phase2.md#p2-9-v6b--v7--v8-abi-compatibility-matrix--closed) and is not re-listed here.

For each item below, the **agreement** line states whether the gap is reproducible in this repo right now (verified via grep / file inspection), and the **why** line states the consumer pattern that breaks if it stays open.

**Status summary:**

| ID | Status |
| --- | --- |
| P3-1 | CLOSED 2026-05-08 (6 of 6 struct cross-checks active; Windows MSVC matrix leg as Phase 4 hardening) |
| P3-2 | CLOSED 2026-05-09 (full 12-bit raw-data encode + decode backend wired) |
| P3-3 | CLOSED 2026-05-06 |
| P3-4 | CLOSED 2026-05-07 |
| P3-5 | CLOSED 2026-05-08 |
| P3-6 | CLOSED 2026-05-08 |

---

## P3-1. ABI Offset Cross-Check Was Decompress-Only — **CLOSED 2026-05-08**

**Status (2026-05-08): closed.** `crates/libjpeg-turbo-rs-capi/tests/abi_offsets.rs` now cross-checks all six originally-planned structs against upstream `jpeglib.h` at `JPEG_LIB_VERSION=80`. All six tests pass on macOS aarch64 (LP64) under `cargo test -p libjpeg-turbo-rs-capi --test abi_offsets --release` and run in CI on Linux x86_64 + macOS aarch64 via the `abi-offsets` matrix job:

- `rust_offsets_match_upstream_jpeglib_h_at_lib_version_80` — `jpeg_decompress_struct` (27 fields + sizeof, P2-4 baseline).
- `rust_offsets_match_jpeg_marker_struct_at_lib_version_80` — `jpeg_marker_struct` (5 fields + sizeof, 2026-05-08).
- `rust_offsets_match_jpeg_destination_mgr_at_lib_version_80` — `jpeg_destination_mgr` (5 fields + sizeof, **NEW 2026-05-08**).
- `rust_offsets_match_jpeg_source_mgr_at_lib_version_80` — `jpeg_source_mgr` (7 fields + sizeof, **NEW 2026-05-08**).
- `rust_offsets_match_jpeg_error_mgr_at_lib_version_80` — `jpeg_error_mgr` (14 fields + sizeof: `error_exit`/`emit_message`/`output_message`/`format_message`/`reset_error_mgr` callbacks, `msg_code`, `msg_parm` union slot, `trace_level`, `num_warnings`, message-table pointers + last-message indices; **NEW 2026-05-08**).
- `rust_offsets_match_jpeg_compress_struct_at_lib_version_80` — `jpeg_compress_struct` (75 fields + sizeof: `jpeg_common_fields` prefix, image description, JPEG_LIB_VERSION ≥ 70/80 extensions, scan state, marker emission, *and* the trailing 11 opaque libjpeg-internal pointers; **NEW 2026-05-08**).

**`sizeof` cross-check (every struct).** Each test additionally probes `sizeof(struct …)` via the same C harness and asserts it equals `mem::size_of::<MirrorStruct>()`. This closes the *false-coverage* gap a per-field check alone leaves open: a per-field check still passes if the Rust mirror truncates the struct's tail (so trailing fields drift silently); the size delta catches that. The error message reads `sizeof: Rust mirror is N bytes, C 'sizeof(struct X)' is M bytes — trailing field(s) are unmirrored or padding diverges`.

**Shared helper.** `cc_offsetof_for_struct(struct_name, &field_names)` in `tests/abi_offsets.rs` builds a one-shot C harness (`#include <setjmp.h>` + `<jpeglib.h>`), prints `field=offset` lines for each requested field plus a `__sizeof__=N` line, runs it through the host `cc`, and parses the output into a `CcProbeResult { offsets, sizeof }`. The harness pulls in `setjmp.h` so callbacks that take a `j_common_ptr` (which contains a `jmp_buf` field via `error_exit`) compile cleanly. Per-struct callsites only have to assemble the field-name list; the build/run/parse plumbing is shared. The closure landed all four large struct callsites unchanged through this helper — no additional MSVC plumbing was needed for green coverage on the existing Linux x86_64 + macOS aarch64 matrix legs.

**Field-name mapping.** All six callsites pass C-side field names directly to `cc_offsetof_for_struct`; the Rust-side `offset_of!(MirrorStruct, rust_field)` then compares to the C-reported offset. A single divergence exists where the Rust mirror renames a field to avoid an identifier collision: C's `struct jpeg_c_main_controller *main` is mirrored as `JpegCompressPublic::main_ctrl`. The cross-check records this as `("main", offset_of!(JpegCompressPublic, main_ctrl))` so the harness still queries `offsetof(struct jpeg_compress_struct, main)` against the upstream header.

**Platform gate.** `cc_offsetof_for_struct` returns `Skip(reason)` when `size_of::<usize>() != 8`, when the upstream submodule is not initialised, or when the host has no runnable `cc`. The 64-bit gate is *not* further restricted to non-Windows: `offset_of!` reads whatever struct layout the Rust compiler chose for the running host (LP64 on Linux/macOS, LLP64 on Windows MSVC) and the C harness compiled on the same host reports the same ABI, so a future Windows-MSVC matrix leg surfaces per-platform Rust↔C drift directly.

**TDD verification (2026-05-08).** Mutation-tested both axes for representative struct (`jpeg_compress_struct`):

1. Field-offset drift: temporarily added `+ 8` to `("main", offset_of!(JpegCompressPublic, main_ctrl))`. Result: `field 'main': Rust says offset 512, C says 504` (the C-side `main` mapping reaches the Rust mirror's `main_ctrl` field correctly). Reverted.
2. Tail truncation: previously verified on the marker struct (the older PR), exact same helper.

**Closure verification:** `cargo test -p libjpeg-turbo-rs-capi --test abi_offsets --release` reports `6 passed; 0 failed; 0 ignored` on the host (macOS aarch64).

**Future hardening (not blocking closure):**

- Windows MSVC leg of the `abi-offsets` matrix — helper currently emits gcc/clang flags only; adding MSVC dispatch (`/I`, `/Fe:`, `/Fo:`, `cl /?`) plus an `ilammy/msvc-dev-cmd@v1` step on `windows-latest` would surface LLP64-specific drift in CI. The runtime gate is already platform-aware (it reads the host's actual struct layout, not a per-platform const block), so a Windows leg would be additive coverage rather than fixing a known gap.
  - **Trigger:** Windows MSVC consumer, or a measurable LLP64-vs-LP64 layout concern.
- `jvirt_barray_control` / `jvirt_sarray_control` — opaque types in upstream `jpeglib.h` (forward-declared, definition in `jmemmgr.c` only). Consumers don't pin field offsets; they treat the `jvirt_*_ptr` as an opaque handle. **No cross-check needed**, kept here as a tracking note for the original P3-1 scope.

The six cross-checked structs cover every classic-API field surface that consumers (cjpeg / Pillow / ImageMagick / libgd / GIMP / FFmpeg / SDL_image) read by name. Encoder-side and decoder-side handles, plus the three manager structs that consumers customise (error / source / destination), are now all gated against upstream layout drift.

---

## P3-2. `jpeg12_write_raw_data` / `jpeg12_read_raw_data` Backend — **CLOSED 2026-05-09**

**Status (2026-05-09): closed — full 12-bit raw-data backend wired.** Both shim entry points now do real work instead of routing to `error_exit`:

- **Encode side** (`jpeg12_write_raw_data` → `run_raw_encoder_12_and_flush`): accumulates per-iMCU-row i16 planes into `CompressPrivate::raw_plane_buffers_12`, then on `jpeg_finish_compress` calls `libjpeg_turbo_rs::raw_data_12::compress_raw_12` with the shim-derived subsampling (`subsampling_from_comp_info`) and the same `quality` the caller installed via `jpeg_set_quality`. The encoded bytes flow through `push_bytes_through_dest_mgr` so caller-installed destination managers (mem-dest, stdio-dest, custom) work uniformly. The dispatch fork lives in `jpeg_finish_compress`: `c.data_precision == 12 && c.raw_data_in != 0` selects the 12-bit path.
- **Decode side** (`jpeg12_read_raw_data`): mirrors the 8-bit `jpeg_read_raw_data` lazily-materialised cache pattern but on i16 planes via `libjpeg_turbo_rs::raw_data_12::decompress_raw_12`. The first call materialises the whole stream into `DecompressPrivate::raw_image_cache_12`; subsequent calls deliver `max_v_samp_factor * DCT_v_scaled_size` rows per call into the caller's JSAMPIMAGE16 (`*mut *mut *mut i16`) array. Returns 0 with `priv_state.last_error` populated on precondition failures (wrong precision, NULL data, undersized `max_lines`); returns rows-per-iMCU on success.

The two error-exit stubs that previously gated this entry point under `JERR_NOTIMPL` are gone. The libjpeg.txt §3 contract — "an unimplemented codepath must routes through `error_exit`" — no longer applies because the codepath is now implemented.

**Implementation** (`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`):

- `CompressPrivate::raw_plane_buffers_12: Vec<Vec<i16>>` — per-component i16 plane accumulator (analog of the existing 8-bit `raw_plane_buffers`).
- `DecompressPrivate::raw_image_cache_12: Option<libjpeg_turbo_rs::raw_data_12::RawImage12>` — lazily-populated decode cache (analog of `raw_image_cache`).
- `run_raw_encoder_12_and_flush(c, priv_state) -> bool` — encoder helper symmetric to `run_raw_encoder_and_flush`, dispatched from `jpeg_finish_compress` when `c.data_precision == 12 && c.raw_data_in != 0`.
- `jpeg12_write_raw_data` — full body validates precision/state, computes per-component plane geometry from `comp_info` + `image_width/height` + `max_h_samp_factor/max_v_samp_factor`, copies caller's iMCU rows into the i16 buffer, advances `c.next_scanline`.
- `jpeg12_read_raw_data` — full body validates precision, lazily materialises `decompress_raw_12`, copies cached i16 rows into the caller's JSAMPIMAGE16 with the same MCU-aligned width contract the 8-bit path uses.

**Verification:**

- `cargo test -p libjpeg-turbo-rs-capi --release --test capi_jpeg12_raw_data_round_trip` → 1 passed. Three independent gates exercise the new code:
  1. Shim encode → native decode: Y-plane DCT round-trip max_diff ≤ 96 (out of 12-bit full scale 4095, ~2.3% — driven by 4:2:0 chroma smoothing + Q=90 quantisation; measured max_diff ≤ 64).
  2. Native encode → shim decode: shim's recovered Y plane is byte-identical to native `decompress_raw_12` output (same JPEG bytes, both backends route through the same code).
  3. Shim encode → shim decode: end-to-end Y max_diff ≤ 96 (proves both new paths compose).
- `cargo test --workspace --release` → 2156 passed / 0 failed / 1 ignored (the 1 ignored is the libtiff `JPEG_HEADER_TABLES_ONLY` shim gap tracked separately).
- `cargo build -p libjpeg-turbo-rs-capi --release` clean.
- `cargo clippy --lib --release -- -D warnings` clean.

**Test deletion (replaces obsolete contract).** `tests/capi_jpeg12_raw_data_error_exit.rs` previously pinned both stubs at "fires `error_exit` with `msg_code = JERR_NOTIMPL` (48)". With both stubs replaced by working backends that no longer route normal calls through `error_exit`, the contract is gone and the file is deleted. The new contract — "round-trip through both entry points produces correct samples within DCT tolerance" — is pinned by `tests/capi_jpeg12_raw_data_round_trip.rs`.

The closure title reflects the actual delta: stub *semantics* moved from "silent zero return" to "loud `error_exit`-driven failure." Symbol-presence and dyld-load compatibility are preserved.

---

## P3-3. Symbol Inventory Allowlist Triage — **CLOSED 2026-05-06**

**Status (2026-05-06): closed for the original Phase 3 release-gate consumer set; corrected 2026-05-10.** The stock tools and downstream smoke harnesses do not require the legacy TurboJPEG 1.x/2.x entry points that remain in `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs::allowlisted_missing_symbols()`, so those names stay intentionally allowlisted with a rationale. The only classic `jpeg_*` name that was still allowlisted, `jpeg_calc_jpeg_dimensions`, was a real C API gap; it is now closed under [P4-1](phase4.md#p4-1-jpeg_calc_jpeg_dimensions-was-documented-but-not-exported--closed-2026-05-10).

**Current contract:** `symbol_inventory` must pass with no missing upstream `jpeg_*` declarations. Legacy TJ aliases may remain allowlisted only when the comment explains the superseding exported API (`tjCompress2`, `tj3*`, etc.).

**Verification (2026-05-10 correction):** `cargo test -p libjpeg-turbo-rs-capi --test symbol_inventory --release -- --nocapture` → `2 passed`.

**Future hardening (not blocking closure):**
- Implement legacy TurboJPEG 1.x/2.x aliases if a concrete downstream binary requires them instead of the already-exported TJ2/TJ3 forms.
- Add per-family dlopen smoke tests for any alias family when it moves from allowlisted to implemented. The symbol-inventory test gates presence, not behaviour.

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

**Shim fix landed alongside the RGB565 fixture (2026-05-08).** The merged-upsample gate now in `src/decode/pipeline_impl/output.rs::Decoder::decode_image_inner` previously bound `out_format == PixelFormat::Rgb`, so `set_merged_upsample(true) + Rgb565` silently fell through to the slow path. Fix: lift the gate to also accept `Rgb565` and route the merged kernel's RGB output through a 5-6-5 truncation pass (matches upstream's `jdmrgext-*-565` semantics for the no-dither case). The dedicated SIMD `_565` kernels remain a Phase 4 perf task — the current path keeps the SIMD merged conversion to RGB and packs to RGB565 in scalar; pixel-correctness is unaffected.

**`cjpeg -sample 3x2,1x1,1x1` baseline.** The non-standard sampling factor matrix at the C cross-check now covers (3,2)/(1,1)/(1,1) and (3,1)/(1,1)/(1,1) — the two configurations the upstream documentation explicitly calls out as "non-standard" while still being valid per ITU-T T.81 Annex B (max sampling factor ≤ 4, sum of products ≤ 10).

**Closure delta:**
- `src/decode/pipeline_impl/output.rs::Decoder::decode_image_inner` — widen the merged-upsample gate from `out_format == PixelFormat::Rgb` to `Rgb || Rgb565`; pack RGB → RGB565 LE after the merged kernel runs.
- `tests/cross_check_p3_6_nonstandard_rgb565.rs` — new file with 4 fixtures matching the acceptance bar.
- `docs/FEATURE_PARITY.md` — update the "Merged upsampling" entry to reflect RGB + RGB565 wiring and call out the deferred SIMD `_565` kernels.

**Acceptance (historical):**
- ~~One fixture per: `3x2` decode, `3x2` encode, `3x1` decode, RGB565 merged-upsample (encode → decode through merged path).~~ **Done** — see test file above.
- ~~Cross-validate against upstream `cjpeg -sample 3x2,1x1,1x1` and `djpeg -rgb565`.~~ **Done** — `cjpeg -sample` for the 3x cases; `djpeg -nosmooth` for the merged-RGB565 chain (djpeg's `-rgb565` flag emits 24-bpp BMP, not a comparable raw 16-bpp file, so the chain via RGB → 5-6-5 truncation is the byte-comparable form).
- ~~If gaps remain after that minimum is in, document them in `docs/FEATURE_PARITY.md`.~~ **Done** — the deferred dedicated `_565` SIMD kernels are recorded there as a Phase 4 perf task.

---

## Phase 3 Suggested Order

1. ~~**P3-1** — Extend `tests/abi_offsets.rs` to `jpeg_compress_struct`, `jpeg_error_mgr`, `jpeg_source_mgr`, `jpeg_destination_mgr`, `jpeg_marker_struct`.~~ **CLOSED 2026-05-08** — all six originally-planned struct cross-checks active in `tests/abi_offsets.rs` (decompress + marker + compress + error_mgr + source_mgr + destination_mgr; 133 fields + 6 sizeof probes). C-side `main` field maps to Rust mirror `main_ctrl` via `(c_field_name, rust_offset)` tuple (only field-name divergence). `jvirt_*_control` opaque upstream → no cross-check needed. Windows MSVC matrix leg deferred as Phase 4 hardening (helper currently emits gcc/clang flags only); `cargo test -p libjpeg-turbo-rs-capi --test abi_offsets --release` reports `6 passed; 0 failed; 0 ignored` on macOS aarch64 + the `abi-offsets` CI matrix.
2. ~~**P3-3** — Audit the symbol inventory allowlist and keep only non-blocking legacy TJ aliases.~~ **CLOSED 2026-05-06; corrected 2026-05-10** — all upstream `jpeg_*` declarations now resolve; legacy TJ1/TJ2 names remain allowlisted with superseding TJ2/TJ3 exports documented in the test.
3. ~~**P3-2** — Either implement `jpeg12_write_raw_data` / `jpeg12_read_raw_data` against the existing 12-bit encode/decode backend, or downgrade them to feature-gated absence. The "stub returning 0" middle ground must end.~~ **CLOSED 2026-05-09** — both entry points wired through `libjpeg_turbo_rs::raw_data_12::{compress,decompress}_raw_12`. Round-trip pinned by `tests/capi_jpeg12_raw_data_round_trip.rs` (3 gates, GREEN).
4. ~~**P3-5** — Classic `jpeglib.h` lifecycle / custom-I/O / suspension C harness (≥ 8 tests).~~ **CLOSED 2026-05-08** — all 8 patterns active in `crates/libjpeg-turbo-rs-capi/tests/capi_classic_lifecycle.rs` (custom src/dst mgr, source suspension, destination-suspension `JERR_CANT_SUSPEND` contract, abort+reuse for both decompress/compress, buffered-image multi-pass, setjmp/longjmp). Pattern #4 surfaced + fixed a real shim defect: `push_bytes_through_dest_mgr` previously ignored `empty_output_buffer`'s `FALSE` return and called `term_destination` anyway, silently dropping the post-suspension bytes. Architectural reality is that the deferred-encode shim cannot honor upstream's per-MCU streaming-suspension contract at `jpeg_write_scanlines` (no encoding happens there), so the closure invokes `cinfo->err->error_exit` with `JERR_CANT_SUSPEND` (upstream code 25, "Suspension not allowed here") rather than inventing a non-upstream resume contract; a `setjmp`/`longjmp` consumer recovers cleanly. The earlier P3-5 pattern #8 fix (2026-05-07) wired `jpeg_read_header` to invoke `error_exit` on EOI-terminated malformed input.
5. ~~**P3-4** — Lift the 4-pixel chroma transform writer gate; close the P2-12 follow-up.~~ **CLOSED 2026-05-07** — gate at `transform_jpeg_with_options::progressive_safe` widened from `max_{h,v} ≤ 2` to `max_{h,v} ∈ {1,2,4}` (the eight standard TJSAMP factors verified by `c_tjtrantest_full`); `tests/c_tjtrantest.rs` skip removed; regression pinned in `tests/regression_progressive_4pixel_chroma_transform.rs` (256 cases). Full matrix runs 12,230 cases without divergence. The 2026-05-07 "1-LSB drift" hypothesis turned out to be an artefact of the encoder-side clamp that P2-11 had already removed; the transform writer inherits the corrected chroma layout via `read_coefficients`. Non-standard 3x sampling stays gated to baseline pending P3-6.
6. ~~**P3-6** — Non-standard sampling / RGB565 merged-upsample minimum fixture set.~~ **CLOSED 2026-05-08** — 4 fixtures (3x2 decode, 3x2 encode, 3x1 decode, RGB565 merged-upsample) all green in `tests/cross_check_p3_6_nonstandard_rgb565.rs`. Shim fix: merged-upsample gate widened from `Rgb` to `Rgb || Rgb565` with a 5-6-5 truncation pass after the merged kernel; dedicated `_565` SIMD kernels deferred as a Phase 4 perf task.

The order is intentional: P3-1 is the cheapest blast-radius reduction (one test file expansion catches a whole class of encode-side ABI drift); P3-3 keeps the symbol inventory honest by separating real classic-API gaps from intentionally-superseded legacy TJ aliases; P3-2 fixes a specific stub; P3-5 is structural but expensive; P3-4 / P3-6 are correctness gaps with narrower consumer impact.
