# Last Mile — libjpeg-turbo-rs vs libjpeg-turbo (C)

> **Date**: 2026-04-27 · **Branch**: `main` · **Reference**: libjpeg-turbo 3.1.3
>
> Goal: ship the remaining gaps so any reasonable libjpeg-turbo caller (TJ3, classic libjpeg, stock djpeg/cjpeg/jpegtran, downstream wrappers like Pillow / ImageMagick / GraphicsMagick) can swap our cdylib in without source-code changes.
>
> **Source of truth** (per `CLAUDE.md`): `docs/FEATURE_PARITY.md` (`[x]`/`[ ]` checklist) and `docs/C_API_REFERENCE.md` (✅/❌/🔶 mapping). This document is a *working triage view* derived from them — it groups gaps by impact, attaches effort estimates and validation steps, and proposes an ordering. When you close a gap, update the canonical row in `FEATURE_PARITY.md` / `C_API_REFERENCE.md` first; this doc may lag and that is fine.

---

## 1. Status Snapshot

| Layer | Coverage | Evidence |
| --- | --- | --- |
| Rust public API | ~complete | All SOF0/2/3/9/10/11 encode+decode; 8/12/16-bit; 12 pixel formats; 7 subsamplings; lossless predictors 1–7; arithmetic 16-table; abbreviated datastreams; ICC; restart resync hook |
| TJ3 C ABI | ~complete | All TJ3 functions + 26 TJPARAMs + legacy TJ1/TJ2 aliases + SONAME (`libjpeg.so.62`, `libturbojpeg.so.0`) + pkg-config |
| Classic `jpeg_*` ABI | partial | 46 `#[no_mangle]` exports cover stock djpeg/cjpeg/jpegtran transcode (no transform), Pillow + ImageMagick smoke. Memory-manager + transupp + several decode markers still absent. |
| Stock-tool parity | byte-exact | `examples/stock_djpeg_cjpeg/build.sh` produces byte-identical output to upstream on `references/libjpeg-turbo/testimages/*.jpg` |
| `tjunittest` | 100% | 1012 subtests pass on our cdylib |
| SIMD parity | bit-exact | NEON / AVX2 / SSE2 / WASM × 5 kernels, scalar↔SIMD bit-exact under 1000-iter PRNG fuzz |

**Net read**: production-shaped for *encode/decode*; not yet a drop-in for `jpegtran -rotate/-flip/-transpose/-crop` or any consumer that hands us virtual coefficient arrays produced by libjpeg's memory manager.

---

## 2. Last-Mile Gap Inventory

Each gap below is a real functional or compatibility gap (NOT a Rust-idiom replacement of a C surface). Items are P0/P1/P2 by user-facing impact.

### 2.1 P0 — Blocks expected workflows

#### G1. `jpegtran` transform path (rotate / flip / transpose / transverse / crop)

- **Symptom**: linking stock `jpegtran` against our cdylib produces `undefined reference to jcopy_markers_setup` / `jpeg_alloc_huff_table` / `jtransform_request_workspace` etc. `-rotate 90`, `-flip horizontal`, `-transpose`, `-transverse`, `-crop WxH+X+Y` therefore fail at link time.
- **Root cause**: `transupp.c` allocates a **destination** virtual coefficient array on `dstinfo->mem`. Our shim has no `jpeg_memory_mgr` table populated on cinfo, no virtual barray model, and no `jcopy_markers_*` helpers.
- **Missing surface** (all need `#[no_mangle] pub extern "C"`):
  - **Memory manager**: `cinfo->mem` populated with `alloc_small`, `alloc_large`, `alloc_sarray`, `alloc_barray`, `request_virt_sarray`, `request_virt_barray`, `realize_virt_arrays`, `access_virt_sarray`, `access_virt_barray`, `free_pool`, `self_destruct`. Plus `jpeg_alloc_huff_table`, `jpeg_alloc_quant_table` constructors.
  - **Backing store hooks** (in-RAM is fine): `jpeg_get_small`, `jpeg_get_large`, `jpeg_free_small`, `jpeg_free_large`, `jpeg_open_backing_store`, `jpeg_mem_init`, `jpeg_mem_term`, `jpeg_mem_available`.
  - **transupp helpers**: `jcopy_markers_setup`, `jcopy_markers_execute`, `jtransform_request_workspace`, `jtransform_adjust_parameters`, `jtransform_execute_transform`, `jtransform_perfect_transform`, `jtransform_parse_crop_spec`.
  - **Foreign coefficient handle recognition**: `jpeg_write_coefficients` (in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:4436`) currently rejects any handle that did not originate from our `jpeg_read_coefficients`. After the memory manager lands, `transupp`-allocated virtual barrays must materialize into our `JpegCoefficients` struct before `run_coefficient_writer_and_flush` runs.
- **Existing scaffolding**: `crates/libjpeg-turbo-rs-capi/src/memmgr.rs` already mirrors the `jpeg_memory_mgr` ABI byte-exact but is internal-only — no exported entry points yet.
- **Effort**: ~14–22 h.
  - Memory manager API (~6–10 h) — pool allocator, virtual array model with row-band access pattern.
  - Backing store + small entry points (~2–4 h).
  - `jcopy_markers_setup`/`jcopy_markers_execute` (~1–2 h) — wrap the existing `jpeg_save_markers` accumulator + `jpeg_write_marker` walker.
  - Foreign-handle recognition in `jpeg_write_coefficients` (~2–3 h).
  - Tests (~3 h) — extend `examples/stock_djpeg_cjpeg/build.sh` with `jpegtran`, byte-compare against upstream on `testorig.jpg` / `testimgari.jpg`, run with `-copy all` / `-copy comments` / `-copy none` / `-copy icc`.
- **Validation**:
  ```bash
  bash examples/stock_djpeg_cjpeg/build.sh
  $OUT/jpegtran -rotate 90      testimages/testorig.jpg > /tmp/our.jpg
  upstream-jpegtran -rotate 90  testimages/testorig.jpg > /tmp/upstream.jpg
  cmp /tmp/our.jpg /tmp/upstream.jpg
  $OUT/jpegtran -copy all input_with_exif.jpg > /tmp/copied.jpg
  exiftool /tmp/copied.jpg | grep EXIF  # markers preserved
  ```
- **Risk**: pool-based memory model needs to be ABI-correct (caller-side `request_virt_*` / `realize_virt_arrays` / `access_virt_*` walk); 12-bit/16-bit precision branches mirror the same pattern; multi-chunk markers (>65 K) need `jpeg_write_marker_chunks`-equivalent handling.

#### G2. `write_coefficients_progressive_arithmetic` cannot emit restart markers

- **Symptom**: caller does `jpegtran -progressive -arithmetic -restart 2B input.jpg`. The shim drops the restart (visible via `cinfo->err->num_warnings` increment + `RS_JWRN_PROG_ARITH_RESTART_DROPPED` since commit `0147a8e`), but the output stream lacks RST markers — not byte-equivalent to upstream.
- **Root cause**: `src/api/coefficient.rs:2080-2082` (the `write_coefficients_progressive_arithmetic` function in the lib crate) explicitly returns `Err` when `coeffs.restart_interval > 0`. Baseline arithmetic + restart was added in commit `98bb17a` (`ArithEncoder::emit_restart` mirrors `jcarith.c::emit_restart`), but the progressive path didn't get the same wiring.
- **Fix shape**: thread per-MCU-group `arith_enc.emit_restart(idx)` + `prev_dc` reset into the progressive scan loop, exactly the way baseline arithmetic does. DRI segment is already emitted at line 2051 when `restart_interval > 0`.
- **Effort**: ~3–5 h. Mostly mechanical — copy the baseline scaffolding, verify DC-band vs AC-band scan ordering, regression-test against `jpegtran -arithmetic -progressive -restart N` against upstream byte-for-byte.
- **Validation**:
  ```bash
  cjpeg -arithmetic -progressive -restart 2B sample.ppm > /tmp/upstream.jpg
  # decode + re-encode through our shim with same flags via tjpegtran
  cmp /tmp/our.jpg /tmp/upstream.jpg
  ```
- **Closes**: the `last_error` / `RS_JWRN_*` warning channel introduced in `b0e99cf` becomes a no-op (warning never fires).

### 2.2 P1 — Compatibility holes for legacy callers

#### G3. Legacy `tjLoadImage` / `tjSaveImage` still stubbed

- **Symptom**: callers that link against the legacy TJ1/TJ2 ABI (still common in older Pillow / GraphicsMagick / mozilla `tools/`) and call `tjLoadImage`/`tjSaveImage` get an error and a NULL-equivalent return — even though we already implement the TJ3 form.
- **Locations**:
  - `crates/libjpeg-turbo-rs-capi/src/legacy.rs:433` — `tjLoadImage: not yet implemented`
  - `crates/libjpeg-turbo-rs-capi/src/legacy.rs:452` — `tjSaveImage: not yet implemented`
  - `crates/libjpeg-turbo-rs-capi/src/legacy.rs:15` (also `tjEncodeYUV3`, `tjDecodeYUV` per legacy alias comment).
- **Fix shape**: thin wrappers delegating to the existing `tj3LoadImage8` / `tj3SaveImage8`. Map the legacy `flags` int to the corresponding TJ3 params (`TJPARAM_BOTTOMUP`, etc.) via the table in `libjpeg-turbo/src/turbojpeg.c::tj1{Load,Save}Image`.
- **Effort**: ~2–3 h.
- **Validation**: existing legacy alias dlopen tests in `tests/legacy_aliases.rs` — extend to call the formerly-stubbed entry points end-to-end.

#### G4. PNG support in image I/O

- **Symptom**: `tj3LoadImage8(handle, "x.png", ...)` / `tj3SaveImage8(...)` fail with the "format not recognized" path. Stock `djpeg -png`, `cjpeg` from a `.png` source — both unavailable.
- **Locations**:
  - `crates/libjpeg-turbo-rs-capi/src/legacy.rs` (no `png` / `RDPNG` / `WRPNG` references — confirms missing).
  - The Rust lib side in `src/image_io.rs` likewise covers BMP/PPM/PGM only.
- **C parity stance**: PNG is conditional on the `PNG_SUPPORTED` build flag in upstream — not part of the *required* core. Marked 🔶 in `docs/C_API_REFERENCE.md:158-159`.
- **Fix shape**: gate behind a `png` feature; depend on `png` crate; add `RDPNG`/`WRPNG`-equivalent encoders/decoders honoring the same TJPF set the C path supports. Keep default off so we don't bloat the codec-only crate.
- **Effort**: ~6–8 h (encode + decode + flag plumbing + tests).
- **Validation**: round-trip a `testimages/testorig.png` if upstream ships one, otherwise cross-validate via `image` crate.
- **Note**: this is the only documented BMP/PPM/PGM/PNG triplet gap; BMP/PPM/PGM are at parity.

### 2.3 P1 — Decode-path classic jpeglib.h gaps

#### G5. Classic ABI symbols still missing for full Pillow/ImageMagick depth

Pillow's `_imagingjpeg` and ImageMagick's `coders/jpeg.c` use symbols beyond what we ship today. Cross-referencing what `jpeg_*` symbols upstream `libjpeg.62.dylib` exports vs our 46 `#[no_mangle]` list:

- **Missing decode-side**:
  - `jpeg_consume_input` — required by Pillow's draft mode and progressive multi-pass decoders.
  - `jpeg_input_complete` — paired with `jpeg_consume_input` for streaming.
  - `jpeg_has_multiple_scans` — Pillow uses this to decide buffered-image mode.
  - `jpeg_start_output` / `jpeg_finish_output` — required for buffered-image multi-pass output.
  - `jpeg_new_colormap` — quantize-mode colormap update.
  - `jpeg_abort_decompress` / `jpeg_abort_compress` / `jpeg_abort` / `jpeg_destroy` — error-path teardown that wrappers expect.
- **Missing encode-side**:
  - `jpeg_set_linear_quality` — alternate quality scaling the C tools accept via `-baseline` / linear quality int.
  - `jpeg_default_qtables` extension paths.
  - `jpeg_write_raw_data` / `jpeg_read_raw_data` — raw planar (YCbCr 4:2:0/4:2:2/...) APIs that ImageMagick's "interlaced raw" path uses.
- **Effort**: ~8–12 h, partitioned across the two halves; mostly thin shims over our existing `Decoder` / `Encoder` trait surfaces.
- **Validation**: Pillow `_imagingjpeg` smoke (`examples/pillow_smoke/`) currently `#[ignore]`d in `tests/capi_pillow_compat.rs` — un-ignore and pass.

### 2.4 Items previously listed here that are NOT gaps

The following were initially flagged but verified already present on `main`. Listed for the audit trail so they don't get re-opened:

- **`Encoder::icc_profile`** — already public at `src/api/encoder.rs:208`. The 🔶 in `docs/C_API_REFERENCE.md` row for `jpeg_write_icc_profile` refers to the *legacy `j_compress_ptr` shape* not being a public Rust idiom; the equivalent capability is shipped.
- **`Encoder::density(unit, x, y)`** — already public at `src/api/encoder.rs:317`. The 🔶 in `docs/FEATURE_PARITY.md` (DPI/density row) was about JFIF density write semantics, which the encoder already covers.
- **`TJPARAM_SAVEMARKERS` through `TjHandle`** — wired end-to-end at `src/api/tj3.rs:269` (set), `:303` (get), `:645-648` (decode behavior dispatch). The 🔶 on this row in `docs/FEATURE_PARITY.md` reflects the historical name; the value is now honored.
- **`TJPARAM_PRECISION` encode dispatch** — upstream `tj3.h` documents this param as read-only; encode precision is selected by which `tj3CompressN` entry point the caller invokes (`tj3Compress8` / `tj3Compress12` / `tj3Compress16`). Our shim mirrors that contract. The 🔶 marker is strictly about the TJ3 ↔ Rust public-API surface mismatch (`compress_8bit()` vs `compress_12bit()` separate functions), not a missing capability.

The 🔶 markers in `docs/FEATURE_PARITY.md` / `docs/C_API_REFERENCE.md` for these rows could be promoted to ✅ in a separate doc-only commit; they are not last-mile blockers.

---

## 3. Out of Scope (Explicit Rust-idiom replacements)

These are 🔶 in the docs but **NOT gaps** — the capability is present via a different Rust surface. Listed so future readers don't re-open them.

- `tj3Alloc` / `tj3Free` — replaced by Rust `Vec` ownership / `Drop`. C ABI shim still exposes `tj3Alloc`/`tj3Free` for FFI callers.
- `tj3GetErrorStr` / `tj3GetErrorCode` — replaced by `Result<T, JpegError>`. C ABI shim still exposes the per-handle getters for FFI callers.
- `jpeg_error_mgr` `format_message` / `output_message` / 5-callback layout — fully implemented in the C ABI shim (`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:629-784`, post `f66aab8`/`b0e99cf`/`0147a8e`). The Rust public-API `ErrorHandler` trait carries 3 callbacks because the others are Rust-idiom no-ops.

---

## 4. Validation Matrix

When closing each gap, confirm against this matrix:

| Workflow | Today | After last-mile | Test driver |
| --- | --- | --- | --- |
| stock `djpeg` decode → byte-exact PPM | ✅ | unchanged | `examples/stock_djpeg_cjpeg/build.sh` |
| stock `cjpeg` encode → byte-exact JPEG | ✅ | unchanged | same |
| stock `jpegtran` transcode (`-progressive`, `-arithmetic`, `-restart N`, `-optimize`) | ✅ (since `74ed710`) | unchanged | `tests/capi_jpeglib_write_coefficients.rs` |
| stock `jpegtran -rotate/-flip/-transpose/-crop` | ❌ link error | ✅ byte-exact | new test in `examples/stock_djpeg_cjpeg/` |
| stock `jpegtran -copy all/comments/icc/none` | ❌ link error | ✅ markers preserved | exiftool diff |
| `jpegtran -arithmetic -progressive -restart N` | ⚠ restart dropped, warning surfaced | ✅ byte-exact | new arith-progressive-restart test |
| Pillow `_imagingjpeg` import + decode | partial | ✅ green | `tests/capi_pillow_compat.rs` (un-ignore) |
| ImageMagick `coders/jpeg.c` import + decode | partial | ✅ green | `tests/capi_imagemagick_compat.rs` (un-ignore) |
| `tjunittest` | ✅ 1012/1012 | unchanged | `tests/tjunittest_link.rs` |
| `examples/cjpeg-test.sh` (libjpeg-turbo upstream test rig) | ❓ not yet wired | should pass | new harness |

---

## 5. Suggested Roadmap

Ordered by ratio of impact / effort. Each step ends with a green commit + codex review pass + push.

1. **G2 — progressive arithmetic + restart** (3–5 h): unblocks one full upstream feature, removes the warning suppression scaffold, mostly mechanical.
2. **G3 — `tjLoadImage` / `tjSaveImage` legacy shim** (2–3 h): closes a stub that legacy wrappers actually call.
3. **G5 — classic decode-side `jpeg_consume_input` family + `jpeg_abort_*` + raw-data API** (8–12 h): unlocks Pillow / ImageMagick smoke-test green.
4. **G1 — `jpegtran` transform path** (14–22 h): the single biggest remaining gap. Memory manager + transupp + handle-recognition + tests.
5. **G4 — PNG support** (6–8 h, gated behind a feature): truly optional given upstream parity stance.

**Total last-mile budget**: ~33–50 hours of focused work. The first three steps (~13–20 h) make our cdylib a drop-in for everything except `jpegtran` transform options.

A separate small doc-only commit can promote the 🔶 markers covered in §2.4 to ✅ in `docs/FEATURE_PARITY.md` / `docs/C_API_REFERENCE.md` (no code change needed).

---

## 6. Definition of Done

A gap is "closed" when **all** of these are true:

1. **Canonical state updated first**: the matching row in `docs/FEATURE_PARITY.md` flips to `[x]` (no qualifier) AND the matching row in `docs/C_API_REFERENCE.md` flips to ✅. These two docs remain the source of truth — this `LAST_MILE.md` is updated *afterwards* (or not at all if the gap simply disappears from the queue).
2. Validation entry in §4 above shows green.
3. New regression test exists in `crates/libjpeg-turbo-rs-capi/tests/` (or the relevant lib test dir) covering the previously-broken caller path.
4. `cargo fmt --all`, `cargo clippy --lib -- -D warnings`, `cargo test` all green.
5. `codex review --commit <SHA>` returns "no introduced correctness issues" (or any P2 it finds is fixed and re-reviewed).
6. Stop-hook does not flag a follow-up issue for the same area.

---

## 7. Pointers

- Detailed transform-path design notes (memory manager pool model, virtual barray access semantics): `docs/NEXT_SESSION_PLAN.md` § 작업 1.5.
- Codex review history for the warning channel chain: commits `74ed710` → `b1db34c` → `f66aab8` → `b0e99cf` → `0147a8e`.
- C reference: `references/libjpeg-turbo/src/transupp.c` (transform), `jmemmgr.c` (memory manager), `jdmarker.c` (marker save/copy), `jcarith.c::emit_restart` (arithmetic restart pattern to mirror).
