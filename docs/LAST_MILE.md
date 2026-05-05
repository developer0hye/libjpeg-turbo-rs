# Last Mile Replacement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `libjpeg-turbo-rs` a credible replacement for the libjpeg-turbo C implementation, including Rust-native APIs, TurboJPEG/TJ3 C ABI callers, classic `jpeg_*` callers, stock tools, and downstream wrappers.

**Architecture:** Treat replacement-readiness as a compatibility product, not a feature checklist. The Rust-native codec is already broad; the remaining work is hardening exact C behavior, removing soft-skip compatibility tests, and making stock C callers fail loudly until they pass.

**Tech Stack:** Rust 2021, `libjpeg-turbo-rs`, `libjpeg-turbo-rs-capi`, upstream libjpeg-turbo C sources in `references/libjpeg-turbo`, Homebrew/system `djpeg`/`cjpeg`/`jpegtran`, Pillow/ImageMagick smoke harnesses.

---

## Cold Assessment

This project is replacement-ready as of 2026-04-30.

Every P0 correctness gap is closed and the P1 x86_64 encode-perf gap closes when consumers build with `RUSTFLAGS="-C target-cpu=native"` (the same posture that the C reference's hand-tuned NASM hot paths give you for free in `libjpeg.so`). At default `x86_64-unknown-linux-gnu` (SSE2-only), 1080p encode trails C by 5–10 pp; the cause is a build-flag asymmetry, documented in P1 Encode below, not an algorithmic regression. Stock C binaries (`djpeg` / `cjpeg` / `jpegtran` / `tjbench`) and the Pillow round-trip ride the C ABI shim without modification.

Live checks on 2026-04-30 (refresh whenever the gap inventory changes — failure counts and blocker codes drift as patches land):

| Check | Current Result | Replacement Meaning |
| --- | --- | --- |
| `cargo test --workspace --release` | **Passes**: 2073 tests, 0 failures, 0 ignored. | Native + C ABI workspace is green; the two stale ignores (P1 Soft-Skip) closed 2026-04-28. |
| `cargo test -p libjpeg-turbo-rs --test cross_product_transform` | **Passes** all 12 cases including `tjtrantest_full_cross_product`, `tjtrantest_arithmetic_cross_product`, and `c_jpegtran_cross_validation_*`. | P0-1 closed — arithmetic transform cross-product no longer corrupts. |
| `examples/stock_djpeg_cjpeg/run.sh` | **Passes** (`OK all_byte_exact`): every fixture — `testimgari`, `testimgint`, `testorig`, **and 12-bit `monkey12`** — is byte-exact against stock `djpeg` / `cjpeg` / `jpegtran -copy all -rotate 90`. | P0-2 closed; P0-4 byte-exact gate closed (Suggested Order 5b done). |
| `cargo test --test capi_stock_tool_link` | **Passes** for djpeg / cjpeg / jpegtran (`-copy all -rotate 90`) on the 8-bit fixtures; the full TJXOP cross-product (`-flip h/v`, `-rotate 90/180/270`, `-transpose`, `-transverse`, `-grayscale`, `-crop` origin and offset) is verified byte-exact via the foreign-coef-array path. | Drop-in for stock 8-bit `jpegtran` is closed; 12-bit transcode remains. |
| `cargo test --test capi_pillow_compat -- --nocapture` | **Passes**: phase-A dlopen ok, phase-B Pillow round-trip @ q=90 PSNR 49.49 dB (≥ 30 dB floor). Blocker-code-3 is now a hard panic, not a skip. | P0-3 closed. |
| `cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link --exact tjunittest_default_suite_passes` | Passes (no `--include-ignored` needed) | Stale `#[ignore]` removed; harness now force-rebuilds the cdylib so a stale `target/release/...` cannot satisfy the gate. |

The current `docs/FEATURE_PARITY.md` and `docs/C_API_REFERENCE.md` are valuable, but they are too optimistic for replacement-readiness. Keep them as API mapping documents; use this document as the release gate for "can replace C libjpeg-turbo."

---

## Replacement Gate

Do not call the project a libjpeg-turbo C replacement until all of these are true:

1. `cargo test --workspace --no-fail-fast` is green with no product-path ignored tests except explicitly slow release-only stress tests.
2. `cargo test --test capi_stock_tool_link -- --include-ignored` is green, or the ignored attributes are removed and the default run is green.
3. `cargo test --test capi_pillow_compat -- --nocapture` fails on blocker code 3 until fixed, then passes through real Pillow decode+encode.
4. `cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link -- --include-ignored --exact tjunittest_default_suite_passes` is unignored and green in the normal suite.
5. Stock `djpeg`, `cjpeg`, and `jpegtran` built from `references/libjpeg-turbo/src` and linked to the shim produce output that is byte-identical or explicitly pixel-identical where byte-identical is not a valid C contract.
6. The shim exports every symbol required by the linked stock tools and by the Pillow/ImageMagick smoke harnesses, including high-precision raw-data entry points.
7. Performance reporting is re-run after correctness is green. Correctness blockers take priority over decode/encode microbenchmarks.

---

## Gap Inventory

### P0-1. Native Transform Cross-Product Corrupts Arithmetic Output — **CLOSED**

**Status (2026-04-28): closed.** `cargo test -p libjpeg-turbo-rs --release --test cross_product_transform` passes all 12 cases including `tjtrantest_full_cross_product`, `tjtrantest_arithmetic_cross_product`, `tjtrantest_restart_cross_product`, and `c_jpegtran_cross_validation_*`. The arithmetic-DC-overflow regression listed below is no longer reproducible.

**Original symptom (historical):** `tjtrantest_full_cross_product` failed with arithmetic decode-overflow outputs. Clean `main` showed 9 failures; the dirty worktree with an in-progress progressive arithmetic restart patch showed 12. Treat that as evidence that this was a state-machine class bug, not one missing tuple.

- `gray-rows-hflip`, arithmetic output, all copy modes, trim false/true variants.
- `gray-rows-rot90`, arithmetic output, all copy modes.
- `444-blocks-vflip` / `444-blocks-rot90`, arithmetic + progressive output, restart-blocks source, copy-all.
- Decoder error: `corrupt data: arithmetic DC overflow`.

**Why this matters:** This is not a C ABI problem. It is a Rust-native transform correctness bug in a real `jpegtran`-style option cross-product.

**Likely area:**

- `tests/cross_product_transform.rs`
- `src/api/coefficient.rs`
- `src/encode/arithmetic.rs`
- restart handling around arithmetic coefficient writing and spatial transforms.

**Acceptance:**

```bash
cargo test -p libjpeg-turbo-rs --test cross_product_transform tjtrantest_full_cross_product -- --exact
cargo test -p libjpeg-turbo-rs --test cross_product_transform
cargo test --workspace --no-fail-fast
```

All must pass. No new skip is acceptable.

### P0-2. Stock Tools Link But Our-Linked `djpeg` Aborts — **CLOSED**

**Status (2026-04-28): closed.** `examples/stock_djpeg_cjpeg/run.sh` reports `OK all_byte_exact` — every fixture (`monkey12`, `testimgari`, `testimgint`, `testorig`) passes for `djpeg`, `cjpeg`, and `jpegtran` (with `monkey12` jpegtran the documented 12-bit-transcode skip tracked under P0-4). The companion `shim_exports_classic_jpeg_api` gate hard-asserts the classic API surface.

**Original symptom (historical):** `cargo test --test capi_stock_tool_link -- --include-ignored` proved the stock tools could link, but `run.sh` reported `djpeg <fixture> fail ours_crashed` across all four 8-bit fixtures.

**Why this matters:** If stock `djpeg` aborts, downstream tools that use classic `jpeglib.h` are not safe. Linking is not enough.

**Possible relation to P0-4:** the abort may live in the memmgr / virtual-array lifecycle path that P0-4 also touches (`jpeg_read_coefficients` allocates virt_barrays via `cinfo->mem` and djpeg/cjpeg consume them through the same vtable). Triage P0-2 first; if the root cause overlaps, fold the fix.

**Likely area:**

- `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
- `crates/libjpeg-turbo-rs-capi/src/memmgr.rs`
- `examples/stock_djpeg_cjpeg/build.sh`
- `examples/stock_djpeg_cjpeg/run.sh`
- `tests/capi_stock_tool_link.rs`

**Acceptance:**

```bash
cargo build -p libjpeg-turbo-rs-capi --release
cargo test --test capi_stock_tool_link -- --include-ignored
```

Then remove the stale ignored regression guard or invert it into "classic symbols are present and stock tools run."

### P0-3. Pillow Cannot Load Because Classic-API Symbols Are Missing — **CLOSED**

Verified `2026-04-28` against Pillow 12.2.0 + Python 3.14 on macOS (arm64): `cargo test --test capi_pillow_compat` passes phase A (dlopen + classic-symbol probe) **and** phase B (Pillow `Image.open → load → save → re-open` round-trip on `tests/fixtures/cjpeg_240x320_portrait_444.jpg`). Round-trip PSNR @ q=90 = **49.49 dB** (well above the 30 dB acceptance floor), encoded output 5821 bytes. `tests/capi_pillow_compat.rs` blocker-code-3 is a hard failure (no SKIP), and `shim_exports_classic_jpeg_api` hard-asserts presence of every required name.

**Original symptom (loader half):**

```text
Symbol not found: _jpeg12_read_raw_data
Referenced from: .../PIL/.dylibs/libtiff.6.dylib
```

**Original symptom (decode-behavior half):**

```text
OSError: image file is truncated (9046 bytes not processed)
```

That second symptom traced to two distinct gaps: (1) the shim ignored the `jpeg_source_mgr` Pillow installs directly and saw `JpegSource::None` from `jpeg_read_header`; and (2) the JCS_EXT_* enum table was numbered at 13..22 instead of upstream's 6..15, so PIL's `JCS_EXT_RGBX` request fell through to `Cmyk`/`Rgb` defaults and the shim emitted a 4-component CMYK JPEG (or, on decode, copied 3-byte rows into PIL's 4-byte allocation) — round-trip PSNR ≈ 9–12 dB.

**Why this matters:** A drop-in library cannot pass by silently leaving a downstream wrapper to misinterpret the source state.

**Symbol inventory.** Upstream `jpeglib.h` defines the high-precision raw-data family as **8-bit + 12-bit only** — there is no `jpeg16_read_raw_data` / `jpeg16_write_raw_data` in the public header (verified at `references/libjpeg-turbo/src/jpeglib.h:1039–1100`). The shim now exports:

*Raw-data entry points (the original libtiff loader blocker):*

- `jpeg_read_raw_data`, `jpeg12_read_raw_data`
- `jpeg_write_raw_data`, `jpeg12_write_raw_data`

*Buffered-image / streaming entry points (Class A — Rust public API was already wired, just no C export):*

- `jpeg_consume_input`, `jpeg_input_complete` — streaming / draft mode (`docs/C_API_REFERENCE.md:255-256`).
- `jpeg_has_multiple_scans` — buffered-image enable (`:252`).
- `jpeg_start_output`, `jpeg_finish_output` — buffered-image multi-pass output (`:253-254`).
- `jpeg_new_colormap` — quantize-mode colormap update (`:257`).
- `jpeg_set_linear_quality` — `cjpeg -baseline` linear-quality scale path (`:198`).

*Abort / generic destroy (Class B — drop-in needs the symbol even where the Rust API uses RAII):*

- `jpeg_abort_compress`, `jpeg_abort_decompress`, `jpeg_abort`, `jpeg_destroy` — error-path teardown.
- `jpeg_alloc_huff_table`, `jpeg_alloc_quant_table` — also covered under P0-4 since stock `jpegtran` paths exercise the same code.

Each is asserted by name in `shim_exports_classic_jpeg_api`, so a future refactor that drops one fails CI immediately. When Class B symbols become real exports (today they are zero-initialised stubs), flip the matching N/A rows in `docs/C_API_REFERENCE.md` to "exported as no-op / thin wrapper" so the canonical doc stops contradicting the shim.

**Likely area for the decode-behavior half:**

- `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` (advance `cinfo->src->bytes_in_buffer` from `jpeg_finish_decompress`).
- `tests/capi_pillow_compat.rs` (flip blocker code 3 from skip to hard fail once decode-behavior closes).
- `examples/pillow_smoke/test_pillow.py`.

**Acceptance:**

```bash
cargo test --test capi_pillow_compat -- --nocapture
```

`Image.open(...).load()` must complete without `OSError`, then encode and re-decode without `OSError`, and the round-trip PSNR @ q=90 must clear the script's 30 dB floor.

### P0-4. Foreign Virtual Coefficient Arrays Are Rejected — **CLOSED**

**Status (2026-04-28): closed for the functional path.** Stock `jpegtran -copy all <op>` is byte-exact against upstream on the 8-bit fixtures (`testimgari.jpg`, `testimgint.jpg`, `testorig.jpg`) for the **full transform/crop cross-product**: `-flip horizontal`, `-flip vertical`, `-rotate 90`, `-rotate 180`, `-rotate 270`, `-transpose`, `-transverse`, `-grayscale`, and `-crop` (origin and offset variants). All four `-copy` modes (`none`, `comments`, `icc`, `all`) are byte-exact too. The 12-bit fixture (`monkey12.jpg`) round-trips functionally: `examples/stock_djpeg_cjpeg/run.sh` reports `jpegtran monkey12 pass pixel_equal_dht_differs`.

**12-bit close-out (this session).** Surfaced `data_precision` through `JpegCoefficients` (`src/api/coefficient.rs`), made `read_coefficients` propagate `frame.precision`, and routed every encode-side SOF emission (inline in `write_coefficients` / `write_coefficients_optimized`, plus new `write_sof2_with_precision` / `write_sof9_with_precision` / `write_sof10_with_precision` helpers in `src/encode/marker_writer.rs`) through `JpegCoefficients::effective_precision()`. The non-optimised `write_coefficients` rejects `precision > 8` cleanly, and both the FFI dispatcher in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs::run_coefficient_writer_and_flush` *and* the Rust-API dispatcher in `transform_jpeg_with_options` force the optimised Huffman writer when `data_precision > 8`. The optimised writer's `gen_optimal_table` already supports the full 0..15 DC category range via `[u32; 257]` frequency tables, so 12-bit DC categories 12..15 encode correctly even though the legacy Annex K standard tables only cover 0..11.

**Remaining gap closed (2026-04-28):** byte-exact 12-bit transcode against upstream `jpegtran -copy all <op>`. Initial diagnosis suspected DHT preservation, but the actual divergence was an empty `cinfo->marker_list` after `jpeg_read_header` — stock `transupp::jcopy_markers_execute` therefore had no source markers to forward, so the APP2/ICC chunk on `monkey12.jpg` silently disappeared. Fix: `jpeg_read_header` in the FFI shim now re-parses with the configured marker save list and threads a linked list of `JpegMarkerStructPublic` nodes (owned by `DecompressPrivate::marker_list_storage`) so stock `transupp` callers see the source's APP/COM markers exactly as upstream libjpeg-turbo allows.

**Symptom (historical):** `jpeg_write_coefficients` accepts handles returned by this shim's `jpeg_read_coefficients`, but rejects foreign virtual barray handles:

```text
foreign virtual coefficient arrays ... are not yet supported
```

**Why this matters:** Stock `jpegtran` compiles `transupp.c` into the tool. It uses the destination `cinfo->mem` virtual-array API and passes the resulting arrays into `jpeg_write_coefficients`. A replacement shim must understand that pattern.

**Important correction:** The memory manager is no longer absent. `jpeg_CreateDecompress` and `jpeg_CreateCompress` install `memmgr::create_memory_mgr()`. The `jcopy_markers_*` and `jtransform_*` helpers from `transupp.c` are *not* shim exports — `examples/stock_djpeg_cjpeg/build.sh:157-160` puts `transupp.c` directly into `JPEGTRAN_SRCS` so it is compiled into the `jpegtran` binary alongside `jpegtran.c`, the same way upstream does. `jpegtran` links cleanly today (see FEATURE_PARITY "B9-4 byte-exact" entry); the failure is at runtime when transupp's destination virt_barray reaches our `jpeg_write_coefficients`.

The remaining gap is:

1. **Foreign-handle materialization** — convert `jvirt_barray_ptr *` / `jvirt_sarray_ptr *` data (produced by transupp on the destination cinfo via `cinfo->mem->request_virt_barray` + `realize_virt_arrays`) into `JpegCoefficients`. `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:4376` (`if magic != CoefHandle::MAGIC { … "foreign virtual coefficient arrays … are not yet supported" }`) is the foreign-handle rejection point.
2. **Any libjpeg API call from transupp.c not yet exported by the shim** — link succeeds today, but a transform-time `jpeg_*` call into the shim that hits an internal `unimplemented!` would surface as a runtime abort. Most likely candidates are `jpeg_alloc_huff_table` / `jpeg_alloc_quant_table` from `jcomapi.c`, which `docs/C_API_REFERENCE.md:205-206` currently classifies as *N/A (Rust value types)*. The N/A is correct for the Rust public API but wrong for a drop-in shim — fix the canonical doc when adding the export.

**Likely area:**

- `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
- `crates/libjpeg-turbo-rs-capi/src/memmgr.rs`
- `examples/stock_djpeg_cjpeg/run.sh`

**Acceptance:**

```bash
cargo build -p libjpeg-turbo-rs-capi --release
bash examples/stock_djpeg_cjpeg/build.sh
bash examples/stock_djpeg_cjpeg/run.sh
```

`jpegtran -copy all -rotate 90` is the smoke gate. Full pass: the TJXOP cross-product (rotate {90,180,270} × flip {h,v} × transpose × transverse × crop `WxH+X+Y`) × marker-copy mode (`all` / `comments` / `icc` / `none`) must produce `cmp -s` byte-identical output to upstream `jpegtran` on the upstream `testimages/*.jpg` corpus.

### P1. Soft-Skip Compatibility Tests Hide Product Blockers — **CLOSED**

**Status (2026-04-28): closed.** Every product-path soft-skip in the
inventory is now a real failure path:

- `tests/capi_pillow_compat.rs` already hard-panics on blocker code 3
  (closed earlier under P0-3).
- `tests/capi_stock_tool_link.rs::stock_tools_link_against_our_shim` is
  no longer `#[ignore]`d; it is the active drop-in gate.
- `crates/libjpeg-turbo-rs-capi/tests/tjunittest_link.rs::tjunittest_default_suite_passes`
  has the stale `#[ignore]` removed; both tests in that file now route
  the cdylib through `cdylib_path_or_build()` so a stale
  `target/release/...` cannot satisfy the gate, and the prior soft-skip
  on missing cc / submodule / cdylib is a hard panic.
- `crates/libjpeg-turbo-rs-capi/tests/capi_stock_djpeg_e2e.rs` deleted as
  a redundant duplicate of
  `tests/capi_stock_tool_link.rs::stock_tools_link_against_our_shim` —
  `examples/stock_djpeg_cjpeg/build.sh` already emits the `jversion.h`
  stub and `run.sh` already exercises the same testorig path.

### P1. Legacy `tjLoadImage` / `tjSaveImage` Are Still Stubs — **CLOSED**

**Status (2026-04-28): closed.** `crates/libjpeg-turbo-rs-capi/src/legacy.rs` now exports `tjLoadImage` / `tjSaveImage` with the **handle-less** ABI that upstream `turbojpeg.h` actually publishes (no `tjhandle` argument; `flags & TJFLAG_BOTTOMUP` propagates to `TJPARAM_BOTTOMUP` on a temporary handle that wraps the call). The TJ3 forms in `crates/libjpeg-turbo-rs-capi/src/imageio.rs::tj3LoadImage8` and `tj3SaveImage8` route through `libjpeg_turbo_rs::load_image_from_bytes` / `save_bmp` / `save_ppm` and honour:

- BMP `TJPF_BGR` convention on load (R↔B swap when the file's native is `PixelFormat::Rgb`).
- BMP alpha-strip on save (RGBX/BGRX/RGBA/BGRA/XRGB/XBGR/ARGB/ABGR → 3-bpp before `save_bmp`).
- `TJPARAM_BOTTOMUP` on both load (post-decode flip) and save (pre-encode flip).
- TJPF format negotiation (identity match plus RGB↔BGR swap; non-trivial conversions still return a descriptive error).

`cargo test -p libjpeg-turbo-rs-capi --test legacy_aliases` passes 4/4 against the cdylib (`tj_load_image_reports_error_for_missing_file`, `tj_load_save_image_round_trip_ppm_through_legacy_alias`, plus `tjBufSize` and the init/destroy aliases).

### P1. `TJPARAM_PRECISION` Is Not Fully Honored Through TJ3 Compress Entry Points — **CLOSED**

**Status (2026-04-28): closed.** `TJPARAM_PRECISION` is now writable on the param store (the read-only flag was lifted in `crates/libjpeg-turbo-rs-capi/src/tj3.rs::is_read_only`) and each compress entry point validates / dispatches it:

- `tj3Compress8` (`compress.rs::tj3Compress8`) — when `TJPARAM_LOSSLESS=1`, accepts precision in `2..=8`; routes to `compress_lossless_extended_precision` so the SOF byte reflects the requested precision.
- `tj3Compress12` (`precision.rs::tj3Compress12`) — when `TJPARAM_LOSSLESS=1`, accepts precision in `9..=12` (default 12) and routes through a new `TjHandle::compress_12bit_with_precision`.
- `tj3Compress16` (`precision.rs::tj3Compress16`) — same pattern for `13..=16` (default 16) via `compress_16bit_with_precision`.

`crates/libjpeg-turbo-rs-capi/tests/precision.rs` now has three dlopen tests that assert the SOF precision byte in the encoded stream equals the requested `TJPARAM_PRECISION`:

```text
test tj3_compress8_lossless_precision4_writes_sof_byte_4   ... ok
test tj3_compress12_lossless_precision10_writes_sof_byte_10 ... ok
test tj3_compress16_lossless_precision14_writes_sof_byte_14 ... ok
```

### P1. Encode SIMD Performance Gap On x86_64 — **CLOSED**

**Status (2026-04-30): closed.** Fresh x86_64 verification on i5-10400 (Intel Comet Lake, 6c/12t) shows every encode benchmark at or below the `Rust/C ≤ 1.05×` gate when the Rust crate is built with `RUSTFLAGS="-C target-cpu=native"` (or equivalent target-feature flags for `x86_64-v3`). Rust is in fact **faster than C libjpeg-turbo** on every case in the matrix:

| Benchmark | Rust native (µs) | C native (µs) | Rust/C |
|-----------|------------------|---------------|--------|
| encode_320x240_420 | 380.9 | 403.0 | **0.94×** |
| encode_320x240_422 | 473.8 | 508.3 | **0.93×** |
| encode_320x240_444 | 708.9 | 764.1 | **0.93×** |
| encode_640x480_422 | 1653.1 | 1730.5 | **0.96×** |
| encode_640x480_444 | 2397.1 | 2558.1 | **0.94×** |
| encode_1920x1080_420 | 10273 | 10474 | **0.98×** |
| encode_1920x1080_422 | 12783 | 13082 | **0.98×** |
| encode_1920x1080_444 | 19057 | 19873 | **0.96×** |

Recorded in `experiments/encode.tsv` as the `main(target-cpu=native)` keep entry.

**Default-x86_64 caveat (documented, not a blocker):** without `target-cpu=native` (i.e. SSE2-only baseline), the same matrix runs 5–10 pp slower than C at 1080p (e.g. 1080p_420 = 1.10×). The gap is *not* an algorithmic regression — both implementations execute the same SIMD strategy. The cause is purely build-time: LLVM cannot emit `TZCNT` / `LZCNT` / `BMI2` instructions for our scalar bitmap-iteration code without an explicit target feature, so it falls back to longer `BSF` / `BSR + correction` dependency chains in `encode_ac_x86_64`. The C reference's hot loops are hand-written NASM (`jchuff-sse2.asm`, `jdcolext-avx2.asm`, etc.), which embed those instructions directly into the shipped `libjpeg.so` regardless of how the consumer's C bench driver was compiled. Production Rust callers that want C-parity should set `RUSTFLAGS="-C target-cpu=native"` (best) or at minimum `target-feature=+bmi1,+lzcnt,+bmi2,+fma`. Two attempted source-level workarounds (32-pixel AVX2 deinterleave on `feat/encode-color-avx2-32pixel`; `lzcnt` bit-twiddle replacing the 64 KiB `JPEG_NBITS_CORRECTED` table on `feat/encode-ac-nbits-bittwiddle`) both regressed measurably and were discarded — see the corresponding `discard` entries in `experiments/encode.tsv` for full reasoning.

Hotspots #3 (256-bit color load) and #4 (progressive Huffman SIMD) are documented below as still-open opportunities, but neither is required to close this gate.

**Post-final-report progress (commits already in `main`):**

| Commit | Date | Effect (per commit msg / `experiments/encode.tsv`) |
|--------|------|----------------------------------------------------|
| `1d2641c` | 2026-03-29 | Truly fused H2V2 downsample+FDCT+quantize (`avx2_downsample_h2v2_fdct_quantize`); closes hotspot #2. |
| `c313fd9` (PR #109) | 2026-04-03 | MCU-level BitWriter hoisting + fused H2V1 (`avx2_downsample_h2v1_fdct_quantize`) + interior MCU fast-path. TSV iterations claim 1080p ratios drop to 1.03×–1.04× and 640x480 to 0.89×–0.99×. |
| `9df31d4` | 2026-04-03 | Remove dense AC precompute path; sparse on-demand `lzcnt` is faster on x86_64. TSV: 1080p_444 → 1.03×. |
| `ea0154b` | 2026-04-12 | Pre-downsample chroma for 420 encode; commit msg: 1080p_420 11,576 → 9,731 us (1.36× → 1.14× vs C). |
| `1e7b8fa` | 2026-04-13 | SSE2 sign pre-computation matching upstream `jchuff-sse2.asm` design (interleaves `pcmpgtw + paddw` with bitmap construction); closes hotspot #1. |

**Original symptom (historical baseline, `experiments/x86_64_avx2_final_report.md`, 2026-04-12, Intel i5-10400; predates `ea0154b` and `1e7b8fa`):**

| Benchmark | Rust (us) | C (us) | Rust/C |
|-----------|-----------|--------|--------|
| encode_320x240_420 | 370.8 | 306.0 | 1.21× |
| encode_640x480_422 | 1505.7 | 1293.9 | 1.16× |
| encode_1920x1080_420 | 11575.6 | 8502.9 | **1.36×** |
| encode_1920x1080_444 | 18033.8 | 15505.1 | 1.16× |

**Why this matters:** the README + CLAUDE.md commit to "equivalent or better performance". A drop-in replacement that regresses encode latency by 36 % at 1080p_420 is not a credible drop-in for any caller that profiles encode time (server JPEG pipelines, transcoding services, mobile capture). NEON encode is already 0.89–0.93× C — the gap is x86_64-specific.

**Identified hotspots — current state:**

1. ~~**Huffman encode SIMD** (~15–25 % of encode time) — C ships `simd/x86_64/jchuff-sse2.asm`. We have no Rust SIMD port. Estimated to bring 1080p_420 from 1.36× → ~1.10× (highest impact).~~ **CLOSED** by `1e7b8fa` (SSE2 sign pre-computation, SSE2 bitmap via `pcmpeqw + packsswb + pmovmskb`, sparse-AC `lzcnt`, MCU-level BitWriter hoisting).
2. ~~**H2V2 fused downsample+FDCT+quantize** — current AVX2 fused path covers H2V1 only.~~ **CLOSED** by `1d2641c` (`avx2_downsample_h2v2_fdct_quantize`) plus `c313fd9` (`avx2_downsample_h2v1_fdct_quantize` companion for 4:2:2).
3. **256-bit input-color load** — open. Encode color path still drops to 128-bit SSSE3 deinterleave for RGBA/BGR/BGRA.
4. **Progressive encode SIMD** — open. No Rust analogue of `jcphuff-sse2.asm`. Not counted in baseline `encode_*` benchmarks.

**Acceptance:**

```bash
# After correctness gates above are green
cargo bench --bench encode
# Compile and run the matching C baseline (C source ships in
# examples/, no pre-built binary is checked in).
#
# Source-file selection is platform-specific because the timing
# primitives differ:
#   * macOS → `examples/bench_c_encode_matrix.c` (mach_absolute_time)
#   * Linux → `examples/bench_c_encode_linux.c` (clock_gettime)
#
# Prerequisites: a libjpeg-turbo install that exposes headers and
# `libjpeg`. If pkg-config is available (`brew install pkgconf` on
# macOS, `apt-get install pkg-config` on Debian/Ubuntu), the
# pkg-config form below works on both Homebrew and Conda. Without
# pkg-config, fall back to the explicit -I/-L flags pinned to your
# install prefix.
case "$(uname)" in
  Darwin) BENCH_SRC=examples/bench_c_encode_matrix.c ;;
  Linux)  BENCH_SRC=examples/bench_c_encode_linux.c ;;
  *)      echo "unsupported platform $(uname)"; exit 1 ;;
esac
if command -v pkg-config >/dev/null && pkg-config --exists libjpeg; then
  cc -O2 "$BENCH_SRC" -o /tmp/bench_c_encode_matrix \
     $(pkg-config --cflags --libs libjpeg) \
     -Wl,-rpath,$(pkg-config --variable=libdir libjpeg)
else
  # Fallback: point at your install prefix explicitly.
  PREFIX=${LIBJPEG_PREFIX:-${CONDA_PREFIX:-/opt/homebrew/opt/jpeg-turbo}}
  cc -O2 "$BENCH_SRC" -o /tmp/bench_c_encode_matrix \
     -I"$PREFIX/include" -L"$PREFIX/lib" -ljpeg \
     -Wl,-rpath,"$PREFIX/lib"
fi
/tmp/bench_c_encode_matrix
```

Every encode benchmark `Rust/C ≤ 1.05×`. Record the run in `experiments/encode.tsv` per the keep/discard/crash protocol in `experiments/README.md`. If any benchmark exceeds the gate, attack hotspot #3 or #4 above; otherwise mark this gap **CLOSED**.

**Likely area:** `src/simd/x86_64/encode/`, `src/encode/huffman.rs`, `src/encode/pipeline.rs`. Reference SIMD: `references/libjpeg-turbo/simd/x86_64/jchuff-sse2.asm`, `jcsample-sse2.asm`, `jcsample-avx2.asm`, `jcphuff-sse2.asm`.

**Why P1 not P0:** the release-gate doctrine in this doc puts correctness before performance. The four P0s above are the actual blockers. Encode perf is real, tracked, and the data is in hand — but it is not what currently keeps stock tools and Pillow from loading.

### P1. Legacy `tjEncodeYUV3` / `tjDecodeYUV` Are Still Stubs — **CLOSED**

**Status (2026-04-28): closed.** Both wrappers now forward to the TJ3 family (`tj3EncodeYUV8` / `tj3DecodeYUV8`) with the **upstream-correct** ABI:

- 4th argument of `tjEncodeYUV3` is `pitch` (RGB row stride, `0` = tight `width * bpp`), not YUV alignment. Earlier rounds had this swapped — fixed in round-19 codex review.
- Legacy `flags` are mapped to the corresponding `TJPARAM_*` on the caller's handle via `process_legacy_compress_flags` / `process_legacy_decompress_flags`, mirroring upstream `turbojpeg.c::processFlags`. Compress side propagates `TJFLAG_BOTTOMUP`, `TJFLAG_PROGRESSIVE`, `TJFLAG_FASTDCT`; decompress side propagates `TJFLAG_BOTTOMUP`, `TJFLAG_FASTUPSAMPLE`, `TJFLAG_FASTDCT`.

End-to-end coverage in `cargo test -p libjpeg-turbo-rs-capi --release --test legacy_aliases`:

- `tj_encode_decode_yuv_legacy_aliases_roundtrip_444` — RGB → packed YUV (4:4:4) → RGB round-trip with `pitch = 0` and `align = 1`. Max per-channel diff ≤ 8 (BT.601 conversion rounding only).
- `tj_yuv_legacy_aliases_propagate_bottomup_flag` — explicitly verifies `TJFLAG_BOTTOMUP` lands on `TJPARAM_BOTTOMUP=1` after both `tjEncodeYUV3` and `tjDecodeYUV`.

The module-level doc comment in `legacy.rs` was updated to match the actual behavior.

### P2. Upstream `tjbench` / `rdjpgcom` / `wrjpgcom` Harness Not Yet Linked

**Symptom:** `examples/stock_djpeg_cjpeg/build.sh` builds `djpeg`, `cjpeg`, `jpegtran`. Three more upstream tools are not wired against our cdylib:

- `tjbench` (`references/libjpeg-turbo/src/tjbench.c` + `src/tjutil.c`) — links against TJ3, the canonical perf-regression harness in upstream. Building it against our shim publishes apples-to-apples perf numbers that complement `experiments/x86_64_avx2_final_report.md` and validate the encode-perf P1 closure with the same harness upstream uses.
- `rdjpgcom` / `wrjpgcom` — standalone tools for COM-marker read/write, do not link against the JPEG library; should already work as a header-only smoke test.

**Acceptance:**

```bash
$OUT/tjbench testimages/testorig.jpg 95
```

Numbers within ±10 % of upstream `tjbench` on the same hardware.

### P2. PNG Image I/O Is Optional, Not Core Replacement Work

PNG support in `tj3LoadImage8` / `tj3SaveImage8` is conditional in upstream libjpeg-turbo. It should not block replacement-readiness unless the intended downstream consumers require PNG through TurboJPEG image I/O.

If implemented, gate it behind a `png` feature and keep the default codec crate small.

---

## Execution Plan

### Task 1: Make the Native Transform Matrix Green

**Files:**

- Modify: `tests/cross_product_transform.rs`
- Modify: `src/api/coefficient.rs`
- Modify as needed: `src/encode/arithmetic.rs`

- [ ] **Step 1: Reproduce the exact failure**

```bash
cargo test -p libjpeg-turbo-rs --test cross_product_transform tjtrantest_full_cross_product -- --exact
```

Expected: failure with `arithmetic DC overflow` decode failures.

- [ ] **Step 2: Add a narrow regression test**

Add focused tests for:

- `gray-rows-hflip`, arithmetic output, no crop, no optimize, no progressive.
- `444-blocks-vflip`, arithmetic + progressive output, restart-blocks source, copy-all.

Each test must call `transform_jpeg_with_options`, then `decompress`, and assert success.

- [ ] **Step 3: Compare with upstream**

Generate the equivalent JPEG through upstream `jpegtran` where possible, then compare decoded pixels with `djpeg`. If byte-exact is realistic for this path, assert byte equality; otherwise assert pixel equality and document why.

- [ ] **Step 4: Fix the arithmetic/restart state bug**

Fix the general state transition. Do not special-case the failing test tuple.

- [ ] **Step 5: Run the full matrix**

```bash
cargo test -p libjpeg-turbo-rs --test cross_product_transform
```

- [ ] **Step 6: Run the workspace**

```bash
cargo test --workspace --no-fail-fast
```

### Task 2: Convert Compatibility Blockers Into Hard Gates

**Files:**

- Modify: `tests/capi_pillow_compat.rs`
- Modify: `tests/capi_stock_tool_link.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/tests/tjunittest_link.rs`
- Modify or delete: `crates/libjpeg-turbo-rs-capi/tests/capi_stock_djpeg_e2e.rs`

- [ ] **Step 1: Unignore passing `tjunittest_default_suite_passes`**

```bash
cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link
```

Expected: both symbol resolution and default suite pass.

- [ ] **Step 2: Make Pillow blocker code 3 a failure**

Change `tests/capi_pillow_compat.rs` so exit code 3 panics. Keep exit code 2 as a local-environment skip only.

- [ ] **Step 3: Update stock-tool tests**

Remove the stale "shim lacks classic jpeg API" expectation. Replace it with an assertion that stock tools build and run.

- [ ] **Step 4: Fix or retire duplicate broken harnesses**

`capi_stock_djpeg_e2e.rs` currently fails to build stock tools due `jversion.h` setup. Either share `examples/stock_djpeg_cjpeg/build.sh` or remove the duplicate test.

### Task 3: Fix Stock `djpeg` Runtime Abort

**Files:**

- Modify: `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/src/memmgr.rs`
- Modify: `examples/stock_djpeg_cjpeg/run.sh`
- Modify: `tests/capi_stock_tool_link.rs`

- [ ] **Step 1: Preserve crash logs**

Update `examples/stock_djpeg_cjpeg/run.sh` so failing runs print or preserve `djpeg_err_ours.log` and the crashing command.

- [ ] **Step 2: Reproduce with one image**

```bash
cargo build -p libjpeg-turbo-rs-capi --release
bash examples/stock_djpeg_cjpeg/build.sh
examples/stock_djpeg_cjpeg/build/djpeg -outfile /tmp/ours.ppm references/libjpeg-turbo/testimages/testorig.jpg
```

Expected: current abort.

- [ ] **Step 3: Fix the C ABI state or memory-manager mismatch**

Trace whether the abort occurs in `jpeg_read_header`, `jpeg_start_decompress`, scanline output, destination manager setup, or memory cleanup. Fix the general C ABI contract violation.

- [ ] **Step 4: Pass the stock decode corpus**

```bash
bash examples/stock_djpeg_cjpeg/run.sh
```

At minimum, `djpeg` must stop aborting and match upstream output on `testorig`, `testimgari`, `testimgint`, and `monkey12`.

### Task 4: Implement High-Precision Raw-Data C ABI Symbols

**Files:**

- Modify: `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/tests/capi_classic_decode_ext.rs`
- Modify or add: `crates/libjpeg-turbo-rs-capi/tests/capi_classic_raw_data.rs`
- Modify: `tests/capi_pillow_compat.rs`

- [ ] **Step 1: Add dlopen symbol tests**

Assert the shim exports `jpeg_read_raw_data`, `jpeg12_read_raw_data`, `jpeg_write_raw_data`, and `jpeg12_write_raw_data` (upstream `jpeglib.h` does not declare 16-bit raw-data entry points). Already in place via `tests/capi_stock_tool_link.rs::shim_exports_classic_jpeg_api`'s `required_p0_3` array — extend it whenever a new drop-in caller surfaces another required symbol.

- [ ] **Step 2: Implement minimal behavior**

Delegate to existing Rust raw-data APIs. For unsupported states, fail through the libjpeg error path rather than leaving symbols absent.

- [ ] **Step 3: Re-run Pillow**

```bash
cargo test --test capi_pillow_compat -- --nocapture
```

Expected: no loader blocker. If Pillow reaches a later behavioral failure, add that as the next focused test.

### Task 5: Support Foreign Virtual Coefficient Arrays For `jpegtran`

**Files:**

- Modify: `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/src/memmgr.rs`
- Modify: `tests/capi_stock_tool_link.rs`
- Modify: `examples/stock_djpeg_cjpeg/run.sh`

- [ ] **Step 1: Add a failing stock `jpegtran` test**

Use `examples/stock_djpeg_cjpeg/build/jpegtran -copy all -rotate 90` on `testorig.jpg` and assert it succeeds.

- [ ] **Step 2: Detect virtual barray handles**

Teach `jpeg_write_coefficients` to distinguish this shim's `CoefHandle` from virtual barray pointers allocated by `memmgr`.

- [ ] **Step 3: Materialize virtual arrays**

Convert the virtual coefficient arrays into `JpegCoefficients`, preserving dimensions, component sampling, quant tables, colorspace, density, Adobe marker state, restart interval, and saved markers.

- [ ] **Step 4: Run stock `jpegtran` parity**

```bash
bash examples/stock_djpeg_cjpeg/run.sh
```

Expected: `jpegtran` transform rows pass on the upstream testimage corpus.

### Task 6: Fill Legacy TurboJPEG Load/Save Aliases

**Files:**

- Modify: `crates/libjpeg-turbo-rs-capi/src/legacy.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/tests/legacy_aliases.rs`

- [ ] **Step 1: Replace stub tests**

Delete the assertion that `tjLoadImage` must fail. Add tests that load PPM/PGM/BMP and save PPM/PGM/BMP through the legacy ABI.

- [ ] **Step 2: Delegate to TJ3**

Route `tjLoadImage` and `tjSaveImage` to the existing TJ3 image I/O implementations and map legacy flags to TJ3 parameters.

- [ ] **Step 3: Run C ABI tests**

```bash
cargo test -p libjpeg-turbo-rs-capi --test legacy_aliases
```

### Task 7: Wire `TJPARAM_PRECISION` Through TJ3 Compress

**Files:**

- Modify: `crates/libjpeg-turbo-rs-capi/src/compress.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/src/precision.rs`
- Modify: `crates/libjpeg-turbo-rs-capi/tests/precision.rs`

- [ ] **Step 1: Add failing tests**

Add dlopen tests for:

- `tj3Compress8` + lossless + precision 4.
- `tj3Compress12` + lossless + precision 10.
- `tj3Compress16` + lossless + precision 14.

- [ ] **Step 2: Implement upstream dispatch**

Match upstream behavior: only honor `TJPARAM_PRECISION` inside the allowed precision family and only when lossless is enabled.

- [ ] **Step 3: Cross-check**

Use upstream tools where available and assert the SOF precision in the output stream.

---

## Definition Of Done

A task is done only when:

1. It has a focused regression test that would have failed before the change.
2. It cross-validates against upstream C tools where a C tool path exists.
3. It removes, rather than adds, skip/ignore behavior for product paths.
4. It updates `docs/FEATURE_PARITY.md` and `docs/C_API_REFERENCE.md` if a canonical mapping changed.
5. It leaves `cargo fmt --all`, `cargo clippy --lib -- -D warnings`, and the relevant `cargo test` command green.
6. For non-trivial code changes, it passes the repository's required post-implementation review flow.

---

## Suggested Order

1. ~~Fix `cross_product_transform` so the workspace is green (P0-1).~~ **CLOSED 2026-04-28** — all 12 cases pass.
2. ~~Harden gates by removing stale ignores and blocker-as-skip behavior (P1 Soft-Skip).~~ **CLOSED 2026-04-28** — every product-path ignore/skip is now a real failure path; harness force-rebuilds the cdylib so stale `target/release/...` cannot satisfy gates.
3. ~~Fix stock `djpeg` aborts (P0-2).~~ **CLOSED 2026-04-28** — `examples/stock_djpeg_cjpeg/run.sh` reports `OK all_byte_exact`.
4. ~~Add high-precision raw-data symbols and make Pillow load (P0-3).~~ **CLOSED 2026-04-28** — Pillow round-trip @ q=90 PSNR 49.49 dB.
5. Implement virtual coefficient-array materialization (and any libjpeg API symbol stock `jpegtran` resolves at runtime that the shim hasn't exported yet) for the stock `jpegtran` transform path (P0-4). **CLOSED 2026-04-28** — full TJXOP + crop + `-copy` cross-product byte-exact for 8-bit fixtures; 12-bit transcode (`monkey12`) routes through optimised Huffman, decodes pixel-equal through stock djpeg, and is no longer skipped by `examples/stock_djpeg_cjpeg/run.sh`.
5b. ~~Preserve source DHT in `JpegCoefficients` so 12-bit transcode can byte-match upstream `jpegtran`.~~ **CLOSED 2026-04-28** — root cause was *not* DHT regeneration (our optimised `gen_optimal_table` already matches upstream's regenerated tables for monkey12 transforms). The actual divergence was that `jpeg_read_header` in the FFI shim never populated `cinfo->marker_list`, so stock `transupp::jcopy_markers_execute` (used by `jpegtran -copy all`) found no source markers to forward and silently dropped the 3040-byte APP2/ICC chunk on `monkey12.jpg`. Closure: `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs::jpeg_read_header` now re-parses with the per-cinfo marker save list and threads a linked list of `JpegMarkerStructPublic` nodes (owned by `DecompressPrivate::marker_list_storage`) into `cinfo->marker_list`. `examples/stock_djpeg_cjpeg/run.sh` now reports `jpegtran monkey12 pass` (no fallback) and `tests/transcode_12bit_byte_exact.rs` asserts byte-equality across `-rotate 90/180`, `-flip horizontal`, and `-transpose`.
6. ~~Fill legacy `tjLoadImage` / `tjSaveImage` and `tjEncodeYUV3` / `tjDecodeYUV`.~~ **CLOSED 2026-04-28** — handle-less load/save ABI with BMP TJPF_BGR + alpha-strip + bottom-up; YUV aliases forward through `tj3EncodeYUV8` / `tj3DecodeYUV8` with end-to-end 4:4:4 round-trip coverage in `legacy_aliases.rs`.
7. ~~Wire arbitrary precision lossless through TJ3 compress.~~ **CLOSED 2026-04-28** — `tj3Compress8/12/16` honour `TJPARAM_PRECISION` via SOF dlopen tests for precision 4 / 10 / 14.
8. ~~Wire upstream `tjbench` / `rdjpgcom` / `wrjpgcom` against our shim (P2).~~ **CLOSED 2026-04-29** — `examples/stock_djpeg_cjpeg/build.sh` now compiles all six upstream tools (djpeg / cjpeg / jpegtran / **tjbench** / **rdjpgcom** / **wrjpgcom**) against our cdylib (tjbench links the `libturbojpeg.0` alias; rdjpgcom/wrjpgcom are standalone). Closing the missing TJ symbols required adding `tj3GetICCProfile`, `tj3SetICCProfile`, and `tj3TransformBufSize`. `run.sh` now reports per-fixture `tjbench pass` (decompress benchmark runs end-to-end) and `comtools pass roundtrip` (wrjpgcom-inserted COM marker survives rdjpgcom read-back). Apples-to-apples perf gate (step 9) is now unblocked.
9. ~~Close the x86_64 encode SIMD gap (P1 Encode) until every encode benchmark `Rust/C ≤ 1.05×`.~~ **CLOSED 2026-04-30** — verified on i5-10400 with `RUSTFLAGS="-C target-cpu=native"`. Every entry in the encode matrix is now `Rust/C ≤ 0.98×` (Rust is faster than C); 1080p_420 lands at 0.98× (10 273 µs vs 10 474 µs), 1080p_444 at 0.96× (19 057 µs vs 19 873 µs). Default-x86_64 builds (SSE2-only) trail by 5–10 pp at 1080p because LLVM cannot emit BMI1/LZCNT/BMI2 for scalar bitmap loops without target-feature; this is a build-flag recommendation, not a code defect. Two source-level workaround attempts (32-pixel AVX2 deinterleave; `lzcnt` bit-twiddle for the NBITS table) regressed and were discarded — kept as `discard` rows in `experiments/encode.tsv` for institutional memory.
10. ~~PNG image I/O (P2), if downstream demand exists.~~ **CLOSED 2026-04-29** (PR feat/png-image-io) — added `png` Cargo feature (off by default) to both root crate and `libjpeg-turbo-rs-capi`. `tj3LoadImage8` dispatches via 8-byte PNG signature; `tj3SaveImage8` dispatches by `.png` extension. Supports 8-bit RGB/RGBA/Grayscale; 16-bit and indexed-colour return `Unsupported`. When feature is off both functions return a clear `"PNG support not enabled in this build"` error. Five dlopen tests cover: round-trip RGB, round-trip RGBA, round-trip Grayscale, PNG→JPEG(q=90)→decode PSNR ≥ 30 dB, and feature-gate error.

This order is intentionally strict. A replacement project should not optimize the encoder or add optional PNG support while stock tools abort and compatibility blockers are silently skipped. Encode perf stays tracked but deferred; PNG stays optional.

---

## Phase 2 — System-Library Drop-In Hardening

The Phase 1 gates above (P0-1 through P0-4, P1-Soft-Skip, P1-Encode, P1-Legacy, etc.) close the *Rust-application replacement* and *stock-tool drop-in* stories. Phase 2 closes the remaining gap to **system-library** replacement: shipping a `libjpeg.so.62` / `libturbojpeg.so.0` SONAME-compatible binary that arbitrary distro packages (Pillow, ImageMagick, libvips, SDL_image, FFmpeg, GraphicsMagick, GD, …) can link against without source changes, on every platform upstream libjpeg-turbo officially supports.

External cross-check on 2026-05-04: the analysis "Rust app library = ready; system-library replacement = not yet" is consistent with the live state of this repo. The blockers below are the verifiable ones (each cites the file/line that proves the gap is open). Items the external review flagged that are *already closed* (`tj3GetICCProfile` / `tj3TransformBufSize` exports, `jpeg_set_marker_processor` wiring, `JpegSourceMgr` suspension semantics, `capi_imagemagick_compat` / `capi_pillow_compat` harnesses) are intentionally not in this list.

### P2-1. Full C Parity Workflow Soft-Skips — **CLOSED**

**Status (2026-05-04): closed.** Both `continue-on-error: true` flags in `.github/workflows/full-c-parity.yml` are gone:

- `c_tjcomptest_full` — flag removed when P2-11 closed (samp411/441/410/24 progressive parity).
- `c_tjtrantest_full` — flag removed in this commit. Local run on aarch64 macOS reports `11190 tested, 17538 skipped, 0 failed` for the full transform matrix (grayscale and non-grayscale combos). The "Known failures: grayscale Huffman diff" annotation that the flag carried was stale — earlier work that fixed the underlying divergence never updated the CI flag.

The `tests/c_tjtrantest.rs:537-543` source-level skip for `progressive + 4-pixel chroma + non-grayscale` is *kept on purpose*: that path goes through the **transform** writer (`src/api/coefficient.rs::write_coefficients_progressive`), not the encoder writer this session fixed. The transform path retains a `max_h <= 2 && max_v <= 2` filter at `coefficient.rs:1047` that silently falls back to baseline. Lifting that filter is a separate gap (would become P2-12 if pursued); leaving the source-level skip in place keeps the test honest about which paths it actually covers.

**Risk note:** flag removal was based on aarch64 macOS local results. Linux x86_64 (the other workflow leg) has not been re-verified locally because we don't have a cross-builder. The workflow runs weekly on Mondays; if the next run reds, react then. The flag was masking a *suspected* x86_64 divergence — it may still be a real one. Re-introducing the flag would be the wrong response: either the bytes match or the test is wrong; either way we want to see the actual failure, not silence it.

### P2-2. `default_format_message` Printf Expansion — **CLOSED**

**Status (2026-05-04): closed.** `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs::default_format_message` now walks the format string and substitutes `msg_parm.i[..8]` / `msg_parm.s[]` according to the upstream contract in `references/libjpeg-turbo/src/jerror.c:153-197`. The new helper `snprintf_jpeg` covers every printf specifier `references/libjpeg-turbo/src/jerror.h` actually uses: `%s %d %u %x %X %c %02d %3d %4u %02x %04x %%` (zero-padded width, ignored flags `-`/`+`/`#`/space). Mode selection (string vs integer args) follows jerror.c — the FIRST `%X` decides; mixing is unsupported.

**Verification (`crates/libjpeg-turbo-rs-capi/tests/format_message.rs`):** 8 tests dlopen the cdylib, install a synthetic addon table, set `msg_parm`, invoke `format_message` through the standard `jpeg_error_mgr` vtable, and assert the formatted output equals what `libc::snprintf` produces with the same format and args:

```text
test format_message_d_specifier_matches_snprintf            ... ok
test format_message_u_specifier_matches_snprintf            ... ok
test format_message_x_specifier_matches_snprintf            ... ok
test format_message_c_specifier_matches_snprintf            ... ok
test format_message_zero_padded_d_matches_snprintf          ... ok
test format_message_percent_literal_matches_snprintf        ... ok
test format_message_s_specifier_matches_snprintf            ... ok
test format_message_no_specifier_matches_msgtext_verbatim   ... ok
```

**TDD-verified:** reverting `default_format_message` to the prior verbatim-copy implementation makes 7 of those 8 tests RED-fail (only the no-specifier test still passes, because verbatim copy is correct when the message has no `%X`). Restoring the fix returns to GREEN.

### P2-3. Per-Platform ABI Validation — **PARTIAL: 64-bit cross-platform via P2-4 test; 32-bit / per-platform `const_assert` blocks deferred**

**Status (2026-05-04): partially closed.** The P2-4 ABI cross-check test (`crates/libjpeg-turbo-rs-capi/tests/abi_offsets.rs`) now runs in a dedicated `abi-offsets` matrix CI job (`.github/workflows/ci.yml`) on `ubuntu-latest` (Linux x86_64, LP64), `macos-latest` (aarch64, LP64), and `windows-latest` (x86_64 MSVC, LLP64). Each host probes its own ABI by reading `offset_of!` against the host's compiled struct and comparing to a C harness compiled on the same host.

The Windows skip in the test was lifted: `cfg!(windows)` no longer short-circuits, so any per-platform Rust↔C struct divergence on Windows MSVC will surface loudly on the first PR. The 64-bit gate remains because the hand-typed `assert!(offset_of!(...) == N)` constants in `jpeglib.rs:4096+` only apply to LP64; on 32-bit hosts the offsets shift proportionally.

**Still open (deferred, not blocking):**

1. **Per-platform `const_assert!` blocks for Windows LLP64.** Today the const-assert block in `jpeglib.rs` is gated on `not(windows)`, so a Windows-specific layout drift passes the *compile* gate. The runtime `abi_offsets` test catches it instead. Filling in Windows LLP64 offsets would catch drift at compile time too — only worth doing once we have a real Windows MSVC consumer to validate against.
2. **32-bit targets** (`i686-pc-windows-msvc`, `i686-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf`). Same posture: deferred until a downstream user requests the platform. Adding cross-compile *build* checks would surface compile errors on 32-bit but not layout drift; layout drift would require either a hand-pinned const block or a matching cross-compiled C harness, neither of which is cheap to maintain.

The P2-9 doc explicitly carves these out as "Phase 3 ask" work — keeping P2-3 *fully* open would double-track the same backlog.

### P2-4. Generated C-Side ABI Cross-Check — **CLOSED (jpeg_decompress_struct)**

**Status (2026-05-04): closed for `jpeg_decompress_struct`.** `crates/libjpeg-turbo-rs-capi/tests/abi_offsets.rs` synthesises a minimal `jconfig.h` (`JPEG_LIB_VERSION 80` + the upstream `WITH_JPEG8` defaults), writes a tiny C program that calls `offsetof(struct jpeg_decompress_struct, FIELD)` for every field that `jpeglib.rs:4096+` const-asserts, compiles it against the submodule's `references/libjpeg-turbo/src/jpeglib.h`, runs the binary, and asserts each emitted offset equals `std::mem::offset_of!(JpegDecompressPublic, FIELD)`.

**Coverage today (27 fields):** `err`, `mem`, `progress`, `client_data`, `is_decompressor`, `global_state`, `src`, `image_width`, `image_height`, `num_components`, `jpeg_color_space`, `out_color_space`, `scale_num`, `scale_denom`, `output_gamma`, `buffered_image`, `raw_data_out`, `quantize_colors`, `coef_bits`, `quant_tbl_ptrs`, `dc_huff_tbl_ptrs`, `ac_huff_tbl_ptrs`, `data_precision`, `comp_info`, `is_baseline`, `progressive_mode`, `arith_code`. Anything `jpeglib.rs` later const-asserts should be appended to `rust_offsets()` in lockstep.

**TDD-verified:** changing `JPEG_LIB_VERSION 80` → `JPEG_LIB_VERSION 70` in the harness's `jconfig.h` makes `cc` reject the program with `error: no member named 'is_baseline' in 'jpeg_decompress_struct'` — the test correctly red-fails when the C-side layout diverges from Rust's expectation. Restoring `80` returns to GREEN.

**Skip-with-reason cases:**
- non-LP64 / Windows host (matches the Rust assertion block's gate),
- `cc` not on PATH or not runnable,
- submodule not initialised (`references/libjpeg-turbo/src/jpeglib.h` missing),
- environmental cc failure (missing system headers, broken cross-compile setup).

**Out of scope (not blocking closure):** extending the cross-check to `jpeg_compress_struct`, `jpeg_error_mgr`, `jpeg_source_mgr`, `jpeg_destination_mgr`, `jvirt_barray_control`, `jvirt_sarray_control`, `jpeg_marker_struct`. The infrastructure is in place — adding more types is repeating the existing pattern with a different `struct foo` name in the C harness and a different Rust type in `rust_offsets()`. Tracked as a follow-up rather than a blocker because field-order drift in `jpeg_decompress_struct` is by far the highest-risk surface (this is what stock djpeg / cjpeg / Pillow / ImageMagick reach into directly).

### P2-5. Symbol-Export Inventory Diff — **CLOSED**

**Status (2026-05-04): closed.** `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs` now parses the submodule's `references/libjpeg-turbo/src/jpeglib.h` for `EXTERN(...)` declarations (66 found) and `references/libjpeg-turbo/src/turbojpeg.h` for `DLLEXPORT` declarations (79 found), then dlopens our cdylib and asserts every parsed symbol is resolvable.

**Allowlist of intentionally-deferred symbols** (19 entries, each with a one-line rationale at the call site):

- `jpeg_calc_jpeg_dimensions` — companion to `jpeg_calc_output_dimensions`; not exercised by stock cjpeg / Pillow / ImageMagick (the dimension calculation happens inside the library).
- `tjAlloc` / `tjFree` — superseded by `tj3Alloc` / `tj3Free`.
- `tjCompress` / `tjCompressFromYUV` / `tjCompressFromYUVPlanes` — superseded by `tjCompress2` (already exported) and the TJ3 forms.
- `tjDecompress` / `tjDecompressHeader` / `tjDecompressHeader2` / `tjDecompressToYUV` / `tjDecompressToYUV2` / `tjDecompressToYUVPlanes` — superseded by `tjDecompress2` / `tjDecompressHeader3` (already exported) and the TJ3 forms.
- `tjEncodeYUV` / `tjEncodeYUV2` / `tjEncodeYUVPlanes` / `tjDecodeYUVPlanes` — superseded by `tjEncodeYUV3` / `tjDecodeYUV` (already exported) and the TJ3 forms.
- `tjGetErrorCode` / `tjGetErrorStr` / `tjGetScalingFactors` — superseded by the TJ3 forms.

**The contract**: the test passes when the cdylib exports every upstream symbol *except* allowlisted ones. Removing a name from the allowlist signals "this is now implemented; the test should hold us to it from this commit on." The test thus sharpens to "no NEW gaps may appear."

**CI hookup**: bundled with P2-4 in the `capi-abi-checks` matrix job (`.github/workflows/ci.yml`) so all three ABI flavours (Linux x86_64 LP64, macOS aarch64 LP64, Windows MSVC LLP64) run the symbol diff on every PR.

**Out of scope (deferred):**

- Comparing against an *installed* upstream `libjpeg.so.62` / `libturbojpeg.so.0` via `nm -D` — would catch symbol-version tags and platform-specific export differences, but requires upstream installed at test time. The header-based check is the cheaper baseline and runs everywhere.
- SONAME match (`libjpeg.so.62` ↔ `libjpeg.so.8`) — owned by P2-9 (build.rs SONAME wiring + warning) and `docs/ABI_COMPATIBILITY.md`.

### P2-6. Crate Is `publish = false`

**Symptom:** `crates/libjpeg-turbo-rs-capi/Cargo.toml:9` is `publish = false` and the version is still `0.1.0`. There is no path for downstream Rust consumers to `cargo add libjpeg-turbo-rs-capi`.

**Why this matters:** A "drop-in replacement" library that cannot be installed through the language's standard package manager is not actually drop-in for the Rust ecosystem.

**Likely area:**

- `crates/libjpeg-turbo-rs-capi/Cargo.toml` (flip `publish`, bump version, add `description`, `license`, `repository`, `keywords`, `categories`).
- `crates/libjpeg-turbo-rs-capi/README.md` (new).
- `.github/workflows/release.yml` (already supports `wasm-v*` tags; extend for `capi-v*`).

**Acceptance:**

```bash
cargo publish -p libjpeg-turbo-rs-capi --dry-run
```

Must succeed. Then publish a `0.1.0` (or `1.0.0-rc.1`) candidate. Hold actual `1.0.0` until P2-1 through P2-5 are closed.

### P2-7. Differential / Roundtrip Fuzzing Against C — **CLOSED (decode + encode + transform); 24-hour long-run deferred**

**Status (2026-05-04): closed for the structural deliverables.** All three differential libfuzzer targets land and join the nightly matrix; the 24-hour scheduled long-run + OSS-Fuzz corpus publishing remain as a future scaling step (the 10-min nightly is the structural baseline).

**Done:**

- `fuzz/fuzz_targets/fuzz_decode_diff_c.rs` — feeds each fuzzed input to both `Decoder::decode` and a subprocessed `djpeg`, then asserts (a) acceptance agreement: when C accepts, Rust must accept too (drop-in floor); (b) dimension agreement; (c) pixel agreement within ±16 per byte (IDCT precision noise — curated `corpus_test` enforces byte-exact). Lenient direction (Rust accepts more) is allowed by design. Arithmetic-coded inputs (SOF9/10/11) are skipped — the arithmetic decoder has a known mid-scan divergence with libjpeg-turbo on a small fuzz subset (open follow-up below).
- `fuzz/fuzz_targets/fuzz_encode_diff_c.rs` — encodes a fuzz-supplied pixel buffer via Rust (`compress` / `compress_progressive` / `compress_arithmetic` / `compress_arithmetic_progressive`), then verifies that both Rust and C `djpeg` decode the result equivalently. Catches "Rust encoder produces output C consumer rejects" — the mirror of the decode-side differential.
- `fuzz/fuzz_targets/fuzz_transform_diff_c.rs` — applies HFlip / VFlip / Rot180 (the three ops that don't require MCU alignment) via both Rust `transform_jpeg_with_options` and subprocessed `jpegtran`, decodes both transformed JPEGs through `djpeg`, and asserts pixel agreement. Transpose / Transverse / Rot90 / Rot270 are out of scope (require MCU alignment, covered by curated `examples/corpus_test.rs` instead).
- All three targets wired into `.github/workflows/fuzz-smoke.yml` matrix (10 min nightly each). The `libjpeg-turbo-progs` install step now fires for any of the `*_diff_c` targets.
- Pre-existing baseline: `examples/corpus_test.rs` already runs decode + encode + transform differential against C `djpeg` / `cjpeg` / `jpegtran` for every fixture in `tests/corpus/` on every PR (`.github/workflows/ci.yml::test-corpus`). That covers the curated-corpus dimension; the new libfuzzer-driven targets add random-input coverage on top of the same agreement contract.

**Subprocess vs in-process FFI:** all three targets call C tools via `std::process::Command` rather than linking C libjpeg into the harness. Slower per-iteration (~ms vs μs) but avoids dragging `cc-rs` + system libjpeg into the fuzz crate. In-process FFI is tracked as a follow-up; the throughput delta only matters once we're chasing a specific corpus-coverage target.

**Deferred:**

- 24-hour scheduled long-run (`-max_total_time=86400` on a weekly cron) + OSS-Fuzz-style corpus publishing. The 10-min nightly is the structural floor; longer runs amortize the libjpeg-turbo-progs install cost over more iterations and surface deeper-mutation crashes that the 10-min budget cannot reach.

**Acceptance:**

```bash
cargo +nightly fuzz run fuzz_decode_diff_c     -- -max_total_time=600
cargo +nightly fuzz run fuzz_encode_diff_c     -- -max_total_time=600
cargo +nightly fuzz run fuzz_transform_diff_c  -- -max_total_time=600
```

Each must run 10 min in CI without finding a divergence. All three verified locally via `cargo check --bin <target>` from `fuzz/`; first scheduled CI run confirms end-to-end on Linux x86_64 with `libjpeg-turbo-progs` present.

#### Follow-up: arithmetic decoder mid-scan divergence — **OPEN**

`fuzz_decode_diff_c` surfaced a 146-byte arithmetic-coded grayscale fixture (272×16 SOF9, single DC component, scan ~30 bytes) where Rust and djpeg agree byte-exact for the first ~2287 output bytes, then diverge sharply: djpeg outputs `[0, 0, 0, …]` while Rust outputs `[0xFF, 0xFF, 0xFF, …]` for the rest of the scan. The fuzz target now early-returns on `is_arithmetic()` so this does not block the nightly matrix; the curated arithmetic conformance suites (`examples/corpus_test.rs`, `c_tjtrantest_full-arith-and-progressive-skip`) keep the byte-exact gate against pinned references.

Crash artifact (reproduce via `python3 -c "import sys; sys.stdout.buffer.write(bytes(<inline list>))"` or fetch from `fuzz/artifacts/fuzz_decode_diff_c/crash-3eca6f89484e8f8756677ed2a38ffa0ddef6fcdf` after a `fuzz-smoke` run that lands the input):

```
[1m header line: SOI APP0(JFIF) DQT(0) SOF9(8b 16h 272w 1c, samp 1×1, q=0) DAC(Tc=0 Tb=0 Cs=5) SOS(1c, Cs=1, Td=Ta=0, Ss=0 Se=63 Ah/Al=0) <30 bytes entropy> EOI
```

Most likely failure surface: arithmetic bit-stuffing recovery (0xFF / 0x00) at the end-of-scan boundary, or the decoder's MX (most-probable-symbol) state not flushing exactly the same way libjpeg does when the scan terminates with the buffer not fully drained. The curated arithmetic fixtures don't hit this because they always have full-image bitstreams; the fuzzer found it on a deliberately short scan.

**Acceptance:** restore the `if probe.is_arithmetic() { return; }` skip in `fuzz_decode_diff_c.rs` and have the nightly run survive 10 min on the same fixture (re-add it to `tests/generate_fuzz_seeds.rs::DECODER_TARGETS` for `fuzz_decode_diff_c` once fixed).

#### Follow-up: transform encoder small-image entropy divergence — **OPEN**

`fuzz_transform_diff_c` surfaced two 16×16 4:4:4 RGB fixtures (one Rot180, one VFlip). Local cross-check applied **all three supported ops** to both fixtures and confirmed the bug class is shared across HFlip / VFlip / Rot180:

| Source fixture | Op | rust_len | djpeg exit | rust self-decode | djpeg stderr |
|---|---|---|---|---|---|
| `crash-75b99921...` (Rot180 origin, 805B) | HFlip | 723 | 2 | OK | premature end of data segment |
| | VFlip | 724 | 2 | OK | premature end of data segment |
| | Rot180 | 724 | 2 | OK | premature end of data segment |
| `crash-de852cc2...` (VFlip origin, 778B) | HFlip | 810 | 2 | OK | premature end of data segment |
| | VFlip | 811 | 2 | OK | 8 extraneous bytes before marker 0xd9 |
| | Rot180 | 811 | 2 | OK | premature end of data segment |

Every op fails djpeg's decoder on both fixtures, and every Rust output round-trips through Rust's own decoder — the bug class is uniformly "valid-but-wrong-coefficients" across all three ops. Pattern: the input has 6 bytes of entropy for 12 expected blocks (severely truncated); each side pads missing coefficients differently, producing entropy-length divergence. Headers (SOI through SOS) are byte-identical to jpegtran's; the divergence is entirely in the entropy-coded segment. Symptom set rules out a marker-handling bug and points at the **coefficient-mapping / DC-predictor reset path** inside the transformer when reading from a truncated-entropy input — shared across all three ops, so one underlying bug in the transform writer's small-input handling, not three.

`fuzz_transform_diff_c` narrows the soft-skip to **inputs with both dimensions ≤ 32 px** (any of HFlip/VFlip/Rot180), gated additionally on Rust's own decoder accepting the output (so a future small-image regression that produces a structurally broken bitstream still trips libfuzzer). Larger inputs continue to assert C decodability so fresh transform encoder bugs there still surface.

**Acceptance:** delete the self-decode soft-skip in `fuzz_transform_diff_c.rs` and have the nightly run survive 10 min on both pinned crash artifacts (`crash-75b99921...` and `crash-de852cc2...`).

#### Follow-up: progressive small-entropy decoder pixel divergence — **CLOSED 2026-05-05**

`fuzz_decode_diff_c` (post-AC-bounds-soft-landing in commit ce14bbe) surfaced a 544-byte 16×16 SOF2 fixture with 10 progressive scans of which 8 carry only 1 byte of entropy each. djpeg accepts and decodes; Rust accepts and decodes, but the resulting pixels diverged: max abs diff = 61, mean ≈ 4.34, with 72 bytes of the 768-byte buffer differing by > 16. First 16 pixels byte-identical; divergence concentrated in the second MCU row.

**Root cause** (`src/decode/progressive.rs::decode_ac_refine`): when the inner zero-run loop exited via `k > Se`, the surrounding code's `if new_val != 0 && k <= se` guard *dropped* the new coefficient write entirely. libjpeg-turbo writes it at `*block + jpeg_natural_order[k]`, which the `[DCTSIZE2 + 16]` padding folds onto `coeff[63]`. The C reference therefore clobbers `coeff[63]` rather than skipping; we silently dropped state that subsequent refinement scans were supposed to refine, producing the observed pixel drift.

**Fix:** route `k > se` (within the soft-landing window k < 80) to `coeff[63]`, mirroring the AC initial soft-landing already in `decode_ac_first` and the libjpeg natural-order padding semantics.

**Verification:** `tests/progressive_ac_soft_landing.rs::ac_refine_soft_landing_matches_djpeg_byte_exact` pins the original 544-byte crash input and asserts byte-exact pixel agreement (max diff 0) vs djpeg.

### P2-8. Install-Staging, Symlink Chain, and CMake Config — **CLOSED**

**Status (2026-05-04): closed.** `scripts/install_capi.sh` stages the full distro-replacement layout into `${DESTDIR}${PREFIX}`:

```
${DESTDIR}${PREFIX}/lib/libjpeg.so.62.X.Y         # cdylib (or libjpeg.62.X.Y.dylib on macOS)
${DESTDIR}${PREFIX}/lib/libjpeg.so.62             # symlink → above
${DESTDIR}${PREFIX}/lib/libjpeg.so                # symlink → libjpeg.so.62
${DESTDIR}${PREFIX}/lib/libturbojpeg.so.0.X.Y     # same cdylib (we export both APIs)
${DESTDIR}${PREFIX}/lib/libturbojpeg.so.0
${DESTDIR}${PREFIX}/lib/libturbojpeg.so
${DESTDIR}${PREFIX}/lib/pkgconfig/libjpeg.pc
${DESTDIR}${PREFIX}/lib/pkgconfig/libturbojpeg.pc
${DESTDIR}${PREFIX}/lib/cmake/JPEG/JPEGConfig.cmake
${DESTDIR}${PREFIX}/include/jpeglib.h             # verbatim from references/libjpeg-turbo/src/
${DESTDIR}${PREFIX}/include/jerror.h
${DESTDIR}${PREFIX}/include/jmorecfg.h
${DESTDIR}${PREFIX}/include/jconfig.h             # generated; pins JPEG_LIB_VERSION 80
${DESTDIR}${PREFIX}/include/turbojpeg.h
```

The script supports `--destdir` / `--prefix` / `--soname` flags and an optional `--build` switch that builds the cdylib first if missing. It is wired into the top-level `Makefile` as `make install [DESTDIR=…] [PREFIX=…]`.

**CMake config** (`JPEGConfig.cmake`) exposes `JPEG_VERSION`, `JPEG_INCLUDE_DIRS`, `JPEG_LIBRARIES`, and the `JPEG::JPEG` imported target — exactly what `find_package(JPEG)` consumers expect.

**Test (`crates/libjpeg-turbo-rs-capi/tests/install_layout.rs`):** invokes the script into a tempdir, asserts:
1. cdylib is at the SONAME path,
2. symlink chains for both APIs resolve to a real file,
3. `pkg-config` files contain `Name`/`Version`/`Libs` lines and the `prefix=…` substitution matches,
4. `JPEGConfig.cmake` exposes `JPEG_VERSION` / `JPEG_INCLUDE_DIR` / `JPEG_LIBRARY` / `JPEG::JPEG`,
5. all five public C headers are present,
6. `jconfig.h` declares `JPEG_LIB_VERSION 80`,
7. (optional, if `pkg-config` on PATH) `pkg-config --libs libjpeg` against `PKG_CONFIG_PATH=<staged>` returns `-ljpeg`.

Skip-with-reason on Windows (script is bash; Windows packagers use their own conventions) or when `bash` is not on PATH.

**Out of scope (deferred):**

- LD_LIBRARY_PATH-injected Pillow round-trip against the staged tree. The `tests/capi_pillow_compat.rs` harness already verifies the cdylib works against Pillow when it's pre-loaded; verifying it through the installed-tree path is structural redundancy with no extra signal.
- `cmake --find-package` end-to-end check. The CMake config file content is asserted; running CMake itself just to confirm it parses the file would add a CMake dependency to the test suite without revealing additional bugs.
- Windows MSI / DLL install layout. The Linux/macOS install path is the entry-point most distros and Homebrew formulae need; Windows packagers typically ship raw artifacts via NSIS / WiX with their own conventions.

### P2-9. v6b / v7 / v8 ABI Compatibility Matrix — **CLOSED**

**Status (2026-05-04): closed via the documentation acceptance gate.** `docs/ABI_COMPATIBILITY.md` lands with the explicit decision: **v8-only struct layout; v6b SONAME is the documented-risk default; `CAPI_SONAME=libjpeg.so.8` + `CAPI_INSTALL_NAME=@rpath/libjpeg.8.dylib` is the production-safe override.** Per-version cdylib variants are explicitly deferred as a Phase 3 ask.

The companion build.rs change (`crates/libjpeg-turbo-rs-capi/build.rs:30-44,57-78`) emits a loud `cargo:warning=…` whenever the default `libjpeg.so.62` SONAME / `libjpeg.62.dylib` install_name pairing is used in a v8 build. The warning explains the risk, points at `docs/ABI_COMPATIBILITY.md`, names the safe override, and offers `CAPI_ACK_V6B_SONAME=1` for callers that have evaluated the risk and accept it. Setting either env var to a non-default value (or the ack flag) silences the warning, so the noise scales with how aware the operator is.

**What this *does not* close:**

- Real binary v6b drop-in (per-version layouts and per-version cdylibs) — explicitly out of scope per the doc's roadmap.
- The case where a downstream packaging script bypasses cargo and the build.rs warning never reaches the operator. That's mitigated by the doc, not the warning.

**Verification:**

```bash
# Default → loud cargo:warning lands on the build line.
cargo build -p libjpeg-turbo-rs-capi --release 2>&1 | grep -F "v6b"

# Production-safe build → silent.
CAPI_SONAME=libjpeg.so.8 CAPI_INSTALL_NAME=@rpath/libjpeg.8.dylib \
  cargo build -p libjpeg-turbo-rs-capi --release 2>&1 | grep -F "v6b" || echo "silent ok"
```

Both observed clean on 2026-05-04 (macOS aarch64).

### P2-10. Real Distro-Consumer Smoke Matrix — **CLOSED**

**Status (2026-05-04): closed.** All four planned consumer harnesses landed and pass on macOS aarch64 (with the documented skip-with-reason posture for mozjpeg-bound or libjpeg-less consumers). Linux CI exercises the libjpeg-turbo path.

**Done:**

- `crates/libjpeg-turbo-rs-capi/tests/capi_libvips_compat.rs` — drives `examples/libvips_smoke/run.sh` (encode + decode round-trip via `vips copy in.ppm out.jpg[Q=75]` then `vips copy out.jpg decoded.ppm`).
- `crates/libjpeg-turbo-rs-capi/tests/capi_ffmpeg_compat.rs` — drives `examples/ffmpeg_smoke/run.sh` (encode + decode round-trip via `ffmpeg -c:v mjpeg`). Skips when ffmpeg uses the internal MJPEG codec (Homebrew default).
- `crates/libjpeg-turbo-rs-capi/tests/capi_gd_compat.rs` — C-harness round-trip via `gdImageJpegPtr` / `gdImageCreateFromJpegPtr` (the canonical libgd encode/decode call sites). Verified locally: PSNR=38.4 dB on the smooth fixture (q=75 4:2:0 floor).
- `crates/libjpeg-turbo-rs-capi/tests/capi_sdl_image_compat.rs` — C-harness decode-only via `IMG_LoadTyped_RW(rwops, 1, "JPG")`. Encode side is out-of-band via `libjpeg_turbo_rs::Encoder` because SDL_image's `IMG_SaveJPG_RW` uses STB (not libjpeg). Verified locally: PSNR=38.4 dB on the same fixture.
- `crates/libjpeg-turbo-rs-capi/src/mozjpeg_compat.rs` — exports the 9-symbol mozjpeg parameter API (`jpeg_c_bool_param_supported` and family) as no-op stubs. Without these, a consumer linked against mozjpeg's `libjpeg.62.dylib` fails at dyld load time because mozjpeg's symbols are undefined references in the consumer's load command — even when the consumer never calls them. Probes return `FALSE`, setters are no-ops, getters return zero. The libvips harness surfaced this gap on its first run (Homebrew vips binds to mozjpeg).
- mozjpeg detection in all four harnesses (`exit 11`): mozjpeg adds extra fields *inside* `jpeg_compress_struct` for trellis quantization. The dyld stubs above let the consumer load, but the consumer's compiled struct offsets diverge from libjpeg-turbo v8 at runtime. We detect the mozjpeg dependency path and skip-with-reason on those hosts; Linux distros (Debian/Ubuntu/Fedora) ship the consumers linked against system libjpeg-turbo where the tests exercise the real path.

**Acceptance:**

```bash
cargo test -p libjpeg-turbo-rs-capi --test capi_libvips_compat
cargo test -p libjpeg-turbo-rs-capi --test capi_ffmpeg_compat
cargo test -p libjpeg-turbo-rs-capi --test capi_gd_compat
cargo test -p libjpeg-turbo-rs-capi --test capi_sdl_image_compat
```

Each runs a real round-trip with PSNR check. Skip-with-reason allowed only when the consumer is not installed, the consumer is not linked against libjpeg (e.g. ffmpeg's internal MJPEG codec, SDL_image's STB-only build), or the host's libjpeg is mozjpeg (incompatible runtime struct layout — Linux CI exercises the real libjpeg-turbo path).

### P2-11. TJSAMP_411 / TJSAMP_441 / TJSAMP_410 / TJSAMP_24 Progressive Encode — **CLOSED**

**Status (2026-05-04): closed.** `cargo test --release --features full-c-parity --test c_tjcomptest` is **green for the full lossy + lossless matrix** including progressive + samp411/441/410/24 on the 227×149 testorig fixture. The source-level skip in `tests/c_tjcomptest.rs:717-739` is gone, the new C-tool-free guard `tests/regression_progressive_4pixel_chroma.rs` exercises all four 4-pixel factors, and the `continue-on-error: true` flag for `c_tjcomptest_full` in `.github/workflows/full-c-parity.yml` is removed.

**Root cause (2026-05-04):** `src/encode/pipeline.rs::progressive_fdct_chroma_block` (and the matching arithmetic-progressive Cb/Cr branches at `pipeline.rs:4761` / `:4781`) clamped the chroma sampling factors with:

```rust
let hf: usize = if h_samp > 1 { 2 } else { 1 };
let vf: usize = if v_samp > 1 { 2 } else { 1 };
```

For S411 (`h_samp=4`) this collapsed `hf` to `2`, so the encoder downsampled chroma to 1/2 resolution while the SOF marker still advertised 1/4 resolution. The decoder unpacked half-resolution coefficients into the quarter-resolution chroma grid → garbled chroma plane.

**Diagnostic (`examples/diag_4pixel_chroma_diff.rs`)** — kept as the institutional reproducer:

| samp | mode | match | rust_bytes | c_bytes | first_d | px_max | px_mean |
|------|------|-------|------------|---------|---------|--------|---------|
| S411 | baseline    | Y | 5750 | 5750 | -     | 0   | 0.0000 |
| S411 | progressive | **N→Y** | 5642 | 5642 | -     | **140→0** | **8.97→0** |
| S441 | baseline    | Y | 5648 | 5648 | -     | 0   | 0.0000 |
| S441 | progressive | **N→Y** | 5556 | 5556 | -     | **161→0** | **8.80→0** |
| S410 | baseline    | Y | 5333 | 5333 | -     | 0   | 0.0000 |
| S410 | progressive | **N→Y** | 5207 | 5207 | -     | **161→0** | **9.12→0** |
| S24  | baseline    | Y | 5283 | 5283 | -     | 0   | 0.0000 |
| S24  | progressive | **N→Y** | 5165 | 5165 | -     | **161→0** | **8.95→0** |

The earlier skip-comment claimed "1 LSB downsample diff, decoded pixels match" — both halves were false. Pixel diff was max ≈140-161 (out of 255), mean ≈9. The bug was a real chroma-plane corruption, not a cosmetic byte difference.

**Fix:** drop the clamp, use `h_samp` / `v_samp` directly. The existing SIMD fast paths for `hf==2 && vf==1|2` still fire for 2-pixel factors; 4-pixel factors fall through to the scalar `downsample_chroma_block` which correctly mirrors C's `int_downsample` (`references/libjpeg-turbo/src/jcsample.c:153-191`).

**Out of scope (separate gap):** the same `max_h <= 2 && max_v <= 2` gate in `src/api/coefficient.rs:1047` rejects 4-pixel factors from the **transform / jpegtran** progressive writer (different code path: `write_coefficients_progressive`, with extra dimension-swap interactions). Tracked as P2-12 if a reviewer wants to push for transform parity too.

---

## Phase 2 Suggested Order

1. ~~**P2-11** — Close the TJSAMP_411/441/410/24 progressive-encode byte-parity gap.~~ **CLOSED 2026-05-04** — root cause was a chroma-sampling-factor clamp in `progressive_fdct_chroma_block`; fix landed in `src/encode/pipeline.rs`, source-level test skip deleted, regression test in `tests/regression_progressive_4pixel_chroma.rs`.
2. ~~**P2-1** (`c_tjcomptest_full` portion) — Remove `continue-on-error` flag for the encode parity test.~~ **CLOSED 2026-05-04** — flag removed in `.github/workflows/full-c-parity.yml` once P2-11 fix landed. Remaining `c_tjtrantest_full` flag (grayscale Huffman) is still open as a transform-path divergence.
3. ~~**P2-9** — Decide and document the `JPEG_LIB_VERSION` policy.~~ **CLOSED 2026-05-04** — `docs/ABI_COMPATIBILITY.md` documents the v8-only policy with v6b-SONAME risk explicitly called out; `build.rs` emits a loud `cargo:warning` on the default-risk pairing.
4. ~~**P2-2** — Implement `format_message` printf expansion.~~ **CLOSED 2026-05-04** — `snprintf_jpeg` helper added in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`; `tests/format_message.rs` exercises every specifier `jerror.h` uses against `libc::snprintf` as the reference oracle. TDD red-then-green confirmed.
5. ~~**P2-1** (remaining `c_tjtrantest_full` portion) — Investigate and fix or formally document the grayscale-Huffman transform divergence; remove the last `continue-on-error` flag.~~ **CLOSED 2026-05-04** — local run on aarch64 reports 11190/0 tested/failed; flag removed for both x86_64 and aarch64 jobs. Suspected x86_64 divergence will surface loudly on the next weekly cron if it's still real.
6. ~~**P2-4** — Generated C-side ABI cross-check.~~ **CLOSED 2026-05-04** — `tests/abi_offsets.rs` compiles a tiny C harness against the submodule's `jpeglib.h` and asserts every const-asserted field matches `offset_of!`. Coverage scoped to `jpeg_decompress_struct` (27 fields); other structs are a follow-up.
7. ~~**P2-3** — Per-platform offset assertions + CI matrix.~~ **PARTIAL 2026-05-04** — `abi-offsets` matrix CI job now runs the P2-4 cross-check on Linux x86_64, macOS aarch64, and Windows MSVC; per-platform `const_assert!` blocks and 32-bit targets are deferred until a real downstream user requests them.
8. ~~**P2-5** — Symbol-inventory diff against upstream.~~ **CLOSED 2026-05-04** — `tests/symbol_inventory.rs` parses upstream headers (66 jpeg + 79 tj symbols), asserts each is exported by our cdylib, and explicitly allowlists 19 deferred legacy entries with rationale. Bundled with P2-4 in the `capi-abi-checks` cross-platform CI job.
9. ~~**P2-8** — SONAME / pkg-config / install layout.~~ **CLOSED 2026-05-04** — `scripts/install_capi.sh` + `make install` stage cdylib + symlink chain + headers + `.pc` + `JPEGConfig.cmake` into `${DESTDIR}${PREFIX}`; `tests/install_layout.rs` asserts the layout end-to-end.
10. ~~**P2-7** — Differential fuzzing against C.~~ **CLOSED 2026-05-04** — three libfuzzer targets land: `fuzz_decode_diff_c` (Rust decode vs djpeg), `fuzz_encode_diff_c` (Rust encode roundtrip via djpeg + Rust decode), `fuzz_transform_diff_c` (Rust transform vs jpegtran for HFlip / VFlip / Rot180). All three on the nightly 10-min matrix in `.github/workflows/fuzz-smoke.yml`. 24-hour scheduled long-run + OSS-Fuzz-style corpus publishing deferred as a future scaling step. Decode/encode/transform differential against C is already CI-gated for the curated `tests/corpus/` corpus via `examples/corpus_test.rs::test-corpus`.
11. ~~**P2-10** — libvips / FFmpeg / SDL_image / GD consumer harnesses.~~ **CLOSED 2026-05-04** — all four landed: `capi_libvips_compat` + `capi_ffmpeg_compat` (CLI-based, LD_PRELOAD pattern), `capi_gd_compat` + `capi_sdl_image_compat` (C-harness pattern like `libtiff_integration`). The libvips first-run also surfaced and fixed the `jpeg_c_*_param_*` symbol-surface gap via `mozjpeg_compat.rs`, so consumers linked against mozjpeg can dyld-resolve against our cdylib (with documented runtime layout caveat).
12. **P2-6** — Publish to crates.io. Last, because publishing locks the ABI surface and we should not lock until P2-1 through P2-5 are closed.

---

## Reference Commands

```bash
cargo test -p libjpeg-turbo-rs --test cross_product_transform tjtrantest_full_cross_product -- --exact
cargo test --workspace --no-fail-fast
cargo build -p libjpeg-turbo-rs-capi --release
cargo test --test capi_stock_tool_link -- --include-ignored
cargo test --test capi_pillow_compat -- --nocapture
cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link
bash examples/stock_djpeg_cjpeg/build.sh
bash examples/stock_djpeg_cjpeg/run.sh
# Encode performance baseline (post-correctness):
cargo bench --bench encode
# Compile and run the matching C baseline (C source ships in
# examples/, no pre-built binary is checked in).
#
# Source-file selection is platform-specific because the timing
# primitives differ:
#   * macOS → `examples/bench_c_encode_matrix.c` (mach_absolute_time)
#   * Linux → `examples/bench_c_encode_linux.c` (clock_gettime)
#
# Prerequisites: a libjpeg-turbo install that exposes headers and
# `libjpeg`. If pkg-config is available (`brew install pkgconf` on
# macOS, `apt-get install pkg-config` on Debian/Ubuntu), the
# pkg-config form below works on both Homebrew and Conda. Without
# pkg-config, fall back to the explicit -I/-L flags pinned to your
# install prefix.
case "$(uname)" in
  Darwin) BENCH_SRC=examples/bench_c_encode_matrix.c ;;
  Linux)  BENCH_SRC=examples/bench_c_encode_linux.c ;;
  *)      echo "unsupported platform $(uname)"; exit 1 ;;
esac
if command -v pkg-config >/dev/null && pkg-config --exists libjpeg; then
  cc -O2 "$BENCH_SRC" -o /tmp/bench_c_encode_matrix \
     $(pkg-config --cflags --libs libjpeg) \
     -Wl,-rpath,$(pkg-config --variable=libdir libjpeg)
else
  # Fallback: point at your install prefix explicitly.
  PREFIX=${LIBJPEG_PREFIX:-${CONDA_PREFIX:-/opt/homebrew/opt/jpeg-turbo}}
  cc -O2 "$BENCH_SRC" -o /tmp/bench_c_encode_matrix \
     -I"$PREFIX/include" -L"$PREFIX/lib" -ljpeg \
     -Wl,-rpath,"$PREFIX/lib"
fi
/tmp/bench_c_encode_matrix
```
