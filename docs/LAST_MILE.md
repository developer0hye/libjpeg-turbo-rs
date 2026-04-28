# Last Mile Replacement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `libjpeg-turbo-rs` a credible replacement for the libjpeg-turbo C implementation, including Rust-native APIs, TurboJPEG/TJ3 C ABI callers, classic `jpeg_*` callers, stock tools, and downstream wrappers.

**Architecture:** Treat replacement-readiness as a compatibility product, not a feature checklist. The Rust-native codec is already broad; the remaining work is hardening exact C behavior, removing soft-skip compatibility tests, and making stock C callers fail loudly until they pass.

**Tech Stack:** Rust 2021, `libjpeg-turbo-rs`, `libjpeg-turbo-rs-capi`, upstream libjpeg-turbo C sources in `references/libjpeg-turbo`, Homebrew/system `djpeg`/`cjpeg`/`jpegtran`, Pillow/ImageMagick smoke harnesses.

---

## Cold Assessment

This project is not replacement-ready today.

It is close as a Rust-native JPEG library, but the last mile for replacing the C implementation is stricter than feature parity. A real replacement must survive unmodified C binaries, unmodified wrapper libraries, and obscure option cross-products without treating loader failures or aborts as skips.

Live checks on 2026-04-28 (refresh whenever the gap inventory changes — failure counts and blocker codes drift as patches land):

| Check | Current Result | Replacement Meaning |
| --- | --- | --- |
| `cargo test --workspace --release` | **Passes**: 2067 tests, 0 failures, 2 ignored. | Native + C ABI workspace is green. |
| `cargo test -p libjpeg-turbo-rs --test cross_product_transform` | **Passes** all 12 cases including `tjtrantest_full_cross_product`, `tjtrantest_arithmetic_cross_product`, and `c_jpegtran_cross_validation_*`. | P0-1 closed — arithmetic transform cross-product no longer corrupts. |
| `examples/stock_djpeg_cjpeg/run.sh` | **Passes** (`OK all_byte_exact`): `djpeg` / `cjpeg` / `jpegtran` byte-exact for `testimgari`, `testimgint`, `testorig`, `monkey12`; `monkey12` jpegtran is the documented 12-bit-transcode skip tracked under P0-4. | P0-2 closed — drop-in for stock C tools holds, except 12-bit jpegtran. |
| `cargo test --test capi_stock_tool_link` | **Passes** for djpeg / cjpeg / jpegtran (`-copy all -rotate 90`) on the 8-bit fixtures; the full TJXOP cross-product (`-flip h/v`, `-rotate 90/180/270`, `-transpose`, `-transverse`, `-grayscale`, `-crop` origin and offset) is verified byte-exact via the foreign-coef-array path. | Drop-in for stock 8-bit `jpegtran` is closed; 12-bit transcode remains. |
| `cargo test --test capi_pillow_compat -- --nocapture` | **Passes**: phase-A dlopen ok, phase-B Pillow round-trip @ q=90 PSNR 49.49 dB (≥ 30 dB floor). Blocker-code-3 is now a hard panic, not a skip. | P0-3 closed. |
| `cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link -- --include-ignored --exact tjunittest_default_suite_passes` | Passes | The ignore on this test is stale and should be removed. |

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

**Remaining gap (separate follow-up):** byte-exact 12-bit transcode against upstream `jpegtran -copy all <op>`. Upstream preserves the source DHT verbatim (no `-optimize` regen), whereas our optimised path always rebuilds DHT from the rotated coefficient distribution — DHT bytes therefore always differ, and the entropy stream that follows differs as a consequence. Closure requires preserving source `dc_huffman_tables` / `ac_huffman_tables` (currently captured by `MarkerReader` but discarded when constructing `JpegCoefficients`) and re-emitting them when `data_precision > 8`. Tracked under "Suggested Order" item 5b.

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

### P1. Soft-Skip Compatibility Tests Hide Product Blockers

**Symptom:** Some compatibility tests document blockers but return success or remain ignored:

- `tests/capi_pillow_compat.rs` treats blocker code 3 as skip.
- `tests/capi_stock_tool_link.rs` is ignored even though it is the key drop-in gate.
- `crates/libjpeg-turbo-rs-capi/tests/tjunittest_link.rs::tjunittest_default_suite_passes` is ignored but passes when forced.
- `crates/libjpeg-turbo-rs-capi/tests/capi_stock_djpeg_e2e.rs` has a separate harness issue around generated headers and should not remain a dead ignored test.

**Acceptance:** Product-path compatibility failures must be real failures in CI. Only slow stress tests may stay ignored by default, and the ignore reason must say how CI exercises them elsewhere.

### P1. Legacy `tjLoadImage` / `tjSaveImage` Are Still Stubs — **CLOSED**

**Status (2026-04-28): closed.** `crates/libjpeg-turbo-rs-capi/src/legacy.rs` now exports `tjLoadImage` / `tjSaveImage` with the **handle-less** ABI that upstream `turbojpeg.h` actually publishes (no `tjhandle` argument; `flags & TJFLAG_BOTTOMUP` propagates to `TJPARAM_BOTTOMUP` on a temporary handle that wraps the call). The TJ3 forms in `crates/libjpeg-turbo-rs-capi/src/imageio.rs::tj3LoadImage8` and `tj3SaveImage8` route through `libjpeg_turbo_rs::load_image_from_bytes` / `save_bmp` / `save_ppm` and honour:

- BMP `TJPF_BGR` convention on load (R↔B swap when the file's native is `PixelFormat::Rgb`).
- BMP alpha-strip on save (RGBX/BGRX/RGBA/BGRA/XRGB/XBGR/ARGB/ABGR → 3-bpp before `save_bmp`).
- `TJPARAM_BOTTOMUP` on both load (post-decode flip) and save (pre-encode flip).
- TJPF format negotiation (identity match plus RGB↔BGR swap; non-trivial conversions still return a descriptive error).

`cargo test -p libjpeg-turbo-rs-capi --test legacy_aliases` passes 4/4 against the cdylib (`tj_load_image_reports_error_for_missing_file`, `tj_load_save_image_round_trip_ppm_through_legacy_alias`, plus `tjBufSize` and the init/destroy aliases).

### P1. `TJPARAM_PRECISION` Is Not Fully Honored Through TJ3 Compress Entry Points

**Symptom:** The Rust library has arbitrary precision lossless APIs, but the C ABI compress entry points need to match upstream dispatch:

- `tj3Compress8` honors lossless precision 2..8.
- `tj3Compress12` honors lossless precision 9..12.
- `tj3Compress16` honors lossless precision 13..16.

**Likely area:**

- `crates/libjpeg-turbo-rs-capi/src/compress.rs`
- `crates/libjpeg-turbo-rs-capi/src/precision.rs`
- `crates/libjpeg-turbo-rs-capi/src/tj3.rs`
- `src/api/precision.rs`

**Acceptance:** Add dlopen tests for precision 4, 10, and 14 and cross-check against upstream tools where available.

### P1. Encode SIMD Performance Gap On x86_64

**Symptom:** `experiments/x86_64_avx2_final_report.md` shows x86_64 AVX2 encode at `Rust/C` ratios:

| Benchmark | Rust (us) | C (us) | Rust/C |
|-----------|-----------|--------|--------|
| encode_320x240_420 | 370.8 | 306.0 | 1.21× |
| encode_640x480_422 | 1505.7 | 1293.9 | 1.16× |
| encode_1920x1080_420 | 11575.6 | 8502.9 | **1.36×** |
| encode_1920x1080_444 | 18033.8 | 15505.1 | 1.16× |

**Why this matters:** the README + CLAUDE.md commit to "equivalent or better performance". A drop-in replacement that regresses encode latency by 36 % at 1080p_420 is not a credible drop-in for any caller that profiles encode time (server JPEG pipelines, transcoding services, mobile capture). NEON encode is already 0.89–0.93× C — the gap is x86_64-specific.

**Identified hotspots:**

1. **Huffman encode SIMD** (~15–25 % of encode time) — C ships `simd/x86_64/jchuff-sse2.asm`. We have no Rust SIMD port. Estimated to bring 1080p_420 from 1.36× → ~1.10× (highest impact).
2. **H2V2 fused downsample+FDCT+quantize** — current AVX2 fused path covers H2V1 only.
3. **256-bit input-color load** — encode color path drops to 128-bit SSSE3 deinterleave.
4. **Progressive encode SIMD** — no Rust analogue of `jcphuff-sse2.asm`.

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

Every encode benchmark `Rust/C ≤ 1.05×`. Record before/after in `experiments/encode.tsv` per the keep/discard/crash protocol in `experiments/README.md`.

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
2. Harden gates by removing stale ignores and blocker-as-skip behavior (P1 Soft-Skip).
3. ~~Fix stock `djpeg` aborts (P0-2).~~ **CLOSED 2026-04-28** — `examples/stock_djpeg_cjpeg/run.sh` reports `OK all_byte_exact`.
4. ~~Add high-precision raw-data symbols and make Pillow load (P0-3).~~ **CLOSED 2026-04-28** — Pillow round-trip @ q=90 PSNR 49.49 dB.
5. Implement virtual coefficient-array materialization (and any libjpeg API symbol stock `jpegtran` resolves at runtime that the shim hasn't exported yet) for the stock `jpegtran` transform path (P0-4). **CLOSED 2026-04-28** — full TJXOP + crop + `-copy` cross-product byte-exact for 8-bit fixtures; 12-bit transcode (`monkey12`) routes through optimised Huffman, decodes pixel-equal through stock djpeg, and is no longer skipped by `examples/stock_djpeg_cjpeg/run.sh`.
5b. Preserve source DHT in `JpegCoefficients` so 12-bit transcode can byte-match upstream `jpegtran` (currently `pixel_equal_dht_differs`). Touchpoints: `JpegCoefficients` struct in `src/api/coefficient.rs`, `read_coefficients` (drop the DHT discard), and `write_coefficients_optimized` (prefer caller-supplied DHT over rebuild). Acceptance: `examples/stock_djpeg_cjpeg/run.sh` reports `jpegtran monkey12 pass byte_exact` (drop the `pixel_equal_dht_differs` allowance in run.sh).
6. ~~Fill legacy `tjLoadImage` / `tjSaveImage` and `tjEncodeYUV3` / `tjDecodeYUV`.~~ **CLOSED 2026-04-28** — handle-less load/save ABI with BMP TJPF_BGR + alpha-strip + bottom-up; YUV aliases forward through `tj3EncodeYUV8` / `tj3DecodeYUV8` with end-to-end 4:4:4 round-trip coverage in `legacy_aliases.rs`.
7. Wire arbitrary precision lossless through TJ3 compress.
8. Wire upstream `tjbench` / `rdjpgcom` / `wrjpgcom` against our shim (P2) — the apples-to-apples gate for step 9.
9. Close the x86_64 encode SIMD gap (P1 Encode) until every encode benchmark `Rust/C ≤ 1.05×`. Run only after correctness and compatibility are green.
10. PNG image I/O (P2), if downstream demand exists.

This order is intentionally strict. A replacement project should not optimize the encoder or add optional PNG support while stock tools abort and compatibility blockers are silently skipped. Encode perf stays tracked but deferred; PNG stays optional.

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
