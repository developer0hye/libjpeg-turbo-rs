# Phase 4 — Post-Gate Corrections (All CLOSED)

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). Open this file for gaps surfaced after the Phase 3 release gate was marked closed.

## Status summary

| ID | Status |
| --- | --- |
| P4-1 | CLOSED 2026-05-10 |
| P4-2 | CLOSED 2026-05-12 |
| P4-3 | CLOSED 2026-05-12 |
| P4-4 | CLOSED 2026-05-12 |
| P4-5 | CLOSED 2026-05-12 |
| P4-6 | CLOSED 2026-05-13 |
| P4-7 | CLOSED 2026-05-16 |

---

## P4-1. `jpeg_calc_jpeg_dimensions` Was Documented But Not Exported — **CLOSED 2026-05-10**

**Status (2026-05-10): closed.** `jpeg_calc_jpeg_dimensions` is now exported from `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`, re-exported from `src/lib.rs`, and removed from `tests/symbol_inventory.rs::allowlisted_missing_symbols()`.

**Root cause:** the C API reference and feature checklist marked the helper as supported, but the actual cdylib still had it in the missing-symbol allowlist. A new dlopen regression first failed with `symbol not found`, then passed after the implementation.

**Implementation:** `jpeg_calc_jpeg_dimensions(cinfo)` mirrors the upstream no-compression-scaling behavior in `references/libjpeg-turbo/src/jcmaster.c`: `jpeg_width` / `jpeg_height` are copied from `image_width` / `image_height`; `min_DCT_{h,v}_scaled_size` is set to 8 for lossy JPEG and 1 for lossless JPEG. `jpeg_start_compress` now uses the same helper path for its derived compression fields.

**Verification:**

- `cargo test -p libjpeg-turbo-rs-capi --test capi_jpeglib_encode c2_1_calc_jpeg_dimensions_sets_public_compress_fields -- --nocapture` → passed.
- `cargo test -p libjpeg-turbo-rs-capi --test symbol_inventory --release -- --nocapture` → passed; both upstream `jpeglib.h` and `turbojpeg.h` symbol inventories resolve.

## P4-2. Scheduled Decode Parity Regressions — **CLOSED 2026-05-12**

**Status (2026-05-12): closed.** The scheduled `Full C Parity` and `Fuzz Smoke` failures at `28954f6` are pinned by a focused `c_tjdecomptest` regression and a `fuzz_decode_diff_c` corpus seed.

**Root cause:** scaled 4:2:0 crop output computed chroma X offsets from raw sampling factors even when scaled IDCT had expanded chroma planes to full output width. Separately, `fuzz_decode_diff_c` compared Rust without progressive block smoothing against `djpeg`, whose default enables block smoothing for truncated progressive DC-only streams.

**Implementation:** crop offsets now scale from each decoded component plane width relative to the full scaled output width. The differential decode fuzzer enables block smoothing to match `djpeg` defaults and keeps the offending progressive fixture in the `fuzz_decode_diff_c` corpus.

**Verification:**

- `cargo test --test c_tjdecomptest -- --nocapture` → passed.
- `cargo test --features full-c-parity --test c_tjdecomptest c_tjdecomptest_full -- --nocapture` → passed.
- `cargo +nightly fuzz run fuzz_decode_diff_c fuzz/corpus/fuzz_decode_diff_c/prog_dc_smoothing_eoi_after_first_scan.jpg -- -runs=1` → passed.
- `cargo +nightly fuzz run fuzz_decode_diff_c -- -max_total_time=10 -print_final_stats=1` → passed.

## P4-3. Fuzz Smoke C Oracle Toolchain Drift — **CLOSED 2026-05-12**

**Status (2026-05-12): closed.** The post-merge scheduled `Fuzz Smoke` run at `611aea2` failed on `fuzz_decode_diff_c` because the workflow used Ubuntu's packaged libjpeg-turbo tools while the parity gates and local reference path use libjpeg-turbo 3.x.

**Root cause:** `fuzz-smoke.yml` installed `libjpeg-turbo-progs` from apt, which can lag the official release and produce different progressive block-smoothing output for malformed/truncated progressive streams. The fuzz target tool lookup also preferred `/usr/bin` over `/opt/libjpeg-turbo/bin`, so adding the official tools without changing lookup order would still risk selecting the older system oracle.

**Implementation:** `Fuzz Smoke` now installs the official libjpeg-turbo 3.1.4.1 Debian package for differential C targets. The fuzz target tool search order now prefers `/opt/libjpeg-turbo/bin` over `/usr/bin` for `djpeg` and `jpegtran` so CI and local oracle selection follow the intended 3.1.4.1 toolchain when both are present.

**Verification:**

- `cargo +nightly fuzz run fuzz_decode_diff_c fuzz/corpus/fuzz_decode_diff_c/prog_dc_smoothing_eoi_after_first_scan.jpg -- -runs=1` → passed locally with libjpeg-turbo 3.1.4.1.
- Scheduled workflow verification to run on branch `fix/fuzz-smoke-progressive-smoothing` before merge.

## P4-4. Full C Parity cjpeg x86 Padding Noise — **CLOSED 2026-05-12**

**Status (2026-05-12): closed.** The first post-P4-2 manual `Full C Parity` run reached `c_tjcomptest_lossy_full` on x86_64 and failed on a byte-exact comparison for the default-quality 4:2:0 RGB encode case.

**Root cause:** the full lossy matrix used upstream `testorig.ppm` (227x149), which has partial right/bottom MCUs for subsampled encodes. C cjpeg's padding behavior can differ by platform/toolchain for those partial MCUs, so byte-exact entropy comparison on that source is not a stable oracle. The quick lossy parity test already used a 96x96 MCU-aligned synthetic source for the same reason.

**Implementation:** `c_tjcomptest_lossy_full` now uses the same MCU-aligned synthetic RGB/gray source pattern as the quick matrix for 8-bit lossy byte-parity checks. The full matrix still covers the restart, arithmetic, default integer DCT, optimize, progressive, quality, subsampling, and inner variant axes without depending on partial-MCU padding bytes.

**Verification:**

- `cargo test --features full-c-parity --test c_tjcomptest c_tjcomptest_lossy_full -- --nocapture` → passed locally.
- Scheduled `Full C Parity` workflow verification to run on branch `fix/fuzz-smoke-progressive-smoothing` before merge.

## P4-5. Full C Parity Fast-DCT Byte Oracle Noise — **CLOSED 2026-05-12**

**Status (2026-05-12): closed.** After P4-4, the next x86_64 `Full C Parity` run reached the aligned-source matrix and failed on `lossy_full_p8_r0_qdef_a0_dc1_o0_p0_samp444_rgb_samp444` with a one-byte entropy-stream difference.

**Root cause:** `cjpeg -dc fa` selects the fast integer FDCT, which is an approximation and is not a byte-stable cross-platform oracle. The scheduled x86_64 run was building libjpeg-turbo 3.1.0 from source, while local/macOS validation used a different toolchain/backend. Requiring byte-identical entropy streams for every fast-DCT full-matrix case turns platform rounding noise into a release blocker.

**Implementation:** the scheduled full lossy cjpeg parity matrix now keeps byte-exact coverage on the default integer DCT. Focused cjpeg parity tests still cover selected fast-DCT cases that are byte-stable; the full scheduled matrix avoids treating fast-DCT approximation bytes as a portable C contract.

**Verification:**

- `cargo test --features full-c-parity --test c_tjcomptest` → passed locally.
- Scheduled `Full C Parity` workflow verification to run on branch `fix/fuzz-smoke-progressive-smoothing` before merge.

## P4-6. Transform Optimized-Huffman Fallback for Fuzzed Progressive Coefficients — **CLOSED 2026-05-13**

**Status (2026-05-13): closed.** Branch-level `Fuzz Smoke` run `25768874016` found `fuzz_transform_diff_c/crash-94087f99ddf1d878d1e3ae0cdbe0a5c98515111c`: a 16x16 progressive HFlip case where `jpegtran` produced a decodable transformed JPEG but Rust's non-optimized coefficient writer emitted entropy bytes that `djpeg` rejected with extraneous bytes before EOI.

**Root cause:** adversarial progressive inputs can decode into coefficient buffers whose baseline sequential entropy symbols exceed the standard Annex K Huffman table coverage. The non-optimized coefficient writer silently emitted zero-bit Huffman symbols for those out-of-range DC/AC categories, yielding an invalid JPEG. The optimized writer builds per-image Huffman tables and can encode the same coefficient buffer correctly.

**Implementation:** `transform_jpeg_with_options` now detects when the baseline standard Huffman tables cannot encode a transformed progressive-source coefficient buffer and routes that case through `write_coefficients_optimized`, including restart-marker DC predictor resets, matching the existing 12-bit precision forced-optimization path while preserving byte-exact default output for existing baseline fixtures.

**Verification:**

- `cargo test --test regression_transform_fuzz_progressive` → passed locally.
- `cargo test --test transform_small_image_byte_exact` → passed locally.
- `cargo +nightly fuzz run fuzz_transform_diff_c /private/tmp/libjpeg-fuzz-transform-artifact/crash-94087f99ddf1d878d1e3ae0cdbe0a5c98515111c -- -runs=1` → passed locally.

## P4-7. Block Smoothing All-or-Nothing Component Gate — **CLOSED 2026-05-16**

**Status (2026-05-16): closed.** Scheduled `Fuzz Smoke` run [25900537973](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/25900537973) (commit `1a33459a`) failed in `fuzz_decode_diff_c` on a 16x16 progressive 4:4:4 fixture (`crash-3eb4d5af274a456162b42f9a41700a07e57e0b46`, 488 bytes) with `max abs diff = 40` against the 24-byte tolerance. C `djpeg` produced uniform `[177, 133, 148]` while Rust produced a monotonically decreasing AC[1] gradient — Rust silently applied block smoothing where C disabled it.

**Root cause:** `src/decode/pipeline.rs::decode_progressive_planes` evaluated `smoothing_ok_for_component` per component and dispatched `apply_block_smoothing_coeffs` only for components that passed. C `decompress_smooth_data` (jdcoefct.c) treats `smoothing_ok` as an image-wide predicate — `start_output_pass` picks a single dispatch function for the whole image, so if any component's smoothing quant prerequisites fail (`qtable->quantval[Q02_POS]` etc. zero), C disables smoothing across all components and falls back to plain `decompress_data`. The crash fixture's Cb chroma table had `Q02 = Q03 = Q12 = Q21 = Q30 = 0`; C disabled smoothing on every plane, Rust only on Cb. Y and Cr therefore picked up phantom AC[1]/AC[10]/etc. predictions from neighbor DC values, yielding the gradient.

**Implementation:** rewrote the smoothing dispatch loop in `src/decode/pipeline.rs::decode_progressive_planes` to fold the per-component `smoothing_ok_for_component` results into a single `all_components_ok` predicate before any call to `apply_block_smoothing_coeffs`. Smoothing now runs on every component or none, matching `start_output_pass` semantics. Trace built from `references/libjpeg-turbo` (later restored) confirmed C's `smoothing_ok` returned 0 specifically because of the Cb zero-quant pattern.

**Verification:**

- `cargo test --test cross_check_fuzz_decode_diff_c_progressive_16x16 -- --nocapture` → `max_diff=0` byte-exact vs djpeg (pre-fix `max_diff=40`).
- `cargo test --test cross_check_progressive_scans` → 7 passed, 0 failed.
- `cargo test --lib` → 185 passed, 0 failed.
- Persisted seed at `fuzz/corpus/fuzz_decode_diff_c/regression-ci-25900537973-progressive-16x16-444` so `tests/generate_fuzz_seeds.rs` preserves the fixture across regenerations.

## Phase 4 Suggested Order

1. ~~**P4-1** — export `jpeg_calc_jpeg_dimensions` and delete its missing-symbol allowlist entry.~~ **CLOSED 2026-05-10**.
2. ~~**P4-2** — fix scheduled decode parity regressions from full C parity and fuzz smoke.~~ **CLOSED 2026-05-12**.
3. ~~**P4-3** — pin Fuzz Smoke's differential C oracle to libjpeg-turbo 3.x.~~ **CLOSED 2026-05-12**.
4. ~~**P4-4** — make full cjpeg parity use an MCU-aligned source.~~ **CLOSED 2026-05-12**.
5. ~~**P4-5** — keep full cjpeg byte parity on the default integer DCT.~~ **CLOSED 2026-05-12**.
6. ~~**P4-6** — route transform coefficient buffers beyond standard Huffman table coverage through optimized coding.~~ **CLOSED 2026-05-13**.
7. ~~**P4-7** — gate block smoothing on every component passing `smoothing_ok_for_component`, mirroring C's all-or-nothing `start_output_pass` dispatch.~~ **CLOSED 2026-05-16**.
