# Phase 4 — Post-Gate Corrections (All CLOSED)

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). Open this file for gaps surfaced after the Phase 3 release gate was marked closed.

## Status summary

| ID | Status |
| --- | --- |
| P4-1 | CLOSED 2026-05-10 |
| P4-2 | CLOSED 2026-05-12 |

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

## Phase 4 Suggested Order

1. ~~**P4-1** — export `jpeg_calc_jpeg_dimensions` and delete its missing-symbol allowlist entry.~~ **CLOSED 2026-05-10**.
2. ~~**P4-2** — fix scheduled decode parity regressions from full C parity and fuzz smoke.~~ **CLOSED 2026-05-12**.
