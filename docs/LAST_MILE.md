# Last Mile Replacement Gate — Index

> **Purpose:** the release gate for "can `libjpeg-turbo-rs` replace C libjpeg-turbo as a system library?" — Rust-native APIs, TurboJPEG/TJ3 C ABI, classic `jpeg_*`, stock tools, downstream wrappers.

> **For agentic workers:** read this index first. Open the phase file(s) below ONLY for the items you are working on. Do not load all phases by default.

**Tech stack:** Rust 2021, `libjpeg-turbo-rs`, `libjpeg-turbo-rs-capi`, upstream sources in `references/libjpeg-turbo`, Homebrew/system `djpeg`/`cjpeg`/`jpegtran`, Pillow/ImageMagick smoke harnesses.

---

## Current Status (2026-05-09)

Project is **replacement-ready** for the Rust-application + stock-tool drop-in story. System-library drop-in (Phase 2) is closed. Long-tail C-compatibility (Phase 3) is fully closed: P3-1 / P3-2 / P3-3 / P3-4 / P3-5 / P3-6 are all CLOSED. P3-2 closed 2026-05-09 with the full 12-bit `jpeg_*_raw_data` backend wired through `libjpeg_turbo_rs::raw_data_12::{compress,decompress}_raw_12`. Phase 4 currently has six closed post-gate corrections: P4-1 exported `jpeg_calc_jpeg_dimensions`; P4-2 fixed scheduled full-parity and fuzz-smoke decode regressions; P4-3 pinned Fuzz Smoke's C oracle to libjpeg-turbo 3.x; P4-4 removed platform-specific padding noise from full cjpeg parity; P4-5 removed fast-DCT byte-oracle noise from scheduled full cjpeg parity; P4-6 fixed transform re-encoding of fuzzed progressive coefficients that require optimized Huffman tables.

**Live gate** (refresh whenever the inventory changes):

| Check | Result |
| --- | --- |
| `cargo test --workspace --release` | **Passes** — 2161 tests, 0 failures, 0 ignored. |
| `cargo test -p libjpeg-turbo-rs --test cross_product_transform` | **Passes** all 12 cases. P0-1 closed. |
| `cargo test -p libjpeg-turbo-rs --test regression_progressive_4pixel_chroma_transform` | **Passes** 256 cases byte-exact vs `jpegtran -progressive -copy all <op>`. P3-4 closed. |
| `cargo test --test cross_check_p3_6_nonstandard_rgb565` | **Passes** 4 fixtures: 3x2 decode (vs `djpeg`), 3x2 encode (vs `cjpeg -sample 3x2,1x1,1x1` + `djpeg`), 3x1 decode, RGB565 merged-upsample (vs `djpeg -nosmooth` + 5-6-5 truncate chain). P3-6 closed. |
| `cargo test --release --features full-c-parity --test c_tjtrantest c_tjtrantest_full` | **Passes** — 12,230 tested, 18,498 skipped (skip set covers unrelated tjtrantest exclusions, not P3-4). |
| `examples/stock_djpeg_cjpeg/run.sh` | `OK all_byte_exact` — every fixture byte-exact vs stock djpeg/cjpeg/jpegtran. P0-2 + P0-4 closed. |
| `cargo test --test capi_stock_tool_link` | **Passes** for djpeg/cjpeg/jpegtran + full TJXOP cross-product. |
| `cargo test --test capi_pillow_compat` | **Passes** — Pillow round-trip @ q=90 PSNR 49.49 dB. P0-3 closed. |
| `cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link` | **Passes** without `--include-ignored`. |
| `cargo test -p libjpeg-turbo-rs-capi --test abi_offsets --release` | **Passes** all 6 cross-checks: `jpeg_decompress_struct` (P2-4) + `jpeg_marker_struct` + `jpeg_compress_struct` + `jpeg_error_mgr` + `jpeg_source_mgr` + `jpeg_destination_mgr` against upstream `jpeglib.h` at `JPEG_LIB_VERSION=80`. P3-1 closed. |
| `cargo test -p libjpeg-turbo-rs-capi --test libtiff_integration --release` | **Passes** end-to-end libtiff COMPRESSION_JPEG round-trip via the cdylib (skips with reason if libtiff/cc are absent). The shim's `jpeg_read_header` walks markers manually to detect tables-only abbreviated datastreams (libjpeg.txt §6) and splices the cached prefix in front of each strip's body so `Decoder::new` parses the unified stream. Previously `#[ignore]`d as a known shim gap; un-ignored 2026-05-09. |

---

## Replacement Gate (7-item checklist)

Do not call the project a libjpeg-turbo C replacement until all are true:

1. `cargo test --workspace --no-fail-fast` is green with no product-path ignored tests except explicitly slow release-only stress tests.
2. `cargo test --test capi_stock_tool_link -- --include-ignored` is green, or the ignored attributes are removed and the default run is green.
3. `cargo test --test capi_pillow_compat -- --nocapture` fails on blocker code 3 until fixed, then passes through real Pillow decode+encode.
4. `cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link -- --include-ignored --exact tjunittest_default_suite_passes` is unignored and green in the normal suite.
5. Stock `djpeg`, `cjpeg`, `jpegtran` built from `references/libjpeg-turbo/src` and linked to the shim produce output that is byte-identical or explicitly pixel-identical where byte-identical is not a valid C contract.
6. The shim exports every symbol required by the linked stock tools and by the Pillow/ImageMagick smoke harnesses, including high-precision raw-data entry points.
7. Performance reporting is re-run after correctness is green. Correctness blockers take priority over decode/encode microbenchmarks.

---

## Open Items

No OPEN or PARTIAL items left in Phase 3 or Phase 4. P3-2 closed 2026-05-09 — `jpeg12_write_raw_data` / `jpeg12_read_raw_data` are wired through the real 12-bit backend, pinned by `tests/capi_jpeg12_raw_data_round_trip.rs` (3 gates: shim-encode→native-decode, native-encode→shim-decode, shim-encode→shim-decode). P4-1 closed 2026-05-10 — `jpeg_calc_jpeg_dimensions` is now exported and covered by `capi_jpeglib_encode` + `symbol_inventory`.

**Next up**: nothing scheduled. The release gate is satisfied for the standard-sampling / classic-lifecycle / lossless-transform / non-standard-sampling / 12-bit-raw-data consumer surfaces. Future phases live in `docs/last_mile/phase5.md` or later if downstream surfaces another gap.

---

## Phase Map

Each phase file is self-contained. Read only the one you need.

| Phase | File | Scope | Status |
| --- | --- | --- | --- |
| **Phase 1** | [last_mile/phase1.md](last_mile/phase1.md) | Original release gate: P0-1..4, P1 (Soft-Skip / Encode SIMD / Legacy / Precision / YUV), Phase-1 P2 (tjbench / PNG), Execution Plan (Tasks 1-7), Definition of Done. | All CLOSED — historical reference. |
| **Phase 2** | [last_mile/phase2.md](last_mile/phase2.md) | System-library drop-in hardening: P2-1..11 (workflow flags, printf expansion, ABI cross-check, symbol inventory, install layout, fuzzing, distro consumers, progressive-encode samp411, crates.io publish). | All CLOSED. |
| **Phase 3** | [last_mile/phase3.md](last_mile/phase3.md) | Long-tail C compatibility: P3-1 (ABI offset cross-check), P3-2 (12-bit raw-data backend), P3-3 (symbol-inventory allowlist triage), P3-4 (4-pixel chroma transform gate), P3-5 (classic lifecycle harness), P3-6 (non-standard sampling / RGB565). | All CLOSED. |
| **Phase 4** | [last_mile/phase4.md](last_mile/phase4.md) | Post-gate corrections surfaced after Phase 3: P4-1..P4-6. | All CLOSED. |
| **Reference** | [last_mile/reference_commands.md](last_mile/reference_commands.md) | Common verification commands (workspace test, stock-tool build, encode bench matrix, etc.). | — |

---

## Phase 3 Suggested Order

(Phase 1 + Phase 2 suggested orders are inside the respective phase files.)

1. ~~**P3-1** — extend `tests/abi_offsets.rs` to compress / error_mgr / source_mgr / destination_mgr / marker.~~ **CLOSED 2026-05-08** — all six originally-planned struct cross-checks active in `tests/abi_offsets.rs` (decompress + marker + compress + error_mgr + source_mgr + destination_mgr; 133 fields + 6 sizeof probes). C-side `main` field maps to Rust mirror `main_ctrl` via `(c_field_name, rust_offset)` tuple (only field-name divergence). `jvirt_*_control` opaque upstream → no cross-check needed. Windows MSVC matrix leg deferred as Phase 4 hardening (helper currently emits gcc/clang flags only); `cargo test … abi_offsets --release` reports `6 passed; 0 failed; 0 ignored` on macOS aarch64 + the `abi-offsets` CI matrix.
2. ~~**P3-3** — audit the symbol inventory allowlist and keep only non-blocking legacy TJ aliases.~~ **CLOSED 2026-05-06; corrected 2026-05-10**.
3. ~~**P3-2** — `jpeg12_*_raw_data` backend.~~ **CLOSED 2026-05-09** — both entry points wired through `libjpeg_turbo_rs::raw_data_12::{compress,decompress}_raw_12`; round-trip pinned by `tests/capi_jpeg12_raw_data_round_trip.rs`.
4. ~~**P3-5** — classic `jpeglib.h` lifecycle / custom-I/O / suspension C harness (≥ 8 tests).~~ **CLOSED 2026-05-08** — all 8 patterns active (custom src/dst mgr, source suspension, destination-suspension `JERR_CANT_SUSPEND` contract, abort+reuse for both, buffered-image multi-pass, setjmp/longjmp). Two shim defects uncovered + fixed: pattern #8 (2026-05-07) made `jpeg_read_header` invoke `error_exit` on EOI-terminated malformed input; pattern #4 (2026-05-08) made `push_bytes_through_dest_mgr` invoke `error_exit` with `JERR_CANT_SUSPEND` instead of silently dropping bytes when a custom dst mgr returns `FALSE` (the deferred-encode shim cannot honor upstream's per-MCU suspension at `jpeg_write_scanlines`).
5. ~~**P3-4** — lift the 4-pixel chroma transform writer gate.~~ **CLOSED 2026-05-07** — gate at `transform_jpeg_with_options::progressive_safe` widened from `max_{h,v} ≤ 2` to `max_{h,v} ∈ {1,2,4}` (the eight standard TJSAMP factors); non-standard 3x sampling stays on baseline pending P3-6. Regression pinned in `tests/regression_progressive_4pixel_chroma_transform.rs` (256 cases, all byte-equal vs `jpegtran -progressive -copy all <op>`); `c_tjtrantest_full` runs 12,230 cases without divergence.
6. ~~**P3-6** — non-standard sampling / RGB565 merged-upsample minimum fixture set.~~ **CLOSED 2026-05-08** — 4 fixtures (3x2 decode, 3x2 encode, 3x1 decode, RGB565 merged-upsample) green in `tests/cross_check_p3_6_nonstandard_rgb565.rs`. Shim fix: merged-upsample gate widened from `Rgb` to `Rgb || Rgb565` with a 5-6-5 truncation pass after the merged kernel.

---

## Maintenance Protocol

The mechanics for adding/closing/archiving entries are documented in the project root `CLAUDE.md` under "LAST_MILE Management" — read that before editing any of these files.
