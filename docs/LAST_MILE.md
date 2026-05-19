# Last Mile Replacement Gate — Index

> **Purpose:** the release gate for "can `libjpeg-turbo-rs` replace C libjpeg-turbo as a system library?" — Rust-native APIs, TurboJPEG/TJ3 C ABI, classic `jpeg_*`, stock tools, downstream wrappers.

> **For agentic workers:** read this index first. Open the phase file(s) below ONLY for the items you are working on. Do not load all phases by default.

**Tech stack:** Rust 2021, `libjpeg-turbo-rs`, `libjpeg-turbo-rs-capi`, upstream sources in `references/libjpeg-turbo`, Homebrew/system `djpeg`/`cjpeg`/`jpegtran`, Pillow/ImageMagick smoke harnesses.

---

## Current Status (2026-05-19)

Project replacement readiness is tiered (see [README.md](../README.md) Replacement tiers):

- **T1.** Rust crate (`use libjpeg_turbo_rs::*;`) — **ready** today.
- **T2.** TurboJPEG cdylib (`libturbojpeg.so.0`) — **ready** for TJ3 consumers (opaque-handle API, no struct ABI risk). Legacy 1.x/2.x surface is partial: **21 legacy aliases wired** in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` (lifecycle `tjInitCompress`/`tjInitDecompress`/`tjInitTransform`/`tjDestroy` + `tjCompress2`/`tjDecompress2`/`tjDecompressHeader3` + `tjTransform`/`tjEncodeYUV3`/`tjDecodeYUV` + buffer-size helpers + `tjLoadImage`/`tjSaveImage` + `tjGetErrorStr2`); **18 still allowlisted-missing** (v1 / un-versioned variants like `tjAlloc`, `tjFree`, `tjCompress`, `tjGetScalingFactors`) per P4-18 (filed 2026-05-18). Consumers compiled against TJ 1.x/2.x headers that touch the missing 18 need P4-18 closure (implement or document-as-permanently-deferred) before T2 is honest for them.
- **T3.** Classic libjpeg v8 cdylib (`libjpeg.so.8`) — **ready** for v8 consumers. The C ABI shim now defaults to `libjpeg.so.8` / `@rpath/libjpeg.8.dylib` (P4-3, 2026-05-17). System-library drop-in (Phase 2) closed; long-tail C-compatibility (Phase 3) fully closed; classic state-machine pathological coverage closed under P4-5 with a real-suspension follow-up filed as P4-17 (2026-05-18). Live divergences from upstream contract surface are tracked as P4-13 (streaming `jpeg_consume_input`) and P4-14 (C-side `max_memory_to_use` enforcement). P4-16 (per-`cinfo` thread-affinity) was closed 2026-05-19 via Option B — the per-thread ownership contract is now documented authoritatively in `docs/ABI_COMPATIBILITY.md`. None block T3 readiness; the remaining two surface before T4 can be honestly framed.
- **T4.** System v6b/v7 drop-in (`libjpeg.so.62` / `libjpeg.so.7`) — **explicit non-goal** until per-ABI cdylib matrix ships. v6b is available only behind explicit `CAPI_SONAME=libjpeg.so.62 + CAPI_ACK_V6B_SONAME=1` opt-in.

Phase 3 history: P3-1 / P3-2 / P3-3 / P3-4 / P3-5 / P3-6 all CLOSED. P3-2 closed 2026-05-09 with the full 12-bit `jpeg_*_raw_data` backend wired through `libjpeg_turbo_rs::raw_data_12::{compress,decompress}_raw_12`. Phase 4 post-gate corrections: P4-1 exported `jpeg_calc_jpeg_dimensions` (closed 2026-05-10); P4-2 introduced the T1–T4 replacement-tier framing (closed 2026-05-17); P4-3 flipped the default C-ABI SONAME from `libjpeg.so.62` to `libjpeg.so.8` (closed 2026-05-17).

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

Filed 2026-05-18 after an independent cold review (Claude + codex, two-pass) surfaced four C-ABI fidelity gaps the prior Phase-4 closures missed and a falsifying flaw in the P4-5 closure's primary suspension test:

| ID | Heading | Phase | State | Note |
| --- | --- | --- | --- | --- |
| P4-13 | [`jpeg_consume_input` returns EOI instead of honoring per-byte source suspension](last_mile/phase4.md#p4-13-jpeg_consume_input-returns-eoi-instead-of-honoring-per-byte-source-suspension--open) | 4 | OPEN | Streaming/suspension contract divergence; documented at `jpeglib.rs:4234-4238` in code. |
| P4-14 | [`max_memory_to_use` is ABI-mirrored but not enforced in the C-side allocation path](last_mile/phase4.md#p4-14-max_memory_to_use-is-abi-mirrored-but-not-enforced-in-the-c-side-allocation-path--open) | 4 | OPEN | Field at correct offset, but zero comparisons against it anywhere in `memmgr.rs`. |
| P4-17 | [`source_mgr_suspends_every_byte` test exercises chunked-refill, not real suspension](last_mile/phase4.md#p4-17-source_mgr_suspends_every_byte-test-exercises-chunked-refill-not-real-suspension--open) | 4 | OPEN | P4-5's primary pattern returns TRUE after one byte; a real `JPEG_SUSPENDED` test is still missing. |
| P4-18 | [18 legacy TurboJPEG 1.x/2.x symbols remain allowlisted-missing](last_mile/phase4.md#p4-18-18-legacy-turbojpeg-1x2x-symbols-remain-allowlisted-missing--open) | 4 | OPEN | 21 legacy TJ aliases wired in `legacy.rs` (v2/v3 + buffer/image helpers); `symbol_inventory.rs:190-207` allowlists 18 still missing (`tjAlloc`/`tjFree`/`tjCompress`/`tjGetScalingFactors`/…). P3-3 scoped these out as "non-blocking" under the TJ3 framing. Belongs to **T2** (TurboJPEG cdylib) — blocks only consumers that call any of the 18 missing functions. |

P4-15 (`jpeg16_*_raw_data` parity audit) was filed and closed-as-N/A in the same 2026-05-18 pass — upstream `jpeglib.h:1039-1041` / `:1096-1098` declares raw-data only for 8/12-bit precision, so our omission mirrors theirs.

History: P3-2 closed 2026-05-09 — `jpeg12_write_raw_data` / `jpeg12_read_raw_data` wired through the real 12-bit backend, pinned by `tests/capi_jpeg12_raw_data_round_trip.rs`. P4-1 closed 2026-05-10 — `jpeg_calc_jpeg_dimensions` exported. P4-2..P4-12 closed 2026-05-17 — T1–T4 tier framing, default SONAME flip to `libjpeg.so.8`, panic guard on all 154 C-ABI entry points (count re-verified 2026-05-18 with strict regex), classic state-machine pathological harness infrastructure + first pattern (note P4-17 follow-up), FEATURE_PARITY wording reconciliation, stale-stub sweep, runtime x86_64 BMI1+LZCNT dispatch audit + README correction, strided/zero-copy architectural filing, downstream compatibility lab filing, OSS-Fuzz project files prepared for upstream submission, hard-case q∈{98,99,100} parity tests. P4-15 closed 2026-05-18 (N/A, mirrors upstream). P4-16 closed 2026-05-19 via Option B — per-thread `cinfo` ownership contract documented authoritatively in `docs/ABI_COMPATIBILITY.md` ("Threading contract" section).

**Next up**: drive P4-13 / P4-14 / P4-17 / P4-18 to closure as adoption pressure surfaces. Long-term backlog items live as P2-F (strided zero-copy refactor), P2-G (downstream lab uplift Qt+OpenCV), P2-H (OSS-Fuzz upstream PR + C-harness sanitizer), P2-I (remaining hard-case parity classes), and the per-ABI cdylib matrix / production packaging / no_std subset / security ops policy / cost-of-feature benchmarks — all tracked in the master plan at `/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md`. None of the remaining P4-13/14/17/18 items block the current T3 release gate; they document divergences from upstream's contract surface that downstream production adoption will need addressed before the T4 (system v6b/v7 drop-in) work can be honestly framed.

---

## Phase Map

Each phase file is self-contained. Read only the one you need.

| Phase | File | Scope | Status |
| --- | --- | --- | --- |
| **Phase 1** | [last_mile/phase1.md](last_mile/phase1.md) | Original release gate: P0-1..4, P1 (Soft-Skip / Encode SIMD / Legacy / Precision / YUV), Phase-1 P2 (tjbench / PNG), Execution Plan (Tasks 1-7), Definition of Done. | All CLOSED — historical reference. |
| **Phase 2** | [last_mile/phase2.md](last_mile/phase2.md) | System-library drop-in hardening: P2-1..11 (workflow flags, printf expansion, ABI cross-check, symbol inventory, install layout, fuzzing, distro consumers, progressive-encode samp411, crates.io publish). | All CLOSED. |
| **Phase 3** | [last_mile/phase3.md](last_mile/phase3.md) | Long-tail C compatibility: P3-1 (ABI offset cross-check), P3-2 (12-bit raw-data backend), P3-3 (symbol-inventory allowlist triage), P3-4 (4-pixel chroma transform gate), P3-5 (classic lifecycle harness), P3-6 (non-standard sampling / RGB565). | All CLOSED. |
| **Phase 4** | [last_mile/phase4.md](last_mile/phase4.md) | Post-gate corrections surfaced after Phase 3: P4-1..P4-12 (jpeg_calc_jpeg_dimensions export, T1–T4 tier framing, SONAME flip to `libjpeg.so.8`, panic guard on 154 entry points, pathological-lifecycle harness, FEATURE_PARITY wording, stub/divergence sweep, x86_64 BMI1+LZCNT dispatch audit, strided/zero-copy filing, downstream-lab filing, OSS-Fuzz project ready, hard-case q∈{98,99,100} parity). P4-13..P4-18 filed 2026-05-18 after cold-review pass; P4-15 closed as N/A; P4-16 closed 2026-05-19 via Option B (documented thread-affinity contract). | P4-1..P4-12 + P4-15 + P4-16 CLOSED; P4-13/14/17/18 OPEN. |
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
