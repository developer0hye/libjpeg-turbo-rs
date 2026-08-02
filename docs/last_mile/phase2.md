# Phase 2 — System-Library Drop-In Hardening (All CLOSED)

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). Open this file only when reading P2-* history.

The Phase 1 gates (P0-1..4, P1-Soft-Skip, P1-Encode, P1-Legacy, etc.) closed the *Rust-application replacement* and *stock-tool drop-in* stories. Phase 2 closed the remaining gap to **system-library** replacement: shipping a `libjpeg.so.62` / `libturbojpeg.so.0` SONAME-compatible binary that arbitrary distro packages (Pillow, ImageMagick, libvips, SDL_image, FFmpeg, GraphicsMagick, GD, …) can link against without source changes.

External cross-check on 2026-05-04: the analysis "Rust app library = ready; system-library replacement = not yet" was consistent with the live state of this repo at that time. Items the external review flagged that were *already closed* (`tj3GetICCProfile` / `tj3TransformBufSize` exports, `jpeg_set_marker_processor` wiring, `JpegSourceMgr` suspension semantics, `capi_imagemagick_compat` / `capi_pillow_compat` harnesses) are intentionally not re-listed here.

---

## P2-1. Full C Parity Workflow Soft-Skips — **CLOSED**

**Status (2026-05-04): closed.** Both `continue-on-error: true` flags in `.github/workflows/full-c-parity.yml` are gone:

- `c_tjcomptest_full` — flag removed when P2-11 closed (samp411/441/410/24 progressive parity).
- `c_tjtrantest_full` — flag removed in this commit. Local run on aarch64 macOS reports `11190 tested, 17538 skipped, 0 failed` for the full transform matrix (grayscale and non-grayscale combos). The "Known failures: grayscale Huffman diff" annotation that the flag carried was stale — earlier work that fixed the underlying divergence never updated the CI flag.

The `tests/c_tjtrantest.rs` source-level skip for `progressive + 4-pixel chroma + non-grayscale` was kept here as a Phase 2 follow-up. **Closed 2026-05-07** under [P3-4](phase3.md#p3-4-4-pixel-chroma-progressive-transform-writer-gate--closed-2026-05-07): the gate in `transform_jpeg_with_options::progressive_safe` was widened from `max_h ≤ 2 && max_v ≤ 2` to `max_{h,v} ∈ {1,2,4}` (the eight standard TJSAMP factors), and the source-level skip was deleted; full `c_tjtrantest_full` now runs 12,230 cases without divergence. Non-standard 3x sampling stays gated to baseline pending P3-6.

**Risk note:** flag removal was based on aarch64 macOS local results. Linux x86_64 (the other workflow leg) has not been re-verified locally because we don't have a cross-builder. The workflow runs weekly on Mondays; if the next run reds, react then. Re-introducing the flag would be the wrong response: either the bytes match or the test is wrong; either way we want to see the actual failure, not silence it.

## P2-2. `default_format_message` Printf Expansion — **CLOSED**

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

## P2-3. Per-Platform ABI Validation — **CLOSED 2026-05-10**

**Status (2026-05-10): closed.** The release-blocking P2-3 scope is now covered by the runtime ABI cross-check in `crates/libjpeg-turbo-rs-capi/tests/abi_offsets.rs` plus the dedicated `capi-abi-checks` CI matrix in `.github/workflows/ci.yml`. P3-1 expanded that gate from the original decompressor-only probe to all six public classic-API structs (`jpeg_decompress_struct`, `jpeg_marker_struct`, `jpeg_compress_struct`, `jpeg_error_mgr`, `jpeg_source_mgr`, `jpeg_destination_mgr`) with per-field `offsetof` checks and `sizeof(struct ...)` checks against upstream `jpeglib.h` at `JPEG_LIB_VERSION=80`.

The remaining platform ideas are future hardening, not OPEN/PARTIAL release-gate work:

1. **Windows/MSVC compile-time `const_assert!` blocks.** The runtime `abi_offsets` gate is platform-aware and compares Rust's host layout to a C harness compiled on the same host. Dedicated Windows compile-time constants would be additive.
2. **32-bit ABI targets** (`i686-pc-windows-msvc`, `i686-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf`). Add these only when a downstream consumer requests that platform; a useful gate needs a matching target C harness, not just a cross-compile build. **Trigger fired 2026-07-30 for `armv7-unknown-linux-gnueabihf`** — issue [#424](https://github.com/developer0hye/libjpeg-turbo-rs/issues/424) asked about ARMv7 performance, and per the rule below the follow-up is Phase 4's [P4-78](phase4.md#p4-78-no-32-bit-arm-aarch32-neon-backend--armv7-is-our-widest-gap-vs-c--open). What landed with it is the codec-side 32-bit gate (`Test (linux-armv7 scalar, emulated)`, cross-build + `qemu-arm`); the **C-ABI** half of this item — a matching armhf C harness for `abi_offsets` — is still untaken, and the `i686-*` pair remains untriggered.

If either future-hardening trigger becomes concrete, open a Phase 4 item instead of reopening this Phase 2 release gate.

## P2-4. Generated C-Side ABI Cross-Check — **CLOSED (jpeg_decompress_struct)**

**Status (2026-05-04): closed for `jpeg_decompress_struct`.** `crates/libjpeg-turbo-rs-capi/tests/abi_offsets.rs` synthesises a minimal `jconfig.h` (`JPEG_LIB_VERSION 80` + the upstream `WITH_JPEG8` defaults), writes a tiny C program that calls `offsetof(struct jpeg_decompress_struct, FIELD)` for every field that `jpeglib.rs:4096+` const-asserts, compiles it against the submodule's `references/libjpeg-turbo/src/jpeglib.h`, runs the binary, and asserts each emitted offset equals `std::mem::offset_of!(JpegDecompressPublic, FIELD)`.

**Coverage today (27 fields):** `err`, `mem`, `progress`, `client_data`, `is_decompressor`, `global_state`, `src`, `image_width`, `image_height`, `num_components`, `jpeg_color_space`, `out_color_space`, `scale_num`, `scale_denom`, `output_gamma`, `buffered_image`, `raw_data_out`, `quantize_colors`, `coef_bits`, `quant_tbl_ptrs`, `dc_huff_tbl_ptrs`, `ac_huff_tbl_ptrs`, `data_precision`, `comp_info`, `is_baseline`, `progressive_mode`, `arith_code`. Anything `jpeglib.rs` later const-asserts should be appended to `rust_offsets()` in lockstep.

**TDD-verified:** changing `JPEG_LIB_VERSION 80` → `JPEG_LIB_VERSION 70` in the harness's `jconfig.h` makes `cc` reject the program with `error: no member named 'is_baseline' in 'jpeg_decompress_struct'` — the test correctly red-fails when the C-side layout diverges from Rust's expectation. Restoring `80` returns to GREEN.

**Skip-with-reason cases:**
- non-LP64 / Windows host (matches the Rust assertion block's gate),
- `cc` not on PATH or not runnable,
- submodule not initialised (`references/libjpeg-turbo/src/jpeglib.h` missing),
- environmental cc failure (missing system headers, broken cross-compile setup).

**Out of scope (not blocking closure):** extending the cross-check to `jpeg_compress_struct`, `jpeg_error_mgr`, `jpeg_source_mgr`, `jpeg_destination_mgr`, `jvirt_barray_control`, `jvirt_sarray_control`, `jpeg_marker_struct`. The infrastructure is in place — adding more types is repeating the existing pattern with a different `struct foo` name in the C harness and a different Rust type in `rust_offsets()`. Tracked as a follow-up in [P3-1](phase3.md#p3-1-abi-offset-cross-check-was-decompress-only--partial-decompress--marker-cross-checked-4-large-structs-deferred) rather than a blocker because field-order drift in `jpeg_decompress_struct` is by far the highest-risk surface.

## P2-5. Symbol-Export Inventory Diff — **CLOSED**

**Status (2026-05-04): closed.** `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs` now parses the submodule's `references/libjpeg-turbo/src/jpeglib.h` for `EXTERN(...)` declarations (66 found) and `references/libjpeg-turbo/src/turbojpeg.h` for `DLLEXPORT` declarations (79 found), then dlopens our cdylib and asserts every parsed symbol is resolvable.

**Allowlist of intentionally-deferred symbols** (corrected 2026-05-10 — see [P3-3](phase3.md#p3-3-symbol-inventory-allowlist-triage--closed-2026-05-06) and [P4-1](phase4.md#p4-1-jpeg_calc_jpeg_dimensions-was-documented-but-not-exported--closed-2026-05-10)). `jpeg_calc_jpeg_dimensions` is now implemented and removed from the allowlist. The legacy TurboJPEG 1.x/2.x names below remain allowlisted unless a concrete downstream binary requires the pre-TJ2 ABI. Historical list:

- `jpeg_calc_jpeg_dimensions` — companion to `jpeg_calc_output_dimensions`; **implemented 2026-05-10 under P4-1**.
- `tjAlloc` / `tjFree` — superseded by `tj3Alloc` / `tj3Free`.
- `tjCompress` / `tjCompressFromYUV` / `tjCompressFromYUVPlanes` — superseded by `tjCompress2` (already exported) and the TJ3 forms.
- `tjDecompress` / `tjDecompressHeader` / `tjDecompressHeader2` / `tjDecompressToYUV` / `tjDecompressToYUV2` / `tjDecompressToYUVPlanes` — superseded by `tjDecompress2` / `tjDecompressHeader3` and the TJ3 forms.
- `tjEncodeYUV` / `tjEncodeYUV2` / `tjEncodeYUVPlanes` / `tjDecodeYUVPlanes` — superseded by `tjEncodeYUV3` / `tjDecodeYUV` and the TJ3 forms.
- `tjGetErrorCode` / `tjGetErrorStr` / `tjGetScalingFactors` — superseded by the TJ3 forms.

**The contract**: the test passes when the cdylib exports every upstream symbol *except* allowlisted ones. Removing a name from the allowlist signals "this is now implemented; the test should hold us to it from this commit on." The test thus sharpens to "no NEW gaps may appear."

**CI hookup**: bundled with P2-4 in the `capi-abi-checks` matrix job (`.github/workflows/ci.yml`) so all three ABI flavours (Linux x86_64 LP64, macOS aarch64 LP64, Windows MSVC LLP64) run the symbol diff on every PR.

**Out of scope (deferred):**

- Comparing against an *installed* upstream `libjpeg.so.62` / `libturbojpeg.so.0` via `nm -D` — would catch symbol-version tags and platform-specific export differences, but requires upstream installed at test time. The header-based check is the cheaper baseline and runs everywhere.
- SONAME match (`libjpeg.so.62` ↔ `libjpeg.so.8`) — owned by P2-9 (build.rs SONAME wiring + warning) and `docs/ABI_COMPATIBILITY.md`.

## P2-6. Crate Is `publish = false` — **CLOSED 2026-05-06**

**Status (2026-05-06): closed.** Both `libjpeg-turbo-rs` (root) and `libjpeg-turbo-rs-capi` are live on crates.io:

- `cargo add libjpeg-turbo-rs` → 0.6.1 (latest, published with v0.6.1 tag)
- `cargo add libjpeg-turbo-rs-capi` → 0.1.0 (published alongside v0.6.1)

The release workflow at `.github/workflows/release.yml` covers `v*` (root + npm + capi), `wasm-v*` (npm only), and `capi-v*` (capi only) tag patterns; v0.6.1 publish job ran successfully on 2026-05-06 across all three publish steps (root crate, npm, capi crate).

## P2-7. Differential / Roundtrip Fuzzing Against C — **CLOSED (decode + encode + transform); 24-hour long-run deferred**

**Status (2026-05-04): closed for the structural deliverables.** All three differential libfuzzer targets land and join the nightly matrix; the 24-hour scheduled long-run + OSS-Fuzz corpus publishing remain as a future scaling step (the 10-min nightly is the structural baseline).

**Done:**

- `fuzz/fuzz_targets/fuzz_decode_diff_c.rs` — feeds each fuzzed input to both `Decoder::decode` and a subprocessed `djpeg`, then asserts (a) acceptance agreement: when C accepts, Rust must accept too (drop-in floor); (b) dimension agreement; (c) pixel agreement within ±16 per byte (IDCT precision noise — curated `corpus_test` enforces byte-exact). Lenient direction (Rust accepts more) is allowed by design. Arithmetic-coded inputs (SOF9/10/11) are skipped — fuzzer-generated streams can carry coefficients whose dequantized magnitudes vastly exceed normal 8-bit JPEG ranges, and djpeg's default `-dct int` IDCT overflows them differently than ours. Coefficient-level correctness for arithmetic is verified byte-exact against djpeg's `jpeg_read_coefficients` by the curated conformance suites; root cause = IDCT integer overflow on malformed inputs, not a decoder bug (see follow-up below).
- `fuzz/fuzz_targets/fuzz_encode_diff_c.rs` — encodes a fuzz-supplied pixel buffer via Rust (`compress` / `compress_progressive` / `compress_arithmetic` / `compress_arithmetic_progressive`), then verifies that both Rust and C `djpeg` decode the result equivalently. Catches "Rust encoder produces output C consumer rejects" — the mirror of the decode-side differential.
- `fuzz/fuzz_targets/fuzz_transform_diff_c.rs` — applies HFlip / VFlip / Rot180 (the three ops that don't require MCU alignment) via both Rust `transform_jpeg_with_options` and subprocessed `jpegtran`, decodes both transformed JPEGs through `djpeg`, and asserts pixel agreement. Transpose / Transverse / Rot90 / Rot270 are out of scope (require MCU alignment, covered by curated `examples/corpus_test.rs` instead).
- All three targets wired into `.github/workflows/fuzz-smoke.yml` matrix (10 min nightly each). The `libjpeg-turbo-progs` install step now fires for any of the `*_diff_c` targets.
- Pre-existing baseline: `examples/corpus_test.rs` already runs decode + encode + transform differential against C `djpeg` / `cjpeg` / `jpegtran` for every fixture in `tests/corpus/` on every PR (`.github/workflows/ci.yml::test-corpus`).

**Subprocess vs in-process FFI:** all three targets call C tools via `std::process::Command` rather than linking C libjpeg into the harness. Slower per-iteration (~ms vs μs) but avoids dragging `cc-rs` + system libjpeg into the fuzz crate. In-process FFI is tracked as a follow-up; the throughput delta only matters once we're chasing a specific corpus-coverage target.

**Deferred:**

- 24-hour scheduled long-run (`-max_total_time=86400` on a weekly cron) + OSS-Fuzz-style corpus publishing. The 10-min nightly is the structural floor; longer runs amortize the libjpeg-turbo-progs install cost over more iterations and surface deeper-mutation crashes that the 10-min budget cannot reach.

**Acceptance:** see commands in [reference_commands.md](reference_commands.md) — each must run 10 min in CI without finding a divergence.

### Follow-up: arithmetic decoder mid-scan divergence — **CLOSED 2026-05-06 (NOT A DECODER BUG)**

`fuzz_decode_diff_c` surfaced a 146-byte arithmetic-coded grayscale fixture (272×16 SOF9, single component, ~30 bytes of entropy) where Rust pixels diverge from djpeg's `-pnm` output by max-diff=255 across rows 8-15. Originally hypothesized as an arithmetic decoder bug.

**Actual root cause:** **integer overflow in IDCT for malformed huge dequantized DC values, not a decoder bug.** Investigation on 2026-05-06:

1. Coefficient comparison: dumped all 68 blocks via Rust `read_coefficients` and via a C harness using `jpeg_read_coefficients`. **Zero mismatches** across all 68 blocks (verified after accounting for Rust's API zigzag-storage convention vs C's natural-order). The arithmetic decoder produces byte-identical coefficients to djpeg.
2. Pixel divergence reproduces with djpeg's *default* `-dct int` and `-dct fast`, but **not** with `-dct float` — float IDCT outputs 255 (saturation) where int/fast output 0 (overflow wrap). The fixture has DC values up to ±2710 (e.g., block 48: DC=2710). With quant table[0]=16, dequantized DC = ±43360 — far beyond the normal 8-bit JPEG coefficient range. The integer IDCTs (jidctint.c, jidctfst.c) carry this through their fixed-point arithmetic and overflow to a different sign than the mathematically-correct float version.
3. Our Rust integer IDCT also handles this overflow, but rounds toward saturation-at-255 instead of djpeg-int's wrap-to-0. Both behaviors are "out of spec"; the fixture is a fuzzer-generated stream of valid arithmetic codes with anomalously huge magnitude category bits.

**Resolution:** keep the `if probe.is_arithmetic() { return; }` skip in `fuzz_decode_diff_c.rs` since matching djpeg-int's overflow wrap would require introducing the same bug in our IDCT — a regression. The curated arithmetic conformance suites (`examples/corpus_test.rs`, `c_tjtrantest_full-arith-and-progressive-skip`) continue to gate byte-exact agreement against pinned well-formed references where overflow is not an issue.

### Follow-up: transform encoder small-image entropy divergence — **CLOSED 2026-05-06**

`fuzz_transform_diff_c` originally surfaced two 16×16 4:4:4 RGB fixtures (one Rot180-origin, one VFlip-origin). All three supported ops on both fixtures looked like a transform-writer bug class — Rust's outputs round-tripped through Rust's own decoder but jpegtran's djpeg rejected them with "premature end of data segment" / "extraneous bytes before marker 0xd9".

**Root cause:** *not* in the transform writer. The bug was in the read-side `BitReader`, which stalled on multi-`0xFF` runs (see the achromatic-output follow-up below). The transform path reads the source coefficients through the same `BitReader`, so the stall delivered corrupted (but Rust-self-consistent) coefficients to the rewriter, which then emitted self-consistent but jpegtran-divergent entropy. With the BitReader fix in commit 1e9c1bb in place, every op on every fixture is byte-exact identical to jpegtran's `-copy all` output:

| Source fixture | HFlip | VFlip | Rot180 |
|---|---|---|---|
| `crash-75b99921...` (Rot180 origin, 806B incl. op-byte) | 719 == 719 ✅ | 723 == 723 ✅ | 719 == 719 ✅ |
| `crash-de852cc2...` (VFlip origin, 778B incl. op-byte) | 722 == 722 ✅ | 721 == 721 ✅ | 717 == 717 ✅ |

**Verification:** `tests/transform_small_image_byte_exact.rs` pins both fixtures and asserts byte-exact equality across HFlip / VFlip / Rot180 vs jpegtran. The soft-skip in `fuzz_transform_diff_c.rs` (the `is_known_small` + Rust-self-decode escape hatch) is removed — any future divergence will now panic the differential as a real regression.

### Follow-up: baseline 16×16 RGB achromatic-output divergence — **CLOSED 2026-05-04**

`fuzz_decode_diff_c` (post-tolerance-bump in cba4674) surfaced a 682-byte baseline (SOF0) 16×16 RGB fixture (`crash-3c70bc73...`) where Rust decoded successfully (no warnings emitted) but produced achromatic output (R=G=B in every triplet) while djpeg produced colored output: max abs diff = 142 / mean ≈ 32.80.

**Root cause** (`src/decode/bitstream.rs::get_byte` + `fill_buffer_slow`): the entropy stream contained `FF 00 FF FF FF 00` and `FF FF FF FF` runs. libjpeg-turbo's `jpeg_fill_bit_buffer` (jdhuff.c:316–331) walks past *any* number of consecutive `0xFF` bytes before classifying the next byte as either a stuffed `0x00` (→ literal `0xFF` data) or a marker. Our `BitReader` only handled the `FF 00` form: on the first `FF FF` the fast path bailed to the "marker — push zero, don't advance" branch, which then looped forever pushing zero bits at the same byte position. Those phantom zero bits over-consumed Y plane bandwidth (Y[3] decoded with nz=60 instead of nz=9 for the diagnostic fixture), then starved Cb/Cr to immediate-EOB (`dc=0, AC=EOB` from the leading 4 zero bits) — producing fully achromatic output.

**Fix:** `get_byte` returns the sentinel `usize::MAX` on `FF FF` so the fast path bails to `fill_buffer_slow`, which now walks past the entire FF run before deciding stuffed-byte vs marker. Pre-fix: 256 of 256 pixels achromatic; post-fix: byte-exact agreement with djpeg on the pinned fixture.

**Verification:** `tests/bitstream_multi_ff_run.rs::multi_ff_run_decodes_chroma_byte_exact_vs_djpeg` pins the 682-byte fixture and asserts both (a) chroma is non-zero and (b) max diff = 0 vs djpeg.

### Follow-up: progressive small-entropy decoder pixel divergence — **CLOSED 2026-05-05**

`fuzz_decode_diff_c` (post-AC-bounds-soft-landing in commit ce14bbe) surfaced a 544-byte 16×16 SOF2 fixture with 10 progressive scans of which 8 carry only 1 byte of entropy each. djpeg accepts and decodes; Rust accepts and decodes, but the resulting pixels diverged: max abs diff = 61, mean ≈ 4.34, with 72 bytes of the 768-byte buffer differing by > 16. First 16 pixels byte-identical; divergence concentrated in the second MCU row.

**Root cause** (`src/decode/progressive.rs::decode_ac_refine`): when the inner zero-run loop exited via `k > Se`, the surrounding code's `if new_val != 0 && k <= se` guard *dropped* the new coefficient write entirely. libjpeg-turbo writes it at `*block + jpeg_natural_order[k]`, which the `[DCTSIZE2 + 16]` padding folds onto `coeff[63]`. The C reference therefore clobbers `coeff[63]` rather than skipping; we silently dropped state that subsequent refinement scans were supposed to refine, producing the observed pixel drift.

**Fix:** route `k > se` (within the soft-landing window k < 80) to `coeff[63]`, mirroring the AC initial soft-landing already in `decode_ac_first` and the libjpeg natural-order padding semantics.

**Verification:** `tests/progressive_ac_soft_landing.rs::ac_refine_soft_landing_matches_djpeg_byte_exact` pins the original 544-byte crash input and asserts byte-exact pixel agreement (max diff 0) vs djpeg.

## P2-8. Install-Staging, Symlink Chain, and CMake Config — **CLOSED**

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

## P2-9. v6b / v7 / v8 ABI Compatibility Matrix — **CLOSED**

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

## P2-10. Real Distro-Consumer Smoke Matrix — **CLOSED**

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

## P2-11. TJSAMP_411 / TJSAMP_441 / TJSAMP_410 / TJSAMP_24 Progressive Encode — **CLOSED**

**Status (2026-05-04): closed.** `cargo test --release --features full-c-parity --test c_tjcomptest` is **green for the full lossy + lossless matrix** including progressive + samp411/441/410/24 on the 227×149 testorig fixture. The source-level skip in `tests/c_tjcomptest.rs:717-739` is gone, the new C-tool-free guard `tests/regression_progressive_4pixel_chroma.rs` exercises all four 4-pixel factors, and the `continue-on-error: true` flag for `c_tjcomptest_full` in `.github/workflows/full-c-parity.yml` is removed.

**Root cause (2026-05-04):** `src/encode/pipeline_impl/progressive_entropy.rs::progressive_fdct_chroma_block` (and the matching branches in `pipeline_impl/arithmetic.rs`) clamped the chroma sampling factors with:

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

**Out of scope (separate gap, since closed):** the same `max_h <= 2 && max_v <= 2` gate in `transform_jpeg_with_options::progressive_safe` previously rejected 4-pixel factors from the **transform / jpegtran** progressive writer. Closed 2026-05-07 under [P3-4](phase3.md#p3-4-4-pixel-chroma-progressive-transform-writer-gate--closed-2026-05-07) — gate widened to `max_{h,v} ∈ {1,2,4}` (the eight standard TJSAMP factors), regression pinned in `tests/regression_progressive_4pixel_chroma_transform.rs`.

---

## Phase 2 Suggested Order (Historical)

1. ~~**P2-11** — Close the TJSAMP_411/441/410/24 progressive-encode byte-parity gap.~~ **CLOSED 2026-05-04** — root cause was a chroma-sampling-factor clamp in `progressive_fdct_chroma_block` (now in `src/encode/pipeline_impl/progressive_entropy.rs`); source-level test skip deleted, regression test in `tests/regression_progressive_4pixel_chroma.rs`.
2. ~~**P2-1** (`c_tjcomptest_full` portion) — Remove `continue-on-error` flag for the encode parity test.~~ **CLOSED 2026-05-04** — flag removed once P2-11 fix landed.
3. ~~**P2-9** — Decide and document the `JPEG_LIB_VERSION` policy.~~ **CLOSED 2026-05-04** — `docs/ABI_COMPATIBILITY.md` documents the v8-only policy with v6b-SONAME risk explicitly called out; `build.rs` emits a loud `cargo:warning` on the default-risk pairing.
4. ~~**P2-2** — Implement `format_message` printf expansion.~~ **CLOSED 2026-05-04** — `snprintf_jpeg` helper added in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`; `tests/format_message.rs` exercises every specifier `jerror.h` uses against `libc::snprintf` as the reference oracle.
5. ~~**P2-1** (remaining `c_tjtrantest_full` portion) — Investigate and fix or formally document the grayscale-Huffman transform divergence; remove the last `continue-on-error` flag.~~ **CLOSED 2026-05-04** — local run on aarch64 reports 11190/0 tested/failed; flag removed for both x86_64 and aarch64 jobs.
6. ~~**P2-4** — Generated C-side ABI cross-check.~~ **CLOSED 2026-05-04** — `tests/abi_offsets.rs` compiles a tiny C harness against the submodule's `jpeglib.h` and asserts every const-asserted field matches `offset_of!`. Coverage scoped to `jpeg_decompress_struct` (27 fields).
7. ~~**P2-3** — Per-platform offset assertions + CI matrix.~~ **CLOSED 2026-05-10** — release-blocking ABI validation is covered by the P3-1-expanded `abi_offsets` runtime gate (six public structs, per-field offsets + `sizeof`) in the `capi-abi-checks` CI matrix; Windows compile-time constants and 32-bit ABI targets are future hardening triggers, not OPEN/PARTIAL last-mile work.
8. ~~**P2-5** — Symbol-inventory diff against upstream.~~ **CLOSED 2026-05-04** — `tests/symbol_inventory.rs` parses upstream headers (66 jpeg + 79 tj symbols), asserts each is exported by our cdylib, allowlists 19 deferred legacy entries with rationale.
9. ~~**P2-8** — SONAME / pkg-config / install layout.~~ **CLOSED 2026-05-04** — `scripts/install_capi.sh` + `make install` stage cdylib + symlink chain + headers + `.pc` + `JPEGConfig.cmake`; `tests/install_layout.rs` asserts the layout end-to-end.
10. ~~**P2-7** — Differential fuzzing against C.~~ **CLOSED 2026-05-04** — three libfuzzer targets land. 24-hour scheduled long-run + OSS-Fuzz-style corpus publishing deferred as a future scaling step.
11. ~~**P2-10** — libvips / FFmpeg / SDL_image / GD consumer harnesses.~~ **CLOSED 2026-05-04** — all four landed. The libvips first-run also surfaced and fixed the `jpeg_c_*_param_*` symbol-surface gap via `mozjpeg_compat.rs`.
12. ~~**P2-6** — Publish to crates.io.~~ **CLOSED 2026-05-06** — both root crate (`libjpeg-turbo-rs` 0.6.1) and CAPI crate (`libjpeg-turbo-rs-capi` 0.1.0) are live.
