# Changelog

All notable changes to the `libjpeg-turbo-rs` workspace are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/). Entries before v0.6.0 predate
this file — see the [GitHub releases](https://github.com/developer0hye/libjpeg-turbo-rs/releases)
and `git log` between tags.

## [Unreleased]

### Changed

- `wasm32` with `+simd128` but **without** the Cargo `simd` feature now uses
  the scalar path, honoring the feature exactly as aarch64/x86_64 always did
  (P4-135 criterion 5, #474). Previously the hand-written SIMD kernels were
  selected on the target feature alone. The published wasm wrapper crate
  enables `simd` and is unaffected; a consumer depending on the core crate
  with `default-features = false` who wants the hand-written kernels must
  enable the `simd` feature (the scalar path still autovectorizes under
  `+simd128`).
- **Breaking (Rust API):** `TjHandle::new()` now initialises `TJPARAM_QUALITY`
  to `-1` and `TJPARAM_SUBSAMP` to `TJSAMP_UNKNOWN` (unset), exactly as
  upstream TurboJPEG does, and every lossy compress path — native and C ABI —
  refuses with upstream's "must be specified" error until the caller supplies
  them (P4-155, #539). Code that relied on the old silent 75 / 4:2:0 defaults
  must set both explicitly; a lossless compress consults neither. `tj3Get` on
  a fresh handle reports the unset values.

### Added
- A pinned Ubuntu 24.04/OpenCV 4.6 replacement harness that proves OpenCV's
  JPEG compression/decompression symbols bind to the Rust `libjpeg.so.8` and
  runs system/Rust bidirectional cross-decodes.

### Fixed
- Nine SIMD wrapper functions whose safety contracts were comment-only now
  enforce them in code (P4-135, #474): `avx2_idct_islow`, `sse2_idct_islow`
  and the AVX2 FDCT+quantize composite check the CPU feature themselves and
  fall back to the scalar reference instead of assuming "verified at
  dispatch time"; the four `wasm_*_to_ycbcr_row` encode wrappers and
  `neon_fancy_h2v2_row` validate slice lengths (`checked_mul`) instead of
  documenting "Caller guarantees…"; and `wasm_fancy_upsample_h2v1` asserts
  its bounds explicitly. A new unguarded parity test asserts that whichever
  arm the executing host takes equals scalar, and the CPUID-masked no-AVX2
  CI leg now runs the lib suite (`--lib`) so the fallback arms execute on a
  CPU that genuinely lacks AVX2, not only the SIMD arms on one that has it.
- Baseline `wasm32` builds (no `+simd128`) no longer emit SIMD128
  instructions (P4-135 criterion 5 / P4-143, #474). Previously the wasm
  backend compiled unconditionally and 13 pipeline call sites dispatched
  to it regardless of the target feature, so a baseline module
  carried 889 SIMD instructions and was rejected at validation by engines
  without WebAssembly SIMD ("SIMD support is not enabled") — the scalar
  fallback documented in the wasm crate's README never ran. Now the arch
  backend modules are gated on the Cargo `simd` feature — wasm32
  additionally on the `simd128` target feature — every call site states the
  same predicate, and a new CI leg builds both wasm targets without the
  repo's forced `+simd128` (denying warnings), covering the configuration
  downstream consumers actually get. Consumers building baseline `wasm32`
  from crates.io are affected; in-repo builds always forced `+simd128` and
  are not.
- The classic decode sequence (`jpeg_read_header` → `jpeg_start_decompress` →
  `jpeg_read_scanlines`) now honors `cinfo->mem->max_memory_to_use` exactly as
  upstream does (P4-14, #467): the budget applies to multi-scan streams
  (progressive or non-interleaved sequential) and buffered-image mode, weighs
  only the whole-image coefficient-array bytes, and refuses at
  `jpeg_start_decompress` with `JERR_NO_BACKING_STORE` (51). Baseline
  single-scan decodes outside buffered-image mode remain unbounded, matching
  stock, and budgets above the coefficient bytes accept even where a
  whole-pipeline estimate would not (`djpeg -maxmemory` parity,
  cross-validated against stock libjpeg-turbo).
- `jpeg_has_multiple_scans()` now reports upstream's definition —
  `(comps_in_scan < num_components) || progressive_mode` — so non-interleaved
  *sequential* streams (the `cjpeg -scans` shape) return TRUE; it previously
  mirrored `progressive_mode` alone.
- `jpeg_mem_dest`'s allocation-failure paths (initial buffer and mid-encode
  growth) are now provably reachable and raise upstream's
  `JERR_OUT_OF_MEMORY` case 10 (P4-120, #467), exercised by a test-only
  fault-injection hook in the shim's allocation funnel.
- The P4-13 progressive-suspension C oracle now discovers `cjpeg`/`djpeg` on
  `PATH`, fails closed on tool/compile errors in CI, and runs as an explicit
  provisioned Linux CI gate instead of reporting a soft-skipped pass.
- Classic C-ABI documentation now distinguishes selected OpenCV/stock-tool
  evidence from general drop-in readiness and tracks the newly audited
  ABI/ownership/state/error/test-integrity gaps through P4-116.
- Classic `jpeg_*` scanline compression now honors the public
  `restart_interval` and `restart_in_rows` fields in baseline, optimized,
  progressive, arithmetic, and lossless modes. The OpenCV replacement
  harness structurally checks DRI/RST markers and produces a byte-identical
  progressive JPEG to Ubuntu's libjpeg-turbo baseline.
- Baseline classic scanline compression also honors `smoothing_factor`
  without implicitly enabling optimized Huffman coding, matching
  `cjpeg -smooth`.
- `Decoder` grayscale output requests now match libjpeg-turbo for JCS_RGB
  input and for legal streams whose component 0 requires upsampling, rather
  than treating red as luma or rejecting the conversion.

## [0.8.0] - 2026-07-28

### Added
- `decompress_from_reader_incremental`: bounded-input-memory decode
  from any `Read` source — interleaved baseline JPEGs decode from a
  sliding window (measured peak input storage: 195,985 bytes of
  allocation capacity — window + header + 64 KiB staging buffer — on a
  1.25 MB 1080p stream, independent of compressed size). Progressive,
  non-interleaved and single-component/grayscale, arithmetic, lossless
  and 12/16-bit streams fall back to buffering. The companion
  `decompress_from_reader_incremental_instrumented` also returns the
  measured peak. The C-ABI suspension core and the new reader now
  share one marker-boundary scanner (`decode::boundary`) (#357).
- One-call header probe `probe(&[u8]) -> Result<JpegInfo>` — dimensions,
  coding mode, subsampling, colorspace, metadata presence, EXIF
  orientation without decoding pixels (#386).
- Chainable decoder configuration: 21 `with_*` counterparts of the
  `Decoder` mutators — the 20 `set_*` methods plus `save_markers`
  (#386).
- `Image::as_bytes()` / `Image::into_vec()`; `Image` and the new
  `JpegInfo` derive `Clone`/`PartialEq`/`Eq` (`SavedMarker` gains
  `PartialEq`/`Eq`); `FrameHeader::width()/height()/dimensions()`
  `usize` accessors (#386).
- Crate-doc "Canonical API map" naming the entry point per task; the
  legacy `compress_*` variants grouped as specialised entry points with
  a stated deprecation intent (next semver-major) (#386).
- EXIF orientation surface (#391): `Decoder::exif_orientation()` /
  `ImageInfo::exif_orientation()` header probes,
  `TransformOp::from_exif_orientation`, and pixel-domain
  `Image::apply_orientation[_value]`.
- `StreamingDecoder::skip_scanlines` now actually skips, C-clamp
  semantics, byte-identical to `djpeg -skip` (#383).
- User-facing `examples/` set with a curated index, plus an
  `image`-bridge example (#388); `cargo check --examples` in CI.
- docs.rs metadata + crate-level quickstart with asserting doctests
  (#387); user-first README with badges and measured zune/C comparisons
  (#385); `CONTRIBUTING.md`.
- `fuzz_decompress_precision` target covering the 12/16-bit and
  arbitrary-precision decoders (#382).
- Windows CI leg building every workspace test target + the two
  C-tool-free suites (#378); Miri CI job over the non-SIMD unit tests
  (#389 phase 1); `publish-check` CI job (#380).
- Regression test for #362 (duplicate report of #314, filed against
  v0.6.3 after the v0.7.0 fix shipped; PR #379 confirmed no further
  product change was needed).
- This changelog; MSRV CI job (rustc 1.87 for the root and capi crates);
  `cargo-deny` supply-chain gate (advisories/bans/licenses/sources) with
  `deny.toml` policy; Dependabot for cargo + GitHub Actions; a README
  for the capi crate (#390).

### Changed
- `Decoder` (and the wrappers embedding it) are now `Send`; the
  `set_marker_processor` / `set_resync_strategy` callbacks require
  `Send` (technically breaking for non-`Send` closures) (#384).
- Crate-level `#![deny(unsafe_op_in_unsafe_fn)]`; encoder NEON/SSE2/
  WASM-SIMD fast paths now honour the `simd` feature being disabled
  (#389 phase 1).
- Bridge crates are publishable (`version` + `path` deps); the `image`
  bridge builds the codec with default features, restoring runtime
  AVX2/NEON dispatch (#380, #381).
- `bench_zune_matrix` calibration stabilized (median-of-3 estimate,
  iteration floor, visible medians) (#376).
- Clippy CI widened from `--lib` to `--workspace` (zero allowances) plus
  `--workspace --all-targets` (three structural lints allowed in test
  code only, tracked as P4-70); the `image`-bridge crate declares
  `rust-version = "1.88"` (inherited from `image@0.25`) (#390).
- The `image` bridge now depends on `image` with
  `default-features = false` — it only uses the codec traits, and the
  defaults pulled the whole format-codec set (including
  `ravif`→`rav1e`, whose pinned `core2`/`paste` carry RUSTSEC
  advisories) into every consumer's build graph. Consumers who relied
  on the bridge transitively enabling image's default formats should
  enable those features on their own `image` dependency (#390).

### Security
- Bumped `crossbeam-epoch` 0.9.18 → 0.9.20 in the dev/bench dependency
  tree (RUSTSEC-2026-0204, invalid pointer dereference in a
  `fmt::Pointer` impl), caught by the new `cargo-deny` gate (#390).

### Fixed
- `set_output_format(Grayscale)` on a colour (YCbCr) JPEG errored while
  the equivalent `set_output_colorspace` route worked; it now decodes,
  byte-identical to `djpeg -grayscale` (#386). The WASM bindings'
  `decode_to(.., Grayscale)` inherits this: previously an error on
  colour sources, now a 1-byte-per-pixel luma image. The override's gray arm
  also panicked on legal streams whose component 0 is subsampled below
  max — now a clean `Unsupported` (full conversion tracked as P4-72) —
  and no longer doubles plane memory past a `set_max_memory` cap when
  no crop shift is needed.
- `examples/pillow_smoke/run.sh` ignored `CARGO_TARGET_DIR` and could
  validate a stale in-repo shim (#386; the wider capi-test variant is
  P4-73).
- Grayscale decode to `Argb`/`Abgr` wrote the alpha byte in the wrong
  slot (#369); 12-bit grayscale decode ignored the requested output
  format (#394); `decompress_to(.., Cmyk)` panicked on non-CMYK sources
  (P4-68).
- 12/16-bit lossless point-transform validation: crafted `Al >=
  precision` streams panicked in debug and mis-decoded in release; a
  corrupt DHT DC category > 16 overflowed a shift (#382).
- The `C Interop` CI job ran zero tests (substring filter) (#377); the
  MSVC UCRT `snprintf` link failure blocked `cargo test --workspace` on
  Windows (#378).

## [0.7.0] - 2026-07-26

### Added
- RGB-direct (`JCS_RGB`) encode modes incl. progressive + arithmetic
  combinations (#345, #346, #348).
- Cross-platform golden tests pinning encoder output byte-stability
  (#319, #337).

### Fixed
- 4:2:0 AVX2 encoder emitted non-dummy trailing-MCU column blocks for
  `width % 16 in 1..=8` (#314); aarch64 DCT parity (#342).
- CMYK encode dropped options and wrote a spurious JFIF marker (#313,
  #339); progressive+arithmetic option combinations (#322).

## [0.6.3] - 2026-07-25

Fuzz-driven robustness batch spanning two months of scheduled
fuzz-smoke findings.

### Added
- Real-world corpus decode/encode gates vs C libjpeg-turbo in CI
  (P4-31, #307).
- Arithmetic multi-scan support (SOF9, interleaved or not) (P4-24);
  streaming `jpeg_consume_input` suspension core (P4-13 partial).

### Fixed
- Multi-scan non-interleaved baseline divergence (P4-22);
  lenient-recovery parity with djpeg on invalid Huffman codes (P4-23);
  IDCT i16-overflow parity with C SIMD (P4-19).
- Scheduled fuzz findings: sparse-DQT-slot remap, category-16
  coefficient rejection, fractional-chroma-ratio reject, SOS
  component-id binding, lossless undifference wrap + `<< Al` scaling
  (P4-34..38); 12-bit sampling-layout overflow (P4-30); block smoothing
  read dummy padding blocks (P4-29); single-component h1v4 block order +
  progressive AC-refine placement (P4-27/28); transform perf + marker
  preservation (#308).

## [0.6.2] - 2026-05-24

C-ABI replacement-tier hardening: P4-2..P4-12 closed (default SONAME
flip to `libjpeg.so.8` with staged loader symlinks, panic guards on all
C entry points, pathological-lifecycle harness, x86_64 dispatch audit),
thread-affinity contract (P4-16) and legacy-TJ migration matrix (P4-18)
documented, cold-review gaps P4-13/14/17 filed, and the scheduled
fuzz-smoke pipeline's repro-artifact workflow with C tools pinned to
libjpeg-turbo 3.x.

## [0.6.1] - 2026-05-06

C-ABI hardening: classic `jpeglib.h` lifecycle/suspension harness
(P3-5), ABI offset cross-checks extended to all six structs (P3-1),
12-bit raw-data backend (P3-2), non-standard sampling + RGB565
merged-upsample fixtures (P3-6). WASM crate v0.2.0 split out.

## [0.6.0] - 2026-05-03

First release with the full T1 (Rust crate) + T2 (TurboJPEG 3 cdylib)
+ T3 (classic libjpeg v8 cdylib) replacement surface: byte-exact stock
djpeg/cjpeg/jpegtran parity, Pillow/libtiff integration, panic guards
on all C entry points.

[Unreleased]: https://github.com/developer0hye/libjpeg-turbo-rs/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/developer0hye/libjpeg-turbo-rs/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/developer0hye/libjpeg-turbo-rs/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/developer0hye/libjpeg-turbo-rs/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/developer0hye/libjpeg-turbo-rs/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/developer0hye/libjpeg-turbo-rs/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/developer0hye/libjpeg-turbo-rs/releases/tag/v0.6.0
