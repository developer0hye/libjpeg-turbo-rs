# Corpus Test Report

> **Historical snapshot (2026-04-07).** This report was generated from a one-time
> corpus run and is preserved for reference. The corpus artifacts (`tests/corpus/`,
> `tests/corpus_results.tsv`) are not committed to the repository. Current
> regression coverage is maintained by CI's `test-corpus` job, which regenerates
> the corpus from scratch each run. Do not treat the numbers below as current
> branch status — run `cargo run --release --example corpus_test` to get live
> results.

Generated: 2026-04-07
Corpus: `tests/corpus/` (2240 JPEG files)
Raw TSV: `tests/corpus_results.tsv`

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total files | 2240 |
| Total test rows | 20,160 (9 operations × 2240) |

### Per-Operation Results

| Operation | Pass | Fail | Crash | Skip | Pass Rate |
|-----------|------|------|-------|------|-----------|
| Decode | 2240 | 0 | 0 | 0 | **100%** |
| Encode | 1659 | 51 | 0 | 530 | 97% (of non-skipped) |
| Transform (7 ops × 2240) | 9992 | 4906 | 833 | 0 | 72% |

**Overall**: Decode is perfect. Encode has a known skip class and a small pixel-diff failure class. Transform has two distinct bug categories.

---

## Failure Analysis

### Category 1: Transform — Byte Mismatch (4906 failures, 727 unique files)

**Pattern**: All transform operations except `transform_flipv` and `transform_rotate180` fail with `byte mismatch: Rust N bytes vs C N bytes`. The Rust output and C jpegtran output differ at the bitstream level, even though both are valid JPEGs.

**Affected files**: 727 unique files spanning all corpus subdirectories:
- 698 generated files
- 18 fuzz_seeds
- 11 fixtures (odd-dimension 420 files: `cjpeg_31x33`, `cjpeg_33x31`, `cjpeg_241x319`, `cjpeg_320x241`, `cjpeg_321x243`, `cjpeg_479x641`, `cjpeg_641x479`, strip images)

**Subsampling breakdown of affected files**:
- 420: 294 files
- 422: 163 files
- 444: 138 files
- Gray: 132 files

**Encoding variant breakdown**: All variants affected (baseline 118, optimized 118, restart 118, progressive 226, arithmetic 278). Not limited to any encoding type.

**Root cause hypothesis**: The Rust lossless-transform implementation re-encodes Huffman tables differently from C jpegtran. Jpegtran uses optimized Huffman table generation during transform; Rust either copies the original tables verbatim or generates them with different statistics. The byte diff is non-zero but the JPEG is structurally valid — this is a **Huffman table encoding difference**, not a pixel-level error. The test compares raw bytes, but the harness does not decode both outputs and compare pixels, so the true semantic error (if any) is unknown.

**Specific transform ops failing more often**: `rotate90`, `rotate270`, `transpose`, `transverse` (727 files each) fail more than `rotate180`, `fliph` (675 each) and `flipv` (597). This pattern is consistent with dimension-swapping transforms producing different MCU-boundary padding behavior.

---

### Category 2: Transform — Crash: "invalid Huffman code" (812 crashes, 116 unique files)

**Pattern**: `Rust transform error: corrupt data: invalid Huffman code` — Rust's transform function reads back its own output for internal re-encoding and fails to decode the progressive Huffman stream.

**Affected files**: 116 unique files, all matching the pattern:
- `*_progressive.jpg` with **odd dimensions** (7×11, 33×17)
- All subsampling types (420, 422, 444) affected
- All quality levels affected

Example files:
- `odd_7x11_420_*_progressive.jpg` (all qualities)
- `odd_33x17_420_*_progressive.jpg` (all qualities)
- `odd_7x11_422_*_progressive.jpg`
- `odd_33x17_422_*_progressive.jpg`

**Root cause hypothesis**: The Rust progressive JPEG transform path has a bug when the image has non-MCU-aligned dimensions. The progressive Huffman decoder fails with "invalid Huffman code" when trying to decode DCT coefficients from the transformed progressive stream, suggesting the transform incorrectly handles the partial MCU blocks at right/bottom edges.

---

### Category 3: Transform — Crash: "AC coefficient index out of bounds" (14 crashes, 2 unique files)

**Pattern**: `Rust transform error: corrupt data: progressive AC coefficient index out of bounds`

**Affected files** (2 files, 7 transform ops each):
- `tests/corpus/generated/testorig_420_50_progressive.jpg`
- `tests/corpus/generated/testorig_422_50_progressive.jpg`

**Root cause hypothesis**: The testorig image (libjpeg-turbo's canonical test image) at quality=50 with progressive encoding triggers an AC coefficient bounds check failure. This is a specific input-dependent bug in the progressive transform decoder — the scan structure of this particular image causes an index to exceed the expected AC coefficient range (0–63).

---

### Category 4: Transform — Crash: Arithmetic Overflow (7 crashes, 1 file)

**Pattern**: `panicked: attempt to add with overflow`

**Affected file**: `tests/corpus/fixtures/photo_3840x2160_420_prog.jpg` — all 7 transform operations crash.

**Root cause hypothesis**: The 3840×2160 image is the largest in the corpus. An integer arithmetic overflow in the transform path (likely computing byte offsets, MCU counts, or buffer sizes) occurs when dimensions approach 4K. This is a **safety bug** — it crashes in debug mode and would silently produce wrong results in release mode with wrapping arithmetic.

---

### Category 5: Encode — Skip: Grayscale PPM Parse Failure (530 skips, 530 unique files)

**Pattern**: `C djpeg error: failed to parse PPM` — the encode harness decodes grayscale JPEGs with `djpeg -ppm` expecting P6 (RGB PPM), but djpeg outputs P5 (PGM grayscale). The PPM parser fails to read the P5 header.

**Affected files**: All 530 grayscale corpus files (identified by `gray` in filename or `gray_8x8.jpg` fixture).

**Root cause**: Harness bug in `examples/corpus_test.rs`. The `run_encode_test` function always calls `decode_with_c_djpeg(..., grayscale: false)`, which runs `djpeg -ppm`. For grayscale JPEGs this produces a PGM file that the PPM parser rejects. The decode test path correctly detects grayscale via `rust_img.pixel_format == PixelFormat::Grayscale`, but the encode path does not.

**Fix**: In `run_encode_test`, first probe the pixel format (e.g. call `decompress` to check `pixel_format`), then call `decode_with_c_djpeg` with `grayscale=true` for grayscale images.

---

### Category 6: Encode — Pixel Diff Failures (51 failures, 14 unique files)

**Pattern**: Rust encoder re-encodes reference pixels at quality=75 S420, but the decoded output differs from C cjpeg's decoded output by up to 39 pixel values.

**Affected files** (selected):
- `photo_1920x1080_422.jpg` / `photo_1920x1080_422_prog.jpg` — max_diff=39
- `photo_1920x1080_444.jpg` / `photo_1920x1080_444_prog.jpg` — max_diff=38
- `photo_1920x1080_420.jpg` / `photo_1920x1080_420_prog.jpg` — max_diff=18
- `cjpeg_7x8_odd_even_{420,422,444}.jpg` — max_diff=5–7
- `strip_1x100_*` (various subsampling/quality) — max_diff=1

**Root cause hypothesis**: The encode test decodes input JPEGs (which may be 422/444/progressive) to RGB pixels, then re-encodes at quality=75 S420. The pixel differences arise from:
1. **Lossy re-encoding of already-lossy data** — the input was compressed at a different quality, re-encoding introduces double-quantization artifacts that differ slightly between Rust and C.
2. **Subsampling conversion differences** — when the source is 444 or 422 and the encoder outputs 420, chroma downsampling may differ.
3. **Odd-dimension handling** — `7x8` and `1x100` strip images may have padding differences.

The max_diff=39 for 422/444 photos indicates the Rust encoder's color conversion or subsampling path deviates from C for non-420 sources.

---

## Root Cause Analysis

| Category | Type | Scope | FEATURE_PARITY.md item |
|----------|------|-------|------------------------|
| Transform byte mismatch | Huffman table generation difference | All transforms, all files | Lossless transform Huffman optimization |
| Transform crash: invalid Huffman (progressive+odd dims) | Bug in progressive transform | Progressive JPEGs with non-MCU-aligned dims | Progressive transform support |
| Transform crash: AC OOB | Decoder bug on specific scan structure | 2 specific files | Progressive scan decoding |
| Transform crash: overflow | Integer overflow on large image | 4K image | Transform buffer sizing |
| Encode skip: grayscale PPM | Harness bug | Grayscale encode tests (530 files) | N/A — harness fix |
| Encode pixel diff | Encoder output deviation | Large photo + odd-dim files | Encode color accuracy |

---

## Prioritized Fix List

### Priority 1: Harness Bug — Grayscale Encode Skip (effort: easy)
**Impact**: 530 skipped encode tests become testable  
**Fix**: In `examples/corpus_test.rs::run_encode_test`, detect grayscale by checking `decompress` output pixel_format before calling `decode_with_c_djpeg`, pass `grayscale=true` for gray images and use S444/gray subsampling in the Rust encoder.  
**File**: `/Users/yhkwon/Documents/Projects/libjpeg-turbo-rs/examples/corpus_test.rs:386–456`

### Priority 2: Transform Crash — Integer Overflow on 4K Image (effort: easy)
**Impact**: 7 crashes eliminated; safety fix (would silently overflow in release)  
**Fix**: Find and fix the usize/u32 arithmetic overflow in the transform path. Profile the 3840×2160 image to locate the panic site. Use `checked_add` or widen types.  
**FEATURE_PARITY**: Transform robustness

### Priority 3: Transform Crash — Progressive + Odd Dimensions (effort: medium)
**Impact**: 812 crashes → 0; fixes 116 unique files  
**Root cause**: Progressive JPEG transform incorrectly handles partial MCU blocks at image boundaries for non-MCU-aligned dimensions  
**Fix**: Audit the progressive Huffman read/write path in the transform implementation for boundary conditions when `width % 8 != 0` or `height % 8 != 0`.

### Priority 4: Transform Crash — AC Coefficient OOB (effort: medium)
**Impact**: 14 crashes → 0; fixes 2 specific files  
**Fix**: Audit the AC coefficient index bounds check in the progressive scan decoder used by the transform path. The `testorig` image has a specific scan structure that triggers index > 63.

### Priority 5: Transform Byte Mismatch — Huffman Table Differences (effort: hard)
**Impact**: 4906 failures → 0; the largest failure category  
**Root cause**: Rust lossless transform does not optimize Huffman tables the way jpegtran does. Jpegtran by default performs Huffman optimization (`-optimize`) during transform.  
**Fix**: Either (a) add Huffman optimization to the Rust transform path to match jpegtran's default behavior, or (b) change the test to decode-and-compare pixels instead of comparing raw bytes. Option (b) is correct for a semantics test; option (a) matches the byte-identical target.  
**FEATURE_PARITY**: Lossless transform with Huffman optimization

### Priority 6: Encode Pixel Diff on Large Photos (effort: medium)
**Impact**: 51 failures → 0  
**Root cause**: Color conversion / chroma downsampling differences when re-encoding 422/444 source as 420  
**Fix**: Investigate the specific pixel that first differs in `photo_1920x1080_422.jpg` → identify whether it's a chroma downsampling, DCT, or quantization difference vs C.

---

## Known Limitations

### C Tools Availability
All C tools (`djpeg`, `cjpeg`, `jpegtran`) were found at `/opt/homebrew/bin/`. No tests were skipped due to missing tools.

### Corpus Coverage Gaps
- No 12-bit precision JPEG files in corpus (libjpeg-turbo supports 12-bit)
- No CMYK/YCCK files in corpus
- No EXIF/ICC profile-bearing files in generated set
- No arithmetic-coded files with odd dimensions (arithmetic+progressive is in the matrix but not tested with odd dims in the fixture set)
- Fuzz seeds are limited to 8×8 gradient/red images at 3 quality levels

### Harness Limitations
- Encode test always uses quality=75 S420 regardless of source encoding — this is by design but means encode quality coverage is narrow
- Transform test compares raw bytes, not decoded pixels; byte differences may not be semantic errors if both outputs decode identically
- No multi-thread stress testing (each file is processed serially)

### Skipped Tests
- 530 encode tests skipped due to grayscale PPM parse bug in harness (all grayscale files)
