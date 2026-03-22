# Test Parity: libjpeg-turbo-rs vs libjpeg-turbo (C)

> Track testing methodology parity. `[x]` = implemented, `[ ]` = missing.
> Source of truth: C libjpeg-turbo's `CMakeLists.txt`, `tjunittest.c`, `tjbench.c`, test scripts in `test/`.

---

## 1. Test Programs / Executables

### tjunittest (TurboJPEG Unit Tests)
- [x] Encode/decode roundtrip (compTest/decompTest) — `tjunittest_compat.rs`
- [x] Multiple pixel formats (RGB, BGR, RGBX, BGRX, XBGR, XRGB, RGBA, BGRA, ABGR, ARGB) — `tjunittest_compat.rs`, `pixel_formats.rs`
- [x] Grayscale encode/decode — `tjunittest_compat.rs`, `grayscale_encode.rs`
- [ ] CMYK pixel format in tjunittest matrix — partial (only in `cmyk_encode.rs`, not in tjunittest matrix)
- [x] All subsampling modes (444, 422, 420, Gray, 440, 411, 441) — `tjunittest_compat.rs`
- [x] 8-bit precision — `tjunittest_compat.rs`
- [x] 12-bit precision — `tjunittest_compat.rs`, `precision.rs`
- [x] 16-bit lossless precision — `tjunittest_compat.rs`, `precision.rs`
- [ ] 2-7, 9-11, 13-15 bit lossless precision (per-precision tests) — only 8/12/16 tested
- [x] YUV encode/decode pipeline — `tjunittest_yuv.rs`, `yuv_api.rs`
- [ ] YUV no-padding mode (`-yuv-nopad`) — not tested
- [x] Lossless JPEG (PSV 1-7, PT variations) — `tjunittest_compat.rs`, `lossless_encode.rs`
- [x] Buffer allocation / overflow tests — `bufsize.rs`, `bufsize_extras.rs`
- [x] BMP I/O tests — `image_io.rs`
- [x] Pixel value verification (checkBuf equivalent) — `tjunittest_compat.rs` verify_roundtrip()
- [x] YUV plane validation (checkBufYUV equivalent) — `tjunittest_yuv.rs`
- [ ] Automatic allocation mode (`-alloc` flag) — not tested (Rust uses Vec, always auto)
- [x] Synthetic test pattern generation — `tjunittest_compat.rs` generate_test_pattern()

### tjbench (Benchmark & Tile Tests)
- [ ] Tiled compression/decompression (8x8, 16x16, 32x32, 64x64, 128x128) — not implemented
- [x] Scaled decompression (1/2, 1/4, 1/8) — `scale_decode.rs`
- [ ] Extended scaling factors (2/1, 15/8, 7/4, 13/8, 3/2, 11/8, 5/4, 9/8, 7/8, 3/4, 5/8, 3/8) — only 1/2, 1/4, 1/8
- [x] Partial decompression (cropping) — `crop_skip.rs`
- [x] Transform + scale combinations — `tjunittest_transform.rs`
- [ ] Merged YUV decompression (420m, 422m) — not tested
- [x] Performance benchmarking (Mpixels/sec) — `benches/decode.rs` (criterion)
- [ ] Frame rate / throughput measurement in tests — benchmark only, not assertion-based

### CLI Tool Equivalents
- [x] cjpeg equivalent (compress API) — `Encoder`, `compress()`
- [x] djpeg equivalent (decompress API) — `decompress()`, `Decoder`
- [x] jpegtran equivalent (transform API) — `transform_jpeg()`, `transform_jpeg_with_options()`
- [ ] rdjpgcom equivalent (read JPEG comments) — `Image.comment` exists, no CLI
- [ ] wrjpgcom equivalent (write JPEG comments) — `Encoder.comment()` exists, no CLI

---

## 2. Validation Methodologies

### Exact Binary Validation
- [x] Byte-for-byte file comparison (`cmp` equivalent) — `bitstream_stability.rs` (deterministic encoding verification)
- [x] MD5 hash validation of compressed output — `bitstream_stability.rs`, `bitstream_regression.rs` (hash-based determinism + regression)
- [x] Expected MD5 hashes per test configuration — `reference_hashes.json` (11 configurations stored)

### Pixel Value Validation
- [x] checkBuf equivalent (pixel-level roundtrip verification) — `verify_roundtrip()` in tjunittest_compat.rs
- [x] Tolerance: 0 for lossless — `tjunittest_compat.rs` lossless tests
- [x] Tolerance: ±2 for lossy — `tjunittest_compat.rs`, `cross_validation.rs`
- [ ] Tolerance: ±1 for 8-bit lossy (C uses 1, we use 2) — we're more lenient
- [x] checkBufYUV equivalent (YUV plane verification) — `tjunittest_yuv.rs`

### Reference Image Comparison
- [x] Decode C-encoded reference JPEGs — `reference_image_compat.rs`, `cross_encoder_compat.rs`
- [x] testorig.jpg decode validation — `cross_encoder_compat.rs`
- [x] testimgari.jpg (arithmetic) validation — `cross_encoder_compat.rs`
- [x] testimgint.jpg (interleaved) validation — `cross_encoder_compat.rs`
- [x] testorig12.jpg (12-bit) validation — `cross_encoder_compat.rs`
- [ ] ImageMagick reference crop comparison (croptest) — not implemented
- [x] djpeg reference output comparison — `cross_validation.rs`

### Dimension/Metadata Validation
- [x] Width/height verification after decompress — throughout all tests
- [x] Color space verification — `conformance.rs`
- [x] Buffer size checks — `bufsize.rs`, `bufsize_extras.rs`
- [x] Integer overflow protection — `malformed_jpeg.rs`

---

## 3. Test Script Equivalents

### tjcomptest.in (Compression Validation ~6900 lossy + ~3360 lossless combos)
- [x] All 6 subsampling modes (444, 422, 440, 420, 411, 441) — `tjunittest_compat.rs`
- [x] Quality levels (default, 1, 100) — `tjunittest_compat.rs`, `encode_boundaries.rs`
- [x] Restart intervals (`-r 1`, `-r 1b`) — `restart_encode.rs`, `tjunittest_compat.rs`
- [x] ICC profile with restart (`-r 1 -icc test3.icc`) — `metadata_write.rs`
- [x] Arithmetic encoding (`-a`) — `tjunittest_compat.rs`, `arithmetic.rs`
- [x] Progressive encoding (`-p`) — `tjunittest_compat.rs`, `progressive_enc.rs`
- [x] Progressive + Arithmetic — `tjunittest_compat.rs`
- [x] Optimized Huffman (`-o`) — `tjunittest_compat.rs`, `huff_opt.rs`
- [x] Lossless PSV 1-7 × PT 0-14 (pt < precision) — `tjunittest_compat.rs`, `lossless_encode.rs`
- [x] 8-bit and 12-bit lossy precision — `tjunittest_compat.rs`, `precision.rs`
- [x] 2-16 bit lossless precision (per-bit) — partial (8/12/16 only)
- [ ] Grayscale-from-RGB encode (`-g` flag) in full matrix — tested individually, not in matrix
- [ ] RGB-direct encode (`-rg` flag, no YCbCr conversion) in full matrix — not in matrix
- [ ] `-baseline` flag forced with quality=1 — `force_baseline` exists but not in matrix
- [ ] `-r 1b` byte-unit restart interval — not distinguished from MCU-row restart
- [ ] DCT method variation (`-dc fa`, `-dc f`) in compress matrix — tested individually
- [ ] Full cross-product: precision × restart × ari × dct × opt × prog × quality × subsamp — we test subsets, not full cross
- [ ] MD5 comparison between our encoder and C cjpeg — not implemented
- [ ] Binary `cmp` between our encoder and C cjpeg — not implemented
- [x] Grayscale input image encode — `tjunittest_compat.rs`, `grayscale_encode.rs`

### tjdecomptest.in (Decompression Validation ~2000-3000 combos)
- [x] 7 subsampling modes (444, 422, 440, 420, 411, 441, gray) — `conformance.rs`
- [x] 4:1:0 (410) subsampling decode — `subsamp_410.rs`
- [x] 5 crop regions (14x14+23+23, 21x21+4+4, 18x18+13+13, 21x21+0+0, 24x26+20+18) — partial (`crop_skip.rs`)
- [ ] Crop × subsampling × scale full cross-product — only individual tests
- [ ] 15 scaling factors (16/8 thru 1/8) — only 1/2, 1/4, 1/8 tested
- [x] No-smooth upsampling (`-nos`) — `decode_toggles.rs`
- [ ] No-smooth × crop × scale cross-product — not tested
- [x] DCT methods (fast, accurate) — `decode_toggles.rs`
- [ ] Fast DCT limited to specific scale+subsamp combos (as in C) — not validated
- [x] Grayscale output from color JPEG (`-g`) — `decode_toggles.rs`
- [ ] Grayscale output in full matrix (only when nosmooth="") — not in matrix
- [x] ICC profile extraction — `metadata_write.rs`
- [ ] ICC profile extraction MD5 comparison against C djpeg — not done
- [ ] 2-16 bit lossless decompression per-precision — only 8/12/16
- [ ] MD5/binary comparison between our decoder and C djpeg — not implemented
- [x] PPM/PGM output format — `cross_encoder_compat.rs`
- [x] RGB output from grayscale JPEG — `decode_toggles.rs`

### tjtrantest.in (Transform Validation)
- [x] All 8 transform types — `tjunittest_transform.rs`, `transform.rs`
- [x] 7 subsampling modes + grayscale — `tjunittest_transform.rs`
- [x] Copy mode: `-c a` (all markers) — `copy_mode.rs` (`MarkerCopyMode::All`)
- [x] Copy mode: `-c n` (no markers) — `copy_mode.rs` (`MarkerCopyMode::None`)
- [x] Copy mode: `-c i` (ICC only) — `copy_mode.rs` (`MarkerCopyMode::IccOnly`)
- [x] 6 crop regions in transform — `tjunittest_transform.rs`, `transform_options.rs`
- [x] Grayscale conversion during transform — `transform_options.rs`
- [x] Progressive output from transform — `tjunittest_transform.rs`
- [x] Arithmetic output from transform — `tjunittest_transform.rs`
- [x] Optimized Huffman output — `transform_options.rs`
- [ ] Restart intervals in transform output — not tested in transform context
- [x] Trim flag — `transform_options.rs`
- [ ] 8-bit and 12-bit precision transforms — only 8-bit tested
- [ ] MD5/binary comparison against C jpegtran — not implemented
- [x] Progressive output — `tjunittest_transform.rs`
- [x] Arithmetic output — `tjunittest_transform.rs`
- [x] Restart intervals in transform — not directly, but restart encode/decode tested
- [x] Trim markers — `transform_options.rs`
- [ ] MD5 comparison between our transform and C jpegtran — not implemented

### croptest.in (Crop Region Validation)
- [x] Basic crop operations — `crop_skip.rs`, `transform_options.rs`
- [ ] Exhaustive crop window iteration (Y 0-16, H 1-16) — not implemented
- [ ] ImageMagick reference comparison — not available
- [ ] Progressive + crop — partial
- [ ] Smooth + crop — not tested

---

## 4. Test Parameter Coverage

### Precision Levels
- [x] 8-bit (primary) — extensive coverage
- [x] 12-bit — `precision.rs`, `tjunittest_compat.rs`
- [x] 16-bit lossless — `precision.rs`, `tjunittest_compat.rs`
- [ ] 2-7, 9-11, 13-15 bit (per-precision lossless) — C tests each individually

### Subsampling Configurations
- [x] 4:4:4
- [x] 4:2:2
- [x] 4:2:0
- [x] 4:4:0
- [x] 4:1:1
- [x] 4:4:1
- [x] 4:1:0 (decompression only, rare) — `subsamp_410.rs`
- [x] Grayscale

### DCT Methods
- [x] islow (integer accurate) — default, tested everywhere
- [x] ifast (integer fast) — `dct_method.rs`, `decode_toggles.rs`
- [x] float — `dct_method.rs`
- [ ] DCT method variation in full test matrix — only tested individually, not cross-product

### Entropy Coding
- [x] Huffman baseline — extensive
- [x] Huffman optimized — `huff_opt.rs`, `tjunittest_compat.rs`
- [x] Arithmetic sequential — `arithmetic.rs`, `tjunittest_compat.rs`
- [x] Progressive Huffman — `progressive_enc.rs`, `tjunittest_compat.rs`
- [x] Progressive + Arithmetic — `tjunittest_compat.rs`, `sof10_encode.rs`

### Color Formats
- [x] RGB, BGR, RGBA, BGRA — extensive
- [x] RGBX, BGRX, XRGB, XBGR, ARGB, ABGR — `pixel_formats.rs`
- [x] Grayscale — extensive
- [x] CMYK — `cmyk_encode.rs`
- [ ] RGB565 encode (decode only) — decode tested in `pixel_formats.rs`
- [x] Dithered/undithered RGB565 — `rgb565_dither.rs`

### Scaling Factors (Decompression)
- [x] 1/1 (no scale) — default
- [x] 1/2 — `scale_decode.rs`
- [x] 1/4 — `scale_decode.rs`
- [x] 1/8 — `scale_decode.rs`
- [ ] 2/1 (upscale) — not supported
- [ ] 15/8, 7/4, 13/8, 3/2, 11/8, 5/4, 9/8 (upscale fractions) — not supported
- [ ] 7/8, 3/4, 5/8, 3/8 (intermediate downscale) — not supported

### Transform Operations
- [x] None, HFlip, VFlip, Transpose, Transverse, Rot90, Rot180, Rot270 — all 8
- [x] Perfect flag — `transform_options.rs`
- [x] Trim flag — `transform_options.rs`
- [x] Crop flag — `transform_options.rs`
- [x] Grayscale flag — `transform_options.rs`
- [x] No-output flag — `transform_options.rs`
- [x] Progressive output — `transform_options.rs`
- [x] Arithmetic output — `transform_options.rs`
- [x] Optimize output — `transform_options.rs`
- [x] Copy-none flag — `transform_options.rs`
- [x] Custom coefficient filter — `coeff_filter.rs`

---

## 5. Edge Case & Robustness Testing

### Malformed Input (C: fuzzing, our: malformed_jpeg.rs)
- [x] Empty input — `malformed_jpeg.rs`
- [x] Missing SOI — `malformed_jpeg.rs`
- [x] Truncated data (10%, 50%, 90%) — `malformed_jpeg.rs`
- [x] Invalid marker sequences — `malformed_jpeg.rs`
- [x] SOF width=0, height=0 — `malformed_jpeg.rs`
- [x] Invalid component count — `malformed_jpeg.rs`
- [x] Invalid sampling factors — `malformed_jpeg.rs`
- [x] SOS component count = 0 — `malformed_jpeg.rs`
- [x] Oversized marker lengths — `malformed_jpeg.rs`
- [x] All-zero entropy data — `malformed_jpeg.rs`
- [x] All-0xFF entropy data — `malformed_jpeg.rs`

### Extreme Dimensions (C: BMPSizeTest, our: extreme_dimensions.rs)
- [x] 1×1 pixel — `extreme_dimensions.rs`
- [x] Non-MCU-aligned dimensions — `extreme_dimensions.rs`
- [x] Prime dimensions — `extreme_dimensions.rs`
- [x] Extreme aspect ratios — `extreme_dimensions.rs`
- [ ] 65535×65535 (max JPEG spec) — not tested (too large for test)
- [x] BMP dimension calculations — `image_io.rs`

### Memory Safety
- [x] Integer overflow in dimensions — `malformed_jpeg.rs`
- [x] Buffer size boundary conditions — `edge_case_inputs.rs`
- [x] max_pixels enforcement — `memory_limits.rs`
- [x] max_memory enforcement — `memory_limits.rs`
- [x] scan_limit enforcement — `memory_limits.rs`
- [x] stop_on_warning enforcement — `memory_limits.rs`

### Concurrency
- [x] Multi-threaded decode — `concurrency.rs`
- [x] Multi-threaded encode — `concurrency.rs`
- [x] Mixed encode/decode — `concurrency.rs`
- [x] SIMD thread safety — `concurrency.rs`
- [x] Send trait verification — `concurrency.rs`
- [ ] Stress test under memory pressure — not tested

### Fuzzing
- [x] Fuzz decompress — `fuzz/fuzz_targets/fuzz_decompress.rs`
- [x] Fuzz decompress lenient — `fuzz/fuzz_targets/fuzz_decompress_lenient.rs`
- [x] Fuzz roundtrip — `fuzz/fuzz_targets/fuzz_roundtrip.rs`
- [x] Fuzz coefficient reader — `fuzz/fuzz_targets/fuzz_read_coefficients.rs`
- [x] Fuzz transform — `fuzz/fuzz_targets/fuzz_transform.rs`
- [x] Fuzz progressive decoder — `fuzz/fuzz_targets/fuzz_progressive_decoder.rs`
- [x] Seed corpus generation — `tests/generate_fuzz_seeds.rs`
- [ ] Continuous fuzzing (OSS-Fuzz integration) — not set up

---

## 6. Metadata & Markers

- [x] JFIF APP0 read/write — `marker_system.rs`, `cross_encoder_compat.rs`
- [x] ICC profile read/write/multi-chunk — `metadata_write.rs`, `icc_exif_edge_cases.rs`
- [x] EXIF APP1 read/write — `metadata_write.rs`, `icc_exif_edge_cases.rs`
- [x] COM marker read/write — `marker_system.rs`, `icc_exif_edge_cases.rs`
- [x] Adobe APP14 (CMYK) — `cmyk_encode.rs`
- [x] Marker preservation through transform — `marker_preservation.rs`
- [x] Custom marker processor callback — `niche_options.rs`
- [x] Configurable marker saving levels — `marker_preservation.rs`
- [x] JFIF thumbnail extraction — `bufsize_extras.rs`
- [ ] JFIF thumbnail embedding — not implemented
- [x] DRI restart markers — `restart_encode.rs`

---

## 7. Progressive JPEG

- [x] Standard progressive encode/decode — `progressive_enc.rs`, `progressive_output.rs`
- [x] Custom scan scripts — `custom_scan.rs`
- [x] ProgressiveDecoder scan-by-scan output — `progressive_output.rs`
- [x] Incomplete progressive (early output) — `progressive_scan_edge_cases.rs`
- [x] Progressive + all subsampling modes — `progressive_scan_edge_cases.rs`
- [x] Progressive + metadata — `progressive_scan_edge_cases.rs`
- [x] Progressive + arithmetic — `sof10_encode.rs`
- [ ] Progressive + tiled decode — not implemented

---

## 8. YUV / Planar Operations

- [x] RGB → YUV → RGB roundtrip — `tjunittest_yuv.rs`
- [x] YUV plane size validation — `tjunittest_yuv.rs`
- [x] compress_from_yuv → decompress — `tjunittest_yuv.rs`
- [x] decompress_to_yuv → compress_from_yuv — `tjunittest_yuv.rs`
- [x] Multiple pixel formats for YUV — `tjunittest_yuv.rs`
- [x] Non-aligned dimensions — `tjunittest_yuv.rs`
- [ ] YUV no-padding mode — not tested
- [ ] Merged YUV decompression (420m, 422m) — not implemented

---

## 9. CMakeLists.txt add_bittest Specific Tests (48 tests)

These are the individual cjpeg/djpeg/jpegtran tests defined via `add_bittest()` macro:

### cjpeg Tests (Encode)
- [x] `444-islow` — baseline 444 integer DCT — `tjunittest_compat.rs`
- [x] `422-ifast-opt` — 422 fast DCT + optimized Huffman — `tjunittest_compat.rs`
- [x] `440-islow` — 440 subsampling — `tjunittest_compat.rs`
- [x] `420-q100-ifast-prog` — 420 quality=100 fast progressive — `tjunittest_compat.rs`
- [x] `gray-islow` — grayscale integer DCT — `tjunittest_compat.rs`
- [x] `420s-islow-opt` — 420 with smoothing + optimized — `niche_options.rs` (smoothing_factor)
- [ ] `3x2-float-prog` — 3x2 sampling float DCT progressive — non-standard sampling not supported
- [ ] `3x2-ifast-prog` — 3x2 sampling fast DCT progressive — non-standard sampling not supported
- [x] `420-islow-ari` — 420 arithmetic encoding — `arithmetic.rs`
- [x] `444-islow-progari` — 444 progressive + arithmetic — `sof10_encode.rs`
- [x] `rgb-islow` with ICC profile — `metadata_write.rs`
- [x] `lossless` (16-bit) — `precision.rs`

### djpeg Tests (Decode)
- [x] `rgb-islow` — baseline decode — `conformance.rs`
- [x] `422-ifast` — fast DCT decode — `decode_toggles.rs`
- [x] `440-islow` — 440 decode — `conformance.rs`
- [ ] `422m-ifast` — merged upsampling 422 — not implemented
- [ ] `420m-q100-ifast-prog` — merged upsampling 420 progressive — not implemented
- [x] `gray-islow` — grayscale decode — `conformance.rs`
- [x] `gray-islow-rgb` — grayscale to RGB output — `decode_toggles.rs`
- [x] `rgb-islow-565` — RGB565 decode (no dither) — `rgb565_dither.rs`
- [x] `rgb-islow-565D` — RGB565 decode (dithered) — `rgb565_dither.rs`
- [ ] `gray-islow-565` — grayscale to RGB565 — not tested
- [ ] `gray-islow-565D` — grayscale to dithered RGB565 — not tested
- [ ] `422m-ifast-565` — merged 422 RGB565 — not implemented
- [ ] `422m-ifast-565D` — merged 422 dithered RGB565 — not implemented
- [ ] `420m-islow-565` — merged 420 RGB565 — not implemented
- [ ] `420m-islow-565D` — merged 420 dithered RGB565 — not implemented
- [x] `420-islow-256` — 256-color quantized decode — `quantize.rs`
- [x] `420-islow-skip15_31` — scanline skip — `scanline_api.rs`
- [x] `420-islow-ari-skip16_139` — arithmetic + skip — partial
- [x] `420-islow-prog-crop62x62_71_71` — progressive + crop — `crop_skip.rs`
- [x] `444-islow-ari-crop37x37_0_0` — arithmetic + crop — `crop_skip.rs`
- [x] `420-islow-ari-crop53x53_4_4` — arithmetic + crop — partial
- [x] `444-islow-skip1_6` — small skip range — `scanline_api.rs`
- [ ] `420m-islow-{scale}` — merged upsampling with 15 scales — not implemented
- [x] `rgb-islow-icc-cmp` — ICC profile extraction — `metadata_write.rs`

### jpegtran Tests (Transform)
- [x] `icc` — ICC copy through transform — `marker_preservation.rs`
- [x] `crop` — lossless crop — `transform_options.rs`
- [x] `420-islow-ari` (as transform) — arithmetic recode — `tjunittest_transform.rs`
- [ ] `420m-ifast-ari` — merged upsampling arithmetic — not implemented

### 16-bit Specific (cjpeg16/djpeg16)
- [x] `lossless` — 16-bit lossless encode/decode — `precision.rs`

---

## 10. Specific C Test Features Not Covered

### Non-Standard Subsampling (3x2)
- [ ] 3x2 horizontal × 2 vertical sampling factor — C tests `3x2-float-prog` and `3x2-ifast-prog`
- [ ] Arbitrary H×V factor combinations beyond standard modes

### RGB565 Output (Dithered/Undithered)
- [x] RGB565 no-dither decode — `rgb565_dither.rs`, `pixel_formats.rs`
- [x] RGB565 dithered decode — `rgb565_dither.rs` (`Decoder::set_dither_565()`)
- [ ] RGB565 with grayscale input
- [ ] RGB565 with merged upsampling

### Merged Upsampling (420m, 422m)
- [ ] 420m decode (merged 2x2 upsample) — not implemented
- [ ] 422m decode (merged 2x1 upsample) — not implemented
- [ ] Merged upsampling + fast DCT
- [ ] Merged upsampling + scaled decode
- [ ] Merged upsampling + RGB565

### Smoothing Factor in Encode
- [x] `420s-islow-opt` (smooth=1) — `niche_options.rs`
- [ ] Smoothing factor validation via MD5/pixel comparison with C reference

### Example Programs
- [ ] `example-8bit-compress` — C example compile/run test
- [ ] `example-8bit-decompress` — C example compile/run test
- [ ] `example-12bit-compress/decompress` — 12-bit example test

### strtest (String Utilities)
- [ ] Internal string function tests — C-specific, not applicable to Rust

### bmpsizetest (BMP Size Calculations)
- [x] BMP dimension calculations — `image_io.rs`
- [ ] In-memory BMP I/O via fmemopen — Rust uses Vec directly

### Floating Point Variance Handling
- [ ] Different expected hashes per FPU (SSE, fp-contract, 387, MSVC)
- [ ] Platform-specific tolerance adjustment
- [ ] Compiler-specific test expectations

### Java API Tests
- [ ] TJUnitTest-bi (Java BufferedImage tests) — not applicable (Rust, no Java)
- [ ] TJUnitTest-bi-yuv — not applicable
- [ ] TJUnitTest-bi-lossless — not applicable

---

## 11. Features Unique to Our Implementation (Not in C)

- [x] Rust-specific: `Result<T, E>` error handling throughout
- [x] Builder pattern for Encoder/Decoder configuration
- [x] ProgressiveDecoder (scan-by-scan API, not in C TurboJPEG)
- [x] ScanlineEncoder/ScanlineDecoder (Rust-native streaming)
- [x] Color quantization with Floyd-Steinberg dithering
- [x] TjHandle (Rust port of C TJ3 handle pattern)
- [x] Thread safety verification (Send/Sync)
- [x] Property-based fuzzing with `arbitrary` crate
- [x] Concurrent encode/decode testing — `concurrency.rs`
- [x] Malformed JPEG robustness (37 tests) — `malformed_jpeg.rs`
- [x] Extreme dimension testing (50 tests) — `extreme_dimensions.rs`

---

## Summary

| Category | Done | Total | % |
|----------|------|-------|---|
| tjunittest equivalents | 14 | 17 | 82% |
| tjbench equivalents | 4 | 8 | 50% |
| Validation methods | 13 | 15 | 87% |
| tjcomptest matrix | 10 | 13 | 77% |
| tjdecomptest matrix | 7 | 11 | 64% |
| tjtrantest matrix | 8 | 10 | 80% |
| croptest | 1 | 5 | 20% |
| CMakeLists bittest (cjpeg) | 10 | 12 | 83% |
| CMakeLists bittest (djpeg) | 11 | 22 | 50% |
| CMakeLists bittest (jpegtran) | 3 | 4 | 75% |
| Precision coverage | 3 | 5 | 60% |
| Subsampling coverage | 7 | 8 | 88% |
| Scaling factors | 4 | 16 | 25% |
| Edge case / robustness | 23 | 25 | 92% |
| Metadata & markers | 10 | 11 | 91% |
| Progressive | 7 | 8 | 88% |
| YUV operations | 5 | 7 | 71% |
| Fuzzing | 7 | 8 | 88% |
| RGB565 specific | 0 | 6 | 0% |
| Merged upsampling | 0 | 5 | 0% |
| Non-standard sampling | 0 | 2 | 0% |
| FP variance handling | 0 | 3 | 0% |

### Key Gaps (Priority Order)
1. **Merged upsampling (420m, 422m)** — C tests extensively; we don't implement this optimization
2. **RGB565 dithered/undithered** — C tests 8 RGB565 combinations; we test 0
3. **MD5/binary comparison** — deterministic encoding + regression hashes implemented (`bitstream_stability.rs`, `bitstream_regression.rs`); cross-encoder comparison with C still missing
4. **Extended scaling factors** — C tests 15 scales; we test 4
5. **Non-standard sampling (3x2)** — C tests 3x2 float/ifast; we don't support arbitrary factors
6. **Tiled operations** — C tests 5 tile sizes; we have none
7. **Per-precision lossless (2-15 bit)** — C tests each; we only test 8/12/16
8. **Exhaustive crop matrix** — C tests 100+ crop regions; we test ~10
9. **DCT method cross-product** — C cross-products DCT × subsampling × quality; we test individually
10. **FP variance handling** — C has per-platform expected values; we don't
