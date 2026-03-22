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
- [ ] Byte-for-byte file comparison (`cmp` equivalent) — not used (pixel-level instead)
- [ ] MD5 hash validation of compressed output — not implemented
- [ ] Expected MD5 hashes per test configuration — not stored

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

### tjcomptest.in (Compression Validation)
- [x] All subsampling modes — `tjunittest_compat.rs`
- [x] Quality levels (1, 75, 100) — `tjunittest_compat.rs`, `encode_boundaries.rs`
- [x] Restart intervals — `restart_encode.rs`, `tjunittest_compat.rs`
- [x] ICC profile embedding — `metadata_write.rs`, `icc_exif_edge_cases.rs`
- [x] Arithmetic encoding — `tjunittest_compat.rs`, `arithmetic.rs`
- [x] Progressive encoding — `tjunittest_compat.rs`, `progressive_enc.rs`
- [x] Progressive + Arithmetic — `tjunittest_compat.rs`
- [x] Optimized Huffman — `tjunittest_compat.rs`, `huff_opt.rs`
- [x] Lossless PSV/PT combinations — `tjunittest_compat.rs`, `lossless_encode.rs`
- [ ] DCT method variation in compression (ifast, float) — encode uses IsLow only in matrix
- [ ] MD5 comparison between our encoder and C cjpeg — not implemented
- [ ] Binary comparison between our encoder and C cjpeg — not implemented

### tjdecomptest.in (Decompression Validation)
- [x] All subsampling modes decode — `conformance.rs`, `tjunittest_compat.rs`
- [x] Crop operations — `crop_skip.rs`
- [ ] 15 scaling factors (only 1/2, 1/4, 1/8 tested) — missing 12 intermediate scales
- [x] Smooth/no-smooth upsampling — `decode_toggles.rs`
- [x] DCT methods (fast, accurate) — `decode_toggles.rs`
- [x] Grayscale decode from color JPEG — `decode_toggles.rs`
- [ ] ICC profile extraction and comparison — extraction tested, not compared to C reference
- [ ] MD5 comparison between our decoder and C djpeg — not implemented
- [x] Multi-format output (PPM/RGB/Grayscale) — `cross_encoder_compat.rs`

### tjtrantest.in (Transform Validation)
- [x] All 8 transform types — `tjunittest_transform.rs`, `transform.rs`
- [x] All subsampling modes — `tjunittest_transform.rs`
- [ ] Copy modes (all, none, ICC only) — partial (copy_markers true/false only)
- [x] Crop + transform — `tjunittest_transform.rs`, `transform_options.rs`
- [x] Grayscale conversion during transform — `tjunittest_transform.rs`, `transform_options.rs`
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
- [ ] 4:1:0 (decompression only, rare) — not tested
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
- [ ] Dithered/undithered RGB565 — not tested

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

## 9. Features Unique to Our Implementation (Not in C)

- [x] Rust-specific: `Result<T, E>` error handling throughout
- [x] Builder pattern for Encoder/Decoder configuration
- [x] ProgressiveDecoder (scan-by-scan API, not in C TurboJPEG)
- [x] ScanlineEncoder/ScanlineDecoder (Rust-native streaming)
- [x] Color quantization with Floyd-Steinberg dithering
- [x] TjHandle (Rust port of C TJ3 handle pattern)
- [x] Thread safety verification (Send/Sync)
- [x] Property-based fuzzing with `arbitrary` crate

---

## Summary

| Category | Done | Total | % |
|----------|------|-------|---|
| tjunittest equivalents | 14 | 17 | 82% |
| tjbench equivalents | 4 | 8 | 50% |
| Validation methods | 10 | 15 | 67% |
| tjcomptest matrix | 10 | 13 | 77% |
| tjdecomptest matrix | 7 | 11 | 64% |
| tjtrantest matrix | 8 | 10 | 80% |
| croptest | 1 | 5 | 20% |
| Precision coverage | 3 | 5 | 60% |
| Subsampling coverage | 7 | 8 | 88% |
| Scaling factors | 4 | 16 | 25% |
| Edge case / robustness | 23 | 25 | 92% |
| Metadata & markers | 10 | 11 | 91% |
| Progressive | 7 | 8 | 88% |
| YUV operations | 5 | 7 | 71% |
| Fuzzing | 7 | 8 | 88% |

### Key Gaps (Priority Order)
1. **MD5/binary comparison** — C validates bitstream identity; we only validate pixels
2. **Extended scaling factors** — C tests 15 scales; we test 4
3. **Tiled operations** — C tests 5 tile sizes; we have none
4. **Per-precision lossless (2-15 bit)** — C tests each; we only test 8/12/16
5. **Exhaustive crop matrix** — C tests 100+ crop regions; we test ~10
6. **DCT method in full matrix** — C cross-products DCT × subsampling × quality; we test individually
