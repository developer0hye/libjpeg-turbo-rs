# Feature Parity: libjpeg-turbo-rs vs libjpeg-turbo (C)

> Track implementation progress. Update checkboxes when features are completed.
> Source of truth: `turbojpeg.h` (TJ3 API), `jpeglib.h` (libjpeg API), `jmorecfg.h`

---

## 1. Frame Types (SOF Markers)

### Encode
- [x] SOF0 — Baseline DCT, Huffman
- [x] SOF2 — Progressive DCT, Huffman
- [x] SOF3 — Lossless, Huffman (grayscale + color, predictor 1-7, pt 0-15)
- [x] SOF9 — Sequential DCT, Arithmetic
- [x] SOF10 — Progressive DCT, Arithmetic
- [x] SOF11 — Lossless, Arithmetic

### Decode
- [x] SOF0 — Baseline DCT, Huffman
- [x] SOF2 — Progressive DCT, Huffman
- [x] SOF3 — Lossless, Huffman (1 and 3 component)
- [x] SOF9 — Sequential DCT, Arithmetic
- [x] SOF10 — Progressive DCT, Arithmetic
- [x] SOF11 — Lossless, Arithmetic

---

## 2. Sample Precision

- [x] 8-bit (`JSAMPLE` / `u8`)
- [x] 12-bit (`J12SAMPLE` / `i16`) — `tj3Compress12`, `tj3Decompress12`, `jpeg12_write_scanlines`, `jpeg12_read_scanlines`
- [x] 16-bit (`J16SAMPLE` / `u16`, lossless only) — `tj3Compress16`, `tj3Decompress16`, `jpeg16_write_scanlines`, `jpeg16_read_scanlines`

---

## 3. Pixel Formats (TJPF)

- [x] TJPF_GRAY — Grayscale (1 bpp)
- [x] TJPF_RGB — RGB (3 bpp)
- [x] TJPF_BGR — BGR (3 bpp)
- [x] TJPF_RGBA — RGBA (4 bpp)
- [x] TJPF_BGRA — BGRA (4 bpp)
- [x] TJPF_CMYK — CMYK (4 bpp)
- [x] TJPF_RGBX — RGB + pad (4 bpp, no alpha)
- [x] TJPF_BGRX — BGR + pad (4 bpp, no alpha)
- [x] TJPF_XBGR — pad + BGR (4 bpp)
- [x] TJPF_XRGB — pad + RGB (4 bpp)
- [x] TJPF_ABGR — alpha + BGR (4 bpp)
- [x] TJPF_ARGB — alpha + RGB (4 bpp)
- [x] TJPF_RGB565 — 5-6-5 packed (decode only)

---

## 4. Chroma Subsampling (TJSAMP)

- [x] TJSAMP_444 (4:4:4)
- [x] TJSAMP_422 (4:2:2)
- [x] TJSAMP_420 (4:2:0)
- [x] TJSAMP_GRAY (grayscale)
- [x] TJSAMP_440 (4:4:0)
- [x] TJSAMP_411 (4:1:1)
- [x] TJSAMP_441 (4:4:1)
- [ ] TJSAMP_UNKNOWN (unusual/custom subsampling detection)

---

## 5. Color Spaces (TJCS / J_COLOR_SPACE)

- [x] JCS_GRAYSCALE
- [x] JCS_YCbCr
- [x] JCS_RGB
- [x] JCS_CMYK
- [x] JCS_YCCK
- [ ] JCS_UNKNOWN (pass-through, no conversion)

---

## 6. Compression Parameters (TJPARAM / jpeg_compress_struct fields)

### Quality & Quantization
- [x] `TJPARAM_QUALITY` — Quality factor 1-100 (`jpeg_set_quality`)
- [ ] `q_scale_factor[NUM_QUANT_TBLS]` — Per-component quality
- [x] `jpeg_add_quant_table()` — Custom quantization table (`Encoder::quant_table()`)
- [ ] `jpeg_set_linear_quality()` — Linear quality scaling
- [ ] `jpeg_default_qtables()` — Reset to default tables
- [ ] `jpeg_quality_scaling()` — Quality to scale factor conversion
- [ ] `force_baseline` parameter — Constrain quant values to 1-255

### Huffman Tables
- [x] Standard DC/AC luminance + chrominance tables
- [x] `TJPARAM_OPTIMIZE` — 2-pass optimized Huffman (`compress_optimized`)
- [x] Custom `dc_huff_tbl_ptrs[4]` — User-supplied DC Huffman tables (`Encoder::huffman_dc_table()`)
- [x] Custom `ac_huff_tbl_ptrs[4]` — User-supplied AC Huffman tables (`Encoder::huffman_ac_table()`)
- [ ] `jpeg_alloc_huff_table()` — Allocate table
- [ ] `jpeg_suppress_tables()` — Table suppression control

### Entropy Coding Mode
- [x] `TJPARAM_PROGRESSIVE` — Progressive mode
- [x] `TJPARAM_ARITHMETIC` — Arithmetic coding
- [x] `TJPARAM_ARITHMETIC` + `TJPARAM_PROGRESSIVE` combined — SOF10 encode

### Lossless Mode
- [x] `TJPARAM_LOSSLESS` — Enable lossless
- [x] `TJPARAM_LOSSLESSPSV` — Predictor selection 1-7 (`Encoder::lossless_predictor()`)
- [x] `TJPARAM_LOSSLESSPT` — Point transform 0-15 (`Encoder::lossless_point_transform()`)
- [x] Lossless multi-component (color) encode (`compress_lossless_extended()`)
- [x] `jpeg_enable_lossless()` — Combined predictor + pt setup (via Encoder builder)

### Restart Markers
- [x] `TJPARAM_RESTARTBLOCKS` — Restart interval in MCU blocks (`Encoder::restart_blocks()`)
- [x] `TJPARAM_RESTARTROWS` — Restart interval in MCU rows (`Encoder::restart_rows()`)
- [x] `restart_interval` field — via Encoder builder
- [x] `restart_in_rows` field — via Encoder builder

### JFIF / Density
- [x] `write_JFIF_header` — JFIF marker (always written, hardcoded 72 DPI)
- [x] `TJPARAM_XDENSITY` — Horizontal pixel density (`DensityInfo`)
- [x] `TJPARAM_YDENSITY` — Vertical pixel density (`DensityInfo`)
- [x] `TJPARAM_DENSITYUNITS` — Units (`DensityUnit` enum)
- [ ] `JFIF_major_version` / `JFIF_minor_version` configurable
- [x] `density_unit` / `X_density` / `Y_density` configurable (read from JFIF, write via `DensityInfo`)

### Adobe Marker
- [x] `write_Adobe_marker` — Adobe APP14 (for CMYK)
- [ ] `write_Adobe_marker` toggle — Enable/disable

### Progressive Scan Control
- [x] `jpeg_simple_progression()` — Standard scan script
- [x] `scan_info` / `num_scans` — Custom scan progression script (`Encoder::scan_script()`)
- [x] `jpeg_scan_info` struct — `ScanScript` struct

### DCT Method
- [x] `JDCT_ISLOW` — Accurate integer DCT
- [x] `JDCT_IFAST` — Fast integer DCT (`DctMethod::IsFast`)
- [x] `JDCT_FLOAT` — Floating-point DCT (`DctMethod::Float`)

### Color Space Control
- [x] Auto YCbCr from RGB/RGBA/BGR/BGRA input
- [x] CMYK direct (no conversion)
- [ ] `jpeg_set_colorspace()` — Explicit colorspace override
- [ ] `jpeg_default_colorspace()` — Reset to default
- [ ] `in_color_space` / `jpeg_color_space` independent control
- [x] Grayscale-from-color encode option (`Encoder::grayscale_from_color()`)

### Input Options
- [ ] `TJPARAM_BOTTOMUP` — Bottom-up row order
- [ ] `raw_data_in` — Encode from raw downsampled component data
- [ ] `smoothing_factor` — Input smoothing (0-100)
- [ ] `do_fancy_downsampling` — Fancy vs simple chroma downsample
- [ ] `CCIR601_sampling` — CCIR 601 sampling convention
- [ ] `input_gamma` — Input gamma correction

### Marker Writing
- [x] JFIF APP0 (automatic)
- [x] EXIF APP1 (`compress_with_metadata`)
- [x] ICC APP2 (`compress_with_metadata`, multi-chunk)
- [x] Adobe APP14 (CMYK encode)
- [x] `jpeg_write_marker()` — Write arbitrary marker data (`marker_writer::write_marker()`)
- [ ] `jpeg_write_m_header()` / `jpeg_write_m_byte()` — Streaming marker write
- [ ] `jpeg_write_icc_profile()` — Standalone ICC write (without full compress)
- [ ] `jpeg_write_tables()` — Write tables-only JPEG
- [x] COM (comment) marker write (`Encoder::comment()`, `marker_writer::write_com()`)

### Scanline-Level Encode API
- [x] `jpeg_start_compress()` — Begin compression (`ScanlineEncoder::new()`)
- [x] `jpeg_write_scanlines()` — Write scanline rows (`ScanlineEncoder::write_scanlines()`)
- [x] `jpeg_finish_compress()` — Finalize compression (`ScanlineEncoder::finish()`)
- [x] `jpeg_write_raw_data()` — Write raw downsampled data (`compress_raw()`)
- [ ] `jpeg12_write_scanlines()` — 12-bit scanlines
- [ ] `jpeg16_write_scanlines()` — 16-bit scanlines
- [ ] `jpeg_calc_jpeg_dimensions()` — Compute output dimensions
- [x] `next_scanline` tracking (`ScanlineEncoder::next_scanline()`)

---

## 7. Decompression Parameters (TJPARAM / jpeg_decompress_struct fields)

### Output Format
- [x] Output pixel format selection (`decompress_to`)
- [x] Scaled IDCT — 1/1, 1/2, 1/4, 1/8 (`set_scale`)
- [x] Crop decode (`decompress_cropped`, `set_crop_region`)
- [ ] `TJPARAM_BOTTOMUP` — Bottom-up row order
- [ ] `out_color_space` — Explicit output colorspace
- [x] YCbCr/YUV raw output (skip color conversion) (`decompress_raw()`)
- [x] `raw_data_out` — Raw downsampled component output (`decompress_raw()`)

### Upsampling / DCT
- [x] Fancy upsampling (default, always on)
- [ ] `TJPARAM_FASTUPSAMPLE` — Nearest-neighbor upsampling toggle
- [ ] `do_fancy_upsampling` toggle
- [ ] `TJPARAM_FASTDCT` — Fast IDCT vs accurate toggle
- [ ] `do_block_smoothing` toggle
- [ ] `dct_method` selection (ISLOW/IFAST/FLOAT)

### Error Handling
- [x] Lenient / error recovery mode (`decompress_lenient`)
- [x] `DecodeWarning` list in Image
- [x] `TJPARAM_STOPONWARNING` — Treat warnings as fatal (`Decoder::set_stop_on_warning()`)
- [x] `TJPARAM_SCANLIMIT` — Max progressive scans before error (`Decoder::set_scan_limit()`)
- [x] Custom error callbacks — `ErrorHandler` trait

### Limits
- [x] `TJPARAM_MAXMEMORY` — Memory limit (`Decoder::set_max_memory()`)
- [x] `TJPARAM_MAXPIXELS` — Image size limit (`Decoder::set_max_pixels()`)

### Marker Handling
- [x] ICC profile reassembly from APP2 chunks
- [x] EXIF extraction + orientation (APP1)
- [x] Adobe APP14 detection (CMYK/YCCK)
- [x] Restart marker (DRI/RST) handling
- [x] `TJPARAM_SAVEMARKERS` — Configurable marker saving (`MarkerSaveConfig` enum: None/All/AppOnly/Specific)
- [x] `jpeg_save_markers()` — Per-marker-type save control (`Decoder::save_markers()`)
- [ ] `jpeg_set_marker_processor()` — Custom marker parser callback
- [x] COM (comment) marker read/expose (`Image.comment`)
- [x] Arbitrary marker access via `marker_list` linked list (`Image.markers()` / `Image.saved_markers`)
- [x] JFIF version / density read (`Image.density`)

### Multi-Scan / Progressive Output
- [x] `jpeg_has_multiple_scans()` — Query progressive (`ProgressiveDecoder::has_multiple_scans()`)
- [x] `buffered_image` mode — Enable scan-by-scan output (`ProgressiveDecoder`)
- [x] `jpeg_start_output()` / `jpeg_finish_output()` — Per-scan output control (`ProgressiveDecoder::output()` / `ProgressiveDecoder::finish()`)
- [x] `jpeg_consume_input()` — Incremental input processing (`ProgressiveDecoder::consume_input()`)
- [x] `jpeg_input_complete()` — Check if all input consumed (`ProgressiveDecoder::input_complete()`)

### Scanline-Level Decode API
- [x] `jpeg_read_header()` — Parse headers (`ScanlineDecoder::new()`)
- [x] `jpeg_start_decompress()` — Begin decompression (`ScanlineDecoder::new()`)
- [x] `jpeg_read_scanlines()` — Read scanline rows (`ScanlineDecoder::read_scanlines()`)
- [x] `jpeg_skip_scanlines()` — Skip rows during decode (`ScanlineDecoder::skip_scanlines()`)
- [ ] `jpeg_crop_scanline()` — Scanline-level horizontal crop
- [x] `jpeg_finish_decompress()` — Finalize decompression (`ScanlineDecoder::finish()`)
- [x] `jpeg_read_raw_data()` — Read raw downsampled data (`decompress_raw()`)
- [ ] `jpeg12_read_scanlines()` / `jpeg12_skip_scanlines()` / `jpeg12_crop_scanline()`
- [ ] `jpeg16_read_scanlines()`
- [ ] `jpeg_calc_output_dimensions()` / `jpeg_core_output_dimensions()`
- [x] `output_scanline` tracking (`ScanlineDecoder::output_scanline()`)

### Color Quantization (8-bit indexed output)
- [x] `quantize_colors` — Enable color quantization (`quantize::quantize()`)
- [x] `desired_number_of_colors` / `actual_number_of_colors` (`QuantizeOptions::num_colors`, `QuantizedImage::palette.len()`)
- [x] `dither_mode` — JDITHER_NONE / JDITHER_ORDERED / JDITHER_FS (`DitherMode` enum)
- [x] `two_pass_quantize` — Two-pass color selection (`QuantizeOptions::two_pass`, median-cut algorithm)
- [x] `colormap` — External colormap input (`QuantizeOptions::colormap`)
- [x] `enable_1pass_quant` / `enable_2pass_quant` / `enable_external_quant` (`QuantizeOptions::two_pass` + `colormap`)
- [ ] `jpeg_new_colormap()` — Update colormap

---

## 8. Metadata

- [x] APP0 JFIF — Read / write
- [x] APP1 EXIF — Read / write (orientation parsing)
- [x] APP2 ICC profile — Read (multi-chunk reassembly) / write (multi-chunk)
- [x] APP14 Adobe — Read / write (CMYK/YCCK signaling)
- [x] COM (comment) — Read (`Image.comment`) / Write (`Encoder::comment()`)
- [x] Arbitrary APP markers — Read (`Decoder::save_markers()` + `Image.markers()`)
- [x] Arbitrary markers — Write (`marker_writer::write_marker()`, `Encoder::saved_marker()`)
- [x] DPI/density — Read (`Image.density`) / Write (`DensityInfo`)
- [ ] JFIF thumbnail extraction
- [x] Marker preservation across transform/re-encode (`TransformOptions.copy_markers`)

---

## 9. Transform API

### Operations (TJXOP)
- [x] TJXOP_NONE
- [x] TJXOP_HFLIP
- [x] TJXOP_VFLIP
- [x] TJXOP_TRANSPOSE
- [x] TJXOP_TRANSVERSE
- [x] TJXOP_ROT90
- [x] TJXOP_ROT180
- [x] TJXOP_ROT270

### Options (TJXOPT flags)
- [x] TJXOPT_PERFECT (1) — Fail if transform is not perfect (partial iMCU) (`TransformOptions.perfect`)
- [x] TJXOPT_TRIM (2) — Discard partial iMCU edges (`TransformOptions.trim`)
- [x] TJXOPT_CROP (4) — Enable lossless cropping region (`TransformOptions.crop`)
- [x] TJXOPT_GRAY (8) — Convert to grayscale during transform (`TransformOptions.grayscale`)
- [x] TJXOPT_NOOUTPUT (16) — Dry run (no output image) (`TransformOptions.no_output`)
- [x] TJXOPT_PROGRESSIVE (32) — Output as progressive JPEG (`TransformOptions.progressive`)
- [x] TJXOPT_COPYNONE (64) — Discard all non-essential markers (`TransformOptions.copy_markers = false`)
- [x] TJXOPT_ARITHMETIC (128) — Output with arithmetic coding (`TransformOptions.arithmetic`)
- [x] TJXOPT_OPTIMIZE (256) — Output with optimized Huffman (`TransformOptions.optimize`)

### Coefficient Access
- [x] `read_coefficients()` — Extract quantized DCT blocks
- [x] `write_coefficients()` — Encode from coefficient blocks
- [x] `transform_jpeg()` — Apply spatial transform
- [ ] `jpeg_copy_critical_parameters()` — Copy tables between compress/decompress
- [x] `tjtransform.customFilter` — User callback for coefficient inspection/modification
- [ ] `tj3TransformBufSize()` — Output buffer size estimation

---

## 10. YUV / Planar API

### RGB → YUV (color conversion only, no JPEG)
- [x] `tj3EncodeYUV8()` — RGB → packed YUV buffer (`yuv::encode_yuv()`)
- [x] `tj3EncodeYUVPlanes8()` — RGB → separate Y/Cb/Cr plane buffers (`yuv::encode_yuv_planes()`)

### YUV → JPEG (compress from YUV)
- [x] `tj3CompressFromYUV8()` — Packed YUV → JPEG (`yuv::compress_from_yuv()`)
- [x] `tj3CompressFromYUVPlanes8()` — Planar YUV → JPEG (`yuv::compress_from_yuv_planes()`)

### JPEG → YUV (decompress to YUV)
- [x] `tj3DecompressToYUV8()` — JPEG → packed YUV buffer (`yuv::decompress_to_yuv()`)
- [x] `tj3DecompressToYUVPlanes8()` — JPEG → separate Y/Cb/Cr plane buffers (`yuv::decompress_to_yuv_planes()`)

### YUV → RGB (color conversion only, no JPEG)
- [x] `tj3DecodeYUV8()` — Packed YUV → RGB (`yuv::decode_yuv()`)
- [x] `tj3DecodeYUVPlanes8()` — Planar YUV → RGB (`yuv::decode_yuv_planes()`)

### Buffer Size Helpers
- [x] `tj3YUVBufSize()` — Total packed YUV buffer size (`yuv_buf_size()`)
- [x] `tj3YUVPlaneSize()` — Single plane buffer size (`yuv_plane_size()`)
- [x] `tj3YUVPlaneWidth()` — Plane width in samples (`yuv_plane_width()`)
- [x] `tj3YUVPlaneHeight()` — Plane height in rows (`yuv_plane_height()`)

---

## 11. SIMD

### aarch64 (ARM NEON)
- [x] IDCT with dequantization (8x8)
- [x] YCbCr → RGB row conversion
- [x] YCbCr → RGBA row conversion
- [x] YCbCr → BGR row conversion
- [x] YCbCr → BGRA row conversion
- [x] Fancy H2V1 upsample
- [x] Fancy H2V2 upsample
- [x] Forward DCT (FDCT) for encoder
- [x] Chroma downsample for encoder
- [x] Quantization for encoder
- [x] Scaled IDCT (4x4, 2x2, 1x1) NEON variants
- [x] RGB → YCbCr (encode-side color conversion)

### x86_64
- [x] SSE2 IDCT
- [x] SSE2 color conversion (YCbCr→RGB)
- [x] SSE2 upsample (H2V1, H2V2)
- [x] AVX2 IDCT
- [x] AVX2 color conversion
- [x] AVX2 upsample

### General
- [x] Scalar fallback for all operations
- [x] Runtime SIMD feature detection (`simd::detect()`)

---

## 12. Memory, I/O, Buffer Management

### Source / Destination
- [x] Memory-to-memory compress (`Vec<u8>` output)
- [x] Memory-to-memory decompress (byte slice → `Image`)
- [x] `jpeg_stdio_dest()` / `jpeg_stdio_src()` — File I/O (`stream::compress_to_file` / `stream::decompress_from_file`)
- [x] `jpeg_mem_dest()` / `jpeg_mem_src()` — C memory I/O (Rust equivalent: already native)
- [x] Custom `jpeg_destination_mgr` — User-defined output stream (`stream::compress_to_writer`)
- [x] Custom `jpeg_source_mgr` — User-defined input stream (`stream::decompress_from_reader`)
- [ ] `TJPARAM_NOREALLOC` — Pre-allocated output buffer

### Buffer Size Calculation
- [ ] `tj3JPEGBufSize()` — Worst-case JPEG output size
- [ ] `tj3YUVBufSize()` — YUV buffer size
- [ ] `tj3TransformBufSize()` — Transform output buffer size

### Image File I/O (BMP/PPM)
- [x] `tj3LoadImage8()` / `tj3LoadImage12()` / `tj3LoadImage16()` — 8-bit implemented (`load_image` / `load_image_from_bytes`)
- [x] `tj3SaveImage8()` / `tj3SaveImage12()` / `tj3SaveImage16()` — 8-bit implemented (`save_bmp` / `save_ppm`)

### Memory Management
- [ ] Custom `jpeg_memory_mgr` — Pool-based allocator
- [ ] `alloc_small` / `alloc_large` / `alloc_sarray` / `alloc_barray`
- [ ] `request_virt_sarray` / `request_virt_barray` / `realize_virt_arrays` / `access_virt_sarray` / `access_virt_barray`
- [ ] `free_pool` / `self_destruct`
- [ ] `max_memory_to_use` / `max_alloc_chunk`
- [ ] `tj3Alloc()` / `tj3Free()` — TurboJPEG allocator

---

## 13. Error Handling

- [x] `Result<T, JpegError>` for all public operations
- [x] `DecodeWarning` list (HuffmanError, TruncatedData) in lenient mode
- [x] Custom error handler — `ErrorHandler` trait
- [x] `error_exit()` callback — `ErrorHandler::error_exit()`
- [x] `emit_message()` callback — `ErrorHandler::emit_warning()` + `ErrorHandler::trace()`
- [ ] `output_message()` callback — Error text display
- [ ] `format_message()` callback — Error string formatting
- [ ] `reset_error_mgr()` callback
- [ ] `trace_level` control
- [ ] `num_warnings` counter
- [ ] `msg_code` / `msg_parm` — Structured error info
- [ ] `jpeg_message_table` / `addon_message_table` — Message customization
- [ ] `tj3GetErrorStr()` / `tj3GetErrorCode()` equivalents
- [ ] `jpeg_resync_to_restart()` — Restart resynchronization

---

## 14. Progress Monitoring

- [x] `jpeg_progress_mgr` struct — `ProgressListener` trait
- [x] `progress_monitor()` callback — `ProgressListener::update()` (closure support)
- [x] `pass_counter` / `pass_limit` — `ProgressInfo.progress`
- [x] `completed_passes` / `total_passes` — `ProgressInfo.pass` / `ProgressInfo.total_passes`

---

## 15. TJ3 Handle / Parameter API

- [x] `tj3Init()` / `tj3Destroy()` — Handle lifecycle (`TjHandle::new()` / Drop)
- [x] `tj3Set()` / `tj3Get()` — Generic parameter get/set (`TjHandle::set()` / `TjHandle::get()`)
- [x] All 26 TJPARAM values as runtime parameters (`TjParam` enum)
- [x] `tj3SetICCProfile()` / `tj3GetICCProfile()` — ICC via handle (`TjHandle::set_icc_profile()` / `TjHandle::icc_profile()`)
- [x] `tj3SetScalingFactor()` / `tj3SetCroppingRegion()` — Decode options via handle (`TjHandle::set_scaling_factor()` / `TjHandle::set_cropping_region()`)
- [x] `tj3GetScalingFactors()` — Query available scaling factors (`TjHandle::scaling_factors()`)

---

## Summary

| Category | Done | Total | % |
|----------|------|-------|---|
| Frame types (encode) | 6 | 6 | 100% |
| Frame types (decode) | 6 | 6 | 100% |
| Sample precision | 3 | 3 | 100% |
| Pixel formats | 13 | 13 | 100% |
| Chroma subsampling | 7 | 8 | 88% |
| Color spaces | 5 | 6 | 83% |
| Compress params | ~40 | ~65 | ~62% |
| Decompress params | ~30 | ~55 | ~55% |
| Metadata | 10 | 10 | 100% |
| Transform ops | 8 | 8 | 100% |
| Transform options | 9 | 9 | 100% |
| Transform misc | 4 | 6 | 67% |
| YUV/Planar API | 12 | 12 | 100% |
| SIMD (aarch64) | 10 | 12 | 83% |
| SIMD (x86_64) | 6 | 6 | 100% |
| Memory & I/O | 8 | ~20 | ~40% |
| Error handling | 5 | ~14 | ~36% |
| Progress | 4 | 4 | 100% |
| TJ3 Handle API | 6 | 6 | 100% |

---

## Priority Roadmap

> Strategy: feature completeness first, then SIMD/performance.

### Phase 4 — Core Feature Gaps ✅ COMPLETE
| # | Feature | PR |
|---|---------|-----|
| 1 | ~~Restart interval encode (DRI)~~ | #18 |
| 2 | ~~COM marker read/write + density~~ | #17 |
| 3 | ~~Lossless encode: color + predictor + pt~~ | #20 |
| 4 | ~~SOF10 arithmetic progressive encode~~ | #25 |
| 5 | ~~SOF11 lossless arithmetic encode/decode~~ | #26 |
| 6 | ~~Grayscale-from-color encode~~ | #22 |
| 7 | ~~Configurable DPI/density~~ | #17 |
| 8 | ~~Arbitrary marker write~~ | #17 |

### Phase 5 — Extended Formats ✅ COMPLETE
| # | Feature | Status |
|---|---------|--------|
| 9 | 12-bit precision | ✅ |
| 10 | 16-bit precision (lossless) | ✅ |
| 11 | ~~Custom quantization tables~~ | ✅ #19 |
| 12 | ~~Custom Huffman tables~~ | ✅ #23 |
| 13 | ~~Custom progressive scan scripts~~ | ✅ #24 |
| 14 | ~~Additional pixel formats~~ | ✅ #28 |
| 15 | ~~Fast DCT (IsFast, Float)~~ | ✅ #27 |
| 16 | ~~S441 subsampling~~ | ✅ #29 |

### Phase 6 — Transform & Advanced ✅ COMPLETE (8/8)
| # | Feature | Status |
|---|---------|--------|
| 17 | ~~All 9 TJXOPT transform flags~~ | ✅ #32 |
| 18 | ~~Coefficient filter callback~~ | ✅ #34 |
| 19 | ~~Marker preservation~~ | ✅ #36 |
| 20 | ~~Scanline-level encode API~~ | ✅ #33 |
| 21 | ~~Scanline-level decode API~~ | ✅ #33 |
| 22 | ~~Progressive output (buffered image)~~ | ✅ |
| 23 | ~~Per-component quality~~ | ✅ #31 |
| 24 | ~~Raw data encode/decode~~ | ✅ #35 |

### Phase 7 — YUV & I/O ✅ COMPLETE
| # | Feature | Status |
|---|---------|--------|
| 25 | ~~YUV planar encode/decode~~ | ✅ #40 |
| 26 | ~~Buffer size calculation~~ | ✅ #30 |
| 27 | ~~Custom source/dest managers~~ | ✅ #39 |
| 28 | ~~File I/O helpers~~ | ✅ #38 |

### Phase 8 — SIMD & Performance ✅ COMPLETE
| # | Feature | Status |
|---|---------|--------|
| 29 | ~~x86_64 SSE2~~ | ✅ #44 |
| 30 | ~~x86_64 AVX2~~ | ✅ #43 |
| 31 | ~~aarch64 NEON extensions~~ | ✅ #42 |

### Phase 9 — Full API Parity ✅ COMPLETE (practical features)
| # | Feature | Status |
|---|---------|--------|
| 32 | ~~TJ3 handle/parameter API~~ | ✅ #45 |
| 33 | Custom error manager | ✅ Already done (ErrorHandler trait, PR #17) |
| 34 | Progress monitoring | ✅ Already done (ProgressListener trait) |
| 35 | ~~Color quantization~~ | ✅ #46 |
| 36 | Custom memory manager | ⬜ N/A in Rust (std allocator) |
