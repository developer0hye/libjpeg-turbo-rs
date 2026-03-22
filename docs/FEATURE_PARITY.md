# Feature Parity: libjpeg-turbo-rs vs libjpeg-turbo (C)

> Track implementation progress. Update checkboxes when features are completed.
> Source of truth: `turbojpeg.h` (TJ3 API), `jpeglib.h` (libjpeg API), `jmorecfg.h`

---

## 1. Frame Types (SOF Markers)

### Encode
- [x] SOF0 — Baseline DCT, Huffman
- [x] SOF2 — Progressive DCT, Huffman
- [x] SOF3 — Lossless, Huffman (grayscale only, predictor 1 only, pt=0 only)
- [x] SOF9 — Sequential DCT, Arithmetic
- [ ] SOF10 — Progressive DCT, Arithmetic
- [ ] SOF11 — Lossless, Arithmetic

### Decode
- [x] SOF0 — Baseline DCT, Huffman
- [x] SOF2 — Progressive DCT, Huffman
- [x] SOF3 — Lossless, Huffman (1 and 3 component)
- [x] SOF9 — Sequential DCT, Arithmetic
- [x] SOF10 — Progressive DCT, Arithmetic (basic)
- [ ] SOF11 — Lossless, Arithmetic

---

## 2. Sample Precision

- [x] 8-bit (`JSAMPLE` / `u8`)
- [ ] 12-bit (`J12SAMPLE` / `i16`) — `tj3Compress12`, `tj3Decompress12`, `jpeg12_write_scanlines`, `jpeg12_read_scanlines`
- [ ] 16-bit (`J16SAMPLE` / `u16`, lossless only) — `tj3Compress16`, `tj3Decompress16`, `jpeg16_write_scanlines`, `jpeg16_read_scanlines`

---

## 3. Pixel Formats (TJPF)

- [x] TJPF_GRAY — Grayscale (1 bpp)
- [x] TJPF_RGB — RGB (3 bpp)
- [x] TJPF_BGR — BGR (3 bpp)
- [x] TJPF_RGBA — RGBA (4 bpp)
- [x] TJPF_BGRA — BGRA (4 bpp)
- [x] TJPF_CMYK — CMYK (4 bpp)
- [ ] TJPF_RGBX — RGB + pad (4 bpp, no alpha)
- [ ] TJPF_BGRX — BGR + pad (4 bpp, no alpha)
- [ ] TJPF_XBGR — pad + BGR (4 bpp)
- [ ] TJPF_XRGB — pad + RGB (4 bpp)
- [ ] TJPF_ABGR — alpha + BGR (4 bpp)
- [ ] TJPF_ARGB — alpha + RGB (4 bpp)
- [ ] TJPF_RGB565 — 5-6-5 packed (decode only)

---

## 4. Chroma Subsampling (TJSAMP)

- [x] TJSAMP_444 (4:4:4)
- [x] TJSAMP_422 (4:2:2)
- [x] TJSAMP_420 (4:2:0)
- [x] TJSAMP_GRAY (grayscale)
- [x] TJSAMP_440 (4:4:0)
- [x] TJSAMP_411 (4:1:1)
- [ ] TJSAMP_441 (4:4:1)
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
- [ ] `jpeg_add_quant_table()` — Custom quantization table
- [ ] `jpeg_set_linear_quality()` — Linear quality scaling
- [ ] `jpeg_default_qtables()` — Reset to default tables
- [ ] `jpeg_quality_scaling()` — Quality to scale factor conversion
- [ ] `force_baseline` parameter — Constrain quant values to 1-255

### Huffman Tables
- [x] Standard DC/AC luminance + chrominance tables
- [x] `TJPARAM_OPTIMIZE` — 2-pass optimized Huffman (`compress_optimized`)
- [ ] Custom `dc_huff_tbl_ptrs[4]` — User-supplied DC Huffman tables
- [ ] Custom `ac_huff_tbl_ptrs[4]` — User-supplied AC Huffman tables
- [ ] `jpeg_alloc_huff_table()` — Allocate table
- [ ] `jpeg_suppress_tables()` — Table suppression control

### Entropy Coding Mode
- [x] `TJPARAM_PROGRESSIVE` — Progressive mode
- [x] `TJPARAM_ARITHMETIC` — Arithmetic coding
- [ ] `TJPARAM_ARITHMETIC` + `TJPARAM_PROGRESSIVE` combined — SOF10 encode

### Lossless Mode
- [x] `TJPARAM_LOSSLESS` — Enable lossless
- [ ] `TJPARAM_LOSSLESSPSV` — Predictor selection 1-7 (only 1 implemented)
- [ ] `TJPARAM_LOSSLESSPT` — Point transform 0-15 (only 0 implemented)
- [ ] Lossless multi-component (color) encode
- [ ] `jpeg_enable_lossless()` — Combined predictor + pt setup

### Restart Markers
- [ ] `TJPARAM_RESTARTBLOCKS` — Restart interval in MCU blocks
- [ ] `TJPARAM_RESTARTROWS` — Restart interval in MCU rows
- [ ] `restart_interval` field in jpeg_compress_struct
- [ ] `restart_in_rows` field in jpeg_compress_struct

### JFIF / Density
- [x] `write_JFIF_header` — JFIF marker (always written, hardcoded 72 DPI)
- [ ] `TJPARAM_XDENSITY` — Horizontal pixel density (configurable)
- [ ] `TJPARAM_YDENSITY` — Vertical pixel density (configurable)
- [ ] `TJPARAM_DENSITYUNITS` — Units (0=unknown, 1=ppi, 2=ppcm)
- [ ] `JFIF_major_version` / `JFIF_minor_version` configurable
- [ ] `density_unit` / `X_density` / `Y_density` configurable

### Adobe Marker
- [x] `write_Adobe_marker` — Adobe APP14 (for CMYK)
- [ ] `write_Adobe_marker` toggle — Enable/disable

### Progressive Scan Control
- [x] `jpeg_simple_progression()` — Standard scan script
- [ ] `scan_info` / `num_scans` — Custom scan progression script
- [ ] `jpeg_scan_info` struct — User-defined scan parameters

### DCT Method
- [x] `JDCT_ISLOW` — Accurate integer DCT (only method)
- [ ] `JDCT_IFAST` — Fast integer DCT
- [ ] `JDCT_FLOAT` — Floating-point DCT

### Color Space Control
- [x] Auto YCbCr from RGB/RGBA/BGR/BGRA input
- [x] CMYK direct (no conversion)
- [ ] `jpeg_set_colorspace()` — Explicit colorspace override
- [ ] `jpeg_default_colorspace()` — Reset to default
- [ ] `in_color_space` / `jpeg_color_space` independent control
- [ ] Grayscale-from-color encode option

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
- [ ] `jpeg_write_marker()` — Write arbitrary marker data
- [ ] `jpeg_write_m_header()` / `jpeg_write_m_byte()` — Streaming marker write
- [ ] `jpeg_write_icc_profile()` — Standalone ICC write (without full compress)
- [ ] `jpeg_write_tables()` — Write tables-only JPEG
- [ ] COM (comment) marker write

### Scanline-Level Encode API
- [ ] `jpeg_start_compress()` — Begin compression
- [ ] `jpeg_write_scanlines()` — Write scanline rows
- [ ] `jpeg_finish_compress()` — Finalize compression
- [ ] `jpeg_write_raw_data()` — Write raw downsampled data
- [ ] `jpeg12_write_scanlines()` — 12-bit scanlines
- [ ] `jpeg16_write_scanlines()` — 16-bit scanlines
- [ ] `jpeg_calc_jpeg_dimensions()` — Compute output dimensions
- [ ] `next_scanline` tracking

---

## 7. Decompression Parameters (TJPARAM / jpeg_decompress_struct fields)

### Output Format
- [x] Output pixel format selection (`decompress_to`)
- [x] Scaled IDCT — 1/1, 1/2, 1/4, 1/8 (`set_scale`)
- [x] Crop decode (`decompress_cropped`, `set_crop_region`)
- [ ] `TJPARAM_BOTTOMUP` — Bottom-up row order
- [ ] `out_color_space` — Explicit output colorspace
- [ ] YCbCr/YUV raw output (skip color conversion)
- [ ] `raw_data_out` — Raw downsampled component output

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
- [ ] `TJPARAM_STOPONWARNING` — Treat warnings as fatal
- [ ] `TJPARAM_SCANLIMIT` — Max progressive scans before error
- [ ] Custom `jpeg_error_mgr` callbacks

### Limits
- [ ] `TJPARAM_MAXMEMORY` — Memory limit
- [ ] `TJPARAM_MAXPIXELS` — Image size limit

### Marker Handling
- [x] ICC profile reassembly from APP2 chunks
- [x] EXIF extraction + orientation (APP1)
- [x] Adobe APP14 detection (CMYK/YCCK)
- [x] Restart marker (DRI/RST) handling
- [ ] `TJPARAM_SAVEMARKERS` — Configurable marker saving (0-4 levels)
- [ ] `jpeg_save_markers()` — Per-marker-type save control
- [ ] `jpeg_set_marker_processor()` — Custom marker parser callback
- [ ] COM (comment) marker read/expose
- [ ] Arbitrary marker access via `marker_list` linked list
- [ ] JFIF version / density read (`saw_JFIF_marker`, `X_density`, `Y_density`)

### Multi-Scan / Progressive Output
- [ ] `jpeg_has_multiple_scans()` — Query progressive
- [ ] `buffered_image` mode — Enable scan-by-scan output
- [ ] `jpeg_start_output()` / `jpeg_finish_output()` — Per-scan output control
- [ ] `jpeg_consume_input()` — Incremental input processing
- [ ] `jpeg_input_complete()` — Check if all input consumed

### Scanline-Level Decode API
- [ ] `jpeg_read_header()` — Parse headers
- [ ] `jpeg_start_decompress()` — Begin decompression
- [ ] `jpeg_read_scanlines()` — Read scanline rows
- [ ] `jpeg_skip_scanlines()` — Skip rows during decode
- [ ] `jpeg_crop_scanline()` — Scanline-level horizontal crop
- [ ] `jpeg_finish_decompress()` — Finalize decompression
- [ ] `jpeg_read_raw_data()` — Read raw downsampled data
- [ ] `jpeg12_read_scanlines()` / `jpeg12_skip_scanlines()` / `jpeg12_crop_scanline()`
- [ ] `jpeg16_read_scanlines()`
- [ ] `jpeg_calc_output_dimensions()` / `jpeg_core_output_dimensions()`
- [ ] `output_scanline` tracking

### Color Quantization (8-bit indexed output)
- [ ] `quantize_colors` — Enable color quantization
- [ ] `desired_number_of_colors` / `actual_number_of_colors`
- [ ] `dither_mode` — JDITHER_NONE / JDITHER_ORDERED / JDITHER_FS
- [ ] `two_pass_quantize` — Two-pass color selection
- [ ] `colormap` — External colormap input
- [ ] `enable_1pass_quant` / `enable_2pass_quant` / `enable_external_quant`
- [ ] `jpeg_new_colormap()` — Update colormap

---

## 8. Metadata

- [x] APP0 JFIF — Read / write
- [x] APP1 EXIF — Read / write (orientation parsing)
- [x] APP2 ICC profile — Read (multi-chunk reassembly) / write (multi-chunk)
- [x] APP14 Adobe — Read / write (CMYK/YCCK signaling)
- [ ] COM (comment) — Read / write
- [ ] Arbitrary APP markers — Read (`jpeg_save_markers` + `marker_list`)
- [ ] Arbitrary markers — Write (`jpeg_write_marker`)
- [ ] DPI/density — Read (from JFIF) / write (configurable)
- [ ] JFIF thumbnail extraction
- [ ] Marker preservation across transform/re-encode

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
- [ ] TJXOPT_PERFECT (1) — Fail if transform is not perfect (partial iMCU)
- [ ] TJXOPT_TRIM (2) — Discard partial iMCU edges
- [ ] TJXOPT_CROP (4) — Enable lossless cropping region
- [ ] TJXOPT_GRAY (8) — Convert to grayscale during transform
- [ ] TJXOPT_NOOUTPUT (16) — Dry run (no output image)
- [ ] TJXOPT_PROGRESSIVE (32) — Output as progressive JPEG
- [ ] TJXOPT_COPYNONE (64) — Discard all non-essential markers
- [ ] TJXOPT_ARITHMETIC (128) — Output with arithmetic coding
- [ ] TJXOPT_OPTIMIZE (256) — Output with optimized Huffman

### Coefficient Access
- [x] `read_coefficients()` — Extract quantized DCT blocks
- [x] `write_coefficients()` — Encode from coefficient blocks
- [x] `transform_jpeg()` — Apply spatial transform
- [ ] `jpeg_copy_critical_parameters()` — Copy tables between compress/decompress
- [ ] `tjtransform.customFilter` — User callback for coefficient inspection/modification
- [ ] `tj3TransformBufSize()` — Output buffer size estimation

---

## 10. YUV / Planar API

### RGB → YUV (color conversion only, no JPEG)
- [ ] `tj3EncodeYUV8()` — RGB → packed YUV buffer
- [ ] `tj3EncodeYUVPlanes8()` — RGB → separate Y/Cb/Cr plane buffers

### YUV → JPEG (compress from YUV)
- [ ] `tj3CompressFromYUV8()` — Packed YUV → JPEG
- [ ] `tj3CompressFromYUVPlanes8()` — Planar YUV → JPEG

### JPEG → YUV (decompress to YUV)
- [ ] `tj3DecompressToYUV8()` — JPEG → packed YUV buffer
- [ ] `tj3DecompressToYUVPlanes8()` — JPEG → separate Y/Cb/Cr plane buffers

### YUV → RGB (color conversion only, no JPEG)
- [ ] `tj3DecodeYUV8()` — Packed YUV → RGB
- [ ] `tj3DecodeYUVPlanes8()` — Planar YUV → RGB

### Buffer Size Helpers
- [ ] `tj3YUVBufSize()` — Total packed YUV buffer size
- [ ] `tj3YUVPlaneSize()` — Single plane buffer size
- [ ] `tj3YUVPlaneWidth()` — Plane width in samples
- [ ] `tj3YUVPlaneHeight()` — Plane height in rows

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
- [ ] Forward DCT (FDCT) for encoder
- [ ] Chroma downsample for encoder
- [ ] Quantization for encoder
- [ ] Scaled IDCT (4x4, 2x2, 1x1) NEON variants
- [ ] RGB → YCbCr (encode-side color conversion)

### x86_64
- [ ] SSE2 IDCT
- [ ] SSE2 color conversion (YCbCr→RGB)
- [ ] SSE2 upsample (H2V1, H2V2)
- [ ] AVX2 IDCT
- [ ] AVX2 color conversion
- [ ] AVX2 upsample

### General
- [x] Scalar fallback for all operations
- [x] Runtime SIMD feature detection (`simd::detect()`)

---

## 12. Memory, I/O, Buffer Management

### Source / Destination
- [x] Memory-to-memory compress (`Vec<u8>` output)
- [x] Memory-to-memory decompress (byte slice → `Image`)
- [ ] `jpeg_stdio_dest()` / `jpeg_stdio_src()` — File I/O
- [ ] `jpeg_mem_dest()` / `jpeg_mem_src()` — C memory I/O (Rust equivalent: already native)
- [ ] Custom `jpeg_destination_mgr` — User-defined output stream
- [ ] Custom `jpeg_source_mgr` — User-defined input stream
- [ ] `TJPARAM_NOREALLOC` — Pre-allocated output buffer

### Buffer Size Calculation
- [ ] `tj3JPEGBufSize()` — Worst-case JPEG output size
- [ ] `tj3YUVBufSize()` — YUV buffer size
- [ ] `tj3TransformBufSize()` — Transform output buffer size

### Image File I/O (BMP/PPM)
- [ ] `tj3LoadImage8()` / `tj3LoadImage12()` / `tj3LoadImage16()`
- [ ] `tj3SaveImage8()` / `tj3SaveImage12()` / `tj3SaveImage16()`

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
- [ ] Custom `jpeg_error_mgr` struct
- [ ] `error_exit()` callback — Fatal error handler
- [ ] `emit_message()` callback — Warning/trace handler
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

- [ ] `jpeg_progress_mgr` struct
- [ ] `progress_monitor()` callback
- [ ] `pass_counter` / `pass_limit`
- [ ] `completed_passes` / `total_passes`

---

## 15. TJ3 Handle / Parameter API

- [ ] `tj3Init()` / `tj3Destroy()` — Handle lifecycle
- [ ] `tj3Set()` / `tj3Get()` — Generic parameter get/set
- [ ] All 26 TJPARAM values as runtime parameters
- [ ] `tj3SetICCProfile()` / `tj3GetICCProfile()` — ICC via handle
- [ ] `tj3SetScalingFactor()` / `tj3SetCroppingRegion()` — Decode options via handle
- [ ] `tj3GetScalingFactors()` — Query available scaling factors

---

## Summary

| Category | Done | Total | % |
|----------|------|-------|---|
| Frame types (encode) | 4 | 6 | 67% |
| Frame types (decode) | 5 | 6 | 83% |
| Sample precision | 1 | 3 | 33% |
| Pixel formats | 6 | 13 | 46% |
| Chroma subsampling | 6 | 8 | 75% |
| Color spaces | 5 | 6 | 83% |
| Compress params | ~15 | ~65 | ~23% |
| Decompress params | ~12 | ~55 | ~22% |
| Metadata | 4 | 10 | 40% |
| Transform ops | 8 | 8 | 100% |
| Transform options | 0 | 9 | 0% |
| Transform misc | 3 | 6 | 50% |
| YUV/Planar API | 0 | 12 | 0% |
| SIMD (aarch64) | 7 | 12 | 58% |
| SIMD (x86_64) | 0 | 6 | 0% |
| Memory & I/O | 2 | ~20 | ~10% |
| Error handling | 2 | ~14 | ~14% |
| Progress | 0 | 4 | 0% |
| TJ3 Handle API | 0 | ~6 | 0% |

---

## Priority Roadmap

> Strategy: feature completeness first, then SIMD/performance.

### Phase 4 — Core Feature Gaps
| # | Feature | Scope |
|---|---------|-------|
| 1 | Restart interval encode (DRI) | `restart_interval` / `restart_in_rows` in all compress paths |
| 2 | COM marker read/write | Decode: expose in Image; Encode: `jpeg_write_marker` equivalent |
| 3 | Lossless encode: color + predictor + pt | SOF3: 3-component, predictors 1-7, point transform 0-15 |
| 4 | SOF10 arithmetic progressive encode | `compress_arithmetic_progressive()` |
| 5 | SOF11 lossless arithmetic encode/decode | Arithmetic entropy for lossless |
| 6 | Grayscale-from-color encode | Drop chroma at encode time |
| 7 | Configurable DPI/density | Read and write X/Y density in JFIF |
| 8 | Arbitrary marker write | `jpeg_write_marker()` equivalent |

### Phase 5 — Extended Formats
| # | Feature | Scope |
|---|---------|-------|
| 9 | 12-bit precision | `i16` sample paths for encode + decode |
| 10 | 16-bit precision (lossless) | `u16` sample paths |
| 11 | Custom quantization tables | User-supplied `JQUANT_TBL` equivalent |
| 12 | Custom Huffman tables | User-supplied `JHUFF_TBL` equivalent |
| 13 | Custom progressive scan scripts | User-defined `jpeg_scan_info` |
| 14 | Additional pixel formats | RGBX, BGRX, XRGB, XBGR, ARGB, ABGR, RGB565 |
| 15 | Fast upsample / fast DCT toggles | `TJPARAM_FASTUPSAMPLE`, `TJPARAM_FASTDCT`, `JDCT_IFAST` |
| 16 | S441 subsampling | 4:4:1 (vertical 4x) |

### Phase 6 — Transform & Advanced
| # | Feature | Scope |
|---|---------|-------|
| 17 | All 9 TJXOPT transform flags | PERFECT, TRIM, CROP, GRAY, NOOUTPUT, PROGRESSIVE, COPYNONE, ARITHMETIC, OPTIMIZE |
| 18 | Coefficient filter callback | `tjtransform.customFilter` |
| 19 | Marker preservation | `TJPARAM_SAVEMARKERS` + `jpeg_save_markers` + copy through transform |
| 20 | Scanline-level encode API | `start_compress` / `write_scanlines` / `finish_compress` |
| 21 | Scanline-level decode API | `start_decompress` / `read_scanlines` / `skip_scanlines` / `crop_scanline` / `finish_decompress` |
| 22 | Progressive output (buffered image) | `buffered_image`, `start_output` / `finish_output` / `consume_input` |
| 23 | Per-component quality | `q_scale_factor[4]` |
| 24 | Raw data encode/decode | `jpeg_write_raw_data` / `jpeg_read_raw_data` |

### Phase 7 — YUV & I/O
| # | Feature | Scope |
|---|---------|-------|
| 25 | YUV planar encode/decode | All 8 `tj3*YUV*` functions |
| 26 | Buffer size calculation | `tj3JPEGBufSize`, `tj3YUVBufSize`, `tj3TransformBufSize` |
| 27 | Custom source/dest managers | Streaming I/O abstraction |
| 28 | File I/O helpers | `tj3LoadImage*` / `tj3SaveImage*` (BMP/PPM) |

### Phase 8 — SIMD & Performance
| # | Feature | Scope |
|---|---------|-------|
| 29 | x86_64 SSE2 | IDCT, color conversion, upsample |
| 30 | x86_64 AVX2 | IDCT, color conversion, upsample |
| 31 | aarch64 NEON extensions | FDCT, scaled IDCT, encode-side color/downsample |

### Phase 9 — Full API Parity
| # | Feature | Scope |
|---|---------|-------|
| 32 | TJ3 handle/parameter API | `tj3Init`/`tj3Set`/`tj3Get`/`tj3Destroy` pattern |
| 33 | Custom error manager | `jpeg_error_mgr` callbacks |
| 34 | Progress monitoring | `jpeg_progress_mgr` callback |
| 35 | Color quantization | Palette reduction, dithering (NONE/ORDERED/FS) |
| 36 | Custom memory manager | Pool-based allocation, virtual arrays |
