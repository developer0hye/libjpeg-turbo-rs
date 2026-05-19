# Feature Parity: libjpeg-turbo-rs vs libjpeg-turbo (C)

> Track implementation progress. Update checkboxes when features are completed.
> Source of truth: `turbojpeg.h` (TJ3 API), `jpeglib.h` (libjpeg API), `jmorecfg.h`
> Reliability rule: `[x]` means the documented public Rust surface is wired end-to-end for the claimed capability.
> Merely storing a `TjHandle` parameter, exposing an internal helper, or supporting a nearby API surface does not count as full parity.

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
- [x] 12-bit (`J12SAMPLE` / `i16`) — `compress_12bit`, `decompress_12bit`, `jpeg12_write_scanlines`, `jpeg12_read_scanlines` (not wired through `TjHandle::compress()` / `decompress()`)
- [x] 16-bit (`J16SAMPLE` / `u16`, lossless only) — `compress_16bit`, `decompress_16bit`, `jpeg16_write_scanlines`, `jpeg16_read_scanlines` (not wired through `TjHandle::compress()` / `decompress()`)

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
- [x] TJSAMP_UNKNOWN (unusual/custom subsampling detection) (`Subsampling::Unknown`)

---

## 5. Color Spaces (TJCS / J_COLOR_SPACE)

- [x] JCS_GRAYSCALE
- [x] JCS_YCbCr
- [x] JCS_RGB
- [x] JCS_CMYK
- [x] JCS_YCCK
- [x] JCS_UNKNOWN (pass-through, no conversion) (`ColorSpace::Unknown`)

---

## 6. Compression Parameters (TJPARAM / jpeg_compress_struct fields)

### Quality & Quantization
- [x] `TJPARAM_QUALITY` — Quality factor 1-100 (`jpeg_set_quality`)
- [x] `q_scale_factor[NUM_QUANT_TBLS]` — Per-component quality (`Encoder::quality_factor()`)
- [x] `jpeg_add_quant_table()` — Custom quantization table (`Encoder::quant_table()`)
- [x] `jpeg_set_linear_quality()` — Linear quality scaling (`Encoder::linear_quality()`)
- [x] `jpeg_default_qtables()` — Reset to default tables (`Encoder::reset_quant_tables()`)
- [x] `jpeg_quality_scaling()` — Quality to scale factor conversion (`quality_scaling()`)
- [x] `force_baseline` parameter — Constrain quant values to 1-255 (`Encoder::force_baseline()`)

### Huffman Tables
- [x] Standard DC/AC luminance + chrominance tables
- [x] `TJPARAM_OPTIMIZE` — 2-pass optimized Huffman (`compress_optimized`)
- [x] Custom `dc_huff_tbl_ptrs[4]` — User-supplied DC Huffman tables (`Encoder::huffman_dc_table()`)
- [x] Custom `ac_huff_tbl_ptrs[4]` — User-supplied AC Huffman tables (`Encoder::huffman_ac_table()`)
- [x] `jpeg_alloc_huff_table()` — N/A for Rust (Huffman tables are value types, no allocation needed)
- [x] `jpeg_suppress_tables()` — N/A for Rust (ownership handles table reuse; no persistent "sent" state needed)

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
- [x] `write_JFIF_header` — JFIF marker (written by default with 1x1 unknown-density fields unless coefficient metadata is rewritten)
- [x] `TJPARAM_XDENSITY` — wired through `TjHandle` compress/decompress + `Encoder::density()`
- [x] `TJPARAM_YDENSITY` — wired through `TjHandle` compress/decompress + `Encoder::density()`
- [x] `TJPARAM_DENSITYUNITS` — wired through `TjHandle` compress/decompress + `Encoder::density()`
- [x] `JFIF_major_version` / `JFIF_minor_version` configurable (`Encoder::jfif_version()`)
- [x] JFIF density read (`Image.density`)
- [x] Low-level density rewrite via coefficient API (`JpegCoefficients.{density_unit,x_density,y_density}`)

### Adobe Marker
- [x] `write_Adobe_marker` — Adobe APP14 (for CMYK)
- [x] `write_Adobe_marker` toggle — Enable/disable (`Encoder::write_adobe_marker()`)

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
- [x] `jpeg_set_colorspace()` — Explicit colorspace override (`Encoder::colorspace()`)
- [x] `jpeg_default_colorspace()` — Reset to auto-detection (`Encoder::reset_colorspace()`)
- [x] `in_color_space` / `jpeg_color_space` — Input format inferred from `PixelFormat`; JPEG colorspace via `Encoder::colorspace()` / `TjHandle` `TJPARAM_COLORSPACE` with `TJCS_DEFAULT=-1`
- [x] Grayscale-from-color encode option (`Encoder::grayscale_from_color()`)

### Input Options
- [x] `TJPARAM_BOTTOMUP` — Bottom-up row order (`Encoder::bottom_up()`)
- [x] `raw_data_in` — Encode from raw downsampled component data (`compress_raw()`)
- [x] `smoothing_factor` — Input smoothing (0-100) (`Encoder::smoothing_factor()`)
- [x] `do_fancy_downsampling` — Fancy vs simple chroma downsample (`Encoder::fancy_downsampling()`)
- [x] `CCIR601_sampling` — N/A (field exists in C struct but never used in libjpeg-turbo encode path)
- [x] `input_gamma` — N/A (gamma correction is user-space preprocessing, not encoder responsibility; C field initialized to 1.0 and never applied)

### Marker Writing
- [x] JFIF APP0 (automatic)
- [x] EXIF APP1 (`compress_with_metadata`)
- [x] ICC APP2 (`compress_with_metadata`, multi-chunk)
- [x] Adobe APP14 (CMYK encode)
- [x] `jpeg_write_marker()` — Write arbitrary marker data (`marker_writer::write_marker()`)
- [x] `jpeg_write_m_header()` / `jpeg_write_m_byte()` — Streaming marker write (`MarkerStreamWriter`)
- [x] `jpeg_write_icc_profile()` — ICC embedded via `Encoder::icc_profile()` / `compress_with_metadata()` / `TjHandle::set_icc_profile()`
- [x] `jpeg_write_tables()` — Write tables-only JPEG (`marker_writer::write_tables_only()`)
- [x] COM (comment) marker write (`Encoder::comment()`, `marker_writer::write_com()`)

### Scanline-Level Encode API
- [x] `jpeg_start_compress()` — Begin compression (`ScanlineEncoder::new()`)
- [x] `jpeg_write_scanlines()` — Write scanline rows (`ScanlineEncoder::write_scanlines()`)
- [x] `jpeg_finish_compress()` — Finalize compression (`ScanlineEncoder::finish()`)
- [x] `jpeg_write_raw_data()` — Write raw downsampled data (`compress_raw()`)
- [x] `jpeg12_write_scanlines()` — 12-bit scanlines (`write_scanlines_12()`)
- [x] `jpeg16_write_scanlines()` — 16-bit scanlines (`write_scanlines_16()`)
- [x] `jpeg_calc_jpeg_dimensions()` — Compute compression-side JPEG dimensions; no compression scaling (`calc_jpeg_dimensions()`, P4-1 2026-05-10)
- [x] `next_scanline` tracking (`ScanlineEncoder::next_scanline()`)

---

## 7. Decompression Parameters (TJPARAM / jpeg_decompress_struct fields)

### Output Format
- [x] Output pixel format selection (`decompress_to`)
- [x] Scaled IDCT — all 16 factors: 1/8 through 2/1 (`set_scale`)
- [x] Crop decode (`decompress_cropped`, `set_crop_region`)
- [x] `TJPARAM_BOTTOMUP` — Bottom-up row order (`ScanlineDecoder::set_bottom_up()`)
- [x] `out_color_space` — Explicit output colorspace (`Decoder::set_output_colorspace()`)
- [x] YCbCr/YUV raw output (skip color conversion) (`decompress_raw()`)
- [x] `raw_data_out` — Raw downsampled component output (`decompress_raw()`)

### Upsampling / DCT
- [x] Fancy upsampling (default, always on)
- [x] `TJPARAM_FASTUPSAMPLE` — Nearest-neighbor upsampling toggle (`Decoder::set_fast_upsample()`)
- [x] `do_fancy_upsampling` toggle (`Decoder::set_fast_upsample()`)
- [x] `TJPARAM_FASTDCT` — Fast IDCT vs accurate toggle (`Decoder::set_fast_dct()`)
- [x] `do_block_smoothing` toggle (`Decoder::set_block_smoothing()`)
- [x] `dct_method` selection (ISLOW/IFAST/FLOAT) (`Decoder::set_dct_method()`)
- [x] RGB565 ordered dithering (`Decoder::set_dither_565()`)
- [x] Merged upsampling (combined upsample + color convert for 422m/420m) (`Decoder::set_merged_upsample()`). RGB and RGB565 output both wired through the SIMD merged kernels (NEON / AVX2 / WASM SIMD128 / scalar). Cross-validated against `djpeg -nosmooth` for S420/S422 in `tests/cross_check_p3_6_nonstandard_rgb565.rs`. Dedicated `_565` SIMD kernels (upstream `jdmrgext-*-565`) are deferred as a Phase 4 perf task — current path packs after the merged conversion.
- [x] 4:1:0 (H=4,V=2) subsampling decode — arbitrary factor upsampling

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
- [x] `TJPARAM_SAVEMARKERS` — Configurable marker saving via `Decoder::save_markers()` / `MarkerSaveConfig` (not yet wired through `TjHandle`)
- [x] `jpeg_save_markers()` — Per-marker-type save control with per-code `length_limit` truncation (`Decoder::save_markers()` / `MarkerSaveConfig::WithLimits`; C ABI shim truncates `cinfo->marker_list` entries to the requested limit)
- [x] `jpeg_set_marker_processor()` — Custom marker parser callback (`Decoder::set_marker_processor()`)
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
- [x] `jpeg_crop_scanline()` — Scanline-level horizontal crop (`ScanlineDecoder::set_crop_x()`)
- [x] `jpeg_finish_decompress()` — Finalize decompression (`ScanlineDecoder::finish()`)
- [x] `jpeg_read_raw_data()` — Read raw downsampled data (`decompress_raw()`)
- [x] `jpeg12_read_scanlines()` / `jpeg12_skip_scanlines()` / `jpeg12_crop_scanline()` (`read_scanlines_12()`)
- [x] `jpeg16_read_scanlines()` (`read_scanlines_16()`)
- [x] `jpeg_calc_output_dimensions()` / `jpeg_core_output_dimensions()` (`calc_output_dimensions()`, `calc_jpeg_dimensions()`)
- [x] `output_scanline` tracking (`ScanlineDecoder::output_scanline()`)

### Color Quantization (8-bit indexed output)
- [x] `quantize_colors` — Enable color quantization (`quantize::quantize()`)
- [x] `desired_number_of_colors` / `actual_number_of_colors` (`QuantizeOptions::num_colors`, `QuantizedImage::palette.len()`)
- [x] `dither_mode` — JDITHER_NONE / JDITHER_ORDERED / JDITHER_FS (`DitherMode` enum)
- [x] `two_pass_quantize` — Two-pass color selection (`QuantizeOptions::two_pass`, median-cut algorithm)
- [x] `colormap` — External colormap input (`QuantizeOptions::colormap`)
- [x] `enable_1pass_quant` / `enable_2pass_quant` / `enable_external_quant` (`QuantizeOptions::two_pass` + `colormap`)
- [x] `jpeg_new_colormap()` — Update colormap (`requantize()`)

---

## 8. Metadata

- [x] APP0 JFIF — Read / write
- [x] APP1 EXIF — Read / write (orientation parsing)
- [x] APP2 ICC profile — Read (multi-chunk reassembly) / write (multi-chunk)
- [x] APP14 Adobe — Read / write (CMYK/YCCK signaling)
- [x] COM (comment) — Read (`Image.comment`) / Write (`Encoder::comment()`)
- [x] Arbitrary APP markers — Read (`Decoder::save_markers()` + `Image.markers()`)
- [x] Arbitrary markers — Write (`marker_writer::write_marker()`, `Encoder::saved_marker()`)
- [x] DPI/density — Read (`Image.density`); low-level rewrite via `JpegCoefficients`, no high-level `Encoder::density()` / `TjHandle` write path
- [x] JFIF thumbnail extraction (`extract_jfif_thumbnail()`)
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
- [x] TJXOPT_COPYNONE (64) — Discard all non-essential markers (`MarkerCopyMode::None`)
- [x] `-copy icc` — Copy only ICC profile markers (`MarkerCopyMode::IccOnly`)
- [x] TJXOPT_ARITHMETIC (128) — Output with arithmetic coding (`TransformOptions.arithmetic`)
- [x] TJXOPT_OPTIMIZE (256) — Output with optimized Huffman (`TransformOptions.optimize`)

### Coefficient Access
- [x] `read_coefficients()` — Extract quantized DCT blocks
- [x] `write_coefficients()` — Encode from coefficient blocks
- [x] `transform_jpeg()` — Apply spatial transform
- [x] `jpeg_copy_critical_parameters()` — Copy tables between compress/decompress (`copy_critical_parameters()`)
- [x] `tjtransform.customFilter` — User callback for coefficient inspection/modification
- [x] `tj3TransformBufSize()` — Output buffer size estimation (`transform_buf_size()`)

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
- [x] Merged H2V1 upsample + color convert (422m)
- [x] Merged H2V2 upsample + color convert (420m)
- [x] Forward DCT (FDCT) for encoder
- [x] Chroma downsample for encoder
- [x] Quantization for encoder
- [x] Scaled IDCT (4x4, 2x2, 1x1) NEON variants
- [x] RGB → YCbCr (encode-side color conversion)

### x86_64
- [x] SSE2 IDCT
- [x] SSE2 color conversion (YCbCr→RGB)
- [x] SSE2 upsample (H2V1, H2V2)
- [x] AVX2 IDCT (full 256-bit ymm, vpmaddwd, DC-only fast path, strided output)
- [x] AVX2 color conversion (i16 mulhi + SSSE3 pshufb interleave)
- [x] AVX2 upsample
- [x] AVX2 vertical blend for H2V2
- [x] AVX2 merged H2V1 upsample + color convert
- [x] AVX2 merged H2V2 upsample + color convert
- [x] Row-streaming H2V2 upsample+color pipeline (fused, no full-plane alloc)
- [x] AVX2 color conversion for RGBA/BGR/BGRA formats (`avx2_ycbcr_to_rgba_row`, `avx2_ycbcr_to_bgr_row`, `avx2_ycbcr_to_bgra_row`)
- [x] SSE2 IDCT DC-only fast path + strided output
- [x] x86_64 encoder SIMD (AVX2 FDCT, RGB→YCbCr, quantization + zigzag)
- [x] AVX2 encode color conversion for RGBA/BGR/BGRA formats
- [x] Fused MCU-row encode pipeline for all pixel formats (RGB/RGBA/BGR/BGRA)

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
- [x] `TJPARAM_NOREALLOC` — Pre-allocated output buffer (`compress_into()`)

### Buffer Size Calculation
- [x] `tj3JPEGBufSize()` — Worst-case JPEG output size (`jpeg_buf_size()`)
- [x] `tj3YUVBufSize()` — YUV buffer size (`yuv_buf_size()`)
- [x] `tj3TransformBufSize()` — Transform output buffer size (`transform_buf_size()`)

### Image File I/O (BMP/PPM/PGM subset)
- [x] `tj3LoadImage8()` / `tj3SaveImage8()` — BMP/PPM/PGM 8-bit (`load_image` / `save_bmp` / `save_ppm`). PNG is conditional in C (`PNG_SUPPORTED` build flag, requires libspng) — mirrored as `--features png` (default off); supports 8-bit RGB/RGBA/Grayscale PNG via `png` crate.
- [x] `tj3LoadImage12()` / `tj3LoadImage16()` — 12/16-bit PPM load (`load_ppm_16bit()` / `load_ppm_16bit_from_bytes()`). C only supports PPM for 12/16-bit.
- [x] `tj3SaveImage12()` / `tj3SaveImage16()` — 12/16-bit PPM save (`save_ppm_16bit()`)

### Memory Management
- [x] Custom `jpeg_memory_mgr` — N/A for Rust (Rust ownership + allocator API replaces C pool-based allocator)
- [x] `alloc_small` / `alloc_large` / `alloc_sarray` / `alloc_barray` — N/A (Rust `Vec`/`Box` replaces C pool allocator)
- [x] `request_virt_sarray` / `request_virt_barray` / virtual array API — N/A (Rust uses direct `Vec<Vec<>>` coefficient storage)
- [x] `free_pool` / `self_destruct` — N/A (Rust Drop trait handles cleanup)
- [x] `max_memory_to_use` / `max_alloc_chunk` — `Decoder::set_max_memory()` / `TjHandle` `TJPARAM_MAXMEMORY`
- [x] `tj3Alloc()` / `tj3Free()` — N/A (Rust ownership; `Vec<u8>` return replaces C caller-managed buffers)

---

## 13. Error Handling

- [x] `Result<T, JpegError>` for all public operations
- [x] `DecodeWarning` list (HuffmanError, TruncatedData) in lenient mode
- [x] Custom error handler — `ErrorHandler` trait
- [x] `error_exit()` callback — `ErrorHandler::error_exit()`
- [x] `emit_message()` callback — `ErrorHandler::emit_warning()` + `ErrorHandler::trace()`
- [x] `output_message()` / `format_message()` — N/A (Rust `Display` trait on `JpegError` replaces C message callbacks)
- [x] `reset_error_mgr()` — N/A (Rust `Result` is stateless; no accumulated error state to reset)
- [x] `trace_level` control — `ErrorHandler::trace()` callback with level parameter
- [x] `num_warnings` counter — `Image.warnings` vec (count via `.len()`)
- [x] `msg_code` / `msg_parm` / `jpeg_message_table` — N/A (Rust uses typed `JpegError` / `DecodeWarning` enums instead of C integer codes + format strings)
- [x] `tj3GetErrorStr()` / `tj3GetErrorCode()` — Rust `Result<T, JpegError>` with `Display` impl replaces C per-handle error getters
- [x] `jpeg_resync_to_restart()` — Internal restart resync handled automatically by decoder; no public hook needed (matches C default behavior)

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
- [x] All 26 TJPARAM values wired end-to-end (`ColorSpace` with `TJCS_DEFAULT=-1`, `Subsampling`, density, ICC populated by `decompress()`; density and `ColorSpace` wired into `compress()`; `SaveMarkers` 0-4 behaviorally wired in decode; `Precision` read-only; `NoRealloc` N/A for Rust `Vec<u8>`)
- [x] `tj3Compress12()` / `tj3Compress16()` / `tj3Decompress12()` / `tj3Decompress16()` — Multi-precision via `TjHandle` (`compress_12bit()` / `compress_16bit()` / `decompress_12bit()` / `decompress_16bit()`)
- [x] `tj3SetICCProfile()` / `tj3GetICCProfile()` — encode-side ICC via handle + decompress populates handle ICC (`TjHandle::set_icc_profile()` / `TjHandle::icc_profile()`)
- [x] `tj3SetScalingFactor()` / `tj3SetCroppingRegion()` — Decode options via handle (`TjHandle::set_scaling_factor()` / `TjHandle::set_cropping_region()`)
- [x] `tj3GetScalingFactors()` — Query available scaling factors (`TjHandle::scaling_factors()`)

---

## Summary

- Percentage rollups were removed from this document. They looked precise, but they mixed core codec support, handle parity, and adjacent Rust-only surfaces in a way that overstated completion.
- Treat the checklist above as the source of truth. Open follow-ups (tracked in `docs/last_mile/phase4.md`): P4-13 (true streaming `jpeg_consume_input` / `JPEG_SUSPENDED` semantics — the existing entry point is a fully-buffered shim per `jpeglib.rs:4234-4238`); P4-14 (C-ABI `max_memory_to_use` enforcement — field is ABI-mirrored but unconsulted); P4-17 (real `JPEG_SUSPENDED` test — the P4-5 `source_mgr_suspends_every_byte` pattern actually exercises chunked refill); P4-18 (18 legacy TurboJPEG 1.x/2.x symbols remain allowlisted-missing per `symbol_inventory.rs:190-207`). P4-4 (panic guard on every C-ABI entry point) and P4-5 (pathological classic-lifecycle coverage) both closed 2026-05-17; P4-16 (per-`cinfo` thread-affinity contract) closed 2026-05-19 via documentation in `docs/ABI_COMPATIBILITY.md` ("Threading contract" section). PNG for `tj3LoadImage8()` / `tj3SaveImage8()` is gated by the optional `png` feature flag, mirroring upstream's `PNG_SUPPORTED` build-time flag.

## Recent Additions (batch reconciliation)

The following gaps were closed in the latest batch of merges:

- **Arithmetic table count widened 4 → 16** per spec F.2.4.3 (DAC parser/writer + `ArithDecoder`/`ArithEncoder`); spec-compliant streams with `tbl_no > 3` now decode/encode correctly.
- **Abbreviated datastream** (JPEG spec F.1.2.4): `jpeg_write_tables()`, `HeaderResult::TablesOnly`, `Decoder::new_with_tables()` inter-session table reuse, `Encoder::suppress_tables(bool)` body-only output.
- **12-bit raw planar I/O**: `compress_raw_12()` / `decompress_raw_12()` covering all 7 subsamplings × baseline/progressive/arithmetic.
- **12/16-bit PPM file I/O**: `load_ppm_12bit()` / `save_ppm_12bit()` / `load_ppm_16bit()` / `save_ppm_16bit()` matching C scope (PPM only).
- **TjHandle gaps**: `TJPARAM_NOREALLOC` (`compress_into()` returns `BufferTooSmall{need,got}`), `TJPARAM_SAVEMARKERS` behavioral wiring, `tj3GetICCProfile()` handle symmetry with `Image.icc_profile()`.
- **Session-reset APIs**: `Encoder::reset_colorspace()` (`jpeg_default_colorspace`), `Encoder::reset_quant_tables(force_baseline)` (`jpeg_default_qtables`).
- **Restart resync hook**: `RestartResyncStrategy` trait + `Decoder::set_resync_strategy()` with `ResyncAction {Continue, Skip, Abort}` — replaces the internal-only implementation of `jpeg_resync_to_restart`.
- **JPEG-in-RAW thumbnail**: `extract_embedded_jpeg()` walks TIFF IFDs (LE/BE, bounds-checked) to extract embedded JPEG thumbnails from ARW/CR2-style files.
- **C ABI shim crate** (`crates/libjpeg-turbo-rs-capi`, cdylib + staticlib): TJ3 API (TJInit/Destroy/Set/Get/Compress8/Decompress8/DecompressHeader/SetScalingFactor/SetCroppingRegion/Transform/YUV x8/ErrorStr/ErrorCode/Alloc/Free/Compress12/Decompress12/Compress16/Decompress16) + 21 legacy TJ1/TJ2 aliases wired in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` (lifecycle `tjInitCompress`/`tjInitDecompress`/`tjInitTransform`/`tjDestroy`; `tjCompress2`/`tjDecompress2`/`tjDecompressHeader3`; `tjTransform`/`tjEncodeYUV3`/`tjDecodeYUV`; buffer-size helpers + image I/O + error string) — 18 other legacy 1.x/2.x symbols (v1 / un-versioned variants like `tjAlloc`, `tjFree`, `tjCompress`, `tjGetScalingFactors`) remain allowlisted-missing per P4-18; + classic `jpeg_*` decode/encode/transform/raw-data lifecycle + SONAME/install_name (default `libjpeg.so.8` / `@rpath/libjpeg.8.dylib` since P4-3 2026-05-17; v6b opt-in via `CAPI_ACK_V6B_SONAME=1`; TurboJPEG `libturbojpeg.so.0`) + pkg-config `.pc` generation. Verified by the LAST_MILE stock-tool, Pillow/ImageMagick, tjunittest, ABI-offset, symbol-inventory, and downstream-consumer gates.

## Second-Batch Reconciliation (parallel follow-up workers)

- **A4 restored** (reverted revert after confirming the earlier corpus-test crash was classification-only, not an A4 regression): arithmetic table count widened 4 → 16 per spec F.2.4.3 is live on `main`.
- **FFI A1-11 jpeglib.h decode subset**: `jpeg_create_decompress`, `jpeg_std_error`, `jpeg_stdio_src`, `jpeg_mem_src`, `jpeg_read_header`, `jpeg_start_decompress`, `jpeg_read_scanlines`, `jpeg_finish_decompress`, `jpeg_destroy_decompress` all `#[no_mangle] extern "C"` in the capi crate. `#[repr(C)] JpegDecompressPublic` subset exposes the fields consumed by these entry points; fuller field layout + classic encode symmetrical (A1-12) + coefficient/marker APIs are tracked as follow-up in `COORDINATOR_NOTES.md`.
- **tj3 auxiliary surfaces** added by B9-5: `tj3GetScalingFactors`, `tj3YUVBufSize`, `tj3YUVPlaneSize`, `tj3YUVPlaneWidth`, `tj3YUVPlaneHeight`, `tj3JPEGBufSize`, `tj3InitVersion`, `tj3LoadImage{8,12,16}`/`tj3SaveImage{8,12,16}`, `TJBUFSIZE`, `TJBUFSIZEYUV`, `tjBufSizeYUV`, process-global no-handle error slot for `tj3GetErrorStr(NULL)`.
- **`tj3Init` enum fix**: `TJINIT_COMPRESS/DECOMPRESS/TRANSFORM` were previously treated as bit flags (1/2/4) instead of enum values (0/1/2); every C caller was getting NULL. Fixed, and exercised by the tjunittest link harness.
- **`tj3Compress8` NOREALLOC in-place fix**: when `TJPARAM_NOREALLOC == 1` and the caller pre-supplies `*jpegBuf`, we now write in place via `copy_nonoverlapping` instead of allocating a fresh libc buffer and swapping the pointer. The old behavior leaked every iteration of the tjunittest `doTest` loop and eventually corrupted the malloc heap (SIGSEGV at `_os_unfair_lock_unlock_slow` inside `mfm_alloc`). Now tjunittest runs to completion on our shim.
- **CI on libjpeg-turbo 3.1.3**: apt ships 2.1.x on Ubuntu 24.04, which lacks `-lossless`, `-precision`, and SOF3 decode. CI now downloads the official `libjpeg-turbo-official_3.1.3_${ARCH}.deb` from upstream GitHub releases and prepends `/opt/libjpeg-turbo/bin` to `$PATH`, so test expectations match the macOS homebrew reality.
- **B1 helper migration completed**: all 90 test files with local `fn djpeg_path()/cjpeg_path()/jpegtran_path()` helpers migrated to `helpers::require_c_tool!()` — CI now fails hard when C tools are missing, local dev still skips. 3 shards (A-E / F-P / Q-Z) × 20 batches.
- **B9-2 Pillow / B9-3 ImageMagick**: link harnesses are active (`examples/pillow_smoke/`, `examples/imagemagick_smoke/`, `tests/capi_pillow_compat.rs`, `crates/libjpeg-turbo-rs-capi/tests/capi_imagemagick_compat.rs`). They are no longer `#[ignore]`d; the tests only soft-skip for missing external tools or macOS loader-injection restrictions.
- **B9-4 stock djpeg–cjpeg–jpegtran — byte-exact parity achieved**: `examples/stock_djpeg_cjpeg/build.sh` builds stock `djpeg`, `cjpeg`, `jpegtran` against our capi and produces output **byte-identical to upstream libjpeg-turbo** on the full `references/libjpeg-turbo/testimages/*.jpg` corpus (8-bit `testorig`, arithmetic `testimgari`, integer-quant `testimgint`, 12-bit `monkey12`). Unblocked by the 33 classic `jpeg_*` symbols (A1-12) + precision-routing fixes in `jpeg_start_decompress` / `jpeg12_read_scanlines` / `jpeg16_read_scanlines` / `jpeg12_skip_scanlines` and the new `jpeg_calc_output_dimensions` export (commit `f70b41f`).
- **B9-5 tjunittest**: now passes 100% on the capi cdylib — `EXIT=0`, 0 ERROR, 0 FAILED, 1012 subtest passes. Closed by four 2026-04-25 fixes: grayscale-to-RGB repack in `tj3Decompress8`, TJSAMP_441/410/24 enum widening end-to-end, CMYK SOF sampling factors honoring TJPARAM_SUBSAMP, and TJPARAM_FASTUPSAMPLE wired into the 4-component decode upsample path.
- **SIMD upsample width=2 kernel guard (closed 2026-04-25)**: added `if in_width == 2` box-filter short-circuit to all four `fancy_upsample_h2v1` kernels (SSE2/AVX2/NEON/WASM) so the raw kernels now match scalar fancy_h2v1 byte-for-byte across every width. Revived `tests/simd_x86.rs::sse2_upsample_edge_cases`, upgraded `tests/simd_avx2.rs::avx2_upsample_width_2` to actually exercise the AVX2 kernel, and widened `UPSAMPLE_WIDTHS` in `tests/simd_parity.rs` to include widths 1 and 2.

## Classic `jpeg_*` API (Third Batch — C1 + C2)

Two parallel workers shipped **36 new `#[no_mangle] extern "C"` symbols** in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`:

- **Decode extensions (C1, 12 symbols)**: `jpeg_skip_scanlines`, `jpeg_crop_scanline`, `jpeg_save_markers`, `jpeg_set_marker_processor`, `jpeg_read_icc_profile`, `jpeg_read_coefficients`, `jpeg_copy_critical_parameters`, `jpeg_core_output_dimensions`, `jpeg12_read_scanlines`, `jpeg12_skip_scanlines`, `jpeg12_crop_scanline`, `jpeg16_read_scanlines`. High-precision state lives in a `thread_local!` side table keyed by the cinfo pointer.
- **Encode side + utilities (C2, 24 symbols)**: `jpeg_CreateCompress`/`jpeg_destroy_compress`, `jpeg_stdio_dest`/`jpeg_mem_dest`, `jpeg_set_defaults`/`jpeg_set_colorspace`/`jpeg_default_colorspace`, `jpeg_set_quality`, `jpeg_start_compress`/`jpeg_write_scanlines`/`jpeg_finish_compress`, `jpeg_quality_scaling`, `jpeg_add_quant_table`, `jpeg_default_qtables`, `jpeg_simple_progression`, `jpeg_enable_lossless`, `jpeg_suppress_tables`, `jpeg_write_marker`/`jpeg_write_m_header`/`jpeg_write_m_byte`, `jpeg_write_icc_profile`, `jpeg_write_tables`, `jpeg12_write_scanlines`/`jpeg16_write_scanlines`, `jpeg_write_coefficients`, `jpeg_resync_to_restart`, `jcopy_block_row`, `jdiv_round_up`.
- **Test count**: 38 (decode) + 15 (encode) new dlopen-and-exercise tests, all green on main.

## Stock-tool byte-exact milestone (B9-4)

- `examples/stock_djpeg_cjpeg/build.sh` builds stock **`djpeg`, `cjpeg`, `jpegtran`** against `libjpeg.62.dylib` / `libjpeg.so.62` produced by our capi crate, **zero undefined symbols** on all three. Per-precision wrappers (`wrppm-{8,12,16}.c`, `wrgif-{8,12}.c`, `rdppm-{8,12,16}.c`, `rdcolmap-{8,12}.c`) built with `-DBITS_IN_JSAMPLE=N` supply the `j12init_*` / `j16init_*` entry points.
- **Byte-exact corpus parity**: every JPEG in `references/libjpeg-turbo/testimages/` now round-trips `cmp -s` against upstream `djpeg` output — including the 12-bit `monkey12.jpg` (149×227 precision=12 PPM).
- Precision routing fixes that made 12-bit work:
  - `jpeg12_read_scanlines` / `jpeg16_read_scanlines` / `jpeg12_skip_scanlines` now advance `cinfo.output_scanline` (was spinning forever inside djpeg's `while (output_scanline < output_height)` loop).
  - `jpeg_start_decompress` fast-path for `data_precision > 8`: no longer drives the 8-bit Rust decoder (which silently succeeded on 12-bit input and clobbered `data_precision` 12→8, misrouting djpeg's precision dispatch).
  - `jpeg_calc_output_dimensions` exported and mirrors `jdmaster.c:267` (non-IDCT-scaling common case + JCS_EXT_* color-space pixelsize).

## Testing Infrastructure Additions

- **Fuzz corpus**: expanded from 22 → ~2,194 seeds across 7 targets (Cartesian product of subsamp × quality × content × entropy-mode); new `fuzz_encode_roundtrip` target; `scripts/fuzz_minimize.sh`; OSS-Fuzz stub at `oss-fuzz/`; nightly `.github/workflows/fuzz-smoke.yml`.
- **Cross-arch CI matrix** (`.github/workflows/cross-arch.yml`): `ubuntu-24.04-arm` (aarch64 NEON), x86_64 AVX2 default, x86_64 SSE2-only via `-C target-feature=-avx2,-sse4.2`, macOS arm64 retained, WASM SIMD128 smoke on every PR.
- **Per-SIMD bit-exact parity suite** (`tests/simd_parity.rs`): 20 kernel × backend combinations (NEON / AVX2 / SSE2 / WASM), 1000-iteration Mulberry32 PRNG, scalar↔SIMD bit-exact assertions.
- **Conformance suite**: `scripts/fetch_conformance.sh` + `tests/worker_b3_conformance_t83*.rs` iterating `references/libjpeg-turbo/testimages/*.jpg` for pixel-exact djpeg comparison + decoded-pixel hash regression in `tests/reference_hashes_conformance.json`.
- **Real-world corpus**: fetch scripts + seed fixtures for Kodak PhotoCD (PSNR round-trip), USC-SIPI Miscellaneous (djpeg byte-exact), EXIF Orientation 1..8, CMYK scanner, JPEG-in-RAW thumbnail.
- **DoS bounds**: cross-platform peak-RSS + wall-clock measure helper, Huffman bomb (max 16-bit codes), progressive 5000-scan bomb with `SCANLIMIT` mitigation, restart-interval=1 4096×4096 bomb. Bounds documented from measured reality per CLAUDE.md tolerance rule.
- **Concurrency stress**: rayon-substituted `std::thread` stress (1000 concurrent decodes, interleaved Encoder/Decoder handoff via mpsc, shared custom quant table), plus loom permutation skeleton gated on `#[cfg(loom)]`.
- **CI C-tool enforcement**: `require_c_tool!` macro panics in CI when `djpeg`/`cjpeg`/`jpegtran` missing; silent skip allowed only for local dev. All 90 test files with local `djpeg_path()/cjpeg_path()/jpegtran_path()` helpers have been migrated (A-E / F-P / Q-Z shards, 20 batches total).
- **libtiff end-to-end test wired** (`examples/libtiff_integration/` + `crates/libjpeg-turbo-rs-capi/tests/libtiff_integration.rs`): a C program opens a TIFF with `COMPRESSION_JPEG`, writes/reads strips via `TIFFWriteEncodedStrip` / `TIFFReadEncodedStrip`, with our cdylib staged as the JPEG provider via `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH`. This exercises the real downstream consumer of `jpeg_write_raw_data` / `jpeg_read_raw_data` (PR #240/#241). The test is active now; it soft-skips only when `cc`/libtiff are absent or on Windows. The former `JPEG_HEADER_TABLES_ONLY` gap is closed by the tables-only prefix splice path recorded in `docs/LAST_MILE.md`.

## Documentation Policy

- If a capability only exists through a different Rust API than the named C surface, describe that explicitly and do not count it as full parity for the original surface.
- If a row depends on a feature-gated or low-level path, say so in the row instead of promoting it to a blanket `[x]`.
