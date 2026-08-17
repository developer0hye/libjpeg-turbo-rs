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
- [x] Native 12-bit (`J12SAMPLE` / `i16`) — `compress_12bit`, `decompress_12bit`, `jpeg12_read_scanlines` (not wired through `TjHandle`)
- [x] Native 16-bit (`J16SAMPLE` / `u16`, lossless only) — `compress_16bit`, `decompress_16bit`, `jpeg16_read_scanlines` (not wired through `TjHandle`)
- [ ] Classic `jpeg12_write_scanlines` / `jpeg16_write_scanlines` finish pipeline — P4-94

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
- [x] TJSAMP_410 (4:1:0; H=4,V=2)
- [x] TJSAMP_24 (2:4; H=2,V=4)
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
- [ ] Classic `q_scale_factor[NUM_QUANT_TBLS]` wiring — native `Encoder::quality_factor()` is ready; classic setup remains P4-85
- [ ] Classic `jpeg_add_quant_table()` output wiring — native `Encoder::quant_table()` is ready; classic scanline output remains P4-85
- [ ] Classic `jpeg_set_linear_quality()` output wiring — native `Encoder::linear_quality()` is ready; classic scanline output remains P4-85
- [ ] Classic `jpeg_default_qtables()` semantics — native `Encoder::reset_quant_tables()` is ready; classic setup remains P4-85
- [x] `jpeg_quality_scaling()` — Quality to scale factor conversion (`quality_scaling()`)
- [ ] Classic `force_baseline` table semantics — native `Encoder::force_baseline()` is ready; classic setup remains P4-85

### Huffman Tables
- [x] Standard DC/AC luminance + chrominance tables
- [x] `TJPARAM_OPTIMIZE` — 2-pass optimized Huffman (`compress_optimized`)
- [ ] Classic custom `dc_huff_tbl_ptrs[4]` wiring — native `Encoder::huffman_dc_table()` is ready; classic scanline output remains P4-85
- [ ] Classic custom `ac_huff_tbl_ptrs[4]` wiring — native `Encoder::huffman_ac_table()` is ready; classic scanline output remains P4-85
- [x] `jpeg_alloc_huff_table()` — N/A for Rust (Huffman tables are value types, no allocation needed)
- [ ] Classic `jpeg_suppress_tables()`/sent-state semantics — native abbreviated streams are ready; classic cinfo reuse remains P4-87

### Entropy Coding Mode
- [x] `TJPARAM_PROGRESSIVE` — Progressive mode
- [x] `TJPARAM_ARITHMETIC` — Arithmetic coding
- [x] `TJPARAM_ARITHMETIC` + `TJPARAM_PROGRESSIVE` combined — SOF10 encode
- [ ] Classic arithmetic DAC conditioning fields — native arithmetic defaults work; public conditioning remains P4-90

### Lossless Mode
- [x] `TJPARAM_LOSSLESS` — Enable lossless
- [x] `TJPARAM_LOSSLESSPSV` — Predictor selection 1-7 (`Encoder::lossless_predictor()`)
- [x] `TJPARAM_LOSSLESSPT` — Point transform 0-15 (`Encoder::lossless_point_transform()`)
- [x] Lossless multi-component (color) encode (`compress_lossless_extended()`)
- [ ] Full classic `jpeg_enable_lossless()` validation/public-state contract — native builder works; P4-107
- [ ] Classic arithmetic+lossless error contract — native SOF11 is supported, but classic compatibility remains P4-89

### Restart Markers
- [x] `TJPARAM_RESTARTBLOCKS` — Restart interval in MCU blocks (`Encoder::restart_blocks()`)
- [x] `TJPARAM_RESTARTROWS` — Restart interval in MCU rows (`Encoder::restart_rows()`)
- [x] `restart_interval` field — via Encoder builder
- [x] `restart_in_rows` field — via Encoder builder

### JFIF / Density
- [ ] Classic `write_JFIF_header`/version/density fields — native marker controls are ready; classic scanline wiring remains P4-88
- [x] `TJPARAM_XDENSITY` — wired through `TjHandle` compress/decompress + `Encoder::density()`
- [x] `TJPARAM_YDENSITY` — wired through `TjHandle` compress/decompress + `Encoder::density()`
- [x] `TJPARAM_DENSITYUNITS` — wired through `TjHandle` compress/decompress + `Encoder::density()`
- [x] Native JFIF version configurable (`Encoder::jfif_version()`)
- [x] JFIF density read (`Image.density`)
- [x] Low-level density rewrite via coefficient API (`JpegCoefficients.{density_unit,x_density,y_density}`)

### Adobe Marker
- [x] Adobe APP14 default (CMYK/RGB-direct native encode)
- [ ] Classic `write_Adobe_marker` toggle — native `Encoder::write_adobe_marker()` is ready; classic scanline wiring remains P4-88

### Progressive Scan Control
- [ ] Full classic `jpeg_simple_progression()` semantics — progressive mode works, but public script installation remains P4-91
- [ ] Classic `scan_info` / `num_scans` wiring — native `Encoder::scan_script()` is ready; classic scanline wiring remains P4-91
- [x] `jpeg_scan_info` struct — `ScanScript` struct

### DCT Method
- [x] `JDCT_ISLOW` — Accurate integer DCT
- [x] `JDCT_IFAST` — Fast integer DCT (`DctMethod::IsFast`)
- [x] `JDCT_FLOAT` — Floating-point DCT (`DctMethod::Float`)
- [ ] Classic scanline `dct_method` forwarding — native methods are ready; classic lossy output remains P4-86

### Color Space Control
- [x] Auto YCbCr from RGB/RGBA/BGR/BGRA input
- [x] CMYK direct (no conversion)
- [ ] Full classic `jpeg_set_colorspace()` / `jpeg_default_colorspace()` scanline behavior — native controls work; P4-93
- [x] Native input/JPEG colorspace control via `Encoder::colorspace()` / `TjHandle`
- [ ] Classic `in_color_space` / `jpeg_color_space` scanline translation — P4-93
- [x] Grayscale-from-color encode option (`Encoder::grayscale_from_color()`)

### Input Options
- [x] `TJPARAM_BOTTOMUP` — Bottom-up row order (`Encoder::bottom_up()`)
- [x] `raw_data_in` — Encode from raw downsampled component data (`compress_raw()`)
- [x] Native `smoothing_factor` — Input smoothing (0-100) (`Encoder::smoothing_factor()`)
- [ ] Classic progressive/arithmetic smoothing composition — P4-84
- [x] Classic `do_fancy_downsampling` — mirrored no-op; upstream silently ignores it because compressor DCT scaling is unsupported. Native `Encoder::fancy_downsampling()` is an extension
- [ ] Classic `CCIR601_sampling` rejection — upstream raises `JERR_CCIR601_NOTIMPL`; shim handling remains P4-88
- [x] `input_gamma` — N/A (gamma correction is user-space preprocessing, not encoder responsibility; C field initialized to 1.0 and never applied)

### Marker Writing
- [x] JFIF APP0 (automatic)
- [x] EXIF APP1 (`compress_with_metadata`)
- [x] ICC APP2 (`compress_with_metadata`, multi-chunk)
- [x] Adobe APP14 (CMYK encode)
- [ ] Full classic `jpeg_write_marker()` state contract — native writer works; P4-105
- [ ] Full classic `jpeg_write_m_header()` / `jpeg_write_m_byte()` length/state contract — native writer works; P4-105
- [ ] Full classic `jpeg_write_icc_profile()` state/error contract — native ICC writing works; P4-105/P4-100
- [ ] Classic `jpeg_write_tables()` installed-table/sent-state semantics — native tables-only output is ready; classic reuse remains P4-87
- [x] COM (comment) marker write (`Encoder::comment()`, `marker_writer::write_com()`)

### Scanline-Level Encode API
- [ ] Full classic `jpeg_start_compress()` contract — basic start works; `write_all_tables` remains P4-87
- [ ] Full classic `jpeg_write_scanlines()` option contract — basic rows work; residual gaps are P4-84..P4-93
- [ ] Full classic `jpeg_finish_compress()` lifecycle/error contract — missing-rows/bad-state lifecycle matches stock (P4-106 closed 2026-08-14); error reporting is P4-100
- [ ] Full classic `jpeg_write_raw_data()` / `jpeg12_write_raw_data()` option contract — default raw encode works; P4-95
- [ ] Classic `jpeg12_write_scanlines()` / `jpeg16_write_scanlines()` encode completion — P4-94
- [x] `jpeg_calc_jpeg_dimensions()` — Compute compression-side JPEG dimensions; no compression scaling (`calc_jpeg_dimensions()`, P4-1 2026-05-10)
- [x] `next_scanline` tracking (`ScanlineEncoder::next_scanline()`)

---

## 7. Decompression Parameters (TJPARAM / jpeg_decompress_struct fields)

### Output Format
- [x] Configurable decoder resource limits (`DecodeLimits`, `Decoder::set_limits` — width/height/pixels/scans/memory, #355)
- [x] `no_std` + `alloc` core codec (`--no-default-features`; CI-built for `thumbv7em-none-eabihf`, #356)
- [x] Output pixel format selection (`decompress_to`)
- [x] XMP metadata accessor with Extended XMP reassembly (`Image::xmp_data`, `Encoder::xmp_data` — #358)
- [x] IPTC IIM accessor from the APP13 Photoshop IRB (`Image::iptc_data`, `Encoder::iptc_data` — #358)
- [x] Decode into caller-owned buffer (`decompress_into`, `output_buffer_size`, `Decoder::decode_image_into` — #354)
- [x] Scaled IDCT — all 16 factors: 1/8 through 2/1 (`set_scale`)
- [x] Crop decode (`decompress_cropped`, `set_crop_region`)
- [x] `TJPARAM_BOTTOMUP` — Bottom-up row order (`ScanlineDecoder::set_bottom_up()`)
- [x] Native explicit output colorspace (`Decoder::set_output_colorspace()`)
- [ ] Classic `out_color_space` translation and error contract — P4-98/P4-99
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
- [ ] Full classic `jpeg_save_markers()` incremental pointer-stability contract — native saving/truncation works; P4-26
- [ ] Classic `jpeg_set_marker_processor()` callback invocation — native callback API exists; P4-112
- [x] COM (comment) marker read/expose (`Image.comment`)
- [x] Arbitrary marker access via `marker_list` linked list (`Image.markers()` / `Image.saved_markers`)
- [x] JFIF version / density read (`Image.density`)

### Multi-Scan / Progressive Output
- [ ] Classic `jpeg_has_multiple_scans()` state semantics — the reported bit matches upstream since 2026-08-13 (progressive ∨ non-interleaved sequential first scan, `jdinput.c:153-156`, pinned by the P4-14 `hms_*` oracle rows); the P4-114 state handling remains
- [x] `buffered_image` mode — Enable scan-by-scan output (`ProgressiveDecoder`)
- [ ] Full classic `jpeg_start_output()` / `jpeg_finish_output()` input-pull/state contract — state guards match stock (P4-104 closed 2026-08-14); input-pull is P4-26, `DSTATE_BUFIMAGE` pass-walking is P4-13
- [ ] Full classic `jpeg_consume_input()` contract — suspension core and state dispatch match stock (P4-104 closed 2026-08-14); deeper streaming fidelity is P4-13/P4-26
- [ ] Full classic `jpeg_input_complete()` state/streaming contract — answers upstream's `eoi_reached` (P4-104 closed 2026-08-14); deeper streaming fidelity is P4-26

### Scanline-Level Decode API
- [ ] Full classic `jpeg_read_header()` public state/tables contract — entry guard and post-parse state match stock (P4-104 closed 2026-08-14); metadata/tables are P4-99/P4-101
- [ ] Full classic `jpeg_start_decompress()` option/state contract — native decoder works; P4-96/P4-99 (published state closed with P4-104 2026-08-14, except `DSTATE_BUFIMAGE` → P4-13)
- [ ] Full classic `jpeg_read_scanlines()` option/error contract — basic rows work; P4-96/P4-99/P4-100
- [x] `jpeg_skip_scanlines()` — Skip rows during decode (`ScanlineDecoder::skip_scanlines()`)
- [ ] Full classic `jpeg_crop_scanline()` iMCU-aligned/state semantics — native exact crop exists; P4-103
- [ ] Full classic `jpeg_finish_decompress()` lifecycle/suspension contract — lifecycle and suspension match stock (P4-104 closed 2026-08-14, oracle-compared); error reporting is P4-100
- [ ] Full classic `jpeg_read_raw_data()` option/state/error contract — native raw decode works; P4-102
- [ ] Full classic `jpeg12_read/skip/crop_scanline` lifecycle/options — native precision decode exists; P4-98
- [ ] Full classic `jpeg16_read_scanlines` lifecycle/options — native precision decode exists; P4-98
- [ ] Classic calculated dimensions matching actual scaled decode — helpers exist; P4-99
- [ ] Classic public component/core dimensions for odd-size sampling — P4-99
- [x] `output_scanline` tracking (`ScanlineDecoder::output_scanline()`)

### Color Quantization (8-bit indexed output)
- [ ] Classic `quantize_colors` output — native `quantize::quantize()` is ready; classic decode remains P4-96
- [x] `desired_number_of_colors` / `actual_number_of_colors` (`QuantizeOptions::num_colors`, `QuantizedImage::palette.len()`)
- [x] `dither_mode` — JDITHER_NONE / JDITHER_ORDERED / JDITHER_FS (`DitherMode` enum)
- [ ] Classic `two_pass_quantize` — native median-cut is ready; classic decode remains P4-96
- [ ] Classic external `colormap` and enable flags — native `QuantizeOptions` is ready; classic decode remains P4-96
- [ ] Classic `jpeg_new_colormap()` — native `requantize()` is ready; classic buffered-image switching remains P4-96

---

## 8. Metadata

- [x] APP0 JFIF — Read / write
- [x] APP1 EXIF — Read / write (orientation parsing)
- [x] APP2 ICC profile — Read (multi-chunk reassembly) / write (multi-chunk)
- [ ] Classic `jpeg_read_icc_profile()` saved-marker/header contract — native ICC reassembly exists; P4-113
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
- [x] TJXOPT_TRIM (2) — Discard partial iMCU edges (`TransformOptions.trim`); an axis holding less than one whole iMCU is left untrimmed rather than rejected, matching `trim_right_edge`/`trim_bottom_edge` (P4-117)
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
- [ ] Full classic `jpeg_copy_critical_parameters()` source-table contract — native copy works; P4-101
- [x] `tjtransform.customFilter` — User callback for coefficient inspection/modification
- [x] `tj3TransformBufSize()` — Output buffer size estimation (`transform_buf_size()`)

---

## 10. YUV / Planar API

### RGB → YUV (color conversion only, no JPEG)
- [x] `tj3EncodeYUV8()` — RGB → packed YUV buffer (runs `yuv::encode_yuv_planes()` and packs the result itself, keeping one plane for `TJSAMP_GRAY`; `encode_yuv()` packs whatever the *pixel* format implies, which is not the same count — P4-165)
- [x] `tj3EncodeYUVPlanes8()` — RGB → separate Y/Cb/Cr plane buffers (`yuv::encode_yuv_planes()`)

### YUV → JPEG (compress from YUV)
- [x] `tj3CompressFromYUV8()` — Packed YUV → JPEG (the entry point splits the packed buffer itself and runs `yuv::compress_from_yuv_planes()`; `compress_from_yuv()` infers the plane count from the buffer length, which cannot express `TJSAMP_GRAY` — P4-165)
- [x] `tj3CompressFromYUVPlanes8()` — Planar YUV → JPEG (`yuv::compress_from_yuv_planes()`)

### JPEG → YUV (decompress to YUV)
- [x] `tj3DecompressToYUV8()` — JPEG → packed YUV buffer (`yuv::decompress_to_yuv()`; the two are not interchangeable — the C entry point rejects 4-component CMYK/YCCK frames per P4-125, the Rust function packs all four planes)
- [x] `tj3DecompressToYUVPlanes8()` — JPEG → separate Y/Cb/Cr plane buffers (`yuv::decompress_to_yuv_planes()`; same divergence — the Rust function returns one plane per SOF component, so four for CMYK/YCCK)

### YUV → RGB (color conversion only, no JPEG)
- [x] `tj3DecodeYUV8()` — Packed YUV → RGB (splits the packed buffer and runs `yuv::decode_yuv_planes()`; same reason as `tj3CompressFromYUV8` — P4-165)
- [x] `tj3DecodeYUVPlanes8()` — Planar YUV → RGB (`yuv::decode_yuv_planes()`)

### Buffer Size Helpers
- [x] `tj3YUVBufSize()` — Total packed YUV buffer size (`yuv_buf_size()`)
- [x] `tj3YUVPlaneSize()` — Single plane buffer size (`yuv_plane_size()`)
- [x] `tj3YUVPlaneWidth()` — Plane width in samples (exported symbol runs the capi-local `plane_width()`, not the root-crate `yuv_plane_width()`, which cannot express C's `componentID >= nc` bound — P4-126)
- [x] `tj3YUVPlaneHeight()` — Plane height in rows (capi-local `plane_height()`; same relationship to `yuv_plane_height()`)

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
- [x] Row-streaming H2V2 / H2V1 / H1V2 upsample+color pipeline (fused, no full-plane alloc)
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
- [x] Classic `jpeg_CreateCompress` / `jpeg_CreateDecompress` version/size guards — P4-110 (closed 2026-08-11; `JERR_BAD_LIB_VERSION` / `JERR_BAD_STRUCT_SIZE` compared against a real libjpeg by `capi_create_abi_guards.rs`)
- [x] Full classic `jpeg_stdio_dest()` contract — short writes, `fflush`, and `ferror` raise `JERR_FILE_WRITE`; foreign-manager reuse raises `JERR_BUFFER_SIZE` (P4-108)
- [x] Full classic `jpeg_stdio_src()` contract — chunked `fread` through the caller's `FILE *`, trace-compared vs stock (P4-109, 2026-08-14)
- [x] Full classic `jpeg_mem_dest()` ownership/reallocation contract — caller capacity honoured, caller buffers never freed, doubling growth into library memory (P4-108)
- [x] Full classic `jpeg_mem_src()` validation/manager-replacement contract — trace-compared vs stock (P4-109, 2026-08-14)
- [x] Custom `jpeg_destination_mgr` — User-defined output stream (`stream::compress_to_writer`)
- [x] Custom `jpeg_source_mgr` — User-defined input stream (`stream::decompress_from_reader`, buffering; `decompress_from_reader_incremental` for bounded input memory on interleaved baseline — P4-58)
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
- [x] `max_memory_to_use` / `max_alloc_chunk` — `Decoder::set_max_memory()` / `TjHandle` `TJPARAM_MAXMEMORY`, **plus classic enforcement (P4-14: vtable since 2026-08-11, decode sequence since 2026-08-13).** `cinfo->mem->max_memory_to_use` is compared against in `realize_virt_arrays` and, for the classic `jpeg_read_header` → `jpeg_start_decompress` sequence, shim-side at start with upstream's scope (multi-scan or buffered-image streams) and accounting (whole-image coefficient bytes only), raising `JERR_NO_BACKING_STORE` ("Memory limit exceeded", 51) exactly as upstream's no-backing-store build does. Its default is upstream's `0` (unlimited), not the `1000000000L` this line used to claim. **Residues** (strip-wise realization, already-allocated overhead, suspending buffered corner) are recorded in P4-14; see the P4-14 section of `docs/ABI_COMPATIBILITY.md`.
- [x] `tj3Alloc()` / `tj3Free()` — N/A (Rust ownership; `Vec<u8>` return replaces C caller-managed buffers)

---

## 13. Error Handling

- [x] `Result<T, JpegError>` for all public operations
- [x] `DecodeWarning` list (HuffmanError, TruncatedData) in lenient mode
- [x] Custom error handler — `ErrorHandler` trait
- [ ] Classic codec failures consistently reach `error_exit()` — native typed errors exist; P4-100
- [x] `emit_message()` callback — `ErrorHandler::emit_warning()` + `ErrorHandler::trace()`
- [x] `output_message()` / `format_message()` — Rust `Display` trait on `JpegError` replaces the callbacks for the native API; the C ABI implements both per `jerror.c` — `format_message` renders upstream's text (P4-146, 2026-08-11), `output_message` writes it to stderr through the installed formatter (P4-146 criterion 4, 2026-08-13)
- [x] `reset_error_mgr()` — N/A (Rust `Result` is stateless; no accumulated error state to reset)
- [x] `trace_level` control — `ErrorHandler::trace()` callback with level parameter
- [x] `num_warnings` counter — `Image.warnings` vec (count via `.len()`)
- [x] `msg_code` / `msg_parm` / `jpeg_message_table` — the native API uses typed `JpegError` / `DecodeWarning` enums instead of C integer codes + format strings; the C ABI carries all three, `jpeg_std_error` installing the 129-entry upstream table (P4-146, 2026-08-11)
- [x] `tj3GetErrorStr()` / `tj3GetErrorCode()` — Rust `Result<T, JpegError>` with `Display` impl replaces C per-handle error getters
- [x] Classic `jpeg_resync_to_restart()` default algorithm — suspending-source trace vs stock; native strategy extension remains available separately (P4-97, 2026-08-14)

---

## 14. Progress Monitoring

- [ ] Classic `jpeg_progress_mgr` wiring — native `ProgressListener` exists; P4-111
- [ ] Classic `progress_monitor()` callback invocation — native listener exists; P4-111
- [ ] Classic `pass_counter` / `pass_limit` publication — P4-111
- [ ] Classic `completed_passes` / `total_passes` publication — P4-111

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
- Treat the checklist above as the feature map and `docs/LAST_MILE.md` as the authoritative live-gap index. P4-13 is partial (its real suspension core is proven; deeper streaming is P4-26), P4-14 remains open, and the classic C-ABI audit is tracked as P4-84..P4-115 (the test-integrity half, P4-116, closed 2026-08-08). Closed historical items stay in `docs/last_mile/phase4.md`. PNG for `tj3LoadImage8()` / `tj3SaveImage8()` remains behind the optional `png` feature, mirroring upstream's build-time flag.

## Recent Additions (batch reconciliation)

The following capabilities landed in earlier batches; any residual scope is
named explicitly rather than implied complete:

- **Arithmetic table count widened 4 → 16** per spec F.2.4.3 (DAC parser/writer + `ArithDecoder`/`ArithEncoder`); spec-compliant streams with `tbl_no > 3` now decode/encode correctly.
- **Abbreviated datastream** (JPEG spec F.1.2.4): `jpeg_write_tables()`, `HeaderResult::TablesOnly`, `Decoder::new_with_tables()` inter-session table reuse, `Encoder::suppress_tables(bool)` body-only output.
- **12-bit raw planar I/O**: `compress_raw_12()` / `decompress_raw_12()` have stock-C odd-size coverage for baseline Huffman S420/S422/S444/S440/S411/S441. S410/S24 and progressive/arithmetic mode coverage remain P4-115.
- **12/16-bit PPM file I/O**: `load_ppm_12bit()` / `save_ppm_12bit()` / `load_ppm_16bit()` / `save_ppm_16bit()` matching C scope (PPM only).
- **TjHandle gaps**: `TJPARAM_NOREALLOC` (`compress_into()` returns `BufferTooSmall{need,got}`), `TJPARAM_SAVEMARKERS` behavioral wiring, `tj3GetICCProfile()` handle symmetry with `Image.icc_profile()`.
- **Session-reset APIs**: `Encoder::reset_colorspace()` (`jpeg_default_colorspace`), `Encoder::reset_quant_tables(force_baseline)` (`jpeg_default_qtables`).
- **Restart resync hook**: `RestartResyncStrategy` trait + `Decoder::set_resync_strategy()` with `ResyncAction {Continue, Skip, Abort}` — replaces the internal-only implementation of `jpeg_resync_to_restart`.
- **JPEG-in-RAW thumbnail**: `extract_embedded_jpeg()` walks TIFF IFDs (LE/BE, bounds-checked) to extract embedded JPEG thumbnails from ARW/CR2-style files.
- **C ABI shim crate** (`crates/libjpeg-turbo-rs-capi`, cdylib + staticlib): exports TJ3, 21 legacy TJ aliases, and a broad classic `jpeg_*` surface with v8 SONAME/pkg-config packaging. Symbol/layout and selected downstream paths are verified; this is not a blanket classic behavioral-parity claim (P4-84..P4-114).

## Second-Batch Reconciliation (parallel follow-up workers)

- **A4 restored** (reverted revert after confirming the earlier corpus-test crash was classification-only, not an A4 regression): arithmetic table count widened 4 → 16 per spec F.2.4.3 is live on `main`.
- **FFI A1-11 export milestone**: the named decode symbols and public v8 struct mirror landed. Their remaining behavioral contracts are now tracked in P4-96..P4-114, not in the historical coordinator notes.
- **tj3 auxiliary surfaces** added by B9-5: `tj3GetScalingFactors`, `tj3YUVBufSize`, `tj3YUVPlaneSize`, `tj3YUVPlaneWidth`, `tj3YUVPlaneHeight`, `tj3JPEGBufSize`, `tj3InitVersion`, `tj3LoadImage{8,12,16}`/`tj3SaveImage{8,12,16}`, `TJBUFSIZE`, `TJBUFSIZEYUV`, `tjBufSizeYUV`, process-global no-handle error slot for `tj3GetErrorStr(NULL)`.
- **`tj3Init` enum fix**: `TJINIT_COMPRESS/DECOMPRESS/TRANSFORM` were previously treated as bit flags (1/2/4) instead of enum values (0/1/2); every C caller was getting NULL. Fixed, and exercised by the tjunittest link harness.
- **`tj3Compress8` NOREALLOC in-place fix**: when `TJPARAM_NOREALLOC == 1` and the caller pre-supplies `*jpegBuf`, we now write in place via `copy_nonoverlapping` instead of allocating a fresh libc buffer and swapping the pointer. The old behavior leaked every iteration of the tjunittest `doTest` loop and eventually corrupted the malloc heap (SIGSEGV at `_os_unfair_lock_unlock_slow` inside `mfm_alloc`). Now tjunittest runs to completion on our shim.
- **Which upstream release each C oracle runs against (P4-130)**: apt ships 2.1.x on Ubuntu 24.04, which lacks `-lossless`, `-precision`, and SOF3 decode, so CI installs upstream's own `libjpeg-turbo-official_${VERSION}_${ARCH}.deb` under `/opt/libjpeg-turbo` instead. Since 2026-08-17 there are **two tool legs, not one**, and `docs/oracle_versions.tsv` is the machine-checked statement of what each one is:

  | Role | Version | What it backs |
  | --- | --- | --- |
  | `tool-baseline` | **3.1.4.1** (2026-03-27) | The behaviour-regression leg: `Integration Tests`, plus the cross-arch, fuzz-smoke and full-C-parity workflows. The release the expectations in this document were written against. |
  | `tool-current` | **3.2.0** (2026-06-30) | The current-parity leg: `Integration Tests (oracle 3.2.0)` runs the root `cargo test --tests` matrix *and* the C-ABI crate's TurboJPEG and stock-tool oracle suites against current upstream stable, selected by `LIBJPEG_TURBO_PREFIX` rather than PATH order. Measured green at 223 suite sections / 2351 tests when the leg landed, plus 19 sections / 107 tests when the C-ABI suites joined it (macOS aarch64, both). |
  | `trace-current` | **3.2.0** (2026-06-30) | The same release built from source with `WITH_JPEG8=1` (`/tmp/ljt320v8/prefix`), for the classic-ABI trace suites: they compile a C oracle and compare traces line by line at the v8 ABI, which upstream's `JPEG_LIB_VERSION 62` deb cannot serve. |
  | `submodule` | **3.1.90** = 3.2 beta1 (2026-03-27) | `references/libjpeg-turbo`: built with `WITH_JPEG8=1` as the *baseline* classic-ABI trace oracle (`/tmp/ljt8/prefix`), and the source every `j*.c:NNN` citation in this repository quotes. |

  So a ✅ backed by a *tool* oracle means "matches 3.1.4.1 **and** 3.2.0" — for the root differential matrix since 2026-08-17, and for the C-ABI crate's oracle suites since the C-ABI half of criterion 1 landed. A ✅ backed by a *source citation* has always meant 3.1.90. That split existed before it was written down — it is what the P4-130 triage found first. `tests/oracle_version_pins.rs` fails if a workflow pins a version this table does not declare, if it declares a leg no workflow installs, or if a C-ABI suite that compares against C runs on the baseline leg alone.

  A single global bump remains explicitly *not* the fix: it would re-baseline every existing expectation at once, making anything it papers over indistinguishable from a pass.

  **Upstream-release tracking policy (P4-130 criterion 4).** 3.2.0 sat unnoticed for two months because nothing was watching. The policy is now mechanical:
  1. **Watch releases, not the calendar.** `.github/workflows/upstream-currency.yml` runs `scripts/check_oracle_currency.sh` weekly (and on any PR touching the manifest or the script): it compares `tool-current` against upstream's latest stable release and fails when they differ. Failing *is* the notification — a human reading release notes is what already failed.
  2. **A new upstream minor opens a triage item, not a bump.** Enumerate the release notes into deltas, and triage each to exactly one of: already at parity (naming the differential test that proves it), a new OPEN LAST_MILE entry, or a recorded non-goal. No delta left untriaged. The 3.2 triage is the worked example, in P4-130's phase-file entry.
  3. **Pins move one leg at a time.** Add the new version alongside the old so a divergence shows up as a difference between two legs. Retire the old leg only once its expectations are known to hold on the new one.
  4. **This table names the version each gate runs against**, so "parity" is never read as a claim about a release nobody tested.
- **B1 root-suite helper migration**: 90 root test files were migrated to `helpers::require_c_tool!()`. A later workspace audit found residual private helpers and result-to-skip paths; those were swept under P4-116, which closed 2026-08-08. Every C-tool lookup and capability probe now fails closed in CI, via `require_c_tool!`, `helpers::optional_c_tool`, or `helpers::skip_missing_c_capability`.
- **B9-2 Pillow / B9-3 ImageMagick**: link harnesses are active (`examples/pillow_smoke/`, `examples/imagemagick_smoke/`, `tests/capi_pillow_compat.rs`, `crates/libjpeg-turbo-rs-capi/tests/capi_imagemagick_compat.rs`). They are no longer `#[ignore]`d; the tests only soft-skip for missing external tools or macOS loader-injection restrictions.
- **B9-4 stock djpeg–cjpeg–jpegtran parity achieved**: `examples/stock_djpeg_cjpeg/build.sh` builds stock `djpeg`, `cjpeg`, and `jpegtran` against our capi over the full `references/libjpeg-turbo/testimages/*.jpg` corpus (8-bit `testorig`, arithmetic `testimgari`, integer-quant `testimgint`, 12-bit `monkey12`). `djpeg` and `jpegtran -copy all -rotate 90` are byte-identical to upstream; cjpeg must either be byte-identical or produce byte-identical output after both files are successfully decoded by stock djpeg. Unblocked by the 33 classic `jpeg_*` symbols (A1-12) + precision-routing fixes in `jpeg_start_decompress` / `jpeg12_read_scanlines` / `jpeg16_read_scanlines` / `jpeg12_skip_scanlines` and the new `jpeg_calc_output_dimensions` export (commit `f70b41f`).
- **B9-5 tjunittest**: now passes 100% on the capi cdylib — `EXIT=0`, 0 ERROR, 0 FAILED, 1012 subtest passes. Closed by four 2026-04-25 fixes: grayscale-to-RGB repack in `tj3Decompress8`, TJSAMP_441/410/24 enum widening end-to-end, CMYK SOF sampling factors honoring TJPARAM_SUBSAMP, and TJPARAM_FASTUPSAMPLE wired into the 4-component decode upsample path.
- **SIMD upsample width=2 kernel guard (closed 2026-04-25)**: added `if in_width == 2` box-filter short-circuit to all four `fancy_upsample_h2v1` kernels (SSE2/AVX2/NEON/WASM) so the raw kernels now match scalar fancy_h2v1 byte-for-byte across every width. Revived `tests/simd_x86.rs::sse2_upsample_edge_cases`, upgraded `tests/simd_avx2.rs::avx2_upsample_width_2` to actually exercise the AVX2 kernel, and widened `UPSAMPLE_WIDTHS` in `tests/simd_parity.rs` to include widths 1 and 2.

## Classic `jpeg_*` API (Third Batch — C1 + C2)

Two parallel workers shipped **36 new `#[no_mangle] extern "C"` symbols** in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`:

- **Decode extensions (C1, 12 symbols)**: `jpeg_skip_scanlines`, `jpeg_crop_scanline`, `jpeg_save_markers`, `jpeg_set_marker_processor`, `jpeg_read_icc_profile`, `jpeg_read_coefficients`, `jpeg_copy_critical_parameters`, `jpeg_core_output_dimensions`, `jpeg12_read_scanlines`, `jpeg12_skip_scanlines`, `jpeg12_crop_scanline`, `jpeg16_read_scanlines`. High-precision state lives in a `thread_local!` side table keyed by the cinfo pointer.
- **Encode side + utilities (C2, 24 symbols)**: `jpeg_CreateCompress`/`jpeg_destroy_compress`, `jpeg_stdio_dest`/`jpeg_mem_dest`, `jpeg_set_defaults`/`jpeg_set_colorspace`/`jpeg_default_colorspace`, `jpeg_set_quality`, `jpeg_start_compress`/`jpeg_write_scanlines`/`jpeg_finish_compress`, `jpeg_quality_scaling`, `jpeg_add_quant_table`, `jpeg_default_qtables`, `jpeg_simple_progression`, `jpeg_enable_lossless`, `jpeg_suppress_tables`, `jpeg_write_marker`/`jpeg_write_m_header`/`jpeg_write_m_byte`, `jpeg_write_icc_profile`, `jpeg_write_tables`, `jpeg12_write_scanlines`/`jpeg16_write_scanlines`, `jpeg_write_coefficients`, `jpeg_resync_to_restart`, `jcopy_block_row`, `jdiv_round_up`.
- **Test count**: 38 (decode) + 15 (encode) new dlopen-and-exercise tests, all green on main.

Those counts record the historical export milestone. P4-116, which tracked
cases where a green test could still represent zero executed comparisons,
closed 2026-08-08.

## Stock-tool byte-exact milestone (B9-4)

- `examples/stock_djpeg_cjpeg/build.sh` builds stock **`djpeg`, `cjpeg`, `jpegtran`** against `libjpeg.62.dylib` / `libjpeg.so.62` produced by our capi crate, **zero undefined symbols** on all three. Per-precision wrappers (`wrppm-{8,12,16}.c`, `wrgif-{8,12}.c`, `rdppm-{8,12,16}.c`, `rdcolmap-{8,12}.c`) built with `-DBITS_IN_JSAMPLE=N` supply the `j12init_*` / `j16init_*` entry points.
- **Byte-exact corpus parity**: every JPEG in `references/libjpeg-turbo/testimages/` now round-trips `cmp -s` against upstream `djpeg` output — including the 12-bit `monkey12.jpg` (149×227 precision=12 PPM).
- Precision routing fixes that made 12-bit work:
  - `jpeg12_read_scanlines` / `jpeg16_read_scanlines` / `jpeg12_skip_scanlines` now advance `cinfo.output_scanline` (was spinning forever inside djpeg's `while (output_scanline < output_height)` loop).
  - `jpeg_start_decompress` fast-path for `data_precision > 8`: no longer drives the 8-bit Rust decoder (which silently succeeded on 12-bit input and clobbered `data_precision` 12→8, misrouting djpeg's precision dispatch).
  - `jpeg_calc_output_dimensions` exported and mirrors `jdmaster.c:267` (non-IDCT-scaling common case + JCS_EXT_* color-space pixelsize).

## Testing Infrastructure Additions

- **Fuzz corpus**: expanded from 22 → ~2,194 seeds across 7 targets (Cartesian product of subsamp × quality × content × entropy-mode); new `fuzz_encode_roundtrip` target; `scripts/fuzz_minimize.sh`; OSS-Fuzz stub at `oss-fuzz/`; nightly `.github/workflows/fuzz-smoke.yml`.
- **Cross-arch CI matrix** (`.github/workflows/cross-arch.yml`): `ubuntu-24.04-arm` (aarch64 NEON), x86_64 AVX2 default, an x86_64 AVX2-disabled **build check** via `-C target-feature=-avx2,-sse4.2` (compile-time only — the workflow's own comment records that it does not exercise the runtime fallback; the CPUID-masked `test-linux-x86_64-no-avx2-emulated` job is what tests it, #320), macOS arm64 retained, WASM SIMD128 smoke on every PR.
- **Per-SIMD bit-exact parity suite** (`src/simd/simd_parity_tests.rs`, relocated from `tests/simd_parity.rs` by P4-135 criterion 2): 20 kernel × backend combinations (NEON / AVX2 / SSE2 / WASM), 1000-iteration Mulberry32 PRNG, scalar↔SIMD bit-exact assertions.
- **Conformance suite**: `scripts/fetch_conformance.sh` + `tests/worker_b3_conformance_t83*.rs` iterating `references/libjpeg-turbo/testimages/*.jpg` for pixel-exact djpeg comparison + decoded-pixel hash regression in `tests/reference_hashes_conformance.json`.
- **Real-world corpus**: fetch scripts + seed fixtures for Kodak PhotoCD (PSNR round-trip), USC-SIPI Miscellaneous (djpeg byte-exact), EXIF Orientation 1..8, CMYK scanner, JPEG-in-RAW thumbnail.
- **DoS bounds**: cross-platform peak-RSS + wall-clock measure helper, Huffman bomb (max 16-bit codes), progressive 5000-scan bomb with `SCANLIMIT` mitigation, restart-interval=1 4096×4096 bomb. Bounds documented from measured reality per CLAUDE.md tolerance rule.
- **Concurrency stress**: rayon-substituted `std::thread` stress (1000 concurrent decodes, interleaved Encoder/Decoder handoff via mpsc, shared custom quant table), plus loom permutation skeleton gated on `#[cfg(loom)]`.
- **CI C-tool enforcement**: `require_c_tool!` panics in CI when a required stock tool is missing, `helpers::optional_c_tool` does the same for a cross-check that is one part of a larger test, and `helpers::skip_missing_c_capability` does it for a probed switch the installed toolchain lacks. The earlier 90-file root-suite migration did not cover every workspace-private helper or exact matrix accounting; that residual audit closed as P4-116 on 2026-08-08.
- **libtiff end-to-end test wired** (`examples/libtiff_integration/` + `crates/libjpeg-turbo-rs-capi/tests/libtiff_integration.rs`): a C program opens a TIFF with `COMPRESSION_JPEG`, writes/reads strips via `TIFFWriteEncodedStrip` / `TIFFReadEncodedStrip`, with our cdylib staged as the JPEG provider via `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH`. This exercises the real downstream consumer of `jpeg_write_raw_data` / `jpeg_read_raw_data` (PR #240/#241). The test is active now; it soft-skips only when `cc`/libtiff are absent or on Windows. The former `JPEG_HEADER_TABLES_ONLY` gap is closed by the tables-only prefix splice path recorded in `docs/LAST_MILE.md`.

## Documentation Policy

- If a capability only exists through a different Rust API than the named C surface, describe that explicitly and do not count it as full parity for the original surface.
- If a row depends on a feature-gated or low-level path, say so in the row instead of promoting it to a blanket `[x]`.
