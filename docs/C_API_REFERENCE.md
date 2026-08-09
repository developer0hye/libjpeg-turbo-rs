# libjpeg-turbo C API → Rust Mapping Reference

> Every public C function from `turbojpeg.h` and `jpeglib.h` with description and Rust equivalent.
> ✅ = end-to-end public equivalent, ❌ = not yet, 🔶 = partial or different surface, N/A = not applicable in Rust
> If support only exists through a different Rust API surface, an internal helper, or a storage-only `TjHandle` field, use `🔶` rather than `✅`.

---

## TurboJPEG 3.0 API (`turbojpeg.h`)

### Handle Lifecycle

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Init(initType)` | Create compress/decompress/transform handle | `TjHandle::new()` | ✅ |
| `tj3Destroy(handle)` | Destroy handle | `Drop` (RAII) | ✅ |

### Parameter Get/Set

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Set(handle, param, value)` | Set integer parameter | `TjHandle::set()` | ✅ |
| `tj3Get(handle, param)` | Get integer parameter | `TjHandle::get()` | ✅ |

`TjHandle` get/set exists, but several parameters below are only partially wired through `compress()` / `decompress()`.

**All 26 TJPARAM values:**

| TJPARAM | Description | Rust | Status |
|---|---|---|---|
| `STOPONWARNING` | Treat warnings as fatal | `Decoder::set_stop_on_warning()` | ✅ |
| `BOTTOMUP` | Bottom-up row order | `Encoder::bottom_up()` / `ScanlineDecoder::set_bottom_up()` | ✅ |
| `NOREALLOC` | Disable output buffer realloc | `TjHandle::compress_into(&buf)` honors NOREALLOC, returns `BufferTooSmall{need,got}` on overflow | ✅ |
| `QUALITY` | Lossy quality 1-100 | `quality: u8` param | ✅ |
| `SUBSAMP` | Chroma subsampling | `subsampling: Subsampling` param | ✅ |
| `JPEGWIDTH` | JPEG image width (read-only) | `Image.width` | ✅ |
| `JPEGHEIGHT` | JPEG image height (read-only) | `Image.height` | ✅ |
| `PRECISION` | Sample precision 2-16 bits | `compress_12bit()`, `compress_16bit()`, `decompress_12bit()`, `decompress_16bit()`, `compress_lossless_arbitrary()` / `decompress_lossless_arbitrary()` | 🔶 |
| `COLORSPACE` | JPEG colorspace | `Encoder::colorspace()` / `Decoder::set_output_colorspace()` | 🔶 |
| `FASTUPSAMPLE` | Nearest-neighbor upsampling | `Decoder::set_fast_upsample()` | ✅ |
| `FASTDCT` | Fast DCT/IDCT algorithm | `Decoder::set_fast_dct()` | ✅ |
| `OPTIMIZE` | Optimized Huffman tables | `compress_optimized()` | ✅ |
| `PROGRESSIVE` | Progressive JPEG mode | `compress_progressive()` | ✅ |
| `SCANLIMIT` | Max progressive scans | `Decoder::set_scan_limit()` | ✅ |
| `ARITHMETIC` | Arithmetic entropy coding | `compress_arithmetic()`, `TransformOptions::arithmetic` | ✅ |
| `LOSSLESS` | Lossless JPEG mode | `compress_lossless()` | ✅ |
| `LOSSLESSPSV` | Lossless predictor 1-7 | `Encoder::lossless_predictor()` | ✅ |
| `LOSSLESSPT` | Lossless point transform 0-15 | `Encoder::lossless_point_transform()` | ✅ |
| `RESTARTBLOCKS` | Restart interval (MCU blocks) | `Encoder::restart_blocks()` | ✅ |
| `RESTARTROWS` | Restart interval (MCU rows) | `Encoder::restart_rows()` | ✅ |
| `XDENSITY` | Horizontal pixel density | `Encoder::density()` + `TjHandle` compress/decompress wiring | ✅ |
| `YDENSITY` | Vertical pixel density | `Encoder::density()` + `TjHandle` compress/decompress wiring | ✅ |
| `DENSITYUNITS` | 0=unknown, 1=ppi, 2=ppcm | `Encoder::density()` + `TjHandle` compress/decompress wiring | ✅ |
| `MAXMEMORY` | Memory limit | `Decoder::set_max_memory()` | ✅ |
| `MAXPIXELS` | Image size limit | `Decoder::set_max_pixels()` | ✅ |
| `SAVEMARKERS` | Marker preservation level 0-4 | `TjHandle` `TJPARAM_SAVEMARKERS` wired through `decompress()` → `Decoder::save_markers()` | ✅ |

### Memory

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Alloc(bytes)` | Allocate buffer | `Vec::with_capacity()` / owned buffers | 🔶 |
| `tj3Free(buffer)` | Free buffer | `drop()` / RAII | 🔶 |

### Buffer Size Calculation

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3JPEGBufSize(w, h, subsamp)` | Worst-case JPEG output size | `jpeg_buf_size()` | ✅ |
| `tj3YUVBufSize(w, align, h, subsamp)` | Total YUV buffer size | `yuv_buf_size()` | ✅ |
| `tj3YUVPlaneSize(comp, w, stride, h, subsamp)` | Single YUV plane size | `yuv_plane_size()` | ✅ |
| `tj3YUVPlaneWidth(comp, w, subsamp)` | YUV plane width | capi-local `plane_width()`; the root-crate `yuv_plane_width()` is the nearest Rust equivalent but takes a `Subsampling` with no grayscale variant, so it cannot apply C's `componentID >= nc` bound (P4-126) | ✅ |
| `tj3YUVPlaneHeight(comp, h, subsamp)` | YUV plane height | capi-local `plane_height()`; same relationship to `yuv_plane_height()` | ✅ |

### ICC Profile

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3SetICCProfile(handle, buf, size)` | Set ICC for encoding | `TjHandle::set_icc_profile()` / `Encoder::icc_profile()` | ✅ |
| `tj3GetICCProfile(handle, &buf, &size)` | Get ICC after decode | `TjHandle::icc_profile()` populated by `decompress()` + symmetric with `Image.icc_profile()` | ✅ |

### Compression (8-bit)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Compress8(handle, src, w, pitch, h, pf, &dst, &size)` | Compress 8-bit pixels to JPEG | `compress()`, `compress_optimized()`, etc. | ✅ |
| `tj3Compress12(handle, src, w, pitch, h, pf, &dst, &size)` | Compress 12-bit pixels | `TjHandle::compress_12bit()` / `compress_12bit()` | ✅ |
| `tj3Compress16(handle, src, w, pitch, h, pf, &dst, &size)` | Compress 16-bit pixels (lossless only) | `TjHandle::compress_16bit()` / `compress_16bit()` | ✅ |

### Compression from YUV

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3CompressFromYUV8(handle, src, w, align, h, &dst, &size)` | Compress packed YUV to JPEG | `yuv::compress_from_yuv()` | ✅ |
| `tj3CompressFromYUVPlanes8(handle, planes, w, strides, h, &dst, &size)` | Compress planar YUV to JPEG | `yuv::compress_from_yuv_planes()` | ✅ |

### Color Encode (RGB → YUV, no JPEG)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3EncodeYUV8(handle, src, w, pitch, h, pf, dst, align)` | RGB → packed YUV | `yuv::encode_yuv()` | ✅ |
| `tj3EncodeYUVPlanes8(handle, src, w, pitch, h, pf, planes, strides)` | RGB → planar YUV | `yuv::encode_yuv_planes()` | ✅ |

### Decompression Header

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecompressHeader(handle, jpeg, size)` | Parse JPEG headers, populate params | `Decoder::new()` / `ScanlineDecoder::new()` | ✅ |

### Scaling & Cropping

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3GetScalingFactors(&count)` | Get list of supported scaling factors | `TjHandle::scaling_factors()` / `ScalingFactor` | ✅ |
| `tj3SetScalingFactor(handle, sf)` | Set output scaling | `Decoder::set_scale()` / `TjHandle::set_scaling_factor()` | ✅ |
| `tj3SetCroppingRegion(handle, region)` | Set crop region | `Decoder::set_crop_region()` / `TjHandle::set_cropping_region()` | ✅ |

### Decompression (8-bit)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Decompress8(handle, jpeg, size, dst, pitch, pf)` | Decompress JPEG to 8-bit pixels | `decompress()`, `decompress_to()`, `decompress_into()` (caller buffer, #354) | ✅ |
| `tj3Decompress12(handle, jpeg, size, dst, pitch, pf)` | Decompress to 12-bit | `TjHandle::decompress_12bit()` / `decompress_12bit()` | ✅ |
| `tj3Decompress16(handle, jpeg, size, dst, pitch, pf)` | Decompress to 16-bit | `TjHandle::decompress_16bit()` / `decompress_16bit()` | ✅ |

### Decompression to YUV

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecompressToYUV8(handle, jpeg, size, dst, align)` | JPEG → packed YUV | `yuv::decompress_to_yuv()` — not interchangeable: the C entry point rejects 4-component CMYK/YCCK frames (P4-125), the Rust function packs all four planes | ✅ |
| `tj3DecompressToYUVPlanes8(handle, jpeg, size, planes, strides)` | JPEG → planar YUV | `yuv::decompress_to_yuv_planes()` — same divergence; the Rust function returns one plane per SOF component, so four for CMYK/YCCK | ✅ |

### Color Decode (YUV → RGB, no JPEG)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecodeYUV8(handle, src, align, dst, w, pitch, h, pf)` | Packed YUV → RGB | `yuv::decode_yuv()` | ✅ |
| `tj3DecodeYUVPlanes8(handle, planes, strides, dst, w, pitch, h, pf)` | Planar YUV → RGB | `yuv::decode_yuv_planes()` | ✅ |

### Lossless Transform

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Transform(handle, jpeg, size, n, &dstBufs, &dstSizes, transforms)` | Lossless transform with options | `transform_jpeg()` / `transform_jpeg_with_options()` (all ops + all TJXOPT flags, including arithmetic/progressive output, + custom filter) | ✅ |
| `tj3TransformBufSize(handle, transform)` | Estimate output buffer size | `transform_buf_size()` | ✅ |

### Error Handling

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3GetErrorStr(handle)` | Get error message string | `JpegError` Display impl (no per-handle getter) | 🔶 |
| `tj3GetErrorCode(handle)` | Get TJERR_WARNING or TJERR_FATAL | `Result<T, JpegError>` / no C-style getter | 🔶 |

### Image File I/O

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3LoadImage8(handle, filename, &w, align, &h, &pf)` | Load 8-bit BMP/PPM/PGM subset | `load_image()` / `load_image_from_bytes()` | 🔶 |
| `tj3SaveImage8(handle, filename, buf, w, pitch, h, pf)` | Save 8-bit BMP/PPM/PGM subset | `save_bmp()` / `save_ppm()` | 🔶 |
| `tj3LoadImage12(...)` / `tj3SaveImage12(...)` | 12-bit file I/O | `load_ppm_12bit()` / `save_ppm_12bit()` (PPM, matching C scope) | ✅ |
| `tj3LoadImage16(...)` / `tj3SaveImage16(...)` | 16-bit file I/O | `load_ppm_16bit()` / `save_ppm_16bit()` (PPM, matching C scope) | ✅ |

---

## libjpeg API (`jpeglib.h`)

### Initialization & Destruction

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_std_error(err)` | Create default error manager | `JpegError` enum | ✅ |
| `jpeg_create_compress(cinfo)` | Create compression struct | `Encoder` / `ScanlineEncoder`; classic version/size guards remain P4-110 | 🔶 |
| `jpeg_create_decompress(cinfo)` | Create decompression struct | `Decoder::new()` / `ScanlineDecoder::new()`; classic version/size guards remain P4-110 | 🔶 |
| `jpeg_destroy_compress(cinfo)` | Destroy compressor | RAII / `Drop` | ✅ |
| `jpeg_destroy_decompress(cinfo)` | Destroy decompressor | RAII / `Drop` | ✅ |
| `jpeg_abort_compress(cinfo)` | Abort compression | N/A (RAII) | N/A |
| `jpeg_abort_decompress(cinfo)` | Abort decompression | N/A (RAII) | N/A |
| `jpeg_abort(cinfo)` | Abort any operation | N/A (RAII) | N/A |
| `jpeg_destroy(cinfo)` | Destroy any handle | N/A (RAII) | N/A |

### Data Source / Destination

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_stdio_dest(cinfo, file)` | Output to FILE* | Full classic contract: `OUTPUT_BUF_SIZE` staging, short-write/`fflush`/`ferror` → `JERR_FILE_WRITE`, foreign-manager reuse → `JERR_BUFFER_SIZE` (P4-108) | ✅ |
| `jpeg_stdio_src(cinfo, file)` | Input from FILE* | Native reader exists; classic FILE buffering/Windows/error semantics remain P4-109 | 🔶 |
| `jpeg_mem_dest(cinfo, &outbuf, &outsize)` | Output to memory buffer | Full classic contract: `*outsize` is caller capacity, caller buffers are filled in place and never freed, growth doubles into library memory (P4-108) | ✅ |
| `jpeg_mem_src(cinfo, inbuf, insize)` | Input from memory buffer | Native `&[u8]` exists; classic validation/manager replacement remains P4-109 | 🔶 |

### Compression Setup

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_set_defaults(cinfo)` | Set default compression params | Automatic in `compress()`; classic public-table setup remains P4-85 | 🔶 |
| `jpeg_set_colorspace(cinfo, cs)` | Set JPEG colorspace | `Encoder::colorspace()`; classic scanline forwarding remains P4-93 | 🔶 |
| `jpeg_default_colorspace(cinfo)` | Reset to default colorspace | `Encoder::reset_colorspace()`; classic scanline forwarding remains P4-93 | 🔶 |
| `jpeg_set_quality(cinfo, quality, force_baseline)` | Set quality factor | `quality: u8` parameter + `Encoder::force_baseline()`; classic public-table/16-bit DQT semantics remain P4-85 | 🔶 |
| `jpeg_set_linear_quality(cinfo, scale, force_baseline)` | Set linear quality scaling | `Encoder::linear_quality()`; classic scanline table wiring is P4-85 | 🔶 |
| `jpeg_default_qtables(cinfo, force_baseline)` | Reset quant tables | `Encoder::reset_quant_tables(force_baseline)`; classic per-slot scales/table setup remain P4-85 | 🔶 |
| `jpeg_add_quant_table(cinfo, which, table, scale, force_baseline)` | Add custom quant table | `Encoder::quant_table()`; classic scanline table wiring/`force_baseline` semantics are P4-85 | 🔶 |
| `jpeg_quality_scaling(quality)` | Convert quality to scale factor | `quality_scaling()` | ✅ |
| `jpeg_enable_lossless(cinfo, psv, pt)` | Enable lossless mode | Native builder exists; classic validation/public-state semantics remain P4-107 | 🔶 |
| `jpeg_simple_progression(cinfo)` | Set standard progressive scan script | Native progressive script exists; classic public-script setup remains P4-91 | 🔶 |
| `jpeg_suppress_tables(cinfo, suppress)` | Control table output | `Encoder::suppress_tables(bool)`; classic cinfo sent/suppression state remains P4-87 | 🔶 |
| `jpeg_alloc_quant_table(cinfo)` | Allocate quant table | N/A (Rust arrays) | N/A |
| `jpeg_alloc_huff_table(cinfo)` | Allocate Huffman table | N/A (Rust structs) | N/A |

### Compression Processing

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_start_compress(cinfo, write_all_tables)` | Begin compression | `ScanlineEncoder::new()`; classic `write_all_tables` remains P4-87 | 🔶 |
| `jpeg_write_scanlines(cinfo, scanlines, num_lines)` | Write scanline rows | `ScanlineEncoder::write_scanlines()`; residual option gaps are P4-84..P4-93 | 🔶 |
| `jpeg12_write_scanlines(...)` | Write 12-bit scanlines | Native 12-bit encode exists; classic finish dispatch remains P4-94 | 🔶 |
| `jpeg16_write_scanlines(...)` | Write 16-bit scanlines | Native 16-bit lossless encode exists; classic finish dispatch remains P4-94 | 🔶 |
| `jpeg_finish_compress(cinfo)` | Finalize compression | Native finish exists; classic incomplete-input/state/error semantics remain P4-100/P4-106 | 🔶 |
| `jpeg_calc_jpeg_dimensions(cinfo)` | Compute compression-side JPEG dimensions; no compression scaling | `calc_jpeg_dimensions()` | ✅ |
| `jpeg_write_raw_data(cinfo, data, num_lines)` | Write raw downsampled data | Default `compress_raw()` path works; full classic options remain P4-95 | 🔶 |
| `jpeg12_write_raw_data(...)` | Write 12-bit raw data | Default `compress_raw_12()` path works; full classic options remain P4-95 | 🔶 |

### Marker Writing

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_write_marker(cinfo, marker, data, len)` | Write arbitrary marker | Native writer exists; classic timing/state validation remains P4-105 | 🔶 |
| `jpeg_write_m_header(cinfo, marker, len)` | Begin streaming marker write | Native writer exists; classic declared-length/state semantics remain P4-105 | 🔶 |
| `jpeg_write_m_byte(cinfo, val)` | Write one byte of marker data | Native writer exists; classic declared-length/state semantics remain P4-105 | 🔶 |
| `jpeg_write_tables(cinfo)` | Write tables-only datastream | Native tables-only output exists; classic installed-table/sent-state semantics remain P4-87 | 🔶 |
| `jpeg_write_icc_profile(cinfo, data, len)` | Write ICC profile | `marker_writer::write_app2_icc()` | 🔶 |

### Decompression

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_header(cinfo, require_image)` | Parse headers | Native parser works; classic metadata/state/tables remain P4-99/P4-101/P4-104 | 🔶 |
| `jpeg_start_decompress(cinfo)` | Begin decompression | Basic `ScanlineDecoder` path works; classic option dispatch remains P4-96/P4-99 | 🔶 |
| `jpeg_read_scanlines(cinfo, scanlines, max_lines)` | Read scanline rows | Basic rows work; classic quantization/options remain P4-96/P4-99 | 🔶 |
| `jpeg12_read_scanlines(...)` | Read 12-bit scanlines | Native precision decode exists; classic lifecycle/options remain P4-98 | 🔶 |
| `jpeg16_read_scanlines(...)` | Read 16-bit scanlines | Native precision decode exists; classic lifecycle/options remain P4-98 | 🔶 |
| `jpeg_skip_scanlines(cinfo, num_lines)` | Skip rows during decode | `ScanlineDecoder::skip_scanlines()`, `StreamingDecoder::skip_scanlines()` (C-matching clamp, issue #383) | ✅ |
| `jpeg12_skip_scanlines(...)` | Skip 12-bit scanlines | Private offset support exists; immediate-after-start behavior remains P4-98 | 🔶 |
| `jpeg_crop_scanline(cinfo, &xoffset, &width)` | Scanline-level crop | Native exact crop exists; classic iMCU alignment/state semantics remain P4-103 | 🔶 |
| `jpeg12_crop_scanline(...)` | 12-bit crop | Private crop support exists; immediate-after-start/output proof remains P4-98 | 🔶 |
| `jpeg_finish_decompress(cinfo)` | Finalize decompression | Native finish exists; classic lifecycle/suspension/error semantics remain P4-100/P4-104 | 🔶 |
| `jpeg_read_raw_data(cinfo, data, max_lines)` | Read raw downsampled data | Native raw decode exists; classic options/state/error semantics remain P4-102 | 🔶 |
| `jpeg12_read_raw_data(...)` | Read 12-bit raw data | Native raw decode exists; classic options/state/error semantics remain P4-102 | 🔶 |

### Buffered Image Mode (Progressive Output)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_has_multiple_scans(cinfo)` | Check if progressive/multi-scan | Native progressive query exists; classic sequential multi-scan/state semantics remain P4-114 | 🔶 |
| `jpeg_start_output(cinfo, scan_number)` | Begin output for specific scan | Native output exists; classic input-pull/state behavior remains P4-26/P4-104 | 🔶 |
| `jpeg_finish_output(cinfo)` | Finish scan output | Native finish exists; classic input-pull/state behavior remains P4-26/P4-104 | 🔶 |
| `jpeg_input_complete(cinfo)` | Check if all input consumed | Native query exists; deeper streaming/state fidelity remains P4-26/P4-104 | 🔶 |
| `jpeg_consume_input(cinfo)` | Process more input data | Suspension core works; deeper streaming/state fidelity remains P4-13/P4-26/P4-104 | 🔶 |
| `jpeg_new_colormap(cinfo)` | Update colormap after quant change | Native `requantize()` exists; classic color quantization remains P4-96 | 🔶 |

### Output Dimensions

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_calc_output_dimensions(cinfo)` | Compute scaled output size | Calculation exists; actual classic decode does not honor it (P4-99) | 🔶 |
| `jpeg_core_output_dimensions(cinfo)` | Core dimension calculation | Helper exists; odd-size public component geometry remains P4-99 | 🔶 |

### Marker Management

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_save_markers(cinfo, marker_code, length_limit)` | Enable marker saving | Native saving works; classic incremental marker-list pointer stability remains P4-26 | 🔶 |
| `jpeg_set_marker_processor(cinfo, marker_code, routine)` | Custom marker parser | Native callback API exists; classic callback invocation remains P4-112 | 🔶 |

### Coefficient Access

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_coefficients(cinfo)` | Read DCT coefficient arrays | `read_coefficients()` | ✅ |
| `jpeg_write_coefficients(cinfo, coef_arrays)` | Write coefficient arrays to JPEG | `write_coefficients()` | ✅ |
| `jpeg_copy_critical_parameters(src, dst)` | Copy quant/Huffman/colorspace between sessions | Native copy exists; classic source public-table publication remains P4-101 | 🔶 |

### Error / Sync

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_resync_to_restart(cinfo, desired)` | Resync to restart marker after error | Native strategy extension exists; classic default algorithm remains P4-97 | 🔶 |

### ICC Profile

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_icc_profile(cinfo, &data, &len)` | Read ICC profile from decoded image | Native ICC reassembly exists; classic saved-marker/header semantics remain P4-113 | 🔶 |

---

## TurboJPEG Legacy API (`turbojpeg.h` — backward compatibility)

> These are older API versions (1.0–2.0). We don't need to port them 1:1 since the TJ3 API
> supersedes them, but they're listed for completeness. Our Rust API covers the same functionality
> through the TJ3-equivalent functions above.

### Legacy Init / Destroy

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjInitCompress()` | `tj3Init(TJINIT_COMPRESS)` | Create compressor |
| `tjInitDecompress()` | `tj3Init(TJINIT_DECOMPRESS)` | Create decompressor |
| `tjInitTransform()` | `tj3Init(TJINIT_TRANSFORM)` | Create transformer |
| `tjDestroy(handle)` | `tj3Destroy(handle)` | Destroy handle |

### Legacy Compress

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjCompress(handle, src, w, pitch, h, pixelSize, dst, &size, subsamp, qual, flags)` | `tj3Compress8` | TJ 1.0 compress |
| `tjCompress2(handle, src, w, pitch, h, pf, &dst, &size, subsamp, qual, flags)` | `tj3Compress8` | TJ 1.2 compress |
| `tjCompressFromYUV(handle, src, w, align, h, subsamp, &dst, &size, qual, flags)` | `tj3CompressFromYUV8` | TJ 1.4 YUV compress |
| `tjCompressFromYUVPlanes(handle, planes, w, strides, h, subsamp, &dst, &size, qual, flags)` | `tj3CompressFromYUVPlanes8` | TJ 1.4 planar compress |

### Legacy Decompress

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjDecompress(handle, jpeg, size, dst, w, pitch, h, pixelSize, flags)` | `tj3Decompress8` | TJ 1.0 decompress |
| `tjDecompress2(handle, jpeg, size, dst, w, pitch, h, pf, flags)` | `tj3Decompress8` | TJ 1.2 decompress |
| `tjDecompressHeader(handle, jpeg, size, &w, &h)` | `tj3DecompressHeader` | TJ 1.0 header |
| `tjDecompressHeader2(handle, jpeg, size, &w, &h, &subsamp)` | `tj3DecompressHeader` | TJ 1.1 header |
| `tjDecompressHeader3(handle, jpeg, size, &w, &h, &subsamp, &cs)` | `tj3DecompressHeader` | TJ 1.4 header |
| `tjDecompressToYUV(handle, jpeg, size, dst, flags)` | `tj3DecompressToYUV8` | TJ 1.1 to YUV |
| `tjDecompressToYUV2(handle, jpeg, size, dst, w, align, h, flags)` | `tj3DecompressToYUV8` | TJ 1.4 to YUV |
| `tjDecompressToYUVPlanes(handle, jpeg, size, planes, w, strides, h, flags)` | `tj3DecompressToYUVPlanes8` | TJ 1.4 to planar |

### Legacy Encode YUV

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjEncodeYUV(handle, src, w, pitch, h, pixelSize, dst, subsamp, flags)` | `tj3EncodeYUV8` | TJ 1.1 encode YUV |
| `tjEncodeYUV2(handle, src, w, pitch, h, pf, dst, subsamp, flags)` | `tj3EncodeYUV8` | TJ 1.2 encode YUV |
| `tjEncodeYUV3(handle, src, w, pitch, h, pf, dst, align, subsamp, flags)` | `tj3EncodeYUV8` | TJ 1.4 encode YUV |
| `tjEncodeYUVPlanes(handle, src, w, pitch, h, pf, planes, strides, subsamp, flags)` | `tj3EncodeYUVPlanes8` | TJ 1.4 encode planar |

### Legacy Decode YUV

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjDecodeYUV(handle, src, align, subsamp, dst, w, pitch, h, pf, flags)` | `tj3DecodeYUV8` | TJ 1.4 decode YUV |
| `tjDecodeYUVPlanes(handle, planes, strides, subsamp, dst, w, pitch, h, pf, flags)` | `tj3DecodeYUVPlanes8` | TJ 1.4 decode planar |

### Legacy Transform

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjTransform(handle, jpeg, size, n, &dstBufs, &dstSizes, transforms, flags)` | `tj3Transform` | TJ 1.2 lossless transform |

### Legacy Buffer Size

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `TJBUFSIZE(w, h)` | `tj3JPEGBufSize` | TJ 1.0 macro |
| `TJBUFSIZEYUV(w, h, subsamp)` | `tj3YUVBufSize` | TJ 1.1 macro |
| `TJBUFSIZEYUV2(w, align, h, subsamp)` | `tj3YUVBufSize` | TJ 1.4 macro |
| `tjBufSize(w, h, subsamp)` | `tj3JPEGBufSize` | TJ 1.2 |
| `tjBufSizeYUV(w, h, subsamp)` | `tj3YUVBufSize` | TJ 1.2 |
| `tjBufSizeYUV2(w, align, h, subsamp)` | `tj3YUVBufSize` | TJ 1.4 |
| `tjPlaneSizeYUV(comp, w, stride, h, subsamp)` | `tj3YUVPlaneSize` | TJ 1.4 |
| `tjPlaneWidth(comp, w, subsamp)` | `tj3YUVPlaneWidth` | TJ 1.4 |
| `tjPlaneHeight(comp, h, subsamp)` | `tj3YUVPlaneHeight` | TJ 1.4 |

### Legacy Memory

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjAlloc(bytes)` | `tj3Alloc` | TJ 1.2 allocate |
| `tjFree(buffer)` | `tj3Free` | TJ 1.2 free |

### Legacy Error

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjGetErrorStr()` | `tj3GetErrorStr` | TJ 1.0 global error string |
| `tjGetErrorStr2(handle)` | `tj3GetErrorStr` | TJ 2.0 per-handle error |
| `tjGetErrorCode(handle)` | `tj3GetErrorCode` | TJ 2.0 error code |
| `tjGetScalingFactors(&count)` | `tj3GetScalingFactors` | TJ 1.2 scaling factors |

### Legacy Image I/O

| C Function | TJ3 Equivalent | Description |
|---|---|---|
| `tjLoadImage(filename, &w, align, &h, &pf, flags)` | `tj3LoadImage8` | TJ 2.0 load BMP/PPM |
| `tjSaveImage(filename, buf, w, pitch, h, pf, flags)` | `tj3SaveImage8` | TJ 2.0 save BMP/PPM |

### Legacy Flags (#define)

| Flag | TJ3 Equivalent | Value |
|---|---|---|
| `TJFLAG_BOTTOMUP` | `TJPARAM_BOTTOMUP` | 2 |
| `TJFLAG_FASTUPSAMPLE` | `TJPARAM_FASTUPSAMPLE` | 256 |
| `TJFLAG_NOREALLOC` | `TJPARAM_NOREALLOC` | 1024 |
| `TJFLAG_FASTDCT` | `TJPARAM_FASTDCT` | 2048 |
| `TJFLAG_ACCURATEDCT` | Default (ISLOW) | 4096 |
| `TJFLAG_STOPONWARNING` | `TJPARAM_STOPONWARNING` | 8192 |
| `TJFLAG_PROGRESSIVE` | `TJPARAM_PROGRESSIVE` | 16384 |
| `TJFLAG_LIMITSCANS` | `TJPARAM_SCANLIMIT` | 32768 |
| `TJFLAG_FORCEMMX` | Removed | 8 |
| `TJFLAG_FORCESSE` | Removed | 16 |
| `TJFLAG_FORCESSE2` | Removed | 32 |
| `TJFLAG_FORCESSE3` | Removed | 128 |

---

## Static Data (`turbojpeg.h`)

| Constant | Description | Rust | Status |
|---|---|---|---|
| `tjMCUWidth[7]` | iMCU width per subsampling | `Subsampling::mcu_width_blocks() * 8` | ✅ |
| `tjMCUHeight[7]` | iMCU height per subsampling | `Subsampling::mcu_height_blocks() * 8` | ✅ |
| `tjPixelSize[12]` | Bytes per pixel per format | `PixelFormat::bytes_per_pixel()` | ✅ |
| `tjRedOffset[12]` | Red channel offset per format | `PixelFormat::red_offset()` | ✅ |
| `tjGreenOffset[12]` | Green channel offset per format | `PixelFormat::green_offset()` | ✅ |
| `tjBlueOffset[12]` | Blue channel offset per format | `PixelFormat::blue_offset()` | ✅ |
| `tjAlphaOffset[12]` | Alpha channel offset per format | `PixelFormat::alpha_offset()` | ✅ |

---

## Structs (`turbojpeg.h`)

| Struct | Description | Rust | Status |
|---|---|---|---|
| `tjscalingfactor` | {num, denom} scaling ratio | `ScalingFactor` | ✅ |
| `tjregion` | {x, y, w, h} crop region | `CropRegion` | ✅ |
| `tjtransform` | {region, op, options, data, customFilter} | `TransformOptions` (all fields incl. `custom_filter`) | ✅ |

---

## Constants & Enums (`jpeglib.h`)

### DCT Method (`J_DCT_METHOD`)
| Value | Description | Rust | Status |
|---|---|---|---|
| `JDCT_ISLOW` | Accurate integer DCT | `DctMethod::IsLow` (default) | ✅ |
| `JDCT_IFAST` | Fast integer DCT (less accurate) | `DctMethod::IsFast`; classic scanline forwarding is P4-86 | 🔶 |
| `JDCT_FLOAT` | Floating-point DCT | `DctMethod::Float`; classic scanline forwarding is P4-86 | 🔶 |

### Dithering (`J_DITHER_MODE`)
| Value | Description | Rust | Status |
|---|---|---|---|
| `JDITHER_NONE` | No dithering | `DitherMode::None` | ✅ |
| `JDITHER_ORDERED` | Ordered dither | `DitherMode::Ordered` | ✅ |
| `JDITHER_FS` | Floyd-Steinberg error diffusion | `DitherMode::FloydSteinberg` | ✅ |

### Return Codes
| Value | Description | Rust | Status |
|---|---|---|---|
| `JPEG_SUSPENDED` (0) | Suspended, need more input | N/A (full-buffer + streaming API) | N/A |
| `JPEG_HEADER_OK` (1) | Valid image found | `Decoder::new()` / `ScanlineDecoder::new()` success | ✅ |
| `JPEG_HEADER_TABLES_ONLY` (2) | Tables-only datastream | `api::abbreviated::HeaderResult::TablesOnly(Box<TablesOnlyState>)` from `read_header()` | ✅ |
| `JPEG_REACHED_SOS` (1) | Start of new scan | Internal | ✅ |
| `JPEG_REACHED_EOI` (2) | End of image | Internal | ✅ |
| `JPEG_ROW_COMPLETED` (3) | Completed one iMCU row | Internal (scanline API) | ✅ |
| `JPEG_SCAN_COMPLETED` (4) | Completed last row of scan | Internal (progressive API) | ✅ |

### Marker Codes
| Value | Description | Rust | Status |
|---|---|---|---|
| `JPEG_RST0` (0xD0) | Restart marker base | Handled in decode + encode | ✅ |
| `JPEG_EOI` (0xD9) | End of image | Handled | ✅ |
| `JPEG_APP0` (0xE0) | APP0 (JFIF) | Read + write | ✅ |
| `JPEG_COM` (0xFE) | Comment marker | Read + write | ✅ |

### Size Constants
| Value | Description | Rust | Status |
|---|---|---|---|
| `DCTSIZE` (8) | Block size | Hardcoded | ✅ |
| `DCTSIZE2` (64) | Block size squared | Hardcoded | ✅ |
| `NUM_QUANT_TBLS` (4) | Max quant tables | 4 in `JpegMetadata` | ✅ |
| `NUM_HUFF_TBLS` (4) | Max Huffman tables | 4 in `JpegMetadata` | ✅ |
| `NUM_ARITH_TBLS` (16) | Max arithmetic tables | 16 in `ArithDecoder` / `ArithEncoder` / DAC writer/parser per spec F.2.4.3 | ✅ |
| `MAX_COMPS_IN_SCAN` (4) | Max components per scan | Handled | ✅ |
| `MAX_SAMP_FACTOR` (4) | Max sampling factor | Handled | ✅ |
| `C_MAX_BLOCKS_IN_MCU` (10) | Max blocks in compressor MCU | Handled | ✅ |
| `D_MAX_BLOCKS_IN_MCU` (10) | Max blocks in decompressor MCU | Handled | ✅ |
| `JPOOL_PERMANENT` (0) | Permanent memory pool | N/A (Rust allocator) | N/A |
| `JPOOL_IMAGE` (1) | Image-scoped memory pool | N/A (Rust allocator) | N/A |

---

## Structs (`jpeglib.h`)

| Struct | Description | Rust | Status |
|---|---|---|---|
| `JQUANT_TBL` | Quantization table (64 values + sent_table) | Internal `[u16; 64]` arrays | ✅ |
| `JHUFF_TBL` | Huffman table (bits[17] + huffval[256]) | `HuffmanTable` / `HuffTable` | ✅ |
| `jpeg_component_info` | Per-component metadata | `ComponentInfo` | ✅ |
| `jpeg_scan_info` | Scan script entry (components, Ss/Se/Ah/Al) | `ScanScript` / `ScanInfo`; classic scanline wiring remains P4-91 | 🔶 |
| `jpeg_marker_struct` | Saved marker (code, length, data, next) | Native markers exist; classic incremental pointer stability remains P4-26 | 🔶 |
| `jpeg_common_struct` | Common fields (err, mem, progress) | Native equivalents exist; classic error/state/progress contracts remain P4-100/P4-104/P4-111 | 🔶 |
| `jpeg_compress_struct` | Full compression state (~50 fields) | `Encoder` / `ScanlineEncoder`; residual classic option/state gaps are P4-84..P4-111 | 🔶 |
| `jpeg_decompress_struct` | Full decompression state (~60 fields) | `Decoder` / `ScanlineDecoder`; residual classic state/options gaps are P4-96..P4-114 | 🔶 |
| `jpeg_error_mgr` | Error handler (5 callbacks + state) | `ErrorHandler` trait (3 callbacks) | 🔶 |
| `jpeg_progress_mgr` | Progress callback + counters | Native `ProgressListener` exists; classic callback/counters remain P4-111 | 🔶 |
| `jpeg_destination_mgr` | Output stream (buffer + 3 callbacks) | Built-in mem/stdio managers are full-contract (P4-108 closed); an application-supplied manager returning `FALSE` from `empty_output_buffer` still gets `JERR_CANT_SUSPEND` instead of suspending (deferred-encode shim, P3-5) | 🔶 |
| `jpeg_source_mgr` | Input stream (buffer + 5 callbacks) | Native readers exist; classic setup/stdio semantics remain P4-109 | 🔶 |
| `jpeg_memory_mgr` | Memory allocator (12 methods) | N/A (Rust allocator) | N/A |

---

## Remaining Gaps / Partial Mappings

These are the highest-signal C API surfaces that still lack end-to-end public parity, even when adjacent Rust APIs exist:

| C Function / Surface | Status | Notes |
|---|---|---|
| `tj3LoadImage8()` / `tj3SaveImage8()` PNG | ✅ | BMP/PPM/PGM implemented; PNG added behind `--features png` (default off). Dispatch by 8-byte magic on load, by `.png` extension on save. 8-bit RGB/RGBA/Grayscale only; 16-bit and indexed-colour return `Unsupported`. |
| `tj3GetErrorStr()` / `tj3GetErrorCode()` | 🔶 | Rust uses `Result` / `JpegError`, not C-style per-handle getters (C ABI shim in `libjpeg-turbo-rs-capi` exposes both for FFI callers) |
| `tj3Alloc()` / `tj3Free()` | 🔶 | N/A — Rust ownership replaces C allocator API (FFI-facing aliases exist in `libjpeg-turbo-rs-capi`) |
| `jpeg_write_icc_profile()` | 🔶 | Native helper and classic export exist; marker state/error contracts remain P4-105/P4-100. |
| `jpeg_create_(de)compress()` + full `jpeg_*` state-machine ABI | 🔶 | The v8 export/layout surface is broad, but create guards, ownership, public state/options, lifecycle, callbacks, and error propagation remain P4-84..P4-114. See `docs/LAST_MILE.md`; symbol presence is not behavioral parity. |
| B9-4 / B9-5 — stock-tool/tjunittest link and behavior gates | ✅ | B9-4 covers the reference image corpus for the operations it runs; B9-5 reaches completion. These are selected consumer gates, not a v6b/v8 general drop-in claim. |
