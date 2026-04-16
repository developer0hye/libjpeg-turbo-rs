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
| `NOREALLOC` | Disable output buffer realloc | `compress_into()` (separate API, not `TjHandle`) | 🔶 |
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
| `SAVEMARKERS` | Marker preservation level 0-4 | `Decoder::save_markers()` / `MarkerSaveConfig` | 🔶 |

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
| `tj3YUVPlaneWidth(comp, w, subsamp)` | YUV plane width | `yuv_plane_width()` | ✅ |
| `tj3YUVPlaneHeight(comp, h, subsamp)` | YUV plane height | `yuv_plane_height()` | ✅ |

### ICC Profile

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3SetICCProfile(handle, buf, size)` | Set ICC for encoding | `TjHandle::set_icc_profile()` / `Encoder::icc_profile()` | ✅ |
| `tj3GetICCProfile(handle, &buf, &size)` | Get ICC after decode | `Image.icc_profile()` | 🔶 |

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
| `tj3Decompress8(handle, jpeg, size, dst, pitch, pf)` | Decompress JPEG to 8-bit pixels | `decompress()`, `decompress_to()` | ✅ |
| `tj3Decompress12(handle, jpeg, size, dst, pitch, pf)` | Decompress to 12-bit | `TjHandle::decompress_12bit()` / `decompress_12bit()` | ✅ |
| `tj3Decompress16(handle, jpeg, size, dst, pitch, pf)` | Decompress to 16-bit | `TjHandle::decompress_16bit()` / `decompress_16bit()` | ✅ |

### Decompression to YUV

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecompressToYUV8(handle, jpeg, size, dst, align)` | JPEG → packed YUV | `yuv::decompress_to_yuv()` | ✅ |
| `tj3DecompressToYUVPlanes8(handle, jpeg, size, planes, strides)` | JPEG → planar YUV | `yuv::decompress_to_yuv_planes()` | ✅ |

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
| `tj3LoadImage12(...)` / `tj3SaveImage12(...)` | 12-bit file I/O | — | ❌ |
| `tj3LoadImage16(...)` / `tj3SaveImage16(...)` | 16-bit file I/O | — | ❌ |

---

## libjpeg API (`jpeglib.h`)

### Initialization & Destruction

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_std_error(err)` | Create default error manager | `JpegError` enum | ✅ |
| `jpeg_create_compress(cinfo)` | Create compression struct | `Encoder` / `ScanlineEncoder` | ✅ |
| `jpeg_create_decompress(cinfo)` | Create decompression struct | `Decoder::new()` / `ScanlineDecoder::new()` | ✅ |
| `jpeg_destroy_compress(cinfo)` | Destroy compressor | RAII / `Drop` | ✅ |
| `jpeg_destroy_decompress(cinfo)` | Destroy decompressor | RAII / `Drop` | ✅ |
| `jpeg_abort_compress(cinfo)` | Abort compression | N/A (RAII) | N/A |
| `jpeg_abort_decompress(cinfo)` | Abort decompression | N/A (RAII) | N/A |
| `jpeg_abort(cinfo)` | Abort any operation | N/A (RAII) | N/A |
| `jpeg_destroy(cinfo)` | Destroy any handle | N/A (RAII) | N/A |

### Data Source / Destination

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_stdio_dest(cinfo, file)` | Output to FILE* | `stream::compress_to_file` / `stream::compress_to_writer` | ✅ |
| `jpeg_stdio_src(cinfo, file)` | Input from FILE* | `stream::decompress_from_file` / `stream::decompress_from_reader` | ✅ |
| `jpeg_mem_dest(cinfo, &outbuf, &outsize)` | Output to memory buffer | `Vec<u8>` output (native) | ✅ |
| `jpeg_mem_src(cinfo, inbuf, insize)` | Input from memory buffer | `&[u8]` input (native) | ✅ |

### Compression Setup

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_set_defaults(cinfo)` | Set default compression params | Automatic in `compress()` | ✅ |
| `jpeg_set_colorspace(cinfo, cs)` | Set JPEG colorspace | `Encoder::colorspace()` | ✅ |
| `jpeg_default_colorspace(cinfo)` | Reset to default colorspace | — | ❌ |
| `jpeg_set_quality(cinfo, quality, force_baseline)` | Set quality factor | `quality: u8` parameter + `Encoder::force_baseline()` | ✅ |
| `jpeg_set_linear_quality(cinfo, scale, force_baseline)` | Set linear quality scaling | `Encoder::linear_quality()` | ✅ |
| `jpeg_default_qtables(cinfo, force_baseline)` | Reset quant tables | — | ❌ |
| `jpeg_add_quant_table(cinfo, which, table, scale, force_baseline)` | Add custom quant table | `Encoder::quant_table()` | ✅ |
| `jpeg_quality_scaling(quality)` | Convert quality to scale factor | `quality_scaling()` | ✅ |
| `jpeg_enable_lossless(cinfo, psv, pt)` | Enable lossless mode | `Encoder::lossless_predictor()` + `Encoder::lossless_point_transform()` | ✅ |
| `jpeg_simple_progression(cinfo)` | Set standard progressive scan script | Used internally in `compress_progressive()` | ✅ |
| `jpeg_suppress_tables(cinfo, suppress)` | Control table output | — | ❌ |
| `jpeg_alloc_quant_table(cinfo)` | Allocate quant table | N/A (Rust arrays) | N/A |
| `jpeg_alloc_huff_table(cinfo)` | Allocate Huffman table | N/A (Rust structs) | N/A |

### Compression Processing

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_start_compress(cinfo, write_all_tables)` | Begin compression | `ScanlineEncoder::new()` | ✅ |
| `jpeg_write_scanlines(cinfo, scanlines, num_lines)` | Write scanline rows | `ScanlineEncoder::write_scanlines()` | ✅ |
| `jpeg12_write_scanlines(...)` | Write 12-bit scanlines | `write_scanlines_12()` | ✅ |
| `jpeg16_write_scanlines(...)` | Write 16-bit scanlines | `write_scanlines_16()` | ✅ |
| `jpeg_finish_compress(cinfo)` | Finalize compression | `ScanlineEncoder::finish()` | ✅ |
| `jpeg_calc_jpeg_dimensions(cinfo)` | Compute output dimensions | `calc_jpeg_dimensions()` | ✅ |
| `jpeg_write_raw_data(cinfo, data, num_lines)` | Write raw downsampled data | `compress_raw()` | ✅ |
| `jpeg12_write_raw_data(...)` | Write 12-bit raw data | — | ❌ |

### Marker Writing

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_write_marker(cinfo, marker, data, len)` | Write arbitrary marker | `marker_writer::write_marker()` | ✅ |
| `jpeg_write_m_header(cinfo, marker, len)` | Begin streaming marker write | `MarkerStreamWriter` | ✅ |
| `jpeg_write_m_byte(cinfo, val)` | Write one byte of marker data | `MarkerStreamWriter` | ✅ |
| `jpeg_write_tables(cinfo)` | Write tables-only datastream | — | ❌ |
| `jpeg_write_icc_profile(cinfo, data, len)` | Write ICC profile | `marker_writer::write_app2_icc()` | 🔶 |

### Decompression

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_header(cinfo, require_image)` | Parse headers | `Decoder::new()` / `ScanlineDecoder::new()` | ✅ |
| `jpeg_start_decompress(cinfo)` | Begin decompression | `ScanlineDecoder::new()` | ✅ |
| `jpeg_read_scanlines(cinfo, scanlines, max_lines)` | Read scanline rows | `ScanlineDecoder::read_scanlines()` | ✅ |
| `jpeg12_read_scanlines(...)` | Read 12-bit scanlines | `read_scanlines_12()` | ✅ |
| `jpeg16_read_scanlines(...)` | Read 16-bit scanlines | `read_scanlines_16()` | ✅ |
| `jpeg_skip_scanlines(cinfo, num_lines)` | Skip rows during decode | `ScanlineDecoder::skip_scanlines()` | ✅ |
| `jpeg12_skip_scanlines(...)` | Skip 12-bit scanlines | `read_scanlines_12()` (skip via offset) | ✅ |
| `jpeg_crop_scanline(cinfo, &xoffset, &width)` | Scanline-level crop | `ScanlineDecoder::set_crop_x()` | ✅ |
| `jpeg12_crop_scanline(...)` | 12-bit crop | `read_scanlines_12()` (crop support) | ✅ |
| `jpeg_finish_decompress(cinfo)` | Finalize decompression | `ScanlineDecoder::finish()` | ✅ |
| `jpeg_read_raw_data(cinfo, data, max_lines)` | Read raw downsampled data | `decompress_raw()` | ✅ |
| `jpeg12_read_raw_data(...)` | Read 12-bit raw data | — | ❌ |

### Buffered Image Mode (Progressive Output)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_has_multiple_scans(cinfo)` | Check if progressive/multi-scan | `ProgressiveDecoder::has_multiple_scans()` | ✅ |
| `jpeg_start_output(cinfo, scan_number)` | Begin output for specific scan | `ProgressiveDecoder::output()` | ✅ |
| `jpeg_finish_output(cinfo)` | Finish scan output | `ProgressiveDecoder::finish()` | ✅ |
| `jpeg_input_complete(cinfo)` | Check if all input consumed | `ProgressiveDecoder::input_complete()` | ✅ |
| `jpeg_consume_input(cinfo)` | Process more input data | `ProgressiveDecoder::consume_input()` | ✅ |
| `jpeg_new_colormap(cinfo)` | Update colormap after quant change | `requantize()` | ✅ |

### Output Dimensions

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_calc_output_dimensions(cinfo)` | Compute scaled output size | `calc_output_dimensions()` | ✅ |
| `jpeg_core_output_dimensions(cinfo)` | Core dimension calculation | `calc_jpeg_dimensions()` | ✅ |

### Marker Management

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_save_markers(cinfo, marker_code, length_limit)` | Enable marker saving | `Decoder::save_markers()` | ✅ |
| `jpeg_set_marker_processor(cinfo, marker_code, routine)` | Custom marker parser | `Decoder::set_marker_processor()` | ✅ |

### Coefficient Access

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_coefficients(cinfo)` | Read DCT coefficient arrays | `read_coefficients()` | ✅ |
| `jpeg_write_coefficients(cinfo, coef_arrays)` | Write coefficient arrays to JPEG | `write_coefficients()` | ✅ |
| `jpeg_copy_critical_parameters(src, dst)` | Copy quant/Huffman/colorspace between sessions | `copy_critical_parameters()` | ✅ |

### Error / Sync

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_resync_to_restart(cinfo, desired)` | Resync to restart marker after error | Internal in decoder | 🔶 |

### ICC Profile

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_icc_profile(cinfo, &data, &len)` | Read ICC profile from decoded image | `Image.icc_profile()` | ✅ |

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
| `JDCT_IFAST` | Fast integer DCT (less accurate) | `DctMethod::IsFast` | ✅ |
| `JDCT_FLOAT` | Floating-point DCT | `DctMethod::Float` | ✅ |

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
| `JPEG_HEADER_TABLES_ONLY` (2) | Tables-only datastream | — | ❌ |
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
| `NUM_ARITH_TBLS` (16) | Max arithmetic tables | 4 in `ArithDecoder` | 🔶 |
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
| `jpeg_scan_info` | Scan script entry (components, Ss/Se/Ah/Al) | `ScanScript` / `ScanInfo` | ✅ |
| `jpeg_marker_struct` | Saved marker (code, length, data, next) | `Image.markers()` / `Image.saved_markers` | ✅ |
| `jpeg_common_struct` | Common fields (err, mem, progress) | Spread across `Decoder` / `Encoder` | ✅ |
| `jpeg_compress_struct` | Full compression state (~50 fields) | `Encoder` / `ScanlineEncoder` | ✅ |
| `jpeg_decompress_struct` | Full decompression state (~60 fields) | `Decoder` / `ScanlineDecoder` | ✅ |
| `jpeg_error_mgr` | Error handler (5 callbacks + state) | `ErrorHandler` trait (3 callbacks) | 🔶 |
| `jpeg_progress_mgr` | Progress callback + counters | `ProgressListener` trait | ✅ |
| `jpeg_destination_mgr` | Output stream (buffer + 3 callbacks) | `stream::compress_to_writer<W: Write>` | ✅ |
| `jpeg_source_mgr` | Input stream (buffer + 5 callbacks) | `stream::decompress_from_reader<R: Read>` | ✅ |
| `jpeg_memory_mgr` | Memory allocator (12 methods) | N/A (Rust allocator) | N/A |

---

## Remaining Gaps / Partial Mappings

These are the highest-signal C API surfaces that still lack end-to-end public parity, even when adjacent Rust APIs exist:

| C Function / Surface | Status | Notes |
|---|---|---|
| `TJPARAM_NOREALLOC` on `TjHandle` | 🔶 | N/A for Rust `Vec<u8>` — stored for API compatibility, no behavioral effect |
| `tj3LoadImage8()` / `tj3SaveImage8()` full parity | 🔶 | Rust only covers BMP/PPM/PGM 8-bit helpers, not PNG or the full handle-driven semantics from C |
| `tj3LoadImage12/16()` / `tj3SaveImage12/16()` | ❌ | Missing |
| `tj3GetErrorStr()` / `tj3GetErrorCode()` | 🔶 | Rust uses `Result` / `JpegError`, not C-style per-handle getters |
| `tj3Alloc()` / `tj3Free()` dedicated allocator API | 🔶 | Idiomatic Rust ownership exists, but not a TurboJPEG allocator entry point |
| `jpeg_write_icc_profile()` | 🔶 | Low-level helper exists, but no libjpeg-style public wrapper around a compression state object |
| `jpeg_resync_to_restart()` | 🔶 | Internal behavior only, no public hook |
| `jpeg_default_colorspace()` | ❌ | Missing |
| `jpeg_default_qtables()` | ❌ | Missing |
| `jpeg_suppress_tables()` | ❌ | Missing |
| `jpeg_write_tables()` | ❌ | Missing |
| `jpeg12_write_raw_data()` | ❌ | Missing |
| `jpeg12_read_raw_data()` | ❌ | Missing |
| `JPEG_HEADER_TABLES_ONLY` | ❌ | Tables-only datastream detection missing |
