# libjpeg-turbo C API → Rust Mapping Reference

> Every public C function from `turbojpeg.h` and `jpeglib.h` with description and Rust equivalent.
> ✅ = implemented, ❌ = not yet, 🔶 = partial

---

## TurboJPEG 3.0 API (`turbojpeg.h`)

### Handle Lifecycle

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Init(initType)` | Create compress/decompress/transform handle | No handle pattern; direct function calls | ❌ |
| `tj3Destroy(handle)` | Destroy handle | N/A (RAII) | ❌ |

### Parameter Get/Set

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Set(handle, param, value)` | Set integer parameter | Function arguments | ❌ |
| `tj3Get(handle, param)` | Get integer parameter | Struct fields | ❌ |

**All 26 TJPARAM values:**

| TJPARAM | Description | Rust | Status |
|---|---|---|---|
| `STOPONWARNING` | Treat warnings as fatal | — | ❌ |
| `BOTTOMUP` | Bottom-up row order | — | ❌ |
| `NOREALLOC` | Disable output buffer realloc | N/A (Vec handles this) | ❌ |
| `QUALITY` | Lossy quality 1-100 | `quality: u8` param | ✅ |
| `SUBSAMP` | Chroma subsampling | `subsampling: Subsampling` param | ✅ |
| `JPEGWIDTH` | JPEG image width (read-only) | `Image.width` | ✅ |
| `JPEGHEIGHT` | JPEG image height (read-only) | `Image.height` | ✅ |
| `PRECISION` | Sample precision 2-16 bits | Hardcoded 8-bit | 🔶 |
| `COLORSPACE` | JPEG colorspace | Auto-detected | 🔶 |
| `FASTUPSAMPLE` | Nearest-neighbor upsampling | Always fancy | ❌ |
| `FASTDCT` | Fast DCT/IDCT algorithm | Always ISLOW | ❌ |
| `OPTIMIZE` | Optimized Huffman tables | `compress_optimized()` | ✅ |
| `PROGRESSIVE` | Progressive JPEG mode | `compress_progressive()` | ✅ |
| `SCANLIMIT` | Max progressive scans | — | ❌ |
| `ARITHMETIC` | Arithmetic entropy coding | `compress_arithmetic()` | ✅ |
| `LOSSLESS` | Lossless JPEG mode | `compress_lossless()` | ✅ |
| `LOSSLESSPSV` | Lossless predictor 1-7 | Hardcoded predictor 1 | 🔶 |
| `LOSSLESSPT` | Lossless point transform 0-15 | Hardcoded pt=0 | 🔶 |
| `RESTARTBLOCKS` | Restart interval (MCU blocks) | — | ❌ |
| `RESTARTROWS` | Restart interval (MCU rows) | — | ❌ |
| `XDENSITY` | Horizontal pixel density | Hardcoded 72 | 🔶 |
| `YDENSITY` | Vertical pixel density | Hardcoded 72 | 🔶 |
| `DENSITYUNITS` | 0=unknown, 1=ppi, 2=ppcm | Hardcoded 1 (ppi) | 🔶 |
| `MAXMEMORY` | Memory limit | — | ❌ |
| `MAXPIXELS` | Image size limit | — | ❌ |
| `SAVEMARKERS` | Marker preservation level 0-4 | ICC/EXIF/Adobe only | 🔶 |

### Memory

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Alloc(bytes)` | Allocate buffer | `Vec::with_capacity()` | ✅ |
| `tj3Free(buffer)` | Free buffer | `drop()` / RAII | ✅ |

### Buffer Size Calculation

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3JPEGBufSize(w, h, subsamp)` | Worst-case JPEG output size | — | ❌ |
| `tj3YUVBufSize(w, align, h, subsamp)` | Total YUV buffer size | — | ❌ |
| `tj3YUVPlaneSize(comp, w, stride, h, subsamp)` | Single YUV plane size | — | ❌ |
| `tj3YUVPlaneWidth(comp, w, subsamp)` | YUV plane width | — | ❌ |
| `tj3YUVPlaneHeight(comp, h, subsamp)` | YUV plane height | — | ❌ |

### ICC Profile

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3SetICCProfile(handle, buf, size)` | Set ICC for encoding | `compress_with_metadata(icc_profile: Some(&data))` | ✅ |
| `tj3GetICCProfile(handle, &buf, &size)` | Get ICC after decode | `Image.icc_profile()` | ✅ |

### Compression (8-bit)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Compress8(handle, src, w, pitch, h, pf, &dst, &size)` | Compress 8-bit pixels to JPEG | `compress()`, `compress_optimized()`, etc. | ✅ |
| `tj3Compress12(handle, src, w, pitch, h, pf, &dst, &size)` | Compress 12-bit pixels | — | ❌ |
| `tj3Compress16(handle, src, w, pitch, h, pf, &dst, &size)` | Compress 16-bit pixels (lossless only) | — | ❌ |

### Compression from YUV

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3CompressFromYUV8(handle, src, w, align, h, &dst, &size)` | Compress packed YUV to JPEG | — | ❌ |
| `tj3CompressFromYUVPlanes8(handle, planes, w, strides, h, &dst, &size)` | Compress planar YUV to JPEG | — | ❌ |

### Color Encode (RGB → YUV, no JPEG)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3EncodeYUV8(handle, src, w, pitch, h, pf, dst, align)` | RGB → packed YUV | — | ❌ |
| `tj3EncodeYUVPlanes8(handle, src, w, pitch, h, pf, planes, strides)` | RGB → planar YUV | — | ❌ |

### Decompression Header

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecompressHeader(handle, jpeg, size)` | Parse JPEG headers, populate params | `Decoder::new()` / `StreamingDecoder::new()` | ✅ |

### Scaling & Cropping

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3GetScalingFactors(&count)` | Get list of supported scaling factors | `ScalingFactor` struct | ✅ |
| `tj3SetScalingFactor(handle, sf)` | Set output scaling | `Decoder::set_scale()` | ✅ |
| `tj3SetCroppingRegion(handle, region)` | Set crop region | `Decoder::set_crop_region()` | ✅ |

### Decompression (8-bit)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Decompress8(handle, jpeg, size, dst, pitch, pf)` | Decompress JPEG to 8-bit pixels | `decompress()`, `decompress_to()` | ✅ |
| `tj3Decompress12(handle, jpeg, size, dst, pitch, pf)` | Decompress to 12-bit | — | ❌ |
| `tj3Decompress16(handle, jpeg, size, dst, pitch, pf)` | Decompress to 16-bit | — | ❌ |

### Decompression to YUV

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecompressToYUV8(handle, jpeg, size, dst, align)` | JPEG → packed YUV | — | ❌ |
| `tj3DecompressToYUVPlanes8(handle, jpeg, size, planes, strides)` | JPEG → planar YUV | — | ❌ |

### Color Decode (YUV → RGB, no JPEG)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3DecodeYUV8(handle, src, align, dst, w, pitch, h, pf)` | Packed YUV → RGB | — | ❌ |
| `tj3DecodeYUVPlanes8(handle, planes, strides, dst, w, pitch, h, pf)` | Planar YUV → RGB | — | ❌ |

### Lossless Transform

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3Transform(handle, jpeg, size, n, &dstBufs, &dstSizes, transforms)` | Lossless transform with options | `transform_jpeg()` (basic ops only) | 🔶 |
| `tj3TransformBufSize(handle, transform)` | Estimate output buffer size | — | ❌ |

### Error Handling

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3GetErrorStr(handle)` | Get error message string | `JpegError` Display impl | ✅ |
| `tj3GetErrorCode(handle)` | Get TJERR_WARNING or TJERR_FATAL | `Result<T, JpegError>` | ✅ |

### Image File I/O

| C Function | Description | Rust | Status |
|---|---|---|---|
| `tj3LoadImage8(handle, filename, &w, align, &h, &pf)` | Load BMP/PPM to 8-bit buffer | `load_image` / `load_image_from_bytes` | ✅ |
| `tj3SaveImage8(handle, filename, buf, w, pitch, h, pf)` | Save 8-bit buffer to BMP/PPM | `save_bmp` / `save_ppm` | ✅ |
| `tj3LoadImage12(...)` / `tj3SaveImage12(...)` | 12-bit file I/O | — | ❌ |
| `tj3LoadImage16(...)` / `tj3SaveImage16(...)` | 16-bit file I/O | — | ❌ |

---

## libjpeg API (`jpeglib.h`)

### Initialization & Destruction

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_std_error(err)` | Create default error manager | `JpegError` enum | ✅ |
| `jpeg_create_compress(cinfo)` | Create compression struct | Direct function call | ✅ |
| `jpeg_create_decompress(cinfo)` | Create decompression struct | `Decoder::new()` | ✅ |
| `jpeg_destroy_compress(cinfo)` | Destroy compressor | RAII / drop | ✅ |
| `jpeg_destroy_decompress(cinfo)` | Destroy decompressor | RAII / drop | ✅ |
| `jpeg_abort_compress(cinfo)` | Abort compression | — | ❌ |
| `jpeg_abort_decompress(cinfo)` | Abort decompression | — | ❌ |
| `jpeg_abort(cinfo)` | Abort any operation | — | ❌ |
| `jpeg_destroy(cinfo)` | Destroy any handle | — | ❌ |

### Data Source / Destination

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_stdio_dest(cinfo, file)` | Output to FILE* | — | ❌ |
| `jpeg_stdio_src(cinfo, file)` | Input from FILE* | — | ❌ |
| `jpeg_mem_dest(cinfo, &outbuf, &outsize)` | Output to memory buffer | `Vec<u8>` output (native) | ✅ |
| `jpeg_mem_src(cinfo, inbuf, insize)` | Input from memory buffer | `&[u8]` input (native) | ✅ |

### Compression Setup

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_set_defaults(cinfo)` | Set default compression params | Automatic in `compress()` | ✅ |
| `jpeg_set_colorspace(cinfo, cs)` | Set JPEG colorspace | Auto-detected from PixelFormat | 🔶 |
| `jpeg_default_colorspace(cinfo)` | Reset to default colorspace | — | ❌ |
| `jpeg_set_quality(cinfo, quality, force_baseline)` | Set quality factor | `quality: u8` parameter | ✅ |
| `jpeg_set_linear_quality(cinfo, scale, force_baseline)` | Set linear quality scaling | — | ❌ |
| `jpeg_default_qtables(cinfo, force_baseline)` | Reset quant tables | — | ❌ |
| `jpeg_add_quant_table(cinfo, which, table, scale, force_baseline)` | Add custom quant table | — | ❌ |
| `jpeg_quality_scaling(quality)` | Convert quality to scale factor | Internal in `tables::quality_scale_quant_table` | ✅ |
| `jpeg_enable_lossless(cinfo, psv, pt)` | Enable lossless mode | `compress_lossless()` (psv=1, pt=0 only) | 🔶 |
| `jpeg_simple_progression(cinfo)` | Set standard progressive scan script | Used internally in `compress_progressive()` | ✅ |
| `jpeg_suppress_tables(cinfo, suppress)` | Control table output | — | ❌ |
| `jpeg_alloc_quant_table(cinfo)` | Allocate quant table | — | ❌ |
| `jpeg_alloc_huff_table(cinfo)` | Allocate Huffman table | — | ❌ |

### Compression Processing

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_start_compress(cinfo, write_all_tables)` | Begin compression | Internal in `compress()` | ✅ |
| `jpeg_write_scanlines(cinfo, scanlines, num_lines)` | Write scanline rows | Whole-image only via `compress()` | 🔶 |
| `jpeg12_write_scanlines(...)` | Write 12-bit scanlines | — | ❌ |
| `jpeg16_write_scanlines(...)` | Write 16-bit scanlines | — | ❌ |
| `jpeg_finish_compress(cinfo)` | Finalize compression | Internal in `compress()` | ✅ |
| `jpeg_calc_jpeg_dimensions(cinfo)` | Compute output dimensions | — | ❌ |
| `jpeg_write_raw_data(cinfo, data, num_lines)` | Write raw downsampled data | — | ❌ |
| `jpeg12_write_raw_data(...)` | Write 12-bit raw data | — | ❌ |

### Marker Writing

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_write_marker(cinfo, marker, data, len)` | Write arbitrary marker | — | ❌ |
| `jpeg_write_m_header(cinfo, marker, len)` | Begin streaming marker write | — | ❌ |
| `jpeg_write_m_byte(cinfo, val)` | Write one byte of marker data | — | ❌ |
| `jpeg_write_tables(cinfo)` | Write tables-only datastream | — | ❌ |
| `jpeg_write_icc_profile(cinfo, data, len)` | Write ICC profile | `compress_with_metadata()` / `marker_writer::write_app2_icc()` | ✅ |

### Decompression

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_header(cinfo, require_image)` | Parse headers | `Decoder::new()` → `MarkerReader::read_markers()` | ✅ |
| `jpeg_start_decompress(cinfo)` | Begin decompression | Internal in `decode_image()` | ✅ |
| `jpeg_read_scanlines(cinfo, scanlines, max_lines)` | Read scanline rows | Whole-image via `decompress()` | 🔶 |
| `jpeg12_read_scanlines(...)` | Read 12-bit scanlines | — | ❌ |
| `jpeg16_read_scanlines(...)` | Read 16-bit scanlines | — | ❌ |
| `jpeg_skip_scanlines(cinfo, num_lines)` | Skip rows during decode | `StreamingDecoder::skip_scanlines()` | 🔶 |
| `jpeg12_skip_scanlines(...)` | Skip 12-bit scanlines | — | ❌ |
| `jpeg_crop_scanline(cinfo, &xoffset, &width)` | Scanline-level crop | `StreamingDecoder::crop_scanline()` | 🔶 |
| `jpeg12_crop_scanline(...)` | 12-bit crop | — | ❌ |
| `jpeg_finish_decompress(cinfo)` | Finalize decompression | Internal | ✅ |
| `jpeg_read_raw_data(cinfo, data, max_lines)` | Read raw downsampled data | — | ❌ |
| `jpeg12_read_raw_data(...)` | Read 12-bit raw data | — | ❌ |

### Buffered Image Mode (Progressive Output)

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_has_multiple_scans(cinfo)` | Check if progressive/multi-scan | `FrameHeader.is_progressive` | ✅ |
| `jpeg_start_output(cinfo, scan_number)` | Begin output for specific scan | — | ❌ |
| `jpeg_finish_output(cinfo)` | Finish scan output | — | ❌ |
| `jpeg_input_complete(cinfo)` | Check if all input consumed | — | ❌ |
| `jpeg_consume_input(cinfo)` | Process more input data | — | ❌ |
| `jpeg_new_colormap(cinfo)` | Update colormap after quant change | — | ❌ |

### Output Dimensions

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_calc_output_dimensions(cinfo)` | Compute scaled output size | `ScalingFactor::scale_dim()` | ✅ |
| `jpeg_core_output_dimensions(cinfo)` | Core dimension calculation | Internal | ✅ |

### Marker Management

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_save_markers(cinfo, marker_code, length_limit)` | Enable marker saving | ICC/EXIF/Adobe hard-coded | 🔶 |
| `jpeg_set_marker_processor(cinfo, marker_code, routine)` | Custom marker parser | — | ❌ |

### Coefficient Access

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_read_coefficients(cinfo)` | Read DCT coefficient arrays | `read_coefficients()` | ✅ |
| `jpeg_write_coefficients(cinfo, coef_arrays)` | Write coefficient arrays to JPEG | `write_coefficients()` | ✅ |
| `jpeg_copy_critical_parameters(src, dst)` | Copy quant/Huffman/colorspace between sessions | — | ❌ |

### Error / Sync

| C Function | Description | Rust | Status |
|---|---|---|---|
| `jpeg_resync_to_restart(cinfo, desired)` | Resync to restart marker after error | Internal in decoder | ✅ |

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
| `tjRedOffset[12]` | Red channel offset per format | Implicit in color conversion | 🔶 |
| `tjGreenOffset[12]` | Green channel offset per format | Implicit | 🔶 |
| `tjBlueOffset[12]` | Blue channel offset per format | Implicit | 🔶 |
| `tjAlphaOffset[12]` | Alpha channel offset per format | Implicit | 🔶 |

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
| `JDCT_ISLOW` | Accurate integer DCT | Only method used | ✅ |
| `JDCT_IFAST` | Fast integer DCT (less accurate) | — | ❌ |
| `JDCT_FLOAT` | Floating-point DCT | — | ❌ |

### Dithering (`J_DITHER_MODE`)
| Value | Description | Rust | Status |
|---|---|---|---|
| `JDITHER_NONE` | No dithering | — | ❌ |
| `JDITHER_ORDERED` | Ordered dither | — | ❌ |
| `JDITHER_FS` | Floyd-Steinberg error diffusion | — | ❌ |

### Return Codes
| Value | Description | Rust | Status |
|---|---|---|---|
| `JPEG_SUSPENDED` (0) | Suspended, need more input | N/A (full-buffer API) | ❌ |
| `JPEG_HEADER_OK` (1) | Valid image found | `Decoder::new()` success | ✅ |
| `JPEG_HEADER_TABLES_ONLY` (2) | Tables-only datastream | — | ❌ |
| `JPEG_REACHED_SOS` (1) | Start of new scan | Internal | 🔶 |
| `JPEG_REACHED_EOI` (2) | End of image | Internal | ✅ |
| `JPEG_ROW_COMPLETED` (3) | Completed one iMCU row | — | ❌ |
| `JPEG_SCAN_COMPLETED` (4) | Completed last row of scan | — | ❌ |

### Marker Codes
| Value | Description | Rust | Status |
|---|---|---|---|
| `JPEG_RST0` (0xD0) | Restart marker base | Handled in decode | ✅ |
| `JPEG_EOI` (0xD9) | End of image | Handled | ✅ |
| `JPEG_APP0` (0xE0) | APP0 (JFIF) | Read + write | ✅ |
| `JPEG_COM` (0xFE) | Comment marker | Read (skip) only | 🔶 |

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
| `JPOOL_PERMANENT` (0) | Permanent memory pool | N/A (Rust allocator) | ❌ |
| `JPOOL_IMAGE` (1) | Image-scoped memory pool | N/A (Rust allocator) | ❌ |

---

## Structs (`jpeglib.h`)

| Struct | Description | Rust | Status |
|---|---|---|---|
| `JQUANT_TBL` | Quantization table (64 values + sent_table) | Internal `[u16; 64]` arrays | ✅ |
| `JHUFF_TBL` | Huffman table (bits[17] + huffval[256]) | `HuffmanTable` / `HuffTable` | ✅ |
| `jpeg_component_info` | Per-component metadata | `ComponentInfo` | ✅ |
| `jpeg_scan_info` | Scan script entry (components, Ss/Se/Ah/Al) | `ScanHeader` / `ScanInfo` | ✅ |
| `jpeg_marker_struct` | Saved marker (code, length, data, next) | `IccChunk`, `exif_data` (partial) | 🔶 |
| `jpeg_common_struct` | Common fields (err, mem, progress) | — | ❌ |
| `jpeg_compress_struct` | Full compression state (~50 fields) | Spread across function params | 🔶 |
| `jpeg_decompress_struct` | Full decompression state (~60 fields) | `Decoder` + `JpegMetadata` | 🔶 |
| `jpeg_error_mgr` | Error handler (5 callbacks + state) | `JpegError` enum | 🔶 |
| `jpeg_progress_mgr` | Progress callback + counters | — | ❌ |
| `jpeg_destination_mgr` | Output stream (buffer + 3 callbacks) | `Vec<u8>` | 🔶 |
| `jpeg_source_mgr` | Input stream (buffer + 5 callbacks) | `&[u8]` | 🔶 |
| `jpeg_memory_mgr` | Memory allocator (12 methods) | Rust allocator | ❌ |
