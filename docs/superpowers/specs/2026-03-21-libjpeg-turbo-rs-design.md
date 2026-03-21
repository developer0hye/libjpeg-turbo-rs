# libjpeg-turbo-rs Design Spec

## Goal

Pure Rust reimplementation of libjpeg-turbo with equivalent or better performance. Feature-complete parity with libjpeg-turbo 3.1.x — every API, every option, every edge case.

## Key Decisions

- **Pure Rust** — no C/assembly dependencies, SIMD via `std::arch` intrinsics
- **Decoding first** — build common modules (DCT, color conversion, Huffman, SIMD backend), then reuse for encoding
- **SIMD targets** — AArch64 NEON first (dev machine: Apple M2), x86_64 SSE2/AVX2 via CI, scalar fallback always available
- **Full JPEG spec** — Baseline, Progressive, Arithmetic coding, Lossless DPCM, 2-16 bit precision
- **Dual API** — high-level TurboJPEG v3-style + streaming libjpeg-style, internal pipeline is streaming
- **Unsafe policy** — safe Rust by default; unsafe only for SIMD intrinsics and proven hot-path bounds check elision; every `unsafe` block requires `// SAFETY:` justification

---

## Complete Feature Matrix (libjpeg-turbo 3.1.x parity target)

### A. Pixel Formats (13 total)

| Format | Bytes/px | Layout | Notes |
|--------|----------|--------|-------|
| RGB | 3 | R, G, B | |
| BGR | 3 | B, G, R | |
| RGBX | 4 | R, G, B, X | X=undefined on decompress |
| BGRX | 4 | B, G, R, X | |
| XBGR | 4 | X, B, G, R | |
| XRGB | 4 | X, R, G, B | |
| GRAY | 1 | luminance | |
| RGBA | 4 | R, G, B, A | A=0xFF on decompress |
| BGRA | 4 | B, G, R, A | |
| ABGR | 4 | A, B, G, R | |
| ARGB | 4 | A, R, G, B | |
| CMYK | 4 | C, M, Y, K | |
| RGB565 | 2 | 5/6/5 packed | decompression only |

### B. JPEG Colorspaces (5 total)

| Colorspace | Components | Subsampling | Notes |
|------------|-----------|-------------|-------|
| YCbCr | 3 | all modes | standard photo JPEG |
| Grayscale | 1 | N/A | luminance only |
| RGB | 3 | none (4:4:4 only) | no color conversion |
| CMYK | 4 | none (4:4:4 only) | no color conversion |
| YCCK | 4 | all modes | CMYK with YCbCr transform |

### C. Subsampling Modes (7 total)

| Mode | H:V | iMCU size | SIMD accelerated | Notes |
|------|-----|-----------|-------------------|-------|
| 4:4:4 | 1:1 | 8×8 | yes | no subsampling |
| 4:2:2 | 2:1 | 16×8 | yes | horizontal 2× |
| 4:2:0 | 2:2 | 16×16 | yes | both 2× |
| Gray | N/A | 8×8 | yes | single component |
| 4:4:0 | 1:2 | 8×16 | partial | vertical 2× |
| 4:1:1 | 4:1 | 32×8 | partial | horizontal 4× |
| 4:4:1 | 1:4 | 8×32 | partial | transposed 4:1:1 |

### D. DCT/IDCT Algorithms (3 methods)

| Method | Precision | Speed | SIMD | Notes |
|--------|-----------|-------|------|-------|
| ISLOW (accurate integer) | highest | moderate | full | default, Loeffler-Ligtenberg-Moschytz |
| IFAST (fast integer) | lower | fastest | partial | Arai-Agui-Nakajima, degrades at quality>97 |
| FLOAT (floating point) | high | variable | SSE only | legacy |

### E. Scaled IDCT (16 scale factors)

1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1/1, 9/8, 5/4, 11/8, 3/2, 13/8, 7/4, 15/8, 2/1

- 1/4 and 1/2 are SIMD-accelerated (reduced 2×2 and 4×4 IDCT)
- Not available for lossless JPEG
- Chroma components may independently scale via IDCT to avoid separate upsampling

### F. Entropy Coding (2 methods)

| Method | Compression | Speed | 8-bit | 12-bit | Progressive |
|--------|-------------|-------|-------|--------|-------------|
| Huffman | standard | fast | yes | yes | yes |
| Arithmetic | ~5-10% better | slower | yes | yes | yes |

Huffman optimization: optional pass to compute optimal Huffman tables (better compression, slower).

### G. Lossless JPEG (ITU-T T.81 predictive DPCM)

**NOT** JPEG-LS (ITU-T T.87) — completely different standard.

7 predictor selection values (PSV):
1. Ra (left)
2. Rb (above)
3. Rc (diagonal upper-left)
4. Ra + Rb − Rc
5. Ra + (Rb − Rc) / 2
6. Rb + (Ra − Rc) / 2
7. (Ra + Rb) / 2

- Point transform (Pt): 0 to precision−1 (0 = fully lossless)
- Data precision: 2–16 bits per sample
- Restrictions: no colorspace conversion, no subsampling (always 4:4:4), no progressive, no arithmetic coding, no scaling, no transforms
- First row uses special predictor `2^(P−Pt−1)`

### H. Progressive JPEG

- DCT coefficients split across multiple scans of increasing quality
- DC scans: `Ss=0, Se=0`, interleaved components (up to 4)
- AC scans: `Ss>0, Se>0`, exactly 1 component per scan
- Successive approximation: `Ah` (last bit), `Al` (current bit)
- Default script: `jpeg_simple_progression()` generates ~10 scans
- Custom scan scripts supported
- Block smoothing: 5×5 DC neighborhood estimation for early display
- Scan limit parameter for security (`TJPARAM_SCANLIMIT`)

### I. Lossless Transforms (8 spatial operations)

| Operation | Description |
|-----------|-------------|
| NONE | no spatial transform |
| HFLIP | horizontal mirror |
| VFLIP | vertical mirror |
| TRANSPOSE | flip along UL-to-LR diagonal |
| TRANSVERSE | flip along UR-to-LL diagonal |
| ROT90 | clockwise 90° |
| ROT180 | 180° |
| ROT270 | counter-clockwise 90° |

Transform option flags:
- `PERFECT` — error if transform is imperfect (partial iMCU)
- `TRIM` — discard untransformable partial iMCUs
- `CROP` — lossless cropping (crop x,y must align to iMCU boundaries)
- `GRAY` — discard color, produce grayscale
- `NOOUTPUT` — run filter only, no output image
- `PROGRESSIVE` — progressive output
- `COPYNONE` — no marker copy
- `ARITHMETIC` — arithmetic coding in output
- `OPTIMIZE` — Huffman optimization in output

Custom DCT filter callback: receives `short*` coefficients in frequency domain with region info.

### J. ICC Profile Support

- Embed profile during compression via APP2 markers
- Multi-marker fragmentation (65519 bytes max per marker)
- Each marker carries sequence number + total count
- Read: reassemble fragments in sequence order, validate consistency
- TurboJPEG: `tj3SetICCProfile()` / `tj3GetICCProfile()`
- libjpeg: `jpeg_write_icc_profile()` / `jpeg_read_icc_profile()`
- Profiles are stored/retrieved but NOT applied (requires external CMS)

### K. Partial / Cropped Decompression

- **Horizontal crop**: `jpeg_crop_scanline()` — xoffset aligned to iMCU column boundaries
- **Vertical skip**: `jpeg_skip_scanlines()` — skip scanlines without decoding
- **TurboJPEG crop**: `tj3SetCroppingRegion()` — full 2D crop, combinable with scaling
- Alignment constraints: crop x must be divisible by scaled iMCU width
- Progressive crop: full coefficients stored, only cropped region gets IDCT

### L. Decompression Scaling

- `scale_num / scale_denom` on decompression (16 valid ratios from 1/8 to 2/1)
- Implemented via reduced-size IDCT (1×1, 2×2, 3×3, 4×4, 5×5, 6×6, 7×7, 8×8, 9×9 through 16×16)
- `jpeg_calc_output_dimensions()` computes actual output size

### M. Color Quantization (colormapped output)

- 1-pass quantizer: fast, uses fixed colormap
- 2-pass quantizer: slower, optimal colormap derived from image
- External colormap: supply colormap before decompression
- Dithering modes: none, ordered, Floyd-Steinberg
- `desired_number_of_colors`: 2–256
- Not available for lossless JPEG

### N. Raw Data Mode

- **Compression**: `raw_data_in = true` → supply pre-downsampled component data
- **Decompression**: `raw_data_out = true` → get raw component data before upsampling/color conversion
- Bypasses color conversion and up/downsampling

### O. Buffered-Image Mode (multi-pass progressive display)

- Enables progressive display during incremental download
- `jpeg_start_output()` / `jpeg_finish_output()` loop for each scan
- `jpeg_consume_input()` to pull more data
- `jpeg_has_multiple_scans()` / `jpeg_input_complete()` state queries
- Color quantization pass switching between scans
- `jpeg_new_colormap()` for supplying external colormaps

### P. Coefficient Access (lossless transcoding)

- `jpeg_read_coefficients()` → raw DCT coefficient arrays
- `jpeg_write_coefficients()` → write from coefficient arrays
- `jpeg_copy_critical_parameters()` → copy tables/headers between instances
- Enables lossless JPEG-to-JPEG operations without decode/re-encode

### Q. YUV Plane Operations (TurboJPEG)

- Encode: RGB → planar YUV (no JPEG)
- Decode: planar YUV → RGB (no JPEG)
- Compress: planar YUV → JPEG
- Decompress: JPEG → planar YUV
- Both unified buffer and separate-planes variants
- Buffer size calculation functions for each format

### R. Marker / Metadata Handling

- Save markers: `jpeg_save_markers()` — save APPn/COM markers to linked list
- Custom marker processors: `jpeg_set_marker_processor()` — per-marker callback
- Write markers: `jpeg_write_marker()` — write arbitrary markers during compression
- Piecemeal writing: `jpeg_write_m_header()` + `jpeg_write_m_byte()`
- JFIF header: version, density units, X/Y density
- Adobe marker: transform type
- TurboJPEG: `TJPARAM_SAVEMARKERS` (0=none, 1=COM, 2=all, 3=all except ICC, 4=ICC only)

### S. Custom Source / Destination Managers

Source manager (decompression):
- `init_source`, `fill_input_buffer`, `skip_input_data`, `resync_to_restart`, `term_source`
- Supports suspension (return false from `fill_input_buffer` to pause)

Destination manager (compression):
- `init_destination`, `empty_output_buffer`, `term_destination`

Built-in: `stdio` (FILE*-based) and `mem` (memory-based) for both.

### T. Restart Markers

- Interval specified in MCUs (`restart_interval`) or MCU rows (`restart_in_rows`)
- RST0-RST7 cycling markers
- On restart: DC predictions reset, bit buffer flushed, entropy state reset
- Enable fault-tolerant decoding (resume after corruption)
- Limited to 16-bit unsigned (JPEG spec constraint)
- Note: optimized Huffman decoder has ~20% slowdown with restart markers

### U. Abbreviated Datastreams

- Tables-only streams: contain only DQT/DHT markers, no image data
- Image-only streams: reference previously loaded tables
- Used in video and streaming applications
- `jpeg_write_tables()` to emit tables-only stream
- `jpeg_read_header()` returns `JPEG_HEADER_TABLES_ONLY` for these

### V. Memory Management

libjpeg-turbo uses pool-based allocation:
- `JPOOL_PERMANENT`: survives for instance lifetime
- `JPOOL_IMAGE`: freed at finish_compress/decompress
- Virtual arrays: demand-paged with temp file backing store when memory limit exceeded
- `max_memory_to_use` / `JPEGMEM` environment variable

### W. Error Handling

libjpeg API:
- `setjmp`/`longjmp` based error recovery
- `jpeg_error_mgr` with function pointers: `error_exit`, `emit_message`, `output_message`, `format_message`, `reset_error_mgr`
- Warning vs fatal distinction

TurboJPEG API:
- Return codes + `tj3GetErrorStr()` / `tj3GetErrorCode()`
- `TJPARAM_STOPONWARNING`: continue or abort on warnings

### X. Compression Options (complete list)

| Option | Range | Notes |
|--------|-------|-------|
| Quality | 1–100 | maps to quantization table scaling |
| Subsampling | 7 modes | see section C |
| DCT method | islow/ifast/float | see section D |
| Huffman optimization | bool | extra pass for optimal tables |
| Progressive | bool | multi-scan output |
| Arithmetic | bool | arithmetic entropy coding |
| Lossless | bool + PSV + Pt | predictive DPCM mode |
| Restart interval | MCUs or rows | see section T |
| Smoothing factor | 0–100 | input smoothing |
| Custom quantization tables | up to 4 | via `jpeg_add_quant_table()` |
| Custom scan scripts | arbitrary | for progressive mode |
| Colorspace | 5 types | see section B |
| Pixel format | 13 types | see section A |
| Data precision | 2–16 bits | 8/12-bit lossy, 2–16-bit lossless |
| Density | X/Y + units | JFIF pixel density |
| Force baseline | bool | constrain tables to baseline compatibility |

### Y. Decompression Options (complete list)

| Option | Range | Notes |
|--------|-------|-------|
| Output colorspace | 17 types | including all extended formats |
| Output pixel format | 13 types | see section A |
| Scale factor | 16 ratios | 1/8 through 2/1 |
| Crop region | x, y, w, h | aligned to iMCU boundaries |
| DCT method | islow/ifast/float | |
| Fancy upsampling | bool | triangle filter vs nearest-neighbor |
| Block smoothing | bool | progressive interblock smoothing |
| Color quantization | bool | colormapped output |
| Dither mode | none/ordered/FS | for quantized output |
| Desired colors | 2–256 | quantized palette size |
| Buffered image | bool | multi-pass progressive display |
| Raw data out | bool | bypass upsampling/color conversion |
| Max memory | MB | limit for coefficient buffers |
| Max scans | count | progressive scan limit (security) |
| Max pixels | count | image size limit (security) |

### Z. Image I/O (TurboJPEG utility)

- Load: BMP, PPM/PGM → pixel buffer
- Save: pixel buffer → BMP, PPM/PGM
- 8/12/16-bit precision variants

---

## SIMD Operations Coverage

Each operation needs separate implementations for each target architecture.

### Operations requiring SIMD (hot path)

| Operation | x86 SSE2 | x86 AVX2 | AArch64 NEON | Scalar |
|-----------|----------|----------|--------------|--------|
| RGB→YCbCr (7 pixel format variants) | yes | yes | yes | yes |
| YCbCr→RGB (7 pixel format variants) | yes | yes | yes | yes |
| RGB→Gray (7 pixel format variants) | yes | yes | yes | yes |
| Forward DCT islow | yes | yes | yes | yes |
| Forward DCT ifast | yes | — | yes | yes |
| Inverse DCT islow 8×8 | yes | yes | yes | yes |
| Inverse DCT ifast 8×8 | yes | — | yes | yes |
| Inverse DCT 4×4 (1/2 scale) | yes | yes | yes | yes |
| Inverse DCT 2×2 (1/4 scale) | yes | yes | yes | yes |
| Quantization (reciprocal multiply) | yes | yes | yes | yes |
| h2v1 downsample | yes | yes | yes | yes |
| h2v2 downsample | yes | yes | yes | yes |
| h2v1 upsample (box) | yes | yes | yes | yes |
| h2v2 upsample (box) | yes | yes | yes | yes |
| h2v1 fancy upsample | yes | yes | yes | yes |
| h2v2 fancy upsample | yes | yes | yes | yes |
| h2v1 merged upsample+CC (7 pf) | yes | yes | yes | yes |
| h2v2 merged upsample+CC (7 pf) | yes | yes | yes | yes |
| Huffman encode (1 block) | yes | yes | yes | yes |
| YCbCr→RGB565 | — | — | yes | yes |
| Sample conversion (8↔12 bit) | yes | yes | yes | yes |

Key insight: each color conversion operation has **7 separate SIMD kernels** (one per extended pixel format).
Merged upsample+color conversion combines two operations into one pass — critical optimization for 4:2:0/4:2:2.

### Quantization SIMD: reciprocal multiplication

Instead of division, libjpeg-turbo stores `reciprocal`, `correction`, `scale`, `shift` per table entry:
`result = ((input + correction) * reciprocal) >> shift`
This enables full SIMD parallelism for what is normally a division-heavy operation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  High-level API (TurboJPEG v3 style)                        │
│  compress() / decompress() / transform()                     │
│  encode_yuv() / decode_yuv() / compress_from_yuv()           │
│  icc_profile() / buffer_size() / load_image() / save_image() │
├──────────────────────────────────────────────────────────────┤
│  Streaming API (libjpeg style)                               │
│  scanline read/write, raw data, coefficient access           │
│  buffered-image mode, suspension, custom src/dst managers    │
├──────────────────────────────────────────────────────────────┤
│  Core Pipeline (trait-based virtual dispatch)                 │
│  ┌─────────┬──────────┬──────────┬──────────┬──────────┐     │
│  │Color    │DCT/IDCT  │Entropy   │Sample    │Lossless  │     │
│  │Convert  │(3 methods│(Huffman/ │Up/Down   │DPCM      │     │
│  │(13 pf)  │+ scaled) │Arithmetic│(7 modes) │(7 PSVs)  │     │
│  └────┬────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘     │
│       │         │          │          │          │            │
│  ┌────▼─────────▼──────────▼──────────▼──────────▼─────┐     │
│  │  SIMD Backend (per-arch, per-operation)              │     │
│  │  AArch64 NEON | x86_64 SSE2/AVX2 | Scalar fallback  │     │
│  └─────────────────────────────────────────────────────┘     │
├──────────────────────────────────────────────────────────────┤
│  Support Systems                                             │
│  Memory manager (pool-based, virtual arrays with backing)    │
│  Marker handler (save, write, custom processors)             │
│  ICC profile handler (multi-marker fragment reassembly)      │
│  Error handler (warning/fatal, recoverable)                  │
│  Coefficient buffer (progressive: full-image virtual arrays) │
└──────────────────────────────────────────────────────────────┘
```

## Crate Structure (expanded)

```
libjpeg-turbo-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── api/
│   │   ├── high_level.rs         # compress(), decompress(), transform()
│   │   ├── streaming.rs          # scanline Decoder/Encoder
│   │   ├── yuv.rs                # encode_yuv(), decode_yuv(), compress_from_yuv()
│   │   ├── transform.rs          # lossless transform API
│   │   ├── image_io.rs           # load_image(), save_image() (BMP, PPM/PGM)
│   │   └── params.rs             # TurboJPEG parameter system (25+ params)
│   ├── decode/
│   │   ├── marker.rs             # marker parsing
│   │   ├── bitstream.rs          # bit reader with byte stuffing
│   │   ├── huffman.rs            # Huffman decoding
│   │   ├── arithmetic.rs         # arithmetic decoding (ITU-T T.81 Annex D)
│   │   ├── entropy.rs            # MCU decoder, DC prediction, restart handling
│   │   ├── dequant.rs            # dequantization
│   │   ├── idct.rs               # 8×8 inverse DCT (islow, ifast, float)
│   │   ├── idct_scaled.rs        # reduced/enlarged IDCT (1×1 through 16×16)
│   │   ├── upsample.rs           # chroma upsampling (simple + fancy)
│   │   ├── merged_upsample.rs    # merged upsample+color conversion
│   │   ├── color.rs              # YCbCr→RGB, Gray, CMYK, YCCK (all 13 pf)
│   │   ├── progressive.rs        # progressive coefficient buffering + block smoothing
│   │   ├── lossless.rs           # lossless DPCM decoding (7 predictors)
│   │   ├── crop.rs               # partial decompression (crop + skip)
│   │   ├── quantize.rs           # color quantization (1-pass, 2-pass, dithering)
│   │   └── pipeline.rs           # full Decoder orchestration
│   ├── encode/
│   │   ├── marker.rs             # marker writing (JFIF, Adobe, custom)
│   │   ├── huffman.rs            # Huffman encoding (+ optimization pass)
│   │   ├── arithmetic.rs         # arithmetic encoding
│   │   ├── entropy.rs            # MCU encoder, DC prediction, restart
│   │   ├── quant.rs              # quantization (quality→table, reciprocal multiply)
│   │   ├── fdct.rs               # forward DCT (islow, ifast, float)
│   │   ├── downsample.rs         # chroma downsampling
│   │   ├── color.rs              # RGB→YCbCr, Gray, CMYK, YCCK (all 13 pf)
│   │   ├── progressive.rs        # progressive scan script execution
│   │   ├── lossless.rs           # lossless DPCM encoding (7 predictors)
│   │   └── pipeline.rs           # full Encoder orchestration
│   ├── transform/
│   │   ├── spatial.rs            # 8 spatial operations on DCT coefficients
│   │   ├── crop.rs               # lossless cropping
│   │   └── filter.rs             # custom DCT filter callback support
│   ├── common/
│   │   ├── types.rs              # ColorSpace, Subsampling, PixelFormat, etc.
│   │   ├── error.rs              # JpegError enum, warning/fatal distinction
│   │   ├── huffman_table.rs      # Huffman table build + fast lookup
│   │   ├── quant_table.rs        # quantization table (standard tables, zigzag)
│   │   ├── icc.rs                # ICC profile multi-marker fragment assembly
│   │   ├── marker.rs             # marker constants, saved marker list
│   │   └── mem.rs                # pool allocator, virtual array manager
│   └── simd/
│       ├── mod.rs                # runtime dispatch, SimdBackend trait
│       ├── scalar.rs             # fallback implementations
│       ├── aarch64/
│       │   ├── mod.rs
│       │   ├── color_convert.rs  # 7 pixel format variants each direction
│       │   ├── dct.rs            # forward + inverse DCT
│       │   ├── idct_scaled.rs    # 2×2, 4×4 reduced IDCT
│       │   ├── sample.rs         # up/downsample
│       │   ├── merged.rs         # merged upsample+CC
│       │   ├── quant.rs          # reciprocal quantization
│       │   └── huffman.rs        # Huffman encode acceleration
│       └── x86_64/
│           ├── mod.rs            # SSE2 + AVX2 runtime detection
│           ├── color_convert.rs
│           ├── dct.rs
│           ├── idct_scaled.rs
│           ├── sample.rs
│           ├── merged.rs
│           ├── quant.rs
│           └── huffman.rs
├── tests/
│   ├── fixtures/
│   ├── conformance/
│   └── ...
├── benches/
│   └── decode.rs
└── fuzz/
    └── decode.rs
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum JpegError {
    #[error("invalid marker: 0xFF{0:02X}")]
    InvalidMarker(u8),
    #[error("unexpected marker: 0xFF{0:02X}")]
    UnexpectedMarker(u8),
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    #[error("corrupt data: {0}")]
    CorruptData(String),
    #[error("buffer too small: need {need}, got {got}")]
    BufferTooSmall { need: usize, got: usize },
    #[error("unexpected end of data")]
    UnexpectedEof,
    #[error("imperfect transform: {0}")]
    ImperfectTransform(String),
    #[error("scan limit exceeded: {0} scans")]
    ScanLimitExceeded(u32),
    #[error("pixel limit exceeded: {0} pixels")]
    PixelLimitExceeded(u64),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Non-fatal warning that allows recovery.
#[derive(Debug)]
pub struct JpegWarning {
    pub message: String,
}
```

## Public API

### High-level (TurboJPEG v3 style)

```rust
// Compression
let jpeg = Compressor::new()
    .quality(85)
    .subsampling(Subsampling::S420)
    .progressive(true)
    .optimize_huffman(true)
    .compress(&rgb_pixels, width, height, PixelFormat::Rgb)?;

// Decompression
let header = Decompressor::read_header(&jpeg_bytes)?;
let image = Decompressor::new()
    .scale(ScalingFactor::new(1, 2))   // 1/2 size
    .crop(Region::new(0, 0, 640, 480))
    .pixel_format(PixelFormat::Bgra)
    .decompress(&jpeg_bytes)?;

// Lossless transform
let transformed = Transformer::new()
    .operation(TransformOp::Rot90)
    .crop(Region::new(0, 0, 640, 480))
    .progressive(true)
    .transform(&jpeg_bytes)?;

// YUV operations
let yuv = encode_yuv(&rgb_pixels, width, height, PixelFormat::Rgb, Subsampling::S420)?;
let jpeg = compress_from_yuv(&yuv_planes, width, height, Subsampling::S420, quality)?;
let yuv = decompress_to_yuv(&jpeg_bytes)?;
let rgb = decode_yuv(&yuv_planes, width, height, Subsampling::S420, PixelFormat::Rgb)?;

// ICC profiles
let mut comp = Compressor::new();
comp.set_icc_profile(&icc_data);
let jpeg = comp.compress(...)?;

let mut decomp = Decompressor::new();
decomp.read_header(&jpeg)?;
let icc = decomp.icc_profile();
```

### Streaming (libjpeg style)

```rust
// Decoding
let mut decoder = StreamingDecoder::new(reader)?;
let header = decoder.header();
decoder.set_output_colorspace(ColorSpace::Rgb)?;
decoder.set_scale(1, 2)?;
decoder.start()?;
while let Some(scanline) = decoder.read_scanline()? { ... }
decoder.finish()?;

// Encoding
let mut encoder = StreamingEncoder::new(writer)?;
encoder.set_image_dimensions(width, height, 3)?;
encoder.set_colorspace(ColorSpace::YCbCr)?;
encoder.set_quality(85)?;
encoder.start()?;
for row in scanlines { encoder.write_scanline(row)?; }
encoder.finish()?;

// Coefficient access (lossless transcoding)
let coefficients = decoder.read_coefficients()?;
// ... modify coefficients ...
encoder.write_coefficients(&coefficients)?;

// Raw data mode
decoder.set_raw_data_out(true)?;
let raw_planes = decoder.read_raw_data()?;

// Buffered-image mode (progressive display)
decoder.set_buffered_image(true)?;
decoder.start()?;
while !decoder.input_complete() {
    decoder.consume_input()?;
    decoder.start_output(decoder.input_scan_number())?;
    while let Some(scanline) = decoder.read_scanline()? { ... }
    decoder.finish_output()?;
}
```

## Testing Strategy

- **Conformance** — decode libjpeg-turbo test images, bit-exact comparison for all subsampling modes, all pixel formats
- **Round-trip** — compress then decompress, verify PSNR is within expected range
- **Fuzz** — `cargo-fuzz` with malformed JPEG inputs (AFL/libFuzzer corpus)
- **Benchmark** — `criterion` comparing against libjpeg-turbo (C) and zune-jpeg
- **SIMD verification** — each SIMD path must produce identical output to scalar fallback
- **Progressive** — verify multi-scan decode produces same result as single-scan reference
- **Lossless** — verify exact reconstruction for all 7 predictors at all precisions
- **Transform** — verify all 8 spatial operations produce bit-identical output to jpegtran
- **Scaling** — verify all 16 scale factors produce correct dimensions and plausible output
- **ICC** — round-trip ICC profile embed/extract
- **Interop** — decode JPEGs from diverse sources (cameras, phones, web, Photoshop, GIMP)

## Implementation Roadmap (revised)

| Phase | Scope | Goal |
|-------|-------|------|
| 1 | Baseline JPEG decoder (scalar) | Correctness ✅ DONE |
| 2 | SIMD optimization (NEON → SSE2/AVX2) | libjpeg-turbo-level decode performance |
| 3 | All pixel formats + extended colorspaces | Full pixel format coverage |
| 4 | Scaled IDCT (1/8 through 2/1) | Decompression scaling |
| 5 | Fancy upsampling + merged upsample+CC | Output quality + performance parity |
| 6 | Progressive JPEG decoding | Full progressive support |
| 7 | Arithmetic decoding | Arithmetic entropy support |
| 8 | Lossless JPEG decoding (7 predictors, 2–16 bit) | Lossless decode |
| 9 | Partial decompression (crop + skip) | Crop support |
| 10 | Baseline JPEG encoder (scalar → SIMD) | Encoding starts |
| 11 | Progressive + Arithmetic + Lossless encoding | Full encoding |
| 12 | Huffman optimization pass | Optimal compression |
| 13 | Lossless transforms (8 ops + crop + filter) | Transform support |
| 14 | YUV plane operations | YUV API |
| 15 | Coefficient access / lossless transcoding | Coefficient API |
| 16 | ICC profile support | ICC read/write |
| 17 | Color quantization + dithering | Colormapped output |
| 18 | Buffered-image mode | Progressive display |
| 19 | Raw data mode | Raw I/O |
| 20 | Custom source/destination managers | Streaming I/O |
| 21 | Abbreviated datastreams | Tables-only support |
| 22 | 12-bit + 16-bit precision | Multi-precision |
| 23 | Image I/O (BMP, PPM/PGM) | File format utilities |
| 24 | Security hardening (scan/pixel limits, fuzz) | Production readiness |

### AA. Thread Safety

- Each compressor/decompressor/transformer instance is fully independent and thread-safe
- No global mutable state between instances
- SIMD dispatch uses lazy one-time initialization (thread-safe)
- In Rust: natural fit — each instance owns its state, `Send + Sync` where appropriate

### AB. Progress Monitoring

- Callback-based progress reporting: `progress_monitor(pass_counter, pass_limit, completed_passes, total_passes)`
- libjpeg-turbo calls this after each scanline or MCU row
- Applications use it for progress bars during large image operations
- In Rust: `Fn(ProgressInfo)` callback parameter on compress/decompress

### AC. Bottom-Up Row Order

- `TJPARAM_BOTTOMUP` — process scanlines from bottom to top
- Required for BMP compatibility (BMP stores rows bottom-up)
- Affects both compression (input order) and decompression (output order)

### AD. No-Realloc Mode

- `TJPARAM_NOREALLOC` — error if output buffer is too small instead of reallocating
- Important for embedded/real-time systems with pre-allocated buffers
- Buffer size calculation functions (`tj3JPEGBufSize`, etc.) help pre-allocate correctly

### AE. Multi-Output Transforms

- `tj3Transform()` can produce N outputs simultaneously from a single input
- Each output can have a different operation, crop region, and options
- Single decode pass, multiple transform outputs — more efficient than N separate calls

### AF. SIMD Runtime Control

- Environment variables for testing/debugging:
  - `JSIMD_FORCENONE=1` — disable all SIMD, use scalar fallback
  - `JSIMD_FORCESSE2=1` — force SSE2 even if AVX2 available
  - `JSIMD_FORCEAVX2=1` — force AVX2
  - `JSIMD_NOHUFFENC=1` — disable SIMD Huffman encoding
  - `JPEGMEM=N` — override max memory limit (MB)
- In Rust: expose via builder pattern or environment variable reading

### AG. Standard Tables (bundled)

- Standard DC/AC Huffman tables (ITU-T T.81 Annex K, Tables K.3–K.6)
- Standard luminance/chrominance quantization tables (Annex K, Table K.1–K.2)
- Quality-to-scale-factor mapping: `jpeg_quality_scaling(quality)`
  - Quality 50 → scale 100 (tables as-is)
  - Quality 1–49 → scale = 5000/quality
  - Quality 51–100 → scale = 200 − 2×quality
- `force_baseline` flag: constrain quantization values to [1, 255] for baseline compatibility

### AH. Fancy Upsampling Rounding Bias

- Triangle filter alternates rounding bias between +1 and +2 across pixel pairs
- Prevents systematic rounding error that would be visible as banding
- h2v1: `left = (3*cur + left_neighbor + 1) >> 2`, `right = (3*cur + right_neighbor + 2) >> 2`
- This is a subtle but important detail for bit-exact compatibility with libjpeg-turbo

### AI. Sample Range Limit Table

- Pre-built 5×(MAXJSAMPLE+1)+CENTERJSAMPLE entry lookup table for fast clamping
- Uses wrap-around trick: out-of-range values index into table regions that return 0 or 255
- Eliminates branch-per-pixel in hot loops (replaced by table lookup)
- Different table sizes for 8-bit (1280 entries), 12-bit, 16-bit precision

### AJ. Suspension Mode (streaming decompression)

- Source manager's `fill_input_buffer()` can return `false` to signal "no data yet"
- `jpeg_read_header()` returns `JPEG_SUSPENDED` (0) when paused
- `jpeg_consume_input()` returns status codes:
  - `JPEG_SUSPENDED` (0) — need more data
  - `JPEG_REACHED_SOS` (1) — reached start of scan
  - `JPEG_REACHED_EOI` (2) — reached end of image
  - `JPEG_ROW_COMPLETED` (3) — completed a row
  - `JPEG_SCAN_COMPLETED` (4) — completed a scan
- Enables incremental decoding over network streams
- In Rust: `Poll`-like or async-compatible interface

### AK. JPEG Constants and Limits

| Constant | Value | Meaning |
|----------|-------|---------|
| `DCTSIZE` | 8 | DCT block dimension |
| `DCTSIZE2` | 64 | coefficients per block |
| `NUM_QUANT_TBLS` | 4 | max quantization tables |
| `NUM_HUFF_TBLS` | 4 | max Huffman tables per class |
| `NUM_ARITH_TBLS` | 16 | max arithmetic coding conditioning tables |
| `MAX_COMPS_IN_SCAN` | 4 | max interleaved components per scan |
| `MAX_SAMP_FACTOR` | 4 | max sampling factor |
| `MAX_COMPONENTS` | 10 | max components per image |
| `JPEG_MAX_DIMENSION` | 65500 | max image dimension |
| `C_MAX_BLOCKS_IN_MCU` | 10 | max blocks per MCU (compressor) |
| `D_MAX_BLOCKS_IN_MCU` | 10 | max blocks per MCU (decompressor) |

### AL. Density / Resolution Info

- JFIF header carries pixel density information:
  - `density_unit`: 0=unknown (aspect ratio only), 1=dots/inch, 2=dots/cm
  - `X_density`, `Y_density`: horizontal/vertical density values
- TurboJPEG: `TJPARAM_DENSITYUNITS`, `TJPARAM_XDENSITY`, `TJPARAM_YDENSITY`
- Preserved through lossless transforms when markers are copied

### AM. Arithmetic Coding Conditioning Parameters

- `arith_dc_L[16]` — lower bound of DC conditioning category
- `arith_dc_U[16]` — upper bound of DC conditioning category
- `arith_ac_K[16]` — AC conditioning parameter (Kx threshold)
- Written in DAC marker segment
- Affect compression efficiency — default values are usually good enough

### AN. Input Smoothing (compression)

- `smoothing_factor` (0–100): low-pass filter applied to input before DCT
- Reduces high-frequency noise at the cost of slight blurring
- Useful for noisy source images to improve compression ratio

### AO. Component Info Details

Full per-component metadata tracked during processing:
- `component_id` — marker-level identifier
- `h_samp_factor`, `v_samp_factor` — sampling factors (1–4)
- `quant_tbl_no` — which quantization table
- `dc_tbl_no`, `ac_tbl_no` — which Huffman tables
- `width_in_blocks`, `height_in_blocks` — dimensions in 8×8 blocks
- `DCT_h_scaled_size`, `DCT_v_scaled_size` — scaled block size for IDCT
- `downsampled_width`, `downsampled_height` — after subsampling
- `MCU_width`, `MCU_height` — blocks per MCU for this component
- `MCU_blocks` — total blocks per MCU for this component
- `MCU_sample_width` — samples per MCU column

### AP. Scan Script Validation

Progressive scan scripts must satisfy:
- DC scans: `Ss=0, Se=0`, up to `MAX_COMPS_IN_SCAN` interleaved components
- AC scans: `Ss>0, Se>0`, exactly 1 component
- Successive approximation: first scan `Ah=0`; refinement `Ah = previous_Al, Al = Ah-1`
- All 64 coefficients (0–63) must be covered for each component
- `last_bitpos[component][coefficient]` tracks precision state

### AQ. Compile-Time Feature Flags (optional features)

In the Rust crate, these become Cargo feature flags:
- `dct-islow` — accurate integer DCT (default, always on)
- `dct-ifast` — fast integer DCT (optional)
- `dct-float` — floating-point DCT (optional)
- `progressive` — progressive JPEG support
- `arithmetic` — arithmetic coding support
- `lossless` — lossless JPEG support
- `color-quantization` — colormapped output support
- `simd` — SIMD acceleration (default on)

### AR. TJSAMP_UNKNOWN

- Represents unusual subsampling configurations (e.g., non-standard sampling factors)
- Can decompress to packed-pixel formats
- Cannot decompress to planar YUV
- Cannot perform lossless cropping
- Must be handled gracefully in all code paths

### AS. `jpegtran -drop` Feature (DCT-domain compositing)

- `jpegtran -drop +X+Y filename` — composites another JPEG into the output at given position
- Operates in DCT domain (lossless for the background)
- The dropped image is re-encoded into the target's coefficient blocks
- Position must align to iMCU boundaries

### AT. Abort / Destroy Operations

- `jpeg_abort_compress()` / `jpeg_abort_decompress()` / `jpeg_abort()` — abort in-progress operation, release working memory (JPOOL_IMAGE) but keep the instance alive for reuse
- `jpeg_destroy_compress()` / `jpeg_destroy_decompress()` / `jpeg_destroy()` — fully destroy instance, release all memory
- Distinction matters: abort allows instance reuse without reallocating, destroy is final
- In Rust: `Drop` trait for destroy, explicit `.abort()` method for mid-operation cleanup

### AU. Restart Marker Resync

- `jpeg_resync_to_restart()` — default restart marker resynchronization after corrupt data
- Called by the source manager when a restart marker is expected but missing
- Scans forward in the stream to find the next valid RST marker
- Can be overridden with custom resync logic via source manager

### AV. Linear Quality & Advanced Quantization Table Control

- `jpeg_set_linear_quality(scale_factor, force_baseline)` — set quality via direct scale factor (not the 1-100 curve)
- `jpeg_default_qtables(force_baseline)` — generate default quantization tables based on `q_scale_factor[]` per component (v7+)
- `jpeg_alloc_quant_table()` / `jpeg_alloc_huff_table()` — allocate table structures within JPEG memory pool
- `jpeg_suppress_tables(suppress)` — mark all current tables as "already sent" or "not yet sent" (for abbreviated datastreams)
- `sent_table` flag on JQUANT_TBL and JHUFF_TBL — tracks whether table has been emitted in the current stream

### AW. Dimension Calculation Functions

- `jpeg_calc_output_dimensions()` — compute `output_width`, `output_height` based on scale, colorspace, etc.
- `jpeg_calc_jpeg_dimensions()` — compute `jpeg_width`, `jpeg_height` (v7+; no-op in libjpeg-turbo)
- `jpeg_core_output_dimensions()` — core dimension calculation (v8+; available in libjpeg-turbo)
- These must be callable after `jpeg_read_header()` and before `jpeg_start_decompress()`

### AX. JFIF / Adobe Marker Control

Compression:
- `write_JFIF_header` — boolean controlling JFIF APP0 marker emission
- `JFIF_major_version`, `JFIF_minor_version` — version fields (typically 1.01 or 1.02)
- `write_Adobe_marker` — boolean controlling Adobe APP14 marker emission

Decompression (read-only):
- `saw_JFIF_marker` — whether JFIF marker was found
- `saw_Adobe_marker` — whether Adobe marker was found
- `Adobe_transform` — Adobe color transform code (0=unknown, 1=YCbCr, 2=YCCK)

### AY. `is_baseline` Flag

- Decompression-only read-only field on `jpeg_decompress_struct`
- Set after `jpeg_read_header()`: `true` if the JPEG is baseline (SOF0), `false` for extended, progressive, lossless
- Useful for applications that need to reject non-baseline images

### AZ. `CCIR601_sampling` Flag

- Affects chroma siting during color conversion
- CCIR 601 uses co-sited chroma (chroma samples aligned with even-numbered luma samples)
- Standard JPEG uses centered chroma (chroma between luma pairs)
- Practically rarely used, but part of the public API surface

### BA. Buffer Size Calculation Functions (TurboJPEG)

| Function | Formula / Purpose |
|----------|-------------------|
| `tj3JPEGBufSize(w, h, subsamp)` | Worst-case compressed JPEG size |
| `tj3YUVBufSize(w, align, h, subsamp)` | Unified planar YUV buffer size |
| `tj3YUVPlaneSize(comp, w, stride, h, subsamp)` | Individual YUV plane size |
| `tj3YUVPlaneWidth(comp, w, subsamp)` | Individual YUV plane width |
| `tj3YUVPlaneHeight(comp, h, subsamp)` | Individual YUV plane height |
| `tj3TransformBufSize(handle, transform)` | Worst-case transform output size |

These are essential for pre-allocating buffers in no-realloc mode.

### BB. Lookup Table Arrays (TurboJPEG)

Static const arrays for pixel format metadata:
- `tjPixelSize[TJ_NUMPF]` — bytes per pixel for each format
- `tjRedOffset[TJ_NUMPF]` — byte offset of red channel
- `tjGreenOffset[TJ_NUMPF]` — byte offset of green channel
- `tjBlueOffset[TJ_NUMPF]` — byte offset of blue channel
- `tjAlphaOffset[TJ_NUMPF]` — byte offset of alpha (-1 if none)
- `tjMCUWidth[TJ_NUMSAMP]` — iMCU width for each subsampling mode
- `tjMCUHeight[TJ_NUMSAMP]` — iMCU height for each subsampling mode

In Rust: implement as `const` arrays or methods on the enum types.

### BC. `unread_marker` Field

- Decompression-only field on `jpeg_decompress_struct`
- Contains the marker code of a marker that was encountered but not yet processed
- Used by suspension mode: when the library suspends, this records where it left off
- Value 0 means no unread marker

### BD. Legacy TurboJPEG API (backward compatibility decision)

libjpeg-turbo maintains full backward compatibility with TJ 1.0 through 2.0 APIs:
- TJ 1.0: `tjInitCompress()`, `tjCompress()`, `tjDecompress()`, etc.
- TJ 1.2: `tjCompress2()`, `tjDecompress2()`, `tjTransform()`, etc.
- TJ 2.0: `tjGetErrorStr2()`, `tjLoadImage()`, `tjSaveImage()`, etc.
- TJ 3.0: `tj3*()` functions (current primary API)

**Decision for Rust:** We implement only the TJ 3.x API surface as native Rust. The legacy APIs exist solely for C ABI backward compatibility and are not meaningful in a Rust library. If C FFI is needed later, a thin `libjpeg-turbo-rs-sys` wrapper crate could expose the legacy names.

### BE. C FFI Compatibility (decision)

**Decision:** The primary deliverable is a native Rust library with idiomatic Rust API. A secondary `libjpeg-turbo-rs-ffi` crate may later provide C-compatible function exports matching the TurboJPEG and/or libjpeg ABI, enabling drop-in replacement of the C library. This is NOT part of the initial implementation phases.

---

## Cross-Reference Verification Summary

Verified against complete symbol extraction from:
- `turbojpeg.h` — 40 tj3 functions, 7 enums (all values), 9 TJXOPT flags, 3 structs, all constants
- `jpeglib.h` — 60+ functions (including 12-bit/16-bit variants), 3 enums, 8 major structs (all fields), all constants
- `jmorecfg.h` — all types, all constants, all feature flags

**Coverage status:**
- All public functions: accounted for (native Rust equivalents or explicit "not in scope" decisions)
- All enum values: accounted for in feature matrix sections A–C, D, I
- All struct fields: accounted for in feature matrix or implementation detail sections
- All constants/limits: accounted for in section AK
- Legacy TJ API: explicit decision in section BD (Rust-native only)
- C FFI: explicit decision in section BE (deferred)

---

## What is explicitly NOT in scope

- **JPEG-LS** (ITU-T T.87) — different standard entirely
- **JPEG 2000** (ITU-T T.800) — different standard
- **JPEG XL** — different standard
- **EXIF structured parsing** — markers saved as raw bytes only
- **Color management / ICC profile application** — profiles stored/retrieved, not applied
- **CMYK↔RGB proper conversion** — requires CMS; only quick/dirty test-quality conversion
- **libjpeg v9-only features** (color_transform, SmartScale variable block sizes) — not in libjpeg-turbo
