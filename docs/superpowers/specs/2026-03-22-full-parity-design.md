# Full libjpeg-turbo Feature Parity — Design Spec

> **Date:** 2026-03-22
> **Scope:** Complete feature parity with libjpeg-turbo C library (x86_64 SIMD and custom memory manager excluded)
> **Approach:** Bottom-up — foundation infrastructure first, then codec features, then advanced APIs
> **API style:** Rust-idiomatic with Encoder/Decoder builder pattern
> **Precision:** Generic `Sample` trait for 8/12/16-bit

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API style | Rust-idiomatic (not C mirror) | Builder pattern, trait-based extensibility |
| Multi-precision | Generic `Sample` trait | Compile-time polymorphism, zero cost for 8-bit |
| Image output type | `Image` + `precision: u8` + raw `Vec<u8>` | Simple, backward compatible |
| Options pattern | `Encoder`/`Decoder` builder, existing functions kept | Gradual adoption, no breaking changes |
| Error/Progress | Rust traits (`ErrorHandler`, `ProgressListener`) | Idiomatic, composable |
| Custom memory mgr | Skip | Rust allocator sufficient; only `max_memory` limit |
| x86_64 SIMD | Skip | No native test environment (M2 Mac) |
| Legacy TJ API | Skip | TJ3 equivalent covered via Rust API |

---

## 1. Sample Trait & Multi-Precision

### Types

```rust
pub trait Sample: Copy + Default + Into<i32> + TryFrom<i32> + Send + Sync + 'static {
    const BITS: u8;
    const MAX_VAL: i32;
    const CENTER: i32;
    const IS_LOSSLESS_ONLY: bool;
    fn from_i32_clamped(v: i32) -> Self;
}

impl Sample for u8  { BITS=8,  MAX_VAL=255,   CENTER=128,   IS_LOSSLESS_ONLY=false }
impl Sample for i16 { BITS=12, MAX_VAL=4095,  CENTER=2048,  IS_LOSSLESS_ONLY=false }
impl Sample for u16 { BITS=16, MAX_VAL=65535, CENTER=32768, IS_LOSSLESS_ONLY=true  }
```

### Image struct

```rust
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    pub precision: u8,          // 8, 12, or 16
    pub data: Vec<u8>,          // raw bytes; cast via bytemuck for 12/16-bit
    pub icc_profile: Option<Vec<u8>>,
    pub exif_data: Option<Vec<u8>>,
    pub comment: Option<String>,
    pub density: DensityInfo,
    pub saved_markers: Vec<SavedMarker>,
    pub warnings: Vec<DecodeWarning>,
}
```

### Impact

- Internal pipeline functions become generic over `S: Sample`
- SIMD paths remain `u8`-specialized; 12/16-bit uses scalar fallback
- Public API: `compress()` stays `u8`; new `Encoder::new(...).precision(12).encode_12bit(&pixels)?`
- Existing user code unchanged

---

## 2. Encoder / Decoder Builder

### Encoder

```rust
pub struct Encoder<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    // Options with defaults
    quality: u8,                              // 75
    subsampling: Subsampling,                 // S420
    optimize_huffman: bool,                   // false
    progressive: bool,                        // false
    arithmetic: bool,                         // false
    lossless: Option<LosslessConfig>,         // None
    restart: RestartInterval,                 // None
    density: DensityInfo,                     // 72 dpi
    precision: u8,                            // 8
    dct_method: DctMethod,                    // IsLow
    smoothing_factor: u8,                     // 0
    grayscale_from_color: bool,               // false
    icc_profile: Option<&'a [u8]>,
    exif_data: Option<&'a [u8]>,
    comment: Option<&'a str>,
    custom_markers: Vec<(u8, &'a [u8])>,
    custom_quant_tables: [Option<[u16; 64]>; 4],
    custom_huffman_dc: [Option<HuffmanTableDef>; 4],
    custom_huffman_ac: [Option<HuffmanTableDef>; 4],
    scan_script: Option<Vec<ScanScript>>,
    error_handler: Option<Box<dyn ErrorHandler>>,
    progress: Option<Box<dyn ProgressListener>>,
    max_memory: Option<usize>,
}
```

All fields added incrementally — only when the feature is implemented.

Builder methods return `Self` for chaining. Terminal method: `.encode() -> Result<Vec<u8>>`.

### Decoder

```rust
pub struct Decoder<'a> {
    // existing fields preserved
    // new options:
    dct_method: DctMethod,
    fancy_upsampling: bool,                   // true
    block_smoothing: bool,                    // true
    stop_on_warning: bool,                    // false
    scan_limit: Option<u32>,
    max_memory: Option<usize>,
    max_pixels: Option<usize>,
    save_markers: MarkerSaveLevel,
    error_handler: Option<Box<dyn ErrorHandler>>,
    progress: Option<Box<dyn ProgressListener>>,
}
```

### Backward compatibility

Existing free functions (`compress`, `decompress`, etc.) remain. Internally they construct `Encoder`/`Decoder` with defaults.

---

## 3. Marker System

### Read side

`MarkerReader` extended:
- Parse COM (0xFE) marker → `Image.comment`
- Parse JFIF density fields → `Image.density`
- Optional: save all APP/COM markers → `Image.saved_markers`
- Controlled by `MarkerSaveLevel`: `Essential` (default), `All`, `Custom(Vec<u8>)`

### Write side

New in `marker_writer.rs`:
```rust
pub fn write_dri(buf: &mut Vec<u8>, interval: u16);
pub fn write_com(buf: &mut Vec<u8>, text: &str);
pub fn write_marker(buf: &mut Vec<u8>, code: u8, data: &[u8]);
pub fn write_app0_jfif_with_density(buf: &mut Vec<u8>, density: &DensityInfo);
```

### Supporting types

```rust
pub struct DensityInfo { pub unit: DensityUnit, pub x: u16, pub y: u16 }
pub enum DensityUnit { Unknown, Dpi, Dpcm }
pub struct SavedMarker { pub code: u8, pub data: Vec<u8> }
pub enum MarkerSaveLevel { Essential, All, Custom(Vec<u8>) }
```

---

## 4. Traits

### ErrorHandler

```rust
pub trait ErrorHandler: Send + Sync {
    fn error_exit(&self, err: &JpegError) -> ! {
        panic!("JPEG error: {}", err);
    }
    fn emit_warning(&self, warning: &DecodeWarning) {}
    fn trace(&self, level: u8, msg: &str) {}
}
```

### ProgressListener

```rust
pub trait ProgressListener: Send + Sync {
    fn update(&self, info: ProgressInfo);
}

pub struct ProgressInfo {
    pub pass: u32,
    pub total_passes: u32,
    pub progress: f32,    // 0.0 ~ 1.0
}

// Closure support
impl<F: Fn(ProgressInfo) + Send + Sync> ProgressListener for F { ... }
```

---

## 5. Codec Completion

### 5.1 Restart Interval Encode (DRI)

- `Encoder::restart_blocks(n)` / `restart_rows(n)`
- All compress paths: MCU counter → every N MCUs: flush bits + RST marker + reset DC
- `write_dri()` marker in output header

### 5.2 SOF10 Arithmetic Progressive Encode

- Combine progressive coefficient buffering + ArithEncoder
- SOF10 (0xCA) marker + DAC conditioning
- Reuse `simple_progression()` scan script

### 5.3 SOF11 Lossless Arithmetic

- Encode: `compress_lossless()` path but ArithEncoder instead of Huffman
- Decode: `decode_lossless_image()` but ArithDecoder instead of Huffman
- SOF11 (0xCB) marker

### 5.4 Lossless Encode Extension

- Multi-component: 3 interleaved components, no color conversion (or optional YCbCr)
- Predictor selection: `LosslessConfig { predictor: 1..=7 }`
- Point transform: `LosslessConfig { point_transform: 0..=15 }`

### 5.5 Grayscale-from-Color

- `Encoder::grayscale_from_color(true)`
- Extract Y component from RGB → encode as 1-component

### 5.6 Custom Quantization Tables

- `Encoder::quant_table(index, [u16; 64])`
- Replaces standard quality-scaled table for that index

### 5.7 Custom Huffman Tables

- `Encoder::huffman_table(class, index, bits, values)`
- Used instead of standard or optimized tables

### 5.8 Custom Progressive Scan Script

- `Encoder::scan_script(Vec<ScanScript>)`
- `ScanScript { components, ss, se, ah, al }`

### 5.9 DCT Method Selection

- `DctMethod::IsLow` — current implementation (accurate integer)
- `DctMethod::IsFast` — fast integer with less accuracy
- `DctMethod::Float` — floating-point

Three implementations each for FDCT and IDCT. Selected via `Encoder::dct_method()` / `Decoder` option.

---

## 6. Pixel Formats & Subsampling

### New pixel formats

| Format | BPP | Channel order | Notes |
|--------|-----|---------------|-------|
| Rgbx | 4 | R,G,B,pad | Pad byte ignored |
| Bgrx | 4 | B,G,R,pad | Pad byte ignored |
| Xrgb | 4 | pad,R,G,B | Pad byte ignored |
| Xbgr | 4 | pad,B,G,R | Pad byte ignored |
| Argb | 4 | A,R,G,B | Alpha-first |
| Abgr | 4 | A,B,G,R | Alpha-first |
| Rgb565 | 2 | 5-6-5 packed | Decode only |

Implementation: offset-based color conversion. Add `red_offset()`, `green_offset()`, `blue_offset()`, `alpha_offset()` to `PixelFormat`.

### S441 subsampling

- MCU size: (8, 32) — 4 Y blocks vertically
- Chroma downsample: 1x4
- Decode upsample: h1v4

---

## 7. Scanline API

### Encode

```rust
pub struct ScanlineEncoder { ... }

impl ScanlineEncoder {
    pub fn new(options: Encoder) -> Result<Self>;
    pub fn write_scanline(&mut self, row: &[u8]) -> Result<()>;
    pub fn write_scanlines(&mut self, rows: &[&[u8]]) -> Result<()>;
    pub fn write_raw_data(&mut self, data: &[&[u8]]) -> Result<()>;
    pub fn finish(self) -> Result<Vec<u8>>;
}
```

### Decode

```rust
pub struct ScanlineDecoder<'a> { ... }

impl<'a> ScanlineDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self>;
    pub fn header(&self) -> &FrameHeader;
    pub fn output_scanline(&self) -> usize;
    pub fn read_scanline(&mut self, buf: &mut [u8]) -> Result<()>;
    pub fn read_scanlines(&mut self, buf: &mut [u8], num_lines: usize) -> Result<usize>;
    pub fn skip_scanlines(&mut self, num_lines: usize) -> Result<usize>;
    pub fn crop_scanline(&mut self, x_offset: &mut usize, width: &mut usize) -> Result<()>;
    pub fn read_raw_data(&mut self, planes: &mut [&mut [u8]]) -> Result<usize>;
    pub fn finish(self) -> Result<()>;
}
```

Internal: decompose existing whole-image pipeline into MCU-row state machine.

---

## 8. YUV Planar API

### Functions

```rust
// Color conversion only (no JPEG)
pub fn encode_yuv(pixels: &[u8], w, h, pf, subsamp) -> Result<Vec<u8>>;
pub fn encode_yuv_planes(pixels: &[u8], w, h, pf, subsamp) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)>;
pub fn decode_yuv(yuv: &[u8], w, h, subsamp, pf) -> Result<Vec<u8>>;
pub fn decode_yuv_planes(y: &[u8], cb: &[u8], cr: &[u8], w, h, subsamp, pf) -> Result<Vec<u8>>;

// JPEG ↔ YUV
pub fn compress_from_yuv(yuv: &[u8], w, h, subsamp, quality) -> Result<Vec<u8>>;
pub fn compress_from_yuv_planes(y, cb, cr, w, h, subsamp, quality) -> Result<Vec<u8>>;
pub fn decompress_to_yuv(jpeg: &[u8]) -> Result<Vec<u8>>;
pub fn decompress_to_yuv_planes(jpeg: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)>;
```

### Buffer size helpers

```rust
pub fn jpeg_buf_size(width: usize, height: usize, subsampling: Subsampling) -> usize;
pub fn yuv_buf_size(width: usize, height: usize, subsampling: Subsampling) -> usize;
pub fn yuv_plane_size(component: usize, width: usize, height: usize, subsampling: Subsampling) -> usize;
pub fn yuv_plane_width(component: usize, width: usize, subsampling: Subsampling) -> usize;
pub fn yuv_plane_height(component: usize, height: usize, subsampling: Subsampling) -> usize;
```

---

## 9. Transform Options & Progressive Output

### Transform

```rust
pub struct TransformOptions {
    pub op: TransformOp,
    pub perfect: bool,
    pub trim: bool,
    pub crop: Option<CropRegion>,
    pub grayscale: bool,
    pub no_output: bool,
    pub progressive: bool,
    pub arithmetic: bool,
    pub optimize: bool,
    pub copy_markers: bool,
    pub custom_filter: Option<Box<dyn FnMut(&mut [i16; 64], FilterInfo) -> bool>>,
}

pub struct FilterInfo {
    pub component_id: u8,
    pub block_x: usize,
    pub block_y: usize,
    pub transform_id: usize,
}

pub fn transform_jpeg_with_options(data: &[u8], options: &TransformOptions) -> Result<Vec<u8>>;
```

### Progressive output

```rust
pub struct ProgressiveDecoder<'a> { ... }

impl<'a> ProgressiveDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self>;
    pub fn header(&self) -> &FrameHeader;
    pub fn num_scans(&self) -> usize;
    pub fn next_scan(&mut self) -> Result<Option<ScanResult>>;
    pub fn finish(self) -> Result<Image>;
}

pub struct ScanResult {
    pub scan_number: usize,
    pub is_final: bool,
}

impl ScanResult {
    pub fn render(&self) -> Result<Image>;  // partial image from accumulated coefficients
}
```

---

## 10. aarch64 NEON Extensions

### New SIMD routines

| Routine | Purpose | Priority |
|---------|---------|----------|
| `neon_fdct_islow` | Forward DCT for encoder | High |
| `neon_rgb_to_ycbcr_row` | Encode-side color conversion | High |
| `neon_downsample_h2v1` | Chroma downsample 2x1 | Medium |
| `neon_downsample_h2v2` | Chroma downsample 2x2 | Medium |
| `neon_idct_4x4` | Scaled IDCT for 1/2 decode | Medium |
| `neon_idct_2x2` | Scaled IDCT for 1/4 decode | Low |
| `neon_idct_1x1` | Scaled IDCT for 1/8 decode | Low |
| `neon_quantize` | Forward quantization | Low |

### SimdRoutines extension

```rust
pub struct SimdRoutines {
    // existing
    pub idct_islow: fn(...),
    pub ycbcr_to_rgb_row: fn(...),
    pub fancy_upsample_h2v1: fn(...),
    // new
    pub fdct_islow: fn(...),
    pub rgb_to_ycbcr_row: fn(...),
    pub downsample_h2v1: fn(...),
    pub downsample_h2v2: fn(...),
    pub idct_4x4: fn(...),
    pub idct_2x2: fn(...),
    pub idct_1x1: fn(...),
}
```

---

## Implementation Order

```
Foundation (infra):
  1. Sample trait + precision field in Image
  2. Encoder builder (minimal: replaces compress() internally)
  3. Decoder builder (extends existing Decoder)
  4. Marker system (COM, density, saved_markers, write_marker)
  5. ErrorHandler + ProgressListener traits

Codec completion:
  6. DRI restart interval encode
  7. Lossless encode extension (color, predictor 1-7, pt)
  8. SOF10 arithmetic progressive encode
  9. SOF11 lossless arithmetic encode/decode
  10. Grayscale-from-color encode
  11. Custom quantization tables
  12. Custom Huffman tables
  13. Custom progressive scan script
  14. DctMethod selection (IsFast, Float)

Format extension:
  15. Additional pixel formats (Rgbx, Bgrx, Xrgb, Xbgr, Argb, Abgr, Rgb565)
  16. S441 subsampling

Advanced API:
  17. Scanline encode API
  18. Scanline decode API
  19. YUV planar encode/decode functions
  20. Buffer size helpers
  21. Transform options (all 9 TJXOPT flags)
  22. Coefficient filter callback
  23. Progressive output (ProgressiveDecoder)

SIMD:
  24. NEON FDCT
  25. NEON RGB→YCbCr
  26. NEON chroma downsample
  27. NEON scaled IDCT variants
```

Each step: TDD (failing test → implement → refactor), commit, update `docs/FEATURE_PARITY.md`.
