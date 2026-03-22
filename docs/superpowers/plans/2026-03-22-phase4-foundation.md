# Phase 4: Foundation Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cross-cutting infrastructure (Sample trait, Encoder/Decoder builder, marker system, ErrorHandler/ProgressListener traits) that all subsequent feature phases depend on.

**Architecture:** Bottom-up — lay type system and builder foundations first, then extend marker handling, then add trait-based extensibility. Each task produces a working, tested library with no regressions. Existing public API preserved via backward-compatible wrappers.

**Tech Stack:** Rust, cargo test, TDD (Red-Green-Refactor)

---

## File Map

| Task | Create | Modify | Test |
|------|--------|--------|------|
| 1. Sample trait | `src/common/sample.rs` | `src/common/mod.rs`, `src/common/types.rs`, `src/decode/pipeline.rs` (Image struct) | `tests/sample_trait.rs` |
| 2. Encoder builder | `src/api/encoder.rs` | `src/api/mod.rs`, `src/api/high_level.rs`, `src/lib.rs`, `src/encode/pipeline.rs` | `tests/encoder_builder.rs` |
| 3. Decoder builder | — | `src/decode/pipeline.rs`, `src/api/high_level.rs` | `tests/decoder_builder.rs` |
| 4. Marker system | — | `src/encode/marker_writer.rs`, `src/decode/marker.rs`, `src/decode/pipeline.rs`, `src/common/types.rs` | `tests/marker_system.rs` |
| 5. Traits | `src/common/traits.rs` | `src/common/mod.rs`, `src/lib.rs` | `tests/traits.rs` |

---

## Task 1: Sample Trait + Precision in Image

Add `Sample` trait for multi-precision (8/12/16-bit) and `precision` field to `Image`.

**Files:**
- Create: `src/common/sample.rs`
- Modify: `src/common/mod.rs` — add `pub mod sample;`
- Modify: `src/common/types.rs` — re-export Sample
- Modify: `src/decode/pipeline.rs` — add `precision: u8` to Image struct
- Modify: `src/lib.rs` — re-export Sample trait
- Create: `tests/sample_trait.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for Sample trait**

```rust
// tests/sample_trait.rs
use libjpeg_turbo_rs::Sample;

#[test]
fn u8_sample_constants() {
    assert_eq!(u8::BITS_PER_SAMPLE, 8);
    assert_eq!(u8::MAX_VAL, 255);
    assert_eq!(u8::CENTER, 128);
    assert!(!u8::IS_LOSSLESS_ONLY);
}

#[test]
fn i16_sample_constants() {
    assert_eq!(i16::BITS_PER_SAMPLE, 12);
    assert_eq!(i16::MAX_VAL, 4095);
    assert_eq!(i16::CENTER, 2048);
    assert!(!i16::IS_LOSSLESS_ONLY);
}

#[test]
fn u16_sample_constants() {
    assert_eq!(u16::BITS_PER_SAMPLE, 16);
    assert_eq!(u16::MAX_VAL, 65535);
    assert_eq!(u16::CENTER, 32768);
    assert!(u16::IS_LOSSLESS_ONLY);
}

#[test]
fn sample_clamp() {
    assert_eq!(u8::from_i32_clamped(300), 255);
    assert_eq!(u8::from_i32_clamped(-10), 0);
    assert_eq!(u8::from_i32_clamped(128), 128);
    assert_eq!(i16::from_i32_clamped(5000), 4095);
    assert_eq!(i16::from_i32_clamped(-1), 0);
    assert_eq!(u16::from_i32_clamped(70000), 65535);
}
```

- [ ] **Step 2: Run test — expect compile error** (`Sample` trait not found)

Run: `cargo test --test sample_trait`

- [ ] **Step 3: Implement Sample trait**

```rust
// src/common/sample.rs

/// Marker trait for JPEG sample types (8-bit, 12-bit, 16-bit).
pub trait Sample: Copy + Default + Into<i32> + Send + Sync + 'static {
    /// Bit depth of this sample type.
    const BITS_PER_SAMPLE: u8;
    /// Maximum representable value.
    const MAX_VAL: i32;
    /// Center value (used as initial predictor in lossless).
    const CENTER: i32;
    /// If true, this precision is only valid for lossless JPEG.
    const IS_LOSSLESS_ONLY: bool;

    /// Clamp an i32 to the valid range and convert.
    fn from_i32_clamped(v: i32) -> Self;
}

impl Sample for u8 {
    const BITS_PER_SAMPLE: u8 = 8;
    const MAX_VAL: i32 = 255;
    const CENTER: i32 = 128;
    const IS_LOSSLESS_ONLY: bool = false;

    #[inline]
    fn from_i32_clamped(v: i32) -> Self {
        v.clamp(0, 255) as u8
    }
}

impl Sample for i16 {
    const BITS_PER_SAMPLE: u8 = 12;
    const MAX_VAL: i32 = 4095;
    const CENTER: i32 = 2048;
    const IS_LOSSLESS_ONLY: bool = false;

    #[inline]
    fn from_i32_clamped(v: i32) -> Self {
        v.clamp(0, 4095) as i16
    }
}

impl Sample for u16 {
    const BITS_PER_SAMPLE: u8 = 16;
    const MAX_VAL: i32 = 65535;
    const CENTER: i32 = 32768;
    const IS_LOSSLESS_ONLY: bool = true;

    #[inline]
    fn from_i32_clamped(v: i32) -> Self {
        v.clamp(0, 65535) as u16
    }
}
```

- [ ] **Step 4: Wire up module**

In `src/common/mod.rs`, add: `pub mod sample;`

In `src/lib.rs`, add `Sample` to re-exports:
```rust
pub use common::sample::Sample;
```

- [ ] **Step 5: Run test — expect PASS**

Run: `cargo test --test sample_trait`

- [ ] **Step 6: Add precision field to Image**

In `src/decode/pipeline.rs`, add `precision: u8` field to the `Image` struct after `pixel_format`:

```rust
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    pub precision: u8,  // NEW: 8, 12, or 16
    pub data: Vec<u8>,
    // ... rest unchanged
}
```

Then find every place that constructs `Image { ... }` and add `precision: 8,`. Use grep for `Ok(Image {` to find all construction sites.

- [ ] **Step 7: Run full test suite — expect PASS**

Run: `cargo test`

- [ ] **Step 8: Format and commit**

```bash
cargo fmt --all && cargo test
git checkout -b feat/sample-trait
git add src/common/sample.rs src/common/mod.rs src/decode/pipeline.rs src/lib.rs tests/sample_trait.rs
git commit -s -m "feat: add Sample trait for multi-precision and precision field to Image"
```

---

## Task 2: Encoder Builder

Replace the fragmented `compress*()` functions with an `Encoder` builder. Existing free functions become thin wrappers.

**Files:**
- Create: `src/api/encoder.rs`
- Modify: `src/api/mod.rs` — add `pub mod encoder;`
- Modify: `src/api/high_level.rs` — rewrite compress functions to use Encoder
- Modify: `src/lib.rs` — re-export `Encoder`
- Create: `tests/encoder_builder.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for Encoder builder**

```rust
// tests/encoder_builder.rs
use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn encoder_basic_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encoder_progressive() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(85)
        .progressive(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_with_metadata() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let icc = vec![0x42u8; 100];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .icc_profile(&icc)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
}

#[test]
fn encoder_arithmetic() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_optimized_huffman() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .optimize_huffman(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}
```

- [ ] **Step 2: Run test — expect compile error**

Run: `cargo test --test encoder_builder`

- [ ] **Step 3: Implement Encoder struct**

```rust
// src/api/encoder.rs
use crate::common::error::Result;
use crate::common::types::{PixelFormat, Subsampling};
use crate::encode::pipeline as encoder;

/// JPEG encoder with builder-pattern configuration.
pub struct Encoder<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    optimize_huffman: bool,
    progressive: bool,
    arithmetic: bool,
    icc_profile: Option<&'a [u8]>,
    exif_data: Option<&'a [u8]>,
}

impl<'a> Encoder<'a> {
    pub fn new(pixels: &'a [u8], width: usize, height: usize, pixel_format: PixelFormat) -> Self {
        Self {
            pixels,
            width,
            height,
            pixel_format,
            quality: 75,
            subsampling: Subsampling::S420,
            optimize_huffman: false,
            progressive: false,
            arithmetic: false,
            icc_profile: None,
            exif_data: None,
        }
    }

    pub fn quality(mut self, quality: u8) -> Self { self.quality = quality; self }
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self { self.subsampling = subsampling; self }
    pub fn optimize_huffman(mut self, optimize: bool) -> Self { self.optimize_huffman = optimize; self }
    pub fn progressive(mut self, progressive: bool) -> Self { self.progressive = progressive; self }
    pub fn arithmetic(mut self, arithmetic: bool) -> Self { self.arithmetic = arithmetic; self }
    pub fn icc_profile(mut self, data: &'a [u8]) -> Self { self.icc_profile = Some(data); self }
    pub fn exif_data(mut self, data: &'a [u8]) -> Self { self.exif_data = Some(data); self }

    /// Encode and return the JPEG byte stream.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let has_metadata = self.icc_profile.is_some() || self.exif_data.is_some();

        let base = if self.arithmetic {
            encoder::compress_arithmetic(
                self.pixels, self.width, self.height,
                self.pixel_format, self.quality, self.subsampling,
            )?
        } else if self.progressive {
            encoder::compress_progressive(
                self.pixels, self.width, self.height,
                self.pixel_format, self.quality, self.subsampling,
            )?
        } else if self.optimize_huffman {
            encoder::compress_optimized(
                self.pixels, self.width, self.height,
                self.pixel_format, self.quality, self.subsampling,
            )?
        } else {
            encoder::compress(
                self.pixels, self.width, self.height,
                self.pixel_format, self.quality, self.subsampling,
            )?
        };

        if has_metadata {
            encoder::inject_metadata(&base, self.icc_profile, self.exif_data)
        } else {
            Ok(base)
        }
    }
}
```

- [ ] **Step 4: Extract inject_metadata from compress_with_metadata**

In `src/encode/pipeline.rs`, extract the metadata injection logic from `compress_with_metadata()` into a standalone public function `inject_metadata()`:

```rust
/// Insert APP1/APP2 markers into an existing JPEG byte stream.
pub fn inject_metadata(
    base: &[u8],
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if icc_profile.is_none() && exif_data.is_none() {
        return Ok(base.to_vec());
    }
    // ... (same insertion logic currently in compress_with_metadata)
}
```

Then rewrite `compress_with_metadata()` to call `compress()` + `inject_metadata()`.

- [ ] **Step 5: Wire up module and re-exports**

In `src/api/mod.rs`, add: `pub mod encoder;`

In `src/lib.rs`, add: `pub use api::encoder::Encoder;`

- [ ] **Step 6: Run tests — expect PASS**

Run: `cargo test --test encoder_builder`

- [ ] **Step 7: Verify existing tests still pass**

Run: `cargo test`

All existing `compress()`, `compress_progressive()`, etc. calls must still work unchanged.

- [ ] **Step 8: Format and commit**

```bash
cargo fmt --all && cargo test
git add src/api/encoder.rs src/api/mod.rs src/api/high_level.rs src/encode/pipeline.rs src/lib.rs tests/encoder_builder.rs
git commit -s -m "feat: add Encoder builder with fluent API"
```

---

## Task 3: Decoder Builder Extension

Extend existing `Decoder` with new option methods. No new struct needed — `Decoder` already exists as a builder.

**Files:**
- Modify: `src/decode/pipeline.rs` — add fields and setter methods to Decoder
- Create: `tests/decoder_builder.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for new Decoder options**

```rust
// tests/decoder_builder.rs
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};
use libjpeg_turbo_rs::decode::pipeline::Decoder;

#[test]
fn decoder_max_pixels_rejects_large_image() {
    let pixels = vec![128u8; 64 * 64 * 3];
    let jpeg = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(32 * 32); // limit to 1024 pixels
    let result = decoder.decode_image();
    assert!(result.is_err());
}

#[test]
fn decoder_stop_on_warning() {
    // Verify the setter compiles and is stored
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_stop_on_warning(true);
    let img = decoder.decode_image().unwrap();
    assert_eq!(img.width, 16);
}
```

- [ ] **Step 2: Run test — expect compile error**

Run: `cargo test --test decoder_builder`

- [ ] **Step 3: Add new fields and methods to Decoder**

In `src/decode/pipeline.rs`, add fields to `Decoder` struct:

```rust
pub struct Decoder<'a> {
    // ... existing fields
    stop_on_warning: bool,
    max_pixels: Option<usize>,
    max_memory: Option<usize>,
    scan_limit: Option<u32>,
}
```

Add setter methods:

```rust
pub fn set_stop_on_warning(&mut self, stop: bool) { self.stop_on_warning = stop; }
pub fn set_max_pixels(&mut self, limit: usize) { self.max_pixels = Some(limit); }
pub fn set_max_memory(&mut self, limit: usize) { self.max_memory = Some(limit); }
pub fn set_scan_limit(&mut self, limit: u32) { self.scan_limit = Some(limit); }
```

Add check in `decode_image()` after reading frame header:

```rust
if let Some(max) = self.max_pixels {
    let total = width * height;
    if total > max {
        return Err(JpegError::Unsupported(format!(
            "image {}x{} ({} pixels) exceeds limit of {}",
            width, height, total, max
        )));
    }
}
```

Initialize new fields in `new()`: `stop_on_warning: false, max_pixels: None, max_memory: None, scan_limit: None`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `cargo test --test decoder_builder && cargo test`

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all && cargo test
git add src/decode/pipeline.rs tests/decoder_builder.rs
git commit -s -m "feat: add max_pixels, stop_on_warning, scan_limit to Decoder"
```

---

## Task 4: Marker System (COM, Density, SavedMarkers, write_marker)

Extend marker reading (COM, density, arbitrary markers) and writing (COM, density, arbitrary markers).

**Files:**
- Modify: `src/common/types.rs` — add DensityInfo, DensityUnit, SavedMarker
- Modify: `src/decode/marker.rs` — parse COM, density; save arbitrary markers
- Modify: `src/decode/pipeline.rs` — add comment, density, saved_markers to Image
- Modify: `src/encode/marker_writer.rs` — add write_com, write_dri, write_marker, write_app0_jfif_with_density
- Create: `tests/marker_system.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for COM roundtrip**

```rust
// tests/marker_system.rs
use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn comment_marker_roundtrip() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment("test comment")
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.comment.as_deref(), Some("test comment"));
}
```

- [ ] **Step 2: Run test — expect compile error**

Run: `cargo test --test marker_system`

- [ ] **Step 3: Add types to common/types.rs**

```rust
/// Pixel density information from JFIF marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensityInfo {
    pub unit: DensityUnit,
    pub x: u16,
    pub y: u16,
}

impl Default for DensityInfo {
    fn default() -> Self {
        Self { unit: DensityUnit::Dpi, x: 72, y: 72 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityUnit {
    Unknown,
    Dpi,
    Dpcm,
}

/// A saved JPEG marker (APP or COM).
#[derive(Debug, Clone)]
pub struct SavedMarker {
    pub code: u8,
    pub data: Vec<u8>,
}
```

- [ ] **Step 4: Add fields to Image struct**

In `src/decode/pipeline.rs`, extend `Image`:

```rust
pub struct Image {
    // ... existing fields
    pub comment: Option<String>,
    pub density: DensityInfo,
    pub saved_markers: Vec<SavedMarker>,
}
```

Update all `Image { ... }` construction sites to include `comment: None, density: DensityInfo::default(), saved_markers: Vec::new()`.

- [ ] **Step 5: Parse COM marker in MarkerReader**

In `src/decode/marker.rs`, in the main marker loop where COM is currently skipped, add:

```rust
0xFE => { // COM
    let length = self.read_u16()? as usize;
    if length >= 2 {
        let data = self.data[self.pos..self.pos + length - 2].to_vec();
        self.pos += length - 2;
        comment = Some(String::from_utf8_lossy(&data).into_owned());
    }
}
```

Add `comment: Option<String>` field to `JpegMetadata`. Pass it through to `Image` in `decode_image()`.

- [ ] **Step 6: Parse density from JFIF APP0**

In the existing `read_app0()` (or wherever JFIF is parsed), extract density fields:

```rust
let density_unit = match data[7] {
    0 => DensityUnit::Unknown,
    1 => DensityUnit::Dpi,
    2 => DensityUnit::Dpcm,
    _ => DensityUnit::Unknown,
};
let x_density = u16::from_be_bytes([data[8], data[9]]);
let y_density = u16::from_be_bytes([data[10], data[11]]);
```

Store in `JpegMetadata.density: DensityInfo`.

- [ ] **Step 7: Add write_com and write_dri to marker_writer**

```rust
/// Write COM (comment) marker.
pub fn write_com(buf: &mut Vec<u8>, text: &str) {
    buf.push(0xFF);
    buf.push(0xFE); // COM
    let length: u16 = (2 + text.len()) as u16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(text.as_bytes());
}

/// Write DRI (restart interval) marker.
pub fn write_dri(buf: &mut Vec<u8>, interval: u16) {
    buf.push(0xFF);
    buf.push(0xDD); // DRI
    buf.extend_from_slice(&4u16.to_be_bytes()); // length = 4
    buf.extend_from_slice(&interval.to_be_bytes());
}

/// Write arbitrary marker.
pub fn write_marker(buf: &mut Vec<u8>, code: u8, data: &[u8]) {
    buf.push(0xFF);
    buf.push(code);
    let length: u16 = (2 + data.len()) as u16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(data);
}

/// Write APP0 JFIF with configurable density.
pub fn write_app0_jfif_with_density(buf: &mut Vec<u8>, density: &DensityInfo) {
    buf.push(0xFF);
    buf.push(0xE0);
    let length: u16 = 16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(b"JFIF\0");
    buf.push(1); buf.push(1); // version 1.01
    buf.push(match density.unit {
        DensityUnit::Unknown => 0,
        DensityUnit::Dpi => 1,
        DensityUnit::Dpcm => 2,
    });
    buf.extend_from_slice(&density.x.to_be_bytes());
    buf.extend_from_slice(&density.y.to_be_bytes());
    buf.push(0); buf.push(0); // no thumbnail
}
```

- [ ] **Step 8: Add comment() and density() to Encoder**

In `src/api/encoder.rs`:

```rust
// New fields
comment: Option<&'a str>,
density: Option<DensityInfo>,

// Setters
pub fn comment(mut self, text: &'a str) -> Self { self.comment = Some(text); self }
pub fn density(mut self, info: DensityInfo) -> Self { self.density = Some(info); self }
```

In `encode()`, after building the base JPEG, inject COM marker (similar to metadata injection).

- [ ] **Step 9: Add more tests**

```rust
#[test]
fn density_roundtrip() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .density(DensityInfo { unit: DensityUnit::Dpi, x: 300, y: 300 })
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.density.x, 300);
    assert_eq!(img.density.y, 300);
    assert_eq!(img.density.unit, DensityUnit::Dpi);
}

#[test]
fn write_dri_marker() {
    let mut buf = Vec::new();
    libjpeg_turbo_rs::encode::marker_writer::write_dri(&mut buf, 100);
    assert_eq!(buf[0], 0xFF);
    assert_eq!(buf[1], 0xDD);
    assert_eq!(u16::from_be_bytes([buf[4], buf[5]]), 100);
}
```

- [ ] **Step 10: Run all tests, format, commit**

```bash
cargo fmt --all && cargo test
git add -A
git commit -s -m "feat: add COM marker, density, saved_markers, write_dri, write_marker"
```

---

## Task 5: ErrorHandler + ProgressListener Traits

Add trait-based extensibility for error handling and progress monitoring.

**Files:**
- Create: `src/common/traits.rs`
- Modify: `src/common/mod.rs` — add `pub mod traits;`
- Modify: `src/lib.rs` — re-export traits
- Create: `tests/traits.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for ErrorHandler**

```rust
// tests/traits.rs
use libjpeg_turbo_rs::{ErrorHandler, ProgressListener, ProgressInfo};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

struct CountingHandler {
    warning_count: AtomicU32,
}

impl ErrorHandler for CountingHandler {
    fn emit_warning(&self, _warning: &libjpeg_turbo_rs::DecodeWarning) {
        self.warning_count.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn custom_error_handler_compiles() {
    let handler = CountingHandler { warning_count: AtomicU32::new(0) };
    handler.emit_warning(&libjpeg_turbo_rs::DecodeWarning::TruncatedData {
        decoded_mcus: 0,
        total_mcus: 10,
    });
    assert_eq!(handler.warning_count.load(Ordering::Relaxed), 1);
}

#[test]
fn progress_closure_compiles() {
    let call_count = Arc::new(AtomicU32::new(0));
    let count = call_count.clone();
    let listener = move |_info: ProgressInfo| {
        count.fetch_add(1, Ordering::Relaxed);
    };
    listener(ProgressInfo { pass: 0, total_passes: 1, progress: 0.5 });
    assert_eq!(call_count.load(Ordering::Relaxed), 1);
}
```

- [ ] **Step 2: Run test — expect compile error**

Run: `cargo test --test traits`

- [ ] **Step 3: Implement traits**

```rust
// src/common/traits.rs
use crate::common::error::{DecodeWarning, JpegError};

/// Customizable error handling for JPEG operations.
pub trait ErrorHandler: Send + Sync {
    /// Called on fatal error. Default: panic.
    fn error_exit(&self, err: &JpegError) -> ! {
        panic!("JPEG fatal error: {err}");
    }

    /// Called on non-fatal warning. Default: ignore.
    fn emit_warning(&self, _warning: &DecodeWarning) {}

    /// Called for trace/debug messages. Default: ignore.
    fn trace(&self, _level: u8, _msg: &str) {}
}

/// Default error handler that does nothing special (Result-based flow).
pub struct DefaultErrorHandler;
impl ErrorHandler for DefaultErrorHandler {}

/// Progress information for encode/decode operations.
#[derive(Debug, Clone, Copy)]
pub struct ProgressInfo {
    /// Current pass (0-based).
    pub pass: u32,
    /// Total number of passes.
    pub total_passes: u32,
    /// Progress within current pass (0.0 to 1.0).
    pub progress: f32,
}

/// Listener for encode/decode progress updates.
pub trait ProgressListener: Send + Sync {
    fn update(&self, info: ProgressInfo);
}

/// Allow closures as ProgressListener.
impl<F: Fn(ProgressInfo) + Send + Sync> ProgressListener for F {
    fn update(&self, info: ProgressInfo) {
        self(info);
    }
}
```

- [ ] **Step 4: Wire up module and re-exports**

In `src/common/mod.rs`, add: `pub mod traits;`

In `src/lib.rs`, add:
```rust
pub use common::traits::{ErrorHandler, DefaultErrorHandler, ProgressInfo, ProgressListener};
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `cargo test --test traits && cargo test`

- [ ] **Step 6: Format and commit**

```bash
cargo fmt --all && cargo test
git add src/common/traits.rs src/common/mod.rs src/lib.rs tests/traits.rs
git commit -s -m "feat: add ErrorHandler and ProgressListener traits"
```

---

## Execution Order

Each task depends on the previous:

```
Task 1 (Sample trait) → standalone, no deps
Task 2 (Encoder builder) → needs Task 1 for precision field awareness
Task 3 (Decoder builder) → standalone from Task 2, but better after
Task 4 (Marker system) → needs Task 2 (Encoder.comment/density)
Task 5 (Traits) → standalone, but placed last to avoid blocking
```

After all 5 tasks: the foundation is complete and Phase 5 (Codec Completion) can begin.
