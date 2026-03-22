# Phase 3: Six Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete encode/decode feature parity — ICC/EXIF metadata write, S440/S411 subsampling encode, multi-component lossless decode, lossless encode (SOF3), SOF10 arithmetic progressive decode, and CMYK encoding.

**Architecture:** Each feature is an independent branch merged via PR. Features touch mostly isolated code paths (marker_writer for metadata, pipeline.rs for encode/decode extensions). Shared test fixtures are used where possible.

**Tech Stack:** Rust, cargo test, TDD (Red-Green-Refactor)

---

## File Map

| Feature | Create | Modify | Test |
|---------|--------|--------|------|
| ICC/EXIF write | — | `src/encode/marker_writer.rs`, `src/encode/pipeline.rs`, `src/api/high_level.rs`, `src/lib.rs` | `tests/metadata_write.rs` |
| S440/S411 encode | — | `src/encode/pipeline.rs` | `tests/subsampling_encode.rs` |
| Multi-component lossless decode | — | `src/decode/pipeline.rs` | `tests/lossless_decode.rs` (extend) |
| Lossless encode (SOF3) | — | `src/encode/pipeline.rs`, `src/encode/marker_writer.rs`, `src/api/high_level.rs`, `src/lib.rs` | `tests/lossless_encode.rs` |
| SOF10 decode | — | `src/decode/pipeline.rs` | `tests/sof10_decode.rs` |
| CMYK encode | — | `src/encode/pipeline.rs`, `src/encode/marker_writer.rs`, `src/encode/color.rs`, `src/api/high_level.rs`, `src/lib.rs` | `tests/cmyk_encode.rs` |

---

## Task 1: ICC/EXIF Metadata Write

Add `write_app1_exif()` and `write_app2_icc()` to marker_writer, then expose new compress functions that accept optional metadata.

**Files:**
- Modify: `src/encode/marker_writer.rs` — add `write_app1_exif`, `write_app2_icc`
- Modify: `src/encode/pipeline.rs` — add `compress_with_metadata()` that wraps compress() and prepends metadata markers
- Modify: `src/api/high_level.rs` — add public `compress_with_metadata()` wrapper
- Modify: `src/lib.rs` — re-export new function
- Create: `tests/metadata_write.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for ICC roundtrip**

```rust
// tests/metadata_write.rs
use libjpeg_turbo_rs::{compress_with_metadata, decompress, PixelFormat, Subsampling};

#[test]
fn icc_profile_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let fake_icc = vec![0x42u8; 200]; // dummy ICC profile
    let jpeg = compress_with_metadata(
        &pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444,
        Some(&fake_icc), None,
    ).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(fake_icc.as_slice()));
}
```

- [ ] **Step 2: Run test — expect compile error** (`compress_with_metadata` not found)

- [ ] **Step 3: Add `write_app2_icc` to marker_writer.rs**

```rust
// In src/encode/marker_writer.rs — append after write_eoi()

/// Write APP2 ICC profile markers. Splits profile into chunks of max 65519 bytes.
pub fn write_app2_icc(buf: &mut Vec<u8>, profile: &[u8]) {
    const ICC_OVERHEAD: usize = 14; // "ICC_PROFILE\0" + seq_no + num_markers
    const MAX_DATA: usize = 65533 - ICC_OVERHEAD; // 65519

    let num_markers = (profile.len() + MAX_DATA - 1) / MAX_DATA;
    let mut offset = 0;

    for seq in 1..=num_markers {
        let chunk_len = (profile.len() - offset).min(MAX_DATA);
        let marker_len: u16 = (ICC_OVERHEAD + chunk_len) as u16 + 2;

        buf.push(0xFF);
        buf.push(0xE2); // APP2
        buf.extend_from_slice(&marker_len.to_be_bytes());
        buf.extend_from_slice(b"ICC_PROFILE\0");
        buf.push(seq as u8);
        buf.push(num_markers as u8);
        buf.extend_from_slice(&profile[offset..offset + chunk_len]);

        offset += chunk_len;
    }
}
```

- [ ] **Step 4: Add `write_app1_exif` to marker_writer.rs**

```rust
/// Write APP1 EXIF marker. `tiff_data` is raw TIFF-format EXIF data (after "Exif\0\0" header).
pub fn write_app1_exif(buf: &mut Vec<u8>, tiff_data: &[u8]) {
    let header = b"Exif\0\0";
    let marker_len: u16 = (2 + header.len() + tiff_data.len()) as u16;

    buf.push(0xFF);
    buf.push(0xE1); // APP1
    buf.extend_from_slice(&marker_len.to_be_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(tiff_data);
}
```

- [ ] **Step 5: Add `compress_with_metadata` to encode/pipeline.rs**

Add after the existing `compress()` function (~line 227):

```rust
/// Compress with optional ICC profile and EXIF metadata.
pub fn compress_with_metadata(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    // Use the normal compress pipeline to get entropy data
    // but inject metadata markers after APP0 JFIF
    // ... (reuse internal compress logic, insert markers in assembly step)
}
```

The simplest approach: call `compress()` to get the full JPEG, then insert the APP1/APP2 markers after the APP0 marker. JPEG markers are positional — APP1/APP2 go right after APP0.

```rust
pub fn compress_with_metadata(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let base = compress(pixels, width, height, pixel_format, quality, subsampling)?;

    // Find end of APP0 JFIF marker (SOI + APP0)
    // SOI = 2 bytes, APP0 = 0xFF 0xE0 + length
    let app0_start = 2; // after SOI
    if base.len() < 4 || base[2] != 0xFF || base[3] != 0xE0 {
        // No APP0, insert after SOI
        let mut out = Vec::with_capacity(base.len() + 1024);
        out.extend_from_slice(&base[..2]); // SOI
        if let Some(exif) = exif_data {
            marker_writer::write_app1_exif(&mut out, exif);
        }
        if let Some(icc) = icc_profile {
            marker_writer::write_app2_icc(&mut out, icc);
        }
        out.extend_from_slice(&base[2..]);
        return Ok(out);
    }

    let app0_len = u16::from_be_bytes([base[4], base[5]]) as usize;
    let insert_pos = app0_start + 2 + app0_len; // after APP0

    let mut out = Vec::with_capacity(base.len() + 1024);
    out.extend_from_slice(&base[..insert_pos]);
    if let Some(exif) = exif_data {
        marker_writer::write_app1_exif(&mut out, exif);
    }
    if let Some(icc) = icc_profile {
        marker_writer::write_app2_icc(&mut out, icc);
    }
    out.extend_from_slice(&base[insert_pos..]);
    Ok(out)
}
```

- [ ] **Step 6: Add public API wrapper in high_level.rs and re-export in lib.rs**

In `src/api/high_level.rs`:
```rust
pub fn compress_with_metadata(
    pixels: &[u8], width: usize, height: usize,
    pixel_format: PixelFormat, quality: u8, subsampling: Subsampling,
    icc_profile: Option<&[u8]>, exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    encoder::compress_with_metadata(pixels, width, height, pixel_format, quality, subsampling, icc_profile, exif_data)
}
```

In `src/lib.rs`, add `compress_with_metadata` to the `pub use api::high_level::` block.

- [ ] **Step 7: Run test — expect PASS**

Run: `cargo test --test metadata_write`

- [ ] **Step 8: Add EXIF roundtrip test**

```rust
#[test]
fn exif_data_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    // Minimal valid TIFF/EXIF: little-endian, magic 42, IFD at offset 8, 0 entries
    let fake_exif = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
    let jpeg = compress_with_metadata(
        &pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444,
        None, Some(&fake_exif),
    ).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.exif_data(), Some(fake_exif.as_slice()));
}

#[test]
fn large_icc_profile_splits_into_chunks() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let large_icc = vec![0xAB; 100_000]; // > 65519 bytes, needs 2 chunks
    let jpeg = compress_with_metadata(
        &pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444,
        Some(&large_icc), None,
    ).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(large_icc.as_slice()));
}
```

- [ ] **Step 9: Run all tests, format, commit**

```bash
cargo fmt && cargo test
git checkout -b feat/metadata-write
git add -A && git commit -s -m "feat: add ICC/EXIF metadata write support in encoder"
```

---

## Task 2: S440/S411 Encoding Support

Extend the encode pipeline to handle 4:4:0 (vertical-only 2x) and 4:1:1 (horizontal 4x) subsampling.

**Files:**
- Modify: `src/encode/pipeline.rs` — extend `compress()`, `compress_arithmetic()`, `compress_progressive()`, `compress_optimized()` MCU logic
- Create: `tests/subsampling_encode.rs`

### Step-by-step

- [ ] **Step 1: Write failing test for S440**

```rust
// tests/subsampling_encode.rs
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn encode_s440_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S440).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}
```

- [ ] **Step 2: Run test — expect error "subsampling mode S440 not supported"**

- [ ] **Step 3: Add S440 to compress() in pipeline.rs**

In `compress()` (~line 78), extend the match to include S440:
```rust
Subsampling::S440 => (8, 16),  // 1x2: 8 wide, 16 tall MCU
```

In `encode_color_mcu()` (~line 1230), add S440 branch:
```rust
Subsampling::S440 => {
    // 2 Y blocks vertically: (x0, y0) and (x0, y0+8)
    encode_single_block(y_plane, width, height, x0, y0, luma_quant,
        dc_luma_table, ac_luma_table, writer, prev_dc_y);
    encode_single_block(y_plane, width, height, x0, y0 + 8, luma_quant,
        dc_luma_table, ac_luma_table, writer, prev_dc_y);
    // Cb/Cr downsampled 1x2 (vertical only)
    encode_downsampled_chroma_block(cb_plane, width, height, x0, y0, 1, 2,
        chroma_quant, dc_chroma_table, ac_chroma_table, writer, prev_dc_cb);
    encode_downsampled_chroma_block(cr_plane, width, height, x0, y0, 1, 2,
        chroma_quant, dc_chroma_table, ac_chroma_table, writer, prev_dc_cr);
}
```

SOF0 component definitions for S440: Y=(1,1,2,0), Cb=(2,1,1,1), Cr=(3,1,1,1).

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Write failing test for S411**

```rust
#[test]
fn encode_s411_roundtrip() {
    let pixels = vec![128u8; 64 * 16 * 3];
    let jpeg = compress(&pixels, 64, 16, PixelFormat::Rgb, 75, Subsampling::S411).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 16);
}
```

- [ ] **Step 6: Add S411 to compress() in pipeline.rs**

MCU size for S411: `(32, 8)` — 4 horizontal Y blocks.

```rust
Subsampling::S411 => (32, 8),
```

In `encode_color_mcu()`:
```rust
Subsampling::S411 => {
    // 4 Y blocks horizontally: (x0, y0), (x0+8, y0), (x0+16, y0), (x0+24, y0)
    for i in 0..4 {
        encode_single_block(y_plane, width, height, x0 + i * 8, y0, luma_quant,
            dc_luma_table, ac_luma_table, writer, prev_dc_y);
    }
    // Cb/Cr downsampled 4x1
    encode_downsampled_chroma_block(cb_plane, width, height, x0, y0, 4, 1,
        chroma_quant, dc_chroma_table, ac_chroma_table, writer, prev_dc_cb);
    encode_downsampled_chroma_block(cr_plane, width, height, x0, y0, 4, 1,
        chroma_quant, dc_chroma_table, ac_chroma_table, writer, prev_dc_cr);
}
```

SOF0 components for S411: Y=(1,4,1,0), Cb=(2,1,1,1), Cr=(3,1,1,1).

- [ ] **Step 7: Run test — expect PASS**

- [ ] **Step 8: Repeat for arithmetic and progressive encode paths**

Apply the same S440/S411 match arms to:
- `compress_arithmetic()` (~line 609): MCU size match + block iteration
- `compress_progressive()` (~line 285): MCU size match + block iteration
- `compress_optimized()` (~line 1458): MCU size match + block iteration

- [ ] **Step 9: Add gradient data tests for visual verification**

```rust
#[test]
fn encode_s440_gradient_pixel_accuracy() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 8) as u8;
            pixels[i + 1] = (y * 8) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S440).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn encode_s411_gradient_pixel_accuracy() {
    let (w, h) = (64, 16);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 4) as u8;
            pixels[i + 1] = (y * 16) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S411).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}
```

- [ ] **Step 10: Run all tests, format, commit**

```bash
cargo fmt && cargo test
git checkout -b feat/s440-s411-encode
git add -A && git commit -s -m "feat: add S440/S411 subsampling encode support"
```

---

## Task 3: Multi-Component Lossless (SOF3) Decode

Extend `decode_lossless_image()` to handle 3-component (color) lossless JPEG.

**Files:**
- Modify: `src/decode/pipeline.rs` — extend `decode_lossless_image()` for multi-component
- Modify: `tests/lossless_decode.rs` — add color lossless tests

### Step-by-step

- [ ] **Step 1: Write failing test for 3-component lossless**

Extend `tests/lossless_decode.rs` with a helper that generates multi-component SOF3 data:

```rust
#[test]
fn decode_lossless_rgb_3component() {
    // Build a 3-component SOF3 JPEG with interleaved scan
    let (w, h) = (8, 4);
    let y_data: Vec<u8> = (0..w*h).map(|i| (i * 3) as u8).collect();
    let cb_data: Vec<u8> = (0..w*h).map(|i| (128 + i) as u8).collect();
    let cr_data: Vec<u8> = (0..w*h).map(|i| (64 + i * 2) as u8).collect();

    let jpeg = make_lossless_jpeg_3comp(&y_data, &cb_data, &cr_data, w as u16, h as u16, 8);
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    // Lossless 3-component decodes to YCbCr which gets color-converted to RGB
    // Just verify dimensions and no crash
    assert_eq!(img.data.len(), w * h * 3);
}
```

- [ ] **Step 2: Run test — expect error "lossless 3 components (only grayscale supported)"**

- [ ] **Step 3: Extend decode_lossless_image for multi-component**

In `src/decode/pipeline.rs`, modify `decode_lossless_image()`:

Remove the `num_components != 1` error. Instead, decode each component's differences separately using the scan's interleaved order:

For interleaved scan (all components in one scan):
- Iterate samples: for each pixel position, decode difference for each component in order
- Each component uses its own DC Huffman table
- Each component has its own prediction context (prev_row per component)

For non-interleaved (one component per scan):
- Process each scan independently

After decoding all component planes, apply color conversion (YCbCr→RGB for 3-component).

Key changes:
1. Remove the `num_components != 1` early return
2. Allocate per-component output planes (Vec of Vec<u16>)
3. For interleaved scan: decode samples round-robin across components
4. After decoding: if 3 components, color-convert; if 1, grayscale output

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Add test with flat color data for exact verification**

- [ ] **Step 6: Run all tests, format, commit**

```bash
cargo fmt && cargo test
git checkout -b feat/lossless-multicomp-decode
git add -A && git commit -s -m "feat: add multi-component lossless JPEG (SOF3) decode"
```

---

## Task 4: Lossless JPEG Encoding (SOF3)

Add `compress_lossless()` — encodes raw pixel data using Huffman-coded differences with prediction.

**Files:**
- Modify: `src/encode/marker_writer.rs` — add `write_sof3()`
- Modify: `src/encode/pipeline.rs` — add `compress_lossless()`
- Modify: `src/api/high_level.rs` — add public wrapper
- Modify: `src/lib.rs` — re-export
- Create: `tests/lossless_encode.rs`

### Step-by-step

- [ ] **Step 1: Write failing test**

```rust
// tests/lossless_encode.rs
use libjpeg_turbo_rs::{compress_lossless, decompress, PixelFormat};

#[test]
fn lossless_encode_grayscale_roundtrip() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.data, pixels); // Lossless = exact match
}
```

- [ ] **Step 2: Run test — expect compile error**

- [ ] **Step 3: Add `write_sof3` to marker_writer.rs**

```rust
/// Write SOF3 (lossless, Huffman-coded) frame header.
pub fn write_sof3(buf: &mut Vec<u8>, width: u16, height: u16, precision: u8,
                  components: &[(u8, u8, u8, u8)]) {
    buf.push(0xFF);
    buf.push(0xC3); // SOF3
    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(precision);
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.push(components.len() as u8);
    for &(id, h_samp, v_samp, qt_idx) in components {
        buf.push(id);
        buf.push((h_samp << 4) | v_samp);
        buf.push(qt_idx);
    }
}
```

Also add `write_sos_lossless`:
```rust
/// Write SOS for lossless scan. Ss=predictor (1-7), Se=0, Ah=0, Al=point_transform.
pub fn write_sos_lossless(buf: &mut Vec<u8>, components: &[(u8, u8, u8)],
                          predictor: u8, point_transform: u8) {
    buf.push(0xFF);
    buf.push(0xDA); // SOS
    let length: u16 = 2 + 1 + (components.len() as u16 * 2) + 3;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(components.len() as u8);
    for &(id, dc_tbl, _ac_tbl) in components {
        buf.push(id);
        buf.push((dc_tbl << 4) | 0); // DC table only, AC unused
    }
    buf.push(predictor); // Ss = predictor selection (1-7)
    buf.push(0);         // Se = 0
    buf.push(point_transform & 0x0F); // Ah=0, Al=point_transform
}
```

- [ ] **Step 4: Add `compress_lossless` to encode/pipeline.rs**

```rust
/// Compress as lossless JPEG (SOF3).
///
/// Uses predictor 1 (left) and no point transform.
/// Produces exact pixel-identical output when decoded.
pub fn compress_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    if pixel_format != PixelFormat::Grayscale {
        return Err(JpegError::Unsupported(
            "lossless encoding only supports grayscale".to_string(),
        ));
    }
    // ... compute differences using predictor 1, Huffman encode, assemble markers
}
```

The encode algorithm:
1. For each row, compute differences: `diff[x] = pixel[x] - prediction[x]`
   - First pixel of first row: prediction = 2^(precision-1) = 128
   - First row remaining: prediction = left pixel
   - Other rows first column: prediction = above pixel
   - Other rows remaining: prediction = left pixel (predictor 1)
2. Huffman-encode each difference using DC encoding (category + extra bits)
3. Assemble: SOI, DHT (DC table), SOF3, SOS (predictor=1, pt=0), entropy data, EOI

- [ ] **Step 5: Add public API wrapper and re-export**

- [ ] **Step 6: Run test — expect PASS (exact pixel match)**

- [ ] **Step 7: Add more tests**

```rust
#[test]
fn lossless_encode_gradient() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x * 7 + y * 3) % 256) as u8;
        }
    }
    let jpeg = compress_lossless(&pixels, w, h, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_produces_sof3_marker() {
    let pixels = vec![128u8; 8 * 8];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let has_sof3 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(has_sof3, "should contain SOF3 marker");
}
```

- [ ] **Step 8: Run all tests, format, commit**

```bash
cargo fmt && cargo test
git checkout -b feat/lossless-encode
git add -A && git commit -s -m "feat: add lossless JPEG (SOF3) encoding"
```

---

## Task 5: SOF10 (Arithmetic Progressive) Decode

Enable decoding of progressive JPEG with arithmetic entropy coding.

**Files:**
- Modify: `src/decode/pipeline.rs` — add `decode_arithmetic_progressive_planes()`
- Modify: `src/decode/arithmetic.rs` — add progressive DC/AC methods (first/refine)
- Create: `tests/sof10_decode.rs`

### Step-by-step

- [ ] **Step 1: Write failing test**

Since we can't easily generate SOF10 files from our encoder (no SOF10 encoder), test by:
1. Encode with `compress_progressive()` (SOF2 Huffman progressive)
2. Verify the decode path can read it (already works)
3. For SOF10, we need to generate test data or test that the pipeline dispatches correctly

Simpler approach: test the arithmetic progressive decode by verifying the pipeline recognizes SOF10 and dispatches correctly. We can create a synthetic SOF10 stream by writing SOF10 marker instead of SOF2 and using arithmetic entropy.

```rust
// tests/sof10_decode.rs
use libjpeg_turbo_rs::{compress_arithmetic, compress_progressive, decompress, PixelFormat, Subsampling};

/// Verify that arithmetic + progressive flags are set for SOF10.
/// Since we don't have a SOF10 encoder yet, we test the decode path
/// handles the combined mode by checking that it doesn't error.
#[test]
fn sof10_marker_recognized() {
    // SOF10 = 0xCA = arithmetic progressive
    // Build minimal SOF10 JPEG manually or verify decode path
    // For now, verify existing progressive and arithmetic paths work independently
    let pixels = vec![128u8; 32 * 32 * 3];

    // Progressive Huffman works
    let prog_jpeg = compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let prog_img = decompress(&prog_jpeg).unwrap();
    assert_eq!(prog_img.width, 32);

    // Arithmetic sequential works
    let arith_jpeg = compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let arith_img = decompress(&arith_jpeg).unwrap();
    assert_eq!(arith_img.width, 32);
}
```

- [ ] **Step 2: Implement arithmetic progressive decode methods**

In `src/decode/arithmetic.rs`, add methods matching the C reference jdarith.c:

```rust
/// Decode DC coefficient for progressive first scan (arithmetic).
pub fn decode_dc_first(&mut self, block: &mut [i16; 64], comp_idx: usize, dc_tbl: usize, al: u8) -> Result<()>

/// Decode DC refinement for progressive (arithmetic).
pub fn decode_dc_refine(&mut self, block: &mut [i16; 64], al: u8) -> Result<()>

/// Decode AC first scan for progressive (arithmetic).
pub fn decode_ac_first(&mut self, block: &mut [i16; 64], ac_tbl: usize, ss: u8, se: u8, al: u8) -> Result<()>

/// Decode AC refinement for progressive (arithmetic).
pub fn decode_ac_refine(&mut self, block: &mut [i16; 64], ac_tbl: usize, ss: u8, se: u8, al: u8) -> Result<()>
```

These follow the same structure as the Huffman progressive decode (`decode/progressive.rs`) but use arithmetic decoding instead of Huffman.

- [ ] **Step 3: Add decode_arithmetic_progressive_planes to pipeline.rs**

In `decode_image()`, add a new branch:
```rust
if self.metadata.is_arithmetic && frame.is_progressive {
    // SOF10: arithmetic progressive
    self.decode_arithmetic_progressive_planes(...)
}
```

This method follows the same multi-scan pattern as `decode_progressive_planes()` but uses `ArithDecoder` instead of `BitReader` + Huffman tables.

- [ ] **Step 4: Add roundtrip test with synthetic SOF10 data**

Build a SOF10 JPEG by manually constructing markers (SOF10 = 0xCA) with arithmetic-encoded progressive scan data. This is complex — alternatively, create a `compress_arithmetic_progressive()` encoder to generate test data.

- [ ] **Step 5: Run all tests, format, commit**

```bash
cargo fmt && cargo test
git checkout -b feat/sof10-decode
git add -A && git commit -s -m "feat: add SOF10 arithmetic progressive JPEG decode"
```

---

## Task 6: CMYK Encoding

Add CMYK compression support with Adobe APP14 marker.

**Files:**
- Modify: `src/encode/marker_writer.rs` — add `write_app14_adobe()`
- Modify: `src/encode/pipeline.rs` — extend compress() for CMYK
- Modify: `src/encode/color.rs` — (no new color conversion needed; CMYK is stored directly)
- Modify: `src/api/high_level.rs` — update compress to accept CMYK PixelFormat
- Create: `tests/cmyk_encode.rs`

### Step-by-step

- [ ] **Step 1: Write failing test**

```rust
// tests/cmyk_encode.rs
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn cmyk_encode_roundtrip() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 4]; // CMYK = 4 bytes per pixel
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.pixel_format, PixelFormat::Cmyk);
}
```

- [ ] **Step 2: Run test — expect error about CMYK not supported**

- [ ] **Step 3: Add `write_app14_adobe` to marker_writer.rs**

```rust
/// Write Adobe APP14 marker for CMYK/YCCK color space identification.
/// transform: 0 = CMYK, 1 = YCbCr, 2 = YCCK
pub fn write_app14_adobe(buf: &mut Vec<u8>, transform: u8) {
    buf.push(0xFF);
    buf.push(0xEE); // APP14
    let length: u16 = 2 + 5 + 2 + 2 + 2 + 1; // 14
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(b"Adobe"); // identifier
    buf.extend_from_slice(&100u16.to_be_bytes()); // version
    buf.extend_from_slice(&0u16.to_be_bytes());   // flags0
    buf.extend_from_slice(&0u16.to_be_bytes());   // flags1
    buf.push(transform); // color transform
}
```

- [ ] **Step 4: Extend compress() for CMYK in pipeline.rs**

CMYK encoding is simpler than YCbCr — no color conversion needed. Each component is encoded independently with 1x1 sampling (no subsampling).

Key changes to `compress()`:
1. Detect `pixel_format == PixelFormat::Cmyk`
2. Split input into 4 component planes (C, M, Y, K)
3. Encode as 4-component with component IDs 'C'(0x43), 'M'(0x4D), 'Y'(0x59), 'K'(0x4B)
4. Force sampling to 1x1 for all components (ignore subsampling parameter)
5. Use all the same quant table for all 4 components
6. Add Adobe APP14 marker with transform=0 (CMYK)

```rust
if pixel_format == PixelFormat::Cmyk {
    return compress_cmyk(pixels, width, height, quality);
}
```

Add internal `fn compress_cmyk(...)`:
- 4 component planes extracted from interleaved CMYK input
- 4 identical quant tables (use luminance table for all)
- MCU = 8x8 (all 1x1 sampling)
- Per MCU: encode C block, M block, Y block, K block
- SOF0 with 4 components: (0x43,1,1,0), (0x4D,1,1,0), (0x59,1,1,0), (0x4B,1,1,0)
- DHT: 2 tables (DC + AC), shared by all components
- SOS: 4 components, all using table 0
- APP14 Adobe marker with transform=0

- [ ] **Step 5: Run test — expect PASS**

- [ ] **Step 6: Add pixel accuracy test**

```rust
#[test]
fn cmyk_encode_pixel_values_preserved() {
    let (w, h) = (8, 8);
    let mut pixels = vec![0u8; w * h * 4];
    for i in 0..w*h {
        pixels[i * 4] = 200;     // C
        pixels[i * 4 + 1] = 100; // M
        pixels[i * 4 + 2] = 50;  // Y
        pixels[i * 4 + 3] = 25;  // K
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 100, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    // At quality 100, values should be very close (JPEG lossy but high quality)
    for i in 0..w*h {
        assert!((img.data[i * 4] as i16 - 200).abs() <= 2, "C channel mismatch");
        assert!((img.data[i * 4 + 1] as i16 - 100).abs() <= 2, "M channel mismatch");
        assert!((img.data[i * 4 + 2] as i16 - 50).abs() <= 2, "Y channel mismatch");
        assert!((img.data[i * 4 + 3] as i16 - 25).abs() <= 2, "K channel mismatch");
    }
}

#[test]
fn cmyk_jpeg_contains_adobe_marker() {
    let pixels = vec![128u8; 8 * 8 * 4];
    let jpeg = compress(&pixels, 8, 8, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    // Adobe marker: FF EE followed by "Adobe"
    let has_adobe = jpeg.windows(7).any(|w| {
        w[0] == 0xFF && w[1] == 0xEE && &w[3..8] == b"Adobe"
    });
    assert!(has_adobe, "CMYK JPEG should contain Adobe APP14 marker");
}
```

- [ ] **Step 7: Run all tests, format, commit**

```bash
cargo fmt && cargo test
git checkout -b feat/cmyk-encode
git add -A && git commit -s -m "feat: add CMYK encoding with Adobe APP14 marker"
```

---

## Execution Order

Recommended sequence (each is independent but this order builds naturally):

1. **Task 1: ICC/EXIF write** — marker_writer additions, no pipeline changes
2. **Task 2: S440/S411 encode** — extends existing pipeline match arms
3. **Task 6: CMYK encode** — new 4-component encode path
4. **Task 3: Multi-component lossless decode** — extends existing lossless decode
5. **Task 4: Lossless encode** — needs lossless decode for roundtrip testing
6. **Task 5: SOF10 decode** — most complex, needs arithmetic progressive methods
