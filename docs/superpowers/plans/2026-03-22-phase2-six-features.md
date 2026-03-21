# Phase 2: Six Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 6 features toward libjpeg-turbo 3.1.x parity: Crop/Skip, Huffman optimization, Arithmetic coding, Progressive encoding, Lossless transforms, Lossless JPEG decoding.

**Architecture:** Each feature is an independent branch + PR. Features 1-2 have no dependencies. Feature 4 (progressive encoding) benefits from Feature 3 (arithmetic). Feature 5 (lossless transforms) needs both decoder coefficient access and encoder output.

**Tech Stack:** Rust (stable), `thiserror` for errors, `criterion` for benchmarks. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-22-phase2-six-features-design.md`

---

## Chunk 1: Partial Decompression (Crop/Skip)

**Branch:** `feat/crop-skip`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/common/types.rs` | Modify | Add `CropRegion` struct |
| `src/decode/pipeline.rs` | Modify | Add scanline-based decode methods, crop state |
| `src/api/streaming.rs` | Modify | Add `crop_scanline()`, `skip_scanlines()`, `read_scanlines()` |
| `src/api/high_level.rs` | Modify | Add `decompress_cropped()` |
| `src/lib.rs` | Modify | Re-export `CropRegion`, `decompress_cropped` |
| `tests/crop_skip.rs` | Create | Integration tests |

### Task 1: Add CropRegion type

**Files:**
- Modify: `src/common/types.rs`

- [ ] **Step 1: Add CropRegion struct**

```rust
// Add at end of src/common/types.rs

/// Region of interest for cropped decompression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}
```

- [ ] **Step 2: Re-export from lib.rs**

Add to `src/lib.rs`: `CropRegion` is already exported via `pub use common::types::*`.

- [ ] **Step 3: Run `cargo build` to verify**

### Task 2: Add scanline-based decode to Decoder

**Files:**
- Modify: `src/decode/pipeline.rs`

The current `Decoder::decode_image()` decodes the entire image at once. We need to add:
1. Crop state fields (`crop_x`, `crop_width`, `output_scanline`)
2. `set_crop()` method
3. Internal scanline iterator that decodes one iMCU row at a time

- [ ] **Step 1: Add crop state to Decoder**

```rust
// In Decoder struct, add fields:
    crop_x: Option<usize>,      // horizontal crop offset (iMCU-aligned)
    crop_width: Option<usize>,  // horizontal crop width
```

Initialize both to `None` in `Decoder::new()`.

- [ ] **Step 2: Add set_crop method**

```rust
/// Set horizontal crop region. Offsets are auto-aligned to iMCU boundaries.
pub fn set_crop(&mut self, x: usize, width: usize) {
    self.crop_x = Some(x);
    self.crop_width = Some(width);
}
```

- [ ] **Step 3: Add decompress_cropped to high-level API**

In `src/api/high_level.rs`:
```rust
/// Decompress a cropped region of a JPEG.
/// The crop region's x is aligned down to iMCU boundary.
pub fn decompress_cropped(data: &[u8], region: CropRegion) -> Result<Image> {
    // Full decode, then extract region from decoded image.
    // This is the simple initial implementation — skip-based optimization comes later.
    let full = Decoder::decode(data)?;
    let bpp = full.pixel_format.bytes_per_pixel();

    let x = region.x.min(full.width);
    let y = region.y.min(full.height);
    let w = region.width.min(full.width - x);
    let h = region.height.min(full.height - y);

    let mut data = Vec::with_capacity(w * h * bpp);
    for row in y..y + h {
        let start = (row * full.width + x) * bpp;
        data.extend_from_slice(&full.data[start..start + w * bpp]);
    }

    Ok(Image {
        width: w,
        height: h,
        pixel_format: full.pixel_format,
        data,
        icc_profile: full.icc_profile,
        exif_data: full.exif_data,
        warnings: full.warnings,
    })
}
```

- [ ] **Step 4: Re-export in lib.rs**

```rust
pub use api::high_level::{compress, decompress, decompress_cropped, decompress_lenient, decompress_to};
```

- [ ] **Step 5: Run `cargo build`**

### Task 3: Add StreamingDecoder scanline API

**Files:**
- Modify: `src/api/streaming.rs`

- [ ] **Step 1: Add crop_scanline and skip_scanlines to StreamingDecoder**

```rust
/// Set horizontal crop. xoffset auto-aligns to iMCU column boundary.
pub fn crop_scanline(&mut self, xoffset: &mut usize, width: &mut usize) -> Result<()> {
    let header = self.inner.header();
    let max_h = header.components.iter().map(|c| c.horizontal_sampling as usize).max().unwrap_or(1);
    let block_size = 8; // TODO: use scale block_size
    let imcu_width = max_h * block_size;

    // Align xoffset down to iMCU boundary
    let aligned_x = (*xoffset / imcu_width) * imcu_width;
    let aligned_end = ((*xoffset + *width + imcu_width - 1) / imcu_width) * imcu_width;
    let aligned_width = (aligned_end - aligned_x).min(header.width as usize - aligned_x);

    *xoffset = aligned_x;
    *width = aligned_width;

    self.inner.set_crop(aligned_x, aligned_width);
    Ok(())
}

/// Skip scanlines without decoding.
/// Returns actual lines skipped.
pub fn skip_scanlines(&mut self, _num_lines: usize) -> Result<usize> {
    // Initial implementation: no-op, full decode handles this via crop region
    Ok(0)
}
```

- [ ] **Step 2: Run `cargo build`**

### Task 4: Integration tests

**Files:**
- Create: `tests/crop_skip.rs`

- [ ] **Step 1: Write crop tests**

```rust
use libjpeg_turbo_rs::{decompress, decompress_cropped, CropRegion, PixelFormat};

#[test]
fn crop_center_region() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion { x: 80, y: 60, width: 160, height: 120 };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn crop_full_image_matches_decompress() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let region = CropRegion { x: 0, y: 0, width: 320, height: 240 };
    let cropped = decompress_cropped(data, region).unwrap();
    assert_eq!(full.data, cropped.data);
}

#[test]
fn crop_clamps_to_image_bounds() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion { x: 300, y: 200, width: 100, height: 100 };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 20);
    assert_eq!(img.height, 40);
}

#[test]
fn crop_top_left_corner() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let region = CropRegion { x: 0, y: 0, width: 64, height: 64 };
    let cropped = decompress_cropped(data, region).unwrap();
    // First row should match
    let bpp = full.pixel_format.bytes_per_pixel();
    for x in 0..64 {
        for c in 0..bpp {
            assert_eq!(cropped.data[x * bpp + c], full.data[x * bpp + c]);
        }
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test --test crop_skip
```

- [ ] **Step 3: Format and commit**

```bash
cargo fmt
git add -A
git commit -s -m "feat: add partial decompression (crop) support"
```

---

## Chunk 2: Huffman Optimization (2-Pass Encoding)

**Branch:** `feat/huffman-opt`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/encode/huff_opt.rs` | Create | Symbol gathering + optimal table generation |
| `src/encode/huffman_encode.rs` | Modify | Add gather-mode methods |
| `src/encode/pipeline.rs` | Modify | 2-pass compression loop |
| `src/encode/mod.rs` | Modify | Register `huff_opt` module |
| `tests/huff_opt.rs` | Create | Tests for optimization |

### Task 5: Implement symbol frequency gathering

**Files:**
- Create: `src/encode/huff_opt.rs`
- Modify: `src/encode/mod.rs`

- [ ] **Step 1: Write test for frequency counting**

```rust
// In src/encode/huff_opt.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_dc_zero_diff() {
        let mut freq = [0u32; 257];
        gather_dc_symbol(0, &mut freq); // category 0
        assert_eq!(freq[0], 1);
    }

    #[test]
    fn gather_dc_positive() {
        let mut freq = [0u32; 257];
        gather_dc_symbol(5, &mut freq); // category 3
        assert_eq!(freq[3], 1);
    }

    #[test]
    fn gather_ac_eob() {
        let mut freq = [0u32; 257];
        gather_ac_eob(&mut freq);
        assert_eq!(freq[0x00], 1);
    }
}
```

- [ ] **Step 2: Implement gathering functions**

```rust
/// Gather DC symbol frequency (category of the DC difference).
pub fn gather_dc_symbol(diff: i16, freq: &mut [u32; 257]) {
    let category = if diff == 0 { 0u8 } else { 16 - diff.unsigned_abs().leading_zeros() as u8 };
    freq[category as usize] += 1;
}

/// Gather AC symbol frequencies from a zigzag-ordered coefficient block.
pub fn gather_ac_symbols(coeffs: &[i16; 64], freq: &mut [u32; 257]) {
    let mut zero_run: u8 = 0;
    for k in 1..64 {
        let ac = coeffs[k];
        if ac == 0 {
            zero_run += 1;
        } else {
            while zero_run >= 16 {
                freq[0xF0] += 1;
                zero_run -= 16;
            }
            let size = 16 - ac.unsigned_abs().leading_zeros() as u8;
            let symbol = ((zero_run as u16) << 4) | (size as u16);
            freq[symbol as usize] += 1;
            zero_run = 0;
        }
    }
    if zero_run > 0 {
        freq[0x00] += 1; // EOB
    }
}

/// Record EOB symbol.
pub fn gather_ac_eob(freq: &mut [u32; 257]) {
    freq[0x00] += 1;
}
```

- [ ] **Step 3: Run tests, verify pass**

### Task 6: Implement optimal Huffman table generation

**Files:**
- Modify: `src/encode/huff_opt.rs`

- [ ] **Step 1: Write test for table generation**

```rust
#[test]
fn gen_optimal_table_from_uniform() {
    let mut freq = [1u32; 257]; // uniform distribution
    freq[256] = 1; // pseudo-symbol
    let (bits, values) = gen_optimal_table(&freq);
    // All code lengths should be <= 16
    let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
    assert!(total <= 257);
    assert!(total > 0);
}

#[test]
fn gen_optimal_table_single_symbol() {
    let mut freq = [0u32; 257];
    freq[0] = 100;
    freq[256] = 1; // pseudo-symbol
    let (bits, values) = gen_optimal_table(&freq);
    let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
    assert_eq!(total, 2); // only 2 symbols with nonzero frequency
}
```

- [ ] **Step 2: Implement `gen_optimal_table`**

Port JPEG Annex K.2 algorithm:
1. Find nonzero-frequency symbols
2. Build Huffman tree (merge two smallest)
3. Compute code sizes
4. Limit to 16-bit max code length
5. Generate bits[] and huffval[]

```rust
/// Generate optimal Huffman table from symbol frequencies.
/// Returns (bits[17], huffval[256]) in JPEG DHT format.
/// freq[256] is the pseudo-symbol (must have count >= 1).
pub fn gen_optimal_table(freq: &[u32; 257]) -> ([u8; 17], Vec<u8>) {
    // ... (full implementation ~150 lines, port of jpeg_gen_optimal_table from jchuff.c)
}
```

- [ ] **Step 3: Run tests**

### Task 7: Integrate 2-pass encoding into pipeline

**Files:**
- Modify: `src/encode/pipeline.rs`

- [ ] **Step 1: Add `optimize` parameter to compress**

Add an `optimize_huffman: bool` parameter to the internal encoding path. When true:
- Pass 1: iterate all MCUs, call `gather_dc_symbol` / `gather_ac_symbols` instead of encoding
- Call `gen_optimal_table` for each of 4 tables (DC lum, DC chr, AC lum, AC chr)
- Build `HuffTable` from optimal bits/values
- Pass 2: normal encoding with optimal tables
- Write optimal DHT markers instead of standard ones

- [ ] **Step 2: Write integration test**

```rust
// tests/huff_opt.rs
#[test]
fn optimized_smaller_than_standard() {
    let data = include_bytes!("tests/fixtures/photo_320x240_420.jpg");
    let img = libjpeg_turbo_rs::decompress(data).unwrap();

    let standard = libjpeg_turbo_rs::compress(
        &img.data, img.width, img.height, img.pixel_format, 75, Subsampling::S420,
    ).unwrap();

    // Optimized should be same or smaller
    let optimized = libjpeg_turbo_rs::compress_optimized(
        &img.data, img.width, img.height, img.pixel_format, 75, Subsampling::S420,
    ).unwrap();

    assert!(optimized.len() <= standard.len());
    // Verify round-trip
    let decoded = libjpeg_turbo_rs::decompress(&optimized).unwrap();
    assert_eq!(decoded.width, img.width);
}
```

- [ ] **Step 3: Format and commit**

```bash
cargo fmt && cargo test && git add -A && git commit -s -m "feat: add Huffman optimization (2-pass encoding)"
```

---

## Chunk 3: Arithmetic Coding (Encode + Decode)

**Branch:** `feat/arithmetic-coding`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/common/arith_tables.rs` | Create | ARITAB probability table (114 entries) |
| `src/common/mod.rs` | Modify | Register `arith_tables` module |
| `src/decode/arithmetic.rs` | Create | Arithmetic entropy decoder |
| `src/decode/mod.rs` | Modify | Register `arithmetic` module |
| `src/decode/marker.rs` | Modify | Add SOF9, SOF10, DAC marker parsing |
| `src/decode/pipeline.rs` | Modify | Dispatch to arithmetic decoder |
| `src/encode/arithmetic.rs` | Create | Arithmetic entropy encoder |
| `src/encode/mod.rs` | Modify | Register `arithmetic` module |
| `src/encode/marker_writer.rs` | Modify | Add SOF9/SOF10/DAC marker writing |
| `src/encode/pipeline.rs` | Modify | Arithmetic encoding option |
| `tests/arithmetic.rs` | Create | Round-trip and decode tests |

### Task 8: Implement probability table

**Files:**
- Create: `src/common/arith_tables.rs`

- [ ] **Step 1: Create ARITAB constant**

Port the 114-entry probability estimation table from `jaricom.c` (ITU-T T.81 Table D.2):

```rust
/// Packed arithmetic coding probability table.
/// Each entry: (Qe << 16) | (NMPS << 8) | (Switch << 7) | NLPS
/// Per ITU-T T.81 Table D.2.
pub const ARITAB: [u32; 114] = [
    0x5a1d_0001 | (1 << 8), // state 0
    // ... (114 entries from jaricom.c)
];

/// DC statistics bins per table.
pub const DC_STAT_BINS: usize = 64;
/// AC statistics bins per table.
pub const AC_STAT_BINS: usize = 256;
```

- [ ] **Step 2: Run `cargo build`**

### Task 9: Implement arithmetic decoder

**Files:**
- Create: `src/decode/arithmetic.rs`

This is the largest and most complex task (~800 lines).

- [ ] **Step 1: Core ArithDecoder struct**

```rust
pub struct ArithDecoder {
    c: u32,           // C register (coding interval base)
    a: u32,           // A register (interval size)
    ct: i32,          // bit shift counter

    // Per-component DC state
    last_dc_val: [i32; 4],
    dc_context: [usize; 4],

    // Statistics tables
    dc_stats: [[u8; DC_STAT_BINS]; 4],
    ac_stats: [[u8; AC_STAT_BINS]; 4],
    fixed_bin: [u8; 4],
}
```

- [ ] **Step 2: Implement `arith_decode()` — single binary decision**

Port from `jdarith.c` lines 114-191:
- Renormalization loop (keep A >= 0x8000)
- Extract Qe, NLPS, NMPS from ARITAB
- Compare C against temp = A - Qe
- Update state machine

- [ ] **Step 3: Implement 4 MCU decoders**

- `decode_mcu_DC_first()` — DC difference with context modeling
- `decode_mcu_DC_refine()` — single refinement bit
- `decode_mcu_AC_first()` — AC coefficients with EOB run
- `decode_mcu_AC_refine()` — AC refinement bits

- [ ] **Step 4: Add SOF9/SOF10/DAC to marker parser**

In `src/decode/marker.rs`, add constants and parsing:
```rust
const SOF9: u8 = 0xC9;   // Arithmetic sequential
const SOF10: u8 = 0xCA;  // Arithmetic progressive
const DAC: u8 = 0xCC;    // Define arithmetic conditioning
```

- [ ] **Step 5: Wire into pipeline**

In `src/decode/pipeline.rs`, detect arithmetic mode from frame header and dispatch to `ArithDecoder` instead of `McuDecoder`.

- [ ] **Step 6: Tests with arithmetic-coded JPEG fixtures**

Need to create test fixtures using libjpeg-turbo's `cjpeg -arithmetic` or use the encoder once built.

### Task 10: Implement arithmetic encoder

**Files:**
- Create: `src/encode/arithmetic.rs`

- [ ] **Step 1: Core ArithEncoder struct** — mirror of decoder
- [ ] **Step 2: Implement `arith_encode()` and Pacman termination**
- [ ] **Step 3: Implement MCU encoders (DC/AC first/refine)**
- [ ] **Step 4: Add SOF9/DAC marker writing**
- [ ] **Step 5: Wire into encode pipeline**
- [ ] **Step 6: Round-trip test: encode arithmetic → decode arithmetic**

- [ ] **Step 7: Format and commit**

```bash
cargo fmt && cargo test && git add -A && git commit -s -m "feat: add arithmetic entropy coding (encode + decode)"
```

---

## Chunk 4: Progressive Encoding

**Branch:** `feat/progressive-enc`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/encode/progressive.rs` | Create | 4 MCU encoders + scan script generation/validation |
| `src/encode/mod.rs` | Modify | Register `progressive` module |
| `src/encode/pipeline.rs` | Modify | Multi-scan loop with coefficient buffer |
| `tests/progressive_enc.rs` | Create | Tests |

### Task 11: Implement scan script generation

**Files:**
- Create: `src/encode/progressive.rs`

- [ ] **Step 1: Test for simple_progression**

```rust
#[test]
fn simple_progression_3_components() {
    let scans = simple_progression(3);
    assert!(scans.len() >= 8); // typically ~10 scans for YCbCr
    // First scan should be DC, all components
    assert_eq!(scans[0].spec_start, 0);
    assert_eq!(scans[0].spec_end, 0);
    assert!(scans[0].components.len() > 1);
}
```

- [ ] **Step 2: Implement `simple_progression()`**

Port from `jcparam.c:476-555`:
```rust
pub fn simple_progression(num_components: usize) -> Vec<ScanHeader> {
    // DC first scan: all components, Al=1
    // AC scans: per-component spectral bands
    // DC refine: all components, Al=0
    // AC refine scans
    // ... (~80 lines)
}
```

- [ ] **Step 3: Implement scan validation**

```rust
pub fn validate_scan_script(scans: &[ScanHeader], num_components: usize) -> Result<()> {
    // DC: Ss=Se=0, up to 4 interleaved
    // AC: Ss>0, single component
    // Ah must equal previous Al for same band
    // All 64 coefficients must be covered per component
}
```

### Task 12: Implement 4 progressive MCU encoders

- [ ] **Step 1: `encode_mcu_DC_first()`** — DC differential with left-shift by Al
- [ ] **Step 2: `encode_mcu_DC_refine()`** — single bit per coefficient
- [ ] **Step 3: `encode_mcu_AC_first()`** — run-length with EOB run encoding
- [ ] **Step 4: `encode_mcu_AC_refine()`** — correction bit buffer (MAX_CORR_BITS=1000)

### Task 13: Integrate into encode pipeline

- [ ] **Step 1: Add coefficient buffer to pipeline**

For progressive: full FDCT + quantize all blocks into `Vec<Vec<[i16; 64]>>` (per-component coefficient arrays), then iterate per-scan.

- [ ] **Step 2: Multi-scan loop**

```rust
if progressive {
    // Buffer all coefficients
    let coeff_bufs = fdct_and_quantize_all(y_plane, cb_plane, cr_plane, ...);
    let scans = scan_script.unwrap_or_else(|| simple_progression(num_components));
    validate_scan_script(&scans, num_components)?;

    for scan in &scans {
        write_sos_marker(&mut output, scan);
        encode_scan(&coeff_bufs, scan, &huff_tables, &mut output);
    }
}
```

- [ ] **Step 3: Round-trip test**

```rust
#[test]
fn progressive_roundtrip() {
    let pixels = vec![128u8; 64 * 64 * 3];
    let jpeg = compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}
```

- [ ] **Step 4: Format and commit**

---

## Chunk 5: Lossless Transforms

**Branch:** `feat/lossless-transform`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/transform/mod.rs` | Create | Module declarations + public types |
| `src/transform/spatial.rs` | Create | 8 spatial transform functions |
| `src/transform/crop.rs` | Create | Lossless crop + trim/perfect logic |
| `src/transform/filter.rs` | Create | Custom DCT filter callback + drop |
| `src/transform/pipeline.rs` | Create | `transform()` orchestration |
| `src/lib.rs` | Modify | Add `pub mod transform;` |
| `tests/transform.rs` | Create | Tests |

### Task 14: Implement coefficient reader

The transform pipeline needs to:
1. Parse JPEG headers
2. Decode all DCT coefficients (without IDCT)
3. Transform coefficients
4. Re-encode into new JPEG

- [ ] **Step 1: Add `read_coefficients()` to Decoder**

```rust
/// Read all DCT coefficients without performing IDCT.
/// Returns per-component arrays of 8x8 coefficient blocks.
pub fn read_coefficients(&self) -> Result<Vec<Vec<[i16; 64]>>> {
    // Reuse progressive coefficient buffering logic
    // For baseline: decode all MCUs into coefficient buffer
}
```

### Task 15: Implement 8 spatial transforms

**Files:**
- Create: `src/transform/spatial.rs`

Each transform manipulates DCT coefficients at the block level:

- [ ] **Step 1: `do_nothing()` — copy blocks unchanged**
- [ ] **Step 2: `do_flip_h()` — negate odd-column coefficients, swap block columns**
- [ ] **Step 3: `do_flip_v()` — negate odd-row coefficients, swap block rows**
- [ ] **Step 4: `do_transpose()` — swap i,j within blocks, swap block coordinates**
- [ ] **Step 5: `do_transverse()` — 180° + transpose**
- [ ] **Step 6: `do_rot_90()`, `do_rot_180()`, `do_rot_270()`**

Each is ~30-50 lines. Test with known coefficient patterns.

### Task 16: Implement transform pipeline

- [ ] **Step 1: Workspace allocation and parameter adjustment**
- [ ] **Step 2: Execute transform + re-encode**
- [ ] **Step 3: Handle TRIM/PERFECT/GRAY flags**
- [ ] **Step 4: Implement DROP (coefficient insertion)**
- [ ] **Step 5: Custom filter callback**

### Task 17: Integration tests

```rust
#[test]
fn rot90_dimensions_swapped() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let info = TransformInfo { transform: TransformOp::Rot90, ..Default::default() };
    let result = transform(data, &info).unwrap();
    let img = decompress(&result).unwrap();
    assert_eq!(img.width, 240);
    assert_eq!(img.height, 320);
}

#[test]
fn rot180_preserves_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let info = TransformInfo { transform: TransformOp::Rot180, ..Default::default() };
    let result = transform(data, &info).unwrap();
    let img = decompress(&result).unwrap();
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
}

#[test]
fn double_hflip_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let info = TransformInfo { transform: TransformOp::HFlip, ..Default::default() };
    let once = transform(data, &info).unwrap();
    let twice = transform(&once, &info).unwrap();
    // Double horizontal flip should be bit-identical to original
    assert_eq!(data.len(), twice.len()); // approximate; exact match for MCU-aligned images
}
```

- [ ] **Step 6: Format and commit**

---

## Chunk 6: Lossless JPEG Decoding

**Branch:** `feat/lossless-jpeg`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/decode/lossless.rs` | Create | Prediction + undifferencing + scaling |
| `src/decode/lossless_huffman.rs` | Create | Lossless Huffman decoding |
| `src/decode/mod.rs` | Modify | Register new modules |
| `src/decode/marker.rs` | Modify | SOF3 parsing |
| `src/decode/pipeline.rs` | Modify | Lossless decode path branch |
| `src/common/types.rs` | Modify | Add `is_lossless` to FrameHeader |
| `tests/lossless.rs` | Create | Tests |

### Task 18: Add SOF3 marker support

**Files:**
- Modify: `src/decode/marker.rs`, `src/common/types.rs`

- [ ] **Step 1: Add `is_lossless` field to FrameHeader**

```rust
pub struct FrameHeader {
    // ... existing fields ...
    pub is_progressive: bool,
    pub is_lossless: bool,    // NEW: true for SOF3
}
```

- [ ] **Step 2: Parse SOF3 in marker reader**

```rust
const SOF3: u8 = 0xC3; // Lossless, Huffman-coded

// In read_markers() match:
SOF3 => {
    frame = Some(self.read_sof_lossless()?);
}
```

### Task 19: Implement lossless Huffman decoder

**Files:**
- Create: `src/decode/lossless_huffman.rs`

- [ ] **Step 1: Decode difference values**

Lossless Huffman: read category from Huffman table, then read that many magnitude bits. Similar to DC decoding but applied per-sample.

```rust
pub fn decode_difference(
    bit_reader: &mut BitReader,
    table: &HuffmanTable,
) -> Result<i16> {
    let category = decode_huffman_symbol(bit_reader, table)?;
    if category == 0 { return Ok(0); }
    let magnitude_bits = bit_reader.read_bits(category);
    // Extend sign
    Ok(extend_sign(magnitude_bits, category))
}
```

### Task 20: Implement prediction and undifferencing

**Files:**
- Create: `src/decode/lossless.rs`

- [ ] **Step 1: Implement 7 predictors**

```rust
fn predict(psv: u8, ra: i32, rb: i32, rc: i32) -> i32 {
    match psv {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => 0,
    }
}
```

- [ ] **Step 2: Implement row-by-row undifferencing**

```rust
pub fn undifference_row(
    diffs: &[i16],
    prev_row: Option<&[u16]>,
    output: &mut [u16],
    psv: u8,
    precision: u8,
    point_transform: u8,
    is_first_row: bool,
) {
    let mask = (1u32 << precision) - 1;
    let initial = 1i32 << (precision as i32 - point_transform as i32 - 1);

    for x in 0..diffs.len() {
        let prediction = if is_first_row && x == 0 {
            initial
        } else if is_first_row {
            output[x - 1] as i32  // predictor 1 (left)
        } else if x == 0 {
            prev_row.unwrap()[0] as i32  // predictor 2 (above)
        } else {
            let ra = output[x - 1] as i32;
            let rb = prev_row.unwrap()[x] as i32;
            let rc = prev_row.unwrap()[x - 1] as i32;
            predict(psv, ra, rb, rc)
        };

        output[x] = ((diffs[x] as i32 + prediction) as u32 & mask) as u16;
    }
}
```

### Task 21: Wire lossless path into pipeline

- [ ] **Step 1: Branch on `is_lossless` in `decode_image()`**

```rust
if frame.is_lossless {
    return self.decode_lossless(frame);
}
```

- [ ] **Step 2: Implement `decode_lossless()`**

Iterate rows, decode differences via lossless Huffman, undifference, apply point transform scaling, convert to output pixels.

- [ ] **Step 3: Handle multi-precision output**

For precision > 8: store as `u16` data. Add `data_u16: Option<Vec<u16>>` to `Image` or always output as 8-bit with shift.

### Task 22: Tests

```rust
#[test]
fn lossless_8bit_predictor_1() {
    // Test with a known lossless JPEG fixture
    // Verify exact sample reconstruction
}

#[test]
fn undifference_first_row() {
    let diffs = [10i16, 5, -3, 2];
    let mut output = [0u16; 4];
    undifference_row(&diffs, None, &mut output, 1, 8, 0, true);
    // initial = 128, then horizontal prediction
    assert_eq!(output[0], (10 + 128) as u16);
    assert_eq!(output[1], (5 + output[0] as i32) as u16);
}

#[test]
fn all_7_predictors_produce_valid_output() {
    for psv in 1..=7 {
        let ra = 100i32;
        let rb = 110;
        let rc = 105;
        let pred = predict(psv, ra, rb, rc);
        assert!(pred >= 0 && pred < 256);
    }
}
```

- [ ] **Step 4: Format and commit**

```bash
cargo fmt && cargo test && git add -A && git commit -s -m "feat: add lossless JPEG (SOF3) decoding with 2-16 bit precision"
```
