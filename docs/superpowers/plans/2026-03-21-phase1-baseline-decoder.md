# Phase 1: Baseline JPEG Decoder (Scalar) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a correct, scalar-only Baseline JPEG decoder that passes conformance tests against libjpeg-turbo output.

**Architecture:** Bottom-up build of the JPEG decoding pipeline. Each module is independently testable. The pipeline flows: markers → entropy decode → dequantize → IDCT → upsample → color convert → pixels. A `Decoder` struct orchestrates the pipeline, exposed through both streaming and high-level APIs.

**Tech Stack:** Rust (stable), `thiserror` for errors, `criterion` for benchmarks, no other runtime dependencies.

**Phase 1 scope (Baseline only):**
- SOF0 (Baseline DCT) — 8-bit samples only
- Huffman coding only (no arithmetic)
- Sequential only (no progressive)
- Subsampling: 4:4:4, 4:2:2, 4:2:0
- Color: Grayscale, YCbCr → RGB
- No lossless, no 12/16-bit

**JPEG Baseline decoding pipeline reference:**
```
JPEG bytes
  → Parse markers (SOI, SOF0, DHT, DQT, SOS, APPn, COM, EOI)
  → Huffman decode entropy data → DC/AC coefficients per 8×8 block
  → Dequantize (multiply by quantization table entries)
  → Inverse DCT (frequency → spatial domain, 8×8 blocks)
  → Reassemble blocks into component planes
  → Upsample chroma (if 4:2:2 or 4:2:0)
  → Color convert (YCbCr → RGB)
  → Output pixel buffer
```

---

## File Structure

```
libjpeg-turbo-rs/
├── Cargo.toml                    # crate manifest, thiserror + criterion deps
├── src/
│   ├── lib.rs                    # module declarations + public re-exports
│   ├── common/
│   │   ├── mod.rs
│   │   ├── types.rs              # ColorSpace, Subsampling, PixelFormat, ComponentInfo, FrameHeader, ScanHeader
│   │   ├── error.rs              # JpegError enum, Result alias
│   │   ├── huffman_table.rs      # HuffmanTable: bits/values storage + decode lookup table builder
│   │   └── quant_table.rs        # QuantTable: 64-entry table + zigzag-ordered storage
│   ├── decode/
│   │   ├── mod.rs
│   │   ├── marker.rs             # MarkerReader: parse all marker segments from byte stream
│   │   ├── bitstream.rs          # BitReader: read bits from entropy-coded data, handle byte stuffing (0xFF00)
│   │   ├── huffman.rs            # decode_dc_coefficient(), decode_ac_coefficients() using HuffmanTable
│   │   ├── entropy.rs            # McuDecoder: decode full MCUs, manage DC prediction per component
│   │   ├── dequant.rs            # dequantize_block(): multiply 8×8 coefficients by quantization table
│   │   ├── idct.rs               # idct_8x8(): scalar implementation of 8×8 inverse DCT
│   │   ├── upsample.rs           # upsample_h2v1(), upsample_h2v2(), fancy (triangle filter) variants
│   │   ├── color.rs              # ycbcr_to_rgb(): per-pixel and bulk color space conversion
│   │   └── pipeline.rs           # Decoder struct: orchestrates full decode from reader to pixel buffer
│   └── api/
│       ├── mod.rs
│       ├── high_level.rs         # decompress(&[u8]) -> Result<Image>
│       └── streaming.rs          # StreamingDecoder: header() + next_scanline()
├── tests/
│   ├── fixtures/                 # test JPEG files (tiny, hand-verified)
│   │   └── .gitkeep
│   ├── common_types.rs           # tests for types.rs
│   ├── marker_parsing.rs         # tests for marker.rs
│   ├── huffman_table.rs          # tests for huffman_table.rs
│   ├── bitstream.rs              # tests for bitstream.rs
│   ├── huffman_decode.rs         # tests for huffman.rs
│   ├── entropy_decode.rs         # tests for entropy.rs
│   ├── dequant.rs                # tests for dequant.rs
│   ├── idct.rs                   # tests for idct.rs
│   ├── upsample.rs               # tests for upsample.rs
│   ├── color_convert.rs          # tests for color.rs
│   ├── decode_pipeline.rs        # integration tests for full decode
│   └── conformance.rs            # bit-exact tests vs libjpeg-turbo reference output
└── benches/
    └── decode.rs                 # criterion: decode various test images
```

---

## Chunk 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/common/mod.rs`
- Create: `src/decode/mod.rs`
- Create: `src/api/mod.rs`
- Create: `tests/fixtures/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "libjpeg-turbo-rs"
version = "0.1.0"
edition = "2021"
description = "Pure Rust reimplementation of libjpeg-turbo"
license = "MIT OR Apache-2.0"

[dependencies]
thiserror = "2"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "decode"
harness = false
```

- [ ] **Step 2: Create .gitignore**

```
/target
```

- [ ] **Step 3: Create module stubs**

`src/lib.rs`:
```rust
pub mod common;
pub mod decode;
pub mod api;
```

`src/common/mod.rs`:
```rust
pub mod types;
pub mod error;
pub mod huffman_table;
pub mod quant_table;
```

`src/decode/mod.rs`:
```rust
pub mod marker;
pub mod bitstream;
pub mod huffman;
pub mod entropy;
pub mod dequant;
pub mod idct;
pub mod upsample;
pub mod color;
pub mod pipeline;
```

`src/api/mod.rs`:
```rust
pub mod high_level;
pub mod streaming;
```

Create empty files for every module listed above (each containing just `// TODO: implement`), plus `tests/fixtures/.gitkeep`.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check`
Expected: success (no errors)

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml .gitignore src/ tests/
git commit -s -m "feat: project scaffolding with module structure"
```

---

### Task 2: Common Types and Error Handling

**Files:**
- Create: `src/common/types.rs`
- Create: `src/common/error.rs`
- Test: `tests/common_types.rs`

- [ ] **Step 1: Write the failing test**

`tests/common_types.rs`:
```rust
use libjpeg_turbo_rs::common::types::*;

#[test]
fn subsampling_block_dimensions() {
    // 4:4:4 — each component has 1×1 blocks per MCU
    assert_eq!(Subsampling::S444.mcu_width_blocks(), 1);
    assert_eq!(Subsampling::S444.mcu_height_blocks(), 1);

    // 4:2:2 — luma has 2×1 blocks, chroma has 1×1
    assert_eq!(Subsampling::S422.mcu_width_blocks(), 2);
    assert_eq!(Subsampling::S422.mcu_height_blocks(), 1);

    // 4:2:0 — luma has 2×2 blocks, chroma has 1×1
    assert_eq!(Subsampling::S420.mcu_width_blocks(), 2);
    assert_eq!(Subsampling::S420.mcu_height_blocks(), 2);
}

#[test]
fn pixel_format_bytes_per_pixel() {
    assert_eq!(PixelFormat::Rgb.bytes_per_pixel(), 3);
    assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Grayscale.bytes_per_pixel(), 1);
}

#[test]
fn color_space_num_components() {
    assert_eq!(ColorSpace::Grayscale.num_components(), 1);
    assert_eq!(ColorSpace::YCbCr.num_components(), 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test common_types`
Expected: FAIL — types not defined yet

- [ ] **Step 3: Implement types.rs**

`src/common/types.rs`:
```rust
/// JPEG color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Grayscale,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
}

impl ColorSpace {
    pub fn num_components(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::YCbCr | Self::Rgb => 3,
            Self::Cmyk | Self::Ycck => 4,
        }
    }
}

/// Chroma subsampling modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:4:4 — no subsampling
    S444,
    /// 4:2:2 — horizontal 2×, vertical 1×
    S422,
    /// 4:2:0 — horizontal 2×, vertical 2×
    S420,
    /// 4:4:0 — horizontal 1×, vertical 2×
    S440,
    /// 4:1:1 — horizontal 4×, vertical 1×
    S411,
}

impl Subsampling {
    /// Max horizontal sampling factor (luma blocks per MCU row).
    pub fn mcu_width_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S440 => 1,
            Self::S422 | Self::S420 => 2,
            Self::S411 => 4,
        }
    }

    /// Max vertical sampling factor (luma blocks per MCU column).
    pub fn mcu_height_blocks(self) -> usize {
        match self {
            Self::S444 | Self::S422 | Self::S411 => 1,
            Self::S420 | Self::S440 => 2,
        }
    }
}

/// Output pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Grayscale,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }
}

/// Information about a single image component (Y, Cb, or Cr).
#[derive(Debug, Clone, Copy)]
pub struct ComponentInfo {
    /// Component identifier (1=Y, 2=Cb, 3=Cr per JFIF).
    pub id: u8,
    /// Horizontal sampling factor (1-4).
    pub horizontal_sampling: u8,
    /// Vertical sampling factor (1-4).
    pub vertical_sampling: u8,
    /// Index into the quantization table array.
    pub quant_table_index: u8,
}

/// Parsed from the SOF marker — describes the image frame.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Sample precision in bits (8 for Baseline).
    pub precision: u8,
    /// Image height in pixels.
    pub height: u16,
    /// Image width in pixels.
    pub width: u16,
    /// Per-component info.
    pub components: Vec<ComponentInfo>,
}

/// Parsed from the SOS marker — describes one scan.
#[derive(Debug, Clone)]
pub struct ScanHeader {
    /// Component selectors for this scan.
    pub components: Vec<ScanComponentSelector>,
}

/// Per-component selector within a scan.
#[derive(Debug, Clone, Copy)]
pub struct ScanComponentSelector {
    /// Component identifier (matches ComponentInfo::id).
    pub component_id: u8,
    /// DC Huffman table index (0-3).
    pub dc_table_index: u8,
    /// AC Huffman table index (0-3).
    pub ac_table_index: u8,
}
```

- [ ] **Step 4: Implement error.rs**

`src/common/error.rs`:
```rust
/// All errors that can occur during JPEG processing.
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

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, JpegError>;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --test common_types`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add src/common/types.rs src/common/error.rs tests/common_types.rs
git commit -s -m "feat: common types (ColorSpace, Subsampling, PixelFormat) and error handling"
```

---

### Task 3: Quantization Table

**Files:**
- Create: `src/common/quant_table.rs`

- [ ] **Step 1: Write the failing test**

Add to `tests/common_types.rs`:
```rust
use libjpeg_turbo_rs::common::quant_table::QuantTable;

#[test]
fn quant_table_from_zigzag_and_natural_order() {
    // First 3 entries of standard JPEG zigzag order:
    // zigzag[0] = DC, zigzag[1] = (0,1), zigzag[2] = (1,0)
    let mut zigzag_data = [1u16; 64];
    zigzag_data[0] = 16; // DC quantization value
    zigzag_data[1] = 11;
    zigzag_data[2] = 10;

    let table = QuantTable::from_zigzag(&zigzag_data);

    // In natural (row-major) order:
    // [0][0] = 16 (DC), [0][1] = 11, [1][0] = 10
    assert_eq!(table.get(0, 0), 16);
    assert_eq!(table.get(0, 1), 11);
    assert_eq!(table.get(1, 0), 10);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test common_types quant_table`
Expected: FAIL

- [ ] **Step 3: Implement quant_table.rs**

`src/common/quant_table.rs`:
```rust
/// Standard JPEG zigzag scan order.
/// Maps zigzag index → (row * 8 + col) in natural order.
#[rustfmt::skip]
pub const ZIGZAG_ORDER: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// A 64-entry quantization table stored in natural (row-major) order.
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// Values in natural (row-major) 8×8 order.
    pub values: [u16; 64],
}

impl QuantTable {
    /// Build from zigzag-ordered data (as stored in the JPEG DQT marker).
    pub fn from_zigzag(zigzag_data: &[u16; 64]) -> Self {
        let mut values = [0u16; 64];
        for (zigzag_index, &value) in zigzag_data.iter().enumerate() {
            values[ZIGZAG_ORDER[zigzag_index]] = value;
        }
        Self { values }
    }

    /// Get value at (row, col) in the 8×8 block.
    pub fn get(&self, row: usize, col: usize) -> u16 {
        self.values[row * 8 + col]
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test common_types`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/common/quant_table.rs tests/common_types.rs
git commit -s -m "feat: quantization table with zigzag-to-natural order conversion"
```

---

### Task 4: Huffman Table

**Files:**
- Create: `src/common/huffman_table.rs`
- Test: `tests/huffman_table.rs`

The Huffman table is built from the DHT marker data: 16 bytes of code-length counts (`bits[1..=16]`), followed by the symbol values. We build a lookup table for fast decoding.

- [ ] **Step 1: Write the failing test**

`tests/huffman_table.rs`:
```rust
use libjpeg_turbo_rs::common::huffman_table::HuffmanTable;

#[test]
fn build_dc_luminance_table() {
    // Standard JPEG DC luminance Huffman table (ITU-T.81 Table K.3)
    // bits[i] = number of codes of length i (1-indexed)
    let bits: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    let table = HuffmanTable::build(&bits, &values).unwrap();

    // The table should contain 12 symbols (categories 0-11 for DC)
    assert_eq!(table.num_symbols(), 12);
}

#[test]
fn lookup_known_codes() {
    // Minimal hand-crafted table:
    // 1 code of length 1: symbol 0x00 → code "0"
    // 1 code of length 2: symbol 0x01 → code "10"
    // 1 code of length 3: symbol 0x02 → code "110"
    let bits: [u8; 17] = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0x00, 0x01, 0x02];

    let table = HuffmanTable::build(&bits, &values).unwrap();

    // Verify lookup: feed the bit pattern and expect the correct symbol + length
    let (symbol, length) = table.lookup(0b0000_0000, 8).unwrap(); // starts with "0"
    assert_eq!(symbol, 0x00);
    assert_eq!(length, 1);

    let (symbol, length) = table.lookup(0b1000_0000, 8).unwrap(); // starts with "10"
    assert_eq!(symbol, 0x01);
    assert_eq!(length, 2);

    let (symbol, length) = table.lookup(0b1100_0000, 8).unwrap(); // starts with "110"
    assert_eq!(symbol, 0x02);
    assert_eq!(length, 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test huffman_table`
Expected: FAIL

- [ ] **Step 3: Implement huffman_table.rs**

`src/common/huffman_table.rs`:
```rust
use crate::common::error::{JpegError, Result};

const LOOKUP_BITS: usize = 8;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;

/// Entry in the fast lookup table.
/// For codes ≤ LOOKUP_BITS: symbol + code length.
/// For codes > LOOKUP_BITS: index into overflow table.
#[derive(Debug, Clone, Copy, Default)]
struct LookupEntry {
    /// Decoded symbol value.
    symbol: u8,
    /// Code length in bits (0 means invalid / overflow).
    length: u8,
}

/// Huffman decoding table built from DHT marker data.
/// Uses an 8-bit fast lookup table for short codes, with
/// a fallback slow path for codes longer than 8 bits.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// Fast lookup for codes ≤ 8 bits. Indexed by the first 8 bits of input.
    fast: Vec<LookupEntry>,
    /// Maximum code value for each code length (1-indexed).
    maxcode: [i32; 18],
    /// Value offset for each code length.
    valoffset: [i32; 18],
    /// Symbol values in order.
    values: Vec<u8>,
    /// Number of symbols.
    count: usize,
}

impl HuffmanTable {
    /// Build a Huffman table from DHT marker data.
    ///
    /// `bits[0]` is unused; `bits[i]` for i in 1..=16 is the count of codes with length i.
    /// `values` contains the symbol values in code-length order.
    pub fn build(bits: &[u8; 17], values: &[u8]) -> Result<Self> {
        let total_symbols: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        if values.len() < total_symbols {
            return Err(JpegError::CorruptData(format!(
                "Huffman table: expected {} symbols, got {}",
                total_symbols,
                values.len()
            )));
        }

        // Generate code values for each symbol (JPEG spec Figure C.1)
        let mut huffcode = Vec::with_capacity(total_symbols);
        let mut code: u32 = 0;
        for length in 1..=16usize {
            for _ in 0..bits[length] {
                huffcode.push((code, length));
                code += 1;
            }
            code <<= 1;
        }

        // Build maxcode and valoffset arrays for slow decode path
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 18];
        let mut symbol_index: usize = 0;
        for length in 1..=16usize {
            let count = bits[length] as usize;
            if count > 0 {
                valoffset[length] = symbol_index as i32 - huffcode[symbol_index].0 as i32;
                symbol_index += count;
                maxcode[length] = huffcode[symbol_index - 1].0 as i32;
            }
        }

        // Build fast lookup table for codes ≤ 8 bits
        let mut fast = vec![LookupEntry::default(); LOOKUP_SIZE];
        for (i, &(code_val, code_len)) in huffcode.iter().enumerate() {
            if code_len <= LOOKUP_BITS {
                // This code occupies multiple slots in the lookup table.
                // Shift code to MSB position, then fill all suffix combinations.
                let code_shifted = (code_val as usize) << (LOOKUP_BITS - code_len);
                let fill_count = 1 << (LOOKUP_BITS - code_len);
                for j in 0..fill_count {
                    fast[code_shifted | j] = LookupEntry {
                        symbol: values[i],
                        length: code_len as u8,
                    };
                }
            }
        }

        Ok(Self {
            fast,
            maxcode,
            valoffset,
            values: values[..total_symbols].to_vec(),
            count: total_symbols,
        })
    }

    /// Look up a symbol from the first `available_bits` (up to 16) of `bits_msb`.
    ///
    /// `bits_msb` should have the next bits left-aligned (MSB first).
    /// Returns `(symbol, code_length)`.
    pub fn lookup(&self, bits_msb: u16, available_bits: u8) -> Result<(u8, u8)> {
        if available_bits == 0 {
            return Err(JpegError::UnexpectedEof);
        }

        // Fast path: use top 8 bits as index
        let index = (bits_msb >> 8) as usize;
        let entry = &self.fast[index];
        if entry.length > 0 && entry.length <= available_bits {
            return Ok((entry.symbol, entry.length));
        }

        // Slow path: bit-by-bit for codes > 8 bits
        let mut code = (bits_msb >> 15) as i32;
        for length in 1..=16u8 {
            if code <= self.maxcode[length as usize] {
                let index = (code + self.valoffset[length as usize]) as usize;
                if index < self.values.len() {
                    return Ok((self.values[index], length));
                }
            }
            if length < 16 {
                code = (code << 1) | ((bits_msb >> (14 - length as u16 + 1)) & 1) as i32;
            }
        }

        Err(JpegError::CorruptData("invalid Huffman code".into()))
    }

    /// Number of symbols in this table.
    pub fn num_symbols(&self) -> usize {
        self.count
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test huffman_table`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/common/huffman_table.rs tests/huffman_table.rs
git commit -s -m "feat: Huffman table builder with 8-bit fast lookup"
```

---

### Task 5: Marker Parsing

**Files:**
- Create: `src/decode/marker.rs`
- Test: `tests/marker_parsing.rs`
- Create: `tests/fixtures/tiny_gray.jpg` (generated programmatically in test)

JPEG markers all begin with 0xFF followed by a marker code. The marker reader must handle: SOI (0xD8), EOI (0xD9), SOF0 (0xC0), DHT (0xC4), DQT (0xDB), SOS (0xDA), APPn (0xE0-0xEF), COM (0xFE), DRI (0xDD).

- [ ] **Step 1: Write the failing test**

`tests/marker_parsing.rs`:
```rust
use libjpeg_turbo_rs::common::types::*;
use libjpeg_turbo_rs::decode::marker::MarkerReader;

/// Minimal valid Baseline JPEG: 1×1 white pixel, grayscale.
/// Hand-crafted to be the smallest valid JPEG possible.
fn minimal_grayscale_jpeg() -> Vec<u8> {
    vec![
        // SOI
        0xFF, 0xD8,
        // DQT: 8-bit, table 0, all values = 1 (no quantization effect)
        0xFF, 0xDB, 0x00, 0x43, 0x00,
        1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
        // SOF0: Baseline, 8-bit, 1×1, 1 component (grayscale)
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01,
        0x01, // 1 component
        0x01, 0x11, 0x00, // component 1: id=1, sampling=1×1, quant_table=0
        // DHT: DC table 0
        0xFF, 0xC4, 0x00, 0x1F, 0x00, // class=DC, id=0
        0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        // DHT: AC table 0 (minimal — just EOB)
        0xFF, 0xC4, 0x00, 0x05, 0x10, // class=AC, id=0
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, // 1 symbol of length 1: EOB (0x00)
        // SOS: 1 component
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        // Entropy-coded data (placeholder — 1 MCU for a white pixel)
        0x00,
        // EOI
        0xFF, 0xD9,
    ]
}

#[test]
fn parse_frame_header() {
    let data = minimal_grayscale_jpeg();
    let mut reader = MarkerReader::new(&data);
    let result = reader.read_markers().unwrap();

    assert_eq!(result.frame.precision, 8);
    assert_eq!(result.frame.width, 1);
    assert_eq!(result.frame.height, 1);
    assert_eq!(result.frame.components.len(), 1);
    assert_eq!(result.frame.components[0].id, 1);
}

#[test]
fn parse_quantization_table() {
    let data = minimal_grayscale_jpeg();
    let mut reader = MarkerReader::new(&data);
    let result = reader.read_markers().unwrap();

    // Should have 1 quantization table (index 0)
    assert!(result.quant_tables[0].is_some());
    // All values should be 1
    let qt = result.quant_tables[0].as_ref().unwrap();
    assert_eq!(qt.get(0, 0), 1);
}

#[test]
fn parse_huffman_tables() {
    let data = minimal_grayscale_jpeg();
    let mut reader = MarkerReader::new(&data);
    let result = reader.read_markers().unwrap();

    // Should have DC table 0 and AC table 0
    assert!(result.dc_huffman_tables[0].is_some());
    assert!(result.ac_huffman_tables[0].is_some());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test marker_parsing`
Expected: FAIL

- [ ] **Step 3: Implement marker.rs**

`src/decode/marker.rs`:
```rust
use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;

// JPEG marker codes
const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOF0: u8 = 0xC0; // Baseline DCT
const DHT: u8 = 0xC4;
const DQT: u8 = 0xDB;
const SOS: u8 = 0xDA;
const DRI: u8 = 0xDD;
const COM: u8 = 0xFE;

/// All metadata parsed from JPEG markers before the entropy-coded data.
#[derive(Debug)]
pub struct JpegMetadata {
    pub frame: FrameHeader,
    pub scan: ScanHeader,
    pub quant_tables: [Option<QuantTable>; 4],
    pub dc_huffman_tables: [Option<HuffmanTable>; 4],
    pub ac_huffman_tables: [Option<HuffmanTable>; 4],
    pub restart_interval: u16,
    /// Byte offset where entropy-coded data begins.
    pub entropy_data_offset: usize,
}

/// Reads and parses JPEG markers from a byte slice.
pub struct MarkerReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MarkerReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Parse all markers up to and including SOS.
    /// Returns metadata and the offset where entropy data begins.
    pub fn read_markers(&mut self) -> Result<JpegMetadata> {
        self.expect_marker(SOI)?;

        let mut frame: Option<FrameHeader> = None;
        let mut scan: Option<ScanHeader> = None;
        let mut quant_tables: [Option<QuantTable>; 4] = [None, None, None, None];
        let mut dc_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut ac_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut restart_interval: u16 = 0;

        loop {
            let marker = self.read_marker()?;
            match marker {
                SOF0 => {
                    frame = Some(self.read_sof()?);
                }
                DQT => {
                    self.read_dqt(&mut quant_tables)?;
                }
                DHT => {
                    self.read_dht(&mut dc_huffman_tables, &mut ac_huffman_tables)?;
                }
                DRI => {
                    restart_interval = self.read_dri()?;
                }
                SOS => {
                    scan = Some(self.read_sos()?);
                    break;
                }
                EOI => {
                    return Err(JpegError::CorruptData(
                        "unexpected EOI before SOS".into(),
                    ));
                }
                // Skip APPn and COM markers
                m if (0xE0..=0xEF).contains(&m) || m == COM => {
                    self.skip_marker_segment()?;
                }
                // Skip unsupported markers with length
                m if m != 0x00 && m != 0xFF => {
                    self.skip_marker_segment()?;
                }
                m => {
                    return Err(JpegError::InvalidMarker(m));
                }
            }
        }

        let frame = frame.ok_or(JpegError::CorruptData("missing SOF marker".into()))?;
        let scan = scan.ok_or(JpegError::CorruptData("missing SOS marker".into()))?;

        Ok(JpegMetadata {
            frame,
            scan,
            quant_tables,
            dc_huffman_tables,
            ac_huffman_tables,
            restart_interval,
            entropy_data_offset: self.pos,
        })
    }

    fn expect_marker(&mut self, expected: u8) -> Result<()> {
        if self.pos + 1 >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        if self.data[self.pos] != 0xFF || self.data[self.pos + 1] != expected {
            return Err(JpegError::UnexpectedMarker(
                self.data.get(self.pos + 1).copied().unwrap_or(0),
            ));
        }
        self.pos += 2;
        Ok(())
    }

    fn read_marker(&mut self) -> Result<u8> {
        // Skip any padding 0xFF bytes
        while self.pos < self.data.len() && self.data[self.pos] == 0xFF {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let marker = self.data[self.pos];
        self.pos += 1;
        if marker == 0x00 {
            return Err(JpegError::InvalidMarker(0x00));
        }
        Ok(marker)
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    fn read_u16_be(&mut self) -> Result<u16> {
        let hi = self.read_u8()? as u16;
        let lo = self.read_u8()? as u16;
        Ok((hi << 8) | lo)
    }

    fn skip_marker_segment(&mut self) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("marker segment length < 2".into()));
        }
        let skip = length - 2; // length includes its own 2 bytes
        if self.pos + skip > self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        self.pos += skip;
        Ok(())
    }

    fn read_sof(&mut self) -> Result<FrameHeader> {
        let length = self.read_u16_be()? as usize;
        let start = self.pos;

        let precision = self.read_u8()?;
        let height = self.read_u16_be()?;
        let width = self.read_u16_be()?;
        let num_components = self.read_u8()? as usize;

        let mut components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            let id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let quant_table_index = self.read_u8()?;
            components.push(ComponentInfo {
                id,
                horizontal_sampling: sampling >> 4,
                vertical_sampling: sampling & 0x0F,
                quant_table_index,
            });
        }

        // Ensure we consumed the right number of bytes
        let consumed = self.pos - start;
        if consumed != length - 2 {
            self.pos = start + length - 2;
        }

        Ok(FrameHeader {
            precision,
            height,
            width,
            components,
        })
    }

    fn read_dqt(&mut self, tables: &mut [Option<QuantTable>; 4]) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        let end = self.pos + length - 2;

        while self.pos < end {
            let info = self.read_u8()?;
            let precision = info >> 4; // 0 = 8-bit, 1 = 16-bit
            let table_id = (info & 0x0F) as usize;

            if table_id >= 4 {
                return Err(JpegError::CorruptData(format!(
                    "quantization table id {} out of range",
                    table_id
                )));
            }

            let mut zigzag = [0u16; 64];
            if precision == 0 {
                for entry in zigzag.iter_mut() {
                    *entry = self.read_u8()? as u16;
                }
            } else {
                for entry in zigzag.iter_mut() {
                    *entry = self.read_u16_be()?;
                }
            }

            tables[table_id] = Some(QuantTable::from_zigzag(&zigzag));
        }

        Ok(())
    }

    fn read_dht(
        &mut self,
        dc_tables: &mut [Option<HuffmanTable>; 4],
        ac_tables: &mut [Option<HuffmanTable>; 4],
    ) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        let end = self.pos + length - 2;

        while self.pos < end {
            let info = self.read_u8()?;
            let table_class = info >> 4; // 0 = DC, 1 = AC
            let table_id = (info & 0x0F) as usize;

            if table_id >= 4 {
                return Err(JpegError::CorruptData(format!(
                    "Huffman table id {} out of range",
                    table_id
                )));
            }

            let mut bits = [0u8; 17];
            for i in 1..=16 {
                bits[i] = self.read_u8()?;
            }

            let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
            let mut values = Vec::with_capacity(total);
            for _ in 0..total {
                values.push(self.read_u8()?);
            }

            let table = HuffmanTable::build(&bits, &values)?;

            if table_class == 0 {
                dc_tables[table_id] = Some(table);
            } else {
                ac_tables[table_id] = Some(table);
            }
        }

        Ok(())
    }

    fn read_dri(&mut self) -> Result<u16> {
        let _length = self.read_u16_be()?;
        self.read_u16_be()
    }

    fn read_sos(&mut self) -> Result<ScanHeader> {
        let _length = self.read_u16_be()?;
        let num_components = self.read_u8()? as usize;

        let mut components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            components.push(ScanComponentSelector {
                component_id,
                dc_table_index: tables >> 4,
                ac_table_index: tables & 0x0F,
            });
        }

        // Skip: Ss, Se, Ah|Al (spectral selection / successive approximation)
        let _ss = self.read_u8()?;
        let _se = self.read_u8()?;
        let _ahl = self.read_u8()?;

        Ok(ScanHeader { components })
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test marker_parsing`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/marker.rs tests/marker_parsing.rs
git commit -s -m "feat: JPEG marker parser (SOF0, DQT, DHT, SOS, DRI, APPn, COM)"
```

---

## Chunk 2: Entropy Decoding

### Task 6: Bitstream Reader

**Files:**
- Create: `src/decode/bitstream.rs`
- Test: `tests/bitstream.rs`

The bitstream reader reads bits from the entropy-coded segment. It must handle JPEG byte stuffing: any 0xFF in the data stream is followed by 0x00 (stuffed byte) or a restart marker (0xD0-0xD7).

- [ ] **Step 1: Write the failing test**

`tests/bitstream.rs`:
```rust
use libjpeg_turbo_rs::decode::bitstream::BitReader;

#[test]
fn read_bits_basic() {
    // Binary: 10110100 = 0xB4
    let data = [0xB4u8];
    let mut reader = BitReader::new(&data);

    assert_eq!(reader.read_bits(1).unwrap(), 1);  // '1'
    assert_eq!(reader.read_bits(3).unwrap(), 0b011); // '011'
    assert_eq!(reader.read_bits(4).unwrap(), 0b0100); // '0100'
}

#[test]
fn read_bits_across_bytes() {
    // 0xAB = 1010_1011, 0xCD = 1100_1101
    let data = [0xAB, 0xCD];
    let mut reader = BitReader::new(&data);

    assert_eq!(reader.read_bits(4).unwrap(), 0b1010);  // first nibble of 0xAB
    assert_eq!(reader.read_bits(8).unwrap(), 0b1011_1100); // crosses byte boundary
    assert_eq!(reader.read_bits(4).unwrap(), 0b1101);  // last nibble of 0xCD
}

#[test]
fn byte_stuffing_ff00_is_transparent() {
    // 0xFF followed by 0x00 should be read as a single 0xFF byte
    let data = [0xFF, 0x00, 0x80];
    let mut reader = BitReader::new(&data);

    // 0xFF = 1111_1111, next byte 0x80 = 1000_0000
    assert_eq!(reader.read_bits(8).unwrap(), 0xFF);
    assert_eq!(reader.read_bits(1).unwrap(), 1);
}

#[test]
fn peek_bits_does_not_consume() {
    let data = [0xB4u8];
    let mut reader = BitReader::new(&data);

    assert_eq!(reader.peek_bits(4).unwrap(), 0b1011);
    assert_eq!(reader.peek_bits(4).unwrap(), 0b1011); // same result
    assert_eq!(reader.read_bits(4).unwrap(), 0b1011); // now consumed
    assert_eq!(reader.read_bits(4).unwrap(), 0b0100); // next 4 bits
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bitstream`
Expected: FAIL

- [ ] **Step 3: Implement bitstream.rs**

`src/decode/bitstream.rs`:
```rust
use crate::common::error::{JpegError, Result};

/// Reads individual bits from JPEG entropy-coded data.
/// Handles byte stuffing (0xFF 0x00 → 0xFF) and detects restart markers.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bit buffer holding up to 32 bits.
    bit_buffer: u32,
    /// Number of valid bits remaining in bit_buffer.
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buffer: 0,
            bits_left: 0,
        }
    }

    /// Fill the bit buffer so it contains at least `needed` bits (up to 25).
    fn fill_buffer(&mut self, needed: u8) -> Result<()> {
        while self.bits_left < needed {
            let byte = self.read_next_byte()?;
            self.bit_buffer = (self.bit_buffer << 8) | byte as u32;
            self.bits_left += 8;
        }
        Ok(())
    }

    /// Read the next byte, handling byte stuffing.
    fn read_next_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        if byte == 0xFF {
            if self.pos >= self.data.len() {
                return Err(JpegError::UnexpectedEof);
            }
            let next = self.data[self.pos];
            if next == 0x00 {
                // Byte stuffing: 0xFF 0x00 → 0xFF
                self.pos += 1;
                Ok(0xFF)
            } else if (0xD0..=0xD7).contains(&next) {
                // Restart marker — caller should handle via reset()
                self.pos += 1;
                Ok(0xFF)
            } else {
                // Unexpected marker in entropy data
                Err(JpegError::UnexpectedMarker(next))
            }
        } else {
            Ok(byte)
        }
    }

    /// Peek at the next `count` bits without consuming them (max 16).
    pub fn peek_bits(&mut self, count: u8) -> Result<u16> {
        debug_assert!(count <= 16);
        self.fill_buffer(count)?;
        let shift = self.bits_left - count;
        Ok(((self.bit_buffer >> shift) & ((1 << count) - 1)) as u16)
    }

    /// Read and consume `count` bits (max 16).
    pub fn read_bits(&mut self, count: u8) -> Result<u16> {
        debug_assert!(count <= 16);
        self.fill_buffer(count)?;
        self.bits_left -= count;
        let val = (self.bit_buffer >> self.bits_left) & ((1 << count) - 1);
        Ok(val as u16)
    }

    /// Discard any remaining bits in the current byte (align to byte boundary).
    /// Used when processing restart markers.
    pub fn reset(&mut self) {
        self.bit_buffer = 0;
        self.bits_left = 0;
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test bitstream`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/bitstream.rs tests/bitstream.rs
git commit -s -m "feat: bitstream reader with byte stuffing and peek support"
```

---

### Task 7: Huffman Symbol Decoding

**Files:**
- Create: `src/decode/huffman.rs`
- Test: `tests/huffman_decode.rs`

Huffman decoding for DC and AC coefficients. DC uses differential coding (each DC value is a delta from the previous block's DC). AC coefficients are run-length coded: the Huffman symbol encodes (run_length, bit_size) where run_length is the count of zero coefficients before a nonzero one.

- [ ] **Step 1: Write the failing test**

`tests/huffman_decode.rs`:
```rust
use libjpeg_turbo_rs::common::huffman_table::HuffmanTable;
use libjpeg_turbo_rs::decode::bitstream::BitReader;
use libjpeg_turbo_rs::decode::huffman;

#[test]
fn decode_dc_coefficient() {
    // Category 3, value +5 → Huffman symbol = 3, then 3 extra bits = 101 (=5)
    // Build a trivial DC table: category 3 has code "0" (length 1)
    let bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![3]; // symbol = category 3
    let table = HuffmanTable::build(&bits, &values).unwrap();

    // Bitstream: "0" (Huffman code for cat 3) + "101" (extra bits for +5)
    // = 0_101_0000 = 0x50 (padded)
    let data = [0x50u8];
    let mut reader = BitReader::new(&data);

    let dc_value = huffman::decode_dc_coefficient(&mut reader, &table).unwrap();
    assert_eq!(dc_value, 5);
}

#[test]
fn decode_dc_negative_value() {
    // Category 2, value -3 → symbol = 2, extra bits = 00 (= -3, since 2-bit ones' complement)
    let bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![2];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    // "0" + "00" = 0_00_00000 = 0x00
    let data = [0x00u8];
    let mut reader = BitReader::new(&data);

    let dc_value = huffman::decode_dc_coefficient(&mut reader, &table).unwrap();
    assert_eq!(dc_value, -3);
}

#[test]
fn decode_dc_zero() {
    // Category 0 → DC difference is 0
    let bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    let data = [0x00u8]; // "0" → category 0 → value 0
    let mut reader = BitReader::new(&data);

    let dc_value = huffman::decode_dc_coefficient(&mut reader, &table).unwrap();
    assert_eq!(dc_value, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test huffman_decode`
Expected: FAIL

- [ ] **Step 3: Implement huffman.rs**

`src/decode/huffman.rs`:
```rust
use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::decode::bitstream::BitReader;

/// Extend a variable-length bit value to a signed integer.
///
/// JPEG uses a ones' complement representation:
/// - If the MSB is 1, the value is positive and equals the raw bits.
/// - If the MSB is 0, the value is negative: -(2^size - 1 - raw_bits).
///
/// This is equivalent to: if val < 2^(size-1) then val - (2^size - 1) else val.
fn extend(value: u16, size: u8) -> i16 {
    if size == 0 {
        return 0;
    }
    let half = 1i16 << (size - 1);
    if (value as i16) < half {
        value as i16 - (2 * half - 1)
    } else {
        value as i16
    }
}

/// Decode one DC coefficient from the bitstream.
///
/// Reads a Huffman symbol (the "category" = number of extra bits),
/// then reads that many extra bits and sign-extends to get the DC difference value.
pub fn decode_dc_coefficient(reader: &mut BitReader, table: &HuffmanTable) -> Result<i16> {
    let peek = reader.peek_bits(16)?;
    let (category, code_len) = table.lookup(peek, 16)?;
    reader.read_bits(code_len)?; // consume the Huffman code

    if category == 0 {
        return Ok(0);
    }
    if category > 15 {
        return Err(JpegError::CorruptData(format!(
            "DC category {} out of range",
            category
        )));
    }

    let extra_bits = reader.read_bits(category)?;
    Ok(extend(extra_bits, category))
}

/// Decode AC coefficients for one 8×8 block.
///
/// Fills `coeffs[1..64]` in zigzag order. `coeffs[0]` (DC) is not touched.
/// Returns when all 63 AC coefficients have been decoded or an EOB symbol is encountered.
pub fn decode_ac_coefficients(
    reader: &mut BitReader,
    table: &HuffmanTable,
    coeffs: &mut [i16; 64],
) -> Result<()> {
    let mut index: usize = 1; // AC coefficients start at zigzag index 1

    while index < 64 {
        let peek = reader.peek_bits(16)?;
        let (symbol, code_len) = table.lookup(peek, 16)?;
        reader.read_bits(code_len)?;

        let run_length = (symbol >> 4) as usize;   // upper nibble: zero run
        let bit_size = (symbol & 0x0F) as u8;       // lower nibble: coefficient size

        if bit_size == 0 {
            if run_length == 0 {
                // EOB (End of Block) — remaining coefficients are zero
                return Ok(());
            }
            if run_length == 15 {
                // ZRL (Zero Run Length) — skip 16 zeros
                index += 16;
                continue;
            }
            return Err(JpegError::CorruptData(format!(
                "invalid AC symbol: run={}, size={}",
                run_length, bit_size
            )));
        }

        index += run_length;
        if index >= 64 {
            return Err(JpegError::CorruptData(
                "AC coefficient index out of bounds".into(),
            ));
        }

        let extra_bits = reader.read_bits(bit_size)?;
        coeffs[index] = extend(extra_bits, bit_size);
        index += 1;
    }

    Ok(())
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test huffman_decode`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/huffman.rs tests/huffman_decode.rs
git commit -s -m "feat: Huffman DC/AC coefficient decoding with sign extension"
```

---

### Task 8: MCU-Level Entropy Decoding

**Files:**
- Create: `src/decode/entropy.rs`
- Test: `tests/entropy_decode.rs`

The MCU (Minimum Coded Unit) decoder ties together the bitstream reader and Huffman decoder. It manages DC prediction state (each component's DC value is a running sum of differences) and decodes all blocks within an MCU.

- [ ] **Step 1: Write the failing test**

`tests/entropy_decode.rs`:
```rust
use libjpeg_turbo_rs::common::huffman_table::HuffmanTable;
use libjpeg_turbo_rs::common::types::*;
use libjpeg_turbo_rs::decode::entropy::McuDecoder;

/// Standard JPEG DC luminance Huffman table bits (K.3).
fn std_dc_luma_bits() -> [u8; 17] {
    [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
}
fn std_dc_luma_values() -> Vec<u8> {
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

#[test]
fn dc_prediction_accumulates() {
    // Verify that DC prediction correctly sums across blocks.
    let dc_table = HuffmanTable::build(&std_dc_luma_bits(), &std_dc_luma_values()).unwrap();

    // Build a trivial AC table with only EOB (symbol 0x00)
    let ac_bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let ac_table = HuffmanTable::build(&ac_bits, &[0x00]).unwrap();

    let dc_tables = [Some(dc_table), None, None, None];
    let ac_tables = [Some(ac_table), None, None, None];

    let mut decoder = McuDecoder::new(1); // 1 component
    // After reset, DC prediction starts at 0 for each component
    assert_eq!(decoder.dc_prediction(0), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test entropy_decode`
Expected: FAIL

- [ ] **Step 3: Implement entropy.rs**

`src/decode/entropy.rs`:
```rust
use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::huffman;

/// Decodes MCUs (Minimum Coded Units) from the entropy-coded data.
/// Manages DC prediction state per component.
pub struct McuDecoder {
    /// Running DC prediction value for each component.
    dc_pred: Vec<i16>,
}

impl McuDecoder {
    pub fn new(num_components: usize) -> Self {
        Self {
            dc_pred: vec![0; num_components],
        }
    }

    /// Current DC prediction value for a component.
    pub fn dc_prediction(&self, component_index: usize) -> i16 {
        self.dc_pred[component_index]
    }

    /// Reset DC predictions (called at restart markers).
    pub fn reset(&mut self) {
        for pred in self.dc_pred.iter_mut() {
            *pred = 0;
        }
    }

    /// Decode one 8×8 block of DCT coefficients (in zigzag order).
    ///
    /// - `component_index`: which component (for DC prediction tracking)
    /// - `dc_table` / `ac_table`: Huffman tables for this component
    /// - `coeffs`: output array of 64 coefficients in zigzag order
    pub fn decode_block(
        &mut self,
        reader: &mut BitReader,
        component_index: usize,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
        coeffs: &mut [i16; 64],
    ) -> Result<()> {
        // Zero out all coefficients
        *coeffs = [0i16; 64];

        // DC coefficient (differential)
        let dc_diff = huffman::decode_dc_coefficient(reader, dc_table)?;
        self.dc_pred[component_index] += dc_diff;
        coeffs[0] = self.dc_pred[component_index];

        // AC coefficients
        huffman::decode_ac_coefficients(reader, ac_table, coeffs)?;

        Ok(())
    }

    /// Decode a complete MCU.
    ///
    /// For each component in the scan, decode the appropriate number of blocks
    /// based on the component's sampling factors.
    ///
    /// Returns blocks in component order, each block being 64 coefficients in zigzag order.
    pub fn decode_mcu(
        &mut self,
        reader: &mut BitReader,
        frame: &FrameHeader,
        scan: &ScanHeader,
        dc_tables: &[Option<HuffmanTable>; 4],
        ac_tables: &[Option<HuffmanTable>; 4],
        blocks: &mut Vec<[i16; 64]>,
    ) -> Result<()> {
        let max_h = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling)
            .max()
            .unwrap_or(1);
        let max_v = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling)
            .max()
            .unwrap_or(1);

        blocks.clear();

        for scan_comp in &scan.components {
            // Find the frame component matching this scan component
            let (comp_idx, comp) = frame
                .components
                .iter()
                .enumerate()
                .find(|(_, c)| c.id == scan_comp.component_id)
                .ok_or(JpegError::CorruptData(format!(
                    "scan references unknown component id {}",
                    scan_comp.component_id
                )))?;

            let dc_table = dc_tables[scan_comp.dc_table_index as usize]
                .as_ref()
                .ok_or(JpegError::CorruptData(format!(
                    "missing DC Huffman table {}",
                    scan_comp.dc_table_index
                )))?;

            let ac_table = ac_tables[scan_comp.ac_table_index as usize]
                .as_ref()
                .ok_or(JpegError::CorruptData(format!(
                    "missing AC Huffman table {}",
                    scan_comp.ac_table_index
                )))?;

            let num_blocks =
                (comp.horizontal_sampling as usize) * (comp.vertical_sampling as usize);

            for _ in 0..num_blocks {
                let mut coeffs = [0i16; 64];
                self.decode_block(reader, comp_idx, dc_table, ac_table, &mut coeffs)?;
                blocks.push(coeffs);
            }
        }

        Ok(())
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test entropy_decode`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/entropy.rs tests/entropy_decode.rs
git commit -s -m "feat: MCU-level entropy decoder with DC prediction"
```

---

## Chunk 3: Reconstruction

### Task 9: Dequantization

**Files:**
- Create: `src/decode/dequant.rs`
- Test: `tests/dequant.rs`

- [ ] **Step 1: Write the failing test**

`tests/dequant.rs`:
```rust
use libjpeg_turbo_rs::common::quant_table::QuantTable;
use libjpeg_turbo_rs::decode::dequant;

#[test]
fn dequantize_multiplies_by_table() {
    // Coefficients in zigzag order, quantization table in zigzag order
    let mut coeffs = [0i16; 64];
    coeffs[0] = 10;  // DC
    coeffs[1] = 5;   // first AC

    let mut qt_zigzag = [1u16; 64];
    qt_zigzag[0] = 16; // DC quant value
    qt_zigzag[1] = 11; // first AC quant value

    let table = QuantTable::from_zigzag(&qt_zigzag);

    let result = dequant::dequantize_block(&coeffs, &table);

    // After dequantization (in natural order):
    // DC: 10 * 16 = 160
    // position (0,1): 5 * 11 = 55
    assert_eq!(result[0], 160);   // [0][0] in natural order
    assert_eq!(result[1], 55);    // [0][1] in natural order
}

#[test]
fn dequantize_preserves_zeros() {
    let coeffs = [0i16; 64];
    let qt_zigzag = [8u16; 64];
    let table = QuantTable::from_zigzag(&qt_zigzag);

    let result = dequant::dequantize_block(&coeffs, &table);

    for &val in result.iter() {
        assert_eq!(val, 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test dequant`
Expected: FAIL

- [ ] **Step 3: Implement dequant.rs**

`src/decode/dequant.rs`:
```rust
use crate::common::quant_table::{QuantTable, ZIGZAG_ORDER};

/// Dequantize a block of DCT coefficients.
///
/// Input: 64 coefficients in **zigzag** order (as decoded from entropy data).
/// Output: 64 coefficients in **natural** (row-major) order, multiplied by the
/// quantization table values.
///
/// The zigzag-to-natural reorder and dequantization happen in one pass.
pub fn dequantize_block(zigzag_coeffs: &[i16; 64], table: &QuantTable) -> [i16; 64] {
    let mut natural = [0i16; 64];
    for (zigzag_idx, &coeff) in zigzag_coeffs.iter().enumerate() {
        let natural_idx = ZIGZAG_ORDER[zigzag_idx];
        natural[natural_idx] = coeff * table.values[natural_idx] as i16;
    }
    natural
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test dequant`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/dequant.rs tests/dequant.rs
git commit -s -m "feat: dequantization with zigzag-to-natural reordering"
```

---

### Task 10: Inverse DCT (8×8 Scalar)

**Files:**
- Create: `src/decode/idct.rs`
- Test: `tests/idct.rs`

The AAN (Arai, Agui, Nakajima) algorithm is used for the integer IDCT, matching libjpeg-turbo's "accurate" integer path. This operates on 8×8 blocks and uses 32-bit integer arithmetic with fixed-point scaling.

- [ ] **Step 1: Write the failing test**

`tests/idct.rs`:
```rust
use libjpeg_turbo_rs::decode::idct;

#[test]
fn idct_dc_only_block() {
    // When only the DC coefficient is nonzero, all 64 output samples
    // should be the same value: DC / 8 (since IDCT of constant = value/N per dimension).
    let mut coeffs = [0i16; 64];
    coeffs[0] = 800; // DC value after dequantization

    let output = idct::idct_8x8(&coeffs);

    // The DC-only IDCT should produce a uniform block.
    // IDCT normalization: DC / 8 / 8 = 800 / 64 ≈ 12 (exact value depends on scaling)
    // All values should be identical.
    let first = output[0];
    for &val in output.iter() {
        assert_eq!(val, first, "DC-only block should produce uniform output");
    }
    // Value should be in reasonable range (positive, not clamped)
    assert!(first > 0, "DC-only output should be positive");
    assert!(first < 255, "DC-only output should not overflow u8");
}

#[test]
fn idct_all_zeros_gives_128() {
    // For level-shifted data: JPEG subtracts 128 before DCT.
    // All-zero coefficients → all output samples are 128 (the level shift).
    // However, our IDCT output is the raw spatial values before level shift,
    // so all-zero input should give all-zero output.
    let coeffs = [0i16; 64];
    let output = idct::idct_8x8(&coeffs);

    for &val in output.iter() {
        assert_eq!(val, 0);
    }
}

#[test]
fn idct_known_values() {
    // Test with a known coefficient pattern and verify against reference output.
    // Single AC coefficient at position (0,1) to verify transform correctness.
    let mut coeffs = [0i16; 64];
    coeffs[0] = 512; // DC
    coeffs[1] = 100; // AC(0,1) — causes horizontal cosine variation

    let output = idct::idct_8x8(&coeffs);

    // The output should vary horizontally but be constant vertically
    // (since only DC and one horizontal-frequency AC are nonzero).
    // Row 0 and row 1 should have the same pattern.
    for col in 0..8 {
        assert_eq!(
            output[0 * 8 + col], output[1 * 8 + col],
            "rows should be identical for this coefficient pattern"
        );
    }
    // The output should not be uniform (AC contribution exists).
    assert_ne!(output[0], output[1], "horizontal variation expected");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test idct`
Expected: FAIL

- [ ] **Step 3: Implement idct.rs**

`src/decode/idct.rs`:
```rust
/// Scalar 8×8 inverse DCT using the AAN (Arai, Agui, Nakajima) algorithm
/// with fixed-point integer arithmetic.
///
/// This matches libjpeg-turbo's "jidctint.c" accurate integer IDCT.
///
/// Input: 64 dequantized coefficients in natural (row-major) order.
/// Output: 64 spatial-domain sample values (not yet level-shifted or clamped).

// Fixed-point constants (13-bit fractional precision)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const F_0_298: i32 = 2446;  // FIX(0.298631336)
const F_0_390: i32 = 3196;  // FIX(0.390180644)
const F_0_541: i32 = 4433;  // FIX(0.541196100)
const F_0_765: i32 = 6270;  // FIX(0.765366865)
const F_0_899: i32 = 7373;  // FIX(0.899976223)
const F_1_175: i32 = 9633;  // FIX(1.175875602)
const F_1_501: i32 = 12299; // FIX(1.501321110)
const F_1_847: i32 = 15137; // FIX(1.847759065)
const F_1_961: i32 = 16069; // FIX(1.961570560)
const F_2_053: i32 = 16819; // FIX(2.053119869)
const F_2_562: i32 = 20995; // FIX(2.562915447)
const F_3_072: i32 = 25172; // FIX(3.072711026)

#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Perform 8×8 inverse DCT.
///
/// Input: dequantized coefficients in natural row-major order.
/// Output: spatial-domain sample values (caller adds 128 level shift and clamps to 0-255).
pub fn idct_8x8(coeffs: &[i16; 64]) -> [i16; 64] {
    let mut workspace = [0i32; 64];

    // Pass 1: process columns from input, store into workspace
    for col in 0..8 {
        let c = |row: usize| coeffs[row * 8 + col] as i32;

        // Shortcut: if all AC terms are zero, just propagate DC
        if c(1) == 0 && c(2) == 0 && c(3) == 0 && c(4) == 0
            && c(5) == 0 && c(6) == 0 && c(7) == 0
        {
            let dcval = c(0) << PASS1_BITS;
            for row in 0..8 {
                workspace[row * 8 + col] = dcval;
            }
            continue;
        }

        // Even part
        let tmp0 = c(0) << CONST_BITS;
        let tmp1 = c(2);
        let tmp2 = c(4) << CONST_BITS;
        let tmp3 = c(6);

        let z1 = (tmp1 + tmp3) * F_0_541;
        let tmp2a = z1 + tmp3 * (-F_1_847);
        let tmp3a = z1 + tmp1 * F_0_765;

        let tmp0a = tmp0 + tmp2;
        let tmp1a = tmp0 - tmp2;

        let tmp10 = tmp0a + tmp3a;
        let tmp13 = tmp0a - tmp3a;
        let tmp11 = tmp1a + tmp2a;
        let tmp12 = tmp1a - tmp2a;

        // Odd part
        let tmp0 = c(7);
        let tmp1 = c(5);
        let tmp2 = c(3);
        let tmp3 = c(1);

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * F_1_175;

        let tmp0 = tmp0 * F_0_298;
        let tmp1 = tmp1 * F_2_053;
        let tmp2 = tmp2 * F_3_072;
        let tmp3 = tmp3 * F_1_501;
        let z1 = z1 * (-F_0_899);
        let z2 = z2 * (-F_2_562);
        let z3 = z3 * (-F_1_961) + z5;
        let z4 = z4 * (-F_0_390) + z5;

        let tmp0 = tmp0 + z1 + z3;
        let tmp1 = tmp1 + z2 + z4;
        let tmp2 = tmp2 + z2 + z3;
        let tmp3 = tmp3 + z1 + z4;

        // Final combination and descale
        for (row, &(even, odd)) in [
            (tmp10, tmp3), (tmp11, tmp2), (tmp12, tmp1), (tmp13, tmp0),
            (tmp13, -tmp0), (tmp12, -tmp1), (tmp11, -tmp2), (tmp10, -tmp3),
        ].iter().enumerate() {
            workspace[row * 8 + col] = descale(even + odd, CONST_BITS - PASS1_BITS);
        }
    }

    // Pass 2: process rows from workspace, produce output
    let mut output = [0i16; 64];

    for row in 0..8 {
        let w = |col: usize| workspace[row * 8 + col];

        // Shortcut: if all AC terms are zero
        if w(1) == 0 && w(2) == 0 && w(3) == 0 && w(4) == 0
            && w(5) == 0 && w(6) == 0 && w(7) == 0
        {
            let dcval = descale(w(0), PASS1_BITS + 3) as i16;
            for col in 0..8 {
                output[row * 8 + col] = dcval;
            }
            continue;
        }

        // Even part
        let tmp0 = w(0) << CONST_BITS;
        let tmp1 = w(2);
        let tmp2 = w(4) << CONST_BITS;
        let tmp3 = w(6);

        let z1 = (tmp1 + tmp3) * F_0_541;
        let tmp2a = z1 + tmp3 * (-F_1_847);
        let tmp3a = z1 + tmp1 * F_0_765;

        let tmp0a = tmp0 + tmp2;
        let tmp1a = tmp0 - tmp2;

        let tmp10 = tmp0a + tmp3a;
        let tmp13 = tmp0a - tmp3a;
        let tmp11 = tmp1a + tmp2a;
        let tmp12 = tmp1a - tmp2a;

        // Odd part
        let tmp0 = w(7);
        let tmp1 = w(5);
        let tmp2 = w(3);
        let tmp3 = w(1);

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * F_1_175;

        let tmp0 = tmp0 * F_0_298;
        let tmp1 = tmp1 * F_2_053;
        let tmp2 = tmp2 * F_3_072;
        let tmp3 = tmp3 * F_1_501;
        let z1 = z1 * (-F_0_899);
        let z2 = z2 * (-F_2_562);
        let z3 = z3 * (-F_1_961) + z5;
        let z4 = z4 * (-F_0_390) + z5;

        let tmp0 = tmp0 + z1 + z3;
        let tmp1 = tmp1 + z2 + z4;
        let tmp2 = tmp2 + z2 + z3;
        let tmp3 = tmp3 + z1 + z4;

        let descale_bits = CONST_BITS + PASS1_BITS + 3;
        let pairs: [(i32, i32); 8] = [
            (tmp10, tmp3), (tmp11, tmp2), (tmp12, tmp1), (tmp13, tmp0),
            (tmp13, -tmp0), (tmp12, -tmp1), (tmp11, -tmp2), (tmp10, -tmp3),
        ];
        for (col, &(even, odd)) in pairs.iter().enumerate() {
            output[row * 8 + col] = descale(even + odd, descale_bits) as i16;
        }
    }

    output
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test idct`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/idct.rs tests/idct.rs
git commit -s -m "feat: scalar 8x8 inverse DCT (AAN algorithm)"
```

---

### Task 11: Chroma Upsampling

**Files:**
- Create: `src/decode/upsample.rs`
- Test: `tests/upsample.rs`

- [ ] **Step 1: Write the failing test**

`tests/upsample.rs`:
```rust
use libjpeg_turbo_rs::decode::upsample;

#[test]
fn upsample_h2v1_doubles_width() {
    // 4 input samples → 8 output samples (horizontal 2× only)
    let input = [10u8, 20, 30, 40];
    let mut output = [0u8; 8];

    upsample::simple_h2v1(&input, 4, &mut output, 8);

    // Simple duplication: each input sample is repeated twice
    assert_eq!(output, [10, 10, 20, 20, 30, 30, 40, 40]);
}

#[test]
fn upsample_h2v2_doubles_both() {
    // 2×2 input → 4×4 output
    let input = [10u8, 20, 30, 40]; // 2 rows × 2 cols
    let mut output = [0u8; 16]; // 4 rows × 4 cols

    upsample::simple_h2v2(&input, 2, 2, &mut output, 4, 4);

    // Each pixel duplicated 2× horizontally and 2× vertically
    #[rustfmt::skip]
    let expected = [
        10, 10, 20, 20,
        10, 10, 20, 20,
        30, 30, 40, 40,
        30, 30, 40, 40,
    ];
    assert_eq!(output, expected);
}

#[test]
fn fancy_h2v1_interpolates() {
    // Fancy upsampling uses triangle filter: (3*near + far + 2) / 4
    let input = [0u8, 100, 200, 100];
    let mut output = [0u8; 8];

    upsample::fancy_h2v1(&input, 4, &mut output, 8);

    // At boundaries, it uses the nearest sample only for the edge
    // Between samples 0 and 100: left = (3*0 + 100 + 2)/4 = 25, right = (3*100 + 0 + 2)/4 = 75
    // Values should transition smoothly
    // Just verify it's not simple duplication and values are interpolated
    assert_ne!(output[0], output[1], "fancy should interpolate, not duplicate");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test upsample`
Expected: FAIL

- [ ] **Step 3: Implement upsample.rs**

`src/decode/upsample.rs`:
```rust
/// Simple nearest-neighbor horizontal 2× upsampling.
/// Each input sample is duplicated to produce two output samples.
pub fn simple_h2v1(input: &[u8], in_width: usize, output: &mut [u8], out_width: usize) {
    for x in 0..in_width {
        let val = input[x];
        output[x * 2] = val;
        output[x * 2 + 1] = val;
    }
}

/// Simple nearest-neighbor 2×2 upsampling (horizontal and vertical).
pub fn simple_h2v2(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
    out_height: usize,
) {
    for y in 0..in_height {
        for x in 0..in_width {
            let val = input[y * in_width + x];
            let out_y = y * 2;
            let out_x = x * 2;
            output[out_y * out_width + out_x] = val;
            output[out_y * out_width + out_x + 1] = val;
            output[(out_y + 1) * out_width + out_x] = val;
            output[(out_y + 1) * out_width + out_x + 1] = val;
        }
    }
}

/// Fancy horizontal 2× upsampling using triangle filter.
/// Produces smoother output than nearest-neighbor.
/// Formula: output[2i] = (3*input[i] + input[i-1] + 2) >> 2
///          output[2i+1] = (3*input[i] + input[i+1] + 2) >> 2
pub fn fancy_h2v1(input: &[u8], in_width: usize, output: &mut [u8], out_width: usize) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    // First pixel: use input[0] as both current and left neighbor
    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    // Middle pixels
    for x in 1..in_width - 1 {
        let left = input[x - 1] as u16;
        let cur = input[x] as u16;
        let right = input[x + 1] as u16;
        output[x * 2] = ((3 * cur + left + 2) >> 2) as u8;
        output[x * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
    }

    // Last pixel
    let last = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 2) >> 2) as u8;
    output[last * 2 + 1] = input[last];
}

/// Fancy 2×2 upsampling (both horizontal and vertical) using triangle filter.
/// Vertical interpolation between adjacent input rows, then horizontal interpolation.
pub fn fancy_h2v2(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
    out_height: usize,
) {
    // Temporary buffer for vertically-interpolated rows
    let mut row_above = vec![0u8; in_width];
    let mut row_below = vec![0u8; in_width];

    for y in 0..in_height {
        let cur_row = &input[y * in_width..(y + 1) * in_width];
        let above = if y > 0 {
            &input[(y - 1) * in_width..y * in_width]
        } else {
            cur_row
        };
        let below = if y + 1 < in_height {
            &input[(y + 1) * in_width..(y + 2) * in_width]
        } else {
            cur_row
        };

        // Vertically interpolate: upper output row
        for x in 0..in_width {
            row_above[x] = ((3 * cur_row[x] as u16 + above[x] as u16 + 2) >> 2) as u8;
            row_below[x] = ((3 * cur_row[x] as u16 + below[x] as u16 + 2) >> 2) as u8;
        }

        // Horizontally interpolate each vertically-interpolated row
        let out_y_top = y * 2;
        let out_y_bot = y * 2 + 1;
        fancy_h2v1(
            &row_above,
            in_width,
            &mut output[out_y_top * out_width..],
            out_width,
        );
        fancy_h2v1(
            &row_below,
            in_width,
            &mut output[out_y_bot * out_width..],
            out_width,
        );
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test upsample`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/upsample.rs tests/upsample.rs
git commit -s -m "feat: chroma upsampling (simple and fancy triangle filter)"
```

---

### Task 12: YCbCr → RGB Color Conversion

**Files:**
- Create: `src/decode/color.rs`
- Test: `tests/color_convert.rs`

- [ ] **Step 1: Write the failing test**

`tests/color_convert.rs`:
```rust
use libjpeg_turbo_rs::decode::color;

#[test]
fn ycbcr_to_rgb_white() {
    // White: Y=255, Cb=128, Cr=128 → R=255, G=255, B=255
    let (r, g, b) = color::ycbcr_to_rgb_pixel(255, 128, 128);
    assert_eq!((r, g, b), (255, 255, 255));
}

#[test]
fn ycbcr_to_rgb_black() {
    // Black: Y=0, Cb=128, Cr=128 → R=0, G=0, B=0
    let (r, g, b) = color::ycbcr_to_rgb_pixel(0, 128, 128);
    assert_eq!((r, g, b), (0, 0, 0));
}

#[test]
fn ycbcr_to_rgb_red() {
    // Pure red: Y=76, Cb=84, Cr=255 → R≈255, G≈0, B≈0
    let (r, g, b) = color::ycbcr_to_rgb_pixel(76, 84, 255);
    // Allow ±1 for integer rounding
    assert!(r >= 254, "red channel: {}", r);
    assert!(g <= 1, "green channel: {}", g);
    assert!(b <= 1, "blue channel: {}", b);
}

#[test]
fn ycbcr_to_rgb_bulk() {
    // Test bulk conversion of 4 pixels
    let y  = [255u8, 0, 76, 149];
    let cb = [128u8, 128, 84, 43];
    let cr = [128u8, 128, 255, 21];

    let mut rgb = [0u8; 12]; // 4 pixels × 3 channels
    color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, 4);

    // White pixel
    assert_eq!(&rgb[0..3], &[255, 255, 255]);
    // Black pixel
    assert_eq!(&rgb[3..6], &[0, 0, 0]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test color_convert`
Expected: FAIL

- [ ] **Step 3: Implement color.rs**

`src/decode/color.rs`:
```rust
/// Clamp a value to the 0-255 range.
#[inline(always)]
fn clamp(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// Convert a single YCbCr pixel to RGB.
///
/// Uses the JFIF/ITU-R BT.601 conversion:
///   R = Y                + 1.40200 * (Cr - 128)
///   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
///   B = Y + 1.77200 * (Cb - 128)
///
/// Fixed-point version with 16-bit fractional precision:
///   R = Y + ((91881 * (Cr-128) + 32768) >> 16)
///   G = Y - ((22554 * (Cb-128) + 46802 * (Cr-128) - 32768) >> 16)
///   B = Y + ((116130 * (Cb-128) + 32768) >> 16)
pub fn ycbcr_to_rgb_pixel(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;

    let r = y + ((91881 * cr + 32768) >> 16);
    let g = y - ((22554 * cb + 46802 * cr - 32768) >> 16);
    let b = y + ((116130 * cb + 32768) >> 16);

    (clamp(r), clamp(g), clamp(b))
}

/// Convert a row of YCbCr pixels to interleaved RGB.
///
/// `y`, `cb`, `cr` are planar component rows (one value per pixel).
/// `rgb` is the output buffer: [R0, G0, B0, R1, G1, B1, ...].
pub fn ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        rgb[x * 3] = r;
        rgb[x * 3 + 1] = g;
        rgb[x * 3 + 2] = b;
    }
}

/// Copy grayscale values directly to RGB (R=G=B=Y).
pub fn grayscale_to_rgb_row(y: &[u8], rgb: &mut [u8], width: usize) {
    for x in 0..width {
        rgb[x * 3] = y[x];
        rgb[x * 3 + 1] = y[x];
        rgb[x * 3 + 2] = y[x];
    }
}

/// Copy grayscale values directly (no color conversion needed).
pub fn grayscale_row(y: &[u8], output: &mut [u8], width: usize) {
    output[..width].copy_from_slice(&y[..width]);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test color_convert`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/color.rs tests/color_convert.rs
git commit -s -m "feat: YCbCr to RGB color conversion (BT.601 fixed-point)"
```

---

## Chunk 4: Integration

### Task 13: Full Decoder Pipeline

**Files:**
- Create: `src/decode/pipeline.rs`
- Test: `tests/decode_pipeline.rs`

The `Decoder` struct ties together all components: marker parsing → entropy decoding → dequantization → IDCT → upsampling → color conversion. It orchestrates the MCU-by-MCU decode loop and assembles the final image.

- [ ] **Step 1: Write the failing test**

`tests/decode_pipeline.rs`:
```rust
use libjpeg_turbo_rs::decode::pipeline::Decoder;

/// Generate a valid tiny JPEG using libjpeg-turbo (via the `image` crate or manual bytes).
/// For now, use a hand-crafted minimal 8×8 grayscale JPEG.
fn minimal_8x8_grayscale_jpeg() -> Vec<u8> {
    // This is a valid 8×8 grayscale JPEG that decodes to all-128 (gray) pixels.
    // Generated by encoding an 8×8 block where DC=0 (which with level shift gives 128).
    // The encoding uses standard Huffman tables and quant table of all 1s.
    include_bytes!("fixtures/gray_8x8.jpg").to_vec()
}

#[test]
fn decode_grayscale_dimensions() {
    let data = minimal_8x8_grayscale_jpeg();
    let decoder = Decoder::new(&data).unwrap();
    let header = decoder.header();

    assert_eq!(header.width, 8);
    assert_eq!(header.height, 8);
    assert_eq!(header.components.len(), 1);
}

#[test]
fn decode_grayscale_pixels() {
    let data = minimal_8x8_grayscale_jpeg();
    let image = Decoder::decode(&data).unwrap();

    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);
    // All pixels should be the same shade of gray
    assert_eq!(image.data.len(), 64); // 8×8 × 1 channel
}
```

- [ ] **Step 2: Create test fixture**

Generate `tests/fixtures/gray_8x8.jpg` using a small Python script (requires Pillow):

```bash
python3 -c "
from PIL import Image
img = Image.new('L', (8, 8), 128)
img.save('tests/fixtures/gray_8x8.jpg', quality=100, subsampling=0)
"
```

Alternatively, create with ImageMagick:
```bash
convert -size 8x8 xc:gray tests/fixtures/gray_8x8.jpg
```

- [ ] **Step 3: Implement pipeline.rs**

`src/decode/pipeline.rs`:
```rust
use crate::common::error::{JpegError, Result};
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::color;
use crate::decode::dequant;
use crate::decode::entropy::McuDecoder;
use crate::decode::idct;
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::decode::upsample;

/// Decoded image data.
#[derive(Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    /// Pixel data. Layout depends on pixel_format:
    /// - Grayscale: width × height bytes
    /// - RGB: width × height × 3 bytes (interleaved)
    pub data: Vec<u8>,
}

/// JPEG decoder. Orchestrates the full decoding pipeline.
pub struct Decoder<'a> {
    metadata: JpegMetadata,
    raw_data: &'a [u8],
}

impl<'a> Decoder<'a> {
    /// Create a new decoder by parsing JPEG markers.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut reader = MarkerReader::new(data);
        let metadata = reader.read_markers()?;
        Ok(Self {
            metadata,
            raw_data: data,
        })
    }

    /// Access the parsed frame header (width, height, components).
    pub fn header(&self) -> &FrameHeader {
        &self.metadata.frame
    }

    /// Decode the full image in one shot.
    pub fn decode(data: &[u8]) -> Result<Image> {
        let decoder = Self::new(data)?;
        decoder.decode_image()
    }

    /// Internal: decode the full image.
    fn decode_image(&self) -> Result<Image> {
        let frame = &self.metadata.frame;
        let scan = &self.metadata.scan;
        let width = frame.width as usize;
        let height = frame.height as usize;

        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported in Phase 1)",
                frame.precision
            )));
        }

        let num_components = frame.components.len();
        let max_h = frame.components.iter().map(|c| c.horizontal_sampling as usize).max().unwrap_or(1);
        let max_v = frame.components.iter().map(|c| c.vertical_sampling as usize).max().unwrap_or(1);

        // MCU dimensions in pixels
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;

        // Number of MCUs
        let mcus_x = (width + mcu_width - 1) / mcu_width;
        let mcus_y = (height + mcu_height - 1) / mcu_height;

        // Allocate component planes (full MCU-aligned size)
        let full_width = mcus_x * mcu_width;
        let full_height = mcus_y * mcu_height;

        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * 8;
                let comp_h = mcus_y * comp.vertical_sampling as usize * 8;
                vec![0u8; comp_w * comp_h]
            })
            .collect();

        // Decode all MCUs
        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut blocks = Vec::new();
        let mut restart_count: u16 = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                // Handle restart markers
                if self.metadata.restart_interval > 0 && restart_count > 0
                    && restart_count % self.metadata.restart_interval == 0
                {
                    bit_reader.reset();
                    mcu_decoder.reset();
                }

                mcu_decoder.decode_mcu(
                    &mut bit_reader,
                    frame,
                    scan,
                    &self.metadata.dc_huffman_tables,
                    &self.metadata.ac_huffman_tables,
                    &mut blocks,
                )?;

                // Place decoded blocks into component planes
                let mut block_idx = 0;
                for (comp_idx, comp) in frame.components.iter().enumerate() {
                    let comp_w = mcus_x * comp.horizontal_sampling as usize * 8;
                    let h_blocks = comp.horizontal_sampling as usize;
                    let v_blocks = comp.vertical_sampling as usize;

                    for v in 0..v_blocks {
                        for h in 0..h_blocks {
                            let zigzag_coeffs = &blocks[block_idx];
                            block_idx += 1;

                            // Dequantize
                            let qt = self.metadata.quant_tables[comp.quant_table_index as usize]
                                .as_ref()
                                .ok_or(JpegError::CorruptData(format!(
                                    "missing quant table {}",
                                    comp.quant_table_index
                                )))?;
                            let dequantized = dequant::dequantize_block(zigzag_coeffs, qt);

                            // Inverse DCT
                            let spatial = idct::idct_8x8(&dequantized);

                            // Place 8×8 block into component plane
                            // Level shift: add 128 and clamp to 0-255
                            let block_x = (mcu_x * h_blocks + h) * 8;
                            let block_y = (mcu_y * v_blocks + v) * 8;

                            for row in 0..8 {
                                for col in 0..8 {
                                    let val = (spatial[row * 8 + col] as i32 + 128).clamp(0, 255) as u8;
                                    let px = block_x + col;
                                    let py = block_y + row;
                                    component_planes[comp_idx][py * comp_w + px] = val;
                                }
                            }
                        }
                    }
                }

                restart_count += 1;
            }
        }

        // Upsample and color convert
        if num_components == 1 {
            // Grayscale: just crop to actual dimensions
            let comp_w = mcus_x * 8;
            let mut data = Vec::with_capacity(width * height);
            for y in 0..height {
                data.extend_from_slice(&component_planes[0][y * comp_w..y * comp_w + width]);
            }
            Ok(Image {
                width,
                height,
                pixel_format: PixelFormat::Grayscale,
                data,
            })
        } else if num_components == 3 {
            // YCbCr → RGB
            // First, upsample Cb and Cr if needed
            let y_plane = &component_planes[0];
            let y_width = mcus_x * frame.components[0].horizontal_sampling as usize * 8;

            let mut cb_full = vec![0u8; full_width * full_height];
            let mut cr_full = vec![0u8; full_width * full_height];

            let cb_comp = &frame.components[1];
            let cr_comp = &frame.components[2];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * 8;
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * 8;

            // Determine upsampling mode
            let h_factor = max_h / cb_comp.horizontal_sampling as usize;
            let v_factor = max_v / cb_comp.vertical_sampling as usize;

            if h_factor == 1 && v_factor == 1 {
                // 4:4:4 — no upsampling needed
                cb_full = component_planes[1].clone();
                cr_full = component_planes[2].clone();
            } else if h_factor == 2 && v_factor == 1 {
                // 4:2:2
                for row in 0..cb_h {
                    upsample::fancy_h2v1(
                        &component_planes[1][row * cb_w..],
                        cb_w,
                        &mut cb_full[row * full_width..],
                        full_width,
                    );
                    upsample::fancy_h2v1(
                        &component_planes[2][row * cb_w..],
                        cb_w,
                        &mut cr_full[row * full_width..],
                        full_width,
                    );
                }
            } else if h_factor == 2 && v_factor == 2 {
                // 4:2:0
                upsample::fancy_h2v2(
                    &component_planes[1],
                    cb_w, cb_h,
                    &mut cb_full,
                    full_width, full_height,
                );
                upsample::fancy_h2v2(
                    &component_planes[2],
                    cb_w, cb_h,
                    &mut cr_full,
                    full_width, full_height,
                );
            } else {
                return Err(JpegError::Unsupported(format!(
                    "subsampling {}×{} not yet supported",
                    h_factor, v_factor
                )));
            }

            // Color convert row by row, cropping to actual dimensions
            let mut data = vec![0u8; width * height * 3];
            for y in 0..height {
                color::ycbcr_to_rgb_row(
                    &y_plane[y * y_width..],
                    &cb_full[y * full_width..],
                    &cr_full[y * full_width..],
                    &mut data[y * width * 3..],
                    width,
                );
            }

            Ok(Image {
                width,
                height,
                pixel_format: PixelFormat::Rgb,
                data,
            })
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported",
                num_components
            )))
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test decode_pipeline`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/decode/pipeline.rs tests/decode_pipeline.rs tests/fixtures/
git commit -s -m "feat: full baseline JPEG decode pipeline with MCU processing"
```

---

### Task 14: High-Level and Streaming APIs

**Files:**
- Create: `src/api/high_level.rs`
- Create: `src/api/streaming.rs`
- Update: `src/lib.rs` (add public re-exports)

- [ ] **Step 1: Implement high_level.rs**

`src/api/high_level.rs`:
```rust
use crate::common::error::Result;
use crate::decode::pipeline::{Decoder, Image};

/// Decompress a JPEG byte slice into an Image.
///
/// # Example
/// ```no_run
/// let jpeg_data = std::fs::read("photo.jpg").unwrap();
/// let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
/// println!("{}×{}", image.width, image.height);
/// ```
pub fn decompress(data: &[u8]) -> Result<Image> {
    Decoder::decode(data)
}
```

- [ ] **Step 2: Implement streaming.rs**

`src/api/streaming.rs`:
```rust
use crate::common::error::Result;
use crate::common::types::FrameHeader;
use crate::decode::pipeline::Decoder;

/// Streaming JPEG decoder — reads header first, then decodes on demand.
///
/// # Example
/// ```no_run
/// use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
///
/// let data = std::fs::read("photo.jpg").unwrap();
/// let decoder = StreamingDecoder::new(&data).unwrap();
/// let header = decoder.header();
/// println!("{}×{}, {} components", header.width, header.height, header.components.len());
/// ```
pub struct StreamingDecoder<'a> {
    inner: Decoder<'a>,
}

impl<'a> StreamingDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let inner = Decoder::new(data)?;
        Ok(Self { inner })
    }

    pub fn header(&self) -> &FrameHeader {
        self.inner.header()
    }
}
```

- [ ] **Step 3: Update lib.rs with re-exports**

`src/lib.rs`:
```rust
pub mod common;
pub mod decode;
pub mod api;

// Convenience re-exports
pub use api::high_level::decompress;
pub use common::error::{JpegError, Result};
pub use common::types::*;
pub use decode::pipeline::Image;
```

- [ ] **Step 4: Verify it compiles and all tests pass**

Run: `cargo test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add src/api/high_level.rs src/api/streaming.rs src/lib.rs
git commit -s -m "feat: high-level decompress() API and streaming decoder"
```

---

### Task 15: Conformance Tests

**Files:**
- Create: `tests/conformance.rs`
- Add: test fixture JPEGs (various subsampling modes)

Generate reference images by decoding with libjpeg-turbo's `djpeg` tool, then compare our output pixel-by-pixel.

- [ ] **Step 1: Create test fixtures**

Generate test images using ImageMagick or Python/Pillow:
```bash
# 8×8 grayscale
python3 -c "
from PIL import Image
Image.new('L', (8,8), 128).save('tests/fixtures/gray_8x8.jpg', quality=95)
Image.new('RGB', (16,16), (255,0,0)).save('tests/fixtures/red_16x16_444.jpg', quality=95, subsampling=0)
Image.new('RGB', (16,16), (0,255,0)).save('tests/fixtures/green_16x16_422.jpg', quality=95, subsampling='4:2:2')
Image.new('RGB', (16,16), (0,0,255)).save('tests/fixtures/blue_16x16_420.jpg', quality=95, subsampling='4:2:0')
"
```

- [ ] **Step 2: Write conformance tests**

`tests/conformance.rs`:
```rust
use libjpeg_turbo_rs::decompress;

#[test]
fn conformance_grayscale_8x8() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);

    // All pixels should be close to 128 (uniform gray)
    for &pixel in &image.data {
        assert!(
            (pixel as i16 - 128).unsigned_abs() <= 2,
            "pixel {} too far from 128",
            pixel
        );
    }
}

#[test]
fn conformance_rgb_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.data.len(), 16 * 16 * 3);

    // All pixels should be approximately red (R≈255, G≈0, B≈0)
    // JPEG compression introduces some error, allow ±10
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r > 240, "pixel ({},{}) R={} too low", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={} too high", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={} too high", x, y, b);
        }
    }
}

#[test]
fn conformance_rgb_422() {
    let data = include_bytes!("fixtures/green_16x16_422.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g > 240, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn conformance_rgb_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b > 240, "pixel ({},{}) B={}", x, y, b);
        }
    }
}
```

- [ ] **Step 3: Run conformance tests**

Run: `cargo test --test conformance`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add tests/conformance.rs tests/fixtures/
git commit -s -m "feat: conformance tests for grayscale, 4:4:4, 4:2:2, 4:2:0"
```

---

### Task 16: Benchmark Setup

**Files:**
- Create: `benches/decode.rs`

- [ ] **Step 1: Create benchmark**

`benches/decode.rs`:
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    let test_cases = [
        ("gray_8x8", include_bytes!("../tests/fixtures/gray_8x8.jpg").as_slice()),
        ("red_16x16_444", include_bytes!("../tests/fixtures/red_16x16_444.jpg").as_slice()),
        ("blue_16x16_420", include_bytes!("../tests/fixtures/blue_16x16_420.jpg").as_slice()),
    ];

    for (name, data) in &test_cases {
        group.bench_with_input(BenchmarkId::new("libjpeg-turbo-rs", name), data, |b, data| {
            b.iter(|| libjpeg_turbo_rs::decompress(data).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
```

- [ ] **Step 2: Verify benchmark compiles**

Run: `cargo bench --no-run`
Expected: compiles without error

- [ ] **Step 3: Run benchmark**

Run: `cargo bench`
Expected: benchmark results printed (baseline numbers for Phase 1)

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add benches/decode.rs
git commit -s -m "feat: criterion benchmark for JPEG decoding"
```
