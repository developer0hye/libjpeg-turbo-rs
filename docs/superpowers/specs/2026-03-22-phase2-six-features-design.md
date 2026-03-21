# Phase 2: Six Features Design Spec

## Context

libjpeg-turbo-rs has completed Phase 1 (baseline+progressive decoder, NEON SIMD, all pixel formats, CMYK/YCCK, scaled IDCT, baseline encoder, ICC/EXIF, error recovery). This spec covers the next 6 features toward full libjpeg-turbo 3.1.x parity, ordered by practical value.

**Reference:** `docs/superpowers/specs/2026-03-21-libjpeg-turbo-rs-design.md` (master design spec)

---

## Feature 1: Partial Decompression (Crop/Skip)

**Goal:** Decode only a region of interest, skipping unnecessary scanlines and columns.

### API

**Low-level (libjpeg style):**
```rust
impl StreamingDecoder {
    /// Set horizontal crop region. xoffset auto-aligns to iMCU boundary.
    /// Must be called after header parse, before first read_scanlines().
    fn crop_scanline(&mut self, xoffset: &mut usize, width: &mut usize) -> Result<()>;

    /// Skip num_lines scanlines without decoding.
    /// Returns actual number of lines skipped.
    fn skip_scanlines(&mut self, num_lines: usize) -> Result<usize>;

    /// Read up to max_lines decoded scanlines into buf.
    fn read_scanlines(&mut self, buf: &mut [u8], max_lines: usize) -> Result<usize>;
}
```

**High-level (TurboJPEG style):**
```rust
pub fn decompress_cropped(data: &[u8], region: CropRegion) -> Result<Image>;

pub struct CropRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}
```

### Architecture

- `crop_scanline()`: aligns xoffset down to iMCU column boundary (`max_h * block_size`), adjusts component-level `first_MCU_col` / `last_MCU_col`, recalculates `downsampled_width` per component
- `skip_scanlines()`: dual path — progressive (coefficient buffer already exists, just advance pointer) vs baseline (decode-and-discard with noop color converter)
- `read_scanlines()`: decode one iMCU row at a time, output only the cropped region
- Scale + crop combinable: crop applies after scale dimension calculation

### Files

- **Modify:** `src/decode/pipeline.rs` (add scanline-based decode), `src/api/streaming.rs`, `src/api/high_level.rs`, `src/common/types.rs` (CropRegion)
- **Estimated:** ~600 lines

---

## Feature 2: Huffman Optimization (2-Pass Encoding)

**Goal:** Generate optimal Huffman tables from actual symbol frequencies for 5-10% better compression.

### API

```rust
impl Compressor {
    fn optimize_huffman(&mut self, enable: bool) -> &mut Self;
}
```

### Architecture

**Pass 1 — Symbol Gathering:**
- `encode_mcu_gather()`: identical MCU traversal as normal encoding but increments frequency counters instead of emitting bits
- 4 frequency arrays (DC_lum, DC_chr, AC_lum, AC_chr), 257 entries each
- Entry 256 = pseudo-symbol with count 1 (prevents all-ones code)

**Table Generation — Annex K.2:**
- `gen_optimal_table(freq) -> (bits[17], huffval[256])`
- Canonical Huffman tree: merge two smallest, increment codesizes
- Code length limiting: if max > 16, redistribute (JPEG spec constraint)
- Assign codes in canonical order

**Pass 2 — Normal Encoding:**
- Use generated tables instead of standard Annex K tables
- Write custom DHT markers

### Files

- **Modify:** `src/encode/huffman_encode.rs` (gather mode), `src/encode/pipeline.rs` (2-pass loop)
- **Create:** `src/encode/huff_opt.rs` (tree generation)
- **Estimated:** ~400 lines

---

## Feature 3: Arithmetic Coding (Encode + Decode)

**Goal:** Full ITU-T T.81 arithmetic entropy coding, both directions.

### Architecture

**Core State Machine:**
- Registers: C (32-bit interval base), A (interval size), ct (bit counter)
- Probability table: `ARITAB[114]` — packed Qe/Next_LPS/Next_MPS/Switch per state
- Statistics bins: DC 64 per table, AC 256 per table. Context-adaptive.
- Binary decision: compare C against (A - Qe), update state, renormalize

**Decoder (`ArithDecoder`):**
- `arith_decode(stat: &mut u8) -> bool` — single binary decision
- 4 MCU decoders: DC_first, DC_refine, AC_first, AC_refine (progressive)
- Renormalization: left-shift A and C, read input bytes with 0xFF stuffing

**Encoder (`ArithEncoder`):**
- `arith_encode(stat: &mut u8, val: bool)` — mirror of decoder
- Pacman termination: emit minimal trailing bytes
- Carry propagation: track stacked 0xFF bytes

**SOF Markers:** SOF9 (arithmetic sequential), SOF10 (arithmetic progressive)
**DAC Marker:** Conditioning parameters (arith_dc_L, arith_dc_U, arith_ac_K)

### Files

- **Create:** `src/decode/arithmetic.rs`, `src/encode/arithmetic.rs`, `src/common/arith_tables.rs`
- **Modify:** `src/decode/marker.rs` (SOF9/SOF10/DAC), `src/decode/entropy.rs` (dispatch), `src/encode/pipeline.rs`
- **Estimated:** ~2200 lines

---

## Feature 4: Progressive Encoding

**Goal:** Multi-scan progressive JPEG encoding with standard and custom scan scripts.

### API

```rust
impl Compressor {
    fn progressive(&mut self, enable: bool) -> &mut Self;
    fn scan_script(&mut self, script: Vec<ScanInfo>) -> &mut Self;
}

pub fn simple_progression(num_components: usize) -> Vec<ScanInfo>;
```

### Architecture

**Scan Script Generation:**
- `simple_progression()`: ~10 scans for 3-component YCbCr
  - Scan 1: DC first, all components, Al=1
  - Scan 2-4: AC first, per-component, various spectral bands
  - Scan 5+: DC/AC refinement scans

**Scan Validation:**
- DC: Ss=Se=0, up to 4 interleaved components
- AC: Ss>0 Se>0, exactly 1 component
- Successive approximation: Ah must equal previous scan's Al for same band
- Track `last_bitpos[component][coefficient]` for completeness check

**4 MCU Encoders:**
- `encode_mcu_DC_first()` — DC differential + left shift by Al
- `encode_mcu_DC_refine()` — single bit per DC coefficient
- `encode_mcu_AC_first()` — run-length + EOB run encoding
- `encode_mcu_AC_refine()` — correction bit buffer (MAX_CORR_BITS=1000)

**Pipeline:**
- Full image → FDCT → quantize → coefficient buffer (all components)
- For each scan: iterate coefficient buffer → entropy encode → emit to output

### Files

- **Create:** `src/encode/progressive.rs` (4 MCU encoders + scan script)
- **Modify:** `src/encode/pipeline.rs` (multi-scan loop, coefficient buffer)
- **Estimated:** ~800 lines

---

## Feature 5: Lossless Transforms

**Goal:** jpegtran-compatible DCT-domain spatial transforms without decode/re-encode.

### API

```rust
pub fn transform(jpeg_data: &[u8], info: &TransformInfo) -> Result<Vec<u8>>;

pub struct TransformInfo {
    pub transform: TransformOp,
    pub perfect: bool,
    pub trim: bool,
    pub force_grayscale: bool,
    pub crop: Option<CropRegion>,
    pub progressive: bool,
    pub arithmetic: bool,
    pub optimize: bool,
    pub drop: Option<DropInfo>,
    pub filter: Option<Box<dyn Fn(&mut [i16; 64], usize, usize)>>,
}

pub enum TransformOp {
    None, HFlip, VFlip, Transpose, Transverse, Rot90, Rot180, Rot270,
}
```

### Architecture

**3-Phase Pipeline:**
1. `request_workspace()` — compute output dimensions, allocate coefficient arrays
2. `adjust_parameters()` — transpose sampling factors, handle grayscale, fix EXIF
3. `execute_transform()` — dispatch to 8 spatial functions

**DCT Block Manipulation:**
- HFLIP: odd-column coefficients negated, blocks swapped L↔R
- VFLIP: odd-row coefficients negated, blocks swapped T↔B
- TRANSPOSE: `dst[j*8+i] = src[i*8+j]` within block + block coordinate swap
- ROT90/180/270, TRANSVERSE: compositions of above

**Partial MCU Handling:**
- PERFECT: error if image size not MCU-aligned
- TRIM: drop partial edge MCUs (output slightly smaller)
- Default: leave edge blocks untouched

**DROP:** Read coefficients from second JPEG, insert at iMCU-aligned position

### Files

- **Create:** `src/transform/mod.rs`, `src/transform/spatial.rs`, `src/transform/crop.rs`, `src/transform/filter.rs`, `src/transform/pipeline.rs`
- **Modify:** `src/lib.rs`
- **Estimated:** ~1200 lines

---

## Feature 6: Lossless JPEG Decoding

**Goal:** Decode ITU-T T.81 lossless JPEG (SOF3) with full 2-16 bit precision.

### Architecture

**Completely separate decode path** — no DCT, no quantization.

**Pipeline:** Entropy decode → difference samples → undifference (prediction) → scale (point transform) → output

**7 Predictors:**

| PSV | Formula | Description |
|-----|---------|-------------|
| 1 | Ra | Left |
| 2 | Rb | Above |
| 3 | Rc | Upper-left diagonal |
| 4 | Ra+Rb-Rc | Plane predictor |
| 5 | Ra+(Rb-Rc)/2 | Weighted horizontal |
| 6 | Rb+(Ra-Rc)/2 | Weighted vertical |
| 7 | (Ra+Rb)/2 | Simple average |

**Special Cases:**
- Row 0, sample 0: `initial = 1 << (precision - point_transform - 1)`
- Row 0, remaining: forced predictor 1 (horizontal)
- Other rows, sample 0: forced predictor 2 (vertical)

**Reconstruction:** `sample = (difference + prediction) & ((1 << precision) - 1)`

**Multi-Precision:**
- Internal: `u16` for all precisions (2-16 bit)
- Point transform Al: right-shift on decode, left-shift on output
- Output: `Image` with `data: Vec<u8>` for 8-bit, new `data_u16: Option<Vec<u16>>` for >8-bit

### Files

- **Create:** `src/decode/lossless.rs`, `src/decode/lossless_huffman.rs`
- **Modify:** `src/decode/marker.rs` (SOF3), `src/decode/pipeline.rs` (lossless branch)
- **Estimated:** ~600 lines

---

## Implementation Order

```
Feature 1 (Crop/Skip)          ← standalone, highest practical value
    │
Feature 2 (Huffman Opt)        ← encoder improvement, no deps
    │
Feature 3 (Arithmetic)         ← new entropy layer, enc+dec
    │
Feature 4 (Progressive Enc)    ← builds on encoder + arithmetic
    │
Feature 5 (Lossless Transforms)← needs coefficient read + encoder
    │
Feature 6 (Lossless JPEG Dec)  ← standalone decode path
```

Features 1-2 are independent. Feature 4 benefits from Feature 3 (arithmetic progressive). Feature 5 needs both decoder (coefficient read) and encoder (re-emit).

## Total Estimated Scope

| Feature | Lines | Difficulty | Branch |
|---------|-------|------------|--------|
| 1. Crop/Skip | ~600 | Medium | `feat/crop-skip` |
| 2. Huffman Opt | ~400 | Medium-High | `feat/huffman-opt` |
| 3. Arithmetic | ~2200 | **High** | `feat/arithmetic-coding` |
| 4. Progressive Enc | ~800 | Medium-High | `feat/progressive-enc` |
| 5. Lossless Transforms | ~1200 | Medium | `feat/lossless-transform` |
| 6. Lossless JPEG Dec | ~600 | Medium | `feat/lossless-jpeg` |
| **Total** | **~5800** | | |
