# Encoding NEON Optimization Design

## Context

The libjpeg-turbo-rs encoder has four NEON-accelerated functions already implemented in `src/simd/aarch64/` but **none are wired into the encoder pipeline**. The pipeline currently calls scalar-only functions directly. The decode side already has a `SimdRoutines` dispatch struct; encoding needs its own.

### Existing NEON Encoder Functions (Unused)

| Stage | File | Function |
|---|---|---|
| RGB→YCbCr | `simd/aarch64/color_encode.rs` | `neon_rgb_to_ycbcr_row()` |
| Forward DCT | `simd/aarch64/fdct.rs` | `neon_fdct()` |
| Quantization | `simd/aarch64/quantize.rs` | `neon_quantize()` |
| Downsample H2V1 | `simd/aarch64/downsample.rs` | `neon_downsample_h2v1()` |
| Downsample H2V2 | `simd/aarch64/downsample.rs` | `neon_downsample_h2v2()` |

### Key Constraints

**FDCT output type mismatch:**
- Scalar FDCT: `fn(&[i16; 64], &mut [i32; 64])` → scalar quantize takes `&[i32; 64]`
- NEON FDCT: `fn(&[i16; 64], &mut [i16; 64])` → NEON quantize takes `&[i16; 64]`

FDCT and quantize must be dispatched together.

**Zigzag reordering mismatch:**
- Scalar `quantize_block` performs zigzag reordering (output in zigzag order)
- NEON `neon_quantize` outputs in natural order (no zigzag)

The fused `fdct_quantize` must produce **zigzag-ordered output** since Huffman encoding expects it. The NEON wrapper must add zigzag reordering after `neon_quantize`, or `neon_quantize` itself must be updated to include it.

**Quantization table scaling:**
- The `quant` parameter passed to `fdct_quantize` must be the **pre-scaled** quantization table (values × 8), matching how `scale_quant_for_fdct()` prepares them in the pipeline. Both scalar and NEON FDCT output values scaled by 8.

## Approach

Benchmark first, then wire. Profile the scalar encoder to establish baseline, then integrate NEON functions incrementally (heaviest hitters first).

## 1. `EncoderSimdRoutines` Dispatch Struct

Create a separate dispatch struct for encoding (not extending the decode `SimdRoutines`):

```rust
// src/simd/mod.rs

pub struct EncoderSimdRoutines {
    /// RGB → YCbCr color conversion, one row.
    /// Only handles interleaved RGB (3 bytes/pixel).
    /// Other pixel formats (RGBA, BGR, BGRA) fall back to scalar in the pipeline.
    pub rgb_to_ycbcr_row: fn(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize),

    /// Combined FDCT (islow) + quantize + zigzag reorder for one 8x8 block.
    /// Fusing avoids the i32-vs-i16 intermediate type mismatch.
    /// `quant` must be the pre-scaled table (values × 8).
    /// Output is in zigzag scan order, ready for Huffman encoding.
    pub fdct_quantize: fn(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]),

    /// Chroma downsample H2V1 (one row).
    pub downsample_h2v1: fn(input: &[u8], in_width: usize, output: &mut [u8]),

    /// Chroma downsample H2V2 (two rows → one row).
    pub downsample_h2v2: fn(row0: &[u8], row1: &[u8], in_width: usize, output: &mut [u8]),
}

pub fn detect_encoder() -> EncoderSimdRoutines { ... }
```

### Scalar Fallback Wrapper

```rust
// src/simd/scalar.rs

pub fn scalar_fdct_quantize(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]) {
    let mut dct = [0i32; 64];
    fdct_islow(input, &mut dct);
    quantize_block(&dct, quant, output); // already produces zigzag output
}
```

### NEON Wrapper

```rust
// src/simd/aarch64/

pub fn neon_fdct_quantize(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]) {
    let mut dct = [0i16; 64];
    neon_fdct(input, &mut dct);
    let mut natural = [0i16; 64];
    neon_quantize(&dct, quant, &mut natural);
    // Reorder from natural to zigzag
    for zigzag_pos in 0..64 {
        output[zigzag_pos] = natural[ZIGZAG_ORDER[zigzag_pos]];
    }
}
```

Note: The zigzag reorder loop is a candidate for future NEON optimization (e.g., `vtbl` lookup). For initial integration, a scalar loop suffices since it operates on only 64 elements.

### DctMethod Handling

The fused `fdct_quantize` only covers `DctMethod::IsLow` (the NEON FDCT is an islow port). When `DctMethod::IsFast` or `DctMethod::Float` is selected, the pipeline falls back to the scalar path (existing `fdct_fn` + `quantize_block`). This can be handled by checking `dct_method` before dispatch.

### Dispatch

- `detect_encoder()` follows the same pattern as `detect()`: check `JSIMD_FORCENONE`, then select NEON on aarch64, scalar elsewhere.

## 2. Encoding Benchmark Harness

### 2a. Micro-benchmarks (`benches/encode.rs`)

Criterion benchmarks for individual stages:
- `fdct_8x8` — scalar vs NEON (toggle via `JSIMD_FORCENONE`)
- `rgb_to_ycbcr_row_640` — single row color conversion
- `fdct_quantize_8x8` — fused FDCT+quantize
- `downsample_h2v1_320` / `downsample_h2v2_320`

### 2b. Full Encode Matrix (`examples/bench_encode_matrix.rs`)

End-to-end compression benchmark:
- **Resolutions:** 64x64, 320x240, 640x480, 1280x720, 1920x1080
- **Subsampling:** 4:4:4, 4:2:2, 4:2:0
- **Input:** Decode test fixture JPEGs to raw RGB, then benchmark encode from the decoded pixels

Output format matches existing `bench_matrix.rs`.

### 2c. C Baseline (`examples/bench_c_encode_matrix.c`)

New C file linking against system libjpeg-turbo for encoding times on the same matrix. Must be created from scratch (no existing C encode benchmark).

### Test Fixtures

Reuse existing test fixture JPEGs decoded to raw RGB — no new fixture files needed. Verify all resolution × subsampling combinations have corresponding fixture files before benchmarking.

## 3. Pipeline Integration

### Current flow in `pipeline.rs`:

```rust
fdct_fn(&block, &mut dct_output);        // i32 output
quant::quantize_block(&dct_output, ...);  // i32 input → zigzag output
```

### New flow:

```rust
let enc_simd = simd::detect_encoder();
(enc_simd.fdct_quantize)(&block, quant_table, &mut quantized);  // fused, zigzag output
(enc_simd.rgb_to_ycbcr_row)(rgb, y, cb, cr, width);             // dispatched (RGB only)
```

### Changes in `pipeline.rs`:

1. Call `simd::detect_encoder()` at the start of each encode entry point
2. Replace `fdct_fn` + `quantize_block` call sites with `enc_simd.fdct_quantize`
3. Replace `color::rgb_to_ycbcr_row()` calls with `enc_simd.rgb_to_ycbcr_row` (RGB pixel format only; RGBA/BGR/BGRA remain scalar)
4. Downsample wiring deferred to Phase 3 — requires bridging row-level NEON functions with block-level pipeline call sites

### Scope: Baseline Sequential Encoding Only

NEON dispatch targets `compress()` (baseline sequential). Progressive, arithmetic, and lossless encoding paths (`compress_progressive`, `compress_arithmetic`, etc.) remain scalar-only for now.

## 4. Phased Rollout

- **Phase 1:** Create benchmark harness, establish scalar baseline, collect C baseline
- **Phase 2:** Wire FDCT+quantize NEON and color conversion NEON, measure gains
- **Phase 3:** Wire downsample NEON (requires refactoring block-level → row-level downsample), measure incremental gains
- **Phase 4:** Profile for further optimization (e.g., NEON zigzag reorder in fused function, Huffman coefficient prep, RGBA color conversion)

## 5. Experiment Tracking

New file: `experiments/encode.tsv`

Experiment sequence:
1. Baseline: scalar full-encode matrix (all resolutions × subsampling)
2. C baseline: libjpeg-turbo C encode for same matrix
3. Wire FDCT+quantize NEON → measure delta
4. Wire color conversion NEON → measure delta
5. Wire downsample NEON → measure delta

Each experiment follows keep/discard protocol. Profile with `samply record` before each optimization.
