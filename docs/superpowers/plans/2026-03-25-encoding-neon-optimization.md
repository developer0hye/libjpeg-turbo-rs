# Encoding NEON Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire existing NEON encoder functions into the encoding pipeline via an `EncoderSimdRoutines` dispatch struct, and create encoding benchmarks to measure gains.

**Architecture:** Create a separate `EncoderSimdRoutines` dispatch struct (parallel to the decode `SimdRoutines`). Fuse FDCT+quantize into a single dispatch point to bridge the i32/i16 type mismatch. Add encoding benchmarks at both micro (per-stage) and macro (full-encode matrix) levels.

**Tech Stack:** Rust, NEON intrinsics (aarch64), Criterion benchmarking, libjpeg-turbo C (for baseline comparison)

**Spec:** `docs/superpowers/specs/2026-03-25-encoding-neon-optimization-design.md`

---

## File Structure

**New files:**
- `benches/encode.rs` — Criterion micro-benchmarks for encoder stages
- `examples/bench_encode_matrix.rs` — Full encode matrix benchmark (Rust)
- `examples/bench_c_encode_matrix.c` — Full encode matrix benchmark (C baseline)
- `experiments/encode.tsv` — Experiment tracking log

**Modified files:**
- `src/simd/mod.rs` — Add `EncoderSimdRoutines` struct + `detect_encoder()`
- `src/simd/scalar.rs` — Add scalar encoder wrappers
- `src/simd/aarch64/mod.rs` — Add `encoder_routines()` with NEON fused fdct_quantize
- `src/encode/pipeline.rs` — Replace direct scalar calls with dispatch through `EncoderSimdRoutines`
- `Cargo.toml` — Add `[[bench]] name = "encode"` target

**Out of scope (remain scalar-only per spec):**
- `compress_progressive`, `compress_progressive_custom` — progressive encoding
- `compress_arithmetic`, `compress_arithmetic_progressive` — arithmetic coding
- `compress_lossless`, `compress_lossless_extended`, `compress_lossless_arithmetic` — lossless (has its own inline `color::rgb_to_ycbcr_row` calls at lines ~1292 and ~1532 that bypass `convert_to_ycbcr`)

---

## Chunk 1: EncoderSimdRoutines Dispatch Layer

### Task 1: Add `EncoderSimdRoutines` struct, scalar wrappers, NEON dispatch, and correctness tests

All three files must be added together so the codebase compiles after the commit.

**Files:**
- Modify: `src/simd/mod.rs`
- Modify: `src/simd/scalar.rs`
- Modify: `src/simd/aarch64/mod.rs`

- [ ] **Step 1: Read all three existing files**

Read `src/simd/mod.rs`, `src/simd/scalar.rs`, `src/simd/aarch64/mod.rs`.

- [ ] **Step 2: Add `EncoderSimdRoutines` struct and `detect_encoder()` to `src/simd/mod.rs`**

Add below the existing `SimdRoutines` struct:

```rust
/// Function-pointer dispatch table for SIMD-accelerated encode operations.
pub struct EncoderSimdRoutines {
    /// RGB → YCbCr color conversion, one row.
    /// Only handles interleaved RGB (3 bytes/pixel).
    pub rgb_to_ycbcr_row: fn(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize),

    /// Combined FDCT (islow) + quantize + zigzag reorder for one 8x8 block.
    /// `quant` must be the pre-scaled divisor table (values x 8).
    /// Output is in zigzag scan order, ready for Huffman encoding.
    pub fdct_quantize: fn(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]),
}
```

Add `detect_encoder()` after `detect()`:

```rust
/// Detect available SIMD features and return the best encoder dispatch table.
pub fn detect_encoder() -> EncoderSimdRoutines {
    if std::env::var("JSIMD_FORCENONE").ok().as_deref() == Some("1") {
        return scalar::encoder_routines();
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return aarch64::encoder_routines();
    }

    // x86_64: no encoder SIMD yet, fall through to scalar
    // TODO: add x86_64 encoder SIMD (SSE2/AVX2 FDCT, color conversion)

    #[allow(unreachable_code)]
    scalar::encoder_routines()
}
```

- [ ] **Step 3: Add scalar encoder wrappers to `src/simd/scalar.rs`**

Add at the bottom:

```rust
use crate::encode::{color as enc_color, fdct, quant};
use crate::simd::EncoderSimdRoutines;

/// Return scalar encoder dispatch table.
pub fn encoder_routines() -> EncoderSimdRoutines {
    EncoderSimdRoutines {
        rgb_to_ycbcr_row: scalar_rgb_to_ycbcr_row_enc,
        fdct_quantize: scalar_fdct_quantize,
    }
}

/// Scalar RGB -> YCbCr row conversion (delegates to encode::color).
fn scalar_rgb_to_ycbcr_row_enc(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    enc_color::rgb_to_ycbcr_row(rgb, y, cb, cr, width);
}

/// Scalar fused FDCT (islow) + quantize + zigzag reorder.
///
/// Calls `fdct_islow` (output i32) then `quantize_block` (zigzag reorder included).
pub(crate) fn scalar_fdct_quantize(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]) {
    let mut dct_output = [0i32; 64];
    fdct::fdct_islow(input, &mut dct_output);
    quant::quantize_block(&dct_output, quant, output);
}
```

Note: `scalar_fdct_quantize` is `pub(crate)` so NEON correctness tests can reference it.

- [ ] **Step 4: Add NEON encoder dispatch to `src/simd/aarch64/mod.rs`**

Add at the bottom:

```rust
use crate::encode::tables::ZIGZAG_ORDER;
use crate::simd::EncoderSimdRoutines;

/// Return NEON-accelerated encoder routines.
pub fn encoder_routines() -> EncoderSimdRoutines {
    EncoderSimdRoutines {
        rgb_to_ycbcr_row: color_encode::neon_rgb_to_ycbcr_row,
        fdct_quantize: neon_fdct_quantize,
    }
}

/// NEON fused FDCT (islow) + quantize + zigzag reorder.
fn neon_fdct_quantize(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]) {
    let mut dct_output = [0i16; 64];
    fdct::neon_fdct(input, &mut dct_output);
    let mut natural = [0i16; 64];
    quantize::neon_quantize(&dct_output, quant, &mut natural);
    // Reorder from natural to zigzag scan order
    for zigzag_pos in 0..64 {
        output[zigzag_pos] = natural[ZIGZAG_ORDER[zigzag_pos]];
    }
}
```

- [ ] **Step 5: Verify it compiles and all existing tests pass**

Run: `cargo test`

Expected: Clean compilation and all tests pass (no behavioral change yet).

- [ ] **Step 6: Commit**

```bash
git add src/simd/mod.rs src/simd/scalar.rs src/simd/aarch64/mod.rs
git commit -s -m "feat(simd): add EncoderSimdRoutines dispatch layer with scalar and NEON backends"
```

---

### Task 2: Write correctness tests — NEON vs scalar produce identical output

**Files:**
- Modify: `src/simd/aarch64/mod.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Write tests for `fdct_quantize` equivalence**

Add at the bottom of `src/simd/aarch64/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    #[test]
    fn neon_fdct_quantize_matches_scalar() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = (i as i16 * 3) - 96;
        }
        let mut quant = [0u16; 64];
        for i in 0..64 {
            quant[i] = (16 + (i as u16) * 2) * 8;
        }

        let mut neon_output = [0i16; 64];
        let mut scalar_output = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar_dc_only() {
        let input = [50i16; 64];
        let quant = [128u16; 64];

        let mut neon_output = [0i16; 64];
        let mut scalar_output = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar_checkerboard() {
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = if (row + col) % 2 == 0 { 100 } else { -100 };
            }
        }
        let quant = [80u16; 64];

        let mut neon_output = [0i16; 64];
        let mut scalar_output = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar() {
        let width = 640;
        let rgb: Vec<u8> = (0..width * 3).map(|i| (i % 256) as u8).collect();

        let mut y_neon = vec![0u8; width];
        let mut cb_neon = vec![0u8; width];
        let mut cr_neon = vec![0u8; width];
        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];

        color_encode::neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);
        crate::encode::color::rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);

        assert_eq!(y_neon, y_scalar, "Y plane mismatch");
        assert_eq!(cb_neon, cb_scalar, "Cb plane mismatch");
        assert_eq!(cr_neon, cr_scalar, "Cr plane mismatch");
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar_edge_values() {
        // Test with extreme values: all 0s, all 255s, and pure R/G/B
        for (r, g, b) in [(0u8, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)] {
            let width = 16;
            let rgb: Vec<u8> = (0..width).flat_map(|_| [r, g, b]).collect();

            let mut y_neon = vec![0u8; width];
            let mut cb_neon = vec![0u8; width];
            let mut cr_neon = vec![0u8; width];
            let mut y_scalar = vec![0u8; width];
            let mut cb_scalar = vec![0u8; width];
            let mut cr_scalar = vec![0u8; width];

            color_encode::neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);
            crate::encode::color::rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);

            assert_eq!(y_neon, y_scalar, "Y mismatch for rgb=({r},{g},{b})");
            assert_eq!(cb_neon, cb_scalar, "Cb mismatch for rgb=({r},{g},{b})");
            assert_eq!(cr_neon, cr_scalar, "Cr mismatch for rgb=({r},{g},{b})");
        }
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p libjpeg-turbo-rs aarch64::tests`

Expected: All 5 tests pass. If any fail, fix the NEON implementation before proceeding.

- [ ] **Step 3: Commit**

```bash
git add src/simd/aarch64/mod.rs
git commit -s -m "test(simd): verify NEON encoder functions match scalar output"
```

---

## Chunk 2: Wire Dispatch into Encoder Pipeline

### Task 3: Refactor `encode_single_block` and `encode_downsampled_chroma_block` to use fused `fdct_quantize`

**Files:**
- Modify: `src/encode/pipeline.rs`

- [ ] **Step 1: Read `encode_single_block`**

Search for `fn encode_single_block` in `src/encode/pipeline.rs`. Read the function.

- [ ] **Step 2: Change signature and body of `encode_single_block`**

Replace the `fdct_fn` parameter:
```rust
    fdct_fn: fn(&[i16; 64], &mut [i32; 64]),
```
With:
```rust
    fdct_quantize: fn(&[i16; 64], &[u16; 64], &mut [i16; 64]),
```

Replace the FDCT + quantize body:
```rust
    let mut dct_output = [0i32; 64];
    fdct_fn(&block, &mut dct_output);

    let mut quantized = [0i16; 64];
    quant::quantize_block(&dct_output, quant_table, &mut quantized);
```
With:
```rust
    let mut quantized = [0i16; 64];
    fdct_quantize(&block, quant_table, &mut quantized);
```

- [ ] **Step 3: Apply identical changes to `encode_downsampled_chroma_block`**

Search for `fn encode_downsampled_chroma_block`. Apply the same parameter and body changes.

- [ ] **Step 4: Update `encode_color_mcu` signature and all its internal calls**

Search for `fn encode_color_mcu`. Change `fdct_fn` parameter to `fdct_quantize` with the new type. Update all internal calls to `encode_single_block` and `encode_downsampled_chroma_block` to pass `fdct_quantize`.

- [ ] **Step 5: Do NOT commit yet** — call sites need updating.

---

### Task 4: Update all baseline entry points to use `detect_encoder()`

**Files:**
- Modify: `src/encode/pipeline.rs`

The following baseline entry points must be updated (search for each by function name):

The baseline entry points fall into three groups by how they currently pass the FDCT function:

**Group A — uses `fdct::select_fdct(dct_method)` → local `fdct_fn` variable:**
| Function | Notes |
|---|---|
| `compress()` | Only function with `dct_method` parameter |

**Group B — passes `fdct::fdct_islow` as inline argument (no local variable):**
| Function | Notes |
|---|---|
| `compress_custom_huffman()` | Passes `fdct::fdct_islow` directly to `encode_single_block`/`encode_color_mcu` |
| `compress_custom_quant()` | Same inline pattern |
| `compress_with_restart()` | Same inline pattern |

**Group C — uses `let fdct_fn = fdct::fdct_islow` (hardcoded, no `dct_method`):**
| Function | Notes |
|---|---|
| `compress_optimized()` | Also has `gather_block`/`gather_downsampled_block` with inline FDCT+quantize |
| `compress_raw()` | Takes pre-converted YCbCr planes (no `convert_to_ycbcr` call) |
| `compress_custom_sampling()` | Hardcodes `fdct_islow` |

- [ ] **Step 1: Update `compress()` (Group A — has `dct_method`)**

At the top of the function body (after validation), add:
```rust
    let enc_simd = crate::simd::detect_encoder();
```

Replace:
```rust
    let fdct_fn: fn(&[i16; 64], &mut [i32; 64]) = fdct::select_fdct(dct_method);
```
With:
```rust
    // NEON fused FDCT+quantize for IsLow (the common case and only NEON-supported variant).
    // IsFast/Float: keep existing scalar fdct + quantize path.
    // WARNING: IsFast/Float currently fall back to scalar_fdct_quantize which uses fdct_islow.
    // This matches the public API (which hardcodes IsLow), so no user-visible regression.
    // TODO: add scalar_fdct_ifast_quantize if IsFast support is needed.
    let fdct_quantize_fn: fn(&[i16; 64], &[u16; 64], &mut [i16; 64]) = if dct_method == DctMethod::IsLow {
        enc_simd.fdct_quantize
    } else {
        crate::simd::scalar::scalar_fdct_quantize
    };
```

Update all calls passing `fdct_fn` to pass `fdct_quantize_fn` instead.

- [ ] **Step 2: Update Group B functions (`compress_custom_huffman`, `compress_custom_quant`, `compress_with_restart`)**

These do NOT have a `let fdct_fn` variable — they pass `fdct::fdct_islow` as an inline argument to `encode_single_block` and `encode_color_mcu`. For each function:

1. Add `let enc_simd = crate::simd::detect_encoder();` at the top
2. Replace every inline `fdct::fdct_islow` argument with `enc_simd.fdct_quantize`

Search for `, fdct::fdct_islow,` or `fdct::fdct_islow)` within each function body.

- [ ] **Step 3: Update Group C functions (`compress_raw`, `compress_custom_sampling`)**

These have `let fdct_fn = fdct::fdct_islow` (a local variable, but no `dct_method`). For each:

1. Add `let enc_simd = crate::simd::detect_encoder();` at the top
2. Replace `let fdct_fn: fn(...) = fdct::fdct_islow;` with `let fdct_quantize_fn = enc_simd.fdct_quantize;`
3. Update all uses of `fdct_fn` to `fdct_quantize_fn`

- [ ] **Step 4: Update `compress_optimized`**

This function has two internal helpers that inline FDCT+quantize:

Search for `fn gather_block` and `fn gather_downsampled_block` inside `compress_optimized`. These have inline calls to `fdct::fdct_islow` + `quant::quantize_block`. Replace them with the fused `fdct_quantize` function pointer, passed as a parameter from the outer function that calls `detect_encoder()`.

- [ ] **Step 5: Leave progressive/arithmetic/lossless entry points scalar-only**

These functions use inline `fdct::fdct_islow` calls or pass it directly. Replace their FDCT + quantize pattern with `crate::simd::scalar::scalar_fdct_quantize` to unify the function signature, but do NOT use `detect_encoder()`. This keeps them scalar-only per spec.

Affected (search for the internal function that both progressive variants delegate to):
- `compress_progressive_with_scans` (the internal function called by both `compress_progressive` and `compress_progressive_custom`)
- `compress_arithmetic`
- `compress_arithmetic_progressive`

- [ ] **Step 6: Verify it compiles**

Run: `cargo check`

Expected: Clean compilation.

- [ ] **Step 7: Run all tests**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/encode/pipeline.rs
git commit -s -m "feat(encode): wire fused fdct_quantize dispatch into encoder pipeline"
```

---

### Task 5: Wire color conversion dispatch into `convert_to_ycbcr`

**Files:**
- Modify: `src/encode/pipeline.rs`

- [ ] **Step 1: Read `convert_to_ycbcr` function**

Search for `fn convert_to_ycbcr` in pipeline.rs.

- [ ] **Step 2: Add `rgb_to_ycbcr_row` parameter**

Change signature from:
```rust
fn convert_to_ycbcr(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)>
```
To:
```rust
fn convert_to_ycbcr(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    rgb_to_ycbcr_row: fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize),
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)>
```

- [ ] **Step 3: Replace `color::rgb_to_ycbcr_row` in the `PixelFormat::Rgb` arm only**

In the `PixelFormat::Rgb` match arm, replace `color::rgb_to_ycbcr_row(...)` with `rgb_to_ycbcr_row(...)`.

All other arms (Rgba, Bgr, Bgra, generic) keep calling their existing scalar functions unchanged.

- [ ] **Step 4: Update ALL `convert_to_ycbcr` call sites**

There are 9 call sites. Each must be updated. Search for `convert_to_ycbcr(` to find them all.

Note: `compress_raw()` takes pre-converted YCbCr planes and does NOT call `convert_to_ycbcr`.

**Baseline entry points — use `enc_simd.rgb_to_ycbcr_row`:**
- `compress()` (~line 80)
- `compress_custom_huffman()` (~line 325)
- `compress_custom_quant()` (~line 515)
- `compress_with_restart()` (~line 720)
- `compress_optimized()` (~line 3659)
- `compress_custom_sampling()` (~line 4587)

These already have `enc_simd` from Task 4. Pass `enc_simd.rgb_to_ycbcr_row` as the last argument.

**Progressive/arithmetic — use scalar:**
- `compress_progressive_with_scans()` (~line 1719) — internal function used by both `compress_progressive` and `compress_progressive_custom`
- `compress_arithmetic()` (~line 2031)
- `compress_arithmetic_progressive()` (~line 2315)

These do not have `enc_simd`. Pass `crate::encode::color::rgb_to_ycbcr_row` directly.

- [ ] **Step 5: Run all tests**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/encode/pipeline.rs
git commit -s -m "feat(encode): wire color conversion NEON dispatch into convert_to_ycbcr"
```

---

## Chunk 3: Encoding Benchmarks

### Task 6: Add encode bench target to `Cargo.toml`

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add bench target**

Add after existing `[[bench]]` entries:
```toml
[[bench]]
name = "encode"
harness = false
```

- [ ] **Step 2: Commit**

```bash
git add Cargo.toml
git commit -s -m "chore: add encode bench target to Cargo.toml"
```

---

### Task 7: Create Criterion micro-benchmarks (`benches/encode.rs`)

**Files:**
- Create: `benches/encode.rs`

- [ ] **Step 1: Create the benchmark file**

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libjpeg_turbo_rs::simd;

fn bench_fdct_quantize_8x8(c: &mut Criterion) {
    let enc = simd::detect_encoder();

    let mut input = [0i16; 64];
    for i in 0..64 {
        input[i] = (i as i16 * 3) - 96;
    }
    let quant = [128u16; 64]; // pre-scaled (16 * 8)
    let mut output = [0i16; 64];

    c.bench_function("fdct_quantize_8x8", |b| {
        b.iter(|| {
            (enc.fdct_quantize)(black_box(&input), black_box(&quant), &mut output);
            black_box(&output);
        })
    });
}

fn bench_rgb_to_ycbcr_row(c: &mut Criterion) {
    let enc = simd::detect_encoder();

    for width in [320, 640, 1920] {
        let rgb: Vec<u8> = (0..width * 3).map(|i| (i % 256) as u8).collect();
        let mut y = vec![0u8; width];
        let mut cb = vec![0u8; width];
        let mut cr = vec![0u8; width];

        c.bench_function(&format!("rgb_to_ycbcr_row_{width}"), |b| {
            b.iter(|| {
                (enc.rgb_to_ycbcr_row)(black_box(&rgb), &mut y, &mut cb, &mut cr, width);
                black_box((&y, &cb, &cr));
            })
        });
    }
}

fn bench_full_encode(c: &mut Criterion) {
    use libjpeg_turbo_rs::common::types::{PixelFormat, Subsampling};

    struct EncodeCase {
        name: &'static str,
        fixture: &'static str,
        subsampling: Subsampling,
    }

    let cases = [
        EncodeCase { name: "encode_320x240_420", fixture: "tests/fixtures/photo_320x240_420.jpg", subsampling: Subsampling::S420 },
        EncodeCase { name: "encode_320x240_422", fixture: "tests/fixtures/photo_320x240_422.jpg", subsampling: Subsampling::S422 },
        EncodeCase { name: "encode_320x240_444", fixture: "tests/fixtures/photo_320x240_444.jpg", subsampling: Subsampling::S444 },
        EncodeCase { name: "encode_640x480_422", fixture: "tests/fixtures/photo_640x480_422.jpg", subsampling: Subsampling::S422 },
        EncodeCase { name: "encode_640x480_444", fixture: "tests/fixtures/photo_640x480_444.jpg", subsampling: Subsampling::S444 },
        EncodeCase { name: "encode_1920x1080_420", fixture: "tests/fixtures/photo_1920x1080_420.jpg", subsampling: Subsampling::S420 },
        EncodeCase { name: "encode_1920x1080_422", fixture: "tests/fixtures/photo_1920x1080_422.jpg", subsampling: Subsampling::S422 },
        EncodeCase { name: "encode_1920x1080_444", fixture: "tests/fixtures/photo_1920x1080_444.jpg", subsampling: Subsampling::S444 },
    ];

    for case in &cases {
        let jpeg_data = match std::fs::read(case.fixture) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();

        c.bench_function(case.name, |b| {
            b.iter(|| {
                let result = libjpeg_turbo_rs::compress(
                    black_box(&image.data),
                    image.width,
                    image.height,
                    PixelFormat::Rgb,
                    75,
                    case.subsampling,
                ).unwrap();
                black_box(&result);
            })
        });
    }
}

criterion_group!(
    benches,
    bench_fdct_quantize_8x8,
    bench_rgb_to_ycbcr_row,
    bench_full_encode,
);
criterion_main!(benches);
```

- [ ] **Step 2: Verify benchmarks compile and run**

Run: `cargo bench --bench encode -- --test`

Expected: Benchmarks compile and run in test mode.

- [ ] **Step 3: Commit**

```bash
git add benches/encode.rs
git commit -s -m "bench: add Criterion micro-benchmarks for encoder stages"
```

---

### Task 8: Create full encode matrix benchmark (`examples/bench_encode_matrix.rs`)

**Files:**
- Create: `examples/bench_encode_matrix.rs`

- [ ] **Step 1: Create the benchmark file**

```rust
/// Standalone encoding benchmark matrix.
/// Usage: cargo run --release --example bench_encode_matrix
fn main() {
    use libjpeg_turbo_rs::common::types::{PixelFormat, Subsampling};

    struct EncodeCase {
        fixture: &'static str,
        subsampling: Subsampling,
        iters: u32,
    }

    let cases: Vec<EncodeCase> = vec![
        // Resolution scaling (4:2:0)
        EncodeCase { fixture: "tests/fixtures/photo_64x64_420.jpg", subsampling: Subsampling::S420, iters: 20000 },
        EncodeCase { fixture: "tests/fixtures/photo_320x240_420.jpg", subsampling: Subsampling::S420, iters: 5000 },
        EncodeCase { fixture: "tests/fixtures/photo_640x480_422.jpg", subsampling: Subsampling::S420, iters: 5000 },
        EncodeCase { fixture: "tests/fixtures/photo_1280x720_420.jpg", subsampling: Subsampling::S420, iters: 2000 },
        EncodeCase { fixture: "tests/fixtures/photo_1920x1080_420.jpg", subsampling: Subsampling::S420, iters: 500 },
        // Subsampling modes (320x240)
        EncodeCase { fixture: "tests/fixtures/photo_320x240_444.jpg", subsampling: Subsampling::S444, iters: 5000 },
        EncodeCase { fixture: "tests/fixtures/photo_320x240_422.jpg", subsampling: Subsampling::S422, iters: 5000 },
        // Subsampling modes (640x480)
        EncodeCase { fixture: "tests/fixtures/photo_640x480_444.jpg", subsampling: Subsampling::S444, iters: 5000 },
        EncodeCase { fixture: "tests/fixtures/photo_640x480_422.jpg", subsampling: Subsampling::S422, iters: 5000 },
        // Subsampling modes (1920x1080)
        EncodeCase { fixture: "tests/fixtures/photo_1920x1080_444.jpg", subsampling: Subsampling::S444, iters: 500 },
        EncodeCase { fixture: "tests/fixtures/photo_1920x1080_422.jpg", subsampling: Subsampling::S422, iters: 500 },
    ];

    println!(
        "{:<50} {:>10} {:>12} {:>8}",
        "Case", "Size", "Time", "Iters"
    );
    println!("{}", "-".repeat(85));

    for case in &cases {
        let jpeg_data = match std::fs::read(case.fixture) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: {} (not found)", case.fixture);
                continue;
            }
        };

        let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        let pixels = &image.data;
        let width = image.width;
        let height = image.height;

        // Warmup
        for _ in 0..100 {
            let _ = libjpeg_turbo_rs::compress(
                pixels, width, height, PixelFormat::Rgb, 75, case.subsampling,
            ).unwrap();
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..case.iters {
            let result = libjpeg_turbo_rs::compress(
                pixels, width, height, PixelFormat::Rgb, 75, case.subsampling,
            ).unwrap();
            std::hint::black_box(&result);
        }
        let elapsed = start.elapsed();
        let us: f64 = elapsed.as_nanos() as f64 / case.iters as f64 / 1000.0;

        let sub_str = match case.subsampling {
            Subsampling::S420 => "420",
            Subsampling::S422 => "422",
            Subsampling::S444 => "444",
            _ => "???",
        };

        println!(
            "encode_{:>4}x{:<4}_{:<3}                                {:>4}x{:<4} {:>10.1} us  ({} iters)",
            width, height, sub_str, width, height, us, case.iters
        );
    }
}
```

- [ ] **Step 2: Build and smoke test**

Run: `cargo build --release --example bench_encode_matrix && cargo run --release --example bench_encode_matrix 2>&1 | head -15`

- [ ] **Step 3: Commit**

```bash
git add examples/bench_encode_matrix.rs
git commit -s -m "bench: add full encode matrix benchmark"
```

---

### Task 9: Create experiment tracking log and C baseline benchmark

**Files:**
- Create: `experiments/encode.tsv`
- Create: `examples/bench_c_encode_matrix.c`

- [ ] **Step 1: Create `experiments/encode.tsv` with header**

```
date	target	description	result	status
```

- [ ] **Step 2: Create C baseline benchmark**

Create `examples/bench_c_encode_matrix.c` — a C file that links against system libjpeg-turbo and benchmarks encoding at the same resolutions/subsampling as the Rust matrix. Use `mach_absolute_time()` for timing on macOS (not `clock_gettime` which has portability issues). Follow the pattern of any existing C baseline in the project.

- [ ] **Step 3: Commit**

```bash
git add experiments/encode.tsv examples/bench_c_encode_matrix.c
git commit -s -m "bench: add encoding experiment log and C baseline benchmark"
```

---

## Chunk 4: Baseline Measurement and Validation

### Task 10: Collect baseline numbers and validate correctness

- [ ] **Step 1: Run encode matrix with NEON disabled (scalar baseline)**

```bash
JSIMD_FORCENONE=1 cargo run --release --example bench_encode_matrix 2>&1 | tee /tmp/encode_scalar_baseline.txt
```

- [ ] **Step 2: Run encode matrix with NEON enabled**

```bash
cargo run --release --example bench_encode_matrix 2>&1 | tee /tmp/encode_neon_baseline.txt
```

- [ ] **Step 3: Record both baselines in `experiments/encode.tsv`**

- [ ] **Step 4: Run full test suite with NEON enabled + disabled**

```bash
cargo test && JSIMD_FORCENONE=1 cargo test
```

Expected: All tests pass in both modes.

- [ ] **Step 5: Commit**

```bash
git add experiments/encode.tsv
git commit -s -m "perf: record encoding baseline measurements (scalar vs NEON)"
```
