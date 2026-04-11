# WASM SIMD128 Baseline Benchmark

**Date**: 2026-04-11
**Environment**: Chromium (Playwright), macOS, Apple Silicon (WASM JIT → NEON)
**Settings**: 50 iterations, encode quality 90, synthetic test images
**WASM build**: wasm-pack --release -O4

## Decode: WASM vs Browser Native (createImageBitmap)

| Resolution | WASM (ms) | Native (ms) | Ratio | Throughput |
|-----------|-----------|-------------|-------|------------|
| 256x256 | 0.30 | 0.40 | 0.75x (faster) | 218.5 MP/s |
| 1024x768 | 2.40 | 2.60 | 0.92x (faster) | 327.7 MP/s |
| 1920x1080 | 5.50 | 5.70 | 0.96x (~same) | 377.0 MP/s |
| 2560x1440 | 9.20 | 10.00 | 0.92x (faster) | 400.7 MP/s |
| 3840x2160 | 18.30 | 21.80 | 0.84x (faster) | 453.2 MP/s |
| 7680x4320 | 63.90 | 83.90 | 0.76x (faster) | 519.2 MP/s |

**Decode summary**: WASM decode is **faster** than native across all resolutions (0.75x-0.96x ratio). Performance advantage increases at higher resolutions.

## Encode: WASM vs Browser Native (canvas.toBlob)

| Resolution | WASM (ms) | Native (ms) | Ratio | Throughput |
|-----------|-----------|-------------|-------|------------|
| 256x256 | 0.50 | 0.40 | 1.25x (slower) | 131.1 MP/s |
| 1024x768 | 5.00 | 2.70 | 1.85x (slower) | 157.3 MP/s |
| 1920x1080 | 13.30 | 6.50 | 2.05x (slower) | 155.9 MP/s |
| 2560x1440 | 23.00 | 20.30 | 1.13x (slower) | 160.3 MP/s |
| 3840x2160 | 50.70 | 30.00 | 1.69x (slower) | 163.6 MP/s |
| 7680x4320 | 203.50 | 92.40 | 2.20x (slower) | 163.0 MP/s |

**Encode summary**: WASM encode is **1.13x-2.20x slower** than native. Largest gap at 7680x4320 (2.20x).

## Scalar Fallback Hotspots (WASM decode pipeline)

These functions fall back to scalar on wasm32, despite having SIMD implementations on NEON/AVX2:

| Function | Location | Impact | Used by |
|----------|----------|--------|---------|
| `idct_ifast` | `simd/wasm32/mod.rs:23` | Low | Non-default DCT method |
| `idct_float` | `simd/wasm32/mod.rs:24` | Low | Non-default DCT method |
| `fancy_h2v2` | `decode/pipeline.rs:938-967` | **HIGH** | All 4:2:0 decode (most common) |
| `merged_h2v1` | `decode/pipeline.rs:856-879` | **HIGH** | H2V1 merged upsample+color path |
| `merged_h2v2` | `decode/pipeline.rs:883-914` | **HIGH** | H2V2 merged upsample+color path |

### Priority Order (by impact on decode performance)
1. **Fancy H2V2 upsample** — used by every 4:2:0 JPEG (>80% of all JPEGs)
2. **Merged H2V2 upsample+color** — fuses H2V2 upsample + YCbCr→RGB, eliminates intermediate buffer for 4:2:0
3. **Merged H2V1 upsample+color** — fuses H2V1 upsample + YCbCr→RGB for 4:2:2
4. **RGB interleave optimization** — current `wasm_ycbcr_to_rgb_row_inner` uses scalar loop for 3-byte store
5. **Multi-format color conversion** — only RGB/RGBA supported; no BGR/BGRA/RGBX/BGRX

### Note on Encode Gap
The 1.13x-2.20x encode slowness likely comes from:
- Huffman encoding is entirely scalar (no SIMD acceleration on any platform)
- The FDCT+quantize pipeline is already SIMD, but the encode overhead is dominated by entropy coding
- Browser's `canvas.toBlob` may use native multi-threaded encoding
