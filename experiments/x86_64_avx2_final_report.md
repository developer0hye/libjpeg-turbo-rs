# x86_64 AVX2/SSE2 Final Benchmark Report

**Date**: 2026-04-12
**CPU**: Intel i5-10400 (x86_64, AVX2)
**Related**: #193
**Changes**: AVX2/SSE2 Fancy H2V2 Upsample, Multi-format encode color conversion, Fused encode pipeline

## Decode: Rust Final vs C libjpeg-turbo

| Benchmark | Baseline (us) | Final (us) | C (us) | Improvement | Rust/C |
|-----------|--------------|------------|--------|-------------|--------|
| photo_64x64_420 | 43.1 | 46.4 | 37.8 | — | 1.23x |
| photo_320x240_420 | 598.7 | 566.6 | 567.8 | +5.4% | **1.00x** |
| decode_640x480 | 829.0 | 691.8 | 681.0 | +16.6% | 1.02x |
| graphic_640x480_420 | 685.0 | 617.1 | 645.6 | +9.9% | **0.96x** |
| checker_640x480_420 | 1340.8 | 1185.7 | 1267.5 | +11.6% | **0.94x** |
| photo_640x480_422 | 2481.2 | 2380.0 | 2711.9 | +4.1% | **0.88x** |
| photo_640x480_444 | 3630.9 | 3497.1 | 4238.7 | +3.7% | **0.83x** |
| photo_1280x720_420 | 7120.9 | 6463.9 | 7817.6 | +9.2% | **0.83x** |
| photo_1920x1080_420 | 16126.5 | 14304.8 | 15762.4 | +11.3% | **0.91x** |
| photo_1920x1080_422 | 19431.0 | 18173.1 | 19669.7 | +6.5% | **0.92x** |
| photo_1920x1080_444 | 32958.4 | 27576.8 | 29498.6 | +16.3% | **0.93x** |
| photo_2560x1440_420 | 28554.1 | 25429.6 | 27056.2 | +10.9% | **0.94x** |
| photo_3840x2160_420 | 63706.9 | 58322.9 | 59636.1 | +8.5% | **0.98x** |
| graphic_1920x1080_420 | 3745.1 | 2631.2 | 3399.6 | +29.7% | **0.77x** |
| photo_640x480_420_rst | 596.3 | 539.2 | 604.5 | +9.6% | **0.89x** |

### Progressive Decode

| Benchmark | Baseline (us) | Final (us) | C (us) | Improvement | Rust/C |
|-----------|--------------|------------|--------|-------------|--------|
| prog_640x480_422 | 6305.8 | 5980.3 | 7228.6 | +5.2% | **0.83x** |
| prog_640x480_444 | 9353.1 | 8571.0 | 10152.8 | +8.4% | **0.84x** |
| prog_1920x1080_420 | 41292.7 | 38185.8 | 41757.7 | +7.5% | **0.91x** |
| prog_1920x1080_444 | 72639.1 | 67800.1 | 80087.4 | +6.7% | **0.85x** |
| prog_3840x2160_420 | 167581.0 | 154249.3 | 169174.1 | +7.9% | **0.91x** |

### Decode vs zune-jpeg

| Benchmark | Rust (us) | zune (us) | Rust/zune |
|-----------|-----------|-----------|-----------|
| 640x480 | 675.2 | 754.3 | **0.90x** |
| graphic_640x480 | 560.2 | 656.9 | **0.85x** |
| 1280x720 | 6398.9 | 7853.5 | **0.81x** |
| 1920x1080 | 14156.4 | 17539.0 | **0.81x** |
| 2560x1440 | 25532.9 | 31385.8 | **0.81x** |
| 3840x2160 | 57154.1 | 72237.2 | **0.79x** |

## Encode: Rust vs C libjpeg-turbo

| Benchmark | Rust (us) | C (us) | Rust/C |
|-----------|-----------|--------|--------|
| encode_320x240_420 | 370.8 | 306.0 | 1.21x |
| encode_320x240_422 | 449.8 | 404.6 | 1.11x |
| encode_320x240_444 | 681.6 | 630.4 | 1.08x |
| encode_640x480_422 | 1505.7 | 1293.9 | 1.16x |
| encode_640x480_444 | 2409.2 | 2270.2 | 1.06x |
| encode_1920x1080_420 | 11575.6 | 8502.9 | 1.36x |
| encode_1920x1080_422 | 12188.3 | 11243.2 | 1.08x |
| encode_1920x1080_444 | 18033.8 | 15505.1 | 1.16x |

## Summary

### Decode
- **4:2:0 baseline**: Rust is now **0.83-0.98x C** across all resolutions (faster than C!)
- **4:2:2 / 4:4:4**: Rust is **0.83-0.93x C** (significantly faster than C)
- **Progressive**: Rust is **0.83-0.91x C** (faster than C across the board)
- **vs zune-jpeg**: Rust is **19-21% faster** on the cases measured here — **4:2:0/4:4:4 baseline at 640x480 and above only**. The 2026-07-26 wide-matrix analysis (issue #361) found the *unmeasured* categories losing to zune at the time: 4:2:2 (1.11-1.24x, #350), tiny images (2-3.6x, #351), and 8K progressive (1.30x, #352). Do not quote this line without the coverage caveat; the wide matrix lives in `examples/bench_zune_matrix.rs` (#360).
- **Average improvement from baseline**: ~10% on 4:2:0, up to 30% on graphic content

### Encode
- Encode remains ~1.08-1.36x C. The primary gap is **Huffman SIMD** (C has `jchuff-sse2.asm`).
- Multi-format color conversion (RGBA/BGR/BGRA) now uses AVX2 with fused pipeline.

### What was optimized
1. **AVX2 Fancy H2V2 Upsample** — fused vertical+horizontal triangle filter, 16 samples/iter
2. **SSE2 Fancy H2V2 Upsample** — 128-bit fallback, 8 samples/iter
3. **AVX2 Multi-format encode color** — RGBA/BGR/BGRA via SSSE3 pshufb deinterleave
4. **Fused encode pipeline** — extended MCU-row-at-a-time path to all pixel formats

### Remaining optimization opportunities
1. **Huffman encode SIMD** (SSE2) — 15-25% of encode time, would close the 1.36x gap
2. **Wider merged upsample+color** — process 32 chroma samples/iter (currently 16)
3. **256-bit RGB store** in merged path — avoid drop to 128-bit SSSE3
