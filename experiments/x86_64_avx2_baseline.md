# x86_64 AVX2/SSE2 Baseline Benchmark Report

**Date**: 2026-04-12
**CPU**: Intel i5-10400 (x86_64, AVX2)
**Related**: #193

## Decode Benchmarks (Rust vs C libjpeg-turbo)

| Benchmark | Rust (us) | C (us) | Ratio | Notes |
|-----------|-----------|--------|-------|-------|
| photo_64x64_420 | 43.1 | 37.8 | 1.14x | |
| photo_320x240_420 | 598.7 | 567.8 | 1.05x | |
| decode_640x480 (gradient) | 829.0 | 681.0 | 1.22x | |
| graphic_640x480_420 | 685.0 | 645.6 | 1.06x | |
| checker_640x480_420 | 1340.8 | 1267.5 | 1.06x | |
| photo_640x480_422 | 2481.2 | 2711.9 | 0.91x | Rust faster |
| photo_640x480_444 | 3630.9 | 4238.7 | 0.86x | Rust faster |
| photo_1280x720_420 | 7120.9 | 7817.6 | 0.91x | Rust faster |
| photo_1920x1080_420 | 16126.5 | 15762.4 | 1.02x | |
| photo_1920x1080_422 | 19431.0 | 19669.7 | 0.99x | ~parity |
| photo_1920x1080_444 | 32958.4 | 29498.6 | 1.12x | |
| photo_2560x1440_420 | 28554.1 | 27056.2 | 1.06x | |
| photo_3840x2160_420 | 63706.9 | 59636.1 | 1.07x | |
| graphic_1920x1080_420 | 3745.1 | 3399.6 | 1.10x | |
| photo_640x480_420_rst | 596.3 | 604.5 | 0.99x | ~parity |

### Progressive Decode

| Benchmark | Rust (us) | C (us) | Ratio | Notes |
|-----------|-----------|--------|-------|-------|
| prog_640x480_422 | 6305.8 | 7228.6 | 0.87x | Rust faster |
| prog_640x480_444 | 9353.1 | 10152.8 | 0.92x | Rust faster |
| prog_1920x1080_420 | 41292.7 | 41757.7 | 0.99x | ~parity |
| prog_1920x1080_444 | 72639.1 | 80087.4 | 0.91x | Rust faster |
| prog_3840x2160_420 | 167581.0 | 169174.1 | 0.99x | ~parity |

### Decode vs zune-jpeg

| Benchmark | Rust (us) | zune (us) | Ratio |
|-----------|-----------|-----------|-------|
| 640x480 | 805.8 | 831.5 | 0.97x |
| graphic_640x480 | 735.6 | 708.4 | 1.04x |
| 1280x720 | 6860.2 | 8297.1 | 0.83x |
| 1920x1080 | 15916.9 | 18952.4 | 0.84x |
| 2560x1440 | 28929.8 | 33022.6 | 0.88x |
| 3840x2160 | 64369.2 | 74652.3 | 0.86x |

## Encode Benchmarks (Rust vs C libjpeg-turbo)

| Benchmark | Rust (us) | C (us) | Ratio | Notes |
|-----------|-----------|--------|-------|-------|
| encode_320x240_420 | 370.8 | 306.0 | 1.21x | |
| encode_320x240_422 | 449.8 | 404.6 | 1.11x | |
| encode_320x240_444 | 681.6 | 630.4 | 1.08x | |
| encode_640x480_422 | 1505.7 | 1293.9 | 1.16x | |
| encode_640x480_444 | 2409.2 | 2270.2 | 1.06x | |
| encode_1920x1080_420 | 11575.6 | 8502.9 | 1.36x | Biggest gap |
| encode_1920x1080_422 | 12188.3 | 11243.2 | 1.08x | |
| encode_1920x1080_444 | 18033.8 | 15505.1 | 1.16x | |

## Analysis

### Decode Performance
- **4:2:0 baseline decode**: ~1.02-1.07x C at high resolutions, near parity
- **4:2:2 and 4:4:4**: Rust is often faster than C (0.86-0.99x)
- **Progressive**: Rust matches or beats C across all cases
- **vs zune-jpeg**: Rust is 12-17% faster at large resolutions

### Decode Hotspot: H2V2 Upsample (Scalar Fallback)
The primary remaining scalar fallback in the decode path is **Fancy H2V2 upsample**, used for all 4:2:0 images (~90% of real-world JPEGs). This explains the ~5-7% gap on large 4:2:0 images (1920x1080, 2560x1440, 3840x2160).

### Encode Performance
- **1920x1080_420 encode**: 1.36x C — the biggest gap. Likely due to missing Huffman SIMD.
- **Small images**: 1.08-1.21x C
- **Large 4:4:4**: 1.16x C

### Priority Optimization Targets
1. **AVX2 Fancy H2V2 Upsample** — biggest decode scalar fallback gap
2. **SSE2 Fancy H2V2 Upsample** — SSE2-only CPU fallback
3. **Encode Huffman SIMD** — explains 1.36x gap on large encode
4. **Multi-format encode color conversion** — extend AVX2 RGB→YCbCr to other formats
