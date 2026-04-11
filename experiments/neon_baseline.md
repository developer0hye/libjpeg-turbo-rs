# NEON Optimization Baseline (2026-04-11)

Platform: macOS aarch64 (Apple Silicon M2), release build with fat LTO.

## Decode: Rust vs C libjpeg-turbo

| Case | Rust (µs) | C (µs) | Rust/C |
|------|-----------|--------|--------|
| photo_64x64_420 | 30.7 | 28.3 | 1.08 |
| photo_320x240_420 | 505.8 | 531.0 | **0.95** |
| gradient_640x480 | 461.0 | 382.5 | 1.21 |
| photo_1280x720_420 | 5672.6 | 5636.4 | 1.01 |
| photo_1920x1080_420 | 12436 | 13112.9 | **0.95** |
| photo_2560x1440_420 | 22204 | 21909.3 | 1.01 |
| photo_3840x2160_420 | 50048 | 47993.9 | 1.04 |
| photo_640x480_444 | 2966.9 | 3089.2 | **0.96** |
| photo_640x480_422 | 1982.7 | 2027.8 | **0.98** |
| photo_1920x1080_444 | 23054 | 23859.1 | **0.97** |
| photo_1920x1080_422 | 15490 | 16245.6 | **0.95** |
| graphic_640x480_420 | 402.5 | — | — |
| checker_640x480_420 | 934.5 | — | — |
| graphic_1920x1080_420 | 2122.8 | — | — |

**Bold** = Rust faster than C.

### Decode Analysis
- **Photo content**: Rust matches or beats C across all resolutions and subsampling modes (0.95x–1.04x).
- **Gradient content**: 1.21x gap — gradient_640x480 has uniform blocks where C's IDCT or pipeline may have advantages.
- **Progressive decode**: Not tested in this baseline (C has progressive variants, Rust bench doesn't include them).
- **Key hotspot**: gradient decode gap suggests IDCT or color conversion for low-entropy blocks could improve.

## Encode: Rust vs C libjpeg-turbo

| Case | Rust (µs) | C (µs) | Rust/C |
|------|-----------|--------|--------|
| encode_320x240_420 | 219.7 | 196.6 | 1.12 |
| encode_320x240_422 | 255.2 | 250.5 | 1.02 |
| encode_320x240_444 | 435.8 | 384.4 | 1.13 |
| encode_640x480_422 | 928.9 | 930.8 | **1.00** |
| encode_640x480_444 | 1414.6 | 1419.4 | **1.00** |
| encode_1920x1080_420 | 6094.7 | 5894.5 | 1.03 |
| encode_1920x1080_422 | 7662.2 | 7598.9 | 1.01 |
| encode_1920x1080_444 | 11415 | 11063.1 | 1.03 |

### Encode Analysis
- **All modes within 1.00x–1.13x of C** — excellent parity.
- **Smallest images (320x240)**: 1.02x–1.13x gap, likely due to setup/overhead amortization.
- **Large images (1080p)**: 1.01x–1.03x — nearly identical to C.
- **640x480 422/444**: Matching or beating C (1.00x).

## Summary

| Category | Best Ratio | Worst Ratio | Avg Ratio |
|----------|-----------|-------------|-----------|
| Decode (photo) | 0.95 | 1.08 | 0.99 |
| Decode (gradient) | 1.21 | 1.21 | 1.21 |
| Encode | 1.00 | 1.13 | 1.04 |

**Overall**: Decode is at parity with C for photo content, with a gap on gradient/low-entropy.
Encode is within 3% of C at 1080p+. Merged upsample (Phase 2) may improve decode further.
