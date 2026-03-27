# Encoding Performance Report

Rust libjpeg-turbo-rs vs C libjpeg-turbo (3.1.0) encoding performance.

- **Platform**: Apple Silicon (aarch64), NEON SIMD
- **Quality**: 75 (default)
- **DCT method**: IsLow (integer slow DCT)
- **Date**: 2026-03-27
- **Branch**: `perf/encode-lto-rawptr-fuse`

## Benchmark Results

### Full Matrix: Rust vs C

| Resolution | Subsampling | Rust (µs) | C (µs) | Ratio | Status |
|------------|-------------|-----------|--------|-------|--------|
| 320×240 | 4:2:0 | 180.7 | 183.4 | **0.985×** | Rust faster |
| 320×240 | 4:2:2 | 223.6 | 233.4 | **0.958×** | Rust faster |
| 320×240 | 4:4:4 | 338.0 | 355.5 | **0.951×** | Rust faster |
| 640×480 | 4:2:2 | 823.0 | 824.7 | **0.998×** | Parity |
| 640×480 | 4:4:4 | 1200.8 | 1214.9 | **0.988×** | Rust faster |
| 1920×1080 | 4:2:0 | 5274.0 | 5075.7 | **1.039×** | C faster |
| 1920×1080 | 4:2:2 | 6471.9 | 6440.7 | **1.005×** | Parity |
| 1920×1080 | 4:4:4 | 9632.9 | 9714.1 | **0.992×** | Rust faster |

### Summary by Subsampling

| Subsampling | Avg Ratio | Notes |
|-------------|-----------|-------|
| 4:4:4 | **0.977×** | Rust consistently faster across all resolutions |
| 4:2:2 | **0.987×** | Rust faster or at parity |
| 4:2:0 | ~**1.01×** | Parity at low res, 4% gap at 1080p |

## Key Findings

1. **Rust matches or beats C in 7 out of 8 benchmark configurations.**
2. The only case where C is measurably faster is 1920×1080 4:2:0 (~4% gap).
3. At lower resolutions, Rust is consistently faster than C across all subsampling modes.

## Root Cause of 1080p 4:2:0 Gap

The ~200µs gap on 1080p 4:2:0 is **structural function-call overhead**, not algorithmic:

- `encode_block` saves 8 callee-saved registers (x19–x28) on every call
- 1080p 4:2:0 has 48,960 blocks/frame (120×68 MCUs × 6 blocks/MCU)
- Overhead: ~4ns/call × 48,960 = ~196µs — accounts for the entire gap
- C libjpeg-turbo inlines `encode_block` into the MCU loop; Rust cannot because inlining the 863-instruction function 6× per MCU blows L1 I-cache (confirmed: 3 failed attempts, all regressed)

This gap is proportionally smaller at lower resolutions (fewer blocks) and for 4:4:4/4:2:2 (fewer blocks per MCU relative to total work).

## Optimization History

Starting point: **1.50× slower than C** (19,147µs vs 4,980µs on 1080p 4:2:0).

### Session 1: NEON Foundation
| Optimization | Savings |
|---|---|
| NEON downsample chroma blocks (vpadalq H2V2/H2V1) | -23.8% |
| Bitmap zero-skip for AC Huffman (u64 leading_zeros) | -20.0% |
| Branchless Huffman + NEON extract_block | -9.6% |
| NEON reciprocal multiply quantize | -9.3% |
| Bulk flush BitWriter (free_bits countdown) | -10.6% |
| Branchless unchecked byte emission | -3.9% |
| **Subtotal** | **19,147 → 7,610µs** |

### Session 2: NEON Vectorized Huffman
| Optimization | Savings |
|---|---|
| NEON vectorized AC encode (vclzq + veorq + bitmap) | -13.7% |
| NEON TBL zigzag reorder (vqtbl4q_u8) | -8.0% |
| Remove put_bits redundant masking | -3.6% |
| **Subtotal** | **7,610 → 5,829µs** |

### Session 3: LTO + Fusion
| Optimization | Savings |
|---|---|
| Fat LTO (cross-module devirtualization) | -3.2% |
| Raw-pointer BitWriter (bypass Vec overhead) | -3.7% (at 444) |
| Fused extract_block + FDCT (eliminate intermediate buffer) | -1.8% |
| C-style local variable hoisting in Huffman loop | -0.8% |
| Sparse AC path (skip NEON pre-compute for ≤8 non-zero ACs) | -3.0% |
| Fused chroma downsample + FDCT + quantize | -1.4% (at 422) |
| **Subtotal** | **5,829 → 5,274µs** |

### Failed Experiments (13+)
See `experiments/encode.tsv` for full details. Notable failures:
- MCU-level hoisting (3 attempts): ABI register pressure negates savings
- `#[inline(always)]` encode_block: I-cache blowup (+7.7%)
- Packed Huffman tables: larger table footprint outweighs saved load
- Unrolled zigzag: register pressure regression

## Profile Breakdown (1080p 4:2:0)

| Component | % of Total | Notes |
|-----------|-----------|-------|
| Huffman encode_block | 62.6% | DC + AC encode + bitmap loop + put_bits |
| FDCT + quantize | 22.5% | neon_rows_fdct_quantize (fused paths) |
| Color convert | 13.7% | neon_rgb_to_ycbcr_row |
| Other | 1.2% | MCU loop overhead |

Both FDCT and color convert are at **algorithmic parity** with C (identical NEON operations, same fixed-point precision, same loop structure).

## Possible Future Improvements

1. **PGO (Profile-Guided Optimization)**: Compiler may optimize branch prediction and reduce prologue/epilogue with profile data. Not yet attempted.
2. **BOLT**: Binary-level hot function layout optimization for I-cache efficiency.
3. **Rust compiler improvements**: Partial inlining support could allow inlining the hot path of encode_block while keeping the cold flush path out-of-line.
