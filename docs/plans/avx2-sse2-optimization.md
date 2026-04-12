# AVX2/SSE2 SIMD Optimization Plan

## Goal
Optimize x86_64 AVX2/SSE2 decode/encode pipeline to match or beat C libjpeg-turbo across all resolutions (64x64 ~ 3840x2160) and subsampling modes (4:2:0/4:2:2/4:4:4).

## Current State

### Implemented (AVX2)
| Component | Function | Status |
|-----------|----------|--------|
| IDCT islow | `avx2_idct_islow` | Wired in dispatch |
| YCbCr→RGB (10 formats) | `avx2_ycbcr_to_rgb_row` | Wired in dispatch |
| Fancy H2V1 | `avx2_fancy_upsample_h2v1` | Wired in dispatch |
| Merged H2V1 upsample+color | `avx2_merged_h2v1_ycbcr_to_rgb` | Wired in decode pipeline |
| Merged H2V2 upsample+color | `avx2_merged_h2v2_ycbcr_to_rgb` | Wired in decode pipeline |
| FDCT islow | `avx2_fdct_islow` | Wired in dispatch |
| FDCT+Quantize+Zigzag | `avx2_fdct_quantize` | Wired in dispatch |
| Extract+FDCT+Quant | `avx2_extract_fdct_quantize` | Wired in dispatch |
| Downsample H2V2+FDCT+Quant | `avx2_downsample_h2v2_fdct_quantize` | Wired in dispatch |
| Downsample H2V1+FDCT+Quant | `avx2_downsample_h2v1_fdct_quantize` | Wired in dispatch |
| RGB→YCbCr | `avx2_rgb_to_ycbcr_row` | Wired in dispatch |

### Implemented (SSE2)
| Component | Function | Status |
|-----------|----------|--------|
| IDCT islow | `sse2_idct_islow` | Wired in dispatch (fallback) |
| YCbCr→RGB | `sse2_ycbcr_to_rgb_row` | Wired in dispatch (fallback) |
| Fancy H2V1 | `sse2_fancy_upsample_h2v1` | Wired in dispatch (fallback) |

### Scalar Fallback (NOT SIMD)
| Component | Gap vs NEON/WASM | Impact |
|-----------|------------------|--------|
| Fancy H2V2 upsample | NEON+WASM have SIMD | **High** (4:2:0 is most common) |
| IDCT ifast | NEON has SIMD | Low (islow is default) |
| IDCT float | NEON has SIMD | Low (rarely used) |
| Multi-format encode color | NEON has partial SIMD | Medium (only RGB→YCbCr) |
| SSE2 merged upsample+color | AVX2 has it, SSE2 doesn't | Medium (SSE2-only CPUs) |
| SSE2 FDCT+quantize | AVX2 has it, SSE2 doesn't | Medium (SSE2-only CPUs) |
| SSE2 RGB→YCbCr | AVX2 has it, SSE2 doesn't | Medium (SSE2-only CPUs) |
| Huffman encode SIMD | C has `jchuff-sse2.asm` | **High** (15-25% of encode time) |

### Baseline Performance (x86_64, i5-10400)
| Benchmark | Rust | C libjpeg-turbo | Ratio |
|-----------|------|-----------------|-------|
| graphic_640x480_420 | 884 us | ~884 us | ~1.00x |
| photo_640x480_420 | 616 us | TBD | TBD |
| photo_1920x1080_420 | 19.86 ms | TBD | TBD |

## Rules
- **Output must be bit-identical to C libjpeg-turbo.** Every SIMD function MUST produce diff=0 vs C djpeg/cjpeg output. Cross-validate using `cargo test`.
- Every change MUST pass `cargo test` before moving to the next task.
- Benchmark using `cargo bench` full matrix + `./bench_c_decode_linux` / `./bench_c_encode_matrix` for C comparison.
- One change at a time. Commit each completed task separately.
- If a change regresses correctness, revert and investigate.
- Profile before optimizing: `samply record` or `perf record` to identify actual hotspots.
- Stable CPU frequency: `echo performance > scaling_governor`, disable turbo boost before benchmarks.
- Reference existing aarch64 NEON and wasm32 SIMD128 implementations for algorithm design.
- Reference C ASM in `references/libjpeg-turbo/simd/x86_64/` for correctness verification.

## Correctness Notes (from @jpeg-expert review)

- **H2V2 rounding**: AVX2 has no rounding-shift instruction. Use WASM/C pattern (explicit bias +8 for even, +7 for odd, then `vpsrlw 4`). Do NOT port NEON's `vrshrn` pattern directly.
- **Lane-crossing**: AVX2 H2V2 horizontal neighbor access crosses 128-bit lane boundary. Use `_mm256_permute2x128_si256` or `_mm256_alignr_epi8` for the shift. 3-cycle latency, not a blocker but must be correct.
- **Boundary safety**: C ASM writes dummy samples past buffer end — Rust cannot do this. Use `col + 16 <= in_width` (SSE2) / `col + 32 <= in_width` (AVX2) loop condition with scalar tail, matching NEON/WASM approach.
- **Edge pixels**: Last-column odd position must use edge replication (`cs_last*4 + 7) >> 4`). Verify scalar tail covers this.
- **Progressive JPEG**: No impact — upsample/color stages are post-IDCT, agnostic to scan structure.

## Boundary Test Matrix (mandatory for all upsample implementations)

| Test Case | Why |
|-----------|-----|
| width=1, 2, 3 | Below SIMD threshold, scalar fallback |
| width=15, 16, 17 | SSE2 boundary (exactly one chunk ± 1) |
| width=31, 32, 33 | AVX2 boundary (exactly one chunk ± 1) |
| height=1 | Edge replication (above=below=current row) |
| Odd output dimensions | chroma width rounding edge cases |
| All-zero / all-255 chroma | Verify no u16 overflow (max colsum horizontal = 4088, fits u16) |
| Restart marker boundary | Upsample state must not leak across restart intervals |
| Merged vs separate path | Cross-validate identical output on non-aligned widths |

---

## Phase 1: Baseline Measurement & Gap Analysis

### Task 1-1: Full decode/encode benchmark baseline
- Set stable CPU frequency (performance governor, no turbo boost)
- Run `cargo bench` full decode matrix (all resolutions, subsampling modes)
- Run `cargo bench` full encode matrix
- Run `./bench_c_decode_linux` and `./bench_c_encode_matrix` for C comparison
- Record: Rust vs C times and ratio for each benchmark
- Save as `experiments/x86_64_avx2_baseline.md`

### Task 1-2: Profile decode hotspots
- `samply record` or `perf record` on `cargo bench -- decode_1920x1080_420`
- Identify top-5 hotspots by time percentage
- Document which functions are SIMD vs scalar fallback
- Prioritize optimization targets by measured impact

---

## Phase 2: Decode Optimization (High Impact)

### Task 2-1: AVX2 Fancy H2V2 Upsample
- **Reference**: `src/simd/wasm32/upsample.rs` (`wasm_fancy_upsample_h2v2`) as primary algorithm reference, `references/libjpeg-turbo/simd/x86_64/jdsample-avx2.asm`, `src/simd/aarch64/upsample.rs` (`neon_fancy_upsample_h2v2`)
- **What**: Implement `avx2_fancy_upsample_h2v2` — fused vertical+horizontal triangle filter for 4:2:0
- **Algorithm**: Single-pass approach (follow WASM/C pattern, NOT NEON vrshrn):
  - Vertical: `colsum = cur*3 + neighbor` in u16
  - Horizontal even: `(thiscs*3 + lastcs + 8) >> 4`
  - Horizontal odd: `(thiscs*3 + nextcs + 7) >> 4`
  - Explicit bias +8/+7 before `vpsrlw 4` (no rounding-shift instruction on x86)
- **Lane-crossing**: Horizontal neighbor at 128-bit boundary requires `_mm256_permute2x128_si256` + `_mm256_alignr_epi8` or `_mm256_permutevar8x32_epi32`
- **Loop condition**: `col + 16 <= in_width` for AVX2 (16 u16 samples per iteration), scalar tail for remainder
- **Why**: 4:2:0 is ~90% of real-world JPEGs; this is the #1 scalar fallback gap
- **Wire up**: Call from decode pipeline where `fancy_h2v2` is invoked
- **Verify**: `cargo test`, diff=0 vs scalar `fancy_h2v2`, run boundary test matrix above

### Task 2-2: SSE2 Fancy H2V2 Upsample
- **Reference**: Same as 2-1 but using 128-bit SSE2 intrinsics
- **What**: Implement `sse2_fancy_upsample_h2v2` for SSE2-only CPU fallback
- **Algorithm**: Same single-pass approach, 8 samples per iteration, `col + 8 <= in_width`
- **Verify**: `cargo test`, diff=0 vs scalar, boundary test matrix

### Task 2-3: Optimize merged upsample+color hot path
- **What**: Profile `avx2_merged_h2v2_ycbcr_to_rgb` and `avx2_merged_h2v1_ycbcr_to_rgb` for bottlenecks
- **Reference**: `references/libjpeg-turbo/simd/x86_64/jdmrgext-avx2.asm`
- **Note**: Merged path uses box-filter (nearest-neighbor chroma duplication), NOT triangle filter. This is correct and matches C libjpeg-turbo design.
- **Actions**:
  - Compare register usage vs C ASM
  - Check for unnecessary loads/stores in the inner loop
  - Verify interleave/store patterns match C's optimized approach
- **Verify**: `cargo test`, benchmark before/after
- **Audit findings (2026-04-12)**:
  - Rust processes 16 chroma samples/iter vs C's 32 (2-pass Y loop). ~25% extra chroma math.
  - Rust drops to 128-bit SSSE3 for RGB interleave/store; C stays 256-bit. ~5-8% loss.
  - Chroma math is identical (same coefficients, same approach).
  - **Deferred to Phase 4 (Task 4-3)**: widen to 32-chroma inner loop + 256-bit RGB store.

---

## Phase 3: Encode Optimization

### Task 3-1: Multi-format encode color conversion (AVX2)
- **Reference**: `src/simd/aarch64/color_encode.rs` (NEON multi-format)
- **What**: Extend `avx2_rgb_to_ycbcr_row` to handle RGBA/BGR/BGRA input formats
- **Approach**: Macro-based generation with different channel deinterleave shuffles (SSSE3 pshufb)
- **Wire up**: Register format-specific variants in encoder dispatch
- **Verify**: `cargo test`, diff=0 vs scalar for all formats

### Task 3-2: Huffman encode SIMD (SSE2)
- **Reference**: `references/libjpeg-turbo/simd/x86_64/jchuff-sse2.asm`, `jcphuff-sse2.asm`
- **What**: SIMD-accelerated Huffman encoding — counting leading zeros and bitstream packing
- **Why**: Huffman encoding is 15-25% of encode time. C has SSE2 SIMD for this, Rust port is scalar-only
- **Verify**: `cargo test`, diff=0 vs scalar, encode benchmark before/after

---

## Phase 4: Performance Tuning (profile-gated)

> **Gate**: Only proceed with Tasks 4-1/4-2 if profiling shows the target function occupies >5% of total time. Otherwise skip directly to Task 4-3.

### Task 4-1: IDCT register pressure and memory access
- **What**: Profile AVX2 IDCT for cache misses, register spills, unnecessary moves
- **Reference**: Compare against `references/libjpeg-turbo/simd/x86_64/jidctint-avx2.asm`
- **Key**: Check if the DC-only fast path triggers frequently enough; optimize transpose operations
- **Verify**: `cargo test`, benchmark before/after

### Task 4-2: Color conversion throughput
- **What**: Profile AVX2 YCbCr→RGB for throughput bottlenecks
- **Reference**: `references/libjpeg-turbo/simd/x86_64/jdcolext-avx2.asm`
- **Key**: Check if interleave/store pattern (3-byte RGB via pshufb) can be improved
- **Verify**: `cargo test`, benchmark before/after

### Task 4-3: Whole-pipeline optimization
- **What**: Profile end-to-end decode/encode for inter-function overhead
- **Key areas**:
  - Memory allocation between pipeline stages
  - Cache locality of coefficient buffers
  - Unnecessary copies between SIMD and scalar boundaries
- **Verify**: `cargo test`, full benchmark matrix before/after

---

## Phase 5: Final Verification & Benchmark

### Task 5-1: Full test suite validation
- Run `cargo test` full suite
- Cross-validate all decode outputs against C `djpeg` (diff=0)
- Cross-validate all encode outputs against C `cjpeg` (diff=0)
- Verify all subsampling modes: 4:2:0, 4:2:2, 4:4:4
- Verify all output formats: RGB, RGBA, BGR, BGRA, RGBX, BGRX
- Run full boundary test matrix from Correctness Notes section

### Task 5-2: Final performance benchmark
- Run `cargo bench` full decode+encode matrix
- Run C comparison benchmarks
- Produce Rust/C ratio table: resolution x subsampling x operation
- Compare against Phase 1 baseline
- Save as `experiments/x86_64_avx2_final_report.md`

### Task 5-3: Documentation update
- Update `docs/FEATURE_PARITY.md` checkboxes for newly implemented features
- Update `docs/C_API_REFERENCE.md` status for new SIMD paths
- Update `README.md` if architecture/performance claims changed

---

## Deferred (low priority)

### SSE2 encode pipeline (FDCT+quantize, RGB→YCbCr)
- **Why deferred**: AVX2-less CPUs are pre-2013 hardware, diminishing returns
- **SSE2 lacks `pmulld`** — FDCT butterfly needs `pmadd`/shift emulation, substantial effort
- **Revisit**: if user demand or CI coverage requires SSE2 encode support
