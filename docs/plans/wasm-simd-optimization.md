# WASM SIMD128 Optimization Plan

## Goal
Optimize WASM simd128 decode/encode pipeline to close the performance gap with browser-native JPEG (C libjpeg-turbo backed). Achieve parity or better across all resolutions (256x256 ~ 3840x2160) and subsampling modes (420/422/444).

## Current State

### Implemented (SIMD128)
| Component | Function | Status |
|-----------|----------|--------|
| IDCT islow | `wasm_idct_islow` | Wired in dispatch |
| YCbCr→RGB | `wasm_ycbcr_to_rgb_row` | Wired in dispatch |
| YCbCr→RGBA | `wasm_ycbcr_to_rgba_row` | **Exists but NOT wired in dispatch** |
| Fancy H2V1 | `wasm_fancy_upsample_h2v1` | Wired in dispatch |
| FDCT | `wasm_fdct` | Used by fdct_quantize |
| FDCT+Quantize | `wasm_fdct_quantize` | Wired in dispatch |
| Extract+FDCT+Quant | `wasm_extract_fdct_quantize` | Wired in dispatch |
| Downsample H2V2+FDCT+Quant | `wasm_downsample_h2v2_fdct_quantize` | Wired in dispatch |
| Downsample H2V1+FDCT+Quant | `wasm_downsample_h2v1_fdct_quantize` | Wired in dispatch |
| RGB→YCbCr | `wasm_rgb_to_ycbcr_row` | Wired in dispatch |

### Scalar Fallback (NOT SIMD)
| Component | Gap vs NEON/AVX2 | Impact |
|-----------|------------------|--------|
| IDCT ifast | NEON/AVX2 have SIMD | Low (islow is default) |
| IDCT float | NEON/AVX2 have SIMD | Low (rarely used) |
| Fancy H2V2 upsample | NEON has SIMD | **High** (4:2:0 is most common) |
| Merged H2V1 upsample+color | NEON+AVX2 have SIMD | **High** (eliminates intermediate buffer) |
| Merged H2V2 upsample+color | NEON+AVX2 have SIMD | **High** (420 decode hot path) |

### Suboptimal SIMD
| Component | Issue | Impact |
|-----------|-------|--------|
| YCbCr→RGB interleave | Uses scalar loop (8 iter) for 3-byte RGB store | **Medium** (every decoded pixel) |
| Multi-format decode color | Only RGB/RGBA; no BGR/BGRA/RGBX/BGRX | Medium (format flexibility) |
| Multi-format encode color | Only RGB→YCbCr; no BGR/RGBA/BGRA | Medium (format flexibility) |

## Rules
- **Output must be bit-identical to scalar.** Every SIMD function MUST produce diff=0 vs scalar fallback across ALL modes. Cross-validate using `cargo test --target wasm32-wasip1`.
- Every change MUST pass `cargo test --target wasm32-wasip1` before moving to the next task.
- Benchmark using Playwright MCP against browser-native JPEG (createImageBitmap/canvas.toBlob).
- One change at a time. Commit each completed task separately.
- If a change regresses correctness, revert and investigate.
- Use `@simd-optimizer` agent for SIMD implementation work and `@jpeg-expert` agent for spec verification.
- Reference existing aarch64 NEON and x86_64 AVX2/SSE2 implementations for algorithm design.

---

## Phase 1: Baseline Measurement & Benchmark Setup

### Task 1-1: WASM decode/encode baseline benchmark
- Build WASM crate: `cd crates/libjpeg-turbo-rs-wasm && wasm-pack build --release --target web`
- Serve `bench/` directory locally (e.g., `python3 -m http.server 8080`)
- Use Playwright MCP to open `http://localhost:8080/bench/index.html` in Chromium
- Click "Run Benchmark" and collect console output
- Record: WASM decode/encode times vs native (createImageBitmap/canvas.toBlob) at all resolutions
- Save baseline as `experiments/wasm_baseline.md`

### Task 1-2: Identify scalar fallback hotspots
- Run `cargo test --target wasm32-wasip1` to verify current state passes
- Grep dispatch table for scalar fallback calls to identify remaining scalar paths
- Document which decode/encode paths still hit scalar code
- Prioritize by frequency: 4:2:0 (H2V2) decode is most common JPEG subsampling

---

## Phase 2: Decode Optimization (High Impact)

### Task 2-1: WASM Fancy H2V2 Upsample
- **Reference**: `src/simd/aarch64/upsample.rs` (`neon_fancy_upsample_h2v2`, `neon_fancy_h2v2_row`)
- **What**: Implement `wasm_fancy_upsample_h2v2` — vertical+horizontal triangle filter for 4:2:0
- **Algorithm**: For each output pair (top row, bottom row):
  - Vertical: weighted average of current row and neighbor row (3:1 ratio)
  - Horizontal: triangle filter same as H2V1 (3:1 with alternating bias)
- **Fused single-pass**: Port the NEON fused approach — compute vertical colsum inline, then horizontal filter, avoiding intermediate buffer
- **Wire up**: No dispatch change needed (H2V2 fancy upsample is called via decode pipeline directly)
- **Verify**: `cargo test --target wasm32-wasip1`, diff=0 vs scalar `fancy_h2v2`

### Task 2-2: WASM Merged H2V1 Upsample+Color
- **Reference**: `src/simd/aarch64/merged.rs` (`neon_merged_h2v1_fn!` macro), `src/simd/x86_64/avx2_merged.rs`
- **What**: Fuse H2V1 chroma upsample + YCbCr→RGB into single pass
- **Why**: Eliminates intermediate upsample buffer allocation and memory traffic
- **Multi-format**: Use macro to generate RGB/RGBA variants (same pattern as NEON)
- **Wire up**: Register in merged upsample dispatch
- **Verify**: `cargo test --target wasm32-wasip1`, diff=0 vs scalar merged path

### Task 2-3: WASM Merged H2V2 Upsample+Color
- **Reference**: `src/simd/aarch64/merged.rs` (`neon_merged_h2v2_fn!` macro), `src/simd/x86_64/avx2_merged.rs`
- **What**: Fuse H2V2 chroma upsample + YCbCr→RGB — processes two output rows per chroma row
- **Algorithm**: Vertical averaging (3:1) + horizontal triangle filter + color conversion, all in one pass
- **Multi-format**: RGB/RGBA variants via macro
- **Wire up**: Register in merged upsample dispatch
- **Verify**: `cargo test --target wasm32-wasip1`, diff=0 vs scalar merged path

### Task 2-4: Optimize RGB Interleave in Color Conversion
- **What**: Replace scalar loop in `wasm_ycbcr_to_rgb_row_inner` (lines 97-110 in `color.rs`) with SIMD shuffle-based 3-byte interleave
- **Current**: After computing R/G/B u8 vectors, stores to temp arrays then does `for i in 0..8` scalar interleave
- **Target**: Use `i8x16_shuffle` to produce packed RGB output directly (similar to RGBA path at lines 196-204)
- **Challenge**: 3-byte-per-pixel (24 bytes for 8 pixels) doesn't divide evenly into 16-byte v128 — need two shuffles + partial store
- **Verify**: `cargo test --target wasm32-wasip1`, diff=0

---

## Phase 3: Encode Optimization

### Task 3-1: WASM Multi-Format Encode Color Conversion
- **Reference**: `src/simd/aarch64/color_encode.rs` (NEON multi-format: RGBA/BGR/BGRA)
- **What**: Extend `wasm_rgb_to_ycbcr_row` to handle RGBA/BGR/BGRA input formats
- **Approach**: Macro-based generation with different channel offsets (same pattern as NEON)
- **Wire up**: Register format-specific variants in encoder dispatch
- **Verify**: `cargo test --target wasm32-wasip1`, diff=0 vs scalar for all formats

### Task 3-2: WASM Multi-Format Decode Color Conversion
- **What**: Extend decode color conversion to handle BGR/BGRA/RGBX/BGRX output formats
- **Approach**: Macro-based generation with different channel reordering shuffles
- **Wire up**: Register format-specific variants in decode dispatch
- **Verify**: `cargo test --target wasm32-wasip1`, diff=0 vs scalar for all formats

---

## Phase 4: Performance Tuning

### Task 4-1: IDCT register pressure optimization
- **What**: Analyze `wasm_idct_islow` for register spills under WASM's virtual register model
- **Reference**: Compare against NEON IDCT (`src/simd/aarch64/idct.rs`) for algorithmic improvements
- **Key**: WASM JIT compilers (V8 Liftoff/TurboFan) have different register allocation than native — minimize live v128 values
- **Verify**: `cargo test --target wasm32-wasip1`, benchmark before/after

### Task 4-2: FDCT register pressure optimization
- **What**: Analyze `wasm_fdct` for similar register pressure issues
- **Reference**: Compare against NEON FDCT (`src/simd/aarch64/fdct.rs`)
- **Verify**: `cargo test --target wasm32-wasip1`, benchmark before/after

### Task 4-3: Quantize zigzag optimization
- **What**: Analyze `wasm_quantize_zigzag` — current approach gathers coefficients in scalar loop before SIMD multiply
- **Target**: Explore SIMD-based zigzag gather or pre-transposed coefficient layout to reduce scalar gather overhead
- **Verify**: `cargo test --target wasm32-wasip1`, benchmark before/after

---

## Phase 5: Final Verification & Benchmark

### Task 5-1: Full test suite validation
- Run `cargo test --target wasm32-wasip1` full suite
- Run `wasm-pack test --node` for WASM-bindgen interop tests
- Verify all decode/encode paths produce diff=0 vs scalar

### Task 5-2: Final performance benchmark
- Build optimized WASM: `wasm-pack build --release --target web`
- Run Playwright benchmark at all resolutions (256x256 ~ 3840x2160)
- Compare against Phase 1 baseline
- Compare against browser-native JPEG (createImageBitmap/canvas.toBlob)
- Produce final report: WASM/Native ratio table (resolution x operation)
- Save as `experiments/wasm_final_report.md`
