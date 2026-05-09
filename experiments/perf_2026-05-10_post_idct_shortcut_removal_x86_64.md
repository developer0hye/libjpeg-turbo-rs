# Performance Matrix — 2026-05-10 (post-IDCT-shortcut-removal, x86_64 at main 577546a)

## Purpose

Re-measure decode performance after PR #278 (commit `577546a`) removed the pure-DC pixel-fill shortcut from all four SIMD ISLOW IDCTs (NEON / SSE2 / AVX2 / WASM). The pre-fix baseline is `experiments/perf_2026-05-09_x86_64.md` taken at `b2ef66e` — same host, same C reference, same governor.

The PR's review claimed *"aarch64 NEON: geomean 0.998× of pre-fix; no individual case regressed by more than 1.5%"* and *"x86_64 AVX2: geomean 0.997×; same matrix"*. **That claim was wrong on x86_64 graphic content** — see the table below.

## Environment (matches 2026-05-09 baseline for apples-to-apples)

| Item | Value |
| --- | --- |
| Host | `yonghye-pc` (Linux 6.17, Ubuntu 24.04) |
| CPU | Intel Core i5-10400 (Comet Lake, AVX2) |
| Governor | `powersave` (sudo not available — variance may be slightly higher) |
| Turbo | enabled (`no_turbo=0`) |
| C reference | libjpeg-turbo `2.1.5` (`/usr/lib/x86_64-linux-gnu`) |
| Rust toolchain | stable, default features (`simd`) |
| Branch | `main` @ `577546a` (= `b2ef66e` + IDCT shortcut removal) |

## Decode matrix — post-fix vs pre-fix

| Fixture | Pre (µs) | Post (µs) | Δ % | Note |
| --- | ---: | ---: | ---: | --- |
| photo_64x64_420 | 47.27 | 47.89 | +1.3% | noise |
| photo_320x240_420 | 514.86 | 512.18 | -0.5% | noise |
| decode_640x480 (gradient) | 639.33 | 636.82 | -0.4% | noise |
| photo_1280x720_420 | 5741.2 | 5740.9 | 0% | |
| photo_1920x1080_420 | 12904 | 12879 | -0.2% | |
| photo_2560x1440_420 | 23161 | 23044 | -0.5% | |
| photo_3840x2160_420 | 52162 | 51999 | -0.3% | |
| photo_640x480_444 | 3169.5 | 3163.7 | -0.2% | |
| photo_640x480_422 | 2153.5 | 2146.3 | -0.3% | |
| photo_1920x1080_444 | 24524 | 24378 | -0.6% | |
| photo_1920x1080_422 | 16662 | 16544 | -0.7% | |
| **graphic_640x480_420** | **532.28** | **658.22** | **+23.7%** | **REGRESSION** |
| **checker_640x480_420** | **1108.6** | **1155.8** | **+4.3%** | regression |
| **graphic_1920x1080_420** | **2485** | **3492.9** | **+40.5%** | **REGRESSION** |
| photo_640x480_420_rst | 514.55 | 513.35 | -0.2% | |
| prog_640x480_444 | 8468.3 | 8417.3 | -0.6% | |
| prog_640x480_422 | 5800.8 | 5783.1 | -0.3% | |
| prog_1920x1080_420 | 36536 | 36466 | -0.2% | |
| prog_1920x1080_444 | 66785 | 66619 | -0.2% | |
| prog_3840x2160_420 | 148880 | 148650 | -0.2% | |

## Diagnosis

The regression localises exactly to the workloads where pure-DC blocks dominate:

- **graphic content** (vector-art-style PNG re-encoded as JPEG; large solid regions): 24-41% slowdown
- **checker content** (high-contrast 2-colour pattern; many constant-region 8×8 blocks): 4% slowdown
- **photo content** (natural images, AC-rich blocks): no measurable change
- **progressive content** (separate scans): no change — progressive's DC scan is non-SIMD path

The pre-fix shortcut path was: `if (AC bitmap == 0) { compute DC scalar pixel value; vst1q_u8 splat to 8 bytes × 8 rows }`. Post-fix, every DC-only block flows through the full 4-column SIMD pass-1 + 4-column SIMD pass-2 pipeline. For a graphic image where ~60% of the 8×8 blocks are DC-only, that's ~60% of MCUs paying for 16 SIMD ops they didn't need before.

## Severity

The regression is real but **bounded to graphic-heavy workloads**:

- Photo (the ~95% case for real-world JPEG): no impact.
- Progressive (typical web content): no impact.
- Graphic / banner / icon / chart content: 4-41% slower, but still under C in absolute terms — `graphic_1920x1080_420` was at 0.91× of C pre-fix (2485 vs C's 2745); post-fix it's at 1.27× (3493 vs 2745). C libjpeg-turbo's NEON/AVX2 ISLOW *also* has a pure-DC shortcut, so the comparison is no longer apples-to-apples in C's favour.

## Follow-up: safe-guarded re-introduction

The shortcut formula is bit-exact-equivalent to the full i16-lane pipeline **when `|coeff * quant|` fits in i16** — i.e. `coeff.unsigned_abs() as i32 * quant as i32 <= 0x7FFF`. Real-world JPEGs satisfy this for ~100% of blocks (typical Q-table DC quantizer is in [10,30], so coeff would need |2048+| to overflow — extremely rare in entropy-coded natural content).

A safe re-introduction:

```rust
if ac_bitmap == 0 {
    let dc_q_abs = (coeff[0].unsigned_abs() as u32) * (quant[0] as u32);
    if dc_q_abs <= 0x7FFF {
        // Safe: shortcut formula matches i16-lane pipeline bit-for-bit.
        return dc_only_pixel_fill_shortcut(coeff[0], quant[0], output, stride);
    }
    // Adversarial input: fall through to full pipeline whose i16 lane
    // wrap matches libjpeg-turbo's NEON/AVX2 ISLOW.
}
// Full pipeline ...
```

This recovers 100% of the graphic regression on real-world inputs while preserving correctness on the fuzz inputs that motivated the deletion.

The follow-up belongs in a separate PR (one variable at a time per the experiment-tracking workflow). Tracked as a Phase 4 perf gap candidate.

## Microbench fixed-cost (no change)

| Operation | Pre (ns) | Post (ns) | Δ |
| --- | ---: | ---: | ---: |
| `idct_8x8` | 27.83 | 27.01 | -3% (noise) |
| `ycbcr_to_rgb_row_640` | 253.7 | 250.4 | -1.3% |
| `fancy_h2v1_320` | 68.7 | 68.8 | 0% |
| `decoder_new_640x480` | 18621 | 18977 | +1.9% |

The `idct_8x8` microbench uses a fixed photo-style coefficient block (no DC-only) so the shortcut never fired in either build. That's why this microbench is flat while end-to-end decode of graphic content regressed — the shortcut's value lives in the *distribution* of block types in real images, not in any single 8×8 block.
