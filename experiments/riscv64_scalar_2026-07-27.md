# RISC-V scalar-path measurement (issue #359 step 1)

**Purpose.** Issue #359 requires this measurement *before* any
portable-SIMD implementation: "build for `riscv64gc-unknown-linux-gnu`
and measure the scalar path against C libjpeg-turbo on the same target.
If we are at or above C, this is not urgent."

## Environment

- Target: `riscv64gc-unknown-linux-gnu` (rustc 1.97.1), `--release` (lto).
- Host: Apple aarch64 running `linux/riscv64` containers under Docker
  emulation. **Absolute times are emulated and meaningless; only the
  ratios on this page are.** Both sides pay the same emulation tax.
- C reference: libjpeg-turbo 2.1.5 (`djpeg`, distro build).
- Ours: `libjpeg_turbo_rs::decompress()`, scalar kernels (no RVV path
  exists in either implementation — this is scalar vs scalar).

## vs C libjpeg-turbo

`djpeg` is a process, so its wall time includes fork/exec plus a PPM
write. Measured fixed overhead on an 8×8 input: **30,951 µs** (dominated
by emulated process startup). Subtracting it from each total gives an
approximate decode-only C figure; the subtraction slightly *favours C
less* than reality, since C also writes the PPM inside the remainder.

| case | C total | C decode-only (est.) | ours (in-process) | ours/C |
|---|---:|---:|---:|---:|
| `photo_640x480_420` | 38,718 | ~7,767 | 22,143 | **~2.9×** |
| `photo_1920x1080_420` | 98,480 | ~67,529 | 168,094 | **~2.5×** |

**We are roughly 2.5–2.9× slower than C libjpeg-turbo on RISC-V.**

## vs zune-jpeg (same target, same harness)

Full `examples/bench_zune_matrix` run: **17 wins / 15 losses / 2 ties**
across 34 scored cases — compared with 31 W / 3 L on aarch64 with NEON.

| case | ours | zune | ratio |
|---|---:|---:|---:|
| `photo_1920x1080_420_prog` | 230,864 | 403,700 | **0.57** |
| `photo_640x480_444_prog` | 60,710 | 99,613 | **0.61** |
| `photo_320x240_444` | 9,246 | 12,962 | **0.71** |
| `photo_640x480_420_rst` | 5,512 | 6,179 | **0.89** |
| `photo_1920x1080_420` | 168,094 | 165,945 | 1.01 |
| `photo_640x480_420` | 22,143 | 19,331 | **1.15** |
| `rw_8k_420_q75` | — | — | **2.04** |
| `graphic_640x480_420` | — | — | **1.80** |

Progressive wins decisively (the #352 `ac_max_k` work is
architecture-independent); dense baseline 4:2:0 loses.

## Conclusion — the issue's premise does not hold

#359 assumed: *"Upstream libjpeg-turbo itself has no RVV/VSX path for
most kernels, so scalar-on-RISC-V is parity with C, not a regression
against it."* **Measured, that is false.** C has no RVV either, yet its
*scalar* kernels beat ours by ~2.5–2.9×. The gap is in our scalar code,
not in a missing vector unit.

That reframes the work:

1. The actionable gap is **scalar-kernel quality** (IDCT, upsample,
   colour convert), which is fixable on stable Rust today and benefits
   every non-SIMD target — RISC-V, POWER, s390x, LoongArch, 32-bit ARM,
   and the `no_std` builds from #356 that dispatch scalar by default.
2. `portable_simd` remains blocked on `core::simd` stabilising
   (nightly-only; we ship stable with `rust-version = "1.87"`), so it
   cannot be the answer for those targets yet regardless.

Recommendation: keep #359 open for the portable-SIMD half, and treat
scalar-kernel optimisation as the near-term item — it is stable-Rust
work with a measured 2.5×+ headroom.

## 2026-07-28 — P4-60 step 1: table-driven YCbCr→RGB (branch feat/issue-359-scalar-color-tables)

Same harness shape (linux/riscv64 container under emulation, `rust:1-slim`
image — note the C side is this image's distro djpeg, a DIFFERENT build
than the 2026-07-27 run, so the ours/C ratios below are comparable only
within this section). The two decode rows carry their 07-27 figures for
orientation, but **no same-env baseline was re-measured**: that pair
crosses container images, so the end-to-end delta it implies is
indicative, not a controlled A/B. The kernel A/B is the controlled
measurement — same binary, same run, one call site swapped.

| measurement | value |
|---|---:|
| kernel A/B, 1080p frame of rows: multiply form | 34,479 µs |
| kernel A/B, same: table form | **21,120 µs (1.63× faster)** |
| decode `photo_640x480_420` (was 22,143 in the 07-27 env) | **19,015 µs** |
| decode `photo_1920x1080_420` (was 168,094) | **147,494 µs** |
| djpeg 640×480, best of 3 minus startup (~34,967) | ~17,050 µs |
| djpeg 1080p, best of 3 minus startup | ~85,802 µs |
| ours/C same-run: 640×480 | **~1.12×** |
| ours/C same-run: 1080p | **~1.72×** |

**Verdict: keep.** The conversion tables (exact precomputation of the
multiply form — bit-identical, proven exhaustively over chroma and by
49 simd-off djpeg cross-checks) cut the kernel 1.63×, and the decode
totals moved 12–14% against the 07-27 env — read that pair as the
direction, not a measured end-to-end win. The C comparison here is
coarse too (3 process-level runs, PPM write included, emulation noise —
the 640×480 ratio should be read as "near parity", not a precise
1.12), but the 1080p gap clearly remains
above the ≤1.2× target: the residual is in other stages (fancy
upsample, IDCT column passes, Huffman decode). P4-60 stays OPEN; next
candidate per C source: `jdsample.c`-style incremental fancy upsample
and Huffman decode table width.

## 2026-07-28 — P4-60 per-stage profile (the acceptance criteria's outstanding step)

Same-run numbers only (this container session is globally faster than
the earlier ones — emulation variance; ratios within the run are the
data). `bench_scalar_p460 profile` micro-benches each scalar kernel at
exactly a 1080p 4:2:0 decode's volume:

| stage (1080p volume) | µs | share of 111,190 µs total |
|---|---:|---:|
| `idct_8x8` × 48,600 blocks | 15,034 | 13.5% |
| `fancy_h2v2_row` × 2,160 rows | 10,198 | 9.2% |
| `ycbcr_to_rgb_row` × 1,080 rows (post-tables) | 15,199 | 13.7% |
| **Σ kernels** | **40,431** | **36.4%** |
| **residual: Huffman entropy decode + BitReader + plane plumbing** | **≈ 70,759** | **≈ 63.6%** |

**Conclusion.** Further kernel tuning has ≤ 36% of the pie to work
with; the scalar gap to C now lives overwhelmingly in entropy decode
(`jdhuff.c`'s two-level lookahead tables vs our `decode/huffman.rs`
path). P4-60's next step is the Huffman decode loop — with this
measurement, no longer a guess.
