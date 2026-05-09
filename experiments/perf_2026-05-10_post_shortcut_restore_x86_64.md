# Performance Matrix — 2026-05-10 (post-shortcut-restore, x86_64 at fix/idct-pure-dc-shortcut-safe-guard bb6ec77)

## Purpose

Verify that re-introducing the pure-DC pixel-fill shortcut with per-backend
lane semantics recovers the graphic-content regression measured in
`experiments/perf_2026-05-10_post_idct_shortcut_removal_x86_64.md`.

## Environment

Same as both prior 2026-05-09 / 2026-05-10 runs (Intel i5-10400, AVX2,
`powersave` governor, libjpeg-turbo 2.1.5).

## Decode matrix — three-way comparison

| Fixture | Pre-PR278 (µs) | Post-PR278 (µs) | Restored (µs) | vs pre-PR278 |
| --- | ---: | ---: | ---: | ---: |
| photo_64x64_420 | 47.27 | 47.89 | 47.04 | -0.5% |
| photo_320x240_420 | 514.86 | 512.18 | 517.87 | +0.6% |
| decode_640x480 (gradient) | 639.33 | 636.82 | 643.67 | +0.7% |
| photo_1280x720_420 | 5741 | 5741 | 5793 | +0.9% |
| photo_1920x1080_420 | 12904 | 12879 | 12987 | +0.6% |
| photo_2560x1440_420 | 23161 | 23044 | 23294 | +0.6% |
| photo_3840x2160_420 | 52162 | 51999 | 52484 | +0.6% |
| photo_640x480_444 | 3170 | 3164 | 3190 | +0.6% |
| photo_640x480_422 | 2154 | 2146 | 2161 | +0.3% |
| photo_1920x1080_444 | 24524 | 24378 | 24550 | +0.1% |
| photo_1920x1080_422 | 16662 | 16544 | 16681 | +0.1% |
| **graphic_640x480_420** | **532.28** | **658.22** | **534.61** | **+0.4%** ✓ |
| **checker_640x480_420** | **1108.6** | **1155.8** | **1108.5** | **0.0%** ✓ |
| **graphic_1920x1080_420** | **2485** | **3493** | **2500** | **+0.6%** ✓ |
| photo_640x480_420_rst | 514.55 | 513.35 | 516.67 | +0.4% |
| prog_640x480_444 | 8468 | 8417 | 8521 | +0.6% |
| prog_640x480_422 | 5801 | 5783 | 5799 | -0.04% |
| prog_1920x1080_420 | 36536 | 36466 | (full run) | — |
| prog_1920x1080_444 | 66785 | 66619 | (full run) | — |
| prog_3840x2160_420 | 148880 | 148650 | (full run) | — |

Photo + progressive: within ±1% of pre-PR278 (run-to-run noise).
Graphic + checker: **fully recovered** to within ±0.6% of pre-PR278.

## Microbench fixed-cost (no change)

| Operation | Pre-PR278 (ns) | Post-PR278 (ns) | Restored (ns) |
| --- | ---: | ---: | ---: |
| `idct_8x8` | 27.83 | 27.01 | 27.99 |
| `ycbcr_to_rgb_row_640` | 253.7 | 250.4 | 251.5 |
| `fancy_h2v1_320` | 68.7 | 68.8 | 68.6 |
| `decoder_new_640x480` | 18621 | 18977 | 18550 |

## Conclusion

The AVX2 i16-wrap shortcut (`wrapping_mul` + `wrapping_shl(2)`) restores the
PR #278 regression. C-libjpeg-turbo decode parity is maintained on graphic
content (2.5 ms / 2.7 ms = 0.93× of C, matching pre-PR278). Photo and
progressive decode unchanged.

The SSE2 + WASM shortcuts in this same commit don't show in the bench
(Comet Lake runtime dispatches AVX2), but their purpose is correctness for
the SSE2-only CI matrix and WASM target. Codex review of the first
restoration attempt caught a missing pass-2 `<< CONST_BITS` wrap in those
backends; both were fixed before this perf log was published.
