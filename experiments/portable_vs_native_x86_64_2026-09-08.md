# Portable vs native encode A/B — x86_64, 2026-09-08 (P4-133 / #464)

**Question.** How much encode speed does a stock `cargo build --release`
leave on the table against `-C target-cpu=native`, feature by feature, and
how does the portable build stand against C libjpeg-turbo? README had carried
"the last few percent (BMI2 PEXT/PDEP, FMA)" as a claim since P4-8 closed
(2026-05-17) without a number behind it.

**Method.** `.github/workflows/perf-portable-vs-native.yml`, one job, one
runner, everything sequential. `examples/bench_encode_matrix` is built seven
times into separate target directories:

| Variant | `RUSTFLAGS` |
| --- | --- |
| `stock` | unset — the portable build a package ships |
| `bmi1-lzcnt` | `-C target-feature=+bmi1,+lzcnt` |
| `bmi2` | `-C target-feature=+bmi2` |
| `fma` | `-C target-feature=+fma` |
| `readme-set` | `-C target-feature=+bmi1,+lzcnt,+bmi2,+fma` (what README recommended) |
| `avx2` | `-C target-feature=+avx2` |
| `native` | `-C target-cpu=native` |

Each binary runs the same eleven-case matrix (quality 75, RGB input, the
`tests/fixtures/photo_*` set) for the default integer DCT; the portable
build and the three variants that can move the float path (`fma`,
`readme-set`, `native`) run it again with `BENCH_DCT_METHOD=float`.
`examples/bench_c_encode_linux` is compiled against upstream's official
3.2.0 package (`/opt/libjpeg-turbo`, `ldd`-asserted) and run the same way.
The portable binary is timed first and again last (`stock-again`);
`scripts/perf_ab_summary.py` reports that pair's per-case spread as the
run's noise and marks any variant inside it `(noise)`.

There is no `-dct float` SIMD in this port on any backend, and the integer
default never touches the float FDCT, so the two tables answer different
questions: the integer table is the one a packager cares about; the float
table isolates the FMA question the issue named.

## Environment

| Item | Value |
| --- | --- |
| Host | GitHub-hosted `ubuntu-latest`, shared runner — governor and turbo cannot be pinned |
| CPU | AMD EPYC 7763 (Zen 3): `avx avx2 bmi1 bmi2 fma abm popcnt sse4_2` |
| C reference | libjpeg-turbo 3.2.0 (build 20260630), official `.deb` |
| Rust | rustc 1.98.1 (2026-09-01), default features (`simd`) |
| Runs | [34149692515](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/34149692515) at `d840042` (first harness; `v3-set` is `readme-set`); [34151304364](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/34151304364) at `9d295e8` on a different runner class, AMD EPYC 9V74 (Zen 4), same flags — the second sample below |

## Run 1 — integer DCT (the default)

Ratios are to the portable build; `C` is the reference's own ratio to
portable, so `1 / C` is the usual Rust-over-C figure. `noise` is
|stock-again / stock − 1|.

| Case | noise | bmi1-lzcnt | bmi2 | fma | readme-set | avx2 | native | stock-again | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64_420 | 0.5% | 0.995 (noise) | 0.984 | 0.984 | 0.995 (noise) | 0.968 | 0.947 | 0.995 | 1.212 |
| 320x240_420 | 2.5% | 0.971 | 0.962 | 0.974 | 0.982 (noise) | 0.973 | 0.968 | 0.975 | 0.948 |
| 320x240_422 | 0.2% | 0.982 | 0.974 | 0.998 | 0.957 | 0.988 | 0.985 | 0.998 | 0.945 |
| 320x240_444 | 1.2% | 0.986 | 0.979 | 0.988 | 0.963 | 0.992 (noise) | 0.979 | 0.988 | 1.000 |
| 640x480_420 | 0.1% | 0.997 | 0.987 | 1.000 (noise) | 0.997 | 0.997 | 0.992 | 0.999 | 0.973 |
| 640x480_422 | 0.6% | 0.984 | 0.971 | 1.000 (noise) | 0.959 | 0.985 | 0.994 | 0.994 | 0.950 |
| 640x480_444 | 1.2% | 0.986 | 0.975 | 0.990 (noise) | 0.966 | 0.989 (noise) | 0.980 | 0.988 | 0.984 |
| 1280x720_420 | 0.1% | 0.993 | 0.987 | 1.001 (noise) | 0.995 | 0.998 | 0.994 | 1.001 | 0.967 |
| 1920x1080_420 | 0.0% | 0.994 | 0.986 | 1.000 (noise) | 0.992 | 0.997 | 0.993 | 1.000 | 0.935 |
| 1920x1080_422 | 0.5% | 0.987 | 0.971 | 1.001 (noise) | 0.957 | 0.990 | 0.991 | 0.995 | 0.925 |
| 1920x1080_444 | 1.2% | 0.985 | 0.972 | 0.988 | 0.961 | 0.989 (noise) | 0.980 | 0.988 | 0.955 |

Absolute times (µs), portable build against C:

| Case | portable | C 3.2.0 | portable / C |
| --- | ---: | ---: | ---: |
| 320x240_420 | 302.7 | 287.0 | 1.05 |
| 320x240_422 | 381.9 | 360.9 | 1.06 |
| 320x240_444 | 565.6 | 565.8 | 1.00 |
| 640x480_420 | 1065.0 | 1036.6 | 1.03 |
| 640x480_422 | 1342.9 | 1275.3 | 1.05 |
| 640x480_444 | 1919.8 | 1889.6 | 1.02 |
| 1280x720_420 | 3618.8 | 3500.8 | 1.03 |
| 1920x1080_420 | 8170.8 | 7643.2 | 1.07 |
| 1920x1080_422 | 10470.5 | 9685.0 | 1.08 |
| 1920x1080_444 | 15495.8 | 14805.5 | 1.05 |

## Run 1 — float DCT (`-dct float`, not the default)

| Case | noise | fma | readme-set | native | stock-again | portable / C | fma / C | native / C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64_420 | 4.4% | 0.781 | 0.769 | 0.816 | 0.956 | 1.39 | 1.09 | 1.14 |
| 320x240_420 | 0.9% | 0.811 | 0.787 | 0.864 | 1.009 | 1.74 | 1.41 | 1.50 |
| 320x240_422 | 1.0% | 0.793 | 0.770 | 0.888 | 0.990 | 1.79 | 1.42 | 1.59 |
| 320x240_444 | 0.6% | 0.795 | 0.775 | 0.867 | 0.994 | 1.72 | 1.37 | 1.49 |
| 640x480_420 | 0.4% | 0.796 | 0.776 | 0.854 | 1.004 | 1.81 | 1.44 | 1.54 |
| 640x480_422 | 0.5% | 0.777 | 0.756 | 0.884 | 0.995 | 1.87 | 1.46 | 1.66 |
| 640x480_444 | 0.5% | 0.766 | 0.746 | 0.847 | 0.995 | 1.88 | 1.44 | 1.59 |
| 1280x720_420 | 0.0% | 0.813 | 0.786 | 0.865 | 1.000 | 1.73 | 1.40 | 1.49 |
| 1920x1080_420 | 0.5% | 0.808 | 0.783 | 0.861 | 0.995 | 1.79 | 1.45 | 1.54 |
| 1920x1080_422 | 0.5% | 0.799 | 0.776 | 0.897 | 1.005 | 1.80 | 1.44 | 1.62 |
| 1920x1080_444 | 0.2% | 0.796 | 0.776 | 0.871 | 0.998 | 1.77 | 1.41 | 1.54 |

## Run 2 — second sample, AMD EPYC 9V74 (Zen 4)

Same harness at `9d295e8` (this is the shape that merged: C is timed first
and last too). Ratios to the portable build; `C` and `C-again` are portable
divided by the reference.

Integer DCT:

| Case | portable µs | bmi1-lzcnt | bmi2 | fma | readme-set | avx2 | native | stock-again | C | C-again |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64_420 | 17.9 | 1.000 | 0.989 | 0.989 | 0.983 | 0.966 | 0.961 | 1.000 | 0.836 | 0.840 |
| 320x240_420 | 264.5 | 1.005 | 0.987 | 0.995 | 0.997 | 0.994 | 0.973 | 1.001 | 1.103 | 1.109 |
| 320x240_422 | 333.9 | 1.002 | 0.987 | 0.993 | 0.976 | 0.986 | 0.963 | 0.995 | 1.128 | 1.126 |
| 320x240_444 | 490.3 | 0.995 | 0.989 | 0.984 | 0.970 | 0.974 | 0.961 | 0.991 | 1.141 | 1.148 |
| 640x480_420 | 1024.0 | 1.022 | 0.987 | 0.997 | 0.991 | 0.998 | 0.986 | 1.000 | 1.092 | 1.106 |
| 640x480_422 | 1298.4 | 1.001 | 0.992 | 0.992 | 0.977 | 0.987 | 0.964 | 0.994 | 1.124 | 1.127 |
| 640x480_444 | 1898.5 | 1.003 | 0.990 | 0.985 | 0.978 | 0.984 | 0.970 | 0.991 | 1.092 | 1.098 |
| 1280x720_420 | 3644.9 | 1.000 | 0.973 | 0.989 | 0.976 | 0.991 | 0.973 | 0.989 | 1.109 | 1.115 |
| 1920x1080_420 | 8191.2 | 1.007 | 0.983 | 0.998 | 0.983 | 0.997 | 0.978 | 0.998 | 1.100 | 1.103 |
| 1920x1080_422 | 10344.7 | 1.000 | 0.987 | 0.993 | 0.979 | 0.989 | 0.972 | 0.996 | 1.103 | 1.110 |
| 1920x1080_444 | 15521.4 | 0.993 | 0.981 | 0.983 | 0.973 | 0.984 | 0.965 | 0.989 | 1.088 | 1.103 |

Float DCT:

| Case | portable µs | fma | readme-set | native | stock-again | C | C-again |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64_420 | 33.0 | 0.742 | 0.745 | 0.767 | 0.985 | 1.381 | 1.441 |
| 320x240_420 | 499.2 | 0.800 | 0.804 | 0.845 | 1.001 | 1.807 | 1.798 |
| 320x240_422 | 654.7 | 0.794 | 0.781 | 0.831 | 0.980 | 1.886 | 1.882 |
| 320x240_444 | 981.2 | 0.792 | 0.781 | 0.826 | 0.985 | 1.937 | 1.929 |
| 640x480_420 | 1996.2 | 0.798 | 0.800 | 0.843 | 1.003 | 1.837 | 1.831 |
| 640x480_422 | 2569.1 | 0.796 | 0.785 | 0.831 | 0.981 | 1.888 | 1.879 |
| 640x480_444 | 3778.8 | 0.788 | 0.784 | 0.832 | 0.988 | 1.863 | 1.857 |
| 1280x720_420 | 6493.8 | 0.815 | 0.812 | 0.860 | 0.995 | 1.737 | 1.731 |
| 1920x1080_420 | 14696.5 | 0.815 | 0.815 | 0.867 | 0.996 | 1.736 | 1.732 |
| 1920x1080_422 | 18899.3 | 0.813 | 0.802 | 0.848 | 0.993 | 1.771 | 1.768 |
| 1920x1080_444 | 27876.6 | 0.816 | 0.809 | 0.856 | 0.997 | 1.729 | 1.734 |

## Agreement across the two samples

Two runner classes, two microarchitectures, the same shape:

| Quantity (1080p unless noted) | Zen 3 (run 1) | Zen 4 (run 2) | Quoted range |
| --- | --- | --- | --- |
| noise, stock vs stock-again (all cases) | 0–2.5 % | 0–1.1 % | 0–2.5 % |
| C vs C-again (all cases) | — | 0.2–1.5 % | — |
| `native` / portable | 0.7–2.0 % faster | 2.2–3.5 % faster | 0.7–3.5 % |
| `bmi2` / portable | 1.4–2.9 % | 1.3–1.9 % | 1.3–2.9 % |
| `bmi1-lzcnt` / portable | 0.6–1.5 % | −0.7–0.7 % (noise) | ≤ 1.5 % |
| `fma` / portable, integer | noise | 0.2–1.7 % | noise to 1.7 % |
| `avx2` / portable | 0.3–1.1 % | 0.3–1.6 % | ≤ 1.6 % |
| `readme-set` / portable | 0.8–4.3 % | 1.7–2.7 % | 0.8–4.3 % |
| portable / C | 1.05–1.08× | 1.09–1.10× | 1.05–1.10× |
| `native` / C | 1.03–1.07× | 1.05–1.08× | 1.03–1.08× |
| float: `fma` / portable | 19–23 % faster | 18–21 % faster | 18–23 % (from 320×240 up) |
| float: portable / C | 1.72–1.88× | 1.73–1.94× | 1.72–1.94× |
| float: `fma` / C | 1.37–1.46× | 1.41–1.53× | 1.37–1.53× |
| float: `native` / `fma` | native slower | native slower | dispatch the feature, not the flag |

Zen 4 shows a wider portable-vs-C gap and a larger native win than Zen 3;
the ordering of the variants is the same on both, and every conclusion in
the Decisions table holds on either sample alone.

## What run 1 says, in detail

The figures below are run 1's (Zen 3); the agreement table above carries
the two-sample ranges the tracking docs quote.

**Integer DCT, the path every default encode takes.**

- The run's own noise is 0–2.5 % per case, mostly under 1.2 %.
- `native` is worth **0.7–2.0 % at 1080p** (0.993 / 0.991 / 0.980), 5 % at
  64×64 where fixed costs dominate. That is the whole "native-only" gap for
  the default path on this CPU.
- `bmi2` alone accounts for most of it: **1.4–2.9 % at 1080p**, outside the
  noise bracket in every 1080p case. Nothing in this port uses PEXT/PDEP;
  the win is the compiler using `SHLX`/`SHRX`/`SARX` for the variable shifts
  in the Huffman bit packer, which the `#[target_feature(enable =
  "bmi1,lzcnt")]` variant does not enable.
- `bmi1-lzcnt` at compile time is worth a further 0.6–1.5 % at 1080p over
  the runtime dispatch — the per-block branch and call P4-8's closing row
  already priced at "about 80 % of the theoretical max".
- `fma` is **noise** on the integer path, as expected: it never touches the
  float FDCT.
- `avx2` at compile time is ≤ 1.1 % at 1080p (3.2 % at 64×64) and inside
  noise in three cases: the explicit AVX2 kernels are already reached by
  runtime dispatch, so auto-vectorising the residue buys almost nothing.
- `readme-set` (`+bmi1,+lzcnt,+bmi2,+fma`) is 0.8–4.3 % at 1080p — and it
  is **not a portable build**: it faults on any CPU without BMI2/FMA, so a
  packager cannot use it, and README no longer suggests that it can.
- **Portable vs C 3.2.0: 1.00–1.08×** from 320×240 up (0.83× at 64×64),
  1.05–1.08× at 1080p. `native` narrows that to 1.03–1.07× and `readme-set`
  to 1.01–1.06×; neither beats C on this Zen 3 box. The 2026-05 i5-10400
  native table in README, which did beat C 3.1.2 by 2–7 %, is a different
  CPU, a different upstream release and a pinned governor; it stays as the
  supplementary figure.

**Float DCT.**

- Compile-time FMA is worth **19–23 %** of the whole encode (0.77–0.81):
  `fdct_float_workspace` uses `f32::mul_add` for parity with clang's
  contraction, and a baseline x86_64 build lowers that to a libm `fmaf`
  call per rotator — 2 per 1-D transform, 16 transforms per block.
- `native` is *slower* than `fma` alone here (0.82–0.90 vs 0.77–0.81);
  `-C target-cpu=znver3` tunes more than it enables, and the float path
  loses some of it. Another reason to dispatch the feature rather than
  recommend the flag.
- Even with FMA the port trails C by **1.37–1.46×** from 320×240 up
  (1.09× at 64×64): upstream runs the float FDCT and quantiser through
  SSE/SSE2 (`jfdctflt-sse.asm` plus `jquantf-sse2.asm`, which also holds
  `jsimd_convsamp_float`), this port runs them scalar on every backend.
  That is a separate gap, filed as P4-187.

## Decisions

| Candidate | Measured | Decision |
| --- | --- | --- |
| BMI2 in the Huffman loop | 1.3–2.9 % at 1080p, integer path, both samples | **Pays.** Add `bmi2` to the elevated Huffman variant's `target_feature` set behind `cpu_has!`, resolved once per plan (P4-133 remainder, coordinate with P4-123 workstream 2). |
| FMA in the float FDCT | 18–23 % of `-dct float` encode, both samples | **Pays.** A `#[target_feature(enable = "fma")]` twin of `scalar_fdct_float_quantize`, selected where `DctMethod::Float` already picks the kernel at plan build (six `DctMethod::Float =>` sites). Bit-exact: hardware FMA and libm `fmaf` both round once. |
| FMA on the integer path | noise | Nothing to do. |
| Compile-time BMI1/LZCNT | 0.6–1.5 % at 1080p over runtime dispatch | Already priced by P4-8; not worth a second mechanism. |
| Compile-time AVX2 | ≤ 1.1 % at 1080p, often noise | Nothing to do; the kernels are dispatched. |
| PEXT/PDEP | not used anywhere | Nothing to measure; the P4-8 follow-up's wording was a guess. |
| Float FDCT/quantise SIMD | 1.4× behind C even with FMA | Out of P4-133's scope; **P4-187**. |

`experiments/encode.tsv` carries the summary row; README's performance
section now quotes the portable build from this run and marks the i5-10400
native table as supplementary.

## Appendix — aarch64 smoke of the float bench (not a measurement)

One run each, Apple M-series host (`aarch64-apple-darwin`), homebrew
`jpeg-turbo` for C, no clock pinning, taken only to check the two benches
parse before the x86_64 job existed. Recorded because P4-187 cites it; the
proper aarch64 number is the P4-187 job's to produce.

| Case | Rust float (µs) | C float (µs) | Rust / C |
| --- | ---: | ---: | ---: |
| 320x240_420 | 410.1 | 265.0 | 1.55 |
| 320x240_422 | 484.1 | 338.5 | 1.43 |
| 320x240_444 | 528.3 | 512.9 | 1.03 |
| 640x480_420 | 1543.5 | 984.0 | 1.57 |
| 640x480_422 | 1792.2 | 1241.8 | 1.44 |
| 640x480_444 | 1885.0 | 1819.1 | 1.04 |
| 1280x720_420 | 4957.6 | 3234.6 | 1.53 |
| 1920x1080_420 | 11148.8 | 7281.4 | 1.53 |
| 1920x1080_422 | 13011.1 | 9333.5 | 1.39 |
| 1920x1080_444 | 14460.1 | 14086.4 | 1.03 |
