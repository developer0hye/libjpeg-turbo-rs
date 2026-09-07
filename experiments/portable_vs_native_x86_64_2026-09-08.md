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

**Follow-up (2026-09-08, PR #602).** Both "Pays" rows landed: a third
`bmi1,lzcnt,bmi2` compilation of the Huffman AC loop, and an FMA-compiled
twin of the float FDCT + quantise kernel installed in `EncoderSimdRoutines`,
each reached from a portable build by `cpu_has!`. Everything measured above
therefore describes the build **before** that dispatch; the re-measurement is
recorded in P4-133's second-milestone entry in `docs/last_mile/phase4.md`.

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

## Run 3 — branch `perf/p4-133-runtime-dispatch` (`ac2c26b`), the dispatch landed

[Run 34157974332](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/34157974332),
AMD EPYC 9V74 (Zen 4), rustc 1.98.1, C libjpeg-turbo 3.2.0 official deb.
Same harness as runs 1–2; `stock` is the branch's plain `cargo build
--release`, now with the BMI2 Huffman tier and the FMA float FDCT reached by
runtime detection. There is no same-run `main` column in this run (added to
the harness afterwards), so integer-DCT deltas against runs 1–2 are
cross-host comparisons.

What it shows:

- **Float DCT: the FMA twin is reached.** `fma / baseline` is 0.982–0.991
  everywhere (1–2 %), where run 2 on the same CPU model had 18–21 %; the
  portable build is 1.39–1.41× C at 1080p (run 2 portable: 1.72–1.94×; run 2
  `+fma`: 1.41–1.53×). `native` and the README set are now 4–9 % *slower*
  than portable on this path.
- **Integer DCT: inconclusive across runs.** `bmi2 / baseline` at 1080p is
  0.978 / 0.996 / 0.990 (noise 0.4–0.6 %) against run 2's 1.3–1.9 %, and
  portable / C is 1.115 / 1.104 / 1.097 against run 2's 1.09–1.10. Those
  differences are the size two hosts differ by; the same-run `main-portable`
  variant exists to answer this.

### Integer DCT (islow) — µs, ratio to portable build, ratio to C 3.2.0

| Case | noise (pair spread) | stock (µs) | stock / baseline | stock / reference | bmi1-lzcnt (µs) | bmi1-lzcnt / baseline | bmi1-lzcnt / reference | bmi2 (µs) | bmi2 / baseline | bmi2 / reference | fma (µs) | fma / baseline | fma / reference | readme-set (µs) | readme-set / baseline | readme-set / reference | avx2 (µs) | avx2 / baseline | avx2 / reference | native (µs) | native / baseline | native / reference | stock-again (µs) | stock-again / baseline | stock-again / reference | C (µs) | C / baseline | C / reference | C-again (µs) | C-again / baseline | C-again / reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x64_420 | 0.0% | 17.7 | 1.000 (noise) | 0.816 | 17.8 | 1.006 | 0.820 | 17.7 | 1.000 (noise) | 0.816 | 17.7 | 1.000 (noise) | 0.816 | 17.6 | 0.994 | 0.811 | 17.0 | 0.960 | 0.783 | 17.0 | 0.960 | 0.783 | 17.7 | 1.000 (noise) | 0.816 | 21.7 | 1.226 | 1.000 | 21.2 | 1.198 | 0.977 |
| 320x240_420 | 0.2% | 261.9 | 1.000 (noise) | 1.096 | 261.5 | 0.998 (noise) | 1.095 | 260.5 | 0.995 | 1.090 | 260.6 | 0.995 | 1.091 | 260.5 | 0.995 | 1.090 | 259.2 | 0.990 | 1.085 | 259.5 | 0.991 | 1.086 | 261.5 | 0.998 (noise) | 1.095 | 238.9 | 0.912 | 1.000 | 237.8 | 0.908 | 0.995 |
| 320x240_422 | 0.5% | 331.8 | 1.000 (noise) | 1.117 | 332.3 | 1.002 (noise) | 1.118 | 333.8 | 1.006 | 1.124 | 326.4 | 0.984 | 1.099 | 325.6 | 0.981 | 1.096 | 325.9 | 0.982 | 1.097 | 321.4 | 0.969 | 1.082 | 330.0 | 0.995 (noise) | 1.111 | 297.1 | 0.895 | 1.000 | 295.2 | 0.890 | 0.994 |
| 320x240_444 | 0.4% | 484.1 | 1.000 (noise) | 1.121 | 483.8 | 0.999 (noise) | 1.120 | 482.4 | 0.996 (noise) | 1.117 | 476.2 | 0.984 | 1.103 | 474.0 | 0.979 | 1.098 | 474.2 | 0.980 | 1.098 | 468.6 | 0.968 | 1.085 | 482.3 | 0.996 (noise) | 1.117 | 431.8 | 0.892 | 1.000 | 429.3 | 0.887 | 0.994 |
| 640x480_420 | 0.2% | 1034.0 | 1.000 (noise) | 1.106 | 1023.3 | 0.990 | 1.094 | 1012.2 | 0.979 | 1.082 | 1012.4 | 0.979 | 1.083 | 1022.5 | 0.989 | 1.093 | 1014.5 | 0.981 | 1.085 | 997.8 | 0.965 | 1.067 | 1032.0 | 0.998 (noise) | 1.104 | 935.1 | 0.904 | 1.000 | 923.1 | 0.893 | 0.987 |
| 640x480_422 | 0.0% | 1291.6 | 1.000 (noise) | 1.098 | 1291.5 | 1.000 (noise) | 1.098 | 1285.8 | 0.996 | 1.093 | 1271.0 | 0.984 | 1.080 | 1266.3 | 0.980 | 1.076 | 1281.1 | 0.992 | 1.089 | 1267.2 | 0.981 | 1.077 | 1291.9 | 1.000 (noise) | 1.098 | 1176.6 | 0.911 | 1.000 | 1148.0 | 0.889 | 0.976 |
| 640x480_444 | 0.4% | 1884.9 | 1.000 (noise) | 1.067 | 1876.1 | 0.995 | 1.062 | 1881.7 | 0.998 (noise) | 1.066 | 1858.9 | 0.986 | 1.053 | 1852.4 | 0.983 | 1.049 | 1864.3 | 0.989 | 1.056 | 1851.1 | 0.982 | 1.048 | 1877.4 | 0.996 (noise) | 1.063 | 1766.0 | 0.937 | 1.000 | 1731.4 | 0.919 | 0.980 |
| 1280x720_420 | 0.2% | 3645.5 | 1.000 (noise) | 1.112 | 3582.0 | 0.983 | 1.092 | 3557.6 | 0.976 | 1.085 | 3570.4 | 0.979 | 1.089 | 3560.1 | 0.977 | 1.086 | 3590.6 | 0.985 | 1.095 | 3550.1 | 0.974 | 1.082 | 3639.2 | 0.998 (noise) | 1.110 | 3279.6 | 0.900 | 1.000 | 3265.9 | 0.896 | 0.996 |
| 1920x1080_420 | 0.6% | 8275.8 | 1.000 (noise) | 1.115 | 8111.6 | 0.980 | 1.093 | 8095.6 | 0.978 | 1.090 | 8075.1 | 0.976 | 1.088 | 8054.7 | 0.973 | 1.085 | 8114.1 | 0.980 | 1.093 | 8039.5 | 0.971 | 1.083 | 8229.0 | 0.994 (noise) | 1.108 | 7424.4 | 0.897 | 1.000 | 7424.7 | 0.897 | 1.000 |
| 1920x1080_422 | 0.4% | 10275.4 | 1.000 (noise) | 1.104 | 10239.8 | 0.997 (noise) | 1.100 | 10229.9 | 0.996 | 1.099 | 10159.9 | 0.989 | 1.091 | 10139.7 | 0.987 | 1.089 | 10202.3 | 0.993 | 1.096 | 10075.0 | 0.980 | 1.082 | 10234.1 | 0.996 (noise) | 1.099 | 9309.3 | 0.906 | 1.000 | 9272.4 | 0.902 | 0.996 |
| 1920x1080_444 | 0.5% | 15399.1 | 1.000 (noise) | 1.097 | 15232.6 | 0.989 | 1.085 | 15251.6 | 0.990 | 1.086 | 15155.9 | 0.984 | 1.080 | 15068.8 | 0.979 | 1.073 | 15191.9 | 0.987 | 1.082 | 14984.9 | 0.973 | 1.067 | 15321.6 | 0.995 (noise) | 1.091 | 14037.9 | 0.912 | 1.000 | 14043.6 | 0.912 | 1.000 |

### Float DCT — µs, ratio to portable build, ratio to C 3.2.0

| Case | noise (pair spread) | stock (µs) | stock / baseline | stock / reference | fma (µs) | fma / baseline | fma / reference | readme-set (µs) | readme-set / baseline | readme-set / reference | native (µs) | native / baseline | native / reference | stock-again (µs) | stock-again / baseline | stock-again / reference | C (µs) | C / baseline | C / reference | C-again (µs) | C-again / baseline | C-again / reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x64_420 | 0.4% | 24.8 | 1.000 (noise) | 1.055 | 24.5 | 0.988 | 1.043 | 27.1 | 1.093 | 1.153 | 25.3 | 1.020 | 1.077 | 24.9 | 1.004 (noise) | 1.060 | 23.5 | 0.948 | 1.000 | 26.5 | 1.069 | 1.128 |
| 320x240_420 | 0.3% | 402.8 | 1.000 (noise) | 1.446 | 397.5 | 0.987 | 1.427 | 444.1 | 1.103 | 1.595 | 421.9 | 1.047 | 1.515 | 401.4 | 0.997 (noise) | 1.441 | 278.5 | 0.691 | 1.000 | 298.7 | 0.742 | 1.073 |
| 320x240_422 | 0.0% | 515.0 | 1.000 (noise) | 1.478 | 509.6 | 0.990 | 1.463 | 544.1 | 1.057 | 1.562 | 538.6 | 1.046 | 1.546 | 514.9 | 1.000 (noise) | 1.478 | 348.4 | 0.677 | 1.000 | 353.6 | 0.687 | 1.015 |
| 320x240_444 | 0.6% | 767.5 | 1.000 (noise) | 1.508 | 757.4 | 0.987 | 1.488 | 834.5 | 1.087 | 1.640 | 809.7 | 1.055 | 1.591 | 763.2 | 0.994 (noise) | 1.500 | 508.9 | 0.663 | 1.000 | 516.1 | 0.672 | 1.014 |
| 640x480_420 | 0.4% | 1605.7 | 1.000 (noise) | 1.461 | 1577.8 | 0.983 | 1.435 | 1768.4 | 1.101 | 1.609 | 1678.9 | 1.046 | 1.527 | 1599.4 | 0.996 (noise) | 1.455 | 1099.2 | 0.685 | 1.000 | 1168.4 | 0.728 | 1.063 |
| 640x480_422 | 0.0% | 2028.0 | 1.000 (noise) | 1.464 | 2005.6 | 0.989 | 1.447 | 2146.3 | 1.058 | 1.549 | 2125.2 | 1.048 | 1.534 | 2028.7 | 1.000 (noise) | 1.464 | 1385.7 | 0.683 | 1.000 | 1472.4 | 0.726 | 1.063 |
| 640x480_444 | 0.6% | 2960.0 | 1.000 (noise) | 1.426 | 2933.6 | 0.991 | 1.413 | 3237.1 | 1.094 | 1.560 | 3140.1 | 1.061 | 1.513 | 2941.6 | 0.994 (noise) | 1.417 | 2075.6 | 0.701 | 1.000 | 2230.4 | 0.754 | 1.075 |
| 1280x720_420 | 0.4% | 5324.1 | 1.000 (noise) | 1.408 | 5226.5 | 0.982 | 1.382 | 5792.7 | 1.088 | 1.532 | 5536.7 | 1.040 | 1.464 | 5304.9 | 0.996 (noise) | 1.403 | 3781.3 | 0.710 | 1.000 | 3923.5 | 0.737 | 1.038 |
| 1920x1080_420 | 0.2% | 12065.0 | 1.000 (noise) | 1.413 | 11856.1 | 0.983 | 1.389 | 13123.1 | 1.088 | 1.537 | 12566.9 | 1.042 | 1.472 | 12037.3 | 0.998 (noise) | 1.410 | 8538.6 | 0.708 | 1.000 | 8857.3 | 0.734 | 1.037 |
| 1920x1080_422 | 0.0% | 15253.8 | 1.000 (noise) | 1.410 | 15097.2 | 0.990 | 1.395 | 16040.7 | 1.052 | 1.482 | 15895.2 | 1.042 | 1.469 | 15252.5 | 1.000 (noise) | 1.410 | 10821.0 | 0.709 | 1.000 | 11323.3 | 0.742 | 1.046 |
| 1920x1080_444 | 0.6% | 22562.4 | 1.000 (noise) | 1.385 | 22337.4 | 0.990 | 1.371 | 24428.3 | 1.083 | 1.500 | 23755.7 | 1.053 | 1.458 | 22437.3 | 0.994 (noise) | 1.378 | 16288.0 | 0.722 | 1.000 | 17340.3 | 0.769 | 1.065 |

### Runner

```
abm
avx
avx2
bmi1
bmi2
fma
popcnt
sse4_2
Model name:                              AMD EPYC 9V74 80-Core Processor
rustc 1.98.1 (48a229cea 2026-09-01)
libjpeg-turbo version 3.2.0 (build 20260630)
```

## Run 4 — same-run `main` vs branch (`69bd3a7`), harness re-run on PR #602

[Run 34159737624](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/34159737624),
Intel Xeon 6973P-C (AVX-512 host), rustc 1.98.1, C 3.2.0 official deb. First
run with the `main-portable` column: the plain `cargo build --release` of
`main` at `34b95ef`, built from a worktree in the same job and timed right
after `stock`. Noise bracket 0.3–11.7 % (integer), 0.3–7.3 % (float) — a
much noisier host than runs 1–3.

What it shows:

- **Float DCT, settled.** `main-portable / baseline` is 1.116–1.268 on every
  case: `main` is 12–27 % slower than the branch, in the same direction on all
  eleven rows and above the bracket on nine. `fma / baseline` is 0.909–1.041,
  i.e. compile-time FMA adds nothing the dispatch does not already reach.
  `native` is 12–33 % slower than portable on this path.
- **Integer DCT, not settled by this host.** `main-portable / baseline` at
  1080p is 1.029 / 1.069 / 1.089, in the branch's favour — but the
  compile-time `+bmi2` binary, timed between them, came out 1.089 at
  1080p 4:2:2 against a 0.5 % bracket, which a BMI2-only difference cannot
  produce. The machine drifted more than its first/last bracket shows, so
  this run says nothing at the 1–3 % level. A quiet Zen runner with this
  column is the measurement that would.

### Integer DCT (islow) — µs, ratio to portable build, ratio to C 3.2.0

| Case | noise (pair spread) | stock (µs) | stock / baseline | stock / reference | main-portable (µs) | main-portable / baseline | main-portable / reference | bmi1-lzcnt (µs) | bmi1-lzcnt / baseline | bmi1-lzcnt / reference | bmi2 (µs) | bmi2 / baseline | bmi2 / reference | fma (µs) | fma / baseline | fma / reference | readme-set (µs) | readme-set / baseline | readme-set / reference | avx2 (µs) | avx2 / baseline | avx2 / reference | native (µs) | native / baseline | native / reference | stock-again (µs) | stock-again / baseline | stock-again / reference | C (µs) | C / baseline | C / reference | C-again (µs) | C-again / baseline | C-again / reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x64_420 | 2.4% | 12.5 | 1.000 (noise) | 0.749 | 12.5 | 1.000 (noise) | 0.749 | 12.0 | 0.960 | 0.719 | 12.2 | 0.976 (noise) | 0.731 | 12.1 | 0.968 | 0.725 | 12.2 | 0.976 (noise) | 0.731 | 13.8 | 1.104 | 0.826 | 14.9 | 1.192 | 0.892 | 12.2 | 0.976 (noise) | 0.731 | 16.7 | 1.336 | 1.000 | 16.6 | 1.328 | 0.994 |
| 320x240_420 | 4.9% | 214.7 | 1.000 (noise) | 1.039 | 204.1 | 0.951 | 0.988 | 199.3 | 0.928 | 0.965 | 199.5 | 0.929 | 0.966 | 201.1 | 0.937 | 0.973 | 205.4 | 0.957 (noise) | 0.994 | 216.0 | 1.006 (noise) | 1.045 | 252.4 | 1.176 | 1.222 | 204.2 | 0.951 (noise) | 0.988 | 206.6 | 0.962 (noise) | 1.000 | 214.8 | 1.000 (noise) | 1.040 |
| 320x240_422 | 2.5% | 263.2 | 1.000 (noise) | 0.968 | 276.6 | 1.051 | 1.017 | 257.2 | 0.977 (noise) | 0.946 | 283.9 | 1.079 | 1.044 | 259.9 | 0.987 (noise) | 0.956 | 259.1 | 0.984 (noise) | 0.953 | 253.6 | 0.964 | 0.933 | 314.8 | 1.196 | 1.158 | 269.7 | 1.025 (noise) | 0.992 | 271.9 | 1.033 | 1.000 | 272.1 | 1.034 | 1.001 |
| 320x240_444 | 6.8% | 412.9 | 1.000 (noise) | 1.057 | 404.8 | 0.980 (noise) | 1.036 | 403.9 | 0.978 (noise) | 1.034 | 430.7 | 1.043 (noise) | 1.103 | 386.8 | 0.937 (noise) | 0.990 | 387.8 | 0.939 (noise) | 0.993 | 385.8 | 0.934 (noise) | 0.988 | 482.9 | 1.170 | 1.236 | 440.8 | 1.068 (noise) | 1.129 | 390.6 | 0.946 (noise) | 1.000 | 427.8 | 1.036 (noise) | 1.095 |
| 640x480_420 | 1.7% | 774.8 | 1.000 (noise) | 1.044 | 859.3 | 1.109 | 1.157 | 768.6 | 0.992 (noise) | 1.035 | 759.0 | 0.980 | 1.022 | 758.0 | 0.978 | 1.021 | 767.2 | 0.990 (noise) | 1.033 | 777.1 | 1.003 (noise) | 1.047 | 909.1 | 1.173 | 1.224 | 761.6 | 0.983 (noise) | 1.026 | 742.5 | 0.958 | 1.000 | 726.4 | 0.938 | 0.978 |
| 640x480_422 | 4.7% | 967.6 | 1.000 (noise) | 1.084 | 963.6 | 0.996 (noise) | 1.079 | 950.4 | 0.982 (noise) | 1.065 | 944.4 | 0.976 (noise) | 1.058 | 943.7 | 0.975 (noise) | 1.057 | 972.2 | 1.005 (noise) | 1.089 | 955.1 | 0.987 (noise) | 1.070 | 1167.8 | 1.207 | 1.308 | 1012.8 | 1.047 (noise) | 1.134 | 892.8 | 0.923 | 1.000 | 940.5 | 0.972 (noise) | 1.053 |
| 640x480_444 | 4.1% | 1445.9 | 1.000 (noise) | 1.088 | 1466.3 | 1.014 (noise) | 1.103 | 1412.6 | 0.977 (noise) | 1.063 | 1380.0 | 0.954 | 1.038 | 1429.0 | 0.988 (noise) | 1.075 | 1353.1 | 0.936 | 1.018 | 1368.3 | 0.946 | 1.030 | 1659.4 | 1.148 | 1.249 | 1387.0 | 0.959 (noise) | 1.044 | 1328.9 | 0.919 | 1.000 | 1294.5 | 0.895 | 0.974 |
| 1280x720_420 | 2.6% | 2710.5 | 1.000 (noise) | 1.052 | 2776.4 | 1.024 (noise) | 1.078 | 2643.4 | 0.975 (noise) | 1.026 | 2624.2 | 0.968 | 1.018 | 2803.4 | 1.034 | 1.088 | 2630.6 | 0.971 | 1.021 | 2674.8 | 0.987 (noise) | 1.038 | 3079.5 | 1.136 | 1.195 | 2641.1 | 0.974 (noise) | 1.025 | 2576.6 | 0.951 | 1.000 | 2566.7 | 0.947 | 0.996 |
| 1920x1080_420 | 11.7% | 5953.9 | 1.000 (noise) | 1.032 | 6123.9 | 1.029 (noise) | 1.061 | 6040.5 | 1.015 (noise) | 1.047 | 5935.1 | 0.997 (noise) | 1.029 | 6164.7 | 1.035 (noise) | 1.069 | 5939.9 | 0.998 (noise) | 1.030 | 6043.5 | 1.015 (noise) | 1.048 | 7542.5 | 1.267 | 1.307 | 6648.8 | 1.117 (noise) | 1.152 | 5769.3 | 0.969 (noise) | 1.000 | 5725.5 | 0.962 (noise) | 0.992 |
| 1920x1080_422 | 0.5% | 7435.4 | 1.000 (noise) | 1.017 | 7951.6 | 1.069 | 1.087 | 7831.3 | 1.053 | 1.071 | 8094.0 | 1.089 | 1.107 | 7526.8 | 1.012 | 1.029 | 7519.5 | 1.011 | 1.028 | 7868.5 | 1.058 | 1.076 | 8900.6 | 1.197 | 1.217 | 7474.0 | 1.005 (noise) | 1.022 | 7313.0 | 0.984 | 1.000 | 7431.4 | 0.999 (noise) | 1.016 |
| 1920x1080_444 | 1.0% | 11079.1 | 1.000 (noise) | 1.004 | 12069.5 | 1.089 | 1.093 | 11306.2 | 1.020 | 1.024 | 11087.1 | 1.001 (noise) | 1.004 | 11095.5 | 1.001 (noise) | 1.005 | 11114.5 | 1.003 (noise) | 1.007 | 10954.0 | 0.989 | 0.992 | 13094.9 | 1.182 | 1.186 | 11187.7 | 1.010 (noise) | 1.013 | 11039.0 | 0.996 (noise) | 1.000 | 11034.3 | 0.996 (noise) | 1.000 |

### Float DCT — µs, ratio to portable build, ratio to C 3.2.0

| Case | noise (pair spread) | stock (µs) | stock / baseline | stock / reference | main-portable (µs) | main-portable / baseline | main-portable / reference | fma (µs) | fma / baseline | fma / reference | readme-set (µs) | readme-set / baseline | readme-set / reference | native (µs) | native / baseline | native / reference | stock-again (µs) | stock-again / baseline | stock-again / reference | C (µs) | C / baseline | C / reference | C-again (µs) | C-again / baseline | C-again / reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x64_420 | 0.6% | 17.5 | 1.000 (noise) | 1.000 | 21.5 | 1.229 | 1.229 | 18.1 | 1.034 | 1.034 | 17.3 | 0.989 | 0.989 | 23.3 | 1.331 | 1.331 | 17.4 | 0.994 (noise) | 0.994 | 17.5 | 1.000 (noise) | 1.000 | 17.4 | 0.994 (noise) | 0.994 |
| 320x240_420 | 2.9% | 301.0 | 1.000 (noise) | 1.306 | 373.2 | 1.240 | 1.620 | 307.4 | 1.021 (noise) | 1.334 | 323.2 | 1.074 | 1.403 | 375.4 | 1.247 | 1.629 | 309.7 | 1.029 (noise) | 1.344 | 230.4 | 0.765 | 1.000 | 230.3 | 0.765 | 1.000 |
| 320x240_422 | 0.3% | 431.8 | 1.000 (noise) | 1.469 | 482.1 | 1.116 | 1.640 | 392.4 | 0.909 | 1.335 | 405.3 | 0.939 | 1.379 | 484.0 | 1.121 | 1.646 | 430.5 | 0.997 (noise) | 1.464 | 294.0 | 0.681 | 1.000 | 333.0 | 0.771 | 1.133 |
| 320x240_444 | 7.2% | 571.1 | 1.000 (noise) | 1.307 | 712.0 | 1.247 | 1.629 | 594.5 | 1.041 (noise) | 1.360 | 585.2 | 1.025 (noise) | 1.339 | 736.5 | 1.290 | 1.685 | 612.5 | 1.072 (noise) | 1.401 | 437.1 | 0.765 | 1.000 | 455.0 | 0.797 | 1.041 |
| 640x480_420 | 0.3% | 1154.7 | 1.000 (noise) | 1.391 | 1450.2 | 1.256 | 1.746 | 1188.2 | 1.029 | 1.431 | 1176.0 | 1.018 | 1.416 | 1477.7 | 1.280 | 1.780 | 1150.7 | 0.997 (noise) | 1.386 | 830.4 | 0.719 | 1.000 | 817.7 | 0.708 | 0.985 |
| 640x480_422 | 2.2% | 1574.3 | 1.000 (noise) | 1.547 | 1919.4 | 1.219 | 1.886 | 1467.2 | 0.932 | 1.442 | 1549.5 | 0.984 (noise) | 1.523 | 1870.7 | 1.188 | 1.839 | 1539.1 | 0.978 (noise) | 1.513 | 1017.5 | 0.646 | 1.000 | 1019.7 | 0.648 | 1.002 |
| 640x480_444 | 0.4% | 2088.7 | 1.000 (noise) | 1.421 | 2648.7 | 1.268 | 1.802 | 2171.7 | 1.040 | 1.478 | 2128.4 | 1.019 | 1.448 | 2722.1 | 1.303 | 1.852 | 2081.1 | 0.996 (noise) | 1.416 | 1469.6 | 0.704 | 1.000 | 1538.4 | 0.737 | 1.047 |
| 1280x720_420 | 4.7% | 3919.4 | 1.000 (noise) | 1.279 | 4795.0 | 1.223 | 1.564 | 3855.5 | 0.984 (noise) | 1.258 | 3831.5 | 0.978 (noise) | 1.250 | 4690.7 | 1.197 | 1.530 | 3735.5 | 0.953 (noise) | 1.219 | 3065.5 | 0.782 | 1.000 | 2923.4 | 0.746 | 0.954 |
| 1920x1080_420 | 1.9% | 8646.3 | 1.000 (noise) | 1.221 | 10824.3 | 1.252 | 1.529 | 8727.6 | 1.009 (noise) | 1.233 | 8973.3 | 1.038 | 1.268 | 10846.2 | 1.254 | 1.532 | 8486.0 | 0.981 (noise) | 1.199 | 7078.8 | 0.819 | 1.000 | 6554.4 | 0.758 | 0.926 |
| 1920x1080_422 | 7.3% | 11610.3 | 1.000 (noise) | 1.410 | 13635.3 | 1.174 | 1.656 | 11392.4 | 0.981 (noise) | 1.384 | 11167.1 | 0.962 (noise) | 1.356 | 13401.0 | 1.154 | 1.628 | 10758.6 | 0.927 (noise) | 1.307 | 8233.1 | 0.709 | 1.000 | 8232.2 | 0.709 | 1.000 |
| 1920x1080_444 | 3.1% | 16304.0 | 1.000 (noise) | 1.292 | 20101.6 | 1.233 | 1.593 | 16457.0 | 1.009 (noise) | 1.304 | 17108.8 | 1.049 | 1.356 | 20395.0 | 1.251 | 1.616 | 15792.3 | 0.969 (noise) | 1.251 | 12621.0 | 0.774 | 1.000 | 12600.8 | 0.773 | 0.998 |

### Runner

```
abm
avx
avx2
avx512f
bmi1
bmi2
fma
popcnt
sse4_2
Model name:                              Intel(R) Xeon(R) 6973P-C
rustc 1.98.1 (48a229cea 2026-09-01)
libjpeg-turbo version 3.2.0 (build 20260630)
```

## Run 5 — same-run `main` vs branch, second dispatch (`69bd3a7`)

[Run 34161016801](https://github.com/developer0hye/libjpeg-turbo-rs/actions/runs/34161016801),
again an Intel Xeon 6973P-C, rustc 1.98.1, C 3.2.0. Dispatched to try for a
quiet Zen host; drew the same model as run 4 with a 1.0–12.4 % integer
bracket, so it adds a second sample rather than a decision.

- **Float DCT:** `main-portable / baseline` 1.156–1.364 on all eleven cases,
  `fma / baseline` 0.92–1.08 (noise), `native` 1.24–1.32 slower — the same
  picture as run 4.
- **Integer DCT:** `main-portable / baseline` at 1080p 0.962 (noise) / 1.023
  (noise) / 1.049 against brackets of 4.2 / 4.4 / 1.0 %; every variant column
  is marked noise on most rows. Nothing at the 1–3 % level survives this
  host. Observation only: `target-cpu=native` is 7–26 % *slower* than
  portable on this AVX-512 box for the integer path, which is not P4-133's
  question but is worth knowing before anyone recommends the flag on Xeon.

Measurement stopped here: three dispatches, two of them with the same-run
column, both on the noisy model. The column stays in the harness; the next
quiet Zen runner answers the BMI2 question for free.

### Integer DCT (islow) — µs, ratio to portable build, ratio to C 3.2.0

| Case | noise (pair spread) | stock (µs) | stock / baseline | stock / reference | main-portable (µs) | main-portable / baseline | main-portable / reference | bmi1-lzcnt (µs) | bmi1-lzcnt / baseline | bmi1-lzcnt / reference | bmi2 (µs) | bmi2 / baseline | bmi2 / reference | fma (µs) | fma / baseline | fma / reference | readme-set (µs) | readme-set / baseline | readme-set / reference | avx2 (µs) | avx2 / baseline | avx2 / reference | native (µs) | native / baseline | native / reference | stock-again (µs) | stock-again / baseline | stock-again / reference | C (µs) | C / baseline | C / reference | C-again (µs) | C-again / baseline | C-again / reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x64_420 | 3.9% | 12.7 | 1.000 (noise) | 0.760 | 12.5 | 0.984 (noise) | 0.749 | 12.1 | 0.953 | 0.725 | 12.2 | 0.961 (noise) | 0.731 | 12.2 | 0.961 (noise) | 0.731 | 12.2 | 0.961 (noise) | 0.731 | 12.3 | 0.969 (noise) | 0.737 | 14.0 | 1.102 | 0.838 | 12.2 | 0.961 (noise) | 0.731 | 16.7 | 1.315 | 1.000 | 16.6 | 1.307 | 0.994 |
| 320x240_420 | 7.1% | 220.1 | 1.000 (noise) | 1.062 | 203.9 | 0.926 | 0.984 | 198.3 | 0.901 | 0.957 | 213.2 | 0.969 (noise) | 1.029 | 216.8 | 0.985 (noise) | 1.046 | 205.2 | 0.932 (noise) | 0.990 | 200.6 | 0.911 | 0.968 | 237.7 | 1.080 | 1.147 | 204.4 | 0.929 (noise) | 0.986 | 207.2 | 0.941 (noise) | 1.000 | 207.7 | 0.944 (noise) | 1.002 |
| 320x240_422 | 11.0% | 263.6 | 1.000 (noise) | 1.005 | 281.1 | 1.066 (noise) | 1.071 | 256.0 | 0.971 (noise) | 0.976 | 269.6 | 1.023 (noise) | 1.027 | 250.7 | 0.951 (noise) | 0.955 | 259.9 | 0.986 (noise) | 0.990 | 253.7 | 0.962 (noise) | 0.967 | 299.8 | 1.137 | 1.143 | 292.5 | 1.110 (noise) | 1.115 | 262.4 | 0.995 (noise) | 1.000 | 265.0 | 1.005 (noise) | 1.010 |
| 320x240_444 | 8.6% | 399.6 | 1.000 (noise) | 0.922 | 464.2 | 1.162 | 1.071 | 400.5 | 1.002 (noise) | 0.924 | 402.5 | 1.007 (noise) | 0.929 | 387.3 | 0.969 (noise) | 0.894 | 389.6 | 0.975 (noise) | 0.899 | 385.6 | 0.965 (noise) | 0.890 | 462.0 | 1.156 | 1.066 | 434.0 | 1.086 (noise) | 1.001 | 433.4 | 1.085 (noise) | 1.000 | 391.6 | 0.980 (noise) | 0.904 |
| 640x480_420 | 5.7% | 805.2 | 1.000 (noise) | 0.979 | 783.3 | 0.973 (noise) | 0.952 | 766.0 | 0.951 (noise) | 0.931 | 759.7 | 0.943 (noise) | 0.924 | 766.1 | 0.951 (noise) | 0.931 | 760.4 | 0.944 (noise) | 0.924 | 767.7 | 0.953 (noise) | 0.933 | 932.9 | 1.159 | 1.134 | 759.4 | 0.943 (noise) | 0.923 | 822.5 | 1.021 (noise) | 1.000 | 725.8 | 0.901 | 0.882 |
| 640x480_422 | 1.1% | 960.4 | 1.000 (noise) | 1.069 | 1006.4 | 1.048 | 1.120 | 947.0 | 0.986 | 1.054 | 961.3 | 1.001 (noise) | 1.070 | 942.7 | 0.982 | 1.050 | 1002.4 | 1.044 | 1.116 | 949.8 | 0.989 | 1.057 | 1208.2 | 1.258 | 1.345 | 970.9 | 1.011 (noise) | 1.081 | 898.2 | 0.935 | 1.000 | 901.1 | 0.938 | 1.003 |
| 640x480_444 | 2.9% | 1391.9 | 1.000 (noise) | 1.054 | 1477.3 | 1.061 | 1.119 | 1389.6 | 0.998 (noise) | 1.052 | 1416.7 | 1.018 (noise) | 1.073 | 1357.8 | 0.976 (noise) | 1.028 | 1359.9 | 0.977 (noise) | 1.030 | 1361.3 | 0.978 (noise) | 1.031 | 1684.0 | 1.210 | 1.275 | 1432.1 | 1.029 (noise) | 1.085 | 1320.3 | 0.949 | 1.000 | 1339.1 | 0.962 | 1.014 |
| 1280x720_420 | 12.4% | 2996.9 | 1.000 (noise) | 1.124 | 2748.8 | 0.917 (noise) | 1.031 | 2660.9 | 0.888 (noise) | 0.998 | 2626.1 | 0.876 (noise) | 0.985 | 2670.5 | 0.891 (noise) | 1.002 | 2660.0 | 0.888 (noise) | 0.998 | 2645.4 | 0.883 (noise) | 0.992 | 3193.7 | 1.066 (noise) | 1.198 | 2625.0 | 0.876 (noise) | 0.985 | 2666.2 | 0.890 (noise) | 1.000 | 2639.0 | 0.881 (noise) | 0.990 |
| 1920x1080_420 | 4.2% | 6380.6 | 1.000 (noise) | 1.090 | 6140.4 | 0.962 (noise) | 1.049 | 6102.8 | 0.956 | 1.043 | 6081.9 | 0.953 | 1.039 | 6032.9 | 0.946 | 1.031 | 5978.1 | 0.937 | 1.021 | 5984.5 | 0.938 | 1.022 | 7111.0 | 1.114 | 1.215 | 6113.7 | 0.958 (noise) | 1.045 | 5852.9 | 0.917 | 1.000 | 6048.4 | 0.948 | 1.033 |
| 1920x1080_422 | 4.4% | 7523.8 | 1.000 (noise) | 1.018 | 7700.4 | 1.023 (noise) | 1.042 | 7641.0 | 1.016 (noise) | 1.034 | 7776.2 | 1.034 (noise) | 1.052 | 7460.3 | 0.992 (noise) | 1.009 | 7638.7 | 1.015 (noise) | 1.034 | 7982.7 | 1.061 | 1.080 | 8835.2 | 1.174 | 1.196 | 7856.0 | 1.044 (noise) | 1.063 | 7390.2 | 0.982 (noise) | 1.000 | 7367.1 | 0.979 (noise) | 0.997 |
| 1920x1080_444 | 1.0% | 11134.3 | 1.000 (noise) | 1.027 | 11685.1 | 1.049 | 1.078 | 11707.3 | 1.051 | 1.080 | 11495.3 | 1.032 | 1.061 | 11546.4 | 1.037 | 1.065 | 11313.5 | 1.016 | 1.044 | 11078.7 | 0.995 (noise) | 1.022 | 13078.6 | 1.175 | 1.207 | 11242.6 | 1.010 (noise) | 1.037 | 10837.7 | 0.973 | 1.000 | 11085.4 | 0.996 (noise) | 1.023 |

### Float DCT — µs, ratio to portable build, ratio to C 3.2.0

| Case | noise (pair spread) | stock (µs) | stock / baseline | stock / reference | main-portable (µs) | main-portable / baseline | main-portable / reference | fma (µs) | fma / baseline | fma / reference | readme-set (µs) | readme-set / baseline | readme-set / reference | native (µs) | native / baseline | native / reference | stock-again (µs) | stock-again / baseline | stock-again / reference | C (µs) | C / baseline | C / reference | C-again (µs) | C-again / baseline | C-again / reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x64_420 | 5.6% | 17.7 | 1.000 (noise) | 1.000 | 21.5 | 1.215 | 1.215 | 17.3 | 0.977 (noise) | 0.977 | 17.5 | 0.989 (noise) | 0.989 | 22.5 | 1.271 | 1.271 | 18.7 | 1.056 (noise) | 1.056 | 17.7 | 1.000 (noise) | 1.000 | 17.7 | 1.000 (noise) | 1.000 |
| 320x240_420 | 0.1% | 303.6 | 1.000 (noise) | 1.302 | 396.9 | 1.307 | 1.702 | 297.6 | 0.980 | 1.276 | 305.5 | 1.006 | 1.310 | 400.1 | 1.318 | 1.716 | 303.3 | 0.999 (noise) | 1.301 | 233.2 | 0.768 | 1.000 | 230.9 | 0.761 | 0.990 |
| 320x240_422 | 7.2% | 418.2 | 1.000 (noise) | 1.407 | 483.3 | 1.156 | 1.626 | 385.9 | 0.923 | 1.298 | 401.3 | 0.960 (noise) | 1.350 | 528.2 | 1.263 | 1.777 | 448.3 | 1.072 (noise) | 1.508 | 297.2 | 0.711 | 1.000 | 312.2 | 0.747 | 1.050 |
| 320x240_444 | 2.0% | 607.4 | 1.000 (noise) | 1.384 | 745.2 | 1.227 | 1.698 | 582.1 | 0.958 | 1.327 | 592.4 | 0.975 | 1.350 | 751.3 | 1.237 | 1.712 | 595.4 | 0.980 (noise) | 1.357 | 438.8 | 0.722 | 1.000 | 489.6 | 0.806 | 1.116 |
| 640x480_420 | 2.1% | 1134.4 | 1.000 (noise) | 1.331 | 1512.2 | 1.333 | 1.774 | 1156.3 | 1.019 (noise) | 1.357 | 1163.0 | 1.025 | 1.364 | 1481.3 | 1.306 | 1.738 | 1157.7 | 1.021 (noise) | 1.358 | 852.4 | 0.751 | 1.000 | 833.8 | 0.735 | 0.978 |
| 640x480_422 | 1.5% | 1466.0 | 1.000 (noise) | 1.402 | 1823.6 | 1.244 | 1.744 | 1529.2 | 1.043 | 1.462 | 1494.9 | 1.020 | 1.429 | 1858.3 | 1.268 | 1.777 | 1488.6 | 1.015 (noise) | 1.423 | 1045.8 | 0.713 | 1.000 | 1017.3 | 0.694 | 0.973 |
| 640x480_444 | 5.4% | 2094.4 | 1.000 (noise) | 1.413 | 2631.1 | 1.256 | 1.775 | 2145.8 | 1.025 (noise) | 1.448 | 2143.7 | 1.024 (noise) | 1.446 | 2762.6 | 1.319 | 1.864 | 2206.9 | 1.054 (noise) | 1.489 | 1482.0 | 0.708 | 1.000 | 1550.9 | 0.740 | 1.046 |
| 1280x720_420 | 1.2% | 3817.9 | 1.000 (noise) | 1.278 | 4850.0 | 1.270 | 1.623 | 4096.1 | 1.073 | 1.371 | 3843.1 | 1.007 (noise) | 1.286 | 4772.0 | 1.250 | 1.597 | 3773.1 | 0.988 (noise) | 1.263 | 2988.4 | 0.783 | 1.000 | 2970.3 | 0.778 | 0.994 |
| 1920x1080_420 | 1.1% | 8468.9 | 1.000 (noise) | 1.263 | 11549.3 | 1.364 | 1.722 | 8743.2 | 1.032 | 1.303 | 8806.5 | 1.040 | 1.313 | 10865.0 | 1.283 | 1.620 | 8560.3 | 1.011 (noise) | 1.276 | 6707.5 | 0.792 | 1.000 | 6804.1 | 0.803 | 1.014 |
| 1920x1080_422 | 4.7% | 10737.1 | 1.000 (noise) | 1.276 | 13979.6 | 1.302 | 1.661 | 11570.4 | 1.078 | 1.375 | 11053.3 | 1.029 (noise) | 1.313 | 13731.4 | 1.279 | 1.631 | 11246.1 | 1.047 (noise) | 1.336 | 8417.5 | 0.784 | 1.000 | 8357.7 | 0.778 | 0.993 |
| 1920x1080_444 | 4.6% | 15955.3 | 1.000 (noise) | 1.228 | 20685.7 | 1.296 | 1.593 | 16180.8 | 1.014 (noise) | 1.246 | 16713.5 | 1.048 | 1.287 | 20721.9 | 1.299 | 1.595 | 16692.1 | 1.046 (noise) | 1.285 | 12988.0 | 0.814 | 1.000 | 12542.2 | 0.786 | 0.966 |

### Runner

```
abm
avx
avx2
avx512f
bmi1
bmi2
fma
popcnt
sse4_2
Model name:                              Intel(R) Xeon(R) 6973P-C
rustc 1.98.1 (48a229cea 2026-09-01)
libjpeg-turbo version 3.2.0 (build 20260630)
```

