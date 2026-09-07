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

