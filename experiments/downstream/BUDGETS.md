# Downstream-consumer regression budgets

**First report:** 2026-10-07, x86_64-linux hosted runner (AMD EPYC 7763,
4 vCPUs), `VARIANT=default`: Cargo's stock `release` profile, no
`RUSTFLAGS`, 30 timed iterations per row.
P4-214 / [#640](https://github.com/developer0hye/libjpeg-turbo-rs/issues/640)
criterion 6.

| report | dispatch | candidate commit |
|---|---|---|
| [`reports/2026-10-07-x86_64-linux/`](reports/2026-10-07-x86_64-linux/report.md) (the first report) | run 37542461917 | `c33650c` |
| [`reports/2026-10-07-x86_64-linux-run2/`](reports/2026-10-07-x86_64-linux-run2/report.md) (reproduction) | run 37543406684 | `e56ace4` |
| [`reports/2026-10-07-aarch64-macos-contaminated/`](reports/2026-10-07-aarch64-macos-contaminated/report.md) (no budget; see below) | run 37542461917 | `c33650c` |
| [`reports/2026-10-07-aarch64-macos-contaminated-run2/`](reports/2026-10-07-aarch64-macos-contaminated-run2/report.md) (no budget; see below) | run 37543406684 | `e56ace4` |

Both candidate commits are `main@80c15d2` plus consumer, CI and documentation
changes only (nothing under `src/`), so the candidate *library* in every report is `main@80c15d2`.
The baseline is the published `libjpeg-turbo-rs` 0.8.0.

**What the hosted reports do not check.** The workflow runs the harness with
`--no-c-oracle`, since the runners have no stock libjpeg-turbo, so the C
column of every hosted report reads "disabled". The C contract was checked
by the local runs recorded in
[P4-214](../../docs/last_mile/phase4.md#p4-214-no-benchmark-measures-a-default-profile-downstream-consumer--closed-2026-10-07).
Against stock 3.2.0 `djpeg` and `cjpeg`, candidate and baseline decodes were
pixel-identical on all eight decode cases, and both `compress` outputs were
byte-identical to `cjpeg -quality 85` (`-sample 1x1` for the 4:4:4 rows).

**aarch64 sets no budget.** Both dispatches gave the arm64 leg a 3-vCPU
`macos-latest` runner that its own pre-run sample showed saturated: load
average 50 (0.3 % idle in the second frame), then 29. Its p10–p90 spreads were
13–80 % of the median in the first dispatch and 6–159 % in the second. Both
are committed as a record of the attempt, and none of their ratios is
budget-grade. `budgets.py` refuses to score an aarch64 report against the
x86_64 first report. The aarch64 budget waits for a
run that passes a load check:
[P4-229](../../docs/last_mile/phase4.md#p4-229-the-downstream-harness-records-machine-load-but-never-acts-on-it-and-the-hosted-macos-runner-was-saturated--open).

## Rules

These are the rules `README.md` "Regression budget" fixed before any data
existed, plus the *lead pairs* below, which that section does not name.
`budgets.py` applies the timing rules to any `report.json`:

```sh
python3 experiments/downstream/budgets.py <new report.json> \
  --first experiments/downstream/reports/2026-10-07-x86_64-linux/report.json
```

- **Time, same run only.** Hosted runners change CPU model and neighbours
  between runs, so no budget compares absolute times across runs. Each budget
  is a ratio of two rows' medians from the **same** run. The rounds
  interleave the rows of a case, so both rows see the same machine.
  - *Band:* `max(2 × the wider of the two rows' (p90 − p10) / median, 3 %)`,
    taken from the first report, which sets the noise floor for later ones.
  - *Parity pairs* (candidate / published 0.8.0) are held to `1 + band`.
    The candidate may not get slower than the release by more than the run
    can resolve.
  - *Lead pairs* (candidate / zune-jpeg or `image`'s codec) are held to the
    first report's ratio × `(1 + its band)`. A lead may shrink only by what
    the first run could not resolve.
  - A row counts as regressed only when it is over budget in **two**
    dispatches.
- **Allocations: zero budget.** Counts, cumulative bytes and peak live bytes
  are deterministic for an architecture. The two x86_64 runs agree on every
  row, and so do the two aarch64 runs. Any increase for the same case is a change to explain.
- **Encode output: identical.** Candidate `compress` bytes must equal the
  baseline's (and C `cjpeg`'s, where C is available). The first report's
  candidate and baseline rows are identical in bytes and PSNR on every encode
  case.
- **Binary size and build time:** the probe contributions below are the
  reference. Growth over 5 % needs a stated reason.

## Timing ratios (x86_64)

| case | pair | kind | run 1 | run 2 | band | budget | verdict |
|---|---|---|---:|---:|---:|---:|---|
| small-64x64-420 | candidate-fresh / baseline-fresh | parity | 1.010 | 1.004 | 4.5% | 1.045 | ok |
| small-64x64-420 | candidate-reuse / baseline-reuse | parity | 0.981 | 0.983 | 4.8% | 1.048 | ok |
| small-64x64-420 | candidate-reuse / zune-reuse | lead | 1.002 | 1.001 | 4.8% | 1.050 | ok |
| small-64x64-420 | candidate-image-adapter / image-builtin | lead | 1.056 | 1.062 | 4.3% | 1.102 | ok |
| phone-4032x3024-420 | candidate-fresh / baseline-fresh | parity | 1.020 | 1.010 | 3.0% | 1.030 | ok |
| phone-4032x3024-420 | candidate-reuse / baseline-reuse | parity | 1.002 | 0.991 | 6.7% | 1.067 | ok |
| phone-4032x3024-420 | candidate-reuse / zune-reuse | lead | 0.801 | 0.800 | 6.7% | 0.855 | ok |
| phone-4032x3024-420 | candidate-image-adapter / image-builtin | lead | 0.782 | 0.783 | 5.2% | 0.823 | ok |
| phone-4032x3024-420-scale-1/4 | candidate-fresh / baseline-fresh | parity | 0.911 | 0.906 | 8.2% | 1.082 | ok |
| phone-4032x3024-420-scale-1/4 | candidate-reuse / baseline-reuse | parity | 0.899 | 0.901 | 12.5% | 1.125 | ok |
| gray-1920x1080 | candidate-fresh / baseline-fresh | parity | 0.998 | 0.995 | 3.0% | 1.030 | ok |
| gray-1920x1080 | candidate-reuse / baseline-reuse | parity | 1.018 | 1.013 | 3.0% | 1.030 | ok |
| gray-1920x1080 | candidate-reuse / zune-reuse | lead | 0.736 | 0.733 | 3.0% | 0.758 | ok |
| gray-1920x1080 | candidate-image-adapter / image-builtin | lead | 0.730 | 0.733 | 3.0% | 0.752 | ok |
| progressive-1920x1080-420 | candidate-fresh / baseline-fresh | parity | 1.003 | 1.002 | 3.0% | 1.030 | ok |
| progressive-1920x1080-420 | candidate-reuse / baseline-reuse | parity | 1.001 | 1.002 | 3.0% | 1.030 | ok |
| progressive-1920x1080-420 | candidate-reuse / zune-reuse | lead | 0.763 | 0.759 | 3.0% | 0.786 | ok |
| progressive-1920x1080-420 | candidate-image-adapter / image-builtin | lead | 0.776 | 0.776 | 3.0% | 0.799 | ok |
| large-7680x4320-420 | candidate-fresh / baseline-fresh | parity | 1.026 | 1.024 | 3.0% | 1.030 | ok |
| large-7680x4320-420 | candidate-reuse / baseline-reuse | parity | 0.992 | 0.990 | 3.0% | 1.030 | ok |
| large-7680x4320-420 | candidate-reuse / zune-reuse | lead | 0.805 | 0.805 | 3.0% | 0.829 | ok |
| large-7680x4320-420 | candidate-image-adapter / image-builtin | lead | 0.800 | 0.799 | 3.4% | 0.827 | ok |
| testorig | candidate-fresh / baseline-fresh | parity | 1.009 | 1.022 | 7.0% | 1.070 | ok |
| testorig | candidate-reuse / baseline-reuse | parity | 0.984 | 0.987 | 11.4% | 1.114 | ok |
| testorig | candidate-reuse / zune-reuse | lead | 0.838 | 0.843 | 11.4% | 0.934 | ok |
| testorig | candidate-image-adapter / image-builtin | lead | 0.815 | 0.813 | 3.0% | 0.840 | ok |
| testimgint | candidate-fresh / baseline-fresh | parity | 1.010 | 1.007 | 5.6% | 1.056 | ok |
| testimgint | candidate-reuse / baseline-reuse | parity | 0.984 | 0.983 | 11.0% | 1.110 | ok |
| testimgint | candidate-reuse / zune-reuse | lead | 0.840 | 0.844 | 11.0% | 0.932 | ok |
| testimgint | candidate-image-adapter / image-builtin | lead | 0.814 | 0.809 | 4.8% | 0.853 | ok |
| encode-64x64 | candidate / baseline | parity | 1.003 | 1.003 | 3.0% | 1.030 | ok |
| encode-64x64 | candidate-444 / baseline-444 | parity | 1.011 | 1.017 | 3.0% | 1.030 | ok |
| encode-64x64 | candidate-444 / image-builtin | lead | 0.313 | 0.314 | 3.0% | 0.323 | ok |
| encode-64x64 | candidate-image-adapter / image-builtin | lead | 0.198 | 0.197 | 3.0% | 0.204 | ok |
| encode-1920x1080 | candidate / baseline | parity | 0.993 | 0.993 | 3.0% | 1.030 | ok |
| encode-1920x1080 | candidate-444 / baseline-444 | parity | 1.004 | 1.015 | 3.0% | 1.030 | ok |
| encode-1920x1080 | candidate-444 / image-builtin | lead | 0.257 | 0.260 | 3.0% | 0.265 | ok |
| encode-1920x1080 | candidate-image-adapter / image-builtin | lead | 0.145 | 0.145 | 3.0% | 0.149 | ok |
| encode-4032x3024 | candidate / baseline | parity | 0.993 | 0.994 | 3.0% | 1.030 | ok |
| encode-4032x3024 | candidate-444 / baseline-444 | parity | 1.004 | 1.016 | 3.0% | 1.030 | ok |
| encode-4032x3024 | candidate-444 / image-builtin | lead | 0.257 | 0.260 | 3.0% | 0.265 | ok |
| encode-4032x3024 | candidate-image-adapter / image-builtin | lead | 0.144 | 0.144 | 3.0% | 0.148 | ok |
| thumbnail | candidate / baseline | parity | 1.060 | 1.060 | 3.0% | 1.030 | **over in both runs** (P4-228) |
| thumbnail | candidate-image-adapter / image-builtin | lead | 0.917 | 0.916 | 5.3% | 0.966 | ok |
| thumbnail | candidate-scaled-decode / image-builtin | lead | 0.265 | 0.265 | 4.1% | 0.276 | ok |

Kinds: *parity* compares with the published release; *lead* compares with
another codec. A lead ratio below 1 means the candidate is faster.

## Allocation reference (x86_64)

Candidate rows of the first report. These are the budget: any increase in
`allocs` or `peak live` for the same case needs a reason. The reference is
per architecture. The encode rows allocate differently on aarch64: on the
64x64 encode, 11 allocations and an 11.3 KiB peak against 14 and 19.8 KiB
here. The aarch64 reference is that report's own rows, which matched across
both aarch64 runs.

| case | row | allocs | peak live |
|---|---|---:|---:|
| small-64x64-420 | candidate-fresh | 19 | 36.5 KiB |
| small-64x64-420 | candidate-reuse | 18 | 24.5 KiB |
| small-64x64-420 | candidate-image-adapter | 27 | 26.4 KiB |
| phone-4032x3024-420 | candidate-fresh | 19 | 52.36 MiB |
| phone-4032x3024-420 | candidate-reuse | 18 | 17.48 MiB |
| phone-4032x3024-420 | candidate-image-adapter | 27 | 20.43 MiB |
| phone-4032x3024-420-scale-1/4 | candidate-fresh | 15 | 4.38 MiB |
| phone-4032x3024-420-scale-1/4 | candidate-reuse | 14 | 2.20 MiB |
| gray-1920x1080 | candidate-fresh | 8 | 1.99 MiB |
| gray-1920x1080 | candidate-reuse | 8 | 1.99 MiB |
| gray-1920x1080 | candidate-image-adapter | 15 | 2.41 MiB |
| progressive-1920x1080-420 | candidate-fresh | 55 | 9.06 MiB |
| progressive-1920x1080-420 | candidate-reuse | 54 | 9.06 MiB |
| progressive-1920x1080-420 | candidate-image-adapter | 80 | 9.52 MiB |
| large-7680x4320-420 | candidate-fresh | 19 | 142.43 MiB |
| large-7680x4320-420 | candidate-reuse | 18 | 47.51 MiB |
| large-7680x4320-420 | candidate-image-adapter | 27 | 55.55 MiB |
| testorig | candidate-fresh | 19 | 174.5 KiB |
| testorig | candidate-reuse | 18 | 75.4 KiB |
| testorig | candidate-image-adapter | 27 | 81.1 KiB |
| testimgint | candidate-fresh | 19 | 174.5 KiB |
| testimgint | candidate-reuse | 18 | 75.4 KiB |
| testimgint | candidate-image-adapter | 27 | 81.1 KiB |
| encode-64x64 | candidate | 14 | 19.8 KiB |
| encode-64x64 | candidate-444 | 11 | 10.8 KiB |
| encode-64x64 | candidate-image-adapter | 15 | 19.8 KiB |
| encode-1920x1080 | candidate | 13 | 4.33 MiB |
| encode-1920x1080 | candidate-444 | 11 | 4.50 MiB |
| encode-1920x1080 | candidate-image-adapter | 14 | 4.33 MiB |
| encode-4032x3024 | candidate | 13 | 25.41 MiB |
| encode-4032x3024 | candidate-444 | 11 | 26.43 MiB |
| encode-4032x3024 | candidate-image-adapter | 14 | 25.41 MiB |
| thumbnail | candidate | 48 | 69.79 MiB |
| thumbnail | candidate-scaled-decode | 38 | 2.16 MiB |
| thumbnail | candidate-image-adapter | 81 | 69.77 MiB |

## Size and build reference

Contribution of each backend to an otherwise empty stock-profile binary
(unstripped bytes over the empty probe). Clean build times come from one
hosted run each and are indicative only.

| backend | x86_64 bytes | aarch64 bytes | x86_64 clean build |
|---|---:|---:|---:|
| libjpeg-turbo-rs 0.8.0 | 625,472 | 467,104 | 11.7 s |
| candidate | 717,344 | 514,368 | 13.0 s |
| candidate + image adapter (+ `image` codec, see README) | 824,592 | 539,600 | 27.6 s |
| `image` (jpeg feature) | 360,672 | 223,792 | 18.8 s |
| zune-jpeg | 217,464 | 173,248 | 1.7 s |

## Where the candidate loses

Listed so a reader does not have to infer them from the tables:

- **Thumbnail, direct API, vs 0.8.0: 1.060 in both runs, over its 1.030
  budget.** The same output, allocations and peak; `apply_orientation` is
  unchanged since 0.8.0. Filed as
  [P4-228](../../docs/last_mile/phase4.md#p4-228-the-candidate-is-6--slower-than-080-on-the-thumbnail-workload--open)
  ([#659](https://github.com/developer0hye/libjpeg-turbo-rs/issues/659)).
- **Thumbnail, direct API, vs `image`'s own codec: 1.17.** The adapter path
  on the same workload is 0.917, so the loss is specific to the direct
  `Decoder` → `Image::apply_orientation` route (P4-228).
- **Fresh decode drifts with size vs 0.8.0:** 1.010–1.020 at 12 MP and
  1.024–1.026 at 33 MP. That is within budget, but the buffer-reuse rows are
  flat. The first hypothesis is the zero-fill in `try_filled_vec` (P4-228).
- **Image adapter vs `image`'s codec on a 64x64 image: 1.056–1.062** (27 vs
  19 allocations). Fixed per-image overhead; within its lead budget, but a
  loss.
- **Peak memory of the buffer-reuse decode:** 17.48 MiB at 12 MP and
  47.51 MiB at 8K, against zune-jpeg's 0.65 MiB and 1.2 MiB.
  [P4-218](../../docs/last_mile/phase4.md#p4-218-the-buffer-reuse-decode-still-allocates-whole-image-component-planes--open).
- **Binary size vs 0.8.0: +14.7 % (x86_64), +10.1 % (aarch64)**, not yet
  attributed.
  [P4-230](../../docs/last_mile/phase4.md#p4-230-the-candidate-adds-1015--more-code-to-a-stock-profile-binary-than-080--open)
  ([#661](https://github.com/developer0hye/libjpeg-turbo-rs/issues/661)).
- **Small-image decode vs zune-jpeg: a tie** (1.001–1.002 at 64x64). Every
  other decode case leads by 16–27 %.
