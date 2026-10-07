# Downstream-consumer regression budgets

P4-214 / [#640](https://github.com/developer0hye/libjpeg-turbo-rs/issues/640)
criterion 6: the first measured report, the cases the candidate loses, and
regression budgets derived from measured spread.

## The reference set

Three x86_64 hosted runs of one consumer build on one CPU model, all
dispatched 2026-10-07 at `dd8afb1`: AMD EPYC 7763 (Zen 3), 4 vCPUs, Cargo's
stock `release` profile, no `RUSTFLAGS`, 30 timed iterations per row.

| report | dispatch |
|---|---|
| [`reports/2026-10-07-ref-zen3-1/`](reports/2026-10-07-ref-zen3-1/report.md) | run 37548314633 |
| [`reports/2026-10-07-ref-zen3-2/`](reports/2026-10-07-ref-zen3-2/report.md) | run 37548320600 |
| [`reports/2026-10-07-ref-zen3-3/`](reports/2026-10-07-ref-zen3-3/report.md) | run 37548332954 |

- **Library:** `main@80c15d2`. `dd8afb1` adds only harness, CI and
  documentation changes on top of it, nothing under `src/`.
- **Baseline:** the published `libjpeg-turbo-rs` 0.8.0.
- **Consumer sources:** `consumer_source_sha256` `aac5b6b3…5253`.

A fourth dispatch at `dd8afb1` (run 37548326976) landed on an AMD EPYC 9V74 (Zen 4) and is
committed as [`reports/2026-10-07-zen4/`](reports/2026-10-07-zen4/report.md).
It is not part of the reference: `budgets.py` refuses to compare across CPU
models.

**No C oracle on the hosted runs.** The workflow passes `--no-c-oracle`,
because the runners have no stock libjpeg-turbo, so every hosted report's C
rows read "disabled". The C contract is checked locally with stock 3.2.0
[(P4-214)](../../docs/last_mile/phase4.md#p4-214-no-benchmark-measures-a-default-profile-downstream-consumer--closed-2026-10-07).
Candidate and baseline decodes are pixel-identical to `djpeg` on every
decode case and on the concurrent section's reference decode. Both
`compress` outputs are byte-identical to `cjpeg -quality 85` (`-sample 1x1`
for the 4:4:4 rows).

**No aarch64 budget.** All eight measured `macos-latest` legs across the 2026-10-07
dispatches ran on a 3-vCPU runner that the harness's own pre-run sample showed
saturated, with load averages of 28–50. Two of those reports are committed
as a record (`reports/2026-10-07-aarch64-macos-contaminated*/`). The aarch64
budget waits for a run that passes a load check:
[P4-229](../../docs/last_mile/phase4.md#p4-229-the-downstream-harness-records-machine-load-but-never-acts-on-it-and-the-hosted-macos-runner-was-saturated--open).

## Rules

```sh
python3 experiments/downstream/budgets.py <new report.json> \
  --first experiments/downstream/reports/2026-10-07-ref-zen3-1/report.json \
  --first experiments/downstream/reports/2026-10-07-ref-zen3-2/report.json \
  --first experiments/downstream/reports/2026-10-07-ref-zen3-3/report.json
```

- **Comparable reports only.** Every report in the set must:
  - be a full run, with no `--smoke` and no `--only`;
  - have the same CPU model, architecture, build variant and recorded runtime
    CPU features;
  - have the same `consumer_source_sha256` and the same `rustc -Vv` output;
  - use the same iterations and warmup, and the same concurrent thread
    and decode counts;
  - time each row in batches within half to double the batch sizes the
    reference used. Batch sizes are adaptive, and the reference's own
    variation is inside the band.

  `budgets.py` refuses anything else (exit 2). A report of a different CPU
  model means re-dispatching, not widening the band.
- **Time: same-run ratios.** Each budgeted figure is the ratio of two rows'
  medians from the same run. The rounds interleave a case's rows, so both
  rows see the same machine. No budget compares absolute times across runs.
  - *Band:* `max(3 %, cross-run range, 2 × the worst within-run (p90 − p10) / median)`
    over the reference runs. README's predeclared rule is
    `max(2 × spread, 3 %)`; the cross-run term was added after the evidence
    under *Why the band has a cross-run term* below.
  - *Limit:* the reference median × `(1 + band)`, for every pair. A later
    report may not be slower than the reference by more than the reference
    runs disagreed among themselves.
  - *Versus 0.8.0:* each parity pair also says whether the reference median is
    behind the published release by more than the band. Those rows are the
    known losing cases. They are listed, not failed. Holding parity rows to
    1.0 would fail every known gap on every run and bury a new regression
    among them.
  - *Unresolvable:* a pair whose cross-run range exceeds 10 % gets no budget
    and is reported as "not resolvable on these runners". The current
    reference has none.
  - A row counts as regressed only when it is over its limit in **two**
    dispatches.
- **Allocations: zero budget.** Count and cumulative bytes are deterministic,
  and so is peak live in the single-threaded sections. All four x86_64 runs
  agree exactly. Any increase for the same case needs a reason. The
  concurrent section's peak live heap also agreed across all four, but it
  depends on how the threads' working sets overlap, and a smoke run
  measured a different value. Its budget is therefore the
  `T ×` single-decode ceiling, not equality.
- **Encode output: identical.** Candidate `compress` bytes equal the
  baseline's on every encode case, and C `cjpeg`'s where C is available.
  The harness asserts this only through `cjpeg`. Without C (every hosted
  report) it records each output's length and PSNR, not its bytes, so
  equal length and PSNR is all a hosted report shows.
- **Binary size and build time:** the probe contributions below are the
  reference. Growth over 5 % needs a stated reason.
- **Regenerate the reference when the consumer or the compiler changes.** A
  new `consumer_source_sha256` or a new stable `rustc` (the workflow uses
  `dtolnay/rust-toolchain@stable`; the reference was built by rustc 1.99.0)
  makes the old set unusable by construction. Dispatch
  three runs, keep those on one CPU model, and replace the set and the tables
  below.

## Reference ratios (x86_64, Zen 3)

Kinds: *parity* compares with the published release, and *lead* compares
with another codec. A ratio below 1 means the candidate is faster. The Zen 4
column is a single run and sets nothing.

| case | pair | kind | run 1 | run 2 | run 3 | range | band | limit | vs 0.8.0 | Zen 4 (one run) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| small-64x64-420 | candidate-fresh / baseline-fresh | parity | 0.997 | 0.994 | 0.993 | 0.4% | 5.8% | 1.052 | within band | 1.001 |
| small-64x64-420 | candidate-reuse / baseline-reuse | parity | 0.975 | 0.972 | 1.021 | 4.8% | 5.0% | 1.024 | within band | 1.002 |
| small-64x64-420 | candidate-reuse / zune-reuse | lead | 0.990 | 0.982 | 0.999 | 1.7% | 4.2% | 1.032 | - | 1.062 |
| small-64x64-420 | candidate-image-adapter / image-builtin | lead | 1.065 | 1.026 | 1.047 | 3.8% | 5.7% | 1.106 | - | 1.114 |
| phone-4032x3024-420 | candidate-fresh / baseline-fresh | parity | 1.024 | 1.008 | 1.021 | 1.7% | 7.0% | 1.093 | within band | 1.042 |
| phone-4032x3024-420 | candidate-reuse / baseline-reuse | parity | 0.978 | 0.985 | 1.031 | 5.3% | 5.8% | 1.042 | within band | 1.034 |
| phone-4032x3024-420 | candidate-reuse / zune-reuse | lead | 0.818 | 0.814 | 0.820 | 0.7% | 5.8% | 0.865 | - | 0.913 |
| phone-4032x3024-420 | candidate-image-adapter / image-builtin | lead | 0.818 | 0.794 | 0.810 | 2.5% | 6.7% | 0.865 | - | 0.921 |
| phone-4032x3024-420-scale-1/4 | candidate-fresh / baseline-fresh | parity | 1.011 | 1.016 | 1.044 | 3.3% | 3.3% | 1.050 | within band | 1.038 |
| phone-4032x3024-420-scale-1/4 | candidate-reuse / baseline-reuse | parity | 1.039 | 1.037 | 1.024 | 1.5% | 3.0% | 1.068 | **behind** | 1.040 |
| gray-1920x1080 | candidate-fresh / baseline-fresh | parity | 0.982 | 0.986 | 1.058 | 7.6% | 7.6% | 1.061 | within band | 1.027 |
| gray-1920x1080 | candidate-reuse / baseline-reuse | parity | 1.000 | 1.010 | 1.037 | 3.7% | 3.7% | 1.048 | within band | 1.027 |
| gray-1920x1080 | candidate-reuse / zune-reuse | lead | 0.749 | 0.745 | 0.723 | 2.7% | 3.2% | 0.769 | - | 0.796 |
| gray-1920x1080 | candidate-image-adapter / image-builtin | lead | 0.743 | 0.744 | 0.726 | 1.9% | 3.5% | 0.769 | - | 0.807 |
| progressive-1920x1080-420 | candidate-fresh / baseline-fresh | parity | 0.997 | 1.004 | 0.999 | 0.6% | 3.6% | 1.035 | within band | 0.995 |
| progressive-1920x1080-420 | candidate-reuse / baseline-reuse | parity | 0.995 | 0.998 | 1.007 | 1.3% | 3.8% | 1.037 | within band | 0.998 |
| progressive-1920x1080-420 | candidate-reuse / zune-reuse | lead | 0.766 | 0.775 | 0.776 | 1.1% | 3.8% | 0.804 | - | 0.760 |
| progressive-1920x1080-420 | candidate-image-adapter / image-builtin | lead | 0.783 | 0.790 | 0.790 | 0.8% | 3.6% | 0.818 | - | 0.791 |
| large-7680x4320-420 | candidate-fresh / baseline-fresh | parity | 1.028 | 1.028 | 1.032 | 0.4% | 3.0% | 1.058 | within band | 1.050 |
| large-7680x4320-420 | candidate-reuse / baseline-reuse | parity | 0.987 | 0.988 | 1.033 | 4.6% | 4.6% | 1.034 | within band | 1.033 |
| large-7680x4320-420 | candidate-reuse / zune-reuse | lead | 0.816 | 0.820 | 0.825 | 0.9% | 3.0% | 0.845 | - | 0.902 |
| large-7680x4320-420 | candidate-image-adapter / image-builtin | lead | 0.816 | 0.816 | 0.830 | 1.4% | 3.0% | 0.841 | - | 0.918 |
| testorig | candidate-fresh / baseline-fresh | parity | 1.008 | 1.012 | 1.011 | 0.4% | 7.8% | 1.090 | within band | 1.022 |
| testorig | candidate-reuse / baseline-reuse | parity | 0.970 | 0.990 | 1.049 | 7.9% | 11.5% | 1.104 | within band | 1.024 |
| testorig | candidate-reuse / zune-reuse | lead | 0.841 | 0.840 | 0.853 | 1.3% | 11.4% | 0.937 | - | 0.943 |
| testorig | candidate-image-adapter / image-builtin | lead | 0.809 | 0.801 | 0.840 | 3.9% | 6.9% | 0.865 | - | 0.964 |
| testimgint | candidate-fresh / baseline-fresh | parity | 1.005 | 1.018 | 1.010 | 1.3% | 7.4% | 1.085 | within band | 1.016 |
| testimgint | candidate-reuse / baseline-reuse | parity | 0.972 | 0.990 | 1.047 | 7.4% | 12.2% | 1.110 | within band | 1.024 |
| testimgint | candidate-reuse / zune-reuse | lead | 0.846 | 0.844 | 0.857 | 1.3% | 10.1% | 0.931 | - | 0.947 |
| testimgint | candidate-image-adapter / image-builtin | lead | 0.810 | 0.810 | 0.841 | 3.2% | 6.0% | 0.858 | - | 0.957 |
| encode-64x64 | candidate / baseline | parity | 0.999 | 1.000 | 0.999 | 0.1% | 3.0% | 1.029 | within band | 1.053 |
| encode-64x64 | candidate-444 / baseline-444 | parity | 1.003 | 0.999 | 0.988 | 1.6% | 3.1% | 1.030 | within band | 1.004 |
| encode-64x64 | candidate-444 / image-builtin | lead | 0.310 | 0.310 | 0.311 | 0.1% | 3.0% | 0.319 | - | 0.281 |
| encode-64x64 | candidate-image-adapter / image-builtin | lead | 0.195 | 0.195 | 0.195 | 0.0% | 3.0% | 0.201 | - | 0.178 |
| encode-1920x1080 | candidate / baseline | parity | 0.990 | 0.993 | 0.992 | 0.2% | 3.0% | 1.021 | within band | 1.069 |
| encode-1920x1080 | candidate-444 / baseline-444 | parity | 0.999 | 0.987 | 0.990 | 1.2% | 3.0% | 1.019 | within band | 1.002 |
| encode-1920x1080 | candidate-444 / image-builtin | lead | 0.253 | 0.253 | 0.253 | 0.1% | 3.0% | 0.261 | - | 0.227 |
| encode-1920x1080 | candidate-image-adapter / image-builtin | lead | 0.142 | 0.142 | 0.143 | 0.0% | 3.0% | 0.147 | - | 0.130 |
| encode-4032x3024 | candidate / baseline | parity | 0.990 | 0.992 | 0.991 | 0.3% | 3.0% | 1.021 | within band | 1.071 |
| encode-4032x3024 | candidate-444 / baseline-444 | parity | 0.999 | 0.988 | 0.991 | 1.1% | 3.0% | 1.020 | within band | 1.001 |
| encode-4032x3024 | candidate-444 / image-builtin | lead | 0.254 | 0.253 | 0.253 | 0.1% | 3.0% | 0.261 | - | 0.227 |
| encode-4032x3024 | candidate-image-adapter / image-builtin | lead | 0.142 | 0.142 | 0.142 | 0.0% | 3.0% | 0.147 | - | 0.130 |
| thumbnail | candidate / baseline | parity | 1.010 | 1.002 | 1.014 | 1.2% | 8.7% | 1.098 | within band | 1.015 |
| thumbnail | candidate-image-adapter / image-builtin | lead | 0.918 | 0.919 | 0.915 | 0.4% | 7.9% | 0.990 | - | 0.941 |
| thumbnail | candidate-scaled-decode / image-builtin | lead | 0.271 | 0.264 | 0.269 | 0.7% | 6.2% | 0.286 | - | 0.298 |
| concurrent-phone-4032x3024-420 | candidate-fresh / baseline-fresh | parity | 1.057 | 1.059 | 1.023 | 3.6% | 8.4% | 1.146 | within band | 1.059 |
| concurrent-phone-4032x3024-420 | candidate-reuse / baseline-reuse | parity | 0.982 | 0.984 | 0.983 | 0.2% | 3.6% | 1.019 | within band | 1.013 |
| concurrent-phone-4032x3024-420 | candidate-reuse / zune-reuse | lead | 0.787 | 0.786 | 0.784 | 0.3% | 4.4% | 0.821 | - | 0.842 |
| concurrent-phone-4032x3024-420 | candidate-image-adapter / image-builtin | lead | 0.782 | 0.788 | 0.776 | 1.2% | 10.3% | 0.862 | - | 0.830 |

## Allocation reference (x86_64)

Candidate rows of `ref-zen3-1`. All four x86_64 runs agree exactly.

| case | row | allocs | cumulative | peak live |
|---|---|---:|---:|---:|
| small-64x64-420 | candidate-fresh | 19 | 36.7 KiB | 36.5 KiB |
| small-64x64-420 | candidate-reuse | 18 | 24.7 KiB | 24.5 KiB |
| small-64x64-420 | candidate-image-adapter | 27 | 44.7 KiB | 26.4 KiB |
| phone-4032x3024-420 | candidate-fresh | 19 | 52.36 MiB | 52.36 MiB |
| phone-4032x3024-420 | candidate-reuse | 18 | 17.48 MiB | 17.48 MiB |
| phone-4032x3024-420 | candidate-image-adapter | 27 | 20.45 MiB | 20.43 MiB |
| phone-4032x3024-420-scale-1/4 | candidate-fresh | 15 | 4.38 MiB | 4.38 MiB |
| phone-4032x3024-420-scale-1/4 | candidate-reuse | 14 | 2.20 MiB | 2.20 MiB |
| gray-1920x1080 | candidate-fresh | 8 | 1.99 MiB | 1.99 MiB |
| gray-1920x1080 | candidate-reuse | 8 | 1.99 MiB | 1.99 MiB |
| gray-1920x1080 | candidate-image-adapter | 15 | 2.42 MiB | 2.41 MiB |
| progressive-1920x1080-420 | candidate-fresh | 55 | 15.00 MiB | 9.06 MiB |
| progressive-1920x1080-420 | candidate-reuse | 54 | 9.06 MiB | 9.06 MiB |
| progressive-1920x1080-420 | candidate-image-adapter | 80 | 9.57 MiB | 9.52 MiB |
| large-7680x4320-420 | candidate-fresh | 19 | 142.43 MiB | 142.43 MiB |
| large-7680x4320-420 | candidate-reuse | 18 | 47.51 MiB | 47.51 MiB |
| large-7680x4320-420 | candidate-image-adapter | 27 | 55.56 MiB | 55.55 MiB |
| testorig | candidate-fresh | 19 | 174.7 KiB | 174.5 KiB |
| testorig | candidate-reuse | 18 | 75.6 KiB | 75.4 KiB |
| testorig | candidate-image-adapter | 27 | 99.4 KiB | 81.1 KiB |
| testimgint | candidate-fresh | 19 | 174.7 KiB | 174.5 KiB |
| testimgint | candidate-reuse | 18 | 75.6 KiB | 75.4 KiB |
| testimgint | candidate-image-adapter | 27 | 99.4 KiB | 81.1 KiB |
| encode-64x64 | candidate | 14 | 21.8 KiB | 19.8 KiB |
| encode-64x64 | candidate-444 | 11 | 12.3 KiB | 10.8 KiB |
| encode-64x64 | candidate-image-adapter | 15 | 23.4 KiB | 19.8 KiB |
| encode-1920x1080 | candidate | 13 | 4.43 MiB | 4.33 MiB |
| encode-1920x1080 | candidate-444 | 11 | 4.54 MiB | 4.50 MiB |
| encode-1920x1080 | candidate-image-adapter | 14 | 4.80 MiB | 4.33 MiB |
| encode-4032x3024 | candidate | 13 | 25.63 MiB | 25.41 MiB |
| encode-4032x3024 | candidate-444 | 11 | 26.53 MiB | 26.43 MiB |
| encode-4032x3024 | candidate-image-adapter | 14 | 27.78 MiB | 25.41 MiB |
| thumbnail | candidate | 48 | 99.31 MiB | 69.79 MiB |
| thumbnail | candidate-scaled-decode | 38 | 3.39 MiB | 2.16 MiB |
| thumbnail | candidate-image-adapter | 81 | 102.34 MiB | 69.77 MiB |

## Concurrent decode (x86_64, Zen 3)

Four threads each decode the 12 MP 4:2:0 photo four times, 16 decodes per
batch, from `ref-zen3-1`.
- **Peak live heap** counts every thread together through the global
  allocator. It is not RSS.
- **Caller buffers** are the reuse rows' four output buffers, allocated
  before the measurement window.
- **Heap + buffers** is what the application holds, so it is the figure to
  compare across rows.

| row | median ms / batch | MP/s | allocs | peak live heap | caller buffers | heap + buffers |
|---|---:|---:|---:|---:|---:|---:|
| baseline-fresh | 276.3 | 706 | 317 | 209.44 MiB | 0.0 KiB | 209.44 MiB |
| baseline-reuse | 270.3 | 722 | 301 | 69.90 MiB | 139.54 MiB | 209.44 MiB |
| candidate-fresh | 291.9 | 668 | 317 | 209.44 MiB | 0.0 KiB | 209.44 MiB |
| candidate-reuse | 265.5 | 735 | 301 | 69.90 MiB | 139.54 MiB | 209.44 MiB |
| candidate-image-adapter | 265.8 | 734 | 445 | 81.73 MiB | 139.54 MiB | 221.27 MiB |
| image-builtin | 340.1 | 574 | 317 | 14.42 MiB | 139.54 MiB | 153.96 MiB |
| zune-fresh | 341.8 | 571 | 301 | 142.13 MiB | 0.0 KiB | 142.13 MiB |
| zune-reuse | 337.2 | 579 | 285 | 2.59 MiB | 139.54 MiB | 142.13 MiB |


## Size and build reference

Contribution of each backend to an otherwise empty stock-profile binary
(unstripped bytes over the empty probe). It is identical in all four x86_64
runs. Clean build times come from `ref-zen3-1` and are indicative only.

| backend | x86_64 bytes | aarch64 bytes (contaminated run) | x86_64 clean build |
|---|---:|---:|---:|
| libjpeg-turbo-rs 0.8.0 | 625,472 | 467,104 | 12.5 s |
| candidate | 717,344 | 514,368 | 13.6 s |
| candidate + image adapter (+ `image` codec, see README) | 824,592 | 539,600 | 29.1 s |
| `image` (jpeg feature) | 360,672 | 223,792 | 19.4 s |
| zune-jpeg | 217,464 | 173,248 | 1.8 s |

## Where the candidate loses

- **Memory under concurrency.** Four concurrent 12 MP decodes hold
  209.4 MiB with the candidate. zune-jpeg holds 142.1 MiB and `image`'s
  codec 154.0 MiB: 67 MiB more than zune-jpeg, all of it whole-image
  component planes. A buffer-reuse decode still holds 69.9 MiB of heap
  against zune-jpeg's 2.6 MiB.
  [P4-218](../../docs/last_mile/phase4.md#p4-218-the-buffer-reuse-decode-still-allocates-whole-image-component-planes--open).
- **Fresh decode of large images vs 0.8.0:**
  - *8K fresh:* 1.028 / 1.028 / 1.032 here, and 1.024–1.050 in the five
    other runs (two earlier consumer builds and Zen 4). That is under the
    3 % floor, but it never changed sign in eight runs.
  - *Concurrent fresh:* 1.057 / 1.059 / 1.023. The reuse rows are at parity.

  The cause was the eager zero-fill in `try_filled_vec`, and it is fixed.
  Two dispatches of the fix against this reference put 8K fresh at
  1.008 / 1.000 and concurrent fresh at 0.997 / 1.002.
  [P4-228](../../docs/last_mile/phase4.md#p4-228-fresh-decode-of-large-images-is-25--slower-than-080--closed-2026-10-07)
  ([#659](https://github.com/developer0hye/libjpeg-turbo-rs/issues/659)).
- **1/4-scaled decode, buffer-reuse vs 0.8.0: 1.024–1.039,** behind by the
  rule. The same library measured 0.90 with the earlier consumer build, so
  this row is binary-sensitive. It is a losing case of this reference, not
  yet a diagnosed regression.
- **Zen 4, one run:**
  - 4:2:0 encode at 1.053–1.071× 0.8.0;
  - 8K fresh at 1.050× and 8K reuse at 1.033×.

  Not confirmed: it is one run, and Zen 4 runners are not the reference.
- **Image adapter vs `image`'s codec on a 64x64 image:** a fixed per-image
  overhead; see the reference table (27 vs 19 allocations).
- **Binary size vs 0.8.0:** +14.7 % (x86_64) and +10.1 % (aarch64), not yet
  attributed.
  [P4-230](../../docs/last_mile/phase4.md#p4-230-the-candidate-adds-1015--more-code-to-a-stock-profile-binary-than-080--open)
  ([#661](https://github.com/developer0hye/libjpeg-turbo-rs/issues/661)).
- **Small-image decode vs zune-jpeg:** a tie at 64x64. Every larger decode
  case leads by 14–28 % in each reference run.

## Why the band has a cross-run term

The first two x86_64 dispatches (`reports/2026-10-07-x86_64-linux*/`, runs
37542461917 and 37543406684) used an earlier consumer build, before the
concurrent section and the full feature line. Their within-run spreads were
0.1–8 %, and they agreed with each other. They put the thumbnail workload at
1.060× 0.8.0 and the 1/4-scale pair at 0.90.

Two runs of the next consumer build, against the same library, put the
thumbnail at 1.016 and 1.003. The 1/4-scale pair moved to 1.005 and 1.205.
One run's p10–p90 therefore says little about the next run, and a single
binary's quirks can masquerade as library regressions. That is why the band
includes the cross-run range, why the reference is three runs, and why
reports are matched on consumer sources. The thumbnail claim first filed as
P4-228 was withdrawn.

