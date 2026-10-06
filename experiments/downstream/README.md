# Downstream-consumer benchmark

P4-214 · GitHub [#640](https://github.com/developer0hye/libjpeg-turbo-rs/issues/640)

This benchmark measures libjpeg-turbo-rs the way an application gets it: a
separate crate, built with Cargo's stock `release` profile and no `RUSTFLAGS`.
It compares that build with the published release and with the pure-Rust
alternatives an application would otherwise pick.

## Why a separate crate

Every other benchmark in this repository runs inside the workspace, so it
inherits two settings no downstream user gets:

- **`[profile.release] lto = true`** in the root `Cargo.toml`. Cargo applies
  profiles only from the workspace root being built and ignores the profiles
  of dependencies ("Cargo only looks at the profile settings in the
  `Cargo.toml` manifest at the root of the workspace. Profile settings defined
  in dependencies will be ignored." —
  <https://doc.rust-lang.org/cargo/reference/profiles.html>). An application
  that depends on `libjpeg-turbo-rs` therefore builds it with `lto = false`
  and 16 codegen units unless the application sets its own profile.
- **`.cargo/config.toml`**, which Cargo reads from every ancestor of the
  directory it builds in (<https://doc.rust-lang.org/cargo/reference/config.html>).

`cargo bench` and `examples/bench_zune_matrix.rs` inherit both, and
`bench_zune_matrix`'s `parity = ok` checks only the output length.

`consumer/` has its own empty `[workspace]` table and no `[profile]` section.
`run.sh` copies it **outside** the repository before building, so neither
setting can reach it.

## What is compared

| Row | What it is |
|---|---|
| `baseline-*` | `libjpeg-turbo-rs = "=0.8.0"` from crates.io, the published release |
| `candidate-*` | this checkout (path dependency, substituted by `run.sh`) |
| `candidate-image-adapter` | this checkout's `crates/libjpeg-turbo-rs-image` through `image`'s `ImageDecoder` / `ImageEncoder` traits |
| `image-builtin` | `image` 0.25's own `codecs::jpeg` (zune-jpeg decodes; `image` encodes) |
| `zune-*` | `zune-jpeg` called directly |

Versions come from the committed `consumer/Cargo.lock`, and every report lists
them. Every row produces the same pixel layout for a case: RGB8, or L8 for the
grayscale case. Each report row also states who owns the output buffer:

- **fresh** rows (`decompress_to`, `zune decode`) return a library-owned `Vec`
  per decode.
- **buffer-reuse** rows (`decompress_into`, `zune decode_into`,
  `ImageDecoder::read_image`) write into one caller-owned buffer that is
  allocated before timing starts. A new decoder is still built for every
  decode, because neither library can reuse a decoder.

### Cases

| Case | Input |
|---|---|
| `small-64x64-420` | 64×64 synthetic, 4:2:0 |
| `phone-4032x3024-420` | 12 MP synthetic "phone photo", q90 4:2:0 |
| `phone-4032x3024-420-scale-1/4` | the same file at DCT scale 1/4 (zune and `image` have no scaled decode, so those rows are N/A) |
| `gray-1920x1080` | grayscale 1080p |
| `progressive-1920x1080-420` | progressive 1080p, 4:2:0 |
| `large-7680x4320-420` | 33 MP (8K UHD), 4:2:0 |
| `testorig`, `testimgint` | upstream `references/libjpeg-turbo/testimages/`, under the IJG License (see that directory's `LICENSE.txt`) |
| `encode-*` | 64×64, 1920×1080 and 4032×3024 RGB, encoded at quality 85 |
| thumbnail | 12 MP photo with EXIF orientation 6: decode → orient → `imageops::resize` (Triangle) to 256 px → encode q85 |

`consumer/src/corpus.rs` generates the synthetic content using integer
arithmetic only, so the pixels are identical on every platform. The **published
baseline** encodes it, never the candidate, so a change to the candidate's
encoder cannot change the decode inputs. Each report records every input
file's SHA-256 and writes the files to `report/corpus/`.

### Correctness checks

These run outside every timed region.

- **Run-aborting invariants.** Two paths through the same library must produce
  identical bytes. The pairs are candidate `decompress_into` vs
  `decompress_to`, the image adapter vs candidate `decompress_to`, baseline
  reuse vs fresh, zune `decode_into` vs `decode`, and the adapter's encoder vs
  candidate `compress`. All rows must also report the same dimensions.
- **Reported, not failed.** Every row's maximum and mean absolute difference
  against `candidate-fresh` is reported. A nonzero candidate-vs-baseline
  difference is a behaviour change to explain. zune-jpeg does not claim
  libjpeg-identical output, so its difference is reported but never fails the
  run.
- **Candidate vs C (asserted).** C libjpeg-turbo is the contract, so when a
  C tool is available a difference **fails the run**:
  - decode: candidate `decompress_to` and baseline 0.8.0 must be
    pixel-identical to `djpeg` on every case, including `-grayscale` and
    `-scale 1/4`. Both matched stock 3.2.0 and Homebrew 3.1.4.1 on all eight
    cases on 2026-10-07;
  - encode: candidate and baseline `compress` must be byte-identical to
    `cjpeg -quality 85` (4:2:0) and to `cjpeg -quality 85 -sample 1x1` for the
    `-444` rows, fed the same pixels as a PPM. A failure says whether the
    streams still differ once APP0 is removed, so a JFIF-header-only
    difference is reported as one rather than loosened silently.
    `image-builtin` makes no libjpeg-compatibility claim and is not compared.

  Tool selection: `--djpeg` / `--cjpeg`, else `DJPEG` / `CJPEG`, else the
  first of `/opt/homebrew/bin`, `/opt/libjpeg-turbo/bin` and `/usr/bin` that
  has the tool. `/usr/local` is never probed, because this repository's own
  C-ABI shim gets installed there. The report records each tool's resolved
  (canonical) path, its `-version` line and its link map (`otool -L` plus
  `LC_RPATH`, or `ldd`), so it shows which libjpeg actually produced the C
  output. Without a tool, the report says so explicitly ("C encode
  comparison: skipped (no cjpeg)").

  The first local smoke run (2026-10-07) compared against Homebrew's 3.1.4.1.
  Later runs used a stock 3.2.0 build, passed as
  `DJPEG=/Volumes/T7/scratch/ljt320/prefix/bin/djpeg
  CJPEG=/Volumes/T7/scratch/ljt320/prefix/bin/cjpeg` (a local build on the
  maintainer's machine). `--no-c-oracle` turns off both tools. The workflow
  passes it, because it provisions no C reference and a tool that happens
  to be on a runner image is an unpinned release.
- **Encode PSNR.** PSNR is measured against the source pixels. The published
  baseline decodes every row's output. Each output's real subsampling is read
  back from its SOF marker. `image`'s encoder writes **4:4:4** at q85 even
  though its doc comments say 4:2:2, so its bytes and PSNR do not compare
  directly with the 4:2:0 rows. The `baseline-444` and `candidate-444` rows
  encode at 4:4:4 to give a like-for-like comparison.

## Reproducing

```sh
git submodule update --init references/libjpeg-turbo   # testorig.jpg / testimgint.jpg
experiments/downstream/run.sh --check                   # fmt, clippy -D warnings, unit tests
experiments/downstream/run.sh --iterations 30 --warmup 5
```

On this project's development Mac, put the work dir on the external SSD when
it is mounted, e.g.
`DOWNSTREAM_WORK_DIR=/Volumes/T7/scratch/downstream-<variant>`. The seven
clean builds (main plus six probes) use several hundred MB of target
directories while they run. `run.sh` deletes them once it has the binary and
the sizes, so a finished work dir holds only the copied consumer,
`Cargo.lock.committed`, the binary, `build-info.txt` and the report (about
18 MB on the 2026-10-07 aarch64 smoke run: 15 MB of report, almost all of
it `report/corpus/`, plus a 2.7 MB binary).

`run.sh` does the following:

1. Copies `consumer/` to a fresh directory outside the repository (a
   `mktemp -d` directory, or `DOWNSTREAM_WORK_DIR` if that is set and empty).
2. Replaces `@CANDIDATE@` in the manifest with this checkout's absolute path,
   and refuses a path containing `|`, `&`, `\` or `"`. It also refuses if the
   manifest has gained a `[profile]` section.
3. Refuses to continue if the copied consumer directory or any of its
   ancestors has a `.cargo/config.toml` (or legacy `.cargo/config`).
   `CARGO_HOME`'s config is the one
   exception, because it applies to every build on the machine, downstream
   builds included. `run.sh` refuses it too if it sets a profile, rustflags,
   a rustc wrapper or `[target]` options.
4. Unsets every `CARGO_PROFILE_*`, `CARGO_BUILD_*` (except `CARGO_BUILD_JOBS`),
   `CARGO_TARGET_*`, `CARGO_INCREMENTAL`, `RUSTC_WRAPPER`,
   `RUSTC_WORKSPACE_WRAPPER`, `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`,
   `RUSTDOCFLAGS` and `CARGO_ENCODED_RUSTDOCFLAGS`, and records which ones
   it cleared. It also records the remaining `CARGO_*` / `RUST*` environment
   except `CARGO_HOME`, dropping credential-named variables and any value
   that is a URL with userinfo. Only the *shape* of `CARGO_HOME`'s config
   goes into `build-info.txt`: table and key names, with every value redacted
   except `[build] jobs` and `[net] offline`.
5. Runs `cargo fetch --locked` so downloads are not counted as build time,
   then `cargo build --release --locked` in a fresh target directory. It
   records the clean build time and the binary size.
6. Builds the six size probes (`probe-none`, `-baseline`, `-candidate`,
   `-adapter`, `-image`, `-zune`), each in its own fresh target directory. A
   probe's size minus `probe-none`'s is that backend's contribution to a stock
   binary. `probe-adapter` includes the candidate and the adapter, and — because the harness's `adapter` feature also enables `image-builtin` — `image`'s own JPEG codec, so its build time and size overstate what an adapter-only application builds; subtract the `image` probe for an estimate.
   Set `SKIP_PROBES=1` to skip this step.
7. Deletes the target directories, then runs the harness from the consumer
   directory, so its `rustc -Vv` resolves the same toolchain the build used.
   The harness writes `report.md`, `report.json` and `corpus/` to
   `DOWNSTREAM_OUT_DIR`, which defaults to `$WORK/report`; `run.sh` then
   copies `build-info.txt` next to them.

`run.sh --check` stops after step 4. Instead of benchmarking, it runs
`cargo fmt --check`, `cargo clippy --locked --release --all-targets -- -D
warnings` and `cargo test --locked --release` on the copied consumer.
`downstream-bench.yml` runs this first, in a separate work dir, and `ci.yml`'s
`downstream-consumer` job runs it on every PR so a candidate API change that
breaks the consumer fails there rather than at the next dispatch.

Build variants are separate, labelled runs. The default is the stock profile.

| `VARIANT` | Sets |
|---|---|
| `default` | nothing (stock `release`) |
| `thin-lto` | `CARGO_PROFILE_RELEASE_LTO=thin` |
| `fat-lto` | `CARGO_PROFILE_RELEASE_LTO=fat` |
| `native` | `RUSTFLAGS=-Ctarget-cpu=native` |

Harness options: `--iterations N`, `--warmup N`, `--smoke` (2 iterations after
1 warmup; this proves the harness works and measures nothing), `--only
<substring>` (case filter), `--djpeg <path>`, `--cjpeg <path>`,
`--no-c-oracle`.

**Before a measured run,** follow the timing-experiment rules in the global
CLAUDE.md: profile the machine first, wait until two consecutive samples are
quiet, and run nothing else. The harness records a `top` and memory sample
just before it starts, but recording load does not remove it.

### Regenerating the lock

The committed `consumer/Cargo.lock` is the reviewed dependency set, and every
build uses `--locked`. If a candidate change adds a dependency or bumps its
own version, the lock no longer matches and `run.sh` stops. Regenerate the
lock **offline** from the workspace lock, so the consumer never resolves a
version the workspace has not already reviewed:

```sh
work=$(mktemp -d)
cp -R experiments/downstream/consumer/. "$work"
sed -i.bak "s|@CANDIDATE@|$PWD|g" "$work/Cargo.toml"
cp Cargo.lock "$work/Cargo.lock"
(cd "$work" && cargo metadata --offline --format-version 1 >/dev/null)
# Packages in the consumer lock that the workspace lock does not have:
list() { awk '/^\[\[package\]\]/{if(n)print n,v,s; n=v=s=""} /^name = /{n=$3} /^version = /{v=$3} /^source = /{s=$3} END{if(n)print n,v,s}' "$1" | sort; }
comm -13 <(list Cargo.lock) <(list "$work/Cargo.lock")
```

The only packages that may be new are the path crates and
`libjpeg-turbo-rs 0.8.0` from the registry. Anything else must pass the
supply-chain review (it must be at least 7 days old and must not look like a
typosquat) before you copy the lock back.

## Reading the report

- **Timing columns** are milliseconds per operation. The timed region covers
  decoder construction, the decode itself, and dropping a library-owned output.
  `median` is the headline figure. `p10`/`p90` (nearest rank) and `min`/`max`
  show the spread. A row whose fastest warmup call took under 1 ms is timed
  in samples of K back-to-back calls, divided by K. The median is then marked
  `(×K)`, and the JSON field is `calls_per_sample`. This keeps timer
  resolution and cold-call effects (first-touch page faults, cold branch
  predictors) out of the small-image rows. The cost is that such a row
  reports warm, repeated-call speed, not the first call an application makes
  after start-up. MP/s is computed over **source** pixels, so the 1/4-scaled
  row's MP/s compares directly with the full decode of the same file.
- **Each case runs its rows interleaved.** Every backend is warmed up, then
  each round times every row once, starting from a different row each round.
  Slow drift in machine load therefore lands on every row. Compare rows within
  one report, not numbers taken from two reports.
- **Allocation columns** come from one extra pass under the counting global
  allocator: allocation events, cumulative bytes requested (a `realloc` counts
  its growth only), and peak live heap above the live heap at the start. The
  allocator counts only inside that pass. In timed regions it costs one
  relaxed atomic load per call, the same for every backend. These numbers are
  deterministic, so machine load does not affect them.
- **N/A** means the backend has no API for the case. The reason appears in the
  row.

## Committing a report

To commit a report under `experiments/`, commit `report.md`, `report.json`
and `build-info.txt`. Leave `corpus/` out: it is regenerated byte for byte
(the report records each file's SHA-256), and the 8K input alone is about
8 MB.

## Regression budget

[`BUDGETS.md`](BUDGETS.md) applies these rules to the first measured report
(#640 criterion 6) and lists the cases the candidate loses;
`budgets.py <report.json> --first reports/2026-10-07-x86_64-linux/report.json`
checks a later report's timing ratios. The rules, fixed before any data existed:

- **Time:** for each row, the relative spread `(p90 - p10) / median` of the
  first report on each runner gives the noise floor. A candidate row regresses
  when its median exceeds the baseline row from the **same run** by more than
  `max(2 × spread, 3 %)`, and the excess reproduces in a second run.
- **Allocations:** these are deterministic, so the budget is zero. Any
  increase in count or peak for the same case is a change to explain.
- **Binary size and build time:** the probe contributions in the first report
  become the reference. Growth over 5 % needs a stated reason.

## Hosted-runner noise

The `workflow_dispatch` job `.github/workflows/downstream-bench.yml` produces
an x86_64 (ubuntu-latest) and an arm64 (macos-latest) report. Shared runners
are noisy: the CPU model can change between runs (each report records it),
turbo and frequency governors cannot be pinned, and neighbouring jobs share
the host. Expect run-to-run swings of several percent and read only
within-run ratios. A regression
claim made from runner data needs either a quiet local machine or repeated
dispatches that agree.

GitHub does not register a dispatch-only workflow until the file exists on
the default branch, so a branch can dispatch it only once `main` has it.
