# Adoption Guide

**Last reviewed:** 2026-10-07, against `main@edf3566`.

This page helps you choose an integration path, decide what to pin, evaluate
the codec on your own images, and roll it out with a way back. It defines no
readiness of its own: readiness lives in [`LAST_MILE.md`](LAST_MILE.md)
(Current Status, tiers T1–T4), and release promises in
[`STABILITY.md`](STABILITY.md) and [`../SECURITY.md`](../SECURITY.md). Where
this page and those disagree, they win.

## 1. Choose a path

| You have | Use | Status (canonical source) |
| --- | --- | --- |
| Rust code you can change | `libjpeg-turbo-rs` (root crate) | **T1** in [`LAST_MILE.md`](LAST_MILE.md): ready for use, not a memory-safety guarantee |
| Rust code built on `image`'s `ImageDecoder` / `ImageEncoder` | `libjpeg-turbo-rs-image` | Opt-in adapter; its [README](../crates/libjpeg-turbo-rs-image/README.md) says which `image` entry points it covers |
| A browser or WASI target | `libjpeg-turbo-rs-wasm` (npm), or the root crate on `wasm32` | [wasm README](../crates/libjpeg-turbo-rs-wasm/README.md); SIMD128 needs an explicit flag (§6) |
| C/C++ using TurboJPEG 3 (`tj3*`) | `libjpeg-turbo-rs-capi` as `libturbojpeg` | **T2**: the primary C target |
| C/C++ compiled against classic libjpeg **v8** | `libjpeg-turbo-rs-capi` as `libjpeg.so.8` | **T3**: experimental and partial; controlled pilots only |
| A binary compiled against libjpeg **v6b or v7** | Stay on upstream, or rebuild against TJ3 / v8 | **T4**: unsupported; the struct layouts differ, so substitution corrupts memory |

The tiers are independent: a green TurboJPEG path says nothing about classic
v8, and a v8 result says nothing about v6b/v7.

## 2. What is published, and what is not

- **Nothing merged since 0.8.0 has been published.** The crates.io set is root
  `0.8.0`, adapter `0.1.0` and C ABI `0.1.2`, all from the `v0.8.0` tree. The
  next root release is **0.9.0**, a breaking minor:
  [`RELEASE.md`](RELEASE.md#recorded-comparisons) records the API comparison,
  and the `[Unreleased]` section of [`../CHANGELOG.md`](../CHANGELOG.md) lists
  the breaking items. The adapter's next release (0.2.0) is breaking too.
- **The published packages carry known memory-safety and abort findings that
  are fixed only on `main`.** Read
  [`security/AFFECTED_VERSIONS.md`](security/AFFECTED_VERSIONS.md) before
  pinning: [§2a](security/AFFECTED_VERSIONS.md#2a-safe-rust-api-libjpeg-turbo-rs)
  covers the root crate (P4-135, P4-136 and P4-192 all affect the current
  release) and [§2b](security/AFFECTED_VERSIONS.md#2b-c-abi-libjpeg-turbo-rs-capi)
  the C ABI, where some rows are still open on `main` as well.
- **No GitHub release has shipped a native bundle yet.** The pipeline that
  builds, checksums and attests them (Sigstore provenance, CycloneDX SBOM) is
  described in [`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md); the latest
  release, `v0.8.0`, predates it and has no assets.

So, today, choose one of:

1. **wait for 0.9.0** (the recommendation for production on untrusted input);
2. **pin a reviewed `main` commit** as a git dependency, with `Cargo.lock`
   committed and every build run with `--locked`;
3. **pin 0.8.0** only after reading AFFECTED_VERSIONS and confirming none of
   its rows is reachable in your use.

## 3. Readiness in one paragraph

As of 2026-10-07 no *known* safe-API memory-safety defect is open: the last
one, P4-192 (#610, a custom scan script writing past stack arrays on x86_64),
closed that day. That is an absence of reports, not a guarantee: P4-139
(layout arithmetic still partly decentralised) and P4-141 (the soundness
verification program) are both PARTIAL, and the findings so far came from
reading code rather than from a gate. The full release-mode workspace run is
red on
[P4-170](last_mile/phase4.md#p4-170-classic-source-manager-parity-fails-in---release-and-passes-in-debug-so-ci-never-sees-it--open)
(classic C ABI). The evidence is the T1 bullet and live-gate table in
[`LAST_MILE.md`](LAST_MILE.md); every `unsafe` item is listed in
[`UNSAFE_INVENTORY.md`](UNSAFE_INVENTORY.md) and
[`UNSAFE_INVENTORY_CAPI.md`](UNSAFE_INVENTORY_CAPI.md).

## 4. Rust API

```sh
cargo add libjpeg-turbo-rs
```

The [README](../README.md) quick start and [`examples/`](../examples/README.md)
cover decode, encode, transforms, metadata and streaming; docs.rs has the
API. [`STABILITY.md`](STABILITY.md) states the SemVer, MSRV, feature, error
and threading policy. Two points matter most when adopting:

- **Only the root re-exports and the named re-export modules are covered
  API.** The `pub` low-level modules (`api`, `common`, `decode`, `encode`,
  `simd`, `transform`) are not, and 0.9.0 already changes some of them. The
  review that classifies their items (P4-222, PR
  [#649](https://github.com/developer0hye/libjpeg-turbo-rs/pull/649)) is
  [`PUBLIC_API_REVIEW.md`](PUBLIC_API_REVIEW.md).
- **Set your own resource budget for untrusted input.** The defaults accept
  what `djpeg` accepts and set no memory ceiling, by design
  ([`STABILITY.md#resource-limits`](STABILITY.md#resource-limits)). Pass the
  limits at construction, because the scan cap applies while markers are
  parsed:

```rust
use libjpeg_turbo_rs::{DecodeLimits, Decoder, PixelFormat};

let limits = DecodeLimits {
    max_memory: Some(512 * 1024 * 1024),
    ..DecodeLimits::strict()
};
let mut decoder = Decoder::new_with_limits(&jpeg_bytes, limits)?;
decoder.set_output_format(PixelFormat::Rgb);
let image = decoder.decode_image()?;
```

The 12-/16-bit `precision::decompress_*` functions take no limits. Which
allocations report `JpegError::AllocationFailed` and which still abort is in
the same STABILITY section; the open gaps are in §8.

## 5. `image` adapter

Use `libjpeg-turbo-rs-image` when your code is written against `image`'s
traits. It does not register itself: `image::open` and friends keep `image`'s
built-in codec, and you construct `JpegDecoder` / `JpegEncoder` explicitly.
The adapter [README](../crates/libjpeg-turbo-rs-image/README.md) is the
contract (construction, `read_image`, `set_limits`, metadata, error mapping,
corrupt data), and `examples/thumbnail_pipeline.rs` there is a complete
migration.

That README describes `main`, which will ship as adapter 0.2.0. **Published
0.1.0 decodes the whole image inside `JpegDecoder::new` and ignores
`image::Limits`** (P4-212 in
[AFFECTED_VERSIONS §2a](security/AFFECTED_VERSIONS.md#2a-safe-rust-api-libjpeg-turbo-rs)),
so do not rely on limits with 0.1.0.

## 6. WebAssembly

This repository's `.cargo/config.toml` enables `+simd128` for in-tree builds
only. A consumer outside the repository must pass
`RUSTFLAGS="-C target-feature=+simd128"`, or the SIMD kernels compile out and
the codec silently runs scalar. Measure module size, instantiation, first
call versus steady state, and linear-memory growth on the runtimes you ship.

## 7. C ABI

**TurboJPEG 3 (T2).** Prefer `tj3*`: opaque handles avoid struct-layout risk.
Check every symbol you use in [`C_API_REFERENCE.md`](C_API_REFERENCE.md).
Legacy 1.x/2.x aliases are partial; [`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md)
has the migration matrix and a one-file shim recipe for the missing ones.
Note P4-199 (#620): the 12-/16-bit decompress paths ignore the handle's
`MAXPIXELS`, `MAXMEMORY` and `SCANLIMIT`.

**Classic v8 (T3).** Pilot only, and only for consumers compiled against
`JPEG_LIB_VERSION = 80`. Before a pilot:

- inventory the `jpeg_*` symbols you import and check each against the open
  classic-ABI items in [`LAST_MILE.md`](LAST_MILE.md#open-items);
- know that the OpenCV evidence tests the cargo cdylib, not the relinked
  library the installer ships
  ([P4-124](last_mile/phase4.md#p4-124-the-opencv-harness-tests-the-cargo-cdylib-not-the-library-we-ship--open)),
  and that GNU symbol versions are
  [P4-81](last_mile/phase4.md#p4-81-linux-cdylib-omits-gnu-libjpeg_80-symbol-versions--partial-nodes-emitted-and-tested-downstream-re-verification-pending);
- threading follows upstream: one thread at a time per `cinfo`
  ([threading contract](ABI_COMPATIBILITY.md#threading-contract));
- load the library from an isolated prefix or process, never as a global
  system replacement, and keep upstream selectable.

Stop on a v6b/v7 consumer, a required symbol that is partial or missing, or a
malformed input whose termination, memory or error behaviour is worse than
upstream's.

Until a release ships native bundles, build from a pinned commit as the
[C-ABI crate README](../crates/libjpeg-turbo-rs-capi/README.md) describes.

## 8. Known gaps that affect adopters

Open on `main` (the [OPEN Items table](LAST_MILE.md#open-items) is the full list):

| Item | Effect |
| --- | --- |
| [P4-236](last_mile/phase4.md#p4-236-encoder-with-non-standard-sampling_factors-silently-drops-progressive-arithmetic-lossless-and-restart-options--open) | `Encoder` with non-standard `sampling_factors` (e.g. 3x2) writes a baseline stream, silently dropping progressive, arithmetic, lossless and restart options. |
| [P4-213](last_mile/phase4.md#p4-213-cmykycck-12-bit-and-lossless-decodes-stage-a-full-size-copy-even-when-given-a-caller-buffer--open) | CMYK/YCCK, 12-bit and lossless decodes stage a full-size copy even into a caller buffer, and `max_memory` does not count it. |
| [P4-216](last_mile/phase4.md#p4-216-the-12-16-bit-decode-entry-points-in-srcapi-still-allocate-geometry-sized-buffers-infallibly--open) | 12-/16-bit decode entry points abort on allocator refusal. |
| [P4-219](last_mile/phase4.md#p4-219-12-bit-decodes-ignore-the-horizontal-crop-and-tjhandles-1216-bit-decompress-ignores-the-cropping-region-entirely--open) | 12-bit decodes ignore the horizontal crop. |
| [P4-221](last_mile/phase4.md#p4-221-encode-path-allocations-are-all-infallible-and-no-gate-tracks-them--open) | Encode-path allocations abort on allocator refusal. |
| [P4-217](last_mile/phase4.md#p4-217-no-written-release-semver-msrv-or-security-policy-no-private-reporting-route-and-no-api-check-before-publish--partial-policy-api-gate-and-affected-version-record-landed-private-reporting-route-not-enabled-oss-fuzz-not-submitted) | GitHub private vulnerability reporting is not enabled yet; [`SECURITY.md`](../SECURITY.md) gives the fallback. |
| [P4-170](last_mile/phase4.md#p4-170-classic-source-manager-parity-fails-in---release-and-passes-in-debug-so-ci-never-sees-it--open) | Classic source-manager parity fails in `--release` builds. |

Also open on `main`, filed by the pull requests linked:

- **P4-218** ([#646](https://github.com/developer0hye/libjpeg-turbo-rs/pull/646)):
  `decompress_into` still allocates whole-image component planes, so the
  buffer-reuse path is not a low-memory path.
- **P4-230** ([#661](https://github.com/developer0hye/libjpeg-turbo-rs/issues/661)):
  the crate adds 10–15 % more code to a release binary than 0.8.0 did.
- **P4-222** ([#649](https://github.com/developer0hye/libjpeg-turbo-rs/pull/649)):
  the low-level modules are de facto API until they are narrowed.
- **P4-220** ([#645](https://github.com/developer0hye/libjpeg-turbo-rs/pull/645)):
  TJ3 entry points never check the handle's instance type.

Architectures: there is no AArch32 NEON backend
([P4-78](last_mile/phase4.md#p4-78-no-32-bit-arm-aarch32-neon-backend--armv7-is-our-widest-gap-vs-c--open)),
and the RISC-V Vector backend is measured under emulation only
([P4-134](last_mile/phase4.md#p4-134-no-risc-v-rvv-simd-backend--upstream-32-ships-one--partial-measured-under-emulation-only-hardware-measurement-outstanding)).
Do not assume x86_64/aarch64 results transfer.

## 9. Evaluate on your own corpus

**Correctness.** Decode your images with this crate and with stock `djpeg`
from the libjpeg-turbo you run today, using matching options (output format,
DCT method, fancy upsampling, scaling, crop). Pixel-identical output is the
project's contract and what CI checks
([`TEST_PARITY.md`](TEST_PARITY.md), [`CORPUS_TEST_REPORT.md`](CORPUS_TEST_REPORT.md));
for encode, compare by decoding both outputs, since compressed bytes may
legitimately differ. Include progressive, grayscale, CMYK/YCCK, restart
markers, EXIF orientation, and the truncated or corrupt files your service
actually receives. Corrupt data is an error by default; lenient decoding is
opt-in.

**Limits.** Feed oversized and adversarial headers with your intended budget
set, and confirm they are refused before allocation instead of aborting.
Check §8 if you decode 12-bit, lossless or CMYK input.

**Performance.** Benchmark your build, not this workspace's: Cargo ignores a
dependency's `[profile.release]`, so this workspace's `lto = true` does not
reach your application, and the README's tables are in-tree builds. The
standalone default-profile consumer harness (`experiments/downstream/`)
measures that build. Its first report and the losing cases are in
[`experiments/downstream/BUDGETS.md`](../experiments/downstream/BUDGETS.md).
It ran on x86_64 hosted runners; no aarch64 figures are budget-grade yet
(P4-229). Still measure inside your application with the release profile you
ship, portable and `target-cpu=native` separately, and record CPU,
toolchain, flags and corpus with every number. Measure peak memory as well as
latency (P4-218).

## 10. Roll out and roll back

1. Run the corpus comparison offline; keep a small redistributable subset as
   a CI canary.
2. Shadow production: decode with both codecs, serve the old result, and
   alert on pixel, error-class, latency or memory differences.
3. Ship behind a switch to a small share of traffic, then widen.
4. Keep the previous codec buildable until the new one has carried full
   traffic through a release cycle.

For Rust, rollback is a dependency or feature-flag change, so keep the old
path compiled in until you remove it on purpose. For the C ABI, keep the
upstream library installed and select by loader path per process; never
overwrite the system library.

## 11. Report problems

Suspected vulnerabilities go through [`SECURITY.md`](../SECURITY.md), never a
public issue. For anything else, open an issue with the crate and version (or
commit), target and features, the API calls, a redistributable input, and the
`djpeg` or upstream result you compared against.
