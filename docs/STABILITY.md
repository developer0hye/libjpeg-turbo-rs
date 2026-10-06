# Stability and Support Policy

What a release promises, per crate. The security half — supported versions,
backports and how to report — is in [`SECURITY.md`](../SECURITY.md).

## Versioning

Every crate follows [Semantic Versioning](https://semver.org/) as Cargo
applies it to `0.x` versions:

- **`0.MINOR.PATCH`**: a *minor* bump (`0.8` → `0.9`) may change the public API
  in incompatible ways; a *patch* bump (`0.8.0` → `0.8.1`) may not. Cargo's
  default `^0.8` requirement therefore picks up patches and never a new minor.
- A breaking change is marked **Breaking** in its `CHANGELOG.md` entry, with
  the migration, in the release that makes it.
- Each crates.io crate is checked by `cargo-semver-checks` against the same
  crate at the previous release tag before anything is published
  (`.github/workflows/release.yml`), so a breaking change under a patch number
  fails before upload. The check covers type signatures of a default-feature
  build, not behaviour and not `png`-gated items; a behaviour change that
  breaks callers is also marked **Breaking**. The npm package is not checked.

The crates version independently: `libjpeg-turbo-rs` (root), `libjpeg-turbo-rs-image`
(the `image` adapter), `libjpeg-turbo-rs-capi` (C ABI) and the npm
`libjpeg-turbo-rs-wasm`. An adapter or C-ABI release names the root version it
was tested with.

### What is public API

- **Covered:** the items re-exported at the crate root (`Decoder`, `Encoder`,
  `decompress*`, `compress*`, `probe`, `TransformOp`, `JpegError`, the types in
  `common::types`, `StreamingDecoder`, `CompressParams`,
  `compress_with_params`, ...) and the named re-export modules `tj3`,
  `precision`, `quantize`, `raw_data_12`, `yuv` and `stream` (with `std`).
- **Not covered:** the low-level modules `api`, `common`, `decode`, `encode`,
  `simd` and `transform` are `pub` for historical reasons and expose pipeline
  and kernel internals. [`PUBLIC_API_REVIEW.md`](PUBLIC_API_REVIEW.md)
  classifies their items; the supported ones are re-exported at the root, and
  the rest will be narrowed in a minor release (P4-222). Reaching into them is
  at your own risk, and the changelog will say when they change.
- **Never covered:** exact error *messages* — match on the variant, not the
  text.
- The C ABI (`libjpeg-turbo-rs-capi`) follows the libjpeg/TurboJPEG ABI it
  implements; its compatibility contract is
  [`docs/ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md).

## Minimum supported Rust version (MSRV)

| Crate | MSRV | Why |
|---|---|---|
| `libjpeg-turbo-rs` | 1.87 | |
| `libjpeg-turbo-rs-capi` | 1.87 | |
| `libjpeg-turbo-rs-image` | 1.88 | `image 0.25` requires it |

CI checks (`cargo check`) each crate with exactly its MSRV toolchain.
Raising an MSRV is a **minor** change, never a patch, and is listed in
`CHANGELOG.md`.

## Cargo features

`std` and `simd` (both default) and `png` are stable: removing one, or
changing what a default build includes, is a breaking change. `simd` selects
the hand-written NEON/SSE2/AVX2/SIMD128 kernels (AVX2 also needs `std` for
runtime detection; SIMD128 needs the `simd128` target feature); without it the
scalar kernels run. `full-c-parity` is internal and test-only: it gates no
library code and is not covered. The C cross-validation suites run on
`simd`-enabled builds only; there the scalar kernels are held to the SIMD
output by `tests/no_std_dispatch.rs` (scalar path forced with
`JSIMD_FORCENONE`) and `src/simd/simd_parity_tests.rs` (scalar and SIMD
kernels called side by side).

## Deprecation

An item that will be removed is first marked `#[deprecated(since, note)]`
naming its replacement, in a release before the one that removes it.
Removal happens in a minor (pre-1.0) or major (post-1.0) release.

## Errors

`JpegError` is `#[non_exhaustive]`: new variants are added in minor *or*
patch releases (a new refusal can be a bug fix), so a `match` on it needs a
wildcard arm. Changing which variant an existing input produces is a
behaviour change and is called out in the changelog.

## Threads

`Decoder`, `Encoder` and `Image` are `Send` (compile-time checked in
`tests/concurrency.rs` and `tests/decode_pipeline_public_api.rs`). The codec
holds no global mutable state apart from one-time initialised read-only
tables, so independent decoders and encoders on separate threads do not
interact. Use one decoder or encoder per thread.

## Resource limits

`DecodeLimits::default()` allows dimensions up to 65,500, up to 2³¹−1 pixels
(about 2.1 gigapixels), 8,192 scans, and no memory ceiling. That accepts
every file `djpeg` accepts in the corpus gates; it refuses only the
pathological corner `djpeg` does not cap by default (`djpeg.c` sets
`max_scans = 0`): frames above 2³¹−1 pixels such as 65,500×65,500, and
streams of more than 8,192 scans. This is deliberate. A default that
refused what `djpeg` accepts would break drop-in use, and the project
decided against shipping separate "untrusted" and "compatible" default
profiles (PR #516).
An application that decodes untrusted uploads should set its own budget —
`Decoder::set_max_pixels`, `set_max_memory` and `set_scan_limit`, or
`DecodeLimits` — before decoding. The 12-/16-bit `precision::decompress_*`
functions take no limits and apply only the default width, height and pixel
checks.

On the `decompress` / `Decoder` paths under `src/decode/`, geometry-sized
allocations report `JpegError::AllocationFailed` instead of aborting the
process (P4-209, held by `tests/decode_alloc_gate.rs`). Not yet: the 12-/16-bit
entry points in `src/api`, which a 12-bit source also reaches through
`Decoder` (P4-216), the lenient-decode warning list (P4-215), and the
encoder's allocations; an allocator refusal there still aborts.

Changing these defaults would be a new policy decision, made in a minor
release and called out as **Breaking**.
