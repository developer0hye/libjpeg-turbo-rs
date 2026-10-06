# Stability and Support Policy

What a release promises, per crate. The security half — supported versions,
backports and how to report — is in [`SECURITY.md`](../SECURITY.md).

## Versioning

Every crate follows [Semantic Versioning](https://semver.org/) as Cargo
applies it to `0.x` versions:

- **`0.MINOR.PATCH`**: a *minor* bump (`0.8` → `0.9`) may change the public API
  in incompatible ways; a *patch* bump (`0.8.0` → `0.8.1`) may not. Cargo's
  default `^0.8` requirement therefore picks up patches and never a new minor.
- A breaking change is listed under `CHANGELOG.md` → **Breaking** with the
  migration, in the release that makes it.
- Each release is checked by `cargo-semver-checks` against the previous
  crates.io release of the same crate before anything is published
  (`.github/workflows/release.yml`), so a breaking change under a patch number
  fails before upload. The check covers type signatures, not behaviour; a
  behaviour change that breaks callers is also called out as **Breaking**.

The crates version independently: `libjpeg-turbo-rs` (root), `libjpeg-turbo-rs-image`
(the `image` adapter), `libjpeg-turbo-rs-capi` (C ABI) and the npm
`libjpeg-turbo-rs-wasm`. An adapter or C-ABI release names the root version it
was tested with.

### What is public API

- **Covered:** the items re-exported at the crate root (`Decoder`, `Encoder`,
  `decompress*`, `compress*`, `probe`, `TransformOp`, `JpegError`, the types in
  `common::types`, ...) and the named re-export modules `tj3`, `precision`,
  `quantize` and `raw_data_12`.
- **Not covered yet:** the low-level modules `api`, `common`, `decode`,
  `encode`, `simd` and `transform` are `pub` for historical reasons and expose
  pipeline and kernel internals. Until the pre-1.0 surface review decides
  which of their items are supported, reaching into them is at your own risk:
  a minor release may change or hide them, and the changelog will say so.
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

CI builds each crate with exactly its MSRV toolchain. Raising an MSRV is a
**minor** change, never a patch, and is listed in `CHANGELOG.md`.

## Cargo features

`std` and `simd` (both default) and `png` are stable: removing one, or
changing what a default build includes, is a breaking change. `simd` selects
the hand-written NEON/SSE2/AVX2/SIMD128 kernels; without it the scalar
kernels run. The integer paths are cross-validated against C either way.

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

`DecodeLimits::default()` accepts every frame C libjpeg-turbo's `djpeg`
accepts: dimensions up to 65,500, about 2.1 gigapixels, 8,192 scans, and no
memory ceiling. This is deliberate. A default that refused what `djpeg`
accepts would break drop-in use, and the project decided against shipping
separate "untrusted" and "compatible" default profiles (PR #516).
An application that decodes untrusted uploads should set its own budget —
`Decoder::set_max_pixels`, `set_max_memory` and `set_scan_limit`, or
`DecodeLimits` — before decoding. Allocations whose size comes from the input
report `JpegError::AllocationFailed` instead of aborting the process.

Changing these defaults would be a new policy decision, made in a minor
release and called out as **Breaking**.
