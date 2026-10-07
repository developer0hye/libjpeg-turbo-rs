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
`DecodeLimits` — before decoding; for 12- and 16-bit frames use
`precision::decompress_12bit_with_limits` / `decompress_16bit_with_limits`
(the plain `decompress_12bit` / `decompress_16bit` apply the defaults).

On the `decompress` / `Decoder` paths under `src/decode/`, geometry-sized
allocations report `JpegError::AllocationFailed` instead of aborting the
process (P4-209, held by `tests/decode_alloc_gate.rs`). Not yet: the 12-/16-bit
entry points in `src/api`, which a 12-bit source also reaches through
`Decoder` (P4-216), the lenient-decode warning list (P4-215), and the
encoder's allocations; an allocator refusal there still aborts.

Changing these defaults would be a new policy decision, made in a minor
release and called out as **Breaking**.

### Limits and how they are applied


Untrusted input should be decoded under an explicit budget.
`DecodeLimits` holds it; `Decoder::new_with_limits` (or `set_limits`,
`set_max_pixels`, `set_max_memory`, `set_scan_limit`) applies it to the 8-bit
pipeline, `precision::decompress_12bit_with_limits` /
`decompress_16bit_with_limits` to the 12- and 16-bit decoders, and a
`TjHandle` builds one from `TJPARAM_MAXPIXELS`, `TJPARAM_MAXMEMORY` (in
megabytes) and `TJPARAM_SCANLIMIT` for `decompress`, `decompress_12bit` and
`decompress_16bit`. The C ABI's `tj3DecompressToYUV8` /
`tj3DecompressToYUVPlanes8` apply `TJPARAM_MAXPIXELS` and `TJPARAM_SCANLIMIT`
through `TjHandle::inspect_header`, not `TJPARAM_MAXMEMORY`: they decode
through the handle-free `decompress_to_yuv_planes`. Every refusal is
`JpegError::LimitExceeded`, raised before any buffer is sized from the
header: `max_scans` during the header walk when the limits are given before
it (`Decoder::new_with_limits`, the `_with_limits` functions, a `TjHandle`),
and when the decode starts when they are set on an already parsed `Decoder`
(`set_scan_limit`, `set_limits`), whose walk ran under the default 8,192; the
others when the decode starts.

| limit | default | what it bounds |
|---|---|---|
| `max_width`, `max_height` | 65,500 (`JPEG_MAX_DIMENSION`) | the SOF's dimensions |
| `max_pixels` | 2³¹−1 (`TJPARAM_MAXPIXELS` 0: none) | SOF width × height, before scaling or cropping |
| `max_scans` | 8,192 (`TJPARAM_SCANLIMIT` 0: none) | SOS segments the header walk finds — the walk itself stops at the cap. An interleaved sequential stream ends the walk at its only SOS, so the cap matters for progressive and multi-scan streams |
| `max_memory` | none (`TJPARAM_MAXMEMORY` 0: none) | an *estimate* of the geometry-sized buffers listed below |

### What the memory budget covers

`max_memory` is compared with an estimate computed from the frame header, not
with what the allocator reports. Each decode path estimates its own buffers:

| path | counted |
|---|---|
| 8-bit `Decoder` / `decompress*` / `TjHandle::decompress` | the output buffer (width × height × output bytes per pixel, owned or caller-supplied); one byte per pixel per component for the component planes; one more full-size plane per subsampled component of an RGB-colour-space frame, or for a subsampled luma plane when YCbCr is decoded to grayscale; for a progressive frame, the coefficient buffer (two bytes per pixel per component plus one byte per 8×8 block) |
| `decompress_12bit_with_limits`, `TjHandle::decompress_12bit` | the per-component planes at their padded iMCU size, the upsampled full-size planes and the interleaved result, two bytes a sample each |
| `decompress_16bit_with_limits`, `TjHandle::decompress_16bit` | the output, plus for a multi-component frame the full-size planes it is interleaved from, two bytes a sample each |
| `TjHandle::decompress_to_yuv_planes` (`tj3DecompressToYUV8` / `tj3DecompressToYUVPlanes8`) | the 8-bit `Decoder` estimate above, as for an RGB decode: the raw path shares its check (P4-225) |
| `tj3Transform` (C ABI only) | what stock's memory manager realizes before the first scan: the source's whole-image coefficient arrays, each component padded to its sampling factors, plus every transform's workspace — none for an in-place flip, the source's size for a vertical flip or 180-degree rotation, transposed for the transpose family, luma only under `TJXOPT_GRAY` — at 128 bytes a block; refused at `estimate >= budget`, which matches stock at the measured boundaries (P4-227) |

Not counted anywhere:

- the compressed input, which the decoder borrows (a C-ABI caller owns it);
- metadata copies: the ICC profile, EXIF, XMP, IPTC, comments and saved
  markers on the `Image`, and the ICC copy a `TjHandle` keeps;
- the full-size staging copy CMYK/YCCK, 12-bit and lossless decodes make even
  when given a caller buffer (P4-213), a vertical crop's copy of the output,
  and `TjHandle`'s `TJPARAM_BOTTOMUP` row flip;
- the 12-bit staging when a 12-bit frame is decoded through `Decoder` /
  `decompress`, which checks only the 8-bit estimate above (P4-224);
- per-row and per-block scratch;
- the warning strings a lenient decode collects (P4-215);
- the classic `jpeg_*` API, which has its own `max_memory_to_use` (P4-14).

The budget is therefore stricter than libjpeg-turbo's in one way and looser in
another. TurboJPEG's `TJPARAM_MAXMEMORY` reaches only the memory manager's
whole-image arrays (`jmemmgr.c` `realize_virt_arrays`, which raises
`JERR_NO_BACKING_STORE`) — never the caller's destination — so a baseline
decode stock TurboJPEG accepts can be refused here. A TurboJPEG handle
refuses an over-budget frame before it decodes, where stock TurboJPEG does so
during the decode; both have published the frame's parameters first. The same
holds for `TJPARAM_SCANLIMIT`: the handle reads the header up to the first
SOS and publishes, then refuses while locating the remaining scans, where
stock refuses from its progress monitor mid-decode. `tj3DecompressHeader`
applies neither limit, as upstream's does not. `tj3Transform` is the
exception to "stricter": its estimate counts what stock's memory manager
counts, so the two refuse the same sources except within stock's few KiB of
pool overhead, which the estimate does not model.
