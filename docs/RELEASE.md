# Release Procedure

How a version of the crates.io crates and the npm package is cut. The
policies a release has to respect — SemVer, MSRV, supported versions — are in
[`STABILITY.md`](STABILITY.md) and [`../SECURITY.md`](../SECURITY.md); what the
native bundles contain is in [`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md).

## 1. Choose the version numbers from evidence

```sh
cargo install cargo-semver-checks --locked --version 0.50.0
scripts/semver_check_release.sh <previous-tag>
```

Run it after bumping the versions in the manifests: it compares each crate's
public API with the previous tag and fails if the bump does not allow what
changed (a breaking change under a 0.x patch number). `release.yml` runs the
same script before any upload, so a wrong choice stops the release rather
than shipping. A breaking *behaviour* change the tool cannot see is still
marked **Breaking** in the changelog and still needs a minor bump.

Each crate has its own version. Bump only the crates that changed; a crate
whose version did not move is skipped by both the check and the publish job.
`libjpeg-turbo-rs-image` and `libjpeg-turbo-rs-capi` must require the root
version they were tested with (`version = "…"` next to `path = "../.."`).

## 2. Write the changelog section

Move the `## [Unreleased]` entries under `## [X.Y.Z] - YYYY-MM-DD`.
`release.yml`'s `changelog-check` job refuses a `v*` tag without that section.
Security fixes are marked **Security** and name the affected versions.

## 3. Rehearse

```sh
gh workflow run release.yml --ref <branch>
```

A dispatch builds and attests the native bundles and runs the API check
against the latest release tag; it publishes nothing.

## 4. Tag

Push `vX.Y.Z` from a commit on `main` whose CI is green. The tag first runs
`changelog-check`, `semver-check` and the native bundle builds; then it
publishes the root crate, then the C-ABI crate followed by the `image`
adapter, with the npm package publishing alongside them once the root crate
is up; the GitHub Release with the native bundles comes last, after all
four. `capi-vX.Y.Z` ships only the C-ABI crate, `wasm-vX.Y.Z` only the npm
package.

## 5. Verify what was published

The published crate is what users get, not the tree that was tagged:

- from a scratch directory **outside** this repository (Cargo reads
  `.cargo/config.toml` from every parent directory), `cargo new consumer`,
  add the released crates by version, build with `--locked` after the first
  resolution, and run the adapter's `examples/thumbnail_pipeline.rs` against
  the registry packages;
- `gh attestation verify` one native bundle as `RELEASE_ARTIFACTS.md`
  describes;
- if the release fixes a vulnerability, publish the GitHub advisory and
  request a RustSec advisory (`SECURITY.md`).

## Recorded comparisons

### `main@7f9e5e5` against `v0.8.0` (2026-10-07)

`libjpeg-turbo-rs`, compared with `cargo rustdoc --locked` output on both
sides: **7 breaking checks fail**, so the next root release is **0.9.0**, not
0.8.1. Every item but one is in the low-level modules
[`STABILITY.md`](STABILITY.md#what-is-public-api) does not yet cover. The
exception is `JpegCoefficients::saw_jfif_marker`: `JpegCoefficients` is
re-exported at the crate root, so a public field added to it breaks covered
API for any caller that builds it with a struct literal:

| Check | Items |
|---|---|
| `function_missing` | 40+ `simd::aarch64::*` and `simd::scalar::*` kernel entry points |
| `struct_missing` | `encode::huffman_encode::BitWriter` |
| `struct_pub_field_missing` | `SimdRoutines::{ycbcr_to_rgb_row, fancy_upsample_h2v1}`, `EncoderSimdRoutines::rgb_to_ycbcr_row` |
| `constructible_struct_adds_field` | `JpegCoefficients::saw_jfif_marker`, `JpegMetadata::{adobe_transform_at_first_sos, saw_jfif_marker_at_first_sos}`, `EncoderSimdRoutines::fdct_float_quantize` |
| `function_parameter_count_changed`, `inherent_method_missing`, `module_missing` | further `simd`-module items |

The changelog's `## [Unreleased]` also records behaviour-level breaks the
tool cannot see (for example `TjHandle::new()` no longer defaulting quality
and subsampling, P4-155).
