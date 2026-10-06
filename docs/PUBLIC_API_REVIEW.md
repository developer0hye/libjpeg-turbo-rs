# Public API Review (pre-1.0)

**Date:** 2026-10-07 · **State reviewed:** `origin/main@c6e792f` (crate 0.8.0) ·
**Tracking:** P4-222 (`docs/last_mile/phase4.md`), #635 Milestone E.

`src/lib.rs` re-exports the supported API at the crate root, but it also
declares six low-level modules `pub` — `api`, `common`, `decode`, `encode`,
`simd`, `transform` — which expose pipeline stages, kernels and tables. This
review classifies what those modules expose so that a breaking release can
narrow them deliberately. [`STABILITY.md`](STABILITY.md#what-is-public-api)
states the resulting promise.

Counts and users below are from a `git grep` of the workspace at the reviewed
commit: "external" means outside the root crate's `src/`.

## Classification

| Module | Not re-exported at the root | External users | Class |
|---|---|---|---|
| `api` | 14 items: `yuv::*` (8 fns), `streaming::StreamingDecoder`, `encoder::RestartConfig`, `quality::scale_quant_table_linear`, `high_level::compress_lossless_extended_precision`, `precision::compress_{12,16}bit_with_precision` | capi (`yuv`), benches (`StreamingDecoder`), 15 test files | **Supported, now at the root:** `yuv::*`, `StreamingDecoder`. The rest internal. |
| `common` | 29 items: `layout::{ImageLayout, checked_span}`, `quant_table::{ZIGZAG_ORDER, NATURAL_ORDER, QuantTable}`, `huffman_table`, `tables`, `arith_tables`, `icc`, `exif` | capi (`layout` ×4, `NATURAL_ORDER`), 5 test files | **Internal, needed by workspace crates:** `layout`, `quant_table` orders. Rest internal. |
| `decode` | 75 items across 16 submodules (kernels, entropy, marker reader, toggles) | capi (`boundary`), one example, 8 test files (25, some overlapping, reach root items via `decode::pipeline`) | **Internal, needed by workspace crates:** `boundary`. Kernels, `toggles` (P4-80) internal. |
| `encode` | 86 items (not counting the 11 `common::tables` constants `encode::tables` re-exports): `pipeline` (31 incl. `CompressParams`), `marker_writer` (30), `fdct`, `color`, … | capi (`pipeline`), benches (`compute_reciprocal`), 4 examples, 11 test files; **README** imported `compress_with_params`/`CompressParams` | **Supported, now at the root:** `CompressParams`, `compress_with_params`. Other `pipeline` fns: internal, needed by capi. Rest internal. |
| `simd` | `detect`, `detect_encoder`, `SimdRoutines`, `EncoderSimdRoutines`, `QuantDivisors` (backends already `pub(crate)` since P4-135) | benches, `tests/simd_dispatch_bounds.rs` | **Internal** (benches and tests only). |
| `transform` | `TransformInfo`, `spatial::*` (18) | none | **Internal**; the three root types stay supported. |

No public signature or field of a root-exported type names a module-only
type, so narrowing the modules leaves no root item that downstream code cannot
name. One more path exposes everything: the capi crate's
`pub use libjpeg_turbo_rs as inner;` (`crates/libjpeg-turbo-rs-capi/src/lib.rs`).

## Done in this change (additive)

- `libjpeg_turbo_rs::yuv::{encode_yuv, encode_yuv_planes, compress_from_yuv,
  compress_from_yuv_planes, decompress_to_yuv, decompress_to_yuv_planes,
  decode_yuv, decode_yuv_planes}`, `libjpeg_turbo_rs::StreamingDecoder` and
  `libjpeg_turbo_rs::{CompressParams, compress_with_params}` are supported
  root paths, pinned by `tests/public_surface_root_paths.rs`. README now
  imports from the root.

## Remaining (breaking — P4-222)

1. Workspace-internal items (`common::layout`, `common::quant_table` orders,
   `decode::boundary`, the `encode::pipeline` functions capi calls) move
   behind one `#[doc(hidden)]` namespace documented as unstable, so the capi
   crate keeps compiling while users stop seeing them.
2. Everything classified **internal** becomes `pub(crate)`. Benches and the
   38 test files that name module-only items (22 of them reach items
   classified internal), plus the 28 that reach root items through module paths such as
   `decode::pipeline::Decoder`, switch to root paths, to the hidden
   namespace, or to `#[cfg(test)]` unit tests. `tests/decode_pipeline_public_api.rs`
   and `tests/encode_pipeline_public_api.rs`, which pin the module paths
   today, are rewritten to pin the root paths.
3. capi drops `pub use libjpeg_turbo_rs as inner`, or narrows it to what its
   own tests need.
4. `cargo-semver-checks` against the previous release records the change,
   and the CHANGELOG entry lists each moved path with its replacement.
