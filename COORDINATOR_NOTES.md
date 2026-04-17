# Coordinator Notes — Mission B1 (CI C-tool enforcement)

## Status

Branch: `feat/b1-ci-c-tools-required` (4 commits; not pushed).

| Commit    | Subtask | Summary                                                                                |
| --------- | ------- | -------------------------------------------------------------------------------------- |
| `d0eaf23` | B1-1    | CI: install `libjpeg-turbo-progs` before `cargo test --tests` and verify on PATH.      |
| `4220cf3` | B1-2    | Add `tests/helpers/c_tools.rs` with `is_ci()` + `require_c_tool()` std-only helpers.   |
| `591d7c7` | B1-3    | Add `require_c_tool!(name)` macro (CI = panic, local = SKIP+return).                   |
| `6e51ffd` | B1-4    | Mechanically rewrite 15 test files (97 sites) from silent-skip match blocks to macro.  |

B1-5 verification: `cargo test` (both `--test helpers_smoke` and the 15
rewritten files) passes locally with no regressions; full `cargo test
--tests` exits with every `test result: ok. N passed; 0 failed` line
visible — no failing suites attributable to this branch.

## Remaining B1 work (NOT done in this session)

Roughly **83 test files** declare their **own** local `djpeg_path()` /
`cjpeg_path()` / `jpegtran_path()` helpers instead of `mod helpers;`.
They contain about **330** additional silent-skip sites that look like:

```rust
let djpeg = match djpeg_path() {          // no `helpers::` prefix
    Some(p) => p,
    None => {
        eprintln!("SKIP: djpeg not found");
        return;
    }
};
```

Converting these requires, per file:

1. Add `mod helpers;` at the top.
2. Delete the local `fn djpeg_path()` / `cjpeg_path()` / etc. definitions
   (typically 10–20 lines each).
3. Run the same mechanical `match ... { Some(p) => p, None => { ... } };` →
   `require_c_tool!("TOOL")` rewrite as B1-4 used.
4. Fix any fallout warnings (unused `use std::process::Command;`, etc.).

Files needing this follow-up (99 total callsite-files minus the 15
already rewritten in B1-4 minus `tests/helpers_smoke.rs` which is
meta):

```
arithmetic.rs                 cross_check_color_quantize.rs    progressive_enc.rs
bitstream_structure.rs        cross_check_coeff_filter.rs      progressive_output.rs
c_indexedcolortest.rs         cross_check_crop_scale.rs        progressive_scan_edge_cases.rs
color_convert.rs              cross_check_dct_decode.rs        quantize.rs
conformance.rs                cross_check_encoder_options.rs   raw_data.rs
copy_mode.rs                  cross_check_extreme_dims.rs      real_world_images.rs
crop_c_compat.rs              cross_check_lossless.rs          reference_image_compat.rs
crop_skip.rs                  cross_check_merged_upsample_formats.rs
cross_check_12bit.rs          cross_check_metadata.rs          restart_byte_unit.rs
cross_encoder_compat.rs       cross_check_metadata_edge.rs     restart_encode.rs
cross_encode_decode.rs        cross_check_misc_gaps.rs         rgb565_dither.rs
cross_product_compress.rs     cross_check_pixel_format_decode.rs
cross_product_decompress.rs   cross_check_pixel_format_encode.rs  s441_encode.rs
cross_product_transform.rs    cross_check_per_quality.rs       scale_decode.rs
custom_huffman.rs             cross_check_progressive_scans.rs scaling_extended.rs
custom_quant.rs               cross_check_raw_decompress.rs    scanline_api.rs
custom_sampling.rs            cross_check_scanline_options.rs  sof10_decode.rs
custom_scan.rs                cross_check_skip_scanlines.rs    sof10_encode.rs
decode_toggles.rs             cross_check_stream_io.rs         sof11.rs
diverse_resolutions.rs        cross_check_tj3.rs               stream_io.rs
easy_wrappers.rs              cross_check_transform.rs         streaming.rs
edge_case_inputs.rs           cross_check_transform_options.rs subsamp_410.rs
encode_boundaries.rs          cross_check_yuv_decompress.rs    subsampling_encode.rs
encoder_builder.rs            image_io.rs                      tjunittest_compat.rs
error_recovery.rs             lossless_decode.rs               tjunittest_transform.rs
huff_opt.rs                   lossless_encode.rs               tjunittest_yuv.rs
marker_preservation.rs        merged_upsample.rs               transform.rs
niche_options.rs              per_quality.rs                   upsample.rs
pixel_formats.rs              yuv_api.rs
```

(Partial list of 83; full enumeration available via
`rg 'fn djpeg_path|fn cjpeg_path|fn jpegtran_path|fn rdjpgcom_path' tests/`.)

Suggested sequencing for a follow-up worker:

1. Commit 1 — add `mod helpers;` and remove the local `_path()` fn in each
   file (keep behavior identical via the still-present helpers::djpeg_path
   re-export).  Confirm `cargo check --tests` is clean.
2. Commit 2 — run the same mechanical rewrite (`match ... { None => {
   eprintln!("SKIP: TOOL not found..."); return; } };` → `require_c_tool!`)
   once every file routes through `helpers::`.
3. Commit 3 — delete now-unused `djpeg_path()` / `cjpeg_path()` /
   `jpegtran_path()` / `rdjpgcom_path()` re-exports from
   `tests/helpers/mod.rs` if no remaining caller references them.

### WORKSPACE_CARGO_ADDITIONS

None — B1 used only `std` (no new crate deps).

## Guardrails acknowledgement

Changes touched only files owned by this worker:

* `.github/workflows/ci.yml` — coordinator-owned (this worker).
* `tests/helpers/mod.rs` — coordinator-owned (this worker).
* `tests/helpers/c_tools.rs` — new coordinator-owned module.
* `tests/helpers_smoke.rs` — tests for the coordinator-owned helpers.
* 15 integration test `tests/*.rs` files listed under B1-4.

Not touched: workspace-root `Cargo.toml`, `README.md`,
`tests/reference_hashes*.json`, `docs/FEATURE_PARITY.md`,
`docs/C_API_REFERENCE.md`, `CLAUDE.md`, `METHODOLOGY.md`,
`FORMATTING.md`, `.githooks/*`, `references/*`.
