# FFI B9-4: Stock djpeg/cjpeg/jpegtran link against our shim — findings

## Executive summary

**Linking stock `djpeg`, `cjpeg`, `jpegtran` against our cdylib
(`liblibjpeg_turbo_rs_capi.dylib` / `libjpeg.so.62`) FAILS at link time
with 22+ missing-symbol errors per binary on macOS (arm64). Identical
outcome expected on Linux because the cause is API-surface coverage,
not platform.**

Host: `arm64-apple-darwin25.4.0`, Apple clang 21.0.0. Reproduced via
`examples/stock_djpeg_cjpeg/build.sh` and the direct `cc` invocations
recorded below. All three logs live next to the build artifacts:

- `examples/stock_djpeg_cjpeg/build/djpeg_link.log`
- `examples/stock_djpeg_cjpeg/build/cjpeg_link.log`
- `examples/stock_djpeg_cjpeg/build/jpegtran_link.log`

## Root cause

`crates/libjpeg-turbo-rs-capi` exports only the **TurboJPEG** API
(`tj3*`, `tj*`). It exports **zero** classic libjpeg (`jpeg_*`) symbols.
Stock `djpeg`/`cjpeg`/`jpegtran` call the classic API directly; they
do not use TurboJPEG. A drop-in replacement for the stock `djpeg`/
`cjpeg`/`jpegtran` binaries therefore requires a separate parallel
API surface.

Verified via:

```
$ rg "pub extern \"C\" fn (\w+)" crates/libjpeg-turbo-rs-capi/src
… only tj3*/tj* functions …
```

## Missing-symbol inventory (as of FFI C2-6)

A1-11 (decode) and C2-1..C2-5 (encode) have landed the following
previously-missing symbols. Items marked `[x]` are exported by the
current cdylib; `[ ]` still block a stock-tool link.

### Classic libjpeg API (required by djpeg + cjpeg + jpegtran)

Core init/destroy:

- `[x]` `jpeg_std_error` (A1-11)
- `[x]` `jpeg_CreateCompress` (C2-1), `[x]` `jpeg_CreateDecompress` (A1-11)
- `[x]` `jpeg_destroy_compress` (C2-1), `[x]` `jpeg_destroy_decompress` (A1-11)

I/O setup:

- `[x]` `jpeg_stdio_src` (A1-11), `[x]` `jpeg_stdio_dest` (C2-1)
- `[x]` `jpeg_mem_src` (A1-11), `[x]` `jpeg_mem_dest` (C2-1)

Decompress flow:

- `[x]` `jpeg_read_header` (A1-11)
- `[x]` `jpeg_start_decompress` (A1-11), `[x]` `jpeg_finish_decompress` (A1-11)
- `[x]` `jpeg_read_scanlines` (A1-11)
- `[ ]` `jpeg_skip_scanlines`, `jpeg_crop_scanline`
- `[ ]` `jpeg_read_icc_profile`
- `[ ]` `jpeg_save_markers`, `jpeg_set_marker_processor`

Compress flow:

- `[x]` `jpeg_set_defaults` (C2-1), `[x]` `jpeg_default_colorspace` (C2-1),
  `[x]` `jpeg_set_colorspace` (C2-1)
- `[x]` `jpeg_start_compress` (C2-2), `[x]` `jpeg_finish_compress` (C2-2)
- `[x]` `jpeg_write_scanlines` (C2-2), `[x]` `jpeg_write_icc_profile` (C2-4)
- `[x]` `jpeg_simple_progression` (C2-3), `[x]` `jpeg_enable_lossless` (C2-3)
- `[x]` `jpeg_set_quality` (C2-1)
- `[x]` `jpeg_suppress_tables` (C2-3)

Transform-specific (jpegtran):

- `[ ]` `jpeg_read_coefficients` (decode-side virtual barray; follow-up)
- `[x]` `jpeg_write_coefficients` (C2-5 stub — records gap error)
- `[ ]` `jpeg_copy_critical_parameters`
- `[ ]` `jpeg_core_output_dimensions`
- `[x]` `jpeg_write_marker` (C2-4), `[x]` `jpeg_write_m_header` (C2-4),
  `[x]` `jpeg_write_m_byte` (C2-4)
- `[x]` `jpeg_write_tables` (C2-4)
- `[x]` `jpeg_resync_to_restart` (C2-5)

Quant/huff helpers (via rdswitch.c):

- `[x]` `jpeg_add_quant_table` (C2-3), `[x]` `jpeg_default_qtables` (C2-3),
  `[x]` `jpeg_quality_scaling` (C2-3)

### 12-bit / 16-bit precision variants (djpeg/cjpeg)

- `[ ]` `jpeg12_read_scanlines`, `jpeg12_skip_scanlines`, `jpeg12_crop_scanline`
- `[x]` `jpeg12_write_scanlines` (C2-5)
- `[ ]` `jpeg16_read_scanlines`
- `[x]` `jpeg16_write_scanlines` (C2-5)

Also `[ ]` `read_color_map_12` — a wrapper defined in the libjpeg-turbo
12-bit wrapper shim (`src/wrapper/rdcolmap-12.c`) that our build does
not emit.

### Internal helpers referenced from transupp.c (jpegtran only)

- `[x]` `jcopy_block_row` (C2-5)
- `[x]` `jdiv_round_up` (C2-5)

Both live in `jutils.c` / `jdatadst.c` inside stock libjpeg-turbo and are
exported as part of `libjpeg.so.62` despite being "internal" utilities.

### Remaining blockers for stock-tool link

After C2-* the outstanding gaps are:

1. Decode-side skip/crop scanline (`jpeg_skip_scanlines`,
   `jpeg_crop_scanline`), marker-processor hooks, and
   `jpeg_read_icc_profile` — needed by djpeg's crop flag and
   JPEG-ICC pass-through, and by jpegtran for marker preservation.
2. Full `jpeg_read_coefficients` + `jpeg_copy_critical_parameters` +
   `jpeg_core_output_dimensions` — required by jpegtran's virtual-
   barray transform pipeline. `jpeg_write_coefficients` ships today as
   a stub; the full barray exchange lands in a follow-up.
3. 12/16-bit read-side scanline entry points — needed by
   `cjpeg -precision 12/16` input and djpeg 12/16-bit output.
4. `read_color_map_12` wrapper for the `-map` flag on 12-bit streams.

## Why this is a real gap, not a build configuration mistake

1. The cc-compile step succeeds for every stock-tool .c source once
   `jconfig.h` / `jconfigint.h` / `jversion.h` are provided. The failure
   is **exclusively** at final link time.
2. The missing symbols are all in the exact shape the public
   `jpeglib.h` declares them (leading `_jpeg_` on macOS = unprefixed
   `jpeg_` on ELF). Our shim's `nm` output does not list a single
   `_jpeg_*` symbol, only `_tj*`.
3. The internal helpers (`jcopy_block_row`, `jdiv_round_up`) are
   declared in the libjpeg-turbo public headers (`jpegint.h`) even
   though they are not formally part of the "API". Stock consumers
   that include `transupp.c` (jpegtran, the tj3Transform shim itself)
   transitively require them.

## Path to fulfilling B9-4

To land this mission green, the shim must additionally expose a classic
libjpeg API layer. Outline:

1. New module `crates/libjpeg-turbo-rs-capi/src/classic.rs` exporting
   each of the ~40 `jpeg_*` symbols listed above with `#[no_mangle] pub
   extern "C"` and the libjpeg calling convention.
2. A matching `struct jpeg_compress_struct` / `struct
   jpeg_decompress_struct` opaque layout compatible with the copies
   stock consumers compile against — must match `sizeof()` / field
   layout of `jpeglib.h` so callers can pass stack-allocated structs in.
3. Error-manager / marker-processor function-pointer vtables must be
   ABI-compatible (C callbacks into Rust).
4. Internal helpers (`jcopy_block_row`, `jdiv_round_up`) also need
   `#[no_mangle]` exports.
5. 12-bit / 16-bit precision variants gate on `BITS_IN_JSAMPLE`; either
   always export all three (`jpeg_`, `jpeg12_`, `jpeg16_`) or emit a
   per-precision cdylib.

Scope estimate: ~40 symbols × ~20 LOC each = ~1000 LOC of wrapper.
The underlying Rust `libjpeg_turbo_rs` crate already implements the
equivalent operations under different signatures, so the work is
purely mechanical ABI bridging.

## Byte-exactness aspiration (once linking works)

Even after the link layer is added, **byte-exact parity is realistic
only for decode** (djpeg). Encode (cjpeg) will likely diverge because
stock libjpeg-turbo uses a fixed quantization table ordering, specific
Huffman-optimization heuristics, and a precise DCT rounding mode that
differ slightly from our Rust encoder. The scripts in this directory
already implement the correct fallback: on encode byte-difference,
roundtrip both outputs through stock djpeg and require pixel-identical
recovered PPMs (equivalent to PSNR = ∞). If that still fails, a PSNR
threshold ≥ 50 dB is acceptable per the mission statement.

Transform (jpegtran) byte-exactness is mostly about coefficient copies
and marker preservation; realistic once classic API parity exists.

## Deliverables in this commit

- `examples/stock_djpeg_cjpeg/build.sh` — configures + invokes `cc`
  to link stock djpeg/cjpeg/jpegtran against `target/release/
  liblibjpeg_turbo_rs_capi.*` with platform-aware flags.
- `examples/stock_djpeg_cjpeg/run.sh` — exercises each built tool
  over `references/libjpeg-turbo/testimages/*.jpg` and diff-compares
  against the system stock binaries. Emits machine-readable TSV.
- `examples/stock_djpeg_cjpeg/build/*.log` — captured linker error
  logs from this run (regenerated on each invocation of `build.sh`).
- `examples/stock_djpeg_cjpeg/build/jconfig.h` / `jconfigint.h` /
  `jversion.h` — minimal in-tree stubs replacing the cmake-generated
  headers; unblocks the compile step without a full upstream build.
- `tests/capi_stock_tool_link.rs` — integration test that runs the
  build+run pair and asserts byte-exact pass rate. Currently
  **documents the blocker**: it runs `build.sh`, checks the return
  code, and reports the missing-symbol count as the failure reason.
