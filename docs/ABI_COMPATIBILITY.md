# ABI Compatibility Policy

> **Audience.** Distro packagers, downstream Rust consumers, anyone shipping a binary that links against `libjpeg-turbo-rs-capi`'s cdylib in place of upstream `libjpeg.so.62` / `libjpeg.so.8` / `libturbojpeg.so.0`. If you are only consuming the Rust API (`use libjpeg_turbo_rs::*;`), this document does not apply — Rust's normal type system gives you binary stability.
>
> **TL;DR.** We target **JPEG_LIB_VERSION = 80** (the v8 ABI). The canonical SONAME for that ABI is `libjpeg.so.8` / `libjpeg.8.dylib`, and that is the **default** since P4-3 (2026-05-17). The historic `libjpeg.so.62` default has been removed because v6b-compiled consumers can silently read garbage from later v8 fields. v6b is still available via a single opt-in switch: `CAPI_ACK_V6B_SONAME=1`. The build script auto-implies `CAPI_SONAME=libjpeg.so.62` and `CAPI_INSTALL_NAME=@rpath/libjpeg.62.dylib` when that flag is set, so SONAME and macOS install_name stay in lockstep.

## Why this document exists

Upstream libjpeg-turbo's CMake build supports three `JPEG_LIB_VERSION` settings (`references/libjpeg-turbo/CMakeLists.txt:264-384`):

| `WITH_JPEG7` | `WITH_JPEG8` | `JPEG_LIB_VERSION` | Default SONAME (Linux) | Notes                             |
|--------------|--------------|--------------------|------------------------|-----------------------------------|
| (off)        | (off)        | 62                 | `libjpeg.so.62`        | Default — most distros            |
| ON           | (off)        | 70                 | `libjpeg.so.7`         | Adds scale_num/scale_denom etc.   |
| (any)        | ON           | 80                 | `libjpeg.so.8`         | Adds is_baseline, block_size etc. |

Each version adds new fields to `struct jpeg_decompress_struct` and `struct jpeg_compress_struct`. A consumer compiled against v6b (`#include <jpeglib.h>` with `JPEG_LIB_VERSION = 62`) sees a *smaller* struct than a consumer compiled against v8. Field offsets for fields that exist in both versions are usually compatible, but only because v8 *appends* new fields to the end — the appended fields don't exist in the v6b consumer's view.

The danger: a library that *advertises* the v6b SONAME (`libjpeg.so.62`) but *exposes* the v8 layout silently passes a wider struct to a narrower consumer. The consumer reads only the v6b prefix correctly. Any v8-only field the consumer doesn't know about is fine. But if the *library* writes to an appended v8 field (e.g. `is_baseline` at offset 312) and a v6b consumer's struct only has `308` bytes, the library is writing past the consumer's allocation. **This is undefined behavior.**

## Our policy

### What we target

- **Struct layout: JPEG_LIB_VERSION = 80** (v8). All offsets in `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` are computed against the v8 LP64 layout. The compile-time assertion block at `jpeglib.rs:3900-3970` pins these.
- **Public symbol surface: TurboJPEG 3.x + classic libjpeg API at v8 level.** Includes `tj3*` (TurboJPEG 3); 21 `tj*` legacy 1.x/2.x aliases (lifecycle `tjInitCompress` / `tjInitDecompress` / `tjInitTransform` / `tjDestroy`; `tjCompress2` / `tjDecompress2` / `tjDecompressHeader3`; `tjTransform` / `tjEncodeYUV3` / `tjDecodeYUV`; buffer-size `tjBufSize` / `TJBUFSIZE` / `TJBUFSIZEYUV` / `tjBufSizeYUV` / `tjBufSizeYUV2` / `tjPlaneSizeYUV` / `tjPlaneWidth` / `tjPlaneHeight`; image I/O `tjLoadImage` / `tjSaveImage`; error string `tjGetErrorStr2`); 18 other legacy 1.x/2.x symbols are still allowlisted-missing — see [P4-18](last_mile/phase4.md#p4-18-18-legacy-turbojpeg-1x2x-symbols-remain-allowlisted-missing--open). And the `jpeg_*` classic API at v8.
- **Default precision: 8-bit/12-bit/16-bit/lossless** as supported through both the TJ3 and classic APIs.

### What we deliberately do *not* target

- **v6b binary compatibility.** A binary compiled against `JPEG_LIB_VERSION = 62` headers cannot safely link against our cdylib unless its struct layout happens to coincide with ours up to the field it touches. We make no guarantee of that.
- **v7 binary compatibility.** Same as v6b — we do not ship per-version cdylib variants.
- **Multi-precision libjpeg `jpeg16_*` / `jpeg12_*` symbols beyond what upstream `jpeglib.h` declares.** Upstream's high-precision raw-data entry points are 8/12 only; we mirror that.

### How to consume us safely

The matrix below shows which `CAPI_SONAME` / `CAPI_INSTALL_NAME` settings are safe for which kind of consumer. Set both via env at `cargo build` time; they are honored by `crates/libjpeg-turbo-rs-capi/build.rs:30-44`.

| Consumer compiled against     | Safe `CAPI_SONAME`            | Safe `CAPI_INSTALL_NAME`         | Notes                                                  |
|-------------------------------|-------------------------------|----------------------------------|--------------------------------------------------------|
| v8 headers (`libjpeg.so.8`)   | `libjpeg.so.8` *(default)*    | `@rpath/libjpeg.8.dylib` *(default)* | **Recommended.** No silent UB. This is the default since P4-3 (2026-05-17). |
| TurboJPEG (`libturbojpeg.so.0`) | `libturbojpeg.so.0`         | `@rpath/libturbojpeg.0.dylib`    | Safe for TJ3 callers — TurboJPEG API is opaque-handle, no struct ABI. Legacy 1.x/2.x surface is partial: 21 aliases wired in `legacy.rs` (mostly v2/v3 variants + buffer/image helpers); 18 still allowlisted-missing (v1 / un-versioned variants like `tjAlloc`, `tjFree`, `tjCompress`, `tjGetScalingFactors`). See [P4-18](last_mile/phase4.md#p4-18-18-legacy-turbojpeg-1x2x-symbols-remain-allowlisted-missing--open) for the implement-vs-deprecate decision matrix. |
| v7 headers (`libjpeg.so.7`)   | (unsupported)                 | (unsupported)                    | Recompile against v8 or use upstream v7.               |
| v6b headers (`libjpeg.so.62`) | `libjpeg.so.62` *opt-in*      | `@rpath/libjpeg.62.dylib` *opt-in* | **Risky / non-default.** Works iff the consumer never touches v7+ fields, and requires the `CAPI_ACK_V6B_SONAME=1` env to silence the build warning. See below. |

### Threading contract

A `jpeg_decompress_struct` / `jpeg_compress_struct` allocated through our C ABI shim **must be used (and freed) on the thread that created it.** Our shim stores per-`cinfo` private state in thread-local side tables keyed by the `cinfo` pointer; transferring `cinfo` ownership across threads silently breaks lookups and leaks the original-thread entry.

Concretely:

- **Safe:** thread A calls `jpeg_create_decompress(cinfo)`, drives the full decode lifecycle through `jpeg_destroy_decompress(cinfo)` on thread A. Likewise for the compress side.
- **Unsafe:** thread A calls `jpeg_create_decompress(cinfo)`; the application then passes `cinfo` (by value or pointer) to thread B; thread B calls `jpeg_read_header(cinfo, …)`. The shim's private-state lookup on thread B returns `None`, and observable behaviour ranges from `JERR_BAD_STATE` to silent miscompilation. `jpeg_destroy_decompress(cinfo)` on thread B will **not** free thread A's entry — the entry leaks until thread A exits.

**Why this contract.** The v8 `struct jpeg_decompress_struct` is ABI-mirrored byte-for-byte (`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:3900-3970` pins the offsets). There is no room to append a `priv_ptr` field without breaking offset compatibility with upstream-compiled consumers, so private state lives in TLS instead. Implementation pointers: `DECOMPRESS_PRIVATE_STATE` at `jpeglib.rs:368-372` (decompress side) + compress equivalent at `:3492-3505`.

**Divergence from upstream.** Upstream libjpeg-turbo's contract is "single-threaded per `cinfo`, but ownership transfer between threads is OK provided the application enforces non-concurrent access." We are stricter: ownership stays on the creating thread. If your consumer needs cross-thread `cinfo` ownership transfer (the canonical example is FFmpeg's frame-thread JPEG path), file an issue with the use case — the migration to a global `OnceLock<RwLock<HashMap>>` is tracked as P4-16 Option A in `docs/last_mile/phase4.md` and we will prioritise based on adoption signal.

### The `libjpeg.so.62` opt-in path

Our build.rs default is **`CAPI_SONAME=libjpeg.so.8`** (P4-3, 2026-05-17). The v6b SONAME `libjpeg.so.62` is no longer the default; it remains available as an opt-in for distro experiments.

The v6b SONAME *works* for the majority of v6b consumers (Pillow 10.x, ImageMagick 7, libtiff 4.x, GD 2.x, FFmpeg 6.x with the JPEG codec) because they only read fields that exist in both v6b and v8 at compatible offsets. But there is a non-empty set of cases where it silently breaks:

1. **A v6b consumer reads `cinfo.scale_num` / `cinfo.scale_denom` / `cinfo.do_fancy_upsampling`** — these are at v8 offsets (68, 72, 96 etc.) but a v6b struct does not have them. Our shim writes there. Result: depending on what the v6b consumer has at *those* byte offsets in *its* struct, we silently corrupt a v6b-only field.
2. **A v6b consumer reads `cinfo.is_baseline` (offset 312, v8-only)** — does not exist in v6b struct. Reading is undefined.
3. **A v6b consumer with a custom `jpeg_error_mgr` whose `format_message` walks an addon table** — works either way; format_message is at offset 0 of the error manager and is ABI-stable since libjpeg v6.

To opt into the v6b SONAME at build time (e.g. for distro replacement experiments where you control every consumer), set the single acknowledgement env:

```bash
CAPI_ACK_V6B_SONAME=1 cargo build -p libjpeg-turbo-rs-capi --release
```

The build script then auto-derives `CAPI_SONAME=libjpeg.so.62` and `CAPI_INSTALL_NAME=@rpath/libjpeg.62.dylib`, keeping the Linux SONAME and macOS install_name in lockstep so dyld can resolve the v6b name. Explicit `CAPI_SONAME` / `CAPI_INSTALL_NAME` overrides still win if you need a non-standard combination.

Without `CAPI_ACK_V6B_SONAME=1`, build.rs emits a loud `cargo:warning=` if v6b SONAME or install_name is requested by hand — the same warning fires if SONAME and install_name disagree on v6b vs v8 (which would silently break load-time resolution on macOS).

## Field-presence reference

The list below is the contract we mirror from `references/libjpeg-turbo/src/jpeglib.h`. Field names that appear under "v6b layout" are present in the v6b ABI; everything below them is appended in later versions.

### `struct jpeg_decompress_struct` (v8 layout, LP64)

```
common (all versions):
  err               offset   0  (jpeg_error_mgr*)
  mem               offset   8  (jpeg_memory_mgr*)
  progress          offset  16  (jpeg_progress_mgr*)
  client_data       offset  24  (void*)
  is_decompressor   offset  32  (boolean)
  global_state      offset  36  (int)

v6b layout (also present in v8):
  src               offset  40  (jpeg_source_mgr*)
  image_width       offset  48
  image_height      offset  52
  num_components    offset  56
  jpeg_color_space  offset  60
  out_color_space   offset  64

v7+ extensions (NOT in v6b):
  scale_num         offset  68
  scale_denom       offset  72
  output_gamma      offset  80
  buffered_image    offset  88
  raw_data_out      offset  92
  ... (continued — see jpeglib.rs:3900-3970)

v8+ extensions (NOT in v6b or v7):
  is_baseline       offset 312   (boolean, JPEG_LIB_VERSION >= 80)
  ... (block_size etc. — see jpeglib.rs:3946+)
```

For the full enumeration of v6b → v7 → v8 differences, see the `#if JPEG_LIB_VERSION >= 70` and `>= 80` blocks in `references/libjpeg-turbo/src/jpeglib.h:191,371,393,419,465,498,654,697,742,1003`.

### `struct jpeg_compress_struct`

Symmetric to the decompress side — refer to `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` and `references/libjpeg-turbo/src/jpeglib.h:300+` for the per-version field list.

## Roadmap — T4 (v6b/v7 drop-in) is tracked as P2-A

T4 ("System v6b/v7 drop-in") is an explicit non-goal under the current
default. The only honest path to closing it is the per-ABI cdylib
matrix tracked as **P2-A** in the master plan
(`/Users/yhkwon/.claude/plans/dreamy-moseying-swing.md`). When triggered
the work is:

1. Add a `CAPI_LIB_VERSION = 62 | 70 | 80` cfg gate.
2. Generate per-version `JpegDecompressPublic` / `JpegCompressPublic` types with conditional fields (likely via `bindgen` or a small build-time code-gen step — hand-maintaining three ~200-field mirrors is known-fragile).
3. Pin per-version offset assertions for each.
4. Build per-version cdylibs (`libjpeg.so.62.cdylib`, `libjpeg.so.7.cdylib`, `libjpeg.so.8.cdylib`).
5. Add a CI matrix entry per version.

This is genuinely large work and is *out of scope* for the current "v8-targeted with documented v6b risk" policy. The decision to take it on should be evidence-driven: at least one named real consumer that we want to support, and an explicit user requirement that opt-in `CAPI_SONAME=libjpeg.so.8` is not acceptable.

## Verification commands

```bash
# Default build (libjpeg.so.8 SONAME, v8 layout — the safe, recommended path).
cargo build -p libjpeg-turbo-rs-capi --release
otool -D target/release/liblibjpeg_turbo_rs_capi.dylib  # macOS — expects libjpeg.8.dylib
readelf -d target/release/liblibjpeg_turbo_rs_capi.so | grep SONAME  # Linux — expects libjpeg.so.8

# v6b opt-in (documented-risk path). One env switches everything:
# the build script derives CAPI_SONAME=libjpeg.so.62 and
# CAPI_INSTALL_NAME=@rpath/libjpeg.62.dylib automatically.
CAPI_ACK_V6B_SONAME=1 cargo build -p libjpeg-turbo-rs-capi --release

# Verify the offset assertions catch any future struct-shape drift.
cargo build -p libjpeg-turbo-rs-capi --release  # const-eval asserts run at compile time
```
