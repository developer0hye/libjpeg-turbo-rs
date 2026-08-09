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
- **Public symbol surface: TurboJPEG 3.x + classic libjpeg API at v8 level.** This describes exported names and struct layout, not complete classic behavioral parity. The remaining classic contract gaps are indexed in `docs/LAST_MILE.md`. The TurboJPEG surface includes `tj3*`, 21 wired legacy aliases, and 18 deliberately deprecated aliases documented in the [migration matrix](#legacy-turbojpeg-1x2x-aliases--partial-coverage-p4-18).
- **Precision:** 8/12/16-bit and lossless paths are supported by the native/TJ3 APIs. Classic high-precision scanline completion and raw option/state fidelity remain P4-94/P4-95/P4-98/P4-102.

### What we deliberately do *not* target

- **v6b binary compatibility.** A binary compiled against `JPEG_LIB_VERSION = 62` headers cannot safely link against our cdylib unless its struct layout happens to coincide with ours up to the field it touches. We make no guarantee of that.
- **v7 binary compatibility.** Same as v6b — we do not ship per-version cdylib variants.
- **Multi-precision libjpeg `jpeg16_*` / `jpeg12_*` symbols beyond what upstream `jpeglib.h` declares.** Upstream's high-precision raw-data entry points are 8/12 only; we mirror that.

### GNU/Linux ELF symbol-version caveat (P4-81)

The current Linux cdylib has the correct default `DT_SONAME` and exports the
required symbol names, but does not yet attach upstream's GNU
`LIBJPEG_8.0`/`LIBJPEGTURBO_8.0` version definitions to them. On the measured
Ubuntu 24.04 glibc loader, a prebuilt OpenCV 4.6 consumer still binds both
versioned create functions to our unversioned exports and completes encode
and decode, while printing:

```text
libjpeg.so.8: no version information available
```

This is functional compatibility evidence for that environment, not a claim
of warning-free distro packaging or compatibility with every ELF loader.
Distro packagers should treat the missing version definitions as an open
replacement gap until P4-81 closes. The reproducible binding evidence is in
`experiments/opencv_downstream_2026-08-02.md`.

### Crate-private version node `LIBJPEGTURBORS_PRIVATE_1.0` (P4-129)

`src/jpeglib.rs` defines 16 `jpeg_capi_test_*` accessors that the shim's own
dlopen-based test suites resolve out of the shared library. They share the
`jpeg_` prefix, so the `jpeg_*` pattern in the reference node used to stamp all
16 as `@@LIBJPEG_8.0` — advertising 16 entry points **no real libjpeg has** as
reference v8 API.

They are now claimed by exact name under **`LIBJPEGTURBORS_PRIVATE_1.0`**, a
node deliberately named so it cannot be mistaken for an upstream one. Exact
names outrank patterns, which is the precedence the `jpeg_mem_dest` /
`jpeg_mem_src` assignment already relies on.

**These are not API.** They carry no stability guarantee, belong to no libjpeg
or TurboJPEG surface, and exist only so the shim's tests can inspect state
through the shared library. Do not link them.

They stay dynamically visible rather than becoming `local:` because eight test
suites resolve them from this cdylib; hiding them would break those without
improving the shipped surface's honesty, which is what the mislabelling
actually cost. `tests/capi_symbol_versions.rs` reads the accessor list out of
`src/jpeglib.rs`, so a 17th added without updating `build.rs` fails CI instead
of silently rejoining the reference node.

### ABI-layout compatibility matrix

The matrix below covers struct-layout/SONAME compatibility only; it does not
promote the partial classic implementation to behavioral drop-in status. Set
both variables at build time; they are honored by
`crates/libjpeg-turbo-rs-capi/build.rs`.

| Consumer compiled against     | Safe `CAPI_SONAME`            | Safe `CAPI_INSTALL_NAME`         | Notes                                                  |
|-------------------------------|-------------------------------|----------------------------------|--------------------------------------------------------|
| v8 headers (`libjpeg.so.8`)   | `libjpeg.so.8` *(default)*    | `@rpath/libjpeg.8.dylib` *(default)* | Correct advertised layout/SONAME. P4-110 still requires exact create-time version/size guards, and other behavioral gaps remain open. |
| TurboJPEG (`libturbojpeg.so.0`) | `libturbojpeg.so.0`         | `@rpath/libturbojpeg.0.dylib`    | Safe for TJ3 callers — TurboJPEG API is opaque-handle, no struct ABI. Legacy 1.x/2.x surface is partial: 21 aliases wired in `legacy.rs` (mostly v2/v3 variants + buffer/image helpers); 18 deliberately deprecated (v1 / un-versioned variants like `tjAlloc`, `tjFree`, `tjCompress`, `tjGetScalingFactors`). See the [Legacy TurboJPEG 1.x/2.x aliases](#legacy-turbojpeg-1x2x-aliases--partial-coverage-p4-18) section below for the per-symbol migration matrix (P4-18 closed 2026-05-19). |
| v7 headers (`libjpeg.so.7`)   | (unsupported)                 | (unsupported)                    | Recompile against v8 or use upstream v7.               |
| v6b headers (`libjpeg.so.62`) | `libjpeg.so.62` *opt-in*      | `@rpath/libjpeg.62.dylib` *opt-in* | **Risky / non-default.** Works iff the consumer never touches v7+ fields, and requires the `CAPI_ACK_V6B_SONAME=1` env to silence the build warning. See below. |

### Threading contract

A `jpeg_decompress_struct` / `jpeg_compress_struct` allocated through our C ABI shim **must be used (and freed) on the thread that created it.** Our shim stores per-`cinfo` private state in thread-local side tables keyed by the `cinfo` pointer; transferring `cinfo` ownership across threads silently breaks lookups and leaks the original-thread entry.

Concretely:

- **Safe:** thread A calls `jpeg_create_decompress(cinfo)`, drives the full decode lifecycle through `jpeg_destroy_decompress(cinfo)` on thread A. Likewise for the compress side.
- **Unsafe:** thread A calls `jpeg_create_decompress(cinfo)`; the application then passes `cinfo` (by value or pointer) to thread B; thread B calls `jpeg_read_header(cinfo, …)`. The shim's private-state lookup on thread B returns `None`, and observable behaviour ranges from `JERR_BAD_STATE` to silent miscompilation. `jpeg_destroy_decompress(cinfo)` on thread B will **not** free thread A's entry — the entry leaks until thread A exits.

**Why this contract.** The v8 `struct jpeg_decompress_struct` is ABI-mirrored byte-for-byte (`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs:3900-3970` pins the offsets). There is no room to append a `priv_ptr` field without breaking offset compatibility with upstream-compiled consumers, so private state lives in TLS instead. Implementation pointers: `DECOMPRESS_PRIVATE_STATE` at `jpeglib.rs:368-372` (decompress side) + compress equivalent at `:3492-3505`.

**Divergence from upstream.** Upstream libjpeg-turbo's contract is "single-threaded per `cinfo`, but ownership transfer between threads is OK provided the application enforces non-concurrent access." We are stricter: ownership stays on the creating thread. If your consumer needs cross-thread `cinfo` ownership transfer (the canonical example is FFmpeg's frame-thread JPEG path), file an issue with the use case — the migration to a global `OnceLock<RwLock<HashMap>>` is tracked as P4-16 Option A in `docs/last_mile/phase4.md` and we will prioritise based on adoption signal.

### Legacy TurboJPEG 1.x/2.x aliases — partial coverage (P4-18)

Our `libturbojpeg.so.0` cdylib exports the full **TurboJPEG 3** API (`tj3*`) and **21 of the 39** legacy 1.x/2.x aliases. The remaining 18 legacy symbols are intentionally not exported and live in `crates/libjpeg-turbo-rs-capi/tests/symbol_inventory.rs:190-207` as "documented-deprecated, migrate to the TJ3 successor". Consumers compiled against TJ 1.x/2.x headers that touch any of the 18 will fail at `dlsym` / link time with `symbol not found`; consumers that touch only the 21 wired aliases work today.

**Migration matrix.** Each missing symbol has a documented successor on the TJ3 surface; in most cases the signatures match closely enough that a thin C shim is one or two lines.

| Missing 1.x/2.x symbol | Recommended successor | Migration notes |
| --- | --- | --- |
| `tjAlloc(int bytes)` | `tj3Alloc(size_t bytes)` | Signature change is `int → size_t`; cast at the call site. Allocates from the same allocator. |
| `tjFree(unsigned char *buf)` | `tj3Free(void *buf)` | Pointer-type widening; no behavioural change. |
| `tjCompress(...)` | `tjCompress2(...)` (wired) → `tj3Compress8(...)` | The v1 entry point predates the buffer-size argument added in v2. Use `tjCompress2` directly; we already export it. |
| `tjCompressFromYUV(...)` | `tj3CompressFromYUV8(...)` | Pass quality / subsamp through `tj3Set(handle, TJPARAM_*, …)` before calling. |
| `tjCompressFromYUVPlanes(...)` | `tj3CompressFromYUVPlanes8(...)` | Same as above; planar variant. |
| `tjDecodeYUVPlanes(...)` | `tj3DecodeYUVPlanes8(...)` | Planar YUV → packed RGB conversion (no JPEG). |
| `tjDecompress(...)` | `tjDecompress2(...)` (wired) → `tj3Decompress8(...)` | Like `tjCompress`: use `tjDecompress2` we already export. |
| `tjDecompressHeader(...)` | `tjDecompressHeader3(...)` (wired) → `tj3DecompressHeader(...)` | Both 1.x and 2.x header-only variants are subsumed by the v3 form, which is wired. |
| `tjDecompressHeader2(...)` | `tjDecompressHeader3(...)` (wired) → `tj3DecompressHeader(...)` | Same as `tjDecompressHeader`. |
| `tjDecompressToYUV(...)` | `tj3DecompressToYUV8(...)` | Allocate output via `tj3YUVBufSize()` first. |
| `tjDecompressToYUV2(...)` | `tj3DecompressToYUV8(...)` | Same as `tjDecompressToYUV`. |
| `tjDecompressToYUVPlanes(...)` | `tj3DecompressToYUVPlanes8(...)` | Plane sizes via `tj3YUVPlaneSize()`. |
| `tjEncodeYUV(...)` | `tjEncodeYUV3(...)` (wired) → `tj3EncodeYUV8(...)` | The v3 form is wired in `legacy.rs`. |
| `tjEncodeYUV2(...)` | `tjEncodeYUV3(...)` (wired) → `tj3EncodeYUV8(...)` | Same as `tjEncodeYUV`. |
| `tjEncodeYUVPlanes(...)` | `tj3EncodeYUVPlanes8(...)` | Planar variant. |
| `tjGetErrorCode(tjhandle h)` | `tj3GetErrorCode(tjhandle h)` | Signature compatible; return codes match `TJERR_*` enum. |
| `tjGetErrorStr(void)` (no-handle) | `tj3GetErrorStr(NULL)` (no-handle form) | TJ3 wraps the no-handle form behind a NULL-handle convention. |
| `tjGetScalingFactors(int *numFactors)` | `tj3GetScalingFactors(int *numFactors)` | Identical signature; return value is the same `tjscalingfactor *` array. |

**Tiny shim recipe.** If your consumer cannot be recompiled against TJ3, wrap the missing symbol in a one-line C function next to its callers and rebuild that translation unit only. Example:

```c
/* shim_legacy_tj.c — re-implement two missing symbols in terms of the TJ3 successors. */
#include <turbojpeg.h>
#include <stdlib.h>

unsigned char *tjAlloc(int bytes) { return tj3Alloc((size_t)bytes); }
void           tjFree (unsigned char *buf) { tj3Free(buf); }
```

Compile and link that file alongside your existing consumer; no source changes required upstream of the shim.

**If you absolutely need the 18 symbols wired in our cdylib itself** (e.g. you cannot ship an extra shim translation unit alongside, or you `LD_PRELOAD` the cdylib directly), see P4-18 Option A in `docs/last_mile/phase4.md`. Each symbol becomes a `pub extern "C" fn` in `crates/libjpeg-turbo-rs-capi/src/legacy.rs` that delegates to its `tj3*` successor — the implementation work is mechanical but not yet scheduled. File an issue with the use case to trigger the work.

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
