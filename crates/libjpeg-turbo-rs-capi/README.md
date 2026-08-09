# libjpeg-turbo-rs-capi

C ABI shim over [libjpeg-turbo-rs](https://crates.io/crates/libjpeg-turbo-rs)
for the two shared libraries C consumers link against.

> **Tiers and status (P4-140).** **TurboJPEG 3 (`libturbojpeg.so.0`) is the
> primary target.** The classic libjpeg leg targets the **v8 identity only**
> (`libjpeg.so.8`) and is experimental and partial. **v6b (`libjpeg.so.62`)
> and v7 are explicit non-goals** — their struct layouts differ, so
> substituting this library for them corrupts memory rather than merely
> failing. The safe-Rust/unsafe-SIMD boundary in the underlying crate is under
> audit (P4-135..P4-139), so no memory-safety guarantee is offered yet, and
> "drop-in replacement" is not a claim this project currently makes.

> **Writing Rust?** You almost certainly want the root
> [libjpeg-turbo-rs](https://crates.io/crates/libjpeg-turbo-rs) crate
> instead — it is the safe, idiomatic API. This crate exists for C/C++
> programs (and language runtimes binding `libjpeg`/`libturbojpeg`) that
> need the C ABI.

- **libjpeg v8 ABI** (`libjpeg.so.8` / `jpeglib.h` surface): `jpeg_std_error`,
  `jpeg_CreateDecompress/Compress`, source/destination managers, marker
  readers, `jpeg_read_header` … `jpeg_finish_decompress`, raw-data and
  coefficient access, `jpeg_simple_progression`, error/message formatting.
- **TurboJPEG 3 API** (`libturbojpeg.so.0` / `turbojpeg.h`): the `tj3*`
  family (8/12/16-bit, lossless, YUV planes, crop/scale, transforms), plus
  21 of the 39 TJ 1.x/2.x legacy aliases; the remaining 18 are documented
  with a migration matrix.

The definitive per-function status table (✅ implemented / 🔶 partial /
❌ missing) is
[`docs/C_API_REFERENCE.md`](https://github.com/developer0hye/libjpeg-turbo-rs/blob/main/docs/C_API_REFERENCE.md)
in the repository root.

## Building

```sh
cargo build --release -p libjpeg-turbo-rs-capi
```

produces `rlib`, `cdylib`, and `staticlib` artifacts. On Linux/macOS the
cdylib can be soname-aliased to `libjpeg.so.8` / `libturbojpeg.so.0` and
substituted for the C libraries; the test suite exercises exactly that via
`dlopen` (`libloading`), including a libtiff integration test.

SONAME selection (v8 default, v6b opt-in via `CAPI_ACK_V6B_SONAME=1`),
the ABI-version matrix, and which settings are safe for which consumers
are documented in
[`docs/ABI_COMPATIBILITY.md`](https://github.com/developer0hye/libjpeg-turbo-rs/blob/main/docs/ABI_COMPATIBILITY.md).

Features:

| flag | default | effect |
|---|---|---|
| `png` | off | enables PNG input/output in `tj3LoadImage8` / `tj3SaveImage8` |

## Compatibility notes

- The codec underneath is cross-validated against C libjpeg-turbo
  (`djpeg` / `cjpeg` / `jpegtran`) by the repository test suite.
- Error messages reproduce the C `format_message` table so callers that
  parse message strings keep working.
- The crate is `unsafe` at the boundary by nature (C ABI); the underlying
  codec is pure Rust.

## License

MIT OR Apache-2.0, same as the root crate.
