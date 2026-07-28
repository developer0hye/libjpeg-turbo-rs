# libjpeg-turbo-rs-capi

C ABI shim over [libjpeg-turbo-rs](https://crates.io/crates/libjpeg-turbo-rs):
a drop-in replacement for the two shared libraries C consumers link against.

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
