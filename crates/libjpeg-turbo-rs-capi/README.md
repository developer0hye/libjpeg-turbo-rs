# libjpeg-turbo-rs-capi

C Application Binary Interface (ABI) shim over
[`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs), providing the
shared-library surfaces used by TurboJPEG and classic libjpeg consumers.

> **Writing Rust?** Use the root `libjpeg-turbo-rs` crate unless you explicitly
> need a C ABI. It provides the idiomatic Rust Application Programming
> Interface (API) without C handles, public C structures, loader behavior, or
> manual buffer ownership.

## Choose the correct C interface

| Consumer | Library surface | Current status |
| --- | --- | --- |
| TurboJPEG 3 application using `tj3*` | `libturbojpeg.so.0` / macOS equivalent | **T2:** primary C ABI target; validate the exact functions and packaged artifact your application uses |
| TurboJPEG 1.x/2.x application | Legacy aliases in `libturbojpeg.so.0` | Partial; some aliases are implemented and the remainder require migration to their TurboJPEG 3 successors |
| Classic libjpeg v8 consumer | `libjpeg.so.8` / `libjpeg.8.dylib` | **T3:** experimental and partial; controlled pilots only |
| Classic libjpeg v6b or v7 consumer | `libjpeg.so.62` / `libjpeg.so.7` | **T4:** not supported as a drop-in replacement |

Start with the decision tree and pilot instructions in
[`docs/ADOPTION_GUIDE.md`](../../docs/ADOPTION_GUIDE.md). The canonical live
readiness gate is [`docs/LAST_MILE.md`](../../docs/LAST_MILE.md).

## Current safety and release status

The underlying codec is implemented in Rust and does not call a C JPEG codec.
It still contains narrowly scoped `unsafe` code in architecture-specific
Single Instruction, Multiple Data (SIMD) kernels; this crate necessarily adds
unsafe C pointer and callback boundaries.

The major safe-Rust Undefined Behavior (UB) defects found in the 2026-08 audit
are recorded as closed, and the live gate currently reports no known remaining
UB reachable through the safe Rust API. A formal memory-safety guarantee still
requires the remaining checked-layout centralization and automated
unsafe-boundary verification work tracked by P4-139 and P4-141.

The full release-mode workspace gate is currently red because of P4-170, a
classic source-manager differential test that passes in debug mode and fails in
release mode. Do not describe the complete release gate as green until the live
gate records a successful re-measurement.

The C ABI surface has additional risks that do not exist in the Rust API:
public structure layout, symbol versioning, loader identity, allocator
ownership, callbacks, lifecycle, suspension, errors, and threading. Read
[`docs/ABI_COMPATIBILITY.md`](../../docs/ABI_COMPATIBILITY.md) before changing a
dynamic-library search path.

## Supported surfaces

### TurboJPEG 3

The primary surface is the opaque-handle `tj3*` API, including the implemented
8/12/16-bit, lossy/lossless, YUV, scaling/crop, transform, parameter, buffer,
and image-loading paths documented in
[`docs/C_API_REFERENCE.md`](../../docs/C_API_REFERENCE.md).

Opaque handles avoid classic libjpeg's version-dependent public-structure ABI,
but they do not eliminate behavioral compatibility work. Check the function
matrix, error contract, allocator ownership, and the exact operation your
application uses.

### Legacy TurboJPEG aliases

Only part of the TurboJPEG 1.x/2.x alias surface is exported. Missing aliases
have recommended TurboJPEG 3 successors and thin-shim guidance in the migration
matrix under
[`docs/ABI_COMPATIBILITY.md`](../../docs/ABI_COMPATIBILITY.md).

A missing symbol fails at link time or `dlsym`; it is not automatically mapped
to a similarly named function.

### Classic libjpeg v8

The classic shim targets `JPEG_LIB_VERSION = 80` and the v8 public structure
layout. It includes substantial encode/decode, raw-data, coefficient, marker,
source/destination manager, error, and helper coverage, but remains
experimental and partial.

Important current constraints include:

- open lifecycle, option, error, precision, suspension, and downstream-coverage
  items in the live gate;
- a stricter threading contract than upstream: each `jpeg_compress_struct` or
  `jpeg_decompress_struct` must remain on the thread that created it, tracked by
  P4-132;
- the need to validate the exact packaged/relinked artifact rather than only
  the raw Cargo `cdylib`, tracked by P4-124;
- no blanket guarantee for arbitrary prebuilt classic libjpeg consumers.

Do not replace a system library globally as the first test. Use the isolated
pilot and rollback sequence in
[`docs/ADOPTION_GUIDE.md`](../../docs/ADOPTION_GUIDE.md).

### v6b and v7 are not drop-in targets

The shim mirrors v8 structures. A consumer compiled against v6b or v7 sees a
different structure contract. Renaming or setting a v6b/v7 SONAME does not
change the actual layout and cannot create binary compatibility.

The explicit v6b SONAME build opt-in exists for narrowly controlled consumers
that are themselves built against v8 headers but resolve a historical SONAME.
It is not permission to load the library into a binary compiled against v6b
headers. The create-time version/size guard rejects that mismatch.

## Building from source

```bash
cargo build --release -p libjpeg-turbo-rs-capi
```

The crate produces Rust library, C dynamic library, and static library artifact
forms according to its Cargo configuration. The raw Cargo dynamic library is
useful for development tests, but it is not identical to every shipped native
bundle: the Linux installation path can relink the static library with the
required symbol-version script and stage SONAME links, headers, package
configuration, and CMake metadata.

When testing release behavior, test the staged artifact users will install.

### Optional PNG support

```bash
cargo build --release -p libjpeg-turbo-rs-capi --features png
```

The `png` feature enables PNG input/output for the relevant TurboJPEG image
loading/saving functions. It is off by default so the codec and C ABI build do
not add that dependency unless requested.

## Native release bundles

Tagged releases publish native bundles for the target matrix listed in
[`docs/RELEASE_ARTIFACTS.md`](../../docs/RELEASE_ARTIFACTS.md). Bundles include,
where applicable:

- `libturbojpeg` and `libjpeg` shared-library identities;
- public headers;
- `pkg-config` metadata;
- CMake package configuration;
- SONAME/install-name links;
- license files, bundle metadata, and checksums.

Current delivery gaps include no Windows C ABI bundle, no signature/build
attestation, no Software Bill of Materials (SBOM), and no first-party deb/rpm
package. Follow the release-artifact guide for verification and installation;
do not copy symbolic-link chains as unrelated regular files.

## Evaluation checklist

Before a TurboJPEG or classic v8 pilot:

- [ ] Identify the exact API family and compiled ABI version.
- [ ] Inventory every imported or dynamically resolved symbol.
- [ ] Check each function in `C_API_REFERENCE.md`.
- [ ] Compile a canary against the headers shipped with the same artifact.
- [ ] Verify SONAME/install name and, on Linux, symbol versions.
- [ ] Compare success/failure, error handling, dimensions, colorspace,
      precision, pixels, metadata, and buffer ownership against upstream.
- [ ] Exercise malformed input, short buffers, custom managers, callbacks,
      abort/reuse, and threading behavior used by the application.
- [ ] Measure portable release performance and peak memory on the deployment
      hardware.
- [ ] Run the exact packaged artifact in an isolated loader environment.
- [ ] Exercise rollback to upstream before serving production output.

The root [`docs/ADOPTION_GUIDE.md`](../../docs/ADOPTION_GUIDE.md) provides a
complete corpus, correctness, performance, rollout, and issue-reporting plan.

## Validation evidence

The repository uses differential C-oracle tests, stock tools, loader/symbol
checks, sanitizers, fuzz targets, cross-architecture jobs, and selected
real-downstream harnesses. The following documents separate implementation
coverage from release readiness:

- [`docs/C_API_REFERENCE.md`](../../docs/C_API_REFERENCE.md) — per-function
  status;
- [`docs/ABI_COMPATIBILITY.md`](../../docs/ABI_COMPATIBILITY.md) — layout,
  SONAME, symbol, allocator, lifecycle, and threading policy;
- [`docs/TEST_PARITY.md`](../../docs/TEST_PARITY.md) — upstream behavior/test
  mapping;
- [`docs/LAST_MILE.md`](../../docs/LAST_MILE.md) — canonical T1-T4 gate;
- [`docs/RELEASE_ARTIFACTS.md`](../../docs/RELEASE_ARTIFACTS.md) — what users
  actually download.

A test through a Rust helper is not sufficient evidence for a C entry point.
C ABI behavior must be exercised through the produced shared library and, when
packaging matters, through the packaged artifact.

## License

MIT OR Apache-2.0, matching the root crate. See the repository's
[`LICENSE-MIT`](../../LICENSE-MIT) and
[`LICENSE-APACHE`](../../LICENSE-APACHE).
