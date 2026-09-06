# Adoption and Migration Guide

**Last reviewed:** 2026-08-27

Use this guide to select the correct `libjpeg-turbo-rs` integration, evaluate
it against your production workload, and roll it out without treating every
Rust and C interface as equally mature.

The canonical readiness source is [`LAST_MILE.md`](LAST_MILE.md). The canonical
C Application Binary Interface (ABI) policy is
[`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md). This guide summarizes those
contracts; it does not override them.

## 1. Choose your integration path

| Application | Recommended path | Current status |
| --- | --- | --- |
| New or existing Rust application | Root `libjpeg-turbo-rs` crate | **T1:** feature-rich and appropriate for evaluation or production use after reviewing the live limitations |
| Rust code built around `image` traits | `libjpeg-turbo-rs-image` | Adapter available; validate color mapping and bridge overhead |
| Browser or WebAssembly System Interface (WASI) application | `libjpeg-turbo-rs-wasm` or the root crate on `wasm32` | Supported; hand-written SIMD128 requires an explicit compiler target feature outside this repository |
| C/C++ application using TurboJPEG 3 (`tj3*`) | `libjpeg-turbo-rs-capi` / `libturbojpeg.so.0` | **T2:** primary C ABI target; evaluate the exact functions and packaged artifact you use |
| C/C++ application compiled against classic libjpeg v8 (`libjpeg.so.8`) | Controlled pilot | **T3:** experimental and partial; not a general system-library replacement |
| Binary compiled against libjpeg v6b (`libjpeg.so.62`) or v7 (`libjpeg.so.7`) | Keep upstream or rebuild against a supported API | **T4:** unsupported as a drop-in replacement |

### Sixty-second decision tree

1. **Can you change Rust source code?** Use the root crate.
2. **Does the code require `image::ImageDecoder` or `image::ImageEncoder`?**
   Use the image bridge and benchmark that complete adapter path.
3. **Are you targeting a browser or WASI?** Use the Wasm package and make the
   `+simd128` build choice explicit.
4. **Can a C/C++ application use TurboJPEG 3?** Prefer the opaque-handle `tj3*`
   API over classic libjpeg.
5. **Must you preserve a prebuilt classic libjpeg ABI?** Continue only for a
   v8-layout consumer and only with the isolated pilot below.
6. **Is the consumer compiled against v6b or v7 headers?** Do not substitute
   this project's v8-layout library.

## 2. Readiness you must understand

### Rust safety status

The Rust codec does not call a C JPEG codec. It still contains narrowly scoped
`unsafe` code in architecture-specific Single Instruction, Multiple Data
(SIMD) kernels, and the optional C ABI shim necessarily handles raw pointers
and callbacks.

The major safe-Rust Undefined Behavior (UB) defects found during the 2026-08
audit are recorded as closed, and the live gate currently reports no known UB
remaining through the safe Rust Application Programming Interface (API). A
formal memory-safety guarantee still requires the remaining checked-layout
centralization and automated unsafe-path verification work tracked by P4-139
and P4-141.

### Current release gate

The full release-mode workspace gate is currently red because of P4-170: two
classic source-manager differential tests pass in debug mode and fail in
release mode. This is not evidence that every Rust-native encode or decode path
fails. It is evidence that the project must not describe the complete release
gate as green before the issue closes and the matrix is re-measured.

Check the current status block in [`LAST_MILE.md`](LAST_MILE.md) before pinning
a version.

### Compatibility tiers are independent

A ready TurboJPEG 3 operation does not make classic libjpeg v8 ready. A working
v8 pilot does not make v6b or v7 compatible. Evaluate the exact API, ABI,
platform, compiler flags, and artifact you will ship.

## 3. Rust-native integration

### Install

```bash
cargo add libjpeg-turbo-rs
```

Or pin the reviewed line explicitly:

```toml
[dependencies]
libjpeg-turbo-rs = "0.8"
```

The root crate's Minimum Supported Rust Version (MSRV) is declared in
[`Cargo.toml`](../Cargo.toml) and enforced in Continuous Integration (CI).

### Decode and encode

```rust
use libjpeg_turbo_rs::{
    compress, decompress_to, JpegError, PixelFormat, Subsampling,
};

fn transcode(jpeg_bytes: &[u8]) -> Result<Vec<u8>, JpegError> {
    let image = decompress_to(jpeg_bytes, PixelFormat::Rgb)?;

    compress(
        &image.data,
        image.width,
        image.height,
        PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
}
```

Set quality and subsampling explicitly so production codec decisions remain
visible in source code.

### Reuse output memory

For frame loops and services, evaluate the caller-owned-buffer path rather than
only one-shot convenience functions:

```rust
use libjpeg_turbo_rs::{decompress_into, output_buffer_size, PixelFormat};

let required = output_buffer_size(&jpeg_bytes, PixelFormat::Rgb)?;
let mut output = vec![0_u8; required];
let info = decompress_into(&jpeg_bytes, PixelFormat::Rgb, &mut output)?;
let pixels = &output[..info.bytes_written];
```

Retain and resize the buffer only when a later image needs more capacity.
Measure allocations, peak resident memory, and throughput with the reuse
strategy your deployment will actually use.

### Select the API by workload

| Need | API direction |
| --- | --- |
| Complete in-memory decode/encode | `decompress*`, `compress`, `Encoder` |
| Reusable caller-owned output | `decompress_into` and sizing helpers |
| Row-at-a-time processing | scanline APIs |
| `std::io` readers and writers | streaming APIs with the default `std` feature |
| Rotate, flip, transpose, or crop without a pixel round trip | coefficient-domain transform APIs |
| Inspect or preserve EXIF, ICC, XMP, IPTC, and comments | metadata and marker APIs |
| Run without the Rust standard library | `default-features = false` with `alloc` available |
| YUV planes or custom JPEG tables | raw/component and advanced builder APIs |

Compile-checked examples live in [`../examples`](../examples) and on
[docs.rs](https://docs.rs/libjpeg-turbo-rs).

### Rust production checklist

- [ ] Pin a reviewed crate version and record the Rust toolchain.
- [ ] Test baseline, progressive, grayscale, CMYK/YCCK, and malformed inputs
      present in your workload.
- [ ] Verify output pixel format, channel order, and alpha/padding semantics.
- [ ] Compare metadata behavior for EXIF/ICC/XMP/IPTC-bearing images.
- [ ] Benchmark a portable release build before adding machine-specific flags.
- [ ] Measure reusable-buffer and streaming paths when they match production.
- [ ] Apply your own dimensions, memory, concurrency, and timeout limits.
- [ ] Keep a rollback version until production traffic has passed acceptance
      criteria.

## 4. `image` crate integration

Use `libjpeg-turbo-rs-image` when the caller expects `image` crate decoder or
encoder traits.

```toml
[dependencies]
libjpeg-turbo-rs-image = "0.1"
image = { version = "0.25", default-features = false }
```

```rust
use image::ImageDecoder;
use libjpeg_turbo_rs_image::JpegDecoder;

let mut decoder = JpegDecoder::new(&jpeg_bytes)?;
let mut pixels = vec![0_u8; decoder.total_bytes() as usize];
decoder.read_image(&mut pixels)?;
```

Review the adapter
[`README.md`](../crates/libjpeg-turbo-rs-image/README.md) for color mapping,
advanced-format access, and limitations.

Benchmark the adapter end to end. A core-codec benchmark excludes trait
adaptation, representation conversion, and surrounding application buffers.
Prefer an explicit bridge dependency rather than a hidden global patch of a
transitive `image` dependency; explicit selection makes review and rollback
simpler.

## 5. WebAssembly integration

Use the Wasm wrapper when JavaScript or TypeScript calls the codec, or depend on
the root crate directly from Rust targeting `wasm32`.

```bash
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --release --target web
```

The repository's local Cargo configuration does not travel with a published
crate. Without the explicit target feature, the hand-written SIMD128 path is
not selected and the build falls back to scalar execution.

Record these separately in a browser/WASI evaluation:

- runtime/version and SIMD128 availability;
- module size and instantiation cost;
- first-call and steady-state latency;
- linear-memory growth and maximum accepted dimensions;
- JavaScript/Wasm boundary copies;
- worker/concurrency model;
- malformed-input timeout and memory limits.

See [`../crates/libjpeg-turbo-rs-wasm/README.md`](../crates/libjpeg-turbo-rs-wasm/README.md).

## 6. TurboJPEG 3 migration

TurboJPEG 3 is the preferred C migration path because its public API uses
opaque handles instead of version-dependent classic libjpeg structures.

### Preconditions

Proceed after confirming:

- the application uses `tj3*`, or every legacy symbol it uses is supported in
  [`C_API_REFERENCE.md`](C_API_REFERENCE.md);
- your platform has a supported bundle or a build you can package and test;
- the workload does not depend on undocumented implementation behavior;
- loader or package selection can be returned to upstream during rollout.

### Obtain and verify the artifact

Tagged releases publish the bundle matrix described in
[`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md). Follow its checksum, unpack,
installation, and package-discovery instructions.

A checksum published beside an artifact is not an offline signature. Until
signing or build provenance is published, record that supply-chain limitation
or build from a pinned commit in your controlled pipeline.

### Compile a canary against shipped headers

Compile against the headers and package metadata from the same bundle as the
shared library. The canary should:

1. create every handle type used in production;
2. set quality, subsampling, and all relevant parameters explicitly;
3. compress and decompress representative images;
4. exercise an invalid input and the error API;
5. destroy handles and free buffers through the documented allocator;
6. verify the library identity selected by the loader.

### Run application-level differential tests

Run each production operation through upstream libjpeg-turbo and this shim,
then compare:

- success/failure classification and error behavior;
- dimensions, subsampling, colorspace, and precision;
- decoded pixels under a predefined comparison rule;
- metadata and transform behavior;
- buffer ownership and reallocation semantics;
- latency, throughput, allocations, and peak memory;
- concurrent use of independent handles.

### Legacy aliases

The library implements the TurboJPEG 3 surface and only part of the legacy
1.x/2.x alias set. The migration matrix in
[`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md) maps missing symbols to supported
successors. Missing symbols fail at link time or dynamic lookup; they are not
substituted automatically.

## 7. Classic libjpeg v8 pilot

The classic v8 shim is experimental and partial. Never replace a system library
globally as the first test.

### Required preconditions

- The consumer is compiled against `JPEG_LIB_VERSION = 80` and expects the v8
  library identity.
- Every imported `jpeg_*` symbol has been inventoried.
- Required behavior is covered in `C_API_REFERENCE.md` and the live gate.
- Every `jpeg_compress_struct` and `jpeg_decompress_struct` stays on the thread
  that created it; the current shim's stricter contract is P4-132.
- Loading can be isolated to a process, container, test prefix, or explicit
  runtime search path.
- Upstream can be restored without rebuilding the entire application.

### Pilot sequence

1. Inventory imported symbols and compiled ABI identity.
2. Build or unpack the exact artifact intended for deployment.
3. Verify SONAME/install name and symbol versions.
4. Run stock-tool and canary tests in an isolated loader environment.
5. Cross-decode both directions and run same-operation differential tests.
6. Exercise lifecycle edges used by the application: suspension, short input,
   custom source/destination managers, callbacks, abort/reuse, errors,
   precision, raw data, coefficients, and threading.
7. Shadow production traffic in a separate process before serving output.
8. Roll out gradually with an automatic fallback to upstream.

### Stop conditions

Stop and remain on upstream when:

- the consumer is v6b/v7 rather than v8;
- a required symbol or behavior is partial or missing;
- the application transfers a `cinfo` across threads;
- the packaged artifact has not passed loader and lifecycle tests;
- malformed input has a worse termination, memory, or error contract;
- rollback requires an unsafe global replacement or a full application rebuild.

## 8. Unsupported or unproved paths

### v6b and v7 drop-in replacement

The classic shim mirrors v8 structures. A different SONAME does not change the
layout. Use upstream, rebuild against TurboJPEG 3, or rebuild against v8 and
run the controlled pilot. Separately built and tested per-ABI variants would be
a future product decision, not an aliasing trick.

### Unmeasured architecture performance

Do not infer x86_64/aarch64 results on armv7, RISC-V Vector (RVV), POWER, or
s390x. Use scalar support only after measuring on the deployment hardware. The
root [`README.md`](../README.md) records current SIMD coverage and gaps.

### Unsigned native artifacts

Current native bundles are checksummed but not yet signed and do not include a
Software Bill of Materials (SBOM). Organizations requiring verified provenance
should build from a pinned commit in a controlled pipeline or wait for the
provenance milestone.

## 9. Build a representative evaluation

A codec evaluation should model production, not a single synthetic image.

### Corpus

Include relevant examples of:

- tiny icons, ordinary camera/web images, 1080p/4K, and maximum dimensions;
- grayscale, 4:4:4, 4:2:2, 4:2:0, and unusual subsampling;
- baseline and progressive JPEG;
- CMYK/YCCK and embedded color profiles;
- EXIF orientation and other metadata;
- lossless, high-precision, or arithmetic streams when accepted;
- truncated, corrupted, oversized, and adversarial inputs;
- outputs produced by the current production encoder.

Record hashes or deterministic generator parameters. Do not commit private
customer images to the public repository.

### Correctness criteria

Define the rule before testing:

- exact bytes when the contract requires identity;
- exact pixels when both implementations promise the same conversion;
- a justified bounded pixel difference only when valid inverse transforms may
  differ;
- cross-decode success when encoded bytes may legitimately differ;
- exact metadata preservation or intentional stripping;
- matching failure class, with no panic, abort, hang, out-of-bounds access, or
  unbounded allocation.

### Performance criteria

Measure at least:

- p50, p95, and p99 latency;
- images or megapixels per second;
- peak resident memory and observable allocations;
- first-call and steady-state behavior;
- one-shot and reusable-buffer paths;
- one thread and production concurrency;
- portable release and machine-specific builds separately.

Use the same optimization class, CPU policy, thread count, input bytes, output
format, and correctness checks for both codecs.

### Operational criteria

Confirm repeatable installation, loader behavior, memory/time limits,
monitoring that identifies the selected codec, and a rollback that has been
exercised rather than merely documented.

## 10. Production rollout

A recommended sequence is:

1. offline differential test on the full corpus;
2. permanent CI canary on a smaller redistributable corpus;
3. shadow mode that computes but does not serve the new result;
4. opt-in or internal traffic with immediate fallback;
5. small-percentage rollout with correctness, latency, memory, and error alerts;
6. progressive increase after a defined observation window;
7. intentional end-of-life decision before removing the rollback dependency.

For Rust, rollback is commonly a Cargo version change. For C ABI replacement,
preserve the previous package or library path and keep loader selection
explicit.

## 11. Report an adoption blocker

Open an issue with this information:

```markdown
## Integration path
Rust API / image bridge / Wasm / TurboJPEG 3 / classic libjpeg v8

## Version and source
crate/release version, commit, artifact filename and checksum

## Environment
OS, architecture, CPU, Rust/C compiler, linker, build flags, runtime

## Production operation
API calls, pixel format, subsampling, precision, dimensions, threading

## Expected and actual behavior
specification/upstream result and observed error/output/regression

## Minimal reproduction
source plus a redistributable fixture or deterministic generator

## Differential evidence
upstream version, commands, raw output, hashes, benchmark methodology

## Adoption impact
blocks evaluation / rollout / production / optimization request
```

Do not publish exploit details in a normal issue. Use GitHub private
vulnerability reporting when the repository exposes it; establishing and
documenting a permanent private disclosure route is a `1.0` requirement in the
Product Requirements Document (PRD).

## 12. Related documents

- [`../PRD.md`](../PRD.md) — product direction and adoption priorities
- [`README.md`](README.md) — documentation index
- [`LAST_MILE.md`](LAST_MILE.md) — canonical T1-T4 release gates
- [`FEATURE_PARITY.md`](FEATURE_PARITY.md) — implemented feature inventory
- [`TEST_PARITY.md`](TEST_PARITY.md) — C behavior/test parity
- [`CORPUS_TEST_REPORT.md`](CORPUS_TEST_REPORT.md) — corpus evidence
- [`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md) — ABI and SONAME policy
- [`C_API_REFERENCE.md`](C_API_REFERENCE.md) — C function status
- [`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md) — bundles and verification gaps
- [`ENCODING_PERFORMANCE.md`](ENCODING_PERFORMANCE.md) — detailed encoder evidence
