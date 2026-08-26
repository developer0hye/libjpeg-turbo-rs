# libjpeg-turbo-rs-wasm

WebAssembly (Wasm) bindings for
[`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs), providing a
Rust-native JPEG codec for browsers and WebAssembly System Interface (WASI)
environments without a C JPEG implementation.

Use this wrapper when JavaScript or TypeScript is the caller. Rust applications
targeting `wasm32` directly may depend on the root crate instead and choose the
appropriate Rust Application Programming Interface (API).

Read [`docs/ADOPTION_GUIDE.md`](../../docs/ADOPTION_GUIDE.md) for the project
readiness, evaluation, memory, rollout, and rollback checklist.

## Build

```bash
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --release --target web
```

The generated `pkg/` directory is an ECMAScript module package suitable for a
browser build. A browser benchmark harness lives under [`bench/`](bench).

## SIMD128 must be enabled explicitly

The repository's `.cargo/config.toml` enables `+simd128` for in-repository
`wasm32` builds. Cargo configuration from this repository does **not** travel
with a crate published to crates.io or npm.

Without this compiler target feature, the hand-written Single Instruction,
Multiple Data (SIMD) kernels are not selected and the codec uses the scalar
path. The build remains functional, but a benchmark result from an in-repo
SIMD build must not be compared with an external scalar build as though they
were configured identically.

For direct Rust builds:

```bash
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --release --target wasm32-unknown-unknown
```

For a deployment that must support clients without Wasm SIMD, publish a scalar
fallback or use feature detection before loading a SIMD-targeted module. A
module compiled with required SIMD instructions is not a universal fallback
binary.

## Core feature selection

The wrapper builds the root codec with default features disabled and the
`simd` Cargo feature enabled. On `wasm32`, SIMD dispatch is a compile-time
target-feature decision rather than the native `std` runtime CPU-feature
probe.

A direct root-crate dependency can choose a scalar build by omitting the
compiler target feature or disable hand-written SIMD through Cargo features as
documented by the root crate. Keep build configuration in source control so a
release and its benchmark use the same settings.

## Distribution

The release workflow supports:

- **npm** as the primary JavaScript distribution channel;
- **crates.io** for Rust-side Wasm consumers;
- root `v*` releases and Wasm-specific release tags according to the repository
  release workflow.

Before publishing or consuming a package, verify the exact package version,
root codec version, compiler flags, generated package metadata, and module
hash. Native C ABI release bundles and their compatibility tiers are unrelated
to this wrapper.

## Evaluation checklist

Record at least:

- browser/runtime and version;
- `wasm-bindgen`/`wasm-pack`, Rust, and LLVM versions;
- whether the loaded module requires and receives SIMD128;
- compressed and uncompressed module size;
- instantiation and first-call cost separately from steady-state throughput;
- p50, p95, and p99 latency on representative images;
- linear-memory growth, peak memory, and maximum accepted dimensions;
- copy cost between JavaScript buffers and Wasm linear memory;
- worker/threading model and number of concurrent codec instances;
- malformed-input timeout and memory limits;
- output pixel format, metadata behavior, and correctness against the current
  production codec.

Measure the JavaScript-to-Wasm wrapper end to end. A root-codec native
benchmark excludes module initialization, boundary copies, garbage collection,
and browser scheduling.

## Production guidance

- Keep one deterministic build command for the package and benchmark.
- Reuse module instances and buffers where the wrapper API permits it.
- Reject dimensions and requested output sizes before committing excessive
  linear memory.
- Run malformed and adversarial inputs inside the same worker and timeout model
  used in production.
- Roll out behind a codec selector or package version that can return to the
  previous implementation.
- Monitor out-of-memory failures, traps, latency, and decode errors separately
  from network or application failures.

## Current project status

The wrapper inherits the root codec's correctness and safe-Rust status. It does
not depend on or promote the optional C Application Binary Interface (ABI)
shim. Canonical readiness and open verification work remain in
[`docs/LAST_MILE.md`](../../docs/LAST_MILE.md); Wasm feature coverage is
summarized in the root [`README.md`](../../README.md).

## License

MIT OR Apache-2.0, matching the root codec. See
[`LICENSE-MIT`](../../LICENSE-MIT) and
[`LICENSE-APACHE`](../../LICENSE-APACHE).
