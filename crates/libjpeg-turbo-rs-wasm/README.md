# libjpeg-turbo-rs-wasm

WASM bindings for [`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs) — a fast, pure-Rust JPEG codec for the browser.

## Build

```sh
wasm-pack build --release --target web
```

The generated `pkg/` directory is a ready-to-use ES module. A browser benchmark harness lives in `bench/`.

**SIMD128 is a compile-time flag, and it is NOT self-contained in this crate.** In-repo builds get it from this repository's `.cargo/config.toml` (`-C target-feature=+simd128` for the wasm32 targets). If you consume this crate from crates.io — outside this repository — that config does not travel with it, and without the flag the codec's WASM SIMD kernels compile out, silently falling back to scalar. Set it yourself:

```sh
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --release --target web
```

## Distribution

Documented with issue #380:

- **npm** — the primary channel. `.github/workflows/release.yml` builds with `wasm-pack` and runs `npm publish` on every `v*` (root release) and `wasm-v*` (bindings-only release) tag; `capi-v*` tags skip it.
- **crates.io** — additionally possible since the dependency on the core codec carries a `version` key (this issue's fix), so Rust-side wasm consumers can `cargo add libjpeg-turbo-rs-wasm`. Publication rides the same maintainer release step as the other crates.

## Feature notes

The core codec is built with `default-features = false, features = ["simd"]`: on `wasm32`, SIMD128 selection is a compile-time `target_feature` decision, so the `std` runtime-CPU-detection rationale that applies to the native `image` bridge (issue #381) does not apply here.
