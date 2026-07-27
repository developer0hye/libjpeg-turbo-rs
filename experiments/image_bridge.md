# image-bridge (libjpeg-turbo-rs-image) experiments

## 2026-07-28 — #381 feature-wiring fix, bridge-vs-zune re-run

**Change.** `crates/libjpeg-turbo-rs-image/Cargo.toml` dependency on the core
codec went from `default-features = false, features = ["simd"]` (no `std`,
therefore no runtime CPU detection — a stock x86-64-baseline build of the
bridge silently ran SSE2-only) to default features (`std` + `simd`).
Regression pinned by
`crates/libjpeg-turbo-rs-image/tests/runtime_dispatch.rs`, which fails under
`cargo test -p libjpeg-turbo-rs-image` on the old manifest (verified
red/green 2026-07-28; feature unification means a whole-workspace build masks
the bad manifest, so the `-p` resolution is the one that matters — it is also
exactly how a crates.io consumer resolves).

**Benchmark.** `cargo run --release -p libjpeg-turbo-rs-image --example
bench_vs_zune` — 1920x1080 4:2:0 q90 synthetic photo, 50 iters/leg, both
legs decoding into a preallocated caller buffer (codex review caught the
first cut charging a ~6 MiB per-iteration allocation to zune only), run
sequentially, macOS aarch64 (Apple Silicon):

| path | ms/decode |
| --- | --- |
| bridge (`libjpeg_turbo_rs_image::JpegDecoder`) | 2.944 |
| `image` built-in (zune-jpeg) | 3.869 |
| ratio (zune/bridge) | **1.31x faster** |

`simd_and_std_features_enabled() == true` printed from the benched artifact
— the dispatch capability is live in the exact binary measured, not assumed.

**Caveat.** On aarch64 NEON selection is compile-time, so this host cannot
show the delta the #381 defect actually caused; the loss was x86-64 AVX2
dispatch. The number above is the post-fix adoption headline for the bridge
(faster than the `image` default), and the dispatch regression itself is
guarded by the mechanism test, not by a timing.
