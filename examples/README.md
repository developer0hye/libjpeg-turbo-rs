# Examples

## Start here (user-facing)

| Example | Shows |
| --- | --- |
| [`decode.rs`](decode.rs) | Decode to RGB / any pixel format |
| [`encode.rs`](encode.rs) | One-shot encode + the `Encoder` builder |
| [`probe_header.rs`](probe_header.rs) | Dimensions & EXIF orientation with no pixel decode |
| [`decode_into.rs`](decode_into.rs) | Decode into a caller-owned, reusable buffer |
| [`transform_lossless.rs`](transform_lossless.rs) | DCT-domain rotate — no decode, no quality loss |

Each runs without arguments against a bundled fixture:
`cargo run --example decode`. The `image`-crate bridge has its own
example: `cargo run -p libjpeg-turbo-rs-image --example decode_with_image`.

These live in the repository only — the published crate keeps its
`exclude = ["examples/", ...]` so `cargo add libjpeg-turbo-rs` stays
slim; browse them here or via the README links (decided with issue #388).

## Everything else (development tooling)

The remaining files are benchmarks, C-differential probes, corpus
generators, and profiling harnesses used by CI and the performance
workflow (`bench_*`, `probe_*`, `diag_*`, `generate_corpus`,
`corpus_test`, oracles `*_c_oracle.c`, ...). They are deliberately kept
at this level because CI jobs and `experiments/` logs reference their
`cargo run --example <name>` invocations; the table above is the
user-facing surface.
