# `tests/inputs/`

Test images deliberately kept **out of** `tests/fixtures/`.

`examples/generate_corpus.rs` copies the whole of `tests/fixtures/`
(recursively) into the C-parity corpus, and `examples/corpus_test.rs` then
compares every corpus file against `djpeg`/`cjpeg`/`jpegtran` **through the
8-bit `decompress()` entry point**. An image that entry point cannot decode is
reported there as a `crash` — the harness's word for "the Rust call returned an
error where C succeeded" — with nothing to say that the corpus's own premise,
rather than the library, is what broke.

That premise is real and worth keeping: `decompress()` handles 8-bit *and*
12-bit sources (`tests/fixtures/real_world/libjpeg_testorig12_227x149_12bit.jpg`
passes the corpus comparison byte-exactly), so nearly every JPEG a fixture
directory would hold belongs in the corpus. What does not belong is a stream
only a precision-specific entry point can read.

Put an image here when it is one of those, or when `decompress()` reads it
wrongly under an open item the corpus would then report as a crash:

| file | why it is here |
|---|---|
| `api_sequence_lossless16_gray_8x8.jpg` | 16-bit lossless (`cjpeg -precision 16 -lossless 1`). Only `decompress_16bit` reads it; `decompress()` returns `Unsupported`. Used by `tests/helpers/api_sequence.rs` as `BUILTIN_INPUTS[2]` — the only input on which `TJPARAM_PRECISION` holds 16. |
| `p4199_noninterleaved12_16x16_444.jpg` | 12-bit, three single-component scans (stock `cjpeg -precision 12 -sample 1x1 -scans` with the script `0; 1; 2;`). `tests/tj3_decomp_parameters.rs` uses it as the 12-bit stream whose header walk counts more than one scan, for `TJPARAM_SCANLIMIT`. The 12-bit decoder reads only its first scan and returns wrong pixels (P4-223), which the corpus comparison would flag. |

The trap this directory exists to avoid is tracked as P4-201.
