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

Put an image here when it is one of those, when `decompress()` reads it
wrongly under an open item the corpus would then report as a `fail`, or when
the corpus's transform pass cannot apply to it:

| file | why it is here |
|---|---|
| `api_sequence_lossless16_gray_8x8.jpg` | 16-bit lossless (`cjpeg -precision 16 -lossless 1`). Only `decompress_16bit` reads it; `decompress()` returns `Unsupported`. Used by `tests/helpers/api_sequence.rs` as `BUILTIN_INPUTS[2]` — the only input on which `TJPARAM_PRECISION` holds 16. |
| `p4199_noninterleaved12_16x16_444.jpg` | 12-bit, three single-component scans (stock `cjpeg -precision 12 -sample 1x1 -scans` with the script `0; 1; 2;`). `tests/tj3_decomp_parameters.rs` uses it as the 12-bit stream whose header walk counts more than one scan, for `TJPARAM_SCANLIMIT`, and `tests/configured_limit_rejections.rs` for `max_scans` on both 12-bit routes. The 12-bit decoder reads only its first scan and returns wrong pixels (P4-223), which the corpus comparison would flag. |
| `decomp_params_lossless8_psv4_pt1_24x16.jpg` | 8-bit lossless, predictor 4, point transform 1. `decompress()` reads it, but `transform()` cannot: a lossless stream has no DCT coefficients, so the corpus records a `skip` for each of its seven transforms, and a skip fails the corpus run. `tests/tj3_decomp_parameters.rs` and `crates/libjpeg-turbo-rs-capi/tests/capi_decomp_parameters.rs` use it for the `LOSSLESSPSV` / `LOSSLESSPT` values `setDecompParameters` publishes. |
| `p4226_two_component_unknown_16x16.jpg` | 8-bit baseline with **two** components, so libjpeg classifies it `JCS_UNKNOWN` and TurboJPEG publishes `TJCS_DEFAULT` / `TJSAMP_UNKNOWN`. Written through stock libjpeg 3.2.0's API (`in_color_space = JCS_UNKNOWN`, `input_components = 2`, `jpeg_set_colorspace(JCS_UNKNOWN)`, default quality); no command-line tool can produce it. `djpeg` refuses it ("PPM output must be grayscale or RGB"), so the corpus would record a failure. `crates/libjpeg-turbo-rs-capi/tests/capi_decomp_parameters.rs` uses it for P4-226's `JCS_UNKNOWN` measurement. |
| `p4226_two_component_progressive_16x16.jpg` | The same two-component `JCS_UNKNOWN` frame, written progressive (`jpeg_simple_progression` added to the program above), so it has more scans than a `TJPARAM_SCANLIMIT` of 2. Stock 3.2.0's `tj3Decompress8` reports "Unsupported color conversion request" for it under that limit — the colour converter is chosen before the scans are absorbed — which `tests/tj3_decomp_parameters.rs` pins. `djpeg` refuses it like the baseline one. |

The trap this directory exists to avoid is tracked as P4-201.
