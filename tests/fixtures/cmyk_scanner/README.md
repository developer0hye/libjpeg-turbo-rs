# CMYK scanner fixture

Hand-crafted 64x64 CMYK (YCCK) JPEG mimicking a flatbed-scanner
capture: four quadrants each highlighting one process colour
(cyan, magenta, yellow, key/black).

| Filename | WxH | Colour space | Subsampling | Quality | Bytes |
| --- | --- | --- | --- | --- | --- |
| `scanner_64x64.jpg` | 64x64 | CMYK (APP14 Adobe, 4-component) | 4:4:4 | 92 | ~956 B |

Well under the 50 KB fixture cap.

The pixel content is deterministic — `worker_b4_cmyk_scanner.rs`
regenerates the fixture on demand if it is missing so the test suite
is self-healing across checkouts.  The content lets us exercise:

* Adobe APP14 marker parsing (required for CMYK JPEGs that lack JFIF).
* 4-component interleaved SOS decode.
* The CMYK output pixel path — our decoder must return raw CMYK bytes
  unchanged.
* Re-encoding decoded CMYK at quality 100 / 4:4:4 is lossless except
  for IDCT/DCT rounding (measured worst-case diff = 1 per channel).

The accompanying test also asserts that C `djpeg` accepts the fixture
without error and produces a PNM with matching geometry.  CMYK-to-RGB
pixel-level comparison across djpeg and our decoder is intentionally
elided because the two tools apply different CMYK→RGB transforms.

## License

Newly generated content inheriting the repository's dual
MIT/Apache-2.0 license.
