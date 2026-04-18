# EXIF Orientation fixtures

Eight synthetic 16x8 red-to-blue gradient JPEGs, each carrying a minimal
APP1 EXIF segment whose IFD0 contains exactly one entry: the TIFF
Orientation tag (0x0112, SHORT, count = 1).

| Filename | Orientation value | Meaning |
| --- | --- | --- |
| `orient_1_16x8.jpg` | 1 | Normal (top-left origin) |
| `orient_2_16x8.jpg` | 2 | Mirror horizontal |
| `orient_3_16x8.jpg` | 3 | Rotate 180 |
| `orient_4_16x8.jpg` | 4 | Mirror vertical |
| `orient_5_16x8.jpg` | 5 | Mirror horizontal + rotate 270 CW (transpose) |
| `orient_6_16x8.jpg` | 6 | Rotate 90 CW |
| `orient_7_16x8.jpg` | 7 | Mirror horizontal + rotate 90 CW (transverse) |
| `orient_8_16x8.jpg` | 8 | Rotate 270 CW (rotate 90 CCW) |

Each file is ~690 bytes, well under the 50 KB fixture cap.

The underlying pixels are identical across all 8 files — the only
difference is the embedded Orientation value.  This lets
`worker_b4_exif_orientation.rs` assert that:

* `Image.exif_orientation()` returns exactly the encoded 1..8 value.
* `Image.exif_data()` exposes the raw TIFF payload starting with the
  `II` or `MM` byte-order marker.
* Decoded pixels are pixel-identical to C djpeg across every value
  (the Orientation tag must NOT alter decoded pixels — it is
  downstream display metadata only).

Generation script lives alongside this README's git history: the
synthetic PPM source is a 16x8 red-to-blue gradient, encoded via
`cjpeg -quality 90 -sample 1x1`, then post-processed to inject the
APP1 segment immediately after SOI.
