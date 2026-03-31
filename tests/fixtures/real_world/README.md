# Real-World JPEG Test Images

## Purpose

Test fixtures for comprehensive JPEG decode/encode cross-validation against C libjpeg-turbo.
Every decode test cross-validates Rust output against `djpeg`, and every encode test against `cjpeg`,
targeting pixel-identical (diff=0) results. These images cover the full spectrum of JPEG features:
baseline, progressive, arithmetic, extended sequential, grayscale, CMYK/YCCK, ICC profiles, EXIF
metadata, restart markers, non-interleaved scans, 12-bit precision, and resolutions from 1x1 to
7680x4320 (8K).

## Sources

### `derived_*` — Generated from libjpeg-turbo test data

Synthetic images generated using C `cjpeg`/`djpeg`/`jpegtran` from source images in the
libjpeg-turbo test suite. The source images come from `references/libjpeg-turbo/testimages/`.

- **URL**: <https://github.com/libjpeg-turbo/libjpeg-turbo>
- **License**: IJG/BSD (Independent JPEG Group license + BSD-style libjpeg-turbo additions)

### `w3c_*` — W3C XHTML Print test suite

Images from the W3C XHTML Print test suite, providing large progressive JPEGs with various
subsampling modes and EXIF metadata.

- **URL**: <https://www.w3.org/MarkUp/Test/xhtml-print/20050519/tests/>
- **License**: W3C Software License (permissive, public)

### `exif_*` — ianare/exif-samples

Real-world camera photos with diverse EXIF metadata from Canon, Fujifilm, Nikon, and iPhone.

- **URL**: <https://github.com/ianare/exif-samples>
- **License**: MIT

### `pil_*` — Python Pillow test suite

Test images from the Python Pillow (PIL fork) image processing library, including CMYK and
grayscale variants.

- **URL**: <https://github.com/python-pillow/Pillow/tree/main/Tests/images>
- **License**: HPND (Historical Permission Notice and Disclaimer)

### `libjpeg_*` — libjpeg-turbo test images

Original test images shipped with libjpeg-turbo, including the canonical `testorig.jpg`,
`testimgint.jpg`, `testimgari.jpg`, and the 12-bit `testorig12.jpg`.

- **URL**: <https://github.com/libjpeg-turbo/libjpeg-turbo/tree/main/testimages>
- **License**: IJG/BSD

### `zune_*` — zune-image test suite

Test images from the zune-image project (zune-jpeg crate), covering edge cases like non-interleaved
scans, unusual sampling factors, MJPEG Huffman tables, CMYK, and YCCK.

- **URL**: <https://github.com/etemesi254/zune-image>
- **Specific path**: `crates/zune-jpeg/tests/` and related test directories
- **License**: MIT / Apache-2.0

### `image_rs_*` — image-rs test suite

Test images from the Rust `image` crate, providing progressive JPEG with ICC profiles and EXIF.

- **URL**: <https://github.com/image-rs/image/tree/main/tests/images/jpg>
- **License**: MIT / Apache-2.0

## Image List

| Filename | WxH | Bytes | Characteristics |
|---|---|---|---|
| derived_1x1_grayscale_q95.jpg | 1x1 | 331 | baseline, grayscale |
| derived_1x1_q95.jpg | 1x1 | 635 | baseline |
| derived_2x3_q95.jpg | 2x3 | 760 | baseline |
| derived_3x2_q95.jpg | 3x2 | 707 | baseline |
| derived_7x7_q90.jpg | 7x7 | 667 | baseline |
| derived_15x9_q85.jpg | 15x9 | 690 | baseline |
| derived_227x149_422_q75.jpg | 227x149 | 6,280 | baseline, 4:2:2 |
| derived_227x149_444_q75.jpg | 227x149 | 7,069 | baseline, 4:4:4 |
| derived_227x149_arithmetic.jpg | 227x149 | 5,153 | arithmetic |
| derived_227x149_grayscale_q90.jpg | 227x149 | 6,068 | baseline, grayscale |
| derived_227x149_icc_profile.jpg | 227x149 | 34,272 | baseline, ICC |
| derived_227x149_optimized.jpg | 227x149 | 5,463 | baseline |
| derived_227x149_progressive.jpg | 227x149 | 5,655 | progressive, 10-scan |
| derived_227x149_progressive_icc.jpg | 227x149 | 34,157 | progressive, ICC, 10-scan |
| derived_227x149_progressive_optimized.jpg | 227x149 | 5,655 | progressive, 10-scan |
| derived_227x149_progressive_restart2.jpg | 227x149 | 5,848 | progressive, DRI, 10-scan |
| derived_227x149_quality1.jpg | 227x149 | 1,387 | extended, quality=1 (lowest) |
| derived_227x149_quality10.jpg | 227x149 | 2,051 | extended, quality=10 (low) |
| derived_227x149_quality100.jpg | 227x149 | 18,228 | baseline, quality=100 (highest) |
| derived_227x149_restart4.jpg | 227x149 | 5,777 | baseline, DRI |
| derived_2048x1536_baseline_q90.jpg | 2048x1536 | 768,769 | baseline |
| derived_3840x2160_4k_420_q85.jpg | 3840x2160 | 187,655 | baseline, 4:2:0, synthetic gradient |
| derived_3840x2160_4k_progressive.jpg | 3840x2160 | 114,489 | progressive, 4:2:0, 10-scan, synthetic gradient |
| derived_7680x4320_8k_420_q75.jpg | 7680x4320 | 579,544 | baseline, 4:2:0, synthetic gradient |
| derived_7680x4320_8k_progressive.jpg | 7680x4320 | 262,005 | progressive, 4:2:0, 10-scan, synthetic gradient |
| derived_exif_canon_arithmetic.jpg | 480x360 | 23,471 | arithmetic |
| derived_exif_canon_progressive.jpg | 480x360 | 24,798 | progressive, 10-scan |
| derived_grayscale_progressive.jpg | 128x128 | 5,190 | progressive, grayscale, 6-scan |
| derived_grayscale_restart2.jpg | 128x128 | 5,648 | baseline, grayscale, DRI |
| derived_hopper_progressive.jpg | 128x128 | 6,363 | progressive, 10-scan |
| derived_w3c_restart1.jpg | 2048x1536 | 760,051 | baseline, DRI |
| derived_w3c_restart8.jpg | 2048x1536 | 759,894 | baseline, DRI |
| exif_canon_powershot.jpg | 480x360 | 32,764 | baseline, EXIF |
| exif_fujifilm.jpg | 59x100 | 2,241 | baseline, EXIF |
| exif_gps_iphone.jpg | 640x480 | 161,713 | baseline, EXIF |
| exif_nikon_d70.jpg | 100x66 | 14,034 | baseline, ICC, EXIF |
| exif_orientation_6.jpg | 600x450 | 136,257 | baseline, ICC, EXIF |
| image_rs_progressive.jpg | 320x240 | 21,474 | progressive, ICC, EXIF, 10-scan |
| libjpeg_testimgari_227x149_arithmetic.jpg | 227x149 | 5,126 | arithmetic |
| libjpeg_testimgint_227x149_baseline.jpg | 227x149 | 5,756 | baseline |
| libjpeg_testorig_227x149_baseline.jpg | 227x149 | 5,770 | baseline |
| libjpeg_testorig12_227x149_12bit.jpg | 227x149 | 12,394 | extended, 12-bit |
| pil_cmyk.jpg | 100x100 | 29,364 | baseline, YCCK, EXIF, DRI |
| pil_grayscale.jpg | 128x128 | 5,353 | baseline, grayscale |
| pil_hopper.jpg | 128x128 | 6,412 | baseline |
| w3c_jpeg420exif.jpg | 2048x1536 | 707,923 | progressive, EXIF, 12-scan |
| w3c_jpeg422jfif.jpg | 2048x1536 | 1,409,739 | progressive, 10-scan |
| w3c_jpeg444.jpg | 256x256 | 2,440 | progressive, 5-scan |
| zune_cmyk_600x397_4comp.jpg | 600x397 | 96,660 | baseline, CMYK |
| zune_grayscale_progressive_900x675.jpg | 900x675 | 109,669 | progressive, grayscale, 6-scan |
| zune_mjpeg_huffman_1280x720.jpg | 1280x720 | 476,629 | baseline, DRI |
| zune_non_interleaved_420_64x64.jpg | 64x64 | 829 | baseline, 3-scan |
| zune_non_interleaved_422_65x65.jpg | 65x65 | 1,364 | baseline, 3-scan |
| zune_non_interleaved_440_64x64.jpg | 64x64 | 897 | baseline, 3-scan |
| zune_non_interleaved_444_64x64.jpg | 64x64 | 1,007 | baseline, 3-scan |
| zune_sampling_factors_400x225.jpg | 400x225 | 10,077 | baseline |
| zune_synthetic_progressive_533x800.jpg | 533x800 | 30,361 | progressive, 6-scan |
| zune_tiny_non_interleaved_444_16x16.jpg | 16x16 | 690 | baseline, 3-scan |
| zune_weird_sampling_600x320.jpg | 600x320 | 39,969 | baseline |
| zune_ycck_1318x611_4comp.jpg | 1318x611 | 142,943 | baseline, YCCK, DRI |
| zune_ycck_progressive_383x740_4comp.jpg | 383x740 | 793,052 | progressive, YCCK, ICC, EXIF, 9-scan |

**Total: 61 images**

## Coverage

- **Encoding types**: baseline (39), arithmetic (3), progressive (18), extended sequential (3)
- **Component types**: 3-comp YCbCr (51), grayscale (6), 4-comp CMYK/YCCK (4)
- **ICC profiles**: 6
- **EXIF metadata**: 9
- **Restart markers (DRI)**: 8
- **Multi-scan (progressive or non-interleaved)**: 23
- **12-bit precision**: 1
- **Non-interleaved scans**: 5
- **Dimension range**: 1x1 to 7680x4320 (8K)
- **Total size**: ~7.5 MB

## License

All images in this directory are redistributable under their respective open-source licenses
(IJG/BSD, W3C Software License, MIT, HPND, Apache-2.0). See the Sources section above for
per-image license details.
