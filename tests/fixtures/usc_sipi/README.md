# USC-SIPI Miscellaneous test images

A small slice of the USC Signal & Image Processing Institute's classic
image database (<https://sipi.usc.edu/database/>).  The three canonical
images — lena, mandrill, and airplane (F-16) — have underpinned image
processing research for decades.  They are distributed for academic /
research use and fetched on-demand by `scripts/fetch_usc_sipi.sh`.

## Checked-in seed fixtures

Only small (80x80) synthetic stand-ins are committed so worker-b4's
USC-SIPI tests compile and run without network access.  Each seed is
< 3 KB.

| Filename | WxH | Quality | Bytes |
| --- | --- | --- | --- |
| `lena_80x80_q75.jpg` | 80x80 | 75 | ~1.1 KB |
| `lena_80x80_q90.jpg` | 80x80 | 90 | ~1.4 KB |
| `mandrill_80x80_q75.jpg` | 80x80 | 75 | ~2.3 KB |
| `mandrill_80x80_q90.jpg` | 80x80 | 90 | ~2.7 KB |
| `airplane_80x80_q75.jpg` | 80x80 | 75 | ~1.2 KB |
| `airplane_80x80_q90.jpg` | 80x80 | 90 | ~1.7 KB |

The seed content is deliberately synthetic (warm skin-tone gradient for
"lena", red-green checker for "mandrill", sky/fuselage/ground strip for
"airplane") to give each test image a distinct spectral character
without redistributing the USC-SIPI originals.

## Full corpus (not checked in)

After running `scripts/fetch_usc_sipi.sh`, this directory additionally
contains the 512x512 derivatives:

    lena_q75.jpg, lena_q90.jpg
    mandrill_q75.jpg, mandrill_q90.jpg
    airplane_q75.jpg, airplane_q90.jpg

The raw TIFFs live in `target/usc_sipi_cache/`.

## License

USC-SIPI images are distributed for research and academic use.
The synthetic seed JPEGs stored here are newly generated content and
inherit the repository's dual MIT/Apache-2.0 license.
