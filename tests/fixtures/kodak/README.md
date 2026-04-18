# Kodak PhotoCD corpus

The 24-image Kodak PhotoCD suite is a classic benchmark for lossy-image
codecs.  It is public domain and hosted at <http://r0k.us/graphics/kodak/>.

Only a tiny seed (two 96x64 derivatives) is checked in so the worker-b4
Kodak tests compile and run without network access.  The full corpus is
intentionally NOT committed — run the fetch script to populate it before
running the full-resolution round-trip suite:

    scripts/fetch_kodak.sh

## Checked-in seed fixtures

| Filename | WxH | Quality | Bytes | Notes |
| --- | --- | --- | --- | --- |
| `kodim01_96x64_q75.jpg` | 96x64 | 75 | ~1.3 KB | Synthetic stand-in generated from a seeded RGB gradient; encoded via libjpeg-turbo `cjpeg -quality 75`. |
| `kodim01_96x64_q90.jpg` | 96x64 | 90 | ~1.7 KB | Same synthetic content re-encoded at quality 90. |

Each checked-in file is well under 50 KB, matching the repo's "small
synthetic fixture" rule.  The content is deliberately synthetic (not a
rescaled copy of `kodim01.png`) to avoid any redistribution obligations
from the Kodak dataset while still exercising the same YCbCr / 4:2:0
code paths our PSNR round-trip test depends on.

## Full corpus (not checked in)

After running `scripts/fetch_kodak.sh`, `tests/fixtures/kodak/` will
additionally contain:

    kodim01_q75.jpg … kodim24_q75.jpg   (full 768x512 / 512x768)
    kodim01_q90.jpg … kodim24_q90.jpg

The raw PNG sources remain in `target/kodak_cache/` and are not needed
beyond one-time transcoding.

## License

Kodak PhotoCD originals are released into the public domain by Kodak.
The synthetic seed JPEGs stored here are newly generated content and
inherit the repository's dual MIT/Apache-2.0 license.
