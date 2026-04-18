# RAW thumbnail fixtures

Synthetic camera-RAW containers (ARW/CR2/NEF-style TIFF 6.0 wrappers)
used to exercise `libjpeg_turbo_rs::extract_embedded_jpeg()`.

| Filename | Bytes | Layout |
| --- | --- | --- |
| `synthetic_arw_24x16.tiff` | 751 | II-TIFF header + IFD0 (Make SHORT + JPEGInterchangeFormat LONG + JPEGInterchangeFormatLength LONG) + 701-byte JPEG payload at offset 50 (24x16 RGB gradient, q85, 4:2:0, produced by `cjpeg`). |

Well under the 50 KB fixture cap.

The fixture mirrors what Sony ARW, Canon CR2, Nikon NEF, Olympus ORF,
and DNG files put in their first IFD: a pointer to an embedded JPEG
preview/thumbnail, indexed by TIFF tags 0x0201 and 0x0202.  Our
extractor walks the IFD chain, locates that pair, validates bounds,
and returns the raw JPEG bytes — which the companion test then feeds
into our decoder to confirm a complete round-trip.

## License

Newly generated content; inherits the repository's dual
MIT/Apache-2.0 licence.
