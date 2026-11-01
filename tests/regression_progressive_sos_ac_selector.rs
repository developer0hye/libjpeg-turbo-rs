//! Issue #696: `write_sos_progressive` wrote `Ta = 0` on every AC scan whose
//! band stopped short of coefficient 63, while the scan's entropy data was
//! coded with the component's own AC table. C `emit_sos` hides `Ta` only when
//! `Se == 0`. A chroma scan with `Se < 63` therefore named the luma table, and
//! both this decoder and djpeg read its coefficients with the wrong codes.

mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{
    decompress_to, read_coefficients, write_coefficients_progressive, Encoder, JpegCoefficients,
    PixelFormat, ScanScript, Subsampling,
};

const PHOTO: &[u8] = include_bytes!("fixtures/photo_320x240_444.jpg");
const YCCK: &[u8] = include_bytes!("fixtures/real_world/pil_cmyk.jpg");

/// One SOS header: per-component `(Cs, Td, Ta)`, then `Ss`, `Se`, `Ah`, `Al`.
#[derive(Debug, PartialEq, Eq)]
struct ScanHeader {
    components: Vec<(u8, u8, u8)>,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
}

/// A scan header plus the AC slot the nearest preceding DHT defined for it.
struct ParsedScan {
    header: ScanHeader,
    ac_slot_defined_before_scan: Option<u8>,
}

/// Walks marker segments only. Entropy data is skipped by scanning for the next
/// marker, so stuffed `FF 00` and restart markers are not mistaken for segments.
fn parse_scans(jpeg: &[u8]) -> Vec<ParsedScan> {
    let mut scans: Vec<ParsedScan> = Vec::new();
    let mut ac_slot: Option<u8> = None;
    let mut pos: usize = 2;
    while pos + 4 <= jpeg.len() {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker: u8 = jpeg[pos + 1];
        if marker == 0x00 || marker == 0xFF || (0xD0..=0xD7).contains(&marker) {
            pos += if marker == 0xFF { 1 } else { 2 };
            continue;
        }
        if marker == 0xD9 {
            break;
        }
        let length: usize = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        if marker == 0xC4 {
            let mut table: usize = pos + 4;
            while table < pos + 2 + length {
                if jpeg[table] >> 4 == 1 {
                    ac_slot = Some(jpeg[table] & 0x0F);
                }
                let symbols: usize = jpeg[table + 1..table + 17]
                    .iter()
                    .map(|&count: &u8| count as usize)
                    .sum();
                table += 17 + symbols;
            }
        } else if marker == 0xDA {
            let count: usize = jpeg[pos + 4] as usize;
            let components: Vec<(u8, u8, u8)> = (0..count)
                .map(|index: usize| {
                    let selector: u8 = jpeg[pos + 6 + 2 * index];
                    (jpeg[pos + 5 + 2 * index], selector >> 4, selector & 0x0F)
                })
                .collect();
            let tail: usize = pos + 5 + 2 * count;
            scans.push(ParsedScan {
                header: ScanHeader {
                    components,
                    ss: jpeg[tail],
                    se: jpeg[tail + 1],
                    ah: jpeg[tail + 2] >> 4,
                    al: jpeg[tail + 2] & 0x0F,
                },
                ac_slot_defined_before_scan: ac_slot.take(),
            });
        }
        pos += 2 + length;
    }
    scans
}

/// The issue's script: chroma AC split at coefficient 2, as jpegli-style band
/// splits do. Only the `1-2` chroma scans have `Se < 63`.
fn band_split_script() -> Vec<ScanScript> {
    let mut script: Vec<ScanScript> = vec![ScanScript {
        components: vec![0, 1, 2],
        ss: 0,
        se: 0,
        ah: 0,
        al: 0,
    }];
    for component in 0..3 {
        script.push(ScanScript {
            components: vec![component],
            ss: 1,
            se: 2,
            ah: 0,
            al: 0,
        });
    }
    for (ah, al) in [(0, 2), (2, 1), (1, 0)] {
        for component in 0..3 {
            script.push(ScanScript {
                components: vec![component],
                ss: 3,
                se: 63,
                ah,
                al,
            });
        }
    }
    script
}

/// `band_split_script` in `cjpeg -scans` syntax.
const BAND_SPLIT_SCANS: &[u8] = b"0 1 2: 0 0 0 0;\n\
    0: 1 2 0 0;\n1: 1 2 0 0;\n2: 1 2 0 0;\n\
    0: 3 63 0 2;\n1: 3 63 0 2;\n2: 3 63 0 2;\n\
    0: 3 63 2 1;\n1: 3 63 2 1;\n2: 3 63 2 1;\n\
    0: 3 63 1 0;\n1: 3 63 1 0;\n2: 3 63 1 0;\n";

fn photo_rgb() -> (usize, usize, Vec<u8>) {
    let image = decompress_to(PHOTO, PixelFormat::Rgb).expect("fixture decodes");
    (image.width, image.height, image.data)
}

fn encode_band_split(rgb: &[u8], width: usize, height: usize, arithmetic: bool) -> Vec<u8> {
    Encoder::new(rgb, width, height, PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .quality(90)
        .progressive(true)
        .arithmetic(arithmetic)
        .scan_script(band_split_script())
        .encode()
        .expect("band-split script encodes")
}

/// Issue #696: the AC slot a scan declares must be the one defined for it.
#[test]
fn sub_band_ac_scans_declare_the_table_they_were_coded_with() {
    let (width, height, rgb) = photo_rgb();
    let jpeg: Vec<u8> = encode_band_split(&rgb, width, height, false);

    let mut checked_chroma_sub_band_scans: usize = 0;
    for scan in parse_scans(&jpeg) {
        if scan.header.ss == 0 {
            continue;
        }
        let defined: u8 = scan
            .ac_slot_defined_before_scan
            .expect("every Huffman AC scan is preceded by its optimized table");
        assert_eq!(
            scan.header.components[0].2, defined,
            "scan {:?} coded with AC table {defined}",
            scan.header
        );
        if scan.header.components[0].0 != 1 && scan.header.se < 63 {
            assert_eq!(defined, 1, "chroma AC tables live in slot 1");
            checked_chroma_sub_band_scans += 1;
        }
    }
    assert_eq!(checked_chroma_sub_band_scans, 2, "the Cb and Cr 1-2 scans");
}

/// Issue #696: before the fix this stream failed to decode here and djpeg
/// produced a damaged image, so pixel identity pins both readers.
#[test]
fn sub_band_stream_decodes_identically_in_rust_and_c() {
    let (width, height, rgb) = photo_rgb();
    let jpeg: Vec<u8> = encode_band_split(&rgb, width, height, false);
    let decoded = decompress_to(&jpeg, PixelFormat::Rgb).expect("the stream is valid");

    // Both scripts end with every coefficient at full precision, so they must
    // decode to the same pixels. This also catches a misread both decoders share.
    let default_script: Vec<u8> = Encoder::new(&rgb, width, height, PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .quality(90)
        .progressive(true)
        .encode()
        .expect("default script encodes");
    let expected = decompress_to(&default_script, PixelFormat::Rgb).expect("default decodes");
    helpers::assert_pixels_identical(
        &decoded.data,
        &expected.data,
        width,
        height,
        3,
        "issue 696 band-split against default script",
    );

    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (c_width, c_height, c_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &jpeg, "issue_696_decode");
    assert_eq!((c_width, c_height), (width, height));
    helpers::assert_pixels_identical(
        &decoded.data,
        &c_pixels,
        width,
        height,
        3,
        "issue 696 band-split decode",
    );
}

/// Issue #696: stock cjpeg is the oracle for every selector byte, in both
/// entropy modes. Arithmetic streams decode either way because each scan resets
/// its statistics and `Kx` is 5 for every table, but the bytes still differ.
#[test]
fn scan_headers_match_cjpeg_for_the_same_script() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let (width, height, rgb) = photo_rgb();
    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend_from_slice(&rgb);
    let script: helpers::TempFile = helpers::TempFile::new("issue_696_scans.txt");
    script.write_bytes(BAND_SPLIT_SCANS);
    let script_path: String = script.path().to_string_lossy().into_owned();

    for arithmetic in [false, true] {
        let mut args: Vec<&str> = vec!["-quality", "90", "-sample", "1x1", "-scans", &script_path];
        if arithmetic {
            args.push("-arithmetic");
        }
        let reference: Vec<u8> =
            helpers::encode_with_c_cjpeg(&cjpeg, &ppm, &args, "issue_696_cjpeg");
        let ours: Vec<u8> = encode_band_split(&rgb, width, height, arithmetic);
        let headers = |jpeg: &[u8]| -> Vec<ScanHeader> {
            parse_scans(jpeg)
                .into_iter()
                .map(|scan: ParsedScan| scan.header)
                .collect()
        };
        assert_eq!(
            headers(&ours),
            headers(&reference),
            "arithmetic = {arithmetic}"
        );
    }
}

/// Issue #696 also reached the default script: `generic_progression` sends every
/// component of a four-component frame through a `1-5` band, and this crate's
/// YCCK transcode codes components 1 to 3 with table 1. A lossless transcode must
/// keep every coefficient of this real Adobe-transform-2 file.
#[test]
fn default_ycck_transcode_preserves_coefficients() {
    let source: JpegCoefficients = read_coefficients(YCCK).expect("fixture reads");
    assert_eq!(source.adobe_transform, Some(2), "fixture is YCCK");

    let transcoded: Vec<u8> =
        write_coefficients_progressive(&source, None).expect("default script transcodes");
    let reread: JpegCoefficients = read_coefficients(&transcoded).expect("transcode reads back");

    assert_eq!(source.components.len(), reread.components.len());
    for (index, (before, after)) in source.components.iter().zip(&reread.components).enumerate() {
        assert!(
            before.blocks == after.blocks,
            "component {index} coefficients changed in the transcode"
        );
    }
}
