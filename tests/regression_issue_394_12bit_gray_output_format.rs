//! Issue #394 (P4-65): 12-bit grayscale decode silently ignored the
//! requested output format. `decode_12bit_as_8bit` hard-coded
//! `pixel_format: Grayscale` for 1-component frames, so
//! `decompress_to(.., Argb)` returned a 1 bpp image while
//! `output_buffer_size()` advertised 4 bpp — a caller trusting the
//! advertised size indexed stale bytes. C expands instead: `djpeg -rgb`
//! on a `cjpeg -precision 12` grayscale JPEG emits a P6 (RGB) PPM.

mod helpers;

use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{decompress_to, output_buffer_size, PixelFormat};

fn cjpeg_supports_precision(cjpeg: &PathBuf) -> bool {
    let output = Command::new(cjpeg).arg("-?").output();
    match output {
        Ok(o) => format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        )
        .contains("-precision"),
        Err(_) => false,
    }
}

/// 12-bit grayscale fixture via our own encoder (C-tool-free).
fn fixture_12bit_gray(width: usize, height: usize) -> Vec<u8> {
    let samples: Vec<i16> = (0..width * height)
        .map(|i| ((i * 97) % 4096) as i16)
        .collect();
    libjpeg_turbo_rs::precision::compress_12bit(
        &samples,
        width,
        height,
        1,
        90,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("12-bit gray encode")
}

/// Issue #394 core: every RGB-family format request must be honoured —
/// same expansion contract as the 8-bit grayscale path, with
/// `output_buffer_size()` agreeing with the returned data.
#[test]
fn issue_394_12bit_gray_honours_output_format() {
    let (width, height): (usize, usize) = (33, 21);
    let jpeg: Vec<u8> = fixture_12bit_gray(width, height);

    // Reference gray values from the (correct) Grayscale path.
    let gray = decompress_to(&jpeg, PixelFormat::Grayscale).expect("grayscale decode");
    assert_eq!(gray.pixel_format, PixelFormat::Grayscale);
    assert_eq!(gray.data.len(), width * height);

    let formats: &[PixelFormat] = &[
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
        PixelFormat::Xrgb,
        PixelFormat::Xbgr,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    for &fmt in formats {
        let advertised: usize = output_buffer_size(&jpeg, fmt).expect("output_buffer_size");
        let img = decompress_to(&jpeg, fmt)
            .unwrap_or_else(|e| panic!("{fmt:?}: decompress_to failed: {e}"));
        assert_eq!(img.pixel_format, fmt, "{fmt:?}: returned pixel_format");
        assert_eq!((img.width, img.height), (width, height), "{fmt:?}: dims");
        assert_eq!(
            img.data.len(),
            advertised,
            "{fmt:?}: data length must equal output_buffer_size"
        );
        // Channel placement: gray value at R/G/B offsets, 255 at pad.
        let bpp: usize = fmt.bytes_per_pixel();
        if bpp >= 3 {
            let r_off: usize = fmt.red_offset().expect("red offset");
            let g_off: usize = fmt.green_offset().expect("green offset");
            let b_off: usize = fmt.blue_offset().expect("blue offset");
            let mut mismatches: usize = 0;
            for (i, &v) in gray.data.iter().enumerate() {
                let px: &[u8] = &img.data[i * bpp..i * bpp + bpp];
                let mut ok: bool = px[r_off] == v && px[g_off] == v && px[b_off] == v;
                if bpp == 4 {
                    ok &= px[6 - r_off - g_off - b_off] == 255;
                }
                if !ok {
                    mismatches += 1;
                }
            }
            assert_eq!(mismatches, 0, "{fmt:?}: channel placement");
        }
    }

    // Rgb565: packed gray.
    let advertised: usize =
        output_buffer_size(&jpeg, PixelFormat::Rgb565).expect("output_buffer_size 565");
    let img565 = decompress_to(&jpeg, PixelFormat::Rgb565).expect("565 decode");
    assert_eq!(img565.pixel_format, PixelFormat::Rgb565);
    assert_eq!((img565.width, img565.height), (width, height), "565 dims");
    assert_eq!(img565.data.len(), advertised);
    let v: u8 = gray.data[0];
    let packed: u16 = (((v as u16) >> 3) << 11) | (((v as u16) >> 2) << 5) | ((v as u16) >> 3);
    assert_eq!(&img565.data[0..2], &packed.to_ne_bytes(), "565 packing");

    // Cmyk is the one target with no 12-bit conversion: typed error, not
    // a panic and not a wrong-format Ok (drift audit on issue #394).
    assert!(
        matches!(
            decompress_to(&jpeg, PixelFormat::Cmyk),
            Err(libjpeg_turbo_rs::JpegError::Unsupported(_))
        ),
        "12-bit gray -> Cmyk must be a typed Unsupported error"
    );

    // decompress_into must agree byte-for-byte with decompress_to on
    // this path (the #369 leg (e) pattern; the 12-bit route stages
    // through a copy today — pin the equality, not the mechanism).
    for &fmt in formats {
        let reference = decompress_to(&jpeg, fmt).expect("decompress_to");
        let mut buf: Vec<u8> = vec![0u8; reference.data.len()];
        libjpeg_turbo_rs::decompress_into(&jpeg, fmt, &mut buf)
            .unwrap_or_else(|e| panic!("{fmt:?}: decompress_into failed: {e}"));
        assert_eq!(buf, reference.data, "{fmt:?}: decompress_into parity");
    }

    // set_dither_565 must be honoured, exactly like the 8-bit grayscale
    // path: the dithered output goes through the same row-aware kernel,
    // so it must differ from plain truncation somewhere on this gradient.
    let mut dec = libjpeg_turbo_rs::Decoder::new(&jpeg).expect("header");
    dec.set_output_format(PixelFormat::Rgb565);
    dec.set_dither_565(true);
    let dithered = dec.decode_image().expect("dithered 565 decode");
    assert_eq!(dithered.data.len(), advertised);
    assert_ne!(
        dithered.data, img565.data,
        "ordered dither must change at least one pixel on this gradient"
    );
}

/// C cross-validation: `djpeg -rgb` on a `cjpeg -precision 12` grayscale
/// JPEG produces a P6 (RGB) PPM whose channels are the gray value
/// tripled — our Rgb output must match it exactly.
#[test]
fn issue_394_c_djpeg_rgb_expansion_matches() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    if !cjpeg_supports_precision(&cjpeg) {
        eprintln!("SKIP: cjpeg lacks -precision (need libjpeg-turbo 3.x)");
        return;
    }

    let (width, height): (usize, usize) = (32, 20);
    // 12-bit PGM (maxval 4095) for cjpeg.
    let samples: Vec<u16> = (0..width * height)
        .map(|i| ((i * 131) % 4096) as u16)
        .collect();
    let mut pgm: Vec<u8> = format!("P5\n{width} {height}\n4095\n").into_bytes();
    for &s in &samples {
        pgm.extend_from_slice(&s.to_be_bytes());
    }
    let pid: u32 = std::process::id();
    let tmp_pgm: PathBuf = std::env::temp_dir().join(format!("issue394_{pid}.pgm"));
    let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("issue394_{pid}.jpg"));
    let tmp_ppm: PathBuf = std::env::temp_dir().join(format!("issue394_{pid}.ppm"));
    std::fs::write(&tmp_pgm, &pgm).expect("write pgm");

    let enc = Command::new(&cjpeg)
        .args(["-precision", "12", "-quality", "92", "-outfile"])
        .arg(&tmp_jpg)
        .arg(&tmp_pgm)
        .output()
        .expect("run cjpeg");
    assert!(
        enc.status.success(),
        "cjpeg -precision 12 failed: {}",
        String::from_utf8_lossy(&enc.stderr)
    );
    let jpeg: Vec<u8> = std::fs::read(&tmp_jpg).expect("read jpg");

    let dec = Command::new(&djpeg)
        .args(["-rgb", "-ppm", "-outfile"])
        .arg(&tmp_ppm)
        .arg(&tmp_jpg)
        .output()
        .expect("run djpeg");
    assert!(
        dec.status.success(),
        "djpeg -rgb failed: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    let ppm: Vec<u8> = std::fs::read(&tmp_ppm).expect("read ppm");
    assert_eq!(&ppm[0..2], b"P6", "djpeg -rgb on 12-bit gray must emit RGB");
    // Parse header: P6 <w> <h> <maxval>\n. For a 12-bit decode djpeg
    // emits maxval 4095 with 2-byte big-endian samples — it does not
    // downscale, so our documented 12->8 scaling (v * 255 / 4095) is
    // applied to C's samples before comparing.
    let (maxval, header_end): (usize, usize) = {
        let mut fields: Vec<usize> = Vec::new();
        let mut idx: usize = 2;
        while fields.len() < 3 {
            while ppm[idx].is_ascii_whitespace() {
                idx += 1;
            }
            let start: usize = idx;
            while ppm[idx].is_ascii_digit() {
                idx += 1;
            }
            fields.push(
                std::str::from_utf8(&ppm[start..idx])
                    .unwrap()
                    .parse()
                    .unwrap(),
            );
        }
        (fields[2], idx + 1)
    };
    assert_eq!(maxval, 4095, "djpeg 12-bit output maxval");
    let payload: &[u8] = &ppm[header_end..];
    assert_eq!(payload.len(), width * height * 3 * 2, "C RGB payload size");
    let c_scaled: Vec<u8> = payload
        .chunks_exact(2)
        .map(|be| {
            let v: u16 = u16::from_be_bytes([be[0], be[1]]);
            (v.min(4095) as u32 * 255 / 4095) as u8
        })
        .collect();

    let ours = decompress_to(&jpeg, PixelFormat::Rgb).expect("our Rgb decode");
    assert_eq!(ours.pixel_format, PixelFormat::Rgb);
    assert_eq!(
        ours.data, c_scaled,
        "12-bit gray -> Rgb must match djpeg -rgb (12->8 scaled) byte-exactly"
    );

    let _ = std::fs::remove_file(&tmp_pgm);
    let _ = std::fs::remove_file(&tmp_jpg);
    let _ = std::fs::remove_file(&tmp_ppm);
}

/// P4-68 (found by the #394 drift audit): requesting Cmyk output from a
/// non-CMYK source used to PANIC in three separate 8-bit arms
/// (baseline colour at the unreachable colour-row match, baseline
/// grayscale, lossless grayscale). C raises JERR_CONVERSION_NOTIMPL
/// (jdcolor.c) — a typed error, never an abort. CMYK sources keep
/// working.
#[test]
fn p4_68_cmyk_request_on_non_cmyk_sources_errors_not_panics() {
    let rgb: Vec<u8> = (0..16 * 8 * 3).map(|i| (i % 256) as u8).collect();
    let gray: Vec<u8> = (0..16 * 8).map(|i| (i % 256) as u8).collect();

    let baseline_color = libjpeg_turbo_rs::compress(
        &rgb,
        16,
        8,
        PixelFormat::Rgb,
        90,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("encode colour");
    let baseline_gray = libjpeg_turbo_rs::compress(
        &gray,
        16,
        8,
        PixelFormat::Grayscale,
        90,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("encode gray");
    let lossless_gray = libjpeg_turbo_rs::compress_lossless(&gray, 16, 8, PixelFormat::Grayscale)
        .expect("encode lossless gray");

    for (label, jpeg) in [
        ("baseline colour", &baseline_color),
        ("baseline grayscale", &baseline_gray),
        ("lossless grayscale", &lossless_gray),
    ] {
        assert!(
            matches!(
                decompress_to(jpeg, PixelFormat::Cmyk),
                Err(libjpeg_turbo_rs::JpegError::Unsupported(_))
            ),
            "{label}: Cmyk request must be a typed error (was a panic, P4-68)"
        );
    }

    // Control: a real CMYK source still decodes to Cmyk.
    let cmyk: Vec<u8> = (0..16 * 8 * 4).map(|i| (i % 256) as u8).collect();
    let cmyk_jpeg = libjpeg_turbo_rs::compress(
        &cmyk,
        16,
        8,
        PixelFormat::Cmyk,
        90,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("encode cmyk");
    let out = decompress_to(&cmyk_jpeg, PixelFormat::Cmyk).expect("cmyk decode");
    assert_eq!(out.pixel_format, PixelFormat::Cmyk);
    assert_eq!(out.data.len(), 16 * 8 * 4);
}
