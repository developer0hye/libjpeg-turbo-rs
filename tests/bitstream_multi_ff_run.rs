//! Regression test for multi-`0xFF` runs inside the entropy-coded segment.
//!
//! libjpeg-turbo's `jpeg_fill_bit_buffer` (jdhuff.c:316–331) walks past any
//! consecutive `0xFF` fill bytes before classifying the next byte:
//!
//! ```text
//!   FF 00          -> stuffed 0xFF data byte
//!   FF FF ... 00   -> still a stuffed 0xFF data byte (extra FFs are
//!                     fill bytes; technically not standards-compliant
//!                     but accepted)
//!   FF ... <marker> -> end of compressed segment
//! ```
//!
//! Our prior `BitReader` only handled the first form: on the first `FF FF`
//! the fast path bailed to the "marker — push zero, don't advance" branch,
//! which then looped forever pushing zeros at the same byte position. The
//! resulting all-zero bits over-consumed bandwidth in the Y plane and
//! starved Cb/Cr to immediate-EOB, producing achromatic (R=G=B) output for
//! any baseline JPEG that contained an `FF FF` run mid-scan.
//!
//! Surfaced by a 682-byte 16×16 baseline 4:2:0 RGB fixture from
//! `fuzz_decode_diff_c` whose entropy stream contains both `FF 00 FF FF FF
//! 00` and `FF FF FF FF` patterns; djpeg decodes a colorful image, the old
//! Rust decoder produced a fully achromatic gradient.

use libjpeg_turbo_rs::{Decoder, PixelFormat};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

mod helpers;

/// 682-byte baseline 16×16 RGB (Y 4:2:0, Cb/Cr 1×1) fixture exercising
/// `FF FF` and `FF FF FF` runs inside the SOS payload.
#[rustfmt::skip]
const MULTI_FF_RUN_FIXTURE: &[u8] = &[
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1,
    1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67,
    0, 3, 2, 2, 3, 2, 2, 3, 3, 3, 3, 4,
    3, 3, 4, 5, 8, 5, 5, 4, 4, 5, 10, 7,
    7, 6, 8, 12, 10, 12, 12, 11, 10, 11, 11, 13,
    14, 18, 16, 13, 14, 17, 14, 11, 11, 16, 22, 16,
    17, 19, 20, 21, 21, 21, 12, 15, 23, 24, 22, 20,
    24, 18, 20, 21, 20, 255, 219, 0, 67, 1, 3, 4,
    4, 5, 4, 5, 9, 5, 5, 9, 20, 13, 11, 13,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 255, 192, 0, 17, 8, 0, 16, 0, 16, 3,
    1, 34, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0,
    31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16, 0,
    2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0,
    0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33,
    49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129,
    145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36,
    51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37,
    38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58,
    67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86,
    87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106,
    115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134,
    135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198,
    199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217,
    218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 196, 0,
    31, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 17, 0,
    2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0,
    1, 2, 119, 0, 1, 2, 3, 17, 4, 5, 33, 49,
    6, 18, 65, 81, 7, 97, 113, 19, 34, 50, 129, 8,
    20, 66, 145, 161, 177, 193, 9, 35, 51, 82, 240, 21,
    98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24,
    25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57,
    58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85,
    86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105,
    106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132,
    133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196,
    197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215,
    216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0,
    12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0, 248,
    247, 193, 159, 8, 126, 231, 238, 127, 74, 247, 159, 5,
    252, 34, 255, 0, 87, 251, 159, 210, 189, 139, 193, 127,
    8, 126, 231, 238, 63, 74, 247, 175, 5, 252, 33, 254,
    255, 255, 255, 255, 0, 87, 251, 143, 210, 140, 30, 51,
    109, 67, 195, 207, 16, 254, 15, 124, 255, 217,
];

fn decode_via_djpeg(djpeg: &PathBuf, jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let mut child = Command::new(djpeg)
        .arg("-pnm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let payload = jpeg.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() {
        return None;
    }
    let pnm = out.stdout;
    let mut i: usize = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && i < pnm.len() {
        while i < pnm.len() && pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < pnm.len() && !pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            tokens.push(String::from_utf8(pnm[start..i].to_vec()).ok()?);
        }
    }
    if tokens.len() < 4 {
        return None;
    }
    let channels = match tokens[0].as_str() {
        "P5" => 1,
        "P6" => 3,
        _ => return None,
    };
    let w: usize = tokens[1].parse().ok()?;
    let h: usize = tokens[2].parse().ok()?;
    if tokens[3] != "255" {
        return None;
    }
    i += 1;
    let needed = w.checked_mul(h)?.checked_mul(channels)?;
    if pnm.len() < i + needed {
        return None;
    }
    Some((w, h, channels, pnm[i..i + needed].to_vec()))
}

#[test]
fn multi_ff_run_decodes_chroma_byte_exact_vs_djpeg() {
    let mut d = Decoder::new(MULTI_FF_RUN_FIXTURE).expect("header parse");
    d.set_lenient(true);
    let img = d
        .decode_image()
        .expect("multi-FF run must not stall the BitReader");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);

    // Fast structural assertion that the BitReader actually advanced past
    // the FF runs and produced colorful chroma — pre-fix output was
    // fully achromatic (R == G == B for every pixel).
    let mut achromatic_pixels: usize = 0;
    for px in img.data.chunks_exact(3) {
        if px[0] == px[1] && px[1] == px[2] {
            achromatic_pixels += 1;
        }
    }
    assert!(
        achromatic_pixels < img.data.len() / 3,
        "all pixels achromatic ({} of {}) — multi-FF run regression: \
         BitReader stalled inside the SOS payload and starved Cb/Cr to EOB",
        achromatic_pixels,
        img.data.len() / 3
    );

    // Locate djpeg via the shared helpers' homebrew → PATH lookup.
    // `require_c_tool!` panics in CI when missing so the byte-exact
    // gate cannot be silently skipped on a CI image without
    // libjpeg-turbo-progs installed; locally it falls back to a
    // logged skip so dev machines without djpeg still pass.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let Some((cw, ch, cc, c_px)) = decode_via_djpeg(&djpeg, MULTI_FF_RUN_FIXTURE) else {
        panic!("djpeg unexpectedly failed on the pinned multi-FF fixture");
    };
    assert_eq!(cw, 16);
    assert_eq!(ch, 16);
    assert_eq!(cc, 3);

    let mut max_diff: i32 = 0;
    for (a, b) in c_px.iter().zip(img.data.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert_eq!(
        max_diff, 0,
        "BitReader multi-FF handling must produce byte-exact pixels vs djpeg"
    );
}
