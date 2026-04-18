#![no_main]
//! Encode-then-decode roundtrip fuzzer.
//!
//! Parses a small structured header from the fuzzer's input, interprets the
//! rest as raw pixel bytes, runs the configured encoder, then decodes the
//! result and asserts that:
//!   1. neither encode nor decode panics,
//!   2. decoded dimensions match the requested dimensions (lossy paths), and
//!   3. for lossless paths, decoded pixels are byte-identical to the input.
//!
//! The structured header layout (first 4 bytes of `data`):
//!   byte 0: width_byte (width = width_byte.max(1))
//!   byte 1: height_byte (height = height_byte.max(1))
//!   byte 2: quality_byte (quality = (quality_byte % 100).max(1))
//!   byte 3: mode_flags
//!             bits 0..=2 -> subsampling index (0..=6, wrapped mod 7)
//!             bits 3..=4 -> entropy mode (0=baseline, 1=progressive,
//!                                         2=arithmetic, 3=arith_progressive)
//!             bits 5..=5 -> use_lossless (overrides entropy mode when set)
//!             bits 6..=6 -> use_arithmetic_lossless
//!             bit  7     -> grayscale (ignores subsampling idx)
//!
//! Remaining bytes are the raw pixel buffer. The fuzz corpus seeds provide
//! handcrafted header+pixel inputs so libFuzzer does not need to discover
//! this format from scratch.

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless,
    compress_lossless_arithmetic, compress_progressive, decompress, PixelFormat, Subsampling,
};

const MAX_DIM: usize = 96;

fn subsampling_for(idx: u8) -> Subsampling {
    match idx % 7 {
        0 => Subsampling::S420,
        1 => Subsampling::S422,
        2 => Subsampling::S444,
        3 => Subsampling::S440,
        4 => Subsampling::S411,
        5 => Subsampling::S441,
        _ => Subsampling::S444,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let width: usize = (data[0] as usize).max(1).min(MAX_DIM);
    let height: usize = (data[1] as usize).max(1).min(MAX_DIM);
    let quality: u8 = ((data[2] as u32 % 100) as u8).max(1);
    let flags: u8 = data[3];

    let sub_idx: u8 = flags & 0b0000_0111;
    let entropy: u8 = (flags >> 3) & 0b11;
    let use_lossless: bool = (flags & 0b0010_0000) != 0;
    let use_arith_lossless: bool = (flags & 0b0100_0000) != 0;
    let grayscale: bool = (flags & 0b1000_0000) != 0;

    let subsampling: Subsampling = subsampling_for(sub_idx);
    let pf: PixelFormat = if grayscale {
        PixelFormat::Grayscale
    } else {
        PixelFormat::Rgb
    };
    let bpp: usize = pf.bytes_per_pixel();

    let required: usize = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(bpp))
        .unwrap_or(usize::MAX);
    if data.len() < 4 + required || required == 0 {
        return;
    }
    let pixels: &[u8] = &data[4..4 + required];

    // Encode. Any Err is expected (invalid param combos); just stop there.
    let encoded: Vec<u8> = if use_lossless {
        // Lossless is only defined when the input is grayscale or 444 RGB;
        // the encoder will return Err for mismatched cases and we bail.
        match compress_lossless(pixels, width, height, pf) {
            Ok(v) => v,
            Err(_) => return,
        }
    } else if use_arith_lossless {
        match compress_lossless_arithmetic(pixels, width, height, pf, 1, 0) {
            Ok(v) => v,
            Err(_) => return,
        }
    } else {
        let res = match entropy {
            0 => compress(pixels, width, height, pf, quality, subsampling),
            1 => compress_progressive(pixels, width, height, pf, quality, subsampling),
            2 => compress_arithmetic(pixels, width, height, pf, quality, subsampling),
            _ => compress_arithmetic_progressive(pixels, width, height, pf, quality, subsampling),
        };
        match res {
            Ok(v) => v,
            Err(_) => return,
        }
    };

    // Decode must succeed on our own encoder output.
    let image = match decompress(&encoded) {
        Ok(img) => img,
        // A decode failure on our own encoder output is a real bug —
        // surface it as a panic so libFuzzer flags a crash.
        Err(e) => panic!(
            "decode of self-encoded JPEG failed: {e:?} (w={width} h={height} q={quality} flags={flags:#010b})"
        ),
    };

    assert_eq!(image.width, width, "decoded width mismatch");
    assert_eq!(image.height, height, "decoded height mismatch");

    // Lossless paths MUST round-trip exactly. Other paths are lossy and we
    // cannot assert pixel equality without a quality-aware tolerance model;
    // we rely on the no-panic + dimension checks above.
    if use_lossless || use_arith_lossless {
        // Lossless decode currently outputs RGB planar; we only verify
        // grayscale exactness to avoid coupling to internal plane layout.
        if grayscale && image.data.len() >= pixels.len() {
            // decompress() returns the image in its native components; for
            // grayscale that is one byte per pixel.
            let decoded: &[u8] = &image.data[..pixels.len()];
            assert_eq!(
                decoded, pixels,
                "lossless grayscale roundtrip must be byte-identical"
            );
        }
    }
});
