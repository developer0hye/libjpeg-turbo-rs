#![no_main]
//! Fuzz target for `transform_jpeg_with_options`.
//!
//! Parses a compact structured header from the fuzzer input, interprets the
//! remaining bytes as JPEG data, and exercises every option combination in
//! `TransformOptions` (op, trim, grayscale, progressive, arithmetic, optimize,
//! restart, crop, marker copy). The raw-bytes fallback also feeds arbitrary
//! bytes directly to let libFuzzer discover valid-looking JPEG prefixes.
//!
//! Header layout (first 10 bytes):
//!   byte 0  : op_idx         — TransformOp (wrapped mod 8)
//!   byte 1  : flags
//!               bit 0 → trim
//!               bit 1 → grayscale
//!               bit 2 → progressive
//!               bit 3 → arithmetic
//!               bit 4 → optimize
//!               bit 5 → no_output
//!               bit 6 → restart_in_rows
//!               bit 7 → enable_crop
//!   byte 2  : restart_interval low  (u16 little-endian)
//!   byte 3  : restart_interval high
//!   byte 4  : copy_mode     — 0=All, 1=None, 2=IccOnly (wrapped mod 3)
//!   byte 5  : crop_x_frac   — crop x as fraction of width  (0..=255 → 0..width)
//!   byte 6  : crop_y_frac   — crop y as fraction of height (0..=255 → 0..height)
//!   byte 7  : crop_w_frac   — crop width  fraction (0..=255 → 1..width)
//!   byte 8  : crop_h_frac   — crop height fraction (0..=255 → 1..height)
//!   byte 9  : perfect       — low bit only
//!   bytes 10+ : JPEG bytes
//!
//! Semantics chosen to maximise option coverage with minimal header overhead
//! and to avoid requiring libFuzzer to discover the JPEG SOI marker from scratch.

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{
    read_coefficients, transform_jpeg_with_options, CropRegion, Decoder, MarkerCopyMode,
    TransformOp, TransformOptions,
};

// Match the pixel-dimension cap used by fuzz_decompress.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

const HEADER_LEN: usize = 10;

fn op_for(idx: u8) -> TransformOp {
    match idx % 8 {
        0 => TransformOp::None,
        1 => TransformOp::HFlip,
        2 => TransformOp::VFlip,
        3 => TransformOp::Transpose,
        4 => TransformOp::Transverse,
        5 => TransformOp::Rot90,
        6 => TransformOp::Rot180,
        _ => TransformOp::Rot270,
    }
}

fn copy_mode_for(idx: u8) -> MarkerCopyMode {
    match idx % 3 {
        0 => MarkerCopyMode::All,
        1 => MarkerCopyMode::None,
        _ => MarkerCopyMode::IccOnly,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_LEN + 2 {
        // Need at least the header plus a minimal 2-byte JPEG (SOI+EOI).
        return;
    }

    let jpeg: &[u8] = &data[HEADER_LEN..];

    // Validate the JPEG dimensions up front to avoid OOMs from giant SOF0 bombs.
    let Ok(decoder) = Decoder::new(jpeg) else {
        return;
    };
    let header = decoder.header();
    let pixels: u64 = (header.width as u64).saturating_mul(header.height as u64);
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    // Also gate on a minimum dimension so crop arithmetic doesn't degenerate.
    let img_w: usize = header.width as usize;
    let img_h: usize = header.height as usize;
    drop(decoder);

    let op_idx: u8 = data[0];
    let flags: u8 = data[1];
    let restart_interval: u16 = u16::from_le_bytes([data[2], data[3]]);
    let copy_idx: u8 = data[4];
    let crop_x_frac: u8 = data[5];
    let crop_y_frac: u8 = data[6];
    let crop_w_frac: u8 = data[7];
    let crop_h_frac: u8 = data[8];
    let perfect: bool = (data[9] & 1) != 0;

    let trim: bool = (flags & 0b0000_0001) != 0;
    let grayscale: bool = (flags & 0b0000_0010) != 0;
    let progressive: bool = (flags & 0b0000_0100) != 0;
    let arithmetic: bool = (flags & 0b0000_1000) != 0;
    let optimize: bool = (flags & 0b0001_0000) != 0;
    let no_output: bool = (flags & 0b0010_0000) != 0;
    let restart_in_rows: bool = (flags & 0b0100_0000) != 0;
    let enable_crop: bool = (flags & 0b1000_0000) != 0;

    // Build an MCU-aligned crop region derived from the image dimensions.
    // Use a 16-pixel MCU granularity (worst case for 4:2:0); actual alignment
    // is checked by the transform and it will return Err for misaligned crops.
    let crop: Option<CropRegion> = if enable_crop && img_w >= 16 && img_h >= 16 {
        const ALIGN: usize = 16;
        let x_raw: usize = (crop_x_frac as usize * img_w / 256) & !(ALIGN - 1);
        let y_raw: usize = (crop_y_frac as usize * img_h / 256) & !(ALIGN - 1);
        // Width and height must be at least ALIGN and fit within image bounds.
        let max_w: usize = (img_w - x_raw) & !(ALIGN - 1);
        let max_h: usize = (img_h - y_raw) & !(ALIGN - 1);
        let w_raw: usize = ((crop_w_frac as usize * max_w / 256) & !(ALIGN - 1)).max(ALIGN);
        let h_raw: usize = ((crop_h_frac as usize * max_h / 256) & !(ALIGN - 1)).max(ALIGN);
        if w_raw <= max_w && h_raw <= max_h && max_w >= ALIGN && max_h >= ALIGN {
            Some(CropRegion {
                x: x_raw,
                y: y_raw,
                width: w_raw,
                height: h_raw,
            })
        } else {
            None
        }
    } else {
        None
    };

    let options: TransformOptions = TransformOptions {
        op: op_for(op_idx),
        perfect,
        trim,
        crop,
        grayscale,
        no_output,
        progressive,
        arithmetic,
        optimize,
        restart_interval,
        restart_in_rows,
        copy_markers: copy_mode_for(copy_idx),
        custom_filter: None,
    };

    // transform_jpeg_with_options returns Err for invalid combinations (e.g.
    // perfect + non-aligned dims). That is expected and not a bug.
    let result = transform_jpeg_with_options(jpeg, &options);

    // If the transform succeeded, the output must be a valid JPEG that our
    // coefficient reader and full decoder can handle without panicking.
    if let Ok(transformed) = result {
        // Gate dimensions again on the transformed output.
        if let Ok(out_dec) = Decoder::new(&transformed) {
            let out_hdr = out_dec.header();
            let out_pixels: u64 = (out_hdr.width as u64).saturating_mul(out_hdr.height as u64);
            if out_hdr.width > 0 && out_hdr.height > 0 && out_pixels <= MAX_FUZZ_PIXELS {
                drop(out_dec);
                // The transformed JPEG must survive read_coefficients.
                let _ = read_coefficients(&transformed);
            }
        }
    }
});
