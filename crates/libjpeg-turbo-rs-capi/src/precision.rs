//! `tj3Compress12` / `tj3Decompress12` / `tj3Compress16` / `tj3Decompress16`.
//!
//! Signatures mirror `turbojpeg.h` exactly; all 12-bit/16-bit samples
//! are passed as contiguous `int16_t` (12-bit) or `uint16_t` (16-bit)
//! arrays with `pitch` stride.
//!
//! 12-bit is lossy (SOF1 / extended sequential DCT) while 16-bit is
//! lossless-only (SOF3 with predictor `LOSSLESSPSV` and point
//! transform `LOSSLESSPT` from the handle).

use std::ffi::{c_int, c_short, c_void};

use libjpeg_turbo_rs::tj3::TjParam;

use crate::alloc::{libc_free, libc_from_slice};
use crate::convert::pixel_format_from_tj;
use crate::tj3::{handle_as_mut, TJERR_FATAL};

fn num_components_from_tjpf(tjpf: c_int) -> Option<usize> {
    // Only Grayscale / RGB / CMYK are supported for non-8-bit precision
    // by libjpeg-turbo itself. Higher-bpp variants with alpha are
    // specifically disallowed by `tj3Compress12/16` in the C header.
    use libjpeg_turbo_rs::PixelFormat;
    match pixel_format_from_tj(tjpf)? {
        PixelFormat::Grayscale => Some(1),
        PixelFormat::Rgb | PixelFormat::Bgr => Some(3),
        PixelFormat::Cmyk => Some(4),
        _ => None,
    }
}

/// `tj3Compress12(handle, srcBuf, width, pitch, height, pixelFormat,
///                jpegBuf, jpegSize) -> int`.
///
/// `srcBuf` is a flat array of signed 12-bit samples; element count is
/// `pitch_samples * height`, where `pitch_samples` is `width *
/// num_components` when `pitch == 0`, else `pitch` (measured in
/// samples, matching `turbojpeg.h` `int pitch` in units of sample
/// count, not bytes).
#[no_mangle]
pub extern "C" fn tj3Compress12(
    handle: *mut c_void,
    src_buf: *const c_short,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    if src_buf.is_null() || jpeg_buf.is_null() || jpeg_size.is_null() {
        inst.set_error(
            "tj3Compress12: NULL srcBuf / jpegBuf / jpegSize",
            TJERR_FATAL,
        );
        return -1;
    }
    if width <= 0 || height <= 0 || pitch < 0 {
        inst.set_error(
            format!("tj3Compress12: invalid dims/pitch ({width}x{height}, pitch={pitch})"),
            TJERR_FATAL,
        );
        return -1;
    }

    let components: usize = match num_components_from_tjpf(pixel_format) {
        Some(c) => c,
        None => {
            inst.set_error(
                format!("tj3Compress12: unsupported TJPF {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    let w: usize = width as usize;
    let h: usize = height as usize;
    let line_samples: usize = if pitch == 0 {
        w * components
    } else {
        pitch as usize
    };
    if line_samples < w * components {
        inst.set_error(
            format!(
                "tj3Compress12: pitch {line_samples} smaller than width*components ({})",
                w * components
            ),
            TJERR_FATAL,
        );
        return -1;
    }

    // SAFETY: caller guarantees `src_buf` is valid for
    // `line_samples * h` samples.
    let total_samples: usize = line_samples.saturating_mul(h);
    let raw: &[i16] = unsafe { std::slice::from_raw_parts(src_buf, total_samples) };

    let dense: Vec<i16> = if line_samples == w * components {
        raw.to_vec()
    } else {
        let mut out: Vec<i16> = Vec::with_capacity(w * components * h);
        for row in 0..h {
            let start: usize = row * line_samples;
            out.extend_from_slice(&raw[start..start + w * components]);
        }
        out
    };

    // Read encoder parameters from the handle. The 12-bit entry-point
    // accepts `TJPARAM_PRECISION` in 9..=12 *when explicitly set*; when
    // the param is left at the handle default (8) we fall back to the
    // entry-point's natural precision (12) so legacy callers that never
    // touch `TJPARAM_PRECISION` keep working.
    //
    // When lossless is active we delegate to the SOF3 lossless encoder
    // (which natively supports an arbitrary precision in 2..=16) by
    // widening the i16 samples to u16 — the bit pattern is preserved
    // because all values sit in the non-negative `precision`-bit range.
    let is_lossless: bool = inst.inner.get(TjParam::Lossless) != 0;
    let raw_precision: i32 = inst.inner.get(TjParam::Precision);
    let stored_precision: i32 = if raw_precision == 8 {
        12
    } else {
        raw_precision
    };
    if !(9..=12).contains(&stored_precision) {
        inst.set_error(
            format!(
                "tj3Compress12: TJPARAM_PRECISION {raw_precision} is out of range 9..=12 for the 12-bit entry point"
            ),
            TJERR_FATAL,
        );
        return -1;
    }

    let jpeg: Vec<u8> = if is_lossless {
        // Widen i16 → u16. `dense` is filled with non-negative samples
        // bounded by `2^stored_precision - 1`, so the cast preserves the
        // numeric value.
        let widened: Vec<u16> = dense.iter().map(|&v| v as u16).collect();
        match inst.inner.compress_16bit_with_precision(
            &widened,
            w,
            h,
            components,
            stored_precision as u8,
        ) {
            Ok(b) => b,
            Err(e) => {
                inst.set_error(format!("tj3Compress12: {e}"), TJERR_FATAL);
                return -1;
            }
        }
    } else {
        match inst.inner.compress_12bit_with_precision(
            &dense,
            w,
            h,
            components,
            stored_precision as u8,
        ) {
            Ok(b) => b,
            Err(e) => {
                inst.set_error(format!("tj3Compress12: {e}"), TJERR_FATAL);
                return -1;
            }
        }
    };

    // SAFETY: out-pointers validated non-NULL above.
    let ptr: *mut u8 = libc_from_slice(&jpeg);
    if ptr.is_null() && !jpeg.is_empty() {
        inst.set_error("tj3Compress12: out-of-memory", TJERR_FATAL);
        return -1;
    }
    unsafe {
        let prior: *mut u8 = *jpeg_buf;
        if !prior.is_null() {
            libc_free(prior);
        }
        *jpeg_buf = ptr;
        *jpeg_size = jpeg.len();
    }
    inst.clear_error();
    0
}

/// `tj3Decompress12(handle, jpegBuf, jpegSize, dstBuf, pitch,
///                  pixelFormat) -> int`.
#[no_mangle]
pub extern "C" fn tj3Decompress12(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut c_short,
    pitch: c_int,
    pixel_format: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    if jpeg_buf.is_null() || dst_buf.is_null() || jpeg_size < 2 || pitch < 0 {
        inst.set_error("tj3Decompress12: invalid NULL / size / pitch", TJERR_FATAL);
        return -1;
    }

    let components: usize = match num_components_from_tjpf(pixel_format) {
        Some(c) => c,
        None => {
            inst.set_error(
                format!("tj3Decompress12: unsupported TJPF {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    // SAFETY: caller guarantees `jpeg_buf` is valid for `jpeg_size` bytes.
    let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };

    let img = match inst.inner.decompress_12bit(jpeg) {
        Ok(i) => i,
        Err(e) => {
            inst.set_error(format!("tj3Decompress12: {e}"), TJERR_FATAL);
            return -1;
        }
    };
    if img.num_components != components {
        inst.set_error(
            format!(
                "tj3Decompress12: TJPF implies {components} components but JPEG has {}",
                img.num_components
            ),
            TJERR_FATAL,
        );
        return -1;
    }

    let line_samples: usize = if pitch == 0 {
        img.width * components
    } else {
        pitch as usize
    };
    if line_samples < img.width * components {
        inst.set_error(
            "tj3Decompress12: pitch too small for width*components",
            TJERR_FATAL,
        );
        return -1;
    }

    // SAFETY: caller guarantees `dst_buf` holds at least
    // `line_samples * height` samples.
    let total: usize = line_samples.saturating_mul(img.height);
    let out: &mut [i16] = unsafe { std::slice::from_raw_parts_mut(dst_buf, total) };
    let row_samples: usize = img.width * components;
    for row in 0..img.height {
        let s: &[i16] = &img.data[row * row_samples..row * row_samples + row_samples];
        let d: &mut [i16] = &mut out[row * line_samples..row * line_samples + row_samples];
        d.copy_from_slice(s);
    }

    inst.clear_error();
    0
}

/// `tj3Compress16(handle, srcBuf, width, pitch, height, pixelFormat,
///                jpegBuf, jpegSize) -> int`.
///
/// 16-bit is lossless-only. Uses the handle's `LOSSLESSPSV` /
/// `LOSSLESSPT` parameters.
#[no_mangle]
pub extern "C" fn tj3Compress16(
    handle: *mut c_void,
    src_buf: *const u16,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    if src_buf.is_null() || jpeg_buf.is_null() || jpeg_size.is_null() {
        inst.set_error(
            "tj3Compress16: NULL srcBuf / jpegBuf / jpegSize",
            TJERR_FATAL,
        );
        return -1;
    }
    if width <= 0 || height <= 0 || pitch < 0 {
        inst.set_error(
            format!("tj3Compress16: invalid dims/pitch ({width}x{height}, pitch={pitch})"),
            TJERR_FATAL,
        );
        return -1;
    }

    let components: usize = match num_components_from_tjpf(pixel_format) {
        Some(c) => c,
        None => {
            inst.set_error(
                format!("tj3Compress16: unsupported TJPF {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    let w: usize = width as usize;
    let h: usize = height as usize;
    let line_samples: usize = if pitch == 0 {
        w * components
    } else {
        pitch as usize
    };
    if line_samples < w * components {
        inst.set_error(
            "tj3Compress16: pitch smaller than width*components",
            TJERR_FATAL,
        );
        return -1;
    }

    let total: usize = line_samples.saturating_mul(h);
    // SAFETY: caller guarantees `src_buf` is valid for `total` u16s.
    let raw: &[u16] = unsafe { std::slice::from_raw_parts(src_buf, total) };
    let dense: Vec<u16> = if line_samples == w * components {
        raw.to_vec()
    } else {
        let mut out: Vec<u16> = Vec::with_capacity(w * components * h);
        for row in 0..h {
            let start: usize = row * line_samples;
            out.extend_from_slice(&raw[start..start + w * components]);
        }
        out
    };

    // 16-bit is always lossless (SOF3). Honour `TJPARAM_PRECISION` for the
    // 16-bit entry-point range (13..=16) so callers can request narrower
    // precision (e.g. 14-bit medical images). When the param is left at
    // the handle default (8) we fall back to the entry-point's natural
    // precision (16) so legacy callers that never touch
    // `TJPARAM_PRECISION` keep working.
    let raw_precision: i32 = inst.inner.get(TjParam::Precision);
    let stored_precision: i32 = if raw_precision == 8 {
        16
    } else {
        raw_precision
    };
    if !(13..=16).contains(&stored_precision) {
        inst.set_error(
            format!(
                "tj3Compress16: TJPARAM_PRECISION {raw_precision} is out of range 13..=16 for the 16-bit entry point"
            ),
            TJERR_FATAL,
        );
        return -1;
    }
    let jpeg: Vec<u8> = match inst.inner.compress_16bit_with_precision(
        &dense,
        w,
        h,
        components,
        stored_precision as u8,
    ) {
        Ok(b) => b,
        Err(e) => {
            inst.set_error(format!("tj3Compress16: {e}"), TJERR_FATAL);
            return -1;
        }
    };

    let ptr: *mut u8 = libc_from_slice(&jpeg);
    if ptr.is_null() && !jpeg.is_empty() {
        inst.set_error("tj3Compress16: out-of-memory", TJERR_FATAL);
        return -1;
    }
    // SAFETY: jpeg_buf / jpeg_size validated non-NULL.
    unsafe {
        let prior: *mut u8 = *jpeg_buf;
        if !prior.is_null() {
            libc_free(prior);
        }
        *jpeg_buf = ptr;
        *jpeg_size = jpeg.len();
    }
    inst.clear_error();
    0
}

/// `tj3Decompress16(handle, jpegBuf, jpegSize, dstBuf, pitch, pixelFormat)
///   -> int`. Lossless 16-bit decompress.
#[no_mangle]
pub extern "C" fn tj3Decompress16(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u16,
    pitch: c_int,
    pixel_format: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    if jpeg_buf.is_null() || dst_buf.is_null() || jpeg_size < 2 || pitch < 0 {
        inst.set_error("tj3Decompress16: invalid NULL / size / pitch", TJERR_FATAL);
        return -1;
    }

    let components: usize = match num_components_from_tjpf(pixel_format) {
        Some(c) => c,
        None => {
            inst.set_error(
                format!("tj3Decompress16: unsupported TJPF {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
    let img = match inst.inner.decompress_16bit(jpeg) {
        Ok(i) => i,
        Err(e) => {
            inst.set_error(format!("tj3Decompress16: {e}"), TJERR_FATAL);
            return -1;
        }
    };
    if img.num_components != components {
        inst.set_error(
            format!(
                "tj3Decompress16: TJPF implies {components} components but JPEG has {}",
                img.num_components
            ),
            TJERR_FATAL,
        );
        return -1;
    }

    let line_samples: usize = if pitch == 0 {
        img.width * components
    } else {
        pitch as usize
    };
    if line_samples < img.width * components {
        inst.set_error(
            "tj3Decompress16: pitch too small for width*components",
            TJERR_FATAL,
        );
        return -1;
    }

    let total: usize = line_samples.saturating_mul(img.height);
    let out: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(dst_buf, total) };
    let row_samples: usize = img.width * components;
    for row in 0..img.height {
        let s: &[u16] = &img.data[row * row_samples..row * row_samples + row_samples];
        let d: &mut [u16] = &mut out[row * line_samples..row * line_samples + row_samples];
        d.copy_from_slice(s);
    }

    inst.clear_error();
    0
}
