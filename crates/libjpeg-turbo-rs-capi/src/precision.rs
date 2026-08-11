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

use crate::alloc::{deliver_compressed_output, OutputDelivery};
use crate::convert::pixel_format_from_tj;
use crate::tj3::{with_handle, TJERR_FATAL};

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
///
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `jpeg_buf`, `jpeg_size` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
///
/// A non-null `*jpeg_buf` is **freed by this function only when
/// `TJPARAM_NOREALLOC` is unset** — see
/// [Ownership transfer](crate#pointer-contract). With the flag set the output
/// is written in place, your pointer comes back unchanged, and `*jpeg_size` is
/// an *input* carrying the buffer's capacity: output that does not fit is an
/// error rather than a resize (P4-145).
#[no_mangle]
pub unsafe extern "C" fn tj3Compress12(
    handle: *mut c_void,
    src_buf: *const c_short,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
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
            let total_samples: usize = match line_samples.checked_mul(h) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3Compress12: pitch * height overflows", TJERR_FATAL);
                    return -1;
                }
            };
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

            // Mirror `references/libjpeg-turbo/src/turbojpeg-mp.c::tj3Compress*`
            // (lines 109-117): the entry-point's natural precision (12 here) is
            // the default; only honour an explicit `TJPARAM_PRECISION` when
            // lossless is active and the value falls inside the entry-point's
            // range (BITS_IN_JSAMPLE - 3 .. BITS_IN_JSAMPLE → 9..=12 here).
            // Out-of-range values silently fall back to the default rather
            // than erroring — upstream relies on the libjpeg layer for the
            // canonical "bad precision" diagnostic.
            let is_lossless: bool = inst.inner.get(TjParam::Lossless) != 0;
            let raw_precision: i32 = inst.inner.get(TjParam::Precision);
            let effective_precision: i32 = if is_lossless && (9..=12).contains(&raw_precision) {
                raw_precision
            } else {
                12
            };

            // ITU-T T.81 / Annex H: in lossless mode the point transform Pt
            // shifts the lower Pt bits off each sample, so Pt must be strictly
            // less than the sample precision (Pt == P would zero every
            // sample). Mirror C's encode-side Al checks in
            // `jcparam.c::jpeg_enable_lossless` (jcparam.c:586-589) and
            // `jcmaster.c::validate_script` (jcmaster.c:396-399).
            if is_lossless {
                let point_transform: i32 = inst.inner.get(TjParam::LosslessPt);
                if point_transform >= effective_precision {
                    inst.set_error(
                    format!(
                        "tj3Compress12: TJPARAM_LOSSLESSPT {point_transform} must be < TJPARAM_PRECISION {effective_precision}"
                    ),
                    TJERR_FATAL,
                );
                    return -1;
                }
            }

            let jpeg: Vec<u8> = if is_lossless {
                // Widen i16 → u16. `dense` is filled with non-negative samples
                // bounded by `2^effective_precision - 1`, so the cast preserves
                // the numeric value.
                let widened: Vec<u16> = dense.iter().map(|&v| v as u16).collect();
                match inst.inner.compress_16bit_with_precision(
                    &widened,
                    w,
                    h,
                    components,
                    effective_precision as u8,
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
                    effective_precision as u8,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        inst.set_error(format!("tj3Compress12: {e}"), TJERR_FATAL);
                        return -1;
                    }
                }
            };

            // SAFETY: out-pointers validated non-NULL above.
            // P4-145: honour `TJPARAM_NOREALLOC`. This used to allocate a fresh
            // buffer and `free()` the previous pointee unconditionally — including
            // when the caller had set the flag, which is exactly when the buffer is
            // not `malloc`-owned. A stack array handed here was freed with the
            // wrong allocator.
            let norealloc: bool = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
            // SAFETY: both out-pointers were validated non-NULL above; the caller's
            // contract covers the buffer's validity and non-aliasing with `jpeg`,
            // which this function owns.
            match unsafe { deliver_compressed_output(&jpeg, jpeg_buf, jpeg_size, norealloc) } {
                OutputDelivery::Delivered => {}
                OutputDelivery::BufferTooSmall { needed, capacity } => {
                    inst.set_error(
                        format!(
                            "tj3Compress12: TJPARAM_NOREALLOC is set and the JPEG buffer is too small \
                             ({needed} bytes needed, {capacity} available)"
                        ),
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::NoBufferSupplied => {
                    inst.set_error(
                        "tj3Compress12: TJPARAM_NOREALLOC is set but no output buffer was supplied",
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::OutOfMemory => {
                    inst.set_error("tj3Compress12: out-of-memory", TJERR_FATAL);
                    return -1;
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

/// `tj3Decompress12(handle, jpegBuf, jpegSize, dstBuf, pitch,
///                  pixelFormat) -> int`.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3Decompress12(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut c_short,
    pitch: c_int,
    pixel_format: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
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
            let total: usize = match line_samples.checked_mul(img.height) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3Decompress12: pitch * height overflows", TJERR_FATAL);
                    return -1;
                }
            };
            let out: &mut [i16] = unsafe { std::slice::from_raw_parts_mut(dst_buf, total) };
            let row_samples: usize = img.width * components;
            for row in 0..img.height {
                let s: &[i16] = &img.data[row * row_samples..row * row_samples + row_samples];
                let d: &mut [i16] = &mut out[row * line_samples..row * line_samples + row_samples];
                d.copy_from_slice(s);
            }

            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

/// `tj3Compress16(handle, srcBuf, width, pitch, height, pixelFormat,
///                jpegBuf, jpegSize) -> int`.
///
/// 16-bit is lossless-only. Uses the handle's `LOSSLESSPSV` /
/// `LOSSLESSPT` parameters.
///
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `jpeg_buf`, `jpeg_size` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
///
/// A non-null `*jpeg_buf` is **freed by this function only when
/// `TJPARAM_NOREALLOC` is unset** — see
/// [Ownership transfer](crate#pointer-contract). With the flag set the output
/// is written in place, your pointer comes back unchanged, and `*jpeg_size` is
/// an *input* carrying the buffer's capacity: output that does not fit is an
/// error rather than a resize (P4-145).
#[no_mangle]
pub unsafe extern "C" fn tj3Compress16(
    handle: *mut c_void,
    src_buf: *const u16,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
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

            let total: usize = match line_samples.checked_mul(h) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3Compress16: pitch * height overflows", TJERR_FATAL);
                    return -1;
                }
            };
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

            // 16-bit is always lossless (SOF3) — upstream
            // `references/libjpeg-turbo/src/turbojpeg-mp.c::tj3Compress16`
            // gates `TJPARAM_PRECISION` on the lossless flag and silently
            // ignores out-of-range values, falling back to BITS_IN_JSAMPLE.
            // We mirror that behaviour so a caller that only ever called
            // `tj3Set(handle, TJPARAM_LOSSLESSPSV, 1)` continues to encode
            // a 16-bit lossless stream.
            let is_lossless: bool = inst.inner.get(TjParam::Lossless) != 0;
            let raw_precision: i32 = inst.inner.get(TjParam::Precision);
            let effective_precision: i32 = if is_lossless && (13..=16).contains(&raw_precision) {
                raw_precision
            } else {
                16
            };
            // ITU-T T.81 / Annex H: in lossless mode the point transform Pt
            // shifts the lower Pt bits off each sample, so Pt must be strictly
            // less than the sample precision. Mirror C's encode-side Al checks
            // in `jcparam.c::jpeg_enable_lossless` (jcparam.c:586-589) and
            // `jcmaster.c::validate_script` (jcmaster.c:396-399).
            let point_transform: i32 = inst.inner.get(TjParam::LosslessPt);
            if point_transform >= effective_precision {
                inst.set_error(
                format!(
                    "tj3Compress16: TJPARAM_LOSSLESSPT {point_transform} must be < TJPARAM_PRECISION {effective_precision}"
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
                effective_precision as u8,
            ) {
                Ok(b) => b,
                Err(e) => {
                    inst.set_error(format!("tj3Compress16: {e}"), TJERR_FATAL);
                    return -1;
                }
            };

            // P4-145: honour `TJPARAM_NOREALLOC`. This used to allocate a fresh
            // buffer and `free()` the previous pointee unconditionally — including
            // when the caller had set the flag, which is exactly when the buffer is
            // not `malloc`-owned. A stack array handed here was freed with the
            // wrong allocator.
            let norealloc: bool = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
            // SAFETY: both out-pointers were validated non-NULL above; the caller's
            // contract covers the buffer's validity and non-aliasing with `jpeg`,
            // which this function owns.
            match unsafe { deliver_compressed_output(&jpeg, jpeg_buf, jpeg_size, norealloc) } {
                OutputDelivery::Delivered => {}
                OutputDelivery::BufferTooSmall { needed, capacity } => {
                    inst.set_error(
                        format!(
                            "tj3Compress16: TJPARAM_NOREALLOC is set and the JPEG buffer is too small \
                             ({needed} bytes needed, {capacity} available)"
                        ),
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::NoBufferSupplied => {
                    inst.set_error(
                        "tj3Compress16: TJPARAM_NOREALLOC is set but no output buffer was supplied",
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::OutOfMemory => {
                    inst.set_error("tj3Compress16: out-of-memory", TJERR_FATAL);
                    return -1;
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

/// `tj3Decompress16(handle, jpegBuf, jpegSize, dstBuf, pitch, pixelFormat)
///   -> int`. Lossless 16-bit decompress.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3Decompress16(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u16,
    pitch: c_int,
    pixel_format: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
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

            let total: usize = match line_samples.checked_mul(img.height) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3Decompress16: pitch * height overflows", TJERR_FATAL);
                    return -1;
                }
            };
            let out: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(dst_buf, total) };
            let row_samples: usize = img.width * components;
            for row in 0..img.height {
                let s: &[u16] = &img.data[row * row_samples..row * row_samples + row_samples];
                let d: &mut [u16] = &mut out[row * line_samples..row * line_samples + row_samples];
                d.copy_from_slice(s);
            }

            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}
