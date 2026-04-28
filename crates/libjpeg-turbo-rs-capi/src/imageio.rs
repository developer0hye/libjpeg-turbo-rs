//! TJ3 `LoadImage*` / `SaveImage*` shims.
//!
//! `tj3LoadImage8` / `tj3SaveImage8` route through the Rust BMP/PPM
//! helpers in `libjpeg_turbo_rs::api::image_io`. Returned buffers are
//! libc-allocated so callers can release them with `tj3Free` or
//! `free()` (matching upstream's contract).
//!
//! 12-bit and 16-bit forms remain stubs because the underlying Rust
//! image I/O is currently 8-bit-only. They record an error and
//! return NULL / `-1` so callers see a clear "not supported yet"
//! message rather than silent corruption.

use std::ffi::{c_char, c_int, c_void, CStr};

use libjpeg_turbo_rs::PixelFormat;

use crate::alloc::{libc_free, libc_from_slice};
use crate::convert::{pixel_format_from_tj, pixel_format_to_tj, TJPF_BGR, TJPF_RGB};
use crate::tj3::{handle_as_mut, TJERR_FATAL};

/// In-place R↔B swap on RGB or BGR row data. `bytes` must be a
/// 3-byte-per-pixel buffer; trailing bytes (e.g. partial last
/// pixel) are ignored. Used to flip between TJPF_RGB and TJPF_BGR.
fn rb_swap_3bpp(bytes: &mut [u8]) {
    for px in bytes.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

/// Reverse the row order of a buffer in-place. `row_bytes` is the
/// stride in bytes; the trailing bytes (if `bytes.len() % row_bytes
/// != 0`) are left alone. Used to honour `TJPARAM_BOTTOMUP`.
fn flip_rows_in_place(bytes: &mut [u8], row_bytes: usize) {
    if row_bytes == 0 {
        return;
    }
    let rows: usize = bytes.len() / row_bytes;
    let mut top: usize = 0;
    let mut bot: usize = rows.saturating_sub(1);
    while top < bot {
        let (lo, hi) = bytes.split_at_mut(bot * row_bytes);
        lo[top * row_bytes..top * row_bytes + row_bytes].swap_with_slice(&mut hi[..row_bytes]);
        top += 1;
        bot -= 1;
    }
}

/// Strip alpha / padding bytes from a 4-bpp packed-pixel buffer to
/// produce dense 3-bpp RGB or BGR. `pf` chooses which 3 channels to
/// keep:
/// - `TJPF_RGBX` / `TJPF_RGBA` → keep bytes 0..3 (R, G, B), drop 3
/// - `TJPF_BGRX` / `TJPF_BGRA` → same as above, but pixels are BGR
/// - `TJPF_XRGB` / `TJPF_ARGB` → keep bytes 1..4 (skip alpha at 0)
/// - `TJPF_XBGR` / `TJPF_ABGR` → keep bytes 1..4 (skip alpha at 0,
///   pixels are BGR)
///
/// Returns the dense buffer plus its 3-bpp format (TJPF_RGB or
/// TJPF_BGR).
fn strip_alpha_to_3bpp(src: &[u8], width: usize, height: usize, src_tj: c_int) -> (Vec<u8>, c_int) {
    let bytes: usize = width * height * 3;
    let mut dst: Vec<u8> = Vec::with_capacity(bytes);
    let (b0, b1, b2, dst_tj): (usize, usize, usize, c_int) = match src_tj {
        crate::convert::TJPF_RGBX | crate::convert::TJPF_RGBA => (0, 1, 2, TJPF_RGB),
        crate::convert::TJPF_BGRX | crate::convert::TJPF_BGRA => (0, 1, 2, TJPF_BGR),
        crate::convert::TJPF_XRGB | crate::convert::TJPF_ARGB => (1, 2, 3, TJPF_RGB),
        crate::convert::TJPF_XBGR | crate::convert::TJPF_ABGR => (1, 2, 3, TJPF_BGR),
        _ => (0, 1, 2, src_tj),
    };
    for px in src.chunks_exact(4) {
        dst.push(px[b0]);
        dst.push(px[b1]);
        dst.push(px[b2]);
    }
    (dst, dst_tj)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

unsafe fn cstr_to_path(filename: *const c_char) -> Option<&'static str> {
    if filename.is_null() {
        return None;
    }
    // SAFETY: caller guarantees a NUL-terminated C string. The lifetime
    // is bounded by the caller's storage; we cast to `'static` only to
    // satisfy the borrow checker on the immediate downstream use, which
    // copies the bytes into an owned `Path` before this function
    // returns.
    let cs: &CStr = unsafe { CStr::from_ptr(filename) };
    cs.to_str().ok()
}

fn align_to(v: usize, align: usize) -> usize {
    if align <= 1 {
        v
    } else {
        v.div_ceil(align) * align
    }
}

// ---------------------------------------------------------------------------
// tj3LoadImage8 / 12 / 16
// ---------------------------------------------------------------------------

/// `tj3LoadImage8(handle, filename, width, align, height, pixelFormat)`.
///
/// Reads a BMP or PPM/PGM file into a freshly libc-allocated buffer.
/// On success: returns the buffer, writes width/height/pixelFormat
/// outputs, and the caller releases the buffer via `tj3Free`. On
/// failure: returns NULL, leaving outputs unchanged, with a
/// descriptive error string installed on the handle.
///
/// Pixel-format negotiation:
///
/// * If `*pixel_format == TJPF_UNKNOWN (-1)` (or unset), we use the
///   file's native format and write that back through the
///   `*pixel_format` out-param.
/// * If `*pixel_format` names a specific TJPF code AND it matches the
///   file's native format, we use it directly.
/// * If the requested format differs from the file's native format,
///   we currently return an error — pixel-format conversion is
///   tracked as follow-up work; stock libjpeg-turbo accepts any
///   TJPF the upstream BMP/PPM reader can produce, but only a subset
///   matches Rust's `PixelFormat` losslessly.
///
/// `align` is the minimum row stride alignment in bytes; we honour
/// it by padding each row out to a multiple of `align`. `align == 0`
/// or `1` means dense (no padding).
#[no_mangle]
pub extern "C" fn tj3LoadImage8(
    handle: *mut c_void,
    filename: *const c_char,
    width: *mut c_int,
    align: c_int,
    height: *mut c_int,
    pixel_format: *mut c_int,
) -> *mut u8 {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return std::ptr::null_mut(),
    };
    let path: &str = match unsafe { cstr_to_path(filename) } {
        Some(p) => p,
        None => {
            inst.set_error("tj3LoadImage8: filename is NULL or not UTF-8", TJERR_FATAL);
            return std::ptr::null_mut();
        }
    };
    let bytes: Vec<u8> = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            inst.set_error(
                format!("tj3LoadImage8: cannot read {path}: {e}"),
                TJERR_FATAL,
            );
            return std::ptr::null_mut();
        }
    };
    // Detect BMP magic before delegating to the parser; upstream
    // TurboJPEG returns BMP pixels in TJPF_BGR even though the
    // Rust loader normalises 24-bit BMP to PixelFormat::Rgb. We
    // swap R↔B post-load so the visible default matches stock.
    let is_bmp: bool = bytes.len() >= 2 && bytes[0] == b'B' && bytes[1] == b'M';
    let mut img: libjpeg_turbo_rs::LoadedImage =
        match libjpeg_turbo_rs::load_image_from_bytes(&bytes) {
            Ok(i) => i,
            Err(e) => {
                inst.set_error(
                    format!("tj3LoadImage8: parse failed for {path}: {e}"),
                    TJERR_FATAL,
                );
                return std::ptr::null_mut();
            }
        };
    let mut native_tj: c_int = pixel_format_to_tj(img.pixel_format);
    if native_tj < 0 {
        inst.set_error(
            "tj3LoadImage8: file pixel format has no TJPF mapping",
            TJERR_FATAL,
        );
        return std::ptr::null_mut();
    }
    // Convert RGB→BGR for BMP sources so upstream TurboJPEG
    // semantics hold. PPM/PGM stay in their native format.
    if is_bmp && native_tj == TJPF_RGB {
        rb_swap_3bpp(&mut img.pixels);
        native_tj = TJPF_BGR;
    }

    // Honour caller-requested pixel format when possible. A negative
    // value means "use file's native format". Today we support the
    // identity match plus the RGB↔BGR swap (both directions) for
    // 3-bpp source data; other conversions return an error.
    let mut effective_tj: c_int = native_tj;
    if !pixel_format.is_null() {
        let req: c_int = unsafe { *pixel_format };
        if req >= 0 && req != native_tj {
            if (native_tj == TJPF_RGB && req == TJPF_BGR)
                || (native_tj == TJPF_BGR && req == TJPF_RGB)
            {
                rb_swap_3bpp(&mut img.pixels);
                effective_tj = req;
            } else {
                inst.set_error(
                    format!(
                        "tj3LoadImage8: pixel-format conversion not yet supported (file is TJPF={native_tj}, requested TJPF={req})"
                    ),
                    TJERR_FATAL,
                );
                return std::ptr::null_mut();
            }
        }
    }
    let native_tj: c_int = effective_tj;

    // Bottom-up row order: stock TurboJPEG returns rows top-to-bottom
    // by default. When the caller set TJPARAM_BOTTOMUP / TJFLAG_BOTTOMUP,
    // flip the row order in-place before we copy out. Use the live
    // `inst` borrow rather than re-dereferencing `handle` to avoid
    // aliasing two `&mut TjInstance` to the same allocation.
    if inst.bottom_up_flag() {
        let bpp_for_flip: usize = img.pixel_format.bytes_per_pixel();
        flip_rows_in_place(&mut img.pixels, img.width * bpp_for_flip);
    }

    // Pad rows out to `align` bytes if requested.
    let bpp: usize = img.pixel_format.bytes_per_pixel();
    let row_dense: usize = img.width * bpp;
    let row_stride: usize = align_to(row_dense, align.max(1) as usize);
    let total: usize = row_stride * img.height;
    let buf_ptr: *mut u8 = if row_stride == row_dense {
        // No padding — copy the dense buffer directly.
        libc_from_slice(&img.pixels)
    } else {
        let p: *mut u8 = crate::alloc::libc_malloc(total);
        if !p.is_null() {
            for y in 0..img.height {
                let src_off: usize = y * row_dense;
                let dst_off: usize = y * row_stride;
                // SAFETY: `p` owns `total` bytes; `row_dense ≤ row_stride`
                // ensures the copy stays inside the destination row.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        img.pixels.as_ptr().add(src_off),
                        p.add(dst_off),
                        row_dense,
                    );
                }
            }
        }
        p
    };
    if buf_ptr.is_null() {
        inst.set_error("tj3LoadImage8: out of memory", TJERR_FATAL);
        return std::ptr::null_mut();
    }

    if !width.is_null() {
        unsafe { *width = img.width as c_int };
    }
    if !height.is_null() {
        unsafe { *height = img.height as c_int };
    }
    if !pixel_format.is_null() {
        unsafe { *pixel_format = native_tj };
    }
    buf_ptr
}

/// `tj3LoadImage12(handle, filename, width, align, height, pixelFormat) -> short *`.
#[no_mangle]
pub extern "C" fn tj3LoadImage12(
    handle: *mut c_void,
    _filename: *const c_char,
    _width: *mut c_int,
    _align: c_int,
    _height: *mut c_int,
    _pixel_format: *mut c_int,
) -> *mut i16 {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error(
            "tj3LoadImage12: 12-bit image load not routed through the Rust shim yet",
            TJERR_FATAL,
        );
    }
    std::ptr::null_mut()
}

/// `tj3LoadImage16(handle, filename, width, align, height, pixelFormat) -> unsigned short *`.
#[no_mangle]
pub extern "C" fn tj3LoadImage16(
    handle: *mut c_void,
    _filename: *const c_char,
    _width: *mut c_int,
    _align: c_int,
    _height: *mut c_int,
    _pixel_format: *mut c_int,
) -> *mut u16 {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error(
            "tj3LoadImage16: 16-bit image load not routed through the Rust shim yet",
            TJERR_FATAL,
        );
    }
    std::ptr::null_mut()
}

// ---------------------------------------------------------------------------
// tj3SaveImage8 / 12 / 16
// ---------------------------------------------------------------------------

/// `tj3SaveImage8(handle, filename, buffer, width, pitch, height, pixelFormat)`.
///
/// Writes BMP (`.bmp` extension) or PPM/PGM (`.ppm` / `.pgm` /
/// fallback) using the Rust helpers in
/// `libjpeg_turbo_rs::api::image_io`. Returns 0 on success, `-1`
/// with an error installed on failure.
///
/// `pitch` is bytes per row in the input buffer. `pitch == 0` means
/// dense (`width * bytes_per_pixel`).
#[no_mangle]
pub extern "C" fn tj3SaveImage8(
    handle: *mut c_void,
    filename: *const c_char,
    buffer: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if buffer.is_null() {
        inst.set_error("tj3SaveImage8: buffer is NULL", TJERR_FATAL);
        return -1;
    }
    if width <= 0 || height <= 0 {
        inst.set_error("tj3SaveImage8: width and height must be > 0", TJERR_FATAL);
        return -1;
    }
    let path: &str = match unsafe { cstr_to_path(filename) } {
        Some(p) => p,
        None => {
            inst.set_error("tj3SaveImage8: filename is NULL or not UTF-8", TJERR_FATAL);
            return -1;
        }
    };
    let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
        Some(p) => p,
        None => {
            inst.set_error(
                format!("tj3SaveImage8: unsupported TJPF code {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    let w: usize = width as usize;
    let h: usize = height as usize;
    let bpp: usize = pf.bytes_per_pixel();
    let row_dense: usize = w * bpp;
    let stride: usize = if pitch <= 0 {
        row_dense
    } else {
        pitch as usize
    };
    if stride < row_dense {
        inst.set_error(
            "tj3SaveImage8: pitch is smaller than width * bytes_per_pixel",
            TJERR_FATAL,
        );
        return -1;
    }

    // Repack into a dense buffer if pitch != width*bpp; the Rust
    // saver expects dense rows.
    let mut dense_bytes: Vec<u8> = if stride == row_dense {
        // SAFETY: caller asserts buffer holds at least
        // `stride * h = row_dense * h` valid bytes.
        unsafe { std::slice::from_raw_parts(buffer, row_dense * h).to_vec() }
    } else {
        let mut v: Vec<u8> = Vec::with_capacity(row_dense * h);
        for y in 0..h {
            // SAFETY: caller asserts buffer holds at least `stride * h`
            // bytes; `y * stride + row_dense ≤ stride * h`.
            let row: &[u8] =
                unsafe { std::slice::from_raw_parts(buffer.add(y * stride), row_dense) };
            v.extend_from_slice(row);
        }
        v
    };

    // Bottom-up row order: when the caller set TJPARAM_BOTTOMUP /
    // TJFLAG_BOTTOMUP, the supplied buffer is in bottom-to-top order;
    // flip it before handing to the Rust savers (which expect
    // top-to-bottom). Use the live `inst` borrow rather than
    // re-dereferencing `handle`, which would alias two
    // `&mut TjInstance` to the same allocation (UB under the
    // aliasing rules).
    if inst.bottom_up_flag() {
        flip_rows_in_place(&mut dense_bytes, row_dense);
    }

    // Choose effective format & buffer for downstream save. For BMP
    // outputs, upstream TurboJPEG's tj3SaveImage8 strips alpha /
    // padding from 4-bpp formats so the on-disk BMP is 24-bit. We
    // mirror that: convert RGBA/BGRA/RGBX/BGRX/ARGB/ABGR/XRGB/XBGR
    // down to 3-bpp RGB or BGR before invoking save_bmp. PPM/PGM
    // accept the original format.
    let lower: String = path.to_ascii_lowercase();
    let is_bmp_out: bool = lower.ends_with(".bmp");
    let (effective_bytes, effective_pf): (Vec<u8>, PixelFormat) = if is_bmp_out && bpp == 4 {
        let (stripped, dst_tj) = strip_alpha_to_3bpp(&dense_bytes, w, h, pixel_format);
        let pf3: PixelFormat = pixel_format_from_tj(dst_tj).unwrap_or(PixelFormat::Rgb);
        (stripped, pf3)
    } else {
        (dense_bytes, pf)
    };

    // Dispatch on extension. BMP for `.bmp`; otherwise PPM (matches
    // upstream tj3SaveImage8's behaviour where unrecognised
    // extensions fall through to PPM/PGM).
    let res: libjpeg_turbo_rs::Result<()> = if is_bmp_out {
        libjpeg_turbo_rs::save_bmp(path, &effective_bytes, w, h, effective_pf)
    } else {
        libjpeg_turbo_rs::save_ppm(path, &effective_bytes, w, h, effective_pf)
    };
    match res {
        Ok(()) => 0,
        Err(e) => {
            inst.set_error(
                format!("tj3SaveImage8: write failed for {path}: {e}"),
                TJERR_FATAL,
            );
            -1
        }
    }
}

/// `tj3SaveImage12(handle, filename, buffer, width, pitch, height, pixelFormat)`.
#[no_mangle]
pub extern "C" fn tj3SaveImage12(
    handle: *mut c_void,
    _filename: *const c_char,
    _buffer: *const i16,
    _width: c_int,
    _pitch: c_int,
    _height: c_int,
    _pixel_format: c_int,
) -> c_int {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error(
            "tj3SaveImage12: 12-bit image save not routed through the Rust shim yet",
            TJERR_FATAL,
        );
    }
    -1
}

/// `tj3SaveImage16(handle, filename, buffer, width, pitch, height, pixelFormat)`.
#[no_mangle]
pub extern "C" fn tj3SaveImage16(
    handle: *mut c_void,
    _filename: *const c_char,
    _buffer: *const u16,
    _width: c_int,
    _pitch: c_int,
    _height: c_int,
    _pixel_format: c_int,
) -> c_int {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error(
            "tj3SaveImage16: 16-bit image save not routed through the Rust shim yet",
            TJERR_FATAL,
        );
    }
    -1
}

// Silence "unused" warning on `libc_free` import in some build
// configurations; `libc_free` is a public crate helper used by other
// modules and may not appear in this file's symbol use.
#[allow(dead_code)]
const _IO_FREE_KEEPALIVE: unsafe fn(*mut u8) = libc_free;
