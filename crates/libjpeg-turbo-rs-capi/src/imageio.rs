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
use crate::convert::{pixel_format_from_tj, pixel_format_to_tj};
use crate::tj3::{handle_as_mut, TJERR_FATAL};

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
                &format!("tj3LoadImage8: cannot read {path}: {e}"),
                TJERR_FATAL,
            );
            return std::ptr::null_mut();
        }
    };
    let img: libjpeg_turbo_rs::LoadedImage = match libjpeg_turbo_rs::load_image_from_bytes(&bytes) {
        Ok(i) => i,
        Err(e) => {
            inst.set_error(
                &format!("tj3LoadImage8: parse failed for {path}: {e}"),
                TJERR_FATAL,
            );
            return std::ptr::null_mut();
        }
    };
    let native_tj: c_int = pixel_format_to_tj(img.pixel_format);
    if native_tj < 0 {
        inst.set_error(
            "tj3LoadImage8: file pixel format has no TJPF mapping",
            TJERR_FATAL,
        );
        return std::ptr::null_mut();
    }

    // Honour caller-requested pixel format when possible. A negative
    // value means "use file's native format".
    if !pixel_format.is_null() {
        let req: c_int = unsafe { *pixel_format };
        if req >= 0 && req != native_tj {
            // Future work: pixel-format conversion when the request
            // disagrees with the file's native format.
            inst.set_error(
                &format!(
                    "tj3LoadImage8: pixel-format conversion not yet supported (file is TJPF={native_tj}, requested TJPF={req})"
                ),
                TJERR_FATAL,
            );
            return std::ptr::null_mut();
        }
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
                &format!("tj3SaveImage8: unsupported TJPF code {pixel_format}"),
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
    let dense_bytes: Vec<u8> = if stride == row_dense {
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

    // Dispatch on extension. BMP for `.bmp`; otherwise PPM (matches
    // upstream tj3SaveImage8's behaviour where unrecognised
    // extensions fall through to PPM/PGM).
    let lower: String = path.to_ascii_lowercase();
    let res: libjpeg_turbo_rs::Result<()> = if lower.ends_with(".bmp") {
        libjpeg_turbo_rs::save_bmp(path, &dense_bytes, w, h, pf)
    } else {
        libjpeg_turbo_rs::save_ppm(path, &dense_bytes, w, h, pf)
    };
    match res {
        Ok(()) => 0,
        Err(e) => {
            inst.set_error(
                &format!("tj3SaveImage8: write failed for {path}: {e}"),
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
