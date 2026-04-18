//! TJ3 `LoadImage*` / `SaveImage*` shims.
//!
//! The default `tjunittest` run does not exercise these (they gate on
//! `-bmp`), but the link step must resolve every TJ3 symbol that the test
//! references, so we expose a minimal surface that:
//!
//! - For `tj3LoadImage*`: returns NULL and records an error — callers in
//!   the default run never exercise this path.
//! - For `tj3SaveImage*`: returns -1 with an error recorded.
//!
//! When the main Rust library exposes a fully-featured BMP/PPM pipeline we
//! can wire it up here; that is out of scope for the FFI link test.

use std::ffi::{c_char, c_int, c_void};

use crate::tj3::{handle_as_mut, TJERR_FATAL};

// ---------------------------------------------------------------------------
// tj3LoadImage8 / 12 / 16
// ---------------------------------------------------------------------------

/// `tj3LoadImage8(handle, filename, width, align, height, pixelFormat)`.
/// Stub: records an error and returns NULL. `tjunittest` only invokes
/// this with `-bmp`, which is not part of the default link test.
#[no_mangle]
pub extern "C" fn tj3LoadImage8(
    handle: *mut c_void,
    _filename: *const c_char,
    _width: *mut c_int,
    _align: c_int,
    _height: *mut c_int,
    _pixel_format: *mut c_int,
) -> *mut u8 {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error(
            "tj3LoadImage8: BMP/PPM load not routed through the Rust shim yet",
            TJERR_FATAL,
        );
    }
    std::ptr::null_mut()
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
#[no_mangle]
pub extern "C" fn tj3SaveImage8(
    handle: *mut c_void,
    _filename: *const c_char,
    _buffer: *const u8,
    _width: c_int,
    _pitch: c_int,
    _height: c_int,
    _pixel_format: c_int,
) -> c_int {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error(
            "tj3SaveImage8: BMP/PPM save not routed through the Rust shim yet",
            TJERR_FATAL,
        );
    }
    -1
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
