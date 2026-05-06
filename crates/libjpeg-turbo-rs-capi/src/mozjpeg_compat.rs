//! mozjpeg parameter-API stubs.
//!
//! `mozjpeg` is the most widely deployed libjpeg-turbo fork (used by
//! Homebrew's libvips, several Linux distros, and Pillow's
//! `pillow-mozjpeg` build). On top of the libjpeg-turbo ABI it adds a
//! parameter-tuning surface (`jpeg_c_*_param_supported` /
//! `jpeg_c_set_*_param` / `jpeg_c_get_*_param`) for trellis quantization,
//! scan optimization, overshoot deringing, and friends.
//!
//! Consumers that link directly against mozjpeg's `libjpeg.62.dylib`
//! carry **undefined references** to these symbols in their dyld load
//! commands. Replacing libjpeg via `LD_PRELOAD` /
//! `DYLD_INSERT_LIBRARIES` then fails at load time — even when the
//! consumer never calls into the mozjpeg path — because dyld cannot
//! resolve the undefined symbols.
//!
//! This module exposes weak no-op stubs:
//! - All `*_param_supported` probes return `FALSE` (0). Consumers
//!   following the mozjpeg pattern probe support before configuring,
//!   so this is the documented "fall back to standard libjpeg" path.
//! - All setters are no-ops (we do not implement mozjpeg-specific
//!   tuning, but a consumer that ignores the `_supported` probe and
//!   sets anyway must not crash).
//! - All getters return zero/false (the value the setter would have
//!   stored, given the no-op semantics).
//!
//! The result: a consumer linked against mozjpeg can dyld-resolve
//! against our cdylib, probe for mozjpeg support, see "not supported",
//! and silently fall through to the standard libjpeg encode path that
//! we *do* implement. This is exactly the "drop-in for the most common
//! libjpeg fork" guarantee P2-10's libvips harness exists to enforce.
//!
//! Reference: <https://github.com/mozilla/mozjpeg/blob/master/jpeglib.h>
//! (search for `jpeg_c_bool_param_supported`).

use std::ffi::{c_float, c_int, c_void};

/// libjpeg's `boolean` is `int` on every platform we ship — `JPEG_LIB_VERSION` 8 layout.
type Boolean = c_int;

const JPEG_FALSE: Boolean = 0;

/// `boolean jpeg_c_bool_param_supported(j_compress_ptr cinfo, J_BOOLEAN_PARAM param)`
///
/// Returns FALSE for every mozjpeg-specific boolean parameter (we do
/// not implement them; consumers must fall back to the standard path).
#[no_mangle]
pub extern "C" fn jpeg_c_bool_param_supported(_cinfo: *mut c_void, _param: c_int) -> Boolean {
    JPEG_FALSE
}

/// `void jpeg_c_set_bool_param(j_compress_ptr cinfo, J_BOOLEAN_PARAM param, boolean value)`
///
/// No-op. A well-behaved consumer probes `*_supported` first; this stub
/// is only here so a consumer that ignores the probe (or a debugging
/// path that always sets) does not crash.
#[no_mangle]
pub extern "C" fn jpeg_c_set_bool_param(_cinfo: *mut c_void, _param: c_int, _value: Boolean) {}

/// `boolean jpeg_c_get_bool_param(j_compress_ptr cinfo, J_BOOLEAN_PARAM param)`
#[no_mangle]
pub extern "C" fn jpeg_c_get_bool_param(_cinfo: *mut c_void, _param: c_int) -> Boolean {
    JPEG_FALSE
}

/// `boolean jpeg_c_int_param_supported(j_compress_ptr cinfo, J_INT_PARAM param)`
#[no_mangle]
pub extern "C" fn jpeg_c_int_param_supported(_cinfo: *mut c_void, _param: c_int) -> Boolean {
    JPEG_FALSE
}

/// `void jpeg_c_set_int_param(j_compress_ptr cinfo, J_INT_PARAM param, int value)`
#[no_mangle]
pub extern "C" fn jpeg_c_set_int_param(_cinfo: *mut c_void, _param: c_int, _value: c_int) {}

/// `int jpeg_c_get_int_param(j_compress_ptr cinfo, J_INT_PARAM param)`
#[no_mangle]
pub extern "C" fn jpeg_c_get_int_param(_cinfo: *mut c_void, _param: c_int) -> c_int {
    0
}

/// `boolean jpeg_c_float_param_supported(j_compress_ptr cinfo, J_FLOAT_PARAM param)`
#[no_mangle]
pub extern "C" fn jpeg_c_float_param_supported(_cinfo: *mut c_void, _param: c_int) -> Boolean {
    JPEG_FALSE
}

/// `void jpeg_c_set_float_param(j_compress_ptr cinfo, J_FLOAT_PARAM param, float value)`
#[no_mangle]
pub extern "C" fn jpeg_c_set_float_param(_cinfo: *mut c_void, _param: c_int, _value: c_float) {}

/// `float jpeg_c_get_float_param(j_compress_ptr cinfo, J_FLOAT_PARAM param)`
#[no_mangle]
pub extern "C" fn jpeg_c_get_float_param(_cinfo: *mut c_void, _param: c_int) -> c_float {
    0.0
}
