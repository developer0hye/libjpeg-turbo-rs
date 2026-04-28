//! TJ3 buffer-sizing and scaling-factor helpers.
//!
//! These are pure computations (no handle required) that complete the TJ3
//! C API surface needed by `tjunittest`:
//!
//! ```c
//! size_t tj3JPEGBufSize(int width, int height, int jpegSubsamp);
//! size_t tj3YUVBufSize(int width, int align, int height, int subsamp);
//! size_t tj3YUVPlaneSize(int componentID, int width, int stride,
//!                        int height, int subsamp);
//! int    tj3YUVPlaneWidth(int componentID, int width, int subsamp);
//! int    tj3YUVPlaneHeight(int componentID, int height, int subsamp);
//! tjscalingfactor *tj3GetScalingFactors(int *numScalingFactors);
//! ```
//!
//! The functions return 0 / -1 on invalid input and populate the
//! process-global error string (matching C reference behavior where a
//! no-handle call can still report the last error via
//! `tj3GetErrorStr(NULL)`).

use std::ffi::{c_int, c_void};
use std::sync::Mutex;

use libjpeg_turbo_rs::tj3::TjHandle;

use crate::header::TjScalingFactor;

/// TJ subsampling index → (MCU width in pixels, MCU height in pixels).
/// Values match the `tjMCUWidth`/`tjMCUHeight` tables in
/// `references/libjpeg-turbo/src/turbojpeg.h`.
const TJ_MCU_WIDTH: [i32; 9] = [8, 16, 16, 8, 8, 32, 8, 32, 16];
const TJ_MCU_HEIGHT: [i32; 9] = [8, 8, 16, 8, 16, 8, 32, 16, 32];

/// Accept any subsampling index in `0..TJ_NUMSAMP` (9) and report
/// whether it is the special gray-only mode.
fn mcu_for_tj(tjsamp: c_int) -> Option<(usize, usize)> {
    if !(0..TJ_MCU_WIDTH.len() as c_int).contains(&tjsamp) {
        return None;
    }
    Some((
        TJ_MCU_WIDTH[tjsamp as usize] as usize,
        TJ_MCU_HEIGHT[tjsamp as usize] as usize,
    ))
}

/// Is the subsampling `TJSAMP_GRAY`? Gray images only have a Y plane, so
/// the chrominance plane sizes are 0.
fn is_gray(tjsamp: c_int) -> bool {
    tjsamp == 3
}

/// `TJ_NUMSAMP` — the upper bound of valid TJSAMP_* indices.
const TJ_NUMSAMP: c_int = 9;

fn pad_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    // Align up to `alignment` (which need not be a power of two for
    // the 32×16 / 16×32 MCUs).
    value.div_ceil(alignment) * alignment
}

// ---------------------------------------------------------------------------
// Process-global last-error state for no-handle calls.
//
// The C API documents that `tj3GetErrorStr(NULL)` returns the message from
// the most recent no-handle TJ3 call. We back that with a single Mutex so
// the state survives across parameter-probing calls like those in
// `bufSizeTest()`.
// ---------------------------------------------------------------------------
static LAST_NO_HANDLE_ERROR: Mutex<Option<std::ffi::CString>> = Mutex::new(None);

pub(crate) fn set_no_handle_error(msg: &str) {
    let sanitized: String = msg.replace('\0', "?");
    if let Ok(cs) = std::ffi::CString::new(sanitized) {
        if let Ok(mut guard) = LAST_NO_HANDLE_ERROR.lock() {
            *guard = Some(cs);
        }
    }
}

/// Look up the global no-handle error string. Returns NULL if none has
/// been recorded yet (callers must then fall back to the default "No
/// error" message).
pub(crate) fn no_handle_error_ptr() -> *const std::ffi::c_char {
    let guard = match LAST_NO_HANDLE_ERROR.lock() {
        Ok(g) => g,
        Err(_) => return std::ptr::null(),
    };
    match guard.as_ref() {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// tj3JPEGBufSize
// ---------------------------------------------------------------------------

/// `tj3JPEGBufSize(width, height, jpegSubsamp) -> size_t`.
///
/// Implements libjpeg-turbo's canonical formula:
///
/// ```text
/// chromasf = subsamp == GRAY ? 0 : 4 * 64 / (mcuw * mcuh)
/// retval   = PAD(width, mcuw) * PAD(height, mcuh) * (2 + chromasf) + 2048
/// ```
///
/// Overflow is checked against `usize::MAX`; on a 64-bit host the
/// overflow gate only trips for truly enormous images (26755×26755×6 at
/// 4:4:4 fits, so the test's overflow check passes through the
/// sufficient-size branch).
#[no_mangle]
pub extern "C" fn tj3JPEGBufSize(width: c_int, height: c_int, jpeg_subsamp: c_int) -> usize {
    if width < 1 || height < 1 || !(-1..TJ_NUMSAMP).contains(&jpeg_subsamp) {
        set_no_handle_error("tj3JPEGBufSize: invalid argument");
        return 0;
    }

    // TJSAMP_UNKNOWN (-1) is treated as TJSAMP_444 for sizing, per the
    // turbojpeg.h contract: "a buffer large enough for no-subsampling is
    // also large enough for anything else".
    let effective_subsamp: c_int = if jpeg_subsamp == -1 { 0 } else { jpeg_subsamp };
    let Some((mcuw, mcuh)): Option<(usize, usize)> = mcu_for_tj(effective_subsamp) else {
        set_no_handle_error("tj3JPEGBufSize: invalid subsampling");
        return 0;
    };

    // GRAY has no chroma; every other subsampling counts 4 chroma samples per
    // `mcuw * mcuh` MCU (matching `4 * 64 / (mcuw * mcuh)` where samples are
    // 8×8 blocks).
    let chromasf: usize = if is_gray(effective_subsamp) {
        0
    } else {
        (4 * 64) / (mcuw * mcuh)
    };

    let padded_w: usize = pad_up(width as usize, mcuw);
    let padded_h: usize = pad_up(height as usize, mcuh);

    padded_w
        .checked_mul(padded_h)
        .and_then(|v| v.checked_mul(2 + chromasf))
        .and_then(|v| v.checked_add(2048))
        .unwrap_or_else(|| {
            set_no_handle_error("tj3JPEGBufSize: image is too large");
            0
        })
}

// ---------------------------------------------------------------------------
// tj3YUVBufSize
// ---------------------------------------------------------------------------

/// `tj3YUVBufSize(width, align, height, subsamp) -> size_t`.
///
/// `align` must be a strictly positive power of two. When `align > 1`
/// each plane row stride is rounded up to a multiple of `align` per the
/// libjpeg-turbo spec.
#[no_mangle]
pub extern "C" fn tj3YUVBufSize(
    width: c_int,
    align: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    if width < 1 || height < 1 {
        set_no_handle_error("tj3YUVBufSize: width/height must be > 0");
        return 0;
    }
    if align < 1 || (align as u32).count_ones() != 1 {
        set_no_handle_error("tj3YUVBufSize: align must be a power of 2");
        return 0;
    }
    if !(0..TJ_NUMSAMP).contains(&subsamp) {
        set_no_handle_error("tj3YUVBufSize: invalid subsampling");
        return 0;
    }

    let n_planes: usize = if is_gray(subsamp) { 1 } else { 3 };
    let mut total: usize = 0;
    for c in 0..n_planes {
        let pw: usize = plane_width(c as c_int, width, subsamp);
        let ph: usize = plane_height(c as c_int, height, subsamp);
        if pw == 0 || ph == 0 {
            set_no_handle_error("tj3YUVBufSize: zero plane dimension");
            return 0;
        }
        let stride: usize = pad_up(pw, align as usize);
        let plane: Option<usize> = stride.checked_mul(ph);
        let plane: usize = match plane {
            Some(v) => v,
            None => {
                set_no_handle_error("tj3YUVBufSize: image is too large");
                return 0;
            }
        };
        total = match total.checked_add(plane) {
            Some(v) => v,
            None => {
                set_no_handle_error("tj3YUVBufSize: image is too large");
                return 0;
            }
        };
    }
    total
}

/// Internal plane-width helper (shared by tj3YUVBufSize / tj3YUVPlaneSize /
/// tj3YUVPlaneWidth to keep the C formula in one place).
fn plane_width(component: c_int, width: c_int, subsamp: c_int) -> usize {
    if width < 1 {
        return 0;
    }
    let Some((mcuw, _)): Option<(usize, usize)> = mcu_for_tj(subsamp) else {
        return 0;
    };
    let ph: usize = mcuw / 8; // luma blocks along width
    let padded: usize = pad_up(width as usize, mcuw);
    if component == 0 {
        padded
    } else if is_gray(subsamp) {
        0
    } else {
        // Chroma horizontal subsampling divides the luma width by ph.
        padded / ph
    }
}

fn plane_height(component: c_int, height: c_int, subsamp: c_int) -> usize {
    if height < 1 {
        return 0;
    }
    let Some((_, mcuh)): Option<(usize, usize)> = mcu_for_tj(subsamp) else {
        return 0;
    };
    let pv: usize = mcuh / 8; // luma blocks along height
    let padded: usize = pad_up(height as usize, mcuh);
    if component == 0 {
        padded
    } else if is_gray(subsamp) {
        0
    } else {
        padded / pv
    }
}

// ---------------------------------------------------------------------------
// tj3YUVPlaneSize
// ---------------------------------------------------------------------------

/// `tj3YUVPlaneSize(componentID, width, stride, height, subsamp) -> size_t`.
#[no_mangle]
pub extern "C" fn tj3YUVPlaneSize(
    component_id: c_int,
    width: c_int,
    stride: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    if !(0..=2).contains(&component_id) {
        set_no_handle_error("tj3YUVPlaneSize: componentID out of range");
        return 0;
    }
    if width < 1 || height < 1 {
        set_no_handle_error("tj3YUVPlaneSize: width/height must be > 0");
        return 0;
    }
    if !(0..TJ_NUMSAMP).contains(&subsamp) {
        set_no_handle_error("tj3YUVPlaneSize: invalid subsampling");
        return 0;
    }

    // Gray: only the Y plane is valid; Cb/Cr return 0 (no error).
    if is_gray(subsamp) && component_id != 0 {
        return 0;
    }

    let pw: usize = plane_width(component_id, width, subsamp);
    let ph: usize = plane_height(component_id, height, subsamp);
    if pw == 0 || ph == 0 {
        set_no_handle_error("tj3YUVPlaneSize: zero plane dimension");
        return 0;
    }

    // Use `stride = pw` when the caller passes 0.
    let effective_stride: usize = if stride == 0 {
        pw
    } else if stride < 0 {
        set_no_handle_error("tj3YUVPlaneSize: stride must be >= 0");
        return 0;
    } else {
        stride as usize
    };
    if effective_stride < pw {
        set_no_handle_error("tj3YUVPlaneSize: stride smaller than plane width");
        return 0;
    }

    effective_stride
        .checked_mul(ph - 1)
        .and_then(|v| v.checked_add(pw))
        .unwrap_or_else(|| {
            set_no_handle_error("tj3YUVPlaneSize: image is too large");
            0
        })
}

// ---------------------------------------------------------------------------
// tj3YUVPlaneWidth / tj3YUVPlaneHeight
// ---------------------------------------------------------------------------

/// `tj3YUVPlaneWidth(componentID, width, subsamp) -> int`.
/// Returns 0 on invalid input, matching C behavior documented in
/// `turbojpeg.h`. `tjunittest`'s `overflowTest` exercises `width = INT_MAX`,
/// which pads up past `INT_MAX` and must therefore produce an overflow
/// error.
#[no_mangle]
pub extern "C" fn tj3YUVPlaneWidth(component_id: c_int, width: c_int, subsamp: c_int) -> c_int {
    if !(0..=2).contains(&component_id) {
        set_no_handle_error("tj3YUVPlaneWidth: componentID out of range");
        return 0;
    }
    if width < 1 {
        set_no_handle_error("tj3YUVPlaneWidth: width must be > 0");
        return 0;
    }
    if !(0..TJ_NUMSAMP).contains(&subsamp) {
        set_no_handle_error("tj3YUVPlaneWidth: invalid subsampling");
        return 0;
    }
    if is_gray(subsamp) && component_id != 0 {
        return 0;
    }
    let pw: usize = plane_width(component_id, width, subsamp);
    if pw == 0 || pw > c_int::MAX as usize {
        set_no_handle_error("tj3YUVPlaneWidth: plane width overflows int");
        return 0;
    }
    pw as c_int
}

/// `tj3YUVPlaneHeight(componentID, height, subsamp) -> int`.
#[no_mangle]
pub extern "C" fn tj3YUVPlaneHeight(component_id: c_int, height: c_int, subsamp: c_int) -> c_int {
    if !(0..=2).contains(&component_id) {
        set_no_handle_error("tj3YUVPlaneHeight: componentID out of range");
        return 0;
    }
    if height < 1 {
        set_no_handle_error("tj3YUVPlaneHeight: height must be > 0");
        return 0;
    }
    if !(0..TJ_NUMSAMP).contains(&subsamp) {
        set_no_handle_error("tj3YUVPlaneHeight: invalid subsampling");
        return 0;
    }
    if is_gray(subsamp) && component_id != 0 {
        return 0;
    }
    let ph: usize = plane_height(component_id, height, subsamp);
    if ph == 0 || ph > c_int::MAX as usize {
        set_no_handle_error("tj3YUVPlaneHeight: plane height overflows int");
        return 0;
    }
    ph as c_int
}

// ---------------------------------------------------------------------------
// tj3GetScalingFactors
// ---------------------------------------------------------------------------

/// Static, per-process table of scaling factors — we hand a raw pointer to
/// C callers with the documented lifetime "valid for the duration of the
/// process". Using a `OnceLock`-wrapped `Vec` keeps us on safe Rust while
/// preserving the C contract.
fn scaling_factor_table() -> &'static [TjScalingFactor] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<TjScalingFactor>> = OnceLock::new();
    TABLE.get_or_init(|| {
        TjHandle::scaling_factors()
            .into_iter()
            .map(|(num, denom)| TjScalingFactor {
                num: num as c_int,
                denom: denom as c_int,
            })
            .collect()
    })
}

/// `tj3GetScalingFactors(numScalingFactors) -> tjscalingfactor *`.
///
/// Returns a pointer to a statically-owned array of `tjscalingfactor`s.
/// The caller MUST NOT free the pointer. `numScalingFactors` is written
/// with the count.
#[no_mangle]
pub extern "C" fn tj3GetScalingFactors(num_scaling_factors: *mut c_int) -> *mut TjScalingFactor {
    let table: &[TjScalingFactor] = scaling_factor_table();
    if !num_scaling_factors.is_null() {
        // SAFETY: caller-provided; NULL was rejected above.
        unsafe {
            *num_scaling_factors = table.len() as c_int;
        }
    }
    // Casting away const is sound because the table never mutates; the C
    // ABI uses a mutable pointer for historical reasons only.
    table.as_ptr() as *mut TjScalingFactor
}

// Silence unused c_void import (used transitively by submodules when
// glob-imported in lib.rs).
#[allow(dead_code)]
const _VOID_MARKER: *const c_void = std::ptr::null();
