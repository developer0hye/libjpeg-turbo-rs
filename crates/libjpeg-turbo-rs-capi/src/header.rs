//! `tj3DecompressHeader`, `tj3SetScalingFactor`, `tj3SetCroppingRegion`.
//!
//! Signatures (from `turbojpeg.h`):
//! ```c
//! int tj3DecompressHeader(tjhandle handle, const unsigned char *jpegBuf,
//!                         size_t jpegSize);
//!
//! typedef struct { int num, denom; } tjscalingfactor;
//! typedef struct { int x, y, w, h; } tjregion;
//!
//! int tj3SetScalingFactor(tjhandle handle, tjscalingfactor factor);
//! int tj3SetCroppingRegion(tjhandle handle, tjregion region);
//! ```
//!
//! `tj3DecompressHeader` parses just enough of the JPEG stream to update
//! the handle's `Width`/`Height`/`Precision`/`ColorSpace`/`Subsampling`
//! parameters. Callers then read them via `tj3Get` to size the output
//! buffer before invoking `tj3Decompress8`.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs::tj3::TjParam;

use crate::tj3::{with_handle, TJERR_FATAL};

/// `tj3DecompressHeader(handle, jpegBuf, jpegSize) -> int`.
#[no_mangle]
pub extern "C" fn tj3DecompressHeader(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
            if jpeg_buf.is_null() || jpeg_size < 2 {
                inst.set_error(
                    "tj3DecompressHeader: NULL jpegBuf or jpegSize < 2",
                    TJERR_FATAL,
                );
                return -1;
            }

            // SAFETY: caller guarantees `jpeg_buf` is valid for `jpeg_size` bytes.
            let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };

            // Header-only path: must report ORIGINAL dimensions (ignoring any
            // `tj3SetScalingFactor`), per the libjpeg-turbo contract that
            // `TJPARAM_JPEGWIDTH`/`TJPARAM_JPEGHEIGHT` reflect the raw JPEG
            // frame size. `decompress_header` applies scaling factor 1:1
            // internally, restoring the caller's scaling factor for subsequent
            // `tj3Decompress*` calls.
            match inst.inner.decompress_header(jpeg) {
                Ok(()) => {
                    inst.clear_error();
                    0
                }
                Err(e) => {
                    inst.set_error(format!("tj3DecompressHeader: {e}"), TJERR_FATAL);
                    -1
                }
            }
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

/// C-layout mirror of `tjscalingfactor` for passing by value from C.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TjScalingFactor {
    pub num: c_int,
    pub denom: c_int,
}

/// C-layout mirror of `tjregion`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TjRegion {
    pub x: c_int,
    pub y: c_int,
    pub w: c_int,
    pub h: c_int,
}

/// `tj3SetScalingFactor(handle, factor) -> int`.
#[no_mangle]
pub extern "C" fn tj3SetScalingFactor(handle: *mut c_void, factor: TjScalingFactor) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
            if factor.num <= 0 || factor.denom <= 0 {
                inst.set_error(
                    format!(
                        "tj3SetScalingFactor: non-positive ratio {}/{}",
                        factor.num, factor.denom
                    ),
                    TJERR_FATAL,
                );
                return -1;
            }

            match inst
                .inner
                .set_scaling_factor(factor.num as u32, factor.denom as u32)
            {
                Ok(()) => {
                    inst.clear_error();
                    0
                }
                Err(e) => {
                    inst.set_error(format!("tj3SetScalingFactor: {e}"), TJERR_FATAL);
                    -1
                }
            }
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

/// `tj3SetCroppingRegion(handle, region) -> int`.
#[no_mangle]
pub extern "C" fn tj3SetCroppingRegion(handle: *mut c_void, region: TjRegion) -> c_int {
    crate::unwind_guard!(-1, {
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
            // The canonical C contract treats {0,0,0,0} as "clear the region".
            if region.x == 0 && region.y == 0 && region.w == 0 && region.h == 0 {
                inst.inner.set_cropping_region(None);
                inst.clear_error();
                return 0;
            }

            if region.x < 0 || region.y < 0 || region.w <= 0 || region.h <= 0 {
                inst.set_error(
                    format!(
                        "tj3SetCroppingRegion: invalid {{x={},y={},w={},h={}}}",
                        region.x, region.y, region.w, region.h
                    ),
                    TJERR_FATAL,
                );
                return -1;
            }

            inst.inner
                .set_cropping_region(Some(libjpeg_turbo_rs::CropRegion {
                    x: region.x as usize,
                    y: region.y as usize,
                    width: region.w as usize,
                    height: region.h as usize,
                }));
            inst.clear_error();
            0
        };

        // SAFETY: as `tj3SetScalingFactor` above.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

// Re-export the `TjParam` import so the module remains self-contained
// even though current implementations don't dispatch on it directly.
#[allow(dead_code)]
const _PARAM_MARKER: TjParam = TjParam::Width;
