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

use crate::tj3::{handle_as_mut, TJERR_FATAL};

/// `tj3DecompressHeader(handle, jpegBuf, jpegSize) -> int`.
#[no_mangle]
pub extern "C" fn tj3DecompressHeader(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    if jpeg_buf.is_null() || jpeg_size < 2 {
        inst.set_error(
            "tj3DecompressHeader: NULL jpegBuf or jpegSize < 2",
            TJERR_FATAL,
        );
        return -1;
    }

    // SAFETY: caller guarantees `jpeg_buf` is valid for `jpeg_size` bytes.
    let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };

    // We perform a full decompress then discard the pixel data. A future
    // optimization can parse just SOF markers, but the current Rust API
    // does not expose a standalone header parser; `decompress()` already
    // updates all the parameters `tj3DecompressHeader` is contracted to
    // populate, so the correctness win outweighs the extra decode cost
    // for small images — which is exactly the header-first path.
    //
    // We call into the Rust handle's `decompress`, which internally
    // updates the handle's Width/Height/Precision/ColorSpace/Subsampling
    // and ICC state. Matches the C semantics that after tj3DecompressHeader
    // all header-derived TJPARAM_* values are queryable via tj3Get.
    match inst.inner.decompress(jpeg) {
        Ok(_img) => {
            inst.clear_error();
            0
        }
        Err(e) => {
            inst.set_error(format!("tj3DecompressHeader: {e}"), TJERR_FATAL);
            -1
        }
    }
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
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

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
}

/// `tj3SetCroppingRegion(handle, region) -> int`.
#[no_mangle]
pub extern "C" fn tj3SetCroppingRegion(handle: *mut c_void, region: TjRegion) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

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
}

// Re-export the `TjParam` import so the module remains self-contained
// even though current implementations don't dispatch on it directly.
#[allow(dead_code)]
const _PARAM_MARKER: TjParam = TjParam::Width;
