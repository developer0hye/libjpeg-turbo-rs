//! Shared conversion helpers between TurboJPEG's integer constants and
//! our Rust-side enums. Keeping these in one place avoids drift as more
//! TJ-family entry points are added.
//!
//! Numeric values here MUST match `turbojpeg.h` exactly; downstream C
//! callers depend on them being ABI-stable.

use std::ffi::c_int;

use libjpeg_turbo_rs::PixelFormat;

/// TJPF_* constants exactly as defined in `turbojpeg.h`.
pub const TJPF_RGB: c_int = 0;
pub const TJPF_BGR: c_int = 1;
pub const TJPF_RGBX: c_int = 2;
pub const TJPF_BGRX: c_int = 3;
pub const TJPF_XBGR: c_int = 4;
pub const TJPF_XRGB: c_int = 5;
pub const TJPF_GRAY: c_int = 6;
pub const TJPF_RGBA: c_int = 7;
pub const TJPF_BGRA: c_int = 8;
pub const TJPF_ABGR: c_int = 9;
pub const TJPF_ARGB: c_int = 10;
pub const TJPF_CMYK: c_int = 11;

/// Convert TJPF_* integer to the Rust `PixelFormat`. Returns `None` for
/// unsupported codes (including `TJPF_UNKNOWN = -1`).
pub fn pixel_format_from_tj(fmt: c_int) -> Option<PixelFormat> {
    Some(match fmt {
        TJPF_RGB => PixelFormat::Rgb,
        TJPF_BGR => PixelFormat::Bgr,
        TJPF_RGBX => PixelFormat::Rgbx,
        TJPF_BGRX => PixelFormat::Bgrx,
        TJPF_XBGR => PixelFormat::Xbgr,
        TJPF_XRGB => PixelFormat::Xrgb,
        TJPF_GRAY => PixelFormat::Grayscale,
        TJPF_RGBA => PixelFormat::Rgba,
        TJPF_BGRA => PixelFormat::Bgra,
        TJPF_ABGR => PixelFormat::Abgr,
        TJPF_ARGB => PixelFormat::Argb,
        TJPF_CMYK => PixelFormat::Cmyk,
        _ => return None,
    })
}

/// Bytes-per-pixel for a TJPF_* code, or `None` for unsupported codes.
/// Callers use this to reconstruct dense row buffers when `pitch == 0`.
pub fn tj_bytes_per_pixel(fmt: c_int) -> Option<usize> {
    pixel_format_from_tj(fmt).map(|pf| pf.bytes_per_pixel())
}
