//! `tj3Compress8` — 8-bit JPEG compression entry point.
//!
//! Signature (from `turbojpeg.h`):
//! ```c
//! int tj3Compress8(tjhandle handle, const unsigned char *srcBuf,
//!                  int width, int pitch, int height, int pixelFormat,
//!                  unsigned char **jpegBuf, size_t *jpegSize);
//! ```
//!
//! Behavior contract:
//! - Returns 0 on success, -1 on failure.
//! - `pitch == 0` means tightly packed rows (`width * bpp`).
//! - On success, `*jpegBuf` is either (a) an input buffer pre-supplied by
//!   the caller (when `TJPARAM_NOREALLOC` is set) or (b) a freshly
//!   libc-allocated buffer that the caller must release with `tj3Free`
//!   / `free`. We always take path (b) for now: `TJPARAM_NOREALLOC` is
//!   accepted but a new buffer is still written so C callers can unify
//!   on `free()`.
//! - `*jpegSize` must be updated to the compressed byte count.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs::PixelFormat;

use crate::alloc::{libc_free, libc_from_slice};
use crate::convert::pixel_format_from_tj;
use crate::tj3::{handle_as_mut, TJERR_FATAL};

/// `tj3Compress8(handle, srcBuf, width, pitch, height, pixelFormat,
///               jpegBuf, jpegSize) -> int`.
#[no_mangle]
pub extern "C" fn tj3Compress8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    // SAFETY: handle validated below; other pointers are validated to be
    // non-NULL before deref.
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    // Argument sanity — every branch stashes an explanatory error.
    if src_buf.is_null() {
        inst.set_error("tj3Compress8: srcBuf is NULL", TJERR_FATAL);
        return -1;
    }
    if jpeg_buf.is_null() || jpeg_size.is_null() {
        inst.set_error(
            "tj3Compress8: jpegBuf / jpegSize out parameter is NULL",
            TJERR_FATAL,
        );
        return -1;
    }
    if width <= 0 || height <= 0 {
        inst.set_error(
            format!("tj3Compress8: width/height must be positive (got {width}x{height})"),
            TJERR_FATAL,
        );
        return -1;
    }
    if pitch < 0 {
        inst.set_error(
            format!("tj3Compress8: pitch must be non-negative (got {pitch})"),
            TJERR_FATAL,
        );
        return -1;
    }

    let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
        Some(p) => p,
        None => {
            inst.set_error(
                format!("tj3Compress8: unsupported TJPF {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    let w: usize = width as usize;
    let h: usize = height as usize;
    let bpp: usize = pf.bytes_per_pixel();
    let effective_pitch: usize = if pitch == 0 { w * bpp } else { pitch as usize };
    if effective_pitch < w * bpp {
        inst.set_error(
            format!(
                "tj3Compress8: pitch {effective_pitch} smaller than width*bpp ({})",
                w * bpp
            ),
            TJERR_FATAL,
        );
        return -1;
    }

    // Reconstruct a dense row-major buffer. The Rust-side encoder expects
    // `width * bpp * height` bytes without per-row padding, so we repack
    // when the caller supplied a non-default pitch.
    //
    // SAFETY: caller guarantees `src_buf` is valid for
    // `effective_pitch * height` bytes laid out row-major with the given
    // pitch.
    let src_len: usize = effective_pitch
        .checked_mul(h)
        .unwrap_or(0)
        .saturating_sub(effective_pitch.saturating_sub(w * bpp));
    let src_slice: &[u8] = unsafe { std::slice::from_raw_parts(src_buf, src_len) };

    let dense: Vec<u8> = if effective_pitch == w * bpp {
        src_slice.to_vec()
    } else {
        let mut packed: Vec<u8> = Vec::with_capacity(w * bpp * h);
        // SAFETY: re-slice the caller's buffer row-by-row using the
        // declared pitch; last row reads exactly `w*bpp` bytes.
        for row in 0..h {
            let row_start: usize = row * effective_pitch;
            // Caller's buffer is valid for (h-1)*pitch + w*bpp bytes.
            let row_slice: &[u8] =
                unsafe { std::slice::from_raw_parts(src_buf.add(row_start), w * bpp) };
            packed.extend_from_slice(row_slice);
        }
        packed
    };

    let jpeg: Vec<u8> = match inst.inner.compress(&dense, w, h, pf) {
        Ok(b) => b,
        Err(e) => {
            inst.set_error(format!("tj3Compress8: {e}"), TJERR_FATAL);
            return -1;
        }
    };

    // Publish to C. Ownership transfers to the caller.
    let out_ptr: *mut u8 = libc_from_slice(&jpeg);
    if out_ptr.is_null() {
        inst.set_error("tj3Compress8: out-of-memory", TJERR_FATAL);
        return -1;
    }

    // SAFETY: jpeg_buf and jpeg_size validated non-NULL above. If the
    // caller had a prior allocation in *jpeg_buf we deliberately leak it:
    // we don't know its allocator. This matches libjpeg-turbo behavior
    // when `TJPARAM_NOREALLOC == 0` — we never reuse the caller's buffer,
    // we always allocate a fresh one. Callers wanting to avoid the leak
    // must pass `*jpeg_buf = NULL` on entry (documented in tj3Compress8).
    unsafe {
        // If the user passed a non-NULL buffer AND NOREALLOC is set,
        // libjpeg-turbo would reuse it in-place. We approximate by freeing
        // the old buffer (best-effort) and writing the new one. This is
        // only safe if the old allocation came from `tj3Alloc`/libc malloc.
        let prior: *mut u8 = *jpeg_buf;
        if !prior.is_null() && inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) == 0 {
            libc_free(prior);
        }
        *jpeg_buf = out_ptr;
        *jpeg_size = jpeg.len();
    }

    inst.clear_error();
    0
}
