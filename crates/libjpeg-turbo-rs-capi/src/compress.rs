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

use libjpeg_turbo_rs::tj3::TjParam;
use libjpeg_turbo_rs::PixelFormat;

use crate::alloc::libc_from_slice;
use crate::convert::pixel_format_from_tj;
use crate::tj3::{with_handle, TJERR_FATAL};

/// `tj3Compress8(handle, srcBuf, width, pitch, height, pixelFormat,
///               jpegBuf, jpegSize) -> int`.
///
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `jpeg_buf`, `jpeg_size` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3Compress8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    crate::unwind_guard!(-1, {
        // The body is a closure so that `with_handle` — not the caller — picks
        // the lifetime of `&mut TjInstance` (P4-137). It is defined outside the
        // `unsafe` block below so its own `unsafe` blocks stay meaningful
        // rather than being nested inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
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

            // Mirror `references/libjpeg-turbo/src/turbojpeg-mp.c::tj3Compress*`
            // (the macro-generated body at lines 109-117 of that file): the
            // entry-point's natural precision (8 here) is the default; only
            // honour an explicit `TJPARAM_PRECISION` when lossless is active
            // and the value falls inside the entry-point's range. Out-of-range
            // values silently fall back to the default rather than erroring —
            // the libjpeg layer is the canonical source of "bad precision"
            // diagnostics, not the TJ ABI.
            let is_lossless: bool = inst.inner.get(TjParam::Lossless) != 0;
            let raw_precision: i32 = inst.inner.get(TjParam::Precision);
            let effective_precision: i32 = if is_lossless && (2..=8).contains(&raw_precision) {
                raw_precision
            } else {
                8
            };

            // For lossless mode with non-default precision, bypass the regular
            // encoder and call the precision-aware lossless path directly.
            let jpeg: Vec<u8> = if is_lossless && effective_precision != 8 {
                use libjpeg_turbo_rs::encode::pipeline::compress_lossless_extended_precision;
                let predictor: u8 = inst.inner.get(TjParam::LosslessPsv) as u8;
                let point_transform: u8 = inst.inner.get(TjParam::LosslessPt) as u8;
                // ITU-T T.81 / Annex H: point transform Pt must be strictly less
                // than the sample precision P (Pt shifts away the lower Pt bits;
                // Pt == P would zero every sample). Mirror upstream
                // `references/libjpeg-turbo/src/jclossls.c::start_pass_lossls` —
                // upstream throws this from libjpeg; we throw from the TJ layer
                // because the precision-aware path here doesn't reach libjpeg.
                if (point_transform as i32) >= effective_precision {
                    inst.set_error(
                    format!(
                        "tj3Compress8: TJPARAM_LOSSLESSPT {point_transform} must be < TJPARAM_PRECISION {effective_precision}"
                    ),
                    TJERR_FATAL,
                );
                    return -1;
                }
                let restart_interval: u16 = {
                    let rb: i32 = inst.inner.get(TjParam::RestartBlocks);
                    let rr: i32 = inst.inner.get(TjParam::RestartRows);
                    if rb > 0 {
                        rb as u16
                    } else if rr > 0 {
                        rr as u16
                    } else {
                        0
                    }
                };
                match compress_lossless_extended_precision(
                    &dense,
                    w,
                    h,
                    pf,
                    predictor,
                    point_transform,
                    restart_interval,
                    effective_precision as u8,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        inst.set_error(format!("tj3Compress8: {e}"), TJERR_FATAL);
                        return -1;
                    }
                }
            } else {
                match inst.inner.compress(&dense, w, h, pf) {
                    Ok(b) => b,
                    Err(e) => {
                        inst.set_error(format!("tj3Compress8: {e}"), TJERR_FATAL);
                        return -1;
                    }
                }
            };

            // SAFETY: jpeg_buf and jpeg_size validated non-NULL above.
            // Two ownership paths per libjpeg-turbo semantics:
            //   (1) NOREALLOC=1 with non-NULL *jpeg_buf: write IN PLACE into the
            //       caller's pre-allocated buffer. The caller retains ownership
            //       and later frees it; we must NOT swap the pointer. The
            //       pre-allocated buffer is at least `tj3JPEGBufSize()` bytes
            //       (the caller's contract).
            //   (2) Else: allocate a fresh libc buffer, transfer ownership; any
            //       prior pointer is treated as stale (leaked — we don't know
            //       its allocator) per the published contract.
            //
            // tjunittest exercises (1) in a tight loop: one `tj3Alloc` at setup,
            // then many `tj3Compress8` calls — if we swap the pointer each call
            // we leak and (worse) the final `tj3Free` releases a different
            // allocation than the one the caller still believes they own.
            let norealloc: bool = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
            let prior: *mut u8 = unsafe { *jpeg_buf };

            if norealloc && !prior.is_null() {
                // Path (1): in-place write into the caller's buffer.
                //
                // `*jpeg_size` is an *input* here: under NOREALLOC it carries
                // the capacity of `prior`, and upstream raises
                // `JERR_BUFFER_SIZE` when the output does not fit
                // (`jdatadst-tj.c:92` — `if (!dest->alloc) ERREXIT(cinfo,
                // JERR_BUFFER_SIZE)`).
                //
                // This used to trust that `prior` was at least
                // `tj3JPEGBufSize(...)` and copy `jpeg.len()` regardless. That
                // is a heap overflow for a caller doing exactly what upstream
                // permits: allocating a smaller buffer, declaring its size, and
                // relying on the library to refuse rather than overrun. Found
                // by the codex review of P4-137 (#476).
                // SAFETY: `jpeg_size` was NULL-checked above.
                let capacity: usize = unsafe { *jpeg_size };
                if jpeg.len() > capacity {
                    inst.set_error(
                        "tj3Compress8: TJPARAM_NOREALLOC is set and the JPEG buffer is too small",
                        TJERR_FATAL,
                    );
                    return -1;
                }
                // SAFETY: caller-supplied buffer, non-aliasing with `jpeg` (which
                // is owned by this function), and `capacity >= jpeg.len()` was
                // just checked rather than assumed.
                unsafe {
                    std::ptr::copy_nonoverlapping(jpeg.as_ptr(), prior, jpeg.len());
                    *jpeg_size = jpeg.len();
                }
            } else {
                // Path (2): allocate + hand off.
                let out_ptr: *mut u8 = libc_from_slice(&jpeg);
                if out_ptr.is_null() {
                    inst.set_error("tj3Compress8: out-of-memory", TJERR_FATAL);
                    return -1;
                }
                unsafe {
                    *jpeg_buf = out_ptr;
                    *jpeg_size = jpeg.len();
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
