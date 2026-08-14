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
//! - On success, `*jpegBuf` is either (a) the buffer the caller pre-supplied,
//!   written **in place**, when `TJPARAM_NOREALLOC` is set, or (b) a freshly
//!   libc-allocated buffer that the caller must release with `tj3Free` /
//!   `free`, the previous pointee having been freed. Which one is decided by
//!   the flag, as upstream decides it (P4-145).
//!
//!   This text used to say path (b) was always taken "so C callers can unify
//!   on `free()`". Unifying was never safe: under the flag the caller may pass
//!   a stack array or a `Vec`, and following that advice would free memory the
//!   library never allocated.
//! - `*jpegSize` must be updated to the compressed byte count.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs::common::layout::{checked_span, ImageLayout};
use libjpeg_turbo_rs::tj3::TjParam;
use libjpeg_turbo_rs::PixelFormat;

use crate::alloc::{deliver_compressed_output, OutputDelivery};
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

            // P4-155 (#539): after argument validation, before any encoding
            // state — upstream's order (`turbojpeg-mp.c:95-98`).
            if !crate::tj3::require_lossy_compress_params(inst, "tj3Compress8") {
                return -1;
            }

            let w: usize = width as usize;
            let h: usize = height as usize;
            let bpp: usize = pf.bytes_per_pixel();
            // Checked: `row_bytes` and the span built from it below size a
            // `slice::from_raw_parts` (P4-139 criterion 3). On a 32-bit target
            // `w * bpp` overflows well inside the dimensions TurboJPEG accepts.
            let row_bytes: usize = match checked_span(&[w, bpp], "tj3Compress8 source row") {
                Ok(bytes) => bytes,
                Err(_) => {
                    inst.set_error(
                        "tj3Compress8: width * bytes_per_pixel overflows",
                        TJERR_FATAL,
                    );
                    return -1;
                }
            };
            let effective_pitch: usize = if pitch == 0 {
                row_bytes
            } else {
                pitch as usize
            };
            if effective_pitch < row_bytes {
                inst.set_error(
                    format!(
                        "tj3Compress8: pitch {effective_pitch} smaller than width*bpp ({row_bytes})"
                    ),
                    TJERR_FATAL,
                );
                return -1;
            }

            // Reconstruct a dense row-major buffer. The Rust-side encoder expects
            // `width * bpp * height` bytes without per-row padding, so we repack
            // when the caller supplied a non-default pitch.
            //
            // The source span is `pitch * (h - 1) + row_bytes`: every row but
            // the last is a full pitch, and the last needs only its own pixels
            // — a caller is not required to allocate padding past the final
            // row. That is exactly `ImageLayout::strided`'s rule, so the
            // formula lives there now rather than being spelled out here
            // (P4-139); this site was the shape the type was modelled on.
            //
            // P4-137 had already replaced the original `checked_mul(h)
            // .unwrap_or(0).saturating_sub(...)` here — worse than it looked,
            // since an overflow did not error but produced a *zero-length*
            // source slice, and the encode proceeded on no input. What this
            // change does is move that corrected formula into `ImageLayout`,
            // not rediscover it.
            let src: ImageLayout =
                match ImageLayout::strided(w, h, bpp, effective_pitch, "tj3Compress8 source") {
                    Ok(layout) => layout,
                    Err(_) => {
                        inst.set_error(
                            "tj3Compress8: pitch * height overflows the source buffer size",
                            TJERR_FATAL,
                        );
                        return -1;
                    }
                };
            // SAFETY: caller guarantees `src_buf` is valid for the pitched
            // extent above, which `ImageLayout` proved fits `isize::MAX` —
            // `from_raw_parts`' own precondition.
            let src_slice: &[u8] =
                unsafe { std::slice::from_raw_parts(src_buf, src.total_bytes()) };

            let dense: Vec<u8> = if src.stride() == src.row_bytes() {
                src_slice.to_vec()
            } else {
                let mut packed: Vec<u8> = Vec::with_capacity(src.packed_bytes());
                // SAFETY: re-slice the caller's buffer row-by-row using the
                // declared pitch; last row reads exactly `row_bytes` bytes, so
                // every read stays inside the extent asserted above.
                for row in 0..h {
                    let row_slice: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            src_buf.add(src.row_offset(row)),
                            src.row_bytes(),
                        )
                    };
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

            // P4-145: one shared implementation for all six compressing entry
            // points. This function had the only correct one; the other five
            // freed the caller's slot unconditionally. Keeping six copies is
            // how the `compress_*` family P4-40 was filed for came about.
            //
            // tjunittest exercises the in-place path in a tight loop: one
            // `tj3Alloc` at setup, then many `tj3Compress8` calls — swapping the
            // pointer each call would leak, and the final `tj3Free` would
            // release a different allocation than the one the caller still
            // believes it owns.
            let norealloc: bool = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
            // SAFETY: both out-pointers were validated non-NULL above; the
            // caller's contract covers buffer validity and non-aliasing with
            // `jpeg`, which this function owns.
            match unsafe { deliver_compressed_output(&jpeg, jpeg_buf, jpeg_size, norealloc) } {
                OutputDelivery::Delivered => {}
                OutputDelivery::BufferTooSmall { needed, capacity } => {
                    inst.set_error(
                        format!(
                            "tj3Compress8: TJPARAM_NOREALLOC is set and the JPEG buffer is too \
                             small ({needed} bytes needed, {capacity} available)"
                        ),
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::NoBufferSupplied => {
                    inst.set_error(
                        "tj3Compress8: TJPARAM_NOREALLOC is set but no output buffer was supplied",
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::OutOfMemory => {
                    inst.set_error("tj3Compress8: out-of-memory", TJERR_FATAL);
                    return -1;
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
