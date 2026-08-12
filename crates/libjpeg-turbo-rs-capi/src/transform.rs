//! `tj3Transform` — lossless JPEG transforms (flip, rotate, crop, ...).
//!
//! Signature (from `turbojpeg.h`):
//! ```c
//! typedef struct {
//!     tjregion r;       /* cropping region */
//!     int op;           /* TJXOP_* operation */
//!     int options;      /* OR of TJXOPT_* flags */
//!     void *data;       /* user pointer for customFilter */
//!     int (*customFilter)(short *coeffs, tjregion r, tjregion p,
//!                         int ci, int i, struct tjtransform *t);
//! } tjtransform;
//!
//! int tj3Transform(tjhandle handle, const unsigned char *jpegBuf,
//!                  size_t jpegSize, int n,
//!                  unsigned char **dstBufs, size_t *dstSizes,
//!                  const tjtransform *transforms);
//! ```
//!
//! Each entry of `transforms[0..n]` produces one output JPEG written to
//! `dstBufs[i]` / `dstSizes[i]`. **Which of the two ownership paths is taken
//! depends on `TJPARAM_NOREALLOC`** (P4-145): with the flag set, the output is
//! written *in place* into the caller's buffer and the pointer comes back
//! unchanged — so it may be a stack array or a `Vec`, and must **not** be
//! passed to `free`. With the flag unset, the output is allocated through libc
//! and the previous pointee freed, so the caller releases the result via
//! `tj3Free` or `free`.
//!
//! This paragraph used to say the outputs are always libc-allocated and always
//! releasable with `free`. Following that under the flag frees memory this
//! library never allocated, which is the invalid free P4-145 fixed. The custom
//! filter callback is not forwarded: wiring it would require converting
//! all `int16` block coefficients back through our Rust interface, which
//! the Rust `TransformOptions::custom_filter` already does internally
//! but not for arbitrary C function pointers. `custom_filter == NULL`
//! is the common case for jpegtran-style operations and fully supported.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, CropRegion, MarkerCopyMode, TransformOp, TransformOptions,
};

use crate::alloc::{deliver_compressed_output, OutputDelivery};
use crate::header::TjRegion;
use crate::tj3::{with_handle, TJERR_FATAL};

// --- TJXOP_* ---
pub const TJXOP_NONE: c_int = 0;
pub const TJXOP_HFLIP: c_int = 1;
pub const TJXOP_VFLIP: c_int = 2;
pub const TJXOP_TRANSPOSE: c_int = 3;
pub const TJXOP_TRANSVERSE: c_int = 4;
pub const TJXOP_ROT90: c_int = 5;
pub const TJXOP_ROT180: c_int = 6;
pub const TJXOP_ROT270: c_int = 7;

// --- TJSAMP_* (subset used for transpose-aware buf-size estimation) ---
const TJSAMP_422: c_int = 1;
const TJSAMP_GRAY: c_int = 3;
const TJSAMP_440: c_int = 4;
const TJSAMP_411: c_int = 5;
const TJSAMP_441: c_int = 6;
const TJSAMP_410: c_int = 7;
const TJSAMP_24: c_int = 8;

// --- TJXOPT_* (bit flags) ---
pub const TJXOPT_PERFECT: c_int = 1;
pub const TJXOPT_TRIM: c_int = 2;
pub const TJXOPT_CROP: c_int = 4;
pub const TJXOPT_GRAY: c_int = 8;
pub const TJXOPT_NOOUTPUT: c_int = 16;
pub const TJXOPT_PROGRESSIVE: c_int = 32;
pub const TJXOPT_COPYNONE: c_int = 64;
pub const TJXOPT_ARITHMETIC: c_int = 128;
pub const TJXOPT_OPTIMIZE: c_int = 256;

/// C-layout `tjtransform`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TjTransform {
    pub r: TjRegion,
    pub op: c_int,
    pub options: c_int,
    pub data: *mut c_void,
    pub custom_filter: Option<
        unsafe extern "C" fn(
            coeffs: *mut i16,
            array_region: TjRegion,
            plane_region: TjRegion,
            component_index: c_int,
            transform_index: c_int,
            transform: *mut TjTransform,
        ) -> c_int,
    >,
}

fn op_from_c(op: c_int) -> Option<TransformOp> {
    Some(match op {
        TJXOP_NONE => TransformOp::None,
        TJXOP_HFLIP => TransformOp::HFlip,
        TJXOP_VFLIP => TransformOp::VFlip,
        TJXOP_TRANSPOSE => TransformOp::Transpose,
        TJXOP_TRANSVERSE => TransformOp::Transverse,
        TJXOP_ROT90 => TransformOp::Rot90,
        TJXOP_ROT180 => TransformOp::Rot180,
        TJXOP_ROT270 => TransformOp::Rot270,
        _ => return None,
    })
}

/// `tj3Transform(handle, jpegBuf, jpegSize, n, dstBufs, dstSizes, transforms)
///   -> int`.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_bufs`, `dst_sizes`, `transforms` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
///
/// Each non-null `*dst_bufs[i]` is **freed by this function only when
/// `TJPARAM_NOREALLOC` is unset**, in which case every one must have come from
/// `tj3Alloc`/`malloc` — see
/// [Ownership transfer](crate#pointer-contract). Note it is the *destination*
/// slots that are freed; `jpeg_buf` is the const source and is never freed.
/// Each `dst_bufs[i]` honours `TJPARAM_NOREALLOC` independently: with the flag
/// set the transform writes in place into that slot, leaves the pointer
/// unchanged, and treats `dst_sizes[i]` as the slot's capacity — too small is
/// an error rather than a resize (P4-145).
#[no_mangle]
pub unsafe extern "C" fn tj3Transform(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    n: c_int,
    dst_bufs: *mut *mut u8,
    dst_sizes: *mut usize,
    transforms: *const TjTransform,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
            if jpeg_buf.is_null() || jpeg_size < 2 {
                inst.set_error("tj3Transform: NULL jpegBuf or jpegSize < 2", TJERR_FATAL);
                return -1;
            }
            if n <= 0 {
                inst.set_error(
                    format!("tj3Transform: n must be positive (got {n})"),
                    TJERR_FATAL,
                );
                return -1;
            }
            if dst_bufs.is_null() || dst_sizes.is_null() || transforms.is_null() {
                inst.set_error(
                    "tj3Transform: NULL dstBufs / dstSizes / transforms",
                    TJERR_FATAL,
                );
                return -1;
            }

            // SAFETY: caller guarantees the three arrays have at least `n` slots
            // and `jpeg_buf` is valid for `jpeg_size` bytes.
            let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
            let txforms: &[TjTransform] =
                unsafe { std::slice::from_raw_parts(transforms, n as usize) };

            // Process each transform independently. libjpeg-turbo does this in a
            // loop too; there's no shared decode state across transforms.
            for (i, t) in txforms.iter().enumerate() {
                let op: TransformOp = match op_from_c(t.op) {
                    Some(o) => o,
                    None => {
                        inst.set_error(
                            format!("tj3Transform[{i}]: unknown TJXOP {}", t.op),
                            TJERR_FATAL,
                        );
                        return -1;
                    }
                };

                if t.custom_filter.is_some() {
                    inst.set_error(
                        format!("tj3Transform[{i}]: customFilter callback is not supported yet"),
                        TJERR_FATAL,
                    );
                    return -1;
                }

                let mut opts: TransformOptions = TransformOptions {
                    op,
                    perfect: (t.options & TJXOPT_PERFECT) != 0,
                    trim: (t.options & TJXOPT_TRIM) != 0,
                    crop: None,
                    grayscale: (t.options & TJXOPT_GRAY) != 0,
                    no_output: (t.options & TJXOPT_NOOUTPUT) != 0,
                    progressive: (t.options & TJXOPT_PROGRESSIVE) != 0,
                    arithmetic: (t.options & TJXOPT_ARITHMETIC) != 0,
                    optimize: (t.options & TJXOPT_OPTIMIZE) != 0,
                    restart_interval: 0,
                    restart_in_rows: false,
                    copy_markers: if (t.options & TJXOPT_COPYNONE) != 0 {
                        MarkerCopyMode::None
                    } else {
                        MarkerCopyMode::All
                    },
                    custom_filter: None,
                };

                if (t.options & TJXOPT_CROP) != 0 {
                    if t.r.x < 0 || t.r.y < 0 || t.r.w <= 0 || t.r.h <= 0 {
                        inst.set_error(
                            format!(
                                "tj3Transform[{i}]: invalid crop region {{x={},y={},w={},h={}}}",
                                t.r.x, t.r.y, t.r.w, t.r.h
                            ),
                            TJERR_FATAL,
                        );
                        return -1;
                    }
                    opts.crop = Some(CropRegion {
                        x: t.r.x as usize,
                        y: t.r.y as usize,
                        width: t.r.w as usize,
                        height: t.r.h as usize,
                    });
                }

                let out: Vec<u8> = match transform_jpeg_with_options(jpeg, &opts) {
                    Ok(v) => v,
                    Err(e) => {
                        inst.set_error(format!("tj3Transform[{i}]: {e}"), TJERR_FATAL);
                        return -1;
                    }
                };

                // Upstream skips destination setup entirely for this option —
                // `if (!(t[i].options & TJXOPT_NOOUTPUT)) jpeg_mem_dest_tj(...)`
                // (`turbojpeg.c:3007`) — so the slots stay exactly as the caller
                // left them and a NULL destination is fine. Delivering here
                // instead would demand a buffer for output that was never
                // produced, and would zero a non-NULL slot's size.
                if (t.options & TJXOPT_NOOUTPUT) != 0 {
                    continue;
                }

                // P4-145: `tj3Transform`'s reusable slots are `dst_bufs[i]`, not
                // `jpeg_buf` — its `jpeg_buf` is the const *source* and is never
                // freed. Each slot honours `TJPARAM_NOREALLOC` per-image, which
                // is what a caller pre-sizing an array of output buffers relies
                // on; before this they were freed unconditionally, with the
                // wrong allocator whenever they were not `malloc`-owned.
                let norealloc: bool =
                    inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
                // SAFETY: dst_bufs/dst_sizes arrays validated non-NULL above and
                // documented by the caller as having `n` slots, so `add(i)` is
                // in bounds for `i < n`.
                unsafe {
                    let slot: *mut *mut u8 = dst_bufs.add(i);
                    let size_slot: *mut usize = dst_sizes.add(i);

                    match deliver_compressed_output(&out, slot, size_slot, norealloc) {
                        OutputDelivery::Delivered => {}
                        OutputDelivery::BufferTooSmall { needed, capacity } => {
                            inst.set_error(
                                format!(
                                    "tj3Transform[{i}]: TJPARAM_NOREALLOC is set and the \
                                     destination buffer is too small ({needed} bytes needed, \
                                     {capacity} available)"
                                ),
                                TJERR_FATAL,
                            );
                            return -1;
                        }
                        OutputDelivery::NoBufferSupplied => {
                            inst.set_error(
                                format!(
                                    "tj3Transform[{i}]: TJPARAM_NOREALLOC is set but no \
                                     destination buffer was supplied"
                                ),
                                TJERR_FATAL,
                            );
                            return -1;
                        }
                        OutputDelivery::OutOfMemory => {
                            inst.set_error(
                                format!("tj3Transform[{i}]: out-of-memory"),
                                TJERR_FATAL,
                            );
                            return -1;
                        }
                    }
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

/// Geometry of the image a transform produces, from geometry alone.
///
/// Mirrors upstream `turbojpeg.c::getTransformedSpecs`. Kept separate from
/// [`tj3TransformBufSize`] because the two callers need different things from
/// it: that entry point adds the handle's stored ICC length to the bound it
/// returns, while the legacy `tjTransform` wrapper must **not** — a caller that
/// sized its destination with `tjTransformBufSize()` gets a buffer derived from
/// geometry only, and adding metadata to the capacity handed to `tj3Transform`
/// overruns it. Measured at a 32x32 source with a 128 KiB ICC profile: an
/// 8192-byte destination against a 132320-byte capacity (P4-151).
///
/// Taking `(w, h, subsamp)` as parameters rather than reading them from a
/// handle is the other half of that separation: the legacy wrapper derives them
/// from a header probe, because parsing into the handle would overwrite
/// compression state the caller set — subsampling, colour space, density, ICC.
pub(crate) fn transformed_specs(
    mut w: c_int,
    mut h: c_int,
    mut subsamp: c_int,
    xform: &TjTransform,
) -> (c_int, c_int, c_int) {
    // `TJXOPT_GRAY` forces grayscale output regardless of source subsampling
    // (mirrors `references/libjpeg-turbo/src/turbojpeg.c::getDstSubsamp`).
    if (xform.options & TJXOPT_GRAY) != 0 {
        subsamp = TJSAMP_GRAY;
    }
    // Transpose-class ops swap the H and V chroma factors, so e.g. 4:2:2 →
    // 4:4:0 and 4:1:1 → 4:4:1. Without this swap the post-transform buffer
    // estimate underflows for non-square chroma layouts and `tj3Transform`
    // can overrun.
    if matches!(
        xform.op,
        TJXOP_TRANSPOSE | TJXOP_TRANSVERSE | TJXOP_ROT90 | TJXOP_ROT270
    ) {
        std::mem::swap(&mut w, &mut h);
        subsamp = match subsamp {
            TJSAMP_422 => TJSAMP_440,
            TJSAMP_440 => TJSAMP_422,
            TJSAMP_411 => TJSAMP_441,
            TJSAMP_441 => TJSAMP_411,
            TJSAMP_410 => TJSAMP_24,
            TJSAMP_24 => TJSAMP_410,
            other => other,
        };
    }
    // Crop detection: upstream gates on `TJXOPT_CROP` and treats `r.w == 0`
    // as "remainder of width post-x". `tjbench` only takes the `IS_CROPPED`
    // path when at least one of x/y/w/h is non-zero, so the simpler "any of
    // r.{w,h} positive" heuristic matches its usage; a stricter caller that
    // wants the remainder-mode behaviour can pass an explicit `r.w` / `r.h`.
    if (xform.options & TJXOPT_CROP) != 0 {
        if xform.r.w > 0 {
            w = xform.r.w;
        }
        if xform.r.h > 0 {
            h = xform.r.h;
        }
    }
    (w, h, subsamp)
}

/// `tj3TransformBufSize(handle, *transform) -> size_t`.
///
/// Returns an upper bound on the bytes needed to hold the JPEG produced
/// by applying `transform` to the source image whose header was last
/// parsed via `tj3DecompressHeader`. Mirrors
/// `references/libjpeg-turbo/src/turbojpeg.h:2358`. Returns `0` on
/// error (NULL handle, NULL transform, or unread header), and stashes
/// a descriptive message reachable via `tj3GetErrorStr`.
///
/// The bound is computed by:
///   1. Pulling cached source dimensions and subsampling out of the
///      handle's TJPARAM_* state (populated by `tj3DecompressHeader`).
///   2. Swapping width/height for the transpose-class ops
///      (`TJXOP_TRANSPOSE`, `TJXOP_TRANSVERSE`, `TJXOP_ROT90`,
///      `TJXOP_ROT270`).
///   3. Honoring the crop region in `transform.r` when set.
///   4. Delegating to `tj3JPEGBufSize` on the resulting (W,H,S).
///
/// # Safety
///
/// C ABI entry point. `handle`, `transform` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3TransformBufSize(
    handle: *mut c_void,
    transform: *const TjTransform,
) -> usize {
    crate::unwind_guard!(0, {
        use libjpeg_turbo_rs::tj3::TjParam;

        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> usize {
            if transform.is_null() {
                inst.set_error("tj3TransformBufSize: transform is NULL", TJERR_FATAL);
                return 0;
            }
            // SAFETY: caller-supplied; non-NULL verified above. The struct is
            // POD-like (raw layout matches `tjtransform`), so reading it is safe.
            let xform: &TjTransform = unsafe { &*transform };

            let w: c_int = inst.inner.get(TjParam::Width);
            let h: c_int = inst.inner.get(TjParam::Height);
            if w <= 0 || h <= 0 {
                inst.set_error(
                    "tj3TransformBufSize: dimensions not set; call tj3DecompressHeader first",
                    TJERR_FATAL,
                );
                return 0;
            }
            let subsamp: c_int = inst.inner.get(TjParam::Subsampling);
            let (w, h, subsamp) = transformed_specs(w, h, subsamp, xform);
            inst.clear_error();
            let base: usize = crate::bufsize::tj3JPEGBufSize(w, h, subsamp);
            // Mirrors upstream `turbojpeg.c::tj3TransformBufSize`: add the stored ICC
            // byte count to the worst-case buffer bound. A caller that set an ICC via
            // `tj3SetICCProfile` must allocate enough space for the ICC APP2 chunks;
            // without this addition the bound would underflow and a downstream
            // tjbench-style consumer could silently undersize its destination buffer.
            // Saturate on overflow: never return less than the bare `tj3JPEGBufSize`.
            let icc_bytes: usize = inst.inner.icc_profile().map_or(0, |b| b.len());
            base.checked_add(icc_bytes).unwrap_or(base)
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(0)
    })
}
