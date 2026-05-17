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
//! `dstBufs[i]` / `dstSizes[i]`. We allocate the outputs through libc
//! so the caller can release them via `tj3Free` or `free`. The custom
//! filter callback is not forwarded: wiring it would require converting
//! all `int16` block coefficients back through our Rust interface, which
//! the Rust `TransformOptions::custom_filter` already does internally
//! but not for arbitrary C function pointers. `custom_filter == NULL`
//! is the common case for jpegtran-style operations and fully supported.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, CropRegion, MarkerCopyMode, TransformOp, TransformOptions,
};

use crate::alloc::{libc_free, libc_from_slice};
use crate::header::TjRegion;
use crate::tj3::{handle_as_mut, TJERR_FATAL};

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
#[no_mangle]
pub extern "C" fn tj3Transform(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    n: c_int,
    dst_bufs: *mut *mut u8,
    dst_sizes: *mut usize,
    transforms: *const TjTransform,
) -> c_int {
    crate::unwind_guard!(-1, {
        let inst = match unsafe { handle_as_mut(handle) } {
            Some(i) => i,
            None => return -1,
        };

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
        let txforms: &[TjTransform] = unsafe { std::slice::from_raw_parts(transforms, n as usize) };

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

            // SAFETY: dst_bufs/dst_sizes arrays validated non-NULL above and
            // documented by the caller as having `n` slots.
            unsafe {
                let slot: *mut *mut u8 = dst_bufs.add(i);
                let size_slot: *mut usize = dst_sizes.add(i);

                let out_ptr: *mut u8 = libc_from_slice(&out);
                if out_ptr.is_null() && !out.is_empty() {
                    inst.set_error(format!("tj3Transform[{i}]: out-of-memory"), TJERR_FATAL);
                    return -1;
                }

                // Free any prior allocation the caller handed us; matches the
                // NOREALLOC-off semantics of libjpeg-turbo.
                let prior: *mut u8 = *slot;
                if !prior.is_null() {
                    libc_free(prior);
                }

                *slot = out_ptr;
                *size_slot = out.len();
            }
        }

        inst.clear_error();
        0
    })
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
#[no_mangle]
pub extern "C" fn tj3TransformBufSize(handle: *mut c_void, transform: *const TjTransform) -> usize {
    crate::unwind_guard!(0, {
        use libjpeg_turbo_rs::tj3::TjParam;

        let inst = match unsafe { handle_as_mut(handle) } {
            Some(i) => i,
            None => return 0,
        };
        if transform.is_null() {
            inst.set_error("tj3TransformBufSize: transform is NULL", TJERR_FATAL);
            return 0;
        }
        // SAFETY: caller-supplied; non-NULL verified above. The struct is
        // POD-like (raw layout matches `tjtransform`), so reading it is safe.
        let xform: &TjTransform = unsafe { &*transform };

        let mut w: c_int = inst.inner.get(TjParam::Width);
        let mut h: c_int = inst.inner.get(TjParam::Height);
        if w <= 0 || h <= 0 {
            inst.set_error(
                "tj3TransformBufSize: dimensions not set; call tj3DecompressHeader first",
                TJERR_FATAL,
            );
            return 0;
        }
        let mut subsamp: c_int = inst.inner.get(TjParam::Subsampling);
        // `TJXOPT_GRAY` forces grayscale output regardless of source subsampling
        // (mirrors `references/libjpeg-turbo/src/turbojpeg.c::getDstSubsamp`).
        if (xform.options & TJXOPT_GRAY) != 0 {
            subsamp = TJSAMP_GRAY;
        }
        // Transpose-class ops swap the H and V chroma factors, so e.g. 4:2:2 →
        // 4:4:0 and 4:1:1 → 4:4:1. Without this swap the post-transform buffer
        // estimate underflows for non-square chroma layouts and `tj3Transform`
        // can overrun. Mirrors upstream `turbojpeg.c::getTransformedSpecs`.
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
        // as "remainder of width post-x". `tjbench` only takes the
        // `IS_CROPPED` path when at least one of x/y/w/h is non-zero, so the
        // simpler "any of r.{w,h} positive" heuristic matches its usage; a
        // stricter caller that wants the remainder-mode behaviour can pass an
        // explicit `r.w` / `r.h`.
        if (xform.options & TJXOPT_CROP) != 0 {
            if xform.r.w > 0 {
                w = xform.r.w;
            }
            if xform.r.h > 0 {
                h = xform.r.h;
            }
        }
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
    })
}
