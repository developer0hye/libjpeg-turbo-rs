//! Legacy TurboJPEG 1.x / 2.x aliases.
//!
//! These are thin wrappers that forward to the canonical TJ3 surface
//! implemented elsewhere in this crate. The goal is binary
//! compatibility with existing C clients (Pillow 10.x, ImageMagick
//! 7.x, stock djpeg/cjpeg shims) that still link against the pre-3.0
//! symbol set.
//!
//! Entry points covered:
//! - `tjInitCompress`, `tjInitDecompress`, `tjInitTransform`
//! - `tjDestroy`
//! - `tjCompress2`, `tjDecompress2`
//! - `tjDecompressHeader3`
//! - `tjTransform`
//! - `tjEncodeYUV3`, `tjDecodeYUV` — forward to `tj3EncodeYUV8` /
//!   `tj3DecodeYUV8` after setting `TJPARAM_SUBSAMP` from `subsamp`
//!   and propagating legacy `flags` (`TJFLAG_BOTTOMUP`,
//!   `TJFLAG_FASTUPSAMPLE` (decode only), `TJFLAG_FASTDCT`,
//!   `TJFLAG_PROGRESSIVE` (encode only)) onto their `TJPARAM_*`
//!   counterparts via `process_legacy_*_flags`, mirroring upstream
//!   `turbojpeg.c::processFlags`. The 4th arg of `tjEncodeYUV3` is
//!   `pitch` (input row stride; `0` = tight `width * bpp`), per
//!   the upstream ABI — it is **not** YUV alignment. The dedicated
//!   `align` argument controls YUV plane row alignment.
//! - `tjBufSize`, `tjBufSizeYUV2`, `tjPlaneSizeYUV`, `tjPlaneWidth`,
//!   `tjPlaneHeight`
//! - `tjLoadImage`, `tjSaveImage` — handle-less (per upstream
//!   `turbojpeg.h` legacy 2.x ABI). Allocate a temporary `tjhandle`,
//!   set `TJPARAM_BOTTOMUP` from `flags & TJFLAG_BOTTOMUP`, delegate
//!   to `tj3LoadImage8` / `tj3SaveImage8`, and copy the temp
//!   handle's last error into the global no-handle slot before
//!   destroying it (so `tjGetErrorStr2(NULL)` works).
//! - `tjGetErrorStr2`

use std::ffi::{c_char, c_int, c_void};

use libjpeg_turbo_rs::{calc_jpeg_dimensions, Subsampling};

use crate::compress::tj3Compress8;
use crate::decompress::tj3Decompress8;
use crate::header::{tj3DecompressHeader, TjRegion};
use crate::tj3::{tj3Destroy, tj3GetErrorStr, tj3Init, tj3Set, with_handle};
use crate::transform::{tj3Transform, TjTransform};

// --- TJINIT values (matching turbojpeg.h `enum TJINIT`) ---
const TJINIT_COMPRESS: c_int = 0;
const TJINIT_DECOMPRESS: c_int = 1;
const TJINIT_TRANSFORM: c_int = 2;

// --- TJPARAM identifiers we drive from the legacy surface ---
/// `turbojpeg.h:2793`. Upstream's `processFlags` maps this to the instance's
/// `noRealloc` for every operation, not only compression (`turbojpeg.c:552`).
const TJFLAG_NOREALLOC: c_int = 1024;
/// The TJ3 parameter the legacy flag maps to.
const TJPARAM_NOREALLOC: c_int = 2;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;

// ---------------------------------------------------------------------------
// Init / Destroy
// ---------------------------------------------------------------------------

/// `tjInitCompress()` — legacy compress-only initializer.
#[no_mangle]
pub extern "C" fn tjInitCompress() -> *mut c_void {
    crate::unwind_guard!(std::ptr::null_mut(), { tj3Init(TJINIT_COMPRESS) })
}

/// `tjInitDecompress()` — legacy decompress-only initializer.
#[no_mangle]
pub extern "C" fn tjInitDecompress() -> *mut c_void {
    crate::unwind_guard!(std::ptr::null_mut(), { tj3Init(TJINIT_DECOMPRESS) })
}

/// `tjInitTransform()` — legacy transform initializer.
///
/// In TurboJPEG 3, `TJINIT_TRANSFORM` already grants both compress and
/// decompress capabilities (see `tj3InitVersion` in `turbojpeg.c`), so
/// we just forward the single enum value.
#[no_mangle]
pub extern "C" fn tjInitTransform() -> *mut c_void {
    crate::unwind_guard!(std::ptr::null_mut(), { tj3Init(TJINIT_TRANSFORM) })
}

/// `tjDestroy(handle)` — identical to `tj3Destroy`.
///
/// # Safety
///
/// Forwards `handle` to [`tj3Destroy`] unchanged, so it carries exactly that
/// function's obligation: NULL or a live handle from `tjInit*`/`tj3Init`,
/// never one already destroyed, and no concurrent call using the same handle.
#[no_mangle]
pub unsafe extern "C" fn tjDestroy(handle: *mut c_void) -> c_int {
    crate::unwind_guard!(-1, {
        // SAFETY: the caller's obligation, restated on this function and
        // discharged unchanged — `handle` is theirs, not one we constructed.
        unsafe { tj3Destroy(handle) };
        0
    })
}

// ---------------------------------------------------------------------------
// Compress / Decompress
// ---------------------------------------------------------------------------

/// `tjCompress2(handle, srcBuf, width, pitch, height, pixelFormat,
///              jpegBuf, jpegSize, jpegSubsamp, jpegQual, flags)`.
///
/// Matches the 2.x signature: `jpegSize` is a `unsigned long *` — we
/// accept `*mut usize` (64-bit on modern targets). Of `flags`, only
/// `TJFLAG_NOREALLOC` is honoured — it maps to `TJPARAM_NOREALLOC`, which
/// decides whether `*jpeg_buf` is written in place or replaced and freed
/// (P4-145). The rest are ignored; TJ3 uses explicit parameters instead.
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
pub unsafe extern "C" fn tjCompress2(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
    jpeg_subsamp: c_int,
    jpeg_qual: c_int,
    flags: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // `TJFLAG_NOREALLOC` must reach the TJ3 parameter, because that is what
        // `tj3Compress8` reads. Discarding it silently made a legacy caller's
        // pre-allocated buffer eligible for `free()` — harmless while
        // `tj3Compress8` merely leaked the prior pointer, and an invalid free
        // for caller-owned storage once P4-145 made that path release it.
        // Validate *before* touching instance state. Upstream rejects a NULL
        // `jpegSize` and out-of-range quality/subsampling with "Invalid
        // argument" before `processFlags` runs (`turbojpeg.c:1274-1280`), and
        // the ordering is observable: setting `TJPARAM_NOREALLOC` and then
        // failing would leave the handle's ownership behaviour changed by a
        // call that returned -1, so the *next* call could free caller-owned
        // storage.
        if jpeg_size.is_null() {
            // Report it *on the handle*. A bare `-1` leaves
            // `tjGetErrorStr2(handle)` saying "No error", where upstream says
            // `tjCompress2(): Invalid argument`; writing only the process-global
            // slot is just as wrong, since a caller with a handle reads the
            // handle's message.
            // SAFETY: `with_handle` NULL-checks the handle itself.
            let _ = unsafe {
                with_handle(handle, |inst: &mut crate::tj3::TjInstance| -> c_int {
                    inst.set_error("tjCompress2: Invalid argument", crate::tj3::TJERR_FATAL);
                    -1
                })
            };
            return -1;
        }
        // Subsampling and quality are set via TJ3 parameters before the
        // actual compress call; both reject out-of-range values.
        if unsafe { tj3Set(handle, TJPARAM_QUALITY, jpeg_qual) } != 0 {
            return -1;
        }
        if unsafe { tj3Set(handle, TJPARAM_SUBSAMP, jpeg_subsamp) } != 0 {
            return -1;
        }
        // `TJFLAG_NOREALLOC` must reach the TJ3 parameter, because that is what
        // `tj3Compress8` reads. Discarding it silently made a legacy caller's
        // pre-allocated buffer eligible for `free()` — harmless while
        // `tj3Compress8` merely leaked the prior pointer, and an invalid free
        // for caller-owned storage once P4-145 made that path release it.
        let norealloc: bool = (flags & TJFLAG_NOREALLOC) != 0;
        if unsafe { tj3Set(handle, TJPARAM_NOREALLOC, norealloc as c_int) } != 0 {
            return -1;
        }
        // The two APIs disagree about what the size slot *means*, and the
        // difference only shows under NOREALLOC. In TJ3 it is an input
        // capacity; in the legacy API it is an **output only** — a caller that
        // sized its buffer with `tjBufSize()` is entitled to leave `*jpegSize`
        // at zero. Forwarding the slot directly turned a valid legacy call
        // into "buffer too small".
        //
        // Upstream resolves it the same way: `size = *jpegSize;` then, under
        // NOREALLOC, `size = tj3JPEGBufSize(width, height, subsamp)`
        // (`turbojpeg.c:1282-1284`) — the worst case the caller was told to
        // allocate.
        // SAFETY: non-NULL, checked above.
        let mut size: usize = unsafe { *jpeg_size };
        if norealloc {
            size = crate::bufsize::tj3JPEGBufSize(width, height, jpeg_subsamp);
        }
        let rc: c_int = unsafe {
            tj3Compress8(
                handle,
                src_buf,
                width,
                pitch,
                height,
                pixel_format,
                jpeg_buf,
                &mut size,
            )
        };
        // Unconditional, as upstream's `*jpegSize = (unsigned long)size;` is:
        // the legacy slot is an output.
        // SAFETY: non-NULL (checked above) and writable per the contract.
        unsafe { *jpeg_size = size };
        rc
    })
}

/// `tjDecompress2(handle, jpegBuf, jpegSize, dstBuf, width, pitch,
///                height, pixelFormat, flags)`.
///
/// The `width`/`height` parameters are legacy artifacts; TJ3 reads them
/// from the header. We honor them as an upper-bound sanity check but do
/// not override the JPEG's real dimensions.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjDecompress2(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u8,
    _width: c_int,
    pitch: c_int,
    _height: c_int,
    pixel_format: c_int,
    _flags: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        unsafe { tj3Decompress8(handle, jpeg_buf, jpeg_size, dst_buf, pitch, pixel_format) }
    })
}

/// `tjDecompressHeader3(handle, jpegBuf, jpegSize, width, height,
///                      jpegSubsamp, jpegColorspace)`.
///
/// TJ3 reads these values via `tj3Get`; the legacy entry point
/// populates caller-provided out-pointers.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `width`, `height`, `jpeg_subsamp`, `jpeg_colorspace` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjDecompressHeader3(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    width: *mut c_int,
    height: *mut c_int,
    jpeg_subsamp: *mut c_int,
    jpeg_colorspace: *mut c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        let rc: c_int = unsafe { tj3DecompressHeader(handle, jpeg_buf, jpeg_size) };
        if rc != 0 {
            return -1;
        }

        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
            // SAFETY: out-pointers are optional per the C contract — skip if NULL.
            unsafe {
                use libjpeg_turbo_rs::tj3::TjParam;
                if !width.is_null() {
                    *width = inst.inner.get(TjParam::Width);
                }
                if !height.is_null() {
                    *height = inst.inner.get(TjParam::Height);
                }
                if !jpeg_subsamp.is_null() {
                    *jpeg_subsamp = inst.inner.get(TjParam::Subsampling);
                }
                if !jpeg_colorspace.is_null() {
                    *jpeg_colorspace = inst.inner.get(TjParam::ColorSpace);
                }
            }
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// `tjTransform(handle, jpegBuf, jpegSize, n, dstBufs, dstSizes,
///              transforms, flags)`.
///
/// Of `flags`, only `TJFLAG_NOREALLOC` is honoured — it maps to
/// `TJPARAM_NOREALLOC`, which decides whether each `dst_bufs[i]` is written in
/// place or replaced and freed (P4-145). The rest are ignored; TJ3 drives
/// options through `TJPARAM_*` on the handle.
///
/// `dst_sizes` are **outputs**, as in upstream. A caller that sized its
/// destinations with `tjTransformBufSize()` may leave them at zero: under
/// `TJFLAG_NOREALLOC` each slot is filled from the transformed image's
/// geometry before the call and overwritten with the produced size afterwards
/// (P4-151), matching `turbojpeg.c:3118-3132`. Without the flag they are
/// forwarded unchanged, since the reallocating path derives its own capacity.
///
/// The substituted capacity comes from geometry alone — never from metadata —
/// so it cannot exceed the buffer a `tjTransformBufSize()`-sized allocation
/// describes. Otherwise identical to `tj3Transform`.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_bufs`, `dst_sizes`, `transforms` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjTransform(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    n: c_int,
    dst_bufs: *mut *mut u8,
    dst_sizes: *mut usize,
    transforms: *const TjTransform,
    flags: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Validate before mutating instance state, as in `tjCompress2`.
        if n <= 0 || dst_bufs.is_null() || dst_sizes.is_null() || transforms.is_null() {
            // On the handle, as above.
            // SAFETY: `with_handle` NULL-checks the handle itself.
            let _ = unsafe {
                with_handle(handle, |inst: &mut crate::tj3::TjInstance| -> c_int {
                    inst.set_error("tjTransform: Invalid argument", crate::tj3::TJERR_FATAL);
                    -1
                })
            };
            return -1;
        }

        // As in `tjCompress2`: the destination slots are freed only when the
        // parameter is unset, so the legacy flag has to reach it.
        let norealloc: bool = (flags & TJFLAG_NOREALLOC) != 0;
        if unsafe { tj3Set(handle, TJPARAM_NOREALLOC, norealloc as c_int) } != 0 {
            return -1;
        }

        // The legacy `dstSizes` are *outputs*, so a caller that sized its
        // buffers with `tjTransformBufSize()` may leave them at zero — and TJ3
        // reads that slot as a capacity. Upstream bridges the gap by filling a
        // temporary array with each transformed image's worst case
        // (`turbojpeg.c:3118-3132`) and copying the real sizes back afterwards.
        //
        // P4-151. Two earlier attempts were rejected in review, and both
        // constraints they exposed are load-bearing here:
        //
        // 1. **The capacity comes from geometry alone.** Upstream uses bare
        //    `tj3JPEGBufSize` on the transformed specs. This port's
        //    `tj3TransformBufSize` also adds the extracted ICC length, so using
        //    it would hand `tj3Transform` a capacity larger than the buffer the
        //    caller sized with `tjTransformBufSize()` — measured at a 32x32
        //    source with a 128 KiB profile as an 8192-byte destination against
        //    a 132320-byte capacity. `transformed_specs` + `tj3JPEGBufSize` is
        //    the geometry-only path.
        //
        // 2. **Deriving the geometry must not touch the handle.** A
        //    `tj3DecompressHeader` here would overwrite compression state the
        //    caller set — subsampling, colour space, density, ICC — so an S420
        //    handle transforming an S444 source would come back reporting S444
        //    and silently compress differently afterwards. `probe` parses the
        //    source header into its own decoder and leaves the instance alone.
        let sizes: Option<Vec<usize>> = if norealloc {
            // `tj3Transform` validates the source itself, but not before this
            // runs — and `from_raw_parts` requires a non-null pointer even for
            // a zero-length slice, so reaching it first turns a documented -1
            // into a non-unwinding abort in debug and UB in release.
            if jpeg_buf.is_null() || jpeg_size == 0 {
                // SAFETY: `with_handle` NULL-checks the handle itself.
                let _ = unsafe {
                    crate::tj3::with_handle(handle, |inst: &mut crate::tj3::TjInstance| {
                        inst.set_error("tjTransform: Invalid argument", crate::tj3::TJERR_FATAL);
                    })
                };
                return -1;
            }
            // SAFETY: non-null and non-empty per the check above; the caller's
            // contract covers validity for `jpeg_size` bytes.
            let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
            let info: libjpeg_turbo_rs::JpegInfo = match libjpeg_turbo_rs::probe(jpeg) {
                Ok(info) => info,
                Err(e) => {
                    // SAFETY: `with_handle` NULL-checks the handle itself.
                    let _ = unsafe {
                        crate::tj3::with_handle(handle, |inst: &mut crate::tj3::TjInstance| {
                            inst.set_error(format!("tjTransform: {e}"), crate::tj3::TJERR_FATAL);
                        })
                    };
                    return -1;
                }
            };
            // A grayscale source must size as `TJSAMP_GRAY`, not as whatever
            // `to_tjsamp` makes of `Subsampling::Unknown`. `probe` reports
            // `Unknown` for single-component images — there are no chroma
            // planes to describe — and mapping that to 4:4:4 would hand
            // `tj3Transform` a capacity *larger* than the caller's buffer,
            // which is sized `tjBufSize(w, h, TJSAMP_GRAY)`. Output landing
            // between the two bounds would then be written past the end of it.
            //
            // The direction matters: over-stating a bound you *allocate* is
            // merely wasteful, while over-stating a capacity you *trust* is an
            // overrun. This is the only place the value is used as the latter.
            const TJSAMP_GRAY: c_int = 3;
            let src_subsamp: c_int = if info.components == 1 {
                TJSAMP_GRAY
            } else {
                info.subsampling.to_tjsamp()
            };
            let mut sizes: Vec<usize> = Vec::with_capacity(n as usize);
            for index in 0..n as usize {
                // SAFETY: `transforms` points to `n` entries per the caller's
                // contract, checked non-NULL above.
                let xform: &TjTransform = unsafe { &*transforms.add(index) };
                let (w, h, subsamp) = crate::transform::transformed_specs(
                    info.width as c_int,
                    info.height as c_int,
                    src_subsamp,
                    xform,
                );
                sizes.push(crate::bufsize::tj3JPEGBufSize(w, h, subsamp));
            }
            Some(sizes)
        } else {
            None
        };

        let rc: c_int = match sizes {
            Some(mut sizes) => {
                // SAFETY: as the direct call below, with a local capacity array
                // standing in for the caller's output slots.
                let rc: c_int = unsafe {
                    tj3Transform(
                        handle,
                        jpeg_buf,
                        jpeg_size,
                        n,
                        dst_bufs,
                        sizes.as_mut_ptr(),
                        transforms,
                    )
                };
                // Upstream copies the produced sizes back unconditionally
                // (`turbojpeg.c:3135-3136`), so a caller reading `dstSizes`
                // after a partial failure sees what was written.
                for (index, size) in sizes.iter().enumerate() {
                    // SAFETY: `dst_sizes` has `n` entries per the contract.
                    unsafe { *dst_sizes.add(index) = *size };
                }
                rc
            }
            // SAFETY: the caller's slots carry their own capacities when the
            // flag is unset, which is the reallocating path.
            None => unsafe {
                tj3Transform(
                    handle, jpeg_buf, jpeg_size, n, dst_bufs, dst_sizes, transforms,
                )
            },
        };
        rc
    })
}

// ---------------------------------------------------------------------------
// YUV (forwards to the A1-7 family in `crate::yuv`)
// ---------------------------------------------------------------------------

/// `tjEncodeYUV3(handle, srcBuf, width, pitch, height, pixelFormat,
///               dstBuf, align, subsamp, flags) -> int`.
///
/// Legacy alias matching the **upstream** `turbojpeg.h` signature:
/// the 4th argument is `pitch` (input RGB row stride in bytes — 0
/// means tight `width * bytes_per_pixel`), **not** YUV alignment.
/// The 8th argument `align` is the YUV plane row-stride alignment.
/// Sets `TJPARAM_SUBSAMP` from `subsamp`, propagates legacy
/// `flags` (`TJFLAG_BOTTOMUP`, `TJFLAG_FASTDCT`) to their
/// `TJPARAM_*` counterparts via `process_legacy_compress_flags`,
/// then forwards to `tj3EncodeYUV8` with the caller's pitch and
/// align preserved.
///
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjEncodeYUV3(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    dst_buf: *mut u8,
    align: c_int,
    subsamp: c_int,
    flags: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        if unsafe { tj3Set(handle, TJPARAM_SUBSAMP, subsamp) } != 0 {
            return -1;
        }
        process_legacy_compress_flags(handle, flags);
        unsafe {
            crate::yuv::tj3EncodeYUV8(
                handle,
                src_buf,
                width,
                pitch,
                height,
                pixel_format,
                dst_buf,
                align.max(1),
            )
        }
    })
}

/// `tjDecodeYUV(handle, srcBuf, align, subsamp, dstBuf, width, pitch,
///              height, pixelFormat, flags) -> int`.
///
/// Legacy alias that forwards to `tj3DecodeYUV8` after setting
/// `TJPARAM_SUBSAMP` and propagating legacy `flags`
/// (`TJFLAG_BOTTOMUP`, `TJFLAG_FASTUPSAMPLE`, `TJFLAG_FASTDCT`) to
/// their `TJPARAM_*` counterparts on the caller's handle, mirroring
/// upstream `turbojpeg.c::processFlags(DECOMPRESS)`.
///
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjDecodeYUV(
    handle: *mut c_void,
    src_buf: *const u8,
    align: c_int,
    subsamp: c_int,
    dst_buf: *mut u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    flags: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        if unsafe { tj3Set(handle, TJPARAM_SUBSAMP, subsamp) } != 0 {
            return -1;
        }
        process_legacy_decompress_flags(handle, flags);
        unsafe {
            crate::yuv::tj3DecodeYUV8(
                handle,
                src_buf,
                align.max(1),
                dst_buf,
                width,
                pitch,
                height,
                pixel_format,
            )
        }
    })
}

// Legacy `flags` bits per upstream `turbojpeg.h`. We translate the
// subset that round-trips through `TJPARAM_*`; unknown bits are
// silently ignored (matching upstream's behaviour for the
// remaining `TJFLAG_FORCE*` SIMD-dispatch hints).
const TJFLAG_FASTUPSAMPLE: c_int = 256;
const TJFLAG_FASTDCT: c_int = 2048;
const TJFLAG_ACCURATEDCT: c_int = 4096;
const TJFLAG_PROGRESSIVE: c_int = 16384;
// `TJFLAG_BOTTOMUP` is also defined module-locally near the
// load/save wrappers; keep that single canonical declaration.
const TJPARAM_PROGRESSIVE: c_int = 12;
const TJPARAM_FASTUPSAMPLE: c_int = 9;
const TJPARAM_FASTDCT: c_int = 10;

/// Mirror of upstream `turbojpeg.c::processFlags(handle, flags,
/// COMPRESS)`. Sets the `TJPARAM_*` counterparts of legacy
/// compress-side flag bits on the caller's handle.
///
/// **FastDCT semantics**: upstream does NOT use `TJFLAG_FASTDCT`
/// directly on COMPRESS. It computes
/// `fastDCT = (quality < 96) && !(flags & TJFLAG_ACCURATEDCT)`.
/// We read the current quality back via
/// `tj3Get(handle, TJPARAM_QUALITY)` and apply the same rule, so
/// `tjEncodeYUV3(..., flags=0)` at quality 75 ends up with
/// `TJPARAM_FASTDCT=1` (matching libjpeg-turbo's default), while
/// quality ≥ 96 or `TJFLAG_ACCURATEDCT` clears it.
fn process_legacy_compress_flags(handle: *mut c_void, flags: c_int) {
    let _ = unsafe {
        tj3Set(
            handle,
            TJPARAM_BOTTOMUP,
            (flags & TJFLAG_BOTTOMUP != 0) as c_int,
        )
    };
    let _ = unsafe {
        tj3Set(
            handle,
            TJPARAM_PROGRESSIVE,
            (flags & TJFLAG_PROGRESSIVE != 0) as c_int,
        )
    };
    let quality: c_int = unsafe { crate::tj3::tj3Get(handle, TJPARAM_QUALITY) };
    let accurate_dct: bool = (flags & TJFLAG_ACCURATEDCT) != 0;
    let fast_dct: bool = quality < 96 && !accurate_dct;
    let _ = unsafe { tj3Set(handle, TJPARAM_FASTDCT, fast_dct as c_int) };
}

/// Mirror of upstream `turbojpeg.c::processFlags(handle, flags,
/// DECOMPRESS)`. On the decompress side, `fastDCT` is driven
/// directly by `TJFLAG_FASTDCT` (no quality/ACCURATEDCT interplay
/// applies — there is no compression quality knob to balance
/// against).
fn process_legacy_decompress_flags(handle: *mut c_void, flags: c_int) {
    let _ = unsafe {
        tj3Set(
            handle,
            TJPARAM_BOTTOMUP,
            (flags & TJFLAG_BOTTOMUP != 0) as c_int,
        )
    };
    let _ = unsafe {
        tj3Set(
            handle,
            TJPARAM_FASTUPSAMPLE,
            (flags & TJFLAG_FASTUPSAMPLE != 0) as c_int,
        )
    };
    let _ = unsafe {
        tj3Set(
            handle,
            TJPARAM_FASTDCT,
            (flags & TJFLAG_FASTDCT != 0) as c_int,
        )
    };
}

// ---------------------------------------------------------------------------
// Buffer sizing (pure computations — no handle required)
// ---------------------------------------------------------------------------

/// `tjBufSize(width, height, jpegSubsamp) -> unsigned long`.
///
/// Mirrors the C wrapper semantics in `turbojpeg.c`: internally delegates
/// to `tj3JPEGBufSize` and returns `(unsigned long)-1` (usize::MAX) when
/// the TJ3 helper returns 0 (invalid input or overflow), so callers that
/// compare against `(unsigned long)-1` — as `tjunittest.c::overflowTest`
/// does — see a stable "error" sentinel.
#[no_mangle]
pub extern "C" fn tjBufSize(width: c_int, height: c_int, jpeg_subsamp: c_int) -> usize {
    crate::unwind_guard!(usize::MAX, {
        let retval: usize = crate::bufsize::tj3JPEGBufSize(width, height, jpeg_subsamp);
        if retval == 0 {
            usize::MAX
        } else {
            retval
        }
    })
}

/// `TJBUFSIZE(width, height) -> unsigned long` — TurboJPEG 1.0 legacy
/// upper-bound sizing helper that assumes 4:4:4 and the widest worst
/// case. Returns `(unsigned long)-1` (usize::MAX) on invalid input, per
/// the historical contract in `turbojpeg.c`.
#[no_mangle]
pub extern "C" fn TJBUFSIZE(width: c_int, height: c_int) -> usize {
    crate::unwind_guard!(usize::MAX, {
        if width < 1 || height < 1 {
            return usize::MAX;
        }
        // Matches turbojpeg.c: PAD(width, 16) * PAD(height, 16) * 6 + 2048.
        let pad_w: usize = ((width as usize) + 15) & !15;
        let pad_h: usize = ((height as usize) + 15) & !15;
        pad_w
            .checked_mul(pad_h)
            .and_then(|v| v.checked_mul(6))
            .and_then(|v| v.checked_add(2048))
            .unwrap_or(usize::MAX)
    })
}

/// `TJBUFSIZEYUV(width, height, subsamp) -> unsigned long` — TurboJPEG
/// 1.1 legacy helper that delegates to `tjBufSizeYUV`.
#[no_mangle]
pub extern "C" fn TJBUFSIZEYUV(width: c_int, height: c_int, subsamp: c_int) -> usize {
    crate::unwind_guard!(usize::MAX, { tjBufSizeYUV(width, height, subsamp) })
}

/// `tjBufSizeYUV(width, height, subsamp) -> unsigned long` — TurboJPEG
/// 1.1 legacy wrapper that hard-codes `align = 4`.
#[no_mangle]
pub extern "C" fn tjBufSizeYUV(width: c_int, height: c_int, subsamp: c_int) -> usize {
    crate::unwind_guard!(usize::MAX, { tjBufSizeYUV2(width, 4, height, subsamp) })
}

/// `tjBufSizeYUV2(width, align, height, subsamp) -> unsigned long`.
///
/// Delegates to `tj3YUVBufSize` and returns `(unsigned long)-1`
/// (usize::MAX) on the 0-return error path, matching the C wrapper in
/// `turbojpeg.c`. `tjunittest.c::overflowTest` relies on this sentinel
/// when `align` is a non-power-of-two or negative value.
#[no_mangle]
pub extern "C" fn tjBufSizeYUV2(
    width: c_int,
    align: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    crate::unwind_guard!(usize::MAX, {
        let retval: usize = crate::bufsize::tj3YUVBufSize(width, align, height, subsamp);
        if retval == 0 {
            usize::MAX
        } else {
            retval
        }
    })
}

/// `tjPlaneSizeYUV(componentID, width, stride, height, subsamp)`.
///
/// Delegates to `tj3YUVPlaneSize` and returns `(unsigned long)-1` on the
/// 0-return error path, matching the C wrapper's sentinel value.
#[no_mangle]
pub extern "C" fn tjPlaneSizeYUV(
    component_id: c_int,
    width: c_int,
    stride: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    crate::unwind_guard!(usize::MAX, {
        let retval: usize =
            crate::bufsize::tj3YUVPlaneSize(component_id, width, stride, height, subsamp);
        if retval == 0 {
            usize::MAX
        } else {
            retval
        }
    })
}

/// `tjPlaneWidth(componentID, width, subsamp)`.
///
/// Delegates to `tj3YUVPlaneWidth` and maps its 0 to the pre-3.0 `-1`
/// sentinel, exactly as upstream does (`turbojpeg.c`):
///
/// ```c
/// int retval = tj3YUVPlaneWidth(componentID, width, subsamp);
/// return (retval == 0) ? -1 : retval;
/// ```
///
/// Delegating is what makes the `componentID >= nc` bound apply here. The
/// previous implementation re-derived the answer from the root-crate
/// `yuv_plane_width`, whose `Subsampling` argument has no grayscale variant, so
/// this layer had to translate `TJSAMP_GRAY` into `S444` and the distinction was
/// gone by the time the helper saw it: grayscale components 1 and 2 came back as
/// full-size planes instead of being rejected (P4-126). Same shape as P4-125 —
/// upstream needs one guard because it delegates; re-deriving needs a second
/// copy of every bound.
#[no_mangle]
pub extern "C" fn tjPlaneWidth(component_id: c_int, width: c_int, subsamp: c_int) -> c_int {
    crate::unwind_guard!(-1, {
        match crate::bufsize::tj3YUVPlaneWidth(component_id, width, subsamp) {
            0 => -1,
            retval => retval,
        }
    })
}

/// `tjPlaneHeight(componentID, height, subsamp)`.
///
/// Delegates to `tj3YUVPlaneHeight` for the reasons given on
/// [`tjPlaneWidth`].
#[no_mangle]
pub extern "C" fn tjPlaneHeight(component_id: c_int, height: c_int, subsamp: c_int) -> c_int {
    crate::unwind_guard!(-1, {
        match crate::bufsize::tj3YUVPlaneHeight(component_id, height, subsamp) {
            0 => -1,
            retval => retval,
        }
    })
}

// ---------------------------------------------------------------------------
// Load / Save — handle-less legacy ABI delegating to tj3LoadImage8 / tj3SaveImage8
// ---------------------------------------------------------------------------

/// Snapshot the temporary handle's last-error message into the
/// process-global no-handle error slot, so that
/// `tj3GetErrorStr(NULL)` / `tjGetErrorStr2(NULL)` return a
/// meaningful diagnostic after the temp handle is destroyed. Mirrors
/// upstream `turbojpeg.c::tjLoadImage` / `tjSaveImage`, which surface
/// the inner error through the global slot for the legacy
/// handle-less ABI.
fn copy_handle_error_to_no_handle_slot(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: handle is non-NULL and was returned by tj3Init above.
    let inst: &crate::tj3::TjInstance = unsafe { &*(handle as *const crate::tj3::TjInstance) };
    if let Ok(s) = inst.last_error.to_str() {
        crate::bufsize::set_no_handle_error(s);
    }
}

/// `tjLoadImage(filename, width, align, height, pixelFormat, flags)`.
///
/// Legacy 2.x signature is **handle-less** — upstream `turbojpeg.c`
/// allocates a temporary `tjhandle`, sets `TJPARAM_BOTTOMUP` from
/// `flags & TJFLAG_BOTTOMUP`, calls `tj3LoadImage8`, then frees the
/// handle. We mirror that exactly, including snapshotting the
/// temp handle's last-error into the no-handle global slot before
/// destroying so callers can recover the diagnostic via
/// `tjGetErrorStr2(NULL)`.
///
/// # Safety
///
/// C ABI entry point. `filename`, `width`, `height`, `pixel_format` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjLoadImage(
    filename: *const c_char,
    width: *mut c_int,
    align: c_int,
    height: *mut c_int,
    pixel_format: *mut c_int,
    flags: c_int,
) -> *mut u8 {
    crate::unwind_guard!(std::ptr::null_mut(), {
        // Create a temporary decompress handle so the underlying TJ3
        // form has somewhere to record errors. `TJINIT_DECOMPRESS = 2`
        // matches `tj3.rs` and `turbojpeg.h`.
        let h: *mut c_void = crate::tj3::tj3Init(2);
        if h.is_null() {
            crate::bufsize::set_no_handle_error("tjLoadImage: tj3Init(TJINIT_DECOMPRESS) failed");
            return std::ptr::null_mut();
        }
        if (flags & TJFLAG_BOTTOMUP) != 0 {
            // TJPARAM_BOTTOMUP = 0 in turbojpeg.h, but use the
            // tj3-published constant via tj3Set to stay layout-independent.
            let _ = unsafe { crate::tj3::tj3Set(h, TJPARAM_BOTTOMUP, 1) };
        }
        let buf: *mut u8 = unsafe {
            crate::imageio::tj3LoadImage8(h, filename, width, align, height, pixel_format)
        };
        if buf.is_null() {
            copy_handle_error_to_no_handle_slot(h);
        }
        // SAFETY: `h` came from tj3Init above, is non-null, has not been
        // destroyed, and no other thread can name it — it never left this
        // function.
        unsafe { crate::tj3::tj3Destroy(h) };
        buf
    })
}

/// `tjSaveImage(filename, buffer, width, pitch, height, pixelFormat, flags)`.
///
/// Legacy 2.x signature: also handle-less. Same handle lifecycle as
/// `tjLoadImage` (temp `tjhandle`, propagate `TJFLAG_BOTTOMUP`,
/// delegate, free, copy error before destroy).
///
/// # Safety
///
/// C ABI entry point. `filename`, `buffer` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjSaveImage(
    filename: *const c_char,
    buffer: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    flags: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        let h: *mut c_void = crate::tj3::tj3Init(1); // TJINIT_COMPRESS
        if h.is_null() {
            crate::bufsize::set_no_handle_error("tjSaveImage: tj3Init(TJINIT_COMPRESS) failed");
            return -1;
        }
        if (flags & TJFLAG_BOTTOMUP) != 0 {
            let _ = unsafe { crate::tj3::tj3Set(h, TJPARAM_BOTTOMUP, 1) };
        }
        let rc: c_int = unsafe {
            crate::imageio::tj3SaveImage8(h, filename, buffer, width, pitch, height, pixel_format)
        };
        if rc != 0 {
            copy_handle_error_to_no_handle_slot(h);
        }
        // SAFETY: `h` came from tj3Init above, is non-null, has not been
        // destroyed, and no other thread can name it — it never left this
        // function.
        unsafe { crate::tj3::tj3Destroy(h) };
        rc
    })
}

/// Legacy `TJFLAG_BOTTOMUP` bit and `TJPARAM_BOTTOMUP` index per
/// upstream `turbojpeg.h`. Kept module-local so legacy translation
/// doesn't pull in additional constant exports.
const TJFLAG_BOTTOMUP: c_int = 2;
const TJPARAM_BOTTOMUP: c_int = 1;

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

/// `tjGetErrorStr2(handle) -> const char *` — identical to
/// `tj3GetErrorStr` with a handle-aware NULL fallback.
///
/// # Safety
///
/// C ABI entry point. `handle` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tjGetErrorStr2(handle: *mut c_void) -> *const c_char {
    crate::unwind_guard!(std::ptr::null(), { unsafe { tj3GetErrorStr(handle) } })
}

// ---------------------------------------------------------------------------
// Width-from-scaled-dimensions helper referenced by some legacy users.
// Exported for binary compat but powered by the Rust `calc_jpeg_dimensions`.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn _calc_reference(w: i32, h: i32) {
    // Kept as a compile-time reference to the sizing helper we depend on.
    let (_w, _h) = calc_jpeg_dimensions(w as usize, h as usize, Subsampling::S420);
}

// Silence a potentially-unused constant on targets where TjRegion is not
// referenced by this module directly (the file imports it for legacy
// consumers that include this header).
#[allow(dead_code)]
const _TJREGION_MARKER: TjRegion = TjRegion {
    x: 0,
    y: 0,
    w: 0,
    h: 0,
};
