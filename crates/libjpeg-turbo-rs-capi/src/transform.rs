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

            // Every transform's arguments are validated before the header is
            // read, as upstream's first loop does (`turbojpeg.c:2963-2988`): a
            // bad entry anywhere in the batch is refused before any output is
            // produced or any limit is consulted.
            // Fallibly, as upstream checks its `malloc` of `xinfo`
            // (`turbojpeg.c:2953-2955`): `n` is the caller's, and an
            // infallible reservation that fails aborts the process.
            let mut batch: Vec<TransformOptions> = Vec::new();
            if batch.try_reserve_exact(txforms.len()).is_err() {
                inst.set_error("tj3Transform(): Memory allocation failure", TJERR_FATAL);
                return -1;
            }
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
                    // Only a negative field is invalid (`turbojpeg.c:2970-2971`);
                    // a zero width or height is `JCROP_UNSET`, "to the edge"
                    // (`:2979-2986`), resolved against the transformed frame
                    // once the header is read (P4-240, #675).
                    if t.r.x < 0 || t.r.y < 0 || t.r.w < 0 || t.r.h < 0 {
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
                apply_output_parameters(inst, &mut opts);
                batch.push(opts);
            }

            // Upstream registers marker processors before reading the header
            // whenever any transform in the batch saves markers
            // (`jcopy_markers_setup` with the handle's saveMarkers option,
            // `turbojpeg.c:2988-2992`); registration is per-handle and
            // permanent. Recorded so the legacy NOREALLOC bridge can tell a
            // cold handle — where upstream's capacity pre-read starves marker
            // saving, the P4-156 ordering quirk — from a warm one
            // (P4-156, #544).
            if txforms
                .iter()
                .any(|t: &TjTransform| (t.options & TJXOPT_COPYNONE) == 0)
                && inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::SaveMarkers) != 0
            {
                inst.transform_markers_registered = true;
            }

            // The handle's limits, against the source, in upstream's order
            // (P4-227, #655).
            if let Some(refusal) = source_refusal(inst, jpeg, txforms, &batch) {
                inst.set_error(refusal, TJERR_FATAL);
                return -1;
            }

            // Process each transform independently. libjpeg-turbo does this in a
            // loop too; there's no shared decode state across transforms.
            for (i, (t, opts)) in txforms.iter().zip(batch.iter()).enumerate() {
                let out: Vec<u8> = match transform_jpeg_with_options(jpeg, opts) {
                    Ok(v) => v,
                    Err(e) => {
                        inst.set_error(format!("tj3Transform[{i}]: {e}"), TJERR_FATAL);
                        return -1;
                    }
                };

                // Upstream skips destination setup entirely for this option —
                // `if (!(t[i].options & TJXOPT_NOOUTPUT)) jpeg_mem_dest_tj(...)`
                // (`turbojpeg.c:3022`) — so the slots stay exactly as the caller
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

/// Fold the handle's output parameters into one transform's options, as
/// upstream does for every transform in the batch (`turbojpeg.c:3029-3037`):
/// `TJPARAM_PROGRESSIVE`, `TJPARAM_ARITHMETIC` and `TJPARAM_OPTIMIZE` are
/// OR-ed with their `TJXOPT_*` twins, and the restart interval comes from
/// `TJPARAM_RESTARTROWS` when set, else `TJPARAM_RESTARTBLOCKS` — the
/// precedence `jcmaster.c` gives `restart_in_rows` (P4-227, #655).
/// `TransformOptions` already drops `optimize` under `arithmetic`, as
/// upstream's `optimize_coding = FALSE` does.
fn apply_output_parameters(inst: &crate::tj3::TjInstance, opts: &mut TransformOptions) {
    use libjpeg_turbo_rs::tj3::TjParam;
    opts.progressive |= inst.inner.get(TjParam::Progressive) != 0;
    opts.arithmetic |= inst.inner.get(TjParam::Arithmetic) != 0;
    opts.optimize |= inst.inner.get(TjParam::Optimize) != 0;
    let rows: c_int = inst.inner.get(TjParam::RestartRows);
    let blocks: c_int = inst.inner.get(TjParam::RestartBlocks);
    if rows > 0 {
        opts.restart_interval = rows as u16;
        opts.restart_in_rows = true;
    } else if blocks > 0 {
        opts.restart_interval = blocks as u16;
        opts.restart_in_rows = false;
    }
}

/// `tjMCUWidth` / `tjMCUHeight` (`turbojpeg.h`), indexed by `TJSAMP_*`.
const TJ_MCU_WIDTH: [usize; 9] = [8, 16, 16, 8, 8, 32, 8, 32, 16];
const TJ_MCU_HEIGHT: [usize; 9] = [8, 8, 16, 8, 16, 8, 32, 16, 32];

/// `jtransform_perfect_transform` (`transupp.c:2415-2450`): which edges must
/// be whole iMCUs for `op` to lose nothing.
fn is_perfect(op: TransformOp, width: usize, height: usize, imcu: (usize, usize)) -> bool {
    let width_whole: bool = width.is_multiple_of(imcu.0);
    let height_whole: bool = height.is_multiple_of(imcu.1);
    match op {
        TransformOp::HFlip | TransformOp::Rot270 => width_whole,
        TransformOp::VFlip | TransformOp::Rot90 => height_whole,
        TransformOp::Transverse | TransformOp::Rot180 => width_whole && height_whole,
        _ => true,
    }
}

/// Whether `jtransform_request_workspace` refuses `opts`' crop region for a
/// `width` x `height` source (`transupp.c:1705-1757`). The region is in the
/// transformed frame. Extending past the frame is allowed only for
/// `TJXOP_NONE`, and only when the offset leaves the original image inside it.
fn crop_is_refused(opts: &TransformOptions, width: usize, height: usize) -> bool {
    let Some(crop) = opts.crop else {
        return false;
    };
    let transposed: bool = matches!(
        opts.op,
        TransformOp::Transpose | TransformOp::Transverse | TransformOp::Rot90 | TransformOp::Rot270
    );
    let (out_width, out_height): (usize, usize) = if transposed {
        (height, width)
    } else {
        (width, height)
    };
    let axis_refused = |offset: usize, extent: usize, frame: usize| -> bool {
        if extent == 0 {
            // `JCROP_UNSET`: to the edge, refused only for an origin outside.
            return offset >= frame;
        }
        if extent > frame {
            opts.op != TransformOp::None || offset >= extent || offset > extent - frame
        } else {
            offset >= frame || offset > frame - extent
        }
    };
    axis_refused(crop.x, crop.width, out_width) || axis_refused(crop.y, crop.height, out_height)
}

/// What libjpeg's memory manager holds for the markers `jcopy_markers_setup`
/// saves under copy option `option` (`TJSM_*` / `JCOPYOPT_*`,
/// `transupp.c:2460-2483`) by the time `jpeg_read_coefficients` realizes its
/// arrays: COM unless NONE or ICC-only, every APPn for ALL, all but APP2 for
/// ALL-EXCEPT-ICC, APP2 alone for ICC. Each saved marker is one `alloc_large`
/// of `sizeof(struct jpeg_marker_struct) + length` (`jdmarker.c:783-784`),
/// which the large pool charges with its header and `ALIGN_SIZE - 1` of slack
/// (`jmemmgr.c`): 32 + 32 + 31 bytes on an LP64 SIMD build. Measured: 32
/// 64 KiB APP5 segments push a 6 MiB source from "accepted at 8 MiB" to
/// "refused at 8, accepted at 9", as on stock 3.2.0.
fn retained_marker_bytes(jpeg: &[u8], option: c_int) -> u64 {
    const PER_MARKER_OVERHEAD: u64 = 32 + 32 + 31;
    const COMMENTS: c_int = 1;
    const ALL: c_int = 2;
    const ALL_EXCEPT_ICC: c_int = 3;
    const ICC: c_int = 4;
    let saves = |code: u8| -> bool {
        match code {
            0xFE => option == COMMENTS || option == ALL || option == ALL_EXCEPT_ICC,
            0xE2 => option == ALL || option == ICC,
            0xE0..=0xEF => option == ALL || option == ALL_EXCEPT_ICC,
            _ => false,
        }
    };
    let mut total: u64 = 0;
    let mut at: usize = 2;
    // Marker segments from SOI to the first SOS; anything malformed ends the
    // walk, as the header parse has already accepted the stream.
    while at + 4 <= jpeg.len() && jpeg[at] == 0xFF {
        let code: u8 = jpeg[at + 1];
        if code == 0xDA {
            break;
        }
        let length: usize = usize::from(jpeg[at + 2]) << 8 | usize::from(jpeg[at + 3]);
        if length < 2 {
            break;
        }
        if saves(code) {
            total = total.saturating_add((length - 2) as u64 + PER_MARKER_OVERHEAD);
        }
        at += 2 + length;
    }
    total
}

/// Bytes in one whole-image coefficient array: `blocks_wide` x `blocks_high`
/// `JBLOCK`s of 64 `JCOEF`s.
fn coefficient_bytes(blocks_wide: usize, blocks_high: usize) -> u64 {
    (blocks_wide as u64)
        .saturating_mul(blocks_high as u64)
        .saturating_mul(128)
}

/// What `tj3Transform` asks of libjpeg's memory manager before reading a
/// single scan: the source's whole-image coefficient arrays
/// (`jinit_d_coef_controller`, each component padded to its sampling factor)
/// plus every transform's workspace (`jtransform_request_workspace`,
/// `transupp.c:1855-1960`), all realized together by
/// `jpeg_read_coefficients`. Stock refuses when this does not fit in
/// `TJPARAM_MAXMEMORY` MiB less what its pools already hold; that overhead is
/// a few KiB and is not modelled, so the estimate refuses at
/// `estimate >= budget` — measured equal to stock at the 6/7 and 12/13 MiB
/// boundaries of a 6 MiB source. A trimmed transform is estimated untrimmed,
/// and a crop by its requested region, so near the boundary those may refuse
/// a little earlier than stock.
///
/// This is upstream's accounting, not this port's: the transform here holds
/// the source coefficients and its own working copies in `Vec`s whose sizes
/// differ from libjpeg's virtual arrays. `TJPARAM_MAXMEMORY` is applied as
/// stock applies it, so the same sources are refused; it is not a ceiling on
/// what the Rust transform allocates.
fn transform_memory_estimate(
    decoder: &libjpeg_turbo_rs::Decoder<'_>,
    batch: &[TransformOptions],
    retained_markers: u64,
) -> u64 {
    let frame = decoder.header();
    let (width, height): (usize, usize) = (frame.width(), frame.height());
    let max_h: usize = frame
        .components
        .iter()
        .map(|c| usize::from(c.horizontal_sampling))
        .max()
        .unwrap_or(1);
    let max_v: usize = frame
        .components
        .iter()
        .map(|c| usize::from(c.vertical_sampling))
        .max()
        .unwrap_or(1);
    let mut estimate: u64 = frame
        .components
        .iter()
        .map(|c| {
            let (h, v): (usize, usize) = (
                usize::from(c.horizontal_sampling),
                usize::from(c.vertical_sampling),
            );
            let blocks_wide: usize = (width * h).div_ceil(max_h * 8).next_multiple_of(h);
            let blocks_high: usize = (height * v).div_ceil(max_v * 8).next_multiple_of(v);
            coefficient_bytes(blocks_wide, blocks_high)
        })
        .fold(retained_markers, u64::saturating_add);
    for opts in batch {
        let gray_only: bool = opts.grayscale
            && frame.components.len() == 3
            && decoder.jpeg_color_space() == libjpeg_turbo_rs::ColorSpace::YCbCr;
        let transposed: bool = matches!(
            opts.op,
            TransformOp::Transpose
                | TransformOp::Transverse
                | TransformOp::Rot90
                | TransformOp::Rot270
        );
        let crop_offset: (usize, usize) = opts.crop.map_or((0, 0), |c| (c.x, c.y));
        // A crop larger than the frame — legal only without a transform — is
        // an expansion, which needs a workspace of the expanded size.
        let expands: bool = opts
            .crop
            .is_some_and(|c| c.width > width || c.height > height);
        let needs_workspace: bool = match opts.op {
            TransformOp::None => crop_offset != (0, 0) || expands,
            // `slow_hflip` is set for any batch of more than one transform.
            TransformOp::HFlip => crop_offset.1 != 0 || batch.len() != 1,
            _ => true,
        };
        if !needs_workspace {
            continue;
        }
        let (mut out_width, mut out_height): (usize, usize) = if transposed {
            (height, width)
        } else {
            (width, height)
        };
        if let Some(crop) = opts.crop {
            // A zero extent runs to the edge (`JCROP_UNSET`).
            let extent = |extent: usize, offset: usize, frame: usize| -> usize {
                if extent == 0 {
                    frame.saturating_sub(offset)
                } else if extent > frame {
                    extent
                } else {
                    extent.min(frame.saturating_sub(offset))
                }
            };
            out_width = extent(crop.width, crop.x, out_width);
            out_height = extent(crop.height, crop.y, out_height);
        }
        let components: usize = if gray_only { 1 } else { frame.components.len() };
        let (imcu_width, imcu_height): (usize, usize) = match (components, transposed) {
            (1, _) => (8, 8),
            (_, true) => (max_v * 8, max_h * 8),
            (_, false) => (max_h * 8, max_v * 8),
        };
        let (imcus_wide, imcus_high): (usize, usize) = (
            out_width.div_ceil(imcu_width),
            out_height.div_ceil(imcu_height),
        );
        for c in frame.components.iter().take(components) {
            let (h, v): (usize, usize) = match (components, transposed) {
                (1, _) => (1, 1),
                (_, true) => (
                    usize::from(c.vertical_sampling),
                    usize::from(c.horizontal_sampling),
                ),
                (_, false) => (
                    usize::from(c.horizontal_sampling),
                    usize::from(c.vertical_sampling),
                ),
            };
            estimate = estimate.saturating_add(coefficient_bytes(
                imcus_wide.saturating_mul(h),
                imcus_high.saturating_mul(v),
            ));
        }
    }
    estimate
}

/// The handle's limits against the source, in the order upstream's
/// `tj3Transform` applies them (P4-227, #655): `TJPARAM_MAXPIXELS` right
/// after the header read (`turbojpeg.c:2995-2998`), then each transform's
/// `TJXOPT_PERFECT` test (`jtransform_request_workspace`, `:3002-3004`), then,
/// inside `jpeg_read_coefficients`, `TJPARAM_MAXMEMORY` when the coefficient
/// arrays are realized and `TJPARAM_SCANLIMIT` as each scan is read
/// (`:2943-2951`). Messages are upstream's.
///
/// `None` when nothing is refused — including a header that does not parse,
/// which the transform itself goes on to report, as it always has.
fn source_refusal(
    inst: &crate::tj3::TjInstance,
    jpeg: &[u8],
    txforms: &[TjTransform],
    batch: &[TransformOptions],
) -> Option<String> {
    use libjpeg_turbo_rs::tj3::TjParam;
    use libjpeg_turbo_rs::{ColorSpace, DecodeLimits, Decoder};

    // No cap but libjpeg's own `JPEG_MAX_DIMENSION`: the handle's limits are
    // applied below, each where upstream applies it.
    let header_limits: DecodeLimits = DecodeLimits {
        max_width: 65_500,
        max_height: 65_500,
        max_pixels: u64::MAX,
        max_scans: usize::MAX,
        max_memory: None,
    };
    // Markers up to the first SOS only, as `jpeg_read_header` reads them: the
    // pixel cap must refuse an oversized progressive source before any of its
    // later scans is walked.
    let decoder: Decoder<'_> = Decoder::new_header_only(jpeg, header_limits).ok()?;
    let frame = decoder.header();
    let (width, height): (usize, usize) = (frame.width(), frame.height());

    let max_pixels: c_int = inst.inner.get(TjParam::MaxPixels);
    if max_pixels > 0 && (width as u64) * (height as u64) > max_pixels as u64 {
        return Some(String::from("tj3Transform(): Image is too large"));
    }

    let max_h: usize = frame
        .components
        .iter()
        .map(|c| usize::from(c.horizontal_sampling))
        .max()
        .unwrap_or(1);
    let max_v: usize = frame
        .components
        .iter()
        .map(|c| usize::from(c.vertical_sampling))
        .max()
        .unwrap_or(1);
    // The source's `getSubsamp`, which the crop alignment below is checked
    // against, read the way every TurboJPEG entry point reads it.
    let source_subsamp: c_int = {
        let mut probe: libjpeg_turbo_rs::tj3::TjHandle = libjpeg_turbo_rs::tj3::TjHandle::new();
        probe.decompress_header_info(jpeg).ok()?;
        probe.get(TjParam::Subsampling)
    };
    // Per transform, in upstream's order (`turbojpeg.c:3000-3016`):
    // `jtransform_request_workspace`'s PERFECT test and crop validation
    // (`transupp.c:1661-1757`, `JERR_BAD_CROP_SPEC`), then tj3Transform's
    // own crop alignment against the destination subsampling — all before
    // `jpeg_read_coefficients`, so before the memory and scan limits below.
    for (transform, opts) in txforms.iter().zip(batch) {
        if opts.perfect {
            let one_component: bool = frame.components.len() == 1
                || (opts.grayscale
                    && frame.components.len() == 3
                    && decoder.jpeg_color_space() == ColorSpace::YCbCr);
            let imcu: (usize, usize) = if one_component {
                (8, 8)
            } else {
                (max_h * 8, max_v * 8)
            };
            if !is_perfect(opts.op, width, height, imcu) {
                return Some(String::from("tj3Transform(): Transform is not perfect"));
            }
        }
        if crop_is_refused(opts, width, height) {
            return Some(String::from("Invalid crop request"));
        }
        if let Some(crop) = opts.crop {
            let (_, _, dst_subsamp) = transformed_specs(0, 0, source_subsamp, transform);
            let Some(&mcu_width) = usize::try_from(dst_subsamp)
                .ok()
                .and_then(|index| TJ_MCU_WIDTH.get(index))
            else {
                return Some(String::from(
                    "tj3Transform(): Could not determine subsampling level of destination image",
                ));
            };
            let mcu_height: usize = TJ_MCU_HEIGHT[dst_subsamp as usize];
            if crop.x % mcu_width != 0 || crop.y % mcu_height != 0 {
                return Some(format!(
                    "tj3Transform(): To crop this JPEG image, x must be a multiple of \
                     {mcu_width}\nand y must be a multiple of {mcu_height}."
                ));
            }
        }
    }

    let max_memory: c_int = inst.inner.get(TjParam::MaxMemory);
    // The markers `jcopy_markers_setup` saved while the header was read sit
    // in the same pools, so they count against the same budget.
    let copy_option: c_int = if batch
        .iter()
        .any(|opts| opts.copy_markers != MarkerCopyMode::None)
    {
        inst.inner.get(TjParam::SaveMarkers)
    } else {
        0
    };
    if max_memory > 0
        && transform_memory_estimate(&decoder, batch, retained_marker_bytes(jpeg, copy_option))
            >= max_memory as u64 * 1_048_576
    {
        return Some(String::from("Memory limit exceeded"));
    }

    let scan_limit: c_int = inst.inner.get(TjParam::ScanLimit);
    if scan_limit > 0 {
        let scan_capped: DecodeLimits = DecodeLimits {
            max_scans: scan_limit as usize,
            ..header_limits
        };
        if let Err(libjpeg_turbo_rs::JpegError::LimitExceeded { what, .. }) =
            Decoder::new_with_limits(jpeg, scan_capped)
        {
            if what.starts_with("scan count") {
                return Some(format!(
                    "Progressive JPEG image has more than {scan_limit} scans"
                ));
            }
        }
    }
    None
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
/// 8192-byte destination against a 139264-byte capacity (P4-151).
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
    // Crop: upstream's `getTransformedSpecs` (`turbojpeg.c:2847-2870`) takes
    // `r.w` / `r.h`, and a zero one as the remainder past `r.x` / `r.y`
    // (`JCROP_UNSET`, P4-240). Its range checks are not repeated here: a
    // region `tj3Transform` would refuse never reaches an output.
    if (xform.options & TJXOPT_CROP) != 0 {
        w = if xform.r.w > 0 {
            xform.r.w
        } else {
            (w - xform.r.x.max(0)).max(1)
        };
        h = if xform.r.h > 0 {
            xform.r.h
        } else {
            (h - xform.r.y.max(0)).max(1)
        };
    }
    (w, h, subsamp)
}

/// `getTransformedSpecs`' crop checks (`turbojpeg.c:2847-2869`), in its
/// order, against the transformed frame: a negative field, an unknown
/// destination subsampling, an origin off the destination iMCU grid, and a
/// region — a zero extent running to the edge — past the frame. `None` when
/// the transform crops nothing or the region is valid.
fn crop_spec_refusal(
    width: c_int,
    height: c_int,
    subsamp: c_int,
    xform: &TjTransform,
) -> Option<String> {
    if (xform.options & TJXOPT_CROP) == 0 {
        return None;
    }
    let r = &xform.r;
    if r.x < 0 || r.y < 0 || r.w < 0 || r.h < 0 {
        return Some(String::from(
            "tj3TransformBufSize(): Invalid cropping region",
        ));
    }
    // The destination geometry without the crop applied.
    let uncropped: TjTransform = TjTransform {
        options: xform.options & !TJXOPT_CROP,
        ..*xform
    };
    let (dst_width, dst_height, dst_subsamp) =
        transformed_specs(width, height, subsamp, &uncropped);
    let Some((&mcu_width, &mcu_height)) = usize::try_from(dst_subsamp)
        .ok()
        .and_then(|index| TJ_MCU_WIDTH.get(index).zip(TJ_MCU_HEIGHT.get(index)))
    else {
        return Some(String::from(
            "tj3TransformBufSize(): Could not determine subsampling level of JPEG image",
        ));
    };
    if !(r.x as usize).is_multiple_of(mcu_width) || !(r.y as usize).is_multiple_of(mcu_height) {
        return Some(format!(
            "tj3TransformBufSize(): To crop this JPEG image, x must be a multiple of \
             {mcu_width}\nand y must be a multiple of {mcu_height}."
        ));
    }
    let exceeds: String = String::from(
        "tj3TransformBufSize(): The cropping region exceeds the destination image dimensions",
    );
    if r.x >= dst_width || r.y >= dst_height {
        return Some(exceeds);
    }
    let cropped_width: i64 = if r.w == 0 {
        i64::from(dst_width - r.x)
    } else {
        i64::from(r.w)
    };
    let cropped_height: i64 = if r.h == 0 {
        i64::from(dst_height - r.y)
    } else {
        i64::from(r.h)
    };
    if i64::from(r.x) + cropped_width > i64::from(dst_width)
        || i64::from(r.y) + cropped_height > i64::from(dst_height)
    {
        return Some(exceeds);
    }
    None
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
            if let Some(refusal) = crop_spec_refusal(w, h, subsamp, xform) {
                inst.set_error(refusal, TJERR_FATAL);
                return 0;
            }
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
