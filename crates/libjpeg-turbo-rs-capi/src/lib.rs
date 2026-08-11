//! C ABI shim for `libjpeg-turbo-rs`.
//!
//! Exposes libjpeg-turbo-compatible `extern "C"` symbols for C consumers
//! (`djpeg`, `cjpeg`, `jpegtran`, Pillow, ImageMagick, …).
//!
//! # Which ABI this targets
//!
//! **TurboJPEG 3 (`libturbojpeg.so.0`) is the primary target.** The classic
//! libjpeg API is offered at the **v8 identity only** (`libjpeg.so.8`,
//! `JPEG_LIB_VERSION 80`), and that leg is experimental and partial — see
//! `docs/ABI_COMPATIBILITY.md` for what it does and does not cover.
//!
//! **libjpeg v6b (`libjpeg.so.62`) and v7 are explicit non-goals**, and this
//! crate must not be substituted for them. The struct layouts differ: a
//! consumer compiled against v6b addresses `jpeg_decompress_struct` fields at
//! v6b offsets while this library lays them out for v8, so every access lands
//! at the wrong offset. That is memory corruption, not a missing feature. The
//! build and install paths reject those identities deliberately (P4-140).
//!
//! # Safety status
//!
//! The safe-Rust / unsafe-SIMD boundary in the underlying crate is under
//! audit, so this crate carries no memory-safety guarantee today. See the open
//! soundness items (P4-135..P4-139) before depending on one.
//!
//! The public Rust surface re-exports the underlying pure-Rust
//! `libjpeg_turbo_rs` crate under `inner` so that downstream Rust
//! consumers can mix-and-match the C and Rust APIs.
//!
//! # Pointer contract
//!
//! Every exported entry point that takes a raw pointer is declared
//! `pub unsafe extern "C" fn`. This crate builds as an `rlib` as well as a
//! `cdylib`, so those signatures are a real Rust API: a safe `fn` taking
//! `*mut c_void` would let safe Rust hand us a dangling pointer, and
//! `tj3Destroy`/`tj3Free` would turn that into a double free or a `free()` of
//! an arbitrary address with no `unsafe` block anywhere (P4-137, #476).
//!
//! Marking them `unsafe` changes nothing about the C ABI. `extern "C"` fixes
//! the calling convention and `#[no_mangle]` fixes the symbol name; `unsafe`
//! adds an obligation for *Rust* callers only. Converting all 132
//! pointer-taking exports left `nm -gU` byte-identical at 157 symbols.
//!
//! Each function's `# Safety` section names its own pointers. The obligations
//! below are common to all of them:
//!
//! - **Validity.** Every pointer is either null — where the function documents
//!   that it rejects null with an error code — or points to a live allocation
//!   for the whole call. We cannot check this, and do not try.
//! - **Retained pointers outlive the call.** A few entry points *store* the
//!   pointer instead of only reading through it, and for those "valid for the
//!   whole call" is far too weak. `jpeg_mem_src` keeps the input buffer and
//!   reads it during every later `jpeg_read_header` / `jpeg_read_scanlines`;
//!   `jpeg_mem_dest` keeps the caller's output slot, and `jpeg_stdio_src` /
//!   `jpeg_stdio_dest` keep the `FILE *`. Each must stay valid, and must not be
//!   aliased or moved, until it is replaced by another call to the same setter
//!   or the `cinfo` is finished, aborted, or destroyed. Dropping a `Vec` after
//!   `jpeg_mem_src` returns and then calling `jpeg_read_header` is a
//!   use-after-free, and nothing in the signature says otherwise.
//! - **Size.** A buffer pointer must be valid for at least the number of bytes
//!   the call reads or writes, as computed from the arguments and the image
//!   header (pitch, height, subsampling, precision). Where the API has a
//!   `*BufSize` helper, that helper is the definition of "enough".
//! - **Alignment.** Pointers must be aligned for the element type. Byte buffers
//!   need no alignment; `*mut c_int` and struct pointers need natural
//!   alignment, and a misaligned struct pointer is UB even if it is readable.
//! - **Ownership transfer.** `tj3Destroy` and `tj3Free` take ownership, but
//!   what they accept differs and conflating them is a real hazard:
//!   - `tj3Destroy` requires a handle **this library returned** from
//!     `tj3Init`. Nothing else is valid, and passing one twice is a double
//!     free.
//!   - `tj3Free` requires memory from **the same libc allocator** — `tj3Alloc`
//!     or a plain `malloc` from the same runtime. It is handed straight to
//!     `free`, so it accepts more than just this library's own allocations,
//!     and correspondingly rejects a stack address, an interior pointer, or
//!     anything from Rust's allocator.
//!
//!   The same rule reaches further than those two functions, and the reach is
//!   easy to miss: the **reusable output slot** is freed by this library. On
//!   `tj3Compress12`, `tj3Compress16`, `tj3CompressFromYUV8` and
//!   `tj3CompressFromYUVPlanes8` that slot is `*jpeg_buf`; on `tj3Transform` it
//!   is each `*dst_bufs[i]` (its `jpeg_buf` is the const *source* and is never
//!   freed). Each is passed to `free()` before a new pointer is stored, so a
//!   non-null slot must have come from `tj3Alloc` (or plain `malloc`) — a stack
//!   array, a `Vec`'s buffer, or anything from Rust's allocator satisfies every
//!   validity, size and alignment rule above and is *still* undefined behaviour
//!   to hand over, because it is freed with the wrong allocator.
//!
//!   **`TJPARAM_NOREALLOC` does not protect you on those five.** Only
//!   `tj3Compress8` consults it (writing in place, and erroring when the buffer
//!   is too small). The other five ignore it and replace the buffer
//!   unconditionally, which diverges from upstream TurboJPEG — tracked as
//!   P4-145. Until that closes, treat a caller-owned output slot on those
//!   entry points as `malloc`-owned memory you are giving away.
//! - **Aliasing.** For the duration of a call, no other live reference may
//!   alias the pointed-to memory. Source and destination buffers must not
//!   overlap unless a function documents otherwise.
//! - **Threading.** A single `tjhandle` or `j_common_ptr` must not be used
//!   concurrently from two threads. Distinct instances are independent. The
//!   classic `jpeg_*` state additionally carries a per-`cinfo` thread-affinity
//!   constraint tracked as P4-132.
//!
//! ## What this crate does and does not guarantee
//!
//! Stated plainly, because the distinction is easy to blur (P4-137
//! criterion 7):
//!
//! - **Not guaranteed:** defence against invalid pointers from C. A caller that
//!   passes a dangling, misaligned, too-small, or already-freed pointer gets
//!   undefined behaviour. This is not a gap to be closed — it is inherent to
//!   the C ABI, and C libjpeg-turbo makes the same bargain.
//! - **Guaranteed (the goal, and what the fuzz and sanitizer gates test):** a
//!   *malformed or hostile JPEG* cannot corrupt memory when the caller honours
//!   the pointer contract above. Bad **data** is our problem; bad **pointers**
//!   are the caller's.

#![deny(unsafe_op_in_unsafe_fn)]
// `clippy::not_unsafe_ptr_arg_deref` was suppressed crate-wide here until
// 2026-08-11. The suppression justified itself with "marking each function
// `unsafe fn` would change the ABI-visible symbol name and break the drop-in
// contract". **That was false, and it is now measured twice.** `extern "C"`
// fixes the calling convention and `#[no_mangle]` fixes the symbol name;
// `unsafe` adds an obligation for *Rust* callers only. Converting all 132
// pointer-taking exports left `nm -gU` byte-identical at 157 symbols
// (P4-137 criterion 1, #476).
//
// The lint is now enforced, so a new export that dereferences a raw pointer
// from a safe `fn` fails the build rather than being waved through.
// Exported symbols must match the C case (`tj3Init`, `tj3Destroy`, ...),
// so we disable the snake_case lint at the crate root.
#![allow(non_snake_case)]

pub use libjpeg_turbo_rs as inner;

// ---------------------------------------------------------------------------
// P4-4: panic guard for every `pub extern "C"` entry point.
//
// A Rust `panic!` that crosses an `extern "C"` boundary is undefined
// behaviour on every target we ship. This macro funnels any panic
// caught in an FFI body into a documented C-style sentinel return
// value (and a one-line stderr message). Every `pub extern "C" fn` in
// the capi crate should wrap its body with `crate::unwind_guard!`.
//
// Usage:
//
//     #[no_mangle]
//     pub extern "C" fn foo(...) -> c_int {
//         crate::unwind_guard!(-1, {
//             // original body, may panic
//         })
//     }
//
// For `()` return:
//
//     pub extern "C" fn bar(...) {
//         crate::unwind_guard!((), { ... })
//     }
//
// We do NOT set `[profile.release] panic = "abort"`: profile-`panic`
// can only be customised at workspace root in stable Cargo, and the
// main Rust crate (`libjpeg_turbo_rs`) keeps the default unwind
// strategy so its `Result<…, JpegError>` callers can recover normally.
// The `catch_unwind` guard below is sufficient to keep panics on the
// Rust side of the FFI boundary.
//
// `#[macro_export]` is the only way to publish a `macro_rules!` macro
// across submodules without `#[macro_use]`; it is intentionally hidden
// from rustdoc because this is a crate-internal helper.
#[macro_export]
#[doc(hidden)]
macro_rules! unwind_guard {
    ($sentinel:expr, $body:block) => {{
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $body)) {
            Ok(__value) => __value,
            Err(__payload) => {
                let __msg: ::std::string::String =
                    if let Some(s) = __payload.downcast_ref::<&'static str>() {
                        (*s).to_string()
                    } else if let Some(s) = __payload.downcast_ref::<::std::string::String>() {
                        s.clone()
                    } else {
                        ::std::string::String::from("<non-string panic payload>")
                    };
                eprintln!(
                    "libjpeg-turbo-rs-capi: panic caught at FFI boundary in `{}`: {}",
                    ::std::module_path!(),
                    __msg
                );
                $sentinel
            }
        }
    }};
}

pub mod alloc;
pub mod bufsize;
pub mod compress;
pub mod convert;
pub mod decompress;
pub mod header;
pub mod imageio;
pub mod jpeglib;
pub mod legacy;
pub mod memmgr;
/// Crate-private: the table is an implementation detail of `jpeg_std_error`,
/// and exporting it would invite consumers to index it directly.
pub(crate) mod message_table;
pub mod mozjpeg_compat;
pub mod precision;
pub mod tj3;
pub mod transform;
pub mod yuv;

// Re-export the `extern "C"` symbols at the crate root for discoverability.
pub use alloc::{tj3Alloc, tj3Free};
pub use bufsize::{
    tj3GetScalingFactors, tj3JPEGBufSize, tj3YUVBufSize, tj3YUVPlaneHeight, tj3YUVPlaneSize,
    tj3YUVPlaneWidth,
};
pub use compress::tj3Compress8;
pub use decompress::tj3Decompress8;
pub use header::{
    tj3DecompressHeader, tj3SetCroppingRegion, tj3SetScalingFactor, TjRegion, TjScalingFactor,
};
pub use imageio::{
    tj3LoadImage12, tj3LoadImage16, tj3LoadImage8, tj3SaveImage12, tj3SaveImage16, tj3SaveImage8,
};
pub use precision::{tj3Compress12, tj3Compress16, tj3Decompress12, tj3Decompress16};
pub use tj3::{
    tj3Destroy, tj3Get, tj3GetErrorCode, tj3GetErrorStr, tj3GetICCProfile, tj3Init, tj3InitVersion,
    tj3Set, tj3SetICCProfile,
};
pub use transform::{tj3Transform, tj3TransformBufSize, TjTransform};
pub use yuv::{
    tj3CompressFromYUV8, tj3CompressFromYUVPlanes8, tj3DecodeYUV8, tj3DecodeYUVPlanes8,
    tj3DecompressToYUV8, tj3DecompressToYUVPlanes8, tj3EncodeYUV8, tj3EncodeYUVPlanes8,
};

// Classic libjpeg-style `jpeg_*` decode entry points (FFI A1-11 / C1-1..3).
pub use jpeglib::{
    jpeg12_crop_scanline, jpeg12_read_scanlines, jpeg12_skip_scanlines, jpeg16_read_scanlines,
    jpeg_CreateDecompress, jpeg_copy_critical_parameters, jpeg_core_output_dimensions,
    jpeg_crop_scanline, jpeg_destroy_decompress, jpeg_finish_decompress, jpeg_mem_src,
    jpeg_read_coefficients, jpeg_read_header, jpeg_read_icc_profile, jpeg_read_scanlines,
    jpeg_save_markers, jpeg_set_marker_processor, jpeg_skip_scanlines, jpeg_start_decompress,
    jpeg_std_error, jpeg_stdio_src, JpegDecompressPublic, JpegErrorMgr, JpegSourceMgr,
    JPEG_HEADER_OK, JPEG_HEADER_TABLES_ONLY, JPEG_SUSPENDED,
};

// Classic libjpeg-style `jpeg_*` encode entry points (FFI C2-*).
pub use jpeglib::{
    jcopy_block_row, jdiv_round_up, jpeg12_write_scanlines, jpeg16_write_scanlines,
    jpeg_CreateCompress, jpeg_add_quant_table, jpeg_calc_jpeg_dimensions, jpeg_create_compress,
    jpeg_default_colorspace, jpeg_default_qtables, jpeg_destroy_compress, jpeg_enable_lossless,
    jpeg_finish_compress, jpeg_mem_dest, jpeg_quality_scaling, jpeg_resync_to_restart,
    jpeg_set_colorspace, jpeg_set_defaults, jpeg_set_quality, jpeg_simple_progression,
    jpeg_start_compress, jpeg_stdio_dest, jpeg_suppress_tables, jpeg_write_coefficients,
    jpeg_write_icc_profile, jpeg_write_m_byte, jpeg_write_m_header, jpeg_write_marker,
    jpeg_write_scanlines, jpeg_write_tables, JpegComponentInfoCompress, JpegCompressPublic,
    JpegDestinationMgr,
};

// mozjpeg parameter-API stubs — let consumers linked against mozjpeg
// (Homebrew libvips, Pillow-mozjpeg, several Linux distros) dyld-resolve
// against our cdylib. All probes return FALSE so the consumer falls
// through to the standard libjpeg encode path we implement.
pub use mozjpeg_compat::{
    jpeg_c_bool_param_supported, jpeg_c_float_param_supported, jpeg_c_get_bool_param,
    jpeg_c_get_float_param, jpeg_c_get_int_param, jpeg_c_int_param_supported,
    jpeg_c_set_bool_param, jpeg_c_set_float_param, jpeg_c_set_int_param,
};

// Legacy TJ1/TJ2 aliases — thin wrappers around the TJ3 surface above.
pub use legacy::{
    tjBufSize, tjBufSizeYUV, tjBufSizeYUV2, tjCompress2, tjDecodeYUV, tjDecompress2,
    tjDecompressHeader3, tjDestroy, tjEncodeYUV3, tjGetErrorStr2, tjInitCompress, tjInitDecompress,
    tjInitTransform, tjLoadImage, tjPlaneHeight, tjPlaneSizeYUV, tjPlaneWidth, tjSaveImage,
    tjTransform, TJBUFSIZE, TJBUFSIZEYUV,
};
