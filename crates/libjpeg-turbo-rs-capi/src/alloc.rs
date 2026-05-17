//! libc-compatible allocation helpers used by the C ABI shim.
//!
//! libjpeg-turbo's `tj3Alloc`/`tj3Free` are thin wrappers around
//! `malloc`/`free`; C callers routinely mix them with their own `free()`
//! (e.g. Pillow does exactly that for the `jpegBuf` returned from
//! `tj3Compress8`). We therefore MUST emit output buffers via the system
//! allocator, not Rust's global allocator. Linking libc directly keeps
//! the two pools coherent on every platform we target.
//!
//! All sizes are in bytes; NULL return signals out-of-memory.

use std::ffi::c_void;
use std::os::raw::c_int;

extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn memcpy(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;
}

/// Allocate `size` bytes through libc `malloc`. Returns NULL on OOM or on
/// `size == 0` (matching TurboJPEG behavior — callers treat NULL as
/// either OOM or "nothing to allocate" and don't distinguish).
pub(crate) fn libc_malloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: `malloc` is a standard C function; we check for NULL before
    // use and ensure the same allocator is used for free().
    let p: *mut c_void = unsafe { malloc(size) };
    p as *mut u8
}

/// Free a pointer previously returned by `libc_malloc`. NULL is a no-op.
pub(crate) fn libc_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: pointer came from `malloc` (via `libc_malloc`).
    unsafe { free(ptr as *mut c_void) }
}

/// Copy `len` bytes from a Rust slice into a fresh libc-allocated buffer.
/// Returns `(ptr, len)`; `ptr` is NULL only on OOM.
pub(crate) fn libc_from_slice(data: &[u8]) -> *mut u8 {
    let p: *mut u8 = libc_malloc(data.len());
    if p.is_null() {
        return p;
    }
    // SAFETY: `p` points to `data.len()` bytes of freshly allocated memory
    // and does not alias the source slice.
    unsafe {
        memcpy(p as *mut c_void, data.as_ptr() as *const c_void, data.len());
    }
    p
}

// ---------------------------------------------------------------------------
// Public extern "C" wrappers (A1-8 scope, but needed here for tj3Compress8 to
// hand out a C-freeable buffer).
// ---------------------------------------------------------------------------

/// `tj3Alloc(bytes)` — libjpeg-turbo-compatible allocator returning a
/// buffer that can be released with `tj3Free` or `free`.
#[no_mangle]
pub extern "C" fn tj3Alloc(bytes: usize) -> *mut c_void {
    crate::unwind_guard!(std::ptr::null_mut(), { libc_malloc(bytes) as *mut c_void })
}

/// `tj3Free(ptr)` — libjpeg-turbo-compatible deallocator. NULL is a no-op.
#[no_mangle]
pub extern "C" fn tj3Free(ptr: *mut c_void) {
    crate::unwind_guard!((), {
        libc_free(ptr as *mut u8);
    })
}

/// Silence "unused" warnings on `c_int` without forcing a re-export.
#[allow(dead_code)]
const _ALLOC_INT_MARKER: c_int = 0;
