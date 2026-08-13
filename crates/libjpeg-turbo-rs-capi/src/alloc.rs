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

std::thread_local! {
    /// P4-120 (#467): countdown to a forced NULL, armed only by tests.
    static FAIL_COUNTDOWN: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

/// Test hook (P4-120, #467): make the `countdown`-th subsequent
/// [`libc_malloc`] on this thread return NULL, exactly as a real allocator
/// under memory pressure would — `0` fails the very next allocation.
/// Disarms itself once it fires. Everything the shim allocates for the
/// classic dest managers funnels through `libc_malloc`, so without this the
/// `JERR_OUT_OF_MEMORY` paths (`jdatadst.c`'s `ERREXIT1(…, 10)` twins) were
/// unreachable from any test. Not part of the C ABI: the symbol is not
/// `extern "C"` and is not exported from the cdylib.
pub fn fail_nth_allocation_for_tests(countdown: u32) {
    FAIL_COUNTDOWN.with(|c| c.set(Some(countdown)));
}

/// Disarm [`fail_nth_allocation_for_tests`] (idempotent).
pub fn disarm_allocation_failure_for_tests() {
    FAIL_COUNTDOWN.with(|c| c.set(None));
}

/// Allocate `size` bytes through libc `malloc`. Returns NULL on OOM or on
/// `size == 0` (matching TurboJPEG behavior — callers treat NULL as
/// either OOM or "nothing to allocate" and don't distinguish).
pub(crate) fn libc_malloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    // P4-120 (#467): the injected failure, indistinguishable to callers
    // from a real NULL. One thread-local read on the production path.
    let inject: bool = FAIL_COUNTDOWN.with(|c| match c.get() {
        Some(0) => {
            c.set(None);
            true
        }
        Some(n) => {
            c.set(Some(n - 1));
            false
        }
        None => false,
    });
    if inject {
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

/// What happened when handing a compressed result to a caller's out-pair.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OutputDelivery {
    /// Written. `*jpeg_buf` / `*jpeg_size` now describe the result.
    Delivered,
    /// `TJPARAM_NOREALLOC` is set and the caller's buffer cannot hold the
    /// output. Nothing was written and nothing was freed.
    BufferTooSmall { needed: usize, capacity: usize },
    /// `TJPARAM_NOREALLOC` is set and the caller supplied no buffer at all.
    ///
    /// Upstream treats this as the same refusal rather than a licence to
    /// allocate: `jdatadst-tj.c:184-192` takes the `*outbuffer == NULL` branch
    /// and, with `alloc` false, raises `JERR_BUFFER_SIZE`. The flag is a
    /// request *not to allocate*, so honouring it half-way — refusing to grow
    /// a buffer but conjuring one when none was given — is the one behaviour
    /// no caller asked for.
    NoBufferSupplied,
    /// The replacement allocation failed. The caller's slot is untouched.
    OutOfMemory,
}

/// Hand `jpeg` to a TurboJPEG `(jpeg_buf, jpeg_size)` out-pair, honouring
/// `TJPARAM_NOREALLOC`.
///
/// Every compressing entry point needs exactly this, and until P4-145 only
/// `tj3Compress8` did it. The other five allocated a fresh buffer and
/// `free()`d the previous pointee unconditionally — including when the caller
/// had set `NOREALLOC`, which is precisely the flag it sets when the buffer is
/// *not* `malloc`-owned. Handing a stack array or a `Vec`'s buffer to those
/// entry points was therefore a free with the wrong allocator, and the caller
/// was doing what upstream permits.
///
/// The two paths, per upstream (`turbojpeg.c` + `jdatadst-tj.c`):
///
/// - **`NOREALLOC` set** — write in place. `*jpeg_size` is an *input* carrying
///   the buffer's capacity; too small is `JERR_BUFFER_SIZE`, not a resize
///   (`jdatadst-tj.c:92`). The caller keeps its pointer, so the slot is
///   neither swapped nor freed. A **NULL** slot is the same refusal, not a
///   licence to allocate (`jdatadst-tj.c:184-192`).
/// - **Otherwise** — allocate, store, and free the previous pointee. Upstream
///   reaches the same state by `realloc`, which also consumes it.
///
/// Six call sites share this rather than six copies; the `compress_*` family
/// that P4-40 was filed for is what that duplication turns into.
///
/// # Safety
///
/// `jpeg_buf` and `jpeg_size` must be non-NULL and valid for read and write.
/// When `norealloc` is set and `*jpeg_buf` is non-NULL, it must be valid for
/// writes of `*jpeg_size` bytes and must not alias `jpeg`. On the other path a
/// non-NULL `*jpeg_buf` must have come from `malloc` / `tj3Alloc`.
pub(crate) unsafe fn deliver_compressed_output(
    jpeg: &[u8],
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
    norealloc: bool,
) -> OutputDelivery {
    // SAFETY: the caller guarantees both out-pointers are valid for access.
    let prior: *mut u8 = unsafe { *jpeg_buf };

    if norealloc {
        if prior.is_null() {
            return OutputDelivery::NoBufferSupplied;
        }
        // SAFETY: as above; under NOREALLOC this slot is an input.
        let capacity: usize = unsafe { *jpeg_size };
        if jpeg.len() > capacity {
            return OutputDelivery::BufferTooSmall {
                needed: jpeg.len(),
                capacity,
            };
        }
        // SAFETY: `capacity >= jpeg.len()` was just checked rather than
        // assumed, and the caller guarantees the buffer does not alias `jpeg`.
        unsafe {
            std::ptr::copy_nonoverlapping(jpeg.as_ptr(), prior, jpeg.len());
            *jpeg_size = jpeg.len();
        }
        return OutputDelivery::Delivered;
    }

    let fresh: *mut u8 = libc_from_slice(jpeg);
    if fresh.is_null() && !jpeg.is_empty() {
        return OutputDelivery::OutOfMemory;
    }
    // SAFETY: `prior` came from this same allocator per the documented
    // ownership contract, and is released only now that its replacement
    // exists — an early free would strand the caller's slot on the OOM path
    // above.
    libc_free(prior);
    // SAFETY: caller guarantees both out-pointers are writable.
    unsafe {
        *jpeg_buf = fresh;
        *jpeg_size = jpeg.len();
    }
    OutputDelivery::Delivered
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
///
/// # Safety
///
/// `ptr` must be NULL, or a pointer returned by [`tj3Alloc`] (or the same
/// allocator's `malloc`) that has not already been freed. It is handed
/// straight to `free`.
///
/// A stack address, an interior pointer, one from a different allocator, or
/// one already freed is undefined behaviour. Nothing here can detect any of
/// those — a raw pointer carries no provenance the callee can check — which is
/// why this is `unsafe fn` rather than a safe one that implies otherwise
/// (P4-137).
///
/// The obligation is the C caller's either way; `unsafe` changes nothing for
/// them. It exists so *Rust* callers of this crate's `rlib` cannot reach
/// `free` without acknowledging it.
#[no_mangle]
pub unsafe extern "C" fn tj3Free(ptr: *mut c_void) {
    crate::unwind_guard!((), {
        libc_free(ptr as *mut u8);
    })
}

/// Silence "unused" warnings on `c_int` without forcing a re-export.
#[allow(dead_code)]
const _ALLOC_INT_MARKER: c_int = 0;
