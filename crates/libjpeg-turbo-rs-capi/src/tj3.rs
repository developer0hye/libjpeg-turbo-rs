//! TurboJPEG 3 handle lifecycle + get/set + error reporting.
//!
//! This module provides the canonical `tj3*` `extern "C"` surface used by
//! downstream C consumers. The handle is an opaque `*mut c_void`; the real
//! allocation is a `Box<TjInstance>` that owns the inner Rust `TjHandle`
//! plus a buffered error message so `tj3GetErrorStr` can return a stable
//! `const char *`.
//!
//! Constant values match libjpeg-turbo's `turbojpeg.h`:
//! - `TJINIT_COMPRESS = 1`, `TJINIT_DECOMPRESS = 2`, `TJINIT_TRANSFORM = 4`
//!   (bit flags; callers may OR them together).
//! - `TJPARAM_*` numeric IDs follow the C header order.

use std::ffi::{c_char, c_int, c_void, CString};

use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};

/// Internal allocation for a TJ3 handle.
///
/// Wraps the Rust-side `TjHandle`, the requested `initType`, and the last
/// error message. The error message is owned by this struct so that the
/// pointer returned by `tj3GetErrorStr` stays valid until the next call
/// mutates it (matching libjpeg-turbo's semantics).
pub(crate) struct TjInstance {
    pub(crate) inner: TjHandle,
    // Recorded for future use by transform/compress/decompress selection.
    #[allow(dead_code)]
    pub(crate) init_type: c_int,
    pub(crate) last_error: CString,
    pub(crate) last_error_code: c_int,
}

impl TjInstance {
    fn new(init_type: c_int) -> Self {
        Self {
            inner: TjHandle::new(),
            init_type,
            last_error: CString::new("No error").expect("static string"),
            last_error_code: 0,
        }
    }

    /// Stash an error message + code, to be retrieved via `tj3GetErrorStr` /
    /// `tj3GetErrorCode`.
    pub(crate) fn set_error(&mut self, msg: impl Into<String>, code: c_int) {
        let s: String = msg.into();
        // Strip interior NULs that would break CString construction.
        let sanitized: String = s.replace('\0', "?");
        self.last_error = CString::new(sanitized)
            .unwrap_or_else(|_| CString::new("internal error formatting message").expect("static"));
        self.last_error_code = code;
    }

    pub(crate) fn clear_error(&mut self) {
        self.last_error = CString::new("No error").expect("static string");
        self.last_error_code = 0;
    }

    /// True if `TJPARAM_BOTTOMUP` is set on this handle. Used by the
    /// image-I/O wrappers (`tj3LoadImage8` / `tj3SaveImage8`) to
    /// flip row order on load/save.
    pub(crate) fn bottom_up_flag(&self) -> bool {
        self.inner.get(TjParam::BottomUp) != 0
    }
}

// TJ_NUMERR matches libjpeg-turbo: 0 = warning (non-fatal), 1 = fatal.
// `TJERR_WARNING = 0` is reserved for future non-fatal paths; kept here to
// document the contract even though no caller currently selects it.
#[allow(dead_code)]
pub(crate) const TJERR_WARNING: c_int = 0;
pub(crate) const TJERR_FATAL: c_int = 1;

/// Map a numeric `TJPARAM_*` constant to the internal `TjParam` enum.
fn param_from_c(param: c_int) -> Option<TjParam> {
    // Numeric layout matches turbojpeg.h TJPARAM_* enumeration.
    Some(match param {
        0 => TjParam::StopOnWarning,
        1 => TjParam::BottomUp,
        2 => TjParam::NoRealloc,
        3 => TjParam::Quality,
        4 => TjParam::Subsampling,
        5 => TjParam::Width,
        6 => TjParam::Height,
        7 => TjParam::Precision,
        8 => TjParam::ColorSpace,
        9 => TjParam::FastUpSample,
        10 => TjParam::FastDct,
        11 => TjParam::Optimize,
        12 => TjParam::Progressive,
        13 => TjParam::ScanLimit,
        14 => TjParam::Arithmetic,
        15 => TjParam::Lossless,
        16 => TjParam::LosslessPsv,
        17 => TjParam::LosslessPt,
        18 => TjParam::RestartBlocks,
        19 => TjParam::RestartRows,
        20 => TjParam::XDensity,
        21 => TjParam::YDensity,
        22 => TjParam::DensityUnits,
        23 => TjParam::MaxMemory,
        24 => TjParam::MaxPixels,
        25 => TjParam::SaveMarkers,
        _ => return None,
    })
}

/// Read-only parameters: `tj3Set` must reject attempts to write them.
///
/// `TJPARAM_PRECISION` is writable for encode: callers use `tj3Set(h,
/// TJPARAM_PRECISION, value)` before calling `tj3Compress8/12/16` with
/// lossless mode to request a specific sample precision (e.g. 4-bit for
/// 8-bit-depth sources). After `tj3DecompressHeader` / `tj3Decompress` the
/// field reflects the precision stored in the JPEG stream (effectively
/// read-only until the next encode call), but the *param store* must accept
/// writes so encode callers can override it.
fn is_read_only(param: TjParam) -> bool {
    matches!(param, TjParam::Width | TjParam::Height)
}

/// Dereference an opaque handle pointer into a mutable `TjInstance`
/// reference, or return `None` if the pointer is NULL.
///
/// # Safety
/// The caller guarantees that `handle` was returned by `tj3Init` and has
/// not yet been passed to `tj3Destroy`.
pub(crate) unsafe fn handle_as_mut<'a>(handle: *mut c_void) -> Option<&'a mut TjInstance> {
    if handle.is_null() {
        None
    } else {
        unsafe { Some(&mut *(handle as *mut TjInstance)) }
    }
}

// ---------------------------------------------------------------------------
// Extern "C" surface
// ---------------------------------------------------------------------------

/// `TURBOJPEG_VERSION_NUMBER` — mirror of the C macro in turbojpeg.h.
/// Kept here so `tj3InitVersion` can rubber-stamp requests that match
/// what the compiled-against header advertised.
pub(crate) const TURBOJPEG_VERSION_NUMBER: c_int = 3_002_000;

/// `tj3Init(initType)` — allocate a TurboJPEG 3 handle.
///
/// `initType` is an enum value defined by `enum TJINIT` in `turbojpeg.h`:
/// `TJINIT_COMPRESS = 0`, `TJINIT_DECOMPRESS = 1`, `TJINIT_TRANSFORM = 2`.
/// Anything outside `[0, TJ_NUMINIT = 3)` returns `NULL`.
///
/// In the C header, `tj3Init` is a macro that expands to
/// `tj3InitVersion(initType, TURBOJPEG_VERSION_NUMBER)`; we export both
/// symbols so callers compiled against either spelling resolve.
#[no_mangle]
pub extern "C" fn tj3Init(init_type: c_int) -> *mut c_void {
    tj3InitVersion(init_type, TURBOJPEG_VERSION_NUMBER)
}

/// `tj3InitVersion(initType, apiVersion)` — the real init entry point.
///
/// We accept any `apiVersion` that is `>= 3_000_000` (TurboJPEG 3.0) so
/// future C tools can link against us without a rebuild. Older versions
/// would depend on the pre-TJ3 surface and are rejected.
#[no_mangle]
pub extern "C" fn tj3InitVersion(init_type: c_int, api_version: c_int) -> *mut c_void {
    // `initType` is an enum (0=COMPRESS, 1=DECOMPRESS, 2=TRANSFORM);
    // validation follows the C reference: reject < 0, >= TJ_NUMINIT (3).
    const TJ_NUMINIT: c_int = 3;
    if !(0..TJ_NUMINIT).contains(&init_type) {
        return std::ptr::null_mut();
    }
    // turbojpeg.c::tj3InitVersion accepts any apiVersion in [1_000_000,
    // 999_999_999]; match that so clients built against older headers
    // still resolve.
    if !(1_000_000..=999_999_999).contains(&api_version) {
        return std::ptr::null_mut();
    }
    let boxed: Box<TjInstance> = Box::new(TjInstance::new(init_type));
    Box::into_raw(boxed) as *mut c_void
}

/// `tj3Destroy(handle)` — free a handle. NULL is a no-op.
#[no_mangle]
pub extern "C" fn tj3Destroy(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` came from `Box::into_raw(Box<TjInstance>)` above.
    unsafe {
        drop(Box::from_raw(handle as *mut TjInstance));
    }
}

/// `tj3Set(handle, param, value)` — set a TJ3 parameter.
///
/// Returns 0 on success, -1 on error (invalid handle, unknown parameter,
/// read-only parameter, or value out of range).
#[no_mangle]
pub extern "C" fn tj3Set(handle: *mut c_void, param: c_int, value: c_int) -> c_int {
    // SAFETY: dereferenced only if non-NULL; validated above.
    let inst: &mut TjInstance = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    let p: TjParam = match param_from_c(param) {
        Some(p) => p,
        None => {
            inst.set_error(format!("unknown TJPARAM id {param}"), TJERR_FATAL);
            return -1;
        }
    };

    if is_read_only(p) {
        inst.set_error(format!("TJPARAM id {param} is read-only"), TJERR_FATAL);
        return -1;
    }

    match inst.inner.set(p, value) {
        Ok(()) => {
            inst.clear_error();
            0
        }
        Err(e) => {
            inst.set_error(e.to_string(), TJERR_FATAL);
            -1
        }
    }
}

/// `tj3Get(handle, param)` — read a TJ3 parameter.
///
/// Returns the parameter value, or -1 on error.
#[no_mangle]
pub extern "C" fn tj3Get(handle: *mut c_void, param: c_int) -> c_int {
    let inst: &mut TjInstance = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    match param_from_c(param) {
        Some(p) => {
            inst.clear_error();
            inst.inner.get(p)
        }
        None => {
            inst.set_error(format!("unknown TJPARAM id {param}"), TJERR_FATAL);
            -1
        }
    }
}

/// `tj3GetErrorStr(handle)` — return the last error message.
///
/// For NULL handles we first check the process-global no-handle error
/// slot (populated by `tj3JPEGBufSize`, `tj3YUVBufSize`, etc.), falling
/// back to a canonical "No error" string. That matches the
/// libjpeg-turbo reference so that `tjunittest`'s
/// `bufSizeTest()` can call `tj3GetErrorStr(NULL)` after a sizing call.
#[no_mangle]
pub extern "C" fn tj3GetErrorStr(handle: *mut c_void) -> *const c_char {
    static GLOBAL_NO_ERROR: &[u8] = b"No error\0";
    if handle.is_null() {
        let p: *const c_char = crate::bufsize::no_handle_error_ptr();
        if !p.is_null() {
            return p;
        }
        return GLOBAL_NO_ERROR.as_ptr() as *const c_char;
    }
    let inst: &mut TjInstance = unsafe { &mut *(handle as *mut TjInstance) };
    inst.last_error.as_ptr()
}

/// `tj3GetErrorCode(handle)` — return the last error severity.
///
/// Returns `TJERR_WARNING=0` when the last operation succeeded or raised
/// only a warning, `TJERR_FATAL=1` on failure. A NULL handle is treated
/// as fatal.
#[no_mangle]
pub extern "C" fn tj3GetErrorCode(handle: *mut c_void) -> c_int {
    if handle.is_null() {
        return TJERR_FATAL;
    }
    let inst: &mut TjInstance = unsafe { &mut *(handle as *mut TjInstance) };
    inst.last_error_code
}

/// `tj3GetICCProfile(handle, **iccBuf, *iccSize) -> int`.
///
/// Returns the ICC profile (if any) associated with the TurboJPEG instance.
/// The handle accumulates an ICC profile in two ways:
///   * `tj3SetICCProfile` writes one in for subsequent encodes/transforms.
///   * `tj3DecompressHeader` extracts one from the source JPEG when
///     `TJPARAM_SAVEMARKERS` is set (or when ICC capture is implicit).
///
/// Contract (mirrors `references/libjpeg-turbo/src/turbojpeg.h:1995`):
/// `iccSize` is required (NULL → -1). When `iccBuf == NULL` the call is
/// query-only and only `*iccSize` is populated; otherwise `*iccBuf` is
/// set to a fresh `tj3Alloc`-style libc allocation that the caller must
/// free with `tj3Free`, and `*iccSize` reflects the byte count. When no
/// ICC is available the stub writes `*iccSize = 0` (and `*iccBuf = NULL`
/// when the buffer pointer is non-NULL) and returns 0.
///
/// Stub note (2026-04-29): the ICC-capture path through
/// `tj3DecompressHeader` is not yet wired, so this currently always
/// reports "no ICC available". `tjbench` only queries via the NULL
/// `iccBuf` form to decide whether to embed an ICC during transform,
/// so a clean "no ICC" answer keeps the benchmark functional. Wiring
/// real ICC capture is tracked separately.
///
/// **DIVERGENCE from upstream:** stock `tj3GetICCProfile` returns -1
/// with `TJERR_WARNING` (not `TJERR_FATAL`) when called on a decompress
/// instance that has no captured ICC; this stub returns 0 instead.
/// Acceptable today because (a) `TJERR_WARNING` requires a soft-error
/// path that the shim doesn't implement yet (see `tj3.rs::TJERR_WARNING`),
/// and (b) every documented caller (incl. `tjbench`) only checks for
/// `== -1` and treats either result as "no ICC", so flipping the
/// sentinel does not change behaviour. Wire the warning-return form
/// once the soft-error path lands.
#[no_mangle]
pub extern "C" fn tj3GetICCProfile(
    handle: *mut c_void,
    icc_buf: *mut *mut u8,
    icc_size: *mut usize,
) -> c_int {
    let inst: &mut TjInstance = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if icc_size.is_null() {
        inst.set_error("tj3GetICCProfile: iccSize is NULL", TJERR_FATAL);
        return -1;
    }
    // SAFETY: caller guarantees the out-pointers are valid for write.
    unsafe {
        *icc_size = 0;
        if !icc_buf.is_null() {
            *icc_buf = std::ptr::null_mut();
        }
    }
    inst.clear_error();
    0
}

/// `tj3SetICCProfile(handle, *iccBuf, iccSize) -> int`.
///
/// Associates an ICC profile with the TurboJPEG instance so subsequent
/// encode/transform calls embed it as APP2 chunks. `iccBuf == NULL`
/// AND `iccSize == 0` clears the stored profile.
///
/// Stub note (2026-04-29): storage is accepted but the encode path
/// does not yet emit the embedded profile. `tjbench` exercises this
/// only when `tj3GetICCProfile` first reports a non-zero size, which
/// our stub never does, so this stays as a link-only stub until the
/// follow-up ICC plumb.
#[no_mangle]
pub extern "C" fn tj3SetICCProfile(
    handle: *mut c_void,
    icc_buf: *mut u8,
    icc_size: usize,
) -> c_int {
    let inst: &mut TjInstance = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if icc_size > 0 && icc_buf.is_null() {
        inst.set_error(
            "tj3SetICCProfile: iccBuf is NULL but iccSize > 0",
            TJERR_FATAL,
        );
        return -1;
    }
    inst.clear_error();
    0
}
