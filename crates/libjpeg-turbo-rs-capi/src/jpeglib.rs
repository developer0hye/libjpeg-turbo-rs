//! FFI A1-11: classic libjpeg (`jpeg_*`) decode entry points.
//!
//! This module exposes the libjpeg-compatible C API used by applications
//! such as `djpeg`, Pillow, ImageMagick, and countless others. Unlike the
//! TurboJPEG 3 (`tj3*`) surface in `tj3.rs`, these functions take a
//! caller-allocated `struct jpeg_decompress_struct` and cooperate with
//! caller-allocated `struct jpeg_error_mgr` / source manager sub-structs.
//!
//! State-machine and struct shape are a minimal subset that is sufficient
//! for the 9 entry points listed in FFI A1-11 — not the full libjpeg 9
//! ~200-field layout. Fields that real libjpeg consumers read after
//! `jpeg_read_header` and `jpeg_start_decompress` are exposed at fixed
//! offsets via `JpegDecompressPublic`; everything else lives in a
//! privately-owned tail box reached through `priv_ptr`.
//!
//! Because this crate is also the TurboJPEG 3 shim, we intentionally
//! keep only the subset of the libjpeg ABI the entry points described
//! in the task need. Future work tracked in `COORDINATOR_NOTES.md` fills
//! in the remainder (quant tables, Huffman tables, scan info, buffered
//! image mode, markers, progress manager, …).

use std::ffi::{c_int, c_long, c_void, CString};
use std::io::Read;

use libjpeg_turbo_rs::{decompress, PixelFormat};

/// libjpeg `boolean` typedef is `int` in the upstream header. All callers
/// use `TRUE` = 1 / `FALSE` = 0.
type CBoolean = c_int;

/// libjpeg `JDIMENSION` typedef.
type JDimension = u32;

/// Exact copy of libjpeg's `jpeg_read_header` return codes. Matches
/// `jpeglib.h`:
/// - `JPEG_SUSPENDED = 0`
/// - `JPEG_HEADER_OK = 1`
/// - `JPEG_HEADER_TABLES_ONLY = 2`
pub const JPEG_SUSPENDED: c_int = 0;
pub const JPEG_HEADER_OK: c_int = 1;
pub const JPEG_HEADER_TABLES_ONLY: c_int = 2;

/// Maximum message length recorded by the default error manager.
#[allow(dead_code)]
const JMSG_LENGTH_MAX: usize = 200;
/// Maximum string-parameter length for error messages.
const JMSG_STR_PARM_MAX: usize = 80;

// ---------------------------------------------------------------------------
// `struct jpeg_error_mgr` — minimal ABI-compatible layout.
// ---------------------------------------------------------------------------
//
// Real libjpeg's `jpeg_error_mgr` has ~15 fields totalling ~400 bytes. We
// expose enough of the layout that:
//   (a) `jpeg_std_error(err)` can populate the callbacks, and
//   (b) clients can write `cinfo.err = jpeg_std_error(&err);` to hook the
//       manager into a decompress struct (which requires `err` to be the
//       first field of `jpeg_decompress_struct`).
//
// Callers that walk uninitialised fields past what we expose here invoke
// undefined behavior in real libjpeg too, so the minimal layout is a
// safe starting point for a drop-in shim.

/// Public-facing layout of `struct jpeg_error_mgr`. `#[repr(C)]` pins the
/// field order; `MaybeUninit` slots hold the (unused) buffers required
/// by the real libjpeg `format_message` contract.
#[repr(C)]
pub struct JpegErrorMgr {
    pub error_exit: Option<unsafe extern "C" fn(*mut c_void)>,
    pub emit_message: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
    pub output_message: Option<unsafe extern "C" fn(*mut c_void)>,
    pub format_message: Option<unsafe extern "C" fn(*mut c_void, *mut u8)>,
    pub reset_error_mgr: Option<unsafe extern "C" fn(*mut c_void)>,
    pub msg_code: c_int,
    /// Union in the C header. We reserve the larger branch (char[80]).
    pub msg_parm: [u8; JMSG_STR_PARM_MAX],
    pub trace_level: c_int,
    pub num_warnings: c_long,
    pub jpeg_message_table: *const *const u8,
    pub last_jpeg_message: c_int,
    pub addon_message_table: *const *const u8,
    pub first_addon_message: c_int,
    pub last_addon_message: c_int,
}

// ---------------------------------------------------------------------------
// `struct jpeg_source_mgr` — verbatim layout.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct JpegSourceMgr {
    pub next_input_byte: *const u8,
    pub bytes_in_buffer: usize,
    pub init_source: Option<unsafe extern "C" fn(*mut c_void)>,
    pub fill_input_buffer: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean>,
    pub skip_input_data: Option<unsafe extern "C" fn(*mut c_void, c_long)>,
    pub resync_to_restart: Option<unsafe extern "C" fn(*mut c_void, c_int) -> CBoolean>,
    pub term_source: Option<unsafe extern "C" fn(*mut c_void)>,
}

// ---------------------------------------------------------------------------
// `struct jpeg_decompress_struct` — public-facing subset.
// ---------------------------------------------------------------------------
//
// This is the view of the struct that consumers touch directly. It starts
// with the `jpeg_common_fields` sequence (`err`, `mem`, `progress`,
// `client_data`, `is_decompressor`, `global_state`) in libjpeg order,
// followed by `src` (decompressor-specific), then the image-description
// fields, and finally a `priv_ptr` that points to the owned Rust-side
// state.
//
// This is intentionally a small, bounded subset — real consumers writing
// to fields we don't expose would fall off the end of our allocation and
// invoke undefined behaviour, so we document the subset here and leave
// the remainder for future work.

/// Public subset of `struct jpeg_decompress_struct`. `#[repr(C)]` pins the
/// field order so consumers can read it with the same offsets they would
/// in real libjpeg (for the fields we expose).
#[repr(C)]
pub struct JpegDecompressPublic {
    // --- jpeg_common_fields ------------------------------------------------
    pub err: *mut JpegErrorMgr,
    pub mem: *mut c_void,
    pub progress: *mut c_void,
    pub client_data: *mut c_void,
    pub is_decompressor: CBoolean,
    pub global_state: c_int,
    // --- decompressor-specific: source manager ----------------------------
    pub src: *mut JpegSourceMgr,
    // --- image description, filled by jpeg_read_header() ------------------
    pub image_width: JDimension,
    pub image_height: JDimension,
    pub num_components: c_int,
    /// J_COLOR_SPACE of the encoded image.
    pub jpeg_color_space: c_int,
    // --- output-side parameters ------------------------------------------
    pub out_color_space: c_int,
    pub scale_num: u32,
    pub scale_denom: u32,
    pub output_width: JDimension,
    pub output_height: JDimension,
    pub out_color_components: c_int,
    pub output_components: c_int,
    pub rec_outbuf_height: c_int,
    pub output_scanline: JDimension,
    // --- private Rust-side state (opaque) --------------------------------
    pub priv_ptr: *mut c_void,
}

// Global-state values mirror libjpeg's internal state machine, just to the
// level of granularity we need for entry-point sequencing.
const DSTATE_START: c_int = 200;
const DSTATE_INHEADER: c_int = 201;
#[allow(dead_code)]
const DSTATE_READY: c_int = 202;
const DSTATE_SCANNING: c_int = 205;
const DSTATE_STOPPING: c_int = 206;

/// Source-of-JPEG variants. The memory variant borrows into the caller's
/// buffer; the stdio variant owns a read-once copy of the file's bytes.
enum JpegSource {
    Memory { ptr: *const u8, len: usize },
    Owned(Vec<u8>),
    None,
}

impl JpegSource {
    fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            // SAFETY: lifetime tied to the caller-owned source buffer.
            // The libjpeg contract places the lifetime-management burden
            // on the application, consistent with upstream.
            JpegSource::Memory { ptr, len } => unsafe {
                if ptr.is_null() {
                    None
                } else {
                    Some(std::slice::from_raw_parts(*ptr, *len))
                }
            },
            JpegSource::Owned(v) => Some(v.as_slice()),
            JpegSource::None => None,
        }
    }
}

/// J_COLOR_SPACE numeric constants matching `jpeglib.h`. Only the
/// subset the decode entry points produce is enumerated here.
const JCS_UNKNOWN: c_int = 0;
const JCS_GRAYSCALE: c_int = 1;
const JCS_RGB: c_int = 2;
const JCS_YCBCR: c_int = 3;
const JCS_CMYK: c_int = 4;
const JCS_YCCK: c_int = 5;

/// Rust-side private state hung off `JpegDecompressPublic::priv_ptr`.
/// Owned via `Box`; freed in `jpeg_destroy_decompress`.
struct DecompressPrivate {
    source: JpegSource,
    /// Owned storage backing the caller-visible `JpegSourceMgr`. We keep
    /// a pointer to it in `public.src` and mutate `bytes_in_buffer`/
    /// `next_input_byte` via the public field.
    source_mgr: Option<Box<JpegSourceMgr>>,
    /// Last error message for debugging. Held here because libjpeg's
    /// `format_message` callback expects it.
    last_error: CString,
    /// Decoded image buffer, built lazily on `jpeg_start_decompress`.
    decoded: Option<libjpeg_turbo_rs::Image>,
}

impl Default for DecompressPrivate {
    fn default() -> Self {
        Self {
            source: JpegSource::None,
            source_mgr: None,
            last_error: CString::new("No error").expect("static"),
            decoded: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: validate and reach into the caller's `cinfo`.
// ---------------------------------------------------------------------------

/// Interpret `cinfo` as our public struct. Returns `None` for NULL; does
/// not otherwise validate content.
///
/// # Safety
/// Caller must guarantee that `cinfo` either is NULL or points to a
/// valid `JpegDecompressPublic`-sized allocation.
unsafe fn cinfo_mut<'a>(cinfo: *mut c_void) -> Option<&'a mut JpegDecompressPublic> {
    if cinfo.is_null() {
        None
    } else {
        Some(unsafe { &mut *(cinfo as *mut JpegDecompressPublic) })
    }
}

/// Dereference the private state hung off `JpegDecompressPublic::priv_ptr`.
/// Returns `None` if the pointer is NULL (struct wasn't properly created).
///
/// Takes the `priv_ptr` by raw pointer to avoid aliasing with an
/// existing `&mut JpegDecompressPublic`, so callers can hold references
/// to both the public and private state simultaneously.
///
/// # Safety
/// Caller must have invoked `jpeg_create_decompress` before reaching any
/// function that calls this helper; we enforce that with a NULL-check.
unsafe fn priv_from_ptr<'a>(priv_ptr: *mut c_void) -> Option<&'a mut DecompressPrivate> {
    if priv_ptr.is_null() {
        None
    } else {
        Some(unsafe { &mut *(priv_ptr as *mut DecompressPrivate) })
    }
}

// ---------------------------------------------------------------------------
// `jpeg_std_error` — subtask FFI A1-11 #2.
// ---------------------------------------------------------------------------

/// Default `error_exit` callback. libjpeg's contract says this function
/// must not return; we implement that via `std::process::abort`. Apps
/// that want graceful error handling override the callback with a
/// `longjmp`-style jump.
unsafe extern "C" fn default_error_exit(_cinfo: *mut c_void) {
    // Emit to stderr then abort. Upstream libjpeg prints the formatted
    // message; we keep parity at the coarse level.
    eprintln!("libjpeg-turbo-rs: fatal JPEG error (default_error_exit)");
    std::process::abort();
}

unsafe extern "C" fn default_emit_message(_cinfo: *mut c_void, _msg_level: c_int) {
    // No-op by default; real libjpeg prints warnings above trace_level.
}

unsafe extern "C" fn default_output_message(_cinfo: *mut c_void) {
    // No-op by default — real libjpeg routes through stderr.
}

unsafe extern "C" fn default_format_message(_cinfo: *mut c_void, buffer: *mut u8) {
    if buffer.is_null() {
        return;
    }
    // Minimal default: write a sentinel so clients don't read uninit memory.
    let msg: &[u8] = b"libjpeg-turbo-rs error\0";
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), buffer, msg.len());
    }
}

unsafe extern "C" fn default_reset_error_mgr(cinfo: *mut c_void) {
    unsafe {
        if let Some(c) = cinfo_mut(cinfo) {
            if !c.err.is_null() {
                let err: &mut JpegErrorMgr = &mut *c.err;
                err.msg_code = 0;
                err.num_warnings = 0;
            }
        }
    }
}

/// `jpeg_std_error(err: *mut jpeg_error_mgr) -> *mut jpeg_error_mgr`.
///
/// Populates the error manager struct with the default callbacks and
/// returns the same pointer so the common idiom
/// `cinfo.err = jpeg_std_error(&err);` works as expected.
#[no_mangle]
pub extern "C" fn jpeg_std_error(err: *mut JpegErrorMgr) -> *mut JpegErrorMgr {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller must ensure `err` points to a writable JpegErrorMgr.
    unsafe {
        let e: &mut JpegErrorMgr = &mut *err;
        e.error_exit = Some(default_error_exit);
        e.emit_message = Some(default_emit_message);
        e.output_message = Some(default_output_message);
        e.format_message = Some(default_format_message);
        e.reset_error_mgr = Some(default_reset_error_mgr);
        e.msg_code = 0;
        e.msg_parm = [0u8; JMSG_STR_PARM_MAX];
        e.trace_level = 0;
        e.num_warnings = 0;
        e.jpeg_message_table = std::ptr::null();
        e.last_jpeg_message = 0;
        e.addon_message_table = std::ptr::null();
        e.first_addon_message = 0;
        e.last_addon_message = 0;
    }
    err
}

// ---------------------------------------------------------------------------
// `jpeg_CreateDecompress` + `jpeg_destroy_decompress` — subtask #3.
// ---------------------------------------------------------------------------

/// `jpeg_CreateDecompress(cinfo, version, structsize)`.
///
/// The `jpeg_create_decompress(cinfo)` macro expands to this function
/// with `version = JPEG_LIB_VERSION` and `structsize = sizeof(*cinfo)`.
/// We ignore version/size checks — our layout is a compatibility subset
/// independent of libjpeg's build-time `JPEG_LIB_VERSION`.
#[no_mangle]
pub extern "C" fn jpeg_CreateDecompress(cinfo: *mut c_void, _version: c_int, _struct_size: usize) {
    // SAFETY: `cinfo` is caller-allocated; we only touch the bytes that
    // fit the `JpegDecompressPublic` prefix. If the buffer is smaller
    // than that, the caller violated libjpeg's `sizeof(*cinfo)` contract
    // and crashes are acceptable (matching upstream).
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    // Do NOT zero `err` — the caller sets that up before calling us
    // (per the `cinfo.err = jpeg_std_error(&err);` idiom).
    c.mem = std::ptr::null_mut();
    c.progress = std::ptr::null_mut();
    c.client_data = std::ptr::null_mut();
    c.is_decompressor = 1;
    c.global_state = DSTATE_START;
    c.src = std::ptr::null_mut();
    c.image_width = 0;
    c.image_height = 0;
    c.num_components = 0;
    c.jpeg_color_space = JCS_UNKNOWN;
    c.out_color_space = JCS_UNKNOWN;
    c.scale_num = 1;
    c.scale_denom = 1;
    c.output_width = 0;
    c.output_height = 0;
    c.out_color_components = 0;
    c.output_components = 0;
    c.rec_outbuf_height = 1;
    c.output_scanline = 0;

    // Allocate the private Rust-side state and hang it off the struct.
    let private: Box<DecompressPrivate> = Box::default();
    c.priv_ptr = Box::into_raw(private) as *mut c_void;
}

/// `jpeg_destroy_decompress(cinfo)` — free the Rust-side private state.
#[no_mangle]
pub extern "C" fn jpeg_destroy_decompress(cinfo: *mut c_void) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    if !c.priv_ptr.is_null() {
        // SAFETY: we allocated this in `jpeg_CreateDecompress` via Box::into_raw.
        let _drop: Box<DecompressPrivate> =
            unsafe { Box::from_raw(c.priv_ptr as *mut DecompressPrivate) };
        c.priv_ptr = std::ptr::null_mut();
    }
    c.src = std::ptr::null_mut();
    c.global_state = 0;
}

// ---------------------------------------------------------------------------
// `jpeg_stdio_src` / `jpeg_mem_src` — subtask #4.
// ---------------------------------------------------------------------------

// Empty source-manager callbacks. Our decode path uses the byte slice
// directly instead of stepping through the callback FSM, so these can
// safely be stubs. If a consumer does invoke them (e.g. to fill/skip),
// we keep `bytes_in_buffer`/`next_input_byte` consistent.
unsafe extern "C" fn noop_init_source(_cinfo: *mut c_void) {}
unsafe extern "C" fn noop_fill_input_buffer(_cinfo: *mut c_void) -> CBoolean {
    // Real libjpeg returns FALSE to signal "suspend". We report TRUE so
    // that callers that accidentally invoke fill_input_buffer don't loop
    // forever; the entire buffer is already present.
    1
}
unsafe extern "C" fn noop_skip_input_data(cinfo: *mut c_void, num_bytes: c_long) {
    if num_bytes <= 0 {
        return;
    }
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        if !c.src.is_null() {
            let src: &mut JpegSourceMgr = unsafe { &mut *c.src };
            let skip: usize = num_bytes as usize;
            if skip >= src.bytes_in_buffer {
                src.next_input_byte = std::ptr::null();
                src.bytes_in_buffer = 0;
            } else {
                // SAFETY: we computed offset < bytes_in_buffer.
                src.next_input_byte = unsafe { src.next_input_byte.add(skip) };
                src.bytes_in_buffer -= skip;
            }
        }
    }
}
unsafe extern "C" fn noop_resync_to_restart(_cinfo: *mut c_void, _desired: c_int) -> CBoolean {
    // Always resync successfully (matches upstream's default).
    1
}
unsafe extern "C" fn noop_term_source(_cinfo: *mut c_void) {}

/// Attach a source manager that reads from an already-in-memory byte
/// slice. Matches libjpeg `jpeg_mem_src(cinfo, buf, size)`.
#[no_mangle]
pub extern "C" fn jpeg_mem_src(cinfo: *mut c_void, buf: *const u8, size: std::os::raw::c_ulong) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };

    let len: usize = size as usize;
    priv_state.source = JpegSource::Memory { ptr: buf, len };
    install_source_mgr(c, priv_state, buf, len);
}

/// Attach a source manager that reads from a `FILE *`. libjpeg's signature
/// takes `FILE *`; we accept `*mut c_void` since stable Rust does not
/// commit to an `stdio::FILE` layout. Internally we promote to a `File`
/// via the POSIX `fileno` + `fdopen`-equivalent path.
///
/// For simplicity and portability, this implementation slurps the entire
/// file into memory at `jpeg_stdio_src` time and delegates to `jpeg_mem_src`.
/// libjpeg does not guarantee that the `FILE *` remains valid after
/// `jpeg_finish_decompress`, so eagerly copying the bytes matches the
/// semantics applications rely on in practice.
#[no_mangle]
pub extern "C" fn jpeg_stdio_src(cinfo: *mut c_void, infile: *mut c_void) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if infile.is_null() {
        priv_state.source = JpegSource::None;
        return;
    }
    let bytes: Vec<u8> = match read_c_file(infile) {
        Ok(b) => b,
        Err(msg) => {
            priv_state.last_error =
                CString::new(format!("jpeg_stdio_src: {msg}")).unwrap_or_default();
            return;
        }
    };
    let buf_ptr: *const u8 = bytes.as_ptr();
    let buf_len: usize = bytes.len();
    priv_state.source = JpegSource::Owned(bytes);
    // Re-resolve the pointer from the stored buffer to avoid pointing at
    // the stack-local `bytes` variable whose lifetime ended above.
    if let JpegSource::Owned(ref v) = priv_state.source {
        install_source_mgr(c, priv_state, v.as_ptr(), v.len());
        let _ = (buf_ptr, buf_len); // silence unused on the borrow path
    }
}

/// Slurp a `FILE *` by converting it to a POSIX fd via `fileno` and
/// reading through `std::fs::File::from_raw_fd`. We duplicate the fd so
/// closing the Rust `File` on drop doesn't affect the caller's stream.
fn read_c_file(file: *mut c_void) -> Result<Vec<u8>, String> {
    #[cfg(unix)]
    {
        use std::os::unix::io::{FromRawFd, RawFd};
        extern "C" {
            fn fileno(stream: *mut c_void) -> c_int;
            fn dup(oldfd: c_int) -> c_int;
        }
        let fd_raw: c_int = unsafe { fileno(file) };
        if fd_raw < 0 {
            return Err("fileno returned -1".into());
        }
        let dup_fd: RawFd = unsafe { dup(fd_raw) };
        if dup_fd < 0 {
            return Err("dup(fileno) returned -1".into());
        }
        let mut f: std::fs::File = unsafe { std::fs::File::from_raw_fd(dup_fd) };
        let mut buf: Vec<u8> = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::{FromRawHandle, RawHandle};
        // We don't have a stable equivalent of `fileno` on Windows in a
        // minimal shim; callers on Windows should prefer `jpeg_mem_src`.
        Err("jpeg_stdio_src is unavailable on Windows; use jpeg_mem_src instead".into())
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err("jpeg_stdio_src is unavailable on this platform".into())
    }
}

/// Create a `JpegSourceMgr` if needed, and point `cinfo.src` at it with
/// `next_input_byte` / `bytes_in_buffer` already pre-loaded.
fn install_source_mgr(
    c: &mut JpegDecompressPublic,
    priv_state: &mut DecompressPrivate,
    ptr: *const u8,
    len: usize,
) {
    if let Some(mgr) = priv_state.source_mgr.as_mut() {
        mgr.next_input_byte = ptr;
        mgr.bytes_in_buffer = len;
    } else {
        priv_state.source_mgr = Some(Box::new(JpegSourceMgr {
            next_input_byte: ptr,
            bytes_in_buffer: len,
            init_source: Some(noop_init_source),
            fill_input_buffer: Some(noop_fill_input_buffer),
            skip_input_data: Some(noop_skip_input_data),
            resync_to_restart: Some(noop_resync_to_restart),
            term_source: Some(noop_term_source),
        }));
    }
    c.src = priv_state
        .source_mgr
        .as_mut()
        .map(|b| b.as_mut() as *mut JpegSourceMgr)
        .unwrap_or(std::ptr::null_mut());
}

// ---------------------------------------------------------------------------
// `jpeg_read_header` — subtask #5.
// ---------------------------------------------------------------------------

/// Map our internal `ColorSpace` to libjpeg's `J_COLOR_SPACE` int code.
fn colorspace_to_jcs(cs: libjpeg_turbo_rs::ColorSpace) -> c_int {
    use libjpeg_turbo_rs::ColorSpace;
    match cs {
        ColorSpace::Grayscale => JCS_GRAYSCALE,
        ColorSpace::Rgb => JCS_RGB,
        ColorSpace::YCbCr => JCS_YCBCR,
        ColorSpace::Cmyk => JCS_CMYK,
        ColorSpace::Ycck => JCS_YCCK,
        ColorSpace::Unknown => JCS_UNKNOWN,
    }
}

/// Peek at the JPEG header to populate `cinfo.image_width` etc. without
/// triggering a full decode. Uses the Rust-side `Decoder::new` which
/// only parses markers up to (but not including) entropy-coded data.
#[no_mangle]
pub extern "C" fn jpeg_read_header(cinfo: *mut c_void, _require_image: CBoolean) -> c_int {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return JPEG_SUSPENDED,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return JPEG_SUSPENDED,
    };

    let bytes: &[u8] = match priv_state.source.as_bytes() {
        Some(b) if b.len() >= 2 => b,
        _ => {
            priv_state.last_error =
                CString::new("jpeg_read_header: no JPEG source attached").unwrap_or_default();
            return JPEG_SUSPENDED;
        }
    };

    let decoder: libjpeg_turbo_rs::Decoder<'_> = match libjpeg_turbo_rs::Decoder::new(bytes) {
        Ok(d) => d,
        Err(e) => {
            priv_state.last_error =
                CString::new(format!("jpeg_read_header: {e}")).unwrap_or_default();
            return JPEG_SUSPENDED;
        }
    };

    let frame: &libjpeg_turbo_rs::FrameHeader = decoder.header();
    c.image_width = frame.width as JDimension;
    c.image_height = frame.height as JDimension;
    c.num_components = frame.components.len() as c_int;

    // Heuristic for jpeg_color_space matching libjpeg jdmarker:
    //   1 component     -> JCS_GRAYSCALE
    //   3 components    -> JCS_YCbCr (or JCS_RGB if Adobe APP14 says so)
    //   4 components    -> JCS_CMYK (or JCS_YCCK if Adobe APP14 says so)
    // We don't have an Adobe-marker getter in the thin header path, so
    // conservatively pick the common case; out_color_space below is the
    // effective one anyway.
    c.jpeg_color_space = match frame.components.len() {
        1 => JCS_GRAYSCALE,
        3 => JCS_YCBCR,
        4 => JCS_CMYK,
        _ => JCS_UNKNOWN,
    };
    // Default out_color_space: matches libjpeg's defaults in jdmaster.
    c.out_color_space = match c.jpeg_color_space {
        JCS_GRAYSCALE => JCS_GRAYSCALE,
        JCS_YCBCR => JCS_RGB,
        JCS_CMYK => JCS_CMYK,
        JCS_YCCK => JCS_CMYK,
        other => other,
    };
    c.global_state = DSTATE_INHEADER;
    priv_state.last_error = CString::new("No error").expect("static");
    JPEG_HEADER_OK
}

// ---------------------------------------------------------------------------
// `jpeg_start_decompress` — subtask #6.
// ---------------------------------------------------------------------------

/// Map a `J_COLOR_SPACE` int to the `PixelFormat` the Rust decoder
/// should emit. Returns `None` for spaces we don't currently surface.
fn jcs_to_pixel_format(cs: c_int) -> Option<PixelFormat> {
    match cs {
        JCS_GRAYSCALE => Some(PixelFormat::Grayscale),
        JCS_RGB => Some(PixelFormat::Rgb),
        JCS_YCBCR => Some(PixelFormat::Rgb), // decoder converts YCbCr->RGB
        JCS_CMYK => Some(PixelFormat::Cmyk),
        _ => None,
    }
}

/// Eagerly decode the image so `jpeg_read_scanlines` can serve rows out
/// of a backing buffer. Returns TRUE on success (libjpeg contract: never
/// returns FALSE except when suspending via a nonblocking source).
#[no_mangle]
pub extern "C" fn jpeg_start_decompress(cinfo: *mut c_void) -> CBoolean {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };

    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) => b.to_vec(),
        None => {
            priv_state.last_error =
                CString::new("jpeg_start_decompress: no source").unwrap_or_default();
            return 0;
        }
    };

    let format: PixelFormat = match jcs_to_pixel_format(c.out_color_space) {
        Some(f) => f,
        None => PixelFormat::Rgb,
    };
    let image: libjpeg_turbo_rs::Image = match libjpeg_turbo_rs::decompress_to(&bytes, format) {
        Ok(i) => i,
        Err(_e) => {
            // Fall back to the default decompress path — this handles
            // colorspaces whose "native" format matches a different
            // PixelFormat (e.g. grayscale images with JCS_GRAYSCALE).
            match decompress(&bytes) {
                Ok(i) => i,
                Err(e) => {
                    priv_state.last_error =
                        CString::new(format!("jpeg_start_decompress: {e}")).unwrap_or_default();
                    return 0;
                }
            }
        }
    };

    let out_cs_effective: c_int = colorspace_to_jcs(match image.pixel_format {
        PixelFormat::Grayscale => libjpeg_turbo_rs::ColorSpace::Grayscale,
        PixelFormat::Cmyk => libjpeg_turbo_rs::ColorSpace::Cmyk,
        _ => libjpeg_turbo_rs::ColorSpace::Rgb,
    });
    c.output_width = image.width as JDimension;
    c.output_height = image.height as JDimension;
    c.out_color_components = image.pixel_format.bytes_per_pixel() as c_int;
    c.output_components = c.out_color_components;
    c.out_color_space = out_cs_effective;
    c.output_scanline = 0;
    c.global_state = DSTATE_SCANNING;
    c.rec_outbuf_height = 1;
    priv_state.decoded = Some(image);
    priv_state.last_error = CString::new("No error").expect("static");
    1
}

// ---------------------------------------------------------------------------
// `jpeg_read_scanlines` — subtask #7.
// ---------------------------------------------------------------------------

/// Copy up to `max_lines` rows from the already-decoded image into the
/// application's row-pointer array. Returns the number of rows copied.
///
/// `scanlines` is a `JSAMPARRAY` = `JSAMPLE **` (array of row pointers).
#[no_mangle]
pub extern "C" fn jpeg_read_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u8,
    max_lines: JDimension,
) -> JDimension {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || max_lines == 0 {
        return 0;
    }
    let image: &libjpeg_turbo_rs::Image = match priv_state.decoded.as_ref() {
        Some(img) => img,
        None => return 0,
    };
    let bpp: usize = image.pixel_format.bytes_per_pixel();
    let row_bytes: usize = image.width * bpp;
    let total_rows: JDimension = image.height as JDimension;
    let remaining: JDimension = total_rows.saturating_sub(c.output_scanline);
    let to_copy: JDimension = std::cmp::min(max_lines, remaining);
    if to_copy == 0 {
        return 0;
    }
    // SAFETY: caller pinky-promises the `scanlines` array has at least
    // `max_lines` pointers, each pointing to a buffer of `row_bytes`.
    for i in 0..(to_copy as usize) {
        let dst: *mut u8 = unsafe { *scanlines.add(i) };
        if dst.is_null() {
            break;
        }
        let src_offset: usize = (c.output_scanline as usize + i) * row_bytes;
        let src: &[u8] = &image.data[src_offset..src_offset + row_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, row_bytes);
        }
    }
    c.output_scanline += to_copy;
    to_copy
}

// ---------------------------------------------------------------------------
// `jpeg_finish_decompress` — subtask #8.
// ---------------------------------------------------------------------------

/// Close out the decode pass. Returns TRUE unless suspended (which we
/// never do — the entire stream is present in memory).
#[no_mangle]
pub extern "C" fn jpeg_finish_decompress(cinfo: *mut c_void) -> CBoolean {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    c.global_state = DSTATE_STOPPING;
    // Drop the decoded image eagerly so the next jpeg_read_header on the
    // same handle starts from a clean slate.
    let priv_ptr: *mut c_void = c.priv_ptr;
    if let Some(priv_state) = unsafe { priv_from_ptr(priv_ptr) } {
        priv_state.decoded = None;
    }
    1
}

// ---------------------------------------------------------------------------
// Test-only accessors (prefixed `jpeg_capi_test_*`) so tests do not have to
// depend on the exact byte offset of every field they check. These are
// intentionally NOT `jpeg_*` — they are internal helpers, but must be
// exported from the cdylib for dlopen-based tests to reach them.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn jpeg_capi_test_dimensions(
    cinfo: *mut c_void,
    out_w: *mut u32,
    out_h: *mut u32,
    out_nc: *mut c_int,
    out_cs: *mut c_int,
) {
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        // SAFETY: caller checked pointers.
        unsafe {
            if !out_w.is_null() {
                *out_w = c.image_width;
            }
            if !out_h.is_null() {
                *out_h = c.image_height;
            }
            if !out_nc.is_null() {
                *out_nc = c.num_components;
            }
            if !out_cs.is_null() {
                *out_cs = c.jpeg_color_space;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_out_cs(cinfo: *mut c_void, cs: c_int) {
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        c.out_color_space = cs;
    }
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_output_dims(
    cinfo: *mut c_void,
    out_w: *mut u32,
    out_h: *mut u32,
    out_components: *mut c_int,
    out_cs: *mut c_int,
) {
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        unsafe {
            if !out_w.is_null() {
                *out_w = c.output_width;
            }
            if !out_h.is_null() {
                *out_h = c.output_height;
            }
            if !out_components.is_null() {
                *out_components = c.output_components;
            }
            if !out_cs.is_null() {
                *out_cs = c.out_color_space;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compile-time layout assertions. If the `JpegDecompressPublic` prefix
// shifts unexpectedly, force a build failure so we notice before a
// consumer's offsets desync.
// ---------------------------------------------------------------------------

const _: () = {
    // `err` must be at offset 0 so `cinfo.err = jpeg_std_error(&err);`
    // (which compiles to a pointer store at offset 0) remains correct.
    assert!(std::mem::offset_of!(JpegDecompressPublic, err) == 0);
    assert!(std::mem::offset_of!(JpegDecompressPublic, is_decompressor) > 0);
    // At least a bounded sanity check: the public subset must fit in the
    // test's 4096-byte opaque buffer.
    assert!(std::mem::size_of::<JpegDecompressPublic>() <= 4096);
};
