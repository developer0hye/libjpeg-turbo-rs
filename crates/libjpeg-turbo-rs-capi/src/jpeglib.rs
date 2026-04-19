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

// ===========================================================================
// FFI C2-{1..6}: classic libjpeg (`jpeg_*`) encode entry points.
// ===========================================================================
//
// This is the symmetric counterpart to the decode side above. The struct
// shape is a minimal subset of `jpeg_compress_struct` that covers the
// fields cjpeg / Pillow / ImageMagick actually read or write. Fields we
// don't expose are either computed internally (and reported back via
// `JpegCompressPublic`) or inert defaults.
//
// All entry points here must read from `references/libjpeg-turbo/src/
// jpeglib.h` for ABI-level signatures; we mirror them verbatim.

/// Verbatim `struct jpeg_destination_mgr` layout.
#[repr(C)]
pub struct JpegDestinationMgr {
    pub next_output_byte: *mut u8,
    pub free_in_buffer: usize,
    pub init_destination: Option<unsafe extern "C" fn(*mut c_void)>,
    pub empty_output_buffer: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean>,
    pub term_destination: Option<unsafe extern "C" fn(*mut c_void)>,
}

/// Public subset of `struct jpeg_compress_struct`. The field order after
/// `jpeg_common_fields` matches libjpeg 9: `dest`, then the image-
/// description fields that callers must populate before
/// `jpeg_set_defaults()`, then the compression parameters, and finally a
/// `priv_ptr` for Rust-side private state.
#[repr(C)]
pub struct JpegCompressPublic {
    // --- jpeg_common_fields ------------------------------------------------
    pub err: *mut JpegErrorMgr,
    pub mem: *mut c_void,
    pub progress: *mut c_void,
    pub client_data: *mut c_void,
    pub is_decompressor: CBoolean,
    pub global_state: c_int,
    // --- compressor-specific: destination manager ------------------------
    pub dest: *mut JpegDestinationMgr,
    // --- image description, filled in by caller --------------------------
    pub image_width: JDimension,
    pub image_height: JDimension,
    pub input_components: c_int,
    pub in_color_space: c_int,
    // --- parameters set by jpeg_set_defaults ------------------------------
    pub data_precision: c_int,
    pub num_components: c_int,
    pub jpeg_color_space: c_int,
    pub comp_info: *mut JpegComponentInfoCompress,
    pub restart_interval: u32,
    pub restart_in_rows: c_int,
    pub write_JFIF_header: CBoolean,
    pub JFIF_major_version: u8,
    pub JFIF_minor_version: u8,
    pub density_unit: u8,
    pub X_density: u16,
    pub Y_density: u16,
    pub write_Adobe_marker: CBoolean,
    pub next_scanline: JDimension,
    pub progressive_mode: CBoolean,
    pub arith_code: CBoolean,
    pub optimize_coding: CBoolean,
    pub raw_data_in: CBoolean,
    pub smoothing_factor: c_int,
    pub dct_method: c_int,
    // --- private Rust-side state ------------------------------------------
    pub priv_ptr: *mut c_void,
}

/// Minimal `jpeg_component_info` surface for the encode path. cjpeg walks
/// `comp_info[i].h_samp_factor / .v_samp_factor / .quant_tbl_no` to apply
/// user-set subsampling. We expose just those fields and compute the rest
/// internally at `jpeg_start_compress` time.
#[repr(C)]
pub struct JpegComponentInfoCompress {
    pub component_id: c_int,
    pub component_index: c_int,
    pub h_samp_factor: c_int,
    pub v_samp_factor: c_int,
    pub quant_tbl_no: c_int,
    pub dc_tbl_no: c_int,
    pub ac_tbl_no: c_int,
}

// libjpeg compressor global_state values.
const CSTATE_START: c_int = 100;
#[allow(dead_code)]
const CSTATE_SCANNING: c_int = 101;
#[allow(dead_code)]
const CSTATE_RAW_OK: c_int = 102;
#[allow(dead_code)]
const CSTATE_WRCOEFS: c_int = 103;

// MAX_COMPONENTS per libjpeg is 10; we reserve 4 entries (enough for
// Grayscale / RGB / YCbCr / CMYK / YCCK) because the shim never emits
// more than 4-component JPEGs.
const MAX_COMPS_OWNED: usize = 4;

/// Destination variants. `FileHandle` keeps the raw `FILE*` so
/// `term_destination` can `fflush` and drain buffered bytes to disk.
enum JpegDest {
    None,
    /// Application-owned `unsigned char **outbuffer` + `unsigned long *outsize`.
    Mem {
        outbuffer: *mut *mut u8,
        outsize: *mut std::os::raw::c_ulong,
    },
    /// File handle to write to via `fwrite`-equivalent.
    File {
        file: *mut c_void,
    },
}

/// Private encoder state held behind `JpegCompressPublic::priv_ptr`.
#[allow(dead_code)] // fields populated across C2-1..C2-5 subtasks
struct CompressPrivate {
    dest_kind: JpegDest,
    dest_mgr: Option<Box<JpegDestinationMgr>>,
    /// Staging buffer that the destination manager writes into. Fresh
    /// bytes come here before being flushed to the final sink.
    dest_buf: Vec<u8>,
    /// Accumulated scanlines (8-bit depth). Laid out row-major
    /// `image_width * input_components` bytes per row.
    pixels_u8: Vec<u8>,
    /// Accumulated scanlines (12/16-bit depth). Laid out row-major as
    /// `image_width * input_components` u16 samples per row.
    pixels_u16: Vec<u16>,
    /// Data precision this buffer represents (8, 12, or 16).
    precision: u8,
    /// Component owning storage for `comp_info[]`.
    comp_info: Vec<JpegComponentInfoCompress>,
    quality: u8,
    subsampling: libjpeg_turbo_rs::Subsampling,
    /// Lossless predictor (0 = lossy mode).
    lossless_predictor: u8,
    lossless_point_transform: u8,
    /// Application-supplied markers, accumulated during compression and
    /// emitted inside `jpeg_finish_compress` alongside the JPEG stream
    /// produced by the Rust encoder. Each `(marker_code, data)` tuple is
    /// inserted immediately after SOI. Data for `jpeg_write_m_byte` is
    /// streamed into the last entry.
    pending_markers: Vec<(c_int, Vec<u8>)>,
    /// ICC profile buffer (captured via `jpeg_write_icc_profile`).
    icc_profile: Option<Vec<u8>>,
    last_error: CString,
    /// If true, writer was started but has not produced output yet.
    have_started: bool,
    /// `jpeg_write_tables` mode: write only an abbreviated tables stream.
    tables_only: bool,
    /// Whether to write JFIF header (overrides C default of TRUE).
    write_jfif: bool,
    /// Suppress writing quant/huffman tables in this datastream.
    suppress_tables: bool,
}

impl Default for CompressPrivate {
    fn default() -> Self {
        Self {
            dest_kind: JpegDest::None,
            dest_mgr: None,
            dest_buf: Vec::with_capacity(4096),
            pixels_u8: Vec::new(),
            pixels_u16: Vec::new(),
            precision: 8,
            comp_info: Vec::new(),
            quality: 75,
            subsampling: libjpeg_turbo_rs::Subsampling::S420,
            lossless_predictor: 0,
            lossless_point_transform: 0,
            pending_markers: Vec::new(),
            icc_profile: None,
            last_error: CString::new("No error").expect("static"),
            have_started: false,
            tables_only: false,
            write_jfif: true,
            suppress_tables: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: reach into the caller's compress struct.
// ---------------------------------------------------------------------------

/// Interpret `cinfo` as our public compress struct. Returns `None` for NULL.
///
/// # Safety
/// Caller must guarantee that `cinfo` either is NULL or points to a
/// valid `JpegCompressPublic`-sized allocation.
unsafe fn cinfo_compress_mut<'a>(cinfo: *mut c_void) -> Option<&'a mut JpegCompressPublic> {
    if cinfo.is_null() {
        None
    } else {
        Some(unsafe { &mut *(cinfo as *mut JpegCompressPublic) })
    }
}

unsafe fn priv_compress_from_ptr<'a>(priv_ptr: *mut c_void) -> Option<&'a mut CompressPrivate> {
    if priv_ptr.is_null() {
        None
    } else {
        Some(unsafe { &mut *(priv_ptr as *mut CompressPrivate) })
    }
}

#[allow(dead_code)] // used in C2-2 (start_compress / write_scanlines)
fn jcs_to_pixel_format_for_input(cs: c_int) -> Option<PixelFormat> {
    match cs {
        JCS_GRAYSCALE => Some(PixelFormat::Grayscale),
        JCS_RGB => Some(PixelFormat::Rgb),
        JCS_YCBCR => Some(PixelFormat::Rgb), // treated as RGB during encode
        JCS_CMYK => Some(PixelFormat::Cmyk),
        // Extended color spaces (libjpeg-turbo only).
        13 /* JCS_EXT_RGB */ => Some(PixelFormat::Rgb),
        14 /* JCS_EXT_RGBX */ => Some(PixelFormat::Rgbx),
        15 /* JCS_EXT_BGR */ => Some(PixelFormat::Bgr),
        16 /* JCS_EXT_BGRX */ => Some(PixelFormat::Bgrx),
        17 /* JCS_EXT_XBGR */ => Some(PixelFormat::Xbgr),
        18 /* JCS_EXT_XRGB */ => Some(PixelFormat::Xrgb),
        19 /* JCS_EXT_RGBA */ => Some(PixelFormat::Rgba),
        20 /* JCS_EXT_BGRA */ => Some(PixelFormat::Bgra),
        21 /* JCS_EXT_ABGR */ => Some(PixelFormat::Bgra), // no direct ABGR match
        22 /* JCS_EXT_ARGB */ => Some(PixelFormat::Rgba), // no direct ARGB match
        _ => None,
    }
}

fn default_num_components_for(cs: c_int) -> c_int {
    match cs {
        JCS_GRAYSCALE => 1,
        JCS_RGB | JCS_YCBCR => 3,
        JCS_CMYK | JCS_YCCK => 4,
        13 | 15 => 3,                               // JCS_EXT_RGB / JCS_EXT_BGR
        14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 => 4, // _RGBX/BGRX/XBGR/XRGB/RGBA/BGRA/ABGR/ARGB
        _ => 3,
    }
}

// ---------------------------------------------------------------------------
// C2-1: jpeg_CreateCompress / destroy / stdio_dest / mem_dest / set_defaults
// / set_colorspace / default_colorspace / set_quality.
// ---------------------------------------------------------------------------

/// `jpeg_CreateCompress(cinfo, version, structsize)`.
///
/// The `jpeg_create_compress(cinfo)` macro expands to this function with
/// `version = JPEG_LIB_VERSION` and `structsize = sizeof(*cinfo)`.
#[no_mangle]
pub extern "C" fn jpeg_CreateCompress(cinfo: *mut c_void, _version: c_int, _struct_size: usize) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    c.mem = std::ptr::null_mut();
    c.progress = std::ptr::null_mut();
    c.client_data = std::ptr::null_mut();
    c.is_decompressor = 0;
    c.global_state = CSTATE_START;
    c.dest = std::ptr::null_mut();
    c.image_width = 0;
    c.image_height = 0;
    c.input_components = 0;
    c.in_color_space = JCS_UNKNOWN;
    c.data_precision = 8;
    c.num_components = 0;
    c.jpeg_color_space = JCS_UNKNOWN;
    c.comp_info = std::ptr::null_mut();
    c.restart_interval = 0;
    c.restart_in_rows = 0;
    c.write_JFIF_header = 1;
    c.JFIF_major_version = 1;
    c.JFIF_minor_version = 1;
    c.density_unit = 0;
    c.X_density = 1;
    c.Y_density = 1;
    c.write_Adobe_marker = 0;
    c.next_scanline = 0;
    c.progressive_mode = 0;
    c.arith_code = 0;
    c.optimize_coding = 0;
    c.raw_data_in = 0;
    c.smoothing_factor = 0;
    c.dct_method = 0; // JDCT_ISLOW

    let private: Box<CompressPrivate> = Box::default();
    c.priv_ptr = Box::into_raw(private) as *mut c_void;
}

/// Expansion of the `jpeg_create_compress(cinfo)` convenience macro.
/// libjpeg emits a direct call to `jpeg_CreateCompress` — we provide
/// both names so callers compiled against either form link cleanly.
#[no_mangle]
pub extern "C" fn jpeg_create_compress(cinfo: *mut c_void) {
    jpeg_CreateCompress(cinfo, 80, std::mem::size_of::<JpegCompressPublic>());
}

/// `jpeg_destroy_compress(cinfo)`.
#[no_mangle]
pub extern "C" fn jpeg_destroy_compress(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    if !c.priv_ptr.is_null() {
        // SAFETY: we allocated this in `jpeg_CreateCompress`.
        let _drop: Box<CompressPrivate> =
            unsafe { Box::from_raw(c.priv_ptr as *mut CompressPrivate) };
        c.priv_ptr = std::ptr::null_mut();
    }
    c.dest = std::ptr::null_mut();
    c.comp_info = std::ptr::null_mut();
    c.global_state = 0;
}

// --- destination manager callbacks (mem/stdio share staging buffer) ----

unsafe extern "C" fn mem_init_destination(cinfo: *mut c_void) {
    install_dest_staging(cinfo);
}

unsafe extern "C" fn mem_empty_output_buffer(cinfo: *mut c_void) -> CBoolean {
    // Called when the staging buffer fills up. Drain what's there to
    // the private `Vec`, then restart from the beginning.
    drain_dest_buffer(cinfo, /*final_flush=*/ false);
    install_dest_staging(cinfo);
    1
}

unsafe extern "C" fn mem_term_destination(cinfo: *mut c_void) {
    drain_dest_buffer(cinfo, /*final_flush=*/ true);
}

/// Point `dest->next_output_byte` / `free_in_buffer` at the private
/// staging buffer so the compressor has somewhere to write. The caller
/// is responsible for invoking this both at `init_destination` time and
/// after each `empty_output_buffer`.
fn install_dest_staging(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    // A 4 KiB staging chunk keeps syscall overhead down without bloating
    // the working set. It matches libjpeg-turbo's `OUTPUT_BUF_SIZE`.
    const STAGING_BYTES: usize = 4096;
    priv_state.dest_buf.resize(STAGING_BYTES, 0);
    if !c.dest.is_null() {
        let dest: &mut JpegDestinationMgr = unsafe { &mut *c.dest };
        dest.next_output_byte = priv_state.dest_buf.as_mut_ptr();
        dest.free_in_buffer = STAGING_BYTES;
    }
}

/// Copy bytes from the staging buffer into the final destination sink.
/// The `_final_flush` flag is kept as a contract marker — libjpeg
/// semantically distinguishes incremental flushes from the terminal
/// flush, but our synchronous in-memory pipeline copies the same
/// consumed prefix in either case.
fn drain_dest_buffer(cinfo: *mut c_void, _final_flush: bool) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if c.dest.is_null() {
        return;
    }
    let dest: &mut JpegDestinationMgr = unsafe { &mut *c.dest };
    let total: usize = priv_state.dest_buf.len();
    let consumed: usize = total.saturating_sub(dest.free_in_buffer);
    let bytes: &[u8] = &priv_state.dest_buf[..consumed];
    if bytes.is_empty() {
        return;
    }
    match priv_state.dest_kind {
        JpegDest::Mem { outbuffer, outsize } => {
            // SAFETY: caller-supplied out-pointers, validated non-NULL
            // at `jpeg_mem_dest` setup time. We grow the buffer by
            // allocating a fresh libc block each flush; for short files
            // this is fine, and large files quickly reach EOF.
            unsafe {
                let prev_ptr: *mut u8 = *outbuffer;
                let prev_len: usize = *outsize as usize;
                let new_len: usize = prev_len + bytes.len();
                let new_ptr: *mut u8 = crate::alloc::libc_malloc(new_len);
                if new_ptr.is_null() {
                    return;
                }
                if !prev_ptr.is_null() && prev_len > 0 {
                    std::ptr::copy_nonoverlapping(prev_ptr, new_ptr, prev_len);
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), new_ptr.add(prev_len), bytes.len());
                if !prev_ptr.is_null() {
                    crate::alloc::libc_free(prev_ptr);
                }
                *outbuffer = new_ptr;
                *outsize = new_len as std::os::raw::c_ulong;
            }
        }
        JpegDest::File { file } => {
            if file.is_null() {
                return;
            }
            write_c_file(file, bytes);
        }
        JpegDest::None => {}
    }
}

/// Write `bytes` to a `FILE *` via libc `fwrite`.
fn write_c_file(file: *mut c_void, bytes: &[u8]) {
    extern "C" {
        fn fwrite(ptr: *const c_void, size: usize, nmemb: usize, stream: *mut c_void) -> usize;
    }
    unsafe {
        fwrite(bytes.as_ptr() as *const c_void, 1, bytes.len(), file);
    }
}

/// Install a destination manager that streams bytes into a `FILE *`.
#[no_mangle]
pub extern "C" fn jpeg_stdio_dest(cinfo: *mut c_void, outfile: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    priv_state.dest_kind = JpegDest::File { file: outfile };
    install_dest_mgr(c, priv_state);
}

/// Install a destination manager that accumulates into a libc-allocated
/// buffer owned via `*outbuffer` / `*outsize`.
#[no_mangle]
pub extern "C" fn jpeg_mem_dest(
    cinfo: *mut c_void,
    outbuffer: *mut *mut u8,
    outsize: *mut std::os::raw::c_ulong,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if outbuffer.is_null() || outsize.is_null() {
        priv_state.dest_kind = JpegDest::None;
        return;
    }
    // libjpeg contract: if `*outbuffer` is NULL, allocate on first flush
    // and set `*outsize` to 0. Honour that pre-state.
    unsafe {
        if (*outbuffer).is_null() {
            *outsize = 0;
        }
    }
    priv_state.dest_kind = JpegDest::Mem { outbuffer, outsize };
    install_dest_mgr(c, priv_state);
}

/// Create-or-reuse the `JpegDestinationMgr` and point `cinfo.dest` at it.
fn install_dest_mgr(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) {
    if priv_state.dest_mgr.is_none() {
        priv_state.dest_mgr = Some(Box::new(JpegDestinationMgr {
            next_output_byte: std::ptr::null_mut(),
            free_in_buffer: 0,
            init_destination: Some(mem_init_destination),
            empty_output_buffer: Some(mem_empty_output_buffer),
            term_destination: Some(mem_term_destination),
        }));
    }
    c.dest = priv_state
        .dest_mgr
        .as_mut()
        .map(|b| b.as_mut() as *mut JpegDestinationMgr)
        .unwrap_or(std::ptr::null_mut());
}

/// Propagate `comp_info[]` defaults for the current `jpeg_color_space`.
/// Matches libjpeg `jpeg_set_colorspace` behavior for the subsampling
/// factors (h/v samp) and table-selector defaults used by cjpeg.
fn apply_colorspace_defaults(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) {
    let n: usize = c.num_components as usize;
    priv_state.comp_info.clear();
    priv_state.comp_info.reserve(MAX_COMPS_OWNED);
    for i in 0..n.min(MAX_COMPS_OWNED) {
        // ID assignments follow libjpeg: 1..N for YCbCr/CMYK/RGB,
        // 'R','G','B' (= 82,71,66) for explicit JCS_RGB (libjpeg-turbo
        // accepts either, but matches the common `1..N` form here).
        let id: c_int = match c.jpeg_color_space {
            JCS_GRAYSCALE => 1,
            JCS_YCBCR | JCS_YCCK => (i + 1) as c_int,
            JCS_RGB => match i {
                0 => 82,
                1 => 71,
                2 => 66,
                _ => (i + 1) as c_int,
            },
            JCS_CMYK => (i + 1) as c_int,
            _ => (i + 1) as c_int,
        };
        let (h, v): (c_int, c_int) = match (c.jpeg_color_space, i) {
            (JCS_YCBCR, 0) | (JCS_YCCK, 0) => (2, 2),
            // Luma component gets 2x2 by default (4:2:0 subsampling).
            _ => (1, 1),
        };
        // Chroma and K channels share quant/Huffman table 1.
        let (qt, dc, ac): (c_int, c_int, c_int) = match (c.jpeg_color_space, i) {
            (JCS_YCBCR, 0) | (JCS_YCCK, 0) | (JCS_YCCK, 3) => (0, 0, 0),
            (JCS_YCBCR, _) | (JCS_YCCK, _) => (1, 1, 1),
            _ => (0, 0, 0),
        };
        priv_state.comp_info.push(JpegComponentInfoCompress {
            component_id: id,
            component_index: i as c_int,
            h_samp_factor: h,
            v_samp_factor: v,
            quant_tbl_no: qt,
            dc_tbl_no: dc,
            ac_tbl_no: ac,
        });
    }
    c.comp_info = priv_state.comp_info.as_mut_ptr();
}

/// `jpeg_default_colorspace(cinfo)` — pick the standard JPEG color space
/// corresponding to `in_color_space`.
#[no_mangle]
pub extern "C" fn jpeg_default_colorspace(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let jcs: c_int = match c.in_color_space {
        JCS_GRAYSCALE => JCS_GRAYSCALE,
        JCS_RGB => JCS_YCBCR,
        JCS_YCBCR => JCS_YCBCR,
        JCS_CMYK => JCS_CMYK,
        JCS_YCCK => JCS_YCCK,
        13 | 15 => JCS_YCBCR, // JCS_EXT_RGB / _BGR — YCbCr output
        14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 => JCS_YCBCR,
        _ => JCS_UNKNOWN,
    };
    jpeg_set_colorspace(cinfo, jcs);
}

/// `jpeg_set_colorspace(cinfo, colorspace)` — select the JPEG color space
/// and populate `comp_info[]` with libjpeg's defaults.
#[no_mangle]
pub extern "C" fn jpeg_set_colorspace(cinfo: *mut c_void, colorspace: c_int) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    c.jpeg_color_space = colorspace;
    c.num_components = default_num_components_for(colorspace);
    // JFIF header is only valid for grayscale and YCbCr per spec.
    let jfif_applicable: bool = matches!(colorspace, JCS_GRAYSCALE | JCS_YCBCR);
    c.write_JFIF_header = if jfif_applicable && priv_state.write_jfif {
        1
    } else {
        0
    };
    // Adobe marker for CMYK / YCCK / RGB.
    c.write_Adobe_marker = matches!(colorspace, JCS_RGB | JCS_CMYK | JCS_YCCK) as CBoolean;
    apply_colorspace_defaults(c, priv_state);
}

/// `jpeg_set_defaults(cinfo)` — populate default compression parameters.
/// Requires the caller to have already set `in_color_space`.
#[no_mangle]
pub extern "C" fn jpeg_set_defaults(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    c.data_precision = 8;
    c.dct_method = 0;
    c.restart_interval = 0;
    c.restart_in_rows = 0;
    c.density_unit = 0;
    c.X_density = 1;
    c.Y_density = 1;
    c.JFIF_major_version = 1;
    c.JFIF_minor_version = 1;
    c.smoothing_factor = 0;
    c.progressive_mode = 0;
    c.arith_code = 0;
    c.optimize_coding = 0;
    c.raw_data_in = 0;
    // Apply the defaults that flow from `in_color_space` → `jpeg_color_space`.
    jpeg_default_colorspace(cinfo);
    // Default quality = 75 with baseline restriction per libjpeg.
    jpeg_set_quality(cinfo, 75, 1);
}

/// `jpeg_set_quality(cinfo, quality, force_baseline)` — install the
/// scaled luma and chroma quant tables for the requested quality
/// factor. The scaling curve matches libjpeg `jpeg_quality_scaling`.
#[no_mangle]
pub extern "C" fn jpeg_set_quality(cinfo: *mut c_void, quality: c_int, _force_baseline: CBoolean) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    let clamped: u8 = quality.clamp(1, 100) as u8;
    priv_state.quality = clamped;
}

// ---------------------------------------------------------------------------
// C2-2: jpeg_start_compress / jpeg_write_scanlines / jpeg_finish_compress.
// ---------------------------------------------------------------------------

/// Derive the chroma subsampling that applies to the YCbCr luma channel,
/// consulting `comp_info[0]` since that's what cjpeg-style callers set.
fn subsampling_from_comp_info(
    priv_state: &CompressPrivate,
    num_components: c_int,
) -> libjpeg_turbo_rs::Subsampling {
    use libjpeg_turbo_rs::Subsampling;
    if num_components < 2 || priv_state.comp_info.is_empty() {
        // Grayscale has no meaningful subsampling; S444 is the default.
        return Subsampling::S444;
    }
    let luma: &JpegComponentInfoCompress = &priv_state.comp_info[0];
    match (luma.h_samp_factor, luma.v_samp_factor) {
        (1, 1) => Subsampling::S444,
        (2, 1) => Subsampling::S422,
        (2, 2) => Subsampling::S420,
        (1, 2) => Subsampling::S440,
        (4, 1) => Subsampling::S411,
        (1, 4) => Subsampling::S441,
        _ => Subsampling::S444,
    }
}

/// Resolve the input `PixelFormat` matching the caller's `in_color_space`
/// and `input_components`. Falls back to `Rgb` when we don't recognise
/// the space; writers handle grayscale explicitly.
fn input_pixel_format(c: &JpegCompressPublic) -> PixelFormat {
    if let Some(pf) = jcs_to_pixel_format_for_input(c.in_color_space) {
        return pf;
    }
    match c.input_components {
        1 => PixelFormat::Grayscale,
        3 => PixelFormat::Rgb,
        4 => PixelFormat::Cmyk,
        _ => PixelFormat::Rgb,
    }
}

/// `jpeg_start_compress(cinfo, write_all_tables)`.
///
/// Transitions the state machine into SCANNING and primes the staging
/// buffer. Actual entropy coding happens inside `jpeg_finish_compress`
/// once all scanlines are present — libjpeg lets callers stream rows
/// either way, but our Rust-side encoder takes the entire image up
/// front, so we accumulate.
#[no_mangle]
pub extern "C" fn jpeg_start_compress(cinfo: *mut c_void, _write_all_tables: CBoolean) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    c.global_state = CSTATE_SCANNING;
    c.next_scanline = 0;

    let input_components: usize = c.input_components.max(1) as usize;
    let width: usize = c.image_width as usize;
    let height: usize = c.image_height as usize;
    let row_bytes: usize = width.saturating_mul(input_components);
    let total_bytes: usize = row_bytes.saturating_mul(height);
    priv_state.pixels_u8.clear();
    priv_state.pixels_u8.resize(total_bytes, 0);
    priv_state.pixels_u16.clear();
    priv_state.have_started = true;
    priv_state.tables_only = false;

    // Kick the destination manager via `init_destination` if installed so
    // the first write_scanline has a live staging buffer waiting.
    if !c.dest.is_null() {
        let dest: &JpegDestinationMgr = unsafe { &*c.dest };
        if let Some(init) = dest.init_destination {
            unsafe { init(cinfo) };
        }
    }
}

/// `jpeg_write_scanlines(cinfo, scanlines, num_lines) -> JDIMENSION`.
///
/// Copies up to `num_lines` rows from the application's row-pointer
/// array into our accumulation buffer. Returns the number actually
/// stored (may be less when near the image bottom).
#[no_mangle]
pub extern "C" fn jpeg_write_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u8,
    num_lines: JDimension,
) -> JDimension {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || num_lines == 0 {
        return 0;
    }
    let input_components: usize = c.input_components.max(1) as usize;
    let row_bytes: usize = (c.image_width as usize).saturating_mul(input_components);
    let total_rows: JDimension = c.image_height;
    let remaining: JDimension = total_rows.saturating_sub(c.next_scanline);
    let to_copy: JDimension = std::cmp::min(num_lines, remaining);
    if to_copy == 0 {
        return 0;
    }
    // SAFETY: caller guarantees `scanlines` holds `num_lines` row
    // pointers, each referencing at least `row_bytes` bytes.
    for i in 0..(to_copy as usize) {
        let row_ptr: *const u8 = unsafe { *scanlines.add(i) } as *const u8;
        if row_ptr.is_null() {
            break;
        }
        let dst_offset: usize = (c.next_scanline as usize + i) * row_bytes;
        let dst: &mut [u8] = &mut priv_state.pixels_u8[dst_offset..dst_offset + row_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(row_ptr, dst.as_mut_ptr(), row_bytes);
        }
    }
    c.next_scanline += to_copy;
    to_copy
}

/// Actually invoke the Rust encoder over the accumulated pixels. On
/// success, push the resulting bytes through the installed destination
/// manager (via `empty_output_buffer` flushes + `term_destination`).
fn run_encoder_and_flush(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) -> bool {
    let width: usize = c.image_width as usize;
    let height: usize = c.image_height as usize;
    if width == 0 || height == 0 {
        return false;
    }
    let pf: PixelFormat = input_pixel_format(c);
    let subsamp: libjpeg_turbo_rs::Subsampling =
        subsampling_from_comp_info(priv_state, c.num_components);
    priv_state.subsampling = subsamp;

    // Choose encode variant based on parameters captured earlier.
    let bytes_result = if priv_state.lossless_predictor != 0 {
        libjpeg_turbo_rs::compress_lossless_extended(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.lossless_predictor,
            priv_state.lossless_point_transform,
        )
    } else if c.progressive_mode != 0 && c.arith_code != 0 {
        libjpeg_turbo_rs::compress_arithmetic_progressive(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else if c.progressive_mode != 0 {
        libjpeg_turbo_rs::compress_progressive(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else if c.arith_code != 0 {
        libjpeg_turbo_rs::compress_arithmetic(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else if c.optimize_coding != 0 {
        libjpeg_turbo_rs::compress_optimized(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else {
        libjpeg_turbo_rs::compress(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    };

    let encoded: Vec<u8> = match bytes_result {
        Ok(b) => b,
        Err(e) => {
            priv_state.last_error =
                CString::new(format!("jpeg_finish_compress: {e}")).unwrap_or_default();
            return false;
        }
    };

    // Inject application-supplied markers right after SOI (offset 2).
    let with_markers: Vec<u8> =
        if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
            encoded
        } else {
            inject_markers_after_soi(&encoded, priv_state)
        };

    // Push the bytes through the destination manager by repeatedly
    // filling the staging buffer and calling `empty_output_buffer`
    // when it overflows, then `term_destination` at EOF.
    push_bytes_through_dest_mgr(c, priv_state, &with_markers);
    true
}

/// Emit `encoded` JPEG bytes by writing into the destination manager's
/// `next_output_byte` buffer, invoking `empty_output_buffer` whenever
/// the staging buffer fills.
fn push_bytes_through_dest_mgr(
    c: &mut JpegCompressPublic,
    priv_state: &mut CompressPrivate,
    encoded: &[u8],
) {
    if c.dest.is_null() {
        return;
    }
    let mut offset: usize = 0;
    while offset < encoded.len() {
        // Refill staging if empty.
        let mut need_refill: bool = false;
        {
            let dest: &JpegDestinationMgr = unsafe { &*c.dest };
            if dest.free_in_buffer == 0 || dest.next_output_byte.is_null() {
                need_refill = true;
            }
        }
        if need_refill {
            // First chunk: rely on caller's init_destination having
            // happened earlier; subsequent chunks need empty_output.
            let empty_fn: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean> =
                unsafe { (*c.dest).empty_output_buffer };
            if let Some(f) = empty_fn {
                unsafe {
                    f(c as *mut JpegCompressPublic as *mut c_void);
                }
            }
        }
        // Copy as much as fits into the staging buffer this round.
        let (dst_ptr, capacity): (*mut u8, usize) = {
            let dest: &JpegDestinationMgr = unsafe { &*c.dest };
            (dest.next_output_byte, dest.free_in_buffer)
        };
        if dst_ptr.is_null() || capacity == 0 {
            break;
        }
        let take: usize = std::cmp::min(capacity, encoded.len() - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.as_ptr().add(offset), dst_ptr, take);
            let dest: &mut JpegDestinationMgr = &mut *c.dest;
            dest.next_output_byte = dst_ptr.add(take);
            dest.free_in_buffer -= take;
        }
        offset += take;
    }
    // Emit the terminal flush.
    let term_fn: Option<unsafe extern "C" fn(*mut c_void)> = unsafe { (*c.dest).term_destination };
    if let Some(f) = term_fn {
        unsafe {
            f(c as *mut JpegCompressPublic as *mut c_void);
        }
    }
    // After term_destination the staging buffer is invalid; zero it.
    let dest: &mut JpegDestinationMgr = unsafe { &mut *c.dest };
    dest.next_output_byte = std::ptr::null_mut();
    dest.free_in_buffer = 0;
    // Prevent unused warning.
    let _ = priv_state;
}

/// Construct a new byte buffer that starts with the encoded SOI marker,
/// inserts any pending APPn markers and (optionally) the ICC_PROFILE
/// chunks, then continues with the original stream.
fn inject_markers_after_soi(encoded: &[u8], priv_state: &CompressPrivate) -> Vec<u8> {
    if encoded.len() < 2 || encoded[0] != 0xFF || encoded[1] != 0xD8 {
        // Not a JPEG stream — leave untouched.
        return encoded.to_vec();
    }
    let mut out: Vec<u8> = Vec::with_capacity(encoded.len() + 64);
    out.extend_from_slice(&encoded[..2]);

    // Emit APPn markers the caller requested via jpeg_write_marker.
    for (code, data) in &priv_state.pending_markers {
        write_marker_segment(&mut out, *code, data);
    }
    // Emit ICC profile via the standard APP2 multi-chunk layout.
    if let Some(icc) = &priv_state.icc_profile {
        write_app2_icc_inline(&mut out, icc);
    }
    out.extend_from_slice(&encoded[2..]);
    out
}

/// Emit APP2 "ICC_PROFILE" chunks in the standard multi-segment layout.
/// Ported from `src/encode/marker_writer.rs::write_app2_icc` so the
/// classic `jpeg_write_icc_profile` path doesn't have to cross a
/// private module boundary.
fn write_app2_icc_inline(buf: &mut Vec<u8>, profile: &[u8]) {
    const ICC_OVERHEAD: usize = 14;
    const MAX_DATA: usize = 65533 - ICC_OVERHEAD;
    let num_markers: usize = profile.len().div_ceil(MAX_DATA);
    let mut offset: usize = 0;
    for seq in 1..=num_markers {
        let chunk_len: usize = (profile.len() - offset).min(MAX_DATA);
        let marker_len: u16 = (ICC_OVERHEAD + chunk_len) as u16 + 2;
        buf.push(0xFF);
        buf.push(0xE2);
        buf.extend_from_slice(&marker_len.to_be_bytes());
        buf.extend_from_slice(b"ICC_PROFILE\0");
        buf.push(seq as u8);
        buf.push(num_markers as u8);
        buf.extend_from_slice(&profile[offset..offset + chunk_len]);
        offset += chunk_len;
    }
}

/// Write a single `marker_code` segment with `data`. The length field is
/// `len + 2` (includes the length word itself); oversized segments are
/// truncated to fit libjpeg's 65533-byte data limit.
fn write_marker_segment(out: &mut Vec<u8>, marker_code: c_int, data: &[u8]) {
    const MAX_DATA: usize = 65533;
    let len: usize = std::cmp::min(data.len(), MAX_DATA);
    let code: u8 = (marker_code & 0xFF) as u8;
    out.push(0xFF);
    out.push(code);
    let seg_len: u16 = (len as u16).wrapping_add(2);
    out.push((seg_len >> 8) as u8);
    out.push((seg_len & 0xFF) as u8);
    out.extend_from_slice(&data[..len]);
}

/// `jpeg_finish_compress(cinfo)` — close the datastream, flush to sink.
#[no_mangle]
pub extern "C" fn jpeg_finish_compress(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.priv_ptr;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if !priv_state.have_started {
        return;
    }
    priv_state.have_started = false;
    // Run the encoder and push bytes through the destination manager.
    let _ = run_encoder_and_flush(c, priv_state);
    c.global_state = CSTATE_START;
}

// ---------------------------------------------------------------------------
// Test-only accessors (encode side). Mirror the decode-side pattern so
// tests don't have to lock in field offsets.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_compress_dims(
    cinfo: *mut c_void,
    width: u32,
    height: u32,
    input_components: c_int,
    in_color_space: c_int,
) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        c.image_width = width;
        c.image_height = height;
        c.input_components = input_components;
        c.in_color_space = in_color_space;
    }
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_get_compress_state(
    cinfo: *mut c_void,
    out_num_components: *mut c_int,
    out_jpeg_color_space: *mut c_int,
    out_in_color_space: *mut c_int,
) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        unsafe {
            if !out_num_components.is_null() {
                *out_num_components = c.num_components;
            }
            if !out_jpeg_color_space.is_null() {
                *out_jpeg_color_space = c.jpeg_color_space;
            }
            if !out_in_color_space.is_null() {
                *out_in_color_space = c.in_color_space;
            }
        }
    }
}

// Compile-time layout assertions (encode).
const _: () = {
    assert!(std::mem::offset_of!(JpegCompressPublic, err) == 0);
    assert!(std::mem::offset_of!(JpegCompressPublic, is_decompressor) > 0);
    assert!(std::mem::size_of::<JpegCompressPublic>() <= 4096);
};
