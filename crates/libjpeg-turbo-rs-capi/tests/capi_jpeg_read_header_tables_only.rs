//! P3-shim regression: `jpeg_read_header` must implement libjpeg.txt §6's
//! tables-only abbreviated-datastream contract.
//!
//! Fixture: a hand-crafted SOI + DQT + DHT + EOI blob — the smallest
//! syntactically-valid tables-only JPEG. The shim's `detect_tables_only`
//! walks markers without descending into table semantics (it doesn't
//! validate that DQT precision / destination is sensible, only that the
//! marker stream contains no SOF/SOS), so a structurally-valid blob is
//! sufficient to exercise both contract branches:
//!
//!   * `jpeg_read_header(cinfo, require_image=FALSE)` returns
//!     `JPEG_HEADER_TABLES_ONLY` (= 2). The cached prefix can then be
//!     spliced in front of subsequent strip data.
//!   * `jpeg_read_header(cinfo, require_image=TRUE)` invokes
//!     `cinfo->err->error_exit(cinfo)` with `msg_code = JERR_NO_IMAGE`
//!     (upstream code 53 at `JPEG_LIB_VERSION=80`, verified empirically
//!     via `cc -DJPEG_LIB_VERSION=80 -I references/libjpeg-turbo/src`)
//!     and never returns `JPEG_HEADER_TABLES_ONLY`.
//!
//! TDD-verified. Removing the `if require_image != 0 { … invoke_error_exit
//! (cinfo, 53); … }` block in `jpeg_read_header` makes
//! `tables_only_with_require_image_true_invokes_jerr_no_image`
//! red-fail with `error_exit fired 0 times, expected 1`.

use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;
use std::ffi::c_int;
use std::os::raw::{c_long, c_void};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use libloading::Library;

#[path = "support/cdylib.rs"]
mod cdylib_support;

const JMSG_STR_PARM_MAX: usize = 80;

#[repr(C)]
struct JpegErrorMgrLayout {
    error_exit: Option<unsafe extern "C" fn(*mut c_void)>,
    emit_message: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
    output_message: Option<unsafe extern "C" fn(*mut c_void)>,
    format_message: Option<unsafe extern "C" fn(*mut c_void, *mut u8)>,
    reset_error_mgr: Option<unsafe extern "C" fn(*mut c_void)>,
    msg_code: c_int,
    msg_parm: [u8; JMSG_STR_PARM_MAX],
    trace_level: c_int,
    num_warnings: c_long,
    jpeg_message_table: *const *const u8,
    last_jpeg_message: c_int,
    addon_message_table: *const *const u8,
    first_addon_message: c_int,
    last_addon_message: c_int,
}

fn cdylib_path() -> std::path::PathBuf {
    cdylib_support::cdylib_path()
}

/// Hand-crafted tables-only JPEG: SOI + minimal DQT + minimal DHT + EOI.
/// The DQT/DHT contents pass `detect_tables_only`'s structural check
/// (which only walks marker headers, not table semantics).
fn build_tables_only_blob() -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    // SOI.
    v.extend_from_slice(&[0xFF, 0xD8]);
    // DQT: 2 (len bytes) + 1 (Pq/Tq) + 64 (8-bit quant values) = 67 = 0x43.
    v.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    v.extend(std::iter::repeat_n(16u8, 64));
    // DHT: 2 (len bytes) + 1 (Tc/Th) + 16 (bits[1..17]) + 12 (huffvals) = 31 = 0x1F.
    v.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x00]);
    // bits[1..17] sums to 12 (one DC code at length 1, six at length 3,
    // five at length 4 = 12). Any layout summing < 256 works for the
    // structural walker.
    v.extend_from_slice(&[
        0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);
    // 12 huffval entries (one per total bits-count above).
    v.extend_from_slice(&[
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    ]);
    // EOI.
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
}

// Per-test atomics + per-test callbacks so cargo test's parallel runner
// cannot race the test paths through a shared callback.
static GOT_FALSE_ERROR_EXIT: AtomicUsize = AtomicUsize::new(0);
static GOT_FALSE_MSG_CODE: AtomicI32 = AtomicI32::new(0);
static GOT_TRUE_ERROR_EXIT: AtomicUsize = AtomicUsize::new(0);
static GOT_TRUE_MSG_CODE: AtomicI32 = AtomicI32::new(0);
static GOT_CONSUME_ERROR_EXIT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn track_false_error_exit(cinfo: *mut c_void) {
    GOT_FALSE_ERROR_EXIT.fetch_add(1, Ordering::SeqCst);
    if !cinfo.is_null() {
        let err_pp: *const *mut JpegErrorMgrLayout = cinfo as *const *mut JpegErrorMgrLayout;
        let err_ptr: *mut JpegErrorMgrLayout = err_pp.read();
        if !err_ptr.is_null() {
            GOT_FALSE_MSG_CODE.store((*err_ptr).msg_code, Ordering::SeqCst);
        }
    }
}

unsafe extern "C" fn track_true_error_exit(cinfo: *mut c_void) {
    GOT_TRUE_ERROR_EXIT.fetch_add(1, Ordering::SeqCst);
    if !cinfo.is_null() {
        let err_pp: *const *mut JpegErrorMgrLayout = cinfo as *const *mut JpegErrorMgrLayout;
        let err_ptr: *mut JpegErrorMgrLayout = err_pp.read();
        if !err_ptr.is_null() {
            GOT_TRUE_MSG_CODE.store((*err_ptr).msg_code, Ordering::SeqCst);
        }
    }
}

unsafe extern "C" fn track_consume_error_exit(_cinfo: *mut c_void) {
    GOT_CONSUME_ERROR_EXIT.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn tables_only_with_require_image_false_returns_header_tables_only() {
    let lib: Library = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    type CreateFn = unsafe extern "C" fn(*mut c_void, c_int, usize);
    type DestroyFn = unsafe extern "C" fn(*mut c_void);
    type StdErrorFn = unsafe extern "C" fn(*mut JpegErrorMgrLayout) -> *mut JpegErrorMgrLayout;
    type MemSrcFn = unsafe extern "C" fn(*mut c_void, *const u8, usize);
    type ReadHeaderFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;

    let create: libloading::Symbol<CreateFn> =
        unsafe { lib.get(b"jpeg_CreateDecompress").unwrap() };
    let destroy: libloading::Symbol<DestroyFn> =
        unsafe { lib.get(b"jpeg_destroy_decompress").unwrap() };
    let std_error: libloading::Symbol<StdErrorFn> = unsafe { lib.get(b"jpeg_std_error").unwrap() };
    let mem_src: libloading::Symbol<MemSrcFn> = unsafe { lib.get(b"jpeg_mem_src").unwrap() };
    let read_header: libloading::Symbol<ReadHeaderFn> =
        unsafe { lib.get(b"jpeg_read_header").unwrap() };

    GOT_FALSE_ERROR_EXIT.store(0, Ordering::SeqCst);
    GOT_FALSE_MSG_CODE.store(0, Ordering::SeqCst);

    let mut err: JpegErrorMgrLayout = unsafe { std::mem::zeroed() };
    unsafe { std_error(&mut err as *mut _) };
    err.error_exit = Some(track_false_error_exit);

    // Allocate cinfo as a 4 KiB buffer (well above the JpegDecompressPublic
    // size). Initialize the err slot at offset 0; the rest is initialised
    // by jpeg_CreateDecompress.
    // P4-110: a `[u8; N]` is only byte-aligned, so casting it to a
    // `j_decompress_ptr` was undefined regardless of size. Naming the mirrored
    // struct fixes the alignment and gives the exact size the guard requires.
    let mut cinfo_buf: std::mem::MaybeUninit<JpegDecompressPublic> =
        std::mem::MaybeUninit::zeroed();
    unsafe {
        let err_slot: *mut *mut JpegErrorMgrLayout =
            cinfo_buf.as_mut_ptr() as *mut *mut JpegErrorMgrLayout;
        err_slot.write(&mut err as *mut _);
    }
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    // JPEG_LIB_VERSION = 80 and the *exact* struct size: since P4-110 the
    // guard requires equality, not "at least as large". The backing allocation
    // may be bigger than the declaration; the declaration may not.
    unsafe { create(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>()) };

    let blob: Vec<u8> = build_tables_only_blob();
    unsafe { mem_src(cinfo_ptr, blob.as_ptr(), blob.len()) };

    // require_image = FALSE (= 0).
    let ret: c_int = unsafe { read_header(cinfo_ptr, 0) };

    assert_eq!(
        ret, 2,
        "expected JPEG_HEADER_TABLES_ONLY (= 2), got {}",
        ret
    );
    assert_eq!(
        GOT_FALSE_ERROR_EXIT.load(Ordering::SeqCst),
        0,
        "error_exit must NOT fire on require_image=FALSE — got {} invocations",
        GOT_FALSE_ERROR_EXIT.load(Ordering::SeqCst),
    );

    unsafe { destroy(cinfo_ptr) };
}

#[test]
fn tables_only_with_require_image_true_invokes_jerr_no_image() {
    let lib: Library = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    type CreateFn = unsafe extern "C" fn(*mut c_void, c_int, usize);
    type DestroyFn = unsafe extern "C" fn(*mut c_void);
    type StdErrorFn = unsafe extern "C" fn(*mut JpegErrorMgrLayout) -> *mut JpegErrorMgrLayout;
    type MemSrcFn = unsafe extern "C" fn(*mut c_void, *const u8, usize);
    type ReadHeaderFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;

    let create: libloading::Symbol<CreateFn> =
        unsafe { lib.get(b"jpeg_CreateDecompress").unwrap() };
    let destroy: libloading::Symbol<DestroyFn> =
        unsafe { lib.get(b"jpeg_destroy_decompress").unwrap() };
    let std_error: libloading::Symbol<StdErrorFn> = unsafe { lib.get(b"jpeg_std_error").unwrap() };
    let mem_src: libloading::Symbol<MemSrcFn> = unsafe { lib.get(b"jpeg_mem_src").unwrap() };
    let read_header: libloading::Symbol<ReadHeaderFn> =
        unsafe { lib.get(b"jpeg_read_header").unwrap() };

    GOT_TRUE_ERROR_EXIT.store(0, Ordering::SeqCst);
    GOT_TRUE_MSG_CODE.store(0, Ordering::SeqCst);

    let mut err: JpegErrorMgrLayout = unsafe { std::mem::zeroed() };
    unsafe { std_error(&mut err as *mut _) };
    err.error_exit = Some(track_true_error_exit);

    // P4-110: a `[u8; N]` is only byte-aligned, so casting it to a
    // `j_decompress_ptr` was undefined regardless of size. Naming the mirrored
    // struct fixes the alignment and gives the exact size the guard requires.
    let mut cinfo_buf: std::mem::MaybeUninit<JpegDecompressPublic> =
        std::mem::MaybeUninit::zeroed();
    unsafe {
        let err_slot: *mut *mut JpegErrorMgrLayout =
            cinfo_buf.as_mut_ptr() as *mut *mut JpegErrorMgrLayout;
        err_slot.write(&mut err as *mut _);
    }
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    unsafe { create(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>()) };

    let blob: Vec<u8> = build_tables_only_blob();
    unsafe { mem_src(cinfo_ptr, blob.as_ptr(), blob.len()) };

    // require_image = TRUE (= 1).
    let _ret: c_int = unsafe { read_header(cinfo_ptr, 1) };

    assert_eq!(
        GOT_TRUE_ERROR_EXIT.load(Ordering::SeqCst),
        1,
        "error_exit must be invoked exactly once on require_image=TRUE + tables-only \
         input — got {} invocations",
        GOT_TRUE_ERROR_EXIT.load(Ordering::SeqCst),
    );
    assert_eq!(
        GOT_TRUE_MSG_CODE.load(Ordering::SeqCst),
        53,
        "msg_code = {}, expected upstream JERR_NO_IMAGE (= 53 at JPEG_LIB_VERSION=80, \
         verified empirically via cc -DJPEG_LIB_VERSION=80 against jerror.h)",
        GOT_TRUE_MSG_CODE.load(Ordering::SeqCst),
    );

    unsafe { destroy(cinfo_ptr) };
}

/// Direct `jpeg_consume_input` on a tables-only stream must return
/// `JPEG_REACHED_EOI` (= 2), NOT invoke `error_exit` with `JERR_NO_IMAGE`.
/// Stock libjpeg's `jpeg_consume_input` (jdapimin.c) accepts a tables-only
/// abbreviated datastream as a valid input — the `require_image` semantics
/// only apply when the *public* `jpeg_read_header` is called by the
/// consumer. The shim's internal call from `jpeg_consume_input` to
/// `jpeg_read_header` must therefore pass `require_image=FALSE` so the
/// tables-only branch returns `JPEG_HEADER_TABLES_ONLY`, which the
/// `jpeg_consume_input` match arm maps to `JPEG_REACHED_EOI`.
///
/// Pre-fix history: codex stop-time review on the initial public-API
/// `require_image=TRUE` wiring caught this regression because the
/// internal call had been hard-coded to `require_image=1`. The fix
/// rewrites that call to pass `0`.
#[test]
fn jpeg_consume_input_on_tables_only_returns_reached_eoi() {
    let lib: Library = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    type CreateFn = unsafe extern "C" fn(*mut c_void, c_int, usize);
    type DestroyFn = unsafe extern "C" fn(*mut c_void);
    type StdErrorFn = unsafe extern "C" fn(*mut JpegErrorMgrLayout) -> *mut JpegErrorMgrLayout;
    type MemSrcFn = unsafe extern "C" fn(*mut c_void, *const u8, usize);
    type ConsumeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
    type InputCompleteFn = unsafe extern "C" fn(*mut c_void) -> c_int;

    let create: libloading::Symbol<CreateFn> =
        unsafe { lib.get(b"jpeg_CreateDecompress").unwrap() };
    let destroy: libloading::Symbol<DestroyFn> =
        unsafe { lib.get(b"jpeg_destroy_decompress").unwrap() };
    let std_error: libloading::Symbol<StdErrorFn> = unsafe { lib.get(b"jpeg_std_error").unwrap() };
    let mem_src: libloading::Symbol<MemSrcFn> = unsafe { lib.get(b"jpeg_mem_src").unwrap() };
    let consume_input: libloading::Symbol<ConsumeFn> =
        unsafe { lib.get(b"jpeg_consume_input").unwrap() };
    let input_complete: libloading::Symbol<InputCompleteFn> =
        unsafe { lib.get(b"jpeg_input_complete").unwrap() };

    GOT_CONSUME_ERROR_EXIT.store(0, Ordering::SeqCst);

    let mut err: JpegErrorMgrLayout = unsafe { std::mem::zeroed() };
    unsafe { std_error(&mut err as *mut _) };
    err.error_exit = Some(track_consume_error_exit);

    // P4-110: a `[u8; N]` is only byte-aligned, so casting it to a
    // `j_decompress_ptr` was undefined regardless of size. Naming the mirrored
    // struct fixes the alignment and gives the exact size the guard requires.
    let mut cinfo_buf: std::mem::MaybeUninit<JpegDecompressPublic> =
        std::mem::MaybeUninit::zeroed();
    unsafe {
        let err_slot: *mut *mut JpegErrorMgrLayout =
            cinfo_buf.as_mut_ptr() as *mut *mut JpegErrorMgrLayout;
        err_slot.write(&mut err as *mut _);
    }
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    unsafe { create(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>()) };

    let blob: Vec<u8> = build_tables_only_blob();
    unsafe { mem_src(cinfo_ptr, blob.as_ptr(), blob.len()) };

    let ret: c_int = unsafe { consume_input(cinfo_ptr) };

    // JPEG_REACHED_EOI = 2.
    assert_eq!(
        ret, 2,
        "expected JPEG_REACHED_EOI (= 2), got {} — the internal jpeg_read_header \
         call from jpeg_consume_input must pass require_image=FALSE so a \
         tables-only blob does not trigger JERR_NO_IMAGE",
        ret,
    );
    assert_eq!(
        GOT_CONSUME_ERROR_EXIT.load(Ordering::SeqCst),
        0,
        "error_exit must NOT fire from jpeg_consume_input on a tables-only \
         input — got {} invocations",
        GOT_CONSUME_ERROR_EXIT.load(Ordering::SeqCst),
    );

    // The documented buffered-image polling loop is
    //     while (!jpeg_input_complete(&cinfo))
    //         (void) jpeg_consume_input(&cinfo);
    // After consume_input returns JPEG_REACHED_EOI on a tables-only
    // stream, the next jpeg_input_complete must return TRUE so the
    // loop terminates. The shim mirrors stock libjpeg's
    // `inputctl->eoi_reached = TRUE` by advancing
    // `cinfo.global_state` past `DSTATE_SCANNING` in the
    // `JPEG_HEADER_TABLES_ONLY` arm of `jpeg_consume_input`.
    let complete: c_int = unsafe { input_complete(cinfo_ptr) };
    assert_ne!(
        complete, 0,
        "jpeg_input_complete must return TRUE after a tables-only EOI return \
         from jpeg_consume_input; otherwise the documented `while \
         (!jpeg_input_complete) jpeg_consume_input(...)` polling loop \
         spins forever"
    );

    unsafe { destroy(cinfo_ptr) };
}
