//! P3-2 regression: `jpeg12_read_raw_data` and `jpeg12_write_raw_data`
//! must invoke `cinfo->err->error_exit(cinfo)` with
//! `msg_code = JERR_NOTIMPL` (upstream code 19) instead of silently
//! returning 0.
//!
//! Pre-fix history. Both stubs set `priv_state.last_error` and
//! returned 0. The 0-return mimicked "no rows ready, retry later"
//! and could spin a caller forever — the libjpeg.txt §3 contract
//! for an unimplemented codepath is to route through `error_exit`
//! so a `setjmp`/`longjmp` consumer recovers cleanly.
//!
//! Test design. Each test dlopens the cdylib, installs a custom
//! `error_exit` on a synthesised `JpegErrorMgr`, points a stack-
//! allocated pseudo-cinfo's `err` slot (offset 0) at that mgr, and
//! calls the function under test with NULL data + 0 lines. The
//! custom handler increments a per-test atomic and records
//! `msg_code`; the test then asserts the handler fired exactly once
//! with code 19 and the function fell through to `0` (the
//! defensive return-when-handler-returns path).
//!
//! TDD-verified. Removing the `invoke_error_exit(cinfo, 19)` line
//! in either `jpeg12_*_raw_data` makes the corresponding test
//! red-fail with `error_exit fired 0 times, expected 1`. Restoring
//! the line returns to GREEN.

use std::ffi::c_int;
use std::os::raw::{c_long, c_void};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use libloading::Library;

const JMSG_STR_PARM_MAX: usize = 80;

/// Byte-exact mirror of `struct jpeg_error_mgr` (JPEG_LIB_VERSION=80
/// LP64). Only the fields the shim reads/writes are typed; the
/// trailing message-table fields are kept here so the layout matches
/// what `jpeg_std_error` initialises.
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
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let candidates = [
        workspace_root.join("target/release/liblibjpeg_turbo_rs_capi.dylib"),
        workspace_root.join("target/release/liblibjpeg_turbo_rs_capi.so"),
        workspace_root.join("target/release/libjpeg_turbo_rs_capi.dll"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "libjpeg-turbo-rs-capi", "--release"])
        .current_dir(&workspace_root)
        .status()
        .expect("cargo build");
    assert!(status.success(), "cargo build failed");
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    panic!("cdylib not found after build");
}

// Per-test atomics (read vs write) so cargo test's default parallel
// runner cannot race the two cases through a shared callback.
static GOT_READ_ERROR_EXIT: AtomicUsize = AtomicUsize::new(0);
static GOT_READ_MSG_CODE: AtomicI32 = AtomicI32::new(0);
static GOT_WRITE_ERROR_EXIT: AtomicUsize = AtomicUsize::new(0);
static GOT_WRITE_MSG_CODE: AtomicI32 = AtomicI32::new(0);

/// Custom `error_exit` for the read test. Increments the read
/// counter, captures `msg_code`, then returns to the caller (the
/// function under test handles the contract-violating
/// "handler-returns" case by falling through to `0`).
unsafe extern "C" fn track_read_error_exit(cinfo: *mut c_void) {
    GOT_READ_ERROR_EXIT.fetch_add(1, Ordering::SeqCst);
    if !cinfo.is_null() {
        let err_pp: *const *mut JpegErrorMgrLayout = cinfo as *const *mut JpegErrorMgrLayout;
        let err_ptr: *mut JpegErrorMgrLayout = err_pp.read();
        if !err_ptr.is_null() {
            let code: c_int = (*err_ptr).msg_code;
            GOT_READ_MSG_CODE.store(code, Ordering::SeqCst);
        }
    }
}

unsafe extern "C" fn track_write_error_exit(cinfo: *mut c_void) {
    GOT_WRITE_ERROR_EXIT.fetch_add(1, Ordering::SeqCst);
    if !cinfo.is_null() {
        let err_pp: *const *mut JpegErrorMgrLayout = cinfo as *const *mut JpegErrorMgrLayout;
        let err_ptr: *mut JpegErrorMgrLayout = err_pp.read();
        if !err_ptr.is_null() {
            let code: c_int = (*err_ptr).msg_code;
            GOT_WRITE_MSG_CODE.store(code, Ordering::SeqCst);
        }
    }
}

#[test]
fn jpeg12_read_raw_data_invokes_error_exit_with_jerr_notimpl() {
    let lib: Library = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    let std_error: libloading::Symbol<
        unsafe extern "C" fn(*mut JpegErrorMgrLayout) -> *mut JpegErrorMgrLayout,
    > = unsafe { lib.get(b"jpeg_std_error").unwrap() };
    let jpeg12_read: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut *mut i16, u32) -> u32,
    > = unsafe { lib.get(b"jpeg12_read_raw_data").unwrap() };

    GOT_READ_ERROR_EXIT.store(0, Ordering::SeqCst);
    GOT_READ_MSG_CODE.store(0, Ordering::SeqCst);

    let mut err: JpegErrorMgrLayout = unsafe { std::mem::zeroed() };
    unsafe { std_error(&mut err as *mut _) };
    err.error_exit = Some(track_read_error_exit);

    // Pseudo-cinfo. The shim's `cinfo_mut` cast lands on this buffer
    // unchecked; only the `err` slot at offset 0 needs a valid
    // pointer for the `invoke_error_exit` walk. The function under
    // test never reads further into the struct because:
    //   * `decompress_private_raw` returns NULL (no thread-local entry).
    //   * `priv_from_ptr(NULL)` returns None — the if-let is skipped.
    //   * Control flows directly to `invoke_error_exit(cinfo, 19)`.
    // 1024 bytes is comfortably larger than `JpegDecompressPublic`.
    let mut cinfo_buf: [u8; 1024] = [0u8; 1024];
    unsafe {
        let err_slot: *mut *mut JpegErrorMgrLayout =
            cinfo_buf.as_mut_ptr() as *mut *mut JpegErrorMgrLayout;
        err_slot.write(&mut err as *mut _);
    }
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    let ret: u32 = unsafe { jpeg12_read(cinfo_ptr, std::ptr::null_mut(), 0) };

    assert_eq!(
        GOT_READ_ERROR_EXIT.load(Ordering::SeqCst),
        1,
        "error_exit fired {} times, expected 1",
        GOT_READ_ERROR_EXIT.load(Ordering::SeqCst),
    );
    assert_eq!(
        GOT_READ_MSG_CODE.load(Ordering::SeqCst),
        19,
        "msg_code = {}, expected upstream JERR_NOTIMPL (19)",
        GOT_READ_MSG_CODE.load(Ordering::SeqCst),
    );
    assert_eq!(
        ret, 0,
        "jpeg12_read_raw_data falls through to 0 only when error_exit returns"
    );
}

#[test]
fn jpeg12_write_raw_data_invokes_error_exit_with_jerr_notimpl() {
    let lib: Library = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    let std_error: libloading::Symbol<
        unsafe extern "C" fn(*mut JpegErrorMgrLayout) -> *mut JpegErrorMgrLayout,
    > = unsafe { lib.get(b"jpeg_std_error").unwrap() };
    let jpeg12_write: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut *mut i16, u32) -> u32,
    > = unsafe { lib.get(b"jpeg12_write_raw_data").unwrap() };

    GOT_WRITE_ERROR_EXIT.store(0, Ordering::SeqCst);
    GOT_WRITE_MSG_CODE.store(0, Ordering::SeqCst);

    let mut err: JpegErrorMgrLayout = unsafe { std::mem::zeroed() };
    unsafe { std_error(&mut err as *mut _) };
    err.error_exit = Some(track_write_error_exit);

    // Pseudo-cinfo for the compress path. `cinfo_compress_mut` casts
    // to `JpegCompressPublic`; both compress and decompress mirrors
    // share `err` at offset 0 (the `jpeg_common_fields` prefix from
    // upstream `jpeglib.h`). The `master` slot the function reads
    // (`c.master`) lives deeper and stays zeroed → NULL →
    // `priv_compress_from_ptr` returns None → if-let skipped → flow
    // to `invoke_error_exit`. 2048 bytes is comfortably larger than
    // `JpegCompressPublic`.
    let mut cinfo_buf: [u8; 2048] = [0u8; 2048];
    unsafe {
        let err_slot: *mut *mut JpegErrorMgrLayout =
            cinfo_buf.as_mut_ptr() as *mut *mut JpegErrorMgrLayout;
        err_slot.write(&mut err as *mut _);
    }
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    let ret: u32 = unsafe { jpeg12_write(cinfo_ptr, std::ptr::null_mut(), 0) };

    assert_eq!(
        GOT_WRITE_ERROR_EXIT.load(Ordering::SeqCst),
        1,
        "error_exit fired {} times, expected 1",
        GOT_WRITE_ERROR_EXIT.load(Ordering::SeqCst),
    );
    assert_eq!(
        GOT_WRITE_MSG_CODE.load(Ordering::SeqCst),
        19,
        "msg_code = {}, expected upstream JERR_NOTIMPL (19)",
        GOT_WRITE_MSG_CODE.load(Ordering::SeqCst),
    );
    assert_eq!(
        ret, 0,
        "jpeg12_write_raw_data falls through to 0 only when error_exit returns"
    );
}
