//! A1-8: consolidated test for the error/allocation helpers
//! `tj3GetErrorStr`, `tj3GetErrorCode`, `tj3Alloc`, `tj3Free`.
//!
//! These symbols are already introduced by A1-2 (errors) and A1-3
//! (allocator); this file adds an end-to-end contract test covering
//! their interaction as a group so the subtask has its own coverage.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::PathBuf;

const TJINIT_COMPRESS: c_int = 1;

fn dlext() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}
fn lib_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}
fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf =
            dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dlext()));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate cdylib near {}", exe.display());
}

#[test]
fn tj3_alloc_free_round_trips_through_malloc() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_alloc: libloading::Symbol<unsafe extern "C" fn(usize) -> *mut c_void> =
            lib.get(b"tj3Alloc").expect("tj3Alloc");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        // Allocate a 4 KiB buffer, write to every byte, free.
        let p: *mut c_void = tj3_alloc(4096);
        assert!(!p.is_null());
        let slice: &mut [u8] = std::slice::from_raw_parts_mut(p as *mut u8, 4096);
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        // Read back a sentinel to confirm writes landed.
        assert_eq!(slice[0], 0);
        assert_eq!(slice[255], 255);
        assert_eq!(slice[256], 0);

        tj3_free(p);

        // Zero-size allocation returns NULL (documented behavior).
        assert!(tj3_alloc(0).is_null());

        // Freeing NULL is a no-op.
        tj3_free(std::ptr::null_mut());
    }
}

#[test]
fn tj3_get_error_str_and_code_reflect_last_call() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set");
        let tj3_err: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *const c_char> =
            lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
        let tj3_code: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"tj3GetErrorCode").expect("tj3GetErrorCode");

        let h = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());

        // Fresh handle: "No error" / code 0.
        let msg: &str = CStr::from_ptr(tj3_err(h)).to_str().expect("utf8");
        assert_eq!(msg, "No error");
        assert_eq!(tj3_code(h), 0);

        // Trigger a validation failure (quality out of range).
        assert_eq!(tj3_set(h, 3 /* QUALITY */, 999), -1);
        let msg2: &str = CStr::from_ptr(tj3_err(h)).to_str().expect("utf8");
        assert!(
            msg2.contains("quality"),
            "expected quality error, got {msg2}"
        );
        assert_eq!(tj3_code(h), 1 /* TJERR_FATAL */);

        // Recover with a valid value; error should clear.
        assert_eq!(tj3_set(h, 3, 50), 0);
        let msg3: &str = CStr::from_ptr(tj3_err(h)).to_str().expect("utf8");
        assert_eq!(msg3, "No error");
        assert_eq!(tj3_code(h), 0);

        // NULL handle: returns a global fallback string / fatal code.
        let msg_null: &str = CStr::from_ptr(tj3_err(std::ptr::null_mut()))
            .to_str()
            .expect("utf8");
        assert!(!msg_null.is_empty());
        assert_eq!(tj3_code(std::ptr::null_mut()), 1);

        tj3_destroy(h);
    }
}
