//! P4-120 (#467): the classic shim's allocation-failure paths are reachable.
//!
//! `JERR_OUT_OF_MEMORY` (56) is raised on two `jpeg_mem_dest` paths that no
//! test could force before: the initial buffer allocation when the caller
//! passes an empty slot (`jdatadst.c:271`, `ERREXIT1(…, 10)`), and the
//! doubling growth in `empty_mem_output_buffer` (`jdatadst.c:132`, the same
//! `ERREXIT1(…, 10)`). Everything funnels through `libc_malloc`, so
//! `fail_nth_allocation_for_tests` — a thread-local countdown armed only
//! here — makes the exact allocation fail the way a real allocator under
//! pressure would.
//!
//! There is no C oracle for these lines: no portable way exists to make
//! stock libjpeg's `malloc` fail on the Nth call. What is compared against
//! upstream instead is the *contract*: the code (56) and the `ERREXIT1`
//! payload (`msg_parm.i[0] == 10`) are read from `jdatadst.c` and pinned
//! here; the message rendering for the code is covered by the P4-146
//! whole-table gate.

use std::cell::Cell;
use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::alloc::{
    disarm_allocation_failure_for_tests, fail_nth_allocation_for_tests,
};
use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_destroy_compress, jpeg_finish_compress, jpeg_mem_dest, jpeg_set_defaults,
    jpeg_start_compress, jpeg_std_error, jpeg_write_scanlines, JpegCompressPublic, JpegErrorMgr,
};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
/// `jpeglib.h`: `JCS_GRAYSCALE`.
const JCS_GRAYSCALE: c_int = 1;
/// `jerror.h` at v8: `JERR_OUT_OF_MEMORY`.
const JERR_OUT_OF_MEMORY: c_int = 56;
/// `jdatadst.c:271` / `:132`: `ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10)`.
const OOM_CASE_MEM_DEST: c_int = 10;

thread_local! {
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static MSG_CODE: Cell<c_int> = const { Cell::new(0) };
    static PARM0: Cell<c_int> = const { Cell::new(0) };
}

unsafe extern "C" fn recording_error_exit(cinfo: *mut c_void) {
    // SAFETY: the error path reads only the leading `err` pointer.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
        if FIRED.with(|f| f.get()) {
            return;
        }
        FIRED.with(|f| f.set(true));
        MSG_CODE.with(|c| c.set(std::ptr::addr_of!((*err_ptr).msg_code).read()));
        let parm: *const u8 = std::ptr::addr_of!((*err_ptr).msg_parm) as *const u8;
        let mut bytes: [u8; std::mem::size_of::<c_int>()] = Default::default();
        std::ptr::copy_nonoverlapping(parm, bytes.as_mut_ptr(), bytes.len());
        PARM0.with(|p| p.set(c_int::from_ne_bytes(bytes)));
    }
}

fn reset_recorder() {
    disarm_allocation_failure_for_tests();
    FIRED.with(|f| f.set(false));
    MSG_CODE.with(|c| c.set(0));
    PARM0.with(|p| p.set(0));
}

/// A compress object with a recording error handler, up to (not including)
/// the destination.
fn compress_scaffold() -> (Box<JpegCompressPublic>, Box<JpegErrorMgr>, *mut c_void) {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: *mut c_void = &mut *cinfo as *mut JpegCompressPublic as *mut c_void;
    // SAFETY: zeroed mirror, classic setup sequence.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err);
        (*errp).error_exit = Some(recording_error_exit);
        cinfo.err = errp;
        libjpeg_turbo_rs_capi::jpeglib::jpeg_create_compress(cinfo_ptr);
    }
    (cinfo, err, cinfo_ptr)
}

/// `jdatadst.c:271`: the empty-slot initial allocation fails → 56, case 10,
/// raised synchronously from `jpeg_mem_dest` itself.
#[test]
fn mem_dest_initial_allocation_failure_reports_oom_case_10() {
    reset_recorder();
    let (_cinfo, _err, cinfo_ptr) = compress_scaffold();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: std::ffi::c_ulong = 0;

    fail_nth_allocation_for_tests(0);
    // SAFETY: valid out-params; the injected failure hits the initial
    // OUTPUT_BUF_SIZE allocation for the empty slot.
    unsafe { jpeg_mem_dest(cinfo_ptr, &mut buf, &mut size) };
    disarm_allocation_failure_for_tests();

    assert!(FIRED.with(|f| f.get()), "error_exit must fire");
    assert_eq!(MSG_CODE.with(|c| c.get()), JERR_OUT_OF_MEMORY);
    assert_eq!(
        PARM0.with(|p| p.get()),
        OOM_CASE_MEM_DEST,
        "ERREXIT1's payload must survive to msg_parm.i[0]"
    );
    // SAFETY: destroy is null-safe for the partially-initialised object.
    unsafe { jpeg_destroy_compress(cinfo_ptr) };
}

/// `jdatadst.c:132`: the doubling growth fails mid-encode → 56, case 10,
/// surfacing through the deferred-encode flush at `jpeg_finish_compress`.
#[test]
fn mem_dest_growth_failure_reports_oom_case_10() {
    reset_recorder();
    let (mut cinfo, _err, cinfo_ptr) = compress_scaffold();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: std::ffi::c_ulong = 0;

    // SAFETY: classic sequence; incompressible noise at 128x128 forces well
    // past the 4096-byte initial buffer so the growth path must run.
    const BIG: usize = 128;
    unsafe {
        jpeg_mem_dest(cinfo_ptr, &mut buf, &mut size);
        cinfo.image_width = BIG as u32;
        cinfo.image_height = BIG as u32;
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
        jpeg_set_defaults(cinfo_ptr);
        jpeg_start_compress(cinfo_ptr, 1);
        let mut row: [u8; BIG] = [0; BIG];
        let mut seed: u32 = 0x1234_5678;
        for _ in 0..BIG {
            for b in row.iter_mut() {
                // xorshift noise — resists compression, guaranteeing growth.
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *b = (seed & 0xFF) as u8;
            }
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            let _ = jpeg_write_scanlines(cinfo_ptr, rows.as_mut_ptr(), 1);
        }
        // Arm now: the deferred encode flushes at finish, and the first
        // allocation it makes is the growth of the 4096-byte initial buffer.
        fail_nth_allocation_for_tests(0);
        jpeg_finish_compress(cinfo_ptr);
        disarm_allocation_failure_for_tests();
    }

    assert!(FIRED.with(|f| f.get()), "error_exit must fire at the flush");
    assert_eq!(MSG_CODE.with(|c| c.get()), JERR_OUT_OF_MEMORY);
    assert_eq!(
        PARM0.with(|p| p.get()),
        OOM_CASE_MEM_DEST,
        "ERREXIT1's payload must survive to msg_parm.i[0]"
    );
    // SAFETY: destroyed once; the manager's own buffer is freed by destroy.
    unsafe { jpeg_destroy_compress(cinfo_ptr) };
}

/// The disarmed hook is inert: the identical sequence succeeds, proving the
/// failures above came from the injection and not from the sequence.
#[test]
fn disarmed_injection_changes_nothing() {
    reset_recorder();
    let (mut cinfo, _err, cinfo_ptr) = compress_scaffold();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: std::ffi::c_ulong = 0;

    // SAFETY: as above, without arming.
    unsafe {
        jpeg_mem_dest(cinfo_ptr, &mut buf, &mut size);
        cinfo.image_width = WIDTH as u32;
        cinfo.image_height = HEIGHT as u32;
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
        jpeg_set_defaults(cinfo_ptr);
        jpeg_start_compress(cinfo_ptr, 1);
        let mut row: [u8; WIDTH] = [128; WIDTH];
        for _ in 0..HEIGHT {
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            let _ = jpeg_write_scanlines(cinfo_ptr, rows.as_mut_ptr(), 1);
        }
        jpeg_finish_compress(cinfo_ptr);
    }
    assert!(!FIRED.with(|f| f.get()), "nothing may fire when disarmed");
    assert!(size > 0, "the encode must have produced output");
    // SAFETY: destroyed once; `buf` is the manager's allocation.
    unsafe {
        jpeg_destroy_compress(cinfo_ptr);
        extern "C" {
            fn free(p: *mut c_void);
        }
        free(buf as *mut c_void);
    }
}
