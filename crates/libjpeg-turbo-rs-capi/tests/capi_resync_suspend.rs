//! P4-97 (#469): `jpeg_resync_to_restart` against a genuinely suspending
//! source manager, trace-compared verbatim against stock libjpeg.
//!
//! The upstream contracts under test (`jdmarker.c` `resync_to_restart` /
//! `next_marker`, libjpeg.txt "Decompression suspension", measured by
//! `examples/classic_resync_suspend_oracle.c` against stock 3.1.4.1):
//!
//! * the decision table over `unread_marker` (desired RST / prior RST /
//!   next-two RSTs / valid non-RST / invalid byte);
//! * `JWRN_MUST_RESYNC` once per call, ahead of the decision loop, and
//!   `JTRC_RECOVERY_ACTION` per chosen action — so a call that scans
//!   forward and re-decides emits the trace more than once — both
//!   publishing `msg_code`/`msg_parm` regardless of `trace_level`;
//! * a `FALSE` fill is a `FALSE` resync return with the source fields at
//!   the last committed restart point — the refill re-presents the
//!   unconsumed tail (`bib` in the fill trace pins the sync discipline);
//! * `discarded_bytes` survives suspension, so the eventual
//!   `JWRN_EXTRANEOUS_DATA` reports the bytes discarded across *all* the
//!   calls of one scan (stock keeps it in the private marker state,
//!   `jpegint.h:395`; scenario s4's `p0 8`).

use std::cell::RefCell;
use std::ffi::{c_int, c_long, c_void};
use std::fmt::Write as _;

mod helpers;

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_destroy_decompress, jpeg_resync_to_restart, jpeg_std_error, JpegDecompressPublic,
    JpegErrorMgr, JpegSourceMgr,
};

/// `boolean` in the C header; the shim keeps its alias private.
type CBoolean = c_int;

thread_local! {
    static TRACE: RefCell<String> = const { RefCell::new(String::new()) };
    static NAME: RefCell<String> = const { RefCell::new(String::new()) };
}

fn trace_line(line: String) {
    TRACE.with(|t| {
        let mut t = t.borrow_mut();
        let name = NAME.with(|n| n.borrow().clone());
        writeln!(t, "{name} {line}").expect("write trace");
    });
}

/// Mirrors the oracle's `record_emit`: every `emit_message` publishes
/// `msg_code` and the first two int parms of the `msg_parm` union.
unsafe extern "C" fn record_emit(cinfo: *mut c_void, msg_level: c_int) {
    // SAFETY: the error path reads only the leading `err` pointer, exactly
    // like stock's j_common_ptr view.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
        let msg_code: c_int = std::ptr::addr_of!((*err_ptr).msg_code).read();
        let parm: *const u8 = std::ptr::addr_of!((*err_ptr).msg_parm).cast::<u8>();
        let p0: c_int = parm.cast::<c_int>().read_unaligned();
        let p1: c_int = parm
            .add(std::mem::size_of::<c_int>())
            .cast::<c_int>()
            .read_unaligned();
        trace_line(format!("msg {msg_code} lvl {msg_level} p0 {p0} p1 {p1}"));
    }
}

unsafe extern "C" fn recording_error_exit(cinfo: *mut c_void) {
    // SAFETY: as `record_emit`.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
        let msg_code: c_int = std::ptr::addr_of!((*err_ptr).msg_code).read();
        trace_line(format!("fatal {msg_code}"));
        panic!("oracle scenarios never reach error_exit (msg_code {msg_code})");
    }
}

unsafe extern "C" fn quiet_output_message(_cinfo: *mut c_void) {}

/// The oracle's suspending source manager: the driver authorizes stream
/// bytes in stages; fill presents `[unconsumed tail + fresh bytes]` or
/// returns `FALSE` when nothing new is staged (libjpeg.txt: the data past
/// the restart point "must NOT be discarded if you suspend").
#[repr(C)]
struct SuspendSrc {
    mgr: JpegSourceMgr,
    stream: *const u8,
    fed: usize,
    stage: usize,
    buf: [u8; 256],
    fill_calls: c_int,
}

unsafe extern "C" fn suspend_init(_cinfo: *mut c_void) {}
unsafe extern "C" fn suspend_skip(_cinfo: *mut c_void, _n: c_long) {}
unsafe extern "C" fn suspend_term(_cinfo: *mut c_void) {}

unsafe extern "C" fn suspend_fill(cinfo: *mut c_void) -> CBoolean {
    // SAFETY: `cinfo` is the live decompress object of the running
    // scenario; its `src` points at the `mgr` field of a `SuspendSrc`
    // (repr(C), first field), so the container cast is layout-sound.
    unsafe {
        let c: *mut JpegDecompressPublic = cinfo as *mut JpegDecompressPublic;
        let s: *mut SuspendSrc = (*c).src as *mut SuspendSrc;
        let tail: usize = (*s).mgr.bytes_in_buffer;
        let fresh: usize = (*s).stage - (*s).fed;

        (*s).fill_calls += 1;
        let calls: c_int = (*s).fill_calls;
        if fresh == 0 {
            trace_line(format!("fill {calls} bib {tail} sus"));
            return 0;
        }
        // Keep the restart-point tail ahead of the new data. The tail is a
        // sub-slice of `buf` (`ptr::copy` = memmove, overlap-safe), and raw
        // pointers throughout — no reference to the raw-deref place.
        let buf_ptr: *mut u8 = (&raw mut (*s).buf).cast::<u8>();
        std::ptr::copy((*s).mgr.next_input_byte, buf_ptr, tail);
        std::ptr::copy_nonoverlapping((*s).stream.add((*s).fed), buf_ptr.add(tail), fresh);
        (*s).fed = (*s).stage;
        (*s).mgr.next_input_byte = buf_ptr;
        (*s).mgr.bytes_in_buffer = tail + fresh;
        trace_line(format!("fill {calls} bib {tail} serve {fresh}"));
        1
    }
}

/// One oracle scenario: create a decompress object, install the suspending
/// manager and the recording error manager, seed `unread_marker`, then
/// drive `jpeg_resync_to_restart` through the staged refills.
fn run_scenario(name: &str, desired: c_int, start_marker: c_int, stream: &[u8], stages: &[usize]) {
    NAME.with(|n| *n.borrow_mut() = name.to_string());

    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    let mut src: Box<SuspendSrc> = Box::new(SuspendSrc {
        mgr: JpegSourceMgr {
            next_input_byte: std::ptr::null(),
            bytes_in_buffer: 0,
            init_source: Some(suspend_init),
            fill_input_buffer: Some(suspend_fill),
            skip_input_data: Some(suspend_skip),
            resync_to_restart: Some(jpeg_resync_to_restart),
            term_source: Some(suspend_term),
        },
        stream: stream.as_ptr(),
        fed: 0,
        stage: stages[0],
        buf: [0; 256],
        fill_calls: 0,
    });

    // Raw pointers are taken ONCE and every later access goes through them —
    // touching the `cinfo` or `SuspendSrc` Box directly after this point would
    // invalidate the raw tags under Stacked Borrows (the shim reads `cinfo`
    // through `cinfo_ptr` on every call, and `suspend_fill` reaches the whole
    // `SuspendSrc` through `cinfo->src`, so both pointers need
    // whole-allocation provenance that outlives every driver-loop mutation).
    // `err` is a separate allocation nothing keeps a raw tag into, so the one
    // `&mut *err` below is harmless.
    let cinfo_typed: *mut JpegDecompressPublic = &raw mut *cinfo;
    let cinfo_ptr: *mut c_void = cinfo_typed.cast::<c_void>();
    let src_ptr: *mut SuspendSrc = &raw mut *src;

    // SAFETY: the classic direct-call harness — a zeroed public mirror
    // initialized by jpeg_CreateDecompress, a Box-owned source manager,
    // and field writes the C oracle performs identically; all through the
    // raw pointers above.
    unsafe {
        (*src_ptr).mgr.next_input_byte = (&raw const (*src_ptr).buf).cast::<u8>();
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err);
        (*errp).error_exit = Some(recording_error_exit);
        (*errp).emit_message = Some(record_emit);
        (*errp).output_message = Some(quiet_output_message);
        (*cinfo_typed).err = errp;
        libjpeg_turbo_rs_capi::jpeglib::jpeg_CreateDecompress(
            cinfo_ptr,
            80,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        (*cinfo_typed).src = &raw mut (*src_ptr).mgr;
        (*cinfo_typed).unread_marker = start_marker;

        let mut step: usize = 0;
        loop {
            let r: CBoolean = jpeg_resync_to_restart(cinfo_ptr, desired);
            trace_line(format!(
                "call {} ret {} um 0x{:02X}",
                step + 1,
                r,
                (*cinfo_typed).unread_marker
            ));
            if r != 0 {
                break;
            }
            step += 1;
            if step >= stages.len() {
                trace_line("exhausted".to_string());
                break;
            }
            (*src_ptr).stage = stages[step];
        }
        trace_line(format!("fills {}", (*src_ptr).fill_calls));
        (*cinfo_typed).src = std::ptr::null_mut();
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

/// The scenario matrix, kept byte-identical to `main()` in
/// `classic_resync_suspend_oracle.c`. Desired restart number 3 (RST3 =
/// 0xD3) throughout.
fn run_all_scenarios() -> String {
    TRACE.with(|t| t.borrow_mut().clear());

    // s1: the desired restart — discard it and resume (action 1).
    run_scenario("s1", 3, 0xD3, &[], &[0]);
    // s2/s3: the next two expected restarts — leave unread (action 3).
    run_scenario("s2", 3, 0xD4, &[], &[0]);
    run_scenario("s3", 3, 0xD5, &[], &[0]);
    // s3b: a valid non-restart marker (EOI) — action 3.
    run_scenario("s3b", 3, 0xD9, &[], &[0]);
    // s4: a prior restart (action 2) scanning through garbage that spans a
    // suspension; JWRN_EXTRANEOUS_DATA on the second call reports all 8.
    run_scenario(
        "s4",
        3,
        0xD2,
        &[0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xFF, 0xD3],
        &[5, 10],
    );
    // s5: an invalid byte (< M_SOF0) scans through a stuffed-zero pair and
    // an FF pad run to EOI — action 3 on the found marker.
    run_scenario("s5", 3, 0x01, &[0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xD9], &[6]);
    // s6: suspension inside an FF pad run — the refill re-presents the two
    // pad FFs (bib 2); the found RST7 is >2 from desired — action 1.
    run_scenario("s6", 3, 0xD1, &[0xFF, 0xFF, 0xFF, 0xD7], &[2, 4]);

    TRACE.with(|t| t.borrow().clone())
}

/// The full suspending-resync trace must match stock verbatim: decision
/// table, message sequence, fill accounting (`bib` = sync discipline), and
/// the suspension-surviving `discarded_bytes` count.
#[test]
fn resync_suspend_trace_matches_stock() {
    let rust_trace: String = run_all_scenarios();

    let Some(oracle) = helpers::build_classic_oracle("classic_resync_suspend_oracle") else {
        eprintln!("SKIP: no stock libjpeg-turbo development install found");
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);

    assert_eq!(
        rust_trace, c_trace,
        "suspending-resync trace diverged from stock libjpeg"
    );
}
