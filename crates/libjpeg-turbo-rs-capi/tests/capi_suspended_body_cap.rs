//! P4-13's 256 MiB suspended-body cap is a *resource limit*, and must be
//! reported as one.
//!
//! The shim will buffer only so much of a suspending source. Until 2026-08-11
//! tripping that bound looked, to every caller-visible signal, exactly like
//! ordinary suspension: `jpeg_consume_input` returned `JPEG_SUSPENDED` and
//! `jpeg_start_decompress` returned FALSE, both of which tell a classic caller
//! "refill and retry". No refill can satisfy a stream that has already exceeded
//! the cap, so a conforming caller loops forever; and because the drain stops
//! short of EOI while clearing its incomplete flag, a caller that instead asked
//! `jpeg_input_complete` was told the truncated stream was fully consumed.
//!
//! Neither answer is available now: the cap raises `JERR_OUT_OF_MEMORY` with a
//! case number outside upstream's `jmemmgr.c` range, so `error_exit` fires and
//! the caller learns that a limit stopped the decode rather than a hiccup in
//! the source.
//!
//! This test really does push more than 256 MiB through the source manager —
//! there is no smaller way to reach the branch, since the bound is a constant.
//! It serves 4 MiB per callback to keep the number of crossings small.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_consume_input, jpeg_destroy_decompress, jpeg_start_decompress,
    jpeg_std_error, JpegDecompressPublic, JpegErrorMgr, JpegSourceMgr,
};

const JPEG_LIB_VERSION: c_int = 80;
/// `jerror.h`: `JERR_OUT_OF_MEMORY`.
const JERR_OUT_OF_MEMORY: c_int = 56;
/// This shim's own case number for the cap, deliberately outside upstream's
/// `jmemmgr.c` range of 1..=10.
const OOM_CASE_SUSPENDED_BODY_CAP: c_int = 100;
/// `jpeglib.h` return codes.
const JPEG_REACHED_SOS: c_int = 1;
const JPEG_REACHED_EOI: c_int = 2;

const MIB: usize = 1024 * 1024;
/// Serve in 4 MiB pieces: ~70 callbacks to cross a 256 MiB bound.
const CHUNK: usize = 4 * MIB;
/// Enough to cross the cap with room to spare.
const TOTAL: usize = 300 * MIB;

/// A source that never ends and never suspends — it keeps producing body bytes
/// until the shim decides it has had enough. That is the only way to reach the
/// cap: a source that suspends or ends takes a different branch.
#[repr(C)]
struct EndlessSource {
    pub_mgr: JpegSourceMgr,
    header: Vec<u8>,
    filler: Vec<u8>,
    served: usize,
    /// How many times to suspend before becoming endless.
    ///
    /// One suspension leaves a parsed header with an incomplete body, which is
    /// the precondition for both body-drain cap sites. A second one makes
    /// `jpeg_start_decompress` give up at `DSTATE_PRELOAD` *without* tripping
    /// the cap, so the drain that finally trips it is the one inside
    /// `jpeg_consume_input` — the third site, which is otherwise unreachable
    /// because startup would have raised first.
    suspends_remaining: usize,
}

unsafe extern "C" fn init_source(_cinfo: *mut c_void) {}

/// Hand over the header first, then `filler` over and over. `filler` contains
/// no `FF` byte, so the boundary scanner never finds a marker and the drain
/// keeps asking for more — which is exactly the shape the cap exists to stop.
unsafe extern "C" fn fill_input_buffer(cinfo: *mut c_void) -> c_int {
    // SAFETY: the shim calls this only with the `cinfo` it was given, whose
    // `src` this test set to a live `EndlessSource` that outlives the decode.
    let src: *mut EndlessSource =
        unsafe { (*(cinfo as *mut JpegDecompressPublic)).src } as *mut EndlessSource;

    // SAFETY: both buffers are owned by that same allocation, so the pointers
    // handed out here stay valid, as a source manager's contract requires.
    unsafe {
        let served: usize = (*src).served;
        if served == 0 {
            (*src).pub_mgr.next_input_byte = (*src).header.as_ptr();
            (*src).pub_mgr.bytes_in_buffer = (*src).header.len();
            (*src).served = (*src).header.len();
            return 1;
        }
        if (*src).suspends_remaining > 0 {
            (*src).suspends_remaining -= 1;
            return 0;
        }
        if served >= TOTAL {
            return 0; // give up rather than run forever if the cap regresses
        }
        (*src).pub_mgr.next_input_byte = (*src).filler.as_ptr();
        (*src).pub_mgr.bytes_in_buffer = (*src).filler.len();
        (*src).served = served + (*src).filler.len();
    }
    1
}

unsafe extern "C" fn skip_input_data(_cinfo: *mut c_void, _num_bytes: std::ffi::c_long) {}
unsafe extern "C" fn term_source(_cinfo: *mut c_void) {}

/// Records the fatal error instead of letting the default handler exit.
/// Per-instance via `client_data` so nothing is shared between tests.
#[repr(C)]
struct Trap {
    fired: bool,
    msg_code: c_int,
    parm: c_int,
}

unsafe extern "C" fn trap_error_exit(cinfo: *mut c_void) {
    // SAFETY: `client_data` is the `Trap` this test installed, and `err` is
    // the live error manager. Read through raw pointers only — an `error_exit`
    // callback must not hold Rust references across the C boundary.
    unsafe {
        let common: *mut JpegDecompressPublic = cinfo as *mut JpegDecompressPublic;
        let trap: *mut Trap = (*common).client_data as *mut Trap;
        let err: *mut JpegErrorMgr = (*common).err;
        if !trap.is_null() && !err.is_null() {
            (*trap).fired = true;
            (*trap).msg_code = (*err).msg_code;
            // `msg_parm` mirrors upstream's `union { int i[8]; char s[80]; }`
            // as a byte array, so read the first `int` back out of it —
            // through `addr_of!`, since taking a reference to a field of a
            // struct behind a raw pointer is exactly what must not happen in
            // a callback the shim may still be mutating through.
            let parm: *const u8 = std::ptr::addr_of!((*err).msg_parm) as *const u8;
            let mut first: [u8; std::mem::size_of::<c_int>()] = Default::default();
            std::ptr::copy_nonoverlapping(parm, first.as_mut_ptr(), first.len());
            (*trap).parm = c_int::from_ne_bytes(first);
        }
    }
    // Returns rather than unwinding: panicking across `extern "C"` aborts.
}

/// A minimal header through the first SOS. The body that follows is the
/// endless filler, so the shim buffers until it hits its own bound.
fn header_through_sos() -> Vec<u8> {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 251) as u8).collect();
    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(80)
        .encode()
        .expect("encode fixture");
    let sos: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("fixture has an SOS");
    let len: usize = usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
    jpeg[..sos + 2 + len].to_vec()
}

#[test]
fn header_drain_cap_reports_a_resource_error_not_suspension() {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let mut trap: Box<Trap> = Box::new(Trap {
        fired: false,
        msg_code: 0,
        parm: 0,
    });
    let mut source: Box<EndlessSource> = Box::new(EndlessSource {
        // SAFETY: all-zero is a valid (null / `None`) value for every field of
        // this `#[repr(C)]` struct, and each is assigned before any call.
        pub_mgr: unsafe { std::mem::zeroed() },
        header: header_through_sos(),
        filler: vec![0x5Au8; CHUNK],
        served: 0,
        suspends_remaining: 0,
    });
    source.pub_mgr.init_source = Some(init_source);
    source.pub_mgr.fill_input_buffer = Some(fill_input_buffer);
    source.pub_mgr.skip_input_data = Some(skip_input_data);
    source.pub_mgr.term_source = Some(term_source);

    // SAFETY: every pointer refers to a live allocation owned here that
    // outlives the last call able to observe it.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        (*errp).error_exit = Some(trap_error_exit);
        cinfo.client_data = &mut *trap as *mut Trap as *mut c_void;
        cinfo.src = &mut source.pub_mgr as *mut JpegSourceMgr;

        // Poll until the shim stops asking for more. The bound is on buffered
        // bytes, so this terminates well before the source's own TOTAL.
        let mut polls: usize = 0;
        loop {
            let rc: c_int = jpeg_consume_input(p);
            polls += 1;
            eprintln!(
                "poll {polls}: rc={rc} state={} served={}",
                cinfo.global_state, source.served
            );
            if trap.fired || rc != 1 || polls > 4096 {
                break;
            }
        }

        assert!(
            trap.fired,
            "the 256 MiB cap must raise a classic error; after {polls} polls \
             nothing was reported, which is the silent-suspension behaviour \
             that makes a conforming refill-and-retry caller loop forever"
        );
        assert_eq!(
            trap.msg_code, JERR_OUT_OF_MEMORY,
            "a buffering limit is an out-of-memory condition"
        );
        assert_eq!(
            trap.parm, OOM_CASE_SUSPENDED_BODY_CAP,
            "the case number must be this shim's own, not one of upstream's \
             jmemmgr.c allocation sites"
        );

        jpeg_destroy_decompress(p);
    }
}

/// The other two cap sites, both of which live behind a completed header.
///
/// `finish_body_drain` (reached from `jpeg_start_decompress`) and the drain
/// loop inside `jpeg_consume_input` each had their own copy of the same
/// silent-suspension bug. Reaching them needs a source that delivers a
/// complete header, suspends once so `body_incomplete` is set, and only then
/// becomes endless.
///
/// The two phases use separate decompressors, run in sequence, so only one
/// 256 MiB accumulation is live at a time.
#[test]
fn body_drain_caps_report_a_resource_error_not_suspension() {
    // Phase 1: jpeg_start_decompress -> finish_body_drain.
    let (fired, code, parm, note) = drive(Phase::Startup);
    assert!(fired, "start_decompress body drain: {note}");
    assert_eq!(code, JERR_OUT_OF_MEMORY, "start_decompress body drain");
    assert_eq!(
        parm, OOM_CASE_SUSPENDED_BODY_CAP,
        "start_decompress body drain"
    );

    // Phase 2: polling jpeg_consume_input from DSTATE_PRELOAD.
    let (fired, code, parm, note) = drive(Phase::PreloadPoll);
    assert!(fired, "consume_input body drain: {note}");
    assert_eq!(code, JERR_OUT_OF_MEMORY, "consume_input body drain");
    assert_eq!(
        parm, OOM_CASE_SUSPENDED_BODY_CAP,
        "consume_input body drain"
    );
}

enum Phase {
    /// Let `jpeg_start_decompress` do the draining.
    Startup,
    /// Suspend startup first, then drain by polling from `DSTATE_PRELOAD`.
    PreloadPoll,
}

/// Run one phase and report what the error handler saw.
fn drive(phase: Phase) -> (bool, c_int, c_int, String) {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let mut trap: Box<Trap> = Box::new(Trap {
        fired: false,
        msg_code: 0,
        parm: 0,
    });
    let mut source: Box<EndlessSource> = Box::new(EndlessSource {
        // SAFETY: all-zero is valid for every field; each is set below.
        pub_mgr: unsafe { std::mem::zeroed() },
        header: header_through_sos(),
        filler: vec![0x5Au8; CHUNK],
        served: 0,
        suspends_remaining: match phase {
            // One suspension: header parsed, body incomplete, then endless —
            // so `finish_body_drain` is what meets the cap.
            Phase::Startup => 1,
            // Two: startup gives up at PRELOAD without reaching the cap, and
            // the polling drain meets it instead.
            Phase::PreloadPoll => 2,
        },
    });
    source.pub_mgr.init_source = Some(init_source);
    source.pub_mgr.fill_input_buffer = Some(fill_input_buffer);
    source.pub_mgr.skip_input_data = Some(skip_input_data);
    source.pub_mgr.term_source = Some(term_source);

    let mut note: String = String::new();
    // SAFETY: every pointer refers to a live allocation owned here that
    // outlives the last call able to observe it.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        (*errp).error_exit = Some(trap_error_exit);
        cinfo.client_data = &mut *trap as *mut Trap as *mut c_void;
        cinfo.src = &mut source.pub_mgr as *mut JpegSourceMgr;

        // Reach the first SOS.
        let mut polls: usize = 0;
        while jpeg_consume_input(p) != JPEG_REACHED_SOS && polls < 64 {
            polls += 1;
        }
        note.push_str(&format!(
            "sos after {polls} polls, state {}; ",
            cinfo.global_state
        ));

        match phase {
            Phase::Startup => {
                let ok: c_int = jpeg_start_decompress(p);
                note.push_str(&format!(
                    "startup returned {ok}, state {}",
                    cinfo.global_state
                ));
            }
            Phase::PreloadPoll => {
                // The one suspension the source is allowed lands here, leaving
                // DSTATE_PRELOAD with the body still incomplete.
                let ok: c_int = jpeg_start_decompress(p);
                note.push_str(&format!(
                    "startup returned {ok}, state {}; ",
                    cinfo.global_state
                ));
                let mut drains: usize = 0;
                while !trap.fired && drains < 4096 {
                    let rc: c_int = jpeg_consume_input(p);
                    drains += 1;
                    if rc == JPEG_REACHED_EOI {
                        break;
                    }
                }
                note.push_str(&format!(
                    "{drains} drain polls, state {}",
                    cinfo.global_state
                ));
            }
        }

        jpeg_destroy_decompress(p);
    }
    (trap.fired, trap.msg_code, trap.parm, note)
}
