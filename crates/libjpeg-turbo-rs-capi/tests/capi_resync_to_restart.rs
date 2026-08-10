//! P4-97 (#469): `jpeg_resync_to_restart` must implement the C default
//! recovery algorithm from `jdmarker.c`, not return a constant `TRUE`.
//!
//! The old implementation ignored `unread_marker`, emitted no warning, never
//! pulled the source manager and never mutated state — it just returned `TRUE`.
//!
//! # Which of these actually falsify the old version
//!
//! Measured by patching `resync_to_restart_impl` back to `return 1` and
//! re-running: **6 of the 10 fail**, namely the action-1 cases (which require
//! `unread_marker` to be *cleared*), both suspension cases, the warning count,
//! and the NULL case.
//!
//! The other four are the action-3 cases. A constant `TRUE` that never touches
//! `unread_marker` happens to produce the same observable as "leave the marker
//! unread", so they pass either way. They are kept deliberately — they pin the
//! *other* half of the decision table, and without them a future change that
//! collapsed everything to action 1 would go unnoticed — but they are not
//! evidence against the old bug, and this note exists so nobody later mistakes
//! them for it.
//!
//! # The algorithm being pinned
//!
//! `read_restart_marker` calls this when it finds a marker other than the
//! restart marker it expected. `unread_marker` is what was found; `desired` is
//! the restart number (0..7) that was wanted. Upstream picks one of three
//! actions:
//!
//! | Found marker | Action | Observable effect |
//! | --- | --- | --- |
//! | `< M_SOF0` (0xC0) — bogus | 2 | scan forward to the next marker |
//! | valid non-restart | 3 | leave `unread_marker` set |
//! | `RST(desired+1)` / `RST(desired+2)` | 3 | leave `unread_marker` set |
//! | `RST(desired-1)` / `RST(desired-2)` | 2 | scan forward |
//! | the desired restart, or > 2 away | 1 | clear `unread_marker`, resume |
//!
//! Actions 1 and 3 are distinguished *only* by whether `unread_marker` is
//! cleared, so that field is the observable this file asserts on.

// `num_warnings` is `c_long`: i64 here, i32 on Windows. Using the alias rather
// than a fixed width keeps this portable and avoids a cast clippy would reject
// as redundant on 64-bit hosts.
use std::ffi::{c_int, c_long, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_destroy_decompress, jpeg_mem_src, jpeg_resync_to_restart,
    jpeg_std_error, JpegDecompressPublic, JpegErrorMgr,
};

const M_RST0: c_int = 0xD0;
/// `JPEG_LIB_VERSION` and struct size the create guard expects.
const JPEG_LIB_VERSION: c_int = 80;

/// Build a decompress object with a std error manager attached, mirroring what
/// a C consumer does before any of this is reachable.
fn make_cinfo() -> (Box<JpegDecompressPublic>, Box<JpegErrorMgr>) {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    // The tripwire this comment used to predict has fired: P4-137 criterion 1
    // landed, so `jpeg_std_error`, `jpeg_CreateDecompress` and
    // `jpeg_destroy_decompress` are `unsafe extern "C" fn` and these calls no
    // longer compile without a block. That is the change working as intended.
    //
    // SAFETY: `err` and `cinfo` are live `Box` allocations owned by this
    // function for the whole call, correctly aligned, and not aliased — the
    // raw pointers are taken from `&mut` borrows that end at each call. The
    // declared struct size matches the type actually passed.
    let err_ptr: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut *err as *mut JpegErrorMgr) };
    cinfo.err = err_ptr;
    unsafe {
        jpeg_CreateDecompress(
            &mut *cinfo as *mut JpegDecompressPublic as *mut c_void,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
    }
    // `jpeg_CreateDecompress` installs its own error manager slot; restore ours
    // so warnings land where this test can count them.
    cinfo.err = err_ptr;
    (cinfo, err)
}

fn destroy(cinfo: &mut JpegDecompressPublic) {
    // SAFETY: `cinfo` is a live, correctly-aligned decompress struct created
    // by `make_cinfo`, and this is its single destroy — the borrow proves no
    // other reference is live for the call.
    unsafe {
        jpeg_destroy_decompress(cinfo as *mut JpegDecompressPublic as *mut c_void);
    }
}

/// Run the resync entry point with a given found-marker / desired pair and
/// report `(returned, unread_marker_after, warnings_emitted)`.
fn resync(found: c_int, desired: c_int) -> (c_int, c_int, c_long) {
    let (mut cinfo, err) = make_cinfo();
    cinfo.unread_marker = found;

    // SAFETY: `cinfo` is live; no source manager is attached, which is fine for
    // the cases below because none of them takes action 2 (scan forward).
    let rc: c_int = unsafe {
        jpeg_resync_to_restart(
            &mut *cinfo as *mut JpegDecompressPublic as *mut c_void,
            desired,
        )
    };
    let after: c_int = cinfo.unread_marker;
    let warnings: c_long = err.num_warnings;
    destroy(&mut cinfo);
    (rc, after, warnings)
}

// ---------------------------------------------------------------------------
// Action 1 — discard the marker and resume
// ---------------------------------------------------------------------------

/// The desired restart itself: discard it, entropy decoding resumes.
#[test]
fn desired_restart_clears_unread_marker() {
    let (rc, after, _) = resync(M_RST0 + 3, 3);
    assert_eq!(rc, 1, "the desired restart must not report suspension");
    assert_eq!(
        after, 0,
        "action 1 must clear `unread_marker`; leaving it set would make the \
         entropy decoder process an empty segment instead of resuming"
    );
}

/// More than two restarts away is untrusted: also action 1.
#[test]
fn restart_more_than_two_away_clears_unread_marker() {
    // desired 0, found RST4 — four ahead, i.e. > 2 in both directions.
    let (rc, after, _) = resync(M_RST0 + 4, 0);
    assert_eq!(rc, 1);
    assert_eq!(
        after, 0,
        "a restart more than two counts away is treated as erroneous (action 1)"
    );
}

// ---------------------------------------------------------------------------
// Action 3 — leave the marker unread
// ---------------------------------------------------------------------------

/// The next expected restart: we missed `desired`, so leave this one to be
/// re-read after an empty segment.
#[test]
fn next_expected_restart_leaves_marker_unread() {
    let found: c_int = M_RST0 + 1;
    let (rc, after, _) = resync(found, 0);
    assert_eq!(rc, 1);
    assert_eq!(
        after, found,
        "RST(desired+1) is action 3 — `unread_marker` must survive so the \
         marker is reprocessed"
    );
}

#[test]
fn second_next_expected_restart_leaves_marker_unread() {
    let found: c_int = M_RST0 + 2;
    let (rc, after, _) = resync(found, 0);
    assert_eq!(rc, 1);
    assert_eq!(after, found, "RST(desired+2) is also action 3");
}

/// `desired` wraps modulo 8, so RST0 is "next" after RST7.
#[test]
fn expected_restart_arithmetic_wraps_modulo_eight() {
    let found: c_int = M_RST0; // RST0
    let (rc, after, _) = resync(found, 7); // desired 7 → next is (7+1)&7 = 0
    assert_eq!(rc, 1);
    assert_eq!(
        after, found,
        "(desired + 1) & 7 must wrap: RST0 is the next restart after RST7"
    );
}

/// A valid non-restart marker (EOI here) keeps us from overrunning the scan.
#[test]
fn valid_non_restart_marker_leaves_marker_unread() {
    let eoi: c_int = 0xD9;
    let (rc, after, _) = resync(eoi, 0);
    assert_eq!(rc, 1);
    assert_eq!(
        after, eoi,
        "a valid non-restart marker is action 3, so EOI is not swallowed"
    );
}

// ---------------------------------------------------------------------------
// The warning is not optional
// ---------------------------------------------------------------------------

/// Upstream emits `JWRN_MUST_RESYNC` unconditionally, before deciding.
#[test]
fn every_resync_emits_a_warning() {
    let (_, _, warnings) = resync(M_RST0 + 3, 3);
    assert_eq!(
        warnings, 1,
        "upstream puts up a warning on every resync attempt; the old constant-\
         TRUE implementation emitted none, so a caller counting `num_warnings` \
         saw a clean stream"
    );
}

// ---------------------------------------------------------------------------
// Suspension
// ---------------------------------------------------------------------------

/// Action 2 with no source manager cannot scan forward, so it must report
/// suspension (`FALSE`) rather than claim success.
///
/// A bogus marker (`< M_SOF0`) selects action 2.
#[test]
fn scan_forward_without_a_source_reports_suspension() {
    let bogus: c_int = 0x02; // below M_SOF0 → invalid marker → action 2
    let (rc, _, _) = resync(bogus, 0);
    assert_eq!(
        rc, 0,
        "action 2 with nothing to read from must return FALSE (suspension). \
         Returning TRUE here is exactly the silent lie P4-97 describes."
    );
}

/// A prior restart also selects action 2.
#[test]
fn prior_restart_scans_forward_and_suspends_without_a_source() {
    // desired 3, found RST2 → (desired - 1) & 7 → action 2.
    let (rc, _, _) = resync(M_RST0 + 2, 3);
    assert_eq!(rc, 0, "RST(desired-1) is action 2, which needs the source");
}

// ---------------------------------------------------------------------------
// NULL handling
// ---------------------------------------------------------------------------

/// A NULL `cinfo` has no state to inspect. It must not claim success.
#[test]
fn null_cinfo_does_not_report_success() {
    // SAFETY: NULL is explicitly handled by the implementation.
    let rc: c_int = unsafe { jpeg_resync_to_restart(std::ptr::null_mut(), 0) };
    assert_eq!(rc, 0, "NULL cinfo must not return TRUE");
}

// ---------------------------------------------------------------------------
// Scan-forward against the built-in memory source
// ---------------------------------------------------------------------------

/// Action 2 over a real (if tiny) `jpeg_mem_src` buffer.
///
/// Stock's memory source answers a request past the end by warning
/// `JWRN_JPEG_EOF` and inserting a fake `FF D9`, so the scan finds EOI and
/// resync returns `TRUE` with `unread_marker == EOI`. Our default manager used
/// to return TRUE while supplying no bytes, which this code read as suspension
/// — `FALSE` on a source that can never resume. Codex review of the first cut
/// caught that divergence; this pins the corrected behaviour.
#[test]
fn scan_forward_past_end_of_memory_source_yields_fake_eoi() {
    let (mut cinfo, _err) = make_cinfo();
    // One non-FF byte: the scan discards it, then needs more input.
    let buf: [u8; 1] = [0x01];
    // SAFETY: `cinfo` is live for the whole call and `buf` outlives the
    // source manager's use of it within this test; the declared length is
    // exactly `buf`'s.
    unsafe {
        jpeg_mem_src(
            &mut *cinfo as *mut JpegDecompressPublic as *mut c_void,
            buf.as_ptr(),
            buf.len() as std::os::raw::c_ulong,
        );
    }

    // RST2 with desired 3 is `(desired - 1) & 7` → action 2, scan forward.
    cinfo.unread_marker = M_RST0 + 2;
    // SAFETY: `cinfo` is live with a memory source attached.
    let rc: c_int = unsafe {
        jpeg_resync_to_restart(&mut *cinfo as *mut JpegDecompressPublic as *mut c_void, 3)
    };
    let after: c_int = cinfo.unread_marker;
    destroy(&mut cinfo);

    assert_eq!(
        rc, 1,
        "a memory source at end-of-buffer synthesises EOI, so resync succeeds; \
         returning FALSE here would report suspension on a source that cannot \
         resume"
    );
    assert_eq!(
        after, 0xD9,
        "the scan must land on the synthetic EOI marker"
    );
}

// ---------------------------------------------------------------------------
// The recovery-action trace is observable state
// ---------------------------------------------------------------------------

/// Upstream runs `TRACEMS2(..., JTRC_RECOVERY_ACTION, marker, action)` before
/// applying each action. `TRACEMS2` publishes `msg_code` and both parameters
/// regardless of `trace_level`, so a C consumer that inspects `err->msg_code`
/// after the call sees `JTRC_RECOVERY_ACTION` (99), not `JWRN_MUST_RESYNC`.
#[test]
fn recovery_action_trace_is_published() {
    let (mut cinfo, err) = make_cinfo();
    cinfo.unread_marker = M_RST0 + 3;
    // SAFETY: `cinfo` is live.
    let rc: c_int = unsafe {
        jpeg_resync_to_restart(&mut *cinfo as *mut JpegDecompressPublic as *mut c_void, 3)
    };
    let code: c_int = unsafe { (*cinfo.err).msg_code };
    destroy(&mut cinfo);
    drop(err);

    assert_eq!(rc, 1);
    assert_eq!(
        code, 99,
        "msg_code must end at JTRC_RECOVERY_ACTION (99); leaving it at \
         JWRN_MUST_RESYNC (124) means the trace was skipped"
    );
}
