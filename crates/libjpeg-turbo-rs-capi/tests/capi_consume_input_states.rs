//! P4-104 (#468): `jpeg_consume_input`'s state/return sequence must match
//! upstream's.
//!
//! Upstream (`jdapimin.c`):
//!
//! - `DSTATE_START` — reset the input controller, init the source, move to
//!   `INHEADER`, fall through;
//! - `DSTATE_INHEADER` — consume; on `JPEG_REACHED_SOS` set the default
//!   decompress params and move to **`DSTATE_READY`**;
//! - `DSTATE_READY` — return `JPEG_REACHED_SOS` and **consume nothing**:
//!   *"Can't advance past first SOS until start_decompress is called."*
//!
//! This shim used to jump to `DSTATE_SCANNING` after the header and report
//! `JPEG_REACHED_EOI` from the second poll onward, because the whole
//! datastream is buffered in memory. That is an implementation fact, not a
//! licence to tell a caller the stream is consumed while it is still sitting
//! at the header.
//!
//! These assertions were first made with a throwaway probe binary, which is
//! why the behaviour was verified but not *pinned*; the first version of this
//! fix shipped with no test covering the path it changed.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_consume_input, jpeg_destroy_decompress, jpeg_mem_src,
    jpeg_std_error, JpegDecompressPublic, JpegErrorMgr,
};

const JPEG_LIB_VERSION: c_int = 80;
/// `jpegint.h`, cross-checked by `every_state_constant_matches_upstream_jpegint_h`.
const DSTATE_START: c_int = 200;
const DSTATE_READY: c_int = 202;
/// `jpeglib.h` return codes.
const JPEG_REACHED_SOS: c_int = 1;

fn encode(progressive: bool) -> Vec<u8> {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 251) as u8).collect();
    Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(80)
        .progressive(progressive)
        .encode()
        .expect("encode fixture")
}

/// Drive `jpeg_consume_input` `polls` times, returning `(returns, states)`.
fn poll_sequence(jpeg: &[u8], polls: usize) -> (Vec<c_int>, Vec<c_int>) {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    // SAFETY: both are live, correctly-aligned allocations owned here; `jpeg`
    // outlives every call, as `jpeg_mem_src`'s retained-pointer contract needs.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        cinfo.err = errp;
        // `error_exit` must return, not unwind: panicking would cross an
        // `extern "C"` frame and abort.
        unsafe extern "C" fn ignore_error_exit(_cinfo: *mut c_void) {}
        (*errp).error_exit = Some(ignore_error_exit);

        assert_eq!(
            cinfo.global_state, DSTATE_START,
            "a fresh decompressor starts at DSTATE_START"
        );
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);

        let mut returns: Vec<c_int> = Vec::new();
        let mut states: Vec<c_int> = Vec::new();
        for _ in 0..polls {
            returns.push(jpeg_consume_input(p));
            states.push(cinfo.global_state);
        }
        jpeg_destroy_decompress(p);
        (returns, states)
    }
}

/// Baseline: every poll after the first reports SOS from READY, forever.
#[test]
fn baseline_polls_stay_at_ready_reporting_sos() {
    let (returns, states) = poll_sequence(&encode(false), 4);
    assert_eq!(
        returns,
        vec![JPEG_REACHED_SOS; 4],
        "upstream returns REACHED_SOS from READY indefinitely; \
         reporting EOI would tell a caller the stream was consumed while it \
         is still at the header"
    );
    assert_eq!(
        states,
        vec![DSTATE_READY; 4],
        "the header parse lands on READY and later polls must not advance it"
    );
}

/// Progressive takes the identical path — upstream does not distinguish them
/// here, and neither may we. Before this fix both reported EOI on poll 2, for
/// the same wrong reason: the datastream was already buffered.
#[test]
fn progressive_polls_match_baseline() {
    let (returns, states) = poll_sequence(&encode(true), 4);
    assert_eq!(returns, vec![JPEG_REACHED_SOS; 4]);
    assert_eq!(states, vec![DSTATE_READY; 4]);
}

/// The first poll is the one that transitions; it must land on READY rather
/// than skipping ahead to SCANNING.
#[test]
fn first_poll_transitions_start_to_ready() {
    let (returns, states) = poll_sequence(&encode(false), 1);
    assert_eq!(returns, vec![JPEG_REACHED_SOS]);
    assert_eq!(
        states,
        vec![DSTATE_READY],
        "DSTATE_START -> DSTATE_READY, per jdapimin.c's \
         'Set global state: ready for start_decompress'"
    );
}

// ---------------------------------------------------------------------------
// C oracle: compare the trace against stock libjpeg rather than against a
// transcription of it.
// ---------------------------------------------------------------------------

mod helpers;

/// Write `bytes` somewhere the oracle can read, uniquely per process and
/// label so parallel tests cannot collide.
fn temp_fixture(label: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path: std::path::PathBuf = std::env::temp_dir().join(format!(
        "libjpeg_turbo_rs_p4104_{}_{label}.jpg",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write oracle fixture");
    path
}

/// The traces this shim produces must equal stock libjpeg's, not a comment's
/// idea of them.
///
/// The sibling tests above assert sequences transcribed from `jdapimin.c`.
/// That is exactly the shape of check that stays green when the transcription
/// is wrong, because the same misreading lands in the implementation and the
/// expectation together — the failure mode CLAUDE.md's C cross-validation rule
/// exists to prevent. Here upstream answers for itself.
#[test]
fn poll_trace_matches_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("consume_input_states_oracle") else {
        eprintln!(
            "SKIP: no stock libjpeg development install found; the C oracle for \
             P4-104's state trace cannot be built. Set LIBJPEG_TURBO_PREFIX to \
             make this a hard failure."
        );
        return;
    };

    for (label, progressive) in [("baseline", false), ("progressive", true)] {
        let jpeg: Vec<u8> = encode(progressive);
        let path: std::path::PathBuf = temp_fixture(label, &jpeg);
        let c_trace: String =
            helpers::run_oracle(&oracle, &["poll", path.to_str().expect("utf-8 temp path")]);
        let _ = std::fs::remove_file(&path);

        let (returns, states) = poll_sequence(&jpeg, 3);
        let mut ours: String = format!("create -1 {DSTATE_START}\n");
        for (rc, state) in returns.iter().zip(states.iter()) {
            ours.push_str(&format!("poll {rc} {state}\n"));
        }

        assert_eq!(
            ours, c_trace,
            "{label}: (return, global_state) trace diverges from stock libjpeg.\n\
             ours:\n{ours}\nC:\n{c_trace}"
        );
    }
}
