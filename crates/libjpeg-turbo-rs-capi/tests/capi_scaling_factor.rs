//! Issue #478 (P4-139 criterion 4): `tj3SetScalingFactor` accepts exactly the
//! factors upstream accepts and refuses the rest with upstream's message.
//!
//! The root crate's `ScalingFactor` now has private fields behind a validating
//! `try_new`, and this entry point is a thin layer over it. Before the change
//! the C-visible text was `tj3SetScalingFactor: corrupt data: unsupported
//! scaling factor N/D`, or a `non-positive ratio` message upstream has no
//! equivalent of; upstream has one refusal, "Unsupported scaling factor", for
//! every pair missing from its table (`turbojpeg.c:2053-2058`).

use std::ffi::{c_int, c_void, CStr};

mod helpers;

use libjpeg_turbo_rs_capi::{
    tj3Destroy, tj3GetErrorStr, tj3GetScalingFactors, tj3Init, tj3SetScalingFactor, TjScalingFactor,
};

/// `turbojpeg.h:99`: `TJINIT_DECOMPRESS`.
const TJINIT_DECOMPRESS: c_int = 1;

/// The trace `examples/scaling_factor_oracle.c` prints, produced through this
/// crate's exports.
fn our_trace() -> String {
    let mut trace: String = String::new();
    let mut count: c_int = 0;
    // SAFETY: `count` is a valid out-pointer; the returned table is static
    // and holds `count` entries.
    let table: &[TjScalingFactor] = unsafe {
        let ptr: *mut TjScalingFactor = tj3GetScalingFactors(&mut count);
        assert!(!ptr.is_null(), "tj3GetScalingFactors");
        std::slice::from_raw_parts(ptr, count as usize)
    };
    for (index, factor) in table.iter().enumerate() {
        trace.push_str(&format!("sf {index} {}/{}\n", factor.num, factor.denom));
    }

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_DECOMPRESS)");
    for num in -1..=20 {
        for denom in -1..=20 {
            // SAFETY: `handle` is a live instance from `tj3Init`, used
            // exclusively here; the error string is read before the next call.
            let (rc, message): (c_int, String) = unsafe {
                let rc: c_int = tj3SetScalingFactor(handle, TjScalingFactor { num, denom });
                let message: String = CStr::from_ptr(tj3GetErrorStr(handle))
                    .to_string_lossy()
                    .into_owned();
                (rc, message)
            };
            let kind: &str = if rc == 0 {
                "none"
            } else if message.contains("Unsupported scaling factor") {
                "unsupported"
            } else {
                "other"
            };
            trace.push_str(&format!("set {num}/{denom} {rc} kind={kind}\n"));
        }
    }
    // SAFETY: `handle` came from `tj3Init` and is not used afterwards.
    unsafe { tj3Destroy(handle) };
    trace
}

/// Issue #478: the exact refusal text, for the classes of input the old code
/// split between two different messages.
#[test]
fn refusal_is_upstreams_message_for_every_kind_of_miss() {
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_DECOMPRESS)");
    for (num, denom) in [(0, 1), (1, 0), (-1, 8), (1, 3), (4, 8), (16, 8)] {
        // SAFETY: as in `our_trace`.
        let (rc, message): (c_int, String) = unsafe {
            let rc: c_int = tj3SetScalingFactor(handle, TjScalingFactor { num, denom });
            let message: String = CStr::from_ptr(tj3GetErrorStr(handle))
                .to_string_lossy()
                .into_owned();
            (rc, message)
        };
        assert_eq!(rc, -1, "{num}/{denom} must be refused");
        assert_eq!(
            message, "tj3SetScalingFactor: Unsupported scaling factor",
            "{num}/{denom}"
        );
    }
    // SAFETY: `handle` came from `tj3Init` and is not used afterwards.
    unsafe { tj3Destroy(handle) };
}

/// Issue #478: the accepted set and the reported table, against real
/// TurboJPEG.
#[test]
fn scaling_factor_acceptance_matches_upstream_turbojpeg() {
    let Some(oracle) = helpers::build_oracle("scaling_factor_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             tj3SetScalingFactor cannot be built. Set LIBJPEG_TURBO_PREFIX to \
             make this a hard failure."
        );
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);
    assert_eq!(
        our_trace(),
        c_trace,
        "tj3SetScalingFactor acceptance diverges from upstream TurboJPEG"
    );
}
