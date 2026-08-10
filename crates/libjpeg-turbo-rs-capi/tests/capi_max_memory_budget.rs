//! P4-14 (#467): `cinfo->mem->max_memory_to_use` must actually bound the
//! virtual-array allocation, the way upstream does.
//!
//! The field sat at the correct ABI offset with **zero comparisons against it
//! anywhere** — a C consumer setting the upstream-documented budget got it
//! silently ignored.
//!
//! The error contract is `JERR_NO_BACKING_STORE` ("Memory limit exceeded",
//! code 51), **not** `JERR_OUT_OF_MEMORY`. Upstream consults the budget in
//! `jpeg_mem_available` (`jmemnobs.c:66-78`) and the spill that follows a
//! shortfall lands on `jmemnobs.c:87-92`. `capi_classic_error_codes.rs`
//! cross-validates both the number and the message against the pinned v8
//! headers — which is how the first draft of this test, written against
//! "Backing store not supported", was caught guessing from the macro name.
//!
//! Upstream ships this same no-backing-store design: `CMakeLists.txt:678`
//! compiles `src/jmemnobs.c` unconditionally. Our "never spills to disk" is
//! not a divergence.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_destroy_decompress, jpeg_std_error, JpegDecompressPublic,
    JpegErrorMgr,
};
use libjpeg_turbo_rs_capi::memmgr::JpegMemoryMgr;

const JPOOL_IMAGE: c_int = 1;
/// `jerror.h:114` — "Memory limit exceeded".
const JERR_NO_BACKING_STORE: c_int = 51;
/// Written into `msg_code` before each run so "nothing raised" is
/// distinguishable from "raised code 0".
const NO_ERROR: c_int = -1;

/// Returns the `msg_code` raised while realizing one virtual sarray of the
/// given geometry under `budget` bytes, or `NO_ERROR` if it succeeded.
///
/// Per-instance state throughout: no `static mut`, because `cargo test` runs
/// these in parallel and a shared capture would let one case read another's
/// result.
fn realize_under_budget(budget: i64, samplesperrow: u32, rows: u32) -> c_int {
    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: JpegDecompressPublic = unsafe { std::mem::zeroed() };
    cinfo.err = unsafe { jpeg_std_error(&mut err) };
    let cinfo_ptr: *mut c_void = &mut cinfo as *mut JpegDecompressPublic as *mut c_void;
    unsafe { jpeg_CreateDecompress(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>()) };

    // A no-op `error_exit`. It returns where a real consumer would `longjmp`;
    // panicking here would cross an `extern "C"` frame, which aborts. Safe for
    // what is observed: the guard raises and then returns immediately.
    unsafe extern "C" fn ignore_error_exit(_cinfo: *mut c_void) {}
    unsafe {
        (*cinfo.err).error_exit = Some(ignore_error_exit);
        (*cinfo.err).msg_code = NO_ERROR;
    }

    // SAFETY: `cinfo.mem` was populated by `jpeg_CreateDecompress` above.
    let mgr: &mut JpegMemoryMgr = unsafe { &mut *(cinfo.mem as *mut JpegMemoryMgr) };
    mgr.max_memory_to_use = budget as std::os::raw::c_long;

    let request = mgr
        .request_virt_sarray
        .expect("request_virt_sarray populated");
    let realize = mgr
        .realize_virt_arrays
        .expect("realize_virt_arrays populated");

    // SAFETY: `cinfo_ptr` is a live decompress struct; the geometry is the
    // subject of the test and is never dereferenced before the budget check.
    unsafe {
        let ctrl = request(cinfo_ptr, JPOOL_IMAGE, 0, samplesperrow, rows, rows);
        assert!(!ctrl.is_null(), "request_virt_sarray returned NULL");
        realize(cinfo_ptr);
    }

    let raised: c_int = unsafe { (*cinfo.err).msg_code };
    unsafe { jpeg_destroy_decompress(cinfo_ptr) };
    raised
}

/// A budget below the array's footprint is refused, with upstream's code.
///
/// 4096 x 4096 samples is 16 MiB; the budget is 1 MiB.
///
/// **This pins our behaviour, which is stricter than upstream's, and the
/// difference is deliberate.** Upstream parcels a shortfall into strips of
/// `maxaccess` rows and only fails when even that minimum will not fit; with
/// `maxaccess == rows_in_array` (as here) its minimum *is* the whole array, so
/// stock libjpeg-turbo would force one full-height buffer and succeed. We
/// allocate full height with no strip machinery, so any shortfall is fatal.
///
/// Honouring a budget that lands *between* the minimum and the full footprint
/// requires strip-wise realization, which P4-14 records as outstanding. Until
/// then the choice is between rejecting what upstream accepts (this) and
/// silently allocating past a budget the caller set (worse). Do not "fix" this
/// test by relaxing the check without implementing strips.
#[test]
fn budget_below_the_working_set_raises_no_backing_store() {
    let raised: c_int = realize_under_budget(1 << 20, 4096, 4096);
    assert_eq!(
        raised, JERR_NO_BACKING_STORE,
        "a 1 MiB budget against a 16 MiB virtual array must raise \
         JERR_NO_BACKING_STORE (51), not succeed and not JERR_OUT_OF_MEMORY (56)"
    );
}

/// The companion, without which the guard could be "satisfied" by refusing
/// everything: an ample budget must still realize.
#[test]
fn ample_budget_still_realizes() {
    let raised: c_int = realize_under_budget(1 << 30, 4096, 4096);
    assert_eq!(
        raised, NO_ERROR,
        "a 1 GiB budget against a 16 MiB virtual array must succeed (raised {raised})"
    );
}

/// `max_memory_to_use <= 0` means unlimited, matching upstream's
/// `if (cinfo->mem->max_memory_to_use)` test. The field is a signed `long`, so
/// a negative value must not be read as an enormous budget — nor as a zero one.
#[test]
fn non_positive_budget_means_unlimited() {
    assert_eq!(
        realize_under_budget(0, 1024, 1024),
        NO_ERROR,
        "budget 0 is upstream's 'no limit', not 'no memory'"
    );
    assert_eq!(
        realize_under_budget(-1, 1024, 1024),
        NO_ERROR,
        "a negative budget is not a limit either"
    );
}
