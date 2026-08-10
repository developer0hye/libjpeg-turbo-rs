//! P4-137 criterion 4 (#476): the handle accessor must not let a caller name
//! the lifetime of `&mut TjInstance`.
//!
//! # What was wrong
//!
//! ```ignore
//! pub(crate) unsafe fn handle_as_mut<'a>(handle: *mut c_void) -> Option<&'a mut TjInstance>
//! ```
//!
//! `'a` was chosen by the *caller*, unconstrained by any input reference. The
//! borrow checker therefore had nothing to tie the reference to, so two calls
//! on one handle produced two simultaneously-live `&mut TjInstance` — instant
//! UB, with no diagnostic. `tj3::with_handle` picks the lifetime itself and
//! confines the borrow to a closure, so that construction no longer typechecks.
//!
//! # What this file can and cannot prove
//!
//! `with_handle` is `pub(crate)`, so an integration test cannot call it and
//! cannot host the compile-fail case. What it *can* do is pin the two things
//! that would silently undo the fix — the accessor's shape, and the absence of
//! the old one — and check that the entry points built on it still behave.
//!
//! Asserting the source rather than using `trybuild` follows the precedent in
//! the root crate's `simd_module_privacy.rs`: an stderr snapshot pins exact
//! rustc diagnostics, which drift between the MSRV and stable toolchains this
//! repo builds on, and the usual fix is to loosen the snapshot until it stops
//! proving anything.

use std::ffi::c_void;
use std::path::PathBuf;

fn capi_source(file: &str) -> String {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The caller-chosen-lifetime accessor must not come back.
#[test]
fn handle_as_mut_is_gone_from_every_module() {
    for file in [
        "tj3.rs",
        "header.rs",
        "compress.rs",
        "decompress.rs",
        "transform.rs",
        "precision.rs",
        "imageio.rs",
        "yuv.rs",
        "legacy.rs",
    ] {
        let source: String = capi_source(file);
        // The doc comment on `with_handle` names the old function to explain
        // what it replaced; only a real call or definition is a regression.
        assert!(
            !source.contains("handle_as_mut(handle)"),
            "{file} still calls `handle_as_mut`, which lets the caller choose \
             the lifetime of `&mut TjInstance` — two aliasing `&mut` to one \
             instance become constructible again (P4-137). Use \
             `tj3::with_handle`."
        );
        assert!(
            !source.contains("fn handle_as_mut"),
            "{file} redefines `handle_as_mut`"
        );
    }
}

/// The replacement must keep the shape that makes the fix work: a closure
/// parameter, so the borrow cannot escape, and no caller-visible lifetime.
#[test]
fn with_handle_confines_the_borrow_to_a_closure() {
    let source: String = capi_source("tj3.rs");

    assert!(
        source.contains("pub(crate) unsafe fn with_handle<R>("),
        "`with_handle` is missing or no longer generic over its return type"
    );
    assert!(
        source.contains("f: impl FnOnce(&mut TjInstance) -> R,"),
        "`with_handle` no longer takes the instance by closure. Returning the \
         reference instead would restore the unbounded-lifetime hazard \
         regardless of what the signature's lifetimes look like (P4-137)."
    );
    // A caller-nameable lifetime on this function is exactly the defect.
    assert!(
        !source.contains("unsafe fn with_handle<'a"),
        "`with_handle` grew a caller-chosen lifetime parameter"
    );
}

// ---------------------------------------------------------------------------
// Behaviour is unchanged through the new accessor
// ---------------------------------------------------------------------------

// Called through the Rust path rather than by dlopening the cdylib, because
// the rlib surface is exactly what P4-137 is about: these are `pub extern "C"`
// items in a crate downstream Rust code can depend on directly.
use libjpeg_turbo_rs_capi::tj3::{tj3Destroy, tj3Get, tj3Init, tj3Set};

/// `TJPARAM_QUALITY` — a plain round-trippable parameter. The numeric layout
/// matches `turbojpeg.h`'s `TJPARAM_*` enumeration, mirrored in
/// `tj3::param_from_c`.
const TJPARAM_QUALITY: std::ffi::c_int = 3;
const TJINIT_COMPRESS: std::ffi::c_int = 0;

#[test]
fn set_and_get_still_round_trip_through_with_handle() {
    // SAFETY: a handle from `tj3Init`, used on one thread, destroyed once.
    unsafe {
        let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
        assert!(!handle.is_null(), "tj3Init returned NULL");

        assert_eq!(tj3Set(handle, TJPARAM_QUALITY, 77), 0, "tj3Set failed");
        assert_eq!(
            tj3Get(handle, TJPARAM_QUALITY),
            77,
            "value did not survive the with_handle round trip"
        );

        tj3Destroy(handle);
    }
}

/// A NULL handle must still be the documented error, not a dereference —
/// `with_handle` returns `None` and each entry point maps it to its sentinel.
///
/// Note what is *missing* here: no `unsafe` block. `tj3Set` and `tj3Get` take a
/// raw pointer and are still safe `pub extern "C" fn`, so safe Rust can hand
/// them any address at all. That is P4-137 criterion 1, which this change does
/// not do — only criterion 4, the borrow scope. Adding `unsafe` here would warn
/// as unnecessary, which is a neat demonstration of the remaining gap: when the
/// exports are converted, this call site will stop compiling until it is
/// wrapped, and that is the intended signal.
#[test]
fn null_handle_still_returns_the_error_sentinel() {
    assert_eq!(
        tj3Set(std::ptr::null_mut(), TJPARAM_QUALITY, 50),
        -1,
        "tj3Set(NULL) must return -1"
    );
    assert_eq!(
        tj3Get(std::ptr::null_mut(), TJPARAM_QUALITY),
        -1,
        "tj3Get(NULL) must return -1"
    );
}
