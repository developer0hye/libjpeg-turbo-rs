//! P4-4 phase-1: prove the `unwind_guard!` macro catches panics in
//! `extern "C"` function bodies and returns the documented sentinel
//! instead of unwinding across the FFI boundary.
//!
//! The macro is published with `#[macro_export]` from the capi crate
//! root so test binaries can use it the same way submodules do.
//!
//! When the panic strategy is `abort` (e.g. a future change to the
//! workspace `[profile.release]`), `catch_unwind` always succeeds and
//! these tests would no longer fire the panic branch — they would
//! still pass because the happy path is checked.

use libjpeg_turbo_rs_capi::unwind_guard;
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counts how many times the `()` panic branch was caught. Validates
/// that the macro genuinely funnels through the panic arm rather than
/// silently dropping the panic on the floor.
static UNIT_PANICS_CAUGHT: AtomicUsize = AtomicUsize::new(0);

/// Test-only entry point that mirrors the real `tj3*` shape: panic
/// behind a flag, return a -1 sentinel on catch.
#[no_mangle]
extern "C" fn panic_safety_int_sentinel(should_panic: c_int) -> c_int {
    unwind_guard!(-1, {
        if should_panic != 0 {
            panic!("deliberate panic from panic_safety_int_sentinel");
        }
        42
    })
}

/// Test-only entry point that returns `*mut c_void` — exercises the
/// pointer-sentinel variant the way `tj3Alloc` does.
#[no_mangle]
extern "C" fn panic_safety_ptr_sentinel(should_panic: c_int) -> *mut c_void {
    unwind_guard!(std::ptr::null_mut(), {
        if should_panic != 0 {
            panic!("deliberate panic from panic_safety_ptr_sentinel");
        }
        // Address `0xDEAD` — distinct from NULL so the happy path is
        // unambiguous.
        0xDEAD_usize as *mut c_void
    })
}

/// Test-only entry point with `()` return — exercises the unit-sentinel
/// variant the way `tj3Free` does.
#[no_mangle]
extern "C" fn panic_safety_unit_sentinel(should_panic: c_int) {
    unwind_guard!((), {
        if should_panic != 0 {
            UNIT_PANICS_CAUGHT.fetch_add(1, Ordering::SeqCst);
            panic!("deliberate panic from panic_safety_unit_sentinel");
        }
    })
}

/// Silence the default `print_to_stderr` panic hook so the deliberate
/// panics in these tests don't pollute `cargo test --nocapture` output.
/// The hook is process-global, so we only install it once.
fn install_quiet_panic_hook() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let prior = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // Only suppress panics that originated in panic_safety_*
            // helpers above. Anything else (a genuine bug in macro
            // expansion, the harness, etc.) goes through the prior
            // hook so the developer sees the real failure.
            let location_ok: bool = info
                .location()
                .map(|loc| loc.file().ends_with("capi_panic_safety.rs"))
                .unwrap_or(false);
            if !location_ok {
                prior(info);
            }
        }));
    });
}

#[test]
fn int_sentinel_happy_path_returns_real_value() {
    install_quiet_panic_hook();
    assert_eq!(panic_safety_int_sentinel(0), 42);
}

#[test]
fn int_sentinel_panic_returns_minus_one() {
    install_quiet_panic_hook();
    assert_eq!(panic_safety_int_sentinel(1), -1);
}

#[test]
fn ptr_sentinel_happy_path_returns_real_pointer() {
    install_quiet_panic_hook();
    let p: *mut c_void = panic_safety_ptr_sentinel(0);
    assert!(!p.is_null());
    assert_eq!(p as usize, 0xDEAD);
}

#[test]
fn ptr_sentinel_panic_returns_null() {
    install_quiet_panic_hook();
    let p: *mut c_void = panic_safety_ptr_sentinel(1);
    assert!(p.is_null());
}

#[test]
fn unit_sentinel_happy_path_runs() {
    install_quiet_panic_hook();
    panic_safety_unit_sentinel(0); // must not panic out
}

#[test]
fn unit_sentinel_panic_is_swallowed() {
    install_quiet_panic_hook();
    let before: usize = UNIT_PANICS_CAUGHT.load(Ordering::SeqCst);
    panic_safety_unit_sentinel(1); // panics — caught, process survives
    let after: usize = UNIT_PANICS_CAUGHT.load(Ordering::SeqCst);
    assert_eq!(after, before + 1, "panic body did not execute exactly once");
}
