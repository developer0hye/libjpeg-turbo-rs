//! A1-2: Smoke test that dlopens the produced cdylib and exercises the
//! TJ3 handle lifecycle via real FFI symbol resolution. This validates
//! that the `extern "C"` symbols are actually exported from the shared
//! object and callable end-to-end from an external process.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::PathBuf;

type TjHandle = *mut c_void;

/// Locate the freshly-built cdylib. `CARGO_CDYLIB_FILE_*` is set by Cargo
/// for tests that depend on the crate's cdylib artifact; for a normal
/// `cargo test -p libjpeg-turbo-rs-capi` we derive the path from
/// `CARGO_MANIFEST_DIR` + `target/<profile>/lib<name><dlext>`.
fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }

    let dlext: &str = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    let prefix: &str = if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    };

    // Walk up from the test exe toward `target/{debug,release}`.
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf = dir.join(format!("{prefix}libjpeg_turbo_rs_capi.{dlext}"));
        if candidate.exists() {
            return candidate;
        }
    }

    panic!(
        "could not locate cdylib for libjpeg_turbo_rs_capi near {}",
        exe.display()
    );
}

// TJPARAM constants — match turbojpeg.h numeric ordering.
// TJPARAM_STOPONWARNING=0, BOTTOMUP=1, NOREALLOC=2, QUALITY=3,
// SUBSAMP=4, JPEGWIDTH=5, JPEGHEIGHT=6, PRECISION=7.
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;

const TJINIT_COMPRESS: c_int = 1; // Matches C: (1 << 0)

#[test]
fn tj3_handle_lifecycle_via_dlopen() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init symbol");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy symbol");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set symbol");
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get symbol");
        let tj3_get_err: libloading::Symbol<unsafe extern "C" fn(TjHandle) -> *const c_char> =
            lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr symbol");

        // Lifecycle.
        let h: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null(), "tj3Init must return a valid handle");

        // Get / Set round-trip on a writable parameter.
        assert_eq!(tj3_set(h, TJPARAM_QUALITY, 77), 0);
        assert_eq!(tj3_get(h, TJPARAM_QUALITY), 77);

        // Subsamp round-trip (0 = TJSAMP_444).
        assert_eq!(tj3_set(h, TJPARAM_SUBSAMP, 0), 0);
        assert_eq!(tj3_get(h, TJPARAM_SUBSAMP), 0);

        // Read-only parameter: attempting to set JPEGWIDTH must fail.
        let rc: c_int = tj3_set(h, TJPARAM_JPEGWIDTH, 512);
        assert_eq!(
            rc, -1,
            "tj3Set on read-only TJPARAM_JPEGWIDTH must return -1"
        );
        let err: *const c_char = tj3_get_err(h);
        assert!(
            !err.is_null(),
            "tj3GetErrorStr must return a non-NULL string after error"
        );
        let msg: &str = CStr::from_ptr(err).to_str().expect("utf8 error message");
        assert!(
            !msg.is_empty(),
            "error message must describe the failure, got empty"
        );

        // NULL handle: tj3Get must return -1 without crashing.
        let null_get: c_int = tj3_get(std::ptr::null_mut(), TJPARAM_QUALITY);
        assert_eq!(null_get, -1, "tj3Get on NULL handle must return -1");

        tj3_destroy(h);

        // Double destroy on NULL is a no-op in libjpeg-turbo; mirror that.
        tj3_destroy(std::ptr::null_mut());
    }
}
