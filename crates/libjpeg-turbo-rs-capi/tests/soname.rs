//! A1-13: verify the produced cdylib carries the libjpeg-compatible
//! SONAME / install_name so dynamic linkers resolve us in place of
//! the stock library.

use std::path::PathBuf;
use std::process::Command;

#[path = "support/cdylib.rs"]
mod cdylib_support;

/// The cdylib this Cargo invocation built; see `support/cdylib.rs` for how
/// it is found on each Cargo build-directory layout (P4-276).
fn cdylib_path() -> PathBuf {
    cdylib_support::cdylib_path()
}

#[cfg(target_os = "macos")]
#[test]
fn cdylib_advertises_libjpeg_compatible_install_name_on_macos() {
    let path: PathBuf = cdylib_path();
    // `otool -D` prints the Mach-O install_name (ID) of the library.
    let which = Command::new("which").arg("otool").output().expect("which");
    if !which.status.success() {
        eprintln!("SKIP: otool not on PATH");
        return;
    }

    let out = Command::new("otool")
        .arg("-D")
        .arg(&path)
        .output()
        .expect("otool -D");
    assert!(
        out.status.success(),
        "otool -D failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
    // P4-3 (2026-05-17): default install_name flipped to the v8 ABI
    // canonical `libjpeg.8.dylib`. v6b is still available via
    // CAPI_SONAME / CAPI_INSTALL_NAME + CAPI_ACK_V6B_SONAME=1.
    assert!(
        stdout.contains("libjpeg.8.dylib"),
        "otool -D did not show libjpeg.8.dylib install_name, got:\n{stdout}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn cdylib_advertises_libjpeg_compatible_soname_on_linux() {
    let path: PathBuf = cdylib_path();
    // Prefer `readelf -d`; fall back to `objdump -p`.
    let which_readelf = Command::new("which")
        .arg("readelf")
        .output()
        .expect("which");
    if !which_readelf.status.success() {
        eprintln!("SKIP: readelf not on PATH");
        return;
    }

    let out = Command::new("readelf")
        .arg("-d")
        .arg(&path)
        .output()
        .expect("readelf -d");
    assert!(
        out.status.success(),
        "readelf -d failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
    // P4-3 (2026-05-17): default SONAME flipped to the v8 ABI
    // canonical `libjpeg.so.8`. v6b is still available via
    // CAPI_SONAME=libjpeg.so.62 + CAPI_ACK_V6B_SONAME=1.
    assert!(
        stdout.contains("libjpeg.so.8"),
        "readelf -d did not show libjpeg.so.8 SONAME, got:\n{stdout}"
    );
}
