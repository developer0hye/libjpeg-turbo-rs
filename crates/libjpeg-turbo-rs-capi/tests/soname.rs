//! A1-13: verify the produced cdylib carries the libjpeg-compatible
//! SONAME / install_name so dynamic linkers resolve us in place of
//! the stock library.

use std::path::PathBuf;
use std::process::Command;

fn dlext() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}
fn lib_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}
fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf =
            dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dlext()));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate cdylib near {}", exe.display());
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
    assert!(
        stdout.contains("libjpeg.62.dylib"),
        "otool -D did not show libjpeg.62.dylib install_name, got:\n{stdout}"
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
    assert!(
        stdout.contains("libjpeg.so.62"),
        "readelf -d did not show libjpeg.so.62 SONAME, got:\n{stdout}"
    );
}
