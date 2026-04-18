// Build script for libjpeg-turbo-rs-capi.
//
// Responsibilities:
// - A1-13: SONAME (Linux/BSD) / install_name (macOS) on the produced
//   cdylib so downstream dynamic linkers resolve us in place of the
//   stock `libjpeg.so.62` / `libturbojpeg.so.0` / `libjpeg.62.dylib`.
// - A1-14 will add pkg-config `.pc` emission into `OUT_DIR`.

use std::env;

fn main() {
    let target_os: String = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // The cdylib we produce is meant to be binary-compatible with both
    // `libjpeg.so.62` (libjpeg API) and `libturbojpeg.so.0` (TurboJPEG
    // API). We pick `libjpeg.so.62` as the SONAME because that's what
    // distro packages name-pin to — a downstream binary that links
    // against stock libjpeg will search for this soname first.
    //
    // Selecting a single soname is intentional: a cdylib target emits
    // exactly one binary on disk. Distributions that want the second
    // soname typically ship a symlink (libturbojpeg.so.0 ->
    // libjpeg.so.62) alongside the library itself; users who need that
    // layout can opt in via the `CAPI_SONAME` env var at build time.
    let soname: String = env::var("CAPI_SONAME").unwrap_or_else(|_| "libjpeg.so.62".to_string());
    let install_name_mac: String =
        env::var("CAPI_INSTALL_NAME").unwrap_or_else(|_| "@rpath/libjpeg.62.dylib".to_string());

    match target_os.as_str() {
        "linux" | "android" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,{soname}");
        }
        "macos" | "ios" | "tvos" | "watchos" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,{install_name_mac}");
        }
        _ => {
            // Windows / WASM / others: no soname concept; nothing to emit.
        }
    }

    // Re-run when the user overrides either variable.
    println!("cargo:rerun-if-env-changed=CAPI_SONAME");
    println!("cargo:rerun-if-env-changed=CAPI_INSTALL_NAME");
    println!("cargo:rerun-if-changed=build.rs");
}
