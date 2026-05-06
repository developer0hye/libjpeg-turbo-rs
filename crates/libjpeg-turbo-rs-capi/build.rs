// Build script for libjpeg-turbo-rs-capi.
//
// Responsibilities:
// - A1-13: SONAME (Linux/BSD) / install_name (macOS) on the produced
//   cdylib so downstream dynamic linkers resolve us in place of the
//   stock `libjpeg.so.62` / `libturbojpeg.so.0` / `libjpeg.62.dylib`.
// - A1-14: pkg-config `.pc` file emission into `OUT_DIR` so packagers
//   can install `libjpeg.pc` / `libturbojpeg.pc` alongside the cdylib
//   and have `pkg-config --libs libjpeg` / `--cflags libturbojpeg`
//   Just Work.

use std::env;
use std::fs;
use std::path::PathBuf;

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
    //
    // CAVEAT (P2-9, see docs/ABI_COMPATIBILITY.md): the default SONAME
    // `libjpeg.so.62` advertises the v6b ABI, but our struct layout is
    // JPEG_LIB_VERSION = 80 (v8). A consumer compiled against v6b
    // headers may silently corrupt v8-only fields (e.g. `is_baseline`).
    // Production deployments should override:
    //   CAPI_SONAME=libjpeg.so.8
    //   CAPI_INSTALL_NAME=@rpath/libjpeg.8.dylib
    let soname: String = env::var("CAPI_SONAME").unwrap_or_else(|_| "libjpeg.so.62".to_string());
    let install_name_mac: String =
        env::var("CAPI_INSTALL_NAME").unwrap_or_else(|_| "@rpath/libjpeg.62.dylib".to_string());

    // Loud warning when the v6b SONAME is paired with v8 struct layout
    // (the documented-risk path). Suppress by setting either CAPI_SONAME
    // explicitly to anything (including the v6b SONAME if you really want
    // it) or CAPI_ACK_V6B_SONAME=1.
    let soname_was_default: bool = env::var("CAPI_SONAME").is_err();
    let install_name_was_default: bool = env::var("CAPI_INSTALL_NAME").is_err();
    let v6b_soname_acknowledged: bool = env::var("CAPI_ACK_V6B_SONAME")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    if (soname_was_default || install_name_was_default) && !v6b_soname_acknowledged {
        println!(
            "cargo:warning=libjpeg-turbo-rs-capi: defaulting SONAME to v6b (`libjpeg.so.62`) \
             but struct layout is JPEG_LIB_VERSION=80 (v8). Consumers compiled against v6b \
             headers may silently corrupt v8-only fields. See docs/ABI_COMPATIBILITY.md. \
             Set CAPI_SONAME=libjpeg.so.8 + CAPI_INSTALL_NAME=@rpath/libjpeg.8.dylib for the \
             safe default, or CAPI_ACK_V6B_SONAME=1 to silence this warning."
        );
    }

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

    // ------------------------------------------------------------------
    // A1-14: write `libjpeg.pc` and `libturbojpeg.pc` into OUT_DIR.
    // ------------------------------------------------------------------
    let out_dir: PathBuf = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let version: &str = env!("CARGO_PKG_VERSION");

    // Prefix defaults to `/usr/local` (standard FHS install target).
    // Packagers typically override with CAPI_PKG_PREFIX=/usr or similar.
    let prefix: String = env::var("CAPI_PKG_PREFIX").unwrap_or_else(|_| "/usr/local".to_string());
    let libdir: String =
        env::var("CAPI_PKG_LIBDIR").unwrap_or_else(|_| "${prefix}/lib".to_string());
    let includedir: String =
        env::var("CAPI_PKG_INCLUDEDIR").unwrap_or_else(|_| "${prefix}/include".to_string());

    let libjpeg_pc: String = format!(
        "prefix={prefix}\n\
         exec_prefix=${{prefix}}\n\
         libdir={libdir}\n\
         includedir={includedir}\n\
         \n\
         Name: libjpeg\n\
         Description: A SIMD-accelerated JPEG codec (libjpeg-turbo-rs shim)\n\
         Version: {version}\n\
         Libs: -L${{libdir}} -ljpeg\n\
         Cflags: -I${{includedir}}\n"
    );

    let libturbojpeg_pc: String = format!(
        "prefix={prefix}\n\
         exec_prefix=${{prefix}}\n\
         libdir={libdir}\n\
         includedir={includedir}\n\
         \n\
         Name: libturbojpeg\n\
         Description: A SIMD-accelerated JPEG codec (TurboJPEG API, libjpeg-turbo-rs shim)\n\
         Version: {version}\n\
         Libs: -L${{libdir}} -lturbojpeg\n\
         Cflags: -I${{includedir}}\n"
    );

    let pc_dir: PathBuf = out_dir.join("pkgconfig");
    fs::create_dir_all(&pc_dir).expect("mkdir pkgconfig");
    fs::write(pc_dir.join("libjpeg.pc"), &libjpeg_pc).expect("write libjpeg.pc");
    fs::write(pc_dir.join("libturbojpeg.pc"), &libturbojpeg_pc).expect("write libturbojpeg.pc");

    // Expose the pkgconfig dir to tests (via env!()) and to downstream
    // packaging scripts (via cargo:metadata output).
    println!("cargo:pkgconfig_dir={}", pc_dir.display());
    println!("cargo:rustc-env=CAPI_PKGCONFIG_DIR={}", pc_dir.display());

    // Re-run when any of the pkg-config-affecting inputs change.
    println!("cargo:rerun-if-env-changed=CAPI_SONAME");
    println!("cargo:rerun-if-env-changed=CAPI_INSTALL_NAME");
    println!("cargo:rerun-if-env-changed=CAPI_ACK_V6B_SONAME");
    println!("cargo:rerun-if-env-changed=CAPI_PKG_PREFIX");
    println!("cargo:rerun-if-env-changed=CAPI_PKG_LIBDIR");
    println!("cargo:rerun-if-env-changed=CAPI_PKG_INCLUDEDIR");
    println!("cargo:rerun-if-changed=build.rs");
}
