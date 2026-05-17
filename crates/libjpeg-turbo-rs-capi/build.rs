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
    // the libjpeg API and `libturbojpeg.so.0` (TurboJPEG API). Our struct
    // layout is JPEG_LIB_VERSION=80 (v8), so the canonical SONAME is
    // `libjpeg.so.8` — that matches what a consumer compiled against v8
    // headers expects. The default install_name on macOS is
    // `@rpath/libjpeg.8.dylib`.
    //
    // Selecting a single soname is intentional: a cdylib target emits
    // exactly one binary on disk. Distributions that want a second
    // soname typically ship a symlink (`libturbojpeg.so.0 ->
    // libjpeg.so.8`) alongside the library itself; users who need a
    // different layout can opt in via the `CAPI_SONAME` env var at
    // build time.
    //
    // V6B OPT-IN (see docs/ABI_COMPATIBILITY.md, P4-3): historically
    // this build defaulted to `libjpeg.so.62` for ease of distro
    // replacement, but a consumer compiled against v6b headers can
    // silently corrupt v8-only fields (e.g. `is_baseline`). The v6b
    // SONAME is still available; the simplest opt-in is:
    //
    //   CAPI_ACK_V6B_SONAME=1 cargo build -p libjpeg-turbo-rs-capi --release
    //
    // which auto-implies `CAPI_SONAME=libjpeg.so.62` and
    // `CAPI_INSTALL_NAME=@rpath/libjpeg.62.dylib` so the SONAME and
    // macOS install_name stay in lockstep. Explicit overrides still
    // win if a packager needs a different combination.
    let v6b_soname_acknowledged: bool = env::var("CAPI_ACK_V6B_SONAME")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let soname: String = env::var("CAPI_SONAME").unwrap_or_else(|_| {
        if v6b_soname_acknowledged {
            "libjpeg.so.62".to_string()
        } else {
            "libjpeg.so.8".to_string()
        }
    });
    let install_name_mac: String = env::var("CAPI_INSTALL_NAME").unwrap_or_else(|_| {
        if v6b_soname_acknowledged {
            "@rpath/libjpeg.62.dylib".to_string()
        } else {
            "@rpath/libjpeg.8.dylib".to_string()
        }
    });

    // Loud warning when v6b SONAME is requested without the
    // acknowledgement env. This is the documented-risk path now; v8
    // is the safe default.
    let v6b_soname_chosen: bool =
        soname.contains(".so.62") || install_name_mac.contains(".62.dylib");
    if v6b_soname_chosen && !v6b_soname_acknowledged {
        println!(
            "cargo:warning=libjpeg-turbo-rs-capi: CAPI_SONAME requests v6b \
             (`libjpeg.so.62`) while struct layout is JPEG_LIB_VERSION=80 (v8). \
             Consumers compiled against v6b headers may silently corrupt v8-only \
             fields. See docs/ABI_COMPATIBILITY.md. Set CAPI_ACK_V6B_SONAME=1 to \
             acknowledge the risk and silence this warning."
        );
    }
    // Sanity check the SONAME / install_name pair so packagers can't
    // accidentally ship one v8 ABI surface and one v6b surface from
    // the same build (a mismatch is silently UB at load time on
    // macOS, where install_name resolution is strict).
    let soname_is_v6b: bool = soname.contains(".so.62");
    let install_name_is_v6b: bool = install_name_mac.contains(".62.dylib");
    if soname_is_v6b != install_name_is_v6b {
        println!(
            "cargo:warning=libjpeg-turbo-rs-capi: CAPI_SONAME ({soname}) and \
             CAPI_INSTALL_NAME ({install_name_mac}) disagree on v6b vs v8 ABI. \
             A v6b SONAME paired with a v8 install_name (or vice versa) will \
             break load-time resolution on macOS. Set CAPI_ACK_V6B_SONAME=1 to \
             pick v6b for both, leave both unset for v8, or set both explicitly."
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
