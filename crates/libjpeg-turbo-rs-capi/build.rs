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

    // Always generate it; only Linux-family targets hand it to the linker.
    let _ = write_version_script(&gnu_version_script());

    match target_os.as_str() {
        "linux" | "android" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,{soname}");
            note_gnu_version_script_owner(&soname);
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

/// P4-81: GNU ELF symbol versions for the classic libjpeg surface.
///
/// The staged `libjpeg.so.8` carried a SONAME but no version nodes, so glibc
/// bound prebuilt consumers (OpenCV, libtiff, GDAL, …) through its
/// unversioned-fallback path and warned `no version information available`.
/// That is not a warning-free distro replacement, and stricter loaders may
/// refuse it outright.
///
/// The map mirrors upstream `src/libjpeg.map.in` — `LIBJPEGTURBO_@ver@` owning
/// the two MEM_SRCDST entry points and localising the SIMD/const internals,
/// `LIBJPEG_@ver@` owning the reference API — with one deliberate deviation.
///
/// **Upstream's `LIBJPEG_8.0` node is `global: *`.** It can afford that
/// because it builds two libraries: `libjpeg.so.8` from this map and
/// `libturbojpeg.so.0` from `src/turbojpeg-mapfile`, which assigns each `tj*`
/// symbol to the `TURBOJPEG_1.0`…`TURBOJPEG_3.0` node it was introduced in.
/// We ship **one** artifact carrying both surfaces, so a `*` catch-all here
/// would stamp every `tj*` symbol as a reference libjpeg export — exactly the
/// mislabelling this item forbids — and re-versioning them under any node of
/// our own would be worse: a consumer linked against a real libturbojpeg
/// requests `tjInitCompress@TURBOJPEG_1.0`, and the loader fails outright when
/// the library offers that name under a different version.
///
/// So the map has no catch-all. Symbols matched by no node keep default,
/// unversioned visibility — which is precisely their status today, so the
/// TurboJPEG and crate-only surfaces are unchanged while the classic API gains
/// the nodes prebuilt consumers look for.
///
/// Only emitted for a v8 libjpeg SONAME: a caller who overrode `CAPI_SONAME`
/// to something else is not building the artifact these nodes describe, and
/// applying a v8 map to it would be a silently wrong label.
/// # Why this does not hand the map to rustc's linker invocation
///
/// It cannot. rustc passes its own version script for every cdylib —
/// `-Wl,--version-script=.../deps/rustc*/list`, an *anonymous* tag
/// (`{ global: …; local: *; };`) that exports the `#[no_mangle]` items and
/// hides the rest — and GNU ld refuses to combine an anonymous version tag
/// with named ones. Adding ours as a second script fails the link outright:
///
/// ```text
/// /usr/bin/ld: anonymous version tag cannot be combined with other version tags
/// ```
///
/// This was verified against binutils 2.47 across every script surface (see
/// P4-81 in `docs/last_mile/phase4.md` for the four-case experiment):
/// a second `--version-script`, a `VERSION { … }` block in a linker-script
/// input file, and `.symver` directives in the object all fail — `.symver`
/// differently, with `version node not found`, because the directive *attaches*
/// a symbol to a node without *defining* one. rustc 1.94.1 exposes no `-C` or
/// `-Z` knob to suppress its own script.
///
/// The one configuration that works is a single named script carrying
/// `local: *` itself, which means whoever owns the link owns the versioning.
/// For a cdylib that is rustc, so the versioned artifact is produced instead by
/// `scripts/install_capi.sh`, which relinks it from the `staticlib` this crate
/// already builds. The acceptance criterion is about the installed library that
/// prebuilt consumers bind to, so that is the right artifact to version.
///
/// The map is still generated here, and its path published as
/// `CAPI_VERSION_SCRIPT`, because the install script and the content tests both
/// consume it.
fn note_gnu_version_script_owner(soname: &str) {
    if !soname.starts_with("libjpeg.so.8") {
        println!(
            "cargo:warning=libjpeg-turbo-rs-capi: SONAME `{soname}` is not the v8 \
             libjpeg identity, so no GNU symbol-version script applies. Prebuilt \
             consumers of this artifact will see no version information (P4-81)."
        );
    }
}

/// Write the map to `OUT_DIR` and publish its path.
///
/// Called on every platform, not only where the linker consumes it, so
/// `tests/capi_symbol_versions.rs` can assert the content a Linux build would
/// get without needing a Linux host — the node names and the deliberate
/// absence of a catch-all are the part most likely to regress, and they are
/// verifiable anywhere.
fn write_version_script(map: &str) -> PathBuf {
    let out_dir: PathBuf = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let map_path: PathBuf = out_dir.join("libjpeg.map");
    fs::write(&map_path, map).expect("write libjpeg.map");
    println!("cargo:rustc-env=CAPI_VERSION_SCRIPT={}", map_path.display());
    map_path
}

/// The GNU version script text. See `note_gnu_version_script_owner` for why it has
/// no catch-all node.
fn gnu_version_script() -> String {
    // Classic exports that are not `jpeg_`-prefixed. Upstream declares these
    // in `jpegint.h` and consumers of the shared library do link them, so they
    // belong in the reference node rather than falling through unversioned.
    const CLASSIC_NON_JPEG_PREFIXED: &[&str] = &["jcopy_block_row", "jdiv_round_up"];

    let mut map: String = String::new();
    map.push_str("LIBJPEGTURBO_8.0 {\n  global:\n");
    // Upstream places the MEM_SRCDST pair here, not in LIBJPEG_8.0. An exact
    // name outranks the `jpeg_*` pattern below, which is the same precedence
    // upstream relies on against its own `*`.
    map.push_str("    jpeg_mem_dest;\n    jpeg_mem_src;\n");
    map.push_str("  local:\n    jsimd_*;\n    jconst_*;\n};\n\n");

    // Crate-private accessors. They share the `jpeg_` prefix, so the `jpeg_*`
    // pattern in LIBJPEG_8.0 below would otherwise stamp all 16 of them as
    // reference libjpeg v8 API — 16 entry points no real libjpeg has, visible
    // to anyone running `readelf --dyn-syms` on our `libjpeg.so.8` (P4-129).
    //
    // They are listed by exact name because an exact match outranks a pattern;
    // that is the same precedence the MEM_SRCDST pair above relies on, and it
    // is deterministic where pattern-vs-pattern ordering is not.
    //
    // They stay dynamically visible rather than being made `local:` because
    // eight dlopen-based suites resolve them out of this very cdylib. Moving
    // them to a distinct, obviously-not-upstream node keeps those tests working
    // while ending the mislabelling, which is the alternative
    // `docs/last_mile/phase4.md`'s P4-129 acceptance criterion 1 permits.
    map.push_str("LIBJPEGTURBORS_PRIVATE_1.0 {\n  global:\n");
    for name in CRATE_PRIVATE_TEST_ACCESSORS {
        map.push_str(&format!("    {name};\n"));
    }
    map.push_str("};\n\n");

    map.push_str("LIBJPEG_8.0 {\n  global:\n    jpeg_*;\n");
    for name in CLASSIC_NON_JPEG_PREFIXED {
        map.push_str(&format!("    {name};\n"));
    }
    map.push_str("};\n");
    map
}

/// The `jpeg_capi_test_*` accessors defined in `src/jpeglib.rs`.
///
/// Kept in one place so the version script and
/// `tests/capi_symbol_versions.rs` cannot drift apart: a 17th accessor added
/// without updating this list falls through to `jpeg_*` and the reference-node
/// allowlist assertion fails.
pub(crate) const CRATE_PRIVATE_TEST_ACCESSORS: &[&str] = &[
    "jpeg_capi_test_arith_code",
    "jpeg_capi_test_density_unit",
    "jpeg_capi_test_dimensions",
    "jpeg_capi_test_get_compress_state",
    "jpeg_capi_test_marker_list",
    "jpeg_capi_test_output_dims",
    "jpeg_capi_test_set_arith_code",
    "jpeg_capi_test_set_compress_dims",
    "jpeg_capi_test_set_optimize_coding",
    "jpeg_capi_test_set_out_cs",
    "jpeg_capi_test_set_progressive",
    "jpeg_capi_test_set_restart_in_rows",
    "jpeg_capi_test_set_restart_interval",
    "jpeg_capi_test_set_smoothing_factor",
    "jpeg_capi_test_x_density",
    "jpeg_capi_test_y_density",
];
