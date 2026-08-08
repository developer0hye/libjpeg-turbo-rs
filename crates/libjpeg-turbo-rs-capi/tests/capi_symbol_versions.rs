//! P4-81: GNU ELF symbol versions on the classic `libjpeg.so.8` artifact.
//!
//! The staged cdylib carried a SONAME but no version nodes, so glibc bound
//! prebuilt consumers through its unversioned-fallback path and warned
//! `no version information available`. That is not a warning-free distro
//! replacement, and stricter loaders may refuse it outright.
//!
//! Two layers, because only one of them needs Linux:
//!
//! * the **script content** — node names, the reference-API assignment, and
//!   the deliberate *absence* of a catch-all — is asserted on every platform
//!   from the generated file, since that is the part most likely to regress
//!   and it is verifiable anywhere;
//! * the **ELF result** — that the linker actually produced those nodes and
//!   attached them to the right symbols — is asserted with `readelf` where
//!   ELF versioning exists.

#[path = "support/cdylib.rs"]
mod cdylib_support;

use std::path::{Path, PathBuf};
use std::process::Command;

/// The version script `build.rs` generated for this build.
///
/// `build.rs` writes it on every platform precisely so this test can read it
/// without a Linux host.
fn version_script() -> String {
    let path: &str = env!("CAPI_VERSION_SCRIPT");
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("could not read the generated version script {path}: {e}"))
}

fn is_ci() -> bool {
    std::env::var("CI")
        .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

/// The two nodes upstream defines, with the same membership.
#[test]
fn version_script_matches_the_upstream_node_layout() {
    let script: String = version_script();

    assert!(
        script.contains("LIBJPEGTURBO_8.0 {"),
        "the LIBJPEGTURBO_8.0 node is missing; prebuilt consumers resolve \
         jpeg_mem_src/jpeg_mem_dest through it:\n{script}"
    );
    assert!(
        script.contains("LIBJPEG_8.0 {"),
        "the LIBJPEG_8.0 node is missing — this is the node whose absence \
         produces `no version information available`:\n{script}"
    );

    // Upstream puts the MEM_SRCDST pair in the TURBO node, not the reference
    // node (src/libjpeg.map.in + the MEM_SRCDST_FUNCTIONS expansion at
    // CMakeLists.txt:349). Assigning them to LIBJPEG_8.0 would label two
    // libjpeg-turbo extensions as reference v8 API.
    let turbo: &str = section(&script, "LIBJPEGTURBO_8.0");
    assert!(
        turbo.contains("jpeg_mem_dest;") && turbo.contains("jpeg_mem_src;"),
        "jpeg_mem_dest / jpeg_mem_src must live in LIBJPEGTURBO_8.0:\n{turbo}"
    );
    assert!(
        turbo.contains("jsimd_*;") && turbo.contains("jconst_*;"),
        "the SIMD and const internals must stay local, as upstream does:\n{turbo}"
    );

    let reference: &str = section(&script, "LIBJPEG_8.0");
    assert!(
        reference.contains("jpeg_*;"),
        "the reference node must claim the classic jpeg_* surface:\n{reference}"
    );
    for name in ["jcopy_block_row", "jdiv_round_up"] {
        assert!(
            reference.contains(&format!("{name};")),
            "classic export `{name}` is not `jpeg_`-prefixed and would fall \
             through unversioned:\n{reference}"
        );
    }
}

/// The script must have no catch-all, which is where it deliberately parts
/// company with upstream's own map.
#[test]
fn version_script_has_no_catch_all_node() {
    let script: String = version_script();

    // Upstream's LIBJPEG_8.0 is `global: *`, which it can afford because it
    // builds libturbojpeg.so.0 from a *separate* map. We ship one artifact
    // carrying both surfaces, so a catch-all here would stamp all 63 `tj*`
    // exports as reference libjpeg API. Re-versioning them under a node of our
    // own would be worse still: a consumer linked against a real libturbojpeg
    // asks for `tjInitCompress@TURBOJPEG_1.0`, and the loader fails outright
    // when the library offers that name under a different version. Leaving
    // them unmatched keeps their present, working, unversioned status.
    let global_star: bool = script
        .lines()
        .map(str::trim)
        .any(|line| line == "*;" || line == "global: *;");
    assert!(
        !global_star,
        "the version script has a catch-all node, which would label every \
         TurboJPEG and crate-only export as a reference libjpeg symbol:\n{script}"
    );
    assert!(
        !script.contains("tj"),
        "TurboJPEG symbols must not be named in the classic map at all:\n{script}"
    );
}

fn section<'a>(script: &'a str, node: &str) -> &'a str {
    let start: usize = script
        .find(&format!("{node} {{"))
        .unwrap_or_else(|| panic!("node {node} not found in:\n{script}"));
    let end: usize = script[start..]
        .find("};")
        .unwrap_or_else(|| panic!("node {node} is unterminated"));
    &script[start..start + end]
}

/// The linker must have produced the nodes, and attached the classic API to
/// them. ELF-only; every other platform reports why it did not run.
#[test]
fn cdylib_exports_the_reference_version_nodes() {
    if !cfg!(target_os = "linux") {
        eprintln!(
            "SKIP: GNU symbol versioning is an ELF feature; this host is {}",
            std::env::consts::OS
        );
        return;
    }
    let readelf: PathBuf = match which("readelf") {
        Some(path) => path,
        None => {
            assert!(
                !is_ci(),
                "CI images ship binutils, so readelf must be present"
            );
            eprintln!("SKIP: readelf is not on PATH");
            return;
        }
    };
    let lib: PathBuf = cdylib_support::cargo_built_cdylib_path()
        .unwrap_or_else(|e| panic!("could not locate the cdylib under test: {e}"));

    let out = Command::new(&readelf)
        .arg("--version-info")
        .arg("--wide")
        .arg(&lib)
        .output()
        .unwrap_or_else(|e| panic!("failed to run readelf: {e}"));
    assert!(
        out.status.success(),
        "readelf --version-info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let info: String = String::from_utf8_lossy(&out.stdout).to_string();

    for node in ["LIBJPEG_8.0", "LIBJPEGTURBO_8.0"] {
        assert!(
            info.contains(node),
            "the linked cdylib defines no `{node}` version node, so consumers \
             still get `no version information available`:\n{info}"
        );
    }

    // Spot-check the assignment rather than only the node's existence: a map
    // that defines the nodes but matches nothing would satisfy the check above
    // while leaving every symbol unversioned.
    let symbols = Command::new(&readelf)
        .arg("--dyn-syms")
        .arg("--wide")
        .arg(&lib)
        .output()
        .unwrap_or_else(|e| panic!("failed to run readelf --dyn-syms: {e}"));
    let dyn_syms: String = String::from_utf8_lossy(&symbols.stdout).to_string();

    for (symbol, node) in [
        ("jpeg_CreateDecompress", "LIBJPEG_8.0"),
        ("jpeg_read_header", "LIBJPEG_8.0"),
        ("jpeg_mem_src", "LIBJPEGTURBO_8.0"),
        ("jpeg_mem_dest", "LIBJPEGTURBO_8.0"),
    ] {
        let line: &str = dyn_syms
            .lines()
            .find(|l| l.contains(symbol))
            .unwrap_or_else(|| panic!("{symbol} is not exported at all:\n{dyn_syms}"));
        assert!(
            line.contains(node),
            "{symbol} must be versioned `@@{node}`, got: {line}"
        );
    }

    // TurboJPEG exports stay unversioned on purpose — see
    // `version_script_has_no_catch_all_node`.
    if let Some(line) = dyn_syms.lines().find(|l| l.contains("tj3Init")) {
        assert!(
            !line.contains("LIBJPEG_8.0") && !line.contains("LIBJPEGTURBO_8.0"),
            "a TurboJPEG export was labelled as reference libjpeg API: {line}"
        );
    }

    eprintln!("cdylib_exports_the_reference_version_nodes: LIBJPEG_8.0 + LIBJPEGTURBO_8.0 present");
}

fn which(tool: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(tool).output().ok()?;
    out.status
        .success()
        .then(|| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string()))
}

/// A non-v8 SONAME must not silently receive the v8 map.
#[test]
fn documented_policy_matches_the_generator() {
    let build_rs: String =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
            .expect("read build.rs");
    assert!(
        build_rs.contains("soname.starts_with(\"libjpeg.so.8\")"),
        "the version script must be gated on the v8 SONAME: applying a v8 map \
         to an artifact built with another identity is a silently wrong label"
    );
}
