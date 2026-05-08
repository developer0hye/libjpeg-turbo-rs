//! Generated C-side ABI cross-check.
//!
//! For every Rust mirror of a public libjpeg struct the shim exposes,
//! this test compiles a tiny C program against the real upstream
//! `jpeglib.h` (from the `references/libjpeg-turbo/` submodule) at
//! `JPEG_LIB_VERSION = 80`, prints `offsetof(struct STRUCT, FIELD)`
//! plus `sizeof(struct STRUCT)` for the named fields, then asserts
//! the values equal what Rust computes via `mem::offset_of!` /
//! `mem::size_of!`.
//!
//! Why this exists. The hand-typed offsets in `jpeglib.rs` were
//! computed by a one-time `offsetof` print and pasted in as constants.
//! If a future upstream `jpeglib.h` shuffles a field — or if our Rust
//! mirror grows a misordered field — neither side notices on its own.
//! This test catches the divergence at `cargo test` time.
//!
//! Why we also probe `sizeof`. A per-field `offsetof` check is blind
//! to *trailing* drift: if the Rust mirror truncates the C struct's
//! tail (a missing pointer, a removed `JPEG_LIB_VERSION ≥ X` field),
//! every named field still resolves at the same offset on both sides,
//! so the per-field check passes. The `sizeof` probe surfaces that
//! tail-truncation bug as `Rust mirror is N bytes, C is M bytes`.
//!
//! Skip-with-reason cases (legitimate dev-machine skips):
//! - No `cc` on PATH.
//! - Cross-compile target where host `cc` cannot match the target ABI.
//! - Non-64-bit host (matches the `cfg(target_pointer_width = "64")`
//!   gate on the Rust assertion block; the offsets change on ILP32).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// Outcome of the C-harness probe: either parsed offsets + sizeof, or
/// a human-readable reason the probe could not run on this host.
struct CcProbeResult {
    offsets: HashMap<String, usize>,
    sizeof: usize,
}

enum CcProbeOutcome {
    Ok(CcProbeResult),
    Skip(String),
}

/// Build a one-shot C harness that prints `name=offset` lines for
/// each requested field of `struct STRUCT_NAME`, plus a final
/// `__sizeof__=N` line, and run it through the system `cc`.
///
/// Returns `Skip(reason)` for environmental gaps (missing compiler,
/// missing submodule, missing system headers); returns `Ok(probe)`
/// once the harness produced parseable output.
fn cc_offsetof_for_struct(struct_name: &str, field_names: &[&str]) -> CcProbeOutcome {
    if std::mem::size_of::<usize>() != 8 {
        return CcProbeOutcome::Skip(format!(
            "ABI cross-check only runs on 64-bit hosts; host has size_of(usize)={}",
            std::mem::size_of::<usize>(),
        ));
    }

    let workspace_root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let upstream_src: PathBuf = workspace_root.join("references/libjpeg-turbo/src");
    let upstream_jpeglib_h: PathBuf = upstream_src.join("jpeglib.h");
    if !upstream_jpeglib_h.exists() {
        return CcProbeOutcome::Skip(format!(
            "upstream jpeglib.h not found at {:?} (submodule not initialized?)",
            upstream_jpeglib_h
        ));
    }

    let cc: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let cc_check: bool = Command::new(&cc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !cc_check {
        return CcProbeOutcome::Skip(format!("C compiler `{}` not found or not runnable", cc));
    }

    let mut c_src: String = String::new();
    c_src.push_str("#include <stdio.h>\n");
    c_src.push_str("#include <stddef.h>\n");
    c_src.push_str("#include <setjmp.h>\n");
    c_src.push_str("#include <jpeglib.h>\n");
    c_src.push_str("int main(void) {\n");
    for name in field_names {
        c_src.push_str(&format!(
            "  printf(\"{name}=%zu\\n\", offsetof(struct {struct_name}, {name}));\n"
        ));
    }
    c_src.push_str(&format!(
        "  printf(\"__sizeof__=%zu\\n\", sizeof(struct {struct_name}));\n"
    ));
    c_src.push_str("  return 0;\n}\n");

    // Minimal `jconfig.h` mirroring the upstream v8 defaults the CMakeLists
    // would have substituted in. JPEG_LIB_VERSION pins the ABI variant the
    // Rust mirror was hand-typed against.
    let jconfig_h: &str = "\
#define JPEG_LIB_VERSION 80
#define LIBJPEG_TURBO_VERSION 3.1.0
#define LIBJPEG_TURBO_VERSION_NUMBER 3001000
#define C_ARITH_CODING_SUPPORTED 1
#define D_ARITH_CODING_SUPPORTED 1
#define MEM_SRCDST_SUPPORTED 1
#define WITH_SIMD 1
#define BITS_IN_JSAMPLE 8
";

    let tmp: tempfile::TempDir = match tempfile::tempdir() {
        Ok(t) => t,
        Err(e) => return CcProbeOutcome::Skip(format!("mkdir tempdir failed: {e}")),
    };
    let jconfig_h_path: PathBuf = tmp.path().join("jconfig.h");
    let c_src_path: PathBuf = tmp.path().join("abi_offsets.c");
    let bin_path: PathBuf = tmp.path().join("abi_offsets");

    if let Err(e) = std::fs::write(&jconfig_h_path, jconfig_h) {
        return CcProbeOutcome::Skip(format!("write jconfig.h failed: {e}"));
    }
    if let Err(e) = std::fs::write(&c_src_path, &c_src) {
        return CcProbeOutcome::Skip(format!("write abi_offsets.c failed: {e}"));
    }

    let compile = match Command::new(&cc)
        .args(["-O0", "-Wno-implicit-function-declaration", "-I"])
        .arg(tmp.path())
        .arg("-I")
        .arg(&upstream_src)
        .arg("-o")
        .arg(&bin_path)
        .arg(&c_src_path)
        .output()
    {
        Ok(o) => o,
        Err(e) => return CcProbeOutcome::Skip(format!("invoking cc failed: {e}")),
    };
    if !compile.status.success() {
        let stderr: String = String::from_utf8_lossy(&compile.stderr).to_string();
        // Environmental compile failures (missing headers, broken
        // toolchain) → skip-with-reason, not test failure. The test is
        // a drift gate, not a toolchain gate.
        if stderr.contains("No such file or directory")
            || stderr.contains("cannot find")
            || stderr.contains("not found")
        {
            return CcProbeOutcome::Skip(format!(
                "cc could not compile the harness for struct {struct_name} (env issue):\n{stderr}"
            ));
        }
        panic!(
            "cc failed to compile abi_offsets.c for struct {struct_name}:\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&compile.stdout),
            stderr
        );
    }

    let run = Command::new(&bin_path)
        .output()
        .unwrap_or_else(|e| panic!("run abi_offsets harness for {struct_name}: {e}"));
    assert!(
        run.status.success(),
        "abi_offsets harness for {struct_name} exited non-zero: {:?}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout: String = String::from_utf8(run.stdout).expect("utf8 stdout");

    let mut offsets: HashMap<String, usize> = HashMap::new();
    let mut sizeof: Option<usize> = None;
    for line in stdout.lines() {
        if let Some((k, v)) = line.split_once('=') {
            if let Ok(off) = v.trim().parse::<usize>() {
                if k == "__sizeof__" {
                    sizeof = Some(off);
                } else {
                    offsets.insert(k.to_string(), off);
                }
            }
        }
    }
    let sizeof: usize = sizeof.unwrap_or_else(|| {
        panic!(
            "C harness did not emit `__sizeof__=N` for struct {struct_name}; raw output:\n{stdout}"
        )
    });

    CcProbeOutcome::Ok(CcProbeResult { offsets, sizeof })
}

/// Compare Rust-side offsets + Rust mirror sizeof against the C probe
/// result and panic with a side-by-side diff if anything diverges.
fn assert_no_drift(
    struct_name: &str,
    rust_fields: &[(&str, usize)],
    rust_sizeof: usize,
    probe: &CcProbeResult,
) {
    let mut mismatches: Vec<String> = Vec::new();
    for (field, rust_off) in rust_fields {
        match probe.offsets.get(*field) {
            None => mismatches.push(format!("field `{field}`: missing from C output")),
            Some(&c_off) if c_off != *rust_off => {
                mismatches.push(format!(
                    "field `{field}`: Rust says offset {rust_off}, C says {c_off}"
                ));
            }
            _ => {}
        }
    }
    if rust_sizeof != probe.sizeof {
        mismatches.push(format!(
            "sizeof: Rust mirror is {rust_sizeof} bytes, C `sizeof(struct {struct_name})` is {} \
             bytes — trailing field(s) are unmirrored or padding diverges",
            probe.sizeof,
        ));
    }
    assert!(
        mismatches.is_empty(),
        "ABI offset / sizeof divergence between Rust mirror and upstream jpeglib.h \
         (JPEG_LIB_VERSION=80) for struct {struct_name}:\n  {}",
        mismatches.join("\n  "),
    );
}

// ---------------------------------------------------------------------------
// `struct jpeg_decompress_struct` cross-check (P2-4 baseline).
// ---------------------------------------------------------------------------

fn rust_offsets_decompress() -> Vec<(&'static str, usize)> {
    use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;
    use std::mem::offset_of;

    vec![
        ("err", offset_of!(JpegDecompressPublic, err)),
        ("mem", offset_of!(JpegDecompressPublic, mem)),
        ("progress", offset_of!(JpegDecompressPublic, progress)),
        ("client_data", offset_of!(JpegDecompressPublic, client_data)),
        (
            "is_decompressor",
            offset_of!(JpegDecompressPublic, is_decompressor),
        ),
        (
            "global_state",
            offset_of!(JpegDecompressPublic, global_state),
        ),
        ("src", offset_of!(JpegDecompressPublic, src)),
        ("image_width", offset_of!(JpegDecompressPublic, image_width)),
        (
            "image_height",
            offset_of!(JpegDecompressPublic, image_height),
        ),
        (
            "num_components",
            offset_of!(JpegDecompressPublic, num_components),
        ),
        (
            "jpeg_color_space",
            offset_of!(JpegDecompressPublic, jpeg_color_space),
        ),
        (
            "out_color_space",
            offset_of!(JpegDecompressPublic, out_color_space),
        ),
        ("scale_num", offset_of!(JpegDecompressPublic, scale_num)),
        ("scale_denom", offset_of!(JpegDecompressPublic, scale_denom)),
        (
            "output_gamma",
            offset_of!(JpegDecompressPublic, output_gamma),
        ),
        (
            "buffered_image",
            offset_of!(JpegDecompressPublic, buffered_image),
        ),
        (
            "raw_data_out",
            offset_of!(JpegDecompressPublic, raw_data_out),
        ),
        (
            "quantize_colors",
            offset_of!(JpegDecompressPublic, quantize_colors),
        ),
        ("coef_bits", offset_of!(JpegDecompressPublic, coef_bits)),
        (
            "quant_tbl_ptrs",
            offset_of!(JpegDecompressPublic, quant_tbl_ptrs),
        ),
        (
            "dc_huff_tbl_ptrs",
            offset_of!(JpegDecompressPublic, dc_huff_tbl_ptrs),
        ),
        (
            "ac_huff_tbl_ptrs",
            offset_of!(JpegDecompressPublic, ac_huff_tbl_ptrs),
        ),
        (
            "data_precision",
            offset_of!(JpegDecompressPublic, data_precision),
        ),
        ("comp_info", offset_of!(JpegDecompressPublic, comp_info)),
        ("is_baseline", offset_of!(JpegDecompressPublic, is_baseline)),
        (
            "progressive_mode",
            offset_of!(JpegDecompressPublic, progressive_mode),
        ),
        ("arith_code", offset_of!(JpegDecompressPublic, arith_code)),
    ]
}

#[test]
fn rust_offsets_match_upstream_jpeglib_h_at_lib_version_80() {
    use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;

    let rust_fields: Vec<(&'static str, usize)> = rust_offsets_decompress();
    let names: Vec<&str> = rust_fields.iter().map(|(n, _)| *n).collect();
    let probe: CcProbeResult = match cc_offsetof_for_struct("jpeg_decompress_struct", &names) {
        CcProbeOutcome::Ok(r) => r,
        CcProbeOutcome::Skip(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    let rust_sizeof: usize = std::mem::size_of::<JpegDecompressPublic>();
    assert_no_drift("jpeg_decompress_struct", &rust_fields, rust_sizeof, &probe);
}

// ---------------------------------------------------------------------------
// `struct jpeg_marker_struct` cross-check (P3-1 deferred work).
// ---------------------------------------------------------------------------
//
// The Rust mirror lives at `JpegMarkerStructPublic` (jpeglib.rs:177-183).
// This struct is the one classic C consumers walk via `jpeg_saved_marker_ptr`
// after `jpeg_save_markers`; the layout matters when, e.g., `jpegtran -copy
// all` reads `marker_list` and iterates `next` chains. Stock `jpegtran -copy
// all` already byte-matches upstream (P0-4 closure), which exercises this
// struct at runtime — but a runtime byte-match check does not catch every
// permutation of "consumer reads `data_length` at the wrong offset and
// silently truncates" that real third-party code might do. The compile-time
// `offsetof` cross-check catches the layout drift directly.

fn rust_offsets_marker() -> Vec<(&'static str, usize)> {
    use libjpeg_turbo_rs_capi::jpeglib::JpegMarkerStructPublic;
    use std::mem::offset_of;

    vec![
        ("next", offset_of!(JpegMarkerStructPublic, next)),
        ("marker", offset_of!(JpegMarkerStructPublic, marker)),
        (
            "original_length",
            offset_of!(JpegMarkerStructPublic, original_length),
        ),
        (
            "data_length",
            offset_of!(JpegMarkerStructPublic, data_length),
        ),
        ("data", offset_of!(JpegMarkerStructPublic, data)),
    ]
}

#[test]
fn rust_offsets_match_jpeg_marker_struct_at_lib_version_80() {
    use libjpeg_turbo_rs_capi::jpeglib::JpegMarkerStructPublic;

    let rust_fields: Vec<(&'static str, usize)> = rust_offsets_marker();
    let names: Vec<&str> = rust_fields.iter().map(|(n, _)| *n).collect();
    let probe: CcProbeResult = match cc_offsetof_for_struct("jpeg_marker_struct", &names) {
        CcProbeOutcome::Ok(r) => r,
        CcProbeOutcome::Skip(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    let rust_sizeof: usize = std::mem::size_of::<JpegMarkerStructPublic>();
    assert_no_drift("jpeg_marker_struct", &rust_fields, rust_sizeof, &probe);
}
