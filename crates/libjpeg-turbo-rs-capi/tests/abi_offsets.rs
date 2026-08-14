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
//!   gates on the Rust assertion blocks — decompress and, since the
//!   P4-139 32-bit fix, encode too; the offsets change on ILP32).

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

// ---------------------------------------------------------------------------
// `struct jpeg_destination_mgr` cross-check (P3-1 deferred work).
// ---------------------------------------------------------------------------
//
// Mirrors `JpegDestinationMgr` in `jpeglib.rs`. C consumers that build a
// custom destination manager (e.g. an in-memory ring buffer) write
// `next_output_byte` and `free_in_buffer` at offsets 0 and 8 (LP64), so a
// layout drift here surfaces as garbled output rather than an obvious
// crash.

fn rust_offsets_destination_mgr() -> Vec<(&'static str, usize)> {
    use libjpeg_turbo_rs_capi::jpeglib::JpegDestinationMgr;
    use std::mem::offset_of;

    vec![
        (
            "next_output_byte",
            offset_of!(JpegDestinationMgr, next_output_byte),
        ),
        (
            "free_in_buffer",
            offset_of!(JpegDestinationMgr, free_in_buffer),
        ),
        (
            "init_destination",
            offset_of!(JpegDestinationMgr, init_destination),
        ),
        (
            "empty_output_buffer",
            offset_of!(JpegDestinationMgr, empty_output_buffer),
        ),
        (
            "term_destination",
            offset_of!(JpegDestinationMgr, term_destination),
        ),
    ]
}

#[test]
fn rust_offsets_match_jpeg_destination_mgr_at_lib_version_80() {
    use libjpeg_turbo_rs_capi::jpeglib::JpegDestinationMgr;

    let rust_fields: Vec<(&'static str, usize)> = rust_offsets_destination_mgr();
    let names: Vec<&str> = rust_fields.iter().map(|(n, _)| *n).collect();
    let probe: CcProbeResult = match cc_offsetof_for_struct("jpeg_destination_mgr", &names) {
        CcProbeOutcome::Ok(r) => r,
        CcProbeOutcome::Skip(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    let rust_sizeof: usize = std::mem::size_of::<JpegDestinationMgr>();
    assert_no_drift("jpeg_destination_mgr", &rust_fields, rust_sizeof, &probe);
}

// ---------------------------------------------------------------------------
// `struct jpeg_source_mgr` cross-check (P3-1 deferred work).
// ---------------------------------------------------------------------------
//
// Mirrors `JpegSourceMgr` in `jpeglib.rs`. The classic streaming-decode
// pattern (`fill_input_buffer` returning `FALSE` for I/O suspension) reads
// `bytes_in_buffer` and `next_input_byte` at well-known offsets; the
// `resync_to_restart` callback at offset ~40 is what `jpeg_resync_to_restart`
// dispatches into. Layout drift here breaks every consumer that ships its
// own source manager.

fn rust_offsets_source_mgr() -> Vec<(&'static str, usize)> {
    use libjpeg_turbo_rs_capi::jpeglib::JpegSourceMgr;
    use std::mem::offset_of;

    vec![
        (
            "next_input_byte",
            offset_of!(JpegSourceMgr, next_input_byte),
        ),
        (
            "bytes_in_buffer",
            offset_of!(JpegSourceMgr, bytes_in_buffer),
        ),
        ("init_source", offset_of!(JpegSourceMgr, init_source)),
        (
            "fill_input_buffer",
            offset_of!(JpegSourceMgr, fill_input_buffer),
        ),
        (
            "skip_input_data",
            offset_of!(JpegSourceMgr, skip_input_data),
        ),
        (
            "resync_to_restart",
            offset_of!(JpegSourceMgr, resync_to_restart),
        ),
        ("term_source", offset_of!(JpegSourceMgr, term_source)),
    ]
}

#[test]
fn rust_offsets_match_jpeg_source_mgr_at_lib_version_80() {
    use libjpeg_turbo_rs_capi::jpeglib::JpegSourceMgr;

    let rust_fields: Vec<(&'static str, usize)> = rust_offsets_source_mgr();
    let names: Vec<&str> = rust_fields.iter().map(|(n, _)| *n).collect();
    let probe: CcProbeResult = match cc_offsetof_for_struct("jpeg_source_mgr", &names) {
        CcProbeOutcome::Ok(r) => r,
        CcProbeOutcome::Skip(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    let rust_sizeof: usize = std::mem::size_of::<JpegSourceMgr>();
    assert_no_drift("jpeg_source_mgr", &rust_fields, rust_sizeof, &probe);
}

// ---------------------------------------------------------------------------
// `struct jpeg_error_mgr` cross-check (P3-1 deferred work).
// ---------------------------------------------------------------------------
//
// Mirrors `JpegErrorMgr` in `jpeglib.rs`. The classic libjpeg consumer
// pattern is to override `error_exit` (offset 0) with a `setjmp`/`longjmp`
// handler and inspect `msg_code` (offset 40 LP64) plus `msg_parm` (offset 44)
// for diagnostics. Layout drift here turns recoverable errors into
// process-aborting bugs in any consumer that walks these fields by name.
//
// `msg_parm` is a `union { int i[8]; char s[80]; }` upstream; the Rust
// mirror reserves the larger arm (`[u8; 80]`) so `mem::size_of` matches
// the union's storage.

fn rust_offsets_error_mgr() -> Vec<(&'static str, usize)> {
    use libjpeg_turbo_rs_capi::jpeglib::JpegErrorMgr;
    use std::mem::offset_of;

    vec![
        ("error_exit", offset_of!(JpegErrorMgr, error_exit)),
        ("emit_message", offset_of!(JpegErrorMgr, emit_message)),
        ("output_message", offset_of!(JpegErrorMgr, output_message)),
        ("format_message", offset_of!(JpegErrorMgr, format_message)),
        ("reset_error_mgr", offset_of!(JpegErrorMgr, reset_error_mgr)),
        ("msg_code", offset_of!(JpegErrorMgr, msg_code)),
        ("msg_parm", offset_of!(JpegErrorMgr, msg_parm)),
        ("trace_level", offset_of!(JpegErrorMgr, trace_level)),
        ("num_warnings", offset_of!(JpegErrorMgr, num_warnings)),
        (
            "jpeg_message_table",
            offset_of!(JpegErrorMgr, jpeg_message_table),
        ),
        (
            "last_jpeg_message",
            offset_of!(JpegErrorMgr, last_jpeg_message),
        ),
        (
            "addon_message_table",
            offset_of!(JpegErrorMgr, addon_message_table),
        ),
        (
            "first_addon_message",
            offset_of!(JpegErrorMgr, first_addon_message),
        ),
        (
            "last_addon_message",
            offset_of!(JpegErrorMgr, last_addon_message),
        ),
    ]
}

#[test]
fn rust_offsets_match_jpeg_error_mgr_at_lib_version_80() {
    use libjpeg_turbo_rs_capi::jpeglib::JpegErrorMgr;

    let rust_fields: Vec<(&'static str, usize)> = rust_offsets_error_mgr();
    let names: Vec<&str> = rust_fields.iter().map(|(n, _)| *n).collect();
    let probe: CcProbeResult = match cc_offsetof_for_struct("jpeg_error_mgr", &names) {
        CcProbeOutcome::Ok(r) => r,
        CcProbeOutcome::Skip(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    let rust_sizeof: usize = std::mem::size_of::<JpegErrorMgr>();
    assert_no_drift("jpeg_error_mgr", &rust_fields, rust_sizeof, &probe);
}

// ---------------------------------------------------------------------------
// `struct jpeg_compress_struct` cross-check (P3-1 deferred work).
// ---------------------------------------------------------------------------
//
// Mirrors `JpegCompressPublic` in `jpeglib.rs`. This is the encoder-side
// counterpart to the `jpeg_decompress_struct` cross-check at the top of
// this file; cjpeg, GIMP's plugin, and a long tail of distro consumers
// read fields like `image_width` / `data_precision` / `comp_info` /
// `optimize_coding` directly through this struct, so any field-order or
// `JPEG_LIB_VERSION ≥ 70`/`≥ 80` drift surfaces as silently corrupted
// encode parameters rather than a clean error.
//
// Field-name mapping note. The Rust mirror calls the
// `struct jpeg_c_main_controller *` slot `main_ctrl` (the C-side name
// `main` collides with Rust's `main` identifier in some scopes); the
// cross-check maps the upstream `main` field to the Rust `main_ctrl` mirror
// via the `(c_field_name, rust_offset)` tuple — same pattern the existing
// `cc_offsetof_for_struct` helper uses, so the harness can keep reading
// `offsetof(struct jpeg_compress_struct, main)` against the upstream
// header while comparing to `offset_of!(JpegCompressPublic, main_ctrl)`.

#[allow(non_snake_case)]
fn rust_offsets_compress() -> Vec<(&'static str, usize)> {
    use libjpeg_turbo_rs_capi::jpeglib::JpegCompressPublic;
    use std::mem::offset_of;

    vec![
        // jpeg_common_fields prefix (offset 0..40 LP64).
        ("err", offset_of!(JpegCompressPublic, err)),
        ("mem", offset_of!(JpegCompressPublic, mem)),
        ("progress", offset_of!(JpegCompressPublic, progress)),
        ("client_data", offset_of!(JpegCompressPublic, client_data)),
        (
            "is_decompressor",
            offset_of!(JpegCompressPublic, is_decompressor),
        ),
        ("global_state", offset_of!(JpegCompressPublic, global_state)),
        // Compressor-specific.
        ("dest", offset_of!(JpegCompressPublic, dest)),
        ("image_width", offset_of!(JpegCompressPublic, image_width)),
        ("image_height", offset_of!(JpegCompressPublic, image_height)),
        (
            "input_components",
            offset_of!(JpegCompressPublic, input_components),
        ),
        (
            "in_color_space",
            offset_of!(JpegCompressPublic, in_color_space),
        ),
        ("input_gamma", offset_of!(JpegCompressPublic, input_gamma)),
        // JPEG_LIB_VERSION ≥ 70.
        ("scale_num", offset_of!(JpegCompressPublic, scale_num)),
        ("scale_denom", offset_of!(JpegCompressPublic, scale_denom)),
        ("jpeg_width", offset_of!(JpegCompressPublic, jpeg_width)),
        ("jpeg_height", offset_of!(JpegCompressPublic, jpeg_height)),
        // Primary compression parameters.
        (
            "data_precision",
            offset_of!(JpegCompressPublic, data_precision),
        ),
        (
            "num_components",
            offset_of!(JpegCompressPublic, num_components),
        ),
        (
            "jpeg_color_space",
            offset_of!(JpegCompressPublic, jpeg_color_space),
        ),
        ("comp_info", offset_of!(JpegCompressPublic, comp_info)),
        // Quantization / Huffman tables.
        (
            "quant_tbl_ptrs",
            offset_of!(JpegCompressPublic, quant_tbl_ptrs),
        ),
        (
            "q_scale_factor",
            offset_of!(JpegCompressPublic, q_scale_factor),
        ),
        (
            "dc_huff_tbl_ptrs",
            offset_of!(JpegCompressPublic, dc_huff_tbl_ptrs),
        ),
        (
            "ac_huff_tbl_ptrs",
            offset_of!(JpegCompressPublic, ac_huff_tbl_ptrs),
        ),
        // Arithmetic-coding tables.
        ("arith_dc_L", offset_of!(JpegCompressPublic, arith_dc_L)),
        ("arith_dc_U", offset_of!(JpegCompressPublic, arith_dc_U)),
        ("arith_ac_K", offset_of!(JpegCompressPublic, arith_ac_K)),
        // Scan scripting.
        ("num_scans", offset_of!(JpegCompressPublic, num_scans)),
        ("scan_info", offset_of!(JpegCompressPublic, scan_info)),
        // Boolean compression options.
        ("raw_data_in", offset_of!(JpegCompressPublic, raw_data_in)),
        ("arith_code", offset_of!(JpegCompressPublic, arith_code)),
        (
            "optimize_coding",
            offset_of!(JpegCompressPublic, optimize_coding),
        ),
        (
            "CCIR601_sampling",
            offset_of!(JpegCompressPublic, CCIR601_sampling),
        ),
        (
            "do_fancy_downsampling",
            offset_of!(JpegCompressPublic, do_fancy_downsampling),
        ),
        (
            "smoothing_factor",
            offset_of!(JpegCompressPublic, smoothing_factor),
        ),
        ("dct_method", offset_of!(JpegCompressPublic, dct_method)),
        // Restart marker control.
        (
            "restart_interval",
            offset_of!(JpegCompressPublic, restart_interval),
        ),
        (
            "restart_in_rows",
            offset_of!(JpegCompressPublic, restart_in_rows),
        ),
        // JFIF / Adobe marker emission parameters.
        (
            "write_JFIF_header",
            offset_of!(JpegCompressPublic, write_JFIF_header),
        ),
        (
            "JFIF_major_version",
            offset_of!(JpegCompressPublic, JFIF_major_version),
        ),
        (
            "JFIF_minor_version",
            offset_of!(JpegCompressPublic, JFIF_minor_version),
        ),
        ("density_unit", offset_of!(JpegCompressPublic, density_unit)),
        ("X_density", offset_of!(JpegCompressPublic, X_density)),
        ("Y_density", offset_of!(JpegCompressPublic, Y_density)),
        (
            "write_Adobe_marker",
            offset_of!(JpegCompressPublic, write_Adobe_marker),
        ),
        // State variable.
        (
            "next_scanline",
            offset_of!(JpegCompressPublic, next_scanline),
        ),
        // Computed at startup.
        (
            "progressive_mode",
            offset_of!(JpegCompressPublic, progressive_mode),
        ),
        (
            "max_h_samp_factor",
            offset_of!(JpegCompressPublic, max_h_samp_factor),
        ),
        (
            "max_v_samp_factor",
            offset_of!(JpegCompressPublic, max_v_samp_factor),
        ),
        (
            "min_DCT_h_scaled_size",
            offset_of!(JpegCompressPublic, min_DCT_h_scaled_size),
        ),
        (
            "min_DCT_v_scaled_size",
            offset_of!(JpegCompressPublic, min_DCT_v_scaled_size),
        ),
        (
            "total_iMCU_rows",
            offset_of!(JpegCompressPublic, total_iMCU_rows),
        ),
        // Per-scan state.
        (
            "comps_in_scan",
            offset_of!(JpegCompressPublic, comps_in_scan),
        ),
        (
            "cur_comp_info",
            offset_of!(JpegCompressPublic, cur_comp_info),
        ),
        ("MCUs_per_row", offset_of!(JpegCompressPublic, MCUs_per_row)),
        (
            "MCU_rows_in_scan",
            offset_of!(JpegCompressPublic, MCU_rows_in_scan),
        ),
        (
            "blocks_in_MCU",
            offset_of!(JpegCompressPublic, blocks_in_MCU),
        ),
        (
            "MCU_membership",
            offset_of!(JpegCompressPublic, MCU_membership),
        ),
        ("Ss", offset_of!(JpegCompressPublic, Ss)),
        ("Se", offset_of!(JpegCompressPublic, Se)),
        ("Ah", offset_of!(JpegCompressPublic, Ah)),
        ("Al", offset_of!(JpegCompressPublic, Al)),
        // JPEG_LIB_VERSION ≥ 80 extensions.
        ("block_size", offset_of!(JpegCompressPublic, block_size)),
        (
            "natural_order",
            offset_of!(JpegCompressPublic, natural_order),
        ),
        ("lim_Se", offset_of!(JpegCompressPublic, lim_Se)),
        // Opaque libjpeg-internal pointers. The C-side `main` field is
        // mirrored by Rust as `main_ctrl` (the trailing `_ctrl` keeps
        // grep'ability and avoids identifier collisions in some scopes
        // — see the file-level note above).
        ("master", offset_of!(JpegCompressPublic, master)),
        ("main", offset_of!(JpegCompressPublic, main_ctrl)),
        ("prep", offset_of!(JpegCompressPublic, prep)),
        ("coef", offset_of!(JpegCompressPublic, coef)),
        ("marker", offset_of!(JpegCompressPublic, marker)),
        ("cconvert", offset_of!(JpegCompressPublic, cconvert)),
        ("downsample", offset_of!(JpegCompressPublic, downsample)),
        ("fdct", offset_of!(JpegCompressPublic, fdct)),
        ("entropy", offset_of!(JpegCompressPublic, entropy)),
        ("script_space", offset_of!(JpegCompressPublic, script_space)),
        (
            "script_space_size",
            offset_of!(JpegCompressPublic, script_space_size),
        ),
    ]
}

#[test]
fn rust_offsets_match_jpeg_compress_struct_at_lib_version_80() {
    use libjpeg_turbo_rs_capi::jpeglib::JpegCompressPublic;

    let rust_fields: Vec<(&'static str, usize)> = rust_offsets_compress();
    let names: Vec<&str> = rust_fields.iter().map(|(n, _)| *n).collect();
    let probe: CcProbeResult = match cc_offsetof_for_struct("jpeg_compress_struct", &names) {
        CcProbeOutcome::Ok(r) => r,
        CcProbeOutcome::Skip(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    let rust_sizeof: usize = std::mem::size_of::<JpegCompressPublic>();
    assert_no_drift("jpeg_compress_struct", &rust_fields, rust_sizeof, &probe);
}
