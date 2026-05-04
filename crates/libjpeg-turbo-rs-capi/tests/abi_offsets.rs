//! P2-4: generated C-side ABI cross-check.
//!
//! For every field that `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
//! const-asserts an offset for, this test compiles a tiny C program
//! against the real upstream `jpeglib.h` (from the
//! `references/libjpeg-turbo/` submodule) at `JPEG_LIB_VERSION = 80`,
//! prints `offsetof(struct jpeg_decompress_struct, FIELD)` for that
//! field, then asserts the value equals what Rust would compute via
//! `std::mem::offset_of!(JpegDecompressPublic, FIELD)`.
//!
//! Why this exists. The hand-typed offsets in `jpeglib.rs:4096+` were
//! computed by a one-time `offsetof` print and pasted in as constants.
//! If a future upstream `jpeglib.h` shuffles a field — or if our Rust
//! mirror grows a misordered field — neither side notices on its own.
//! This test catches the divergence at `cargo test` time.
//!
//! Skip-with-reason cases:
//! - No `cc` on PATH.
//! - Cross-compile target where host `cc` cannot match the target ABI.
//! - Non-LP64 host (matches the `cfg(target_pointer_width = "64",
//!   not(windows))` gate on the Rust assertion block; the offsets
//!   change on ILP32 / Windows LLP64).

use std::path::PathBuf;
use std::process::Command;

/// Field name → expected Rust offset (must equal `offset_of!`).
fn rust_offsets() -> Vec<(&'static str, usize)> {
    // SAFETY-equivalent: pure `const` reads, no allocation.
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
    // Gate: 64-bit hosts only. The Rust assertion block in
    // `jpeglib.rs` only pins LP64 offsets, but `offset_of!` reads the
    // actual struct layout on the running host — so this test probes
    // whatever ABI the target uses (LP64 on Linux/macOS, LLP64 on
    // Windows MSVC). Any per-platform divergence between our Rust
    // mirror and upstream `jpeglib.h` shows up here as a per-platform
    // mismatch.
    //
    // On 32-bit hosts the struct shrinks proportionally and the
    // hand-typed offsets in jpeglib.rs do not apply; skip with a
    // descriptive reason rather than fail.
    if std::mem::size_of::<usize>() != 8 {
        eprintln!(
            "SKIP: ABI cross-check only runs on 64-bit hosts; \
             host has size_of(usize)={}",
            std::mem::size_of::<usize>(),
        );
        return;
    }

    // Locate the submodule headers.
    let workspace_root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let upstream_src: PathBuf = workspace_root.join("references/libjpeg-turbo/src");
    let upstream_jpeglib_h: PathBuf = upstream_src.join("jpeglib.h");
    if !upstream_jpeglib_h.exists() {
        eprintln!(
            "SKIP: upstream jpeglib.h not found at {:?} (submodule not initialized?)",
            upstream_jpeglib_h
        );
        return;
    }

    // Locate a C compiler.
    let cc: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let cc_check: bool = Command::new(&cc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !cc_check {
        eprintln!("SKIP: C compiler `{}` not found or not runnable", cc);
        return;
    }

    // Build the field list and the C harness.
    let fields: Vec<(&'static str, usize)> = rust_offsets();

    let mut c_src: String = String::new();
    c_src.push_str("#include <stdio.h>\n");
    c_src.push_str("#include <stddef.h>\n");
    c_src.push_str("#include <jpeglib.h>\n");
    c_src.push_str("int main(void) {\n");
    for (name, _) in &fields {
        c_src.push_str(&format!(
            "  printf(\"{name}=%zu\\n\", offsetof(struct jpeg_decompress_struct, {name}));\n"
        ));
    }
    c_src.push_str("  return 0;\n}\n");

    // Build a minimal `jconfig.h` with the v8 defaults the upstream
    // CMakeLists.txt uses (jconfig.h.in @-substitutions that matter for
    // compilation: JPEG_LIB_VERSION + LIBJPEG_TURBO_VERSION_NUMBER +
    // MEM_SRCDST + arithmetic).
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

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let jconfig_h_path: PathBuf = tmp.path().join("jconfig.h");
    let c_src_path: PathBuf = tmp.path().join("abi_offsets.c");
    let bin_path: PathBuf = tmp.path().join("abi_offsets");

    std::fs::write(&jconfig_h_path, jconfig_h).expect("write jconfig.h");
    std::fs::write(&c_src_path, &c_src).expect("write abi_offsets.c");

    // Compile. `-I<tmp>` for our jconfig.h, `-I<upstream_src>` for jpeglib.h.
    let compile_status = Command::new(&cc)
        .args(["-O0", "-Wno-implicit-function-declaration", "-I"])
        .arg(tmp.path())
        .arg("-I")
        .arg(&upstream_src)
        .arg("-o")
        .arg(&bin_path)
        .arg(&c_src_path)
        .output()
        .expect("invoke cc");
    if !compile_status.status.success() {
        let stderr: String = String::from_utf8_lossy(&compile_status.stderr).to_string();
        // If compilation failed for an environmental reason (missing
        // headers, broken cross-compile setup), skip-with-reason rather
        // than panic — the test exists to gate against drift, not to
        // double as a compile-toolchain test.
        if stderr.contains("No such file or directory")
            || stderr.contains("cannot find")
            || stderr.contains("not found")
        {
            eprintln!(
                "SKIP: cc could not compile the harness (env issue):\n{}",
                stderr
            );
            return;
        }
        panic!(
            "cc failed to compile abi_offsets.c:\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&compile_status.stdout),
            stderr
        );
    }

    // Run, capture stdout.
    let run = Command::new(&bin_path).output().expect("run abi_offsets");
    assert!(
        run.status.success(),
        "abi_offsets exited non-zero: {:?}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout: String = String::from_utf8(run.stdout).expect("utf8 stdout");

    // Parse `name=offset` lines into a map.
    let mut c_offsets: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for line in stdout.lines() {
        if let Some((k, v)) = line.split_once('=') {
            if let Ok(off) = v.trim().parse::<usize>() {
                c_offsets.insert(k.to_string(), off);
            }
        }
    }

    let mut mismatches: Vec<String> = Vec::new();
    for (field, rust_off) in &fields {
        match c_offsets.get(*field) {
            None => mismatches.push(format!("field `{field}`: missing from C output")),
            Some(&c_off) if c_off != *rust_off => {
                mismatches.push(format!(
                    "field `{field}`: Rust says offset {rust_off}, C says {c_off}"
                ));
            }
            _ => {}
        }
    }

    assert!(
        mismatches.is_empty(),
        "ABI offset divergence between Rust mirror and upstream jpeglib.h \
         (JPEG_LIB_VERSION=80):\n  {}\nRaw C output:\n{}",
        mismatches.join("\n  "),
        stdout
    );
}
