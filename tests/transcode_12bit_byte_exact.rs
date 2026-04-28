//! Byte-exact 12-bit JPEG transcode against upstream `jpegtran`.
//!
//! Closes LAST_MILE.md "Suggested Order" item 5b. The transform
//! `transform_jpeg_with_options(monkey12.jpg, -rotate 90, -copy all)` —
//! and the analogous flip / rotate ops — must produce output that is
//! byte-identical to upstream `jpegtran -copy all <op>`. The 12-bit
//! `monkey12.jpg` fixture has an APP2 ICC profile chunk; preserving
//! that chunk verbatim through the C-API path ensures the FFI shim
//! is faithful to stock libjpeg-turbo's `transupp::jcopy_markers_execute`
//! marker forwarding.

mod helpers;

use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, MarkerCopyMode, TransformOp, TransformOptions,
};

/// Apply `jpegtran -copy all <op>` via upstream and via our Rust
/// `transform_jpeg_with_options`, and assert the byte streams match.
fn assert_byte_exact_transcode(fixture: &str, op: TransformOp, jpegtran_arg: &[&str]) {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let src_path = PathBuf::from(format!("references/libjpeg-turbo/testimages/{fixture}"));
    let src: Vec<u8> = std::fs::read(&src_path).unwrap_or_else(|e| {
        panic!("read {}: {e}", src_path.display());
    });

    let tmp_dir = std::env::temp_dir();
    let op_tag: String = jpegtran_arg.join("_").replace([' ', '/'], "_");
    let tmp_out = tmp_dir.join(format!(
        "ljt_12bit_xform_{}_{}_{}.jpg",
        std::process::id(),
        op_tag,
        fixture
    ));
    let mut cmd = Command::new(&jpegtran);
    cmd.arg("-copy").arg("all");
    for a in jpegtran_arg {
        cmd.arg(a);
    }
    cmd.arg("-outfile").arg(&tmp_out).arg(&src_path);
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("spawn upstream jpegtran: {e}"));
    assert!(status.success(), "upstream jpegtran failed for {fixture}");
    let upstream: Vec<u8> = std::fs::read(&tmp_out).expect("read upstream output");
    let _ = std::fs::remove_file(&tmp_out);

    let opts = TransformOptions {
        op,
        copy_markers: MarkerCopyMode::All,
        ..TransformOptions::default()
    };
    let ours: Vec<u8> = transform_jpeg_with_options(&src, &opts)
        .unwrap_or_else(|e| panic!("Rust transform of {fixture}: {e}"));

    if ours != upstream {
        let first_diff = ours
            .iter()
            .zip(upstream.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(ours.len().min(upstream.len()));
        panic!(
            "{fixture}: byte mismatch (ours={} upstream={}) at offset {first_diff}: \
             ours[..]={:02X?}, upstream[..]={:02X?}",
            ours.len(),
            upstream.len(),
            &ours[first_diff..(first_diff + 16).min(ours.len())],
            &upstream[first_diff..(first_diff + 16).min(upstream.len())]
        );
    }
}

#[test]
fn monkey12_rotate90_byte_exact_against_upstream_jpegtran() {
    assert_byte_exact_transcode("monkey12.jpg", TransformOp::Rot90, &["-rotate", "90"]);
}

#[test]
fn monkey12_rotate180_byte_exact_against_upstream_jpegtran() {
    assert_byte_exact_transcode("monkey12.jpg", TransformOp::Rot180, &["-rotate", "180"]);
}

#[test]
fn monkey12_flip_h_byte_exact_against_upstream_jpegtran() {
    assert_byte_exact_transcode("monkey12.jpg", TransformOp::HFlip, &["-flip", "horizontal"]);
}

#[test]
fn monkey12_transpose_byte_exact_against_upstream_jpegtran() {
    assert_byte_exact_transcode("monkey12.jpg", TransformOp::Transpose, &["-transpose"]);
}
