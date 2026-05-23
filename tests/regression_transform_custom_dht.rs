//! Regression test for fuzz_transform_diff_c crash b4de9a0d.
//!
//! A baseline JPEG with custom DHT tables defining AC categories > 10
//! and DC categories > 11 was re-encoded with standard Annex K tables
//! that lack codes for those categories, producing corrupt output that
//! djpeg rejected. Fixed by always routing through the optimized
//! Huffman writer when out-of-range symbols are present, regardless
//! of whether the source was progressive.

#![cfg(not(target_family = "wasm"))]

use libjpeg_turbo_rs::{
    decompress, transform_jpeg_with_options, MarkerCopyMode, TransformOp, TransformOptions,
};

#[test]
fn transform_baseline_custom_dht_produces_valid_jpeg() {
    let crash_path = format!(
        "{}/fuzz/artifacts/fuzz_transform_diff_c/crash-b4de9a0d01a69a9aac6c34dd3351e69a8a2bf1c2",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = match std::fs::read(&crash_path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: crash artifact not found at {crash_path}");
            return;
        }
    };

    let jpeg = &data[1..]; // byte 0 is the op selector

    for op in [TransformOp::HFlip, TransformOp::VFlip, TransformOp::Rot180] {
        let opts = TransformOptions {
            op,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        };
        let transformed = transform_jpeg_with_options(jpeg, &opts)
            .unwrap_or_else(|e| panic!("transform {op:?} failed: {e}"));

        // The transformed output must be a decodable JPEG.
        decompress(&transformed)
            .unwrap_or_else(|e| panic!("transform {op:?} output not decodable: {e}"));
    }
}
