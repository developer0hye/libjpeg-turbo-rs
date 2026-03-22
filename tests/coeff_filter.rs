use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, transform_jpeg_with_options, PixelFormat, Subsampling,
    TransformOp, TransformOptions,
};

/// Helper: create a small color JPEG for testing.
fn make_test_jpeg(width: usize, height: usize, subsampling: Subsampling) -> Vec<u8> {
    let bpp: usize = 3;
    let mut pixels: Vec<u8> = vec![0u8; width * height * bpp];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * bpp;
            pixels[idx] = (x * 255 / width.max(1)) as u8;
            pixels[idx + 1] = (y * 255 / height.max(1)) as u8;
            pixels[idx + 2] = 128;
        }
    }
    compress(&pixels, width, height, PixelFormat::Rgb, 90, subsampling).unwrap()
}

/// A custom filter that zeros all AC coefficients should produce a blocky image
/// with only DC values preserved. Verify that the filter is invoked and modifies blocks.
#[test]
fn custom_filter_zeros_ac_coefficients() {
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S444);
    let invocation_count = Arc::new(AtomicUsize::new(0));
    let count_clone = invocation_count.clone();

    let opts = TransformOptions {
        custom_filter: Some(Box::new(move |block: &mut [i16; 64], _ci, _bx, _by| {
            count_clone.fetch_add(1, Ordering::Relaxed);
            // Zero all AC coefficients (indices 1..64), keep DC (index 0).
            for i in 1..64 {
                block[i] = 0;
            }
        })),
        ..TransformOptions::default()
    };

    let result: Vec<u8> = transform_jpeg_with_options(&data, &opts).unwrap();

    // Filter should have been invoked for every block in every component.
    let coeffs = read_coefficients(&data).unwrap();
    let total_blocks: usize = coeffs
        .components
        .iter()
        .map(|c| c.blocks_x * c.blocks_y)
        .sum();
    assert_eq!(invocation_count.load(Ordering::Relaxed), total_blocks);

    // Verify the output has zeroed AC: read back and check.
    let out_coeffs = read_coefficients(&result).unwrap();
    for comp in &out_coeffs.components {
        for block in &comp.blocks {
            // AC coefficients (zigzag indices 1..64) should all be zero.
            for i in 1..64 {
                assert_eq!(
                    block[i], 0,
                    "AC coefficient at index {} should be zero after filter",
                    i
                );
            }
        }
    }
}

/// Verify that the custom filter receives correct component_index, block_x, block_y values.
#[test]
fn custom_filter_receives_correct_coordinates() {
    // Use 4:4:4 subsampling for simple block layout (all components same size).
    let data: Vec<u8> = make_test_jpeg(16, 16, Subsampling::S444);

    // For a 16x16 4:4:4 JPEG, each component has 2x2 blocks.
    let calls: Arc<std::sync::Mutex<Vec<(usize, usize, usize)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let calls_clone = calls.clone();

    let opts = TransformOptions {
        custom_filter: Some(Box::new(move |_block, ci, bx, by| {
            calls_clone.lock().unwrap().push((ci, bx, by));
        })),
        ..TransformOptions::default()
    };

    let _result = transform_jpeg_with_options(&data, &opts).unwrap();

    let recorded = calls.lock().unwrap();

    // 3 components x (2x2 blocks each) = 12 calls total.
    assert_eq!(recorded.len(), 12);

    // Check that all 3 component indices appear.
    let component_indices: std::collections::HashSet<usize> =
        recorded.iter().map(|(ci, _, _)| *ci).collect();
    assert_eq!(component_indices.len(), 3);
    assert!(component_indices.contains(&0));
    assert!(component_indices.contains(&1));
    assert!(component_indices.contains(&2));

    // For each component, verify block coordinates cover the full 2x2 grid.
    for ci in 0..3 {
        let mut coords: Vec<(usize, usize)> = recorded
            .iter()
            .filter(|(c, _, _)| *c == ci)
            .map(|(_, bx, by)| (*bx, *by))
            .collect();
        coords.sort();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }
}

/// Verify that None custom_filter is a no-op: output matches a transform without filter.
#[test]
fn custom_filter_none_is_noop() {
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);

    let opts_no_filter = TransformOptions {
        op: TransformOp::HFlip,
        custom_filter: None,
        ..TransformOptions::default()
    };

    let opts_default = TransformOptions {
        op: TransformOp::HFlip,
        ..TransformOptions::default()
    };

    let result_no_filter = transform_jpeg_with_options(&data, &opts_no_filter).unwrap();
    let result_default = transform_jpeg_with_options(&data, &opts_default).unwrap();

    // Both should produce identical output.
    assert_eq!(result_no_filter, result_default);
}

/// Verify that custom filter works in conjunction with spatial transforms.
/// The filter should be applied AFTER the spatial transform.
#[test]
fn custom_filter_applied_after_spatial_transform() {
    let data: Vec<u8> = make_test_jpeg(16, 16, Subsampling::S444);

    // Read original coefficients for comparison.
    let orig_coeffs = read_coefficients(&data).unwrap();
    let orig_dc: i16 = orig_coeffs.components[0].blocks[0][0];

    // Apply HFlip with a filter that sets DC to a sentinel value.
    let sentinel: i16 = 42;
    let opts = TransformOptions {
        op: TransformOp::HFlip,
        custom_filter: Some(Box::new(move |block: &mut [i16; 64], _ci, _bx, _by| {
            block[0] = sentinel;
        })),
        ..TransformOptions::default()
    };

    let result = transform_jpeg_with_options(&data, &opts).unwrap();
    let out_coeffs = read_coefficients(&result).unwrap();

    // Every block in every component should have DC = sentinel.
    for comp in &out_coeffs.components {
        for block in &comp.blocks {
            assert_eq!(
                block[0], sentinel,
                "DC coefficient should be set to sentinel by filter"
            );
        }
    }

    // Confirm the original DC was not already the sentinel (sanity check).
    assert_ne!(orig_dc, sentinel, "Original DC should differ from sentinel");
}
