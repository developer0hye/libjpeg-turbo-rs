use libjpeg_turbo_rs::{decompress, decompress_cropped, CropRegion};

#[test]
fn crop_full_image_matches_decompress() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let region = CropRegion {
        x: 0,
        y: 0,
        width: 320,
        height: 240,
    };
    let cropped = decompress_cropped(data, region).unwrap();
    assert_eq!(full.data, cropped.data);
}

#[test]
fn crop_center_region() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion {
        x: 80,
        y: 60,
        width: 160,
        height: 120,
    };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn crop_clamps_to_image_bounds() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion {
        x: 300,
        y: 200,
        width: 100,
        height: 100,
    };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 20);
    assert_eq!(img.height, 40);
}

#[test]
fn crop_top_left_corner_matches_full_decode() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let region = CropRegion {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    let cropped = decompress_cropped(data, region).unwrap();
    let bpp = full.pixel_format.bytes_per_pixel();
    // First row of cropped should match first 64 pixels of full
    for x in 0..64 {
        for c in 0..bpp {
            assert_eq!(cropped.data[x * bpp + c], full.data[x * bpp + c]);
        }
    }
}

#[test]
fn crop_zero_size_returns_empty() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 0);
    assert_eq!(img.height, 0);
    assert_eq!(img.data.len(), 0);
}

#[test]
fn crop_region_pixel_values_match_full_decode() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let bpp = full.pixel_format.bytes_per_pixel();
    let region = CropRegion {
        x: 50,
        y: 30,
        width: 100,
        height: 80,
    };
    let cropped = decompress_cropped(data, region).unwrap();

    // Every pixel in cropped should match the corresponding pixel in full
    for row in 0..80 {
        for col in 0..100 {
            let crop_idx = (row * 100 + col) * bpp;
            let full_idx = ((30 + row) * 320 + (50 + col)) * bpp;
            for c in 0..bpp {
                assert_eq!(
                    cropped.data[crop_idx + c],
                    full.data[full_idx + c],
                    "Mismatch at row={row}, col={col}, channel={c}"
                );
            }
        }
    }
}
