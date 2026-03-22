use libjpeg_turbo_rs::{
    calc_jpeg_dimensions, calc_output_dimensions, compress, copy_critical_parameters,
    extract_jfif_thumbnail, jpeg_buf_size, read_coefficients, transform_buf_size, PixelFormat,
    Subsampling, TransformOp,
};

// --- transform_buf_size ---

#[test]
fn transform_buf_size_none_matches_jpeg_buf_size() {
    let size: usize = transform_buf_size(640, 480, Subsampling::S420, TransformOp::None);
    let expected: usize = jpeg_buf_size(640, 480, Subsampling::S420);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_hflip_preserves_dimensions() {
    let size: usize = transform_buf_size(640, 480, Subsampling::S420, TransformOp::HFlip);
    let expected: usize = jpeg_buf_size(640, 480, Subsampling::S420);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_rot90_swaps_dimensions() {
    let size: usize = transform_buf_size(640, 480, Subsampling::S420, TransformOp::Rot90);
    let expected: usize = jpeg_buf_size(480, 640, Subsampling::S420);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_rot270_swaps_dimensions() {
    let size: usize = transform_buf_size(640, 480, Subsampling::S420, TransformOp::Rot270);
    let expected: usize = jpeg_buf_size(480, 640, Subsampling::S420);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_transpose_swaps_dimensions() {
    let size: usize = transform_buf_size(1920, 1080, Subsampling::S422, TransformOp::Transpose);
    let expected: usize = jpeg_buf_size(1080, 1920, Subsampling::S422);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_transverse_swaps_dimensions() {
    let size: usize = transform_buf_size(800, 600, Subsampling::S444, TransformOp::Transverse);
    let expected: usize = jpeg_buf_size(600, 800, Subsampling::S444);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_rot180_preserves_dimensions() {
    let size: usize = transform_buf_size(640, 480, Subsampling::S420, TransformOp::Rot180);
    let expected: usize = jpeg_buf_size(640, 480, Subsampling::S420);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_vflip_preserves_dimensions() {
    let size: usize = transform_buf_size(640, 480, Subsampling::S420, TransformOp::VFlip);
    let expected: usize = jpeg_buf_size(640, 480, Subsampling::S420);
    assert_eq!(size, expected);
}

#[test]
fn transform_buf_size_asymmetric_shows_difference() {
    // S411 MCU=32x8: pad(100,32)=128, pad(50,8)=56 vs pad(50,32)=64, pad(100,8)=104
    let no_swap: usize = transform_buf_size(100, 50, Subsampling::S411, TransformOp::None);
    let swapped: usize = transform_buf_size(100, 50, Subsampling::S411, TransformOp::Rot90);
    assert_ne!(no_swap, swapped);
}

// --- calc_output_dimensions ---

#[test]
fn calc_output_dimensions_identity() {
    let (w, h) = calc_output_dimensions(640, 480, 1, 1);
    assert_eq!((w, h), (640, 480));
}

#[test]
fn calc_output_dimensions_half() {
    let (w, h) = calc_output_dimensions(640, 480, 1, 2);
    assert_eq!((w, h), (320, 240));
}

#[test]
fn calc_output_dimensions_quarter() {
    let (w, h) = calc_output_dimensions(640, 480, 1, 4);
    assert_eq!((w, h), (160, 120));
}

#[test]
fn calc_output_dimensions_eighth() {
    let (w, h) = calc_output_dimensions(640, 480, 1, 8);
    assert_eq!((w, h), (80, 60));
}

#[test]
fn calc_output_dimensions_rounds_up() {
    let (w, h) = calc_output_dimensions(641, 481, 1, 2);
    assert_eq!((w, h), (321, 241));
}

#[test]
fn calc_output_dimensions_scale_up() {
    let (w, h) = calc_output_dimensions(640, 480, 3, 2);
    assert_eq!((w, h), (960, 720));
}

// --- calc_jpeg_dimensions ---

#[test]
fn calc_jpeg_dimensions_s444_no_padding() {
    let (w, h) = calc_jpeg_dimensions(640, 480, Subsampling::S444);
    assert_eq!((w, h), (640, 480));
}

#[test]
fn calc_jpeg_dimensions_s420_no_padding() {
    let (w, h) = calc_jpeg_dimensions(640, 480, Subsampling::S420);
    assert_eq!((w, h), (640, 480));
}

#[test]
fn calc_jpeg_dimensions_s420_pads_to_mcu() {
    let (w, h) = calc_jpeg_dimensions(641, 481, Subsampling::S420);
    assert_eq!((w, h), (656, 496));
}

#[test]
fn calc_jpeg_dimensions_s411_pads_to_mcu() {
    let (w, h) = calc_jpeg_dimensions(641, 481, Subsampling::S411);
    assert_eq!((w, h), (672, 488));
}

// --- extract_jfif_thumbnail ---

#[test]
fn extract_jfif_thumbnail_absent() {
    let pixels: Vec<u8> = vec![255u8; 64 * 64 * 3];
    let jpeg_data: Vec<u8> = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420)
        .expect("compress failed");
    let thumb: Option<Vec<u8>> = extract_jfif_thumbnail(&jpeg_data);
    assert!(thumb.is_none());
}

#[test]
fn extract_jfif_thumbnail_with_embedded_thumb() {
    let thumb_w: u8 = 2;
    let thumb_h: u8 = 2;
    let thumb_pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];

    let mut app0_data: Vec<u8> = Vec::new();
    app0_data.extend_from_slice(b"JFIF\0");
    app0_data.push(1);
    app0_data.push(2);
    app0_data.push(1);
    app0_data.extend_from_slice(&72u16.to_be_bytes());
    app0_data.extend_from_slice(&72u16.to_be_bytes());
    app0_data.push(thumb_w);
    app0_data.push(thumb_h);
    app0_data.extend_from_slice(&thumb_pixels);

    let app0_length: u16 = (app0_data.len() + 2) as u16;

    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let base_jpeg: Vec<u8> =
        compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S420).expect("compress failed");

    let mut new_jpeg: Vec<u8> = Vec::new();
    new_jpeg.push(0xFF);
    new_jpeg.push(0xD8);
    new_jpeg.push(0xFF);
    new_jpeg.push(0xE0);
    new_jpeg.extend_from_slice(&app0_length.to_be_bytes());
    new_jpeg.extend_from_slice(&app0_data);

    let mut pos: usize = 2;
    if base_jpeg.len() > 4 && base_jpeg[2] == 0xFF && base_jpeg[3] == 0xE0 {
        let orig_len: usize = u16::from_be_bytes([base_jpeg[4], base_jpeg[5]]) as usize + 2;
        pos = 2 + 2 + orig_len;
    }
    new_jpeg.extend_from_slice(&base_jpeg[pos..]);

    let thumb: Option<Vec<u8>> = extract_jfif_thumbnail(&new_jpeg);
    assert!(thumb.is_some(), "thumbnail should be found");
    assert_eq!(thumb.unwrap(), thumb_pixels);
}

#[test]
fn extract_jfif_thumbnail_zero_size() {
    let mut app0_data: Vec<u8> = Vec::new();
    app0_data.extend_from_slice(b"JFIF\0");
    app0_data.push(1);
    app0_data.push(2);
    app0_data.push(1);
    app0_data.extend_from_slice(&72u16.to_be_bytes());
    app0_data.extend_from_slice(&72u16.to_be_bytes());
    app0_data.push(0);
    app0_data.push(0);

    let app0_length: u16 = (app0_data.len() + 2) as u16;

    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let base_jpeg: Vec<u8> =
        compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S420).expect("compress failed");

    let mut new_jpeg: Vec<u8> = Vec::new();
    new_jpeg.push(0xFF);
    new_jpeg.push(0xD8);
    new_jpeg.push(0xFF);
    new_jpeg.push(0xE0);
    new_jpeg.extend_from_slice(&app0_length.to_be_bytes());
    new_jpeg.extend_from_slice(&app0_data);

    let mut pos: usize = 2;
    if base_jpeg.len() > 4 && base_jpeg[2] == 0xFF && base_jpeg[3] == 0xE0 {
        let orig_len: usize = u16::from_be_bytes([base_jpeg[4], base_jpeg[5]]) as usize + 2;
        pos = 2 + 2 + orig_len;
    }
    new_jpeg.extend_from_slice(&base_jpeg[pos..]);

    let thumb: Option<Vec<u8>> = extract_jfif_thumbnail(&new_jpeg);
    assert!(thumb.is_none());
}

// --- copy_critical_parameters ---

#[test]
fn copy_critical_parameters_preserves_quant_tables() {
    let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];
    let jpeg_data: Vec<u8> = compress(&pixels, 64, 64, PixelFormat::Rgb, 90, Subsampling::S420)
        .expect("compress failed");
    let coeffs = read_coefficients(&jpeg_data).expect("read_coefficients failed");

    let config = copy_critical_parameters(&coeffs);

    assert_eq!(config.quant_tables.len(), coeffs.quant_tables.len());
    for (i, table) in config.quant_tables.iter().enumerate() {
        assert_eq!(table, &coeffs.quant_tables[i], "quant table {} mismatch", i);
    }
}

#[test]
fn copy_critical_parameters_preserves_dimensions() {
    let pixels: Vec<u8> = vec![128u8; 100 * 80 * 3];
    let jpeg_data: Vec<u8> = compress(&pixels, 100, 80, PixelFormat::Rgb, 75, Subsampling::S420)
        .expect("compress failed");
    let coeffs = read_coefficients(&jpeg_data).expect("read_coefficients failed");

    let config = copy_critical_parameters(&coeffs);

    assert_eq!(config.width, 100);
    assert_eq!(config.height, 80);
}

#[test]
fn copy_critical_parameters_preserves_sampling_factors() {
    let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];
    let jpeg_data: Vec<u8> = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420)
        .expect("compress failed");
    let coeffs = read_coefficients(&jpeg_data).expect("read_coefficients failed");

    let config = copy_critical_parameters(&coeffs);

    assert_eq!(config.num_components, coeffs.components.len());
    for (i, comp) in config.component_info.iter().enumerate() {
        assert_eq!(comp.h_sampling, coeffs.components[i].h_sampling);
        assert_eq!(comp.v_sampling, coeffs.components[i].v_sampling);
        assert_eq!(
            comp.quant_table_index,
            coeffs.components[i].quant_table_index
        );
    }
}
