use libjpeg_turbo_rs::common::quant_table::QuantTable;
use libjpeg_turbo_rs::common::types::*;

#[test]
fn subsampling_block_dimensions() {
    assert_eq!(Subsampling::S444.mcu_width_blocks(), 1);
    assert_eq!(Subsampling::S444.mcu_height_blocks(), 1);
    assert_eq!(Subsampling::S422.mcu_width_blocks(), 2);
    assert_eq!(Subsampling::S422.mcu_height_blocks(), 1);
    assert_eq!(Subsampling::S420.mcu_width_blocks(), 2);
    assert_eq!(Subsampling::S420.mcu_height_blocks(), 2);
}

#[test]
fn pixel_format_bytes_per_pixel() {
    assert_eq!(PixelFormat::Rgb.bytes_per_pixel(), 3);
    assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Grayscale.bytes_per_pixel(), 1);
}

#[test]
fn color_space_num_components() {
    assert_eq!(ColorSpace::Grayscale.num_components(), 1);
    assert_eq!(ColorSpace::YCbCr.num_components(), 3);
}

#[test]
fn quant_table_from_zigzag_and_natural_order() {
    let mut zigzag_data = [1u16; 64];
    zigzag_data[0] = 16;
    zigzag_data[1] = 11;
    zigzag_data[2] = 10;

    let table = QuantTable::from_zigzag(&zigzag_data);

    assert_eq!(table.get(0, 0), 16);
    assert_eq!(table.get(0, 1), 11);
    assert_eq!(table.get(1, 0), 10);
}
