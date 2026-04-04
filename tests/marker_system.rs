use libjpeg_turbo_rs::{decompress, DensityInfo, DensityUnit, Encoder, PixelFormat};

#[test]
fn comment_marker_roundtrip() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment("hello world")
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.comment.as_deref(), Some("hello world"));
}

#[test]
fn default_density_is_72dpi() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.density.x, 1);
    assert_eq!(img.density.y, 1);
    assert_eq!(img.density.unit, DensityUnit::Unknown);
}

#[test]
fn write_dri_marker() {
    let mut buf = Vec::new();
    libjpeg_turbo_rs::encode::marker_writer::write_dri(&mut buf, 100);
    assert_eq!(buf[0], 0xFF);
    assert_eq!(buf[1], 0xDD);
    assert_eq!(u16::from_be_bytes([buf[2], buf[3]]), 4);
    assert_eq!(u16::from_be_bytes([buf[4], buf[5]]), 100);
}

#[test]
fn write_com_marker() {
    let mut buf = Vec::new();
    libjpeg_turbo_rs::encode::marker_writer::write_com(&mut buf, "test");
    assert_eq!(buf[0], 0xFF);
    assert_eq!(buf[1], 0xFE);
    assert_eq!(&buf[4..8], b"test");
}

#[test]
fn density_info_default() {
    let d = DensityInfo::default();
    assert_eq!(d.unit, DensityUnit::Unknown);
    assert_eq!(d.x, 1);
    assert_eq!(d.y, 1);
}
