use libjpeg_turbo_rs::FrameHeader;

#[test]
fn frame_header_has_lossless_field() {
    // Verify FrameHeader exposes is_lossless
    let header = FrameHeader {
        precision: 8,
        height: 100,
        width: 100,
        components: vec![],
        is_progressive: false,
        is_lossless: true,
    };
    assert!(header.is_lossless);
}
