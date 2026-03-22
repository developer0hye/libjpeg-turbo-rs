use libjpeg_turbo_rs::{
    jpeg_buf_size, yuv_buf_size, yuv_plane_height, yuv_plane_size, yuv_plane_width, Subsampling,
};

// --- jpeg_buf_size ---

#[test]
fn jpeg_buf_size_s420_640x480() {
    // MCU = 16x16, PAD(640,16)=640, PAD(480,16)=480
    // chromasf = 4*64/(16*16) = 1
    // 640 * 480 * (2+1) + 2048 = 921_600 + 2048 = 923_648... wait
    // Actually: 640 * 480 * 3 + 2048 = 921_600 + 2048 = nope
    // retval = PAD(w, mcuw) * PAD(h, mcuh) * (2 + chromasf) + 2048
    // = 640 * 480 * 3 + 2048 = 922_048... let me compute:
    // 640*480 = 307_200, * 3 = 921_600, + 2048 = 923_648
    let size: usize = jpeg_buf_size(640, 480, Subsampling::S420);
    assert_eq!(size, 923_648);
}

#[test]
fn jpeg_buf_size_s444_640x480() {
    // MCU = 8x8, PAD(640,8)=640, PAD(480,8)=480
    // chromasf = 4*64/(8*8) = 4
    // 640 * 480 * (2+4) + 2048 = 307_200 * 6 + 2048 = 1_843_200 + 2048 = 1_845_248
    let size: usize = jpeg_buf_size(640, 480, Subsampling::S444);
    assert_eq!(size, 1_845_248);
}

#[test]
fn jpeg_buf_size_s422_640x480() {
    // MCU = 16x8, chromasf = 4*64/(16*8) = 2
    // PAD(640,16)=640, PAD(480,8)=480
    // 640 * 480 * (2+2) + 2048 = 307_200 * 4 + 2048 = 1_228_800 + 2048 = 1_230_848
    let size: usize = jpeg_buf_size(640, 480, Subsampling::S422);
    assert_eq!(size, 1_230_848);
}

#[test]
fn jpeg_buf_size_s411_640x480() {
    // MCU = 32x8, chromasf = 4*64/(32*8) = 1
    // PAD(640,32)=640, PAD(480,8)=480
    // 640 * 480 * 3 + 2048 = 923_648
    let size: usize = jpeg_buf_size(640, 480, Subsampling::S411);
    assert_eq!(size, 923_648);
}

#[test]
fn jpeg_buf_size_non_aligned_dimensions() {
    // 641x481 with S420: MCU=16x16, PAD(641,16)=656, PAD(481,16)=496
    // chromasf = 1, 656 * 496 * 3 + 2048 = 325_376 * 3 + 2048 = 976_128 + 2048 = 978_176
    let size: usize = jpeg_buf_size(641, 481, Subsampling::S420);
    assert_eq!(size, 978_176);
}

#[test]
fn jpeg_buf_size_always_positive() {
    let size: usize = jpeg_buf_size(1, 1, Subsampling::S420);
    assert!(size > 0);
}

// --- yuv_plane_width ---

#[test]
fn yuv_plane_width_s444_luma() {
    // mcuw/8 = 1, PAD(640, 1) = 640, comp0 => 640
    assert_eq!(yuv_plane_width(0, 640, Subsampling::S444), 640);
}

#[test]
fn yuv_plane_width_s444_chroma() {
    // mcuw/8 = 1, PAD(640, 1) = 640, comp1 => 640 * 8 / 8 = 640
    assert_eq!(yuv_plane_width(1, 640, Subsampling::S444), 640);
}

#[test]
fn yuv_plane_width_s420_luma() {
    // mcuw/8 = 2, PAD(640, 2) = 640, comp0 => 640
    assert_eq!(yuv_plane_width(0, 640, Subsampling::S420), 640);
}

#[test]
fn yuv_plane_width_s420_chroma() {
    // mcuw/8 = 2, PAD(640, 2) = 640, comp1 => 640 * 8 / 16 = 320
    assert_eq!(yuv_plane_width(1, 640, Subsampling::S420), 320);
}

#[test]
fn yuv_plane_width_s411_chroma() {
    // mcuw/8 = 4, PAD(640, 4) = 640, comp1 => 640 * 8 / 32 = 160
    assert_eq!(yuv_plane_width(1, 640, Subsampling::S411), 160);
}

#[test]
fn yuv_plane_width_s420_odd_width() {
    // mcuw/8 = 2, PAD(641, 2) = 642, comp0 => 642
    assert_eq!(yuv_plane_width(0, 641, Subsampling::S420), 642);
    // comp1 => 642 * 8 / 16 = 321
    assert_eq!(yuv_plane_width(1, 641, Subsampling::S420), 321);
}

#[test]
fn yuv_plane_width_s411_odd_width() {
    // mcuw/8 = 4, PAD(641, 4) = 644, comp0 => 644
    assert_eq!(yuv_plane_width(0, 641, Subsampling::S411), 644);
    // comp1 => 644 * 8 / 32 = 161
    assert_eq!(yuv_plane_width(1, 641, Subsampling::S411), 161);
}

// --- yuv_plane_height ---

#[test]
fn yuv_plane_height_s444_luma() {
    assert_eq!(yuv_plane_height(0, 480, Subsampling::S444), 480);
}

#[test]
fn yuv_plane_height_s444_chroma() {
    assert_eq!(yuv_plane_height(1, 480, Subsampling::S444), 480);
}

#[test]
fn yuv_plane_height_s420_luma() {
    assert_eq!(yuv_plane_height(0, 480, Subsampling::S420), 480);
}

#[test]
fn yuv_plane_height_s420_chroma() {
    // mcuh/8 = 2, PAD(480, 2) = 480, comp1 => 480 * 8 / 16 = 240
    assert_eq!(yuv_plane_height(1, 480, Subsampling::S420), 240);
}

#[test]
fn yuv_plane_height_s441_chroma() {
    // mcuh/8 = 4, PAD(480, 4) = 480, comp1 => 480 * 8 / 32 = 120
    assert_eq!(yuv_plane_height(1, 480, Subsampling::S441), 120);
}

#[test]
fn yuv_plane_height_s420_odd_height() {
    // mcuh/8 = 2, PAD(481, 2) = 482, comp0 => 482
    assert_eq!(yuv_plane_height(0, 481, Subsampling::S420), 482);
    // comp1 => 482 * 8 / 16 = 241
    assert_eq!(yuv_plane_height(1, 481, Subsampling::S420), 241);
}

// --- yuv_plane_size ---

#[test]
fn yuv_plane_size_s420_luma() {
    // pw=640, ph=480, stride=pw => stride * (ph-1) + pw = 640*479 + 640 = 640*480 = 307_200
    assert_eq!(yuv_plane_size(0, 640, 480, Subsampling::S420), 307_200);
}

#[test]
fn yuv_plane_size_s420_chroma() {
    // pw=320, ph=240 => 320 * (240-1) + 320 = 320*240 = 76_800
    assert_eq!(yuv_plane_size(1, 640, 480, Subsampling::S420), 76_800);
}

// --- yuv_buf_size ---

#[test]
fn yuv_buf_size_s420_640x480() {
    // Y: 640*480=307_200, Cb: 320*240=76_800, Cr: 320*240=76_800
    // Total = 307_200 + 76_800 + 76_800 = 460_800
    let size: usize = yuv_buf_size(640, 480, Subsampling::S420);
    assert_eq!(size, 460_800);
}

#[test]
fn yuv_buf_size_s444_640x480() {
    // All planes: 640*480 each = 307_200 * 3 = 921_600
    let size: usize = yuv_buf_size(640, 480, Subsampling::S444);
    assert_eq!(size, 921_600);
}

#[test]
fn yuv_buf_size_consistent_with_plane_sizes() {
    for &sub in &[
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
        Subsampling::S441,
    ] {
        let total: usize = yuv_buf_size(640, 480, sub);
        let y: usize = yuv_plane_size(0, 640, 480, sub);
        let cb: usize = yuv_plane_size(1, 640, 480, sub);
        let cr: usize = yuv_plane_size(2, 640, 480, sub);
        assert_eq!(total, y + cb + cr, "mismatch for {:?}", sub);
    }
}

#[test]
fn yuv_buf_size_odd_dimensions() {
    // Verify non-aligned dimensions work without panic
    let size: usize = yuv_buf_size(641, 481, Subsampling::S420);
    assert!(size > 0);
}
