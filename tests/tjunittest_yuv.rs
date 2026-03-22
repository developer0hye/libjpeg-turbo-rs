/// YUV conversion validation tests.
use libjpeg_turbo_rs::api::yuv;
use libjpeg_turbo_rs::{
    compress, decompress_to, yuv_buf_size, yuv_plane_height, yuv_plane_size, yuv_plane_width,
    PixelFormat, Subsampling,
};

fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8);
            pixels.push(((y * 255) / height.max(1)) as u8);
            pixels.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    pixels
}

const COLOR_SUBSAMPLING: [Subsampling; 6] = [
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S420,
    Subsampling::S440,
    Subsampling::S411,
    Subsampling::S441,
];

fn yuv_roundtrip_helper(subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let original: Vec<u8> = gradient_rgb(w, h);
    let yuv_packed: Vec<u8> = yuv::encode_yuv(&original, w, h, PixelFormat::Rgb, subsamp).unwrap();
    assert_eq!(
        yuv_packed.len(),
        yuv_buf_size(w, h, subsamp),
        "size {:?}",
        subsamp
    );
    let decoded: Vec<u8> = yuv::decode_yuv(&yuv_packed, w, h, subsamp, PixelFormat::Rgb).unwrap();
    assert_eq!(decoded.len(), original.len());
    for i in 0..original.len() {
        let diff: i16 = original[i] as i16 - decoded[i] as i16;
        assert!(
            diff.abs() <= 12,
            "YUV {:?} byte {} diff={}",
            subsamp,
            i,
            diff
        );
    }
}

#[test]
fn tjunittest_yuv_roundtrip_444() {
    yuv_roundtrip_helper(Subsampling::S444);
}
#[test]
fn tjunittest_yuv_roundtrip_422() {
    yuv_roundtrip_helper(Subsampling::S422);
}
#[test]
fn tjunittest_yuv_roundtrip_420() {
    yuv_roundtrip_helper(Subsampling::S420);
}
#[test]
fn tjunittest_yuv_roundtrip_440() {
    yuv_roundtrip_helper(Subsampling::S440);
}
#[test]
fn tjunittest_yuv_roundtrip_411() {
    yuv_roundtrip_helper(Subsampling::S411);
}
#[test]
fn tjunittest_yuv_roundtrip_441() {
    yuv_roundtrip_helper(Subsampling::S441);
}

#[test]
fn tjunittest_yuv_roundtrip_various_sizes() {
    for &(w, h) in &[(16usize, 16usize), (35, 27), (48, 48), (100, 1)] {
        for &ss in &[Subsampling::S444, Subsampling::S420] {
            let orig: Vec<u8> = gradient_rgb(w, h);
            let yuv: Vec<u8> = yuv::encode_yuv(&orig, w, h, PixelFormat::Rgb, ss).unwrap();
            let dec: Vec<u8> = yuv::decode_yuv(&yuv, w, h, ss, PixelFormat::Rgb).unwrap();
            assert_eq!(dec.len(), orig.len(), "{}x{} {:?}", w, h, ss);
            for i in 0..orig.len() {
                let diff: i16 = orig[i] as i16 - dec[i] as i16;
                assert!(
                    diff.abs() <= 12,
                    "{}x{} {:?} byte {} diff={}",
                    w,
                    h,
                    ss,
                    i,
                    diff
                );
            }
        }
    }
}

#[test]
fn tjunittest_yuv_grayscale_roundtrip() {
    let (w, h): (usize, usize) = (48, 48);
    let mut orig: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            orig.push(((x + y) * 4) as u8);
        }
    }
    let yuv: Vec<u8> =
        yuv::encode_yuv(&orig, w, h, PixelFormat::Grayscale, Subsampling::S444).unwrap();
    assert_eq!(yuv.len(), w * h);
    for i in 0..orig.len() {
        assert_eq!(yuv[i], orig[i], "gray Y byte {}", i);
    }
}

#[test]
fn tjunittest_yuv_plane_sizes_aligned() {
    let (w, h): (usize, usize) = (48, 48);
    let cases: [(Subsampling, usize, usize); 6] = [
        (Subsampling::S444, 48, 48),
        (Subsampling::S422, 24, 48),
        (Subsampling::S420, 24, 24),
        (Subsampling::S440, 48, 24),
        (Subsampling::S411, 12, 48),
        (Subsampling::S441, 48, 12),
    ];
    for &(ss, cw, ch) in &cases {
        assert_eq!(yuv_plane_width(0, w, ss), w, "Y width {:?}", ss);
        assert_eq!(yuv_plane_height(0, h, ss), h, "Y height {:?}", ss);
        assert_eq!(yuv_plane_width(1, w, ss), cw, "Cb width {:?}", ss);
        assert_eq!(yuv_plane_height(1, h, ss), ch, "Cb height {:?}", ss);
        let total: usize =
            yuv_plane_size(0, w, h, ss) + yuv_plane_size(1, w, h, ss) + yuv_plane_size(2, w, h, ss);
        assert_eq!(total, yuv_buf_size(w, h, ss), "total {:?}", ss);
    }
}

#[test]
fn tjunittest_yuv_plane_sizes_nonaligned() {
    let (w, h): (usize, usize) = (35, 27);
    for &ss in &COLOR_SUBSAMPLING {
        assert!(yuv_plane_width(0, w, ss) >= w, "Y width {:?}", ss);
        assert!(yuv_plane_height(0, h, ss) >= h, "Y height {:?}", ss);
        assert!(yuv_plane_width(1, w, ss) > 0, "Cb width {:?}", ss);
        assert!(yuv_buf_size(w, h, ss) > 0, "total {:?}", ss);
    }
}

fn compress_from_yuv_helper(subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let orig: Vec<u8> = gradient_rgb(w, h);
    let yuv: Vec<u8> = yuv::encode_yuv(&orig, w, h, PixelFormat::Rgb, subsamp).unwrap();
    let jpeg: Vec<u8> = yuv::compress_from_yuv(&yuv, w, h, subsamp, 90).unwrap();
    assert!(!jpeg.is_empty());
    let img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
    assert_eq!((img.width, img.height), (w, h));
    let direct: Vec<u8> = compress(&orig, w, h, PixelFormat::Rgb, 90, subsamp).unwrap();
    let dimg = decompress_to(&direct, PixelFormat::Rgb).unwrap();
    for i in 0..orig.len() {
        let diff: i16 = img.data[i] as i16 - dimg.data[i] as i16;
        assert!(
            diff.abs() <= 12,
            "yuv vs direct {:?} byte {} diff={}",
            subsamp,
            i,
            diff
        );
    }
}

#[test]
fn tjunittest_compress_from_yuv_444() {
    compress_from_yuv_helper(Subsampling::S444);
}
#[test]
fn tjunittest_compress_from_yuv_422() {
    compress_from_yuv_helper(Subsampling::S422);
}
#[test]
fn tjunittest_compress_from_yuv_420() {
    compress_from_yuv_helper(Subsampling::S420);
}

fn decompress_to_yuv_helper(subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let orig: Vec<u8> = gradient_rgb(w, h);
    let jpeg: Vec<u8> = compress(&orig, w, h, PixelFormat::Rgb, 90, subsamp).unwrap();
    let direct: Vec<u8> = decompress_to(&jpeg, PixelFormat::Rgb).unwrap().data;
    let (yuv, yw, yh, ys) = yuv::decompress_to_yuv(&jpeg).unwrap();
    let via: Vec<u8> = yuv::decode_yuv(&yuv, yw, yh, ys, PixelFormat::Rgb).unwrap();
    assert_eq!(via.len(), direct.len());
    for i in 0..direct.len() {
        let diff: i16 = via[i] as i16 - direct[i] as i16;
        assert!(
            diff.abs() <= 12,
            "yuv decode {:?} byte {} diff={}",
            subsamp,
            i,
            diff
        );
    }
}

#[test]
fn tjunittest_decompress_to_yuv_444() {
    decompress_to_yuv_helper(Subsampling::S444);
}
#[test]
fn tjunittest_decompress_to_yuv_422() {
    decompress_to_yuv_helper(Subsampling::S422);
}
#[test]
fn tjunittest_decompress_to_yuv_420() {
    decompress_to_yuv_helper(Subsampling::S420);
}

#[test]
fn tjunittest_yuv_encode_multiple_pixel_formats() {
    let (w, h): (usize, usize) = (48, 48);
    for &pf in &[
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
    ] {
        let bpp: usize = pf.bytes_per_pixel();
        let mut px: Vec<u8> = vec![0u8; w * h * bpp];
        for y in 0..h {
            for x in 0..w {
                let idx: usize = (y * w + x) * bpp;
                px[idx + pf.red_offset().unwrap()] = ((x * 255) / w) as u8;
                px[idx + pf.green_offset().unwrap()] = ((y * 255) / h) as u8;
                px[idx + pf.blue_offset().unwrap()] = 128;
                if bpp == 4 {
                    let ao = 6
                        - pf.red_offset().unwrap()
                        - pf.green_offset().unwrap()
                        - pf.blue_offset().unwrap();
                    if ao < bpp {
                        px[idx + ao] = 255;
                    }
                }
            }
        }
        let yuv: Vec<u8> = yuv::encode_yuv(&px, w, h, pf, Subsampling::S444).unwrap();
        assert_eq!(yuv.len(), yuv_buf_size(w, h, Subsampling::S444), "{:?}", pf);
        assert!(yuv.iter().any(|&v| v > 0), "{:?}", pf);
    }
}
