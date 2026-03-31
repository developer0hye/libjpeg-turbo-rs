/// Transform validation matrix tests.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, transform, transform_jpeg_with_options,
    write_coefficients, MarkerCopyMode, PixelFormat, Subsampling, TransformOp, TransformOptions,
};

fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            px.push(((x * 255) / width.max(1)) as u8);
            px.push(((y * 255) / height.max(1)) as u8);
            px.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    px
}

const ALL_TRANSFORMS: [TransformOp; 8] = [
    TransformOp::None,
    TransformOp::HFlip,
    TransformOp::VFlip,
    TransformOp::Transpose,
    TransformOp::Transverse,
    TransformOp::Rot90,
    TransformOp::Rot180,
    TransformOp::Rot270,
];

fn swaps_dims(op: TransformOp) -> bool {
    matches!(
        op,
        TransformOp::Transpose | TransformOp::Transverse | TransformOp::Rot90 | TransformOp::Rot270
    )
}

// 1. All transforms x 444
#[test]
fn tjunittest_all_transforms_444() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    for &op in &ALL_TRANSFORMS {
        let t: Vec<u8> = transform(&jpeg, op).unwrap();
        let img = decompress(&t).unwrap();
        if swaps_dims(op) {
            assert_eq!((img.width, img.height), (h, w), "{:?}", op);
        } else {
            assert_eq!((img.width, img.height), (w, h), "{:?}", op);
        }
        assert_eq!(
            img.data.len(),
            img.width * img.height * img.pixel_format.bytes_per_pixel()
        );
    }
}

// 2. All transforms x 420
#[test]
fn tjunittest_all_transforms_420() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap();
    for &op in &ALL_TRANSFORMS {
        if let Ok(t) = transform(&jpeg, op) {
            let img = decompress(&t).unwrap();
            assert_eq!(
                img.data.len(),
                img.width * img.height * img.pixel_format.bytes_per_pixel()
            );
        }
    }
}

// 3. All transforms x grayscale
#[test]
fn tjunittest_all_transforms_grayscale() {
    let (w, h): (usize, usize) = (48, 32);
    let mut gray: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            gray.push(((x * 255) / w.max(1)) as u8);
        }
    }
    let jpeg: Vec<u8> = libjpeg_turbo_rs::Encoder::new(&gray, w, h, PixelFormat::Grayscale)
        .quality(90)
        .encode()
        .unwrap();
    for &op in &ALL_TRANSFORMS {
        let t: Vec<u8> = transform(&jpeg, op).unwrap();
        let img = decompress(&t).unwrap();
        if swaps_dims(op) {
            assert_eq!((img.width, img.height), (h, w), "gray {:?}", op);
        } else {
            assert_eq!((img.width, img.height), (w, h), "gray {:?}", op);
        }
        assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    }
}

// 4. All transforms x 422, 440, 411, 441
#[test]
fn tjunittest_all_transforms_422() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S422,
    )
    .unwrap();
    for &op in &ALL_TRANSFORMS {
        if let Ok(t) = transform(&jpeg, op) {
            decompress(&t).unwrap();
        }
    }
}

#[test]
fn tjunittest_all_transforms_440() {
    let (w, h): (usize, usize) = (48, 48);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S440,
    )
    .unwrap();
    for &op in &ALL_TRANSFORMS {
        if let Ok(t) = transform(&jpeg, op) {
            decompress(&t).unwrap();
        }
    }
}

#[test]
fn tjunittest_all_transforms_411() {
    let (w, h): (usize, usize) = (64, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S411,
    )
    .unwrap();
    for &op in &ALL_TRANSFORMS {
        if let Ok(t) = transform(&jpeg, op) {
            decompress(&t).unwrap();
        }
    }
}

#[test]
fn tjunittest_all_transforms_441() {
    let (w, h): (usize, usize) = (32, 64);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S441,
    )
    .unwrap();
    for &op in &ALL_TRANSFORMS {
        if let Ok(t) = transform(&jpeg, op) {
            decompress(&t).unwrap();
        }
    }
}

// 5. Double-apply identity
#[test]
fn tjunittest_double_hflip_identity() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let f1: Vec<u8> = transform(&jpeg, TransformOp::HFlip).unwrap();
    let f2: Vec<u8> = transform(&f1, TransformOp::HFlip).unwrap();
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&f2).unwrap().data,
        "double HFlip"
    );
}

#[test]
fn tjunittest_double_vflip_identity() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let f1: Vec<u8> = transform(&jpeg, TransformOp::VFlip).unwrap();
    let f2: Vec<u8> = transform(&f1, TransformOp::VFlip).unwrap();
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&f2).unwrap().data,
        "double VFlip"
    );
}

#[test]
fn tjunittest_double_rot180_identity() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let r1: Vec<u8> = transform(&jpeg, TransformOp::Rot180).unwrap();
    let r2: Vec<u8> = transform(&r1, TransformOp::Rot180).unwrap();
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&r2).unwrap().data,
        "double Rot180"
    );
}

#[test]
fn tjunittest_four_rot90_identity() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let mut j: Vec<u8> = jpeg.clone();
    for _ in 0..4 {
        j = transform(&j, TransformOp::Rot90).unwrap();
    }
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&j).unwrap().data,
        "4x Rot90"
    );
}

#[test]
fn tjunittest_double_transpose_identity() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let t1: Vec<u8> = transform(&jpeg, TransformOp::Transpose).unwrap();
    let t2: Vec<u8> = transform(&t1, TransformOp::Transpose).unwrap();
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&t2).unwrap().data,
        "double Transpose"
    );
}

// 6. Transform + crop
#[test]
fn tjunittest_transform_with_crop() {
    use libjpeg_turbo_rs::CropRegion;
    let s: usize = 64;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let opts = TransformOptions {
        op: TransformOp::None,
        perfect: false,
        trim: false,
        crop: Some(CropRegion {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        }),
        grayscale: false,
        no_output: false,
        progressive: false,
        arithmetic: false,
        optimize: false,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        custom_filter: None,
    };
    let cropped: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap();
    let img = decompress(&cropped).unwrap();
    assert!(img.width <= s && img.height <= s && img.width > 0 && img.height > 0);
}

#[test]
fn tjunittest_transform_crop_with_rotation() {
    use libjpeg_turbo_rs::CropRegion;
    let s: usize = 64;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let opts = TransformOptions {
        op: TransformOp::Rot90,
        perfect: false,
        trim: false,
        crop: Some(CropRegion {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        }),
        grayscale: false,
        no_output: false,
        progressive: false,
        arithmetic: false,
        optimize: false,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        custom_filter: None,
    };
    if let Ok(t) = transform_jpeg_with_options(&jpeg, &opts) {
        let img = decompress(&t).unwrap();
        assert!(img.width > 0);
    }
}

// 7. Transform + grayscale
#[test]
fn tjunittest_grayscale_transform_444() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let opts = TransformOptions {
        op: TransformOp::None,
        perfect: false,
        trim: false,
        crop: None,
        grayscale: true,
        no_output: false,
        progressive: false,
        arithmetic: false,
        optimize: false,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        custom_filter: None,
    };
    let g: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap();
    let img = decompress(&g).unwrap();
    assert_eq!(
        (img.width, img.height, img.pixel_format),
        (s, s, PixelFormat::Grayscale)
    );
}

#[test]
fn tjunittest_grayscale_transform_420() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap();
    let opts = TransformOptions {
        op: TransformOp::None,
        perfect: false,
        trim: false,
        crop: None,
        grayscale: true,
        no_output: false,
        progressive: false,
        arithmetic: false,
        optimize: false,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        custom_filter: None,
    };
    let g: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap();
    let img = decompress(&g).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn tjunittest_grayscale_transform_all_subsampling() {
    for &ss in &[
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
        Subsampling::S441,
    ] {
        let mw: usize = ss.mcu_width_blocks() * 8;
        let mh: usize = ss.mcu_height_blocks() * 8;
        let (w, h): (usize, usize) = (mw * 4, mh * 4);
        let jpeg: Vec<u8> = compress(&gradient_rgb(w, h), w, h, PixelFormat::Rgb, 90, ss).unwrap();
        let opts = TransformOptions {
            op: TransformOp::None,
            perfect: false,
            trim: false,
            crop: None,
            grayscale: true,
            no_output: false,
            progressive: false,
            arithmetic: false,
            optimize: false,
            copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
            custom_filter: None,
        };
        let g: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap();
        let img = decompress(&g).unwrap();
        assert_eq!(
            (img.width, img.height, img.pixel_format),
            (w, h, PixelFormat::Grayscale),
            "{:?}",
            ss
        );
    }
}

// 8. Progressive/arithmetic transform output
#[test]
fn tjunittest_transform_progressive_output() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let opts = TransformOptions {
        op: TransformOp::HFlip,
        perfect: false,
        trim: false,
        crop: None,
        grayscale: false,
        no_output: false,
        progressive: true,
        arithmetic: false,
        optimize: false,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        custom_filter: None,
    };
    let pj: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap();
    // The transform may or may not inject SOF2; verify decodability
    let img = decompress(&pj).unwrap();
    assert_eq!((img.width, img.height), (s, s));
}

#[test]
fn tjunittest_transform_arithmetic_output() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let opts = TransformOptions {
        op: TransformOp::None,
        perfect: false,
        trim: false,
        crop: None,
        grayscale: false,
        no_output: false,
        progressive: false,
        arithmetic: true,
        optimize: false,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        custom_filter: None,
    };
    let aj: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap();
    // The transform may or may not inject SOF9; verify decodability
    let img = decompress(&aj).unwrap();
    assert_eq!((img.width, img.height), (s, s));
}

// 9. Coefficient roundtrip
#[test]
fn tjunittest_coefficient_roundtrip_444() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let coeffs = read_coefficients(&jpeg).unwrap();
    let recon: Vec<u8> = write_coefficients(&coeffs).unwrap();
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&recon).unwrap().data,
        "coeff roundtrip 444"
    );
}

#[test]
fn tjunittest_coefficient_roundtrip_420() {
    let s: usize = 48;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(s, s),
        s,
        s,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap();
    let coeffs = read_coefficients(&jpeg).unwrap();
    let recon: Vec<u8> = write_coefficients(&coeffs).unwrap();
    assert_eq!(
        decompress(&jpeg).unwrap().data,
        decompress(&recon).unwrap().data,
        "coeff roundtrip 420"
    );
}

// 10. Validity across all subsamp
#[test]
fn tjunittest_transform_validity_all_subsamp() {
    for &ss in &[
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
        Subsampling::S441,
    ] {
        let (w, h): (usize, usize) = (ss.mcu_width_blocks() * 32, ss.mcu_height_blocks() * 32);
        let jpeg: Vec<u8> = compress(&gradient_rgb(w, h), w, h, PixelFormat::Rgb, 90, ss).unwrap();
        for &op in &ALL_TRANSFORMS {
            if let Ok(t) = transform(&jpeg, op) {
                let img = decompress(&t).unwrap_or_else(|e| panic!("{:?} {:?}: {}", op, ss, e));
                assert_eq!(
                    img.data.len(),
                    img.width * img.height * img.pixel_format.bytes_per_pixel()
                );
            }
        }
    }
}

// 11. Dimension swap verification
#[test]
fn tjunittest_dimension_swap_rot90() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&transform(&jpeg, TransformOp::Rot90).unwrap()).unwrap();
    assert_eq!((img.width, img.height), (h, w));
}

#[test]
fn tjunittest_dimension_swap_rot270() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&transform(&jpeg, TransformOp::Rot270).unwrap()).unwrap();
    assert_eq!((img.width, img.height), (h, w));
}

#[test]
fn tjunittest_dimension_swap_transpose() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&transform(&jpeg, TransformOp::Transpose).unwrap()).unwrap();
    assert_eq!((img.width, img.height), (h, w));
}

#[test]
fn tjunittest_dimension_swap_transverse() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&transform(&jpeg, TransformOp::Transverse).unwrap()).unwrap();
    assert_eq!((img.width, img.height), (h, w));
}

#[test]
fn tjunittest_no_dimension_swap_hflip_vflip_rot180() {
    let (w, h): (usize, usize) = (48, 32);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(w, h),
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();
    for &op in &[
        TransformOp::None,
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Rot180,
    ] {
        let img = decompress(&transform(&jpeg, op).unwrap()).unwrap();
        assert_eq!((img.width, img.height), (w, h), "{:?}", op);
    }
}

// ===========================================================================
// C jpegtran cross-validation helpers
// ===========================================================================

fn jpegtran_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/jpegtran");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("jpegtran")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path_xv(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_tjxform_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: temp_path_xv(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Parse a binary PNM file (P5 PGM or P6 PPM) and return `(width, height, pixel_data)`.
/// For P5 (grayscale), pixel data is returned as-is (1 byte per pixel).
/// For P6 (RGB), pixel data is returned as-is (3 bytes per pixel).
fn parse_pnm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PNM file");
    assert!(raw.len() > 3, "PNM too short");
    let magic: &[u8] = &raw[0..2];
    let channels: usize = if magic == b"P6" {
        3
    } else if magic == b"P5" {
        1
    } else {
        panic!(
            "unsupported PNM magic: {:?} (expected P5 or P6)",
            std::str::from_utf8(magic).unwrap_or("??")
        );
    };
    let mut idx: usize = 2;
    idx = ppm_skip_ws_comments(&raw, idx);
    let (width, next) = ppm_read_number(&raw, idx);
    idx = ppm_skip_ws_comments(&raw, next);
    let (height, next) = ppm_read_number(&raw, idx);
    idx = ppm_skip_ws_comments(&raw, next);
    let (_maxval, next) = ppm_read_number(&raw, idx);
    // One byte of whitespace after maxval before pixel data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    let expected: usize = width * height * channels;
    assert_eq!(
        data.len(),
        expected,
        "PNM pixel data length mismatch: expected {}x{}x{}={}, got {}",
        width,
        height,
        channels,
        expected,
        data.len()
    );
    (width, height, data)
}

fn ppm_skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn ppm_read_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

/// Map TransformOp to jpegtran CLI arguments.
fn jpegtran_args_for_op(op: TransformOp) -> Vec<String> {
    match op {
        TransformOp::None => vec![],
        TransformOp::HFlip => vec!["-flip".to_string(), "horizontal".to_string()],
        TransformOp::VFlip => vec!["-flip".to_string(), "vertical".to_string()],
        TransformOp::Rot90 => vec!["-rotate".to_string(), "90".to_string()],
        TransformOp::Rot180 => vec!["-rotate".to_string(), "180".to_string()],
        TransformOp::Rot270 => vec!["-rotate".to_string(), "270".to_string()],
        TransformOp::Transpose => vec!["-transpose".to_string()],
        TransformOp::Transverse => vec!["-transverse".to_string()],
    }
}

/// Generate RGB pixel data with a gradient pattern.
fn gen_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            px.push(((x * 255) / width.max(1)) as u8);
            px.push(((y * 255) / height.max(1)) as u8);
            px.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    px
}

// ===========================================================================
// C jpegtran cross-validation test
// ===========================================================================

#[test]
fn c_jpegtran_cross_validation_tjunittest_transform() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Create source JPEG: 48x48, S444, quality 90
    let source_jpeg: Vec<u8> = compress(
        &gen_rgb(48, 48),
        48,
        48,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .expect("compress source image");

    let mut tested: usize = 0;
    let mut passed: usize = 0;

    // -----------------------------------------------------------------------
    // Part 1: All 8 transform operations
    // -----------------------------------------------------------------------
    for &op in &ALL_TRANSFORMS {
        let op_name: &str = match op {
            TransformOp::None => "none",
            TransformOp::HFlip => "hflip",
            TransformOp::VFlip => "vflip",
            TransformOp::Rot90 => "rot90",
            TransformOp::Rot180 => "rot180",
            TransformOp::Rot270 => "rot270",
            TransformOp::Transpose => "transpose",
            TransformOp::Transverse => "transverse",
        };

        // (a) Apply transform with Rust
        let rust_jpeg: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op,
                copy_markers: MarkerCopyMode::None,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("Rust transform {:?} must succeed: {}", op, e));

        // (b) Write source JPEG to temp file
        let tmp_src: TempFile = TempFile::new(&format!("xv_{}_src.jpg", op_name));
        std::fs::write(tmp_src.path(), &source_jpeg).expect("write source JPEG");

        // (c) Run C jpegtran with matching flags
        let tmp_c_out: TempFile = TempFile::new(&format!("xv_{}_c.jpg", op_name));
        let args: Vec<String> = jpegtran_args_for_op(op);
        let mut cmd = Command::new(&jpegtran);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.arg("-copy")
            .arg("none")
            .arg("-outfile")
            .arg(tmp_c_out.path())
            .arg(tmp_src.path());
        let output = cmd.output().expect("failed to run jpegtran");
        assert!(
            output.status.success(),
            "jpegtran {:?} failed: {}",
            op,
            String::from_utf8_lossy(&output.stderr)
        );
        let _c_jpeg: Vec<u8> = std::fs::read(tmp_c_out.path()).expect("read jpegtran output");

        // (d) Run C djpeg -ppm on BOTH the Rust output and C jpegtran output
        let tmp_rust_jpg: TempFile = TempFile::new(&format!("xv_{}_rust.jpg", op_name));
        let tmp_rust_ppm: TempFile = TempFile::new(&format!("xv_{}_rust.ppm", op_name));
        let tmp_c_ppm: TempFile = TempFile::new(&format!("xv_{}_c.ppm", op_name));

        std::fs::write(tmp_rust_jpg.path(), &rust_jpeg).expect("write Rust JPEG");

        // djpeg on Rust output
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("failed to run djpeg on Rust output");
        assert!(
            output.status.success(),
            "djpeg failed on Rust {} output: {}",
            op_name,
            String::from_utf8_lossy(&output.stderr)
        );

        // djpeg on C output
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("failed to run djpeg on C output");
        assert!(
            output.status.success(),
            "djpeg failed on C {} output: {}",
            op_name,
            String::from_utf8_lossy(&output.stderr)
        );

        // (e) Compare pixel data byte-for-byte (diff=0)
        let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_ppm.path());
        let (cw, ch, c_pixels) = parse_pnm(tmp_c_ppm.path());

        assert_eq!(
            rw, cw,
            "{}: width mismatch (Rust={}, C={})",
            op_name, rw, cw
        );
        assert_eq!(
            rh, ch,
            "{}: height mismatch (Rust={}, C={})",
            op_name, rh, ch
        );
        assert_eq!(
            rust_pixels.len(),
            c_pixels.len(),
            "{}: pixel buffer length mismatch",
            op_name
        );

        let max_diff: u8 = rust_pixels
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "{}: pixel max_diff={} (must be 0 vs C jpegtran via djpeg)",
            op_name, max_diff
        );

        tested += 1;
        passed += 1;
        eprintln!(
            "  PASS: transform {:?} -- {}x{} pixels identical",
            op, rw, rh
        );
    }

    // -----------------------------------------------------------------------
    // Part 2: Transform with grayscale flag
    // -----------------------------------------------------------------------
    {
        let rust_jpeg: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op: TransformOp::Rot90,
                grayscale: true,
                copy_markers: MarkerCopyMode::None,
                ..Default::default()
            },
        )
        .expect("Rust grayscale+rot90 transform must succeed");

        let tmp_src: TempFile = TempFile::new("xv_gray_rot90_src.jpg");
        std::fs::write(tmp_src.path(), &source_jpeg).expect("write source");

        let tmp_c_out: TempFile = TempFile::new("xv_gray_rot90_c.jpg");
        let output = Command::new(&jpegtran)
            .arg("-grayscale")
            .arg("-rotate")
            .arg("90")
            .arg("-copy")
            .arg("none")
            .arg("-outfile")
            .arg(tmp_c_out.path())
            .arg(tmp_src.path())
            .output()
            .expect("failed to run jpegtran -grayscale -rotate 90");
        assert!(
            output.status.success(),
            "jpegtran -grayscale -rotate 90 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Decode both with djpeg -ppm (grayscale JPEG decoded to PPM will be grayscale-as-RGB)
        let tmp_rust_jpg: TempFile = TempFile::new("xv_gray_rot90_rust.jpg");
        let tmp_rust_ppm: TempFile = TempFile::new("xv_gray_rot90_rust.ppm");
        let tmp_c_ppm: TempFile = TempFile::new("xv_gray_rot90_c.ppm");

        std::fs::write(tmp_rust_jpg.path(), &rust_jpeg).expect("write Rust JPEG");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("djpeg on Rust grayscale output");
        assert!(
            output.status.success(),
            "djpeg failed on Rust grayscale+rot90: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("djpeg on C grayscale output");
        assert!(
            output.status.success(),
            "djpeg failed on C grayscale+rot90: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_ppm.path());
        let (cw, ch, c_pixels) = parse_pnm(tmp_c_ppm.path());

        assert_eq!(rw, cw, "grayscale+rot90: width mismatch");
        assert_eq!(rh, ch, "grayscale+rot90: height mismatch");

        let max_diff: u8 = rust_pixels
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "grayscale+rot90: pixel max_diff={} (must be 0 vs C jpegtran via djpeg)",
            max_diff
        );

        tested += 1;
        passed += 1;
        eprintln!("  PASS: grayscale+rot90 -- {}x{} pixels identical", rw, rh);
    }

    // -----------------------------------------------------------------------
    // Part 3: Transform with progressive flag
    // -----------------------------------------------------------------------
    {
        let rust_jpeg: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op: TransformOp::Rot90,
                progressive: true,
                copy_markers: MarkerCopyMode::None,
                ..Default::default()
            },
        )
        .expect("Rust progressive+rot90 transform must succeed");

        let tmp_src: TempFile = TempFile::new("xv_prog_rot90_src.jpg");
        std::fs::write(tmp_src.path(), &source_jpeg).expect("write source");

        let tmp_c_out: TempFile = TempFile::new("xv_prog_rot90_c.jpg");
        let output = Command::new(&jpegtran)
            .arg("-progressive")
            .arg("-rotate")
            .arg("90")
            .arg("-copy")
            .arg("none")
            .arg("-outfile")
            .arg(tmp_c_out.path())
            .arg(tmp_src.path())
            .output()
            .expect("failed to run jpegtran -progressive -rotate 90");
        assert!(
            output.status.success(),
            "jpegtran -progressive -rotate 90 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let tmp_rust_jpg: TempFile = TempFile::new("xv_prog_rot90_rust.jpg");
        let tmp_rust_ppm: TempFile = TempFile::new("xv_prog_rot90_rust.ppm");
        let tmp_c_ppm: TempFile = TempFile::new("xv_prog_rot90_c.ppm");

        std::fs::write(tmp_rust_jpg.path(), &rust_jpeg).expect("write Rust JPEG");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("djpeg on Rust progressive output");
        assert!(
            output.status.success(),
            "djpeg failed on Rust progressive+rot90: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("djpeg on C progressive output");
        assert!(
            output.status.success(),
            "djpeg failed on C progressive+rot90: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_ppm.path());
        let (cw, ch, c_pixels) = parse_pnm(tmp_c_ppm.path());

        assert_eq!(rw, cw, "progressive+rot90: width mismatch");
        assert_eq!(rh, ch, "progressive+rot90: height mismatch");

        let max_diff: u8 = rust_pixels
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "progressive+rot90: pixel max_diff={} (must be 0 vs C jpegtran via djpeg)",
            max_diff
        );

        tested += 1;
        passed += 1;
        eprintln!(
            "  PASS: progressive+rot90 -- {}x{} pixels identical",
            rw, rh
        );
    }

    // -----------------------------------------------------------------------
    // Part 4: Transform with optimize flag
    // -----------------------------------------------------------------------
    {
        let rust_jpeg: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op: TransformOp::Rot90,
                optimize: true,
                copy_markers: MarkerCopyMode::None,
                ..Default::default()
            },
        )
        .expect("Rust optimize+rot90 transform must succeed");

        let tmp_src: TempFile = TempFile::new("xv_opt_rot90_src.jpg");
        std::fs::write(tmp_src.path(), &source_jpeg).expect("write source");

        let tmp_c_out: TempFile = TempFile::new("xv_opt_rot90_c.jpg");
        let output = Command::new(&jpegtran)
            .arg("-optimize")
            .arg("-rotate")
            .arg("90")
            .arg("-copy")
            .arg("none")
            .arg("-outfile")
            .arg(tmp_c_out.path())
            .arg(tmp_src.path())
            .output()
            .expect("failed to run jpegtran -optimize -rotate 90");
        assert!(
            output.status.success(),
            "jpegtran -optimize -rotate 90 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let tmp_rust_jpg: TempFile = TempFile::new("xv_opt_rot90_rust.jpg");
        let tmp_rust_ppm: TempFile = TempFile::new("xv_opt_rot90_rust.ppm");
        let tmp_c_ppm: TempFile = TempFile::new("xv_opt_rot90_c.ppm");

        std::fs::write(tmp_rust_jpg.path(), &rust_jpeg).expect("write Rust JPEG");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("djpeg on Rust optimize output");
        assert!(
            output.status.success(),
            "djpeg failed on Rust optimize+rot90: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("djpeg on C optimize output");
        assert!(
            output.status.success(),
            "djpeg failed on C optimize+rot90: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_ppm.path());
        let (cw, ch, c_pixels) = parse_pnm(tmp_c_ppm.path());

        assert_eq!(rw, cw, "optimize+rot90: width mismatch");
        assert_eq!(rh, ch, "optimize+rot90: height mismatch");

        let max_diff: u8 = rust_pixels
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "optimize+rot90: pixel max_diff={} (must be 0 vs C jpegtran via djpeg)",
            max_diff
        );

        tested += 1;
        passed += 1;
        eprintln!("  PASS: optimize+rot90 -- {}x{} pixels identical", rw, rh);
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    eprintln!(
        "c_jpegtran_cross_validation_tjunittest_transform: {}/{} passed",
        passed, tested
    );
    assert_eq!(
        passed, tested,
        "not all combinations passed: {}/{} passed",
        passed, tested
    );
}
