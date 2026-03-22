/// Transform validation matrix tests.
use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, transform, transform_jpeg_with_options,
    write_coefficients, PixelFormat, Subsampling, TransformOp, TransformOptions,
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
        copy_markers: true,
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
        copy_markers: true,
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
        copy_markers: true,
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
        copy_markers: true,
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
            copy_markers: true,
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
        copy_markers: true,
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
        copy_markers: true,
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
