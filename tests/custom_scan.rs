use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, ScanScript, Subsampling};

#[test]
fn custom_scan_script_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let script = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        }, // DC first, al=1
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Y AC
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Cb AC
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // Cr AC
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        }, // DC refine
    ];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn custom_scan_script_grayscale_roundtrip() {
    let pixels = vec![200u8; 32 * 32];
    let script = vec![
        ScanScript {
            components: vec![0],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        }, // DC first
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        }, // AC full
        ScanScript {
            components: vec![0],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        }, // DC refine
    ];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(90)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn custom_scan_script_differs_from_default() {
    // A minimal 2-scan script (DC + AC) should produce different output
    // than the default multi-pass progression.
    let pixels = vec![100u8; 8 * 8 * 3];

    let default_jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .encode()
        .unwrap();

    let simple_script = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        }, // DC, no successive approx
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
    ];
    let custom_jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(simple_script)
        .encode()
        .unwrap();

    // Both should decode, but byte streams differ because scan scripts differ
    assert_ne!(default_jpeg, custom_jpeg);
    let img = decompress(&custom_jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
}
