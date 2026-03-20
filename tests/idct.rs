use libjpeg_turbo_rs::decode::idct;

#[test]
fn idct_dc_only_block() {
    let mut coeffs = [0i16; 64];
    coeffs[0] = 800;

    let output = idct::idct_8x8(&coeffs);

    let first = output[0];
    for &val in output.iter() {
        assert_eq!(val, first, "DC-only block should produce uniform output");
    }
    assert!(first > 0, "DC-only output should be positive");
}

#[test]
fn idct_all_zeros() {
    let coeffs = [0i16; 64];
    let output = idct::idct_8x8(&coeffs);

    for &val in output.iter() {
        assert_eq!(val, 0);
    }
}

#[test]
fn idct_known_values() {
    let mut coeffs = [0i16; 64];
    coeffs[0] = 512;
    coeffs[1] = 100;

    let output = idct::idct_8x8(&coeffs);

    // Vertically constant for this pattern
    for col in 0..8 {
        assert_eq!(
            output[0 * 8 + col],
            output[1 * 8 + col],
            "rows should be identical for this coefficient pattern"
        );
    }
    // Horizontal variation expected
    assert_ne!(output[0], output[1]);
}
