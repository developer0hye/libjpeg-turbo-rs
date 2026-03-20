use libjpeg_turbo_rs::decode::upsample;

#[test]
fn upsample_h2v1_doubles_width() {
    let input = [10u8, 20, 30, 40];
    let mut output = [0u8; 8];
    upsample::simple_h2v1(&input, 4, &mut output, 8);
    assert_eq!(output, [10, 10, 20, 20, 30, 30, 40, 40]);
}

#[test]
fn upsample_h2v2_doubles_both() {
    let input = [10u8, 20, 30, 40];
    let mut output = [0u8; 16];
    upsample::simple_h2v2(&input, 2, 2, &mut output, 4, 4);
    #[rustfmt::skip]
    let expected = [
        10, 10, 20, 20,
        10, 10, 20, 20,
        30, 30, 40, 40,
        30, 30, 40, 40,
    ];
    assert_eq!(output, expected);
}

#[test]
fn fancy_h2v1_interpolates() {
    let input = [0u8, 100, 200, 100];
    let mut output = [0u8; 8];
    upsample::fancy_h2v1(&input, 4, &mut output, 8);
    assert_ne!(
        output[0], output[1],
        "fancy should interpolate, not duplicate"
    );
}
