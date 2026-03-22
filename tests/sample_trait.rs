use libjpeg_turbo_rs::Sample;

#[test]
fn u8_sample_constants() {
    assert_eq!(u8::BITS_PER_SAMPLE, 8);
    assert_eq!(u8::MAX_VAL, 255);
    assert_eq!(u8::CENTER, 128);
    assert!(!u8::IS_LOSSLESS_ONLY);
}

#[test]
fn i16_sample_constants() {
    assert_eq!(i16::BITS_PER_SAMPLE, 12);
    assert_eq!(i16::MAX_VAL, 4095);
    assert_eq!(i16::CENTER, 2048);
    assert!(!i16::IS_LOSSLESS_ONLY);
}

#[test]
fn u16_sample_constants() {
    assert_eq!(u16::BITS_PER_SAMPLE, 16);
    assert_eq!(u16::MAX_VAL, 65535);
    assert_eq!(u16::CENTER, 32768);
    assert!(u16::IS_LOSSLESS_ONLY);
}

#[test]
fn sample_clamp() {
    assert_eq!(u8::from_i32_clamped(300), 255);
    assert_eq!(u8::from_i32_clamped(-10), 0);
    assert_eq!(u8::from_i32_clamped(128), 128);
    assert_eq!(i16::from_i32_clamped(5000), 4095);
    assert_eq!(i16::from_i32_clamped(-1), 0);
    assert_eq!(u16::from_i32_clamped(70000), 65535);
    assert_eq!(u16::from_i32_clamped(-5), 0);
}
