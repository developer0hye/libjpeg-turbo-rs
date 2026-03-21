use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::decompress;

fn assert_streaming_matches_high_level(data: &[u8]) {
    let decoder = StreamingDecoder::new(data).unwrap();
    let streaming = decoder.decode().unwrap();
    let high_level = decompress(data).unwrap();

    assert_eq!(streaming.width, high_level.width);
    assert_eq!(streaming.height, high_level.height);
    assert_eq!(streaming.pixel_format, high_level.pixel_format);
    assert_eq!(streaming.data, high_level.data);
}

#[test]
fn streaming_decoder_decode_matches_high_level_api() {
    let data = include_bytes!("fixtures/gradient_640x480.jpg");
    assert_streaming_matches_high_level(data);
}

#[test]
fn streaming_decoder_decode_matches_high_level_api_422() {
    let data = include_bytes!("fixtures/green_16x16_422.jpg");
    assert_streaming_matches_high_level(data);
}

#[test]
fn streaming_decoder_decode_matches_high_level_api_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    assert_streaming_matches_high_level(data);
}

#[test]
fn streaming_decoder_can_decode_multiple_times() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");

    let decoder = StreamingDecoder::new(data).unwrap();
    let first = decoder.decode().unwrap();
    let second = decoder.decode().unwrap();

    assert_eq!(first.width, second.width);
    assert_eq!(first.height, second.height);
    assert_eq!(first.pixel_format, second.pixel_format);
    assert_eq!(first.data, second.data);
}
