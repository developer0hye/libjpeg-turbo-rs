//! The items the P4-222 public-surface review promoted to supported root
//! paths (#635 Milestone E). Pinned here so the later narrowing of the
//! low-level modules (`api`, `common`, `decode`, `encode`, `simd`,
//! `transform`) cannot remove the supported way to reach them.

use libjpeg_turbo_rs::yuv;
use libjpeg_turbo_rs::{
    compress_with_params, CompressParams, PixelFormat, StreamingDecoder, Subsampling,
};

/// The root paths resolve and behave like the module paths they re-export:
/// encoding through either produces identical bytes.
#[test]
fn promoted_items_resolve_at_the_root() {
    let rgb: Vec<u8> = (0..(16 * 16 * 3)).map(|i: usize| (i % 251) as u8).collect();
    let jpeg: Vec<u8> = compress_with_params(&CompressParams::new(
        &rgb,
        16,
        16,
        PixelFormat::Rgb,
        85,
        Subsampling::S420,
    ))
    .expect("baseline encode through the root path");
    let via_module: Vec<u8> = libjpeg_turbo_rs::encode::pipeline::compress_with_params(
        &libjpeg_turbo_rs::encode::pipeline::CompressParams::new(
            &rgb,
            16,
            16,
            PixelFormat::Rgb,
            85,
            Subsampling::S420,
        ),
    )
    .expect("baseline encode through the module path");
    assert_eq!(jpeg, via_module);

    let (planes, width, height, subsampling): (Vec<Vec<u8>>, usize, usize, Subsampling) =
        yuv::decompress_to_yuv_planes(&jpeg).expect("YUV planes through the root path");
    assert_eq!((width, height, subsampling), (16, 16, Subsampling::S420));
    assert_eq!(planes.len(), 3);

    let streaming: StreamingDecoder<'_> =
        StreamingDecoder::new(&jpeg).expect("streaming decoder through the root path");
    drop(streaming);
}
