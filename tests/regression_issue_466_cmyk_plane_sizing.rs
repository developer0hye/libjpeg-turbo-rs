//! Issue #466 (P4-126 criteria 1 and 4): the root-crate YUV plane helpers must
//! reject a component index the subsampling has no plane for, and
//! `decompress_to_yuv_planes` must stop asking them for one.
//!
//! `yuv_plane_width`/`yuv_plane_height` treated *every* `component != 0` as
//! chroma with no upper bound, so `yuv_plane_width(3, ..)` answered with a
//! chroma-sized plane instead of signalling an invalid argument. C rejects it:
//! `tj3YUVPlaneWidth` raises `THROWG("Invalid argument", 0)` for
//! `componentID >= nc` (`references/libjpeg-turbo/src/turbojpeg.c:1123-1125`).
//!
//! The only caller that reached index 3 was `decompress_to_yuv_planes`, on
//! every 4-component frame — and it was getting a *wrong* answer there, not
//! merely an unchecked one. In CMYK and YCCK the fourth component (K) carries
//! full resolution, like luma; sizing it as chroma silently truncated it
//! whenever the frame had subsampled chroma. That is P4-126 criterion 4's
//! decision, resolved here in favour of keeping four planes and sizing the
//! fourth correctly, rather than rejecting CMYK in the public Rust API.

use libjpeg_turbo_rs::api::yuv::decompress_to_yuv_planes;
use libjpeg_turbo_rs::{compress, yuv_plane_height, yuv_plane_width, PixelFormat, Subsampling};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;

/// A 4-component frame whose chroma really is subsampled, so a K plane sized by
/// the chroma rule is measurably smaller than the correct full-resolution one.
/// With 4:4:4 the two rules coincide and the bug is invisible.
fn cmyk_jpeg_420() -> Vec<u8> {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * 4).map(|i| (i % 251) as u8).collect();
    compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        90,
        Subsampling::S420,
    )
    .expect("compress 4:2:0 CMYK fixture")
}

/// P4-126 criterion 1: an index at or above the plane count is an invalid
/// argument, reported as C reports it — 0, the documented "invalid input"
/// return of `tj3YUVPlaneWidth`/`tj3YUVPlaneHeight`.
#[test]
fn plane_helpers_reject_a_component_index_past_the_plane_count() {
    for ss in [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
    ] {
        for comp in [3usize, 4, 9] {
            assert_eq!(
                yuv_plane_width(comp, WIDTH, ss),
                0,
                "yuv_plane_width({comp}, .., {ss:?}) must report an invalid component index"
            );
            assert_eq!(
                yuv_plane_height(comp, HEIGHT, ss),
                0,
                "yuv_plane_height({comp}, .., {ss:?}) must report an invalid component index"
            );
        }
        // The valid indices keep their meaning — this must not resize anything.
        assert_eq!(yuv_plane_width(0, WIDTH, ss), yuv_plane_width(0, WIDTH, ss));
        assert!(yuv_plane_width(0, WIDTH, ss) > 0, "luma stays valid");
        assert!(yuv_plane_width(2, WIDTH, ss) > 0, "chroma stays valid");
    }
}

/// P4-126 criterion 4: a 4-component frame still yields four planes, and the
/// fourth is full resolution. Sized by the old chroma rule it came back at half
/// width and half height for a 4:2:0 frame, silently dropping three quarters of
/// the K channel.
#[test]
fn cmyk_fourth_plane_is_full_resolution_not_chroma_sized() {
    let jpeg: Vec<u8> = cmyk_jpeg_420();
    let (planes, width, height, subsampling): (Vec<Vec<u8>>, usize, usize, Subsampling) =
        decompress_to_yuv_planes(&jpeg).expect("decompress 4:2:0 CMYK to planes");

    assert_eq!(planes.len(), 4, "one plane per SOF component");
    assert_eq!((width, height), (WIDTH, HEIGHT));

    let luma_w: usize = yuv_plane_width(0, width, subsampling);
    let luma_h: usize = yuv_plane_height(0, height, subsampling);
    let chroma_w: usize = yuv_plane_width(1, width, subsampling);
    assert!(
        chroma_w < luma_w,
        "fixture must actually be subsampled or this test proves nothing \
         (luma {luma_w}, chroma {chroma_w})"
    );

    assert_eq!(
        planes[3].len(),
        luma_w * luma_h,
        "the K plane carries full resolution in CMYK/YCCK; sizing it as chroma \
         truncates it (got {} bytes, luma plane is {})",
        planes[3].len(),
        luma_w * luma_h
    );
}
