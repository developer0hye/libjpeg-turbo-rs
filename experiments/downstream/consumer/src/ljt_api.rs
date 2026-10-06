//! The libjpeg-turbo-rs calls the harness makes, written once and expanded
//! for both the published baseline and the candidate.
//!
//! The two crates have identical public APIs at the time of writing, so one
//! macro body serves both and the two rows cannot drift apart in what they
//! ask the library to do. If the candidate ever breaks this API, the harness
//! stops compiling — which is itself the finding a downstream benchmark is
//! for: every application written against 0.8.0 would break the same way.

/// Chroma subsampling for the encode rows, independent of either crate's
/// own `Subsampling` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    /// What thumbnailers and cameras use; the default comparison.
    S420,
    /// What `image`'s built-in encoder always writes, for a like-for-like row.
    S444,
}

/// `$scale` builds the crate's `ScalingFactor` from `(num, denom)`: the
/// published 0.8.0 has an infallible `new`, while the candidate validates
/// through `try_new` (P4-139). The harness only asks for supported factors.
macro_rules! ljt_api {
    ($module:ident, $krate:ident, $scale:expr) => {
        pub mod $module {
            use crate::corpus::OutputLayout;
            use $krate::{Decoder, PixelFormat, ScalingFactor, Subsampling};

            fn pixel_format(layout: OutputLayout) -> PixelFormat {
                match layout {
                    OutputLayout::Rgb => PixelFormat::Rgb,
                    OutputLayout::Gray => PixelFormat::Grayscale,
                }
            }

            fn configured_decoder<'a>(
                jpeg: &'a [u8],
                layout: OutputLayout,
                scale: Option<(u32, u32)>,
            ) -> Decoder<'a> {
                let mut decoder: Decoder<'a> = Decoder::new(jpeg).unwrap_or_else(|error| {
                    panic!(
                        concat!(stringify!($krate), " header parse failed: {}"),
                        error
                    )
                });
                decoder.set_output_format(pixel_format(layout));
                if let Some((numerator, denominator)) = scale {
                    let make_scale: fn(u32, u32) -> ScalingFactor = $scale;
                    decoder.set_scale(make_scale(numerator, denominator));
                }
                decoder
            }

            /// Library-owned output: the one-call API a typical application
            /// reaches for first (`decompress_to`), or the `Decoder` builder
            /// when a scale is requested (there is no one-call scaled API).
            pub fn decode_fresh(
                jpeg: &[u8],
                layout: OutputLayout,
                scale: Option<(u32, u32)>,
            ) -> (usize, usize, Vec<u8>) {
                let image: $krate::Image = match scale {
                    None => $krate::decompress_to(jpeg, pixel_format(layout)),
                    Some(_) => configured_decoder(jpeg, layout, scale).decode_image(),
                }
                .unwrap_or_else(|error| {
                    panic!(concat!(stringify!($krate), " decode failed: {}"), error)
                });
                (image.width, image.height, image.data)
            }

            /// Bytes a caller-owned buffer needs; queried once, outside the
            /// timed loop.
            pub fn output_size(
                jpeg: &[u8],
                layout: OutputLayout,
                scale: Option<(u32, u32)>,
            ) -> usize {
                configured_decoder(jpeg, layout, scale)
                    .output_buffer_size()
                    .unwrap_or_else(|error| {
                        panic!(
                            concat!(stringify!($krate), " output_buffer_size failed: {}"),
                            error
                        )
                    })
            }

            /// Caller-owned output, reused across iterations
            /// (`decompress_into`, or the builder's `decode_image_into` when
            /// scaled). A fresh decoder is still constructed per call: no
            /// decoder-reuse API exists, so "reuse" here means buffer reuse.
            pub fn decode_into(
                jpeg: &[u8],
                layout: OutputLayout,
                scale: Option<(u32, u32)>,
                out: &mut [u8],
            ) -> (usize, usize) {
                let info: $krate::ImageInfo = match scale {
                    None => $krate::decompress_into(jpeg, pixel_format(layout), out),
                    Some(_) => configured_decoder(jpeg, layout, scale).decode_image_into(out),
                }
                .unwrap_or_else(|error| {
                    panic!(
                        concat!(stringify!($krate), " decode_into failed: {}"),
                        error
                    )
                });
                (info.width, info.height)
            }

            /// Decode to RGB and apply the stream's own EXIF orientation —
            /// the first two steps of the thumbnail workload.
            pub fn decode_oriented_rgb(
                jpeg: &[u8],
                scale: Option<(u32, u32)>,
            ) -> (usize, usize, Vec<u8>) {
                let image: $krate::Image = configured_decoder(jpeg, OutputLayout::Rgb, scale)
                    .decode_image()
                    .unwrap_or_else(|error| {
                        panic!(concat!(stringify!($krate), " decode failed: {}"), error)
                    })
                    .apply_orientation();
                (image.width, image.height, image.data)
            }

            /// Baseline (sequential) RGB encode at the given chroma
            /// subsampling.
            pub fn compress_rgb(
                rgb: &[u8],
                width: usize,
                height: usize,
                quality: u8,
                chroma: crate::ljt_api::Chroma,
            ) -> Vec<u8> {
                let subsampling: Subsampling = match chroma {
                    crate::ljt_api::Chroma::S420 => Subsampling::S420,
                    crate::ljt_api::Chroma::S444 => Subsampling::S444,
                };
                $krate::compress(rgb, width, height, PixelFormat::Rgb, quality, subsampling)
                    .unwrap_or_else(|error| {
                        panic!(concat!(stringify!($krate), " encode failed: {}"), error)
                    })
            }
        }
    };
}

ljt_api!(baseline, ljt_baseline, |num: u32, denom: u32| {
    ljt_baseline::ScalingFactor::new(num, denom)
});
ljt_api!(candidate, ljt_candidate, |num: u32, denom: u32| {
    ljt_candidate::ScalingFactor::try_new(num, denom).expect("the harness asks only for supported factors")
});
