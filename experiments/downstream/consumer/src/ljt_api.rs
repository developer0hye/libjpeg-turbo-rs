//! The libjpeg-turbo-rs calls the harness makes, written once and expanded
//! for both the published baseline and the candidate.
//!
//! The two crates have identical public APIs at the time of writing, so one
//! macro body serves both and the two rows cannot drift apart in what they
//! ask the library to do. If the candidate ever breaks this API, the harness
//! stops compiling — which is itself the finding a downstream benchmark is
//! for: every application written against 0.8.0 would break the same way.

macro_rules! ljt_api {
    ($module:ident, $krate:ident) => {
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
                    decoder.set_scale(ScalingFactor::new(numerator, denominator));
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

            /// 4:2:0 baseline encode — the encode comparison's settings.
            pub fn compress_rgb(rgb: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
                $krate::compress(
                    rgb,
                    width,
                    height,
                    PixelFormat::Rgb,
                    quality,
                    Subsampling::S420,
                )
                .unwrap_or_else(|error| {
                    panic!(concat!(stringify!($krate), " encode failed: {}"), error)
                })
            }
        }
    };
}

ljt_api!(baseline, ljt_baseline);
ljt_api!(candidate, ljt_candidate);
