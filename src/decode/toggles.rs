//! Decode toggle features: fast upsample, block smoothing, colorspace override.

use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::types::*;
use crate::decode::pipeline::Image;

/// Generic nearest-neighbor upsampling for any h_factor x v_factor.
#[allow(clippy::too_many_arguments)]
pub fn upsample_nearest(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
    h_factor: usize,
    v_factor: usize,
) {
    for y in 0..in_height {
        for x in 0..in_width {
            let val: u8 = input[y * in_width + x];
            for vy in 0..v_factor {
                for hx in 0..h_factor {
                    output[(y * v_factor + vy) * out_width + x * h_factor + hx] = val;
                }
            }
        }
    }
}

/// Apply inter-block smoothing to a decoded component plane.
pub fn apply_block_smoothing(plane: &mut [u8], pw: usize, ph: usize) {
    // Smooth horizontal block edges
    for y in 0..ph {
        for bx in 1..(pw / 8) {
            let ex: usize = bx * 8;
            if ex >= pw {
                break;
            }
            let li: usize = y * pw + ex - 1;
            let ri: usize = y * pw + ex;
            let av: u8 = ((plane[li] as u16 + plane[ri] as u16 + 1) >> 1) as u8;
            plane[li] = av;
            plane[ri] = av;
        }
    }
    // Smooth vertical block edges
    for by in 1..(ph / 8) {
        let ey: usize = by * 8;
        if ey >= ph {
            break;
        }
        for x in 0..pw {
            let ti: usize = (ey - 1) * pw + x;
            let bi: usize = ey * pw + x;
            let av: u8 = ((plane[ti] as u16 + plane[bi] as u16 + 1) >> 1) as u8;
            plane[ti] = av;
            plane[bi] = av;
        }
    }
}

/// Decode with output colorspace override.
#[allow(clippy::too_many_arguments)]
pub fn decode_with_colorspace_override(
    target_cs: ColorSpace,
    component_planes: &[Vec<u8>],
    frame: &FrameHeader,
    out_width: usize,
    out_height: usize,
    mcus_x: usize,
    block_size: usize,
    icc_profile: Option<Vec<u8>>,
    exif_data: Option<Vec<u8>>,
    comment: Option<String>,
    density: DensityInfo,
    saved_markers: Vec<SavedMarker>,
    warnings: Vec<DecodeWarning>,
) -> Result<Image> {
    match target_cs {
        ColorSpace::Grayscale => {
            let cw: usize = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;
            let mut data: Vec<u8> = Vec::with_capacity(out_width * out_height);
            for y in 0..out_height {
                data.extend_from_slice(&component_planes[0][y * cw..y * cw + out_width]);
            }
            Ok(Image {
                width: out_width,
                height: out_height,
                pixel_format: PixelFormat::Grayscale,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment,
                density,
                saved_markers,
                warnings,
            })
        }
        ColorSpace::YCbCr => {
            if frame.components.len() < 3 {
                return Err(JpegError::Unsupported(
                    "YCbCr output requires 3+ components".into(),
                ));
            }
            let cw: usize = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;
            let cbw: usize = mcus_x * frame.components[1].horizontal_sampling as usize * block_size;
            let hf: usize = frame.components[0].horizontal_sampling as usize
                / frame.components[1].horizontal_sampling as usize;
            let vf: usize = frame.components[0].vertical_sampling as usize
                / frame.components[1].vertical_sampling as usize;
            let mut data: Vec<u8> = Vec::with_capacity(out_width * out_height * 3);
            for y in 0..out_height {
                for x in 0..out_width {
                    data.push(component_planes[0][y * cw + x]);
                    let cx: usize = (x / hf).min(cbw.saturating_sub(1));
                    let cy: usize =
                        (y / vf).min((component_planes[1].len() / cbw).saturating_sub(1));
                    data.push(component_planes[1][cy * cbw + cx]);
                    data.push(component_planes[2][cy * cbw + cx]);
                }
            }
            Ok(Image {
                width: out_width,
                height: out_height,
                pixel_format: PixelFormat::Rgb,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment,
                density,
                saved_markers,
                warnings,
            })
        }
        _ => Err(JpegError::Unsupported(format!(
            "output colorspace {:?} not supported",
            target_cs
        ))),
    }
}
