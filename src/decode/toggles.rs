//! Decode toggle features: fast upsample, block smoothing, colorspace override.

use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::quant_table::QuantTable;
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

// ============================================================================
// Coefficient-level block smoothing (matching C libjpeg-turbo's
// decompress_smooth_data / smoothing_ok from jdcoefct.c)
// ============================================================================

/// Number of saved coefficient precision entries per component.
/// Corresponds to C's `SAVED_COEFS` (DC + first 9 AC in zigzag order).
const SAVED_COEFS: usize = 10;

/// Quantization table positions (natural order) for the 10 coefficients.
/// Q00=0, Q01=1, Q10=8, Q20=16, Q11=9, Q02=2, Q03=3, Q12=10, Q21=17, Q30=24
const Q00_POS: usize = 0;
const Q01_POS: usize = 1;
const Q10_POS: usize = 8;
const Q20_POS: usize = 16;
const Q11_POS: usize = 9;
const Q02_POS: usize = 2;
const Q03_POS: usize = 3;
const Q12_POS: usize = 10;
const Q21_POS: usize = 17;
const Q30_POS: usize = 24;

/// Compute per-component coefficient precision from progressive scan headers.
///
/// Returns a `Vec<[i32; SAVED_COEFS]>` where entry `[ci][k]` is:
///   - `-1`: coefficient k of component ci has never been seen in any scan
///   - `0`: coefficient k is fully accurate (successive approximation complete)
///   - `>0`: the Al (successive approximation low bit) value from the last scan
///     that refined this coefficient
///
/// This mimics C libjpeg-turbo's `coef_bits` array, which is built up by
/// `start_pass_phuff_decoder` in `jdphuff.c`.
pub fn compute_coef_bits(
    scans: &[crate::decode::marker::ScanInfo],
    frame: &FrameHeader,
) -> Vec<[i32; SAVED_COEFS]> {
    let num_components: usize = frame.components.len();
    // Initialize all coefficients to -1 (not yet seen), matching C's jinit_phuff_decoder.
    let mut coef_bits: Vec<[i32; 64]> = vec![[-1i32; 64]; num_components];

    for scan_info in scans {
        let scan: &ScanHeader = &scan_info.header;
        let ss: u8 = scan.spec_start;
        let se: u8 = scan.spec_end;
        let al: u8 = scan.succ_low;

        for scan_comp in &scan.components {
            let comp_idx: Option<usize> = frame
                .components
                .iter()
                .position(|fc| fc.id == scan_comp.component_id);
            let comp_idx: usize = match comp_idx {
                Some(ci) => ci,
                None => continue,
            };

            // Update coef_bits for the spectral range in this scan
            for entry in coef_bits[comp_idx]
                .iter_mut()
                .take(se as usize + 1)
                .skip(ss as usize)
            {
                *entry = al as i32;
            }
        }
    }

    // Extract just the SAVED_COEFS (zigzag indices 0..9) for each component.
    // coef_bits is indexed by zigzag position, which is what we want.
    let mut result: Vec<[i32; SAVED_COEFS]> = Vec::with_capacity(num_components);
    for comp_bits in coef_bits.iter().take(num_components) {
        let mut saved: [i32; SAVED_COEFS] = [-1i32; SAVED_COEFS];
        saved[..SAVED_COEFS].copy_from_slice(&comp_bits[..SAVED_COEFS]);
        result.push(saved);
    }
    result
}

/// Check if block smoothing should be applied for a given component.
///
/// Returns `true` if smoothing is useful (some AC coefficients are imprecise)
/// and the required quantization table entries are nonzero.
/// Matches C's `smoothing_ok()` logic.
pub fn smoothing_ok_for_component(coef_bits: &[i32; SAVED_COEFS], quant: &QuantTable) -> bool {
    // DC values must be at least partly known
    if coef_bits[0] < 0 {
        return false;
    }

    // Verify DC & first 9 AC quantizers are nonzero to avoid zero-divide
    if quant.values[Q00_POS] == 0
        || quant.values[Q01_POS] == 0
        || quant.values[Q10_POS] == 0
        || quant.values[Q20_POS] == 0
        || quant.values[Q11_POS] == 0
        || quant.values[Q02_POS] == 0
        || quant.values[Q03_POS] == 0
        || quant.values[Q12_POS] == 0
        || quant.values[Q21_POS] == 0
        || quant.values[Q30_POS] == 0
    {
        return false;
    }

    // Block smoothing is useful if some AC coefficients remain inaccurate
    coef_bits
        .iter()
        .skip(1)
        .take(SAVED_COEFS - 1)
        .any(|&b| b != 0)
}

/// Apply coefficient-level block smoothing to one component's coefficient buffer.
///
/// This is a port of C libjpeg-turbo's `decompress_smooth_data()` from jdcoefct.c.
/// It modifies the DCT coefficient blocks in-place, using a 5x5 DC neighbor window
/// to predict imprecise AC coefficients before IDCT.
///
/// Parameters:
///   - `coeff_buf`: coefficient blocks in natural (row-major) order, `blocks_x * blocks_y` blocks
///   - `blocks_x`: number of blocks horizontally
///   - `blocks_y`: number of blocks vertically
///   - `coef_bits`: per-coefficient precision for this component (SAVED_COEFS entries)
///   - `quant`: quantization table for this component
#[allow(clippy::too_many_lines)]
pub fn apply_block_smoothing_coeffs(
    coeff_buf: &mut [[i16; 64]],
    blocks_x: usize,
    blocks_y: usize,
    coef_bits: &[i32; SAVED_COEFS],
    quant: &QuantTable,
) {
    // Determine if we do DC interpolation (all AC bits unknown)
    let change_dc: bool = coef_bits[1] == -1
        && coef_bits[2] == -1
        && coef_bits[3] == -1
        && coef_bits[4] == -1
        && coef_bits[5] == -1
        && coef_bits[6] == -1
        && coef_bits[7] == -1
        && coef_bits[8] == -1
        && coef_bits[9] == -1;

    let q00: i64 = quant.values[Q00_POS] as i64;
    let q01: i64 = quant.values[Q01_POS] as i64;
    let q10: i64 = quant.values[Q10_POS] as i64;
    let q20: i64 = quant.values[Q20_POS] as i64;
    let q11: i64 = quant.values[Q11_POS] as i64;
    let q02: i64 = quant.values[Q02_POS] as i64;
    let q03: i64 = if change_dc {
        quant.values[Q03_POS] as i64
    } else {
        0
    };
    let q12: i64 = if change_dc {
        quant.values[Q12_POS] as i64
    } else {
        0
    };
    let q21: i64 = if change_dc {
        quant.values[Q21_POS] as i64
    } else {
        0
    };
    let q30: i64 = if change_dc {
        quant.values[Q30_POS] as i64
    } else {
        0
    };

    /// Helper: get DC value (natural position 0) from a block.
    #[inline(always)]
    fn dc_val(coeff_buf: &[[i16; 64]], blocks_x: usize, row: usize, col: usize) -> i32 {
        coeff_buf[row * blocks_x + col][0] as i32
    }

    /// Helper: compute prediction value with rounding and Al clamping.
    #[inline(always)]
    fn compute_pred(num: i64, q_coef: i64, al: i32) -> i16 {
        let pred: i32 = if num >= 0 {
            let mut p: i32 = (((q_coef << 7) + num) / (q_coef << 8)) as i32;
            if al > 0 && p >= (1 << al) {
                p = (1 << al) - 1;
            }
            p
        } else {
            let mut p: i32 = (((q_coef << 7) - num) / (q_coef << 8)) as i32;
            if al > 0 && p >= (1 << al) {
                p = (1 << al) - 1;
            }
            -p
        };
        pred as i16
    }

    // We need to work on a copy of each block to avoid reading modified neighbors.
    // Process row by row. The C code uses a workspace copy per block.
    for by in 0..blocks_y {
        // Determine neighboring row indices (clamped to image bounds)
        let row_pp: usize = if by >= 2 {
            by - 2
        } else if by >= 1 {
            by - 1
        } else {
            by
        };
        let row_p: usize = if by >= 1 { by - 1 } else { by };
        let row_n: usize = if by + 1 < blocks_y { by + 1 } else { by };
        let row_nn: usize = if by + 2 < blocks_y {
            by + 2
        } else if by + 1 < blocks_y {
            by + 1
        } else {
            by
        };

        for bx in 0..blocks_x {
            // Copy current block to workspace
            let block_idx: usize = by * blocks_x + bx;
            let mut workspace: [i16; 64] = coeff_buf[block_idx];

            // Column indices for the 5-wide DC window (clamped)
            let col_pp: usize = if bx >= 2 {
                bx - 2
            } else if bx >= 1 {
                bx - 1
            } else {
                bx
            };
            let col_p: usize = if bx >= 1 { bx - 1 } else { bx };
            let col_n: usize = if bx + 1 < blocks_x { bx + 1 } else { bx };
            let col_nn: usize = if bx + 2 < blocks_x {
                bx + 2
            } else if bx + 1 < blocks_x {
                bx + 1
            } else {
                bx
            };

            // Gather 25 DC values from the 5x5 neighborhood
            let dc01: i32 = dc_val(coeff_buf, blocks_x, row_pp, col_pp);
            let dc02: i32 = dc_val(coeff_buf, blocks_x, row_pp, col_p);
            let dc03: i32 = dc_val(coeff_buf, blocks_x, row_pp, bx);
            let dc04: i32 = dc_val(coeff_buf, blocks_x, row_pp, col_n);
            let dc05: i32 = dc_val(coeff_buf, blocks_x, row_pp, col_nn);

            let dc06: i32 = dc_val(coeff_buf, blocks_x, row_p, col_pp);
            let dc07: i32 = dc_val(coeff_buf, blocks_x, row_p, col_p);
            let dc08: i32 = dc_val(coeff_buf, blocks_x, row_p, bx);
            let dc09: i32 = dc_val(coeff_buf, blocks_x, row_p, col_n);
            let dc10: i32 = dc_val(coeff_buf, blocks_x, row_p, col_nn);

            let dc11: i32 = dc_val(coeff_buf, blocks_x, by, col_pp);
            let dc12: i32 = dc_val(coeff_buf, blocks_x, by, col_p);
            let dc13: i32 = dc_val(coeff_buf, blocks_x, by, bx);
            let dc14: i32 = dc_val(coeff_buf, blocks_x, by, col_n);
            let dc15: i32 = dc_val(coeff_buf, blocks_x, by, col_nn);

            let dc16: i32 = dc_val(coeff_buf, blocks_x, row_n, col_pp);
            let dc17: i32 = dc_val(coeff_buf, blocks_x, row_n, col_p);
            let dc18: i32 = dc_val(coeff_buf, blocks_x, row_n, bx);
            let dc19: i32 = dc_val(coeff_buf, blocks_x, row_n, col_n);
            let dc20: i32 = dc_val(coeff_buf, blocks_x, row_n, col_nn);

            let dc21: i32 = dc_val(coeff_buf, blocks_x, row_nn, col_pp);
            let dc22: i32 = dc_val(coeff_buf, blocks_x, row_nn, col_p);
            let dc23: i32 = dc_val(coeff_buf, blocks_x, row_nn, bx);
            let dc24: i32 = dc_val(coeff_buf, blocks_x, row_nn, col_n);
            let dc25: i32 = dc_val(coeff_buf, blocks_x, row_nn, col_nn);

            // AC01 (natural position 1)
            let al: i32 = coef_bits[1];
            if al != 0 && workspace[1] == 0 {
                let num: i64 = q00
                    * if change_dc {
                        (-dc01 - dc02 + dc04 + dc05 - 3 * dc06 + 13 * dc07 - 13 * dc09 + 3 * dc10
                            - 3 * dc11
                            + 38 * dc12
                            - 38 * dc14
                            + 3 * dc15
                            - 3 * dc16
                            + 13 * dc17
                            - 13 * dc19
                            + 3 * dc20
                            - dc21
                            - dc22
                            + dc24
                            + dc25) as i64
                    } else {
                        (-7 * dc11 + 50 * dc12 - 50 * dc14 + 7 * dc15) as i64
                    };
                workspace[1] = compute_pred(num, q01, al);
            }

            // AC10 (natural position 8)
            let al: i32 = coef_bits[2];
            if al != 0 && workspace[8] == 0 {
                let num: i64 = q00
                    * if change_dc {
                        (-dc01 - 3 * dc02 - 3 * dc03 - 3 * dc04 - dc05 - dc06
                            + 13 * dc07
                            + 38 * dc08
                            + 13 * dc09
                            - dc10
                            + dc16
                            - 13 * dc17
                            - 38 * dc18
                            - 13 * dc19
                            + dc20
                            + dc21
                            + 3 * dc22
                            + 3 * dc23
                            + 3 * dc24
                            + dc25) as i64
                    } else {
                        (-7 * dc03 + 50 * dc08 - 50 * dc18 + 7 * dc23) as i64
                    };
                workspace[8] = compute_pred(num, q10, al);
            }

            // AC20 (natural position 16)
            let al: i32 = coef_bits[3];
            if al != 0 && workspace[16] == 0 {
                let num: i64 = q00
                    * if change_dc {
                        (dc03 + 2 * dc07 + 7 * dc08 + 2 * dc09 - 5 * dc12 - 14 * dc13 - 5 * dc14
                            + 2 * dc17
                            + 7 * dc18
                            + 2 * dc19
                            + dc23) as i64
                    } else {
                        (-dc03 + 13 * dc08 - 24 * dc13 + 13 * dc18 - dc23) as i64
                    };
                workspace[16] = compute_pred(num, q20, al);
            }

            // AC11 (natural position 9)
            let al: i32 = coef_bits[4];
            if al != 0 && workspace[9] == 0 {
                let num: i64 = q00
                    * if change_dc {
                        (-dc01 + dc05 + 9 * dc07 - 9 * dc09 - 9 * dc17 + 9 * dc19 + dc21 - dc25)
                            as i64
                    } else {
                        (dc10 + dc16 - 10 * dc17 + 10 * dc19 - dc02 - dc20 + dc22 - dc24 + dc04
                            - dc06
                            + 10 * dc07
                            - 10 * dc09) as i64
                    };
                workspace[9] = compute_pred(num, q11, al);
            }

            // AC02 (natural position 2)
            let al: i32 = coef_bits[5];
            if al != 0 && workspace[2] == 0 {
                let num: i64 = q00
                    * if change_dc {
                        (2 * dc07 - 5 * dc08 + 2 * dc09 + dc11 + 7 * dc12 - 14 * dc13
                            + 7 * dc14
                            + dc15
                            + 2 * dc17
                            - 5 * dc18
                            + 2 * dc19) as i64
                    } else {
                        (-dc11 + 13 * dc12 - 24 * dc13 + 13 * dc14 - dc15) as i64
                    };
                workspace[2] = compute_pred(num, q02, al);
            }

            if change_dc {
                // AC03 (natural position 3)
                let al: i32 = coef_bits[6];
                if al != 0 && workspace[3] == 0 {
                    let num: i64 = q00 * (dc07 - dc09 + 2 * dc12 - 2 * dc14 + dc17 - dc19) as i64;
                    workspace[3] = compute_pred(num, q03, al);
                }

                // AC12 (natural position 10)
                let al: i32 = coef_bits[7];
                if al != 0 && workspace[10] == 0 {
                    let num: i64 = q00 * (dc07 - 3 * dc08 + dc09 - dc17 + 3 * dc18 - dc19) as i64;
                    workspace[10] = compute_pred(num, q12, al);
                }

                // AC21 (natural position 17)
                let al: i32 = coef_bits[8];
                if al != 0 && workspace[17] == 0 {
                    let num: i64 = q00 * (dc07 - dc09 - 3 * dc12 + 3 * dc14 + dc17 - dc19) as i64;
                    workspace[17] = compute_pred(num, q21, al);
                }

                // AC30 (natural position 24)
                let al: i32 = coef_bits[9];
                if al != 0 && workspace[24] == 0 {
                    let num: i64 = q00 * (dc07 + 2 * dc08 + dc09 - dc17 - 2 * dc18 - dc19) as i64;
                    workspace[24] = compute_pred(num, q30, al);
                }

                // DC interpolation using Gaussian-like kernel
                let num: i64 = q00
                    * (-2 * dc01 - 6 * dc02 - 8 * dc03 - 6 * dc04 - 2 * dc05 - 6 * dc06
                        + 6 * dc07
                        + 42 * dc08
                        + 6 * dc09
                        - 6 * dc10
                        - 8 * dc11
                        + 42 * dc12
                        + 152 * dc13
                        + 42 * dc14
                        - 8 * dc15
                        - 6 * dc16
                        + 6 * dc17
                        + 42 * dc18
                        + 6 * dc19
                        - 6 * dc20
                        - 2 * dc21
                        - 6 * dc22
                        - 8 * dc23
                        - 6 * dc24
                        - 2 * dc25) as i64;
                let pred: i32 = if num >= 0 {
                    (((q00 << 7) + num) / (q00 << 8)) as i32
                } else {
                    -(((q00 << 7) - num) / (q00 << 8)) as i32
                };
                workspace[0] = pred as i16;
            }

            // Write back the modified workspace
            coeff_buf[block_idx] = workspace;
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
    comp_block_sizes: &[usize],
    icc_profile: Option<Vec<u8>>,
    exif_data: Option<Vec<u8>>,
    comment: Option<String>,
    density: DensityInfo,
    saved_markers: Vec<SavedMarker>,
    warnings: Vec<DecodeWarning>,
) -> Result<Image> {
    match target_cs {
        ColorSpace::Grayscale => {
            let cw: usize =
                mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];
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
            let cw: usize =
                mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];
            let cbw: usize =
                mcus_x * frame.components[1].horizontal_sampling as usize * comp_block_sizes[1];
            // Effective upsample factors considering per-component IDCT sizes
            let hf: usize = cw / cbw;
            let vf: usize = (frame.components[0].vertical_sampling as usize * comp_block_sizes[0])
                / (frame.components[1].vertical_sampling as usize * comp_block_sizes[1]);
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
