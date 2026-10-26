//! Decode toggle features: fast upsample, block smoothing, colorspace override.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::pipeline::Image;
#[allow(unused_imports)]
use alloc::{format, vec};
#[allow(unused_imports)]
use alloc::{string::String, vec::Vec};

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

/// The coefficient bits each component had before the last scan that coded
/// it, as C's `smoothing_ok()` latches them for block smoothing
/// (`prev_coef_bits_latch`, jdcoefct.c:451-456). Block smoothing uses these
/// instead of [`compute_coef_bits`] for the iMCU rows past the last one the
/// final scan decoded with data, which a truncated stream leaves only partly
/// coded (jdcoefct.c:556-563; P4-255).
///
/// `start_pass_phuff_decoder` (jdphuff.c:132-137) saves entries
/// `min(Ss, 1)..=max(Se, 9)` before each scan updates its band, and saves
/// 0 for the first scan; the latch is -1 throughout when the stream has a
/// single scan. Only entries 1..=9 are latched, and only those are read;
/// entry 0 is left at -1.
pub(crate) fn compute_prev_coef_bits(
    scans: &[crate::decode::marker::ScanInfo],
    frame: &FrameHeader,
) -> Vec<[i32; SAVED_COEFS]> {
    let num_components: usize = frame.components.len();
    let mut coef_bits: Vec<[i32; 64]> = vec![[-1i32; 64]; num_components];
    let mut prev_coef_bits: Vec<[i32; 64]> = vec![[-1i32; 64]; num_components];

    for (scan_index, scan_info) in scans.iter().enumerate() {
        let scan: &ScanHeader = &scan_info.header;
        let ss: usize = scan.spec_start as usize;
        let se: usize = scan.spec_end as usize;
        for scan_comp in &scan.components {
            let Some(comp_idx) = frame
                .components
                .iter()
                .position(|fc| fc.id == scan_comp.component_id)
            else {
                continue;
            };
            for coefi in ss.min(1)..=se.clamp(9, 63) {
                prev_coef_bits[comp_idx][coefi] = if scan_index > 0 {
                    coef_bits[comp_idx][coefi]
                } else {
                    0
                };
            }
            for entry in coef_bits[comp_idx].iter_mut().take(se + 1).skip(ss) {
                *entry = i32::from(scan.succ_low);
            }
        }
    }

    prev_coef_bits
        .iter()
        .map(|prev| {
            let mut latch: [i32; SAVED_COEFS] = [-1i32; SAVED_COEFS];
            if scans.len() > 1 {
                latch[1..].copy_from_slice(&prev[1..SAVED_COEFS]);
            }
            latch
        })
        .collect()
}

/// Check the per-component prerequisites C's `smoothing_ok()` requires before
/// any plane is smoothed.
///
/// C iterates over every component and returns `FALSE` for the entire image
/// as soon as one fails these checks:
///   - quantization table is allocated (we assume non-null in Rust);
///   - first 10 zigzag quant entries are nonzero (avoids `(Q << 7) / (Q << 8)`
///     dividing by zero in the prediction formula);
///   - DC has been seen at least once (`coef_bits[0] >= 0`).
///
/// Note that C's `smoothing_useful` flag (any `coef_bits[1..9] != 0`) is
/// **OR-folded across all components**, not per-component, so it is evaluated
/// at the call site rather than here.
pub fn smoothing_prerequisites_ok_for_component(
    coef_bits: &[i32; SAVED_COEFS],
    quant: &QuantTable,
) -> bool {
    // DC values must be at least partly known.
    if coef_bits[0] < 0 {
        return false;
    }

    // Verify DC & first 9 AC quantizers are nonzero to avoid zero-divide.
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

    true
}

/// Returns true if smoothing has anything to predict for this component,
/// i.e. at least one of `coef_bits[1..SAVED_COEFS]` is nonzero (either
/// `-1` for "never seen" or a positive Al for "partially refined").
///
/// C OR-folds this across all components into a single image-wide
/// `smoothing_useful` flag — see the call site in
/// `decode_progressive_planes`.
pub fn smoothing_useful_for_component(coef_bits: &[i32; SAVED_COEFS]) -> bool {
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
///   - `coeff_buf`: coefficient blocks in natural (row-major) order, laid out
///     `row_stride` blocks per row (the iMCU-padded grid width)
///   - `row_stride`: buffer row pitch in blocks (`mcus_x * h_samp`)
///   - `blocks_x`: number of *real* blocks horizontally (`width_in_blocks`)
///   - `blocks_y`: number of *real* blocks vertically (`height_in_blocks`)
///   - `v_samp`: vertical sampling factor (block rows per iMCU row)
///   - `coef_bits`: per-coefficient precision for this component (SAVED_COEFS entries)
///   - `quant`: quantization table for this component
///
/// Only the real `blocks_x * blocks_y` grid is smoothed, and the 5x5 DC
/// neighbor window selection replicates C's per-iMCU-row processing
/// (fuzz smoke run 28921468958, P4-29). C clamps its column window at
/// `width_in_blocks - 1`, so dummy padding columns are never read; its
/// row window, however, is driven by the scaled per-iMCU-row indices
///
/// ```c
/// image_block_row  = output_iMCU_row * block_rows + block_row;
/// image_block_rows = block_rows * total_iMCU_rows;
/// ```
///
/// where `block_rows` is `v_samp` except on the last iMCU row (partial
/// height). Two consequences faithfully reproduced here: on non-final
/// iMCU rows the `next-next` row can reach *into* the trailing dummy
/// rows (whose DC values interleaved scans decode for real), and on a
/// partial last iMCU row the scaled indices clamp `prev`/`prev-prev`
/// earlier than plain image-row arithmetic would. A naive clamp at the
/// real grid edge matches C only when `total_iMCU_rows == 1`; the padded
/// whole-grid clamp used before this port smoothed dummy rows and read
/// them where C does not, producing phantom AC predictions in the last
/// real block row.
///
/// Neighbor DC values are read from a snapshot taken before any block is
/// modified: C runs IDCT straight off the smoothed workspace and never
/// writes it back, so its neighbor reads always see original DC values.
/// Without the snapshot the `change_dc` path (DC interpolation) would feed
/// already-smoothed DC values into later blocks' predictions.
pub fn apply_block_smoothing_coeffs(
    coeff_buf: &mut [[i16; 64]],
    row_stride: usize,
    blocks_x: usize,
    blocks_y: usize,
    v_samp: usize,
    coef_bits: &[i32; SAVED_COEFS],
    quant: &QuantTable,
) {
    // This public signature returns nothing, so the snapshot cannot report a
    // refused allocation and is reserved infallibly here, as it always was.
    // The decode pipeline calls `try_apply_block_smoothing_coeffs` instead
    // (P4-209): there the size comes from a hostile stream's block grid.
    let mut dc_snapshot: Vec<i32> =
        Vec::with_capacity(dc_snapshot_len(coeff_buf.len(), row_stride, blocks_x));
    smooth_with_dc_snapshot(
        coeff_buf,
        row_stride,
        blocks_x,
        blocks_y,
        v_samp,
        coef_bits,
        coef_bits,
        usize::MAX,
        quant,
        &mut dc_snapshot,
    );
}

/// [`apply_block_smoothing_coeffs`] with the DC snapshot reserved through
/// `common::try_alloc`, so an allocator refusal is
/// [`JpegError::AllocationFailed`] rather than an abort (P4-209). The
/// snapshot holds one `i32` per block of the component's padded grid.
///
/// iMCU rows after `last_good_imcu_row` are smoothed with `prev_coef_bits`
/// ([`compute_prev_coef_bits`]): the final scan ran short before reaching
/// them, so they hold only what the scans before it coded (P4-255). Pass
/// `usize::MAX` when every row was decoded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_apply_block_smoothing_coeffs(
    coeff_buf: &mut [[i16; 64]],
    row_stride: usize,
    blocks_x: usize,
    blocks_y: usize,
    v_samp: usize,
    coef_bits: &[i32; SAVED_COEFS],
    prev_coef_bits: &[i32; SAVED_COEFS],
    last_good_imcu_row: usize,
    quant: &QuantTable,
) -> Result<()> {
    let mut dc_snapshot: Vec<i32> = crate::common::try_alloc::try_with_capacity(
        dc_snapshot_len(coeff_buf.len(), row_stride, blocks_x),
        "block smoothing DC snapshot",
    )?;
    smooth_with_dc_snapshot(
        coeff_buf,
        row_stride,
        blocks_x,
        blocks_y,
        v_samp,
        coef_bits,
        prev_coef_bits,
        last_good_imcu_row,
        quant,
        &mut dc_snapshot,
    );
    Ok(())
}

/// Entries `smooth_with_dc_snapshot` pushes: one per real column of every
/// padded block row. Zero when `row_stride` or `blocks_x` is zero; the other
/// degenerate grids it returns early for (`blocks_y`, `v_samp` or the iMCU
/// row count zero) still reserve, which costs a small allocation on input
/// that is never smoothed, never a short one.
fn dc_snapshot_len(coeff_len: usize, row_stride: usize, blocks_x: usize) -> usize {
    coeff_len
        .checked_div(row_stride)
        .map_or(0, |padded_rows: usize| blocks_x * padded_rows)
}

/// The smoothing pass itself. `dc_snapshot` arrives empty with capacity for
/// [`dc_snapshot_len`] entries, so filling it never reallocates — which is
/// what lets the fallible caller's reservation be the only allocation.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn smooth_with_dc_snapshot(
    coeff_buf: &mut [[i16; 64]],
    row_stride: usize,
    blocks_x: usize,
    blocks_y: usize,
    v_samp: usize,
    current_coef_bits: &[i32; SAVED_COEFS],
    prev_coef_bits: &[i32; SAVED_COEFS],
    last_good_imcu_row: usize,
    quant: &QuantTable,
    dc_snapshot: &mut Vec<i32>,
) {
    let q00: i64 = quant.values[Q00_POS] as i64;
    let q01: i64 = quant.values[Q01_POS] as i64;
    let q10: i64 = quant.values[Q10_POS] as i64;
    let q20: i64 = quant.values[Q20_POS] as i64;
    let q11: i64 = quant.values[Q11_POS] as i64;
    let q02: i64 = quant.values[Q02_POS] as i64;

    // C picks the coefficient bits per iMCU row: those latched from the
    // previous scan for a row past `last_good_iMCU_row` (the final scan ran
    // short before it), the current ones otherwise (jdcoefct.c:556-563).
    // DC interpolation and the extra quantizers follow from that choice.
    let row_params = |coef_bits: &[i32; SAVED_COEFS]| -> (bool, i64, i64, i64, i64) {
        let change_dc: bool = coef_bits[1] == -1
            && coef_bits[2] == -1
            && coef_bits[3] == -1
            && coef_bits[4] == -1
            && coef_bits[5] == -1
            && coef_bits[6] == -1
            && coef_bits[7] == -1
            && coef_bits[8] == -1
            && coef_bits[9] == -1;
        if change_dc {
            (
                true,
                quant.values[Q03_POS] as i64,
                quant.values[Q12_POS] as i64,
                quant.values[Q21_POS] as i64,
                quant.values[Q30_POS] as i64,
            )
        } else {
            (false, 0, 0, 0, 0)
        }
    };
    let current_params = row_params(current_coef_bits);
    let prev_params = row_params(prev_coef_bits);

    // Nothing to smooth on a degenerate grid (e.g. a malformed SOF with
    // zero height reaches here with an empty coefficient buffer); bail
    // out before the iMCU geometry below underflows.
    if blocks_x == 0 || blocks_y == 0 || row_stride == 0 || v_samp == 0 {
        return;
    }

    // The padded grid height and the iMCU-row geometry drive C's row
    // window selection below. All real rows are contiguous at the top of
    // the buffer; dummy rows (if any) sit at the bottom of the last iMCU
    // row and are readable as `next-next` neighbors but never smoothed.
    let padded_rows: usize = coeff_buf.len() / row_stride;
    let total_imcu_rows: usize = padded_rows / v_samp;
    if total_imcu_rows == 0 {
        return;
    }
    let last_imcu_row: usize = total_imcu_rows - 1;

    // Snapshot the DC values (padded rows included — C reads dummy-row
    // DCs as neighbors in the cases described above) so neighbor reads
    // are unaffected by blocks already smoothed this pass (C never writes
    // the smoothed workspace back to the buffer it reads DC neighbors
    // from).
    debug_assert!(
        dc_snapshot.is_empty() && dc_snapshot.capacity() >= blocks_x * padded_rows,
        "caller must reserve dc_snapshot_len() entries"
    );
    for by in 0..padded_rows {
        for bx in 0..blocks_x {
            dc_snapshot.push(coeff_buf[by * row_stride + bx][0] as i32);
        }
    }
    // Read-only from here on, as the local `Vec` it replaced was.
    let dc_snapshot: &[i32] = dc_snapshot.as_slice();

    /// Helper: get an original (pre-smoothing) DC value from the snapshot.
    #[inline(always)]
    fn dc_val(dc_snapshot: &[i32], blocks_x: usize, row: usize, col: usize) -> i32 {
        dc_snapshot[row * blocks_x + col]
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
        // Determine neighboring row indices exactly the way C's
        // decompress_smooth_data does: per-iMCU-row `block_rows` (partial
        // on the last iMCU row) feeds the scaled image_block_row /
        // image_block_rows guards, while the actual neighbor offsets are
        // plain +/-1/+/-2 rows in the padded buffer.
        let imcu_row: usize = by / v_samp;
        let block_row: usize = by % v_samp;
        let (coef_bits, (change_dc, q03, q12, q21, q30)) = if imcu_row > last_good_imcu_row {
            (prev_coef_bits, prev_params)
        } else {
            (current_coef_bits, current_params)
        };
        let block_rows: usize = if imcu_row < last_imcu_row {
            v_samp
        } else {
            blocks_y - last_imcu_row * v_samp
        };
        let image_block_row: usize = imcu_row * block_rows + block_row;
        let image_block_rows: usize = block_rows * total_imcu_rows;

        let row_p: usize = if image_block_row > 0 { by - 1 } else { by };
        let row_pp: usize = if image_block_row > 1 { by - 2 } else { row_p };
        let row_n: usize = if image_block_row + 1 < image_block_rows {
            by + 1
        } else {
            by
        };
        let row_nn: usize = if image_block_row + 2 < image_block_rows {
            by + 2
        } else {
            row_n
        };

        for bx in 0..blocks_x {
            // Copy current block to workspace
            let block_idx: usize = by * row_stride + bx;
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
            let dc01: i32 = dc_val(dc_snapshot, blocks_x, row_pp, col_pp);
            let dc02: i32 = dc_val(dc_snapshot, blocks_x, row_pp, col_p);
            let dc03: i32 = dc_val(dc_snapshot, blocks_x, row_pp, bx);
            let dc04: i32 = dc_val(dc_snapshot, blocks_x, row_pp, col_n);
            let dc05: i32 = dc_val(dc_snapshot, blocks_x, row_pp, col_nn);

            let dc06: i32 = dc_val(dc_snapshot, blocks_x, row_p, col_pp);
            let dc07: i32 = dc_val(dc_snapshot, blocks_x, row_p, col_p);
            let dc08: i32 = dc_val(dc_snapshot, blocks_x, row_p, bx);
            let dc09: i32 = dc_val(dc_snapshot, blocks_x, row_p, col_n);
            let dc10: i32 = dc_val(dc_snapshot, blocks_x, row_p, col_nn);

            let dc11: i32 = dc_val(dc_snapshot, blocks_x, by, col_pp);
            let dc12: i32 = dc_val(dc_snapshot, blocks_x, by, col_p);
            let dc13: i32 = dc_val(dc_snapshot, blocks_x, by, bx);
            let dc14: i32 = dc_val(dc_snapshot, blocks_x, by, col_n);
            let dc15: i32 = dc_val(dc_snapshot, blocks_x, by, col_nn);

            let dc16: i32 = dc_val(dc_snapshot, blocks_x, row_n, col_pp);
            let dc17: i32 = dc_val(dc_snapshot, blocks_x, row_n, col_p);
            let dc18: i32 = dc_val(dc_snapshot, blocks_x, row_n, bx);
            let dc19: i32 = dc_val(dc_snapshot, blocks_x, row_n, col_n);
            let dc20: i32 = dc_val(dc_snapshot, blocks_x, row_n, col_nn);

            let dc21: i32 = dc_val(dc_snapshot, blocks_x, row_nn, col_pp);
            let dc22: i32 = dc_val(dc_snapshot, blocks_x, row_nn, col_p);
            let dc23: i32 = dc_val(dc_snapshot, blocks_x, row_nn, bx);
            let dc24: i32 = dc_val(dc_snapshot, blocks_x, row_nn, col_n);
            let dc25: i32 = dc_val(dc_snapshot, blocks_x, row_nn, col_nn);

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

/// Decode with an output colorspace override.
///
/// This public low-level entry point predates `Decoder`'s complete grayscale
/// and YCbCr conversion paths, and `Decoder` no longer calls it. Both arms
/// retain their legacy behavior for API compatibility: grayscale copies
/// component 0, and YCbCr replicates each chroma sample (nearest neighbour)
/// instead of upsampling it as libjpeg does. Callers that need libjpeg's
/// output — colorspace-aware RGB-to-gray, or component upsampling — should
/// configure [`crate::Decoder`] instead (P4-72, P4-244).
///
/// Geometry the legacy arms cannot handle is refused with
/// [`JpegError::Unsupported`], never a panic: for grayscale, a missing or
/// subsampled component 0 plane, or one that does not cover the output; for
/// YCbCr, fewer than three
/// planes, Cb and Cr sampled differently, luma not a whole multiple of
/// chroma (including chroma denser than luma), a component 0 plane that does
/// not cover the output, or a chroma plane shorter than one row. A chroma
/// plane shorter than the output needs repeats its last row.
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
    xmp_data: Option<Vec<u8>>,
    iptc_data: Option<Vec<u8>>,
    comment: Option<String>,
    density: DensityInfo,
    saved_markers: Vec<SavedMarker>,
    warnings: Vec<DecodeWarning>,
) -> Result<Image> {
    if target_cs == ColorSpace::Grayscale {
        if frame.components.is_empty() || component_planes.is_empty() || comp_block_sizes.is_empty()
        {
            return Err(JpegError::Unsupported(
                "legacy grayscale override requires a component 0 plane".into(),
            ));
        }
        let cw: usize =
            mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];
        if cw < out_width || component_planes[0].len() < out_height.saturating_mul(cw) {
            return Err(JpegError::Unsupported(format!(
                "legacy grayscale override requires a full-resolution component 0 plane (plane width {cw} vs output {out_width}x{out_height}); use Decoder's grayscale output instead"
            )));
        }
        let mut data: Vec<u8> = crate::common::try_alloc::try_reserved_vec(
            out_width * out_height,
            "colorspace override output",
        )?;
        for y in 0..out_height {
            data.extend_from_slice(&component_planes[0][y * cw..y * cw + out_width]);
        }
        return Ok(Image {
            xmp_data,
            iptc_data,
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
        });
    }
    if target_cs != ColorSpace::YCbCr {
        return Err(JpegError::Unsupported(format!(
            "output colorspace {target_cs:?} not supported"
        )));
    }
    if frame.components.len() < 3 || component_planes.len() < 3 || comp_block_sizes.len() < 3 {
        return Err(JpegError::Unsupported(
            "YCbCr output requires 3+ components".into(),
        ));
    }
    let legacy_refusal = |why: &str| {
        Err(JpegError::Unsupported(format!(
            "legacy YCbCr override {why}; use Decoder's YCbCr output instead"
        )))
    };
    // (plane width, rows per MCU row), with per-component IDCT sizes.
    let geometry = |index: usize| -> (usize, usize) {
        let component = &frame.components[index];
        (
            mcus_x * component.horizontal_sampling as usize * comp_block_sizes[index],
            component.vertical_sampling as usize * comp_block_sizes[index],
        )
    };
    let (cw, luma_rows): (usize, usize) = geometry(0);
    let (cbw, cb_rows): (usize, usize) = geometry(1);
    // Issue #683: Cr used to be read with Cb's geometry, and the factors
    // below divided by zero when chroma out-sampled luma.
    if geometry(2) != (cbw, cb_rows) {
        return legacy_refusal("needs Cb and Cr sampled alike");
    }
    if cbw == 0
        || cb_rows == 0
        || cw < cbw
        || luma_rows < cb_rows
        || !cw.is_multiple_of(cbw)
        || !luma_rows.is_multiple_of(cb_rows)
    {
        return legacy_refusal("needs luma sampled at a whole multiple of chroma");
    }
    // Effective upsample factors considering per-component IDCT sizes
    let hf: usize = cw / cbw;
    let vf: usize = luma_rows / cb_rows;
    if cw < out_width
        || out_height
            .checked_mul(cw)
            .is_none_or(|needed| component_planes[0].len() < needed)
    {
        return legacy_refusal("requires a component 0 plane covering the output");
    }
    let chroma_rows: usize = component_planes[1].len().min(component_planes[2].len()) / cbw;
    if chroma_rows == 0 && out_width > 0 && out_height > 0 {
        return legacy_refusal("requires chroma planes at least one row long");
    }
    let output_len: usize = out_width
        .checked_mul(out_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            JpegError::Unsupported(format!(
                "YCbCr output {out_width}x{out_height} overflows usize"
            ))
        })?;
    let mut data: Vec<u8> =
        crate::common::try_alloc::try_reserved_vec(output_len, "colorspace override output")?;
    for y in 0..out_height {
        for x in 0..out_width {
            data.push(component_planes[0][y * cw + x]);
            let cx: usize = (x / hf).min(cbw - 1);
            let cy: usize = (y / vf).min(chroma_rows - 1);
            data.push(component_planes[1][cy * cbw + cx]);
            data.push(component_planes[2][cy * cbw + cx]);
        }
    }
    Ok(Image {
        xmp_data,
        iptc_data,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed SOF with zero height reaches smoothing with an empty
    /// coefficient buffer (`mcus_y == 0`); the iMCU geometry must bail
    /// out instead of underflowing `total_imcu_rows - 1` (codex review
    /// of the P4-29 fix).
    #[test]
    fn block_smoothing_empty_grid_does_not_panic() {
        let mut empty: Vec<[i16; 64]> = Vec::new();
        let quant = QuantTable::from_zigzag(&[1u16; 64]);
        let coef_bits: [i32; SAVED_COEFS] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        apply_block_smoothing_coeffs(&mut empty, 0, 0, 0, 1, &coef_bits, &quant);
        apply_block_smoothing_coeffs(&mut empty, 2, 2, 0, 4, &coef_bits, &quant);
    }

    fn scan(components: &[u8], ss: u8, se: u8, ah: u8, al: u8) -> crate::decode::marker::ScanInfo {
        crate::decode::marker::ScanInfo {
            header: ScanHeader {
                components: components
                    .iter()
                    .map(
                        |&component_id| crate::common::types::ScanComponentSelector {
                            component_id,
                            dc_table_index: 0,
                            ac_table_index: 0,
                        },
                    )
                    .collect(),
                spec_start: ss,
                spec_end: se,
                succ_high: ah,
                succ_low: al,
            },
            data_offset: 0,
            dc_huffman_tables: Default::default(),
            ac_huffman_tables: Default::default(),
            restart_interval: 0,
        }
    }

    /// `compute_prev_coef_bits` follows `start_pass_phuff_decoder`'s save
    /// (jdphuff.c:132-137) and `smoothing_ok`'s latch (jdcoefct.c:451-456):
    /// the bits from before each component's last scan, 0 when that was the
    /// stream's first scan, -1 throughout for a single-scan stream (P4-255).
    #[test]
    fn prev_coef_bits_are_the_bits_before_each_components_last_scan() {
        let frame = FrameHeader {
            precision: 8,
            height: 16,
            width: 16,
            components: [1u8, 2]
                .iter()
                .map(|&id| crate::common::types::ComponentInfo {
                    id,
                    horizontal_sampling: 1,
                    vertical_sampling: 1,
                    quant_table_index: 0,
                })
                .collect(),
            is_progressive: true,
            is_lossless: false,
        };
        let scans = [
            scan(&[1, 2], 0, 0, 0, 1), // DC first, both components
            scan(&[1], 1, 5, 0, 2),    // component 1: AC 1-5
            scan(&[1], 1, 63, 2, 1),   // component 1: AC 1-63 refinement
        ];
        let current: Vec<[i32; SAVED_COEFS]> = compute_coef_bits(&scans, &frame);
        let prev: Vec<[i32; SAVED_COEFS]> = compute_prev_coef_bits(&scans, &frame);
        assert_eq!(current[0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        // Before its last scan, component 1 had AC 1-5 at Al 2 and 6-9 unseen.
        assert_eq!(prev[0], [-1, 2, 2, 2, 2, 2, -1, -1, -1, -1]);
        // Component 2's last scan was the stream's first: C saves 0.
        assert_eq!(prev[1], [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        // A single-scan stream latches -1.
        assert_eq!(
            compute_prev_coef_bits(&scans[..1], &frame)[0],
            [-1; SAVED_COEFS]
        );
    }
}
