#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use super::downsample_chroma_block_h2v1_ssse3;
use super::{
    downsample_chroma_block, extract_block, may_use_islow_simd_kernel, vec, BitWriter, HuffTable,
    HuffmanEncoder, QuantDivisors, Subsampling,
};

/// Encode a single 8x8 block through the DCT -> quantize -> Huffman pipeline.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn encode_single_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant_table: &QuantDivisors,
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // C jccoefct.c:178-199 — when a block is entirely outside the image
    // (subsampled MCU stride exceeds image dimensions), emit a dummy block:
    // DC = previous block's DC (so DC diff = 0), AC all zero. This matches
    // upstream's "Create a row of dummy blocks at the bottom of the image"
    // path and keeps the DC frequency distribution byte-identical to cjpeg
    // for non-444 subsamplings whose MCU height/width does not divide the
    // image dimensions evenly.
    if block_x >= plane_width || block_y >= plane_height {
        let mut dummy = [0i16; 64];
        dummy[0] = *prev_dc;
        HuffmanEncoder::encode_block(writer, &dummy, prev_dc, dc_table, ac_table);
        return;
    }

    let mut quantized = [0i16; 64];

    // The fused SIMD path uses islow FDCT internally. Skip it for ifast/float
    // so the caller-provided fdct_quantize_fn (with correct divisors) is used.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // Fused path for interior blocks: load u8 → FDCT → quantize → zigzag
    // without intermediate [i16; 64] buffer between extract and FDCT.
    if use_fused_simd && block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "simd"))]
        {
            unsafe {
                crate::simd::wasm32::wasm_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
    }

    // Border blocks: pad to a local 8×8 buffer with replicated-last-pixel,
    // then use the NEON/AVX2 fused path.  This ensures byte-identical output
    // with C libjpeg-turbo's expand_right_edge + NEON convsamp/fdct path.
    let is_edge: bool = block_x + 8 > plane_width || block_y + 8 > plane_height;
    if is_edge {
        let mut local_buf = [0u8; 64]; // 8×8 padded block
        for row in 0..8usize {
            let src_y: usize = (block_y + row).min(plane_height - 1);
            for col in 0..8usize {
                let src_x: usize = (block_x + col).min(plane_width - 1);
                local_buf[row * 8 + col] = plane[src_y * plane_width + src_x];
            }
        }

        if use_fused_simd {
            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            {
                unsafe {
                    crate::simd::aarch64::neon_extract_fdct_quantize(
                        local_buf.as_ptr(),
                        8,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if crate::cpu_has!("avx2") {
                    unsafe {
                        crate::simd::x86_64::avx2_extract_fdct_quantize(
                            local_buf.as_ptr(),
                            8,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                    return;
                }
            }
        }
    }

    // Generic path: extract block + caller-provided FDCT+quantize.
    // Used for ifast, float, and non-SIMD fallback.
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}

/// Whether an ISLOW-only fused SIMD FDCT can replace the requested transform.
#[inline]
pub(super) fn can_use_fused_islow(
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> bool {
    !core::ptr::eq(
        fdct_quantize_fn as *const (),
        crate::simd::scalar::scalar_fdct_ifast_quantize as *const (),
    ) && !core::ptr::eq(
        fdct_quantize_fn as *const (),
        crate::simd::scalar::scalar_fdct_float_quantize as *const (),
    )
}

/// Encode a full color MCU (multiple Y blocks + Cb + Cr blocks).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn encode_color_mcu(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    subsampling: Subsampling,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    match subsampling {
        Subsampling::S444 | Subsampling::Unknown => {
            // 1 Y + 1 Cb + 1 Cr = 3 blocks, MCU-level hoisting saves 2 begin/end pairs
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                encode_mcu_444_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
            {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    cb_plane,
                    width,
                    height,
                    x0,
                    y0,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S422 => {
            // 2 Y + 1 Cb + 1 Cr = 4 blocks, MCU-level hoisting saves 3 begin/end pairs
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                encode_mcu_422_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
            {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_downsampled_chroma_block(
                    cb_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    1,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_downsampled_chroma_block(
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    1,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S420 => {
            // 4 Y blocks (2x2 arrangement) + 1 downsampled Cb + 1 downsampled Cr
            // Optimized path: do all FDCT+quantize first, then all Huffman encoding
            // with a single hoisted begin_block/end_block per MCU (saves 5 pairs).
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                encode_mcu_420_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
            {
                // Y blocks in order: top-left, top-right, bottom-left, bottom-right
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0 + 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + 8,
                    y0 + 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                // Downsample chroma: 2x2 box filter
                encode_downsampled_chroma_block(
                    cb_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    2,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_downsampled_chroma_block(
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    2,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S440 => {
            // 2 Y blocks vertically: (x0, y0) and (x0, y0+8)
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
                fdct_quantize_fn,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0 + 8,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
                fdct_quantize_fn,
            );
            // Cb/Cr downsampled 1x2
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                1,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                1,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
        Subsampling::S411 => {
            // 4 Y blocks horizontally
            for i in 0..4 {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + i * 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
            // Cb/Cr downsampled 4x1
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                4,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                4,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
        Subsampling::S441 => {
            // 4 Y blocks vertically
            for i in 0..4 {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0 + i * 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
            // Cb/Cr downsampled 1x4
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                1,
                4,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                1,
                4,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
        Subsampling::S410 => {
            // 4 Y horizontal × 2 vertical = 8 luma blocks per MCU
            for dy in [0usize, 8] {
                for dx in [0usize, 8, 16, 24] {
                    encode_single_block(
                        y_plane,
                        width,
                        height,
                        x0 + dx,
                        y0 + dy,
                        luma_quant,
                        dc_luma_table,
                        ac_luma_table,
                        writer,
                        prev_dc_y,
                        fdct_quantize_fn,
                    );
                }
            }
            // Cb/Cr downsampled 4x2
            for (plane, prev_dc) in [(cb_plane, &mut *prev_dc_cb), (cr_plane, &mut *prev_dc_cr)] {
                encode_downsampled_chroma_block(
                    plane,
                    width,
                    height,
                    x0,
                    y0,
                    4,
                    2,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S24 => {
            // 2 Y horizontal × 4 vertical = 8 luma blocks per MCU
            for dy in [0usize, 8, 16, 24] {
                for dx in [0usize, 8] {
                    encode_single_block(
                        y_plane,
                        width,
                        height,
                        x0 + dx,
                        y0 + dy,
                        luma_quant,
                        dc_luma_table,
                        ac_luma_table,
                        writer,
                        prev_dc_y,
                        fdct_quantize_fn,
                    );
                }
            }
            // Cb/Cr downsampled 2x4
            for (plane, prev_dc) in [(cb_plane, &mut *prev_dc_cb), (cr_plane, &mut *prev_dc_cr)] {
                encode_downsampled_chroma_block(
                    plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    4,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc,
                    fdct_quantize_fn,
                );
            }
        }
    }
}

/// Check if a Y block at the given pixel position is a dummy block
/// (beyond the real image boundary in either dimension).
/// C libjpeg-turbo creates dummy blocks with AC=0, DC=prev for these positions.
#[inline]
pub(super) fn is_y_dummy(block_x_px: usize, block_y_px: usize, y_wib: usize, y_hib: usize) -> bool {
    block_x_px / 8 >= y_wib || block_y_px / 8 >= y_hib
}

/// Encode a dummy block (AC=0, DC=previous block's DC) matching C jccoefct.c.
#[inline]
pub(super) fn encode_dummy_block(
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
) {
    let mut dummy: [i16; 64] = [0i16; 64];
    dummy[0] = *prev_dc;
    HuffmanEncoder::encode_block(writer, &dummy, prev_dc, dc_table, ac_table);
}

/// Encode a color MCU with dummy Y blocks for the last MCU column.
///
/// C libjpeg-turbo creates "dummy" blocks beyond `width_in_blocks`: all AC=0,
/// DC = previous block's DC (jccoefct.c lines 184-191). This produces smaller
/// output than FDCT'ing the padded pixel data.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn encode_color_mcu_with_dummies(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    subsampling: Subsampling,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    eff_col_width: usize,
    eff_row_height: usize,
) {
    let (h_samp, v_samp) = subsampling.sampling_factors();
    let y_mcu_width: usize = h_samp as usize;
    let y_mcu_height: usize = v_samp as usize;

    // Encode Y blocks: real blocks where vy < eff_row_height && hx < eff_col_width,
    // dummy blocks elsewhere (AC=0, DC=prev_dc, matching C jccoefct.c lines 184-199).
    for vy in 0..y_mcu_height {
        let is_dummy_row: bool = vy >= eff_row_height;
        for hx in 0..y_mcu_width {
            let is_dummy_col: bool = hx >= eff_col_width;
            if is_dummy_row || is_dummy_col {
                // Dummy block: AC=0, DC=previous block's DC
                let mut dummy = [0i16; 64];
                dummy[0] = *prev_dc_y;
                HuffmanEncoder::encode_block(
                    writer,
                    &dummy,
                    prev_dc_y,
                    dc_luma_table,
                    ac_luma_table,
                );
            } else {
                let bx: usize = x0 + hx * 8;
                let by: usize = y0 + vy * 8;
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    bx,
                    by,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
        }
    }

    // Chroma blocks: always encode normally (chroma MCU_width=1 for S422/S420)
    encode_downsampled_chroma_block(
        cb_plane,
        width,
        height,
        x0,
        y0,
        h_samp as usize,
        v_samp as usize,
        chroma_quant,
        dc_chroma_table,
        ac_chroma_table,
        writer,
        prev_dc_cb,
        fdct_quantize_fn,
    );
    encode_downsampled_chroma_block(
        cr_plane,
        width,
        height,
        x0,
        y0,
        h_samp as usize,
        v_samp as usize,
        chroma_quant,
        dc_chroma_table,
        ac_chroma_table,
        writer,
        prev_dc_cr,
        fdct_quantize_fn,
    );
}

/// Helper: FDCT+quantize a single block (interior: fused SIMD, border: scalar fallback).
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn fdct_quantize_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    out: &mut [i16; 64],
) {
    if block_x + 8 <= plane_width
        && block_y + 8 <= plane_height
        && crate::cpu_has!("avx2")
        && may_use_islow_simd_kernel(fdct_quantize_fn)
    {
        unsafe {
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                plane.as_ptr().add(block_y * plane_width + block_x),
                plane_width,
                quant,
                out,
            );
        }
    } else {
        let mut block = [0i16; 64];
        extract_block(
            plane,
            plane_width,
            plane_height,
            block_x,
            block_y,
            &mut block,
        );
        fdct_quantize_fn(&mut block, quant, out);
    }
}

/// Helper: FDCT+quantize a downsampled H2V1 chroma block.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn fdct_quantize_chroma_h2v1(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    out: &mut [i16; 64],
) {
    // Fused path: downsample + FDCT + quantize in one pass (AVX2)
    if can_use_fused_islow(fdct_quantize_fn)
        && block_x + 16 <= plane_width
        && block_y + 8 <= plane_height
        && crate::cpu_has!("avx2")
        && may_use_islow_simd_kernel(fdct_quantize_fn)
    {
        unsafe {
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                plane.as_ptr().add(block_y * plane_width + block_x),
                plane_width,
                quant,
                out,
            );
        }
        return;
    }
    // Separate downsample + FDCT (SSSE3 downsample only)
    if block_x + 16 <= plane_width
        && block_y + 8 <= plane_height
        && crate::cpu_has!("ssse3")
        && may_use_islow_simd_kernel(fdct_quantize_fn)
    {
        let mut block = [0i16; 64];
        unsafe {
            downsample_chroma_block_h2v1_ssse3(plane, plane_width, block_x, block_y, &mut block);
        }
        fdct_quantize_fn(&mut block, quant, out);
    } else {
        let mut block = [0i16; 64];
        downsample_chroma_block(
            plane,
            plane_width,
            plane_height,
            block_x,
            block_y,
            2,
            1,
            &mut block,
        );
        fdct_quantize_fn(&mut block, quant, out);
    }
}

/// Optimized 4:4:4 MCU encoding with MCU-level BitWriter hoisting.
///
/// 3 blocks (Y + Cb + Cr), saves 2 begin/end pairs per MCU.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_mcu_444_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 3] = [[0i16; 64]; 3];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool = crate::cpu_has!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);
    let interior: bool = x0 + 8 <= width && y0 + 8 <= height;

    if can_use_fused_islow(fdct_quantize_fn) && interior && has_avx2 {
        unsafe {
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_plane.as_ptr().add(y0 * width + x0),
                width,
                luma_quant,
                &mut q[0],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[2],
            );
        }
    } else {
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[0],
        );
        fdct_quantize_block(
            cb_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[1],
        );
        fdct_quantize_block(
            cr_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[2],
        );
    }

    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(1536);
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[0],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[1],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[2],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Optimized 4:2:2 MCU encoding with MCU-level BitWriter hoisting.
///
/// 4 blocks (2 Y + Cb + Cr), saves 3 begin/end pairs per MCU.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_mcu_422_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 4] = [[0i16; 64]; 4];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool = crate::cpu_has!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);
    // Interior check: 2 Y blocks (16 wide) + H2V1 chroma (16 wide, 8 tall)
    let interior: bool = x0 + 16 <= width && y0 + 8 <= height;

    if can_use_fused_islow(fdct_quantize_fn) && interior && has_avx2 {
        unsafe {
            let y_ptr: *const u8 = y_plane.as_ptr().add(y0 * width + x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, width, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                width,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[3],
            );
        }
    } else {
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[0],
        );
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0 + 8,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[1],
        );
        fdct_quantize_chroma_h2v1(
            cb_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[2],
        );
        fdct_quantize_chroma_h2v1(
            cr_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[3],
        );
    }

    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(2048);
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[0],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[1],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[2],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[3],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Optimized 4:2:0 MCU encoding with MCU-level BitWriter hoisting.
///
/// Does all FDCT+quantize for 6 blocks first, then all Huffman encoding in one
/// hoisted begin_block/end_block pair. Saves 5 begin/end pairs per MCU.
/// 6 blocks × 128 bytes = 768 bytes of quantized data fits in L1.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_mcu_420_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // Phase 1: FDCT + quantize all 6 blocks (4 Y + 1 Cb + 1 Cr)
    // Cache feature detection once per MCU (not per block).
    let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool = crate::cpu_has!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);

    // Check if all 4 Y blocks and both chroma blocks are interior (common case).
    // For 1080p with 16x16 MCUs, only edge MCUs fail this check.
    let interior: bool = x0 + 16 <= width && y0 + 16 <= height;

    let use_fused_islow: bool = can_use_fused_islow(fdct_quantize_fn);

    if use_fused_islow && interior && has_avx2 {
        // Fast path: all blocks are interior, use fused SIMD for everything
        unsafe {
            let y_ptr: *const u8 = y_plane.as_ptr().add(y0 * width + x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, width, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                width,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * width),
                width,
                luma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * width + 8),
                width,
                luma_quant,
                &mut q[3],
            );
            crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[4],
            );
            crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[5],
            );
        }
    } else {
        // Slow path: handle edge MCUs with per-block bounds checking
        let y_offsets: [(usize, usize); 4] =
            [(x0, y0), (x0 + 8, y0), (x0, y0 + 8), (x0 + 8, y0 + 8)];
        for (idx, &(bx, by)) in y_offsets.iter().enumerate() {
            if use_fused_islow && has_avx2 && bx + 8 <= width && by + 8 <= height {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        y_plane.as_ptr().add(by * width + bx),
                        width,
                        luma_quant,
                        &mut q[idx],
                    );
                }
            } else {
                let mut block = [0i16; 64];
                extract_block(y_plane, width, height, bx, by, &mut block);
                fdct_quantize_fn(&mut block, luma_quant, &mut q[idx]);
            }
        }
        if use_fused_islow && has_avx2 && x0 + 16 <= width && y0 + 16 <= height {
            unsafe {
                crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                    cb_plane.as_ptr().add(y0 * width + x0),
                    width,
                    chroma_quant,
                    &mut q[4],
                );
            }
        } else {
            let mut block = [0i16; 64];
            downsample_chroma_block(cb_plane, width, height, x0, y0, 2, 2, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[4]);
        }
        if use_fused_islow && has_avx2 && x0 + 16 <= width && y0 + 16 <= height {
            unsafe {
                crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                    cr_plane.as_ptr().add(y0 * width + x0),
                    width,
                    chroma_quant,
                    &mut q[5],
                );
            }
        } else {
            let mut block = [0i16; 64];
            downsample_chroma_block(cr_plane, width, height, x0, y0, 2, 2, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[5]);
        }
    }

    // Phase 2: Huffman encode all 6 blocks with MCU-level hoisted state.
    // 3072 bytes = 6 blocks × 512 bytes worst-case per block.
    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(3072);

        // 4 Y blocks
        for block in q.iter().take(4) {
            HuffmanEncoder::encode_block_hoisted(
                &mut pb,
                &mut fb,
                &mut buf,
                block,
                prev_dc_y,
                dc_luma_table,
                ac_luma_table,
            );
        }
        // Cb
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[4],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        // Cr
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[5],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );

        writer.end_block(pb, fb, buf);
    }
}

/// Encode one 420 MCU using pre-downsampled half-resolution chroma buffers.
///
/// Y blocks are read from full-resolution `y_plane` (stride = `y_stride`).
/// Cb/Cr blocks are read from half-resolution buffers (stride = `chroma_stride`).
/// Since chroma is already downsampled, we use `avx2_extract_fdct_quantize`
/// instead of the heavier `avx2_downsample_h2v2_fdct_quantize`.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) fn encode_mcu_420_half_chroma(
    y_plane: &[u8],
    y_stride: usize,
    cb_half: &[u8],
    cr_half: &[u8],
    chroma_stride: usize,
    y_x0: usize,
    y_y0: usize,
    chroma_x0: usize,
    chroma_y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool = crate::cpu_has!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);

    // Check if all blocks are interior (common case for non-edge MCUs)
    let y_interior: bool = y_x0 + 16 <= y_stride && y_y0 + 16 <= 16;
    let c_interior: bool = chroma_x0 + 8 <= chroma_stride && chroma_y0 + 8 <= 8;

    let use_fused_islow: bool = can_use_fused_islow(fdct_quantize_fn);

    if use_fused_islow && y_interior && c_interior && has_avx2 {
        unsafe {
            // 4 Y blocks from full-res plane
            let y_ptr: *const u8 = y_plane.as_ptr().add(y_y0 * y_stride + y_x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, y_stride, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                y_stride,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * y_stride),
                y_stride,
                luma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * y_stride + 8),
                y_stride,
                luma_quant,
                &mut q[3],
            );
            // Cb/Cr from half-res plane (already downsampled, just extract 8×8)
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cb_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                chroma_stride,
                chroma_quant,
                &mut q[4],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cr_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                chroma_stride,
                chroma_quant,
                &mut q[5],
            );
        }
    } else {
        // Fallback: scalar extract for edge MCUs
        let y_offsets: [(usize, usize); 4] = [
            (y_x0, y_y0),
            (y_x0 + 8, y_y0),
            (y_x0, y_y0 + 8),
            (y_x0 + 8, y_y0 + 8),
        ];
        for (idx, &(bx, by)) in y_offsets.iter().enumerate() {
            if use_fused_islow && has_avx2 && bx + 8 <= y_stride && by + 8 <= 16 {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        y_plane.as_ptr().add(by * y_stride + bx),
                        y_stride,
                        luma_quant,
                        &mut q[idx],
                    );
                }
            } else {
                let mut block = [0i16; 64];
                extract_block(y_plane, y_stride, 16, bx, by, &mut block);
                fdct_quantize_fn(&mut block, luma_quant, &mut q[idx]);
            }
        }
        // Chroma from half-res
        if use_fused_islow && has_avx2 && chroma_x0 + 8 <= chroma_stride && chroma_y0 + 8 <= 8 {
            unsafe {
                crate::simd::x86_64::avx2_extract_fdct_quantize(
                    cb_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                    chroma_stride,
                    chroma_quant,
                    &mut q[4],
                );
                crate::simd::x86_64::avx2_extract_fdct_quantize(
                    cr_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                    chroma_stride,
                    chroma_quant,
                    &mut q[5],
                );
            }
        } else {
            let mut block = [0i16; 64];
            extract_block(cb_half, chroma_stride, 8, chroma_x0, chroma_y0, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[4]);
            extract_block(cr_half, chroma_stride, 8, chroma_x0, chroma_y0, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[5]);
        }
    }

    // Huffman encode all 6 blocks with MCU-level hoisted state
    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(3072);
        for block in q.iter().take(4) {
            HuffmanEncoder::encode_block_hoisted(
                &mut pb,
                &mut fb,
                &mut buf,
                block,
                prev_dc_y,
                dc_luma_table,
                ac_luma_table,
            );
        }
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[4],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[5],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Encode a downsampled chroma block through the full pipeline.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn encode_downsampled_chroma_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    quant_table: &QuantDivisors,
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // The fused SIMD paths use islow FDCT; skip for ifast/float.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // Fused NEON path: downsample + FDCT + quantize + zigzag in one pass,
    // eliminating the intermediate [i16; 64] downsampled block.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    if use_fused_simd {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if block_x + src_w <= plane_width && block_y + src_h <= plane_height {
            let plane_ptr: *const u8 =
                unsafe { plane.as_ptr().add(block_y * plane_width + block_x) };
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        plane_ptr,
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        plane_ptr,
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // x86_64 fused path: AVX2 downsample+FDCT+quantize+zigzag
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if use_fused_simd {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if crate::cpu_has!("avx2")
            && block_x + src_w <= plane_width
            && block_y + src_h <= plane_height
        {
            // Fused downsample+FDCT+quantize for H2V2
            if h_factor == 2 && v_factor == 2 {
                let mut quantized = [0i16; 64];
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            // Fused downsample+FDCT+quantize for H2V1
            if h_factor == 2 && v_factor == 1 {
                let mut quantized = [0i16; 64];
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // Edge block: pad source area locally and use NEON/AVX2 fused path.
    // This matches C libjpeg-turbo's expand_right_edge + SIMD downsample behavior.
    let src_w: usize = 8 * h_factor;
    let src_h: usize = 8 * v_factor;
    let active_rows: usize = plane_height.saturating_sub(block_y).min(src_h);
    let last_downsample_group_start: usize = active_rows
        .div_ceil(v_factor)
        .saturating_sub(1)
        .saturating_mul(v_factor);
    let mut local_buf = vec![0u8; src_w * src_h];
    for row in 0..src_h {
        let source_row: usize = if row < active_rows || v_factor == 1 {
            row.min(active_rows.saturating_sub(1))
        } else {
            // C pads the input only to complete the final vertical row group,
            // downsamples it, then replicates that final downsampled row to
            // the bottom of the iMCU. Repeating the final input group here is
            // equivalent and avoids downsampling duplicated last samples.
            let group_offset: usize = (row - last_downsample_group_start) % v_factor;
            (last_downsample_group_start + group_offset).min(active_rows - 1)
        };
        let src_y: usize = block_y + source_row;
        for col in 0..src_w {
            let src_x: usize = (block_x + col).min(plane_width - 1);
            local_buf[row * src_w + col] = plane[src_y * plane_width + src_x];
        }
    }

    // Try NEON/AVX2 fused downsample+FDCT+quantize on the padded local buffer
    if use_fused_simd {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
                let mut quantized = [0i16; 64];
                if h_factor == 2 && v_factor == 2 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                            local_buf.as_ptr(),
                            src_w,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                    return;
                }
                if h_factor == 2 && v_factor == 1 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                            local_buf.as_ptr(),
                            src_w,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                    return;
                }
            }
        }
    }

    // Scalar fallback (non-SIMD platforms): downsample from padded buffer
    let mut block = [0i16; 64];
    downsample_chroma_block(
        &local_buf, src_w, src_h, 0, 0, h_factor, v_factor, &mut block,
    );

    let mut quantized = [0i16; 64];
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);

    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}
