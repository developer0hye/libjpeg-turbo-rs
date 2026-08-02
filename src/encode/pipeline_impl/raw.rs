use super::{
    build_huff_table, encode_single_block, format, marker_writer, scale_quant_for_fdct, tables,
    vec, BitWriter, HuffTable, JpegError, QuantDivisors, Result, Subsampling, ToString, Vec,
};

/// Compress JPEG from raw downsampled component planes.
///
/// Bypasses color conversion and chroma downsampling — the caller provides
/// data already in the YCbCr color space at the correct subsampled dimensions.
/// This matches libjpeg-turbo's `jpeg_write_raw_data()` functionality.
#[allow(clippy::too_many_arguments)]
pub fn compress_raw(
    planes: &[&[u8]],
    plane_widths: &[usize],
    plane_heights: &[usize],
    image_width: usize,
    image_height: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    if image_width == 0 || image_height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if planes.len() != plane_widths.len() || planes.len() != plane_heights.len() {
        return Err(JpegError::CorruptData(
            "planes, plane_widths, and plane_heights must have the same length".to_string(),
        ));
    }
    let is_grayscale: bool = planes.len() == 1;
    if is_grayscale && subsampling != Subsampling::S444 {
        return Err(JpegError::CorruptData(format!(
            "1 plane (grayscale) is only valid with S444 subsampling, got {:?}",
            subsampling
        )));
    }
    if !is_grayscale && planes.len() != 3 {
        return Err(JpegError::CorruptData(format!(
            "expected 1 (grayscale) or 3 (YCbCr) planes, got {}",
            planes.len()
        )));
    }
    let (h_samp, v_samp): (u8, u8) = subsampling.sampling_factors();
    if !is_grayscale {
        let expected_cb_w: usize = image_width.div_ceil(h_samp as usize);
        let expected_cb_h: usize = image_height.div_ceil(v_samp as usize);
        if plane_widths[0] != image_width || plane_heights[0] != image_height {
            return Err(JpegError::CorruptData(format!(
                "Y plane dimensions {}x{} do not match image dimensions {}x{}",
                plane_widths[0], plane_heights[0], image_width, image_height
            )));
        }
        for comp_idx in 1..3 {
            let comp_name: &str = if comp_idx == 1 { "Cb" } else { "Cr" };
            if plane_widths[comp_idx] != expected_cb_w || plane_heights[comp_idx] != expected_cb_h {
                return Err(JpegError::CorruptData(format!(
                    "{} plane dimensions {}x{} do not match expected {}x{} for {:?} subsampling",
                    comp_name,
                    plane_widths[comp_idx],
                    plane_heights[comp_idx],
                    expected_cb_w,
                    expected_cb_h,
                    subsampling
                )));
            }
        }
    }
    for (i, plane) in planes.iter().enumerate() {
        let expected_size: usize = plane_widths[i] * plane_heights[i];
        if plane.len() < expected_size {
            return Err(JpegError::BufferTooSmall {
                need: expected_size,
                got: plane.len(),
            });
        }
    }
    let luma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors: QuantDivisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors: QuantDivisors = scale_quant_for_fdct(&chroma_quant);
    let dc_luma_table: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table: HuffTable =
        build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table: HuffTable =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table: HuffTable =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);
    let (mcu_w, mcu_h): (usize, usize) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };
    let mcus_x: usize = image_width.div_ceil(mcu_w);
    let mcus_y: usize = image_height.div_ceil(mcu_h);
    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn = enc_simd.fdct_quantize;
    let mut bit_writer: BitWriter = BitWriter::new(image_width * image_height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;
            if is_grayscale {
                encode_single_block(
                    planes[0],
                    plane_widths[0],
                    plane_heights[0],
                    x0,
                    y0,
                    &luma_divisors,
                    &dc_luma_table,
                    &ac_luma_table,
                    &mut bit_writer,
                    &mut prev_dc_y,
                    fdct_quantize_fn,
                );
            } else {
                let h: usize = h_samp as usize;
                let v: usize = v_samp as usize;
                for vy in 0..v {
                    for hx in 0..h {
                        encode_single_block(
                            planes[0],
                            plane_widths[0],
                            plane_heights[0],
                            x0 + hx * 8,
                            y0 + vy * 8,
                            &luma_divisors,
                            &dc_luma_table,
                            &ac_luma_table,
                            &mut bit_writer,
                            &mut prev_dc_y,
                            fdct_quantize_fn,
                        );
                    }
                }
                let chroma_x: usize = x0 / h;
                let chroma_y: usize = y0 / v;
                encode_single_block(
                    planes[1],
                    plane_widths[1],
                    plane_heights[1],
                    chroma_x,
                    chroma_y,
                    &chroma_divisors,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    planes[2],
                    plane_widths[2],
                    plane_heights[2],
                    chroma_x,
                    chroma_y,
                    &chroma_divisors,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
    }
    bit_writer.flush();
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);
    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }
    if is_grayscale {
        let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
        marker_writer::write_sof0(
            &mut output,
            image_width as u16,
            image_height as u16,
            &components,
        );
    } else {
        let components: Vec<(u8, u8, u8, u8)> =
            vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof0(
            &mut output,
            image_width as u16,
            image_height as u16,
            &components,
        );
    }
    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );
    marker_writer::write_dht(
        &mut output,
        1,
        0,
        &tables::AC_LUMINANCE_BITS,
        &tables::AC_LUMINANCE_VALUES,
    );
    if !is_grayscale {
        marker_writer::write_dht(
            &mut output,
            0,
            1,
            &tables::DC_CHROMINANCE_BITS,
            &tables::DC_CHROMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut output,
            1,
            1,
            &tables::AC_CHROMINANCE_BITS,
            &tables::AC_CHROMINANCE_VALUES,
        );
    }
    if is_grayscale {
        marker_writer::write_sos(&mut output, &[(1, 0, 0)]);
    } else {
        marker_writer::write_sos(&mut output, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
    }
    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);
    Ok(output)
}
