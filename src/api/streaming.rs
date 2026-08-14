use crate::common::error::{JpegError, Result};
use crate::common::types::{FrameHeader, PixelFormat, ScalingFactor};
use crate::decode::pipeline::{Decoder, Image};
#[allow(unused_imports)]
use alloc::format;

/// Streaming JPEG decoder — reads header first, then decodes on demand.
pub struct StreamingDecoder<'a> {
    inner: Decoder<'a>,
    /// Rows dropped from the front of the next `decode()` output
    /// (issue #383). Applied through the pipeline's vertical crop, so
    /// fully-skipped iMCU rows also skip their IDCT.
    skip_rows: usize,
}

impl<'a> StreamingDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let inner = Decoder::new(data)?;
        Ok(Self {
            inner,
            skip_rows: 0,
        })
    }

    pub fn header(&self) -> &FrameHeader {
        self.inner.header()
    }

    /// Set the output pixel format.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.inner.set_output_format(format);
    }

    /// Set the decompression scaling factor.
    pub fn set_scale(&mut self, scale: ScalingFactor) {
        self.inner.set_scale(scale);
    }

    /// Enable lenient mode: continue on errors, filling corrupt areas with gray.
    pub fn set_lenient(&mut self, lenient: bool) {
        self.inner.set_lenient(lenient);
    }

    /// Set horizontal crop. xoffset is auto-aligned down to iMCU column boundary.
    /// Updates xoffset and width in-place to reflect aligned values.
    ///
    /// A crop origin past the right edge, or an `xoffset + width` that is not
    /// representable, is refused rather than wrapped: the arithmetic below
    /// assumes the origin lies inside the image, and this function already
    /// returns `Result` (P4-139 chunk 2). A window that merely *ends* past the
    /// right edge keeps its long-standing behaviour and is clamped to the
    /// image; unlike C's `jpeg_crop_scanline`, which errors
    /// (`jdapistd.c: JERR_WIDTH_OVERFLOW`), so does an origin exactly *at* the
    /// width and a zero width. Narrowing to C's rule would change what
    /// existing callers get back and belongs to P4-103, which owns
    /// `jpeg_crop_scanline` semantics.
    ///
    /// Both bounds are in **output** (post-scale) space, which is what
    /// `set_crop` consumes — call `set_scale` first. The unscaled
    /// `header.width` and the unscaled block size 8 were used here until
    /// P4-139 chunk 2, so an upscaled decode aligned to the wrong grid and
    /// measured the edge against the wrong width.
    pub fn crop_scanline(&mut self, xoffset: &mut usize, width: &mut usize) -> Result<()> {
        let output_width: usize = self.inner.output_width();
        let block_size: usize = self.inner.output_block_size();
        let max_h = self
            .inner
            .header()
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let imcu_width = max_h * block_size;

        if *xoffset > output_width {
            return Err(JpegError::Unsupported(format!(
                "crop origin {} lies past the output width {}",
                *xoffset, output_width
            )));
        }
        let crop_end: usize = xoffset.checked_add(*width).ok_or_else(|| {
            JpegError::Unsupported(format!(
                "crop region {}..+{} is not representable",
                *xoffset, *width
            ))
        })?;

        // Align xoffset down to the iMCU boundary. The end is clamped to the
        // image before alignment so the rounding cannot leave the address
        // space; for any window that already fitted, clamping first and
        // clamping after give the same aligned width.
        let aligned_x = (*xoffset / imcu_width) * imcu_width;
        let aligned_end = crop_end.min(output_width).div_ceil(imcu_width) * imcu_width;
        let aligned_width = (aligned_end - aligned_x).min(output_width - aligned_x);

        *xoffset = aligned_x;
        *width = aligned_width;

        self.inner.set_crop(aligned_x, aligned_width);
        Ok(())
    }

    /// Skip scanlines: every subsequent `decode()` starts `num_lines`
    /// rows further down (the skip is sticky, not consumed by a decode —
    /// unlike `ScanlineDecoder::skip_scanlines`, which advances a cursor
    /// over an already-decoded image). Consecutive calls accumulate.
    ///
    /// `num_lines` counts **output** rows: call `set_scale` (and
    /// `crop_scanline`) *before* skipping, exactly as C freezes scaling
    /// before `jpeg_skip_scanlines` may run. Returns the number of rows
    /// actually skipped, clamped at the image end the way C clamps
    /// (jdapistd.c: `num_lines = output_height - output_scanline`) — so
    /// skipping everything is allowed and a later `decode()` yields a
    /// zero-height image, matching `djpeg -skip 0,H-1`'s empty PPM.
    ///
    /// Implemented through the pipeline's vertical crop: iMCU rows that
    /// fall entirely inside the skipped region never run their IDCT, so
    /// the per-skipped-row work is bounded (issue #383 shipped this as a
    /// success-reporting no-op).
    pub fn skip_scanlines(&mut self, num_lines: usize) -> Result<usize> {
        let out_h: usize = self.inner.output_height();
        let remaining: usize = out_h.saturating_sub(self.skip_rows);
        let actual: usize = num_lines.min(remaining);
        self.skip_rows += actual;
        if self.skip_rows > 0 {
            self.inner
                .set_crop_y(self.skip_rows, out_h - self.skip_rows);
        }
        Ok(actual)
    }

    /// Decode the JPEG payload using the already-parsed metadata.
    pub fn decode(&self) -> Result<Image> {
        self.inner.decode_image()
    }
}
