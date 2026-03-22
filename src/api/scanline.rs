/// Scanline-level encode and decode API for row-by-row JPEG processing.
///
/// `ScanlineDecoder` wraps the existing `Decoder` and exposes a scanline-at-a-time
/// read interface. Internally it performs a lazy full decode on first access, then
/// serves individual rows from the decoded buffer.
///
/// `ScanlineEncoder` accumulates pixel rows one at a time, then delegates to the
/// existing compression pipeline on `finish()`.
use crate::common::error::{JpegError, Result};
use crate::common::types::{ColorSpace, DctMethod, FrameHeader, PixelFormat, Subsampling};
use crate::decode::pipeline::{Decoder, Image};

/// Row-by-row JPEG decoder.
pub struct ScanlineDecoder<'a> {
    decoder: Decoder<'a>,
    decoded_image: Option<Image>,
    current_line: usize,
    crop_x: Option<(usize, usize)>,
}

impl<'a> ScanlineDecoder<'a> {
    /// Create a new scanline decoder from raw JPEG data.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let decoder: Decoder<'a> = Decoder::new(data)?;
        Ok(Self {
            decoder,
            decoded_image: None,
            current_line: 0,
            crop_x: None,
        })
    }

    /// Returns the JPEG frame header.
    pub fn header(&self) -> &FrameHeader {
        self.decoder.header()
    }

    /// Returns the number of scanlines read so far.
    pub fn output_scanline(&self) -> usize {
        self.current_line
    }

    /// Set the output pixel format before starting decode.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.decoder.set_output_format(format);
    }

    /// Enable or disable fast (nearest-neighbor) upsampling.
    pub fn set_fast_upsample(&mut self, fast: bool) {
        self.decoder.fast_upsample = fast;
    }

    /// Enable or disable fast DCT for decoding.
    pub fn set_fast_dct(&mut self, fast: bool) {
        self.decoder.fast_dct = fast;
    }

    /// Set the DCT/IDCT method for decoding.
    pub fn set_dct_method(&mut self, method: DctMethod) {
        self.decoder.dct_method = method;
    }

    /// Enable or disable inter-block smoothing.
    pub fn set_block_smoothing(&mut self, smooth: bool) {
        self.decoder.block_smoothing = smooth;
    }

    /// Override the output color space.
    pub fn set_output_colorspace(&mut self, cs: ColorSpace) {
        self.decoder.output_colorspace = Some(cs);
    }

    /// Set horizontal crop region for scanline-level decoding.
    pub fn set_crop_x(&mut self, x: usize, width: usize) {
        self.crop_x = Some((x, width));
    }

    /// Decode and copy one scanline into `buf`.
    pub fn read_scanline(&mut self, buf: &mut [u8]) -> Result<()> {
        self.ensure_decoded()?;
        let img: &Image = self.decoded_image.as_ref().unwrap();
        let bpp: usize = img.pixel_format.bytes_per_pixel();
        let row_bytes: usize = img.width * bpp;
        if self.current_line >= img.height {
            return Err(JpegError::Unsupported("no more scanlines".into()));
        }
        let start: usize = self.current_line * row_bytes;
        buf[..row_bytes].copy_from_slice(&img.data[start..start + row_bytes]);
        self.current_line += 1;
        Ok(())
    }

    /// Skip scanlines without copying data.
    pub fn skip_scanlines(&mut self, count: usize) -> Result<usize> {
        self.ensure_decoded()?;
        let img: &Image = self.decoded_image.as_ref().unwrap();
        let remaining: usize = img.height - self.current_line;
        let actual: usize = count.min(remaining);
        self.current_line += actual;
        Ok(actual)
    }

    /// Finalize and return the complete decoded Image.
    pub fn finish(mut self) -> Result<Image> {
        self.ensure_decoded()?;
        Ok(self.decoded_image.take().unwrap())
    }

    fn ensure_decoded(&mut self) -> Result<()> {
        if self.decoded_image.is_none() {
            let mut img: Image = self.decoder.decode_image()?;
            if let Some((crop_x_offset, crop_width)) = self.crop_x {
                img = Self::apply_horizontal_crop(img, crop_x_offset, crop_width)?;
            }
            self.decoded_image = Some(img);
        }
        Ok(())
    }

    fn apply_horizontal_crop(img: Image, x: usize, width: usize) -> Result<Image> {
        let bpp: usize = img.pixel_format.bytes_per_pixel();
        if x + width > img.width {
            return Err(JpegError::Unsupported(format!(
                "crop region {}..{} exceeds image width {}",
                x,
                x + width,
                img.width
            )));
        }
        let src_row_bytes: usize = img.width * bpp;
        let dst_row_bytes: usize = width * bpp;
        let mut data: Vec<u8> = Vec::with_capacity(dst_row_bytes * img.height);
        for y in 0..img.height {
            let src_start: usize = y * src_row_bytes + x * bpp;
            data.extend_from_slice(&img.data[src_start..src_start + dst_row_bytes]);
        }
        Ok(Image {
            width,
            height: img.height,
            pixel_format: img.pixel_format,
            precision: img.precision,
            data,
            icc_profile: img.icc_profile,
            exif_data: img.exif_data,
            comment: img.comment,
            density: img.density,
            saved_markers: img.saved_markers,
            warnings: img.warnings,
        })
    }
}

/// Row-by-row JPEG encoder.
pub struct ScanlineEncoder {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    current_line: usize,
}

impl ScanlineEncoder {
    /// Create a new scanline encoder.
    pub fn new(width: usize, height: usize, pixel_format: PixelFormat) -> Self {
        let bpp: usize = pixel_format.bytes_per_pixel();
        Self {
            pixels: vec![0u8; width * height * bpp],
            width,
            height,
            pixel_format,
            quality: 75,
            subsampling: Subsampling::S420,
            current_line: 0,
        }
    }

    /// Set JPEG quality factor (1-100).
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality;
    }

    /// Set chroma subsampling mode.
    pub fn set_subsampling(&mut self, subsampling: Subsampling) {
        self.subsampling = subsampling;
    }

    /// Returns the index of the next scanline to be written.
    pub fn next_scanline(&self) -> usize {
        self.current_line
    }

    /// Write one row of pixel data.
    pub fn write_scanline(&mut self, row: &[u8]) -> Result<()> {
        if self.current_line >= self.height {
            return Err(JpegError::Unsupported("all scanlines written".into()));
        }
        let bpp: usize = self.pixel_format.bytes_per_pixel();
        let row_bytes: usize = self.width * bpp;
        let start: usize = self.current_line * row_bytes;
        self.pixels[start..start + row_bytes].copy_from_slice(&row[..row_bytes]);
        self.current_line += 1;
        Ok(())
    }

    /// Compress all accumulated scanlines into a JPEG byte stream.
    pub fn finish(self) -> Result<Vec<u8>> {
        if self.current_line != self.height {
            return Err(JpegError::Unsupported(format!(
                "not all scanlines written: {} of {}",
                self.current_line, self.height
            )));
        }
        crate::encode::pipeline::compress(
            &self.pixels,
            self.width,
            self.height,
            self.pixel_format,
            self.quality,
            self.subsampling,
            crate::common::types::DctMethod::IsLow,
        )
    }
}
