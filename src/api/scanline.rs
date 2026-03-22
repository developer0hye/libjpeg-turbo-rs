/// Scanline-level encode and decode API for row-by-row JPEG processing.
///
/// `ScanlineDecoder` wraps the existing `Decoder` and exposes a scanline-at-a-time
/// read interface. Internally it performs a lazy full decode on first access, then
/// serves individual rows from the decoded buffer.
///
/// `ScanlineEncoder` accumulates pixel rows one at a time, then delegates to the
/// existing compression pipeline on `finish()`.
use crate::common::error::{JpegError, Result};
use crate::common::types::{FrameHeader, PixelFormat, Subsampling};
use crate::decode::pipeline::{Decoder, Image};

/// Row-by-row JPEG decoder.
///
/// Uses a lazy-decode strategy: the full image is decoded on the first call to
/// `read_scanline`, `skip_scanlines`, or `finish`. Subsequent reads serve rows
/// directly from the in-memory buffer.
///
/// # Example
/// ```no_run
/// use libjpeg_turbo_rs::{ScanlineDecoder, PixelFormat};
///
/// let jpeg_data: &[u8] = &[]; // your JPEG bytes
/// let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
/// dec.set_output_format(PixelFormat::Rgb);
/// let width = dec.header().width as usize;
/// let mut row = vec![0u8; width * 3];
/// while dec.output_scanline() < dec.header().height as usize {
///     dec.read_scanline(&mut row).unwrap();
/// }
/// ```
pub struct ScanlineDecoder<'a> {
    decoder: Decoder<'a>,
    decoded_image: Option<Image>,
    current_line: usize,
}

impl<'a> ScanlineDecoder<'a> {
    /// Create a new scanline decoder from raw JPEG data.
    ///
    /// Parses the JPEG header immediately but defers pixel decoding until
    /// the first scanline read or `finish()` call.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let decoder: Decoder<'a> = Decoder::new(data)?;
        Ok(Self {
            decoder,
            decoded_image: None,
            current_line: 0,
        })
    }

    /// Returns the JPEG frame header (dimensions, components, etc.).
    pub fn header(&self) -> &FrameHeader {
        self.decoder.header()
    }

    /// Returns the number of scanlines read so far.
    pub fn output_scanline(&self) -> usize {
        self.current_line
    }

    /// Set the output pixel format before starting decode.
    ///
    /// Must be called before the first `read_scanline` or `finish`. Has no
    /// effect after decoding has started.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.decoder.set_output_format(format);
    }

    /// Decode (lazily on first call) and copy one scanline into `buf`.
    ///
    /// The buffer must be at least `width * bytes_per_pixel` bytes long.
    /// Returns an error if all scanlines have already been read.
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
    ///
    /// Returns the actual number of lines skipped (clamped to remaining lines).
    pub fn skip_scanlines(&mut self, count: usize) -> Result<usize> {
        self.ensure_decoded()?;
        let img: &Image = self.decoded_image.as_ref().unwrap();
        let remaining: usize = img.height - self.current_line;
        let actual: usize = count.min(remaining);
        self.current_line += actual;
        Ok(actual)
    }

    /// Finalize and return the complete decoded Image.
    ///
    /// Triggers decoding if it hasn't happened yet.
    pub fn finish(mut self) -> Result<Image> {
        self.ensure_decoded()?;
        Ok(self.decoded_image.take().unwrap())
    }

    /// Trigger lazy decode if not already done.
    fn ensure_decoded(&mut self) -> Result<()> {
        if self.decoded_image.is_none() {
            self.decoded_image = Some(self.decoder.decode_image()?);
        }
        Ok(())
    }
}

/// Row-by-row JPEG encoder.
///
/// Accumulates pixel data one scanline at a time, then compresses the complete
/// image on `finish()`.
///
/// # Example
/// ```no_run
/// use libjpeg_turbo_rs::{ScanlineEncoder, PixelFormat, Subsampling};
///
/// let mut enc = ScanlineEncoder::new(640, 480, PixelFormat::Rgb);
/// enc.set_quality(85);
/// enc.set_subsampling(Subsampling::S422);
/// let row = vec![128u8; 640 * 3];
/// for _ in 0..480 {
///     enc.write_scanline(&row).unwrap();
/// }
/// let jpeg_bytes = enc.finish().unwrap();
/// ```
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
    /// Create a new scanline encoder for the given image dimensions and format.
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

    /// Set JPEG quality factor (1-100, where 100 is best quality).
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality;
    }

    /// Set chroma subsampling mode.
    pub fn set_subsampling(&mut self, subsampling: Subsampling) {
        self.subsampling = subsampling;
    }

    /// Returns the index of the next scanline to be written (0-based).
    pub fn next_scanline(&self) -> usize {
        self.current_line
    }

    /// Write one row of pixel data.
    ///
    /// The `row` slice must contain at least `width * bytes_per_pixel` bytes.
    /// Returns an error if all scanlines have already been written.
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
    ///
    /// Returns an error if not all scanlines have been written.
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
