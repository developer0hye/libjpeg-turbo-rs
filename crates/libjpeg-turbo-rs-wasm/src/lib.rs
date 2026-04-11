use wasm_bindgen::prelude::*;

use libjpeg_turbo_rs as jpeg;

// ===== Error conversion =====

fn to_js_err(e: jpeg::JpegError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ===== Enums =====

/// Output pixel format.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Grayscale = 0,
    Rgb = 1,
    Rgba = 2,
    Bgr = 3,
    Bgra = 4,
    Rgbx = 5,
    Bgrx = 6,
    Xrgb = 7,
    Xbgr = 8,
    Argb = 9,
    Abgr = 10,
}

impl From<PixelFormat> for jpeg::PixelFormat {
    fn from(f: PixelFormat) -> Self {
        match f {
            PixelFormat::Grayscale => jpeg::PixelFormat::Grayscale,
            PixelFormat::Rgb => jpeg::PixelFormat::Rgb,
            PixelFormat::Rgba => jpeg::PixelFormat::Rgba,
            PixelFormat::Bgr => jpeg::PixelFormat::Bgr,
            PixelFormat::Bgra => jpeg::PixelFormat::Bgra,
            PixelFormat::Rgbx => jpeg::PixelFormat::Rgbx,
            PixelFormat::Bgrx => jpeg::PixelFormat::Bgrx,
            PixelFormat::Xrgb => jpeg::PixelFormat::Xrgb,
            PixelFormat::Xbgr => jpeg::PixelFormat::Xbgr,
            PixelFormat::Argb => jpeg::PixelFormat::Argb,
            PixelFormat::Abgr => jpeg::PixelFormat::Abgr,
        }
    }
}

impl From<jpeg::PixelFormat> for PixelFormat {
    fn from(f: jpeg::PixelFormat) -> Self {
        match f {
            jpeg::PixelFormat::Grayscale => PixelFormat::Grayscale,
            jpeg::PixelFormat::Rgb => PixelFormat::Rgb,
            jpeg::PixelFormat::Rgba => PixelFormat::Rgba,
            jpeg::PixelFormat::Bgr => PixelFormat::Bgr,
            jpeg::PixelFormat::Bgra => PixelFormat::Bgra,
            jpeg::PixelFormat::Rgbx => PixelFormat::Rgbx,
            jpeg::PixelFormat::Bgrx => PixelFormat::Bgrx,
            jpeg::PixelFormat::Xrgb => PixelFormat::Xrgb,
            jpeg::PixelFormat::Xbgr => PixelFormat::Xbgr,
            jpeg::PixelFormat::Argb => PixelFormat::Argb,
            jpeg::PixelFormat::Abgr => PixelFormat::Abgr,
            other => panic!(
                "PixelFormat::{:?} is not supported in the WASM API; use decode_to() with an explicit format",
                other
            ),
        }
    }
}

/// Chroma subsampling mode.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:4:4 — no subsampling
    S444 = 0,
    /// 4:2:2 — horizontal 2x
    S422 = 1,
    /// 4:2:0 — horizontal 2x, vertical 2x
    S420 = 2,
    /// 4:4:0 — vertical 2x
    S440 = 3,
    /// 4:1:1 — horizontal 4x
    S411 = 4,
}

impl From<Subsampling> for jpeg::Subsampling {
    fn from(s: Subsampling) -> Self {
        match s {
            Subsampling::S444 => jpeg::Subsampling::S444,
            Subsampling::S422 => jpeg::Subsampling::S422,
            Subsampling::S420 => jpeg::Subsampling::S420,
            Subsampling::S440 => jpeg::Subsampling::S440,
            Subsampling::S411 => jpeg::Subsampling::S411,
        }
    }
}

/// DCT algorithm selection.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DctMethod {
    /// Accurate integer DCT (default).
    IsLow = 0,
    /// Fast integer DCT with reduced accuracy.
    IsFast = 1,
    /// Floating-point DCT.
    Float = 2,
}

impl From<DctMethod> for jpeg::DctMethod {
    fn from(m: DctMethod) -> Self {
        match m {
            DctMethod::IsLow => jpeg::DctMethod::IsLow,
            DctMethod::IsFast => jpeg::DctMethod::IsFast,
            DctMethod::Float => jpeg::DctMethod::Float,
        }
    }
}

// ===== DecodedImage =====

/// Decoded JPEG image with pixel data and metadata.
#[wasm_bindgen]
pub struct DecodedImage {
    inner: jpeg::Image,
}

#[wasm_bindgen]
impl DecodedImage {
    /// Image width in pixels.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.inner.width as u32
    }

    /// Image height in pixels.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.inner.height as u32
    }

    /// Pixel format of the decoded data.
    #[wasm_bindgen(getter)]
    pub fn format(&self) -> PixelFormat {
        self.inner.pixel_format.into()
    }

    /// Raw pixel data as Uint8Array.
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(self.inner.data.as_slice())
    }

    /// Pointer to pixel data in WASM linear memory (for zero-copy access).
    #[wasm_bindgen(getter, js_name = "dataPtr")]
    pub fn data_ptr(&self) -> u32 {
        self.inner.data.as_ptr() as u32
    }

    /// Length of pixel data in bytes.
    #[wasm_bindgen(getter, js_name = "dataLen")]
    pub fn data_len(&self) -> u32 {
        self.inner.data.len() as u32
    }

    /// ICC color profile data, if present.
    #[wasm_bindgen(getter, js_name = "iccProfile")]
    pub fn icc_profile(&self) -> Option<js_sys::Uint8Array> {
        self.inner
            .icc_profile
            .as_ref()
            .map(|d| js_sys::Uint8Array::from(d.as_slice()))
    }

    /// EXIF metadata, if present.
    #[wasm_bindgen(getter, js_name = "exifData")]
    pub fn exif_data(&self) -> Option<js_sys::Uint8Array> {
        self.inner
            .exif_data
            .as_ref()
            .map(|d| js_sys::Uint8Array::from(d.as_slice()))
    }

    /// JPEG comment string, if present.
    #[wasm_bindgen(getter)]
    pub fn comment(&self) -> Option<String> {
        self.inner.comment.clone()
    }

    /// Bytes per pixel for the current format.
    #[wasm_bindgen(getter, js_name = "bytesPerPixel")]
    pub fn bytes_per_pixel(&self) -> u32 {
        self.inner.pixel_format.bytes_per_pixel() as u32
    }
}

// ===== Top-level decode functions =====

/// Decode a JPEG image from raw bytes. Returns RGB pixel data by default.
#[wasm_bindgen]
pub fn decode(data: &[u8]) -> Result<DecodedImage, JsValue> {
    let image: jpeg::Image = jpeg::decompress(data).map_err(to_js_err)?;
    Ok(DecodedImage { inner: image })
}

/// Decode a JPEG image to a specific pixel format.
#[wasm_bindgen(js_name = "decodeTo")]
pub fn decode_to(data: &[u8], format: PixelFormat) -> Result<DecodedImage, JsValue> {
    let image: jpeg::Image = jpeg::decompress_to(data, format.into()).map_err(to_js_err)?;
    Ok(DecodedImage { inner: image })
}

// ===== Top-level encode function =====

/// Encode raw pixels to JPEG. Returns compressed JPEG data as Uint8Array.
#[wasm_bindgen]
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<js_sys::Uint8Array, JsValue> {
    let data: Vec<u8> = jpeg::compress(
        pixels,
        width as usize,
        height as usize,
        format.into(),
        quality,
        subsampling.into(),
    )
    .map_err(to_js_err)?;
    Ok(js_sys::Uint8Array::from(data.as_slice()))
}

// ===== JpegEncoder (builder pattern) =====

/// JPEG encoder with configurable options.
///
/// ```js
/// const enc = new JpegEncoder(pixels, 640, 480, PixelFormat.Rgb);
/// enc.quality(90);
/// enc.subsampling(Subsampling.S444);
/// enc.optimizeHuffman(true);
/// const jpeg = enc.encode();
/// ```
#[wasm_bindgen(js_name = "JpegEncoder")]
pub struct WasmEncoder {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    pixel_format: jpeg::PixelFormat,
    quality: u8,
    subsampling: jpeg::Subsampling,
    optimize_huffman: bool,
    progressive: bool,
    arithmetic: bool,
    dct_method: jpeg::DctMethod,
    icc_profile: Option<Vec<u8>>,
    exif_data: Option<Vec<u8>>,
    comment: Option<String>,
}

#[wasm_bindgen(js_class = "JpegEncoder")]
impl WasmEncoder {
    /// Create a new encoder for the given pixel data.
    #[wasm_bindgen(constructor)]
    pub fn new(pixels: &[u8], width: u32, height: u32, format: PixelFormat) -> Self {
        Self {
            pixels: pixels.to_vec(),
            width: width as usize,
            height: height as usize,
            pixel_format: format.into(),
            quality: 75,
            subsampling: jpeg::Subsampling::S420,
            optimize_huffman: false,
            progressive: false,
            arithmetic: false,
            dct_method: jpeg::DctMethod::IsLow,
            icc_profile: None,
            exif_data: None,
            comment: None,
        }
    }

    /// Set JPEG quality (1-100, default 75).
    pub fn quality(&mut self, q: u8) {
        self.quality = q;
    }

    /// Set chroma subsampling mode.
    pub fn subsampling(&mut self, s: Subsampling) {
        self.subsampling = s.into();
    }

    /// Enable/disable Huffman table optimization (smaller files, slower encode).
    #[wasm_bindgen(js_name = "optimizeHuffman")]
    pub fn optimize_huffman(&mut self, optimize: bool) {
        self.optimize_huffman = optimize;
    }

    /// Enable/disable progressive JPEG encoding.
    pub fn progressive(&mut self, progressive: bool) {
        self.progressive = progressive;
    }

    /// Enable/disable arithmetic coding.
    pub fn arithmetic(&mut self, arithmetic: bool) {
        self.arithmetic = arithmetic;
    }

    /// Set DCT algorithm.
    #[wasm_bindgen(js_name = "dctMethod")]
    pub fn dct_method(&mut self, method: DctMethod) {
        self.dct_method = method.into();
    }

    /// Set ICC color profile data.
    #[wasm_bindgen(js_name = "iccProfile")]
    pub fn icc_profile(&mut self, data: &[u8]) {
        self.icc_profile = Some(data.to_vec());
    }

    /// Set EXIF metadata.
    #[wasm_bindgen(js_name = "exifData")]
    pub fn exif_data(&mut self, data: &[u8]) {
        self.exif_data = Some(data.to_vec());
    }

    /// Set JPEG comment string.
    pub fn comment(&mut self, text: &str) {
        self.comment = Some(text.to_string());
    }

    /// Encode the image and return compressed JPEG data.
    pub fn encode(&self) -> Result<js_sys::Uint8Array, JsValue> {
        let mut encoder: jpeg::Encoder<'_> =
            jpeg::Encoder::new(&self.pixels, self.width, self.height, self.pixel_format)
                .quality(self.quality)
                .subsampling(self.subsampling)
                .optimize_huffman(self.optimize_huffman)
                .progressive(self.progressive)
                .arithmetic(self.arithmetic)
                .dct_method(self.dct_method);

        if let Some(ref icc) = self.icc_profile {
            encoder = encoder.icc_profile(icc);
        }
        if let Some(ref exif) = self.exif_data {
            encoder = encoder.exif_data(exif);
        }
        if let Some(ref comment) = self.comment {
            encoder = encoder.comment(comment);
        }

        let data: Vec<u8> = encoder.encode().map_err(to_js_err)?;
        Ok(js_sys::Uint8Array::from(data.as_slice()))
    }
}

// ===== Utility functions =====

/// Get the dimensions of a JPEG without fully decoding it.
/// Returns [width, height] as a Uint32Array.
#[wasm_bindgen(js_name = "jpegDimensions")]
pub fn jpeg_dimensions(data: &[u8]) -> Result<js_sys::Uint32Array, JsValue> {
    let decoder: jpeg::ScanlineDecoder<'_> = jpeg::ScanlineDecoder::new(data).map_err(to_js_err)?;
    let header: &jpeg::FrameHeader = decoder.header();
    let dims: [u32; 2] = [header.width as u32, header.height as u32];
    Ok(js_sys::Uint32Array::from(&dims[..]))
}
