/// Lossless JPEG transforms operating on DCT coefficients.
///
/// Implements spatial transforms (flip, rotate, transpose) that manipulate
/// DCT coefficients without decoding/re-encoding, preserving image quality.
pub mod spatial;

use crate::common::types::CropRegion;

/// Lossless transform operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformOp {
    /// No transform (copy).
    None,
    /// Horizontal flip (mirror left-right).
    HFlip,
    /// Vertical flip (mirror top-bottom).
    VFlip,
    /// Transpose (swap rows and columns).
    Transpose,
    /// Transverse transpose (rotate 180 + transpose).
    Transverse,
    /// Rotate 90 degrees clockwise.
    Rot90,
    /// Rotate 180 degrees.
    Rot180,
    /// Rotate 270 degrees clockwise.
    Rot270,
}

/// Configuration for a lossless transform operation.
#[derive(Debug, Clone)]
pub struct TransformInfo {
    pub transform: TransformOp,
}

impl Default for TransformInfo {
    fn default() -> Self {
        Self {
            transform: TransformOp::None,
        }
    }
}

/// Marker copy behavior during transform.
///
/// Controls which APP and COM markers are preserved when writing transformed output.
/// Corresponds to jpegtran's `-copy` option: `a` (all), `n` (none), `i` (ICC only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarkerCopyMode {
    /// Copy all APP and COM markers (default). Corresponds to `-copy all` / TJXOPT default.
    #[default]
    All,
    /// Copy no markers. Corresponds to `-copy none` / TJXOPT_COPYNONE.
    None,
    /// Copy only ICC profile (APP2) markers. Corresponds to `-copy icc`.
    IccOnly,
}

impl From<bool> for MarkerCopyMode {
    fn from(value: bool) -> Self {
        if value {
            MarkerCopyMode::All
        } else {
            MarkerCopyMode::None
        }
    }
}

/// All 9 TJXOPT transform flags matching libjpeg-turbo's jpegtran options,
/// plus a user callback for coefficient inspection/modification.
///
/// Controls lossless JPEG transforms including spatial operations, partial MCU
/// handling, grayscale conversion, and output encoding options.
pub struct TransformOptions {
    /// Spatial transform to apply (flip, rotate, transpose, etc.).
    pub op: TransformOp,
    /// Fail if image dimensions are not iMCU-aligned for the requested transform.
    /// Corresponds to TJXOPT_PERFECT.
    pub perfect: bool,
    /// Discard partial iMCU blocks at image edges before transforming.
    /// Corresponds to TJXOPT_TRIM.
    pub trim: bool,
    /// Crop to the specified region (in MCU-aligned coordinates).
    /// Corresponds to TJXOPT_CROP.
    pub crop: Option<CropRegion>,
    /// Drop chroma components, producing a grayscale JPEG.
    /// Corresponds to TJXOPT_GRAY.
    pub grayscale: bool,
    /// Dry run: perform validation but return empty output.
    /// Corresponds to TJXOPT_NOOUTPUT.
    pub no_output: bool,
    /// Re-encode output as progressive JPEG.
    /// Corresponds to TJXOPT_PROGRESSIVE.
    pub progressive: bool,
    /// Re-encode output with arithmetic entropy coding.
    /// Corresponds to TJXOPT_ARITHMETIC (libjpeg-turbo 3.x).
    pub arithmetic: bool,
    /// Re-encode output with optimized Huffman tables (2-pass).
    /// Corresponds to TJXOPT_OPTIMIZE (libjpeg-turbo 3.x).
    pub optimize: bool,
    /// Marker copy behavior: All (default), None (TJXOPT_COPYNONE), or IccOnly.
    /// See [`MarkerCopyMode`] for details.
    pub copy_markers: MarkerCopyMode,
    /// User callback for inspecting/modifying DCT coefficients during transform.
    /// Called once per block after the spatial transform is applied.
    /// Arguments: (block: &mut [i16; 64], component_index: usize, block_x: usize, block_y: usize)
    /// Corresponds to `tjtransform.customFilter`.
    #[allow(clippy::type_complexity)]
    pub custom_filter: Option<Box<dyn Fn(&mut [i16; 64], usize, usize, usize)>>,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            op: TransformOp::None,
            perfect: false,
            trim: false,
            crop: None,
            grayscale: false,
            no_output: false,
            progressive: false,
            arithmetic: false,
            optimize: false,
            copy_markers: MarkerCopyMode::All,
            custom_filter: None,
        }
    }
}

impl std::fmt::Debug for TransformOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformOptions")
            .field("op", &self.op)
            .field("perfect", &self.perfect)
            .field("trim", &self.trim)
            .field("crop", &self.crop)
            .field("grayscale", &self.grayscale)
            .field("no_output", &self.no_output)
            .field("progressive", &self.progressive)
            .field("arithmetic", &self.arithmetic)
            .field("optimize", &self.optimize)
            .field("copy_markers", &self.copy_markers)
            .field(
                "custom_filter",
                &self.custom_filter.as_ref().map(|_| "<filter fn>"),
            )
            .finish()
    }
}
