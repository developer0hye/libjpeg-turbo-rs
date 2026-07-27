// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::boxed::Box;
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

impl TransformOp {
    /// The lossless transform that makes an image with this EXIF
    /// orientation (tag 0x0112, values 1-8) display upright — the
    /// standard mapping every photo application otherwise hand-rolls
    /// (issue #391). Returns `None` for values outside 1..=8.
    ///
    /// Lossless caveats (same as `jpegtran`): when the dimensions are
    /// not iMCU-aligned (e.g. a 600x450 4:2:0 photo), coefficient-domain
    /// transforms cannot fully reorient the partial edge blocks — use
    /// [`crate::TransformOptions`] with `perfect` to reject such images,
    /// `trim` to drop the edge, or the pixel-domain
    /// `Image::apply_orientation` for exact output at any size. And the
    /// default marker copying preserves the original orientation tag on
    /// the transformed output, which would make EXIF-aware viewers
    /// rotate twice — pair the transform with
    /// [`crate::MarkerCopyMode::None`] (or rewrite the tag) when the
    /// output is meant for display.
    pub fn from_exif_orientation(orientation: u8) -> Option<Self> {
        match orientation {
            1 => Some(Self::None),
            2 => Some(Self::HFlip),
            3 => Some(Self::Rot180),
            4 => Some(Self::VFlip),
            5 => Some(Self::Transpose),
            6 => Some(Self::Rot90),
            7 => Some(Self::Transverse),
            8 => Some(Self::Rot270),
            _ => None,
        }
    }
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
    /// Restart interval. 0 = disabled.
    ///
    /// When `restart_in_rows == false` (default): interpreted as MCU blocks (DRI value),
    /// matching jpegtran `-restart Nb`.
    /// When `restart_in_rows == true`: interpreted as MCU rows; the actual DRI value
    /// is recomputed as `restart_interval * output_mcus_per_row` after applying the
    /// spatial transform / crop / trim, matching jpegtran `-restart N`.
    pub restart_interval: u16,
    /// Treat `restart_interval` as MCU rows (true) instead of MCU blocks (false).
    /// When true, the implementation recomputes the actual DRI based on the output
    /// MCU grid produced by the transform. Required for parity with jpegtran `-restart N`
    /// when the output dimensions differ from the source (rotation, crop, trim).
    pub restart_in_rows: bool,
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
            restart_interval: 0,
            restart_in_rows: false,
            copy_markers: MarkerCopyMode::All,
            custom_filter: None,
        }
    }
}

impl core::fmt::Debug for TransformOptions {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
            .field("restart_interval", &self.restart_interval)
            .field("restart_in_rows", &self.restart_in_rows)
            .field("copy_markers", &self.copy_markers)
            .field(
                "custom_filter",
                &self.custom_filter.as_ref().map(|_| "<filter fn>"),
            )
            .finish()
    }
}
