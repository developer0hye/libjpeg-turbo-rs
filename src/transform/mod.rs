/// Lossless JPEG transforms operating on DCT coefficients.
///
/// Implements spatial transforms (flip, rotate, transpose) that manipulate
/// DCT coefficients without decoding/re-encoding, preserving image quality.
pub mod spatial;

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
