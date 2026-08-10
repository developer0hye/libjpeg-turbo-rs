// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::string::String;
/// All errors that can occur during JPEG processing.
///
/// `#[non_exhaustive]`: variants may be added in minor releases (the
/// #355 `LimitExceeded` addition was source-breaking for exhaustive
/// downstream matches — this prevents a repeat).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JpegError {
    #[error("invalid marker: 0xFF{0:02X}")]
    InvalidMarker(u8),

    #[error("unexpected marker: 0xFF{0:02X}")]
    UnexpectedMarker(u8),

    #[error("unsupported feature: {0}")]
    Unsupported(String),

    #[error("corrupt data: {0}")]
    CorruptData(String),

    #[error("buffer too small: need {need}, got {got}")]
    BufferTooSmall { need: usize, got: usize },

    #[error("{what} {actual} exceeds limit {limit}")]
    LimitExceeded {
        what: &'static str,
        actual: u64,
        limit: u64,
    },

    /// The allocator refused a buffer whose size is derived from the input.
    ///
    /// `vec![0u8; n]` and friends *abort the process* when the allocator says
    /// no, so a hostile header could turn into a denial of service that no
    /// caller can catch. Sizes taken from the JPEG stream are allocated with
    /// `try_reserve_exact` instead and surface here (P4-136 criterion 4).
    #[error("allocation of {bytes} bytes for {what} failed")]
    AllocationFailed { what: &'static str, bytes: u64 },

    #[error("unexpected end of data")]
    UnexpectedEof,

    #[cfg(feature = "std")]
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = core::result::Result<T, JpegError>;

/// Non-fatal warning that allows recovery in lenient mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeWarning {
    /// Huffman decode error at the given MCU position.
    HuffmanError {
        mcu_x: usize,
        mcu_y: usize,
        message: String,
    },
    /// Data ended before all MCUs were decoded.
    TruncatedData {
        decoded_mcus: usize,
        total_mcus: usize,
    },
    /// A spec-valid but unsupported feature was encountered and recovered from
    /// in lenient mode by emitting a best-effort (neutral-filled) raster — e.g.
    /// non-standard sampling where a chroma component out-samples luma (the
    /// upsample pipeline assumes luma is the maximally-sampled component; see
    /// LAST_MILE P4-21). Strict mode rejects these instead.
    UnsupportedRecovered { detail: String },
}

impl core::fmt::Display for DecodeWarning {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HuffmanError {
                mcu_x,
                mcu_y,
                message,
            } => {
                write!(
                    f,
                    "Huffman decode error at MCU ({}, {}): {}",
                    mcu_x, mcu_y, message
                )
            }
            Self::TruncatedData {
                decoded_mcus,
                total_mcus,
            } => {
                write!(
                    f,
                    "truncated data: decoded {}/{} MCUs",
                    decoded_mcus, total_mcus
                )
            }
            Self::UnsupportedRecovered { detail } => {
                write!(f, "unsupported feature recovered (lenient): {}", detail)
            }
        }
    }
}
