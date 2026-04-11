/// All errors that can occur during JPEG processing.
#[derive(Debug, thiserror::Error)]
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

    #[error("unexpected end of data")]
    UnexpectedEof,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, JpegError>;

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
}

impl std::fmt::Display for DecodeWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        }
    }
}
