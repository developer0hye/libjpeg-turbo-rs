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
