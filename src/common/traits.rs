use crate::common::error::{DecodeWarning, JpegError};

/// Customizable error handling for JPEG operations.
pub trait ErrorHandler: Send + Sync {
    /// Called on fatal error. Default: panic.
    fn error_exit(&self, err: &JpegError) -> ! {
        panic!("JPEG fatal error: {err}");
    }

    /// Called on non-fatal warning. Default: ignore.
    fn emit_warning(&self, _warning: &DecodeWarning) {}

    /// Called for trace/debug messages. Default: ignore.
    fn trace(&self, _level: u8, _msg: &str) {}
}

/// Default error handler that uses Result-based flow.
pub struct DefaultErrorHandler;
impl ErrorHandler for DefaultErrorHandler {}

/// Progress information for encode/decode operations.
#[derive(Debug, Clone, Copy)]
pub struct ProgressInfo {
    /// Current pass (0-based).
    pub pass: u32,
    /// Total number of passes.
    pub total_passes: u32,
    /// Progress within current pass (0.0 to 1.0).
    pub progress: f32,
}

/// Listener for encode/decode progress updates.
pub trait ProgressListener: Send + Sync {
    fn update(&self, info: ProgressInfo);
}

/// Allow closures as ProgressListener.
impl<F: Fn(ProgressInfo) + Send + Sync> ProgressListener for F {
    fn update(&self, info: ProgressInfo) {
        self(info);
    }
}
