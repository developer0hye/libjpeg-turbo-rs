use libjpeg_turbo_rs::{DecodeWarning, ErrorHandler, ProgressInfo, ProgressListener};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

struct CountingHandler {
    warning_count: AtomicU32,
}

impl ErrorHandler for CountingHandler {
    fn emit_warning(&self, _warning: &DecodeWarning) {
        self.warning_count.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn custom_error_handler_compiles_and_works() {
    let handler = CountingHandler {
        warning_count: AtomicU32::new(0),
    };
    handler.emit_warning(&DecodeWarning::TruncatedData {
        decoded_mcus: 0,
        total_mcus: 10,
    });
    assert_eq!(handler.warning_count.load(Ordering::Relaxed), 1);
}

#[test]
fn progress_closure_compiles_and_works() {
    let call_count = Arc::new(AtomicU32::new(0));
    let count = call_count.clone();
    let listener = move |_info: ProgressInfo| {
        count.fetch_add(1, Ordering::Relaxed);
    };
    listener.update(ProgressInfo {
        pass: 0,
        total_passes: 1,
        progress: 0.5,
    });
    assert_eq!(call_count.load(Ordering::Relaxed), 1);
}

#[test]
fn default_error_handler_exists() {
    let _handler = libjpeg_turbo_rs::DefaultErrorHandler;
}

#[test]
fn progress_info_fields() {
    let info = ProgressInfo {
        pass: 1,
        total_passes: 3,
        progress: 0.75,
    };
    assert_eq!(info.pass, 1);
    assert_eq!(info.total_passes, 3);
    assert!((info.progress - 0.75).abs() < f32::EPSILON);
}
