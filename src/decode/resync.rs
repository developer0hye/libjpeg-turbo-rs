//! Pluggable RST marker resync strategy (A6-3, mirrors C libjpeg-turbo's
//! `jpeg_resync_to_restart` hook).
//!
//! When a baseline/sequential JPEG uses restart intervals, each group of
//! `restart_interval` MCUs is terminated by a restart marker `0xFF 0xDn`
//! where `n` is a 3-bit counter that cycles 0→7. If the stream is
//! truncated, padded, or corrupted, the RST number actually observed may
//! not match what the decoder expects next. C libjpeg-turbo exposes
//! `jpeg_resync_to_restart(cinfo, desired)` so callers can plug in custom
//! recovery logic. This module provides the Rust equivalent.
//!
//! The default behavior (no strategy set) continues to skip past any
//! RST marker unconditionally — matching the existing Rust decoder and
//! `ResyncAction::Continue`. Callers can install a strategy via
//! `Decoder::set_resync_strategy()` to override that.

/// Action to take when a restart-marker desync is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResyncAction {
    /// Accept the observed marker as the new synchronization point and
    /// continue decoding. The decoder's RST counter is realigned to the
    /// observed number. This mirrors C's `jpeg_resync_to_restart` returning
    /// `TRUE` with `suspension = FALSE`.
    Continue,
    /// Skip past the current (bad) RST marker and scan forward until the
    /// next RST marker is found, then resume decoding there. Entropy state
    /// is reset at the recovery point, so pixels between the skipped
    /// markers will be garbled but the remaining scan can complete.
    Skip,
    /// Give up — propagate a `JpegError::CorruptData` error to the caller.
    Abort,
}

/// User-supplied callback consulted when a restart-marker desync occurs.
///
/// A "desync" is any of the following:
/// - The RST number at the next marker boundary is not the one the decoder
///   expected (e.g., decoder expected RST2 but read RST5).
/// - No RST marker was found at the expected bitstream position (EOF or
///   a different marker such as EOI).
///
/// The strategy returns the action the decoder should take.
pub trait RestartResyncStrategy {
    /// Called once per desync event.
    ///
    /// * `expected` — the RST number the decoder was expecting (0..=7).
    /// * `found` — the RST number actually observed at the current
    ///   position, or `None` if no RST marker was there.
    fn on_desync(&mut self, expected: u8, found: Option<u8>) -> ResyncAction;
}

/// Default strategy: always `Continue` — preserves the historical Rust
/// behavior of unconditionally skipping past any RST marker.
///
/// Installed automatically when `Decoder::set_resync_strategy` is not
/// called by the user.
pub struct DefaultResyncStrategy;

impl RestartResyncStrategy for DefaultResyncStrategy {
    fn on_desync(&mut self, _expected: u8, _found: Option<u8>) -> ResyncAction {
        ResyncAction::Continue
    }
}
