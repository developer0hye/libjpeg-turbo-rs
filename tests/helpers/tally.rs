//! Planned-versus-executed accounting for C cross-validation matrices.
//!
//! P4-116: a matrix driver that reports green tells you the loop finished, not
//! that anything was compared. Every suite in this crate had at least one path
//! — a missing fixture, an oracle that exited non-zero, a dimension mismatch,
//! a Rust error — that `continue`d or `return`ed, so a run in which *zero*
//! cases actually reached a byte comparison was indistinguishable from a run
//! in which all of them did. One such regression shipped: the P4-13 harness
//! reported `1 passed` while skipping entirely, because its private tool
//! lookup ignored a working `cjpeg` on `PATH`.
//!
//! [`ComparisonTally`] closes that hole by making the accounting explicit.
//! A case ends in exactly one of two states:
//!
//! * **compared** — a real Rust-versus-C comparison ran to completion;
//! * **excluded** — the case is deliberately outside the matrix (a
//!   subsampling with no Rust equivalent, an option the installed C tool does
//!   not support), recorded *with its reason* so the exclusion set is visible
//!   in the test output rather than inferred from a silent `continue`.
//!
//! There is deliberately no "attempted" or "skipped" state. Anything that is
//! neither compared nor consciously excluded is a failure, and
//! [`ComparisonTally::finish`] says so.

use std::collections::BTreeMap;

/// Accounting for one cross-validation matrix.
///
/// Construct with the number of cases the matrix intends to run, record each
/// case as it resolves, and call [`finish`](Self::finish) at the end. `finish`
/// panics unless every planned case was accounted for and at least one real
/// comparison happened.
pub struct ComparisonTally {
    label: String,
    planned: usize,
    compared: usize,
    /// Exclusion reason → count, so the summary names *why* cases were
    /// dropped instead of only how many.
    excluded: BTreeMap<String, usize>,
    /// Set by [`finish`](Self::finish). A tally that goes out of scope without
    /// it asserted nothing, so `Drop` panics.
    ///
    /// A type-level `#[must_use]` cannot enforce this: it only fires when the
    /// value is discarded in statement position, and every real call site binds
    /// it to a `let`. `Drop` catches the case that actually happens — an early
    /// `return` inserted between construction and `finish()`.
    finished: bool,
}

impl Drop for ComparisonTally {
    fn drop(&mut self) {
        // Never panic while another panic is unwinding: that aborts the
        // process and replaces a readable failure with a crash. A test that is
        // already failing needs no help from this guard.
        if !self.finished && !std::thread::panicking() {
            panic!(
                "{}: ComparisonTally dropped without finish() — {} of {} planned \
                 comparisons went unverified. Some path returned early.",
                self.label, self.compared, self.planned
            );
        }
    }
}

impl ComparisonTally {
    /// Start accounting for a matrix that intends to run `planned` cases.
    ///
    /// `planned` must be computed from the matrix definition (the product of
    /// the option axes, the fixture count, …), never incremented inside the
    /// loop — a counter that grows as the loop runs can always be satisfied by
    /// a loop that does nothing.
    pub fn new(label: impl Into<String>, planned: usize) -> Self {
        Self {
            label: label.into(),
            planned,
            compared: 0,
            excluded: BTreeMap::new(),
            finished: false,
        }
    }

    /// Record one completed Rust-versus-C comparison.
    pub fn compared(&mut self) {
        self.compared += 1;
    }

    /// Record `n` completed comparisons at once, for inner loops that compare
    /// per row or per pixel block.
    pub fn compared_n(&mut self, n: usize) {
        self.compared += n;
    }

    /// Record one case as deliberately outside the matrix, with the reason.
    pub fn excluded(&mut self, reason: impl Into<String>) {
        self.excluded_n(1, reason);
    }

    /// Record `n` cases excluded for the same reason, for a whole axis that a
    /// missing feature removes at once. Stating the count is the point: a
    /// silently dropped axis reads as "covered" in the summary.
    pub fn excluded_n(&mut self, n: usize, reason: impl Into<String>) {
        *self.excluded.entry(reason.into()).or_insert(0) += n;
    }

    /// Comparisons completed so far.
    pub fn comparisons(&self) -> usize {
        self.compared
    }

    /// Assert the matrix ran as planned, and report what it did.
    ///
    /// Panics when the plan was empty, when nothing was compared, or when
    /// compared + excluded does not equal the plan — the three shapes a
    /// vacuous pass takes.
    pub fn finish(mut self) {
        self.finished = true;
        let excluded_total: usize = self.excluded.values().sum();
        let accounted: usize = self.compared + excluded_total;
        let summary: String = if self.excluded.is_empty() {
            String::new()
        } else {
            let detail: Vec<String> = self
                .excluded
                .iter()
                .map(|(reason, count)| format!("{count}x {reason}"))
                .collect();
            format!(" (excluded: {})", detail.join("; "))
        };
        assert!(
            self.planned > 0,
            "{}: the matrix planned zero cases, so a green result proves nothing",
            self.label
        );
        assert!(
            self.compared > 0,
            "{}: {}/{} cases planned but none reached a comparison{}",
            self.label,
            accounted,
            self.planned,
            summary
        );
        assert_eq!(
            accounted,
            self.planned,
            "{}: {} of {} planned cases are unaccounted for — {} compared, \
             {} excluded{}. Every case must either complete a comparison or be \
             recorded as a deliberate exclusion.",
            self.label,
            self.planned.saturating_sub(accounted),
            self.planned,
            self.compared,
            excluded_total,
            summary
        );
        eprintln!(
            "{}: {} comparisons completed out of {} planned{}",
            self.label, self.compared, self.planned, summary
        );
    }
}
