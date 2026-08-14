//! Centralized memory-layout arithmetic for image spans (P4-139, #478).
//!
//! Every buffer this crate sizes is some arrangement of
//! `width × height × bytes_per_pixel`, possibly row-padded by a stride.
//! Before this module, that product was written out at each call site with
//! inconsistent overflow behaviour — checked in one file, saturating in a
//! second, plain multiplication in a third — and the same formula existed
//! in disagreeing checked/unchecked twins. [`ImageLayout`] is the one
//! implementation: construction validates the geometry, and every span it
//! reports came through checked arithmetic with the element size *inside*
//! the chain.
//!
//! What this type enforces is the **universal** contract every span shares:
//!
//! * no intermediate product wraps (`checked_mul`/`checked_add` throughout);
//! * the total fits `isize::MAX` — `slice::from_raw_parts`' documented
//!   precondition and the allocator's object-size ceiling, so a span that
//!   passes here can back both a `Vec` and a raw slice;
//! * a caller-supplied stride is at least the row's own bytes, and the
//!   strided total is `(height − 1) · stride + row_bytes` — the last row
//!   needs only its pixels, exactly upstream TurboJPEG's rule.
//!
//! What it deliberately does **not** enforce are the boundary-specific
//! ceilings, which belong to the boundary because their *error codes* do:
//! the classic C ABI's `JDIMENSION` width cap raises `JERR_WIDTH_OVERFLOW`
//! (`jcmaster.c:190-194`) and stays with `checked_samples_per_row` in the
//! capi crate; `DecodeLimits::max_pixels` raises `LimitExceeded` from
//! `check_frame` with its own diagnostics. Both compose: validate the
//! boundary ceiling first, then build the layout here.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::vec::Vec;

use crate::common::error::{JpegError, Result};

/// A validated 2-D pixel-buffer layout: geometry in, checked spans out.
///
/// Construction is the proof: an `ImageLayout` existing means its
/// `row_bytes`, `stride` and `total_bytes` were computed without overflow
/// and the total fits `isize::MAX`. Fields are private so no site can
/// recombine the raw factors with ad-hoc arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLayout {
    /// Pixels per row.
    width: usize,
    /// Rows.
    height: usize,
    /// Bytes per pixel (or per sample for planar/high-precision buffers).
    bytes_per_pixel: usize,
    /// Bytes from one row's start to the next. Equals `row_bytes` for a
    /// tightly-packed layout.
    stride: usize,
}

impl ImageLayout {
    /// A tightly-packed layout: `stride == width · bytes_per_pixel`.
    ///
    /// `what` names the buffer in the error, as `try_alloc::try_filled_vec`
    /// does. (Named rather than linked: that helper is crate-private and an
    /// intra-doc link from a public item to it does not resolve.)
    pub fn packed(
        width: usize,
        height: usize,
        bytes_per_pixel: usize,
        what: &'static str,
    ) -> Result<Self> {
        let row_bytes: usize = checked_span(&[width, bytes_per_pixel], what)?;
        let layout = Self {
            width,
            height,
            bytes_per_pixel,
            stride: row_bytes,
        };
        // Validate the total now so a constructed layout cannot fail later.
        let _ = layout.checked_total(what)?;
        Ok(layout)
    }

    /// A row-padded layout. `stride` is in bytes; `0` means tightly packed
    /// (the TurboJPEG pitch convention). A stride below the row's own bytes
    /// is refused — it would make later rows alias earlier ones.
    pub fn strided(
        width: usize,
        height: usize,
        bytes_per_pixel: usize,
        stride: usize,
        what: &'static str,
    ) -> Result<Self> {
        let row_bytes: usize = checked_span(&[width, bytes_per_pixel], what)?;
        let stride: usize = if stride == 0 { row_bytes } else { stride };
        if stride < row_bytes {
            return Err(JpegError::Unsupported(alloc::format!(
                "{what}: stride {stride} is less than the row's {row_bytes} bytes"
            )));
        }
        let layout = Self {
            width,
            height,
            bytes_per_pixel,
            stride,
        };
        let _ = layout.checked_total(what)?;
        Ok(layout)
    }

    /// Pixels per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Bytes one row's pixels occupy, excluding padding.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        // Cannot wrap: `packed`/`strided` proved it at construction.
        self.width * self.bytes_per_pixel
    }

    /// Bytes from one row's start to the next.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The whole buffer's span: `(height − 1) · stride + row_bytes`, or
    /// `0` for zero-area geometry. Proven `<= isize::MAX` at construction,
    /// so the result can size a `Vec` or bound a raw slice directly.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        if self.height == 0 || self.width == 0 || self.bytes_per_pixel == 0 {
            return 0;
        }
        // Cannot wrap: construction ran `checked_total` on these values.
        (self.height - 1) * self.stride + self.row_bytes()
    }

    /// The same geometry with the row padding removed: `height · row_bytes`.
    ///
    /// This is what a strided caller's buffer gets repacked *into*, so it is
    /// the size of the copy rather than of the source. Cannot wrap:
    /// `stride >= row_bytes` makes it no larger than [`Self::total_bytes`],
    /// which construction already proved fits `isize::MAX`.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.height * self.row_bytes()
    }

    /// Byte offset of `row`'s first pixel. `row` must be `< height`;
    /// callers index rows they iterate, so this is a debug-checked
    /// invariant, not input validation.
    #[must_use]
    pub fn row_offset(&self, row: usize) -> usize {
        debug_assert!(row < self.height.max(1), "row {row} out of {}", self.height);
        // Cannot wrap: `row < height` and the full extent was proven.
        row * self.stride
    }

    /// The checked form of [`Self::total_bytes`], used at construction.
    ///
    /// The head — `(height − 1) · stride` — is validated *before* the
    /// zero-area exit, not after, because [`Self::row_offset`] promises
    /// `row · stride` cannot wrap for any `row < height` and this multiply is
    /// the only thing that proves it. Returning early on a zero factor would
    /// leave that promise unbacked for a layout like
    /// `strided(0, 1000, 3, usize::MAX / 2, …)`: `row_bytes` is 0, so the
    /// under-stride test passes, the total is legitimately 0, and yet
    /// `row_offset(999)` wraps. Callers turn that offset into a `ptr::add`.
    fn checked_total(&self, what: &'static str) -> Result<usize> {
        // Checked, not saturating: this module is the one place that may not
        // hand a saturated product to a span (`tests/sizing_arithmetic_gate.rs`
        // enforces that repo-wide). Zero rows means no offsets exist, so
        // there is nothing left to prove.
        let rows_before_last: usize = match self.height.checked_sub(1) {
            Some(rows) => rows,
            None => return Ok(0),
        };
        let head: usize = rows_before_last
            .checked_mul(self.stride)
            .ok_or_else(|| too_big(what))?;
        if head > isize::MAX as usize {
            return Err(JpegError::LimitExceeded {
                what,
                actual: head as u64,
                limit: isize::MAX as u64,
            });
        }
        if self.width == 0 || self.bytes_per_pixel == 0 {
            return Ok(0);
        }
        let total: usize = head
            .checked_add(self.row_bytes())
            .ok_or_else(|| too_big(what))?;
        if total > isize::MAX as usize {
            return Err(JpegError::LimitExceeded {
                what,
                actual: total as u64,
                limit: isize::MAX as u64,
            });
        }
        Ok(total)
    }
}

/// Product of arbitrary span factors, refusing wrap and totals past
/// `isize::MAX` — the free-function form for spans that are not 2-D
/// (plane counts, block grids, coefficient arrays).
///
/// The empty product is `1`, as arithmetic requires; no caller passes an
/// empty slice, and one that did would be asking for a factorless span.
///
/// This absorbed `progressive_output`'s private `checked_plane_size`, which
/// was contract-identical. Chunk 2 deleted that twin and repointed its twelve
/// call sites here, so the formula has one implementation rather than two that
/// could drift.
#[must_use = "the checked span is the result; discarding it discards the check"]
pub fn checked_span(factors: &[usize], what: &'static str) -> Result<usize> {
    let mut total: usize = 1;
    for factor in factors {
        total = total.checked_mul(*factor).ok_or_else(|| too_big(what))?;
    }
    if total > isize::MAX as usize {
        return Err(JpegError::LimitExceeded {
            what,
            actual: total as u64,
            limit: isize::MAX as u64,
        });
    }
    Ok(total)
}

/// The overflow error, shaped like the existing guards': the actual value
/// is unrepresentable, so report the limit it broke.
fn too_big(what: &'static str) -> JpegError {
    JpegError::LimitExceeded {
        what,
        actual: u64::MAX,
        limit: isize::MAX as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_total_is_the_plain_product() {
        let l = ImageLayout::packed(640, 480, 3, "test").expect("small geometry");
        assert_eq!(l.row_bytes(), 1920);
        assert_eq!(l.stride(), 1920);
        assert_eq!(l.total_bytes(), 640 * 480 * 3);
    }

    #[test]
    fn zero_stride_means_tightly_packed() {
        let l = ImageLayout::strided(100, 10, 4, 0, "test").expect("pitch 0");
        assert_eq!(l.stride(), 400);
        assert_eq!(l.total_bytes(), 4000);
    }

    #[test]
    fn strided_total_charges_padding_for_all_but_the_last_row() {
        let l = ImageLayout::strided(100, 10, 1, 128, "test").expect("padded");
        assert_eq!(l.total_bytes(), 9 * 128 + 100);
        assert_eq!(l.row_offset(9), 9 * 128);
    }

    #[test]
    fn packed_bytes_drops_the_padding_total_bytes_charges() {
        let l = ImageLayout::strided(100, 10, 1, 128, "test").expect("padded");
        assert_eq!(l.total_bytes(), 9 * 128 + 100);
        assert_eq!(l.packed_bytes(), 10 * 100);
        // For a layout that is already tight the two agree.
        let tight = ImageLayout::packed(100, 10, 1, "test").expect("tight");
        assert_eq!(tight.packed_bytes(), tight.total_bytes());
    }

    #[test]
    fn stride_below_row_bytes_is_refused() {
        let err = ImageLayout::strided(100, 10, 3, 299, "test").expect_err("under-stride");
        assert!(
            matches!(err, JpegError::Unsupported(_)),
            "want Unsupported, got {err:?}"
        );
    }

    #[test]
    fn wrapping_products_are_refused_not_wrapped() {
        let err = ImageLayout::packed(usize::MAX / 2, 3, 1, "test").expect_err("wraps");
        assert!(matches!(err, JpegError::LimitExceeded { .. }));
        let err = ImageLayout::strided(4, usize::MAX / 2, 1, usize::MAX / 4, "test")
            .expect_err("strided head wraps");
        assert!(matches!(err, JpegError::LimitExceeded { .. }));
    }

    /// The `isize::MAX` arm distinct from the wrap arm: a product that fits
    /// `usize` but exceeds the allocator/raw-slice ceiling. On 64-bit that
    /// needs huge factors; the same shape is what a 32-bit target hits at
    /// ~2 GiB, which `adversarial_geometry_matches_the_u128_model` below
    /// covers on whatever width it runs under.
    #[test]
    fn totals_past_isize_max_are_refused() {
        let err =
            ImageLayout::packed(isize::MAX as usize, 1, 2, "test").expect_err("over isize::MAX");
        assert!(matches!(
            err,
            JpegError::LimitExceeded { limit, .. } if limit == isize::MAX as u64
        ));
    }

    #[test]
    fn zero_area_is_zero_bytes_not_an_error() {
        let l = ImageLayout::packed(0, 480, 3, "test").expect("zero width");
        assert_eq!(l.total_bytes(), 0);
        let l = ImageLayout::strided(100, 0, 3, 1024, "test").expect("zero height");
        assert_eq!(l.total_bytes(), 0);
    }

    /// Zero area does *not* excuse an unrepresentable row offset. A zero row
    /// makes the under-stride test vacuous and the total honestly 0, but
    /// `row_offset` is still defined for every `row < height` and callers
    /// turn it into a `ptr::add` — so `(height - 1) * stride` has to be
    /// proven whatever the total is.
    #[test]
    fn zero_area_with_an_unrepresentable_row_offset_is_refused() {
        let err = ImageLayout::strided(0, 1000, 3, usize::MAX / 2, "test")
            .expect_err("row offsets wrap even though the total is 0");
        assert!(
            matches!(err, JpegError::LimitExceeded { .. }),
            "want LimitExceeded, got {err:?}"
        );
        // The same geometry with a stride the offsets can actually reach is
        // still fine, so the guard is about representability, not zero area.
        let l = ImageLayout::strided(0, 1000, 3, 64, "test").expect("small stride");
        assert_eq!(l.total_bytes(), 0);
        assert_eq!(l.row_offset(999), 999 * 64);
    }

    #[test]
    fn checked_span_matches_the_old_checked_plane_size_contract() {
        assert_eq!(checked_span(&[4, 5, 6], "test").expect("small"), 120);
        assert!(checked_span(&[usize::MAX, 2], "test").is_err());
        assert!(checked_span(&[], "test").expect("empty is 1") == 1);
    }

    /// Reference model in `u128`, which cannot overflow for any `usize`
    /// factors: the strided total, or `None` where the layout must refuse.
    fn model(w: usize, h: usize, bpp: usize, stride: usize) -> Option<u128> {
        let row: u128 = (w as u128) * (bpp as u128);
        // Validation before the zero-area early-out, matching `strided`:
        // a row that cannot exist as a slice (or an under-stride) is
        // invalid geometry even when zero rows would span zero bytes —
        // callers use `row_bytes` for per-row arithmetic regardless of
        // height. On 32-bit this arm is what the sweep actually hits.
        if row > isize::MAX as u128 {
            return None;
        }
        let stride: u128 = if stride == 0 { row } else { stride as u128 };
        if stride < row {
            return None;
        }
        // Also before the zero-area exit: `row_offset` is defined for every
        // `row < height` whatever the total is, so `(h - 1) * stride` must be
        // representable even when the total is zero. Without this arm the
        // model accepts `(0, 1000, 3, usize::MAX / 2)`, whose `row_offset`
        // wraps — the hole this sweep exists to catch.
        if h == 0 {
            return Some(0);
        }
        let head: u128 = (h as u128 - 1) * stride;
        if head > isize::MAX as u128 {
            return None;
        }
        if w == 0 || bpp == 0 {
            return Some(0);
        }
        let total: u128 = head + row;
        (total <= isize::MAX as u128).then_some(total)
    }

    /// P4-139 criterion 5: adversarial geometry swept against the `u128`
    /// model on whatever pointer width this test runs under — the same
    /// assertions hold on the 64-bit legs and the 32-bit ones
    /// (armv7/wasm32 run the lib tests), where the interesting boundary
    /// moves to ~2 GiB. Deterministic Mulberry32 sweep plus a hand-picked
    /// edge matrix; no `proptest` dependency, matching this repo's
    /// preference for reproducible CI.
    #[test]
    fn adversarial_geometry_matches_the_u128_model() {
        // Hand-picked edges: JPEG dimension cap, u32-wrapping products,
        // isize::MAX-adjacent totals, stride just under/over row bytes.
        let edges: &[(usize, usize, usize, usize)] = &[
            (65535, 65535, 4, 0),
            (65535, 65535, 2, 0),
            (65500, 40000, 1, 0), // the checked_staging_span 32-bit example
            (1 << 16, 1 << 16, 1, 0),
            (usize::MAX, 1, 1, 0),
            (1, usize::MAX, 1, 0),
            (usize::MAX / 2, 2, 1, 0),
            (isize::MAX as usize, 1, 1, 0),
            (isize::MAX as usize / 2, 1, 2, 0),
            (100, 10, 3, 299), // stride one under row bytes
            (100, 10, 3, 300), // stride exactly row bytes
            (100, 10, 3, 301), // stride one over
            (3, 3, 1, usize::MAX / 2),
            (0, usize::MAX, usize::MAX, 0),
            (65535, 1, 0, 17), // zero bpp, nonzero stride
            // Zero area with a huge *explicit* stride: the total is honestly
            // 0, but `row_offset(height - 1)` is not, so these must be
            // refused. The `(0, usize::MAX, usize::MAX, 0)` row above misses
            // it — stride 0 normalises to `row_bytes`, which is also 0.
            (0, 1000, 3, usize::MAX / 2),
            (100, 0, 3, usize::MAX / 2),
            (100, 3, 0, usize::MAX / 2),
        ];
        // Mulberry32, the same generator the SIMD parity tests use, so the
        // sweep is identical on every platform and every run.
        let mut state: u32 = 0x9e3779b9;
        let mut next = move || -> u32 {
            state = state.wrapping_add(0x6D2B79F5);
            let mut z: u32 = state;
            z = (z ^ (z >> 15)).wrapping_mul(z | 1);
            z ^= z.wrapping_add((z ^ (z >> 7)).wrapping_mul(z | 61));
            z ^ (z >> 14)
        };
        let mut cases: Vec<(usize, usize, usize, usize)> = edges.to_vec();
        for _ in 0..4096 {
            // Bias toward boundary-magnitude values: mix small dims, JPEG
            // caps, and width-dependent extremes.
            let pick = |r: u32| -> usize {
                match r % 6 {
                    0 => (r as usize) % 128,
                    1 => 65535 - ((r as usize) % 64),
                    // Shifts bounded by the pointer width so the sweep is
                    // valid on 32-bit targets too.
                    2 => (usize::MAX >> ((r % (usize::BITS - 1)) + 1)).max(1),
                    3 => (isize::MAX as usize) >> (r % usize::BITS),
                    4 => 1usize << (r % (usize::BITS - 1)),
                    _ => (r as usize).max(1),
                }
            };
            let w = pick(next());
            let h = pick(next());
            let bpp = [0usize, 1, 2, 3, 4, 8][(next() % 6) as usize];
            let stride = match next() % 3 {
                0 => 0,
                1 => pick(next()),
                // `wrapping_mul`, not `saturating_mul`: this is an adversarial
                // *input* generator, so a wrapped stride is a case worth
                // feeding in — and saturating span arithmetic is barred
                // repo-wide by `tests/sizing_arithmetic_gate.rs`, including
                // here, where it would read as an endorsement.
                _ => (w.wrapping_mul(bpp))
                    .wrapping_add((next() % 3) as usize)
                    .wrapping_sub(1),
            };
            cases.push((w, h, bpp, stride));
        }

        for &(w, h, bpp, stride) in &cases {
            let got = ImageLayout::strided(w, h, bpp, stride, "sweep");
            match model(w, h, bpp, stride) {
                Some(total) => {
                    let layout = got.unwrap_or_else(|e| {
                        panic!("({w},{h},{bpp},{stride}) must be accepted, got {e:?}")
                    });
                    assert_eq!(
                        layout.total_bytes() as u128,
                        total,
                        "total mismatch at ({w},{h},{bpp},{stride})"
                    );
                }
                None => {
                    assert!(
                        got.is_err(),
                        "({w},{h},{bpp},{stride}) must be refused (model says overflow/under-stride)"
                    );
                }
            }
        }
    }
}
