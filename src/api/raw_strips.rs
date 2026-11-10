//! Raw component planes, one iMCU row at a time, borrowed from the reader.
//!
//! [`crate::Decoder::into_raw_strips`] returns a [`RawStripReader`]. Each
//! [`RawStripReader::next_strip`] call decodes the next iMCU row and lends it
//! out as a [`RawStrip`]: one [`RawStripPlane`] per component, at the
//! component's own (possibly subsampled) resolution, with no upsampling and
//! no color conversion. The caller copies the rows wherever it wants them
//! (its own planes at its own stride, an interleaved NV12 chroma plane, a GPU
//! upload, a Y4M writer), so no layout is imposed and no whole-image buffer
//! is required. This is libjpeg's `jpeg_read_raw_data`, which also returns
//! one iMCU row per call, without the row-pointer arrays.
//!
//! A strip borrows the reader, so the next call cannot happen while it is
//! alive; that is what lets the reader reuse one scratch strip instead of
//! allocating per row. For the same reason this is a method and not an
//! [`Iterator`]: a standard iterator cannot lend items out of itself.
//!
//! **Working set.** For a single-scan Huffman stream (baseline or extended
//! sequential, interleaved or single-component) the reader decodes each row
//! on demand and holds one iMCU row of every component. A progressive or
//! multi-scan stream cannot produce any final row until all its scans are
//! read, and arithmetic-coded streams are not streamed yet (P4-268), so for
//! all three the reader decodes the whole frame on its first strip (as
//! [`crate::Decoder::decode_raw`]) and lends strips out of those planes.
//!
//! **Samples.** A strip's active rows ([`RawStripPlane::rows`]) are identical
//! to libjpeg's raw output for the same rows. Its padded rows
//! ([`RawStripPlane::padded_rows`]) extend to the MCU-aligned width and the
//! full strip height; the samples beyond the active area are the decoder's
//! IDCT of the edge and MCU padding blocks (or 128 where a single-component
//! scan codes no block), identical to [`crate::RawImage`]'s planes, and are
//! not part of the image.
//!
//! **Limits.** Like `decode_raw`, no block smoothing is applied to
//! progressive streams that stop short of full refinement (P4-249), and
//! only 8-bit DCT frames at 1/1 scale without cropping are accepted.

use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::decode::pipeline::{Decoder, RawStreamState};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

/// Frame geometry of a raw strip decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawGeometry {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Number of strips (iMCU rows): `ceil(height / strip_height)`.
    pub strip_count: usize,
    /// Image rows each strip covers: `max_v_samp_factor * 8`.
    pub strip_height: usize,
    /// One entry per frame component, in frame order.
    pub components: Vec<RawComponentGeometry>,
}

/// Geometry of one component's raw plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawComponentGeometry {
    /// Component identifier from the SOF marker.
    pub id: u8,
    /// Horizontal sampling factor.
    pub horizontal_sampling: usize,
    /// Vertical sampling factor.
    pub vertical_sampling: usize,
    /// Active width in samples: libjpeg's `downsampled_width`,
    /// `ceil(image_width * h / max_h)`.
    pub width: usize,
    /// Active height in rows: libjpeg's `downsampled_height`.
    pub height: usize,
    /// Width of a padded row, aligned to the MCU grid. Also the distance
    /// between rows in [`RawStripPlane::padded_data`].
    pub padded_width: usize,
    /// Rows of this component in one strip: `v_samp_factor * 8`.
    pub rows_per_strip: usize,
}

/// Where the strips come from (see the module docs). The decoder lives in
/// the variants that still need it, so no state can lack one.
enum Source<'a> {
    /// Single-scan Huffman stream, decoded one row per call.
    Streaming {
        decoder: Decoder<'a>,
        state: RawStreamState<'a>,
    },
    /// Multi-scan stream whose whole-frame decode runs on the first strip.
    Pending(Decoder<'a>),
    /// The whole-frame decode's warnings; its planes are in `buffers`.
    Buffered(Vec<DecodeWarning>),
}

/// Pull-based reader of raw component planes, one iMCU row per call.
///
/// Created by [`crate::Decoder::into_raw_strips`]; see the
/// [module documentation](crate::api::raw_strips) for what a strip holds and
/// what memory the reader keeps.
///
/// ```no_run
/// # fn main() -> libjpeg_turbo_rs::Result<()> {
/// # let jpeg: Vec<u8> = Vec::new();
/// use libjpeg_turbo_rs::Decoder;
///
/// let mut reader = Decoder::new(&jpeg)?.into_raw_strips()?;
/// let mut planes: Vec<Vec<u8>> = reader
///     .geometry()
///     .components
///     .iter()
///     .map(|c| Vec::with_capacity(c.width * c.height))
///     .collect();
/// while let Some(strip) = reader.next_strip()? {
///     for plane in strip.planes() {
///         for row in plane.rows() {
///             planes[plane.component()].extend_from_slice(row);
///         }
///     }
/// }
/// let _warnings = reader.finish()?;
/// # Ok(())
/// # }
/// ```
pub struct RawStripReader<'a> {
    extraneous_bytes: usize,
    geometry: RawGeometry,
    source: Source<'a>,
    /// One strip per component (streaming) or whole planes (buffered).
    buffers: Vec<Vec<u8>>,
    next_index: usize,
    /// Set by the first error; every later call reports it again rather than
    /// resuming from a half-decoded state.
    failure: Option<String>,
}

impl<'a> core::fmt::Debug for RawStripReader<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RawStripReader")
            .field("geometry", &self.geometry)
            .field("next_index", &self.next_index)
            .field("failed", &self.failure.is_some())
            .finish_non_exhaustive()
    }
}

impl<'a> RawStripReader<'a> {
    pub(crate) fn new(decoder: Decoder<'a>) -> Result<Self> {
        let geometry: RawGeometry = decoder.raw_strip_geometry()?;
        let extraneous_bytes: usize = decoder.raw_strip_extraneous_bytes();
        let (source, buffers): (Source<'a>, Vec<Vec<u8>>) =
            match decoder.raw_stream_begin(&geometry)? {
                Some(state) => (
                    Source::Streaming { decoder, state },
                    Decoder::raw_stream_scratch(&geometry)?,
                ),
                None => (Source::Pending(decoder), Vec::new()),
            };
        Ok(Self {
            extraneous_bytes,
            geometry,
            source,
            buffers,
            next_index: 0,
            failure: None,
        })
    }

    /// The frame and per-component geometry, available before any strip is
    /// decoded, so destination buffers can be sized up front.
    pub fn geometry(&self) -> &RawGeometry {
        &self.geometry
    }

    /// Strips not yet returned.
    pub fn strips_remaining(&self) -> usize {
        self.geometry.strip_count - self.next_index
    }

    /// Decode and lend the next strip, or `Ok(None)` after the last one.
    ///
    /// # Errors
    ///
    /// A corrupt stream fails at the strip that holds the damage (for a
    /// streamed decode the earlier strips have already been returned), unless
    /// the decoder is lenient ([`Decoder::set_lenient`]), in which case the
    /// damage is concealed and reported by [`Self::finish`]. The first error
    /// is returned as it occurred; after it, every further call (including
    /// [`Self::finish`]) fails with `CorruptData` naming that first error,
    /// rather than resuming from a half-decoded state.
    pub fn next_strip(&mut self) -> Result<Option<RawStrip<'_>>> {
        if let Some(message) = &self.failure {
            return Err(JpegError::CorruptData(format!(
                "raw strip reader stopped by an earlier error: {message}"
            )));
        }
        if self.next_index >= self.geometry.strip_count {
            return Ok(None);
        }
        if let Err(error) = self.decode_strip(self.next_index) {
            self.failure = Some(error.to_string());
            return Err(error);
        }
        let index: usize = self.next_index;
        self.next_index += 1;
        Ok(Some(RawStrip {
            index,
            geometry: &self.geometry,
            buffers: &self.buffers,
            holds_whole_planes: !matches!(self.source, Source::Streaming { .. }),
        }))
    }

    fn decode_strip(&mut self, index: usize) -> Result<()> {
        if let Source::Streaming { decoder, state } = &mut self.source {
            return decoder.raw_stream_strip(state, &self.geometry, index, &mut self.buffers);
        }
        // Every strip after the first is already in `buffers`. Checked before
        // the `replace` below, which would otherwise discard the fallback
        // decode's warnings on each of those strips.
        if !matches!(self.source, Source::Pending(_)) {
            return Ok(());
        }
        // Take the decoder out of `Pending`. The placeholder is only seen if
        // the decode below fails, and then `failure` fuses the reader.
        let Source::Pending(decoder) =
            core::mem::replace(&mut self.source, Source::Buffered(Vec::new()))
        else {
            return Ok(());
        };
        // `decode_raw` re-runs the marker, limit and precision checks `new`
        // already passed: header-only work, and keeping it means the planes
        // are exactly `decode_raw`'s.
        let (raw, warnings): (crate::RawImage, Vec<DecodeWarning>) =
            decoder.decode_raw_scaled_with_warnings(8)?;
        // The geometry came from the same header; a disagreement would
        // misindex every strip, so it is checked, not assumed.
        let is_consistent: bool = raw.planes.len() == self.geometry.components.len()
            && self.geometry.components.iter().enumerate().all(|(ci, c)| {
                raw.plane_widths[ci] == c.padded_width
                    && raw.plane_heights[ci] == self.geometry.strip_count * c.rows_per_strip
                    && raw.planes[ci].len() == c.padded_width * raw.plane_heights[ci]
            });
        if !is_consistent {
            return Err(JpegError::Unsupported(String::from(
                "raw decode geometry differs from the strip geometry",
            )));
        }
        self.buffers = raw.planes;
        self.source = Source::Buffered(warnings);
        Ok(())
    }

    /// Call `f` with every remaining strip in order, then [`Self::finish`].
    /// An error from `f` stops the decode and is returned as is.
    pub fn for_each_strip<F>(mut self, mut f: F) -> Result<Vec<DecodeWarning>>
    where
        F: FnMut(RawStrip<'_>) -> Result<()>,
    {
        while let Some(strip) = self.next_strip()? {
            f(strip)?;
        }
        self.finish()
    }

    /// Decode any strips not yet read (discarding them) and return the
    /// warnings the decode recovered from, in the order and form
    /// [`Decoder::decode_raw_with_warnings`] reports them. To abandon a
    /// decode without paying for the rest, drop the reader instead.
    pub fn finish(mut self) -> Result<Vec<DecodeWarning>> {
        while self.next_strip()?.is_some() {}
        let mut warnings: Vec<DecodeWarning> = match self.source {
            Source::Streaming { state, .. } => state.finish(),
            Source::Buffered(warnings) => warnings,
            // Not reached: `raw_strip_geometry` refuses an empty frame, so
            // there is a first strip, and the drain above decoded it.
            Source::Pending(_) => Vec::new(),
        };
        if self.extraneous_bytes != 0 {
            warnings
                .try_reserve(1)
                .map_err(|_| JpegError::AllocationFailed {
                    what: "marker warning",
                    bytes: core::mem::size_of::<DecodeWarning>() as u64,
                })?;
            warnings.push(DecodeWarning::ExtraneousData {
                bytes: self.extraneous_bytes,
            });
        }
        Ok(warnings)
    }
}

/// One iMCU row of every component, borrowed from a [`RawStripReader`].
#[derive(Debug, Clone, Copy)]
pub struct RawStrip<'r> {
    index: usize,
    geometry: &'r RawGeometry,
    buffers: &'r [Vec<u8>],
    /// `buffers` are whole planes (fallback decode) rather than one strip.
    holds_whole_planes: bool,
}

impl<'r> RawStrip<'r> {
    /// This strip's position, `0..strip_count`.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Component `component`'s rows in this strip, or `None` past the last
    /// component.
    pub fn plane(&self, component: usize) -> Option<RawStripPlane<'r>> {
        (component < self.geometry.components.len()).then(|| self.plane_at(component))
    }

    /// Every component's rows in this strip, in frame order.
    pub fn planes(&self) -> impl ExactSizeIterator<Item = RawStripPlane<'r>> + 'r {
        let strip: RawStrip<'r> = *self;
        (0..self.geometry.components.len()).map(move |ci| strip.plane_at(ci))
    }

    /// `component` must be below the component count. The slice cannot go
    /// out of range: the reader sized each streamed buffer to exactly one
    /// strip (`raw_stream_strip` re-checks it) and checked a fallback
    /// decode's planes against `strip_count` strips before lending any.
    fn plane_at(&self, component: usize) -> RawStripPlane<'r> {
        let comp: &RawComponentGeometry = &self.geometry.components[component];
        let strip_bytes: usize = comp.padded_width * comp.rows_per_strip;
        let start: usize = if self.holds_whole_planes {
            self.index * strip_bytes
        } else {
            0
        };
        let first_row: usize = self.index * comp.rows_per_strip;
        RawStripPlane {
            component,
            first_row,
            active_rows: comp
                .rows_per_strip
                .min(comp.height.saturating_sub(first_row)),
            width: comp.width,
            padded_width: comp.padded_width,
            data: &self.buffers[component][start..start + strip_bytes],
        }
    }
}

/// One component's rows within a [`RawStrip`].
#[derive(Debug, Clone, Copy)]
pub struct RawStripPlane<'r> {
    component: usize,
    first_row: usize,
    active_rows: usize,
    width: usize,
    padded_width: usize,
    data: &'r [u8],
}

impl<'r> RawStripPlane<'r> {
    /// Index of the component in frame order.
    pub fn component(&self) -> usize {
        self.component
    }

    /// Row of the component's plane that this strip's first row is.
    pub fn first_row(&self) -> usize {
        self.first_row
    }

    /// Active width in samples (the component's `width`).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Padded row width, also the stride of [`Self::padded_data`].
    pub fn padded_width(&self) -> usize {
        self.padded_width
    }

    /// The strip's rows inside the image, each `width` samples. Fewer than
    /// `rows_per_strip` on the last strip when the height is not
    /// MCU-aligned.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &'r [u8]> + 'r {
        let width: usize = self.width;
        self.data
            .chunks_exact(self.padded_width)
            .take(self.active_rows)
            .map(move |row| &row[..width])
    }

    /// Every row of the strip, each `padded_width` samples, MCU padding
    /// included (see the module docs for what the padding holds).
    pub fn padded_rows(&self) -> core::slice::ChunksExact<'r, u8> {
        self.data.chunks_exact(self.padded_width)
    }

    /// The whole strip, `rows_per_strip` rows of `padded_width` samples.
    pub fn padded_data(&self) -> &'r [u8] {
        self.data
    }
}
