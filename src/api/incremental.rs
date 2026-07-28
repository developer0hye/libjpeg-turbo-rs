//! Bounded-input-memory decode from a [`std::io::Read`] source
//! (P4-58, issue #357).
//!
//! Interleaved single-scan Huffman baseline streams — the overwhelming
//! majority of real-world JPEGs — decode from a **sliding window** of
//! compressed bytes: the window grows only while a single MCU row's
//! entropy data straddles its end, and consumed bytes are dropped as
//! rows commit, so peak input memory is a fixed window rather than the
//! whole stream. Progressive, non-interleaved (including grayscale),
//! arithmetic, lossless, and 12/16-bit streams require the full entropy
//! stream by construction; for those this entry point transparently
//! falls back to buffering, exactly like
//! [`decompress_from_reader`](crate::stream::decompress_from_reader).
//!
//! The window/boundary machinery is shared with the C-ABI
//! `jpeg_consume_input` suspension core
//! ([`crate::decode::boundary`]) — one streaming mechanism, not
//! three (the P4-26 co-design constraint).

use crate::common::error::{JpegError, Result};
use crate::decode::boundary;
use crate::decode::pipeline::{Decoder, Image};
use std::io::Read;

/// Bytes requested from the reader per refill.
const READ_CHUNK: usize = 64 * 1024;

/// Minimum bytes to accumulate after a starved row before retrying it —
/// a retry re-decodes the whole row, so tiny refills would be
/// quadratic against drip-feed readers.
const MIN_RETRY_REFILL: usize = 4 * 1024;

/// Hard cap on the buffered header prefix (SOI through the first SOS),
/// mirroring the C-ABI suspension core's 256 MiB body cap: a stream
/// that never presents an SOS must not grow the buffer unboundedly.
const MAX_HEADER_BYTES: usize = 256 * 1024 * 1024;

/// Decode a JPEG from a byte stream with bounded input memory.
///
/// For interleaved baseline streams the input window stays small and
/// independent of the compressed size (measured in
/// `tests/regression_issue_357_incremental_reader.rs`); other coding
/// modes fall back to full buffering. Output is identical to
/// [`decompress`](crate::decompress) on the same bytes in every case.
///
/// Decode options (output format, scaling, cropping) are not yet
/// plumbed through this entry point — it decodes with defaults, like
/// `decompress`. Use the slice API when you need options.
pub fn decompress_from_reader_incremental<R: Read>(reader: R) -> Result<Image> {
    decode_incremental(reader).map(|(image, _)| image)
}

/// [`decompress_from_reader_incremental`] plus the peak number of
/// compressed bytes held at any point — the observable that P4-58's
/// bounded-memory acceptance test asserts on. Exposed (rather than a
/// test-only hook) so callers tuning window behaviour can measure too.
pub fn decompress_from_reader_incremental_instrumented<R: Read>(
    reader: R,
) -> Result<(Image, usize)> {
    decode_incremental(reader)
}

/// Read up to [`READ_CHUNK`] more bytes into `buf`. Returns the byte
/// count (0 = end of stream); propagates reader errors as
/// [`JpegError::Io`].
fn read_some<R: Read>(reader: &mut R, buf: &mut Vec<u8>) -> Result<usize> {
    // Take + read_to_end loops over however many small reads the
    // source needs to produce up to READ_CHUNK bytes — a drip-feed
    // reader costs one function call per drip, not one buffer resize
    // (the naive resize-then-read version made a 1-byte/read source
    // memset 64 KiB per byte).
    reader
        .take(READ_CHUNK as u64)
        .read_to_end(buf)
        .map_err(JpegError::Io)
}

fn decode_incremental<R: Read>(mut reader: R) -> Result<(Image, usize)> {
    let mut buf: Vec<u8> = Vec::new();
    let mut peak: usize = 0;
    let mut source_done: bool = false;

    // Phase 1: accumulate the header prefix through the first SOS. The
    // shared boundary scanner reports when the prefix is complete.
    let entropy_start: usize = loop {
        if let Some(off) = boundary::find_first_sos(&buf) {
            break off;
        }
        if source_done {
            return Err(JpegError::UnexpectedEof);
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err(JpegError::CorruptData(
                "no SOS marker within the 256 MiB header cap".to_string(),
            ));
        }
        let n: usize = read_some(&mut reader, &mut buf)?;
        peak = peak.max(buf.len());
        if n == 0 {
            source_done = true;
        }
    };

    // Phase 2: eligibility probe on the header prefix alone. A parse
    // error here is NOT terminal: progressive and non-interleaved
    // streams walk every scan during header parse, which runs past the
    // prefix — exactly the shapes that take the buffered fallback,
    // where the full-stream parse delivers the real verdict.
    let eligible: bool = match Decoder::new(&buf[..entropy_start]) {
        Ok(probe) => probe.baseline_stream_eligible(),
        Err(_) => false,
    };

    if !eligible {
        // Fallback: buffer the remainder and take the slice path. This
        // is the documented behaviour for stream shapes whose entropy
        // layout requires the full stream (progressive multi-pass,
        // non-interleaved block rasters, arithmetic conditioning).
        loop {
            let n: usize = read_some(&mut reader, &mut buf)?;
            peak = peak.max(buf.len());
            if n == 0 {
                break;
            }
        }
        let image: Image = crate::api::high_level::decompress(&buf)?;
        return Ok((image, peak));
    }

    // Phase 3: windowed row decode. The header prefix moves to its own
    // allocation so `buf` can shrink to just the entropy window.
    let header: Vec<u8> = buf[..entropy_start].to_vec();
    buf.drain(..entropy_start);
    let decoder: Decoder = Decoder::new(&header)?;
    let mut st = decoder.baseline_stream_begin()?;
    let mut is_final: bool = source_done;

    loop {
        peak = peak.max(header.len() + buf.len());
        if decoder.baseline_stream_step(&buf, is_final, &mut st)? {
            break;
        }
        if is_final {
            // Defensive: a final window never starves (the step errors
            // on truncation instead), so this is unreachable — but a
            // livelock would be worse than a spurious error.
            return Err(JpegError::UnexpectedEof);
        }
        // Drop committed bytes from the window front, then refill.
        let consumable: usize = st.consumable();
        if consumable > 0 {
            buf.drain(..consumable);
            st.rebase(consumable);
        }
        // Accumulate a meaningful refill before retrying: each retry
        // re-decodes a full MCU row, so advancing one byte at a time
        // against a drip-feed reader would be quadratic in row size.
        let mut refilled: usize = 0;
        while refilled < MIN_RETRY_REFILL {
            let n: usize = read_some(&mut reader, &mut buf)?;
            peak = peak.max(header.len() + buf.len());
            if n == 0 {
                is_final = true;
                break;
            }
            refilled += n;
        }
    }

    decoder.set_prefilled_baseline_planes(st.into_planes());
    let image: Image = decoder.decode_image()?;
    Ok((image, peak))
}
