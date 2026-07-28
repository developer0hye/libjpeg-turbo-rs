//! Bounded-input-memory decode from a [`std::io::Read`] source
//! (P4-58, issue #357).
//!
//! Interleaved single-scan Huffman baseline streams — the overwhelming
//! majority of real-world JPEGs — decode from a **sliding window** of
//! compressed bytes: the window grows only while a single MCU row's
//! entropy data straddles its end, and consumed bytes are dropped as
//! rows commit, so peak input memory is a fixed window rather than the
//! whole stream. Progressive, non-interleaved multi-component,
//! arithmetic, lossless, and 12/16-bit streams require the full entropy
//! stream by construction, and single-component (grayscale) streams
//! decode through the one-block raster of P4-27, which the row loop
//! does not model; for all of those this entry point transparently
//! falls back to buffering, exactly like
//! [`decompress_from_reader`](crate::stream::decompress_from_reader).
//!
//! The marker-boundary scanner is shared with the C-ABI
//! `jpeg_consume_input` suspension core ([`crate::decode::boundary`]),
//! so this is not a third independent streaming mechanism (the P4-26
//! co-design constraint); the sliding window and the row checkpoints
//! are specific to this path.

use crate::common::error::{JpegError, Result};
use crate::decode::boundary;
use crate::decode::pipeline::{Decoder, Image};
use std::io::Read;

/// Bytes requested from the reader per refill.
const READ_CHUNK: usize = 64 * 1024;

/// Refill batching floor for FAST sources only: while `read()` keeps
/// returning full chunks, accumulate this much before re-attempting a
/// starved row. A source that short-reads breaks the batch immediately
/// (we cannot wait for more without risking a block on a stream that
/// already delivered the whole image), so this constant does NOT
/// protect against drip-feed retry cost — see the retry-granularity
/// note on [`decompress_from_reader_incremental`].
const MIN_RETRY_REFILL: usize = 4 * 1024;

/// Hard cap on the buffered header prefix (SOI through the first SOS),
/// mirroring the C-ABI suspension core's 256 MiB body cap: a stream
/// that never presents an SOS must not grow the buffer unboundedly.
const MAX_HEADER_BYTES: usize = 256 * 1024 * 1024;

/// Hard cap on the entropy window while a single MCU row refuses to
/// commit (codex P1: an attacker streaming endless `0xFF` fill after
/// SOS would otherwise grow the window to OOM). A legal row's entropy
/// is bounded far below this — worst case ~18 MiB for a maximum-width
/// 4:4:4 frame — so hitting the cap is a corrupt/hostile stream.
const MAX_WINDOW_BYTES: usize = 64 * 1024 * 1024;

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
///
/// **Retry granularity.** A starved MCU row is re-attempted after every
/// short read (waiting for more would block on a source that already
/// delivered the image), so retry cost scales with how few bytes each
/// `read()` returns: a pathological 1-byte-per-read source re-decodes
/// each row once per byte. Sources that deliver kilobytes per read —
/// files, buffered sockets — are unaffected.
pub fn decompress_from_reader_incremental<R: Read>(reader: R) -> Result<Image> {
    decode_incremental(reader).map(|(image, _)| image)
}

/// [`decompress_from_reader_incremental`] plus the peak input storage
/// held at any point, measured as allocation CAPACITY (not just live
/// bytes): header prefix + entropy window + the fixed 64 KiB
/// read-staging buffer. This is the observable P4-58's bounded-memory
/// acceptance test asserts on. Exposed (rather than a test-only hook)
/// so callers tuning window behaviour can measure too.
pub fn decompress_from_reader_incremental_instrumented<R: Read>(
    reader: R,
) -> Result<(Image, usize)> {
    decode_incremental(reader)
}

/// One `read` call into `scratch`, appended to `buf`. Returns the byte
/// count (0 = end of stream); propagates reader errors as
/// [`JpegError::Io`]. Deliberately a SINGLE read: looping until a
/// fixed chunk fills (`Take::read_to_end` style) would block on a
/// persistent socket that has already delivered the complete image but
/// stays open (codex P1).
fn read_some<R: Read>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    scratch: &mut [u8; READ_CHUNK],
) -> Result<usize> {
    let n: usize = loop {
        match reader.read(&mut scratch[..]) {
            Ok(n) => break n,
            // The Read contract expects callers to retry Interrupted
            // (read_to_end did so implicitly); anything else is real.
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(JpegError::Io(e)),
        }
    };
    buf.extend_from_slice(&scratch[..n]);
    Ok(n)
}

fn decode_incremental<R: Read>(mut reader: R) -> Result<(Image, usize)> {
    let mut buf: Vec<u8> = Vec::new();
    let mut scratch: Box<[u8; READ_CHUNK]> = Box::new([0u8; READ_CHUNK]);
    // The staging buffer holds compressed bytes too; count it from the
    // start so the public metric reflects every byte of input storage
    // (codex P2).
    let mut peak: usize = READ_CHUNK;
    let mut source_done: bool = false;

    // Phase 1: accumulate the header prefix through the first SOS. The
    // shared boundary scanner reports when the prefix is complete. A
    // stream that ends with no SOS (non-JPEG bytes, tables-only,
    // truncated header) falls through to the buffered path so the
    // error VARIANT matches `decompress` on the same bytes exactly
    // (reviewer P2: this used to blanket-report UnexpectedEof).
    let entropy_start: Option<usize> = loop {
        if let Some(off) = boundary::find_first_sos(&buf) {
            break Some(off);
        }
        if source_done {
            break None;
        }
        // Guard BEFORE appending: one more chunk at the cap boundary
        // would trigger Vec's amortized doubling (~2x the cap in one
        // allocation request), defeating the limit (codex P2).
        if buf.len() + READ_CHUNK > MAX_HEADER_BYTES {
            return Err(JpegError::CorruptData(
                "no SOS marker within the 256 MiB header cap".to_string(),
            ));
        }
        let n: usize = read_some(&mut reader, &mut buf, &mut scratch)?;
        peak = peak.max(READ_CHUNK + buf.capacity());
        if n == 0 {
            source_done = true;
        }
    };

    // Phase 2: eligibility probe on the header prefix alone. A parse
    // error here is NOT terminal: progressive and non-interleaved
    // streams walk every scan during header parse, which runs past the
    // prefix — exactly the shapes that take the buffered fallback,
    // where the full-stream parse delivers the real verdict.
    let eligible: bool = match entropy_start {
        Some(off) => match Decoder::new(&buf[..off]) {
            Ok(probe) => probe.baseline_stream_eligible(),
            Err(_) => false,
        },
        None => false,
    };

    if !eligible {
        // Fallback: buffer the remainder and take the slice path. This
        // is the documented behaviour for stream shapes whose entropy
        // layout requires the full stream (progressive multi-pass,
        // non-interleaved block rasters, arithmetic conditioning).
        loop {
            // Bounded like every other accumulation path in this module
            // (reviewer P3): `decompress_from_reader` itself has no cap,
            // so this is deliberately stricter than "exactly like" it —
            // a hostile endless source cannot OOM the fallback.
            if buf.len() + READ_CHUNK > MAX_HEADER_BYTES {
                return Err(JpegError::CorruptData(
                    "stream exceeded the 256 MiB buffering cap".to_string(),
                ));
            }
            let n: usize = read_some(&mut reader, &mut buf, &mut scratch)?;
            peak = peak.max(READ_CHUNK + buf.capacity());
            if n == 0 {
                break;
            }
        }
        let image: Image = crate::api::high_level::decompress(&buf)?;
        return Ok((image, peak));
    }
    let entropy_start: usize = entropy_start.expect("eligible implies an SOS offset was found");

    // Phase 3: windowed row decode. Split rather than clone: `buf`
    // keeps its allocation as the header, the entropy tail moves to a
    // new window vector — no moment holds two copies of the header
    // (codex P2 on peak accounting).
    let window_tail: Vec<u8> = buf[entropy_start..].to_vec();
    peak = peak.max(READ_CHUNK + buf.capacity() + window_tail.capacity());
    buf.truncate(entropy_start);
    let mut header: Vec<u8> = buf;
    // The header keeps the capacity of the whole accumulated prefix;
    // release the tail's worth so the retained allocation matches what
    // the metric reports.
    header.shrink_to_fit();
    let mut buf: Vec<u8> = window_tail;
    let decoder: Decoder = Decoder::new(&header)?;
    let mut st = decoder.baseline_stream_begin()?;
    let mut is_final: bool = source_done;

    loop {
        peak = peak.max(READ_CHUNK + header.capacity() + buf.capacity());
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
        if buf.len() + READ_CHUNK + MIN_RETRY_REFILL > MAX_WINDOW_BYTES {
            return Err(JpegError::CorruptData(format!(
                "entropy window exceeded {} bytes without an MCU row committing",
                MAX_WINDOW_BYTES
            )));
        }
        // Batch refills only while the source keeps delivering full
        // reads: a short read means the source has nothing more ready
        // (drip feed or idle socket), and looping for more would either
        // waste retries or block on a stream that already delivered the
        // image — retry the row with what arrived instead.
        let mut refilled: usize = 0;
        loop {
            let n: usize = read_some(&mut reader, &mut buf, &mut scratch)?;
            peak = peak.max(READ_CHUNK + header.capacity() + buf.capacity());
            if n == 0 {
                is_final = true;
                break;
            }
            refilled += n;
            if refilled >= MIN_RETRY_REFILL || n < READ_CHUNK {
                break;
            }
        }
    }

    decoder.set_prefilled_baseline_planes(st.into_planes());
    let image: Image = decoder.decode_image()?;
    Ok((image, peak))
}
