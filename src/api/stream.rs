//! Reader/writer convenience wrappers for JPEG compression and
//! decompression.
//!
//! **These are not incremental**: every `*_from_reader` / `*_from_file`
//! function buffers the entire input with `read_to_end` before handing
//! it to the in-memory decoder, so peak memory is the full compressed
//! size plus all decode intermediates (issue #357). They are I/O-trait
//! conveniences, not an equivalent of libjpeg-turbo's suspending
//! `jpeg_source_mgr`. For bounded input memory on interleaved baseline
//! streams use
//! [`decompress_from_reader_incremental`](crate::decompress_from_reader_incremental)
//! (P4-58); the slice-based `decompress()` core stays the default
//! because its bounds-checked slice fetches are a measured throughput
//! win over trait-object reads.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::vec::Vec;
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use std::fs::File;
use std::io::{Read, Write};
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use std::path::Path;

use crate::common::error::Result;
use crate::common::types::{PixelFormat, Subsampling};
use crate::decode::pipeline::Image;

/// Compress pixels and write JPEG output to a writer.
///
/// Delegates to the in-memory `compress()` and writes the result to the
/// provided writer. Equivalent to libjpeg-turbo's `jpeg_stdio_dest()` or
/// a custom `jpeg_destination_mgr`.
pub fn compress_to_writer<W: Write>(
    writer: &mut W,
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<()> {
    let jpeg_data: Vec<u8> = crate::api::high_level::compress(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
    )?;
    writer.write_all(&jpeg_data)?;
    Ok(())
}

/// Read JPEG data from a reader and decompress.
///
/// **Buffers the entire stream first**: reads all bytes with
/// `read_to_end`, then delegates to the in-memory `decompress()`. Peak
/// memory is the full compressed size plus decode intermediates — this
/// is *not* an incremental source like libjpeg-turbo's suspending
/// `jpeg_source_mgr` (issue #357). For a bounded input window on
/// interleaved baseline streams use
/// [`decompress_from_reader_incremental`](crate::decompress_from_reader_incremental)
/// (P4-58).
pub fn decompress_from_reader<R: Read>(reader: &mut R) -> Result<Image> {
    let mut buffer: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buffer)?;
    crate::api::high_level::decompress(&buffer)
}

/// Compress pixels and write JPEG to a file path.
///
/// Opens the file for writing, compresses the pixel data, and writes the
/// JPEG output. Equivalent to using libjpeg-turbo's `jpeg_stdio_dest()`
/// with `fopen()`.
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
pub fn compress_to_file<P: AsRef<Path>>(
    path: P,
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<()> {
    let mut file: File = File::create(path)?;
    compress_to_writer(
        &mut file,
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
    )
}

/// Read and decompress JPEG from a file path.
///
/// **Buffers the whole file into memory first** (see
/// [`decompress_from_reader`]) — not an incremental read (issue #357).
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
pub fn decompress_from_file<P: AsRef<Path>>(path: P) -> Result<Image> {
    let mut file: File = File::open(path)?;
    decompress_from_reader(&mut file)
}
