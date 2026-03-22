//! Streaming I/O for JPEG compression and decompression.
//!
//! Provides functions to compress/decompress JPEG data using `std::io::Read`
//! and `std::io::Write` traits, plus convenience functions for file paths.
//! This is the Rust equivalent of libjpeg-turbo's `jpeg_source_mgr` /
//! `jpeg_destination_mgr` custom I/O abstraction.

use std::fs::File;
use std::io::{Read, Write};
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
/// Reads all bytes from the reader into memory, then delegates to the
/// in-memory `decompress()`. Equivalent to libjpeg-turbo's `jpeg_stdio_src()`
/// or a custom `jpeg_source_mgr`.
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
/// Opens the file for reading, reads all JPEG data, and decompresses it.
/// Equivalent to using libjpeg-turbo's `jpeg_stdio_src()` with `fopen()`.
pub fn decompress_from_file<P: AsRef<Path>>(path: P) -> Result<Image> {
    let mut file: File = File::open(path)?;
    decompress_from_reader(&mut file)
}
