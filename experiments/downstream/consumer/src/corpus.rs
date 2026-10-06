//! Benchmark corpus: deterministic synthetic images plus two upstream test
//! images.
//!
//! Synthetic content avoids any licensing question and makes every input
//! reproducible from this source file alone. It is generated with integer
//! arithmetic only — no `sin`/`exp` — so the pixels are bit-identical on every
//! platform and libm, and it is *encoded with the published baseline*
//! (`libjpeg-turbo-rs 0.8.0`), never the candidate: if the candidate encoded
//! the corpus, every encoder change would move the decode inputs and two
//! reports could no longer be compared case by case. The SHA-256 of every
//! JPEG lands in the report, so a changed input is visible.

use ljt_baseline::{PixelFormat, Subsampling};

/// Fixed seed: the corpus is part of the benchmark definition.
const CORPUS_SEED: u32 = 0x640_2026;

/// JPEG quality of the synthetic decode inputs: a typical phone camera setting.
pub const CORPUS_QUALITY: u8 = 90;

/// Pixel layout every backend is asked to produce for a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputLayout {
    Rgb,
    Gray,
}

impl OutputLayout {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            OutputLayout::Rgb => 3,
            OutputLayout::Gray => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            OutputLayout::Rgb => "RGB8",
            OutputLayout::Gray => "L8",
        }
    }
}

/// Where a corpus file came from — recorded verbatim in the report.
#[derive(Debug, Clone)]
pub struct CorpusFile {
    pub id: String,
    pub jpeg: Vec<u8>,
    pub origin: String,
    pub licence: String,
}

/// Raw pixels kept for encode cases (the encoders' input and the PSNR
/// reference).
pub struct SourcePixels {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

fn hash32(x: u32, y: u32, seed: u32) -> u32 {
    let mut h: u32 = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ seed;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    h
}

/// Bilinearly interpolated lattice noise in 0..=255 with `cell`-pixel cells:
/// large cells give soft blotches (sky, skin), small cells give foliage-like
/// detail.
fn value_noise(x: u32, y: u32, cell: u32, seed: u32) -> u32 {
    let cell_x: u32 = x / cell;
    let cell_y: u32 = y / cell;
    let frac_x: u32 = x % cell;
    let frac_y: u32 = y % cell;
    let corner = |dx: u32, dy: u32| -> u32 { hash32(cell_x + dx, cell_y + dy, seed) & 0xFF };
    let top: u32 = corner(0, 0) * (cell - frac_x) + corner(1, 0) * frac_x;
    let bottom: u32 = corner(0, 1) * (cell - frac_x) + corner(1, 1) * frac_x;
    (top * (cell - frac_y) + bottom * frac_y) / (cell * cell)
}

/// Triangle wave in 0..=255 with the given period — a periodic texture
/// (fabric, brickwork) that stresses the high-frequency AC coefficients.
fn triangle(t: u32, period: u32) -> u32 {
    let phase: u32 = t % period;
    let half: u32 = period / 2;
    let rising: u32 = if phase < half { phase } else { period - phase };
    (rising * 255 / half.max(1)).min(255)
}

/// One RGB pixel of the "photo": a lit gradient, two noise octaves, a
/// patterned region, a few hard-edged objects and sensor grain.
fn synthetic_pixel(x: u32, y: u32, width: u32, height: u32) -> [u8; 3] {
    // Lighting: brighter towards the top-left, like an outdoor shot.
    let light: u32 = 255 - (x * 90 / width + y * 120 / height);
    let mut channels: [u32; 3] = [0; 3];
    for (channel, value) in channels.iter_mut().enumerate() {
        let seed: u32 = CORPUS_SEED.wrapping_add(channel as u32 * 0x1000);
        let broad: u32 = value_noise(x, y, 96, seed);
        let medium: u32 = value_noise(x, y, 17, seed ^ 0xA5A5);
        let fine: u32 = value_noise(x, y, 4, seed ^ 0x5A5A);
        *value = (light * 3 + broad * 3 + medium * 2 + fine) / 9;
    }
    // A patterned band across the lower third (diagonal weave).
    if y > height * 2 / 3 && x < width * 3 / 4 {
        let weave: u32 = triangle(x + y, 23).min(triangle(x + height - y, 31));
        for value in channels.iter_mut() {
            *value = (*value + weave) / 2;
        }
    }
    // Hard-edged objects: six rectangles at seeded positions.
    for object in 0..6u32 {
        let h: u32 = hash32(object, 7, CORPUS_SEED);
        let left: u32 = (h & 0xFF) * width / 256;
        let top: u32 = ((h >> 8) & 0xFF) * height / 256;
        let object_width: u32 = width / 10 + ((h >> 16) & 0x3F) * width / 512;
        let object_height: u32 = height / 10 + ((h >> 22) & 0x3F) * height / 512;
        if x >= left && x < left + object_width && y >= top && y < top + object_height {
            let tint: [u32; 3] = [
                hash32(object, 1, CORPUS_SEED) & 0xFF,
                hash32(object, 2, CORPUS_SEED) & 0xFF,
                hash32(object, 3, CORPUS_SEED) & 0xFF,
            ];
            for (value, tint_value) in channels.iter_mut().zip(tint) {
                *value = (*value + tint_value * 3) / 4;
            }
        }
    }
    // Sensor grain, independent per channel: +-8.
    let mut pixel: [u8; 3] = [0; 3];
    for (channel, out) in pixel.iter_mut().enumerate() {
        let grain: i32 = (hash32(x, y, CORPUS_SEED ^ (channel as u32 + 11)) & 0x0F) as i32 - 8;
        *out = (channels[channel] as i32 + grain).clamp(0, 255) as u8;
    }
    pixel
}

pub fn synthetic_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            pixels.extend_from_slice(&synthetic_pixel(x, y, width as u32, height as u32));
        }
    }
    pixels
}

/// BT.601 luma of the same scene, integer arithmetic.
pub fn synthetic_gray(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let [r, g, b] = synthetic_pixel(x, y, width as u32, height as u32);
            let luma: u32 = (299 * r as u32 + 587 * g as u32 + 114 * b as u32 + 500) / 1000;
            pixels.push(luma as u8);
        }
    }
    pixels
}

/// Minimal little-endian TIFF block (what follows `Exif\0\0` in APP1) with one
/// IFD0 entry: Orientation (0x0112) = `orientation`. 6 means "rotate 90° CW to
/// display", the usual portrait phone shot.
pub fn exif_orientation_tiff(orientation: u16) -> Vec<u8> {
    let mut tiff: Vec<u8> = Vec::with_capacity(26);
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
    tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
    tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    tiff.extend_from_slice(&1u32.to_le_bytes()); // count
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0]); // value padding to 4 bytes
    tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    tiff
}

fn synthetic_origin(description: &str) -> String {
    format!(
        "synthetic ({description}), generated by experiments/downstream/consumer/src/corpus.rs \
         seed {CORPUS_SEED:#x}, encoded by the published baseline libjpeg-turbo-rs 0.8.0"
    )
}

const SYNTHETIC_LICENCE: &str = "generated by this harness; no third-party content";

pub fn encode_baseline_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    progressive: bool,
    exif: Option<&[u8]>,
) -> Vec<u8> {
    let result = match (progressive, exif) {
        (true, None) => ljt_baseline::compress_progressive(
            rgb,
            width,
            height,
            PixelFormat::Rgb,
            CORPUS_QUALITY,
            Subsampling::S420,
        ),
        (false, None) => ljt_baseline::compress(
            rgb,
            width,
            height,
            PixelFormat::Rgb,
            CORPUS_QUALITY,
            Subsampling::S420,
        ),
        (false, Some(tiff)) => ljt_baseline::compress_with_metadata(
            rgb,
            width,
            height,
            PixelFormat::Rgb,
            CORPUS_QUALITY,
            Subsampling::S420,
            None,
            Some(tiff),
        ),
        (true, Some(_)) => panic!("the corpus has no progressive EXIF case"),
    };
    result.unwrap_or_else(|error| panic!("baseline encode of the corpus failed: {error}"))
}

pub fn synthetic_rgb_jpeg(
    id: &str,
    width: usize,
    height: usize,
    progressive: bool,
    exif_orientation: Option<u16>,
) -> CorpusFile {
    let rgb: Vec<u8> = synthetic_rgb(width, height);
    let tiff: Option<Vec<u8>> = exif_orientation.map(exif_orientation_tiff);
    let jpeg: Vec<u8> = encode_baseline_rgb(&rgb, width, height, progressive, tiff.as_deref());
    let mode: &str = if progressive {
        "progressive"
    } else {
        "baseline"
    };
    let exif_note: String = exif_orientation
        .map(|value| format!(", EXIF orientation {value}"))
        .unwrap_or_default();
    CorpusFile {
        id: id.to_string(),
        jpeg,
        origin: synthetic_origin(&format!(
            "{width}x{height} RGB, q{CORPUS_QUALITY} 4:2:0 {mode}{exif_note}"
        )),
        licence: SYNTHETIC_LICENCE.to_string(),
    }
}

pub fn synthetic_gray_jpeg(id: &str, width: usize, height: usize) -> CorpusFile {
    let gray: Vec<u8> = synthetic_gray(width, height);
    let jpeg: Vec<u8> = ljt_baseline::compress(
        &gray,
        width,
        height,
        PixelFormat::Grayscale,
        CORPUS_QUALITY,
        Subsampling::S444,
    )
    .unwrap_or_else(|error| panic!("baseline grayscale encode of the corpus failed: {error}"));
    CorpusFile {
        id: id.to_string(),
        jpeg,
        origin: synthetic_origin(&format!(
            "{width}x{height} grayscale, q{CORPUS_QUALITY} baseline"
        )),
        licence: SYNTHETIC_LICENCE.to_string(),
    }
}

/// An upstream libjpeg-turbo test image from the candidate checkout's
/// `references/libjpeg-turbo` submodule. A missing file is a hard error: the
/// report would otherwise silently describe a smaller corpus.
pub fn upstream_test_image(id: &str, repo_root: &std::path::Path, file_name: &str) -> CorpusFile {
    let path: std::path::PathBuf = repo_root
        .join("references/libjpeg-turbo/testimages")
        .join(file_name);
    let jpeg: Vec<u8> = std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "{} is required (initialise the references/libjpeg-turbo submodule): {error}",
            path.display()
        )
    });
    CorpusFile {
        id: id.to_string(),
        jpeg,
        origin: format!(
            "references/libjpeg-turbo/testimages/{file_name} (upstream libjpeg-turbo test image)"
        ),
        licence: "IJG License (references/libjpeg-turbo/testimages/LICENSE.txt, README.ijg)"
            .to_string(),
    }
}
