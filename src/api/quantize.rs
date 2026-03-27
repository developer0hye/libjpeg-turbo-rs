//! Color quantization for 8-bit indexed/palette output.
//!
//! Provides median-cut palette generation (two-pass) or uniform palette (one-pass),
//! with optional dithering (none, ordered Bayer, Floyd-Steinberg error diffusion).
//! Compatible with libjpeg-turbo's `quantize_colors`, `dither_mode`, `two_pass_quantize`,
//! and `colormap` features.

use crate::common::error::{JpegError, Result};

/// Dithering mode for color quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherMode {
    /// No dithering -- nearest color in palette.
    None,
    /// Ordered dithering (4x4 Bayer matrix).
    Ordered,
    /// Floyd-Steinberg error diffusion dithering.
    FloydSteinberg,
}

/// Options controlling color quantization behavior.
pub struct QuantizeOptions {
    /// Target number of colors (1..=256, default 256).
    pub num_colors: usize,
    /// Dithering mode.
    pub dither_mode: DitherMode,
    /// Use two-pass quantization (median-cut, better quality, slower). Default: true.
    pub two_pass: bool,
    /// External colormap to use instead of generating one.
    /// When provided, the palette is used as-is and no palette generation occurs.
    pub colormap: Option<Vec<[u8; 3]>>,
}

impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            num_colors: 256,
            dither_mode: DitherMode::None,
            two_pass: true,
            colormap: None,
        }
    }
}

/// An image quantized to a color palette.
pub struct QuantizedImage {
    /// Palette indices (one per pixel, row-major).
    pub indices: Vec<u8>,
    /// Color palette (up to 256 RGB entries).
    pub palette: Vec<[u8; 3]>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

/// Quantize RGB pixel data to an indexed palette image.
///
/// `pixels` must be packed RGB (3 bytes per pixel), length = `width * height * 3`.
pub fn quantize(
    pixels: &[u8],
    width: usize,
    height: usize,
    options: &QuantizeOptions,
) -> Result<QuantizedImage> {
    let expected_len: usize = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| JpegError::CorruptData("image dimensions overflow".into()))?;

    if pixels.len() != expected_len {
        return Err(JpegError::BufferTooSmall {
            need: expected_len,
            got: pixels.len(),
        });
    }

    if options.num_colors == 0 || options.num_colors > 256 {
        return Err(JpegError::CorruptData(format!(
            "num_colors must be 1..=256, got {}",
            options.num_colors
        )));
    }

    let num_pixels: usize = width * height;

    // Build or use provided palette
    let palette: Vec<[u8; 3]> = if let Some(ref cmap) = options.colormap {
        cmap.clone()
    } else if options.two_pass {
        build_palette_median_cut(pixels, options.num_colors)
    } else {
        build_palette_uniform(options.num_colors)
    };

    // Map pixels to palette indices with optional dithering
    let indices: Vec<u8> = match options.dither_mode {
        DitherMode::None => map_nearest(pixels, &palette, num_pixels),
        DitherMode::Ordered => map_ordered_dither(pixels, &palette, width, height),
        DitherMode::FloydSteinberg => map_floyd_steinberg(pixels, &palette, width, height),
    };

    Ok(QuantizedImage {
        indices,
        palette,
        width,
        height,
    })
}

/// Re-quantize an already-quantized image with a new colormap/palette.
///
/// Dequantizes the image (palette lookup) then re-quantizes with the new palette.
/// This implements `jpeg_new_colormap()` functionality.
pub fn requantize(
    image: &QuantizedImage,
    new_palette: &[[u8; 3]],
    dither: DitherMode,
) -> QuantizedImage {
    // Dequantize to RGB pixels
    let pixels: Vec<u8> = dequantize(image);

    // Re-map pixels to new palette
    let indices: Vec<u8> = match dither {
        DitherMode::None => map_nearest(&pixels, new_palette, image.width * image.height),
        DitherMode::Ordered => map_ordered_dither(&pixels, new_palette, image.width, image.height),
        DitherMode::FloydSteinberg => {
            map_floyd_steinberg(&pixels, new_palette, image.width, image.height)
        }
    };

    QuantizedImage {
        indices,
        palette: new_palette.to_vec(),
        width: image.width,
        height: image.height,
    }
}

/// Convert a quantized indexed image back to packed RGB pixels.
pub fn dequantize(image: &QuantizedImage) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(image.indices.len() * 3);
    for &idx in &image.indices {
        let color: [u8; 3] = image.palette[idx as usize];
        pixels.extend_from_slice(&color);
    }
    pixels
}

// ---------------------------------------------------------------------------
// Palette generation: median-cut algorithm (two-pass)
// ---------------------------------------------------------------------------

/// A bounding box of colors used by the median-cut algorithm.
struct ColorBox {
    /// Indices into the deduplicated color list.
    colors: Vec<usize>,
}

/// Build an optimal N-color palette from pixel data using median-cut.
fn build_palette_median_cut(pixels: &[u8], num_colors: usize) -> Vec<[u8; 3]> {
    // Collect unique colors with counts
    let mut color_counts: std::collections::HashMap<[u8; 3], u64> =
        std::collections::HashMap::new();
    for chunk in pixels.chunks_exact(3) {
        let color: [u8; 3] = [chunk[0], chunk[1], chunk[2]];
        *color_counts.entry(color).or_insert(0) += 1;
    }

    let unique_colors: Vec<[u8; 3]> = color_counts.keys().copied().collect();
    let counts: Vec<u64> = unique_colors.iter().map(|c| color_counts[c]).collect();

    if unique_colors.len() <= num_colors {
        // Fewer unique colors than requested -- return them all
        return unique_colors;
    }

    // Start with one box containing all colors
    let initial_box = ColorBox {
        colors: (0..unique_colors.len()).collect(),
    };
    let mut boxes: Vec<ColorBox> = vec![initial_box];

    // Split boxes until we have enough
    while boxes.len() < num_colors {
        // Find the box with the largest weighted range to split
        let split_idx: Option<usize> = find_largest_box(&boxes, &unique_colors, &counts);
        let split_idx: usize = match split_idx {
            Some(idx) => idx,
            // No more splittable boxes (all boxes have 1 color)
            Option::None => break,
        };

        let current_box: ColorBox = boxes.remove(split_idx);
        let (box_a, box_b) = split_box(current_box, &unique_colors, &counts);
        boxes.push(box_a);
        boxes.push(box_b);
    }

    // Compute weighted average color for each box
    boxes
        .iter()
        .map(|b| box_average(&b.colors, &unique_colors, &counts))
        .collect()
}

/// Find the box with the largest range (weighted by pixel count) in any channel.
fn find_largest_box(boxes: &[ColorBox], colors: &[[u8; 3]], counts: &[u64]) -> Option<usize> {
    let mut best_idx: Option<usize> = Option::None;
    let mut best_score: u64 = 0;

    for (i, b) in boxes.iter().enumerate() {
        if b.colors.len() < 2 {
            continue;
        }
        let (range, _channel) = box_largest_range(&b.colors, colors);
        // Weight by total pixel count in the box
        let total_count: u64 = b.colors.iter().map(|&ci| counts[ci]).sum();
        let score: u64 = range as u64 * total_count;
        if score > best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }

    best_idx
}

/// Find the channel (0=R, 1=G, 2=B) with the largest range in a box,
/// returning (range, channel).
fn box_largest_range(indices: &[usize], colors: &[[u8; 3]]) -> (u8, usize) {
    let mut min_rgb: [u8; 3] = [255, 255, 255];
    let mut max_rgb: [u8; 3] = [0, 0, 0];

    for &ci in indices {
        let c: [u8; 3] = colors[ci];
        for ch in 0..3 {
            if c[ch] < min_rgb[ch] {
                min_rgb[ch] = c[ch];
            }
            if c[ch] > max_rgb[ch] {
                max_rgb[ch] = c[ch];
            }
        }
    }

    let mut best_ch: usize = 0;
    let mut best_range: u8 = 0;
    for ch in 0..3 {
        let range: u8 = max_rgb[ch] - min_rgb[ch];
        if range > best_range {
            best_range = range;
            best_ch = ch;
        }
    }

    (best_range, best_ch)
}

/// Split a color box at the weighted median along its widest channel.
fn split_box(b: ColorBox, colors: &[[u8; 3]], counts: &[u64]) -> (ColorBox, ColorBox) {
    let (_range, channel) = box_largest_range(&b.colors, colors);

    // Sort by the chosen channel
    let mut sorted: Vec<usize> = b.colors;
    sorted.sort_by_key(|&ci| colors[ci][channel]);

    // Find the weighted median split point
    let total_count: u64 = sorted.iter().map(|&ci| counts[ci]).sum();
    let half: u64 = total_count / 2;
    let mut running: u64 = 0;
    let mut split_pos: usize = 1; // Ensure at least 1 in the first box

    for (i, &ci) in sorted.iter().enumerate() {
        running += counts[ci];
        if running >= half && i > 0 {
            split_pos = i;
            break;
        }
    }

    // Ensure both halves are non-empty
    if split_pos == 0 {
        split_pos = 1;
    }
    if split_pos >= sorted.len() {
        split_pos = sorted.len() - 1;
    }

    let box_a = ColorBox {
        colors: sorted[..split_pos].to_vec(),
    };
    let box_b = ColorBox {
        colors: sorted[split_pos..].to_vec(),
    };

    (box_a, box_b)
}

/// Compute the weighted average color for a box.
fn box_average(indices: &[usize], colors: &[[u8; 3]], counts: &[u64]) -> [u8; 3] {
    let mut sum_r: u64 = 0;
    let mut sum_g: u64 = 0;
    let mut sum_b: u64 = 0;
    let mut total: u64 = 0;

    for &ci in indices {
        let c: [u8; 3] = colors[ci];
        let w: u64 = counts[ci];
        sum_r += c[0] as u64 * w;
        sum_g += c[1] as u64 * w;
        sum_b += c[2] as u64 * w;
        total += w;
    }

    if total == 0 {
        return [0, 0, 0];
    }

    [
        (sum_r / total) as u8,
        (sum_g / total) as u8,
        (sum_b / total) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Palette generation: uniform cube (one-pass)
// ---------------------------------------------------------------------------

/// Build a uniform RGB palette with approximately `num_colors` entries.
/// Uses an NxNxN cube where N = cbrt(num_colors).
fn build_palette_uniform(num_colors: usize) -> Vec<[u8; 3]> {
    let n: usize = (num_colors as f64).cbrt().floor() as usize;
    let n: usize = n.clamp(1, 6); // 6^3 = 216 max

    let mut palette: Vec<[u8; 3]> = Vec::with_capacity(n * n * n);
    for r in 0..n {
        for g in 0..n {
            for b in 0..n {
                let rv: u8 = if n > 1 {
                    (r * 255 / (n - 1)) as u8
                } else {
                    128
                };
                let gv: u8 = if n > 1 {
                    (g * 255 / (n - 1)) as u8
                } else {
                    128
                };
                let bv: u8 = if n > 1 {
                    (b * 255 / (n - 1)) as u8
                } else {
                    128
                };
                palette.push([rv, gv, bv]);
            }
        }
    }

    palette
}

// ---------------------------------------------------------------------------
// Pixel-to-palette mapping
// ---------------------------------------------------------------------------

/// Find the nearest palette entry by squared Euclidean distance.
fn nearest_palette_index(r: u8, g: u8, b: u8, palette: &[[u8; 3]]) -> u8 {
    let mut best_idx: u8 = 0;
    let mut best_dist: u32 = u32::MAX;

    for (i, &color) in palette.iter().enumerate() {
        let dr: i32 = r as i32 - color[0] as i32;
        let dg: i32 = g as i32 - color[1] as i32;
        let db: i32 = b as i32 - color[2] as i32;
        let dist: u32 = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }

    best_idx
}

/// Map each pixel to its nearest palette color (no dithering).
fn map_nearest(pixels: &[u8], palette: &[[u8; 3]], num_pixels: usize) -> Vec<u8> {
    let mut indices: Vec<u8> = Vec::with_capacity(num_pixels);
    for chunk in pixels.chunks_exact(3) {
        indices.push(nearest_palette_index(chunk[0], chunk[1], chunk[2], palette));
    }
    indices
}

// ---------------------------------------------------------------------------
// Ordered (Bayer) dithering
// ---------------------------------------------------------------------------

/// 4x4 Bayer threshold matrix, normalized to [-0.5, +0.5) range.
/// Standard Bayer matrix: (M[row][col] / 16.0 - 0.5).
/// Scaled to an appropriate spread at use site.
const BAYER_4X4: [[f32; 4]; 4] = [
    [-0.5, 0.0, -0.375, 0.125],
    [0.25, -0.25, 0.375, -0.125],
    [-0.3125, 0.1875, -0.4375, 0.0625],
    [0.4375, -0.0625, 0.3125, -0.1875],
];

/// Map pixels with ordered (Bayer) dithering.
///
/// The spread is computed from the palette to scale Bayer thresholds appropriately:
/// larger palette gaps mean larger dither amplitudes.
fn map_ordered_dither(pixels: &[u8], palette: &[[u8; 3]], width: usize, height: usize) -> Vec<u8> {
    // Compute a spread based on average palette step size.
    // For N evenly spaced colors over 0-255, step = 255/(N-1).
    let spread: f32 = if palette.len() > 1 {
        255.0 / (palette.len() as f32 - 1.0)
    } else {
        128.0
    };

    let mut indices: Vec<u8> = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let offset: usize = (y * width + x) * 3;
            let threshold: f32 = BAYER_4X4[y % 4][x % 4] * spread;

            let r: u8 = (pixels[offset] as f32 + threshold)
                .round()
                .clamp(0.0, 255.0) as u8;
            let g: u8 = (pixels[offset + 1] as f32 + threshold)
                .round()
                .clamp(0.0, 255.0) as u8;
            let b: u8 = (pixels[offset + 2] as f32 + threshold)
                .round()
                .clamp(0.0, 255.0) as u8;

            indices.push(nearest_palette_index(r, g, b, palette));
        }
    }

    indices
}

// ---------------------------------------------------------------------------
// Floyd-Steinberg error diffusion dithering
// ---------------------------------------------------------------------------

/// Map pixels with Floyd-Steinberg error diffusion dithering.
///
/// Uses `f32` accumulators to avoid precision loss from integer division
/// in the 7/16, 3/16, 5/16, 1/16 error distribution.
fn map_floyd_steinberg(pixels: &[u8], palette: &[[u8; 3]], width: usize, height: usize) -> Vec<u8> {
    let num_pixels: usize = width * height;
    let mut buffer: Vec<[f32; 3]> = Vec::with_capacity(num_pixels);

    // Initialize with original pixel values as f32
    for chunk in pixels.chunks_exact(3) {
        buffer.push([chunk[0] as f32, chunk[1] as f32, chunk[2] as f32]);
    }

    let mut indices: Vec<u8> = vec![0u8; num_pixels];

    for y in 0..height {
        for x in 0..width {
            let idx: usize = y * width + x;

            // Clamp the error-adjusted pixel
            let r: u8 = buffer[idx][0].round().clamp(0.0, 255.0) as u8;
            let g: u8 = buffer[idx][1].round().clamp(0.0, 255.0) as u8;
            let b: u8 = buffer[idx][2].round().clamp(0.0, 255.0) as u8;

            let palette_idx: u8 = nearest_palette_index(r, g, b, palette);
            indices[idx] = palette_idx;

            let chosen: [u8; 3] = palette[palette_idx as usize];

            // Quantization error (difference between desired and chosen color)
            let err: [f32; 3] = [
                r as f32 - chosen[0] as f32,
                g as f32 - chosen[1] as f32,
                b as f32 - chosen[2] as f32,
            ];

            // Distribute error to neighbors using Floyd-Steinberg coefficients:
            //            *    7/16
            //  3/16   5/16   1/16
            for ch in 0..3 {
                let e: f32 = err[ch];
                if x + 1 < width {
                    buffer[idx + 1][ch] += e * (7.0 / 16.0);
                }
                if y + 1 < height {
                    if x > 0 {
                        buffer[(y + 1) * width + (x - 1)][ch] += e * (3.0 / 16.0);
                    }
                    buffer[(y + 1) * width + x][ch] += e * (5.0 / 16.0);
                    if x + 1 < width {
                        buffer[(y + 1) * width + (x + 1)][ch] += e * (1.0 / 16.0);
                    }
                }
            }
        }
    }

    indices
}
