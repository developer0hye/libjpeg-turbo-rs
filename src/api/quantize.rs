//! Color quantization for 8-bit indexed/palette output.
//!
//! Provides median-cut palette generation (two-pass) or uniform palette (one-pass),
//! with optional dithering (none, ordered Bayer, Floyd-Steinberg error diffusion).
//! Compatible with libjpeg-turbo's `quantize_colors`, `dither_mode`, `two_pass_quantize`,
//! and `colormap` features.

// The C-compatible algorithm uses index variables to address 3-D histogram arrays.
// Clippy's needless_range_loop lint fires on these but the indices are essential.
#![allow(clippy::needless_range_loop)]

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
    /// Use the C libjpeg-turbo compatible algorithm (jquant2.c).
    ///
    /// When enabled, produces pixel-identical output to C djpeg -quantize.
    /// Uses 5-6-5 histogram, perceptual R=2/G=3/B=1 weights, midpoint box
    /// splitting with population-then-volume phase, histogram-cell-center
    /// color averaging, and C-compatible serpentine Floyd-Steinberg dithering
    /// with integer error accumulation and error limiting.
    pub c_compatible: bool,
}

impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            num_colors: 256,
            dither_mode: DitherMode::None,
            two_pass: true,
            colormap: None,
            c_compatible: false,
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

    // Dispatch to C-compatible algorithm when requested
    if options.c_compatible {
        return quantize_c_compatible(pixels, width, height, options);
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
// C-compatible quantization (jquant2.c algorithm)
// ---------------------------------------------------------------------------
//
// This section implements the exact algorithm from libjpeg-turbo's jquant2.c
// to produce pixel-identical output to C djpeg -quantize.
//
// Key parameters from jquant2.c:
//   HIST_C0_BITS = 5  (R channel, shift = 3)
//   HIST_C1_BITS = 6  (G channel, shift = 2)
//   HIST_C2_BITS = 5  (B channel, shift = 3)
//   Histogram size: 32 x 64 x 32
//   R_SCALE = 2, G_SCALE = 3, B_SCALE = 1 (perceptual weights)

const HIST_C0_BITS: u32 = 5; // R bits
const HIST_C1_BITS: u32 = 6; // G bits
const HIST_C2_BITS: u32 = 5; // B bits

const HIST_C0_ELEMS: usize = 1 << HIST_C0_BITS; // 32
const HIST_C1_ELEMS: usize = 1 << HIST_C1_BITS; // 64
const HIST_C2_ELEMS: usize = 1 << HIST_C2_BITS; // 32

const C0_SHIFT: u32 = 8 - HIST_C0_BITS; // 3
const C1_SHIFT: u32 = 8 - HIST_C1_BITS; // 2
const C2_SHIFT: u32 = 8 - HIST_C2_BITS; // 3

const R_SCALE: i64 = 2;
const G_SCALE: i64 = 3;
const B_SCALE: i64 = 1;

// Sub-box size for inverse colormap: 1/8 of histogram in each direction
const BOX_C0_LOG: u32 = HIST_C0_BITS - 3; // 2
const BOX_C1_LOG: u32 = HIST_C1_BITS - 3; // 3
const BOX_C2_LOG: u32 = HIST_C2_BITS - 3; // 2

const BOX_C0_ELEMS: usize = 1 << BOX_C0_LOG; // 4
const BOX_C1_ELEMS: usize = 1 << BOX_C1_LOG; // 8
const BOX_C2_ELEMS: usize = 1 << BOX_C2_LOG; // 4

const BOX_C0_SHIFT: u32 = C0_SHIFT + BOX_C0_LOG; // 5
const BOX_C1_SHIFT: u32 = C1_SHIFT + BOX_C1_LOG; // 5
const BOX_C2_SHIFT: u32 = C2_SHIFT + BOX_C2_LOG; // 5

const STEP_C0: i64 = (1 << C0_SHIFT) * R_SCALE; // 8 * 2 = 16
const STEP_C1: i64 = (1 << C1_SHIFT) * G_SCALE; // 4 * 3 = 12
const STEP_C2: i64 = (1 << C2_SHIFT) * B_SCALE; // 8 * 1 = 8

/// A histogram box used in median-cut, mirrors C `box` struct.
#[derive(Clone)]
struct CBox {
    c0min: usize,
    c0max: usize,
    c1min: usize,
    c1max: usize,
    c2min: usize,
    c2max: usize,
    /// Scaled 2-norm volume (determines splittability: 0 = not splittable).
    volume: i64,
    /// Number of nonzero histogram cells in this box.
    colorcount: i64,
}

/// Build the 32x64x32 histogram from pixel data.
fn build_histogram(pixels: &[u8]) -> Box<[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS]> {
    // Use heap-allocated 3D array to avoid stack overflow (32*64*32*2 = 131072 bytes)
    let mut histogram = Box::new([[[0u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS]);
    for chunk in pixels.chunks_exact(3) {
        let c0: usize = (chunk[0] >> C0_SHIFT) as usize;
        let c1: usize = (chunk[1] >> C1_SHIFT) as usize;
        let c2: usize = (chunk[2] >> C2_SHIFT) as usize;
        let cell: &mut u16 = &mut histogram[c0][c1][c2];
        // Saturate at u16::MAX (C does: if ++(*histp) <= 0 then (*histp)--)
        *cell = cell.saturating_add(1);
    }
    histogram
}

/// Shrink box bounds to enclose only nonzero histogram cells, recompute volume and population.
/// Mirrors C `update_box()`.
fn update_box(histogram: &[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS], boxp: &mut CBox) {
    let (mut c0min, mut c0max) = (boxp.c0min, boxp.c0max);
    let (mut c1min, mut c1max) = (boxp.c1min, boxp.c1max);
    let (mut c2min, mut c2max) = (boxp.c2min, boxp.c2max);

    // Shrink c0min
    'outer_c0min: for c0 in c0min..=c0max {
        for c1 in c1min..=c1max {
            for c2 in c2min..=c2max {
                if histogram[c0][c1][c2] != 0 {
                    c0min = c0;
                    break 'outer_c0min;
                }
            }
        }
    }
    // Shrink c0max
    'outer_c0max: for c0 in (c0min..=c0max).rev() {
        for c1 in c1min..=c1max {
            for c2 in c2min..=c2max {
                if histogram[c0][c1][c2] != 0 {
                    c0max = c0;
                    break 'outer_c0max;
                }
            }
        }
    }
    // Shrink c1min
    'outer_c1min: for c1 in c1min..=c1max {
        for c0 in c0min..=c0max {
            for c2 in c2min..=c2max {
                if histogram[c0][c1][c2] != 0 {
                    c1min = c1;
                    break 'outer_c1min;
                }
            }
        }
    }
    // Shrink c1max
    'outer_c1max: for c1 in (c1min..=c1max).rev() {
        for c0 in c0min..=c0max {
            for c2 in c2min..=c2max {
                if histogram[c0][c1][c2] != 0 {
                    c1max = c1;
                    break 'outer_c1max;
                }
            }
        }
    }
    // Shrink c2min
    'outer_c2min: for c2 in c2min..=c2max {
        for c0 in c0min..=c0max {
            for c1 in c1min..=c1max {
                if histogram[c0][c1][c2] != 0 {
                    c2min = c2;
                    break 'outer_c2min;
                }
            }
        }
    }
    // Shrink c2max
    'outer_c2max: for c2 in (c2min..=c2max).rev() {
        for c0 in c0min..=c0max {
            for c1 in c1min..=c1max {
                if histogram[c0][c1][c2] != 0 {
                    c2max = c2;
                    break 'outer_c2max;
                }
            }
        }
    }

    boxp.c0min = c0min;
    boxp.c0max = c0max;
    boxp.c1min = c1min;
    boxp.c1max = c1max;
    boxp.c2min = c2min;
    boxp.c2max = c2max;

    // Compute volume using scaled 2-norm (matches C exactly)
    let dist0: i64 = ((c0max - c0min) as i64) * (1 << C0_SHIFT) * R_SCALE;
    let dist1: i64 = ((c1max - c1min) as i64) * (1 << C1_SHIFT) * G_SCALE;
    let dist2: i64 = ((c2max - c2min) as i64) * (1 << C2_SHIFT) * B_SCALE;
    boxp.volume = dist0 * dist0 + dist1 * dist1 + dist2 * dist2;

    // Count nonzero cells
    let mut ccount: i64 = 0;
    for c0 in c0min..=c0max {
        for c1 in c1min..=c1max {
            for c2 in c2min..=c2max {
                if histogram[c0][c1][c2] != 0 {
                    ccount += 1;
                }
            }
        }
    }
    boxp.colorcount = ccount;
}

/// Find the splittable box with the largest color population. Mirrors C `find_biggest_color_pop()`.
fn find_biggest_color_pop(boxes: &[CBox]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut maxc: i64 = 0;
    for (i, b) in boxes.iter().enumerate() {
        if b.colorcount > maxc && b.volume > 0 {
            best = Some(i);
            maxc = b.colorcount;
        }
    }
    best
}

/// Find the splittable box with the largest volume. Mirrors C `find_biggest_volume()`.
fn find_biggest_volume(boxes: &[CBox]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut maxv: i64 = 0;
    for (i, b) in boxes.iter().enumerate() {
        if b.volume > maxv {
            best = Some(i);
            maxv = b.volume;
        }
    }
    best
}

/// Perform median cut to produce `desired_colors` boxes.
/// Mirrors C `median_cut()` with its two-phase population/volume strategy.
fn median_cut(
    histogram: &[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS],
    boxes: &mut Vec<CBox>,
    desired_colors: usize,
) {
    while boxes.len() < desired_colors {
        // Phase 1 (first half): split by population; Phase 2 (second half): split by volume
        let b1_idx: Option<usize> = if boxes.len() * 2 <= desired_colors {
            find_biggest_color_pop(boxes)
        } else {
            find_biggest_volume(boxes)
        };

        let b1_idx: usize = match b1_idx {
            Some(i) => i,
            None => break,
        };

        // Clone b1 to create b2 with same bounds
        let mut b2: CBox = boxes[b1_idx].clone();
        let b1: &mut CBox = &mut boxes[b1_idx];

        // Choose axis: longest scaled axis. Tie-breaking: green > red > blue
        // (matches C code for RGB color order where rgb_red=0, rgb_green=1, rgb_blue=2)
        let c0: i64 = ((b1.c0max - b1.c0min) as i64) * (1 << C0_SHIFT) * R_SCALE;
        let c1: i64 = ((b1.c1max - b1.c1min) as i64) * (1 << C1_SHIFT) * G_SCALE;
        let c2: i64 = ((b1.c2max - b1.c2min) as i64) * (1 << C2_SHIFT) * B_SCALE;

        // C code for rgb_red[cinfo->out_color_space] == 0 (standard RGB):
        //   cmax = c1 (G), n = 1
        //   if c0 > cmax: cmax = c0, n = 0
        //   if c2 > cmax: n = 2
        // This gives tie priority: G wins ties with R, B loses ties with G or R
        let mut cmax: i64 = c1;
        let n: usize = if c0 > cmax {
            cmax = c0;
            if c2 > cmax {
                2
            } else {
                0
            }
        } else if c2 > cmax {
            2
        } else {
            1
        };

        // Split at midpoint
        match n {
            0 => {
                let lb: usize = (b1.c0max + b1.c0min) / 2;
                b1.c0max = lb;
                b2.c0min = lb + 1;
            }
            1 => {
                let lb: usize = (b1.c1max + b1.c1min) / 2;
                b1.c1max = lb;
                b2.c1min = lb + 1;
            }
            _ => {
                let lb: usize = (b1.c2max + b1.c2min) / 2;
                b1.c2max = lb;
                b2.c2min = lb + 1;
            }
        }

        // Update both boxes
        update_box(histogram, &mut boxes[b1_idx]);
        update_box(histogram, &mut b2);
        boxes.push(b2);
    }
}

/// Compute the representative color for a box using histogram cell centers.
/// Mirrors C `compute_color()` with half-cell offset for cell center coordinates.
fn compute_color(
    histogram: &[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS],
    boxp: &CBox,
) -> [u8; 3] {
    let mut total: i64 = 0;
    let mut c0total: i64 = 0;
    let mut c1total: i64 = 0;
    let mut c2total: i64 = 0;

    for c0 in boxp.c0min..=boxp.c0max {
        for c1 in boxp.c1min..=boxp.c1max {
            for c2 in boxp.c2min..=boxp.c2max {
                let count: i64 = histogram[c0][c1][c2] as i64;
                if count != 0 {
                    total += count;
                    // Cell center = (index << shift) + (1 << shift) / 2
                    // This matches C: ((c0 << C0_SHIFT) + ((1 << C0_SHIFT) >> 1)) * count
                    c0total += ((c0 as i64) * (1 << C0_SHIFT) + (1 << C0_SHIFT) / 2) * count;
                    c1total += ((c1 as i64) * (1 << C1_SHIFT) + (1 << C1_SHIFT) / 2) * count;
                    c2total += ((c2 as i64) * (1 << C2_SHIFT) + (1 << C2_SHIFT) / 2) * count;
                }
            }
        }
    }

    if total == 0 {
        return [0, 0, 0];
    }

    // Rounding: (sum + total/2) / total -- matches C: (c0total + (total >> 1)) / total
    [
        ((c0total + (total >> 1)) / total) as u8,
        ((c1total + (total >> 1)) / total) as u8,
        ((c2total + (total >> 1)) / total) as u8,
    ]
}

/// Select colors using the C-compatible two-pass median-cut algorithm.
fn select_colors_c(
    histogram: &[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS],
    desired_colors: usize,
) -> Vec<[u8; 3]> {
    // Initialize one box containing the whole color space (histogram bounds)
    let initial = CBox {
        c0min: 0,
        c0max: 255 >> C0_SHIFT, // = 31
        c1min: 0,
        c1max: 255 >> C1_SHIFT, // = 63
        c2min: 0,
        c2max: 255 >> C2_SHIFT, // = 31
        volume: 0,
        colorcount: 0,
    };

    let mut boxes: Vec<CBox> = vec![initial];
    update_box(histogram, &mut boxes[0]);

    median_cut(histogram, &mut boxes, desired_colors);

    boxes.iter().map(|b| compute_color(histogram, b)).collect()
}

/// Build the error-limit table as in C `init_error_limit()`.
/// Maps raw FS errors to limited values to prevent error cascade.
/// Returns a Vec indexed by (error + 255), covering -255..=255.
fn build_error_limit_table() -> Vec<i32> {
    // Table covers -255..=255 (511 entries), indexed as table[err + 255]
    const MAXJSAMPLE: i32 = 255;
    const STEPSIZE: i32 = (MAXJSAMPLE + 1) / 16; // = 16

    let mut table: Vec<i32> = vec![0i32; (MAXJSAMPLE * 2 + 1) as usize];
    // table[i] = limited error for input error (i - 255)

    // Map 1:1 for |err| < STEPSIZE
    let mut out: i32 = 0;
    for inp in 0..STEPSIZE {
        table[(255 + inp) as usize] = out;
        table[(255 - inp) as usize] = -out;
        out += 1;
    }
    // Map 1:2 for STEPSIZE <= |err| < 3*STEPSIZE
    let mut inp: i32 = STEPSIZE;
    while inp < STEPSIZE * 3 {
        table[(255 + inp) as usize] = out;
        table[(255 - inp) as usize] = -out;
        inp += 1;
        if inp & 1 == 0 {
            out += 1;
        }
    }
    // Clamp the rest
    for inp2 in inp..=MAXJSAMPLE {
        table[(255 + inp2) as usize] = out;
        table[(255 - inp2) as usize] = -out;
    }

    table
}

/// Find the nearest palette index using C-compatible perceptual distance
/// (R_SCALE=2, G_SCALE=3, B_SCALE=1) -- used only when cache misses in
/// the no-dither path require a brute-force search.
#[inline(always)]
fn nearest_palette_index_c(r: u8, g: u8, b: u8, palette: &[[u8; 3]]) -> u8 {
    let mut best_idx: u8 = 0;
    let mut best_dist: i64 = i64::MAX;

    for (i, &color) in palette.iter().enumerate() {
        let dr: i64 = (r as i64 - color[0] as i64) * R_SCALE;
        let dg: i64 = (g as i64 - color[1] as i64) * G_SCALE;
        let db: i64 = (b as i64 - color[2] as i64) * B_SCALE;
        let dist: i64 = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }

    best_idx
}

/// Build the inverse colormap cache using the sub-box method from C.
/// The cache (32x64x32 u16) stores palette-index+1, 0 = not yet computed.
/// This mirrors C's `fill_inverse_cmap()`, `find_nearby_colors()`, `find_best_colors()`.
fn build_inverse_colormap(
    palette: &[[u8; 3]],
    cache: &mut [[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS],
) {
    let numcolors: usize = palette.len();

    // Iterate over all sub-boxes (groups of BOX_C0_ELEMS x BOX_C1_ELEMS x BOX_C2_ELEMS cells)
    let num_boxes_c0: usize = HIST_C0_ELEMS / BOX_C0_ELEMS;
    let num_boxes_c1: usize = HIST_C1_ELEMS / BOX_C1_ELEMS;
    let num_boxes_c2: usize = HIST_C2_ELEMS / BOX_C2_ELEMS;

    for bc0 in 0..num_boxes_c0 {
        for bc1 in 0..num_boxes_c1 {
            for bc2 in 0..num_boxes_c2 {
                // Compute the center coordinate of the lower-left histogram cell in this sub-box
                // minc = (box_id << BOX_SHIFT) + (1 << C_SHIFT) / 2
                let minc0: i64 = ((bc0 << BOX_C0_SHIFT) + (1 << C0_SHIFT) / 2) as i64;
                let minc1: i64 = ((bc1 << BOX_C1_SHIFT) + (1 << C1_SHIFT) / 2) as i64;
                let minc2: i64 = ((bc2 << BOX_C2_SHIFT) + (1 << C2_SHIFT) / 2) as i64;

                // Upper bounds of the sub-box volume
                let maxc0: i64 = minc0 + ((1 << BOX_C0_SHIFT) as i64) - ((1 << C0_SHIFT) as i64);
                let maxc1: i64 = minc1 + ((1 << BOX_C1_SHIFT) as i64) - ((1 << C1_SHIFT) as i64);
                let maxc2: i64 = minc2 + ((1 << BOX_C2_SHIFT) as i64) - ((1 << C2_SHIFT) as i64);

                let centerc0: i64 = (minc0 + maxc0) >> 1;
                let centerc1: i64 = (minc1 + maxc1) >> 1;
                let centerc2: i64 = (minc2 + maxc2) >> 1;

                // Step 1: find_nearby_colors -- Heckbert criterion to select candidates
                let mut mindist: Vec<i64> = vec![0i64; numcolors];
                let mut minmaxdist: i64 = i64::MAX;

                for (i, &color) in palette.iter().enumerate() {
                    let x0: i64 = color[0] as i64;
                    let x1: i64 = color[1] as i64;
                    let x2: i64 = color[2] as i64;

                    let (min0, max0) = if x0 < minc0 {
                        let td: i64 = (x0 - minc0) * R_SCALE;
                        let mn: i64 = td * td;
                        let td2: i64 = (x0 - maxc0) * R_SCALE;
                        (mn, td2 * td2)
                    } else if x0 > maxc0 {
                        let td: i64 = (x0 - maxc0) * R_SCALE;
                        let mn: i64 = td * td;
                        let td2: i64 = (x0 - minc0) * R_SCALE;
                        (mn, td2 * td2)
                    } else {
                        let mx: i64 = if x0 <= centerc0 {
                            let td: i64 = (x0 - maxc0) * R_SCALE;
                            td * td
                        } else {
                            let td: i64 = (x0 - minc0) * R_SCALE;
                            td * td
                        };
                        (0, mx)
                    };

                    let (min1, max1) = if x1 < minc1 {
                        let td: i64 = (x1 - minc1) * G_SCALE;
                        let mn: i64 = td * td;
                        let td2: i64 = (x1 - maxc1) * G_SCALE;
                        (mn, td2 * td2)
                    } else if x1 > maxc1 {
                        let td: i64 = (x1 - maxc1) * G_SCALE;
                        let mn: i64 = td * td;
                        let td2: i64 = (x1 - minc1) * G_SCALE;
                        (mn, td2 * td2)
                    } else {
                        let mx: i64 = if x1 <= centerc1 {
                            let td: i64 = (x1 - maxc1) * G_SCALE;
                            td * td
                        } else {
                            let td: i64 = (x1 - minc1) * G_SCALE;
                            td * td
                        };
                        (0, mx)
                    };

                    let (min2, max2) = if x2 < minc2 {
                        let td: i64 = (x2 - minc2) * B_SCALE;
                        let mn: i64 = td * td;
                        let td2: i64 = (x2 - maxc2) * B_SCALE;
                        (mn, td2 * td2)
                    } else if x2 > maxc2 {
                        let td: i64 = (x2 - maxc2) * B_SCALE;
                        let mn: i64 = td * td;
                        let td2: i64 = (x2 - minc2) * B_SCALE;
                        (mn, td2 * td2)
                    } else {
                        let mx: i64 = if x2 <= centerc2 {
                            let td: i64 = (x2 - maxc2) * B_SCALE;
                            td * td
                        } else {
                            let td: i64 = (x2 - minc2) * B_SCALE;
                            td * td
                        };
                        (0, mx)
                    };

                    mindist[i] = min0 + min1 + min2;
                    let max_total: i64 = max0 + max1 + max2;
                    if max_total < minmaxdist {
                        minmaxdist = max_total;
                    }
                }

                // Collect candidates (those within minmaxdist)
                let mut colorlist: Vec<usize> = Vec::with_capacity(numcolors);
                for i in 0..numcolors {
                    if mindist[i] <= minmaxdist {
                        colorlist.push(i);
                    }
                }

                // Step 2: find_best_colors -- Thomas incremental distance method
                let box_cells: usize = BOX_C0_ELEMS * BOX_C1_ELEMS * BOX_C2_ELEMS;
                let mut bestdist: Vec<i64> = vec![i64::MAX; box_cells];
                let mut bestcolor: Vec<u8> = vec![0u8; box_cells];

                for &icolor in &colorlist {
                    let color: &[u8; 3] = &palette[icolor];
                    // Initial distance from minc to this color
                    let inc0: i64 = (minc0 - color[0] as i64) * R_SCALE;
                    let inc1: i64 = (minc1 - color[1] as i64) * G_SCALE;
                    let inc2: i64 = (minc2 - color[2] as i64) * B_SCALE;
                    let dist0_base: i64 = inc0 * inc0 + inc1 * inc1 + inc2 * inc2;

                    // Increments (Thomas method)
                    let mut xinc0: i64 = inc0 * (2 * STEP_C0) + STEP_C0 * STEP_C0;
                    let xinc1_base: i64 = inc1 * (2 * STEP_C1) + STEP_C1 * STEP_C1;
                    let xinc2_base: i64 = inc2 * (2 * STEP_C2) + STEP_C2 * STEP_C2;

                    let mut bptr: usize = 0;
                    let mut dist0: i64 = dist0_base;
                    for _ic0 in 0..BOX_C0_ELEMS {
                        let mut dist1: i64 = dist0;
                        let mut xx1: i64 = xinc1_base;
                        for _ic1 in 0..BOX_C1_ELEMS {
                            let mut dist2: i64 = dist1;
                            let mut xx2: i64 = xinc2_base;
                            for _ic2 in 0..BOX_C2_ELEMS {
                                if dist2 < bestdist[bptr] {
                                    bestdist[bptr] = dist2;
                                    bestcolor[bptr] = icolor as u8;
                                }
                                dist2 += xx2;
                                xx2 += 2 * STEP_C2 * STEP_C2;
                                bptr += 1;
                            }
                            dist1 += xx1;
                            xx1 += 2 * STEP_C1 * STEP_C1;
                        }
                        dist0 += xinc0;
                        xinc0 += 2 * STEP_C0 * STEP_C0;
                    }
                }

                // Store results into cache (value = colorindex + 1)
                let c0_base: usize = bc0 * BOX_C0_ELEMS;
                let c1_base: usize = bc1 * BOX_C1_ELEMS;
                let c2_base: usize = bc2 * BOX_C2_ELEMS;
                let mut bptr: usize = 0;
                for ic0 in 0..BOX_C0_ELEMS {
                    for ic1 in 0..BOX_C1_ELEMS {
                        for ic2 in 0..BOX_C2_ELEMS {
                            cache[c0_base + ic0][c1_base + ic1][c2_base + ic2] =
                                bestcolor[bptr] as u16 + 1;
                            bptr += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Map pixels to palette indices using the inverse colormap cache (no dithering).
fn map_no_dither_c(
    pixels: &[u8],
    palette: &[[u8; 3]],
    cache: &[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS],
    num_pixels: usize,
) -> Vec<u8> {
    let mut indices: Vec<u8> = Vec::with_capacity(num_pixels);
    for chunk in pixels.chunks_exact(3) {
        let c0: usize = (chunk[0] >> C0_SHIFT) as usize;
        let c1: usize = (chunk[1] >> C1_SHIFT) as usize;
        let c2: usize = (chunk[2] >> C2_SHIFT) as usize;
        let cached: u16 = cache[c0][c1][c2];
        let idx: u8 = if cached != 0 {
            (cached - 1) as u8
        } else {
            // Should not happen if cache is fully pre-built, but fallback safely
            nearest_palette_index_c(chunk[0], chunk[1], chunk[2], palette)
        };
        indices.push(idx);
    }
    indices
}

/// Map pixels with C-compatible FS dithering.
///
/// Mirrors C `pass2_fs_dither()` from jquant2.c exactly:
/// - Serpentine scanning (LTR on even rows, RTL on odd rows)
/// - Integer×16 error accumulation with RIGHT_SHIFT rounding
/// - Error limiting via lookup table
/// - Inverse colormap cache for nearest-color lookup
///
/// The fserrors array has (width + 2) * 3 entries (INT16 in C, i32 here for range).
/// The layout mirrors C: errorptr starts BEFORE the first column (at the dummy entry),
/// reads from `errorptr[dir3]` (current column), writes to `errorptr[0]` (previous column),
/// then advances `errorptr += dir3`.
fn map_fs_dither_c(
    pixels: &[u8],
    palette: &[[u8; 3]],
    cache: &[[[u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS],
    error_limit: &[i32],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let num_pixels: usize = width * height;
    let mut indices: Vec<u8> = vec![0u8; num_pixels];

    // fserrors array: (width + 2) * 3 i32 entries, all zeroed.
    // C uses FSERROR (INT16), but we use i32 to avoid overflow during accumulation.
    // Entry layout: flat array, entry i occupies fserrors[i*3..i*3+3].
    // Entries 0..=(width+1); 0 and width+1 are dummy slots at the row ends.
    let fs_len: usize = (width + 2) * 3;
    let mut fserrors: Vec<i32> = vec![0i32; fs_len];

    // on_odd_row: C initializes to FALSE (first row is LTR).
    let mut on_odd_row: bool = false;

    for row in 0..height {
        let row_base: usize = row * width;

        // C chooses direction based on on_odd_row, then flips it.
        // For odd row (RTL): errorptr starts at fserrors + (width+1)*3, dir3 = -3
        // For even row (LTR): errorptr starts at fserrors + 0, dir3 = +3
        //
        // In C: "errorptr points to *previous* column's array entry"
        // Reads: errorptr[dir3 + component] = current column's carried error
        // Writes: errorptr[component] = 3/16 distribution for previous column's next-row
        // Advance: errorptr += dir3 (now points to current column's entry)
        //
        // We model this with errorptr as a signed index into fserrors (divided by 3).
        let (dir, dir3, mut ep, mut inptr, mut outptr): (isize, isize, isize, isize, isize) =
            if on_odd_row {
                // RTL
                let ep_start: isize = (width + 1) as isize;
                let in_start: isize = (row_base + width - 1) as isize * 3;
                let out_start: isize = (row_base + width - 1) as isize;
                (-1, -1, ep_start, in_start, out_start)
            } else {
                // LTR
                let ep_start: isize = 0;
                let in_start: isize = (row_base) as isize * 3;
                let out_start: isize = (row_base) as isize;
                (1, 1, ep_start, in_start, out_start)
            };
        // C: "flip for next time" — done before processing
        on_odd_row = !on_odd_row;

        let mut cur0: i32 = 0;
        let mut cur1: i32 = 0;
        let mut cur2: i32 = 0;
        let mut belowerr0: i32 = 0;
        let mut belowerr1: i32 = 0;
        let mut belowerr2: i32 = 0;
        let mut bpreverr0: i32 = 0;
        let mut bpreverr1: i32 = 0;
        let mut bpreverr2: i32 = 0;

        for _col in 0..width {
            // Read current column's below-row errors.
            // C: cur0 = RIGHT_SHIFT(cur0 + errorptr[dir3 + 0] + 8, 4)
            // errorptr[dir3] is the CURRENT column's error entry (ep + dir3 in entry units).
            let read_ep: usize = (ep + dir3) as usize;
            let rb: usize = read_ep * 3;

            cur0 = (cur0 + fserrors[rb] + 8) >> 4;
            cur1 = (cur1 + fserrors[rb + 1] + 8) >> 4;
            cur2 = (cur2 + fserrors[rb + 2] + 8) >> 4;

            // Error limiting: table indexed by [-255..=255] mapped to [0..=510]
            cur0 = error_limit[(255 + cur0.clamp(-255, 255)) as usize];
            cur1 = error_limit[(255 + cur1.clamp(-255, 255)) as usize];
            cur2 = error_limit[(255 + cur2.clamp(-255, 255)) as usize];

            // Add pixel value and clamp (C: range_limit[cur + pixel])
            cur0 = (cur0 + pixels[inptr as usize] as i32).clamp(0, 255);
            cur1 = (cur1 + pixels[inptr as usize + 1] as i32).clamp(0, 255);
            cur2 = (cur2 + pixels[inptr as usize + 2] as i32).clamp(0, 255);

            // Nearest color via inverse colormap cache
            let c0i: usize = (cur0 as usize) >> C0_SHIFT as usize;
            let c1i: usize = (cur1 as usize) >> C1_SHIFT as usize;
            let c2i: usize = (cur2 as usize) >> C2_SHIFT as usize;
            let cached: u16 = cache[c0i][c1i][c2i];
            let pixcode: usize = if cached != 0 {
                (cached - 1) as usize
            } else {
                nearest_palette_index_c(cur0 as u8, cur1 as u8, cur2 as u8, palette) as usize
            };
            indices[outptr as usize] = pixcode as u8;

            // Representation error
            cur0 -= palette[pixcode][0] as i32;
            cur1 -= palette[pixcode][1] as i32;
            cur2 -= palette[pixcode][2] as i32;

            // Distribute error into the next-row error array.
            // C writes to errorptr[0] (the PREVIOUS column's entry, at ep).
            // After this block, errorptr advances to current column (ep += dir3).
            // The next row's error contributions:
            //   errorptr[0] = bpreverr + cur * 3  → 3/16 going to below-prev
            //   bpreverr    = belowerr + cur * 5  → 5/16 going to below-current
            //   belowerr    = cur                 → 1/16 going to below-next
            //   cur         *= 7                  → 7/16 going to next on same row
            let wb: usize = (ep as usize) * 3; // write to previous entry

            let bnexterr0: i32 = cur0;
            fserrors[wb] = bpreverr0 + cur0 * 3;
            bpreverr0 = belowerr0 + cur0 * 5;
            belowerr0 = bnexterr0;
            cur0 *= 7;

            let bnexterr1: i32 = cur1;
            fserrors[wb + 1] = bpreverr1 + cur1 * 3;
            bpreverr1 = belowerr1 + cur1 * 5;
            belowerr1 = bnexterr1;
            cur1 *= 7;

            let bnexterr2: i32 = cur2;
            fserrors[wb + 2] = bpreverr2 + cur2 * 3;
            bpreverr2 = belowerr2 + cur2 * 5;
            belowerr2 = bnexterr2;
            cur2 *= 7;

            // Advance all pointers
            inptr += dir3 * 3;
            outptr += dir;
            ep += dir3; // errorptr += dir3 in C
        }

        // Post-loop: unload bpreverr into the entry errorptr currently points to.
        // C: errorptr[0] = bpreverr (errorptr now at last processed column's entry).
        let wb: usize = (ep as usize) * 3;
        fserrors[wb] = bpreverr0;
        fserrors[wb + 1] = bpreverr1;
        fserrors[wb + 2] = bpreverr2;
    }

    indices
}

/// Top-level C-compatible quantization dispatcher.
fn quantize_c_compatible(
    pixels: &[u8],
    width: usize,
    height: usize,
    options: &QuantizeOptions,
) -> Result<QuantizedImage> {
    let num_pixels: usize = width * height;

    // Build histogram
    let histogram = build_histogram(pixels);

    // Select palette using C-compatible median-cut
    let palette: Vec<[u8; 3]> = if let Some(ref cmap) = options.colormap {
        cmap.clone()
    } else {
        select_colors_c(&histogram, options.num_colors)
    };

    // Build the full inverse colormap cache upfront
    // (C does lazy per-subbox fill, but pre-building is equivalent and simpler)
    let mut cache = Box::new([[[0u16; HIST_C2_ELEMS]; HIST_C1_ELEMS]; HIST_C0_ELEMS]);
    build_inverse_colormap(&palette, &mut cache);

    let indices: Vec<u8> = match options.dither_mode {
        DitherMode::None | DitherMode::Ordered => {
            // C jquant2.c: ordered dither falls back to FS
            // No-dither uses inverse colormap cache
            map_no_dither_c(pixels, &palette, &cache, num_pixels)
        }
        DitherMode::FloydSteinberg => {
            let error_limit = build_error_limit_table();
            map_fs_dither_c(pixels, &palette, &cache, &error_limit, width, height)
        }
    };

    Ok(QuantizedImage {
        indices,
        palette,
        width,
        height,
    })
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
