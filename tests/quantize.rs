use libjpeg_turbo_rs::quantize::{dequantize, quantize, DitherMode, QuantizeOptions};

/// Helper: compute mean squared error between two RGB buffers.
fn mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum();
    sum / a.len() as f64
}

/// Helper: generate a horizontal RGB gradient (left = black, right = white).
fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = (x * 255 / (width - 1).max(1)) as u8;
            pixels.push(val);
            pixels.push(val);
            pixels.push(val);
        }
    }
    pixels
}

#[test]
fn uniform_color_image_quantizes_to_one_entry() {
    let width = 8;
    let height = 8;
    // Solid red image
    let pixels: Vec<u8> = vec![255, 0, 0].repeat(width * height);

    let options = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert_eq!(result.width, width);
    assert_eq!(result.height, height);
    assert_eq!(result.indices.len(), width * height);
    // A uniform image should produce exactly 1 palette entry
    assert_eq!(result.palette.len(), 1);
    assert_eq!(result.palette[0], [255, 0, 0]);
    // All indices should point to the same entry
    assert!(result.indices.iter().all(|&i| i == 0));
}

#[test]
fn gradient_palette_size_matches_requested() {
    let width = 256;
    let height = 4;
    let pixels = make_gradient(width, height);

    let options = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert!(result.palette.len() <= 16);
    // A grayscale gradient should use close to 16 colors
    assert!(
        result.palette.len() >= 8,
        "palette too small: {}",
        result.palette.len()
    );
}

#[test]
fn dither_modes_produce_different_outputs() {
    let width = 64;
    let height = 64;
    let pixels = make_gradient(width, height);

    let opts_none = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let opts_ordered = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::Ordered,
        two_pass: true,
        colormap: None,
    };
    let opts_fs = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::FloydSteinberg,
        two_pass: true,
        colormap: None,
    };

    let result_none = quantize(&pixels, width, height, &opts_none).unwrap();
    let result_ordered = quantize(&pixels, width, height, &opts_ordered).unwrap();
    let result_fs = quantize(&pixels, width, height, &opts_fs).unwrap();

    // The palettes may be the same, but the index patterns must differ
    assert_ne!(
        result_none.indices, result_ordered.indices,
        "None and Ordered should differ"
    );
    assert_ne!(
        result_none.indices, result_fs.indices,
        "None and FS should differ"
    );
    assert_ne!(
        result_ordered.indices, result_fs.indices,
        "Ordered and FS should differ"
    );
}

#[test]
fn dequantize_roundtrip_preserves_palette_colors() {
    let width = 4;
    let height = 4;
    // 4 colors: red, green, blue, white
    let mut pixels = Vec::new();
    let colors = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
    for row in 0..height {
        for col in 0..width {
            let c = colors[(row * width + col) % 4];
            pixels.extend_from_slice(&c);
        }
    }

    let options = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let quantized = quantize(&pixels, width, height, &options).unwrap();
    let restored = dequantize(&quantized);

    // With 256 colors and only 4 unique, roundtrip should be perfect
    assert_eq!(pixels, restored);
}

#[test]
fn external_colormap_is_used() {
    let width = 4;
    let height = 4;
    // Pixels are all (128, 128, 128)
    let pixels: Vec<u8> = vec![128, 128, 128].repeat(width * height);

    let colormap = vec![[0, 0, 0], [128, 128, 128], [255, 255, 255]];
    let options = QuantizeOptions {
        num_colors: 3,
        dither_mode: DitherMode::None,
        two_pass: false,
        colormap: Some(colormap.clone()),
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    // Should use the provided colormap exactly
    assert_eq!(result.palette, colormap);
    // All pixels should map to index 1 (128,128,128)
    assert!(result.indices.iter().all(|&i| i == 1));
}

#[test]
fn floyd_steinberg_distributes_error_across_gradient() {
    // Floyd-Steinberg error diffusion should create smoother transitions
    // by distributing quantization error to neighboring pixels.
    // On a gradient with few palette colors, FS produces more varied index
    // patterns (fewer long runs of the same index) than no dithering.
    let width = 128;
    let height = 1;
    let pixels = make_gradient(width, height);

    // Use a fixed 4-color grayscale palette for deterministic comparison
    let palette = vec![[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]];

    let opts_none = QuantizeOptions {
        num_colors: 4,
        dither_mode: DitherMode::None,
        two_pass: false,
        colormap: Some(palette.clone()),
    };
    let opts_fs = QuantizeOptions {
        num_colors: 4,
        dither_mode: DitherMode::FloydSteinberg,
        two_pass: false,
        colormap: Some(palette),
    };

    let result_none = quantize(&pixels, width, height, &opts_none).unwrap();
    let result_fs = quantize(&pixels, width, height, &opts_fs).unwrap();

    // Count index transitions (how often the palette index changes between adjacent pixels).
    // FS dithering should produce more transitions than nearest-neighbor.
    let transitions_none = result_none
        .indices
        .windows(2)
        .filter(|w| w[0] != w[1])
        .count();
    let transitions_fs = result_fs
        .indices
        .windows(2)
        .filter(|w| w[0] != w[1])
        .count();

    assert!(
        transitions_fs > transitions_none,
        "FS should produce more index transitions ({transitions_fs}) than None ({transitions_none})"
    );

    // FS should produce different index patterns than None
    assert_ne!(result_none.indices, result_fs.indices);
}

#[test]
fn two_pass_vs_one_pass_quality_difference() {
    let width = 64;
    let height = 64;
    // Create a colorful image with various hues
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 4) as u8;
            let g = (y * 4) as u8;
            let b = ((x + y) * 2) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let opts_two_pass = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let opts_one_pass = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::None,
        two_pass: false,
        colormap: None,
    };

    let result_two = quantize(&pixels, width, height, &opts_two_pass).unwrap();
    let result_one = quantize(&pixels, width, height, &opts_one_pass).unwrap();

    let restored_two = dequantize(&result_two);
    let restored_one = dequantize(&result_one);

    let mse_two = mse(&pixels, &restored_two);
    let mse_one = mse(&pixels, &restored_one);

    // Two-pass (median cut) should produce better quality than one-pass (uniform)
    assert!(
        mse_two < mse_one,
        "two-pass MSE ({mse_two:.2}) should be less than one-pass MSE ({mse_one:.2})"
    );
}

#[test]
fn num_colors_one() {
    let width = 8;
    let height = 8;
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 32) as u8);
            pixels.push((y * 32) as u8);
            pixels.push(128);
        }
    }

    let options = QuantizeOptions {
        num_colors: 1,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert_eq!(result.palette.len(), 1);
    assert!(result.indices.iter().all(|&i| i == 0));
}

#[test]
fn num_colors_256() {
    let width = 32;
    let height = 32;
    // Generate image with more than 256 unique colors
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push(((x + y) * 4) as u8);
        }
    }

    let options = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert!(result.palette.len() <= 256);
    assert!(result.palette.len() > 1);
}

#[test]
fn grayscale_quantization() {
    let width = 64;
    let height = 1;
    // 64 shades of gray as RGB
    let mut pixels = Vec::new();
    for x in 0..width {
        let val = (x * 4) as u8;
        pixels.push(val);
        pixels.push(val);
        pixels.push(val);
    }

    let options = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert!(result.palette.len() <= 8);
    // Each palette entry should be a gray (R == G == B)
    for color in &result.palette {
        assert_eq!(
            color[0], color[1],
            "palette entry should be gray: {:?}",
            color
        );
        assert_eq!(
            color[1], color[2],
            "palette entry should be gray: {:?}",
            color
        );
    }
}

#[test]
fn invalid_pixel_buffer_size_returns_error() {
    let width = 4;
    let height = 4;
    // Buffer too short (need 4*4*3 = 48 bytes, give 10)
    let pixels = vec![0u8; 10];

    let options = QuantizeOptions::default();
    let result = quantize(&pixels, width, height, &options);
    assert!(result.is_err());
}

#[test]
fn num_colors_zero_returns_error() {
    let pixels = vec![128u8; 3 * 4 * 4];
    let options = QuantizeOptions {
        num_colors: 0,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let result = quantize(&pixels, 4, 4, &options);
    assert!(result.is_err());
}

#[test]
fn num_colors_exceeds_256_returns_error() {
    let pixels = vec![128u8; 3 * 4 * 4];
    let options = QuantizeOptions {
        num_colors: 257,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let result = quantize(&pixels, 4, 4, &options);
    assert!(result.is_err());
}

#[test]
fn ordered_dither_produces_spatial_pattern() {
    let width = 16;
    let height = 16;
    // Uniform mid-gray: quantizing to 2 colors with ordered dither should produce a pattern.
    // Use an external colormap so the palette has exactly 2 entries (black and white).
    let pixels: Vec<u8> = vec![128, 128, 128].repeat(width * height);

    let options = QuantizeOptions {
        num_colors: 2,
        dither_mode: DitherMode::Ordered,
        two_pass: false,
        colormap: Some(vec![[0, 0, 0], [255, 255, 255]]),
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    // With ordered dither on a mid-tone between black and white, we should see a mix
    let count_0 = result.indices.iter().filter(|&&i| i == 0).count();
    let count_1 = result.indices.iter().filter(|&&i| i == 1).count();
    assert!(
        count_0 > 0 && count_1 > 0,
        "ordered dither should use both palette entries (0={count_0}, 1={count_1})"
    );
}

#[test]
fn quantized_image_dimensions_match() {
    let width = 13;
    let height = 7;
    let pixels: Vec<u8> = vec![100, 150, 200].repeat(width * height);

    let options = QuantizeOptions::default();
    let result = quantize(&pixels, width, height, &options).unwrap();

    assert_eq!(result.width, width);
    assert_eq!(result.height, height);
    assert_eq!(result.indices.len(), width * height);
}
