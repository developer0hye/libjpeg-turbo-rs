//! Reference hash regression detection.
//!
//! Compares current encoder output hashes against stored known-good values
//! in `tests/reference_hashes.json`. When hashes are `null`, the test records
//! the current value and prints instructions to update the file.
//!
//! To regenerate all hashes after an intentional encoder change:
//!   UPDATE_HASHES=1 cargo test bitstream_regression -- --nocapture
//!
//! Then commit the updated `reference_hashes.json`.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless,
    compress_optimized, compress_progressive, compress_with_metadata, PixelFormat, Subsampling,
};

fn hash_bytes(data: &[u8]) -> String {
    let mut hasher: DefaultHasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn generate_test_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 5 + y * 3) % 256) as u8);
            pixels.push(((x * 3 + y * 7) % 256) as u8);
            pixels.push(((x * 7 + y * 11) % 256) as u8);
        }
    }
    pixels
}

fn generate_grayscale_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8);
        }
    }
    pixels
}

/// Load the reference hashes from the JSON file.
fn load_reference_hashes() -> HashMap<String, Option<String>> {
    let json_str: &str = include_str!("reference_hashes.json");
    // Minimal JSON parsing: extract key-value pairs.
    // The file has a simple flat structure: { "key": "value" | null, ... }
    let mut hashes: HashMap<String, Option<String>> = HashMap::new();
    for line in json_str.lines() {
        let trimmed: &str = line.trim();
        if !trimmed.starts_with('"') || trimmed.starts_with("\"_") {
            continue;
        }
        // Parse: "key": "value" or "key": null
        let parts: Vec<&str> = trimmed.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key: String = parts[0].trim().trim_matches('"').to_string();
        let value_str: &str = parts[1].trim().trim_end_matches(',');
        let value: Option<String> = if value_str == "null" {
            None
        } else {
            Some(value_str.trim_matches('"').to_string())
        };
        hashes.insert(key, value);
    }
    hashes
}

/// Build the map of configuration name -> current hash.
fn compute_current_hashes() -> Vec<(String, String)> {
    let rgb_pixels: Vec<u8> = generate_test_pattern(64, 64);
    let gray_pixels: Vec<u8> = generate_grayscale_pattern(64, 64);
    let fake_icc: Vec<u8> = vec![0x42u8; 200];
    let fake_exif: Vec<u8> = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];

    vec![
        (
            "baseline_64x64_q75_420".to_string(),
            hash_bytes(
                &compress(&rgb_pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
            ),
        ),
        (
            "baseline_64x64_q75_444".to_string(),
            hash_bytes(
                &compress(&rgb_pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap(),
            ),
        ),
        (
            "baseline_64x64_q50_420".to_string(),
            hash_bytes(
                &compress(&rgb_pixels, 64, 64, PixelFormat::Rgb, 50, Subsampling::S420).unwrap(),
            ),
        ),
        (
            "baseline_64x64_q100_420".to_string(),
            hash_bytes(
                &compress(
                    &rgb_pixels,
                    64,
                    64,
                    PixelFormat::Rgb,
                    100,
                    Subsampling::S420,
                )
                .unwrap(),
            ),
        ),
        (
            "progressive_64x64_q75_444".to_string(),
            hash_bytes(
                &compress_progressive(&rgb_pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444)
                    .unwrap(),
            ),
        ),
        (
            "arithmetic_64x64_q75_420".to_string(),
            hash_bytes(
                &compress_arithmetic(&rgb_pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420)
                    .unwrap(),
            ),
        ),
        (
            "arithmetic_progressive_64x64_q75_444".to_string(),
            hash_bytes(
                &compress_arithmetic_progressive(
                    &rgb_pixels,
                    64,
                    64,
                    PixelFormat::Rgb,
                    75,
                    Subsampling::S444,
                )
                .unwrap(),
            ),
        ),
        (
            "lossless_64x64_gray".to_string(),
            hash_bytes(&compress_lossless(&gray_pixels, 64, 64, PixelFormat::Grayscale).unwrap()),
        ),
        (
            "optimized_64x64_q75_420".to_string(),
            hash_bytes(
                &compress_optimized(&rgb_pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420)
                    .unwrap(),
            ),
        ),
        (
            "grayscale_64x64_q75".to_string(),
            hash_bytes(
                &compress(
                    &gray_pixels,
                    64,
                    64,
                    PixelFormat::Grayscale,
                    75,
                    Subsampling::S444,
                )
                .unwrap(),
            ),
        ),
        (
            "metadata_icc_exif_64x64_q75_444".to_string(),
            hash_bytes(
                &compress_with_metadata(
                    &rgb_pixels,
                    64,
                    64,
                    PixelFormat::Rgb,
                    75,
                    Subsampling::S444,
                    Some(&fake_icc),
                    Some(&fake_exif),
                )
                .unwrap(),
            ),
        ),
    ]
}

#[test]
fn regression_check_reference_hashes() {
    let reference: HashMap<String, Option<String>> = load_reference_hashes();
    let current: Vec<(String, String)> = compute_current_hashes();

    let update_mode: bool = std::env::var("UPDATE_HASHES").is_ok();
    let mut needs_update: Vec<(String, String)> = Vec::new();
    let mut mismatches: Vec<String> = Vec::new();

    for (name, current_hash) in &current {
        match reference.get(name) {
            Some(Some(expected_hash)) => {
                if current_hash != expected_hash {
                    mismatches.push(format!(
                        "  {name}: expected={expected_hash}, got={current_hash}"
                    ));
                }
            }
            Some(None) | None => {
                // Hash not yet recorded
                needs_update.push((name.clone(), current_hash.clone()));
            }
        }
    }

    if !needs_update.is_empty() {
        eprintln!();
        eprintln!("=== Reference hashes not yet recorded ===");
        eprintln!("Update tests/reference_hashes.json with these values:");
        eprintln!();
        for (name, hash) in &needs_update {
            eprintln!("    \"{name}\": \"{hash}\",");
        }
        eprintln!();

        if update_mode {
            eprintln!("UPDATE_HASHES=1 is set. Copy the above values into reference_hashes.json.");
        }
    }

    if !mismatches.is_empty() {
        panic!(
            "Reference hash regression detected!\n\
             The following configurations produce different output than recorded:\n\
             {}\n\
             If this is intentional (encoder change), run:\n\
             UPDATE_HASHES=1 cargo test bitstream_regression -- --nocapture\n\
             and update tests/reference_hashes.json.",
            mismatches.join("\n")
        );
    }
}

/// Verify that the hash function itself is stable within a single run.
#[test]
fn hash_function_self_consistency() {
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let h1: String = hash_bytes(&data);
    let h2: String = hash_bytes(&data);
    assert_eq!(h1, h2, "hash function must be deterministic within a run");
}
