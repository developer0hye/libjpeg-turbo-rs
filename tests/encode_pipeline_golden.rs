//! Byte-exact characterization harness for the `encode::pipeline::compress_*` family.
//!
//! Purpose: this suite pins the **exact bytes** produced by every public
//! `compress_*` entry point across the axes those entry points differ on
//! (pixel format, subsampling, quality, DCT method, restart interval,
//! smoothing factor, custom quant tables, custom Huffman tables).
//!
//! It exists so that structural refactors of `src/encode/pipeline.rs` — which
//! historically grew as ~10 copy-pasted variants of one algorithm — can be
//! proven output-identical rather than merely "tests still pass". A pure
//! refactor must not move a single byte; if it does, this test fails loudly
//! and names the exact case.
//!
//! The golden table lives in `tests/fixtures/encode_pipeline_golden.txt`, one
//! `case-id<TAB>len<TAB>hash` record per line. Regenerate deliberately with:
//!
//! ```text
//! REGEN_ENCODE_GOLDEN=1 cargo test --test encode_pipeline_golden
//! ```
//!
//! Regenerating is only correct when the encoder's output is *intended* to
//! change; review the diff of the fixture file as carefully as a code diff.

use libjpeg_turbo_rs::encode::pipeline;
use libjpeg_turbo_rs::{DctMethod, HuffmanTableDef, PixelFormat, Subsampling};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// FNV-1a 64-bit. Dependency-free and stable across platforms and Rust
/// versions, which matters because the fixture is committed and compared on
/// every CI target.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Deterministic synthetic image with photo-like local structure: a smooth
/// two-axis gradient (exercises DC prediction and low-frequency coefficients),
/// a hard-edged rectangle (exercises high-frequency AC coefficients and
/// ringing), and reproducible per-pixel noise (defeats degenerate all-zero
/// blocks that would hide entropy-coding differences).
fn synthetic_pixels(width: usize, height: usize, bytes_per_pixel: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * bytes_per_pixel];
    // Linear congruential generator (Numerical Recipes constants) — fixed seed
    // so every run and every platform produces the identical input image.
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;

            let gradient_x: i32 = (x * 255 / width.max(1)) as i32;
            let gradient_y: i32 = (y * 255 / height.max(1)) as i32;
            // Hard-edged block covering the middle ninth of the image.
            let in_rect: bool =
                x * 3 >= width && x * 3 < width * 2 && y * 3 >= height && y * 3 < height * 2;
            let edge: i32 = if in_rect { 200 } else { 0 };

            let red: u8 = (gradient_x + noise + edge).clamp(0, 255) as u8;
            let green: u8 = (gradient_y + noise).clamp(0, 255) as u8;
            let blue: u8 = ((gradient_x + gradient_y) / 2 - noise + edge / 2).clamp(0, 255) as u8;

            let offset: usize = (y * width + x) * bytes_per_pixel;
            for channel in 0..bytes_per_pixel {
                pixels[offset + channel] = match channel {
                    0 => red,
                    1 => green,
                    2 => blue,
                    // 4th channel: alpha / padding / K — a distinct pattern so
                    // formats that must ignore it are distinguishable from
                    // formats that must consume it.
                    _ => red ^ green,
                };
            }
        }
    }
    pixels
}

/// The image geometries under test. Non-MCU-aligned and 1-pixel dimensions are
/// the cases where edge padding, dummy blocks and row-group replication differ
/// between the variants, so they carry most of the regression risk.
const GEOMETRIES: &[(usize, usize)] = &[(1, 1), (7, 5), (16, 16), (33, 17), (64, 48)];

const SUBSAMPLINGS: &[(&str, Subsampling)] = &[
    ("444", Subsampling::S444),
    ("422", Subsampling::S422),
    ("420", Subsampling::S420),
    ("440", Subsampling::S440),
    ("411", Subsampling::S411),
    ("441", Subsampling::S441),
    ("410", Subsampling::S410),
    ("24", Subsampling::S24),
];

const FORMATS: &[(&str, PixelFormat, usize)] = &[
    ("gray", PixelFormat::Grayscale, 1),
    ("rgb", PixelFormat::Rgb, 3),
    ("rgba", PixelFormat::Rgba, 4),
    ("bgr", PixelFormat::Bgr, 3),
    ("bgra", PixelFormat::Bgra, 4),
    ("cmyk", PixelFormat::Cmyk, 4),
];

const DCT_METHODS: &[(&str, DctMethod)] = &[
    ("islow", DctMethod::IsLow),
    ("ifast", DctMethod::IsFast),
    ("float", DctMethod::Float),
];

/// Quality values chosen to straddle the documented divergence points: q<=19
/// forces scaled quant entries past 255 (the 16-bit DQT path), q=100 forces
/// the all-ones table, and q=50 is the unscaled baseline.
const QUALITIES: &[u8] = &[1, 10, 25, 50, 90, 100];

/// Records one case's observable outcome. Errors are recorded too — a variant
/// that rejects an input today must keep rejecting it after the refactor.
fn record(
    results: &mut BTreeMap<String, String>,
    case_id: String,
    outcome: &Result<Vec<u8>, impl std::fmt::Debug>,
) {
    let value: String = match outcome {
        Ok(bytes) => format!("{}\t{:016x}", bytes.len(), fnv1a64(bytes)),
        // Only the error *variant* is pinned, not its message text, so that
        // rewording a diagnostic does not read as an output regression.
        Err(error) => {
            let debug: String = format!("{:?}", error);
            let variant: &str = debug.split(['(', ' ', '{']).next().unwrap_or("Err");
            format!("ERR\t{}", variant)
        }
    };
    assert!(
        results.insert(case_id.clone(), value).is_none(),
        "duplicate case id generated: {case_id}"
    );
}

/// A non-default but valid Huffman table pair, used to prove the
/// custom-Huffman entry point threads its tables through unchanged.
fn custom_huffman_tables() -> ([Option<HuffmanTableDef>; 4], [Option<HuffmanTableDef>; 4]) {
    // A flat 16-symbol DC table: 16 codes of length 4. Valid per JPEG Annex C
    // and deliberately different from the standard Annex K table.
    let mut dc_bits: [u8; 17] = [0; 17];
    dc_bits[4] = 16;
    let dc_def = HuffmanTableDef {
        bits: dc_bits,
        values: (0u8..16).collect(),
    };
    // A 32-symbol AC table: 16 codes of length 5, 16 of length 6.
    let mut ac_bits: [u8; 17] = [0; 17];
    ac_bits[5] = 16;
    ac_bits[6] = 16;
    let ac_def = HuffmanTableDef {
        bits: ac_bits,
        values: (0u8..32).collect(),
    };
    (
        [Some(dc_def.clone()), Some(dc_def), None, None],
        [Some(ac_def.clone()), Some(ac_def), None, None],
    )
}

/// A non-default quantization table set, including entries above 255 so the
/// 16-bit DQT emission path is exercised.
fn custom_quant_tables() -> [Option<[u16; 64]>; 4] {
    let mut luma: [u16; 64] = [0; 64];
    let mut chroma: [u16; 64] = [0; 64];
    for index in 0..64 {
        // Ramp from 1 upward; the tail exceeds 255 to force 16-bit DQT.
        luma[index] = (index as u16 * 9) + 1;
        chroma[index] = (index as u16 * 13) + 3;
    }
    [Some(luma), Some(chroma), None, None]
}

/// Runs the full matrix and returns the case-id -> outcome map.
fn collect_all_cases() -> BTreeMap<String, String> {
    let mut results: BTreeMap<String, String> = BTreeMap::new();

    for &(width, height) in GEOMETRIES {
        for &(format_name, pixel_format, bytes_per_pixel) in FORMATS {
            let pixels: Vec<u8> = synthetic_pixels(width, height, bytes_per_pixel);

            for &(subsampling_name, subsampling) in SUBSAMPLINGS {
                for &quality in QUALITIES {
                    // --- compress: vary DCT method (its distinguishing axis) ---
                    for &(dct_name, dct_method) in DCT_METHODS {
                        let case_id = format!(
                            "compress|{width}x{height}|{format_name}|{subsampling_name}|q{quality}|{dct_name}"
                        );
                        let outcome = pipeline::compress(
                            &pixels,
                            width,
                            height,
                            pixel_format,
                            quality,
                            subsampling,
                            dct_method,
                        );
                        record(&mut results, case_id, &outcome);
                    }

                    // --- compress_with_restart: vary restart interval ---
                    for &restart_interval in &[0u16, 1, 3] {
                        let case_id = format!(
                            "restart|{width}x{height}|{format_name}|{subsampling_name}|q{quality}|ri{restart_interval}"
                        );
                        let outcome = pipeline::compress_with_restart(
                            &pixels,
                            width,
                            height,
                            pixel_format,
                            quality,
                            subsampling,
                            restart_interval,
                            DctMethod::IsLow,
                        );
                        record(&mut results, case_id, &outcome);
                    }

                    // --- compress_optimized: vary smoothing x restart ---
                    for &smoothing_factor in &[0u8, 50, 100] {
                        for &restart_interval in &[0u16, 2] {
                            let case_id = format!(
                                "optimized|{width}x{height}|{format_name}|{subsampling_name}|q{quality}|sm{smoothing_factor}|ri{restart_interval}"
                            );
                            let outcome = pipeline::compress_optimized(
                                &pixels,
                                width,
                                height,
                                pixel_format,
                                quality,
                                subsampling,
                                smoothing_factor,
                                DctMethod::IsLow,
                                restart_interval,
                            );
                            record(&mut results, case_id, &outcome);
                        }
                    }

                    // --- compress_custom_quant ---
                    {
                        let quant = custom_quant_tables();
                        let case_id = format!(
                            "customquant|{width}x{height}|{format_name}|{subsampling_name}|q{quality}"
                        );
                        let outcome = pipeline::compress_custom_quant(
                            &pixels,
                            width,
                            height,
                            pixel_format,
                            quality,
                            subsampling,
                            &quant,
                        );
                        record(&mut results, case_id, &outcome);
                    }

                    // --- compress_custom_huffman ---
                    {
                        let (dc, ac) = custom_huffman_tables();
                        let case_id = format!(
                            "customhuff|{width}x{height}|{format_name}|{subsampling_name}|q{quality}"
                        );
                        let outcome = pipeline::compress_custom_huffman(
                            &pixels,
                            width,
                            height,
                            pixel_format,
                            quality,
                            subsampling,
                            &dc,
                            &ac,
                        );
                        record(&mut results, case_id, &outcome);
                    }
                }
            }
        }
    }

    results
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("encode_pipeline_golden.txt")
}

fn serialize(results: &BTreeMap<String, String>) -> String {
    let mut text = String::new();
    for (case_id, value) in results {
        let _ = writeln!(text, "{case_id}\t{value}");
    }
    text
}

#[test]
fn compress_variants_are_byte_identical_to_golden() {
    let results: BTreeMap<String, String> = collect_all_cases();
    assert!(
        results.len() > 5_000,
        "matrix collapsed to {} cases — the harness is no longer covering the \
         compress_* axes it is supposed to pin",
        results.len()
    );

    let path: PathBuf = fixture_path();

    if std::env::var_os("REGEN_ENCODE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("failed to create fixtures directory");
        std::fs::write(&path, serialize(&results)).expect("failed to write golden fixture");
        eprintln!(
            "REGENERATED {} with {} cases",
            path.display(),
            results.len()
        );
        return;
    }

    let golden_text: String = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "golden fixture {} is missing or unreadable ({error}); generate it with \
             REGEN_ENCODE_GOLDEN=1 cargo test --test encode_pipeline_golden",
            path.display()
        )
    });

    let mut golden: BTreeMap<String, String> = BTreeMap::new();
    for line in golden_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (case_id, value) = line
            .split_once('\t')
            .unwrap_or_else(|| panic!("malformed golden record: {line}"));
        golden.insert(case_id.to_string(), value.to_string());
    }

    // Report every divergence at once — a refactor that shifts output usually
    // shifts a whole family of cases, and the pattern names the root cause far
    // better than the first failure alone.
    let mut mismatches: Vec<String> = Vec::new();
    for (case_id, actual) in &results {
        match golden.get(case_id) {
            Some(expected) if expected == actual => {}
            Some(expected) => mismatches.push(format!(
                "  {case_id}\n    golden: {expected}\n    actual: {actual}"
            )),
            None => mismatches.push(format!(
                "  {case_id}\n    golden: <absent>\n    actual: {actual}"
            )),
        }
    }
    for case_id in golden.keys() {
        if !results.contains_key(case_id) {
            mismatches.push(format!(
                "  {case_id}\n    golden: present\n    actual: <case no longer generated>"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} compress_* cases diverged from the golden fixture:\n{}",
        mismatches.len(),
        results.len(),
        mismatches
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
