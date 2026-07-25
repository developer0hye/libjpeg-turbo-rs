//! Issue #314: 4:2:0 encode diverged from `cjpeg` whenever `ceil(width/8)` is odd.
//!
//! The x86_64 AVX2 4:2:0 row fast path (`src/encode/pipeline.rs:373`) FDCTs
//! every block of every MCU unconditionally. It guarded the last partial MCU
//! *row* but not the last partial MCU *column*, so for any width whose luma
//! block count is odd it forward-transformed replicated edge pixels where C
//! emits a dummy block — zeroed, with the previous block's DC copied in and no
//! AC at all (`jccoefct.c:292-312`).
//!
//! That covers roughly half of all widths, including ordinary photo sizes such
//! as 500x375, 1000x750 and 1080x1080. It escaped detection because the fast
//! path is x86_64+AVX2-only (the project was developed on aarch64, which always
//! takes the generic path) and because the byte-exact encoder cross-check used
//! only the MCU-aligned 48x48.
//!
//! Two layers of coverage here:
//!   1. `*_matches_cjpeg` — byte-exact against stock `cjpeg`, skipped with a
//!      reason when C tools are absent.
//!   2. `*_pinned_bytes` — the same expectation frozen as length + hash, so the
//!      regression is caught on machines without the C toolchain. Every pinned
//!      value below was taken from output verified byte-identical to `cjpeg`.

mod helpers;

use libjpeg_turbo_rs::encode::pipeline;
use libjpeg_turbo_rs::{DctMethod, PixelFormat, Subsampling};

/// FNV-1a 64-bit, matching `tests/encode_pipeline_golden.rs`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Must stay identical to the generator in `examples/probe_fused_vs_fullplane.rs`,
/// which produced the PPM files that `cjpeg` encoded to derive the pinned values.
fn synthetic_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let gradient_x: i32 = (x * 255 / width.max(1)) as i32;
            let gradient_y: i32 = (y * 255 / height.max(1)) as i32;
            let in_rect: bool =
                x * 3 >= width && x * 3 < width * 2 && y * 3 >= height && y * 3 < height * 2;
            let edge: i32 = if in_rect { 200 } else { 0 };
            let red: u8 = (gradient_x + noise + edge).clamp(0, 255) as u8;
            let green: u8 = (gradient_y + noise).clamp(0, 255) as u8;
            let blue: u8 = ((gradient_x + gradient_y) / 2 - noise + edge / 2).clamp(0, 255) as u8;
            let offset: usize = (y * width + x) * 3;
            pixels[offset] = red;
            pixels[offset + 1] = green;
            pixels[offset + 2] = blue;
        }
    }
    pixels
}

fn encode_420(width: usize, height: usize) -> Vec<u8> {
    let pixels: Vec<u8> = synthetic_pixels(width, height);
    pipeline::compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        50,
        Subsampling::S420,
        DctMethod::IsLow,
    )
    .unwrap_or_else(|error| panic!("{width}x{height} 4:2:0 encode failed: {error:?}"))
}

/// `(width, height, expected_len, expected_fnv1a64)`.
///
/// The first four have `ceil(width/8)` odd — these are the regressions. The last
/// two have it even and were always correct; they are controls proving the fix
/// did not perturb the previously-passing path.
const PINNED: &[(usize, usize, usize, u64)] = &[
    (500, 375, 14829, 0xb91f_a3dd_2aac_3ede), // ceil(500/8)  = 63  odd
    (1000, 750, 51259, 0x5a99_4bfe_5fc7_ce32), // ceil(1000/8) = 125 odd
    (1000, 1000, 67407, 0xd2fc_055a_4865_7f9e), // ceil(1000/8) = 125 odd
    (1080, 1080, 77119, 0xa28b_88a4_045e_7f65), // ceil(1080/8) = 135 odd
    (1200, 900, 72276, 0x69bf_152c_68e7_1887), // ceil(1200/8) = 150 even (control)
    (1920, 1080, 135493, 0xb08b_5058_4758_f0db), // ceil(1920/8) = 240 even (control)
];

#[test]
fn issue_314_odd_block_width_420_pinned_bytes() {
    let mut failures: Vec<String> = Vec::new();

    for &(width, height, expected_len, expected_hash) in PINNED {
        let encoded: Vec<u8> = encode_420(width, height);
        let actual_hash: u64 = fnv1a64(&encoded);
        if encoded.len() != expected_len || actual_hash != expected_hash {
            let blocks_across: usize = width.div_ceil(8);
            failures.push(format!(
                "  {width}x{height} (ceil(w/8)={blocks_across}, {}): \
                 expected {expected_len} bytes / {expected_hash:016x}, \
                 got {} bytes / {actual_hash:016x}",
                if blocks_across.is_multiple_of(2) {
                    "even"
                } else {
                    "odd"
                },
                encoded.len(),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "4:2:0 output no longer matches the cjpeg-verified reference (issue #314):\n{}",
        failures.join("\n")
    );
}

#[test]
fn issue_314_odd_block_width_420_matches_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");

    // Small enough to keep the test fast, chosen so every partial-MCU residue
    // that flips `ceil(width/8)` parity is represented.
    let geometries: &[(usize, usize)] = &[
        (7, 16),
        (8, 24),
        (17, 32),
        (23, 17),
        (24, 48),
        (33, 17),
        (500, 375),
        // Controls: `ceil(width/8)` even, correct before the fix too.
        (16, 16),
        (32, 32),
        (48, 48),
    ];

    let mut failures: Vec<String> = Vec::new();

    for &(width, height) in geometries {
        let pixels: Vec<u8> = synthetic_pixels(width, height);
        let rust_jpeg: Vec<u8> = encode_420(width, height);

        let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.extend_from_slice(&pixels);
        let c_jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
            &cjpeg,
            &ppm,
            &["-quality", "50", "-sample", "2x2", "-dct", "int"],
            &format!("issue314_{width}x{height}"),
        );

        if rust_jpeg != c_jpeg {
            let blocks_across: usize = width.div_ceil(8);
            let first_diff: usize = rust_jpeg
                .iter()
                .zip(c_jpeg.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(rust_jpeg.len().min(c_jpeg.len()));
            failures.push(format!(
                "  {width}x{height} (ceil(w/8)={blocks_across}): rust={} bytes, \
                 c={} bytes, first difference at offset {first_diff}",
                rust_jpeg.len(),
                c_jpeg.len(),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "4:2:0 encode diverged from cjpeg at {} of {} geometries (issue #314):\n{}",
        failures.len(),
        geometries.len(),
        failures.join("\n")
    );
}

/// Issue #316: `compress_with_restart` / `compress_custom_quant` /
/// `compress_custom_huffman` used a full-plane strategy that never implemented
/// the dummy-block contract, so they diverged from `cjpeg` on 204 of 576 swept
/// geometries — on every platform, unlike #314.
///
/// Fixed structurally by routing all four baseline entry points through the
/// `CompressParams` core (P4-40) rather than patching each copy.
#[test]
fn issue_316_full_plane_variants_match_cjpeg_at_partial_mcus() {
    let cjpeg = require_c_tool!("cjpeg");

    // Chroma-subsampled modes only: 4:4:4 has no dummy blocks. Geometries mix
    // `ceil(w/8)` and `ceil(h/8)` parities so both column and row dummies fire.
    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S422, "2x1"),
        (Subsampling::S420, "2x2"),
        (Subsampling::S440, "1x2"),
    ];
    let geometries: &[(usize, usize)] = &[
        (7, 8),
        (8, 24),
        (17, 17),
        (23, 32),
        (24, 15),
        (33, 33),
        (16, 16),
        (48, 48),
    ];

    let mut failures: Vec<String> = Vec::new();

    for &(subsampling, sample) in subsamplings {
        for &(width, height) in geometries {
            let pixels: Vec<u8> = synthetic_pixels(width, height);
            let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
            ppm.extend_from_slice(&pixels);

            // restart_interval = 0: isolates the buffering strategy, since no
            // RST markers are emitted and the stream should equal plain cjpeg.
            let rust_no_restart: Vec<u8> = pipeline::compress_with_restart(
                &pixels,
                width,
                height,
                PixelFormat::Rgb,
                50,
                subsampling,
                0,
                DctMethod::IsLow,
            )
            .expect("restart-capable encode");
            let c_no_restart: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &ppm,
                &["-quality", "50", "-sample", sample, "-dct", "int"],
                &format!("issue316_{width}x{height}_{sample}"),
            );
            if rust_no_restart != c_no_restart {
                failures.push(format!(
                    "  ri=0 {width}x{height} {sample}: rust={} c={}",
                    rust_no_restart.len(),
                    c_no_restart.len()
                ));
            }

            // A real restart interval. cjpeg's `B` suffix counts MCU blocks,
            // which is the unit our restart_interval uses.
            let rust_restart: Vec<u8> = pipeline::compress_with_restart(
                &pixels,
                width,
                height,
                PixelFormat::Rgb,
                50,
                subsampling,
                3,
                DctMethod::IsLow,
            )
            .expect("restart encode");
            let c_restart: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &ppm,
                &[
                    "-quality", "50", "-sample", sample, "-dct", "int", "-restart", "3B",
                ],
                &format!("issue316_rst_{width}x{height}_{sample}"),
            );
            if rust_restart != c_restart {
                failures.push(format!(
                    "  ri=3 {width}x{height} {sample}: rust={} c={}",
                    rust_restart.len(),
                    c_restart.len()
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "full-plane variants diverged from cjpeg at {} of {} checks (issue #316):\n{}",
        failures.len(),
        subsamplings.len() * geometries.len() * 2,
        failures.join("\n")
    );
}
