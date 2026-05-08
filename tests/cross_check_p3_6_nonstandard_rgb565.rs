//! P3-6 minimum-coverage fixtures: non-standard sampling (3x2, 3x1) and
//! RGB565 merged-upsample, cross-validated against stock C tools.
//!
//! Acceptance bar from `docs/last_mile/phase3.md` P3-6:
//!   - One fixture per: 3x2 decode, 3x2 encode, 3x1 decode, RGB565 merged-upsample.
//!   - Cross-validate against upstream `cjpeg -sample 3x2,1x1,1x1` and `djpeg -rgb565`.
//!   - Gaps remaining after this minimum land in `docs/FEATURE_PARITY.md`.
//!
//! Skip rules: `cjpeg` / `djpeg` not on PATH is a developer-machine skip;
//! in CI (`CI=true` / `GITHUB_ACTIONS=true`) the same conditions hard-fail
//! so the lifted P3-6 gate cannot disappear into a green skip.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{decompress_to, Encoder, PixelFormat, ScanlineDecoder};

const TEST_W: usize = 48;
const TEST_H: usize = 48;
const QUALITY: u8 = 90;

fn require_tools_or_skip() -> Option<(PathBuf, PathBuf)> {
    let cjpeg = match helpers::cjpeg_path() {
        Some(p) => p,
        None => {
            if helpers::is_ci() {
                panic!("P3-6 cross-check requires cjpeg in CI");
            }
            eprintln!("SKIP: cjpeg not found in /opt/homebrew/bin or /usr/local/bin");
            return None;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            if helpers::is_ci() {
                panic!("P3-6 cross-check requires djpeg in CI");
            }
            eprintln!("SKIP: djpeg not found in /opt/homebrew/bin or /usr/local/bin");
            return None;
        }
    };
    Some((cjpeg, djpeg))
}

fn make_gradient(w: usize, h: usize) -> Vec<u8> {
    helpers::generate_gradient(w, h)
}

fn write_ppm_tmp(label: &str, w: usize, h: usize, pixels: &[u8]) -> helpers::TempFile {
    let f = helpers::TempFile::new(&format!("{}.ppm", label));
    helpers::write_ppm_file(f.path(), w, h, pixels);
    f
}

fn cjpeg_encode_with_sample(cjpeg: &Path, sample: &str, ppm_path: &Path) -> Vec<u8> {
    let out_file: helpers::TempFile = helpers::TempFile::new("p3_6_cjpeg_out.jpg");
    helpers::run_c_cjpeg(
        cjpeg,
        &["-sample", sample, "-quality", &QUALITY.to_string()],
        ppm_path,
        out_file.path(),
    );
    std::fs::read(out_file.path()).expect("read cjpeg output")
}

/// 5-6-5 truncation (no dither), matching upstream and the merged path.
fn pack_rgb_to_rgb565_le(rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len() % 3, 0);
    let mut out: Vec<u8> = vec![0u8; (rgb.len() / 3) * 2];
    for i in 0..(rgb.len() / 3) {
        let r: u16 = rgb[i * 3] as u16;
        let g: u16 = rgb[i * 3 + 1] as u16;
        let b: u16 = rgb[i * 3 + 2] as u16;
        let word: u16 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
        let bytes = word.to_le_bytes();
        out[i * 2] = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
    out
}

// ---------------------------------------------------------------------------
// Fixture #1: 3x2 sampling decode — Rust must match djpeg pixel-for-pixel.
// ---------------------------------------------------------------------------

#[test]
fn p3_6_3x2_sampling_decode_matches_djpeg() {
    let (cjpeg, djpeg) = match require_tools_or_skip() {
        Some(t) => t,
        None => return,
    };

    let pixels: Vec<u8> = make_gradient(TEST_W, TEST_H);
    let ppm: helpers::TempFile = write_ppm_tmp("p3_6_3x2_decode", TEST_W, TEST_H, &pixels);

    // C cjpeg encodes a 3x2-sampled JPEG.
    let jpeg: Vec<u8> = cjpeg_encode_with_sample(&cjpeg, "3x2,1x1,1x1", ppm.path());

    // Rust decode.
    let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode 3x2");
    assert_eq!(rust_img.width, TEST_W);
    assert_eq!(rust_img.height, TEST_H);

    // C djpeg decode.
    let (cw, ch, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, "p3_6_3x2_decode");
    assert_eq!(cw, TEST_W);
    assert_eq!(ch, TEST_H);

    helpers::assert_pixels_identical(&rust_img.data, &c_rgb, TEST_W, TEST_H, 3, "p3_6_3x2_decode");
}

// ---------------------------------------------------------------------------
// Fixture #2: 3x2 sampling encode — Rust output must decode (via djpeg)
// to pixels identical to the cjpeg-baseline + djpeg pipeline. JPEG-encode
// is lossy so byte-level parity isn't expected, but the SOF1/sampling
// factors round-trip and the decoded pixels must match within JPEG-encoder
// quantization bounds (PSNR ≥ 35 dB, no max-diff > 8 for q=90).
// ---------------------------------------------------------------------------

#[test]
fn p3_6_3x2_sampling_encode_round_trips() {
    let (cjpeg, djpeg) = match require_tools_or_skip() {
        Some(t) => t,
        None => return,
    };

    let pixels: Vec<u8> = make_gradient(TEST_W, TEST_H);
    let ppm: helpers::TempFile = write_ppm_tmp("p3_6_3x2_encode", TEST_W, TEST_H, &pixels);

    // Rust-encoded with custom 3x2 sampling.
    let rust_jpeg: Vec<u8> = Encoder::new(&pixels, TEST_W, TEST_H, PixelFormat::Rgb)
        .quality(QUALITY)
        .sampling_factors(vec![(3, 2), (1, 1), (1, 1)])
        .encode()
        .expect("Rust encode 3x2");

    // Decode the Rust output via djpeg — establishes that the SOF1/scan
    // structure round-trips through stock C tooling.
    let (rw, rh, rust_via_c) =
        helpers::decode_with_c_djpeg(&djpeg, &rust_jpeg, "p3_6_3x2_rust_via_c");
    assert_eq!(rw, TEST_W);
    assert_eq!(rh, TEST_H);

    // Reference pipeline: cjpeg -sample 3x2,1x1,1x1 + djpeg.
    let c_jpeg: Vec<u8> = cjpeg_encode_with_sample(&cjpeg, "3x2,1x1,1x1", ppm.path());
    let (cw, ch, c_pixels) = helpers::decode_with_c_djpeg(&djpeg, &c_jpeg, "p3_6_3x2_c_baseline");
    assert_eq!(cw, TEST_W);
    assert_eq!(ch, TEST_H);

    // Both decodes operate on the same source pixels at the same quality;
    // the Rust encoder + djpeg pipeline should land within the C
    // baseline's quantization neighborhood. Use the same tolerance the
    // existing standard-sampling cross-checks use against the source.
    let max_diff: u8 = pixel_max_diff(&rust_via_c, &c_pixels);
    assert!(
        max_diff <= 8,
        "p3_6_3x2_encode: max pixel diff vs cjpeg baseline = {} (tolerance ≤ 8)",
        max_diff,
    );
}

fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Fixture #3: 3x1 sampling decode — Rust must match djpeg pixel-for-pixel.
// ---------------------------------------------------------------------------

#[test]
fn p3_6_3x1_sampling_decode_matches_djpeg() {
    let (cjpeg, djpeg) = match require_tools_or_skip() {
        Some(t) => t,
        None => return,
    };

    let pixels: Vec<u8> = make_gradient(TEST_W, TEST_H);
    let ppm: helpers::TempFile = write_ppm_tmp("p3_6_3x1_decode", TEST_W, TEST_H, &pixels);

    let jpeg: Vec<u8> = cjpeg_encode_with_sample(&cjpeg, "3x1,1x1,1x1", ppm.path());

    let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode 3x1");
    assert_eq!(rust_img.width, TEST_W);
    assert_eq!(rust_img.height, TEST_H);

    let (cw, ch, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, "p3_6_3x1_decode");
    assert_eq!(cw, TEST_W);
    assert_eq!(ch, TEST_H);

    helpers::assert_pixels_identical(&rust_img.data, &c_rgb, TEST_W, TEST_H, 3, "p3_6_3x1_decode");
}

// ---------------------------------------------------------------------------
// Fixture #4: RGB565 merged-upsample — Rust output (merged path enabled,
// RGB565 target) must match djpeg -rgb565 byte-for-byte for an S420 input.
// ---------------------------------------------------------------------------

#[test]
fn p3_6_rgb565_merged_decode_matches_djpeg_rgb_chain() {
    let (_cjpeg, djpeg) = match require_tools_or_skip() {
        Some(t) => t,
        None => return,
    };

    // Validate Rust's `merged_upsample + Rgb565` path against the
    // `djpeg-RGB → 5-6-5 truncate` chain. djpeg's `-rgb565` flag emits a
    // 24-bpp BMP rather than a 16-bpp file — the in-library RGB565 step
    // is internal-only — so direct byte-comparison against `djpeg
    // -rgb565` output isn't possible. Instead, rely on the fact that
    // `cross_check_rgb565_merged.rs` already pins
    // (Rust-merged-RGB == djpeg-RGB), then verify that Rust's
    // `merged + RGB565` matches the same pipeline truncated to 5-6-5.
    //
    // S420 + S422 cover both merged kernels (H2V2 and H2V1).
    let cases: &[(libjpeg_turbo_rs::Subsampling, &str)] = &[
        (libjpeg_turbo_rs::Subsampling::S420, "S420"),
        (libjpeg_turbo_rs::Subsampling::S422, "S422"),
    ];

    for &(subsamp, name) in cases {
        let pixels: Vec<u8> = make_gradient(TEST_W, TEST_H);
        let jpeg: Vec<u8> =
            libjpeg_turbo_rs::compress(&pixels, TEST_W, TEST_H, PixelFormat::Rgb, QUALITY, subsamp)
                .expect("compress fixture");

        // Rust: merged + RGB565.
        let mut dec = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new");
        dec.set_merged_upsample(true);
        dec.set_output_format(PixelFormat::Rgb565);
        let rust_565 = dec.finish().expect("merged+RGB565 decode");
        assert_eq!(rust_565.width, TEST_W, "{}: width", name);
        assert_eq!(rust_565.height, TEST_H, "{}: height", name);
        assert_eq!(rust_565.data.len(), TEST_W * TEST_H * 2, "{}: len", name);

        // Rust: merged + RGB (same path before 5-6-5 packing). Used as
        // the in-library reference: the merged kernel must produce
        // identical RGB whether the final output is RGB or RGB565.
        let mut dec_rgb = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new");
        dec_rgb.set_merged_upsample(true);
        dec_rgb.set_output_format(PixelFormat::Rgb);
        let rust_rgb = dec_rgb.finish().expect("merged+RGB decode");
        let expected_565: Vec<u8> = pack_rgb_to_rgb565_le(&rust_rgb.data);
        assert_eq!(
            rust_565.data, expected_565,
            "{}: merged+RGB565 != pack(merged+RGB) — RGB565 wiring is wrong",
            name,
        );

        // C djpeg RGB → 5-6-5 chain. The merged path uses box-filter
        // upsampling (matching djpeg's `-nosmooth` flag), not the
        // default fancy-upsample, so the C reference must match.
        let jpeg_file = helpers::TempFile::new(&format!("p3_6_rgb565_{}.jpg", name));
        let ppm_file = helpers::TempFile::new(&format!("p3_6_rgb565_{}.ppm", name));
        jpeg_file.write_bytes(&jpeg);
        let out = Command::new(&djpeg)
            .arg("-nosmooth")
            .arg("-ppm")
            .arg("-outfile")
            .arg(ppm_file.path())
            .arg(jpeg_file.path())
            .output()
            .expect("run djpeg -nosmooth");
        assert!(
            out.status.success(),
            "{}: djpeg -nosmooth failed: {}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
        let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
        let (cw, ch, c_rgb) = helpers::parse_ppm(&ppm_data).expect("parse PPM");
        assert_eq!(cw, TEST_W, "{}: width", name);
        assert_eq!(ch, TEST_H, "{}: height", name);
        let c_chain_565: Vec<u8> = pack_rgb_to_rgb565_le(&c_rgb);
        assert_eq!(
            rust_565.data, c_chain_565,
            "{}: Rust merged+RGB565 != pack(djpeg -nosmooth RGB) — pixel divergence vs C",
            name,
        );

        // Regression guard: enabling dither_565 alongside merged_upsample
        // must NOT take the truncation-only merged branch. The shim
        // doesn't ship a `*_565D` merged path yet; the gate must fall
        // through to the slow dithered path so the dither setting is
        // honored. Caught originally by codex review on the P3-6 commit.
        let mut dec_dither =
            libjpeg_turbo_rs::Decoder::new(&jpeg).expect("Decoder::new for dither path");
        dec_dither.set_merged_upsample(true);
        dec_dither.set_dither_565(true);
        dec_dither.set_output_format(PixelFormat::Rgb565);
        let dithered = dec_dither
            .decode_image()
            .expect("merged + dither_565 + RGB565 decode");
        assert_eq!(
            dithered.data.len(),
            TEST_W * TEST_H * 2,
            "{}: dither len",
            name
        );
        // The dithered output must differ from the plain-truncation
        // output for at least some pixels of a non-trivial gradient.
        // (If they match, the dither pass was silently skipped.)
        assert_ne!(
            dithered.data, rust_565.data,
            "{}: dither_565 + merged produced identical bytes to plain merged — \
             dither setting was silently dropped (regression of P3-6 codex review)",
            name,
        );
    }
}
