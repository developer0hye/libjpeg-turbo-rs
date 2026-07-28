//! Programmatic fuzz seed generator.
//!
//! Populates `fuzz/corpus/<target>/` with a large variety of structurally valid
//! JPEG files so libFuzzer can start mutation from meaningful inputs rather
//! than pure random bytes. Seeds cover the Cartesian product of
//! subsampling * quality * content * entropy-coding-mode, plus a handful of
//! hand-rolled minimal-JPEG byte sequences for edge cases.
//!
//! Run with:
//!   cargo test --test generate_fuzz_seeds
//!
//! The test is intentionally no longer `#[ignore]` — it runs in a few seconds
//! and keeping the committed corpus in sync with the generator matters for
//! reproducible fuzzing.

use std::fs;
use std::path::{Path, PathBuf};

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless,
    compress_lossless_arithmetic, compress_lossless_extended, compress_progressive, PixelFormat,
    Subsampling,
};

/// Width/height used for every synthetic seed. Small enough to keep the
/// corpus cheap, large enough to exercise >1 MCU for every subsampling.
const SEED_W: usize = 16;
const SEED_H: usize = 16;

/// Decoder-facing fuzz targets that receive every generated JPEG. The
/// `fuzz_roundtrip` and `fuzz_encode_roundtrip` targets use `Arbitrary`, but
/// structured JPEG bytes still make useful mutation bases.
///
/// `fuzz_transform_options` expects a 10-byte structured header prepended to
/// the JPEG; it is seeded separately via `transform_options_seeds()` below.
const DECODER_TARGETS: &[&str] = &[
    "fuzz_decompress",
    "fuzz_decompress_lenient",
    "fuzz_read_coefficients",
    "fuzz_transform",
    "fuzz_progressive_decoder",
    // P2-7: differential decode-vs-djpeg target. Same input shape as
    // the other decoder targets (raw JPEG bytes, no header), so it
    // benefits from the same seed corpus.
    "fuzz_decode_diff_c",
    // Issue #382: 12/16-bit + arbitrary-precision decode entry points.
    // Raw JPEG bytes like the rest; the generated corpus gives it more
    // than its three hand-crafted lossless seeds.
    "fuzz_decompress_precision",
];

#[derive(Clone, Copy)]
enum Content {
    Gradient,
    Checker,
    SyntheticPhoto,
}

impl Content {
    fn label(self) -> &'static str {
        match self {
            Content::Gradient => "grad",
            Content::Checker => "chk",
            Content::SyntheticPhoto => "photo",
        }
    }

    fn pixels_rgb(self, w: usize, h: usize) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(w * h * 3);
        match self {
            Content::Gradient => {
                for y in 0..h {
                    for x in 0..w {
                        let r: u8 = ((x * 255) / w.max(1)) as u8;
                        let g: u8 = ((y * 255) / h.max(1)) as u8;
                        let b: u8 = (((x + y) * 255) / (w + h).max(1)) as u8;
                        buf.push(r);
                        buf.push(g);
                        buf.push(b);
                    }
                }
            }
            Content::Checker => {
                let tile: usize = 4;
                for y in 0..h {
                    for x in 0..w {
                        let on: bool = ((x / tile) + (y / tile)).is_multiple_of(2);
                        let v: u8 = if on { 230 } else { 30 };
                        buf.push(v);
                        buf.push(v);
                        buf.push(v);
                    }
                }
            }
            Content::SyntheticPhoto => {
                // Deterministic smooth-ish signal so the content compresses
                // like a real photo (mid-frequency energy).
                for y in 0..h {
                    for x in 0..w {
                        let fx: f32 = x as f32 / w.max(1) as f32;
                        let fy: f32 = y as f32 / h.max(1) as f32;
                        let r: u8 = ((128.0 + 80.0 * (fx * 6.0).sin() + 40.0 * (fy * 3.0).cos())
                            .clamp(0.0, 255.0)) as u8;
                        let g: u8 =
                            ((128.0 + 70.0 * (fx * 4.0 + 1.2).cos() + 50.0 * (fy * 5.0).sin())
                                .clamp(0.0, 255.0)) as u8;
                        let b: u8 = ((128.0
                            + 60.0 * (fx * 2.5 + fy * 2.5).sin()
                            + 40.0 * (fy * 7.0 + 0.5).cos())
                        .clamp(0.0, 255.0)) as u8;
                        buf.push(r);
                        buf.push(g);
                        buf.push(b);
                    }
                }
            }
        }
        buf
    }

    fn pixels_gray(self, w: usize, h: usize) -> Vec<u8> {
        // Rec.601 luma of the RGB version, so content stays comparable.
        let rgb: Vec<u8> = self.pixels_rgb(w, h);
        let mut gray: Vec<u8> = Vec::with_capacity(w * h);
        for px in rgb.chunks_exact(3) {
            let y: u32 =
                (299 * px[0] as u32 + 587 * px[1] as u32 + 114 * px[2] as u32 + 500) / 1000;
            gray.push(y.min(255) as u8);
        }
        gray
    }
}

#[derive(Clone, Copy)]
enum SubsampLabel {
    S420,
    S422,
    S444,
    S440,
    S411,
    S441,
    Gray,
}

impl SubsampLabel {
    fn all() -> &'static [SubsampLabel] {
        &[
            SubsampLabel::S420,
            SubsampLabel::S422,
            SubsampLabel::S444,
            SubsampLabel::S440,
            SubsampLabel::S411,
            SubsampLabel::S441,
            SubsampLabel::Gray,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            SubsampLabel::S420 => "420",
            SubsampLabel::S422 => "422",
            SubsampLabel::S444 => "444",
            SubsampLabel::S440 => "440",
            SubsampLabel::S411 => "411",
            SubsampLabel::S441 => "441",
            SubsampLabel::Gray => "gray",
        }
    }

    fn subsampling(self) -> Subsampling {
        match self {
            SubsampLabel::S420 => Subsampling::S420,
            SubsampLabel::S422 => Subsampling::S422,
            SubsampLabel::S444 => Subsampling::S444,
            SubsampLabel::S440 => Subsampling::S440,
            SubsampLabel::S411 => Subsampling::S411,
            SubsampLabel::S441 => Subsampling::S441,
            // Gray encodes as a 1-component JPEG; the value is ignored.
            SubsampLabel::Gray => Subsampling::S444,
        }
    }

    fn is_gray(self) -> bool {
        matches!(self, SubsampLabel::Gray)
    }
}

#[derive(Clone, Copy)]
enum Entropy {
    Baseline,
    Progressive,
    Arithmetic,
    ArithProgressive,
    Lossless,
    LosslessArithmetic,
}

impl Entropy {
    fn all() -> &'static [Entropy] {
        &[
            Entropy::Baseline,
            Entropy::Progressive,
            Entropy::Arithmetic,
            Entropy::ArithProgressive,
            Entropy::Lossless,
            Entropy::LosslessArithmetic,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Entropy::Baseline => "base",
            Entropy::Progressive => "prog",
            Entropy::Arithmetic => "arith",
            Entropy::ArithProgressive => "aprog",
            Entropy::Lossless => "ls",
            Entropy::LosslessArithmetic => "lsar",
        }
    }
}

/// Encode the requested combination. Returns `None` when the combination is
/// unsupported by the current encoder (e.g. lossless with subsampled chroma);
/// the caller should simply skip.
fn encode_seed(
    content: Content,
    sub: SubsampLabel,
    quality: u8,
    entropy: Entropy,
) -> Option<Vec<u8>> {
    let w: usize = SEED_W;
    let h: usize = SEED_H;

    if sub.is_gray() {
        let pixels: Vec<u8> = content.pixels_gray(w, h);
        let pf: PixelFormat = PixelFormat::Grayscale;
        let subs: Subsampling = Subsampling::S444;
        return match entropy {
            Entropy::Baseline => compress(&pixels, w, h, pf, quality, subs).ok(),
            Entropy::Progressive => compress_progressive(&pixels, w, h, pf, quality, subs).ok(),
            Entropy::Arithmetic => compress_arithmetic(&pixels, w, h, pf, quality, subs).ok(),
            Entropy::ArithProgressive => {
                compress_arithmetic_progressive(&pixels, w, h, pf, quality, subs).ok()
            }
            Entropy::Lossless => compress_lossless(&pixels, w, h, pf).ok(),
            Entropy::LosslessArithmetic => {
                compress_lossless_arithmetic(&pixels, w, h, pf, 1, 0).ok()
            }
        };
    }

    let pixels: Vec<u8> = content.pixels_rgb(w, h);
    let pf: PixelFormat = PixelFormat::Rgb;
    let subs: Subsampling = sub.subsampling();

    match entropy {
        Entropy::Baseline => compress(&pixels, w, h, pf, quality, subs).ok(),
        Entropy::Progressive => compress_progressive(&pixels, w, h, pf, quality, subs).ok(),
        Entropy::Arithmetic => compress_arithmetic(&pixels, w, h, pf, quality, subs).ok(),
        Entropy::ArithProgressive => {
            compress_arithmetic_progressive(&pixels, w, h, pf, quality, subs).ok()
        }
        // Lossless only well-defined with no subsampling; skip the rest.
        Entropy::Lossless => {
            if matches!(sub, SubsampLabel::S444) {
                compress_lossless_extended(&pixels, w, h, pf, 1, 0).ok()
            } else {
                None
            }
        }
        Entropy::LosslessArithmetic => {
            if matches!(sub, SubsampLabel::S444) {
                compress_lossless_arithmetic(&pixels, w, h, pf, 1, 0).ok()
            } else {
                None
            }
        }
    }
}

/// The committed corpus is canonicalized to this platform's generator
/// output. P4-33 root cause: encoder output is backend-dependent for
/// partial-MCU 4:2:0 shapes (the 8×64 `aspect_wide_*` seeds encode to
/// different — pixel-identical — bytes under NEON than under the x86
/// paths), so a byte-exact committed corpus can only ever match ONE
/// backend. x86_64-linux is the canonical one (matches the main CI
/// runners); everywhere else the generator must leave committed seeds
/// untouched instead of churning the working tree.
const CANONICAL_SEED_PLATFORM: bool = cfg!(all(target_arch = "x86_64", target_os = "linux"));

/// Seeds whose on-disk bytes differed from the freshly generated bytes
/// on the canonical platform. P4-33: the committed `aspect_*` seeds were
/// generated on an aarch64 host and never matched the x86_64 generator
/// output; every full test run silently rewrote them. The test now fails
/// on canonical-platform drift (see the assertion in `generate_seeds`)
/// so a generator/encoder change that alters seed bytes must re-commit
/// the seeds in the PR that introduces it.
///
/// The guard relies on two invariants of this file: (1) `generate_seeds`
/// is the ONLY `#[test]` in this binary — a second parallel test sharing
/// this static could race the end-of-test drain; (2) each `(target,
/// name)` pair is written at most once per run (or always with identical
/// bytes) — a same-path/different-bytes double write would record
/// spurious drift even on a fresh checkout.
static DRIFTED_SEEDS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Write a seed file into one or more target corpus directories, skipping
/// no-op rewrites so mtime stays stable when the content is unchanged.
///
/// On the canonical platform, overwrites of existing files with different
/// content are recorded in [`DRIFTED_SEEDS`]; brand-new files are not (a
/// fresh checkout without the corpus must still generate cleanly). On
/// every other platform an existing file is left untouched — its bytes
/// are the canonical ones, and backend-dependent encodings must not
/// churn the tree (they decode pixel-identically; either variant is an
/// equally good fuzz seed).
fn fan_out_write(name: &str, bytes: &[u8], corpus_base: &Path, extra_targets: &[&str]) {
    for target in extra_targets {
        let target_dir: PathBuf = corpus_base.join(target);
        fs::create_dir_all(&target_dir).expect("failed to create corpus directory");
        let dest: PathBuf = target_dir.join(name);
        if let Ok(existing) = fs::read(&dest) {
            if existing == bytes {
                continue;
            }
            if !CANONICAL_SEED_PLATFORM {
                continue;
            }
            DRIFTED_SEEDS
                .lock()
                .expect("drift list poisoned")
                .push(dest.display().to_string());
        }
        fs::write(&dest, bytes).expect("failed to write seed file");
    }
}

/// Handcrafted byte sequences targeting specific decoder edge cases.
fn structural_edge_seeds() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        // Bare SOI+EOI — shortest legal JPEG skeleton.
        ("soi_eoi.bin", vec![0xFF, 0xD8, 0xFF, 0xD9]),
        // SOI + COM(0 length) + EOI.
        (
            "soi_com_eoi.bin",
            vec![0xFF, 0xD8, 0xFF, 0xFE, 0x00, 0x02, 0xFF, 0xD9],
        ),
        // SOI + truncated SOF0 marker — exercises short-read recovery.
        (
            "truncated_sof0.bin",
            vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x05, 0x08],
        ),
        // SOI + APP0 JFIF header + EOI (no frame).
        (
            "soi_app0_eoi.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x02, 0x00,
                0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9,
            ],
        ),
        // SOI + DRI with a non-zero restart interval + EOI.
        (
            "soi_dri_eoi.bin",
            vec![0xFF, 0xD8, 0xFF, 0xDD, 0x00, 0x04, 0x00, 0x08, 0xFF, 0xD9],
        ),
        // SOI + DNL marker (define number of lines) + EOI.
        // DNL is a rarely-parsed segment; exercises that code path.
        (
            "soi_dnl_eoi.bin",
            vec![0xFF, 0xD8, 0xFF, 0xDC, 0x00, 0x04, 0x00, 0x10, 0xFF, 0xD9],
        ),
        // SOI + two COM segments + EOI — multiple markers of same type.
        (
            "soi_two_com_eoi.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xFE, 0x00, 0x04, b'A', b'B', 0xFF, 0xFE, 0x00, 0x04, b'C', b'D',
                0xFF, 0xD9,
            ],
        ),
        // SOI + APP1 (Exif header only, no IFD) + EOI.
        (
            "soi_app1_exif_eoi.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x08, b'E', b'x', b'i', b'f', 0x00, 0x00, 0xFF, 0xD9,
            ],
        ),
        // SOI + APP2 (ICC profile signature only) + EOI.
        (
            "soi_app2_icc_eoi.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xE2, 0x00, 0x10, b'I', b'C', b'C', b'_', b'P', b'R', b'O', b'F',
                b'I', b'L', b'E', 0x00, 0x01, 0x01, 0xFF, 0xD9,
            ],
        ),
        // SOI + APP14 Adobe color transform marker (value 0=unknown) + EOI.
        (
            "soi_app14_adobe_eoi.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x00, 0x64,
                0x00, 0x00, 0x00, 0x00, 0xFF, 0xD9,
            ],
        ),
        // SOI + SOF0 with zero-width image — tests dimension-zero guard.
        (
            "sof0_zero_width.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x00, 0x01, 0x01, 0x11,
                0x00, 0xFF, 0xD9,
            ],
        ),
        // SOI + SOF0 with zero-height image — tests dimension-zero guard.
        (
            "sof0_zero_height.bin",
            vec![
                0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x00, 0x00, 0x08, 0x01, 0x01, 0x11,
                0x00, 0xFF, 0xD9,
            ],
        ),
        // Bare SOI with no EOI and no subsequent bytes — exercises EOF-mid-header path.
        ("bare_soi.bin", vec![0xFF, 0xD8]),
        // 0xFF bytes only — exercises the fill-byte skip path in the marker scanner.
        (
            "all_ff_bytes.bin",
            vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
        // Random non-JPEG bytes — fuzzer should not panic on totally invalid input.
        (
            "not_jpeg.bin",
            vec![0x00, 0x01, 0x02, 0x03, 0x7F, 0x80, 0xFE, 0x42],
        ),
    ]
}

/// Build a JPEG with restart markers (DRI + RST0..RST7) for a given content
/// type. Restart intervals stress the resync/restart-marker parsing paths.
fn make_restart_jpeg(content: Content, sub: SubsampLabel, quality: u8) -> Option<Vec<u8>> {
    // We use the encode_seed path to get a valid baseline JPEG, then inject a
    // DRI segment directly after SOI. This is simpler than wiring restart
    // through the public API and still exercises the decoder's restart path.
    let baseline: Vec<u8> = encode_seed(content, sub, quality, Entropy::Baseline)?;
    // Insert DRI(interval=2) immediately after the SOI marker (bytes 0..2).
    let dri: [u8; 6] = [0xFF, 0xDD, 0x00, 0x04, 0x00, 0x02]; // DRI, len=4, interval=2
    let mut out: Vec<u8> = Vec::with_capacity(baseline.len() + dri.len());
    out.extend_from_slice(&baseline[..2]); // SOI
    out.extend_from_slice(&dri);
    out.extend_from_slice(&baseline[2..]);
    Some(out)
}

/// Build seeds for `fuzz_transform_options` by prepending the 10-byte
/// structured header to various JPEG payloads.
///
/// Header layout (see fuzz_targets/fuzz_transform_options.rs):
///   [0] op_idx, [1] flags, [2..3] restart_interval (LE), [4] copy_mode,
///   [5..8] crop fractions, [9] perfect flag.
fn transform_options_seeds(corpus_base: &Path) {
    let target: &str = "fuzz_transform_options";

    // Helper: prepend a 10-byte header to a JPEG and write to target corpus.
    let write_seed = |name: &str, header: [u8; 10], jpeg: &[u8]| {
        let mut bytes: Vec<u8> = Vec::with_capacity(10 + jpeg.len());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(jpeg);
        fan_out_write(name, &bytes, corpus_base, &[target]);
    };

    // Produce source JPEGs for various dimensions and modes.
    let sources: &[(Content, SubsampLabel, u8, Entropy)] = &[
        // Square 16×16 — minimal MCU size that exercises all ops.
        (Content::Gradient, SubsampLabel::S420, 75, Entropy::Baseline),
        (Content::Checker, SubsampLabel::S444, 75, Entropy::Baseline),
        (
            Content::SyntheticPhoto,
            SubsampLabel::S420,
            90,
            Entropy::Progressive,
        ),
        (
            Content::Gradient,
            SubsampLabel::S422,
            50,
            Entropy::Arithmetic,
        ),
        // Grayscale — exercises grayscale-to-grayscale transform.
        (Content::Checker, SubsampLabel::Gray, 75, Entropy::Baseline),
        // Lossless RGB 444 — exercises lossless transform.
        (
            Content::SyntheticPhoto,
            SubsampLabel::S444,
            75,
            Entropy::Lossless,
        ),
    ];

    // TransformOp indices: 0=None,1=HFlip,2=VFlip,3=Transpose,4=Transverse,
    //                       5=Rot90,6=Rot180,7=Rot270.
    // flags: trim=0x01, grayscale=0x02, progressive=0x04, arithmetic=0x08,
    //        optimize=0x10, no_output=0x20, restart_in_rows=0x40, crop=0x80.
    let option_combos: &[(&str, [u8; 10])] = &[
        // No-op transform, all defaults.
        ("opt_none_default.bin", [0, 0x00, 0, 0, 0, 0, 0, 0, 0, 0]),
        // HFlip with copy_markers=None.
        ("opt_hflip_copynone.bin", [1, 0x00, 0, 0, 1, 0, 0, 0, 0, 0]),
        // VFlip with trim.
        ("opt_vflip_trim.bin", [2, 0x01, 0, 0, 0, 0, 0, 0, 0, 0]),
        // Rot90, grayscale output.
        ("opt_rot90_gray.bin", [5, 0x02, 0, 0, 0, 0, 0, 0, 0, 0]),
        // Rot180, progressive re-encode.
        ("opt_rot180_prog.bin", [6, 0x04, 0, 0, 0, 0, 0, 0, 0, 0]),
        // Rot270, arithmetic re-encode.
        ("opt_rot270_arith.bin", [7, 0x08, 0, 0, 0, 0, 0, 0, 0, 0]),
        // Transpose, optimize.
        ("opt_transpose_opt.bin", [3, 0x10, 0, 0, 0, 0, 0, 0, 0, 0]),
        // HFlip with restart interval 4 (restart_in_rows=false).
        ("opt_hflip_restart4.bin", [1, 0x00, 4, 0, 0, 0, 0, 0, 0, 0]),
        // None with restart_in_rows flag + interval 2.
        (
            "opt_none_restart_rows.bin",
            [0, 0x40, 2, 0, 0, 0, 0, 0, 0, 0],
        ),
        // Crop enabled at roughly centre quarter (fracs: x=64,y=64,w=128,h=128).
        (
            "opt_hflip_crop_centre.bin",
            [1, 0x80, 0, 0, 0, 64, 64, 128, 128, 0],
        ),
        // Crop near origin (small region).
        (
            "opt_rot90_crop_origin.bin",
            [5, 0x80, 0, 0, 0, 0, 0, 32, 32, 0],
        ),
        // No-op with no_output flag — validation only.
        ("opt_nooutput.bin", [0, 0x20, 0, 0, 0, 0, 0, 0, 0, 0]),
        // Copy IccOnly markers.
        ("opt_hflip_icc.bin", [1, 0x00, 0, 0, 2, 0, 0, 0, 0, 0]),
        // Perfect flag set (will Err if dims not MCU-aligned, that's fine).
        ("opt_rot90_perfect.bin", [5, 0x00, 0, 0, 0, 0, 0, 0, 0, 1]),
        // Combination: progressive + optimize + trim.
        (
            "opt_none_prog_opt_trim.bin",
            [0, 0x15, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ];

    for &(content, sub, quality, entropy) in sources {
        let Some(jpeg) = encode_seed(content, sub, quality, entropy) else {
            continue;
        };
        let src_label: String = format!(
            "{c}_{s}_{quality}_{e}",
            c = content.label(),
            s = sub.label(),
            e = entropy.label(),
        );
        for &(opt_label, header) in option_combos {
            let name: String = format!("txopt_{src_label}_{opt_label}");
            write_seed(&name, header, &jpeg);
        }
    }

    // Also seed with the structural edge-case JPEGs to exercise error paths.
    for (edge_name, edge_bytes) in structural_edge_seeds() {
        // Use a no-op header so the interesting bytes are in the JPEG part.
        let header: [u8; 10] = [0, 0x00, 0, 0, 0, 0, 0, 0, 0, 0];
        let name: String = format!("txopt_edge_{edge_name}");
        write_seed(&name, header, &edge_bytes);
    }
}

#[test]
fn generate_seeds() {
    let corpus_base: &Path = Path::new("fuzz/corpus");

    // Create every target directory up front.
    let mut all_targets: Vec<&str> = DECODER_TARGETS.to_vec();
    all_targets.push("fuzz_roundtrip");
    all_targets.push("fuzz_encode_roundtrip");
    all_targets.push("fuzz_transform_options");
    // P2-7 follow-ups: structured-input differential targets need their
    // own corpus directories so cargo-fuzz finds the seeds we plant
    // below. Same input shapes as the unprefixed targets:
    //   fuzz_encode_diff_c    — 4-byte header + raw pixels (mirrors
    //                           fuzz_encode_roundtrip)
    //   fuzz_transform_diff_c — 1-byte op + JPEG (subset of
    //                           fuzz_transform_options' header)
    all_targets.push("fuzz_encode_diff_c");
    all_targets.push("fuzz_transform_diff_c");
    for t in &all_targets {
        fs::create_dir_all(corpus_base.join(t)).expect("failed to create corpus directory");
    }

    // Preserve existing real-world fixture seeds when available.
    let fixtures: &[&str] = &[
        "tests/fixtures/gray_8x8.jpg",
        "tests/fixtures/red_16x16_444.jpg",
        "tests/fixtures/blue_16x16_420.jpg",
        "tests/fixtures/photo_64x64_420.jpg",
        "tests/fixtures/blue_16x16_420_prog.jpg",
        "tests/fixtures/checker_640x480_420.jpg",
        "tests/fixtures/checker_640x480_420_prog.jpg",
    ];
    for fixture_path in fixtures {
        let src: &Path = Path::new(fixture_path);
        if !src.exists() {
            continue;
        }
        let filename: &str = src
            .file_name()
            .and_then(|s| s.to_str())
            .expect("fixture filename");
        let bytes: Vec<u8> = fs::read(src).expect("failed to read fixture");
        fan_out_write(filename, &bytes, corpus_base, DECODER_TARGETS);
    }

    // Full matrix: content × subsampling × quality × entropy.
    let contents: &[Content] = &[Content::Gradient, Content::Checker, Content::SyntheticPhoto];
    let qualities: &[u8] = &[10, 50, 90];

    let mut generated: usize = 0;
    let mut skipped: usize = 0;

    for &content in contents {
        for &sub in SubsampLabel::all() {
            for &quality in qualities {
                for &entropy in Entropy::all() {
                    let Some(jpeg) = encode_seed(content, sub, quality, entropy) else {
                        skipped += 1;
                        continue;
                    };
                    let name: String = format!(
                        "{c}_{sub}_q{q}_{e}.jpg",
                        c = content.label(),
                        sub = sub.label(),
                        q = quality,
                        e = entropy.label(),
                    );
                    fan_out_write(&name, &jpeg, corpus_base, DECODER_TARGETS);
                    fan_out_write(&name, &jpeg, corpus_base, &["fuzz_roundtrip"]);
                    fan_out_write(&name, &jpeg, corpus_base, &["fuzz_encode_roundtrip"]);
                    // P2-7: same trick as fuzz_encode_roundtrip — the JPEG
                    // bytes are interpreted as `[4-byte header][pixel bytes]`
                    // by the differential target. This gives libfuzzer a
                    // realistic byte distribution to mutate from instead
                    // of starting from a blank corpus.
                    fan_out_write(&name, &jpeg, corpus_base, &["fuzz_encode_diff_c"]);
                    generated += 1;
                }
            }
        }
    }

    // Structural edge-case seeds for every decoder target.
    for (name, bytes) in structural_edge_seeds() {
        fan_out_write(name, &bytes, corpus_base, DECODER_TARGETS);
    }

    // Wide-aspect seeds: 1 MCU tall (8×64) and 1 MCU wide (64×8), exercising
    // dimension-extreme paths that normal square seeds never trigger.
    for (w, h, label) in [(8usize, 64usize, "wide"), (64usize, 8usize, "tall")] {
        for &content in &[Content::Gradient, Content::Checker] {
            let pixels: Vec<u8> = content.pixels_rgb(w, h);
            for &(sub, entropy) in &[
                (SubsampLabel::S420, Entropy::Baseline),
                (SubsampLabel::S444, Entropy::Progressive),
            ] {
                if let Some(jpeg) = {
                    let pf: PixelFormat = PixelFormat::Rgb;
                    let subs: Subsampling = sub.subsampling();
                    match entropy {
                        Entropy::Baseline => compress(&pixels, w, h, pf, 75, subs).ok(),
                        Entropy::Progressive => {
                            compress_progressive(&pixels, w, h, pf, 75, subs).ok()
                        }
                        _ => None,
                    }
                } {
                    let name: String = format!(
                        "aspect_{label}_{c}_{s}_{e}.jpg",
                        c = content.label(),
                        s = sub.label(),
                        e = entropy.label(),
                    );
                    fan_out_write(&name, &jpeg, corpus_base, DECODER_TARGETS);
                    fan_out_write(&name, &jpeg, corpus_base, &["fuzz_transform_options"]);
                }
            }
        }
    }

    // Restart-marker seeds: inject a DRI segment so the decoder exercises
    // restart-marker parsing and resync paths.
    for &content in &[Content::Gradient, Content::SyntheticPhoto] {
        for &(sub, label) in &[(SubsampLabel::S420, "420"), (SubsampLabel::S444, "444")] {
            if let Some(jpeg) = make_restart_jpeg(content, sub, 75) {
                let name: String =
                    format!("restart_{c}_{sub}.jpg", c = content.label(), sub = label,);
                fan_out_write(&name, &jpeg, corpus_base, DECODER_TARGETS);
                fan_out_write(&name, &jpeg, corpus_base, &["fuzz_transform_options"]);
            }
        }
    }

    // Seeds for the new fuzz_transform_options target (structured header + JPEG).
    transform_options_seeds(corpus_base);

    // P2-7: structured seeds for fuzz_encode_diff_c. Codex round-2 P2:
    // fan-out of JPEG bytes alone left this target effectively unseeded
    // — the 4-byte header derives e.g. width=63 height=24 grayscale,
    // requiring 1516 bytes total, but the JPEG seeds are mostly < 600
    // bytes so the target returns before exercising the encoder. We
    // generate explicit [header][pixel-buffer] seeds covering the
    // entropy × subsampling × pixel-format axes the target actually
    // exercises, so libfuzzer starts from inputs that already trigger
    // the differential.
    let encdiff_target: &[&str] = &["fuzz_encode_diff_c"];
    // (width, height, quality, flags_byte, label)
    // flags layout (matches fuzz_encode_diff_c::fuzz_target):
    //   bits 0..1: subsampling idx (0=420, 1=422, 2=444, 3=440)
    //   bits 2..3: entropy        (0=baseline, 1=progressive, 2=arith, 3=arith+prog)
    //   bit  7   : grayscale (1=Grayscale, 0=Rgb)
    // Cover every (sub_idx 0..=3) × (entropy 0..=3) branch the target
    // exercises so libfuzzer hits every code path from iteration 0
    // (codex round-3 P3). The grayscale entries cover the bpp=1 size
    // accounting separately from the RGB entries.
    let encdiff_configs: &[(u8, u8, u8, u8, &str)] = &[
        (32, 32, 75, 0b0000_0000, "rgb_baseline_32x32_q75_s420"),
        (32, 32, 75, 0b0000_0001, "rgb_baseline_32x32_q75_s422"),
        (32, 32, 75, 0b0000_0010, "rgb_baseline_32x32_q75_s444"),
        (32, 32, 75, 0b0000_0011, "rgb_baseline_32x32_q75_s440"),
        (32, 32, 90, 0b0000_0100, "rgb_progressive_32x32_q90_s420"),
        (32, 32, 50, 0b0000_1000, "rgb_arith_32x32_q50_s420"),
        (
            32,
            32,
            75,
            0b0000_1100,
            "rgb_arith_progressive_32x32_q75_s420",
        ),
        (16, 16, 75, 0b1000_0000, "gray_baseline_16x16_q75"),
        (24, 24, 90, 0b1000_0100, "gray_progressive_24x24_q90"),
    ];
    for (w, h, q, flags, label) in encdiff_configs {
        let bpp: usize = if (flags & 0b1000_0000) != 0 { 1 } else { 3 };
        let pixel_buf_len: usize = (*w as usize) * (*h as usize) * bpp;
        // Smooth gradient: gives the encoder real signal without
        // worst-case quantization noise.
        let pixel_buf: Vec<u8> = (0..pixel_buf_len)
            .map(|i| ((i * 251) % 256) as u8)
            .collect();
        let mut bytes: Vec<u8> = Vec::with_capacity(4 + pixel_buf_len);
        bytes.extend_from_slice(&[*w, *h, *q, *flags]);
        bytes.extend_from_slice(&pixel_buf);
        let name: String = format!("encdiff_{label}.bin");
        fan_out_write(&name, &bytes, corpus_base, encdiff_target);
    }

    // P2-7: seeds for fuzz_transform_diff_c. Header is a single op byte
    // (0=HFlip, 1=VFlip, 2=Rot180; the target wraps op_idx % 3) followed
    // by valid JPEG bytes. Without explicit seeds the target spends its
    // 10-min budget on inputs whose byte 1 onward is rarely a valid JPEG
    // — codex P2 finding.
    let diff_c_target: &[&str] = &["fuzz_transform_diff_c"];
    let diff_c_sources: &[(Content, SubsampLabel, u8, Entropy, &str)] = &[
        (
            Content::Gradient,
            SubsampLabel::S420,
            75,
            Entropy::Baseline,
            "grad_420_q75_baseline",
        ),
        (
            Content::Checker,
            SubsampLabel::S444,
            75,
            Entropy::Baseline,
            "chk_444_q75_baseline",
        ),
        (
            Content::SyntheticPhoto,
            SubsampLabel::S420,
            90,
            Entropy::Progressive,
            "photo_420_q90_progressive",
        ),
    ];
    for (content, sub, q, entropy, label) in diff_c_sources {
        let Some(jpeg) = encode_seed(*content, *sub, *q, *entropy) else {
            continue;
        };
        for op_byte in [0u8, 1, 2] {
            let mut bytes: Vec<u8> = Vec::with_capacity(1 + jpeg.len());
            bytes.push(op_byte);
            bytes.extend_from_slice(&jpeg);
            let op_label: &str = match op_byte {
                0 => "hflip",
                1 => "vflip",
                _ => "rot180",
            };
            let name: String = format!("xform_{label}_{op_label}.bin");
            fan_out_write(&name, &bytes, corpus_base, diff_c_target);
        }
    }

    // fuzz_transform-specific inputs: JPEGs wrapped with junk bytes, JPEGs
    // with scrambled marker segments, and JPEGs whose DHT/DQT headers have
    // unusual but legal lengths. All exercise the marker scanner and the
    // coefficient reader without relying on random mutation.
    if let Some(base_jpeg) = encode_seed(
        Content::SyntheticPhoto,
        SubsampLabel::S420,
        75,
        Entropy::Baseline,
    ) {
        let mut with_prefix: Vec<u8> = b"LEADING_JUNK\x00\x00\x00".to_vec();
        with_prefix.extend_from_slice(&base_jpeg);
        fan_out_write(
            "transform_with_prefix.bin",
            &with_prefix,
            corpus_base,
            &["fuzz_transform"],
        );

        let mut with_suffix: Vec<u8> = base_jpeg.clone();
        with_suffix.extend_from_slice(b"\x00TRAILING_JUNK");
        fan_out_write(
            "transform_with_suffix.bin",
            &with_suffix,
            corpus_base,
            &["fuzz_transform"],
        );

        // Stuffed 0xFF bytes between SOI and next marker exercise the marker
        // scanner's fill-byte skip path.
        let mut fill_stuffed: Vec<u8> = vec![0xFF, 0xD8];
        fill_stuffed.extend(std::iter::repeat_n(0xFF, 8));
        fill_stuffed.extend_from_slice(&base_jpeg[2..]);
        fan_out_write(
            "transform_fill_stuffed.bin",
            &fill_stuffed,
            corpus_base,
            &["fuzz_transform"],
        );

        // Extra APP14 Adobe header prepended — covers the Adobe colorspace
        // heuristic path.
        let app14_adobe: [u8; 16] = [
            0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x80, 0x00, 0x80,
            0x00, 0x00,
        ];
        let mut adobe_head: Vec<u8> = vec![0xFF, 0xD8];
        adobe_head.extend_from_slice(&app14_adobe);
        adobe_head.extend_from_slice(&base_jpeg[2..]);
        fan_out_write(
            "transform_app14_adobe.bin",
            &adobe_head,
            corpus_base,
            &["fuzz_transform"],
        );
    }

    // fuzz_progressive_decoder-specific seeds: progressive JPEGs that have
    // been truncated mid-scan so the decoder exercises its partial-input
    // handling paths. Only available when we produced a progressive seed.
    if let Some(prog_jpeg) = encode_seed(
        Content::SyntheticPhoto,
        SubsampLabel::S420,
        75,
        Entropy::Progressive,
    ) {
        // Keep only the first ~75% of bytes — typically cuts off mid-scan.
        let cut: usize = (prog_jpeg.len() * 3) / 4;
        let truncated_mid: Vec<u8> = prog_jpeg[..cut].to_vec();
        fan_out_write(
            "progressive_truncated_mid.bin",
            &truncated_mid,
            corpus_base,
            &["fuzz_progressive_decoder"],
        );

        // First ~15% — should stop before any scan data.
        let cut_early: usize = (prog_jpeg.len() / 7).max(16);
        let truncated_early: Vec<u8> = prog_jpeg[..cut_early].to_vec();
        fan_out_write(
            "progressive_truncated_early.bin",
            &truncated_early,
            corpus_base,
            &["fuzz_progressive_decoder"],
        );

        // Strip the trailing EOI (0xFF 0xD9) to force end-of-stream handling.
        if prog_jpeg.len() > 2 {
            let no_eoi: Vec<u8> = prog_jpeg[..prog_jpeg.len() - 2].to_vec();
            fan_out_write(
                "progressive_no_eoi.bin",
                &no_eoi,
                corpus_base,
                &["fuzz_progressive_decoder"],
            );
        }
    }

    // fuzz_encode_roundtrip gets a small pool of raw-pixel header seeds so
    // libFuzzer starts from structurally meaningful inputs for the encode
    // path. Each is interpreted by the target as a 4-byte header + pixels.
    let raw_seeds: [(&str, Vec<u8>); 3] = [
        ("encode_tiny_gradient.bin", {
            let mut v: Vec<u8> = vec![8, 8, 75, 0];
            v.extend_from_slice(&Content::Gradient.pixels_rgb(8, 8));
            v
        }),
        ("encode_16x16_photo.bin", {
            let mut v: Vec<u8> = vec![16, 16, 90, 1];
            v.extend_from_slice(&Content::SyntheticPhoto.pixels_rgb(16, 16));
            v
        }),
        ("encode_16x16_checker.bin", {
            let mut v: Vec<u8> = vec![16, 16, 50, 2];
            v.extend_from_slice(&Content::Checker.pixels_rgb(16, 16));
            v
        }),
    ];
    for (name, bytes) in &raw_seeds {
        fan_out_write(name, bytes, corpus_base, &["fuzz_encode_roundtrip"]);
    }

    eprintln!(
        "generated={generated} skipped_unsupported={skipped} targets={:?}",
        all_targets
    );
    // 3 content × 7 sub × 3 quality × 6 entropy = 378 attempts; lossless
    // skips most non-444 combos, so the realistic floor is ~200.
    assert!(
        generated >= 200,
        "expected >=200 generated seeds, got {generated}",
    );

    // P4-33 drift guard: if the generator rewrote any existing seed with
    // different bytes, the committed corpus no longer matches the current
    // encoder/generator. The files on disk have already been updated —
    // review the diff (decoded pixels should be unchanged unless the
    // encoder change was meant to alter output) and commit the
    // regenerated seeds in the same PR as the change that caused this.
    let drifted: Vec<String> = std::mem::take(&mut *DRIFTED_SEEDS.lock().expect("drift list"));
    assert!(
        drifted.is_empty(),
        "{} committed fuzz seed(s) drifted from the current generator output; \
         review and commit the regenerated files:\n  {}",
        drifted.len(),
        drifted.join("\n  "),
    );
}
