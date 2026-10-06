//! Downstream-consumer benchmark (P4-214, issue #640).
//!
//! Measures libjpeg-turbo-rs the way an application sees it: a separate crate
//! on Cargo's default `release` profile, depending on the published baseline
//! (0.8.0 from crates.io), the candidate checkout, the `image` adapter,
//! `image`'s built-in JPEG codec and `zune-jpeg`. See ../README.md for what is
//! compared, how to reproduce it and how to read the report.
//!
//! Structure of a run, per case:
//! 1. correctness pass — every backend decodes once; outputs are compared
//!    outside any timed region (exact-equality invariants abort the run,
//!    cross-library differences are reported);
//! 2. allocation pass — one decode per backend under the counting allocator;
//! 3. warmup, then N timed rounds in which every backend runs once, in a
//!    rotating order, so slow drift in machine load hits all rows alike.
//!
//! The concurrent decode section (`concurrent.rs`) follows the same three
//! passes with a batch of T threads × K decodes in place of one decode.

mod alloc_counter;
mod concurrent;
mod corpus;
mod decode;
mod environment;
mod json;
mod ljt_api;
mod measure;
mod report;
mod workloads;

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use alloc_counter::{AllocStats, CountingAllocator};
use corpus::{CorpusFile, OutputLayout, SourcePixels};
use decode::{DecodeBackend, DecodeCase, Preparation, PreparedDecode};
use ljt_api::Chroma;
use measure::{FrameFacts, PixelDiff, TimedSamples, TimingSummary};
use report::{
    ConcurrentReport, ConcurrentRow, CorpusRecord, CorrectnessRecord, DecodeCaseReport, DecodeRow,
    EncodeCaseReport, EncodeRow, Report, ThumbnailReport, ThumbnailRow,
};
use workloads::{EncodeBackend, ThumbnailBackend};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Command-line options. `--smoke` exists to prove the harness end to end in
/// seconds; its numbers are not measurements and the report says so.
struct Options {
    iterations: usize,
    warmup: usize,
    smoke: bool,
    repo: PathBuf,
    out_dir: PathBuf,
    build_info: Option<PathBuf>,
    lockfile: PathBuf,
    only: Option<String>,
    /// Explicit `--djpeg` until `main` resolves it, then the tool in use.
    djpeg: Option<PathBuf>,
    /// Explicit `--cjpeg` until `main` resolves it, then the tool in use.
    cjpeg: Option<PathBuf>,
    /// `--no-c-oracle` turns off both C tools.
    use_c_oracle: bool,
}

const USAGE: &str = "usage: downstream-consumer --repo <candidate checkout> [--out-dir DIR] \
[--iterations N] [--warmup N] [--smoke] [--only SUBSTRING] [--build-info FILE] \
[--lockfile FILE] [--djpeg PATH] [--cjpeg PATH] [--no-c-oracle]";

fn parse_options() -> Options {
    let mut options: Options = Options {
        iterations: 30,
        warmup: 5,
        smoke: false,
        repo: PathBuf::new(),
        out_dir: PathBuf::from("downstream-report"),
        build_info: None,
        lockfile: Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
        only: None,
        djpeg: None,
        cjpeg: None,
        use_c_oracle: true,
    };
    let mut repo: Option<PathBuf> = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = |name: &str| -> String {
            arguments
                .next()
                .unwrap_or_else(|| panic!("{name} needs a value\n{USAGE}"))
        };
        match argument.as_str() {
            "--iterations" => {
                options.iterations = value("--iterations").parse().expect("--iterations N")
            }
            "--warmup" => options.warmup = value("--warmup").parse().expect("--warmup N"),
            "--smoke" => options.smoke = true,
            "--repo" => repo = Some(PathBuf::from(value("--repo"))),
            "--out-dir" => options.out_dir = PathBuf::from(value("--out-dir")),
            "--build-info" => options.build_info = Some(PathBuf::from(value("--build-info"))),
            "--lockfile" => options.lockfile = PathBuf::from(value("--lockfile")),
            "--only" => options.only = Some(value("--only")),
            "--djpeg" => options.djpeg = Some(PathBuf::from(value("--djpeg"))),
            "--cjpeg" => options.cjpeg = Some(PathBuf::from(value("--cjpeg"))),
            "--no-c-oracle" => options.use_c_oracle = false,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}\n{USAGE}"),
        }
    }
    if options.smoke {
        options.iterations = 2;
        options.warmup = 1;
    }
    assert!(options.iterations >= 1, "--iterations must be at least 1");
    options.repo = repo.unwrap_or_else(|| panic!("--repo is required\n{USAGE}"));
    options
}

/// Shortest sample worth timing on its own. Below this a single call is
/// dominated by timer resolution and cold-call effects (first-touch page
/// faults, branch predictors), so such rows are timed in batches.
const MIN_SAMPLE: Duration = Duration::from_millis(1);
const MAX_CALLS_PER_SAMPLE: usize = 1000;

/// Warm every runner, then run `iterations` rounds in which each runner is
/// timed once, starting from a different runner each round. The timed region
/// includes dropping the output: freeing a library-owned buffer is part of
/// the fresh path's cost to an application.
///
/// The warmup calls are timed too, and a runner whose fastest warmup call
/// took under [`MIN_SAMPLE`] is timed `K` calls per sample, with `K` chosen so
/// one sample spans about `MIN_SAMPLE`; every reported figure is per call.
fn time_interleaved(
    runners: &mut [Box<dyn FnMut() + '_>],
    warmup: usize,
    iterations: usize,
) -> Vec<TimedSamples> {
    let mut calls_per_sample: Vec<usize> = Vec::with_capacity(runners.len());
    for runner in runners.iter_mut() {
        let mut fastest: Option<Duration> = None;
        for _ in 0..warmup {
            let start: Instant = Instant::now();
            runner();
            let elapsed: Duration = start.elapsed();
            fastest = Some(fastest.map_or(elapsed, |f| f.min(elapsed)));
        }
        let batch: usize = match fastest {
            Some(call) if call < MIN_SAMPLE => {
                let nanos: u128 = call.as_nanos().max(1);
                (MIN_SAMPLE.as_nanos().div_ceil(nanos) as usize).clamp(1, MAX_CALLS_PER_SAMPLE)
            }
            _ => 1,
        };
        calls_per_sample.push(batch);
    }
    let count: usize = runners.len();
    let mut samples: Vec<TimedSamples> = calls_per_sample
        .iter()
        .map(|batch| TimedSamples {
            per_call: Vec::with_capacity(iterations),
            calls_per_sample: *batch,
        })
        .collect();
    for round in 0..iterations {
        for offset in 0..count {
            let index: usize = (round + offset) % count;
            let batch: usize = samples[index].calls_per_sample;
            let start: Instant = Instant::now();
            for _ in 0..batch {
                (runners[index])();
            }
            let per_call: Duration = start.elapsed() / batch as u32;
            samples[index].per_call.push(per_call);
        }
    }
    samples
}

/// FNV-1a, for the exact-equality invariants without keeping every output
/// (the 8K case is ~100 MB per backend).
fn fingerprint(bytes: &[u8]) -> (usize, u64) {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (bytes.len(), hash)
}

/// Over *source* pixels, so a scaled decode's MP/s compares directly with the
/// full-size decode of the same file.
fn megapixels_per_second(width: usize, height: usize, timing: &TimingSummary) -> f64 {
    (width * height) as f64 / 1e6 / (timing.median_ms / 1e3)
}

fn selected(options: &Options, id: &str) -> bool {
    options
        .only
        .as_deref()
        .is_none_or(|needle| id.contains(needle))
}

struct CorpusStore {
    dir: PathBuf,
    records: Vec<CorpusRecord>,
}

impl CorpusStore {
    /// Write the file next to the report (so any tool, djpeg included, can
    /// re-decode it) and record its checksum and origin.
    fn add(&mut self, file: &CorpusFile) -> PathBuf {
        let path: PathBuf = self.dir.join(format!("{}.jpg", file.id));
        if !self.records.iter().any(|record| record.id == file.id) {
            std::fs::write(&path, &file.jpeg)
                .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
            self.records.push(CorpusRecord {
                id: file.id.clone(),
                bytes: file.jpeg.len(),
                sha256: environment::sha256_hex(&path),
                frame: measure::inspect_frame(&file.jpeg),
                origin: file.origin.clone(),
                licence: file.licence.clone(),
            });
        }
        path
    }
}

/// Decode with the C reference and return its pixels, or why not.
fn c_reference(
    djpeg: &Path,
    jpeg_path: &Path,
    case: &DecodeCase,
    scratch: &Path,
) -> Result<(usize, usize, Vec<u8>), String> {
    let output: PathBuf = scratch.join(format!("{}.c-reference.pnm", case.id.replace('/', "_")));
    let mut command: std::process::Command = std::process::Command::new(djpeg);
    if let Some((numerator, denominator)) = case.scale {
        command
            .arg("-scale")
            .arg(format!("{numerator}/{denominator}"));
    }
    if case.layout == OutputLayout::Gray {
        command.arg("-grayscale");
    }
    command.arg("-outfile").arg(&output).arg(jpeg_path);
    let result: std::process::Output = command
        .output()
        .map_err(|error| format!("djpeg did not run: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "djpeg failed: {}",
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    let bytes: Vec<u8> = std::fs::read(&output).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(&output);
    let (width, height, channels, pixels) = measure::parse_pnm(&bytes)
        .ok_or_else(|| "djpeg output is not 8-bit PPM/PGM".to_string())?;
    if channels != case.layout.bytes_per_pixel() {
        return Err(format!(
            "djpeg wrote {channels} channels, the case expects {}",
            case.layout.bytes_per_pixel()
        ));
    }
    Ok((width, height, pixels))
}

fn run_decode_case(
    case: &DecodeCase,
    options: &Options,
    jpeg_path: &Path,
    djpeg: Option<&Path>,
) -> DecodeCaseReport {
    eprintln!("[decode] {}", case.id);
    let mut prepared: Vec<PreparedDecode> = Vec::new();
    let mut rows: Vec<DecodeRow> = Vec::new();
    for backend in DecodeBackend::ALL {
        match PreparedDecode::prepare(backend, case) {
            Preparation::Ready(ready) => prepared.push(ready),
            Preparation::NotApplicable(reason) => {
                rows.push(DecodeRow::not_applicable(backend, reason))
            }
        }
    }

    // 1. Correctness. The candidate's library-owned decode is the reference
    //    every other row is compared with.
    let reference_index: usize = prepared
        .iter()
        .position(|ready| ready.backend == DecodeBackend::CandidateFresh)
        .expect("the candidate supports every case");
    let (reference_width, reference_height, reference_pixels) = {
        let decoded = prepared[reference_index].decode();
        let pixels: Vec<u8> = prepared[reference_index].pixels(&decoded).to_vec();
        (decoded.width, decoded.height, pixels)
    };
    assert_eq!(
        reference_pixels.len(),
        reference_width * reference_height * case.layout.bytes_per_pixel(),
        "candidate output length disagrees with its dimensions on {}",
        case.id
    );

    let mut correctness: Vec<CorrectnessRecord> = Vec::new();
    let mut fingerprints: Vec<(DecodeBackend, (usize, u64))> = Vec::new();
    for ready in prepared.iter_mut() {
        let decoded = ready.decode();
        let pixels: &[u8] = ready.pixels(&decoded);
        // Every backend was asked for the same format and size; a mismatch
        // is a harness or library bug, not a quality difference.
        assert_eq!(
            (decoded.width, decoded.height),
            (reference_width, reference_height),
            "{} produced different dimensions on {}",
            ready.backend.id(),
            case.id
        );
        assert_eq!(
            pixels.len(),
            reference_pixels.len(),
            "{} output length on {}",
            ready.backend.id(),
            case.id
        );
        fingerprints.push((ready.backend, fingerprint(pixels)));
        if ready.backend != DecodeBackend::CandidateFresh {
            let diff: PixelDiff = measure::pixel_diff(&reference_pixels, pixels);
            correctness.push(CorrectnessRecord::measured(
                ready.backend.id(),
                "candidate-fresh",
                diff,
            ));
        }
    }
    // Same library and settings, different buffer ownership: these must be
    // byte-identical, or one of the two paths is wrong.
    let exact_pairs: [(DecodeBackend, DecodeBackend); 4] = [
        (DecodeBackend::CandidateReuse, DecodeBackend::CandidateFresh),
        (
            DecodeBackend::CandidateImageAdapter,
            DecodeBackend::CandidateFresh,
        ),
        (DecodeBackend::BaselineReuse, DecodeBackend::BaselineFresh),
        (DecodeBackend::ZuneReuse, DecodeBackend::ZuneFresh),
    ];
    let fingerprint_of = |wanted: DecodeBackend| -> Option<(usize, u64)> {
        fingerprints
            .iter()
            .find(|(backend, _)| *backend == wanted)
            .map(|(_, print)| *print)
    };
    for (left, right) in exact_pairs {
        if let (Some(a), Some(b)) = (fingerprint_of(left), fingerprint_of(right)) {
            assert_eq!(
                a,
                b,
                "{} and {} must be byte-identical on {}",
                left.id(),
                right.id(),
                case.id
            );
        }
    }
    // C is the contract: the project's goal is output byte-identical to
    // `djpeg` with default settings, and every case here — baseline,
    // progressive, grayscale, 1/4 DCT scaling (`djpeg -scale 1/4`) — is inside
    // it. So with a C decoder present, a difference fails the run. The
    // published baseline is held to it too: on 2026-10-07 it matched stock
    // 3.2.0 and Homebrew 3.1.4.1 on all eight cases.
    if let Some(djpeg) = djpeg {
        let (width, height, c_pixels) = c_reference(djpeg, jpeg_path, case, &options.out_dir)
            .unwrap_or_else(|reason| panic!("C djpeg could not decode {}: {reason}", case.id));
        assert_eq!(
            (width, height),
            (reference_width, reference_height),
            "C djpeg and the candidate disagree on dimensions for {}",
            case.id
        );
        let diff: PixelDiff = measure::pixel_diff(&c_pixels, &reference_pixels);
        assert_eq!(
            diff.max_abs, 0,
            "candidate differs from C djpeg on {} ({diff:?}); byte-identical output is the contract",
            case.id
        );
        correctness.push(CorrectnessRecord::note(
            "candidate-fresh",
            "C djpeg",
            "identical (asserted)".to_string(),
        ));
        assert_eq!(
            fingerprint_of(DecodeBackend::BaselineFresh),
            Some(fingerprint(&c_pixels)),
            "baseline 0.8.0 differs from C djpeg on {}",
            case.id
        );
        correctness.push(CorrectnessRecord::note(
            "baseline-fresh",
            "C djpeg",
            "identical (asserted)".to_string(),
        ));
    }
    drop(reference_pixels);

    // 2. Allocation pass.
    let mut allocations: Vec<AllocStats> = Vec::with_capacity(prepared.len());
    for ready in prepared.iter_mut() {
        let (decoded, stats) = alloc_counter::measure(|| ready.decode());
        drop(decoded);
        allocations.push(stats);
    }

    // 3. Timing.
    let backends: Vec<DecodeBackend> = prepared.iter().map(|ready| ready.backend).collect();
    let samples: Vec<TimedSamples> = {
        let mut runners: Vec<Box<dyn FnMut() + '_>> = prepared
            .iter_mut()
            .map(|ready| {
                Box::new(move || {
                    let decoded = ready.decode();
                    black_box(&decoded);
                }) as Box<dyn FnMut() + '_>
            })
            .collect();
        time_interleaved(&mut runners, options.warmup, options.iterations)
    };
    for ((backend, samples), stats) in backends.iter().zip(&samples).zip(&allocations) {
        let timing: TimingSummary = measure::summarize(samples);
        rows.push(DecodeRow {
            backend: backend.id().to_string(),
            api: backend.api().to_string(),
            buffer: backend.buffer().to_string(),
            path: backend.path().to_string(),
            output: format!(
                "{} {reference_width}x{reference_height}",
                case.layout.label()
            ),
            timing: Some(timing),
            megapixels_per_second: Some(megapixels_per_second(
                case.source_width,
                case.source_height,
                &timing,
            )),
            alloc: Some(*stats),
            not_applicable: None,
        });
    }
    // Fixed backend order in the report regardless of preparation order.
    rows.sort_by_key(|row| {
        DecodeBackend::ALL
            .iter()
            .position(|backend| backend.id() == row.backend)
    });

    DecodeCaseReport {
        id: case.id.clone(),
        corpus_id: case.corpus_id.clone(),
        description: case.description.clone(),
        source_width: case.source_width,
        source_height: case.source_height,
        layout: case.layout.label().to_string(),
        scale: case.scale.map(|(n, d)| format!("{n}/{d}")),
        rows,
        correctness,
    }
}

/// Encode `ppm` with C `cjpeg -quality 85` (4:2:0, cjpeg's default for RGB
/// input), or `-sample 1x1` for 4:4:4.
fn c_encode(cjpeg: &Path, ppm: &Path, chroma: Chroma, scratch: &Path) -> Result<Vec<u8>, String> {
    let output: PathBuf = scratch.join("c-encode.jpg");
    let mut command: std::process::Command = std::process::Command::new(cjpeg);
    command
        .arg("-quality")
        .arg(workloads::ENCODE_QUALITY.to_string());
    if chroma == Chroma::S444 {
        command.arg("-sample").arg("1x1");
    }
    command.arg("-outfile").arg(&output).arg(ppm);
    let result: std::process::Output = command
        .output()
        .map_err(|error| format!("cjpeg did not run: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "cjpeg failed: {}",
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    let bytes: Vec<u8> = std::fs::read(&output).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(&output);
    Ok(bytes)
}

/// Where two JPEG streams first differ, and whether they still differ once
/// every APP0 (JFIF) segment is removed — so a header-only difference is
/// reported as such rather than hidden or treated as a codec difference.
fn describe_byte_difference(ours: &[u8], theirs: &[u8]) -> String {
    let first: usize = ours
        .iter()
        .zip(theirs)
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(theirs.len()));
    let without_app0 = |jpeg: &[u8]| -> Vec<u8> {
        let mut out: Vec<u8> = jpeg[..2.min(jpeg.len())].to_vec();
        let mut position: usize = 2;
        while position + 4 <= jpeg.len() && jpeg[position] == 0xFF && jpeg[position + 1] != 0xDA {
            let length: usize =
                u16::from_be_bytes([jpeg[position + 2], jpeg[position + 3]]) as usize;
            let end: usize = (position + 2 + length).min(jpeg.len());
            if jpeg[position + 1] != 0xE0 {
                out.extend_from_slice(&jpeg[position..end]);
            }
            position = end;
        }
        out.extend_from_slice(&jpeg[position.min(jpeg.len())..]);
        out
    };
    let same_without_app0: bool = without_app0(ours) == without_app0(theirs);
    format!(
        "sizes {} vs {} bytes, first difference at offset {first}; {}",
        ours.len(),
        theirs.len(),
        if same_without_app0 {
            "identical once APP0 is removed (header-only difference)"
        } else {
            "still different once APP0 is removed"
        }
    )
}

fn run_encode_case(source: &SourcePixels, options: &Options) -> EncodeCaseReport {
    eprintln!("[encode] {}", source.id);
    let mut rows: Vec<EncodeRow> = Vec::new();
    let mut fingerprints: Vec<(EncodeBackend, (usize, u64))> = Vec::new();
    let mut outputs: Vec<(EncodeBackend, Vec<u8>)> = Vec::new();
    for backend in EncodeBackend::ALL {
        let jpeg: Vec<u8> = backend.encode(&source.rgb, source.width, source.height);
        // One reference decoder for every row's PSNR — the pinned published
        // baseline — so the PSNR column differs only by encoder.
        let (width, height, decoded) =
            ljt_api::baseline::decode_fresh(&jpeg, OutputLayout::Rgb, None);
        assert_eq!(
            (width, height),
            (source.width, source.height),
            "{} changed the dimensions",
            backend.id()
        );
        let facts: Option<FrameFacts> = measure::inspect_frame(&jpeg);
        rows.push(EncodeRow {
            backend: backend.id().to_string(),
            api: backend.api().to_string(),
            output_bytes: jpeg.len(),
            subsampling: facts
                .map(|f| f.subsampling)
                .unwrap_or_else(|| "unreadable".to_string()),
            psnr_db: measure::psnr(&source.rgb, &decoded),
            timing: None,
            megapixels_per_second: None,
            alloc: AllocStats::default(),
        });
        fingerprints.push((backend, fingerprint(&jpeg)));
        outputs.push((backend, jpeg));
    }
    // The adapter is a thin wrapper over the candidate's `compress` with the
    // same quality and subsampling: identical bytes, or the wrapper changed
    // something it should not have.
    let candidate_bytes = fingerprints
        .iter()
        .find(|(b, _)| *b == EncodeBackend::Candidate);
    let adapter_bytes = fingerprints
        .iter()
        .find(|(b, _)| *b == EncodeBackend::CandidateImageAdapter);
    assert_eq!(
        candidate_bytes.map(|(_, f)| f),
        adapter_bytes.map(|(_, f)| f),
        "the image adapter's encoder must produce the candidate's exact bytes on {}",
        source.id
    );

    // C cross-check. `compress` at q85 is documented to match `cjpeg
    // -quality 85` byte for byte (4:2:0 by default, `-sample 1x1` for 4:4:4),
    // so with a C encoder present a candidate difference fails the run.
    let mut c_comparison: Vec<CorrectnessRecord> = Vec::new();
    match &options.cjpeg {
        None => c_comparison.push(CorrectnessRecord::note(
            "all rows",
            "C cjpeg",
            if options.use_c_oracle {
                "C encode comparison: skipped (no cjpeg)".to_string()
            } else {
                "C encode comparison: disabled (--no-c-oracle)".to_string()
            },
        )),
        Some(cjpeg) => {
            let ppm: PathBuf = options.out_dir.join(format!("{}.ppm", source.id));
            let mut ppm_bytes: Vec<u8> =
                format!("P6\n{} {}\n255\n", source.width, source.height).into_bytes();
            ppm_bytes.extend_from_slice(&source.rgb);
            std::fs::write(&ppm, &ppm_bytes)
                .unwrap_or_else(|error| panic!("write {}: {error}", ppm.display()));
            drop(ppm_bytes);
            for chroma in [Chroma::S420, Chroma::S444] {
                let c_bytes: Vec<u8> = c_encode(cjpeg, &ppm, chroma, &options.out_dir)
                    .unwrap_or_else(|reason| {
                        panic!("C cjpeg could not encode {}: {reason}", source.id)
                    });
                let label: &str = match chroma {
                    Chroma::S420 => "C cjpeg -quality 85",
                    Chroma::S444 => "C cjpeg -quality 85 -sample 1x1",
                };
                let (baseline, candidate) = match chroma {
                    Chroma::S420 => (EncodeBackend::Baseline, EncodeBackend::Candidate),
                    Chroma::S444 => (EncodeBackend::Baseline444, EncodeBackend::Candidate444),
                };
                for backend in [baseline, candidate] {
                    let ours: &[u8] = &outputs
                        .iter()
                        .find(|(b, _)| *b == backend)
                        .expect("every backend encoded")
                        .1;
                    if ours != c_bytes.as_slice() {
                        panic!(
                            "{} differs from {label} on {}: {}",
                            backend.id(),
                            source.id,
                            describe_byte_difference(ours, &c_bytes)
                        );
                    }
                    c_comparison.push(CorrectnessRecord::note(
                        backend.id(),
                        label,
                        "byte-identical (asserted)".to_string(),
                    ));
                }
            }
            let _ = std::fs::remove_file(&ppm);
            c_comparison.push(CorrectnessRecord::note(
                "candidate-image-adapter",
                "candidate",
                "byte-identical (asserted above), hence identical to C".to_string(),
            ));
            c_comparison.push(CorrectnessRecord::note(
                "image-builtin",
                "—",
                "not compared: image's own encoder makes no libjpeg-compatibility claim"
                    .to_string(),
            ));
        }
    }
    drop(outputs);

    for (row, backend) in rows.iter_mut().zip(EncodeBackend::ALL) {
        let (jpeg, stats) =
            alloc_counter::measure(|| backend.encode(&source.rgb, source.width, source.height));
        drop(jpeg);
        row.alloc = stats;
    }
    let samples: Vec<TimedSamples> = {
        let mut runners: Vec<Box<dyn FnMut() + '_>> = EncodeBackend::ALL
            .iter()
            .map(|backend| {
                let backend: EncodeBackend = *backend;
                Box::new(move || {
                    let jpeg: Vec<u8> = backend.encode(&source.rgb, source.width, source.height);
                    black_box(&jpeg);
                }) as Box<dyn FnMut() + '_>
            })
            .collect();
        time_interleaved(&mut runners, options.warmup, options.iterations)
    };
    for (row, samples) in rows.iter_mut().zip(&samples) {
        let timing: TimingSummary = measure::summarize(samples);
        row.megapixels_per_second =
            Some(megapixels_per_second(source.width, source.height, &timing));
        row.timing = Some(timing);
    }
    EncodeCaseReport {
        id: source.id.clone(),
        width: source.width,
        height: source.height,
        c_comparison,
        rows,
    }
}

/// One thumbnail decoded back for the correctness columns.
struct ThumbnailOutput {
    backend: ThumbnailBackend,
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    jpeg_bytes: usize,
}

fn run_thumbnail(
    file: &CorpusFile,
    width: usize,
    height: usize,
    options: &Options,
) -> ThumbnailReport {
    eprintln!("[thumbnail] {}", file.id);
    let applicable: Vec<ThumbnailBackend> = ThumbnailBackend::ALL
        .into_iter()
        .filter(|backend| backend.not_applicable().is_none())
        .collect();
    // EXIF 6 turns the landscape source portrait before the resize.
    let (expected_width, expected_height) = workloads::thumbnail_size(height as u32, width as u32);

    // Correctness: decode every thumbnail with the pinned baseline and
    // compare with the candidate's.
    let outputs: Vec<ThumbnailOutput> = applicable
        .iter()
        .map(|backend| {
            let jpeg: Vec<u8> = workloads::thumbnail(*backend, &file.jpeg, width, height);
            let (out_width, out_height, pixels) =
                ljt_api::baseline::decode_fresh(&jpeg, OutputLayout::Rgb, None);
            ThumbnailOutput {
                backend: *backend,
                width: out_width,
                height: out_height,
                pixels,
                jpeg_bytes: jpeg.len(),
            }
        })
        .collect();
    let candidate_output: &ThumbnailOutput = outputs
        .iter()
        .find(|output| output.backend == ThumbnailBackend::Candidate)
        .expect("the candidate row always runs");

    let allocations: Vec<AllocStats> = applicable
        .iter()
        .map(|backend| {
            let (jpeg, stats) = alloc_counter::measure(|| {
                workloads::thumbnail(*backend, &file.jpeg, width, height)
            });
            drop(jpeg);
            stats
        })
        .collect();
    let samples: Vec<TimedSamples> = {
        let mut runners: Vec<Box<dyn FnMut() + '_>> = applicable
            .iter()
            .map(|backend| {
                let backend: ThumbnailBackend = *backend;
                let jpeg: &[u8] = &file.jpeg;
                Box::new(move || {
                    let thumbnail: Vec<u8> = workloads::thumbnail(backend, jpeg, width, height);
                    black_box(&thumbnail);
                }) as Box<dyn FnMut() + '_>
            })
            .collect();
        time_interleaved(&mut runners, options.warmup, options.iterations)
    };

    let mut rows: Vec<ThumbnailRow> = Vec::new();
    for ((output, stats), samples) in outputs.iter().zip(&allocations).zip(&samples) {
        let timing: TimingSummary = measure::summarize(samples);
        let same_size: bool =
            (output.width, output.height) == (candidate_output.width, candidate_output.height);
        rows.push(ThumbnailRow {
            backend: output.backend.id().to_string(),
            api: output.backend.api().to_string(),
            output: format!("{}x{}", output.width, output.height),
            output_bytes: output.jpeg_bytes,
            orientation_applied: (output.width as u32, output.height as u32)
                == (expected_width, expected_height),
            diff_vs_candidate: same_size
                .then(|| measure::pixel_diff(&candidate_output.pixels, &output.pixels)),
            timing: Some(timing),
            megapixels_per_second: Some(megapixels_per_second(width, height, &timing)),
            alloc: Some(*stats),
            not_applicable: None,
        });
    }
    for backend in ThumbnailBackend::ALL {
        if let Some(reason) = backend.not_applicable() {
            rows.push(ThumbnailRow::not_applicable(backend, reason));
        }
    }
    ThumbnailReport {
        corpus_id: file.id.clone(),
        source_width: width,
        source_height: height,
        expected_output: format!("{expected_width}x{expected_height}"),
        rows,
    }
}

/// T threads × K decodes of the 12 MP photo per backend; see `concurrent.rs`.
fn run_concurrent_decode(case: &DecodeCase, options: &Options) -> ConcurrentReport {
    let threads: usize = concurrent::worker_count(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let decodes: usize = concurrent::decodes_per_thread(options.smoke);
    eprintln!("[concurrent] {} ({threads} threads x {decodes})", case.id);
    // One PreparedDecode per thread, so each reuse thread owns its buffer.
    // All rows' pools coexist because the timing rounds interleave rows.
    let mut pools: Vec<Vec<PreparedDecode>> = Vec::new();
    let mut rows: Vec<ConcurrentRow> = Vec::new();
    for backend in DecodeBackend::ALL {
        let mut pool: Vec<PreparedDecode> = Vec::with_capacity(threads);
        let mut not_applicable: Option<&'static str> = None;
        for _ in 0..threads {
            match PreparedDecode::prepare(backend, case) {
                Preparation::Ready(ready) => pool.push(ready),
                Preparation::NotApplicable(reason) => {
                    not_applicable = Some(reason);
                    break;
                }
            }
        }
        match not_applicable {
            Some(reason) => rows.push(ConcurrentRow {
                backend: backend.id().to_string(),
                path: backend.path().to_string(),
                timing: None,
                megapixels_per_second: None,
                alloc: None,
                caller_buffer_bytes: None,
                outputs_compared: None,
                not_applicable: Some(reason.to_string()),
            }),
            None => pools.push(pool),
        }
    }

    // 1. Correctness: the single-threaded output of the same backend is the
    //    reference for every output of one concurrent batch.
    let mut outputs_compared: Vec<usize> = Vec::with_capacity(pools.len());
    for pool in pools.iter_mut() {
        let backend: DecodeBackend = pool[0].backend;
        let (width, height, reference) = {
            let decoded = pool[0].decode();
            let pixels: Vec<u8> = pool[0].pixels(&decoded).to_vec();
            (decoded.width, decoded.height, pixels)
        };
        let compared: usize = concurrent::check_batch(pool, decodes, (width, height, &reference))
            .unwrap_or_else(|difference| {
                panic!(
                    "{} differs from its single-threaded output under {threads} threads on {}: {difference}",
                    backend.id(),
                    case.id
                )
            });
        outputs_compared.push(compared);
    }

    // 2. Allocation pass: one batch, all threads in one window. run_batch
    //    joins every thread before returning, so the counts are exact.
    let allocations: Vec<AllocStats> = pools
        .iter_mut()
        .map(|pool| alloc_counter::measure(|| concurrent::run_batch(pool, decodes)).1)
        .collect();

    // 3. Timing: one runner per backend runs a whole batch.
    let samples: Vec<TimedSamples> = {
        let mut runners: Vec<Box<dyn FnMut() + '_>> = pools
            .iter_mut()
            .map(|pool| {
                Box::new(move || concurrent::run_batch(pool, decodes)) as Box<dyn FnMut() + '_>
            })
            .collect();
        time_interleaved(&mut runners, options.warmup, options.iterations)
    };
    for (((pool, samples), stats), compared) in pools
        .iter()
        .zip(&samples)
        .zip(&allocations)
        .zip(&outputs_compared)
    {
        let backend: DecodeBackend = pool[0].backend;
        let timing: TimingSummary = measure::summarize(samples);
        rows.push(ConcurrentRow {
            backend: backend.id().to_string(),
            path: backend.path().to_string(),
            timing: Some(timing),
            megapixels_per_second: Some(concurrent::aggregate_megapixels_per_second(
                case.source_width,
                case.source_height,
                threads,
                decodes,
                &timing,
            )),
            alloc: Some(*stats),
            caller_buffer_bytes: Some(
                pool.iter()
                    .map(|ready| ready.caller_buffer_bytes() as u64)
                    .sum(),
            ),
            outputs_compared: Some(*compared),
            not_applicable: None,
        });
    }
    rows.sort_by_key(|row| {
        DecodeBackend::ALL
            .iter()
            .position(|backend| backend.id() == row.backend)
    });
    ConcurrentReport {
        id: concurrent::SECTION_ID.to_string(),
        corpus_id: case.corpus_id.clone(),
        source_width: case.source_width,
        source_height: case.source_height,
        threads,
        decodes_per_thread: decodes,
        rows,
    }
}

/// `(case id, corpus id, description, output layout, DCT scale)`.
type DecodeSpec = (
    &'static str,
    &'static str,
    &'static str,
    OutputLayout,
    Option<(u32, u32)>,
);

const DECODE_SPECS: [DecodeSpec; 8] = [
    (
        "small-64x64-420",
        "synthetic-64x64-420",
        "small image, 4:2:0",
        OutputLayout::Rgb,
        None,
    ),
    (
        "phone-4032x3024-420",
        "synthetic-4032x3024-420",
        "12 MP phone-size photo, 4:2:0",
        OutputLayout::Rgb,
        None,
    ),
    (
        "phone-4032x3024-420-scale-1/4",
        "synthetic-4032x3024-420",
        "the same photo, 1/4-scaled decode",
        OutputLayout::Rgb,
        Some((1, 4)),
    ),
    (
        "gray-1920x1080",
        "synthetic-1920x1080-gray",
        "grayscale 1080p",
        OutputLayout::Gray,
        None,
    ),
    (
        "progressive-1920x1080-420",
        "synthetic-1920x1080-420-progressive",
        "progressive 1080p, 4:2:0",
        OutputLayout::Rgb,
        None,
    ),
    (
        "large-7680x4320-420",
        "synthetic-7680x4320-420",
        "33 MP (8K UHD) image, 4:2:0",
        OutputLayout::Rgb,
        None,
    ),
    (
        "testorig",
        "testorig",
        "upstream testorig.jpg",
        OutputLayout::Rgb,
        None,
    ),
    (
        "testimgint",
        "testimgint",
        "upstream testimgint.jpg",
        OutputLayout::Rgb,
        None,
    ),
];

fn corpus_file(corpus_id: &str, repo: &Path) -> CorpusFile {
    match corpus_id {
        "synthetic-64x64-420" => corpus::synthetic_rgb_jpeg(corpus_id, 64, 64, false, None),
        "synthetic-4032x3024-420" => corpus::synthetic_rgb_jpeg(corpus_id, 4032, 3024, false, None),
        "synthetic-1920x1080-gray" => corpus::synthetic_gray_jpeg(corpus_id, 1920, 1080),
        "synthetic-1920x1080-420-progressive" => {
            corpus::synthetic_rgb_jpeg(corpus_id, 1920, 1080, true, None)
        }
        "synthetic-7680x4320-420" => corpus::synthetic_rgb_jpeg(corpus_id, 7680, 4320, false, None),
        "testorig" => corpus::upstream_test_image(corpus_id, repo, "testorig.jpg"),
        "testimgint" => corpus::upstream_test_image(corpus_id, repo, "testimgint.jpg"),
        other => unreachable!("unknown corpus id {other}"),
    }
}

fn main() {
    let mut options: Options = parse_options();
    // Sample the machine before this process does anything heavy.
    let load_sample: String = environment::load_sample();
    let started_at: String = environment::shell("date -u +%Y-%m-%dT%H:%M:%SZ");
    std::fs::create_dir_all(options.out_dir.join("corpus")).expect("create the output directory");
    let mut store: CorpusStore = CorpusStore {
        dir: options.out_dir.join("corpus"),
        records: Vec::new(),
    };
    if options.use_c_oracle {
        options.djpeg = environment::find_c_tool("djpeg", options.djpeg.as_deref(), "DJPEG");
        options.cjpeg = environment::find_c_tool("cjpeg", options.cjpeg.as_deref(), "CJPEG");
    } else {
        options.djpeg = None;
        options.cjpeg = None;
    }
    let describe_tool = |tool: &Option<PathBuf>, name: &str, variable: &str| -> String {
        match tool {
            Some(path) => format!("{} — {}", path.display(), environment::c_tool_version(path)),
            None if !options.use_c_oracle => "disabled (--no-c-oracle)".to_string(),
            None => format!("not found (set {variable} or pass --{name} to add the C comparison)"),
        }
    };
    let c_oracle: String = describe_tool(&options.djpeg, "djpeg", "DJPEG");
    let c_encoder: String = describe_tool(&options.cjpeg, "cjpeg", "CJPEG");
    let c_oracle_link_map: Option<String> =
        options.djpeg.as_deref().map(environment::c_tool_link_map);
    let c_encoder_link_map: Option<String> =
        options.cjpeg.as_deref().map(environment::c_tool_link_map);
    let djpeg: Option<PathBuf> = options.djpeg.clone();

    // Decode cases. A corpus file is generated only when a selected case uses
    // it, so `--only` also keeps a run small.
    let mut decode_reports: Vec<DecodeCaseReport> = Vec::new();
    let mut cached: Option<CorpusFile> = None;
    for (case_id, corpus_id, description, layout, scale) in DECODE_SPECS {
        if !selected(&options, case_id) {
            continue;
        }
        // Specs sharing a corpus file are adjacent, so a one-entry cache
        // avoids regenerating the 12 MP photo without holding the 8K one
        // longer than its case.
        if cached.as_ref().is_none_or(|file| file.id != corpus_id) {
            cached = Some(corpus_file(corpus_id, &options.repo));
        }
        let file: &CorpusFile = cached.as_ref().expect("just filled");
        let jpeg_path: PathBuf = store.add(file);
        let facts: FrameFacts =
            measure::inspect_frame(&file.jpeg).expect("corpus files have a frame header");
        let case: DecodeCase = DecodeCase {
            id: case_id.to_string(),
            corpus_id: corpus_id.to_string(),
            description: description.to_string(),
            jpeg: file.jpeg.clone(),
            layout,
            scale,
            source_width: facts.width as usize,
            source_height: facts.height as usize,
        };
        decode_reports.push(run_decode_case(
            &case,
            &options,
            &jpeg_path,
            djpeg.as_deref(),
        ));
    }
    drop(cached);

    // Encode cases: the same synthetic content as raw RGB.
    let mut encode_reports: Vec<EncodeCaseReport> = Vec::new();
    for (id, width, height) in [
        ("encode-64x64", 64usize, 64usize),
        ("encode-1920x1080", 1920, 1080),
        ("encode-4032x3024", 4032, 3024),
    ] {
        if !selected(&options, id) {
            continue;
        }
        let source: SourcePixels = SourcePixels {
            id: id.to_string(),
            width,
            height,
            rgb: corpus::synthetic_rgb(width, height),
        };
        encode_reports.push(run_encode_case(&source, &options));
    }

    // Thumbnail workload: a portrait phone shot stored landscape + EXIF 6.
    let mut thumbnail_report: Option<ThumbnailReport> = None;
    if selected(&options, "thumbnail-4032x3024-exif6") {
        let file: CorpusFile =
            corpus::synthetic_rgb_jpeg("synthetic-4032x3024-420-exif6", 4032, 3024, false, Some(6));
        store.add(&file);
        thumbnail_report = Some(run_thumbnail(&file, 4032, 3024, &options));
    }

    let mut concurrent_report: Option<ConcurrentReport> = None;
    if selected(&options, concurrent::SECTION_ID) {
        // Regenerated rather than kept from the decode loop, whose one-entry
        // cache holds at most one file; `store.add` records it once.
        let file: CorpusFile = corpus_file(concurrent::CORPUS_ID, &options.repo);
        store.add(&file);
        let facts: FrameFacts =
            measure::inspect_frame(&file.jpeg).expect("corpus files have a frame header");
        let case: DecodeCase = DecodeCase {
            id: concurrent::SECTION_ID.to_string(),
            corpus_id: file.id.clone(),
            description: "12 MP phone-size photo, 4:2:0, decoded concurrently".to_string(),
            jpeg: file.jpeg,
            layout: OutputLayout::Rgb,
            scale: None,
            source_width: facts.width as usize,
            source_height: facts.height as usize,
        };
        concurrent_report = Some(run_concurrent_decode(&case, &options));
    }

    let report: Report = Report {
        started_at,
        smoke: options.smoke,
        iterations: options.iterations,
        warmup: options.warmup,
        only: options.only.clone(),
        repo: options.repo.display().to_string(),
        build_info: environment::read_build_info(options.build_info.as_deref()),
        rustc: environment::shell("rustc -Vv"),
        cpu: environment::cpu_model(),
        logical_cpus: environment::logical_cpus(),
        os: environment::operating_system(),
        runtime_cpu_features: environment::runtime_cpu_features(),
        candidate_simd_and_std: ljt_candidate::simd_and_std_features_enabled(),
        baseline_simd_and_std: ljt_baseline::simd_and_std_features_enabled(),
        locked_packages: environment::read_lock(&options.lockfile),
        lockfile: options.lockfile.display().to_string(),
        load_sample,
        c_oracle,
        c_oracle_link_map,
        c_encoder,
        c_encoder_link_map,
        corpus: store.records,
        decode: decode_reports,
        encode: encode_reports,
        thumbnail: thumbnail_report,
        concurrent: concurrent_report,
    };
    let markdown_path: PathBuf = options.out_dir.join("report.md");
    let json_path: PathBuf = options.out_dir.join("report.json");
    std::fs::write(&markdown_path, report.markdown()).expect("write report.md");
    std::fs::write(&json_path, report.json().render()).expect("write report.json");
    println!("report: {}", markdown_path.display());
    println!("json:   {}", json_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_jfif_header_only_difference_is_named_as_such() {
        // SOI, APP0 (len 4, payload differs), DQT stub, SOS stub + data, EOI.
        let ours: [u8; 14] = [
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x01, 0x01, 0xFF, 0xDA, 0x00, 0x02, 0xFF, 0xD9,
        ];
        let mut theirs: [u8; 14] = ours;
        theirs[7] = 0x02;
        assert!(describe_byte_difference(&ours, &theirs).contains("header-only"));
        theirs = ours;
        theirs[11] = 0x03;
        assert!(describe_byte_difference(&ours, &theirs).contains("still different"));
    }
}
