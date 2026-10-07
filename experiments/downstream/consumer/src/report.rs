//! Report model and its Markdown / JSON renderings.

use std::fmt::Write;

use crate::alloc_counter::AllocStats;
use crate::decode::DecodeBackend;
use crate::environment::LockedPackage;
use crate::json::Json;
use crate::measure::{FrameFacts, PixelDiff, TimingSummary};
use crate::workloads::{EncodeBackend, ThumbnailBackend};

pub struct CorpusRecord {
    pub id: String,
    pub bytes: usize,
    pub sha256: String,
    pub frame: Option<FrameFacts>,
    pub origin: String,
    pub licence: String,
}

/// One comparison made outside the timed region.
pub struct CorrectnessRecord {
    pub subject: String,
    pub compared_to: String,
    pub diff: Option<PixelDiff>,
    pub note: Option<String>,
}

impl CorrectnessRecord {
    pub fn measured(subject: &str, compared_to: &str, diff: PixelDiff) -> Self {
        CorrectnessRecord {
            subject: subject.to_string(),
            compared_to: compared_to.to_string(),
            diff: Some(diff),
            note: None,
        }
    }

    pub fn note(subject: &str, compared_to: &str, note: String) -> Self {
        CorrectnessRecord {
            subject: subject.to_string(),
            compared_to: compared_to.to_string(),
            diff: None,
            note: Some(note),
        }
    }
}

pub struct DecodeRow {
    pub backend: String,
    pub api: String,
    pub buffer: String,
    pub path: String,
    pub output: String,
    pub timing: Option<TimingSummary>,
    pub megapixels_per_second: Option<f64>,
    pub alloc: Option<AllocStats>,
    pub not_applicable: Option<String>,
}

impl DecodeRow {
    pub fn not_applicable(backend: DecodeBackend, reason: &str) -> Self {
        DecodeRow {
            backend: backend.id().to_string(),
            api: backend.api().to_string(),
            buffer: backend.buffer().to_string(),
            path: backend.path().to_string(),
            output: "—".to_string(),
            timing: None,
            megapixels_per_second: None,
            alloc: None,
            not_applicable: Some(reason.to_string()),
        }
    }
}

pub struct DecodeCaseReport {
    pub id: String,
    pub corpus_id: String,
    pub description: String,
    pub source_width: usize,
    pub source_height: usize,
    pub layout: String,
    pub scale: Option<String>,
    pub rows: Vec<DecodeRow>,
    pub correctness: Vec<CorrectnessRecord>,
}

pub struct EncodeRow {
    pub backend: String,
    pub api: String,
    pub output_bytes: usize,
    pub subsampling: String,
    pub psnr_db: f64,
    pub timing: Option<TimingSummary>,
    pub megapixels_per_second: Option<f64>,
    pub alloc: AllocStats,
}

pub struct EncodeCaseReport {
    pub id: String,
    pub width: usize,
    pub height: usize,
    /// Byte-for-byte comparison with C `cjpeg`, or why it was skipped.
    pub c_comparison: Vec<CorrectnessRecord>,
    pub rows: Vec<EncodeRow>,
}

pub struct ThumbnailRow {
    pub backend: String,
    pub api: String,
    pub output: String,
    pub output_bytes: usize,
    pub orientation_applied: bool,
    pub diff_vs_candidate: Option<PixelDiff>,
    pub timing: Option<TimingSummary>,
    pub megapixels_per_second: Option<f64>,
    pub alloc: Option<AllocStats>,
    pub not_applicable: Option<String>,
}

impl ThumbnailRow {
    pub fn not_applicable(backend: ThumbnailBackend, reason: &str) -> Self {
        ThumbnailRow {
            backend: backend.id().to_string(),
            api: backend.api().to_string(),
            output: "—".to_string(),
            output_bytes: 0,
            orientation_applied: false,
            diff_vs_candidate: None,
            timing: None,
            megapixels_per_second: None,
            alloc: None,
            not_applicable: Some(reason.to_string()),
        }
    }
}

pub struct ThumbnailReport {
    pub corpus_id: String,
    pub source_width: usize,
    pub source_height: usize,
    pub expected_output: String,
    pub rows: Vec<ThumbnailRow>,
}

/// One backend's row in the concurrent section. Timing is per batch (T
/// threads × K decodes each); MP/s is the whole batch's.
pub struct ConcurrentRow {
    pub backend: String,
    pub path: String,
    pub timing: Option<TimingSummary>,
    pub megapixels_per_second: Option<f64>,
    /// One batch under the counting allocator, all threads together.
    pub alloc: Option<AllocStats>,
    /// Caller-owned output buffers, one per thread, allocated before the
    /// allocation window and so not in `alloc`. An application holds them
    /// too: compare `peak_live + caller_buffer_bytes` across rows.
    pub caller_buffer_bytes: Option<u64>,
    /// Outputs compared with the same backend's single-threaded output, all
    /// byte-identical (a difference aborts the run).
    pub outputs_compared: Option<usize>,
    pub not_applicable: Option<String>,
}

pub struct ConcurrentReport {
    pub id: String,
    pub corpus_id: String,
    pub source_width: usize,
    pub source_height: usize,
    pub threads: usize,
    pub decodes_per_thread: usize,
    pub rows: Vec<ConcurrentRow>,
    /// The single-threaded references against each other and C djpeg, on
    /// the decode section's contract; or why C was not compared.
    pub correctness: Vec<CorrectnessRecord>,
}

pub struct Report {
    pub started_at: String,
    pub smoke: bool,
    pub iterations: usize,
    pub warmup: usize,
    pub only: Option<String>,
    pub repo: String,
    pub build_info: Vec<(String, String)>,
    pub rustc: String,
    pub cpu: String,
    pub logical_cpus: String,
    pub os: String,
    pub runtime_cpu_features: String,
    pub candidate_simd_and_std: bool,
    pub baseline_simd_and_std: bool,
    pub locked_packages: Vec<LockedPackage>,
    pub lockfile: String,
    pub load_sample: String,
    pub c_oracle: String,
    pub c_oracle_link_map: Option<String>,
    pub c_encoder: String,
    pub c_encoder_link_map: Option<String>,
    pub corpus: Vec<CorpusRecord>,
    pub decode: Vec<DecodeCaseReport>,
    pub encode: Vec<EncodeCaseReport>,
    pub thumbnail: Option<ThumbnailReport>,
    pub concurrent: Option<ConcurrentReport>,
}

/// Bumped when a consumer of report.json must change: /2 added `concurrent`.
const SCHEMA: &str = "downstream-consumer-report/2";

/// Dependency features as declared in the consumer's Cargo.toml. Static on
/// purpose: the manifest is the single place they are chosen.
const DECLARED_FEATURES: [(&str, &str); 4] = [
    (
        "libjpeg-turbo-rs (baseline and candidate)",
        "default features (`std`, `simd`)",
    ),
    ("libjpeg-turbo-rs-image (candidate)", "default features"),
    (
        "image",
        "`default-features = false`, `features = [\"jpeg\"]`",
    ),
    ("zune-jpeg", "default features"),
];

fn mib(bytes: u64) -> String {
    if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn ms(value: f64) -> String {
    if value >= 100.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    }
}

fn psnr_text(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.2}")
    } else {
        "inf".to_string()
    }
}

fn timing_cells(timing: Option<&TimingSummary>, megapixels_per_second: Option<f64>) -> String {
    match (timing, megapixels_per_second) {
        (Some(t), Some(rate)) => format!(
            "{}{} | {} | {} | {} | {} | {:.1}",
            ms(t.median_ms),
            if t.calls_per_sample > 1 {
                format!(" (×{})", t.calls_per_sample)
            } else {
                String::new()
            },
            ms(t.p10_ms),
            ms(t.p90_ms),
            ms(t.min_ms),
            ms(t.max_ms),
            rate
        ),
        _ => "— | — | — | — | — | —".to_string(),
    }
}

fn alloc_cells(alloc: Option<&AllocStats>) -> String {
    match alloc {
        Some(a) => format!("{} | {} | {}", a.count, mib(a.bytes), mib(a.peak_live)),
        None => "— | — | —".to_string(),
    }
}

fn diff_text(diff: Option<&PixelDiff>) -> String {
    match diff {
        Some(d) if d.max_abs == 0 => "identical".to_string(),
        Some(d) => format!(
            "max {} / mean {:.4} ({} samples differ)",
            d.max_abs, d.mean_abs, d.differing_samples
        ),
        None => "—".to_string(),
    }
}

fn timing_json(timing: Option<&TimingSummary>) -> Json {
    Json::opt(timing, |t| {
        Json::object(vec![
            ("iterations", Json::int(t.iterations as u64)),
            ("calls_per_sample", Json::int(t.calls_per_sample as u64)),
            ("median_ms", Json::num(t.median_ms)),
            ("p10_ms", Json::num(t.p10_ms)),
            ("p90_ms", Json::num(t.p90_ms)),
            ("min_ms", Json::num(t.min_ms)),
            ("max_ms", Json::num(t.max_ms)),
        ])
    })
}

fn alloc_json(alloc: Option<&AllocStats>) -> Json {
    Json::opt(alloc, |a| {
        Json::object(vec![
            ("count", Json::int(a.count)),
            ("bytes", Json::int(a.bytes)),
            ("peak_live_bytes", Json::int(a.peak_live)),
        ])
    })
}

fn diff_json(diff: Option<&PixelDiff>) -> Json {
    Json::opt(diff, |d| {
        Json::object(vec![
            ("max_abs", Json::int(d.max_abs)),
            ("mean_abs", Json::num(d.mean_abs)),
            ("differing_samples", Json::int(d.differing_samples as u64)),
        ])
    })
}

/// The decode and concurrent sections share one correctness contract, so
/// they share its rendering too.
fn correctness_markdown(out: &mut String, records: &[CorrectnessRecord]) {
    let _ = writeln!(
        out,
        "\nCorrectness (outside the timed region):\n\n| row | compared with | result |\n|---|---|---|"
    );
    for record in records {
        let result: String = match (&record.diff, &record.note) {
            (Some(diff), _) => diff_text(Some(diff)),
            (None, Some(note)) => note.clone(),
            (None, None) => "—".to_string(),
        };
        let _ = writeln!(
            out,
            "| {} | {} | {} |",
            record.subject, record.compared_to, result
        );
    }
}

fn correctness_json(records: &[CorrectnessRecord]) -> Json {
    Json::Array(
        records
            .iter()
            .map(|record| {
                Json::object(vec![
                    ("subject", Json::str(&record.subject)),
                    ("compared_to", Json::str(&record.compared_to)),
                    ("diff", diff_json(record.diff.as_ref())),
                    ("note", Json::opt(record.note.as_deref(), Json::str)),
                ])
            })
            .collect(),
    )
}

fn concurrent_json(section: &ConcurrentReport) -> Json {
    Json::object(vec![
        ("id", Json::str(&section.id)),
        ("corpus_id", Json::str(&section.corpus_id)),
        ("source_width", Json::int(section.source_width as u64)),
        ("source_height", Json::int(section.source_height as u64)),
        ("threads", Json::int(section.threads as u64)),
        (
            "decodes_per_thread",
            Json::int(section.decodes_per_thread as u64),
        ),
        ("memory_measure", Json::str(CONCURRENT_MEMORY_MEASURE)),
        (
            "rows",
            Json::Array(
                section
                    .rows
                    .iter()
                    .map(|row| {
                        Json::object(vec![
                            ("backend", Json::str(&row.backend)),
                            ("path", Json::str(&row.path)),
                            ("batch_timing", timing_json(row.timing.as_ref())),
                            (
                                "aggregate_megapixels_per_second",
                                Json::opt(row.megapixels_per_second, Json::num),
                            ),
                            ("alloc", alloc_json(row.alloc.as_ref())),
                            (
                                "caller_buffer_bytes",
                                Json::opt(row.caller_buffer_bytes, Json::int),
                            ),
                            (
                                "outputs_identical_to_single_threaded",
                                Json::opt(row.outputs_compared, |count| Json::int(count as u64)),
                            ),
                            (
                                "not_applicable",
                                Json::opt(row.not_applicable.as_deref(), Json::str),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("correctness", correctness_json(&section.correctness)),
    ])
}

/// Said in both renderings, so a reader of either cannot take the peak for
/// resident memory.
const CONCURRENT_MEMORY_MEASURE: &str = "peak live heap through the global allocator, all threads \
     together, above the heap at the batch's start; not RSS (allocator retention, fragmentation, \
     thread stacks and code pages are not in it)";

fn concurrent_markdown(out: &mut String, section: &ConcurrentReport) {
    let _ = writeln!(
        out,
        "\n## Concurrent decode\n\n`{}`: {} threads, each decoding `{}` ({}x{}, RGB) {} \
             times back to back; one batch is all {} decodes. Each thread owns its decoder per \
             decode and, on reuse rows, one output buffer. Times are ms per batch (wall, \
             thread spawn and join included); MP/s is the whole batch's source pixels at the \
             median. Allocation columns are one batch under the counting allocator: **{}**. \
             `caller buffers` is the reuse rows' T output buffers, allocated before that \
             window — an application holds them too, so compare `peak live heap + caller \
             buffers` across rows. Every output of the correctness batch is compared with the \
             same backend's single-threaded output, and those references are held to the \
             decode section's contract (table below); a difference aborts the run.\n",
        section.id,
        section.threads,
        section.corpus_id,
        section.source_width,
        section.source_height,
        section.decodes_per_thread,
        section.threads * section.decodes_per_thread,
        CONCURRENT_MEMORY_MEASURE
    );
    let _ = writeln!(
            out,
            "| row | path | median | p10 | p90 | min | max | MP/s | allocs | alloc bytes | peak live heap | caller buffers | vs single-threaded |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        );
    for row in &section.rows {
        match &row.not_applicable {
            Some(reason) => {
                let _ = writeln!(
                    out,
                    "| {} | {} | N/A: {} | — | — | — | — | — | — | — | — | — | — |",
                    row.backend, row.path, reason
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} |",
                    row.backend,
                    row.path,
                    timing_cells(row.timing.as_ref(), row.megapixels_per_second),
                    alloc_cells(row.alloc.as_ref()),
                    row.caller_buffer_bytes
                        .map(mib)
                        .unwrap_or_else(|| "—".to_string()),
                    row.outputs_compared
                        .map(|count| format!("{count} identical (asserted)"))
                        .unwrap_or_else(|| "—".to_string())
                );
            }
        }
    }
    correctness_markdown(out, &section.correctness);
}

fn frame_text(frame: Option<&FrameFacts>) -> String {
    frame
        .map(|f| format!("{}x{} {} {}", f.width, f.height, f.process, f.subsampling))
        .unwrap_or_else(|| "unreadable".to_string())
}

impl Report {
    pub fn markdown(&self) -> String {
        let mut out: String = String::new();
        let _ = writeln!(out, "# Downstream-consumer benchmark report\n");
        if self.smoke {
            let _ = writeln!(
                out,
                "> **SMOKE RUN — not a measurement.** {} timed iteration(s) after {} warmup: \
                 this run proves the harness and its correctness checks, nothing about speed.\n",
                self.iterations, self.warmup
            );
        }
        let _ = writeln!(out, "- Started (UTC): {}", self.started_at);
        let _ = writeln!(
            out,
            "- Iterations: {} timed, {} warmup per row; rounds interleave all rows of a case",
            self.iterations, self.warmup
        );
        if let Some(only) = &self.only {
            let _ = writeln!(out, "- Case filter: `--only {only}` (partial run)");
        }
        let _ = writeln!(out, "- Candidate checkout: `{}`", self.repo);
        let _ = writeln!(
            out,
            "- How to read this: `experiments/downstream/README.md`\n"
        );

        let _ = writeln!(out, "## Build\n");
        let _ = writeln!(out, "| key | value |\n|---|---|");
        for (key, value) in &self.build_info {
            let _ = writeln!(out, "| {key} | `{value}` |");
        }

        let _ = writeln!(out, "\n## Environment\n");
        let _ = writeln!(
            out,
            "- CPU: {} ({} logical CPUs)",
            self.cpu, self.logical_cpus
        );
        let _ = writeln!(out, "- OS: {}", self.os);
        let _ = writeln!(out, "- Runtime ISA: {}", self.runtime_cpu_features);
        let _ = writeln!(
            out,
            "- `simd_and_std_features_enabled()`: baseline {}, candidate {}",
            self.baseline_simd_and_std, self.candidate_simd_and_std
        );
        let _ = writeln!(out, "- C reference decoder: {}", self.c_oracle);
        let _ = writeln!(out, "- C reference encoder: {}", self.c_encoder);
        let _ = writeln!(out, "- Dependency features (from the consumer manifest):");
        for (crate_name, features) in DECLARED_FEATURES {
            let _ = writeln!(out, "  - {crate_name}: {features}");
        }
        let _ = writeln!(out, "\n```\n$ rustc -Vv\n{}\n```", self.rustc);
        if let Some(link_map) = &self.c_oracle_link_map {
            let _ = writeln!(
                out,
                "\nC reference decoder's link map:\n\n```\n{link_map}\n```"
            );
        }
        if let Some(link_map) = &self.c_encoder_link_map {
            let _ = writeln!(
                out,
                "\nC reference encoder's link map:\n\n```\n{link_map}\n```"
            );
        }

        let _ = writeln!(
            out,
            "\n## Resolved crate versions\n\nFrom `{}`.\n\n| package | version | source |\n|---|---|---|",
            self.lockfile
        );
        for package in &self.locked_packages {
            let _ = writeln!(
                out,
                "| {} | {} | {} |",
                package.name, package.version, package.source
            );
        }

        let _ = writeln!(
            out,
            "\n## Corpus\n\n| id | frame | bytes | sha256 | origin | licence |\n|---|---|---|---|---|---|"
        );
        for record in &self.corpus {
            let _ = writeln!(
                out,
                "| {} | {} | {} | `{}` | {} | {} |",
                record.id,
                frame_text(record.frame.as_ref()),
                record.bytes,
                record.sha256,
                record.origin,
                record.licence
            );
        }

        let _ = writeln!(
            out,
            "\n## Machine load before the run\n\n```\n{}\n```",
            self.load_sample
        );

        let _ = writeln!(out, "\n## Decode\n\nBackends (one row each per case):\n");
        let _ = writeln!(out, "| row | API | output buffer |\n|---|---|---|");
        for backend in DecodeBackend::ALL {
            let _ = writeln!(
                out,
                "| {} | {} | {} |",
                backend.id(),
                backend.api(),
                backend.buffer()
            );
        }
        let _ = writeln!(
            out,
            "\nTimes in ms per decode (decoder construction, decode, and dropping a \
             library-owned output included). A median marked (×K) comes from samples of \
             K back-to-back calls (rows whose single call is under 1 ms), divided by K. \
             MP/s is over source pixels at the median. \
             Allocation columns are one decode under the counting allocator: events, \
             cumulative bytes, peak live bytes above the start."
        );
        for case in &self.decode {
            let scale: String = case
                .scale
                .as_ref()
                .map(|s| format!(", DCT scale {s}"))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "\n### {} — {}\n\nSource {}x{} (`{}`), output {}{}.\n",
                case.id,
                case.description,
                case.source_width,
                case.source_height,
                case.corpus_id,
                case.layout,
                scale
            );
            let _ = writeln!(
                out,
                "| row | path | output | median | p10 | p90 | min | max | MP/s | allocs | alloc bytes | peak live |"
            );
            let _ = writeln!(
                out,
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            );
            for row in &case.rows {
                match &row.not_applicable {
                    Some(reason) => {
                        let _ = writeln!(
                            out,
                            "| {} | {} | N/A: {} | — | — | — | — | — | — | — | — | — |",
                            row.backend, row.path, reason
                        );
                    }
                    None => {
                        let _ = writeln!(
                            out,
                            "| {} | {} | {} | {} | {} |",
                            row.backend,
                            row.path,
                            row.output,
                            timing_cells(row.timing.as_ref(), row.megapixels_per_second),
                            alloc_cells(row.alloc.as_ref())
                        );
                    }
                }
            }
            correctness_markdown(&mut out, &case.correctness);
        }

        let _ = writeln!(out, "\n## Encode\n\nBackends:\n\n| row | API |\n|---|---|");
        for backend in EncodeBackend::ALL {
            let _ = writeln!(out, "| {} | {} |", backend.id(), backend.api());
        }
        let _ = writeln!(
            out,
            "\nQuality 85 everywhere. `subsampling` is read back from each output's SOF \
             marker: where it differs between rows, bytes and PSNR are not like for like. \
             `image-builtin` writes 4:4:4, so compare it with `baseline-444` / `candidate-444`. \
             PSNR is against the source pixels, every output decoded by the same decoder \
             (the published baseline)."
        );
        for case in &self.encode {
            let _ = writeln!(
                out,
                "\n### {} — {}x{} RGB\n\n| row | subsampling | bytes | PSNR dB | median | p10 | p90 | min | max | MP/s | allocs | alloc bytes | peak live |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                case.id, case.width, case.height
            );
            for row in &case.rows {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} |",
                    row.backend,
                    row.subsampling,
                    row.output_bytes,
                    psnr_text(row.psnr_db),
                    timing_cells(row.timing.as_ref(), row.megapixels_per_second),
                    alloc_cells(Some(&row.alloc))
                );
            }
            let _ = writeln!(
                out,
                "\nC cross-check (outside the timed region):\n\n| row | compared with | result |\n|---|---|---|"
            );
            for record in &case.c_comparison {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} |",
                    record.subject,
                    record.compared_to,
                    record.note.as_deref().unwrap_or("—")
                );
            }
        }

        if let Some(thumbnail) = &self.thumbnail {
            let _ = writeln!(
                out,
                "\n## Thumbnail workload\n\nDecode `{}` ({}x{}, EXIF orientation 6) → apply orientation → \
                 `image::imageops::resize` (Triangle) to fit 256 px → encode q85. Expected output \
                 {} (portrait). Thumbnails are decoded back by the published baseline and compared \
                 with the candidate row.\n",
                thumbnail.corpus_id,
                thumbnail.source_width,
                thumbnail.source_height,
                thumbnail.expected_output
            );
            let _ = writeln!(out, "| row | API |\n|---|---|");
            for backend in ThumbnailBackend::ALL {
                let _ = writeln!(out, "| {} | {} |", backend.id(), backend.api());
            }
            let _ = writeln!(
                out,
                "\n| row | output | orientation applied | bytes | vs candidate | median | p10 | p90 | min | max | MP/s | allocs | alloc bytes | peak live |\n|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            );
            for row in &thumbnail.rows {
                match &row.not_applicable {
                    Some(reason) => {
                        let _ = writeln!(
                            out,
                            "| {} | N/A: {} | — | — | — | — | — | — | — | — | — | — | — | — |",
                            row.backend, reason
                        );
                    }
                    None => {
                        let vs_candidate: String = match &row.diff_vs_candidate {
                            Some(diff) => diff_text(Some(diff)),
                            None => "different size".to_string(),
                        };
                        let _ = writeln!(
                            out,
                            "| {} | {} | {} | {} | {} | {} | {} |",
                            row.backend,
                            row.output,
                            if row.orientation_applied {
                                "yes"
                            } else {
                                "**no**"
                            },
                            row.output_bytes,
                            vs_candidate,
                            timing_cells(row.timing.as_ref(), row.megapixels_per_second),
                            alloc_cells(row.alloc.as_ref())
                        );
                    }
                }
            }
        }

        if let Some(section) = &self.concurrent {
            concurrent_markdown(&mut out, section);
        }
        out
    }

    pub fn json(&self) -> Json {
        let decode: Vec<Json> = self
            .decode
            .iter()
            .map(|case| {
                Json::object(vec![
                    ("id", Json::str(&case.id)),
                    ("corpus_id", Json::str(&case.corpus_id)),
                    ("description", Json::str(&case.description)),
                    ("source_width", Json::int(case.source_width as u64)),
                    ("source_height", Json::int(case.source_height as u64)),
                    ("output_layout", Json::str(&case.layout)),
                    ("scale", Json::opt(case.scale.as_deref(), Json::str)),
                    (
                        "rows",
                        Json::Array(
                            case.rows
                                .iter()
                                .map(|row| {
                                    Json::object(vec![
                                        ("backend", Json::str(&row.backend)),
                                        ("api", Json::str(&row.api)),
                                        ("buffer", Json::str(&row.buffer)),
                                        ("path", Json::str(&row.path)),
                                        ("output", Json::str(&row.output)),
                                        ("timing", timing_json(row.timing.as_ref())),
                                        (
                                            "megapixels_per_second",
                                            Json::opt(row.megapixels_per_second, Json::num),
                                        ),
                                        ("alloc", alloc_json(row.alloc.as_ref())),
                                        (
                                            "not_applicable",
                                            Json::opt(row.not_applicable.as_deref(), Json::str),
                                        ),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                    ("correctness", correctness_json(&case.correctness)),
                ])
            })
            .collect();
        let encode: Vec<Json> = self
            .encode
            .iter()
            .map(|case| {
                Json::object(vec![
                    ("id", Json::str(&case.id)),
                    ("width", Json::int(case.width as u64)),
                    ("height", Json::int(case.height as u64)),
                    (
                        "c_comparison",
                        Json::Array(
                            case.c_comparison
                                .iter()
                                .map(|record| {
                                    Json::object(vec![
                                        ("subject", Json::str(&record.subject)),
                                        ("compared_to", Json::str(&record.compared_to)),
                                        ("result", Json::opt(record.note.as_deref(), Json::str)),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                    (
                        "rows",
                        Json::Array(
                            case.rows
                                .iter()
                                .map(|row| {
                                    Json::object(vec![
                                        ("backend", Json::str(&row.backend)),
                                        ("api", Json::str(&row.api)),
                                        ("output_bytes", Json::int(row.output_bytes as u64)),
                                        ("subsampling", Json::str(&row.subsampling)),
                                        ("psnr_db", Json::num(row.psnr_db)),
                                        ("timing", timing_json(row.timing.as_ref())),
                                        (
                                            "megapixels_per_second",
                                            Json::opt(row.megapixels_per_second, Json::num),
                                        ),
                                        ("alloc", alloc_json(Some(&row.alloc))),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ])
            })
            .collect();
        let thumbnail: Json = Json::opt(self.thumbnail.as_ref(), |thumbnail| {
            Json::object(vec![
                ("corpus_id", Json::str(&thumbnail.corpus_id)),
                ("source_width", Json::int(thumbnail.source_width as u64)),
                ("source_height", Json::int(thumbnail.source_height as u64)),
                ("expected_output", Json::str(&thumbnail.expected_output)),
                (
                    "rows",
                    Json::Array(
                        thumbnail
                            .rows
                            .iter()
                            .map(|row| {
                                Json::object(vec![
                                    ("backend", Json::str(&row.backend)),
                                    ("api", Json::str(&row.api)),
                                    ("output", Json::str(&row.output)),
                                    ("output_bytes", Json::int(row.output_bytes as u64)),
                                    ("orientation_applied", Json::Bool(row.orientation_applied)),
                                    (
                                        "diff_vs_candidate",
                                        diff_json(row.diff_vs_candidate.as_ref()),
                                    ),
                                    ("timing", timing_json(row.timing.as_ref())),
                                    (
                                        "megapixels_per_second",
                                        Json::opt(row.megapixels_per_second, Json::num),
                                    ),
                                    ("alloc", alloc_json(row.alloc.as_ref())),
                                    (
                                        "not_applicable",
                                        Json::opt(row.not_applicable.as_deref(), Json::str),
                                    ),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ])
        });
        Json::object(vec![
            ("schema", Json::str(SCHEMA)),
            ("started_at", Json::str(&self.started_at)),
            ("smoke", Json::Bool(self.smoke)),
            ("iterations", Json::int(self.iterations as u64)),
            ("warmup", Json::int(self.warmup as u64)),
            ("only", Json::opt(self.only.as_deref(), Json::str)),
            ("candidate_checkout", Json::str(&self.repo)),
            (
                "build",
                Json::Object(
                    self.build_info
                        .iter()
                        .map(|(key, value)| (key.clone(), Json::str(value)))
                        .collect(),
                ),
            ),
            (
                "environment",
                Json::object(vec![
                    ("rustc", Json::str(&self.rustc)),
                    ("cpu", Json::str(&self.cpu)),
                    ("logical_cpus", Json::str(&self.logical_cpus)),
                    ("os", Json::str(&self.os)),
                    (
                        "runtime_cpu_features",
                        Json::str(&self.runtime_cpu_features),
                    ),
                    (
                        "baseline_simd_and_std",
                        Json::Bool(self.baseline_simd_and_std),
                    ),
                    (
                        "candidate_simd_and_std",
                        Json::Bool(self.candidate_simd_and_std),
                    ),
                    ("c_oracle", Json::str(&self.c_oracle)),
                    (
                        "c_oracle_link_map",
                        Json::opt(self.c_oracle_link_map.as_deref(), Json::str),
                    ),
                    ("c_encoder", Json::str(&self.c_encoder)),
                    (
                        "c_encoder_link_map",
                        Json::opt(self.c_encoder_link_map.as_deref(), Json::str),
                    ),
                    (
                        "declared_features",
                        Json::Object(
                            DECLARED_FEATURES
                                .iter()
                                .map(|(name, features)| (name.to_string(), Json::str(features)))
                                .collect(),
                        ),
                    ),
                    ("load_sample", Json::str(&self.load_sample)),
                ]),
            ),
            ("lockfile", Json::str(&self.lockfile)),
            (
                "resolved_packages",
                Json::Array(
                    self.locked_packages
                        .iter()
                        .map(|package| {
                            Json::object(vec![
                                ("name", Json::str(&package.name)),
                                ("version", Json::str(&package.version)),
                                ("source", Json::str(&package.source)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "corpus",
                Json::Array(
                    self.corpus
                        .iter()
                        .map(|record| {
                            Json::object(vec![
                                ("id", Json::str(&record.id)),
                                ("frame", Json::str(&frame_text(record.frame.as_ref()))),
                                ("bytes", Json::int(record.bytes as u64)),
                                ("sha256", Json::str(&record.sha256)),
                                ("origin", Json::str(&record.origin)),
                                ("licence", Json::str(&record.licence)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("decode", Json::Array(decode)),
            ("encode", Json::Array(encode)),
            ("thumbnail", thumbnail),
            (
                "concurrent",
                Json::opt(self.concurrent.as_ref(), concurrent_json),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section() -> ConcurrentReport {
        let batch: TimingSummary = TimingSummary {
            iterations: 2,
            calls_per_sample: 1,
            median_ms: 400.0,
            p10_ms: 390.0,
            p90_ms: 410.0,
            min_ms: 390.0,
            max_ms: 410.0,
        };
        ConcurrentReport {
            id: "concurrent-phone-4032x3024-420".to_string(),
            corpus_id: "synthetic-4032x3024-420".to_string(),
            source_width: 4032,
            source_height: 3024,
            threads: 4,
            decodes_per_thread: 4,
            rows: vec![
                ConcurrentRow {
                    backend: "candidate-reuse".to_string(),
                    path: "buffer-reuse".to_string(),
                    timing: Some(batch),
                    megapixels_per_second: Some(487.7),
                    alloc: Some(AllocStats {
                        count: 64,
                        bytes: 3 << 20,
                        peak_live: 1 << 20,
                    }),
                    caller_buffer_bytes: Some(4 * 36_578_304),
                    outputs_compared: Some(16),
                    not_applicable: None,
                },
                ConcurrentRow {
                    backend: "zune-fresh".to_string(),
                    path: "fresh".to_string(),
                    timing: None,
                    megapixels_per_second: None,
                    alloc: None,
                    caller_buffer_bytes: None,
                    outputs_compared: None,
                    not_applicable: Some("no API".to_string()),
                },
            ],
            correctness: vec![CorrectnessRecord::note(
                "all rows",
                "C djpeg",
                "C decode comparison: disabled (--no-c-oracle)".to_string(),
            )],
        }
    }

    fn field<'a>(object: &'a Json, key: &str) -> &'a Json {
        match object {
            Json::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value)
                .unwrap_or_else(|| panic!("missing key {key}")),
            other => panic!("not an object: {other:?}"),
        }
    }

    #[test]
    fn concurrent_json_records_the_pool_shape_and_per_row_memory() {
        let json: Json = concurrent_json(&section());
        assert!(matches!(field(&json, "threads"), Json::Integer(4)));
        assert!(matches!(
            field(&json, "decodes_per_thread"),
            Json::Integer(4)
        ));
        assert!(
            matches!(field(&json, "memory_measure"), Json::String(text) if text.contains("not RSS"))
        );
        let rows: &Vec<Json> = match field(&json, "rows") {
            Json::Array(rows) => rows,
            other => panic!("rows: {other:?}"),
        };
        assert_eq!(rows.len(), 2);
        let measured: &Json = &rows[0];
        assert!(matches!(
            field(field(measured, "batch_timing"), "median_ms"),
            Json::Number(value) if *value == 400.0
        ));
        assert!(matches!(
            field(field(measured, "alloc"), "peak_live_bytes"),
            Json::Integer(value) if *value == 1 << 20
        ));
        assert!(matches!(
            field(measured, "caller_buffer_bytes"),
            Json::Integer(146_313_216)
        ));
        assert!(matches!(
            field(measured, "outputs_identical_to_single_threaded"),
            Json::Integer(16)
        ));
        assert!(matches!(field(&rows[1], "batch_timing"), Json::Null));
        assert!(matches!(field(&rows[1], "not_applicable"), Json::String(_)));
        let correctness: &Vec<Json> = match field(&json, "correctness") {
            Json::Array(records) => records,
            other => panic!("correctness: {other:?}"),
        };
        assert!(matches!(
            field(&correctness[0], "note"),
            Json::String(text) if text.contains("disabled (--no-c-oracle)")
        ));
    }

    #[test]
    fn concurrent_markdown_says_heap_not_rss_and_fills_every_column() {
        let mut out: String = String::new();
        concurrent_markdown(&mut out, &section());
        assert!(out.contains("## Concurrent decode"));
        assert!(out.contains("4 threads"));
        assert!(out.contains("not RSS"));
        let header_columns: usize = out
            .lines()
            .find(|line| line.starts_with("| row |"))
            .expect("table header")
            .matches('|')
            .count();
        let rows: Vec<&str> = out
            .lines()
            .filter(|line| {
                line.starts_with("| candidate-reuse") || line.starts_with("| zune-fresh")
            })
            .collect();
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(row.matches('|').count(), header_columns, "{row}");
        }
        assert!(out.contains("| 139.5 MiB | 16 identical (asserted) |"));
        assert!(
            out.contains("| all rows | C djpeg | C decode comparison: disabled (--no-c-oracle) |")
        );
    }
}
