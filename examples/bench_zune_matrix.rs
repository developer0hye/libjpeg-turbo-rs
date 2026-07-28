//! Wide-matrix decode comparison vs `zune-jpeg` (issue #360).
//!
//! Covers every subsampling (4:2:0 / 4:2:2 / 4:4:4 / 4:4:0 / grayscale)
//! × {baseline, progressive} × resolution tiers from 8×8 to 8K, plus
//! restart-interval and content-type cases — the categories whose
//! absence from `benches/compare.rs` hid the #350/#351/#352 losses.
//!
//! Per case it reports best-of-N wall clock, allocation count, and
//! allocated bytes for BOTH decoders, asserts output-length parity
//! (loudly flagging mismatches instead of reporting them as speed
//! differences), and ends with a win/loss summary so a single-mode
//! regression cannot hide in an average.
//!
//! Run with: `cargo run --release --example bench_zune_matrix`
//! (benchmarks must run alone — no parallel builds or tests.)

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

fn record(size: usize) {
    COUNTING.with(|c| {
        if c.get() {
            ALLOC_COUNT.with(|n| n.set(n.get() + 1));
            ALLOC_BYTES.with(|b| b.set(b.get() + size));
        }
    });
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocs<F: FnOnce()>(f: F) -> (usize, usize) {
    ALLOC_COUNT.with(|n| n.set(0));
    ALLOC_BYTES.with(|b| b.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    (ALLOC_COUNT.with(|n| n.get()), ALLOC_BYTES.with(|b| b.get()))
}

/// One timed measurement: best-of-N and median-of-N wall clock, plus the
/// iteration count actually used.
struct Timing {
    best_us: f64,
    median_us: f64,
    iters: usize,
}

/// Minimum measured iterations for any case. A sample this small is the
/// issue #376 failure mode (single perturbed shot → collapsed count →
/// noisy best-of-few), so anything that would fall below it is raised to
/// it and flagged in the output.
const MIN_ITERS: usize = 20;

/// Best/median-of-N wall clock with a per-case time budget: keeps tiny
/// images statistically meaningful (thousands of iterations) without
/// letting 8K cases run for minutes.
///
/// Issue #376: the iteration count was sized from ONE untimed warm-up
/// shot; a single cold-cache/scheduler/thermal perturbation collapsed it
/// and the reported best-of-few swung 0.97x..1.37x across runs of the
/// same commit. The estimate is now the median of three warm-up shots,
/// the count has a hard floor, and the median is reported next to the
/// best so residual instability is visible instead of silent.
fn best_of_budget<F: FnMut()>(mut f: F) -> Timing {
    // Warm-up + median-of-3 estimate.
    let mut est: [f64; 3] = [0.0; 3];
    for slot in est.iter_mut() {
        let t = Instant::now();
        f();
        *slot = t.elapsed().as_secs_f64() * 1e6;
    }
    est.sort_by(|a, b| a.total_cmp(b));
    let est_us: f64 = est[1];
    let iters: usize = ((1_500_000.0 / est_us.max(1.0)) as usize).clamp(MIN_ITERS, 20_000);
    for _ in 0..iters / 20 {
        f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    Timing {
        best_us: samples[0],
        median_us: samples[iters / 2],
        iters,
    }
}

struct CaseResult {
    name: &'static str,
    ours: Timing,
    zune: Timing,
    ours_allocs: usize,
    ours_bytes: usize,
    zune_allocs: usize,
    zune_bytes: usize,
    len_ours: usize,
    len_zune: usize,
}

const CASES: &[(&str, &str)] = &[
    // --- tiny (fixed cost dominates) ---
    ("gray_8x8", "tests/fixtures/gray_8x8.jpg"),
    ("blue_16x16_420", "tests/fixtures/blue_16x16_420.jpg"),
    (
        "blue_16x16_420_prog",
        "tests/fixtures/blue_16x16_420_prog.jpg",
    ),
    ("photo_64x64_420", "tests/fixtures/photo_64x64_420.jpg"),
    (
        "photo_64x64_420_prog",
        "tests/fixtures/photo_64x64_420_prog.jpg",
    ),
    (
        "nonint_440_64x64",
        "tests/fixtures/real_world/zune_non_interleaved_440_64x64.jpg",
    ),
    // --- small ---
    ("photo_320x240_420", "tests/fixtures/photo_320x240_420.jpg"),
    ("photo_320x240_422", "tests/fixtures/photo_320x240_422.jpg"),
    ("photo_320x240_444", "tests/fixtures/photo_320x240_444.jpg"),
    (
        "photo_320x240_420_prog",
        "tests/fixtures/photo_320x240_420_prog.jpg",
    ),
    (
        "gray_227x149",
        "tests/fixtures/real_world/derived_227x149_grayscale_q90.jpg",
    ),
    // --- medium ---
    ("photo_640x480_420", "tests/fixtures/photo_640x480_420.jpg"),
    ("photo_640x480_422", "tests/fixtures/photo_640x480_422.jpg"),
    ("photo_640x480_444", "tests/fixtures/photo_640x480_444.jpg"),
    (
        "photo_640x480_420_rst",
        "tests/fixtures/photo_640x480_420_rst.jpg",
    ),
    (
        "photo_640x480_422_prog",
        "tests/fixtures/photo_640x480_422_prog.jpg",
    ),
    (
        "photo_640x480_444_prog",
        "tests/fixtures/photo_640x480_444_prog.jpg",
    ),
    ("gradient_640x480", "tests/fixtures/gradient_640x480.jpg"),
    (
        "graphic_640x480_420",
        "tests/fixtures/graphic_640x480_420.jpg",
    ),
    (
        "checker_640x480_420",
        "tests/fixtures/checker_640x480_420.jpg",
    ),
    // --- HD ---
    (
        "photo_1280x720_420",
        "tests/fixtures/photo_1280x720_420.jpg",
    ),
    (
        "photo_1920x1080_420",
        "tests/fixtures/photo_1920x1080_420.jpg",
    ),
    (
        "photo_1920x1080_422",
        "tests/fixtures/photo_1920x1080_422.jpg",
    ),
    (
        "photo_1920x1080_444",
        "tests/fixtures/photo_1920x1080_444.jpg",
    ),
    (
        "graphic_1920x1080_420",
        "tests/fixtures/graphic_1920x1080_420.jpg",
    ),
    (
        "photo_1920x1080_420_prog",
        "tests/fixtures/photo_1920x1080_420_prog.jpg",
    ),
    (
        "photo_1920x1080_422_prog",
        "tests/fixtures/photo_1920x1080_422_prog.jpg",
    ),
    (
        "photo_1920x1080_444_prog",
        "tests/fixtures/photo_1920x1080_444_prog.jpg",
    ),
    (
        "gray_900x675_prog",
        "tests/fixtures/real_world/zune_grayscale_progressive_900x675.jpg",
    ),
    // --- QHD / 4K / 8K ---
    (
        "photo_2560x1440_420",
        "tests/fixtures/photo_2560x1440_420.jpg",
    ),
    (
        "photo_3840x2160_420",
        "tests/fixtures/photo_3840x2160_420.jpg",
    ),
    (
        "photo_3840x2160_420_prog",
        "tests/fixtures/photo_3840x2160_420_prog.jpg",
    ),
    (
        "rw_2048x1536_q90",
        "tests/fixtures/real_world/derived_2048x1536_baseline_q90.jpg",
    ),
    (
        "rw_4k_420_q85",
        "tests/fixtures/real_world/derived_3840x2160_4k_420_q85.jpg",
    ),
    (
        "rw_4k_progressive",
        "tests/fixtures/real_world/derived_3840x2160_4k_progressive.jpg",
    ),
    (
        "rw_8k_420_q75",
        "tests/fixtures/real_world/derived_7680x4320_8k_420_q75.jpg",
    ),
    (
        "rw_8k_progressive",
        "tests/fixtures/real_world/derived_7680x4320_8k_progressive.jpg",
    ),
];

fn main() {
    let mut results: Vec<CaseResult> = Vec::new();

    for &(name, path) in CASES {
        let jpeg = std::fs::read(path).unwrap_or_else(|_| panic!("{path} fixture required"));

        // Allocation profile: one decode per decoder under the counter.
        let mut len_ours = 0usize;
        let (ours_allocs, ours_bytes) = count_allocs(|| {
            let img = libjpeg_turbo_rs::decompress(&jpeg).expect("ours decode");
            len_ours = img.data.len();
        });
        let mut len_zune = 0usize;
        let (zune_allocs, zune_bytes) = count_allocs(|| {
            let cursor = std::io::Cursor::new(&jpeg);
            let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
            let pixels = decoder.decode().expect("zune decode");
            len_zune = pixels.len();
        });

        let ours = best_of_budget(|| {
            let img = libjpeg_turbo_rs::decompress(std::hint::black_box(&jpeg)).unwrap();
            std::hint::black_box(&img.data);
        });
        let zune = best_of_budget(|| {
            let cursor = std::io::Cursor::new(std::hint::black_box(&jpeg));
            let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
            let pixels = decoder.decode().unwrap();
            std::hint::black_box(&pixels);
        });

        results.push(CaseResult {
            name,
            ours,
            zune,
            ours_allocs,
            ours_bytes,
            zune_allocs,
            zune_bytes,
            len_ours,
            len_zune,
        });
    }

    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>10} {:>6}  {:>7} {:>12}  {:>7} {:>12}  parity",
        "case",
        "ours(us)",
        "o.med",
        "zune(us)",
        "z.med",
        "ratio",
        "allocs",
        "bytes",
        "z.allocs",
        "z.bytes"
    );
    let (mut wins, mut losses, mut ties, mut mismatches) = (0usize, 0usize, 0usize, 0usize);
    let mut loss_names: Vec<String> = Vec::new();
    let mut floored: Vec<&str> = Vec::new();
    for r in &results {
        let ratio = r.ours.best_us / r.zune.best_us;
        // Announce degraded samples instead of hiding them (issue #376):
        // a case pinned at the iteration floor was too slow for its
        // budget, so its best-of-N rests on the minimum sample size.
        if r.ours.iters == MIN_ITERS || r.zune.iters == MIN_ITERS {
            floored.push(r.name);
        }
        // Output-length parity: a decoder emitting a different pixel
        // format must be flagged, not scored (criterion 4 of #360).
        let parity = if r.len_ours == r.len_zune {
            "ok".to_string()
        } else {
            mismatches += 1;
            format!("LEN-MISMATCH ours={} zune={}", r.len_ours, r.len_zune)
        };
        if r.len_ours == r.len_zune {
            if ratio <= 0.98 {
                wins += 1;
            } else if ratio >= 1.02 {
                losses += 1;
                loss_names.push(format!("{} ({ratio:.2}x)", r.name));
            } else {
                ties += 1;
            }
        }
        println!(
            "{:<26} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>6.2}  {:>7} {:>12}  {:>7} {:>12}  {}",
            r.name,
            r.ours.best_us,
            r.ours.median_us,
            r.zune.best_us,
            r.zune.median_us,
            ratio,
            r.ours_allocs,
            r.ours_bytes,
            r.zune_allocs,
            r.zune_bytes,
            parity
        );
    }

    println!(
        "\nsummary: {wins} wins / {losses} losses / {ties} ties (±2% threshold, \
         {} scored cases) + {mismatches} format-mismatch cases (unscored)",
        wins + losses + ties
    );
    if !loss_names.is_empty() {
        println!("losses: {}", loss_names.join(", "));
    }
    if !floored.is_empty() {
        println!(
            "note: iteration floor ({MIN_ITERS}) hit for: {} — best-of-N for these \
             rests on the minimum sample size; treat their ratios with care.",
            floored.join(", ")
        );
    }
    if mismatches > 0 {
        println!(
            "note: LEN-MISMATCH cases compare different output formats \
             (e.g. zune expands grayscale to RGB); their timings are \
             printed for reference but excluded from win/loss."
        );
    }
}
