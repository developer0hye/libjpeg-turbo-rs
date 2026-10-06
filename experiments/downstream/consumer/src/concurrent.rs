//! Concurrent decode workload (issue #635, Milestone C): T worker threads,
//! each decoding the same 12 MP photo K times back to back, the way an
//! application with a small decode pool does. The single-threaded rows say
//! nothing about what such an application holds at once — T decoders' working
//! sets plus T outputs — which is what this section reports.

use std::hint::black_box;
use std::thread;

use crate::decode::PreparedDecode;
use crate::measure::{self, PixelDiff, TimingSummary};

/// The section's id, matched by `--only`.
pub const SECTION_ID: &str = "concurrent-phone-4032x3024-420";

/// The 12 MP phone photo: large enough that per-thread working sets and
/// outputs dominate the heap, and the size a photo-handling service sees.
pub const CORPUS_ID: &str = "synthetic-4032x3024-420";

/// A small pool, as a request handler or thumbnailer would run. More workers
/// than cores would time the scheduler rather than the decoders, so the
/// machine caps it; the hosted arm64 runners have 3.
const MAX_WORKERS: usize = 4;

pub fn worker_count(available_parallelism: usize) -> usize {
    available_parallelism.clamp(1, MAX_WORKERS)
}

/// Back-to-back decodes per thread and batch. More than one, so the reuse
/// rows actually reuse their buffer and a library that leaks or grows state
/// between decodes on one thread shows it. The smoke run only needs to prove
/// that path once.
pub fn decodes_per_thread(smoke: bool) -> usize {
    if smoke {
        2
    } else {
        4
    }
}

/// Source megapixels decoded by the whole batch per second of batch wall
/// time, at the median.
pub fn aggregate_megapixels_per_second(
    width: usize,
    height: usize,
    threads: usize,
    decodes_per_thread: usize,
    batch: &TimingSummary,
) -> f64 {
    (width * height * threads * decodes_per_thread) as f64 / 1e6 / (batch.median_ms / 1e3)
}

/// `None` when `output` is byte-identical to `reference`.
pub fn output_difference(reference: &[u8], output: &[u8]) -> Option<PixelDiff> {
    if reference == output {
        return None;
    }
    if reference.len() != output.len() {
        // pixel_diff needs equal lengths; a length change is a total mismatch.
        return Some(PixelDiff {
            max_abs: u8::MAX,
            mean_abs: f64::NAN,
            differing_samples: reference.len().max(output.len()),
        });
    }
    Some(measure::pixel_diff(reference, output))
}

/// One batch: every worker on its own thread, `decodes` decodes each, outputs
/// dropped as they are produced. `thread::scope` joins every worker before
/// returning, which is what makes the allocation counters exact when this
/// runs inside `alloc_counter::measure` (see that module).
pub fn run_batch(workers: &mut [PreparedDecode<'_>], decodes: usize) {
    thread::scope(|scope| {
        for worker in workers.iter_mut() {
            scope.spawn(move || {
                for _ in 0..decodes {
                    let decoded = worker.decode();
                    black_box(&decoded);
                }
            });
        }
    });
}

/// The correctness batch: like [`run_batch`], but every output of every
/// thread is compared with `reference` (the same backend's single-threaded
/// output, dimensions included). Returns how many outputs were compared, or
/// the first difference found.
pub fn check_batch(
    workers: &mut [PreparedDecode<'_>],
    decodes: usize,
    reference: (usize, usize, &[u8]),
) -> Result<usize, String> {
    let (reference_width, reference_height, reference_pixels) = reference;
    let results: Vec<Result<usize, String>> = thread::scope(|scope| {
        let handles: Vec<thread::ScopedJoinHandle<'_, Result<usize, String>>> = workers
            .iter_mut()
            .enumerate()
            .map(|(thread_index, worker)| {
                scope.spawn(move || {
                    for decode_index in 0..decodes {
                        let decoded = worker.decode();
                        if (decoded.width, decoded.height) != (reference_width, reference_height) {
                            return Err(format!(
                                "thread {thread_index} decode {decode_index}: {}x{}, single-threaded {reference_width}x{reference_height}",
                                decoded.width, decoded.height
                            ));
                        }
                        if let Some(diff) =
                            output_difference(reference_pixels, worker.pixels(&decoded))
                        {
                            return Err(format!(
                                "thread {thread_index} decode {decode_index}: {diff:?}"
                            ));
                        }
                    }
                    Ok(decodes)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a decode worker panicked"))
            .collect()
    });
    results.into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_is_capped_by_the_pool_size_and_the_machine() {
        assert_eq!(worker_count(1), 1);
        assert_eq!(worker_count(3), 3);
        assert_eq!(worker_count(4), 4);
        assert_eq!(worker_count(64), 4);
        // available_parallelism never returns 0, but a clamp must not either.
        assert_eq!(worker_count(0), 1);
    }

    #[test]
    fn smoke_runs_fewer_but_still_repeated_decodes() {
        assert!(decodes_per_thread(true) >= 2);
        assert!(decodes_per_thread(true) < decodes_per_thread(false));
    }

    #[test]
    fn aggregate_rate_counts_every_decode_of_the_batch() {
        let batch: TimingSummary = TimingSummary {
            iterations: 3,
            calls_per_sample: 1,
            median_ms: 500.0,
            p10_ms: 400.0,
            p90_ms: 600.0,
            min_ms: 400.0,
            max_ms: 600.0,
        };
        // 4 threads × 4 decodes of 4032×3024 (12.192768 MP) in 0.5 s.
        let rate: f64 = aggregate_megapixels_per_second(4032, 3024, 4, 4, &batch);
        assert!((rate - 12.192768 * 16.0 / 0.5).abs() < 1e-9, "{rate}");
        // Halving the threads halves the work in the same wall time.
        let half: f64 = aggregate_megapixels_per_second(4032, 3024, 2, 4, &batch);
        assert!((rate / half - 2.0).abs() < 1e-12);
    }

    fn small_case() -> crate::decode::DecodeCase {
        let file: crate::corpus::CorpusFile =
            crate::corpus::synthetic_rgb_jpeg("synthetic-64x64-420", 64, 64, false, None);
        crate::decode::DecodeCase {
            id: "concurrent-test".to_string(),
            corpus_id: file.id.clone(),
            description: "64x64 4:2:0".to_string(),
            jpeg: file.jpeg,
            layout: crate::corpus::OutputLayout::Rgb,
            scale: None,
            source_width: 64,
            source_height: 64,
        }
    }

    fn workers(
        backend: crate::decode::DecodeBackend,
        case: &crate::decode::DecodeCase,
        threads: usize,
    ) -> Vec<PreparedDecode<'_>> {
        (0..threads)
            .map(|_| match PreparedDecode::prepare(backend, case) {
                crate::decode::Preparation::Ready(ready) => ready,
                crate::decode::Preparation::NotApplicable(reason) => panic!("{reason}"),
            })
            .collect()
    }

    #[test]
    fn every_backend_matches_itself_across_threads() {
        let _serial = crate::alloc_counter::HEAVY_ALLOCATION_TESTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let case: crate::decode::DecodeCase = small_case();
        for backend in crate::decode::DecodeBackend::ALL {
            let mut pool: Vec<PreparedDecode<'_>> = workers(backend, &case, 3);
            let decoded = pool[0].decode();
            let reference: Vec<u8> = pool[0].pixels(&decoded).to_vec();
            let compared: Result<usize, String> =
                check_batch(&mut pool, 2, (decoded.width, decoded.height, &reference));
            assert_eq!(compared, Ok(6), "{}", backend.id());
        }
    }

    #[test]
    fn a_differing_or_resized_output_fails_the_batch() {
        let _serial = crate::alloc_counter::HEAVY_ALLOCATION_TESTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let case: crate::decode::DecodeCase = small_case();
        let mut pool: Vec<PreparedDecode<'_>> =
            workers(crate::decode::DecodeBackend::CandidateReuse, &case, 2);
        let decoded = pool[0].decode();
        let mut reference: Vec<u8> = pool[0].pixels(&decoded).to_vec();
        reference[100] ^= 1;
        let error: String = check_batch(&mut pool, 2, (64, 64, &reference)).expect_err("differs");
        assert!(error.contains("max_abs: 1"), "{error}");
        reference[100] ^= 1;
        let error: String = check_batch(&mut pool, 2, (32, 64, &reference)).expect_err("resized");
        assert!(error.contains("64x64"), "{error}");
    }

    #[test]
    fn output_difference_reports_value_and_length_mismatches() {
        assert!(output_difference(&[1, 2, 3], &[1, 2, 3]).is_none());
        let changed: PixelDiff = output_difference(&[1, 2, 3], &[1, 5, 3]).expect("differs");
        assert_eq!((changed.max_abs, changed.differing_samples), (3, 1));
        let truncated: PixelDiff = output_difference(&[1, 2, 3], &[1, 2]).expect("differs");
        assert_eq!(truncated.differing_samples, 3);
    }
}
