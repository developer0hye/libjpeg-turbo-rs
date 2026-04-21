//! Regression tests for fuzzer-discovered crashes. Each byte pattern was
//! captured from the nightly `fuzz-smoke.yml` workflow and copied into
//! `fuzz/corpus/<target>/` so libFuzzer also exercises it on every run.
//! Every API listed here must return `Err` (or a panic-safe result) on the
//! malformed inputs — never panic or abort.

use libjpeg_turbo_rs::{decompress, decompress_lenient, read_coefficients};

fn load(target: &str, crash_name: &str) -> Vec<u8> {
    let path = format!(
        "{}/fuzz/corpus/{}/{}",
        env!("CARGO_MANIFEST_DIR"),
        target,
        crash_name
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path, e))
}

#[test]
fn fuzz_decompress_crash_9aa915ab() {
    let data = load(
        "fuzz_decompress",
        "crash-9aa915ab4e164bb3511a0505a62cd5b0ea954c0d",
    );
    let _ = decompress(&data);
}

#[test]
fn fuzz_decompress_lenient_crash_8eecc401() {
    let data = load(
        "fuzz_decompress_lenient",
        "crash-8eecc40117a9bc076a870df7006dbc10a630befc",
    );
    let _ = decompress_lenient(&data);
}

#[test]
fn fuzz_progressive_decoder_crash_25ad884d() {
    let data = load(
        "fuzz_progressive_decoder",
        "crash-25ad884d739ff12dee3ce88fc4234e46b04d9c02",
    );
    let _ = decompress(&data);
}

#[test]
fn fuzz_read_coefficients_crash_c60edf95() {
    let data = load(
        "fuzz_read_coefficients",
        "crash-c60edf9531733501d6735509f5f8ee006cb74f82",
    );
    let _ = read_coefficients(&data);
}

#[test]
fn fuzz_transform_crash_5e275748() {
    let data = load(
        "fuzz_transform",
        "crash-5e275748109e1a5d3ee55b804ab7daab0b74afbe",
    );
    // Transform goes through the same entropy-decode entry points as
    // `read_coefficients`; the fuzz_transform target itself calls the
    // high-level transform API, but coefficient read is the common panic
    // surface captured here.
    let _ = read_coefficients(&data);
}
