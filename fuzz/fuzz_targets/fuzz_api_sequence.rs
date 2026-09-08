#![no_main]
//! API-sequence fuzzer: ordered `TjHandle` lifecycles rather than one buffer
//! handed to one entry point.
//!
//! P4-141 criterion 3 (#480). None of the other twelve targets in this
//! directory touches `TjHandle`, mixes operation kinds on one handle, or
//! compares against a handle built fresh from the same configuration, so no
//! defect that needs *two* calls to appear — an operation reading what an
//! earlier one left on the handle — is reachable from any of them. This target
//! drives
//! `new → configure → probe → decode → reset → decode → transform → destroy`
//! orderings and compares every result against the same call on a handle built
//! fresh from the configuration alone.
//!
//! The engine, the wire format and the oracle live in
//! `tests/helpers/api_sequence.rs`, included by path here and by
//! `tests/api_sequence_state.rs`. The deterministic test proves the property
//! on every pull request; this target searches for inputs that break it.
//!
//! Input layout:
//!   byte 0            : operation count, taken modulo `MAX_OPS + 1`
//!   bytes 1..1+4*n    : one 4-byte record per operation
//!   remaining bytes   : the first of the images the program operates on
//!
//! Every `BUILTIN_INPUTS` fixture is available alongside it, at the indices
//! `fuzz_inputs` documents, so a program can switch between images that
//! disagree on every parameter a decode publishes. Without a second image,
//! comparing a published parameter against a fresh handle compares a number
//! with itself.

#[path = "../../tests/helpers/api_sequence.rs"]
mod api_sequence;

use api_sequence::{fuzz_inputs, program_from_bytes, run_program, Limits};
use libfuzzer_sys::fuzz_target;

/// A sixteenth of the ceiling the byte-decode targets use, on purpose.
///
/// Those targets decode **once** per input, so 1 MP is a sensible per-input
/// budget for them. This one runs a *program*: with `MAX_OPS = 16`, every
/// decode-family operation also builds a reference handle and repeats the
/// call, `Op::DecompressHeader` is a full decode in this port (P4-142) and
/// `Op::Compress` replays the last publishing decode — so a single input can
/// ask for well over thirty full decodes. At 1 MP against libFuzzer's
/// `-timeout=30` (`.github/workflows/fuzz-smoke.yml`) that is a hang report
/// whose cause is the harness rather than the library, and the interesting
/// state here is in the *ordering*, not in the resolution. 256 x 256 keeps
/// every committed seed and every `BUILTIN_INPUTS` entry — the largest is the
/// 12-bit fixture at 227 x 149 = 33,823 px — well inside the budget.
///
/// The engine blanks an input above it and sets it as `TJPARAM_MAXPIXELS` on
/// every handle, and `MAXPIXELS` / `MAXMEMORY` are excluded from the fuzzable
/// parameters so a program cannot raise its own cap.
const LIMITS: Limits = Limits {
    max_pixels: 65_536,
    prefilter_headers: true,
};

fuzz_target!(|data: &[u8]| {
    let (program, body) = program_from_bytes(data);
    if program.is_empty() {
        return;
    }
    let inputs: Vec<&[u8]> = fuzz_inputs(body);
    let _ = run_program(&inputs, &program, &LIMITS);
});
