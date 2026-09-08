//! P4-141 criterion 3 (#480): the deterministic half of the API-sequence
//! harness.
//!
//! No byte fuzzer in `fuzz/fuzz_targets/` touches `TjHandle`, mixes operation
//! kinds on one handle, or compares a result against a handle built fresh from
//! the same configuration. The defect class that needs *two* calls to
//! appear — a decode that reads what a previous decode, probe or compress left
//! on the handle — has therefore never been reachable by any of them. The
//! criterion asks for a fuzzer driving
//! `new → configure → probe → decode → reset → decode → transform → destroy`
//! orderings; this file drives the same engine deterministically so the
//! property is proved on every pull request rather than only on the 6-hourly
//! `Fuzz Smoke` schedule.
//!
//! The engine and its oracle live in [`api_sequence`], included by path here
//! and from `fuzz/fuzz_targets/fuzz_api_sequence.rs`, so the property CI
//! proves and the property libFuzzer searches are literally the same code.

#[path = "helpers/api_sequence.rs"]
mod api_sequence;

use api_sequence::{
    decode_outcome, encode_program, fuzz_inputs, program_from_bytes, run_program, run_program_with,
    Limits, Op, ReferencePolicy, ALL_PARAMS, BUILTIN_INPUTS, FUZZ_INPUT_BODY,
    FUZZ_INPUT_COLOR_DENSE, FUZZ_INPUT_GRAY, FUZZ_INPUT_LOSSLESS16, FUZZ_INPUT_LOSSY12,
};
use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
use libjpeg_turbo_rs::{CropRegion, PixelFormat, Subsampling};

/// `fuzz_api_sequence`'s frame-header ceiling, kept in step with it so the
/// deterministic runs and the fuzzed ones take the same path through
/// `run_program`. Every fixture here is far below it — the largest is the
/// 12-bit built-in at 227 x 149 = 33,823 px — so it never fires in this file;
/// the tests that need it to fire set their own, lower value.
const LIMITS: Limits = Limits {
    max_pixels: 65_536,
    prefilter_headers: true,
};

/// The committed fixtures, plus two built here because the tree has no
/// arithmetic-coded and no ICC-carrying fixture small enough to be cheap.
fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut corpus: Vec<(&'static str, Vec<u8>)> = vec![
        ("gray_8x8", include_bytes!("fixtures/gray_8x8.jpg").to_vec()),
        (
            "blue_16x16_420",
            include_bytes!("fixtures/blue_16x16_420.jpg").to_vec(),
        ),
        (
            "blue_16x16_420_prog",
            include_bytes!("fixtures/blue_16x16_420_prog.jpg").to_vec(),
        ),
        (
            "cjpeg_7x7_square_444",
            include_bytes!("fixtures/cjpeg_7x7_square_444.jpg").to_vec(),
        ),
        (
            "photo_64x64_420",
            include_bytes!("fixtures/photo_64x64_420.jpg").to_vec(),
        ),
        (
            "color_16x16_422_dense",
            include_bytes!("fixtures/api_sequence_color_16x16_422_dense.jpg").to_vec(),
        ),
        (
            "lossless16_gray_8x8",
            include_bytes!("inputs/api_sequence_lossless16_gray_8x8.jpg").to_vec(),
        ),
        // The only 12-bit source in the tree, and so the only input on which
        // `Op::Decompress12` returns `Ok` — see `BUILTIN_INPUTS`.
        (
            "lossy12_227x149",
            include_bytes!("fixtures/real_world/libjpeg_testorig12_227x149_12bit.jpg").to_vec(),
        ),
        // Truncated: the error paths are part of the sequence space, and a
        // failed decode must leave the handle no less predictable than a
        // successful one.
        ("truncated", {
            let full: &[u8] = include_bytes!("fixtures/photo_64x64_420.jpg");
            full[..full.len() / 2].to_vec()
        }),
    ];

    let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 7 % 251) as u8).collect();
    corpus.push((
        "arithmetic_32x32",
        libjpeg_turbo_rs::compress_arithmetic(
            &pixels,
            32,
            32,
            PixelFormat::Rgb,
            80,
            Subsampling::S422,
        )
        .expect("arithmetic encode must succeed"),
    ));
    corpus.push((
        "icc_32x32",
        libjpeg_turbo_rs::Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
            .quality(80)
            .subsampling(Subsampling::S420)
            .icc_profile(&[0xAB; 96])
            .encode()
            .expect("ICC encode must succeed"),
    ));
    corpus
}

/// The subset used by the exhaustive ordering sweep, which runs 8^3 prefixes
/// per fixture: the six cheapest images, so the sweep stays a couple of
/// seconds in debug and does not dominate the suite under `wasm32-wasip1`,
/// where the whole integration suite runs unoptimised under wasmtime. Every
/// fixture is still swept by the other tests in this file.
fn cheap_corpus() -> Vec<(&'static str, Vec<u8>)> {
    corpus()
        .into_iter()
        .filter(|(label, _)| {
            matches!(
                *label,
                "gray_8x8"
                    | "blue_16x16_420"
                    | "blue_16x16_420_prog"
                    | "cjpeg_7x7_square_444"
                    | "color_16x16_422_dense"
                    | "lossless16_gray_8x8"
            )
        })
        .collect()
}

/// The exact ordering the criterion names, spelled out rather than generated.
fn criterion_named_program() -> Vec<Op> {
    vec![
        // configure
        Op::Set {
            param: TjParam::Quality,
            value: 82,
        },
        Op::Set {
            param: TjParam::Subsampling,
            value: 2,
        },
        Op::Set {
            param: TjParam::SaveMarkers,
            value: 2,
        },
        Op::SetScaling { index: 12 },
        Op::SetCrop {
            region: Some(CropRegion {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            }),
        },
        // probe
        Op::DecompressHeader,
        Op::InspectHeader,
        // decode
        Op::Decompress,
        // reset (destroy + init)
        Op::Reset,
        // decode again, on a handle that must behave like a brand-new one
        Op::Decompress,
        // transform
        Op::Transform { op: 5, flags: 0 },
        // destroy — the engine drops the handle when the program ends
    ]
}

#[test]
fn criterion_named_sequence_leaves_no_state_behind() {
    let program: Vec<Op> = criterion_named_program();
    for (label, jpeg) in corpus() {
        let inputs: Vec<&[u8]> = fuzz_inputs(&jpeg);
        let report = run_program(&inputs, &program, &LIMITS);
        assert_eq!(
            report.executed,
            program.len(),
            "{label}: the named sequence dropped an operation"
        );
    }
}

/// The direct driver for P4: decode one image, then another that disagrees on
/// every published parameter, then the first again. A decode that publishes
/// only once — or that leaves one field from the previous image in place —
/// diverges from a fresh handle at the second `Decompress`.
#[test]
fn published_parameters_do_not_survive_a_change_of_image() {
    let corpus: Vec<(&'static str, Vec<u8>)> = corpus();
    let program: Vec<Op> = vec![
        Op::Decompress,
        Op::SelectInput { index: 1 },
        Op::Decompress,
        Op::SelectInput { index: 0 },
        Op::Decompress,
        Op::DecompressHeader,
        Op::SelectInput { index: 1 },
        Op::DecompressHeader,
        Op::Decompress12,
        Op::Decompress16,
        Op::SelectInput { index: 0 },
        Op::Decompress12,
        Op::Decompress16,
    ];
    for (first_label, first) in &corpus {
        for (second_label, second) in &corpus {
            let inputs: [&[u8]; 2] = [first, second];
            let report = run_program(&inputs, &program, &LIMITS);
            assert!(
                report.executed > 0,
                "{first_label} then {second_label}: nothing executed"
            );
        }
    }
}

#[test]
fn decode_is_independent_of_every_three_operation_prefix() {
    // Every ordering of three operations drawn from the alphabet below,
    // each followed by a decode whose result is compared against a fresh
    // handle. 8^3 = 512 prefixes per fixture.
    let alphabet: [Op; 8] = [
        Op::DecompressHeader,
        Op::InspectHeader,
        Op::Decompress,
        Op::Decompress12,
        Op::Compress {
            width: 8,
            height: 8,
            format: 1,
        },
        Op::Transform { op: 1, flags: 0 },
        Op::SelectInput { index: 1 },
        Op::Reset,
    ];
    let corpus: Vec<(&'static str, Vec<u8>)> = cheap_corpus();
    assert_eq!(corpus.len(), 6, "the cheap corpus filter lost a fixture");
    for (label, jpeg) in &corpus {
        for a in &alphabet {
            for b in &alphabet {
                for c in &alphabet {
                    let program: Vec<Op> = vec![
                        Op::Set {
                            param: TjParam::Quality,
                            value: 90,
                        },
                        Op::Set {
                            param: TjParam::Subsampling,
                            value: 1,
                        },
                        a.clone(),
                        b.clone(),
                        c.clone(),
                        Op::Decompress,
                    ];
                    let inputs: Vec<&[u8]> = fuzz_inputs(jpeg);
                    let report = run_program(&inputs, &program, &LIMITS);
                    assert_eq!(
                        report.executed,
                        program.len(),
                        "{label}: dropped an operation"
                    );
                }
            }
        }
    }
}

#[test]
fn byte_programs_run_clean() {
    // The fuzzer's own path: bytes in, program out, program run. A
    // deterministic xorshift stands in for libFuzzer so the same decoding is
    // exercised on every pull request.
    let corpus: Vec<(&'static str, Vec<u8>)> = corpus();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let next = |state: &mut u64| -> u8 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 24) as u8
    };
    for round in 0..256u32 {
        // Exactly one length byte plus its records, so `inputs[0]` is the
        // fixture verbatim rather than a fixture behind a random prefix — a
        // program whose every decode hits an error path exercises much less
        // than it looks like it does.
        let announced: u8 = next(&mut state);
        let count: usize = announced as usize % (api_sequence::MAX_OPS + 1);
        let mut header: Vec<u8> = vec![announced];
        for _ in 0..count * 4 {
            header.push(next(&mut state));
        }
        let (_, jpeg) = &corpus[round as usize % corpus.len()];
        let mut input: Vec<u8> = header;
        input.extend_from_slice(jpeg);
        let (program, body) = program_from_bytes(&input);
        // `fuzz_inputs`, not a hand-built list: this is the stand-in for
        // libFuzzer and the only test that runs *generated* `SelectInput`
        // indices, so a shorter list here folds index 3 back onto the body
        // and the 16-bit image is never selected.
        let inputs: Vec<&[u8]> = fuzz_inputs(body);
        let report = run_program(&inputs, &program, &LIMITS);
        assert!(
            !report.skipped_oversize,
            "round {round}: fixtures are small"
        );
        assert_eq!(
            report.executed,
            program.len(),
            "round {round}: an operation was dropped without being counted"
        );
    }
}

#[test]
fn program_decoding_is_total_and_deterministic() {
    // `program_from_bytes` is the fuzzer's front door: it must accept any
    // byte string, including empty and truncated ones, and must be a
    // function of its input.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for length in 0..96usize {
        let mut bytes: Vec<u8> = Vec::with_capacity(length);
        for _ in 0..length {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push(state as u8);
        }
        let (first, first_body) = program_from_bytes(&bytes);
        let (second, second_body) = program_from_bytes(&bytes);
        assert_eq!(first, second, "length {length}: decoding is not a function");
        assert_eq!(first_body, second_body, "length {length}: body differs");
        assert!(
            first.len() <= api_sequence::MAX_OPS,
            "length {length}: program exceeds MAX_OPS"
        );
    }
}

/// The premise the whole of P4 rests on: the built-in images must be able to
/// put a *different* value in each published slot than the operation under
/// test will write. Where they cannot, the comparison is a number against
/// itself.
///
/// Measured rather than asserted from the fixture names — the first version of
/// this harness used two images that both reported `PRECISION = 8` and the
/// JFIF default density `0 / 1 / 1`, which are also `TjHandle::new()`'s
/// initial values, so deleting the entire density write-back from
/// `TjHandle::decompress` left every test and all 168 seeds it then had green.
#[test]
fn the_builtin_inputs_can_move_every_published_parameter() {
    let published: &[TjParam] = &[
        TjParam::Width,
        TjParam::Height,
        TjParam::Precision,
        TjParam::ColorSpace,
        TjParam::Subsampling,
        TjParam::XDensity,
        TjParam::YDensity,
        TjParam::DensityUnits,
    ];
    let mut observed: Vec<Vec<i32>> = vec![Vec::new(); published.len()];
    let mut decoded: usize = 0;
    for input in BUILTIN_INPUTS {
        // Whichever decode-family entry point accepts this image; a built-in
        // no entry point accepts contributes nothing and is a defect in the
        // set, which the count below catches.
        let mut handle: TjHandle = TjHandle::new();
        let accepted: bool = handle.decompress(input).is_ok()
            || handle.decompress_16bit(input).is_ok()
            || handle.decompress_12bit(input).is_ok();
        assert!(accepted, "a built-in input that no entry point decodes");
        decoded += 1;
        for (slot, &param) in published.iter().enumerate() {
            observed[slot].push(handle.get(param));
        }
    }
    assert_eq!(decoded, BUILTIN_INPUTS.len());
    for (slot, &param) in published.iter().enumerate() {
        let values: &Vec<i32> = &observed[slot];
        assert!(
            values.iter().any(|value| *value != values[0]),
            "every built-in input publishes {param:?} = {}, so P4's comparison \
             of it cannot fail — add an image that differs",
            values[0]
        );
    }
}

/// `ALL_PARAMS` drives P4's "and nothing else" half, so a parameter missing
/// from it is one an operation could write unnoticed.
#[test]
fn all_params_lists_every_tjparam() {
    // Exhaustive by construction: a new `TjParam` variant makes this match
    // fail to compile, and the count below then fails until `ALL_PARAMS`
    // grows to match.
    fn is_a_parameter(param: TjParam) -> bool {
        match param {
            TjParam::Quality
            | TjParam::Subsampling
            | TjParam::Width
            | TjParam::Height
            | TjParam::Precision
            | TjParam::ColorSpace
            | TjParam::FastUpSample
            | TjParam::FastDct
            | TjParam::Optimize
            | TjParam::Progressive
            | TjParam::ScanLimit
            | TjParam::Arithmetic
            | TjParam::Lossless
            | TjParam::LosslessPsv
            | TjParam::LosslessPt
            | TjParam::RestartBlocks
            | TjParam::RestartRows
            | TjParam::XDensity
            | TjParam::YDensity
            | TjParam::DensityUnits
            | TjParam::MaxMemory
            | TjParam::MaxPixels
            | TjParam::BottomUp
            | TjParam::NoRealloc
            | TjParam::StopOnWarning
            | TjParam::SaveMarkers => true,
        }
    }
    assert_eq!(
        ALL_PARAMS.len(),
        26,
        "TjParam gained or lost a variant; ALL_PARAMS must list every one"
    );
    for &param in ALL_PARAMS {
        assert!(is_a_parameter(param));
        assert_eq!(
            ALL_PARAMS.iter().filter(|other| **other == param).count(),
            1,
            "{param:?} is listed twice"
        );
    }
}

/// The exact program `fuzz_api_sequence` failed on ninety seconds into its
/// first run, kept as a regression on the *harness*: P2's reference replayed
/// the configuration and then the last publishing decode, which reorders
/// `set(SUBSAMP) → decode` into `decode → set(SUBSAMP)`. A decode publishes
/// `SUBSAMP`, so the two orders leave different values behind and the
/// reference produced a different JPEG than the live handle for a reason that
/// was entirely the oracle's fault. The prefix is replayed in order now.
#[test]
fn the_reference_replays_configuration_and_the_decode_in_their_original_order() {
    let program: Vec<Op> = vec![
        Op::Set {
            param: TjParam::Quality,
            value: 90,
        },
        Op::Decompress,
        Op::Set {
            param: TjParam::Quality,
            value: 90,
        },
        Op::Set {
            param: TjParam::Subsampling,
            value: 0,
        },
        Op::Compress {
            width: 15,
            height: 7,
            format: 0,
        },
    ];
    for (label, jpeg) in corpus() {
        let inputs: [&[u8]; 3] = [&jpeg, BUILTIN_INPUTS[0], BUILTIN_INPUTS[1]];
        let report = run_program(&inputs, &program, &LIMITS);
        assert_eq!(
            report.executed,
            program.len(),
            "{label}: dropped an operation"
        );
    }
}

/// The seed generator writes `Op::SelectInput` indices against
/// [`fuzz_inputs`]'s layout, and nothing in the type system ties an index to
/// the image it is supposed to name. Decode each one and check.
///
/// The first version of the fuzz target passed two of the three built-ins
/// while the seeds addressed three, so the `precision_transition` program's
/// `Decompress16` calls landed on an 8-bit image and failed. Everything still
/// passed: the premise test checks `BUILTIN_INPUTS`, not what the target
/// actually hands the engine.
#[test]
fn the_fuzz_input_indices_name_the_images_they_claim() {
    let body: &[u8] = include_bytes!("fixtures/gray_8x8.jpg");
    let inputs: Vec<&[u8]> = fuzz_inputs(body);
    assert_eq!(
        inputs.len(),
        1 + BUILTIN_INPUTS.len(),
        "fuzz_inputs must expose every built-in"
    );
    assert_eq!(inputs[usize::from(FUZZ_INPUT_BODY)], body);

    let mut gray: TjHandle = TjHandle::new();
    let gray_image = gray
        .decompress(inputs[usize::from(FUZZ_INPUT_GRAY)])
        .expect("FUZZ_INPUT_GRAY must be an 8-bit image");
    assert_eq!((gray_image.width, gray_image.height), (8, 8));
    assert_eq!(gray.get(TjParam::ColorSpace), 2, "TJCS_GRAY");

    let mut dense: TjHandle = TjHandle::new();
    dense
        .decompress(inputs[usize::from(FUZZ_INPUT_COLOR_DENSE)])
        .expect("FUZZ_INPUT_COLOR_DENSE must be an 8-bit image");
    assert_eq!(
        (
            dense.get(TjParam::DensityUnits),
            dense.get(TjParam::XDensity),
            dense.get(TjParam::YDensity)
        ),
        (1, 72, 71),
        "FUZZ_INPUT_COLOR_DENSE is the only input carrying a non-default JFIF density"
    );

    let mut lossless: TjHandle = TjHandle::new();
    let sixteen = lossless
        .decompress_16bit(inputs[usize::from(FUZZ_INPUT_LOSSLESS16)])
        .expect("FUZZ_INPUT_LOSSLESS16 must be decodable by decompress_16bit");
    assert_eq!(sixteen.precision, 16);
    assert_eq!(
        lossless.get(TjParam::Precision),
        16,
        "this input is the only way PRECISION ever holds anything but 8"
    );

    let mut twelve: TjHandle = TjHandle::new();
    let lossy12 = twelve
        .decompress_12bit(inputs[usize::from(FUZZ_INPUT_LOSSY12)])
        .expect("FUZZ_INPUT_LOSSY12 must be decodable by decompress_12bit");
    assert_eq!((lossy12.width, lossy12.height), (227, 149));
    assert_eq!(
        twelve.get(TjParam::Precision),
        12,
        "this input is the only one `Decompress12` accepts, so it is the only \
         one on which P4's write-back comparison for that opcode runs"
    );
}

/// The 16-bit built-in must stay out of `tests/fixtures/`.
///
/// `examples/generate_corpus.rs` copies that tree wholesale into the C-parity
/// corpus, and `examples/corpus_test.rs` compares every corpus file through
/// the **8-bit** `decompress()`. This stream is 16-bit lossless, which that
/// entry point refuses — and the corpus harness records a Rust error as a
/// `crash`, so putting the file under `tests/fixtures/` turns a premise
/// mismatch into a job failure that names neither the file's precision nor the
/// premise. It did exactly that on the first push of this branch.
///
/// Both halves are asserted, because either alone is satisfiable by accident:
/// that the 8-bit entry point really does refuse it (the reason), and that the
/// file is not under `tests/fixtures/` (the consequence). The trap itself is
/// P4-201.
#[test]
fn the_sixteen_bit_builtin_stays_out_of_the_c_parity_corpus() {
    let sixteen_bit: &[u8] = BUILTIN_INPUTS[2];
    assert!(
        libjpeg_turbo_rs::decompress(sixteen_bit).is_err(),
        "if the 8-bit entry point ever reads this stream, the reason for \
         keeping it out of the corpus is gone and this test should go with it"
    );
    assert!(
        TjHandle::new().decompress_16bit(sixteen_bit).is_ok(),
        "and the precision entry point must still read it"
    );

    // The corpus copies by path, so the placement is what the corpus sees.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let inside: std::path::PathBuf =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let offenders: Vec<String> = walk(&inside)
            .into_iter()
            .filter(|path| {
                std::fs::read(path)
                    .map(|b| b == sixteen_bit)
                    .unwrap_or(false)
            })
            .map(|path| path.display().to_string())
            .collect();
        assert!(
            offenders.is_empty(),
            "the 16-bit lossless input is inside the C-parity corpus tree: {offenders:?}"
        );
    }
}

/// Every file under `dir`, recursively. Only used by the placement check
/// above, which needs the whole fixture tree because `copy_jpgs` recurses.
#[cfg(not(target_arch = "wasm32"))]
fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out: Vec<std::path::PathBuf> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path: std::path::PathBuf = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

/// `Op::Decompress12`'s write-back comparison must actually run.
///
/// P4's first half is gated on the live call succeeding, and
/// `decompress_12bit` refuses an 8-bit source (P4-171) and a 16-bit one — so
/// before `FUZZ_INPUT_LOSSY12` existed, that opcode returned `Err` on every
/// image the engine could select, in every deterministic test and in all 189
/// seeds, and its arm of P4 was dead. `rust-code-reviewer` measured it. This
/// pins the repair the only way that can fail: by asserting the call
/// *succeeded*, which is what makes the comparison execute.
#[test]
fn the_twelve_bit_write_back_comparison_runs_on_some_input() {
    let inputs: Vec<&[u8]> = fuzz_inputs(BUILTIN_INPUTS[0]);
    let program: Vec<Op> = vec![
        // Publish 8-bit facts first, so the 12-bit write-back has a different
        // value to overwrite rather than agreeing by coincidence.
        Op::SelectInput {
            index: FUZZ_INPUT_GRAY,
        },
        Op::Decompress,
        Op::SelectInput {
            index: FUZZ_INPUT_LOSSY12,
        },
        Op::Decompress12,
    ];
    let report = run_program(&inputs, &program, &LIMITS);
    assert_eq!(report.executed, program.len());
    assert_eq!(
        report.decode_errors, 0,
        "both decodes must succeed — a failing Decompress12 skips P4's \
         write-back comparison entirely"
    );

    // And the values it compares are genuinely different from what the
    // previous decode left, so the comparison is not a number against itself.
    let mut handle: TjHandle = TjHandle::new();
    handle
        .decompress(inputs[usize::from(FUZZ_INPUT_GRAY)])
        .expect("the grayscale built-in decodes");
    let before: (i32, i32, i32) = (
        handle.get(TjParam::Width),
        handle.get(TjParam::Height),
        handle.get(TjParam::Precision),
    );
    handle
        .decompress_12bit(inputs[usize::from(FUZZ_INPUT_LOSSY12)])
        .expect("the 12-bit built-in decodes");
    let after: (i32, i32, i32) = (
        handle.get(TjParam::Width),
        handle.get(TjParam::Height),
        handle.get(TjParam::Precision),
    );
    assert_ne!(
        before, after,
        "the 12-bit built-in must publish different values than the 8-bit one"
    );
}

/// P2 is free of a merged-ICC false positive only because
/// `decompress_header` is implemented as `decompress`.
///
/// `TjHandle` keeps one ICC field where upstream keeps two (P4-198, #619), so
/// `compress` reads whatever the last decode left. P2's reference replays only
/// the *most recent* publishing operation, so if `DecompressHeader` and
/// `Decompress` treated that field differently, the program
/// `SetIcc(v), Decompress, DecompressHeader, Compress` would leave the two
/// handles carrying different profiles and P2 would panic on the harness
/// rather than on the library — from four operations, well inside libFuzzer's
/// reach.
///
/// Making `decompress_header` header-only is P4-142, which is open. This
/// fails the moment it lands without carrying the ICC write with it, which is
/// a named test failure instead of a mystery crash.
#[test]
fn the_two_publishing_operations_agree_on_the_icc_profile() {
    let icc: Vec<u8> = vec![0x5A; 32];
    for (label, jpeg) in corpus() {
        for level in [0, 1, 2, 3, 4] {
            let mut header_side: TjHandle = TjHandle::new();
            header_side.set_icc_profile(Some(icc.clone()));
            header_side
                .set(TjParam::SaveMarkers, level)
                .expect("SAVEMARKERS accepts 0..=4");
            let header_ok: bool = header_side.decompress_header(&jpeg).is_ok();

            let mut decode_side: TjHandle = TjHandle::new();
            decode_side.set_icc_profile(Some(icc.clone()));
            decode_side
                .set(TjParam::SaveMarkers, level)
                .expect("SAVEMARKERS accepts 0..=4");
            let decode_ok: bool = decode_side.decompress(&jpeg).is_ok();

            assert_eq!(
                header_ok, decode_ok,
                "{label} @ SAVEMARKERS={level}: the two publishers disagree on \
                 whether the stream decodes"
            );
            assert_eq!(
                header_side.icc_profile().map(<[u8]>::to_vec),
                decode_side.icc_profile().map(<[u8]>::to_vec),
                "{label} @ SAVEMARKERS={level}: `decompress_header` and \
                 `decompress` must leave the same ICC profile on the handle, \
                 or P2's reference replay reports a harness bug as a library \
                 crash (P4-198 #619 via P4-142)"
            );
        }
    }
}

/// A header the pre-parse cannot read *because of a limit* must be kept out
/// of the run, not waved through.
///
/// `Decoder::new` refuses a stream whose scan count exceeds its own default of
/// 8192, and `TjHandle::decompress_12bit` / `decompress_16bit` read nothing
/// from the handle — they build their own decoder with `DecodeLimits::default()`
/// (P4-199, #620) — so `TJPARAM_MAXPIXELS` does not reach them at all. Treating
/// every parse failure as "safe to forward", which the first version did, let a
/// 16-bit lossless frame of any declared size through the ceiling.
/// `codex review` found it.
///
/// Every *other* parse failure is still forwarded on purpose: a malformed
/// stream is rejected cheaply by every entry point, and those error paths are
/// most of what a program of decode operations exercises.
#[test]
fn a_header_the_prefilter_cannot_read_because_of_a_limit_is_kept_out() {
    let jpeg: Vec<u8> = progressive_stream_with_excess_scans();
    assert!(
        matches!(
            libjpeg_turbo_rs::Decoder::new(&jpeg),
            Err(libjpeg_turbo_rs::JpegError::LimitExceeded { .. })
        ),
        "the fixture must be one the pre-parse refuses *for a limit*, or it \
         proves nothing"
    );
    assert!(
        TjHandle::new().inspect_header(&jpeg).is_ok(),
        "and one an uncapped handle answers — that is what made it a bypass"
    );

    let ceiling: Limits = Limits {
        max_pixels: 32,
        prefilter_headers: true,
    };
    let alone: [&[u8]; 1] = [&jpeg];
    let report = run_program(&alone, &[Op::Decompress, Op::InspectHeader], &ceiling);
    assert!(
        report.skipped_oversize,
        "as the only input it leaves nothing to run"
    );
    assert_eq!(report.executed, 0);

    // Alongside the built-ins it is blanked rather than removed, so the
    // 16-bit image keeps its index and the stream reaches no entry point.
    let inputs: Vec<&[u8]> = fuzz_inputs(&jpeg);
    let reachable = run_program(
        &inputs,
        &[
            Op::SelectInput {
                index: FUZZ_INPUT_BODY,
            },
            Op::Decompress,
            Op::Decompress16,
            Op::SelectInput {
                index: FUZZ_INPUT_LOSSLESS16,
            },
            Op::Decompress16,
        ],
        &LIMITS,
    );
    assert!(!reachable.skipped_oversize);
    assert_eq!(
        reachable.decode_errors, 2,
        "the blanked body decodes nothing at either precision, and the 16-bit \
         built-in still decodes"
    );
}

/// Every handle the engine builds carries the run's ceiling — including the
/// one `Op::Reset` creates, which the first version left uncapped while every
/// reference kept it. `codex review` reproduced the resulting P1 false
/// positive on `[Reset, InspectHeader]`.
///
/// The assertion is on the *live* handle's behaviour rather than on the
/// mechanism: a program whose every operation runs after a `Reset` must still
/// see the ceiling. With the pre-parse now excluding over-ceiling inputs, the
/// handle cap is belt-and-braces for `MAXPIXELS` — no input distinguishes the
/// two — so what this pins is the symmetry: whatever ceiling the references
/// carry, the reset handle carries too, which is what keeps P1 from firing on
/// the harness.
#[test]
fn a_handle_built_by_reset_carries_the_same_ceiling_as_the_references() {
    let inputs: Vec<&[u8]> = fuzz_inputs(BUILTIN_INPUTS[0]);
    let program: Vec<Op> = vec![
        Op::Decompress,
        Op::Reset,
        Op::Decompress,
        Op::InspectHeader,
        Op::SelectInput {
            index: FUZZ_INPUT_COLOR_DENSE,
        },
        Op::Decompress,
        Op::Reset,
        Op::Decompress,
    ];
    let report = run_program(&inputs, &program, &LIMITS);
    assert_eq!(report.executed, program.len());
    assert_eq!(
        report.decode_errors, 0,
        "the built-ins are well under the ceiling at every point in the program"
    );

    // And with the pre-parse off, so the handle's TJPARAM_MAXPIXELS is the
    // only ceiling, every decode after a `Reset` must still be refused. This
    // is the arrangement in which the reset path forgetting the cap is
    // observable at all — with the pre-parse on, nothing can tell the two
    // mechanisms apart.
    let handle_only: Limits = Limits {
        max_pixels: 32,
        prefilter_headers: false,
    };
    let after_reset = run_program(
        &inputs,
        &[
            Op::Reset,
            Op::Decompress,
            Op::InspectHeader,
            Op::DecompressHeader,
        ],
        &handle_only,
    );
    assert!(!after_reset.skipped_oversize, "the pre-parse is off");
    assert_eq!(after_reset.executed, 4);
    assert_eq!(
        after_reset.decode_errors, 3,
        "a handle created by Reset must carry the same ceiling as the first \
         one and as every reference"
    );
}

/// A progressive stream carrying more scans than `Decoder`'s default limit of
/// 8192, built by repeating a real encoder's scan section.
///
/// Hand-rolling SOS segments does not work: the scan walk reads entropy data,
/// so a structurally plausible but empty scan fails as `UnexpectedEof` before
/// the count matters, and the fixture would then prove nothing. Repeating the
/// scans of an 8x8 progressive JPEG gives 6 scans per copy in 186 bytes.
fn progressive_stream_with_excess_scans() -> Vec<u8> {
    let pixels: Vec<u8> = (0..8usize * 8).map(|i| (i * 7 % 251) as u8).collect();
    let base: Vec<u8> = libjpeg_turbo_rs::compress_progressive(
        &pixels,
        8,
        8,
        PixelFormat::Grayscale,
        50,
        Subsampling::S444,
    )
    .expect("progressive encode must succeed");
    let first_scan: usize = base
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("a progressive stream has an SOS marker");
    let scans: &[u8] = &base[first_scan..base.len() - 2];
    // 1400 copies x 6 scans is 8400, comfortably past the 8192 limit.
    let mut stream: Vec<u8> = base[..first_scan].to_vec();
    for _ in 0..1400 {
        stream.extend_from_slice(scans);
    }
    stream.extend_from_slice(&[0xFF, 0xD9]);
    stream
}

/// An input above the ceiling keeps its slot, so the named indices keep
/// naming the same images.
///
/// Compacting the list instead renumbers everything behind the dropped entry:
/// with a 64x64 body dropped, `FUZZ_INPUT_LOSSLESS16` (3) folds to `3 % 3`
/// and selects the grayscale image, so the only 16-bit decode in the seed
/// corpus quietly stops happening. `codex review` reproduced it by enlarging
/// an otherwise unrelated body.
#[test]
fn an_oversize_input_does_not_renumber_the_others() {
    let oversize: &[u8] = include_bytes!("fixtures/photo_64x64_420.jpg");
    let inputs: Vec<&[u8]> = fuzz_inputs(oversize);
    // Below the 64x64 body's 4096 pixels and the 12-bit built-in's 33,823,
    // above the three small ones — so an entry is blanked at the head of the
    // list *and* at its tail, and the three between them keep their indices.
    let ceiling: Limits = Limits {
        max_pixels: 256,
        prefilter_headers: true,
    };
    let program: Vec<Op> = vec![
        Op::SelectInput {
            index: FUZZ_INPUT_LOSSLESS16,
        },
        Op::Decompress16,
    ];
    let report = run_program(&inputs, &program, &ceiling);
    assert!(
        !report.skipped_oversize,
        "the built-ins are under the ceiling"
    );
    assert_eq!(
        report.decode_errors, 0,
        "FUZZ_INPUT_LOSSLESS16 must still select the 16-bit image when an \
         earlier input is blanked"
    );

    // The body is blanked, not removed: selecting it decodes nothing.
    let body_program: Vec<Op> = vec![
        Op::SelectInput {
            index: FUZZ_INPUT_BODY,
        },
        Op::Decompress,
    ];
    let blanked = run_program(&inputs, &body_program, &ceiling);
    assert_eq!(blanked.decode_errors, 1, "a blanked input decodes nothing");

    // Same at the tail: the 12-bit built-in is above this ceiling too, and it
    // is blanked in place rather than dropped — if it were dropped the list
    // would still hold four entries and index 4 would fold to 0.
    let tail_program: Vec<Op> = vec![
        Op::SelectInput {
            index: FUZZ_INPUT_LOSSY12,
        },
        Op::Decompress12,
    ];
    let tail = run_program(&inputs, &tail_program, &ceiling);
    assert_eq!(
        tail.decode_errors, 1,
        "a blanked tail input decodes nothing, and must not have been \
         replaced by the body"
    );
}

/// The comparator has to see metadata *bytes*, not metadata sizes.
///
/// Summarising a decode as `icc=Some(96) exif=None xmp=Some(5) ...` — which
/// the first version did — makes two images whose metadata differs but
/// happens to be the same length compare equal, so a decode returning the
/// previous image's XMP is invisible to P1, and to P4 too because none of
/// these is a `TjParam`. `codex review` demonstrated it with the packets
/// `first` and `other`; this is that pair.
#[test]
fn the_comparator_distinguishes_metadata_of_equal_length() {
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 7 % 251) as u8).collect();
    let encode = |xmp: &[u8], icc: &[u8], comment: &str| -> Vec<u8> {
        libjpeg_turbo_rs::Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
            .quality(80)
            .subsampling(Subsampling::S420)
            .xmp_data(xmp)
            .icc_profile(icc)
            .comment(comment)
            .encode()
            .expect("metadata encode must succeed")
    };
    let first: Vec<u8> = encode(b"first", &[0xAA; 64], "aaa");
    let second: Vec<u8> = encode(b"other", &[0xBB; 64], "bbb");
    assert_eq!(
        first.len(),
        second.len(),
        "the two streams must differ only in metadata content"
    );

    // With SAVEMARKERS off, the raw APP segments are not carried in
    // `saved_markers`, so the only thing that can distinguish these two
    // decodes is the parsed metadata itself. Comparing under the default
    // level would pass even if the parsed fields were reduced to lengths,
    // because the marker bytes would still differ.
    let no_markers: [Op; 1] = [Op::Set {
        param: TjParam::SaveMarkers,
        value: 0,
    }];
    let outcome_first = decode_outcome(&first, &no_markers, &LIMITS);
    let outcome_second = decode_outcome(&second, &no_markers, &LIMITS);
    assert_ne!(
        outcome_first, outcome_second,
        "two decodes differing only in metadata content must not compare equal — \
         P1 cannot see a stale profile otherwise"
    );
    assert_eq!(
        outcome_first,
        decode_outcome(&first, &no_markers, &LIMITS),
        "and the comparator must still be a function of the input"
    );
}

/// The seed generator assembles programs with [`encode_program`]; the fuzzer
/// reads them back with `program_from_bytes`. If the two halves of the wire
/// format ever disagree, every committed seed silently becomes a different
/// program than the one its name claims.
#[test]
fn program_round_trips_through_its_wire_format() {
    let jpeg: &[u8] = include_bytes!("fixtures/gray_8x8.jpg");
    let mut programs: Vec<Vec<Op>> = vec![criterion_named_program()];
    // Every one of the thirteen variants, including the two `None`
    // discriminator arms — opcode 2's `a % 2 == 0` and opcode 3's `a == 0` —
    // which are exactly where the encoder and decoder can disagree while
    // every `Some` case still round-trips.
    programs.push(vec![
        Op::SetIcc { profile: None },
        Op::SetCrop { region: None },
        Op::SetIcc {
            profile: Some(vec![0x5A; 128]),
        },
        Op::SetCrop {
            region: Some(CropRegion {
                x: 0,
                y: 15,
                width: 16,
                height: 16,
            }),
        },
        Op::SetScaling { index: 255 },
        Op::Set {
            param: TjParam::ScanLimit,
            value: -32768,
        },
        Op::Compress {
            width: 255,
            height: 255,
            format: 255,
        },
        Op::Transform {
            op: 255,
            flags: 255,
        },
        Op::SelectInput { index: 255 },
        Op::Decompress12,
        Op::Decompress16,
        Op::InspectHeader,
        Op::Reset,
    ]);
    for program in &programs {
        let mut bytes: Vec<u8> = encode_program(program);
        bytes.extend_from_slice(jpeg);
        let (decoded, body) = program_from_bytes(&bytes);
        assert_eq!(&decoded, program, "wire format is not an inverse");
        assert_eq!(body, jpeg, "the body must survive the round trip");
    }
}

/// The oracle must be able to fail. A reference handle that replays no
/// configuration decodes a different image whenever the configuration
/// changes the output, so `NoConfig` has to be caught — if it is not, the
/// comparison in `ConfigOnly` is comparing nothing.
///
/// `catch_unwind` aborts under `wasm32-wasip1`, so the mutation check is
/// native-only; the oracle it checks runs on every target.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn the_oracle_rejects_a_reference_that_skips_configuration() {
    let jpeg: &[u8] = include_bytes!("fixtures/photo_64x64_420.jpg");
    // 1/2 scaling changes every output byte, so a reference that never
    // applies it cannot agree with the live handle.
    let program: Vec<Op> = vec![Op::SetScaling { index: 12 }, Op::Decompress];

    let inputs: [&[u8]; 1] = [jpeg];
    let honest = std::panic::catch_unwind(|| {
        run_program_with(&inputs, &program, &LIMITS, ReferencePolicy::ConfigOnly)
    });
    assert!(
        honest.is_ok(),
        "the oracle must pass on the configuration it is given"
    );

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let broken = std::panic::catch_unwind(|| {
        run_program_with(&inputs, &program, &LIMITS, ReferencePolicy::NoConfig)
    });
    std::panic::set_hook(previous);
    assert!(
        broken.is_err(),
        "a reference handle that drops the configuration must be detected — \
         the oracle is vacuous otherwise"
    );
}

/// P2's own can-fail proof. `ReferencePolicy::NoWriteBackReplay` builds the
/// reference from the configuration alone, so a compress that reads a fact the
/// preceding decode published must diverge. If it does not, P2 is comparing a
/// handle against a copy of itself — which is what the first draft of this
/// harness did, by skipping the comparison entirely once anything had been
/// published.
///
/// `catch_unwind` aborts under `wasm32-wasip1`, so this is native-only; the
/// property it checks runs on every target.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn the_oracle_rejects_a_reference_that_skips_the_last_published_header() {
    // The dense built-in publishes SUBSAMP = 1 (4:2:2) and density 1/72x71;
    // a handle that never saw it has SUBSAMP unset, which `compress` refuses
    // outright (`turbojpeg-mp.c:95-98`).
    let inputs: [&[u8]; 1] = [BUILTIN_INPUTS[1]];
    let program: Vec<Op> = vec![
        Op::Set {
            param: TjParam::Quality,
            value: 80,
        },
        Op::Decompress,
        Op::Compress {
            width: 15,
            height: 15,
            format: 1,
        },
    ];

    let honest = std::panic::catch_unwind(|| {
        run_program_with(&inputs, &program, &LIMITS, ReferencePolicy::ConfigOnly)
    });
    assert!(
        honest.is_ok(),
        "replaying the last published header must reproduce the compress"
    );

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let broken = std::panic::catch_unwind(|| {
        run_program_with(
            &inputs,
            &program,
            &LIMITS,
            ReferencePolicy::NoWriteBackReplay,
        )
    });
    std::panic::set_hook(previous);
    assert!(
        broken.is_err(),
        "a reference that skips the last published header must be detected — \
         P2 is vacuous otherwise"
    );
}
