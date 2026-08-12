//! P4-148 (#526): no byte array in this crate's tests may stand in for a
//! libjpeg struct — enforced, not merely swept once.
//!
//! The error manager used to be allocated as `MaybeUninit<[u8; 512]>` and cast
//! to a `*mut jpeg_error_mgr` for `jpeg_std_error` to write a struct through.
//! `[u8; N]` has alignment 1. Nothing guaranteed the pointer met
//! `align_of::<JpegErrorMgr>()`; it worked because stack slots happen to be
//! over-aligned, which is the kind of accident a compiler version or a target
//! is free to stop providing. **43 sites** — the 42 `MaybeUninit` ones the item
//! counted, plus a `Box<[u8; 512]>` in `capi_classic_decode_ext.rs` that sat
//! directly beneath a comment explaining why the *cinfo* beside it had been
//! converted for exactly this reason.
//!
//! **Why a source gate rather than the Miri run the item asked for.** Every one
//! of the eleven affected suites locates and `dlopen`s the cdylib through
//! `libloading`, so none of them runs under Miri at all:
//!
//! ```text
//! $ cargo +nightly miri test -p libjpeg-turbo-rs-capi --test arith_code_flag
//! panicked at tests/arith_code_flag.rs:55:5:
//! could not locate cdylib near .../nightly-aarch64-apple-darwin/bin/miri
//! ```
//!
//! That is not a gap this fix can close — Miri has no FFI, and these tests
//! exist precisely to exercise the shared object a C caller links. The
//! criterion asked for Miri because Miri rejects misaligned references, making
//! it a *mechanism* rather than a convention. Naming the mirrored struct is a
//! stronger mechanism than either: alignment stops being a property anything
//! can check at run time and becomes one the compiler guarantees, because the
//! storage now *is* the struct. What no compiler can prevent is someone
//! reintroducing the byte-array idiom later, and that is what this gate holds.
//!
//! **It matches the bug, not one spelling of it.** The first version pinned
//! literals, so `MaybeUninit::<[u8; 512]>::zeroed()` — the more common
//! turbofish construction — matched nothing, and a `type ErrBlob = [u8; 512];`
//! alias would have laundered any form past it. Lines are normalised before
//! matching and byte-array aliases are banned outright; the self-check below
//! carries six real spellings that must be flagged and five legitimate lines
//! that must not.
//!
//! **What this gate is not.** A determined author can still defeat a source
//! scan. The guarantee is the type; this only catches the accidental
//! copy-paste of the old idiom, which is the realistic regression.
//!
//! **Environment:** this reads the repository source tree, so it is skipped
//! where that tree is not reachable — `wasm32-wasip1` under wasmtime (which
//! preopens only `.` and `/tmp`) and a packaged crate. It runs on every native
//! leg, which is where a developer would write the pattern in the first place.

use std::path::{Path, PathBuf};

/// Storage shapes that have no legitimate use in this crate's tests, written
/// against [`normalize`]d text so a different but equivalent spelling does not
/// slip past.
///
/// Each is a fixed-size byte array used as *struct backing*, which is the
/// defect. A genuine byte **buffer** — a JPEG stream, a scanline row, a
/// four-byte marker — is a bare `[u8; N]`, a `Vec<u8>` or a slice, and those
/// are untouched: several tests legitimately declare `let src: [u8; 12]`. What
/// makes these three different is that the array is standing in for a struct
/// the library will write through, which is what makes its alignment load
/// bearing.
const BANNED_SHAPES: [&str; 4] = [
    "MaybeUninit<[u8;",
    "Box<[u8;",
    "as*mut[u8;",
    // A type alias would otherwise launder any of the above past a substring
    // scan: `type ErrBlob = [u8; 512];` then `MaybeUninit<ErrBlob>`. There is
    // no legitimate byte-array alias in these tests, so banning the alias
    // itself closes the hole without having to resolve it.
    "=[u8;",
];

/// Collapse the spellings Rust accepts for the same type into one form.
///
/// Removing whitespace folds `MaybeUninit < [u8 ; 512] >` together with the
/// rustfmt-normalised form, and dropping the turbofish `::` before `<` folds
/// `MaybeUninit::<[u8; 512]>::zeroed()` — the common construction spelling —
/// into `MaybeUninit<[u8;512]>`. Without this the gate would pin one way of
/// writing the bug rather than the bug, which is the failure mode of every
/// substring lint.
fn normalize(line: &str) -> String {
    let dense: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    dense.replace("::<", "<")
}

/// This file, which necessarily contains the banned shapes as *data* — the
/// list above and the self-check below. Excluding it is the usual cost of a
/// source gate written in the language it inspects; the sibling test covers
/// the risk by proving the scan still reaches real suites.
const SELF: &str = "err_mgr_alignment_gate.rs";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn tests_dir() -> PathBuf {
    crate_root().join("tests")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// A line counts only if it is code. The module docs above quote the banned
/// shapes to explain them, and a gate that tripped on its own explanation
/// would be removed rather than obeyed.
fn is_comment(line: &str) -> bool {
    let trimmed: &str = line.trim_start();
    trimmed.starts_with("//") || trimmed.starts_with("*") || trimmed.starts_with("/*")
}

#[test]
fn no_byte_array_stands_in_for_a_libjpeg_struct() {
    let dir: PathBuf = tests_dir();
    if !dir.is_dir() {
        eprintln!(
            "SKIP: {} is not readable. This gate inspects repository sources, \
             which a packaged crate and a sandboxed target (wasm32-wasip1) do \
             not provide. It runs on every native leg.",
            dir.display()
        );
        return;
    }

    let mut sources: Vec<PathBuf> = Vec::new();
    walk(&dir, &mut sources);
    assert!(
        !sources.is_empty(),
        "no test sources found under {} — the gate would pass vacuously",
        dir.display()
    );

    let mut offenders: Vec<String> = Vec::new();
    for path in &sources {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let name: &str = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        if name == SELF {
            continue;
        }
        for (index, line) in text.lines().enumerate() {
            if is_comment(line) {
                continue;
            }
            let normalized: String = normalize(line);
            for shape in BANNED_SHAPES {
                if normalized.contains(shape) {
                    offenders.push(format!("  {name}:{}\t{}", index + 1, line.trim()));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "a byte array is being used as storage for a libjpeg struct.\n\n{}\n\n\
         `[u8; N]` has alignment 1, so casting it to a struct pointer is \
         undefined however large it is — it works only while stack slots happen \
         to be over-aligned. Name the mirrored struct instead, so the compiler \
         guarantees the alignment:\n\n    \
         let mut err: MaybeUninit<JpegErrorMgr> = MaybeUninit::zeroed();\n\n\
         These suites `dlopen` the cdylib, so Miri cannot check them (P4-148); \
         this gate is what stands in for it.",
        offenders.join("\n")
    );
}

/// The gate must be able to see the pattern it bans.
///
/// A scanner with a broken path, a wrong extension filter or an over-eager
/// comment skip passes silently forever — the failure mode of every source
/// gate. This feeds it the real shapes and requires a hit.
#[test]
fn the_gate_detects_the_pattern_it_bans() {
    /// Real lines a future test could plausibly write, each recreating the
    /// under-aligned storage in a *different* spelling. Pinning only the one
    /// form this change happened to remove would be overfitting: the turbofish
    /// construction below is the more common way to write it, and matched
    /// nothing until `normalize` was added.
    const MUST_FLAG: [&str; 6] = [
        "        let mut err: MaybeUninit<[u8; 512]> = MaybeUninit::zeroed();",
        "        let mut err = MaybeUninit::<[u8; 512]>::zeroed();",
        "        let mut err = MaybeUninit :: < [u8 ; 512] > :: uninit();",
        "        let err: Box<[u8; 512]> = Box::new([0u8; 512]);",
        "        let err_box = unsafe { Box::from_raw(err_ptr as *mut [u8; 512]) };",
        "        type ErrBlob = [u8; 512];",
    ];
    for line in MUST_FLAG {
        let normalized: String = normalize(line);
        assert!(
            !is_comment(line) && BANNED_SHAPES.iter().any(|s| normalized.contains(s)),
            "the scanner would not flag: {line}"
        );
    }

    /// Lines that must **not** trip it: genuine byte buffers, which several
    /// suites legitimately declare, and the prose that explains the ban. A gate
    /// that fired on these would be deleted rather than obeyed.
    const MUST_NOT_FLAG: [&str; 5] = [
        "        let src: [u8; 12] = [0u8; 12];",
        "        let mut dst: [u8; 64] = [0u8; 64];",
        "    let data: [u8; 4] = [0xFF, 0xD8, 0xFF, 0xD9];",
        "        let mut err: MaybeUninit<JpegErrorMgr> = MaybeUninit::zeroed();",
        "    // a `MaybeUninit<[u8; 512]>` is align-1",
    ];
    for line in MUST_NOT_FLAG {
        let normalized: String = normalize(line);
        assert!(
            is_comment(line) || !BANNED_SHAPES.iter().any(|s| normalized.contains(s)),
            "the scanner would wrongly flag: {line}"
        );
    }

    // The scan must reach the suites that carried the defect. Checking for a
    // real one rather than for this file matters: this file is skipped, so
    // finding it would have proved nothing about coverage.
    let mut sources: Vec<PathBuf> = Vec::new();
    walk(&tests_dir(), &mut sources);
    if sources.is_empty() {
        eprintln!("SKIP: repository sources not readable; see the sibling test.");
        return;
    }
    for expected in ["arith_code_flag.rs", "capi_classic_decode_ext.rs"] {
        assert!(
            sources
                .iter()
                .any(|p| p.file_name().is_some_and(|n| n == expected)),
            "the scan did not reach {expected}, one of the suites P4-148 converted, \
             so its coverage is not what it appears"
        );
    }
}
