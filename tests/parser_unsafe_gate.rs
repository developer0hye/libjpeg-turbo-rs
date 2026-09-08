//! P4-141 criterion 5: parser and control-plane sources contain no `unsafe`
//! — enforced, not merely asserted in prose.
//!
//! Malformed-input handling is the codec's largest attack surface: marker
//! length parsing, the bit reader, Huffman and arithmetic entropy decoding,
//! progressive scan state (EOB runs, spectral ranges, coefficient indexing),
//! restart-marker resync, custom scan scripts, and the coefficient-order
//! tables those paths index by. Every byte of a hostile file flows through
//! these modules before any kernel sees it, so the criterion asks for them to
//! be entirely safe Rust: a bounds bug there must be a clean panic or a typed
//! error, never undefined behaviour.
//!
//! **The mechanism is this test.** Each file in `PARSER_FILES` and
//! `UNSAFE_FREE_KERNELS` must contain no `unsafe` token outside comments. A
//! new `unsafe` in any of them fails here until it is removed again — the
//! criterion's "replace with safe indexing and show the generated code is
//! unchanged, or justify with a benchmark" — so the property cannot quietly
//! regress the way a SAFETY comment would. `decode/huffman.rs` (2026-08-10,
//! #500) and `decode/progressive.rs` (this gate's first landing) were the two
//! files that had unchecked zigzag indexing; the rest already qualified and
//! are listed so they stay that way. The gate does not fail open on a new
//! module either: every `.rs` under `src/decode/` must be classified into one
//! of the three lists below, so adding a file is a conscious choice.
//!
//! Out of scope, deliberately: `common/huffman_table.rs` still holds the
//! `OnceBox` lazy-init `unsafe` that P4-141 criterion 6 names as the blocker
//! for `forbid(unsafe_code)` under `not(feature = "simd")`; it joins
//! `PARSER_FILES` when that lands. The files in `INVENTORY_OWNED` hold
//! strided-IDCT destination pointers and SIMD kernels — dispatch, not parsing
//! — and are owned by the criterion-4 `unsafe` inventory
//! (`docs/UNSAFE_INVENTORY.md`, gated by `tests/unsafe_inventory_gate.rs`),
//! not by this gate.
//!
//! **What the scanner understands.** The lexer lives in
//! `tests/helpers/unsafe_scan.rs`, shared with the criterion-4 inventory
//! gate (`tests/unsafe_inventory_gate.rs`): `//` comments, nested `/* */`
//! comments, `"…"` strings with backslash escapes, raw strings (`r"…"`,
//! `r#"…"#`, with `b`/`c` prefixes), and char/byte literals (including
//! `'"'`) are skipped, so an `unsafe` hidden after a `"\"//"` or inside a
//! multi-line literal is still found, and prose about `unsafe` in any
//! comment form is not. Each of those shapes is pinned here by
//! `scanner_distinguishes_code_from_comments_and_literals`. The deliberate
//! residue: a raw identifier `r#unsafe` is reported (a loud false positive,
//! and not something these files should contain anyway).
//!
//! **Environment:** this reads the repository source tree, so it is skipped
//! where that tree is not reachable — `wasm32-wasip1` under wasmtime (which
//! preopens only `.` and `/tmp`) and a packaged crate. It runs on every
//! native leg, which is where the `unsafe` would be introduced.

use std::path::{Path, PathBuf};

#[path = "helpers/unsafe_scan.rs"]
mod unsafe_scan;
use unsafe_scan::unsafe_token_lines;

/// Parser and control-plane sources that must stay free of `unsafe`
/// (P4-141 criterion 5). Repository-relative, forward slashes.
const PARSER_FILES: [&str; 26] = [
    "src/api/coefficient.rs",
    "src/api/encoder.rs",
    "src/common/quant_table.rs",
    "src/common/tables.rs",
    "src/common/types.rs",
    "src/decode/arithmetic.rs",
    "src/decode/bitstream.rs",
    "src/decode/boundary.rs",
    "src/decode/entropy.rs",
    "src/decode/huffman.rs",
    "src/decode/lossless.rs",
    "src/decode/marker.rs",
    "src/decode/mod.rs",
    "src/decode/pipeline.rs",
    "src/decode/pipeline_impl/api.rs",
    "src/decode/pipeline_impl/colorspace.rs",
    "src/decode/pipeline_impl/lossless.rs",
    "src/decode/pipeline_impl/output.rs",
    "src/decode/pipeline_impl/raw.rs",
    "src/decode/pipeline_impl/scan.rs",
    "src/decode/progressive.rs",
    "src/decode/resync.rs",
    "src/decode/toggles.rs",
    "src/encode/arithmetic.rs",
    "src/encode/marker_writer.rs",
    "src/encode/progressive.rs",
];

/// Scalar decode kernels that are `unsafe`-free today and must stay so —
/// they are the reference every SIMD kernel is checked against, and the
/// non-SIMD build aims at `forbid(unsafe_code)` (criterion 6).
const UNSAFE_FREE_KERNELS: [&str; 5] = [
    "src/decode/color.rs",
    "src/decode/dequant.rs",
    "src/decode/idct.rs",
    "src/decode/merged_upsample.rs",
    "src/decode/upsample.rs",
];

/// Decode sources that legitimately hold `unsafe` — strided-IDCT destination
/// pointers and their scan drivers. Owned by the criterion-4 inventory; this
/// gate only insists they are classified, not that they are clean.
const INVENTORY_OWNED: [&str; 7] = [
    "src/decode/idct_extended.rs",
    "src/decode/idct_scaled.rs",
    "src/decode/pipeline_impl/arithmetic.rs",
    "src/decode/pipeline_impl/baseline.rs",
    "src/decode/pipeline_impl/color.rs",
    "src/decode/pipeline_impl/progressive.rs",
    "src/decode/pipeline_impl/streaming.rs",
];

/// The directory whose every source must appear in one of the lists above.
const CLASSIFIED_ROOT: &str = "src/decode";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `false` when the repository source tree is not reachable — a packaged
/// crate, or a sandboxed target such as `wasm32-wasip1`. Reported explicitly
/// rather than silently, so a green run always means the gate either ran or
/// said why it did not.
fn repository_tree_is_readable() -> bool {
    repo_root().join("src").is_dir()
}

fn relative(path: &Path) -> String {
    unsafe_scan::relative(&repo_root(), path)
}

/// Every code `unsafe` token in `path`, as `line-number: text`.
fn unsafe_sites(path: &Path) -> Vec<String> {
    unsafe_scan::unsafe_sites(path)
        .into_iter()
        .map(|site| format!("{}: {}", site.line, site.text))
        .collect()
}

fn skip_reason() -> String {
    format!(
        "SKIP: the source tree is not readable from {}. This gate inspects \
         repository sources, which a packaged crate and a sandboxed target \
         (wasm32-wasip1) do not provide. It runs wherever a workflow runs \
         `cargo test --tests`.",
        repo_root().display()
    )
}

#[test]
fn parser_and_control_plane_sources_contain_no_unsafe() {
    if !repository_tree_is_readable() {
        eprintln!("{}", skip_reason());
        return;
    }

    let mut offenders: Vec<String> = Vec::new();
    for relative_path in PARSER_FILES.iter().chain(UNSAFE_FREE_KERNELS.iter()) {
        let path: PathBuf = repo_root().join(relative_path);
        assert!(
            path.is_file(),
            "{relative_path} is listed in this gate but does not exist; \
             if it was renamed or removed, update the list"
        );
        for site in unsafe_sites(&path) {
            offenders.push(format!("  {relative_path}:{site}"));
        }
    }

    assert!(
        offenders.is_empty(),
        "`unsafe` in a parser / control-plane source (P4-141 criterion 5).\n\n\
         Malformed-input handling must be safe Rust: replace the site with \
         safe indexing and show the generated code is unchanged, or justify \
         the cost with a benchmark and keep the check anyway — a bounds bug \
         here must panic or return an error, never be undefined behaviour.\n\n\
         {}\n",
        offenders.join("\n")
    );
}

/// The gate must not fail open: a new module under `src/decode/` that is in
/// none of the lists is a classification the author has not made yet.
#[test]
fn every_decode_source_is_classified() {
    if !repository_tree_is_readable() {
        eprintln!("{}", skip_reason());
        return;
    }

    let mut files: Vec<PathBuf> = Vec::new();
    unsafe_scan::rust_sources(&repo_root().join(CLASSIFIED_ROOT), &mut files);
    assert!(
        !files.is_empty(),
        "no sources found under {CLASSIFIED_ROOT}"
    );

    let unclassified: Vec<String> = files
        .iter()
        .map(|path| relative(path))
        .filter(|rel| {
            !PARSER_FILES.contains(&rel.as_str())
                && !UNSAFE_FREE_KERNELS.contains(&rel.as_str())
                && !INVENTORY_OWNED.contains(&rel.as_str())
        })
        .collect();
    assert!(
        unclassified.is_empty(),
        "sources under {CLASSIFIED_ROOT} are not classified in \
         tests/parser_unsafe_gate.rs (P4-141 criterion 5). Add each to \
         PARSER_FILES (parser / control plane: must be unsafe-free), \
         UNSAFE_FREE_KERNELS (scalar kernel, unsafe-free today), or \
         INVENTORY_OWNED (holds kernel-dispatch unsafe; criterion 4):\n\n  {}\n",
        unclassified.join("\n  ")
    );

    for relative_path in INVENTORY_OWNED {
        assert!(
            repo_root().join(relative_path).is_file(),
            "{relative_path} is listed in INVENTORY_OWNED but does not exist; \
             if it was renamed or removed, update the list"
        );
    }
}

/// The scanner itself must be discriminating in both directions: a
/// whole-word `unsafe` in code is caught wherever a literal or comment tries
/// to hide it, and prose, lint names and identifiers are not. Without this,
/// an over-eager or under-eager lexer would make the gate above vacuous.
#[test]
fn scanner_distinguishes_code_from_comments_and_literals() {
    let hits = |text: &str| unsafe_token_lines(text);

    // Code, in the shapes the codebase uses.
    assert_eq!(hits("    unsafe { *p = 1 }"), vec![1]);
    assert_eq!(hits("pub unsafe fn f() {} // ok"), vec![1]);
    assert_eq!(hits("let x = unsafe{g()};"), vec![1]);
    assert_eq!(hits("let a = 1;\nlet b = 2;\nunsafe { h() }\n"), vec![3]);

    // Literals must not hide code that follows them.
    assert_eq!(hits(r#"let s = "a//b"; unsafe { g() }"#), vec![1]);
    assert_eq!(hits(r#"let s = "\"//"; unsafe { g() }"#), vec![1]);
    assert_eq!(hits(r#"let s = "\\"; unsafe { g() }"#), vec![1]);
    assert_eq!(
        hits("let s = \"line one\nline two\"; unsafe { g() }"),
        vec![2]
    );
    assert_eq!(hits(r##"let s = r#"a "//" b"#; unsafe { g() }"##), vec![1]);
    assert_eq!(hits(r#"let c = '"'; unsafe { g() }"#), vec![1]);
    assert_eq!(hits(r#"let c = '\''; unsafe { g() }"#), vec![1]);
    assert_eq!(hits("fn f<'a>(x: &'a T) { unsafe { g() } }"), vec![1]);
    assert_eq!(hits("/* a */ unsafe { g() }"), vec![1]);
    assert_eq!(hits("/* a\n b */ unsafe { g() }"), vec![2]);

    // Prose, lint names and identifiers are not code `unsafe`.
    assert!(hits("// the unsafe path is gone").is_empty());
    assert!(hits("    /// no `unsafe` here").is_empty());
    assert!(hits("#![deny(unsafe_op_in_unsafe_fn)]").is_empty());
    assert!(hits("let unsafe_count = 0;").is_empty());
    assert!(hits("let n = 1; // was unsafe once").is_empty());
    assert!(hits(r#"let s = "a//b"; // unsafe"#).is_empty());
    assert!(hits(r#"let s = "unsafe";"#).is_empty());
    assert!(hits(r#"let s = "multi\nline unsafe\n";"#).is_empty());
    assert!(hits(r##"let s = r#"unsafe { g() }"#;"##).is_empty());
    assert!(hits("/* unsafe { g() } */").is_empty());
    assert!(hits("/* outer /* unsafe */ still comment */ let x = 1;").is_empty());
    assert!(hits("/* unsafe\n unsafe\n */").is_empty());
}

/// The file-reading half must also be exercised on a file that *has* sites,
/// or a broken read/report path would leave the gate above green and
/// meaningless. A throwaway fixture with known contents pins both the
/// non-empty result and the reported line numbers.
#[test]
fn scanner_reads_files_and_reports_line_numbers() {
    if !repository_tree_is_readable() {
        eprintln!("{}", skip_reason());
        return;
    }
    let fixture: PathBuf =
        std::env::temp_dir().join(format!("parser_unsafe_gate_{}.rs", std::process::id()));
    std::fs::write(
        &fixture,
        "// prose about unsafe\nfn f() {}\nunsafe fn g() {}\nlet s = \"unsafe\";\nfn h() { unsafe { g() } }\n",
    )
    .expect("write fixture");
    let sites: Vec<String> = unsafe_sites(&fixture);
    let _ = std::fs::remove_file(&fixture);
    assert_eq!(
        sites,
        vec![
            "3: unsafe fn g() {}".to_string(),
            "5: fn h() { unsafe { g() } }".to_string()
        ],
        "read/report path must list exactly the code sites with their lines"
    );
}
