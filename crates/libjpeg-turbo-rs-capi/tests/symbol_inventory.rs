//! P2-5: symbol-inventory diff against upstream.
//!
//! Parses `references/libjpeg-turbo/src/jpeglib.h` for `EXTERN(...)`
//! declarations and `references/libjpeg-turbo/src/turbojpeg.h` for
//! `DLLEXPORT` declarations to extract the canonical upstream symbol
//! list, then dlopens our cdylib and asserts each upstream-declared
//! symbol is resolvable. The set of *intentionally unsupported*
//! symbols is enumerated explicitly so a future addition that fills
//! one in upgrades the test from "expected gap" to "validated symbol"
//! by deletion from the allowlist.
//!
//! This is the structural complement to `tests/capi_stock_tool_link.rs::
//! shim_exports_classic_jpeg_api` (which only counts how many `jpeg_*`
//! symbols we export — useful for catching catastrophic regressions
//! but not for proving drop-in completeness).

use std::collections::HashSet;
use std::path::PathBuf;

use libloading::Library;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn dylib_ext() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

fn lib_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}

fn cdylib_filename() -> String {
    format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dylib_ext())
}

fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let filename: String = cdylib_filename();
    let mut dir: PathBuf = exe;
    while dir.pop() {
        let candidate: PathBuf = dir.join(&filename);
        if candidate.exists() {
            return candidate;
        }
    }

    let workspace = workspace_root();
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "libjpeg-turbo-rs-capi", "--release"])
        .current_dir(&workspace)
        .status()
        .expect("cargo build");
    assert!(status.success(), "cargo build failed");
    for rel_dir in ["target/release/deps", "target/release"] {
        let candidate: PathBuf = workspace.join(rel_dir).join(&filename);
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("cdylib not found after build");
}

/// Parse upstream `jpeglib.h` for `EXTERN(...)` declarations and pull
/// out the function names. Walks the source line-by-line: each line
/// starting with `EXTERN(` has the function name on the same line
/// after the closing `)` and a space.
fn extract_jpeg_symbols(jpeglib_h: &str) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    for line in jpeglib_h.lines() {
        let trimmed: &str = line.trim_start();
        if !trimmed.starts_with("EXTERN(") {
            continue;
        }
        // Find the matching `)` for the EXTERN( type.
        let after_extern: &str = &trimmed["EXTERN(".len()..];
        let mut depth: i32 = 1;
        let mut close_idx: Option<usize> = None;
        for (i, ch) in after_extern.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close_idx = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close_idx else {
            continue;
        };
        // After the `)` and a space (or sometimes whitespace), comes
        // the function name, then `(`.
        let after_paren: &str = &after_extern[close + 1..];
        let after_paren: &str = after_paren.trim_start();
        let name_end: usize = after_paren
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(after_paren.len());
        if name_end > 0 {
            let name: &str = &after_paren[..name_end];
            if !name.is_empty() {
                out.insert(name.to_string());
            }
        }
    }
    out
}

/// Parse upstream `turbojpeg.h` for `DLLEXPORT` declarations.
/// `DLLEXPORT <return type> <name>(...)`.
fn extract_tj_symbols(turbojpeg_h: &str) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    for line in turbojpeg_h.lines() {
        let trimmed: &str = line.trim_start();
        if !trimmed.starts_with("DLLEXPORT") {
            continue;
        }
        // After `DLLEXPORT`, skip the return type. Find the FIRST `(`,
        // then walk back from there to find the start of the function
        // name.
        let Some(open_paren_idx) = trimmed.find('(') else {
            continue;
        };
        let head: &str = &trimmed[..open_paren_idx];
        // The function name is the last whitespace-delimited token.
        let name: &str = head.split_whitespace().last().unwrap_or("");
        // Strip leading `*` (return-type pointer markers like `char *`).
        let name: &str = name.trim_start_matches('*');
        if name.starts_with("tj")
            && name
                .chars()
                .all(|c: char| c.is_ascii_alphanumeric() || c == '_')
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// Symbols that we deliberately do not yet expose. Each entry has a
/// rationale; deleting an entry is the test's signal that the gap
/// closed (and a future regression would re-trip the test).
///
/// The legacy `tj*` aliases below are the **deprecated** upstream
/// TurboJPEG 1.x/2.x ABI. Upstream still ships them as forwarding
/// wrappers around the v2/v3 forms because old downstream code links
/// to them; we have not yet ported the wrappers because:
/// (a) every TJ3 successor is implemented (tj3Compress8, tj3Decompress8,
///     tj3DecompressHeader, tj3GetErrorStr, tj3GetErrorCode,
///     tj3GetScalingFactors, tj3Alloc, tj3Free), so a caller that wants
///     the modern API gets it,
/// (b) `crates/libjpeg-turbo-rs-capi/src/legacy.rs` already exports
///     the v2/v3 variants (tjCompress2, tjDecompress2,
///     tjDecompressHeader3, tjEncodeYUV3, tjDecodeYUV, tjGetErrorStr2)
///     for the most-recent legacy generation,
/// (c) closing each row is a thin forwarding wrapper; they can be
///     filled one at a time as real downstream code surfaces a need.
///
/// Removing a name below is the contract: it means "this is now
/// implemented; the test should hold us to it from this commit on."
fn allowlisted_missing_symbols() -> HashSet<&'static str> {
    [
        // Classic libjpeg API: (none currently allowlisted — P3-3
        // emptied this list 2026-05-06).
        //
        // Legacy TurboJPEG 1.x/2.x ABI — superseded by the TJ3 forms
        // and the `*2` / `*3` variants we already export. The 18
        // entries below are tracked under P4-18 (filed 2026-05-18
        // after a cold review surfaced them as a blocker for
        // downstream C/C++ consumers compiled against TurboJPEG
        // 1.x/2.x headers); see `docs/last_mile/phase4.md` for the
        // implement-vs-deprecate decision matrix.
        "tjAlloc",                 // → tj3Alloc
        "tjFree",                  // → tj3Free
        "tjCompress",              // → tjCompress2 → tj3Compress8
        "tjCompressFromYUV",       // → tj3CompressFromYUV8
        "tjCompressFromYUVPlanes", // → tj3CompressFromYUVPlanes8
        "tjDecodeYUVPlanes",       // → tj3DecodeYUVPlanes8
        "tjDecompress",            // → tjDecompress2 → tj3Decompress8
        "tjDecompressHeader",      // → tjDecompressHeader3 → tj3DecompressHeader
        "tjDecompressHeader2",     // → tjDecompressHeader3 → tj3DecompressHeader
        "tjDecompressToYUV",       // → tj3DecompressToYUV8
        "tjDecompressToYUV2",      // → tj3DecompressToYUV8
        "tjDecompressToYUVPlanes", // → tj3DecompressToYUVPlanes8
        "tjEncodeYUV",             // → tjEncodeYUV3 → tj3EncodeYUV8
        "tjEncodeYUV2",            // → tjEncodeYUV3 → tj3EncodeYUV8
        "tjEncodeYUVPlanes",       // → tj3EncodeYUVPlanes8
        "tjGetErrorCode",          // → tj3GetErrorCode
        "tjGetErrorStr",           // → tj3GetErrorStr (no-handle form)
        "tjGetScalingFactors",     // → tj3GetScalingFactors
    ]
    .into_iter()
    .collect()
}

#[test]
fn cdylib_exports_every_upstream_jpeglib_h_symbol() {
    let workspace = workspace_root();
    let jpeglib_h_path = workspace.join("references/libjpeg-turbo/src/jpeglib.h");
    if !jpeglib_h_path.exists() {
        eprintln!(
            "SKIP: upstream jpeglib.h not found at {:?} (submodule not initialized?)",
            jpeglib_h_path
        );
        return;
    }
    let jpeglib_h_text: String = std::fs::read_to_string(&jpeglib_h_path).expect("read jpeglib.h");
    let upstream_symbols: HashSet<String> = extract_jpeg_symbols(&jpeglib_h_text);
    assert!(
        upstream_symbols.len() >= 30,
        "Parser regressed: only {} upstream jpeg_* symbols extracted from jpeglib.h. \
         Expected ≥ 30; was the EXTERN() macro renamed upstream?",
        upstream_symbols.len()
    );
    eprintln!(
        "Parsed {} upstream jpeg_* symbol declarations from jpeglib.h",
        upstream_symbols.len()
    );

    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    let allowlist: HashSet<&'static str> = allowlisted_missing_symbols();
    let mut missing: Vec<String> = Vec::new();
    for sym in &upstream_symbols {
        if allowlist.contains(sym.as_str()) {
            continue;
        }
        let resolved: bool = unsafe {
            // libloading caches errors; calling .get() and discarding
            // is the canonical "does it resolve?" check.
            lib.get::<unsafe extern "C" fn()>(sym.as_bytes()).is_ok()
        };
        if !resolved {
            missing.push(sym.clone());
        }
    }
    missing.sort();

    assert!(
        missing.is_empty(),
        "Drop-in completeness gap: {} jpeg_* symbol(s) declared by upstream \
         jpeglib.h but not exported by our cdylib. Either implement them or \
         add them to `allowlisted_missing_symbols()` with a rationale.\n\
         Missing: {:#?}",
        missing.len(),
        missing
    );
}

#[test]
fn cdylib_exports_every_upstream_turbojpeg_h_symbol() {
    let workspace = workspace_root();
    let turbojpeg_h_path = workspace.join("references/libjpeg-turbo/src/turbojpeg.h");
    if !turbojpeg_h_path.exists() {
        eprintln!(
            "SKIP: upstream turbojpeg.h not found at {:?}",
            turbojpeg_h_path
        );
        return;
    }
    let turbojpeg_h_text: String =
        std::fs::read_to_string(&turbojpeg_h_path).expect("read turbojpeg.h");
    let upstream_symbols: HashSet<String> = extract_tj_symbols(&turbojpeg_h_text);
    assert!(
        upstream_symbols.len() >= 30,
        "Parser regressed: only {} upstream tj* symbols extracted from turbojpeg.h",
        upstream_symbols.len()
    );
    eprintln!(
        "Parsed {} upstream tj* symbol declarations from turbojpeg.h",
        upstream_symbols.len()
    );

    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };

    let allowlist: HashSet<&'static str> = allowlisted_missing_symbols();
    let mut missing: Vec<String> = Vec::new();
    for sym in &upstream_symbols {
        if allowlist.contains(sym.as_str()) {
            continue;
        }
        let resolved: bool = unsafe { lib.get::<unsafe extern "C" fn()>(sym.as_bytes()).is_ok() };
        if !resolved {
            missing.push(sym.clone());
        }
    }
    missing.sort();

    assert!(
        missing.is_empty(),
        "Drop-in completeness gap: {} tj* symbol(s) declared by upstream \
         turbojpeg.h but not exported by our cdylib. Either implement them \
         or add them to `allowlisted_missing_symbols()` with a rationale.\n\
         Missing: {:#?}",
        missing.len(),
        missing
    );
}
