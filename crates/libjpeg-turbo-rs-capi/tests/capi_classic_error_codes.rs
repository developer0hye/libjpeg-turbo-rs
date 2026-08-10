//! P4-100: every `JERR_*` code the classic shim emits, checked against the
//! pinned upstream headers.
//!
//! A consumer compares `cinfo->err->msg_code` against the enum from *its own*
//! `jerror.h`, and `format_message` indexes the message table with it. So a
//! constant that is one off does not fail loudly — it silently reports a
//! different error than the one that happened, which is the failure mode
//! `JERR_OUT_OF_MEMORY` sat in until now (documented as unpinned in
//! `jpeglib.rs`, and the reason P4-120 exists).
//!
//! Deriving the values is not enough: `jerror.h`'s enum is *positional*, built
//! from a `JMESSAGE` list with version-gated entries, so an off-by-one is easy
//! to introduce and easy to miss. Two hand-written derivations of this list
//! disagreed by one during P4-100's own development. This test therefore asks
//! the C compiler, which is the only authority that cannot be argued with.
//!
//! It also asserts the *message text*, because a correct number attached to
//! the wrong message would still mislead every consumer that formats it.

use std::ffi::c_int;
use std::path::PathBuf;
use std::process::Command;

/// Every code the shim defines, with the upstream message it must resolve to.
///
/// Keep this in step with the `const JERR_*` block in
/// `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`; the completeness check below
/// fails if the shim grows one this table does not cover.
const EXPECTED: &[(&str, i32, &str)] = &[
    (
        "JERR_BAD_DCT_COEF",
        6,
        "DCT coefficient (lossy) or spatial difference (lossless) out of range",
    ),
    ("JERR_BAD_LENGTH", 12, "Bogus marker length"),
    (
        "JERR_BAD_PRECISION",
        16,
        "Unsupported JPEG data precision %d",
    ),
    (
        "JERR_BAD_STATE",
        21,
        "Improper call to JPEG library in state %d",
    ),
    (
        "JERR_BUFFER_SIZE",
        24,
        "Buffer passed to JPEG library is too small",
    ),
    ("JERR_CANT_SUSPEND", 25, "Suspension not allowed here"),
    ("JERR_FILE_READ", 37, "Input file read error"),
    (
        "JERR_FILE_WRITE",
        38,
        "Output file write error --- out of disk space?",
    ),
    ("JERR_INPUT_EMPTY", 43, "Empty input file"),
    (
        "JERR_IMAGE_TOO_BIG",
        42,
        "Maximum supported image dimension is %u pixels",
    ),
    ("JERR_INPUT_EOF", 44, "Premature end of input file"),
    ("JERR_NOTIMPL", 48, "Requested features are incompatible"),
    ("JERR_NO_IMAGE", 53, "JPEG datastream contains no image"),
    ("JERR_OUT_OF_MEMORY", 56, "Insufficient memory (case %d)"),
    (
        "JERR_TOO_LITTLE_DATA",
        69,
        "Application transferred too few scanlines",
    ),
    ("JERR_UNKNOWN_MARKER", 70, "Unsupported marker type 0x%02x"),
    // Added with P4-14: what upstream raises when `max_memory_to_use` cannot
    // cover the virtual arrays. NOT `JERR_OUT_OF_MEMORY` — the budget is
    // consulted by `jpeg_mem_available` (`jmemnobs.c:66-78`) and the spill that
    // follows hits `jmemnobs.c:87-92`. Cross-checked here because the issue
    // that requested this enforcement named the wrong code *and* the wrong
    // number for it.
    ("JERR_NO_BACKING_STORE", 51, "Memory limit exceeded"),
    // Added with P4-139: upstream raises this — not `JERR_IMAGE_TOO_BIG` — when
    // `image_width * input_components` is not representable as `JDIMENSION`
    // (`jcmaster.c:190-194`). The two are different failures and the messages
    // differ in arity, so conflating them emits a `%u` slot with nothing in it.
    (
        "JERR_WIDTH_OVERFLOW",
        72,
        "Image too wide for this implementation",
    ),
];

fn find_cc() -> Option<PathBuf> {
    if let Ok(cc) = std::env::var("CC") {
        if !cc.is_empty() {
            return Some(PathBuf::from(cc));
        }
    }
    for candidate in ["cc", "clang", "gcc"] {
        if let Ok(out) = Command::new("which").arg(candidate).output() {
            if out.status.success() {
                let path: String = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
            }
        }
    }
    None
}

fn upstream_src_dir() -> Option<PathBuf> {
    let dir: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/libjpeg-turbo/src");
    dir.join("jerror.h").exists().then(|| {
        dir.canonicalize()
            .expect("canonicalize the upstream src dir")
    })
}

const JCONFIG_H: &str = "\
#define JPEG_LIB_VERSION 80
#define LIBJPEG_TURBO_VERSION 3.1.0
#define LIBJPEG_TURBO_VERSION_NUMBER 3001000
#define C_ARITH_CODING_SUPPORTED 1
#define D_ARITH_CODING_SUPPORTED 1
#define MEM_SRCDST_SUPPORTED 1
#define WITH_SIMD 1
#define BITS_IN_JSAMPLE 8
";

fn is_ci() -> bool {
    std::env::var("CI")
        .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

/// Ask the C compiler for each code's numeric value and message text.
fn upstream_codes() -> Option<Vec<(String, i32, String)>> {
    let cc: PathBuf = find_cc()?;
    let src_dir: PathBuf = upstream_src_dir()?;
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("jconfig.h"), JCONFIG_H).expect("write jconfig.h");

    // `jerror.h` builds both the enum and, with a redefined JMESSAGE, the
    // message table — so one translation unit can print name, value and text
    // together and nothing has to be transcribed by hand.
    // `jerror.h`'s message list ends with JMSG_COPYRIGHT, whose text comes from
    // the generated `jversion.h`. The submodule ships only `jversion.h.in`, so
    // supply the macro directly rather than running upstream's CMake configure
    // for one string that this test never inspects.
    let mut program: String = String::from(
        "#include <stdio.h>\n\
         #define JCOPYRIGHT \"unused by this probe\"\n\
         #define JVERSION \"unused by this probe\"\n\
         #include \"jpeglib.h\"\n\
         #include \"jerror.h\"\n\
         #define REPORT(sym) printf(\"%s\\t%d\\t%s\\n\", #sym, (int)(sym), messages[(int)(sym)]);\n\
         static const char * const messages[] = {\n\
         #define JMESSAGE(code, string) string,\n\
         #include \"jerror.h\"\n\
         };\n\
         int main(void) {\n",
    );
    for (name, _, _) in EXPECTED {
        program.push_str(&format!("  REPORT({name})\n"));
    }
    program.push_str("  return 0;\n}\n");

    let src: PathBuf = tmp.path().join("jerr_codes.c");
    std::fs::write(&src, program).expect("write probe source");
    let bin: PathBuf = tmp.path().join("jerr_codes");
    let compile = Command::new(&cc)
        .arg("-O0")
        .arg("-I")
        .arg(tmp.path())
        .arg("-I")
        .arg(&src_dir)
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .output()
        .expect("invoke cc");
    assert!(
        compile.status.success(),
        "the JERR probe failed to compile against the pinned headers:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin).output().expect("run the JERR probe");
    assert!(run.status.success(), "the JERR probe exited non-zero");
    let stdout: String = String::from_utf8_lossy(&run.stdout).to_string();
    Some(
        stdout
            .lines()
            .filter_map(|line| {
                let mut parts = line.splitn(3, '\t');
                let name: String = parts.next()?.to_string();
                let value: i32 = parts.next()?.parse().ok()?;
                let message: String = parts.next()?.to_string();
                Some((name, value, message))
            })
            .collect(),
    )
}

/// Every constant the shim emits must equal the value the pinned upstream
/// headers give it, and carry the message a consumer will format.
#[test]
fn classic_error_codes_match_upstream() {
    let observed: Vec<(String, i32, String)> = match upstream_codes() {
        Some(codes) => codes,
        None => {
            assert!(
                !is_ci(),
                "CI checks out submodules and provides a C compiler, so this \
                 cross-check must run there"
            );
            eprintln!(
                "SKIP: no C compiler or the libjpeg-turbo submodule is not checked out; \
                 cannot ask upstream for its JERR_* values"
            );
            return;
        }
    };

    assert_eq!(
        observed.len(),
        EXPECTED.len(),
        "the probe reported {} codes but {} were requested",
        observed.len(),
        EXPECTED.len()
    );

    for ((want_name, want_code, want_msg), (got_name, got_code, got_msg)) in
        EXPECTED.iter().zip(observed.iter())
    {
        assert_eq!(want_name, got_name, "probe output is out of order");
        assert_eq!(
            *want_code, *got_code,
            "{want_name}: the shim uses {want_code}, upstream defines {got_code} — \
             every consumer comparing msg_code would see the wrong error"
        );
        assert_eq!(
            *want_msg, *got_msg,
            "{want_name}: the shim documents a different message than upstream's, \
             so `format_message` would render something the code does not mean"
        );
    }

    eprintln!(
        "classic_error_codes_match_upstream: {} codes verified against the pinned v8 headers",
        EXPECTED.len()
    );
}

/// The table above must cover every `JERR_*` the shim defines. Without this a
/// newly added constant is unverified, which is exactly how `JERR_OUT_OF_MEMORY`
/// stayed unpinned.
///
/// Scans **every** shim module that defines error constants, not just
/// `jpeglib.rs`. P4-14 added `JERR_NO_BACKING_STORE` and `JERR_OUT_OF_MEMORY`
/// to `memmgr.rs`, where a single-file scan would not have seen them — and
/// `JERR_OUT_OF_MEMORY` would have looked covered because `jpeglib.rs` happens
/// to define a constant by the same name. "Every shim error constant" has to
/// mean every file, or the claim is only as good as the file list.
#[test]
fn every_shim_error_constant_is_covered() {
    const SCANNED_MODULES: [&str; 2] = ["src/jpeglib.rs", "src/memmgr.rs"];

    let mut defined: Vec<String> = Vec::new();
    for module in SCANNED_MODULES {
        let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(module);
        let source: String =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {module}: {e}"));
        // Parse the *value* too. Comparing names alone would let
        // `memmgr.rs`'s `JERR_NO_BACKING_STORE` drift from 51 to anything
        // while both tests stayed green — `EXPECTED` is checked against
        // upstream independently, so a name match proves nothing about the
        // number the shim actually raises.
        let found: Vec<(String, c_int)> = source
            .lines()
            .filter_map(|line| {
                let rest: &str = line.trim().strip_prefix("const JERR_")?;
                let (name, tail) = rest.split_once(':')?;
                let value: c_int = tail
                    .split('=')
                    .nth(1)?
                    .trim()
                    .trim_end_matches(';')
                    .trim()
                    .parse()
                    .ok()?;
                Some((format!("JERR_{name}"), value))
            })
            .collect();
        for (name, value) in &found {
            if let Some((_, expected, _)) = EXPECTED.iter().find(|(n, _, _)| n == name) {
                assert_eq!(
                    value, expected,
                    "{module} defines {name} = {value}, but the upstream-verified \
                     table says {expected}"
                );
            }
        }
        let found: Vec<String> = found.into_iter().map(|(name, _)| name).collect();
        assert!(
            !found.is_empty(),
            "found no `const JERR_*` in {module} — this test's parser has drifted, \
             or the constants moved and this list needs updating"
        );
        defined.extend(found);
    }
    defined.sort();
    defined.dedup();

    let covered: Vec<&str> = EXPECTED.iter().map(|(name, _, _)| *name).collect();
    let missing: Vec<&String> = defined
        .iter()
        .filter(|name| !covered.contains(&name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these shim error constants are not cross-checked against upstream: {missing:?}"
    );
}
