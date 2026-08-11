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

use libjpeg_turbo_rs_capi::jpeglib::JpegErrorMgr;
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
        "JERR_BAD_LIB_VERSION",
        13,
        "Wrong JPEG library version: library is %d, caller expects %d",
    ),
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
        "JERR_BAD_STRUCT_SIZE",
        22,
        "JPEG parameter struct mismatch: library thinks size is %u, caller expects %u",
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

/// A `j_common_ptr`-shaped stub with the alignment the cast requires.
///
/// A `Vec<u8>` was used here first, which is byte-aligned: casting its data
/// pointer to `*mut *mut JpegErrorMgr` and writing through it is undefined
/// behaviour even when the system allocator happens to over-align, and Miri
/// rejects it. `#[repr(C)]` with `err` first gives the real layout the
/// callbacks read.
#[repr(C)]
struct CommonStub {
    err: *mut JpegErrorMgr,
    /// libjpeg's `j_common_ptr` continues past `err`; nothing under test
    /// reads it, but the padding keeps a stray offset read inside our own
    /// allocation rather than off the end of a one-pointer struct.
    tail: [usize; 127],
}

impl CommonStub {
    fn new(err: *mut JpegErrorMgr) -> Self {
        Self {
            err,
            tail: [0usize; 127],
        }
    }
}

/// Resolve `JMSG_COPYRIGHT` (75) and `JMSG_VERSION` (76) from the submodule's
/// templates, since `jversion.h` itself is generated at configure time.
///
/// `JVERSION` comes from `jversion.h.in`'s `JPEG_LIB_VERSION >= 80` arm;
/// `JCOPYRIGHT` from the same file with `@COPYRIGHT_YEAR@` substituted from
/// `CMakeLists.txt`. Parsed rather than hardcoded so a submodule bump that
/// changes either one fails here instead of silently disagreeing with the
/// table.
fn configured_version_message(code: usize) -> String {
    let root: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/libjpeg-turbo");
    let template: String =
        std::fs::read_to_string(root.join("src/jversion.h.in")).expect("read jversion.h.in");
    let cmake: String =
        std::fs::read_to_string(root.join("CMakeLists.txt")).expect("read CMakeLists.txt");

    let year: String = cmake
        .lines()
        .find_map(|l| {
            let rest = l.trim().strip_prefix("set(COPYRIGHT_YEAR")?;
            Some(
                rest.trim()
                    .trim_matches(|c| c == '"' || c == ')')
                    .to_string(),
            )
        })
        .expect("COPYRIGHT_YEAR in CMakeLists.txt");

    if code == 76 {
        // The first JVERSION after `#if JPEG_LIB_VERSION >= 80`.
        let after: &str = template
            .split_once("#if JPEG_LIB_VERSION >= 80")
            .expect("v8 arm in jversion.h.in")
            .1;
        let line: &str = after
            .lines()
            .find(|l| l.trim_start().starts_with("#define JVERSION"))
            .expect("JVERSION define");
        line.split('"').nth(1).expect("JVERSION string").to_string()
    } else {
        let raw: String = template
            .split_once("#define JCOPYRIGHT")
            .expect("JCOPYRIGHT define")
            .1
            .split('"')
            .nth(1)
            .expect("JCOPYRIGHT string")
            .to_string();
        raw.replace("@COPYRIGHT_YEAR@", &year)
    }
}

/// Read the table the shim installs, exactly as a C consumer would: through
/// `err->jpeg_message_table`, indexed by code.
fn installed_message_table() -> Vec<String> {
    use libjpeg_turbo_rs_capi::jpeglib::jpeg_std_error;

    let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
    // SAFETY: `jerr` is a live, correctly-aligned error manager owned here.
    let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut jerr as *mut JpegErrorMgr) };
    assert!(!errp.is_null(), "jpeg_std_error");

    // SAFETY: `jpeg_std_error` installs a table of `last_jpeg_message + 1`
    // `'static` NUL-terminated strings, which is what is walked here.
    unsafe {
        let table: *const *const u8 = (*errp).jpeg_message_table;
        assert!(
            !table.is_null(),
            "jpeg_std_error must install a message table"
        );
        let count: usize = ((*errp).last_jpeg_message + 1) as usize;
        (0..count)
            .map(|i| {
                let entry: *const u8 = table.add(i).read();
                assert!(!entry.is_null(), "message table entry {i} is null");
                let mut len: usize = 0;
                while entry.add(len).read() != 0 {
                    len += 1;
                }
                String::from_utf8_lossy(std::slice::from_raw_parts(entry, len)).into_owned()
            })
            .collect()
    }
}

/// Render `code` the way a C consumer does: `msg_code` plus
/// `err->format_message`. No parameter is supplied, so parameterised messages
/// come back with their `%…` spec unfilled — callers compare the prefix.
fn render_through_shim(code: c_int) -> String {
    use libjpeg_turbo_rs_capi::jpeglib::jpeg_std_error;
    const JMSG_LENGTH_MAX: usize = 200;

    let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
    // SAFETY: `jerr` is a live, correctly-aligned error manager owned here.
    let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut jerr as *mut JpegErrorMgr) };
    assert!(!errp.is_null(), "jpeg_std_error");

    let mut cinfo: CommonStub = CommonStub::new(errp);
    let mut buf: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];
    // SAFETY: `cinfo`'s first pointer-sized field is the `err` slot, which is
    // all `format_message` reads; `buf` is the contract's required size.
    unsafe {
        (*errp).msg_code = code;
        let format = (*errp).format_message.expect("format_message installed");
        format(
            &mut cinfo as *mut CommonStub as *mut std::ffi::c_void,
            buf.as_mut_ptr(),
        );
    }
    let len: usize = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
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

        // ...and the same text must come out of *our formatter*, not just out
        // of `jerror.h`. Until P4-146 (#518) this file compared the header to a
        // table in this file and nothing else: `jpeg_message_table` was null,
        // every error rendered as "bogus message code", and this test reported
        // 18 codes "verified" throughout. Checking the header against a literal
        // is a parity check of the *constants*; only this line makes it one of
        // the *rendering*.
        //
        // Parameterised messages are compared up to their first `%`: supplying
        // each one's parameter is `capi_error_message_rendering.rs`'s job, and
        // duplicating it here would couple this test to argument shapes it does
        // not otherwise care about.
        let rendered: String = render_through_shim(*got_code);
        let prefix: &str = want_msg.split('%').next().unwrap_or(want_msg);
        assert!(
            rendered.starts_with(prefix),
            "{want_name} ({got_code}): upstream says {want_msg:?} but our \
             `format_message` rendered {rendered:?} — a C consumer's \
             `output_message` shows the latter"
        );
    }

    eprintln!(
        "classic_error_codes_match_upstream: {} codes verified against the pinned v8 headers",
        EXPECTED.len()
    );
}

/// The **whole** generated message table must match upstream's, entry for
/// entry — not just the 18 codes `EXPECTED` names.
///
/// This is what pins `message_table.rs` against a submodule bump. The table is
/// indexed by `msg_code`, so an entry that shifts does not go missing: every
/// message after it becomes a *different, plausible* message. Checking only the
/// codes we happen to enumerate would leave 111 of the 129 entries free to
/// drift silently.
///
/// It also re-derives the table the same way `message_table.rs` was generated —
/// by compiling `jerror.h` twice at `JPEG_LIB_VERSION 80` — so the count itself
/// is verified. A line-order parse of that header sees 134 `JMESSAGE` lines
/// because several are version-conditional and a few appear twice under
/// opposite guards; 129 is the number that survives the preprocessor.
#[test]
fn the_whole_message_table_matches_upstream() {
    let observed: Vec<String> = match upstream_message_table() {
        Some(t) => t,
        None => {
            assert!(
                !is_ci(),
                "CI checks out submodules and provides a C compiler, so this \
                 cross-check must run there"
            );
            eprintln!(
                "SKIP: no C compiler or the libjpeg-turbo submodule is not checked out; \
                 cannot ask upstream for its message table"
            );
            return;
        }
    };

    assert_eq!(
        observed.len(),
        129,
        "upstream's v8 table has 129 entries; the probe saw {}. If this changed, \
         `message_table.rs` must be regenerated, not adjusted by hand",
        observed.len()
    );

    // Compare the *raw format strings*, not rendered output. Rendering and
    // then comparing the text before the first `%` would let `%u pixels`
    // become `%d bananas` — the argument shape and the entire suffix would go
    // unchecked, which is most of what can drift.
    //
    // The strings are read back out of the installed table through the public
    // `jpeg_message_table` field, which is exactly the pointer a C consumer
    // indexes.
    let installed: Vec<String> = installed_message_table();
    assert_eq!(
        installed.len(),
        observed.len(),
        "installed table has {} entries, upstream has {}",
        installed.len(),
        observed.len()
    );

    for (code, (want, got)) in observed.iter().zip(installed.iter()).enumerate() {
        // 75/76 come from `jversion.h`, which the submodule ships only as
        // `jversion.h.in`; the probe substitutes placeholders for them, so it
        // cannot be the oracle here. `message_table.rs` documents where their
        // real values come from.
        if code == 75 || code == 76 {
            // `jversion.h` is generated, so the probe cannot be the oracle:
            // the submodule ships only `jversion.h.in`. Resolve the two
            // values the same way CMake does and compare exactly — accepting
            // "anything without the word probe" would let a stale version
            // string pass a test that claims all 129 entries are pinned.
            let want: String = configured_version_message(code);
            assert_eq!(
                *got, want,
                "index {code} must match the configured value from \
                 jversion.h.in / CMakeLists.txt"
            );
            continue;
        }
        assert_eq!(
            want, got,
            "message table diverges at index {code}. A shifted entry means every \
             later code renders someone else's message"
        );
    }

    eprintln!(
        "the_whole_message_table_matches_upstream: {} entries verified verbatim \
         (75/76 against jversion.h.in + CMakeLists.txt)",
        observed.len()
    );
}

/// Dump upstream's complete message table, in `msg_code` order.
fn upstream_message_table() -> Option<Vec<String>> {
    let cc: PathBuf = find_cc()?;
    let src_dir: PathBuf = upstream_src_dir()?;
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("jconfig.h"), JCONFIG_H).expect("write jconfig.h");

    let program: &str = "#include <stdio.h>\n\
         #define JCOPYRIGHT \"unused by this probe\"\n\
         #define JVERSION \"unused by this probe\"\n\
         #include \"jpeglib.h\"\n\
         #include \"jerror.h\"\n\
         static const char * const messages[] = {\n\
         #define JMESSAGE(code, string) string,\n\
         #include \"jerror.h\"\n\
         };\n\
         int main(void) {\n\
           int n = (int)(sizeof(messages) / sizeof(messages[0]));\n\
           for (int i = 0; i < n; i++) printf(\"%s\\n\", messages[i]);\n\
           return 0;\n\
         }\n";

    let src: PathBuf = tmp.path().join("jerr_table.c");
    std::fs::write(&src, program).expect("write probe source");
    let bin: PathBuf = tmp.path().join("jerr_table");
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
        "the message-table probe failed to compile against the pinned headers:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin)
        .output()
        .expect("run the message-table probe");
    assert!(
        run.status.success(),
        "the message-table probe exited non-zero"
    );
    Some(
        String::from_utf8_lossy(&run.stdout)
            .lines()
            .map(|l| l.to_string())
            .collect(),
    )
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
