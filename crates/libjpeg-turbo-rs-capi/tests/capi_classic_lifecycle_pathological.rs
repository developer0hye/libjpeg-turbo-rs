//! P4-5 (phase-1): pathological classic libjpeg lifecycle patterns that
//! extend the eight P3-5 patterns in `capi_classic_lifecycle.rs`.
//!
//! Where `capi_classic_lifecycle.rs` exercises *valid* but uncommon
//! state-machine paths (custom source/destination managers, single
//! suspension events, abort+reuse), the patterns here are intentionally
//! adversarial:
//!
//!   1. `source_mgr_suspends_every_byte` — the custom `jpeg_source_mgr`
//!      hands the decoder one byte at a time, returning `FALSE` on every
//!      `fill_input_buffer` call so the decoder must walk through hundreds
//!      of suspension/resume transitions for a single image. The existing
//!      P3-5 suspension test only suspends once.
//!
//!   2. `dest_mgr_rejects_first_flush` — the custom
//!      `jpeg_destination_mgr` returns `FALSE` from
//!      `empty_output_buffer`. The upstream contract documents this as
//!      *fatal*: there is no suspended-output protocol on the compress
//!      side, so the shim must surface `JERR_CANT_SUSPEND` (msg_code 25)
//!      via the installed error manager rather than crash, hang, or
//!      silently corrupt the output stream. The C harness installs a
//!      setjmp/longjmp error_exit and asserts the documented error fires.
//!
//!   3. `save_markers_truncates_multichunk_icc` — an APP2 ICC profile
//!      spanning three chunks is saved with `length_limit = 1`, forcing
//!      the marker list to retain the marker header but only one byte of
//!      payload per chunk. Verifies the shim's `marker_list` truncation
//!      logic matches upstream behavior.
//!
//! Phase-2 (deferred) patterns: marker_processor that longjmps via
//! setjmp error_mgr, virt_barray reuse after `jpeg_abort_decompress`,
//! abbreviated stream re-read with cached prefix changing tables between
//! sessions.
//!
//! Each test compiles a small C harness against the submodule's
//! `references/libjpeg-turbo/src/jpeglib.h` and links it through symlinks
//! that name both v8 (P4-3 default) and v6b (legacy) SONAMEs.

use std::path::{Path, PathBuf};
use std::process::Command;

// ---------- shared helpers (mirror capi_classic_lifecycle.rs) ----------

fn dlext() -> &'static str {
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

fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf =
            dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dlext()));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!(
        "could not locate cdylib near {}; build with `cargo build -p libjpeg-turbo-rs-capi --release`",
        exe.display()
    );
}

fn find_cc() -> Option<PathBuf> {
    if let Ok(test_cc) = std::env::var("CAPI_TEST_CC") {
        if !test_cc.is_empty() {
            return Some(PathBuf::from(test_cc));
        }
    }
    // These harnesses must link natively against the Rust cdylib. A Conda
    // build-wide CC can target an older sysroot and reject current glibc
    // symbols, so prefer the host compiler unless the test-specific override
    // above is set.
    for candidate in [
        "/usr/bin/cc",
        "/usr/bin/clang",
        "/usr/local/bin/cc",
        "/opt/homebrew/opt/llvm/bin/clang",
    ] {
        let path: PathBuf = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(env_cc) = std::env::var("CC") {
        if !env_cc.is_empty() {
            return Some(PathBuf::from(env_cc));
        }
    }
    for candidate in ["cc", "clang", "gcc"] {
        if let Ok(out) = Command::new("which").arg(candidate).output() {
            if out.status.success() {
                let s: String = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return Some(PathBuf::from(s));
                }
            }
        }
    }
    None
}

fn is_ci() -> bool {
    std::env::var("CI")
        .map(|value: String| {
            let normalized: String = value.to_ascii_lowercase();
            !normalized.is_empty() && normalized != "0" && normalized != "false"
        })
        .unwrap_or(false)
}

fn find_c_tool(name: &str) -> Option<PathBuf> {
    for directory in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/libjpeg-turbo/bin",
    ] {
        let candidate: PathBuf = Path::new(directory).join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    std::env::var_os("PATH").and_then(|path: std::ffi::OsString| {
        std::env::split_paths(&path)
            .map(|directory: PathBuf| directory.join(name))
            .find(|candidate: &PathBuf| candidate.is_file())
    })
}

fn require_c_tool(name: &str, test_name: &str) -> Option<PathBuf> {
    if let Some(path) = find_c_tool(name) {
        return Some(path);
    }
    if is_ci() {
        panic!("{test_name}: required C oracle `{name}` was not found on PATH");
    }
    eprintln!("SKIP {test_name}: required C oracle `{name}` was not found on PATH");
    None
}

#[cfg(unix)]
fn compiler_tool_path(cc: &Path) -> std::ffi::OsString {
    let mut directories: Vec<PathBuf> = Vec::new();
    if let Some(parent) = cc.parent() {
        if !parent.as_os_str().is_empty() {
            directories.push(parent.to_path_buf());
        }
    }
    for directory in [PathBuf::from("/usr/bin"), PathBuf::from("/bin")] {
        if !directories.contains(&directory) {
            directories.push(directory);
        }
    }
    std::env::join_paths(directories).expect("construct compiler helper PATH")
}

#[cfg(unix)]
fn setup_symlinks(lib: &Path, parent: &Path) -> PathBuf {
    let subdir: PathBuf = parent.join("symlinks");
    std::fs::create_dir_all(&subdir).expect("mkdir symlinks");
    let names: &[&str] = if cfg!(target_os = "macos") {
        &["libjpeg.8.dylib", "libjpeg.62.dylib", "libjpeg.dylib"]
    } else {
        &["libjpeg.so.8", "libjpeg.so.62", "libjpeg.so"]
    };
    for name in names {
        let link = subdir.join(name);
        if !link.exists() {
            std::os::unix::fs::symlink(lib, &link).expect("symlink");
        }
    }
    subdir
}

#[cfg(not(unix))]
fn setup_symlinks(_lib: &Path, parent: &Path) -> PathBuf {
    parent.to_path_buf()
}

fn upstream_src_dir() -> Option<PathBuf> {
    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate: PathBuf = manifest_dir.join("../../references/libjpeg-turbo/src");
    if candidate.join("jpeglib.h").exists() {
        Some(candidate.canonicalize().expect("canonicalize upstream_src"))
    } else {
        None
    }
}

/// Produce a minimal jconfig.h matching the v8 ABI our cdylib advertises.
fn write_jconfig(dir: &Path) -> PathBuf {
    let path: PathBuf = dir.join("jconfig.h");
    let content: &str = "/* generated by capi_classic_lifecycle_pathological.rs */\n\
        #define JPEG_LIB_VERSION 80\n\
        #define LIBJPEG_TURBO_VERSION 3.1.0\n\
        #define LIBJPEG_TURBO_VERSION_NUMBER 3001000\n\
        #define BITS_IN_JSAMPLE 8\n\
        #define MEM_SRCDST_SUPPORTED 1\n\
        #define WITH_SIMD 1\n\
        #define C_ARITH_CODING_SUPPORTED 1\n\
        #define D_ARITH_CODING_SUPPORTED 1\n";
    std::fs::write(&path, content).expect("write jconfig");
    path
}

#[cfg(unix)]
fn compile_and_run_c(c_source: &str, test_name: &str) -> Option<i32> {
    let cc: PathBuf = match find_cc() {
        Some(p) => p,
        None => {
            if is_ci() {
                panic!("{test_name}: no C compiler on PATH");
            }
            eprintln!("SKIP {test_name}: no C compiler on PATH");
            return None;
        }
    };
    let upstream: PathBuf = match upstream_src_dir() {
        Some(p) => p,
        None => {
            if is_ci() {
                panic!(
                    "{test_name}: references/libjpeg-turbo/src missing; initialize the submodule"
                );
            }
            eprintln!(
                "SKIP {test_name}: references/libjpeg-turbo/src missing — run \
                 `git submodule update --init --depth 1 references/libjpeg-turbo`"
            );
            return None;
        }
    };

    let cdylib: PathBuf = cdylib_path();
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let symlink_dir: PathBuf = setup_symlinks(&cdylib, tmp.path());
    let jconfig_dir: &Path = tmp.path();
    write_jconfig(jconfig_dir);

    let src_path: PathBuf = tmp.path().join(format!("{test_name}.c"));
    std::fs::write(&src_path, c_source).expect("write C source");
    let exe: PathBuf = tmp.path().join(test_name);

    let mut cmd = Command::new(&cc);
    cmd.arg(&src_path)
        .arg("-O2")
        .arg(format!("-I{}", upstream.display()))
        .arg(format!("-I{}", jconfig_dir.display()))
        .arg("-o")
        .arg(&exe)
        .arg(format!("-L{}", symlink_dir.display()))
        .arg("-ljpeg")
        .arg(format!("-Wl,-rpath,{}", symlink_dir.display()))
        // C oracle discovery still honors the caller's PATH, but a compiler
        // must not accidentally pick Conda's incompatible `ld` ahead of the
        // system linker. Keep the compiler's own directory plus system tools.
        .env("PATH", compiler_tool_path(&cc));
    let compile = cmd.output().expect("cc");
    if !compile.status.success() {
        panic!(
            "{test_name}: C harness compilation failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr),
        );
    }

    let run = Command::new(&exe).output().expect("run harness");
    if !run.status.success() {
        eprintln!(
            "{test_name} harness FAILED (exit {:?}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            run.status.code(),
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
    Some(run.status.code().unwrap_or(-1))
}

#[cfg(not(unix))]
fn compile_and_run_c(_c_source: &str, test_name: &str) -> Option<i32> {
    eprintln!("SKIP {test_name}: pathological harness is unix-only");
    None
}

// ---------- pattern 1: source_mgr that suspends every byte ----------

const PATTERN_SOURCE_MGR_SUSPENDS_EVERY_BYTE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

/* 8x8 baseline RGB JPEG (created with cjpeg -quality 75; embedded as the
 * minimum 145-byte SOI..EOI stream that exercises the full state
 * machine).  We don't need a specific image — only that the decoder
 * works through SOI/DQT/SOF/DHT/SOS/EOI under aggressive suspension. */
static const unsigned char SAMPLE_JPEG[] = {
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J', 'F', 'I', 'F', 0x00,
    0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
    0xFF, 0xDB, 0x00, 0x43, 0x00,
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99,
    0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
    0xFF, 0xC4, 0x00, 0x1F, 0x00,
    0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0,
    0,1,2,3,4,5,6,7,8,9,10,11,
    0xFF, 0xC4, 0x00, 0xB5, 0x10,
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
    0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
    0xFC, 0xFF, 0xD9
};

/* Source manager that releases one byte per fill_input_buffer call, and
 * suspends (returns FALSE) on every refill. The decoder must walk the
 * suspension state machine through every single byte of the stream. */
struct slow_src {
    struct jpeg_source_mgr pub_mgr;
    const unsigned char *data;
    size_t total;
    size_t pos;
    JOCTET one_byte;
    size_t refill_calls;
};

static void slow_init(j_decompress_ptr cinfo) {
    struct slow_src *s = (struct slow_src *)cinfo->src;
    s->pos = 0;
    s->refill_calls = 0;
    s->pub_mgr.next_input_byte = NULL;
    s->pub_mgr.bytes_in_buffer = 0;
}

static boolean slow_fill(j_decompress_ptr cinfo) {
    struct slow_src *s = (struct slow_src *)cinfo->src;
    s->refill_calls++;
    if (s->pos >= s->total) {
        return FALSE; /* suspend; caller should treat as truncated */
    }
    s->one_byte = s->data[s->pos++];
    s->pub_mgr.next_input_byte = &s->one_byte;
    s->pub_mgr.bytes_in_buffer = 1;
    /* Return TRUE so the decoder consumes the one byte. The "suspends
     * every byte" character comes from the fact that we'll need to be
     * called again for the NEXT byte. */
    return TRUE;
}

static void slow_skip(j_decompress_ptr cinfo, long num_bytes) {
    struct slow_src *s = (struct slow_src *)cinfo->src;
    if (num_bytes > 0) {
        size_t skip = (size_t)num_bytes;
        if (skip > s->pub_mgr.bytes_in_buffer) {
            skip -= s->pub_mgr.bytes_in_buffer;
            s->pos += skip;
            s->pub_mgr.bytes_in_buffer = 0;
        } else {
            s->pub_mgr.next_input_byte += skip;
            s->pub_mgr.bytes_in_buffer -= skip;
        }
    }
}

static void slow_term(j_decompress_ptr cinfo) {
    (void)cinfo;
}

int main(void) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    struct slow_src src;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    src.pub_mgr.init_source = slow_init;
    src.pub_mgr.fill_input_buffer = slow_fill;
    src.pub_mgr.skip_input_data = slow_skip;
    src.pub_mgr.resync_to_restart = jpeg_resync_to_restart;
    src.pub_mgr.term_source = slow_term;
    src.pub_mgr.next_input_byte = NULL;
    src.pub_mgr.bytes_in_buffer = 0;
    src.data = SAMPLE_JPEG;
    src.total = sizeof(SAMPLE_JPEG);
    src.pos = 0;
    src.refill_calls = 0;
    src.one_byte = 0;
    cinfo.src = (struct jpeg_source_mgr *)&src;

    int header_result = jpeg_read_header(&cinfo, TRUE);
    if (header_result != JPEG_HEADER_OK && header_result != JPEG_SUSPENDED) {
        fprintf(stderr, "unexpected header result %d\n", header_result);
        jpeg_destroy_decompress(&cinfo);
        return 1;
    }

    /* Many refill calls expected — at least one per byte. */
    if (src.refill_calls < src.total / 2) {
        fprintf(stderr,
            "expected aggressive refill (~%zu), saw %zu\n",
            src.total, src.refill_calls);
        jpeg_destroy_decompress(&cinfo);
        return 2;
    }

    jpeg_destroy_decompress(&cinfo);
    return 0;
}
"#;

#[test]
fn source_mgr_suspends_every_byte() {
    let rc = match compile_and_run_c(
        PATTERN_SOURCE_MGR_SUSPENDS_EVERY_BYTE,
        "source_mgr_suspends_every_byte",
    ) {
        Some(rc) => rc,
        None => return,
    };
    assert_eq!(
        rc, 0,
        "source-mgr-suspends-every-byte harness exited with code {rc}"
    );
}

// ---------- pattern 2: destination_mgr that rejects the first flush ----------

const PATTERN_DEST_MGR_REJECTS_FIRST_FLUSH: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <jerror.h>

/* Destination manager whose `empty_output_buffer` always returns FALSE.
 *
 * Upstream libjpeg-turbo documents this as a fatal contract violation:
 * unlike the decompress side, the compress side has *no* suspended-
 * output protocol. The first FALSE return MUST raise JERR_CANT_SUSPEND
 * (msg_code 25) via the error manager — anything else (silent loop,
 * advancing the buffer pointer despite FALSE, a heap-overflow into the
 * undersized window) is a state-machine bug.
 *
 * Test goal: confirm the shim surfaces JERR_CANT_SUSPEND through the
 * standard setjmp/longjmp error_exit path so a defensive C consumer can
 * catch the failure instead of crashing. */

#define WINDOW_BYTES 8
struct rejecting_dest {
    struct jpeg_destination_mgr pub_mgr;
    unsigned char window[WINDOW_BYTES];
    int empty_calls;
};

struct setjmp_err_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf escape;
    int captured_code;
};

static void rej_init(j_compress_ptr cinfo) {
    struct rejecting_dest *d = (struct rejecting_dest *)cinfo->dest;
    d->pub_mgr.next_output_byte = d->window;
    d->pub_mgr.free_in_buffer = WINDOW_BYTES;
    d->empty_calls = 0;
}

static boolean rej_empty(j_compress_ptr cinfo) {
    struct rejecting_dest *d = (struct rejecting_dest *)cinfo->dest;
    d->empty_calls++;
    /* Always FALSE — the documented contract violation. */
    return FALSE;
}

static void rej_term(j_compress_ptr cinfo) {
    /* Should never be reached: error_exit must longjmp first. */
    (void)cinfo;
}

static void trapping_error_exit(j_common_ptr cinfo) {
    struct setjmp_err_mgr *err = (struct setjmp_err_mgr *)cinfo->err;
    err->captured_code = err->pub.msg_code;
    /* Don't print — we expect this path. */
    longjmp(err->escape, 1);
}

static void quiet_emit(j_common_ptr cinfo, int msg_level) {
    (void)cinfo;
    (void)msg_level;
}

int main(void) {
    struct jpeg_compress_struct cinfo;
    struct setjmp_err_mgr jerr;
    struct rejecting_dest dest;

    memset(&dest, 0, sizeof(dest));
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = trapping_error_exit;
    jerr.pub.emit_message = quiet_emit;
    jerr.captured_code = -1;

    if (setjmp(jerr.escape)) {
        /* error_exit fired. The safety property we pin: JERR_CANT_SUSPEND
         * is the documented response to a destination that cannot accept
         * the encoded stream. The shim is free to detect this eagerly
         * (refusing the destination on first write) or lazily (only
         * after rej_empty returns FALSE) — both paths are safe and
         * catchable; we don't pin which one. We use the symbolic
         * JERR_CANT_SUSPEND from <jerror.h> rather than its numeric
         * value because the enum shifts between JPEG_LIB_VERSION
         * targets (v6b is the only one that *includes*
         * JERR_ARITH_NOTIMPL — jerror.h gates it on
         * `JPEG_LIB_VERSION < 70` — so every code after it sits one
         * higher there than it does at v7/v8). */
        if (jerr.captured_code != JERR_CANT_SUSPEND) {
            fprintf(stderr,
                "expected JERR_CANT_SUSPEND (%d), got %d\n",
                (int)JERR_CANT_SUSPEND, jerr.captured_code);
            jpeg_destroy_compress(&cinfo);
            return 2;
        }
        jpeg_destroy_compress(&cinfo);
        return 0; /* success: the shim surfaced the documented error */
    }

    jpeg_create_compress(&cinfo);
    dest.pub_mgr.init_destination = rej_init;
    dest.pub_mgr.empty_output_buffer = rej_empty;
    dest.pub_mgr.term_destination = rej_term;
    cinfo.dest = (struct jpeg_destination_mgr *)&dest;

    /* A small grayscale encode is enough — the 8-byte window guarantees
     * the encoder hits empty_output_buffer before SOI..APP0 finishes. */
    cinfo.image_width = 8;
    cinfo.image_height = 8;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);
    cinfo.optimize_coding = TRUE;

    JSAMPLE row[8];
    for (int i = 0; i < 8; i++) row[i] = (JSAMPLE)(i * 32);

    /* If the contract holds, jpeg_start_compress will call rej_empty,
     * see FALSE, and trapping_error_exit will longjmp back to the
     * setjmp above. We should never reach jpeg_finish_compress. */
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW r = row;
        jpeg_write_scanlines(&cinfo, &r, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    fprintf(stderr,
        "encode finished without firing JERR_CANT_SUSPEND — "
        "destination contract violated\n");
    return 4;
}
"#;

#[test]
fn dest_mgr_rejects_first_flush() {
    let rc = match compile_and_run_c(
        PATTERN_DEST_MGR_REJECTS_FIRST_FLUSH,
        "dest_mgr_rejects_first_flush",
    ) {
        Some(rc) => rc,
        None => return,
    };
    assert_eq!(
        rc, 0,
        "dest-mgr-rejects-first-flush harness exited with code {rc}"
    );
}

// ---------- pattern 3: save_markers truncates a multi-chunk ICC profile ----------

const PATTERN_SAVE_MARKERS_TRUNCATES_MULTICHUNK_ICC: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <jerror.h>

/* A minimal JPEG with two APP2 segments carrying an "ICC_PROFILE\0" marker
 * (chunks 1/2 and 2/2). Each chunk's payload header is the standard
 * 14 bytes "ICC_PROFILE\0" + chunk_num + chunk_count, followed by
 * 16 bytes of synthetic ICC payload. We then attach a SOI/SOF/SOS/EOI
 * tail so the stream parses cleanly through to the SOS.
 *
 * The intent: when the caller requests jpeg_save_markers(cinfo, JPEG_APP0+2,
 * length_limit=1), the marker list must retain both APP2 segments — one
 * jpeg_marker_struct per segment — with data_length=1 (i.e. only one byte
 * of payload kept past the marker header). The shim's marker_list
 * truncation logic is the code under test.
 *
 * The CHUNK_PAYLOAD_BYTES per APP2 segment includes:
 *   14 bytes "ICC_PROFILE\0" + chunk_num (1 byte) + chunk_count (1 byte)
 *   16 bytes synthetic ICC data
 *   --
 *   30 bytes payload  →  APP2 marker length field = 30+2 = 32 = 0x0020 */
static const unsigned char SAMPLE_JPEG[] = {
    0xFF, 0xD8,                                    /* SOI */
    /* APP2 #1: chunk 1 of 2 */
    0xFF, 0xE2, 0x00, 0x20,
    'I','C','C','_','P','R','O','F','I','L','E', 0x00,
    0x01, 0x02,                                    /* chunk 1 of 2 */
    0xAA,0xBB,0xCC,0xDD, 0x11,0x22,0x33,0x44,
    0x55,0x66,0x77,0x88, 0x99,0xAA,0xBB,0xCC,
    /* APP2 #2: chunk 2 of 2 */
    0xFF, 0xE2, 0x00, 0x20,
    'I','C','C','_','P','R','O','F','I','L','E', 0x00,
    0x02, 0x02,                                    /* chunk 2 of 2 */
    0xDD,0xEE,0xFF,0x00, 0x11,0x22,0x33,0x44,
    0x55,0x66,0x77,0x88, 0x99,0xAA,0xBB,0xCC,
    /* Minimal DQT */
    0xFF, 0xDB, 0x00, 0x43, 0x00,
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99,
    /* SOF0 8x8 grayscale */
    0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
    /* Minimal DHT DC */
    0xFF, 0xC4, 0x00, 0x1F, 0x00,
    0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0,
    0,1,2,3,4,5,6,7,8,9,10,11,
    /* Minimal DHT AC */
    0xFF, 0xC4, 0x00, 0xB5, 0x10,
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
    /* SOS */
    0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
    0xFC, 0xFF, 0xD9
};

int main(void) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    /* length_limit = 1: keep the marker headers but only one byte of
     * payload per chunk. The shim must still preserve TWO distinct
     * marker_list entries (one per APP2 segment) rather than coalescing. */
    jpeg_save_markers(&cinfo, JPEG_APP0 + 2, 1);

    jpeg_mem_src(&cinfo, (unsigned char *)SAMPLE_JPEG, sizeof(SAMPLE_JPEG));
    int hdr = jpeg_read_header(&cinfo, TRUE);
    if (hdr != JPEG_HEADER_OK) {
        fprintf(stderr, "jpeg_read_header returned %d\n", hdr);
        jpeg_destroy_decompress(&cinfo);
        return 1;
    }

    /* Walk the marker list. Expect exactly two APP2 markers, each with
     * data_length = 1 (the truncation limit). */
    int app2_count = 0;
    jpeg_saved_marker_ptr mk = cinfo.marker_list;
    while (mk != NULL) {
        if (mk->marker == JPEG_APP0 + 2) {
            app2_count++;
            if (mk->data_length != 1) {
                fprintf(stderr,
                    "APP2 marker has data_length=%u, expected 1\n",
                    mk->data_length);
                jpeg_destroy_decompress(&cinfo);
                return 2;
            }
            /* original_length should reflect the full 30-byte payload. */
            if (mk->original_length != 30) {
                fprintf(stderr,
                    "APP2 marker original_length=%u, expected 30\n",
                    mk->original_length);
                jpeg_destroy_decompress(&cinfo);
                return 3;
            }
        }
        mk = mk->next;
    }
    if (app2_count != 2) {
        fprintf(stderr,
            "expected 2 APP2 markers in the list, saw %d\n", app2_count);
        jpeg_destroy_decompress(&cinfo);
        return 4;
    }

    jpeg_destroy_decompress(&cinfo);
    return 0;
}
"#;

#[test]
fn save_markers_truncates_multichunk_icc() {
    let rc = match compile_and_run_c(
        PATTERN_SAVE_MARKERS_TRUNCATES_MULTICHUNK_ICC,
        "save_markers_truncates_multichunk_icc",
    ) {
        Some(rc) => rc,
        None => return,
    };
    assert_eq!(
        rc, 0,
        "save-markers-truncates-multichunk-icc harness exited with code {rc}"
    );
}

// ---------- pattern: P4-13 — real per-marker consume_input suspension ----------
//
// A *real* suspending source manager (`fill_input_buffer` returns FALSE when
// its drip buffer is empty — not the chunked-refill of
// `source_mgr_suspends_every_byte`) drives a multi-scan PROGRESSIVE JPEG through
// the buffered-image polling idiom. The harness asserts that:
//   * `jpeg_read_header` returns `JPEG_HEADER_OK` once the header (through the
//     first SOS) has been delivered, suspending until then;
//   * `jpeg_consume_input` returns `JPEG_SUSPENDED` when the drip buffer is dry
//     mid-body, `JPEG_REACHED_SOS` at each scan boundary, and `JPEG_REACHED_EOI`
//     at end-of-image — resuming after each suspension by delivering more bytes;
//   * the resumed decode is byte-for-byte identical to a full-buffer (`mem_src`)
//     decode of the same JPEG.
// P4-104 separately tracks finish-decompress state/reset fidelity.
// The progressive fixture is generated with `cjpeg -progressive` and embedded.
const PATTERN_CONSUME_INPUT_SUSPEND_PROGRESSIVE_TEMPLATE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <jerror.h>

/*{PROG_JPEG}*/

/* Reference RGB pixels produced by STOCK libjpeg-turbo `djpeg -pnm` on the same
 * JPEG (decoded by the Rust test at build time and embedded here), so the
 * drip-fed decode is cross-validated against the C oracle, not just our own
 * mem_src path. */
/*{DJPEG_REF}*/
#define DJPEG_REF_LEN ((int)sizeof(DJPEG_REF))

/* Suspending source: delivers bytes only up to `avail`; fill returns FALSE
 * (real suspension) once `pos` reaches `avail`. The driver raises `avail`
 * after each suspension to model bytes arriving over time. */
struct drip_src {
    struct jpeg_source_mgr pub_mgr;
    const unsigned char *data;
    size_t total;
    size_t pos;
    size_t avail;
    JOCTET buf[64];
};

static void drip_init(j_decompress_ptr cinfo) {
    struct drip_src *s = (struct drip_src *)cinfo->src;
    s->pos = 0;
    s->pub_mgr.next_input_byte = NULL;
    s->pub_mgr.bytes_in_buffer = 0;
}
static boolean drip_fill(j_decompress_ptr cinfo) {
    struct drip_src *s = (struct drip_src *)cinfo->src;
    if (s->pos >= s->avail) return FALSE;        /* real suspension */
    size_t n = s->avail - s->pos;
    if (n > sizeof(s->buf)) n = sizeof(s->buf);
    memcpy(s->buf, s->data + s->pos, n);
    s->pub_mgr.next_input_byte = s->buf;
    s->pub_mgr.bytes_in_buffer = n;
    s->pos += n;
    return TRUE;
}
static void drip_skip(j_decompress_ptr cinfo, long num_bytes) {
    struct drip_src *s = (struct drip_src *)cinfo->src;
    if (num_bytes <= 0) return;
    size_t skip = (size_t)num_bytes;
    if (skip <= s->pub_mgr.bytes_in_buffer) {
        s->pub_mgr.next_input_byte += skip;
        s->pub_mgr.bytes_in_buffer -= skip;
    } else {
        skip -= s->pub_mgr.bytes_in_buffer;
        s->pub_mgr.bytes_in_buffer = 0;
        s->pos += skip;
    }
}
static void drip_term(j_decompress_ptr cinfo) { (void)cinfo; }

/* Full-buffer reference decode via mem_src. */
static int decode_full(unsigned char *out, size_t out_cap,
                       JDIMENSION *w, JDIMENSION *h, int *comps) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, (unsigned char *)PROG_JPEG, sizeof(PROG_JPEG));
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo); return -1;
    }
    if (!jpeg_start_decompress(&cinfo)) { jpeg_destroy_decompress(&cinfo); return -2; }
    *w = cinfo.output_width; *h = cinfo.output_height; *comps = cinfo.output_components;
    size_t row_bytes = (size_t)cinfo.output_width * cinfo.output_components;
    if (row_bytes * cinfo.output_height > out_cap) { jpeg_destroy_decompress(&cinfo); return -3; }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *rowp = out + (size_t)cinfo.output_scanline * row_bytes;
        JSAMPROW rows[1]; rows[0] = rowp;
        if (jpeg_read_scanlines(&cinfo, rows, 1) != 1) break;
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}

static unsigned char g_ref[4 * 1024 * 1024];
static unsigned char g_drip[4 * 1024 * 1024];

int main(void) {
    JDIMENSION rw = 0, rh = 0; int rcomp = 0;
    if (decode_full(g_ref, sizeof(g_ref), &rw, &rh, &rcomp) != 0) {
        fprintf(stderr, "reference (mem_src) decode failed\n"); return 10;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    struct drip_src src;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    memset(&src, 0, sizeof(src));
    src.pub_mgr.init_source = drip_init;
    src.pub_mgr.fill_input_buffer = drip_fill;
    src.pub_mgr.skip_input_data = drip_skip;
    src.pub_mgr.resync_to_restart = jpeg_resync_to_restart;
    src.pub_mgr.term_source = drip_term;
    src.data = PROG_JPEG;
    src.total = sizeof(PROG_JPEG);
    src.pos = 0;
    src.avail = 2; /* start tiny: even the header must wait for more bytes */
    cinfo.src = (struct jpeg_source_mgr *)&src;

    /* Header: suspend until the through-SOS prefix has been delivered. */
    int hdr_suspends = 0, r;
    while ((r = jpeg_read_header(&cinfo, TRUE)) == JPEG_SUSPENDED) {
        hdr_suspends++;
        src.avail += 16; if (src.avail > src.total) src.avail = src.total;
        if (hdr_suspends > (int)src.total + 16) { fprintf(stderr, "header loop runaway\n"); return 11; }
    }
    if (r != JPEG_HEADER_OK) { fprintf(stderr, "header result %d\n", r); return 12; }

    cinfo.buffered_image = TRUE;
    if (!jpeg_start_decompress(&cinfo)) { fprintf(stderr, "start_decompress failed\n"); return 13; }

    /* Body: drive consume_input, resuming after each real suspension. */
    int consume_suspends = 0, reached_sos = 0, reached_eoi = 0, guard = 0;
    while (!jpeg_input_complete(&cinfo)) {
        int cr = jpeg_consume_input(&cinfo);
        if (cr == JPEG_SUSPENDED) {
            consume_suspends++;
            src.avail += 16; if (src.avail > src.total) src.avail = src.total;
        } else if (cr == JPEG_REACHED_SOS) {
            reached_sos++;
        } else if (cr == JPEG_REACHED_EOI) {
            reached_eoi++;
        }
        if (++guard > 1000000) { fprintf(stderr, "consume loop runaway\n"); return 14; }
    }
    if (consume_suspends < 1) { fprintf(stderr, "no real body suspension exercised\n"); return 15; }
    if (reached_sos < 1) { fprintf(stderr, "no JPEG_REACHED_SOS (expected progressive multi-scan)\n"); return 16; }
    if (reached_eoi < 1) { fprintf(stderr, "no JPEG_REACHED_EOI\n"); return 17; }
    /* input_scan_number must track the SOS events in lock-step: read_header set
     * it to 1 at the first SOS, and each REACHED_SOS bumped it by one. */
    if (cinfo.input_scan_number != 1 + reached_sos) {
        fprintf(stderr, "input_scan_number=%d, expected %d (1 + %d REACHED_SOS)\n",
                cinfo.input_scan_number, 1 + reached_sos, reached_sos);
        return 23;
    }

    if (cinfo.output_width != rw || cinfo.output_height != rh || cinfo.output_components != rcomp) {
        fprintf(stderr, "dim mismatch drip=%ux%ux%d ref=%ux%ux%d\n",
                cinfo.output_width, cinfo.output_height, cinfo.output_components, rw, rh, rcomp);
        return 18;
    }
    size_t row_bytes = (size_t)rw * rcomp;
    size_t total = row_bytes * rh;
    if (total > sizeof(g_drip)) { fprintf(stderr, "image too big\n"); return 19; }

    int final_scan = cinfo.input_scan_number;
    jpeg_start_output(&cinfo, final_scan);
    /* jpeg_start_output must record the requested scan so the documented
     * buffered-image termination (input_scan_number == output_scan_number)
     * holds for this suspending stream. */
    if (cinfo.output_scan_number != final_scan) {
        fprintf(stderr, "output_scan_number=%d, expected %d\n",
                cinfo.output_scan_number, final_scan);
        return 24;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *rowp = g_drip + (size_t)cinfo.output_scanline * row_bytes;
        JSAMPROW rows[1]; rows[0] = rowp;
        if (jpeg_read_scanlines(&cinfo, rows, 1) != 1) break;
    }
    jpeg_finish_output(&cinfo);

    if (memcmp(g_ref, g_drip, total) != 0) {
        fprintf(stderr, "PIXEL MISMATCH: drip-fed decode != full-buffer decode\n");
        return 20;
    }

    /* Cross-validate against the STOCK libjpeg-turbo (djpeg) reference — the
     * repo's mandatory C oracle. Catches a shim-vs-stock divergence that the
     * shim-only comparison above would miss. */
    if (total != (size_t)DJPEG_REF_LEN) {
        fprintf(stderr, "stock djpeg ref length %d != decoded %zu\n", DJPEG_REF_LEN, total);
        return 25;
    }
    {
        int maxd = 0;
        for (size_t k = 0; k < total; k++) {
            int d = (int)g_drip[k] - (int)DJPEG_REF[k];
            if (d < 0) d = -d;
            if (d > maxd) maxd = d;
        }
        /* P4-13 acceptance requires byte-identical cross-validation; our decoder
         * is byte-exact with libjpeg-turbo's islow IDCT, so demand maxd == 0. */
        if (maxd != 0) {
            fprintf(stderr, "drip-fed decode differs from stock djpeg by %d (require 0)\n", maxd);
            return 26;
        }
    }

    if (!jpeg_finish_decompress(&cinfo)) { fprintf(stderr, "finish_decompress failed\n"); return 21; }
    jpeg_destroy_decompress(&cinfo);
    return 0;
}
"#;

/// Decode `jpeg` with stock `djpeg -pnm` and return the raw interleaved RGB
/// pixels (P6 PPM body). The C oracle for the P4-13 cross-validation.
fn djpeg_decode_rgb(djpeg: &Path, jpeg: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write as _;
    use std::process::Stdio;
    let mut child = Command::new(djpeg)
        .arg("-pnm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(jpeg).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    // Parse a binary PPM (P6): magic, width, height, maxval, then w*h*3 bytes.
    let b = &out.stdout;
    if b.len() < 2 || &b[..2] != b"P6" {
        return None;
    }
    let mut i = 2usize;
    let mut toks: Vec<usize> = Vec::new();
    while toks.len() < 3 && i < b.len() {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            toks.push(std::str::from_utf8(&b[start..i]).ok()?.parse().ok()?);
        }
    }
    if toks.len() < 3 {
        return None;
    }
    i += 1; // single whitespace after maxval
    let need = toks[0].checked_mul(toks[1])?.checked_mul(3)?;
    if b.len() < i + need {
        return None;
    }
    Some(b[i..i + need].to_vec())
}

/// Generate a multi-scan progressive JPEG with `cjpeg -progressive` from a
/// 32x32 RGB gradient.
fn make_progressive_jpeg(cjpeg: &Path) -> Option<Vec<u8>> {
    use std::io::Write as _;
    use std::process::Stdio;
    let (w, h) = (32usize, 32usize);
    let mut ppm: Vec<u8> = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            ppm.push((x * 8) as u8);
            ppm.push((y * 8) as u8);
            ppm.push(((x + y) * 4) as u8);
        }
    }
    let mut child = Command::new(cjpeg)
        .args(["-progressive", "-quality", "90", "-sample", "1x1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(&ppm).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() || out.stdout.len() < 100 {
        return None;
    }
    Some(out.stdout)
}

fn bytes_to_c_array(name: &str, bytes: &[u8]) -> String {
    let mut s = format!("static const unsigned char {name}[] = {{\n");
    for (i, b) in bytes.iter().enumerate() {
        if i % 16 == 0 {
            s.push_str("    ");
        }
        s.push_str(&format!("0x{b:02X},"));
        if i % 16 == 15 {
            s.push('\n');
        }
    }
    s.push_str("\n};\n");
    s
}

#[test]
fn consume_input_suspends_through_progressive_body() {
    const TEST_NAME: &str = "consume_input_suspends_through_progressive_body";
    let Some(cjpeg) = require_c_tool("cjpeg", TEST_NAME) else {
        return;
    };
    let Some(djpeg) = require_c_tool("djpeg", TEST_NAME) else {
        return;
    };
    let jpeg: Vec<u8> = make_progressive_jpeg(&cjpeg)
        .expect("cjpeg was found but failed to produce the progressive fixture");
    // The fixture must actually be progressive (SOF2) to have multiple scans.
    assert!(
        jpeg.windows(2).any(|w| w == [0xFF, 0xC2]),
        "generated fixture is not progressive (no SOF2 marker)"
    );
    // Stock C oracle: decode the same JPEG with djpeg and embed the reference
    // pixels so the harness cross-validates against libjpeg-turbo, not just our
    // own mem_src path (repo rule + P4-13 acceptance criterion).
    let djpeg_ref: Vec<u8> = djpeg_decode_rgb(&djpeg, &jpeg)
        .expect("djpeg was found but failed to decode the progressive fixture");
    let c_array = bytes_to_c_array("PROG_JPEG", &jpeg);
    let ref_array = bytes_to_c_array("DJPEG_REF", &djpeg_ref);
    let c_source = PATTERN_CONSUME_INPUT_SUSPEND_PROGRESSIVE_TEMPLATE
        .replace("/*{PROG_JPEG}*/", &c_array)
        .replace("/*{DJPEG_REF}*/", &ref_array);
    let rc = match compile_and_run_c(&c_source, "consume_input_suspend_progressive") {
        Some(rc) => rc,
        None => return,
    };
    assert_eq!(
        rc, 0,
        "P4-13 consume_input suspension harness exited with code {rc}"
    );
}
