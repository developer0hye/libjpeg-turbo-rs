//! P3-5: classic `jpeglib.h` lifecycle / custom-I/O / suspension C harness.
//!
//! These tests exist because the existing C-consumer harnesses
//! (`capi_pillow_compat`, `capi_imagemagick_compat`, `capi_libvips_compat`,
//! `capi_ffmpeg_compat`, `capi_gd_compat`, `capi_sdl_image_compat`,
//! `libtiff_integration`) all drive the cdylib through canned
//! `jpeg_mem_src` / `jpeg_mem_dest` paths plus the raw-data API. They do
//! *not* exercise the classic state-machine edge cases an arbitrary C
//! consumer can construct: custom source managers, custom destination
//! managers, suspension semantics, abort+reuse lifecycles, buffered-image
//! multi-pass progressive output, and `setjmp`/`longjmp` error cleanup
//! with a custom `error_exit`.
//!
//! Each test:
//!   1. Synthesises a minimal `jconfig.h` and compiles a small C harness
//!      against the submodule's `references/libjpeg-turbo/src/jpeglib.h`.
//!   2. Links it against our cdylib through symlinks (`libjpeg.so.62` /
//!      `libjpeg.62.dylib`).
//!   3. Runs the binary and asserts exit code 0. The C harness performs
//!      its own correctness check (pixel equality vs a baseline path) and
//!      writes a diagnostic to stderr on mismatch.
//!
//! Skip-with-reason cases:
//! - `cc` not on PATH (legitimate dev-machine skip).
//! - Submodule not initialised (`references/libjpeg-turbo/src/jpeglib.h`
//!   missing).
//! - Compile failed for environmental reasons (missing system headers).
//!
//! Hard-fail cases (per CLAUDE.md C cross-validation rules):
//! - Compile succeeded but binary exits non-zero (real lifecycle bug).
//! - Cdylib is missing — that's a build-system regression, not an env issue.
//!
//! Patterns covered (one `#[test]` each, per `docs/last_mile/phase3.md`
//! P3-5 acceptance):
//!   1. Custom `jpeg_source_mgr` (callback-driven, small chunks).
//!   2. Custom `jpeg_destination_mgr` with `empty_output_buffer` flush stress.
//!   3. Source suspension (`fill_input_buffer` returns `FALSE`).
//!   4. Destination suspension / partial flush.
//!   5. `jpeg_abort_decompress` followed by re-use of the same struct.
//!   6. `jpeg_abort_compress` followed by re-use.
//!   7. Buffered-image multi-pass progressive scan loop.
//!   8. `setjmp`/`longjmp` error-cleanup with a custom `error_exit`.

use std::path::{Path, PathBuf};
use std::process::Command;

// ---------- shared helpers (pattern from roundtrip_c_client.rs) ----------

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
        "could not locate cdylib near {} — build it first with `cargo build -p libjpeg-turbo-rs-capi --release`",
        exe.display()
    );
}

fn find_cc() -> Option<PathBuf> {
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

#[cfg(unix)]
fn setup_symlinks(lib: &Path, parent: &Path) -> PathBuf {
    let subdir: PathBuf = parent.join("symlinks");
    std::fs::create_dir_all(&subdir).expect("mkdir symlinks");
    let (versioned, short): (&str, &str) = if cfg!(target_os = "macos") {
        ("libjpeg.62.dylib", "libjpeg.dylib")
    } else {
        ("libjpeg.so.62", "libjpeg.so")
    };
    for name in [versioned, short] {
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
    let header: PathBuf = candidate.join("jpeglib.h");
    if header.exists() {
        Some(candidate.canonicalize().expect("canonicalize upstream_src"))
    } else {
        None
    }
}

/// Minimal `jconfig.h` matching the v8 defaults the upstream
/// `CMakeLists.txt` uses. Same content as `tests/abi_offsets.rs`.
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

/// Compile the C source against the upstream headers, link against our
/// cdylib via symlinks, run it, and return (status, stdout, stderr).
///
/// On any environmental failure (no cc, no submodule, missing system
/// headers) emits a `SKIP:` line and returns `Err(skip_reason)`. On a
/// hard cdylib-missing failure or compile error that isn't environmental,
/// panics — that is a real regression, not a skip.
fn compile_and_run(test_name: &str, c_src: &str, exe_args: &[&str]) -> Result<(), String> {
    let cc: PathBuf = match find_cc() {
        Some(p) => p,
        None => {
            return Err("no C compiler (cc/clang/gcc) found on PATH".to_string());
        }
    };

    let upstream_src: PathBuf = match upstream_src_dir() {
        Some(p) => p,
        None => {
            return Err(
                "submodule not initialised: references/libjpeg-turbo/src/jpeglib.h missing"
                    .to_string(),
            );
        }
    };

    let lib: PathBuf = cdylib_path();
    let lib_dir: &Path = lib.parent().expect("cdylib parent");
    let lib_stem: String = lib
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| {
            if cfg!(target_os = "windows") {
                s.to_string()
            } else if let Some(rest) = s.strip_prefix("lib") {
                rest.to_string()
            } else {
                s.to_string()
            }
        })
        .expect("lib stem");

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let jconfig_path: PathBuf = tmp.path().join("jconfig.h");
    let c_src_path: PathBuf = tmp.path().join(format!("{test_name}.c"));
    let bin_path: PathBuf = tmp.path().join(test_name);

    std::fs::write(&jconfig_path, JCONFIG_H).expect("write jconfig.h");
    std::fs::write(&c_src_path, c_src).expect("write C source");

    let symlink_dir: PathBuf = setup_symlinks(&lib, tmp.path());

    // Compile.
    let mut cmd = Command::new(&cc);
    cmd.arg("-O0")
        .arg("-Wall")
        .arg("-Wno-implicit-function-declaration")
        .arg("-I")
        .arg(tmp.path())
        .arg("-I")
        .arg(&upstream_src)
        .arg("-o")
        .arg(&bin_path)
        .arg(&c_src_path);
    if cfg!(unix) {
        cmd.arg(format!("-L{}", symlink_dir.display()))
            .arg("-ljpeg")
            .arg(format!("-Wl,-rpath,{}", symlink_dir.display()));
    } else {
        cmd.arg(format!("-L{}", lib_dir.display()))
            .arg(format!("-l{lib_stem}"));
    }
    let compile = cmd.output().expect("invoke cc");
    if !compile.status.success() {
        let stderr: String = String::from_utf8_lossy(&compile.stderr).to_string();
        // Environmental skip: missing system headers, broken cross-compile.
        if stderr.contains("No such file or directory")
            || stderr.contains("cannot find")
            || stderr.contains("not found")
        {
            return Err(format!("cc compile failed (env issue):\n{stderr}"));
        }
        // Anything else is a real regression — the C source is wrong, or
        // the upstream header / our cdylib link surface drifted.
        panic!(
            "C harness for `{test_name}` failed to compile:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&compile.stdout),
            stderr
        );
    }

    // Run.
    let mut run_cmd = Command::new(&bin_path);
    run_cmd
        .args(exe_args)
        .env("LD_LIBRARY_PATH", &symlink_dir)
        .env("DYLD_LIBRARY_PATH", &symlink_dir);
    let run = run_cmd.output().expect("run binary");
    if !run.status.success() {
        panic!(
            "C harness `{test_name}` exited with code {}:\nstdout: {}\nstderr: {}",
            run.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
    Ok(())
}

fn run_or_skip(test_name: &str, c_src: &str) {
    match compile_and_run(test_name, c_src, &[]) {
        Ok(()) => {}
        Err(reason) => {
            eprintln!("SKIP {test_name}: {reason}");
        }
    }
}

// ---------- shared C preamble: TJ3 fixture builder + helpers ----------

/// Common preamble: TJ3 prototypes for fixture creation, plus a small
/// helper that synthesises a 64x64 RGB gradient and encodes it via
/// `tj3Compress8` so the test is self-contained (no external fixture
/// files). Each test C source concatenates this with its pattern-specific
/// code and a `main`.
const C_PREAMBLE: &str = r#"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jpeglib.h"
#include "jerror.h"

typedef void *tjhandle;
#define TJINIT_COMPRESS 1
#define TJPARAM_QUALITY 3
#define TJPARAM_SUBSAMP 4
#define TJPF_RGB 0
#define TJSAMP_444 0

extern tjhandle tj3Init(int);
extern void tj3Destroy(tjhandle);
extern int tj3Set(tjhandle, int, int);
extern int tj3Compress8(tjhandle, const unsigned char *, int, int, int, int,
                        unsigned char **, size_t *);
extern void tj3Free(void *);

static const int FIX_W = 64;
static const int FIX_H = 64;
static const int FIX_BPP = 3;

/* Build a 64x64 RGB gradient and encode it via TJ3. Caller must
 * tj3Free the returned buffer. Returns NULL on failure. */
static unsigned char *make_fixture(size_t *out_size, unsigned char **out_src) {
    int w = FIX_W, h = FIX_H, bpp = FIX_BPP;
    unsigned char *src = (unsigned char *)malloc((size_t)w * h * bpp);
    if (!src) return NULL;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char *p = src + ((size_t)y * w + x) * bpp;
            p[0] = (unsigned char)((x * 255) / (w - 1));
            p[1] = (unsigned char)((y * 255) / (h - 1));
            p[2] = (unsigned char)(((x + y) * 255) / (w + h - 2));
        }
    }
    tjhandle enc = tj3Init(TJINIT_COMPRESS);
    if (!enc) { free(src); return NULL; }
    tj3Set(enc, TJPARAM_QUALITY, 90);
    tj3Set(enc, TJPARAM_SUBSAMP, TJSAMP_444);
    unsigned char *jpeg = NULL;
    size_t jpeg_size = 0;
    int rc = tj3Compress8(enc, src, w, 0, h, TJPF_RGB, &jpeg, &jpeg_size);
    tj3Destroy(enc);
    if (rc != 0) { free(src); return NULL; }
    *out_size = jpeg_size;
    *out_src = src;
    return jpeg;
}

/* Decode via the canned mem_src path. Returns malloc'd RGB buffer or
 * NULL. Caller must free(). */
static unsigned char *decode_via_mem_src(const unsigned char *jpeg, size_t jpeg_size) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, jpeg, jpeg_size);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    cinfo.out_color_space = JCS_RGB;
    if (!jpeg_start_decompress(&cinfo)) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    int row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char *dst = (unsigned char *)malloc((size_t)cinfo.output_height * row_stride);
    if (!dst) { jpeg_destroy_decompress(&cinfo); return NULL; }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row_ptr = dst + (size_t)cinfo.output_scanline * row_stride;
        if (jpeg_read_scanlines(&cinfo, &row_ptr, 1) != 1) {
            free(dst);
            jpeg_destroy_decompress(&cinfo);
            return NULL;
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return dst;
}
"#;

// ---------- pattern #1: custom jpeg_source_mgr (callback-driven) ----------

const PATTERN_1_CUSTOM_SRC_MGR: &str = r#"
/* Custom source manager that delivers the JPEG bytes in fixed-size chunks,
 * exercising the fill_input_buffer callback path many times per decode. */
typedef struct {
    struct jpeg_source_mgr pub;   /* public part — must come first */
    const JOCTET *backing;        /* full JPEG */
    size_t backing_len;
    size_t backing_pos;           /* bytes already delivered */
    size_t chunk;                 /* bytes per fill */
    JOCTET local[256];            /* bounded local buffer */
    int fill_calls;               /* diagnostic counter */
} chunk_src_mgr;

static void chunk_init_source(j_decompress_ptr cinfo) {
    chunk_src_mgr *s = (chunk_src_mgr *)cinfo->src;
    s->backing_pos = 0;
    s->fill_calls = 0;
}

static boolean chunk_fill_input_buffer(j_decompress_ptr cinfo) {
    chunk_src_mgr *s = (chunk_src_mgr *)cinfo->src;
    s->fill_calls += 1;
    size_t remaining = s->backing_len - s->backing_pos;
    if (remaining == 0) {
        /* End of input — emit fake EOI marker per upstream contract
         * (jdatasrc.c::fill_input_buffer fallback). */
        s->local[0] = (JOCTET)0xFF;
        s->local[1] = (JOCTET)JPEG_EOI;
        s->pub.next_input_byte = s->local;
        s->pub.bytes_in_buffer = 2;
        WARNMS(cinfo, JWRN_JPEG_EOF);
        return TRUE;
    }
    size_t to_copy = remaining < s->chunk ? remaining : s->chunk;
    memcpy(s->local, s->backing + s->backing_pos, to_copy);
    s->backing_pos += to_copy;
    s->pub.next_input_byte = s->local;
    s->pub.bytes_in_buffer = to_copy;
    return TRUE;
}

static void chunk_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
    chunk_src_mgr *s = (chunk_src_mgr *)cinfo->src;
    if (num_bytes <= 0) return;
    size_t n = (size_t)num_bytes;
    /* Consume from the in-memory window first. */
    if (n < s->pub.bytes_in_buffer) {
        s->pub.next_input_byte += n;
        s->pub.bytes_in_buffer -= n;
        return;
    }
    n -= s->pub.bytes_in_buffer;
    s->pub.bytes_in_buffer = 0;
    /* Then advance the backing position. */
    if (n > s->backing_len - s->backing_pos) {
        s->backing_pos = s->backing_len;
    } else {
        s->backing_pos += n;
    }
}

static void chunk_term_source(j_decompress_ptr cinfo) {
    (void)cinfo;
}

static void install_chunk_src(j_decompress_ptr cinfo, chunk_src_mgr *s,
                              const JOCTET *buf, size_t len, size_t chunk) {
    s->pub.init_source = chunk_init_source;
    s->pub.fill_input_buffer = chunk_fill_input_buffer;
    s->pub.skip_input_data = chunk_skip_input_data;
    s->pub.resync_to_restart = jpeg_resync_to_restart;
    s->pub.term_source = chunk_term_source;
    s->pub.bytes_in_buffer = 0;
    s->pub.next_input_byte = NULL;
    s->backing = buf;
    s->backing_len = len;
    s->chunk = chunk;
    s->backing_pos = 0;
    s->fill_calls = 0;
    cinfo->src = (struct jpeg_source_mgr *)s;
}

/* Decode via custom source mgr with a small chunk size so the consumer's
 * fill_input_buffer fires many times during a single decode. */
static unsigned char *decode_via_chunk_src(const unsigned char *jpeg, size_t jpeg_size,
                                           size_t chunk, int *out_fill_calls) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    chunk_src_mgr src_mgr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    install_chunk_src(&cinfo, &src_mgr, jpeg, jpeg_size, chunk);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    cinfo.out_color_space = JCS_RGB;
    if (!jpeg_start_decompress(&cinfo)) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    int row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char *dst = (unsigned char *)malloc((size_t)cinfo.output_height * row_stride);
    if (!dst) { jpeg_destroy_decompress(&cinfo); return NULL; }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row_ptr = dst + (size_t)cinfo.output_scanline * row_stride;
        if (jpeg_read_scanlines(&cinfo, &row_ptr, 1) != 1) {
            free(dst);
            jpeg_destroy_decompress(&cinfo);
            return NULL;
        }
    }
    jpeg_finish_decompress(&cinfo);
    *out_fill_calls = src_mgr.fill_calls;
    jpeg_destroy_decompress(&cinfo);
    return dst;
}

int main(void) {
    unsigned char *src = NULL;
    size_t jpeg_size = 0;
    unsigned char *jpeg = make_fixture(&jpeg_size, &src);
    if (!jpeg) {
        fprintf(stderr, "make_fixture failed\n");
        return 2;
    }
    /* Fixture must be larger than the chunk so fill_input_buffer fires
     * more than once — otherwise this test isn't actually exercising
     * the callback path. */
    if (jpeg_size < 64) {
        fprintf(stderr, "fixture suspiciously small: %zu bytes\n", jpeg_size);
        tj3Free(jpeg); free(src);
        return 3;
    }

    unsigned char *baseline = decode_via_mem_src(jpeg, jpeg_size);
    if (!baseline) {
        fprintf(stderr, "baseline mem_src decode failed\n");
        tj3Free(jpeg); free(src);
        return 4;
    }

    int fill_calls = 0;
    unsigned char *via_chunk = decode_via_chunk_src(jpeg, jpeg_size, 17, &fill_calls);
    if (!via_chunk) {
        fprintf(stderr, "custom chunk_src decode failed\n");
        free(baseline); tj3Free(jpeg); free(src);
        return 5;
    }

    /* Sanity: a 17-byte chunk over a >= 64-byte fixture must invoke
     * fill_input_buffer at least twice. If it didn't, the shim is
     * silently consuming the entire buffer up-front, which would defeat
     * the purpose of this test. */
    if (fill_calls < 2) {
        fprintf(stderr,
                "fill_input_buffer fired only %d time(s) for a %zu-byte JPEG "
                "with chunk=17 — custom source mgr is not being driven\n",
                fill_calls, jpeg_size);
        free(via_chunk); free(baseline); tj3Free(jpeg); free(src);
        return 6;
    }

    /* The two decodes must produce byte-identical RGB output. */
    size_t pixel_bytes = (size_t)FIX_W * FIX_H * FIX_BPP;
    size_t mismatches = 0;
    int max_diff = 0;
    for (size_t i = 0; i < pixel_bytes; ++i) {
        int d = (int)baseline[i] - (int)via_chunk[i];
        if (d < 0) d = -d;
        if (d != 0) {
            mismatches += 1;
            if (d > max_diff) max_diff = d;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr,
                "custom_src vs mem_src pixel mismatch: %zu/%zu bytes differ, max diff %d\n",
                mismatches, pixel_bytes, max_diff);
        free(via_chunk); free(baseline); tj3Free(jpeg); free(src);
        return 7;
    }

    fprintf(stderr,
            "OK custom_src_mgr: %zu-byte JPEG, chunk=17, fill_calls=%d, pixels match\n",
            jpeg_size, fill_calls);
    free(via_chunk); free(baseline); tj3Free(jpeg); free(src);
    return 0;
}
"#;

#[test]
fn pattern_1_custom_jpeg_source_mgr_drives_fill_input_buffer() {
    let c_src = format!("{C_PREAMBLE}\n{PATTERN_1_CUSTOM_SRC_MGR}");
    run_or_skip("lifecycle_custom_src", &c_src);
}

// ---------- patterns #2-#8: tracked, deferred to follow-up commits ----------

#[test]
#[ignore = "P3-5 follow-up: custom jpeg_destination_mgr with empty_output_buffer flush"]
fn pattern_2_custom_jpeg_destination_mgr_drives_empty_output_buffer() {}

#[test]
#[ignore = "P3-5 follow-up: source suspension (fill_input_buffer returns FALSE)"]
fn pattern_3_source_suspension_returns_control_to_consumer() {}

#[test]
#[ignore = "P3-5 follow-up: destination suspension / partial flush"]
fn pattern_4_destination_suspension_partial_flush() {}

#[test]
#[ignore = "P3-5 follow-up: jpeg_abort_decompress + reuse of same struct"]
fn pattern_5_jpeg_abort_decompress_then_reuse() {}

#[test]
#[ignore = "P3-5 follow-up: jpeg_abort_compress + reuse"]
fn pattern_6_jpeg_abort_compress_then_reuse() {}

#[test]
#[ignore = "P3-5 follow-up: buffered-image multi-pass progressive (jpeg_consume_input + jpeg_start_output + jpeg_finish_output)"]
fn pattern_7_buffered_image_multi_pass_progressive() {}

#[test]
#[ignore = "P3-5 follow-up: setjmp/longjmp error cleanup with custom error_exit"]
fn pattern_8_setjmp_longjmp_error_cleanup() {}
