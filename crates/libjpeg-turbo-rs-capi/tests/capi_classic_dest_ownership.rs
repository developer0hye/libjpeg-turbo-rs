//! P4-108: classic destination-manager buffer ownership and stdio I/O errors.
//!
//! `jpeg_mem_dest` has a documented ownership contract (jdatadst.c:222-234):
//! the caller may supply its own initial buffer, `*outsize` is that buffer's
//! **capacity** (not a payload length), and *"the library does not free such
//! buffer when allocating a larger buffer"*. `jpeg_stdio_dest` has an equally
//! sharp I/O contract: every short `fwrite` and any `ferror` after the final
//! `fflush` must raise `JERR_FILE_WRITE` through the error manager.
//!
//! Both contracts are memory-safety- and data-loss-relevant for a drop-in
//! replacement, and neither is observable from Rust-side tests: they only show
//! up when a real C consumer hands the library a stack/static/heap buffer it
//! still owns, or a stream that fails to accept bytes. So every case here is a
//! C canary compiled against the pinned upstream `jpeglib.h` at
//! `JPEG_LIB_VERSION == 80` and linked against the staged cdylib.
//!
//! The canary reports contract facts as `key=value` lines on stdout rather
//! than raw byte counts, so the very same binary can be linked against a
//! reference libjpeg-turbo v8 build and the two outputs compared verbatim.
//! Point `LIBJPEG_TURBO_REFERENCE_DIR` at a build tree (expects
//! `<dir>/lib` + `<dir>/include`, as installed by `cmake --install`) to enable
//! that differential leg; without it the canaries still run fail-closed
//! against our shim alone and the reference comparison reports an explicit
//! skip reason.
//!
//! Scenarios (one `#[test]` each):
//!   1. sufficient caller buffer on the stack — not moved, not freed
//!   2. sufficient caller buffer in static storage — same contract
//!   3. insufficient caller buffer — grows into library memory, encoded
//!      prefix preserved, caller buffer still caller-owned (`free`d by the
//!      canary itself, which aborts on a double free)
//!   4. `*outbuffer == NULL` — library allocates at `jpeg_mem_dest` time
//!   5. NULL out-parameters — `JERR_BUFFER_SIZE`
//!   6. non-NULL buffer with `*outsize == 0` — treated as no buffer at all
//!   7. repeated use of one compress handle across two images
//!   8. switching an installed stdio destination to `jpeg_mem_dest` —
//!      `JERR_BUFFER_SIZE`
//!   9. stdio short write below the staging threshold — `JERR_FILE_WRITE`
//!  10. stdio short write above the staging threshold (drives
//!      `empty_output_buffer`) — `JERR_FILE_WRITE`
//!  11. `/dev/full`: writes succeed but `fflush`/`ferror` fail —
//!      `JERR_FILE_WRITE` (reports an explicit skip where `/dev/full` does not
//!      exist)
//!  12. success control — bytes are on disk before `fclose`, stream has no
//!      error flag, and the file decodes back to the source pixels
//!
//! Every scenario that produces output also decodes it and compares against
//! the source pixels: SOI/EOI bookends alone would still pass if a
//! destination manager dropped the entire entropy-coded payload.

use std::path::{Path, PathBuf};
use std::process::Command;

// ---------- shared harness (pattern from capi_classic_lifecycle.rs) ----------

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
        "could not locate cdylib near {} — build it first with `cargo build -p libjpeg-turbo-rs-capi`",
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
    let names: &[&str] = if cfg!(target_os = "macos") {
        &["libjpeg.8.dylib", "libjpeg.dylib"]
    } else {
        &["libjpeg.so.8", "libjpeg.so"]
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
    let header: PathBuf = candidate.join("jpeglib.h");
    if header.exists() {
        Some(candidate.canonicalize().expect("canonicalize upstream_src"))
    } else {
        None
    }
}

/// Minimal `jconfig.h` matching the v8 defaults upstream's `CMakeLists.txt`
/// uses. Same content as `tests/abi_offsets.rs` and
/// `tests/capi_classic_lifecycle.rs`; the canary itself `#error`s if
/// `JPEG_LIB_VERSION` is ever anything but 80.
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

/// A libjpeg build the canary can link against: either our staged cdylib or a
/// reference libjpeg-turbo v8 install.
struct LinkTarget {
    /// `-I` directory holding `jpeglib.h` / `jerror.h`.
    include_dir: PathBuf,
    /// `-L` / rpath directory holding the shared library.
    lib_dir: PathBuf,
    /// A `jconfig.h` must be synthesised into the compile include path when
    /// the header directory does not already ship one (our submodule source
    /// tree does not; an installed reference build does).
    needs_synthetic_jconfig: bool,
}

/// The staged Rust cdylib under test.
fn shim_target(tmp: &Path) -> LinkTarget {
    let lib: PathBuf = cdylib_path();
    LinkTarget {
        include_dir: upstream_src_dir().expect("checked by caller"),
        lib_dir: setup_symlinks(&lib, tmp),
        needs_synthetic_jconfig: true,
    }
}

/// Reference libjpeg-turbo v8 install, when `LIBJPEG_TURBO_REFERENCE_DIR`
/// points at one. Returns the reason it is unavailable otherwise, so callers
/// can report an explicit skip instead of silently dropping the comparison.
fn reference_target() -> Result<LinkTarget, String> {
    let root: String = std::env::var("LIBJPEG_TURBO_REFERENCE_DIR")
        .map_err(|_| "LIBJPEG_TURBO_REFERENCE_DIR is not set".to_string())?;
    let root: PathBuf = PathBuf::from(root);
    let include_dir: PathBuf = root.join("include");
    if !include_dir.join("jpeglib.h").is_file() {
        return Err(format!(
            "LIBJPEG_TURBO_REFERENCE_DIR={} has no include/jpeglib.h",
            root.display()
        ));
    }
    // `cmake --install` lands the shared library under lib/ or lib64/
    // depending on the distribution's GNUInstallDirs defaults. Require an
    // actual v8 libjpeg in it: a directory that merely *exists* would let
    // `-L<dir> -ljpeg` fall through to the system libjpeg, silently turning
    // the differential leg into a comparison against some other build.
    let versioned: &[&str] = if cfg!(target_os = "macos") {
        &["libjpeg.8.dylib"]
    } else if cfg!(target_os = "windows") {
        &["jpeg8.lib", "jpeg.lib"]
    } else {
        &["libjpeg.so.8"]
    };
    let lib_dir: PathBuf = ["lib", "lib64"]
        .iter()
        .map(|d| root.join(d))
        .find(|d| versioned.iter().any(|n| d.join(n).exists()))
        .ok_or_else(|| {
            format!(
                "LIBJPEG_TURBO_REFERENCE_DIR={} has no {} under lib/ or lib64/ — \
                 build it with -DWITH_JPEG8=1",
                root.display(),
                versioned.join(" or ")
            )
        })?;
    // A reference install without its own `jconfig.h` cannot be trusted: we
    // would have to synthesise one, hardcoding `JPEG_LIB_VERSION 80` and
    // masking the very mismatch the canary's `#if` guard exists to catch.
    if !include_dir.join("jconfig.h").is_file() {
        return Err(format!(
            "LIBJPEG_TURBO_REFERENCE_DIR={} has no include/jconfig.h; refusing to \
             synthesise one, because that would assume the v8 ABI instead of proving it",
            root.display()
        ));
    }
    Ok(LinkTarget {
        include_dir,
        lib_dir,
        needs_synthetic_jconfig: false,
    })
}

/// Compile the canary against `target` and run one scenario.
///
/// Returns the canary's stdout on success. Panics when the canary compiles
/// but exits non-zero — that is the real contract violation this file exists
/// to catch, never an environment problem.
fn run_canary(target: &LinkTarget, label: &str, scenario: &str, scratch: &Path) -> String {
    let cc: PathBuf = find_cc().expect("checked by caller");
    let build: PathBuf = scratch.join(format!("build-{label}-{scenario}"));
    std::fs::create_dir_all(&build).expect("mkdir build dir");

    let c_src_path: PathBuf = build.join("p4108_dest_canary.c");
    std::fs::write(&c_src_path, CANARY_C).expect("write C source");
    if target.needs_synthetic_jconfig {
        std::fs::write(build.join("jconfig.h"), JCONFIG_H).expect("write jconfig.h");
    }
    let bin_path: PathBuf = build.join(if cfg!(target_os = "windows") {
        "p4108_dest_canary.exe"
    } else {
        "p4108_dest_canary"
    });

    let mut cmd = Command::new(&cc);
    cmd.arg("-O0")
        .arg("-Wall")
        // -Werror so a scenario that loses its dispatch arm fails the build as
        // -Wunused-function instead of quietly never running.
        .arg("-Werror")
        .arg("-I")
        .arg(&build)
        .arg("-I")
        .arg(&target.include_dir)
        .arg("-o")
        .arg(&bin_path)
        .arg(&c_src_path)
        .arg(format!("-L{}", target.lib_dir.display()))
        .arg("-ljpeg");
    if cfg!(unix) {
        cmd.arg(format!("-Wl,-rpath,{}", target.lib_dir.display()));
    }
    let compile = cmd.output().expect("invoke cc");
    if !compile.status.success() {
        panic!(
            "P4-108 canary failed to compile against {label}:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr)
        );
    }

    let workdir: PathBuf = build.join("work");
    std::fs::create_dir_all(&workdir).expect("mkdir workdir");
    let run = Command::new(&bin_path)
        .arg(scenario)
        .arg(&workdir)
        .env("LD_LIBRARY_PATH", &target.lib_dir)
        .env("DYLD_LIBRARY_PATH", &target.lib_dir)
        .output()
        .expect("run canary");
    if !run.status.success() {
        panic!(
            "P4-108 canary scenario `{scenario}` against {label} exited with {}:\nstdout: {}\nstderr: {}",
            run.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
    String::from_utf8_lossy(&run.stdout).to_string()
}

/// Run one scenario against the shim and — when a reference build is
/// configured — against upstream, requiring byte-identical contract reports.
///
/// Environmental prerequisites (no `cc`, submodule not initialised, cdylib
/// missing) produce an explicit `SKIP:` line, per the project's C
/// cross-validation rules. A contract mismatch never skips.
fn assert_scenario(scenario: &str) {
    if find_cc().is_none() {
        eprintln!("SKIP: {scenario}: no C compiler (cc/clang/gcc) on PATH");
        return;
    }
    if upstream_src_dir().is_none() {
        eprintln!(
            "SKIP: {scenario}: submodule not initialised (references/libjpeg-turbo/src/jpeglib.h missing)"
        );
        return;
    }
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let shim: LinkTarget = shim_target(tmp.path());
    let ours: String = run_canary(&shim, "shim", scenario, tmp.path());
    assert!(
        !ours.trim().is_empty(),
        "scenario `{scenario}` reported no contract facts — the canary must \
         print at least one key=value line, otherwise a vacuous run passes"
    );
    // Echo the contract report so `-- --nocapture` shows which facts were
    // actually established; CI greps this to prove /dev/full really ran.
    for line in ours.lines() {
        println!("{scenario}: {line}");
    }
    // A scenario that reported only its own unavailability asserted nothing.
    // Say so through the same `SKIP:` channel CI already fails closed on,
    // rather than letting it read as a passing case.
    if ours.contains("dev_full_available=0") {
        eprintln!(
            "SKIP: {scenario}: /dev/full does not exist on this platform, so the \
             fflush/ferror path was not exercised"
        );
    }

    match reference_target() {
        Ok(reference) => {
            let theirs: String = run_canary(&reference, "reference", scenario, tmp.path());
            assert_eq!(
                contract_facts(&ours),
                contract_facts(&theirs),
                "scenario `{scenario}`: destination contract diverges from the \
                 reference libjpeg-turbo v8 build\n  ours:      {}\n  reference: {}",
                ours.trim().replace('\n', " | "),
                theirs.trim().replace('\n', " | ")
            );
            compare_metrics(scenario, &ours, &theirs);
        }
        Err(reason) => {
            eprintln!(
                "SKIP (reference leg only): {scenario}: {reason}; \
                 shim-side contract assertions still ran and passed:\n{}",
                ours.trim()
            );
        }
    }
}

/// Contract facts: the `key=value` lines compared verbatim between legs.
fn contract_facts(report: &str) -> Vec<&str> {
    report
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// Metrics: `#key=value` lines carrying implementation-derived numbers.
///
/// These cannot be compared for equality — two conforming encoders produce
/// different byte counts and different quantisation error — but they are the
/// only values in the report that a broken destination manager can move
/// independently of the reference, so they are compared with a tolerance
/// instead of being dropped.
fn metrics(report: &str) -> Vec<(String, f64)> {
    report
        .lines()
        .map(str::trim)
        .filter_map(|l| l.strip_prefix('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            Some((k.to_string(), v.parse::<f64>().ok()?))
        })
        .collect()
}

/// Every metric must exist on both sides and stay within a factor of two of
/// the reference. A destination manager that dropped the entropy-coded payload
/// shrinks `outsize` by an order of magnitude and pushes `meanabsdiff` far
/// past the reference, so this catches exactly the failure that marker
/// bookends cannot.
fn compare_metrics(scenario: &str, ours: &str, theirs: &str) {
    let mine: Vec<(String, f64)> = metrics(ours);
    let reference: Vec<(String, f64)> = metrics(theirs);
    assert_eq!(
        mine.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        reference
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>(),
        "scenario `{scenario}`: the two legs reported different metric keys"
    );
    for ((key, ours_v), (_, ref_v)) in mine.iter().zip(reference.iter()) {
        // meanabsdiff can legitimately be 0.0-ish; guard the ratio.
        let lo: f64 = ref_v * 0.5 - 0.5;
        let hi: f64 = ref_v * 2.0 + 0.5;
        assert!(
            *ours_v >= lo && *ours_v <= hi,
            "scenario `{scenario}`: metric `{key}` = {ours_v} is outside \
             [{lo}, {hi}] derived from the reference value {ref_v}"
        );
    }
}

// ---------- scenarios ----------

#[test]
fn mem_dest_sufficient_stack_buffer_is_not_moved_or_freed() {
    assert_scenario("mem_sufficient_stack");
}

#[test]
fn mem_dest_sufficient_static_buffer_is_not_moved_or_freed() {
    assert_scenario("mem_sufficient_static");
}

#[test]
fn mem_dest_insufficient_buffer_grows_without_freeing_caller_memory() {
    assert_scenario("mem_grow");
}

#[test]
fn mem_dest_null_buffer_is_allocated_by_the_library() {
    assert_scenario("mem_null_alloc");
}

#[test]
fn mem_dest_rejects_null_out_parameters() {
    assert_scenario("mem_null_args");
}

#[test]
fn mem_dest_zero_capacity_buffer_is_replaced_not_written_into() {
    assert_scenario("mem_zero_capacity");
}

#[test]
fn mem_dest_handle_reuse_writes_two_independent_images() {
    assert_scenario("mem_reuse");
}

#[test]
fn mem_dest_rejects_a_foreign_installed_destination_manager() {
    assert_scenario("dest_switch");
}

#[test]
fn stdio_dest_reports_short_write_from_term_destination() {
    assert_scenario("stdio_write_error_small");
}

#[test]
fn stdio_dest_reports_short_write_from_empty_output_buffer() {
    assert_scenario("stdio_write_error_large");
}

#[test]
fn stdio_dest_reports_flush_failure_on_a_full_device() {
    assert_scenario("stdio_dev_full");
}

#[test]
fn stdio_dest_flushes_before_returning_from_finish_compress() {
    assert_scenario("stdio_success");
}

// ---------- the canary ----------

const CANARY_C: &str = r##"
/*
 * P4-108 destination-manager ownership / I-O canary.
 *
 * Every scenario prints implementation-independent `key=value` contract facts
 * so the same binary can be diffed between our cdylib and a reference
 * libjpeg-turbo v8 build. Byte counts are deliberately never printed: two
 * conforming encoders produce different sizes, only the contract must match.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include "jpeglib.h"
#include "jerror.h"

#if JPEG_LIB_VERSION != 80
#error "P4-108 canary must be compiled against the classic v8 ABI"
#endif

#define IMG_W 64
#define IMG_H 64
/* Large enough that the encoded stream exceeds libjpeg's 4096-byte stdio
 * staging buffer, so `empty_output_buffer` runs at least once. */
#define BIG_W 320
#define BIG_H 320

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      return 1;                                                             \
    }                                                                       \
  } while (0)

struct canary_err {
  struct jpeg_error_mgr pub;
  jmp_buf jb;
  int code;
  int fired;
};

static void canary_error_exit(j_common_ptr cinfo)
{
  struct canary_err *e = (struct canary_err *)cinfo->err;
  e->code = cinfo->err->msg_code;
  e->fired = 1;
  longjmp(e->jb, 1);
}

static void canary_output_message(j_common_ptr cinfo)
{
  (void)cinfo;
}

static void install_err(struct jpeg_compress_struct *cinfo,
                        struct canary_err *e)
{
  cinfo->err = jpeg_std_error(&e->pub);
  e->pub.error_exit = canary_error_exit;
  e->pub.output_message = canary_output_message;
  e->code = -1;
  e->fired = 0;
}

static void install_derr(struct jpeg_decompress_struct *dinfo,
                         struct canary_err *e)
{
  dinfo->err = jpeg_std_error(&e->pub);
  e->pub.error_exit = canary_error_exit;
  e->pub.output_message = canary_output_message;
  e->code = -1;
  e->fired = 0;
}

static unsigned char *make_pixels(int w, int h)
{
  int x, y;
  unsigned char *px = (unsigned char *)malloc((size_t)w * (size_t)h * 3);
  if (px == NULL)
    return NULL;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      unsigned char *p = px + ((size_t)y * (size_t)w + (size_t)x) * 3;
      p[0] = (unsigned char)(x * 3 + y);
      p[1] = (unsigned char)(y * 5 + x);
      p[2] = (unsigned char)((x ^ y) * 7);
    }
  }
  return px;
}

static void compress_image(struct jpeg_compress_struct *cinfo,
                           unsigned char *px, int w, int h)
{
  JSAMPROW row[1];
  cinfo->image_width = (JDIMENSION)w;
  cinfo->image_height = (JDIMENSION)h;
  cinfo->input_components = 3;
  cinfo->in_color_space = JCS_RGB;
  jpeg_set_defaults(cinfo);
  jpeg_set_quality(cinfo, 90, TRUE);
  jpeg_start_compress(cinfo, TRUE);
  while (cinfo->next_scanline < cinfo->image_height) {
    row[0] = (JSAMPROW)(px + (size_t)cinfo->next_scanline * (size_t)w * 3);
    (void)jpeg_write_scanlines(cinfo, row, 1);
  }
  jpeg_finish_compress(cinfo);
}

static int is_complete_jpeg(const unsigned char *buf, unsigned long len)
{
  if (buf == NULL || len < 4)
    return 0;
  if (buf[0] != 0xFF || buf[1] != 0xD8)
    return 0;
  if (buf[len - 2] != 0xFF || buf[len - 1] != 0xD9)
    return 0;
  return 1;
}

/* Marker bookends prove nothing about the payload: a destination manager that
 * dropped every entropy-coded byte in between still produces SOI...EOI. So
 * decode the result and compare it to the pixels we fed in. This runs inside
 * the canary, against whichever libjpeg it is linked to, so both legs verify
 * their own output.
 *
 * Returns the mean absolute per-sample difference, or -1 on any decode
 * failure / dimension mismatch. */
static double decoded_mean_abs_diff(const unsigned char *jpeg, unsigned long len,
                                    const unsigned char *src, int w, int h)
{
  struct jpeg_decompress_struct dinfo;
  struct canary_err jerr;
  unsigned char *row = NULL;
  JSAMPROW rows[1];
  double total = 0.0;
  unsigned long samples = 0;
  int y;

  install_derr(&dinfo, &jerr);
  if (setjmp(jerr.jb)) {
    free(row);
    jpeg_destroy_decompress(&dinfo);
    return -1.0;
  }
  jpeg_create_decompress(&dinfo);
  jpeg_mem_src(&dinfo, (unsigned char *)jpeg, len);
  if (jpeg_read_header(&dinfo, TRUE) != JPEG_HEADER_OK)
    return -1.0;
  dinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&dinfo);
  if ((int)dinfo.output_width != w || (int)dinfo.output_height != h ||
      dinfo.output_components != 3)
    return -1.0;

  row = (unsigned char *)malloc((size_t)w * 3);
  if (row == NULL)
    return -1.0;
  y = 0;
  while (dinfo.output_scanline < dinfo.output_height) {
    int x;
    rows[0] = row;
    if (jpeg_read_scanlines(&dinfo, rows, 1) != 1)
      break;
    for (x = 0; x < w * 3; x++) {
      int a = row[x];
      int b = src[(size_t)y * (size_t)w * 3 + (size_t)x];
      total += (double)(a > b ? a - b : b - a);
      samples++;
    }
    y++;
  }
  free(row);
  jpeg_finish_decompress(&dinfo);
  jpeg_destroy_decompress(&dinfo);
  if (y != h || samples == 0)
    return -1.0;
  return total / (double)samples;
}

/* Assert the encoded stream is both well-formed and carries the image. The
 * `#`-prefixed lines are metrics, not contract facts: the Rust driver compares
 * them across legs with a tolerance rather than for string equality, because
 * two conforming encoders legitimately differ on size and exact error. */
static int check_payload(const char *tag, const unsigned char *buf,
                         unsigned long len, const unsigned char *src, int w, int h)
{
  double mad;
  if (!is_complete_jpeg(buf, len)) {
    fprintf(stderr, "FAIL %s: not a complete JPEG (len=%lu)\n", tag, len);
    return 1;
  }
  mad = decoded_mean_abs_diff(buf, len, src, w, h);
  if (mad < 0.0) {
    fprintf(stderr, "FAIL %s: output did not decode back to %dx%d RGB\n", tag, w, h);
    return 1;
  }
  /* Measured, not guessed: this fixture is deliberately high-frequency
   * ((x^y)*7 in blue), so default 4:2:0 chroma subsampling at q=90 costs a
   * real amount. Both our shim and libjpeg-turbo 3.1.4.1 measure 9.180 on it;
   * 11.0 leaves ~20% headroom for SIMD-path rounding differences across
   * architectures. A destination manager that dropped the entropy-coded
   * payload lands far above this — the surviving image would be flat grey.
   * The sharp check is the cross-leg metric comparison in the Rust driver;
   * this bound only has to catch gross corruption in a single leg. */
  if (mad > 11.0) {
    fprintf(stderr, "FAIL %s: decoded mean abs diff %.3f exceeds 11.0\n", tag, mad);
    return 1;
  }
  printf("payload_decodes=1\n");
  printf("#%s_outsize=%lu\n", tag, len);
  printf("#%s_meanabsdiff=%.3f\n", tag, mad);
  return 0;
}

static unsigned char static_sink[65536];

/* Scenarios 1 + 2: a caller buffer big enough for the whole stream is used in
 * place. libjpeg must not move it and must not free it (jdatadst.c:231-233).
 * Passing stack and static storage makes an erroneous free() an immediate
 * allocator abort rather than a silent heap corruption. */
static int scn_mem_sufficient(int use_static)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  unsigned char stack_sink[sizeof static_sink];
  unsigned char *sink = use_static ? static_sink : stack_sink;
  unsigned char *out;
  unsigned long outsize;
  unsigned char *px;

  memset(sink, 0xA5, sizeof static_sink);
  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb)) {
    fprintf(stderr, "FAIL unexpected error_exit, msg_code=%d\n", jerr.code);
    return 1;
  }
  jpeg_create_compress(&cinfo);
  out = sink;
  outsize = (unsigned long)sizeof static_sink;
  jpeg_mem_dest(&cinfo, &out, &outsize);
  CHECK(out == sink);
  CHECK(outsize == (unsigned long)sizeof static_sink);

  compress_image(&cinfo, px, IMG_W, IMG_H);

  CHECK(out == sink);
  CHECK(outsize > 4);
  CHECK(outsize < (unsigned long)sizeof static_sink);
  if (check_payload("sufficient", sink, outsize, px, IMG_W, IMG_H) != 0)
    return 1;
  /* Nothing beyond the reported length may be disturbed, and the stream must
   * start at offset 0 rather than after a phantom capacity-sized prefix. */
  CHECK(sink[outsize] == 0xA5);
  CHECK(sink[sizeof static_sink - 1] == 0xA5);

  jpeg_destroy_compress(&cinfo);
  free(px);
  printf("buffer_moved=0\n");
  printf("soi_at_offset_0=1\n");
  printf("outsize_within_capacity=1\n");
  printf("tail_untouched=1\n");
  return 0;
}

/* Scenario 3: a caller buffer too small to hold the stream. libjpeg allocates
 * its own larger buffer, copies the prefix already written into the caller's
 * buffer, and leaves that buffer for the caller to free. */
static int scn_mem_grow(void)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  const size_t cap = 128;
  unsigned char *caller;
  unsigned char *out;
  unsigned long outsize;
  unsigned char *px;

  caller = (unsigned char *)malloc(cap);
  CHECK(caller != NULL);
  memset(caller, 0xA5, cap);
  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb)) {
    fprintf(stderr, "FAIL unexpected error_exit, msg_code=%d\n", jerr.code);
    return 1;
  }
  jpeg_create_compress(&cinfo);
  out = caller;
  outsize = (unsigned long)cap;
  jpeg_mem_dest(&cinfo, &out, &outsize);
  CHECK(out == caller);

  compress_image(&cinfo, px, IMG_W, IMG_H);

  CHECK(out != caller);
  CHECK(outsize > (unsigned long)cap);
  if (check_payload("grown", out, outsize, px, IMG_W, IMG_H) != 0)
    return 1;
  /* The bytes libjpeg had already placed in the caller's buffer must reappear
   * at the head of the grown buffer, and must still be readable in the
   * caller's buffer — proof it was copied, not freed and reused.
   *
   * Note this deliberately reads `caller` under the hypothesis that the
   * library may have freed it. That is the point: under a sanitizer the read
   * aborts and the canary fails loudly, and without one the following free()
   * catches the same defect as a double free. */
  CHECK(memcmp(out, caller, cap) == 0);

  jpeg_destroy_compress(&cinfo);
  /* Aborts here if the library freed the caller's buffer. */
  free(caller);
  free(out);
  free(px);
  printf("grew_into_library_buffer=1\n");
  printf("encoded_prefix_preserved=1\n");
  printf("caller_buffer_still_owned=1\n");
  return 0;
}

/* Scenario 4b: NULL out-parameters are a caller bug, not a request to
 * allocate. Upstream rejects them before touching anything (jdatadst.c:242). */
static int scn_mem_null_args(void)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  unsigned long outsize = 0;

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb) == 0) {
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, NULL, &outsize);
    fprintf(stderr, "FAIL jpeg_mem_dest accepted a NULL outbuffer\n");
    return 1;
  }
  CHECK(jerr.fired == 1);
  CHECK(jerr.code == JERR_BUFFER_SIZE);
  printf("null_outbuffer_rejected=1\n");
  printf("msg_code=%d\n", jerr.code);
  return 0;
}

/* Scenario 4: `*outbuffer == NULL` — libjpeg allocates immediately inside
 * jpeg_mem_dest and publishes the capacity in `*outsize` (jdatadst.c:267-273),
 * long before any compression happens. */
static int scn_mem_null_alloc(void)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  unsigned char *out;
  unsigned long outsize;
  unsigned char *px;
  unsigned long initial_capacity;

  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb)) {
    fprintf(stderr, "FAIL unexpected error_exit, msg_code=%d\n", jerr.code);
    return 1;
  }
  jpeg_create_compress(&cinfo);
  out = NULL;
  outsize = 0;
  jpeg_mem_dest(&cinfo, &out, &outsize);
  CHECK(out != NULL);
  CHECK(outsize > 0);
  initial_capacity = outsize;

  compress_image(&cinfo, px, IMG_W, IMG_H);

  /* term_destination must have replaced the advertised capacity with the
   * real byte count; leaving it at the capacity is the classic silent bug. */
  CHECK(outsize != initial_capacity);
  if (check_payload("null_alloc", out, outsize, px, IMG_W, IMG_H) != 0)
    return 1;

  jpeg_destroy_compress(&cinfo);
  free(out);
  free(px);
  printf("allocated_at_setup=1\n");
  printf("initial_capacity=%lu\n", initial_capacity);
  printf("complete_jpeg=1\n");
  return 0;
}

/* Scenario 5: a non-NULL buffer advertised with zero capacity is not a usable
 * buffer. libjpeg replaces it with its own allocation and never writes through
 * the caller's pointer. */
static int scn_mem_zero_capacity(void)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  unsigned char *caller;
  unsigned char *out;
  unsigned long outsize;
  unsigned char *px;
  unsigned long initial_capacity;

  caller = (unsigned char *)malloc(64);
  CHECK(caller != NULL);
  memset(caller, 0xA5, 64);
  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb)) {
    fprintf(stderr, "FAIL unexpected error_exit, msg_code=%d\n", jerr.code);
    return 1;
  }
  jpeg_create_compress(&cinfo);
  out = caller;
  outsize = 0;
  jpeg_mem_dest(&cinfo, &out, &outsize);
  CHECK(out != NULL);
  CHECK(out != caller);
  CHECK(outsize > 0);
  initial_capacity = outsize;

  compress_image(&cinfo, px, IMG_W, IMG_H);

  if (check_payload("zero_capacity", out, outsize, px, IMG_W, IMG_H) != 0)
    return 1;
  /* The whole caller block must be untouched, not just its first byte. */
  {
    int i;
    for (i = 0; i < 64; i++)
      CHECK(caller[i] == 0xA5);
  }

  jpeg_destroy_compress(&cinfo);
  free(caller);
  free(out);
  free(px);
  printf("zero_capacity_replaced=1\n");
  printf("initial_capacity=%lu\n", initial_capacity);
  printf("caller_buffer_untouched=1\n");
  return 0;
}

/* Scenario 6: one compress handle, two images. Re-executing jpeg_mem_dest
 * rebinds the destination without leaking the previous output or reporting
 * the previous image's bytes. */
static int scn_mem_reuse(void)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  unsigned char *out1, *out2;
  unsigned long size1, size2;
  unsigned char *px;

  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb)) {
    fprintf(stderr, "FAIL unexpected error_exit, msg_code=%d\n", jerr.code);
    return 1;
  }
  jpeg_create_compress(&cinfo);

  out1 = NULL;
  size1 = 0;
  jpeg_mem_dest(&cinfo, &out1, &size1);
  compress_image(&cinfo, px, IMG_W, IMG_H);
  if (check_payload("reuse_first", out1, size1, px, IMG_W, IMG_H) != 0)
    return 1;

  out2 = NULL;
  size2 = 0;
  jpeg_mem_dest(&cinfo, &out2, &size2);
  compress_image(&cinfo, px, IMG_W, IMG_H);
  if (check_payload("reuse_second", out2, size2, px, IMG_W, IMG_H) != 0)
    return 1;

  /* Distinct buffers, and the second image reports only its own bytes — a
   * destination that kept appending would report a larger second size. */
  CHECK(out1 != out2);
  CHECK(size1 == size2);
  CHECK(memcmp(out1, out2, (size_t)size1) == 0);

  jpeg_destroy_compress(&cinfo);
  free(out1);
  free(out2);
  free(px);
  printf("second_image_complete=1\n");
  printf("buffers_distinct=1\n");
  printf("sizes_equal=1\n");
  return 0;
}

/* Scenario 7: an already-installed destination manager that jpeg_mem_dest did
 * not create cannot be reinterpreted as a memory destination — its private
 * area may be a different size (jdatadst.c:252-257). */
static int scn_dest_switch(const char *dir)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  volatile unsigned char *out;
  volatile unsigned long outsize;
  char path[1024];
  FILE *f;

  snprintf(path, sizeof path, "%s/switch.jpg", dir);
  f = fopen(path, "wb");
  CHECK(f != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb) == 0) {
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, f);
    out = NULL;
    outsize = 0;
    jpeg_mem_dest(&cinfo, (unsigned char **)&out, (unsigned long *)&outsize);
    fprintf(stderr, "FAIL jpeg_mem_dest accepted a stdio destination manager\n");
    return 1;
  }
  CHECK(jerr.fired == 1);
  CHECK(jerr.code == JERR_BUFFER_SIZE);
  fclose(f);
  printf("rejected=1\n");
  printf("msg_code=%d\n", jerr.code);
  return 0;
}

/* Scenarios 8 + 9: every short write must surface as JERR_FILE_WRITE. A stream
 * opened read-only accepts no bytes at all, so `small` exercises the final
 * flush inside term_destination and `large` exercises empty_output_buffer. */
static int scn_stdio_write_error(const char *dir, int large)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  char path[1024];
  FILE *f;
  unsigned char *px;
  int w = large ? BIG_W : IMG_W;
  int h = large ? BIG_H : IMG_H;

  snprintf(path, sizeof path, "%s/readonly-%d.bin", dir, large);
  f = fopen(path, "wb");
  CHECK(f != NULL);
  fclose(f);
  f = fopen(path, "rb");
  CHECK(f != NULL);

  px = make_pixels(w, h);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb) == 0) {
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, f);
    compress_image(&cinfo, px, w, h);
    fprintf(stderr, "FAIL compression to an unwritable stream reported success\n");
    return 1;
  }
  CHECK(jerr.fired == 1);
  CHECK(jerr.code == JERR_FILE_WRITE);
  jpeg_destroy_compress(&cinfo);
  fclose(f);
  free(px);
  printf("write_error_reported=1\n");
  printf("msg_code=%d\n", jerr.code);
  return 0;
}

/* Scenario 10: on /dev/full every fwrite "succeeds" into the stdio buffer and
 * the failure only appears at fflush/ferror time. Linux-only; elsewhere the
 * scenario reports that it did not run rather than silently passing. */
static int scn_stdio_dev_full(void)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  FILE *f;
  unsigned char *px;

  f = fopen("/dev/full", "wb");
  if (f == NULL) {
    printf("dev_full_available=0\n");
    return 0;
  }
  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb) == 0) {
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, f);
    compress_image(&cinfo, px, IMG_W, IMG_H);
    fprintf(stderr, "FAIL compression to /dev/full reported success\n");
    return 1;
  }
  CHECK(jerr.fired == 1);
  CHECK(jerr.code == JERR_FILE_WRITE);
  jpeg_destroy_compress(&cinfo);
  fclose(f);
  free(px);
  printf("dev_full_available=1\n");
  printf("write_error_reported=1\n");
  printf("msg_code=%d\n", jerr.code);
  return 0;
}

/* Scenario 11: the success control. term_destination must fflush, so the whole
 * stream is on disk when jpeg_finish_compress returns — before the caller
 * closes the stream — and the stream must carry no error flag. */
static int scn_stdio_success(const char *dir)
{
  struct jpeg_compress_struct cinfo;
  struct canary_err jerr;
  char path[1024];
  FILE *f;
  FILE *probe;
  unsigned char *px;
  long on_disk;
  unsigned char head[2];
  unsigned char tail[2];

  snprintf(path, sizeof path, "%s/ok.jpg", dir);
  f = fopen(path, "wb");
  CHECK(f != NULL);
  px = make_pixels(IMG_W, IMG_H);
  CHECK(px != NULL);

  install_err(&cinfo, &jerr);
  if (setjmp(jerr.jb)) {
    fprintf(stderr, "FAIL unexpected error_exit, msg_code=%d\n", jerr.code);
    return 1;
  }
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, f);
  compress_image(&cinfo, px, IMG_W, IMG_H);

  CHECK(ferror(f) == 0);
  /* Read the file through a second handle while the writing handle is still
   * open: anything still sitting in the stdio buffer would be invisible. */
  probe = fopen(path, "rb");
  CHECK(probe != NULL);
  CHECK(fseek(probe, 0, SEEK_END) == 0);
  on_disk = ftell(probe);
  CHECK(on_disk > 4);
  CHECK(fseek(probe, 0, SEEK_SET) == 0);
  CHECK(fread(head, 1, 2, probe) == 2);
  CHECK(fseek(probe, -2, SEEK_END) == 0);
  CHECK(fread(tail, 1, 2, probe) == 2);
  CHECK(fseek(probe, 0, SEEK_SET) == 0);
  {
    unsigned char *whole = (unsigned char *)malloc((size_t)on_disk);
    CHECK(whole != NULL);
    CHECK(fread(whole, 1, (size_t)on_disk, probe) == (size_t)on_disk);
    if (check_payload("stdio", whole, (unsigned long)on_disk, px, IMG_W, IMG_H) != 0)
      return 1;
    free(whole);
  }
  fclose(probe);
  CHECK(head[0] == 0xFF && head[1] == 0xD8);
  CHECK(tail[0] == 0xFF && tail[1] == 0xD9);

  jpeg_destroy_compress(&cinfo);
  fclose(f);
  free(px);
  printf("flushed_before_fclose=1\n");
  printf("stream_error_flag=0\n");
  printf("complete_jpeg=1\n");
  return 0;
}

int main(int argc, char **argv)
{
  const char *scenario;
  const char *dir;

  if (argc < 3) {
    fprintf(stderr, "usage: %s <scenario> <workdir>\n", argv[0]);
    return 2;
  }
  scenario = argv[1];
  dir = argv[2];

  if (strcmp(scenario, "mem_sufficient_stack") == 0)
    return scn_mem_sufficient(0);
  if (strcmp(scenario, "mem_sufficient_static") == 0)
    return scn_mem_sufficient(1);
  if (strcmp(scenario, "mem_grow") == 0)
    return scn_mem_grow();
  if (strcmp(scenario, "mem_null_alloc") == 0)
    return scn_mem_null_alloc();
  if (strcmp(scenario, "mem_null_args") == 0)
    return scn_mem_null_args();
  if (strcmp(scenario, "mem_zero_capacity") == 0)
    return scn_mem_zero_capacity();
  if (strcmp(scenario, "mem_reuse") == 0)
    return scn_mem_reuse();
  if (strcmp(scenario, "dest_switch") == 0)
    return scn_dest_switch(dir);
  if (strcmp(scenario, "stdio_write_error_small") == 0)
    return scn_stdio_write_error(dir, 0);
  if (strcmp(scenario, "stdio_write_error_large") == 0)
    return scn_stdio_write_error(dir, 1);
  if (strcmp(scenario, "stdio_dev_full") == 0)
    return scn_stdio_dev_full();
  if (strcmp(scenario, "stdio_success") == 0)
    return scn_stdio_success(dir);

  fprintf(stderr, "unknown scenario `%s`\n", scenario);
  return 2;
}
"##;
