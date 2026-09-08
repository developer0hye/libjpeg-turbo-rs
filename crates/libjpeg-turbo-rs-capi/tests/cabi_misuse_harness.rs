//! P4-141 criterion 3, second half: the process-isolated C-ABI harness.
//!
//! `examples/cabi_misuse_harness.c` is a C driver that `dlopen`s one shared
//! library, runs **one named case**, and exits. This file spawns it: once per
//! case against our cdylib, and once more against a stock `libturbojpeg` when
//! one is installed, requiring the two stdout transcripts to agree.
//!
//! **Why a child process per case.** The scenarios this criterion names are the
//! ones that may fault — an overrun past a destination buffer, a handle used
//! after teardown, a sanitizer report at the FFI boundary. In one process the
//! first case that misbehaves takes every later case with it and the cases that
//! are *supposed* to fault cannot be written at all. A child per case turns a
//! fault into an exit status attributed to exactly one scenario, which is what
//! makes `selftest_guard_page` — a deliberate one-byte overrun that must die by
//! a signal — a test rather than a crash.
//!
//! **Why differential.** Asserting our own return codes would pin current
//! behaviour, which is the failure mode P4-141 exists to retire. The harness
//! takes the library path on the command line, so the *same binary* drives both
//! implementations and the comparison is between two transcripts rather than
//! between behaviour and a comment. Where no install is present the per-case
//! contract assertions below still run; they are read off upstream's source,
//! cited per case, not off our output.
//!
//! **The three divergences this harness found on its first run** are in
//! [`KNOWN_DIVERGENCES`], each keyed to the LAST_MILE item that owns it, and
//! each required to *still* diverge by
//! [`known_divergences_are_live`] — so a fix deletes its entry instead of
//! leaving a stale exemption behind.
//!
//! **The C driver carries its own invariant checks, and they decide its exit
//! status.** `sanitizers.yml` runs the cases without this runner, reading only
//! the exit status, so a case that merely *printed* `canary=corrupt` and
//! returned 0 would leave that leg green on real corruption — `codex review`
//! demonstrated exactly that with an injected short-stride write. The driver's
//! `require()` covers only what holds on *both* implementations; everything in
//! [`KNOWN_DIVERGENCES`] is deliberately left to the comparison below, since
//! failing on it would turn the oracle run red. The leading canary is checked
//! by `guarded_free` itself rather than by each case, so a case that forgets to
//! ask is still covered — which is how the one buffer no case happened to check
//! was found.
//!
//! Sanitizer coverage: `.github/workflows/sanitizers.yml`'s `c_boundary_asan`
//! job runs every case in [`CASES`], plus `selftest_guard_page`, against an
//! ASan-instrumented cdylib; `selftest_canary` runs here only. Before this
//! harness existed that job resolved five `tj3*` symbols over three fixtures,
//! allocated with its own `malloc`/`free` and touched no compress and no 12-bit
//! path — the gaps recorded under P4-141 criterion 2, which `alloc_ownership`,
//! `undersized_output` and `precision12` close.

// dlopen, mmap guard pages and pthreads are POSIX; the harness is not built on
// Windows. The C-ABI surface itself is covered there by the rest of this
// crate's suites.
#![cfg(unix)]

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

mod helpers;

#[path = "support/cdylib.rs"]
mod cdylib;

/// Cases that take a JPEG fixture and are compared against the stock library.
const CASES: &[&str] = &[
    "lifecycle",
    "handle_defaults",
    "parameter_applicability",
    "null_handle",
    "pitch_boundaries",
    "undersized_output",
    "max_dimensions",
    "concurrent_handles",
    "alloc_ownership",
    "precision12",
];

/// Cases that exercise the harness's own instrumentation rather than the
/// library. They are not run against the oracle: what they assert is that the
/// guard page and the canary are armed, which is a property of this harness.
const SELFTESTS: &[&str] = &["selftest_guard_page", "selftest_canary"];

/// A transcript line our shim and stock TurboJPEG are known to disagree on.
///
/// Every entry names the item that owns the divergence and must **still**
/// diverge — [`known_divergences_are_live`] fails when one stops, so closing
/// the item deletes the entry rather than leaving an exemption that outlives
/// the defect.
struct KnownDivergence {
    case: &'static str,
    key: &'static str,
    item: &'static str,
}

const KNOWN_DIVERGENCES: &[KnownDivergence] = &[
    // P4-200 (#621) gains a criterion for the initial values: upstream seeds
    // `jpegWidth`/`jpegHeight` to -1 in `tj3InitVersion`
    // (`references/libjpeg-turbo/src/turbojpeg.c:600-601`); ours leave them 0,
    // so a caller using the documented -1 sentinel for "not read yet" sees a
    // legal dimension instead.
    KnownDivergence {
        case: "handle_defaults",
        key: "default_compress_JPEGWIDTH",
        item: "P4-200 (#621)",
    },
    KnownDivergence {
        case: "handle_defaults",
        key: "default_compress_JPEGHEIGHT",
        item: "P4-200 (#621)",
    },
    KnownDivergence {
        case: "handle_defaults",
        key: "default_decompress_JPEGWIDTH",
        item: "P4-200 (#621)",
    },
    KnownDivergence {
        case: "handle_defaults",
        key: "default_decompress_JPEGHEIGHT",
        item: "P4-200 (#621)",
    },
    KnownDivergence {
        case: "handle_defaults",
        key: "default_transform_JPEGWIDTH",
        item: "P4-200 (#621)",
    },
    KnownDivergence {
        case: "handle_defaults",
        key: "default_transform_JPEGHEIGHT",
        item: "P4-200 (#621)",
    },
    // P4-202 (#624): `tj3Compress8` accepts a width or height above libjpeg's
    // JPEG_MAX_DIMENSION and emits a frame stock `djpeg` refuses to read.
    KnownDivergence {
        case: "max_dimensions",
        key: "compress_over_max_width_rc",
        item: "P4-202 (#624)",
    },
    KnownDivergence {
        case: "max_dimensions",
        key: "compress_over_max_height_rc",
        item: "P4-202 (#624)",
    },
    // P4-203 (#625): `TJPARAM_PRECISION` reports the decode path's output
    // precision, not the SOF's `data_precision`.
    KnownDivergence {
        case: "precision12",
        key: "precision",
        item: "P4-203 (#625)",
    },
    // P4-205 (#627): `tj3Alloc(0)` returns NULL; upstream is a bare
    // `malloc(bytes)` (`turbojpeg.c:935-938`), which returns a unique freeable
    // pointer for 0 on every platform this project ships to.
    KnownDivergence {
        case: "alloc_ownership",
        key: "alloc_zero",
        item: "P4-205 (#627)",
    },
    // P4-206 (#628): under TJPARAM_NOREALLOC we accept a buffer exactly the
    // size of the output; upstream needs one byte more, because libjpeg's
    // `emit_byte` calls `dump_buffer` when `free_in_buffer` hits zero and
    // `empty_output_buffer` with `alloc == FALSE` is `JERR_BUFFER_SIZE`.
    KnownDivergence {
        case: "undersized_output",
        key: "norealloc_exact_rc",
        item: "P4-206 (#628)",
    },
    KnownDivergence {
        case: "undersized_output",
        key: "norealloc_exact_fits",
        item: "P4-206 (#628)",
    },
    KnownDivergence {
        case: "undersized_output",
        key: "norealloc_exact_soi",
        item: "P4-206 (#628)",
    },
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture() -> PathBuf {
    workspace_root()
        .join("references")
        .join("libjpeg-turbo")
        .join("testimages")
        .join("testorig.jpg")
}

/// Compile the C harness once per test binary.
///
/// A compile failure is fatal rather than a skip: every case in this file goes
/// through it, so a silent skip would turn the whole suite green while running
/// nothing.
fn harness_binary() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| {
        let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("cabi_misuse_harness.c");
        assert!(source.is_file(), "harness source missing: {source:?}");

        let out_dir: PathBuf =
            std::env::temp_dir().join(format!("cabi_misuse_harness_{}", std::process::id()));
        std::fs::create_dir_all(&out_dir).expect("create harness build dir");
        let binary: PathBuf = out_dir.join("cabi_misuse_harness");

        let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let output = Command::new(&compiler)
            .args(["-O1", "-g", "-Wall", "-Wextra"])
            .arg("-o")
            .arg(&binary)
            .arg(&source)
            .args(["-ldl", "-lpthread"])
            .output()
            .unwrap_or_else(|error| panic!("failed to run {compiler}: {error}"));
        assert!(
            output.status.success(),
            "failed to compile {source:?} with {compiler}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        binary
    })
}

fn our_cdylib() -> PathBuf {
    cdylib::cdylib_path()
}

/// The stock `libturbojpeg` shared object, when a TurboJPEG 3 development
/// install is present.
///
/// The harness `dlopen`s it, so a static archive is not usable here even
/// though [`helpers::find_turbojpeg_dev`] accepts one for the link-time
/// oracles. `LIBJPEG_TURBO_PREFIX` still makes the comparison mandatory: the
/// CI step that runs this crate's tests sets it, and a prefix that turns out to
/// ship no shared object is a broken provisioning step, not a machine without
/// TurboJPEG.
fn stock_turbojpeg() -> Option<PathBuf> {
    let Some(install) = helpers::find_turbojpeg_dev() else {
        // The same fail-closed rule `helpers::build_oracle` applies: an
        // explicit prefix is a statement that an install is provisioned, and
        // returning None here would leave both comparisons green while
        // comparing nothing.
        assert!(
            !helpers::oracle_is_required(),
            "no TurboJPEG 3 development install found under \
             LIBJPEG_TURBO_PREFIX={:?} — that variable says one is \
             provisioned, and skipping the differential comparison would pass \
             this gate without checking anything",
            std::env::var_os("LIBJPEG_TURBO_PREFIX")
        );
        return None;
    };
    let candidates: [&str; 4] = [
        "libturbojpeg.so",
        "libturbojpeg.dylib",
        "libturbojpeg.so.0",
        "libturbojpeg.0.dylib",
    ];
    let found: Option<PathBuf> = candidates
        .iter()
        .map(|name| install.lib_dir.join(name))
        .find(|path| path.exists());
    assert!(
        found.is_some() || !helpers::oracle_is_required(),
        "LIBJPEG_TURBO_PREFIX names an install whose {:?} holds no loadable \
         libturbojpeg; this harness dlopens the oracle, so a static-only \
         install cannot serve it",
        install.lib_dir
    );
    let library: PathBuf = found?;
    if !exports_tj3_init_version(&library) {
        // `docs/oracle_versions.tsv` pins `tool-current` at 3.2.0 and that is
        // what the contract here is read against. The distinction is not
        // cosmetic: 3.1.4.1 *accepts* a pitch below `width * pixelSize` and
        // then writes rows at that stride, which `pitch_boundaries` catches
        // with a guard page — an upstream defect fixed in 3.2.0 and not a
        // statement about this port. Comparing against it would report
        // upstream's bug as ours.
        assert!(
            !helpers::oracle_is_required(),
            "LIBJPEG_TURBO_PREFIX names {library:?}, which does not export \
             tj3InitVersion and is therefore older than the 3.2.0 \
             `tool-current` pin in docs/oracle_versions.tsv; this harness \
             compares against that release's contract"
        );
        eprintln!(
            "SKIP: {library:?} predates TurboJPEG 3.2 (no tj3InitVersion); \
             set LIBJPEG_TURBO_PREFIX to a 3.2.0 install to run the \
             differential comparison"
        );
        return None;
    }
    Some(library)
}

/// `tj3InitVersion` is TurboJPEG 3.2's addition (`turbojpeg.c:583`), so
/// resolving it is a mechanical "is this oracle at least the pinned
/// `tool-current` release?" that needs no version string to parse.
fn exports_tj3_init_version(library: &Path) -> bool {
    let loaded = match unsafe { libloading::Library::new(library) } {
        Ok(loaded) => loaded,
        Err(error) => panic!("the oracle {library:?} does not load: {error}"),
    };
    let symbol: Result<libloading::Symbol<*const ()>, _> =
        unsafe { loaded.get(b"tj3InitVersion\0") };
    symbol.is_ok()
}

struct Run {
    stdout: String,
    stderr: String,
    code: Option<i32>,
    signal: Option<i32>,
}

impl Run {
    fn outcome(&self) -> String {
        match (self.code, self.signal) {
            (Some(code), _) => format!("exit {code}"),
            (_, Some(signal)) => format!("killed by signal {signal}"),
            _ => "no status".to_string(),
        }
    }

    /// The transcript as ordered `key=value` pairs. Order is part of the
    /// comparison: a case that stops early drops later keys, and that must read
    /// as a difference rather than as an unchanged subset.
    fn transcript(&self) -> Vec<(String, String)> {
        self.stdout
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| match line.split_once('=') {
                Some((key, value)) => (key.to_string(), value.to_string()),
                None => panic!("harness printed a line that is not key=value: {line:?}"),
            })
            .collect()
    }

    fn value(&self, key: &str) -> String {
        self.transcript()
            .into_iter()
            .find(|(name, _)| name == key)
            .unwrap_or_else(|| {
                panic!(
                    "transcript has no `{key}` line; it printed:\n{}",
                    self.stdout
                )
            })
            .1
    }
}

fn run(library: &Path, case: &str) -> Run {
    let output = Command::new(harness_binary())
        .arg(library)
        .arg(case)
        .arg(fixture())
        .output()
        .unwrap_or_else(|error| panic!("failed to spawn the harness for `{case}`: {error}"));
    Run {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        code: output.status.code(),
        signal: output.status.signal(),
    }
}

/// Run a case against our shim and require it to have completed cleanly.
///
/// Empty stderr is part of "cleanly" on purpose: a sanitizer diagnostic in
/// *recovering* mode prints and lets the process exit 0, so an exit-code check
/// alone would report a UBSan finding as a pass.
fn run_ours(case: &str) -> Run {
    let outcome = run(&our_cdylib(), case);
    assert_eq!(
        outcome.code,
        Some(0),
        "case `{case}` against our cdylib: {}\nstdout:\n{}\nstderr:\n{}",
        outcome.outcome(),
        outcome.stdout,
        outcome.stderr
    );
    assert!(
        outcome.stderr.is_empty(),
        "case `{case}` exited 0 but wrote to stderr, which is where a \
         recovering sanitizer diagnostic lands:\n{}",
        outcome.stderr
    );
    assert!(
        !outcome.stdout.is_empty(),
        "case `{case}` exited 0 with an empty transcript"
    );
    outcome
}

fn assert_values(run: &Run, case: &str, expected: &[(&str, &str)]) {
    for (key, want) in expected {
        let got: String = run.value(key);
        assert_eq!(
            &got.as_str(),
            want,
            "case `{case}`: {key} is {got}, expected {want}\nfull transcript:\n{}",
            run.stdout
        );
    }
}

// --------------------------------------------------------- instrumentation --

/// The trailing guard page is armed.
///
/// Without this the overrun checks in every other case would pass while
/// measuring nothing — the shape #320 named. `selftest_guard_page` writes one
/// byte past a `guarded_buf`; the process must die rather than return.
#[test]
fn the_guard_page_is_armed() {
    let outcome = run(&our_cdylib(), "selftest_guard_page");
    assert!(
        outcome.signal.is_some(),
        "the deliberate one-byte overrun did not fault: {}\nstdout:\n{}",
        outcome.outcome(),
        outcome.stdout
    );
    assert!(
        outcome.stdout.contains("about_to_overrun=1"),
        "the harness did not reach the overrun: {}",
        outcome.stdout
    );
    assert!(
        !outcome.stdout.contains("overrun_survived=1"),
        "the write past the payload returned instead of faulting: {}",
        outcome.stdout
    );
}

/// The leading canary is checked. A short-stride write lands in the slack
/// before the payload, which no guard page can cover, so the canary is the
/// mechanism — and this proves it reports rather than passes.
#[test]
fn the_canary_is_checked() {
    let outcome = run(&our_cdylib(), "selftest_canary");
    assert_eq!(
        outcome.code,
        Some(2),
        "corrupting the canary should exit 2, got {}\nstdout:\n{}",
        outcome.outcome(),
        outcome.stdout
    );
    assert_eq!(outcome.value("canary"), "corrupt");
}

/// Every case the C driver dispatches is either compared against the oracle or
/// named as a selftest. A case added to the harness and to nothing here would
/// run in no CI leg at all.
#[test]
fn every_harness_case_is_driven() {
    let source: String = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("cabi_misuse_harness.c"),
    )
    .expect("read the harness source");

    let mut dispatched: Vec<String> = Vec::new();
    for fragment in source.split("strcmp(case_name, \"").skip(1) {
        let name: &str = fragment
            .split('"')
            .next()
            .expect("a dispatch arm names its case");
        dispatched.push(name.to_string());
    }
    dispatched.sort();
    dispatched.dedup();

    let mut driven: Vec<String> = CASES
        .iter()
        .chain(SELFTESTS.iter())
        .map(|case| case.to_string())
        .collect();
    driven.sort();

    assert_eq!(
        dispatched, driven,
        "the harness dispatches cases this file does not run (or vice versa); \
         add the case to CASES or to SELFTESTS"
    );
}

// ------------------------------------------------------------ the contract --

/// `init → destroy → destroy`, in the part that has a contract.
///
/// `tj3Destroy(NULL)` returns without touching anything
/// (`references/libjpeg-turbo/src/turbojpeg.c:641`) and `tj3InitVersion`
/// rejects an `initType` outside `enum TJINIT` with NULL (`:589-591`). A second
/// destroy of the same live pointer is a double free upstream too, so the
/// harness does not drive it — see its header.
#[test]
fn lifecycle_contract() {
    let outcome = run_ours("lifecycle");
    assert_values(
        &outcome,
        "lifecycle",
        &[
            ("destroy_null_before", "returned"),
            ("init_decompress", "handle"),
            ("destroy", "returned"),
            // Four non-default parameters are written before the destroy, so
            // this compares a re-init against a handle that *could* have
            // inherited them rather than against a second copy of the same
            // constructor.
            ("dirtied", "4"),
            ("destroy_null_after", "returned"),
            ("reinit", "handle"),
            ("reinit_matches_fresh", "yes"),
            ("init_neg", "null"),
            ("init_numinit", "null"),
            ("init_large", "null"),
            ("cycles_100", "ok"),
        ],
    );
}

/// The initial value of every `TJPARAM`, on all three init types.
///
/// The *values* as a whole are compared against the oracle. What is asserted
/// here without one is narrower and worth stating exactly: that the harness
/// printed a line for all 26 parameters on each init type (so a transcript
/// that lost half its keys is not silently a subset), and that the eleven
/// sentinels `tj3InitVersion` seeds explicitly are the seeded values rather
/// than the zeroes a `Default` would give — which is where a port that forgot
/// to seed one shows up.
#[test]
fn handle_defaults_are_readable() {
    let outcome = run_ours("handle_defaults");
    let transcript = outcome.transcript();
    assert_eq!(
        transcript.len(),
        26 * 3,
        "expected every TJPARAM on each init type:\n{}",
        outcome.stdout
    );
    assert_values(
        &outcome,
        "handle_defaults",
        &[
            // The sentinels `tj3InitVersion` seeds explicitly
            // (`turbojpeg.c:598-608`) — the ones a zeroed struct would get
            // wrong.
            ("default_compress_QUALITY", "-1"),
            ("default_compress_SUBSAMP", "-1"),
            ("default_compress_COLORSPACE", "-1"),
            ("default_compress_PRECISION", "8"),
            ("default_compress_LOSSLESSPSV", "1"),
            ("default_compress_XDENSITY", "1"),
            ("default_compress_YDENSITY", "1"),
            ("default_compress_SAVEMARKERS", "2"),
        ],
    );
}

/// `tj3Set`'s applicability rule, over the whole 26 x 3 matrix.
///
/// The sixteen pairs where we accept what upstream refuses are P4-207 (#629)
/// and belong to the oracle comparison. What is asserted here is the part both
/// implementations agree on and that needs no oracle: a transform instance is
/// initialised for both roles, so every *role-conditioned* guard passes on it
/// and only the two read-only parameters are refused — and an out-of-range
/// index is refused on either side of the enum.
#[test]
fn parameter_applicability_matrix_is_complete() {
    let outcome = run_ours("parameter_applicability");
    let transcript = outcome.transcript();
    assert_eq!(
        transcript.len(),
        26 * 3 + 4,
        "expected every TJPARAM on each init type plus the four range probes:\n{}",
        outcome.stdout
    );
    // `JPEGWIDTH` / `JPEGHEIGHT` are published by a decode, never written by a
    // caller, so `tj3Set` refuses them whatever the instance is.
    const READ_ONLY: [&str; 2] = ["set_transform_JPEGWIDTH", "set_transform_JPEGHEIGHT"];
    for (key, value) in &transcript {
        if key.starts_with("set_transform_") {
            let expected: &str = if READ_ONLY.contains(&key.as_str()) {
                "-1"
            } else {
                "0"
            };
            assert_eq!(
                value, expected,
                "{key}: a transform instance is initialised for compression \
                 and decompression, so every parameter but the two read-only \
                 ones is accepted on it"
            );
        }
    }
    assert_values(
        &outcome,
        "parameter_applicability",
        &[
            ("set_negative_param", "-1"),
            ("set_past_last_param", "-1"),
            ("get_negative_param", "-1"),
            ("get_past_last_param", "-1"),
        ],
    );
}

/// Every entry point taking a handle rejects NULL with its documented error
/// value rather than dereferencing it (`GET_TJINSTANCE` and friends,
/// `turbojpeg.c:300-345`), and `tj3GetErrorStr(NULL)` falls back to the global
/// string (`:678-687`).
#[test]
fn null_handles_are_rejected() {
    let outcome = run_ours("null_handle");
    assert_values(
        &outcome,
        "null_handle",
        &[
            ("get", "-1"),
            ("set", "-1"),
            ("decompress_header", "-1"),
            ("decompress8", "-1"),
            ("compress8", "-1"),
            ("set_scaling_factor", "-1"),
            ("set_cropping_region", "-1"),
            ("get_error_str_null", "string"),
            // A rejected compress must not have touched the caller's output
            // slots.
            ("compress8_out_ptr", "untouched"),
            ("compress8_out_len", "0"),
            // The destination is a guarded buffer: a library that skipped the
            // NULL-handle check would write the fixture's 101 KB into three
            // bytes and fault here rather than corrupting this process.
            ("dst_canary", "intact"),
            ("dst_untouched", "yes"),
        ],
    );
}

/// `tj3Decompress8`'s pitch contract (`turbojpeg-mp.c:229-231`, with the
/// negative-pitch and NULL rejection at `:170-172`): 0 means
/// `output_width * pixelSize`, less than that is "Invalid argument", a negative
/// pitch is refused, and padding above it is honoured.
///
/// Each accepted pitch decodes into exactly `pitch * height` bytes with a
/// `PROT_NONE` page immediately after the last one, so "does not overrun" is
/// enforced by the hardware. The four pixel digests must agree: padding changes
/// where rows start and must not change what is in them.
#[test]
fn pitch_boundaries_are_enforced() {
    let outcome = run_ours("pitch_boundaries");
    assert_values(
        &outcome,
        "pitch_boundaries",
        &[
            ("width", "227"),
            ("height", "149"),
            ("pitch_implicit_rc", "0"),
            ("pitch_exact_rc", "0"),
            ("pitch_padded_rc", "0"),
            ("pitch_double_rc", "0"),
            ("pitch_implicit_canary", "intact"),
            ("pitch_exact_canary", "intact"),
            ("pitch_padded_canary", "intact"),
            ("pitch_double_canary", "intact"),
            // Below `width * pixelSize`, and negative.
            ("pitch_short_rc", "-1"),
            ("pitch_one_rc", "-1"),
            ("pitch_negative_rc", "-1"),
            ("pitch_reject_canary", "intact"),
            ("pitch_reject_untouched", "yes"),
            ("pf_negative_rc", "-1"),
            ("pf_numpf_rc", "-1"),
            ("null_dst_rc", "-1"),
            ("null_src_rc", "-1"),
            ("zero_size_rc", "-1"),
            // A callee that wrote *before* the destination and then returned
            // the expected error passed everything else here: the write lands
            // in accessible mmap slack that no guard page and no sanitizer
            // covers.
            ("arg_reject_canary", "intact"),
            ("arg_reject_untouched", "yes"),
        ],
    );
    let baseline: String = outcome.value("pitch_implicit_pixels");
    for label in ["exact", "padded", "double"] {
        assert_eq!(
            outcome.value(&format!("pitch_{label}_pixels")),
            baseline,
            "pitch `{label}` changed the pixels, not just their placement:\n{}",
            outcome.stdout
        );
    }
}

/// Undersized output buffers on the compression side, which is where
/// `TJPARAM_NOREALLOC` turns "too small" into a contract: the caller's buffer is
/// used in place and `jpegSize` is its capacity, so a short buffer must be an
/// error rather than a resize or an overrun (P4-145). The buffer is guarded, so
/// "rather than an overrun" is checked and not assumed.
#[test]
fn undersized_output_buffers_are_refused() {
    let outcome = run_ours("undersized_output");
    assert_values(
        &outcome,
        "undersized_output",
        &[
            ("compressed_rc", "0"),
            // One byte under what the image actually compresses to — the case
            // measures that first rather than guessing, so the boundary does
            // not drift when the encoder's output moves.
            ("norealloc_one_rc", "-1"),
            ("norealloc_tiny_rc", "-1"),
            ("norealloc_one_short_rc", "-1"),
            ("norealloc_worst_case_rc", "0"),
            ("norealloc_worst_case_fits", "yes"),
            ("norealloc_worst_case_soi", "yes"),
        ],
    );
    // Whatever the outcome, the caller's pointer is never moved and nothing is
    // written before the payload — that is the NOREALLOC contract (P4-145) and
    // it holds on the refusing capacities too.
    for label in ["one", "tiny", "one_short", "exact", "worst_case"] {
        assert_values(
            &outcome,
            "undersized_output",
            &[
                (&format!("norealloc_{label}_slot"), "unchanged"),
                (&format!("norealloc_{label}_canary"), "intact"),
            ],
        );
    }
    // `norealloc_exact_*` is a known divergence (P4-206) and belongs to the
    // oracle comparison, not here.
    let compressed: usize = outcome.value("compressed").parse().expect("size");
    let worst_case: usize = outcome.value("bufsize").parse().expect("bufsize");
    assert!(
        compressed >= 2 && compressed < worst_case,
        "the measured output ({compressed}) must sit strictly inside \
         tj3JPEGBufSize ({worst_case}) for the boundary slots to be distinct"
    );
}

/// Maximum dimensions: the arithmetic that sizes a destination, and the
/// boundary libjpeg refuses (`JPEG_MAX_DIMENSION`, `jcmaster.c:186-189`), plus
/// the decompression-side ceiling `TJPARAM_MAXPIXELS`
/// (`turbojpeg-mp.c:195-198`).
///
/// `compress_over_max_*` is a known divergence — see [`KNOWN_DIVERGENCES`] —
/// so it is deliberately not asserted here; the oracle comparison owns it.
#[test]
fn maximum_dimensions_are_bounded() {
    let outcome = run_ours("max_dimensions");
    assert_values(
        &outcome,
        "max_dimensions",
        &[
            ("bufsize_zero", "0"),
            ("bufsize_negative", "0"),
            ("bufsize_bad_subsamp", "0"),
            ("compress_zero_width_rc", "-1"),
            ("compress_zero_height_rc", "-1"),
            ("compress_negative_width_rc", "-1"),
            ("compress_negative_height_rc", "-1"),
            ("compress_negative_pitch_rc", "-1"),
            ("maxpixels_set_one", "0"),
            ("maxpixels_one_rc", "-1"),
            ("maxpixels_exact_rc", "0"),
            ("maxpixels_short_rc", "-1"),
            ("maxpixels_canary", "intact"),
        ],
    );
    // 65500 x 65500 at 4:4:4 is ~25.7 GB as a number and must not wrap on the
    // way to one. The literals are 64-bit `size_t` values; on a 32-bit host
    // both implementations truncate and the oracle comparison is what holds
    // the line, so the absolute check is gated rather than deleted.
    #[cfg(target_pointer_width = "64")]
    assert_values(
        &outcome,
        "max_dimensions",
        &[
            ("bufsize_max_444", "25744646144"),
            ("bufsize_max_420", "12872324096"),
        ],
    );
}

/// Concurrent calls in the shape the API supports: one handle per thread,
/// eight of them, against the process-wide state below the handle — dispatch
/// tables and the lazily built Huffman tables. Every thread must reproduce the
/// single-threaded digest.
///
/// **The ordering is the whole test.** The eight threads are created, parked on
/// a condition variable, counted, and released together once all eight have
/// arrived, so the first call into the library in that process is made by eight
/// threads at once; the serial reference is decoded *afterwards*. The arrival
/// count is in the transcript (`parked`) rather than assumed — broadcasting as
/// soon as the last `pthread_create` returned let a slow starter walk past an
/// already-open gate, which `codex review` measured at seven of eight. An earlier version probed the header and
/// decoded the reference on the main thread first, which warmed every lazily
/// built table — so each worker took the already-initialised path and the case
/// could not have seen a race in the initialisation it names. `codex review`
/// found that; it is #320's shape, a coverage claim the mechanism did not
/// provide.
///
/// Two threads sharing one handle is deliberately not driven: upstream mutates
/// `tjinstance` from every entry point without synchronisation, so there is no
/// observable contract to compare against — only a data race in both
/// implementations. `capi_thread_affinity.rs` records the same reasoning for
/// the classic `cinfo`.
#[test]
fn concurrent_handles_agree() {
    let outcome = run_ours("concurrent_handles");
    assert_values(
        &outcome,
        "concurrent_handles",
        &[
            // All eight were parked on the condition variable before it opened,
            // so the process's first library call really was concurrent. The
            // count is reported rather than assumed: broadcasting as soon as
            // the last `pthread_create` returned let a slow starter walk past
            // an already-open gate, which measured seven.
            ("parked", "8"),
            // Printed before the harness's own guard, so a failing reference
            // decode reaches this assertion instead of an empty transcript.
            ("reference_rc", "0"),
            ("workers_agree", "yes"),
        ],
    );
}

/// `tj3Alloc` / `tj3Free` across the ABI, in both directions: a destination the
/// library allocated and the caller releases, and an output the library
/// allocated during a compress. The pre-existing sanitizer harness allocates
/// with its own `malloc`/`free`, so this contract had never crossed the
/// boundary — recorded in `docs/UNSAFE_INVENTORY_CAPI.md`'s `tj3Free` row and
/// in P4-141 criterion 2.
#[test]
fn allocator_ownership_crosses_the_boundary() {
    let outcome = run_ours("alloc_ownership");
    assert_values(
        &outcome,
        "alloc_ownership",
        &[
            ("free_null", "returned"),
            ("alloc_4096", "pointer"),
            ("free_4096", "returned"),
            ("free_zero", "returned"),
            ("decode_into_tj3alloc_rc", "0"),
            ("compress_alloc_rc", "0"),
            ("compress_alloc_out", "allocated"),
            ("compress_alloc_soi", "yes"),
        ],
    );
}

/// A 12-bit round trip across the ABI. No 12-bit entry point crossed the C
/// boundary under a sanitizer before this case existed — the other half of the
/// C-boundary gap P4-141 criterion 2 records.
///
/// `precision` is a known divergence (see [`KNOWN_DIVERGENCES`]) and is
/// asserted by the oracle comparison rather than here.
#[test]
fn twelve_bit_round_trip_crosses_the_boundary() {
    let outcome = run_ours("precision12");
    assert_values(
        &outcome,
        "precision12",
        &[
            ("compress12_rc", "0"),
            ("compress12_nonempty", "yes"),
            ("header12_rc", "0"),
            ("jpegwidth", "64"),
            ("jpegheight", "48"),
            ("decompress12_rc", "0"),
            ("decompress12_canary", "intact"),
            // The pitch clause applies at 12-bit precision too, in samples.
            ("decompress12_short_pitch_rc", "-1"),
        ],
    );
}

// --------------------------------------------------------- the comparison --

/// P4-207 (#629): the sixteen (instance type, parameter) pairs `tj3Set`
/// accepts and upstream refuses as "not applicable".
///
/// Listed as pairs rather than as sixteen `KnownDivergence` structs because
/// they are one defect with one owner; [`known_divergences_are_live`] holds
/// each of them to the same "must still diverge" rule, so closing P4-207 still
/// has to delete them.
const APPLICABILITY_DIVERGENCES: &[(&str, &str)] = &[
    ("compress", "FASTUPSAMPLE"),
    ("compress", "SCANLIMIT"),
    ("decompress", "NOREALLOC"),
    ("decompress", "QUALITY"),
    ("decompress", "COLORSPACE"),
    ("decompress", "OPTIMIZE"),
    ("decompress", "PROGRESSIVE"),
    ("decompress", "ARITHMETIC"),
    ("decompress", "LOSSLESS"),
    ("decompress", "LOSSLESSPSV"),
    ("decompress", "LOSSLESSPT"),
    ("decompress", "RESTARTBLOCKS"),
    ("decompress", "RESTARTROWS"),
    ("decompress", "XDENSITY"),
    ("decompress", "YDENSITY"),
    ("decompress", "DENSITYUNITS"),
];

const APPLICABILITY_ITEM: &str = "P4-207 (#629)";

/// The known divergences, as (case, key, owning item), with the applicability
/// pairs expanded into the transcript keys they name.
fn known_divergences() -> Vec<(&'static str, String, &'static str)> {
    let mut all: Vec<(&'static str, String, &'static str)> = KNOWN_DIVERGENCES
        .iter()
        .map(|entry| (entry.case, entry.key.to_string(), entry.item))
        .collect();
    all.extend(APPLICABILITY_DIVERGENCES.iter().map(|(init, param)| {
        (
            "parameter_applicability",
            format!("set_{init}_{param}"),
            APPLICABILITY_ITEM,
        )
    }));
    all
}

fn divergence_for(case: &str, key: &str) -> Option<&'static str> {
    known_divergences()
        .into_iter()
        .find(|(entry_case, entry_key, _)| *entry_case == case && entry_key == key)
        .map(|(_, _, item)| item)
}

/// Every case produces the same transcript on our shim and on stock
/// TurboJPEG, except the lines [`KNOWN_DIVERGENCES`] names.
#[test]
fn transcripts_match_stock_turbojpeg() {
    let Some(oracle) = stock_turbojpeg() else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install with a loadable \
             libturbojpeg; set LIBJPEG_TURBO_PREFIX to make this mandatory"
        );
        return;
    };
    let ours: PathBuf = our_cdylib();

    for case in CASES {
        let mine = run(&ours, case);
        let theirs = run(&oracle, case);
        // A floor, without which two children that both die before printing
        // anything compare two empty transcripts and pass. The nine contract
        // tests above assert exit 0 per case today, but that backstop is
        // incidental — this one is structural and covers a case added later.
        assert_eq!(
            mine.code,
            Some(0),
            "case `{case}` did not complete against our cdylib: {}\nstdout:\n{}\nstderr:\n{}",
            mine.outcome(),
            mine.stdout,
            mine.stderr
        );
        assert!(
            !mine.transcript().is_empty() && !theirs.transcript().is_empty(),
            "case `{case}` produced an empty transcript on at least one side \
             (ours {} lines, stock {} lines); there is nothing to compare",
            mine.transcript().len(),
            theirs.transcript().len()
        );
        assert_eq!(
            mine.outcome(),
            theirs.outcome(),
            "case `{case}` ended differently: ours {} vs stock {}\n\
             ours stdout:\n{}\nours stderr:\n{}\nstock stdout:\n{}",
            mine.outcome(),
            theirs.outcome(),
            mine.stdout,
            mine.stderr,
            theirs.stdout
        );

        let mine_lines = mine.transcript();
        let theirs_lines = theirs.transcript();
        let mine_keys: Vec<&String> = mine_lines.iter().map(|(key, _)| key).collect();
        let theirs_keys: Vec<&String> = theirs_lines.iter().map(|(key, _)| key).collect();
        assert_eq!(
            mine_keys, theirs_keys,
            "case `{case}` printed a different sequence of keys, so one side \
             stopped early or grew a line the other has not:\nours:\n{}\nstock:\n{}",
            mine.stdout, theirs.stdout
        );

        for ((key, mine_value), (_, their_value)) in mine_lines.iter().zip(theirs_lines.iter()) {
            if let Some(item) = divergence_for(case, key) {
                assert_ne!(
                    mine_value, their_value,
                    "`{case}`/`{key}` is listed as a known divergence owned by \
                     {item} but the two libraries now agree; delete the entry"
                );
                continue;
            }
            assert_eq!(
                mine_value, their_value,
                "`{case}`/`{key}`: ours {mine_value}, stock TurboJPEG \
                 {their_value}\nours:\n{}\nstock:\n{}",
                mine.stdout, theirs.stdout
            );
        }
    }
}

/// Every listed divergence still diverges, and names a live item.
///
/// Without this the list would become a place where a fixed defect keeps an
/// exemption that quietly covers the *next* regression on the same line — the
/// failure mode `deferrals_are_live_and_named` was written for in the
/// `unsafe`-inventory gate.
#[test]
fn known_divergences_are_live() {
    let Some(oracle) = stock_turbojpeg() else {
        eprintln!("SKIP: no loadable stock libturbojpeg to compare against");
        return;
    };
    let ours: PathBuf = our_cdylib();

    for (case, key, item) in known_divergences() {
        assert!(
            CASES.contains(&case),
            "known divergence names `{case}`, which is not a compared case"
        );
        assert!(
            item.starts_with("P4-"),
            "known divergence `{case}`/`{key}` must name the LAST_MILE item \
             that owns it, got {item:?}"
        );
        let mine = run(&ours, case);
        let theirs = run(&oracle, case);
        assert_ne!(
            mine.value(&key),
            theirs.value(&key),
            "`{case}`/`{key}` no longer diverges — {item} is fixed, so delete \
             its entry and let the comparison hold the line"
        );
    }
}
