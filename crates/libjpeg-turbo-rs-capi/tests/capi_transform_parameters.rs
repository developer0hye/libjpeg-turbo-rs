//! P4-227 (#655): which handle parameters `tj3Transform` applies, compared
//! verbatim against stock TurboJPEG.
//!
//! Upstream's `tj3Transform` (`references/libjpeg-turbo/src/turbojpeg.c:2920`)
//! applies the handle's `TJPARAM_MAXPIXELS`, `TJPARAM_SCANLIMIT` and
//! `TJPARAM_MAXMEMORY` to the source and its `TJPARAM_PROGRESSIVE`,
//! `TJPARAM_ARITHMETIC`, `TJPARAM_OPTIMIZE` and restart interval to the
//! output. The port read none of them.
//!
//! The trace is produced by `examples/transform_parameters_oracle.c` against
//! real TurboJPEG and mirrored here through this crate's own exports, over the
//! same JPEG bytes. Each line carries the output's SOF marker, size and
//! FNV-1a hash, so it pins the transformed *bytes*, not just the return code.

use std::ffi::{c_int, c_void};
use std::path::PathBuf;

mod helpers;

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::transform::{
    tj3Transform, TjTransform, TJXOPT_ARITHMETIC, TJXOPT_CROP, TJXOPT_GRAY, TJXOPT_PERFECT,
    TJXOPT_PROGRESSIVE, TJXOP_HFLIP, TJXOP_NONE, TJXOP_ROT180, TJXOP_ROT90, TJXOP_TRANSPOSE,
    TJXOP_TRANSVERSE, TJXOP_VFLIP,
};
use libjpeg_turbo_rs_capi::{
    tj3DecompressHeader, tj3Destroy, tj3Free, tj3Get, tj3GetErrorStr, tj3Init, tj3Set, TjRegion,
};

const TJINIT_DECOMPRESS: c_int = 1;
const TJINIT_TRANSFORM: c_int = 2;

const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJPARAM_OPTIMIZE: c_int = 11;
const TJPARAM_PROGRESSIVE: c_int = 12;
const TJPARAM_SCANLIMIT: c_int = 13;
const TJPARAM_ARITHMETIC: c_int = 14;
const TJPARAM_RESTARTBLOCKS: c_int = 18;
const TJPARAM_RESTARTROWS: c_int = 19;
const TJPARAM_MAXMEMORY: c_int = 23;
const TJPARAM_MAXPIXELS: c_int = 24;

/// No parameter: `run`'s `param < 0`.
const NONE: (c_int, c_int) = (-1, 0);

const FIXTURES: [(&str, &[u8]); 4] = [
    (
        "base",
        include_bytes!("../../../tests/fixtures/photo_64x64_420.jpg"),
    ),
    (
        "prog",
        include_bytes!("../../../tests/fixtures/photo_64x64_420_prog.jpg"),
    ),
    (
        "gray",
        include_bytes!("../../../tests/fixtures/gray_8x8.jpg"),
    ),
    // Carries a DRI of 200 MCUs, which no transform output inherits.
    (
        "rst",
        include_bytes!("../../../tests/fixtures/photo_640x480_420_rst.jpg"),
    ),
];

/// The oracle's `big` label: 1024x1024 4:4:4, so its source coefficient
/// arrays are exactly 6 MiB. Generated here and handed to the oracle as
/// bytes, so both sides transform the same stream.
fn big() -> Vec<u8> {
    let (width, height): (usize, usize) = (1024, 1024);
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i / 3) % width + i / (3 * width)) as u8)
        .collect();
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        50,
        Subsampling::S444,
    )
    .expect("encode the 1024x1024 source")
}

fn inputs() -> Vec<(String, Vec<u8>)> {
    let mut inputs: Vec<(String, Vec<u8>)> = FIXTURES
        .iter()
        .map(|(label, jpeg)| (label.to_string(), jpeg.to_vec()))
        .collect();
    inputs.push(("big".to_string(), big()));
    inputs
}

/// The oracle's `sof_marker`.
fn sof_marker(jpeg: &[u8]) -> u8 {
    jpeg.windows(2)
        .skip(2)
        .find(|pair| {
            pair[0] == 0xFF
                && (0xC0..=0xCF).contains(&pair[1])
                && ![0xC4, 0xC8, 0xCC].contains(&pair[1])
        })
        .map_or(0, |pair| pair[1])
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(14_695_981_039_346_656_037, |hash: u64, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
        })
}

/// The oracle's `run`: one `tj3Transform` on a fresh handle.
fn run(
    label: &str,
    case_name: &str,
    jpeg: &[u8],
    params: [(c_int, c_int); 2],
    op: c_int,
    options: c_int,
) -> String {
    run_region(label, case_name, jpeg, params, op, options, (0, 0, 0, 0))
}

/// [`run`] with a crop region, applied when `options` has `TJXOPT_CROP`.
fn run_region(
    label: &str,
    case_name: &str,
    jpeg: &[u8],
    params: [(c_int, c_int); 2],
    op: c_int,
    options: c_int,
    (x, y, w, h): (c_int, c_int, c_int, c_int),
) -> String {
    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null());
    let transform: TjTransform = TjTransform {
        r: TjRegion { x, y, w, h },
        op,
        options,
        data: std::ptr::null_mut(),
        custom_filter: None,
    };
    let mut dst: *mut u8 = std::ptr::null_mut();
    let mut dst_size: usize = 0;
    // SAFETY: live handle; one transform, one output slot.
    let (rc, width): (c_int, c_int) = unsafe {
        for (param, value) in params {
            if param >= 0 {
                tj3Set(handle, param, value);
            }
        }
        let rc: c_int = tj3Transform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            &mut dst,
            &mut dst_size,
            &transform,
        );
        (rc, tj3Get(handle, TJPARAM_JPEGWIDTH))
    };
    let line: String = if rc == 0 {
        // SAFETY: `tj3Transform` succeeded, so `dst` holds `dst_size` bytes.
        let out: &[u8] = unsafe { std::slice::from_raw_parts(dst, dst_size) };
        format!(
            "{label} {case_name} rc={rc} sof={:02x} size={} hash={:016x} jw={width}\n",
            sof_marker(out),
            out.len(),
            fnv1a(out)
        )
    } else {
        format!(
            "{label} {case_name} rc={rc} sof=00 size=0 hash={:016x} jw={width}\n",
            0
        )
    };
    // SAFETY: `dst` is NULL or came from this library's allocator.
    unsafe {
        tj3Free(dst.cast::<c_void>());
        tj3Destroy(handle);
    }
    line
}

fn trace_label(label: &str, jpeg: &[u8]) -> String {
    let probe: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    // SAFETY: live handle; `jpeg` is a live slice.
    let pixels: c_int = unsafe {
        assert_eq!(tj3DecompressHeader(probe, jpeg.as_ptr(), jpeg.len()), 0);
        let pixels: c_int = tj3Get(probe, TJPARAM_JPEGWIDTH) * tj3Get(probe, TJPARAM_JPEGHEIGHT);
        tj3Destroy(probe);
        pixels
    };
    let cases: [(&str, [(c_int, c_int); 2], c_int, c_int); 17] = [
        ("plain", [NONE, NONE], TJXOP_NONE, 0),
        ("rot90", [NONE, NONE], TJXOP_ROT90, 0),
        (
            "maxpixels_under",
            [(TJPARAM_MAXPIXELS, pixels - 1), NONE],
            TJXOP_NONE,
            0,
        ),
        (
            "maxpixels_at",
            [(TJPARAM_MAXPIXELS, pixels), NONE],
            TJXOP_NONE,
            0,
        ),
        (
            "maxpixels_perfect",
            [(TJPARAM_MAXPIXELS, 1), NONE],
            TJXOP_TRANSVERSE,
            TJXOPT_PERFECT,
        ),
        ("scanlimit", [(TJPARAM_SCANLIMIT, 2), NONE], TJXOP_NONE, 0),
        (
            "maxmemory_small",
            [(TJPARAM_MAXMEMORY, 1), NONE],
            TJXOP_NONE,
            0,
        ),
        (
            "progressive",
            [(TJPARAM_PROGRESSIVE, 1), NONE],
            TJXOP_NONE,
            0,
        ),
        ("arithmetic", [(TJPARAM_ARITHMETIC, 1), NONE], TJXOP_NONE, 0),
        (
            "progressive_arithmetic",
            [(TJPARAM_PROGRESSIVE, 1), (TJPARAM_ARITHMETIC, 1)],
            TJXOP_ROT90,
            0,
        ),
        ("optimize", [(TJPARAM_OPTIMIZE, 1), NONE], TJXOP_NONE, 0),
        (
            "arithmetic_optimize",
            [(TJPARAM_ARITHMETIC, 1), (TJPARAM_OPTIMIZE, 1)],
            TJXOP_NONE,
            0,
        ),
        (
            "restartblocks",
            [(TJPARAM_RESTARTBLOCKS, 4), NONE],
            TJXOP_NONE,
            0,
        ),
        (
            "restartrows",
            [(TJPARAM_RESTARTROWS, 1), NONE],
            TJXOP_ROT90,
            0,
        ),
        (
            "restartrows_then_blocks",
            [(TJPARAM_RESTARTROWS, 1), (TJPARAM_RESTARTBLOCKS, 4)],
            TJXOP_NONE,
            0,
        ),
        (
            "opt_progressive",
            [NONE, NONE],
            TJXOP_NONE,
            TJXOPT_PROGRESSIVE,
        ),
        (
            "opt_arithmetic",
            [NONE, NONE],
            TJXOP_NONE,
            TJXOPT_ARITHMETIC,
        ),
    ];
    let mut trace: String = String::new();
    for (case_name, params, op, options) in cases {
        if case_name == "opt_progressive" {
            for (crop_name, crop_op, region) in [
                ("crop_to_edge", TJXOP_NONE, (16, 16, 0, 0)),
                ("crop_width_to_edge", TJXOP_NONE, (16, 0, 0, 16)),
                ("crop_to_edge_rot90", TJXOP_ROT90, (16, 0, 0, 0)),
                ("crop_past_edge", TJXOP_NONE, (48, 0, 32, 32)),
                ("crop_origin_outside", TJXOP_NONE, (64, 0, 0, 0)),
            ] {
                trace.push_str(&run_region(
                    label,
                    crop_name,
                    jpeg,
                    [NONE, NONE],
                    crop_op,
                    TJXOPT_CROP,
                    region,
                ));
            }
        }
        trace.push_str(&run(label, case_name, jpeg, params, op, options));
    }
    if label == "big" {
        for (case_name, op, megabytes) in [
            ("maxmemory_none_6", TJXOP_NONE, 6),
            ("maxmemory_none_7", TJXOP_NONE, 7),
            ("maxmemory_hflip_6", TJXOP_HFLIP, 6),
            ("maxmemory_hflip_7", TJXOP_HFLIP, 7),
            ("maxmemory_vflip_12", TJXOP_VFLIP, 12),
            ("maxmemory_vflip_13", TJXOP_VFLIP, 13),
            ("maxmemory_rot90_12", TJXOP_ROT90, 12),
            ("maxmemory_rot90_13", TJXOP_ROT90, 13),
            ("maxmemory_rot180_12", TJXOP_ROT180, 12),
            ("maxmemory_rot180_13", TJXOP_ROT180, 13),
            ("maxmemory_transpose_12", TJXOP_TRANSPOSE, 12),
            ("maxmemory_transpose_13", TJXOP_TRANSPOSE, 13),
        ] {
            trace.push_str(&run(
                label,
                case_name,
                jpeg,
                [(TJPARAM_MAXMEMORY, megabytes), NONE],
                op,
                0,
            ));
        }
        for (case_name, megabytes) in [("maxmemory_rot90_gray_8", 8), ("maxmemory_rot90_gray_9", 9)]
        {
            trace.push_str(&run(
                label,
                case_name,
                jpeg,
                [(TJPARAM_MAXMEMORY, megabytes), NONE],
                TJXOP_ROT90,
                TJXOPT_GRAY,
            ));
        }
    }
    trace
}

fn our_trace() -> String {
    inputs()
        .iter()
        .map(|(label, jpeg)| trace_label(label, jpeg))
        .collect()
}

fn line_for<'a>(trace: &'a str, label: &str, case_name: &str) -> &'a str {
    let prefix: String = format!("{label} {case_name} ");
    trace
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{trace}"))
}

fn rc_of(line: &str) -> &str {
    line.split_whitespace()
        .find(|field| field.starts_with("rc="))
        .expect("an rc field")
}

/// The whole trace, compared verbatim against real TurboJPEG.
#[test]
fn transform_applies_what_stock_turbojpeg_applies() {
    let Some(oracle) = helpers::build_oracle("transform_parameters_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             P4-227's transform parameters cannot be built. Set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let workdir: PathBuf =
        std::env::temp_dir().join(format!("libjpeg_turbo_rs_p4227_{}", std::process::id()));
    std::fs::create_dir_all(&workdir).expect("create oracle workdir");
    let mut args: Vec<String> = vec![workdir.to_str().expect("utf-8 workdir").to_string()];
    for (label, jpeg) in inputs() {
        std::fs::write(workdir.join(format!("{label}.jpg")), &jpeg).expect("write fixture");
        args.push(label);
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let c_trace: String = helpers::run_oracle(&oracle, &arg_refs);
    let _ = std::fs::remove_dir_all(&workdir);

    assert_eq!(
        our_trace(),
        c_trace,
        "tj3Transform applies different handle parameters than stock TurboJPEG \
         (turbojpeg.c:2920-3086)"
    );
}

// --- The contract stated without the oracle --------------------------------
//
// Every expected value below is the stock 3.2.0 trace's.

/// Issue #655: `TJPARAM_MAXPIXELS` refuses a source one pixel over it — before
/// the `TJXOPT_PERFECT` check — and `TJPARAM_SCANLIMIT` a stream with more
/// scans; a transform publishes nothing.
#[test]
fn transform_honours_maxpixels_and_scanlimit() {
    let trace: String = our_trace();
    for label in ["base", "prog", "gray", "rst", "big"] {
        assert_eq!(rc_of(line_for(&trace, label, "maxpixels_under")), "rc=-1");
        assert_eq!(rc_of(line_for(&trace, label, "maxpixels_at")), "rc=0");
        assert_eq!(rc_of(line_for(&trace, label, "maxpixels_perfect")), "rc=-1");
        assert!(line_for(&trace, label, "plain").ends_with(" jw=-1"));
    }
    assert_eq!(rc_of(line_for(&trace, "prog", "scanlimit")), "rc=-1");
    assert_eq!(rc_of(line_for(&trace, "base", "scanlimit")), "rc=0");
}

/// Issue #655: `TJPARAM_MAXMEMORY` bounds the source's coefficient arrays
/// plus the transform's workspace — 6 MiB in place, 12 MiB with a workspace,
/// 8 MiB for a grayscale rotation — refusing at the budget and accepting a
/// MiB above, as stock does.
#[test]
fn transform_honours_maxmemory() {
    let trace: String = our_trace();
    for (case_name, rc) in [
        ("maxmemory_none_6", "rc=-1"),
        ("maxmemory_none_7", "rc=0"),
        ("maxmemory_hflip_6", "rc=-1"),
        ("maxmemory_hflip_7", "rc=0"),
        ("maxmemory_vflip_12", "rc=-1"),
        ("maxmemory_vflip_13", "rc=0"),
        ("maxmemory_rot90_12", "rc=-1"),
        ("maxmemory_rot90_13", "rc=0"),
        ("maxmemory_rot180_12", "rc=-1"),
        ("maxmemory_rot180_13", "rc=0"),
        ("maxmemory_transpose_12", "rc=-1"),
        ("maxmemory_transpose_13", "rc=0"),
        ("maxmemory_rot90_gray_8", "rc=-1"),
        ("maxmemory_rot90_gray_9", "rc=0"),
        ("maxmemory_small", "rc=-1"),
    ] {
        assert_eq!(rc_of(line_for(&trace, "big", case_name)), rc, "{case_name}");
    }
    assert_eq!(rc_of(line_for(&trace, "base", "maxmemory_small")), "rc=0");
}

/// Issue #655: the handle's `TJPARAM_PROGRESSIVE` / `TJPARAM_ARITHMETIC`
/// produce exactly what their `TJXOPT_*` twins produce, and the handle's
/// `TJPARAM_OPTIMIZE` and restart interval change the output. All of these
/// were ignored.
#[test]
fn transform_honours_the_output_parameters() {
    let trace: String = our_trace();
    let output = |case_name: &str| -> String {
        let line: &str = line_for(&trace, "base", case_name);
        line[line.find(" rc=").expect("rc")..].to_string()
    };
    assert_eq!(output("progressive"), output("opt_progressive"));
    assert_eq!(output("arithmetic"), output("opt_arithmetic"));
    assert!(output("progressive").contains(" sof=c2 "));
    assert!(output("arithmetic").contains(" sof=c9 "));
    assert!(output("progressive_arithmetic").contains(" sof=ca "));
    // Arithmetic coding drops optimisation, as upstream's
    // `optimize_coding = FALSE` does.
    assert_eq!(output("arithmetic_optimize"), output("arithmetic"));
    for case_name in ["optimize", "restartblocks"] {
        assert_ne!(output(case_name), output("plain"), "{case_name}");
    }
    // The last restart parameter set wins: blocks after rows is blocks.
    assert_eq!(output("restartrows_then_blocks"), output("restartblocks"));
}

/// Issue #655 (codex review): a transform's output has a restart interval
/// only when the handle asks for one. `jpeg_copy_critical_parameters` does
/// not copy `restart_interval`, so stock drops the source's DRI; the port
/// kept it.
#[test]
fn transform_drops_the_source_restart_interval() {
    let source: &[u8] = FIXTURES[3].1;
    let restart_interval = |jpeg: &[u8]| -> Option<u16> {
        let mut at: usize = 2;
        while at + 4 < jpeg.len() && jpeg[at] == 0xFF && jpeg[at + 1] != 0xDA {
            let length: usize = usize::from(jpeg[at + 2]) << 8 | usize::from(jpeg[at + 3]);
            if jpeg[at + 1] == 0xDD {
                return Some(u16::from(jpeg[at + 4]) << 8 | u16::from(jpeg[at + 5]));
            }
            at += 2 + length;
        }
        None
    };
    assert_eq!(restart_interval(source), Some(200));
    for (blocks, expected) in [(0, None), (4, Some(4))] {
        let output: Vec<u8> = transform_once(source, TJPARAM_RESTARTBLOCKS, blocks, no_crop())
            .expect("the transform succeeds");
        assert_eq!(
            restart_interval(&output),
            expected,
            "RESTARTBLOCKS={blocks}"
        );
    }
}

/// Issue #655 (codex review): a crop region `jtransform_request_workspace`
/// refuses is reported before `TJPARAM_MAXMEMORY`, as upstream validates it
/// before `jpeg_read_coefficients` realizes the arrays the budget bounds.
#[test]
fn an_invalid_crop_outranks_maxmemory() {
    let crop: TjTransform = TjTransform {
        r: TjRegion {
            x: 1024,
            y: 0,
            w: 16,
            h: 16,
        },
        options: TJXOPT_CROP,
        ..no_crop()
    };
    let message: String = transform_once(&big(), TJPARAM_MAXMEMORY, 1, crop).expect_err("refused");
    assert_eq!(message, "Invalid crop request");
}

fn no_crop() -> TjTransform {
    TjTransform {
        r: TjRegion {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        op: TJXOP_NONE,
        options: 0,
        data: std::ptr::null_mut(),
        custom_filter: None,
    }
}

/// One `tj3Transform` with one parameter set: the output, or the error string.
fn transform_once(
    jpeg: &[u8],
    param: c_int,
    value: c_int,
    transform: TjTransform,
) -> Result<Vec<u8>, String> {
    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null());
    let mut dst: *mut u8 = std::ptr::null_mut();
    let mut dst_size: usize = 0;
    // SAFETY: live handle; one transform, one output slot, freed below.
    unsafe {
        tj3Set(handle, param, value);
        let rc: c_int = tj3Transform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            &mut dst,
            &mut dst_size,
            &transform,
        );
        let result: Result<Vec<u8>, String> = if rc == 0 {
            Ok(std::slice::from_raw_parts(dst, dst_size).to_vec())
        } else {
            Err(std::ffi::CStr::from_ptr(tj3GetErrorStr(handle))
                .to_string_lossy()
                .into_owned())
        };
        tj3Free(dst.cast::<c_void>());
        tj3Destroy(handle);
        result
    }
}

/// Codex review of #655: `TJPARAM_MAXPIXELS` is applied from the header, as
/// upstream applies it right after `jpeg_read_header`, before any later scan
/// is walked — so a progressive source cut after its first SOS is refused for
/// its size, not for the scans that are missing.
#[test]
fn maxpixels_is_applied_before_any_later_scan_is_read() {
    let progressive: &[u8] = FIXTURES[1].1;
    let first_sos: usize = progressive
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("an SOS marker");
    let segment_length: usize =
        usize::from(progressive[first_sos + 2]) << 8 | usize::from(progressive[first_sos + 3]);
    let truncated: &[u8] = &progressive[..first_sos + 2 + segment_length + 8];
    let message: String =
        transform_once(truncated, TJPARAM_MAXPIXELS, 1, no_crop()).expect_err("refused");
    assert_eq!(message, "tj3Transform(): Image is too large");
}

/// Codex review of #655: tj3Transform's own crop-alignment refusal
/// (`turbojpeg.c:3007-3015`) also precedes the memory limit, and a crop that
/// enlarges the frame — legal without a transform — counts its expanded
/// workspace: stock refuses the 1100x1100 expansion of the 1024x1024 source
/// at 12 MiB and accepts it at 13 MiB (measured), as does the estimate.
#[test]
fn crop_alignment_and_expansion_meet_the_memory_limit_as_stock_does() {
    let source: Vec<u8> = big();
    let crop = |x: usize, w: usize| -> TjTransform {
        TjTransform {
            r: TjRegion {
                x: x as c_int,
                y: 0,
                w: w as c_int,
                h: w as c_int,
            },
            options: TJXOPT_CROP,
            ..no_crop()
        }
    };
    assert_eq!(
        transform_once(&source, TJPARAM_MAXMEMORY, 1, crop(4, 16)).expect_err("refused"),
        "tj3Transform(): To crop this JPEG image, x must be a multiple of 8\n\
         and y must be a multiple of 8."
    );
    assert_eq!(
        transform_once(&source, TJPARAM_MAXMEMORY, 12, crop(0, 1100)).expect_err("refused"),
        "Memory limit exceeded"
    );
    assert!(transform_once(&source, TJPARAM_MAXMEMORY, 13, crop(0, 1100)).is_ok());
}

/// Codex review of #655: the markers upstream saves for copying sit in the
/// pools the memory budget covers. With 32 64 KiB APP5 segments in front of
/// the 6 MiB source, stock 3.2.0 refuses an identity transform at 8 MiB and
/// accepts it at 9; with `TJXOPT_COPYNONE` nothing is saved and 7 MiB is
/// enough (measured).
#[test]
fn saved_markers_count_against_maxmemory() {
    let source: Vec<u8> = big();
    let mut with_markers: Vec<u8> = source[..2].to_vec();
    for _ in 0..32 {
        with_markers.extend_from_slice(&[0xFF, 0xE5, 0xFF, 0xFF]);
        with_markers.extend(std::iter::repeat_n(0u8, 65_533));
    }
    with_markers.extend_from_slice(&source[2..]);
    let copy_none: TjTransform = TjTransform {
        options: libjpeg_turbo_rs_capi::transform::TJXOPT_COPYNONE,
        ..no_crop()
    };
    assert_eq!(
        transform_once(&with_markers, TJPARAM_MAXMEMORY, 8, no_crop()).expect_err("refused"),
        "Memory limit exceeded"
    );
    assert!(transform_once(&with_markers, TJPARAM_MAXMEMORY, 9, no_crop()).is_ok());
    assert!(transform_once(&with_markers, TJPARAM_MAXMEMORY, 7, copy_none).is_ok());
}

/// Issue #675 (P4-240): a zero crop width or height is `JCROP_UNSET`, "to the
/// edge" (`turbojpeg.c:2979-2986`), in `tj3Transform` and in
/// `tj3TransformBufSize` (`getTransformedSpecs`, `:2862-2865`); the port
/// refused the first and sized the second for the whole frame. The bytes are
/// held to stock's by the oracle trace's `crop_*` cases.
#[test]
fn a_zero_crop_extent_runs_to_the_edge() {
    let trace: String = our_trace();
    for case_name in ["crop_to_edge", "crop_width_to_edge", "crop_to_edge_rot90"] {
        assert_eq!(
            rc_of(line_for(&trace, "base", case_name)),
            "rc=0",
            "{case_name}"
        );
    }
    for case_name in ["crop_past_edge", "crop_origin_outside"] {
        assert_eq!(
            rc_of(line_for(&trace, "base", case_name)),
            "rc=-1",
            "{case_name}"
        );
    }
    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    let transform: TjTransform = TjTransform {
        r: TjRegion {
            x: 16,
            y: 16,
            w: 0,
            h: 0,
        },
        options: TJXOPT_CROP,
        ..no_crop()
    };
    let photo: &[u8] = FIXTURES[0].1;
    // SAFETY: live handle; `photo` is a live slice; `transform` outlives the call.
    let (bound, expected): (usize, usize) = unsafe {
        assert_eq!(tj3DecompressHeader(handle, photo.as_ptr(), photo.len()), 0);
        let bound: usize =
            libjpeg_turbo_rs_capi::transform::tj3TransformBufSize(handle, &transform);
        tj3Destroy(handle);
        (bound, libjpeg_turbo_rs_capi::tj3JPEGBufSize(48, 48, 2))
    };
    assert_eq!(bound, expected, "a 48x48 4:2:0 destination");
}

/// Issue #675 (codex review): `tj3TransformBufSize` validates the crop as
/// upstream's `getTransformedSpecs` does (`turbojpeg.c:2847-2869`) — a zero
/// extent runs to the edge, a region past the destination returns 0 with
/// upstream's message. Every value below is stock 3.2.0's for
/// `photo_64x64_420.jpg`.
#[test]
fn transform_buf_size_validates_the_crop_as_stock_does() {
    let photo: &[u8] = FIXTURES[0].1;
    let exceeds: &str =
        "tj3TransformBufSize(): The cropping region exceeds the destination image dimensions";
    let cases: [(c_int, (c_int, c_int, c_int, c_int), usize, &str); 9] = [
        (TJXOP_NONE, (16, 16, 0, 0), 8960, ""),
        (TJXOP_NONE, (64, 0, 0, 0), 0, exceeds),
        (TJXOP_NONE, (0, 64, 0, 16), 0, exceeds),
        (TJXOP_NONE, (48, 0, 32, 32), 0, exceeds),
        (
            TJXOP_NONE,
            (8, 0, 16, 16),
            0,
            "tj3TransformBufSize(): To crop this JPEG image, x must be a multiple of 16\n\
             and y must be a multiple of 16.",
        ),
        (TJXOP_ROT90, (16, 0, 0, 0), 11264, ""),
        (
            TJXOP_NONE,
            (-1, 0, 0, 0),
            0,
            "tj3TransformBufSize(): Invalid cropping region",
        ),
        (TJXOP_NONE, (0, 0, 64, 64), 14336, ""),
        (TJXOP_NONE, (0, 0, 0, 0), 14336, ""),
    ];
    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    // SAFETY: live handle; `photo` is a live slice.
    unsafe { assert_eq!(tj3DecompressHeader(handle, photo.as_ptr(), photo.len()), 0) };
    for (op, (x, y, w, h), expected, message) in cases {
        let transform: TjTransform = TjTransform {
            r: TjRegion { x, y, w, h },
            op,
            options: TJXOPT_CROP,
            ..no_crop()
        };
        // SAFETY: live handle; `transform` outlives the call.
        let (bound, error): (usize, String) = unsafe {
            let bound: usize =
                libjpeg_turbo_rs_capi::transform::tj3TransformBufSize(handle, &transform);
            let error: String = std::ffi::CStr::from_ptr(tj3GetErrorStr(handle))
                .to_string_lossy()
                .into_owned();
            (bound, error)
        };
        assert_eq!(bound, expected, "op {op} region {:?}", (x, y, w, h));
        if expected == 0 {
            assert_eq!(error, message, "op {op} region {:?}", (x, y, w, h));
        }
    }
    destroy_handle(handle);
}

fn destroy_handle(handle: *mut c_void) {
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
}

#[test]
fn oracle_source_is_present() {
    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("transform_parameters_oracle.c");
    assert!(
        source.exists(),
        "missing oracle source {}",
        source.display()
    );
}
