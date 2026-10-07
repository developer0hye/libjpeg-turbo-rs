//! P4-199 (#620), P4-200 (#621), P4-203 (#625): what a decompress publishes
//! into the handle, compared verbatim against stock TurboJPEG.
//!
//! `setDecompParameters` (`references/libjpeg-turbo/src/turbojpeg.c:514-536`)
//! writes thirteen parameters from the frame header, and the shared body of
//! `tj3Decompress{8,12,16}` calls it before the `TJPARAM_MAXPIXELS` refusal
//! (`turbojpeg-mp.c:190`, `:195-198`). The port published eight from the
//! 8-bit path — the *output's* dimensions and precision among them — and three
//! from the 12/16-bit paths, which also ignored the handle's limits.
//!
//! The trace is produced by `examples/decomp_parameters_oracle.c` against real
//! TurboJPEG and mirrored here through this crate's own `tj3*` exports, over
//! the same JPEG bytes: the fixtures are embedded with `include_bytes!` and
//! written to the oracle's workdir. Cases that diverge by design or under
//! another item are listed in the oracle's header comment and are not traced.

use std::ffi::{c_int, c_void};
use std::path::PathBuf;

mod helpers;

use libjpeg_turbo_rs_capi::{
    tj3Decompress12, tj3Decompress16, tj3Decompress8, tj3DecompressHeader, tj3Destroy, tj3Get,
    tj3Init, tj3Set, tj3SetCroppingRegion, tj3SetScalingFactor, TjRegion, TjScalingFactor,
};

const TJINIT_DECOMPRESS: c_int = 1;

/// `turbojpeg.h` `TJPARAM_*`, in `setDecompParameters`' order.
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJPARAM_PRECISION: c_int = 7;
const TJPARAM_COLORSPACE: c_int = 8;
const TJPARAM_PROGRESSIVE: c_int = 12;
const TJPARAM_SCANLIMIT: c_int = 13;
const TJPARAM_ARITHMETIC: c_int = 14;
const TJPARAM_LOSSLESS: c_int = 15;
const TJPARAM_LOSSLESSPSV: c_int = 16;
const TJPARAM_LOSSLESSPT: c_int = 17;
const TJPARAM_XDENSITY: c_int = 20;
const TJPARAM_YDENSITY: c_int = 21;
const TJPARAM_DENSITYUNITS: c_int = 22;
const TJPARAM_MAXPIXELS: c_int = 24;

const PARAMS: [(c_int, &str); 13] = [
    (TJPARAM_SUBSAMP, "subsamp"),
    (TJPARAM_JPEGWIDTH, "jw"),
    (TJPARAM_JPEGHEIGHT, "jh"),
    (TJPARAM_PRECISION, "prec"),
    (TJPARAM_COLORSPACE, "cs"),
    (TJPARAM_PROGRESSIVE, "prog"),
    (TJPARAM_ARITHMETIC, "arith"),
    (TJPARAM_LOSSLESS, "lossless"),
    (TJPARAM_LOSSLESSPSV, "psv"),
    (TJPARAM_LOSSLESSPT, "pt"),
    (TJPARAM_XDENSITY, "xd"),
    (TJPARAM_YDENSITY, "yd"),
    (TJPARAM_DENSITYUNITS, "du"),
];

const TJCS_GRAY: c_int = 2;
const TJCS_CMYK: c_int = 3;
const TJCS_YCCK: c_int = 4;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;
const TJPF_CMYK: c_int = 11;

/// `(label, bytes)` — the oracle reads `<workdir>/<label>.jpg`.
const FIXTURES: [(&str, &[u8]); 9] = [
    (
        "gray",
        include_bytes!("../../../tests/fixtures/gray_8x8.jpg"),
    ),
    (
        "dense",
        include_bytes!("../../../tests/fixtures/api_sequence_color_16x16_422_dense.jpg"),
    ),
    (
        "prog",
        include_bytes!("../../../tests/fixtures/photo_64x64_420_prog.jpg"),
    ),
    (
        "arith",
        include_bytes!(
            "../../../tests/fixtures/real_world/libjpeg_testimgari_227x149_arithmetic.jpg"
        ),
    ),
    (
        "progarith",
        include_bytes!("../../../tests/fixtures/decomp_params_prog_arith_24x16_420.jpg"),
    ),
    (
        "lossless8",
        include_bytes!("../../../tests/inputs/decomp_params_lossless8_psv4_pt1_24x16.jpg"),
    ),
    (
        "lossy12",
        include_bytes!("../../../tests/fixtures/real_world/libjpeg_testorig12_227x149_12bit.jpg"),
    ),
    (
        "lossless16",
        include_bytes!("../../../tests/inputs/api_sequence_lossless16_gray_8x8.jpg"),
    ),
    (
        "cmyk",
        include_bytes!("../../../tests/fixtures/real_world/pil_cmyk.jpg"),
    ),
];

fn get(handle: *mut c_void, param: c_int) -> c_int {
    // SAFETY: `handle` is a live instance used only by this thread.
    unsafe { tj3Get(handle, param) }
}

fn instance() -> *mut c_void {
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_DECOMPRESS)");
    handle
}

fn destroy(handle: *mut c_void) {
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
}

fn emit(label: &str, case_name: &str, rc: c_int, handle: *mut c_void) -> String {
    let mut line: String = format!("{label} {case_name} rc={rc}");
    for (param, name) in PARAMS {
        line.push_str(&format!(" {name}={}", get(handle, param)));
    }
    line.push('\n');
    line
}

fn pixel_format_for(handle: *mut c_void) -> c_int {
    match get(handle, TJPARAM_COLORSPACE) {
        TJCS_GRAY => TJPF_GRAY,
        TJCS_CMYK | TJCS_YCCK => TJPF_CMYK,
        _ => TJPF_RGB,
    }
}

/// The oracle's `routed_decompress`: the entry point `precision` selects.
/// `buffer` holds `width * height * 4` two-byte samples.
fn routed_decompress(
    handle: *mut c_void,
    jpeg: &[u8],
    buffer: &mut [u16],
    precision: c_int,
    pixel_format: c_int,
) -> c_int {
    // SAFETY: `jpeg` is a live slice; `buffer` holds at least
    // width*height*4 samples of two bytes, more than any format used here
    // needs at pitch 0, for the frame whose header sized it.
    unsafe {
        if precision <= 8 {
            tj3Decompress8(
                handle,
                jpeg.as_ptr(),
                jpeg.len(),
                buffer.as_mut_ptr().cast::<u8>(),
                0,
                pixel_format,
            )
        } else if precision <= 12 {
            tj3Decompress12(
                handle,
                jpeg.as_ptr(),
                jpeg.len(),
                buffer.as_mut_ptr().cast::<i16>(),
                0,
                pixel_format,
            )
        } else {
            tj3Decompress16(
                handle,
                jpeg.as_ptr(),
                jpeg.len(),
                buffer.as_mut_ptr(),
                0,
                pixel_format,
            )
        }
    }
}

fn header(handle: *mut c_void, jpeg: &[u8]) -> c_int {
    // SAFETY: `jpeg` is a live slice of `jpeg.len()` bytes.
    unsafe { tj3DecompressHeader(handle, jpeg.as_ptr(), jpeg.len()) }
}

/// Every case for one fixture, in the oracle's order.
fn trace_label(label: &str, jpeg: &[u8], sequence: *mut c_void) -> String {
    let mut trace: String = String::new();
    if label.starts_with("headeronly_") {
        let handle: *mut c_void = instance();
        let rc: c_int = header(handle, jpeg);
        trace.push_str(&emit(label, "header", rc, handle));
        destroy(handle);
        return trace;
    }

    let probe: *mut c_void = instance();
    assert_eq!(header(probe, jpeg), 0, "{label}: probe header");
    let width: usize = get(probe, TJPARAM_JPEGWIDTH) as usize;
    let height: usize = get(probe, TJPARAM_JPEGHEIGHT) as usize;
    let precision: c_int = get(probe, TJPARAM_PRECISION);
    let lossless: c_int = get(probe, TJPARAM_LOSSLESS);
    let pixel_format: c_int = pixel_format_for(probe);
    destroy(probe);
    let mut buffer: Vec<u16> = vec![0; width * height * 4];

    let handle: *mut c_void = instance();
    let rc: c_int = header(handle, jpeg);
    trace.push_str(&emit(label, "header", rc, handle));
    let routed_precision: c_int = get(handle, TJPARAM_PRECISION);
    let rc: c_int = routed_decompress(handle, jpeg, &mut buffer, routed_precision, pixel_format);
    trace.push_str(&emit(label, "routed", rc, handle));
    destroy(handle);

    let handle: *mut c_void = instance();
    let rc: c_int = routed_decompress(handle, jpeg, &mut buffer, precision, pixel_format);
    trace.push_str(&emit(label, "direct", rc, handle));
    destroy(handle);

    let handle: *mut c_void = instance();
    // SAFETY: live handle, valid parameter.
    unsafe { tj3Set(handle, TJPARAM_MAXPIXELS, 1) };
    let rc: c_int = header(handle, jpeg);
    trace.push_str(&emit(label, "maxpixels_header", rc, handle));
    destroy(handle);
    let handle: *mut c_void = instance();
    // SAFETY: live handle, valid parameter.
    unsafe { tj3Set(handle, TJPARAM_MAXPIXELS, 1) };
    let rc: c_int = routed_decompress(handle, jpeg, &mut buffer, precision, pixel_format);
    trace.push_str(&emit(label, "maxpixels", rc, handle));
    destroy(handle);

    let handle: *mut c_void = instance();
    // SAFETY: live handle, valid parameter.
    unsafe { tj3Set(handle, TJPARAM_SCANLIMIT, 2) };
    let rc: c_int = routed_decompress(handle, jpeg, &mut buffer, precision, pixel_format);
    trace.push_str(&emit(label, "scanlimit", rc, handle));
    destroy(handle);

    let rc: c_int = routed_decompress(sequence, jpeg, &mut buffer, precision, pixel_format);
    trace.push_str(&emit(label, "sequence", rc, sequence));

    if precision == 8 && lossless == 0 {
        let handle: *mut c_void = instance();
        // SAFETY: live handle; `buffer` is large enough for the unscaled
        // frame, so for the half-scaled one too.
        let rc: c_int = unsafe {
            tj3SetScalingFactor(handle, TjScalingFactor { num: 1, denom: 2 });
            tj3Decompress8(
                handle,
                jpeg.as_ptr(),
                jpeg.len(),
                buffer.as_mut_ptr().cast::<u8>(),
                0,
                pixel_format,
            )
        };
        trace.push_str(&emit(label, "scaled", rc, handle));
        destroy(handle);

        let handle: *mut c_void = instance();
        let mut rc: c_int = header(handle, jpeg);
        if rc == 0 {
            // SAFETY: live handle whose header has been read.
            rc = unsafe {
                tj3SetCroppingRegion(
                    handle,
                    TjRegion {
                        x: 0,
                        y: 0,
                        w: 8,
                        h: 8,
                    },
                )
            };
        }
        if rc == 0 {
            // SAFETY: as above; the region is smaller than the frame.
            rc = unsafe {
                tj3Decompress8(
                    handle,
                    jpeg.as_ptr(),
                    jpeg.len(),
                    buffer.as_mut_ptr().cast::<u8>(),
                    0,
                    pixel_format,
                )
            };
        }
        trace.push_str(&emit(label, "cropped", rc, handle));
        destroy(handle);
    }
    trace
}

/// Replace the first scan's entropy data with stuffed all-ones bytes
/// (`FF 00` pairs), leaving the header intact: every Huffman lookup then reads
/// an all-ones code, which no JPEG table assigns (ITU T.81 C.2).
fn corrupt_entropy(jpeg: &[u8]) -> Vec<u8> {
    let first_sos: usize = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("an SOS marker");
    let segment_length: usize =
        (usize::from(jpeg[first_sos + 2]) << 8) | usize::from(jpeg[first_sos + 3]);
    let entropy_start: usize = first_sos + 2 + segment_length;
    let entropy_len: usize = jpeg.len() - 2 - entropy_start;
    let mut stream: Vec<u8> = jpeg[..entropy_start].to_vec();
    for _ in 0..entropy_len / 2 {
        stream.extend_from_slice(&[0xFF, 0x00]);
    }
    stream.extend_from_slice(&[0xFF, 0xD9]);
    stream
}

/// `jpeg` with its SOF0 fields patched: `precision`, `height` and `width`.
fn patched_sof(jpeg: &[u8], precision: u8, height: u16, width: u16) -> Vec<u8> {
    let sof: usize = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xC0])
        .expect("an SOF0 marker");
    let mut stream: Vec<u8> = jpeg.to_vec();
    // FF C0, Lf (2), P, Y (2), X (2).
    stream[sof + 4] = precision;
    stream[sof + 5..sof + 7].copy_from_slice(&height.to_be_bytes());
    stream[sof + 7..sof + 9].copy_from_slice(&width.to_be_bytes());
    stream
}

/// `jpeg` cut just after its first scan's SOS header and a few entropy bytes:
/// every later scan, and EOI, is gone.
fn truncated_after_first_sos(jpeg: &[u8]) -> Vec<u8> {
    let first_sos: usize = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("an SOS marker");
    let segment_length: usize =
        (usize::from(jpeg[first_sos + 2]) << 8) | usize::from(jpeg[first_sos + 3]);
    jpeg[..first_sos + 2 + segment_length + 8].to_vec()
}

/// Every traced stream: the embedded fixtures, then header-only cases
/// (P4-142) traced through `tj3DecompressHeader` alone — garbage entropy data
/// behind an intact header, a progressive stream cut after its first SOS
/// (both read), and frames `get_sof` / `initial_setup` refuse before anything
/// is published: 65,501 pixels wide, a lossy precision of 9, and a height of
/// 0 (`JERR_EMPTY_IMAGE`; no DNL support on either side).
fn inputs() -> Vec<(String, Vec<u8>)> {
    let mut inputs: Vec<(String, Vec<u8>)> = FIXTURES
        .iter()
        .map(|(label, jpeg)| (label.to_string(), jpeg.to_vec()))
        .collect();
    let baseline: &[u8] = include_bytes!("../../../tests/fixtures/photo_64x64_420.jpg");
    let progressive: &[u8] = include_bytes!("../../../tests/fixtures/photo_64x64_420_prog.jpg");
    inputs.push(("headeronly_corrupt".to_string(), corrupt_entropy(baseline)));
    inputs.push((
        "headeronly_truncprog".to_string(),
        truncated_after_first_sos(progressive),
    ));
    inputs.push((
        "headeronly_toowide".to_string(),
        patched_sof(baseline, 8, 64, 65_501),
    ));
    inputs.push((
        "headeronly_precision9".to_string(),
        patched_sof(baseline, 9, 64, 64),
    ));
    inputs.push((
        "headeronly_empty".to_string(),
        patched_sof(baseline, 8, 0, 64),
    ));
    inputs
}

fn our_trace() -> String {
    let sequence: *mut c_void = instance();
    let mut trace: String = String::new();
    for (label, jpeg) in inputs() {
        trace.push_str(&trace_label(&label, &jpeg, sequence));
    }
    destroy(sequence);
    trace
}

fn line_for<'a>(trace: &'a str, label: &str, case_name: &str) -> &'a str {
    let prefix: String = format!("{label} {case_name} ");
    trace
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{trace}"))
}

/// The whole trace, compared verbatim against real TurboJPEG.
#[test]
fn decompress_publishes_what_stock_turbojpeg_publishes() {
    let Some(oracle) = helpers::build_oracle("decomp_parameters_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             P4-199's published parameters cannot be built. Set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let workdir: PathBuf =
        std::env::temp_dir().join(format!("libjpeg_turbo_rs_p4199_{}", std::process::id()));
    std::fs::create_dir_all(&workdir).expect("create oracle workdir");
    let mut args: Vec<String> = vec![workdir.to_str().expect("utf-8 workdir").to_string()];
    for (label, jpeg) in inputs() {
        std::fs::write(workdir.join(format!("{label}.jpg")), &jpeg).expect("write fixture");
        args.push(label);
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let c_trace: String = helpers::run_oracle(&oracle, &arg_refs);
    let _ = std::fs::remove_dir_all(&workdir);

    let rust_trace: String = our_trace();
    assert_eq!(
        rust_trace, c_trace,
        "a decompress publishes different TJPARAM values than stock TurboJPEG \
         (setDecompParameters, turbojpeg.c:514-536)"
    );
}

// --- The contract stated without the oracle --------------------------------
//
// These keep the gate's teeth on a machine with no TurboJPEG development
// install. Every expected value below is the stock 3.2.0 trace's.

/// A progressive stream publishes `PROGRESSIVE = 1` and, because
/// `LOSSLESSPT` is the first scan's `Al`, the DC scan's point transform.
#[test]
fn a_progressive_decode_publishes_the_first_scans_point_transform() {
    let trace: String = our_trace();
    assert_eq!(
        line_for(&trace, "prog", "direct"),
        "prog direct rc=0 subsamp=2 jw=64 jh=64 prec=8 cs=1 prog=1 arith=0 \
         lossless=0 psv=0 pt=1 xd=1 yd=1 du=0"
    );
    assert_eq!(
        line_for(&trace, "progarith", "sequence"),
        "progarith sequence rc=0 subsamp=2 jw=24 jh=16 prec=8 cs=1 prog=1 arith=1 \
         lossless=0 psv=0 pt=1 xd=1 yd=1 du=0"
    );
    assert_eq!(
        line_for(&trace, "lossless8", "direct"),
        "lossless8 direct rc=0 subsamp=0 jw=24 jh=16 prec=8 cs=0 prog=0 arith=0 \
         lossless=1 psv=4 pt=1 xd=1 yd=1 du=0"
    );
}

/// Issue #620: `tj3Decompress12` / `tj3Decompress16` publish all thirteen and
/// refuse a frame over the handle's `TJPARAM_MAXPIXELS` — having published,
/// as upstream's shared body does.
#[test]
fn the_12_and_16_bit_entry_points_publish_and_honour_maxpixels() {
    let trace: String = our_trace();
    assert_eq!(
        line_for(&trace, "lossy12", "direct"),
        "lossy12 direct rc=0 subsamp=2 jw=227 jh=149 prec=12 cs=1 prog=0 arith=0 \
         lossless=0 psv=0 pt=0 xd=1 yd=1 du=0"
    );
    assert_eq!(
        line_for(&trace, "lossy12", "maxpixels"),
        "lossy12 maxpixels rc=-1 subsamp=2 jw=227 jh=149 prec=12 cs=1 prog=0 arith=0 \
         lossless=0 psv=0 pt=0 xd=1 yd=1 du=0"
    );
    assert_eq!(
        line_for(&trace, "lossless16", "maxpixels"),
        "lossless16 maxpixels rc=-1 subsamp=3 jw=8 jh=8 prec=16 cs=2 prog=0 arith=0 \
         lossless=1 psv=1 pt=0 xd=1 yd=1 du=0"
    );
}

/// P4-142: the header of a stream with garbage entropy data is read — and
/// a decompress of the same bytes fails, or the fixture would prove nothing.
#[test]
fn the_header_is_read_without_decoding() {
    let trace: String = our_trace();
    assert_eq!(
        line_for(&trace, "headeronly_corrupt", "header"),
        "headeronly_corrupt header rc=0 subsamp=2 jw=64 jh=64 prec=8 cs=1 prog=0 \
         arith=0 lossless=0 psv=0 pt=0 xd=1 yd=1 du=0"
    );
    let corrupt: Vec<u8> = corrupt_entropy(include_bytes!(
        "../../../tests/fixtures/photo_64x64_420.jpg"
    ));
    let handle: *mut c_void = instance();
    let mut buffer: Vec<u8> = vec![0; 64 * 64 * 3];
    // SAFETY: live handle; `buffer` holds a 64x64 RGB image at pitch 0.
    let rc: c_int = unsafe {
        tj3Decompress8(
            handle,
            corrupt.as_ptr(),
            corrupt.len(),
            buffer.as_mut_ptr(),
            0,
            TJPF_RGB,
        )
    };
    destroy(handle);
    assert_eq!(rc, -1, "the entropy data must be corrupt for a decode");
}

/// P4-200: scaling and cropping change the output, not `JPEGWIDTH` /
/// `JPEGHEIGHT`.
#[test]
fn scaled_and_cropped_decodes_publish_the_frame_dimensions() {
    let trace: String = our_trace();
    for case_name in ["scaled", "cropped"] {
        assert!(
            line_for(&trace, "prog", case_name).starts_with(&format!(
                "prog {case_name} rc=0 subsamp=2 jw=64 jh=64 prec=8 "
            )),
            "{}",
            line_for(&trace, "prog", case_name)
        );
    }
}

#[test]
fn oracle_source_is_present() {
    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("decomp_parameters_oracle.c");
    assert!(
        source.exists(),
        "missing oracle source {}",
        source.display()
    );
}
