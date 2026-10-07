//! P4-234 (#667): `tj3DecompressToYUV8` / `tj3DecompressToYUVPlanes8` at every
//! scaling factor, compared verbatim against stock TurboJPEG.
//!
//! `examples/scaled_yuv_oracle.c` decodes each stream at all sixteen factors
//! into exactly the documented `tj3YUVBufSize` / `tj3YUVPlaneSize` bytes and
//! prints an FNV-1a digest; this file runs the same loop through this crate's
//! exports. The streams span every TurboJPEG subsampling — 4:4:4, 4:2:2,
//! 4:2:0, 4:4:0, 4:1:1, 4:4:1, 4:1:0 and 2:4 — at sizes that leave ragged
//! iMCUs, plus progressive, arithmetic, restart and grayscale ones, so the
//! padding samples a plane carries beyond the image are compared too.

use std::ffi::{c_int, c_void};
use std::path::PathBuf;

mod helpers;

use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
use libjpeg_turbo_rs_capi::yuv::{tj3DecompressToYUV8, tj3DecompressToYUVPlanes8};
use libjpeg_turbo_rs_capi::{
    tj3DecompressHeader, tj3Destroy, tj3Get, tj3GetScalingFactors, tj3Init, tj3SetScalingFactor,
    tj3YUVBufSize, tj3YUVPlaneSize, TjScalingFactor,
};

const TJINIT_DECOMPRESS: c_int = 1;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJSAMP_GRAY: c_int = 3;

fn fnv1a(bytes: &[u8], hash: u64) -> u64 {
    bytes.iter().fold(hash, |hash: u64, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
    })
}

/// A deterministic RGB image with texture, so every IDCT size has detail.
fn pixels(width: usize, height: usize) -> Vec<u8> {
    (0..width * height * 3)
        .map(|i| {
            let (x, y, c) = ((i / 3) % width, i / (3 * width), i % 3);
            ((x * 7 + y * 13 + c * 40 + (x * y) % 17) % 256) as u8
        })
        .collect()
}

/// `(label, jpeg)` for every traced stream.
fn inputs() -> Vec<(String, Vec<u8>)> {
    let samplings: [(&str, (u8, u8)); 8] = [
        ("444", (1, 1)),
        ("422", (2, 1)),
        ("420", (2, 2)),
        ("440", (1, 2)),
        ("411", (4, 1)),
        ("441", (1, 4)),
        ("410", (4, 2)),
        ("24", (2, 4)),
    ];
    let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
    for (width, height) in [(227, 149), (35, 27), (17, 9)] {
        let rgb: Vec<u8> = pixels(width, height);
        for (name, luma) in samplings {
            let jpeg: Vec<u8> = Encoder::new(&rgb, width, height, PixelFormat::Rgb)
                .quality(85)
                .sampling_factors(vec![luma, (1, 1), (1, 1)])
                .encode()
                .expect("encode");
            inputs.push((format!("s{name}_{width}x{height}"), jpeg));
        }
        for (name, encoder) in [
            (
                "prog",
                Encoder::new(&rgb, width, height, PixelFormat::Rgb)
                    .sampling_factors(vec![(2, 2), (1, 1), (1, 1)])
                    .progressive(true),
            ),
            (
                "ari",
                Encoder::new(&rgb, width, height, PixelFormat::Rgb)
                    .sampling_factors(vec![(2, 1), (1, 1), (1, 1)])
                    .arithmetic(true),
            ),
            (
                "rst",
                Encoder::new(&rgb, width, height, PixelFormat::Rgb)
                    .sampling_factors(vec![(2, 2), (1, 1), (1, 1)])
                    .restart_blocks(1),
            ),
        ] {
            inputs.push((
                format!("{name}_{width}x{height}"),
                encoder.encode().expect("encode"),
            ));
        }
        let gray: Vec<u8> = rgb.iter().step_by(3).copied().collect();
        inputs.push((
            format!("gray_{width}x{height}"),
            Encoder::new(&gray, width, height, PixelFormat::Grayscale)
                .encode()
                .expect("encode"),
        ));
    }
    inputs
}

fn our_trace() -> String {
    let mut count: c_int = 0;
    // SAFETY: `count` is a live out-parameter.
    let table: *mut TjScalingFactor = unsafe { tj3GetScalingFactors(&mut count) };
    // SAFETY: the table holds `count` entries.
    let factors: &[TjScalingFactor] = unsafe { std::slice::from_raw_parts(table, count as usize) };
    let mut trace: String = String::new();
    for (label, jpeg) in inputs() {
        for factor in factors {
            let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
            assert!(!handle.is_null());
            // SAFETY: live handle; every buffer is the size the documented
            // sizing functions give, and the decoders are asked to write the
            // planes those functions describe.
            let line: String = unsafe {
                tj3DecompressHeader(handle, jpeg.as_ptr(), jpeg.len());
                tj3SetScalingFactor(handle, *factor);
                let scale = |dimension: c_int| -> c_int {
                    (dimension * factor.num + factor.denom - 1) / factor.denom
                };
                let width: c_int = scale(tj3Get(handle, TJPARAM_JPEGWIDTH));
                let height: c_int = scale(tj3Get(handle, TJPARAM_JPEGHEIGHT));
                let subsamp: c_int = tj3Get(handle, TJPARAM_SUBSAMP);
                let plane_count: usize = if subsamp == TJSAMP_GRAY { 1 } else { 3 };
                let mut packed: Vec<u8> = vec![0; tj3YUVBufSize(width, 1, height, subsamp)];
                let mut planes: Vec<Vec<u8>> = (0..plane_count)
                    .map(|p| vec![0; tj3YUVPlaneSize(p as c_int, width, 0, height, subsamp)])
                    .collect();
                let rc: c_int =
                    tj3DecompressToYUV8(handle, jpeg.as_ptr(), jpeg.len(), packed.as_mut_ptr(), 1);
                let mut pointers: Vec<*mut u8> =
                    planes.iter_mut().map(|plane| plane.as_mut_ptr()).collect();
                pointers.resize(3, std::ptr::null_mut());
                let planar_rc: c_int = tj3DecompressToYUVPlanes8(
                    handle,
                    jpeg.as_ptr(),
                    jpeg.len(),
                    pointers.as_mut_ptr(),
                    std::ptr::null(),
                );
                let basis: u64 = 14_695_981_039_346_656_037;
                let packed_hash: u64 = if rc == 0 {
                    fnv1a(&packed, basis)
                } else {
                    basis
                };
                let planar_hash: u64 = if planar_rc == 0 {
                    planes.iter().fold(basis, |hash, plane| fnv1a(plane, hash))
                } else {
                    basis
                };
                tj3Destroy(handle);
                format!(
                    "{label} {}/{} {width}x{height} rc={rc} {packed_hash:016x} planar_rc={planar_rc} {planar_hash:016x}\n",
                    factor.num, factor.denom
                )
            };
            trace.push_str(&line);
        }
    }
    trace
}

/// Issue #667: every scaled YUV decode equals stock's, byte for byte.
#[test]
fn scaled_yuv_planes_match_stock_turbojpeg() {
    let Some(oracle) = helpers::build_oracle("scaled_yuv_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             P4-234's scaled YUV planes cannot be built. Set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let workdir: PathBuf =
        std::env::temp_dir().join(format!("libjpeg_turbo_rs_p4234_{}", std::process::id()));
    std::fs::create_dir_all(&workdir).expect("create oracle workdir");
    let mut args: Vec<String> = vec![workdir.to_str().expect("utf-8 workdir").to_string()];
    for (label, jpeg) in inputs() {
        std::fs::write(workdir.join(format!("{label}.jpg")), &jpeg).expect("write stream");
        args.push(label);
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let c_trace: String = helpers::run_oracle(&oracle, &arg_refs);
    let _ = std::fs::remove_dir_all(&workdir);

    let rust_trace: String = our_trace();
    assert_eq!(rust_trace.lines().count(), 36 * 16);
    assert!(
        !rust_trace.contains("rc=-1"),
        "every stream decodes at every factor:\n{rust_trace}"
    );
    assert_eq!(
        rust_trace, c_trace,
        "scaled YUV planes differ from stock TurboJPEG"
    );
}

#[test]
fn oracle_source_is_present() {
    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("scaled_yuv_oracle.c");
    assert!(
        source.exists(),
        "missing oracle source {}",
        source.display()
    );
}
