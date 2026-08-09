//! P4-126: the YUV plane-dimension entry points must bound `componentID` the
//! way stock libjpeg-turbo does, across *every* `TJSAMP_*`.
//!
//! Upstream applies the bound once, in `tj3YUVPlaneWidth` /
//! `tj3YUVPlaneHeight` (`references/libjpeg-turbo/src/turbojpeg.c:1115`):
//!
//! ```c
//!   nc = (subsamp == TJSAMP_GRAY ? 1 : 3);
//!   if (componentID < 0 || componentID >= nc)
//!     THROWG("Invalid argument", 0);
//! ```
//!
//! and the legacy `tjPlaneWidth` / `tjPlaneHeight` inherit it by **delegating**:
//! `return (tj3YUVPlaneWidth(...) == 0) ? -1 : retval;`. This port did not
//! delegate — the legacy pair called the root-crate `yuv_plane_width` /
//! `yuv_plane_height` directly, and those take a `Subsampling` that has no
//! grayscale variant (`TJSAMP_GRAY` maps to `S444` in `subsamp_from_c`). So a
//! grayscale image's non-existent chroma planes were reported as full-size
//! planes instead of being rejected. Same shape as P4-125: upstream needs one
//! guard because it delegates; this port needs the delegation or a second
//! guard.
//!
//! Rather than assert a handful of hand-picked cases, this drives
//! `examples/yuv_plane_index_c_oracle.c` over the entire
//! (`TJSAMP_*` x `componentID` in -1..=4) matrix and requires every cell to
//! match.

mod helpers;

use std::ffi::c_int;

use libjpeg_turbo_rs_capi::{tj3YUVPlaneHeight, tj3YUVPlaneWidth, tjPlaneHeight, tjPlaneWidth};

/// Deliberately not a multiple of any subsampling factor, so a plane that is
/// wrongly sized as chroma cannot coincide with the luma value through
/// rounding.
const WIDTH: c_int = 100;
const HEIGHT: c_int = 100;

/// One row of the oracle's matrix.
#[derive(Debug, PartialEq, Eq)]
struct Row {
    subsamp: c_int,
    component_id: c_int,
    plane_width: c_int,
    plane_height: c_int,
    yuv_plane_width: c_int,
    yuv_plane_height: c_int,
}

fn parse(stdout: &str) -> Vec<Row> {
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let f: Vec<c_int> = line
                .split_whitespace()
                .map(|t| {
                    t.parse::<c_int>()
                        .unwrap_or_else(|e| panic!("oracle emitted non-integer {t:?}: {e}"))
                })
                .collect();
            assert_eq!(f.len(), 6, "oracle row has {} fields: {line:?}", f.len());
            Row {
                subsamp: f[0],
                component_id: f[1],
                plane_width: f[2],
                plane_height: f[3],
                yuv_plane_width: f[4],
                yuv_plane_height: f[5],
            }
        })
        .collect()
}

/// Our exported symbols, evaluated for the same cell the oracle reported.
fn ours(subsamp: c_int, component_id: c_int) -> Row {
    Row {
        subsamp,
        component_id,
        plane_width: tjPlaneWidth(component_id, WIDTH, subsamp),
        plane_height: tjPlaneHeight(component_id, HEIGHT, subsamp),
        yuv_plane_width: tj3YUVPlaneWidth(component_id, WIDTH, subsamp),
        yuv_plane_height: tj3YUVPlaneHeight(component_id, HEIGHT, subsamp),
    }
}

/// P4-126: every (subsamp, componentID) cell must agree with stock
/// libjpeg-turbo. Measured against 3.1.4.1, the pre-fix mismatches were the
/// grayscale chroma indices: `tjPlaneWidth(1, 100, TJSAMP_GRAY)` returned 100
/// where C returns -1.
#[test]
fn plane_dimension_component_bound_matches_c() {
    let Some(oracle) = helpers::build_oracle("yuv_plane_index_c_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; set LIBJPEG_TURBO_PREFIX to make \
             this parity check mandatory"
        );
        return;
    };

    let stdout: String = helpers::run_oracle(&oracle, &[&WIDTH.to_string(), &HEIGHT.to_string()]);
    let expected: Vec<Row> = parse(&stdout);
    assert!(
        !expected.is_empty(),
        "oracle produced no rows — a silently empty matrix would pass every assertion below"
    );

    let mut mismatches: Vec<String> = Vec::new();
    for row in &expected {
        let got: Row = ours(row.subsamp, row.component_id);
        if got != *row {
            mismatches.push(format!(
                "  subsamp={} componentID={}\n    C  : tjPlaneWidth={} tjPlaneHeight={} \
                 tj3YUVPlaneWidth={} tj3YUVPlaneHeight={}\n    ours: tjPlaneWidth={} \
                 tjPlaneHeight={} tj3YUVPlaneWidth={} tj3YUVPlaneHeight={}",
                row.subsamp,
                row.component_id,
                row.plane_width,
                row.plane_height,
                row.yuv_plane_width,
                row.yuv_plane_height,
                got.plane_width,
                got.plane_height,
                got.yuv_plane_width,
                got.yuv_plane_height,
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} plane-dimension cells disagree with stock libjpeg-turbo:\n{}",
        mismatches.len(),
        expected.len(),
        mismatches.join("\n")
    );
}
