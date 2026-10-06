//! Issue #478 (P4-139 criterion 4): `ScalingFactor` is constructed only through
//! a validated `try_new`, and no public method on it panics.
//!
//! Before this change the fields were public and `ScalingFactor::new` accepted
//! anything, so `ScalingFactor { num: 1, denom: 0 }.block_size()` — or
//! `Decoder::set_scale` with it, then a decode — hit an `assert!` on public
//! input. The validity rule is upstream TurboJPEG's: `tj3SetScalingFactor`
//! (`turbojpeg.c:2053-2058`) accepts a factor only when it is *exactly* one of
//! the sixteen entries of the `sf` table (`turbojpeg.c:199-217`), compared
//! field by field, so `4/8` is refused even though it equals `1/2`.

mod helpers;

use libjpeg_turbo_rs::tj3::TjHandle;
use libjpeg_turbo_rs::{
    calc_output_dimensions, compress, JpegError, PixelFormat, ScalingFactor, Subsampling,
};

/// Upstream's table, transcribed in its order from `turbojpeg.c:199-217`.
const UPSTREAM_SF: [(u32, u32); 16] = [
    (2, 1),
    (15, 8),
    (7, 4),
    (13, 8),
    (3, 2),
    (11, 8),
    (5, 4),
    (9, 8),
    (1, 1),
    (7, 8),
    (3, 4),
    (5, 8),
    (1, 2),
    (3, 8),
    (1, 4),
    (1, 8),
];

/// Issue #478: the supported table is upstream's, entry for entry and in order,
/// because `tj3GetScalingFactors` hands it to C callers by index.
#[test]
fn supported_table_is_upstreams_sf_table_in_order() {
    let ours: Vec<(u32, u32)> = ScalingFactor::SUPPORTED
        .iter()
        .map(|sf: &ScalingFactor| (sf.num(), sf.denom()))
        .collect();
    assert_eq!(ours, UPSTREAM_SF.to_vec());
    assert_eq!(TjHandle::scaling_factors(), UPSTREAM_SF.to_vec());
}

/// Issue #478: every factor upstream accepts constructs, round-trips through
/// the accessors, and maps to the IDCT block size `num * 8 / denom` — an exact
/// integer for every table entry, which is what keeps scaled decode output
/// unchanged by this API change.
#[test]
fn try_new_accepts_all_sixteen_upstream_factors() {
    // Upstream's table is N/8 for N = 16 down to 1, in that order.
    let block_sizes: [usize; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    for ((num, denom), expected_block) in UPSTREAM_SF.into_iter().zip(block_sizes) {
        let sf: ScalingFactor = ScalingFactor::try_new(num, denom)
            .unwrap_or_else(|e| panic!("{num}/{denom} is in upstream's table: {e}"));
        assert_eq!((sf.num(), sf.denom()), (num, denom));
        assert_eq!(sf.block_size(), expected_block, "{num}/{denom}");
    }
}

/// Issue #478: a zero denominator is refused at construction, so it can never
/// reach the division in `block_size`/`scale_dim` that used to `assert!`.
#[test]
fn try_new_refuses_a_zero_denominator() {
    for num in [0u32, 1, 2, 8, 16, u32::MAX] {
        let result: libjpeg_turbo_rs::Result<ScalingFactor> = ScalingFactor::try_new(num, 0);
        assert!(
            matches!(result, Err(JpegError::Unsupported(_))),
            "{num}/0 must be refused, got {result:?}"
        );
    }
}

/// Issue #478: factors upstream refuses are refused — including ones equal in
/// *value* to a supported factor, because upstream compares the two fields,
/// not the ratio (`turbojpeg.c:2054`).
#[test]
fn try_new_refuses_factors_upstream_refuses() {
    let refused: [(u32, u32); 12] = [
        (0, 1),
        (1, 3),
        (4, 8),
        (8, 8),
        (2, 2),
        (16, 8),
        (2, 16),
        (17, 8),
        (3, 1),
        (1, 16),
        (u32::MAX, 1),
        (1, u32::MAX),
    ];
    for (num, denom) in refused {
        let result: libjpeg_turbo_rs::Result<ScalingFactor> = ScalingFactor::try_new(num, denom);
        assert!(
            matches!(result, Err(JpegError::Unsupported(_))),
            "{num}/{denom} must be refused, got {result:?}"
        );
    }
}

/// Issue #478: over a 0..=20 square, the accepted set is exactly the table,
/// and no public method panics on any value a caller can construct — including
/// `scale_dim` at `usize::MAX`, where the old unchecked `input_dim * num`
/// overflowed. Results are checked against a `u128` model that cannot overflow:
/// the exact `ceil(dim * num / denom)` when it fits `usize`, else 0, the crate's
/// refusal value for an unrepresentable size.
#[test]
fn no_public_method_panics_over_a_num_denom_sweep() {
    let mut accepted: Vec<(u32, u32)> = Vec::new();
    for num in 0u32..=20 {
        for denom in 0u32..=20 {
            let Ok(sf) = ScalingFactor::try_new(num, denom) else {
                continue;
            };
            accepted.push((num, denom));
            let block: usize = sf.block_size();
            assert!((1..=16).contains(&block), "{num}/{denom}: block {block}");
            for dim in [0usize, 1, 7, 8, 9, 65_535, usize::MAX / 16, usize::MAX] {
                let model: u128 = (dim as u128 * num as u128).div_ceil(denom as u128);
                let expected: usize = usize::try_from(model).unwrap_or(0);
                assert_eq!(sf.scale_dim(dim), expected, "{num}/{denom} at {dim}");
            }
        }
    }
    let mut table: Vec<(u32, u32)> = UPSTREAM_SF.to_vec();
    table.sort_unstable();
    accepted.sort_unstable();
    assert_eq!(
        accepted, table,
        "accepted set must be exactly upstream's table"
    );
}

/// Issue #478: the default stays 1/1 and is a member of the table.
#[test]
fn default_is_one_to_one() {
    let sf: ScalingFactor = ScalingFactor::default();
    assert_eq!((sf.num(), sf.denom()), (1, 1));
    assert!(ScalingFactor::SUPPORTED.contains(&sf));
}

/// Issue #478: `TjHandle::set_scaling_factor` refuses exactly what `try_new`
/// refuses, so the two entry points cannot drift apart.
#[test]
fn tj_handle_agrees_with_try_new() {
    let mut handle: TjHandle = TjHandle::new();
    for num in 0u32..=20 {
        for denom in 0u32..=20 {
            assert_eq!(
                handle.set_scaling_factor(num, denom).is_ok(),
                ScalingFactor::try_new(num, denom).is_ok(),
                "{num}/{denom}"
            );
        }
    }
}

/// Issue #478: a refused factor leaves the handle's previous one in place, as
/// upstream's `THROW` before `this->scalingFactor = scalingFactor` does
/// (`turbojpeg.c:2057-2060`).
#[test]
fn a_refused_factor_keeps_the_previous_one() {
    let (w, h): (usize, usize) = (64, 48);
    let pixels: Vec<u8> = vec![128u8; w * h * 3];
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("encode fixture");
    let mut handle: TjHandle = TjHandle::new();
    handle.set_scaling_factor(1, 2).expect("1/2 is supported");
    assert!(handle.set_scaling_factor(4, 8).is_err());
    assert!(handle.set_scaling_factor(1, 0).is_err());
    let image: libjpeg_turbo_rs::Image = handle.decompress(&jpeg).expect("decode at 1/2");
    assert_eq!((image.width, image.height), (32, 24));
}

/// Issue #478: every accepted factor decodes, at the dimensions
/// `scale_dim` predicts, through the safe `Decoder` path the factor feeds.
#[test]
fn every_accepted_factor_decodes_at_scale_dim() {
    let (w, h): (usize, usize) = (37, 29);
    let pixels: Vec<u8> = (0..w * h * 3).map(|i: usize| (i * 7 % 251) as u8).collect();
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).expect("encode fixture");
    for sf in ScalingFactor::SUPPORTED {
        let mut decoder: libjpeg_turbo_rs::Decoder = libjpeg_turbo_rs::Decoder::new(&jpeg)
            .unwrap_or_else(|e| panic!("header for {}/{}: {e}", sf.num(), sf.denom()));
        decoder.set_scale(sf);
        let image: libjpeg_turbo_rs::Image = decoder
            .decode_image()
            .unwrap_or_else(|e| panic!("decode at {}/{}: {e}", sf.num(), sf.denom()));
        assert_eq!(
            (image.width, image.height),
            (sf.scale_dim(w), sf.scale_dim(h)),
            "{}/{}",
            sf.num(),
            sf.denom()
        );
    }
}

/// Issue #478: `calc_output_dimensions` takes a raw `num`/`denom` the way
/// `jpeg_calc_output_dimensions` reads `scale_num`/`scale_denom`, and used to
/// `assert!(scale_denom != 0)`. It now follows `jpeg_core_output_dimensions`
/// (`jdmaster.c`): pick the smallest block size `N` in 1..=16 with
/// `num * 8 <= denom * N` (16 if none), then output `ceil(dim * N / 8)`. That
/// rule has an answer for every input, zero denominators included, and for a
/// non-table factor it differs from the old `ceil(dim * num / denom)` — 1/3 of
/// 320 is 120 in C, not 107. Cross-checked against `djpeg -scale`.
#[test]
fn calc_output_dimensions_matches_djpeg_on_edge_pairs() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let jpeg_path: std::path::PathBuf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/photo_320x240_420.jpg");
    let cases: [(u32, u32); 16] = [
        (1, 1),
        (1, 2),
        (3, 2),
        (1, 3),
        (5, 3),
        (4, 8),
        (17, 8),
        (3, 16),
        (1, 0),
        (0, 1),
        (0, 0),
        (2, 1),
        (100, 1),
        (1, 100),
        // `scale_num * 8` wraps in C's `unsigned int` to 8: block size 8.
        (536_870_913, 1),
        // `scale_num * 8` is 0x8000_0008 and `scale_denom * N` wraps from
        // N = 2 on; C settles on N = 9 (360x270), unwrapped arithmetic on 2.
        (268_435_457, 2_147_483_649),
    ];
    for (num, denom) in cases {
        let output = std::process::Command::new(&djpeg)
            .arg("-scale")
            .arg(format!("{num}/{denom}"))
            .arg("-ppm")
            .arg(&jpeg_path)
            .output()
            .unwrap_or_else(|e| panic!("run djpeg: {e}"));
        assert!(
            output.status.success(),
            "djpeg -scale {num}/{denom}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let (c_w, c_h, _): (usize, usize, Vec<u8>) = helpers::parse_ppm(&output.stdout)
            .unwrap_or_else(|| panic!("parse djpeg PPM at {num}/{denom}"));
        assert_eq!(
            calc_output_dimensions(320, 240, num, denom),
            (c_w, c_h),
            "{num}/{denom}"
        );
    }
}

/// Issue #478: `calc_output_dimensions` reports 0 for an axis whose scaled
/// size is not representable, rather than panicking or wrapping.
#[test]
fn calc_output_dimensions_never_panics() {
    for num in [0u32, 1, 7, 16, u32::MAX] {
        for denom in [0u32, 1, 8, u32::MAX] {
            let _ = calc_output_dimensions(usize::MAX, 0, num, denom);
        }
    }
    assert_eq!(calc_output_dimensions(usize::MAX, 8, 2, 1), (0, 16));
}

/// Issue #478: `calc_output_dimensions` reports 0 only when the scaled size
/// itself does not fit. Near `usize::MAX`, multiplying first overflowed for
/// shrinking factors whose answer is representable; it must agree with
/// `ScalingFactor::scale_dim` for every supported factor at every magnitude.
#[test]
fn calc_output_dimensions_agrees_with_scale_dim_near_usize_max() {
    use libjpeg_turbo_rs::{calc_output_dimensions, ScalingFactor};
    let dims: [usize; 6] = [
        usize::MAX,
        usize::MAX - 7,
        usize::MAX / 2,
        usize::MAX / 4,
        usize::MAX / 16 + 3,
        1usize << (usize::BITS - 2),
    ];
    for factor in ScalingFactor::SUPPORTED {
        for dim in dims {
            let (width, height): (usize, usize) =
                calc_output_dimensions(dim, dim, factor.num(), factor.denom());
            let expected: usize = factor.scale_dim(dim);
            assert_eq!(
                (width, height),
                (expected, expected),
                "{}/{} of {dim}",
                factor.num(),
                factor.denom()
            );
        }
    }
    // The case the old code got wrong: 1/2 of 2^(BITS-2) is 2^(BITS-3), not 0.
    let quarter: usize = 1usize << (usize::BITS - 2);
    assert_eq!(calc_output_dimensions(quarter, 1, 1, 2).0, quarter / 2);
}
