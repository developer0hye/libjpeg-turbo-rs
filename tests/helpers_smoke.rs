//! Smoke test for shared test helpers module.

mod helpers;

#[test]
fn helpers_c_tool_discovery() {
    // P4-116: this used to look up djpeg and print SKIP when it was absent,
    // asserting nothing either way — a test that cannot fail. Assert the
    // contract the helper actually has: whatever it returns must be a path
    // that exists, and on CI it must return one.
    match helpers::djpeg_path() {
        Some(path) => assert!(
            path.exists(),
            "c_tool_path returned {path:?}, which does not exist"
        ),
        None => assert!(
            !helpers::is_ci(),
            "CI provisions libjpeg-turbo, so djpeg must be discoverable"
        ),
    }
}

#[test]
fn helpers_is_ci_returns_bool() {
    // Smoke check: `is_ci()` must not panic and must return a bool.
    // We intentionally do not assert the value, because both CI runners
    // and `cargo test` without `CI` set are valid environments for this
    // suite to run in.
    let _: bool = helpers::is_ci();
}

#[test]
fn helpers_require_c_tool_err_for_missing() {
    // `require_c_tool` must return a NotFound error for a non-existent
    // binary name.  This exercises the library-style helper that the
    // `require_c_tool!` macro delegates to.
    let err: std::io::Error = helpers::require_c_tool("definitely_not_a_real_tool_xyz_42")
        .expect_err("missing binary must yield Err");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn helpers_require_c_tool_macro_skips_locally() {
    // Locally (no `CI` env var set to a truthy value), the `require_c_tool!`
    // macro must print `SKIP: ... not found` to stderr and `return` from the
    // enclosing function rather than panic when the tool cannot be located.
    //
    // On CI the macro panics, which would cause this test to fail — skip
    // this test on CI because we cannot safely exercise the skip branch
    // there without tampering with process-global env vars (which would
    // race with other parallel tests).
    if helpers::is_ci() {
        eprintln!("SKIP: macro local-skip branch cannot be exercised on CI");
        return;
    }
    let _path: std::path::PathBuf = require_c_tool!("definitely_not_a_real_tool_xyz_42");
    // The macro must have early-returned via its SKIP path; reaching this
    // line would mean the macro did not return, which is a bug.
    unreachable!("require_c_tool! must have returned via SKIP path");
}

#[test]
fn helpers_temp_file_lifecycle() {
    let tf = helpers::TempFile::new("smoke_test.txt");
    tf.write_bytes(b"hello");
    assert!(tf.path().exists());
    let path = tf.path().to_owned();
    drop(tf);
    assert!(!path.exists(), "TempFile should auto-delete on drop");
}

#[test]
fn helpers_generate_gradient() {
    let pixels = helpers::generate_gradient(16, 16);
    assert_eq!(pixels.len(), 16 * 16 * 3);
    // Top-left pixel should be (0, 0, 0)
    assert_eq!(pixels[0], 0);
    assert_eq!(pixels[1], 0);
    assert_eq!(pixels[2], 0);
}

#[test]
fn helpers_parse_ppm_roundtrip() {
    let width: usize = 4;
    let height: usize = 3;
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);
    let ppm: Vec<u8> = helpers::build_ppm(&pixels, width, height);
    let (w, h, data) = helpers::parse_ppm(&ppm).expect("parse_ppm should succeed");
    assert_eq!(w, width);
    assert_eq!(h, height);
    assert_eq!(data, pixels);
}

#[test]
fn helpers_parse_pgm_roundtrip() {
    let width: usize = 4;
    let height: usize = 3;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
    let pgm: Vec<u8> = helpers::build_pgm(&pixels, width, height);
    let (w, h, data) = helpers::parse_pgm(&pgm).expect("parse_pgm should succeed");
    assert_eq!(w, width);
    assert_eq!(h, height);
    assert_eq!(data, pixels);
}

#[test]
fn helpers_pixel_max_diff() {
    let a: Vec<u8> = vec![100, 200, 50];
    let b: Vec<u8> = vec![100, 203, 48];
    assert_eq!(helpers::pixel_max_diff(&a, &b), 3);

    let c: Vec<u8> = vec![100, 200, 50];
    assert_eq!(helpers::pixel_max_diff(&a, &c), 0);
}

#[test]
fn helpers_assert_pixels_identical_passes() {
    let pixels: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    helpers::assert_pixels_identical(&pixels, &pixels, 2, 1, 3, "identical_test");
}

#[test]
fn helpers_build_ppm_format() {
    let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
    let ppm: Vec<u8> = helpers::build_ppm(&pixels, 3, 1);
    assert!(ppm.starts_with(b"P6\n3 1\n255\n"));
    assert_eq!(ppm.len(), "P6\n3 1\n255\n".len() + 9);
}

// ---------------------------------------------------------------------------
// ComparisonTally guard tests (P4-116)
//
// These live here rather than in `helpers/tally.rs` so they run exactly once.
// `mod helpers;` is included by dozens of integration-test binaries, and a
// `#[cfg(test)]` module inside it is compiled into every one of them — which
// would report the same five results a hundred-odd times and inflate the
// workspace test count.
// ---------------------------------------------------------------------------

#[test]
fn full_coverage_passes() {
    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new("t", 3);
    tally.compared();
    tally.compared();
    tally.compared();
    tally.finish();
}

#[test]
fn exclusions_count_toward_the_plan() {
    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new("t", 3);
    tally.compared();
    tally.excluded("no Rust equivalent");
    tally.excluded("no Rust equivalent");
    tally.finish();
}

#[test]
#[should_panic(expected = "unaccounted for")]
fn a_dropped_case_fails() {
    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new("t", 3);
    tally.compared();
    tally.finish();
}

#[test]
#[should_panic(expected = "none reached a comparison")]
fn excluding_everything_fails() {
    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new("t", 2);
    tally.excluded("tool missing");
    tally.excluded("tool missing");
    tally.finish();
}

#[test]
#[should_panic(expected = "planned zero cases")]
fn an_empty_plan_fails() {
    helpers::ComparisonTally::new("t", 0).finish();
}

/// The `Drop` guard must fire when a tally never reaches `finish()` — the
/// case a type-level `#[must_use]` cannot see, because every real call site
/// binds the tally to a variable.
#[test]
#[should_panic(expected = "dropped without finish()")]
fn a_tally_that_is_never_finished_fails() {
    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new("t", 2);
    tally.compared();
    // Falls out of scope here, exactly as an early `return` would leave it.
}
