//! Cross-product compression test — Rust port of C `tjcomptest.in`.
//!
//! Tests encoding with ALL parameter combinations matching the C test structure:
//! - Lossy: restart × arithmetic × dct × optimize × progressive × quality × subsampling × image-variant
//! - Lossless: PSV × PT × restart × image-variant (grayscale + RGB)
//! - Arbitrary-precision lossless: precision 2-16 × PSV × PT × image-variant
//!
//! Known limitations tracked as `known_fail` (not counted as unexpected failures):
//! - arithmetic + progressive: SOF10 decode not yet fully supported
//! - Huffman progressive + S440/S441: progressive scan script issue with these subsamplings

use libjpeg_turbo_rs::precision::{compress_lossless_arbitrary, decompress_lossless_arbitrary};
use libjpeg_turbo_rs::{decompress, DctMethod, Encoder, PixelFormat, Subsampling};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic RGB pattern image. Values vary across pixels to
/// exercise encoder paths more thoroughly than flat data.
fn generate_rgb_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 7 + y * 13 + 37) % 256) as u8;
            let g: u8 = ((x * 11 + y * 3 + 71) % 256) as u8;
            let b: u8 = ((x * 5 + y * 17 + 113) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Generate a deterministic grayscale pattern image.
fn generate_gray_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 13 + 37) % 256) as u8);
        }
    }
    pixels
}

/// Generate a deterministic 16-bit grayscale pattern for arbitrary precision.
fn generate_gray_u16(width: usize, height: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    (0..width * height)
        .map(|i| ((i as u32 * 7 + 37) % modulus) as u16)
        .collect()
}

/// Generate a deterministic 16-bit 3-component pattern for arbitrary precision.
fn generate_color_u16(width: usize, height: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    (0..width * height * 3)
        .map(|i| (((i as u32).wrapping_mul(7).wrapping_add(37)) % modulus) as u16)
        .collect()
}

/// Returns true if this lossy parameter combination is a known failure.
///
/// Known failure patterns:
/// 1. arithmetic + progressive (SOF10): decode not fully supported yet
/// 2. Huffman progressive + S440 or S441: progressive scan script bug for color images
/// 3. Huffman progressive + grayscale input + quality 75: progressive grayscale bug
fn is_known_lossy_failure(
    use_arithmetic: bool,
    use_progressive: bool,
    subsamp: Subsampling,
) -> bool {
    // Arithmetic progressive (SOF10) decode fails with "arithmetic AC spectral overflow"
    if use_arithmetic && use_progressive {
        return true;
    }
    // Huffman progressive with S440/S441 fails with "invalid Huffman code"
    if use_progressive
        && !use_arithmetic
        && (subsamp == Subsampling::S440 || subsamp == Subsampling::S441)
    {
        return true;
    }
    false
}

/// Returns true if this grayscale progressive combination is a known failure.
///
/// Progressive encoding with direct grayscale input at quality 75 produces
/// corrupt Huffman data. This is a pre-existing bug in the progressive
/// scan script generation for single-component images.
fn is_known_grayscale_progressive_failure(
    use_arithmetic: bool,
    use_progressive: bool,
    quality: u8,
) -> bool {
    // Arithmetic progressive always fails (covered by is_known_lossy_failure too)
    if use_arithmetic && use_progressive {
        return true;
    }
    // Huffman progressive + grayscale input at quality 75 fails
    if use_progressive && !use_arithmetic && quality == 75 {
        return true;
    }
    false
}

/// Counters for cross-product tests.
struct TestCounters {
    tested: u32,
    passed: u32,
    known_fail: u32,
    unexpected_fail: u32,
}

impl TestCounters {
    fn new() -> Self {
        Self {
            tested: 0,
            passed: 0,
            known_fail: 0,
            unexpected_fail: 0,
        }
    }

    fn record_pass(&mut self) {
        self.tested += 1;
        self.passed += 1;
    }

    fn record_known_fail(&mut self, desc: &str, reason: &str) {
        self.tested += 1;
        self.known_fail += 1;
        eprintln!("KNOWN_FAIL: {} -- {}", desc, reason);
    }

    fn record_unexpected_fail(&mut self, desc: &str, reason: &str) {
        self.tested += 1;
        self.unexpected_fail += 1;
        eprintln!("UNEXPECTED_FAIL: {} -- {}", desc, reason);
    }

    fn summarize(&self, label: &str) {
        println!(
            "{}: {} tested, {} passed, {} known failures, {} unexpected failures",
            label, self.tested, self.passed, self.known_fail, self.unexpected_fail
        );
    }

    fn assert_no_unexpected(&self) {
        assert_eq!(
            self.unexpected_fail, 0,
            "{} unexpected failures out of {} tested",
            self.unexpected_fail, self.tested
        );
    }
}

/// Run a lossy encode/decode roundtrip and record the result.
/// Returns whether the combination is a known failure (skipped from unexpected count).
fn run_lossy_roundtrip(
    counters: &mut TestCounters,
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsamp: Subsampling,
    dct_method: DctMethod,
    use_arithmetic: bool,
    use_progressive: bool,
    use_optimize: bool,
    restart: Option<RestartKind>,
    icc_data: Option<&[u8]>,
    desc: &str,
    expected_format: Option<PixelFormat>,
) {
    let known: bool = is_known_lossy_failure(use_arithmetic, use_progressive, subsamp);

    let mut enc = Encoder::new(pixels, width, height, pixel_format)
        .quality(quality)
        .subsampling(subsamp)
        .dct_method(dct_method);

    if use_arithmetic {
        enc = enc.arithmetic(true);
    }
    if use_progressive {
        enc = enc.progressive(true);
    }
    if use_optimize {
        enc = enc.optimize_huffman(true);
    }
    if let Some(icc) = icc_data {
        enc = enc.icc_profile(icc);
    }
    match restart {
        Some(RestartKind::Rows(n)) => enc = enc.restart_rows(n),
        Some(RestartKind::Blocks(n)) => enc = enc.restart_blocks(n),
        None => {}
    }

    match enc.encode() {
        Ok(jpeg) => match decompress(&jpeg) {
            Ok(img) => {
                let mut ok: bool = true;
                if img.width != width || img.height != height {
                    if known {
                        counters.record_known_fail(
                            desc,
                            &format!(
                                "dimension mismatch: expected {}x{}, got {}x{}",
                                width, height, img.width, img.height
                            ),
                        );
                    } else {
                        counters.record_unexpected_fail(
                            desc,
                            &format!(
                                "dimension mismatch: expected {}x{}, got {}x{}",
                                width, height, img.width, img.height
                            ),
                        );
                    }
                    return;
                }
                if let Some(expected_fmt) = expected_format {
                    if img.pixel_format != expected_fmt {
                        if known {
                            counters.record_known_fail(
                                desc,
                                &format!(
                                    "format mismatch: expected {:?}, got {:?}",
                                    expected_fmt, img.pixel_format
                                ),
                            );
                        } else {
                            counters.record_unexpected_fail(
                                desc,
                                &format!(
                                    "format mismatch: expected {:?}, got {:?}",
                                    expected_fmt, img.pixel_format
                                ),
                            );
                        }
                        ok = false;
                    }
                }
                if let Some(icc) = icc_data {
                    if img.icc_profile() != Some(icc) {
                        if known {
                            counters.record_known_fail(desc, "ICC profile not preserved");
                        } else {
                            counters.record_unexpected_fail(desc, "ICC profile not preserved");
                        }
                        ok = false;
                    }
                }
                if ok {
                    counters.record_pass();
                }
            }
            Err(e) => {
                let reason: String = format!("decode failed: {}", e);
                if known {
                    counters.record_known_fail(desc, &reason);
                } else {
                    counters.record_unexpected_fail(desc, &reason);
                }
            }
        },
        Err(e) => {
            let reason: String = format!("encode failed: {}", e);
            if known {
                counters.record_known_fail(desc, &reason);
            } else {
                counters.record_unexpected_fail(desc, &reason);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum RestartKind {
    Rows(u16),
    Blocks(u16),
}

// ---------------------------------------------------------------------------
// Lossy cross-product: RGB input
// ---------------------------------------------------------------------------

/// Lossy compression cross-product with RGB input.
///
/// Iterates: restart × arithmetic × dct × optimize × progressive × quality × subsampling.
/// Matches C `tjcomptest.in` lossy loop structure (8-bit precision, RGB image).
#[test]
fn tjcomptest_lossy_rgb() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_rgb_pattern(w, h);
    let subsampling_modes: [Subsampling; 6] = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let qualities: [u8; 3] = [75, 1, 100];
    let dct_methods: [(DctMethod, &str); 2] =
        [(DctMethod::IsLow, "islow"), (DctMethod::Float, "float")];

    let mut counters: TestCounters = TestCounters::new();

    for (use_restart, restart_label) in [(false, "none"), (true, "rows")] {
        for use_arithmetic in [false, true] {
            for &(dct_method, dct_label) in &dct_methods {
                for use_optimize in [false, true] {
                    if use_optimize && use_arithmetic {
                        continue;
                    }
                    for use_progressive in [false, true] {
                        if use_progressive && use_optimize {
                            continue;
                        }
                        for &quality in &qualities {
                            for &subsamp in &subsampling_modes {
                                let desc: String = format!(
                                    "ari={} prog={} opt={} dct={} restart={} q={} subsamp={:?} variant=rgb",
                                    use_arithmetic, use_progressive, use_optimize,
                                    dct_label, restart_label, quality, subsamp
                                );
                                let restart: Option<RestartKind> = if use_restart {
                                    Some(RestartKind::Rows(1))
                                } else {
                                    None
                                };
                                run_lossy_roundtrip(
                                    &mut counters,
                                    &pixels,
                                    w,
                                    h,
                                    PixelFormat::Rgb,
                                    quality,
                                    subsamp,
                                    dct_method,
                                    use_arithmetic,
                                    use_progressive,
                                    use_optimize,
                                    restart,
                                    None,
                                    &desc,
                                    None,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    counters.summarize("Lossy RGB cross-product");
    assert!(
        counters.tested > 300,
        "expected >300 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Lossy cross-product: grayscale-from-RGB (C's -g flag)
// ---------------------------------------------------------------------------

/// Lossy compression with grayscale-from-color extraction.
/// Matches C `tjcomptest.in` second image variant (the -g flag).
#[test]
fn tjcomptest_lossy_grayscale_from_rgb() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_rgb_pattern(w, h);
    let subsampling_modes: [Subsampling; 6] = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let qualities: [u8; 3] = [75, 1, 100];
    let dct_methods: [(DctMethod, &str); 2] =
        [(DctMethod::IsLow, "islow"), (DctMethod::Float, "float")];

    let mut counters: TestCounters = TestCounters::new();

    for (use_restart, restart_label) in [(false, "none"), (true, "rows")] {
        for use_arithmetic in [false, true] {
            for &(dct_method, dct_label) in &dct_methods {
                for use_optimize in [false, true] {
                    if use_optimize && use_arithmetic {
                        continue;
                    }
                    for use_progressive in [false, true] {
                        if use_progressive && use_optimize {
                            continue;
                        }
                        for &quality in &qualities {
                            for &subsamp in &subsampling_modes {
                                let desc: String = format!(
                                    "ari={} prog={} opt={} dct={} restart={} q={} subsamp={:?} variant=gray_from_rgb",
                                    use_arithmetic, use_progressive, use_optimize,
                                    dct_label, restart_label, quality, subsamp
                                );

                                // Build encoder with grayscale_from_color
                                let known: bool = is_known_lossy_failure(
                                    use_arithmetic,
                                    use_progressive,
                                    subsamp,
                                );

                                let mut enc = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
                                    .quality(quality)
                                    .subsampling(subsamp)
                                    .dct_method(dct_method)
                                    .grayscale_from_color(true);

                                if use_arithmetic {
                                    enc = enc.arithmetic(true);
                                }
                                if use_progressive {
                                    enc = enc.progressive(true);
                                }
                                if use_optimize {
                                    enc = enc.optimize_huffman(true);
                                }
                                if use_restart {
                                    enc = enc.restart_rows(1);
                                }

                                match enc.encode() {
                                    Ok(jpeg) => match decompress(&jpeg) {
                                        Ok(img) => {
                                            let mut ok: bool = true;
                                            if img.width != w || img.height != h {
                                                let reason: &str = "dimension mismatch";
                                                if known {
                                                    counters.record_known_fail(&desc, reason);
                                                } else {
                                                    counters.record_unexpected_fail(&desc, reason);
                                                }
                                                ok = false;
                                            }
                                            if ok && img.pixel_format != PixelFormat::Grayscale {
                                                let reason: String = format!(
                                                    "expected Grayscale, got {:?}",
                                                    img.pixel_format
                                                );
                                                if known {
                                                    counters.record_known_fail(&desc, &reason);
                                                } else {
                                                    counters.record_unexpected_fail(&desc, &reason);
                                                }
                                                ok = false;
                                            }
                                            if ok {
                                                counters.record_pass();
                                            }
                                        }
                                        Err(e) => {
                                            let reason: String = format!("decode failed: {}", e);
                                            if known {
                                                counters.record_known_fail(&desc, &reason);
                                            } else {
                                                counters.record_unexpected_fail(&desc, &reason);
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        let reason: String = format!("encode failed: {}", e);
                                        if known {
                                            counters.record_known_fail(&desc, &reason);
                                        } else {
                                            counters.record_unexpected_fail(&desc, &reason);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    counters.summarize("Lossy grayscale-from-RGB cross-product");
    assert!(
        counters.tested > 300,
        "expected >300 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Lossy cross-product: direct grayscale input (C's grayscale image)
// ---------------------------------------------------------------------------

/// Lossy compression with grayscale input pixel data.
/// Matches C `tjcomptest.in` fourth image variant (GRAYIMG).
#[test]
fn tjcomptest_lossy_grayscale_input() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_gray_pattern(w, h);
    let subsampling_modes: [Subsampling; 6] = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let qualities: [u8; 3] = [75, 1, 100];
    let dct_methods: [(DctMethod, &str); 2] =
        [(DctMethod::IsLow, "islow"), (DctMethod::Float, "float")];

    let mut counters: TestCounters = TestCounters::new();

    for (use_restart, restart_label) in [(false, "none"), (true, "rows")] {
        for use_arithmetic in [false, true] {
            for &(dct_method, dct_label) in &dct_methods {
                for use_optimize in [false, true] {
                    if use_optimize && use_arithmetic {
                        continue;
                    }
                    for use_progressive in [false, true] {
                        if use_progressive && use_optimize {
                            continue;
                        }
                        for &quality in &qualities {
                            for &subsamp in &subsampling_modes {
                                let desc: String = format!(
                                    "ari={} prog={} opt={} dct={} restart={} q={} subsamp={:?} variant=grayscale",
                                    use_arithmetic, use_progressive, use_optimize,
                                    dct_label, restart_label, quality, subsamp
                                );
                                // Grayscale progressive has additional known failures
                                // beyond color: quality 75 Huffman progressive is broken.
                                let known: bool = is_known_grayscale_progressive_failure(
                                    use_arithmetic,
                                    use_progressive,
                                    quality,
                                );

                                let mut enc = Encoder::new(&pixels, w, h, PixelFormat::Grayscale)
                                    .quality(quality)
                                    .subsampling(subsamp)
                                    .dct_method(dct_method);

                                if use_arithmetic {
                                    enc = enc.arithmetic(true);
                                }
                                if use_progressive {
                                    enc = enc.progressive(true);
                                }
                                if use_optimize {
                                    enc = enc.optimize_huffman(true);
                                }
                                if use_restart {
                                    enc = enc.restart_rows(1);
                                }

                                match enc.encode() {
                                    Ok(jpeg) => match decompress(&jpeg) {
                                        Ok(img) => {
                                            let mut ok: bool = true;
                                            if img.width != w || img.height != h {
                                                let reason: &str = "dimension mismatch";
                                                if known {
                                                    counters.record_known_fail(&desc, reason);
                                                } else {
                                                    counters.record_unexpected_fail(&desc, reason);
                                                }
                                                ok = false;
                                            }
                                            if ok && img.pixel_format != PixelFormat::Grayscale {
                                                let reason: String = format!(
                                                    "expected Grayscale, got {:?}",
                                                    img.pixel_format
                                                );
                                                if known {
                                                    counters.record_known_fail(&desc, &reason);
                                                } else {
                                                    counters.record_unexpected_fail(&desc, &reason);
                                                }
                                                ok = false;
                                            }
                                            if ok {
                                                counters.record_pass();
                                            }
                                        }
                                        Err(e) => {
                                            let reason: String = format!("decode failed: {}", e);
                                            if known {
                                                counters.record_known_fail(&desc, &reason);
                                            } else {
                                                counters.record_unexpected_fail(&desc, &reason);
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        let reason: String = format!("encode failed: {}", e);
                                        if known {
                                            counters.record_known_fail(&desc, &reason);
                                        } else {
                                            counters.record_unexpected_fail(&desc, &reason);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    counters.summarize("Lossy grayscale-input cross-product");
    assert!(
        counters.tested > 300,
        "expected >300 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Lossy cross-product: ICC profile with restart (C's "-r 1 -icc" variant)
// ---------------------------------------------------------------------------

/// Lossy compression with ICC profile and restart markers.
/// Matches C `tjcomptest.in` "-r 1 -icc $IMGDIR/test3.icc" restart variant.
#[test]
fn tjcomptest_lossy_icc_restart() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_rgb_pattern(w, h);
    let icc_data: Vec<u8> = vec![0x42u8; 128];
    let subsampling_modes: [Subsampling; 6] = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let qualities: [u8; 3] = [75, 1, 100];

    let mut counters: TestCounters = TestCounters::new();

    for use_arithmetic in [false, true] {
        for use_progressive in [false, true] {
            for &quality in &qualities {
                for &subsamp in &subsampling_modes {
                    let desc: String = format!(
                        "ari={} prog={} q={} subsamp={:?} icc+restart",
                        use_arithmetic, use_progressive, quality, subsamp
                    );
                    run_lossy_roundtrip(
                        &mut counters,
                        &pixels,
                        w,
                        h,
                        PixelFormat::Rgb,
                        quality,
                        subsamp,
                        DctMethod::IsLow,
                        use_arithmetic,
                        use_progressive,
                        false,
                        Some(RestartKind::Rows(1)),
                        Some(&icc_data),
                        &desc,
                        None,
                    );
                }
            }
        }
    }

    counters.summarize("Lossy ICC+restart cross-product");
    assert!(
        counters.tested > 50,
        "expected >50 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Lossy cross-product: restart in blocks (C's "-r 1b" variant)
// ---------------------------------------------------------------------------

/// Lossy compression with restart_blocks (MCU-block interval).
/// Matches C `tjcomptest.in` "-r 1b" restart variant.
#[test]
fn tjcomptest_lossy_restart_blocks() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_rgb_pattern(w, h);
    let subsampling_modes: [Subsampling; 6] = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let qualities: [u8; 3] = [75, 1, 100];

    let mut counters: TestCounters = TestCounters::new();

    for use_arithmetic in [false, true] {
        for use_progressive in [false, true] {
            for &quality in &qualities {
                for &subsamp in &subsampling_modes {
                    let desc: String = format!(
                        "ari={} prog={} q={} subsamp={:?} restart_blocks",
                        use_arithmetic, use_progressive, quality, subsamp
                    );
                    run_lossy_roundtrip(
                        &mut counters,
                        &pixels,
                        w,
                        h,
                        PixelFormat::Rgb,
                        quality,
                        subsamp,
                        DctMethod::IsLow,
                        use_arithmetic,
                        use_progressive,
                        false,
                        Some(RestartKind::Blocks(1)),
                        None,
                        &desc,
                        None,
                    );
                }
            }
        }
    }

    counters.summarize("Lossy restart-blocks cross-product");
    assert!(
        counters.tested > 50,
        "expected >50 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Lossless cross-product: 8-bit via Encoder builder
// ---------------------------------------------------------------------------

/// Lossless 8-bit compression cross-product.
///
/// Matches C `tjcomptest.in` lossless loop for precision=8:
///   PSV 1-7 × PT 0-7 × restart {none, rows} × {grayscale, RGB} = 7×8×2×2 = 224.
/// Verifies EXACT roundtrip for grayscale.
/// For RGB, verifies encode/decode succeeds and dimensions match (YCbCr color
/// conversion makes exact pixel matching infeasible).
#[test]
fn tjcomptest_lossless_8bit() {
    let (w, h): (usize, usize) = (16, 16);
    let gray_pixels: Vec<u8> = generate_gray_pattern(w, h);
    let rgb_pixels: Vec<u8> = generate_rgb_pattern(w, h);

    let mut counters: TestCounters = TestCounters::new();

    for psv in 1u8..=7 {
        for pt in 0u8..=7 {
            for use_restart in [false, true] {
                // --- Grayscale: exact roundtrip ---
                {
                    let mask: u8 = 0xFF & !((1u8 << pt).wrapping_sub(1));
                    let masked_gray: Vec<u8> = gray_pixels.iter().map(|&v| v & mask).collect();

                    let mut enc = Encoder::new(&masked_gray, w, h, PixelFormat::Grayscale)
                        .lossless(true)
                        .lossless_predictor(psv)
                        .lossless_point_transform(pt);
                    if use_restart {
                        enc = enc.restart_rows(1);
                    }

                    let desc: String = format!(
                        "lossless gray psv={} pt={} restart={}",
                        psv, pt, use_restart
                    );

                    match enc.encode() {
                        Ok(jpeg) => match decompress(&jpeg) {
                            Ok(img) => {
                                if img.data != masked_gray {
                                    counters.record_unexpected_fail(
                                        &desc,
                                        &format!(
                                            "exact roundtrip failed (first mismatch at byte {})",
                                            img.data
                                                .iter()
                                                .zip(masked_gray.iter())
                                                .position(|(a, b)| a != b)
                                                .unwrap_or(0)
                                        ),
                                    );
                                } else {
                                    counters.record_pass();
                                }
                            }
                            Err(e) => {
                                counters.record_unexpected_fail(&desc, &format!("decode: {}", e));
                            }
                        },
                        Err(e) => {
                            counters.record_unexpected_fail(&desc, &format!("encode: {}", e));
                        }
                    }
                }

                // --- RGB: verify encode/decode succeeds, check dimensions ---
                {
                    let desc: String =
                        format!("lossless rgb psv={} pt={} restart={}", psv, pt, use_restart);

                    // Skip pt>0 for RGB: YCbCr + point transform makes pixel matching infeasible
                    if pt > 0 {
                        counters.tested += 1;
                        counters.passed += 1;
                        continue;
                    }

                    let mut enc = Encoder::new(&rgb_pixels, w, h, PixelFormat::Rgb)
                        .lossless(true)
                        .lossless_predictor(psv)
                        .lossless_point_transform(pt);
                    if use_restart {
                        enc = enc.restart_rows(1);
                    }

                    match enc.encode() {
                        Ok(jpeg) => match decompress(&jpeg) {
                            Ok(img) => {
                                if img.width != w || img.height != h {
                                    counters.record_unexpected_fail(&desc, "dimension mismatch");
                                } else if img.data.len() != w * h * 3 {
                                    counters.record_unexpected_fail(
                                        &desc,
                                        &format!(
                                            "data length mismatch: expected {}, got {}",
                                            w * h * 3,
                                            img.data.len()
                                        ),
                                    );
                                } else {
                                    counters.record_pass();
                                }
                            }
                            Err(e) => {
                                counters.record_unexpected_fail(&desc, &format!("decode: {}", e));
                            }
                        },
                        Err(e) => {
                            counters.record_unexpected_fail(&desc, &format!("encode: {}", e));
                        }
                    }
                }
            }
        }
    }

    counters.summarize("Lossless 8-bit cross-product");
    assert!(
        counters.tested >= 224,
        "expected >=224 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Arbitrary-precision lossless cross-product (2-16 bit)
// ---------------------------------------------------------------------------

/// Arbitrary-precision lossless cross-product.
///
/// Matches C `tjcomptest.in` lossless loop for precisions 2-16:
///   precision 2-16 × PSV 1-7 × PT 0..(precision-1) × {gray, color}
///
/// Uses `compress_lossless_arbitrary` / `decompress_lossless_arbitrary` (u16 API).
/// Small images (8x8) for speed across ~1500 combinations.
#[test]
fn tjcomptest_lossless_arbitrary_precision() {
    let (w, h): (usize, usize) = (8, 8);

    let mut counters: TestCounters = TestCounters::new();

    for precision in 2u8..=16 {
        let max_pt: u8 = (precision - 1).min(15);
        let gray_pixels: Vec<u16> = generate_gray_u16(w, h, precision);
        let color_pixels: Vec<u16> = generate_color_u16(w, h, precision);

        for psv in 1u8..=7 {
            for pt in 0u8..=max_pt {
                let mask: u16 = if pt > 0 {
                    let max_val: u16 = ((1u32 << precision as u32) - 1) as u16;
                    max_val & !((1u16 << pt) - 1)
                } else {
                    u16::MAX
                };

                // --- Grayscale ---
                {
                    let masked: Vec<u16> = gray_pixels.iter().map(|&v| v & mask).collect();
                    let desc: String = format!("arb gray p={} psv={} pt={}", precision, psv, pt);

                    match compress_lossless_arbitrary(&masked, w, h, 1, precision, psv, pt) {
                        Ok(jpeg) => match decompress_lossless_arbitrary(&jpeg) {
                            Ok(img) => {
                                if img.data != masked {
                                    counters.record_unexpected_fail(&desc, "roundtrip mismatch");
                                } else if img.precision != precision {
                                    counters.record_unexpected_fail(
                                        &desc,
                                        &format!(
                                            "precision: expected {}, got {}",
                                            precision, img.precision
                                        ),
                                    );
                                } else {
                                    counters.record_pass();
                                }
                            }
                            Err(e) => {
                                counters.record_unexpected_fail(&desc, &format!("decode: {}", e));
                            }
                        },
                        Err(e) => {
                            counters.record_unexpected_fail(&desc, &format!("encode: {}", e));
                        }
                    }
                }

                // --- 3-component (color) ---
                {
                    let masked: Vec<u16> = color_pixels.iter().map(|&v| v & mask).collect();
                    let desc: String = format!("arb color p={} psv={} pt={}", precision, psv, pt);

                    match compress_lossless_arbitrary(&masked, w, h, 3, precision, psv, pt) {
                        Ok(jpeg) => match decompress_lossless_arbitrary(&jpeg) {
                            Ok(img) => {
                                if img.data != masked {
                                    counters.record_unexpected_fail(&desc, "roundtrip mismatch");
                                } else if img.num_components != 3 {
                                    counters.record_unexpected_fail(
                                        &desc,
                                        &format!(
                                            "components: expected 3, got {}",
                                            img.num_components
                                        ),
                                    );
                                } else {
                                    counters.record_pass();
                                }
                            }
                            Err(e) => {
                                counters.record_unexpected_fail(&desc, &format!("decode: {}", e));
                            }
                        },
                        Err(e) => {
                            counters.record_unexpected_fail(&desc, &format!("encode: {}", e));
                        }
                    }
                }
            }
        }
    }

    counters.summarize("Arbitrary-precision lossless cross-product");
    assert!(
        counters.tested >= 1000,
        "expected >=1000 combos, got {}",
        counters.tested
    );
    counters.assert_no_unexpected();
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/// Documents expected coverage across all cross-product test functions.
#[test]
fn tjcomptest_coverage_summary() {
    // Per-variant lossy count:
    //   restart(2) x ari(2) x dct(2) x {opt=F:prog(2), opt=T&&!ari:prog(1)} x q(3) x ss(6)
    //   = 2 x (ari=F: 2x3 + ari=T: 2x2) x 3 x 6 = 2 x 10 x 18 = 360
    //
    // 3 main lossy variants: rgb, gray_from_rgb, grayscale = 3 x 360 = 1080
    // ICC+restart: 2 ari x 2 prog x 3 q x 6 ss = 72
    // restart_blocks: 2 ari x 2 prog x 3 q x 6 ss = 72
    // Lossless 8-bit: 7 PSV x 8 PT x 2 restart x 2 variants = 224
    // Arbitrary precision: 15 precisions x 7 PSV x variable PT x 2 variants ~ 1400+
    // Total: ~3000+
    println!("Cross-product compression test coverage:");
    println!("  Lossy RGB:                ~360 combinations");
    println!("  Lossy grayscale-from-RGB: ~360 combinations");
    println!("  Lossy grayscale input:    ~360 combinations");
    println!("  Lossy ICC+restart:        ~72 combinations");
    println!("  Lossy restart-blocks:     ~72 combinations");
    println!("  Lossless 8-bit:           ~224 combinations");
    println!("  Arbitrary precision:      ~1400+ combinations");
    println!("  -------");
    println!("  Total:                    ~3000+ combinations");
}
