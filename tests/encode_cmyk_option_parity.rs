//! Issue #313: the CMYK encode path silently drops per-variant options.
//!
//! The four baseline `compress_*` entry points share one early return into
//! `compress_cmyk(pixels, width, height, quality, subsampling)` — a signature
//! that cannot carry restart intervals, custom tables, or a DCT method — so
//! those options are discarded without any error reaching the caller.
//! `compress_optimized`, which owns smoothing, rejects CMYK outright instead.
//!
//! Upstream libjpeg-turbo gates none of these on colorspace: `optimize_coding`
//! (`jcmaster.c`), `restart_interval` (`jchuff.c`), `smoothing_factor`
//! (`jcsample.c`, per-component with a `smoothok` fallback), quantization
//! slots (`jcparam.c`) and `dct_method` (`jcdctmgr.c`) all apply to CMYK.
//!
//! These tests assert the *correct* (C-matching) behaviour and are therefore
//! expected to fail until issue #313 is fixed. They are `#[ignore]`d rather
//! than weakened, per the project's strict-assertion rules.

use libjpeg_turbo_rs::encode::pipeline;
use libjpeg_turbo_rs::{DctMethod, Encoder, HuffmanTableDef, PixelFormat, Subsampling};

/// Deterministic 4-channel CMYK test image with both smooth and hard-edged
/// content, so that quantization and entropy-coding changes are observable.
fn cmyk_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 4];
    let mut rng_state: u32 = 0x9e37_79b9;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let in_rect: bool = x * 3 >= width && x * 3 < width * 2;
            let offset: usize = (y * width + x) * 4;
            pixels[offset] = ((x * 255 / width.max(1)) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / height.max(1)) as i32 - noise).clamp(0, 255) as u8;
            pixels[offset + 2] = (if in_rect { 220 } else { 40 } + noise).clamp(0, 255) as u8;
            pixels[offset + 3] = ((x + y) as i32 % 256).clamp(0, 255) as u8;
        }
    }
    pixels
}

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

/// Counts RST0..RST7 markers (0xFFD0..0xFFD7) in an entropy-coded stream.
fn count_restart_markers(jpeg: &[u8]) -> usize {
    jpeg.windows(2)
        .filter(|window| window[0] == 0xFF && (0xD0..=0xD7).contains(&window[1]))
        .count()
}

#[test]
#[ignore = "Issue #313: compress_with_restart drops restart_interval for CMYK"]
fn cmyk_honours_restart_interval() {
    let pixels: Vec<u8> = cmyk_image(WIDTH, HEIGHT);
    let with_restart: Vec<u8> = pipeline::compress_with_restart(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
        2,
        DctMethod::IsLow,
    )
    .expect("CMYK encode with restart interval must succeed");

    let restarts: usize = count_restart_markers(&with_restart);
    // 64x48 at 4:4:4 is 8x6 = 48 MCUs; a restart interval of 2 yields 23
    // restart markers (one after every 2 MCUs except the final group).
    assert!(
        restarts > 0,
        "expected RST markers in CMYK output with restart_interval=2, found none \
         — the interval was silently discarded (issue #313)"
    );
    assert_eq!(
        restarts, 23,
        "CMYK restart marker count must match the MCU count / interval"
    );
}

#[test]
#[ignore = "Issue #313: compress_custom_quant drops custom tables for CMYK"]
fn cmyk_honours_custom_quant_tables() {
    let pixels: Vec<u8> = cmyk_image(WIDTH, HEIGHT);

    let mut coarse: [u16; 64] = [0; 64];
    for (index, entry) in coarse.iter_mut().enumerate() {
        // Deliberately far coarser than any quality-75 default table, so the
        // encoded size must drop sharply if the table is actually applied.
        *entry = 200 + index as u16;
    }
    let custom: [Option<[u16; 64]>; 4] = [Some(coarse), Some(coarse), None, None];

    let with_custom: Vec<u8> = pipeline::compress_custom_quant(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
        &custom,
    )
    .expect("CMYK encode with custom quant tables must succeed");

    let default: Vec<u8> = pipeline::compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    )
    .expect("CMYK default encode must succeed");

    assert_ne!(
        with_custom, default,
        "CMYK output with custom quant tables is byte-identical to the default \
         encode — the tables were silently discarded (issue #313)"
    );
    assert!(
        with_custom.len() < default.len(),
        "a much coarser quant table must produce a smaller file: custom={} default={}",
        with_custom.len(),
        default.len()
    );
}

#[test]
#[ignore = "Issue #313: compress_custom_huffman drops custom tables for CMYK"]
fn cmyk_honours_custom_huffman_tables() {
    let pixels: Vec<u8> = cmyk_image(WIDTH, HEIGHT);

    // A valid but deliberately non-standard flat table: 16 DC codes of length 4.
    let mut dc_bits: [u8; 17] = [0; 17];
    dc_bits[4] = 16;
    let dc_def = HuffmanTableDef {
        bits: dc_bits,
        values: (0u8..16).collect(),
    };
    let mut ac_bits: [u8; 17] = [0; 17];
    ac_bits[5] = 16;
    ac_bits[6] = 16;
    let ac_def = HuffmanTableDef {
        bits: ac_bits,
        values: (0u8..32).collect(),
    };

    let with_custom: Vec<u8> = pipeline::compress_custom_huffman(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
        &[Some(dc_def.clone()), Some(dc_def), None, None],
        &[Some(ac_def.clone()), Some(ac_def), None, None],
    )
    .expect("CMYK encode with custom Huffman tables must succeed");

    let default: Vec<u8> = pipeline::compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    )
    .expect("CMYK default encode must succeed");

    assert_ne!(
        with_custom, default,
        "CMYK output with custom Huffman tables is byte-identical to the default \
         encode — the tables were silently discarded (issue #313)"
    );
}

#[test]
#[ignore = "Issue #313: compress_optimized rejects CMYK, so Encoder::optimize_huffman fails"]
fn cmyk_supports_optimized_huffman() {
    let pixels: Vec<u8> = cmyk_image(WIDTH, HEIGHT);

    let optimized: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .quality(75)
        .subsampling(Subsampling::S444)
        .optimize_huffman(true)
        .encode()
        .expect("Encoder::optimize_huffman must work on CMYK, as cjpeg -optimize does");

    let default: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("CMYK default encode must succeed");

    assert!(
        optimized.len() < default.len(),
        "optimized-Huffman CMYK output must be smaller than default-table output: \
         optimized={} default={}",
        optimized.len(),
        default.len()
    );
}

#[test]
#[ignore = "Issue #313: compress_optimized rejects CMYK, so Encoder::smoothing_factor fails"]
fn cmyk_supports_smoothing_factor() {
    let pixels: Vec<u8> = cmyk_image(WIDTH, HEIGHT);

    let smoothed: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .quality(75)
        .subsampling(Subsampling::S444)
        .smoothing_factor(50)
        .encode()
        .expect("Encoder::smoothing_factor must work on CMYK, as cjpeg -smooth does");

    let unsmoothed: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("CMYK default encode must succeed");

    assert_ne!(
        smoothed, unsmoothed,
        "smoothing_factor=50 produced byte-identical CMYK output — smoothing was \
         not applied (issue #313)"
    );
}

#[test]
#[ignore = "Issue #313: compress_cmyk ignores dct_method"]
fn cmyk_honours_dct_method() {
    let pixels: Vec<u8> = cmyk_image(WIDTH, HEIGHT);

    let encode = |dct_method: DctMethod| -> Vec<u8> {
        pipeline::compress(
            &pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Cmyk,
            75,
            Subsampling::S444,
            dct_method,
        )
        .unwrap_or_else(|error| panic!("CMYK encode with {dct_method:?} failed: {error:?}"))
    };

    let islow: Vec<u8> = encode(DctMethod::IsLow);
    let ifast: Vec<u8> = encode(DctMethod::IsFast);
    let float: Vec<u8> = encode(DctMethod::Float);

    // IsFast uses the AA&N algorithm with 8-bit fixed point; it cannot produce
    // coefficients identical to the 13-bit IsLow path on real image content.
    assert_ne!(
        islow, ifast,
        "IsLow and IsFast produced byte-identical CMYK output — dct_method was \
         ignored (issue #313)"
    );
    assert_ne!(
        islow, float,
        "IsLow and Float produced byte-identical CMYK output — dct_method was \
         ignored (issue #313)"
    );
}
