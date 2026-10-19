//! Differential fuzzing of `precision::decompress_lossless_arbitrary` (SOF3,
//! precision 2–16) against libjpeg-turbo's own lossless decoder, through
//! `examples/lossless_decode_c_oracle.c`.
//!
//! `fuzz_decompress_precision` drives this decoder for panics only, so it
//! cannot see a wrong sample, and `fuzz_decode_diff_c` compares the 8-bit
//! `Decoder` path, not this one (P4-66). The input is a JPEG; only SOF3
//! frames are compared.
//!
//! Asserts, when libjpeg decodes the frame without a warning:
//! 1. Rust accepts it too (drop-in floor);
//! 2. width, height and component count agree;
//! 3. every sample is identical — lossless decoding has no tolerance.
//!
//! Known divergences are skipped unless `LTRS_FUZZ_INCLUDE_KNOWN=1`:
//! - P4-247: a multi-component frame with any sampling factor other than
//!   1×1 (wrong samples, `Ok`), and a first scan coding fewer components than
//!   the frame, which libjpeg decodes across its later scans and this entry
//!   point refuses (P4-75 made that a `CorruptData` refusal; 0.8.0 panicked).
//!   Restart intervals, its part 1, are compared: they are fixed.
//! - P4-250: a frame-header Tq or first-scan Ta above 3, which libjpeg
//!   ignores in lossless mode and Rust refuses.
//! - P4-66: below 16 bits, where libjpeg carries a corrupt difference's sum
//!   modulo 2^16 and Rust masks it to the precision.
//!
//! libjpeg is handed a lone component's sampling factors as 1×1 (see
//! [`oracle_input`]); a libjpeg bug would otherwise fail restart streams that
//! Rust decodes correctly.

#![no_main]

#[path = "common/c_oracle.rs"]
mod c_oracle;

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::precision::decompress_lossless_arbitrary;

const MAX_FUZZ_SAMPLES: u64 = 1_048_576;

fn oracle() -> &'static PathBuf {
    static ORACLE: OnceLock<PathBuf> = OnceLock::new();
    ORACLE.get_or_init(|| c_oracle::oracle_path("LOSSLESS_C_ORACLE"))
}

/// What the marker segments say, read without decoding.
struct Layout {
    width: u64,
    height: u64,
    /// `(h, v)` sampling factors per component.
    sampling: Vec<(u8, u8)>,
    first_scan_components: Option<u8>,
    /// The largest Tq in the frame header and Ta in the first scan header.
    max_unused_selector: u8,
    /// Offset of the sampling-factor byte of a one-component frame.
    lone_sampling_at: Option<usize>,
}

/// Walk the marker segments up to the first SOS. Returns `None` unless the
/// first frame header is SOF3 (Huffman lossless).
fn read_layout(data: &[u8]) -> Option<Layout> {
    if data.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut i: usize = 2;
    let mut layout: Option<Layout> = None;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let marker: u8 = data[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        let length: usize = usize::from(u16::from_be_bytes([data[i + 2], data[i + 3]]));
        let segment: &[u8] = data.get(i + 4..i + 2 + length)?;
        match marker {
            0xC3 if layout.is_none() => {
                let components: usize = usize::from(*segment.get(5)?);
                let sampling: Vec<(u8, u8)> = (0..components)
                    .map(|k| segment.get(7 + 3 * k).map(|&s| (s >> 4, s & 0x0F)))
                    .collect::<Option<_>>()?;
                layout = Some(Layout {
                    width: u64::from(u16::from_be_bytes([*segment.get(3)?, *segment.get(4)?])),
                    height: u64::from(u16::from_be_bytes([*segment.get(1)?, *segment.get(2)?])),
                    sampling,
                    first_scan_components: None,
                    max_unused_selector: (0..components)
                        .filter_map(|k| segment.get(8 + 3 * k).copied())
                        .max()
                        .unwrap_or(0),
                    lone_sampling_at: (components == 1).then_some(i + 4 + 7),
                });
            }
            // A DCT or arithmetic frame before any SOF3: not this decoder's.
            0xC0..=0xCF if marker != 0xC4 && marker != 0xC8 && marker != 0xCC => {
                layout.as_ref()?;
            }
            0xDA => {
                let mut layout: Layout = layout?;
                layout.first_scan_components = segment.first().copied();
                let ns: usize = usize::from(segment.first().copied().unwrap_or(0));
                layout.max_unused_selector = (0..ns)
                    .filter_map(|k| segment.get(2 + 2 * k).map(|&tdta| tdta & 0x0F))
                    .chain([layout.max_unused_selector])
                    .max()
                    .unwrap_or(0);
                return Some(layout);
            }
            _ => {}
        }
        i += 2 + length;
    }
    None
}

/// What the C oracle decodes: `data`, with a lone component's sampling
/// factors set to 1×1 when they are valid.
///
/// They do not change a one-component scan, which is non-interleaved (T.81
/// A.2.2), and Rust ignores them. libjpeg does not: for a factor v > 1 it
/// decodes v rows per iMCU row and only then undifferences them
/// (`jddiffct.c` `decompress_data`), so a restart's prediction reset lands
/// on the iMCU row's first row. Any restart interval that is not a multiple
/// of v rows then decodes wrong — checked 2026-10-17 with libjpeg-turbo
/// 3.2.0: a `cjpeg -lossless 1 -restart 2` stream relabelled 1×3 gives 444
/// of 703 samples wrong, where Rust returns the source. libjpeg's own
/// encoder never writes such a frame: it forces 1×1 in lossless mode.
///
/// Factors outside 1–4 stay as they are: both decoders refuse them
/// (libjpeg's `JERR_BAD_SAMPLING`, `jdinput.c` `initial_setup`), and the
/// rewrite would let only libjpeg decode the frame.
fn oracle_input<'a>(data: &'a [u8], layout: &Layout) -> Cow<'a, [u8]> {
    let valid = |factor: u8| (1..=4).contains(&factor);
    match layout.lone_sampling_at {
        Some(at) if data[at] != 0x11 && valid(data[at] >> 4) && valid(data[at] & 0x0F) => {
            let mut patched: Vec<u8> = data.to_vec();
            patched[at] = 0x11;
            Cow::Owned(patched)
        }
        _ => Cow::Borrowed(data),
    }
}

fuzz_target!(|data: &[u8]| {
    c_oracle::ensure_sigpipe_ignored();
    let oracle: &PathBuf = oracle();

    let Some(layout) = read_layout(data) else {
        return;
    };
    let components: u64 = layout.sampling.len() as u64;
    if layout.width == 0
        || layout.height == 0
        || components == 0
        || layout.width * layout.height * components > MAX_FUZZ_SAMPLES
    {
        return;
    }
    if !c_oracle::include_known_gaps() {
        // P4-247 (3): a first scan coding fewer components than the frame is
        // refused (P4-75), where libjpeg decodes the later scans too.
        if layout.first_scan_components != Some(layout.sampling.len() as u8) {
            return;
        }
        // P4-247 (2): subsampled multi-component frames decode wrong.
        if layout.sampling.len() > 1 && layout.sampling.iter().any(|&s| s != (1, 1)) {
            return;
        }
        // P4-250: Tq or Ta above 3 is refused, though neither selects
        // anything in a lossless frame and libjpeg ignores both.
        if layout.max_unused_selector > 3 {
            return;
        }
    }

    let c_result = c_oracle::run_oracle(
        oracle,
        &["/dev/stdin".to_string()],
        &oracle_input(data, &layout),
    );
    let rust_result = decompress_lossless_arbitrary(data);

    let c = match c_result {
        None => return,
        Some(c) if c.warned => return,
        Some(c) => c,
    };
    let (header, body) = c_oracle::split_header(&c.stdout).expect("lossless oracle header");
    let (cw, ch, cc, precision): (usize, usize, usize, u32) = match header[..] {
        ["LL", w, h, n, p] => (
            w.parse().unwrap(),
            h.parse().unwrap(),
            n.parse().unwrap(),
            p.parse().unwrap(),
        ),
        _ => panic!("malformed lossless oracle header {header:?}"),
    };
    let c_samples: Vec<u16> = body
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| u16::from_le_bytes(pair))
        .collect();
    let image = match rust_result {
        Ok(image) => image,
        Err(error) => panic!(
            "drop-in regression: libjpeg decoded this {cw}x{ch}x{cc} P{precision} lossless frame \
             cleanly but Rust rejected it: {error}"
        ),
    };
    assert_eq!(
        (image.width, image.height, image.num_components),
        (cw, ch, cc),
        "lossless dimensions diverge (P{precision})"
    );
    assert_eq!(image.data.len(), c_samples.len(), "lossless sample count");
    if image.data == c_samples {
        return;
    }
    let differing: Vec<usize> = (0..c_samples.len())
        .filter(|&i| image.data[i] != c_samples[i])
        .collect();
    // P4-66: libjpeg keeps a wrapped sum's high bits below 16-bit precision;
    // Rust masks them. Every difference of that kind agrees after masking.
    if precision < 16 && !c_oracle::include_known_gaps() {
        let mask: u16 = ((1u32 << precision) - 1) as u16;
        if differing
            .iter()
            .all(|&i| c_samples[i] & mask == image.data[i])
        {
            return;
        }
    }
    let first: Vec<String> = differing
        .iter()
        .take(6)
        .map(|&i| {
            format!(
                "[sample {} ch {}] rust {} c {}",
                i / cc,
                i % cc,
                image.data[i],
                c_samples[i]
            )
        })
        .collect();
    // Below 16 bits a wrapped sum also changes later predictions (P4-66), so
    // a P4-66 divergence can spread past samples the filter above can match;
    // replaying the input against a build that wraps modulo 2^16 separates
    // those from anything new.
    panic!(
        "lossless samples diverge from libjpeg ({cw}x{ch}x{cc} P{precision}): {} of {} differ; first: {}",
        differing.len(),
        c_samples.len(),
        first.join(", ")
    );
});
