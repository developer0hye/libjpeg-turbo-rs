//! Metamorphic option matrix: every encoder option must have an effect, and
//! that effect must not depend on which other options are set.
//!
//! Two properties are checked over the full cross-product of builder options
//! and pixel formats:
//!
//! - **effect** — `encode(base + O)` differs from `encode(base)`. An option
//!   that changes nothing is either being dropped or is a no-op nobody noticed.
//! - **independence** — `encode(base + O1 + O2)` differs from
//!   `encode(base + O2)`, for every ordered pair. If `O1` visibly works alone
//!   but stops working once `O2` is set, `O2` is masking it.
//!
//! Neither property needs an external oracle: they compare the implementation
//! against itself. That is the whole point — this is the cheapest class of bug
//! detection available, and it is what would have caught #313 and #322
//! automatically instead of by hand.
//!
//! # Known violations
//!
//! `KNOWN_VIOLATIONS` lists the pairs that fail today, each citing the issue
//! that tracks it. The list is enforced in **both directions**: an unlisted
//! violation fails the test, and a listed one that has started passing also
//! fails, with a message to delete the entry. An exemption list that silently
//! rots into a list of things that used to be broken is worse than no list.

use libjpeg_turbo_rs::{Encoder, HuffmanTableDef, PixelFormat, Subsampling};
use std::collections::BTreeSet;

const WIDTH: usize = 48;
const HEIGHT: usize = 32;

fn pixels_for(format: PixelFormat) -> Vec<u8> {
    let channels: usize = format.bytes_per_pixel();
    let mut buffer: Vec<u8> = vec![0u8; WIDTH * HEIGHT * channels];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x3f) - 32;
            let offset: usize = (y * WIDTH + x) * channels;
            for channel in 0..channels {
                let base: i32 = match channel {
                    0 => (x * 255 / WIDTH) as i32,
                    1 => (y * 255 / HEIGHT) as i32,
                    2 => ((x ^ y) & 0xff) as i32,
                    _ => 128,
                };
                buffer[offset + channel] = (base + noise).clamp(0, 255) as u8;
            }
        }
    }
    buffer
}

/// A clearly non-default quantization table.
///
/// The base is 40, not something much coarser. At 180 every coefficient
/// quantizes to ~0, which makes `dct_method` unobservable *on top of* this
/// option — and whether it happens to remain observable turns out to be
/// backend-dependent: x86_64 still differed, aarch64 did not, so CI failed on
/// one architecture only. 40 keeps the table's own effect obvious (1143 -> 805
/// bytes on the RGB fixture) while leaving enough AC energy for the DCT choice
/// to still matter, on both backends.
fn coarse_quant() -> [u16; 64] {
    let mut table: [u16; 64] = [0; 64];
    for (index, entry) in table.iter_mut().enumerate() {
        *entry = 40 + index as u16;
    }
    table
}

fn nonstandard_dc() -> HuffmanTableDef {
    let mut bits: [u8; 17] = [0; 17];
    bits[4] = 16;
    HuffmanTableDef {
        bits,
        values: (0u8..16).collect(),
    }
}

fn nonstandard_ac() -> HuffmanTableDef {
    let mut bits: [u8; 17] = [0; 17];
    bits[5] = 16;
    bits[6] = 16;
    HuffmanTableDef {
        bits,
        values: (0u8..32).collect(),
    }
}

/// One builder option, as a name plus the mutation it applies.
struct OptionSpec {
    name: &'static str,
    apply: fn(Encoder<'_>) -> Encoder<'_>,
}

/// The options under test. Deliberately excludes ones whose effect is not a
/// pure function of the encode (metadata injection, which is additive and
/// trivially observable) and mutually exclusive mode switches such as
/// `lossless`, which cannot compose with `progressive` by construction.
const OPTIONS: &[OptionSpec] = &[
    OptionSpec {
        name: "restart_blocks",
        apply: |e| e.restart_blocks(2),
    },
    OptionSpec {
        name: "quant_table",
        apply: |e| {
            e.quant_table(0, coarse_quant())
                .quant_table(1, coarse_quant())
        },
    },
    OptionSpec {
        name: "huffman_tables",
        apply: |e| {
            e.huffman_dc_table(0, nonstandard_dc())
                .huffman_ac_table(0, nonstandard_ac())
        },
    },
    OptionSpec {
        name: "dct_method_ifast",
        apply: |e| e.dct_method(libjpeg_turbo_rs::DctMethod::IsFast),
    },
    OptionSpec {
        name: "optimize_huffman",
        apply: |e| e.optimize_huffman(true),
    },
    OptionSpec {
        name: "smoothing_factor",
        apply: |e| e.smoothing_factor(50),
    },
    OptionSpec {
        name: "progressive",
        apply: |e| e.progressive(true),
    },
    OptionSpec {
        name: "arithmetic",
        apply: |e| e.arithmetic(true),
    },
];

const FORMATS: &[(&str, PixelFormat)] = &[
    ("rgb", PixelFormat::Rgb),
    ("gray", PixelFormat::Grayscale),
    ("cmyk", PixelFormat::Cmyk),
];

/// Violations that exist today, each tied to the issue tracking it, or to
/// `by-design` where the combination is genuinely meaningless rather than
/// dropped:
///
/// - `huffman_tables after arithmetic` — arithmetic coding carries no Huffman tables
/// - `optimize_huffman after arithmetic` — arithmetic coding carries no Huffman tables
/// - `huffman_tables after optimize_huffman` — optimize_coding computes tables, overriding supplied ones (libjpeg semantics)
/// - `optimize_huffman after progressive` — the progressive path already optimizes (jcmaster.c:770-774)
/// - `huffman_tables after progressive` — a progressive scan covers one
///   coefficient band, so tables are derived per scan from that scan's own
///   statistics; a single supplied pair cannot express them. C behaves the
///   same way (`jcmaster.c:770-774` forces `optimize_coding` for progressive
///   when tables are absent, and `cjpeg -progressive` always optimizes).
/// - `optimize_huffman after smoothing_factor` — smoothing routes through the optimized path, so optimization is already on
///
///
/// Format: `"<format>|<property>|<detail>"`. Enforced in both directions —
/// see the module docs.
const KNOWN_VIOLATIONS: &[(&str, &str)] = &[
    // ---- by design: the combination is meaningless, not dropped ----
    (
        "gray|independence|huffman_tables after arithmetic",
        "by-design",
    ),
    (
        "gray|independence|huffman_tables after optimize_huffman",
        "by-design",
    ),
    (
        "gray|independence|huffman_tables after progressive",
        "by-design",
    ),
    (
        "gray|independence|optimize_huffman after arithmetic",
        "by-design",
    ),
    (
        "gray|independence|optimize_huffman after progressive",
        "by-design",
    ),
    (
        "rgb|independence|huffman_tables after arithmetic",
        "by-design",
    ),
    (
        "rgb|independence|huffman_tables after optimize_huffman",
        "by-design",
    ),
    (
        "rgb|independence|huffman_tables after progressive",
        "by-design",
    ),
    (
        "rgb|independence|optimize_huffman after arithmetic",
        "by-design",
    ),
    (
        "rgb|independence|optimize_huffman after progressive",
        "by-design",
    ),
];

fn encode(
    pixels: &[u8],
    format: PixelFormat,
    mutations: &[&OptionSpec],
) -> Result<Vec<u8>, String> {
    let mut encoder = Encoder::new(pixels, WIDTH, HEIGHT, format)
        .quality(75)
        .subsampling(Subsampling::S420);
    for mutation in mutations {
        encoder = (mutation.apply)(encoder);
    }
    encoder.encode().map_err(|error| format!("{error:?}"))
}

/// Runs the matrix and returns the set of observed violation keys.
fn collect_violations() -> BTreeSet<String> {
    let mut violations: BTreeSet<String> = BTreeSet::new();

    for &(format_name, format) in FORMATS {
        let pixels: Vec<u8> = pixels_for(format);

        let Ok(base) = encode(&pixels, format, &[]) else {
            // A format that cannot encode at all is out of scope here; other
            // suites cover that.
            continue;
        };

        // --- property: effect ---
        let mut single: Vec<(&OptionSpec, Option<Vec<u8>>)> = Vec::new();
        for option in OPTIONS {
            match encode(&pixels, format, &[option]) {
                Ok(encoded) => {
                    if encoded == base {
                        violations.insert(format!("{format_name}|effect|{}", option.name));
                    }
                    single.push((option, Some(encoded)));
                }
                // An option the format rejects outright is a visible, honest
                // failure — not the silent drop this suite hunts for.
                Err(_) => single.push((option, None)),
            }
        }

        // --- property: independence ---
        for (first, first_alone) in &single {
            // Only meaningful if the option demonstrably works on its own.
            let Some(first_alone) = first_alone else {
                continue;
            };
            if first_alone == &base {
                continue; // already reported under `effect`
            }
            for (second, second_alone) in &single {
                if std::ptr::eq(*first, *second) || second_alone.is_none() {
                    continue;
                }
                let (Ok(with_second), Ok(with_both)) = (
                    encode(&pixels, format, &[second]),
                    encode(&pixels, format, &[second, first]),
                ) else {
                    continue;
                };
                if with_both == with_second {
                    violations.insert(format!(
                        "{format_name}|independence|{} after {}",
                        first.name, second.name
                    ));
                }
            }
        }
    }

    violations
}

#[test]
fn every_encoder_option_has_an_effect_and_composes() {
    let observed: BTreeSet<String> = collect_violations();
    let known: BTreeSet<String> = KNOWN_VIOLATIONS
        .iter()
        .map(|(key, _)| (*key).to_string())
        .collect();

    let unexpected: Vec<&String> = observed.difference(&known).collect();
    let fixed: Vec<&String> = known.difference(&observed).collect();

    let mut report = String::new();
    if !unexpected.is_empty() {
        report.push_str(&format!(
            "\n{} NEW violation(s) — an option is being dropped or masked:\n",
            unexpected.len()
        ));
        for key in &unexpected {
            report.push_str(&format!("    {key}\n"));
        }
    }
    if !fixed.is_empty() {
        report.push_str(&format!(
            "\n{} known violation(s) no longer reproduce — delete them from \
             KNOWN_VIOLATIONS so the matrix stays strict:\n",
            fixed.len()
        ));
        for key in &fixed {
            let issue: &str = KNOWN_VIOLATIONS
                .iter()
                .find(|(k, _)| *k == key.as_str())
                .map(|(_, issue)| *issue)
                .unwrap_or("?");
            report.push_str(&format!("    {key}  ({issue})\n"));
        }
    }

    assert!(
        report.is_empty(),
        "encoder option matrix changed:{report}\n\
         Properties: `effect` = setting an option must change the output; \
         `independence` = 'A after B' means A's effect vanished once B was set."
    );
}
