//! P4-141 criterion 1: the concurrent one-time initialisation, raced in a
//! process where it has not happened yet.
//!
//! `common::huffman_table::std_huffman_tables` is a leaked-`Box` once-cell over
//! an `AtomicPtr` — six of the `unsafe` sites criterion 6 is blocked on, and the
//! only one-time initialisation Miri can see at all (the SIMD dispatch caches
//! are `feature = "simd"`, which the Miri legs build without). Its claim is that
//! racing initialisers each build a value, exactly one `compare_exchange`
//! publishes, and the losers drop theirs — so every reader in the process
//! observes the same `&'static`.
//!
//! Nothing tested that. `tests/concurrency.rs` decodes on ten threads, but it
//! computes a single-threaded reference decode *first*, which initialises the
//! cell before any thread starts: every one of those threads takes the
//! already-published fast path. The race is only available in a process that has
//! not touched the tables yet, which is why the race lives in its own binary
//! and does nothing before the barrier releases.
//!
//! The binary holds one other test, the `djpeg` oracle below, and it decodes —
//! so on the *native* legs, where libtest runs both in parallel, the race can
//! find the cell already published and degenerate into six fast-path reads.
//! Under Miri, which is where the claim is being checked, the oracle is
//! `#[cfg_attr(miri, ignore)]`d for want of a process to spawn, so the race is
//! the only thing in the process (`rust-code-reviewer`, 2026-09-09).
//!
//! Both routes to the cell run in that one race, because they fail differently:
//! a direct call is the raw `get_or_init`, and a baseline decode reaches it
//! through `fill_default_huffman_tables`, which clones the published `Arc`s into
//! the metadata and per-scan snapshots — a refcount bump on a pointer another
//! thread may have just published.
//!
//! **The decode fixture carries no DHT segments, and that is the mechanism.**
//! `fill_default_huffman_tables` only writes slots that marker parsing left
//! `None`, so a decode of an ordinary `compress` output *calls* the cell and
//! then clones nothing — the tables it decodes with are the ones it parsed.
//! Stripping every DHT leaves all four slots unset, so the published `Arc`s are
//! what the entropy decoder actually reads. And the stripped stream still
//! decodes to bytes identical to the unstripped one, which is the assertion
//! that pins it: that equality holds only because the tables the file omits are
//! the Annex K tables the cell publishes.
//!
//! The assertions are what makes this more than a smoke test:
//!
//! * every direct observer must see the **same four `Arc` pointers**, which is
//!   the "exactly one publish wins" claim stated as an equality. A `get_or_init`
//!   that leaked one box per caller would satisfy every other assertion here and
//!   fail this one.
//! * every decoder must produce identical bytes, and they must equal a decode
//!   of the *unstripped* fixture performed after the join — see above for why
//!   that equality is the interesting part.
//!
//! Miri adds what no native run can: its data-race detector sees an
//! unsynchronised read of a pointer another thread wrote, and
//! `-Zmiri-many-seeds` re-runs the whole test under different interleavings so
//! the loser branch — `Err(winner)`, reclaim ours, use theirs — is searched
//! rather than hoped for. That branch is reached, not hoped for: instrumenting
//! it with a counter and running it (macOS aarch64, 2026-09-09), the default
//! seed takes it 5 times and the eight seeds the CI step passes take it 40. Five
//! is the maximum — every thread but the winner — and not a structural
//! guarantee: a thread that arrives after publication takes the `Acquire` fast
//! path and never reaches the `compare_exchange` at all. The figures are
//! scheduler-dependent; what they establish is that the branch is executed
//! rather than merely compiled.
//!
//! Excluded on wasm32: WASI has no `std::thread::spawn`.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Arc, Barrier};

use libjpeg_turbo_rs::common::huffman_table::{std_huffman_tables, HuffmanTable};
use libjpeg_turbo_rs::{compress, decompress, Image, PixelFormat, Subsampling};

mod helpers;

/// Threads calling `std_huffman_tables()` directly.
const DIRECT_THREADS: usize = 4;
/// Threads reaching the same cell through a baseline decode.
const DECODE_THREADS: usize = 2;

/// 16x16 keeps the whole race inside a Miri run measured in seconds; the cell is
/// process-global, so the image size has nothing to do with what is being tested.
const SIDE: usize = 16;

/// A 16x16 baseline JPEG carrying its own Huffman tables.
fn fixture_with_tables() -> Vec<u8> {
    let pixels: Vec<u8> = (0..SIDE * SIDE * 3)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    compress(&pixels, SIDE, SIDE, PixelFormat::Rgb, 75, Subsampling::S420).expect("baseline encode")
}

/// The same stream with every DHT segment removed, so `SOS` names four table
/// slots marker parsing leaves unset and `fill_default_huffman_tables` has to
/// clone the published standard tables into them.
///
/// This is the M-JPEG shape `tests/huffman_table_defaults.rs` pins against
/// `djpeg`'s behaviour on a committed 428-byte fixture; here it is built rather
/// than embedded so the pixels stay the ones this suite compares.
fn strip_huffman_tables(jpeg: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(jpeg.len());
    out.extend_from_slice(&jpeg[..2]); // SOI
    let mut at: usize = 2;
    while at + 1 < jpeg.len() {
        assert_eq!(jpeg[at], 0xFF, "marker expected at offset {at}");
        let marker: u8 = jpeg[at + 1];
        if marker == 0xDA {
            // Entropy-coded data follows the scan header; copy the rest whole.
            out.extend_from_slice(&jpeg[at..]);
            return out;
        }
        if (0xD0..=0xD9).contains(&marker) || marker == 0x01 {
            out.extend_from_slice(&jpeg[at..at + 2]);
            at += 2;
            continue;
        }
        let length: usize = usize::from(u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]));
        if marker != 0xC4 {
            out.extend_from_slice(&jpeg[at..at + 2 + length]);
        }
        at += 2 + length;
    }
    out
}

/// The four published table addresses, as integers so they can cross a thread
/// boundary without carrying a reference.
fn table_addresses(tables: &'static [Arc<HuffmanTable>; 4]) -> [usize; 4] {
    [
        Arc::as_ptr(&tables[0]) as usize,
        Arc::as_ptr(&tables[1]) as usize,
        Arc::as_ptr(&tables[2]) as usize,
        Arc::as_ptr(&tables[3]) as usize,
    ]
}

#[test]
fn racing_initialisers_publish_exactly_one_set_of_standard_tables() {
    let with_tables: Vec<u8> = fixture_with_tables();
    let jpeg: Arc<Vec<u8>> = Arc::new(strip_huffman_tables(&with_tables));
    assert!(
        jpeg.len() < with_tables.len(),
        "the fixture kept its DHT segments, so the decode threads would parse \
         their own tables instead of reading the published ones"
    );
    let barrier: Arc<Barrier> = Arc::new(Barrier::new(DIRECT_THREADS + DECODE_THREADS));

    let direct: Vec<std::thread::JoinHandle<[usize; 4]>> = (0..DIRECT_THREADS)
        .map(|_| {
            let barrier: Arc<Barrier> = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                table_addresses(std_huffman_tables())
            })
        })
        .collect();

    let decoders: Vec<std::thread::JoinHandle<Vec<u8>>> = (0..DECODE_THREADS)
        .map(|_| {
            let barrier: Arc<Barrier> = Arc::clone(&barrier);
            let jpeg: Arc<Vec<u8>> = Arc::clone(&jpeg);
            std::thread::spawn(move || {
                barrier.wait();
                decompress(&jpeg).expect("concurrent decode").data
            })
        })
        .collect();

    let observed: Vec<[usize; 4]> = direct
        .into_iter()
        .map(|handle| handle.join().expect("direct thread panicked"))
        .collect();
    let decoded: Vec<Vec<u8>> = decoders
        .into_iter()
        .map(|handle| handle.join().expect("decode thread panicked"))
        .collect();

    let first: [usize; 4] = observed[0];
    // From 1: comparing `observed[0]` with itself is the one iteration that
    // cannot fail (rust-code-reviewer, 2026-09-09).
    for (index, addresses) in observed.iter().enumerate().skip(1) {
        assert_eq!(
            *addresses, first,
            "thread {index} saw a different set of tables: {addresses:?} vs {first:?} — \
             more than one initialiser published"
        );
    }
    // Not a tautology on the line above: it rules out the degenerate case where
    // every slot is the same address, which equality alone would accept.
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        assert_ne!(
            first[a], first[b],
            "slots {a} and {b} are the same allocation"
        );
    }

    // The *unstripped* stream, decoded after the race: it carries the tables the
    // stripped one omits, so agreement means the auto-filled tables were the
    // Annex K tables the cell published, not merely that the threads agreed
    // with each other.
    let reference: Image = decompress(&with_tables).expect("post-join reference decode");
    for (index, data) in decoded.iter().enumerate() {
        assert_eq!(
            *data, reference.data,
            "decode thread {index} disagrees with a decode of the same image \
             carrying its own Huffman tables"
        );
    }
}

/// The C oracle for the table-less fixture: `djpeg` auto-fills the same standard
/// tables, so it must produce the same pixels.
///
/// Ignored under Miri, which cannot spawn a process — the concurrency assertions
/// are what the interpreter is here for, and byte parity is what `djpeg` is here
/// for.
#[cfg_attr(miri, ignore = "Miri cannot spawn a process")]
#[test]
fn djpeg_auto_fills_the_same_standard_tables() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let stripped: Vec<u8> = strip_huffman_tables(&fixture_with_tables());

    let (width, height, c_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &stripped, "miri_once_init_no_dht");
    let ours: Image = decompress(&stripped).expect("decode the table-less fixture");
    assert_eq!((width, height), (ours.width, ours.height));
    helpers::assert_pixels_identical(
        &c_pixels,
        &ours.data,
        width,
        height,
        3,
        "miri_once_init_no_dht",
    );
}
