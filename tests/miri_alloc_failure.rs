//! P4-141 criterion 1: what the library does *after* the allocator refuses.
//!
//! The criterion names "post-allocation-failure state" as a surface no job
//! covered, and the reason is that nothing in this repository could produce the
//! state at all: every existing test asks the allocator for what it can supply.
//! The fallible-allocation work (P4-136, P4-144) routes every input-derived size
//! through `common::try_alloc` so a refusal becomes
//! [`JpegError::AllocationFailed`] instead of an `abort`, and
//! `tests/decode_limits.rs` proves the *limit* checks reject oversized
//! geometry — but a limit refusing a header is not the allocator refusing a
//! buffer. The difference is the half-built object: which buffers exist, which
//! are partially written, and whether the next call reads them.
//!
//! So this suite injects the refusal. A `#[global_allocator]` fails any
//! allocation of an **exact size** armed **on the calling thread only**, so one
//! test's injection cannot reach another's (libtest runs them in parallel) and
//! the harness's own allocations are never refused. The size is exact rather
//! than a threshold because every buffer refused here comes from
//! `try_reserve_exact`, and a `>=` rule would also catch whatever else happens
//! to be large — including an *infallible* allocation, which aborts the process
//! instead of failing an assertion. Then it asserts three
//! things per case, in this order:
//!
//! 1. the call reports `AllocationFailed` rather than aborting or succeeding;
//! 2. the injector actually fired — a refusal count, because "returns an error
//!    while armed" is also what a limit check does, and a test that cannot tell
//!    those apart is measuring nothing;
//! 3. the object is still usable: the *same* decoder, after disarming, produces
//!    exactly the bytes a never-refused decode produces.
//!
//! Under Miri the third point is the one that matters. A refusal path that
//! leaves a `Vec` with a length covering unwritten bytes, or frees a buffer the
//! object still references, is invisible to a native run that happens not to
//! read the wrong byte; the interpreter reports it. This suite is in the Miri
//! job for that reason, and it runs on every native leg too, where it is a
//! plain error-path test.
//!
//! `selftest_the_injector_refuses_exactly_what_it_is_armed_for` is the committed proof
//! that the mechanism is armed — the shape the C-ABI misuse harness needed after
//! its guard pages turned out to be inert (P4-141, 2026-09-09).

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use libjpeg_turbo_rs::{
    compress, decompress, Encoder, Image, JpegError, PixelFormat, ProgressiveDecoder, Subsampling,
};

thread_local! {
    /// The exact allocation size refused on this thread, if any.
    ///
    /// An exact size rather than a threshold. A `size >= threshold` rule refuses
    /// whatever else happens to be large, and every *infallible* allocation it
    /// catches aborts the process — which is a `SIGABRT` with no assertion
    /// message, the failure mode P4-209 demonstrates. Every size this suite
    /// refuses comes from `try_reserve_exact` or `try_filled_vec`, so it is
    /// exactly predictable, and naming it means an unrelated allocation of a
    /// different size cannot be caught by accident (rust-code-reviewer,
    /// 2026-09-09).
    static REFUSE_SIZE: Cell<Option<usize>> = const { Cell::new(None) };
    /// How many allocations this thread has refused.
    static REFUSALS: Cell<usize> = const { Cell::new(0) };
}

struct RefusingAllocator;

// SAFETY: every path either forwards to `System` unchanged or returns null,
// which `GlobalAlloc` documents as the failure signal; `dealloc` is never
// intercepted, `realloc` refuses on the *new* size and leaves the caller's block
// untouched and owned, and nothing here forges a reference or assumes a layout.
//
// Consulting the thread-local state cannot re-enter the allocator: a
// `const`-initialised `Cell` with no destructor lowers to native
// `#[thread_local]` storage on every target this suite runs on (Linux, macOS and
// Windows; wasm32 is excluded above), so `LocalKey::with` neither boxes the value
// on first access nor registers a destructor. On a hypothetical target without
// native TLS the first access would allocate, and refusing that allocation would
// recurse.
unsafe impl GlobalAlloc for RefusingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if refuse(layout.size()) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A grow through `try_reserve_exact` on a `Vec` that already owns
        // capacity arrives here, not in `alloc`. Refusing only `alloc` would
        // have left every `Vec::try_reserve` on a non-empty buffer unrefusable.
        if refuse(new_size) {
            return core::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if refuse(layout.size()) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
}

fn refuse(size: usize) -> bool {
    REFUSE_SIZE.with(|armed| {
        if armed.get() == Some(size) {
            REFUSALS.with(|count| count.set(count.get() + 1));
            true
        } else {
            false
        }
    })
}

#[global_allocator]
static GLOBAL: RefusingAllocator = RefusingAllocator;

/// Disarms the injector when it goes out of scope.
///
/// A `Drop` guard rather than a straight-line restore, so that an unwinding
/// `body` cannot leave the thread armed for whatever libtest runs next.
///
/// What it does **not** do is protect the panic runtime: a payload is allocated
/// and a message formatted inside `body`'s own frame, before unwinding reaches
/// this guard. The first version of this comment claimed otherwise
/// (rust-code-reviewer, 2026-09-09). Refusing exactly one size, rather than
/// everything above a threshold, is what keeps that allocation out of the
/// injector's way.
struct Armed;

impl Drop for Armed {
    fn drop(&mut self) {
        REFUSE_SIZE.with(|armed| armed.set(None));
    }
}

/// Run `body` with allocations of exactly `size` bytes refused, returning its
/// value and how many refusals it caused.
///
/// The thread-local cells are touched before arming so that their own
/// first-access bookkeeping, if any, is never the allocation that gets refused.
fn with_refusals_of<T>(size: usize, body: impl FnOnce() -> T) -> (T, usize) {
    REFUSALS.with(|count| count.set(0));
    REFUSE_SIZE.with(|armed| armed.set(None));

    let guard: Armed = Armed;
    REFUSE_SIZE.with(|armed| armed.set(Some(size)));
    let value: T = body();
    drop(guard);

    (value, REFUSALS.with(|count| count.get()))
}

/// Side of the progressive fixtures.
const SIDE: usize = 32;

/// The progressive destination image: `width * height * 3` bytes, allocated by
/// `try_filled_vec` in `api::progressive_output` and therefore refusable.
const PIXELS_BYTES: usize = SIDE * SIDE * 3;

/// Bytes of ICC profile to embed for the metadata case. `common::icc`
/// reassembles the chunks into one `try_reserved_vec` of exactly this size.
///
/// Deliberately not a round number. The exact-size rule exists so an
/// *infallible* allocation is never refused — that is a `SIGABRT` with no
/// assertion message rather than a test failure — and a power of two is exactly
/// what `RawVec`'s doubling growth and fixed-size buffers land on. 16 KiB was
/// the first draft; the profile is synthetic, so nothing is lost by choosing a
/// size no growth policy produces (rust-code-reviewer, 2026-09-09). The
/// destination sizes below cannot be chosen so freely — they are fixed by the
/// fixture geometry that gives those images a C oracle — which is what the
/// refusal-count assertions are for.
const ICC_BYTES: usize = 16_007;

fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8);
            pixels.push(((x * 3 + y * 7 + 50) % 256) as u8);
            pixels.push(((x * 5 + y * 5 + 100) % 256) as u8);
        }
    }
    pixels
}

/// Side of the fixture in the ignored mainline test. 64 rather than 32 so its
/// 12288-byte destination is a size nothing else in that decode allocates —
/// at 32x32 the destination is 3072 bytes, and the ICC case already refuses on
/// that path's numbers.
const MAINLINE_SIDE: usize = 64;

/// The reference decode below is our own decoder, so say where the C oracle for
/// these bytes lives: without an ICC profile this is byte-for-byte the fixture
/// `tests/miri_public_api.rs` hands to `djpeg` in
/// `djpeg_agrees_with_every_fixture_this_suite_builds` (same pixels, quality and
/// subsampling), and that test requires diff = 0.
fn progressive_fixture(icc: Option<&[u8]>) -> Vec<u8> {
    let pixels: Vec<u8> = gradient_rgb(SIDE, SIDE);
    let mut encoder: Encoder<'_> = Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .progressive(true);
    if let Some(profile) = icc {
        encoder = encoder.icc_profile(profile);
    }
    encoder.encode().expect("progressive encode")
}

/// A deterministic blob the size of a real ICC profile, with the four-byte
/// big-endian size field a profile starts with so the decoder keeps it.
fn icc_profile() -> Vec<u8> {
    let mut profile: Vec<u8> = Vec::with_capacity(ICC_BYTES);
    profile.extend_from_slice(&(ICC_BYTES as u32).to_be_bytes());
    while profile.len() < ICC_BYTES {
        profile.push((profile.len() % 251) as u8);
    }
    profile
}

/// The mechanism, proved before anything relies on it.
#[test]
fn selftest_the_injector_refuses_exactly_what_it_is_armed_for() {
    let armed: usize = 4096;

    let (outcomes, count) = with_refusals_of(armed, || {
        let mut refused: Vec<u8> = Vec::new();
        let mut allowed: Vec<u8> = Vec::new();
        (
            refused.try_reserve_exact(armed).is_err(),
            // One byte off the armed size: an exact-match injector must let it
            // through, which is what keeps an unrelated allocation from being
            // caught by accident.
            allowed.try_reserve_exact(armed + 1).is_ok(),
        )
    });
    assert!(
        outcomes.0,
        "the injector allowed an allocation it was armed for"
    );
    assert!(
        outcomes.1,
        "the injector refused a size it was not armed for"
    );
    assert_eq!(count, 1, "exactly one refusal expected");

    // And it is off again afterwards, so a later test in this binary is not
    // running against a still-armed allocator.
    let mut buffer: Vec<u8> = Vec::new();
    buffer
        .try_reserve_exact(armed)
        .expect("allocator refuses after disarming");
    assert_eq!(
        REFUSALS.with(|count| count.get()),
        1,
        "no refusal may happen after disarming"
    );
}

/// The half-built object case: a `ProgressiveDecoder` whose `output()` was
/// refused mid-reconstruction is still a valid decoder.
///
/// `output()` allocates the destination image, fills it component by component
/// and hands it over. A refusal partway leaves the decoder's own coefficient
/// state untouched *if* the buffer being filled was the only thing lost —
/// nothing but an interpreter can tell that claim from a native run that
/// happens not to read a stale byte, which is why this suite is in the Miri
/// job.
#[test]
fn a_refused_progressive_output_leaves_the_decoder_usable() {
    let jpeg: Vec<u8> = progressive_fixture(None);
    let reference: Image = decompress(&jpeg).expect("unrefused reference decode");

    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).expect("open progressive");
    while decoder.consume_input().expect("consume scan") {}
    assert!(decoder.input_complete());

    let (result, refusals) = with_refusals_of(PIXELS_BYTES, || decoder.output());
    let error: JpegError = result.expect_err("a refused allocation must not produce an image");
    // Naming the buffer, not just the variant: `AllocationFailed` from some
    // other call site would satisfy `matches!` while proving nothing about the
    // destination image this test is aimed at.
    assert!(
        matches!(
            &error,
            JpegError::AllocationFailed { what, bytes }
                if *what == "progressive output image" && *bytes == (SIDE * SIDE * 3) as u64
        ),
        "want the destination image refused, got {error:?}"
    );
    assert_eq!(
        refusals, 1,
        "exactly the destination image must be refused; a different count means \
         the injector caught something else"
    );

    // Same decoder, same coefficients, allocator no longer refusing.
    let image: Image = decoder.output().expect("output after a refused output");
    assert_eq!(image.data, reference.data);
}

/// The metadata half of the same question, on a different fallible call site.
///
/// A multi-chunk ICC profile is reassembled by `common::icc` into one buffer
/// sized by the *file*, not by the geometry — the amplification P4-144 was filed
/// for. Refusing it exercises a refusal that happens after the pixels are
/// already decoded and while the assembled `Image` is being built, which is a
/// different post-failure state from the one above: the coefficients are spent
/// and the component planes are reconstructed, but the reassembly runs *before*
/// the destination image is allocated (`api::progressive_output`, the
/// `try_reassemble_icc_profile` call above `assemble_ycbcr`), so the refusal
/// lands ahead of that allocation rather than at it.
#[test]
fn a_refused_icc_reassembly_leaves_the_decoder_usable() {
    let profile: Vec<u8> = icc_profile();
    let jpeg: Vec<u8> = progressive_fixture(Some(&profile));
    let reference: Image = decompress(&jpeg).expect("unrefused reference decode");

    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).expect("open progressive");
    while decoder.consume_input().expect("consume scan") {}

    let (result, refusals) = with_refusals_of(ICC_BYTES, || decoder.output());
    let error: JpegError = result.expect_err("a refused profile must not produce an image");
    assert!(
        matches!(
            &error,
            JpegError::AllocationFailed { what, bytes }
                if *what == "ICC profile" && *bytes == ICC_BYTES as u64
        ),
        "want the ICC reassembly refused, got {error:?}"
    );
    assert_eq!(refusals, 1, "exactly the profile buffer must be refused");

    let image: Image = decoder.output().expect("output after a refused reassembly");
    assert_eq!(image.data, reference.data);
    assert_eq!(
        image.icc_profile.as_deref(),
        Some(profile.as_slice()),
        "the profile must survive its own refused reassembly"
    );
}

/// The contract for the *primary* decode entry point, which does not hold today.
///
/// `decompress` allocates its destination with `vec![0u8; size]`
/// (`decode/pipeline_impl/output.rs`, `take_out_buf`) where `size` comes from
/// the SOF, so an allocator refusal aborts the process instead of reporting
/// [`JpegError::AllocationFailed`] — the uncatchable denial of service P4-136
/// criterion 4 and P4-144 removed from the progressive, arithmetic, ICC and
/// marker paths. Verified by patching that one site to `try_filled_vec`, after
/// which this test passes unchanged.
///
/// Ignored rather than deleted, and rather than asserting the abort: an
/// assertion that the process dies would pin the defect as the contract. This is
/// the regression test for P4-209 (#632), and deleting the `#[ignore]` is what
/// closes it.
#[test]
#[ignore = "P4-209 (#632): the mainline decode destination is allocated infallibly, so an allocator refusal aborts"]
fn the_mainline_decode_reports_refusal_instead_of_aborting() {
    let side: usize = MAINLINE_SIDE;
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(side, side),
        side,
        side,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("baseline encode");
    let reference: Image = decompress(&jpeg).expect("unrefused reference decode");

    // The destination is `width * height * 3` = 12288 bytes for this fixture.
    let (result, refusals) = with_refusals_of(side * side * 3, || decompress(&jpeg));
    let error: JpegError = result.expect_err("a refused allocation must not decode");
    assert!(
        matches!(error, JpegError::AllocationFailed { .. }),
        "want AllocationFailed, got {error:?}"
    );
    assert_eq!(refusals, 1, "exactly the destination must be refused");

    let again: Image = decompress(&jpeg).expect("decode after a refused decode");
    assert_eq!(again.data, reference.data);
}
