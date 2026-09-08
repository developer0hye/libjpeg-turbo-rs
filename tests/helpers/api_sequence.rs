//! API-sequence engine: ordered `TjHandle` lifecycles with a state-leakage
//! oracle.
//!
//! P4-141 criterion 3 (#480) asks for "an API-sequence fuzzer alongside the
//! byte fuzzers, driving `new → configure → probe → decode → reset → decode →
//! transform → destroy` orderings". No existing byte fuzzer touches
//! `TjHandle`, mixes operation kinds on one handle, or compares a result
//! against a handle built fresh from the same configuration — the two that do
//! call twice (`fuzz_decompress`'s owned-then-sink decodes,
//! `fuzz_progressive_decoder`'s `consume_input` loop) only repeat one
//! operation — so the defect class that needs *two* calls to appear, an
//! operation that reads what an earlier one left on the handle, is
//! unreachable from all twelve of them.
//!
//! This file is included by path from three places. The first two are so that
//! the property CI proves on every pull request and the property libFuzzer
//! searches on the 6-hourly schedule are the same code:
//!
//! * `tests/api_sequence_state.rs` — deterministic programs over committed
//!   fixtures;
//! * `fuzz/fuzz_targets/fuzz_api_sequence.rs` — [`program_from_bytes`] turns
//!   libFuzzer's byte string into a program and the rest of the input into the
//!   first of the JPEGs the program operates on;
//! * `tests/generate_fuzz_seeds.rs` — [`encode_program`] assembles the
//!   committed seeds, so the wire format has one definition rather than two.
//!
//! # Why a program holds more than one input
//!
//! A program that only ever decodes *one* image cannot see a stale published
//! parameter: the value an earlier call left behind is the value this call
//! would write anyway, so P4 below compares a number against itself. The
//! engine therefore takes a list of inputs and [`Op::SelectInput`] switches
//! between them, and [`BUILTIN_INPUTS`] guarantees at least two that differ in
//! dimensions, component count, colour space and subsampling even when the
//! caller supplies one. Injecting a "publish only on the first decode" bug into
//! `TjHandle::decompress` is caught by P4 with the switch and passes without
//! it.
//!
//! Two images are not enough either. Measured through `TjHandle::decompress`,
//! the committed fixtures all report `PRECISION = 8` and the JFIF default
//! density `0 / 1 / 1`, which are also what `TjHandle::new()` initialises —
//! so four of P4's eight comparisons were a number against itself *and*
//! against the fresh-handle default. [`BUILTIN_INPUTS`] therefore carries
//! three images chosen so every published parameter can take at least two
//! values across them, and
//! `the_builtin_inputs_can_move_every_published_parameter` in
//! `tests/api_sequence_state.rs` fails if that stops being true.
//!
//! # The oracle
//!
//! Four properties, each stated in terms of the documented contract rather
//! than of the current implementation:
//!
//! * **P1, decode purity.** For every decode-family operation, the result on a
//!   handle that has already been used equals the result on a handle created
//!   fresh and given only the *configuration* operations of the prefix.
//!   `TjHandle::decompress` reads scaling, cropping, `STOPONWARNING`,
//!   `FASTUPSAMPLE`, `FASTDCT`, `SAVEMARKERS`, `BOTTOMUP` and the resource
//!   limits — all writable parameters — so a fresh handle carrying the same
//!   writes must produce the same bytes. Anything else it reads is state that
//!   leaked from an earlier call.
//! * **P2, compress depends on the configuration and the most recent
//!   published header, and on nothing else.** `tj3DecompressHeader` and
//!   `tj3Decompress8` deliberately publish header facts into the handle
//!   (`JPEGWIDTH`, `JPEGHEIGHT`, `PRECISION`, `COLORSPACE`, `SUBSAMP`, the
//!   densities), and a later compress is documented to read them —
//!   `setCompDefaults` takes the densities from the handle at
//!   `turbojpeg.c:376-378` and the sampling factors at `:418-422`. So the
//!   reference for a compress is a fresh handle replaying the configuration
//!   *plus the last publishing operation that succeeded*, on the image it ran
//!   on. Skipping the compress comparison after any write-back — the first
//!   draft — removed every sequence in which it could have differed, which is
//!   the same compare-nothing defect the multi-input section describes.
//!   `tj3Decompress12` / `tj3Decompress16` are not publishing operations for
//!   this purpose: they write `JPEGWIDTH`, `JPEGHEIGHT` and `PRECISION`, none
//!   of which `configure_encoder` reads.
//! * **P3, determinism of the handle-free entry points.**
//!   `transform_jpeg_with_options` takes no handle at all, so two identical
//!   calls differing is global state. It is deliberately *not* a statement
//!   about handle state — P1 covers repetition of a stateful operation,
//!   because the live handle's n-th decode is compared against a fresh
//!   handle's first.
//! * **P4, write-back agreement, in both directions.** After a *successful*
//!   decode-family operation, every parameter that operation documents it
//!   writes holds the same value on the used handle as on the fresh one — and
//!   every parameter it does *not* document writing is unchanged from before
//!   the call. The first half catches a value an earlier call left behind that
//!   this one failed to overwrite; the second catches an operation that writes
//!   a parameter its contract does not mention, which is how `Op::InspectHeader`
//!   is held to writing nothing at all.
//!
//! # What it deliberately does not assert
//!
//! Upstream keeps two ICC buffers — `iccBuf`, written only by
//! `tj3SetICCProfile` and read by the compress and transform paths
//! (`turbojpeg.c:111` declares both; the readers are `tj3Compress*` at
//! `turbojpeg-mp.c:126`, `tj3CompressFromYUVPlanes8` at `turbojpeg.c:1367`
//! and `tj3Transform` at `:3045`), and `decompICCBuf`, written only by
//! the decompressor (`:1909-1913`) — so in C a decode can never change the profile
//! a later compress embeds. `TjHandle` merges them into one field, which is
//! filed as its own gap; P2 does not cover it because P2 stops at the first
//! write-back. That property joins this oracle when the gap closes, rather
//! than being pinned here in its current shape.

#![allow(dead_code)]

use libjpeg_turbo_rs::tj3::{FrameInfo, TjHandle, TjParam};
use libjpeg_turbo_rs::{
    transform_jpeg_with_options, CropRegion, Decoder, MarkerCopyMode, PixelFormat, TransformOp,
    TransformOptions,
};

/// Upper bound on the operations one program may hold.
///
/// Each compared operation builds a fresh reference handle and replays the
/// configuration *setters* onto it before making one call, so the decode cost
/// is linear in the operation count and only the `set()` calls are quadratic —
/// and those are free next to a decode.
pub const MAX_OPS: usize = 16;

/// Bytes each operation occupies in the fuzzer's wire format.
const OP_RECORD_LEN: usize = 4;

/// Distinct opcodes; `record[0] % OPCODE_COUNT` selects one.
///
/// One per [`Op`] variant. A new variant must bump this *and* join the
/// round-trip program in `tests/api_sequence_state.rs`: leaving it here would
/// make [`encode_op`] emit a code `op_from_record` folds back onto `Op::Set`,
/// and that round-trip test is the only thing that would notice.
const OPCODE_COUNT: u8 = 13;

/// The parameters a program may write.
///
/// `MAXMEMORY` and `MAXPIXELS` are excluded on purpose: they are the caps this
/// harness uses to keep one input from allocating without bound, and a fuzzed
/// value would turn a finding into an out-of-memory kill of the whole run.
const FUZZABLE_PARAMS: &[TjParam] = &[
    TjParam::Quality,
    TjParam::Subsampling,
    TjParam::Width,
    TjParam::Height,
    TjParam::Precision,
    TjParam::ColorSpace,
    TjParam::FastUpSample,
    TjParam::FastDct,
    TjParam::Optimize,
    TjParam::Progressive,
    TjParam::ScanLimit,
    TjParam::Arithmetic,
    TjParam::Lossless,
    TjParam::LosslessPsv,
    TjParam::LosslessPt,
    TjParam::RestartBlocks,
    TjParam::RestartRows,
    TjParam::XDensity,
    TjParam::YDensity,
    TjParam::DensityUnits,
    TjParam::BottomUp,
    TjParam::NoRealloc,
    TjParam::StopOnWarning,
    TjParam::SaveMarkers,
];

/// Parameters `tj3DecompressHeader` / `tj3Decompress8` publish from the frame
/// header. Compared after a successful 8-bit decode-family operation (P4).
const WRITE_BACK_8BIT: &[TjParam] = &[
    TjParam::Width,
    TjParam::Height,
    TjParam::Precision,
    TjParam::ColorSpace,
    TjParam::Subsampling,
    TjParam::XDensity,
    TjParam::YDensity,
    TjParam::DensityUnits,
];

/// What `TjHandle::decompress_12bit` / `decompress_16bit` document they write.
const WRITE_BACK_PRECISION: &[TjParam] = &[TjParam::Width, TjParam::Height, TjParam::Precision];

/// Every parameter `TjHandle::get` answers.
///
/// P4's second half asserts that an operation leaves everything outside its
/// documented write set untouched, so this list has to be complete rather than
/// interesting: a parameter missing here is one an operation could scribble on
/// unnoticed. `every_parameter_is_in_all_params` in
/// `tests/api_sequence_state.rs` fails if a new `TjParam` variant does not
/// reach it.
pub const ALL_PARAMS: &[TjParam] = &[
    TjParam::Quality,
    TjParam::Subsampling,
    TjParam::Width,
    TjParam::Height,
    TjParam::Precision,
    TjParam::ColorSpace,
    TjParam::FastUpSample,
    TjParam::FastDct,
    TjParam::Optimize,
    TjParam::Progressive,
    TjParam::ScanLimit,
    TjParam::Arithmetic,
    TjParam::Lossless,
    TjParam::LosslessPsv,
    TjParam::LosslessPt,
    TjParam::RestartBlocks,
    TjParam::RestartRows,
    TjParam::XDensity,
    TjParam::YDensity,
    TjParam::DensityUnits,
    TjParam::MaxMemory,
    TjParam::MaxPixels,
    TjParam::BottomUp,
    TjParam::NoRealloc,
    TjParam::StopOnWarning,
    TjParam::SaveMarkers,
];

/// Pixel formats a compress operation may be handed. `Rgb565` is decode-output
/// only (`common/types.rs`), so it is not an encoder input.
const COMPRESS_FORMATS: &[PixelFormat] = &[
    PixelFormat::Grayscale,
    PixelFormat::Rgb,
    PixelFormat::Rgba,
    PixelFormat::Bgr,
    PixelFormat::Bgra,
    PixelFormat::Cmyk,
    PixelFormat::Rgbx,
    PixelFormat::Bgrx,
    PixelFormat::Xrgb,
    PixelFormat::Xbgr,
    PixelFormat::Argb,
    PixelFormat::Abgr,
];

/// Images every program can reach regardless of what the caller supplied.
///
/// Chosen so that each parameter a decode publishes takes at least two values
/// across them — without that, P4 compares a number against itself. Measured
/// through `TjHandle` as `[Width, Height, Precision, ColorSpace, Subsampling,
/// XDensity, YDensity, DensityUnits]`:
///
/// | input | op that succeeds | published |
/// |---|---|---|
/// | `gray_8x8.jpg` | `decompress` | `8, 8, 8, 2, 3, 1, 1, 0` |
/// | `api_sequence_color_16x16_422_dense.jpg` | `decompress` | `16, 16, 8, 1, 1, 72, 71, 1` |
/// | `api_sequence_lossless16_gray_8x8.jpg` | `decompress_16bit` | `8, 8, 16, -1, -1, 1, 1, 0` |
///
/// The second is `cjpeg -quality 80 -sample 2x1` output whose JFIF APP0
/// density field was then set to `units = 1, 72 x 71`: `cjpeg` has no
/// `-density` switch, and every committed fixture carries the JFIF default
/// `0 / 1 / 1`, which is also `TjHandle::new()`'s initial value — so with
/// those alone the three density comparisons could not fail. The third is
/// `cjpeg -precision 16 -lossless 1`, and it is the only way `PRECISION` ever
/// holds anything but 8 before an 8-bit decode publishes over it.
pub const BUILTIN_INPUTS: &[&[u8]] = &[
    include_bytes!("../fixtures/gray_8x8.jpg"),
    include_bytes!("../fixtures/api_sequence_color_16x16_422_dense.jpg"),
    include_bytes!("../fixtures/api_sequence_lossless16_gray_8x8.jpg"),
];

/// The input list the fuzz target builds: the fuzzed body first, then every
/// [`BUILTIN_INPUTS`] entry in order.
///
/// The seed generator writes `Op::SelectInput` indices against this layout and
/// the target consumes it, so it is defined here rather than in either — the
/// first version hard-coded two of the three built-ins in the target while the
/// seeds addressed three, which silently pointed the precision-transition
/// program's `Decompress16` at an 8-bit image.
pub fn fuzz_inputs<'a>(body: &'a [u8]) -> Vec<&'a [u8]> {
    let mut inputs: Vec<&'a [u8]> = Vec::with_capacity(1 + BUILTIN_INPUTS.len());
    inputs.push(body);
    inputs.extend_from_slice(BUILTIN_INPUTS);
    inputs
}

/// `Op::SelectInput` index of the fuzzed body under [`fuzz_inputs`].
pub const FUZZ_INPUT_BODY: u8 = 0;
/// `Op::SelectInput` index of `gray_8x8.jpg` under [`fuzz_inputs`].
pub const FUZZ_INPUT_GRAY: u8 = 1;
/// `Op::SelectInput` index of the density-carrying 4:2:2 image.
pub const FUZZ_INPUT_COLOR_DENSE: u8 = 2;
/// `Op::SelectInput` index of the 16-bit lossless image — the only one
/// `Decompress16` accepts, and so the only way `PRECISION` reaches 16.
pub const FUZZ_INPUT_LOSSLESS16: u8 = 3;

/// One call in a program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// `tj3Set(param, value)`. Configuration: replayed onto the reference.
    Set { param: TjParam, value: i32 },
    /// `tj3SetScalingFactor`, by index into `TjHandle::scaling_factors()`.
    SetScaling { index: u8 },
    /// `tj3SetCroppingRegion`.
    SetCrop { region: Option<CropRegion> },
    /// `tj3SetICCProfile`.
    SetIcc { profile: Option<Vec<u8>> },
    /// `tj3DecompressHeader` — the criterion's "probe".
    DecompressHeader,
    /// `TjHandle::inspect_header` — the header-only probe, which takes
    /// `&self` and therefore writes nothing back.
    InspectHeader,
    /// `tj3Decompress8`.
    Decompress,
    /// `tj3Decompress12`.
    Decompress12,
    /// `tj3Decompress16`.
    Decompress16,
    /// `tj3Compress8` over a synthetic image of the given size and format.
    Compress { width: u8, height: u8, format: u8 },
    /// `tj3Transform`.
    Transform { op: u8, flags: u8 },
    /// Point the following operations at another of the caller's images.
    ///
    /// Not configuration: it is which buffer the caller passes to the next
    /// call, so the reference handle is handed the same one.
    SelectInput { index: u8 },
    /// `tj3Destroy` immediately followed by `tj3Init` — the criterion's
    /// "reset". Discards the handle and the configuration replayed onto the
    /// reference.
    Reset,
}

impl Op {
    /// Whether this operation is configuration — the part of a prefix a fresh
    /// reference handle is allowed to see.
    pub fn is_config(&self) -> bool {
        matches!(
            self,
            Op::Set { .. } | Op::SetScaling { .. } | Op::SetCrop { .. } | Op::SetIcc { .. }
        )
    }

    /// Whether this operation publishes the header facts a later *compress*
    /// reads.
    ///
    /// `tj3Decompress12` / `tj3Decompress16` are excluded deliberately: they
    /// write `JPEGWIDTH`, `JPEGHEIGHT` and `PRECISION`, and
    /// `configure_encoder` reads none of the three. Including them would make
    /// P2's reference replay an operation that changes nothing it compares,
    /// while hiding the last operation that does.
    pub fn publishes_for_compress(&self) -> bool {
        matches!(self, Op::DecompressHeader | Op::Decompress)
    }
}

/// How the reference handle is rebuilt before each compared operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferencePolicy {
    /// The oracle: replay exactly the configuration operations issued since
    /// the last [`Op::Reset`].
    ConfigOnly,
    /// Deliberately broken — replay nothing. Used only by the harness's own
    /// mutation check, which requires this to be *detected*; without it,
    /// `ConfigOnly` passing would say nothing about whether the comparison
    /// can fail at all.
    NoConfig,
    /// Deliberately broken the other way — replay the configuration but not
    /// the last publishing operation, so a compress reading a published fact
    /// must diverge. The committed can-fail proof for P2, which is otherwise
    /// the one property with no mutation check of its own.
    NoWriteBackReplay,
}

impl ReferencePolicy {
    fn replays_config(self) -> bool {
        !matches!(self, ReferencePolicy::NoConfig)
    }

    fn replays_write_back(self) -> bool {
        matches!(self, ReferencePolicy::ConfigOnly)
    }
}

/// Resource ceiling for one program. Kept out of [`FUZZABLE_PARAMS`] so a
/// program cannot raise its own cap.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Frame-header pixels — width x height as the SOF declares them, before
    /// any scaling — above which an input is dropped from the program.
    pub max_pixels: u64,
}

/// What one [`run_program`] call did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    /// Operations executed on the live handle.
    pub executed: usize,
    /// Every input's frame header exceeded [`Limits::max_pixels`], so nothing
    /// ran. A single oversize input is blanked instead — it keeps its slot, so
    /// the `Op::SelectInput` indices still name the images they name, and the
    /// rest of a fuzzer's program stays useful when only `inputs[0]` is
    /// enormous.
    pub skipped_oversize: bool,
    /// Decode-family operations whose live call returned `Err`.
    ///
    /// Reported so a test can assert that a *refusal* happened rather than
    /// asserting on the mechanism that produced it: without it, a test for the
    /// pixel ceiling can only check the ingredients and passes with the
    /// ceiling removed.
    pub decode_errors: usize,
}

/// Decode a fuzzer byte string into a program plus the JPEG it operates on.
///
/// Wire format: one length byte, then `OP_RECORD_LEN` bytes per operation,
/// then the JPEG. Every byte string decodes — a short one simply yields a
/// shorter program — so libFuzzer never wastes an input on a parse failure.
pub fn program_from_bytes(data: &[u8]) -> (Vec<Op>, &[u8]) {
    let Some((&announced, rest)) = data.split_first() else {
        return (Vec::new(), data);
    };
    let available: usize = rest.len() / OP_RECORD_LEN;
    let count: usize = (announced as usize % (MAX_OPS + 1)).min(available);
    let (records, tail) = rest.split_at(count * OP_RECORD_LEN);
    // Re-anchor the body on SOI. Byte 0 governs both the record count and the
    // body offset, so without this a libFuzzer insertion or deletion anywhere
    // before the image shifts `FF D8` out of alignment and turns a good corpus
    // entry into a garbage body — the mutant would explore error paths only
    // and never build on the seed's coverage.
    let body: &[u8] = match tail.windows(2).position(|pair| pair == [0xFF, 0xD8]) {
        Some(soi) => &tail[soi..],
        None => tail,
    };
    // Indexed rather than `chunks_exact`: nightly clippy's
    // `chunks_exact_to_as_chunks` fires on a const chunk size, and the
    // `as_chunks` it suggests is newer than this crate's MSRV, so neither the
    // call nor an `allow` for a lint stable clippy does not know can stand.
    let ops: Vec<Op> = (0..count)
        .map(|op| op_from_record(&records[op * OP_RECORD_LEN..(op + 1) * OP_RECORD_LEN]))
        .collect();
    (ops, body)
}

/// The exact inverse of [`program_from_bytes`] for the operations a committed
/// seed needs.
///
/// `tests/generate_fuzz_seeds.rs` writes the seed corpus for
/// `fuzz_api_sequence`, and hand-assembling opcode bytes there would put a
/// second copy of this wire format in the tree — the shape that goes stale the
/// first time an opcode moves. `program_round_trips_through_its_wire_format`
/// in `tests/api_sequence_state.rs` pins the two halves together.
///
/// Returns `None` for an operation outside the encodable range (a `Set` value
/// wider than `i16`, a cropping region this format cannot express, an ICC
/// profile that is not a run of one non-zero byte).
pub fn encode_op(op: &Op) -> Option<[u8; 4]> {
    match op {
        Op::Set { param, value } => {
            let index: usize = FUZZABLE_PARAMS.iter().position(|p| p == param)?;
            let value: i16 = i16::try_from(*value).ok()?;
            let [low, high]: [u8; 2] = value.to_le_bytes();
            Some([0, u8::try_from(index).ok()?, low, high])
        }
        Op::SetScaling { index } => Some([1, *index, 0, 0]),
        Op::SetCrop { region: None } => Some([2, 0, 0, 0]),
        Op::SetCrop {
            region: Some(region),
        } => {
            if region.x != 0 || region.y >= 16 {
                return None;
            }
            let width: usize = region.width.checked_sub(1)?;
            let height: usize = region.height.checked_sub(1)?;
            if width >= 16 || height >= 16 {
                return None;
            }
            Some([
                2,
                1,
                u8::try_from(width * 16).ok()?,
                u8::try_from(height * 16 + region.y).ok()?,
            ])
        }
        Op::SetIcc { profile: None } => Some([3, 0, 0, 0]),
        Op::SetIcc {
            profile: Some(profile),
        } => {
            let &byte = profile.first()?;
            if byte == 0 || profile.len() > 128 || profile.iter().any(|&b| b != byte) {
                return None;
            }
            Some([3, byte, u8::try_from(profile.len() - 1).ok()?, 0])
        }
        Op::DecompressHeader => Some([4, 0, 0, 0]),
        Op::InspectHeader => Some([5, 0, 0, 0]),
        Op::Decompress => Some([6, 0, 0, 0]),
        Op::Decompress12 => Some([7, 0, 0, 0]),
        Op::Decompress16 => Some([8, 0, 0, 0]),
        Op::Compress {
            width,
            height,
            format,
        } => Some([9, *width, *height, *format]),
        Op::Transform { op, flags } => Some([10, *op, *flags, 0]),
        Op::SelectInput { index } => Some([11, *index, 0, 0]),
        Op::Reset => Some([12, 0, 0, 0]),
    }
}

/// Assemble a program into the byte prefix [`program_from_bytes`] reads.
///
/// Panics on an operation [`encode_op`] cannot represent, so a seed that
/// silently stopped being what it claims is a test failure rather than a
/// quietly different corpus entry.
pub fn encode_program(ops: &[Op]) -> Vec<u8> {
    assert!(
        ops.len() <= MAX_OPS,
        "a program holds at most {MAX_OPS} operations, got {}",
        ops.len()
    );
    let mut bytes: Vec<u8> = Vec::with_capacity(1 + ops.len() * OP_RECORD_LEN);
    bytes.push(ops.len() as u8);
    for op in ops {
        let record: [u8; 4] =
            encode_op(op).unwrap_or_else(|| panic!("{op:?} is outside the wire format"));
        bytes.extend_from_slice(&record);
    }
    bytes
}

fn op_from_record(record: &[u8]) -> Op {
    let (code, a, b, c): (u8, u8, u8, u8) =
        (record[0] % OPCODE_COUNT, record[1], record[2], record[3]);
    match code {
        0 => Op::Set {
            param: FUZZABLE_PARAMS[a as usize % FUZZABLE_PARAMS.len()],
            value: i32::from(i16::from_le_bytes([b, c])),
        },
        1 => Op::SetScaling { index: a },
        2 if a % 2 == 0 => Op::SetCrop { region: None },
        // `x` is pinned to 0 while P4-197 (#618) is open: a left boundary
        // at or past the *scaled* output width narrows the decode to zero
        // columns, and the vertical-crop step then trips a `debug_assert!`
        // in `pipeline_impl/output.rs` whose comment calls the empty-data
        // case unreachable. Upstream refuses the region instead
        // (`turbojpeg.c:2106-2109`). Any non-zero `x` is reachable for some
        // scale in the 1/8..2/1 range a program may also set, so the whole
        // offset is excluded rather than bounded. Delete this pin — and
        // `x: usize::from(b % 16)` returns — when that item closes.
        2 => Op::SetCrop {
            region: Some(CropRegion {
                x: 0,
                y: usize::from(c % 16),
                width: usize::from(b / 16) + 1,
                height: usize::from(c / 16) + 1,
            }),
        },
        3 if a == 0 => Op::SetIcc { profile: None },
        3 => Op::SetIcc {
            profile: Some(vec![a; usize::from(b) % 128 + 1]),
        },
        4 => Op::DecompressHeader,
        5 => Op::InspectHeader,
        6 => Op::Decompress,
        7 => Op::Decompress12,
        8 => Op::Decompress16,
        9 => Op::Compress {
            width: a,
            height: b,
            format: c,
        },
        10 => Op::Transform { op: a, flags: b },
        11 => Op::SelectInput { index: a },
        _ => Op::Reset,
    }
}

/// Run a program with the real oracle.
pub fn run_program(inputs: &[&[u8]], ops: &[Op], limits: &Limits) -> Run {
    run_program_with(inputs, ops, limits, ReferencePolicy::ConfigOnly)
}

/// Run a program, choosing how the reference handle is built.
///
/// Panics — loudly, with the program, the operation index and the difference —
/// the moment an oracle fails, so libFuzzer reports it as a crash and
/// `cargo test` as a failure.
pub fn run_program_with(
    inputs: &[&[u8]],
    ops: &[Op],
    limits: &Limits,
    policy: ReferencePolicy,
) -> Run {
    assert!(!inputs.is_empty(), "a program needs at least one input");
    // An input above the ceiling is blanked, not removed. In the fuzzer
    // `inputs[0]` is attacker-controlled, so removing it would renumber every
    // built-in behind it — `FUZZ_INPUT_LOSSLESS16` (3) would fold to `3 % 3`
    // and select the grayscale image, silently deleting the only coverage of
    // a 16-bit decode. `codex review` reproduced that by enlarging an
    // otherwise unrelated body. An empty slice is a stream every entry point
    // rejects without allocating, so the slot survives and means nothing.
    let mut usable: Vec<&[u8]> = Vec::with_capacity(inputs.len());
    let mut blanked: usize = 0;
    for input in inputs {
        if exceeds_limits(input, limits) {
            usable.push(&[]);
            blanked += 1;
        } else {
            usable.push(input);
        }
    }
    if blanked == usable.len() {
        return Run {
            executed: 0,
            skipped_oversize: true,
            decode_errors: 0,
        };
    }
    let inputs: Vec<&[u8]> = usable;

    let mut live: TjHandle = new_capped_handle(limits);
    // The prefix a reference handle replays, in the order the live handle saw
    // it: every configuration call, plus the most recent decode that published
    // the header facts a compress reads.
    let mut prefix: Vec<ReplayStep> = Vec::new();
    let mut executed: usize = 0;
    let mut selected: usize = 0;
    let mut decode_errors: usize = 0;

    for (index, op) in ops.iter().enumerate() {
        if let Op::SelectInput { index: choice } = op {
            selected = usize::from(*choice) % inputs.len();
            executed += 1;
            continue;
        }
        let jpeg: &[u8] = inputs[selected];
        let before: Vec<i32> = ALL_PARAMS.iter().map(|&param| live.get(param)).collect();
        let live_outcome: Outcome = apply(&mut live, op, jpeg, limits);
        executed += 1;

        match op {
            Op::Set { .. } | Op::SetScaling { .. } | Op::SetCrop { .. } | Op::SetIcc { .. } => {
                prefix.push(ReplayStep {
                    op: op.clone(),
                    input: selected,
                    publishes: false,
                });
            }
            Op::Reset => {
                prefix.clear();
            }
            // Handled above: it changes the argument, not the handle.
            Op::SelectInput { .. } => unreachable!("SelectInput is consumed before dispatch"),
            Op::DecompressHeader
            | Op::InspectHeader
            | Op::Decompress
            | Op::Decompress12
            | Op::Decompress16 => {
                // P1.
                let mut reference: TjHandle =
                    build_reference(&configuration_of(&prefix), &inputs, limits, policy);
                let reference_outcome: Outcome = apply(&mut reference, op, jpeg, limits);
                compare(
                    "P1 decode purity",
                    index,
                    op,
                    ops,
                    jpeg,
                    &live_outcome,
                    &reference_outcome,
                );
                if !live_outcome.succeeded() {
                    decode_errors += 1;
                }
                // P4, first half: what the operation documents it writes.
                let published: &[TjParam] = match op {
                    Op::Decompress12 | Op::Decompress16 => WRITE_BACK_PRECISION,
                    Op::InspectHeader => &[],
                    _ => WRITE_BACK_8BIT,
                };
                if live_outcome.succeeded() {
                    compare_params(index, op, ops, jpeg, &live, &reference, published);
                }
                // P4, second half: and nothing else. Asserted whether or not
                // the call succeeded — a failure is not a licence to scribble.
                compare_untouched(index, op, ops, jpeg, &live, &before, published);
                if op.publishes_for_compress() && live_outcome.succeeded() {
                    // Only the most recent publisher matters, and it takes the
                    // position it had in the live sequence — so a `set` issued
                    // after it still wins on replay, exactly as it does live.
                    prefix.retain(|step| !step.publishes);
                    prefix.push(ReplayStep {
                        op: op.clone(),
                        input: selected,
                        publishes: true,
                    });
                }
            }
            Op::Compress { .. } => {
                // P2.
                let mut reference: TjHandle = build_reference(&prefix, &inputs, limits, policy);
                let reference_outcome: Outcome = apply(&mut reference, op, jpeg, limits);
                compare(
                    "P2 compress depends only on configuration and the last published header",
                    index,
                    op,
                    ops,
                    jpeg,
                    &live_outcome,
                    &reference_outcome,
                );
                // P4, second half: `tj3Compress8` takes `&self`, so it must
                // leave every parameter where it found it.
                compare_untouched(index, op, ops, jpeg, &live, &before, &[]);
            }
            Op::Transform { .. } => {
                // P3. `transform_jpeg_with_options` takes no handle, so a
                // difference across two identical calls is global state. This
                // says nothing about handle state, by construction.
                let repeat: Outcome = apply(&mut live, op, jpeg, limits);
                compare(
                    "P3 transform determinism",
                    index,
                    op,
                    ops,
                    jpeg,
                    &live_outcome,
                    &repeat,
                );
                compare_untouched(index, op, ops, jpeg, &live, &before, &[]);
            }
        }
    }

    Run {
        executed,
        skipped_oversize: false,
        decode_errors,
    }
}

/// Whether the frame header declares more pixels than this run allows.
///
/// Mirrors `fuzz_decompress_precision`: a stream whose header does not parse
/// falls through to the entry points, which must reject it with a typed error.
fn exceeds_limits(jpeg: &[u8], limits: &Limits) -> bool {
    let Ok(decoder) = Decoder::new(jpeg) else {
        return false;
    };
    let header = decoder.header();
    let pixels: u64 = (header.width as u64).saturating_mul(header.height as u64);
    pixels > limits.max_pixels
}

/// One entry of the prefix a reference handle replays.
///
/// Replay is *in order*, and that is load-bearing rather than tidy: a decode
/// publishes `SUBSAMP`, `COLORSPACE` and the densities, which are also
/// settable, so `set(SUBSAMP) → decode` and `decode → set(SUBSAMP)` leave
/// different values behind. A first version replayed the configuration and
/// then the last publishing operation, which reordered exactly those pairs;
/// the fuzzer found it in ninety seconds, as a P2 failure on
/// `[Set(Quality), Decompress, Set(Quality), Set(Subsampling), Compress]`.
#[derive(Clone, Debug)]
struct ReplayStep {
    op: Op,
    /// Index into the program's inputs, meaningful only for a publisher.
    input: usize,
    /// True for the decode whose published header a later compress reads.
    publishes: bool,
}

/// A handle carrying the run's pixel ceiling.
///
/// The pre-filter alone is not the safeguard: `Decoder::new` refuses a stream
/// whose *scan count* exceeds its own default of 8192 while a fresh
/// `TjHandle` leaves `TJPARAM_SCANLIMIT` unset, so a header the pre-filter
/// could not read is one the handle will happily decode. `codex review`
/// reproduced exactly that — a 2048x2048 progressive frame with 8193 scans
/// that failed pre-parsing, passed `inspect_header`, and decoded past the
/// ceiling. `TJPARAM_MAXPIXELS` is deliberately absent from
/// [`FUZZABLE_PARAMS`], so no program can raise it back, and both the live and
/// the reference handle carry it — which keeps every comparison exact.
fn new_capped_handle(limits: &Limits) -> TjHandle {
    let mut handle: TjHandle = TjHandle::new();
    let ceiling: i32 = i32::try_from(limits.max_pixels).unwrap_or(i32::MAX);
    handle
        .set(TjParam::MaxPixels, ceiling)
        .expect("TJPARAM_MAXPIXELS accepts any i32");
    handle
}

fn build_reference(
    prefix: &[ReplayStep],
    inputs: &[&[u8]],
    limits: &Limits,
    policy: ReferencePolicy,
) -> TjHandle {
    let mut handle: TjHandle = new_capped_handle(limits);
    if !policy.replays_config() {
        return handle;
    }
    for step in prefix {
        if step.publishes && !policy.replays_write_back() {
            continue;
        }
        let jpeg: &[u8] = if step.publishes {
            inputs[step.input]
        } else {
            &[]
        };
        let _ = apply(&mut handle, &step.op, jpeg, limits);
    }
    handle
}

/// The prefix a *decode* is compared against: configuration only.
///
/// A decode reads no parameter any operation publishes (see P1), so replaying
/// the publisher would only add cost.
fn configuration_of(prefix: &[ReplayStep]) -> Vec<ReplayStep> {
    prefix
        .iter()
        .filter(|step| !step.publishes)
        .cloned()
        .collect()
}

/// The comparable result of one operation.
#[derive(Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A configuration call: `Ok(())` or the rejection message.
    Config(Result<(), String>),
    /// A `Reset`, which cannot fail.
    Reset,
    /// A header probe: for `InspectHeader` the returned [`FrameInfo`], for
    /// `DecompressHeader` the parameters it published, or the error.
    Header(Result<String, String>),
    /// An 8-bit decode: the frame facts, every metadata byte, every pixel.
    Pixels8(Result<DecodeSummary, String>),
    /// A 12-bit decode.
    Pixels12(Result<(String, Vec<i16>), String>),
    /// A 16-bit decode.
    Pixels16(Result<(String, Vec<u16>), String>),
    /// A compress or transform: the whole JPEG stream, or the error.
    Bytes(Result<Vec<u8>, String>),
}

/// Everything a caller can observe about one 8-bit decode.
///
/// Metadata is carried as *bytes*, not as lengths. Summarising it as
/// `icc=Some(96) exif=None ...` — the first version — makes two images whose
/// profiles differ but happen to be the same size compare equal, so a decode
/// returning the previous image's XMP would be invisible to P1 and, since it
/// is not a `TjParam`, to P4 as well. `codex review` demonstrated it with two
/// otherwise identical streams carrying the XMP packets `first` and `other`.
#[derive(Clone, PartialEq, Eq)]
pub struct DecodeSummary {
    /// Dimensions, pixel format, precision and density.
    pub facts: String,
    /// ICC, EXIF, XMP, IPTC, the comment, every saved marker and every
    /// warning, length-prefixed so the concatenation is unambiguous.
    pub metadata: Vec<u8>,
    /// The output pixels.
    pub pixels: Vec<u8>,
}

impl Outcome {
    fn succeeded(&self) -> bool {
        match self {
            Outcome::Config(result) => result.is_ok(),
            Outcome::Reset => true,
            Outcome::Header(result) => result.is_ok(),
            Outcome::Pixels8(result) => result.is_ok(),
            Outcome::Pixels12(result) => result.is_ok(),
            Outcome::Pixels16(result) => result.is_ok(),
            Outcome::Bytes(result) => result.is_ok(),
        }
    }
}

/// Prints metadata verbatim and payloads as length + digest, so a fuzz report
/// stays readable when the payload is a megapixel image.
impl core::fmt::Debug for Outcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Outcome::Config(Ok(())) => write!(f, "Config(ok)"),
            Outcome::Config(Err(message)) => write!(f, "Config(err: {message})"),
            Outcome::Reset => write!(f, "Reset"),
            Outcome::Header(Ok(meta)) => write!(f, "Header({meta})"),
            Outcome::Header(Err(message)) => write!(f, "Header(err: {message})"),
            Outcome::Pixels8(Ok(summary)) => write!(
                f,
                "Pixels8({}, metadata {} bytes {:#018x}, pixels {} bytes {:#018x})",
                summary.facts,
                summary.metadata.len(),
                digest_u8(&summary.metadata),
                summary.pixels.len(),
                digest_u8(&summary.pixels),
            ),
            Outcome::Pixels8(Err(message)) => write!(f, "Pixels8(err: {message})"),
            Outcome::Pixels12(Ok((meta, data))) => write!(
                f,
                "Pixels12({meta}, {} samples, {:#018x})",
                data.len(),
                digest_i16(data)
            ),
            Outcome::Pixels12(Err(message)) => write!(f, "Pixels12(err: {message})"),
            Outcome::Pixels16(Ok((meta, data))) => write!(
                f,
                "Pixels16({meta}, {} samples, {:#018x})",
                data.len(),
                digest_u16(data)
            ),
            Outcome::Pixels16(Err(message)) => write!(f, "Pixels16(err: {message})"),
            Outcome::Bytes(Ok(data)) => {
                write!(f, "Bytes({} bytes, {:#018x})", data.len(), digest_u8(data))
            }
            Outcome::Bytes(Err(message)) => write!(f, "Bytes(err: {message})"),
        }
    }
}

/// Execute one operation.
///
/// `limits` is threaded through only for [`Op::Reset`], which builds a new
/// handle: a `TjHandle::new()` there would drop the pixel ceiling on the live
/// handle while every reference kept it, and P1 would then fail on a header
/// the reference refuses and the live handle answers. `codex review`
/// reproduced exactly that with `[Reset, InspectHeader]`.
fn apply(handle: &mut TjHandle, op: &Op, jpeg: &[u8], limits: &Limits) -> Outcome {
    match op {
        Op::Set { param, value } => {
            Outcome::Config(handle.set(*param, *value).map_err(|e| format!("{e:?}")))
        }
        Op::SetScaling { index } => {
            let factors: Vec<(u32, u32)> = TjHandle::scaling_factors();
            let (num, denom): (u32, u32) = factors[usize::from(*index) % factors.len()];
            Outcome::Config(
                handle
                    .set_scaling_factor(num, denom)
                    .map_err(|e| format!("{e:?}")),
            )
        }
        Op::SetCrop { region } => {
            handle.set_cropping_region(*region);
            Outcome::Config(Ok(()))
        }
        Op::SetIcc { profile } => {
            handle.set_icc_profile(profile.clone());
            Outcome::Config(Ok(()))
        }
        // The published facts, not `Ok(())`: with an empty string the P1
        // comparison for this operation carried exactly one bit, and every
        // other statement about it rested on P4's hand-maintained list.
        Op::DecompressHeader => {
            let result: Result<(), String> = handle
                .decompress_header(jpeg)
                .map_err(|error| format!("{error:?}"));
            Outcome::Header(result.map(|()| describe_published(handle)))
        }
        Op::InspectHeader => Outcome::Header(
            handle
                .inspect_header(jpeg)
                .map(describe_frame)
                .map_err(|e| format!("{e:?}")),
        ),
        Op::Decompress => Outcome::Pixels8(
            handle
                .decompress(jpeg)
                .map(summarize_decode)
                .map_err(|e| format!("{e:?}")),
        ),
        Op::Decompress12 => Outcome::Pixels12(
            handle
                .decompress_12bit(jpeg)
                .map(|image| {
                    (
                        format!(
                            "{}x{} comps={}",
                            image.width, image.height, image.num_components
                        ),
                        image.data,
                    )
                })
                .map_err(|e| format!("{e:?}")),
        ),
        Op::Decompress16 => Outcome::Pixels16(
            handle
                .decompress_16bit(jpeg)
                .map(|image| {
                    (
                        format!(
                            "{}x{} comps={} prec={}",
                            image.width, image.height, image.num_components, image.precision
                        ),
                        image.data,
                    )
                })
                .map_err(|e| format!("{e:?}")),
        ),
        Op::Compress {
            width,
            height,
            format,
        } => {
            let format: PixelFormat =
                COMPRESS_FORMATS[usize::from(*format) % COMPRESS_FORMATS.len()];
            let (width, height): (usize, usize) =
                (usize::from(*width) % 16 + 1, usize::from(*height) % 16 + 1);
            let pixels: Vec<u8> = synthetic_pixels(width * height * format.bytes_per_pixel());
            Outcome::Bytes(
                handle
                    .compress(&pixels, width, height, format)
                    .map_err(|e| format!("{e:?}")),
            )
        }
        Op::Transform { op, flags } => Outcome::Bytes(
            transform_jpeg_with_options(jpeg, &transform_options(*op, *flags))
                .map_err(|e| format!("{e:?}")),
        ),
        Op::Reset => {
            *handle = new_capped_handle(limits);
            Outcome::Reset
        }
        Op::SelectInput { .. } => unreachable!("SelectInput is consumed by the driver"),
    }
}

fn summarize_decode(image: libjpeg_turbo_rs::Image) -> DecodeSummary {
    let mut metadata: Vec<u8> = Vec::new();
    push_field(&mut metadata, image.icc_profile.as_deref());
    push_field(&mut metadata, image.exif_data.as_deref());
    push_field(&mut metadata, image.xmp_data.as_deref());
    push_field(&mut metadata, image.iptc_data.as_deref());
    push_field(&mut metadata, image.comment.as_deref().map(str::as_bytes));
    for marker in &image.saved_markers {
        metadata.push(marker.code);
        push_field(&mut metadata, Some(marker.data.as_slice()));
    }
    for warning in &image.warnings {
        push_field(&mut metadata, Some(format!("{warning:?}").as_bytes()));
    }
    DecodeSummary {
        facts: format!(
            "{}x{} fmt={:?} prec={} density={:?} markers={} warnings={}",
            image.width,
            image.height,
            image.pixel_format,
            image.precision,
            image.density,
            image.saved_markers.len(),
            image.warnings.len(),
        ),
        metadata,
        pixels: image.data,
    }
}

/// Append an optional field, length-prefixed so `Some(b"ab") + None` and
/// `Some(b"a") + Some(b"b")` cannot serialise to the same bytes.
fn push_field(blob: &mut Vec<u8>, field: Option<&[u8]>) {
    match field {
        None => blob.push(0),
        Some(bytes) => {
            blob.push(1);
            blob.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            blob.extend_from_slice(bytes);
        }
    }
}

/// Decode one image the way [`Op::Decompress`] does, after applying `config`,
/// for tests that need to check the comparator itself rather than a whole
/// program.
pub fn decode_outcome(jpeg: &[u8], config: &[Op], limits: &Limits) -> Outcome {
    let mut handle: TjHandle = new_capped_handle(limits);
    for op in config {
        assert!(op.is_config(), "decode_outcome takes configuration only");
        let _ = apply(&mut handle, op, &[], limits);
    }
    apply(&mut handle, &Op::Decompress, jpeg, limits)
}

/// The parameters `tj3DecompressHeader` publishes, formatted for comparison.
fn describe_published(handle: &TjHandle) -> String {
    let mut description: String = String::new();
    for &param in WRITE_BACK_8BIT {
        description.push_str(&format!("{param:?}={} ", handle.get(param)));
    }
    description
}

fn describe_frame(frame: FrameInfo) -> String {
    format!(
        "{}x{} comps={} sub={:?}",
        frame.width, frame.height, frame.num_components, frame.subsampling
    )
}

/// Deterministic, non-uniform filler: a constant buffer would make several
/// encoder paths produce the same stream regardless of the parameters.
fn synthetic_pixels(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i.wrapping_mul(31) + 7) as u8).collect()
}

fn transform_options(op: u8, flags: u8) -> TransformOptions {
    TransformOptions {
        op: match op % 8 {
            0 => TransformOp::None,
            1 => TransformOp::HFlip,
            2 => TransformOp::VFlip,
            3 => TransformOp::Transpose,
            4 => TransformOp::Transverse,
            5 => TransformOp::Rot90,
            6 => TransformOp::Rot180,
            _ => TransformOp::Rot270,
        },
        perfect: flags & 0x01 != 0,
        trim: flags & 0x02 != 0,
        crop: None,
        grayscale: flags & 0x04 != 0,
        no_output: flags & 0x08 != 0,
        progressive: flags & 0x10 != 0,
        arithmetic: flags & 0x20 != 0,
        optimize: flags & 0x40 != 0,
        restart_interval: 0,
        restart_in_rows: false,
        copy_markers: if flags & 0x80 != 0 {
            MarkerCopyMode::None
        } else {
            MarkerCopyMode::All
        },
        ..TransformOptions::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn compare(
    property: &str,
    index: usize,
    op: &Op,
    program: &[Op],
    jpeg: &[u8],
    live: &Outcome,
    reference: &Outcome,
) {
    if live == reference {
        return;
    }
    panic!(
        "{property} violated at operation {index} ({op:?})\n\
         used handle: {live:?}\n\
         fresh handle: {reference:?}\n\
         program: {program:?}\n\
         jpeg: {} bytes, {:#018x}",
        jpeg.len(),
        digest_u8(jpeg),
    );
}

#[allow(clippy::too_many_arguments)]
fn compare_params(
    index: usize,
    op: &Op,
    program: &[Op],
    jpeg: &[u8],
    live: &TjHandle,
    reference: &TjHandle,
    published: &[TjParam],
) {
    for &param in published {
        let (used, fresh): (i32, i32) = (live.get(param), reference.get(param));
        assert!(
            used == fresh,
            "P4 write-back agreement violated at operation {index} ({op:?}): \
             {param:?} is {used} on the used handle and {fresh} on a fresh one — \
             the operation left an earlier call's value in place\n\
             program: {program:?}\n\
             jpeg: {} bytes, {:#018x}",
            jpeg.len(),
            digest_u8(jpeg),
        );
    }
}

/// P4's second half: every parameter outside the operation's documented write
/// set must hold the value it held before the call.
///
/// This is what holds `Op::InspectHeader` and `Op::Compress` — both `&self` —
/// to writing nothing at all, and what would catch a decode that writes a
/// parameter its contract does not mention. Twenty-six `get()` calls are free
/// next to a decode.
#[allow(clippy::too_many_arguments)]
fn compare_untouched(
    index: usize,
    op: &Op,
    program: &[Op],
    jpeg: &[u8],
    live: &TjHandle,
    before: &[i32],
    published: &[TjParam],
) {
    for (&param, &was) in ALL_PARAMS.iter().zip(before) {
        if published.contains(&param) {
            continue;
        }
        let now: i32 = live.get(param);
        assert!(
            now == was,
            "P4 write-back agreement violated at operation {index} ({op:?}): \
             {param:?} changed from {was} to {now}, and this operation does not \
             document writing it\n\
             program: {program:?}\n\
             jpeg: {} bytes, {:#018x}",
            jpeg.len(),
            digest_u8(jpeg),
        );
    }
}

/// FNV-1a. Payload identity is asserted by `==` on the whole buffer; this is
/// only so a failure message can name the buffers it is talking about.
fn digest_u8(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn digest_i16(samples: &[i16]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &sample in samples {
        for byte in sample.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn digest_u16(samples: &[u16]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &sample in samples {
        for byte in sample.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
