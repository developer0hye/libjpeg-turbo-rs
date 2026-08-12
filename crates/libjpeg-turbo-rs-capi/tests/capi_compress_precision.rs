//! P4-150 (#531): a lossy 16-bit compress must be refused, the way upstream
//! refuses it.
//!
//! `tj3Compress16` used to treat 16-bit as "always lossless": whatever the
//! caller had configured, it encoded a lossless stream and returned success.
//! Upstream sets `cinfo->data_precision = 16` and lets `jpeg_start_compress`
//! decide, and `jcmaster.c:199-208` admits precision 8 or 12 for a lossy
//! compress and 2..=16 only for a lossless one — so the call fails with
//! `JERR_BAD_PRECISION` before any output exists.
//!
//! Two things made that worth fixing rather than documenting. A caller who set
//! quality 80 and 4:4:4 got neither: the output was a lossless SOF3 stream,
//! silently, in place of the documented error. And silent acceptance is the
//! failure mode with nothing to notice — the same class as P4-39 (#313), where
//! CMYK options were dropped without a word. A wrong error code at least
//! announces itself.
//!
//! The assertions here are compared against real TurboJPEG rather than
//! transcribed from it, because the rule is not visible in the layer the
//! function lives in: `turbojpeg-mp.c` contains no precision check at all.
//! Reading only that file — which is what produced the bug — suggests 16-bit
//! lossy is fine.
//!
//! The matrix also covers *where in the refusal chain* the check sits, which a
//! gate placed naively last gets wrong. Upstream installs the destination
//! before `jpeg_start_compress`, so a `TJPARAM_NOREALLOC` slot that cannot be
//! used at all loses to the buffer error — but a slot that is merely too small
//! does not, because its capacity is only tested when output overflows it, and
//! nothing is written once the compress is refused. And earlier still,
//! `setCompDefaults` validates the lossless parameters before the destination
//! exists, so an out-of-range point transform beats both. The three groups
//! disagree with each other, so no single ordering satisfies them by accident.

use std::ffi::{c_char, c_int, c_void, CStr};

mod helpers;

use libjpeg_turbo_rs_capi::{
    tj3Alloc, tj3Compress12, tj3Compress16, tj3Compress8, tj3Destroy, tj3Free, tj3GetErrorStr,
    tj3Init, tj3Set,
};

/// `turbojpeg.h`: `TJINIT_COMPRESS` — a plain enum, so 0.
const TJINIT_COMPRESS: c_int = 0;
/// `turbojpeg.h:596`: `TJPARAM_PRECISION`.
const TJPARAM_PRECISION: c_int = 7;
/// `turbojpeg.h:748`: `TJPARAM_LOSSLESS`.
const TJPARAM_LOSSLESS: c_int = 15;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJSAMP_444: c_int = 0;
const TJPF_RGB: c_int = 0;

const WIDTH: c_int = 32;
const HEIGHT: c_int = 32;

/// libjpeg's `JERR_BAD_PRECISION` text, formatted for 16-bit samples.
///
/// TurboJPEG's own errors carry a `function():` prefix (`SNPRINTF(this->errStr,
/// ..., "%s(): %s", FUNCTION_NAME, ...)`), but one raised *inside* libjpeg
/// arrives through `CATCH_LIBJPEG`, which copies the message verbatim. The
/// absence of the prefix is therefore part of the observable contract, not an
/// accident of formatting.
const BAD_PRECISION_16: &str = "Unsupported JPEG data precision 16";

/// A compressor with the parameters every case shares. `precision` of `None`
/// leaves `TJPARAM_PRECISION` alone, and `point_transform` of `None` does the
/// same for `TJPARAM_LOSSLESSPT`; both are the common configuration.
fn compressor(
    lossless: bool,
    precision: Option<c_int>,
    point_transform: Option<c_int>,
) -> *mut c_void {
    /// `turbojpeg.h`: `TJPARAM_LOSSLESSPT`.
    const TJPARAM_LOSSLESSPT: c_int = 17;

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    // SAFETY: `handle` is a live instance from `tj3Init`, used exclusively here.
    unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_QUALITY, 80), 0, "set quality");
        assert_eq!(
            tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444),
            0,
            "set subsamp"
        );
        if lossless {
            assert_eq!(tj3Set(handle, TJPARAM_LOSSLESS, 1), 0, "set lossless");
        }
        if let Some(bits) = precision {
            assert_eq!(
                tj3Set(handle, TJPARAM_PRECISION, bits),
                0,
                "set precision {bits}"
            );
        }
        if let Some(pt) = point_transform {
            assert_eq!(
                tj3Set(handle, TJPARAM_LOSSLESSPT, pt),
                0,
                "set point transform {pt}"
            );
        }
    }
    handle
}

/// The handle's current error string, as the C caller would read it.
fn error_string(handle: *mut c_void) -> String {
    // SAFETY: `handle` is live; `tj3GetErrorStr` returns a NUL-terminated
    // buffer owned by the instance and valid until the next call on it.
    let raw: *const c_char = unsafe { tj3GetErrorStr(handle) };
    assert!(!raw.is_null(), "tj3GetErrorStr returned NULL");
    // SAFETY: as above.
    unsafe { CStr::from_ptr(raw) }
        .to_str()
        .expect("error string is not UTF-8")
        .to_string()
}

/// Which rule refused the call, as opposed to how it phrased it.
///
/// The precision message is one this port owes byte for byte, and does. The
/// *precedence* between it and the destination's refusal is a separate
/// contract — which check runs first — and pinning that by raw text would drag
/// in messages the two libraries have never agreed on: TurboJPEG's own
/// `function():` errors carry per-call detail this port words differently, on
/// purpose (the same reason `norealloc_oracle.c` does not compare byte counts).
/// Classifying keeps the ordering assertion honest without pretending
/// unrelated strings match.
fn error_kind(message: &str) -> &'static str {
    if message.contains("data precision") {
        "precision"
    } else if message.contains("too small") || message.contains("NOREALLOC") {
        "buffer"
    } else if message.contains("lossless parameters") || message.contains("LOSSLESSPT") {
        // Upstream words this "Invalid progressive/lossless parameters Ss=..
        // Al=.."; this port names the two TJ parameters the caller actually
        // set. Same rule, different phrasing, so it is classified rather than
        // compared.
        "lossless_params"
    } else if message.contains("TJPARAM_QUALITY must be specified") {
        // P4-155 (#539): the substring is the documented part of upstream's
        // message; the `function():` prefix is not.
        "quality_unset"
    } else if message.contains("TJPARAM_SUBSAMP must be specified") {
        "subsamp_unset"
    } else {
        "other"
    }
}

/// One case in the oracle's line format: `label rc kind=<k> err="<message>"`.
///
/// A successful call carries no message, so both fields are empty then —
/// reading a stale `errStr` would compare state neither library promises.
/// `err` is printed only for the precision refusal, the one message the port
/// owes byte for byte; for every other outcome the kind is the contract.
fn trace_line(label: &str, handle: *mut c_void, rc: c_int) -> String {
    let message: String = if rc == 0 {
        String::new()
    } else {
        error_string(handle)
    };
    let kind: &str = if rc == 0 {
        "none"
    } else {
        error_kind(&message)
    };
    let exact: &str = if kind == "precision" { &message } else { "" };
    format!("{label} {rc} kind={kind} err=\"{exact}\"\n")
}

/// How the output slot is provided, which decides whether the destination's
/// refusal or the precision rule is the one the caller sees.
#[derive(Clone, Copy)]
enum Slot {
    /// The library allocates. The plain case.
    LibraryAllocated,
    /// `TJPARAM_NOREALLOC` set, slot left empty.
    NoReallocEmpty,
    /// `TJPARAM_NOREALLOC` set with a caller buffer of this many bytes.
    NoReallocSized(usize),
}

fn compress16_case(
    label: &str,
    lossless: bool,
    precision: Option<c_int>,
    slot: Slot,
    point_transform: Option<c_int>,
) -> String {
    /// `turbojpeg.h`: `TJPARAM_NOREALLOC`.
    const TJPARAM_NOREALLOC: c_int = 2;

    let handle: *mut c_void = compressor(lossless, precision, point_transform);
    let src: Vec<u16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 65535) as u16)
        .collect();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;

    if let Slot::NoReallocEmpty | Slot::NoReallocSized(_) = slot {
        // SAFETY: live handle.
        unsafe {
            assert_eq!(tj3Set(handle, TJPARAM_NOREALLOC, 1), 0, "set NOREALLOC");
        }
        if let Slot::NoReallocSized(capacity) = slot {
            let allocated: *mut c_void = tj3Alloc(capacity);
            assert!(!allocated.is_null(), "tj3Alloc({capacity})");
            buf = allocated as *mut u8;
            size = capacity;
        }
    }

    // SAFETY: `src` covers `WIDTH * HEIGHT * 3` samples at the default pitch and
    // outlives the call; `buf`/`size` are the output slot configured above.
    let rc: c_int = unsafe {
        tj3Compress16(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    if !buf.is_null() {
        // SAFETY: allocated by the library on the success path, freed once.
        unsafe { tj3Free(buf as *mut c_void) };
    }
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn compress12_case(label: &str, lossless: bool) -> String {
    let handle: *mut c_void = compressor(lossless, None, None);
    let src: Vec<i16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 4096) as i16)
        .collect();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;

    // SAFETY: as `compress16_case`, with 12-bit samples in `i16`.
    let rc: c_int = unsafe {
        tj3Compress12(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    if !buf.is_null() {
        // SAFETY: allocated by the library, freed once.
        unsafe { tj3Free(buf as *mut c_void) };
    }
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn compress8_case(label: &str, lossless: bool) -> String {
    let handle: *mut c_void = compressor(lossless, None, None);
    let src: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;

    // SAFETY: as above, with 8-bit samples.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    if !buf.is_null() {
        // SAFETY: allocated by the library, freed once.
        unsafe { tj3Free(buf as *mut c_void) };
    }
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

/// Every case the oracle traces, in its order, as one string.
fn our_trace() -> String {
    let mut trace: String = String::new();
    trace.push_str(&compress16_case(
        "c16_lossy",
        false,
        None,
        Slot::LibraryAllocated,
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossless",
        true,
        None,
        Slot::LibraryAllocated,
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossless_prec13",
        true,
        Some(13),
        Slot::LibraryAllocated,
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossless_prec12",
        true,
        Some(12),
        Slot::LibraryAllocated,
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossy_prec12",
        false,
        Some(12),
        Slot::LibraryAllocated,
        None,
    ));
    // Precedence against the destination, which upstream installs before
    // `jpeg_start_compress`. A slot that cannot be used at all loses to the
    // buffer error; a slot that is merely too small does not, because its
    // capacity is never tested once the compress is refused.
    trace.push_str(&compress16_case(
        "c16_lossy_norealloc_null",
        false,
        None,
        Slot::NoReallocEmpty,
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossy_norealloc_cramped",
        false,
        None,
        Slot::NoReallocSized(16),
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossless_norealloc_null",
        true,
        None,
        Slot::NoReallocEmpty,
        None,
    ));
    trace.push_str(&compress16_case(
        "c16_lossless_norealloc_roomy",
        true,
        None,
        Slot::NoReallocSized(64 * 1024),
        None,
    ));
    // Precedence against the *lossless parameters*, which upstream validates
    // earlier still: `setCompDefaults` calls `jpeg_enable_lossless` before
    // `jpeg_mem_dest_tj` (`turbojpeg-mp.c:117-120`). A point transform that is
    // not less than the precision therefore wins over the buffer error even
    // though the slot is unusable — so the destination preflight cannot simply
    // be hoisted to the front of the function.
    trace.push_str(&compress16_case(
        "c16_pt_ge_prec_norealloc_null",
        true,
        Some(13),
        Slot::NoReallocEmpty,
        Some(13),
    ));
    trace.push_str(&compress16_case(
        "c16_pt_ge_prec_roomy",
        true,
        Some(13),
        Slot::LibraryAllocated,
        Some(13),
    ));
    // A legal point transform, so the buffer error returns once the lossless
    // parameters stop being the first thing wrong.
    trace.push_str(&compress16_case(
        "c16_pt_lt_prec_norealloc_null",
        true,
        Some(13),
        Slot::NoReallocEmpty,
        Some(12),
    ));
    trace.push_str(&compress12_case("c12_lossy", false));
    trace.push_str(&compress12_case("c12_lossless", true));
    trace.push_str(&compress8_case("c8_lossy", false));
    trace.push_str(&compress8_case("c8_lossless", true));

    // P4-155 (#539): the unset-parameter matrix.
    trace.push_str(&fresh_get_case("p4155_fresh_get"));
    trace.push_str(&unset8_case("p4155_c8_unset_both", false, false, false));
    trace.push_str(&unset8_case("p4155_c8_unset_quality", false, true, false));
    trace.push_str(&unset8_case("p4155_c8_unset_subsamp", true, false, false));
    // Argument validation still wins over the musts (step 1 vs step 2).
    trace.push_str(&unset8_case("p4155_c8_arg_precedence", false, false, true));
    trace.push_str(&unset12_case("p4155_c12_unset_both"));
    trace.push_str(&unset16_lossless_case("p4155_c16_lossless_unset"));
    trace.push_str(&unset_encodeyuv8_case("p4155_encodeyuv8_unset"));
    trace.push_str(&unset_fromyuvplanes8_case("p4155_fromyuvplanes8_unset"));
    trace.push_str(&unset_decodeyuvplanes8_case("p4155_decodeyuvplanes8_unset"));
    // #548-review round: the packed wrappers, discriminating lines — an
    // out-of-range pixelFormat proves the subsampling gate precedes the
    // format check (the check lives in the ...Planes8 delegates upstream),
    // and a non-power-of-two align proves argument validation beats the
    // gates.
    trace.push_str(&unset_fromyuv8_case("p4155_fromyuv8_unset", 1));
    trace.push_str(&unset_fromyuv8_case("p4155_fromyuv8_badalign", 3));
    trace.push_str(&unset_encodeyuv8_badpf_case("p4155_encodeyuv8_badpf_unset"));
    trace.push_str(&unset_decodeyuv8_badpf_case("p4155_decodeyuv8_badpf_unset"));
    trace.push_str(&unset_encodeyuvplanes8_case("p4155_encodeyuvplanes8_unset"));
    trace.push_str(&legacy_encodeyuv3_align0_case(
        "p4155_legacy_encodeyuv3_align0",
    ));
    trace.push_str(&legacy_decodeyuv_align0_case(
        "p4155_legacy_decodeyuv_align0",
    ));
    trace
}

/// The divergence itself, stated without reference to the oracle so the gate
/// still has teeth on a machine with no TurboJPEG development install.
#[test]
fn lossy_16bit_compress_is_refused() {
    let line: String = compress16_case("c16_lossy", false, None, Slot::LibraryAllocated, None);
    assert_eq!(
        line,
        format!("c16_lossy -1 kind=precision err=\"{BAD_PRECISION_16}\"\n"),
        "a lossy 16-bit compress must fail with libjpeg's JERR_BAD_PRECISION \
         text; encoding a lossless stream instead gives the caller output where \
         upstream gives a documented error"
    );
}

/// `TJPARAM_PRECISION` is read only when `TJPARAM_LOSSLESS` is set
/// (`turbojpeg-mp.c:111-115`), so asking for 12 does not turn a lossy 16-bit
/// call into a legal one. This is the case a fix keyed off the requested
/// precision rather than the lossless flag would get wrong.
#[test]
fn requesting_12bit_precision_does_not_rescue_a_lossy_16bit_compress() {
    let line: String = compress16_case(
        "c16_lossy_prec12",
        false,
        Some(12),
        Slot::LibraryAllocated,
        None,
    );
    assert_eq!(
        line,
        format!("c16_lossy_prec12 -1 kind=precision err=\"{BAD_PRECISION_16}\"\n"),
        "TJPARAM_PRECISION is ignored unless TJPARAM_LOSSLESS is set, so the \
         effective precision here is still 16 and the call is still refused"
    );
}

/// The rule must not be over-applied. Twelve-bit is one of the two precisions
/// `jcmaster.c:206` admits for a lossy compress, so refusing it would trade
/// one divergence for another — and a fix written as "wide samples imply
/// lossless" would do exactly that.
#[test]
fn lossy_12bit_and_8bit_compress_still_succeed() {
    assert_eq!(
        compress12_case("c12_lossy", false),
        "c12_lossy 0 kind=none err=\"\"\n",
        "12-bit lossy is legal upstream"
    );
    assert_eq!(
        compress8_case("c8_lossy", false),
        "c8_lossy 0 kind=none err=\"\"\n",
        "8-bit lossy is the ordinary path"
    );
}

/// Lossless 16-bit — the configuration the precision exists for — must keep
/// working, including with `TJPARAM_PRECISION` inside and outside the window
/// upstream honours.
#[test]
fn lossless_16bit_compress_still_succeeds() {
    for (label, precision) in [
        ("c16_lossless", None),
        ("c16_lossless_prec13", Some(13)),
        // Outside the 13..=16 window `turbojpeg-mp.c:114` honours: ignored,
        // not an error.
        ("c16_lossless_prec12", Some(12)),
    ] {
        assert_eq!(
            compress16_case(label, true, precision, Slot::LibraryAllocated, None),
            format!("{label} 0 kind=none err=\"\"\n"),
            "lossless 16-bit must still encode"
        );
    }
}

/// The whole matrix against real TurboJPEG.
///
/// The per-case tests above pin what this port does; this pins that it is the
/// same thing upstream does. The distinction is not academic here — the bug
/// being fixed was introduced by reading `turbojpeg-mp.c`, which contains no
/// precision check, and concluding correctly that TurboJPEG imposes none. The
/// check is in libjpeg, two calls down.
#[test]
fn precision_rules_match_upstream_turbojpeg() {
    let Some(oracle) = helpers::build_oracle("compress_precision_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             P4-150's precision rules cannot be built. Set LIBJPEG_TURBO_PREFIX \
             to make this a hard failure."
        );
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);

    assert_eq!(
        our_trace(),
        c_trace,
        "compress precision acceptance diverges from upstream TurboJPEG"
    );
}

// --- P4-155 (#539): the "must be specified" gates over unset parameters. ---
//
// A fresh handle carries `TJPARAM_QUALITY = -1` and `TJPARAM_SUBSAMP =
// TJSAMP_UNKNOWN`, and every *lossy* compress path refuses until the caller
// supplies them (`turbojpeg-mp.c:95-98`); the YUV encode/decode paths gate on
// SUBSAMP alone. This port defaulted to 75 / 4:2:0, so the branch was
// unreachable and a caller who forgot a parameter silently got substitutes.

/// A handle with only the named parameters set.
fn partial_compressor(set_quality: bool, set_subsamp: bool, lossless: bool) -> *mut c_void {
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    // SAFETY: `handle` is a live instance from `tj3Init`, used exclusively here.
    unsafe {
        if set_quality {
            assert_eq!(tj3Set(handle, TJPARAM_QUALITY, 80), 0, "set quality");
        }
        if set_subsamp {
            assert_eq!(
                tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444),
                0,
                "set subsamp"
            );
        }
        if lossless {
            assert_eq!(tj3Set(handle, TJPARAM_LOSSLESS, 1), 0, "set lossless");
        }
    }
    handle
}

/// Criterion 1: the fresh handle's own report of both parameters.
fn fresh_get_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3Get;
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    // SAFETY: live handle.
    let line: String = unsafe {
        format!(
            "{label} 0 kind=get err=\"quality={} subsamp={}\"\n",
            tj3Get(handle, TJPARAM_QUALITY),
            tj3Get(handle, TJPARAM_SUBSAMP)
        )
    };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn unset8_case(label: &str, set_quality: bool, set_subsamp: bool, null_src: bool) -> String {
    let handle: *mut c_void = partial_compressor(set_quality, set_subsamp, false);
    let src: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: `src` outlives the call; out-params are locals.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            if null_src {
                std::ptr::null()
            } else {
                src.as_ptr()
            },
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: library-allocated or null.
    unsafe {
        tj3Free(buf as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

fn unset12_case(label: &str) -> String {
    let handle: *mut c_void = partial_compressor(false, false, false);
    let src: Vec<i16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 4096) as i16)
        .collect();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: as `unset8_case`.
    let rc: c_int = unsafe {
        tj3Compress12(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: as `unset8_case`.
    unsafe {
        tj3Free(buf as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

/// A lossless compress consults neither parameter (`turbojpeg-mp.c:95-98`
/// gates on `!this->lossless`), so both may stay unset.
fn unset16_lossless_case(label: &str) -> String {
    let handle: *mut c_void = partial_compressor(false, false, true);
    let src: Vec<u16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 65535) as u16)
        .collect();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: as `unset8_case`.
    let rc: c_int = unsafe {
        tj3Compress16(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: as `unset8_case`.
    unsafe {
        tj3Free(buf as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

fn unset_encodeyuv8_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3EncodeYUV8;
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    let src: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();
    // Generous: large enough for any subsampling had the call proceeded.
    let mut dst: Vec<u8> = vec![0; WIDTH as usize * HEIGHT as usize * 4];
    // SAFETY: buffers outlive the call.
    let rc: c_int = unsafe {
        tj3EncodeYUV8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            dst.as_mut_ptr(),
            1,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn unset_fromyuvplanes8_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3CompressFromYUVPlanes8;
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    let plane: usize = WIDTH as usize * HEIGHT as usize;
    let y: Vec<u8> = (0..plane).map(|i| (i % 251) as u8).collect();
    let cb: Vec<u8> = vec![128; plane];
    let cr: Vec<u8> = vec![128; plane];
    let planes: [*const u8; 3] = [y.as_ptr(), cb.as_ptr(), cr.as_ptr()];
    let strides: [c_int; 3] = [WIDTH, WIDTH, WIDTH];
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: three live planes with matching strides.
    let rc: c_int = unsafe {
        tj3CompressFromYUVPlanes8(
            handle,
            planes.as_ptr(),
            WIDTH,
            strides.as_ptr(),
            HEIGHT,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: as `unset8_case`.
    unsafe {
        tj3Free(buf as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

fn unset_decodeyuvplanes8_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3DecodeYUVPlanes8;
    /// `turbojpeg.h`: `TJINIT_DECOMPRESS`.
    const TJINIT_DECOMPRESS: c_int = 1;
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_DECOMPRESS)");
    let plane: usize = WIDTH as usize * HEIGHT as usize;
    let y: Vec<u8> = (0..plane).map(|i| (i % 251) as u8).collect();
    let cb: Vec<u8> = vec![128; plane];
    let cr: Vec<u8> = vec![128; plane];
    let planes: [*const u8; 3] = [y.as_ptr(), cb.as_ptr(), cr.as_ptr()];
    let strides: [c_int; 3] = [WIDTH, WIDTH, WIDTH];
    let mut dst: Vec<u8> = vec![0; plane * 4];
    // SAFETY: three live planes; `dst` sized for RGBX at full resolution.
    let rc: c_int = unsafe {
        tj3DecodeYUVPlanes8(
            handle,
            planes.as_ptr(),
            strides.as_ptr(),
            dst.as_mut_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn unset_fromyuv8_case(label: &str, align: c_int) -> String {
    use libjpeg_turbo_rs_capi::tj3CompressFromYUV8;
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    let src: Vec<u8> = vec![128; WIDTH as usize * HEIGHT as usize * 4];
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: `src` outlives the call; out-params are locals.
    let rc: c_int = unsafe {
        tj3CompressFromYUV8(
            handle,
            src.as_ptr(),
            WIDTH,
            align,
            HEIGHT,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: library-allocated or null; destroyed once.
    unsafe {
        tj3Free(buf as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

fn unset_encodeyuv8_badpf_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3EncodeYUV8;
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    let src: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();
    let mut dst: Vec<u8> = vec![0; WIDTH as usize * HEIGHT as usize * 4];
    // SAFETY: buffers outlive the call; 99 is deliberately out of TJPF range.
    let rc: c_int = unsafe {
        tj3EncodeYUV8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            99,
            dst.as_mut_ptr(),
            1,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn unset_decodeyuv8_badpf_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3DecodeYUV8;
    /// `turbojpeg.h`: `TJINIT_DECOMPRESS`.
    const TJINIT_DECOMPRESS: c_int = 1;
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_DECOMPRESS)");
    let src: Vec<u8> = vec![128; WIDTH as usize * HEIGHT as usize * 4];
    let mut dst: Vec<u8> = vec![0; WIDTH as usize * HEIGHT as usize * 4];
    // SAFETY: buffers outlive the call; 99 is deliberately out of TJPF range.
    let rc: c_int = unsafe {
        tj3DecodeYUV8(
            handle,
            src.as_ptr(),
            1,
            dst.as_mut_ptr(),
            WIDTH,
            0,
            HEIGHT,
            99,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn unset_encodeyuvplanes8_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tj3EncodeYUVPlanes8;
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    let src: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();
    let plane: usize = WIDTH as usize * HEIGHT as usize;
    let mut y: Vec<u8> = vec![0; plane];
    let mut cb: Vec<u8> = vec![0; plane];
    let mut cr: Vec<u8> = vec![0; plane];
    let mut planes: [*mut u8; 3] = [y.as_mut_ptr(), cb.as_mut_ptr(), cr.as_mut_ptr()];
    let strides: [c_int; 3] = [WIDTH, WIDTH, WIDTH];
    // SAFETY: three live planes with matching strides; `src` outlives.
    let rc: c_int = unsafe {
        tj3EncodeYUVPlanes8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            planes.as_mut_ptr(),
            strides.as_ptr(),
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

/// P4-155 (#539): the C-ABI gates fire without a TurboJPEG install.
///
/// The oracle trace above cross-validates against real TurboJPEG where one
/// exists; this standalone test keeps the gates covered on hosts (and CI
/// legs) without it — the first version of this block had every new case
/// reachable only through the skippable trace.
#[test]
fn unset_params_refuse_standalone() {
    let both = unset8_case("standalone_unset_both", false, false, false);
    assert!(
        both.contains("kind=quality_unset"),
        "quality gate first: {both}"
    );
    let no_subsamp = unset8_case("standalone_unset_subsamp", true, false, false);
    assert!(
        no_subsamp.contains("kind=subsamp_unset"),
        "subsamp gate second: {no_subsamp}"
    );
    let lossless = unset16_lossless_case("standalone_lossless_bypass");
    assert!(
        lossless.contains(" 0 kind=none"),
        "lossless consults neither: {lossless}"
    );
    let packed = unset_fromyuv8_case("standalone_fromyuv8", 1);
    assert!(
        packed.contains("kind=subsamp_unset"),
        "packed FromYUV8 gates subsampling in the entry itself: {packed}"
    );
}

/// The legacy wrappers forward `align`/`pad` raw; a zero value must reach the
/// TJ3 entry's validation rather than being clamped to 1 — upstream refuses
/// (#539 re-review). Both parameters are set, isolating the align path.
fn legacy_encodeyuv3_align0_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::{tjEncodeYUV3, tjInitCompress};
    const TJSAMP_420: c_int = 2;
    let handle: *mut c_void = tjInitCompress();
    assert!(!handle.is_null(), "tjInitCompress");
    let src: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();
    let mut dst: Vec<u8> = vec![0; WIDTH as usize * HEIGHT as usize * 4];
    // SAFETY: buffers outlive the call; align 0 is the case under test.
    let rc: c_int = unsafe {
        tjEncodeYUV3(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            dst.as_mut_ptr(),
            0,
            TJSAMP_420,
            0,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}

fn legacy_decodeyuv_align0_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::{tjDecodeYUV, tjInitDecompress};
    const TJSAMP_420: c_int = 2;
    let handle: *mut c_void = tjInitDecompress();
    assert!(!handle.is_null(), "tjInitDecompress");
    let src: Vec<u8> = vec![128; WIDTH as usize * HEIGHT as usize * 4];
    let mut dst: Vec<u8> = vec![0; WIDTH as usize * HEIGHT as usize * 4];
    // SAFETY: buffers outlive the call; align 0 is the case under test.
    let rc: c_int = unsafe {
        tjDecodeYUV(
            handle,
            src.as_ptr(),
            0,
            TJSAMP_420,
            dst.as_mut_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            0,
        )
    };
    let line: String = trace_line(label, handle, rc);
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    line
}
