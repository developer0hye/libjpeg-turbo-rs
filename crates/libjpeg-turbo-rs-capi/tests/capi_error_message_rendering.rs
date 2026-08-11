//! P4-146 (#518): `format_message` must render upstream's text, not a fallback.
//!
//! `jpeg_std_error` left `jpeg_message_table` null, so `default_format_message`
//! always took its `"libjpeg-turbo-rs: bogus message code"` branch. Every
//! classic error rendered as that string for a C consumer using
//! `err->format_message` or `err->output_message` — the standard error path —
//! whatever `msg_code` said.
//!
//! **Why the existing gate did not catch it.** `capi_classic_error_codes.rs`
//! checks each code's number *and* message against the pinned upstream
//! headers, and has caught real mistakes. But it reads the expected message
//! from `jerror.h` and compares it to a table in the test — never to our
//! formatter. It is a genuine parity check of the *constants* and a false-green
//! on the *rendering*. These tests close that by going through the shim.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{jpeg_std_error, JpegErrorMgr};

/// The C contract: `format_message` writes at most `JMSG_LENGTH_MAX` bytes.
const JMSG_LENGTH_MAX: usize = 200;

/// A `j_common_ptr`-shaped stub with the alignment the cast requires.
///
/// A `Vec<u8>` was used here first, which is byte-aligned: casting its data
/// pointer to `*mut *mut JpegErrorMgr` and writing through it is undefined
/// behaviour even when the system allocator happens to over-align, and Miri
/// rejects it. `#[repr(C)]` with `err` first gives the real layout the
/// callbacks read.
#[repr(C)]
struct CommonStub {
    err: *mut JpegErrorMgr,
    /// libjpeg's `j_common_ptr` continues past `err`; nothing under test
    /// reads it, but the padding keeps a stray offset read inside our own
    /// allocation rather than off the end of a one-pointer struct.
    tail: [usize; 127],
}

impl CommonStub {
    fn new(err: *mut JpegErrorMgr) -> Self {
        Self {
            err,
            tail: [0usize; 127],
        }
    }
}

/// Render `code` (with optional integer parameter) exactly as a C consumer
/// would: set `msg_code`/`msg_parm`, then call `err->format_message`.
fn render(code: c_int, parm: Option<c_int>) -> String {
    let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
    // SAFETY: `jerr` is a live, correctly-aligned error manager owned here.
    let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut jerr as *mut JpegErrorMgr) };
    assert!(!errp.is_null(), "jpeg_std_error");

    // A `j_common_ptr`-shaped stub: its first field is the `err` pointer, which
    // is all `format_message` reads.
    let mut cinfo: CommonStub = CommonStub::new(errp);
    let mut buf: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];

    // SAFETY: `cinfo` is a 1 KiB allocation whose first pointer-sized field we
    // set to `errp`; `buf` is exactly the `JMSG_LENGTH_MAX` bytes the contract
    // requires. Both outlive the call.
    unsafe {
        (*errp).msg_code = code;
        if let Some(value) = parm {
            // `msg_parm` is a union whose `i` arm is `c_int[8]`; writing the
            // first slot is what `ERREXIT1` does.
            let parm_ptr: *mut c_int = (*errp).msg_parm.as_mut_ptr() as *mut c_int;
            parm_ptr.write(value);
        }
        let format = (*errp).format_message.expect("format_message installed");
        format(
            &mut cinfo as *mut CommonStub as *mut c_void,
            buf.as_mut_ptr(),
        );
    }

    let len: usize = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

/// A parameterless code renders its exact upstream string.
///
/// `JERR_NO_BACKING_STORE` is the one P4-14's budget guard raises, so this is
/// also the end-to-end proof that a C consumer hitting that guard now sees
/// something actionable.
#[test]
fn parameterless_code_renders_upstream_text() {
    assert_eq!(render(51, None), "Memory limit exceeded");
    assert_eq!(render(72, None), "Image too wide for this implementation");
    assert_eq!(render(44, None), "Premature end of input file");
}

/// `%d` substitution from `msg_parm.i[0]` — upstream's `ERREXIT1` shape.
#[test]
fn integer_parameter_is_substituted() {
    // "Insufficient memory (case %d)" with upstream's alloc-too-large case.
    assert_eq!(render(56, Some(8)), "Insufficient memory (case 8)");
}

/// `%u` is a distinct format spec and must not be dropped or mis-rendered.
#[test]
fn unsigned_parameter_is_substituted() {
    assert_eq!(
        render(42, Some(65500)),
        "Maximum supported image dimension is 65500 pixels"
    );
}

/// An out-of-range code falls back to entry 0 — *and reports which code it
/// was*, which is the whole point of the message.
///
/// Upstream's fallback is `msg_parm.i[0] = msg_code; msgtext = table[0]`
/// (`jerror.c:173-175`), and entry 0 is "Bogus message code %d". We used to
/// substitute a fixed string that dropped the number.
#[test]
fn out_of_range_code_falls_back_and_names_the_code() {
    assert_eq!(render(100_000, None), "Bogus message code 100000");
    // Negative codes take the same path; the table index must never be signed.
    assert_eq!(render(-1, None), "Bogus message code -1");
    // Code 0 is itself entry 0, and upstream renders it the same way.
    assert_eq!(render(0, None), "Bogus message code 0");
}

/// The boundary: the last valid index renders, one past it does not.
#[test]
fn last_message_index_is_the_boundary() {
    let last: c_int = {
        let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
        // SAFETY: live error manager owned here.
        let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut jerr as *mut JpegErrorMgr) };
        unsafe { (*errp).last_jpeg_message }
    };
    assert_eq!(
        last, 128,
        "v8 table has 129 entries, so the last index is 128"
    );
    assert!(
        !render(last, None).starts_with("Bogus message code"),
        "the last table entry must render its own text"
    );
    assert_eq!(
        render(last + 1, None),
        format!("Bogus message code {}", last + 1),
        "one past the last entry must fall back"
    );
}

/// The end-to-end case criterion 3 actually asks for: trigger a **real
/// library failure** and format the message from the same error manager the
/// failure used.
///
/// Every other test here writes `msg_code` by hand, which proves the formatter
/// and the table agree but says nothing about whether a failure populates
/// them. A broken error-to-formatter integration would pass all of them.
///
/// The failure used is P4-14's memory-budget guard: it is reachable through
/// the public vtable, deterministic, and its message carries no parameter, so
/// the assertion is the exact upstream string.
#[test]
fn a_real_failure_renders_its_message() {
    use libjpeg_turbo_rs_capi::jpeglib::{
        jpeg_CreateDecompress, jpeg_destroy_decompress, JpegDecompressPublic,
    };
    use libjpeg_turbo_rs_capi::memmgr::JpegMemoryMgr;

    const JPOOL_IMAGE: c_int = 1;

    let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: JpegDecompressPublic = unsafe { std::mem::zeroed() };
    // SAFETY: both are live, correctly-aligned structs owned by this test, and
    // the declared size matches the type passed.
    unsafe {
        cinfo.err = jpeg_std_error(&mut jerr as *mut JpegErrorMgr);
        let cinfo_ptr: *mut c_void = &mut cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>());

        // `error_exit` must return rather than `longjmp`; panicking would
        // cross an `extern "C"` frame and abort.
        unsafe extern "C" fn ignore_error_exit(_cinfo: *mut c_void) {}
        (*cinfo.err).error_exit = Some(ignore_error_exit);
        (*cinfo.err).msg_code = -1;

        // A 1 MiB budget against a 16 MiB virtual array (P4-14).
        let mgr: &mut JpegMemoryMgr = &mut *(cinfo.mem as *mut JpegMemoryMgr);
        mgr.max_memory_to_use = 1 << 20;
        let request = mgr.request_virt_sarray.expect("request_virt_sarray");
        let realize = mgr.realize_virt_arrays.expect("realize_virt_arrays");
        let ctrl = request(cinfo_ptr, JPOOL_IMAGE, 0, 4096, 4096, 4096);
        assert!(!ctrl.is_null(), "request_virt_sarray");
        realize(cinfo_ptr);

        // Now format from that same manager, exactly as a C error handler does.
        let mut buf: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];
        let mut stub: CommonStub = CommonStub::new(cinfo.err);
        let format = (*cinfo.err).format_message.expect("format_message");
        format(
            &mut stub as *mut CommonStub as *mut c_void,
            buf.as_mut_ptr(),
        );

        let len: usize = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let rendered: String = String::from_utf8_lossy(&buf[..len]).into_owned();
        assert_eq!(
            rendered, "Memory limit exceeded",
            "a real budget failure must render upstream's text, not a fallback"
        );

        jpeg_destroy_decompress(cinfo_ptr);
    }
}
