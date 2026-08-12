//! P4-146 criterion 4 (issue #518): `output_message` writes to stderr in
//! upstream's format, and `trace_level`-gated `emit_message` calls stay
//! silent by default.
//!
//! Upstream's `output_message` (`jerror.c:95-110`) formats through the
//! *installed* `format_message` and appends a newline on stderr. The message
//! text itself is already pinned verbatim against the pinned `jerror.h` by
//! `capi_classic_error_codes::the_whole_message_table_matches_upstream`, so
//! what these tests add is the routing and the shape: installed formatter,
//! stderr, trailing `\n`, and the `emit_message` display policy
//! (`jerror.c:113-143`).
//!
//! Every case runs in a child process (the pattern of
//! `capi_create_abi_guards::default_error_handler_reports_both_parameters_and_aborts`)
//! because the assertion is about the *process's* stderr — capturing it
//! in-process would test a different mechanism than the one a C consumer sees.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{jpeg_std_error, JpegErrorMgr};

/// `jerror.h` at v8: `JERR_NO_SOI` ("Not a JPEG file: starts with 0x%02x 0x%02x").
const JERR_NO_SOI: c_int = 55;
/// `jerror.h` at v8: `JMSG_VERSION` (the version string), parameterless.
const JMSG_VERSION: c_int = 76;
/// `jerror.h` at v8: `JTRC_16BIT_TABLES`.
const JTRC_16BIT_TABLES: c_int = 77;

const JMSG_LENGTH_MAX: usize = 200;

/// A minimal cinfo: the error path reads only the leading `err` pointer.
#[repr(C)]
struct MinimalCinfo {
    err: *mut JpegErrorMgr,
}

fn with_child(
    child_env: &str,
    test_name: &str,
    child: impl FnOnce(),
) -> Option<std::process::Output> {
    if std::env::var_os(child_env).is_some() {
        child();
        std::process::exit(0);
    }
    let exe: std::path::PathBuf = std::env::current_exe().expect("locate test binary");
    Some(
        std::process::Command::new(exe)
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env(child_env, "1")
            .output()
            .expect("spawn child"),
    )
}

/// Renders `msg_code` through the *installed* `format_message`, exactly as
/// upstream's `output_message` does before printing.
fn formatted(err: *mut JpegErrorMgr, cinfo: *mut c_void) -> String {
    let mut buf: [u8; JMSG_LENGTH_MAX] = [0; JMSG_LENGTH_MAX];
    // SAFETY: `err` was initialized by `jpeg_std_error`; the buffer meets the
    // JMSG_LENGTH_MAX contract.
    unsafe {
        let fmt = (*err).format_message.expect("format_message installed");
        fmt(cinfo, buf.as_mut_ptr());
    }
    let len: usize = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

/// The default `output_message` prints the installed formatter's text plus a
/// newline to stderr — `fprintf(stderr, "%s\n", buffer)` (`jerror.c:108`).
#[test]
fn default_output_message_writes_formatted_text_and_newline_to_stderr() {
    let output = with_child(
        "P4146_OUTPUT_MESSAGE_CHILD",
        "default_output_message_writes_formatted_text_and_newline_to_stderr",
        || {
            let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
            let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut err) };
            let mut cinfo: MinimalCinfo = MinimalCinfo { err: errp };
            let cinfo_ptr: *mut c_void = &mut cinfo as *mut MinimalCinfo as *mut c_void;
            // JERR_NO_SOI carries two %02x parameters — set them so the child
            // proves parameter substitution survives the stderr route.
            unsafe {
                (*errp).msg_code = JERR_NO_SOI;
                let parm: *mut u8 = std::ptr::addr_of_mut!((*errp).msg_parm) as *mut u8;
                let w: usize = std::mem::size_of::<c_int>();
                parm.copy_from_nonoverlapping(0x12_i32.to_ne_bytes().as_ptr(), w);
                parm.add(w)
                    .copy_from_nonoverlapping(0x34_i32.to_ne_bytes().as_ptr(), w);
            }
            // What upstream would print is `formatted` + "\n"; emit the expectation
            // on *stdout* so the parent never has to transcribe jerror.h.
            println!("EXPECT:{}", formatted(errp, cinfo_ptr));
            unsafe {
                let out = (*errp).output_message.expect("output_message installed");
                out(cinfo_ptr);
            }
        },
    );
    let Some(output) = output else { return };

    assert!(output.status.success(), "child failed: {:?}", output.status);
    let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
    let expected: &str = stdout
        .lines()
        .find_map(|l| l.strip_prefix("EXPECT:"))
        .expect("child printed its expectation");
    assert!(
        expected.contains("0x12") && expected.contains("0x34"),
        "precondition: the formatter substituted both parameters: {expected:?}"
    );
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(
        stderr,
        format!("{expected}\n"),
        "output_message must print exactly the installed formatter's text \
         plus a newline on stderr, as jerror.c:108 does"
    );
}

/// `output_message` must route through the *installed* `format_message`, not
/// the default — a caller that overrides only the formatter still owns the
/// text (`jerror.c:100` reads the function pointer, not the function).
#[test]
fn default_output_message_uses_an_overridden_format_message() {
    unsafe extern "C" fn custom_format(_cinfo: *mut c_void, buffer: *mut u8) {
        const MSG: &[u8] = b"CUSTOM FORMATTER TEXT\0";
        // SAFETY: the buffer is JMSG_LENGTH_MAX bytes per the contract.
        unsafe { buffer.copy_from_nonoverlapping(MSG.as_ptr(), MSG.len()) };
    }

    let output = with_child(
        "P4146_CUSTOM_FORMAT_CHILD",
        "default_output_message_uses_an_overridden_format_message",
        || {
            let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
            let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut err) };
            let mut cinfo: MinimalCinfo = MinimalCinfo { err: errp };
            let cinfo_ptr: *mut c_void = &mut cinfo as *mut MinimalCinfo as *mut c_void;
            unsafe {
                (*errp).format_message = Some(custom_format);
                let out = (*errp).output_message.expect("output_message installed");
                out(cinfo_ptr);
            }
        },
    );
    let Some(output) = output else { return };

    assert!(output.status.success(), "child failed: {:?}", output.status);
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(
        stderr, "CUSTOM FORMATTER TEXT\n",
        "output_message must render through the installed format_message"
    );
}

/// The `emit_message` display policy (`jerror.c:113-137`): traces above
/// `trace_level` are silent, level-0 advisories always show, and only the
/// first warning prints at default trace level — while `num_warnings` counts
/// every one.
#[test]
fn emit_message_trace_gating_matches_upstream_policy() {
    let output = with_child(
        "P4146_EMIT_MESSAGE_CHILD",
        "emit_message_trace_gating_matches_upstream_policy",
        || {
            let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
            let errp: *mut JpegErrorMgr = unsafe { jpeg_std_error(&mut err) };
            let mut cinfo: MinimalCinfo = MinimalCinfo { err: errp };
            let cinfo_ptr: *mut c_void = &mut cinfo as *mut MinimalCinfo as *mut c_void;
            unsafe {
                let emit = (*errp).emit_message.expect("emit_message installed");

                // Trace level 1 at default trace_level 0: silent.
                (*errp).msg_code = JTRC_16BIT_TABLES;
                emit(cinfo_ptr, 1);

                // Level 0 advisory: always shown (trace_level >= 0).
                (*errp).msg_code = JMSG_VERSION;
                println!("EXPECT:{}", formatted(errp, cinfo_ptr));
                emit(cinfo_ptr, 0);

                // Two warnings: only the first prints by default; both count.
                (*errp).msg_code = JERR_NO_SOI;
                let parm: *mut u8 = std::ptr::addr_of_mut!((*errp).msg_parm) as *mut u8;
                let w: usize = std::mem::size_of::<c_int>();
                parm.copy_from_nonoverlapping(0xAB_i32.to_ne_bytes().as_ptr(), w);
                parm.add(w)
                    .copy_from_nonoverlapping(0xCD_i32.to_ne_bytes().as_ptr(), w);
                println!("EXPECT:{}", formatted(errp, cinfo_ptr));
                emit(cinfo_ptr, -1);
                emit(cinfo_ptr, -1);
                println!(
                    "WARNINGS:{}",
                    std::ptr::addr_of!((*errp).num_warnings).read()
                );
            }
        },
    );
    let Some(output) = output else { return };

    assert!(output.status.success(), "child failed: {:?}", output.status);
    let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
    let expects: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("EXPECT:"))
        .collect();
    assert_eq!(expects.len(), 2, "child printed both expectations");
    let warnings: &str = stdout
        .lines()
        .find_map(|l| l.strip_prefix("WARNINGS:"))
        .expect("child reported num_warnings");
    assert_eq!(warnings, "2", "both warnings must be counted");
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(
        stderr,
        format!("{}\n{}\n", expects[0], expects[1]),
        "exactly the level-0 advisory and the first warning print by \
         default: traces above trace_level are silent, the second warning \
         is suppressed (jerror.c:129-142)"
    );
}
