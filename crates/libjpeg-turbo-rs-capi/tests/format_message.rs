//! P2-2 regression: `default_format_message` must implement the printf
//! expansion documented in `references/libjpeg-turbo/src/jerror.c::format_message`.
//!
//! For each specifier seen in `jerror.h` (`%s %d %u %x %X %c %02d %3d
//! %4u %02x %04x %%`), build a synthetic addon table containing one
//! format string, set `msg_parm`, call the shim's `format_message`
//! through the standard `jpeg_error_mgr` vtable, and assert the
//! formatted output equals what libc snprintf would produce — using
//! the *same* format string and the *same* arguments. The test is C-free
//! at the libjpeg level (does not invoke `cjpeg`/`djpeg`); it only uses
//! `libc::snprintf` as the reference oracle.
//!
//! TDD-verified: prior to the fix, `default_format_message` copied the
//! format string verbatim. With the fix, every assertion below passes;
//! reverting the fix makes the `%d` / `%u` / `%x` / `%c` / `%02d` /
//! `%3d` cases red-fail because the verbatim copy still contains the
//! `%X` literal.

use std::ffi::{c_char, c_int};
use std::os::raw::c_void;

use libloading::Library;

#[path = "support/cdylib.rs"]
mod cdylib_support;

const JMSG_LENGTH_MAX: usize = 200;
const JMSG_STR_PARM_MAX: usize = 80;

#[repr(C)]
struct JpegErrorMgrLayout {
    error_exit: Option<unsafe extern "C" fn(*mut c_void)>,
    emit_message: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
    output_message: Option<unsafe extern "C" fn(*mut c_void)>,
    format_message: Option<unsafe extern "C" fn(*mut c_void, *mut u8)>,
    reset_error_mgr: Option<unsafe extern "C" fn(*mut c_void)>,
    msg_code: c_int,
    msg_parm: [u8; JMSG_STR_PARM_MAX],
    trace_level: c_int,
    num_warnings: std::ffi::c_long,
    jpeg_message_table: *const *const u8,
    last_jpeg_message: c_int,
    addon_message_table: *const *const u8,
    first_addon_message: c_int,
    last_addon_message: c_int,
}

// The MSVC UCRT declares `snprintf` as an inline in <stdio.h> and does
// not export it from the import libraries, so linking the raw symbol
// fails with LNK2019 and — because that is a build failure — no
// workspace test could even build on a Windows host (issue #378,
// P4-62). `legacy_stdio_definitions.lib` ships out-of-line exported
// definitions of the inline printf family exactly for external-symbol
// consumers like this oracle; linking it preserves the P2-2
// printf-expansion coverage on MSVC instead of cfg-ing the tests away.
#[cfg_attr(target_env = "msvc", link(name = "legacy_stdio_definitions"))]
extern "C" {
    fn snprintf(buf: *mut c_char, size: usize, fmt: *const c_char, ...) -> c_int;
}

fn cdylib_path() -> std::path::PathBuf {
    cdylib_support::cdylib_path()
}

/// Set up a JpegErrorMgr with `format_message` patched in via
/// `jpeg_std_error`, then invoke it for the given format / args and
/// return the resulting formatted string (without trailing NUL).
fn invoke_shim_format_message(
    lib: &Library,
    format_str: &[u8],
    string_arg: Option<&[u8]>,
    int_args: &[c_int],
) -> Vec<u8> {
    // Build a single-entry addon table. The shim indexes by
    // `msg_code - first_addon_message`, so if first==msg_code==1000 the
    // shim reads slot 0 — that's where we put the format string.
    let format_with_nul: Vec<u8> = {
        let mut v = format_str.to_vec();
        v.push(0);
        v
    };
    let entries: Vec<*const u8> = vec![format_with_nul.as_ptr()];

    let std_error: libloading::Symbol<
        unsafe extern "C" fn(*mut JpegErrorMgrLayout) -> *mut JpegErrorMgrLayout,
    > = unsafe { lib.get(b"jpeg_std_error").unwrap() };

    let mut err: JpegErrorMgrLayout = unsafe { std::mem::zeroed() };
    unsafe { std_error(&mut err as *mut _) };

    // Wire the addon table to cover code 1000.
    err.first_addon_message = 1000;
    err.last_addon_message = 1000;
    err.addon_message_table = entries.as_ptr();
    err.msg_code = 1000;

    // Fill msg_parm. For string mode, copy the string into the union's
    // `s` arm (truncate-with-NUL at JMSG_STR_PARM_MAX-1 like upstream).
    // For int mode, pack each c_int into its 4-byte slot.
    err.msg_parm = [0u8; JMSG_STR_PARM_MAX];
    if let Some(s) = string_arg {
        let n: usize = s.len().min(JMSG_STR_PARM_MAX - 1);
        err.msg_parm[..n].copy_from_slice(&s[..n]);
        err.msg_parm[n] = 0;
    } else {
        for (i, &v) in int_args.iter().enumerate().take(8) {
            let off = i * std::mem::size_of::<c_int>();
            let bytes = v.to_ne_bytes();
            err.msg_parm[off..off + bytes.len()].copy_from_slice(&bytes);
        }
    }

    // The shim's default_format_message reads `cinfo` as `*mut *mut JpegErrorMgr`
    // (i.e. err is at offset 0 of the cinfo struct). Synthesize a tiny
    // pseudo-cinfo that satisfies that contract.
    #[repr(C)]
    struct PseudoCinfo {
        // Read by the shim's `default_format_message` via raw-pointer
        // indirection at offset 0 — invisible to Rust's reachability
        // analysis but load-bearing at runtime.
        #[allow(dead_code)]
        err: *mut JpegErrorMgrLayout,
    }
    let mut pseudo = PseudoCinfo {
        err: &mut err as *mut _,
    };

    let mut buf: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];
    unsafe {
        let f = err.format_message.expect("format_message slot installed");
        f(&mut pseudo as *mut _ as *mut c_void, buf.as_mut_ptr());
    }
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(JMSG_LENGTH_MAX);
    buf[..nul].to_vec()
}

/// Reference: format `format_str` with `args` via libc snprintf and
/// return the bytes (no NUL).
fn snprintf_ref_string(format_str: &[u8], string_arg: &[u8]) -> Vec<u8> {
    // Make NUL-terminated cstrings.
    let mut fmt: Vec<u8> = format_str.to_vec();
    fmt.push(0);
    let mut s_in: Vec<u8> = string_arg.to_vec();
    s_in.push(0);
    let mut buf: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];
    unsafe {
        snprintf(
            buf.as_mut_ptr() as *mut c_char,
            JMSG_LENGTH_MAX,
            fmt.as_ptr() as *const c_char,
            s_in.as_ptr() as *const c_char,
        );
    }
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(JMSG_LENGTH_MAX);
    buf[..nul].to_vec()
}

fn snprintf_ref_ints_8(format_str: &[u8], args: [c_int; 8]) -> Vec<u8> {
    let mut fmt: Vec<u8> = format_str.to_vec();
    fmt.push(0);
    let mut buf: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];
    unsafe {
        snprintf(
            buf.as_mut_ptr() as *mut c_char,
            JMSG_LENGTH_MAX,
            fmt.as_ptr() as *const c_char,
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        );
    }
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(JMSG_LENGTH_MAX);
    buf[..nul].to_vec()
}

fn assert_formats_match_int(lib: &Library, format_str: &[u8], args: [c_int; 8]) {
    let expected = snprintf_ref_ints_8(format_str, args);
    let actual = invoke_shim_format_message(lib, format_str, None, &args);
    assert_eq!(
        actual,
        expected,
        "format `{}` with args {:?}: shim emitted `{}`, libc emitted `{}`",
        String::from_utf8_lossy(format_str),
        args,
        String::from_utf8_lossy(&actual),
        String::from_utf8_lossy(&expected),
    );
}

fn assert_formats_match_string(lib: &Library, format_str: &[u8], string_arg: &[u8]) {
    let expected = snprintf_ref_string(format_str, string_arg);
    let actual = invoke_shim_format_message(lib, format_str, Some(string_arg), &[]);
    assert_eq!(
        actual,
        expected,
        "format `{}` with string arg `{}`: shim emitted `{}`, libc emitted `{}`",
        String::from_utf8_lossy(format_str),
        String::from_utf8_lossy(string_arg),
        String::from_utf8_lossy(&actual),
        String::from_utf8_lossy(&expected),
    );
}

#[test]
fn format_message_d_specifier_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    assert_formats_match_int(&lib, b"got %d", [42, 0, 0, 0, 0, 0, 0, 0]);
    assert_formats_match_int(&lib, b"got %d, %d", [-7, 99, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn format_message_u_specifier_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    assert_formats_match_int(&lib, b"%u bytes", [12345, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn format_message_x_specifier_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    assert_formats_match_int(&lib, b"hex %x", [0xDEAD, 0, 0, 0, 0, 0, 0, 0]);
    assert_formats_match_int(&lib, b"%04x %02x", [0x1234, 0xff, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn format_message_c_specifier_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    assert_formats_match_int(&lib, b"marker 0x%2c%2c", [0xFF, 0xD8, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn format_message_zero_padded_d_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    assert_formats_match_int(&lib, b"line %02d:%02d", [3, 7, 0, 0, 0, 0, 0, 0]);
    assert_formats_match_int(&lib, b"row %3d", [42, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn format_message_percent_literal_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    // The C contract: jerror.c chooses int-mode (%% has no specifier)
    // and passes msg_parm.i[..]. snprintf with no consumed args returns
    // the literal string. Our shim must do the same.
    assert_formats_match_int(&lib, b"100%% sure", [0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn format_message_s_specifier_matches_snprintf() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    assert_formats_match_string(&lib, b"file: %s", b"my_test.jpg");
    assert_formats_match_string(&lib, b"%s", b"hello world");
}

#[test]
fn format_message_no_specifier_matches_msgtext_verbatim() {
    let lib = unsafe { Library::new(cdylib_path()).expect("dlopen cdylib") };
    let msg: &[u8] = b"Premature end of JPEG file";
    let actual = invoke_shim_format_message(&lib, msg, None, &[0; 8]);
    assert_eq!(actual, msg);
}
