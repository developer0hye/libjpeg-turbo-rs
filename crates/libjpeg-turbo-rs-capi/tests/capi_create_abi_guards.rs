//! P4-110: `jpeg_CreateCompress` / `jpeg_CreateDecompress` must validate
//! `version` and `structsize` before writing the caller's object.
//!
//! This is the item's P0. Both functions ignored their two guard parameters
//! and wrote the shim's full Rust mirror regardless, so a caller that declared
//! a smaller struct — an older ABI, a different build, or simply a wrong
//! `sizeof` — had memory written past the end of its allocation. Several tests
//! in this crate passed a 4096-byte blob and declared 4096, which normalised
//! the gap rather than exposing it.
//!
//! Rejecting correctly matters as much as detecting: which error code, which
//! two parameters in which order, which check wins when both are wrong, and
//! how much of the caller's object may be touched before the guard fires.
//! `examples/create_abi_guards_oracle.c` answers all of that from a real
//! libjpeg, and this test compares the shim's behaviour against it line for
//! line rather than against a reading of `jdapimin.c`.
//!
//! The observable contract, as measured:
//!
//! ```text
//! ok               fired=0                          canary=1 err_kept=1 client_kept=1
//! badversion_low   fired=1 code=13 parm0=80 parm1=70 canary=1 mem_null=1
//! badsize_small    fired=1 code=22 parm0=<real> parm1=<declared> canary=1 mem_null=1
//! both_wrong       fired=1 code=13 ...              version is checked first
//! ```

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateCompress, jpeg_CreateDecompress, jpeg_abort, jpeg_destroy, jpeg_destroy_compress,
    jpeg_destroy_decompress, jpeg_std_error, JpegCompressPublic, JpegDecompressPublic,
    JpegErrorMgr,
};

mod helpers;

/// Matches the oracle's constants exactly; the two harnesses must agree byte
/// for byte or the comparison is meaningless.
const CANARY: usize = 64;
const CANARY_BYTE: u8 = 0x5A;
const FILL_BYTE: u8 = 0xAA;
/// A recognisable non-null value for `client_data`. It is compared, never
/// dereferenced, so it is built with `without_provenance_mut` — an
/// int-to-pointer cast would warn under Miri's strict provenance for no reason.
const CLIENT_DATA_SENTINEL: usize = 0x1234_ABCD;
const JPEG_LIB_VERSION: c_int = 80;

/// Error manager plus recorded outcome, recovered inside the callback by
/// casting `cinfo->err` — the same trick the C harness uses, and the reason
/// nothing here needs a `static`, which would race across parallel tests.
#[repr(C)]
struct TrapErrorMgr {
    pub_mgr: JpegErrorMgr,
    fired: c_int,
    msg_code: c_int,
    parm0: c_int,
    parm1: c_int,
}

/// The pointer handed to `jpeg_std_error` must be derived from the **whole**
/// `TrapErrorMgr`, not from its `pub_mgr` field. Deriving it from the field
/// gives provenance over only that field's bytes, and [`trap_error_exit`] then
/// writes `fired` just past them — which Miri rejects (and which this test hit
/// on its first Miri run). C's `struct my_error_mgr { struct jpeg_error_mgr
/// pub; ... }` idiom is the same shape; the cast has to start from the outer
/// object.
///
/// Records the error and returns. A Rust `error_exit` cannot `longjmp` the way
/// the C harness does, and must not unwind across `extern "C"`; the shim's
/// contract is that it stops touching the caller's object once the handler has
/// been invoked, which is exactly what the canary checks.
unsafe extern "C" fn trap_error_exit(cinfo: *mut c_void) {
    // SAFETY: every caller below sets `err` to a live `TrapErrorMgr` whose
    // error manager is its first field. Raw reads only: the shim may still be
    // writing through `cinfo`, so no Rust reference may span this callback.
    unsafe {
        let err: *mut TrapErrorMgr =
            (*(cinfo as *mut JpegDecompressPublic)).err as *mut TrapErrorMgr;
        if err.is_null() {
            return;
        }
        (*err).fired = 1;
        (*err).msg_code = (*err).pub_mgr.msg_code;
        (*err).parm0 = read_parm(&(*err).pub_mgr, 0);
        (*err).parm1 = read_parm(&(*err).pub_mgr, 1);
    }
}

/// `msg_parm` mirrors upstream's `union { int i[8]; char s[80]; }` as raw
/// bytes; pull the `n`th `int` back out of it.
fn read_parm(err: &JpegErrorMgr, n: usize) -> c_int {
    let width: usize = std::mem::size_of::<c_int>();
    let start: usize = n * width;
    let mut bytes: [u8; std::mem::size_of::<c_int>()] = Default::default();
    bytes.copy_from_slice(&err.msg_parm[start..start + width]);
    c_int::from_ne_bytes(bytes)
}

unsafe extern "C" fn quiet_output_message(_cinfo: *mut c_void) {}

/// An allocation of exactly `declared` bytes plus a canary, aligned for the
/// struct. Deliberately *not* a `Vec<JpegDecompressPublic>`: the point is that
/// the caller's object may be smaller than the shim's mirror, and a test that
/// always allocates the full mirror can never catch a write past its end.
struct Block {
    ptr: *mut u8,
    layout: Layout,
    declared: usize,
}

impl Block {
    fn new(declared: usize, align: usize) -> Self {
        let layout: Layout =
            Layout::from_size_align(declared + CANARY, align).expect("valid layout");
        // SAFETY: non-zero size, valid alignment.
        let ptr: *mut u8 = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        // SAFETY: `ptr` is valid for `declared + CANARY` bytes.
        unsafe {
            std::ptr::write_bytes(ptr, FILL_BYTE, declared);
            std::ptr::write_bytes(ptr.add(declared), CANARY_BYTE, CANARY);
        }
        Block {
            ptr,
            layout,
            declared,
        }
    }

    /// 1 when nothing was written past the object the caller declared.
    fn canary_intact(&self) -> c_int {
        // SAFETY: the canary bytes are inside this allocation.
        let tail: &[u8] =
            unsafe { std::slice::from_raw_parts(self.ptr.add(self.declared), CANARY) };
        c_int::from(tail.iter().all(|&b| b == CANARY_BYTE))
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`layout` are the pair returned by `alloc` above.
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

/// One decompressor case, rendered in the oracle's line format.
fn run_decompress(label: &str, version: c_int, declared: usize) -> String {
    let real: usize = std::mem::size_of::<JpegDecompressPublic>();
    let block: Block = Block::new(declared, std::mem::align_of::<JpegDecompressPublic>());
    let p: *mut c_void = block.ptr as *mut c_void;

    let mut trap: Box<TrapErrorMgr> = Box::new(TrapErrorMgr {
        // SAFETY: `JpegErrorMgr` is `#[repr(C)]` plain data; all-zero is a
        // valid value and `jpeg_std_error` overwrites what matters.
        pub_mgr: unsafe { std::mem::zeroed() },
        fired: 0,
        msg_code: 0,
        parm0: 0,
        parm1: 0,
    });

    // SAFETY: `block` is valid for `declared + CANARY` bytes and `trap`
    // outlives every call. Field writes go through raw pointers because the
    // object may be smaller than the full mirror — forming a
    // `&mut JpegDecompressPublic` over it would itself be the bug under test.
    let line: String = unsafe {
        let errp: *mut JpegErrorMgr =
            jpeg_std_error(&mut *trap as *mut TrapErrorMgr as *mut JpegErrorMgr);
        (*errp).error_exit = Some(trap_error_exit);
        (*errp).output_message = Some(quiet_output_message);

        let base: *mut JpegDecompressPublic = block.ptr as *mut JpegDecompressPublic;
        std::ptr::addr_of_mut!((*base).err).write(errp);
        std::ptr::addr_of_mut!((*base).client_data).write(
            std::ptr::without_provenance_mut::<c_void>(CLIENT_DATA_SENTINEL),
        );

        jpeg_CreateDecompress(p, version, declared);

        let mut line: String = format!(
            "{label} fired={} code={} parm0={} parm1={} canary={}",
            trap.fired,
            trap.msg_code,
            trap.parm0,
            trap.parm1,
            block.canary_intact()
        );
        if trap.fired == 0 {
            line.push_str(&format!(
                " err_kept={} client_kept={} is_decompressor={} global_state={} \
                src_zero={} width_zero={}",
                c_int::from(std::ptr::addr_of!((*base).err).read() == errp),
                c_int::from(
                    std::ptr::addr_of!((*base).client_data).read() as usize == CLIENT_DATA_SENTINEL
                ),
                std::ptr::addr_of!((*base).is_decompressor).read(),
                std::ptr::addr_of!((*base).global_state).read(),
                c_int::from(std::ptr::addr_of!((*base).src).read().is_null()),
                c_int::from(std::ptr::addr_of!((*base).image_width).read() == 0),
            ));
            jpeg_destroy_decompress(p);
        } else {
            line.push_str(&format!(
                " mem_null={}",
                c_int::from(std::ptr::addr_of!((*base).mem).read().is_null())
            ));
            // Must be safe after a rejected create — that is what nulling
            // `mem` first is for.
            jpeg_destroy_decompress(p);
        }
        line.push_str(&format!(" real={real}\n"));
        line
    };
    line
}

/// One compressor case, rendered in the oracle's line format.
fn run_compress(label: &str, version: c_int, declared: usize) -> String {
    let real: usize = std::mem::size_of::<JpegCompressPublic>();
    let block: Block = Block::new(declared, std::mem::align_of::<JpegCompressPublic>());
    let p: *mut c_void = block.ptr as *mut c_void;

    let mut trap: Box<TrapErrorMgr> = Box::new(TrapErrorMgr {
        // SAFETY: as above.
        pub_mgr: unsafe { std::mem::zeroed() },
        fired: 0,
        msg_code: 0,
        parm0: 0,
        parm1: 0,
    });

    // SAFETY: as in `run_decompress`; `err` and `client_data` sit at the same
    // offsets in both structs (the `jpeg_common_fields` prefix).
    let line: String = unsafe {
        let errp: *mut JpegErrorMgr =
            jpeg_std_error(&mut *trap as *mut TrapErrorMgr as *mut JpegErrorMgr);
        (*errp).error_exit = Some(trap_error_exit);
        (*errp).output_message = Some(quiet_output_message);

        let base: *mut JpegCompressPublic = block.ptr as *mut JpegCompressPublic;
        std::ptr::addr_of_mut!((*base).err).write(errp);
        std::ptr::addr_of_mut!((*base).client_data).write(
            std::ptr::without_provenance_mut::<c_void>(CLIENT_DATA_SENTINEL),
        );

        jpeg_CreateCompress(p, version, declared);

        let mut line: String = format!(
            "{label} fired={} code={} parm0={} parm1={} canary={}",
            trap.fired,
            trap.msg_code,
            trap.parm0,
            trap.parm1,
            block.canary_intact()
        );
        if trap.fired == 0 {
            line.push_str(&format!(
                " err_kept={} client_kept={} is_decompressor={} global_state={} \
                dest_zero={} width_zero={}",
                c_int::from(std::ptr::addr_of!((*base).err).read() == errp),
                c_int::from(
                    std::ptr::addr_of!((*base).client_data).read() as usize == CLIENT_DATA_SENTINEL
                ),
                std::ptr::addr_of!((*base).is_decompressor).read(),
                std::ptr::addr_of!((*base).global_state).read(),
                c_int::from(std::ptr::addr_of!((*base).dest).read().is_null()),
                c_int::from(std::ptr::addr_of!((*base).image_width).read() == 0),
            ));
            jpeg_destroy_compress(p);
        } else {
            line.push_str(&format!(
                " mem_null={}",
                c_int::from(std::ptr::addr_of!((*base).mem).read().is_null())
            ));
            jpeg_destroy_compress(p);
        }
        line.push_str(&format!(" real={real}\n"));
        line
    };
    line
}

fn decompress_trace() -> String {
    let real: usize = std::mem::size_of::<JpegDecompressPublic>();
    let mut out: String = String::new();
    out.push_str(&run_decompress("ok", JPEG_LIB_VERSION, real));
    out.push_str(&run_decompress(
        "badversion_low",
        JPEG_LIB_VERSION - 10,
        real,
    ));
    out.push_str(&run_decompress(
        "badversion_high",
        JPEG_LIB_VERSION + 10,
        real,
    ));
    out.push_str(&run_decompress("badsize_small", JPEG_LIB_VERSION, real - 8));
    out.push_str(&run_decompress("badsize_large", JPEG_LIB_VERSION, real + 8));
    out.push_str(&run_decompress(
        "both_wrong",
        JPEG_LIB_VERSION - 10,
        real - 8,
    ));
    out
}

fn compress_trace() -> String {
    let real: usize = std::mem::size_of::<JpegCompressPublic>();
    let mut out: String = String::new();
    out.push_str(&run_compress("ok", JPEG_LIB_VERSION, real));
    out.push_str(&run_compress("badversion_low", JPEG_LIB_VERSION - 10, real));
    out.push_str(&run_compress(
        "badversion_high",
        JPEG_LIB_VERSION + 10,
        real,
    ));
    out.push_str(&run_compress("badsize_small", JPEG_LIB_VERSION, real - 8));
    out.push_str(&run_compress("badsize_large", JPEG_LIB_VERSION, real + 8));
    out.push_str(&run_compress("both_wrong", JPEG_LIB_VERSION - 10, real - 8));
    out
}

/// The whole contract, compared against a real libjpeg.
///
/// Includes `real=<sizeof>` on every line, so a mirror whose size has drifted
/// from the C struct fails here too — the guard would otherwise start
/// rejecting correct callers, which is worse than the bug it fixes.
#[test]
#[cfg_attr(
    miri,
    ignore = "builds and runs a C oracle binary; Miri cannot spawn a process"
)]
fn create_guards_match_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("create_abi_guards_oracle") else {
        eprintln!(
            "SKIP: no stock libjpeg development install found; the C oracle for \
             P4-110's create guards cannot be built. Set LIBJPEG_TURBO_PREFIX \
             to make this a hard failure."
        );
        return;
    };

    let c_decompress: String = helpers::run_oracle(&oracle, &["decompress"]);
    assert_eq!(
        decompress_trace(),
        c_decompress,
        "jpeg_CreateDecompress guard behaviour diverges from stock libjpeg"
    );

    let c_compress: String = helpers::run_oracle(&oracle, &["compress"]);
    assert_eq!(
        compress_trace(),
        c_compress,
        "jpeg_CreateCompress guard behaviour diverges from stock libjpeg"
    );
}

/// The memory-safety claim, stated without reference to the oracle so it
/// survives a machine that has no C libjpeg.
///
/// A caller declaring a smaller struct must get an error and an untouched
/// object beyond what it declared — the P0. Before the guard, the shim wrote
/// its whole mirror here, past the end of the caller's allocation.
#[test]
fn undersized_struct_is_rejected_without_writing_past_it() {
    let real: usize = std::mem::size_of::<JpegDecompressPublic>();
    for shrink in [8usize, 64, real / 2] {
        let line: String = run_decompress("undersized", JPEG_LIB_VERSION, real - shrink);
        assert!(
            line.contains("fired=1 code=22"),
            "declaring {} bytes fewer than the real struct must raise \
             JERR_BAD_STRUCT_SIZE (22): {line}",
            shrink
        );
        assert!(
            line.contains("canary=1"),
            "nothing may be written past the caller's declared object: {line}"
        );
        assert!(
            line.contains("mem_null=1"),
            "`mem` must be NULL so jpeg_destroy stays safe after rejection: {line}"
        );
    }
}

/// The accepted path's two preserved fields, likewise oracle-independent.
///
/// Upstream zeroes the whole struct and puts `err` and `client_data` back;
/// this shim used to null `client_data`, silently dropping the pointer an
/// application had already attached for its callbacks.
#[test]
fn accepted_create_preserves_err_and_client_data() {
    let real: usize = std::mem::size_of::<JpegDecompressPublic>();
    let line: String = run_decompress("ok", JPEG_LIB_VERSION, real);
    assert!(
        line.contains("fired=0"),
        "exact size must be accepted: {line}"
    );
    assert!(
        line.contains("err_kept=1"),
        "err must survive create: {line}"
    );
    assert!(
        line.contains("client_kept=1"),
        "client_data must survive create — an application sets it before \
         calling, and its callbacks read it back: {line}"
    );
    assert!(
        line.contains("width_zero=1"),
        "fields the caller did not set must be zeroed, not left as whatever \
         the allocation held: {line}"
    );
}

/// The rejection path must not *read* past the caller's object either.
///
/// The cases above allocate a canary past the declared size, so the allocation
/// is always at least as large as the shim's mirror — which means they cannot
/// detect a reference formed over the full mirror, only a write through one.
/// This one allocates **exactly** what the caller declared, so any oversized
/// borrow is a genuine out-of-bounds access. Under Miri that is a hard error;
/// natively it is the case that would fault on a page boundary.
///
/// It covers the whole rejection path, error handler included: the handler is
/// the shim's own `default_error_exit` here, which used to build a full
/// `&mut JpegDecompressPublic` just to read `err` at offset 0.
#[test]
fn rejection_path_touches_only_the_declared_object() {
    for (label, version, shrink) in [
        ("undersized", JPEG_LIB_VERSION, 8usize),
        ("undersized-half", JPEG_LIB_VERSION, 128),
        ("badversion", JPEG_LIB_VERSION - 10, 8),
    ] {
        let real: usize = std::mem::size_of::<JpegDecompressPublic>();
        let declared: usize = real - shrink;
        let layout: Layout =
            Layout::from_size_align(declared, std::mem::align_of::<JpegDecompressPublic>())
                .expect("valid layout");
        // SAFETY: non-zero size and a valid alignment.
        let block: *mut u8 = unsafe { alloc(layout) };
        assert!(!block.is_null(), "allocation failed");

        let mut trap: Box<TrapErrorMgr> = Box::new(TrapErrorMgr {
            // SAFETY: all-zero is valid for this `#[repr(C)]` plain-data struct.
            pub_mgr: unsafe { std::mem::zeroed() },
            fired: 0,
            msg_code: 0,
            parm0: 0,
            parm1: 0,
        });

        // SAFETY: `block` is valid for exactly `declared` bytes — deliberately
        // fewer than the mirror — and `err` sits at offset 0, inside it.
        unsafe {
            std::ptr::write_bytes(block, FILL_BYTE, declared);
            let errp: *mut JpegErrorMgr =
                jpeg_std_error(&mut *trap as *mut TrapErrorMgr as *mut JpegErrorMgr);
            (*errp).error_exit = Some(trap_error_exit);
            (*errp).output_message = Some(quiet_output_message);
            (block as *mut *mut JpegErrorMgr).write(errp);

            jpeg_CreateDecompress(block as *mut c_void, version, declared);
            assert_eq!(trap.fired, 1, "{label}: the declaration must be rejected");

            // Teardown is what a caller's error handler does next, and it
            // must stay inside the object too — through *both* doors. The
            // generic `jpeg_destroy` / `jpeg_abort` pick their path by reading
            // `is_decompressor`, which a rejected create leaves indeterminate:
            // only `err` (the caller) and `mem` (the guard) are initialized.
            jpeg_abort(block as *mut c_void);
            jpeg_destroy(block as *mut c_void);
            jpeg_destroy_decompress(block as *mut c_void);

            dealloc(block, layout);
        }
    }
}

/// The *default* error handler is on this path too, and it aborts.
///
/// Every test above installs its own `error_exit`, so none of them reach
/// `jpeg_std_error`'s default — which is what a caller that never sets one
/// gets, and which `jpeg_CreateDecompress`'s guards therefore invoke. It
/// aborts the process by design, so it can only be observed from a child.
///
/// What this pins: the rejection reaches the default handler, and the handler
/// reports **both** `ERREXIT2` parameters. Reporting only the first turns
/// "library is 80, caller expects 70" into a bare 80, which is the number the
/// reader already knew.
///
/// It does not prove the handler stays inside the caller's object — Miri
/// cannot follow a process abort. That half rests on the handler reading only
/// `err`, at offset 0 of the common prefix, through a raw pointer.
#[test]
#[cfg_attr(
    miri,
    ignore = "re-execs this test binary; Miri cannot spawn a process"
)]
fn default_error_handler_reports_both_parameters_and_aborts() {
    const CHILD_ENV: &str = "P4110_DEFAULT_HANDLER_CHILD";

    if std::env::var_os(CHILD_ENV).is_some() {
        // Child: undersize the declaration and let the default handler run.
        let real: usize = std::mem::size_of::<JpegDecompressPublic>();
        let declared: usize = real - 8;
        let layout: Layout =
            Layout::from_size_align(declared, std::mem::align_of::<JpegDecompressPublic>())
                .expect("valid layout");
        // SAFETY: non-zero size, valid alignment.
        let block: *mut u8 = unsafe { alloc(layout) };
        assert!(!block.is_null());
        let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
        // SAFETY: `block` is valid for `declared` bytes and `err` sits at
        // offset 0. No `error_exit` override: the default one is the point.
        unsafe {
            std::ptr::write_bytes(block, FILL_BYTE, declared);
            let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
            (block as *mut *mut JpegErrorMgr).write(errp);
            jpeg_CreateDecompress(block as *mut c_void, JPEG_LIB_VERSION, declared);
        }
        unreachable!("the default error handler must abort before returning");
    }

    let exe: std::path::PathBuf = std::env::current_exe().expect("locate test binary");
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("default_error_handler_reports_both_parameters_and_aborts")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn child");

    assert!(
        !output.status.success(),
        "child must abort, exited {:?}",
        output.status
    );
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        stderr.contains("msg_code=22"),
        "child must report JERR_BAD_STRUCT_SIZE (22): {stderr}"
    );
    let expected_real: usize = std::mem::size_of::<JpegDecompressPublic>();
    assert!(
        stderr.contains(&format!("parm0={expected_real}")),
        "parm0 must be the library's own sizeof: {stderr}"
    );
    assert!(
        stderr.contains(&format!("parm1={}", expected_real - 8)),
        "parm1 must be what the caller declared — reporting only parm0 tells \
         the reader a number they already knew: {stderr}"
    );
}

/// The standard idiom: allocate the struct, set `err`, call create — and leave
/// everything else, `client_data` included, **uninitialized**.
///
/// This is what stock `djpeg` does (`djpeg.c:573-589`), and upstream says so
/// itself: the application "may have set client_data", with a note that "tools
/// like Purify may complain here". C tolerates copying an indeterminate value;
/// Rust does not — a typed load of uninitialized memory is undefined whatever
/// happens to the value afterwards. Create therefore zeroes *around* the two
/// preserved slots rather than through them.
///
/// The other tests all pre-fill the whole allocation, so none of them can
/// catch this: the memory is initialized before create ever sees it. Here the
/// backing store is deliberately left as `alloc` returned it, so under Miri
/// any load of the untouched slot is a hard error.
#[test]
fn create_never_loads_an_uninitialized_client_data() {
    let real: usize = std::mem::size_of::<JpegDecompressPublic>();
    let layout: Layout =
        Layout::from_size_align(real, std::mem::align_of::<JpegDecompressPublic>())
            .expect("valid layout");
    // SAFETY: non-zero size, valid alignment. Deliberately *not* zeroed and
    // deliberately not written past the `err` slot.
    let block: *mut u8 = unsafe { alloc(layout) };
    assert!(!block.is_null(), "allocation failed");

    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });

    // SAFETY: `block` is valid for `real` bytes; only `err` (offset 0) is
    // initialized before the call, exactly as the documented idiom leaves it.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        (*errp).error_exit = Some(trap_error_exit_boxed);
        (*errp).output_message = Some(quiet_output_message);
        (block as *mut *mut JpegErrorMgr).write(errp);

        jpeg_CreateDecompress(block as *mut c_void, JPEG_LIB_VERSION, real);

        // Accepted, and the fields create is responsible for are set.
        assert_eq!(
            std::ptr::addr_of!((*(block as *mut JpegDecompressPublic)).is_decompressor).read(),
            1,
            "create must have run"
        );
        assert_eq!(
            std::ptr::addr_of!((*(block as *mut JpegDecompressPublic)).image_width).read(),
            0,
            "unrelated fields must be zeroed even from uninitialized memory"
        );
        // `client_data` is deliberately *not* read here: it is still whatever
        // the allocator left, which is the caller's business and no one else's.

        jpeg_destroy_decompress(block as *mut c_void);
        dealloc(block, layout);
    }
}

/// An `error_exit` for the case above: it must not touch `cinfo`, whose
/// `client_data` is uninitialized on purpose.
unsafe extern "C" fn trap_error_exit_boxed(_cinfo: *mut c_void) {}
