//! P4-132 (#463): a classic `cinfo` may change threads between calls.
//!
//! Upstream's contract is "single-threaded per `cinfo`, but ownership transfer
//! between threads is OK provided the application enforces non-concurrent
//! access" — FFmpeg's frame-threaded JPEG path hands a codec context from the
//! thread that opened it to the thread that decodes with it. The shim used to
//! key the decompressor's private state on the `cinfo` address in a
//! `thread_local!` map, so the same sequence on our library looked up nothing
//! on the second thread and leaked the first thread's entry.
//!
//! What is pinned here, in the order the issue lists it:
//!
//! 1. create on thread A, drive and destroy on thread B — both directions of
//!    the API, decompress and compress;
//! 2. the destroy on thread B *releases* the private state (counted through
//!    the crate's non-exported test hooks, so a leak is a number, not a
//!    guess);
//! 3. a `cinfo` destroyed and re-created **at the same address** starts from
//!    fresh state — the collision a pointer-keyed table has to defend
//!    against;
//! 4. distinct `cinfo`s on distinct threads run concurrently without
//!    synchronisation — the shape that *is* supported. Concurrent use of one
//!    `cinfo` from two threads is undefined behaviour, exactly as upstream
//!    documents it, and is not exercised: there is no observable contract to
//!    assert, only a data race.
//!
//! The pixels a moved `cinfo` produces are cross-validated against `djpeg`
//! byte for byte where the oracle is installed, and against the same-thread
//! run unconditionally — moving threads must change nothing.
//!
//! One more test runs the transfer sequence under Miri (see
//! [`moved_decode_sequence_is_sound_under_miri`]); the rest need the crate's
//! SIMD encoder and a spawned `djpeg`, which Miri cannot interpret, so they
//! carry `cfg_attr(miri, ignore)`.

use std::ffi::{c_int, c_void};
use std::os::raw::c_ulong;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::sync::{Mutex, MutexGuard};
use std::thread;

use libjpeg_turbo_rs::{PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateCompress, jpeg_CreateDecompress, jpeg_destroy_compress, jpeg_destroy_decompress,
    jpeg_finish_compress, jpeg_finish_decompress, jpeg_mem_dest, jpeg_mem_src, jpeg_read_header,
    jpeg_read_scanlines, jpeg_save_markers, jpeg_set_defaults, jpeg_set_quality,
    jpeg_start_compress, jpeg_start_decompress, jpeg_std_error, jpeg_write_scanlines,
    live_compress_private_count_for_tests, live_decompress_private_count_for_tests,
    JpegCompressPublic, JpegDecompressPublic, JpegErrorMgr,
};

extern "C" {
    /// `jpeg_mem_dest` hands back a `malloc`ed buffer the caller frees.
    fn free(ptr: *mut c_void);
}

const JPEG_LIB_VERSION: c_int = 80;
const JPEG_HEADER_OK: c_int = 1;
const JCS_RGB: c_int = 2;
/// `JPEG_APP0 + 1`, the marker the reuse test asks the first lifecycle to
/// keep and the second one not to.
const JPEG_APP1: c_int = 0xE1;

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const QUALITY: u8 = 90;

/// The live-instance counters are process-wide and `cargo test` runs the
/// tests of one binary as parallel threads, so every test that reads a
/// counter holds this while it does. A poisoned lock is still a usable lock:
/// the test that poisoned it has already failed on its own.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A raw pointer that may cross a thread boundary. The tests are the
/// application upstream's contract talks about: they enforce that only one
/// thread touches the object at a time by joining before the next use.
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
// Hand-written so the pointer is `Copy` for every `T`; `derive` would add a
// `T: Copy` bound the ABI mirrors do not meet.
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

/// Error manager plus a record of whether `error_exit` fired. Recovered
/// inside the callback by casting `cinfo->err`, the same way a C consumer
/// embeds `struct jpeg_error_mgr` as the first member of its own struct.
#[repr(C)]
struct TrapErrorMgr {
    pub_mgr: JpegErrorMgr,
    fired: c_int,
}

impl TrapErrorMgr {
    fn boxed() -> Box<TrapErrorMgr> {
        let mut trap: Box<TrapErrorMgr> = Box::new(TrapErrorMgr {
            pub_mgr: unsafe { std::mem::zeroed() },
            fired: 0,
        });
        // The pointer is derived from the whole box so the callback's write
        // to `fired` stays inside the provenance of what `jpeg_std_error` was
        // given (Miri rejects the field-derived form).
        let whole: *mut TrapErrorMgr = &mut *trap;
        unsafe {
            jpeg_std_error(whole as *mut JpegErrorMgr);
            (*whole).pub_mgr.error_exit = Some(trap_error_exit);
            (*whole).pub_mgr.output_message = Some(quiet_output_message);
        }
        trap
    }
}

/// Records the failure and returns. A Rust `error_exit` cannot `longjmp`, and
/// the shim's contract is that it stops touching the object once the handler
/// has run — the return codes checked below are what a C caller would see.
unsafe extern "C" fn trap_error_exit(cinfo: *mut c_void) {
    unsafe {
        let err: *mut TrapErrorMgr =
            (*(cinfo as *mut JpegDecompressPublic)).err as *mut TrapErrorMgr;
        if !err.is_null() {
            (*err).fired = 1;
        }
    }
}

unsafe extern "C" fn quiet_output_message(_cinfo: *mut c_void) {}

/// A 64×48 RGB gradient with enough structure that a wrong row or a wrong
/// component is a visible diff, not a coincidental match.
fn source_pixels() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push((x * 4) as u8);
            pixels.push((y * 5) as u8);
            pixels.push(((x + y) * 2) as u8);
        }
    }
    pixels
}

/// The JPEG every decode below reads: the crate's own encoder at a fixed
/// quality and 4:2:0, so the fixture needs no file and no oracle.
fn fixture_jpeg() -> Vec<u8> {
    libjpeg_turbo_rs::compress(
        &source_pixels(),
        WIDTH,
        HEIGHT,
        PixelFormat::Rgb,
        QUALITY,
        Subsampling::S420,
    )
    .expect("encode the fixture")
}

/// The fixture with a two-byte APP1 segment spliced in after SOI. Every
/// decoder skips it; only a `jpeg_save_markers(APP1)` lifecycle keeps it.
fn fixture_jpeg_with_app1() -> Vec<u8> {
    let jpeg: Vec<u8> = fixture_jpeg();
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "fixture starts with SOI");
    let mut spliced: Vec<u8> = Vec::with_capacity(jpeg.len() + 6);
    spliced.extend_from_slice(&jpeg[..2]);
    spliced.extend_from_slice(&[0xFF, 0xE1, 0x00, 0x04, b'A', b'B']);
    spliced.extend_from_slice(&jpeg[2..]);
    spliced
}

// ---------- classic lifecycles, callable from any thread ----------

/// `jpeg_create_decompress` on a caller-owned object. The object's memory is
/// owned by the test, not by the thread that creates on it, which is what
/// lets the reuse test put a second lifecycle at the same address.
unsafe fn create_decompress(cinfo: *mut JpegDecompressPublic, err: *mut TrapErrorMgr) {
    unsafe {
        std::ptr::write_bytes(cinfo, 0, 1);
        (*cinfo).err = err as *mut JpegErrorMgr;
        jpeg_CreateDecompress(
            cinfo as *mut c_void,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
    }
    assert!(
        !unsafe { (*cinfo).mem }.is_null(),
        "jpeg_CreateDecompress must accept the v8 mirror"
    );
}

/// `jpeg_mem_src` through `jpeg_finish_decompress`, returning the RGB rows.
/// Deliberately does **not** destroy: the tests choose where that happens.
/// `after_header` runs between `jpeg_read_header` and `jpeg_start_decompress`,
/// where `marker_list` is populated (the shim clears it on finish).
unsafe fn decode_with(
    cinfo: *mut JpegDecompressPublic,
    jpeg: &[u8],
    after_header: impl FnOnce(&JpegDecompressPublic),
) -> Vec<u8> {
    let cv: *mut c_void = cinfo as *mut c_void;
    let mut pixels: Vec<u8> = Vec::new();
    unsafe {
        jpeg_mem_src(cv, jpeg.as_ptr(), jpeg.len() as c_ulong);
        assert_eq!(
            jpeg_read_header(cv, 1),
            JPEG_HEADER_OK,
            "jpeg_read_header on thread {:?}",
            thread::current().id()
        );
        after_header(&*cinfo);
        assert_eq!(jpeg_start_decompress(cv), 1, "jpeg_start_decompress");
        let width: usize = (*cinfo).output_width as usize;
        let height: usize = (*cinfo).output_height as usize;
        let components: usize = (*cinfo).output_components as usize;
        assert_eq!((width, height, components), (WIDTH, HEIGHT, 3));
        let stride: usize = width * components;
        pixels.resize(stride * height, 0);
        // `loop` rather than `while`: the counter advances inside the shim,
        // behind a raw pointer, which clippy cannot see.
        loop {
            let row: usize = (*cinfo).output_scanline as usize;
            if row >= height {
                break;
            }
            let mut row_ptr: *mut u8 = pixels.as_mut_ptr().add(row * stride);
            assert_eq!(
                jpeg_read_scanlines(cv, &mut row_ptr, 1),
                1,
                "jpeg_read_scanlines stalled at row {row}"
            );
        }
        assert_eq!(jpeg_finish_decompress(cv), 1, "jpeg_finish_decompress");
    }
    pixels
}

unsafe fn create_compress(cinfo: *mut JpegCompressPublic, err: *mut TrapErrorMgr) {
    unsafe {
        std::ptr::write_bytes(cinfo, 0, 1);
        (*cinfo).err = err as *mut JpegErrorMgr;
        jpeg_CreateCompress(
            cinfo as *mut c_void,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegCompressPublic>(),
        );
    }
    assert!(
        !unsafe { (*cinfo).mem }.is_null(),
        "jpeg_CreateCompress must accept the v8 mirror"
    );
}

/// `jpeg_mem_dest` through `jpeg_finish_compress`, returning the JPEG bytes.
/// Like [`decode_with`], leaves the destroy to the caller.
unsafe fn encode_with(cinfo: *mut JpegCompressPublic, pixels: &[u8]) -> Vec<u8> {
    let cv: *mut c_void = cinfo as *mut c_void;
    let mut out_buf: *mut u8 = std::ptr::null_mut();
    let mut out_size: c_ulong = 0;
    unsafe {
        jpeg_mem_dest(cv, &mut out_buf, &mut out_size);
        (*cinfo).image_width = WIDTH as u32;
        (*cinfo).image_height = HEIGHT as u32;
        (*cinfo).input_components = 3;
        (*cinfo).in_color_space = JCS_RGB;
        jpeg_set_defaults(cv);
        jpeg_set_quality(cv, QUALITY as c_int, 1);
        jpeg_start_compress(cv, 1);
        let stride: usize = WIDTH * 3;
        loop {
            let row: usize = (*cinfo).next_scanline as usize;
            if row >= HEIGHT {
                break;
            }
            let mut row_ptr: *mut u8 = pixels.as_ptr().add(row * stride) as *mut u8;
            assert_eq!(
                jpeg_write_scanlines(cv, &mut row_ptr, 1),
                1,
                "jpeg_write_scanlines stalled at row {row} on thread {:?}",
                thread::current().id()
            );
        }
        jpeg_finish_compress(cv);
        assert!(
            !out_buf.is_null() && out_size > 0,
            "jpeg_mem_dest produced nothing"
        );
        let jpeg: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        free(out_buf as *mut c_void);
        jpeg
    }
}

fn assert_no_error(err: &TrapErrorMgr, what: &str) {
    assert_eq!(
        err.fired, 0,
        "{what}: error_exit fired with msg_code {}",
        err.pub_mgr.msg_code
    );
}

// ---------- the C oracle ----------

fn find_c_tool(name: &str) -> Option<PathBuf> {
    if let Some(prefix) = std::env::var_os("LIBJPEG_TURBO_PREFIX") {
        // Exclusive, as in every other oracle suite of this crate: a pinned
        // release that is missing is missing, not a reason to use another.
        let pinned: PathBuf = PathBuf::from(prefix).join("bin").join(name);
        return pinned.is_file().then_some(pinned);
    }
    for directory in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/libjpeg-turbo/bin",
    ] {
        let candidate: PathBuf = Path::new(directory).join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|directory: PathBuf| directory.join(name))
            .find(|candidate: &PathBuf| candidate.is_file())
    })
}

/// `djpeg -rgb -pnm` on `jpeg`, as raw RGB rows — or `None` when the oracle
/// is not installed, which is the only skip this suite allows.
fn djpeg_rgb(jpeg: &[u8]) -> Option<Vec<u8>> {
    let djpeg: PathBuf = match find_c_tool("djpeg") {
        Some(path) => path,
        None => {
            eprintln!("SKIP: C oracle `djpeg` not found; pixel comparison is same-thread only");
            return None;
        }
    };
    let dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let input: PathBuf = dir.path().join("moved.jpg");
    let output: PathBuf = dir.path().join("moved.ppm");
    std::fs::write(&input, jpeg).expect("write the oracle input");
    let status = Command::new(&djpeg)
        .args(["-rgb", "-pnm", "-outfile"])
        .arg(&output)
        .arg(&input)
        .status()
        .expect("run djpeg");
    assert!(
        status.success(),
        "djpeg failed on the shim's output: {status}"
    );
    let ppm: Vec<u8> = std::fs::read(&output).expect("read the oracle output");
    // `P6\n<w> <h>\n255\n` then `w*h*3` bytes.
    let header_end: usize = {
        let mut newlines: usize = 0;
        let mut end: usize = 0;
        for (i, byte) in ppm.iter().enumerate() {
            if *byte == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    end = i + 1;
                    break;
                }
            }
        }
        end
    };
    let header: &str = std::str::from_utf8(&ppm[..header_end]).expect("ascii PPM header");
    assert_eq!(
        header,
        format!("P6\n{WIDTH} {HEIGHT}\n255\n"),
        "djpeg emitted an unexpected PPM header"
    );
    Some(ppm[header_end..].to_vec())
}

fn assert_pixels_match_oracle(pixels: &[u8], jpeg: &[u8], what: &str) {
    if let Some(oracle) = djpeg_rgb(jpeg) {
        let max_diff: u8 = pixels
            .iter()
            .zip(oracle.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert_eq!(
            pixels.len(),
            oracle.len(),
            "{what}: pixel count differs from djpeg"
        );
        assert_eq!(
            max_diff, 0,
            "{what}: pixels differ from djpeg (max diff {max_diff})"
        );
    }
}

// ---------- 1 + 2: ownership transfer, both directions, with release ----------

/// Create on thread A, decode and destroy on thread B. The pixels must be the
/// ones a single-threaded lifecycle produces, and thread B's destroy must
/// release what thread A's create allocated.
#[test]
#[cfg_attr(miri, ignore = "needs the SIMD encoder and a spawned djpeg")]
fn decompress_created_on_one_thread_runs_and_is_destroyed_on_another() {
    let _guard: MutexGuard<'static, ()> = serial();
    let jpeg: Vec<u8> = fixture_jpeg();

    // Reference: the whole lifecycle on this thread.
    let mut reference_err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
    let mut reference_cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let expected: Vec<u8> = unsafe {
        create_decompress(&mut *reference_cinfo, &mut *reference_err);
        let pixels: Vec<u8> = decode_with(&mut *reference_cinfo, &jpeg, |_| {});
        jpeg_destroy_decompress(&mut *reference_cinfo as *mut JpegDecompressPublic as *mut c_void);
        pixels
    };
    assert_no_error(&reference_err, "same-thread lifecycle");

    let live_before: usize = live_decompress_private_count_for_tests();

    let mut err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: SendPtr<JpegDecompressPublic> = SendPtr(&mut *cinfo);
    let err_ptr: SendPtr<TrapErrorMgr> = SendPtr(&mut *err);

    // Thread A only creates. Joining it before thread B starts is the
    // application-enforced hand-off upstream's contract requires.
    let creator: thread::JoinHandle<()> = thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegDecompressPublic> = cinfo_ptr;
        let err_ptr: SendPtr<TrapErrorMgr> = err_ptr;
        unsafe { create_decompress(cinfo_ptr.0, err_ptr.0) };
    });
    creator.join().expect("thread A");
    assert_eq!(
        live_decompress_private_count_for_tests(),
        live_before + 1,
        "thread A's create registered one private state"
    );

    let jpeg_for_b: Vec<u8> = jpeg.clone();
    let consumer: thread::JoinHandle<Vec<u8>> = thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegDecompressPublic> = cinfo_ptr;
        unsafe {
            let pixels: Vec<u8> = decode_with(cinfo_ptr.0, &jpeg_for_b, |_| {});
            jpeg_destroy_decompress(cinfo_ptr.0 as *mut c_void);
            pixels
        }
    });
    let moved: Vec<u8> = consumer.join().expect("thread B");
    assert_no_error(&err, "moved lifecycle");

    assert_eq!(
        moved, expected,
        "a cinfo moved to another thread must decode exactly what it decodes at home"
    );
    assert_pixels_match_oracle(&moved, &jpeg, "decode on the receiving thread");
    assert_eq!(
        live_decompress_private_count_for_tests(),
        live_before,
        "thread B's destroy must release the private state thread A created"
    );
}

/// The compress side has kept its private state behind `cinfo->master` since
/// it was written and never had the constraint; this pins that it stays that
/// way, with the same release check.
#[test]
#[cfg_attr(miri, ignore = "needs the SIMD encoder and a spawned djpeg")]
fn compress_created_on_one_thread_runs_and_is_destroyed_on_another() {
    let _guard: MutexGuard<'static, ()> = serial();
    let pixels: Vec<u8> = source_pixels();

    let mut reference_err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
    let mut reference_cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let expected: Vec<u8> = unsafe {
        create_compress(&mut *reference_cinfo, &mut *reference_err);
        let jpeg: Vec<u8> = encode_with(&mut *reference_cinfo, &pixels);
        jpeg_destroy_compress(&mut *reference_cinfo as *mut JpegCompressPublic as *mut c_void);
        jpeg
    };
    assert_no_error(&reference_err, "same-thread compress lifecycle");

    let live_before: usize = live_compress_private_count_for_tests();

    let mut err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: SendPtr<JpegCompressPublic> = SendPtr(&mut *cinfo);
    let err_ptr: SendPtr<TrapErrorMgr> = SendPtr(&mut *err);

    thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegCompressPublic> = cinfo_ptr;
        let err_ptr: SendPtr<TrapErrorMgr> = err_ptr;
        unsafe { create_compress(cinfo_ptr.0, err_ptr.0) };
    })
    .join()
    .expect("thread A");
    assert_eq!(live_compress_private_count_for_tests(), live_before + 1);

    let pixels_for_b: Vec<u8> = pixels.clone();
    let moved: Vec<u8> = thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegCompressPublic> = cinfo_ptr;
        unsafe {
            let jpeg: Vec<u8> = encode_with(cinfo_ptr.0, &pixels_for_b);
            jpeg_destroy_compress(cinfo_ptr.0 as *mut c_void);
            jpeg
        }
    })
    .join()
    .expect("thread B");
    assert_no_error(&err, "moved compress lifecycle");

    assert_eq!(
        moved, expected,
        "a compressor moved to another thread must emit byte-identical output"
    );
    // The stream is real JPEG as far as the C oracle is concerned.
    if let Some(oracle) = djpeg_rgb(&moved) {
        assert_eq!(oracle.len(), WIDTH * HEIGHT * 3);
    }
    assert_eq!(
        live_compress_private_count_for_tests(),
        live_before,
        "thread B's destroy must release the private state thread A created"
    );
}

// ---------- 3: the same address, twice ----------

/// Destroy on another thread, then create again in the **same** allocation,
/// without clearing it first — the C reuse pattern. The second lifecycle must
/// not see the first one's `jpeg_save_markers` request, and the live count
/// must show the cross-thread destroy actually released lifecycle 1: that
/// count is what discriminates the old design (its map missed the destroy and
/// kept the entry alive on the creating thread), while the marker check pins
/// the invariant that a re-create starts from fresh state whatever sat in the
/// allocation before.
#[test]
#[cfg_attr(miri, ignore = "needs the SIMD encoder and a spawned djpeg")]
fn reallocated_cinfo_at_the_same_address_does_not_inherit_the_old_state() {
    let _guard: MutexGuard<'static, ()> = serial();
    let jpeg: Vec<u8> = fixture_jpeg_with_app1();
    let live_before: usize = live_decompress_private_count_for_tests();

    let mut err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: SendPtr<JpegDecompressPublic> = SendPtr(&mut *cinfo);
    let err_ptr: SendPtr<TrapErrorMgr> = SendPtr(&mut *err);
    let address: usize = cinfo_ptr.0 as usize;

    // Lifecycle 1, here: ask for APP1 and prove the request is honoured —
    // otherwise lifecycle 2's null list would prove nothing.
    unsafe {
        create_decompress(cinfo_ptr.0, err_ptr.0);
        jpeg_save_markers(cinfo_ptr.0 as *mut c_void, JPEG_APP1, 0xFFFF);
        let _pixels: Vec<u8> = decode_with(cinfo_ptr.0, &jpeg, |c: &JpegDecompressPublic| {
            let saved = c.marker_list;
            assert!(
                !saved.is_null(),
                "control: lifecycle 1 must have saved the APP1 marker"
            );
            assert_eq!((*saved).marker, 0xE1, "control: the saved marker is APP1");
        });
    }
    assert_no_error(&err, "lifecycle 1");

    // Destroyed elsewhere.
    thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegDecompressPublic> = cinfo_ptr;
        unsafe { jpeg_destroy_decompress(cinfo_ptr.0 as *mut c_void) };
    })
    .join()
    .expect("destroying thread");
    assert_eq!(
        live_decompress_private_count_for_tests(),
        live_before,
        "the cross-thread destroy released lifecycle 1"
    );

    // Lifecycle 2, same bytes, same address, no marker request. Not zeroed:
    // `jpeg_CreateDecompress` keeps `err` and initialises everything else,
    // as upstream's does over a struct that has just been destroyed.
    assert_eq!(cinfo_ptr.0 as usize, address, "the object did not move");
    unsafe {
        jpeg_CreateDecompress(
            cinfo_ptr.0 as *mut c_void,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        assert!(
            !(*cinfo_ptr.0).mem.is_null(),
            "re-create on the destroyed object"
        );
        let _pixels: Vec<u8> = decode_with(cinfo_ptr.0, &jpeg, |c: &JpegDecompressPublic| {
            assert!(
                c.marker_list.is_null(),
                "lifecycle 2 at the same address surfaced lifecycle 1's marker_save state"
            );
        });
        jpeg_destroy_decompress(cinfo_ptr.0 as *mut c_void);
    }
    assert_no_error(&err, "lifecycle 2");
    assert_eq!(live_decompress_private_count_for_tests(), live_before);
}

// ---------- 4: distinct instances, concurrently ----------

/// Eight threads, eight `cinfo`s, no synchronisation between them. This is
/// the concurrency upstream supports and the shape a shared private-state
/// table would have serialised. One `cinfo` on two threads at once is a
/// data race on both libraries and is documented, not tested.
#[test]
#[cfg_attr(miri, ignore = "needs the SIMD encoder and a spawned djpeg")]
fn distinct_cinfos_decode_concurrently_on_separate_threads() {
    let _guard: MutexGuard<'static, ()> = serial();
    let jpeg: Vec<u8> = fixture_jpeg();
    let expected: Vec<u8> = {
        let mut err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
        let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
        let pixels: Vec<u8> = unsafe {
            create_decompress(&mut *cinfo, &mut *err);
            let pixels: Vec<u8> = decode_with(&mut *cinfo, &jpeg, |_| {});
            jpeg_destroy_decompress(&mut *cinfo as *mut JpegDecompressPublic as *mut c_void);
            pixels
        };
        assert_no_error(&err, "reference");
        pixels
    };
    let live_before: usize = live_decompress_private_count_for_tests();

    const THREADS: usize = 8;
    const ROUNDS: usize = 4;
    let (tx, rx) = mpsc::channel::<(usize, Vec<u8>)>();
    let workers: Vec<thread::JoinHandle<()>> = (0..THREADS)
        .map(|index: usize| {
            let jpeg: Vec<u8> = jpeg.clone();
            let tx: mpsc::Sender<(usize, Vec<u8>)> = tx.clone();
            thread::spawn(move || {
                let mut err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
                let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
                for _ in 0..ROUNDS {
                    let pixels: Vec<u8> = unsafe {
                        create_decompress(&mut *cinfo, &mut *err);
                        let pixels: Vec<u8> = decode_with(&mut *cinfo, &jpeg, |_| {});
                        jpeg_destroy_decompress(
                            &mut *cinfo as *mut JpegDecompressPublic as *mut c_void,
                        );
                        pixels
                    };
                    assert_no_error(&err, "worker");
                    tx.send((index, pixels)).expect("collector alive");
                }
            })
        })
        .collect();
    drop(tx);
    let mut received: usize = 0;
    for (index, pixels) in rx {
        assert_eq!(pixels, expected, "worker {index} decoded something else");
        received += 1;
    }
    for worker in workers {
        worker.join().expect("worker thread");
    }
    assert_eq!(received, THREADS * ROUNDS);
    assert_eq!(
        live_decompress_private_count_for_tests(),
        live_before,
        "every worker released its own state"
    );
}

// ---------- the sequence under Miri ----------

/// The transfer sequence under Miri, on a fixture small enough to stay on
/// the scalar paths — Miri cannot interpret vendor SIMD intrinsics, and the
/// crate's own encoder reaches them, so the fixture is a file.
///
/// This is the test that would have caught the first draft of P4-132: its
/// lookup read `mem` and `master` through the raw `cinfo` while every entry
/// point held a `&mut` to the same struct, which Stacked Borrows rejects on
/// the first reborrow after create. `.github/workflows/ci.yml` runs this
/// binary under Miri for that reason.
#[test]
fn moved_decode_sequence_is_sound_under_miri() {
    let _guard: MutexGuard<'static, ()> = serial();
    let jpeg: &'static [u8] = include_bytes!("../../../tests/fixtures/cjpeg_1x1_422.jpg");
    let live_before: usize = live_decompress_private_count_for_tests();

    let mut err: Box<TrapErrorMgr> = TrapErrorMgr::boxed();
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: SendPtr<JpegDecompressPublic> = SendPtr(&mut *cinfo);
    let err_ptr: SendPtr<TrapErrorMgr> = SendPtr(&mut *err);

    thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegDecompressPublic> = cinfo_ptr;
        let err_ptr: SendPtr<TrapErrorMgr> = err_ptr;
        unsafe { create_decompress(cinfo_ptr.0, err_ptr.0) };
    })
    .join()
    .expect("thread A");

    let rows: usize = thread::spawn(move || {
        let cinfo_ptr: SendPtr<JpegDecompressPublic> = cinfo_ptr;
        let cv: *mut c_void = cinfo_ptr.0 as *mut c_void;
        unsafe {
            jpeg_save_markers(cv, JPEG_APP1, 0xFFFF);
            jpeg_mem_src(cv, jpeg.as_ptr(), jpeg.len() as c_ulong);
            assert_eq!(jpeg_read_header(cv, 1), JPEG_HEADER_OK);
            assert_eq!(jpeg_start_decompress(cv), 1);
            let width: usize = (*cinfo_ptr.0).output_width as usize;
            let height: usize = (*cinfo_ptr.0).output_height as usize;
            let components: usize = (*cinfo_ptr.0).output_components as usize;
            assert_eq!((width, height, components), (1, 1, 3));
            let mut pixels: Vec<u8> = vec![0u8; width * components * height];
            let mut rows: usize = 0;
            loop {
                let row: usize = (*cinfo_ptr.0).output_scanline as usize;
                if row >= height {
                    break;
                }
                let mut row_ptr: *mut u8 = pixels.as_mut_ptr().add(row * width * components);
                rows += jpeg_read_scanlines(cv, &mut row_ptr, 1) as usize;
            }
            assert_eq!(jpeg_finish_decompress(cv), 1);
            jpeg_destroy_decompress(cv);
            rows
        }
    })
    .join()
    .expect("thread B");
    assert_no_error(&err, "moved lifecycle under Miri");
    assert_eq!(rows, 1);
    assert_eq!(live_decompress_private_count_for_tests(), live_before);
}
