//! P4-104 (#468): a `jpeg_start_decompress` that suspends must not leave the
//! caller in `DSTATE_READY`.
//!
//! `jpeg_consume_input` refuses to advance from READY — *"Can't advance past
//! first SOS until start_decompress is called"* (`jdapimin.c`). That guard is
//! upstream-correct, but it is only safe because upstream leaves READY
//! **before** anything that can suspend: `jdapistd.c:65` publishes
//! `DSTATE_PRELOAD` and only then runs the absorb loop that may return FALSE.
//! `jpeg_consume_input` keeps draining from PRELOAD (`jdapimin.c:346-353`).
//!
//! Skip that transition and a legitimate sequence deadlocks: a custom source
//! reaches its first SOS, `jpeg_start_decompress` returns FALSE part-way
//! through the body, the application feeds more bytes and resumes by polling
//! `jpeg_consume_input` — which now answers `JPEG_REACHED_SOS` forever without
//! reading a byte. No amount of new data helps. This test pins both halves:
//! the state a suspended startup leaves behind, and that polling from it makes
//! progress.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_consume_input, jpeg_destroy_decompress, jpeg_input_complete,
    jpeg_mem_src, jpeg_read_coefficients, jpeg_start_decompress, jpeg_std_error,
    JpegDecompressPublic, JpegErrorMgr, JpegSourceMgr,
};

const JPEG_LIB_VERSION: c_int = 80;
/// `jpegint.h`, cross-checked by `every_state_constant_matches_upstream_jpegint_h`.
const DSTATE_READY: c_int = 202;
const DSTATE_PRELOAD: c_int = 203;
const DSTATE_SCANNING: c_int = 205;
const DSTATE_STOPPING: c_int = 210;
/// `jpeglib.h` return codes.
const JPEG_SUSPENDED: c_int = 0;
const JPEG_REACHED_SOS: c_int = 1;
const JPEG_REACHED_EOI: c_int = 2;

/// Bytes handed over per successful `fill_input_buffer`. Small enough that a
/// single scan spans several callbacks, which is the shape a real network
/// source has.
const CHUNK: usize = 128;

/// A classic suspending source: it serves `data[..limit]` in `CHUNK`-sized
/// pieces and returns FALSE — libjpeg's "suspend" — once it has served
/// everything it is currently allowed to. Raising `limit` models new bytes
/// arriving.
///
/// `pub_mgr` is first so a `*mut ChunkSource` and the `*mut JpegSourceMgr`
/// stored in `cinfo.src` are the same address, the same trick every C source
/// manager in libjpeg uses.
#[repr(C)]
struct ChunkSource {
    pub_mgr: JpegSourceMgr,
    data: Vec<u8>,
    served: usize,
    limit: usize,
}

unsafe extern "C" fn init_source(_cinfo: *mut c_void) {}

/// Serve the next chunk, or suspend. Never touches `cinfo` beyond reading
/// `src`, and holds no Rust reference across the boundary.
unsafe extern "C" fn fill_input_buffer(cinfo: *mut c_void) -> c_int {
    // SAFETY: the shim only calls this with the `cinfo` it was handed, whose
    // `src` this test set to a live `ChunkSource` that outlives the decode.
    let src: *mut ChunkSource =
        unsafe { (*(cinfo as *mut JpegDecompressPublic)).src } as *mut ChunkSource;
    debug_assert!(!src.is_null());

    let served: usize = unsafe { (*src).served };
    let limit: usize = unsafe { (*src).limit };
    if served >= limit {
        // Suspension: leave `next_input_byte` / `bytes_in_buffer` untouched,
        // per `jpeglib.h`'s contract for a suspending data source.
        return 0;
    }
    let end: usize = (served + CHUNK).min(limit);
    // SAFETY: `data` is owned by the same allocation and outlives the decode,
    // so the pointer stays valid after this returns — the retained-pointer
    // contract a source manager must satisfy.
    unsafe {
        let base: *const u8 = (*src).data.as_ptr();
        (*src).pub_mgr.next_input_byte = base.add(served);
        (*src).pub_mgr.bytes_in_buffer = end - served;
        (*src).served = end;
    }
    1
}

unsafe extern "C" fn skip_input_data(cinfo: *mut c_void, num_bytes: std::ffi::c_long) {
    if num_bytes <= 0 {
        return;
    }
    // SAFETY: as in `fill_input_buffer`.
    let src: *mut ChunkSource =
        unsafe { (*(cinfo as *mut JpegDecompressPublic)).src } as *mut ChunkSource;
    let mut remaining: usize = num_bytes as usize;
    unsafe {
        let in_buffer: usize = (*src).pub_mgr.bytes_in_buffer;
        let step: usize = remaining.min(in_buffer);
        (*src).pub_mgr.next_input_byte = (*src).pub_mgr.next_input_byte.add(step);
        (*src).pub_mgr.bytes_in_buffer = in_buffer - step;
        remaining -= step;
        (*src).served = ((*src).served + remaining).min((*src).limit);
    }
}

unsafe extern "C" fn term_source(_cinfo: *mut c_void) {}

fn progressive_fixture() -> Vec<u8> {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    // 96×96 so the entropy-coded body is comfortably larger than the header,
    // and progressive so the datastream genuinely has multiple scans — the
    // case upstream absorbs inside `jpeg_start_decompress`.
    let pixels: Vec<u8> = (0..96 * 96 * 3).map(|i| (i % 251) as u8).collect();
    Encoder::new(&pixels, 96, 96, PixelFormat::Rgb)
        .quality(85)
        .progressive(true)
        .encode()
        .expect("encode fixture")
}

/// Offset just past the first SOS header, so a source capped here has
/// delivered a complete header and an incomplete body.
fn first_sos_end(jpeg: &[u8]) -> usize {
    let sos: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("fixture must contain an SOS marker");
    // 2 marker bytes + the 2-byte segment length, then the segment itself.
    let len: usize = usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
    let end: usize = sos + 2 + len;
    assert!(
        end < jpeg.len(),
        "fixture body must extend past the first SOS"
    );
    end
}

#[test]
fn suspended_startup_leaves_preload_and_consume_input_still_drains() {
    let jpeg: Vec<u8> = progressive_fixture();
    let header_only: usize = first_sos_end(&jpeg);
    let total: usize = jpeg.len();

    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let mut source: Box<ChunkSource> = Box::new(ChunkSource {
        // SAFETY: `JpegSourceMgr` is a plain `#[repr(C)]` struct of pointers,
        // sizes and `Option<fn>`s; all-zero is a valid (null / `None`) value
        // for every field, and each one is assigned below before use.
        pub_mgr: unsafe { std::mem::zeroed() },
        data: jpeg,
        served: 0,
        limit: header_only,
    });
    source.pub_mgr.init_source = Some(init_source);
    source.pub_mgr.fill_input_buffer = Some(fill_input_buffer);
    source.pub_mgr.skip_input_data = Some(skip_input_data);
    source.pub_mgr.term_source = Some(term_source);

    // SAFETY: every pointer below refers to a live allocation owned by this
    // function, and each outlives the last call that can observe it.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        cinfo.src = &mut source.pub_mgr as *mut JpegSourceMgr;

        // Reach the first SOS. A suspending source may need several polls.
        let mut reached_sos: bool = false;
        for _ in 0..64 {
            if jpeg_consume_input(p) == JPEG_REACHED_SOS {
                reached_sos = true;
                break;
            }
        }
        assert!(reached_sos, "consume_input never reached the first SOS");
        assert_eq!(
            cinfo.global_state, DSTATE_READY,
            "reaching SOS must land on READY (jdapimin.c)"
        );

        // The body is still short, so startup must report suspension.
        assert_eq!(
            jpeg_start_decompress(p),
            0,
            "start_decompress must suspend while the body is incomplete"
        );
        assert_eq!(
            cinfo.global_state, DSTATE_PRELOAD,
            "a suspended startup must leave PRELOAD, not READY — from READY, \
             jpeg_consume_input refuses to consume and the caller can never \
             make progress (jdapistd.c:65)"
        );

        // More bytes arrive; the application resumes by polling.
        source.limit = total;
        let mut returns: Vec<c_int> = Vec::new();
        for _ in 0..512 {
            let rc: c_int = jpeg_consume_input(p);
            returns.push(rc);
            if rc == JPEG_REACHED_EOI {
                break;
            }
            assert_ne!(
                rc, JPEG_SUSPENDED,
                "source has the whole file; suspension here means the drain \
                 is not reading it: {returns:?}"
            );
        }
        assert_eq!(
            returns.last().copied(),
            Some(JPEG_REACHED_EOI),
            "polling from a suspended startup must drain to EOI; it instead \
             repeated {:?}",
            returns.last()
        );

        // Reaching EOI does not by itself make the decoder scanline-ready.
        // Upstream returns EOI from PRELOAD without touching the state
        // (`jdapimin.c`); the caller still owes the `jpeg_start_decompress`
        // retry that runs `output_pass_setup`. Publishing SCANNING here would
        // advertise a pass that has not run.
        assert_eq!(
            cinfo.global_state, DSTATE_PRELOAD,
            "EOI from PRELOAD must leave the state alone until \
             jpeg_start_decompress is retried"
        );
        // The input *is* complete, though — upstream's `eoi_reached` is TRUE —
        // so the standard polling idiom terminates rather than spinning.
        assert_eq!(
            jpeg_input_complete(p),
            1,
            "the whole datastream has been consumed, whatever the state says"
        );

        // And the retry the contract asks for now succeeds and publishes the
        // scanline-ready state.
        assert_eq!(
            jpeg_start_decompress(p),
            1,
            "startup must succeed once the body is fully buffered"
        );
        assert_eq!(
            cinfo.global_state, DSTATE_SCANNING,
            "the completed startup pass is what publishes SCANNING"
        );

        jpeg_destroy_decompress(p);
    }
}

/// The same hazard on the transcode path. `jpeg_read_coefficients` absorbs the
/// whole datastream, so it too must leave READY — upstream walks
/// READY → `DSTATE_RDCOEFS` → `DSTATE_STOPPING` (`jdtrans.c:54-82`), the last
/// of which is what `jpeg_finish_decompress` expects to find.
///
/// A `jpegtran`-shaped caller that reaches SOS by polling `jpeg_consume_input`
/// before transcoding would otherwise finish the coefficient read still at
/// READY: `jpeg_input_complete` reports FALSE and further polling repeats
/// `REACHED_SOS` with nothing left to consume.
#[test]
fn coefficient_read_from_ready_lands_on_stopping() {
    let jpeg: Vec<u8> = progressive_fixture();

    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    // SAFETY: `err`, `cinfo` and `jpeg` are live for every call below, and
    // `jpeg` outlives the decode as `jpeg_mem_src`'s retained-pointer
    // contract requires.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);

        assert_eq!(
            jpeg_consume_input(p),
            JPEG_REACHED_SOS,
            "first poll must reach SOS"
        );
        assert_eq!(cinfo.global_state, DSTATE_READY);

        assert!(
            !jpeg_read_coefficients(p).is_null(),
            "coefficient read must succeed"
        );
        assert_eq!(
            cinfo.global_state, DSTATE_STOPPING,
            "a completed coefficient read must reach STOPPING (jdtrans.c), not \
             sit at READY where consume_input refuses to advance"
        );
        assert_eq!(
            jpeg_input_complete(p),
            1,
            "the whole datastream is in the coefficient buffer"
        );
        assert_eq!(
            jpeg_consume_input(p),
            JPEG_REACHED_EOI,
            "nothing left to consume, so polling must say so"
        );

        jpeg_destroy_decompress(p);
    }
}

// ---------------------------------------------------------------------------
// C oracle: the same two sequences, run against stock libjpeg.
// ---------------------------------------------------------------------------

mod helpers;

fn temp_fixture(label: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path: std::path::PathBuf = std::env::temp_dir().join(format!(
        "libjpeg_turbo_rs_p4104_{}_{label}.jpg",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write oracle fixture");
    path
}

/// Run the suspend/resume sequence and record it in the oracle's line format.
fn preload_trace(jpeg: &[u8]) -> String {
    let header_only: usize = first_sos_end(jpeg);
    let total: usize = jpeg.len();

    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let mut source: Box<ChunkSource> = Box::new(ChunkSource {
        // SAFETY: as in the test above — all-zero is a valid value for every
        // field, and the callbacks are assigned before any call can run.
        pub_mgr: unsafe { std::mem::zeroed() },
        data: jpeg.to_vec(),
        served: 0,
        limit: header_only,
    });
    source.pub_mgr.init_source = Some(init_source);
    source.pub_mgr.fill_input_buffer = Some(fill_input_buffer);
    source.pub_mgr.skip_input_data = Some(skip_input_data);
    source.pub_mgr.term_source = Some(term_source);

    let mut trace: String = String::new();
    // SAFETY: every pointer refers to a live allocation owned here that
    // outlives the last call able to observe it.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        trace.push_str(&format!("create -1 {}\n", cinfo.global_state));
        cinfo.src = &mut source.pub_mgr as *mut JpegSourceMgr;

        let mut rc: c_int = JPEG_SUSPENDED;
        for _ in 0..64 {
            rc = jpeg_consume_input(p);
            if rc == JPEG_REACHED_SOS {
                break;
            }
        }
        trace.push_str(&format!("sos {rc} {}\n", cinfo.global_state));

        let ok: c_int = jpeg_start_decompress(p);
        trace.push_str(&format!("startup {ok} {}\n", cinfo.global_state));

        source.limit = total;
        for _ in 0..512 {
            rc = jpeg_consume_input(p);
            if rc == JPEG_REACHED_EOI {
                break;
            }
        }
        trace.push_str(&format!("drained {rc} {}\n", cinfo.global_state));
        trace.push_str(&format!(
            "complete {} {}\n",
            jpeg_input_complete(p),
            cinfo.global_state
        ));

        let ok: c_int = jpeg_start_decompress(p);
        trace.push_str(&format!("retry {ok} {}\n", cinfo.global_state));

        jpeg_destroy_decompress(p);
    }
    trace
}

/// Run the transcode sequence and record it in the oracle's line format.
fn coefs_trace(jpeg: &[u8]) -> String {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    let mut trace: String = String::new();
    // SAFETY: as above; `jpeg` outlives the decode, which `jpeg_mem_src`'s
    // retained-pointer contract requires.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = errp;
        let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
        jpeg_CreateDecompress(
            p,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        trace.push_str(&format!("create -1 {}\n", cinfo.global_state));
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);

        let rc: c_int = jpeg_consume_input(p);
        trace.push_str(&format!("poll {rc} {}\n", cinfo.global_state));

        let coefs: *mut c_void = jpeg_read_coefficients(p);
        trace.push_str(&format!(
            "coefs {} {}\n",
            c_int::from(!coefs.is_null()),
            cinfo.global_state
        ));
        trace.push_str(&format!(
            "complete {} {}\n",
            jpeg_input_complete(p),
            cinfo.global_state
        ));
        let rc: c_int = jpeg_consume_input(p);
        trace.push_str(&format!("poll {rc} {}\n", cinfo.global_state));

        jpeg_destroy_decompress(p);
    }
    trace
}

/// Both sequences, compared step by step against stock libjpeg.
///
/// The assertions in the tests above encode upstream's behaviour from a
/// reading of `jdapistd.c` / `jdtrans.c`. This one makes upstream produce the
/// trace itself, so a misreading fails here instead of being ratified by a
/// green suite.
#[test]
fn state_traces_match_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("consume_input_states_oracle") else {
        eprintln!(
            "SKIP: no stock libjpeg development install found; the C oracle for \
             P4-104's PRELOAD and coefficient traces cannot be built. Set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };

    let jpeg: Vec<u8> = progressive_fixture();
    let path: std::path::PathBuf = temp_fixture("suspend", &jpeg);
    let file: &str = path.to_str().expect("utf-8 temp path");

    let c_preload: String = helpers::run_oracle(&oracle, &["preload", file]);
    let c_coefs: String = helpers::run_oracle(&oracle, &["coefs", file]);
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        preload_trace(&jpeg),
        c_preload,
        "suspend/resume trace diverges from stock libjpeg"
    );
    assert_eq!(
        coefs_trace(&jpeg),
        c_coefs,
        "coefficient-read trace diverges from stock libjpeg"
    );
}
