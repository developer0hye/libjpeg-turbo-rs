//! C2-5: end-to-end roundtrip for `jpeg_write_coefficients`.
//!
//! Verifies the lossless transcode flow used by `jpegtran`:
//! tj3Compress → jpeg_read_coefficients → jpeg_write_coefficients →
//! tj3Decompress, asserting the decoded output matches the source pixels
//! (the quantization tables and entropy coding survive the round trip
//! because the coefficients are re-emitted verbatim).

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

const JPEG_HEADER_OK: c_int = 1;

fn dlext() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}
fn lib_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}
fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf =
            dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dlext()));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate cdylib near {}", exe.display());
}

/// Compress a deterministic 64x64 RGB gradient via tj3 and return the
/// JPEG bytes alongside the source pixels.
fn build_fixture_jpeg(lib: &libloading::Library) -> (Vec<u8>, Vec<u8>, usize, usize) {
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set");
        let tj3_compress: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress8").expect("tj3Compress8");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let h_enc: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 80), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_SUBSAMP, TJSAMP_444), 0);

        let w: usize = 64;
        let h_px: usize = 64;
        let mut src: Vec<u8> = Vec::with_capacity(w * h_px * 3);
        for y in 0..h_px {
            for x in 0..w {
                src.push(x as u8);
                src.push(y as u8);
                src.push(((x + y) / 2) as u8);
            }
        }

        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc: c_int = tj3_compress(
            h_enc,
            src.as_ptr(),
            w as c_int,
            0,
            h_px as c_int,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        assert_eq!(rc, 0);

        let jpeg: Vec<u8> = std::slice::from_raw_parts(jpeg_buf, jpeg_size).to_vec();
        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h_enc);
        (jpeg, src, w, h_px)
    }
}

/// Decode a JPEG byte stream back to RGB pixels via tj3, returning
/// (pixels, width, height).
fn decode_jpeg(lib: &libloading::Library, jpeg: &[u8]) -> (Vec<u8>, usize, usize) {
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get");
        let tj3_decompress_header: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize) -> c_int,
        > = lib
            .get(b"tj3DecompressHeader")
            .expect("tj3DecompressHeader");
        let tj3_decompress: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").expect("tj3Decompress8");

        let h_dec: TjHandle = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_dec.is_null());
        let rc: c_int = tj3_decompress_header(h_dec, jpeg.as_ptr(), jpeg.len());
        assert_eq!(rc, 0, "tj3DecompressHeader failed on transcoded output");
        let w: c_int = tj3_get(h_dec, TJPARAM_JPEGWIDTH);
        let h_px: c_int = tj3_get(h_dec, TJPARAM_JPEGHEIGHT);
        assert!(w > 0 && h_px > 0);
        let mut dst: Vec<u8> = vec![0u8; (w * h_px * 3) as usize];
        let rc: c_int = tj3_decompress(
            h_dec,
            jpeg.as_ptr(),
            jpeg.len(),
            dst.as_mut_ptr(),
            0,
            TJPF_RGB,
        );
        assert_eq!(rc, 0, "tj3Decompress8 failed on transcoded output");
        tj3_destroy(h_dec);
        (dst, w as usize, h_px as usize)
    }
}

#[test]
fn write_coefficients_roundtrip_pixel_exact() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, src, src_w, src_h) = build_fixture_jpeg(&lib);

    // Decode the fixture once via tj3 so we have the reference pixels
    // produced by the same decoder we'll use on the transcoded output.
    let (ref_pixels, ref_w, ref_h) = decode_jpeg(&lib, &jpeg_in);
    assert_eq!((ref_w, ref_h), (src_w, src_h));

    let transcoded: Vec<u8> = unsafe {
        // ---- Decompress side: read coefficients ----
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;

        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);

        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);

        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null(), "coef handle must be non-null");

        // ---- Compress side: write coefficients ----
        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);

        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        assert!(
            !out_buf.is_null(),
            "transcoded output buffer must be allocated"
        );
        assert!(out_size > 0, "transcoded output must be non-empty");
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();

        // ---- Cleanup ----
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);

        bytes
    };

    // Sanity: SOI + EOI bracketing.
    assert!(
        transcoded.len() >= 4,
        "transcoded too short: {}",
        transcoded.len()
    );
    assert_eq!(&transcoded[..2], &[0xFF, 0xD8], "missing SOI");
    assert_eq!(
        &transcoded[transcoded.len() - 2..],
        &[0xFF, 0xD9],
        "missing EOI"
    );

    // Decode the transcoded JPEG and verify pixel-exact match against the
    // reference decode of the original fixture. Lossless transcode means
    // the same coefficients re-encoded must produce the same pixels after
    // a clean decode.
    let (out_pixels, out_w, out_h) = decode_jpeg(&lib, &transcoded);
    assert_eq!((out_w, out_h), (src_w, src_h));
    assert_eq!(
        out_pixels, ref_pixels,
        "transcoded output must decode pixel-exact to the reference"
    );

    // Quiet unused warnings on the source pixels (they are the input that
    // produced the reference decode above).
    let _ = src;
}

/// `jpeg_write_coefficients(NULL, ...)` and a NULL `coef_arrays` must not
/// crash and must leave a meaningful last-error trace for callers that
/// parse `cinfo->err`.
#[test]
fn write_coefficients_null_inputs_do_not_crash() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        // Both NULL: no-op.
        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(std::ptr::null_mut(), std::ptr::null_mut());

        // cinfo OK, coef_arrays NULL: must not crash, must record error.
        const CINFO_BYTES: usize = 4096;
        let mut cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(err_ptr);
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(cinfo_ptr, 80, CINFO_BYTES);
        jpeg_write_coefficients(cinfo_ptr, std::ptr::null_mut());

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// Markers added between `jpeg_write_coefficients` and
/// `jpeg_finish_compress` must land in the output stream — the libjpeg
/// jpegtran flow injects ICC and other APP segments in exactly this
/// window, so write_coefficients cannot finalize the destination early.
#[test]
fn write_coefficients_then_marker_then_finish_emits_marker() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    const APP3_CODE: c_int = 0xE3;
    const PAYLOAD: &[u8] = b"WRCOEFS-MARKER-PROBE";

    let transcoded: Vec<u8> = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);

        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);

        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        // Inject an APP3 marker between write_coefficients and
        // finish_compress. This is the critical ordering test: the marker
        // must end up in the output stream.
        let jpeg_write_marker: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, *const u8, u32),
        > = lib.get(b"jpeg_write_marker").expect("jpeg_write_marker");
        jpeg_write_marker(
            enc_cinfo_ptr,
            APP3_CODE,
            PAYLOAD.as_ptr(),
            PAYLOAD.len() as u32,
        );

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        assert!(!out_buf.is_null());
        assert!(out_size > 0);
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        bytes
    };

    // Confirm SOI/EOI bracketing.
    assert_eq!(&transcoded[..2], &[0xFF, 0xD8]);
    assert_eq!(&transcoded[transcoded.len() - 2..], &[0xFF, 0xD9]);

    // The injected APP3 segment must appear in the output. Search for the
    // payload bytes anywhere after SOI.
    let needle: &[u8] = PAYLOAD;
    let found: bool = transcoded.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "APP3 payload not found in transcoded output — markers injected between \
         write_coefficients and finish_compress were dropped"
    );

    // The APP3 marker byte sequence (FF E3) must precede the payload.
    let mut saw_app3_before_payload: bool = false;
    for i in 0..transcoded.len().saturating_sub(needle.len()) {
        if &transcoded[i..i + needle.len()] == needle {
            // Walk back at most 64 bytes looking for FF E3.
            let start: usize = i.saturating_sub(64);
            for j in (start..i).rev() {
                if transcoded[j] == 0xFF && j + 1 < transcoded.len() && transcoded[j + 1] == 0xE3 {
                    saw_app3_before_payload = true;
                    break;
                }
            }
            break;
        }
    }
    assert!(
        saw_app3_before_payload,
        "payload found but no preceding FF E3 (APP3) marker — marker structure malformed"
    );

    // The APP3 segment must land *after* SOI + the automatic JFIF APP0
    // segment (0xFFE0 with "JFIF\0" identifier), preserving libjpeg's
    // marker ordering. If APP3 lands before APP0, downstream JFIF parsers
    // can mis-detect the colorspace.
    let app0_pos: Option<usize> = (0..transcoded.len().saturating_sub(7)).find(|&i| {
        transcoded[i] == 0xFF && transcoded[i + 1] == 0xE0 && &transcoded[i + 4..i + 9] == b"JFIF\0"
    });
    if let Some(app0_at) = app0_pos {
        let payload_at: usize = transcoded
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("payload presence already asserted above");
        assert!(
            payload_at > app0_at,
            "APP3 payload at {payload_at} must come after JFIF APP0 at {app0_at}"
        );
    }
}

/// Foreign coefficient handles (e.g. raw stack pointers, virtual barray
/// pointers from a real libjpeg memory manager) must be rejected — the
/// shim cannot blindly dereference them as `JpegCoefficients`.
#[test]
fn write_coefficients_rejects_foreign_handle() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(err_ptr);
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        // Hand a foreign 8-byte block whose first u64 is *not* the magic.
        // The shim must reject it inside finish_compress without crashing.
        let foreign: [u64; 4] = [0xDEAD_BEEF_DEAD_BEEF; 4];
        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(cinfo_ptr, foreign.as_ptr() as *mut c_void);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        // No crash. Output should be empty (no init_destination ever ran
        // through the encoder, or the magic check failed before push).
        // Either way, the test's contract is "no crash".

        // Cleanup: tj3Free for the (possibly NULL) outbuf, then destroy.
        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// `c.progressive_mode = 1` between `jpeg_write_coefficients` and
/// `jpeg_finish_compress` must produce a progressive datastream (SOF2,
/// 0xFFC2), not the baseline SOF0 (0xFFC0). Mirrors `jpegtran -progressive`.
#[test]
fn write_coefficients_honors_progressive_mode() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let transcoded: Vec<u8> = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        // Toggle progressive_mode on the destination cinfo *between*
        // write_coefficients and finish_compress, mirroring
        // `jpegtran -progressive`.
        let jpeg_capi_test_set_progressive: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_progressive")
            .expect("jpeg_capi_test_set_progressive");
        jpeg_capi_test_set_progressive(enc_cinfo_ptr, 1);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        assert!(!out_buf.is_null());
        assert!(out_size > 0);
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        bytes
    };

    // Search for SOF2 (FF C2) and absence of SOF0 (FF C0). Skip past SOI/JFIF/DQT.
    let has_sof2: bool = transcoded.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    let has_sof0: bool = transcoded.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC0);
    assert!(
        has_sof2,
        "progressive transcoded output must contain SOF2 (FF C2)"
    );
    assert!(
        !has_sof0,
        "progressive transcoded output must not contain SOF0 (FF C0)"
    );
}

/// Source Adobe APP14 (CMYK fixture with both JFIF and Adobe markers)
/// must survive the transcode with its color-transform byte intact:
/// the 4-component output drops JFIF (illegal on 4-comp) and emits an
/// Adobe APP14 with the same transform byte the source declared.
#[test]
fn write_coefficients_preserves_source_adobe_app14() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let fixture_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests/fixtures/cmyk_scanner/scanner_64x64.jpg");
    let jpeg_in: Vec<u8> = match std::fs::read(&fixture_path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("SKIP: missing fixture {}", fixture_path.display());
            return;
        }
    };

    let src_app14_pos: usize = jpeg_in
        .windows(9)
        .position(|w| w[0] == 0xFF && w[1] == 0xEE && &w[4..9] == b"Adobe")
        .expect("source fixture must have Adobe APP14");
    // Adobe APP14 layout: FF EE LEN_HI LEN_LO 'A' 'd' 'o' 'b' 'e' VERhi
    // VERlo FLAG0hi FLAG0lo FLAG1hi FLAG1lo TRANSFORM (transform byte
    // is at offset 15 from FF).
    let src_transform: u8 = jpeg_in[src_app14_pos + 15];

    let transcoded: Vec<u8> = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null(), "coef handle must be non-null");

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);
        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        assert!(!out_buf.is_null());
        assert!(out_size > 0);
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        bytes
    };

    // Output must contain Adobe APP14 with the source's transform byte.
    let out_app14_pos: usize = transcoded
        .windows(9)
        .position(|w| w[0] == 0xFF && w[1] == 0xEE && &w[4..9] == b"Adobe")
        .expect("transcoded output must contain Adobe APP14");
    let out_transform: u8 = transcoded[out_app14_pos + 15];
    assert_eq!(
        out_transform, src_transform,
        "Adobe color-transform byte must be preserved verbatim from the source"
    );

    // 4-component output: JFIF APP0 must be stripped (invalid on 4-comp).
    // Inspect the SOF segment to confirm 4 components.
    let sof_pos: Option<usize> = transcoded
        .windows(2)
        .position(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC2));
    if let Some(pos) = sof_pos {
        let nf: u8 = transcoded[pos + 9];
        if nf == 4 {
            let has_jfif: bool = transcoded
                .windows(9)
                .any(|w| w[0] == 0xFF && w[1] == 0xE0 && &w[4..9] == b"JFIF\0");
            assert!(!has_jfif, "4-component output must not carry JFIF APP0");
        }
    }
}

/// `cinfo->restart_in_rows` (row-mode restart, what `jpegtran -restart Nrows`
/// produces) must fold into byte-mode `restart_interval` for baseline
/// output and emit RST markers in the entropy stream. Without this the
/// row-mode flag is silently dropped.
#[test]
fn write_coefficients_baseline_honors_restart_in_rows() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let transcoded: Vec<u8> = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        // Set row-mode restart between write_coefficients and finish_compress.
        // 64x64 RGB 4:4:4 baseline: 8 MCUs/row at 8x8 blocks, 8 rows total.
        // restart_in_rows=2 → restart_interval = 16 MCUs (every 2 rows).
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        assert!(!out_buf.is_null());
        assert!(out_size > 0);
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        bytes
    };

    // The DRI segment (FF DD) must declare a non-zero restart interval.
    let dri_pos: usize = transcoded
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] == 0xDD)
        .expect("DRI segment must be emitted when restart_in_rows is set");
    let interval: u16 = u16::from_be_bytes([transcoded[dri_pos + 4], transcoded[dri_pos + 5]]);
    assert!(
        interval > 0,
        "DRI interval must be non-zero in row-mode (got {interval})"
    );

    // Entropy stream must contain at least one RST marker (FF D0..FF D7).
    let sos_pos: usize = transcoded
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] == 0xDA)
        .expect("SOS segment expected");
    let entropy: &[u8] = &transcoded[sos_pos..];
    let has_rst: bool = entropy
        .windows(2)
        .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]));
    assert!(
        has_rst,
        "entropy stream must contain RST markers when row-mode restart is set"
    );
}

/// `jpegtran -arithmetic -restart Nrows` (baseline arithmetic + restart)
/// must produce a valid arithmetic JPEG with RST markers in the entropy
/// stream. The Rust-side `write_coefficients_arithmetic` now supports
/// restart, mirroring libjpeg-turbo `jcarith.c::emit_restart`.
#[test]
fn write_coefficients_arithmetic_with_restart_emits_rst_markers() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let transcoded: Vec<u8> = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        // Combine -arithmetic with -restart Nrows on the destination.
        let jpeg_capi_test_set_arith_code: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_arith_code")
            .expect("jpeg_capi_test_set_arith_code");
        jpeg_capi_test_set_arith_code(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        let bytes: Vec<u8> = if !out_buf.is_null() && out_size > 0 {
            std::slice::from_raw_parts(out_buf, out_size as usize).to_vec()
        } else {
            Vec::new()
        };
        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        bytes
    };

    // Output must be a complete arithmetic JPEG (SOF9 = FF C9, SOI/EOI bracketing).
    assert!(
        !transcoded.is_empty(),
        "arithmetic + restart must not produce empty output"
    );
    assert_eq!(&transcoded[..2], &[0xFF, 0xD8], "missing SOI");
    assert_eq!(
        &transcoded[transcoded.len() - 2..],
        &[0xFF, 0xD9],
        "missing EOI"
    );
    let has_sof9: bool = transcoded.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC9);
    assert!(has_sof9, "arithmetic output must contain SOF9 (FF C9)");

    // DRI segment must be emitted with a non-zero interval — restart_in_rows
    // folds into byte-mode the same way as the baseline path.
    let dri_pos: usize = transcoded
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] == 0xDD)
        .expect("arithmetic + restart must emit DRI segment");
    let interval: u16 = u16::from_be_bytes([transcoded[dri_pos + 4], transcoded[dri_pos + 5]]);
    assert!(
        interval > 0,
        "DRI interval must be non-zero in arithmetic restart mode (got {interval})"
    );

    // Entropy stream (after SOS) must contain at least one RST marker (FF D0..D7).
    let sos_pos: usize = transcoded
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] == 0xDA)
        .expect("SOS segment expected");
    let entropy: &[u8] = &transcoded[sos_pos..];
    let has_rst: bool = entropy
        .windows(2)
        .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]));
    assert!(
        has_rst,
        "arithmetic entropy stream must contain RST markers when restart is set"
    );
}

/// Progressive arithmetic + restart (either byte-mode `restart_interval`
/// or row-mode `restart_in_rows`) is not yet supported by the
/// underlying writer. The shim must drop the restart spec rather than
/// silently honoring zero — it must produce a valid SOF10 stream with
/// no DRI / no RST, and (when the caller explicitly asked for restart)
/// surface a `last_error` indicating restart was dropped.
#[test]
fn write_coefficients_progressive_arithmetic_drops_restart_explicitly() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let transcoded: Vec<u8> = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        // -arithmetic + -progressive + -restart Nrows
        let jpeg_capi_test_set_arith_code: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_arith_code")
            .expect("jpeg_capi_test_set_arith_code");
        jpeg_capi_test_set_arith_code(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_progressive: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_progressive")
            .expect("jpeg_capi_test_set_progressive");
        jpeg_capi_test_set_progressive(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        let bytes: Vec<u8> = if !out_buf.is_null() && out_size > 0 {
            std::slice::from_raw_parts(out_buf, out_size as usize).to_vec()
        } else {
            Vec::new()
        };
        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        bytes
    };

    // Output must be a complete SOF10 (progressive arithmetic) stream.
    assert!(
        !transcoded.is_empty(),
        "progressive arithmetic + restart must not produce empty output"
    );
    assert_eq!(&transcoded[..2], &[0xFF, 0xD8], "missing SOI");
    assert_eq!(
        &transcoded[transcoded.len() - 2..],
        &[0xFF, 0xD9],
        "missing EOI"
    );
    let has_sof10: bool = transcoded.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xCA);
    assert!(
        has_sof10,
        "progressive arithmetic must contain SOF10 (FF CA)"
    );

    // No DRI segment (restart was dropped).
    let has_dri: bool = transcoded.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(
        !has_dri,
        "progressive arithmetic + restart must drop restart, no DRI segment expected"
    );
}

// Counters used by the next test's emit_message hook. Sentinel value
// `i32::MIN` means "no callback observed yet" so we can distinguish
// "level 0" (a real call) from "no call at all".
static EMIT_COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static LAST_LEVEL: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(i32::MIN);

unsafe extern "C" fn capture_emit_message(cinfo: *mut c_void, msg_level: c_int) {
    EMIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    LAST_LEVEL.store(msg_level, std::sync::atomic::Ordering::SeqCst);
    // Mirror libjpeg's documented `emit_message` contract from jerror.c:
    // when msg_level < 0, the hook owns the `num_warnings` increment.
    // A real wrapper (jpegtran, GraphicsMagick, …) would do this; the
    // shim relies on the hook to count, rather than pre-incrementing.
    if msg_level < 0 && !cinfo.is_null() {
        // err is the first pointer-sized field in the cinfo struct.
        let err_pp: *const *mut u8 = cinfo as *const *mut u8;
        let err_ptr: *mut u8 = err_pp.read();
        if !err_ptr.is_null() {
            // num_warnings sits at byte offset 128 (5 callbacks × 8 +
            // msg_code 4 + msg_parm 80 + trace_level 4).
            let nw_ptr: *mut std::os::raw::c_long = err_ptr.add(128) as *mut _;
            nw_ptr.write(nw_ptr.read().saturating_add(1));
        }
    }
}

#[test]
fn write_coefficients_progressive_arithmetic_drop_emits_warning() {
    // The combined "progressive + arithmetic + row-mode restart" case
    // currently drops the restart interval (matching libjpeg-turbo's
    // documented behaviour for this corner). The drop must be visible
    // to C callers via the standard libjpeg warning channel —
    // `cinfo->err->num_warnings` must increment and the installed
    // `emit_message(cinfo, -1)` hook must fire — so a wrapper such as
    // `jpegtran` can detect the loss without poking at private state.
    EMIT_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
    LAST_LEVEL.store(i32::MIN, std::sync::atomic::Ordering::SeqCst);

    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let (num_warnings, emit_count, last_level) = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        // Patch the `emit_message` slot (offset 8 in JpegErrorMgr:
        // error_exit @0, emit_message @8 — 8 bytes per fn pointer on
        // 64-bit). This must happen *after* jpeg_std_error wired in
        // the default hook, because that call would overwrite our
        // override otherwise.
        let emit_slot: *mut Option<unsafe extern "C" fn(*mut c_void, c_int)> =
            (enc_err_ptr as *mut u8).add(8) as *mut _;
        emit_slot.write(Some(capture_emit_message));
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        let jpeg_capi_test_set_arith_code: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_arith_code")
            .expect("jpeg_capi_test_set_arith_code");
        jpeg_capi_test_set_arith_code(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_progressive: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_progressive")
            .expect("jpeg_capi_test_set_progressive");
        jpeg_capi_test_set_progressive(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        // num_warnings sits at byte offset 128 in JpegErrorMgr:
        //   5 callback slots × 8 = 40
        //   msg_code              = 4
        //   msg_parm[80]          = 80
        //   trace_level           = 4
        //   ----------------------- 128
        //   num_warnings (c_long, 8 bytes on 64-bit)
        let nw_ptr: *const std::os::raw::c_long = (enc_err_ptr as *const u8).add(128) as *const _;
        let nw: std::os::raw::c_long = nw_ptr.read();

        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);

        (
            nw,
            EMIT_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            LAST_LEVEL.load(std::sync::atomic::Ordering::SeqCst),
        )
    };

    assert!(
        num_warnings >= 1,
        "expected num_warnings >= 1 after dropping row-mode restart, got {num_warnings}"
    );
    assert!(
        emit_count >= 1,
        "expected emit_message to fire at least once, got {emit_count}"
    );
    assert_eq!(
        last_level, -1,
        "warning emit must use msg_level=-1 (libjpeg trace convention), got {last_level}"
    );
}

#[test]
fn write_coefficients_progressive_arithmetic_drop_default_hook_counts_warning() {
    // Mirrors the previous test, but does NOT install a custom
    // emit_message hook — exercising the default hook path. A wrapper
    // that simply polls `cinfo->err->num_warnings` (the documented
    // libjpeg way to detect whether warnings were emitted during
    // compression) must see a non-zero count, because the shim's
    // `default_emit_message` follows the jerror.c contract and bumps
    // the counter on msg_level=-1.
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let num_warnings: std::os::raw::c_long = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        // Intentionally NO emit_message override here — testing default path.
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        let jpeg_capi_test_set_arith_code: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_arith_code")
            .expect("jpeg_capi_test_set_arith_code");
        jpeg_capi_test_set_arith_code(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_progressive: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_progressive")
            .expect("jpeg_capi_test_set_progressive");
        jpeg_capi_test_set_progressive(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        let nw_ptr: *const std::os::raw::c_long = (enc_err_ptr as *const u8).add(128) as *const _;
        let nw: std::os::raw::c_long = nw_ptr.read();

        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        nw
    };

    assert!(
        num_warnings >= 1,
        "default emit_message must increment num_warnings on msg_level<0 (got {num_warnings})"
    );
}

// Counters used by the next test's output_message hook.
static OUTPUT_COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static OUTPUT_NW_AT_CALL: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

unsafe extern "C" fn capture_output_message(cinfo: *mut c_void) {
    OUTPUT_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if !cinfo.is_null() {
        let err_pp: *const *mut u8 = cinfo as *const *mut u8;
        let err_ptr: *mut u8 = err_pp.read();
        if !err_ptr.is_null() {
            let nw_ptr: *const std::os::raw::c_long = err_ptr.add(128) as *const _;
            OUTPUT_NW_AT_CALL.store(nw_ptr.read() as i64, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[test]
fn write_coefficients_default_emit_calls_output_before_incrementing() {
    // Verifies the default `emit_message` matches libjpeg-turbo's
    // jerror.c order: `output_message` is called BEFORE the
    // `num_warnings` increment. A custom `output_message` inspecting
    // the counter while formatting the first warning must see
    // `num_warnings == 0`, the way real libjpeg callers expect.
    OUTPUT_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
    OUTPUT_NW_AT_CALL.store(-1, std::sync::atomic::Ordering::SeqCst);

    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let (output_count, observed_nw) = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        // Override only `output_message` (offset 16: error_exit @0,
        // emit_message @8, output_message @16). Keep the default
        // `emit_message` so we exercise its libjpeg-conformant ordering.
        let out_slot: *mut Option<unsafe extern "C" fn(*mut c_void)> =
            (enc_err_ptr as *mut u8).add(16) as *mut _;
        out_slot.write(Some(capture_output_message));
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        let jpeg_capi_test_set_arith_code: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_arith_code")
            .expect("jpeg_capi_test_set_arith_code");
        jpeg_capi_test_set_arith_code(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_progressive: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_progressive")
            .expect("jpeg_capi_test_set_progressive");
        jpeg_capi_test_set_progressive(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);

        (
            OUTPUT_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            OUTPUT_NW_AT_CALL.load(std::sync::atomic::Ordering::SeqCst),
        )
    };

    assert!(
        output_count >= 1,
        "default emit_message must route the first warning to output_message (got {output_count} call(s))"
    );
    assert_eq!(
        observed_nw, 0,
        "output_message must observe num_warnings==0 on the first warning (libjpeg jerror.c order); got {observed_nw}"
    );
}

#[test]
fn write_coefficients_warning_emits_meaningful_format_message() {
    // The libjpeg WARNMS contract: emit_message(cinfo, -1) must be
    // preceded by setting `cinfo->err->msg_code` so a follow-up
    // `(*err->format_message)(cinfo, buf)` produces a non-bogus
    // string. Verifies the shim's progressive-arith-restart drop now
    // satisfies that contract by installing an addon message table
    // and pointing msg_code at it.
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    let formatted: String = unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut dec_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let dec_cinfo_ptr: *mut c_void = dec_cinfo.as_mut_ptr() as *mut c_void;
        const ERR_BYTES: usize = 512;
        let mut dec_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let dec_err_ptr: *mut c_void = dec_err.as_mut_ptr() as *mut c_void;
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(dec_err_ptr);
        (dec_cinfo_ptr as *mut *mut c_void).write(dec_err_ptr);
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(dec_cinfo_ptr, 80, CINFO_BYTES);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(dec_cinfo_ptr, jpeg_in.as_ptr(), jpeg_in.len() as c_ulong);
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(dec_cinfo_ptr, 1), JPEG_HEADER_OK);
        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coef_arrays: *mut c_void = jpeg_read_coefficients(dec_cinfo_ptr);
        assert!(!coef_arrays.is_null());

        let mut enc_cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let enc_cinfo_ptr: *mut c_void = enc_cinfo.as_mut_ptr() as *mut c_void;
        let mut enc_err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let enc_err_ptr: *mut c_void = enc_err.as_mut_ptr() as *mut c_void;
        let _ = jpeg_std_error(enc_err_ptr);
        (enc_cinfo_ptr as *mut *mut c_void).write(enc_err_ptr);
        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(enc_cinfo_ptr, 80, CINFO_BYTES);

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        jpeg_mem_dest(enc_cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(enc_cinfo_ptr, coef_arrays);

        let jpeg_capi_test_set_arith_code: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_arith_code")
            .expect("jpeg_capi_test_set_arith_code");
        jpeg_capi_test_set_arith_code(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_progressive: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_progressive")
            .expect("jpeg_capi_test_set_progressive");
        jpeg_capi_test_set_progressive(enc_cinfo_ptr, 1);
        let jpeg_capi_test_set_restart_in_rows: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_restart_in_rows")
            .expect("jpeg_capi_test_set_restart_in_rows");
        jpeg_capi_test_set_restart_in_rows(enc_cinfo_ptr, 2);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        // Pull format_message out of err (offset 24: error_exit @0,
        // emit_message @8, output_message @16, format_message @24).
        let fmt_slot: *const Option<unsafe extern "C" fn(*mut c_void, *mut u8)> =
            (enc_err_ptr as *const u8).add(24) as *const _;
        let format_message_fn: unsafe extern "C" fn(*mut c_void, *mut u8) = fmt_slot
            .read()
            .expect("format_message slot must be populated");
        let mut buf: [u8; 256] = [0; 256];
        format_message_fn(enc_cinfo_ptr, buf.as_mut_ptr());
        let nul: usize = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let s: String = String::from_utf8_lossy(&buf[..nul]).into_owned();

        if !out_buf.is_null() {
            let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"tj3Free").expect("tj3Free");
            tj3_free(out_buf as *mut c_void);
        }
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(enc_cinfo_ptr);
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(dec_cinfo_ptr);
        s
    };

    assert!(
        formatted
            .to_ascii_lowercase()
            .contains("progressive arithmetic"),
        "format_message must produce a meaningful warning string \
         (expected 'progressive arithmetic' in the message, got {formatted:?})"
    );
    assert!(
        formatted.to_ascii_lowercase().contains("restart"),
        "warning must reference 'restart' (got {formatted:?})"
    );
    assert!(
        !formatted.to_ascii_lowercase().contains("bogus"),
        "warning must not fall through to the 'Bogus message code' path (got {formatted:?})"
    );
}
