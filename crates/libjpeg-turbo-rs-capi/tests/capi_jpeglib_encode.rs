//! FFI C2-*: end-to-end test for the libjpeg-style `jpeg_*` encode API.
//!
//! Tests the encode state machine (create/destroy, dest managers,
//! defaults/quality/colorspace setters, start/write/finish) via
//! `dlopen`, then decodes the result via the decode-side entry points
//! and cross-checks pixels.

#[path = "../../../tests/helpers/mod.rs"]
mod helpers;

use std::ffi::{c_int, c_uint, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

const JCS_RGB: c_int = 2;

#[repr(C)]
struct JpegCompressPrefix {
    err: *mut c_void,
    mem: *mut c_void,
    progress: *mut c_void,
    client_data: *mut c_void,
    is_decompressor: c_int,
    global_state: c_int,
    dest: *mut c_void,
    image_width: u32,
    image_height: u32,
    input_components: c_int,
    in_color_space: c_int,
    input_gamma: f64,
    scale_num: c_uint,
    scale_denom: c_uint,
    jpeg_width: u32,
    jpeg_height: u32,
}

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

#[derive(Clone, Copy)]
struct ClassicEncodeCase {
    label: &'static str,
    progressive: bool,
    arithmetic: bool,
    optimize: bool,
    lossless: bool,
    smoothing: c_int,
    restart_blocks: c_uint,
    restart_rows: c_int,
    cjpeg_args: &'static [&'static str],
    expect_byte_exact: bool,
}

fn classic_scanline_encode(
    lib: &libloading::Library,
    pixels: &[u8],
    width: usize,
    height: usize,
    case: ClassicEncodeCase,
) -> Vec<u8> {
    unsafe {
        const CINFO_BYTES: usize = 4096;
        const ERR_BYTES: usize = 512;
        let mut cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;
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

        let set_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        set_dims(cinfo_ptr, width as u32, height as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);
        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(cinfo_ptr, 90, 1);

        if case.progressive {
            let set_progressive: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
                .get(b"jpeg_simple_progression")
                .expect("jpeg_simple_progression");
            set_progressive(cinfo_ptr);
        }
        if case.arithmetic {
            let set_arithmetic: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
                .get(b"jpeg_capi_test_set_arith_code")
                .expect("jpeg_capi_test_set_arith_code");
            set_arithmetic(cinfo_ptr, 1);
        }
        if case.optimize {
            let set_optimize: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
                .get(b"jpeg_capi_test_set_optimize_coding")
                .expect("jpeg_capi_test_set_optimize_coding");
            set_optimize(cinfo_ptr, 1);
        }
        if case.smoothing > 0 {
            let set_smoothing: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
                .get(b"jpeg_capi_test_set_smoothing_factor")
                .expect("jpeg_capi_test_set_smoothing_factor");
            set_smoothing(cinfo_ptr, case.smoothing);
        }
        if case.lossless {
            let enable_lossless: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, c_int, c_int),
            > = lib
                .get(b"jpeg_enable_lossless")
                .expect("jpeg_enable_lossless");
            enable_lossless(cinfo_ptr, 1, 0);
        }
        if case.restart_blocks > 0 {
            let set_restart: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_uint)> = lib
                .get(b"jpeg_capi_test_set_restart_interval")
                .expect("jpeg_capi_test_set_restart_interval");
            set_restart(cinfo_ptr, case.restart_blocks);
        }
        if case.restart_rows > 0 {
            let set_restart_rows: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
                lib.get(b"jpeg_capi_test_set_restart_in_rows")
                    .expect("jpeg_capi_test_set_restart_in_rows");
            set_restart_rows(cinfo_ptr, case.restart_rows);
        }

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        jpeg_start_compress(cinfo_ptr, 1);
        let jpeg_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");
        let mut written: usize = 0;
        while written < height {
            let row_ptr: *mut u8 = pixels[written * width * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_write_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(got > 0, "{}: scanline encoder stalled", case.label);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);
        assert!(!out_buf.is_null(), "{}: output buffer is null", case.label);
        let encoded: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
        encoded
    }
}

fn dri_intervals(jpeg: &[u8]) -> Vec<u16> {
    jpeg.windows(6)
        .filter(|window| window[0..4] == [0xff, 0xdd, 0x00, 0x04])
        .map(|window| u16::from_be_bytes([window[4], window[5]]))
        .collect()
}

fn restart_marker_count(jpeg: &[u8]) -> usize {
    jpeg.windows(2)
        .filter(|window| window[0] == 0xff && (0xd0..=0xd7).contains(&window[1]))
        .count()
}

/// C2-1: create -> set_defaults -> set_quality -> destroy is crash-free
/// and leaves the cinfo in a usable shape (num_components populated,
/// comp_info pointer set).
#[test]
fn c2_1_compress_create_setup_destroy() {
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

        // Populate the 3 fields cjpeg sets before jpeg_set_defaults.
        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, 64, 64, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(cinfo_ptr, 75, 1);

        // Verify: num_components = 3, in_color_space preserved.
        let jpeg_capi_test_get_compress_state: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int, *mut c_int),
        > = lib
            .get(b"jpeg_capi_test_get_compress_state")
            .expect("jpeg_capi_test_get_compress_state");
        let mut num_components: c_int = 0;
        let mut jpeg_cs: c_int = 0;
        let mut in_cs: c_int = 0;
        jpeg_capi_test_get_compress_state(cinfo_ptr, &mut num_components, &mut jpeg_cs, &mut in_cs);
        assert_eq!(num_components, 3);
        assert_eq!(in_cs, JCS_RGB);
        // jpeg_set_defaults -> jpeg_default_colorspace: RGB input → YCbCr JPEG.
        assert_eq!(jpeg_cs, 3 /* JCS_YCbCr */);

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// `jpeg_calc_jpeg_dimensions` is an exported libjpeg 7+ helper that lets
/// callers compute `jpeg_width` / `jpeg_height` before `jpeg_start_compress`.
#[test]
fn c2_1_calc_jpeg_dimensions_sets_public_compress_fields() {
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

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, 641, 479, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let state: &mut JpegCompressPrefix = &mut *(cinfo_ptr as *mut JpegCompressPrefix);
        state.jpeg_width = 0;
        state.jpeg_height = 0;

        let jpeg_calc_jpeg_dimensions: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_calc_jpeg_dimensions")
            .expect("jpeg_calc_jpeg_dimensions");
        jpeg_calc_jpeg_dimensions(cinfo_ptr);

        assert_eq!(state.jpeg_width, 641);
        assert_eq!(state.jpeg_height, 479);

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// C2-1 mem_dest: a NULL caller buffer is allocated by the library *inside*
/// `jpeg_mem_dest`, before any compression runs (jdatadst.c:267-273), and the
/// caller's stale `*outsize` is replaced by the real capacity.
///
/// P4-108: this previously asserted the opposite — that `*outbuffer` stays
/// NULL and `*outsize` becomes 0 — pinning shim-only behaviour that no C
/// libjpeg has, so a consumer reading `*outbuffer` between `jpeg_mem_dest` and
/// `jpeg_finish_compress` saw NULL. The exact allocation size is not asserted
/// here; `capi_classic_dest_ownership.rs` cross-checks it against a reference
/// v8 build.
#[test]
fn c2_1_mem_dest_installs_cleanly() {
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

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0xDEAD;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);
        assert!(
            !out_buf.is_null(),
            "jpeg_mem_dest must allocate immediately when *outbuffer is NULL"
        );
        assert!(
            out_size > 0 && out_size != 0xDEAD,
            "*outsize must be replaced by the allocated capacity, got {out_size}"
        );
        // The buffer is the caller's to free once jpeg_mem_dest has published
        // it, exactly as after a full compress.
        let allocated: *mut u8 = out_buf;

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
        // Release the libc-malloc'd buffer through the same allocator the
        // library used.
        let libc_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        libc_free(allocated as *mut c_void);
    }
}

/// C2-2: the start/write/finish pipeline produces a JPEG that our own
/// decode side can read back pixel-identically to the Rust-native
/// compress function.
#[test]
fn c2_2_write_scanlines_roundtrip_pixel_matches_rust_native() {
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

        // Fill a 64x64 RGB gradient, identical to the decode-side fixture.
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

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(cinfo_ptr, 75, 1);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        jpeg_start_compress(cinfo_ptr, 1);

        let jpeg_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");

        let mut written: usize = 0;
        while written < h_px {
            let row_ptr: *mut u8 = src[written * w * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_write_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(
                got >= 1,
                "jpeg_write_scanlines returned 0 with rows remaining"
            );
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        // After finish, the mem-dest outbuffer should hold a valid JPEG.
        assert!(!out_buf.is_null(), "mem_dest outbuffer was not populated");
        assert!(out_size > 100, "output too small ({out_size} bytes)");
        // SOI / EOI sanity.
        let encoded: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        assert_eq!(&encoded[..2], &[0xFF, 0xD8], "missing SOI");
        assert_eq!(&encoded[encoded.len() - 2..], &[0xFF, 0xD9], "missing EOI");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        // Cross-check: the byte stream must decode via the Rust-native
        // decoder to a pixel grid that closely matches the source gradient.
        let decoded = libjpeg_turbo_rs::decompress(&encoded).expect("decompress roundtrip");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h_px);

        // Quality=75 4:2:0 roundtrip: max per-channel diff ≤ 20 for
        // smooth gradients. We measure observed diff to detect regression
        // without overfitting to a single quality factor.
        let mut max_diff: u8 = 0;
        for (&a, &b) in src.iter().zip(decoded.data.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff <= 20, "roundtrip max diff {max_diff} exceeds 20");

        // The C-path JPEG and a Rust-path JPEG from the same pixels and
        // parameters should also round-trip to the same decoded output,
        // i.e. both paths call into the same encode pipeline.
        let native_jpeg = libjpeg_turbo_rs::compress(
            &src,
            w,
            h_px,
            libjpeg_turbo_rs::PixelFormat::Rgb,
            75,
            libjpeg_turbo_rs::Subsampling::S420,
        )
        .expect("native compress");
        assert_eq!(
            encoded, native_jpeg,
            "classic jpeg_* bytes diverge from Rust-native bytes"
        );

        // Release the libc-malloc'd buffer.
        let libc_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        libc_free(out_buf as *mut c_void);
    }
}

/// C2-3: `jpeg_quality_scaling` matches the libjpeg scaling curve.
///
/// libjpeg formula:
///   quality < 50: scale = 5000 / quality
///   quality >= 50: scale = 200 - 2 * quality
#[test]
fn c2_3_quality_scaling_matches_libjpeg_formula() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_quality_scaling: libloading::Symbol<unsafe extern "C" fn(c_int) -> c_int> = lib
            .get(b"jpeg_quality_scaling")
            .expect("jpeg_quality_scaling");
        // Spot-check the curve at three representative points.
        // libjpeg clamps values outside 1..100 to the nearest endpoint.
        assert_eq!(jpeg_quality_scaling(100), 0);
        assert_eq!(jpeg_quality_scaling(75), 50);
        assert_eq!(jpeg_quality_scaling(50), 100);
        assert_eq!(jpeg_quality_scaling(25), 200);
        // Below 50: 5000 / q.
        assert_eq!(jpeg_quality_scaling(10), 500);
    }
}

/// C2-3: `jpeg_simple_progression` flips `progressive_mode` on so the
/// next `jpeg_finish_compress` emits SOF2 instead of SOF0.
#[test]
fn c2_3_simple_progression_emits_progressive_stream() {
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

        let w: usize = 32;
        let h_px: usize = 32;
        let src: Vec<u8> = (0..w * h_px * 3).map(|i| (i % 256) as u8).collect();

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(cinfo_ptr, 75, 1);

        let jpeg_simple_progression: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_simple_progression")
            .expect("jpeg_simple_progression");
        jpeg_simple_progression(cinfo_ptr);

        let set_restart: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_uint)> = lib
            .get(b"jpeg_capi_test_set_restart_interval")
            .expect("jpeg_capi_test_set_restart_interval");
        set_restart(cinfo_ptr, 4);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        jpeg_start_compress(cinfo_ptr, 1);

        let jpeg_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");
        let mut written: usize = 0;
        while written < h_px {
            let row_ptr: *mut u8 = src[written * w * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_write_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(got >= 1);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);
        assert!(!out_buf.is_null());

        // Scan for the SOF2 marker (0xFF 0xC2) — the progressive SOF.
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        let has_sof2: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC2]);
        let has_sof0: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC0]);
        assert!(has_sof2, "expected SOF2 marker in progressive stream");
        assert!(!has_sof0, "progressive stream must not contain SOF0");
        assert!(
            bytes
                .windows(6)
                .any(|w| w == [0xFF, 0xDD, 0x00, 0x04, 0x00, 0x04]),
            "progressive stream must preserve cinfo.restart_interval as DRI=4"
        );
        assert!(
            bytes
                .windows(2)
                .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1])),
            "progressive stream with DRI=4 must contain restart markers"
        );

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// P4-82/P4-83: the classic scanline boundary must preserve public restart
/// fields across every pixel-encode entropy dispatcher and baseline input
/// smoothing independently of Huffman optimization. Each case is
/// cross-validated against stock C `cjpeg` on identical PPM input.
#[test]
fn c2_3_scanline_option_dispatch_matches_cjpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let help = std::process::Command::new(&cjpeg)
        .arg("-help")
        .output()
        .expect("cjpeg -help");
    let help_text: String = format!(
        "{}{}",
        String::from_utf8_lossy(&help.stdout),
        String::from_utf8_lossy(&help.stderr)
    );
    assert!(
        help_text.contains("-arithmetic") && help_text.contains("-lossless"),
        "P4-82 C oracle must support arithmetic and lossless modes"
    );

    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let width: usize = 96;
    let height: usize = 80;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 13 + y * 7 + 19) & 0xff) as u8);
            pixels.push(((x * 3 + y * 17 + 71) & 0xff) as u8);
            pixels.push(((x * 11 + y * 5 + 137) & 0xff) as u8);
        }
    }
    let ppm: Vec<u8> = helpers::build_ppm(&pixels, width, height);

    let cases: &[ClassicEncodeCase] = &[
        ClassicEncodeCase {
            label: "baseline-blocks",
            progressive: false,
            arithmetic: false,
            optimize: false,
            lossless: false,
            smoothing: 0,
            restart_blocks: 4,
            restart_rows: 0,
            cjpeg_args: &["-restart", "4b"],
            expect_byte_exact: true,
        },
        ClassicEncodeCase {
            label: "baseline-rows",
            restart_blocks: 0,
            restart_rows: 2,
            cjpeg_args: &["-restart", "2"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "optimized-blocks",
            optimize: true,
            restart_blocks: 4,
            cjpeg_args: &["-optimize", "-restart", "4b"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "optimized-rows",
            optimize: true,
            restart_rows: 2,
            cjpeg_args: &["-optimize", "-restart", "2"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "smoothing-blocks",
            smoothing: 25,
            restart_blocks: 4,
            cjpeg_args: &["-smooth", "25", "-restart", "4b"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "smoothing-rows",
            smoothing: 25,
            restart_rows: 2,
            cjpeg_args: &["-smooth", "25", "-restart", "2"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "progressive-blocks",
            progressive: true,
            restart_blocks: 4,
            cjpeg_args: &["-progressive", "-restart", "4b"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "progressive-rows",
            progressive: true,
            restart_rows: 2,
            cjpeg_args: &["-progressive", "-restart", "2"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "arithmetic-blocks",
            arithmetic: true,
            restart_blocks: 4,
            cjpeg_args: &["-arithmetic", "-restart", "4b"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "arithmetic-rows",
            arithmetic: true,
            restart_rows: 2,
            cjpeg_args: &["-arithmetic", "-restart", "2"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "arithmetic-progressive-blocks",
            progressive: true,
            arithmetic: true,
            restart_blocks: 4,
            cjpeg_args: &["-arithmetic", "-progressive", "-restart", "4b"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "arithmetic-progressive-rows",
            progressive: true,
            arithmetic: true,
            restart_rows: 2,
            cjpeg_args: &["-arithmetic", "-progressive", "-restart", "2"],
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "lossless-blocks",
            lossless: true,
            // C lossless requires a whole number of MCU rows in block mode;
            // with one-sample MCUs at width 96, 96b is one row.
            restart_blocks: 96,
            cjpeg_args: &["-lossless", "1,0", "-restart", "96b"],
            expect_byte_exact: false,
            ..cases_default()
        },
        ClassicEncodeCase {
            label: "lossless-rows",
            lossless: true,
            restart_rows: 2,
            cjpeg_args: &["-lossless", "1,0", "-restart", "2"],
            expect_byte_exact: false,
            ..cases_default()
        },
    ];

    for &case in cases {
        let rust_jpeg: Vec<u8> = classic_scanline_encode(&lib, &pixels, width, height, case);
        let mut c_args: Vec<&str> = vec!["-quality", "90", "-sample", "2x2", "-dct", "int"];
        c_args.extend_from_slice(case.cjpeg_args);
        let c_jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(&cjpeg, &ppm, &c_args, case.label);

        let rust_dri: Vec<u16> = dri_intervals(&rust_jpeg);
        let c_dri: Vec<u16> = dri_intervals(&c_jpeg);
        assert!(
            !rust_dri.is_empty(),
            "{}: Rust output lacks DRI",
            case.label
        );
        assert_eq!(rust_dri, c_dri, "{}: DRI sequence differs", case.label);
        let rust_rst: usize = restart_marker_count(&rust_jpeg);
        let c_rst: usize = restart_marker_count(&c_jpeg);
        assert!(rust_rst > 0, "{}: Rust output lacks RST", case.label);
        assert_eq!(rust_rst, c_rst, "{}: RST count differs", case.label);

        if case.expect_byte_exact {
            assert_eq!(rust_jpeg, c_jpeg, "{}: bytes differ from cjpeg", case.label);
        } else {
            let (rust_width, rust_height, rust_pixels): (usize, usize, Vec<u8>) =
                helpers::decode_with_c_djpeg(&djpeg, &rust_jpeg, &format!("{}-rust", case.label));
            let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
                helpers::decode_with_c_djpeg(&djpeg, &c_jpeg, &format!("{}-c", case.label));
            assert_eq!(rust_width, width, "{}: Rust JPEG width", case.label);
            assert_eq!(rust_height, height, "{}: Rust JPEG height", case.label);
            assert_eq!(c_width, width, "{}: C JPEG width", case.label);
            assert_eq!(c_height, height, "{}: C JPEG height", case.label);
            assert_eq!(
                rust_pixels, c_pixels,
                "{}: stock djpeg decoded pixels",
                case.label
            );
            assert_eq!(
                rust_pixels, pixels,
                "{}: lossless source pixels",
                case.label
            );
        }
    }
}

const fn cases_default() -> ClassicEncodeCase {
    ClassicEncodeCase {
        label: "",
        progressive: false,
        arithmetic: false,
        optimize: false,
        lossless: false,
        smoothing: 0,
        restart_blocks: 0,
        restart_rows: 0,
        cjpeg_args: &[],
        expect_byte_exact: true,
    }
}

/// C2-3: add_quant_table / default_qtables / enable_lossless / suppress_tables
/// do not crash on reasonable inputs.
#[test]
fn c2_3_helpers_null_and_basic_guards() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_add_quant_table: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, *const u32, c_int, c_int),
        > = lib
            .get(b"jpeg_add_quant_table")
            .expect("jpeg_add_quant_table");
        jpeg_add_quant_table(std::ptr::null_mut(), 0, std::ptr::null(), 100, 1);

        let jpeg_default_qtables: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
            lib.get(b"jpeg_default_qtables")
                .expect("jpeg_default_qtables");
        jpeg_default_qtables(std::ptr::null_mut(), 1);

        let jpeg_enable_lossless: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, c_int),
        > = lib
            .get(b"jpeg_enable_lossless")
            .expect("jpeg_enable_lossless");
        jpeg_enable_lossless(std::ptr::null_mut(), 1, 0);

        let jpeg_suppress_tables: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
            lib.get(b"jpeg_suppress_tables")
                .expect("jpeg_suppress_tables");
        jpeg_suppress_tables(std::ptr::null_mut(), 1);
    }
}

/// Null-guard: destroy and setup functions must accept NULL without
/// crashing.
#[test]
fn c2_1_null_arguments_return_safely() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(std::ptr::null_mut());

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(std::ptr::null_mut());

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(std::ptr::null_mut(), 75, 1);
    }
}

/// C2-4: a custom COM marker written via `jpeg_write_marker` shows up
/// in the output stream immediately after the SOI. Mirrors how cjpeg's
/// `-comment` flag plumbs text through the classic API.
#[test]
fn c2_4_write_marker_inserts_custom_segment_after_soi() {
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

        let w: usize = 16;
        let h_px: usize = 16;
        let src: Vec<u8> = vec![128u8; w * h_px * 3];

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        jpeg_start_compress(cinfo_ptr, 1);

        // Write a COM (0xFE) marker containing ASCII text.
        let marker_payload: &[u8] = b"hello-from-jpeg-write-marker";
        let jpeg_write_marker: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, *const u8, std::os::raw::c_uint),
        > = lib.get(b"jpeg_write_marker").expect("jpeg_write_marker");
        jpeg_write_marker(
            cinfo_ptr,
            0xFE,
            marker_payload.as_ptr(),
            marker_payload.len() as std::os::raw::c_uint,
        );

        // Also exercise the piecemeal writers: write_m_header + write_m_byte.
        let jpeg_write_m_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, std::os::raw::c_uint),
        > = lib
            .get(b"jpeg_write_m_header")
            .expect("jpeg_write_m_header");
        jpeg_write_m_header(cinfo_ptr, 0xE1, 4);
        let jpeg_write_m_byte: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
            lib.get(b"jpeg_write_m_byte").expect("jpeg_write_m_byte");
        for b in b"TEST" {
            jpeg_write_m_byte(cinfo_ptr, *b as c_int);
        }

        // Fill scanlines.
        let jpeg_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");
        let mut written: usize = 0;
        while written < h_px {
            let row_ptr: *mut u8 = src[written * w * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_write_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(got >= 1);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        // Expect our payload in the stream.
        let needle: &[u8] = marker_payload;
        let found: bool = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "jpeg_write_marker payload missing from output");
        let app1_payload: &[u8] = b"TEST";
        let found_app1: bool = bytes.windows(app1_payload.len()).any(|w| w == app1_payload);
        assert!(found_app1, "jpeg_write_m_byte payload missing from output");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// C2-4: `jpeg_write_icc_profile` results in an APP2 `ICC_PROFILE\0`
/// segment on the stream and the decoded `Image` surfaces the same
/// profile bytes via the Rust-native decoder.
#[test]
fn c2_4_write_icc_profile_roundtrips_bytes() {
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

        let w: usize = 16;
        let h_px: usize = 16;
        let src: Vec<u8> = vec![64u8; w * h_px * 3];

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        jpeg_start_compress(cinfo_ptr, 1);

        // Synthetic ICC profile (just arbitrary bytes — the shim doesn't
        // validate ICC content, only that it surfaces through APP2).
        let icc: Vec<u8> = (0..256u32).map(|i| (i & 0xFF) as u8).collect();
        let jpeg_write_icc_profile: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, std::os::raw::c_uint),
        > = lib
            .get(b"jpeg_write_icc_profile")
            .expect("jpeg_write_icc_profile");
        jpeg_write_icc_profile(cinfo_ptr, icc.as_ptr(), icc.len() as std::os::raw::c_uint);

        let jpeg_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");
        let mut written: usize = 0;
        while written < h_px {
            let row_ptr: *mut u8 = src[written * w * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_write_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(got >= 1);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        // Look for the ICC_PROFILE signature within an APP2 segment.
        let sig: &[u8] = b"ICC_PROFILE\0";
        let found_sig: bool = bytes.windows(sig.len()).any(|w| w == sig);
        assert!(found_sig, "ICC_PROFILE signature missing from output");

        // Decode and ensure the ICC bytes round-trip.
        let img = libjpeg_turbo_rs::decompress(&bytes).expect("decompress");
        assert_eq!(
            img.icc_profile.as_deref(),
            Some(icc.as_slice()),
            "ICC profile did not round-trip through decode"
        );

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// C2-5: `jdiv_round_up(a, b)` is ceiling-divide with a zero-guard.
#[test]
fn c2_5_jdiv_round_up_matches_libjpeg_formula() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jdiv_round_up: libloading::Symbol<
            unsafe extern "C" fn(
                std::os::raw::c_long,
                std::os::raw::c_long,
            ) -> std::os::raw::c_long,
        > = lib.get(b"jdiv_round_up").expect("jdiv_round_up");
        assert_eq!(jdiv_round_up(7, 3), 3);
        assert_eq!(jdiv_round_up(6, 3), 2);
        assert_eq!(jdiv_round_up(0, 5), 0);
        assert_eq!(jdiv_round_up(1, 1), 1);
        // zero-divisor guard
        assert_eq!(jdiv_round_up(5, 0), 0);
    }
}

/// C2-5: `jcopy_block_row` copies exactly num_blocks * 64 i16 samples.
#[test]
fn c2_5_jcopy_block_row_copies_full_blocks() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jcopy_block_row: libloading::Symbol<unsafe extern "C" fn(*const i16, *mut i16, u32)> =
            lib.get(b"jcopy_block_row").expect("jcopy_block_row");
        let src: Vec<i16> = (0..128i16).collect(); // 2 blocks
        let mut dst: Vec<i16> = vec![0i16; 128];
        jcopy_block_row(src.as_ptr(), dst.as_mut_ptr(), 2);
        assert_eq!(dst, src);

        // num_blocks=0 must be a no-op.
        let mut guard: Vec<i16> = vec![-1i16; 64];
        jcopy_block_row(src.as_ptr(), guard.as_mut_ptr(), 0);
        assert!(guard.iter().all(|&x| x == -1));
    }
}

/// C2-5: 12-bit and 16-bit scanline symbols have null guards. P4-94 tracks
/// real-row buffering and finish-compress coverage; this is intentionally not
/// evidence for the high-precision encode pipeline.
#[test]
fn c2_5_high_precision_write_scanlines_null_guards() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg12_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u16, u32) -> u32,
        > = lib
            .get(b"jpeg12_write_scanlines")
            .expect("jpeg12_write_scanlines");
        let jpeg16_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u16, u32) -> u32,
        > = lib
            .get(b"jpeg16_write_scanlines")
            .expect("jpeg16_write_scanlines");
        // NULL cinfo → 0 rows.
        assert_eq!(
            jpeg12_write_scanlines(std::ptr::null_mut(), std::ptr::null_mut(), 1),
            0
        );
        assert_eq!(
            jpeg16_write_scanlines(std::ptr::null_mut(), std::ptr::null_mut(), 1),
            0
        );
    }
}

/// C2-4: `jpeg_write_tables` emits a standalone tables datastream
/// (SOI ... EOI, with only DQT/DHT segments, no SOF). Consumers of
/// the abbreviated-file convention depend on this shape.
#[test]
fn c2_4_write_tables_emits_tables_only_datastream() {
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
        // Minimum setup so set_quality has a valid struct.
        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, 8, 8, 1, 1 /* JCS_GRAYSCALE */);
        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_tables: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_write_tables").expect("jpeg_write_tables");
        jpeg_write_tables(cinfo_ptr);

        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "tables stream must start SOI");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            &[0xFF, 0xD9],
            "tables stream must end EOI"
        );
        // No SOF0/SOF2 must appear inside a tables-only datastream.
        let has_sof0: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC0]);
        let has_sof2: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC2]);
        assert!(!has_sof0, "tables-only stream must not contain SOF0");
        assert!(!has_sof2, "tables-only stream must not contain SOF2");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// SA2-6: run the full encode pipeline on an EXACT 584-byte cinfo buffer —
/// the canonical `sizeof(struct jpeg_compress_struct)` on LP64 for
/// `JPEG_LIB_VERSION >= 80`. This simulates a stock cjpeg build that
/// allocates `struct jpeg_compress_struct cinfo;` on the stack.
///
/// A passing test proves that our shim:
/// 1. Reads/writes every field at its canonical libjpeg offset
/// 2. Never touches bytes beyond offset 583 (which would be caller memory)
/// 3. Produces a valid JPEG roundtrip that the Rust decoder recovers
#[test]
fn sa2_6_stock_abi_cinfo_size_encode_pipeline_works() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        // Exact libjpeg v80 LP64 sizeof — see jpeglib.rs offset assertions.
        const CINFO_BYTES: usize = 584;
        // Embed the cinfo in a larger buffer with red-zone bytes around
        // it so we can detect out-of-bounds writes from the shim.
        const REDZONE: usize = 32;
        let mut backing: Vec<u8> = vec![0xAAu8; REDZONE * 2 + CINFO_BYTES];
        let cinfo_ptr: *mut c_void = backing.as_mut_ptr().add(REDZONE) as *mut c_void;
        // Zero just the cinfo region (mirrors the `memset` in jcapimin).
        std::ptr::write_bytes(cinfo_ptr as *mut u8, 0, CINFO_BYTES);

        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(err_ptr);
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        let jpeg_create_compress_fn: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress_fn(cinfo_ptr, 80, CINFO_BYTES);

        let w: usize = 32;
        let h_px: usize = 24;
        let mut src: Vec<u8> = Vec::with_capacity(w * h_px * 3);
        for y in 0..h_px {
            for x in 0..w {
                src.push((x * 4) as u8);
                src.push((y * 8) as u8);
                src.push(((x + y) * 3) as u8);
            }
        }

        let set_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        set_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let set_defaults_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        set_defaults_fn(cinfo_ptr);

        let set_quality_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        set_quality_fn(cinfo_ptr, 75, 1);

        let mem_dest_fn: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        mem_dest_fn(cinfo_ptr, &mut out_buf, &mut out_size);

        let start_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        start_fn(cinfo_ptr, 1);

        let write_fn: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");
        let mut written: usize = 0;
        while written < h_px {
            let row_ptr: *mut u8 = src[written * w * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = write_fn(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(got >= 1, "write_scanlines returned 0 at row {written}");
            written += got as usize;
        }

        let finish_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        finish_fn(cinfo_ptr);

        assert!(!out_buf.is_null(), "encoded buffer is null");
        assert!(out_size > 80, "encoded stream too small ({out_size})");
        let encoded: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        assert_eq!(&encoded[..2], &[0xFF, 0xD8], "missing SOI");
        assert_eq!(&encoded[encoded.len() - 2..], &[0xFF, 0xD9], "missing EOI");

        let destroy_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        destroy_fn(cinfo_ptr);

        // Verify the red-zone: bytes outside [REDZONE, REDZONE+CINFO_BYTES)
        // must remain 0xAA — any corruption means the shim wrote past the
        // canonical 584-byte envelope.
        for i in 0..REDZONE {
            assert_eq!(
                backing[i], 0xAA,
                "red-zone byte {i} corrupted (shim wrote before cinfo)"
            );
            assert_eq!(
                backing[REDZONE + CINFO_BYTES + i],
                0xAA,
                "red-zone byte {i} corrupted (shim wrote past canonical 584-byte struct)"
            );
        }

        let decoded = libjpeg_turbo_rs::decompress(&encoded).expect("decode roundtrip");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h_px);

        let mut max_diff: u8 = 0;
        for (&a, &b) in src.iter().zip(decoded.data.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff <= 20, "roundtrip max diff {max_diff} > 20");

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}
