//! P0-4: end-to-end exercise of the foreign-virtual-coefficient-array
//! protocol used by stock libjpeg-turbo's `transupp.c` /
//! `jtransform_*` helpers.
//!
//! The shim's `jpeg_read_coefficients` returns a `jvirt_barray_ptr*`
//! shaped array (one entry per component) backed by the cinfo's
//! `JpegMemoryMgr`. This test simulates the in-process equivalent of
//! `jpegtran -copy none` (an identity transform with no rotation) by:
//!
//!   1. parsing source coefficients via `jpeg_read_coefficients`,
//!   2. allocating a parallel `jvirt_barray_ptr*` array on the
//!      destination cinfo via the public `cinfo->mem->request_virt_barray`
//!      vtable (mirroring `jtransform_request_workspace`),
//!   3. populating the destination arrays by copying blocks from the
//!      source arrays through `cinfo->mem->access_virt_barray` reads
//!      and `dst_cinfo->mem->access_virt_barray` writes,
//!   4. linking critical parameters with `jpeg_copy_critical_parameters`,
//!   5. handing the destination array off to `jpeg_write_coefficients`
//!      and emitting bytes through `jpeg_finish_compress`.
//!
//! Step 3 puts the pointer outside `coef_array_to_handle_table`, so
//! the `run_coefficient_writer_and_flush` shortcut misses and the
//! foreign-array materialisation path runs end-to-end. Round-trip
//! pixel equality vs the reference decode locks in correctness.

use std::ffi::{c_int, c_long, c_uint, c_void};
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

// ---------------------------------------------------------------------------
// Mirror of the C `jpeg_memory_mgr` vtable.
//
// Only the fields touched by this test are typed strictly; the rest
// are kept as `*const c_void` so the layout matches stock libjpeg
// without locking in our internal Rust signatures.
// ---------------------------------------------------------------------------

const JPOOL_IMAGE: c_int = 1;
type JDimension = c_uint;
type CBoolean = c_int;

#[repr(C)]
#[derive(Default)]
struct JVirtBarrayControl {
    mem_buffer: *mut *mut [i16; 64],
    rows_in_array: JDimension,
    blocksperrow: JDimension,
    maxaccess: JDimension,
    rows_in_mem: JDimension,
    pre_zero: CBoolean,
    dirty: CBoolean,
}

#[repr(C)]
struct MemMgr {
    alloc_small: Option<unsafe extern "C" fn(*mut c_void, c_int, usize) -> *mut c_void>,
    alloc_large: Option<unsafe extern "C" fn(*mut c_void, c_int, usize) -> *mut c_void>,
    alloc_sarray: *const c_void,
    alloc_barray: *const c_void,
    request_virt_sarray: *const c_void,
    request_virt_barray: Option<
        unsafe extern "C" fn(
            *mut c_void,
            c_int,
            CBoolean,
            JDimension,
            JDimension,
            JDimension,
        ) -> *mut JVirtBarrayControl,
    >,
    realize_virt_arrays: Option<unsafe extern "C" fn(*mut c_void)>,
    access_virt_sarray: *const c_void,
    access_virt_barray: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut JVirtBarrayControl,
            JDimension,
            JDimension,
            CBoolean,
        ) -> *mut *mut [i16; 64],
    >,
    free_pool: *const c_void,
    self_destruct: *const c_void,
    max_memory_to_use: c_long,
    max_alloc_chunk: c_long,
}

// ---------------------------------------------------------------------------
// Fixture helpers shared with other capi test modules.
// ---------------------------------------------------------------------------

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
        let rc: c_int = tj3_decompress_header(h_dec, jpeg.as_ptr(), jpeg.len());
        assert_eq!(rc, 0);
        let w: usize = tj3_get(h_dec, TJPARAM_JPEGWIDTH) as usize;
        let h_px: usize = tj3_get(h_dec, TJPARAM_JPEGHEIGHT) as usize;
        let mut dst: Vec<u8> = vec![0u8; w * h_px * 3];
        let rc: c_int = tj3_decompress(
            h_dec,
            jpeg.as_ptr(),
            jpeg.len(),
            dst.as_mut_ptr(),
            0,
            TJPF_RGB,
        );
        assert_eq!(rc, 0);
        tj3_destroy(h_dec);
        (dst, w, h_px)
    }
}

// ---------------------------------------------------------------------------
// `cinfo->mem` is the second pointer-sized slot on `jpeg_decompress_struct`
// (after `err`). Read it directly so we can drive the public memmgr
// vtable from outside the shim.
// ---------------------------------------------------------------------------

unsafe fn cinfo_mem(cinfo: *mut c_void) -> *mut MemMgr {
    let words: *mut *mut c_void = cinfo as *mut *mut c_void;
    *words.add(1) as *mut MemMgr
}

unsafe fn cinfo_num_components(cinfo: *mut c_void) -> c_int {
    // num_components sits at offset 56 inside `jpeg_decompress_struct`
    // after err/mem/progress/client_data/is_decompressor/global_state/src
    // /image_width/image_height. We dispatch via the dedicated test
    // accessor instead so the offset is whatever the shim agrees with.
    // Fall back to reading through the published accessor if present.
    let _ = cinfo;
    -1
}

#[test]
fn foreign_coef_arrays_full_round_trip_pixel_exact() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    let (jpeg_in, _src, src_w, src_h) = build_fixture_jpeg(&lib);
    let (ref_pixels, _, _) = decode_jpeg(&lib, &jpeg_in);

    let transcoded: Vec<u8> = unsafe {
        // Decompress side
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
        let src_arrays: *mut *mut JVirtBarrayControl =
            jpeg_read_coefficients(dec_cinfo_ptr) as *mut *mut JVirtBarrayControl;
        assert!(
            !src_arrays.is_null(),
            "jpeg_read_coefficients returned NULL"
        );

        // Compress side
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

        // Mirror jtransform_request_workspace: walk src components,
        // request a parallel barray in dst's pool, and copy blocks.
        // 3 components for a YCbCr 4:4:4 fixture.
        let n: usize = 3;
        let dec_mem: *mut MemMgr = cinfo_mem(dec_cinfo_ptr);
        let enc_mem: *mut MemMgr = cinfo_mem(enc_cinfo_ptr);
        assert!(!dec_mem.is_null());
        assert!(!enc_mem.is_null());
        let dec_access = (*dec_mem)
            .access_virt_barray
            .expect("dec access_virt_barray");
        let enc_alloc_small = (*enc_mem).alloc_small.expect("enc alloc_small");
        let enc_request = (*enc_mem)
            .request_virt_barray
            .expect("enc request_virt_barray");
        let enc_realize = (*enc_mem).realize_virt_arrays.expect("enc realize");
        let enc_access = (*enc_mem).access_virt_barray.expect("enc access");

        // Allocate the foreign array on dst's pool.
        let dst_arrays: *mut *mut JVirtBarrayControl = enc_alloc_small(
            enc_cinfo_ptr,
            JPOOL_IMAGE,
            n * std::mem::size_of::<*mut JVirtBarrayControl>(),
        ) as *mut *mut JVirtBarrayControl;
        assert!(!dst_arrays.is_null());

        // For each component, peek at the src barray's metadata, then
        // allocate a same-shape barray in dst.
        for ci in 0..n {
            let src_barray: *mut JVirtBarrayControl = *src_arrays.add(ci);
            assert!(!src_barray.is_null());
            let blocks_x: JDimension = (*src_barray).blocksperrow;
            let blocks_y: JDimension = (*src_barray).rows_in_array;
            let v_samp: JDimension = (*src_barray).maxaccess;
            let dst_barray: *mut JVirtBarrayControl = enc_request(
                enc_cinfo_ptr,
                JPOOL_IMAGE,
                /*pre_zero=*/ 0,
                blocks_x,
                blocks_y,
                v_samp.max(1),
            );
            assert!(!dst_barray.is_null());
            *dst_arrays.add(ci) = dst_barray;
        }
        enc_realize(enc_cinfo_ptr);

        // Copy blocks from src to dst arrays.
        for ci in 0..n {
            let src_barray: *mut JVirtBarrayControl = *src_arrays.add(ci);
            let dst_barray: *mut JVirtBarrayControl = *dst_arrays.add(ci);
            let blocks_x: usize = (*src_barray).blocksperrow as usize;
            let blocks_y: JDimension = (*src_barray).rows_in_array;
            let src_rows: *mut *mut [i16; 64] =
                dec_access(dec_cinfo_ptr, src_barray, 0, blocks_y, /*writable=*/ 0);
            let dst_rows: *mut *mut [i16; 64] =
                enc_access(enc_cinfo_ptr, dst_barray, 0, blocks_y, /*writable=*/ 1);
            assert!(!src_rows.is_null() && !dst_rows.is_null());
            for r in 0..blocks_y as usize {
                let src_row: *mut [i16; 64] = *src_rows.add(r);
                let dst_row: *mut [i16; 64] = *dst_rows.add(r);
                std::ptr::copy_nonoverlapping(src_row, dst_row, blocks_x);
            }
        }

        // Critical-parameter copy: image dims, sampling, quant, density.
        let jpeg_copy_critical_parameters: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_copy_critical_parameters")
            .expect("jpeg_copy_critical_parameters");
        jpeg_copy_critical_parameters(dec_cinfo_ptr, enc_cinfo_ptr);

        // Wire dst output buffer.
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
        jpeg_write_coefficients(enc_cinfo_ptr, dst_arrays as *mut c_void);

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(enc_cinfo_ptr);

        assert!(!out_buf.is_null(), "out_buf must be allocated");
        assert!(out_size > 0, "out_size must be non-zero");
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();

        // Cleanup
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
        let _ = cinfo_num_components; // silence dead_code
        bytes
    };

    assert!(transcoded.len() >= 4);
    assert_eq!(&transcoded[..2], &[0xFF, 0xD8]);
    assert_eq!(&transcoded[transcoded.len() - 2..], &[0xFF, 0xD9]);

    let (out_pixels, out_w, out_h) = decode_jpeg(
        &unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen2"),
        &transcoded,
    );
    assert_eq!((out_w, out_h), (src_w, src_h));
    // Lossless transcode: pixels must be identical.
    assert_eq!(
        out_pixels, ref_pixels,
        "foreign-array transcode must be pixel-exact vs reference decode"
    );
}
