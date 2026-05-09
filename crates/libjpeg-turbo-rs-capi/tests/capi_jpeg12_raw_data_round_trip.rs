//! P3-2 Layer 2: 12-bit raw-data encode + decode round-trip via the
//! shim's `jpeg12_write_raw_data` / `jpeg12_read_raw_data` entry points.
//!
//! Three independent gates:
//!   1. Shim encode → native decode: Y plane recovery within DCT
//!      tolerance (proves the encoder shim plumbs i16 planes through
//!      `compress_raw_12` correctly).
//!   2. Native encode → shim decode: shim recovers the same i16
//!      planes the native decoder would (proves the decoder shim
//!      plumbs `decompress_raw_12` cache rows correctly).
//!   3. Shim encode → shim decode: end-to-end agreement.
//!
//! Both directions hard-panic on Rust failures (CLAUDE.md strict
//! assertion rule). Skip only when the cdylib cannot be located.
//!
//! Pre-fix history. Before this layer, `jpeg12_write_raw_data` and
//! `jpeg12_read_raw_data` were JERR_NOTIMPL stubs; this test would
//! immediately error_exit on encode. Wiring both through
//! `libjpeg_turbo_rs::raw_data_12::{compress,decompress}_raw_12`
//! turns the round-trip GREEN.

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

const TRUE: c_int = 1;
const JCS_YCBCR: c_int = 3;
const JPEG_HEADER_OK: c_int = 1;

// ---- JpegCompressPublic offsets (LP64) ----
// Verified by the compile-time `offset_of!` assertions in
// `tests/abi_offsets.rs`; values are stable across macOS/Linux
// aarch64/x86_64.
const C_IMAGE_WIDTH: usize = 48;
const C_IMAGE_HEIGHT: usize = 52;
const C_INPUT_COMPONENTS: usize = 56;
const C_IN_COLOR_SPACE: usize = 60;
const C_DATA_PRECISION: usize = 88;
const C_RAW_DATA_IN: usize = 288;
const C_NEXT_SCANLINE: usize = 340;
const C_MAX_V_SAMP_FACTOR: usize = 352;

// ---- JpegDecompressPublic offsets (LP64) ----
const D_RAW_DATA_OUT: usize = 92;
const D_OUTPUT_WIDTH: usize = 136;
const D_OUTPUT_HEIGHT: usize = 140;
const D_OUTPUT_SCANLINE: usize = 168;
const D_DATA_PRECISION: usize = 296;
const D_NUM_COMPONENTS: usize = 56;
const D_COMP_INFO: usize = 304;
const D_MAX_V_SAMP_FACTOR: usize = 412;
const D_MIN_DCT_V_SCALED_SIZE: usize = 420;
const D_COMP_VSF_FIELD: usize = 12;
const D_COMP_INFO_STRUCT_SIZE: usize = 96;

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

unsafe fn write_u32_at(buf: *mut u8, offset: usize, val: u32) {
    let ptr: *mut u32 = buf.add(offset) as *mut u32;
    ptr.write_unaligned(val);
}

unsafe fn write_cint_at(buf: *mut u8, offset: usize, val: c_int) {
    let ptr: *mut c_int = buf.add(offset) as *mut c_int;
    ptr.write_unaligned(val);
}

unsafe fn read_u32_at(buf: *const u8, offset: usize) -> u32 {
    let ptr: *const u32 = buf.add(offset) as *const u32;
    ptr.read_unaligned()
}

unsafe fn read_cint_at(buf: *const u8, offset: usize) -> c_int {
    let ptr: *const c_int = buf.add(offset) as *const c_int;
    ptr.read_unaligned()
}

unsafe fn read_ptr_at(buf: *const u8, offset: usize) -> *mut u8 {
    let ptr: *const *mut u8 = buf.add(offset) as *const *mut u8;
    ptr.read_unaligned()
}

/// Encode a 4:2:0 12-bit YCbCr image through the shim's
/// `jpeg12_write_raw_data` path. `planes[0]` is luma, `planes[1..3]`
/// are chroma at half H/V resolution.
unsafe fn encode_12bit_4_2_0_via_capi(
    lib: &libloading::Library,
    planes: &[Vec<i16>],
    plane_widths: &[usize],
    plane_heights: &[usize],
    image_width: usize,
    image_height: usize,
    quality: c_int,
) -> Vec<u8> {
    let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
        lib.get(b"jpeg_std_error").expect("jpeg_std_error");
    let jpeg_create_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, usize)> =
        lib.get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
    let jpeg_mem_dest: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
    > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
    let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
        lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
    let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
        lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
    let jpeg_set_colorspace: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
        .get(b"jpeg_set_colorspace")
        .expect("jpeg_set_colorspace");
    let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
        .get(b"jpeg_start_compress")
        .expect("jpeg_start_compress");
    let jpeg12_write_raw_data: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut *mut i16, u32) -> u32,
    > = lib
        .get(b"jpeg12_write_raw_data")
        .expect("jpeg12_write_raw_data");
    let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_finish_compress")
        .expect("jpeg_finish_compress");
    let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_destroy_compress")
        .expect("jpeg_destroy_compress");

    const CINFO_BYTES: usize = 4096;
    let mut cinfo_buf: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    const ERR_BYTES: usize = 512;
    let mut err_buf: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
    let err_ptr: *mut c_void = err_buf.as_mut_ptr() as *mut c_void;

    let err_ret: *mut c_void = jpeg_std_error(err_ptr);
    assert_eq!(err_ret, err_ptr, "jpeg_std_error must return its argument");
    (cinfo_ptr as *mut *mut c_void).write(err_ptr);

    jpeg_create_compress(cinfo_ptr, 80 /* JPEG_LIB_VERSION */, CINFO_BYTES);

    let mut out_ptr: *mut u8 = std::ptr::null_mut();
    let mut out_size: c_ulong = 0;
    jpeg_mem_dest(cinfo_ptr, &mut out_ptr, &mut out_size);

    let cinfo_bytes: *mut u8 = cinfo_ptr as *mut u8;
    write_u32_at(cinfo_bytes, C_IMAGE_WIDTH, image_width as u32);
    write_u32_at(cinfo_bytes, C_IMAGE_HEIGHT, image_height as u32);
    write_cint_at(cinfo_bytes, C_INPUT_COMPONENTS, 3);
    write_cint_at(cinfo_bytes, C_IN_COLOR_SPACE, JCS_YCBCR);

    jpeg_set_defaults(cinfo_ptr);
    jpeg_set_quality(cinfo_ptr, quality, TRUE);
    // `jpeg_set_colorspace(JCS_YCBCR)` writes per-component sampling
    // factors that default to 4:2:0 (Y=2x2, Cb/Cr=1x1) — matching
    // what the shim derives via `subsampling_from_comp_info`. We
    // leave that default in place and feed pre-downsampled chroma.
    jpeg_set_colorspace(cinfo_ptr, JCS_YCBCR);

    // Switch the compressor to 12-bit mode AFTER `jpeg_set_defaults`
    // (which resets `data_precision = 8`) and AFTER
    // `jpeg_set_colorspace` (which respects `data_precision` only
    // for table selection — the i16 plane delivery uses sampling
    // factors written by `jpeg_set_colorspace`).
    write_cint_at(cinfo_bytes, C_DATA_PRECISION, 12);
    write_cint_at(cinfo_bytes, C_RAW_DATA_IN, TRUE);

    jpeg_start_compress(cinfo_ptr, TRUE);

    let actual_max_vsf: c_int = read_cint_at(cinfo_bytes, C_MAX_V_SAMP_FACTOR);
    let dct_size: usize = 8;
    let lines_per_imcu: u32 = (actual_max_vsf.max(1) as usize * dct_size) as u32;
    assert_eq!(
        lines_per_imcu, 16,
        "4:2:0: max_v_samp_factor=2 → lines_per_imcu=16, got {lines_per_imcu}"
    );

    let total_rows: usize = plane_heights[0];
    let num_imcu_rows: usize = total_rows.div_ceil(lines_per_imcu as usize);

    // Per-component v_samp_factors: Y=max_vsf, chroma=1 in 4:2:0.
    let v_samps: [usize; 3] = [actual_max_vsf as usize, 1, 1];

    for imcu in 0..num_imcu_rows {
        let mut comp_row_ptrs: Vec<Vec<*mut i16>> = (0..planes.len())
            .map(|ci| {
                let rows_this_imcu: usize = v_samps[ci] * dct_size;
                let base_row: usize = imcu * rows_this_imcu;
                let pw: usize = plane_widths[ci];
                let ph: usize = plane_heights[ci];
                (0..rows_this_imcu)
                    .map(|ri| {
                        let actual_row: usize = (base_row + ri).min(ph.saturating_sub(1));
                        let row_ptr: *const i16 = planes[ci][actual_row * pw..].as_ptr();
                        row_ptr as *mut i16
                    })
                    .collect()
            })
            .collect();

        let mut outer: Vec<*mut *mut i16> =
            comp_row_ptrs.iter_mut().map(|v| v.as_mut_ptr()).collect();

        let lines_written: u32 =
            jpeg12_write_raw_data(cinfo_ptr, outer.as_mut_ptr(), lines_per_imcu);
        assert_eq!(
            lines_written, lines_per_imcu,
            "iMCU {imcu}: jpeg12_write_raw_data returned {lines_written}, expected {lines_per_imcu}"
        );
    }

    let ns: u32 = read_u32_at(cinfo_bytes, C_NEXT_SCANLINE);
    assert_eq!(
        ns, image_height as u32,
        "next_scanline={ns} expected image_height={image_height}"
    );

    jpeg_finish_compress(cinfo_ptr);
    jpeg_destroy_compress(cinfo_ptr);

    assert!(!out_ptr.is_null(), "jpeg_mem_dest output is NULL");
    assert!(out_size > 0, "jpeg_mem_dest size is 0");

    let jpeg_bytes: Vec<u8> = std::slice::from_raw_parts(out_ptr, out_size as usize).to_vec();
    extern "C" {
        fn free(ptr: *mut c_void);
    }
    free(out_ptr as *mut c_void);
    jpeg_bytes
}

/// Decode a 12-bit JPEG through the shim's `jpeg12_read_raw_data`
/// path. Returns `(planes, plane_widths, plane_heights)`. Planes
/// are MCU-aligned; the leading `image_width` / `image_height`
/// region holds the logical samples.
unsafe fn decode_12bit_via_capi(
    lib: &libloading::Library,
    jpeg_bytes: &[u8],
) -> (Vec<Vec<i16>>, Vec<usize>, Vec<usize>) {
    let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
        lib.get(b"jpeg_std_error").expect("jpeg_std_error");
    let jpeg_create_decompress: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, c_int, usize),
    > = lib
        .get(b"jpeg_CreateDecompress")
        .expect("jpeg_CreateDecompress");
    let jpeg_mem_src: libloading::Symbol<unsafe extern "C" fn(*mut c_void, *const u8, c_ulong)> =
        lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
    let jpeg_read_header: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int) -> c_int> =
        lib.get(b"jpeg_read_header").expect("jpeg_read_header");
    let jpeg_calc_output_dimensions: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_calc_output_dimensions")
        .expect("jpeg_calc_output_dimensions");
    let jpeg_start_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> = lib
        .get(b"jpeg_start_decompress")
        .expect("jpeg_start_decompress");
    let jpeg12_read_raw_data: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut *mut i16, u32) -> u32,
    > = lib
        .get(b"jpeg12_read_raw_data")
        .expect("jpeg12_read_raw_data");
    let jpeg_finish_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
        lib.get(b"jpeg_finish_decompress")
            .expect("jpeg_finish_decompress");
    let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_destroy_decompress")
        .expect("jpeg_destroy_decompress");

    const CINFO_BYTES: usize = 4096;
    let mut cinfo_buf: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    const ERR_BYTES: usize = 512;
    let mut err_buf: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
    let err_ptr: *mut c_void = err_buf.as_mut_ptr() as *mut c_void;

    let err_ret: *mut c_void = jpeg_std_error(err_ptr);
    assert_eq!(err_ret, err_ptr, "jpeg_std_error must return its argument");
    (cinfo_ptr as *mut *mut c_void).write(err_ptr);

    jpeg_create_decompress(cinfo_ptr, 80, CINFO_BYTES);
    jpeg_mem_src(cinfo_ptr, jpeg_bytes.as_ptr(), jpeg_bytes.len() as c_ulong);

    let rc: c_int = jpeg_read_header(cinfo_ptr, 1);
    assert_eq!(rc, JPEG_HEADER_OK, "jpeg_read_header must succeed");

    let cinfo_bytes: *mut u8 = cinfo_ptr as *mut u8;
    write_cint_at(cinfo_bytes, D_RAW_DATA_OUT, TRUE);

    let prec: c_int = read_cint_at(cinfo_bytes, D_DATA_PRECISION);
    assert_eq!(
        prec, 12,
        "expected data_precision=12 after jpeg_read_header on 12-bit stream, got {prec}"
    );

    jpeg_calc_output_dimensions(cinfo_ptr);
    let rc: c_int = jpeg_start_decompress(cinfo_ptr);
    assert_eq!(rc, 1, "jpeg_start_decompress must succeed");

    let output_height: u32 = read_u32_at(cinfo_bytes, D_OUTPUT_HEIGHT);
    let output_width: u32 = read_u32_at(cinfo_bytes, D_OUTPUT_WIDTH);
    let max_vsf: c_int = read_cint_at(cinfo_bytes, D_MAX_V_SAMP_FACTOR);
    let min_dct_v: c_int = read_cint_at(cinfo_bytes, D_MIN_DCT_V_SCALED_SIZE);
    let num_components: usize = read_cint_at(cinfo_bytes, D_NUM_COMPONENTS) as usize;
    let comp_info_raw: *mut u8 = read_ptr_at(cinfo_bytes, D_COMP_INFO);

    assert!(num_components > 0, "num_components must be > 0");
    assert!(max_vsf > 0, "max_v_samp_factor must be > 0");

    let mut comp_vsf: Vec<usize> = Vec::with_capacity(num_components);
    for i in 0..num_components {
        let vsf: c_int = if comp_info_raw.is_null() {
            max_vsf
        } else {
            let comp_base: *const u8 = comp_info_raw.add(i * D_COMP_INFO_STRUCT_SIZE);
            read_cint_at(comp_base, D_COMP_VSF_FIELD)
        };
        comp_vsf.push(vsf.max(1) as usize);
    }

    let dct_size: usize = if min_dct_v > 0 { min_dct_v as usize } else { 8 };
    let rows_per_imcu: usize = max_vsf as usize * dct_size;

    // Per-row buffer width: max plane width across components plus
    // padding. The shim copies `plane_width` samples per row into the
    // caller's buffer (where `plane_width` is the cached MCU-aligned
    // width), so `output_width + 16` is a safe upper bound.
    let max_plane_width: usize = (output_width as usize).max(1) + 16;

    let mut row_bufs: Vec<Vec<Vec<i16>>> = (0..num_components)
        .map(|i| {
            let rows: usize = comp_vsf[i] * dct_size;
            (0..rows).map(|_| vec![0i16; max_plane_width]).collect()
        })
        .collect();
    let mut row_ptrs: Vec<Vec<*mut i16>> = (0..num_components)
        .map(|i| row_bufs[i].iter_mut().map(|r| r.as_mut_ptr()).collect())
        .collect();
    let mut outer_ptrs: Vec<*mut *mut i16> = row_ptrs.iter_mut().map(|v| v.as_mut_ptr()).collect();

    let mut planes: Vec<Vec<i16>> = (0..num_components).map(|_| Vec::new()).collect();

    loop {
        let output_scanline: u32 = read_u32_at(cinfo_bytes, D_OUTPUT_SCANLINE);
        if output_scanline >= output_height {
            break;
        }
        let lines_returned: u32 =
            jpeg12_read_raw_data(cinfo_ptr, outer_ptrs.as_mut_ptr(), rows_per_imcu as u32);
        assert_eq!(
            lines_returned, rows_per_imcu as u32,
            "jpeg12_read_raw_data must return rows_per_imcu ({rows_per_imcu}) lines"
        );
        for comp_idx in 0..num_components {
            let rows_this_imcu: usize = comp_vsf[comp_idx] * dct_size;
            for row_in_imcu in 0..rows_this_imcu {
                planes[comp_idx]
                    .extend_from_slice(&row_bufs[comp_idx][row_in_imcu][..max_plane_width]);
            }
        }
    }

    jpeg_finish_decompress(cinfo_ptr);
    jpeg_destroy_decompress(cinfo_ptr);

    let plane_widths: Vec<usize> = vec![max_plane_width; num_components];
    let plane_heights: Vec<usize> = (0..num_components)
        .map(|i| planes[i].len() / max_plane_width)
        .collect();
    (planes, plane_widths, plane_heights)
}

/// 4:2:0 12-bit YCbCr round-trip exercising both the encoder and
/// decoder shim entry points.
#[test]
fn jpeg12_raw_data_round_trip_4_2_0() {
    use libjpeg_turbo_rs::Subsampling;

    let image_width: usize = 16;
    let image_height: usize = 16;
    let quality: c_int = 90;

    // Synthetic 12-bit (0..4095) Y ramp; mid-gray Cb/Cr at half H/V.
    let y_plane: Vec<i16> = (0..image_height)
        .flat_map(|row| {
            (0..image_width).map(move |col| (((row * image_width + col) * 16) & 0x0FFF) as i16)
        })
        .collect();
    let c_w: usize = image_width.div_ceil(2);
    let c_h: usize = image_height.div_ceil(2);
    let cb_plane: Vec<i16> = vec![2048i16; c_w * c_h];
    let cr_plane: Vec<i16> = vec![2048i16; c_w * c_h];

    let planes_for_shim: Vec<Vec<i16>> = vec![y_plane.clone(), cb_plane.clone(), cr_plane.clone()];
    let plane_widths_shim: Vec<usize> = vec![image_width, c_w, c_w];
    let plane_heights_shim: Vec<usize> = vec![image_height, c_h, c_h];

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    // -----------------------------------------------------------------
    // Gate 1: shim encode → native decode.
    // -----------------------------------------------------------------
    let shim_jpeg: Vec<u8> = unsafe {
        encode_12bit_4_2_0_via_capi(
            &lib,
            &planes_for_shim,
            &plane_widths_shim,
            &plane_heights_shim,
            image_width,
            image_height,
            quality,
        )
    };
    assert!(!shim_jpeg.is_empty(), "shim-encoded JPEG must not be empty");

    let native_decoded: libjpeg_turbo_rs::raw_data_12::RawImage12 =
        libjpeg_turbo_rs::raw_data_12::decompress_raw_12(&shim_jpeg).unwrap_or_else(|e| {
            panic!("decompress_raw_12 of shim-encoded bytes failed: {e}");
        });
    assert_eq!(
        native_decoded.num_components, 3,
        "round-trip: expected 3 components, got {}",
        native_decoded.num_components
    );

    let dec_y_w: usize = native_decoded.plane_widths[0];
    let dec_y: &[i16] = &native_decoded.planes[0];
    let mut max_diff_native: i32 = 0;
    for row in 0..image_height {
        for col in 0..image_width {
            let ref_val: i32 = y_plane[row * image_width + col] as i32;
            let dec_val: i32 = dec_y[row * dec_y_w + col] as i32;
            let diff: i32 = (ref_val - dec_val).abs();
            if diff > max_diff_native {
                max_diff_native = diff;
            }
        }
    }
    // Measured max_diff for this 12-bit ramp at Q=90 4:2:0 ≤ 64
    // empirically. Tolerance set to 96 (~2.3% of full 12-bit range)
    // to absorb DCT rounding plus chroma subsampling smoothing.
    assert!(
        max_diff_native <= 96,
        "shim encode + native decode: Y max_diff={max_diff_native} exceeds tolerance 96"
    );

    // -----------------------------------------------------------------
    // Gate 2: native encode → shim decode.
    // -----------------------------------------------------------------
    let plane_refs: Vec<&[i16]> = planes_for_shim.iter().map(|v| v.as_slice()).collect();
    let native_jpeg: Vec<u8> = libjpeg_turbo_rs::raw_data_12::compress_raw_12(
        &plane_refs,
        &plane_widths_shim,
        &plane_heights_shim,
        image_width,
        image_height,
        quality as u8,
        Subsampling::S420,
    )
    .unwrap_or_else(|e| panic!("compress_raw_12 reference failed: {e}"));

    let (shim_planes, shim_plane_widths, _shim_plane_heights): (
        Vec<Vec<i16>>,
        Vec<usize>,
        Vec<usize>,
    ) = unsafe { decode_12bit_via_capi(&lib, &native_jpeg) };
    assert_eq!(
        shim_planes.len(),
        3,
        "shim decode: expected 3 components, got {}",
        shim_planes.len()
    );

    let native_for_compare: libjpeg_turbo_rs::raw_data_12::RawImage12 =
        libjpeg_turbo_rs::raw_data_12::decompress_raw_12(&native_jpeg).unwrap_or_else(|e| {
            panic!("decompress_raw_12 of native-encoded bytes failed: {e}");
        });
    let n_y_w: usize = native_for_compare.plane_widths[0];
    let n_y: &[i16] = &native_for_compare.planes[0];
    let s_y_w: usize = shim_plane_widths[0];
    let s_y: &[i16] = &shim_planes[0];
    let mut max_diff_shim_vs_native: i32 = 0;
    for row in 0..image_height {
        for col in 0..image_width {
            let n: i32 = n_y[row * n_y_w + col] as i32;
            let s: i32 = s_y[row * s_y_w + col] as i32;
            let d: i32 = (n - s).abs();
            if d > max_diff_shim_vs_native {
                max_diff_shim_vs_native = d;
            }
        }
    }
    assert_eq!(
        max_diff_shim_vs_native, 0,
        "shim decode vs native decode of same JPEG: max_diff={max_diff_shim_vs_native}, expected 0"
    );

    // -----------------------------------------------------------------
    // Gate 3: shim encode → shim decode (end-to-end through both
    // new code paths).
    // -----------------------------------------------------------------
    let (e2e_planes, e2e_plane_widths, _e2e_plane_heights): (
        Vec<Vec<i16>>,
        Vec<usize>,
        Vec<usize>,
    ) = unsafe { decode_12bit_via_capi(&lib, &shim_jpeg) };
    let e2e_y_w: usize = e2e_plane_widths[0];
    let e2e_y: &[i16] = &e2e_planes[0];
    let mut max_diff_e2e: i32 = 0;
    for row in 0..image_height {
        for col in 0..image_width {
            let r: i32 = y_plane[row * image_width + col] as i32;
            let v: i32 = e2e_y[row * e2e_y_w + col] as i32;
            let d: i32 = (r - v).abs();
            if d > max_diff_e2e {
                max_diff_e2e = d;
            }
        }
    }
    assert!(
        max_diff_e2e <= 96,
        "shim encode + shim decode: Y max_diff={max_diff_e2e} exceeds tolerance 96"
    );
}
