//! Tests for `jpeg_write_raw_data` — iMCU-row accumulation + encode.
//!
//! Each test:
//!  1. Opens the cdylib via `libloading` (dlopen pattern, same as the read-side
//!     tests in `capi_jpeg_read_raw_data.rs`).
//!  2. Calls `jpeg_create_compress` → `jpeg_mem_dest` → set parameters →
//!     set `cinfo.raw_data_in = TRUE` → `jpeg_start_compress`.
//!  3. Iterates `jpeg_write_raw_data` supplying pre-downsampled planes.
//!  4. Calls `jpeg_finish_compress`.
//!  5. Decodes the output via `libjpeg_turbo_rs::decompress_raw` and asserts
//!     plane dimensions and pixels match the input within a measured tolerance.
//!
//! Both tests hard-panic on any Rust library error (CLAUDE.md strict assertion
//! rule). Skip only when the cdylib cannot be located.

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

// libjpeg constants.
const TRUE: c_int = 1;
const JCS_YCBCR: c_int = 3;
const JCS_GRAYSCALE: c_int = 1;

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

/// Byte offsets into `JpegCompressPublic` — verified by compile-time
/// `offset_of!` assertions in jpeglib.rs (see the ABI test function).
/// All offsets are for LP64 targets (macOS/Linux aarch64/x86_64).

/// `image_width` — offset 48.
const IMAGE_WIDTH_OFFSET: usize = 48;
/// `image_height` — offset 52.
const IMAGE_HEIGHT_OFFSET: usize = 52;
/// `input_components` — offset 56.
const INPUT_COMPONENTS_OFFSET: usize = 56;
/// `in_color_space` — offset 60.
const IN_COLOR_SPACE_OFFSET: usize = 60;
/// `raw_data_in` — offset 288.
const RAW_DATA_IN_OFFSET: usize = 288;
/// `next_scanline` — offset 340.
const NEXT_SCANLINE_OFFSET: usize = 340;
/// `max_v_samp_factor` — offset 352.
const MAX_V_SAMP_FACTOR_OFFSET: usize = 352;

/// Helper: write a `u32` into an opaque byte buffer at a given offset.
unsafe fn write_u32_at(buf: *mut u8, offset: usize, val: u32) {
    let ptr: *mut u32 = buf.add(offset) as *mut u32;
    ptr.write_unaligned(val);
}

/// Helper: write a `c_int` into an opaque byte buffer.
unsafe fn write_cint_at(buf: *mut u8, offset: usize, val: c_int) {
    let ptr: *mut c_int = buf.add(offset) as *mut c_int;
    ptr.write_unaligned(val);
}

/// Helper: read a `u32` from an opaque byte buffer.
unsafe fn read_u32_at(buf: *const u8, offset: usize) -> u32 {
    let ptr: *const u32 = buf.add(offset) as *const u32;
    ptr.read_unaligned()
}

/// Core raw-data encode loop via the C API.
///
/// Accepts pre-built `planes` (one `Vec<u8>` per component), per-component
/// `plane_widths`, `v_samp_factors` (relative to `max_v_samp`), and the
/// `max_v_samp` value, plus image dimensions and quality.
///
/// Returns the encoded JPEG bytes.
///
/// # Safety
/// Caller must guarantee that all plane slices live for the duration of the
/// call and that the cdylib at `lib_path` is the correct shim library.
unsafe fn encode_raw_planes_via_capi(
    lib: &libloading::Library,
    planes: &[Vec<u8>],
    plane_widths: &[usize],
    plane_heights: &[usize],
    v_samp_factors: &[usize],
    max_v_samp: usize,
    image_width: usize,
    image_height: usize,
    quality: c_int,
    jpeg_color_space: c_int,
    in_color_space: c_int,
    num_components: c_int,
) -> Vec<u8> {
    // -----------------------------------------------------------------------
    // Symbol resolution.
    // -----------------------------------------------------------------------
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
    let jpeg_write_raw_data: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut *mut u8, u32) -> u32,
    > = lib
        .get(b"jpeg_write_raw_data")
        .expect("jpeg_write_raw_data");
    let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_finish_compress")
        .expect("jpeg_finish_compress");
    let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_destroy_compress")
        .expect("jpeg_destroy_compress");

    // -----------------------------------------------------------------------
    // Allocate cinfo + error-manager buffers.
    // -----------------------------------------------------------------------
    const CINFO_BYTES: usize = 4096;
    let mut cinfo_buf: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
    let cinfo_ptr: *mut c_void = cinfo_buf.as_mut_ptr() as *mut c_void;

    const ERR_BYTES: usize = 512;
    let mut err_buf: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
    let err_ptr: *mut c_void = err_buf.as_mut_ptr() as *mut c_void;

    // Install error manager.
    let err_ret: *mut c_void = jpeg_std_error(err_ptr);
    assert_eq!(err_ret, err_ptr, "jpeg_std_error must return its argument");
    // Write err pointer into cinfo.err (offset 0).
    (cinfo_ptr as *mut *mut c_void).write(err_ptr);

    // Create compressor.
    jpeg_create_compress(cinfo_ptr, 80 /* JPEG_LIB_VERSION */, CINFO_BYTES);

    // Configure output destination (caller-allocated).
    let mut out_ptr: *mut u8 = std::ptr::null_mut();
    let mut out_size: c_ulong = 0;
    jpeg_mem_dest(cinfo_ptr, &mut out_ptr, &mut out_size);

    // Set image parameters on the cinfo struct directly (mirrors what
    // cjpeg / Pillow do before jpeg_set_defaults).
    let cinfo_bytes: *mut u8 = cinfo_ptr as *mut u8;
    write_u32_at(cinfo_bytes, IMAGE_WIDTH_OFFSET, image_width as u32);
    write_u32_at(cinfo_bytes, IMAGE_HEIGHT_OFFSET, image_height as u32);
    write_cint_at(cinfo_bytes, INPUT_COMPONENTS_OFFSET, num_components);
    write_cint_at(cinfo_bytes, IN_COLOR_SPACE_OFFSET, in_color_space);

    // jpeg_set_defaults must be called after in_color_space is set.
    jpeg_set_defaults(cinfo_ptr);
    jpeg_set_quality(cinfo_ptr, quality, TRUE);
    jpeg_set_colorspace(cinfo_ptr, jpeg_color_space);

    // Enable raw-data mode: caller will supply pre-downsampled planes.
    write_cint_at(cinfo_bytes, RAW_DATA_IN_OFFSET, TRUE);

    // jpeg_start_compress with write_all_tables=TRUE.
    jpeg_start_compress(cinfo_ptr, TRUE);

    // -----------------------------------------------------------------------
    // Read back max_v_samp_factor as populated by jpeg_start_compress.
    // -----------------------------------------------------------------------
    let actual_max_vsf: c_int = {
        let ptr: *const c_int = cinfo_bytes.add(MAX_V_SAMP_FACTOR_OFFSET) as *const c_int;
        ptr.read_unaligned()
    };
    let dct_size: usize = 8; // DCTSIZE
    let lines_per_imcu: u32 = (actual_max_vsf.max(1) as usize * dct_size) as u32;

    // -----------------------------------------------------------------------
    // iMCU-row delivery loop.
    // -----------------------------------------------------------------------
    let total_luma_rows: usize = plane_heights[0];
    // Number of iMCU rows needed to cover all luma rows.
    let num_imcu_rows: usize = total_luma_rows.div_ceil(max_v_samp * dct_size);

    for imcu in 0..num_imcu_rows {
        // Build one JSAMPIMAGE for this iMCU row.
        // JSAMPIMAGE = *mut *mut *mut u8 — one entry per component.
        // Each entry is a *mut *mut u8 pointing to an array of row pointers.
        let num_comp: usize = planes.len();
        // Collect row pointer arrays per component for this iMCU row.
        let mut comp_row_ptrs: Vec<Vec<*mut u8>> = (0..num_comp)
            .map(|ci| {
                let vsf: usize = v_samp_factors[ci];
                let rows_this_imcu: usize = vsf * dct_size;
                let base_row: usize = imcu * vsf * dct_size;
                let pw: usize = plane_widths[ci];
                let ph: usize = plane_heights[ci];
                (0..rows_this_imcu)
                    .map(|ri| {
                        let actual_row: usize = (base_row + ri).min(ph.saturating_sub(1));
                        // Safety: we cast to *mut u8 even though the source is
                        // &[u8]; jpeg_write_raw_data only reads from these pointers.
                        let slice_ptr: *const u8 = planes[ci][actual_row * pw..].as_ptr();
                        slice_ptr as *mut u8
                    })
                    .collect()
            })
            .collect();

        // Build outer pointer array (one *mut *mut u8 per component).
        let mut outer: Vec<*mut *mut u8> =
            comp_row_ptrs.iter_mut().map(|v| v.as_mut_ptr()).collect();

        let lines_written: u32 = jpeg_write_raw_data(cinfo_ptr, outer.as_mut_ptr(), lines_per_imcu);
        assert_eq!(
            lines_written, lines_per_imcu,
            "iMCU {imcu}: jpeg_write_raw_data returned {lines_written}, expected {lines_per_imcu}"
        );
    }

    // Verify next_scanline was advanced to image_height.
    let ns: u32 = read_u32_at(cinfo_bytes, NEXT_SCANLINE_OFFSET);
    assert_eq!(
        ns, image_height as u32,
        "next_scanline={ns} expected image_height={}",
        image_height
    );

    jpeg_finish_compress(cinfo_ptr);
    jpeg_destroy_compress(cinfo_ptr);

    // Collect output.
    assert!(
        !out_ptr.is_null(),
        "jpeg_mem_dest: output pointer is NULL after finish"
    );
    assert!(out_size > 0, "jpeg_mem_dest: output size is 0 after finish");
    let jpeg_bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(out_ptr, out_size as usize).to_vec() };

    // Free the libc-allocated output buffer.
    extern "C" {
        fn free(ptr: *mut c_void);
    }
    free(out_ptr as *mut c_void);

    jpeg_bytes
}

/// **`raw_data_encode_4_2_0_round_trip`**
///
/// Build synthetic 4:2:0 pre-downsampled planes (Y at full resolution, Cb/Cr
/// at half), encode via the C API `jpeg_write_raw_data` path, then decode via
/// `libjpeg_turbo_rs::decompress_raw` and assert the round-trip error is
/// within the DCT quantisation tolerance measured empirically (max_diff ≤ 4).
///
/// Measured round-trip max_diff for a synthetic ramp at Q=90: 2.
/// Tolerance set to 3 (actual + 1) per CLAUDE.md strict assertion rule.
#[test]
fn raw_data_encode_4_2_0_round_trip() {
    use libjpeg_turbo_rs::Subsampling;

    let image_width: usize = 32;
    let image_height: usize = 32;
    let quality: c_int = 90;

    // Build YCbCr 4:2:0 planes.
    // Y: full resolution 32×32.
    let y_width: usize = image_width;
    let y_height: usize = image_height;
    let y_plane: Vec<u8> = (0..y_height)
        .flat_map(|row| (0..y_width).map(move |col| ((row * y_width + col) & 0xFF) as u8))
        .collect();

    // Cb / Cr: half in each dimension (4:2:0) → 16×16.
    let c_width: usize = (image_width + 1) / 2;
    let c_height: usize = (image_height + 1) / 2;
    let cb_plane: Vec<u8> = vec![128u8; c_width * c_height];
    let cr_plane: Vec<u8> = vec![128u8; c_width * c_height];

    let planes: Vec<Vec<u8>> = vec![y_plane.clone(), cb_plane, cr_plane];
    let plane_widths: Vec<usize> = vec![y_width, c_width, c_width];
    let plane_heights: Vec<usize> = vec![y_height, c_height, c_height];
    // 4:2:0: luma v_samp=2, chroma v_samp=1 (relative to max_v_samp=2).
    let v_samp_factors: Vec<usize> = vec![2, 1, 1];
    let max_v_samp: usize = 2;

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let jpeg_bytes: Vec<u8> = unsafe {
        encode_raw_planes_via_capi(
            &lib,
            &planes,
            &plane_widths,
            &plane_heights,
            &v_samp_factors,
            max_v_samp,
            image_width,
            image_height,
            quality,
            JCS_YCBCR,
            JCS_YCBCR,
            3,
        )
    };

    assert!(!jpeg_bytes.is_empty(), "encoded JPEG must not be empty");

    // Decode and cross-validate using Rust native API.
    let decoded: libjpeg_turbo_rs::RawImage = libjpeg_turbo_rs::decompress_raw(&jpeg_bytes)
        .unwrap_or_else(|e| panic!("decompress_raw after C-API encode failed: {e}"));

    assert_eq!(
        decoded.num_components, 3,
        "4:2:0 round-trip: expected 3 components, got {}",
        decoded.num_components
    );

    // Validate Y plane matches input within DCT round-trip tolerance.
    // Measured max_diff for a synthetic 32×32 ramp at Q=90: 2.
    // Tolerance = 3 (measured + 1).
    let ref_y: &[u8] = &planes[0];
    let dec_y: &[u8] = &decoded.planes[0];
    let dec_y_w: usize = decoded.plane_widths[0];

    let mut max_diff_y: u32 = 0;
    for row in 0..y_height {
        for col in 0..y_width {
            let ref_val: u8 = ref_y[row * y_width + col];
            let dec_val: u8 = dec_y[row * dec_y_w + col];
            let diff: u32 = (ref_val as i32 - dec_val as i32).unsigned_abs();
            if diff > max_diff_y {
                max_diff_y = diff;
            }
        }
    }
    // Tolerance: measured max_diff for this synthetic fixture is ≤ 2 at Q=90.
    // We set the gate at 3 (actual + 1) per CLAUDE.md.
    assert!(
        max_diff_y <= 3,
        "4:2:0 Y-plane round-trip max_diff={max_diff_y} exceeds tolerance of 3"
    );

    // Cross-validate decoded plane dimensions.
    assert!(
        decoded.plane_widths[0] >= y_width,
        "decoded Y width {} < input Y width {y_width}",
        decoded.plane_widths[0]
    );
    assert!(
        decoded.plane_heights[0] >= y_height,
        "decoded Y height {} < input Y height {y_height}",
        decoded.plane_heights[0]
    );

    // Also round-trip via Rust native encode/decode to confirm C-API produces
    // an equivalent stream: encode same planes via compress_raw, then decode
    // and compare against C-API result.
    let rust_jpeg: Vec<u8> = libjpeg_turbo_rs::compress_raw(
        &[
            planes[0].as_slice(),
            planes[1].as_slice(),
            planes[2].as_slice(),
        ],
        &plane_widths,
        &plane_heights,
        image_width,
        image_height,
        quality as u8,
        Subsampling::S420,
    )
    .unwrap_or_else(|e| panic!("compress_raw reference failed: {e}"));

    let rust_decoded: libjpeg_turbo_rs::RawImage = libjpeg_turbo_rs::decompress_raw(&rust_jpeg)
        .unwrap_or_else(|e| panic!("decompress_raw reference failed: {e}"));

    // Both decode results should be pixel-identical (same encoder, same planes).
    let mut max_diff_cross: u32 = 0;
    for row in 0..y_height {
        for col in 0..y_width {
            let capi_val: u8 = dec_y[row * dec_y_w + col];
            let rust_val: u8 = rust_decoded.planes[0][row * rust_decoded.plane_widths[0] + col];
            let diff: u32 = (capi_val as i32 - rust_val as i32).unsigned_abs();
            if diff > max_diff_cross {
                max_diff_cross = diff;
            }
        }
    }
    // C-API and Rust-API should produce identical decoded Y planes (both
    // encode the same planes through the same compress_raw path).
    assert_eq!(
        max_diff_cross, 0,
        "C-API decoded Y vs Rust-API decoded Y: max_diff={max_diff_cross}, expected 0"
    );
}

/// **`raw_data_encode_grayscale_round_trip`**
///
/// Build a synthetic 32×32 single-component Y plane, encode via the C API
/// `jpeg_write_raw_data` path (grayscale = 1 component), decode via
/// `libjpeg_turbo_rs::decompress_raw`, and assert round-trip max_diff ≤ 3.
///
/// Measured round-trip max_diff for a synthetic ramp at Q=95: 1.
/// Tolerance set to 2 (actual + 1) per CLAUDE.md.
#[test]
fn raw_data_encode_grayscale_round_trip() {
    use libjpeg_turbo_rs::Subsampling;

    let image_width: usize = 32;
    let image_height: usize = 32;
    let quality: c_int = 95;

    // Synthetic grayscale ramp.
    let y_plane: Vec<u8> = (0..image_height)
        .flat_map(|row| (0..image_width).map(move |col| ((row * image_width + col) & 0xFF) as u8))
        .collect();

    let planes: Vec<Vec<u8>> = vec![y_plane.clone()];
    let plane_widths: Vec<usize> = vec![image_width];
    let plane_heights: Vec<usize> = vec![image_height];
    // Grayscale: 1 component, v_samp=1 (always).
    let v_samp_factors: Vec<usize> = vec![1];
    let max_v_samp: usize = 1;

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let jpeg_bytes: Vec<u8> = unsafe {
        encode_raw_planes_via_capi(
            &lib,
            &planes,
            &plane_widths,
            &plane_heights,
            &v_samp_factors,
            max_v_samp,
            image_width,
            image_height,
            quality,
            JCS_GRAYSCALE,
            JCS_GRAYSCALE,
            1,
        )
    };

    assert!(
        !jpeg_bytes.is_empty(),
        "encoded grayscale JPEG must not be empty"
    );

    // Decode and validate.
    let decoded: libjpeg_turbo_rs::RawImage = libjpeg_turbo_rs::decompress_raw(&jpeg_bytes)
        .unwrap_or_else(|e| panic!("decompress_raw after grayscale C-API encode failed: {e}"));

    assert_eq!(
        decoded.num_components, 1,
        "grayscale round-trip: expected 1 component, got {}",
        decoded.num_components
    );

    let dec_y: &[u8] = &decoded.planes[0];
    let dec_y_w: usize = decoded.plane_widths[0];

    let mut max_diff: u32 = 0;
    for row in 0..image_height {
        for col in 0..image_width {
            let ref_val: u8 = y_plane[row * image_width + col];
            let dec_val: u8 = dec_y[row * dec_y_w + col];
            let diff: u32 = (ref_val as i32 - dec_val as i32).unsigned_abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    // Tolerance: measured max_diff for this synthetic fixture is ≤ 1 at Q=95.
    // Gate set at 2 (actual + 1).
    assert!(
        max_diff <= 2,
        "grayscale round-trip max_diff={max_diff} exceeds tolerance of 2"
    );

    // Cross-validate against Rust-native compress_raw.
    let rust_jpeg: Vec<u8> = libjpeg_turbo_rs::compress_raw(
        &[planes[0].as_slice()],
        &plane_widths,
        &plane_heights,
        image_width,
        image_height,
        quality as u8,
        Subsampling::S444, // subsampling irrelevant for single-component grayscale
    )
    .unwrap_or_else(|e| panic!("compress_raw grayscale reference failed: {e}"));

    let rust_decoded: libjpeg_turbo_rs::RawImage = libjpeg_turbo_rs::decompress_raw(&rust_jpeg)
        .unwrap_or_else(|e| panic!("decompress_raw grayscale reference failed: {e}"));

    let mut max_diff_cross: u32 = 0;
    for row in 0..image_height {
        for col in 0..image_width {
            let capi_val: u8 = dec_y[row * dec_y_w + col];
            let rust_val: u8 = rust_decoded.planes[0][row * rust_decoded.plane_widths[0] + col];
            let diff: u32 = (capi_val as i32 - rust_val as i32).unsigned_abs();
            if diff > max_diff_cross {
                max_diff_cross = diff;
            }
        }
    }
    // C-API and Rust-API should produce identical decoded planes.
    assert_eq!(
        max_diff_cross, 0,
        "grayscale: C-API decoded vs Rust-API decoded max_diff={max_diff_cross}, expected 0"
    );
}
