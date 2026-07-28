//! Tests for `jpeg_read_raw_data` — iMCU-row delivery of raw component planes.
//!
//! Each test:
//!  1. Opens the cdylib via `libloading` (dlopen pattern).
//!  2. Calls `jpeg_create_decompress` → `jpeg_mem_src` → `jpeg_read_header`.
//!  3. Sets `cinfo.raw_data_out = TRUE`, calls `jpeg_start_decompress`.
//!  4. Iterates `jpeg_read_raw_data` until `output_scanline >= output_height`.
//!  5. Cross-validates the collected planes against
//!     `libjpeg_turbo_rs::decompress_raw(same_bytes)`.
//!
//! Both tests hard-panic on any Rust library error (CLAUDE.md strict
//! assertion rule). Skip only when a fixture file is absent (submodule
//! not initialised).

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

// libjpeg return codes.
const JPEG_HEADER_OK: c_int = 1;
// libjpeg `TRUE` / `FALSE`.
const TRUE: c_int = 1;

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

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// Byte offsets into `JpegDecompressPublic` — verified by the compile-time
// `offset_of!` assertions and a one-shot runtime print in jpeglib.rs.
// These values are for LP64 targets (macOS/Linux aarch64/x86_64).

/// `raw_data_out` — asserted by ABI test in jpeglib.rs.
const RAW_DATA_OUT_OFFSET: usize = 92;

/// `output_scanline` — measured: 168.
const OUTPUT_SCANLINE_OFFSET: usize = 168;

/// `output_height` — measured: 140.
const OUTPUT_HEIGHT_OFFSET: usize = 140;

/// `output_width` — measured: 136.
const OUTPUT_WIDTH_OFFSET: usize = 136;

/// `max_v_samp_factor` — measured: 412.
const MAX_V_SAMP_FACTOR_OFFSET: usize = 412;

/// `min_DCT_v_scaled_size` — measured: 420.
const MIN_DCT_V_SCALED_SIZE_OFFSET: usize = 420;

/// `num_components` — asserted by ABI test: 56.
const NUM_COMPONENTS_OFFSET: usize = 56;

/// `comp_info` pointer — asserted by ABI test: 304.
const COMP_INFO_OFFSET: usize = 304;

/// `v_samp_factor` within `JpegComponentInfoPublic` — measured: 12.
const COMP_VSF_FIELD_OFFSET: usize = 12;

/// Size of `JpegComponentInfoPublic` — measured: 96.
const COMP_INFO_STRUCT_SIZE: usize = 96;

/// Helper: read a `u32` from an opaque byte buffer at a given byte offset.
unsafe fn read_u32_at(buf: *const u8, offset: usize) -> u32 {
    let ptr: *const u32 = buf.add(offset) as *const u32;
    ptr.read_unaligned()
}

/// Helper: read a pointer-sized value from an opaque byte buffer.
unsafe fn read_ptr_at(buf: *const u8, offset: usize) -> *mut u8 {
    let ptr: *const *mut u8 = buf.add(offset) as *const *mut u8;
    ptr.read_unaligned()
}

/// Helper: read a `c_int` from an opaque byte buffer.
unsafe fn read_cint_at(buf: *const u8, offset: usize) -> c_int {
    let ptr: *const c_int = buf.add(offset) as *const c_int;
    ptr.read_unaligned()
}

/// Core raw-data decode loop via the C API.
///
/// Returns `(planes, plane_widths, plane_heights, num_components)` where
/// `planes[i]` contains all rows for component `i` in natural order.
///
/// # Safety
/// Caller must ensure `jpeg_bytes` lives for the duration of the call.
unsafe fn collect_raw_planes_via_capi(
    lib: &libloading::Library,
    jpeg_bytes: &[u8],
) -> (Vec<Vec<u8>>, Vec<usize>, Vec<usize>, usize) {
    // -----------------------------------------------------------------------
    // Symbol resolution.
    // -----------------------------------------------------------------------
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
    let jpeg_read_raw_data: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *mut *mut *mut u8, u32) -> u32,
    > = lib.get(b"jpeg_read_raw_data").expect("jpeg_read_raw_data");
    let jpeg_finish_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
        lib.get(b"jpeg_finish_decompress")
            .expect("jpeg_finish_decompress");
    let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_destroy_decompress")
        .expect("jpeg_destroy_decompress");

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
    (cinfo_ptr as *mut *mut c_void).write(err_ptr);

    // Create decompressor.
    jpeg_create_decompress(cinfo_ptr, 80 /* JPEG_LIB_VERSION */, CINFO_BYTES);

    // Attach in-memory source.
    jpeg_mem_src(cinfo_ptr, jpeg_bytes.as_ptr(), jpeg_bytes.len() as c_ulong);

    // Parse header.
    let rc: c_int = jpeg_read_header(cinfo_ptr, 1);
    assert_eq!(rc, JPEG_HEADER_OK, "jpeg_read_header must succeed");

    // Set raw_data_out = TRUE before jpeg_start_decompress.
    let cinfo_bytes: *mut u8 = cinfo_ptr as *mut u8;
    cinfo_bytes
        .add(RAW_DATA_OUT_OFFSET)
        .cast::<c_int>()
        .write_unaligned(TRUE);

    // Calculate output dimensions (updates max_v_samp_factor etc.).
    jpeg_calc_output_dimensions(cinfo_ptr);

    // Start decompress.
    let rc: c_int = jpeg_start_decompress(cinfo_ptr);
    assert_eq!(rc, 1, "jpeg_start_decompress must succeed");

    // -----------------------------------------------------------------------
    // Read dimension / sampling info from the public struct.
    // -----------------------------------------------------------------------
    let output_height: u32 = read_u32_at(cinfo_bytes, OUTPUT_HEIGHT_OFFSET);
    let max_vsf: c_int = read_cint_at(cinfo_bytes, MAX_V_SAMP_FACTOR_OFFSET);
    let min_dct_v: c_int = read_cint_at(cinfo_bytes, MIN_DCT_V_SCALED_SIZE_OFFSET);
    let num_components: usize = read_cint_at(cinfo_bytes, NUM_COMPONENTS_OFFSET) as usize;
    let comp_info_raw: *mut u8 = read_ptr_at(cinfo_bytes, COMP_INFO_OFFSET);

    assert!(
        num_components > 0,
        "num_components must be > 0 after jpeg_read_header"
    );
    assert!(
        max_vsf > 0,
        "max_v_samp_factor must be > 0 after jpeg_calc_output_dimensions"
    );

    // Collect per-component v_samp_factor values.
    let mut comp_vsf: Vec<usize> = Vec::with_capacity(num_components);
    for i in 0..num_components {
        let vsf: c_int = if comp_info_raw.is_null() {
            // Fallback: assume all components have max_vsf for a grayscale
            // stream, or max_vsf / 2 for chroma in 4:2:0.
            if i == 0 {
                max_vsf
            } else {
                (max_vsf + 1) / 2
            }
        } else {
            let comp_base: *const u8 = comp_info_raw.add(i * COMP_INFO_STRUCT_SIZE);
            read_cint_at(comp_base, COMP_VSF_FIELD_OFFSET)
        };
        comp_vsf.push(vsf.max(1) as usize);
    }

    // Use min_DCT_v_scaled_size when available (>0), fall back to DCTSIZE=8.
    let dct_size: usize = if min_dct_v > 0 { min_dct_v as usize } else { 8 };
    let rows_per_imcu: usize = max_vsf as usize * dct_size;

    // -----------------------------------------------------------------------
    // Compute plane dimensions: height = ceil(image_height * vsf /
    // max_vsf), aligned to dct_size * vsf. Width: read from comp_info
    // if available, otherwise use output_width from cinfo.
    // -----------------------------------------------------------------------
    // We derive plane widths by collecting them from the Rust-level
    // `decompress_raw` result pre-call rather than reading width_in_blocks
    // here — the test validates against that result anyway.  For the
    // allocation we use a generous upper bound: output_width * max_vsf
    // rounded up to 8.
    let output_width: u32 = read_u32_at(cinfo_bytes, OUTPUT_WIDTH_OFFSET);

    // Allocate accumulation buffers (one Vec per component).
    let mut planes: Vec<Vec<u8>> = (0..num_components).map(|_| Vec::new()).collect();

    // Allocate row-pointer arenas for one iMCU row (max component rows).
    // Each component needs `comp_vsf[i] * dct_size` row pointers and
    // matching row buffers. We allocate the maximum possible width.
    let max_plane_width: usize = (output_width as usize).max(1) + 16;
    // Row storage: [num_components][max_rows_per_imcu][max_plane_width].
    let mut row_bufs: Vec<Vec<Vec<u8>>> = (0..num_components)
        .map(|i| {
            let rows: usize = comp_vsf[i] * dct_size;
            (0..rows).map(|_| vec![0u8; max_plane_width]).collect()
        })
        .collect();
    // Row-pointer arrays for each component: &mut *mut u8.
    let mut row_ptrs: Vec<Vec<*mut u8>> = (0..num_components)
        .map(|i| row_bufs[i].iter_mut().map(|r| r.as_mut_ptr()).collect())
        .collect();
    // Outer pointer array: one *mut *mut u8 per component.
    let mut outer_ptrs: Vec<*mut *mut u8> = row_ptrs.iter_mut().map(|v| v.as_mut_ptr()).collect();

    // -----------------------------------------------------------------------
    // iMCU-row delivery loop.
    // -----------------------------------------------------------------------
    loop {
        let output_scanline: u32 = read_u32_at(cinfo_bytes, OUTPUT_SCANLINE_OFFSET);
        if output_scanline >= output_height {
            break;
        }

        let lines_returned: u32 =
            jpeg_read_raw_data(cinfo_ptr, outer_ptrs.as_mut_ptr(), rows_per_imcu as u32);
        assert_eq!(
            lines_returned, rows_per_imcu as u32,
            "jpeg_read_raw_data must return rows_per_imcu ({rows_per_imcu}) lines"
        );

        // Harvest rows: for each component, read `comp_vsf[i] * dct_size`
        // rows. Keep all rows including MCU padding — this matches what
        // the C API actually delivers and what `decompress_raw` returns
        // (both are MCU-aligned per upstream's raw-data contract).
        for comp_idx in 0..num_components {
            let rows_this_imcu: usize = comp_vsf[comp_idx] * dct_size;
            for row_in_imcu in 0..rows_this_imcu {
                let src: &[u8] = &row_bufs[comp_idx][row_in_imcu][..max_plane_width];
                planes[comp_idx].extend_from_slice(src);
            }
        }
    }

    jpeg_finish_decompress(cinfo_ptr);
    jpeg_destroy_decompress(cinfo_ptr);

    // Compute actual plane widths (plane bytes / rows = plane_width).
    let plane_heights_out: Vec<usize> = (0..num_components)
        .map(|i| planes[i].len() / max_plane_width)
        .collect();

    (
        planes,
        vec![max_plane_width; num_components],
        plane_heights_out,
        num_components,
    )
}

/// **`raw_data_decode_4_2_0_matches_upstream`**
///
/// Load `references/libjpeg-turbo/testimages/testorig.jpg` (8-bit 4:2:0
/// baseline), decode via `jpeg_read_raw_data`, and cross-validate against
/// `libjpeg_turbo_rs::decompress_raw`. Asserts:
/// - `num_components` matches.
/// - Per-plane pixel-exact equality (max_diff == 0) for the samples that
///   fall within the true (non-padded) plane dimensions.
#[test]
fn raw_data_decode_4_2_0_matches_upstream() {
    let fixture: PathBuf =
        manifest_dir().join("../../references/libjpeg-turbo/testimages/testorig.jpg");
    if !fixture.exists() {
        eprintln!(
            "SKIP: testorig.jpg not found at {} — submodule not initialised",
            fixture.display()
        );
        return;
    }
    let jpeg_bytes: Vec<u8> = std::fs::read(&fixture)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", fixture.display()));

    // Rust-native reference decode.
    let reference: libjpeg_turbo_rs::RawImage = libjpeg_turbo_rs::decompress_raw(&jpeg_bytes)
        .unwrap_or_else(|e| panic!("libjpeg_turbo_rs::decompress_raw failed: {e}"));

    assert!(
        reference.num_components > 0,
        "reference must have > 0 components"
    );

    // C-API decode via the shim.
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let (capi_planes, capi_widths, capi_heights, capi_ncomp) =
        unsafe { collect_raw_planes_via_capi(&lib, &jpeg_bytes) };

    assert_eq!(
        capi_ncomp, reference.num_components,
        "num_components mismatch: C-API={capi_ncomp} reference={}",
        reference.num_components
    );

    // Cross-validate each component plane.
    for comp_idx in 0..reference.num_components {
        let ref_width: usize = reference.plane_widths[comp_idx];
        let ref_height: usize = reference.plane_heights[comp_idx];
        let capi_w: usize = capi_widths[comp_idx];
        let capi_h: usize = capi_heights[comp_idx];

        assert!(
            capi_h >= ref_height,
            "comp[{comp_idx}] C-API height {capi_h} < reference height {ref_height}"
        );
        assert!(
            capi_w >= ref_width,
            "comp[{comp_idx}] C-API width {capi_w} < reference width {ref_width}"
        );

        let ref_plane: &[u8] = &reference.planes[comp_idx];
        let capi_plane: &[u8] = &capi_planes[comp_idx];

        let mut max_diff: u32 = 0;
        for row in 0..ref_height {
            for col in 0..ref_width {
                let ref_val: u8 = ref_plane[row * ref_width + col];
                let capi_val: u8 = capi_plane[row * capi_w + col];
                let diff: u32 = (ref_val as i32 - capi_val as i32).unsigned_abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        // Target: pixel-identical (max_diff == 0).
        assert_eq!(
            max_diff, 0,
            "comp[{comp_idx}]: pixel mismatch between C-API and Rust decompress_raw \
             (max_diff={max_diff}); expected 0 (bit-exact)"
        );
    }
}

/// **`raw_data_decode_grayscale`**
///
/// Encode a small synthetic 8-bit grayscale JPEG via
/// `libjpeg_turbo_rs::compress_raw` (single-component), then decode
/// it via the C API `jpeg_read_raw_data` and cross-validate against
/// `libjpeg_turbo_rs::decompress_raw`. Single-component path.
///
/// We create the fixture inline so the test runs even without an
/// external grayscale file.
#[test]
fn raw_data_decode_grayscale() {
    use libjpeg_turbo_rs::Subsampling;

    // Synthetic 32x32 grayscale ramp.
    let width: usize = 32;
    let height: usize = 32;
    let plane: Vec<u8> = (0..height)
        .flat_map(|row| (0..width).map(move |col| ((row * width + col) & 0xFF) as u8))
        .collect();

    // Encode via Rust native API (quality 95 to keep near-lossless).
    let jpeg_bytes: Vec<u8> = libjpeg_turbo_rs::compress_raw(
        &[plane.as_slice()],
        &[width],
        &[height],
        width,
        height,
        95,
        Subsampling::S444, // subsampling is irrelevant for a single-component grayscale stream
    )
    .unwrap_or_else(|e| panic!("compress_raw grayscale failed: {e}"));

    // Rust-native reference decode.
    let reference: libjpeg_turbo_rs::RawImage = libjpeg_turbo_rs::decompress_raw(&jpeg_bytes)
        .unwrap_or_else(|e| panic!("decompress_raw reference failed: {e}"));
    assert_eq!(
        reference.num_components, 1,
        "grayscale must have 1 component"
    );

    // C-API decode via the shim.
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let (capi_planes, capi_widths, capi_heights, capi_ncomp) =
        unsafe { collect_raw_planes_via_capi(&lib, &jpeg_bytes) };

    assert_eq!(
        capi_ncomp, 1,
        "grayscale: C-API reported {capi_ncomp} components, expected 1"
    );

    let ref_width: usize = reference.plane_widths[0];
    let ref_height: usize = reference.plane_heights[0];
    let capi_w: usize = capi_widths[0];
    let capi_h: usize = capi_heights[0];

    assert!(
        capi_h >= ref_height,
        "grayscale C-API height {capi_h} < reference height {ref_height}"
    );
    assert!(
        capi_w >= ref_width,
        "grayscale C-API width {capi_w} < reference width {ref_width}"
    );

    let ref_plane: &[u8] = &reference.planes[0];
    let capi_plane: &[u8] = &capi_planes[0];

    let mut max_diff: u32 = 0;
    for row in 0..ref_height {
        for col in 0..ref_width {
            let ref_val: u8 = ref_plane[row * ref_width + col];
            let capi_val: u8 = capi_plane[row * capi_w + col];
            let diff: u32 = (ref_val as i32 - capi_val as i32).unsigned_abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    // Grayscale Q=95 round-trip via compress_raw → decompress_raw: the
    // test validates C-API vs Rust-API, both decoding the same JPEG bytes,
    // so both results must be identical (max_diff == 0).
    assert_eq!(
        max_diff, 0,
        "grayscale: pixel mismatch between C-API and Rust decompress_raw \
         (max_diff={max_diff}); expected 0 (bit-exact)"
    );
}
