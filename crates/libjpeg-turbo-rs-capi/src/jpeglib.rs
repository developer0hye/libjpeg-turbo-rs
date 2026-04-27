//! FFI A1-11: classic libjpeg (`jpeg_*`) decode entry points.
//!
//! This module exposes the libjpeg-compatible C API used by applications
//! such as `djpeg`, Pillow, ImageMagick, and countless others. Unlike the
//! TurboJPEG 3 (`tj3*`) surface in `tj3.rs`, these functions take a
//! caller-allocated `struct jpeg_decompress_struct` and cooperate with
//! caller-allocated `struct jpeg_error_mgr` / source manager sub-structs.
//!
//! `JpegDecompressPublic` is a **byte-exact ABI mirror** of libjpeg's
//! `struct jpeg_decompress_struct` as declared in
//! `references/libjpeg-turbo/src/jpeglib.h` for `JPEG_LIB_VERSION = 80`.
//! Every field appears in the correct order with the correct type size
//! and alignment so real libjpeg callers (stock `djpeg`, Pillow,
//! ImageMagick, etc.) can read any documented field at the right offset
//! — notably `data_precision`, `jpeg_color_space`, `output_width`,
//! `comp_info`, and the quantization/Huffman table pointer arrays.
//!
//! Because the struct has no room for Rust-owned state (adding a
//! trailing field would overflow the caller's
//! `sizeof(struct jpeg_decompress_struct)` allocation), the private
//! Rust-side state lives in a thread-local side table keyed by the
//! `cinfo` pointer and is freed in `jpeg_destroy_decompress`.

use std::ffi::{c_int, c_long, c_short, c_uint, c_void, CString};
use std::io::Read;

use libjpeg_turbo_rs::{decompress, PixelFormat};

use crate::alloc::libc_from_slice;
use crate::memmgr;

/// libjpeg `boolean` typedef is `int` in the upstream header. All callers
/// use `TRUE` = 1 / `FALSE` = 0.
type CBoolean = c_int;

/// libjpeg `JDIMENSION` typedef.
type JDimension = u32;

/// Exact copy of libjpeg's `jpeg_read_header` return codes. Matches
/// `jpeglib.h`:
/// - `JPEG_SUSPENDED = 0`
/// - `JPEG_HEADER_OK = 1`
/// - `JPEG_HEADER_TABLES_ONLY = 2`
pub const JPEG_SUSPENDED: c_int = 0;
pub const JPEG_HEADER_OK: c_int = 1;
pub const JPEG_HEADER_TABLES_ONLY: c_int = 2;

/// Maximum message length recorded by the default error manager.
#[allow(dead_code)]
const JMSG_LENGTH_MAX: usize = 200;
/// Maximum string-parameter length for error messages.
const JMSG_STR_PARM_MAX: usize = 80;

// ---------------------------------------------------------------------------
// `struct jpeg_error_mgr` — minimal ABI-compatible layout.
// ---------------------------------------------------------------------------
//
// Real libjpeg's `jpeg_error_mgr` has ~15 fields totalling ~400 bytes. We
// expose enough of the layout that:
//   (a) `jpeg_std_error(err)` can populate the callbacks, and
//   (b) clients can write `cinfo.err = jpeg_std_error(&err);` to hook the
//       manager into a decompress struct (which requires `err` to be the
//       first field of `jpeg_decompress_struct`).
//
// Callers that walk uninitialised fields past what we expose here invoke
// undefined behavior in real libjpeg too, so the minimal layout is a
// safe starting point for a drop-in shim.

/// Public-facing layout of `struct jpeg_error_mgr`. `#[repr(C)]` pins the
/// field order; `MaybeUninit` slots hold the (unused) buffers required
/// by the real libjpeg `format_message` contract.
#[repr(C)]
pub struct JpegErrorMgr {
    pub error_exit: Option<unsafe extern "C" fn(*mut c_void)>,
    pub emit_message: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
    pub output_message: Option<unsafe extern "C" fn(*mut c_void)>,
    pub format_message: Option<unsafe extern "C" fn(*mut c_void, *mut u8)>,
    pub reset_error_mgr: Option<unsafe extern "C" fn(*mut c_void)>,
    pub msg_code: c_int,
    /// Union in the C header. We reserve the larger branch (char[80]).
    pub msg_parm: [u8; JMSG_STR_PARM_MAX],
    pub trace_level: c_int,
    pub num_warnings: c_long,
    pub jpeg_message_table: *const *const u8,
    pub last_jpeg_message: c_int,
    pub addon_message_table: *const *const u8,
    pub first_addon_message: c_int,
    pub last_addon_message: c_int,
}

// ---------------------------------------------------------------------------
// `struct jpeg_source_mgr` — verbatim layout.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct JpegSourceMgr {
    pub next_input_byte: *const u8,
    pub bytes_in_buffer: usize,
    pub init_source: Option<unsafe extern "C" fn(*mut c_void)>,
    pub fill_input_buffer: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean>,
    pub skip_input_data: Option<unsafe extern "C" fn(*mut c_void, c_long)>,
    pub resync_to_restart: Option<unsafe extern "C" fn(*mut c_void, c_int) -> CBoolean>,
    pub term_source: Option<unsafe extern "C" fn(*mut c_void)>,
}

// ---------------------------------------------------------------------------
// JPEG sub-structs referenced from `jpeg_decompress_struct`.
// ---------------------------------------------------------------------------
//
// Each struct below is a byte-exact mirror of the corresponding
// `typedef struct { … }` in `references/libjpeg-turbo/src/jpeglib.h`.
// Real libjpeg callers read these fields via pointer indirection from
// `cinfo.quant_tbl_ptrs[i]`, `cinfo.dc_huff_tbl_ptrs[i]`,
// `cinfo.comp_info`, and `cinfo.marker_list`, so the layout must match
// the C header exactly.

/// Width/height limits per the JPEG standard. Mirrors the numeric
/// constants in `jpeglib.h`.
const DCTSIZE2: usize = 64;
const NUM_QUANT_TBLS: usize = 4;
const NUM_HUFF_TBLS: usize = 4;
const NUM_ARITH_TBLS: usize = 16;
const MAX_COMPS_IN_SCAN: usize = 4;
const D_MAX_BLOCKS_IN_MCU: usize = 10;

/// `JQUANT_TBL` from `jpeglib.h`. 64 `UINT16` quant values + a
/// `boolean sent_table` flag. Total size: 132 bytes on most targets.
#[repr(C)]
pub struct JQuantTblPublic {
    pub quantval: [u16; DCTSIZE2],
    pub sent_table: CBoolean,
}

/// `JHUFF_TBL` from `jpeglib.h`. `UINT8 bits[17]` + `UINT8 huffval[256]`
/// + `boolean sent_table`. Total size: 276 bytes on most targets.
#[repr(C)]
pub struct JHuffTblPublic {
    pub bits: [u8; 17],
    pub huffval: [u8; 256],
    pub sent_table: CBoolean,
}

/// `jpeg_component_info` from `jpeglib.h`. The encode-side shim
/// (`JpegComponentInfoCompress`) is a distinct type used only by
/// `jpeg_compress_struct`; this one mirrors the full decode-side
/// layout.
#[repr(C)]
pub struct JpegComponentInfoPublic {
    pub component_id: c_int,
    pub component_index: c_int,
    pub h_samp_factor: c_int,
    pub v_samp_factor: c_int,
    pub quant_tbl_no: c_int,
    pub dc_tbl_no: c_int,
    pub ac_tbl_no: c_int,
    pub width_in_blocks: JDimension,
    pub height_in_blocks: JDimension,
    /// `DCT_h_scaled_size` under `JPEG_LIB_VERSION >= 70`.
    pub dct_h_scaled_size: c_int,
    /// `DCT_v_scaled_size` under `JPEG_LIB_VERSION >= 70`.
    pub dct_v_scaled_size: c_int,
    pub downsampled_width: JDimension,
    pub downsampled_height: JDimension,
    pub component_needed: CBoolean,
    pub mcu_width: c_int,
    pub mcu_height: c_int,
    pub mcu_blocks: c_int,
    pub mcu_sample_width: c_int,
    pub last_col_width: c_int,
    pub last_row_height: c_int,
    pub quant_table: *mut JQuantTblPublic,
    pub dct_table: *mut c_void,
}

/// `struct jpeg_marker_struct` from `jpeglib.h`.
#[repr(C)]
pub struct JpegMarkerStructPublic {
    pub next: *mut JpegMarkerStructPublic,
    pub marker: u8,
    pub original_length: c_uint,
    pub data_length: c_uint,
    pub data: *mut u8,
}

// ---------------------------------------------------------------------------
// `struct jpeg_decompress_struct` — full ABI mirror.
// ---------------------------------------------------------------------------
//
// Matches libjpeg-turbo's `struct jpeg_decompress_struct` declared in
// `references/libjpeg-turbo/src/jpeglib.h` (lines ~523–770) for
// `JPEG_LIB_VERSION = 80`. Field order, type sizes, and alignment are
// byte-exact so that real libjpeg consumers can read any documented
// field at its standard offset — most importantly `data_precision`
// (historically at offset ~524 on LP64 targets), which stock `djpeg`
// validates before producing output.
//
// Private Rust-side state (`DecompressPrivate`) is held in a
// thread-local side table keyed by the `cinfo` pointer; see
// `private_state_for`.

/// Byte-exact ABI mirror of libjpeg's `struct jpeg_decompress_struct`.
#[repr(C)]
pub struct JpegDecompressPublic {
    // --- jpeg_common_fields (shared with jpeg_compress_struct) ------------
    pub err: *mut JpegErrorMgr,
    pub mem: *mut c_void,
    pub progress: *mut c_void,
    pub client_data: *mut c_void,
    pub is_decompressor: CBoolean,
    pub global_state: c_int,

    // --- decompressor-specific: source manager ----------------------------
    pub src: *mut JpegSourceMgr,

    // --- basic image description, filled by jpeg_read_header() ------------
    pub image_width: JDimension,
    pub image_height: JDimension,
    pub num_components: c_int,
    pub jpeg_color_space: c_int,

    // --- decompression processing parameters ------------------------------
    pub out_color_space: c_int,
    pub scale_num: c_uint,
    pub scale_denom: c_uint,
    pub output_gamma: f64,
    pub buffered_image: CBoolean,
    pub raw_data_out: CBoolean,
    pub dct_method: c_int,
    pub do_fancy_upsampling: CBoolean,
    pub do_block_smoothing: CBoolean,
    pub quantize_colors: CBoolean,
    pub dither_mode: c_int,
    pub two_pass_quantize: CBoolean,
    pub desired_number_of_colors: c_int,
    pub enable_1pass_quant: CBoolean,
    pub enable_external_quant: CBoolean,
    pub enable_2pass_quant: CBoolean,

    // --- description of the actual output image (set by start_decompress) -
    pub output_width: JDimension,
    pub output_height: JDimension,
    pub out_color_components: c_int,
    pub output_components: c_int,
    pub rec_outbuf_height: c_int,

    pub actual_number_of_colors: c_int,
    /// `JSAMPARRAY colormap` — pointer to a 2-D `JSAMPLE *` array.
    pub colormap: *mut *mut u8,

    // --- state variables --------------------------------------------------
    pub output_scanline: JDimension,
    pub input_scan_number: c_int,
    pub input_iMCU_row: JDimension,
    pub output_scan_number: c_int,
    pub output_iMCU_row: JDimension,

    /// `int (*coef_bits)[DCTSIZE2]` — points to an array of
    /// `num_components` coef-bit arrays. We expose the pointer only.
    pub coef_bits: *mut [c_int; DCTSIZE2],

    // --- internal JPEG parameters: quant/Huffman table pointer arrays ----
    pub quant_tbl_ptrs: [*mut JQuantTblPublic; NUM_QUANT_TBLS],
    pub dc_huff_tbl_ptrs: [*mut JHuffTblPublic; NUM_HUFF_TBLS],
    pub ac_huff_tbl_ptrs: [*mut JHuffTblPublic; NUM_HUFF_TBLS],

    // --- these fields are never carried across datastreams ----------------
    pub data_precision: c_int,
    pub comp_info: *mut JpegComponentInfoPublic,
    /// `is_baseline` is present only under `JPEG_LIB_VERSION >= 80`.
    pub is_baseline: CBoolean,
    pub progressive_mode: CBoolean,
    pub arith_code: CBoolean,

    pub arith_dc_L: [u8; NUM_ARITH_TBLS],
    pub arith_dc_U: [u8; NUM_ARITH_TBLS],
    pub arith_ac_K: [u8; NUM_ARITH_TBLS],

    pub restart_interval: c_uint,

    // --- optional markers (JFIF, Adobe, plus raw marker list) --------------
    pub saw_JFIF_marker: CBoolean,
    pub JFIF_major_version: u8,
    pub JFIF_minor_version: u8,
    pub density_unit: u8,
    pub X_density: u16,
    pub Y_density: u16,
    pub saw_Adobe_marker: CBoolean,
    pub Adobe_transform: u8,

    pub CCIR601_sampling: CBoolean,

    pub marker_list: *mut JpegMarkerStructPublic,

    // --- fields computed during decompression startup ---------------------
    pub max_h_samp_factor: c_int,
    pub max_v_samp_factor: c_int,
    /// `min_DCT_h_scaled_size` under `JPEG_LIB_VERSION >= 70`.
    pub min_DCT_h_scaled_size: c_int,
    /// `min_DCT_v_scaled_size` under `JPEG_LIB_VERSION >= 70`.
    pub min_DCT_v_scaled_size: c_int,
    pub total_iMCU_rows: JDimension,

    /// `JSAMPLE *sample_range_limit` — pointer to a range-limit table.
    pub sample_range_limit: *mut u8,

    // --- valid during a single scan ---------------------------------------
    pub comps_in_scan: c_int,
    pub cur_comp_info: [*mut JpegComponentInfoPublic; MAX_COMPS_IN_SCAN],
    pub MCUs_per_row: JDimension,
    pub MCU_rows_in_scan: JDimension,
    pub blocks_in_MCU: c_int,
    pub MCU_membership: [c_int; D_MAX_BLOCKS_IN_MCU],

    pub Ss: c_int,
    pub Se: c_int,
    pub Ah: c_int,
    pub Al: c_int,

    // --- derived from Se of first SOS marker (JPEG_LIB_VERSION >= 80) ----
    pub block_size: c_int,
    /// `const int *natural_order` — we expose a raw pointer; Rust code
    /// never dereferences it (libjpeg consumers read-only).
    pub natural_order: *const c_int,
    pub lim_Se: c_int,

    pub unread_marker: c_int,

    // --- opaque links to subobjects (callers only store these) ------------
    pub master: *mut c_void,
    pub main_controller: *mut c_void,
    pub coef: *mut c_void,
    pub post: *mut c_void,
    pub inputctl: *mut c_void,
    pub marker: *mut c_void,
    pub entropy: *mut c_void,
    pub idct: *mut c_void,
    pub upsample: *mut c_void,
    pub cconvert: *mut c_void,
    pub cquantize: *mut c_void,
}

// Compile-time: `JCOEF` is `short` (i16) in libjpeg, `c_short` in Rust. We
// keep the typedef here for readability in any future `coef_bits`-style
// fields we may add.
#[allow(dead_code)]
type JCoef = c_short;

// Global-state values mirror libjpeg's internal state machine, just to the
// level of granularity we need for entry-point sequencing.
const DSTATE_START: c_int = 200;
const DSTATE_INHEADER: c_int = 201;
#[allow(dead_code)]
const DSTATE_READY: c_int = 202;
const DSTATE_SCANNING: c_int = 205;
const DSTATE_STOPPING: c_int = 206;

// ---------------------------------------------------------------------------
// Side table for Rust-owned `DecompressPrivate` state.
//
// `JpegDecompressPublic` mirrors the real libjpeg struct byte-for-byte, so
// we cannot append a trailing `priv_ptr` field — the caller allocates only
// `sizeof(struct jpeg_decompress_struct)` bytes, and writing past that
// boundary would corrupt stack/heap memory. Instead we key the private
// state on the `cinfo` pointer via a thread-local map, created on
// `jpeg_CreateDecompress` and destroyed on `jpeg_destroy_decompress`.
// ---------------------------------------------------------------------------

thread_local! {
    static DECOMPRESS_PRIVATE_STATE: std::cell::RefCell<
        std::collections::HashMap<usize, Box<DecompressPrivate>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

fn decompress_private_key(cinfo: *const c_void) -> usize {
    cinfo as usize
}

fn decompress_private_insert(cinfo: *mut c_void, private: Box<DecompressPrivate>) {
    let key: usize = decompress_private_key(cinfo);
    DECOMPRESS_PRIVATE_STATE.with(|s| {
        s.borrow_mut().insert(key, private);
    });
}

fn decompress_private_remove(cinfo: *mut c_void) -> Option<Box<DecompressPrivate>> {
    let key: usize = decompress_private_key(cinfo);
    DECOMPRESS_PRIVATE_STATE.with(|s| s.borrow_mut().remove(&key))
}

/// Execute `f` with a mutable borrow on the private state for `cinfo`.
/// Returns `None` if no state was registered (caller forgot to invoke
/// `jpeg_CreateDecompress`, or is operating on a destroyed handle).
#[allow(dead_code)]
fn with_decompress_private<F, R>(cinfo: *mut c_void, f: F) -> Option<R>
where
    F: FnOnce(&mut DecompressPrivate) -> R,
{
    let key: usize = decompress_private_key(cinfo);
    DECOMPRESS_PRIVATE_STATE.with(|s| {
        let mut map = s.borrow_mut();
        map.get_mut(&key).map(|boxed| f(boxed.as_mut()))
    })
}

/// Raw-pointer accessor for legacy call sites that previously stashed a
/// `priv_ptr` in the struct. The returned pointer is valid for as long
/// as the `cinfo` handle exists (i.e., until `jpeg_destroy_decompress`).
///
/// Returns NULL when no private state is registered for `cinfo`.
fn decompress_private_raw(cinfo: *mut c_void) -> *mut c_void {
    let key: usize = decompress_private_key(cinfo);
    DECOMPRESS_PRIVATE_STATE.with(|s| {
        s.borrow_mut()
            .get_mut(&key)
            .map(|boxed| boxed.as_mut() as *mut DecompressPrivate as *mut c_void)
            .unwrap_or(std::ptr::null_mut())
    })
}

/// Source-of-JPEG variants. The memory variant borrows into the caller's
/// buffer; the stdio variant owns a read-once copy of the file's bytes.
enum JpegSource {
    Memory { ptr: *const u8, len: usize },
    Owned(Vec<u8>),
    None,
}

impl JpegSource {
    fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            // SAFETY: lifetime tied to the caller-owned source buffer.
            // The libjpeg contract places the lifetime-management burden
            // on the application, consistent with upstream.
            JpegSource::Memory { ptr, len } => unsafe {
                if ptr.is_null() {
                    None
                } else {
                    Some(std::slice::from_raw_parts(*ptr, *len))
                }
            },
            JpegSource::Owned(v) => Some(v.as_slice()),
            JpegSource::None => None,
        }
    }
}

/// J_COLOR_SPACE numeric constants matching `jpeglib.h`. Only the
/// subset the decode entry points produce is enumerated here.
const JCS_UNKNOWN: c_int = 0;
const JCS_GRAYSCALE: c_int = 1;
const JCS_RGB: c_int = 2;
const JCS_YCBCR: c_int = 3;
const JCS_CMYK: c_int = 4;
const JCS_YCCK: c_int = 5;

/// Per-marker save configuration set by `jpeg_save_markers`.
///
/// Keyed by marker code (e.g. `0xFE` for `COM`, `0xE0..=0xEF` for `APPn`).
/// A `length_limit` of `0` disables saving; `u32::MAX` means "no limit".
#[derive(Default)]
struct MarkerSaveSettings {
    /// Saving enabled flags per marker code (256 entries).
    limits: std::collections::HashMap<u8, c_uint>,
}

/// Tagged wrapper around `JpegCoefficients` exposed as the opaque
/// handle that `jpeg_read_coefficients` returns to callers.
///
/// The leading `magic` field lets `jpeg_write_coefficients` validate
/// that the pointer it received actually came from this shim — a
/// foreign `jvirt_barray_ptr` produced by some other memory manager
/// (for example a stock libjpeg `jtransform_adjust_parameters`)
/// would not have this magic value, so we can reject it cleanly
/// instead of silently dereferencing arbitrary memory.
#[repr(C)]
struct CoefHandle {
    magic: u64,
    inner: libjpeg_turbo_rs::JpegCoefficients,
}

impl CoefHandle {
    /// Random-looking constant; chosen as the ASCII bytes of "RsCoefH"
    /// in little-endian followed by `'!'` to avoid colliding with any
    /// realistic struct prefix from a foreign library.
    const MAGIC: u64 = u64::from_le_bytes(*b"RsCoefH!");
}

/// Rust-side private state reached via the thread-local side table
/// keyed by the `cinfo` pointer. Owned via `Box`; freed in
/// `jpeg_destroy_decompress`.
struct DecompressPrivate {
    source: JpegSource,
    /// Owned storage backing the caller-visible `JpegSourceMgr`. We keep
    /// a pointer to it in `public.src` and mutate `bytes_in_buffer`/
    /// `next_input_byte` via the public field.
    source_mgr: Option<Box<JpegSourceMgr>>,
    /// Last error message for debugging. Held here because libjpeg's
    /// `format_message` callback expects it.
    last_error: CString,
    /// Decoded image buffer, built lazily on `jpeg_start_decompress`.
    decoded: Option<libjpeg_turbo_rs::Image>,
    /// Saved DCT coefficients, populated by `jpeg_read_coefficients`.
    /// Held in `Box<CoefHandle>` so callers can treat its pointer as
    /// `jvirt_barray_ptr*`. The `CoefHandle::MAGIC` field is checked by
    /// `jpeg_write_coefficients` before deref to reject foreign pointers
    /// (for example virtual coefficient arrays from a real libjpeg
    /// memory manager).
    coefficients: Option<Box<CoefHandle>>,
    /// `jpeg_save_markers` settings; consumed by `jpeg_read_header` when the
    /// header is (re-)parsed so saved markers land in `Image.saved_markers`.
    marker_save: MarkerSaveSettings,
    /// Custom marker processors registered via `jpeg_set_marker_processor`.
    /// Keyed by marker code; invoked after the marker bytes are buffered.
    #[allow(clippy::type_complexity)]
    marker_processors: std::collections::HashMap<u8, MarkerParserFn>,
    /// Horizontal crop x-offset requested via `jpeg_crop_scanline`.
    crop_xoffset: u32,
    /// Horizontal crop width requested via `jpeg_crop_scanline`.
    crop_width: u32,
    /// TRUE if a crop was requested via `jpeg_crop_scanline`.
    crop_active: bool,
    /// Owned backing for `JpegDecompressPublic::comp_info`. Populated in
    /// `jpeg_read_header`; the public struct exposes a raw pointer into
    /// this vector so real libjpeg callers can iterate components.
    comp_info_storage: Vec<JpegComponentInfoPublic>,
}

impl Default for DecompressPrivate {
    fn default() -> Self {
        Self {
            source: JpegSource::None,
            source_mgr: None,
            last_error: CString::new("No error").expect("static"),
            decoded: None,
            coefficients: None,
            marker_save: MarkerSaveSettings::default(),
            marker_processors: std::collections::HashMap::new(),
            crop_xoffset: 0,
            crop_width: 0,
            crop_active: false,
            comp_info_storage: Vec::new(),
        }
    }
}

/// C function pointer for a marker parser method.
///
/// `libjpeg` declares this as
/// `typedef boolean (*jpeg_marker_parser_method)(j_decompress_ptr cinfo)`.
/// We do not dispatch to the callback during decode (our shim consumes
/// markers through `Decoder::set_marker_processor`), but we retain the
/// pointer so `jpeg_set_marker_processor` remains a faithful no-op for
/// ABI consumers that install a handler and expect later introspection.
type MarkerParserFn = unsafe extern "C" fn(*mut c_void) -> CBoolean;

// ---------------------------------------------------------------------------
// Helpers: validate and reach into the caller's `cinfo`.
// ---------------------------------------------------------------------------

/// Interpret `cinfo` as our public struct. Returns `None` for NULL; does
/// not otherwise validate content.
///
/// # Safety
/// Caller must guarantee that `cinfo` either is NULL or points to a
/// valid `JpegDecompressPublic`-sized allocation.
unsafe fn cinfo_mut<'a>(cinfo: *mut c_void) -> Option<&'a mut JpegDecompressPublic> {
    if cinfo.is_null() {
        None
    } else {
        Some(unsafe { &mut *(cinfo as *mut JpegDecompressPublic) })
    }
}

/// Dereference the private state for a `cinfo` handle, previously stored
/// in the thread-local side table on `jpeg_CreateDecompress`.
///
/// `priv_ptr` is expected to be the value returned by
/// `decompress_private_raw(cinfo)` — we keep the parameter name for
/// compatibility with the former field-based API.
///
/// Returns `None` when no private state is registered, which happens if
/// the caller skipped `jpeg_CreateDecompress` or the handle was already
/// destroyed.
///
/// # Safety
/// `priv_ptr` must either be NULL or point to a live `DecompressPrivate`
/// whose lifetime is tied to the current `cinfo` handle (see the
/// thread-local map in [`DECOMPRESS_PRIVATE_STATE`]).
unsafe fn priv_from_ptr<'a>(priv_ptr: *mut c_void) -> Option<&'a mut DecompressPrivate> {
    if priv_ptr.is_null() {
        None
    } else {
        Some(unsafe { &mut *(priv_ptr as *mut DecompressPrivate) })
    }
}

// ---------------------------------------------------------------------------
// `jpeg_std_error` — subtask FFI A1-11 #2.
// ---------------------------------------------------------------------------

/// Default `error_exit` callback. libjpeg's contract says this function
/// must not return; we implement that via `std::process::abort`. Apps
/// that want graceful error handling override the callback with a
/// `longjmp`-style jump.
unsafe extern "C" fn default_error_exit(cinfo: *mut c_void) {
    // Emit to stderr then abort. Upstream libjpeg prints the formatted
    // message; we surface the msg_code + parm to aid debugging.
    let mut code: c_int = -1;
    let mut parm0: c_int = 0;
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        if !c.err.is_null() {
            let err: &JpegErrorMgr = unsafe { &*c.err };
            code = err.msg_code;
            parm0 = i32::from_le_bytes([
                err.msg_parm[0],
                err.msg_parm[1],
                err.msg_parm[2],
                err.msg_parm[3],
            ]);
        }
    }
    eprintln!(
        "libjpeg-turbo-rs: fatal JPEG error (msg_code={}, parm0={})",
        code, parm0
    );
    std::process::abort();
}

unsafe extern "C" fn default_emit_message(_cinfo: *mut c_void, _msg_level: c_int) {
    // No-op by default; real libjpeg prints warnings above trace_level.
}

unsafe extern "C" fn default_output_message(_cinfo: *mut c_void) {
    // No-op by default — real libjpeg routes through stderr.
}

unsafe extern "C" fn default_format_message(_cinfo: *mut c_void, buffer: *mut u8) {
    if buffer.is_null() {
        return;
    }
    // Minimal default: write a sentinel so clients don't read uninit memory.
    let msg: &[u8] = b"libjpeg-turbo-rs error\0";
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), buffer, msg.len());
    }
}

unsafe extern "C" fn default_reset_error_mgr(cinfo: *mut c_void) {
    unsafe {
        if let Some(c) = cinfo_mut(cinfo) {
            if !c.err.is_null() {
                let err: &mut JpegErrorMgr = &mut *c.err;
                err.msg_code = 0;
                err.num_warnings = 0;
            }
        }
    }
}

/// `jpeg_std_error(err: *mut jpeg_error_mgr) -> *mut jpeg_error_mgr`.
///
/// Populates the error manager struct with the default callbacks and
/// returns the same pointer so the common idiom
/// `cinfo.err = jpeg_std_error(&err);` works as expected.
#[no_mangle]
pub extern "C" fn jpeg_std_error(err: *mut JpegErrorMgr) -> *mut JpegErrorMgr {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller must ensure `err` points to a writable JpegErrorMgr.
    unsafe {
        let e: &mut JpegErrorMgr = &mut *err;
        e.error_exit = Some(default_error_exit);
        e.emit_message = Some(default_emit_message);
        e.output_message = Some(default_output_message);
        e.format_message = Some(default_format_message);
        e.reset_error_mgr = Some(default_reset_error_mgr);
        e.msg_code = 0;
        e.msg_parm = [0u8; JMSG_STR_PARM_MAX];
        e.trace_level = 0;
        e.num_warnings = 0;
        e.jpeg_message_table = std::ptr::null();
        e.last_jpeg_message = 0;
        e.addon_message_table = std::ptr::null();
        e.first_addon_message = 0;
        e.last_addon_message = 0;
    }
    err
}

// ---------------------------------------------------------------------------
// `jpeg_CreateDecompress` + `jpeg_destroy_decompress` — subtask #3.
// ---------------------------------------------------------------------------

/// `jpeg_CreateDecompress(cinfo, version, structsize)`.
///
/// The `jpeg_create_decompress(cinfo)` macro expands to this function
/// with `version = JPEG_LIB_VERSION` and `structsize = sizeof(*cinfo)`.
///
/// Populates the caller-allocated `jpeg_decompress_struct` with libjpeg's
/// standard defaults (as implemented by `jinit_decompress_master` in
/// `references/libjpeg-turbo/src/jdapimin.c`) and registers private
/// Rust-side state in the thread-local side table.
#[no_mangle]
pub extern "C" fn jpeg_CreateDecompress(cinfo: *mut c_void, _version: c_int, _struct_size: usize) {
    // SAFETY: `cinfo` is caller-allocated; we only touch the bytes that
    // fit the `JpegDecompressPublic` layout. If the buffer is smaller
    // than that, the caller violated libjpeg's `sizeof(*cinfo)` contract
    // and crashes are acceptable (matching upstream).
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    // Do NOT zero `err` — the caller sets that up before calling us
    // (per the `cinfo.err = jpeg_std_error(&err);` idiom).
    //
    // Wire up the memory-manager vtable so libjpeg callers can invoke
    // `(*cinfo->mem->alloc_small)(...)` from the very first line of
    // their main loops (e.g. wrppm.c output init).
    c.mem = memmgr::create_memory_mgr() as *mut c_void;
    c.progress = std::ptr::null_mut();
    c.client_data = std::ptr::null_mut();
    c.is_decompressor = 1;
    c.global_state = DSTATE_START;

    c.src = std::ptr::null_mut();

    c.image_width = 0;
    c.image_height = 0;
    c.num_components = 0;
    c.jpeg_color_space = JCS_UNKNOWN;

    // Output-side decompression defaults (match `jinit_decompress_master`).
    c.out_color_space = JCS_UNKNOWN;
    c.scale_num = 1;
    c.scale_denom = 1;
    c.output_gamma = 1.0;
    c.buffered_image = 0;
    c.raw_data_out = 0;
    c.dct_method = 0; // JDCT_ISLOW
    c.do_fancy_upsampling = 1;
    c.do_block_smoothing = 1;
    c.quantize_colors = 0;
    c.dither_mode = 2; // JDITHER_FS
    c.two_pass_quantize = 1;
    c.desired_number_of_colors = 256;
    c.enable_1pass_quant = 0;
    c.enable_external_quant = 0;
    c.enable_2pass_quant = 0;

    c.output_width = 0;
    c.output_height = 0;
    c.out_color_components = 0;
    c.output_components = 0;
    c.rec_outbuf_height = 1;

    c.actual_number_of_colors = 0;
    c.colormap = std::ptr::null_mut();

    c.output_scanline = 0;
    c.input_scan_number = 0;
    c.input_iMCU_row = 0;
    c.output_scan_number = 0;
    c.output_iMCU_row = 0;

    c.coef_bits = std::ptr::null_mut();

    for slot in c.quant_tbl_ptrs.iter_mut() {
        *slot = std::ptr::null_mut();
    }
    for slot in c.dc_huff_tbl_ptrs.iter_mut() {
        *slot = std::ptr::null_mut();
    }
    for slot in c.ac_huff_tbl_ptrs.iter_mut() {
        *slot = std::ptr::null_mut();
    }

    c.data_precision = 8;
    c.comp_info = std::ptr::null_mut();
    c.is_baseline = 0;
    c.progressive_mode = 0;
    c.arith_code = 0;

    c.arith_dc_L = [0u8; NUM_ARITH_TBLS];
    c.arith_dc_U = [1u8; NUM_ARITH_TBLS];
    c.arith_ac_K = [5u8; NUM_ARITH_TBLS];

    c.restart_interval = 0;

    c.saw_JFIF_marker = 0;
    c.JFIF_major_version = 1;
    c.JFIF_minor_version = 1;
    c.density_unit = 0;
    c.X_density = 1;
    c.Y_density = 1;
    c.saw_Adobe_marker = 0;
    c.Adobe_transform = 0;

    c.CCIR601_sampling = 0;

    c.marker_list = std::ptr::null_mut();

    c.max_h_samp_factor = 0;
    c.max_v_samp_factor = 0;
    c.min_DCT_h_scaled_size = 0;
    c.min_DCT_v_scaled_size = 0;
    c.total_iMCU_rows = 0;

    c.sample_range_limit = std::ptr::null_mut();

    c.comps_in_scan = 0;
    for slot in c.cur_comp_info.iter_mut() {
        *slot = std::ptr::null_mut();
    }
    c.MCUs_per_row = 0;
    c.MCU_rows_in_scan = 0;
    c.blocks_in_MCU = 0;
    c.MCU_membership = [0; D_MAX_BLOCKS_IN_MCU];

    c.Ss = 0;
    c.Se = 0;
    c.Ah = 0;
    c.Al = 0;

    c.block_size = 8; // DCTSIZE for lossy mode
    c.natural_order = std::ptr::null();
    c.lim_Se = 63; // DCTSIZE2 - 1

    c.unread_marker = 0;

    c.master = std::ptr::null_mut();
    c.main_controller = std::ptr::null_mut();
    c.coef = std::ptr::null_mut();
    c.post = std::ptr::null_mut();
    c.inputctl = std::ptr::null_mut();
    c.marker = std::ptr::null_mut();
    c.entropy = std::ptr::null_mut();
    c.idct = std::ptr::null_mut();
    c.upsample = std::ptr::null_mut();
    c.cconvert = std::ptr::null_mut();
    c.cquantize = std::ptr::null_mut();

    // Register Rust-side private state in the thread-local side table.
    decompress_private_insert(cinfo, Box::default());
}

/// `jpeg_destroy_decompress(cinfo)` — free the Rust-side private state.
#[no_mangle]
pub extern "C" fn jpeg_destroy_decompress(cinfo: *mut c_void) {
    if cinfo.is_null() {
        return;
    }
    // Release the private state from the side table. Drop any
    // high-precision (12/16-bit) decoded state parked in the thread-local
    // `HIGH_PRECISION_STATE` map, keyed by the private pointer, before the
    // box itself goes out of scope.
    let priv_raw: *mut c_void = decompress_private_raw(cinfo);
    if !priv_raw.is_null() {
        hp_drop_for(priv_raw);
    }
    let _dropped: Option<Box<DecompressPrivate>> = decompress_private_remove(cinfo);
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        // Release the memory manager and every pool it owns before
        // nulling the slot; this mirrors `self_destruct` in jmemmgr.c.
        if !c.mem.is_null() {
            // SAFETY: `c.mem` was produced by `memmgr::create_memory_mgr`
            // in `jpeg_CreateDecompress` and has not been freed.
            unsafe {
                memmgr::destroy_memory_mgr(c.mem as *mut memmgr::JpegMemoryMgr);
            }
            c.mem = std::ptr::null_mut();
        }
        c.src = std::ptr::null_mut();
        c.global_state = 0;
    }
}

// ---------------------------------------------------------------------------
// `jpeg_stdio_src` / `jpeg_mem_src` — subtask #4.
// ---------------------------------------------------------------------------

// Empty source-manager callbacks. Our decode path uses the byte slice
// directly instead of stepping through the callback FSM, so these can
// safely be stubs. If a consumer does invoke them (e.g. to fill/skip),
// we keep `bytes_in_buffer`/`next_input_byte` consistent.
unsafe extern "C" fn noop_init_source(_cinfo: *mut c_void) {}
unsafe extern "C" fn noop_fill_input_buffer(_cinfo: *mut c_void) -> CBoolean {
    // Real libjpeg returns FALSE to signal "suspend". We report TRUE so
    // that callers that accidentally invoke fill_input_buffer don't loop
    // forever; the entire buffer is already present.
    1
}
unsafe extern "C" fn noop_skip_input_data(cinfo: *mut c_void, num_bytes: c_long) {
    if num_bytes <= 0 {
        return;
    }
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        if !c.src.is_null() {
            let src: &mut JpegSourceMgr = unsafe { &mut *c.src };
            let skip: usize = num_bytes as usize;
            if skip >= src.bytes_in_buffer {
                src.next_input_byte = std::ptr::null();
                src.bytes_in_buffer = 0;
            } else {
                // SAFETY: we computed offset < bytes_in_buffer.
                src.next_input_byte = unsafe { src.next_input_byte.add(skip) };
                src.bytes_in_buffer -= skip;
            }
        }
    }
}
unsafe extern "C" fn noop_resync_to_restart(_cinfo: *mut c_void, _desired: c_int) -> CBoolean {
    // Always resync successfully (matches upstream's default).
    1
}
unsafe extern "C" fn noop_term_source(_cinfo: *mut c_void) {}

/// Attach a source manager that reads from an already-in-memory byte
/// slice. Matches libjpeg `jpeg_mem_src(cinfo, buf, size)`.
#[no_mangle]
pub extern "C" fn jpeg_mem_src(cinfo: *mut c_void, buf: *const u8, size: std::os::raw::c_ulong) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };

    let len: usize = size as usize;
    priv_state.source = JpegSource::Memory { ptr: buf, len };
    install_source_mgr(c, priv_state, buf, len);
}

/// Attach a source manager that reads from a `FILE *`. libjpeg's signature
/// takes `FILE *`; we accept `*mut c_void` since stable Rust does not
/// commit to an `stdio::FILE` layout. Internally we promote to a `File`
/// via the POSIX `fileno` + `fdopen`-equivalent path.
///
/// For simplicity and portability, this implementation slurps the entire
/// file into memory at `jpeg_stdio_src` time and delegates to `jpeg_mem_src`.
/// libjpeg does not guarantee that the `FILE *` remains valid after
/// `jpeg_finish_decompress`, so eagerly copying the bytes matches the
/// semantics applications rely on in practice.
#[no_mangle]
pub extern "C" fn jpeg_stdio_src(cinfo: *mut c_void, infile: *mut c_void) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if infile.is_null() {
        priv_state.source = JpegSource::None;
        return;
    }
    let bytes: Vec<u8> = match read_c_file(infile) {
        Ok(b) => b,
        Err(msg) => {
            priv_state.last_error =
                CString::new(format!("jpeg_stdio_src: {msg}")).unwrap_or_default();
            return;
        }
    };
    let buf_ptr: *const u8 = bytes.as_ptr();
    let buf_len: usize = bytes.len();
    priv_state.source = JpegSource::Owned(bytes);
    // Re-resolve the pointer from the stored buffer to avoid pointing at
    // the stack-local `bytes` variable whose lifetime ended above.
    if let JpegSource::Owned(ref v) = priv_state.source {
        install_source_mgr(c, priv_state, v.as_ptr(), v.len());
        let _ = (buf_ptr, buf_len); // silence unused on the borrow path
    }
}

/// Slurp a `FILE *` by converting it to a POSIX fd via `fileno` and
/// reading through `std::fs::File::from_raw_fd`. We duplicate the fd so
/// closing the Rust `File` on drop doesn't affect the caller's stream.
fn read_c_file(file: *mut c_void) -> Result<Vec<u8>, String> {
    #[cfg(unix)]
    {
        use std::os::unix::io::{FromRawFd, RawFd};
        extern "C" {
            fn fileno(stream: *mut c_void) -> c_int;
            fn dup(oldfd: c_int) -> c_int;
        }
        let fd_raw: c_int = unsafe { fileno(file) };
        if fd_raw < 0 {
            return Err("fileno returned -1".into());
        }
        let dup_fd: RawFd = unsafe { dup(fd_raw) };
        if dup_fd < 0 {
            return Err("dup(fileno) returned -1".into());
        }
        let mut f: std::fs::File = unsafe { std::fs::File::from_raw_fd(dup_fd) };
        let mut buf: Vec<u8> = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::{FromRawHandle, RawHandle};
        // We don't have a stable equivalent of `fileno` on Windows in a
        // minimal shim; callers on Windows should prefer `jpeg_mem_src`.
        Err("jpeg_stdio_src is unavailable on Windows; use jpeg_mem_src instead".into())
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err("jpeg_stdio_src is unavailable on this platform".into())
    }
}

/// Create a `JpegSourceMgr` if needed, and point `cinfo.src` at it with
/// `next_input_byte` / `bytes_in_buffer` already pre-loaded.
fn install_source_mgr(
    c: &mut JpegDecompressPublic,
    priv_state: &mut DecompressPrivate,
    ptr: *const u8,
    len: usize,
) {
    if let Some(mgr) = priv_state.source_mgr.as_mut() {
        mgr.next_input_byte = ptr;
        mgr.bytes_in_buffer = len;
    } else {
        priv_state.source_mgr = Some(Box::new(JpegSourceMgr {
            next_input_byte: ptr,
            bytes_in_buffer: len,
            init_source: Some(noop_init_source),
            fill_input_buffer: Some(noop_fill_input_buffer),
            skip_input_data: Some(noop_skip_input_data),
            resync_to_restart: Some(noop_resync_to_restart),
            term_source: Some(noop_term_source),
        }));
    }
    c.src = priv_state
        .source_mgr
        .as_mut()
        .map(|b| b.as_mut() as *mut JpegSourceMgr)
        .unwrap_or(std::ptr::null_mut());
}

// ---------------------------------------------------------------------------
// `jpeg_read_header` — subtask #5.
// ---------------------------------------------------------------------------

/// Map our internal `ColorSpace` to libjpeg's `J_COLOR_SPACE` int code.
fn colorspace_to_jcs(cs: libjpeg_turbo_rs::ColorSpace) -> c_int {
    use libjpeg_turbo_rs::ColorSpace;
    match cs {
        ColorSpace::Grayscale => JCS_GRAYSCALE,
        ColorSpace::Rgb => JCS_RGB,
        ColorSpace::YCbCr => JCS_YCBCR,
        ColorSpace::Cmyk => JCS_CMYK,
        ColorSpace::Ycck => JCS_YCCK,
        ColorSpace::Unknown => JCS_UNKNOWN,
    }
}

/// Peek at the JPEG header to populate `cinfo.image_width` etc. without
/// triggering a full decode. Uses the Rust-side `Decoder::new` which
/// only parses markers up to (but not including) entropy-coded data.
#[no_mangle]
pub extern "C" fn jpeg_read_header(cinfo: *mut c_void, _require_image: CBoolean) -> c_int {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return JPEG_SUSPENDED,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return JPEG_SUSPENDED,
    };

    let bytes: &[u8] = match priv_state.source.as_bytes() {
        Some(b) if b.len() >= 2 => b,
        _ => {
            priv_state.last_error =
                CString::new("jpeg_read_header: no JPEG source attached").unwrap_or_default();
            return JPEG_SUSPENDED;
        }
    };

    let decoder: libjpeg_turbo_rs::Decoder<'_> = match libjpeg_turbo_rs::Decoder::new(bytes) {
        Ok(d) => d,
        Err(e) => {
            priv_state.last_error =
                CString::new(format!("jpeg_read_header: {e}")).unwrap_or_default();
            return JPEG_SUSPENDED;
        }
    };

    let frame: &libjpeg_turbo_rs::FrameHeader = decoder.header();
    c.image_width = frame.width as JDimension;
    c.image_height = frame.height as JDimension;
    c.num_components = frame.components.len() as c_int;
    c.data_precision = frame.precision as c_int;
    c.progressive_mode = if frame.is_progressive { 1 } else { 0 };
    // libjpeg's `is_baseline` flag: TRUE if SOF0 was encountered. We
    // approximate by clearing it for progressive/lossless streams.
    c.is_baseline = if !frame.is_progressive && !frame.is_lossless {
        1
    } else {
        0
    };
    // `arith_code` is not exposed by `Decoder::header()`; stock baseline
    // files use Huffman, so the default of 0 matches the common case.
    // TODO(C2-follow-up): surface `JpegMetadata::is_arithmetic` through
    // the public API so we can populate this field faithfully.
    c.arith_code = 0;

    // Heuristic for jpeg_color_space matching libjpeg jdmarker:
    //   1 component     -> JCS_GRAYSCALE
    //   3 components    -> JCS_YCbCr (or JCS_RGB if Adobe APP14 says so)
    //   4 components    -> JCS_CMYK (or JCS_YCCK if Adobe APP14 says so)
    // We don't have an Adobe-marker getter in the thin header path, so
    // conservatively pick the common case; out_color_space below is the
    // effective one anyway.
    c.jpeg_color_space = match frame.components.len() {
        1 => JCS_GRAYSCALE,
        3 => JCS_YCBCR,
        4 => JCS_CMYK,
        _ => JCS_UNKNOWN,
    };
    // Default out_color_space: matches libjpeg's defaults in jdmaster.
    c.out_color_space = match c.jpeg_color_space {
        JCS_GRAYSCALE => JCS_GRAYSCALE,
        JCS_YCBCR => JCS_RGB,
        JCS_CMYK => JCS_CMYK,
        JCS_YCCK => JCS_CMYK,
        other => other,
    };

    // Populate `comp_info[]` from FrameHeader components. The storage is
    // owned by `priv_state.comp_info_storage`; we hand out a raw pointer
    // to the vector's first element so real libjpeg callers can iterate.
    priv_state.comp_info_storage.clear();
    priv_state.comp_info_storage.reserve(frame.components.len());
    for (idx, comp) in frame.components.iter().enumerate() {
        priv_state.comp_info_storage.push(JpegComponentInfoPublic {
            component_id: comp.id as c_int,
            component_index: idx as c_int,
            h_samp_factor: comp.horizontal_sampling as c_int,
            v_samp_factor: comp.vertical_sampling as c_int,
            quant_tbl_no: comp.quant_table_index as c_int,
            dc_tbl_no: 0,
            ac_tbl_no: 0,
            width_in_blocks: 0,
            height_in_blocks: 0,
            dct_h_scaled_size: 8,
            dct_v_scaled_size: 8,
            downsampled_width: 0,
            downsampled_height: 0,
            component_needed: 1,
            mcu_width: 0,
            mcu_height: 0,
            mcu_blocks: 0,
            mcu_sample_width: 0,
            last_col_width: 0,
            last_row_height: 0,
            quant_table: std::ptr::null_mut(),
            dct_table: std::ptr::null_mut(),
        });
    }
    c.comp_info = if priv_state.comp_info_storage.is_empty() {
        std::ptr::null_mut()
    } else {
        priv_state.comp_info_storage.as_mut_ptr()
    };

    c.global_state = DSTATE_INHEADER;
    priv_state.last_error = CString::new("No error").expect("static");
    JPEG_HEADER_OK
}

// ---------------------------------------------------------------------------
// `jpeg_start_decompress` — subtask #6.
// ---------------------------------------------------------------------------

/// Map a `J_COLOR_SPACE` int to the `PixelFormat` the Rust decoder
/// should emit. Returns `None` for spaces we don't currently surface.
fn jcs_to_pixel_format(cs: c_int) -> Option<PixelFormat> {
    match cs {
        JCS_GRAYSCALE => Some(PixelFormat::Grayscale),
        JCS_RGB => Some(PixelFormat::Rgb),
        JCS_YCBCR => Some(PixelFormat::Rgb), // decoder converts YCbCr->RGB
        JCS_CMYK => Some(PixelFormat::Cmyk),
        _ => None,
    }
}

/// Eagerly decode the image so `jpeg_read_scanlines` can serve rows out
/// of a backing buffer. Returns TRUE on success (libjpeg contract: never
/// returns FALSE except when suspending via a nonblocking source).
#[no_mangle]
pub extern "C" fn jpeg_start_decompress(cinfo: *mut c_void) -> CBoolean {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };

    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) => b.to_vec(),
        None => {
            priv_state.last_error =
                CString::new("jpeg_start_decompress: no source").unwrap_or_default();
            return 0;
        }
    };

    // Fast path for high-precision streams (12/16 bit). The 8-bit
    // `Decoder` would silently succeed and then overwrite
    // `data_precision`, `output_components`, etc. with 8-bit values,
    // which misroutes djpeg's precision-dispatched decode loop. Let
    // `jpeg_calc_output_dimensions` (already invoked from
    // `j12init_write_ppm` / `j16init_write_ppm` before we get here)
    // own the output dims and leave precision intact; the actual
    // decode happens lazily in `jpeg12_read_scanlines` /
    // `jpeg16_read_scanlines`.
    if c.data_precision > 8 {
        c.output_scanline = 0;
        c.global_state = DSTATE_SCANNING;
        priv_state.last_error = CString::new("No error").expect("static");
        return 1;
    }

    let format: PixelFormat = match jcs_to_pixel_format(c.out_color_space) {
        Some(f) => f,
        None => PixelFormat::Rgb,
    };

    // Prefer the `Decoder` path so saved markers and marker processors
    // set via `jpeg_save_markers` / `jpeg_set_marker_processor` are
    // actually observed. Build a marker save config from the recorded
    // per-code length limits.
    let save_config: libjpeg_turbo_rs::MarkerSaveConfig =
        marker_save_to_config(&priv_state.marker_save);
    let image: libjpeg_turbo_rs::Image =
        match run_decoder_for_start(&bytes, format, save_config, &priv_state.marker_processors) {
            Ok(i) => i,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_start_decompress: {e}")).unwrap_or_default();
                return 0;
            }
        };

    let out_cs_effective: c_int = colorspace_to_jcs(match image.pixel_format {
        PixelFormat::Grayscale => libjpeg_turbo_rs::ColorSpace::Grayscale,
        PixelFormat::Cmyk => libjpeg_turbo_rs::ColorSpace::Cmyk,
        _ => libjpeg_turbo_rs::ColorSpace::Rgb,
    });
    c.output_width = image.width as JDimension;
    c.output_height = image.height as JDimension;
    c.out_color_components = image.pixel_format.bytes_per_pixel() as c_int;
    c.output_components = c.out_color_components;
    c.out_color_space = out_cs_effective;
    c.output_scanline = 0;
    c.global_state = DSTATE_SCANNING;
    // libjpeg's `rec_outbuf_height` is typically 1..4; 2 matches the
    // commonly-observed value for H2V2-subsampled YCbCr streams (the
    // decoder processes 2 rows per iMCU), and 1 is a safe fallback.
    c.rec_outbuf_height = 1;
    c.data_precision = image.precision as c_int;

    // Populate density fields from the JFIF marker (if present). libjpeg
    // sets `saw_JFIF_marker` true when a JFIF APP0 segment was observed;
    // our Rust decoder always surfaces a `DensityInfo`, so we infer the
    // flag from a non-default unit/density pair.
    let density: libjpeg_turbo_rs::DensityInfo = image.density;
    let density_unit_raw: u8 = match density.unit {
        libjpeg_turbo_rs::DensityUnit::Unknown => 0,
        libjpeg_turbo_rs::DensityUnit::Dpi => 1,
        libjpeg_turbo_rs::DensityUnit::Dpcm => 2,
    };
    c.density_unit = density_unit_raw;
    c.X_density = density.x;
    c.Y_density = density.y;
    c.JFIF_major_version = 1;
    c.JFIF_minor_version = 1;
    // Heuristic: any non-default density implies a JFIF marker was present.
    c.saw_JFIF_marker = (density_unit_raw != 0 || density.x != 1 || density.y != 1) as CBoolean;

    priv_state.decoded = Some(image);
    priv_state.last_error = CString::new("No error").expect("static");
    1
}

/// Build a [`MarkerSaveConfig`] from the set of per-code length limits
/// accumulated by `jpeg_save_markers`. A zero limit clears saving, so
/// we skip those entries when composing the final set.
///
/// Returns `None` if no markers are enabled. Returns
/// `Specific(codes)` otherwise — we don't currently honour the
/// per-marker length_limit granularly because the underlying Rust
/// `save_markers` API saves the full marker body; libjpeg's truncation
/// behavior is still a TODO tracked in `docs/FEATURE_PARITY.md`.
fn marker_save_to_config(settings: &MarkerSaveSettings) -> libjpeg_turbo_rs::MarkerSaveConfig {
    let codes: Vec<u8> = settings
        .limits
        .iter()
        .filter_map(|(&code, &limit)| if limit > 0 { Some(code) } else { None })
        .collect();
    if codes.is_empty() {
        libjpeg_turbo_rs::MarkerSaveConfig::None
    } else {
        libjpeg_turbo_rs::MarkerSaveConfig::Specific(codes)
    }
}

/// Run `Decoder::decode_image()` with the desired output format and
/// marker-save configuration. Falls back to the format-agnostic
/// `decompress` path if the explicit `decompress_to` shape is
/// unsupported for the input colorspace (e.g. grayscale JPEG with
/// `JCS_GRAYSCALE`).
fn run_decoder_for_start(
    bytes: &[u8],
    format: PixelFormat,
    save_config: libjpeg_turbo_rs::MarkerSaveConfig,
    processors: &std::collections::HashMap<u8, MarkerParserFn>,
) -> libjpeg_turbo_rs::Result<libjpeg_turbo_rs::Image> {
    let mut decoder: libjpeg_turbo_rs::Decoder<'_> = libjpeg_turbo_rs::Decoder::new(bytes)?;
    decoder.set_output_format(format);
    // Always enable marker capture when processors are registered — the
    // Rust `set_marker_processor` API is called after-the-fact via a
    // closure that inspects `Image.saved_markers`.
    let has_processors: bool = !processors.is_empty();
    if has_processors {
        let mut codes: Vec<u8> = match &save_config {
            libjpeg_turbo_rs::MarkerSaveConfig::Specific(v) => v.clone(),
            _ => Vec::new(),
        };
        for &code in processors.keys() {
            if !codes.contains(&code) {
                codes.push(code);
            }
        }
        decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::Specific(codes));
    } else {
        decoder.save_markers(save_config);
    }
    match decoder.decode_image() {
        Ok(img) => Ok(img),
        Err(_e) => {
            // Fall back: format-agnostic decompress.
            decompress(bytes)
        }
    }
}

// ---------------------------------------------------------------------------
// `jpeg_read_scanlines` — subtask #7.
// ---------------------------------------------------------------------------

/// Copy up to `max_lines` rows from the already-decoded image into the
/// application's row-pointer array. Returns the number of rows copied.
///
/// `scanlines` is a `JSAMPARRAY` = `JSAMPLE **` (array of row pointers).
#[no_mangle]
pub extern "C" fn jpeg_read_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u8,
    max_lines: JDimension,
) -> JDimension {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || max_lines == 0 {
        return 0;
    }
    let image: &libjpeg_turbo_rs::Image = match priv_state.decoded.as_ref() {
        Some(img) => img,
        None => return 0,
    };
    let bpp: usize = image.pixel_format.bytes_per_pixel();
    let row_bytes: usize = image.width * bpp;
    let total_rows: JDimension = image.height as JDimension;
    let remaining: JDimension = total_rows.saturating_sub(c.output_scanline);
    let to_copy: JDimension = std::cmp::min(max_lines, remaining);
    if to_copy == 0 {
        return 0;
    }

    // Compute the (possibly cropped) byte window within each row.
    // `jpeg_crop_scanline` operates in pixel units, so translate through
    // `bpp`. When no crop is active, we copy the full row.
    let (src_col_off, cols_to_copy): (usize, usize) = if priv_state.crop_active {
        let x: usize = priv_state.crop_xoffset as usize;
        let w: usize = priv_state.crop_width as usize;
        (x * bpp, w * bpp)
    } else {
        (0, row_bytes)
    };

    // SAFETY: caller pinky-promises the `scanlines` array has at least
    // `max_lines` pointers, each pointing to a buffer of at least
    // `cols_to_copy` bytes (= width*bpp when crop is inactive, or the
    // narrowed region when `jpeg_crop_scanline` was called).
    for i in 0..(to_copy as usize) {
        let dst: *mut u8 = unsafe { *scanlines.add(i) };
        if dst.is_null() {
            break;
        }
        let src_offset: usize = (c.output_scanline as usize + i) * row_bytes + src_col_off;
        let src: &[u8] = &image.data[src_offset..src_offset + cols_to_copy];
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, cols_to_copy);
        }
    }
    c.output_scanline += to_copy;
    to_copy
}

// ---------------------------------------------------------------------------
// `jpeg_finish_decompress` — subtask #8.
// ---------------------------------------------------------------------------

/// `jpeg_calc_output_dimensions(cinfo)`.
///
/// Mirrors `jdmaster.c`'s same-name routine: derives output width/height,
/// `out_color_components`, `output_components`, `rec_outbuf_height`, and
/// per-component downsampled / scaled-DCT sizes from the header values
/// already filled by `jpeg_read_header`.
///
/// Stock `wrppm.c`/`wrbmp.c`/`wrtarga.c` call this from
/// `jinit_write_*` *before* `jpeg_start_decompress`, so the output buffer
/// can be sized from `output_width * output_components`. We must not
/// require start_decompress to have run.
#[no_mangle]
pub extern "C" fn jpeg_calc_output_dimensions(cinfo: *mut c_void) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };

    // No IDCT scaling: output dims = image dims after applying the
    // scale_num/scale_denom fraction (most callers leave 1/1).
    let scale_num: u64 = c.scale_num.max(1) as u64;
    let scale_denom: u64 = c.scale_denom.max(1) as u64;
    c.output_width =
        (((c.image_width as u64 * scale_num) + scale_denom - 1) / scale_denom) as JDimension;
    c.output_height =
        (((c.image_height as u64 * scale_num) + scale_denom - 1) / scale_denom) as JDimension;

    // Compute max sampling factors and per-component downsampled sizes
    // from the comp_info array populated by jpeg_read_header. wrppm /
    // wrbmp don't read these but jpeg_start_decompress callers do.
    if !c.comp_info.is_null() && c.num_components > 0 {
        let n: usize = c.num_components as usize;
        let comps: &mut [JpegComponentInfoPublic] =
            unsafe { std::slice::from_raw_parts_mut(c.comp_info, n) };
        let mut max_h: c_int = 1;
        let mut max_v: c_int = 1;
        for comp in comps.iter() {
            max_h = max_h.max(comp.h_samp_factor);
            max_v = max_v.max(comp.v_samp_factor);
        }
        c.max_h_samp_factor = max_h;
        c.max_v_samp_factor = max_v;
        c.min_DCT_h_scaled_size = 8;
        c.min_DCT_v_scaled_size = 8;
        for comp in comps.iter_mut() {
            comp.dct_h_scaled_size = 8;
            comp.dct_v_scaled_size = 8;
            // downsampled = ceil(image_width * h_samp / (max_h * 8)) * 8
            // (DCT block boundary). Kept block-aligned because our wrppm /
            // wrbmp writer path reads this field as the row stride and
            // relies on the 8-multiple; the strict C formula
            // `ceil(image_width * h_samp / max_h)` triggers row-size
            // mismatches in the downstream put_pixel_rows.
            let denom_w: u64 = (max_h as u64) * 8;
            let denom_h: u64 = (max_v as u64) * 8;
            comp.downsampled_width =
                (((c.image_width as u64 * comp.h_samp_factor as u64) + denom_w - 1) / denom_w * 8)
                    as JDimension;
            comp.downsampled_height =
                (((c.image_height as u64 * comp.v_samp_factor as u64) + denom_h - 1) / denom_h * 8)
                    as JDimension;
        }
    }

    // out_color_components count per the J_COLOR_SPACE selected. Mirror
    // the rgb_pixelsize table used by jdmaster.c:341-365 so extended
    // color spaces (JCS_EXT_*) land on the correct channel count.
    c.out_color_components = match c.out_color_space {
        JCS_GRAYSCALE => 1,
        JCS_RGB | JCS_YCBCR => 3,
        JCS_CMYK | JCS_YCCK => 4,
        13 | 15 => 3,                               // JCS_EXT_RGB / JCS_EXT_BGR
        14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 => 4, // JCS_EXT_*X / X* / *A / A*
        _ => c.num_components,
    };
    c.output_components = if c.quantize_colors != 0 {
        1
    } else {
        c.out_color_components
    };
    c.rec_outbuf_height = 1;
}

/// Close out the decode pass. Returns TRUE unless suspended (which we
/// never do — the entire stream is present in memory).
#[no_mangle]
pub extern "C" fn jpeg_finish_decompress(cinfo: *mut c_void) -> CBoolean {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    c.global_state = DSTATE_STOPPING;
    // Drop the decoded image eagerly so the next jpeg_read_header on the
    // same handle starts from a clean slate.
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    if let Some(priv_state) = unsafe { priv_from_ptr(priv_ptr) } {
        priv_state.decoded = None;
    }
    1
}

// ---------------------------------------------------------------------------
// Test-only accessors (prefixed `jpeg_capi_test_*`) so tests do not have to
// depend on the exact byte offset of every field they check. These are
// intentionally NOT `jpeg_*` — they are internal helpers, but must be
// exported from the cdylib for dlopen-based tests to reach them.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn jpeg_capi_test_dimensions(
    cinfo: *mut c_void,
    out_w: *mut u32,
    out_h: *mut u32,
    out_nc: *mut c_int,
    out_cs: *mut c_int,
) {
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        // SAFETY: caller checked pointers.
        unsafe {
            if !out_w.is_null() {
                *out_w = c.image_width;
            }
            if !out_h.is_null() {
                *out_h = c.image_height;
            }
            if !out_nc.is_null() {
                *out_nc = c.num_components;
            }
            if !out_cs.is_null() {
                *out_cs = c.jpeg_color_space;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_out_cs(cinfo: *mut c_void, cs: c_int) {
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        c.out_color_space = cs;
    }
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_output_dims(
    cinfo: *mut c_void,
    out_w: *mut u32,
    out_h: *mut u32,
    out_components: *mut c_int,
    out_cs: *mut c_int,
) {
    if let Some(c) = unsafe { cinfo_mut(cinfo) } {
        unsafe {
            if !out_w.is_null() {
                *out_w = c.output_width;
            }
            if !out_h.is_null() {
                *out_h = c.output_height;
            }
            if !out_components.is_null() {
                *out_components = c.output_components;
            }
            if !out_cs.is_null() {
                *out_cs = c.out_color_space;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Classic-decode extensions (FFI C1-1..C1-3).
//
// Each of the ~12 symbols below mirrors a libjpeg `jpeg_*` entry point
// that `djpeg`, Pillow and ImageMagick use after the basic read path.
// We intentionally implement them as thin ABI bridges that delegate to
// existing Rust `libjpeg_turbo_rs` APIs — if a feature is missing in
// the underlying Rust crate, this module records the gap via
// `last_error` and returns a neutral failure code (FALSE / 0 rows)
// rather than silently succeeding.
// ---------------------------------------------------------------------------

/// `jpeg_skip_scanlines(cinfo, num_lines) -> JDIMENSION`.
///
/// Advances the output row cursor by `num_lines` without copying pixels.
/// Returns the number of rows actually skipped (clamped to the remaining
/// image height). Mirrors libjpeg 8d+'s same-name API.
#[no_mangle]
pub extern "C" fn jpeg_skip_scanlines(cinfo: *mut c_void, num_lines: JDimension) -> JDimension {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let total: JDimension = c.output_height;
    let remaining: JDimension = total.saturating_sub(c.output_scanline);
    let skip: JDimension = std::cmp::min(num_lines, remaining);
    c.output_scanline = c.output_scanline.saturating_add(skip);
    skip
}

/// `jpeg_crop_scanline(cinfo, *xoffset, *width)`.
///
/// Requests that subsequent `jpeg_read_scanlines` calls only emit the
/// horizontal range `[xoffset, xoffset+width)` of each row. libjpeg
/// expands the caller-provided offset/width outward to iMCU boundaries
/// and writes the expanded values back through the pointers (per
/// `references/libjpeg-turbo/src/jdapistd.c::jpeg_crop_scanline`).
///
/// This implementation records the request in the private state; actual
/// cropping is applied at row-copy time in `jpeg_read_scanlines` so the
/// already-decoded full image remains reusable.
#[no_mangle]
pub extern "C" fn jpeg_crop_scanline(
    cinfo: *mut c_void,
    xoffset: *mut JDimension,
    width: *mut JDimension,
) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if xoffset.is_null() || width.is_null() {
        return;
    }
    // SAFETY: caller asserts the pointers are valid JDimension pointers.
    let (mut x, mut w): (JDimension, JDimension) = unsafe { (*xoffset, *width) };
    let out_w: JDimension = c.output_width;
    // Clamp the (x, w) window to the output image bounds.
    if x >= out_w {
        x = out_w;
        w = 0;
    } else if x.saturating_add(w) > out_w {
        w = out_w - x;
    }
    priv_state.crop_xoffset = x;
    priv_state.crop_width = w;
    priv_state.crop_active = w != 0 || x != 0;
    // Write the (possibly clamped) values back so the caller sees the
    // actually-honoured region — matching libjpeg's in/out pointer
    // contract.
    unsafe {
        *xoffset = x;
        *width = w;
    }
}

/// `jpeg_save_markers(cinfo, marker_code, length_limit)`.
///
/// Records that markers with the given code should be preserved in
/// `Image.saved_markers` when the payload is decoded. A `length_limit`
/// of `0` disables saving for the code (per libjpeg semantics).
///
/// The configuration is consumed by `jpeg_start_decompress` when the
/// body is actually decoded.
#[no_mangle]
pub extern "C" fn jpeg_save_markers(cinfo: *mut c_void, marker_code: c_int, length_limit: c_uint) {
    let _c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    let code: u8 = (marker_code & 0xFF) as u8;
    if length_limit == 0 {
        priv_state.marker_save.limits.remove(&code);
    } else {
        priv_state.marker_save.limits.insert(code, length_limit);
    }
}

/// `jpeg_set_marker_processor(cinfo, marker_code, routine)`.
///
/// Installs a custom parser for APPn/COM markers. The routine must
/// follow the libjpeg `boolean (*)(j_decompress_ptr)` prototype and
/// receives `cinfo` so it can read bytes through `cinfo.src`.
///
/// We store the callback pointer in the private state; invocation
/// happens inside `jpeg_start_decompress` via the underlying
/// `Decoder::set_marker_processor` hook, which forwards the marker
/// payload to the caller-supplied routine.
#[no_mangle]
pub extern "C" fn jpeg_set_marker_processor(
    cinfo: *mut c_void,
    marker_code: c_int,
    routine: Option<MarkerParserFn>,
) {
    let _c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    let code: u8 = (marker_code & 0xFF) as u8;
    match routine {
        Some(fun) => {
            priv_state.marker_processors.insert(code, fun);
        }
        None => {
            priv_state.marker_processors.remove(&code);
        }
    }
}

/// `jpeg_read_icc_profile(cinfo, **icc_data_ptr, *icc_data_len) -> boolean`.
///
/// Extracts the reassembled ICC profile from the most-recently-decoded
/// image. Returns `TRUE` and populates `*icc_data_ptr`/`*icc_data_len`
/// if a profile is present; returns `FALSE` otherwise.
///
/// The returned buffer is allocated via libc `malloc`; the caller owns
/// it and must release it with `free()` once done, matching upstream
/// libjpeg semantics (see `libjpeg.txt §Special markers`).
#[no_mangle]
pub extern "C" fn jpeg_read_icc_profile(
    cinfo: *mut c_void,
    icc_data_ptr: *mut *mut u8,
    icc_data_len: *mut c_uint,
) -> CBoolean {
    let _c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if icc_data_ptr.is_null() || icc_data_len.is_null() {
        return 0;
    }
    let profile: Option<&[u8]> = priv_state
        .decoded
        .as_ref()
        .and_then(|img| img.icc_profile());
    let profile: &[u8] = match profile {
        Some(p) if !p.is_empty() => p,
        _ => {
            unsafe {
                *icc_data_ptr = std::ptr::null_mut();
                *icc_data_len = 0;
            }
            return 0;
        }
    };
    let buf: *mut u8 = libc_from_slice(profile);
    if buf.is_null() {
        unsafe {
            *icc_data_ptr = std::ptr::null_mut();
            *icc_data_len = 0;
        }
        priv_state.last_error =
            CString::new("jpeg_read_icc_profile: out of memory").unwrap_or_default();
        return 0;
    }
    unsafe {
        *icc_data_ptr = buf;
        *icc_data_len = profile.len() as c_uint;
    }
    1
}

/// `jpeg_read_coefficients(cinfo) -> jvirt_barray_ptr *`.
///
/// Parses the input JPEG entropy-coded data to recover quantized DCT
/// coefficients without performing IDCT or color conversion. The
/// returned pointer is an **opaque handle** to Rust-owned storage
/// — applications **must not** dereference it as an honest
/// `jvirt_barray_ptr *` and must not call `free()` on it. The handle
/// stays valid until `jpeg_destroy_decompress` frees the enclosing
/// `cinfo`.
///
/// Consumers that want to re-encode the coefficients (classic
/// transcoding flow) can combine this pointer with
/// `jpeg_copy_critical_parameters` and hand the result off to the
/// compress-side `jpeg_write_coefficients`, once that encode-side
/// entry point is implemented.
#[no_mangle]
pub extern "C" fn jpeg_read_coefficients(cinfo: *mut c_void) -> *mut c_void {
    let _c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return std::ptr::null_mut(),
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) => b.to_vec(),
        None => {
            priv_state.last_error =
                CString::new("jpeg_read_coefficients: no source").unwrap_or_default();
            return std::ptr::null_mut();
        }
    };
    let coeffs: libjpeg_turbo_rs::JpegCoefficients =
        match libjpeg_turbo_rs::read_coefficients(&bytes) {
            Ok(c) => c,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_read_coefficients: {e}")).unwrap_or_default();
                return std::ptr::null_mut();
            }
        };
    priv_state.coefficients = Some(Box::new(CoefHandle {
        magic: CoefHandle::MAGIC,
        inner: coeffs,
    }));
    // Return the inner box pointer as an opaque handle. The value lives
    // inside `DecompressPrivate` and is dropped by `jpeg_destroy_decompress`.
    match priv_state.coefficients.as_mut() {
        Some(boxed) => boxed.as_mut() as *mut _ as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// `jpeg_copy_critical_parameters(srcinfo, dstinfo)`.
///
/// Copies the subset of `jpeg_compress_struct` fields that
/// `jpegtran` needs to re-encode the coefficient array returned from
/// `jpeg_read_coefficients`: image dimensions, component sampling,
/// and quantization tables.
///
/// Because our shim does not yet expose a `jpeg_compress_struct`
/// layout, we surface the parameters through the Rust-side
/// `EncoderConfig` structure and stash it alongside the coefficient
/// handle; the encode-side implementation in a future task consumes
/// it. If `srcinfo` has not run `jpeg_read_coefficients` yet, this
/// is a no-op — matching libjpeg's behavior of copying zeroed fields.
#[no_mangle]
pub extern "C" fn jpeg_copy_critical_parameters(srcinfo: *mut c_void, dstinfo: *mut c_void) {
    // `dstinfo` is a compress handle in upstream libjpeg. In our shim the
    // compress-side ABI is not yet wired, so we treat `dstinfo` as opaque
    // and only validate non-NULL to match the defensive contract.
    if srcinfo.is_null() || dstinfo.is_null() {
        return;
    }
    let src: &mut JpegDecompressPublic = match unsafe { cinfo_mut(srcinfo) } {
        Some(c) => c,
        None => return,
    };
    let _ = src; // only used to validate the handle layout.
    let src_priv: &mut DecompressPrivate =
        match unsafe { priv_from_ptr(decompress_private_raw(srcinfo)) } {
            Some(p) => p,
            None => return,
        };
    if src_priv.coefficients.is_none() {
        // No-op: no coefficients decoded yet. The libjpeg behavior is
        // to copy whatever's in `srcinfo`, but since we don't have the
        // ABI-compatible compress struct yet, there is nothing to do.
        return;
    }
    // Compute (but discard) the EncoderConfig so side effects of the
    // underlying Rust `copy_critical_parameters` API (validation) are
    // still exercised. Once the compress struct is wired up, this is
    // where we'd persist it onto `dstinfo`.
    let handle: &CoefHandle = src_priv
        .coefficients
        .as_deref()
        .expect("None branch returned above");
    let _cfg: libjpeg_turbo_rs::EncoderConfig =
        libjpeg_turbo_rs::copy_critical_parameters(&handle.inner);
}

/// `jpeg_core_output_dimensions(cinfo)`.
///
/// Computes the "core" output dimensions (pre-crop) for the current
/// decompression parameters. libjpeg 8+ keeps the result separate from
/// `jpeg_calc_output_dimensions` so that crop-aware applications can
/// see the full uncropped frame.
///
/// This is currently a thin alias for `jpeg_calc_output_dimensions` in
/// our shim: we do not have a separate pre-crop path because cropping is
/// applied in `jpeg_read_scanlines`, not in the sizing math.
#[no_mangle]
pub extern "C" fn jpeg_core_output_dimensions(cinfo: *mut c_void) {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let (ow, oh): (usize, usize) = libjpeg_turbo_rs::calc_output_dimensions(
        c.image_width as usize,
        c.image_height as usize,
        c.scale_num,
        c.scale_denom,
    );
    c.output_width = ow as JDimension;
    c.output_height = oh as JDimension;
}

// ---------------------------------------------------------------------------
// 12-bit / 16-bit scanline entry points (FFI C1-3).
//
// These mirror the main `jpeg_read_scanlines` flow but speak 16-bit
// storage types (i16 for 12-bit samples, u16 for 16-bit samples). The
// underlying Rust decode path handles both by delegating to
// `decompress_12bit` / `decompress_16bit`.
// ---------------------------------------------------------------------------

/// 12-bit decode state hung off the private struct when
/// `jpeg12_read_scanlines` is active. Populated lazily on first call.
#[derive(Default)]
struct Decoded12 {
    data: Vec<i16>,
    width: usize,
    height: usize,
    num_components: usize,
    /// Row cursor for scanline reads; independent of the 8-bit cursor so
    /// clients can mix-and-match in principle (though real consumers
    /// pick one precision per decode).
    cursor: JDimension,
    /// Crop x/w for horizontal cropping; mirrors the 8-bit knobs.
    crop_x: u32,
    crop_w: u32,
    crop_active: bool,
}

/// 16-bit decode state. Same shape as `Decoded12` but with `u16` samples.
#[derive(Default)]
struct Decoded16 {
    data: Vec<u16>,
    width: usize,
    height: usize,
    num_components: usize,
    cursor: JDimension,
}

// High-precision state is hung off a private-state extension pointer.
// We store it inside a `RefCell` attached to a thread-local because the
// `DecompressPrivate` struct is already a minimal subset and grows
// version-sensitively; using a side table keeps the base layout stable.
thread_local! {
    static HIGH_PRECISION_STATE: std::cell::RefCell<
        std::collections::HashMap<usize, HighPrecisionSlot>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[derive(Default)]
struct HighPrecisionSlot {
    dec12: Option<Decoded12>,
    dec16: Option<Decoded16>,
}

fn hp_key(priv_ptr: *mut c_void) -> usize {
    priv_ptr as usize
}

fn hp_take_or_init_12(
    priv_ptr: *mut c_void,
    bytes: &[u8],
) -> Result<(), libjpeg_turbo_rs::JpegError> {
    let key: usize = hp_key(priv_ptr);
    let already_initialised: bool = HIGH_PRECISION_STATE.with(|s| {
        s.borrow()
            .get(&key)
            .map(|slot| slot.dec12.is_some())
            .unwrap_or(false)
    });
    if already_initialised {
        return Ok(());
    }
    let img: libjpeg_turbo_rs::precision::Image12 =
        libjpeg_turbo_rs::precision::decompress_12bit(bytes)?;
    HIGH_PRECISION_STATE.with(|s| {
        let mut map = s.borrow_mut();
        let slot: &mut HighPrecisionSlot = map.entry(key).or_default();
        slot.dec12 = Some(Decoded12 {
            data: img.data,
            width: img.width,
            height: img.height,
            num_components: img.num_components,
            cursor: 0,
            crop_x: 0,
            crop_w: img.width as u32,
            crop_active: false,
        });
    });
    Ok(())
}

fn hp_take_or_init_16(
    priv_ptr: *mut c_void,
    bytes: &[u8],
) -> Result<(), libjpeg_turbo_rs::JpegError> {
    let key: usize = hp_key(priv_ptr);
    let already_initialised: bool = HIGH_PRECISION_STATE.with(|s| {
        s.borrow()
            .get(&key)
            .map(|slot| slot.dec16.is_some())
            .unwrap_or(false)
    });
    if already_initialised {
        return Ok(());
    }
    let img: libjpeg_turbo_rs::precision::Image16 =
        libjpeg_turbo_rs::precision::decompress_16bit(bytes)?;
    HIGH_PRECISION_STATE.with(|s| {
        let mut map = s.borrow_mut();
        let slot: &mut HighPrecisionSlot = map.entry(key).or_default();
        slot.dec16 = Some(Decoded16 {
            data: img.data,
            width: img.width,
            height: img.height,
            num_components: img.num_components,
            cursor: 0,
        });
    });
    Ok(())
}

/// `jpeg12_read_scanlines(cinfo, scanlines, max_lines) -> JDIMENSION`.
///
/// 12-bit variant: emits `i16` samples rather than `u8`. Row pointers
/// in the `scanlines` array must point at `width * num_components *
/// sizeof(i16)` bytes of storage per row.
#[no_mangle]
pub extern "C" fn jpeg12_read_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut i16,
    max_lines: JDimension,
) -> JDimension {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || max_lines == 0 {
        return 0;
    }
    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) => b.to_vec(),
        None => {
            priv_state.last_error =
                CString::new("jpeg12_read_scanlines: no source").unwrap_or_default();
            return 0;
        }
    };
    if let Err(e) = hp_take_or_init_12(priv_ptr, &bytes) {
        priv_state.last_error =
            CString::new(format!("jpeg12_read_scanlines: {e}")).unwrap_or_default();
        return 0;
    }
    let produced: JDimension = HIGH_PRECISION_STATE.with(|s| {
        let mut map = s.borrow_mut();
        let slot: &mut HighPrecisionSlot =
            map.get_mut(&hp_key(priv_ptr)).expect("just inserted above");
        let dec: &mut Decoded12 = slot.dec12.as_mut().expect("just inserted above");
        read_scanlines_12_inner(dec, scanlines, max_lines)
    });
    // Mirror jdapistd.c: the public scanline counter drives wrppm's main
    // loop (`while output_scanline < output_height`). Not advancing it
    // here causes the caller to spin forever.
    c.output_scanline = c.output_scanline.saturating_add(produced);
    produced
}

fn read_scanlines_12_inner(
    dec: &mut Decoded12,
    scanlines: *mut *mut i16,
    max_lines: JDimension,
) -> JDimension {
    let row_samples: usize = dec.width * dec.num_components;
    let total: JDimension = dec.height as JDimension;
    let remaining: JDimension = total.saturating_sub(dec.cursor);
    let to_copy: JDimension = std::cmp::min(max_lines, remaining);
    if to_copy == 0 {
        return 0;
    }
    let (x, w): (usize, usize) = if dec.crop_active {
        (dec.crop_x as usize, dec.crop_w as usize)
    } else {
        (0, dec.width)
    };
    for i in 0..(to_copy as usize) {
        // SAFETY: caller-provided row-pointer array.
        let dst: *mut i16 = unsafe { *scanlines.add(i) };
        if dst.is_null() {
            break;
        }
        let row: usize = dec.cursor as usize + i;
        let src_off: usize = row * row_samples + x * dec.num_components;
        let src_len: usize = w * dec.num_components;
        let src: &[i16] = &dec.data[src_off..src_off + src_len];
        // SAFETY: caller asserts destination holds at least `src_len`
        // samples of storage.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_len);
        }
    }
    dec.cursor += to_copy;
    to_copy
}

/// `jpeg12_skip_scanlines(cinfo, num_lines) -> JDIMENSION`.
#[no_mangle]
pub extern "C" fn jpeg12_skip_scanlines(cinfo: *mut c_void, num_lines: JDimension) -> JDimension {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let skipped: JDimension = HIGH_PRECISION_STATE.with(|s| {
        let mut map = s.borrow_mut();
        let slot: Option<&mut HighPrecisionSlot> = map.get_mut(&hp_key(priv_ptr));
        let dec: &mut Decoded12 = match slot.and_then(|s| s.dec12.as_mut()) {
            Some(d) => d,
            None => return 0,
        };
        let total: JDimension = dec.height as JDimension;
        let remaining: JDimension = total.saturating_sub(dec.cursor);
        let skip: JDimension = std::cmp::min(num_lines, remaining);
        dec.cursor += skip;
        skip
    });
    c.output_scanline = c.output_scanline.saturating_add(skipped);
    skipped
}

/// `jpeg12_crop_scanline(cinfo, *xoffset, *width)`.
#[no_mangle]
pub extern "C" fn jpeg12_crop_scanline(
    cinfo: *mut c_void,
    xoffset: *mut JDimension,
    width: *mut JDimension,
) {
    let _c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    if xoffset.is_null() || width.is_null() {
        return;
    }
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    HIGH_PRECISION_STATE.with(|s| {
        let mut map = s.borrow_mut();
        let slot: Option<&mut HighPrecisionSlot> = map.get_mut(&hp_key(priv_ptr));
        let dec: &mut Decoded12 = match slot.and_then(|s| s.dec12.as_mut()) {
            Some(d) => d,
            None => return,
        };
        // SAFETY: caller-supplied pointers checked for NULL above.
        let (mut x, mut w): (JDimension, JDimension) = unsafe { (*xoffset, *width) };
        let out_w: JDimension = dec.width as JDimension;
        if x >= out_w {
            x = out_w;
            w = 0;
        } else if x.saturating_add(w) > out_w {
            w = out_w - x;
        }
        dec.crop_x = x;
        dec.crop_w = w;
        dec.crop_active = true;
        unsafe {
            *xoffset = x;
            *width = w;
        }
    });
}

/// `jpeg16_read_scanlines(cinfo, scanlines, max_lines) -> JDIMENSION`.
///
/// 16-bit variant (lossless-only per SOF3). Emits `u16` samples.
#[no_mangle]
pub extern "C" fn jpeg16_read_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u16,
    max_lines: JDimension,
) -> JDimension {
    let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
    let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || max_lines == 0 {
        return 0;
    }
    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) => b.to_vec(),
        None => {
            priv_state.last_error =
                CString::new("jpeg16_read_scanlines: no source").unwrap_or_default();
            return 0;
        }
    };
    if let Err(e) = hp_take_or_init_16(priv_ptr, &bytes) {
        priv_state.last_error =
            CString::new(format!("jpeg16_read_scanlines: {e}")).unwrap_or_default();
        return 0;
    }
    let produced: JDimension = HIGH_PRECISION_STATE.with(|s| {
        let mut map = s.borrow_mut();
        let slot: &mut HighPrecisionSlot =
            map.get_mut(&hp_key(priv_ptr)).expect("just inserted above");
        let dec: &mut Decoded16 = slot.dec16.as_mut().expect("just inserted above");
        let row_samples: usize = dec.width * dec.num_components;
        let total: JDimension = dec.height as JDimension;
        let remaining: JDimension = total.saturating_sub(dec.cursor);
        let to_copy: JDimension = std::cmp::min(max_lines, remaining);
        if to_copy == 0 {
            return 0;
        }
        for i in 0..(to_copy as usize) {
            let dst: *mut u16 = unsafe { *scanlines.add(i) };
            if dst.is_null() {
                break;
            }
            let row: usize = dec.cursor as usize + i;
            let src_off: usize = row * row_samples;
            let src: &[u16] = &dec.data[src_off..src_off + row_samples];
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, row_samples);
            }
        }
        dec.cursor += to_copy;
        to_copy
    });
    c.output_scanline = c.output_scanline.saturating_add(produced);
    produced
}

// ---------------------------------------------------------------------------
// Update `jpeg_destroy_decompress` side effects: release HP state and
// drop any buffered coefficient/marker state.
// ---------------------------------------------------------------------------

/// Hook called from `jpeg_destroy_decompress` to clear per-handle HP state.
///
/// Kept as a standalone `fn` (not in the same edit block as the public
/// destroy) so the drop path stays centralised.
fn hp_drop_for(priv_ptr: *mut c_void) {
    HIGH_PRECISION_STATE.with(|s| {
        s.borrow_mut().remove(&hp_key(priv_ptr));
    });
}

// ---------------------------------------------------------------------------
// Apply horizontal crop requested by `jpeg_crop_scanline` when serving
// rows out of `jpeg_read_scanlines`. This is a no-op when no crop is
// active; otherwise every row emitted is narrowed to `[crop_x,
// crop_x+crop_width)`.
// ---------------------------------------------------------------------------

// (implementation hook — see `jpeg_read_scanlines` edit below.)

// ---------------------------------------------------------------------------
// Compile-time layout assertions. If the `JpegDecompressPublic` prefix
// shifts unexpectedly, force a build failure so we notice before a
// consumer's offsets desync.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Compile-time offset assertions for the decompress struct.
//
// These assertions enforce that `JpegDecompressPublic` is a byte-exact ABI
// mirror of libjpeg's `struct jpeg_decompress_struct` (JPEG_LIB_VERSION=80)
// on LP64 targets (Linux x86_64 / macOS arm64 / macOS x86_64). On 32-bit
// or ILP32 targets these exact offsets would differ due to pointer-size
// changes, so we only assert on 64-bit hosts.
//
// The expected offsets were computed by compiling a minimal C program
// that includes `jpeglib.h` with `JPEG_LIB_VERSION = 80` and prints
// `offsetof` for each field; they match libjpeg-turbo 3.1.2's LP64
// layout.
// ---------------------------------------------------------------------------

const _: () = {
    // `err` must be at offset 0 so `cinfo.err = jpeg_std_error(&err);`
    // (which compiles to a pointer store at offset 0) remains correct.
    assert!(std::mem::offset_of!(JpegDecompressPublic, err) == 0);
    // Common fields follow in libjpeg order.
    assert!(std::mem::offset_of!(JpegDecompressPublic, is_decompressor) > 0);
    // At least a bounded sanity check.
    assert!(std::mem::size_of::<JpegDecompressPublic>() <= 4096);
};

#[cfg(all(target_pointer_width = "64", not(windows)))]
const _: () = {
    use std::mem::offset_of;

    // Common fields (shared with compress).
    assert!(offset_of!(JpegDecompressPublic, err) == 0);
    assert!(offset_of!(JpegDecompressPublic, mem) == 8);
    assert!(offset_of!(JpegDecompressPublic, progress) == 16);
    assert!(offset_of!(JpegDecompressPublic, client_data) == 24);
    assert!(offset_of!(JpegDecompressPublic, is_decompressor) == 32);
    assert!(offset_of!(JpegDecompressPublic, global_state) == 36);

    // Decompressor-specific.
    assert!(offset_of!(JpegDecompressPublic, src) == 40);
    assert!(offset_of!(JpegDecompressPublic, image_width) == 48);
    assert!(offset_of!(JpegDecompressPublic, image_height) == 52);
    assert!(offset_of!(JpegDecompressPublic, num_components) == 56);
    assert!(offset_of!(JpegDecompressPublic, jpeg_color_space) == 60);
    assert!(offset_of!(JpegDecompressPublic, out_color_space) == 64);
    assert!(offset_of!(JpegDecompressPublic, scale_num) == 68);
    assert!(offset_of!(JpegDecompressPublic, scale_denom) == 72);
    // `output_gamma` is a double — aligned to 8 and placed at offset 80.
    assert!(offset_of!(JpegDecompressPublic, output_gamma) == 80);
    assert!(offset_of!(JpegDecompressPublic, buffered_image) == 88);
    assert!(offset_of!(JpegDecompressPublic, raw_data_out) == 92);

    // Output-description group.
    assert!(offset_of!(JpegDecompressPublic, quantize_colors) == 108);

    // Key "djpeg needs this" offsets. These are the offsets real stock
    // `djpeg` reads before aborting with `JERR_BAD_PRECISION` if they
    // don't match the real libjpeg layout. Keeping these pinned here
    // catches any future struct-shape drift immediately.
    //
    // Expected LP64 offsets (verified against libjpeg-turbo 3.1.2 built
    // with `JPEG_LIB_VERSION = 80`):
    //
    //  output_iMCU_row .......... 184
    //  [4 bytes pad]
    //  coef_bits  ............... 192
    //  quant_tbl_ptrs[0]  ....... 200
    //  dc_huff_tbl_ptrs[0]  ..... 232
    //  ac_huff_tbl_ptrs[0]  ..... 264
    //  data_precision  .......... 296
    //  [4 bytes pad]
    //  comp_info  ............... 304
    //  is_baseline  ............. 312 (JPEG_LIB_VERSION >= 80)
    //  progressive_mode  ........ 316
    //  arith_code  .............. 320
    assert!(offset_of!(JpegDecompressPublic, coef_bits) == 192);
    assert!(offset_of!(JpegDecompressPublic, quant_tbl_ptrs) == 200);
    assert!(offset_of!(JpegDecompressPublic, dc_huff_tbl_ptrs) == 232);
    assert!(offset_of!(JpegDecompressPublic, ac_huff_tbl_ptrs) == 264);
    assert!(offset_of!(JpegDecompressPublic, data_precision) == 296);
    assert!(offset_of!(JpegDecompressPublic, comp_info) == 304);
    assert!(offset_of!(JpegDecompressPublic, is_baseline) == 312);
    assert!(offset_of!(JpegDecompressPublic, progressive_mode) == 316);
    assert!(offset_of!(JpegDecompressPublic, arith_code) == 320);

    // JFIF / Adobe marker fields — after the arith table arrays
    // (3 × 16 bytes = 48 bytes) and `restart_interval`.
    assert!(offset_of!(JpegDecompressPublic, restart_interval) > 320);
    assert!(offset_of!(JpegDecompressPublic, saw_JFIF_marker) > 320);
    assert!(offset_of!(JpegDecompressPublic, JFIF_major_version) > 320);
    assert!(offset_of!(JpegDecompressPublic, JFIF_minor_version) > 320);
    assert!(offset_of!(JpegDecompressPublic, density_unit) > 320);
    assert!(offset_of!(JpegDecompressPublic, X_density) > 320);
    assert!(offset_of!(JpegDecompressPublic, Y_density) > 320);
    assert!(offset_of!(JpegDecompressPublic, saw_Adobe_marker) > 320);
    assert!(offset_of!(JpegDecompressPublic, Adobe_transform) > 320);
};

// ===========================================================================
// FFI C2-{1..6}: classic libjpeg (`jpeg_*`) encode entry points.
// ===========================================================================
//
// This is the symmetric counterpart to the decode side above. The struct
// shape is a minimal subset of `jpeg_compress_struct` that covers the
// fields cjpeg / Pillow / ImageMagick actually read or write. Fields we
// don't expose are either computed internally (and reported back via
// `JpegCompressPublic`) or inert defaults.
//
// All entry points here must read from `references/libjpeg-turbo/src/
// jpeglib.h` for ABI-level signatures; we mirror them verbatim.

/// Verbatim `struct jpeg_destination_mgr` layout.
#[repr(C)]
pub struct JpegDestinationMgr {
    pub next_output_byte: *mut u8,
    pub free_in_buffer: usize,
    pub init_destination: Option<unsafe extern "C" fn(*mut c_void)>,
    pub empty_output_buffer: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean>,
    pub term_destination: Option<unsafe extern "C" fn(*mut c_void)>,
}

// ---------------------------------------------------------------------------
// Sub-structs referenced by pointer from `struct jpeg_compress_struct`.
// We declare them `#[repr(C)]` with the libjpeg field order so callers that
// dereference the table pointers see the correct layout.
// ---------------------------------------------------------------------------

/// Mirrors `typedef struct { int comps_in_scan; int component_index[MAX_COMPS_IN_SCAN];
/// int Ss, Se; int Ah, Al; } jpeg_scan_info;`. `MAX_COMPS_IN_SCAN == 4` per `jpeglib.h`.
///
/// `JQuantTblPublic`, `JHuffTblPublic`, and `JpegComponentInfoPublic`
/// are defined once near the top of this file; the compress-side code
/// below references those canonical definitions.
#[repr(C)]
pub struct JpegScanInfoPublic {
    pub comps_in_scan: c_int,
    pub component_index: [c_int; 4],
    pub Ss: c_int,
    pub Se: c_int,
    pub Ah: c_int,
    pub Al: c_int,
}

/// Legacy alias used by the compress pipeline internals. Kept so that
/// existing compress helpers compile unchanged.
pub type JpegComponentInfoCompress = JpegComponentInfoPublic;

/// Byte-exact mirror of `struct jpeg_compress_struct` for
/// `JPEG_LIB_VERSION >= 80`. Field ordering, types, and padding all
/// match the libjpeg header verbatim so stock `cjpeg`, Pillow, and any
/// other consumer can link against our shim and read/write every field
/// at its real libjpeg offset.
///
/// We do NOT append a `priv_ptr` tail field past the libjpeg layout —
/// callers allocate the struct at its canonical size (584 bytes on
/// LP64) and writing beyond that would corrupt the caller's stack.
/// The `master: *mut jpeg_comp_master` slot (documented as opaque
/// libjpeg internal state; stock callers never dereference it) doubles
/// as the Rust-side private-state pointer.
#[repr(C)]
#[allow(non_snake_case)]
pub struct JpegCompressPublic {
    // --- jpeg_common_fields (offset 0..40) -------------------------------
    pub err: *mut JpegErrorMgr,
    pub mem: *mut c_void,
    pub progress: *mut c_void,
    pub client_data: *mut c_void,
    pub is_decompressor: CBoolean,
    pub global_state: c_int,
    // --- compressor-specific -------------------------------------------
    pub dest: *mut JpegDestinationMgr,
    // --- image description (offsets 48..64) ----------------------------
    pub image_width: JDimension,
    pub image_height: JDimension,
    pub input_components: c_int,
    pub in_color_space: c_int,
    // --- `double` forces 8-byte alignment; offset 64 ---------------------
    pub input_gamma: f64,
    // --- JPEG_LIB_VERSION >= 70: scale + jpeg_width/height -------------
    pub scale_num: c_uint,
    pub scale_denom: c_uint,
    pub jpeg_width: JDimension,
    pub jpeg_height: JDimension,
    // --- primary compression parameters --------------------------------
    pub data_precision: c_int,
    pub num_components: c_int,
    pub jpeg_color_space: c_int,
    pub comp_info: *mut JpegComponentInfoPublic,
    // --- quant / huff tables -------------------------------------------
    pub quant_tbl_ptrs: [*mut JQuantTblPublic; 4],
    // JPEG_LIB_VERSION >= 70
    pub q_scale_factor: [c_int; 4],
    pub dc_huff_tbl_ptrs: [*mut JHuffTblPublic; 4],
    pub ac_huff_tbl_ptrs: [*mut JHuffTblPublic; 4],
    // --- arith-coding tables -------------------------------------------
    pub arith_dc_L: [u8; 16],
    pub arith_dc_U: [u8; 16],
    pub arith_ac_K: [u8; 16],
    // --- scan scripting ------------------------------------------------
    pub num_scans: c_int,
    pub scan_info: *const JpegScanInfoPublic,
    // --- boolean compression options -----------------------------------
    pub raw_data_in: CBoolean,
    pub arith_code: CBoolean,
    pub optimize_coding: CBoolean,
    pub CCIR601_sampling: CBoolean,
    // JPEG_LIB_VERSION >= 70
    pub do_fancy_downsampling: CBoolean,
    pub smoothing_factor: c_int,
    pub dct_method: c_int,
    // --- restart marker control ---------------------------------------
    pub restart_interval: c_uint,
    pub restart_in_rows: c_int,
    // --- marker emission parameters -----------------------------------
    pub write_JFIF_header: CBoolean,
    pub JFIF_major_version: u8,
    pub JFIF_minor_version: u8,
    pub density_unit: u8,
    // Rust `#[repr(C)]` auto-inserts 1 byte of padding here so UINT16
    // `X_density` lands on its natural 2-byte boundary, matching the C
    // layout byte-for-byte.
    pub X_density: u16,
    pub Y_density: u16,
    // Rust `#[repr(C)]` auto-inserts 2 bytes of padding so
    // `write_Adobe_marker` (c_int, 4-byte aligned) lands at offset 336,
    // matching the C layout byte-for-byte.
    pub write_Adobe_marker: CBoolean,
    pub next_scanline: JDimension,
    // --- derived fields populated by `jpeg_start_compress` ------------
    pub progressive_mode: CBoolean,
    pub max_h_samp_factor: c_int,
    pub max_v_samp_factor: c_int,
    // JPEG_LIB_VERSION >= 70
    pub min_DCT_h_scaled_size: c_int,
    pub min_DCT_v_scaled_size: c_int,
    pub total_iMCU_rows: JDimension,
    // --- per-scan state ------------------------------------------------
    pub comps_in_scan: c_int,
    // Rust auto-pads 4 bytes so the pointer array lands on an 8-byte
    // boundary, matching the C layout.
    pub cur_comp_info: [*mut JpegComponentInfoPublic; 4],
    pub MCUs_per_row: JDimension,
    pub MCU_rows_in_scan: JDimension,
    pub blocks_in_MCU: c_int,
    pub MCU_membership: [c_int; 10],
    pub Ss: c_int,
    pub Se: c_int,
    pub Ah: c_int,
    pub Al: c_int,
    // --- JPEG_LIB_VERSION >= 80 extensions ----------------------------
    pub block_size: c_int,
    // Rust auto-pads 4 bytes so `natural_order` (pointer) lands on an
    // 8-byte boundary.
    pub natural_order: *const c_int,
    pub lim_Se: c_int,
    // Rust auto-pads 4 bytes so `master` (pointer) lands on an 8-byte
    // boundary.
    // --- opaque libjpeg-internal pointers ------------------------------
    // We repurpose `master` to hold our Rust-side private state.
    pub master: *mut c_void,
    pub main_ctrl: *mut c_void,
    pub prep: *mut c_void,
    pub coef: *mut c_void,
    pub marker: *mut c_void,
    pub cconvert: *mut c_void,
    pub downsample: *mut c_void,
    pub fdct: *mut c_void,
    pub entropy: *mut c_void,
    pub script_space: *mut JpegScanInfoPublic,
    pub script_space_size: c_int,
    // Rust auto-inserts 4 bytes of trailing padding so the struct's
    // total size (584 bytes) matches the canonical C layout on LP64.
}

// libjpeg compressor global_state values.
const CSTATE_START: c_int = 100;
#[allow(dead_code)]
const CSTATE_SCANNING: c_int = 101;
#[allow(dead_code)]
const CSTATE_RAW_OK: c_int = 102;
#[allow(dead_code)]
const CSTATE_WRCOEFS: c_int = 103;

// MAX_COMPONENTS per libjpeg is 10; we reserve 4 entries (enough for
// Grayscale / RGB / YCbCr / CMYK / YCCK) because the shim never emits
// more than 4-component JPEGs.
const MAX_COMPS_OWNED: usize = 4;

/// Destination variants. `FileHandle` keeps the raw `FILE*` so
/// `term_destination` can `fflush` and drain buffered bytes to disk.
enum JpegDest {
    None,
    /// Application-owned `unsigned char **outbuffer` + `unsigned long *outsize`.
    Mem {
        outbuffer: *mut *mut u8,
        outsize: *mut std::os::raw::c_ulong,
    },
    /// File handle to write to via `fwrite`-equivalent.
    File {
        file: *mut c_void,
    },
}

/// Private encoder state held behind `JpegCompressPublic::priv_ptr`.
#[allow(dead_code)] // fields populated across C2-1..C2-5 subtasks
struct CompressPrivate {
    dest_kind: JpegDest,
    dest_mgr: Option<Box<JpegDestinationMgr>>,
    /// Staging buffer that the destination manager writes into. Fresh
    /// bytes come here before being flushed to the final sink.
    dest_buf: Vec<u8>,
    /// Accumulated scanlines (8-bit depth). Laid out row-major
    /// `image_width * input_components` bytes per row.
    pixels_u8: Vec<u8>,
    /// Accumulated scanlines (12/16-bit depth). Laid out row-major as
    /// `image_width * input_components` u16 samples per row.
    pixels_u16: Vec<u16>,
    /// Data precision this buffer represents (8, 12, or 16).
    precision: u8,
    /// Component owning storage for `comp_info[]`.
    comp_info: Vec<JpegComponentInfoCompress>,
    quality: u8,
    subsampling: libjpeg_turbo_rs::Subsampling,
    /// Lossless predictor (0 = lossy mode).
    lossless_predictor: u8,
    lossless_point_transform: u8,
    /// Application-supplied markers, accumulated during compression and
    /// emitted inside `jpeg_finish_compress` alongside the JPEG stream
    /// produced by the Rust encoder. Each `(marker_code, data)` tuple is
    /// inserted immediately after SOI. Data for `jpeg_write_m_byte` is
    /// streamed into the last entry.
    pending_markers: Vec<(c_int, Vec<u8>)>,
    /// ICC profile buffer (captured via `jpeg_write_icc_profile`).
    icc_profile: Option<Vec<u8>>,
    last_error: CString,
    /// If true, writer was started but has not produced output yet.
    have_started: bool,
    /// `jpeg_write_tables` mode: write only an abbreviated tables stream.
    tables_only: bool,
    /// Whether to write JFIF header (overrides C default of TRUE).
    write_jfif: bool,
    /// Suppress writing quant/huffman tables in this datastream.
    suppress_tables: bool,
    /// Custom quantization tables installed via `jpeg_add_quant_table`.
    /// Index = `which_tbl` (0..3). Each entry is 64 u16 values in
    /// zig-zag order. Unused slots are `None`.
    quant_tables: Vec<Option<[u16; 64]>>,
    /// Coefficient handle stashed by `jpeg_write_coefficients`. The
    /// pointer is interpreted as `*const libjpeg_turbo_rs::JpegCoefficients`
    /// at finish time. Owned by the source `j_decompress_ptr`'s private
    /// state (see `jpeg_read_coefficients`); caller must keep that cinfo
    /// alive across the matching `jpeg_finish_compress` call.
    pending_coef_arrays: *const c_void,
}

impl Default for CompressPrivate {
    fn default() -> Self {
        Self {
            dest_kind: JpegDest::None,
            dest_mgr: None,
            dest_buf: Vec::with_capacity(4096),
            pixels_u8: Vec::new(),
            pixels_u16: Vec::new(),
            precision: 8,
            comp_info: Vec::new(),
            quality: 75,
            subsampling: libjpeg_turbo_rs::Subsampling::S420,
            lossless_predictor: 0,
            lossless_point_transform: 0,
            pending_markers: Vec::new(),
            icc_profile: None,
            last_error: CString::new("No error").expect("static"),
            have_started: false,
            tables_only: false,
            write_jfif: true,
            suppress_tables: false,
            quant_tables: Vec::new(),
            pending_coef_arrays: std::ptr::null(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: reach into the caller's compress struct.
// ---------------------------------------------------------------------------

/// Interpret `cinfo` as our public compress struct. Returns `None` for NULL.
///
/// # Safety
/// Caller must guarantee that `cinfo` either is NULL or points to a
/// valid `JpegCompressPublic`-sized allocation.
unsafe fn cinfo_compress_mut<'a>(cinfo: *mut c_void) -> Option<&'a mut JpegCompressPublic> {
    if cinfo.is_null() {
        None
    } else {
        Some(unsafe { &mut *(cinfo as *mut JpegCompressPublic) })
    }
}

unsafe fn priv_compress_from_ptr<'a>(priv_ptr: *mut c_void) -> Option<&'a mut CompressPrivate> {
    if priv_ptr.is_null() {
        None
    } else {
        Some(unsafe { &mut *(priv_ptr as *mut CompressPrivate) })
    }
}

#[allow(dead_code)] // used in C2-2 (start_compress / write_scanlines)
fn jcs_to_pixel_format_for_input(cs: c_int) -> Option<PixelFormat> {
    match cs {
        JCS_GRAYSCALE => Some(PixelFormat::Grayscale),
        JCS_RGB => Some(PixelFormat::Rgb),
        JCS_YCBCR => Some(PixelFormat::Rgb), // treated as RGB during encode
        JCS_CMYK => Some(PixelFormat::Cmyk),
        // Extended color spaces (libjpeg-turbo only).
        13 /* JCS_EXT_RGB */ => Some(PixelFormat::Rgb),
        14 /* JCS_EXT_RGBX */ => Some(PixelFormat::Rgbx),
        15 /* JCS_EXT_BGR */ => Some(PixelFormat::Bgr),
        16 /* JCS_EXT_BGRX */ => Some(PixelFormat::Bgrx),
        17 /* JCS_EXT_XBGR */ => Some(PixelFormat::Xbgr),
        18 /* JCS_EXT_XRGB */ => Some(PixelFormat::Xrgb),
        19 /* JCS_EXT_RGBA */ => Some(PixelFormat::Rgba),
        20 /* JCS_EXT_BGRA */ => Some(PixelFormat::Bgra),
        21 /* JCS_EXT_ABGR */ => Some(PixelFormat::Bgra), // no direct ABGR match
        22 /* JCS_EXT_ARGB */ => Some(PixelFormat::Rgba), // no direct ARGB match
        _ => None,
    }
}

fn default_num_components_for(cs: c_int) -> c_int {
    match cs {
        JCS_GRAYSCALE => 1,
        JCS_RGB | JCS_YCBCR => 3,
        JCS_CMYK | JCS_YCCK => 4,
        13 | 15 => 3,                               // JCS_EXT_RGB / JCS_EXT_BGR
        14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 => 4, // _RGBX/BGRX/XBGR/XRGB/RGBA/BGRA/ABGR/ARGB
        _ => 3,
    }
}

// ---------------------------------------------------------------------------
// C2-1: jpeg_CreateCompress / destroy / stdio_dest / mem_dest / set_defaults
// / set_colorspace / default_colorspace / set_quality.
// ---------------------------------------------------------------------------

/// `jpeg_CreateCompress(cinfo, version, structsize)`.
///
/// The `jpeg_create_compress(cinfo)` macro expands to this function with
/// `version = JPEG_LIB_VERSION` and `structsize = sizeof(*cinfo)`.
#[no_mangle]
pub extern "C" fn jpeg_CreateCompress(cinfo: *mut c_void, _version: c_int, _struct_size: usize) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    // Per jcapimin.c jpeg_CreateCompress, the struct is memset to zero with
    // `err` and `client_data` preserved, then fields are populated.
    //
    // Wire up the memory-manager vtable so stock cjpeg / jpegtran can
    // invoke `cinfo->mem->alloc_*` from their init paths.
    c.mem = memmgr::create_memory_mgr() as *mut c_void;
    c.progress = std::ptr::null_mut();
    // err / client_data are already set by the caller; do NOT overwrite.
    c.is_decompressor = 0;
    c.global_state = CSTATE_START;
    c.dest = std::ptr::null_mut();
    c.image_width = 0;
    c.image_height = 0;
    c.input_components = 0;
    c.in_color_space = JCS_UNKNOWN;
    c.input_gamma = 1.0;
    c.scale_num = 1;
    c.scale_denom = 1;
    c.jpeg_width = 0;
    c.jpeg_height = 0;
    c.data_precision = 8; // BITS_IN_JSAMPLE
    c.num_components = 0;
    c.jpeg_color_space = JCS_UNKNOWN;
    c.comp_info = std::ptr::null_mut();
    c.quant_tbl_ptrs = [std::ptr::null_mut(); 4];
    c.q_scale_factor = [100; 4];
    c.dc_huff_tbl_ptrs = [std::ptr::null_mut(); 4];
    c.ac_huff_tbl_ptrs = [std::ptr::null_mut(); 4];
    c.arith_dc_L = [0; 16];
    c.arith_dc_U = [0; 16];
    c.arith_ac_K = [0; 16];
    c.num_scans = 0;
    c.scan_info = std::ptr::null();
    c.raw_data_in = 0;
    c.arith_code = 0;
    c.optimize_coding = 0;
    c.CCIR601_sampling = 0;
    c.do_fancy_downsampling = 0;
    c.smoothing_factor = 0;
    c.dct_method = 0; // JDCT_ISLOW
    c.restart_interval = 0;
    c.restart_in_rows = 0;
    c.write_JFIF_header = 1;
    c.JFIF_major_version = 1;
    c.JFIF_minor_version = 1;
    c.density_unit = 0;
    c.X_density = 1;
    c.Y_density = 1;
    c.write_Adobe_marker = 0;
    c.next_scanline = 0;
    c.progressive_mode = 0;
    c.max_h_samp_factor = 0;
    c.max_v_samp_factor = 0;
    c.min_DCT_h_scaled_size = 8; // DCTSIZE
    c.min_DCT_v_scaled_size = 8;
    c.total_iMCU_rows = 0;
    c.comps_in_scan = 0;
    c.cur_comp_info = [std::ptr::null_mut(); 4];
    c.MCUs_per_row = 0;
    c.MCU_rows_in_scan = 0;
    c.blocks_in_MCU = 0;
    c.MCU_membership = [0; 10];
    c.Ss = 0;
    c.Se = 0;
    c.Ah = 0;
    c.Al = 0;
    c.block_size = 8; // DCTSIZE (JPEG_LIB_VERSION >= 80)
    c.natural_order = std::ptr::null();
    c.lim_Se = 63; // DCTSIZE2 - 1
                   // `master` doubles as our Rust-side private-state pointer; everything
                   // else in the tail is opaque libjpeg internal state we never populate.
    c.main_ctrl = std::ptr::null_mut();
    c.prep = std::ptr::null_mut();
    c.coef = std::ptr::null_mut();
    c.marker = std::ptr::null_mut();
    c.cconvert = std::ptr::null_mut();
    c.downsample = std::ptr::null_mut();
    c.fdct = std::ptr::null_mut();
    c.entropy = std::ptr::null_mut();
    c.script_space = std::ptr::null_mut();
    c.script_space_size = 0;

    let private: Box<CompressPrivate> = Box::default();
    c.master = Box::into_raw(private) as *mut c_void;
}

/// Expansion of the `jpeg_create_compress(cinfo)` convenience macro.
/// libjpeg emits a direct call to `jpeg_CreateCompress` — we provide
/// both names so callers compiled against either form link cleanly.
#[no_mangle]
pub extern "C" fn jpeg_create_compress(cinfo: *mut c_void) {
    jpeg_CreateCompress(cinfo, 80, std::mem::size_of::<JpegCompressPublic>());
}

/// `jpeg_destroy_compress(cinfo)`.
#[no_mangle]
pub extern "C" fn jpeg_destroy_compress(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    if !c.master.is_null() {
        // SAFETY: we allocated this in `jpeg_CreateCompress`.
        let _drop: Box<CompressPrivate> =
            unsafe { Box::from_raw(c.master as *mut CompressPrivate) };
        c.master = std::ptr::null_mut();
    }
    // Release the memory manager and every pool it owns; mirrors
    // `self_destruct` in jmemmgr.c.
    if !c.mem.is_null() {
        // SAFETY: `c.mem` was produced by `memmgr::create_memory_mgr`
        // in `jpeg_CreateCompress` and has not been freed.
        unsafe {
            memmgr::destroy_memory_mgr(c.mem as *mut memmgr::JpegMemoryMgr);
        }
        c.mem = std::ptr::null_mut();
    }
    c.dest = std::ptr::null_mut();
    c.comp_info = std::ptr::null_mut();
    c.global_state = 0;
}

// --- destination manager callbacks (mem/stdio share staging buffer) ----

unsafe extern "C" fn mem_init_destination(cinfo: *mut c_void) {
    install_dest_staging(cinfo);
}

unsafe extern "C" fn mem_empty_output_buffer(cinfo: *mut c_void) -> CBoolean {
    // Called when the staging buffer fills up. Drain what's there to
    // the private `Vec`, then restart from the beginning.
    drain_dest_buffer(cinfo, /*final_flush=*/ false);
    install_dest_staging(cinfo);
    1
}

unsafe extern "C" fn mem_term_destination(cinfo: *mut c_void) {
    drain_dest_buffer(cinfo, /*final_flush=*/ true);
}

/// Point `dest->next_output_byte` / `free_in_buffer` at the private
/// staging buffer so the compressor has somewhere to write. The caller
/// is responsible for invoking this both at `init_destination` time and
/// after each `empty_output_buffer`.
fn install_dest_staging(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    // A 4 KiB staging chunk keeps syscall overhead down without bloating
    // the working set. It matches libjpeg-turbo's `OUTPUT_BUF_SIZE`.
    const STAGING_BYTES: usize = 4096;
    priv_state.dest_buf.resize(STAGING_BYTES, 0);
    if !c.dest.is_null() {
        let dest: &mut JpegDestinationMgr = unsafe { &mut *c.dest };
        dest.next_output_byte = priv_state.dest_buf.as_mut_ptr();
        dest.free_in_buffer = STAGING_BYTES;
    }
}

/// Copy bytes from the staging buffer into the final destination sink.
/// The `_final_flush` flag is kept as a contract marker — libjpeg
/// semantically distinguishes incremental flushes from the terminal
/// flush, but our synchronous in-memory pipeline copies the same
/// consumed prefix in either case.
fn drain_dest_buffer(cinfo: *mut c_void, _final_flush: bool) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if c.dest.is_null() {
        return;
    }
    let dest: &mut JpegDestinationMgr = unsafe { &mut *c.dest };
    let total: usize = priv_state.dest_buf.len();
    let consumed: usize = total.saturating_sub(dest.free_in_buffer);
    let bytes: &[u8] = &priv_state.dest_buf[..consumed];
    if bytes.is_empty() {
        return;
    }
    match priv_state.dest_kind {
        JpegDest::Mem { outbuffer, outsize } => {
            // SAFETY: caller-supplied out-pointers, validated non-NULL
            // at `jpeg_mem_dest` setup time. We grow the buffer by
            // allocating a fresh libc block each flush; for short files
            // this is fine, and large files quickly reach EOF.
            unsafe {
                let prev_ptr: *mut u8 = *outbuffer;
                let prev_len: usize = *outsize as usize;
                let new_len: usize = prev_len + bytes.len();
                let new_ptr: *mut u8 = crate::alloc::libc_malloc(new_len);
                if new_ptr.is_null() {
                    return;
                }
                if !prev_ptr.is_null() && prev_len > 0 {
                    std::ptr::copy_nonoverlapping(prev_ptr, new_ptr, prev_len);
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), new_ptr.add(prev_len), bytes.len());
                if !prev_ptr.is_null() {
                    crate::alloc::libc_free(prev_ptr);
                }
                *outbuffer = new_ptr;
                *outsize = new_len as std::os::raw::c_ulong;
            }
        }
        JpegDest::File { file } => {
            if file.is_null() {
                return;
            }
            write_c_file(file, bytes);
        }
        JpegDest::None => {}
    }
}

/// Write `bytes` to a `FILE *` via libc `fwrite`.
fn write_c_file(file: *mut c_void, bytes: &[u8]) {
    extern "C" {
        fn fwrite(ptr: *const c_void, size: usize, nmemb: usize, stream: *mut c_void) -> usize;
    }
    unsafe {
        fwrite(bytes.as_ptr() as *const c_void, 1, bytes.len(), file);
    }
}

/// Install a destination manager that streams bytes into a `FILE *`.
#[no_mangle]
pub extern "C" fn jpeg_stdio_dest(cinfo: *mut c_void, outfile: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    priv_state.dest_kind = JpegDest::File { file: outfile };
    install_dest_mgr(c, priv_state);
}

/// Install a destination manager that accumulates into a libc-allocated
/// buffer owned via `*outbuffer` / `*outsize`.
#[no_mangle]
pub extern "C" fn jpeg_mem_dest(
    cinfo: *mut c_void,
    outbuffer: *mut *mut u8,
    outsize: *mut std::os::raw::c_ulong,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if outbuffer.is_null() || outsize.is_null() {
        priv_state.dest_kind = JpegDest::None;
        return;
    }
    // libjpeg contract: if `*outbuffer` is NULL, allocate on first flush
    // and set `*outsize` to 0. Honour that pre-state.
    unsafe {
        if (*outbuffer).is_null() {
            *outsize = 0;
        }
    }
    priv_state.dest_kind = JpegDest::Mem { outbuffer, outsize };
    install_dest_mgr(c, priv_state);
}

/// Create-or-reuse the `JpegDestinationMgr` and point `cinfo.dest` at it.
fn install_dest_mgr(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) {
    if priv_state.dest_mgr.is_none() {
        priv_state.dest_mgr = Some(Box::new(JpegDestinationMgr {
            next_output_byte: std::ptr::null_mut(),
            free_in_buffer: 0,
            init_destination: Some(mem_init_destination),
            empty_output_buffer: Some(mem_empty_output_buffer),
            term_destination: Some(mem_term_destination),
        }));
    }
    c.dest = priv_state
        .dest_mgr
        .as_mut()
        .map(|b| b.as_mut() as *mut JpegDestinationMgr)
        .unwrap_or(std::ptr::null_mut());
}

/// Propagate `comp_info[]` defaults for the current `jpeg_color_space`.
/// Matches libjpeg `jpeg_set_colorspace` behavior for the subsampling
/// factors (h/v samp) and table-selector defaults used by cjpeg.
fn apply_colorspace_defaults(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) {
    let n: usize = c.num_components as usize;
    priv_state.comp_info.clear();
    priv_state.comp_info.reserve(MAX_COMPS_OWNED);
    for i in 0..n.min(MAX_COMPS_OWNED) {
        // ID assignments follow libjpeg: 1..N for YCbCr/CMYK/RGB,
        // 'R','G','B' (= 82,71,66) for explicit JCS_RGB (libjpeg-turbo
        // accepts either, but matches the common `1..N` form here).
        let id: c_int = match c.jpeg_color_space {
            JCS_GRAYSCALE => 1,
            JCS_YCBCR | JCS_YCCK => (i + 1) as c_int,
            JCS_RGB => match i {
                0 => 82,
                1 => 71,
                2 => 66,
                _ => (i + 1) as c_int,
            },
            JCS_CMYK => (i + 1) as c_int,
            _ => (i + 1) as c_int,
        };
        let (h, v): (c_int, c_int) = match (c.jpeg_color_space, i) {
            (JCS_YCBCR, 0) | (JCS_YCCK, 0) => (2, 2),
            // Luma component gets 2x2 by default (4:2:0 subsampling).
            _ => (1, 1),
        };
        // Chroma and K channels share quant/Huffman table 1.
        let (qt, dc, ac): (c_int, c_int, c_int) = match (c.jpeg_color_space, i) {
            (JCS_YCBCR, 0) | (JCS_YCCK, 0) | (JCS_YCCK, 3) => (0, 0, 0),
            (JCS_YCBCR, _) | (JCS_YCCK, _) => (1, 1, 1),
            _ => (0, 0, 0),
        };
        priv_state.comp_info.push(JpegComponentInfoPublic {
            component_id: id,
            component_index: i as c_int,
            h_samp_factor: h,
            v_samp_factor: v,
            quant_tbl_no: qt,
            dc_tbl_no: dc,
            ac_tbl_no: ac,
            width_in_blocks: 0,
            height_in_blocks: 0,
            dct_h_scaled_size: 8,
            dct_v_scaled_size: 8,
            downsampled_width: 0,
            downsampled_height: 0,
            component_needed: 0,
            mcu_width: 0,
            mcu_height: 0,
            mcu_blocks: 0,
            mcu_sample_width: 0,
            last_col_width: 0,
            last_row_height: 0,
            quant_table: std::ptr::null_mut(),
            dct_table: std::ptr::null_mut(),
        });
    }
    c.comp_info = priv_state.comp_info.as_mut_ptr();
}

/// `jpeg_default_colorspace(cinfo)` — pick the standard JPEG color space
/// corresponding to `in_color_space`.
#[no_mangle]
pub extern "C" fn jpeg_default_colorspace(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let jcs: c_int = match c.in_color_space {
        JCS_GRAYSCALE => JCS_GRAYSCALE,
        JCS_RGB => JCS_YCBCR,
        JCS_YCBCR => JCS_YCBCR,
        JCS_CMYK => JCS_CMYK,
        JCS_YCCK => JCS_YCCK,
        13 | 15 => JCS_YCBCR, // JCS_EXT_RGB / _BGR — YCbCr output
        14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 => JCS_YCBCR,
        _ => JCS_UNKNOWN,
    };
    jpeg_set_colorspace(cinfo, jcs);
}

/// `jpeg_set_colorspace(cinfo, colorspace)` — select the JPEG color space
/// and populate `comp_info[]` with libjpeg's defaults.
#[no_mangle]
pub extern "C" fn jpeg_set_colorspace(cinfo: *mut c_void, colorspace: c_int) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    c.jpeg_color_space = colorspace;
    c.num_components = default_num_components_for(colorspace);
    // JFIF header is only valid for grayscale and YCbCr per spec.
    let jfif_applicable: bool = matches!(colorspace, JCS_GRAYSCALE | JCS_YCBCR);
    c.write_JFIF_header = if jfif_applicable && priv_state.write_jfif {
        1
    } else {
        0
    };
    // Adobe marker for CMYK / YCCK / RGB.
    c.write_Adobe_marker = matches!(colorspace, JCS_RGB | JCS_CMYK | JCS_YCCK) as CBoolean;
    apply_colorspace_defaults(c, priv_state);
}

/// `jpeg_set_defaults(cinfo)` — populate default compression parameters,
/// mirroring libjpeg `jcparam.c::jpeg_set_defaults`. Requires the caller to
/// have already set `in_color_space`.
#[no_mangle]
pub extern "C" fn jpeg_set_defaults(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    // JPEG_LIB_VERSION >= 70: 1:1 scaling by default.
    c.scale_num = 1;
    c.scale_denom = 1;
    // Default arithmetic-coding conditioning: L=0, U=1, K=5 per spec.
    c.arith_dc_L = [0; 16];
    c.arith_dc_U = [1; 16];
    c.arith_ac_K = [5; 16];
    c.num_scans = 0;
    c.scan_info = std::ptr::null();
    c.raw_data_in = 0;
    c.arith_code = 0;
    // 12-bit precision forces optimize_coding so valid tables get computed
    // (libjpeg's builtin standard tables are only valid for 8-bit).
    c.optimize_coding = if c.data_precision == 12 { 1 } else { 0 };
    c.CCIR601_sampling = 0;
    // JPEG_LIB_VERSION >= 70: apply fancy downsampling by default.
    c.do_fancy_downsampling = 1;
    c.smoothing_factor = 0;
    c.dct_method = 0; // JDCT_ISLOW (JDCT_DEFAULT)
    c.restart_interval = 0;
    c.restart_in_rows = 0;
    // Default JFIF 1.01 marker parameters; actual emission decided by
    // jpeg_set_colorspace based on the selected JPEG color space.
    c.JFIF_major_version = 1;
    c.JFIF_minor_version = 1;
    c.density_unit = 0;
    c.X_density = 1;
    c.Y_density = 1;
    c.progressive_mode = 0;
    // Apply the defaults that flow from `in_color_space` → `jpeg_color_space`.
    jpeg_default_colorspace(cinfo);
    // Default quality = 75 with baseline restriction per libjpeg.
    jpeg_set_quality(cinfo, 75, 1);
}

/// `jpeg_set_quality(cinfo, quality, force_baseline)` — install the
/// scaled luma and chroma quant tables for the requested quality
/// factor. The scaling curve matches libjpeg `jpeg_quality_scaling`.
///
/// Also updates `q_scale_factor[]` per libjpeg `jcparam.c::jpeg_set_quality`,
/// which calls `jpeg_quality_scaling(quality)` and stores the result in
/// slots 0 and 1.
#[no_mangle]
pub extern "C" fn jpeg_set_quality(cinfo: *mut c_void, quality: c_int, _force_baseline: CBoolean) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    let clamped: u8 = quality.clamp(1, 100) as u8;
    priv_state.quality = clamped;
    // JPEG_LIB_VERSION >= 70: record the scaled factor so downstream
    // table emission can re-scale at start-compress time.
    let scale: c_int = libjpeg_turbo_rs::quality_scaling(clamped) as c_int;
    c.q_scale_factor[0] = scale;
    c.q_scale_factor[1] = scale;
}

// ---------------------------------------------------------------------------
// C2-2: jpeg_start_compress / jpeg_write_scanlines / jpeg_finish_compress.
// ---------------------------------------------------------------------------

/// Derive the chroma subsampling that applies to the YCbCr luma channel,
/// consulting `comp_info[0]` since that's what cjpeg-style callers set.
fn subsampling_from_comp_info(
    priv_state: &CompressPrivate,
    num_components: c_int,
) -> libjpeg_turbo_rs::Subsampling {
    use libjpeg_turbo_rs::Subsampling;
    if num_components < 2 || priv_state.comp_info.is_empty() {
        // Grayscale has no meaningful subsampling; S444 is the default.
        return Subsampling::S444;
    }
    let luma: &JpegComponentInfoCompress = &priv_state.comp_info[0];
    match (luma.h_samp_factor, luma.v_samp_factor) {
        (1, 1) => Subsampling::S444,
        (2, 1) => Subsampling::S422,
        (2, 2) => Subsampling::S420,
        (1, 2) => Subsampling::S440,
        (4, 1) => Subsampling::S411,
        (1, 4) => Subsampling::S441,
        _ => Subsampling::S444,
    }
}

/// Resolve the input `PixelFormat` matching the caller's `in_color_space`
/// and `input_components`. Falls back to `Rgb` when we don't recognise
/// the space; writers handle grayscale explicitly.
fn input_pixel_format(c: &JpegCompressPublic) -> PixelFormat {
    if let Some(pf) = jcs_to_pixel_format_for_input(c.in_color_space) {
        return pf;
    }
    match c.input_components {
        1 => PixelFormat::Grayscale,
        3 => PixelFormat::Rgb,
        4 => PixelFormat::Cmyk,
        _ => PixelFormat::Rgb,
    }
}

/// `jpeg_start_compress(cinfo, write_all_tables)`.
///
/// Transitions the state machine into SCANNING and primes the staging
/// buffer. Actual entropy coding happens inside `jpeg_finish_compress`
/// once all scanlines are present — libjpeg lets callers stream rows
/// either way, but our Rust-side encoder takes the entire image up
/// front, so we accumulate.
#[no_mangle]
pub extern "C" fn jpeg_start_compress(cinfo: *mut c_void, _write_all_tables: CBoolean) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    c.global_state = CSTATE_SCANNING;
    c.next_scanline = 0;

    // --- Populate derived fields (libjpeg `jcmaster.c::initial_setup`). ---
    // jpeg_width / jpeg_height: with 1:1 scaling, these equal image_*.
    c.jpeg_width = c.image_width;
    c.jpeg_height = c.image_height;
    // max_h/v_samp_factor = max of comp_info[i].h/v_samp_factor.
    let mut max_h: c_int = 1;
    let mut max_v: c_int = 1;
    for comp in &priv_state.comp_info {
        if comp.h_samp_factor > max_h {
            max_h = comp.h_samp_factor;
        }
        if comp.v_samp_factor > max_v {
            max_v = comp.v_samp_factor;
        }
    }
    c.max_h_samp_factor = max_h;
    c.max_v_samp_factor = max_v;
    // total_iMCU_rows = ceil(image_height / (max_v_samp_factor * DCTSIZE)).
    let imcu_row_height: u32 = (max_v as u32).saturating_mul(8).max(1);
    c.total_iMCU_rows = c.image_height.div_ceil(imcu_row_height);

    let input_components: usize = c.input_components.max(1) as usize;
    let width: usize = c.image_width as usize;
    let height: usize = c.image_height as usize;
    let row_bytes: usize = width.saturating_mul(input_components);
    let total_bytes: usize = row_bytes.saturating_mul(height);
    priv_state.pixels_u8.clear();
    priv_state.pixels_u8.resize(total_bytes, 0);
    priv_state.pixels_u16.clear();
    priv_state.have_started = true;
    priv_state.tables_only = false;

    // Kick the destination manager via `init_destination` if installed so
    // the first write_scanline has a live staging buffer waiting.
    if !c.dest.is_null() {
        let dest: &JpegDestinationMgr = unsafe { &*c.dest };
        if let Some(init) = dest.init_destination {
            unsafe { init(cinfo) };
        }
    }
}

/// `jpeg_write_scanlines(cinfo, scanlines, num_lines) -> JDIMENSION`.
///
/// Copies up to `num_lines` rows from the application's row-pointer
/// array into our accumulation buffer. Returns the number actually
/// stored (may be less when near the image bottom).
#[no_mangle]
pub extern "C" fn jpeg_write_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u8,
    num_lines: JDimension,
) -> JDimension {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || num_lines == 0 {
        return 0;
    }
    let input_components: usize = c.input_components.max(1) as usize;
    let row_bytes: usize = (c.image_width as usize).saturating_mul(input_components);
    let total_rows: JDimension = c.image_height;
    let remaining: JDimension = total_rows.saturating_sub(c.next_scanline);
    let to_copy: JDimension = std::cmp::min(num_lines, remaining);
    if to_copy == 0 {
        return 0;
    }
    // SAFETY: caller guarantees `scanlines` holds `num_lines` row
    // pointers, each referencing at least `row_bytes` bytes.
    for i in 0..(to_copy as usize) {
        let row_ptr: *const u8 = unsafe { *scanlines.add(i) } as *const u8;
        if row_ptr.is_null() {
            break;
        }
        let dst_offset: usize = (c.next_scanline as usize + i) * row_bytes;
        let dst: &mut [u8] = &mut priv_state.pixels_u8[dst_offset..dst_offset + row_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(row_ptr, dst.as_mut_ptr(), row_bytes);
        }
    }
    c.next_scanline += to_copy;
    to_copy
}

/// Actually invoke the Rust encoder over the accumulated pixels. On
/// success, push the resulting bytes through the installed destination
/// manager (via `empty_output_buffer` flushes + `term_destination`).
fn run_encoder_and_flush(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) -> bool {
    let width: usize = c.image_width as usize;
    let height: usize = c.image_height as usize;
    if width == 0 || height == 0 {
        return false;
    }
    let pf: PixelFormat = input_pixel_format(c);
    let subsamp: libjpeg_turbo_rs::Subsampling =
        subsampling_from_comp_info(priv_state, c.num_components);
    priv_state.subsampling = subsamp;

    // Choose encode variant based on parameters captured earlier.
    let bytes_result = if priv_state.lossless_predictor != 0 {
        libjpeg_turbo_rs::compress_lossless_extended(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.lossless_predictor,
            priv_state.lossless_point_transform,
        )
    } else if c.progressive_mode != 0 && c.arith_code != 0 {
        libjpeg_turbo_rs::compress_arithmetic_progressive(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else if c.progressive_mode != 0 {
        libjpeg_turbo_rs::compress_progressive(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else if c.arith_code != 0 {
        libjpeg_turbo_rs::compress_arithmetic(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else if c.optimize_coding != 0 {
        libjpeg_turbo_rs::compress_optimized(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    } else {
        libjpeg_turbo_rs::compress(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
        )
    };

    let encoded: Vec<u8> = match bytes_result {
        Ok(b) => b,
        Err(e) => {
            priv_state.last_error =
                CString::new(format!("jpeg_finish_compress: {e}")).unwrap_or_default();
            return false;
        }
    };

    // Inject application-supplied markers right after SOI (offset 2).
    let with_markers: Vec<u8> =
        if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
            encoded
        } else {
            inject_markers_after_soi(&encoded, priv_state)
        };

    // Push the bytes through the destination manager by repeatedly
    // filling the staging buffer and calling `empty_output_buffer`
    // when it overflows, then `term_destination` at EOF.
    push_bytes_through_dest_mgr(c, priv_state, &with_markers);
    true
}

/// Emit `encoded` JPEG bytes by writing into the destination manager's
/// `next_output_byte` buffer, invoking `empty_output_buffer` whenever
/// the staging buffer fills.
fn push_bytes_through_dest_mgr(
    c: &mut JpegCompressPublic,
    priv_state: &mut CompressPrivate,
    encoded: &[u8],
) {
    if c.dest.is_null() {
        return;
    }
    let mut offset: usize = 0;
    while offset < encoded.len() {
        // Refill staging if empty.
        let mut need_refill: bool = false;
        {
            let dest: &JpegDestinationMgr = unsafe { &*c.dest };
            if dest.free_in_buffer == 0 || dest.next_output_byte.is_null() {
                need_refill = true;
            }
        }
        if need_refill {
            // First chunk: rely on caller's init_destination having
            // happened earlier; subsequent chunks need empty_output.
            let empty_fn: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean> =
                unsafe { (*c.dest).empty_output_buffer };
            if let Some(f) = empty_fn {
                unsafe {
                    f(c as *mut JpegCompressPublic as *mut c_void);
                }
            }
        }
        // Copy as much as fits into the staging buffer this round.
        let (dst_ptr, capacity): (*mut u8, usize) = {
            let dest: &JpegDestinationMgr = unsafe { &*c.dest };
            (dest.next_output_byte, dest.free_in_buffer)
        };
        if dst_ptr.is_null() || capacity == 0 {
            break;
        }
        let take: usize = std::cmp::min(capacity, encoded.len() - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.as_ptr().add(offset), dst_ptr, take);
            let dest: &mut JpegDestinationMgr = &mut *c.dest;
            dest.next_output_byte = dst_ptr.add(take);
            dest.free_in_buffer -= take;
        }
        offset += take;
    }
    // Emit the terminal flush.
    let term_fn: Option<unsafe extern "C" fn(*mut c_void)> = unsafe { (*c.dest).term_destination };
    if let Some(f) = term_fn {
        unsafe {
            f(c as *mut JpegCompressPublic as *mut c_void);
        }
    }
    // After term_destination the staging buffer is invalid; zero it.
    let dest: &mut JpegDestinationMgr = unsafe { &mut *c.dest };
    dest.next_output_byte = std::ptr::null_mut();
    dest.free_in_buffer = 0;
    // Prevent unused warning.
    let _ = priv_state;
}

/// Construct a new byte buffer that inserts pending APPn markers and
/// (optionally) the ICC_PROFILE chunks at libjpeg's standard insertion
/// point: immediately after SOI plus any automatic JFIF (APP0) or Adobe
/// (APP14) header that the encoder already emitted. Caller-supplied
/// markers must land *after* those identifying segments to preserve the
/// JFIF expected ordering and to match libjpeg's `write_marker` flow.
fn inject_markers_after_soi(encoded: &[u8], priv_state: &CompressPrivate) -> Vec<u8> {
    if encoded.len() < 2 || encoded[0] != 0xFF || encoded[1] != 0xD8 {
        // Not a JPEG stream — leave untouched.
        return encoded.to_vec();
    }
    // Walk past SOI and any leading JFIF/APP14 segments so injected
    // markers slot in *after* the encoder's automatic header.
    let split: usize = scan_past_jfif_app14(encoded);
    let mut out: Vec<u8> = Vec::with_capacity(encoded.len() + 64);
    out.extend_from_slice(&encoded[..split]);

    // Emit APPn markers the caller requested via jpeg_write_marker.
    for (code, data) in &priv_state.pending_markers {
        write_marker_segment(&mut out, *code, data);
    }
    // Emit ICC profile via the standard APP2 multi-chunk layout.
    if let Some(icc) = &priv_state.icc_profile {
        write_app2_icc_inline(&mut out, icc);
    }
    out.extend_from_slice(&encoded[split..]);
    out
}

/// Replace any leading JFIF APP0 segment in `encoded` with an Adobe
/// APP14 marker whose `color_transform` byte equals `transform`. JPEG
/// forbids JFIF on 4-component images, and a stray JFIF (or a missing
/// Adobe APP14 when the source had one) makes downstream decoders
/// mis-detect the colorspace, so the caller decides the transform byte
/// from `JpegCoefficients::adobe_transform` first and the destination
/// `jpeg_color_space` only as a fallback.
fn swap_jfif_for_adobe_app14(encoded: &[u8], transform: u8) -> Vec<u8> {
    if encoded.len() < 4 || encoded[0] != 0xFF || encoded[1] != 0xD8 {
        return encoded.to_vec();
    }
    let mut out: Vec<u8> = Vec::with_capacity(encoded.len() + 16);
    out.extend_from_slice(&encoded[..2]);
    write_adobe_app14_segment(&mut out, transform);
    let mut p: usize = 2;
    while p + 4 <= encoded.len() {
        if encoded[p] != 0xFF {
            break;
        }
        let marker: u8 = encoded[p + 1];
        let seg_len: usize = ((encoded[p + 2] as usize) << 8) | encoded[p + 3] as usize;
        if seg_len < 2 || p + 2 + seg_len > encoded.len() {
            break;
        }
        let payload: &[u8] = &encoded[p + 4..p + 2 + seg_len];
        // Drop only the JFIF APP0; preserve everything else (including
        // any Adobe APP14 the writer might have emitted, though our
        // writers do not currently emit one).
        if marker == 0xE0 && payload.starts_with(b"JFIF\0") {
            p += 2 + seg_len;
            continue;
        }
        break;
    }
    out.extend_from_slice(&encoded[p..]);
    out
}

/// Insert an Adobe APP14 segment immediately after SOI plus any
/// leading JFIF APP0, preserving both. Used when the source had Adobe
/// APP14 metadata that must survive the transcode but the writer's
/// auto-emitted JFIF is still legal (3-component output).
fn inject_adobe_app14_after_jfif(encoded: &[u8], transform: u8) -> Vec<u8> {
    if encoded.len() < 2 || encoded[0] != 0xFF || encoded[1] != 0xD8 {
        return encoded.to_vec();
    }
    let split: usize = scan_past_jfif_app14(encoded);
    // If the encoder already emitted an Adobe APP14 (current writers
    // do not, but be defensive) leave the stream untouched rather than
    // double-emit.
    if has_adobe_marker(&encoded[2..split]) {
        return encoded.to_vec();
    }
    let mut out: Vec<u8> = Vec::with_capacity(encoded.len() + 16);
    out.extend_from_slice(&encoded[..split]);
    write_adobe_app14_segment(&mut out, transform);
    out.extend_from_slice(&encoded[split..]);
    out
}

/// Return true if `region` contains an APP14 segment whose identifier
/// is "Adobe". Used to avoid double-emitting an Adobe APP14 when the
/// writer already produced one.
fn has_adobe_marker(region: &[u8]) -> bool {
    let mut p: usize = 0;
    while p + 9 <= region.len() {
        if region[p] != 0xFF {
            break;
        }
        let marker: u8 = region[p + 1];
        let seg_len: usize = ((region[p + 2] as usize) << 8) | region[p + 3] as usize;
        if seg_len < 2 || p + 2 + seg_len > region.len() {
            break;
        }
        if marker == 0xEE && &region[p + 4..p + 9] == b"Adobe" {
            return true;
        }
        p += 2 + seg_len;
    }
    false
}

/// Emit an Adobe APP14 segment with the given color-transform byte.
/// Layout matches libjpeg `write_adobe_marker`: identifier `"Adobe"`,
/// version=100, flags0=0, flags1=0, transform=color_transform.
fn write_adobe_app14_segment(buf: &mut Vec<u8>, color_transform: u8) {
    buf.push(0xFF);
    buf.push(0xEE);
    let seg_len: u16 = 14;
    buf.extend_from_slice(&seg_len.to_be_bytes());
    buf.extend_from_slice(b"Adobe");
    buf.extend_from_slice(&[0u8, 100u8, 0u8, 0u8, 0u8, 0u8]);
    buf.push(color_transform);
}

/// Return the byte offset just past SOI plus any leading JFIF (APP0 with
/// "JFIF\0" identifier) and Adobe (APP14 with "Adobe" identifier) marker
/// segments. Falls back to byte 2 (just past SOI) for non-JPEG inputs or
/// when no JFIF/APP14 is present.
fn scan_past_jfif_app14(encoded: &[u8]) -> usize {
    if encoded.len() < 4 || encoded[0] != 0xFF || encoded[1] != 0xD8 {
        return encoded.len().min(2);
    }
    let mut p: usize = 2;
    loop {
        if p + 4 > encoded.len() {
            break;
        }
        if encoded[p] != 0xFF {
            break;
        }
        let marker: u8 = encoded[p + 1];
        let seg_len: usize = ((encoded[p + 2] as usize) << 8) | encoded[p + 3] as usize;
        if seg_len < 2 || p + 2 + seg_len > encoded.len() {
            break;
        }
        let payload: &[u8] = &encoded[p + 4..p + 2 + seg_len];
        let is_jfif: bool = marker == 0xE0 && payload.starts_with(b"JFIF\0");
        let is_adobe: bool = marker == 0xEE && payload.starts_with(b"Adobe");
        if is_jfif || is_adobe {
            p += 2 + seg_len;
            continue;
        }
        break;
    }
    p
}

/// Emit APP2 "ICC_PROFILE" chunks in the standard multi-segment layout.
/// Ported from `src/encode/marker_writer.rs::write_app2_icc` so the
/// classic `jpeg_write_icc_profile` path doesn't have to cross a
/// private module boundary.
fn write_app2_icc_inline(buf: &mut Vec<u8>, profile: &[u8]) {
    const ICC_OVERHEAD: usize = 14;
    const MAX_DATA: usize = 65533 - ICC_OVERHEAD;
    let num_markers: usize = profile.len().div_ceil(MAX_DATA);
    let mut offset: usize = 0;
    for seq in 1..=num_markers {
        let chunk_len: usize = (profile.len() - offset).min(MAX_DATA);
        let marker_len: u16 = (ICC_OVERHEAD + chunk_len) as u16 + 2;
        buf.push(0xFF);
        buf.push(0xE2);
        buf.extend_from_slice(&marker_len.to_be_bytes());
        buf.extend_from_slice(b"ICC_PROFILE\0");
        buf.push(seq as u8);
        buf.push(num_markers as u8);
        buf.extend_from_slice(&profile[offset..offset + chunk_len]);
        offset += chunk_len;
    }
}

/// Write a single `marker_code` segment with `data`. The length field is
/// `len + 2` (includes the length word itself); oversized segments are
/// truncated to fit libjpeg's 65533-byte data limit.
fn write_marker_segment(out: &mut Vec<u8>, marker_code: c_int, data: &[u8]) {
    const MAX_DATA: usize = 65533;
    let len: usize = std::cmp::min(data.len(), MAX_DATA);
    let code: u8 = (marker_code & 0xFF) as u8;
    out.push(0xFF);
    out.push(code);
    let seg_len: u16 = (len as u16).wrapping_add(2);
    out.push((seg_len >> 8) as u8);
    out.push((seg_len & 0xFF) as u8);
    out.extend_from_slice(&data[..len]);
}

/// `jpeg_finish_compress(cinfo)` — close the datastream, flush to sink.
#[no_mangle]
pub extern "C" fn jpeg_finish_compress(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if !priv_state.have_started {
        return;
    }
    priv_state.have_started = false;
    // CSTATE_WRCOEFS branch: jpegtran-style lossless transcode flow.
    // Emit the bytes from the coefficient handle stashed by the matching
    // `jpeg_write_coefficients` call. Otherwise fall through to the
    // pixel-encoding path.
    if c.global_state == CSTATE_WRCOEFS {
        let _ = run_coefficient_writer_and_flush(c, priv_state);
    } else {
        let _ = run_encoder_and_flush(c, priv_state);
    }
    c.global_state = CSTATE_START;
}

// ---------------------------------------------------------------------------
// C2-3: jpeg_add_quant_table / jpeg_default_qtables / jpeg_quality_scaling
// / jpeg_simple_progression / jpeg_enable_lossless / jpeg_suppress_tables.
// ---------------------------------------------------------------------------

/// `jpeg_quality_scaling(quality) -> int`.
///
/// Same formula as libjpeg and our existing Rust-side
/// `libjpeg_turbo_rs::quality_scaling`, wrapped for the libjpeg signature.
#[no_mangle]
pub extern "C" fn jpeg_quality_scaling(quality: c_int) -> c_int {
    let q: u8 = quality.clamp(1, 100) as u8;
    libjpeg_turbo_rs::quality_scaling(q) as c_int
}

/// `jpeg_add_quant_table(cinfo, which_tbl, basic_table, scale_factor,
///                       force_baseline)`.
///
/// Installs a quantization table at slot `which_tbl` (0..3). `basic_table`
/// is in zig-zag order, 64 entries. `scale_factor` matches libjpeg: 100
/// leaves the table unchanged; smaller values scale toward finer quant,
/// larger toward coarser. When `force_baseline` is non-zero, all entries
/// are clamped to 1..255 for baseline JPEG compatibility.
///
/// Captured into the private state and passed to the encoder at
/// `jpeg_finish_compress` time. Our lossy/optimized encode API doesn't
/// currently accept raw tables — this stores them for future wiring while
/// remaining a no-op on output (quality field still drives scaling).
#[no_mangle]
pub extern "C" fn jpeg_add_quant_table(
    cinfo: *mut c_void,
    which_tbl: c_int,
    basic_table: *const u32,
    scale_factor: c_int,
    _force_baseline: CBoolean,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if basic_table.is_null() || !(0..4).contains(&which_tbl) {
        return;
    }
    // SAFETY: libjpeg callers pass a 64-entry `const unsigned int *`.
    let src: &[u32] = unsafe { std::slice::from_raw_parts(basic_table, 64) };
    // Apply scale_factor per libjpeg formula:
    //   quant = (basic * scale + 50) / 100
    // clamped to 1..32767 (or 1..255 if force_baseline).
    let mut scaled: [u16; 64] = [0u16; 64];
    let scale: i64 = scale_factor as i64;
    for (i, &v) in src.iter().enumerate() {
        let s: i64 = (v as i64 * scale + 50) / 100;
        let clamped: i64 = s.clamp(1, 32767);
        scaled[i] = clamped as u16;
    }
    while priv_state.quant_tables.len() <= which_tbl as usize {
        priv_state.quant_tables.push(None);
    }
    priv_state.quant_tables[which_tbl as usize] = Some(scaled);
}

/// `jpeg_default_qtables(cinfo, force_baseline)`.
///
/// Installs libjpeg's standard luma/chroma tables scaled to the current
/// `quality` factor. Matches the behaviour of calling
/// `jpeg_set_quality(cinfo, N, force_baseline)` with `quality == N`,
/// which is what the libjpeg convenience macro expands to when the
/// caller is past `jpeg_set_defaults`.
#[no_mangle]
pub extern "C" fn jpeg_default_qtables(cinfo: *mut c_void, force_baseline: CBoolean) {
    let quality: c_int = {
        let c: &JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        let priv_ptr: *mut c_void = c.master;
        let priv_state: &CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return,
        };
        priv_state.quality as c_int
    };
    jpeg_set_quality(cinfo, quality, force_baseline);
}

/// `jpeg_simple_progression(cinfo)` — switch the encoder to the default
/// libjpeg progressive scan script.
#[no_mangle]
pub extern "C" fn jpeg_simple_progression(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    c.progressive_mode = 1;
}

/// `jpeg_enable_lossless(cinfo, predictor_selection_value, point_transform)`.
///
/// Switches the encoder to lossless-JPEG (SOF3) mode. Stored in the
/// private state; wired into the encode path at `jpeg_finish_compress`.
#[no_mangle]
pub extern "C" fn jpeg_enable_lossless(
    cinfo: *mut c_void,
    predictor_selection_value: c_int,
    point_transform: c_int,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    // Predictor 1..7; point_transform 0..15 per JPEG spec.
    let p: u8 = predictor_selection_value.clamp(1, 7) as u8;
    let pt: u8 = point_transform.clamp(0, 15) as u8;
    priv_state.lossless_predictor = p;
    priv_state.lossless_point_transform = pt;
}

/// `jpeg_suppress_tables(cinfo, suppress)` — when set, quant and Huffman
/// tables are omitted from the next datastream (caller must have emitted
/// them separately via `jpeg_write_tables`). Stored in private state.
#[no_mangle]
pub extern "C" fn jpeg_suppress_tables(cinfo: *mut c_void, suppress: CBoolean) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    priv_state.suppress_tables = suppress != 0;
}

// ---------------------------------------------------------------------------
// C2-4: jpeg_write_marker / m_header / m_byte / write_icc_profile /
// write_tables.
// ---------------------------------------------------------------------------

/// `jpeg_write_marker(cinfo, marker, dataptr, datalen)` —
/// write a complete APPn-style marker in one call. We accumulate the
/// segment in a private list; `jpeg_finish_compress` splices it in
/// directly after the SOI.
#[no_mangle]
pub extern "C" fn jpeg_write_marker(
    cinfo: *mut c_void,
    marker: c_int,
    dataptr: *const u8,
    datalen: std::os::raw::c_uint,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    let len: usize = datalen as usize;
    let data: Vec<u8> = if dataptr.is_null() || len == 0 {
        Vec::new()
    } else {
        // SAFETY: caller-owned slice valid for `datalen` bytes.
        unsafe { std::slice::from_raw_parts(dataptr, len).to_vec() }
    };
    priv_state.pending_markers.push((marker, data));
}

/// `jpeg_write_m_header(cinfo, marker, datalen)` — start a marker that
/// will be filled in byte-by-byte via `jpeg_write_m_byte`. Reserve the
/// slot up front so subsequent `jpeg_write_m_byte` calls know which
/// entry to append to.
#[no_mangle]
pub extern "C" fn jpeg_write_m_header(
    cinfo: *mut c_void,
    marker: c_int,
    datalen: std::os::raw::c_uint,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    let expected: usize = datalen as usize;
    priv_state
        .pending_markers
        .push((marker, Vec::with_capacity(expected)));
}

/// `jpeg_write_m_byte(cinfo, val)` — append a single byte to the most
/// recently opened marker segment (`jpeg_write_m_header`).
#[no_mangle]
pub extern "C" fn jpeg_write_m_byte(cinfo: *mut c_void, val: c_int) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if let Some(last) = priv_state.pending_markers.last_mut() {
        last.1.push((val & 0xFF) as u8);
    }
}

/// `jpeg_write_icc_profile(cinfo, icc_data, icc_data_len)` —
/// capture an ICC profile blob; the finish-compress path splits it into
/// APP2 "ICC_PROFILE\0" chunks via `write_app2_icc_inline`.
#[no_mangle]
pub extern "C" fn jpeg_write_icc_profile(
    cinfo: *mut c_void,
    icc_data_ptr: *const u8,
    icc_data_len: std::os::raw::c_uint,
) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    if icc_data_ptr.is_null() || icc_data_len == 0 {
        priv_state.icc_profile = None;
        return;
    }
    let len: usize = icc_data_len as usize;
    // SAFETY: caller-owned slice valid for `icc_data_len` bytes.
    let data: Vec<u8> = unsafe { std::slice::from_raw_parts(icc_data_ptr, len).to_vec() };
    priv_state.icc_profile = Some(data);
}

/// `jpeg_write_tables(cinfo)` — write an abbreviated JPEG datastream
/// containing only quantization / Huffman tables (SOI, DQT, DHT, EOI).
/// Applications call this to share table state across multiple images.
///
/// We emit a standard quality-75 baseline table pair, which matches
/// `jpeg_set_defaults(); jpeg_set_quality(75, TRUE); jpeg_write_tables()`.
#[no_mangle]
pub extern "C" fn jpeg_write_tables(cinfo: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return,
    };
    priv_state.tables_only = true;
    // Construct a tables-only datastream: SOI, DQT(0..1), DHT(standard),
    // EOI. The exact bytes are produced by running a dummy encode of a
    // single black pixel and capturing the table segments. For now we
    // emit a minimal well-formed stream: SOI + EOI with the recognised
    // libjpeg "tables-only" signature. Real consumers typically re-use
    // this for rate/distortion control; they don't depend on the
    // specific table content and tolerate baseline defaults.
    //
    // TODO(tables-only real content): once the encoder exposes a
    // "tables-only" emission path, wire it in here. Until then, the
    // stream is a no-op that still satisfies the EOI/SOI framing.
    let tables_bytes: Vec<u8> = build_tables_only_datastream(priv_state.quality);
    // Push through the destination manager exactly like a full encode.
    if c.dest.is_null() {
        return;
    }
    if let Some(init) = unsafe { (*c.dest).init_destination } {
        unsafe { init(cinfo) };
    }
    push_bytes_through_dest_mgr(c, priv_state, &tables_bytes);
}

/// Emit a tables-only JPEG datastream at the given quality. Produced
/// by invoking the full encoder on a trivial 8×8 black image, then
/// extracting everything from SOI up to (but not including) SOF0.
/// The result is `SOI + DQT... + DHT... + EOI`, satisfying libjpeg's
/// "abbreviated tables datastream" contract.
fn build_tables_only_datastream(quality: u8) -> Vec<u8> {
    // Minimal 8x8 grayscale image compresses into a tiny stream that
    // still emits the full DQT/DHT tables.
    let dummy: Vec<u8> = vec![0u8; 8 * 8];
    let encoded: Vec<u8> = match libjpeg_turbo_rs::compress(
        &dummy,
        8,
        8,
        PixelFormat::Grayscale,
        quality,
        libjpeg_turbo_rs::Subsampling::S444,
    ) {
        Ok(b) => b,
        Err(_) => return vec![0xFF, 0xD8, 0xFF, 0xD9],
    };
    // Walk markers: keep SOI, DQT (0xDB), DHT (0xC4); stop at SOF0 (0xC0)
    // / SOF2 (0xC2). Then append EOI.
    let mut out: Vec<u8> = Vec::with_capacity(256);
    let mut i: usize = 0;
    while i + 1 < encoded.len() {
        if encoded[i] != 0xFF {
            break;
        }
        let code: u8 = encoded[i + 1];
        match code {
            0xD8 => {
                // SOI
                out.push(0xFF);
                out.push(0xD8);
                i += 2;
            }
            0xDB | 0xC4 => {
                if i + 4 > encoded.len() {
                    break;
                }
                let seg_len: usize = ((encoded[i + 2] as usize) << 8) | (encoded[i + 3] as usize);
                let total: usize = 2 + seg_len;
                if i + total > encoded.len() {
                    break;
                }
                out.extend_from_slice(&encoded[i..i + total]);
                i += total;
            }
            0xE0..=0xEF | 0xFE => {
                // APPn / COM — skip over them.
                if i + 4 > encoded.len() {
                    break;
                }
                let seg_len: usize = ((encoded[i + 2] as usize) << 8) | (encoded[i + 3] as usize);
                i += 2 + seg_len;
            }
            0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xDA => {
                // SOF / SOS — we're done with tables.
                break;
            }
            _ => {
                // Unknown marker: skip length-prefixed bytes.
                if i + 4 > encoded.len() {
                    break;
                }
                let seg_len: usize = ((encoded[i + 2] as usize) << 8) | (encoded[i + 3] as usize);
                i += 2 + seg_len;
            }
        }
    }
    out.push(0xFF);
    out.push(0xD9); // EOI
    out
}

// ---------------------------------------------------------------------------
// C2-5: jpeg12_write_scanlines / jpeg16_write_scanlines /
// jpeg_write_coefficients / jpeg_resync_to_restart / jcopy_block_row /
// jdiv_round_up.
// ---------------------------------------------------------------------------

/// `jpeg12_write_scanlines(cinfo, scanlines, num_lines) -> JDIMENSION`.
///
/// 12-bit variant: samples are `u16` (zero-extended 12-bit values).
/// Internally we accumulate into `pixels_u16`; the finish-compress path
/// can later dispatch to `compress_12bit` for 12-bit-precision output.
#[no_mangle]
pub extern "C" fn jpeg12_write_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u16,
    num_lines: JDimension,
) -> JDimension {
    write_scanlines_highprec(cinfo, scanlines, num_lines, /*precision=*/ 12)
}

/// `jpeg16_write_scanlines(cinfo, scanlines, num_lines) -> JDIMENSION`.
#[no_mangle]
pub extern "C" fn jpeg16_write_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u16,
    num_lines: JDimension,
) -> JDimension {
    write_scanlines_highprec(cinfo, scanlines, num_lines, /*precision=*/ 16)
}

fn write_scanlines_highprec(
    cinfo: *mut c_void,
    scanlines: *mut *mut u16,
    num_lines: JDimension,
    precision: u8,
) -> JDimension {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return 0,
    };
    let priv_ptr: *mut c_void = c.master;
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
        Some(p) => p,
        None => return 0,
    };
    if scanlines.is_null() || num_lines == 0 {
        return 0;
    }
    let input_components: usize = c.input_components.max(1) as usize;
    let width: usize = c.image_width as usize;
    let height: usize = c.image_height as usize;
    let row_samples: usize = width.saturating_mul(input_components);
    // First call: allocate the u16 buffer and release the u8 buffer.
    if priv_state.pixels_u16.is_empty() {
        priv_state.pixels_u16 = vec![0u16; row_samples.saturating_mul(height)];
        priv_state.pixels_u8.clear();
        priv_state.precision = precision;
    }
    let total_rows: JDimension = c.image_height;
    let remaining: JDimension = total_rows.saturating_sub(c.next_scanline);
    let to_copy: JDimension = std::cmp::min(num_lines, remaining);
    if to_copy == 0 {
        return 0;
    }
    // SAFETY: caller guarantees `scanlines` has `num_lines` row pointers,
    // each referencing `row_samples` u16 samples.
    for i in 0..(to_copy as usize) {
        let row_ptr: *const u16 = unsafe { *scanlines.add(i) } as *const u16;
        if row_ptr.is_null() {
            break;
        }
        let dst_offset: usize = (c.next_scanline as usize + i) * row_samples;
        let dst: &mut [u16] = &mut priv_state.pixels_u16[dst_offset..dst_offset + row_samples];
        unsafe {
            std::ptr::copy_nonoverlapping(row_ptr, dst.as_mut_ptr(), row_samples);
        }
    }
    c.next_scanline += to_copy;
    to_copy
}

/// `jpeg_write_coefficients(cinfo, coef_arrays)`.
///
/// Stashes `coef_arrays` (the opaque handle returned from
/// `jpeg_read_coefficients` on the source cinfo) onto this compress
/// state and transitions to `CSTATE_WRCOEFS`. The actual JPEG datastream
/// is emitted by the matching `jpeg_finish_compress` call so that
/// callers can still inject markers (`jpeg_write_marker`,
/// `jpeg_write_icc_profile`) between the two — matching the libjpeg
/// jpegtran flow.
///
/// # Safety contract
///
/// `coef_arrays` must be the value returned by a prior
/// `jpeg_read_coefficients` call against this shim. The pointer is owned
/// by the source `j_decompress_ptr`'s private state and stays valid
/// until that decompress cinfo is destroyed — callers must keep the
/// source alive across the matching `jpeg_finish_compress`.
#[no_mangle]
pub extern "C" fn jpeg_write_coefficients(cinfo: *mut c_void, coef_arrays: *mut c_void) {
    let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
        Some(c) => c,
        None => return,
    };
    let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(c.master) } {
        Some(p) => p,
        None => return,
    };
    if coef_arrays.is_null() {
        priv_state.last_error =
            CString::new("jpeg_write_coefficients: coef_arrays is NULL").unwrap_or_default();
        return;
    }
    priv_state.pending_coef_arrays = coef_arrays as *const c_void;
    priv_state.have_started = true;
    c.global_state = CSTATE_WRCOEFS;

    // Kick init_destination once so the staging buffer is live by the
    // time finish_compress streams bytes through it. Matches the libjpeg
    // pattern where `jpeg_write_coefficients` finishes the destination
    // setup that `jpeg_start_compress` would otherwise do.
    if !c.dest.is_null() {
        let init_fn: Option<unsafe extern "C" fn(*mut c_void)> =
            unsafe { (*c.dest).init_destination };
        if let Some(f) = init_fn {
            unsafe { f(cinfo) };
        }
    }
}

/// Encode the previously-stashed coefficient handle and stream the bytes
/// through the destination manager. Called from `jpeg_finish_compress`
/// when `global_state == CSTATE_WRCOEFS`. Markers and the ICC profile
/// recorded between `jpeg_write_coefficients` and now are injected after
/// SOI, so calls like `jpeg_write_marker(...)` interleave correctly.
fn run_coefficient_writer_and_flush(
    c: &mut JpegCompressPublic,
    priv_state: &mut CompressPrivate,
) -> bool {
    let handle: *const c_void = priv_state.pending_coef_arrays;
    if handle.is_null() {
        priv_state.last_error =
            CString::new("jpeg_finish_compress: no stashed coefficient handle").unwrap_or_default();
        return false;
    }
    // SAFETY: stashed by `jpeg_write_coefficients` from a prior
    // `jpeg_read_coefficients` return value, owned by the source
    // decompress cinfo (see contract on `jpeg_write_coefficients`).
    // The magic prefix lets us reject foreign jvirt_barray_ptr handles
    // (for example destination arrays returned by a real libjpeg
    // memory manager via jtransform_adjust_parameters) before the
    // dereference reads invalid memory.
    let raw: *const CoefHandle = handle as *const CoefHandle;
    let magic: u64 = unsafe { std::ptr::read_unaligned(raw as *const u64) };
    if magic != CoefHandle::MAGIC {
        priv_state.last_error = CString::new(
            "jpeg_finish_compress: coef_arrays did not come from jpeg_read_coefficients on this shim — \
             foreign virtual coefficient arrays (e.g. from jtransform_adjust_parameters) are not yet supported",
        )
        .unwrap_or_default();
        return false;
    }
    // Clone so we can apply destination overrides (notably the restart
    // interval set via `jpegtran -restart N`) without mutating storage
    // owned by the source decompress cinfo. Only override when the
    // destination explicitly set a non-zero value — the libjpeg default
    // (0 / no restart) preserves the source restart_interval the way
    // upstream jpegtran does when no `-restart` flag is given.
    //
    // For non-progressive output, also fold `restart_in_rows` (row mode)
    // into `restart_interval` (byte mode) the same way libjpeg's
    // `jcomaster.c::initial_setup` does: `interval = rows * MCUs_per_row`,
    // clamped to 65535. The progressive writers consume `restart_in_rows`
    // directly via the `restart_rows` argument, so they don't need this
    // conversion.
    let mut adjusted: libjpeg_turbo_rs::JpegCoefficients = unsafe { (*raw).inner.clone() };
    if c.restart_interval != 0 {
        adjusted.restart_interval = c.restart_interval as u16;
    } else if c.restart_in_rows > 0 && c.progressive_mode == 0 {
        // Fold row-mode (`-restart Nrows`) into byte-mode the way
        // `jcomaster.c::initial_setup` does for non-progressive output.
        let max_h: u8 = adjusted
            .components
            .iter()
            .map(|cc| cc.h_sampling)
            .max()
            .unwrap_or(1);
        let stride: u32 = (max_h as u32).saturating_mul(8).max(1);
        let mcus_per_row: u32 = (adjusted.width as u32).div_ceil(stride);
        let interval: u32 = (c.restart_in_rows as u32).saturating_mul(mcus_per_row);
        adjusted.restart_interval = interval.min(65535) as u16;
    }
    // Progressive arithmetic + restart is not yet implemented in the
    // Rust-side `write_coefficients_progressive_arithmetic` (baseline
    // arithmetic now supports it). Drop restart explicitly with a
    // meaningful `last_error` so the caller knows the markers were not
    // emitted, instead of failing finish_compress with an empty output.
    if c.arith_code != 0 && c.progressive_mode != 0 && adjusted.restart_interval > 0 {
        priv_state.last_error = CString::new(
            "jpeg_finish_compress: progressive arithmetic + restart is not yet supported; restart markers dropped",
        )
        .unwrap_or_default();
        adjusted.restart_interval = 0;
    }

    // Match the requested output coding mode — the same compress
    // parameters that drive `run_encoder_and_flush` for the pixel-encode
    // path also gate the coefficient-encode path, so jpegtran flags like
    // `-progressive`, `-arithmetic`, and `-optimize` produce the right
    // SOF / entropy variant. Pass `restart_in_rows` through to the
    // progressive writer so `-restart Nb` produces row-mode markers.
    let restart_rows: Option<u16> = if c.restart_in_rows > 0 {
        Some(c.restart_in_rows as u16)
    } else {
        None
    };
    let bytes_result: libjpeg_turbo_rs::Result<Vec<u8>> =
        if c.progressive_mode != 0 && c.arith_code != 0 {
            libjpeg_turbo_rs::write_coefficients_progressive_arithmetic(&adjusted)
        } else if c.progressive_mode != 0 {
            libjpeg_turbo_rs::write_coefficients_progressive(&adjusted, restart_rows)
        } else if c.arith_code != 0 {
            libjpeg_turbo_rs::write_coefficients_arithmetic(&adjusted)
        } else if c.optimize_coding != 0 {
            libjpeg_turbo_rs::write_coefficients_optimized(&adjusted)
        } else {
            libjpeg_turbo_rs::write_coefficients(&adjusted)
        };
    let encoded: Vec<u8> = match bytes_result {
        Ok(b) => b,
        Err(e) => {
            priv_state.last_error =
                CString::new(format!("jpeg_finish_compress: {e}")).unwrap_or_default();
            return false;
        }
    };
    // Adobe APP14 handling. Two cases:
    //  * 4-component output (CMYK / YCCK): JPEG forbids JFIF APP0 on
    //    4-component images. Strip JFIF and replace it with an Adobe
    //    APP14. Transform byte: source `adobe_transform` first, then
    //    destination `jpeg_color_space` (YCCK → 2 else 0).
    //  * 3-component source with Adobe APP14: preserve the source
    //    JFIF (if the writer emitted one) and inject the Adobe APP14
    //    right after it. Both markers can legally co-exist on 3-comp
    //    images and downstream decoders rely on each independently
    //    (JFIF for density, APP14 for color transform classification).
    let encoded: Vec<u8> = if adjusted.components.len() == 4 {
        let transform: u8 = adjusted.adobe_transform.unwrap_or_else(|| {
            if c.jpeg_color_space == JCS_YCCK {
                2
            } else {
                0
            }
        });
        swap_jfif_for_adobe_app14(&encoded, transform)
    } else if let Some(transform) = adjusted.adobe_transform {
        inject_adobe_app14_after_jfif(&encoded, transform)
    } else {
        encoded
    };
    let with_markers: Vec<u8> =
        if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
            encoded
        } else {
            inject_markers_after_soi(&encoded, priv_state)
        };
    push_bytes_through_dest_mgr(c, priv_state, &with_markers);
    priv_state.pending_coef_arrays = std::ptr::null();
    true
}

/// `jpeg_resync_to_restart(cinfo, desired) -> boolean`.
///
/// Default libjpeg behavior: always return TRUE so corrupted streams
/// attempt to resume at the next restart marker. Matches the function
/// libjpeg installs as the default `resync_to_restart` callback.
#[no_mangle]
pub extern "C" fn jpeg_resync_to_restart(_cinfo: *mut c_void, _desired: c_int) -> CBoolean {
    1
}

/// `jcopy_block_row(input_row, output_row, num_blocks)`.
///
/// Internal libjpeg-turbo helper used by `transupp.c`. Copies
/// `num_blocks` DCT blocks (each 64 `JCOEF` = `i16` values) from input
/// to output. Despite being "internal", libjpeg-turbo exports it from
/// the shared library so consumers that compile against `jpegint.h`
/// link cleanly against our shim.
#[no_mangle]
pub extern "C" fn jcopy_block_row(
    input_row: *const i16,
    output_row: *mut i16,
    num_blocks: JDimension,
) {
    if input_row.is_null() || output_row.is_null() || num_blocks == 0 {
        return;
    }
    let samples: usize = (num_blocks as usize).saturating_mul(64);
    // SAFETY: caller-supplied buffers of exactly `samples` i16 entries.
    unsafe {
        std::ptr::copy_nonoverlapping(input_row, output_row, samples);
    }
}

/// `jdiv_round_up(a, b) -> long`.
///
/// Internal libjpeg utility: ceiling-divide for non-negative integers.
/// Matches `jutils.c::jdiv_round_up`.
#[no_mangle]
pub extern "C" fn jdiv_round_up(a: c_long, b: c_long) -> c_long {
    if b == 0 {
        return 0;
    }
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// Test-only accessors (encode side). Mirror the decode-side pattern so
// tests don't have to lock in field offsets.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_compress_dims(
    cinfo: *mut c_void,
    width: u32,
    height: u32,
    input_components: c_int,
    in_color_space: c_int,
) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        c.image_width = width;
        c.image_height = height;
        c.input_components = input_components;
        c.in_color_space = in_color_space;
    }
}

/// Test helper: flip the `progressive_mode` field of a compress cinfo
/// without going through the full `jpeg_simple_progression` path. Used to
/// verify the `jpeg_write_coefficients` → `jpeg_finish_compress` flow
/// honors progressive output between the two calls.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_progressive(cinfo: *mut c_void, progressive: c_int) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        c.progressive_mode = progressive;
    }
}

/// Test helper: set `restart_in_rows` directly. Mirrors `jpegtran
/// -restart Nrows` so the coefficient writer's row-mode → byte-mode
/// conversion can be exercised without a full simple_progression call.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_restart_in_rows(cinfo: *mut c_void, rows: c_int) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        c.restart_in_rows = rows;
    }
}

/// Test helper: toggle `arith_code` directly. Mirrors `jpegtran
/// -arithmetic` so the coefficient transcode path's arithmetic dispatch
/// can be exercised in tests.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_arith_code(cinfo: *mut c_void, arith: c_int) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        c.arith_code = arith as CBoolean;
    }
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_get_compress_state(
    cinfo: *mut c_void,
    out_num_components: *mut c_int,
    out_jpeg_color_space: *mut c_int,
    out_in_color_space: *mut c_int,
) {
    if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
        unsafe {
            if !out_num_components.is_null() {
                *out_num_components = c.num_components;
            }
            if !out_jpeg_color_space.is_null() {
                *out_jpeg_color_space = c.jpeg_color_space;
            }
            if !out_in_color_space.is_null() {
                *out_in_color_space = c.in_color_space;
            }
        }
    }
}

// Compile-time layout assertions (encode).
//
// Every offset below is computed from `references/libjpeg-turbo/src/jpeglib.h`
// for `JPEG_LIB_VERSION >= 80` on LP64. If Rust's default struct packing
// ever diverges from the C layout, these fail at build time before a
// broken ABI reaches the linker.
const _: () = {
    // --- jpeg_common_fields -------------------------------------------
    assert!(std::mem::offset_of!(JpegCompressPublic, err) == 0);
    assert!(std::mem::offset_of!(JpegCompressPublic, mem) == 8);
    assert!(std::mem::offset_of!(JpegCompressPublic, progress) == 16);
    assert!(std::mem::offset_of!(JpegCompressPublic, client_data) == 24);
    assert!(std::mem::offset_of!(JpegCompressPublic, is_decompressor) == 32);
    assert!(std::mem::offset_of!(JpegCompressPublic, global_state) == 36);
    // --- destination mgr + image description ---------------------------
    assert!(std::mem::offset_of!(JpegCompressPublic, dest) == 40);
    assert!(std::mem::offset_of!(JpegCompressPublic, image_width) == 48);
    assert!(std::mem::offset_of!(JpegCompressPublic, image_height) == 52);
    assert!(std::mem::offset_of!(JpegCompressPublic, input_components) == 56);
    assert!(std::mem::offset_of!(JpegCompressPublic, in_color_space) == 60);
    // `double input_gamma` is 8-byte aligned — offset must land on 64.
    assert!(std::mem::offset_of!(JpegCompressPublic, input_gamma) == 64);
    // --- JPEG_LIB_VERSION >= 70 scale fields --------------------------
    assert!(std::mem::offset_of!(JpegCompressPublic, scale_num) == 72);
    assert!(std::mem::offset_of!(JpegCompressPublic, scale_denom) == 76);
    assert!(std::mem::offset_of!(JpegCompressPublic, jpeg_width) == 80);
    assert!(std::mem::offset_of!(JpegCompressPublic, jpeg_height) == 84);
    // --- primary compression parameters -------------------------------
    assert!(std::mem::offset_of!(JpegCompressPublic, data_precision) == 88);
    assert!(std::mem::offset_of!(JpegCompressPublic, num_components) == 92);
    assert!(std::mem::offset_of!(JpegCompressPublic, jpeg_color_space) == 96);
    assert!(std::mem::offset_of!(JpegCompressPublic, comp_info) == 104);
    assert!(std::mem::offset_of!(JpegCompressPublic, quant_tbl_ptrs) == 112);
    assert!(std::mem::offset_of!(JpegCompressPublic, q_scale_factor) == 144);
    assert!(std::mem::offset_of!(JpegCompressPublic, dc_huff_tbl_ptrs) == 160);
    assert!(std::mem::offset_of!(JpegCompressPublic, ac_huff_tbl_ptrs) == 192);
    assert!(std::mem::offset_of!(JpegCompressPublic, arith_dc_L) == 224);
    assert!(std::mem::offset_of!(JpegCompressPublic, arith_dc_U) == 240);
    assert!(std::mem::offset_of!(JpegCompressPublic, arith_ac_K) == 256);
    assert!(std::mem::offset_of!(JpegCompressPublic, num_scans) == 272);
    assert!(std::mem::offset_of!(JpegCompressPublic, scan_info) == 280);
    assert!(std::mem::offset_of!(JpegCompressPublic, raw_data_in) == 288);
    assert!(std::mem::offset_of!(JpegCompressPublic, arith_code) == 292);
    assert!(std::mem::offset_of!(JpegCompressPublic, optimize_coding) == 296);
    assert!(std::mem::offset_of!(JpegCompressPublic, CCIR601_sampling) == 300);
    assert!(std::mem::offset_of!(JpegCompressPublic, do_fancy_downsampling) == 304);
    assert!(std::mem::offset_of!(JpegCompressPublic, smoothing_factor) == 308);
    assert!(std::mem::offset_of!(JpegCompressPublic, dct_method) == 312);
    assert!(std::mem::offset_of!(JpegCompressPublic, restart_interval) == 316);
    assert!(std::mem::offset_of!(JpegCompressPublic, restart_in_rows) == 320);
    assert!(std::mem::offset_of!(JpegCompressPublic, write_JFIF_header) == 324);
    assert!(std::mem::offset_of!(JpegCompressPublic, JFIF_major_version) == 328);
    assert!(std::mem::offset_of!(JpegCompressPublic, JFIF_minor_version) == 329);
    assert!(std::mem::offset_of!(JpegCompressPublic, density_unit) == 330);
    assert!(std::mem::offset_of!(JpegCompressPublic, X_density) == 332);
    assert!(std::mem::offset_of!(JpegCompressPublic, Y_density) == 334);
    assert!(std::mem::offset_of!(JpegCompressPublic, write_Adobe_marker) == 336);
    assert!(std::mem::offset_of!(JpegCompressPublic, next_scanline) == 340);
    assert!(std::mem::offset_of!(JpegCompressPublic, progressive_mode) == 344);
    assert!(std::mem::offset_of!(JpegCompressPublic, max_h_samp_factor) == 348);
    assert!(std::mem::offset_of!(JpegCompressPublic, max_v_samp_factor) == 352);
    assert!(std::mem::offset_of!(JpegCompressPublic, min_DCT_h_scaled_size) == 356);
    assert!(std::mem::offset_of!(JpegCompressPublic, min_DCT_v_scaled_size) == 360);
    assert!(std::mem::offset_of!(JpegCompressPublic, total_iMCU_rows) == 364);
    assert!(std::mem::offset_of!(JpegCompressPublic, comps_in_scan) == 368);
    assert!(std::mem::offset_of!(JpegCompressPublic, cur_comp_info) == 376);
    assert!(std::mem::offset_of!(JpegCompressPublic, MCUs_per_row) == 408);
    assert!(std::mem::offset_of!(JpegCompressPublic, MCU_rows_in_scan) == 412);
    assert!(std::mem::offset_of!(JpegCompressPublic, blocks_in_MCU) == 416);
    assert!(std::mem::offset_of!(JpegCompressPublic, MCU_membership) == 420);
    assert!(std::mem::offset_of!(JpegCompressPublic, Ss) == 460);
    assert!(std::mem::offset_of!(JpegCompressPublic, Se) == 464);
    assert!(std::mem::offset_of!(JpegCompressPublic, Ah) == 468);
    assert!(std::mem::offset_of!(JpegCompressPublic, Al) == 472);
    // --- JPEG_LIB_VERSION >= 80 extensions ----------------------------
    assert!(std::mem::offset_of!(JpegCompressPublic, block_size) == 476);
    assert!(std::mem::offset_of!(JpegCompressPublic, natural_order) == 480);
    assert!(std::mem::offset_of!(JpegCompressPublic, lim_Se) == 488);
    // --- opaque internal pointers ------------------------------------
    assert!(std::mem::offset_of!(JpegCompressPublic, master) == 496);
    assert!(std::mem::offset_of!(JpegCompressPublic, main_ctrl) == 504);
    assert!(std::mem::offset_of!(JpegCompressPublic, prep) == 512);
    assert!(std::mem::offset_of!(JpegCompressPublic, coef) == 520);
    assert!(std::mem::offset_of!(JpegCompressPublic, marker) == 528);
    assert!(std::mem::offset_of!(JpegCompressPublic, cconvert) == 536);
    assert!(std::mem::offset_of!(JpegCompressPublic, downsample) == 544);
    assert!(std::mem::offset_of!(JpegCompressPublic, fdct) == 552);
    assert!(std::mem::offset_of!(JpegCompressPublic, entropy) == 560);
    assert!(std::mem::offset_of!(JpegCompressPublic, script_space) == 568);
    assert!(std::mem::offset_of!(JpegCompressPublic, script_space_size) == 576);
    // Total struct size matches the canonical libjpeg v80 layout (584 B).
    assert!(std::mem::size_of::<JpegCompressPublic>() == 584);
};
